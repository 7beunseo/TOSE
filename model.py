from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_1d_sincos_pos_embed(embed_dim: int, length: int, cls_token: bool = False) -> torch.Tensor:
    """Construct a one-dimensional sinusoidal positional embedding."""
    assert embed_dim % 2 == 0, "embed_dim must be even."
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)
    position = np.arange(length, dtype=np.float32)
    out = np.einsum("m,d->md", position, omega)
    pos_embed = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros((1, embed_dim), dtype=np.float32), pos_embed], axis=0)
    return torch.from_numpy(pos_embed).float()


class ObservedSetSelfBlock(nn.Module):
    """Contextualizes observed feature tokens as an unordered conditioning set."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    @staticmethod
    def safe_padding_mask(observed_bool: torch.Tensor) -> torch.Tensor:
        padding_mask = ~observed_bool.bool()
        all_padded = padding_mask.all(dim=1)
        if all_padded.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_padded] = False
        return padding_mask

    def forward(self, tokens: torch.Tensor, observed_bool: torch.Tensor) -> torch.Tensor:
        padding_mask = self.safe_padding_mask(observed_bool)
        normed = self.norm1(tokens)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=padding_mask, need_weights=False)
        tokens = tokens + self.dropout(attn_out)
        tokens = tokens + self.dropout(self.ffn(self.norm2(tokens)))
        return tokens


class TargetConditionalBlock(nn.Module):
    """Lets each target feature query retrieve evidence from observed features."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        enable_target_self_attention: bool = True,
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn = (
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            if enable_target_self_attention
            else None
        )
        self.cross_q_norm = nn.LayerNorm(embed_dim)
        self.cross_kv_norm = nn.LayerNorm(embed_dim)
        self.self_norm = nn.LayerNorm(embed_dim) if enable_target_self_attention else None
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(
        self,
        target_tokens: torch.Tensor,
        observed_tokens: torch.Tensor,
        observed_bool: torch.Tensor,
        use_target_conditioning: bool = True,
        source_aggregation: str = "learned",
        shared_attention_query: torch.Tensor | None = None,
        target_column_tokens: torch.Tensor | None = None,
        source_column_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        observed_norm = self.cross_kv_norm(observed_tokens)
        if source_aggregation == "uniform":
            cross_out = self._uniform_source_context(observed_norm, observed_bool)
            cross_out = cross_out.expand(-1, target_tokens.size(1), -1)
        elif source_aggregation == "static_column":
            if target_column_tokens is None or source_column_tokens is None:
                raise ValueError(
                    "Static column attention requires target and source column tokens."
                )
            cross_out = self._static_column_context(
                target_column_tokens=target_column_tokens,
                source_column_tokens=source_column_tokens,
                observed_values=observed_norm,
                observed_bool=observed_bool,
            )
        elif source_aggregation == "learned":
            padding_mask = ObservedSetSelfBlock.safe_padding_mask(observed_bool)
            normalized_targets = self.cross_q_norm(target_tokens)
            if shared_attention_query is not None:
                # The actual Q passed to cross-attention is identical for every
                # target; target identity remains only on the residual path.
                query = self.cross_q_norm(shared_attention_query)
            else:
                # A singleton shared query learns row-specific source importance
                # without allowing the source weights to depend on target identity.
                query = (
                    normalized_targets
                    if use_target_conditioning
                    else normalized_targets.mean(dim=1, keepdim=True)
                )
            cross_out, _ = self.cross_attn(
                query=query,
                key=observed_norm,
                value=observed_norm,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            if query.size(1) == 1:
                cross_out = cross_out.expand(-1, target_tokens.size(1), -1)
        else:
            raise ValueError(f"Unsupported source aggregation: {source_aggregation}")
        target_tokens = target_tokens + self.dropout(cross_out)
        if self.self_attn is not None:
            if self.self_norm is None:
                raise RuntimeError("Target self-attention normalization is unavailable.")
            target_norm = self.self_norm(target_tokens)
            self_out, _ = self.self_attn(target_norm, target_norm, target_norm, need_weights=False)
            target_tokens = target_tokens + self.dropout(self_out)
        target_tokens = target_tokens + self.dropout(self.ffn(self.ffn_norm(target_tokens)))
        return target_tokens

    def _uniform_source_context(
        self,
        observed_tokens: torch.Tensor,
        observed_bool: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the attention value/output projections with uniform observed-source weights."""
        effective_observed = observed_bool.bool()
        empty_rows = ~effective_observed.any(dim=1)
        if empty_rows.any():
            effective_observed = effective_observed.clone()
            effective_observed[empty_rows] = True

        if self.cross_attn.in_proj_weight is None:
            raise RuntimeError("Uniform aggregation requires packed QKV projection weights.")
        value_weight = self.cross_attn.in_proj_weight.chunk(3, dim=0)[2]
        value_bias = None
        if self.cross_attn.in_proj_bias is not None:
            value_bias = self.cross_attn.in_proj_bias.chunk(3, dim=0)[2]
        projected_values = F.linear(observed_tokens, value_weight, value_bias)

        source_weights = effective_observed.to(projected_values.dtype)
        source_weights = source_weights / source_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_values = torch.sum(projected_values * source_weights.unsqueeze(-1), dim=1, keepdim=True)
        return F.linear(
            pooled_values,
            self.cross_attn.out_proj.weight,
            self.cross_attn.out_proj.bias,
        )

    def _static_column_context(
        self,
        target_column_tokens: torch.Tensor,
        source_column_tokens: torch.Tensor,
        observed_values: torch.Tensor,
        observed_bool: torch.Tensor,
    ) -> torch.Tensor:
        """Attend by column identity while aggregating the row's actual source values."""
        if self.cross_attn.in_proj_weight is None:
            raise RuntimeError("Static column attention requires packed QKV projection weights.")

        query_weight, key_weight, value_weight = self.cross_attn.in_proj_weight.chunk(3, dim=0)
        query_bias = key_bias = value_bias = None
        if self.cross_attn.in_proj_bias is not None:
            query_bias, key_bias, value_bias = self.cross_attn.in_proj_bias.chunk(3, dim=0)

        query = F.linear(
            self.cross_q_norm(target_column_tokens), query_weight, query_bias
        )
        key = F.linear(
            self.cross_kv_norm(source_column_tokens), key_weight, key_bias
        )
        value = F.linear(observed_values, value_weight, value_bias)

        batch_size, target_count, embed_dim = query.shape
        source_count = key.size(1)
        num_heads = self.cross_attn.num_heads
        head_dim = embed_dim // num_heads

        query = query.view(batch_size, target_count, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, source_count, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, source_count, num_heads, head_dim).transpose(1, 2)

        effective_observed = observed_bool.bool()
        empty_rows = ~effective_observed.any(dim=1)
        if empty_rows.any():
            effective_observed = effective_observed.clone()
            effective_observed[empty_rows] = True

        attention_logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        attention_logits = attention_logits.masked_fill(
            ~effective_observed[:, None, None, :],
            torch.finfo(attention_logits.dtype).min,
        )
        attention_weights = torch.softmax(attention_logits, dim=-1)
        attention_weights = F.dropout(
            attention_weights,
            p=self.cross_attn.dropout,
            training=self.training,
        )
        context = torch.matmul(attention_weights, value)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, target_count, embed_dim
        )
        return F.linear(
            context,
            self.cross_attn.out_proj.weight,
            self.cross_attn.out_proj.bias,
        )


class SharedRowBlock(nn.Module):
    """Capacity-matched observed-row block used when target-wise TOSE is removed."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.context_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.context_q_norm = nn.LayerNorm(embed_dim)
        self.context_kv_norm = nn.LayerNorm(embed_dim)
        self.self_norm = nn.LayerNorm(embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, tokens: torch.Tensor, observed_bool: torch.Tensor) -> torch.Tensor:
        padding_mask = ObservedSetSelfBlock.safe_padding_mask(observed_bool)
        context_query = self.context_q_norm(tokens)
        context_memory = self.context_kv_norm(tokens)
        context_out, _ = self.context_attn(
            context_query,
            context_memory,
            context_memory,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        tokens = tokens + self.dropout(context_out)
        self_norm = self.self_norm(tokens)
        self_out, _ = self.self_attn(
            self_norm,
            self_norm,
            self_norm,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        tokens = tokens + self.dropout(self_out)
        tokens = tokens + self.dropout(self.ffn(self.ffn_norm(tokens)))
        return tokens


class TargetConditionalObservedSetEncoder(nn.Module):
    """
    Target-conditional observed-set encoder for imputation.

    Every feature has a target query. The query cross-attends to the currently
    observed feature set, so the representation for feature j is directly
    optimized for reconstructing feature j rather than being a generic row
    embedding.
    """

    def __init__(
        self,
        input_dim: int,
        num_dim: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        mechanism_hidden_dim: int,
        cross_attn_layers: int,
        ablation_mode: str = "full",
    ) -> None:
        super().__init__()
        valid_ablation_modes = {
            "full",
            "no_target_conditioning",
            "no_observed_self_attention",
            "no_state_embedding",
            "no_tose_shared_row",
            "uniform_source_aggregation",
            "shared_target_query",
            "static_column_attention",
            "uniform_source_no_target_set_sa",
            "no_target_aware_pooling",
            "both_uniform",
            "tose_1_block",
            "tose_2_blocks",
            "tose_4_blocks",
            "observed1_tose_1_block",
            "observed1_tose_3_blocks",
            "observed1_tose_6_blocks",
            "observed1_tose1_uniform_source",
            "observed1_tose1_uniform_source_no_target_set_sa",
            "observed1_tose1_uniform_source_no_observed_set_sa",
            "observed1_tose1_all_three_removed",
            "observed1_tose1_no_target_set_sa",
            "observed1_tose1_no_observed_set_sa",
            "observed1_tose1_shared_target_query",
            "observed1_tose1_static_column_attention",
            "linear_prediction_heads",
        }
        if ablation_mode not in valid_ablation_modes:
            raise ValueError(f"Unsupported ablation mode: {ablation_mode}")
        self.input_dim = input_dim
        self.num_dim = int(num_dim)
        self.embed_dim = embed_dim
        self.ablation_mode = ablation_mode
        self.feature_embedding = nn.Embedding(input_dim, embed_dim)
        # Kept for checkpoint compatibility with earlier public forward-only runs.
        self.type_embedding = nn.Embedding(2, embed_dim)
        feature_type_ids = torch.zeros(input_dim, dtype=torch.long)
        feature_type_ids[self.num_dim :] = 1
        self.register_buffer("feature_type_ids", feature_type_ids, persistent=False)
        self.value_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.observed_state_embedding = nn.Embedding(2, embed_dim)
        self.target_state_embedding = nn.Embedding(2, embed_dim)
        if ablation_mode == "no_tose_shared_row":
            self.register_parameter("query_seed", None)
        else:
            self.query_seed = nn.Parameter(torch.zeros(1, input_dim, embed_dim))
        self.register_buffer("feature_ids", torch.arange(input_dim), persistent=False)
        observed_depth = max(1, depth // 2)
        target_depth = max(1, depth + max(0, cross_attn_layers - 1))
        if ablation_mode == "tose_1_block":
            target_depth = 1
        elif ablation_mode == "tose_2_blocks":
            target_depth = 2
        elif ablation_mode == "tose_4_blocks":
            target_depth = 4
        elif ablation_mode in {
            "observed1_tose_1_block",
            "observed1_tose_3_blocks",
            "observed1_tose_6_blocks",
            "observed1_tose1_uniform_source",
            "observed1_tose1_uniform_source_no_target_set_sa",
            "observed1_tose1_uniform_source_no_observed_set_sa",
            "observed1_tose1_all_three_removed",
            "observed1_tose1_no_target_set_sa",
            "observed1_tose1_no_observed_set_sa",
            "observed1_tose1_shared_target_query",
            "observed1_tose1_static_column_attention",
        }:
            observed_depth = 1
            target_depth = {
                "observed1_tose_1_block": 1,
                "observed1_tose_3_blocks": 3,
                "observed1_tose_6_blocks": 6,
                "observed1_tose1_uniform_source": 1,
                "observed1_tose1_uniform_source_no_target_set_sa": 1,
                "observed1_tose1_uniform_source_no_observed_set_sa": 1,
                "observed1_tose1_all_three_removed": 1,
                "observed1_tose1_no_target_set_sa": 1,
                "observed1_tose1_no_observed_set_sa": 1,
                "observed1_tose1_shared_target_query": 1,
                "observed1_tose1_static_column_attention": 1,
            }[ablation_mode]
            if ablation_mode in {
                "observed1_tose1_uniform_source_no_observed_set_sa",
                "observed1_tose1_all_three_removed",
                "observed1_tose1_no_observed_set_sa",
            }:
                observed_depth = 0
        self.observed_blocks = nn.ModuleList(
            [ObservedSetSelfBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(observed_depth)]
        )
        self.target_blocks = nn.ModuleList()
        self.shared_row_blocks = nn.ModuleList()
        if ablation_mode == "no_tose_shared_row":
            self.shared_row_blocks.extend(
                SharedRowBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(target_depth)
            )
            self.shared_row_to_field = nn.Linear(embed_dim, embed_dim)
        else:
            self.target_blocks.extend(
                TargetConditionalBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    dropout,
                    enable_target_self_attention=(
                        ablation_mode
                        not in {
                            "uniform_source_no_target_set_sa",
                            "observed1_tose1_uniform_source_no_target_set_sa",
                            "observed1_tose1_all_three_removed",
                            "observed1_tose1_no_target_set_sa",
                        }
                    ),
                )
                for _ in range(target_depth)
            )
            self.shared_row_to_field = nn.Identity()
        self.field_norm = nn.LayerNorm(embed_dim)
        self.value_summary_norm = nn.LayerNorm(embed_dim)
        self.mask_summary_norm = nn.LayerNorm(embed_dim)
        self.summary_norm = nn.LayerNorm(embed_dim)
        self.observed_pool_score = nn.Linear(embed_dim, 1)
        self.target_pool_score = None
        if ablation_mode not in {
            "no_tose_shared_row",
            "no_target_aware_pooling",
            "both_uniform",
        }:
            self.target_pool_score = nn.Sequential(
                nn.Linear(embed_dim * 2, mechanism_hidden_dim),
                nn.GELU(),
                nn.Linear(mechanism_hidden_dim, 1),
            )
        self.mask_summary_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self._latest_fusion_stats = {
            "target_pool_entropy": 0.0,
            "observed_ratio": 1.0,
            "target_depth": float(target_depth),
            "shared_row_depth": float(len(self.shared_row_blocks)),
        }
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.query_seed is not None:
            nn.init.normal_(self.query_seed, std=0.02)
        nn.init.xavier_uniform_(self.observed_pool_score.weight)
        nn.init.zeros_(self.observed_pool_score.bias)

    def latest_fusion_stats(self) -> dict[str, float]:
        return dict(self._latest_fusion_stats)

    def _build_observed_tokens(
        self,
        feature_embed: torch.Tensor,
        value_embed: torch.Tensor,
        state_embed: torch.Tensor,
    ) -> torch.Tensor:
        return feature_embed + value_embed + state_embed

    def _build_target_tokens(
        self,
        query_seed: torch.Tensor,
        feature_embed: torch.Tensor,
        target_state: torch.Tensor,
    ) -> torch.Tensor:
        return query_seed + feature_embed + target_state

    def _record_stats(
        self,
        target_weights: torch.Tensor | None,
        observed_bool: torch.Tensor,
    ) -> None:
        normalized_entropy = 0.0
        if target_weights is not None:
            entropy = -(target_weights * torch.log(target_weights.clamp_min(1e-8))).sum(dim=1)
            max_entropy = math.log(max(2, target_weights.size(1)))
            normalized_entropy = float((entropy / max_entropy).mean().detach().cpu())
        self._latest_fusion_stats = {
            "target_pool_entropy": normalized_entropy,
            "observed_ratio": float(observed_bool.float().mean().detach().cpu()),
            "target_depth": float(len(self.target_blocks)),
            "shared_row_depth": float(len(self.shared_row_blocks)),
        }

    def forward(self, input_x: torch.Tensor, missing_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = input_x.size(0)
        observed_bool = missing_mask < 0.5
        use_state_embedding = self.ablation_mode != "no_state_embedding"
        feature_embed = self.feature_embedding(self.feature_ids).unsqueeze(0).expand(batch_size, -1, -1)
        observed_state = (
            self.observed_state_embedding(observed_bool.long())
            if use_state_embedding
            else torch.zeros_like(feature_embed)
        )
        observed_tokens = self._build_observed_tokens(
            feature_embed=feature_embed,
            value_embed=self.value_encoder(input_x.unsqueeze(-1)),
            state_embed=observed_state,
        )
        if self.ablation_mode not in {
            "no_observed_self_attention",
            "observed1_tose1_uniform_source_no_observed_set_sa",
            "observed1_tose1_all_three_removed",
            "observed1_tose1_no_observed_set_sa",
        }:
            for block in self.observed_blocks:
                observed_tokens = block(observed_tokens, observed_bool)

        if self.ablation_mode == "no_tose_shared_row":
            for block in self.shared_row_blocks:
                observed_tokens = block(observed_tokens, observed_bool)

        target_state = (
            self.target_state_embedding(missing_mask.long())
            if use_state_embedding
            else torch.zeros_like(feature_embed)
        )

        obs_logits = self.observed_pool_score(observed_tokens).squeeze(-1).masked_fill(~observed_bool, -1e4)
        obs_weights = torch.softmax(obs_logits, dim=1)
        value_summary = torch.sum(obs_weights.unsqueeze(-1) * observed_tokens, dim=1)
        value_summary = self.value_summary_norm(value_summary)

        mask_context = self.mask_summary_mlp(torch.cat([feature_embed, target_state], dim=-1))
        mask_summary = self.mask_summary_norm(mask_context.mean(dim=1))

        if self.ablation_mode == "no_tose_shared_row":
            shared_field_context = self.shared_row_to_field(value_summary).unsqueeze(1)
            field_tokens = self.field_norm(feature_embed + target_state + shared_field_context)
            pooled_repr = value_summary
            z_init = self.summary_norm(value_summary)
            self._record_stats(target_weights=None, observed_bool=observed_bool)
            return {
                "field_tokens": field_tokens,
                "cls_repr": z_init,
                "pooled_repr": pooled_repr,
                "value_summary": value_summary,
                "mask_summary": mask_summary,
                "z_init": z_init,
            }

        if self.query_seed is None:
            raise RuntimeError("Target query seed is unavailable outside the shared-row ablation.")
        target_tokens = self._build_target_tokens(
            query_seed=self.query_seed.expand(batch_size, -1, -1),
            feature_embed=feature_embed,
            target_state=target_state,
        )
        shared_attention_query = None
        if self.ablation_mode in {
            "shared_target_query",
            "observed1_tose1_shared_target_query",
        }:
            # Reuse the full model's query seeds so this ablation adds no
            # parameters. Their learned mean contains no target ID, state, or
            # intermediate target representation and is identical for every target.
            shared_attention_query = self.query_seed.mean(dim=1, keepdim=True).expand(
                batch_size, -1, -1
            )
        for block in self.target_blocks:
            target_tokens = block(
                target_tokens,
                observed_tokens,
                observed_bool,
                use_target_conditioning=self.ablation_mode
                not in {
                    "no_target_conditioning",
                    "shared_target_query",
                    "observed1_tose1_shared_target_query",
                },
                source_aggregation=(
                    "static_column"
                    if self.ablation_mode
                    in {
                        "static_column_attention",
                        "observed1_tose1_static_column_attention",
                    }
                    else "uniform"
                    if self.ablation_mode in {
                        "uniform_source_aggregation",
                        "uniform_source_no_target_set_sa",
                        "both_uniform",
                        "observed1_tose1_uniform_source",
                        "observed1_tose1_uniform_source_no_target_set_sa",
                        "observed1_tose1_uniform_source_no_observed_set_sa",
                        "observed1_tose1_all_three_removed",
                    }
                    else "learned"
                ),
                shared_attention_query=shared_attention_query,
                target_column_tokens=feature_embed,
                source_column_tokens=feature_embed,
            )
        field_tokens = self.field_norm(target_tokens)
        if self.ablation_mode in {"no_target_aware_pooling", "both_uniform"}:
            target_weights = torch.ones(
                field_tokens.shape[:2],
                dtype=field_tokens.dtype,
                device=field_tokens.device,
            )
            target_weights = target_weights / max(1, field_tokens.size(1))
        else:
            if self.target_pool_score is None:
                raise RuntimeError("Target pooling is unavailable for this ablation mode.")
            target_logits = self.target_pool_score(
                torch.cat([field_tokens, target_state], dim=-1)
            ).squeeze(-1)
            target_weights = torch.softmax(target_logits, dim=1)
        pooled_repr = torch.sum(target_weights.unsqueeze(-1) * field_tokens, dim=1)
        z_init = self.summary_norm(pooled_repr)
        self._record_stats(target_weights=target_weights, observed_bool=observed_bool)
        return {
            "field_tokens": field_tokens,
            "cls_repr": z_init,
            "pooled_repr": pooled_repr,
            "value_summary": value_summary,
            "mask_summary": mask_summary,
            "z_init": z_init,
        }


class SimpleFieldwiseHead(nn.Module):
    """Lightweight field-wise decoder used by TOSE."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        dropout: float,
        linear_readout: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.linear_readout = bool(linear_readout)
        self.row_proj = (
            nn.Identity()
            if self.linear_readout
            else nn.Linear(embed_dim, embed_dim)
        )
        self.mask_embedding = nn.Embedding(2, embed_dim)
        self.task_bias = nn.Parameter(torch.zeros(1, 1, embed_dim))
        pos_embed = get_1d_sincos_pos_embed(embed_dim, input_dim, cls_token=False)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)
        self.head = (
            nn.Linear(embed_dim, 1)
            if self.linear_readout
            else nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, 1),
            )
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if isinstance(self.row_proj, nn.Linear):
            nn.init.xavier_uniform_(self.row_proj.weight)
            nn.init.zeros_(self.row_proj.bias)
        nn.init.normal_(self.task_bias, std=0.02)
        final = self.head if isinstance(self.head, nn.Linear) else self.head[-1]
        nn.init.xavier_uniform_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        row_repr: torch.Tensor,
        field_tokens: torch.Tensor,
        condition_mask: torch.Tensor,
        use_state_embedding: bool = True,
    ) -> torch.Tensor:
        row_context = self.row_proj(row_repr).unsqueeze(1)
        state_embedding = (
            self.mask_embedding(condition_mask.long())
            if use_state_embedding
            else torch.zeros_like(field_tokens)
        )
        tokens = (
            field_tokens
            + row_context
            + state_embedding
            + self.pos_embed
            + self.task_bias
        )
        return self.head(tokens).squeeze(-1)


@dataclass
class TOSEOutputs:
    z_init: torch.Tensor
    z_final: torch.Tensor
    forward_num: torch.Tensor
    forward_cat_logits: torch.Tensor
    forward_cat_class_logits: list[torch.Tensor]


class TargetWiseCategoricalHead(nn.Module):
    """Predicts each categorical feature directly from TOSE target tokens."""

    def __init__(
        self,
        num_dim: int,
        cat_bin_num: torch.Tensor | list[int] | tuple[int, ...],
        cat_codebook_sizes: list[int] | tuple[int, ...],
        embed_dim: int,
        dropout: float,
        linear_readout: bool = False,
    ) -> None:
        super().__init__()
        self.num_dim = int(num_dim)
        self.linear_readout = bool(linear_readout)
        self.cat_bin_num = [int(value) for value in cat_bin_num]
        self.cat_bit_slices: list[tuple[int, int]] = []
        start = self.num_dim
        for bit_len in self.cat_bin_num:
            end = start + bit_len
            self.cat_bit_slices.append((start, end))
            start = end

        self.row_proj = (
            nn.Identity()
            if self.linear_readout
            else nn.Linear(embed_dim, embed_dim)
        )
        self.mask_embedding = nn.Embedding(2, embed_dim)
        self.type_embedding = nn.Embedding(1, embed_dim)
        self.heads = nn.ModuleList(
            [
                (
                    nn.Linear(embed_dim, int(num_classes))
                    if self.linear_readout
                    else nn.Sequential(
                        nn.LayerNorm(embed_dim),
                        nn.Linear(embed_dim, embed_dim * 2),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(embed_dim * 2, int(num_classes)),
                    )
                )
                for num_classes in cat_codebook_sizes
            ]
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if isinstance(self.row_proj, nn.Linear):
            nn.init.xavier_uniform_(self.row_proj.weight)
            nn.init.zeros_(self.row_proj.bias)
        for head in self.heads:
            final = head if isinstance(head, nn.Linear) else head[-1]
            nn.init.xavier_uniform_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(
        self,
        row_repr: torch.Tensor,
        field_tokens: torch.Tensor,
        condition_mask: torch.Tensor,
        use_state_embedding: bool = True,
    ) -> list[torch.Tensor]:
        if len(self.heads) == 0:
            return []

        row_context = self.row_proj(row_repr)
        outputs: list[torch.Tensor] = []
        for column_idx, (start, end) in enumerate(self.cat_bit_slices):
            token_j = field_tokens[:, start:end].mean(dim=1)
            mask_j = condition_mask[:, start:end].amax(dim=1).long()
            if use_state_embedding:
                token_j = token_j + self.mask_embedding(mask_j)
            token_j = token_j + row_context + self.type_embedding.weight[0]
            outputs.append(self.heads[column_idx](token_j))
        return outputs


class TOSEModel(nn.Module):
    """Target-conditional observed-set model for tabular imputation."""

    def __init__(
        self,
        input_dim: int,
        num_dim: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        mechanism_hidden_dim: int,
        cross_attn_layers: int,
        cat_bin_num: torch.Tensor | list[int] | tuple[int, ...] = (),
        cat_codebook_sizes: list[int] | tuple[int, ...] = (),
        ablation_mode: str = "full",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_dim = num_dim
        self.cat_dim = input_dim - num_dim
        self.ablation_mode = ablation_mode
        linear_readout = ablation_mode == "linear_prediction_heads"

        self.encoder = TargetConditionalObservedSetEncoder(
            input_dim=input_dim,
            num_dim=num_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            mechanism_hidden_dim=mechanism_hidden_dim,
            cross_attn_layers=cross_attn_layers,
            ablation_mode=ablation_mode,
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        self.forward_decoder = SimpleFieldwiseHead(
            input_dim=input_dim,
            embed_dim=embed_dim,
            dropout=dropout,
            linear_readout=linear_readout,
        )
        self.forward_cat_classifier = TargetWiseCategoricalHead(
            num_dim=num_dim,
            cat_bin_num=cat_bin_num,
            cat_codebook_sizes=cat_codebook_sizes,
            embed_dim=embed_dim,
            dropout=dropout,
            linear_readout=linear_readout,
        )

    def forward(
        self,
        input_x: torch.Tensor,
        missing_mask: torch.Tensor,
        decoder_forward_mask: torch.Tensor | None = None,
    ) -> TOSEOutputs:
        enc = self.encoder(input_x=input_x, missing_mask=missing_mask)
        z = self.final_norm(enc["z_init"])
        field_tokens = enc["field_tokens"]

        if decoder_forward_mask is None:
            decoder_forward_mask = missing_mask

        forward_fields = self.forward_decoder(
            row_repr=z,
            field_tokens=field_tokens,
            condition_mask=decoder_forward_mask,
            use_state_embedding=self.ablation_mode != "no_state_embedding",
        )
        forward_num = forward_fields[:, : self.num_dim]
        forward_cat_logits = forward_fields[:, self.num_dim :]
        forward_cat_class_logits = self.forward_cat_classifier(
            row_repr=z,
            field_tokens=field_tokens,
            condition_mask=decoder_forward_mask,
            use_state_embedding=self.ablation_mode != "no_state_embedding",
        )

        return TOSEOutputs(
            z_init=enc["z_init"],
            z_final=z,
            forward_num=forward_num,
            forward_cat_logits=forward_cat_logits,
            forward_cat_class_logits=forward_cat_class_logits,
        )
