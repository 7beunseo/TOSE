from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from config import TOSEConfig
from dataset import BenchmarkBundle, DeterministicBatchPlanner, build_batch, move_batch_to_device
from losses import TOSEForwardLoss
from metrics import compose_raw_prediction, compute_forward_metrics
from model import TOSEModel
from utils import (
    append_text_line,
    atomic_torch_save,
    capture_rng_state,
    format_float,
    make_versioned_dir,
    resolve_device,
    restore_rng_state,
    save_json,
)
from visualization import save_epoch_dashboard


MPGR_ENABLED = False
TOSE_ENABLED = True
ROW_MISSINGNESS_CONTEXT_ENABLED = False
REMASK_POLICY = "uniform_without_replacement"
CATEGORY_HEAD_POLICY = "target_wise_direct_classification"


class TOSETrainer:
    def __init__(self, config: TOSEConfig, bundle: BenchmarkBundle) -> None:
        self.config = config
        self.bundle = bundle
        self.device = resolve_device(config.device)
        self.use_amp = bool(config.amp) and self.device.type == "cuda"
        self.disable_tqdm = os.environ.get("TQDM_DISABLE", "0") == "1"
        if config.selection_split == "test":
            raise ValueError(
                "selection_split='test' is unsupported when test prediction during training is disabled. "
                "Use selection_split='valid'."
            )

        model_kwargs = dict(
            input_dim=bundle.train_split.model_x.shape[1],
            num_dim=bundle.num_dim,
            embed_dim=config.embed_dim,
            depth=config.encoder_depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            mechanism_hidden_dim=config.mechanism_hidden_dim,
            cross_attn_layers=config.cross_attn_layers,
            ablation_mode=config.ablation_mode,
            cat_bin_num=bundle.cat_bin_num,
            cat_codebook_sizes=[int(codebook.shape[0]) for codebook in bundle.cat_codebooks],
        )
        self.model = TOSEModel(**model_kwargs).to(self.device)
        self.loss_module = TOSEForwardLoss(
            num_dim=bundle.num_dim,
            forward_keep_loss_weight=config.forward_keep_loss_weight,
            forward_aux_loss_weight=config.forward_aux_loss_weight,
            loss_balance_mode=config.loss_balance_mode,
            lambda_num=config.lambda_num,
            lambda_cat=config.lambda_cat,
            cat_bit_beta=config.cat_bit_beta,
            cat_direct_ce_loss_weight=config.cat_direct_ce_loss_weight,
            cat_codebook_loss_weight=config.cat_codebook_loss_weight,
            cat_bit_aux_loss_weight=config.cat_bit_aux_loss_weight,
            cat_bin_num=bundle.cat_bin_num,
            cat_codebooks=bundle.cat_codebooks,
        ).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.train_planner = DeterministicBatchPlanner(
            num_samples=bundle.train_split.num_samples,
            batch_size=config.batch_size,
            seed=config.seed,
            shuffle=True,
        )
        self.eval_planner_train = DeterministicBatchPlanner(
            num_samples=bundle.train_split.num_samples,
            batch_size=config.eval_batch_size,
            seed=config.seed,
            shuffle=False,
        )
        self.eval_planner_valid = DeterministicBatchPlanner(
            num_samples=bundle.valid_split.num_samples,
            batch_size=config.eval_batch_size,
            seed=config.seed,
            shuffle=False,
        )
        self.eval_planner_test = DeterministicBatchPlanner(
            num_samples=bundle.test_split.num_samples,
            batch_size=config.eval_batch_size,
            seed=config.seed,
            shuffle=False,
        )
        restart_steps = max(1, self.train_planner.num_batches * config.restart_epochs)
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=restart_steps,
            T_mult=config.restart_mult,
            eta_min=config.min_lr,
        )

        self.global_step = 0
        self.start_epoch = 0
        self.start_batch_in_epoch = 0
        self.history: list[dict[str, Any]] = []
        self.best_score = None
        self.best_report: dict[str, Any] | None = None
        self.epochs_since_improvement = 0
        self.stopped_early = False
        self.stop_reason: str | None = None
        self.run_dir = self._prepare_run_dir()
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.plot_dir = self.run_dir / "plots"
        self.log_path = self.run_dir / "train_log.txt"
        self.history_path = self.run_dir / "history.json"
        self.report_path = self.run_dir / "report.json"
        self.config_path = self.run_dir / "config.json"
        self.last_ckpt_path = self.ckpt_dir / "last.pt"
        self.best_ckpt_path = self.ckpt_dir / "best.pt"
        self.step_ckpt_path = self.ckpt_dir / "last_step.pt"

        save_json(self.config_path, asdict(config))
        architecture = self._architecture_metadata()
        append_text_line(
            self.log_path,
            "[model] "
            f"loss_balance_mode={config.loss_balance_mode}, "
            "objective=forward_only, "
            f"ablation_variant={config.ablation_mode}, "
            f"mpgr_enabled={MPGR_ENABLED}, "
            f"tose_enabled={architecture['tose_enabled']}, "
            f"row_missingness_context_enabled={ROW_MISSINGNESS_CONTEXT_ENABLED}, "
            "row_missingness_context_implementation=absent, "
            f"remask_policy={REMASK_POLICY}, "
            f"encoder_policy={architecture['encoder_policy']}, "
            f"source_aggregation_policy={architecture['source_aggregation_policy']}, "
            f"category_head={CATEGORY_HEAD_POLICY}",
        )

        if config.resume_checkpoint is not None:
            self._load_checkpoint(Path(config.resume_checkpoint))

    def _prepare_run_dir(self) -> Path:
        if self.config.resume_checkpoint is not None:
            return Path(self.config.resume_checkpoint).resolve().parent.parent
        base_dir = (
            self.config.output_root
            / self.config.dataset
            / f"rate{self.config.ratio}"
            / self.config.mask_type
            / f"split_{self.config.split_idx}"
            / self.config.run_tag
        )
        return make_versioned_dir(base_dir)

    def _architecture_metadata(self) -> dict[str, Any]:
        mode = self.config.ablation_mode
        no_tose = mode == "no_tose_shared_row"
        uniform_sources = mode in {
            "uniform_source_aggregation",
            "uniform_source_no_target_set_sa",
            "both_uniform",
            "observed1_tose1_uniform_source",
            "observed1_tose1_uniform_source_no_target_set_sa",
            "observed1_tose1_uniform_source_no_observed_set_sa",
            "observed1_tose1_all_three_removed",
        }
        no_target_set_sa = mode in {
            "uniform_source_no_target_set_sa",
            "observed1_tose1_uniform_source_no_target_set_sa",
            "observed1_tose1_all_three_removed",
            "observed1_tose1_no_target_set_sa",
        }
        no_observed_set_sa = mode in {
            "no_observed_self_attention",
            "observed1_tose1_uniform_source_no_observed_set_sa",
            "observed1_tose1_all_three_removed",
            "observed1_tose1_no_observed_set_sa",
        }
        shared_target_query = mode == "observed1_tose1_shared_target_query"
        static_column_attention = mode == "observed1_tose1_static_column_attention"
        linear_prediction_heads = mode == "linear_prediction_heads"
        uniform_target_pooling = mode in {"no_target_aware_pooling", "both_uniform"}
        state_enabled = mode != "no_state_embedding"
        target_conditioning = mode not in {
            "no_target_conditioning",
            "no_tose_shared_row",
            "observed1_tose1_shared_target_query",
        }

        if no_tose:
            encoder_policy = "shared_observed_row_capacity_matched"
            source_aggregation = "learned_global_observed_pool"
        elif mode == "both_uniform":
            encoder_policy = "target_conditional_observed_set:both_uniform"
            source_aggregation = "uniform_over_observed_sources"
        elif mode == "observed1_tose1_all_three_removed":
            encoder_policy = "target_conditional_raw_observed:uniform_source_no_target_set_sa"
            source_aggregation = "uniform_over_raw_observed_sources"
        elif mode == "observed1_tose1_uniform_source_no_observed_set_sa":
            encoder_policy = "target_conditional_raw_observed:uniform_source_aggregation"
            source_aggregation = "uniform_over_raw_observed_sources"
        elif mode == "observed1_tose1_no_observed_set_sa":
            encoder_policy = "target_conditional_raw_observed:learned_cross_attention"
            source_aggregation = "learned_cross_attention"
        elif mode == "observed1_tose1_no_target_set_sa":
            encoder_policy = "target_conditional_observed_set:learned_source_independent_targets"
            source_aggregation = "learned_cross_attention"
        elif shared_target_query:
            encoder_policy = "target_specific_residual:shared_target_cross_attention_query"
            source_aggregation = "learned_cross_attention_shared_target_query"
        elif static_column_attention:
            encoder_policy = "target_conditional_observed_set:static_column_attention"
            source_aggregation = "static_target_source_column_attention"
        elif linear_prediction_heads:
            encoder_policy = "target_conditional_observed_set:full"
            source_aggregation = "learned_cross_attention"
        elif no_target_set_sa:
            encoder_policy = "target_conditional_observed_set:uniform_source_independent_targets"
            source_aggregation = "uniform_over_observed_sources"
        elif uniform_sources:
            encoder_policy = "target_conditional_observed_set:uniform_source_aggregation"
            source_aggregation = "uniform_over_observed_sources"
        else:
            encoder_policy = f"target_conditional_observed_set:{mode}"
            source_aggregation = "learned_cross_attention"

        return {
            "tose_enabled": not no_tose,
            "tose_target_block_count": len(self.model.encoder.target_blocks),
            "observed_set_block_count": len(self.model.encoder.observed_blocks),
            "shared_row_encoder_enabled": no_tose,
            "shared_row_block_count": len(self.model.encoder.shared_row_blocks),
            "capacity_matched_shared_row_blocks": no_tose,
            "target_query_enabled": not no_tose,
            "shared_target_query_enabled": shared_target_query,
            "target_query_policy": (
                "shared_learned_query"
                if shared_target_query
                else "target_column_identity_query"
                if static_column_attention
                else "target_specific_query"
                if not no_tose
                else "not_applicable"
            ),
            "target_cross_attention_enabled": not no_tose and not uniform_sources,
            "target_set_self_attention_enabled": (
                not no_tose and not no_target_set_sa
            ),
            "target_aware_pooling_enabled": not no_tose and not uniform_target_pooling,
            "target_pooling_policy": (
                "uniform_over_targets"
                if uniform_target_pooling
                else "learned_target_state_conditioned"
                if not no_tose
                else "not_applicable"
            ),
            "target_pool_state_concat_enabled": (
                state_enabled and not no_tose and not uniform_target_pooling
            ),
            "head_mask_embedding_enabled": state_enabled,
            "observed_state_embedding_enabled": state_enabled,
            "target_state_embedding_enabled": state_enabled,
            "target_conditioning_enabled": target_conditioning,
            "target_conditioned_source_selection_enabled": (
                target_conditioning and not uniform_sources
            ),
            "static_column_attention_enabled": static_column_attention,
            "row_adaptive_source_selection_enabled": (
                not no_tose and not uniform_sources and not static_column_attention
            ),
            "source_attention_score_inputs": (
                "target_and_source_column_identity"
                if static_column_attention
                else "none_uniform"
                if uniform_sources
                else "target_and_observed_row_representations"
                if not no_tose
                else "not_applicable"
            ),
            "source_value_inputs": (
                "actual_observed_row_tokens" if not no_tose else "not_applicable"
            ),
            "observed_self_attention_enabled": not no_observed_set_sa,
            "source_aggregation_policy": source_aggregation,
            "encoder_policy": encoder_policy,
            "prediction_head_policy": (
                "single_linear_readout" if linear_prediction_heads else "nonlinear_mlp"
            ),
            "prediction_head_hidden_layers": 0 if linear_prediction_heads else 1,
        }

    def _next_position(self, epoch: int, batch_in_epoch: int, total_batches: int) -> tuple[int, int]:
        next_batch = batch_in_epoch + 1
        next_epoch = epoch
        if next_batch >= total_batches:
            next_epoch += 1
            next_batch = 0
        return next_epoch, next_batch

    def _checkpoint_payload(self, epoch: int, batch_in_epoch: int) -> dict[str, Any]:
        config_dict = asdict(self.config)
        config_dict["data_root"] = str(config_dict["data_root"])
        config_dict["output_root"] = str(config_dict["output_root"])
        return {
            "config": config_dict,
            "model": self.model.state_dict(),
            "loss_module": self.loss_module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "global_step": self.global_step,
            "history": self.history,
            "best_score": self.best_score,
            "best_report": self.best_report,
            "epochs_since_improvement": self.epochs_since_improvement,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "run_dir": str(self.run_dir),
            "rng_state": capture_rng_state(),
        }

    def _save_checkpoint(self, path: Path, epoch: int, batch_in_epoch: int) -> None:
        atomic_torch_save(path, self._checkpoint_payload(epoch=epoch, batch_in_epoch=batch_in_epoch))

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        try:
            self.model.load_state_dict(checkpoint["model"])
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load model checkpoint. The checkpoint is not architecture-compatible with this public "
                "forward-only model. Re-run without --resume-checkpoint or resume from a matching checkpoint."
            ) from exc
        self.loss_module.load_state_dict(checkpoint["loss_module"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.start_epoch = int(checkpoint["epoch"])
        self.start_batch_in_epoch = int(checkpoint["batch_in_epoch"])
        self.global_step = int(checkpoint["global_step"])
        self.history = list(checkpoint.get("history", []))
        self.best_score = checkpoint.get("best_score")
        self.best_report = checkpoint.get("best_report")
        self.epochs_since_improvement = int(checkpoint.get("epochs_since_improvement", 0))
        self.stopped_early = bool(checkpoint.get("stopped_early", False))
        self.stop_reason = checkpoint.get("stop_reason")
        if "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"])
        append_text_line(
            self.log_path,
            f"[resume] epoch={self.start_epoch} batch={self.start_batch_in_epoch} global_step={self.global_step}",
        )

    def _log(self, text: str) -> None:
        print(text, flush=True)
        append_text_line(self.log_path, text)

    def _score_from_metrics(self, metrics: dict[str, float]) -> float:
        rmse = metrics.get("rmse", float("nan"))
        bit_f1 = metrics.get("bit_f1", float("nan"))
        cat_acc = metrics.get("cat_accuracy", float("nan"))
        rmse_term = 1e6 if rmse != rmse else rmse
        f1_term = 0.0 if bit_f1 != bit_f1 else bit_f1
        acc_term = 0.0 if cat_acc != cat_acc else cat_acc
        return rmse_term - 0.1 * f1_term - 0.05 * acc_term

    def _selection_metrics_from_epoch(
        self,
        valid_metrics: dict[str, float],
        test_metrics: dict[str, float] | None,
    ) -> dict[str, float]:
        if self.config.selection_split == "valid":
            return valid_metrics
        if test_metrics is None:
            raise RuntimeError("selection_split='test' requires test metrics during training.")
        return test_metrics

    def _load_model_from_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        try:
            self.model.load_state_dict(checkpoint["model"])
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load model weights. The checkpoint architecture differs from this public forward-only "
                "model; start a fresh run or resume from a matching checkpoint."
            ) from exc
        self.loss_module.load_state_dict(checkpoint["loss_module"])

    def _forward_model(
        self,
        input_x: torch.Tensor,
        missing_mask: torch.Tensor,
        decoder_forward_mask: torch.Tensor | None = None,
    ):
        return self.model(
            input_x=input_x,
            missing_mask=missing_mask,
            decoder_forward_mask=decoder_forward_mask,
        )

    def _sample_uniform_remask(self, missing_mask: torch.Tensor) -> torch.Tensor:
        """Uniformly hide 25% of currently observed logical features."""
        ratio = float(self.config.guided_remask_ratio)
        if ratio <= 0.0:
            return torch.zeros_like(missing_mask)

        group_missing = self.bundle.collapse_feature_mask_to_groups(missing_mask.float())
        observed_group_mask = group_missing < 0.5
        aux_group_mask = torch.zeros_like(group_missing)

        for row_idx in range(observed_group_mask.size(0)):
            observed_idx = torch.nonzero(observed_group_mask[row_idx], as_tuple=False).squeeze(1)
            if observed_idx.numel() == 0:
                continue

            k = min(int(round(observed_idx.numel() * ratio)), int(observed_idx.numel()))
            if k <= 0:
                continue

            permutation = torch.randperm(observed_idx.numel(), device=observed_idx.device)
            chosen = observed_idx[permutation[:k]]
            aux_group_mask[row_idx, chosen] = 1.0

        return self.bundle.expand_group_mask(aux_group_mask)

    def _predict_split(self, split_name: str) -> tuple[dict[str, float], np.ndarray]:
        self.model.eval()
        split = self.bundle.get_split(split_name)
        if split_name == "train":
            planner = self.eval_planner_train
        elif split_name == "valid":
            planner = self.eval_planner_valid
        elif split_name == "test":
            planner = self.eval_planner_test
        else:
            raise ValueError(f"Unsupported split name: {split_name}")

        pred_num_chunks = []
        pred_cat_chunks = []
        pred_cat_class_chunks: list[list[torch.Tensor]] | None = None

        with torch.no_grad():
            epoch_indices = planner.build_epoch_indices(epoch=0)
            iterator = tqdm(epoch_indices, desc=f"{self.config.dataset}-eval-{split_name}", leave=False)
            if self.disable_tqdm:
                iterator.disable = True
            for indices in iterator:
                batch = build_batch(self.bundle, split, indices)
                batch = move_batch_to_device(batch, self.device)
                outputs = self._forward_model(
                    input_x=batch["input_x"],
                    missing_mask=batch["missing_mask"],
                )
                pred_num_chunks.append(outputs.forward_num.detach().cpu())
                pred_cat_chunks.append(outputs.forward_cat_logits.detach().cpu())
                if len(outputs.forward_cat_class_logits) > 0:
                    if pred_cat_class_chunks is None:
                        pred_cat_class_chunks = [[] for _ in outputs.forward_cat_class_logits]
                    for column_idx, logits_j in enumerate(outputs.forward_cat_class_logits):
                        pred_cat_class_chunks[column_idx].append(logits_j.detach().cpu())

        pred_num = torch.cat(pred_num_chunks, dim=0)
        pred_cat_logits = (
            torch.cat(pred_cat_chunks, dim=0) if self.bundle.bit_dim > 0 else torch.zeros((split.num_samples, 0))
        )
        pred_cat_class_logits = None
        if pred_cat_class_chunks is not None:
            pred_cat_class_logits = [torch.cat(chunks, dim=0) for chunks in pred_cat_class_chunks]

        pred_bundle = compose_raw_prediction(
            bundle=self.bundle,
            split_name=split_name,
            pred_num_std=pred_num,
            pred_cat_logits=pred_cat_logits,
            pred_cat_class_logits=pred_cat_class_logits,
        )
        metrics = compute_forward_metrics(self.bundle, split_name=split_name, prediction=pred_bundle)
        return metrics, pred_bundle.pred_raw

    def _write_epoch_artifacts(self, epoch_summary: dict[str, Any]) -> None:
        save_json(self.history_path, self.history)
        save_epoch_dashboard(self.history, self.plot_dir / f"epoch_{epoch_summary['epoch']:04d}.png")
        save_epoch_dashboard(self.history, self.plot_dir / "latest.png")

        architecture = self._architecture_metadata()
        report = {
            "model_variant": f"tose_ablation_{self.config.ablation_mode}",
            "ablation_variant": self.config.ablation_mode,
            "mpgr_enabled": MPGR_ENABLED,
            "row_missingness_context_enabled": ROW_MISSINGNESS_CONTEXT_ENABLED,
            "row_missingness_context_implementation": "absent",
            **architecture,
            "double_row_normalization_enabled": True,
            "remask_policy": REMASK_POLICY,
            "category_head_policy": CATEGORY_HEAD_POLICY,
            "dataset": self.config.dataset,
            "split_idx": self.config.split_idx,
            "mask_type": self.config.mask_type,
            "ratio": self.config.ratio,
            "run_dir": str(self.run_dir),
            "selection_split": self.config.selection_split,
            "split_sizes": self.bundle.split_sizes(),
            "best_score": self.best_score,
            "best_report": self.best_report,
            "epochs_since_improvement": self.epochs_since_improvement,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "latest_epoch": epoch_summary["epoch"],
            "latest_metrics": epoch_summary,
        }
        save_json(self.report_path, report)

    def train(self) -> dict[str, Any]:
        total_start = time.time()
        self._log(
            f"[start] dataset={self.config.dataset} split={self.config.split_idx} "
            f"device={self.device} run_dir={self.run_dir}"
        )

        for epoch in range(self.start_epoch, self.config.max_epochs):
            epoch_indices = self.train_planner.build_epoch_indices(epoch)
            total_batches = len(epoch_indices)
            batch_start = self.start_batch_in_epoch if epoch == self.start_epoch else 0
            self.model.train()
            self.loss_module.train()

            meter = {
                "loss_total": 0.0,
                "loss_forward": 0.0,
                "loss_forward_num": 0.0,
                "loss_forward_cat": 0.0,
                "loss_forward_cat_direct": 0.0,
                "loss_forward_cat_codebook": 0.0,
                "loss_forward_cat_bit": 0.0,
                "loss_forward_keep": 0.0,
                "loss_forward_aux": 0.0,
                "loss_forward_aux_num": 0.0,
                "loss_forward_aux_cat": 0.0,
                "loss_forward_aux_cat_direct": 0.0,
                "loss_forward_aux_cat_codebook": 0.0,
                "loss_forward_aux_cat_bit": 0.0,
                "steps": 0,
            }

            progress = tqdm(
                range(batch_start, total_batches),
                desc=f"{self.config.dataset}-train-epoch-{epoch + 1}",
                leave=False,
                disable=self.disable_tqdm,
            )
            epoch_start = time.time()
            for batch_pos in progress:
                indices = epoch_indices[batch_pos]
                batch = build_batch(self.bundle, self.bundle.train_split, indices)
                batch = move_batch_to_device(batch, self.device)
                aux_mask = self._sample_uniform_remask(batch["missing_mask"])
                aux_cat_mask = aux_mask[:, self.bundle.num_dim :].float()
                aux_cat_col_mask = self.bundle.collapse_cat_bit_mask(aux_cat_mask).float()
                batch["guided_aux_mask"] = aux_mask.float()
                batch["guided_aux_num_mask"] = aux_mask[:, : self.bundle.num_dim].float()
                batch["guided_aux_cat_mask"] = aux_cat_mask
                batch["guided_aux_cat_col_mask"] = aux_cat_col_mask
                keep_mask = (batch["observed_mask"] - aux_mask.float()).clamp_min(0.0)
                batch["guided_keep_mask"] = keep_mask.float()
                batch["guided_keep_num_mask"] = keep_mask[:, : self.bundle.num_dim].float()
                batch["guided_keep_cat_mask"] = keep_mask[:, self.bundle.num_dim :].float()
                batch["guided_keep_cat_col_mask"] = (
                    batch["observed_cat_col_mask"] - aux_cat_col_mask
                ).clamp_min(0.0)
                model_input_x = batch["input_x"].masked_fill(aux_mask.bool(), 0.0)
                model_missing_mask = torch.maximum(batch["missing_mask"], aux_mask.float())
                self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self._forward_model(
                        input_x=model_input_x,
                        missing_mask=model_missing_mask,
                        decoder_forward_mask=model_missing_mask,
                    )
                    loss, loss_stats = self.loss_module(outputs, batch, epoch=epoch)

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss detected at epoch={epoch + 1}, batch={batch_pos + 1}, loss={loss.item()}"
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                self.global_step += 1
                meter["steps"] += 1
                for key in [
                    "loss_total",
                    "loss_forward",
                    "loss_forward_num",
                    "loss_forward_cat",
                    "loss_forward_cat_direct",
                    "loss_forward_cat_codebook",
                    "loss_forward_cat_bit",
                    "loss_forward_keep",
                    "loss_forward_aux",
                    "loss_forward_aux_num",
                    "loss_forward_aux_cat",
                    "loss_forward_aux_cat_direct",
                    "loss_forward_aux_cat_codebook",
                    "loss_forward_aux_cat_bit",
                ]:
                    value = loss_stats[key]
                    meter[key] += float(value.item() if hasattr(value, "item") else value)

                avg_total = meter["loss_total"] / meter["steps"]
                progress.set_postfix(
                    loss=f"{avg_total:.4f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    step=self.global_step,
                )

                if self.global_step % self.config.log_every_steps == 0:
                    self._log(
                        f"[step] dataset={self.config.dataset} epoch={epoch + 1} "
                        f"batch={batch_pos + 1}/{total_batches} "
                        f"global_step={self.global_step} loss={avg_total:.6f} "
                        f"lr={self.optimizer.param_groups[0]['lr']:.6g}"
                    )

                next_epoch, next_batch = self._next_position(epoch, batch_pos, total_batches)
                self._save_checkpoint(self.step_ckpt_path, epoch=next_epoch, batch_in_epoch=next_batch)
                if self.global_step % self.config.checkpoint_every_steps == 0:
                    self._save_checkpoint(self.last_ckpt_path, epoch=next_epoch, batch_in_epoch=next_batch)

            self.start_batch_in_epoch = 0

            train_metrics, train_pred_raw = self._predict_split("train")
            valid_metrics, valid_pred_raw = self._predict_split("valid")
            test_metrics = None

            epoch_summary = {
                "epoch": epoch + 1,
                "global_step": self.global_step,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "epoch_seconds": time.time() - epoch_start,
                "total_seconds": time.time() - total_start,
                "train_loss_total": meter["loss_total"] / max(1, meter["steps"]),
                "train_loss_forward": meter["loss_forward"] / max(1, meter["steps"]),
                "train_loss_forward_num": meter["loss_forward_num"] / max(1, meter["steps"]),
                "train_loss_forward_cat": meter["loss_forward_cat"] / max(1, meter["steps"]),
                "train_loss_forward_cat_direct": meter["loss_forward_cat_direct"] / max(1, meter["steps"]),
                "train_loss_forward_cat_codebook": meter["loss_forward_cat_codebook"] / max(1, meter["steps"]),
                "train_loss_forward_cat_bit": meter["loss_forward_cat_bit"] / max(1, meter["steps"]),
                "train_loss_forward_keep": meter["loss_forward_keep"] / max(1, meter["steps"]),
                "train_loss_forward_aux": meter["loss_forward_aux"] / max(1, meter["steps"]),
                "train_loss_forward_aux_num": meter["loss_forward_aux_num"] / max(1, meter["steps"]),
                "train_loss_forward_aux_cat": meter["loss_forward_aux_cat"] / max(1, meter["steps"]),
                "train_loss_forward_aux_cat_direct": meter["loss_forward_aux_cat_direct"] / max(1, meter["steps"]),
                "train_loss_forward_aux_cat_codebook": meter["loss_forward_aux_cat_codebook"] / max(1, meter["steps"]),
                "train_loss_forward_aux_cat_bit": meter["loss_forward_aux_cat_bit"] / max(1, meter["steps"]),
                "train_rmse": train_metrics.get("rmse"),
                "train_mae": train_metrics.get("mae"),
                "train_bit_f1": train_metrics.get("bit_f1"),
                "train_cat_accuracy": train_metrics.get("cat_accuracy"),
                "train_invalid_code_rate": train_metrics.get("invalid_code_rate"),
                "valid_rmse": valid_metrics.get("rmse"),
                "valid_mae": valid_metrics.get("mae"),
                "valid_bit_f1": valid_metrics.get("bit_f1"),
                "valid_cat_accuracy": valid_metrics.get("cat_accuracy"),
                "valid_invalid_code_rate": valid_metrics.get("invalid_code_rate"),
            }
            if test_metrics is not None:
                epoch_summary.update(
                    {
                        "test_rmse": test_metrics.get("rmse"),
                        "test_mae": test_metrics.get("mae"),
                        "test_bit_f1": test_metrics.get("bit_f1"),
                        "test_cat_accuracy": test_metrics.get("cat_accuracy"),
                        "test_invalid_code_rate": test_metrics.get("invalid_code_rate"),
                    }
                )
            self.history.append(epoch_summary)

            np.save(self.run_dir / "train_pred_raw.npy", train_pred_raw)
            np.save(self.run_dir / "valid_pred_raw.npy", valid_pred_raw)

            selection_metrics = self._selection_metrics_from_epoch(valid_metrics=valid_metrics, test_metrics=test_metrics)
            self._log(
                "[epoch] "
                f"dataset={self.config.dataset} "
                f"epoch={epoch_summary['epoch']}/{self.config.max_epochs} "
                f"loss={format_float(epoch_summary['train_loss_total'])} "
                f"forward={format_float(epoch_summary['train_loss_forward'])} "
                f"num={format_float(epoch_summary['train_loss_forward_num'])} "
                f"cat={format_float(epoch_summary['train_loss_forward_cat'])} "
                f"cat_ce={format_float(epoch_summary['train_loss_forward_cat_direct'])} "
                f"keep={format_float(epoch_summary['train_loss_forward_keep'])} "
                f"aux={format_float(epoch_summary['train_loss_forward_aux'])} "
                f"train_rmse={format_float(epoch_summary['train_rmse'])} "
                f"{self.config.selection_split}_rmse={format_float(selection_metrics.get('rmse'))} "
                f"{self.config.selection_split}_f1={format_float(selection_metrics.get('bit_f1'))} "
                f"{self.config.selection_split}_acc={format_float(selection_metrics.get('cat_accuracy'))} "
                f"{self.config.selection_split}_invalid={format_float(selection_metrics.get('invalid_code_rate'))} "
                f"lr={format_float(epoch_summary['lr'])}"
            )

            current_score = self._score_from_metrics(selection_metrics)
            improved = self.best_score is None or current_score < (self.best_score - self.config.early_stop_min_delta)
            if improved:
                self.best_score = current_score
                self.best_report = {
                    "epoch": epoch + 1,
                    "score": current_score,
                    "selection_split": self.config.selection_split,
                    "train_metrics": train_metrics,
                    "valid_metrics": valid_metrics,
                }
                self.epochs_since_improvement = 0
                self._save_checkpoint(self.best_ckpt_path, epoch=epoch + 1, batch_in_epoch=0)
            else:
                self.epochs_since_improvement += 1

            if (
                self.config.early_stop_patience > 0
                and self.epochs_since_improvement >= self.config.early_stop_patience
            ):
                self.stopped_early = True
                self.stop_reason = (
                    "early_stop_patience_exhausted"
                    f"(patience={self.config.early_stop_patience}, "
                    f"min_delta={self.config.early_stop_min_delta}, "
                    f"epochs_since_improvement={self.epochs_since_improvement})"
                )
                self._log(
                    f"[early-stop] epoch={epoch + 1} "
                    f"patience={self.config.early_stop_patience} "
                    f"min_delta={format_float(self.config.early_stop_min_delta)} "
                    f"selection_split={self.config.selection_split} "
                    f"best_epoch={None if self.best_report is None else self.best_report['epoch']}"
                )

            self._save_checkpoint(self.last_ckpt_path, epoch=epoch + 1, batch_in_epoch=0)
            self._write_epoch_artifacts(epoch_summary)

            if self.stopped_early:
                break

        if self.best_ckpt_path.exists():
            self._load_model_from_checkpoint(self.best_ckpt_path)
            best_train_metrics, best_train_pred_raw = self._predict_split("train")
            best_valid_metrics, best_valid_pred_raw = self._predict_split("valid")
            best_test_metrics, best_test_pred_raw = self._predict_split("test")
            final_best_score = self._score_from_metrics(
                best_valid_metrics if self.config.selection_split == "valid" else best_test_metrics
            )
            np.save(self.run_dir / "train_pred_raw.npy", best_train_pred_raw)
            np.save(self.run_dir / "valid_pred_raw.npy", best_valid_pred_raw)
            np.save(self.run_dir / "test_pred_raw.npy", best_test_pred_raw)
            if self.best_report is None:
                self.best_report = {}
            self.best_report.update(
                {
                    "selection_split": self.config.selection_split,
                    "train_metrics": best_train_metrics,
                    "valid_metrics": best_valid_metrics,
                    "test_metrics": best_test_metrics,
                    "score": final_best_score,
                }
            )
            self.best_score = final_best_score

        architecture = self._architecture_metadata()
        final_report = {
            "model_variant": f"tose_ablation_{self.config.ablation_mode}",
            "ablation_variant": self.config.ablation_mode,
            "mpgr_enabled": MPGR_ENABLED,
            "row_missingness_context_enabled": ROW_MISSINGNESS_CONTEXT_ENABLED,
            "row_missingness_context_implementation": "absent",
            **architecture,
            "double_row_normalization_enabled": True,
            "remask_policy": REMASK_POLICY,
            "category_head_policy": CATEGORY_HEAD_POLICY,
            "dataset": self.config.dataset,
            "split_idx": self.config.split_idx,
            "mask_type": self.config.mask_type,
            "ratio": self.config.ratio,
            "run_dir": str(self.run_dir),
            "history_path": str(self.history_path),
            "selection_split": self.config.selection_split,
            "split_sizes": self.bundle.split_sizes(),
            "best_score": self.best_score,
            "best_report": self.best_report,
            "epochs_since_improvement": self.epochs_since_improvement,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "history": self.history,
        }
        save_json(self.report_path, final_report)
        self._log(
            f"[done] dataset={self.config.dataset} split={self.config.split_idx} "
            f"best_epoch={None if self.best_report is None else self.best_report['epoch']} "
            f"run_dir={self.run_dir}"
        )
        return final_report
