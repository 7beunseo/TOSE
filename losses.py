from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import masked_mean


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.numel() == 0:
        return target.new_tensor(0.0)
    return masked_mean((pred - target) ** 2, mask)


def masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0:
        return target.new_tensor(0.0)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return masked_mean(loss, mask)


def categorical_codebook_ce_loss(
    cat_logits: torch.Tensor,
    target_cat_idx: torch.Tensor,
    cat_col_mask: torch.Tensor,
    cat_codebooks: list[torch.Tensor],
    cat_bin_num: torch.Tensor,
) -> torch.Tensor:
    if (
        cat_logits.numel() == 0
        or target_cat_idx.numel() == 0
        or cat_col_mask.numel() == 0
        or len(cat_codebooks) == 0
    ):
        return cat_logits.new_tensor(0.0)

    losses = []
    start = 0
    for column_idx, bit_len in enumerate(cat_bin_num.tolist()):
        end = start + int(bit_len)
        logits_j = cat_logits[:, start:end]
        codebook_j = cat_codebooks[column_idx].to(logits_j.device)
        target_j = target_cat_idx[:, column_idx]
        mask_j = cat_col_mask[:, column_idx].float()
        mask_sum = mask_j.sum()
        if mask_sum <= 0:
            start = end
            continue

        logits_expanded = logits_j.unsqueeze(1)
        codebook_expanded = codebook_j.unsqueeze(0)
        log_prob_1 = F.logsigmoid(logits_expanded)
        log_prob_0 = F.logsigmoid(-logits_expanded)
        scores = (codebook_expanded * log_prob_1 + (1.0 - codebook_expanded) * log_prob_0).sum(dim=-1)
        ce = F.cross_entropy(scores, target_j, reduction="none")
        losses.append((ce * mask_j).sum() / mask_sum.clamp_min(1.0))
        start = end

    if not losses:
        return cat_logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def categorical_direct_ce_loss(
    cat_class_logits: list[torch.Tensor],
    target_cat_idx: torch.Tensor,
    cat_col_mask: torch.Tensor,
) -> torch.Tensor:
    if (
        len(cat_class_logits) == 0
        or target_cat_idx.numel() == 0
        or cat_col_mask.numel() == 0
    ):
        if len(cat_class_logits) > 0:
            return cat_class_logits[0].new_tensor(0.0)
        return target_cat_idx.new_tensor(0.0, dtype=torch.float32)

    losses = []
    for column_idx, logits_j in enumerate(cat_class_logits):
        target_j = target_cat_idx[:, column_idx]
        mask_j = cat_col_mask[:, column_idx].float()
        mask_sum = mask_j.sum()
        if mask_sum <= 0:
            continue
        ce = F.cross_entropy(logits_j, target_j, reduction="none")
        losses.append((ce * mask_j).sum() / mask_sum.clamp_min(1.0))

    if not losses:
        return cat_class_logits[0].new_tensor(0.0)
    return torch.stack(losses).mean()


class TOSEForwardLoss(nn.Module):
    """TOSE forward-only reconstruction objective.

    The public TOSE objective intentionally contains only the proposed
    forward reconstruction losses. All non-forward auxiliary objectives and
    learned objective-scale terms are absent from this implementation.
    """

    def __init__(
        self,
        num_dim: int,
        forward_keep_loss_weight: float,
        forward_aux_loss_weight: float,
        loss_balance_mode: str,
        lambda_num: float,
        lambda_cat: float,
        cat_bit_beta: float,
        cat_direct_ce_loss_weight: float,
        cat_codebook_loss_weight: float,
        cat_bit_aux_loss_weight: float,
        cat_bin_num: torch.Tensor,
        cat_codebooks: list[torch.Tensor],
    ) -> None:
        super().__init__()
        self.num_dim = int(num_dim)
        self.forward_keep_loss_weight = float(forward_keep_loss_weight)
        self.forward_aux_loss_weight = float(forward_aux_loss_weight)
        self.loss_balance_mode = str(loss_balance_mode)

        lambda_sum = max(float(lambda_num) + float(lambda_cat), 1e-8)
        self.lambda_num = float(lambda_num) / lambda_sum
        self.lambda_cat = float(lambda_cat) / lambda_sum
        self.cat_bit_beta = float(cat_bit_beta)
        self.cat_direct_ce_loss_weight = float(cat_direct_ce_loss_weight)
        self.cat_codebook_loss_weight = float(cat_codebook_loss_weight)
        self.cat_bit_aux_loss_weight = float(cat_bit_aux_loss_weight)

        self.register_buffer("cat_bin_num", cat_bin_num.long().clone())
        self._cat_codebook_names: list[str] = []
        for idx, codebook in enumerate(cat_codebooks):
            name = f"cat_codebook_{idx}"
            self.register_buffer(name, codebook.float().clone())
            self._cat_codebook_names.append(name)

    def reconstruction_components(
        self,
        pred_num: torch.Tensor,
        pred_cat_logits: torch.Tensor,
        pred_cat_class_logits: list[torch.Tensor],
        target_num: torch.Tensor,
        target_cat: torch.Tensor,
        target_cat_idx: torch.Tensor,
        num_mask: torch.Tensor,
        cat_mask: torch.Tensor,
        cat_col_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        num_loss = masked_mse(pred_num, target_num, num_mask)
        cat_codebook_loss = pred_cat_logits.new_tensor(0.0)
        if self.cat_codebook_loss_weight > 0.0:
            cat_codebook_loss = categorical_codebook_ce_loss(
                cat_logits=pred_cat_logits,
                target_cat_idx=target_cat_idx,
                cat_col_mask=cat_col_mask,
                cat_codebooks=[getattr(self, name) for name in self._cat_codebook_names],
                cat_bin_num=self.cat_bin_num,
            )
        cat_direct_loss = categorical_direct_ce_loss(
            cat_class_logits=pred_cat_class_logits,
            target_cat_idx=target_cat_idx,
            cat_col_mask=cat_col_mask,
        )
        bit_loss_weight = (
            self.cat_bit_beta if self.loss_balance_mode == "type_mean" else self.cat_bit_aux_loss_weight
        )
        cat_bit_aux_loss = pred_cat_logits.new_tensor(0.0)
        if bit_loss_weight > 0.0:
            cat_bit_aux_loss = masked_bce_with_logits(pred_cat_logits, target_cat, cat_mask)

        if self.loss_balance_mode == "type_mean":
            cat_loss = (
                self.cat_direct_ce_loss_weight * cat_direct_loss
                + self.cat_codebook_loss_weight * cat_codebook_loss
                + bit_loss_weight * cat_bit_aux_loss
            )
            total_loss = self.lambda_num * num_loss + self.lambda_cat * cat_loss
        else:
            cat_loss = (
                self.cat_direct_ce_loss_weight * cat_direct_loss
                + self.cat_codebook_loss_weight * cat_codebook_loss
                + bit_loss_weight * cat_bit_aux_loss
            )
            total_loss = num_loss + cat_loss

        return {
            "total": total_loss,
            "num": num_loss,
            "cat": cat_loss,
            "cat_direct": cat_direct_loss,
            "cat_codebook": cat_codebook_loss,
            "cat_bit": cat_bit_aux_loss,
        }

    def forward(self, outputs, batch: dict[str, torch.Tensor], epoch: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        keep = self.reconstruction_components(
            pred_num=outputs.forward_num,
            pred_cat_logits=outputs.forward_cat_logits,
            pred_cat_class_logits=outputs.forward_cat_class_logits,
            target_num=batch["target_num"],
            target_cat=batch["target_cat"],
            target_cat_idx=batch["target_cat_idx"],
            num_mask=batch["guided_keep_num_mask"],
            cat_mask=batch["guided_keep_cat_mask"],
            cat_col_mask=batch["guided_keep_cat_col_mask"],
        )
        aux = self.reconstruction_components(
            pred_num=outputs.forward_num,
            pred_cat_logits=outputs.forward_cat_logits,
            pred_cat_class_logits=outputs.forward_cat_class_logits,
            target_num=batch["target_num"],
            target_cat=batch["target_cat"],
            target_cat_idx=batch["target_cat_idx"],
            num_mask=batch["guided_aux_num_mask"],
            cat_mask=batch["guided_aux_cat_mask"],
            cat_col_mask=batch["guided_aux_cat_col_mask"],
        )

        forward_loss = self.forward_keep_loss_weight * keep["total"] + self.forward_aux_loss_weight * aux["total"]
        stats = {
            "loss_total": forward_loss.detach(),
            "loss_forward": forward_loss.detach(),
            "loss_forward_num": keep["num"].detach(),
            "loss_forward_cat": keep["cat"].detach(),
            "loss_forward_cat_direct": keep["cat_direct"].detach(),
            "loss_forward_cat_codebook": keep["cat_codebook"].detach(),
            "loss_forward_cat_bit": keep["cat_bit"].detach(),
            "loss_forward_keep": keep["total"].detach(),
            "loss_forward_aux": aux["total"].detach(),
            "loss_forward_aux_num": aux["num"].detach(),
            "loss_forward_aux_cat": aux["cat"].detach(),
            "loss_forward_aux_cat_direct": aux["cat_direct"].detach(),
            "loss_forward_aux_cat_codebook": aux["cat_codebook"].detach(),
            "loss_forward_aux_cat_bit": aux["cat_bit"].detach(),
        }
        return forward_loss, stats
