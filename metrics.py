from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score

from dataset import BenchmarkBundle


def _log_sigmoid_np(values: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -values)


def _log_sigmoid_neg_np(values: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, values)


def recover_categories_from_codebooks(
    cat_logits: np.ndarray,
    cat_codebooks: list[np.ndarray],
    cat_bin_num: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(cat_bin_num) == 0:
        return (
            np.zeros((cat_logits.shape[0], 0), dtype=np.int64),
            np.zeros((cat_logits.shape[0], 0), dtype=np.int64),
        )

    pred_columns = []
    pred_bit_columns = []
    start = 0
    for column_idx, bit_len in enumerate(cat_bin_num.tolist()):
        end = start + int(bit_len)
        logits_j = cat_logits[:, start:end]
        codebook_j = cat_codebooks[column_idx].astype(np.float32)

        logits_expanded = logits_j[:, None, :]
        codebook_expanded = codebook_j[None, :, :]
        log_prob_1 = _log_sigmoid_np(logits_expanded)
        log_prob_0 = _log_sigmoid_neg_np(logits_expanded)
        scores = (codebook_expanded * log_prob_1 + (1.0 - codebook_expanded) * log_prob_0).sum(axis=-1)

        pred_idx = scores.argmax(axis=1).astype(np.int64)
        pred_bits = codebook_j[pred_idx].astype(np.int64)
        pred_columns.append(pred_idx)
        pred_bit_columns.append(pred_bits)
        start = end

    return np.stack(pred_columns, axis=1), np.concatenate(pred_bit_columns, axis=1)


def recover_categories_from_class_logits(
    cat_class_logits: list[torch.Tensor] | list[np.ndarray],
    cat_codebooks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if len(cat_class_logits) == 0:
        return (
            np.zeros((0, 0), dtype=np.int64),
            np.zeros((0, 0), dtype=np.int64),
        )

    pred_columns = []
    pred_bit_columns = []
    for column_idx, logits_j in enumerate(cat_class_logits):
        if isinstance(logits_j, torch.Tensor):
            logits_np = logits_j.detach().cpu().numpy()
        else:
            logits_np = np.asarray(logits_j)
        pred_idx = logits_np.argmax(axis=1).astype(np.int64)
        codebook_j = cat_codebooks[column_idx].astype(np.int64)
        pred_bits = codebook_j[pred_idx]
        pred_columns.append(pred_idx)
        pred_bit_columns.append(pred_bits)

    return np.stack(pred_columns, axis=1), np.concatenate(pred_bit_columns, axis=1)


def compute_invalid_code_rate(
    pred_cat_prob: np.ndarray,
    cat_codebooks: list[np.ndarray],
    cat_bin_num: np.ndarray,
    cat_col_mask: np.ndarray,
) -> float:
    if len(cat_bin_num) == 0 or pred_cat_prob.size == 0 or cat_col_mask.sum() == 0:
        return float("nan")

    threshold_bits = (pred_cat_prob > 0.5).astype(np.int64)
    invalid_columns = []
    start = 0
    for column_idx, bit_len in enumerate(cat_bin_num.tolist()):
        end = start + int(bit_len)
        bits_j = threshold_bits[:, start:end]
        codebook_j = cat_codebooks[column_idx].astype(np.int64)
        matches = (bits_j[:, None, :] == codebook_j[None, :, :]).all(axis=-1)
        invalid_columns.append((~matches.any(axis=1)).astype(np.float32))
        start = end

    invalid_mask = np.stack(invalid_columns, axis=1)
    return float(invalid_mask[cat_col_mask].mean())


@dataclass
class PredictionBundle:
    pred_raw: np.ndarray
    pred_num_std: np.ndarray
    pred_cat_logits: np.ndarray
    pred_cat_prob: np.ndarray
    pred_cat_bits: np.ndarray
    pred_cat_idx: np.ndarray


def compose_raw_prediction(
    bundle: BenchmarkBundle,
    split_name: str,
    pred_num_std: torch.Tensor,
    pred_cat_logits: torch.Tensor,
    pred_cat_class_logits: list[torch.Tensor] | None = None,
) -> PredictionBundle:
    split = bundle.get_split(split_name)
    pred_num_std_np = pred_num_std.detach().cpu().numpy()
    pred_cat_logits_np = (
        pred_cat_logits.detach().cpu().numpy()
        if pred_cat_logits.numel() > 0
        else np.zeros((split.num_samples, 0), dtype=np.float32)
    )
    if pred_cat_class_logits is not None and len(pred_cat_class_logits) > 0:
        pred_cat_idx_np, pred_cat_bits_np = recover_categories_from_class_logits(
            cat_class_logits=pred_cat_class_logits,
            cat_codebooks=bundle.cat_codebooks_np,
        )
        pred_cat_prob_np = pred_cat_bits_np.astype(np.float32)
    else:
        pred_cat_prob_np = (
            np.exp(_log_sigmoid_np(pred_cat_logits_np)) if pred_cat_logits_np.size > 0 else pred_cat_logits_np
        )
        pred_cat_idx_np, pred_cat_bits_np = recover_categories_from_codebooks(
            cat_logits=pred_cat_logits_np,
            cat_codebooks=bundle.cat_codebooks_np,
            cat_bin_num=bundle.cat_bin_num.cpu().numpy(),
        )

    pred_num_raw = pred_num_std_np * (bundle.std[: bundle.num_dim].cpu().numpy()[None, :] * 2.0) + bundle.mean[
        : bundle.num_dim
    ].cpu().numpy()[None, :]
    pred_raw = split.raw_x.cpu().numpy().copy()

    missing_ext = split.missing_mask_ext.cpu().numpy()
    if bundle.num_dim > 0:
        pred_raw[:, : bundle.num_dim][missing_ext[:, : bundle.num_dim]] = pred_num_raw[
            missing_ext[:, : bundle.num_dim]
        ]
    if bundle.bit_dim > 0:
        pred_raw[:, bundle.num_dim :][missing_ext[:, bundle.num_dim :]] = pred_cat_bits_np[
            missing_ext[:, bundle.num_dim :]
        ]

    return PredictionBundle(
        pred_raw=pred_raw,
        pred_num_std=pred_num_std_np,
        pred_cat_logits=pred_cat_logits_np,
        pred_cat_prob=pred_cat_prob_np,
        pred_cat_bits=pred_cat_bits_np,
        pred_cat_idx=pred_cat_idx_np,
    )


def compute_forward_metrics(
    bundle: BenchmarkBundle,
    split_name: str,
    prediction: PredictionBundle,
) -> dict[str, float]:
    split = bundle.get_split(split_name)
    true_raw = split.raw_x.cpu().numpy()
    true_norm = split.model_x.cpu().numpy()
    mask_original = split.original_mask.cpu().numpy()
    missing_ext = split.missing_mask_ext.cpu().numpy()

    metrics: dict[str, float] = {}
    num_mask = missing_ext[:, : bundle.num_dim]
    if bundle.num_dim > 0 and num_mask.sum() > 0:
        pred_num_norm = (
            prediction.pred_raw[:, : bundle.num_dim] - bundle.mean[: bundle.num_dim].cpu().numpy()[None, :]
        ) / (bundle.std[: bundle.num_dim].cpu().numpy()[None, :] * 2.0)
        diff = pred_num_norm[num_mask] - true_norm[:, : bundle.num_dim][num_mask]
        metrics["mse"] = float(np.mean(diff**2))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        metrics["mae"] = float(np.mean(np.abs(diff)))
        r2_values = []
        for column in range(bundle.num_dim):
            column_mask = num_mask[:, column]
            if int(column_mask.sum()) < 2:
                continue
            y_true = true_norm[:, column][column_mask].astype(np.float64)
            y_pred = pred_num_norm[:, column][column_mask].astype(np.float64)
            if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
                r2_values = []
                break
            denominator = float(np.sum((y_true - y_true.mean()) ** 2))
            if denominator <= np.finfo(np.float64).eps:
                continue
            numerator = float(np.sum((y_true - y_pred) ** 2))
            r2_values.append(1.0 - numerator / denominator)
        metrics["r2"] = float(np.mean(r2_values)) if r2_values else float("nan")
    else:
        metrics["mse"] = float("nan")
        metrics["rmse"] = float("nan")
        metrics["mae"] = float("nan")
        metrics["r2"] = float("nan")

    cat_mask = missing_ext[:, bundle.num_dim :]
    if bundle.bit_dim > 0 and cat_mask.sum() > 0:
        true_bits = true_raw[:, bundle.num_dim :].astype(np.int64)
        metrics["bit_f1_legacy_bitwise"] = float(
            f1_score(
                true_bits[cat_mask].reshape(-1),
                prediction.pred_cat_bits[cat_mask].reshape(-1),
                zero_division=0,
            )
        )
    else:
        metrics["bit_f1_legacy_bitwise"] = float("nan")

    if split.cat_idx is not None and len(bundle.cat_col_idx) > 0 and len(bundle.cat_bin_num) > 0:
        cat_col_mask = mask_original[:, bundle.cat_col_idx]
        if cat_col_mask.sum() > 0:
            true_tokens = []
            pred_tokens = []
            cat_col_idx_np = np.asarray(bundle.cat_col_idx, dtype=np.int64)
            for local_idx, original_col_idx in enumerate(cat_col_idx_np.tolist()):
                row_mask = cat_col_mask[:, local_idx]
                if not row_mask.any():
                    continue
                true_idx = split.cat_idx[:, local_idx][row_mask].astype(np.int64)
                pred_idx = prediction.pred_cat_idx[:, local_idx][row_mask].astype(np.int64)
                true_tokens.extend(f"col{int(original_col_idx)}=idx{int(value)}" for value in true_idx.tolist())
                pred_tokens.extend(f"col{int(original_col_idx)}=idx{int(value)}" for value in pred_idx.tolist())
            if len(true_tokens) > 0:
                metrics["cat_f1_weighted"] = float(
                    f1_score(true_tokens, pred_tokens, average="weighted", zero_division=0)
                )
                metrics["cat_f1_macro"] = float(
                    f1_score(true_tokens, pred_tokens, average="macro", zero_division=0)
                )
            else:
                metrics["cat_f1_weighted"] = float("nan")
                metrics["cat_f1_macro"] = float("nan")
            metrics["cat_f1"] = metrics["cat_f1_weighted"]
            metrics["f1"] = metrics["cat_f1_weighted"]
            metrics["bit_f1"] = metrics["cat_f1_weighted"]
            metrics["cat_accuracy"] = float(
                (prediction.pred_cat_idx[cat_col_mask] == split.cat_idx[cat_col_mask]).mean()
            )
            metrics["accuracy"] = metrics["cat_accuracy"]
        else:
            metrics["cat_f1"] = float("nan")
            metrics["cat_f1_weighted"] = float("nan")
            metrics["cat_f1_macro"] = float("nan")
            metrics["f1"] = float("nan")
            metrics["bit_f1"] = float("nan")
            metrics["cat_accuracy"] = float("nan")
            metrics["accuracy"] = float("nan")
        metrics["invalid_code_rate"] = compute_invalid_code_rate(
            pred_cat_prob=prediction.pred_cat_prob,
            cat_codebooks=bundle.cat_codebooks_np,
            cat_bin_num=bundle.cat_bin_num.cpu().numpy(),
            cat_col_mask=cat_col_mask,
        )
    else:
        metrics["cat_f1"] = float("nan")
        metrics["cat_f1_weighted"] = float("nan")
        metrics["cat_f1_macro"] = float("nan")
        metrics["f1"] = float("nan")
        metrics["bit_f1"] = float("nan")
        metrics["cat_accuracy"] = float("nan")
        metrics["accuracy"] = float("nan")
        metrics["invalid_code_rate"] = float("nan")

    metrics["MAE"] = metrics["mae"]
    metrics["RMSE"] = metrics["rmse"]
    metrics["NRMSE"] = metrics["rmse"]
    metrics["R2"] = metrics["r2"]
    metrics["F1"] = metrics["cat_f1"]
    metrics["Accuracy"] = metrics["cat_accuracy"]
    metrics["numeric_source"] = "prepared_npz_x_norm"
    metrics["scaler_source"] = "prepared_npz_mean_std"
    metrics["metric_space"] = "tose_normalized"
    metrics["numeric_metric_unit"] = "prepared_normalized_unit"
    metrics["numeric_metric_formula"] = "pred_num_norm - true_x_norm"
    metrics["numeric_metric_scale"] = "tose_x_norm_formula_(x-mean)/std/2.0"
    metrics["categorical_protocol"] = "model_native"
    metrics["categorical_metric_level"] = "column_prefixed_category_index"
    metrics["categorical_f1_average"] = "weighted_over_column_prefixed_category_indices"
    metrics["cat_f1_average"] = "weighted_over_column_prefixed_category_indices"
    metrics["categorical_token_format"] = "col{original_column_index}=idx{category_index}"
    metrics["missing_rate"] = float(split.missing_mask_ext.float().mean().item())
    return metrics
