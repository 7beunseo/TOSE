from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from selection_dataset import BenchmarkMNARBundle


SELECTION_PROTOCOL = "baseline_normalized"
SELECTION_FORMULA = (
    "weighted_mean(rmse/reference_mean_rmse, "
    "(1-f1)/(1-reference_mode_f1), "
    "(1-accuracy)/(1-reference_mode_accuracy))"
)
NUMERIC_WEIGHT = 0.50
F1_WEIGHT = 0.25
ACCURACY_WEIGHT = 0.25
EPSILON = 1e-6

_active_reference: dict[str, Any] | None = None
_active_context: dict[str, Any] | None = None


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))


def _compute_reference(bundle: BenchmarkMNARBundle) -> dict[str, Any]:
    reference: dict[str, Any] = {
        "protocol": "train_mean_mode_evaluated_on_valid_missing_positions",
        "uses_test": False,
        "numeric_rmse": float("nan"),
        "categorical_f1_weighted": float("nan"),
        "categorical_accuracy": float("nan"),
    }

    if bundle.num_dim > 0:
        train_values = bundle.train_split.model_x[:, : bundle.num_dim].cpu().numpy()
        train_missing = bundle.train_split.missing_mask_ext[:, : bundle.num_dim].cpu().numpy().astype(bool)
        valid_values = bundle.valid_split.model_x[:, : bundle.num_dim].cpu().numpy()
        valid_missing = bundle.valid_split.missing_mask_ext[:, : bundle.num_dim].cpu().numpy().astype(bool)

        means = np.zeros(bundle.num_dim, dtype=np.float64)
        for feature_idx in range(bundle.num_dim):
            observed = ~train_missing[:, feature_idx]
            if observed.any():
                means[feature_idx] = float(np.mean(train_values[observed, feature_idx]))
        if valid_missing.any():
            prediction = np.broadcast_to(means[None, :], valid_values.shape)
            diff = prediction[valid_missing] - valid_values[valid_missing]
            reference["numeric_rmse"] = float(np.sqrt(np.mean(diff**2)))
        reference["numeric_train_mean"] = means.tolist()

    has_categories = (
        bundle.train_split.cat_idx is not None
        and bundle.valid_split.cat_idx is not None
        and len(bundle.cat_col_idx) > 0
    )
    if has_categories:
        train_cat_idx = bundle.train_split.cat_idx
        valid_cat_idx = bundle.valid_split.cat_idx
        train_original_mask = bundle.train_split.original_mask.cpu().numpy().astype(bool)
        valid_original_mask = bundle.valid_split.original_mask.cpu().numpy().astype(bool)
        modes: list[int] = []
        true_tokens: list[str] = []
        pred_tokens: list[str] = []
        correct = 0
        total = 0

        for local_idx, original_col_idx in enumerate(bundle.cat_col_idx):
            train_observed = ~train_original_mask[:, int(original_col_idx)]
            observed_values = train_cat_idx[train_observed, local_idx].astype(np.int64)
            cardinality = int(bundle.cat_codebooks_np[local_idx].shape[0])
            mode = 0 if observed_values.size == 0 else int(
                np.argmax(np.bincount(observed_values, minlength=cardinality))
            )
            modes.append(mode)

            valid_missing = valid_original_mask[:, int(original_col_idx)]
            if not valid_missing.any():
                continue
            true_values = valid_cat_idx[valid_missing, local_idx].astype(np.int64)
            pred_values = np.full(true_values.shape, mode, dtype=np.int64)
            true_tokens.extend(
                f"col{int(original_col_idx)}=idx{int(value)}" for value in true_values.tolist()
            )
            pred_tokens.extend(
                f"col{int(original_col_idx)}=idx{int(value)}" for value in pred_values.tolist()
            )
            correct += int(np.sum(true_values == pred_values))
            total += int(true_values.size)

        if true_tokens:
            reference["categorical_f1_weighted"] = float(
                f1_score(true_tokens, pred_tokens, average="weighted", zero_division=0)
            )
        if total > 0:
            reference["categorical_accuracy"] = float(correct / total)
        reference["categorical_train_mode"] = modes
        reference["categorical_token_format"] = "col{original_column_index}=idx{category_index}"

    return reference


def configure_selection(
    *,
    data_root: str | Path,
    dataset: str,
    mask_type: str,
    split_idx: int,
    ratio: str | int = "30",
    seed: int = 20260430,
    valid_max_fraction: float = 0.15,
) -> dict[str, Any]:
    global _active_context, _active_reference

    bundle = BenchmarkMNARBundle(
        dataset_name=dataset,
        mask_type=mask_type,
        split_idx=int(split_idx),
        ratio=str(ratio),
        data_root=data_root,
        seed=int(seed) + int(split_idx),
        valid_max_fraction=float(valid_max_fraction),
    )
    _active_reference = _compute_reference(bundle)
    _active_context = {
        "dataset": str(dataset),
        "split_idx": int(split_idx),
        "mask_type": str(mask_type),
        "ratio": str(ratio),
    }
    return selection_metadata()


def normalized_score(metrics: dict[str, Any]) -> float:
    score, _ = normalized_score_with_components(metrics)
    return score


def normalized_score_with_components(metrics: dict[str, Any]) -> tuple[float, dict[str, float]]:
    if _active_reference is None:
        raise RuntimeError("configure_selection() must be called before checkpoint scoring.")

    rmse = metrics.get("rmse", metrics.get("RMSE", float("nan")))
    f1 = metrics.get(
        "cat_f1_weighted",
        metrics.get("cat_f1", metrics.get("f1", metrics.get("F1", metrics.get("bit_f1", float("nan"))))),
    )
    accuracy = metrics.get(
        "cat_accuracy",
        metrics.get("accuracy", metrics.get("Accuracy", float("nan"))),
    )
    reference_rmse = _active_reference.get("numeric_rmse", float("nan"))
    reference_f1 = _active_reference.get("categorical_f1_weighted", float("nan"))
    reference_accuracy = _active_reference.get("categorical_accuracy", float("nan"))

    terms: list[tuple[str, float, float]] = []
    if _finite(rmse) and _finite(reference_rmse):
        terms.append(("numeric_relative_rmse", NUMERIC_WEIGHT, float(rmse) / max(float(reference_rmse), EPSILON)))
    if _finite(f1) and _finite(reference_f1):
        terms.append(
            (
                "categorical_relative_f1_error",
                F1_WEIGHT,
                (1.0 - float(f1)) / max(1.0 - float(reference_f1), EPSILON),
            )
        )
    if _finite(accuracy) and _finite(reference_accuracy):
        terms.append(
            (
                "categorical_relative_accuracy_error",
                ACCURACY_WEIGHT,
                (1.0 - float(accuracy)) / max(1.0 - float(reference_accuracy), EPSILON),
            )
        )

    if not terms:
        return 1e6, {}
    weight_sum = sum(weight for _, weight, _ in terms)
    components = {name: float(value) for name, _, value in terms}
    components["available_weight_sum"] = float(weight_sum)
    score = sum(weight * value for _, weight, value in terms) / max(weight_sum, EPSILON)
    return float(score), components


def selection_metadata() -> dict[str, Any]:
    return {
        "checkpoint_selection": SELECTION_PROTOCOL,
        "selection_formula": SELECTION_FORMULA,
        "selection_weights": {
            "numeric_rmse": NUMERIC_WEIGHT,
            "categorical_f1_error": F1_WEIGHT,
            "categorical_accuracy_error": ACCURACY_WEIGHT,
        },
        "selection_reference": _active_reference,
        "selection_context": _active_context,
        "selection_uses_test": False,
    }
