import argparse
import copy
import csv
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


ADAPTER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ADAPTER_ROOT.parents[1]
OFFICIAL_DSAN_ROOT = Path(
    os.environ.get("DSAN_SOURCE_ROOT", PROJECT_ROOT / ".baseline_sources" / "dsan")
).resolve()
MODEL_DIR = OFFICIAL_DSAN_ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from dsan import Mapper, Model, MVIDataset  # noqa: E402

SHARED_DIR = next(
    parent / "_shared"
    for parent in Path(__file__).resolve().parents
    if (parent / "_shared" / "baseline_selection.py").exists()
)
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from baseline_selection import configure_selection, normalized_score, selection_metadata  # noqa: E402


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "dsan"
DEFAULT_DATASETS = [
    "adult",
    "bean",
    "default",
    "gesture",
    "letter",
    "magic",
    "news",
    "shoppers",
    "abalone",
    "bank",
    "breast",
    "california",
    "german",
    "mushroom",
]


class EmptyScaler:
    mean_ = np.array([], dtype=np.float32)
    scale_ = np.array([], dtype=np.float32)
    var_ = np.array([], dtype=np.float32)
    n_features_in_ = 0

    def fit(self, x):
        return self

    def transform(self, x):
        return np.asarray(x, dtype=np.float32)

    def inverse_transform(self, x):
        return np.asarray(x, dtype=np.float32)


class IdentityScaler:
    """Pass-through scaler for numeric values already stored in the prepared npz."""

    def __init__(self, n_features: int) -> None:
        self.mean_ = np.zeros(int(n_features), dtype=np.float32)
        self.scale_ = np.ones(int(n_features), dtype=np.float32)
        self.var_ = np.ones(int(n_features), dtype=np.float32)
        self.n_features_in_ = int(n_features)

    def fit(self, x):
        return self

    def transform(self, x):
        return np.asarray(x, dtype=np.float32)

    def inverse_transform(self, x):
        return np.asarray(x, dtype=np.float32)


def stable_array_digest(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values)
    import hashlib

    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slugify(text: str) -> str:
    chars = []
    for ch in text.strip().lower():
        chars.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "run"


def safe_mean(x: np.ndarray) -> float:
    return float("nan") if x.size == 0 else float(x.mean())


def build_train_valid_indices(
    train_size: int,
    test_size: int,
    seed: int,
    valid_max_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if train_size <= 1 or valid_max_fraction <= 0.0:
        return np.arange(train_size, dtype=np.int64), np.empty(0, dtype=np.int64)

    requested_valid_size = int(math.floor(train_size * valid_max_fraction))
    requested_valid_size = max(requested_valid_size, 1)
    valid_size = min(test_size, requested_valid_size, train_size - 1)
    if valid_size <= 0:
        return np.arange(train_size, dtype=np.int64), np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(int(seed) + 17)
    permutation = rng.permutation(train_size)
    valid_idx = np.sort(permutation[:valid_size]).astype(np.int64)
    train_idx = np.sort(permutation[valid_size:]).astype(np.int64)
    return train_idx, valid_idx


class BenchmarkDataBundle:
    def __init__(
        self,
        dataset_name: str,
        mask_type: str,
        split_idx: int,
        ratio: str,
        data_root: Path,
        seed: int,
        valid_max_fraction: float,
        require_strict_split: bool = True,
    ) -> None:
        self.dataset_name = dataset_name
        self.mask_type = mask_type
        self.split_idx = int(split_idx)
        self.ratio = str(ratio)
        self.data_root = Path(data_root)
        self.seed = int(seed)
        self.valid_max_fraction = float(valid_max_fraction)
        self.require_strict_split = bool(require_strict_split)
        self.dataset_dir = self.data_root / dataset_name
        self.info_path = self.data_root / "Info" / f"{dataset_name}.json"
        self.normalized_path = (
            self.dataset_dir / "normalized" / f"rate{self.ratio}" / mask_type / f"split_{split_idx}.npz"
        )

        with open(self.info_path, "r", encoding="utf-8") as f:
            self.info = json.load(f)
        self.num_col_idx = list(self.info["num_col_idx"])
        self.cat_col_idx = list(self.info["cat_col_idx"])

        normalized = np.load(self.normalized_path)
        self.metric_mean = normalized["mean"].astype(np.float32)
        self.metric_std = normalized["std"].astype(np.float32)
        normalized_files = set(normalized.files)
        self.strict_split = {"valid_x_norm", "valid_mask", "train_indices", "valid_indices"}.issubset(
            normalized_files
        )
        if self.require_strict_split and not self.strict_split:
            raise FileNotFoundError(
                f"Strict split arrays are required but missing from {self.normalized_path}. "
                "Expected valid_x_norm, valid_mask, train_indices, and valid_indices."
            )
        if self.strict_split:
            train_idx = normalized["train_indices"].astype(np.int64)
            valid_idx = normalized["valid_indices"].astype(np.int64)
            train_x_norm = normalized["train_x_norm"].astype(np.float32)
            valid_x_norm = normalized["valid_x_norm"].astype(np.float32)
            test_x_norm = normalized["test_x_norm"].astype(np.float32)
        else:
            full_train_x_norm = normalized["train_x_norm"].astype(np.float32)
            test_x_norm = normalized["test_x_norm"].astype(np.float32)
            train_idx, valid_idx = build_train_valid_indices(
                train_size=int(full_train_x_norm.shape[0]),
                test_size=int(test_x_norm.shape[0]),
                seed=self.seed,
                valid_max_fraction=self.valid_max_fraction,
            )
            train_x_norm = full_train_x_norm[train_idx]
            valid_x_norm = full_train_x_norm[valid_idx]

        raw_data = self._load_raw_eval_data()
        full_train_mask_original = raw_data["train_mask_original"]
        full_train_cat_idx = raw_data["train_cat_idx"]

        self.train_indices = train_idx
        self.valid_indices = valid_idx
        self.train_mask_original = full_train_mask_original[train_idx]
        self.valid_mask_original = full_train_mask_original[valid_idx]
        self.test_mask_original = raw_data["test_mask_original"]
        self.train_cat_idx = None if full_train_cat_idx is None else full_train_cat_idx[train_idx]
        self.valid_cat_idx = None if full_train_cat_idx is None else full_train_cat_idx[valid_idx]
        self.test_cat_idx = raw_data["test_cat_idx"]
        self.num_dim = len(self.num_col_idx)
        self.cat_cardinalities = raw_data["cat_cardinalities"]
        self.cat_columns = raw_data["cat_columns"]
        self.idx_to_category = raw_data["idx_to_category"]
        self.train_x_raw = self._compose_native_values(
            raw_data["train_num_raw"][train_idx],
            self.train_cat_idx,
        )
        self.valid_x_raw = self._compose_native_values(
            raw_data["train_num_raw"][valid_idx],
            self.valid_cat_idx,
        )
        self.test_x_raw = self._compose_native_values(
            raw_data["test_num_raw"],
            self.test_cat_idx,
        )
        self.cat_dim = int(self.train_x_raw.shape[1] - self.num_dim)
        self.train_mask_ext = self._build_native_model_mask(self.train_mask_original)
        self.valid_mask_ext = self._build_native_model_mask(self.valid_mask_original)
        self.test_mask_ext = self._build_native_model_mask(self.test_mask_original)
        self.numeric_scaler, self.numeric_standardization = self._build_dsan_native_numeric_metadata(
            normalized,
        )
        self.fairness_checks = self._build_fairness_checks()
        self.categorical_preprocessing = {
            "model_family": "baseline_dsan_native",
            "categorical_representation": "native_label_index",
            "fit_transform_train": "map_idx label encoding from dataset metadata; missing cells set to NaN only in model input",
            "transform_valid_test": "same map_idx label encoding fitted by benchmark metadata",
            "inverse_transform_predictions": "sigmoid threshold for binary categorical tasks, argmax for multi-class tasks, then map_idx id",
            "uses_eunseo_binary_codebook": False,
            "uses_eunseo_codebook_loss": False,
            "uses_eunseo_codebook_argmax_decoding": False,
            "loss": "DSAN native BCE for binary categorical columns and cross-entropy for multi-class categorical columns",
            "evaluation_basis": "original categorical variable label index",
            "categorical_f1": "column-prefixed category-index weighted F1 over missing categorical cells",
            "categorical_metric_level": "column_prefixed_category_index",
            "categorical_f1_average": "weighted_over_column_prefixed_category_indices",
            "categorical_accuracy": "flattened accuracy over column-prefixed missing categorical cells",
        }
        self._verify_mask_files(full_train_mask_original, train_idx, valid_idx)

    def split_sizes(self) -> dict:
        return {
            "train": int(self.train_x_raw.shape[0]),
            "valid": int(self.valid_x_raw.shape[0]),
            "test": int(self.test_x_raw.shape[0]),
        }

    def _build_native_model_mask(self, original_mask: np.ndarray) -> np.ndarray:
        columns = []
        if len(self.num_col_idx) > 0:
            columns.append(original_mask[:, self.num_col_idx])
        if len(self.cat_col_idx) > 0:
            columns.append(original_mask[:, self.cat_col_idx])
        if not columns:
            return np.zeros((original_mask.shape[0], 0), dtype=bool)
        return np.concatenate(columns, axis=1).astype(bool)

    def _select_npz_numeric(self, x_norm: np.ndarray, split_name: str) -> np.ndarray:
        if self.num_dim == 0:
            return np.empty((x_norm.shape[0], 0), dtype=np.float32)
        if x_norm.shape[1] < self.num_dim:
            raise ValueError(
                f"{self.dataset_name} {split_name} normalized array has {x_norm.shape[1]} columns, "
                f"but {self.num_dim} numeric columns are expected."
            )
        return x_norm[:, : self.num_dim].astype(np.float32, copy=True)

    def _compose_native_values(self, num_values: np.ndarray, cat_idx: np.ndarray | None) -> np.ndarray:
        if cat_idx is None:
            return num_values.astype(np.float32, copy=False)
        return np.concatenate([num_values.astype(np.float32), cat_idx.astype(np.float32)], axis=1)

    def _build_dsan_native_numeric_metadata(
        self,
        normalized,
    ) -> tuple[IdentityScaler | EmptyScaler, dict]:
        if self.num_dim == 0:
            return EmptyScaler(), {
                "numeric_source": "raw_csv_numeric_with_dsan_native_standard_scaler_model_input",
                "scaler_source": "model_input_dsan_standard_scaler_and_metric_prepared_npz_mean_std",
                "metric_space": "eunseo_normalized",
                "categorical_protocol": "dsan_native_label_index_and_category_index_metric",
                "npz_path": str(self.normalized_path),
                "numeric_columns_from": "no numeric columns",
                "scaler_applied_in_runner": "none",
                "scale_changed_by_runner": False,
            }

        npz_mean = normalized["mean"][: self.num_dim].astype(np.float32).tolist() if "mean" in normalized.files else None
        npz_std = normalized["std"][: self.num_dim].astype(np.float32).tolist() if "std" in normalized.files else None
        metadata = {
            "source": "raw_csv_numeric_with_dsan_native_standard_scaler_model_input",
            "numeric_source": "raw_csv_numeric",
            "model_numeric_preprocessing": "dsan_native_standard_scaler",
            "model_numeric_standardizer_fit": "train_split_numeric_values_after_applying_training_missing_mask",
            "metric_scaler_source": "prepared_npz_mean_std",
            "scaler_source": "model_input_dsan_standard_scaler_and_metric_prepared_npz_mean_std",
            "metric_space": "eunseo_normalized",
            "categorical_protocol": "dsan_native_label_index_and_category_index_metric",
            "npz_path": str(self.normalized_path),
            "numeric_columns_from": "raw train.csv/test.csv numeric columns selected by Info/{dataset}.json",
            "scaler_applied_in_runner": "sklearn.preprocessing.StandardScaler",
            "scale_changed_by_runner": True,
            "raw_unit_inverse_transform_for_metrics": True,
            "normalization_formula_for_metrics": "(x_raw - prepared_npz_mean) / prepared_npz_std / 2.0",
            "npz_mean": npz_mean,
            "npz_std": npz_std,
            "train_numeric_values_hash": stable_array_digest(self.train_x_raw[:, : self.num_dim].astype(np.float32)),
            "valid_numeric_values_hash": stable_array_digest(self.valid_x_raw[:, : self.num_dim].astype(np.float32)),
            "test_numeric_values_hash": stable_array_digest(self.test_x_raw[:, : self.num_dim].astype(np.float32)),
            "prepared_npz_x_norm_used_as_model_input": False,
            "prepared_npz_mean_std_used_for_metric": True,
            "numeric_input_first_num_dim_matches_npz": False,
            "raw_csv_numeric_scaler_fit": True,
        }
        return IdentityScaler(self.num_dim), metadata

    def _build_fairness_checks(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "mask_type": self.mask_type,
            "ratio": self.ratio,
            "split_idx": int(self.split_idx),
            "same_split_id_logged": True,
            "train_indices_hash": stable_array_digest(self.train_indices.astype(np.int64)),
            "valid_indices_hash": stable_array_digest(self.valid_indices.astype(np.int64)),
            "train_missing_mask_hash": stable_array_digest(self.train_mask_ext.astype(bool)),
            "valid_missing_mask_hash": stable_array_digest(self.valid_mask_ext.astype(bool)),
            "test_missing_mask_hash": stable_array_digest(self.test_mask_ext.astype(bool)),
            "train_original_mask_hash": stable_array_digest(self.train_mask_original.astype(bool)),
            "valid_original_mask_hash": stable_array_digest(self.valid_mask_original.astype(bool)),
            "test_original_mask_hash": stable_array_digest(self.test_mask_original.astype(bool)),
            "evaluation_mask_hash": stable_array_digest(self.test_mask_ext.astype(bool)),
            "numeric_standardization": self.numeric_standardization,
            "categorical_evaluation_basis": "original_label_index",
        }

    def _load_raw_eval_data(self) -> dict:
        train_df = pd.read_csv(self.dataset_dir / "train.csv")
        test_df = pd.read_csv(self.dataset_dir / "test.csv")

        train_mask_original = np.load(
            self.dataset_dir / "masks" / f"rate{self.ratio}" / self.mask_type / f"train_mask_{self.split_idx}.npy"
        ).astype(bool)
        test_mask_original = np.load(
            self.dataset_dir / "masks" / f"rate{self.ratio}" / self.mask_type / f"test_mask_{self.split_idx}.npy"
        ).astype(bool)

        cols = train_df.columns
        train_num_raw = train_df[cols[self.num_col_idx]].values.astype(np.float32)
        test_num_raw = test_df[cols[self.num_col_idx]].values.astype(np.float32)

        if len(self.cat_col_idx) == 0:
            train_cat_idx = None
            test_cat_idx = None
            cat_cardinalities = np.array([], dtype=np.int64)
            cat_columns_list: list[str] = []
            idx_to_category: list[list[str]] = []
        else:
            cat_columns = cols[self.cat_col_idx]
            train_cat = train_df[cat_columns].astype(str)
            test_cat = test_df[cat_columns].astype(str)

            train_cat_idx = []
            test_cat_idx = []
            cat_cardinalities = []
            cat_columns_list = [str(column) for column in cat_columns]
            idx_to_category = []

            for column in cat_columns:
                with open(self.dataset_dir / f"{column}_map_idx.json", "r", encoding="utf-8") as f:
                    category_to_idx = json.load(f)

                train_cat_idx_i = train_cat[column].map(category_to_idx).to_numpy()
                test_cat_idx_i = test_cat[column].map(category_to_idx).to_numpy()
                if pd.isna(train_cat_idx_i).any() or pd.isna(test_cat_idx_i).any():
                    raise ValueError(f"Unmapped category found in {self.dataset_name}.{column}")

                train_cat_idx.append(train_cat_idx_i.astype(np.int64))
                test_cat_idx.append(test_cat_idx_i.astype(np.int64))
                num_classes = len(category_to_idx)
                cat_cardinalities.append(num_classes)
                inverse = [None] * num_classes
                for category, category_idx in category_to_idx.items():
                    inverse[int(category_idx)] = str(category)
                idx_to_category.append(inverse)

            train_cat_idx = np.stack(train_cat_idx, axis=1)
            test_cat_idx = np.stack(test_cat_idx, axis=1)
            cat_cardinalities = np.array(cat_cardinalities, dtype=np.int64)

        return {
            "train_num_raw": train_num_raw,
            "test_num_raw": test_num_raw,
            "train_mask_original": train_mask_original,
            "test_mask_original": test_mask_original,
            "train_cat_idx": None if train_cat_idx is None else train_cat_idx.astype(np.int64),
            "test_cat_idx": None if test_cat_idx is None else test_cat_idx.astype(np.int64),
            "cat_cardinalities": cat_cardinalities.astype(np.int64),
            "cat_columns": cat_columns_list,
            "idx_to_category": idx_to_category,
        }

    def _verify_mask_files(
        self,
        full_train_mask_original: np.ndarray,
        train_idx: np.ndarray,
        valid_idx: np.ndarray,
    ) -> None:
        if not self.strict_split:
            return
        mask_dir = self.dataset_dir / "masks" / f"rate{self.ratio}" / self.mask_type
        checks = [
            (mask_dir / f"strict_train_mask_{self.split_idx}.npy", full_train_mask_original[train_idx]),
            (mask_dir / f"valid_mask_{self.split_idx}.npy", full_train_mask_original[valid_idx]),
            (mask_dir / f"full_train_mask_{self.split_idx}.npy", full_train_mask_original),
        ]
        for path, expected in checks:
            if path.exists() and not np.array_equal(np.load(path).astype(bool), expected.astype(bool)):
                raise ValueError(f"Mask file mismatch for {self.dataset_name}: {path}")


def evaluate_arrays(
    pred_raw: np.ndarray,
    true_raw: np.ndarray,
    mask_original: np.ndarray,
    num_col_idx: list[int],
    cat_col_idx: list[int],
    num_dim: int,
    cat_cardinalities: np.ndarray,
    truth_cat_idx: np.ndarray | None,
    mean_num: np.ndarray,
    std_num: np.ndarray,
) -> dict:
    if num_dim > 0:
        num_mask = mask_original[:, num_col_idx]
        pred_num_norm = (pred_raw[:, :num_dim] - mean_num[None, :]) / (std_num[None, :] * 2.0)
        true_num_norm = (true_raw[:, :num_dim] - mean_num[None, :]) / (std_num[None, :] * 2.0)
        num_diff = pred_num_norm[num_mask] - true_num_norm[num_mask]
        mae = safe_mean(np.abs(num_diff))
        mse = safe_mean(num_diff**2)
        rmse = float(np.sqrt(mse)) if not np.isnan(mse) else float("nan")
        raw_diff = pred_raw[:, :num_dim][num_mask] - true_raw[:, :num_dim][num_mask]
        mae_raw = safe_mean(np.abs(raw_diff))
        mse_raw = safe_mean(raw_diff**2)
        rmse_raw = float(np.sqrt(mse_raw)) if not np.isnan(mse_raw) else float("nan")
    else:
        mae = float("nan")
        mse = float("nan")
        rmse = float("nan")
        mae_raw = float("nan")
        mse_raw = float("nan")
        rmse_raw = float("nan")

    cat_f1 = float("nan")
    cat_f1_macro = float("nan")
    cat_f1_weighted = float("nan")
    cat_accuracy = float("nan")
    accuracy = float("nan")
    if truth_cat_idx is not None and len(cat_col_idx) > 0 and len(cat_cardinalities) > 0:
        cat_mask = mask_original[:, cat_col_idx]
        pred_cat_idx = np.rint(pred_raw[:, num_dim:]).astype(np.int64)
        for column_idx, cardinality in enumerate(cat_cardinalities.tolist()):
            pred_cat_idx[:, column_idx] = np.clip(pred_cat_idx[:, column_idx], 0, int(cardinality) - 1)

        y_true_tokens = []
        y_pred_tokens = []
        for column_idx in range(len(cat_col_idx)):
            column_mask = cat_mask[:, column_idx]
            if column_mask.sum() <= 0:
                continue
            y_true = truth_cat_idx[:, column_idx][column_mask]
            y_pred = pred_cat_idx[:, column_idx][column_mask]
            original_col = int(cat_col_idx[column_idx])
            y_true_tokens.extend([f"col{original_col}=idx{int(value)}" for value in y_true.tolist()])
            y_pred_tokens.extend([f"col{original_col}=idx{int(value)}" for value in y_pred.tolist()])
        if y_true_tokens:
            cat_f1_weighted = float(f1_score(y_true_tokens, y_pred_tokens, average="weighted", zero_division=0))
            cat_f1_macro = float(f1_score(y_true_tokens, y_pred_tokens, average="macro", zero_division=0))
            cat_f1 = cat_f1_weighted
            cat_accuracy = float(np.mean(np.asarray(y_true_tokens, dtype=object) == np.asarray(y_pred_tokens, dtype=object)))
            accuracy = cat_accuracy

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mae_raw": mae_raw,
        "mse_raw": mse_raw,
        "rmse_raw": rmse_raw,
        "cat_f1": cat_f1,
        "f1": cat_f1,
        "cat_f1_macro": cat_f1_macro,
        "cat_f1_weighted": cat_f1_weighted,
        "bit_f1": cat_f1,
        "cat_accuracy": cat_accuracy,
        "accuracy": accuracy,
        "MAE": mae,
        "RMSE": rmse,
        "F1": cat_f1,
        "Accuracy": accuracy,
        "metric_basis": "eunseo_normalized_numeric_and_original_categorical_label_index",
        "numeric_source": "raw_csv_numeric_with_dsan_native_standard_scaler_model_input",
        "scaler_source": "model_input_dsan_standard_scaler_and_metric_prepared_npz_mean_std",
        "metric_space": "eunseo_normalized",
        "numeric_metric_unit": "prepared_normalized_unit",
        "numeric_metric_formula": "((pred_raw - prepared_npz_mean) / prepared_npz_std / 2.0) - ((true_raw - prepared_npz_mean) / prepared_npz_std / 2.0)",
        "numeric_metric_scale": "eunseo_x_norm_formula_(x-mean)/std/2.0",
        "categorical_protocol": "dsan_native_label_index_and_category_index_metric",
        "categorical_metric_level": "column_prefixed_category_index",
        "categorical_f1_basis": "category_index",
        "categorical_f1_average": "weighted_over_column_prefixed_category_indices",
        "cat_f1_average": "weighted_over_column_prefixed_category_indices",
        "categorical_token_format": "col{original_column_index}=idx{category_index}",
        "categorical_metric_aggregation": "flatten_column_prefixed_missing_categorical_cells",
    }


def evaluate_predictions(bundle: BenchmarkDataBundle, pred_raw: np.ndarray, split: str) -> dict:
    if split == "train":
        true_raw = bundle.train_x_raw
        mask_original = bundle.train_mask_original
        truth_cat_idx = bundle.train_cat_idx
    elif split == "valid":
        true_raw = bundle.valid_x_raw
        mask_original = bundle.valid_mask_original
        truth_cat_idx = bundle.valid_cat_idx
    elif split == "test":
        true_raw = bundle.test_x_raw
        mask_original = bundle.test_mask_original
        truth_cat_idx = bundle.test_cat_idx
    else:
        raise ValueError(f"Invalid split: {split}")

    return evaluate_arrays(
        pred_raw=pred_raw,
        true_raw=true_raw,
        mask_original=mask_original,
        num_col_idx=bundle.num_col_idx,
        cat_col_idx=bundle.cat_col_idx,
        num_dim=bundle.num_dim,
        cat_cardinalities=bundle.cat_cardinalities,
        truth_cat_idx=truth_cat_idx,
        mean_num=bundle.metric_mean[: bundle.num_dim],
        std_num=bundle.metric_std[: bundle.num_dim],
    )


class BaselineNativeCategoricalAdapter:
    """Adapter for baseline DSAN native categorical preprocessing and decoding."""

    def __init__(self, bundle: BenchmarkDataBundle) -> None:
        self.bundle = bundle
        self.metadata = bundle.categorical_preprocessing

    def fit_transform_train(self) -> np.ndarray:
        train_x = self.bundle.train_x_raw.copy()
        train_x[self.bundle.train_mask_ext] = np.nan
        return train_x.astype(np.float32)

    def transform_valid_test(self) -> tuple[np.ndarray, np.ndarray]:
        valid_x = self.bundle.valid_x_raw.copy()
        test_x = self.bundle.test_x_raw.copy()
        valid_x[self.bundle.valid_mask_ext] = np.nan
        test_x[self.bundle.test_mask_ext] = np.nan
        return valid_x.astype(np.float32), test_x.astype(np.float32)

    def inverse_transform_predictions(self, pred_raw: np.ndarray) -> np.ndarray:
        restored = pred_raw.astype(np.float32, copy=True)
        if self.bundle.cat_dim <= 0:
            return restored
        pred_cat = np.rint(restored[:, self.bundle.num_dim :]).astype(np.int64)
        for column_idx, cardinality in enumerate(self.bundle.cat_cardinalities.tolist()):
            pred_cat[:, column_idx] = np.clip(pred_cat[:, column_idx], 0, int(cardinality) - 1)
        restored[:, self.bundle.num_dim :] = pred_cat.astype(np.float32)
        return restored


def score_from_metrics(metrics: dict) -> float:
    return normalized_score(metrics)


class DSANFairRunner:
    def __init__(self, args, bundle: BenchmarkDataBundle, run_dir: Path, device: torch.device) -> None:
        self.args = args
        self.bundle = bundle
        self.run_dir = run_dir
        self.device = device
        self.n_col = int(bundle.train_x_raw.shape[1])
        self.num_vars = list(range(bundle.num_dim))
        self.cat_vars = list(range(bundle.num_dim, self.n_col))
        self.tasks = [int(v) for v in bundle.cat_cardinalities.tolist()]
        self.scaler = bundle.numeric_scaler
        self.preprocessing_adapter = BaselineNativeCategoricalAdapter(bundle)
        self.mapper = None
        self.model = None
        self.optimizer = None

    def build_missing_inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        train_x = self.preprocessing_adapter.fit_transform_train()
        valid_x, test_x = self.preprocessing_adapter.transform_valid_test()
        return train_x, valid_x, test_x

    def initialize_model(self, train_x: np.ndarray) -> None:
        if len(self.num_vars) > 0:
            self.scaler = StandardScaler()
            self.scaler.fit(train_x[:, self.num_vars])
            self.bundle.numeric_standardization["model_scaler_mean"] = self.scaler.mean_.astype(np.float32).tolist()
            self.bundle.numeric_standardization["model_scaler_scale"] = self.scaler.scale_.astype(np.float32).tolist()
            self.bundle.numeric_standardization["model_scaler_var"] = self.scaler.var_.astype(np.float32).tolist()
        else:
            self.scaler = EmptyScaler()
        self.mapper = Mapper(X=train_x, num_vars=self.num_vars, cat_vars=self.cat_vars)
        self.model = Model(
            num_vars=self.num_vars,
            cat_vars=self.cat_vars,
            tasks=self.tasks,
            rep_dim=self.args.rep_dim,
            vocab_size=len(self.mapper.f2i),
            num_heads=self.args.num_heads,
            n_hidden=self.args.n_hidden,
            dropout=self.args.dropout,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

    def make_loader(self, x: np.ndarray, batch_size: int, shuffle: bool, noise_percent: int) -> DataLoader:
        dataset = MVIDataset(
            X=x,
            cat_vars=self.cat_vars,
            num_vars=self.num_vars,
            scaler=self.scaler,
            mapper=self.mapper,
            noise_percent=noise_percent,
        )
        generator = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(int(self.args.seed))
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            generator=generator,
        )

    def compute_loss(self, preds, targets, masks) -> torch.Tensor:
        loss_terms = []
        eps = 1e-10
        if len(self.num_vars) > 0:
            num_mask = masks[0].float()
            observed = num_mask.sum()
            if observed.item() > 0:
                loss_mse = F.mse_loss(preds[0], targets[0], reduction="none")
                loss_terms.append((loss_mse * num_mask).sum() / (observed + eps))

        for output_idx, task in enumerate(self.tasks, start=1):
            mask = masks[output_idx].float()
            observed = mask.sum()
            if observed.item() <= 0:
                continue
            if task == 2:
                loss = F.binary_cross_entropy(
                    input=preds[output_idx].reshape(mask.shape),
                    target=targets[output_idx].float(),
                    reduction="none",
                )
            else:
                loss = F.cross_entropy(
                    input=preds[output_idx],
                    target=targets[output_idx],
                    reduction="none",
                ).reshape(mask.shape)
            loss_terms.append((loss * mask).sum() / (observed + eps))

        if not loss_terms:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        return torch.stack(loss_terms).mean()

    def train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_rows = 0
        for batch in loader:
            nums, tokens, targets, masks = batch
            nums = nums.to(self.device)
            tokens = tokens.to(self.device)
            targets = [target.to(self.device) for target in targets]
            masks = [mask.to(self.device) for mask in masks]
            batch_rows = int(tokens.shape[0])

            preds = self.model(nums, tokens)
            loss = self.compute_loss(preds, targets, masks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item()) * batch_rows
            total_rows += batch_rows
        return total_loss / max(total_rows, 1)

    def transform(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        loader = self.make_loader(x=x, batch_size=batch_size, shuffle=False, noise_percent=0)
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                nums, tokens, _, _ = batch
                nums = nums.to(self.device)
                tokens = tokens.to(self.device)
                outputs = self.model(nums, tokens)
                predictions.append(self.outputs_to_raw(outputs))
        return self.preprocessing_adapter.inverse_transform_predictions(np.vstack(predictions).astype(np.float32))

    def outputs_to_raw(self, outputs) -> np.ndarray:
        batch_size = int(outputs[0].shape[0])
        result = np.zeros((batch_size, self.n_col), dtype=np.float32)
        if len(self.num_vars) > 0:
            rec_num = self.scaler.inverse_transform(outputs[0].detach().cpu().numpy())
            result[:, self.num_vars] = rec_num.astype(np.float32)
        for offset, output in enumerate(outputs[1:]):
            column_idx = self.cat_vars[offset]
            if output.shape[1] == 1:
                values = (output.detach().cpu().numpy() > 0.5).astype(np.float32).reshape(-1)
            else:
                values = np.argmax(output.detach().cpu().numpy(), axis=1).astype(np.float32)
            result[:, column_idx] = values
        return result

    def checkpoint_payload(self, epoch: int, config: dict, best_score: float, history: list[dict]) -> dict:
        scaler_payload = None
        if hasattr(self.scaler, "mean_"):
            scaler_payload = {
                "mean_": self.scaler.mean_.tolist(),
                "scale_": self.scaler.scale_.tolist(),
                "var_": self.scaler.var_.tolist(),
                "n_features_in_": int(self.scaler.n_features_in_),
            }
        return {
            "epoch": epoch,
            "config": config,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_score": best_score,
            "num_vars": self.num_vars,
            "cat_vars": self.cat_vars,
            "tasks": self.tasks,
            "scaler": scaler_payload,
            "mapper_f2i": self.mapper.f2i,
            "history": history,
        }

    def run(self) -> dict:
        args = self.args
        self.run_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = self.run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = checkpoints_dir / "best.pt"
        last_ckpt_path = checkpoints_dir / "last.pt"

        train_x, valid_x, test_x = self.build_missing_inputs()
        self.initialize_model(train_x)
        train_loader = self.make_loader(
            x=train_x,
            batch_size=args.batch_size,
            shuffle=True,
            noise_percent=args.noise_percent,
        )

        config_payload = {
            "dataset": self.bundle.dataset_name,
            "data_root": str(args.data_root),
            "output_root": str(args.output_root),
            "mask_type": self.bundle.mask_type,
            "ratio": self.bundle.ratio,
            "split_idx": self.bundle.split_idx,
            "run_tag": args.run_tag,
            "seed": args.seed,
            "device": str(self.device),
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "max_epochs": args.max_epochs,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "selection_split": "valid",
            "valid_max_fraction": args.valid_max_fraction,
            "noise_percent": args.noise_percent,
            "rep_dim": args.rep_dim,
            "num_heads": args.num_heads,
            "n_hidden": args.n_hidden,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            **selection_metadata(),
            "score_formula": selection_metadata()["selection_formula"],
            "numeric_source": "raw_csv_numeric_with_dsan_native_standard_scaler_model_input",
            "scaler_source": "model_input_dsan_standard_scaler_and_metric_prepared_npz_mean_std",
            "metric_space": "eunseo_normalized",
            "categorical_protocol": "dsan_native_label_index_and_category_index_metric",
            "num_dim": self.bundle.num_dim,
            "cat_dim": self.bundle.cat_dim,
            "cat_cardinalities": self.bundle.cat_cardinalities.tolist(),
            "numeric_standardization": self.bundle.numeric_standardization,
            "categorical_preprocessing": self.bundle.categorical_preprocessing,
            "fairness_checks": self.bundle.fairness_checks,
        }
        save_json(self.run_dir / "config.json", config_payload)

        best_score = None
        best_epoch = None
        best_state = None
        best_train_metrics = None
        best_valid_metrics = None
        latest_metrics = None
        epochs_since_improvement = 0
        stopped_early = False
        stop_reason = None
        history = []
        start_epoch = 1

        resume_ckpt_path = find_checkpoint(checkpoints_dir, "last")
        best_load_path = find_checkpoint(checkpoints_dir, "best")
        if not args.no_resume and not args.overwrite and resume_ckpt_path is not None:
            resume_state = torch.load(resume_ckpt_path, map_location=self.device, weights_only=False)
            resume_config = resume_state.get("config", {})
            resume_cat = resume_config.get("categorical_preprocessing", {})
            resume_num = resume_config.get("numeric_standardization", {})
            resume_compatible = (
                resume_cat.get("categorical_representation") == "native_label_index"
                and resume_num.get("source") == "raw_csv_numeric_with_dsan_native_standard_scaler_model_input"
                and resume_config.get("cat_cardinalities") == self.bundle.cat_cardinalities.tolist()
                and resume_config.get("num_dim") == self.bundle.num_dim
                and resume_config.get("cat_dim") == self.bundle.cat_dim
            )
            if not resume_compatible:
                print(
                    f"[resume-skip] dataset={self.bundle.dataset_name} split={self.bundle.split_idx} "
                    "checkpoint uses an incompatible preprocessing protocol; starting fresh. "
                    f"checkpoint={resume_ckpt_path}",
                    flush=True,
                )
            else:
                self.model.load_state_dict(resume_state["model_state_dict"])
                self.optimizer.load_state_dict(resume_state["optimizer_state_dict"])
                best_score = resume_state.get("best_score")
                history = resume_state.get("history", [])
                last_epoch = int(resume_state.get("epoch", len(history)))
                start_epoch = last_epoch + 1
                if history:
                    latest_metrics = {
                        "epoch": history[-1].get("epoch"),
                        "score": history[-1].get("score"),
                        "selection_split": "valid",
                        "train_metrics": history[-1].get("train_metrics"),
                        "valid_metrics": history[-1].get("valid_metrics"),
                        "test_metrics": None,
                    }
                    epochs_since_improvement = int(history[-1].get("epochs_since_improvement", 0))
                if best_load_path is not None:
                    best_checkpoint = torch.load(best_load_path, map_location=self.device, weights_only=False)
                    best_epoch = int(best_checkpoint.get("epoch", last_epoch))
                    best_state = copy.deepcopy(best_checkpoint["model_state_dict"])
                print(
                    f"[resume] dataset={self.bundle.dataset_name} split={self.bundle.split_idx} "
                    f"from_epoch={last_epoch} next_epoch={start_epoch} best_epoch={best_epoch} "
                    f"no_improve={epochs_since_improvement} checkpoint={resume_ckpt_path}",
                    flush=True,
                )

        for epoch in range(start_epoch, args.max_epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            print(f"[predict] dataset={self.bundle.dataset_name} split={self.bundle.split_idx} epoch={epoch} train")
            train_pred = self.transform(train_x, args.eval_batch_size)
            print(f"[predict] dataset={self.bundle.dataset_name} split={self.bundle.split_idx} epoch={epoch} valid")
            valid_pred = self.transform(valid_x, args.eval_batch_size)
            train_metrics = evaluate_predictions(self.bundle, train_pred, split="train")
            valid_metrics = evaluate_predictions(self.bundle, valid_pred, split="valid")
            current_score = score_from_metrics(valid_metrics)

            improved = best_score is None or current_score < (best_score - args.early_stop_min_delta)
            if improved:
                best_score = current_score
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                best_train_metrics = train_metrics
                best_valid_metrics = valid_metrics
                epochs_since_improvement = 0
                safe_torch_save(
                    self.checkpoint_payload(epoch, config_payload, best_score, history),
                    best_ckpt_path,
                    epoch=epoch,
                )
                np.save(self.run_dir / "best_train_pred_raw.npy", train_pred)
                np.save(self.run_dir / "best_valid_pred_raw.npy", valid_pred)
            else:
                epochs_since_improvement += 1

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "score": current_score,
                "selection_split": "valid",
                "improved": bool(improved),
                "train_metrics": train_metrics,
                "valid_metrics": valid_metrics,
                "epochs_since_improvement": epochs_since_improvement,
            }
            history.append(epoch_record)
            latest_metrics = {
                "epoch": epoch,
                "score": current_score,
                "selection_split": "valid",
                "train_metrics": train_metrics,
                "valid_metrics": valid_metrics,
                "test_metrics": None,
            }
            safe_torch_save(
                self.checkpoint_payload(epoch, config_payload, best_score, history),
                last_ckpt_path,
                epoch=epoch,
            )
            save_json(self.run_dir / "history.json", history)

            print(
                f"[eval] dataset={self.bundle.dataset_name} split={self.bundle.split_idx} "
                f"epoch={epoch} train_loss={train_loss:.6f} score={current_score:.6f} "
                f"valid_rmse={valid_metrics.get('rmse')} valid_f1={valid_metrics.get('cat_f1')} "
                f"valid_acc={valid_metrics.get('cat_accuracy')} improved={improved}",
                flush=True,
            )

            if args.early_stop_patience > 0 and epochs_since_improvement >= args.early_stop_patience:
                stopped_early = True
                stop_reason = (
                    "early_stop_patience_exhausted"
                    f"(patience={args.early_stop_patience}, "
                    f"min_delta={args.early_stop_min_delta}, "
                    f"epochs_since_improvement={epochs_since_improvement})"
                )
                print(
                    f"[early-stop] dataset={self.bundle.dataset_name} split={self.bundle.split_idx} "
                    f"epoch={epoch} best_epoch={best_epoch} selection_split=valid",
                    flush=True,
                )
                break

        if best_state is None:
            best_state = copy.deepcopy(self.model.state_dict())
            best_epoch = latest_metrics["epoch"]
            best_score = latest_metrics["score"]
            best_train_metrics = latest_metrics["train_metrics"]
            best_valid_metrics = latest_metrics["valid_metrics"]

        self.model.load_state_dict(best_state)
        print(f"[predict] dataset={self.bundle.dataset_name} split={self.bundle.split_idx} best train", flush=True)
        best_train_pred = self.transform(train_x, args.eval_batch_size)
        print(f"[predict] dataset={self.bundle.dataset_name} split={self.bundle.split_idx} best valid", flush=True)
        best_valid_pred = self.transform(valid_x, args.eval_batch_size)
        print(f"[predict] dataset={self.bundle.dataset_name} split={self.bundle.split_idx} best test", flush=True)
        best_test_pred = self.transform(test_x, args.eval_batch_size)
        best_train_metrics = evaluate_predictions(self.bundle, best_train_pred, split="train")
        best_valid_metrics = evaluate_predictions(self.bundle, best_valid_pred, split="valid")
        best_test_metrics = evaluate_predictions(self.bundle, best_test_pred, split="test")
        best_score = score_from_metrics(best_valid_metrics)

        np.save(self.run_dir / "train_pred_raw.npy", best_train_pred)
        np.save(self.run_dir / "valid_pred_raw.npy", best_valid_pred)
        np.save(self.run_dir / "test_pred_raw.npy", best_test_pred)
        safe_torch_save(
            self.checkpoint_payload(best_epoch, config_payload, best_score, history),
            best_ckpt_path,
            epoch=best_epoch,
        )

        best_report = {
            "epoch": best_epoch,
            "score": best_score,
            "selection_split": "valid",
            "train_metrics": best_train_metrics,
            "valid_metrics": best_valid_metrics,
            "test_metrics": best_test_metrics,
        }
        report = {
            "dataset": self.bundle.dataset_name,
            "mask_type": self.bundle.mask_type,
            "split_idx": self.bundle.split_idx,
            "numeric_source": "raw_csv_numeric_with_dsan_native_standard_scaler_model_input",
            "scaler_source": "model_input_dsan_standard_scaler_and_metric_prepared_npz_mean_std",
            "metric_space": "eunseo_normalized",
            "categorical_protocol": "dsan_native_label_index_and_category_index_metric",
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_report": best_report,
            "val_metrics": best_valid_metrics,
            "val_protocol": {
                "type": "valid_score_same_as_hybrid",
                "score_formula": config_payload["score_formula"],
                "early_stop_patience": args.early_stop_patience,
                "early_stop_min_delta": args.early_stop_min_delta,
                "valid_max_fraction": args.valid_max_fraction,
            },
            "preprocessing_protocol": {
                "numeric": self.bundle.numeric_standardization,
                "categorical": self.bundle.categorical_preprocessing,
            },
            "fairness_checks": self.bundle.fairness_checks,
            "selection_split": "valid",
            **selection_metadata(),
            "split_sizes": self.bundle.split_sizes(),
            "train_metrics": best_train_metrics,
            "valid_metrics": best_valid_metrics,
            "test_metrics": best_test_metrics,
            "latest_epoch": history[-1]["epoch"] if history else None,
            "latest_metrics": best_report,
            "epochs_since_improvement": epochs_since_improvement,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason if stop_reason is not None else "max_epochs_reached",
            "completed": True,
            "num_dim": self.bundle.num_dim,
            "cat_dim": self.bundle.cat_dim,
            "cat_columns": self.bundle.cat_columns,
            "cat_cardinalities": self.bundle.cat_cardinalities.tolist(),
        }
        save_json(self.run_dir / "report.json", report)
        return report


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, np.generic):
        return sanitize_for_json(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(payload), f, indent=2, allow_nan=False)


def find_checkpoint(checkpoints_dir: Path, stem: str) -> Path | None:
    candidates = []
    canonical = checkpoints_dir / f"{stem}.pt"
    if canonical.exists():
        candidates.append(canonical)
    candidates.extend(checkpoints_dir.glob(f"{stem}_epoch_*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def safe_torch_save(payload: dict, path: Path, epoch: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    torch.save(payload, tmp_path)

    last_error = None
    for _ in range(5):
        try:
            tmp_path.replace(path)
            return path
        except OSError as exc:
            last_error = exc
            time.sleep(0.5)

    epoch_part = "unknown" if epoch is None else f"{int(epoch):04d}"
    fallback = path.with_name(f"{path.stem}_epoch_{epoch_part}_{os.getpid()}_{time.time_ns()}{path.suffix}")
    tmp_path.replace(fallback)
    print(
        f"[checkpoint-warning] could not replace {path}; saved fallback={fallback} error={last_error}",
        flush=True,
    )
    return fallback


def aggregate_metrics(reports: list[dict]) -> dict:
    fields = [
        "mae",
        "mse",
        "rmse",
        "f1",
        "cat_f1",
        "cat_f1_macro",
        "cat_f1_weighted",
        "bit_f1",
        "cat_accuracy",
        "accuracy",
    ]
    aggregate = {}
    for field in fields:
        values = []
        for report in reports:
            value = report["test_metrics"].get(field)
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                values.append(float(value))
        if not values:
            aggregate[field] = {"mean": None, "std": None}
        else:
            arr = np.array(values, dtype=np.float64)
            aggregate[field] = {"mean": float(arr.mean()), "std": float(arr.std())}
    return aggregate


def write_dataset_csv(path: Path, reports: list[dict], run_tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "mask_type",
        "split_idx",
        "run_tag",
        "best_epoch",
        "best_score",
        "train_mae",
        "train_rmse",
        "train_cat_f1",
        "train_cat_f1_weighted",
        "train_bit_f1",
        "train_cat_accuracy",
        "valid_mae",
        "valid_rmse",
        "valid_cat_f1",
        "valid_cat_f1_weighted",
        "valid_bit_f1",
        "valid_cat_accuracy",
        "test_mae",
        "test_rmse",
        "test_cat_f1",
        "test_cat_f1_weighted",
        "test_bit_f1",
        "test_cat_accuracy",
        "test_accuracy",
        "categorical_preprocessing",
        "categorical_decoding",
        "stopped_early",
        "completed",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    "dataset": report["dataset"],
                    "mask_type": report["mask_type"],
                    "split_idx": report["split_idx"],
                    "run_tag": run_tag,
                    "best_epoch": report["best_epoch"],
                    "best_score": report["best_score"],
                    "train_mae": report["train_metrics"].get("mae"),
                    "train_rmse": report["train_metrics"].get("rmse"),
                    "train_cat_f1": report["train_metrics"].get("cat_f1"),
                    "train_cat_f1_weighted": report["train_metrics"].get("cat_f1_weighted"),
                    "train_bit_f1": report["train_metrics"].get("bit_f1"),
                    "train_cat_accuracy": report["train_metrics"].get("cat_accuracy"),
                    "valid_mae": report["valid_metrics"].get("mae"),
                    "valid_rmse": report["valid_metrics"].get("rmse"),
                    "valid_cat_f1": report["valid_metrics"].get("cat_f1"),
                    "valid_cat_f1_weighted": report["valid_metrics"].get("cat_f1_weighted"),
                    "valid_bit_f1": report["valid_metrics"].get("bit_f1"),
                    "valid_cat_accuracy": report["valid_metrics"].get("cat_accuracy"),
                    "test_mae": report["test_metrics"].get("mae"),
                    "test_rmse": report["test_metrics"].get("rmse"),
                    "test_cat_f1": report["test_metrics"].get("cat_f1"),
                    "test_cat_f1_weighted": report["test_metrics"].get("cat_f1_weighted"),
                    "test_bit_f1": report["test_metrics"].get("bit_f1"),
                    "test_cat_accuracy": report["test_metrics"].get("cat_accuracy"),
                    "test_accuracy": report["test_metrics"].get("accuracy"),
                    "categorical_preprocessing": report.get("preprocessing_protocol", {})
                    .get("categorical", {})
                    .get("categorical_representation"),
                    "categorical_decoding": report.get("preprocessing_protocol", {})
                    .get("categorical", {})
                    .get("inverse_transform_predictions"),
                    "stopped_early": report["stopped_early"],
                    "completed": report["completed"],
                }
            )


def write_global_csv(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "run_tag",
        "test_mae_mean",
        "test_mae_std",
        "test_rmse_mean",
        "test_rmse_std",
        "test_cat_f1_mean",
        "test_cat_f1_std",
        "test_cat_f1_weighted_mean",
        "test_cat_f1_weighted_std",
        "test_bit_f1_mean",
        "test_bit_f1_std",
        "test_cat_accuracy_mean",
        "test_cat_accuracy_std",
        "test_accuracy_mean",
        "test_accuracy_std",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset, payload in summary.items():
            agg = payload["aggregate_test_metrics"]
            writer.writerow(
                {
                    "dataset": dataset,
                    "run_tag": payload["selected_tag"],
                    "test_mae_mean": agg["mae"]["mean"],
                    "test_mae_std": agg["mae"]["std"],
                    "test_rmse_mean": agg["rmse"]["mean"],
                    "test_rmse_std": agg["rmse"]["std"],
                    "test_cat_f1_mean": agg["cat_f1"]["mean"],
                    "test_cat_f1_std": agg["cat_f1"]["std"],
                    "test_cat_f1_weighted_mean": agg["cat_f1_weighted"]["mean"],
                    "test_cat_f1_weighted_std": agg["cat_f1_weighted"]["std"],
                    "test_bit_f1_mean": agg["bit_f1"]["mean"],
                    "test_bit_f1_std": agg["bit_f1"]["std"],
                    "test_cat_accuracy_mean": agg["cat_accuracy"]["mean"],
                    "test_cat_accuracy_std": agg["cat_accuracy"]["std"],
                    "test_accuracy_mean": agg["accuracy"]["mean"],
                    "test_accuracy_std": agg["accuracy"]["std"],
                }
            )


def get_run_dir(args, dataset: str, split_idx: int) -> Path:
    return args.output_root / dataset / f"rate{args.ratio}" / args.mask_type / f"split_{split_idx}" / args.run_tag


def load_completed_report(run_dir: Path) -> dict | None:
    report_path = run_dir / "report.json"
    if not report_path.exists():
        return None
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    if not report.get("completed"):
        return None

    categorical = report.get("preprocessing_protocol", {}).get("categorical", {})
    numeric = report.get("preprocessing_protocol", {}).get("numeric", {})
    metrics = report.get("test_metrics", {})
    compatible = (
        categorical.get("categorical_representation") == "native_label_index"
        and categorical.get("uses_eunseo_binary_codebook") is False
        and categorical.get("uses_eunseo_codebook_loss") is False
        and categorical.get("uses_eunseo_codebook_argmax_decoding") is False
        and numeric.get("source") == "raw_csv_numeric_with_dsan_native_standard_scaler_model_input"
        and metrics.get("numeric_source") == "raw_csv_numeric_with_dsan_native_standard_scaler_model_input"
        and metrics.get("scaler_source") == "model_input_dsan_standard_scaler_and_metric_prepared_npz_mean_std"
        and metrics.get("metric_basis") == "eunseo_normalized_numeric_and_original_categorical_label_index"
        and metrics.get("categorical_f1_basis") == "category_index"
        and metrics.get("categorical_f1_average") == "weighted_over_column_prefixed_category_indices"
        and metrics.get("categorical_token_format") == "col{original_column_index}=idx{category_index}"
    )
    if not compatible:
        print(
            f"[skip-disabled] existing report does not match current baseline preprocessing protocol: {report_path}",
            flush=True,
        )
        return None
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DSAN with native model preprocessing and Eunseo-compatible final metrics."
    )
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mask-type", type=str, required=True)
    parser.add_argument("--ratio", type=str, default="30")
    parser.add_argument("--splits", nargs="*", type=int, default=[0])
    parser.add_argument("--run-tag", type=str, default="dsan_native_protocol_e600_es")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--early-stop-patience", type=int, default=45)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0002)
    parser.add_argument("--valid-max-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--noise-percent", type=int, default=30)
    parser.add_argument("--rep-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--allow-legacy-valid-split",
        action="store_true",
        help="Allow fallback valid split creation when prepared strict split arrays are absent.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Restart incomplete runs instead of loading last.pt.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.low_memory:
        args.batch_size = 64
        args.eval_batch_size = 256
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    summary_root = args.output_root / "_summaries"
    csv_root = summary_root / f"{slugify(args.run_tag)}-csv"
    benchmark_summary = {}

    for dataset in args.datasets:
        print(f"\n=== Dataset: {dataset} ===", flush=True)
        reports = []
        for split_idx in args.splits:
            split_seed = int(args.seed) + int(split_idx)
            set_seed(split_seed)
            run_dir = get_run_dir(args, dataset, split_idx)
            existing = None if args.overwrite else load_completed_report(run_dir)
            if existing is not None:
                print(f"[skip] dataset={dataset} split={split_idx} report={run_dir / 'report.json'}", flush=True)
                reports.append(existing)
                continue

            configure_selection(
                data_root=args.data_root,
                dataset=dataset,
                mask_type=args.mask_type,
                split_idx=split_idx,
                ratio=args.ratio,
                seed=args.seed,
                valid_max_fraction=args.valid_max_fraction,
            )
            print(
                f"[selection] dataset={dataset} split={split_idx} "
                "protocol=baseline_normalized uses_test=False",
                flush=True,
            )

            print(
                f"[run] dataset={dataset} mask={args.mask_type} split={split_idx} "
                f"seed={split_seed} batch={args.batch_size} eval_batch={args.eval_batch_size}",
                flush=True,
            )
            bundle = BenchmarkDataBundle(
                dataset_name=dataset,
                mask_type=args.mask_type,
                split_idx=split_idx,
                ratio=args.ratio,
                data_root=args.data_root,
                seed=split_seed,
                valid_max_fraction=args.valid_max_fraction,
                require_strict_split=not args.allow_legacy_valid_split,
            )
            runner = DSANFairRunner(args=args, bundle=bundle, run_dir=run_dir, device=device)
            report = runner.run()
            reports.append(report)
            print(
                f"[done] dataset={dataset} split={split_idx} best_epoch={report['best_epoch']} "
                f"test_rmse={report['test_metrics'].get('rmse')} "
                f"test_f1={report['test_metrics'].get('cat_f1')} "
                f"test_acc={report['test_metrics'].get('cat_accuracy')}",
                flush=True,
            )

            del runner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        dataset_summary = {
            "dataset": dataset,
            "selected_tag": args.run_tag,
            "aggregate_test_metrics": aggregate_metrics(reports),
            "per_split_reports": reports,
        }
        benchmark_summary[dataset] = dataset_summary

        mask_slug = args.mask_type.lower()
        save_json(summary_root / f"{dataset}_{mask_slug}_{slugify(args.run_tag)}_summary.json", dataset_summary)
        write_dataset_csv(csv_root / f"{dataset}_{mask_slug}_split_metrics.csv", reports, args.run_tag)

    mask_slug = args.mask_type.lower()
    save_json(summary_root / f"{mask_slug}_{slugify(args.run_tag)}_benchmark_summary.json", benchmark_summary)
    write_global_csv(csv_root / f"{mask_slug}_summary.csv", benchmark_summary)
    print(f"\nSaved summary under {summary_root}", flush=True)


if __name__ == "__main__":
    main()
