from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


SHARED_DIR = next(
    parent / "_shared"
    for parent in Path(__file__).resolve().parents
    if (parent / "_shared" / "baseline_selection.py").exists()
)
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from baseline_selection import configure_selection, normalized_score, selection_metadata  # noqa: E402


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


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def array_hash(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values)
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]
    return header, rows


def _numeric_matrix(rows: list[list[str]], col_idx: list[int]) -> np.ndarray:
    if not col_idx:
        return np.empty((len(rows), 0), dtype=np.float32)
    return np.asarray([[float(row[idx]) for idx in col_idx] for row in rows], dtype=np.float32)


def _categorical_matrix(rows: list[list[str]], col_idx: list[int]) -> np.ndarray:
    if not col_idx:
        return np.empty((len(rows), 0), dtype=object)
    return np.asarray([[row[idx] for idx in col_idx] for row in rows], dtype=object)


def _safe_mean(values: np.ndarray) -> float:
    return float("nan") if values.size == 0 else float(values.mean())


def _weighted_f1(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    true_flat = true_labels.reshape(-1)
    pred_flat = pred_labels.reshape(-1)
    if true_flat.size == 0:
        return float("nan")
    classes = sorted(set(true_flat.tolist()) | set(pred_flat.tolist()))
    weighted_sum = 0.0
    total_support = 0.0
    for cls in classes:
        true_is = true_flat == cls
        support = float(true_is.sum())
        if support == 0.0:
            continue
        pred_is = pred_flat == cls
        tp = float(np.logical_and(true_is, pred_is).sum())
        fp = float(np.logical_and(~true_is, pred_is).sum())
        fn = float(np.logical_and(true_is, ~pred_is).sum())
        denom = 2.0 * tp + fp + fn
        f1 = 0.0
        if denom > 0.0:
            f1 = (2.0 * tp) / denom
        weighted_sum += support * f1
        total_support += support
    return float("nan") if total_support == 0.0 else float(weighted_sum / total_support)


def _score_from_metrics(metrics: dict[str, float]) -> float:
    return normalized_score(metrics)


def _slugify_mask(mask_type: str) -> str:
    return mask_type.lower()


def _auto_run_tag(mask_type: str, split_idx: int, max_epochs: int) -> str:
    return f"gain_native_cat_{_slugify_mask(mask_type)}_split{split_idx}_e{max_epochs}_es"


@dataclass
class RawSplit:
    num_raw: np.ndarray
    num_norm: np.ndarray | None
    cat_raw: np.ndarray
    original_missing_mask: np.ndarray

    @property
    def num_samples(self) -> int:
        return int(self.original_missing_mask.shape[0])


@dataclass
class ModelSplit:
    x_model: np.ndarray
    missing_mask_ext: np.ndarray
    num_raw: np.ndarray
    cat_raw: np.ndarray
    original_missing_mask: np.ndarray

    @property
    def num_samples(self) -> int:
        return int(self.x_model.shape[0])


class StrictNumericStandardizer:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.observed_count_: np.ndarray | None = None
        self.method_: str = "uninitialized_prepared_npz_required"
        self.model_input_source_: str = "uninitialized_requires_prepared_npz_x_norm"
        self.normalization_divisor_: float = 1.0

    def fit(self, train_num: np.ndarray, train_num_missing: np.ndarray) -> "StrictNumericStandardizer":
        raise RuntimeError(
            "Raw numeric scaler fitting is disabled for the fair protocol. "
            "Use prepared npz train_x_norm/valid_x_norm/test_x_norm and npz mean/std."
        )

    def fit_from_npz_stats(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        train_num_missing: np.ndarray,
    ) -> "StrictNumericStandardizer":
        dim = mean.shape[0]
        self.mean_ = mean.astype(np.float32)
        safe_std = std.astype(np.float32).copy()
        safe_std[safe_std <= 1e-12] = 1.0
        self.std_ = safe_std
        self.observed_count_ = (
            np.sum(~train_num_missing, axis=0).astype(np.int64)
            if dim > 0
            else np.zeros(0, dtype=np.int64)
        )
        self.method_ = "normalized_npz_mean_std"
        self.model_input_source_ = "normalized_npz_train_valid_test_x_norm_numeric_columns"
        self.normalization_divisor_ = 2.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StrictNumericStandardizer is not fitted.")
        if values.shape[1] == 0:
            return values.astype(np.float32)
        return ((values - self.mean_[None, :]) / self.std_[None, :] / self.normalization_divisor_).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StrictNumericStandardizer is not fitted.")
        if values.shape[1] == 0:
            return values.astype(np.float32)
        return (values * self.normalization_divisor_ * self.std_[None, :] + self.mean_[None, :]).astype(np.float32)

    def metadata(self) -> dict[str, Any]:
        return {
            "method": self.method_,
            "model_input_source": self.model_input_source_,
            "numeric_source": "prepared_npz_x_norm",
            "scaler_source": "prepared_npz_mean_std",
            "metric_space": "eunseo_normalized",
            "categorical_protocol": "model_native",
            "raw_csv_numeric_scaler_fit": False,
            "normalization_formula": f"(x - mean) / std / {self.normalization_divisor_}",
            "inverse_formula": f"x_norm * {self.normalization_divisor_} * std + mean",
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "std": None if self.std_ is None else self.std_.tolist(),
            "observed_count": None if self.observed_count_ is None else self.observed_count_.tolist(),
            "uses_test_statistics": False,
        }


class BaselineScalarCategoricalAdapter:
    """Encode each categorical feature as one Min-Max-scaled ordinal scalar."""

    def __init__(self) -> None:
        self.vocabularies_: list[list[str]] = []
        self.feature_names_: list[str] = []
        self.cardinalities_: list[int] = []

    def fit(
        self,
        train_cat: np.ndarray,
        train_cat_missing: np.ndarray,
        feature_names: list[str],
    ) -> "BaselineScalarCategoricalAdapter":
        self.feature_names_ = list(feature_names)
        self.vocabularies_ = []
        self.cardinalities_ = []
        for col in range(train_cat.shape[1]):
            observed = train_cat[~train_cat_missing[:, col], col].astype(str)
            if observed.size == 0:
                observed = train_cat[:, col].astype(str)
            vocab = sorted(set(observed.tolist()))
            if not vocab:
                vocab = ["__UNK__"]
            self.vocabularies_.append(vocab)
            self.cardinalities_.append(len(vocab))
        return self

    @property
    def output_dim(self) -> int:
        return len(self.vocabularies_)

    def transform(self, values: np.ndarray) -> np.ndarray:
        if values.shape[1] == 0:
            return np.empty((values.shape[0], 0), dtype=np.float32)
        encoded = np.zeros(values.shape, dtype=np.float32)
        for col, vocab in enumerate(self.vocabularies_):
            lookup = {value: idx for idx, value in enumerate(vocab)}
            for row, raw_value in enumerate(values[:, col].astype(str)):
                idx = lookup.get(raw_value)
                if idx is not None:
                    encoded[row, col] = float(idx) / float(max(1, len(vocab) - 1))
        return encoded

    def expand_missing_mask(self, cat_missing: np.ndarray) -> np.ndarray:
        return cat_missing.astype(bool, copy=True)

    def inverse_transform_predictions(self, pred_scalar: np.ndarray) -> np.ndarray:
        if not self.vocabularies_:
            return np.empty((pred_scalar.shape[0], 0), dtype=object)
        columns = []
        for col, vocab in enumerate(self.vocabularies_):
            idx = np.rint(pred_scalar[:, col] * float(max(1, len(vocab) - 1))).astype(np.int64)
            idx = np.clip(idx, 0, len(vocab) - 1)
            columns.append(np.asarray([vocab[int(i)] for i in idx], dtype=object))
        return np.stack(columns, axis=1)

    def column_prefixed_category_index_tokens(self, values: np.ndarray, original_col_idx: list[int]) -> np.ndarray:
        if values.shape[1] == 0:
            return np.empty((values.shape[0], 0), dtype=object)
        tokens = np.empty(values.shape, dtype=object)
        for col, (vocab, original_col) in enumerate(zip(self.vocabularies_, original_col_idx)):
            lookup = {value: idx for idx, value in enumerate(vocab)}
            for row, raw_value in enumerate(values[:, col].astype(str)):
                idx = lookup.get(raw_value, -1)
                tokens[row, col] = f"col{original_col}=idx{idx}"
        return tokens

    def metadata(self) -> dict[str, Any]:
        return {
            "method": "ordinal_scalar_minmax",
            "fit_scope": "strict_train_observed_categories_only",
            "encoding": "category_index / max(1, cardinality - 1)",
            "decoding": "inverse_minmax_then_round_and_clip_to_original_label",
            "f1_metric": "column_prefixed_category_index_weighted_f1",
            "uses_eunseo_codebook": False,
            "uses_cat_bin_num": False,
            "uses_codebook_loss": False,
            "vocabulary_sizes": list(self.cardinalities_),
            "feature_names": list(self.feature_names_),
        }


class BaselinePreprocessor:
    def __init__(
        self,
        num_col_idx: list[int],
        cat_col_idx: list[int],
        cat_feature_names: list[str],
        numeric_mean: np.ndarray | None = None,
        numeric_std: np.ndarray | None = None,
    ) -> None:
        self.num_col_idx = list(num_col_idx)
        self.cat_col_idx = list(cat_col_idx)
        self.numeric = StrictNumericStandardizer()
        self.categorical = BaselineScalarCategoricalAdapter()
        self.cat_feature_names = list(cat_feature_names)
        self.numeric_mean = numeric_mean
        self.numeric_std = numeric_std

    def fit_transform_train(self, split: RawSplit) -> ModelSplit:
        num_missing = split.original_missing_mask[:, self.num_col_idx] if self.num_col_idx else np.empty((split.num_samples, 0), dtype=bool)
        cat_missing = split.original_missing_mask[:, self.cat_col_idx] if self.cat_col_idx else np.empty((split.num_samples, 0), dtype=bool)
        if split.num_norm is not None and self.numeric_mean is not None and self.numeric_std is not None:
            self.numeric.fit_from_npz_stats(self.numeric_mean, self.numeric_std, num_missing)
            num_model = split.num_norm.astype(np.float32)
        else:
            raise RuntimeError("Prepared npz numeric values are required; raw numeric scaler fitting is disabled.")
        cat_model = self.categorical.fit(split.cat_raw, cat_missing, self.cat_feature_names).transform(split.cat_raw)
        return self._build_model_split(split, num_model, cat_model, num_missing, cat_missing)

    def transform_valid_test(self, split: RawSplit) -> ModelSplit:
        num_missing = split.original_missing_mask[:, self.num_col_idx] if self.num_col_idx else np.empty((split.num_samples, 0), dtype=bool)
        cat_missing = split.original_missing_mask[:, self.cat_col_idx] if self.cat_col_idx else np.empty((split.num_samples, 0), dtype=bool)
        if split.num_norm is None:
            raise RuntimeError("Prepared npz numeric values are required for valid/test transforms.")
        num_model = split.num_norm.astype(np.float32)
        cat_model = self.categorical.transform(split.cat_raw)
        return self._build_model_split(split, num_model, cat_model, num_missing, cat_missing)

    def _build_model_split(
        self,
        split: RawSplit,
        num_model: np.ndarray,
        cat_model: np.ndarray,
        num_missing: np.ndarray,
        cat_missing: np.ndarray,
    ) -> ModelSplit:
        x_model = np.concatenate([num_model, cat_model], axis=1).astype(np.float32)
        cat_missing_ext = self.categorical.expand_missing_mask(cat_missing)
        missing_mask_ext = np.concatenate([num_missing, cat_missing_ext], axis=1).astype(bool)
        return ModelSplit(
            x_model=x_model,
            missing_mask_ext=missing_mask_ext,
            num_raw=split.num_raw.astype(np.float32),
            cat_raw=split.cat_raw.copy(),
            original_missing_mask=split.original_missing_mask.astype(bool),
        )

    def inverse_transform_predictions(self, pred_model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        num_dim = len(self.num_col_idx)
        pred_num = self.numeric.inverse_transform(pred_model[:, :num_dim])
        pred_cat = self.categorical.inverse_transform_predictions(pred_model[:, num_dim:])
        return pred_num, pred_cat

    def metadata(self) -> dict[str, Any]:
        return {
            "numeric_preprocessing": self.numeric.metadata(),
            "categorical_preprocessing": self.categorical.metadata(),
            "eunseo_codebook_objects_present": False,
        }


class NativeCategoricalBundle:
    def __init__(
        self,
        dataset_name: str,
        mask_type: str,
        split_idx: int,
        ratio: str,
        data_root: str | Path,
    ) -> None:
        self.dataset_name = dataset_name
        self.mask_type = mask_type
        self.split_idx = int(split_idx)
        self.ratio = str(ratio)
        self.data_root = Path(data_root)
        self.dataset_dir = self.data_root / dataset_name
        self.info_path = self.data_root / "Info" / f"{dataset_name}.json"
        self.normalized_path = (
            self.dataset_dir / "normalized" / f"rate{self.ratio}" / mask_type / f"split_{split_idx}.npz"
        )
        with open(self.info_path, "r", encoding="utf-8") as f:
            self.info = json.load(f)
        self.num_col_idx = [int(v) for v in self.info["num_col_idx"]]
        self.cat_col_idx = [int(v) for v in self.info["cat_col_idx"]]

        normalized = np.load(self.normalized_path)
        if "train_indices" not in normalized.files or "valid_indices" not in normalized.files:
            raise ValueError(f"{self.normalized_path} does not contain strict train/valid indices.")
        self.train_indices = normalized["train_indices"].astype(np.int64)
        self.valid_indices = normalized["valid_indices"].astype(np.int64)
        num_dim = len(self.num_col_idx)
        if num_dim > 0:
            required_numeric_keys = ["train_x_norm", "valid_x_norm", "test_x_norm", "mean", "std"]
            missing_keys = [key for key in required_numeric_keys if key not in normalized.files]
            if missing_keys:
                raise ValueError(f"{self.normalized_path} is missing numeric normalized keys: {missing_keys}")
            self.train_num_norm = normalized["train_x_norm"][:, :num_dim].astype(np.float32)
            self.valid_num_norm = normalized["valid_x_norm"][:, :num_dim].astype(np.float32)
            self.test_num_norm = normalized["test_x_norm"][:, :num_dim].astype(np.float32)
            self.numeric_mean = normalized["mean"][:num_dim].astype(np.float32)
            self.numeric_std = normalized["std"][:num_dim].astype(np.float32)
            if self.train_num_norm.shape[0] != self.train_indices.shape[0]:
                raise ValueError(f"{self.normalized_path}: train_x_norm rows do not match train_indices.")
            if self.valid_num_norm.shape[0] != self.valid_indices.shape[0]:
                raise ValueError(f"{self.normalized_path}: valid_x_norm rows do not match valid_indices.")
        else:
            self.train_num_norm = np.empty((self.train_indices.shape[0], 0), dtype=np.float32)
            self.valid_num_norm = np.empty((self.valid_indices.shape[0], 0), dtype=np.float32)
            test_rows_for_empty = normalized["test_x_norm"].shape[0] if "test_x_norm" in normalized.files else 0
            self.test_num_norm = np.empty((test_rows_for_empty, 0), dtype=np.float32)
            self.numeric_mean = np.empty(0, dtype=np.float32)
            self.numeric_std = np.empty(0, dtype=np.float32)

        train_header, train_rows = _read_csv(self.dataset_dir / "train.csv")
        test_header, test_rows = _read_csv(self.dataset_dir / "test.csv")
        if train_header != test_header:
            raise ValueError(f"{dataset_name}: train/test CSV headers do not match.")
        self.header = train_header
        self.cat_feature_names = [train_header[idx] for idx in self.cat_col_idx]

        train_num_full = _numeric_matrix(train_rows, self.num_col_idx)
        test_num = _numeric_matrix(test_rows, self.num_col_idx)
        train_cat_full = _categorical_matrix(train_rows, self.cat_col_idx)
        test_cat = _categorical_matrix(test_rows, self.cat_col_idx)

        mask_dir = self.dataset_dir / "masks" / f"rate{self.ratio}" / mask_type
        train_mask_full = np.load(mask_dir / f"train_mask_{self.split_idx}.npy").astype(bool)
        test_mask = np.load(mask_dir / f"test_mask_{self.split_idx}.npy").astype(bool)
        self.mask_paths = {
            "train": str(mask_dir / f"train_mask_{self.split_idx}.npy"),
            "test": str(mask_dir / f"test_mask_{self.split_idx}.npy"),
        }

        self.train_raw = RawSplit(
            num_raw=train_num_full[self.train_indices].astype(np.float32),
            num_norm=self.train_num_norm,
            cat_raw=train_cat_full[self.train_indices],
            original_missing_mask=train_mask_full[self.train_indices],
        )
        self.valid_raw = RawSplit(
            num_raw=train_num_full[self.valid_indices].astype(np.float32),
            num_norm=self.valid_num_norm,
            cat_raw=train_cat_full[self.valid_indices],
            original_missing_mask=train_mask_full[self.valid_indices],
        )
        self.test_raw = RawSplit(
            num_raw=test_num.astype(np.float32),
            num_norm=self.test_num_norm,
            cat_raw=test_cat,
            original_missing_mask=test_mask,
        )

    def split_sizes(self) -> dict[str, int]:
        return {
            "train": self.train_raw.num_samples,
            "valid": self.valid_raw.num_samples,
            "test": self.test_raw.num_samples,
        }

    def fairness_log(self, preprocessor: BaselinePreprocessor) -> dict[str, Any]:
        train_eval_mask = self.train_raw.original_missing_mask[:, self.num_col_idx + self.cat_col_idx]
        valid_eval_mask = self.valid_raw.original_missing_mask[:, self.num_col_idx + self.cat_col_idx]
        test_eval_mask = self.test_raw.original_missing_mask[:, self.num_col_idx + self.cat_col_idx]
        return {
            "split_idx": self.split_idx,
            "mask_type": self.mask_type,
            "ratio": self.ratio,
            "normalized_split_path": str(self.normalized_path),
            "mask_paths": self.mask_paths,
            "train_indices_sha256": array_hash(self.train_indices),
            "valid_indices_sha256": array_hash(self.valid_indices),
            "train_eval_mask_sha256": array_hash(train_eval_mask),
            "valid_eval_mask_sha256": array_hash(valid_eval_mask),
            "test_eval_mask_sha256": array_hash(test_eval_mask),
            "same_split_id_verified": True,
            "same_missing_mask_source_verified": True,
            "same_evaluation_positions_verified": True,
            "numeric_statistics_source": "normalized_npz_mean_std",
            "numeric_model_input_source": "normalized_npz_train_valid_test_x_norm_numeric_columns",
            "numeric_source": "prepared_npz_x_norm",
            "scaler_source": "prepared_npz_mean_std",
            "metric_space": "eunseo_normalized",
            "categorical_protocol": "model_native",
            "gain_internal_normalization": "none",
            "numeric_uses_test_statistics": False,
            "categorical_evaluation_level": "original_label",
            "categorical_preprocessing": preprocessor.categorical.metadata(),
        }


def verify_prepared_numeric_inputs(
    bundle: NativeCategoricalBundle,
    train_split: ModelSplit,
    valid_split: ModelSplit,
    test_split: ModelSplit,
) -> dict[str, Any]:
    num_dim = len(bundle.num_col_idx)
    if num_dim == 0:
        return {
            "enabled": True,
            "num_dim": 0,
            "train_max_abs_diff": 0.0,
            "valid_max_abs_diff": 0.0,
            "test_max_abs_diff": 0.0,
            "passed": True,
        }

    checks = {
        "train": (train_split.x_model[:, :num_dim], bundle.train_num_norm),
        "valid": (valid_split.x_model[:, :num_dim], bundle.valid_num_norm),
        "test": (test_split.x_model[:, :num_dim], bundle.test_num_norm),
    }
    result: dict[str, Any] = {"enabled": True, "num_dim": num_dim, "passed": True}
    for split_name, (actual, expected) in checks.items():
        if actual.shape != expected.shape:
            raise ValueError(
                f"{bundle.dataset_name} split_{bundle.split_idx} {split_name}: "
                f"numeric model input shape {actual.shape} does not match prepared npz {expected.shape}."
            )
        max_abs_diff = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
        result[f"{split_name}_max_abs_diff"] = max_abs_diff
        if max_abs_diff != 0.0:
            result["passed"] = False
    if not result["passed"]:
        raise ValueError(
            f"{bundle.dataset_name} split_{bundle.split_idx}: numeric model input differs from prepared npz: {result}"
        )
    return result


def _identity_minmax_params(dim: int) -> dict[str, Any]:
    return {
        "method": "identity_prepared_npz_x_norm",
        "min_val": np.zeros(dim, dtype=np.float32),
        "range_val": np.ones(dim, dtype=np.float32),
    }


def _apply_minmax(values: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    if params.get("method") == "identity_prepared_npz_x_norm":
        return values.astype(np.float32, copy=True)
    return (values - params["min_val"][None, :]) / (params["range_val"][None, :] + 1e-6)


def _invert_minmax(values: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    if params.get("method") == "identity_prepared_npz_x_norm":
        return values.astype(np.float32, copy=True)
    return values * (params["range_val"][None, :] + 1e-6) + params["min_val"][None, :]


class GainModel:
    def __init__(
        self,
        dim: int,
        hint_rate: float,
        alpha: float,
        learning_rate: float,
        seed: int,
        allow_gpu_growth: bool,
        disable_gpu: bool,
    ) -> None:
        self.dim = int(dim)
        self.hint_rate = float(hint_rate)
        self.alpha = float(alpha)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.seed)
            self._build_graph()
            config = tf.ConfigProto()
            if disable_gpu:
                config.device_count["GPU"] = 0
            if allow_gpu_growth:
                config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.graph, config=config)
            self.sess.run(tf.global_variables_initializer())

    def _xavier_init(self, size: list[int]) -> tf.Tensor:
        in_dim = float(size[0])
        stddev = 1.0 / tf.sqrt(in_dim / 2.0)
        return tf.random_normal(shape=size, stddev=stddev)

    def _build_graph(self) -> None:
        dim = self.dim
        h_dim = max(1, dim)
        self.x_ph = tf.placeholder(tf.float32, shape=[None, dim], name="X")
        self.m_ph = tf.placeholder(tf.float32, shape=[None, dim], name="M")
        self.h_ph = tf.placeholder(tf.float32, shape=[None, dim], name="H")

        with tf.variable_scope("discriminator"):
            d_w1 = tf.Variable(self._xavier_init([dim * 2, h_dim]), name="W1")
            d_b1 = tf.Variable(tf.zeros(shape=[h_dim]), name="b1")
            d_w2 = tf.Variable(self._xavier_init([h_dim, h_dim]), name="W2")
            d_b2 = tf.Variable(tf.zeros(shape=[h_dim]), name="b2")
            d_w3 = tf.Variable(self._xavier_init([h_dim, dim]), name="W3")
            d_b3 = tf.Variable(tf.zeros(shape=[dim]), name="b3")
        with tf.variable_scope("generator"):
            g_w1 = tf.Variable(self._xavier_init([dim * 2, h_dim]), name="W1")
            g_b1 = tf.Variable(tf.zeros(shape=[h_dim]), name="b1")
            g_w2 = tf.Variable(self._xavier_init([h_dim, h_dim]), name="W2")
            g_b2 = tf.Variable(tf.zeros(shape=[h_dim]), name="b2")
            g_w3 = tf.Variable(self._xavier_init([h_dim, dim]), name="W3")
            g_b3 = tf.Variable(tf.zeros(shape=[dim]), name="b3")

        theta_d = [d_w1, d_w2, d_w3, d_b1, d_b2, d_b3]
        theta_g = [g_w1, g_w2, g_w3, g_b1, g_b2, g_b3]

        def generator(x: tf.Tensor, m: tf.Tensor) -> tf.Tensor:
            inputs = tf.concat(values=[x, m], axis=1)
            h1 = tf.nn.relu(tf.matmul(inputs, g_w1) + g_b1)
            h2 = tf.nn.relu(tf.matmul(h1, g_w2) + g_b2)
            return tf.nn.sigmoid(tf.matmul(h2, g_w3) + g_b3)

        def discriminator(x: tf.Tensor, h: tf.Tensor) -> tf.Tensor:
            inputs = tf.concat(values=[x, h], axis=1)
            h1 = tf.nn.relu(tf.matmul(inputs, d_w1) + d_b1)
            h2 = tf.nn.relu(tf.matmul(h1, d_w2) + d_b2)
            return tf.nn.sigmoid(tf.matmul(h2, d_w3) + d_b3)

        self.g_sample = generator(self.x_ph, self.m_ph)
        hat_x = self.x_ph * self.m_ph + self.g_sample * (1.0 - self.m_ph)
        d_prob = discriminator(hat_x, self.h_ph)
        self.d_loss = -tf.reduce_mean(
            self.m_ph * tf.log(d_prob + 1e-8) + (1.0 - self.m_ph) * tf.log(1.0 - d_prob + 1e-8)
        )
        g_loss_temp = -tf.reduce_mean((1.0 - self.m_ph) * tf.log(d_prob + 1e-8))
        self.mse_loss = tf.reduce_mean((self.m_ph * self.x_ph - self.m_ph * self.g_sample) ** 2) / tf.reduce_mean(
            self.m_ph
        )
        self.g_loss = g_loss_temp + self.alpha * self.mse_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.d_solver = optimizer.minimize(self.d_loss, var_list=theta_d)
        self.g_solver = optimizer.minimize(self.g_loss, var_list=theta_g)
        self.saver = tf.train.Saver(max_to_keep=3)

    def train_batch(self, x_norm: np.ndarray, observed_m: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
        rows = int(x_norm.shape[0])
        z = rng.uniform(0.0, 0.01, size=(rows, self.dim)).astype(np.float32)
        h_temp = (rng.uniform(0.0, 1.0, size=(rows, self.dim)) < self.hint_rate).astype(np.float32)
        h = observed_m * h_temp
        x = observed_m * x_norm + (1.0 - observed_m) * z
        _, d_loss = self.sess.run([self.d_solver, self.d_loss], feed_dict={self.x_ph: x, self.m_ph: observed_m, self.h_ph: h})
        _, g_loss, mse_loss = self.sess.run(
            [self.g_solver, self.g_loss, self.mse_loss],
            feed_dict={self.x_ph: x, self.m_ph: observed_m, self.h_ph: h},
        )
        return float(d_loss), float(g_loss), float(mse_loss)

    def predict(
        self,
        x_model: np.ndarray,
        missing_mask_ext: np.ndarray,
        minmax_params: dict[str, np.ndarray],
        eval_batch_size: int,
        seed: int,
    ) -> np.ndarray:
        if minmax_params is None:
            x_norm = x_model.astype(np.float32)
        else:
            x_norm = _apply_minmax(x_model, minmax_params).astype(np.float32)
        observed_m = (~missing_mask_ext).astype(np.float32)
        out_norm = np.empty_like(x_norm)
        rng = np.random.default_rng(seed)
        for start in range(0, x_norm.shape[0], int(eval_batch_size)):
            end = min(start + int(eval_batch_size), x_norm.shape[0])
            rows = end - start
            z = rng.uniform(0.0, 0.01, size=(rows, self.dim)).astype(np.float32)
            x = observed_m[start:end] * x_norm[start:end] + (1.0 - observed_m[start:end]) * z
            g = self.sess.run(self.g_sample, feed_dict={self.x_ph: x, self.m_ph: observed_m[start:end]})
            out_norm[start:end] = observed_m[start:end] * x_norm[start:end] + (1.0 - observed_m[start:end]) * g
        if minmax_params is None:
            return out_norm.astype(np.float32)
        return _invert_minmax(out_norm, minmax_params).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.saver.save(self.sess, str(path))

    def restore(self, path: Path) -> None:
        self.saver.restore(self.sess, str(path))

    def close(self) -> None:
        self.sess.close()


def evaluate_predictions(
    preprocessor: BaselinePreprocessor,
    split: ModelSplit,
    pred_model: np.ndarray,
    num_col_idx: list[int],
    cat_col_idx: list[int],
) -> dict[str, float]:
    pred_num_raw, pred_cat_labels = preprocessor.inverse_transform_predictions(pred_model)
    metrics: dict[str, float] = {}
    if num_col_idx:
        num_mask = split.original_missing_mask[:, num_col_idx]
        if num_mask.sum() > 0:
            pred_num_norm = pred_model[:, : len(num_col_idx)]
            true_num_norm = split.x_model[:, : len(num_col_idx)]
            diff = pred_num_norm[num_mask] - true_num_norm[num_mask]
            mse = _safe_mean(diff**2)
            metrics["mae"] = _safe_mean(np.abs(diff))
            metrics["mse"] = mse
            metrics["rmse"] = float(np.sqrt(mse)) if math.isfinite(mse) else float("nan")

            raw_diff = pred_num_raw[num_mask] - split.num_raw[num_mask]
            raw_mse = _safe_mean(raw_diff**2)
            metrics["mae_raw"] = _safe_mean(np.abs(raw_diff))
            metrics["mse_raw"] = raw_mse
            metrics["rmse_raw"] = float(np.sqrt(raw_mse)) if math.isfinite(raw_mse) else float("nan")
        else:
            metrics["mae"] = float("nan")
            metrics["mse"] = float("nan")
            metrics["rmse"] = float("nan")
            metrics["mae_raw"] = float("nan")
            metrics["mse_raw"] = float("nan")
            metrics["rmse_raw"] = float("nan")
    else:
        metrics["mae"] = float("nan")
        metrics["mse"] = float("nan")
        metrics["rmse"] = float("nan")
        metrics["mae_raw"] = float("nan")
        metrics["mse_raw"] = float("nan")
        metrics["rmse_raw"] = float("nan")

    if cat_col_idx:
        cat_mask = split.original_missing_mask[:, cat_col_idx]
        if cat_mask.sum() > 0:
            true_token_matrix = preprocessor.categorical.column_prefixed_category_index_tokens(split.cat_raw, cat_col_idx)
            pred_token_matrix = preprocessor.categorical.column_prefixed_category_index_tokens(pred_cat_labels, cat_col_idx)
            true_labels = np.empty(int(cat_mask.sum()), dtype=object)
            pred_labels = np.empty(int(cat_mask.sum()), dtype=object)
            pos = 0
            for cat_pos, _original_col in enumerate(cat_col_idx):
                rows = np.where(cat_mask[:, cat_pos])[0]
                n = rows.size
                if n == 0:
                    continue
                true_labels[pos : pos + n] = true_token_matrix[rows, cat_pos]
                pred_labels[pos : pos + n] = pred_token_matrix[rows, cat_pos]
                pos += n
            true_labels = true_labels[:pos]
            pred_labels = pred_labels[:pos]
            metrics["f1"] = _weighted_f1(true_labels, pred_labels)
            metrics["cat_f1"] = metrics["f1"]
            metrics["cat_f1_weighted"] = metrics["f1"]
            metrics["cat_f1_macro"] = float(f1_score(true_labels, pred_labels, average="macro", zero_division=0))
            metrics["bit_f1"] = metrics["f1"]
            metrics["accuracy"] = float(np.mean(true_labels == pred_labels)) if true_labels.size > 0 else float("nan")
            metrics["cat_accuracy"] = metrics["accuracy"]
        else:
            metrics["f1"] = float("nan")
            metrics["cat_f1"] = float("nan")
            metrics["cat_f1_weighted"] = float("nan")
            metrics["bit_f1"] = float("nan")
            metrics["cat_f1_macro"] = float("nan")
            metrics["accuracy"] = float("nan")
            metrics["cat_accuracy"] = float("nan")
    else:
        metrics["f1"] = float("nan")
        metrics["cat_f1"] = float("nan")
        metrics["cat_f1_weighted"] = float("nan")
        metrics["bit_f1"] = float("nan")
        metrics["accuracy"] = float("nan")
        metrics["cat_accuracy"] = float("nan")

    metrics["categorical_metric_level"] = "column_prefixed_category_index"
    metrics["categorical_f1_average"] = "weighted_over_column_prefixed_category_indices"
    metrics["cat_f1_average"] = "weighted_over_column_prefixed_category_indices"
    metrics["categorical_token_format"] = "col{original_column_index}=idx{category_index}"
    metrics["MAE"] = metrics["mae"]
    metrics["RMSE"] = metrics["rmse"]
    metrics["F1"] = metrics["cat_f1"]
    metrics["Accuracy"] = metrics["cat_accuracy"]
    metrics["numeric_source"] = "prepared_npz_x_norm"
    metrics["scaler_source"] = "prepared_npz_mean_std"
    metrics["metric_space"] = "eunseo_normalized"
    metrics["numeric_metric_unit"] = "prepared_normalized_unit"
    metrics["numeric_metric_formula"] = "pred_num_norm - true_x_norm"
    metrics["numeric_metric_scale"] = "eunseo_x_norm_formula_(x-mean)/std/2.0"
    metrics["categorical_protocol"] = "model_native"
    return metrics


class GainNativeCategoricalRunner:
    def __init__(self, args: argparse.Namespace, dataset: str, mask_type: str, run_tag: str) -> None:
        self.args = args
        self.dataset = dataset
        self.mask_type = mask_type
        self.run_tag = run_tag
        self.run_dir = (
            Path(args.output_root)
            / dataset
            / f"rate{args.ratio}"
            / mask_type
            / f"split_{args.split_idx}"
            / run_tag
        )
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.best_ckpt_path = self.checkpoint_dir / "best.ckpt"
        self.last_ckpt_path = self.checkpoint_dir / "last.ckpt"
        self.history_path = self.run_dir / "history.json"
        self.report_path = self.run_dir / "report.json"
        self.history: list[dict[str, Any]] = []

    def run(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        configure_selection(
            data_root=self.args.data_root,
            dataset=self.dataset,
            mask_type=self.mask_type,
            split_idx=self.args.split_idx,
            ratio=self.args.ratio,
            seed=self.args.seed,
            valid_max_fraction=self.args.valid_max_fraction,
        )
        print(
            f"[selection] dataset={self.dataset} split={self.args.split_idx} "
            "protocol=baseline_normalized uses_test=False",
            flush=True,
        )
        bundle = NativeCategoricalBundle(
            dataset_name=self.dataset,
            mask_type=self.mask_type,
            split_idx=self.args.split_idx,
            ratio=str(self.args.ratio),
            data_root=self.args.data_root,
        )
        preprocessor = BaselinePreprocessor(
            num_col_idx=bundle.num_col_idx,
            cat_col_idx=bundle.cat_col_idx,
            cat_feature_names=bundle.cat_feature_names,
            numeric_mean=bundle.numeric_mean,
            numeric_std=bundle.numeric_std,
        )
        train_split = preprocessor.fit_transform_train(bundle.train_raw)
        valid_split = preprocessor.transform_valid_test(bundle.valid_raw)
        test_split = preprocessor.transform_valid_test(bundle.test_raw)
        numeric_input_verification = verify_prepared_numeric_inputs(bundle, train_split, valid_split, test_split)
        minmax_params = None
        train_norm = train_split.x_model.astype(np.float32)
        train_observed_m = (~train_split.missing_mask_ext).astype(np.float32)
        fairness_log = bundle.fairness_log(preprocessor)
        fairness_log["numeric_input_verification"] = numeric_input_verification

        config_payload = self._config_payload(bundle, train_split, preprocessor, fairness_log)
        save_json(self.run_dir / "config.json", config_payload)
        save_json(self.run_dir / "fairness_log.json", fairness_log)

        set_global_seed(self.args.seed)
        model = None
        try:
            model = GainModel(
                dim=train_split.x_model.shape[1],
                hint_rate=self.args.hint_rate,
                alpha=self.args.alpha,
                learning_rate=self.args.learning_rate,
                seed=self.args.seed,
                allow_gpu_growth=not self.args.no_gpu_growth,
                disable_gpu=self.args.disable_gpu,
            )
            rng = np.random.default_rng(self.args.seed)
            best_epoch: int | None = None
            best_score: float | None = None
            best_report: dict[str, Any] | None = None
            epochs_since_improvement = 0
            stopped_early = False
            stop_reason: str | None = None
            total_start = time.time()
            print(
                f"[start] dataset={self.dataset} mask={self.mask_type} split={self.args.split_idx} "
                f"train={train_split.num_samples} valid={valid_split.num_samples} test={test_split.num_samples} "
                f"dim={train_split.x_model.shape[1]} cat=one_hot run_dir={self.run_dir}",
                flush=True,
            )

            for epoch in range(1, self.args.max_epochs + 1):
                epoch_start = time.time()
                order = rng.permutation(train_split.num_samples)
                d_losses, g_losses, mse_losses = [], [], []
                for start in range(0, order.size, self.args.batch_size):
                    batch_idx = order[start : start + self.args.batch_size]
                    d_loss, g_loss, mse_loss = model.train_batch(train_norm[batch_idx], train_observed_m[batch_idx], rng)
                    d_losses.append(d_loss)
                    g_losses.append(g_loss)
                    mse_losses.append(mse_loss)

                train_pred = model.predict(train_split.x_model, train_split.missing_mask_ext, minmax_params, self.args.eval_batch_size, self.args.seed + 101)
                valid_pred = model.predict(valid_split.x_model, valid_split.missing_mask_ext, minmax_params, self.args.eval_batch_size, self.args.seed + 202)
                train_metrics = evaluate_predictions(preprocessor, train_split, train_pred, bundle.num_col_idx, bundle.cat_col_idx)
                valid_metrics = evaluate_predictions(preprocessor, valid_split, valid_pred, bundle.num_col_idx, bundle.cat_col_idx)
                current_score = _score_from_metrics(valid_metrics)
                improved = best_score is None or current_score < (best_score - self.args.early_stop_min_delta)
                if improved:
                    best_epoch = epoch
                    best_score = current_score
                    best_report = {
                        "epoch": epoch,
                        "score": current_score,
                        "selection_split": self.args.selection_split,
                        "train_metrics": train_metrics,
                        "valid_metrics": valid_metrics,
                        "test_metrics": None,
                    }
                    epochs_since_improvement = 0
                    model.save(self.best_ckpt_path)
                    np.save(self.run_dir / "best_train_pred_model.npy", train_pred)
                    np.save(self.run_dir / "best_valid_pred_model.npy", valid_pred)
                else:
                    epochs_since_improvement += 1

                epoch_record = {
                    "epoch": epoch,
                    "epoch_seconds": time.time() - epoch_start,
                    "total_seconds": time.time() - total_start,
                    "d_loss": _safe_mean(np.asarray(d_losses, dtype=np.float64)),
                    "g_loss": _safe_mean(np.asarray(g_losses, dtype=np.float64)),
                    "mse_loss": _safe_mean(np.asarray(mse_losses, dtype=np.float64)),
                    "train_rmse": train_metrics.get("rmse"),
                    "train_mae": train_metrics.get("mae"),
                    "train_f1": train_metrics.get("f1"),
                    "train_accuracy": train_metrics.get("accuracy"),
                    "valid_rmse": valid_metrics.get("rmse"),
                    "valid_mae": valid_metrics.get("mae"),
                    "valid_f1": valid_metrics.get("f1"),
                    "valid_accuracy": valid_metrics.get("accuracy"),
                    "score": current_score,
                    "improved": bool(improved),
                    "best_epoch": best_epoch,
                    "epochs_since_improvement": epochs_since_improvement,
                }
                self.history.append(epoch_record)
                model.save(self.last_ckpt_path)
                print(
                    f"[epoch] dataset={self.dataset} epoch={epoch}/{self.args.max_epochs} "
                    f"score={current_score:.6f} valid_rmse={valid_metrics.get('rmse')} "
                    f"valid_f1={valid_metrics.get('f1')} valid_acc={valid_metrics.get('accuracy')} "
                    f"best_epoch={best_epoch}",
                    flush=True,
                )

                if self.args.early_stop_patience > 0 and epochs_since_improvement >= self.args.early_stop_patience:
                    stopped_early = True
                    stop_reason = (
                        "early_stop_patience_exhausted"
                        f"(patience={self.args.early_stop_patience}, "
                        f"min_delta={self.args.early_stop_min_delta}, "
                        f"epochs_since_improvement={epochs_since_improvement}, "
                        f"selection_split={self.args.selection_split})"
                    )
                self._write_report(
                    bundle,
                    preprocessor,
                    fairness_log,
                    best_epoch,
                    best_score,
                    best_report,
                    stopped_early,
                    stop_reason,
                    completed=False,
                )
                if stopped_early:
                    print(f"[early-stop] dataset={self.dataset} epoch={epoch} best_epoch={best_epoch}", flush=True)
                    break

            if best_epoch is None:
                raise RuntimeError("No best checkpoint was selected.")
            model.restore(self.best_ckpt_path)
            best_train_pred = model.predict(train_split.x_model, train_split.missing_mask_ext, minmax_params, self.args.eval_batch_size, self.args.seed + 101)
            best_valid_pred = model.predict(valid_split.x_model, valid_split.missing_mask_ext, minmax_params, self.args.eval_batch_size, self.args.seed + 202)
            best_test_pred = model.predict(test_split.x_model, test_split.missing_mask_ext, minmax_params, self.args.eval_batch_size, self.args.seed + 303)
            train_metrics = evaluate_predictions(preprocessor, train_split, best_train_pred, bundle.num_col_idx, bundle.cat_col_idx)
            valid_metrics = evaluate_predictions(preprocessor, valid_split, best_valid_pred, bundle.num_col_idx, bundle.cat_col_idx)
            test_metrics = evaluate_predictions(preprocessor, test_split, best_test_pred, bundle.num_col_idx, bundle.cat_col_idx)
            best_score = _score_from_metrics(valid_metrics)
            best_report = {
                "epoch": best_epoch,
                "score": best_score,
                "selection_split": self.args.selection_split,
                "train_metrics": train_metrics,
                "valid_metrics": valid_metrics,
                "test_metrics": test_metrics,
            }
            np.save(self.run_dir / "train_pred_model.npy", best_train_pred)
            np.save(self.run_dir / "valid_pred_model.npy", best_valid_pred)
            np.save(self.run_dir / "test_pred_model.npy", best_test_pred)
            report = self._write_report(
                bundle,
                preprocessor,
                fairness_log,
                best_epoch,
                best_score,
                best_report,
                stopped_early,
                stop_reason,
                completed=True,
            )
            print(
                f"[done] dataset={self.dataset} best_epoch={best_epoch} "
                f"test_rmse={test_metrics.get('rmse')} test_f1={test_metrics.get('f1')} "
                f"test_acc={test_metrics.get('accuracy')}",
                flush=True,
            )
            return report
        finally:
            if model is not None:
                model.close()

    def _config_payload(
        self,
        bundle: NativeCategoricalBundle,
        train_split: ModelSplit,
        preprocessor: BaselinePreprocessor,
        fairness_log: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "model": "GAIN",
            "baseline_protocol": "official_gain_with_ordinal_scalar_categorical_adapter",
            "numeric_source": "prepared_npz_x_norm",
            "scaler_source": "prepared_npz_mean_std",
            "metric_space": "eunseo_normalized",
            "categorical_protocol": "model_native",
            "dataset": self.dataset,
            "mask_type": self.mask_type,
            "split_idx": self.args.split_idx,
            "ratio": str(self.args.ratio),
            "data_root": str(self.args.data_root),
            "output_root": str(self.args.output_root),
            "run_tag": self.run_tag,
            "seed": self.args.seed,
            "selection_split": self.args.selection_split,
            **selection_metadata(),
            "max_epochs": self.args.max_epochs,
            "early_stop_patience": self.args.early_stop_patience,
            "early_stop_min_delta": self.args.early_stop_min_delta,
            "batch_size": self.args.batch_size,
            "eval_batch_size": self.args.eval_batch_size,
            "hint_rate": self.args.hint_rate,
            "alpha": self.args.alpha,
            "learning_rate": self.args.learning_rate,
            "split_sizes": bundle.split_sizes(),
            "num_dim": len(bundle.num_col_idx),
            "cat_original_dim": len(bundle.cat_col_idx),
            "model_dim": int(train_split.x_model.shape[1]),
            "gain_internal_normalization": "none_model_receives_prepared_npz_x_norm",
            "preprocessing": preprocessor.metadata(),
            "fairness_log": fairness_log,
        }

    def _write_report(
        self,
        bundle: NativeCategoricalBundle,
        preprocessor: BaselinePreprocessor,
        fairness_log: dict[str, Any],
        best_epoch: int | None,
        best_score: float | None,
        best_report: dict[str, Any] | None,
        stopped_early: bool,
        stop_reason: str | None,
        completed: bool,
    ) -> dict[str, Any]:
        selected = best_report or {}
        report = {
            "dataset": self.dataset,
            "mask_type": self.mask_type,
            "split_idx": self.args.split_idx,
            "ratio": str(self.args.ratio),
            "run_tag": self.run_tag,
            "run_dir": str(self.run_dir),
            "selection_split": self.args.selection_split,
            **selection_metadata(),
            "best_epoch": best_epoch,
            "best_score": best_score,
            "train_metrics": selected.get("train_metrics"),
            "valid_metrics": selected.get("valid_metrics"),
            "test_metrics": selected.get("test_metrics"),
            "best_report": best_report,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "completed": completed,
            "numeric_source": "prepared_npz_x_norm",
            "scaler_source": "prepared_npz_mean_std",
            "metric_space": "eunseo_normalized",
            "categorical_protocol": "model_native",
            "split_sizes": bundle.split_sizes(),
            "history_path": str(self.history_path),
            "history": self.history,
            "preprocessing": preprocessor.metadata(),
            "fairness_log": fairness_log,
            "categorical_handling": preprocessor.categorical.metadata(),
            "numeric_handling": preprocessor.numeric.metadata(),
            "metric_protocol": {
                "numeric": "prepared_npz_normalized_space_rmse_mae_on_missing_positions",
                "numeric_raw_optional": "rmse_raw_mae_raw_are_inverse_to_original_units_with_prepared_npz_mean_std",
                "categorical": "native_decode_to_column_prefixed_category_indices_then_weighted_f1_and_accuracy_on_missing_positions",
            },
            "paper_protocol_sentence": PAPER_PROTOCOL_SENTENCE,
        }
        save_json(self.history_path, self.history)
        save_json(self.report_path, report)
        return report



PAPER_PROTOCOL_SENTENCE = (
    "Numeric variables use prepared npz train_x_norm/valid_x_norm/test_x_norm values "
    "with prepared mean/std metadata. Each categorical variable is represented by "
    "one Min-Max-scaled ordinal scalar and decoded by inverse scaling, rounding, and "
    "clipping. Final metrics are computed in the common evaluation "
    "space: numeric RMSE/MAE on prepared normalized values and categorical weighted "
    "F1/accuracy on column-prefixed category-index labels over masked positions."
)


def is_completed_report(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        report = load_json(path)
        return (
            bool(report.get("completed", False))
            and report.get("numeric_source") == "prepared_npz_x_norm"
            and report.get("scaler_source") == "prepared_npz_mean_std"
            and report.get("metric_space") == "eunseo_normalized"
            and report.get("categorical_protocol") == "model_native"
            and isinstance(report.get("test_metrics"), dict)
            and report["test_metrics"].get("categorical_metric_level") == "column_prefixed_category_index"
            and report["test_metrics"].get("categorical_f1_average") == "weighted_over_column_prefixed_category_indices"
        )
    except Exception:
        return False


def metric_value(report: dict[str, Any], split: str, key: str) -> Any:
    metrics = report.get(f"{split}_metrics") or {}
    return metrics.get(key)


def write_reports_csv(path: Path, reports: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "mask_type",
        "split_idx",
        "run_tag",
        "best_epoch",
        "best_score",
        "selection_split",
        "stopped_early",
        "completed",
        "test_rmse",
        "test_mae",
        "test_f1",
        "test_accuracy",
        "categorical_handling",
        "categorical_decoding",
        "numeric_handling",
        "run_dir",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            cat_meta = report.get("categorical_handling") or {}
            num_meta = report.get("numeric_handling") or {}
            writer.writerow(
                {
                    "dataset": report.get("dataset"),
                    "mask_type": report.get("mask_type"),
                    "split_idx": report.get("split_idx"),
                    "run_tag": report.get("run_tag"),
                    "best_epoch": report.get("best_epoch"),
                    "best_score": report.get("best_score"),
                    "selection_split": report.get("selection_split"),
                    "stopped_early": report.get("stopped_early"),
                    "completed": report.get("completed"),
                    "test_rmse": metric_value(report, "test", "rmse"),
                    "test_mae": metric_value(report, "test", "mae"),
                    "test_f1": metric_value(report, "test", "f1"),
                    "test_accuracy": metric_value(report, "test", "accuracy"),
                    "categorical_handling": cat_meta.get("method"),
                    "categorical_decoding": cat_meta.get("decoding"),
                    "numeric_handling": num_meta.get("method"),
                    "run_dir": report.get("run_dir"),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official GAIN with scalar categorical adaptation.")
    public_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-root", default=str(public_root / "data"))
    parser.add_argument(
        "--output-root",
        default=str(public_root / "results" / "gain"),
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--mask-types",
        nargs="+",
        required=True,
        choices=[
            "MCAR",
            "MAR",
            "MNAR",
            "MCAR_v2_strict",
            "MAR_logistic_v2_strict",
            "MNAR_logistic_T2_v2_strict_hard",
        ],
    )
    parser.add_argument("--ratio", default="30")
    parser.add_argument("--split-idx", type=int, default=0)
    parser.add_argument(
        "--split-idxs",
        nargs="+",
        type=int,
        default=None,
        help="Run multiple explicit split indices, e.g. --split-idxs 0 1 2 3 4 5 6 7 8 9.",
    )
    parser.add_argument("--all-splits", action="store_true", help="Run split_0 through split_9.")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--selection-split", choices=["valid"], default="valid")
    parser.add_argument("--valid-max-fraction", type=float, default=0.15)
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--early-stop-patience", type=int, default=45)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0002)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hint-rate", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable-gpu", action="store_true")
    parser.add_argument("--no-gpu-growth", action="store_true")
    return parser.parse_args()


def resolve_split_indices(args: argparse.Namespace) -> list[int]:
    if args.all_splits and args.split_idxs is not None:
        raise ValueError("Use either --all-splits or --split-idxs, not both.")
    if args.all_splits:
        return list(range(10))
    if args.split_idxs is not None:
        seen: set[int] = set()
        split_indices: list[int] = []
        for split_idx in args.split_idxs:
            if split_idx < 0:
                raise ValueError(f"split index must be non-negative: {split_idx}")
            if split_idx not in seen:
                split_indices.append(split_idx)
                seen.add(split_idx)
        return split_indices
    if args.split_idx < 0:
        raise ValueError(f"split index must be non-negative: {args.split_idx}")
    return [args.split_idx]


def resolve_run_tag(args: argparse.Namespace, mask_type: str, split_idx: int, multi_split: bool) -> str:
    if args.run_tag is None:
        return _auto_run_tag(mask_type, split_idx, args.max_epochs)
    if "{split_idx}" in args.run_tag or "{split}" in args.run_tag:
        return args.run_tag.format(split_idx=split_idx, split=split_idx)
    if multi_split:
        return f"{args.run_tag}_split{split_idx}"
    return args.run_tag


def main() -> None:
    args = parse_args()
    split_indices = resolve_split_indices(args)
    multi_split = len(split_indices) > 1
    reports: list[dict[str, Any]] = []
    for mask_type in args.mask_types:
        for split_idx in split_indices:
            run_args = copy.copy(args)
            run_args.split_idx = split_idx
            run_tag = resolve_run_tag(args, mask_type, split_idx, multi_split)
            for dataset in args.datasets:
                runner = GainNativeCategoricalRunner(
                    args=run_args,
                    dataset=dataset,
                    mask_type=mask_type,
                    run_tag=run_tag,
                )
                if not args.overwrite and is_completed_report(runner.report_path):
                    report = load_json(runner.report_path)
                    print(
                        f"[skip] dataset={dataset} mask={mask_type} split={split_idx} report={runner.report_path}",
                        flush=True,
                    )
                else:
                    report = runner.run()
                reports.append(copy.deepcopy(report))

    summary_root = Path(args.output_root) / "_summaries"
    if args.run_tag:
        summary_tag = args.run_tag.replace("{split_idx}", "splits").replace("{split}", "splits")
    elif args.all_splits:
        summary_tag = "auto_tags_splits0_9"
    elif args.split_idxs is not None:
        summary_tag = "auto_tags_splits" + "_".join(str(split_idx) for split_idx in split_indices)
    else:
        summary_tag = "auto_tags"
    save_json(summary_root / f"gain_native_categorical_reports_{summary_tag}.json", reports)
    write_reports_csv(summary_root / f"gain_native_categorical_reports_{summary_tag}.csv", reports)
    print(PAPER_PROTOCOL_SENTENCE, flush=True)
    print(f"[summary] {summary_root}", flush=True)


if __name__ == "__main__":
    main()
