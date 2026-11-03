from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


SHARED_DIR = next(
    parent / "_shared"
    for parent in Path(__file__).resolve().parents
    if (parent / "_shared" / "baseline_selection.py").exists()
)
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from baseline_selection import configure_selection, normalized_score, selection_metadata  # noqa: E402


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "miwae"
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
DEFAULT_MASK_TYPES = [
    "MCAR",
    "MAR",
    "MNAR",
]

_TF_GRAPH_SEED: int | None = None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2)


def _slugify_tag(text: str) -> str:
    allowed = []
    for ch in text.strip().lower():
        if ch.isalnum() or ch in {"-", "_"}:
            allowed.append(ch)
        else:
            allowed.append("_")
    slug = "".join(allowed).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "run"


def install_tf1_compat() -> None:
    """Patch TF2 into the TF1 surface expected by the original repo."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]

    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()

    def reset_default_graph_with_seed() -> None:
        tf.compat.v1.reset_default_graph()
        if _TF_GRAPH_SEED is not None:
            tf.compat.v1.set_random_seed(_TF_GRAPH_SEED)

    tf.reset_default_graph = reset_default_graph_with_seed  # type: ignore[attr-defined]
    tf.placeholder = tf.compat.v1.placeholder  # type: ignore[attr-defined]
    tf.variable_scope = tf.compat.v1.variable_scope  # type: ignore[attr-defined]
    tf.get_variable = tf.compat.v1.get_variable  # type: ignore[attr-defined]
    tf.ConfigProto = tf.compat.v1.ConfigProto  # type: ignore[attr-defined]
    tf.Session = tf.compat.v1.Session  # type: ignore[attr-defined]
    tf.global_variables_initializer = tf.compat.v1.global_variables_initializer  # type: ignore[attr-defined]
    tf.trainable_variables = tf.compat.v1.trainable_variables  # type: ignore[attr-defined]
    tf.Summary = tf.compat.v1.Summary  # type: ignore[attr-defined]
    tf.summary = tf.compat.v1.summary  # type: ignore[assignment]
    tf.train.AdamOptimizer = tf.compat.v1.train.AdamOptimizer  # type: ignore[attr-defined]
    tf.train.Saver = tf.compat.v1.train.Saver  # type: ignore[attr-defined]
    tf.set_random_seed = tf.compat.v1.set_random_seed  # type: ignore[attr-defined]
    if not hasattr(tf, "log"):
        tf.log = tf.math.log  # type: ignore[attr-defined]
    if not hasattr(tf, "distributions"):
        tf.distributions = tf.compat.v1.distributions  # type: ignore[attr-defined]

    try:
        import tensorflow_probability  # noqa: F401
    except ModuleNotFoundError:
        tfp = types.ModuleType("tensorflow_probability")
        tfp.distributions = types.SimpleNamespace(
            Normal=tf.compat.v1.distributions.Normal,
            Bernoulli=tf.compat.v1.distributions.Bernoulli,
            StudentT=tf.compat.v1.distributions.StudentT,
        )
        tfp.bijectors = types.SimpleNamespace()
        sys.modules["tensorflow_probability"] = tfp


def set_seed(seed: int) -> None:
    global _TF_GRAPH_SEED
    _TF_GRAPH_SEED = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.compat.v1.set_random_seed(seed)
        tf.random.set_seed(seed)
    except Exception:
        pass


def _extend_mask(mask: np.ndarray, bin_num: np.ndarray) -> np.ndarray:
    num_rows, _ = mask.shape
    cum_sum = bin_num.cumsum()
    cum_sum = np.insert(cum_sum, 0, 0)
    result = np.zeros((num_rows, int(bin_num.sum())), dtype=bool)
    for idx in range(len(bin_num)):
        result[:, cum_sum[idx] : cum_sum[idx + 1]] = np.tile(mask[:, idx][:, None], int(bin_num[idx]))
    return result


def _recover_categories_from_bits(bit_values: np.ndarray, cat_bin_num: np.ndarray) -> np.ndarray:
    if len(cat_bin_num) == 0:
        return np.zeros((bit_values.shape[0], 0), dtype=np.int64)

    rounded = (bit_values > 0.5).astype(np.int64)
    cum_sum = cat_bin_num.cumsum()
    cum_sum = np.insert(cum_sum, 0, 0)
    columns = []
    for idx in range(len(cat_bin_num)):
        bits = rounded[:, cum_sum[idx] : cum_sum[idx + 1]]
        powers = 2 ** np.arange(bits.shape[1] - 1, -1, -1)
        columns.append((bits * powers).sum(axis=1))
    return np.stack(columns, axis=1)


def _recover_categories_from_onehot(bit_values: np.ndarray, cat_bin_num: np.ndarray) -> np.ndarray:
    if len(cat_bin_num) == 0:
        return np.zeros((bit_values.shape[0], 0), dtype=np.int64)

    cum_sum = cat_bin_num.cumsum()
    cum_sum = np.insert(cum_sum, 0, 0)
    columns = []
    for idx in range(len(cat_bin_num)):
        group = bit_values[:, cum_sum[idx]:cum_sum[idx + 1]]
        columns.append(np.argmax(group, axis=1).astype(np.int64))
    return np.stack(columns, axis=1)


def _safe_mean(x: np.ndarray) -> float | None:
    return None if x.size == 0 else float(x.mean())


def _metric_value(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _fit_native_numeric_standardizer(train_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if train_values.shape[1] == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty
    mean = train_values.mean(axis=0).astype(np.float32)
    std = train_values.std(axis=0).astype(np.float32)
    std[~np.isfinite(std)] = 1.0
    std[std < 1e-6] = 1.0
    return mean, std


def _build_train_valid_indices(
    train_size: int,
    test_size: int,
    seed: int,
    valid_max_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if train_size <= 1 or valid_max_fraction <= 0.0:
        return np.arange(train_size, dtype=np.int64), np.empty(0, dtype=np.int64)

    requested_valid_size = int(math.floor(train_size * valid_max_fraction))
    if requested_valid_size <= 0:
        requested_valid_size = 1
    valid_size = min(test_size, requested_valid_size, train_size - 1)
    if valid_size <= 0:
        return np.arange(train_size, dtype=np.int64), np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(int(seed) + 17)
    permutation = rng.permutation(train_size)
    valid_idx = np.sort(permutation[:valid_size]).astype(np.int64)
    train_idx = np.sort(permutation[valid_size:]).astype(np.int64)
    return train_idx, valid_idx


class FairDataBundle:
    categorical_encoding = "codebook"

    def __init__(
        self,
        dataset_name: str,
        mask_type: str,
        split_idx: int,
        ratio: str = "30",
        data_root: str | Path = DEFAULT_DATA_ROOT,
        seed: int = 20260430,
        valid_max_fraction: float = 0.15,
    ) -> None:
        self.dataset_name = dataset_name
        self.mask_type = mask_type
        self.split_idx = int(split_idx)
        self.ratio = str(ratio)
        self.data_root = Path(data_root)
        self.seed = int(seed)
        self.valid_max_fraction = float(valid_max_fraction)
        self.dataset_dir = self.data_root / dataset_name
        self.info_path = self.data_root / "Info" / f"{dataset_name}.json"
        self.normalized_path = (
            self.dataset_dir / "normalized" / f"rate{self.ratio}" / mask_type / f"split_{split_idx}.npz"
        )

        with open(self.info_path, "r", encoding="utf-8") as f:
            self.info = json.load(f)

        self.num_col_idx = [int(x) for x in self.info["num_col_idx"]]
        self.cat_col_idx = [int(x) for x in self.info["cat_col_idx"]]
        self.num_dim = len(self.num_col_idx)

        self._load_normalized_split()
        self._load_raw_eval_data()

    def _load_normalized_split(self) -> None:
        if not self.normalized_path.exists():
            raise FileNotFoundError(f"Prepared normalized split npz not found: {self.normalized_path}")

        normalized = np.load(self.normalized_path, allow_pickle=True)
        required = {
            "train_x_norm",
            "valid_x_norm",
            "test_x_norm",
            "train_mask",
            "valid_mask",
            "test_mask",
            "train_indices",
            "valid_indices",
            "mean",
            "std",
            "cat_bin_num",
        }
        missing = sorted(required - set(normalized.files))
        if missing:
            raise KeyError(f"Prepared normalized split npz is missing keys {missing}: {self.normalized_path}")

        self.strict_split = True
        self._prepared_train_x_norm = normalized["train_x_norm"].astype(np.float32)
        self._prepared_valid_x_norm = normalized["valid_x_norm"].astype(np.float32)
        self._prepared_test_x_norm = normalized["test_x_norm"].astype(np.float32)
        self.train_x_norm = self._prepared_train_x_norm
        self.valid_x_norm = self._prepared_valid_x_norm
        self.test_x_norm = self._prepared_test_x_norm
        self.train_mask_ext = normalized["train_mask"].astype(bool)
        self.valid_mask_ext = normalized["valid_mask"].astype(bool)
        self.test_mask_ext = normalized["test_mask"].astype(bool)
        self.train_indices = normalized["train_indices"].astype(np.int64)
        self.valid_indices = normalized["valid_indices"].astype(np.int64)
        self.mean = normalized["mean"].astype(np.float32)
        self.std = normalized["std"].astype(np.float32)
        self.cat_bin_num = normalized["cat_bin_num"].astype(np.int64)
        self.bit_dim = int(self.train_x_norm.shape[1] - self.num_dim)

    def _load_raw_eval_data(self) -> None:
        train_df = pd.read_csv(self.dataset_dir / "train.csv")
        test_df = pd.read_csv(self.dataset_dir / "test.csv")
        data_df = pd.read_csv(self.dataset_dir / "data.csv")

        train_mask_original = np.load(
            self.dataset_dir / "masks" / f"rate{self.ratio}" / self.mask_type / f"train_mask_{self.split_idx}.npy"
        ).astype(bool)
        test_mask_original = np.load(
            self.dataset_dir / "masks" / f"rate{self.ratio}" / self.mask_type / f"test_mask_{self.split_idx}.npy"
        ).astype(bool)

        cols = train_df.columns
        train_num = train_df[cols[self.num_col_idx]].values.astype(np.float32)
        test_num = test_df[cols[self.num_col_idx]].values.astype(np.float32)

        if len(self.cat_col_idx) == 0:
            full_train_x_raw = train_num
            test_x_raw = test_num
            full_train_cat_idx = None
            test_cat_idx = None
        else:
            cat_columns = cols[self.cat_col_idx]
            train_cat = train_df[cat_columns].astype(str)
            test_cat = test_df[cat_columns].astype(str)
            data_cat = data_df[cat_columns].astype(str)

            train_cat_bin = []
            test_cat_bin = []
            train_cat_idx = []
            test_cat_idx = []
            cat_bin_num = []

            for column in data_cat.columns:
                with open(self.dataset_dir / f"{column}_map_bin.json", "r", encoding="utf-8") as f:
                    category_to_binary = json.load(f)
                with open(self.dataset_dir / f"{column}_map_idx.json", "r", encoding="utf-8") as f:
                    category_to_idx = json.load(f)

                train_cat_enc = train_cat[column].map(category_to_binary).to_numpy()
                test_cat_enc = test_cat[column].map(category_to_binary).to_numpy()
                if pd.isna(train_cat_enc).any() or pd.isna(test_cat_enc).any():
                    raise ValueError(f"Unknown category while encoding {self.dataset_name}.{column}")

                train_cat_bin_i = np.array([list(map(int, bits)) for bits in train_cat_enc], dtype=np.float32)
                test_cat_bin_i = np.array([list(map(int, bits)) for bits in test_cat_enc], dtype=np.float32)
                train_cat_idx_i = train_cat[column].map(category_to_idx).to_numpy().astype(np.int64)
                test_cat_idx_i = test_cat[column].map(category_to_idx).to_numpy().astype(np.int64)

                train_cat_bin.append(train_cat_bin_i)
                test_cat_bin.append(test_cat_bin_i)
                train_cat_idx.append(train_cat_idx_i)
                test_cat_idx.append(test_cat_idx_i)
                cat_bin_num.append(train_cat_bin_i.shape[1])

            full_train_cat_bin = np.concatenate(train_cat_bin, axis=1)
            test_cat_bin_arr = np.concatenate(test_cat_bin, axis=1)
            full_train_cat_idx = np.stack(train_cat_idx, axis=1)
            test_cat_idx = np.stack(test_cat_idx, axis=1)
            loaded_cat_bin_num = np.array(cat_bin_num, dtype=np.int64)
            if not np.array_equal(loaded_cat_bin_num, self.cat_bin_num):
                raise ValueError(
                    f"cat_bin_num mismatch for {self.dataset_name}: "
                    f"normalized={self.cat_bin_num.tolist()} raw={loaded_cat_bin_num.tolist()}"
                )

            full_train_x_raw = np.concatenate([train_num, full_train_cat_bin], axis=1)
            test_x_raw = np.concatenate([test_num, test_cat_bin_arr], axis=1)

        self.train_x_raw = full_train_x_raw[self.train_indices].astype(np.float32)
        self.valid_x_raw = full_train_x_raw[self.valid_indices].astype(np.float32)
        self.test_x_raw = test_x_raw.astype(np.float32)
        self.train_mask_original = train_mask_original[self.train_indices]
        self.valid_mask_original = train_mask_original[self.valid_indices]
        self.test_mask_original = test_mask_original
        self.train_cat_idx = None if full_train_cat_idx is None else full_train_cat_idx[self.train_indices]
        self.valid_cat_idx = None if full_train_cat_idx is None else full_train_cat_idx[self.valid_indices]
        self.test_cat_idx = test_cat_idx

    def split_sizes(self) -> dict[str, int]:
        return {
            "train": int(self.train_x_norm.shape[0]),
            "valid": int(self.valid_x_norm.shape[0]),
            "test": int(self.test_x_norm.shape[0]),
        }

    def x_nan(self, split: str) -> np.ndarray:
        x_norm, mask = self.norm_and_mask(split)
        x_nan = x_norm.copy()
        x_nan[mask] = np.nan
        return x_nan

    def norm_and_mask(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        if split == "train":
            return self.train_x_norm, self.train_mask_ext
        if split == "valid":
            return self.valid_x_norm, self.valid_mask_ext
        if split == "test":
            return self.test_x_norm, self.test_mask_ext
        raise ValueError(f"Invalid split: {split}")

    def norm_to_raw(self, x_norm: np.ndarray) -> np.ndarray:
        return (x_norm * (self.std[None, :] * 2.0) + self.mean[None, :]).astype(np.float32)

    def validate_numeric_protocol(self) -> dict:
        if self.num_dim == 0:
            train_match = valid_match = test_match = True
        else:
            expected = {
                "train": self._prepared_train_x_norm[:, : self.num_dim],
                "valid": self._prepared_valid_x_norm[:, : self.num_dim],
                "test": self._prepared_test_x_norm[:, : self.num_dim],
            }
            actual = {
                "train": self.train_x_norm[:, : self.num_dim],
                "valid": self.valid_x_norm[:, : self.num_dim],
                "test": self.test_x_norm[:, : self.num_dim],
            }
            matches = {split: bool(np.array_equal(actual[split], expected[split])) for split in expected}
            if not all(matches.values()):
                max_abs_diff = {
                    split: float(np.max(np.abs(actual[split] - expected[split])))
                    for split in expected
                    if not matches[split]
                }
                raise ValueError(
                    "Numeric model inputs do not exactly match prepared npz x_norm first num_dim columns: "
                    f"matches={matches}, max_abs_diff={max_abs_diff}"
                )
            train_match = matches["train"]
            valid_match = matches["valid"]
            test_match = matches["test"]
        return {
            "train": train_match,
            "valid": valid_match,
            "test": test_match,
        }

    def numeric_protocol_metadata(self) -> dict:
        matches = self.validate_numeric_protocol()
        return {
            "numeric_source": "raw_csv_numeric_with_notmiwae_native_train_zscore_model_input",
            "scaler_source": "model_input_train_zscore_and_metric_prepared_npz_mean_std",
            "metric_space": "eunseo_normalized",
            "categorical_protocol": "wrapper_onehot_input_original_notmiwae_decoder",
            "normalization_formula": "(x_raw - mean) / std / 2.0",
            "inverse_formula": "x_norm * 2.0 * std + mean",
            "numeric_input_first_num_dim_matches_npz": matches,
        }


class OneHotFairDataBundle(FairDataBundle):
    """Fair split loader that builds categorical one-hot groups without map_bin/map_idx files."""

    categorical_encoding = "onehot"

    def _load_normalized_split(self) -> None:
        if not self.normalized_path.exists():
            raise FileNotFoundError(f"Prepared normalized split npz not found: {self.normalized_path}")

        normalized = np.load(self.normalized_path, allow_pickle=True)
        required = {
            "train_x_norm",
            "valid_x_norm",
            "test_x_norm",
            "train_mask",
            "valid_mask",
            "test_mask",
            "train_indices",
            "valid_indices",
            "mean",
            "std",
        }
        missing = sorted(required - set(normalized.files))
        if missing:
            raise KeyError(f"Prepared normalized split npz is missing keys {missing}: {self.normalized_path}")

        self.strict_split = True
        self._prepared_train_x_norm = normalized["train_x_norm"].astype(np.float32)
        self._prepared_valid_x_norm = normalized["valid_x_norm"].astype(np.float32)
        self._prepared_test_x_norm = normalized["test_x_norm"].astype(np.float32)
        self._train_x_norm_source = self._prepared_train_x_norm
        self._valid_x_norm_source = self._prepared_valid_x_norm
        self._test_x_norm_source = self._prepared_test_x_norm
        self._train_mask_source = normalized["train_mask"].astype(bool)
        self._valid_mask_source = normalized["valid_mask"].astype(bool)
        self._test_mask_source = normalized["test_mask"].astype(bool)
        self.train_indices = normalized["train_indices"].astype(np.int64)
        self.valid_indices = normalized["valid_indices"].astype(np.int64)
        self.mean = normalized["mean"].astype(np.float32)
        self.std = normalized["std"].astype(np.float32)

    @staticmethod
    def _one_hot(indices: np.ndarray, dim: int) -> np.ndarray:
        out = np.zeros((indices.shape[0], dim), dtype=np.float32)
        out[np.arange(indices.shape[0]), indices] = 1.0
        return out

    def _load_raw_eval_data(self) -> None:
        train_df = pd.read_csv(self.dataset_dir / "train.csv")
        test_df = pd.read_csv(self.dataset_dir / "test.csv")
        data_df = pd.read_csv(self.dataset_dir / "data.csv")

        train_mask_original = np.load(
            self.dataset_dir / "masks" / f"rate{self.ratio}" / self.mask_type / f"train_mask_{self.split_idx}.npy"
        ).astype(bool)
        test_mask_original = np.load(
            self.dataset_dir / "masks" / f"rate{self.ratio}" / self.mask_type / f"test_mask_{self.split_idx}.npy"
        ).astype(bool)

        cols = train_df.columns
        train_num_raw = train_df[cols[self.num_col_idx]].values.astype(np.float32)
        test_num_raw = test_df[cols[self.num_col_idx]].values.astype(np.float32)
        train_num_raw_split = train_num_raw[self.train_indices]
        valid_num_raw_split = train_num_raw[self.valid_indices]
        self.model_numeric_mean, self.model_numeric_std = _fit_native_numeric_standardizer(train_num_raw_split)
        if self.num_dim > 0:
            train_num_model = ((train_num_raw_split - self.model_numeric_mean[None, :])
                               / self.model_numeric_std[None, :]).astype(np.float32)
            valid_num_model = ((valid_num_raw_split - self.model_numeric_mean[None, :])
                               / self.model_numeric_std[None, :]).astype(np.float32)
            test_num_model = ((test_num_raw - self.model_numeric_mean[None, :])
                              / self.model_numeric_std[None, :]).astype(np.float32)
        else:
            train_num_model = np.zeros((len(self.train_indices), 0), dtype=np.float32)
            valid_num_model = np.zeros((len(self.valid_indices), 0), dtype=np.float32)
            test_num_model = np.zeros((len(test_df), 0), dtype=np.float32)
        train_num_mask = self._train_mask_source[:, :self.num_dim]
        valid_num_mask = self._valid_mask_source[:, :self.num_dim]
        test_num_mask = self._test_mask_source[:, :self.num_dim]

        full_train_cat_idx = None
        valid_cat_idx = None
        test_cat_idx = None
        full_train_cat_oh = np.zeros((len(train_df), 0), dtype=np.float32)
        test_cat_oh = np.zeros((len(test_df), 0), dtype=np.float32)
        self.cat_bin_num = np.array([], dtype=np.int64)
        self.category_maps: dict[str, list[str]] = {}

        if len(self.cat_col_idx) > 0:
            cat_columns = cols[self.cat_col_idx]
            train_cat = train_df[cat_columns].astype(str)
            test_cat = test_df[cat_columns].astype(str)
            data_cat = data_df[cat_columns].astype(str)

            train_cat_oh_parts = []
            test_cat_oh_parts = []
            train_cat_idx_parts = []
            test_cat_idx_parts = []
            cat_dims = []

            for column in data_cat.columns:
                categories = list(data_cat[column].dropna().astype(str).unique())
                category_to_idx = {category: idx for idx, category in enumerate(categories)}
                train_idx = train_cat[column].map(category_to_idx).to_numpy()
                test_idx = test_cat[column].map(category_to_idx).to_numpy()
                if pd.isna(train_idx).any() or pd.isna(test_idx).any():
                    raise ValueError(f"Unknown category while one-hot encoding {self.dataset_name}.{column}")

                train_idx = train_idx.astype(np.int64)
                test_idx = test_idx.astype(np.int64)
                dim = len(categories)
                train_cat_oh_parts.append(self._one_hot(train_idx, dim))
                test_cat_oh_parts.append(self._one_hot(test_idx, dim))
                train_cat_idx_parts.append(train_idx)
                test_cat_idx_parts.append(test_idx)
                cat_dims.append(dim)
                self.category_maps[str(column)] = categories

            full_train_cat_oh = np.concatenate(train_cat_oh_parts, axis=1).astype(np.float32)
            test_cat_oh = np.concatenate(test_cat_oh_parts, axis=1).astype(np.float32)
            full_train_cat_idx = np.stack(train_cat_idx_parts, axis=1).astype(np.int64)
            test_cat_idx = np.stack(test_cat_idx_parts, axis=1).astype(np.int64)
            self.cat_bin_num = np.array(cat_dims, dtype=np.int64)

        self.bit_dim = int(self.cat_bin_num.sum())
        self.train_x_raw = np.concatenate(
            [train_num_raw[self.train_indices], full_train_cat_oh[self.train_indices]], axis=1
        ).astype(np.float32)
        self.valid_x_raw = np.concatenate(
            [train_num_raw[self.valid_indices], full_train_cat_oh[self.valid_indices]], axis=1
        ).astype(np.float32)
        self.test_x_raw = np.concatenate([test_num_raw, test_cat_oh], axis=1).astype(np.float32)

        train_cat_model = full_train_cat_oh[self.train_indices]
        valid_cat_model = full_train_cat_oh[self.valid_indices]
        test_cat_model = test_cat_oh
        self.train_x_norm = np.concatenate([train_num_model, train_cat_model], axis=1).astype(np.float32)
        self.valid_x_norm = np.concatenate([valid_num_model, valid_cat_model], axis=1).astype(np.float32)
        self.test_x_norm = np.concatenate([test_num_model, test_cat_model], axis=1).astype(np.float32)

        self.train_mask_original = train_mask_original[self.train_indices]
        self.valid_mask_original = train_mask_original[self.valid_indices]
        self.test_mask_original = test_mask_original

        train_cat_mask = _extend_mask(self.train_mask_original[:, self.cat_col_idx], self.cat_bin_num)
        valid_cat_mask = _extend_mask(self.valid_mask_original[:, self.cat_col_idx], self.cat_bin_num)
        test_cat_mask = _extend_mask(self.test_mask_original[:, self.cat_col_idx], self.cat_bin_num)
        self.train_mask_ext = np.concatenate([train_num_mask, train_cat_mask], axis=1).astype(bool)
        self.valid_mask_ext = np.concatenate([valid_num_mask, valid_cat_mask], axis=1).astype(bool)
        self.test_mask_ext = np.concatenate([test_num_mask, test_cat_mask], axis=1).astype(bool)

        self.train_cat_idx = None if full_train_cat_idx is None else full_train_cat_idx[self.train_indices]
        self.valid_cat_idx = None if full_train_cat_idx is None else full_train_cat_idx[self.valid_indices]
        self.test_cat_idx = test_cat_idx

    def norm_to_raw(self, x_norm: np.ndarray) -> np.ndarray:
        if self.num_dim > 0:
            num_raw = (x_norm[:, :self.num_dim] * self.model_numeric_std[None, :]
                       + self.model_numeric_mean[None, :])
        else:
            num_raw = np.zeros((x_norm.shape[0], 0), dtype=np.float32)
        cat_raw = x_norm[:, self.num_dim:]
        return np.concatenate([num_raw, cat_raw], axis=1).astype(np.float32)

    def numeric_protocol_metadata(self) -> dict:
        return {
            "model_numeric_source": "raw_csv_numeric",
            "model_numeric_preprocessing": "notMIWAE_native_train_split_zscore",
            "model_numeric_standardizer_fit": "train_split_complete_numeric_values_only",
            "metric_numeric_source": "raw_numeric_inverse_transformed_to_eunseo_normalized_space",
            "metric_scaler_source": "prepared_npz_mean_std",
            "metric_space": "eunseo_normalized",
            "normalization_formula_for_metrics": "(x_raw - prepared_npz_mean) / prepared_npz_std / 2.0",
            "model_inverse_formula": "x_model * model_train_std + model_train_mean",
            "categorical_model_input": "onehot_wrapper_input",
            "categorical_model_internal_head": "none_original_notmiwae_gaussian_decoder",
            "categorical_metric_level": "column_prefixed_category_index",
            "categorical_f1_average": "weighted_over_column_prefixed_category_indices",
        }


def evaluate_predictions(bundle: FairDataBundle, pred_model: np.ndarray, split: str) -> dict:
    if split == "train":
        true_raw = bundle.train_x_raw
        true_norm = bundle.train_x_norm
        mask_original = bundle.train_mask_original
        truth_cat_idx = bundle.train_cat_idx
    elif split == "valid":
        true_raw = bundle.valid_x_raw
        true_norm = bundle.valid_x_norm
        mask_original = bundle.valid_mask_original
        truth_cat_idx = bundle.valid_cat_idx
    elif split == "test":
        true_raw = bundle.test_x_raw
        true_norm = bundle.test_x_norm
        mask_original = bundle.test_mask_original
        truth_cat_idx = bundle.test_cat_idx
    else:
        raise ValueError(f"Invalid split: {split}")

    return evaluate_arrays(
        pred_model=pred_model,
        true_raw=true_raw,
        true_norm=true_norm,
        mask_original=mask_original,
        num_col_idx=bundle.num_col_idx,
        cat_col_idx=bundle.cat_col_idx,
        num_dim=bundle.num_dim,
        cat_bin_num=bundle.cat_bin_num,
        truth_cat_idx=truth_cat_idx,
        metric_mean_num=bundle.mean[: bundle.num_dim],
        metric_std_num=bundle.std[: bundle.num_dim],
        model_mean_num=getattr(bundle, "model_numeric_mean", bundle.mean[: bundle.num_dim]),
        model_std_num=getattr(bundle, "model_numeric_std", bundle.std[: bundle.num_dim] * 2.0),
        categorical_encoding=getattr(bundle, "categorical_encoding", "codebook"),
    )


def evaluate_arrays(
    pred_model: np.ndarray,
    true_raw: np.ndarray,
    true_norm: np.ndarray,
    mask_original: np.ndarray,
    num_col_idx: list[int],
    cat_col_idx: list[int],
    num_dim: int,
    cat_bin_num: np.ndarray,
    truth_cat_idx: np.ndarray | None,
    metric_mean_num: np.ndarray,
    metric_std_num: np.ndarray,
    model_mean_num: np.ndarray,
    model_std_num: np.ndarray,
    categorical_encoding: str = "codebook",
) -> dict:
    mae = None
    mse = None
    rmse = None
    mae_raw = None
    mse_raw = None
    rmse_raw = None
    if num_dim > 0:
        num_mask = mask_original[:, num_col_idx]
        if num_mask.sum() > 0:
            pred_num_model = pred_model[:, :num_dim]
            pred_num_raw = pred_num_model * model_std_num[None, :] + model_mean_num[None, :]
            pred_num_eval = (pred_num_raw - metric_mean_num[None, :]) / (metric_std_num[None, :] * 2.0)
            true_num_eval = (true_raw[:, :num_dim] - metric_mean_num[None, :]) / (metric_std_num[None, :] * 2.0)
            num_diff = pred_num_eval[num_mask] - true_num_eval[num_mask]
            mae = _safe_mean(np.abs(num_diff))
            mse = _safe_mean(num_diff**2)
            rmse = None if mse is None else float(np.sqrt(mse))
            raw_diff = pred_num_raw[num_mask] - true_raw[:, :num_dim][num_mask]
            mae_raw = _safe_mean(np.abs(raw_diff))
            mse_raw = _safe_mean(raw_diff**2)
            rmse_raw = None if mse_raw is None else float(np.sqrt(mse_raw))

    bit_f1 = None
    cat_f1_macro = None
    cat_f1_weighted = None
    cat_accuracy = None
    accuracy = None
    if len(cat_col_idx) > 0 and len(cat_bin_num) > 0:
        cat_mask = mask_original[:, cat_col_idx]
        bit_mask = _extend_mask(cat_mask, cat_bin_num)
        if bit_mask.sum() > 0:
            pred_bits = (pred_model[:, num_dim:] > 0.5).astype(np.int64)
            true_bits = true_raw[:, num_dim:].astype(np.int64)
            bit_f1 = float(
                f1_score(true_bits[bit_mask].reshape(-1), pred_bits[bit_mask].reshape(-1), zero_division=0)
            )

    if truth_cat_idx is not None and len(cat_col_idx) > 0 and len(cat_bin_num) > 0:
        cat_mask = mask_original[:, cat_col_idx]
        if cat_mask.sum() > 0:
            if categorical_encoding == "onehot":
                pred_cat_idx = _recover_categories_from_onehot(pred_model[:, num_dim:], cat_bin_num)
            else:
                raise ValueError("notMIWAE fair baseline requires native one-hot categorical encoding.")
            cat_accuracy = float((pred_cat_idx[cat_mask] == truth_cat_idx[cat_mask]).mean())
            accuracy = cat_accuracy
            y_true = []
            y_pred = []
            for col_pos, original_col in enumerate(cat_col_idx):
                rows = np.where(cat_mask[:, col_pos])[0]
                if rows.size == 0:
                    continue
                y_true.extend([f"col{original_col}=idx{int(v)}" for v in truth_cat_idx[rows, col_pos].tolist()])
                y_pred.extend([f"col{original_col}=idx{int(v)}" for v in pred_cat_idx[rows, col_pos].tolist()])
            if y_true:
                cat_f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
                cat_f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mae_raw": mae_raw,
        "mse_raw": mse_raw,
        "rmse_raw": rmse_raw,
        "f1": cat_f1_weighted,
        "cat_f1": cat_f1_weighted,
        "accuracy": accuracy,
        "bit_f1": cat_f1_weighted,
        "onehot_bit_f1": bit_f1,
        "cat_f1_macro": cat_f1_macro,
        "cat_f1_weighted": cat_f1_weighted,
        "cat_accuracy": cat_accuracy,
        "MAE": mae,
        "RMSE": rmse,
        "F1": cat_f1_weighted,
        "Accuracy": accuracy,
        "numeric_source": "raw_csv_numeric_with_notmiwae_native_train_zscore_model_input",
        "scaler_source": "model_input_train_zscore_and_metric_prepared_npz_mean_std",
        "metric_space": "eunseo_normalized",
        "categorical_protocol": "wrapper_onehot_input_original_notmiwae_decoder",
        "categorical_metric_level": "column_prefixed_category_index",
        "cat_f1_average": "weighted_over_column_prefixed_category_indices",
        "categorical_f1_average": "weighted_over_column_prefixed_category_indices",
        "categorical_token_format": "col{original_column_index}=idx{category_index}",
        "numeric_metric_unit": "eunseo_prepared_normalized_unit",
        "numeric_metric_formula": "((pred_raw - prepared_npz_mean) / prepared_npz_std / 2.0) - ((true_raw - prepared_npz_mean) / prepared_npz_std / 2.0)",
        "numeric_metric_scale": "eunseo_x_norm_formula_(x-mean)/std/2.0",
    }


def score_from_metrics(metrics: dict) -> float:
    return normalized_score(metrics)


def run_training_epoch(model: Any, batch_size: int, rng: np.random.Generator) -> float:
    order = rng.permutation(model.n)
    total_loss = 0.0
    total_count = 0
    for start in range(0, model.n, batch_size):
        idx = order[start : start + batch_size]
        feed = {
            model.x_pl: model.X[idx],
            model.s_pl: model.S[idx],
            model.n_pl: model.n_samples,
        }
        _, loss, _ = model.sess.run([model.train_op, model.loss, model.global_step], feed)
        total_loss += float(loss) * len(idx)
        total_count += len(idx)
    return total_loss / max(total_count, 1)


def evaluate_loss(model: Any, x_nan: np.ndarray, batch_size: int, n_samples: int) -> float:
    observed = (~np.isnan(x_nan)).astype(np.float32)
    x_zero = np.nan_to_num(x_nan, nan=0.0).astype(np.float32)
    total_loss = 0.0
    total_count = 0
    for start in range(0, len(x_zero), batch_size):
        xb = x_zero[start : start + batch_size]
        sb = observed[start : start + batch_size]
        loss = model.sess.run(model.loss, {model.x_pl: xb, model.s_pl: sb, model.n_pl: n_samples})
        total_loss += float(loss) * len(xb)
        total_count += len(xb)
    return total_loss / max(total_count, 1)


def impute_norm(model: Any, x_nan: np.ndarray, n_samples: int, batch_size: int) -> np.ndarray:
    observed = (~np.isnan(x_nan)).astype(np.float32)
    x_zero = np.nan_to_num(x_nan, nan=0.0).astype(np.float32)
    completed = np.empty_like(x_zero, dtype=np.float32)
    has_missing_process = hasattr(model, "log_p_s_given_x")

    for start in range(0, len(x_zero), batch_size):
        xb = x_zero[start : start + batch_size]
        sb = observed[start : start + batch_size]
        fetches = [model.l_out_mu, model.log_p_x_given_z, model.log_p_z, model.log_q_z_given_x]
        if has_missing_process:
            fetches.append(model.log_p_s_given_x)
        outputs = model.sess.run(fetches, {model.x_pl: xb, model.s_pl: sb, model.n_pl: n_samples})
        l_out_mu, log_p_x_given_z, log_p_z, log_q_z_given_x = outputs[:4]
        log_w = log_p_x_given_z + log_p_z - log_q_z_given_x
        if has_missing_process:
            log_w = log_w + outputs[4]
        log_w = log_w - np.max(log_w, axis=1, keepdims=True)
        weights = np.exp(log_w)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        xm = np.sum(l_out_mu * weights[:, :, None], axis=1)
        completed[start : start + len(xb)] = xb + xm * (1.0 - sb)
    return completed


def evaluate_model_predictions(
    model: Any,
    bundle: FairDataBundle,
    split: str,
    n_samples: int,
    eval_batch_size: int,
) -> tuple[dict, np.ndarray]:
    pred_norm = impute_norm(model, bundle.x_nan(split), n_samples=n_samples, batch_size=eval_batch_size)
    metrics = evaluate_predictions(bundle, pred_norm, split)
    pred_raw = bundle.norm_to_raw(pred_norm)
    return metrics, pred_raw


def get_run_dir(args: argparse.Namespace, dataset: str, split_idx: int) -> Path:
    return args.output_root / dataset / f"rate{args.ratio}" / args.mask_type / f"split_{split_idx}" / args.run_tag


def load_completed_report(run_dir: Path) -> dict | None:
    report_path = run_dir / "report.json"
    if not report_path.exists():
        return None
    with open(report_path, "r", encoding="utf-8-sig") as f:
        report = json.load(f)
    if not report.get("completed"):
        return None
    test_metrics = report.get("test_metrics")
    config = report.get("config") if isinstance(report.get("config"), dict) else {}
    if (
        report.get("numeric_source") != "raw_csv_numeric_with_notmiwae_native_train_zscore_model_input"
        or report.get("scaler_source") != "model_input_train_zscore_and_metric_prepared_npz_mean_std"
        or report.get("metric_space") != "eunseo_normalized"
        or report.get("categorical_protocol") != "wrapper_onehot_input_original_notmiwae_decoder"
        or config.get("model_variant") != "native"
        or config.get("categorical_encoding") != "onehot"
        or not isinstance(test_metrics, dict)
        or test_metrics.get("metric_space") != "eunseo_normalized"
        or test_metrics.get("categorical_metric_level") != "column_prefixed_category_index"
        or test_metrics.get("cat_f1_average") != "weighted_over_column_prefixed_category_indices"
        or test_metrics.get("categorical_token_format") != "col{original_column_index}=idx{category_index}"
        or test_metrics.get("cat_f1_weighted") is None
        or "rmse_raw" not in test_metrics
        or "mae_raw" not in test_metrics
    ):
        return None
    return report


def load_json_if_exists(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def checkpoint_exists(prefix: Path) -> bool:
    return Path(str(prefix) + ".index").exists()


def history_latest_epoch(history: list[dict]) -> int:
    epochs = [int(record["epoch"]) for record in history if "epoch" in record]
    return max(epochs) if epochs else 0


def resume_state_from_history(history: list[dict]) -> dict:
    if not history:
        return {
            "latest_epoch": 0,
            "best_epoch": None,
            "best_score": None,
            "epochs_since_improvement": 0,
            "stopped_early": False,
            "stop_reason": None,
        }

    latest = history[-1]
    improved_records = [record for record in history if record.get("improved")]
    if improved_records:
        best_record = improved_records[-1]
    else:
        best_record = min(history, key=lambda record: float(record.get("score", float("inf"))))

    return {
        "latest_epoch": int(latest["epoch"]),
        "best_epoch": int(best_record["epoch"]),
        "best_score": best_record.get("score"),
        "epochs_since_improvement": int(latest.get("epochs_since_improvement", 0)),
        "stopped_early": False,
        "stop_reason": None,
    }


def is_resume_compatible(resume_state: dict, config_payload: dict) -> bool:
    resume_config = resume_state.get("config")
    if not isinstance(resume_config, dict):
        return False

    keys = [
        "model",
        "model_variant",
        "out_dist",
        "cat_likelihood",
        "categorical_encoding",
        "numeric_metric_scale",
        "input_dim",
        "n_latent",
        "n_hidden",
        "n_samples",
    ]
    for key in keys:
        if resume_config.get(key) != config_payload.get(key):
            return False
    if list(resume_config.get("cat_dims", [])) != list(config_payload.get("cat_dims", [])):
        return False
    return True


def build_config_payload(args: argparse.Namespace, dataset: str, split_idx: int, bundle: FairDataBundle) -> dict:
    n_latent = (
        args.n_latent
        if args.n_latent is not None
        else max(1, math.ceil(bundle.train_x_norm.shape[1] / 2))
    )
    model_out_dist = args.out_dist
    return {
        "dataset": dataset,
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "mask_type": args.mask_type,
        "ratio": args.ratio,
        "split_idx": split_idx,
        "run_tag": args.run_tag,
        "seed": args.seed,
        "model": args.model_kind,
        "model_variant": args.model_variant,
        "n_latent": n_latent,
        "n_hidden": args.n_hidden,
        "n_samples": args.n_samples,
        "missing_process": args.missing_process,
        "out_dist": model_out_dist,
        "cat_likelihood": "not_applicable_original_notmiwae_has_no_categorical_head",
        "categorical_encoding": getattr(bundle, "categorical_encoding", "codebook"),
        "cat_dims": bundle.cat_bin_num.tolist(),
        "input_dim": int(bundle.train_x_norm.shape[1]),
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_epochs": args.max_epochs,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "selection_split": "valid",
        "selection_metric": args.selection_metric,
        **selection_metadata(),
        "score_formula": selection_metadata()["selection_formula"],
        "numeric_metric_scale": "eunseo_normalized",
        "numeric_source": "raw_csv_numeric_with_notmiwae_native_train_zscore_model_input",
        "scaler_source": "model_input_train_zscore_and_metric_prepared_npz_mean_std",
        "metric_space": "eunseo_normalized",
        "categorical_protocol": "wrapper_onehot_input_original_notmiwae_decoder",
        "numeric_protocol": bundle.numeric_protocol_metadata(),
        "selection_impute_samples": args.selection_impute_samples,
        "final_impute_samples": args.final_impute_samples,
        "valid_max_fraction": args.valid_max_fraction,
        "auto_resume": not args.no_resume and not args.overwrite,
        "split_sizes": bundle.split_sizes(),
        "num_dim": bundle.num_dim,
        "bit_dim": bundle.bit_dim,
        "strict_split": bundle.strict_split,
    }


def run_single_split(args: argparse.Namespace, dataset: str, split_idx: int) -> dict:
    set_seed(args.seed)
    run_dir = get_run_dir(args, dataset, split_idx)
    if not args.overwrite:
        existing = load_completed_report(run_dir)
        if existing is not None:
            print(f"[skip] dataset={dataset} split={split_idx} report={run_dir / 'report.json'}", flush=True)
            return existing

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

    print(f"[load] dataset={dataset} split={split_idx} mask={args.mask_type}", flush=True)
    if args.model_variant != "native":
        raise ValueError("This wrapper preserves the original notMIWAE architecture; use --model-variant native.")
    bundle_cls = OneHotFairDataBundle
    bundle = bundle_cls(
        dataset_name=dataset,
        mask_type=args.mask_type,
        split_idx=split_idx,
        ratio=args.ratio,
        data_root=args.data_root,
        seed=args.seed,
        valid_max_fraction=args.valid_max_fraction,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config_payload = build_config_payload(args, dataset, split_idx, bundle)
    save_json(run_dir / "config.json", config_payload)

    if args.dry_run:
        report = {
            "dataset": dataset,
            "mask_type": args.mask_type,
            "split_idx": split_idx,
            "best_epoch": None,
            "best_score": None,
            "selection_split": "valid",
            **selection_metadata(),
            "split_sizes": bundle.split_sizes(),
            "train_metrics": None,
            "valid_metrics": None,
            "test_metrics": None,
            "stopped_early": False,
            "stop_reason": "dry_run",
            "completed": False,
            "config": config_payload,
        }
        save_json(run_dir / "report.json", report)
        return report

    if args.model_kind == "miwae":
        from MIWAE import MIWAE as ModelClass
    else:
        from notMIWAE import notMIWAE as ModelClass

    train_x_nan = bundle.x_nan("train")
    valid_x_nan = bundle.x_nan("valid")

    print(
        f"[fit] dataset={dataset} train={train_x_nan.shape} valid={valid_x_nan.shape} "
        f"test={bundle.test_x_norm.shape}",
        flush=True,
    )
    model_kwargs = {
        "n_latent": config_payload["n_latent"],
        "n_samples": args.n_samples,
        "n_hidden": args.n_hidden,
        "out_dist": config_payload["out_dist"],
        "name": str(run_dir),
    }
    if args.model_kind == "notmiwae":
        model_kwargs["missing_process"] = args.missing_process
    model = ModelClass(train_x_nan, valid_x_nan, **model_kwargs)

    rng = np.random.default_rng(args.seed)
    history_path = run_dir / "history.json"
    resume_state_path = run_dir / "resume_state.json"
    history: list[dict] = [] if args.overwrite else load_json_if_exists(history_path, [])
    if not isinstance(history, list):
        history = []
    best_score: float | None = None
    best_epoch: int | None = None
    epochs_since_improvement = 0
    stopped_early = False
    stop_reason = "max_epochs_reached"
    best_ckpt_prefix = run_dir / "checkpoints" / "best"
    last_ckpt_prefix = run_dir / "checkpoints" / "last"
    best_ckpt_prefix.parent.mkdir(parents=True, exist_ok=True)

    resume_state = None
    resume_latest_epoch = 0
    can_resume = (
        not args.overwrite
        and not args.no_resume
        and checkpoint_exists(last_ckpt_prefix)
        and len(history) > 0
    )
    if can_resume:
        resume_state = load_json_if_exists(resume_state_path, None)
        if not isinstance(resume_state, dict):
            resume_state = resume_state_from_history(history)
        if not is_resume_compatible(resume_state, config_payload):
            print(
                f"[resume-incompatible] dataset={dataset} split={split_idx} "
                "existing checkpoint config differs from current native/original settings; starting fresh",
                flush=True,
            )
            can_resume = False
            history = []
            resume_state = None
            resume_latest_epoch = 0
        else:
            resume_latest_epoch = int(resume_state.get("latest_epoch") or history_latest_epoch(history))
        if can_resume and resume_latest_epoch > 0:
            print(
                f"[resume] dataset={dataset} split={split_idx} "
                f"checkpoint={last_ckpt_prefix} latest_epoch={resume_latest_epoch}",
                flush=True,
            )
            model.load(str(last_ckpt_prefix))
            best_score = resume_state.get("best_score")
            best_epoch = resume_state.get("best_epoch")
            epochs_since_improvement = int(resume_state.get("epochs_since_improvement", 0))
            stopped_early = bool(resume_state.get("stopped_early", False))
            stop_reason = resume_state.get("stop_reason") or stop_reason
            rng = np.random.default_rng(args.seed)
            for _ in range(resume_latest_epoch):
                rng.permutation(model.n)
        elif can_resume:
            can_resume = False
    elif len(history) > 0 and not args.overwrite and not args.no_resume:
        print(
            f"[resume-skip] dataset={dataset} split={split_idx} "
            f"history_exists_but_last_checkpoint_missing={last_ckpt_prefix}",
            flush=True,
        )
        history = []

    def build_progress_report(completed: bool, latest_epoch: int | None = None) -> dict:
        return {
            "dataset": dataset,
            "mask_type": args.mask_type,
            "split_idx": split_idx,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "selection_split": "valid",
            "split_sizes": bundle.split_sizes(),
            "train_metrics": None,
            "valid_metrics": None,
            "test_metrics": None,
            "latest_epoch": latest_epoch if latest_epoch is not None else history_latest_epoch(history),
            "latest_metrics": history[-1] if history else None,
            "epochs_since_improvement": epochs_since_improvement,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "completed": completed,
            "num_dim": bundle.num_dim,
            "bit_dim": bundle.bit_dim,
            "config": config_payload,
        }

    def save_resume_artifacts(latest_epoch: int) -> None:
        state = {
            "dataset": dataset,
            "mask_type": args.mask_type,
            "split_idx": split_idx,
            "run_tag": args.run_tag,
            "latest_epoch": latest_epoch,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "epochs_since_improvement": epochs_since_improvement,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "max_epochs": args.max_epochs,
            "selection_split": "valid",
            "selection_metric": args.selection_metric,
            "config": config_payload,
        }
        save_json(resume_state_path, state)
        save_json(run_dir / "report.json", build_progress_report(completed=False, latest_epoch=latest_epoch))

    try:
        start_epoch = resume_latest_epoch + 1 if can_resume else 1
        if stopped_early:
            print(
                f"[resume-finalize] dataset={dataset} split={split_idx} "
                f"already_stopped_early best_epoch={best_epoch}",
                flush=True,
            )

        for epoch in range(start_epoch, args.max_epochs + 1):
            if stopped_early:
                break
            train_loss = run_training_epoch(model, args.batch_size, rng)
            valid_loss = evaluate_loss(model, valid_x_nan, args.eval_batch_size, args.n_samples)

            valid_metrics = None
            if args.selection_metric == "score":
                valid_metrics, _ = evaluate_model_predictions(
                    model,
                    bundle,
                    "valid",
                    n_samples=args.selection_impute_samples,
                    eval_batch_size=args.eval_batch_size,
                )
                current_score = score_from_metrics(valid_metrics)
            else:
                current_score = valid_loss

            improved = best_score is None or current_score < (best_score - args.early_stop_min_delta)
            if improved:
                best_score = current_score
                best_epoch = epoch
                epochs_since_improvement = 0
                model.save(str(best_ckpt_prefix))
            else:
                epochs_since_improvement += 1

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "score": current_score,
                "improved": improved,
                "epochs_since_improvement": epochs_since_improvement,
            }
            if valid_metrics is not None:
                for key, value in valid_metrics.items():
                    record[f"valid_{key}"] = value
            history.append(record)
            model.save(str(last_ckpt_prefix))
            save_json(history_path, history)
            save_resume_artifacts(epoch)

            print(
                f"[epoch] dataset={dataset} epoch={epoch}/{args.max_epochs} "
                f"train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} "
                f"score={current_score:.6f} best_epoch={best_epoch} "
                f"wait={epochs_since_improvement}",
                flush=True,
            )

            if epochs_since_improvement >= args.early_stop_patience:
                stopped_early = True
                stop_reason = (
                    "early_stop_patience_exhausted"
                    f"(patience={args.early_stop_patience}, "
                    f"min_delta={args.early_stop_min_delta}, "
                    f"epochs_since_improvement={epochs_since_improvement})"
                )
                print(
                    f"[early-stop] dataset={dataset} epoch={epoch} best_epoch={best_epoch} "
                    f"selection_split=valid",
                    flush=True,
                )
                save_resume_artifacts(epoch)
                break

        if best_epoch is None:
            best_epoch = len(history)
            best_score = history[-1]["score"] if history else None
            if checkpoint_exists(last_ckpt_prefix):
                model.save(str(best_ckpt_prefix))

        print(f"[restore] dataset={dataset} best_epoch={best_epoch}", flush=True)
        restore_ckpt_prefix = best_ckpt_prefix if checkpoint_exists(best_ckpt_prefix) else last_ckpt_prefix
        if not checkpoint_exists(restore_ckpt_prefix):
            raise FileNotFoundError(f"No checkpoint available for final evaluation: {restore_ckpt_prefix}")
        model.load(str(restore_ckpt_prefix))

        print(f"[predict] dataset={dataset} split=train samples={args.final_impute_samples}", flush=True)
        train_metrics, train_pred_raw = evaluate_model_predictions(
            model, bundle, "train", args.final_impute_samples, args.eval_batch_size
        )
        print(f"[predict] dataset={dataset} split=valid samples={args.final_impute_samples}", flush=True)
        valid_metrics, valid_pred_raw = evaluate_model_predictions(
            model, bundle, "valid", args.final_impute_samples, args.eval_batch_size
        )
        print(f"[predict] dataset={dataset} split=test samples={args.final_impute_samples}", flush=True)
        test_metrics, test_pred_raw = evaluate_model_predictions(
            model, bundle, "test", args.final_impute_samples, args.eval_batch_size
        )

        final_best_score = score_from_metrics(valid_metrics) if args.selection_metric == "score" else best_score
        np.save(run_dir / "train_pred_raw.npy", train_pred_raw)
        np.save(run_dir / "valid_pred_raw.npy", valid_pred_raw)
        np.save(run_dir / "test_pred_raw.npy", test_pred_raw)

        best_report = {
            "epoch": best_epoch,
            "score": final_best_score,
            "selection_split": "valid",
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        }
        report = {
            "dataset": dataset,
            "mask_type": args.mask_type,
            "split_idx": split_idx,
            "best_epoch": best_epoch,
            "best_score": final_best_score,
            "best_report": best_report,
            **selection_metadata(),
            "val_metrics": valid_metrics,
            "val_protocol": {
                "type": "valid_score_same_as_hybrid"
                if args.selection_metric == "score"
                else "valid_loss_notmiwae",
                "score_formula": config_payload["score_formula"],
                "early_stop_patience": args.early_stop_patience,
                "early_stop_min_delta": args.early_stop_min_delta,
                "valid_max_fraction": args.valid_max_fraction,
                "selection_impute_samples": args.selection_impute_samples,
                "final_impute_samples": args.final_impute_samples,
            },
            "selection_split": "valid",
            "split_sizes": bundle.split_sizes(),
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
            "latest_epoch": history[-1]["epoch"] if history else None,
            "latest_metrics": best_report,
            "epochs_since_improvement": epochs_since_improvement,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "completed": True,
            "num_dim": bundle.num_dim,
            "bit_dim": bundle.bit_dim,
            "numeric_source": "raw_csv_numeric_with_notmiwae_native_train_zscore_model_input",
            "scaler_source": "model_input_train_zscore_and_metric_prepared_npz_mean_std",
            "metric_space": "eunseo_normalized",
            "categorical_protocol": "wrapper_onehot_input_original_notmiwae_decoder",
            "numeric_protocol": bundle.numeric_protocol_metadata(),
            "resumed": bool(can_resume),
            "resume_checkpoint": str(last_ckpt_prefix) if can_resume else None,
            "config": config_payload,
        }
        save_json(run_dir / "report.json", report)
        return _json_safe(report)
    finally:
        try:
            model.sess.close()
        except Exception:
            pass
        del model
        gc.collect()


def aggregate_metrics(reports: list[dict]) -> dict:
    fields = [
        "mae",
        "mse",
        "rmse",
        "mae_raw",
        "mse_raw",
        "rmse_raw",
        "f1",
        "accuracy",
        "bit_f1",
        "onehot_bit_f1",
        "cat_f1_macro",
        "cat_f1_weighted",
        "cat_accuracy",
    ]
    result = {}
    for field in fields:
        values = []
        for report in reports:
            metrics = report.get("test_metrics") or {}
            value = _metric_value(metrics.get(field))
            if value is not None:
                values.append(value)
        if not values:
            result[field] = {"mean": None, "std": None, "var": None}
        else:
            arr = np.asarray(values, dtype=np.float64)
            result[field] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "var": float(arr.var()),
            }
    return result


def write_dataset_csv(path: Path, reports: list[dict], run_tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "mask_type",
        "split_idx",
        "run_tag",
        "best_epoch",
        "best_score",
        "train_mae",
        "train_rmse",
        "train_mae_raw",
        "train_rmse_raw",
        "train_f1",
        "train_accuracy",
        "valid_mae",
        "valid_rmse",
        "valid_mae_raw",
        "valid_rmse_raw",
        "valid_f1",
        "valid_accuracy",
        "test_mae",
        "test_rmse",
        "test_mae_raw",
        "test_rmse_raw",
        "test_f1",
        "test_accuracy",
        "completed",
        "stopped_early",
        "stop_reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            train = report.get("train_metrics") or {}
            valid = report.get("valid_metrics") or {}
            test = report.get("test_metrics") or {}
            writer.writerow(
                _json_safe(
                    {
                        "dataset": report.get("dataset"),
                        "mask_type": report.get("mask_type"),
                        "split_idx": report.get("split_idx"),
                        "run_tag": run_tag,
                        "best_epoch": report.get("best_epoch"),
                        "best_score": report.get("best_score"),
                        "train_mae": train.get("mae"),
                        "train_rmse": train.get("rmse"),
                        "train_mae_raw": train.get("mae_raw"),
                        "train_rmse_raw": train.get("rmse_raw"),
                        "train_f1": train.get("f1"),
                        "train_accuracy": train.get("accuracy"),
                        "valid_mae": valid.get("mae"),
                        "valid_rmse": valid.get("rmse"),
                        "valid_mae_raw": valid.get("mae_raw"),
                        "valid_rmse_raw": valid.get("rmse_raw"),
                        "valid_f1": valid.get("f1"),
                        "valid_accuracy": valid.get("accuracy"),
                        "test_mae": test.get("mae"),
                        "test_rmse": test.get("rmse"),
                        "test_mae_raw": test.get("mae_raw"),
                        "test_rmse_raw": test.get("rmse_raw"),
                        "test_f1": test.get("f1"),
                        "test_accuracy": test.get("accuracy"),
                        "completed": report.get("completed"),
                        "stopped_early": report.get("stopped_early"),
                        "stop_reason": report.get("stop_reason"),
                    }
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MIWAE on fixed tabular imputation benchmark splits.")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mask-type", choices=DEFAULT_MASK_TYPES, default=DEFAULT_MASK_TYPES[0])
    parser.add_argument("--ratio", type=str, default="30")
    parser.add_argument("--split-idx", type=int, default=0)
    parser.add_argument("--run-tag", type=str, default="miwae_student_t_e600_es")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--early-stop-patience", type=int, default=45)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0002)
    parser.add_argument("--valid-max-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--n-hidden", type=int, default=128)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--n-latent", type=int, default=None)
    parser.add_argument("--model-kind", choices=["notmiwae", "miwae"], default="miwae")
    parser.add_argument("--model-variant", choices=["native"], default="native")
    parser.add_argument("--missing-process", type=str, default="selfmasking_known")
    parser.add_argument("--out-dist", choices=["t", "t-distribution"], default="t")
    parser.add_argument("--cat-likelihood", choices=["not_applicable"], default="not_applicable")
    parser.add_argument("--selection-metric", choices=["score", "loss"], default="score")
    parser.add_argument("--selection-impute-samples", type=int, default=1000)
    parser.add_argument("--final-impute-samples", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="ignore partial history/checkpoints unless --overwrite is used")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    install_tf1_compat()
    reports = []
    for dataset in args.datasets:
        report = run_single_split(args, dataset, args.split_idx)
        reports.append(report)

    summary_root = args.output_root / "_summaries"
    run_tag_slug = _slugify_tag(args.run_tag)
    summary_payload = {
        "run_tag": args.run_tag,
        "mask_type": args.mask_type,
        "split_idx": args.split_idx,
        "datasets": args.datasets,
        "aggregate_test_metrics": aggregate_metrics([r for r in reports if r.get("completed")]),
        "reports": reports,
    }
    save_json(summary_root / f"{args.mask_type}_{run_tag_slug}_summary.json", summary_payload)
    write_dataset_csv(summary_root / f"{args.mask_type}_{run_tag_slug}_split_metrics.csv", reports, args.run_tag)
    print(f"[done] summary={summary_root}", flush=True)


if __name__ == "__main__":
    main()
