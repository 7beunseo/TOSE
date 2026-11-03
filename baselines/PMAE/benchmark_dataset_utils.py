import json
import math
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score


def _sha1_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    digest = hashlib.sha1()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


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


def _safe_mean(x: np.ndarray) -> float | None:
    if x.size == 0:
        return None
    value = float(np.mean(x))
    return None if not math.isfinite(value) else value


def _safe_metric(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _flatten_masked_cat_labels(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = []
    for col_idx in range(mask.shape[1]):
        col_mask = mask[:, col_idx]
        if col_mask.sum() > 0:
            values.extend([f"{col_idx}:{value}" for value in labels[col_mask, col_idx].astype(str)])
    return np.asarray(values, dtype=object)


def _load_protocol_indices(normalized_dir: Path, split_idx: int) -> tuple[np.ndarray, np.ndarray]:
    combined_path = normalized_dir / f"split_{split_idx}.npz"
    if combined_path.exists():
        normalized = np.load(combined_path)
        files = set(normalized.files)
        if {"train_indices", "valid_indices"}.issubset(files):
            return normalized["train_indices"].astype(np.int64), normalized["valid_indices"].astype(np.int64)

    train_npz = np.load(normalized_dir / f"split_{split_idx}_train.npz")
    valid_npz = np.load(normalized_dir / f"split_{split_idx}_valid.npz")
    return train_npz["indices"].astype(np.int64), valid_npz["indices"].astype(np.int64)


def _load_normalized_numeric_stats(normalized_dir: Path, split_idx: int, num_dim: int) -> tuple[np.ndarray, np.ndarray]:
    if num_dim == 0:
        return np.zeros(0, dtype=np.float32), np.ones(0, dtype=np.float32)

    combined_path = normalized_dir / f"split_{split_idx}.npz"
    if combined_path.exists():
        normalized = np.load(combined_path)
        files = set(normalized.files)
        if {"mean", "std"}.issubset(files):
            mean = normalized["mean"].astype(np.float32)[:num_dim]
            std = normalized["std"].astype(np.float32)[:num_dim]
            return mean, np.where(std > 1e-12, std, 1.0).astype(np.float32)

    train_npz = np.load(normalized_dir / f"split_{split_idx}_train.npz")
    mean = train_npz["mean"].astype(np.float32)[:num_dim]
    std = train_npz["std"].astype(np.float32)[:num_dim]
    return mean, np.where(std > 1e-12, std, 1.0).astype(np.float32)


def _load_normalized_numeric_columns(
    normalized_dir: Path,
    split_idx: int,
    num_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined_path = normalized_dir / f"split_{split_idx}.npz"
    if combined_path.exists():
        normalized = np.load(combined_path)
        files = set(normalized.files)
        keys = {"train_x_norm", "valid_x_norm", "test_x_norm"}
        if keys.issubset(files):
            return (
                normalized["train_x_norm"].astype(np.float32)[:, :num_dim],
                normalized["valid_x_norm"].astype(np.float32)[:, :num_dim],
                normalized["test_x_norm"].astype(np.float32)[:, :num_dim],
            )

    train_npz = np.load(normalized_dir / f"split_{split_idx}_train.npz")
    valid_npz = np.load(normalized_dir / f"split_{split_idx}_valid.npz")
    test_npz = np.load(normalized_dir / f"split_{split_idx}_test.npz")
    return (
        train_npz["x_norm"].astype(np.float32)[:, :num_dim],
        valid_npz["x_norm"].astype(np.float32)[:, :num_dim],
        test_npz["x_norm"].astype(np.float32)[:, :num_dim],
    )


def _normalized_numeric_stats_source(normalized_dir: Path, split_idx: int) -> Path:
    combined_path = normalized_dir / f"split_{split_idx}.npz"
    if combined_path.exists():
        normalized = np.load(combined_path)
        if {"mean", "std"}.issubset(set(normalized.files)):
            return combined_path
    return normalized_dir / f"split_{split_idx}_train.npz"


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


class BenchmarkDataBundle:
    """Loads the prepared benchmark split/masks without regenerating anything."""

    def __init__(
        self,
        dataset_name: str,
        mask_type: str,
        split_idx: int,
        ratio: str = "30",
        data_root: str | Path = Path(__file__).resolve().parents[1] / "data",
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
        self.normalized_dir = self.dataset_dir / "normalized" / f"rate{self.ratio}" / mask_type

        with open(self.info_path, "r", encoding="utf-8") as f:
            self.info = json.load(f)

        self.num_col_idx = self.info["num_col_idx"]
        self.cat_col_idx = self.info["cat_col_idx"]

        split_payload = self._load_normalized_split()
        self.strict_split = bool(split_payload["strict_split"])
        self.mean = torch.from_numpy(split_payload["mean"]).float()
        self.std = torch.from_numpy(split_payload["std"]).float()
        self.cat_bin_num = torch.from_numpy(split_payload["cat_bin_num"]).long()

        raw_data = self._load_raw_eval_data()
        full_train_x_raw = torch.from_numpy(raw_data["train_x_raw"]).float()
        test_x_raw = torch.from_numpy(raw_data["test_x_raw"]).float()
        full_train_mask_original = torch.from_numpy(raw_data["train_mask_original"].astype(np.bool_))
        test_mask_original = torch.from_numpy(raw_data["test_mask_original"].astype(np.bool_))
        full_train_cat_idx = raw_data["train_cat_idx"]
        test_cat_idx = raw_data["test_cat_idx"]
        self.num_dim = raw_data["num_dim"]
        self.bit_dim = int(full_train_x_raw.shape[1] - self.num_dim)

        self.train_indices = torch.from_numpy(split_payload["train_indices"])
        self.valid_indices = torch.from_numpy(split_payload["valid_indices"])

        self.train_x_norm = torch.from_numpy(split_payload["train_x_norm"]).float()
        self.valid_x_norm = torch.from_numpy(split_payload["valid_x_norm"]).float()
        self.test_x_norm = torch.from_numpy(split_payload["test_x_norm"]).float()

        self.train_mask_ext = torch.from_numpy(split_payload["train_mask"].astype(np.bool_))
        self.valid_mask_ext = torch.from_numpy(split_payload["valid_mask"].astype(np.bool_))
        self.test_mask_ext = torch.from_numpy(split_payload["test_mask"].astype(np.bool_))

        self.train_x_raw = full_train_x_raw[self.train_indices]
        self.valid_x_raw = full_train_x_raw[self.valid_indices]
        self.test_x_raw = test_x_raw

        self.train_mask_original = full_train_mask_original[self.train_indices]
        self.valid_mask_original = full_train_mask_original[self.valid_indices]
        self.test_mask_original = test_mask_original

        self.train_cat_idx = None if full_train_cat_idx is None else full_train_cat_idx[self.train_indices.numpy()]
        self.valid_cat_idx = None if full_train_cat_idx is None else full_train_cat_idx[self.valid_indices.numpy()]
        self.test_cat_idx = test_cat_idx

        expected_dim = int(self.train_x_raw.shape[1])
        for name, mask in {
            "train": self.train_mask_ext,
            "valid": self.valid_mask_ext,
            "test": self.test_mask_ext,
        }.items():
            if int(mask.shape[1]) != expected_dim:
                raise ValueError(
                    f"{dataset_name} {mask_type} split_{split_idx} {name} mask has dim {mask.shape[1]}, "
                    f"expected encoded feature dim {expected_dim}"
                )

    def split_sizes(self) -> dict[str, int]:
        return {
            "train": int(self.train_x_raw.shape[0]),
            "valid": int(self.valid_x_raw.shape[0]),
            "test": int(self.test_x_raw.shape[0]),
        }

    def _load_normalized_split(self) -> dict[str, Any]:
        combined_path = self.normalized_dir / f"split_{self.split_idx}.npz"
        if combined_path.exists():
            normalized = np.load(combined_path)
            files = set(normalized.files)
            strict_keys = {"train_x_norm", "valid_x_norm", "test_x_norm", "train_mask", "valid_mask", "test_mask"}
            if strict_keys.issubset(files):
                return {
                    "strict_split": bool(normalized.get("strict_split", True)),
                    "train_indices": normalized["train_indices"].astype(np.int64),
                    "valid_indices": normalized["valid_indices"].astype(np.int64),
                    "train_x_norm": normalized["train_x_norm"].astype(np.float32),
                    "valid_x_norm": normalized["valid_x_norm"].astype(np.float32),
                    "test_x_norm": normalized["test_x_norm"].astype(np.float32),
                    "train_mask": normalized["train_mask"].astype(bool),
                    "valid_mask": normalized["valid_mask"].astype(bool),
                    "test_mask": normalized["test_mask"].astype(bool),
                    "mean": normalized["mean"].astype(np.float32),
                    "std": normalized["std"].astype(np.float32),
                    "cat_bin_num": normalized["cat_bin_num"].astype(np.int64),
                }

        train_npz = np.load(self.normalized_dir / f"split_{self.split_idx}_train.npz")
        valid_npz = np.load(self.normalized_dir / f"split_{self.split_idx}_valid.npz")
        test_npz = np.load(self.normalized_dir / f"split_{self.split_idx}_test.npz")
        train_indices = train_npz["indices"].astype(np.int64)
        valid_indices = valid_npz["indices"].astype(np.int64)
        return {
            "strict_split": True,
            "train_indices": train_indices,
            "valid_indices": valid_indices,
            "train_x_norm": train_npz["x_norm"].astype(np.float32),
            "valid_x_norm": valid_npz["x_norm"].astype(np.float32),
            "test_x_norm": test_npz["x_norm"].astype(np.float32),
            "train_mask": train_npz["mask"].astype(bool),
            "valid_mask": valid_npz["mask"].astype(bool),
            "test_mask": test_npz["mask"].astype(bool),
            "mean": train_npz["mean"].astype(np.float32),
            "std": train_npz["std"].astype(np.float32),
            "cat_bin_num": train_npz["cat_bin_num"].astype(np.int64),
        }

    def _load_raw_eval_data(self) -> dict[str, Any]:
        train_df = pd.read_csv(self.dataset_dir / "train.csv")
        test_df = pd.read_csv(self.dataset_dir / "test.csv")
        data_df = pd.read_csv(self.dataset_dir / "data.csv")

        mask_dir = self.dataset_dir / "masks" / f"rate{self.ratio}" / self.mask_type
        train_mask_original = np.load(mask_dir / f"train_mask_{self.split_idx}.npy").astype(bool)
        test_mask_original = np.load(mask_dir / f"test_mask_{self.split_idx}.npy").astype(bool)

        cols = train_df.columns
        train_num = train_df[cols[self.num_col_idx]].values.astype(np.float32)
        test_num = test_df[cols[self.num_col_idx]].values.astype(np.float32)

        if len(self.cat_col_idx) == 0:
            train_x_raw = train_num
            test_x_raw = test_num
            train_cat_idx = None
            test_cat_idx = None
            cat_bin_num = np.array([], dtype=np.int64)
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
                    raise ValueError(f"Unmapped category found in {self.dataset_name}.{column}")

                train_cat_bin_i = np.array([list(map(int, bits)) for bits in train_cat_enc], dtype=np.float32)
                test_cat_bin_i = np.array([list(map(int, bits)) for bits in test_cat_enc], dtype=np.float32)
                train_cat_idx_i = train_cat[column].map(category_to_idx).to_numpy().astype(np.int64)
                test_cat_idx_i = test_cat[column].map(category_to_idx).to_numpy().astype(np.int64)

                train_cat_bin.append(train_cat_bin_i)
                test_cat_bin.append(test_cat_bin_i)
                train_cat_idx.append(train_cat_idx_i)
                test_cat_idx.append(test_cat_idx_i)
                cat_bin_num.append(train_cat_bin_i.shape[1])

            train_cat_bin = np.concatenate(train_cat_bin, axis=1)
            test_cat_bin = np.concatenate(test_cat_bin, axis=1)
            train_cat_idx = np.stack(train_cat_idx, axis=1)
            test_cat_idx = np.stack(test_cat_idx, axis=1)
            cat_bin_num = np.array(cat_bin_num, dtype=np.int64)

            train_x_raw = np.concatenate([train_num, train_cat_bin], axis=1)
            test_x_raw = np.concatenate([test_num, test_cat_bin], axis=1)

        return {
            "train_x_raw": train_x_raw.astype(np.float32),
            "test_x_raw": test_x_raw.astype(np.float32),
            "train_mask_original": train_mask_original.astype(bool),
            "test_mask_original": test_mask_original.astype(bool),
            "train_cat_idx": None if train_cat_idx is None else train_cat_idx.astype(np.int64),
            "test_cat_idx": None if test_cat_idx is None else test_cat_idx.astype(np.int64),
            "num_dim": len(self.num_col_idx),
            "cat_bin_num": cat_bin_num.astype(np.int64),
        }


class BaselineOneHotDataBundle:
    """Same split/mask protocol with baseline one-hot categorical handling.

    This adapter intentionally does not use Eunseo binary codebooks. It fits
    categorical vocabularies on the strict training rows only, represents
    categories as one-hot groups, and decodes predictions by group argmax.
    """

    categorical_preprocessing_name = "baseline_one_hot"

    def __init__(
        self,
        dataset_name: str,
        mask_type: str,
        split_idx: int,
        ratio: str = "30",
        data_root: str | Path = Path(__file__).resolve().parents[1] / "data",
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
        self.normalized_dir = self.dataset_dir / "normalized" / f"rate{self.ratio}" / mask_type

        with open(self.info_path, "r", encoding="utf-8") as f:
            self.info = json.load(f)

        self.num_col_idx = self.info["num_col_idx"]
        self.cat_col_idx = self.info["cat_col_idx"]

        self.train_indices_np, self.valid_indices_np = _load_protocol_indices(self.normalized_dir, self.split_idx)
        self.train_indices = torch.from_numpy(self.train_indices_np)
        self.valid_indices = torch.from_numpy(self.valid_indices_np)

        self._load_raw_frames_and_masks()
        self._fit_numeric_standardizer()
        self._fit_categorical_one_hot()
        self._build_model_arrays()
        self._build_metadata()

    def _load_raw_frames_and_masks(self) -> None:
        train_df = pd.read_csv(self.dataset_dir / "train.csv")
        test_df = pd.read_csv(self.dataset_dir / "test.csv")

        mask_dir = self.dataset_dir / "masks" / f"rate{self.ratio}" / self.mask_type
        full_train_mask_original = np.load(mask_dir / f"train_mask_{self.split_idx}.npy").astype(bool)
        test_mask_original = np.load(mask_dir / f"test_mask_{self.split_idx}.npy").astype(bool)

        cols = train_df.columns
        self.num_columns = [str(c) for c in cols[self.num_col_idx]]
        self.cat_columns = [str(c) for c in cols[self.cat_col_idx]]

        full_train_num = (
            train_df[self.num_columns].values.astype(np.float32)
            if len(self.num_columns) > 0
            else np.zeros((len(train_df), 0), dtype=np.float32)
        )
        test_num = (
            test_df[self.num_columns].values.astype(np.float32)
            if len(self.num_columns) > 0
            else np.zeros((len(test_df), 0), dtype=np.float32)
        )
        full_train_cat = (
            train_df[self.cat_columns].astype(str).to_numpy()
            if len(self.cat_columns) > 0
            else np.empty((len(train_df), 0), dtype=object)
        )
        test_cat = (
            test_df[self.cat_columns].astype(str).to_numpy()
            if len(self.cat_columns) > 0
            else np.empty((len(test_df), 0), dtype=object)
        )

        self.full_train_num_raw = full_train_num
        self.test_num_raw = test_num
        self.full_train_cat_labels = full_train_cat
        self.test_cat_labels = test_cat

        self.train_num_raw = full_train_num[self.train_indices_np]
        self.valid_num_raw = full_train_num[self.valid_indices_np]
        self.train_cat_labels = full_train_cat[self.train_indices_np]
        self.valid_cat_labels = full_train_cat[self.valid_indices_np]

        self.train_mask_original_np = full_train_mask_original[self.train_indices_np]
        self.valid_mask_original_np = full_train_mask_original[self.valid_indices_np]
        self.test_mask_original_np = test_mask_original

        self.train_mask_original = torch.from_numpy(self.train_mask_original_np.astype(np.bool_))
        self.valid_mask_original = torch.from_numpy(self.valid_mask_original_np.astype(np.bool_))
        self.test_mask_original = torch.from_numpy(self.test_mask_original_np.astype(np.bool_))

    def _fit_numeric_standardizer(self) -> None:
        num_dim = len(self.num_col_idx)
        mean, std = _load_normalized_numeric_stats(self.normalized_dir, self.split_idx, num_dim)
        train_num_norm, valid_num_norm, test_num_norm = _load_normalized_numeric_columns(
            self.normalized_dir,
            self.split_idx,
            num_dim,
        )
        self.numeric_stats_source = _normalized_numeric_stats_source(self.normalized_dir, self.split_idx)
        self.mean = torch.from_numpy(mean).float()
        self.std = torch.from_numpy(std).float()
        self.train_num_norm = train_num_norm
        self.valid_num_norm = valid_num_norm
        self.test_num_norm = test_num_norm
        self._validate_numeric_npz_inputs()

    def _standardize_num(self, values: np.ndarray) -> np.ndarray:
        if values.shape[1] == 0:
            return values.astype(np.float32)
        mean = self.mean.cpu().numpy()[None, :]
        std = self.std.cpu().numpy()[None, :]
        return ((values.astype(np.float32) - mean) / std / 2.0).astype(np.float32)

    def _validate_numeric_npz_inputs(self) -> None:
        if len(self.num_col_idx) == 0:
            self.numeric_input_matches_npz = True
            self.numeric_input_max_abs_diff = 0.0
            return
        reloaded_train, reloaded_valid, reloaded_test = _load_normalized_numeric_columns(
            self.normalized_dir,
            self.split_idx,
            len(self.num_col_idx),
        )
        diffs = [
            float(np.max(np.abs(self.train_num_norm - reloaded_train))),
            float(np.max(np.abs(self.valid_num_norm - reloaded_valid))),
            float(np.max(np.abs(self.test_num_norm - reloaded_test))),
        ]
        self.numeric_input_max_abs_diff = max(diffs)
        self.numeric_input_matches_npz = bool(
            np.array_equal(self.train_num_norm, reloaded_train)
            and np.array_equal(self.valid_num_norm, reloaded_valid)
            and np.array_equal(self.test_num_norm, reloaded_test)
        )
        if not self.numeric_input_matches_npz:
            raise ValueError(
                f"{self.dataset_name} {self.mask_type} split_{self.split_idx}: numeric input does not match "
                f"prepared npz x_norm columns; max_abs_diff={self.numeric_input_max_abs_diff}"
            )

    def _fit_categorical_one_hot(self) -> None:
        self.cat_vocabularies: list[list[str]] = []
        self.cat_label_to_index: list[dict[str, int]] = []
        self.cat_onehot_num: list[int] = []
        if len(self.cat_col_idx) == 0:
            self.cat_onehot_num_tensor = torch.zeros(0, dtype=torch.long)
            return

        strict_train_labels = self.train_cat_labels
        for j in range(strict_train_labels.shape[1]):
            values = sorted({str(v) for v in strict_train_labels[:, j]})
            if len(values) == 0:
                values = ["__UNK__"]
            mapping = {label: idx for idx, label in enumerate(values)}
            self.cat_vocabularies.append(values)
            self.cat_label_to_index.append(mapping)
            self.cat_onehot_num.append(len(values))
        self.cat_onehot_num_tensor = torch.from_numpy(np.asarray(self.cat_onehot_num, dtype=np.int64)).long()

    def _encode_one_hot(self, labels: np.ndarray) -> np.ndarray:
        if len(self.cat_col_idx) == 0:
            return np.zeros((labels.shape[0], 0), dtype=np.float32)
        columns = []
        for j, mapping in enumerate(self.cat_label_to_index):
            width = len(mapping)
            encoded = np.zeros((labels.shape[0], width), dtype=np.float32)
            for row_idx, value in enumerate(labels[:, j]):
                idx = mapping.get(str(value))
                if idx is not None:
                    encoded[row_idx, idx] = 1.0
            columns.append(encoded)
        return np.concatenate(columns, axis=1).astype(np.float32)

    def _extend_original_mask_to_model(self, mask_original: np.ndarray) -> np.ndarray:
        num_mask = mask_original[:, self.num_col_idx] if len(self.num_col_idx) > 0 else np.zeros((mask_original.shape[0], 0), dtype=bool)
        if len(self.cat_col_idx) == 0:
            return num_mask.astype(bool)
        cat_mask = mask_original[:, self.cat_col_idx]
        cat_mask_ext = _extend_mask(cat_mask, np.asarray(self.cat_onehot_num, dtype=np.int64))
        return np.concatenate([num_mask, cat_mask_ext], axis=1).astype(bool)

    def _build_model_arrays(self) -> None:
        train_num_std = self.train_num_norm
        valid_num_std = self.valid_num_norm
        test_num_std = self.test_num_norm

        train_cat_oh = self._encode_one_hot(self.train_cat_labels)
        valid_cat_oh = self._encode_one_hot(self.valid_cat_labels)
        test_cat_oh = self._encode_one_hot(self.test_cat_labels)

        self.train_x_raw = torch.from_numpy(np.concatenate([train_num_std, train_cat_oh], axis=1)).float()
        self.valid_x_raw = torch.from_numpy(np.concatenate([valid_num_std, valid_cat_oh], axis=1)).float()
        self.test_x_raw = torch.from_numpy(np.concatenate([test_num_std, test_cat_oh], axis=1)).float()

        # Kept for compatibility with summary code; these are model-space arrays.
        self.train_x_norm = self.train_x_raw.clone()
        self.valid_x_norm = self.valid_x_raw.clone()
        self.test_x_norm = self.test_x_raw.clone()

        self.train_mask_ext = torch.from_numpy(self._extend_original_mask_to_model(self.train_mask_original_np))
        self.valid_mask_ext = torch.from_numpy(self._extend_original_mask_to_model(self.valid_mask_original_np))
        self.test_mask_ext = torch.from_numpy(self._extend_original_mask_to_model(self.test_mask_original_np))

        self.num_dim = len(self.num_col_idx)
        self.bit_dim = int(self.train_x_raw.shape[1] - self.num_dim)

    def _build_metadata(self) -> None:
        self.preprocessing_metadata = {
            "model_family": "baseline",
            "numeric_preprocessing": "normalized_npz_x_norm_numeric_columns",
            "numeric_fit_scope": "normalized_npz_mean_std",
            "numeric_stats_source": str(self.numeric_stats_source),
            "numeric_input_source": "train_x_norm_valid_x_norm_test_x_norm_numeric_columns",
            "numeric_source": "prepared_npz_x_norm",
            "scaler_source": "prepared_npz_mean_std",
            "metric_space": "eunseo_normalized",
            "numeric_metric_unit": "prepared_normalized_unit",
            "numeric_metric_formula": "pred_num_norm - true_x_norm",
            "numeric_metric_scale": "eunseo_x_norm_formula_(x-mean)/std/2.0",
            "categorical_protocol": "model_native",
            "numeric_transform_formula": "(x - mean) / std / 2.0",
            "numeric_inverse_formula": "x_norm * 2.0 * std + mean",
            "categorical_preprocessing": "one_hot",
            "categorical_fit_scope": "strict_train_rows_only",
            "categorical_decoding": "one_hot_group_argmax_to_original_label",
            "categorical_metric_level": "column_prefixed_category_index",
            "categorical_f1_average": "weighted_over_column_prefixed_category_indices",
            "cat_f1_average": "weighted_over_column_prefixed_category_indices",
            "categorical_token_format": "col{original_column_index}=idx{category_index}",
            "uses_eunseo_codebook": False,
            "uses_eunseo_codebook_loss": False,
            "uses_eunseo_codebook_argmax": False,
            "cat_columns": self.cat_columns,
            "cat_onehot_num": list(map(int, self.cat_onehot_num)),
        }
        self.protocol_checks = {
            "split_idx": self.split_idx,
            "train_indices_sha1": _sha1_array(self.train_indices_np.astype(np.int64)),
            "valid_indices_sha1": _sha1_array(self.valid_indices_np.astype(np.int64)),
            "train_missing_mask_original_sha1": _sha1_array(self.train_mask_original_np.astype(np.bool_)),
            "valid_missing_mask_original_sha1": _sha1_array(self.valid_mask_original_np.astype(np.bool_)),
            "test_missing_mask_original_sha1": _sha1_array(self.test_mask_original_np.astype(np.bool_)),
            "train_eval_mask_original_sha1": _sha1_array(self.train_mask_original_np.astype(np.bool_)),
            "valid_eval_mask_original_sha1": _sha1_array(self.valid_mask_original_np.astype(np.bool_)),
            "test_eval_mask_original_sha1": _sha1_array(self.test_mask_original_np.astype(np.bool_)),
            "numeric_stats_fit_on_strict_train_observed_only": False,
            "numeric_stats_loaded_from_normalized_npz": True,
            "numeric_columns_loaded_from_npz_x_norm": True,
            "raw_csv_numeric_scaler_fit": False,
            "scaler_refit_in_runner": False,
            "categorical_eval_on_original_labels": True,
            "numeric_input_first_num_dim_matches_npz": self.numeric_input_matches_npz,
            "numeric_input_max_abs_diff": self.numeric_input_max_abs_diff,
        }

    def split_sizes(self) -> dict[str, int]:
        return {
            "train": int(self.train_x_raw.shape[0]),
            "valid": int(self.valid_x_raw.shape[0]),
            "test": int(self.test_x_raw.shape[0]),
        }

    def _split_truth(self, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if split == "train":
            return self.train_num_norm, self.train_num_raw, self.train_cat_labels, self.train_mask_original_np
        if split == "valid":
            return self.valid_num_norm, self.valid_num_raw, self.valid_cat_labels, self.valid_mask_original_np
        if split == "test":
            return self.test_num_norm, self.test_num_raw, self.test_cat_labels, self.test_mask_original_np
        raise ValueError(f"Invalid split: {split}")

    def decode_categorical_predictions(self, pred_model: np.ndarray) -> np.ndarray:
        if len(self.cat_col_idx) == 0:
            return np.empty((pred_model.shape[0], 0), dtype=object)
        start = self.num_dim
        columns = []
        for vocab, width in zip(self.cat_vocabularies, self.cat_onehot_num):
            logits = pred_model[:, start : start + width]
            chosen = np.argmax(logits, axis=1)
            columns.append(np.asarray([vocab[int(idx)] for idx in chosen], dtype=object))
            start += width
        return np.stack(columns, axis=1)

    def _categorical_labels_to_indices(self, labels: np.ndarray) -> np.ndarray:
        if len(self.cat_col_idx) == 0:
            return np.empty((labels.shape[0], 0), dtype=np.int64)
        indices = np.full((labels.shape[0], len(self.cat_label_to_index)), -1, dtype=np.int64)
        for col_pos, mapping in enumerate(self.cat_label_to_index):
            for row_idx, value in enumerate(labels[:, col_pos]):
                indices[row_idx, col_pos] = int(mapping.get(str(value), -1))
        return indices

    def evaluate_predictions(self, pred_model: np.ndarray, split: str) -> dict[str, float | None]:
        true_num_norm, true_num_raw, true_cat_labels, mask_original = self._split_truth(split)
        num_dim = self.num_dim

        if num_dim > 0:
            num_mask = mask_original[:, self.num_col_idx]
            pred_num_norm = pred_model[:, :num_dim]
            diff = pred_num_norm[num_mask] - true_num_norm[num_mask]
            mae = _safe_mean(np.abs(diff))
            mse = _safe_mean(diff**2)
            rmse = None if mse is None else float(np.sqrt(mse))

            pred_num_raw = pred_num_norm * 2.0 * self.std.cpu().numpy()[None, :] + self.mean.cpu().numpy()[None, :]
            raw_diff = pred_num_raw[num_mask] - true_num_raw[num_mask]
            mae_raw = _safe_mean(np.abs(raw_diff))
            mse_raw = _safe_mean(raw_diff**2)
            rmse_raw = None if mse_raw is None else float(np.sqrt(mse_raw))
        else:
            mae = None
            mse = None
            rmse = None
            mae_raw = None
            mse_raw = None
            rmse_raw = None

        cat_f1 = None
        cat_f1_macro = None
        cat_f1_weighted = None
        cat_accuracy = None
        if len(self.cat_col_idx) > 0:
            pred_cat_labels = self.decode_categorical_predictions(pred_model)
            true_cat_idx = self._categorical_labels_to_indices(true_cat_labels)
            pred_cat_idx = self._categorical_labels_to_indices(pred_cat_labels)
            cat_mask = mask_original[:, self.cat_col_idx]
            if cat_mask.sum() > 0:
                cat_accuracy = float((pred_cat_idx[cat_mask] == true_cat_idx[cat_mask]).mean())
                y_true = []
                y_pred = []
                for col_pos, original_col in enumerate(self.cat_col_idx):
                    rows = np.where(cat_mask[:, col_pos])[0]
                    if rows.size == 0:
                        continue
                    y_true.extend(
                        f"col{int(original_col)}=idx{int(value)}"
                        for value in true_cat_idx[rows, col_pos].tolist()
                    )
                    y_pred.extend(
                        f"col{int(original_col)}=idx{int(value)}"
                        for value in pred_cat_idx[rows, col_pos].tolist()
                    )
                cat_f1_macro = _safe_metric(f1_score(y_true, y_pred, average="macro", zero_division=0))
                cat_f1_weighted = _safe_metric(f1_score(y_true, y_pred, average="weighted", zero_division=0))
                cat_f1 = cat_f1_weighted

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "cat_f1": cat_f1,
            "cat_f1_macro": cat_f1_macro,
            "cat_f1_weighted": cat_f1_weighted,
            "bit_f1": cat_f1,
            "cat_accuracy": cat_accuracy,
            "accuracy": cat_accuracy,
            "mae_raw": mae_raw,
            "mse_raw": mse_raw,
            "rmse_raw": rmse_raw,
            "numeric_source": "prepared_npz_x_norm",
            "scaler_source": "prepared_npz_mean_std",
            "metric_space": "eunseo_normalized",
            "numeric_metric_unit": "prepared_normalized_unit",
            "numeric_metric_formula": "pred_num_norm - true_x_norm",
            "numeric_metric_scale": "eunseo_x_norm_formula_(x-mean)/std/2.0",
            "categorical_protocol": "model_native",
            "categorical_metric_level": "column_prefixed_category_index",
            "categorical_f1_average": "weighted_over_column_prefixed_category_indices",
            "cat_f1_average": "weighted_over_column_prefixed_category_indices",
            "categorical_token_format": "col{original_column_index}=idx{category_index}",
            "MAE": mae,
            "RMSE": rmse,
            "f1": cat_f1,
            "F1": cat_f1,
            "Accuracy": cat_accuracy,
        }


class PMAELabelRoundDataBundle(BaselineOneHotDataBundle):
    """PMAE paper-style categorical handling.

    The released PMAE implementation treats inputs as numerical values and
    evaluates categorical columns by rounding imputed values. This adapter keeps
    that model-side convention: numeric inputs are raw CSV values, categorical
    inputs are raw category indices, and PMAE's own MinMax wrapper scales them
    for training. Metrics are computed after mapping numeric outputs back into
    the shared Eunseo normalized evaluation space.
    """

    categorical_preprocessing_name = "pmae_label_round"

    def _fit_categorical_one_hot(self) -> None:
        self.cat_vocabularies: list[list[str]] = []
        self.cat_label_to_index: list[dict[str, int]] = []
        self.cat_label_num: list[int] = []
        if len(self.cat_col_idx) == 0:
            self.cat_label_num_tensor = torch.zeros(0, dtype=torch.long)
            return

        data_df = pd.read_csv(self.dataset_dir / "data.csv")
        data_cat = data_df[self.cat_columns].astype(str).to_numpy()
        for j in range(data_cat.shape[1]):
            values = sorted({str(v) for v in data_cat[:, j]})
            if len(values) == 0:
                values = ["__UNK__"]
            mapping = {label: idx for idx, label in enumerate(values)}
            self.cat_vocabularies.append(values)
            self.cat_label_to_index.append(mapping)
            self.cat_label_num.append(len(values))
        self.cat_label_num_tensor = torch.from_numpy(np.asarray(self.cat_label_num, dtype=np.int64)).long()

    def _encode_one_hot(self, labels: np.ndarray) -> np.ndarray:
        if len(self.cat_col_idx) == 0:
            return np.zeros((labels.shape[0], 0), dtype=np.float32)
        encoded = np.zeros((labels.shape[0], len(self.cat_label_to_index)), dtype=np.float32)
        for j, mapping in enumerate(self.cat_label_to_index):
            for row_idx, value in enumerate(labels[:, j]):
                key = str(value)
                if key not in mapping:
                    raise ValueError(f"Unmapped category found in {self.dataset_name}.{self.cat_columns[j]}: {key}")
                encoded[row_idx, j] = float(mapping[key])
        return encoded

    def _extend_original_mask_to_model(self, mask_original: np.ndarray) -> np.ndarray:
        num_mask = (
            mask_original[:, self.num_col_idx]
            if len(self.num_col_idx) > 0
            else np.zeros((mask_original.shape[0], 0), dtype=bool)
        )
        cat_mask = (
            mask_original[:, self.cat_col_idx]
            if len(self.cat_col_idx) > 0
            else np.zeros((mask_original.shape[0], 0), dtype=bool)
        )
        return np.concatenate([num_mask, cat_mask], axis=1).astype(bool)

    def _build_model_arrays(self) -> None:
        train_num_raw = self.train_num_raw.astype(np.float32)
        valid_num_raw = self.valid_num_raw.astype(np.float32)
        test_num_raw = self.test_num_raw.astype(np.float32)

        train_cat_idx = self._encode_one_hot(self.train_cat_labels)
        valid_cat_idx = self._encode_one_hot(self.valid_cat_labels)
        test_cat_idx = self._encode_one_hot(self.test_cat_labels)

        self.train_x_raw = torch.from_numpy(np.concatenate([train_num_raw, train_cat_idx], axis=1)).float()
        self.valid_x_raw = torch.from_numpy(np.concatenate([valid_num_raw, valid_cat_idx], axis=1)).float()
        self.test_x_raw = torch.from_numpy(np.concatenate([test_num_raw, test_cat_idx], axis=1)).float()

        # Compatibility alias only. For PMAE, these tensors are model-space
        # raw inputs; shared normalized metrics are computed explicitly below.
        self.train_x_norm = self.train_x_raw.clone()
        self.valid_x_norm = self.valid_x_raw.clone()
        self.test_x_norm = self.test_x_raw.clone()

        self.train_mask_ext = torch.from_numpy(self._extend_original_mask_to_model(self.train_mask_original_np))
        self.valid_mask_ext = torch.from_numpy(self._extend_original_mask_to_model(self.valid_mask_original_np))
        self.test_mask_ext = torch.from_numpy(self._extend_original_mask_to_model(self.test_mask_original_np))

        self.num_dim = len(self.num_col_idx)
        self.bit_dim = int(self.train_x_raw.shape[1] - self.num_dim)

    def _build_metadata(self) -> None:
        self.preprocessing_metadata = {
            "model_family": "pmae_baseline",
            "numeric_preprocessing": "raw_csv_numeric_columns_then_pmae_native_minmax",
            "numeric_fit_scope": "pmae_imputer_strict_train_observed",
            "numeric_stats_source": str(self.numeric_stats_source),
            "numeric_input_source": "raw_csv_numeric_columns",
            "numeric_source": "prepared_npz_x_norm",
            "scaler_source": "prepared_npz_mean_std",
            "metric_space": "eunseo_normalized",
            "numeric_metric_unit": "prepared_normalized_unit",
            "numeric_metric_formula": "((pred_raw - npz_mean) / npz_std / 2.0) - true_x_norm",
            "numeric_metric_scale": "eunseo_x_norm_formula_(x-mean)/std/2.0",
            "categorical_protocol": "model_native",
            "numeric_train_transform": "pmae_native_minmax_fit_on_train_observed",
            "numeric_transform_formula": "PMAE internal MinMax: (x - train_min) / (train_max - train_min)",
            "numeric_inverse_formula": "PMAE inverse MinMax before shared metric evaluation",
            "categorical_preprocessing": "label_index_then_pmae_native_minmax",
            "categorical_fit_scope": "data_csv_category_vocabulary_to_match_pmae_preprocessing",
            "categorical_transform_formula": "category_index",
            "categorical_input_space": "raw_category_index_then_pmae_minmax",
            "categorical_decoding": "round(pred_category_index)_clip_to_valid_range_inverse_label",
            "categorical_metric_level": "column_prefixed_category_index",
            "categorical_f1_average": "weighted_over_column_prefixed_category_indices",
            "cat_f1_average": "weighted_over_column_prefixed_category_indices",
            "categorical_token_format": "col{original_column_index}=idx{category_index}",
            "uses_eunseo_codebook": False,
            "uses_eunseo_codebook_loss": False,
            "uses_eunseo_codebook_argmax": False,
            "cat_columns": self.cat_columns,
            "cat_label_num": list(map(int, self.cat_label_num)),
        }
        self.protocol_checks = {
            "split_idx": self.split_idx,
            "train_indices_sha1": _sha1_array(self.train_indices_np.astype(np.int64)),
            "valid_indices_sha1": _sha1_array(self.valid_indices_np.astype(np.int64)),
            "train_missing_mask_original_sha1": _sha1_array(self.train_mask_original_np.astype(np.bool_)),
            "valid_missing_mask_original_sha1": _sha1_array(self.valid_mask_original_np.astype(np.bool_)),
            "test_missing_mask_original_sha1": _sha1_array(self.test_mask_original_np.astype(np.bool_)),
            "train_eval_mask_original_sha1": _sha1_array(self.train_mask_original_np.astype(np.bool_)),
            "valid_eval_mask_original_sha1": _sha1_array(self.valid_mask_original_np.astype(np.bool_)),
            "test_eval_mask_original_sha1": _sha1_array(self.test_mask_original_np.astype(np.bool_)),
            "numeric_stats_fit_on_strict_train_observed_only": False,
            "numeric_stats_loaded_from_normalized_npz": True,
            "numeric_columns_loaded_from_npz_x_norm": False,
            "raw_csv_numeric_scaler_fit": True,
            "pmae_native_minmax_fit_on_train_observed": True,
            "pmae_numeric_minmax_fit_on_strict_train_observed_only": True,
            "scaler_refit_in_runner": True,
            "categorical_eval_on_original_labels": True,
            "numeric_truth_x_norm_matches_npz": self.numeric_input_matches_npz,
            "numeric_truth_x_norm_max_abs_diff": self.numeric_input_max_abs_diff,
            "pmae_original_rounding_eval": True,
            "categorical_raw_index_for_pmae_output_range": True,
        }

    def decode_categorical_indices(self, pred_model: np.ndarray) -> np.ndarray:
        if len(self.cat_col_idx) == 0:
            return np.empty((pred_model.shape[0], 0), dtype=np.int64)
        cat_values = pred_model[:, self.num_dim : self.num_dim + len(self.cat_label_to_index)]
        columns = []
        for j, vocab in enumerate(self.cat_vocabularies):
            rounded = np.rint(cat_values[:, j]).astype(np.int64)
            columns.append(np.clip(rounded, 0, len(vocab) - 1))
        return np.stack(columns, axis=1)

    def decode_categorical_predictions(self, pred_model: np.ndarray) -> np.ndarray:
        if len(self.cat_col_idx) == 0:
            return np.empty((pred_model.shape[0], 0), dtype=object)
        cat_indices = self.decode_categorical_indices(pred_model)
        columns = []
        for j, vocab in enumerate(self.cat_vocabularies):
            columns.append(np.asarray([vocab[int(idx)] for idx in cat_indices[:, j]], dtype=object))
        return np.stack(columns, axis=1)

    def evaluate_predictions(self, pred_model: np.ndarray, split: str) -> dict[str, float | None]:
        true_num_norm, true_num_raw, true_cat_labels, mask_original = self._split_truth(split)
        num_dim = self.num_dim

        if num_dim > 0:
            num_mask = mask_original[:, self.num_col_idx]
            pred_num_raw = pred_model[:, :num_dim]
            mean = self.mean.cpu().numpy()[None, :]
            std = self.std.cpu().numpy()[None, :]
            pred_num_norm = ((pred_num_raw - mean) / std / 2.0).astype(np.float32)
            diff = pred_num_norm[num_mask] - true_num_norm[num_mask]
            mae = _safe_mean(np.abs(diff))
            mse = _safe_mean(diff**2)
            rmse = None if mse is None else float(np.sqrt(mse))

            raw_diff = pred_num_raw[num_mask] - true_num_raw[num_mask]
            mae_raw = _safe_mean(np.abs(raw_diff))
            mse_raw = _safe_mean(raw_diff**2)
            rmse_raw = None if mse_raw is None else float(np.sqrt(mse_raw))
        else:
            mae = None
            mse = None
            rmse = None
            mae_raw = None
            mse_raw = None
            rmse_raw = None

        cat_f1 = None
        cat_f1_macro = None
        cat_f1_weighted = None
        cat_accuracy = None
        if len(self.cat_col_idx) > 0:
            true_cat_idx = self._categorical_labels_to_indices(true_cat_labels)
            pred_cat_idx = self.decode_categorical_indices(pred_model)
            cat_mask = mask_original[:, self.cat_col_idx]
            if cat_mask.sum() > 0:
                cat_accuracy = float((pred_cat_idx[cat_mask] == true_cat_idx[cat_mask]).mean())
                y_true = []
                y_pred = []
                for col_pos, original_col in enumerate(self.cat_col_idx):
                    rows = np.where(cat_mask[:, col_pos])[0]
                    if rows.size == 0:
                        continue
                    y_true.extend(
                        f"col{int(original_col)}=idx{int(value)}"
                        for value in true_cat_idx[rows, col_pos].tolist()
                    )
                    y_pred.extend(
                        f"col{int(original_col)}=idx{int(value)}"
                        for value in pred_cat_idx[rows, col_pos].tolist()
                    )
                cat_f1_macro = _safe_metric(f1_score(y_true, y_pred, average="macro", zero_division=0))
                cat_f1_weighted = _safe_metric(f1_score(y_true, y_pred, average="weighted", zero_division=0))
                cat_f1 = cat_f1_weighted

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "cat_f1": cat_f1,
            "cat_f1_macro": cat_f1_macro,
            "cat_f1_weighted": cat_f1_weighted,
            "bit_f1": cat_f1,
            "cat_accuracy": cat_accuracy,
            "accuracy": cat_accuracy,
            "mae_raw": mae_raw,
            "mse_raw": mse_raw,
            "rmse_raw": rmse_raw,
            "numeric_source": "prepared_npz_x_norm",
            "scaler_source": "prepared_npz_mean_std",
            "metric_space": "eunseo_normalized",
            "numeric_metric_unit": "prepared_normalized_unit",
            "numeric_metric_formula": "((pred_raw - npz_mean) / npz_std / 2.0) - true_x_norm",
            "numeric_metric_scale": "eunseo_x_norm_formula_(x-mean)/std/2.0",
            "categorical_protocol": "model_native",
            "categorical_metric_level": "column_prefixed_category_index",
            "categorical_f1_average": "weighted_over_column_prefixed_category_indices",
            "cat_f1_average": "weighted_over_column_prefixed_category_indices",
            "categorical_token_format": "col{original_column_index}=idx{category_index}",
            "MAE": mae,
            "RMSE": rmse,
            "f1": cat_f1,
            "F1": cat_f1,
            "Accuracy": cat_accuracy,
        }


def evaluate_predictions(bundle: Any, pred_raw: np.ndarray, split: str) -> dict[str, float | None]:
    if hasattr(bundle, "evaluate_predictions") and not isinstance(bundle, BenchmarkDataBundle):
        return bundle.evaluate_predictions(pred_raw, split)

    if split == "train":
        true_raw = bundle.train_x_raw.cpu().numpy()
        true_norm = bundle.train_x_norm.cpu().numpy()
        mask_original = bundle.train_mask_original.cpu().numpy()
        truth_cat_idx = bundle.train_cat_idx
    elif split == "valid":
        true_raw = bundle.valid_x_raw.cpu().numpy()
        true_norm = bundle.valid_x_norm.cpu().numpy()
        mask_original = bundle.valid_mask_original.cpu().numpy()
        truth_cat_idx = bundle.valid_cat_idx
    elif split == "test":
        true_raw = bundle.test_x_raw.cpu().numpy()
        true_norm = bundle.test_x_norm.cpu().numpy()
        mask_original = bundle.test_mask_original.cpu().numpy()
        truth_cat_idx = bundle.test_cat_idx
    else:
        raise ValueError(f"Invalid split: {split}")

    return evaluate_arrays(
        pred_raw=pred_raw,
        true_raw=true_raw,
        true_norm=true_norm,
        mask_original=mask_original,
        num_col_idx=bundle.num_col_idx,
        cat_col_idx=bundle.cat_col_idx,
        num_dim=bundle.num_dim,
        cat_bin_num=bundle.cat_bin_num.cpu().numpy(),
        truth_cat_idx=truth_cat_idx,
        mean_num=bundle.mean[: bundle.num_dim].cpu().numpy(),
        std_num=bundle.std[: bundle.num_dim].cpu().numpy(),
    )


def evaluate_arrays(
    pred_raw: np.ndarray,
    true_raw: np.ndarray,
    true_norm: np.ndarray,
    mask_original: np.ndarray,
    num_col_idx: list[int],
    cat_col_idx: list[int],
    num_dim: int,
    cat_bin_num: np.ndarray,
    truth_cat_idx: np.ndarray | None,
    mean_num: np.ndarray,
    std_num: np.ndarray,
) -> dict[str, float | None]:
    if num_dim > 0:
        num_mask = mask_original[:, num_col_idx]
        pred_num_norm = (pred_raw[:, :num_dim] - mean_num[None, :]) / std_num[None, :] / 2.0
        num_diff = pred_num_norm[num_mask] - true_norm[:, :num_dim][num_mask]
        mae = _safe_mean(np.abs(num_diff))
        mse = _safe_mean(num_diff**2)
        rmse = None if mse is None else float(np.sqrt(mse))
    else:
        mae = None
        mse = None
        rmse = None

    legacy_bit_f1 = None
    cat_f1 = None
    cat_f1_macro = None
    cat_f1_weighted = None
    cat_accuracy = None
    if len(cat_col_idx) > 0 and len(cat_bin_num) > 0:
        cat_mask = mask_original[:, cat_col_idx]
        bit_mask = _extend_mask(cat_mask, cat_bin_num)
        if bit_mask.sum() > 0:
            pred_bits = (pred_raw[:, num_dim:] > 0.5).astype(np.int64)
            true_bits = true_raw[:, num_dim:].astype(np.int64)
            legacy_bit_f1 = float(
                f1_score(true_bits[bit_mask].reshape(-1), pred_bits[bit_mask].reshape(-1), zero_division=0)
            )

    if truth_cat_idx is not None and len(cat_col_idx) > 0 and len(cat_bin_num) > 0:
        pred_cat_idx = _recover_categories_from_bits(pred_raw[:, num_dim:], cat_bin_num)
        cat_mask = mask_original[:, cat_col_idx]
        if cat_mask.sum() > 0:
            cat_accuracy = float((pred_cat_idx[cat_mask] == truth_cat_idx[cat_mask]).mean())
            y_true = []
            y_pred = []
            for col_pos, original_col in enumerate(cat_col_idx):
                rows = np.where(cat_mask[:, col_pos])[0]
                if rows.size == 0:
                    continue
                y_true.extend(
                    f"col{int(original_col)}=idx{int(value)}"
                    for value in truth_cat_idx[rows, col_pos].tolist()
                )
                y_pred.extend(
                    f"col{int(original_col)}=idx{int(value)}"
                    for value in pred_cat_idx[rows, col_pos].tolist()
                )
            if y_true:
                cat_f1_macro = _safe_metric(f1_score(y_true, y_pred, average="macro", zero_division=0))
                cat_f1_weighted = _safe_metric(f1_score(y_true, y_pred, average="weighted", zero_division=0))
                cat_f1 = cat_f1_weighted

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "bit_f1": cat_f1,
        "cat_f1": cat_f1,
        "cat_f1_macro": cat_f1_macro,
        "cat_f1_weighted": cat_f1_weighted,
        "legacy_bit_f1": legacy_bit_f1,
        "cat_accuracy": cat_accuracy,
        "accuracy": cat_accuracy,
        "MAE": mae,
        "RMSE": rmse,
        "f1": cat_f1,
        "F1": cat_f1,
        "Accuracy": cat_accuracy,
        "numeric_source": "prepared_npz_x_norm",
        "scaler_source": "prepared_npz_mean_std",
        "metric_space": "eunseo_normalized",
        "numeric_metric_unit": "prepared_normalized_unit",
        "numeric_metric_formula": "pred_num_norm - true_x_norm",
        "numeric_metric_scale": "eunseo_x_norm_formula_(x-mean)/std/2.0",
        "categorical_protocol": "model_native",
        "categorical_metric_level": "column_prefixed_category_index",
        "categorical_f1_average": "weighted_over_column_prefixed_category_indices",
        "cat_f1_average": "weighted_over_column_prefixed_category_indices",
        "categorical_token_format": "col{original_column_index}=idx{category_index}",
    }
