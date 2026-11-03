from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


@dataclass
class SplitData:
    """Model-ready arrays and masks for one data partition."""

    model_x: torch.Tensor
    raw_x: torch.Tensor
    observed_x: torch.Tensor
    missing_mask_ext: torch.Tensor
    original_mask: torch.Tensor
    cat_idx: np.ndarray | None

    @property
    def num_samples(self) -> int:
        return int(self.model_x.shape[0])


class BenchmarkMNARBundle:
    """Load one prepared benchmark repetition without regenerating masks."""

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
        self.split_idx = split_idx
        self.ratio = str(ratio)
        self.data_root = Path(data_root)
        self.seed = int(seed)
        self.valid_max_fraction = float(valid_max_fraction)
        self.dataset_dir = self.data_root / dataset_name
        self.info_path = self.data_root / "Info" / f"{dataset_name}.json"
        self.normalized_path = (
            self.dataset_dir / "normalized" / f"rate{self.ratio}" / mask_type / f"split_{split_idx}.npz"
        )

        with open(self.info_path, "r", encoding="utf-8") as file:
            self.info = json.load(file)

        self.num_col_idx = self.info["num_col_idx"]
        self.cat_col_idx = self.info["cat_col_idx"]

        normalized = np.load(self.normalized_path)
        normalized_files = set(normalized.files)
        self.strict_split = {"valid_x_norm", "valid_mask", "train_indices", "valid_indices"}.issubset(
            normalized_files
        )
        self.mean = torch.from_numpy(normalized["mean"]).float()
        self.std = torch.from_numpy(normalized["std"]).float()
        self.cat_bin_num = torch.from_numpy(normalized["cat_bin_num"]).long()
        if self.strict_split:
            train_x_norm = torch.from_numpy(normalized["train_x_norm"]).float()
            valid_x_norm = torch.from_numpy(normalized["valid_x_norm"]).float()
            test_x_norm = torch.from_numpy(normalized["test_x_norm"]).float()
            train_mask_ext = torch.from_numpy(normalized["train_mask"].astype(np.bool_))
            valid_mask_ext = torch.from_numpy(normalized["valid_mask"].astype(np.bool_))
            test_mask_ext = torch.from_numpy(normalized["test_mask"].astype(np.bool_))
            self.train_indices = torch.from_numpy(normalized["train_indices"].astype(np.int64))
            self.valid_indices = torch.from_numpy(normalized["valid_indices"].astype(np.int64))
        else:
            full_train_x_norm = torch.from_numpy(normalized["train_x_norm"]).float()
            test_x_norm = torch.from_numpy(normalized["test_x_norm"]).float()
            full_train_mask_ext = torch.from_numpy(normalized["train_mask"].astype(np.bool_))
            test_mask_ext = torch.from_numpy(normalized["test_mask"].astype(np.bool_))
            self.train_indices, self.valid_indices = self._build_train_valid_indices(
                train_size=int(full_train_x_norm.shape[0]),
                test_size=int(test_x_norm.shape[0]),
            )
            train_x_norm = full_train_x_norm[self.train_indices]
            valid_x_norm = full_train_x_norm[self.valid_indices]
            train_mask_ext = full_train_mask_ext[self.train_indices]
            valid_mask_ext = full_train_mask_ext[self.valid_indices]

        raw = self._load_raw_eval_data()
        self.num_dim = raw["num_dim"]
        self.bit_dim = int(raw["train_x_raw"].shape[1] - self.num_dim)
        self.cat_codebooks = [torch.from_numpy(codebook).float() for codebook in raw["cat_codebooks"]]
        self.cat_codebooks_np = [codebook.astype(np.int64) for codebook in raw["cat_codebooks"]]
        self.cat_bit_slices = self._build_cat_bit_slices()
        self.feature_group_slices = self._build_feature_group_slices()
        full_train_x_raw = torch.from_numpy(raw["train_x_raw"]).float()
        test_x_raw = torch.from_numpy(raw["test_x_raw"]).float()
        full_train_mask_original = torch.from_numpy(raw["train_mask_original"].astype(np.bool_))
        test_mask_original = torch.from_numpy(raw["test_mask_original"].astype(np.bool_))
        full_train_cat_idx = raw["train_cat_idx"]
        test_cat_idx = raw["test_cat_idx"]

        self.valid_size = int(self.valid_indices.numel())
        self.test_size = int(test_x_norm.shape[0])
        self.train_size = int(self.train_indices.numel())
        train_x_raw = full_train_x_raw[self.train_indices]
        valid_x_raw = full_train_x_raw[self.valid_indices]
        train_mask_original = full_train_mask_original[self.train_indices]
        valid_mask_original = full_train_mask_original[self.valid_indices]
        train_cat_idx = self._slice_optional_array(full_train_cat_idx, self.train_indices)
        valid_cat_idx = self._slice_optional_array(full_train_cat_idx, self.valid_indices)

        self.train_split = self._build_split(
            model_x=self._build_model_space_tensor(train_x_norm, train_x_raw),
            raw_x=train_x_raw,
            missing_mask_ext=train_mask_ext,
            original_mask=train_mask_original,
            cat_idx=train_cat_idx,
        )
        self.valid_split = self._build_split(
            model_x=self._build_model_space_tensor(valid_x_norm, valid_x_raw),
            raw_x=valid_x_raw,
            missing_mask_ext=valid_mask_ext,
            original_mask=valid_mask_original,
            cat_idx=valid_cat_idx,
        )
        self.test_split = self._build_split(
            model_x=self._build_model_space_tensor(test_x_norm, test_x_raw),
            raw_x=test_x_raw,
            missing_mask_ext=test_mask_ext,
            original_mask=test_mask_original,
            cat_idx=test_cat_idx,
        )
        self.group_missing_prior, self.group_co_missing = self._build_pattern_statistics()

    def _build_train_valid_indices(self, train_size: int, test_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if train_size <= 1 or self.valid_max_fraction <= 0.0:
            train_indices = torch.arange(train_size, dtype=torch.long)
            valid_indices = torch.empty(0, dtype=torch.long)
            return train_indices, valid_indices

        requested_valid_size = int(math.floor(train_size * self.valid_max_fraction))
        if requested_valid_size <= 0:
            requested_valid_size = 1
        valid_size = min(test_size, requested_valid_size, train_size - 1)
        if valid_size <= 0:
            train_indices = torch.arange(train_size, dtype=torch.long)
            valid_indices = torch.empty(0, dtype=torch.long)
            return train_indices, valid_indices

        rng = np.random.default_rng(self.seed + 17)
        permutation = rng.permutation(train_size)
        valid_idx_np = np.sort(permutation[:valid_size])
        train_idx_np = np.sort(permutation[valid_size:])
        train_indices = torch.from_numpy(train_idx_np.astype(np.int64))
        valid_indices = torch.from_numpy(valid_idx_np.astype(np.int64))
        return train_indices, valid_indices

    def _slice_optional_array(self, values: np.ndarray | None, indices: torch.Tensor) -> np.ndarray | None:
        if values is None:
            return None
        return values[indices.cpu().numpy()]

    def _build_cat_bit_slices(self) -> list[tuple[int, int]]:
        slices: list[tuple[int, int]] = []
        start = 0
        for bit_len in self.cat_bin_num.tolist():
            end = start + int(bit_len)
            slices.append((start, end))
            start = end
        return slices

    def _build_feature_group_slices(self) -> list[tuple[int, int]]:
        groups = [(idx, idx + 1) for idx in range(self.num_dim)]
        groups.extend((self.num_dim + start, self.num_dim + end) for start, end in self.cat_bit_slices)
        return groups

    def _build_model_space_tensor(self, x_norm: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        """Combine normalized numeric values with binary categorical bits."""
        if self.bit_dim == 0:
            return x_norm[:, : self.num_dim].clone()
        return torch.cat([x_norm[:, : self.num_dim], x_raw[:, self.num_dim :]], dim=1)

    def _build_split(
        self,
        model_x: torch.Tensor,
        raw_x: torch.Tensor,
        missing_mask_ext: torch.Tensor,
        original_mask: torch.Tensor,
        cat_idx: np.ndarray | None,
    ) -> SplitData:
        observed_x = model_x.clone()
        observed_x[missing_mask_ext] = 0.0
        return SplitData(
            model_x=model_x.float(),
            raw_x=raw_x.float(),
            observed_x=observed_x.float(),
            missing_mask_ext=missing_mask_ext.bool(),
            original_mask=original_mask.bool(),
            cat_idx=cat_idx,
        )

    def _load_raw_eval_data(self) -> dict:
        train_df = pd.read_csv(self.dataset_dir / "train.csv")
        test_df = pd.read_csv(self.dataset_dir / "test.csv")
        data_df = pd.read_csv(self.dataset_dir / "data.csv")

        train_mask_original = np.load(
            self.dataset_dir / "masks" / f"rate{self.ratio}" / self.mask_type / f"train_mask_{self.split_idx}.npy"
        )
        test_mask_original = np.load(
            self.dataset_dir / "masks" / f"rate{self.ratio}" / self.mask_type / f"test_mask_{self.split_idx}.npy"
        )

        if train_mask_original.dtype == np.float32:
            train_mask_original = train_mask_original.astype(bool)
        if test_mask_original.dtype == np.float32:
            test_mask_original = test_mask_original.astype(bool)

        columns = train_df.columns
        train_num = train_df[columns[self.num_col_idx]].values.astype(np.float32)
        test_num = test_df[columns[self.num_col_idx]].values.astype(np.float32)

        if len(self.cat_col_idx) == 0:
            train_x_raw = train_num
            test_x_raw = test_num
            train_cat_idx = None
            test_cat_idx = None
            cat_bin_num = np.array([], dtype=np.int64)
            cat_codebooks: list[np.ndarray] = []
        else:
            cat_columns = columns[self.cat_col_idx]
            train_cat = train_df[cat_columns].astype(str)
            test_cat = test_df[cat_columns].astype(str)
            data_cat = data_df[cat_columns].astype(str)

            train_cat_bin = []
            test_cat_bin = []
            train_cat_idx = []
            test_cat_idx = []
            cat_bin_num = []
            cat_codebooks = []

            for column in data_cat.columns:
                with open(self.dataset_dir / f"{column}_map_bin.json", "r", encoding="utf-8") as file:
                    category_to_binary = json.load(file)
                with open(self.dataset_dir / f"{column}_map_idx.json", "r", encoding="utf-8") as file:
                    category_to_idx = json.load(file)

                train_cat_enc = train_cat[column].map(category_to_binary).to_numpy()
                test_cat_enc = test_cat[column].map(category_to_binary).to_numpy()

                train_cat_bin_i = np.array([list(map(int, bits)) for bits in train_cat_enc], dtype=np.float32)
                test_cat_bin_i = np.array([list(map(int, bits)) for bits in test_cat_enc], dtype=np.float32)
                train_cat_idx_i = train_cat[column].map(category_to_idx).to_numpy().astype(np.int64)
                test_cat_idx_i = test_cat[column].map(category_to_idx).to_numpy().astype(np.int64)

                train_cat_bin.append(train_cat_bin_i)
                test_cat_bin.append(test_cat_bin_i)
                train_cat_idx.append(train_cat_idx_i)
                test_cat_idx.append(test_cat_idx_i)
                cat_bin_num.append(train_cat_bin_i.shape[1])

                num_classes = len(category_to_idx)
                bit_len = train_cat_bin_i.shape[1]
                codebook = np.zeros((num_classes, bit_len), dtype=np.float32)
                for category, category_idx in category_to_idx.items():
                    bits = np.array(list(map(int, category_to_binary[category])), dtype=np.float32)
                    codebook[int(category_idx)] = bits
                cat_codebooks.append(codebook)

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
            "cat_codebooks": cat_codebooks,
            "num_dim": len(self.num_col_idx),
            "cat_bin_num": cat_bin_num.astype(np.int64),
        }

    def collapse_cat_bit_mask(self, cat_mask: torch.Tensor) -> torch.Tensor:
        if len(self.cat_bit_slices) == 0:
            return cat_mask.new_zeros((cat_mask.size(0), 0))

        columns = []
        for start, end in self.cat_bit_slices:
            columns.append(cat_mask[:, start:end].amax(dim=1, keepdim=True))
        return torch.cat(columns, dim=1)

    def expand_cat_col_mask(self, cat_col_mask: torch.Tensor) -> torch.Tensor:
        if len(self.cat_bit_slices) == 0:
            return cat_col_mask.new_zeros((cat_col_mask.size(0), 0))

        expanded = []
        for column_idx, (start, end) in enumerate(self.cat_bit_slices):
            width = end - start
            expanded.append(cat_col_mask[:, column_idx : column_idx + 1].expand(-1, width))
        return torch.cat(expanded, dim=1)

    def collapse_feature_mask_to_groups(self, feature_mask: torch.Tensor) -> torch.Tensor:
        if len(self.feature_group_slices) == 0:
            return feature_mask.new_zeros((feature_mask.size(0), 0))

        groups = []
        for start, end in self.feature_group_slices:
            groups.append(feature_mask[:, start:end].amax(dim=1, keepdim=True))
        return torch.cat(groups, dim=1)

    def expand_group_mask(self, group_mask: torch.Tensor) -> torch.Tensor:
        if len(self.feature_group_slices) == 0:
            return group_mask.new_zeros((group_mask.size(0), 0))

        expanded = []
        for group_idx, (start, end) in enumerate(self.feature_group_slices):
            width = end - start
            expanded.append(group_mask[:, group_idx : group_idx + 1].expand(-1, width))
        return torch.cat(expanded, dim=1)

    def _build_pattern_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate training-only missingness priors for guided remasking."""
        missing = self.collapse_feature_mask_to_groups(self.train_split.missing_mask_ext.float())
        feature_missing_prior = missing.mean(dim=0)
        co_missing = missing.transpose(0, 1) @ missing
        co_missing.fill_diagonal_(0.0)

        row_sum = co_missing.sum(dim=1, keepdim=True)
        normalized = co_missing / row_sum.clamp_min(1e-6)
        fallback = feature_missing_prior.unsqueeze(0).expand_as(normalized)
        normalized = torch.where(row_sum > 0, normalized, fallback)
        return feature_missing_prior.float(), normalized.float()

    def get_split(self, split_name: str) -> SplitData:
        if split_name == "train":
            return self.train_split
        if split_name == "valid":
            return self.valid_split
        if split_name == "test":
            return self.test_split
        raise ValueError(f"Unsupported split name: {split_name}")

    def split_sizes(self) -> dict[str, int]:
        return {
            "train": self.train_split.num_samples,
            "valid": self.valid_split.num_samples,
            "test": self.test_split.num_samples,
        }


class DeterministicBatchPlanner:
    """Recreate each epoch's batch order from only the seed and epoch."""

    def __init__(self, num_samples: int, batch_size: int, seed: int, shuffle: bool = True) -> None:
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)

    @property
    def num_batches(self) -> int:
        return int(math.ceil(self.num_samples / max(1, self.batch_size)))

    def build_epoch_indices(self, epoch: int) -> list[torch.Tensor]:
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            indices = torch.randperm(self.num_samples, generator=generator)
        else:
            indices = torch.arange(self.num_samples)
        return list(indices.split(self.batch_size))


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True)
    return moved


def build_batch(bundle: BenchmarkMNARBundle, split: SplitData, indices: torch.Tensor) -> dict[str, torch.Tensor]:
    model_x = split.model_x[indices]
    observed_x = split.observed_x[indices]
    missing_mask = split.missing_mask_ext[indices].float()
    observed_mask = 1.0 - missing_mask
    missing_cat_mask = missing_mask[:, bundle.num_dim :]
    observed_cat_mask = observed_mask[:, bundle.num_dim :]

    if split.cat_idx is None:
        target_cat_idx = torch.empty((indices.numel(), 0), dtype=torch.long)
    else:
        target_cat_idx = torch.from_numpy(split.cat_idx[indices.cpu().numpy()]).long()

    batch = {
        "indices": indices.long(),
        "target_num": model_x[:, : bundle.num_dim].float(),
        "target_cat": model_x[:, bundle.num_dim :].float(),
        "target_cat_idx": target_cat_idx,
        "input_x": observed_x.float(),
        "input_num": observed_x[:, : bundle.num_dim].float(),
        "input_cat": observed_x[:, bundle.num_dim :].float(),
        "missing_mask": missing_mask.float(),
        "missing_num_mask": missing_mask[:, : bundle.num_dim].float(),
        "missing_cat_mask": missing_cat_mask.float(),
        "missing_cat_col_mask": bundle.collapse_cat_bit_mask(missing_cat_mask).float(),
        "observed_mask": observed_mask.float(),
        "observed_num_mask": observed_mask[:, : bundle.num_dim].float(),
        "observed_cat_mask": observed_cat_mask.float(),
        "observed_cat_col_mask": bundle.collapse_cat_bit_mask(observed_cat_mask).float(),
        "raw_x": split.raw_x[indices].float(),
    }
    return batch
