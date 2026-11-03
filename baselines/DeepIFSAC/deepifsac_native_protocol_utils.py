from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset


def _extend_mask(mask: np.ndarray, bin_num: np.ndarray) -> np.ndarray:
    num_rows, _ = mask.shape
    cum_sum = bin_num.cumsum()
    cum_sum = np.insert(cum_sum, 0, 0)
    result = np.zeros((num_rows, int(bin_num.sum())), dtype=bool)
    for idx in range(len(bin_num)):
        result[:, cum_sum[idx] : cum_sum[idx + 1]] = np.tile(mask[:, idx][:, None], int(bin_num[idx]))
    return result


def _safe_mean(values: np.ndarray) -> float:
    return float("nan") if values.size == 0 else float(values.mean())


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


def _fit_deepifsac_numeric_standardizer(train_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if train_values.shape[1] == 0:
        return np.zeros(0, dtype=np.float32), np.ones(0, dtype=np.float32)
    mean = train_values.astype(np.float32).mean(axis=0).astype(np.float32)
    std = train_values.astype(np.float32).std(axis=0).astype(np.float32)
    std = np.where(np.isfinite(std), std, 1.0).astype(np.float32)
    std = np.where(std < 1e-6, 1e-6, std).astype(np.float32)
    return mean, std


@dataclass
class SameProtocolSplit:
    name: str
    model_x: torch.Tensor
    raw_x: torch.Tensor
    missing_mask_ext: torch.Tensor
    original_mask: torch.Tensor
    cat_idx: np.ndarray | None

    @property
    def num_samples(self) -> int:
        return int(self.model_x.shape[0])


class DeepIFSACSameProtocolDataset(Dataset):
    def __init__(self, split: SameProtocolSplit, num_dim: int) -> None:
        self.split = split
        self.num_dim = int(num_dim)
        self.cat_dim = int(split.model_x.shape[1] - self.num_dim)

    def __len__(self) -> int:
        return self.split.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        model_x = self.split.model_x[idx]
        missing_mask = self.split.missing_mask_ext[idx].bool()

        target_num = model_x[: self.num_dim].float()
        target_cat = model_x[self.num_dim :].long()

        input_num = target_num.clone()
        if self.num_dim > 0:
            input_num[missing_mask[: self.num_dim]] = 0.0

        input_cat = target_cat.clone()
        if self.cat_dim > 0:
            input_cat[missing_mask[self.num_dim :]] = 0

        cls = torch.zeros(1, dtype=torch.long)
        x_categ = torch.cat([cls, input_cat.long()], dim=0)

        cat_observed = (~missing_mask[self.num_dim :]).long()
        cat_mask = torch.cat([torch.ones(1, dtype=torch.long), cat_observed], dim=0)
        con_mask = (~missing_mask[: self.num_dim]).long()

        return {
            "x_categ": x_categ.long(),
            "x_cont": input_num.float(),
            "cat_mask": cat_mask.long(),
            "con_mask": con_mask.long(),
            "target_num": target_num.float(),
            "target_cat": target_cat.long(),
            "missing_num_mask": missing_mask[: self.num_dim].bool(),
            "missing_cat_mask": missing_mask[self.num_dim :].bool(),
        }


class DeepIFSACBenchmarkBundle:
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
        self.normalized_path = self.normalized_dir / f"split_{split_idx}.npz"

        with open(self.info_path, "r", encoding="utf-8") as file:
            self.info = json.load(file)

        self.num_col_idx = list(self.info["num_col_idx"])
        self.cat_col_idx = list(self.info["cat_col_idx"])
        self.num_dim = len(self.num_col_idx)

        split_arrays = self._load_normalized_splits()
        self.mean = torch.from_numpy(split_arrays["mean"]).float()
        self.std = torch.from_numpy(split_arrays["std"]).float()
        raw = self._load_raw_eval_data()
        self.cat_dim = len(self.cat_col_idx)
        self.cat_cardinalities = torch.from_numpy(raw["cat_cardinalities"].astype(np.int64)).long()
        self.categorical_preprocessing = {
            "adapter": "deepifsac_native_label_tokens",
            "input_encoding": "per-categorical-column integer label ids from *_map_idx.json",
            "loss": "per-categorical-column cross_entropy on missing categorical entries",
            "decoding": "argmax over each categorical column head",
            "metric_level": "column_prefixed_category_index",
            "f1_average": "weighted_over_column_prefixed_category_indices",
            "uses_eunseo_codebook": False,
            "uses_eunseo_codebook_loss": False,
            "uses_eunseo_codebook_decode": False,
        }

        full_train_raw = torch.from_numpy(raw["train_x_raw"]).float()
        test_raw = torch.from_numpy(raw["test_x_raw"]).float()
        full_train_original_mask = torch.from_numpy(raw["train_mask_original"].astype(np.bool_))
        test_original_mask = torch.from_numpy(raw["test_mask_original"].astype(np.bool_))
        full_train_cat_idx = raw["train_cat_idx"]
        test_cat_idx = raw["test_cat_idx"]

        train_indices = split_arrays["train_indices"]
        valid_indices = split_arrays["valid_indices"]
        self.train_indices = torch.from_numpy(train_indices.astype(np.int64))
        self.valid_indices = torch.from_numpy(valid_indices.astype(np.int64))

        train_raw = full_train_raw[self.train_indices]
        valid_raw = full_train_raw[self.valid_indices]
        train_original_mask = full_train_original_mask[self.train_indices]
        valid_original_mask = full_train_original_mask[self.valid_indices]
        train_cat_idx = None if full_train_cat_idx is None else full_train_cat_idx[train_indices]
        valid_cat_idx = None if full_train_cat_idx is None else full_train_cat_idx[valid_indices]

        model_numeric_mean_np, model_numeric_std_np = _fit_deepifsac_numeric_standardizer(
            train_raw[:, : self.num_dim].cpu().numpy()
        )
        self.model_numeric_mean = torch.from_numpy(model_numeric_mean_np).float()
        self.model_numeric_std = torch.from_numpy(model_numeric_std_np).float()

        self.numeric_preprocessing = {
            "numeric_source": "raw_csv_numeric_with_deepifsac_native_train_zscore_model_input",
            "model_numeric_source": "raw_csv_numeric",
            "model_numeric_preprocessing": "deepifsac_native_train_split_zscore",
            "model_numeric_standardizer_fit": "train_split_complete_numeric_values_only",
            "metric_scaler_source": "prepared_npz_mean_std",
            "scaler_source": "model_input_train_zscore_and_metric_prepared_npz_mean_std",
            "metric_space": "eunseo_normalized",
            "categorical_protocol": "deepifsac_native_label_embedding_and_category_index_metric",
            "standardization": "deepifsac_native_for_model_input_prepared_npz_for_metric",
            "model_mean_source": "raw train split numeric mean",
            "model_std_source": "raw train split numeric std",
            "metric_mean_source": "prepared npz mean",
            "metric_std_source": "prepared npz std",
            "model_scale_formula": "(x_raw - model_train_mean) / model_train_std",
            "model_inverse_formula": "x_model * model_train_std + model_train_mean",
            "normalization_formula_for_metrics": "(x_raw - prepared_npz_mean) / prepared_npz_std / 2.0",
        }

        self.train_split = self._make_split(
            name="train",
            x_norm=self._raw_numeric_to_model_space(train_raw),
            raw_x=train_raw,
            missing_mask=train_original_mask[:, self.num_col_idx + self.cat_col_idx],
            original_mask=train_original_mask,
            cat_idx=train_cat_idx,
        )
        self.valid_split = self._make_split(
            name="valid",
            x_norm=self._raw_numeric_to_model_space(valid_raw),
            raw_x=valid_raw,
            missing_mask=valid_original_mask[:, self.num_col_idx + self.cat_col_idx],
            original_mask=valid_original_mask,
            cat_idx=valid_cat_idx,
        )
        self.test_split = self._make_split(
            name="test",
            x_norm=self._raw_numeric_to_model_space(test_raw),
            raw_x=test_raw,
            missing_mask=test_original_mask[:, self.num_col_idx + self.cat_col_idx],
            original_mask=test_original_mask,
            cat_idx=test_cat_idx,
        )
        self._verify_numeric_inputs_against_npz(
            split_arrays["train_x_norm"][:, : self.num_dim],
            split_arrays["valid_x_norm"][:, : self.num_dim],
            split_arrays["test_x_norm"][:, : self.num_dim],
        )
        self.numeric_preprocessing["numeric_input_verification"] = self.numeric_input_verification

    def _load_normalized_splits(self) -> dict[str, np.ndarray]:
        if not self.normalized_path.exists():
            return self._load_legacy_split_files()

        normalized = np.load(self.normalized_path)
        files = set(normalized.files)
        strict_keys = {
            "train_x_norm",
            "valid_x_norm",
            "test_x_norm",
            "train_mask",
            "valid_mask",
            "test_mask",
            "train_indices",
            "valid_indices",
        }
        if strict_keys.issubset(files):
            return {
                "train_x_norm": normalized["train_x_norm"],
                "valid_x_norm": normalized["valid_x_norm"],
                "test_x_norm": normalized["test_x_norm"],
                "train_mask": normalized["train_mask"],
                "valid_mask": normalized["valid_mask"],
                "test_mask": normalized["test_mask"],
                "train_indices": normalized["train_indices"].astype(np.int64),
                "valid_indices": normalized["valid_indices"].astype(np.int64),
                "mean": normalized["mean"],
                "std": normalized["std"],
            }

        return self._load_legacy_split_files()

    def _load_legacy_split_files(self) -> dict[str, np.ndarray]:
        train_npz = np.load(self.normalized_dir / f"split_{self.split_idx}_train.npz")
        valid_npz = np.load(self.normalized_dir / f"split_{self.split_idx}_valid.npz")
        test_npz = np.load(self.normalized_dir / f"split_{self.split_idx}_test.npz")

        full_train_size = int(train_npz["indices"].shape[0] + valid_npz["indices"].shape[0])
        train_indices, valid_indices = _build_train_valid_indices(
            train_size=full_train_size,
            test_size=int(test_npz["x_norm"].shape[0]),
            seed=self.seed,
            valid_max_fraction=self.valid_max_fraction,
        )
        if train_npz["indices"].shape[0] == train_indices.shape[0]:
            train_indices = train_npz["indices"].astype(np.int64)
        if valid_npz["indices"].shape[0] == valid_indices.shape[0]:
            valid_indices = valid_npz["indices"].astype(np.int64)

        return {
            "train_x_norm": train_npz["x_norm"],
            "valid_x_norm": valid_npz["x_norm"],
            "test_x_norm": test_npz["x_norm"],
            "train_mask": train_npz["mask"],
            "valid_mask": valid_npz["mask"],
            "test_mask": test_npz["mask"],
            "train_indices": train_indices,
            "valid_indices": valid_indices,
            "mean": train_npz["mean"],
            "std": train_npz["std"],
        }

    def _load_raw_eval_data(self) -> dict[str, Any]:
        train_df = pd.read_csv(self.dataset_dir / "train.csv")
        test_df = pd.read_csv(self.dataset_dir / "test.csv")

        train_mask_original = np.load(
            self.dataset_dir / "masks" / f"rate{self.ratio}" / self.mask_type / f"train_mask_{self.split_idx}.npy"
        )
        test_mask_original = np.load(
            self.dataset_dir / "masks" / f"rate{self.ratio}" / self.mask_type / f"test_mask_{self.split_idx}.npy"
        )
        train_mask_original = train_mask_original.astype(bool)
        test_mask_original = test_mask_original.astype(bool)

        columns = train_df.columns
        train_num = train_df[columns[self.num_col_idx]].values.astype(np.float32)
        test_num = test_df[columns[self.num_col_idx]].values.astype(np.float32)

        if len(self.cat_col_idx) == 0:
            train_x_raw = train_num
            test_x_raw = test_num
            train_cat_idx = None
            test_cat_idx = None
            cat_cardinalities = np.zeros(0, dtype=np.int64)
        else:
            cat_columns = columns[self.cat_col_idx]
            train_cat = train_df[cat_columns].astype(str)
            test_cat = test_df[cat_columns].astype(str)

            train_cat_idx = []
            test_cat_idx = []
            cat_cardinalities = []

            for column in cat_columns:
                with open(self.dataset_dir / f"{column}_map_idx.json", "r", encoding="utf-8") as file:
                    category_to_idx = json.load(file)

                train_cat_idx_i = train_cat[column].map(category_to_idx).to_numpy().astype(np.int64)
                test_cat_idx_i = test_cat[column].map(category_to_idx).to_numpy().astype(np.int64)

                train_cat_idx.append(train_cat_idx_i)
                test_cat_idx.append(test_cat_idx_i)
                cat_cardinalities.append(len(category_to_idx))

            train_cat_idx = np.stack(train_cat_idx, axis=1).astype(np.int64)
            test_cat_idx = np.stack(test_cat_idx, axis=1).astype(np.int64)
            train_x_raw = np.concatenate([train_num, train_cat_idx.astype(np.float32)], axis=1)
            test_x_raw = np.concatenate([test_num, test_cat_idx.astype(np.float32)], axis=1)
            cat_cardinalities = np.asarray(cat_cardinalities, dtype=np.int64)

        return {
            "train_x_raw": train_x_raw.astype(np.float32),
            "test_x_raw": test_x_raw.astype(np.float32),
            "train_mask_original": train_mask_original,
            "test_mask_original": test_mask_original,
            "train_cat_idx": train_cat_idx,
            "test_cat_idx": test_cat_idx,
            "cat_cardinalities": cat_cardinalities,
        }

    def _raw_numeric_to_model_space(self, raw_x: torch.Tensor) -> torch.Tensor:
        if self.num_dim == 0:
            return raw_x.new_zeros((raw_x.shape[0], 0))
        raw_num = raw_x[:, : self.num_dim].cpu().numpy().astype(np.float32)
        mean = self.model_numeric_mean.cpu().numpy()[None, :]
        std = self.model_numeric_std.cpu().numpy()[None, :]
        model_num = (raw_num - mean) / std
        return torch.from_numpy(model_num.astype(np.float32))

    def _make_split(
        self,
        name: str,
        x_norm: torch.Tensor,
        raw_x: torch.Tensor,
        missing_mask: torch.Tensor,
        original_mask: torch.Tensor,
        cat_idx: np.ndarray | None,
    ) -> SameProtocolSplit:
        if self.num_dim > 0:
            num_norm = x_norm[:, : self.num_dim].float().clone()
        else:
            num_norm = raw_x.new_zeros((raw_x.shape[0], 0))

        if self.cat_dim > 0:
            model_x = torch.cat([num_norm, raw_x[:, self.num_dim :].long().float()], dim=1)
        else:
            model_x = num_norm
        return SameProtocolSplit(
            name=name,
            model_x=model_x.float(),
            raw_x=raw_x.float(),
            missing_mask_ext=missing_mask.bool(),
            original_mask=original_mask.bool(),
            cat_idx=cat_idx,
        )

    def _verify_numeric_inputs_against_npz(
        self,
        train_num_norm: np.ndarray,
        valid_num_norm: np.ndarray,
        test_num_norm: np.ndarray,
    ) -> None:
        if self.num_dim == 0:
            self.numeric_input_verification = {
                "numeric_model_input_protocol": "no_numeric_columns",
                "numeric_input_first_num_dim_matches_npz": True,
                "numeric_input_max_abs_diff": 0.0,
            }
            return
        checks = {
            "train": (self.train_split.model_x[:, : self.num_dim].cpu().numpy(), train_num_norm),
            "valid": (self.valid_split.model_x[:, : self.num_dim].cpu().numpy(), valid_num_norm),
            "test": (self.test_split.model_x[:, : self.num_dim].cpu().numpy(), test_num_norm),
        }
        diffs = {name: float(np.max(np.abs(actual - expected))) for name, (actual, expected) in checks.items()}
        max_diff = max(diffs.values()) if diffs else 0.0
        self.numeric_input_verification = {
            "numeric_model_input_protocol": "deepifsac_native_train_zscore",
            "numeric_input_first_num_dim_matches_npz": False,
            "numeric_input_max_abs_diff": max_diff,
            "numeric_input_max_abs_diff_by_split": diffs,
            "prepared_npz_x_norm_used_as_model_input": False,
            "prepared_npz_mean_std_used_for_metric": True,
            "raw_csv_numeric_scaler_fit": True,
            "model_numeric_mean": self.model_numeric_mean.cpu().numpy(),
            "model_numeric_std": self.model_numeric_std.cpu().numpy(),
        }

    def get_split(self, split_name: str) -> SameProtocolSplit:
        if split_name == "train":
            return self.train_split
        if split_name == "valid":
            return self.valid_split
        if split_name == "test":
            return self.test_split
        raise ValueError(f"Unsupported split: {split_name}")

    def split_sizes(self) -> dict[str, int]:
        return {
            "train": self.train_split.num_samples,
            "valid": self.valid_split.num_samples,
            "test": self.test_split.num_samples,
        }


def compose_raw_prediction(
    bundle: DeepIFSACBenchmarkBundle,
    split_name: str,
    pred_num_norm: np.ndarray,
    pred_cat_idx: np.ndarray,
) -> np.ndarray:
    split = bundle.get_split(split_name)
    pred_raw = split.raw_x.cpu().numpy().copy()
    missing = split.missing_mask_ext.cpu().numpy()

    if bundle.num_dim > 0:
        mean = bundle.model_numeric_mean.cpu().numpy()[None, :]
        std = bundle.model_numeric_std.cpu().numpy()[None, :]
        pred_num_raw = pred_num_norm * std + mean
        num_mask = missing[:, : bundle.num_dim]
        pred_raw[:, : bundle.num_dim][num_mask] = pred_num_raw[num_mask]

    if bundle.cat_dim > 0:
        cat_mask = missing[:, bundle.num_dim :]
        pred_raw[:, bundle.num_dim :][cat_mask] = pred_cat_idx[cat_mask]

    return pred_raw.astype(np.float32)


def evaluate_predictions(bundle: DeepIFSACBenchmarkBundle, pred_raw: np.ndarray, split_name: str) -> dict[str, float]:
    split = bundle.get_split(split_name)
    return evaluate_arrays(
        pred_raw=pred_raw,
        true_raw=split.raw_x.cpu().numpy(),
        true_norm=split.model_x.cpu().numpy(),
        mask_original=split.original_mask.cpu().numpy(),
        num_col_idx=bundle.num_col_idx,
        cat_col_idx=bundle.cat_col_idx,
        num_dim=bundle.num_dim,
        truth_cat_idx=split.cat_idx,
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
    truth_cat_idx: np.ndarray | None,
    mean_num: np.ndarray,
    std_num: np.ndarray,
) -> dict[str, float]:
    num_mask = mask_original[:, num_col_idx] if len(num_col_idx) > 0 else np.zeros((mask_original.shape[0], 0), dtype=bool)
    if num_dim > 0 and num_mask.sum() > 0:
        pred_num_norm = (pred_raw[:, :num_dim] - mean_num[None, :]) / (std_num[None, :] * 2.0)
        true_num_norm = (true_raw[:, :num_dim] - mean_num[None, :]) / (std_num[None, :] * 2.0)
        num_diff = pred_num_norm[num_mask] - true_num_norm[num_mask]
        mae = _safe_mean(np.abs(num_diff))
        mse = _safe_mean(num_diff**2)
        rmse = float(np.sqrt(mse)) if not np.isnan(mse) else float("nan")

        raw_diff = pred_raw[:, :num_dim][num_mask] - true_raw[:, :num_dim][num_mask]
        mae_raw = _safe_mean(np.abs(raw_diff))
        mse_raw = _safe_mean(raw_diff**2)
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
    if truth_cat_idx is not None and len(cat_col_idx) > 0:
        pred_cat_idx = np.rint(pred_raw[:, num_dim:]).astype(np.int64)
        cat_mask = mask_original[:, cat_col_idx]
        if cat_mask.sum() > 0:
            cat_accuracy = float((pred_cat_idx[cat_mask] == truth_cat_idx[cat_mask]).mean())
            accuracy = cat_accuracy
            true_tokens = []
            pred_tokens = []
            for col_pos, original_col in enumerate(cat_col_idx):
                col_mask = cat_mask[:, col_pos]
                if col_mask.sum() > 0:
                    true_values = truth_cat_idx[:, col_pos][col_mask]
                    pred_values = pred_cat_idx[:, col_pos][col_mask]
                    true_tokens.extend(
                        [f"col{int(original_col)}=idx{int(value)}" for value in true_values.tolist()]
                    )
                    pred_tokens.extend(
                        [f"col{int(original_col)}=idx{int(value)}" for value in pred_values.tolist()]
                    )
            if true_tokens:
                cat_f1_macro = float(f1_score(true_tokens, pred_tokens, average="macro", zero_division=0))
                cat_f1_weighted = float(f1_score(true_tokens, pred_tokens, average="weighted", zero_division=0))
                cat_f1 = cat_f1_weighted

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mae_raw": mae_raw,
        "mse_raw": mse_raw,
        "rmse_raw": rmse_raw,
        "numeric_source": "raw_csv_numeric_with_deepifsac_native_train_zscore_model_input",
        "scaler_source": "model_input_train_zscore_and_metric_prepared_npz_mean_std",
        "metric_space": "eunseo_normalized",
        "categorical_protocol": "deepifsac_native_label_embedding_and_category_index_metric",
        "cat_f1": cat_f1,
        "cat_f1_macro": cat_f1_macro,
        "cat_f1_weighted": cat_f1_weighted,
        "f1": cat_f1_weighted,
        "bit_f1": cat_f1_weighted,
        "cat_accuracy": cat_accuracy,
        "accuracy": accuracy,
        "MAE": mae,
        "RMSE": rmse,
        "F1": cat_f1_weighted,
        "Accuracy": accuracy,
        "numeric_metric_unit": "prepared_normalized_unit",
        "numeric_metric_formula": "((pred_raw - prepared_npz_mean) / prepared_npz_std / 2.0) - ((true_raw - prepared_npz_mean) / prepared_npz_std / 2.0)",
        "numeric_metric_scale": "eunseo_x_norm_formula_(x-mean)/std/2.0",
        "categorical_metric_level": "column_prefixed_category_index",
        "cat_f1_average": "weighted_over_column_prefixed_category_indices",
        "categorical_f1_average": "weighted_over_column_prefixed_category_indices",
        "categorical_token_format": "col{original_column_index}=idx{category_index}",
    }


def score_from_metrics(metrics: dict[str, float]) -> float:
    rmse = metrics.get("rmse", float("nan"))
    cat_f1 = metrics.get("cat_f1_weighted", metrics.get("cat_f1", metrics.get("bit_f1", float("nan"))))
    cat_acc = metrics.get("cat_accuracy", metrics.get("accuracy", float("nan")))
    rmse_term = 1e6 if rmse != rmse else float(rmse)
    f1_term = 0.0 if cat_f1 != cat_f1 else float(cat_f1)
    acc_term = 0.0 if cat_acc != cat_acc else float(cat_acc)
    return rmse_term - 0.1 * f1_term - 0.05 * acc_term


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_sanitize(value.tolist())
    if isinstance(value, torch.Tensor):
        return json_sanitize(value.detach().cpu().tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(json_sanitize(payload), file, indent=2)
