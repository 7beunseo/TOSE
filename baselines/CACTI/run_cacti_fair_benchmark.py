from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from fair_protocol_dataset import BenchmarkMNARBundle, SplitData
from src.loaders.embedcopyloader import EmbedCopyMaskedDataset
from src.models.cacti import CACTImodel
from src.modules.scheduler import StepWiseWarmupCosineAnnealingScheduler


DATASETS = [
    "adult", "bean", "default", "gesture", "letter", "magic", "news", "shoppers",
    "abalone", "bank", "breast", "california", "german", "mushroom",
]
OFFICIAL_REPOSITORY = "https://github.com/sriramlab/CACTI"
OFFICIAL_SOURCE_ROOT = Path(
    os.environ.get(
        "CACTI_SOURCE_ROOT",
        Path(__file__).resolve().parents[2] / ".baseline_sources" / "cacti",
    )
).resolve()
SELECTION_PROTOCOL = "baseline_normalized"
SELECTION_FORMULA = (
    "weighted_mean(rmse/reference_mean_rmse, "
    "(1-f1)/(1-reference_mode_f1), "
    "(1-accuracy)/(1-reference_mode_accuracy))"
)
SELECTION_NUM_WEIGHT = 0.50
SELECTION_F1_WEIGHT = 0.25
SELECTION_ACCURACY_WEIGHT = 0.25
SELECTION_EPSILON = 1e-6


@dataclass
class LogicalSplit:
    values: np.ndarray
    observed: np.ndarray
    missing: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official CACTI under the shared strict protocol.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--splits", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--mask-type", required=True)
    parser.add_argument("--ratio", default="30")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--run-tag", default="cacti_official_mtcm_gte_e600_es")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--early-stop-patience", type=int, default=45)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0002)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mask-ratio", type=float, default=0.9)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--nencoder", type=int, default=10)
    parser.add_argument("--ndecoder", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-last-checkpoint", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2, allow_nan=True)
        replace_with_retry(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def replace_with_retry(tmp_path: Path, path: Path, attempts: int = 30) -> None:
    """Tolerate short Windows file locks from scanners and indexing services."""
    last_error: OSError | None = None
    delay = 0.1
    for attempt in range(attempts):
        try:
            os.replace(tmp_path, path)
            return
        except OSError as exc:
            last_error = exc
            if attempt == 0 or (attempt + 1) % 5 == 0:
                print(
                    f"[io-retry] target={path} attempt={attempt + 1}/{attempts} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
            if attempt + 1 < attempts:
                time.sleep(delay)
                delay = min(delay * 1.5, 2.0)
    if last_error is not None:
        raise last_error


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        torch.save(payload, tmp_path)
        replace_with_retry(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def official_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_logical_split(bundle: BenchmarkMNARBundle, split_name: str) -> LogicalSplit:
    split = bundle.get_split(split_name)
    numeric = split.model_x[:, : bundle.num_dim].cpu().numpy().astype(np.float32)
    if split.cat_idx is None:
        categorical = np.zeros((split.num_samples, 0), dtype=np.float32)
    else:
        categorical = split.cat_idx.astype(np.float32)
    values = np.concatenate([numeric, categorical], axis=1)
    missing = bundle.collapse_feature_mask_to_groups(split.missing_mask_ext.float()).cpu().numpy().astype(bool)
    if values.shape != missing.shape:
        raise RuntimeError(f"Logical data/mask mismatch: values={values.shape}, mask={missing.shape}")
    observed = values.copy()
    observed[missing] = np.nan
    return LogicalSplit(values=values, observed=observed, missing=missing)


def fit_minmax(train_observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mins = np.nanmin(train_observed, axis=0).astype(np.float32)
    maxs = np.nanmax(train_observed, axis=0).astype(np.float32)
    if not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs)):
        raise RuntimeError("At least one feature has no observed training value.")
    return mins, maxs


def scale_values(values: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    return ((values - mins[None, :]) / (maxs - mins + 1e-6)[None, :]).astype(np.float32)


def unscale_values(values: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    return (values * (maxs - mins + 1e-6)[None, :] + mins[None, :]).astype(np.float32)


def decode_categories(bundle: BenchmarkMNARBundle, predictions: np.ndarray) -> np.ndarray:
    decoded = predictions.copy()
    for local_idx, codebook in enumerate(bundle.cat_codebooks_np):
        column_idx = bundle.num_dim + local_idx
        decoded[:, column_idx] = np.clip(
            np.rint(decoded[:, column_idx]), 0, int(codebook.shape[0]) - 1
        )
    return decoded


def compute_metrics(
    bundle: BenchmarkMNARBundle,
    split_name: str,
    predictions: np.ndarray,
) -> dict[str, float | str]:
    split = bundle.get_split(split_name)
    true_norm = split.model_x[:, : bundle.num_dim].cpu().numpy()
    num_mask = split.missing_mask_ext[:, : bundle.num_dim].cpu().numpy().astype(bool)
    metrics: dict[str, float | str] = {}
    if bundle.num_dim > 0 and num_mask.any():
        diff = predictions[:, : bundle.num_dim][num_mask] - true_norm[num_mask]
        mse = float(np.mean(diff**2))
        metrics.update(mse=mse, rmse=float(np.sqrt(mse)), mae=float(np.mean(np.abs(diff))))
    else:
        metrics.update(mse=float("nan"), rmse=float("nan"), mae=float("nan"))

    if split.cat_idx is not None and len(bundle.cat_col_idx) > 0:
        cat_mask = split.original_mask[:, bundle.cat_col_idx].cpu().numpy().astype(bool)
        pred_idx = predictions[:, bundle.num_dim :].astype(np.int64)
        if cat_mask.any():
            true_tokens: list[str] = []
            pred_tokens: list[str] = []
            for local_idx, original_idx in enumerate(bundle.cat_col_idx):
                rows = cat_mask[:, local_idx]
                if not rows.any():
                    continue
                true_values = split.cat_idx[:, local_idx][rows].astype(np.int64)
                pred_values = pred_idx[:, local_idx][rows]
                true_tokens.extend(f"col{original_idx}=idx{value}" for value in true_values.tolist())
                pred_tokens.extend(f"col{original_idx}=idx{value}" for value in pred_values.tolist())
            weighted_f1 = float(f1_score(true_tokens, pred_tokens, average="weighted", zero_division=0))
            macro_f1 = float(f1_score(true_tokens, pred_tokens, average="macro", zero_division=0))
            accuracy = float((pred_idx[cat_mask] == split.cat_idx[cat_mask]).mean())
            metrics.update(
                cat_f1=weighted_f1,
                cat_f1_weighted=weighted_f1,
                cat_f1_macro=macro_f1,
                f1=weighted_f1,
                bit_f1=weighted_f1,
                cat_accuracy=accuracy,
                accuracy=accuracy,
                invalid_code_rate=0.0,
            )
        else:
            metrics.update(
                cat_f1=float("nan"), cat_f1_weighted=float("nan"), cat_f1_macro=float("nan"),
                f1=float("nan"), bit_f1=float("nan"), cat_accuracy=float("nan"),
                accuracy=float("nan"), invalid_code_rate=float("nan"),
            )
    else:
        metrics.update(
            cat_f1=float("nan"), cat_f1_weighted=float("nan"), cat_f1_macro=float("nan"),
            f1=float("nan"), bit_f1=float("nan"), cat_accuracy=float("nan"),
            accuracy=float("nan"), invalid_code_rate=float("nan"),
        )

    metrics.update(
        MAE=metrics["mae"], RMSE=metrics["rmse"], F1=metrics["cat_f1"],
        Accuracy=metrics["cat_accuracy"], numeric_source="prepared_npz_x_norm",
        scaler_source="CACTI_train_only_minmax_over_prepared_numeric_and_ordinal_categorical",
        metric_space="eunseo_normalized", numeric_metric_unit="prepared_normalized_unit",
        numeric_metric_formula="pred_num_norm - true_x_norm",
        categorical_protocol="CACTI_native_scalar_ordinal_prediction_rounded_to_valid_category",
        categorical_metric_level="column_prefixed_category_index",
        categorical_f1_average="weighted_over_column_prefixed_category_indices",
    )
    return metrics


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))


def compute_selection_reference(bundle: BenchmarkMNARBundle) -> dict[str, Any]:
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
            difference = prediction[valid_missing] - valid_values[valid_missing]
            reference["numeric_rmse"] = float(np.sqrt(np.mean(difference**2)))
        reference["numeric_train_mean"] = means.tolist()

    if (
        bundle.train_split.cat_idx is not None
        and bundle.valid_split.cat_idx is not None
        and len(bundle.cat_col_idx) > 0
    ):
        train_cat_idx = bundle.train_split.cat_idx
        valid_cat_idx = bundle.valid_split.cat_idx
        train_mask = bundle.train_split.original_mask.cpu().numpy().astype(bool)
        valid_mask = bundle.valid_split.original_mask.cpu().numpy().astype(bool)
        modes: list[int] = []
        true_tokens: list[str] = []
        pred_tokens: list[str] = []
        correct = 0
        total = 0
        for local_idx, original_idx in enumerate(bundle.cat_col_idx):
            observed = ~train_mask[:, int(original_idx)]
            observed_values = train_cat_idx[observed, local_idx].astype(np.int64)
            cardinality = int(bundle.cat_codebooks_np[local_idx].shape[0])
            mode = 0 if observed_values.size == 0 else int(
                np.argmax(np.bincount(observed_values, minlength=cardinality))
            )
            modes.append(mode)
            missing = valid_mask[:, int(original_idx)]
            if not missing.any():
                continue
            true_values = valid_cat_idx[missing, local_idx].astype(np.int64)
            pred_values = np.full(true_values.shape, mode, dtype=np.int64)
            true_tokens.extend(f"col{int(original_idx)}=idx{int(value)}" for value in true_values.tolist())
            pred_tokens.extend(f"col{int(original_idx)}=idx{int(value)}" for value in pred_values.tolist())
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


def selection_score(
    metrics: dict[str, Any], reference: dict[str, Any]
) -> tuple[float, dict[str, float]]:
    rmse = metrics.get("rmse", metrics.get("RMSE", float("nan")))
    f1 = metrics.get("cat_f1_weighted", metrics.get("cat_f1", metrics.get("F1", float("nan"))))
    accuracy = metrics.get("cat_accuracy", metrics.get("Accuracy", float("nan")))
    terms: list[tuple[str, float, float]] = []
    if finite_number(rmse) and finite_number(reference["numeric_rmse"]):
        terms.append(
            (
                "numeric_relative_rmse",
                SELECTION_NUM_WEIGHT,
                float(rmse) / max(float(reference["numeric_rmse"]), SELECTION_EPSILON),
            )
        )
    if finite_number(f1) and finite_number(reference["categorical_f1_weighted"]):
        terms.append(
            (
                "categorical_relative_f1_error",
                SELECTION_F1_WEIGHT,
                (1.0 - float(f1))
                / max(1.0 - float(reference["categorical_f1_weighted"]), SELECTION_EPSILON),
            )
        )
    if finite_number(accuracy) and finite_number(reference["categorical_accuracy"]):
        terms.append(
            (
                "categorical_relative_accuracy_error",
                SELECTION_ACCURACY_WEIGHT,
                (1.0 - float(accuracy))
                / max(1.0 - float(reference["categorical_accuracy"]), SELECTION_EPSILON),
            )
        )
    if not terms:
        return 1e6, {}
    weight_sum = sum(weight for _, weight, _ in terms)
    components = {name: float(value) for name, _, value in terms}
    components["available_weight_sum"] = float(weight_sum)
    score = sum(weight * value for _, weight, value in terms) / max(weight_sum, SELECTION_EPSILON)
    return float(score), components


def forward_encoder_eval_batched(
    model: CACTImodel,
    table: torch.Tensor,
    missing_mask: torch.Tensor,
    info_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorize upstream batch-size-1 inference without changing its mathematics."""
    token_embeddings = model.tab_embed(table)
    context_embeddings = model.info_embed_enc(info_embeddings.to(torch.float32))
    encoded = torch.cat((token_embeddings, context_embeddings), dim=2)
    encoded = encoded + model.pos_embed[:, 1:, :]

    batch_size, table_len, embedding_dim = encoded.shape
    num_observed = (1 - missing_mask.to(torch.int32)).sum(dim=1)
    if not torch.all(num_observed == num_observed[0]):
        raise RuntimeError("CACTI inference batches must have an equal observed-feature count.")
    observed_len = int(num_observed[0])

    threshold = 0.99
    interval = threshold / table_len
    ordered_noise = torch.arange(0, threshold, interval, device=encoded.device)
    if ordered_noise.numel() != table_len:
        ordered_noise = ordered_noise[:table_len]
    # Upstream uses expand() and then writes into the view. clone() preserves each
    # row's native batch-size-1 result while avoiding cross-row storage aliasing.
    noise = ordered_noise.unsqueeze(0).expand(batch_size, -1).clone()
    noise[missing_mask > threshold] = 1.0
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :observed_len]

    masked = encoded.clone()
    masked[missing_mask > threshold] = model.pad_token
    masked = torch.gather(
        masked,
        dim=1,
        index=ids_keep.unsqueeze(-1).repeat(1, 1, embedding_dim),
    )
    observed_mask = torch.ones((batch_size, table_len), device=encoded.device)
    observed_mask[:, observed_len:] = 0
    observed_mask = torch.gather(observed_mask, dim=1, index=ids_restore)
    observed_mask *= 1 - missing_mask.to(observed_mask.dtype)

    cls_token = (model.cls_token + model.pos_embed[:, :1, :]).expand(batch_size, -1, -1)
    encoded = torch.cat((cls_token, masked), dim=1)
    for block in model.encode_blocks:
        encoded = block(encoded)
    return model.norm(encoded), ids_restore, observed_mask


def predict_scaled(
    model: CACTImodel,
    observed_scaled: np.ndarray,
    info_embeddings: torch.Tensor,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
    label: str,
) -> np.ndarray:
    missing = np.isnan(observed_scaled)
    observed_counts = (~missing).sum(axis=1)
    output = np.empty_like(observed_scaled, dtype=np.float32)
    model.eval()

    with torch.inference_mode():
        progress = tqdm(total=len(observed_scaled), desc=label, unit="rows")
        for count in np.unique(observed_counts):
            group_indices = np.flatnonzero(observed_counts == count)
            for start in range(0, len(group_indices), batch_size):
                indices = group_indices[start : start + batch_size]
                batch_values = np.nan_to_num(observed_scaled[indices], nan=-1.0)
                table = torch.from_numpy(batch_values).to(device).unsqueeze(1)
                miss_map = torch.from_numpy(missing[indices]).to(device)
                contexts = info_embeddings.unsqueeze(0).expand(len(indices), -1, -1)
                # Upstream CACTI transform() runs inference in FP32 even when fit()
                # uses AMP. Keep validation and test on that native precision path.
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    latent, restore_map, obs_mask = forward_encoder_eval_batched(
                        model, table, miss_map, contexts
                    )
                    decoded = model.forward_decoder(latent, restore_map, obs_mask, contexts)
                    predicted = model.decode_predictor(decoded).squeeze(1)
                output[indices] = predicted.float().cpu().numpy()
                progress.update(len(indices))
        progress.close()
    return output


def prediction_to_raw(
    bundle: BenchmarkMNARBundle,
    split_name: str,
    predictions: np.ndarray,
) -> np.ndarray:
    split = bundle.get_split(split_name)
    raw = split.raw_x.cpu().numpy().copy()
    missing_ext = split.missing_mask_ext.cpu().numpy().astype(bool)
    if bundle.num_dim > 0:
        pred_num_raw = predictions[:, : bundle.num_dim] * (
            bundle.std[: bundle.num_dim].cpu().numpy()[None, :] * 2.0
        ) + bundle.mean[: bundle.num_dim].cpu().numpy()[None, :]
        raw[:, : bundle.num_dim][missing_ext[:, : bundle.num_dim]] = pred_num_raw[
            missing_ext[:, : bundle.num_dim]
        ]
    bit_start = bundle.num_dim
    for local_idx, codebook in enumerate(bundle.cat_codebooks_np):
        width = int(codebook.shape[1])
        pred_idx = predictions[:, bundle.num_dim + local_idx].astype(np.int64)
        pred_bits = codebook[pred_idx]
        bit_mask = missing_ext[:, bit_start : bit_start + width]
        raw[:, bit_start : bit_start + width][bit_mask] = pred_bits[bit_mask]
        bit_start += width
    return raw.astype(np.float32)


def checkpoint_payload(
    model: CACTImodel,
    optimizer: torch.optim.Optimizer,
    scheduler: StepWiseWarmupCosineAnnealingScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_score: float | None,
    best_report: dict[str, Any] | None,
    no_improve: int,
    history: list[dict[str, Any]],
    mins: np.ndarray,
    maxs: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_score": best_score,
        "best_report": best_report,
        "no_improve": no_improve,
        "history": history,
        "mins": mins,
        "maxs": maxs,
        "config": config,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def load_resume(
    path: Path,
    model: CACTImodel,
    optimizer: torch.optim.Optimizer,
    scheduler: StepWiseWarmupCosineAnnealingScheduler,
    scaler: torch.amp.GradScaler,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_random_state") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
    return checkpoint


def run_single(args: argparse.Namespace, dataset: str, split_idx: int) -> dict[str, Any]:
    seed = int(args.seed) + int(split_idx)
    run_dir = (
        args.output_root / dataset / f"rate{args.ratio}" / args.mask_type /
        f"split_{split_idx}" / args.run_tag
    )
    report_path = run_dir / "report.json"
    if report_path.exists() and not args.overwrite:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("best_report", {}).get("test_metrics"):
                print(f"[skip] dataset={dataset} split={split_idx} report={report_path}", flush=True)
                return report
        except (OSError, json.JSONDecodeError):
            pass

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    history_path = run_dir / "history.json"

    set_seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    use_amp = bool(args.amp and device.type == "cuda")
    bundle = BenchmarkMNARBundle(
        dataset_name=dataset,
        mask_type=args.mask_type,
        split_idx=split_idx,
        ratio=args.ratio,
        data_root=args.data_root,
        seed=seed,
    )
    logical = {name: build_logical_split(bundle, name) for name in ("train", "valid", "test")}
    selection_reference = compute_selection_reference(bundle)
    mins, maxs = fit_minmax(logical["train"].observed)
    scaled = {name: scale_values(split.observed, mins, maxs) for name, split in logical.items()}

    embedding_path = args.embedding_root / dataset / "embeddings_colnames_GTEenMLMlarge.npz"
    if not embedding_path.exists():
        raise FileNotFoundError(f"Missing CACTI column embedding: {embedding_path}")
    embedding_npz = np.load(embedding_path)
    embeddings_np = embedding_npz["colembed"].astype(np.float32)
    if embeddings_np.shape[0] != logical["train"].values.shape[1]:
        raise RuntimeError(
            f"Embedding/feature mismatch for {dataset}: {embeddings_np.shape[0]} vs "
            f"{logical['train'].values.shape[1]}"
        )
    info_embeddings = torch.from_numpy(embeddings_np).to(device)

    nhead = args.embed_dim // 8 if args.embed_dim >= 64 else 4
    model = CACTImodel(
        table_len=logical["train"].values.shape[1],
        mask_ratio=args.mask_ratio,
        embed_dim=args.embed_dim,
        depth=args.nencoder,
        num_heads=nhead,
        decoder_depth=args.ndecoder,
        decoder_num_heads=nhead,
        context_embed_dim=embeddings_np.shape[1],
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    train_tensor = torch.from_numpy(scaled["train"])
    train_dataset = EmbedCopyMaskedDataset(train_tensor, args.mask_ratio, str(embedding_path))
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / args.batch_size))
    scheduler = StepWiseWarmupCosineAnnealingScheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        min_lr=args.min_lr,
        steps_per_epoch=steps_per_epoch,
        warmup_start_lr=args.min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = {
        **vars(args),
        "dataset": dataset,
        "split_idx": split_idx,
        "seed": seed,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": official_commit(OFFICIAL_SOURCE_ROOT),
        "category_input": "one_scalar_ordinal_index_per_logical_categorical_column",
        "inference_batching": (
            "rows_grouped_by_observed_count_with_external_non_aliasing_contextify; "
            "verified_equivalent_to_native_batch_size_1"
        ),
        "inference_precision": "fp32_matching_official_transform",
        "selection_split": "valid",
        "checkpoint_selection": SELECTION_PROTOCOL,
        "selection_formula": SELECTION_FORMULA,
        "selection_weights": {
            "numeric_rmse": SELECTION_NUM_WEIGHT,
            "categorical_f1_error": SELECTION_F1_WEIGHT,
            "categorical_accuracy_error": SELECTION_ACCURACY_WEIGHT,
        },
        "selection_reference": selection_reference,
        "test_used_during_training": False,
    }
    atomic_json_dump(run_dir / "config.json", config)

    start_epoch = 0
    global_step = 0
    best_score: float | None = None
    best_report: dict[str, Any] | None = None
    no_improve = 0
    history: list[dict[str, Any]] = []
    if last_path.exists() and not args.overwrite:
        checkpoint = load_resume(last_path, model, optimizer, scheduler, scaler)
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_score = checkpoint.get("best_score")
        best_report = checkpoint.get("best_report")
        no_improve = int(checkpoint.get("no_improve", 0))
        history = list(checkpoint.get("history", []))
        print(
            f"[resume] dataset={dataset} split={split_idx} next_epoch={start_epoch + 1} "
            f"best_epoch={None if best_report is None else best_report.get('epoch')}",
            flush=True,
        )

    print(
        f"[start] model=CACTI-official dataset={dataset} split={split_idx} "
        f"train={len(logical['train'].values)} valid={len(logical['valid'].values)} "
        f"test={len(logical['test'].values)} features={logical['train'].values.shape[1]} "
        f"num={bundle.num_dim} cat={len(bundle.cat_col_idx)} device={device}",
        flush=True,
    )

    for epoch in range(start_epoch, args.max_epochs):
        generator = torch.Generator().manual_seed(seed + epoch * 1009)
        dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            generator=generator,
        )
        model.train()
        total_loss = 0.0
        finite_batches = 0
        progress = tqdm(dataloader, desc=f"{dataset} epoch {epoch + 1}/{args.max_epochs}", unit="batch")
        for batch in progress:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss, _ = model(batch["table"], batch["miss_map"], batch["mask_map"], batch["info_embed"])
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            total_loss += float(loss.detach().cpu())
            finite_batches += 1
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.5f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        progress.close()
        if finite_batches == 0:
            raise RuntimeError(f"No finite CACTI training loss in epoch {epoch + 1}.")

        valid_scaled_pred = predict_scaled(
            model, scaled["valid"], info_embeddings, device, args.eval_batch_size, use_amp,
            f"{dataset} valid epoch {epoch + 1}",
        )
        valid_pred = decode_categories(bundle, unscale_values(valid_scaled_pred, mins, maxs))
        valid_metrics = compute_metrics(bundle, "valid", valid_pred)
        score, score_components = selection_score(valid_metrics, selection_reference)
        improved = best_score is None or score < best_score - args.early_stop_min_delta
        if improved:
            best_score = score
            no_improve = 0
            best_report = {
                "epoch": epoch + 1,
                "score": score,
                "score_components": score_components,
                "valid_metrics": valid_metrics,
            }
            atomic_torch_save(
                best_path,
                {"model": model.state_dict(), "epoch": epoch + 1, "score": score, "mins": mins, "maxs": maxs},
            )
        else:
            no_improve += 1

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / finite_batches,
            "valid_score": score,
            "valid_score_components": score_components,
            "valid_metrics": valid_metrics,
            "best_score": best_score,
            "best_epoch": None if best_report is None else best_report["epoch"],
            "no_improve": no_improve,
            "lr": scheduler.get_last_lr()[0],
            "epoch_sec": None,
        }
        history.append(epoch_record)
        atomic_json_dump(history_path, history)
        atomic_torch_save(
            last_path,
            checkpoint_payload(
                model, optimizer, scheduler, scaler, epoch + 1, global_step, best_score,
                best_report, no_improve, history, mins, maxs, config,
            ),
        )
        print(
            f"[epoch] dataset={dataset} split={split_idx} epoch={epoch + 1}/{args.max_epochs} "
            f"loss={epoch_record['train_loss']:.6f} score={score:.6f} "
            f"valid_rmse={valid_metrics['rmse']} valid_f1={valid_metrics['cat_f1']} "
            f"valid_acc={valid_metrics['cat_accuracy']} improved={improved} "
            f"best_epoch={epoch_record['best_epoch']} no_improve={no_improve}",
            flush=True,
        )
        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(
                f"[early-stop] dataset={dataset} split={split_idx} epoch={epoch + 1} "
                f"best_epoch={None if best_report is None else best_report['epoch']}",
                flush=True,
            )
            break

    if not best_path.exists() or best_report is None:
        raise RuntimeError(f"Best checkpoint was not produced for {dataset} split {split_idx}.")
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])

    split_metrics: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "valid", "test"):
        scaled_pred = predict_scaled(
            model, scaled[split_name], info_embeddings, device, args.eval_batch_size, use_amp,
            f"{dataset} {split_name} final",
        )
        prediction = decode_categories(bundle, unscale_values(scaled_pred, mins, maxs))
        split_metrics[split_name] = compute_metrics(bundle, split_name, prediction)
        np.save(run_dir / f"{split_name}_pred_logical.npy", prediction.astype(np.float32))
        np.save(run_dir / f"{split_name}_pred_raw.npy", prediction_to_raw(bundle, split_name, prediction))

    best_report.update(
        train_metrics=split_metrics["train"],
        valid_metrics=split_metrics["valid"],
        test_metrics=split_metrics["test"],
        score=selection_score(split_metrics["valid"], selection_reference)[0],
        score_components=selection_score(split_metrics["valid"], selection_reference)[1],
        selection_split="valid",
    )
    final_report = {
        "model": "CACTI",
        "model_variant": "official_CACTI_MT-CM_GTE_fair_protocol",
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": config["official_commit"],
        "official_model_source_modified": False,
        "dataset": dataset,
        "split_idx": split_idx,
        "mask_type": args.mask_type,
        "ratio": args.ratio,
        "seed": seed,
        "run_dir": str(run_dir),
        "selection_split": "valid",
        "checkpoint_selection": SELECTION_PROTOCOL,
        "test_used_during_training": False,
        "best_score": best_report["score"],
        "best_report": best_report,
        "train_metrics": split_metrics["train"],
        "valid_metrics": split_metrics["valid"],
        "test_metrics": split_metrics["test"],
        "history": history,
        "split_sizes": {name: int(len(split.values)) for name, split in logical.items()},
        "protocol": config,
    }
    atomic_json_dump(report_path, final_report)
    if last_path.exists() and not args.keep_last_checkpoint:
        last_path.unlink()
    print(
        f"[done] dataset={dataset} split={split_idx} best_epoch={best_report['epoch']} "
        f"test_rmse={split_metrics['test']['rmse']} test_f1={split_metrics['test']['cat_f1']} "
        f"test_acc={split_metrics['test']['cat_accuracy']} report={report_path}",
        flush=True,
    )
    return final_report


def main() -> None:
    args = parse_args()
    if args.ndecoder % 2 != 0:
        raise ValueError("Official CACTI requires an even decoder depth.")
    for split_idx in args.splits:
        for dataset in args.datasets:
            run_single(args, dataset, int(split_idx))


if __name__ == "__main__":
    main()
