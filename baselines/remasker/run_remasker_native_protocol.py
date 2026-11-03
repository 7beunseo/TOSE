import argparse
import copy
import json
import math
import random
import sys
import time
from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

# ReMasker upstream uses the deprecated np.float alias in its positional
# embedding helper. Keep the original source intact and patch compatibility here.
if not hasattr(np, "float"):
    np.float = float

import model_mae
from utils import MAEDataset, adjust_learning_rate


SCRIPT_DIR = Path(__file__).resolve().parent
PMAE_DIR = SCRIPT_DIR.parent / "PMAE"
if str(PMAE_DIR) not in sys.path:
    sys.path.insert(1, str(PMAE_DIR))

from benchmark_dataset_utils import BaselineOneHotDataBundle  # noqa: E402

SHARED_DIR = next(
    parent / "_shared"
    for parent in Path(__file__).resolve().parents
    if (parent / "_shared" / "baseline_selection.py").exists()
)
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from baseline_selection import configure_selection, normalized_score, selection_metadata  # noqa: E402


EPS = 1e-8
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

DATASET_CONFIGS = {
    "abalone": dict(embed_dim=32, depth=4, decoder_depth=2, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "adult": dict(embed_dim=32, depth=6, decoder_depth=4, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "bank": dict(embed_dim=32, depth=6, decoder_depth=4, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "bean": dict(embed_dim=32, depth=4, decoder_depth=2, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "breast": dict(embed_dim=32, depth=4, decoder_depth=2, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "california": dict(embed_dim=32, depth=6, decoder_depth=4, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "default": dict(embed_dim=32, depth=6, decoder_depth=4, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "german": dict(embed_dim=32, depth=6, decoder_depth=4, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "gesture": dict(embed_dim=32, depth=6, decoder_depth=4, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "letter": dict(embed_dim=64, depth=8, decoder_depth=6, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "magic": dict(embed_dim=32, depth=4, decoder_depth=2, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "mushroom": dict(embed_dim=64, depth=8, decoder_depth=6, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "news": dict(embed_dim=64, depth=8, decoder_depth=6, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "shoppers": dict(embed_dim=32, depth=6, decoder_depth=4, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def metric_value(metrics: dict[str, Any] | None, key: str) -> float | None:
    if not metrics:
        return None
    value = metrics.get(key)
    if value is None and key == "cat_f1":
        value = metrics.get("cat_f1_weighted", metrics.get("f1"))
    if value is None and key == "cat_accuracy":
        value = metrics.get("accuracy")
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def score_from_metrics(metrics: dict[str, Any]) -> float:
    return normalized_score(metrics)


def aggregate_metrics(reports: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    fields = ["mae", "rmse", "cat_f1", "cat_f1_macro", "cat_f1_weighted", "cat_accuracy", "mae_raw", "rmse_raw"]
    out: dict[str, dict[str, float | None]] = {}
    for field in fields:
        vals = []
        for report in reports:
            metrics = report.get("test_metrics") or {}
            value = metrics.get(field)
            if value is not None and math.isfinite(float(value)):
                vals.append(float(value))
        if not vals:
            out[field] = {"mean": None, "std": None, "var": None}
        else:
            arr = np.asarray(vals, dtype=np.float64)
            out[field] = {"mean": float(arr.mean()), "std": float(arr.std()), "var": float(arr.var())}
    return out


def run_dir_for(args: argparse.Namespace, dataset: str, split_idx: int) -> Path:
    return (
        Path(args.output_root)
        / dataset
        / f"rate{args.ratio}"
        / args.mask_type
        / f"split_{split_idx}"
        / args.run_tag
    )


def is_completed(run_dir: Path) -> bool:
    report_path = run_dir / "report.json"
    pred_path = run_dir / "test_pred_raw.npy"
    if pred_path.exists():
        return True
    if not report_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(report.get("completed")) and isinstance(report.get("test_metrics"), dict)


class ReMaskerProtocolImputer:
    """ReMasker model wrapper with shared valid-based selection and metrics."""

    def __init__(self, args: Namespace) -> None:
        config = DATASET_CONFIGS.get(args.dataset, DATASET_CONFIGS["california"])
        self.dataset_name = args.dataset
        self.batch_size = int(args.batch_size)
        self.eval_batch_size = int(args.eval_batch_size)
        self.accum_iter = int(args.accum_iter)
        self.max_epochs = int(args.max_epochs)
        self.min_lr = float(args.min_lr)
        self.lr = args.lr
        self.blr = float(args.blr)
        self.weight_decay = float(args.weight_decay)
        self.warmup_epochs = int(args.warmup_epochs)
        self.norm_field_loss = bool(args.norm_field_loss)
        self.embed_dim = config["embed_dim"] if args.embed_dim is None else int(args.embed_dim)
        self.depth = config["depth"] if args.depth is None else int(args.depth)
        self.decoder_depth = config["decoder_depth"] if args.decoder_depth is None else int(args.decoder_depth)
        self.num_heads = config["num_heads"] if args.num_heads is None else int(args.num_heads)
        self.mlp_ratio = config["mlp_ratio"] if args.mlp_ratio is None else float(args.mlp_ratio)
        self.mask_ratio = config["mask_ratio"] if args.mask_ratio is None else float(args.mask_ratio)
        self.encode_func = "linear" if args.encode_func is None else args.encode_func
        self.device = resolve_device(args.device)
        self.use_amp = bool(args.amp) and self.device.type == "cuda"
        self.max_grad_norm = float(args.max_grad_norm)
        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.norm_parameters: dict[str, Any] | None = None
        self.training_history: list[dict[str, Any]] = []

    def fit_normalizer(self, x_raw: torch.Tensor) -> torch.Tensor:
        x = x_raw.clone().cpu().float()
        x_np = x.numpy().astype(np.float32, copy=True)
        dim = int(x_np.shape[1])
        min_val = np.zeros(dim, dtype=np.float32)
        max_val = np.ones(dim, dtype=np.float32)
        for col_idx in range(dim):
            column = x_np[:, col_idx]
            finite = np.isfinite(column)
            if not finite.any():
                min_val[col_idx] = 0.0
                max_val[col_idx] = 1.0
                continue
            min_val[col_idx] = float(np.nanmin(column))
            max_val[col_idx] = float(np.nanmax(column))
            if abs(max_val[col_idx] - min_val[col_idx]) < EPS:
                max_val[col_idx] = min_val[col_idx] + 1.0
            x[:, col_idx] = (x[:, col_idx] - min_val[col_idx]) / (max_val[col_idx] - min_val[col_idx] + EPS)
        self.norm_parameters = {
            "min": min_val,
            "max": max_val,
            "method": "remasker_native_minmax_fit_on_train_observed",
        }
        return x

    def init_model(self, dim: int) -> None:
        self.model = model_mae.MaskedAutoencoder(
            rec_len=dim,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            decoder_embed_dim=self.embed_dim,
            decoder_depth=self.decoder_depth,
            decoder_num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=EPS),
            norm_field_loss=self.norm_field_loss,
            encode_func=self.encode_func,
        )
        self.model.to(self.device)

    def training_state_dict(self, epoch: int) -> dict[str, Any]:
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Training state unavailable before fit.")
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "norm_parameters": self.norm_parameters,
            "training_history": copy.deepcopy(self.training_history),
            "base_lr": float(self.lr),
            "epoch": int(epoch),
        }

    def fit(self, x_raw: torch.Tensor, epoch_end_callback: Any | None = None, resume_state: dict | None = None) -> None:
        x = self.fit_normalizer(x_raw)
        observed_mask = (1 - (1 * torch.isnan(x))).float()
        x = torch.nan_to_num(x).float()
        dim = int(x.shape[1])
        self.init_model(dim)
        assert self.model is not None

        eff_batch_size = self.batch_size * self.accum_iter
        if self.lr is None:
            self.lr = self.blr * eff_batch_size / 64
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay,
        )
        start_epoch = 0
        if resume_state is not None:
            self.model.load_state_dict(resume_state["model_state_dict"])
            self.optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            if resume_state.get("scaler_state_dict") is not None:
                self.scaler.load_state_dict(resume_state["scaler_state_dict"])
            if resume_state.get("norm_parameters") is not None:
                self.norm_parameters = resume_state["norm_parameters"]
            self.training_history = copy.deepcopy(resume_state.get("training_history", []))
            self.lr = float(resume_state.get("base_lr", self.lr))
            start_epoch = int(resume_state.get("epoch", 0))
            print(f"[resume] dataset={self.dataset_name} start_epoch={start_epoch + 1}", flush=True)

        train_dataset = MAEDataset(x, observed_mask)
        loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=self.batch_size)
        print(
            f"[train-start] dataset={self.dataset_name} rows={x.shape[0]} dim={dim} "
            f"epochs={self.max_epochs} device={self.device} amp={self.use_amp}",
            flush=True,
        )
        for epoch in range(start_epoch, self.max_epochs):
            self.model.train()
            total_loss = 0.0
            self.optimizer.zero_grad(set_to_none=True)
            iter_idx = 0
            for iter_idx, (samples, masks) in enumerate(loader):
                if iter_idx % self.accum_iter == 0:
                    adjust_learning_rate(
                        self.optimizer,
                        iter_idx / len(loader) + epoch,
                        float(self.lr),
                        self.min_lr,
                        self.max_epochs,
                        self.warmup_epochs,
                    )
                samples = samples.unsqueeze(dim=1).to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    loss, _, _, _ = self.model(samples, masks, mask_ratio=self.mask_ratio)
                    loss_value = float(loss.item())
                if not math.isfinite(loss_value):
                    raise RuntimeError(f"Non-finite loss: {loss_value}")
                total_loss += loss_value
                loss = loss / self.accum_iter
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    if (iter_idx + 1) % self.accum_iter == 0:
                        self.scaler.unscale_(self.optimizer)
                        if self.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                else:
                    loss.backward()
                    if (iter_idx + 1) % self.accum_iter == 0:
                        if self.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

            train_loss = float(math.sqrt(total_loss / (iter_idx + 1)))
            record = {"epoch": epoch + 1, "train_loss": train_loss, "lr": float(self.optimizer.param_groups[0]["lr"])}
            self.training_history.append(record)
            print(
                f"[train] dataset={self.dataset_name} epoch={epoch + 1}/{self.max_epochs} "
                f"loss={train_loss:.6f} lr={record['lr']:.6g}",
                flush=True,
            )
            if epoch_end_callback is not None and bool(epoch_end_callback(self, epoch + 1, record)):
                break

    def transform(self, x_raw: torch.Tensor) -> torch.Tensor:
        if self.model is None or self.norm_parameters is None:
            raise RuntimeError("Model is not trained.")
        x_original = x_raw.clone().cpu().float()
        x = x_original.clone()
        min_val = torch.as_tensor(self.norm_parameters["min"], dtype=torch.float32)
        max_val = torch.as_tensor(self.norm_parameters["max"], dtype=torch.float32)
        x = (x - min_val[None, :]) / (max_val[None, :] - min_val[None, :] + EPS)
        observed_mask = (1 - (1 * torch.isnan(x))).float()
        x = torch.nan_to_num(x).float()
        loader = DataLoader(TensorDataset(x, observed_mask), batch_size=self.eval_batch_size, shuffle=False)
        preds = []
        self.model.eval()
        with torch.no_grad():
            for samples, masks in loader:
                samples = samples.unsqueeze(dim=1).to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                _, pred, _, _ = self.model(samples, masks)
                preds.append(pred.squeeze(dim=2).detach().cpu())
        imputed = torch.cat(preds, dim=0)
        imputed = imputed * (max_val[None, :] - min_val[None, :] + EPS) + min_val[None, :]
        return observed_mask.cpu() * torch.nan_to_num(x_original) + (1 - observed_mask.cpu()) * imputed


def build_model_args(args: argparse.Namespace, dataset: str) -> Namespace:
    return Namespace(
        dataset=dataset,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        accum_iter=args.accum_iter,
        max_epochs=args.max_epochs,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        embed_dim=args.embed_dim,
        depth=args.depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        mask_ratio=args.mask_ratio,
        encode_func=args.encode_func,
        norm_field_loss=args.norm_field_loss,
        weight_decay=args.weight_decay,
        lr=args.lr,
        blr=args.blr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        device=args.device,
        amp=args.amp,
        max_grad_norm=args.max_grad_norm,
    )


def run_single_split(args: argparse.Namespace, dataset: str, split_idx: int) -> dict[str, Any]:
    run_dir = run_dir_for(args, dataset, split_idx)
    if args.skip_completed and not args.overwrite and is_completed(run_dir):
        print(f"[skip-completed] dataset={dataset} split={split_idx} run_dir={run_dir}", flush=True)
        with open(run_dir / "report.json", "r", encoding="utf-8") as f:
            return json.load(f)
    if args.overwrite and run_dir.exists():
        raise RuntimeError("Refusing to delete existing run_dir. Move/delete it manually before --overwrite rerun.")
    run_dir.mkdir(parents=True, exist_ok=True)

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

    seed = int(args.seed) + int(split_idx)
    set_all_seeds(seed)
    bundle = BaselineOneHotDataBundle(
        dataset_name=dataset,
        mask_type=args.mask_type,
        split_idx=split_idx,
        ratio=args.ratio,
        data_root=args.data_root,
        seed=seed,
        valid_max_fraction=args.valid_max_fraction,
    )
    train_x = bundle.train_x_raw.cpu().numpy().copy()
    valid_x = bundle.valid_x_raw.cpu().numpy().copy()
    test_x = bundle.test_x_raw.cpu().numpy().copy()
    train_x[bundle.train_mask_ext.cpu().numpy()] = np.nan
    valid_x[bundle.valid_mask_ext.cpu().numpy()] = np.nan
    test_x[bundle.test_mask_ext.cpu().numpy()] = np.nan
    train_x_t = torch.tensor(train_x, dtype=torch.float32)
    valid_x_t = torch.tensor(valid_x, dtype=torch.float32)
    test_x_t = torch.tensor(test_x, dtype=torch.float32)

    imputer = ReMaskerProtocolImputer(build_model_args(args, dataset))
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = checkpoints_dir / "best.pt"
    last_ckpt = checkpoints_dir / "last.pt"

    config = {
        "dataset": dataset,
        "mask_type": args.mask_type,
        "ratio": args.ratio,
        "split_idx": split_idx,
        "run_tag": args.run_tag,
        "seed": seed,
        "model_family": "remasker",
        "numeric_source": "prepared_npz_x_norm",
        "scaler_source": "prepared_npz_mean_std",
        "metric_space": "eunseo_normalized",
        "categorical_protocol": "model_native",
        "preprocessing_metadata": {
            **getattr(bundle, "preprocessing_metadata", {}),
            "model_family": "remasker_baseline",
            "numeric_train_transform": "remasker_native_minmax_fit_on_train_observed",
            "numeric_inverse_formula": "ReMasker inverse MinMax before shared metric evaluation",
            "categorical_preprocessing": "one_hot",
            "categorical_decoding": "one_hot_group_argmax_to_original_label",
        },
        "protocol_checks": getattr(bundle, "protocol_checks", {}),
        "max_epochs": args.max_epochs,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "selection_split": "valid",
        **selection_metadata(),
        "score_formula": selection_metadata()["selection_formula"],
    }
    save_json(run_dir / "config.json", config)

    resume_state = None
    if args.resume_incomplete and last_ckpt.exists() and not args.overwrite:
        print(f"[auto-resume] dataset={dataset} split={split_idx} checkpoint={last_ckpt}", flush=True)
        resume_state = torch.load(last_ckpt, map_location=resolve_device(args.device), weights_only=False)

    tracking = {
        "best_score": None if resume_state is None else resume_state.get("best_score"),
        "best_epoch": None if resume_state is None else resume_state.get("best_epoch"),
        "best_report": None if resume_state is None else resume_state.get("best_report"),
        "epochs_since_improvement": 0 if resume_state is None else int(resume_state.get("epochs_since_improvement", 0)),
        "latest_metrics": None if resume_state is None else resume_state.get("latest_metrics"),
        "completed": False,
        "stopped_early": False,
        "stop_reason": None,
    }

    def report_payload() -> dict[str, Any]:
        selected = tracking["best_report"] or tracking["latest_metrics"] or {}
        return {
            **config,
            "best_epoch": tracking["best_epoch"],
            "best_score": tracking["best_score"],
            "best_report": tracking["best_report"],
            "train_metrics": selected.get("train_metrics"),
            "valid_metrics": selected.get("valid_metrics"),
            "test_metrics": selected.get("test_metrics"),
            "latest_metrics": tracking["latest_metrics"],
            "epochs_since_improvement": tracking["epochs_since_improvement"],
            "stopped_early": tracking["stopped_early"],
            "stop_reason": tracking["stop_reason"],
            "completed": tracking["completed"],
            "split_sizes": bundle.split_sizes(),
            "num_dim": bundle.num_dim,
            "bit_dim": bundle.bit_dim,
        }

    def save_last(epoch_idx: int) -> None:
        state = imputer.training_state_dict(epoch_idx)
        state.update(
            {
                "config": config,
                "best_score": tracking["best_score"],
                "best_epoch": tracking["best_epoch"],
                "best_report": tracking["best_report"],
                "epochs_since_improvement": tracking["epochs_since_improvement"],
                "latest_metrics": tracking["latest_metrics"],
            }
        )
        torch.save(state, last_ckpt)
        save_json(run_dir / "history.json", imputer.training_history)
        save_json(run_dir / "report.json", report_payload())

    def on_epoch_end(current: ReMaskerProtocolImputer, epoch_idx: int, record: dict[str, Any]) -> bool:
        train_pred = current.transform(train_x_t).detach().cpu().numpy()
        valid_pred = current.transform(valid_x_t).detach().cpu().numpy()
        train_metrics = bundle.evaluate_predictions(train_pred, split="train")
        valid_metrics = bundle.evaluate_predictions(valid_pred, split="valid")
        score = score_from_metrics(valid_metrics)
        tracking["latest_metrics"] = {
            "epoch": epoch_idx,
            "score": score,
            "selection_split": "valid",
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "test_metrics": None,
        }
        improved = tracking["best_score"] is None or score < float(tracking["best_score"]) - args.early_stop_min_delta
        if improved:
            tracking["best_score"] = score
            tracking["best_epoch"] = epoch_idx
            tracking["best_report"] = copy.deepcopy(tracking["latest_metrics"])
            tracking["epochs_since_improvement"] = 0
            best_state = current.training_state_dict(epoch_idx)
            best_state.update({"config": config, "best_score": score, "best_epoch": epoch_idx, "best_report": tracking["best_report"]})
            torch.save(best_state, best_ckpt)
            np.save(run_dir / "best_train_pred_raw.npy", train_pred)
            np.save(run_dir / "best_valid_pred_raw.npy", valid_pred)
        else:
            tracking["epochs_since_improvement"] += 1
        print(
            f"[eval] dataset={dataset} split={split_idx} epoch={epoch_idx} "
            f"score={score:.6f} valid_rmse={valid_metrics.get('rmse')} "
            f"valid_f1={valid_metrics.get('cat_f1')} valid_acc={valid_metrics.get('cat_accuracy')}",
            flush=True,
        )
        save_last(epoch_idx)
        if args.early_stop_patience > 0 and tracking["epochs_since_improvement"] >= args.early_stop_patience:
            tracking["stopped_early"] = True
            tracking["stop_reason"] = "early_stop_patience_exhausted"
            save_last(epoch_idx)
            return True
        return False

    imputer.fit(train_x_t, epoch_end_callback=on_epoch_end, resume_state=resume_state)

    if not best_ckpt.exists():
        raise RuntimeError(f"best checkpoint missing: {best_ckpt}")
    best_state = torch.load(best_ckpt, map_location=resolve_device(args.device), weights_only=False)
    imputer.fit_normalizer(train_x_t)
    imputer.init_model(train_x_t.shape[1])
    assert imputer.model is not None
    imputer.model.load_state_dict(best_state["model_state_dict"])
    imputer.norm_parameters = best_state["norm_parameters"]

    train_pred = imputer.transform(train_x_t).detach().cpu().numpy()
    valid_pred = imputer.transform(valid_x_t).detach().cpu().numpy()
    test_pred = imputer.transform(test_x_t).detach().cpu().numpy()
    train_metrics = bundle.evaluate_predictions(train_pred, split="train")
    valid_metrics = bundle.evaluate_predictions(valid_pred, split="valid")
    test_metrics = bundle.evaluate_predictions(test_pred, split="test")
    score = score_from_metrics(valid_metrics)
    tracking["best_score"] = score
    tracking["best_epoch"] = int(best_state.get("epoch", tracking["best_epoch"] or 0))
    tracking["best_report"] = {
        "epoch": tracking["best_epoch"],
        "score": score,
        "selection_split": "valid",
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }
    tracking["latest_metrics"] = tracking["best_report"]
    tracking["completed"] = True
    np.save(run_dir / "test_pred_raw.npy", test_pred)
    np.save(run_dir / "train_pred_raw.npy", train_pred)
    np.save(run_dir / "valid_pred_raw.npy", valid_pred)
    save_json(run_dir / "report.json", report_payload())
    print(
        f"[done] dataset={dataset} split={split_idx} test_rmse={test_metrics.get('rmse')} "
        f"test_f1={test_metrics.get('cat_f1')} test_acc={test_metrics.get('cat_accuracy')}",
        flush=True,
    )
    return report_payload()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ReMasker with native minmax training and shared Eunseo metrics.")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    public_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-root", default=str(public_root / "data"))
    parser.add_argument("--output-root", default=str(public_root / "results" / "remasker"))
    parser.add_argument("--mask-type", required=True)
    parser.add_argument("--ratio", default="30")
    parser.add_argument("--splits", nargs="*", type=int, default=list(range(10)))
    parser.add_argument("--run-tag", default="remasker_native_e600_es")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--early-stop-patience", type=int, default=45)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0002)
    parser.add_argument("--valid-max-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--accum-iter", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--decoder-depth", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--mlp-ratio", type=float, default=None)
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--encode-func", default=None)
    parser.add_argument("--norm-field-loss", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=40)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--resume-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = []
    for dataset in args.datasets:
        for split_idx in args.splits:
            reports.append(run_single_split(args, dataset, int(split_idx)))
    summary_dir = Path(args.output_root) / "_summaries"
    save_json(summary_dir / f"{args.run_tag}_{args.mask_type}_reports.json", reports)
    save_json(
        summary_dir / f"{args.run_tag}_{args.mask_type}_summary.json",
        {"aggregate_test_metrics": aggregate_metrics(reports), "num_reports": len(reports)},
    )


if __name__ == "__main__":
    main()
