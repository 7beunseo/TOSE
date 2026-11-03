import argparse
import csv
import gc
import json
import math
import random
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmark_dataset_utils import (
    BaselineOneHotDataBundle,
    PMAELabelRoundDataBundle,
    evaluate_predictions,
)
from pmae_same_protocol import PMAEProtocolImputer, set_all_seeds


SHARED_DIR = next(
    parent / "_shared"
    for parent in Path(__file__).resolve().parents
    if (parent / "_shared" / "baseline_selection.py").exists()
)
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from baseline_selection import configure_selection, normalized_score  # noqa: E402


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "pmae"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PMAE on prepared same-protocol missing-value benchmark splits.")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mask-type", type=str, default="MNAR_logistic_T2_v2_strict")
    parser.add_argument("--ratio", type=str, default="30")
    parser.add_argument("--splits", nargs="*", type=int, default=[0])
    parser.add_argument("--run-tag", type=str, default="pmae_same_protocol_split0_e600_es")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--accum-iter", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=45)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0002)
    parser.add_argument("--selection-split", type=str, default="valid", choices=["valid"])
    parser.add_argument("--valid-max-fraction", type=float, default=0.15)
    parser.add_argument(
        "--categorical-preprocessing",
        type=str,
        default="pmae_label_round",
        choices=["pmae_label_round", "baseline_one_hot"],
        help=(
            "pmae_label_round follows the released PMAE evaluation style for categorical columns; "
            "baseline_one_hot keeps PMAE free of Eunseo codebooks and evaluates original labels"
        ),
    )
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-summaries", action="store_true", help="skip writing shared summary files")

    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--decoder-depth", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--mlp-ratio", type=float, default=None)
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--encode-func", type=str, default=None)
    parser.add_argument("--norm-field-loss", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=40)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument(
        "--input-normalization",
        type=str,
        default="minmax",
        choices=["minmax", "identity"],
        help=(
            "PMAE-native training preprocessing. Use minmax to fit a PMAE MinMax transform on "
            "the strict training inputs; metrics are still computed in the shared Eunseo x_norm space."
        ),
    )
    parser.add_argument("--pmae-a", type=float, default=0.05)
    parser.add_argument("--pmae-b", type=float, default=0.5)
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, allow_nan=False)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _score_from_metrics(metrics: dict[str, Any]) -> float:
    return normalized_score(metrics)


def aggregate_metrics(reports: list[dict]) -> dict[str, dict[str, float | None]]:
    metrics = {}
    for field in [
        "mae",
        "mse",
        "rmse",
        "mae_raw",
        "mse_raw",
        "rmse_raw",
        "cat_f1",
        "cat_f1_macro",
        "cat_f1_weighted",
        "bit_f1",
        "cat_accuracy",
        "accuracy",
    ]:
        values = []
        for report in reports:
            value = report["test_metrics"].get(field)
            if value is not None:
                values.append(float(value))
        if len(values) == 0:
            metrics[field] = {"mean": None, "var": None, "std": None}
        else:
            arr = np.asarray(values, dtype=np.float64)
            metrics[field] = {"mean": float(arr.mean()), "var": float(arr.var()), "std": float(arr.std())}
    return metrics


def build_model_args(args: argparse.Namespace, dataset: str) -> Namespace:
    return Namespace(
        dataset=dataset,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_epochs=args.max_epochs,
        accum_iter=args.accum_iter,
        norm_field_loss=args.norm_field_loss,
        weight_decay=args.weight_decay,
        lr=args.lr,
        blr=args.blr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        mask_ratio=args.mask_ratio,
        embed_dim=args.embed_dim,
        depth=args.depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        encode_func=args.encode_func,
        device=args.device,
        log_interval=args.log_interval,
        amp=args.amp,
        max_grad_norm=args.max_grad_norm,
        input_normalization=args.input_normalization,
        pmae_a=args.pmae_a,
        pmae_b=args.pmae_b,
    )


def get_run_dir(args: argparse.Namespace, dataset: str, split_idx: int) -> Path:
    return args.output_root / dataset / f"rate{args.ratio}" / args.mask_type / f"split_{split_idx}" / args.run_tag


def load_existing_report(run_dir: Path) -> dict | None:
    report_path = run_dir / "report.json"
    if not report_path.exists():
        return None
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_completed_report(report: dict | None) -> bool:
    if not report or not report.get("completed", False):
        return False
    test_metrics = report.get("test_metrics")
    preprocessing_metadata = report.get("preprocessing_metadata") or {}
    return (
        report.get("numeric_source") == "prepared_npz_x_norm"
        and report.get("scaler_source") == "prepared_npz_mean_std"
        and report.get("metric_space") == "eunseo_normalized"
        and report.get("categorical_protocol") == "model_native"
        and preprocessing_metadata.get("numeric_train_transform") == "pmae_native_minmax_fit_on_train_observed"
        and preprocessing_metadata.get("categorical_input_space") == "raw_category_index_then_pmae_minmax"
        and preprocessing_metadata.get("categorical_transform_formula") == "category_index"
        and isinstance(test_metrics, dict)
        and test_metrics.get("categorical_metric_level") == "column_prefixed_category_index"
        and test_metrics.get("categorical_f1_average") == "weighted_over_column_prefixed_category_indices"
        and test_metrics.get("categorical_token_format") == "col{original_column_index}=idx{category_index}"
    )


def is_resume_state_compatible(resume_state: dict | None, config_payload: dict) -> bool:
    if not isinstance(resume_state, dict):
        return False
    resume_config = resume_state.get("config")
    if not isinstance(resume_config, dict):
        return False
    required_keys = ["numeric_source", "scaler_source", "metric_space", "categorical_protocol", "input_normalization"]
    if any(resume_config.get(key) != config_payload.get(key) for key in required_keys):
        return False
    resume_preprocessing = resume_config.get("preprocessing_metadata") or {}
    current_preprocessing = config_payload.get("preprocessing_metadata") or {}
    preprocessing_keys = ["categorical_input_space", "categorical_transform_formula", "categorical_decoding"]
    if any(resume_preprocessing.get(key) != current_preprocessing.get(key) for key in preprocessing_keys):
        return False
    norm_parameters = resume_state.get("norm_parameters") or {}
    if norm_parameters and norm_parameters.get("method") != "pmae_native_minmax_fit_on_train_observed":
        return False
    return True


def write_dataset_csv(path: Path, reports: list[dict], run_tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "mask_type",
                "split_idx",
                "run_tag",
                "train_mae",
                "train_rmse",
                "train_f1",
                "train_accuracy",
                "valid_mae",
                "valid_rmse",
                "valid_f1",
                "valid_accuracy",
                "test_mae",
                "test_rmse",
                "test_f1",
                "test_accuracy",
            ],
        )
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    "dataset": report["dataset"],
                    "mask_type": report["mask_type"],
                    "split_idx": report["split_idx"],
                    "run_tag": run_tag,
                    "train_mae": report["train_metrics"].get("mae"),
                    "train_rmse": report["train_metrics"].get("rmse"),
                    "train_f1": report["train_metrics"].get("cat_f1", report["train_metrics"].get("bit_f1")),
                    "train_accuracy": report["train_metrics"].get("cat_accuracy"),
                    "valid_mae": report["valid_metrics"].get("mae"),
                    "valid_rmse": report["valid_metrics"].get("rmse"),
                    "valid_f1": report["valid_metrics"].get("cat_f1", report["valid_metrics"].get("bit_f1")),
                    "valid_accuracy": report["valid_metrics"].get("cat_accuracy"),
                    "test_mae": report["test_metrics"].get("mae"),
                    "test_rmse": report["test_metrics"].get("rmse"),
                    "test_f1": report["test_metrics"].get("cat_f1", report["test_metrics"].get("bit_f1")),
                    "test_accuracy": report["test_metrics"].get("cat_accuracy"),
                }
            )


def write_global_csv(path: Path, benchmark_summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "run_tag",
                "test_mae_mean",
                "test_mae_std",
                "test_rmse_mean",
                "test_rmse_std",
                "test_f1_mean",
                "test_f1_std",
                "test_accuracy_mean",
                "test_accuracy_std",
            ],
        )
        writer.writeheader()
        for dataset, summary in benchmark_summary.items():
            agg = summary["aggregate_test_metrics"]
            writer.writerow(
                {
                    "dataset": dataset,
                    "run_tag": summary["selected_tag"],
                    "test_mae_mean": agg["mae"]["mean"],
                    "test_mae_std": agg["mae"]["std"],
                    "test_rmse_mean": agg["rmse"]["mean"],
                    "test_rmse_std": agg["rmse"]["std"],
                    "test_f1_mean": (agg.get("cat_f1") or agg["bit_f1"])["mean"],
                    "test_f1_std": (agg.get("cat_f1") or agg["bit_f1"])["std"],
                    "test_accuracy_mean": agg["cat_accuracy"]["mean"],
                    "test_accuracy_std": agg["cat_accuracy"]["std"],
                }
            )


def run_single_split(args: argparse.Namespace, dataset: str, split_idx: int) -> dict:
    if args.selection_split != "valid":
        raise ValueError("This same-protocol PMAE runner only allows valid-based model selection.")

    seed = args.seed + split_idx
    set_all_seeds(seed)
    run_dir = get_run_dir(args, dataset, split_idx)
    existing_report = None if args.overwrite else load_existing_report(run_dir)
    if is_completed_report(existing_report):
        print(f"[skip] dataset={dataset} split={split_idx} existing_report={run_dir / 'report.json'}", flush=True)
        return existing_report

    selection_metadata = configure_selection(
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

    print(f"[prep] dataset={dataset} split={split_idx} seed={seed}", flush=True)
    bundle_cls = PMAELabelRoundDataBundle
    if args.categorical_preprocessing == "baseline_one_hot":
        bundle_cls = BaselineOneHotDataBundle
    bundle = bundle_cls(
        dataset_name=dataset,
        mask_type=args.mask_type,
        split_idx=split_idx,
        ratio=args.ratio,
        data_root=args.data_root,
        seed=seed,
        valid_max_fraction=args.valid_max_fraction,
    )
    model_args = build_model_args(args, dataset)
    imputer = PMAEProtocolImputer(model_args)

    train_x = bundle.train_x_raw.cpu().numpy().copy()
    valid_x = bundle.valid_x_raw.cpu().numpy().copy()
    test_x = bundle.test_x_raw.cpu().numpy().copy()
    train_x[bundle.train_mask_ext.cpu().numpy()] = np.nan
    valid_x[bundle.valid_mask_ext.cpu().numpy()] = np.nan
    test_x[bundle.test_mask_ext.cpu().numpy()] = np.nan

    train_x_t = torch.tensor(train_x, dtype=torch.float32)
    valid_x_t = torch.tensor(valid_x, dtype=torch.float32)
    test_x_t = torch.tensor(test_x, dtype=torch.float32)

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoints_dir / "best.pt"
    last_ckpt_path = checkpoints_dir / "last.pt"

    config_payload = {
        "dataset": dataset,
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "mask_type": args.mask_type,
        "ratio": args.ratio,
        "split_idx": split_idx,
        "run_tag": args.run_tag,
        "seed": seed,
        "device": str(imputer.device),
        "batch_size": model_args.batch_size,
        "eval_batch_size": model_args.eval_batch_size,
        "max_epochs": imputer.max_epochs,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "selection_split": args.selection_split,
        **selection_metadata,
        "valid_max_fraction": args.valid_max_fraction,
        "score_formula": selection_metadata["selection_formula"],
        "numeric_source": "prepared_npz_x_norm",
        "scaler_source": "prepared_npz_mean_std",
        "metric_space": "eunseo_normalized",
        "categorical_protocol": "model_native",
        "input_normalization": args.input_normalization,
        "categorical_preprocessing": args.categorical_preprocessing,
        "preprocessing_metadata": getattr(bundle, "preprocessing_metadata", {
            "model_family": "baseline_reproduction",
            "categorical_preprocessing": args.categorical_preprocessing,
            "categorical_metric_level": "column_prefixed_category_index",
            "categorical_f1_average": "weighted_over_column_prefixed_category_indices",
            "cat_f1_average": "weighted_over_column_prefixed_category_indices",
            "categorical_token_format": "col{original_column_index}=idx{category_index}",
            "uses_eunseo_codebook": False,
        }),
        "protocol_checks": getattr(bundle, "protocol_checks", {}),
        "embed_dim": imputer.embed_dim,
        "depth": imputer.depth,
        "decoder_depth": imputer.decoder_depth,
        "num_heads": imputer.num_heads,
        "mlp_ratio": imputer.mlp_ratio,
        "encode_func": imputer.encode_func,
        "pmae_a": imputer.a,
        "pmae_b": imputer.b,
    }
    save_json(run_dir / "config.json", config_payload)

    resume_state = None
    resume_checkpoint = args.resume_checkpoint
    if resume_checkpoint is None and not args.overwrite and last_ckpt_path.exists():
        resume_checkpoint = last_ckpt_path
        print(f"[auto-resume] dataset={dataset} split={split_idx} checkpoint={resume_checkpoint}", flush=True)
    if resume_checkpoint is not None:
        try:
            resume_state = torch.load(resume_checkpoint, map_location=imputer.device, weights_only=False)
            if not is_resume_state_compatible(resume_state, config_payload):
                print(
                    f"[resume-incompatible] dataset={dataset} split={split_idx} checkpoint={resume_checkpoint} "
                    "does not match PMAE native-MinMax / shared-metric protocol; restarting.",
                    flush=True,
                )
                resume_state = None
            else:
                print(
                    f"[resume-load] dataset={dataset} split={split_idx} checkpoint={resume_checkpoint} "
                    f"epoch={resume_state.get('epoch')}",
                    flush=True,
                )
        except Exception as exc:
            print(
                f"[resume-failed] dataset={dataset} split={split_idx} checkpoint={resume_checkpoint} "
                f"error={type(exc).__name__}: {exc}. Restarting in the same run directory.",
                flush=True,
            )
            resume_state = None
    tracking = {
        "best_score": None,
        "best_report": None,
        "best_epoch": None,
        "best_train_pred_raw": None,
        "best_valid_pred_raw": None,
        "best_test_pred_raw": None,
        "latest_epoch": None,
        "latest_metrics": None,
        "epochs_since_improvement": 0,
        "stopped_early": False,
        "stop_reason": None,
        "completed": False,
    }
    if resume_state is not None:
        tracking["best_score"] = resume_state.get("best_score")
        tracking["best_report"] = resume_state.get("best_report")
        tracking["best_epoch"] = resume_state.get("best_epoch")
        tracking["epochs_since_improvement"] = int(resume_state.get("epochs_since_improvement", 0))
        tracking["stopped_early"] = bool(resume_state.get("stopped_early", False))
        tracking["stop_reason"] = resume_state.get("stop_reason")
        tracking["latest_epoch"] = resume_state.get("latest_epoch")
        tracking["latest_metrics"] = resume_state.get("latest_metrics")
        tracking["completed"] = bool(resume_state.get("completed", False))

    def build_report_payload() -> dict:
        selected = tracking["best_report"] or tracking["latest_metrics"] or {}
        return {
            "dataset": dataset,
            "mask_type": args.mask_type,
            "ratio": args.ratio,
            "split_idx": split_idx,
            "run_tag": args.run_tag,
            "numeric_source": "prepared_npz_x_norm",
            "scaler_source": "prepared_npz_mean_std",
            "metric_space": "eunseo_normalized",
            "categorical_protocol": "model_native",
            "best_epoch": tracking["best_epoch"],
            "best_score": tracking["best_score"],
            "best_report": tracking["best_report"],
            "val_metrics": selected.get("valid_metrics"),
            "val_protocol": {
                "type": f"{args.selection_split}_score_same_as_baselines",
                "score_formula": config_payload["score_formula"],
                "early_stop_patience": args.early_stop_patience,
                "early_stop_min_delta": args.early_stop_min_delta,
                "valid_max_fraction": args.valid_max_fraction,
            },
            "selection_split": args.selection_split,
            **selection_metadata,
            "split_sizes": bundle.split_sizes(),
            "preprocessing_metadata": config_payload["preprocessing_metadata"],
            "protocol_checks": config_payload["protocol_checks"],
            "train_metrics": selected.get("train_metrics"),
            "valid_metrics": selected.get("valid_metrics"),
            "test_metrics": selected.get("test_metrics"),
            "latest_epoch": tracking["latest_epoch"],
            "latest_metrics": tracking["latest_metrics"],
            "epochs_since_improvement": tracking["epochs_since_improvement"],
            "stopped_early": tracking["stopped_early"],
            "stop_reason": tracking["stop_reason"],
            "completed": tracking["completed"],
            "num_dim": bundle.num_dim,
            "bit_dim": bundle.bit_dim,
        }

    def save_resume_checkpoint(epoch_idx: int) -> None:
        checkpoint = imputer.training_state_dict(epoch=epoch_idx)
        checkpoint.update(
            {
                "config": config_payload,
                "rng_state": capture_rng_state(),
                "best_score": tracking["best_score"],
                "best_epoch": tracking["best_epoch"],
                "best_report": tracking["best_report"],
                "latest_epoch": tracking["latest_epoch"],
                "latest_metrics": tracking["latest_metrics"],
                "epochs_since_improvement": tracking["epochs_since_improvement"],
                "stopped_early": tracking["stopped_early"],
                "stop_reason": tracking["stop_reason"],
                "completed": tracking["completed"],
            }
        )
        last_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, last_ckpt_path)

    def save_progress_report() -> None:
        save_json(run_dir / "history.json", imputer.training_history)
        save_json(run_dir / "report.json", build_report_payload())

    def on_epoch_end(current_imputer: PMAEProtocolImputer, epoch_idx: int, epoch_record: dict) -> bool:
        print(f"[predict] dataset={dataset} split={split_idx} epoch={epoch_idx} train", flush=True)
        train_pred_raw = current_imputer.transform(train_x_t).detach().cpu().numpy()
        print(f"[predict] dataset={dataset} split={split_idx} epoch={epoch_idx} valid", flush=True)
        valid_pred_raw = current_imputer.transform(valid_x_t).detach().cpu().numpy()

        train_metrics = evaluate_predictions(bundle, train_pred_raw, split="train")
        valid_metrics = evaluate_predictions(bundle, valid_pred_raw, split="valid")
        test_metrics = None
        test_pred_raw = None
        if args.selection_split == "test":
            print(f"[predict] dataset={dataset} split={split_idx} epoch={epoch_idx} test", flush=True)
            test_pred_raw = current_imputer.transform(test_x_t).detach().cpu().numpy()
            test_metrics = evaluate_predictions(bundle, test_pred_raw, split="test")

        selection_metrics = valid_metrics if args.selection_split == "valid" else test_metrics
        current_score = _score_from_metrics(selection_metrics)
        epoch_record.update(
            {
                "train_mae": train_metrics.get("mae"),
                "train_rmse": train_metrics.get("rmse"),
                "train_f1": train_metrics.get("cat_f1", train_metrics.get("bit_f1")),
                "train_accuracy": train_metrics.get("cat_accuracy"),
                "valid_mae": valid_metrics.get("mae"),
                "valid_rmse": valid_metrics.get("rmse"),
                "valid_f1": valid_metrics.get("cat_f1", valid_metrics.get("bit_f1")),
                "valid_accuracy": valid_metrics.get("cat_accuracy"),
                "score": current_score,
            }
        )

        tracking["latest_epoch"] = epoch_idx
        tracking["latest_metrics"] = {
            "epoch": epoch_idx,
            "score": current_score,
            "selection_split": args.selection_split,
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        }

        improved = tracking["best_score"] is None or current_score < (
            float(tracking["best_score"]) - args.early_stop_min_delta
        )
        epoch_record["improved"] = bool(improved)
        if improved:
            tracking["best_score"] = current_score
            tracking["best_epoch"] = epoch_idx
            tracking["best_report"] = {
                "epoch": epoch_idx,
                "score": current_score,
                "selection_split": args.selection_split,
                "train_metrics": train_metrics,
                "valid_metrics": valid_metrics,
                "test_metrics": test_metrics,
            }
            tracking["best_train_pred_raw"] = train_pred_raw
            tracking["best_valid_pred_raw"] = valid_pred_raw
            tracking["best_test_pred_raw"] = test_pred_raw
            tracking["epochs_since_improvement"] = 0
            checkpoint = current_imputer.training_state_dict(epoch=epoch_idx)
            checkpoint.update(
                {
                    "config": config_payload,
                    "best_score": current_score,
                    "best_epoch": epoch_idx,
                    "best_report": tracking["best_report"],
                    "selected_by": f"{args.selection_split}_score",
                    "rng_state": capture_rng_state(),
                }
            )
            torch.save(checkpoint, best_ckpt_path)
            np.save(run_dir / "best_train_pred_raw.npy", train_pred_raw)
            np.save(run_dir / "best_valid_pred_raw.npy", valid_pred_raw)
        else:
            tracking["epochs_since_improvement"] += 1

        print(
            f"[eval] dataset={dataset} split={split_idx} epoch={epoch_idx} "
            f"score={current_score:.6f} valid_rmse={valid_metrics.get('rmse')} "
            f"valid_f1={valid_metrics.get('cat_f1', valid_metrics.get('bit_f1'))} "
            f"valid_acc={valid_metrics.get('cat_accuracy')}",
            flush=True,
        )

        if args.early_stop_patience > 0 and tracking["epochs_since_improvement"] >= args.early_stop_patience:
            tracking["stopped_early"] = True
            tracking["stop_reason"] = (
                "early_stop_patience_exhausted"
                f"(patience={args.early_stop_patience}, "
                f"min_delta={args.early_stop_min_delta}, "
                f"epochs_since_improvement={tracking['epochs_since_improvement']})"
            )
            print(
                f"[early-stop] dataset={dataset} split={split_idx} epoch={epoch_idx} "
                f"best_epoch={tracking['best_epoch']} selection_split={args.selection_split}",
                flush=True,
            )
            save_resume_checkpoint(epoch_idx)
            save_progress_report()
            return True

        save_resume_checkpoint(epoch_idx)
        save_progress_report()
        return False

    print(
        f"[fit] dataset={dataset} split={split_idx} train_shape={tuple(train_x_t.shape)} "
        f"valid_shape={tuple(valid_x_t.shape)} test_shape={tuple(test_x_t.shape)}",
        flush=True,
    )
    imputer.fit(train_x_t, epoch_end_callback=on_epoch_end, resume_state=resume_state)

    if best_ckpt_path.exists():
        best_checkpoint = torch.load(best_ckpt_path, map_location=imputer.device, weights_only=False)
        imputer.load_state_dict(best_checkpoint["model_state_dict"])
        if best_checkpoint.get("norm_parameters") is not None:
            imputer.norm_parameters = best_checkpoint["norm_parameters"]

        best_train_pred_raw = imputer.transform(train_x_t).detach().cpu().numpy()
        best_valid_pred_raw = imputer.transform(valid_x_t).detach().cpu().numpy()
        best_test_pred_raw = imputer.transform(test_x_t).detach().cpu().numpy()
        best_train_metrics = evaluate_predictions(bundle, best_train_pred_raw, split="train")
        best_valid_metrics = evaluate_predictions(bundle, best_valid_pred_raw, split="valid")
        best_test_metrics = evaluate_predictions(bundle, best_test_pred_raw, split="test")
        final_best_score = _score_from_metrics(
            best_valid_metrics if args.selection_split == "valid" else best_test_metrics
        )

        tracking["best_train_pred_raw"] = best_train_pred_raw
        tracking["best_valid_pred_raw"] = best_valid_pred_raw
        tracking["best_test_pred_raw"] = best_test_pred_raw
        tracking["best_score"] = final_best_score
        tracking["best_epoch"] = best_checkpoint.get("epoch")
        tracking["best_report"] = {
            "epoch": best_checkpoint.get("epoch"),
            "score": final_best_score,
            "selection_split": args.selection_split,
            "train_metrics": best_train_metrics,
            "valid_metrics": best_valid_metrics,
            "test_metrics": best_test_metrics,
        }

    tracking["latest_metrics"] = tracking["best_report"] or tracking["latest_metrics"]
    np.save(run_dir / "train_pred_raw.npy", tracking["best_train_pred_raw"])
    np.save(run_dir / "valid_pred_raw.npy", tracking["best_valid_pred_raw"])
    np.save(run_dir / "test_pred_raw.npy", tracking["best_test_pred_raw"])
    tracking["completed"] = True
    save_resume_checkpoint(len(imputer.training_history))
    save_progress_report()

    del imputer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return build_report_payload()


def main() -> None:
    args = parse_args()
    benchmark_summary = {}
    run_tag_slug = _slugify_tag(args.run_tag)
    summary_root = args.output_root / "_summaries"
    csv_root = summary_root / f"{run_tag_slug}-csv"

    for dataset in args.datasets:
        print(f"\n=== Dataset: {dataset} ===", flush=True)
        reports = []
        for split_idx in args.splits:
            print(f"[run] dataset={dataset} split={split_idx} mask_type={args.mask_type} tag={args.run_tag}", flush=True)
            report = run_single_split(args, dataset, split_idx)
            reports.append(report)
            test_metrics = report["test_metrics"]
            print(
                f"[done] dataset={dataset} split={split_idx} "
                f"test_rmse={test_metrics.get('rmse')} "
                f"test_f1={test_metrics.get('cat_f1', test_metrics.get('bit_f1'))} "
                f"test_acc={test_metrics.get('cat_accuracy')}",
                flush=True,
            )

        dataset_summary = {
            "dataset": dataset,
            "selected_tag": args.run_tag,
            "aggregate_test_metrics": aggregate_metrics(reports),
            "per_split_reports": reports,
        }
        benchmark_summary[dataset] = dataset_summary
        if not args.no_summaries:
            mask_slug = args.mask_type.lower()
            save_json(summary_root / f"{dataset}_{mask_slug}_{run_tag_slug}_summary.json", dataset_summary)
            write_dataset_csv(csv_root / f"{dataset}_{mask_slug}_split_metrics.csv", reports, args.run_tag)

    if not args.no_summaries:
        mask_slug = args.mask_type.lower()
        save_json(summary_root / f"{mask_slug}_{run_tag_slug}_benchmark_summary.json", benchmark_summary)
        write_global_csv(csv_root / f"{mask_slug}_summary.csv", benchmark_summary)
        print("\nSaved aggregate benchmark summary.", flush=True)


if __name__ == "__main__":
    main()
