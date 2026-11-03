from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from config import build_config, get_arg_parser
from dataset import BenchmarkBundle
from trainer import TOSETrainer
from utils import enable_torch_performance_flags, load_json, resolve_device, save_json, seed_everything


def write_summary_csv(path: Path, reports: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset",
                "split_idx",
                "run_dir",
                "best_epoch",
                "test_rmse",
                "test_mae",
                "test_bit_f1",
                "test_cat_accuracy",
            ],
        )
        writer.writeheader()
        for report in reports:
            best = report.get("best_report") or {}
            test_metrics = best.get("test_metrics", {})
            writer.writerow(
                {
                    "dataset": report["dataset"],
                    "split_idx": report["split_idx"],
                    "run_dir": report["run_dir"],
                    "best_epoch": None if not best else best.get("epoch"),
                    "test_rmse": test_metrics.get("rmse"),
                    "test_mae": test_metrics.get("mae"),
                    "test_bit_f1": test_metrics.get("bit_f1"),
                    "test_cat_accuracy": test_metrics.get("cat_accuracy"),
                }
            )


def sample_standard_deviation(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    return float(np.std(finite, ddof=1))


def aggregate_reports(reports: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for report in reports:
        grouped.setdefault(report["dataset"], []).append(report)

    summary: dict[str, dict] = {}
    for dataset, dataset_reports in grouped.items():
        rmse_values = np.array(
            [report.get("best_report", {}).get("test_metrics", {}).get("rmse", np.nan) for report in dataset_reports],
            dtype=np.float64,
        )
        f1_values = np.array(
            [report.get("best_report", {}).get("test_metrics", {}).get("bit_f1", np.nan) for report in dataset_reports],
            dtype=np.float64,
        )
        acc_values = np.array(
            [
                report.get("best_report", {}).get("test_metrics", {}).get("cat_accuracy", np.nan)
                for report in dataset_reports
            ],
            dtype=np.float64,
        )
        summary[dataset] = {
            "num_runs": len(dataset_reports),
            "rmse_mean": None if np.isnan(rmse_values).all() else float(np.nanmean(rmse_values)),
            "rmse_std": sample_standard_deviation(rmse_values),
            "bit_f1_mean": None if np.isnan(f1_values).all() else float(np.nanmean(f1_values)),
            "bit_f1_std": sample_standard_deviation(f1_values),
            "cat_accuracy_mean": None if np.isnan(acc_values).all() else float(np.nanmean(acc_values)),
            "cat_accuracy_std": sample_standard_deviation(acc_values),
        }
    return summary


def _version_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    if name.startswith("v") and name[1:].isdigit():
        return (int(name[1:]), name)
    return (-1, name)


def task_base_dir(
    output_root: Path,
    dataset: str,
    ratio: str,
    mask_type: str,
    split_idx: int,
    run_tag: str,
) -> Path:
    return output_root / dataset / f"rate{ratio}" / mask_type / f"split_{split_idx}" / run_tag


def has_existing_output(
    output_root: Path,
    dataset: str,
    ratio: str,
    mask_type: str,
    split_idx: int,
    run_tag: str,
) -> bool:
    base_dir = task_base_dir(output_root, dataset, ratio, mask_type, split_idx, run_tag)
    return base_dir.exists() and any(path.is_dir() for path in base_dir.iterdir())


def find_existing_report(
    output_root: Path,
    dataset: str,
    ratio: str,
    mask_type: str,
    split_idx: int,
    run_tag: str,
) -> dict | None:
    base_dir = task_base_dir(output_root, dataset, ratio, mask_type, split_idx, run_tag)
    if not base_dir.exists():
        return None

    version_dirs = sorted(
        [path for path in base_dir.iterdir() if path.is_dir()],
        key=_version_sort_key,
        reverse=True,
    )
    for version_dir in version_dirs:
        report_path = version_dir / "report.json"
        if report_path.exists():
            try:
                return load_json(report_path)
            except (OSError, ValueError) as error:
                print(
                    f"[skip-unreadable-report] path={report_path} error={error}",
                    file=sys.stderr,
                )
    return None


def main() -> None:
    parser = get_arg_parser()
    args = parser.parse_args()

    if args.resume_checkpoint is not None and (len(args.datasets) != 1 or len(args.splits) != 1):
        raise ValueError("--resume-checkpoint requires exactly one dataset and one split.")

    reports = []
    for dataset in args.datasets:
        for split_idx in args.splits:
            if args.resume_checkpoint is None:
                existing_report = find_existing_report(
                    output_root=args.output_root,
                    dataset=dataset,
                    ratio=str(args.ratio),
                    mask_type=args.mask_type,
                    split_idx=split_idx,
                    run_tag=args.run_tag,
                )
                if existing_report is not None:
                    if args.skip_completed:
                        print(
                            f"[skip] dataset={dataset} split={split_idx} "
                            f"run_dir={existing_report.get('run_dir')}"
                        )
                        reports.append(existing_report)
                        continue
                    if not args.force_new_version:
                        raise RuntimeError(
                            "Existing completed report found for "
                            f"dataset={dataset} split={split_idx}. "
                            "Use --skip-completed to reuse it, --resume-checkpoint to continue a checkpoint, "
                            "or --force-new-version only when an intentional fresh rerun is needed."
                        )
                elif has_existing_output(
                    output_root=args.output_root,
                    dataset=dataset,
                    ratio=str(args.ratio),
                    mask_type=args.mask_type,
                    split_idx=split_idx,
                    run_tag=args.run_tag,
                ) and not args.force_new_version:
                    raise RuntimeError(
                        "Existing partial output found for "
                        f"dataset={dataset} split={split_idx}. "
                        "Resume from a checkpoint with --resume-checkpoint, or pass --force-new-version "
                        "only when intentionally discarding the partial run."
                    )

            config = build_config(args, dataset=dataset, split_idx=split_idx)
            seed_everything(config.seed)
            device = resolve_device(config.device)
            enable_torch_performance_flags(device)

            bundle = BenchmarkBundle(
                dataset_name=dataset,
                mask_type=config.mask_type,
                split_idx=split_idx,
                ratio=config.ratio,
                data_root=config.data_root,
                seed=config.seed,
                valid_max_fraction=config.valid_max_fraction,
            )
            trainer = TOSETrainer(config=config, bundle=bundle)
            reports.append(trainer.train())

    summary_root = args.output_root / "_summaries"
    save_json(summary_root / "tose_benchmark_summary.json", aggregate_reports(reports))
    save_json(summary_root / "tose_benchmark_reports.json", reports)
    write_summary_csv(summary_root / "tose_benchmark_reports.csv", reports)


if __name__ == "__main__":
    main()
