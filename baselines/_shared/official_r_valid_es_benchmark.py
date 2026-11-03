from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _load_gain_protocol_module() -> Any:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "GAIN" / "run_gain_native_categorical_benchmark.py",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location("official_r_valid_es_gain_protocol", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    raise FileNotFoundError("Could not locate the shared GAIN fair-protocol module.")


gain_protocol = _load_gain_protocol_module()

DEFAULT_DATASETS = gain_protocol.DEFAULT_DATASETS
NativeCategoricalBundle = gain_protocol.NativeCategoricalBundle
BaselinePreprocessor = gain_protocol.BaselinePreprocessor
evaluate_predictions = gain_protocol.evaluate_predictions

SHARED_DIR = Path(__file__).resolve().parent
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from baseline_selection import (  # noqa: E402
    configure_selection,
    normalized_score_with_components,
    selection_metadata,
)


METHOD_CONFIG = {
    "mice": {
        "model": "MICE-OfficialR-ValidES",
        "implementation": "official_R_package_mice_mids_newdata",
        "package": "mice",
        "official_github": "https://github.com/amices/mice",
        "fit_protocol": "official_mice_train_fit_mice_mids_newdata_valid_es",
        "categorical_protocol": "official_R_factor_native",
    },
    "missforest": {
        "model": "missForest-OfficialR-TrainOnly-ValidES",
        "implementation": "official_R_algorithm_train_only_adapter",
        "package": "missForest",
        "official_github": "https://github.com/stekhoven/missForest",
        "fit_protocol": "official_missforest_1.6.1_ranger_loop_train_only_valid_es",
        "categorical_protocol": "official_R_factor_native",
    },
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
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


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_rscript(explicit: str) -> str:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Rscript does not exist: {path}")
        return str(path)
    found = shutil.which("Rscript")
    if found:
        return found
    candidates = sorted(Path(r"C:\Program Files\R").glob(r"R-*\bin\Rscript.exe"), reverse=True)
    if candidates:
        return str(candidates[0])
    raise RuntimeError("Rscript was not found. Pass --rscript with the full Rscript.exe path.")


def _write_levels_csv(path: Path, cat_columns: list[str], vocabularies: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["column", "level"])
        for column, vocabulary in zip(cat_columns, vocabularies):
            for level in vocabulary:
                writer.writerow([column, str(level)])


def _write_input_csv(
    path: Path,
    num_values: np.ndarray,
    cat_values: np.ndarray,
    num_missing: np.ndarray,
    cat_missing: np.ndarray,
    num_columns: list[str],
    cat_columns: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_count = num_values.shape[0] if num_values.ndim == 2 else cat_values.shape[0]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(num_columns + cat_columns)
        for row_idx in range(row_count):
            row: list[str] = []
            for col_idx in range(len(num_columns)):
                row.append("" if num_missing[row_idx, col_idx] else f"{float(num_values[row_idx, col_idx]):.10g}")
            for col_idx in range(len(cat_columns)):
                row.append("" if cat_missing[row_idx, col_idx] else str(cat_values[row_idx, col_idx]))
            writer.writerow(row)


def _read_output_csv(
    path: Path,
    num_columns: list[str],
    cat_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    pred_num = np.empty((len(rows), len(num_columns)), dtype=np.float32)
    pred_cat = np.empty((len(rows), len(cat_columns)), dtype=object)
    for row_idx, row in enumerate(rows):
        for col_idx, column in enumerate(num_columns):
            try:
                pred_num[row_idx, col_idx] = float(row.get(column, ""))
            except Exception:
                pred_num[row_idx, col_idx] = np.nan
        for col_idx, column in enumerate(cat_columns):
            value = str(row.get(column, ""))
            pred_cat[row_idx, col_idx] = "" if value in {"", "NA", "NaN", "nan", "<NA>"} else value
    return pred_num, pred_cat


def _read_runtime_metadata(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        return {str(row["key"]): str(row["value"]) for row in csv.DictReader(handle)}


def _original_missing(raw_split: Any, columns: list[int]) -> np.ndarray:
    if not columns:
        return np.empty((raw_split.num_samples, 0), dtype=bool)
    return raw_split.original_missing_mask[:, columns].astype(bool)


class OfficialRValidationRunner:
    def __init__(
        self,
        args: argparse.Namespace,
        method: str,
        r_script: Path,
        dataset: str,
        split_idx: int,
    ) -> None:
        self.args = args
        self.method = method
        self.config = METHOD_CONFIG[method]
        self.r_script = r_script.resolve()
        self.dataset = dataset
        self.split_idx = int(split_idx)
        self.run_tag = args.run_tag or self._default_run_tag()
        self.run_dir = (
            Path(args.output_root)
            / dataset
            / f"rate{args.ratio}"
            / args.mask_type
            / f"split_{self.split_idx}"
            / self.run_tag
        )
        self.report_path = self.run_dir / "report.json"
        self.history_path = self.run_dir / "history.json"
        self.rscript = _find_rscript(args.rscript)

    def _default_run_tag(self) -> str:
        mode = "self" if "self" in self.args.mask_type.lower() else "t2"
        if self.method == "mice":
            return f"mice_official_r_valid_es_hard_{mode}_iter{self.args.max_iterations}_normsel"
        return f"missforest_official_r_train_only_valid_es_hard_{mode}_iter{self.args.max_iterations}_normsel"

    def is_completed(self) -> bool:
        if not self.report_path.exists():
            return False
        try:
            report = _load_json(self.report_path)
        except Exception:
            return False
        metrics = report.get("test_metrics")
        return (
            bool(report.get("completed"))
            and report.get("implementation") == self.config["implementation"]
            and report.get("fit_transform_protocol") == self.config["fit_protocol"]
            and report.get("checkpoint_selection") == "baseline_normalized"
            and isinstance(metrics, dict)
            and metrics.get("categorical_metric_level") == "column_prefixed_category_index"
            and metrics.get("categorical_f1_average") == "weighted_over_column_prefixed_category_indices"
        )

    def _phase_arrays(
        self,
        raw_split: Any,
        model_split: Any,
        preprocessor: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num_dim = len(preprocessor.num_col_idx)
        num_values = model_split.x_model[:, :num_dim].astype(np.float32)
        cat_values = raw_split.cat_raw.copy()
        num_missing = _original_missing(raw_split, preprocessor.num_col_idx)
        cat_missing = _original_missing(raw_split, preprocessor.cat_col_idx)
        return num_values, cat_values, num_missing, cat_missing

    def _run_r_phase(
        self,
        phase_name: str,
        train_raw: Any,
        train_model: Any,
        target_raw: Any,
        target_model: Any,
        preprocessor: Any,
        max_iterations: int,
    ) -> tuple[Path, list[str], list[str], dict[str, str]]:
        num_columns = [f"num_{idx}" for idx in range(len(preprocessor.num_col_idx))]
        cat_columns = [f"cat_{idx}" for idx in range(len(preprocessor.cat_col_idx))]
        train_arrays = self._phase_arrays(train_raw, train_model, preprocessor)
        target_arrays = self._phase_arrays(target_raw, target_model, preprocessor)
        combined = [np.concatenate([train_item, target_item], axis=0) for train_item, target_item in zip(train_arrays, target_arrays)]

        phase_dir = self.run_dir / "r_workspace" / phase_name
        input_csv = phase_dir / "input.csv"
        levels_csv = phase_dir / "levels.csv"
        output_dir = phase_dir / "iterations"
        _write_input_csv(
            input_csv,
            num_values=combined[0],
            cat_values=combined[1],
            num_missing=combined[2],
            cat_missing=combined[3],
            num_columns=num_columns,
            cat_columns=cat_columns,
        )
        _write_levels_csv(levels_csv, cat_columns, preprocessor.categorical.vocabularies_)
        output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            self.rscript,
            str(self.r_script),
            str(input_csv),
            str(output_dir),
            str(levels_csv),
            ",".join(cat_columns) if cat_columns else "__NONE__",
            str(train_model.num_samples),
            str(int(max_iterations)),
            str(int(self.args.seed + self.split_idx)),
            str(int(self.args.rf_estimators)),
            str(int(self.args.r_num_threads)),
            str(int(self.args.verbose)),
            str(self.args.mice_categorical_method),
        ]
        print(
            f"[r-phase-start] method={self.config['model']} dataset={self.dataset} split={self.split_idx} "
            f"phase={phase_name} train_rows={train_model.num_samples} target_rows={target_model.num_samples} "
            f"iterations={max_iterations}",
            flush=True,
        )
        completed = subprocess.run(command, cwd=str(self.run_dir), check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"R phase failed with exit={completed.returncode}: {' '.join(command)}"
            )
        metadata = _read_runtime_metadata(output_dir / "runtime_metadata.csv")
        return output_dir, num_columns, cat_columns, metadata

    def _category_fallbacks(self, bundle: Any, preprocessor: Any) -> list[str]:
        fallbacks: list[str] = []
        train_missing = _original_missing(bundle.train_raw, preprocessor.cat_col_idx)
        for local_idx, vocabulary in enumerate(preprocessor.categorical.vocabularies_):
            observed = bundle.train_raw.cat_raw[~train_missing[:, local_idx], local_idx].astype(str)
            if observed.size:
                values, counts = np.unique(observed, return_counts=True)
                fallbacks.append(str(values[int(np.argmax(counts))]))
            else:
                fallbacks.append(str(vocabulary[0]) if vocabulary else "__UNK__")
        return fallbacks

    def _assemble_prediction(
        self,
        raw_split: Any,
        model_split: Any,
        pred_num: np.ndarray,
        pred_cat: np.ndarray,
        preprocessor: Any,
        category_fallbacks: list[str],
    ) -> np.ndarray:
        num_dim = len(preprocessor.num_col_idx)
        cat_dim = len(preprocessor.cat_col_idx)
        if pred_num.shape != (model_split.num_samples, num_dim):
            raise ValueError(f"Unexpected numeric output shape: {pred_num.shape}")
        if pred_cat.shape != (model_split.num_samples, cat_dim):
            raise ValueError(f"Unexpected categorical output shape: {pred_cat.shape}")

        num_missing = _original_missing(raw_split, preprocessor.num_col_idx)
        if num_dim:
            missing_nonfinite = num_missing & ~np.isfinite(pred_num)
            if missing_nonfinite.any():
                raise RuntimeError(
                    f"R imputer left {int(missing_nonfinite.sum())} numeric missing predictions unresolved."
                )
            pred_num[~num_missing] = model_split.x_model[:, :num_dim][~num_missing]

        cat_missing = _original_missing(raw_split, preprocessor.cat_col_idx)
        if cat_dim:
            for local_idx, vocabulary in enumerate(preprocessor.categorical.vocabularies_):
                known = set(str(item) for item in vocabulary)
                fallback = category_fallbacks[local_idx]
                for row_idx in np.where(cat_missing[:, local_idx])[0]:
                    value = str(pred_cat[row_idx, local_idx])
                    if value not in known:
                        pred_cat[row_idx, local_idx] = fallback
                observed_rows = ~cat_missing[:, local_idx]
                pred_cat[observed_rows, local_idx] = raw_split.cat_raw[observed_rows, local_idx]
            pred_cat_model = preprocessor.categorical.transform(pred_cat)
        else:
            pred_cat_model = np.empty((model_split.num_samples, 0), dtype=np.float32)
        return np.concatenate([pred_num.astype(np.float32), pred_cat_model.astype(np.float32)], axis=1)

    def _read_iteration_predictions(
        self,
        output_dir: Path,
        iteration: int,
        num_columns: list[str],
        cat_columns: list[str],
        train_rows: int,
        train_raw: Any,
        train_model: Any,
        target_raw: Any,
        target_model: Any,
        preprocessor: Any,
        category_fallbacks: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        output_csv = output_dir / f"iteration_{iteration:03d}.csv"
        if not output_csv.exists():
            raise FileNotFoundError(f"Expected R iteration output was not created: {output_csv}")
        pred_num, pred_cat = _read_output_csv(output_csv, num_columns, cat_columns)
        expected_rows = train_rows + target_model.num_samples
        if pred_num.shape[0] != expected_rows:
            raise ValueError(f"Expected {expected_rows} rows in {output_csv}, found {pred_num.shape[0]}.")
        train_prediction = self._assemble_prediction(
            train_raw,
            train_model,
            pred_num[:train_rows].copy(),
            pred_cat[:train_rows].copy(),
            preprocessor,
            category_fallbacks,
        )
        target_prediction = self._assemble_prediction(
            target_raw,
            target_model,
            pred_num[train_rows:].copy(),
            pred_cat[train_rows:].copy(),
            preprocessor,
            category_fallbacks,
        )
        return train_prediction, target_prediction

    def run(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        selection_info = configure_selection(
            data_root=self.args.data_root,
            dataset=self.dataset,
            mask_type=self.args.mask_type,
            split_idx=self.split_idx,
            ratio=self.args.ratio,
            seed=self.args.seed,
            valid_max_fraction=self.args.valid_max_fraction,
        )
        bundle = NativeCategoricalBundle(
            dataset_name=self.dataset,
            mask_type=self.args.mask_type,
            split_idx=self.split_idx,
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
        numeric_verification = gain_protocol.verify_prepared_numeric_inputs(
            bundle, train_split, valid_split, test_split
        )
        category_fallbacks = self._category_fallbacks(bundle, preprocessor)
        fairness_log = bundle.fairness_log(preprocessor)
        fairness_log.update(
            {
                "numeric_input_verification": numeric_verification,
                "train_only_fit": True,
                "valid_used_for_iteration_selection_only": True,
                "test_used_during_selection": False,
                "test_evaluated_once_after_selection": True,
                "selection_protocol": "valid_iteration_early_stopping",
                "fit_transform_protocol": self.config["fit_protocol"],
                **selection_info,
            }
        )

        print(
            f"[start] method={self.config['model']} dataset={self.dataset} split={self.split_idx} "
            f"train={train_split.num_samples} valid={valid_split.num_samples} test={test_split.num_samples} "
            f"max_iterations={self.args.max_iterations} run_dir={self.run_dir}",
            flush=True,
        )
        valid_output, num_columns, cat_columns, runtime_metadata = self._run_r_phase(
            "train_valid_selection",
            bundle.train_raw,
            train_split,
            bundle.valid_raw,
            valid_split,
            preprocessor,
            self.args.max_iterations,
        )

        best_iteration: int | None = None
        best_score: float | None = None
        best_train_prediction: np.ndarray | None = None
        best_valid_prediction: np.ndarray | None = None
        best_train_metrics: dict[str, Any] | None = None
        best_valid_metrics: dict[str, Any] | None = None
        no_improve = 0
        history: list[dict[str, Any]] = []

        for iteration in range(1, self.args.max_iterations + 1):
            train_prediction, valid_prediction = self._read_iteration_predictions(
                valid_output,
                iteration,
                num_columns,
                cat_columns,
                train_split.num_samples,
                bundle.train_raw,
                train_split,
                bundle.valid_raw,
                valid_split,
                preprocessor,
                category_fallbacks,
            )
            train_metrics = evaluate_predictions(
                preprocessor, train_split, train_prediction, bundle.num_col_idx, bundle.cat_col_idx
            )
            valid_metrics = evaluate_predictions(
                preprocessor, valid_split, valid_prediction, bundle.num_col_idx, bundle.cat_col_idx
            )
            score, components = normalized_score_with_components(valid_metrics)
            improved = best_score is None or score < best_score - self.args.early_stop_min_delta
            if improved:
                best_iteration = iteration
                best_score = float(score)
                best_train_prediction = train_prediction.copy()
                best_valid_prediction = valid_prediction.copy()
                best_train_metrics = train_metrics
                best_valid_metrics = valid_metrics
                no_improve = 0
                np.save(self.run_dir / "best_train_pred_model.npy", best_train_prediction)
                np.save(self.run_dir / "best_valid_pred_model.npy", best_valid_prediction)
            else:
                no_improve += 1
            record = {
                "iteration": iteration,
                "score": float(score),
                "score_components": components,
                "improved": improved,
                "no_improve": no_improve,
                "best_iteration": best_iteration,
                "best_score": best_score,
                "train_metrics": train_metrics,
                "valid_metrics": valid_metrics,
                "elapsed_seconds": time.time() - started,
            }
            history.append(record)
            _save_json(self.history_path, history)
            print(
                f"[valid-es] method={self.config['model']} dataset={self.dataset} split={self.split_idx} "
                f"iteration={iteration}/{self.args.max_iterations} score={score:.6f} "
                f"valid_rmse={valid_metrics.get('rmse')} valid_f1={valid_metrics.get('f1')} "
                f"valid_acc={valid_metrics.get('accuracy')} improved={improved} no_improve={no_improve}",
                flush=True,
            )
            if self.args.early_stop_patience > 0 and no_improve >= self.args.early_stop_patience:
                break

        if (
            best_iteration is None
            or best_score is None
            or best_train_prediction is None
            or best_valid_prediction is None
            or best_train_metrics is None
            or best_valid_metrics is None
        ):
            raise RuntimeError("Validation selection did not produce a best iteration.")

        test_output, test_num_columns, test_cat_columns, test_runtime_metadata = self._run_r_phase(
            "train_test_final",
            bundle.train_raw,
            train_split,
            bundle.test_raw,
            test_split,
            preprocessor,
            best_iteration,
        )
        final_train_prediction, test_prediction = self._read_iteration_predictions(
            test_output,
            best_iteration,
            test_num_columns,
            test_cat_columns,
            train_split.num_samples,
            bundle.train_raw,
            train_split,
            bundle.test_raw,
            test_split,
            preprocessor,
            category_fallbacks,
        )
        test_metrics = evaluate_predictions(
            preprocessor, test_split, test_prediction, bundle.num_col_idx, bundle.cat_col_idx
        )
        max_train_phase_difference = float(
            np.max(np.abs(final_train_prediction - best_train_prediction))
        ) if final_train_prediction.size else 0.0

        np.save(self.run_dir / "train_pred_model.npy", best_train_prediction)
        np.save(self.run_dir / "valid_pred_model.npy", best_valid_prediction)
        np.save(self.run_dir / "test_pred_model.npy", test_prediction)
        selected_record = next(item for item in history if item["iteration"] == best_iteration)
        best_report = {
            "iteration": best_iteration,
            "epoch": best_iteration,
            "score": best_score,
            "selection_split": "valid",
            "score_components": selected_record["score_components"],
            "train_metrics": best_train_metrics,
            "valid_metrics": best_valid_metrics,
            "test_metrics": test_metrics,
        }
        method_note = (
            "Uses the official mice package. The chain is fitted on training rows only, then validation/test "
            "rows are imputed through the official mice.mids(newdata=...) path, which excludes new rows from "
            "conditional-model estimation."
            if self.method == "mice"
            else
            "The official missForest API has no ignore/newdata transform. This R adapter follows the "
            "missForest 1.6.1 ranger update loop but restricts every forest fit and initialization statistic "
            "to training rows; validation and test rows are prediction-only."
        )
        report = {
            "model": self.config["model"],
            "method": self.method,
            "implementation": self.config["implementation"],
            "official_package": self.config["package"],
            "official_github": self.config["official_github"],
            "official_r_runtime": runtime_metadata,
            "official_r_test_runtime": test_runtime_metadata,
            "official_implementation_note": method_note,
            "dataset": self.dataset,
            "mask_type": self.args.mask_type,
            "split_idx": self.split_idx,
            "ratio": str(self.args.ratio),
            "run_tag": self.run_tag,
            "run_dir": str(self.run_dir),
            "selection_split": "valid",
            "selection_protocol": "valid_iteration_early_stopping",
            **selection_metadata(),
            "best_epoch": best_iteration,
            "best_iteration": best_iteration,
            "best_score": best_score,
            "train_metrics": best_train_metrics,
            "valid_metrics": best_valid_metrics,
            "test_metrics": test_metrics,
            "best_report": best_report,
            "history": history,
            "history_path": str(self.history_path),
            "max_iterations": self.args.max_iterations,
            "early_stop_patience": self.args.early_stop_patience,
            "early_stop_min_delta": self.args.early_stop_min_delta,
            "last_iteration": int(history[-1]["iteration"]),
            "stopped_early": len(history) < self.args.max_iterations,
            "completed": True,
            "numeric_source": "prepared_npz_x_norm",
            "scaler_source": "prepared_npz_mean_std",
            "metric_space": "eunseo_normalized",
            "categorical_protocol": self.config["categorical_protocol"],
            "categorical_handling": {
                "model_input": "R_factor_with_train_vocabulary",
                "model_output": "R_factor_label",
                "evaluation_decode": "train_vocabulary_original_category_index",
                "f1_metric": "column_prefixed_category_index_weighted_f1",
                "uses_eunseo_codebook": False,
            },
            "fit_transform_protocol": self.config["fit_protocol"],
            "fit_scope": "train_rows_only",
            "valid_scope": "prediction_and_iteration_selection_only",
            "test_scope": "prediction_once_after_best_iteration_selection",
            "test_used_during_selection": False,
            "train_phase_max_abs_difference": max_train_phase_difference,
            "split_sizes": bundle.split_sizes(),
            "preprocessing": preprocessor.metadata(),
            "fairness_log": fairness_log,
            "r_script": str(self.r_script),
            "rscript_executable": self.rscript,
            "total_seconds": time.time() - started,
        }
        _save_json(self.run_dir / "fairness_log.json", fairness_log)
        _save_json(self.run_dir / "config.json", vars(self.args))
        _save_json(self.report_path, report)
        print(
            f"[done] method={self.config['model']} dataset={self.dataset} split={self.split_idx} "
            f"best_iteration={best_iteration} test_rmse={test_metrics.get('rmse')} "
            f"test_f1={test_metrics.get('f1')} test_acc={test_metrics.get('accuracy')}",
            flush=True,
        )
        return report


def _resolve_splits(args: argparse.Namespace) -> list[int]:
    if args.all_splits and args.splits:
        raise ValueError("Use either --all-splits or --splits, not both.")
    if args.all_splits:
        return list(range(10))
    if args.splits:
        return list(dict.fromkeys(int(item) for item in args.splits))
    return [0]


def _build_parser(method: str, default_output_root: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run {METHOD_CONFIG[method]['model']} with common Eunseo validation and metrics."
    )
    parser.add_argument("--rscript", default="")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[2] / "data"))
    parser.add_argument("--output-root", default=default_output_root)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--mask-type", required=True)
    parser.add_argument("--ratio", default="30")
    parser.add_argument("--splits", nargs="+", type=int, default=None)
    parser.add_argument("--all-splits", action="store_true")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--valid-max-fraction", type=float, default=0.15)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=45)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0002)
    parser.add_argument("--rf-estimators", type=int, default=100)
    parser.add_argument("--r-num-threads", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--mice-categorical-method", choices=["default", "cart"], default="default")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run_cli(method: str, r_script: Path, default_output_root: str) -> None:
    method = method.lower()
    if method not in METHOD_CONFIG:
        raise ValueError(f"Unsupported method: {method}")
    args = _build_parser(method, default_output_root).parse_args()
    if args.max_iterations <= 0:
        raise ValueError("--max-iterations must be positive.")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be non-negative.")
    if not r_script.exists():
        raise FileNotFoundError(f"R implementation not found: {r_script}")

    reports: list[dict[str, Any]] = []
    for split_idx in _resolve_splits(args):
        for dataset in args.datasets:
            run_args = copy.copy(args)
            runner = OfficialRValidationRunner(run_args, method, r_script, dataset, split_idx)
            if not args.overwrite and runner.is_completed():
                print(
                    f"[skip-completed] method={METHOD_CONFIG[method]['model']} dataset={dataset} "
                    f"split={split_idx} report={runner.report_path}",
                    flush=True,
                )
                report = _load_json(runner.report_path)
            else:
                report = runner.run()
            reports.append(report)

    summary_dir = Path(args.output_root) / "_summaries"
    summary_name = f"{method}_official_r_valid_es_reports.json"
    _save_json(summary_dir / summary_name, reports)
    print(f"[summary] {summary_dir / summary_name}", flush=True)
