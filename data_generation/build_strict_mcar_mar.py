from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from strict_masking import sample_exact_split_masks
except ImportError:  # pragma: no cover - package-style import
    from .strict_masking import sample_exact_split_masks


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

MCAR_MASK = "MCAR_v2_strict"
MAR_MASK = "MAR_logistic_v2_strict"


@dataclass
class StrictMechanism:
    kind: str
    predictor_indices: np.ndarray
    target_indices: np.ndarray
    centers: np.ndarray
    scales: np.ndarray
    coeffs: np.ndarray
    intercepts: np.ndarray
    expected_train_rate: np.ndarray
    expected_valid_rate: np.ndarray
    expected_test_rate: np.ndarray
    expected_max_gap: float
    tries: int
    target_missing_rate: float


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Generate strict MCAR and logistic-MAR masks using the same split, "
            "normalization, and file protocol as the TOSE strict benchmark."
        )
    )
    parser.add_argument("--support-dir", type=Path, default=here)
    parser.add_argument("--data-root", type=Path, default=here.parent / "data" / "hard_mnar")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--splits", nargs="*", type=int, default=list(range(10)))
    parser.add_argument("--mask-types", nargs="*", choices=["mcar", "mar"], default=["mcar", "mar"])
    parser.add_argument("--missing-rate", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--valid-max-fraction", type=float, default=0.15)
    parser.add_argument("--mar-observed-fraction", type=float, default=0.30)
    parser.add_argument("--mar-logit-strength", type=float, default=1.00)
    parser.add_argument("--max-expected-gap", type=float, default=0.08)
    parser.add_argument("--max-rate-gap", type=float, default=0.10)
    parser.add_argument("--min-observed-rate", type=float, default=0.02)
    parser.add_argument("--min-missing-rate", type=float, default=0.02)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--max-mechanism-tries", type=int, default=2000)
    parser.add_argument("--max-sample-tries", type=int, default=2000)
    parser.add_argument("--overwrite-masks", action="store_true")
    return parser.parse_args()


def import_generation_modules(support_dir: Path, data_root: Path):
    support_dir = support_dir.resolve()
    if str(support_dir) not in sys.path:
        sys.path.insert(0, str(support_dir))

    import prepare_diffputer_dataset as prep  # type: ignore
    import normalize_diffputer_preprocessed as norm  # type: ignore
    import dataset_v2 as v2  # type: ignore

    prep.DATA_DIR = data_root
    prep.INFO_DIR = data_root / "Info"
    norm.DATA_DIR = data_root
    return prep, norm, v2


def mechanism_probabilities(
    x: np.ndarray,
    mechanism: StrictMechanism,
    missing_rate: float,
    v2,
) -> np.ndarray:
    n, d = x.shape
    if mechanism.kind == "mcar":
        return np.full((n, d), float(missing_rate), dtype=np.float64)

    if mechanism.kind != "mar_logistic":
        raise ValueError(f"Unsupported mechanism kind: {mechanism.kind}")

    probabilities = np.zeros((n, d), dtype=np.float64)
    z = v2.standardize(x.astype(np.float64), mechanism.centers, mechanism.scales)
    logits = (
        z[:, mechanism.predictor_indices].dot(mechanism.coeffs)
        + mechanism.intercepts.reshape(1, -1)
    )
    probabilities[:, mechanism.target_indices] = v2.sigmoid_np(logits)
    return probabilities


def draw_mcar_mechanism(
    train_x: np.ndarray,
    test_x: np.ndarray,
    split_indices,
    missing_rate: float,
) -> StrictMechanism:
    d = train_x.shape[1]
    expected = np.full(d, float(missing_rate), dtype=np.float64)
    return StrictMechanism(
        kind="mcar",
        predictor_indices=np.empty(0, dtype=np.int64),
        target_indices=np.arange(d, dtype=np.int64),
        centers=np.zeros(d, dtype=np.float64),
        scales=np.ones(d, dtype=np.float64),
        coeffs=np.empty((0, d), dtype=np.float64),
        intercepts=np.full(d, np.log(missing_rate / (1.0 - missing_rate)), dtype=np.float64),
        expected_train_rate=expected.copy(),
        expected_valid_rate=expected.copy(),
        expected_test_rate=expected.copy(),
        expected_max_gap=0.0,
        tries=1,
        target_missing_rate=float(missing_rate),
    )


def draw_mar_mechanism(
    train_x: np.ndarray,
    test_x: np.ndarray,
    split_indices,
    missing_rate: float,
    observed_fraction: float,
    logit_strength: float,
    max_expected_gap: float,
    max_tries: int,
    rng: np.random.Generator,
    v2,
) -> StrictMechanism:
    strict_train_x = train_x[split_indices.train_idx].astype(np.float64)
    valid_x = train_x[split_indices.valid_idx].astype(np.float64)
    centers, scales = v2.make_standardizer(strict_train_x)
    strict_train_z = v2.standardize(strict_train_x, centers, scales)
    d = strict_train_x.shape[1]
    if d < 2:
        raise ValueError("Logistic MAR requires at least two logical features")

    predictor_count = max(int(round(float(observed_fraction) * d)), 1)
    predictor_count = min(predictor_count, d - 1)
    target_count = d - predictor_count
    target_missing_rate = float(missing_rate) * d / target_count
    if target_missing_rate >= 1.0:
        raise ValueError(
            "MAR observed fraction leaves too few target features for the requested overall missing rate"
        )

    best: StrictMechanism | None = None
    best_score = float("inf")
    all_indices = np.arange(d, dtype=np.int64)
    for attempt in range(1, max_tries + 1):
        predictor_indices = np.sort(
            rng.choice(d, size=predictor_count, replace=False)
        ).astype(np.int64)
        target_indices = np.setdiff1d(all_indices, predictor_indices, assume_unique=True)

        coeffs = rng.normal(size=(predictor_count, target_count)).astype(np.float64)
        linear_train = strict_train_z[:, predictor_indices].dot(coeffs)
        linear_scale = np.nanstd(linear_train, axis=0)
        linear_scale[~np.isfinite(linear_scale)] = 1.0
        linear_scale[linear_scale < 1e-6] = 1.0
        coeffs = coeffs / linear_scale.reshape(1, -1) * float(logit_strength)
        linear_train = strict_train_z[:, predictor_indices].dot(coeffs)
        intercepts = np.asarray(
            [
                v2.fit_intercept_linear(linear_train[:, column], target_missing_rate)
                for column in range(target_count)
            ],
            dtype=np.float64,
        )

        candidate = StrictMechanism(
            kind="mar_logistic",
            predictor_indices=predictor_indices,
            target_indices=target_indices,
            centers=centers,
            scales=scales,
            coeffs=coeffs,
            intercepts=intercepts,
            expected_train_rate=np.empty(d, dtype=np.float64),
            expected_valid_rate=np.empty(d, dtype=np.float64),
            expected_test_rate=np.empty(d, dtype=np.float64),
            expected_max_gap=float("inf"),
            tries=attempt,
            target_missing_rate=target_missing_rate,
        )
        train_probabilities = mechanism_probabilities(
            strict_train_x, candidate, missing_rate, v2
        )
        valid_probabilities = mechanism_probabilities(valid_x, candidate, missing_rate, v2)
        candidate.expected_train_rate = train_probabilities.mean(axis=0)
        candidate.expected_valid_rate = valid_probabilities.mean(axis=0)
        candidate.expected_max_gap = float(
            np.max(
                np.abs(
                    candidate.expected_train_rate - candidate.expected_valid_rate
                )
            )
        )
        if candidate.expected_max_gap < best_score:
            best = candidate
            best_score = candidate.expected_max_gap
        if candidate.expected_max_gap <= max_expected_gap:
            best = candidate
            break

    assert best is not None
    best.expected_test_rate = mechanism_probabilities(
        test_x, best, missing_rate, v2
    ).mean(axis=0)
    return best


def min_counts_ok(
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    split_indices,
    missing_columns: np.ndarray,
    min_observed_rate: float,
    min_missing_rate: float,
    min_count: int,
) -> bool:
    for mask in (
        train_mask[split_indices.train_idx],
        train_mask[split_indices.valid_idx],
        test_mask,
    ):
        n = mask.shape[0]
        min_observed = min(n, max(int(np.ceil(n * min_observed_rate)), int(min_count)))
        min_missing = min(n, max(int(np.ceil(n * min_missing_rate)), int(min_count)))
        if np.any((~mask).sum(axis=0) < min_observed):
            return False
        if np.any(mask[:, missing_columns].sum(axis=0) < min_missing):
            return False
    return True


def sample_masks_with_rejection(
    train_x: np.ndarray,
    test_x: np.ndarray,
    mechanism: StrictMechanism,
    split_indices,
    rng: np.random.Generator,
    missing_rate: float,
    max_rate_gap: float,
    min_observed_rate: float,
    min_missing_rate: float,
    min_count: int,
    max_tries: int,
    v2,
):
    train_probabilities = mechanism_probabilities(train_x, mechanism, missing_rate, v2)
    test_probabilities = mechanism_probabilities(test_x, mechanism, missing_rate, v2)
    return sample_exact_split_masks(
        train_probabilities=train_probabilities,
        test_probabilities=test_probabilities,
        train_indices=split_indices.train_idx,
        valid_indices=split_indices.valid_idx,
        missing_columns=mechanism.target_indices,
        overall_missing_rate=missing_rate,
        max_train_valid_gap=max_rate_gap,
        min_observed_rate=min_observed_rate,
        min_missing_rate=min_missing_rate,
        min_count=min_count,
        max_tries=max_tries,
        rng=rng,
    )


def metadata_payload(
    args: argparse.Namespace,
    dataset: str,
    split_idx: int,
    mask_name: str,
    effective_seed: int,
    mechanism: StrictMechanism,
    mask_result,
) -> dict[str, object]:
    notes = [
        "The supervised target is excluded before missingness generation.",
        "Mechanism standardization and calibration use strict-train rows only.",
        "One fixed mechanism is applied to strict-train, validation, and test.",
        "Each partition uses the nearest feasible exact overall missing-cell count.",
        "Test values and test rates are not used to select mechanism candidates.",
        "Normalization statistics use observed strict-train values only.",
    ]
    if mechanism.kind == "mar_logistic":
        notes.append("MAR predictor features remain fully observed; only target features are amputed.")
    else:
        notes.append("MCAR uses the same missing probability for every logical cell.")

    return {
        "dataset": dataset,
        "mask_type": mask_name,
        "split_idx": split_idx,
        "missing_rate": args.missing_rate,
        "seed": args.seed,
        "effective_seed": effective_seed,
        "mask_kind": mechanism.kind,
        "valid_max_fraction": args.valid_max_fraction,
        "mar_observed_fraction": args.mar_observed_fraction,
        "mar_logit_strength": args.mar_logit_strength,
        "max_expected_gap": args.max_expected_gap,
        "max_rate_gap": args.max_rate_gap,
        "min_observed_rate": args.min_observed_rate,
        "min_missing_rate": args.min_missing_rate,
        "min_count": args.min_count,
        "max_mechanism_tries": args.max_mechanism_tries,
        "max_sample_tries": args.max_sample_tries,
        "strict_split_files": True,
        "strict_protocol_version": "tose_public_v1_feature_only_train_fit_exact_rate",
        "normalization": "strict_train_observed_only_after_valid_split",
        "mechanism_tries": mechanism.tries,
        "sample_tries": mask_result.tries,
        "target_missing_rate": mechanism.target_missing_rate,
        "expected_max_gap": mechanism.expected_max_gap,
        "sampled_max_gap": mask_result.max_gap,
        "sampled_train_valid_max_gap": mask_result.train_valid_max_gap,
        "expected_train_rate": mechanism.expected_train_rate.tolist(),
        "expected_valid_rate": mechanism.expected_valid_rate.tolist(),
        "expected_test_rate": mechanism.expected_test_rate.tolist(),
        "sampled_train_rate": mask_result.train_rate.tolist(),
        "sampled_valid_rate": mask_result.valid_rate.tolist(),
        "sampled_test_rate": mask_result.test_rate.tolist(),
        "predictor_indices": mechanism.predictor_indices.tolist(),
        "target_indices": mechanism.target_indices.tolist(),
        "centers": mechanism.centers.tolist(),
        "scales": mechanism.scales.tolist(),
        "coeffs": mechanism.coeffs.tolist(),
        "intercepts": mechanism.intercepts.tolist(),
        "overall_train_rate": float(
            mask_result.train_mask.mean()
        ),
        "overall_strict_train_rate": float(
            mask_result.train_mask.mean()
        ),
        "overall_test_rate": float(mask_result.test_mask.mean()),
        "notes": notes,
    }


def expected_outputs(data_root: Path, dataset: str, mask_name: str, split_idx: int, rate_name: str) -> list[Path]:
    mask_dir = data_root / dataset / "masks" / rate_name / mask_name
    normalized_dir = data_root / dataset / "normalized" / rate_name / mask_name
    return [
        mask_dir / f"train_mask_{split_idx}.npy",
        mask_dir / f"test_mask_{split_idx}.npy",
        mask_dir / f"full_train_mask_{split_idx}.npy",
        mask_dir / f"strict_train_mask_{split_idx}.npy",
        mask_dir / f"valid_mask_{split_idx}.npy",
        mask_dir / f"mechanism_{split_idx}.json",
        normalized_dir / f"split_{split_idx}.npz",
        normalized_dir / f"split_{split_idx}_train.npz",
        normalized_dir / f"split_{split_idx}_valid.npz",
        normalized_dir / f"split_{split_idx}_test.npz",
        normalized_dir / f"split_{split_idx}.json",
    ]


def generate_one(
    args: argparse.Namespace,
    prep,
    norm,
    v2,
    dataset: str,
    split_idx: int,
    mechanism_name: str,
) -> None:
    mask_name = MCAR_MASK if mechanism_name == "mcar" else MAR_MASK
    rate_dir = v2.rate_name(args.missing_rate)
    outputs = expected_outputs(args.data_root, dataset, mask_name, split_idx, rate_dir)
    if all(path.exists() for path in outputs) and not args.overwrite_masks:
        print(f"[skip-complete] dataset={dataset} mask={mask_name} split={split_idx}")
        return

    train_x, test_x = prep.load_dataset(dataset)
    train_x = np.asarray(train_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)
    split_indices = v2.build_train_valid_indices(
        train_size=train_x.shape[0],
        test_size=test_x.shape[0],
        seed=args.seed,
        valid_max_fraction=args.valid_max_fraction,
    )
    effective_seed = int(args.seed) + int(split_idx) * 100_003 + sum(ord(ch) for ch in dataset)
    rng = np.random.default_rng(effective_seed)

    if mechanism_name == "mcar":
        mechanism = draw_mcar_mechanism(
            train_x=train_x,
            test_x=test_x,
            split_indices=split_indices,
            missing_rate=args.missing_rate,
        )
    else:
        mechanism = draw_mar_mechanism(
            train_x=train_x,
            test_x=test_x,
            split_indices=split_indices,
            missing_rate=args.missing_rate,
            observed_fraction=args.mar_observed_fraction,
            logit_strength=args.mar_logit_strength,
            max_expected_gap=args.max_expected_gap,
            max_tries=args.max_mechanism_tries,
            rng=rng,
            v2=v2,
        )

    mask_result = sample_masks_with_rejection(
        train_x=train_x,
        test_x=test_x,
        mechanism=mechanism,
        split_indices=split_indices,
        rng=rng,
        missing_rate=args.missing_rate,
        max_rate_gap=args.max_rate_gap,
        min_observed_rate=args.min_observed_rate,
        min_missing_rate=args.min_missing_rate,
        min_count=args.min_count,
        max_tries=args.max_sample_tries,
        v2=v2,
    )

    mask_dir = args.data_root / dataset / "masks" / rate_dir / mask_name
    mask_dir.mkdir(parents=True, exist_ok=True)
    np.save(mask_dir / f"train_mask_{split_idx}.npy", mask_result.train_mask)
    np.save(mask_dir / f"test_mask_{split_idx}.npy", mask_result.test_mask)
    np.save(mask_dir / f"full_train_mask_{split_idx}.npy", mask_result.train_mask)
    np.save(
        mask_dir / f"strict_train_mask_{split_idx}.npy",
        mask_result.train_mask[split_indices.train_idx],
    )
    np.save(
        mask_dir / f"valid_mask_{split_idx}.npy",
        mask_result.train_mask[split_indices.valid_idx],
    )
    payload = metadata_payload(
        args=args,
        dataset=dataset,
        split_idx=split_idx,
        mask_name=mask_name,
        effective_seed=effective_seed,
        mechanism=mechanism,
        mask_result=mask_result,
    )
    payload["overall_strict_train_rate"] = float(
        mask_result.train_mask[split_indices.train_idx].mean()
    )
    payload["overall_valid_rate"] = float(
        mask_result.train_mask[split_indices.valid_idx].mean()
    )
    (mask_dir / f"mechanism_{split_idx}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    save_args = argparse.Namespace(
        data_root=args.data_root,
        missing_rate=args.missing_rate,
        mask_name=mask_name,
    )
    normalized_path = v2.save_strict_normalized_split(
        norm=norm,
        args=save_args,
        dataset=dataset,
        split_idx=split_idx,
        split_indices=split_indices,
    )
    print(
        f"[generated] dataset={dataset} mask={mask_name} split={split_idx} "
        f"rate={payload['overall_strict_train_rate']:.4f}/"
        f"{payload['overall_valid_rate']:.4f}/{payload['overall_test_rate']:.4f} "
        f"gap={mask_result.max_gap:.4f} normalized={normalized_path}"
    )


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 < args.missing_rate < 1.0):
        raise ValueError("missing_rate must be in (0, 1)")
    if not (0.0 < args.mar_observed_fraction < 1.0):
        raise ValueError("mar_observed_fraction must be in (0, 1)")
    if args.mar_logit_strength <= 0.0:
        raise ValueError("mar_logit_strength must be positive")
    unknown = sorted(set(args.datasets) - set(DEFAULT_DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    invalid_splits = [split for split in args.splits if split < 0]
    if invalid_splits:
        raise ValueError(f"Invalid splits: {invalid_splits}")


def main() -> None:
    args = parse_args()
    args.support_dir = args.support_dir.resolve()
    args.data_root = args.data_root.resolve()
    args.datasets = list(dict.fromkeys(args.datasets))
    args.splits = sorted(set(args.splits))
    args.mask_types = list(dict.fromkeys(args.mask_types))
    validate_args(args)

    prep, norm, v2 = import_generation_modules(args.support_dir, args.data_root)
    for dataset in args.datasets:
        for split_idx in args.splits:
            for mechanism_name in args.mask_types:
                generate_one(
                    args=args,
                    prep=prep,
                    norm=norm,
                    v2=v2,
                    dataset=dataset,
                    split_idx=split_idx,
                    mechanism_name=mechanism_name,
                )
    print("[done] strict MCAR/MAR generation complete")


if __name__ == "__main__":
    main()
