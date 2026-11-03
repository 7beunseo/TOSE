import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import optimize

try:
    from strict_masking import sample_exact_split_masks
except ImportError:  # pragma: no cover - package-style import
    from .strict_masking import sample_exact_split_masks

DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = DEFAULT_SOURCE_ROOT / "dataset"
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


@dataclass
class SplitIndices:
    train_idx: np.ndarray
    valid_idx: np.ndarray


@dataclass
class Mechanism:
    kind: str
    idxs_params: list[int]
    idxs_targets: list[int]
    centers: np.ndarray
    scales: np.ndarray
    coeffs: np.ndarray
    intercepts: np.ndarray
    expected_train_rate: np.ndarray
    expected_valid_rate: np.ndarray
    expected_test_rate: np.ndarray
    expected_max_gap: float
    tries: int


@dataclass
class MaskResult:
    train_mask: np.ndarray
    test_mask: np.ndarray
    train_rate: np.ndarray
    valid_rate: np.ndarray
    test_rate: np.ndarray
    max_gap: float
    tries: int


def to_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def import_original_modules(source_root: Path, data_root: Path):
    source_root = source_root.resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    import prepare_diffputer_dataset as prep  # type: ignore
    import normalize_diffputer_preprocessed as norm  # type: ignore

    prep.DATA_DIR = data_root
    prep.INFO_DIR = data_root / "Info"
    norm.DATA_DIR = data_root
    return prep, norm


def rate_name(p: float) -> str:
    return f"rate{int(round(p * 100))}"


def build_train_valid_indices(
    train_size: int,
    test_size: int,
    seed: int,
    valid_max_fraction: float,
) -> SplitIndices:
    if train_size <= 1 or valid_max_fraction <= 0.0:
        return SplitIndices(np.arange(train_size, dtype=np.int64), np.empty(0, dtype=np.int64))

    requested_valid_size = int(math.floor(train_size * valid_max_fraction))
    if requested_valid_size <= 0:
        requested_valid_size = 1
    valid_size = min(test_size, requested_valid_size, train_size - 1)
    if valid_size <= 0:
        return SplitIndices(np.arange(train_size, dtype=np.int64), np.empty(0, dtype=np.int64))

    rng = np.random.default_rng(int(seed) + 17)
    permutation = rng.permutation(train_size)
    valid_idx = np.sort(permutation[:valid_size]).astype(np.int64)
    train_idx = np.sort(permutation[valid_size:]).astype(np.int64)
    return SplitIndices(train_idx=train_idx, valid_idx=valid_idx)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def fit_intercept_1d(x: np.ndarray, coeff: float, target_p: float) -> float:
    target_p = float(np.clip(target_p, 1e-6, 1.0 - 1e-6))

    def objective(bias: float) -> float:
        return float(sigmoid_np(x * coeff + bias).mean() - target_p)

    try:
        return float(optimize.bisect(objective, -50.0, 50.0, maxiter=100))
    except ValueError:
        return float(np.log(target_p / (1.0 - target_p)))


def make_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers = np.nanmean(x, axis=0)
    centers[~np.isfinite(centers)] = 0.0
    scales = np.nanstd(x, axis=0)
    scales[~np.isfinite(scales)] = 1.0
    scales[scales < 1e-6] = 1.0
    return centers.astype(np.float64), scales.astype(np.float64)


def standardize(x: np.ndarray, centers: np.ndarray, scales: np.ndarray) -> np.ndarray:
    z = (x.astype(np.float64) - centers.reshape(1, -1)) / scales.reshape(1, -1)
    z[~np.isfinite(z)] = 0.0
    return z


def self_probabilities(z: np.ndarray, coeffs: np.ndarray, intercepts: np.ndarray) -> np.ndarray:
    return sigmoid_np(z * coeffs.reshape(1, -1) + intercepts.reshape(1, -1))


def mechanism_probabilities(x: np.ndarray, mechanism: Mechanism, base_missing_rate: float) -> np.ndarray:
    z = standardize(x.astype(np.float64), mechanism.centers, mechanism.scales)
    if mechanism.kind == "self_logistic":
        return self_probabilities(z, mechanism.coeffs, mechanism.intercepts)

    if mechanism.kind == "logistic_t2":
        n, d = z.shape
        probs = np.full((n, d), float(base_missing_rate), dtype=np.float64)
        idxs_params = np.asarray(mechanism.idxs_params, dtype=np.int64)
        idxs_targets = np.asarray(mechanism.idxs_targets, dtype=np.int64)
        if idxs_targets.size > 0:
            logits = z[:, idxs_params].dot(mechanism.coeffs) + mechanism.intercepts.reshape(1, -1)
            probs[:, idxs_targets] = sigmoid_np(logits)
        return probs

    raise ValueError(f"Unknown mechanism kind: {mechanism.kind}")


def draw_self_logistic_mechanism(
    train_x: np.ndarray,
    test_x: np.ndarray,
    split_indices: SplitIndices,
    missing_rate: float,
    rng: np.random.Generator,
    effect_scale: float,
    max_expected_gap: float,
    max_tries: int,
) -> Mechanism:
    strict_train_x = train_x[split_indices.train_idx].astype(np.float64)
    centers, scales = make_standardizer(strict_train_x)
    train_sub_z = standardize(strict_train_x, centers, scales)
    valid_z = standardize(train_x[split_indices.valid_idx].astype(np.float64), centers, scales)

    best: Mechanism | None = None
    best_score = float("inf")
    d = strict_train_x.shape[1]

    for attempt in range(1, max_tries + 1):
        coeffs = rng.choice(np.array([-1.0, 1.0]), size=d) * float(effect_scale)
        intercepts = np.array(
            [fit_intercept_1d(train_sub_z[:, j], coeffs[j], missing_rate) for j in range(d)],
            dtype=np.float64,
        )

        p_train = self_probabilities(train_sub_z, coeffs, intercepts)
        p_valid = self_probabilities(valid_z, coeffs, intercepts)
        train_rate = p_train.mean(axis=0)
        valid_rate = p_valid.mean(axis=0)
        max_gap = float(np.max(np.abs(train_rate - valid_rate)))
        score = max_gap
        if score < best_score:
            best_score = score
            best = Mechanism(
                kind="self_logistic",
                idxs_params=list(range(d)),
                idxs_targets=list(range(d)),
                centers=centers,
                scales=scales,
                coeffs=coeffs,
                intercepts=intercepts,
                expected_train_rate=train_rate,
                expected_valid_rate=valid_rate,
                expected_test_rate=np.empty(d, dtype=np.float64),
                expected_max_gap=max_gap,
                tries=attempt,
            )
        if max_gap <= max_expected_gap:
            break

    assert best is not None
    best.expected_test_rate = mechanism_probabilities(
        test_x.astype(np.float64), best, missing_rate
    ).mean(axis=0)
    return best


def fit_intercept_linear(linear_term: np.ndarray, target_p: float) -> float:
    target_p = float(np.clip(target_p, 1e-6, 1.0 - 1e-6))

    def objective(bias: float) -> float:
        return float(sigmoid_np(linear_term + bias).mean() - target_p)

    try:
        return float(optimize.bisect(objective, -50.0, 50.0, maxiter=100))
    except ValueError:
        return float(np.log(target_p / (1.0 - target_p)))


def draw_t2_logistic_mechanism(
    train_x: np.ndarray,
    test_x: np.ndarray,
    split_indices: SplitIndices,
    missing_rate: float,
    rng: np.random.Generator,
    p_params: float,
    effect_scale: float,
    max_expected_gap: float,
    max_tries: int,
) -> Mechanism:
    train_sub_x = train_x[split_indices.train_idx].astype(np.float64)
    valid_x = train_x[split_indices.valid_idx].astype(np.float64)
    centers, scales = make_standardizer(train_sub_x)
    train_sub_z = standardize(train_sub_x, centers, scales)
    d = train_sub_x.shape[1]

    best: Mechanism | None = None
    best_score = float("inf")

    for attempt in range(1, max_tries + 1):
        d_params = max(int(round(float(p_params) * d)), 1)
        if d_params >= d:
            d_params = max(d - 1, 1)
        idxs_params = np.sort(rng.choice(d, size=d_params, replace=False)).astype(np.int64)
        idxs_targets = np.asarray([j for j in range(d) if j not in set(idxs_params.tolist())], dtype=np.int64)
        if idxs_targets.size == 0:
            idxs_targets = np.arange(d, dtype=np.int64)

        coeffs = rng.normal(loc=0.0, scale=float(effect_scale), size=(idxs_params.size, idxs_targets.size))
        linear_train = train_sub_z[:, idxs_params].dot(coeffs)
        scale = np.nanstd(linear_train, axis=0, keepdims=True)
        scale[~np.isfinite(scale)] = 1.0
        scale[scale < 1e-6] = 1.0
        coeffs = coeffs / scale
        linear_train = train_sub_z[:, idxs_params].dot(coeffs)
        intercepts = np.array(
            [fit_intercept_linear(linear_train[:, j], missing_rate) for j in range(idxs_targets.size)],
            dtype=np.float64,
        )

        mechanism = Mechanism(
            kind="logistic_t2",
            idxs_params=idxs_params.tolist(),
            idxs_targets=idxs_targets.tolist(),
            centers=centers,
            scales=scales,
            coeffs=coeffs,
            intercepts=intercepts,
            expected_train_rate=np.array([]),
            expected_valid_rate=np.array([]),
            expected_test_rate=np.array([]),
            expected_max_gap=float("inf"),
            tries=attempt,
        )
        p_train = mechanism_probabilities(train_sub_x, mechanism, missing_rate)
        p_valid = mechanism_probabilities(valid_x, mechanism, missing_rate)
        train_rate = p_train.mean(axis=0)
        valid_rate = p_valid.mean(axis=0)
        max_gap = float(np.max(np.abs(train_rate - valid_rate)))
        mechanism.expected_train_rate = train_rate
        mechanism.expected_valid_rate = valid_rate
        mechanism.expected_max_gap = max_gap

        if max_gap < best_score:
            best_score = max_gap
            best = mechanism
        if max_gap <= max_expected_gap:
            best = mechanism
            break

    assert best is not None
    best.expected_test_rate = mechanism_probabilities(
        test_x.astype(np.float64), best, missing_rate
    ).mean(axis=0)
    return best


def rate_gap(train_rate: np.ndarray, valid_rate: np.ndarray, test_rate: np.ndarray) -> float:
    if train_rate.size == 0:
        return 0.0
    return float(
        np.max(
            np.maximum.reduce(
                [
                    np.abs(train_rate - valid_rate),
                    np.abs(train_rate - test_rate),
                    np.abs(valid_rate - test_rate),
                ]
            )
        )
    )


def rates_from_masks(train_mask: np.ndarray, test_mask: np.ndarray, split_indices: SplitIndices):
    train_rate = train_mask[split_indices.train_idx].mean(axis=0)
    valid_rate = train_mask[split_indices.valid_idx].mean(axis=0)
    test_rate = test_mask.mean(axis=0)
    return train_rate, valid_rate, test_rate


def min_counts_ok(
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    split_indices: SplitIndices,
    min_observed_rate: float,
    min_missing_rate: float,
    min_count: int,
) -> bool:
    splits = [
        train_mask[split_indices.train_idx],
        train_mask[split_indices.valid_idx],
        test_mask,
    ]
    for mask in splits:
        n = mask.shape[0]
        min_observed = min(n, max(int(math.ceil(n * min_observed_rate)), int(min_count)))
        min_missing = min(n, max(int(math.ceil(n * min_missing_rate)), int(min_count)))
        observed_count = (~mask).sum(axis=0)
        missing_count = mask.sum(axis=0)
        if np.any(observed_count < min_observed):
            return False
        if np.any(missing_count < min_missing):
            return False
    return True


def sample_masks_with_rejection(
    train_x: np.ndarray,
    test_x: np.ndarray,
    mechanism: Mechanism,
    split_indices: SplitIndices,
    rng: np.random.Generator,
    missing_rate: float,
    max_rate_gap: float,
    min_observed_rate: float,
    min_missing_rate: float,
    min_count: int,
    max_tries: int,
) -> MaskResult:
    p_train = mechanism_probabilities(train_x.astype(np.float64), mechanism, missing_rate)
    p_test = mechanism_probabilities(test_x.astype(np.float64), mechanism, missing_rate)
    return sample_exact_split_masks(
        train_probabilities=p_train,
        test_probabilities=p_test,
        train_indices=split_indices.train_idx,
        valid_indices=split_indices.valid_idx,
        missing_columns=np.arange(train_x.shape[1], dtype=np.int64),
        overall_missing_rate=missing_rate,
        max_train_valid_gap=max_rate_gap,
        min_observed_rate=min_observed_rate,
        min_missing_rate=min_missing_rate,
        min_count=min_count,
        max_tries=max_tries,
        rng=rng,
    )


def save_mask_metadata(
    save_dir: Path,
    split_idx: int,
    dataset: str,
    mask_name: str,
    missing_rate: float,
    mechanism: Mechanism,
    mask_result: MaskResult,
    args: argparse.Namespace,
) -> None:
    payload = {
        "dataset": dataset,
        "mask_type": mask_name,
        "split_idx": split_idx,
        "missing_rate": missing_rate,
        "seed": args.seed,
        "mask_kind": args.mask_kind,
        "valid_max_fraction": args.valid_max_fraction,
        "p_params": args.p_params,
        "effect_scale": args.effect_scale,
        "max_expected_gap": args.max_expected_gap,
        "max_rate_gap": args.max_rate_gap,
        "strict_split_files": bool(args.strict_split_files),
        "mechanism_tries": mechanism.tries,
        "sample_tries": mask_result.tries,
        "expected_max_gap": mechanism.expected_max_gap,
        "sampled_max_gap": mask_result.max_gap,
        "expected_train_rate": mechanism.expected_train_rate.tolist(),
        "expected_valid_rate": mechanism.expected_valid_rate.tolist(),
        "expected_test_rate": mechanism.expected_test_rate.tolist(),
        "sampled_train_rate": mask_result.train_rate.tolist(),
        "sampled_valid_rate": mask_result.valid_rate.tolist(),
        "sampled_test_rate": mask_result.test_rate.tolist(),
        "idxs_params": mechanism.idxs_params,
        "idxs_targets": mechanism.idxs_targets,
        "centers": mechanism.centers.tolist(),
        "scales": mechanism.scales.tolist(),
        "coeffs": mechanism.coeffs.tolist(),
        "intercepts": mechanism.intercepts.tolist(),
        "notes": [
            "Missingness mechanism parameters are fixed before train/valid/test mask sampling.",
            "The same mechanism is applied to train/valid/test.",
            "Masks are sampled independently and accepted only when split feature rates are sufficiently similar.",
            "When strict_split_files is true, valid is stored explicitly and normalization uses train-observed values only.",
        ],
    }
    with open(save_dir / f"mechanism_{split_idx}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_strict_normalized_split(
    norm,
    args: argparse.Namespace,
    dataset: str,
    split_idx: int,
    split_indices: SplitIndices,
) -> Path:
    ratio = str(int(round(args.missing_rate * 100)))
    train_x, test_x, train_mask, test_mask, cat_bin_num = norm.load_diffputer_features(
        dataname=dataset,
        split_idx=split_idx,
        mask_type=args.mask_name,
        ratio=ratio,
    )

    train_idx = split_indices.train_idx.astype(np.int64)
    valid_idx = split_indices.valid_idx.astype(np.int64)
    strict_train_x = train_x[train_idx]
    valid_x = train_x[valid_idx]
    strict_train_mask = train_mask[train_idx]
    valid_mask = train_mask[valid_idx]

    # Strict mode excludes validation rows from normalization statistics.
    mean, std = norm.mean_std(strict_train_x, strict_train_mask)
    strict_train_x_norm = to_float32((strict_train_x - mean) / std / 2.0)
    valid_x_norm = to_float32((valid_x - mean) / std / 2.0)
    test_x_norm = to_float32((test_x - mean) / std / 2.0)

    strict_train_x_init = to_float32((1.0 - strict_train_mask.astype(np.float32)) * strict_train_x_norm)
    valid_x_init = to_float32((1.0 - valid_mask.astype(np.float32)) * valid_x_norm)
    test_x_init = to_float32((1.0 - test_mask.astype(np.float32)) * test_x_norm)

    save_dir = args.data_root / dataset / "normalized" / rate_name(args.missing_rate) / args.mask_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"split_{split_idx}.npz"

    np.savez_compressed(
        save_path,
        strict_split=np.array(True),
        train_indices=train_idx,
        valid_indices=valid_idx,
        train_x_norm=strict_train_x_norm,
        valid_x_norm=valid_x_norm,
        test_x_norm=test_x_norm,
        train_x_init=strict_train_x_init,
        valid_x_init=valid_x_init,
        test_x_init=test_x_init,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        train_mask=strict_train_mask.astype(bool),
        valid_mask=valid_mask.astype(bool),
        test_mask=test_mask.astype(bool),
        cat_bin_num=cat_bin_num.astype(np.int64),
    )

    split_payloads = {
        "train": {
            "x_norm": strict_train_x_norm,
            "x_init": strict_train_x_init,
            "mask": strict_train_mask.astype(bool),
            "indices": train_idx,
        },
        "valid": {
            "x_norm": valid_x_norm,
            "x_init": valid_x_init,
            "mask": valid_mask.astype(bool),
            "indices": valid_idx,
        },
        "test": {
            "x_norm": test_x_norm,
            "x_init": test_x_init,
            "mask": test_mask.astype(bool),
            "indices": np.arange(test_x.shape[0], dtype=np.int64),
        },
    }
    for split_name, payload in split_payloads.items():
        np.savez_compressed(
            save_dir / f"split_{split_idx}_{split_name}.npz",
            split=np.array(split_name),
            x_norm=payload["x_norm"],
            x_init=payload["x_init"],
            mask=payload["mask"],
            indices=payload["indices"],
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            cat_bin_num=cat_bin_num.astype(np.int64),
        )

    meta = {
        "dataset": dataset,
        "ratio": ratio,
        "mask_type": args.mask_name,
        "split_idx": split_idx,
        "strict_split": True,
        "normalization": "train_observed_only_after_valid_split",
        "train_shape": list(strict_train_x_norm.shape),
        "valid_shape": list(valid_x_norm.shape),
        "test_shape": list(test_x_norm.shape),
        "num_missing_train": int(strict_train_mask.sum()),
        "num_missing_valid": int(valid_mask.sum()),
        "num_missing_test": int(test_mask.sum()),
        "train_indices_path": f"split_{split_idx}.npz::train_indices",
        "valid_indices_path": f"split_{split_idx}.npz::valid_indices",
        "split_files": [
            f"split_{split_idx}_train.npz",
            f"split_{split_idx}_valid.npz",
            f"split_{split_idx}_test.npz",
        ],
    }
    with open(save_dir / f"split_{split_idx}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return save_path


def generate_one_dataset(args: argparse.Namespace, prep, norm, dataset: str) -> None:
    train_x, test_x = prep.load_dataset(dataset)
    train_x = np.asarray(train_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)
    folder_name = rate_name(args.missing_rate)
    save_dir = args.data_root / dataset / "masks" / folder_name / args.mask_name
    normalized_dir = args.data_root / dataset / "normalized" / folder_name / args.mask_name
    save_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    for split_idx in args.splits:
        train_mask_path = save_dir / f"train_mask_{split_idx}.npy"
        test_mask_path = save_dir / f"test_mask_{split_idx}.npy"
        if (
            train_mask_path.exists()
            and test_mask_path.exists()
            and not args.overwrite
            and not args.dry_run
        ):
            print(f"[skip] masks already exist: {train_mask_path}")
            continue

        seed = int(args.seed) + int(split_idx) * 100_003 + sum(ord(ch) for ch in dataset)
        rng = np.random.default_rng(seed)
        split_indices = build_train_valid_indices(
            train_size=train_x.shape[0],
            test_size=test_x.shape[0],
            seed=args.seed,
            valid_max_fraction=args.valid_max_fraction,
        )
        if args.mask_kind == "self_logistic":
            mechanism = draw_self_logistic_mechanism(
                train_x=train_x,
                test_x=test_x,
                split_indices=split_indices,
                missing_rate=args.missing_rate,
                rng=rng,
                effect_scale=args.effect_scale,
                max_expected_gap=args.max_expected_gap,
                max_tries=args.max_mechanism_tries,
            )
        elif args.mask_kind == "logistic_t2":
            mechanism = draw_t2_logistic_mechanism(
                train_x=train_x,
                test_x=test_x,
                split_indices=split_indices,
                missing_rate=args.missing_rate,
                rng=rng,
                p_params=args.p_params,
                effect_scale=args.effect_scale,
                max_expected_gap=args.max_expected_gap,
                max_tries=args.max_mechanism_tries,
            )
        else:
            raise ValueError(f"Unsupported mask_kind: {args.mask_kind}")
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
        )

        print(
            f"[v2] dataset={dataset} split={split_idx} "
            f"expected_gap={mechanism.expected_max_gap:.4f} sampled_gap={mask_result.max_gap:.4f} "
            f"mechanism_tries={mechanism.tries} sample_tries={mask_result.tries}"
        )
        if mechanism.expected_max_gap > args.max_expected_gap:
            print(
                f"[warn] dataset={dataset} split={split_idx} expected gap exceeded threshold "
                f"({mechanism.expected_max_gap:.4f} > {args.max_expected_gap:.4f}); best candidate was used."
            )
        if mask_result.max_gap > args.max_rate_gap:
            print(
                f"[warn] dataset={dataset} split={split_idx} sampled gap exceeded threshold "
                f"({mask_result.max_gap:.4f} > {args.max_rate_gap:.4f}); best sample was used."
            )

        if args.dry_run:
            continue

        np.save(train_mask_path, mask_result.train_mask)
        np.save(test_mask_path, mask_result.test_mask)
        if args.strict_split_files:
            np.save(save_dir / f"full_train_mask_{split_idx}.npy", mask_result.train_mask)
            np.save(save_dir / f"strict_train_mask_{split_idx}.npy", mask_result.train_mask[split_indices.train_idx])
            np.save(save_dir / f"valid_mask_{split_idx}.npy", mask_result.train_mask[split_indices.valid_idx])
        save_mask_metadata(
            save_dir=save_dir,
            split_idx=split_idx,
            dataset=dataset,
            mask_name=args.mask_name,
            missing_rate=args.missing_rate,
            mechanism=mechanism,
            mask_result=mask_result,
            args=args,
        )
        print(f"[saved] {train_mask_path}")
        print(f"[saved] {test_mask_path}")

        if args.normalize:
            if args.strict_split_files:
                npz_path = save_strict_normalized_split(
                    norm=norm,
                    args=args,
                    dataset=dataset,
                    split_idx=split_idx,
                    split_indices=split_indices,
                )
            else:
                npz_path = norm.normalize_one(dataset, split_idx, args.mask_name, str(int(round(args.missing_rate * 100))))
            print(f"[normalized] {npz_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create fixed-mechanism MNAR self-logistic v2 masks with similar "
            "train/valid/test feature-level missing rates."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--splits", nargs="*", type=int, default=[0])
    parser.add_argument("--mask-kind", choices=["self_logistic", "logistic_t2"], default="self_logistic")
    parser.add_argument("--mask-name", default=None)
    parser.add_argument("--missing-rate", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--valid-max-fraction", type=float, default=0.15)
    parser.add_argument("--p-params", type=float, default=0.30)
    parser.add_argument("--effect-scale", type=float, default=0.50)
    parser.add_argument("--max-expected-gap", type=float, default=0.08)
    parser.add_argument("--max-rate-gap", type=float, default=0.10)
    parser.add_argument("--min-observed-rate", type=float, default=0.02)
    parser.add_argument("--min-missing-rate", type=float, default=0.02)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--max-mechanism-tries", type=int, default=2000)
    parser.add_argument("--max-sample-tries", type=int, default=2000)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--strict-split-files", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.source_root = args.source_root.resolve()
    args.data_root = args.data_root.resolve()
    if args.mask_name is None:
        args.mask_name = "MNAR_self_logistic_v2" if args.mask_kind == "self_logistic" else "MNAR_logistic_T2_v2"
    prep, norm = import_original_modules(args.source_root, args.data_root)
    for dataset in args.datasets:
        generate_one_dataset(args, prep, norm, dataset)


if __name__ == "__main__":
    main()
