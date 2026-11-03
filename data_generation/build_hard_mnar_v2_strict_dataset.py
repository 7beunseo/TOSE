from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from urllib import request

import numpy as np


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

T2_HARD_MASK = "MNAR_logistic_T2_v2_strict_hard"
SELF_HARD_MASK = "MNAR_self_logistic_v2_strict_hard"
GERMAN_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build the strict MNAR-T2 benchmark for TOSE."
    )
    parser.add_argument("--support-dir", type=Path, default=here)
    parser.add_argument("--output-root", type=Path, default=here.parent / "data" / "hard_mnar")
    parser.add_argument("--copy-base-from", type=Path, default=None)
    parser.add_argument("--prepare-base", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--splits", nargs="*", type=int, default=list(range(10)))
    parser.add_argument("--mask-types", nargs="*", choices=["t2", "self"], default=["t2"])
    parser.add_argument("--missing-rate", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--valid-max-fraction", type=float, default=0.15)
    parser.add_argument("--p-params", type=float, default=0.30)
    parser.add_argument("--t2-logit-strength", type=float, default=2.0)
    parser.add_argument("--self-effect-scale", type=float, default=1.5)
    parser.add_argument("--max-expected-gap", type=float, default=0.12)
    parser.add_argument("--max-rate-gap", type=float, default=0.15)
    parser.add_argument("--min-observed-rate", type=float, default=0.02)
    parser.add_argument("--min-missing-rate", type=float, default=0.02)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--max-mechanism-tries", type=int, default=5000)
    parser.add_argument("--max-sample-tries", type=int, default=5000)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-process", action="store_true")
    parser.add_argument("--overwrite-info", action="store_true")
    parser.add_argument("--overwrite-masks", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    return parser.parse_args()


def import_generation_modules(support_dir: Path, output_root: Path):
    support_dir = support_dir.resolve()
    if str(support_dir) not in sys.path:
        sys.path.insert(0, str(support_dir))

    import prepare_diffputer_dataset as prep  # type: ignore
    import normalize_diffputer_preprocessed as norm  # type: ignore
    import dataset_v2 as v2  # type: ignore

    prep.DATA_DIR = output_root
    prep.INFO_DIR = output_root / "Info"
    prep.LOCAL_GERMAN_PATH = output_root / "_raw_sources" / "german.data"
    norm.DATA_DIR = output_root
    return prep, norm, v2


def download_file(url: str, destination: Path, force: bool = False) -> None:
    if destination.exists() and not force:
        print(f"[exists] {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    req = request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"[download] {url}")
    with request.urlopen(req) as response, tmp_path.open("wb") as file:
        file.write(response.read())
    if tmp_path.stat().st_size == 0:
        raise IOError(f"Downloaded zero-byte file from {url}")
    tmp_path.replace(destination)
    print(f"[saved] {destination}")


def copy_base_files(source_root: Path, output_root: Path, datasets: list[str]) -> None:
    source_root = source_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    info_src = source_root / "Info"
    info_dst = output_root / "Info"
    info_dst.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        src_dir = source_root / dataset
        dst_dir = output_root / dataset
        if not src_dir.exists():
            raise FileNotFoundError(f"Missing source dataset folder: {src_dir}")
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in ["data.csv", "train.csv", "test.csv"]:
            src = src_dir / name
            if not src.exists():
                raise FileNotFoundError(f"Missing source file: {src}")
            shutil.copy2(src, dst_dir / name)
        for src in src_dir.glob("*_map_*.json"):
            shutil.copy2(src, dst_dir / src.name)
        info_file = info_src / f"{dataset}.json"
        if not info_file.exists():
            raise FileNotFoundError(f"Missing info file: {info_file}")
        shutil.copy2(info_file, info_dst / info_file.name)
        print(f"[base-copy] {dataset}")


def prepare_base_datasets(args: argparse.Namespace, prep) -> None:
    if "german" in args.datasets:
        download_file(GERMAN_DATA_URL, prep.LOCAL_GERMAN_PATH, force=args.force_download)

    for dataset in args.datasets:
        print("=" * 80)
        print(f"[base] {dataset}")
        prep.prepare_dataset(
            name=dataset,
            train_ratio=0.70,
            missing_rate=args.missing_rate,
            mask_types=[],
            mask_num=max(args.splits) + 1 if args.splits else 10,
            normalize=False,
            force_download=args.force_download,
            force_process=args.force_process,
            overwrite_info=args.overwrite_info,
            reproduce_masks=False,
        )


def draw_hard_t2_logistic_mechanism(
    v2,
    train_x: np.ndarray,
    test_x: np.ndarray,
    split_indices,
    missing_rate: float,
    rng: np.random.Generator,
    p_params: float,
    logit_strength: float,
    max_expected_gap: float,
    max_tries: int,
):
    train_sub_x = train_x[split_indices.train_idx].astype(np.float64)
    valid_x = train_x[split_indices.valid_idx].astype(np.float64)
    centers, scales = v2.make_standardizer(train_sub_x)
    train_sub_z = v2.standardize(train_sub_x, centers, scales)
    d = train_sub_x.shape[1]

    best = None
    best_score = float("inf")
    for attempt in range(1, max_tries + 1):
        d_params = max(int(round(float(p_params) * d)), 1)
        if d_params >= d:
            d_params = max(d - 1, 1)
        idxs_params = np.sort(rng.choice(d, size=d_params, replace=False)).astype(np.int64)
        idxs_targets = np.asarray([j for j in range(d) if j not in set(idxs_params.tolist())], dtype=np.int64)
        if idxs_targets.size == 0:
            idxs_targets = np.arange(d, dtype=np.int64)

        coeffs = rng.normal(loc=0.0, scale=1.0, size=(idxs_params.size, idxs_targets.size))
        linear_train = train_sub_z[:, idxs_params].dot(coeffs)
        scale = np.nanstd(linear_train, axis=0, keepdims=True)
        scale[~np.isfinite(scale)] = 1.0
        scale[scale < 1e-6] = 1.0
        coeffs = coeffs / scale * float(logit_strength)
        linear_train = train_sub_z[:, idxs_params].dot(coeffs)
        intercepts = np.array(
            [v2.fit_intercept_linear(linear_train[:, j], missing_rate) for j in range(idxs_targets.size)],
            dtype=np.float64,
        )

        mechanism = v2.Mechanism(
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
        p_train = v2.mechanism_probabilities(train_sub_x, mechanism, missing_rate)
        p_valid = v2.mechanism_probabilities(valid_x, mechanism, missing_rate)
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
    best.expected_test_rate = v2.mechanism_probabilities(
        test_x.astype(np.float64), best, missing_rate
    ).mean(axis=0)
    return best


def save_hard_mask_metadata(save_dir: Path, split_idx: int, dataset: str, mask_name: str, args, mechanism, mask_result) -> None:
    payload = {
        "dataset": dataset,
        "mask_type": mask_name,
        "split_idx": split_idx,
        "missing_rate": args.missing_rate,
        "seed": args.seed,
        "mask_kind": mechanism.kind,
        "valid_max_fraction": args.valid_max_fraction,
        "p_params": args.p_params,
        "self_effect_scale": args.self_effect_scale,
        "t2_logit_strength": args.t2_logit_strength,
        "max_expected_gap": args.max_expected_gap,
        "max_rate_gap": args.max_rate_gap,
        "min_observed_rate": args.min_observed_rate,
        "min_missing_rate": args.min_missing_rate,
        "min_count": args.min_count,
        "max_mechanism_tries": args.max_mechanism_tries,
        "max_sample_tries": args.max_sample_tries,
        "strict_split_files": True,
        "hard_protocol": True,
        "strict_protocol_version": "tose_public_v1_feature_only_train_fit_exact_rate",
        "mechanism_tries": mechanism.tries,
        "sample_tries": mask_result.tries,
        "expected_max_gap": mechanism.expected_max_gap,
        "sampled_max_gap": mask_result.max_gap,
        "sampled_train_valid_max_gap": mask_result.train_valid_max_gap,
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
            "Hard MNAR stress-test setting separated from the main v2_strict benchmark.",
            "The supervised target is excluded before missingness generation.",
            "Mechanism standardization and calibration use strict-train rows only.",
            "The same mechanism is applied to train/valid/test.",
            "Each partition uses the nearest feasible exact overall missing-cell count.",
            "Test values and test rates are not used to select mechanism candidates.",
            "Strict normalization uses strict-train observed values only.",
        ],
    }
    with open(save_dir / f"mechanism_{split_idx}.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def generate_one_mask(args, prep, norm, v2, dataset: str, mask_kind: str, mask_name: str) -> None:
    train_x, test_x = prep.load_dataset(dataset)
    train_x = np.asarray(train_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)
    folder_name = v2.rate_name(args.missing_rate)
    save_dir = args.output_root / dataset / "masks" / folder_name / mask_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for split_idx in sorted(set(args.splits)):
        train_mask_path = save_dir / f"train_mask_{split_idx}.npy"
        test_mask_path = save_dir / f"test_mask_{split_idx}.npy"
        if train_mask_path.exists() and test_mask_path.exists() and not args.overwrite_masks:
            print(f"[skip] masks already exist: {train_mask_path}")
            continue

        seed = int(args.seed) + int(split_idx) * 100_003 + sum(ord(ch) for ch in dataset)
        rng = np.random.default_rng(seed)
        split_indices = v2.build_train_valid_indices(
            train_size=train_x.shape[0],
            test_size=test_x.shape[0],
            seed=args.seed,
            valid_max_fraction=args.valid_max_fraction,
        )
        if mask_kind == "self_logistic":
            mechanism = v2.draw_self_logistic_mechanism(
                train_x=train_x,
                test_x=test_x,
                split_indices=split_indices,
                missing_rate=args.missing_rate,
                rng=rng,
                effect_scale=args.self_effect_scale,
                max_expected_gap=args.max_expected_gap,
                max_tries=args.max_mechanism_tries,
            )
        elif mask_kind == "logistic_t2":
            mechanism = draw_hard_t2_logistic_mechanism(
                v2=v2,
                train_x=train_x,
                test_x=test_x,
                split_indices=split_indices,
                missing_rate=args.missing_rate,
                rng=rng,
                p_params=args.p_params,
                logit_strength=args.t2_logit_strength,
                max_expected_gap=args.max_expected_gap,
                max_tries=args.max_mechanism_tries,
            )
        else:
            raise ValueError(f"Unsupported mask kind: {mask_kind}")

        mask_result = v2.sample_masks_with_rejection(
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
            f"[hard] dataset={dataset} mask={mask_name} split={split_idx} "
            f"expected_gap={mechanism.expected_max_gap:.4f} sampled_gap={mask_result.max_gap:.4f} "
            f"mechanism_tries={mechanism.tries} sample_tries={mask_result.tries}"
        )
        if mechanism.expected_max_gap > args.max_expected_gap:
            print(
                f"[warn] expected gap exceeded threshold "
                f"({mechanism.expected_max_gap:.4f} > {args.max_expected_gap:.4f}); best candidate used."
            )
        if mask_result.max_gap > args.max_rate_gap:
            print(
                f"[warn] sampled gap exceeded threshold "
                f"({mask_result.max_gap:.4f} > {args.max_rate_gap:.4f}); best sample used."
            )

        np.save(train_mask_path, mask_result.train_mask)
        np.save(test_mask_path, mask_result.test_mask)
        np.save(save_dir / f"full_train_mask_{split_idx}.npy", mask_result.train_mask)
        np.save(save_dir / f"strict_train_mask_{split_idx}.npy", mask_result.train_mask[split_indices.train_idx])
        np.save(save_dir / f"valid_mask_{split_idx}.npy", mask_result.train_mask[split_indices.valid_idx])
        save_hard_mask_metadata(save_dir, split_idx, dataset, mask_name, args, mechanism, mask_result)

        save_args = argparse.Namespace(
            data_root=args.output_root,
            missing_rate=args.missing_rate,
            mask_name=mask_name,
        )
        npz_path = v2.save_strict_normalized_split(
            norm=norm,
            args=save_args,
            dataset=dataset,
            split_idx=split_idx,
            split_indices=split_indices,
        )
        print(f"[normalized] {npz_path}")


def summarize_strength(args, v2, prep) -> None:
    summary = {}
    for mask_name in [T2_HARD_MASK, SELF_HARD_MASK]:
        rows = []
        for dataset in args.datasets:
            train_x, test_x = prep.load_dataset(dataset)
            x = np.concatenate([np.asarray(train_x, dtype=np.float64), np.asarray(test_x, dtype=np.float64)])
            for split_idx in sorted(set(args.splits)):
                meta_path = args.output_root / dataset / "masks" / v2.rate_name(args.missing_rate) / mask_name / f"mechanism_{split_idx}.json"
                if not meta_path.exists():
                    continue
                with open(meta_path, "r", encoding="utf-8") as file:
                    meta = json.load(file)
                mechanism = v2.Mechanism(
                    kind=meta["mask_kind"],
                    idxs_params=meta["idxs_params"],
                    idxs_targets=meta["idxs_targets"],
                    centers=np.asarray(meta["centers"], dtype=np.float64),
                    scales=np.asarray(meta["scales"], dtype=np.float64),
                    coeffs=np.asarray(meta["coeffs"], dtype=np.float64),
                    intercepts=np.asarray(meta["intercepts"], dtype=np.float64),
                    expected_train_rate=np.asarray(meta["expected_train_rate"], dtype=np.float64),
                    expected_valid_rate=np.asarray(meta["expected_valid_rate"], dtype=np.float64),
                    expected_test_rate=np.asarray(meta["expected_test_rate"], dtype=np.float64),
                    expected_max_gap=float(meta["expected_max_gap"]),
                    tries=int(meta["mechanism_tries"]),
                )
                probs = v2.mechanism_probabilities(x, mechanism, args.missing_rate).reshape(-1)
                q10, q90 = np.quantile(probs, [0.10, 0.90])
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split_idx,
                        "mean_p": float(probs.mean()),
                        "p90_minus_p10": float(q90 - q10),
                        "prob_std": float(probs.std()),
                        "sampled_max_gap": float(meta["sampled_max_gap"]),
                    }
                )
        if rows:
            summary[mask_name] = {
                "n": len(rows),
                "mean_p": float(np.mean([r["mean_p"] for r in rows])),
                "mean_p90_minus_p10": float(np.mean([r["p90_minus_p10"] for r in rows])),
                "median_p90_minus_p10": float(np.median([r["p90_minus_p10"] for r in rows])),
                "max_p90_minus_p10": float(np.max([r["p90_minus_p10"] for r in rows])),
                "mean_prob_std": float(np.mean([r["prob_std"] for r in rows])),
                "mean_sampled_max_gap": float(np.mean([r["sampled_max_gap"] for r in rows])),
                "max_sampled_max_gap": float(np.max([r["sampled_max_gap"] for r in rows])),
            }
    out = args.output_root / "hard_mnar_strength_summary.json"
    with open(out, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[summary] {out}")


def main() -> None:
    args = parse_args()
    args.support_dir = args.support_dir.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.datasets = list(dict.fromkeys(args.datasets))
    args.splits = sorted(set(args.splits))

    prep, norm, v2 = import_generation_modules(args.support_dir, args.output_root)

    if args.copy_base_from is not None:
        copy_base_files(args.copy_base_from, args.output_root, args.datasets)
    elif args.prepare_base:
        prepare_base_datasets(args, prep)

    if "t2" in args.mask_types:
        for dataset in args.datasets:
            generate_one_mask(args, prep, norm, v2, dataset, "logistic_t2", T2_HARD_MASK)
    if "self" in args.mask_types:
        for dataset in args.datasets:
            generate_one_mask(args, prep, norm, v2, dataset, "self_logistic", SELF_HARD_MASK)

    if args.summarize:
        summarize_strength(args, v2, prep)


if __name__ == "__main__":
    main()
