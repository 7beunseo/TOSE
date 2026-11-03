import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("dataset")
DEFAULT_MASK_TYPES = ["MCAR", "MAR", "MNAR_logistic_T2", "MNAR_self_logistic"]
DEFAULT_RATIO = "30"
DEFAULT_SPLITS = list(range(10))


def ensure_category_maps(data_dir: Path, data_df: pd.DataFrame, cat_columns: pd.Index) -> None:
    if len(cat_columns) == 0:
        return

    first_map = data_dir / f"{cat_columns[0]}_map_bin.json"
    if first_map.exists():
        return

    for column in cat_columns:
        categories = data_df[column].astype(str).unique()
        num_categories = len(categories)
        num_bits = (num_categories - 1).bit_length()

        category_to_binary = {
            category: format(index, "0" + str(num_bits) + "b")
            for index, category in enumerate(categories)
        }
        category_to_idx = {category: index for index, category in enumerate(categories)}

        with open(data_dir / f"{column}_map_bin.json", "w", encoding="utf-8") as f:
            json.dump(category_to_binary, f)
        with open(data_dir / f"{column}_map_idx.json", "w", encoding="utf-8") as f:
            json.dump(category_to_idx, f)


def extend_mask(mask: np.ndarray, bin_num: np.ndarray) -> np.ndarray:
    num_rows, num_cols = mask.shape
    cum_sum = bin_num.cumsum()
    cum_sum = np.insert(cum_sum, 0, 0)
    result = np.zeros((num_rows, int(bin_num.sum())), dtype=bool)

    for idx in range(num_cols):
        tiled = np.tile(mask[:, idx][:, np.newaxis], int(bin_num[idx]))
        result[:, cum_sum[idx]:cum_sum[idx + 1]] = tiled

    return result


def select_feature_mask(
    mask: np.ndarray,
    feature_indices: list[int],
    requested_indices: list[int],
    original_column_count: int,
) -> np.ndarray:
    """Select logical feature masks from corrected or legacy mask files."""
    if mask.shape[1] == len(feature_indices):
        local_index = {original: local for local, original in enumerate(feature_indices)}
        return mask[:, [local_index[index] for index in requested_indices]]
    if mask.shape[1] == original_column_count:
        return mask[:, requested_indices]
    raise ValueError(
        "Mask width does not match either the feature-only protocol or the legacy table: "
        f"got {mask.shape[1]}, expected {len(feature_indices)} or {original_column_count}"
    )


def load_diffputer_features(dataname: str, split_idx: int, mask_type: str, ratio: str):
    data_dir = DATA_DIR / dataname
    info_path = DATA_DIR / "Info" / f"{dataname}.json"

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    feature_indices = sorted(set(num_col_idx) | set(cat_col_idx))

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    data_df = pd.read_csv(data_dir / "data.csv")

    train_mask = np.load(data_dir / "masks" / f"rate{ratio}" / mask_type / f"train_mask_{split_idx}.npy")
    test_mask = np.load(data_dir / "masks" / f"rate{ratio}" / mask_type / f"test_mask_{split_idx}.npy")

    cols = train_df.columns
    train_num = train_df[cols[num_col_idx]].values.astype(np.float32)
    test_num = test_df[cols[num_col_idx]].values.astype(np.float32)

    if len(cat_col_idx) == 0:
        train_x = train_num
        test_x = test_num
        ext_train_mask = select_feature_mask(
            train_mask, feature_indices, num_col_idx, len(cols)
        )
        ext_test_mask = select_feature_mask(
            test_mask, feature_indices, num_col_idx, len(cols)
        )
        cat_bin_num = np.array([], dtype=np.int64)
    else:
        cat_columns = cols[cat_col_idx]
        ensure_category_maps(data_dir, data_df, cat_columns)

        train_cat = train_df[cat_columns].astype(str)
        test_cat = test_df[cat_columns].astype(str)
        data_cat = data_df[cat_columns].astype(str)

        train_cat_bin = []
        test_cat_bin = []
        cat_bin_num = []

        for column in data_cat.columns:
            with open(data_dir / f"{column}_map_bin.json", "r", encoding="utf-8") as f:
                category_to_binary = json.load(f)

            train_cat_enc = train_cat[column].map(category_to_binary).to_numpy()
            test_cat_enc = test_cat[column].map(category_to_binary).to_numpy()

            train_cat_bin_i = np.array([list(map(int, binary)) for binary in train_cat_enc], dtype=np.float32)
            test_cat_bin_i = np.array([list(map(int, binary)) for binary in test_cat_enc], dtype=np.float32)

            train_cat_bin.append(train_cat_bin_i)
            test_cat_bin.append(test_cat_bin_i)
            cat_bin_num.append(train_cat_bin_i.shape[1])

        train_cat_bin = np.concatenate(train_cat_bin, axis=1)
        test_cat_bin = np.concatenate(test_cat_bin, axis=1)
        cat_bin_num = np.array(cat_bin_num, dtype=np.int64)

        train_x = np.concatenate([train_num, train_cat_bin], axis=1)
        test_x = np.concatenate([test_num, test_cat_bin], axis=1)

        train_num_mask = select_feature_mask(
            train_mask, feature_indices, num_col_idx, len(cols)
        )
        test_num_mask = select_feature_mask(
            test_mask, feature_indices, num_col_idx, len(cols)
        )
        train_cat_mask = extend_mask(
            select_feature_mask(train_mask, feature_indices, cat_col_idx, len(cols)),
            cat_bin_num,
        )
        test_cat_mask = extend_mask(
            select_feature_mask(test_mask, feature_indices, cat_col_idx, len(cols)),
            cat_bin_num,
        )

        ext_train_mask = np.concatenate([train_num_mask, train_cat_mask], axis=1)
        ext_test_mask = np.concatenate([test_num_mask, test_cat_mask], axis=1)

    return train_x, test_x, ext_train_mask.astype(bool), ext_test_mask.astype(bool), cat_bin_num


def mean_std(data: np.ndarray, mask: np.ndarray):
    observed = (~mask).astype(np.float32)
    observed_count = observed.sum(0)
    observed_count[observed_count == 0] = 1
    mean = (data * observed).sum(0) / observed_count
    var = ((data - mean) ** 2 * observed).sum(0) / observed_count
    std = np.sqrt(var)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_one(dataname: str, split_idx: int, mask_type: str, ratio: str) -> Path:
    train_x, test_x, train_mask, test_mask, cat_bin_num = load_diffputer_features(
        dataname=dataname,
        split_idx=split_idx,
        mask_type=mask_type,
        ratio=ratio,
    )

    mean, std = mean_std(train_x, train_mask)
    train_x_norm = ((train_x - mean) / std / 2).astype(np.float32)
    test_x_norm = ((test_x - mean) / std / 2).astype(np.float32)

    train_x_init = ((1.0 - train_mask.astype(np.float32)) * train_x_norm).astype(np.float32)
    test_x_init = ((1.0 - test_mask.astype(np.float32)) * test_x_norm).astype(np.float32)

    save_dir = DATA_DIR / dataname / "normalized" / f"rate{ratio}" / mask_type
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"split_{split_idx}.npz"

    np.savez_compressed(
        save_path,
        train_x_norm=train_x_norm,
        test_x_norm=test_x_norm,
        train_x_init=train_x_init,
        test_x_init=test_x_init,
        mean=mean,
        std=std,
        train_mask=train_mask,
        test_mask=test_mask,
        cat_bin_num=cat_bin_num,
    )

    meta = {
        "dataset": dataname,
        "ratio": ratio,
        "mask_type": mask_type,
        "split_idx": split_idx,
        "train_shape": list(train_x_norm.shape),
        "test_shape": list(test_x_norm.shape),
        "num_missing_train": int(train_mask.sum()),
        "num_missing_test": int(test_mask.sum()),
    }
    with open(save_dir / f"split_{split_idx}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return save_path


def available_datasets():
    return sorted(
        path.name
        for path in DATA_DIR.iterdir()
        if path.is_dir() and path.name not in {"Info"} and (path / "data.csv").exists()
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize DiffPuter-style tabular datasets.")
    parser.add_argument("--datasets", nargs="*", default=available_datasets())
    parser.add_argument("--ratio", default=DEFAULT_RATIO)
    parser.add_argument("--mask-types", nargs="*", default=DEFAULT_MASK_TYPES)
    parser.add_argument("--splits", nargs="*", type=int, default=DEFAULT_SPLITS)
    return parser.parse_args()


def main():
    args = parse_args()

    for dataname in args.datasets:
        for mask_type in args.mask_types:
            for split_idx in args.splits:
                save_path = normalize_one(
                    dataname=dataname,
                    split_idx=split_idx,
                    mask_type=mask_type,
                    ratio=args.ratio,
                )
                print(f"Saved normalized preprocessing to {save_path}")


if __name__ == "__main__":
    main()
