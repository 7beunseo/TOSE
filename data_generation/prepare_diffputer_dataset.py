import argparse
import json
import shutil
import zipfile
from pathlib import Path
from urllib import request

import numpy as np
import pandas as pd
import torch
from scipy import optimize
from sklearn.datasets import fetch_california_housing, fetch_openml

try:
    from normalize_diffputer_preprocessed import normalize_one
except ImportError:  # pragma: no cover - package-style import
    from .normalize_diffputer_preprocessed import normalize_one


DATA_DIR = Path("dataset")
INFO_DIR = DATA_DIR / "Info"
LOCAL_GERMAN_PATH = Path("german.data")
torch.set_default_dtype(torch.float32)

DEFAULT_MASK_TYPES: list[str] = []
DEFAULT_MISSING_RATE = 0.3
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_MASK_NUM = 10


DATASET_SPECS = {
    "adult": {
        "source": "uci_zip",
        "url": "https://archive.ics.uci.edu/static/public/2/adult.zip",
        "raw_files": ["adult.data"],
        "info": {
            "name": "adult",
            "task_type": "binclass",
            "header": None,
            "num_col_idx": [0, 2, 4, 10, 11, 12],
            "cat_col_idx": [1, 3, 5, 6, 7, 8, 9, 13],
            "target_col_idx": [14],
        },
    },
    "default": {
        "source": "uci_zip",
        "url": "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
        "raw_files": ["default of credit card clients.xls"],
        "info": {
            "name": "default",
            "num_col_idx": [0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            "cat_col_idx": [1, 2, 3, 5, 6, 7, 8, 9, 10],
            "target_col_idx": [23],
        },
    },
    "magic": {
        "source": "uci_zip",
        "url": "https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip",
        "raw_files": ["magic04.data"],
        "info": {
            "name": "magic",
            "header": None,
            "num_col_idx": list(range(10)),
            "cat_col_idx": [],
            "target_col_idx": [10],
        },
    },
    "shoppers": {
        "source": "uci_zip",
        "url": "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip",
        "raw_files": ["online_shoppers_intention.csv"],
        "info": {
            "name": "shoppers",
            "num_col_idx": list(range(10)),
            "cat_col_idx": [10, 11, 12, 13, 14, 15, 16],
            "target_col_idx": [17],
        },
    },
    "news": {
        "source": "uci_zip",
        "url": "https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip",
        "raw_files": ["OnlineNewsPopularity/OnlineNewsPopularity.csv"],
        "info": {
            "name": "news",
            "num_col_idx": list(range(45)),
            "cat_col_idx": [46, 47],
            "target_col_idx": [45],
        },
    },
    "gesture": {
        "source": "uci_zip",
        "url": "https://archive.ics.uci.edu/static/public/302/gesture+phase+segmentation.zip",
        "raw_files": ["a1_va3.csv", "c3_va3.csv"],
        "info": {
            "name": "gesture",
            "header": None,
            "num_col_idx": list(range(32)),
            "cat_col_idx": [],
            "target_col_idx": [],
        },
    },
    "letter": {
        "source": "uci_zip",
        "url": "https://archive.ics.uci.edu/static/public/59/letter+recognition.zip",
        "raw_files": ["letter-recognition.data"],
        "info": {
            "name": "letter",
            "header": None,
            "num_col_idx": list(range(16)),
            "cat_col_idx": [],
            "target_col_idx": [16],
        },
    },
    "bean": {
        "source": "uci_zip",
        "url": "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip",
        "raw_files": ["DryBeanDataset/Dry_Bean_Dataset.xlsx"],
        "info": {
            "name": "bean",
            "header": None,
            "num_col_idx": list(range(16)),
            "cat_col_idx": [],
            "target_col_idx": [16],
        },
    },
    "german": {
        "source": "local_german",
        "raw_files": ["german.data"],
        "info": {
            "name": "german",
            "task_type": "binclass",
            "header": None,
            "num_col_idx": [1, 4, 7, 10, 12, 15, 17],
            "cat_col_idx": [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19],
            "target_col_idx": [20],
        },
    },
    "bank": {
        "source": "openml_fetch",
        "raw_files": ["bank_raw.csv"],
        "openml_id": 1461,
        "info": {
            "name": "bank",
            "task_type": "binclass",
            "header": None,
            "num_col_idx": [0, 5, 9, 11, 12, 13, 14],
            "cat_col_idx": [1, 2, 3, 4, 6, 7, 8, 10, 15],
            "target_col_idx": [16],
        },
    },
    "abalone": {
        "source": "openml_fetch",
        "raw_files": ["abalone_raw.csv"],
        "openml_id": 183,
        "info": {
            "name": "abalone",
            "task_type": "regression",
            "header": None,
            "num_col_idx": [1, 2, 3, 4, 5, 6, 7],
            "cat_col_idx": [0],
            "target_col_idx": [8],
        },
    },
    "mushroom": {
        "source": "openml_fetch",
        "raw_files": ["mushroom_raw.csv"],
        "openml_id": 24,
        "info": {
            "name": "mushroom",
            "task_type": "binclass",
            "header": None,
            "num_col_idx": [],
            "cat_col_idx": list(range(22)),
            "target_col_idx": [22],
        },
    },
    "breast": {
        "source": "openml_fetch",
        "raw_files": ["breast_raw.csv"],
        "openml_id": 1510,
        "info": {
            "name": "breast",
            "task_type": "binclass",
            "header": None,
            "num_col_idx": list(range(30)),
            "cat_col_idx": [],
            "target_col_idx": [30],
        },
    },
    "california": {
        "source": "sklearn_fetch",
        "raw_files": ["california_housing.csv"],
        "info": {
            "name": "california",
            "task_type": "regression",
            "header": None,
            "num_col_idx": list(range(8)),
            "cat_col_idx": [],
            "target_col_idx": [8],
        },
    },
}


def dataset_dir(name: str) -> Path:
    return DATA_DIR / name


def rate_name(p: float) -> str:
    return f"rate{int(round(p * 100))}"


def unzip_file(zip_filepath: Path, dest_path: Path) -> None:
    with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
        zip_ref.extractall(dest_path)


def has_raw_files(name: str) -> bool:
    save_dir = dataset_dir(name)
    raw_files = DATASET_SPECS[name].get("raw_files", [])
    return all((save_dir / rel_path).exists() for rel_path in raw_files)


def download_to_path(url: str, dest_path: Path) -> None:
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    req = request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with request.urlopen(req) as response, open(tmp_path, "wb") as f:
        shutil.copyfileobj(response, f)
    if tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        raise IOError(f"Downloaded zero-byte file from {url}")
    tmp_path.replace(dest_path)


def download_from_uci(name: str, force: bool = False) -> None:
    spec = DATASET_SPECS[name]
    save_dir = dataset_dir(name)
    save_dir.mkdir(parents=True, exist_ok=True)
    zip_path = save_dir / f"{name}.zip"

    if has_raw_files(name) and not force:
        print(f"{name}: raw files already present at {save_dir}")
        return

    if zip_path.exists() and not force:
        print(f"{name}: raw archive already present at {zip_path}")
    else:
        print(f"Start downloading dataset {name} from UCI.")
        download_to_path(spec["url"], zip_path)
        print(f"Finish downloading dataset from {spec['url']}, data has been saved to {zip_path}.")

    for attempt in range(2):
        try:
            unzip_file(zip_path, save_dir)
            print(f"Finish unzipping {name}.")
            if not has_raw_files(name):
                raise FileNotFoundError(
                    f"Expected raw files {spec.get('raw_files', [])} were not found after extracting {zip_path}."
                )
            return
        except (zipfile.BadZipFile, EOFError, FileNotFoundError) as exc:
            if attempt == 1:
                raise
            print(f"{name}: extraction failed with {type(exc).__name__}; re-downloading archive once.")
            if zip_path.exists():
                zip_path.unlink()
            download_to_path(spec["url"], zip_path)

    raise RuntimeError(f"Failed to prepare archive for {name}")


def download_direct_files(name: str, force: bool = False) -> None:
    spec = DATASET_SPECS[name]
    save_dir = dataset_dir(name)
    save_dir.mkdir(parents=True, exist_ok=True)

    if has_raw_files(name) and not force:
        print(f"{name}: raw files already present at {save_dir}")
        return

    for rel_path, url in spec["downloads"].items():
        target_path = save_dir / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() and not force:
            print(f"{name}: file already present at {target_path}")
            continue
        print(f"{name}: downloading {url}")
        download_to_path(url, target_path)
        print(f"{name}: saved {target_path}")

    if not has_raw_files(name):
        raise FileNotFoundError(
            f"Expected raw files {spec.get('raw_files', [])} were not downloaded for {name}."
        )


def download_from_openml(name: str, force: bool = False) -> None:
    spec = DATASET_SPECS[name]
    save_dir = dataset_dir(name)
    save_dir.mkdir(parents=True, exist_ok=True)

    if has_raw_files(name) and not force:
        print(f"{name}: raw files already present at {save_dir}")
        return

    openml_dataset = fetch_openml(
        data_id=spec["openml_id"],
        as_frame=True,
        parser="auto",
    )

    feature_df = openml_dataset.data.copy()
    target = openml_dataset.target
    if not isinstance(target, pd.Series):
        target = pd.Series(target, name="target")
    if target.name is None:
        target = target.rename("target")

    raw_df = pd.concat([feature_df, target], axis=1)
    raw_csv_path = save_dir / spec["raw_files"][0]
    raw_df.to_csv(raw_csv_path, index=False)

    descr_path = save_dir / "openml.DESCR.txt"
    descr_text = getattr(openml_dataset, "DESCR", None)
    if descr_text:
        descr_path.write_text(descr_text, encoding="utf-8")

    print(f"{name}: saved OpenML raw data to {raw_csv_path}")


def prepare_local_german(force: bool = False) -> None:
    save_dir = dataset_dir("german")
    save_dir.mkdir(parents=True, exist_ok=True)
    target_path = save_dir / "german.data"
    if target_path.exists() and not force:
        print(f"german: raw file already present at {target_path}")
        return
    if not LOCAL_GERMAN_PATH.exists():
        raise FileNotFoundError(
            f"Could not find local german source at {LOCAL_GERMAN_PATH.resolve()}."
        )
    shutil.copy2(LOCAL_GERMAN_PATH, target_path)
    print(f"Copied local german.data to {target_path}")


def download_california(force: bool = False) -> None:
    save_dir = dataset_dir("california")
    save_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = save_dir / "california_housing.csv"
    descr_path = save_dir / "california_housing.DESCR.txt"

    if raw_csv_path.exists() and descr_path.exists() and not force:
        print(f"california: raw files already present at {save_dir}")
        return

    dataset = fetch_california_housing(as_frame=True)
    dataset.frame.to_csv(raw_csv_path, index=False)
    descr_path.write_text(dataset.DESCR, encoding="utf-8")
    print(f"Saved California Housing raw data to {raw_csv_path}")


def ensure_dataset_info(name: str, overwrite: bool = False) -> Path:
    INFO_DIR.mkdir(parents=True, exist_ok=True)
    info_path = INFO_DIR / f"{name}.json"
    if info_path.exists() and not overwrite:
        return info_path
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(DATASET_SPECS[name]["info"], f, indent=4)
    print(f"Saved dataset info to {info_path}")
    return info_path


def process_gesture() -> None:
    file_names = ["a1_va3", "a2_va3", "a3_va3", "b1_va3", "b1_va3", "c1_va3", "c3_va3"]
    datas = []
    for name in file_names:
        df = pd.read_csv(dataset_dir("gesture") / f"{name}.csv")
        datas.append(df.to_numpy())

    data = np.concatenate(datas, axis=0)
    pd.DataFrame(data).to_csv(dataset_dir("gesture") / "data.csv", index=False)


def process_letter() -> None:
    dataname = "letter"
    path = dataset_dir(dataname) / f"{dataname}-recognition.data"
    save_path = dataset_dir(dataname) / "data.csv"
    df = pd.read_csv(path, header=None)

    cols = df.columns.tolist()
    cols = cols[1:] + cols[:1]
    df = df[cols]
    df.to_csv(save_path, index=False, header=True)


def process_news() -> None:
    path = dataset_dir("news") / "OnlineNewsPopularity" / "OnlineNewsPopularity.csv"
    save_path = dataset_dir("news") / "data.csv"

    data_df = pd.read_csv(path)
    data_df = data_df.drop("url", axis=1)

    columns = np.array(data_df.columns.tolist())
    cat_columns1 = columns[list(range(12, 18))]
    cat_columns2 = columns[list(range(30, 38))]

    cat_col1 = data_df[cat_columns1].astype(int).to_numpy().argmax(axis=1)
    cat_col2 = data_df[cat_columns2].astype(int).to_numpy().argmax(axis=1)

    data_df = data_df.drop(cat_columns2, axis=1)
    data_df = data_df.drop(cat_columns1, axis=1)
    data_df["data_channel"] = cat_col1
    data_df["weekday"] = cat_col2
    data_df.to_csv(save_path, index=False)


def process_adult() -> None:
    path = dataset_dir("adult") / "adult.data"
    save_path = dataset_dir("adult") / "data.csv"
    data_df = pd.read_csv(path, header=None)
    data_df.dropna().to_csv(save_path, index=False)


def process_shoppers() -> None:
    path = dataset_dir("shoppers") / "online_shoppers_intention.csv"
    save_path = dataset_dir("shoppers") / "data.csv"
    data_df = pd.read_csv(path)
    data_df.dropna().to_csv(save_path, index=False)


def process_default() -> None:
    path = dataset_dir("default") / "default of credit card clients.xls"
    save_path = dataset_dir("default") / "data.csv"
    data_df = pd.read_excel(path, sheet_name="Data", header=1)
    data_df = data_df.drop("ID", axis=1)
    data_df.dropna().to_csv(save_path, index=False)


def process_magic() -> None:
    path = dataset_dir("magic") / "magic04.data"
    save_path = dataset_dir("magic") / "data.csv"
    data_df = pd.read_csv(path, header=None)
    data_df.dropna().to_csv(save_path, index=False)


def process_bean() -> None:
    path = dataset_dir("bean") / "DryBeanDataset" / "Dry_Bean_Dataset.xlsx"
    save_path = dataset_dir("bean") / "data.csv"
    data_df = pd.read_excel(path, sheet_name="Dry_Beans_Dataset", header=1)
    data_df.dropna().to_csv(save_path, index=False)


def process_german() -> None:
    path = dataset_dir("german") / "german.data"
    save_path = dataset_dir("german") / "data.csv"
    data_df = pd.read_csv(path, sep=r"\s+", header=None)
    data_df.dropna().to_csv(save_path, index=False)


def process_bank() -> None:
    path = dataset_dir("bank") / "bank_raw.csv"
    save_path = dataset_dir("bank") / "data.csv"
    data_df = pd.read_csv(path)
    data_df.dropna().to_csv(save_path, index=False)


def process_abalone() -> None:
    path = dataset_dir("abalone") / "abalone_raw.csv"
    save_path = dataset_dir("abalone") / "data.csv"
    data_df = pd.read_csv(path)
    data_df.dropna().to_csv(save_path, index=False)


def process_mushroom() -> None:
    path = dataset_dir("mushroom") / "mushroom_raw.csv"
    save_path = dataset_dir("mushroom") / "data.csv"
    data_df = pd.read_csv(path, na_values="?")
    data_df = data_df.dropna()
    data_df.to_csv(save_path, index=False)


def process_breast() -> None:
    path = dataset_dir("breast") / "breast_raw.csv"
    save_path = dataset_dir("breast") / "data.csv"
    data_df = pd.read_csv(path)
    data_df.dropna().to_csv(save_path, index=False)


def process_california() -> None:
    path = dataset_dir("california") / "california_housing.csv"
    save_path = dataset_dir("california") / "data.csv"
    data_df = pd.read_csv(path)
    data_df.dropna().to_csv(save_path, index=False)


DOWNLOADERS = {
    "uci_zip": download_from_uci,
    "direct_files": download_direct_files,
    "local_german": lambda name, force=False: prepare_local_german(force=force),
    "openml_fetch": lambda name, force=False: download_from_openml(name, force=force),
    "sklearn_fetch": lambda name, force=False: download_california(force=force),
}


PROCESSORS = {
    name: globals()[f"process_{name}"]
    for name in DATASET_SPECS
    if f"process_{name}" in globals()
}


def train_test_split(dataname: str, ratio: float = DEFAULT_TRAIN_RATIO) -> None:
    data_dir = dataset_dir(dataname)
    path = data_dir / "data.csv"
    info_path = INFO_DIR / f"{dataname}.json"

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    cat_idx = info["cat_col_idx"]
    data_df = pd.read_csv(path)
    total_num = data_df.shape[0]

    if len(cat_idx) == 0:
        data_values = data_df.values[:, :-1].astype(np.float32)
        nan_idx = np.isnan(data_values).nonzero()[0]
        keep_idx = list(set(np.arange(data_values.shape[0])) - set(list(nan_idx)))
        keep_idx = np.array(keep_idx)
    else:
        keep_idx = np.arange(total_num)

    num_train = int(keep_idx.shape[0] * ratio)
    num_test = total_num - num_train

    np.random.seed(1234)
    np.random.shuffle(keep_idx)

    train_idx = keep_idx[:num_train]
    test_idx = keep_idx[-num_test:]

    train_df = data_df.loc[train_idx]
    test_df = data_df.loc[test_idx]

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Splitting training and testing data for {dataname} is done.")
    print(f"Training data shape: {train_df.shape}, Testing data shape: {test_df.shape}")
    print(f"Training data saved at {train_path}, Testing data saved at {test_path}.")


def load_dataset(dataname: str):
    """Load logical input features for missingness generation.

    The supervised target is intentionally excluded.  Missingness mechanisms
    must be defined on exactly the same logical variables that are presented to
    the imputation model.
    """
    data_dir = dataset_dir(dataname)
    info_path = INFO_DIR / f"{dataname}.json"

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    data_df = pd.read_csv(data_dir / "data.csv")
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    cols = train_df.columns
    numeric = set(num_col_idx)
    categorical = set(cat_col_idx)
    feature_indices = sorted(numeric | categorical)
    if not feature_indices:
        raise ValueError(f"Dataset {dataname!r} has no input features")

    train_features = []
    test_features = []
    for column_idx in feature_indices:
        column = cols[column_idx]
        if column_idx in numeric:
            train_values = train_df[column].to_numpy(dtype=np.float32)
            test_values = test_df[column].to_numpy(dtype=np.float32)
        else:
            map_path_bin = data_dir / f"{column}_map_bin.json"
            map_path_idx = data_dir / f"{column}_map_idx.json"
            categories = data_df[column].astype(str).unique().tolist()
            category_to_idx = {category: index for index, category in enumerate(categories)}
            num_bits = max(1, (len(categories) - 1).bit_length())
            category_to_binary = {
                category: format(index, f"0{num_bits}b")
                for category, index in category_to_idx.items()
            }
            if not map_path_idx.exists():
                map_path_idx.write_text(json.dumps(category_to_idx), encoding="utf-8")
            if not map_path_bin.exists():
                map_path_bin.write_text(json.dumps(category_to_binary), encoding="utf-8")
            train_values = (
                train_df[column].astype(str).map(category_to_idx).to_numpy(dtype=np.float32)
            )
            test_values = (
                test_df[column].astype(str).map(category_to_idx).to_numpy(dtype=np.float32)
            )
            if not np.isfinite(train_values).all() or not np.isfinite(test_values).all():
                raise ValueError(f"Unknown category encountered in {dataname}.{column}")

        train_features.append(train_values)
        test_features.append(test_values)

    return (
        np.stack(train_features, axis=1).astype(np.float32),
        np.stack(test_features, axis=1).astype(np.float32),
    )


def concat_indices(*index_groups: list[int]) -> np.ndarray:
    merged = []
    for group in index_groups:
        merged.extend(group)
    return np.asarray(merged, dtype=np.int64)


def quantile(X, q, dim=None):
    return X.kthvalue(int(q * len(X)), dim=dim)[0]


def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
    n, d = X.shape
    if self_mask:
        coeffs = torch.randn(d)
        wx = X * coeffs
        scale = torch.std(wx, 0)
        scale[~torch.isfinite(scale)] = 1.0
        scale[scale < 1e-6] = 1.0
        coeffs /= scale
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na, dtype=X.dtype)
        wx = X[:, idxs_obs].mm(coeffs)
        scale = torch.std(wx, 0, keepdim=True)
        scale[~torch.isfinite(scale)] = 1.0
        scale[scale < 1e-6] = 1.0
        coeffs /= scale
    return coeffs


def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(d)
        for j in range(d):
            x_col = X[:, j]
            coeff_j = coeffs[j]

            def f(x):
                return torch.sigmoid(x_col * coeff_j + x).mean().item() - p

            try:
                intercepts[j] = optimize.bisect(f, -50, 50)
            except ValueError:
                base_p = min(max(float(p), 1e-6), 1.0 - 1e-6)
                intercepts[j] = float(np.log(base_p / (1.0 - base_p)))
    else:
        _, d_na = coeffs.shape
        intercepts = torch.zeros(d_na)
        for j in range(d_na):
            def f(x):
                return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p
            try:
                intercepts[j] = optimize.bisect(f, -500, 500)
            except ValueError:
                base_p = min(max(float(p), 1e-6), 1.0 - 1e-6)
                intercepts[j] = float(np.log(base_p / (1.0 - base_p)))
    return intercepts


def mar_mask(X, p, p_obs):
    n, d = X.shape
    if not torch.is_tensor(X):
        X = torch.from_numpy(X)
    mask = torch.zeros(n, d).bool()
    d_obs = max(int(p_obs * d), 1)
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)
    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)
    mask[:, idxs_nas] = torch.rand(n, d - d_obs) < ps
    return mask.numpy()


def mnar_mask_logistic(X, p, p_params=0.3, exclude_inputs=True):
    n, d = X.shape
    if not torch.is_tensor(X):
        X = torch.from_numpy(X)
    mask = torch.zeros(n, d).bool()
    d_params = max(int(p_params * d), 1) if exclude_inputs else d
    idxs_params = np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_params]) if exclude_inputs else np.arange(d)
    coeffs = pick_coeffs(X, idxs_params, idxs_nas)
    intercepts = fit_intercepts(X[:, idxs_params], coeffs, p)
    ps = torch.sigmoid(X[:, idxs_params].mm(coeffs) + intercepts)
    mask[:, idxs_nas] = torch.rand(n, len(idxs_nas)) < ps
    if exclude_inputs:
        mask[:, idxs_params] = torch.rand(n, d_params) < p
    return mask.numpy()


def mnar_self_mask_logistic(X, p):
    n, d = X.shape
    if not torch.is_tensor(X):
        X = torch.from_numpy(X)
    coeffs = pick_coeffs(X, self_mask=True)
    intercepts = fit_intercepts(X, coeffs, p, self_mask=True)
    ps = torch.sigmoid(X * coeffs + intercepts)
    mask = torch.rand(n, d) < ps
    return mask.numpy()


def ensure_min_observed_per_feature(
    mask: np.ndarray,
    min_observed_rate: float = 0.01,
    min_observed_count: int = 10,
    min_missing_rate: float = 0.01,
    min_missing_count: int = 10,
) -> np.ndarray:
    """Prevent degenerate MNAR-self masks where a feature is always observed or always missing."""
    mask = mask.copy()
    n, d = mask.shape
    required_observed = min(n, max(int(np.ceil(n * min_observed_rate)), int(min_observed_count)))
    required_missing = min(n, max(int(np.ceil(n * min_missing_rate)), int(min_missing_count)))
    if required_observed <= 0 and required_missing <= 0:
        return mask

    for col in range(d):
        observed_count = int((~mask[:, col]).sum())
        if observed_count < required_observed:
            need = required_observed - observed_count
            missing_rows = np.flatnonzero(mask[:, col])
            if missing_rows.size > 0:
                chosen = np.random.choice(missing_rows, size=min(need, missing_rows.size), replace=False)
                mask[chosen, col] = False
                print(
                    f"[mask-adjust] feature={col} observed_count={observed_count} "
                    f"-> {int((~mask[:, col]).sum())}",
                    flush=True,
                )

        missing_count = int(mask[:, col].sum())
        if missing_count < required_missing:
            need = required_missing - missing_count
            observed_rows = np.flatnonzero(~mask[:, col])
            if observed_rows.size > 0:
                chosen = np.random.choice(observed_rows, size=min(need, observed_rows.size), replace=False)
                mask[chosen, col] = True
                print(
                    f"[mask-adjust] feature={col} missing_count={missing_count} "
                    f"-> {int(mask[:, col].sum())}",
                    flush=True,
                )
    return mask


def generate_mask(dataname: str, mask_type: str, p: float, mask_num: int, reproduce: bool = True):
    train_X, test_X = load_dataset(dataname)
    print(f"{dataname} missing probability: {p}")

    q = 0.3
    if p > 0.3:
        q = 0.1

    for mask_idx in range(mask_num):
        if mask_type == "MCAR":
            train_mask = np.random.rand(*train_X.shape) < p
            test_mask = np.random.rand(*test_X.shape) < p
        elif mask_type == "MAR":
            train_mask = mar_mask(train_X, p=p / (1 - q), p_obs=q)
            test_mask = mar_mask(test_X, p=p / (1 - q), p_obs=q)
        elif mask_type == "MNAR_logistic_T2":
            train_mask = mnar_mask_logistic(train_X, p=p, p_params=q, exclude_inputs=True)
            test_mask = mnar_mask_logistic(test_X, p=p, p_params=q, exclude_inputs=True)
        elif mask_type in {"MNAR_self_logistic", "MNAR_self_logistic_guarded"}:
            train_mask = mnar_self_mask_logistic(train_X, p=p)
            test_mask = mnar_self_mask_logistic(test_X, p=p)
            train_mask = ensure_min_observed_per_feature(train_mask)
            test_mask = ensure_min_observed_per_feature(test_mask)
        else:
            raise ValueError("Invalid mask type")

        folder_name = rate_name(p)
        data_dir = dataset_dir(dataname) / "masks" / folder_name / mask_type
        data_dir.mkdir(parents=True, exist_ok=True)

        train_mask_path = data_dir / f"train_mask_{mask_idx}.npy"
        test_mask_path = data_dir / f"test_mask_{mask_idx}.npy"

        if train_mask_path.exists() and test_mask_path.exists() and not reproduce:
            print(f"Masks already exist at {train_mask_path}")
            continue

        np.save(train_mask_path, train_mask)
        np.save(test_mask_path, test_mask)
        print(f"Saved train/test masks for {dataname} {mask_type} #{mask_idx}")


def normalize_dataset(dataname: str, mask_types: list[str], ratio: str, splits: int, force: bool = False) -> None:
    for mask_type in mask_types:
        save_dir = dataset_dir(dataname) / "normalized" / f"rate{ratio}" / mask_type
        save_dir.mkdir(parents=True, exist_ok=True)
        for split_idx in range(splits):
            npz_path = save_dir / f"split_{split_idx}.npz"
            json_path = save_dir / f"split_{split_idx}.json"
            if npz_path.exists() and json_path.exists() and not force:
                print(f"Normalized split already exists at {npz_path}")
                continue
            save_path = normalize_one(dataname, split_idx, mask_type, ratio)
            print(f"Saved normalized preprocessing to {save_path}")


def prepare_dataset(
    name: str,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    missing_rate: float = DEFAULT_MISSING_RATE,
    mask_types: list[str] | None = None,
    mask_num: int = DEFAULT_MASK_NUM,
    normalize: bool = True,
    force_download: bool = False,
    force_process: bool = False,
    overwrite_info: bool = False,
    reproduce_masks: bool = False,
) -> None:
    if name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {name}")

    if mask_types is None:
        mask_types = DEFAULT_MASK_TYPES

    ensure_dataset_info(name, overwrite=overwrite_info)

    source = DATASET_SPECS[name]["source"]
    DOWNLOADERS[source](name, force=force_download)

    data_path = dataset_dir(name) / "data.csv"
    if force_process or not data_path.exists():
        PROCESSORS[name]()
        print(f"Processed raw files into {data_path}")
    else:
        print(f"{name}: processed data already present at {data_path}")

    train_path = dataset_dir(name) / "train.csv"
    test_path = dataset_dir(name) / "test.csv"
    if force_process or not train_path.exists() or not test_path.exists():
        train_test_split(name, ratio=train_ratio)
    else:
        print(f"{name}: train/test split already present.")

    for mask_type in mask_types:
        generate_mask(
            dataname=name,
            mask_type=mask_type,
            mask_num=mask_num,
            p=missing_rate,
            reproduce=reproduce_masks,
        )

    if normalize:
        normalize_dataset(
            dataname=name,
            mask_types=mask_types,
            ratio=str(int(round(missing_rate * 100))),
            splits=mask_num,
            force=force_process,
        )


def available_datasets() -> list[str]:
    return sorted(DATASET_SPECS.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download and preprocess base tabular datasets. Use the strict MCAR/MAR/MNAR "
            "generators for public benchmark masks."
        )
    )
    parser.add_argument("--datasets", nargs="*", default=available_datasets())
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--missing-rate", type=float, default=DEFAULT_MISSING_RATE)
    parser.add_argument("--mask-types", nargs="*", default=DEFAULT_MASK_TYPES)
    parser.add_argument("--mask-num", type=int, default=DEFAULT_MASK_NUM)
    parser.add_argument("--skip-normalize", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-process", action="store_true")
    parser.add_argument("--overwrite-info", action="store_true")
    parser.add_argument("--reproduce-masks", action="store_true")
    return parser.parse_args()


def os_path_exists(path: Path) -> bool:
    return path.exists()


def main():
    args = parse_args()

    for name in args.datasets:
        print("=" * 80)
        print(f"Preparing dataset: {name}")
        prepare_dataset(
            name=name,
            train_ratio=args.train_ratio,
            missing_rate=args.missing_rate,
            mask_types=args.mask_types,
            mask_num=args.mask_num,
            normalize=not args.skip_normalize,
            force_download=args.force_download,
            force_process=args.force_process,
            overwrite_info=args.overwrite_info,
            reproduce_masks=args.reproduce_masks,
        )


if __name__ == "__main__":
    main()
