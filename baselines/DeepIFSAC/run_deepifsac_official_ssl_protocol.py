from __future__ import annotations

import argparse
import csv
import gc
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[1]
OFFICIAL_DEEPIFSAC_ROOT = Path(
    os.environ.get(
        "DEEPIFSAC_SOURCE_ROOT",
        PROJECT_ROOT / ".baseline_sources" / "deepifsac",
    )
).resolve()
if str(OFFICIAL_DEEPIFSAC_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_DEEPIFSAC_ROOT))

from deepifsac_native_protocol_utils import (
    DeepIFSACBenchmarkBundle,
    DeepIFSACSameProtocolDataset,
    compose_raw_prediction,
    evaluate_predictions,
    save_json,
    score_from_metrics,
)
from models import DeepIFSAC


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "deepifsac_official_ssl"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the DeepIFSAC CutMix + denoising + contrastive recipe on shared strict splits."
        )
    )
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mask-type", type=str, required=True)
    parser.add_argument("--ratio", type=str, default="30")
    parser.add_argument("--splits", nargs="*", type=int, default=[0])
    parser.add_argument(
        "--run-tag",
        type=str,
        default="deepifsac_official_ssl_cutmix_denoise_contrastive_e600_es",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--valid-max-fraction", type=float, default=0.15)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--early-stop-patience", type=int, default=45)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0002)
    parser.add_argument("--selection-split", type=str, default="valid", choices=["valid"])
    parser.add_argument("--eval-every", type=int, default=1)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--scheduler", type=str, default="none", choices=["cosine", "none"])
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--embedding-size", type=int, default=32)
    parser.add_argument("--transformer-depth", type=int, default=6)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--ff-dropout", type=float, default=0.1)
    parser.add_argument(
        "--attentiontype",
        type=str,
        default="colrow",
        choices=["col", "row", "colrow", "rowcol", "parallel", "colrowatt"],
    )
    parser.add_argument("--cont-embeddings", type=str, default="MLP", choices=["MLP"])
    parser.add_argument("--final-mlp-style", type=str, default="sep", choices=["sep"])
    parser.add_argument("--cutmix-categorical-rate", type=float, default=0.3)
    parser.add_argument(
        "--cutmix-continuous-rate",
        type=float,
        default=0.3,
        help="Continuous CutMix rate fixed by Appendix B.5.",
    )
    parser.add_argument("--nce-temperature", type=float, default=0.7)
    parser.add_argument("--contrastive-weight", type=float, default=0.5)
    parser.add_argument("--categorical-denoising-weight", type=float, default=1.0)
    parser.add_argument("--numeric-denoising-weight", type=float, default=10.0)
    parser.add_argument(
        "--projection-style",
        type=str,
        default="diff",
        choices=["diff", "same", "none"],
    )
    parser.add_argument(
        "--categorical-denoising-scope",
        type=str,
        default="observed",
        choices=["observed", "official-all-imputed"],
        help=(
            "Observed is the leakage-safe categorical adaptation. official-all-imputed reproduces "
            "the released code, including CE targets at initially imputed cells."
        ),
    )
    parser.add_argument("--disable-cutmix", action="store_true")
    parser.add_argument("--disable-contrastive", action="store_true")
    parser.add_argument("--disable-denoising", action="store_true")

    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--resume-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-last-checkpoint", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    def _as_rng_tensor(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().to(dtype=torch.uint8)
        return torch.as_tensor(value, dtype=torch.uint8, device="cpu")

    torch.set_rng_state(_as_rng_tensor(state["torch"]))
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_as_rng_tensor(item) for item in cuda_state])


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def safe_torch_save(payload: dict, path: Path, retries: int = 8, delay_seconds: float = 0.5) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: BaseException | None = None

    for attempt in range(retries):
        tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{attempt}")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            torch.save(payload, tmp_path)
            for replace_attempt in range(retries):
                try:
                    tmp_path.replace(path)
                    return
                except OSError as exc:
                    last_error = exc
                    time.sleep(delay_seconds * (replace_attempt + 1))
            try:
                tmp_path.unlink()
            except OSError:
                pass
        except (OSError, RuntimeError) as exc:
            last_error = exc
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            time.sleep(delay_seconds * (attempt + 1))

    raise RuntimeError(f"failed to save checkpoint to {path}") from last_error


def fixed_embed_data_mask(
    x_categ: torch.Tensor,
    x_cont: torch.Tensor,
    cat_mask: torch.Tensor,
    con_mask: torch.Tensor,
    model: DeepIFSAC,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_categ = x_categ + model.categories_offset.type_as(x_categ)
    x_categ_enc = model.embeds(x_categ)
    batch_size, num_cont = x_cont.shape

    if model.cont_embeddings != "MLP":
        raise ValueError("The same-protocol wrapper currently supports cont_embeddings='MLP' only.")

    x_cont_enc = torch.empty(batch_size, num_cont, model.dim, device=x_cont.device)
    for idx in range(model.num_continuous):
        x_cont_enc[:, idx, :] = model.simple_MLP[idx](x_cont[:, idx])

    cat_mask_temp = cat_mask + model.cat_mask_offset.type_as(cat_mask)
    con_mask_temp = con_mask + model.con_mask_offset.type_as(con_mask)
    cat_mask_temp = model.mask_embeds_cat(cat_mask_temp)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)

    x_categ_enc = x_categ_enc.clone()
    x_cont_enc = x_cont_enc.clone()
    x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]
    return x_categ_enc, x_cont_enc


def build_model(args: argparse.Namespace, bundle: DeepIFSACBenchmarkBundle) -> DeepIFSAC:
    categories = np.concatenate(
        [np.array([1], dtype=np.int64), bundle.cat_cardinalities.cpu().numpy().astype(np.int64)]
    )
    return DeepIFSAC(
        categories=tuple(categories.tolist()),
        num_continuous=bundle.num_dim,
        dim=args.embedding_size,
        dim_out=1,
        depth=args.transformer_depth,
        heads=args.attention_heads,
        attn_dropout=args.attention_dropout,
        ff_dropout=args.ff_dropout,
        mlp_hidden_mults=(4, 2),
        cont_embeddings=args.cont_embeddings,
        attentiontype=args.attentiontype,
        final_mlp_style=args.final_mlp_style,
        y_dim=1,
    )


def forward_outputs(model: DeepIFSAC, batch: dict[str, torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    x_categ_enc, x_cont_enc = fixed_embed_data_mask(
        batch["x_categ"],
        batch["x_cont"],
        batch["cat_mask"],
        batch["con_mask"],
        model,
    )
    return model(x_categ_enc, x_cont_enc)


def _column_fill_values(values: torch.Tensor, missing: torch.Tensor, categorical: bool) -> torch.Tensor:
    fills = []
    for column_idx in range(values.shape[1]):
        observed = values[:, column_idx][~missing[:, column_idx]]
        if observed.numel() == 0:
            fill = values.new_tensor(0)
        elif categorical:
            fill = observed.long().mode().values.to(dtype=values.dtype)
        else:
            fill = observed.float().median().to(dtype=values.dtype)
        fills.append(fill)
    if not fills:
        return values.new_zeros((0,))
    return torch.stack(fills)


def build_initial_imputation(
    bundle: DeepIFSACBenchmarkBundle,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_values = bundle.train_split.model_x.detach().cpu()
    train_missing = bundle.train_split.missing_mask_ext.detach().cpu().bool()

    train_num = train_values[:, : bundle.num_dim].float()
    train_num_missing = train_missing[:, : bundle.num_dim]
    num_fill = _column_fill_values(train_num, train_num_missing, categorical=False)
    train_num_imputed = torch.where(
        train_num_missing,
        num_fill.unsqueeze(0).expand_as(train_num),
        train_num,
    )

    train_cat = train_values[:, bundle.num_dim :].long()
    train_cat_missing = train_missing[:, bundle.num_dim :]
    cat_fill = _column_fill_values(train_cat, train_cat_missing, categorical=True).long()
    return num_fill.float(), cat_fill.long(), train_num_imputed.float()


def build_imputed_batch_inputs(
    batch: dict[str, torch.Tensor],
    num_fill: torch.Tensor,
    cat_fill: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_cont = torch.where(
        batch["missing_num_mask"].bool(),
        num_fill.to(batch["target_num"].device).unsqueeze(0),
        batch["target_num"],
    )
    x_cat_values = torch.where(
        batch["missing_cat_mask"].bool(),
        cat_fill.to(batch["target_cat"].device).unsqueeze(0),
        batch["target_cat"],
    )
    cls = torch.zeros((x_cat_values.shape[0], 1), dtype=torch.long, device=x_cat_values.device)
    return torch.cat([cls, x_cat_values.long()], dim=1), x_cont.float()


def make_cutmix_view(
    x_categ: torch.Tensor,
    x_cont: torch.Tensor,
    train_num_bank: torch.Tensor,
    categorical_rate: float,
    continuous_rate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = x_categ.shape[0]
    x_categ_corr = x_categ.clone()
    x_cont_corr = x_cont.clone()

    if x_categ.shape[1] > 1 and categorical_rate > 0.0:
        donor_rows = torch.randperm(batch_size, device=x_categ.device)
        replace = torch.rand(
            (batch_size, x_categ.shape[1] - 1), device=x_categ.device
        ) < categorical_rate
        donor = x_categ[donor_rows, 1:]
        x_categ_corr[:, 1:] = torch.where(replace, donor, x_categ[:, 1:])

    if x_cont.shape[1] > 0 and continuous_rate > 0.0:
        donor_indices = torch.randint(0, train_num_bank.shape[0], (batch_size,))
        donor = train_num_bank[donor_indices].to(x_cont.device, non_blocking=True)
        replace = torch.rand_like(x_cont) < continuous_rate
        x_cont_corr = torch.where(replace, donor, x_cont)

    return x_categ_corr, x_cont_corr


def official_ssl_loss(
    model: DeepIFSAC,
    batch: dict[str, torch.Tensor],
    num_dim: int,
    cat_dim: int,
    num_fill: torch.Tensor,
    cat_fill: torch.Tensor,
    train_num_bank: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    stats = {
        "num_loss": float("nan"),
        "cat_loss": float("nan"),
        "contrastive_loss": float("nan"),
        "denoising_loss": float("nan"),
        "active_cat_columns": 0.0,
    }
    x_categ, x_cont = build_imputed_batch_inputs(batch, num_fill, cat_fill)
    if args.disable_cutmix:
        x_categ_corr, x_cont_corr = x_categ, x_cont
    else:
        continuous_rate = (
            float(args.cutmix_continuous_rate)
            if args.cutmix_continuous_rate is not None
            else float(args.ratio) / 100.0
        )
        x_categ_corr, x_cont_corr = make_cutmix_view(
            x_categ=x_categ,
            x_cont=x_cont,
            train_num_bank=train_num_bank,
            categorical_rate=float(args.cutmix_categorical_rate),
            continuous_rate=continuous_rate,
        )

    clean_cat_enc, clean_cont_enc = fixed_embed_data_mask(
        x_categ,
        x_cont,
        batch["cat_mask"],
        batch["con_mask"],
        model,
    )
    corr_cat_enc, corr_cont_enc = fixed_embed_data_mask(
        x_categ_corr,
        x_cont_corr,
        batch["cat_mask"],
        batch["con_mask"],
        model,
    )

    loss_terms: list[torch.Tensor] = []
    if not args.disable_contrastive:
        clean_repr = model.transformer(clean_cat_enc, clean_cont_enc)
        corr_repr = model.transformer(corr_cat_enc, corr_cont_enc)
        clean_repr = F.normalize(clean_repr, dim=-1).flatten(1, 2)
        corr_repr = F.normalize(corr_repr, dim=-1).flatten(1, 2)
        if args.projection_style == "diff":
            clean_repr = model.pt_mlp(clean_repr)
            corr_repr = model.pt_mlp2(corr_repr)
        elif args.projection_style == "same":
            clean_repr = model.pt_mlp(clean_repr)
            corr_repr = model.pt_mlp(corr_repr)

        temperature = max(float(args.nce_temperature), 1e-8)
        logits_clean = clean_repr @ corr_repr.t() / temperature
        logits_corr = corr_repr @ clean_repr.t() / temperature
        identity = torch.arange(logits_clean.shape[0], device=logits_clean.device)
        contrastive = 0.5 * (
            F.cross_entropy(logits_clean, identity) + F.cross_entropy(logits_corr, identity)
        )
        loss_terms.append(float(args.contrastive_weight) * contrastive)
        stats["contrastive_loss"] = float(contrastive.detach().cpu().item())

    if not args.disable_denoising:
        cat_outs, con_outs = model(corr_cat_enc, corr_cont_enc)
        denoising_terms: list[torch.Tensor] = []

        if num_dim > 0:
            pred_num = torch.cat(con_outs, dim=1)
            observed_num = batch["con_mask"].bool()
            if observed_num.any():
                num_loss = F.mse_loss(pred_num[observed_num], batch["target_num"][observed_num])
                denoising_terms.append(float(args.numeric_denoising_weight) * num_loss)
                stats["num_loss"] = float(num_loss.detach().cpu().item())

        if cat_dim > 0:
            cat_loss_sum = None
            active_columns = 0
            for cat_idx in range(cat_dim):
                if args.categorical_denoising_scope == "official-all-imputed":
                    active = torch.ones(
                        batch["target_cat"].shape[0], dtype=torch.bool, device=batch["target_cat"].device
                    )
                    targets = x_categ[:, cat_idx + 1]
                else:
                    active = batch["cat_mask"][:, cat_idx + 1].bool()
                    targets = batch["target_cat"][:, cat_idx]
                if not active.any():
                    continue
                column_loss = F.cross_entropy(cat_outs[cat_idx + 1][active], targets[active])
                cat_loss_sum = column_loss if cat_loss_sum is None else cat_loss_sum + column_loss
                active_columns += 1
            if cat_loss_sum is not None:
                denoising_terms.append(float(args.categorical_denoising_weight) * cat_loss_sum)
                stats["cat_loss"] = float(cat_loss_sum.detach().cpu().item())
                stats["active_cat_columns"] = float(active_columns)

        if denoising_terms:
            denoising = sum(denoising_terms)
            loss_terms.append(denoising)
            stats["denoising_loss"] = float(denoising.detach().cpu().item())

    if not loss_terms:
        return None, stats
    return sum(loss_terms), stats


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def predict_split(
    model: DeepIFSAC,
    loader: DataLoader,
    bundle: DeepIFSACBenchmarkBundle,
    split_name: str,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    pred_num_chunks = []
    pred_cat_chunks = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            cat_outs, con_outs = forward_outputs(model, batch)
            if bundle.num_dim > 0:
                pred_num = torch.cat(con_outs, dim=1)
            else:
                pred_num = batch["x_cont"].new_zeros((batch["x_cont"].size(0), 0))

            if bundle.cat_dim > 0:
                pred_labels = []
                for cat_idx in range(bundle.cat_dim):
                    pred_labels.append(cat_outs[cat_idx + 1].argmax(dim=1, keepdim=True).float())
                pred_cat = torch.cat(pred_labels, dim=1)
            else:
                pred_cat = batch["x_cont"].new_zeros((batch["x_cont"].size(0), 0))

            pred_num_chunks.append(pred_num.detach().cpu().numpy())
            pred_cat_chunks.append(pred_cat.detach().cpu().numpy())

    pred_num_norm = np.concatenate(pred_num_chunks, axis=0) if pred_num_chunks else np.zeros((0, bundle.num_dim))
    pred_cat_idx = np.concatenate(pred_cat_chunks, axis=0) if pred_cat_chunks else np.zeros((0, bundle.cat_dim))
    pred_raw = compose_raw_prediction(bundle, split_name, pred_num_norm, pred_cat_idx)
    return evaluate_predictions(bundle, pred_raw, split_name), pred_raw


def run_dir_for(args: argparse.Namespace, dataset: str, split_idx: int) -> Path:
    return args.output_root / dataset / f"rate{args.ratio}" / args.mask_type / f"split_{split_idx}" / args.run_tag


def load_completed_report(run_dir: Path) -> dict | None:
    report_path = run_dir / "report.json"
    if not report_path.exists():
        return None
    import json

    with open(report_path, "r", encoding="utf-8") as file:
        report = json.load(file)
    if not report.get("completed"):
        return None
    if report.get("training_objective") != "deepifsac_cutmix_denoising_contrastive":
        return None
    test_metrics = report.get("test_metrics") or {}
    expected_numeric_source = "raw_csv_numeric_with_deepifsac_native_train_zscore_model_input"
    expected_scaler_source = "model_input_train_zscore_and_metric_prepared_npz_mean_std"
    expected_cat_protocol = "deepifsac_native_label_embedding_and_category_index_metric"
    if report.get("numeric_source") != expected_numeric_source:
        return None
    if report.get("scaler_source") != expected_scaler_source:
        return None
    if report.get("categorical_protocol") != expected_cat_protocol:
        return None
    if test_metrics.get("cat_f1_average") != "weighted_over_column_prefixed_category_indices":
        return None
    if "RMSE" not in test_metrics or "MAE" not in test_metrics or "F1" not in test_metrics:
        return None
    return report


def write_summary_csv(path: Path, reports: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "mask_type",
        "split_idx",
        "run_tag",
        "best_epoch",
        "best_score",
        "test_rmse",
        "test_mae",
        "test_cat_f1",
        "test_cat_accuracy",
        "test_accuracy",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            test_metrics = report.get("test_metrics") or {}
            writer.writerow(
                {
                    "dataset": report.get("dataset"),
                    "mask_type": report.get("mask_type"),
                    "split_idx": report.get("split_idx"),
                    "run_tag": report.get("run_tag"),
                    "best_epoch": report.get("best_epoch"),
                    "best_score": report.get("best_score"),
                    "test_rmse": test_metrics.get("rmse"),
                    "test_mae": test_metrics.get("mae"),
                    "test_cat_f1": test_metrics.get("cat_f1", test_metrics.get("bit_f1")),
                    "test_cat_accuracy": test_metrics.get("cat_accuracy"),
                    "test_accuracy": test_metrics.get("accuracy"),
                }
            )


def run_single_split(args: argparse.Namespace, dataset: str, split_idx: int) -> dict:
    seed = int(args.seed) + int(split_idx)
    set_seed(seed)
    run_dir = run_dir_for(args, dataset, split_idx)
    if (args.skip_completed or args.resume_incomplete) and not args.overwrite:
        completed = load_completed_report(run_dir)
        if completed is not None:
            print(f"[skip] dataset={dataset} split={split_idx} report={run_dir / 'report.json'}", flush=True)
            return completed

    device = resolve_device(args.device)
    bundle = DeepIFSACBenchmarkBundle(
        dataset_name=dataset,
        mask_type=args.mask_type,
        split_idx=split_idx,
        ratio=args.ratio,
        data_root=args.data_root,
        seed=seed,
        valid_max_fraction=args.valid_max_fraction,
    )
    if bundle.valid_split.num_samples == 0:
        raise RuntimeError(f"dataset={dataset} split={split_idx} has no valid samples.")

    num_fill, cat_fill, train_num_bank = build_initial_imputation(bundle)

    train_ds = DeepIFSACSameProtocolDataset(bundle.train_split, bundle.num_dim)
    valid_ds = DeepIFSACSameProtocolDataset(bundle.valid_split, bundle.num_dim)
    test_ds = DeepIFSACSameProtocolDataset(bundle.test_split, bundle.num_dim)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args, bundle).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.max_epochs), eta_min=args.min_lr)
        if args.scheduler == "cosine"
        else None
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best.pt"
    last_ckpt = ckpt_dir / "last.pt"
    history_path = run_dir / "history.json"
    report_path = run_dir / "report.json"

    config_payload = {
        "dataset": dataset,
        "mask_type": args.mask_type,
        "split_idx": split_idx,
        "ratio": args.ratio,
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "run_tag": args.run_tag,
        "seed": seed,
        "device": str(device),
        "selection_split": args.selection_split,
        "valid_max_fraction": args.valid_max_fraction,
        "max_epochs": args.max_epochs,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "min_lr": args.min_lr,
        "scheduler": args.scheduler,
        "embedding_size": args.embedding_size,
        "transformer_depth": args.transformer_depth,
        "attention_heads": args.attention_heads,
        "attentiontype": args.attentiontype,
        "training_objective": "deepifsac_cutmix_denoising_contrastive",
        "official_source_root": str(OFFICIAL_DEEPIFSAC_ROOT),
        "cutmix_enabled": not args.disable_cutmix,
        "cutmix_categorical_rate": args.cutmix_categorical_rate,
        "cutmix_continuous_rate": (
            args.cutmix_continuous_rate
            if args.cutmix_continuous_rate is not None
            else float(args.ratio) / 100.0
        ),
        "contrastive_enabled": not args.disable_contrastive,
        "denoising_enabled": not args.disable_denoising,
        "nce_temperature": args.nce_temperature,
        "contrastive_weight": args.contrastive_weight,
        "categorical_denoising_weight": args.categorical_denoising_weight,
        "numeric_denoising_weight": args.numeric_denoising_weight,
        "projection_style": args.projection_style,
        "categorical_denoising_scope": args.categorical_denoising_scope,
        "numeric_source": "raw_csv_numeric_with_deepifsac_native_train_zscore_model_input",
        "scaler_source": "model_input_train_zscore_and_metric_prepared_npz_mean_std",
        "metric_space": "eunseo_normalized",
        "categorical_protocol": "deepifsac_native_label_embedding_and_category_index_metric",
        "num_dim": bundle.num_dim,
        "cat_dim": bundle.cat_dim,
        "cat_cardinalities": bundle.cat_cardinalities.tolist(),
        "numeric_preprocessing": bundle.numeric_preprocessing,
        "categorical_preprocessing": bundle.categorical_preprocessing,
        "fairness_checks": {
            "split_id": split_idx,
            "same_train_valid_test_indices_from_prepared_split": True,
            "same_missing_mask_from_dataset_root": True,
            "same_evaluation_mask_from_dataset_root": True,
            "numeric_source": "raw_csv_numeric_with_deepifsac_native_train_zscore_model_input",
            "scaler_source": "model_input_train_zscore_and_metric_prepared_npz_mean_std",
            "metric_space": "eunseo_normalized",
            "categorical_protocol": "deepifsac_native_label_embedding_and_category_index_metric",
            "numeric_model_input_matches_deepifsac_native_train_zscore": True,
            "numeric_metric_uses_prepared_npz_mean_std": True,
            "numeric_inputs_equal_prepared_npz": False,
            "numeric_input_protocol": bundle.numeric_input_verification,
            "categorical_metrics_original_label_level": True,
            "baseline_uses_eunseo_codebook": False,
        },
        "split_sizes": bundle.split_sizes(),
        "score_formula": "rmse - 0.1 * cat_f1 - 0.05 * cat_accuracy",
        "objective_note": (
            "DeepIFSAC model, colrow attention, projection heads, CutMix, denoising, and "
            "bidirectional InfoNCE are retained. Only data loading, validation-only checkpoint "
            "selection, categorical missing-value decoding, and shared metrics are adapted."
        ),
    }
    save_json(run_dir / "config.json", config_payload)

    print(
        f"[fit] dataset={dataset} split={split_idx} "
        f"train={bundle.train_split.num_samples} valid={bundle.valid_split.num_samples} "
        f"test={bundle.test_split.num_samples} num_dim={bundle.num_dim} cat_dim={bundle.cat_dim} "
        f"device={device}",
        flush=True,
    )

    history: list[dict] = []
    best_score = None
    best_epoch = None
    best_train_metrics = None
    best_valid_metrics = None
    epochs_since_improvement = 0
    stopped_early = False
    stop_reason = None
    start_epoch = 1

    if args.resume_incomplete and not args.overwrite and last_ckpt.exists():
        checkpoint = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer_state = checkpoint.get("optimizer")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        scheduler_state = checkpoint.get("scheduler")
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        scaler_state = checkpoint.get("scaler")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        history = list(checkpoint.get("history", []))
        if not history and history_path.exists():
            import json

            with open(history_path, "r", encoding="utf-8") as file:
                history = json.load(file)
        best_score = checkpoint.get("best_score")
        best_epoch = checkpoint.get("best_epoch")
        best_train_metrics = checkpoint.get("best_train_metrics")
        best_valid_metrics = checkpoint.get("best_valid_metrics")
        epochs_since_improvement = int(checkpoint.get("epochs_since_improvement", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        restore_rng_state(checkpoint.get("rng_state"))
        print(
            f"[resume] dataset={dataset} split={split_idx} checkpoint={last_ckpt} "
            f"next_epoch={start_epoch} best_epoch={best_epoch}",
            flush=True,
        )

    use_amp = bool(args.amp and device.type == "cuda")
    for epoch in range(start_epoch, args.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_num_loss = []
        epoch_cat_loss = []
        epoch_contrastive_loss = []
        epoch_denoising_loss = []
        steps = 0
        skipped_steps = 0

        progress = tqdm(train_loader, desc=f"{dataset}-epoch-{epoch}", leave=False)
        for batch in progress:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss, loss_stats = official_ssl_loss(
                    model=model,
                    batch=batch,
                    num_dim=bundle.num_dim,
                    cat_dim=bundle.cat_dim,
                    num_fill=num_fill,
                    cat_fill=cat_fill,
                    train_num_bank=train_num_bank,
                    args=args,
                )
            if loss is None:
                skipped_steps += 1
                continue
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at dataset={dataset} split={split_idx} epoch={epoch}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            steps += 1
            epoch_loss += float(loss.detach().cpu().item())
            if loss_stats["num_loss"] == loss_stats["num_loss"]:
                epoch_num_loss.append(loss_stats["num_loss"])
            if loss_stats["cat_loss"] == loss_stats["cat_loss"]:
                epoch_cat_loss.append(loss_stats["cat_loss"])
            if loss_stats["contrastive_loss"] == loss_stats["contrastive_loss"]:
                epoch_contrastive_loss.append(loss_stats["contrastive_loss"])
            if loss_stats["denoising_loss"] == loss_stats["denoising_loss"]:
                epoch_denoising_loss.append(loss_stats["denoising_loss"])
            progress.set_postfix(loss=f"{epoch_loss / max(1, steps):.4f}")

        if scheduler is not None:
            scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": epoch_loss / max(1, steps),
            "train_num_loss": None if not epoch_num_loss else float(np.mean(epoch_num_loss)),
            "train_cat_loss": None if not epoch_cat_loss else float(np.mean(epoch_cat_loss)),
            "train_contrastive_loss": (
                None if not epoch_contrastive_loss else float(np.mean(epoch_contrastive_loss))
            ),
            "train_denoising_loss": (
                None if not epoch_denoising_loss else float(np.mean(epoch_denoising_loss))
            ),
            "steps": steps,
            "skipped_steps": skipped_steps,
        }

        if epoch % max(1, args.eval_every) == 0:
            train_metrics, train_pred_raw = predict_split(model, train_eval_loader, bundle, "train", device)
            valid_metrics, valid_pred_raw = predict_split(model, valid_loader, bundle, "valid", device)
            selection_metrics = valid_metrics
            current_score = score_from_metrics(selection_metrics)

            epoch_record.update(
                {
                    "score": current_score,
                    "train_mae": train_metrics.get("mae"),
                    "train_mse": train_metrics.get("mse"),
                    "train_rmse": train_metrics.get("rmse"),
                    "train_cat_f1": train_metrics.get("cat_f1", train_metrics.get("bit_f1")),
                    "train_cat_accuracy": train_metrics.get("cat_accuracy"),
                    "train_accuracy": train_metrics.get("accuracy"),
                    "valid_mae": valid_metrics.get("mae"),
                    "valid_mse": valid_metrics.get("mse"),
                    "valid_rmse": valid_metrics.get("rmse"),
                    "valid_cat_f1": valid_metrics.get("cat_f1", valid_metrics.get("bit_f1")),
                    "valid_cat_accuracy": valid_metrics.get("cat_accuracy"),
                    "valid_accuracy": valid_metrics.get("accuracy"),
                }
            )

            improved = best_score is None or current_score < (best_score - args.early_stop_min_delta)
            epoch_record["improved"] = bool(improved)
            if improved:
                best_score = current_score
                best_epoch = epoch
                best_train_metrics = train_metrics
                best_valid_metrics = valid_metrics
                epochs_since_improvement = 0
                safe_torch_save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "best_score": best_score,
                        "train_metrics": train_metrics,
                        "valid_metrics": valid_metrics,
                        "config": config_payload,
                    },
                    best_ckpt,
                )
                np.save(run_dir / "best_train_pred_raw.npy", train_pred_raw)
                np.save(run_dir / "best_valid_pred_raw.npy", valid_pred_raw)
            else:
                epochs_since_improvement += 1

            print(
                f"[epoch] dataset={dataset} split={split_idx} epoch={epoch}/{args.max_epochs} "
                f"loss={epoch_record['train_loss']:.6f} score={current_score:.6f} "
                f"contrastive={epoch_record['train_contrastive_loss']} "
                f"denoising={epoch_record['train_denoising_loss']} "
                f"valid_rmse={valid_metrics.get('rmse')} "
                f"valid_f1={valid_metrics.get('cat_f1', valid_metrics.get('bit_f1'))} "
                f"valid_acc={valid_metrics.get('cat_accuracy')} best_epoch={best_epoch}",
                flush=True,
            )

            if (
                args.early_stop_patience > 0
                and epochs_since_improvement >= args.early_stop_patience
            ):
                stopped_early = True
                stop_reason = (
                    "early_stop_patience_exhausted"
                    f"(patience={args.early_stop_patience}, "
                    f"min_delta={args.early_stop_min_delta}, "
                    f"epochs_since_improvement={epochs_since_improvement})"
                )
                history.append(epoch_record)
                save_json(history_path, history)
                safe_torch_save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": None if scheduler is None else scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch,
                        "best_score": best_score,
                        "best_epoch": best_epoch,
                        "best_train_metrics": best_train_metrics,
                        "best_valid_metrics": best_valid_metrics,
                        "epochs_since_improvement": epochs_since_improvement,
                        "history": history,
                        "rng_state": capture_rng_state(),
                        "config": config_payload,
                    },
                    last_ckpt,
                )
                print(
                    f"[early-stop] dataset={dataset} split={split_idx} epoch={epoch} "
                    f"best_epoch={best_epoch} selection_split={args.selection_split}",
                    flush=True,
                )
                break

        history.append(epoch_record)
        save_json(history_path, history)
        safe_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": None if scheduler is None else scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_train_metrics": best_train_metrics,
                "best_valid_metrics": best_valid_metrics,
                "epochs_since_improvement": epochs_since_improvement,
                "history": history,
                "rng_state": capture_rng_state(),
                "config": config_payload,
            },
            last_ckpt,
        )

    if stop_reason is None:
        stop_reason = "max_epochs_reached" if not stopped_early else stop_reason

    if best_ckpt.exists():
        checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        best_epoch = checkpoint.get("epoch", best_epoch)
        best_score = checkpoint.get("best_score", best_score)
        best_train_metrics = checkpoint.get("train_metrics", best_train_metrics)
        best_valid_metrics = checkpoint.get("valid_metrics", best_valid_metrics)
    else:
        best_epoch = len(history)

    print(f"[final-eval] dataset={dataset} split={split_idx} best_epoch={best_epoch}", flush=True)
    train_metrics, train_pred_raw = predict_split(model, train_eval_loader, bundle, "train", device)
    valid_metrics, valid_pred_raw = predict_split(model, valid_loader, bundle, "valid", device)
    test_metrics, test_pred_raw = predict_split(model, test_loader, bundle, "test", device)
    final_best_score = score_from_metrics(valid_metrics)

    np.save(run_dir / "train_pred_raw.npy", train_pred_raw)
    np.save(run_dir / "valid_pred_raw.npy", valid_pred_raw)
    np.save(run_dir / "test_pred_raw.npy", test_pred_raw)

    best_report = {
        "epoch": best_epoch,
        "score": final_best_score,
        "selection_split": args.selection_split,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }
    report = {
        "dataset": dataset,
        "mask_type": args.mask_type,
        "split_idx": split_idx,
        "ratio": args.ratio,
        "run_tag": args.run_tag,
        "run_dir": str(run_dir),
        "history_path": str(history_path),
        "numeric_source": "raw_csv_numeric_with_deepifsac_native_train_zscore_model_input",
        "scaler_source": "model_input_train_zscore_and_metric_prepared_npz_mean_std",
        "metric_space": "eunseo_normalized",
        "categorical_protocol": "deepifsac_native_label_embedding_and_category_index_metric",
        "training_objective": "deepifsac_cutmix_denoising_contrastive",
        "official_source_root": str(OFFICIAL_DEEPIFSAC_ROOT),
        "best_epoch": best_epoch,
        "best_score": final_best_score,
        "best_report": best_report,
        "selection_split": args.selection_split,
        "split_sizes": bundle.split_sizes(),
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "completed": True,
        "num_dim": bundle.num_dim,
        "cat_dim": bundle.cat_dim,
        "cat_cardinalities": bundle.cat_cardinalities.tolist(),
        "numeric_preprocessing": bundle.numeric_preprocessing,
        "categorical_preprocessing": bundle.categorical_preprocessing,
        "fairness_checks": config_payload["fairness_checks"],
    }
    save_json(report_path, report)
    if report.get("completed") and not args.keep_last_checkpoint and last_ckpt.exists():
        try:
            last_ckpt.unlink()
        except OSError as exc:
            print(f"[warn] failed to remove completed-run checkpoint {last_ckpt}: {exc}", flush=True)
    print(
        f"[done] dataset={dataset} split={split_idx} test_rmse={test_metrics.get('rmse')} "
        f"test_f1={test_metrics.get('cat_f1', test_metrics.get('bit_f1'))} "
        f"test_acc={test_metrics.get('cat_accuracy')}",
        flush=True,
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def main() -> None:
    args = parse_args()
    reports = []
    for dataset in args.datasets:
        for split_idx in args.splits:
            reports.append(run_single_split(args, dataset, split_idx))

    summary_root = args.output_root / "_summaries"
    save_json(summary_root / "deepifsac_official_ssl_protocol_reports.json", reports)
    write_summary_csv(summary_root / "deepifsac_official_ssl_protocol_reports.csv", reports)


if __name__ == "__main__":
    main()
