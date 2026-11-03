from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results"
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
class TOSEConfig:
    dataset: str
    split_idx: int
    mask_type: str
    ratio: str
    data_root: Path
    output_root: Path
    run_tag: str
    device: str
    seed: int
    batch_size: int
    eval_batch_size: int
    max_epochs: int
    early_stop_patience: int
    early_stop_min_delta: float
    selection_split: str
    valid_max_fraction: float
    lr: float
    weight_decay: float
    min_lr: float
    restart_epochs: int
    restart_mult: int
    grad_clip: float
    amp: bool
    checkpoint_every_steps: int
    log_every_steps: int
    embed_dim: int
    encoder_depth: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    mechanism_hidden_dim: int
    cross_attn_layers: int
    ablation_mode: str
    guided_remask_ratio: float
    guided_remask_temperature: float
    forward_keep_loss_weight: float
    forward_aux_loss_weight: float
    loss_balance_mode: str
    lambda_num: float
    lambda_cat: float
    cat_bit_beta: float
    cat_direct_ce_loss_weight: float
    cat_codebook_loss_weight: float
    cat_bit_aux_loss_weight: float
    resume_checkpoint: str | None


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TOSE mixed-type tabular imputation benchmark"
    )
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--splits", nargs="*", type=int, default=[0])
    parser.add_argument("--mask-type", type=str, default="MCAR_v2_strict")
    parser.add_argument("--ratio", type=str, default="30")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-tag", type=str, default="tose_full")
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--force-new-version", action="store_true")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--early-stop-patience", type=int, default=45)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0002)
    parser.add_argument("--selection-split", type=str, default="valid", choices=["valid"])
    parser.add_argument("--valid-max-fraction", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=0.005)
    parser.add_argument("--min-lr", type=float, default=0.000005)
    parser.add_argument("--restart-epochs", type=int, default=30)
    parser.add_argument("--restart-mult", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpoint-every-steps", type=int, default=500)
    parser.add_argument("--log-every-steps", type=int, default=500)

    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--encoder-depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument("--mechanism-hidden-dim", type=int, default=64)
    parser.add_argument("--cross-attn-layers", type=int, default=3)
    parser.add_argument(
        "--ablation-mode",
        type=str,
        default="full",
        choices=[
            "full",
            "no_target_conditioning",
            "no_observed_self_attention",
            "no_state_embedding",
            "no_tose_shared_row",
            "uniform_source_aggregation",
            "uniform_source_no_target_set_sa",
            "no_target_aware_pooling",
            "both_uniform",
            "tose_1_block",
            "tose_2_blocks",
            "tose_4_blocks",
            "observed1_tose_1_block",
            "observed1_tose_3_blocks",
            "observed1_tose_6_blocks",
            "observed1_tose1_uniform_source",
            "observed1_tose1_uniform_source_no_target_set_sa",
            "observed1_tose1_uniform_source_no_observed_set_sa",
            "observed1_tose1_all_three_removed",
            "observed1_tose1_no_target_set_sa",
            "observed1_tose1_no_observed_set_sa",
            "observed1_tose1_shared_target_query",
            "observed1_tose1_static_column_attention",
            "linear_prediction_heads",
        ],
    )

    parser.add_argument("--guided-remask-ratio", type=float, default=0.25)
    parser.add_argument("--guided-remask-temperature", type=float, default=0.5)
    parser.add_argument("--forward-keep-loss-weight", type=float, default=1.0)
    parser.add_argument("--forward-aux-loss-weight", type=float, default=0.6)
    parser.add_argument("--loss-balance-mode", type=str, default="legacy", choices=["legacy", "type_mean"])
    parser.add_argument("--lambda-num", type=float, default=0.5)
    parser.add_argument("--lambda-cat", type=float, default=0.5)
    parser.add_argument("--cat-bit-beta", type=float, default=0.0)
    parser.add_argument("--cat-direct-ce-loss-weight", type=float, default=1.0)
    parser.add_argument("--cat-codebook-loss-weight", type=float, default=0.0)
    parser.add_argument("--cat-bit-aux-loss-weight", type=float, default=0.0)
    return parser


def build_config(args: argparse.Namespace, dataset: str, split_idx: int) -> TOSEConfig:
    return TOSEConfig(
        dataset=dataset,
        split_idx=split_idx,
        mask_type=args.mask_type,
        ratio=str(args.ratio),
        data_root=args.data_root,
        output_root=args.output_root,
        run_tag=args.run_tag,
        device=args.device,
        seed=args.seed + split_idx,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_epochs=args.max_epochs,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        selection_split=args.selection_split,
        valid_max_fraction=args.valid_max_fraction,
        lr=args.lr,
        weight_decay=args.weight_decay,
        min_lr=args.min_lr,
        restart_epochs=args.restart_epochs,
        restart_mult=args.restart_mult,
        grad_clip=args.grad_clip,
        amp=bool(args.amp),
        checkpoint_every_steps=args.checkpoint_every_steps,
        log_every_steps=args.log_every_steps,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        mechanism_hidden_dim=args.mechanism_hidden_dim,
        cross_attn_layers=args.cross_attn_layers,
        ablation_mode=args.ablation_mode,
        guided_remask_ratio=args.guided_remask_ratio,
        guided_remask_temperature=args.guided_remask_temperature,
        forward_keep_loss_weight=args.forward_keep_loss_weight,
        forward_aux_loss_weight=args.forward_aux_loss_weight,
        loss_balance_mode=args.loss_balance_mode,
        lambda_num=args.lambda_num,
        lambda_cat=args.lambda_cat,
        cat_bit_beta=args.cat_bit_beta,
        cat_direct_ce_loss_weight=args.cat_direct_ce_loss_weight,
        cat_codebook_loss_weight=args.cat_codebook_loss_weight,
        cat_bit_aux_loss_weight=args.cat_bit_aux_loss_weight,
        resume_checkpoint=args.resume_checkpoint,
    )
