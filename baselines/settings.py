"""Appendix B.4-B.7 settings used by the public baseline adapters."""

from __future__ import annotations

COMMON_TRAINING = {
    "max_epochs": 600,
    "early_stop_patience": 45,
    "early_stop_min_delta": 0.0002,
    "checkpoint_selection": "validation_only",
}

BASELINE_SETTINGS = {
    "remasker": {
        "optimizer": "AdamW",
        "betas": (0.9, 0.95),
        "lr": 1e-3,
        "weight_decay": 0.01,
        "batch_size": 64,
        "architecture": {
            "heads": 4,
            "mlp_ratio": 4.0,
            "dataset_dependent_depth": True,
        },
        "method": {
            "mask_ratio": 0.5,
            "scheduler": "cosine",
            "warmup_epochs": 40,
            "min_lr": 1e-5,
        },
        "categorical": "one_hot_groups_then_groupwise_argmax",
    },
    "pmae": {
        "optimizer": "AdamW",
        "betas": (0.9, 0.95),
        "lr": 1e-3,
        "weight_decay": 0.01,
        "batch_size": 64,
        "architecture": {
            "heads": 4,
            "mlp_ratio": 4.0,
            "dataset_dependent_depth": True,
        },
        "method": {
            "masking": "proportional",
            "a": 0.05,
            "b": 0.5,
            "scheduler": "cosine",
            "warmup_epochs": 40,
            "min_lr": 1e-5,
        },
        "categorical": "integer_scalar_minmax_then_inverse_round",
    },
    "cacti": {
        "optimizer": "AdamW",
        "betas": (0.9, 0.95),
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "batch_size": 128,
        "architecture": {
            "embedding_dim": 64,
            "encoder_depth": 10,
            "decoder_depth": 4,
            "heads": 8,
            "column_context": True,
        },
        "method": {
            "masking": "median_truncated_copy",
            "mask_ratio": 0.9,
            "scheduler": "stepwise_warmup_cosine",
            "warmup_epochs": 50,
            "min_lr": 5e-6,
            "gradient_clip": 5.0,
        },
        "categorical": "ordinal_scalar_minmax_then_inverse_round_clip",
    },
    "gain": {
        "optimizer": "Adam",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 128,
        "architecture": {
            "generator_hidden_layers": 2,
            "discriminator_hidden_layers": 2,
            "hidden_width": "input_dim",
            "hidden_activation": "relu",
            "output_activation": "sigmoid",
        },
        "method": {"hint_rate": 0.9, "alpha": 100.0, "input_noise": (0.0, 0.01)},
        "categorical": "integer_scalar_minmax_then_inverse_round",
    },
    "miwae": {
        "optimizer": "Adam",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 64,
        "architecture": {
            "hidden_dim": 128,
            "encoder_hidden_layers": 2,
            "decoder_hidden_layers": 2,
            "activation": "relu",
            "observation_model": "student_t",
            "latent_dim": "ceil(input_dim / 2)",
        },
        "method": {
            "training_importance_samples": 20,
            "validation_imputation_samples": 1000,
            "final_imputation_samples": 10000,
        },
        "categorical": "train_vocabulary_one_hot_then_groupwise_argmax",
    },
    "dsan": {
        "optimizer": "Adam",
        "lr": 3e-3,
        "weight_decay": 1e-5,
        "batch_size": 128,
        "architecture": {
            "representation_dim": 32,
            "heads": 8,
            "n_hidden": 2,
            "dropout": 0.5,
        },
        "method": {"denoising_corruption": 0.30},
        "categorical": "column_embeddings_with_bce_or_cross_entropy",
    },
    "deepifsac": {
        "optimizer": "AdamW",
        "lr": 1e-4,
        "weight_decay": 1e-2,
        "batch_size": 128,
        "architecture": {
            "embedding_dim": 32,
            "depth": 6,
            "heads": 8,
            "attention": "colrow",
            "dropout": 0.1,
        },
        "method": {
            "cutmix_categorical_rate": 0.3,
            "cutmix_continuous_rate": 0.3,
            "temperature": 0.7,
            "contrastive_weight": 0.5,
            "categorical_denoising_weight": 1.0,
            "numeric_denoising_weight": 10.0,
        },
        "categorical": "native_column_embeddings_and_cross_entropy_heads",
    },
    "mice": {
        "implementation": "mice 3.19.0",
        "max_iterations": 10,
        "m": 1,
        "methods": ("pmm", "logreg", "polyreg", "polr"),
        "nnet_maxnwts": 100000,
    },
    "missforest": {
        "implementation": "missForest 1.6.1 with ranger",
        "max_iterations": 10,
        "trees": 100,
        "mtry": "floor(sqrt(p))",
        "replace": True,
        "sample_fraction": 1.0,
    },
}


def get_settings(method: str) -> dict:
    key = method.strip().lower()
    if key not in BASELINE_SETTINGS:
        raise KeyError(f"Unknown baseline: {method}")
    return BASELINE_SETTINGS[key]
