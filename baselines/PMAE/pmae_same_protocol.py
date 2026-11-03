import copy
import math
import random
import time
from argparse import Namespace
from functools import partial
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

import pMAE
from utils import MAEDataset, adjust_learning_rate


EPS = 1e-8


DATASET_CONFIGS = {
    "abalone": dict(embed_dim=32, depth=4, decoder_depth=2, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "adult": dict(embed_dim=32, depth=6, decoder_depth=4, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "bank": dict(embed_dim=32, depth=6, decoder_depth=4, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "bean": dict(embed_dim=32, depth=4, decoder_depth=2, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "breast": dict(embed_dim=32, depth=4, decoder_depth=2, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "california": dict(embed_dim=32, depth=6, decoder_depth=4, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "default": dict(embed_dim=32, depth=6, decoder_depth=4, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "german": dict(embed_dim=32, depth=6, decoder_depth=4, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "gesture": dict(embed_dim=32, depth=6, decoder_depth=4, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "letter": dict(embed_dim=64, depth=8, decoder_depth=6, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "magic": dict(embed_dim=32, depth=4, decoder_depth=2, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "mushroom": dict(embed_dim=64, depth=8, decoder_depth=6, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "news": dict(embed_dim=64, depth=8, decoder_depth=6, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
    "shoppers": dict(embed_dim=32, depth=6, decoder_depth=4, num_heads=4, mlp_ratio=4.0, mask_ratio=0.5),
}


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PMAEProtocolImputer:
    """PMAE training wrapper with valid-based checkpoint selection."""

    def __init__(self, args: Namespace) -> None:
        config = DATASET_CONFIGS.get(args.dataset, DATASET_CONFIGS["california"])

        self.batch_size = int(args.batch_size)
        self.eval_batch_size = int(getattr(args, "eval_batch_size", 512))
        self.accum_iter = int(args.accum_iter)
        self.min_lr = float(args.min_lr)
        self.norm_field_loss = bool(args.norm_field_loss)
        self.weight_decay = float(args.weight_decay)
        self.lr = args.lr
        self.blr = float(args.blr)
        self.warmup_epochs = int(args.warmup_epochs)
        self.model = None
        self.norm_parameters = None
        self.input_normalization = getattr(args, "input_normalization", "minmax")
        if self.input_normalization not in {"identity", "minmax"}:
            raise ValueError("input_normalization must be 'identity' or 'minmax'")
        self.dataset_name = args.dataset
        self.log_interval = max(1, int(getattr(args, "log_interval", 1)))
        self.use_amp = bool(getattr(args, "amp", False)) and resolve_device(args.device).type == "cuda"
        self.max_grad_norm = float(getattr(args, "max_grad_norm", 5.0))
        self.max_epoch_retries = int(getattr(args, "max_epoch_retries", 3))

        self.embed_dim = config["embed_dim"] if args.embed_dim is None else int(args.embed_dim)
        self.depth = config["depth"] if args.depth is None else int(args.depth)
        self.decoder_depth = config["decoder_depth"] if args.decoder_depth is None else int(args.decoder_depth)
        self.num_heads = config["num_heads"] if args.num_heads is None else int(args.num_heads)
        self.mlp_ratio = config["mlp_ratio"] if args.mlp_ratio is None else float(args.mlp_ratio)
        self.max_epochs = int(args.max_epochs)
        self.mask_ratio = config["mask_ratio"] if args.mask_ratio is None else float(args.mask_ratio)
        self.encode_func = "linear" if args.encode_func is None else args.encode_func
        self.device = resolve_device(args.device)
        self.training_history = []
        self.optimizer = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.current_epoch = 0

        self.a = float(getattr(args, "pmae_a", 0.05))
        self.b = float(getattr(args, "pmae_b", 0.5))
        self.block_mlp = None
        self.old_loss = False

    def _fit_normalizer(self, x_raw: torch.Tensor) -> torch.Tensor:
        x = x_raw.clone().cpu()
        dim = int(x.shape[1])
        if self.input_normalization == "identity":
            self.norm_parameters = {
                "min": np.zeros(dim, dtype=np.float32),
                "max": np.ones(dim, dtype=np.float32),
                "method": "identity",
            }
            return x

        x_np = x.numpy().astype(np.float32, copy=True)
        min_val = np.zeros(dim, dtype=np.float32)
        max_val = np.ones(dim, dtype=np.float32)
        for col_idx in range(dim):
            column = x_np[:, col_idx]
            finite = np.isfinite(column)
            if not finite.any():
                min_val[col_idx] = 0.0
                max_val[col_idx] = 1.0
                continue
            min_val[col_idx] = float(np.nanmin(column))
            max_val[col_idx] = float(np.nanmax(column))
            if abs(max_val[col_idx] - min_val[col_idx]) < EPS:
                max_val[col_idx] = min_val[col_idx] + 1.0
            x[:, col_idx] = (x[:, col_idx] - min_val[col_idx]) / (max_val[col_idx] - min_val[col_idx] + EPS)

        self.norm_parameters = {
            "min": min_val,
            "max": max_val,
            "method": "pmae_native_minmax_fit_on_train_observed",
        }
        return x

    def _init_model(self, dim: int) -> None:
        self.model = pMAE.MaskedAutoencoder(
            rec_len=dim,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            decoder_embed_dim=self.embed_dim,
            decoder_depth=self.decoder_depth,
            decoder_num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=EPS),
            norm_field_loss=self.norm_field_loss,
            encode_func=self.encode_func,
            block_mlp=self.block_mlp,
            old_loss=self.old_loss,
        )
        self.model.to(self.device)

    def fit(
        self,
        x_raw: torch.Tensor,
        epoch_end_callback: Any | None = None,
        resume_state: dict | None = None,
    ) -> "PMAEProtocolImputer":
        x = self._fit_normalizer(x_raw)
        dim = int(x.shape[1])
        no = int(x.shape[0])

        observed_mask = (1 - (1 * torch.isnan(x))).float()
        x = torch.nan_to_num(x).float()

        self._init_model(dim)
        eff_batch_size = self.batch_size * self.accum_iter
        if self.lr is None:
            self.lr = self.blr * eff_batch_size / self.batch_size

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay,
        )
        self.optimizer = optimizer
        dataset = MAEDataset(x, observed_mask)
        dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=self.batch_size)
        start_epoch = 0

        if resume_state is not None:
            self.model.load_state_dict(resume_state["model_state_dict"])
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            scaler_state = resume_state.get("scaler_state_dict")
            if scaler_state:
                try:
                    self.scaler.load_state_dict(scaler_state)
                except RuntimeError as exc:
                    print(
                        f"[resume-scaler-skip] dataset={self.dataset_name} "
                        f"reason={exc}",
                        flush=True,
                    )
            else:
                print(
                    f"[resume-scaler-skip] dataset={self.dataset_name} "
                    "reason=empty_or_missing_grad_scaler_state",
                    flush=True,
                )
            if resume_state.get("norm_parameters") is not None:
                self.norm_parameters = resume_state["norm_parameters"]
            self.training_history = copy.deepcopy(resume_state.get("training_history", []))
            self.lr = float(resume_state.get("base_lr", self.lr))
            start_epoch = int(resume_state.get("epoch", 0))
            self.current_epoch = start_epoch
            rng_state = resume_state.get("rng_state")
            if rng_state is not None:
                random.setstate(rng_state["python"])
                np.random.set_state(rng_state["numpy"])
                torch.set_rng_state(rng_state["torch"].cpu())
                torch_cuda_state = rng_state.get("torch_cuda")
                if torch_cuda_state is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all([state.cpu() for state in torch_cuda_state])
            print(
                f"[resume] dataset={self.dataset_name} start_epoch={start_epoch + 1} "
                f"history_len={len(self.training_history)} base_lr={self.lr:.6g}",
                flush=True,
            )
        else:
            self.training_history = []
            self.current_epoch = 0

        fit_start = time.time()
        print(
            f"[train-start] dataset={self.dataset_name} rows={no} dim={dim} device={self.device} "
            f"batch_size={self.batch_size} eval_batch_size={self.eval_batch_size} epochs={self.max_epochs} "
            f"amp={self.use_amp}",
            flush=True,
        )

        for epoch in range(start_epoch, self.max_epochs):
            self.model.train()
            epoch_start = time.time()
            retry_count = 0

            while True:
                epoch_model_state = copy.deepcopy(self.model.state_dict())
                epoch_optimizer_state = copy.deepcopy(optimizer.state_dict())
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                epoch_failed = False
                iter_idx = 0

                for iter_idx, (samples, masks) in enumerate(dataloader):
                    if iter_idx % self.accum_iter == 0:
                        adjust_learning_rate(
                            optimizer,
                            iter_idx / len(dataloader) + epoch,
                            float(self.lr),
                            self.min_lr,
                            self.max_epochs,
                            self.warmup_epochs,
                        )

                    samples = samples.to(self.device, non_blocking=True)
                    masks = masks.to(self.device, non_blocking=True)

                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        obs_mean_vec = masks.float().mean(axis=0).float()
                        mask_ratio = 1.0 * (obs_mean_vec > 0.9999) + 1.0 * (obs_mean_vec <= 0.9999) * (
                            self.a * torch.log((obs_mean_vec + 1e-20) / (1 - obs_mean_vec + 1e-20)) + self.b
                        )
                        loss, _, _, _, _ = self.model(samples, masks, mask_ratio=mask_ratio)
                        loss_value = float(loss.item())

                    if not math.isfinite(loss_value):
                        epoch_failed = True
                        print(
                            f"[warn] dataset={self.dataset_name} epoch={epoch + 1} iter={iter_idx + 1} "
                            f"loss={loss_value} lr={optimizer.param_groups[0]['lr']:.6g} retry={retry_count + 1}",
                            flush=True,
                        )
                        break

                    total_loss += loss_value
                    loss = loss / self.accum_iter
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        if (iter_idx + 1) % self.accum_iter == 0:
                            self.scaler.unscale_(optimizer)
                            if self.max_grad_norm > 0:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                            self.scaler.step(optimizer)
                            self.scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                    else:
                        loss.backward()
                        if (iter_idx + 1) % self.accum_iter == 0:
                            if self.max_grad_norm > 0:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)

                if not epoch_failed:
                    total_loss = float(math.sqrt(total_loss / (iter_idx + 1)))
                    break

                retry_count += 1
                if retry_count > self.max_epoch_retries:
                    raise RuntimeError(
                        f"dataset={self.dataset_name} epoch={epoch + 1} exceeded retry limit at lr={self.lr}"
                    )

                self.model.load_state_dict(epoch_model_state)
                optimizer.load_state_dict(epoch_optimizer_state)
                self.lr = max(self.min_lr, float(self.lr) * 0.5)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.lr
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"[retry] dataset={self.dataset_name} epoch={epoch + 1} restarting with base_lr={self.lr:.6g}",
                    flush=True,
                )

            self.training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": total_loss,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
            )
            if ((epoch + 1) % self.log_interval == 0) or (epoch == 0) or (epoch + 1 == self.max_epochs):
                print(
                    f"[train] dataset={self.dataset_name} epoch={epoch + 1}/{self.max_epochs} "
                    f"loss={total_loss:.6f} lr={optimizer.param_groups[0]['lr']:.6g} "
                    f"epoch_sec={time.time() - epoch_start:.1f} total_sec={time.time() - fit_start:.1f}",
                    flush=True,
                )

            if epoch_end_callback is not None:
                should_stop = bool(epoch_end_callback(self, epoch + 1, self.training_history[-1]))
                if should_stop:
                    print(f"[train-stop] dataset={self.dataset_name} epoch={epoch + 1}/{self.max_epochs}", flush=True)
                    break
            self.current_epoch = epoch + 1

        print(
            f"[train-end] dataset={self.dataset_name} epochs={len(self.training_history)} "
            f"total_sec={time.time() - fit_start:.1f}",
            flush=True,
        )
        return self

    def transform(self, x_raw: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
        if self.model is None or self.norm_parameters is None:
            raise RuntimeError("Model has not been trained yet.")

        x_original = x_raw.clone().cpu().float()
        x = x_original.clone()
        min_val = torch.as_tensor(self.norm_parameters["min"], dtype=torch.float32)
        max_val = torch.as_tensor(self.norm_parameters["max"], dtype=torch.float32)
        if self.norm_parameters.get("method") != "identity":
            x = (x - min_val[None, :]) / (max_val[None, :] - min_val[None, :] + EPS)
        observed_mask = (1 - (1 * torch.isnan(x))).float()
        x = torch.nan_to_num(x).float()

        loader = DataLoader(
            TensorDataset(x, observed_mask),
            batch_size=self.eval_batch_size if batch_size is None else int(batch_size),
            shuffle=False,
        )
        preds = []
        self.model.eval()
        with torch.no_grad():
            for samples, masks in loader:
                samples = samples.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                _, pred, _, _, _ = self.model(samples, masks)
                preds.append(pred.squeeze(dim=2).detach().cpu())

        imputed_data = torch.cat(preds, dim=0)
        if self.norm_parameters.get("method") != "identity":
            imputed_data = imputed_data * (max_val[None, :] - min_val[None, :] + EPS) + min_val[None, :]
        if np.all(np.isnan(imputed_data.numpy())):
            raise RuntimeError("The imputed result contains only NaN values.")
        return observed_mask.cpu() * torch.nan_to_num(x_original) + (1 - observed_mask.cpu()) * imputed_data

    def training_state_dict(self, epoch: int | None = None) -> dict[str, Any]:
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Training state is unavailable before fit starts.")
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "norm_parameters": self.norm_parameters,
            "training_history": copy.deepcopy(self.training_history),
            "base_lr": float(self.lr),
            "epoch": len(self.training_history) if epoch is None else int(epoch),
        }

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> "PMAEProtocolImputer":
        if self.model is None:
            raise RuntimeError("Model has not been initialized yet.")
        self.model.load_state_dict(state_dict)
        return self
