from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def resolve_device(requested: str) -> torch.device:
    """Return the requested device, falling back to CPU when needed."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_torch_performance_flags(device: torch.device) -> None:
    """Enable standard PyTorch performance settings."""
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def _make_json_safe(payload: Any) -> Any:
    if is_dataclass(payload):
        payload = asdict(payload)
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {key: _make_json_safe(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_make_json_safe(value) for value in payload]
    return payload


def save_json(path: Path, payload: Any) -> None:
    """Save common Python containers and dataclasses as readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(_make_json_safe(payload), file, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def append_text_line(path: Path, text: str) -> None:
    """Append one line to a log file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(text.rstrip() + "\n")


def atomic_torch_save(path: Path, payload: Any) -> None:
    """Write checkpoints through a temporary file and atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(5):
        # Use a unique temp file so concurrent/resumed runs cannot steal each other's
        # checkpoint before the atomic replace on Windows.
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
            return
        except OSError as error:
            last_error = error
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            time.sleep(0.2 * (attempt + 1))
    raise last_error if last_error is not None else RuntimeError(f"Failed to save checkpoint: {path}")


def make_versioned_dir(base_dir: Path) -> Path:
    """
    Create v1, v2, and later directories to avoid overwriting prior runs.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    version = 1
    while True:
        candidate = base_dir / f"v{version}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        version += 1


def capture_rng_state() -> dict[str, Any]:
    """Capture random-number generator states for exact resumption."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore random states, including dropout and shuffling streams."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = state["torch"]
    if isinstance(torch_state, torch.Tensor):
        torch_state = torch_state.detach().to(device="cpu", dtype=torch.uint8)
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "torch_cuda" in state:
        cuda_states = []
        for cuda_state in state["torch_cuda"]:
            if isinstance(cuda_state, torch.Tensor):
                cuda_state = cuda_state.detach().to(device="cpu", dtype=torch.uint8)
            cuda_states.append(cuda_state)
        torch.cuda.set_rng_state_all(cuda_states)


def format_float(value: float | None) -> str:
    if value is None:
        return "None"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return f"{value}"
    return f"{value:.6f}"


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Average values only where the mask equals one."""
    mask = mask.float()
    denom = mask.sum().clamp_min(eps)
    return (values * mask).sum() / denom
