from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from settings import BASELINE_SETTINGS, COMMON_TRAINING


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "baselines"


def verify_sources(source_root: Path) -> list[str]:
    expected = json.loads((BASELINES / "official_sources.json").read_text(encoding="utf-8"))
    messages = []
    for name, metadata in expected.items():
        checkout = source_root / name
        if not (checkout / ".git").exists():
            raise FileNotFoundError(f"Missing official source checkout: {checkout}")
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != metadata["commit"]:
            raise RuntimeError(f"{name}: expected {metadata['commit']}, found {actual}")
        messages.append(f"{name}: {actual}")
    return messages


def verify_settings() -> None:
    assert COMMON_TRAINING == {
        "max_epochs": 600,
        "early_stop_patience": 45,
        "early_stop_min_delta": 0.0002,
        "checkpoint_selection": "validation_only",
    }
    assert BASELINE_SETTINGS["remasker"]["weight_decay"] == 0.01
    assert BASELINE_SETTINGS["pmae"]["batch_size"] == 64
    assert BASELINE_SETTINGS["miwae"]["architecture"]["observation_model"] == "student_t"
    assert BASELINE_SETTINGS["miwae"]["architecture"]["latent_dim"] == "ceil(input_dim / 2)"
    assert BASELINE_SETTINGS["deepifsac"]["method"]["cutmix_continuous_rate"] == 0.3
    assert BASELINE_SETTINGS["missforest"]["trees"] == 100


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the Appendix B baseline protocol.")
    parser.add_argument(
        "--source-root", type=Path, default=ROOT / ".baseline_sources"
    )
    parser.add_argument("--skip-sources", action="store_true")
    args = parser.parse_args()

    verify_settings()
    print("[verified] Appendix B.4-B.7 settings")
    if not args.skip_sources:
        for message in verify_sources(args.source_root.resolve()):
            print(f"[verified] {message}")


if __name__ == "__main__":
    main()
