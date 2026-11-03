from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "_shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from official_r_valid_es_benchmark import run_cli  # noqa: E402


if __name__ == "__main__":
    run_cli(
        method="missforest",
        r_script=HERE / "r" / "official_missforest_train_only_valid_es.R",
        default_output_root=str(HERE.parents[1] / "results" / "missforest"),
    )
