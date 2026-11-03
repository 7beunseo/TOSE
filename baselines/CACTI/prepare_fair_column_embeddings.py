from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from fair_protocol_contexts import ordered_column_contexts


DATASETS = [
    "adult", "bean", "default", "gesture", "letter", "magic", "news", "shoppers",
    "abalone", "bank", "breast", "california", "german", "mushroom",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare official CACTI column-context embeddings.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--embed-model", default="Alibaba-NLP/gte-en-mlm-large")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_KERAS", "1")

    # The upstream GTE checkpoint pins Transformers 4.37.2. Keep that compatibility
    # bundle local to CACTI so the shared benchmark environment is not downgraded.
    compatibility_root = Path(__file__).resolve().parent / "fair_protocol" / "transformers_4_37"
    if compatibility_root.exists():
        sys.path.insert(0, str(compatibility_root))

    from transformers import AutoModel, AutoTokenizer
    import transformers

    device = choose_device(args.device)
    print(
        f"[embedding-model] name={args.embed_model} device={device} "
        f"transformers={transformers.__version__}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.embed_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.embed_model, trust_remote_code=True)
    model.to(device)
    model.eval()

    for dataset in args.datasets:
        info_path = args.data_root / "Info" / f"{dataset}.json"
        with info_path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
        colnames, descriptions = ordered_column_contexts(dataset, info)
        dataset_dir = args.output_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        output_path = dataset_dir / "embeddings_colnames_GTEenMLMlarge.npz"

        if output_path.exists() and not args.overwrite:
            try:
                existing = np.load(output_path)
                if existing["colembed"].shape[0] == len(colnames):
                    print(f"[embedding-skip] dataset={dataset} path={output_path}", flush=True)
                    continue
            except (OSError, KeyError, ValueError):
                pass

        vectors: list[np.ndarray] = []
        with torch.inference_mode():
            for idx, description in enumerate(descriptions):
                context = f"{colnames[idx]} (is defined as) : {description}"
                tokens = tokenizer(
                    context,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                tokens = {key: value.to(device) for key, value in tokens.items()}
                output = model(**tokens)
                # The official CACTI utility mean-pools all tokens for each column independently.
                vectors.append(output.last_hidden_state.mean(dim=1).squeeze(0).float().cpu().numpy())

        embeddings = np.stack(vectors).astype(np.float32)
        tmp_path = output_path.with_suffix(".tmp.npz")
        np.savez(tmp_path, colembed=embeddings, colnames=np.asarray(colnames))
        os.replace(tmp_path, output_path)
        print(
            f"[embedding-done] dataset={dataset} shape={embeddings.shape} path={output_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
