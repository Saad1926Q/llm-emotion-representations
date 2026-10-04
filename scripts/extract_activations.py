from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from emotion_vectors import MODEL_ID
from emotion_vectors.extraction import extract_activations
from emotion_vectors.generation import load_model_and_tokenizer

DEFAULT_RUN_DIR = Path("data/runs/qwen35_2b")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]




def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and save response activations from generated stories."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"Generation run directory (default: {DEFAULT_RUN_DIR}).",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override the model ID stored in the generation records.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Torch dtype used while loading the model.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device map (default: auto).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Extract only the first N generation records; useful for a pilot run.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    generations_path = args.run_dir / "generations.jsonl"
    if not generations_path.exists():
        raise SystemExit(f"generation file does not exist: {generations_path}")

    records = _read_jsonl(generations_path)
    if not records:
        raise SystemExit(f"generation file is empty: {generations_path}")
    if args.limit is not None:
        records = records[: args.limit]

    records = [
        record
        for record in records
        if record.get("validation", {}).get("valid", True)
    ]
    if not records:
        print("no valid records to extract")
        return

    model_id = args.model_id or records[0].get("model_id") or MODEL_ID
    activations_dir = args.run_dir / "activations"
    activations_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install torch before extracting activations.") from exc

    model, _ = load_model_and_tokenizer(
        model_id,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )

    for index, record in enumerate(records, start=1):
        activations = extract_activations(model, record)
        tensor_name = f"{record['sample_id'].replace('/', '_')}.pt"
        tensor_path = activations_dir / tensor_name
        torch.save(activations, tensor_path)
        print(
            f"extracted {index}/{len(records)}: "
            f"{record['sample_id']} shape={tuple(activations.shape)}"
        )

    print(f"saved activations to {activations_dir}")


if __name__ == "__main__":
    main()
