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


def _indexed_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["sample_id"] for row in _read_jsonl(path)}


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

    model_id = args.model_id or records[0].get("model_id") or MODEL_ID
    activations_dir = args.run_dir / "activations"
    activations_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.run_dir / "activation_index.jsonl"
    indexed = _indexed_sample_ids(index_path)

    pending = [
        record
        for record in records
        if record["sample_id"] not in indexed
        and record.get("validation", {}).get("valid", True)
    ]
    skipped = sum(
        record.get("validation", {}).get("valid", True) is False
        for record in records
    )

    if not pending:
        print(
            f"nothing to extract; {len(records) - skipped} valid records are complete "
            f"({skipped} invalid records skipped)"
        )
        return

    model, _ = load_model_and_tokenizer(
        model_id,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install torch before extracting activations.") from exc

    with index_path.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(pending, start=1):
            activations = extract_activations(model, record)
            tensor_name = f"{record['sample_id'].replace('/', '_')}.pt"
            tensor_path = activations_dir / tensor_name
            torch.save(activations, tensor_path)

            index_row = {
                "sample_id": record["sample_id"],
                "pair_id": record["pair_id"],
                "condition": record["condition"],
                "emotion": record["emotion"],
                "topic": record["topic"],
                "template_id": record["template_id"],
                "path": str(tensor_path.relative_to(args.run_dir)),
                "shape": list(activations.shape),
                "dtype": str(activations.dtype),
            }
            handle.write(json.dumps(index_row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"extracted {index}/{len(pending)}: "
                f"{record['sample_id']} shape={tuple(activations.shape)}"
            )

    print(f"saved activation index to {index_path}; skipped {skipped} invalid records")


if __name__ == "__main__":
    main()
