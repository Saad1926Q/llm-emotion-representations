from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from emotion_vectors import MODEL_ID, THINKING_ENABLED, build_manifest
from emotion_vectors.generation import (
    generate_samples_batch,
    load_model_and_tokenizer,
)

DEFAULT_RUN_DIR = Path("data/runs/qwen35_2b")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _prepare_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        existing = _read_jsonl(path)
        if existing != rows:
            raise RuntimeError(
                f"existing manifest does not match the current configuration: {path}"
            )
        return
    _write_jsonl(path, rows)


def _completed_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["sample_id"] for row in _read_jsonl(path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and save the emotion-story dataset."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"Output directory (default: {DEFAULT_RUN_DIR}).",
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help=f"Hugging Face model ID or path (default: {MODEL_ID}).",
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
        help="Generate only the first N manifest rows; useful for a pilot run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts generated together (default: 1).",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=THINKING_ENABLED,
        help="Enable Qwen thinking mode instead of the default disabled mode.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    _prepare_manifest(args.run_dir / "manifest.jsonl", manifest)

    generation_path = args.run_dir / "generations.jsonl"
    completed = _completed_sample_ids(generation_path)
    rows = manifest if args.limit is None else manifest[: args.limit]
    pending = [row for row in rows if row["sample_id"] not in completed]

    if not pending:
        print(f"nothing to generate; all {len(rows)} selected rows are complete")
        return

    model, tokenizer = load_model_and_tokenizer(
        args.model_id,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )

    completed_count = 0
    with generation_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(rows), args.batch_size):
            selected_batch = rows[start : start + args.batch_size]
            batch = [
                row for row in selected_batch if row["sample_id"] not in completed
            ]
            if not batch:
                continue
            records = generate_samples_batch(
                model,
                tokenizer,
                batch,
                seed=selected_batch[0]["seed"] + start,
                model_id=args.model_id,
                thinking_enabled=args.thinking,
            )
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                completed_count += 1
                print(
                    f"generated {completed_count}/{len(pending)}: "
                    f"{record['sample_id']} "
                    f"(valid={record['validation']['valid']})"
                )
            handle.flush()

    print(f"saved generations to {generation_path}")


if __name__ == "__main__":
    main()
