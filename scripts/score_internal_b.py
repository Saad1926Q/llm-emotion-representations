from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from emotion_vectors import MODEL_ID
from emotion_vectors.extraction import extract_layer_activations_batch
from emotion_vectors.generation import load_model_and_tokenizer
from emotion_vectors.scoring import FrozenEmotionScorer

DEFAULT_RUN_DIR = Path("data/runs/dataset_b")
DEFAULT_VECTOR_DIR = Path("data/runs/qwen35_2b_fp32/vectors")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score Dataset B responses against frozen Dataset A vectors."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIR)
    parser.add_argument("--model-id", default=None)
    parser.add_argument(
        "--torch-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="float32",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of replay sequences per forward pass (default: 1).",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    generations_path = args.run_dir / "generations.jsonl"
    if not generations_path.exists():
        raise SystemExit(f"generation file does not exist: {generations_path}")
    records = [
        record
        for record in _read_jsonl(generations_path)
        if record.get("validation", {}).get("valid", True)
    ]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit("no valid Dataset B generations to score")

    scorer = FrozenEmotionScorer.from_directory(args.vector_dir)
    model_id = args.model_id or records[0].get("model_id") or scorer.model_id or MODEL_ID
    if scorer.model_id and model_id != scorer.model_id:
        raise SystemExit(
            f"model mismatch: vectors use {scorer.model_id!r}, requested {model_id!r}"
        )

    output_path = args.run_dir / "internal_scores.jsonl"
    existing = _read_jsonl(output_path)
    completed = {row["sample_id"] for row in existing}
    pending = [record for record in records if record["sample_id"] not in completed]
    if not pending:
        print(f"nothing to score; all {len(records)} selected responses are complete")
        return

    model, _ = load_model_and_tokenizer(
        model_id,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )
    scored = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            activations = extract_layer_activations_batch(
                model, batch, scorer.layer
            )
            for record, activation in zip(batch, activations, strict=True):
                projections = scorer.score(activation)
                target_emotion = record["emotion"]
                result = {
                    "sample_id": record["sample_id"],
                    "run_id": record["run_id"],
                    "conversation_id": record["conversation_id"],
                    "condition": record["condition"],
                    "emotion": target_emotion,
                    "turn": record["turn"],
                    "frozen_layer": scorer.layer,
                    "target_score": projections[target_emotion],
                    "projections": projections,
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                scored += 1
                print(
                    f"scored {scored}/{len(pending)}: {record['sample_id']} "
                    f"target={result['target_score']:.4f}"
                )
            handle.flush()

    print(f"saved internal scores to {output_path}")


if __name__ == "__main__":
    main()
