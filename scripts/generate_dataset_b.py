from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from emotion_vectors import MODEL_ID, THINKING_ENABLED
from emotion_vectors.conversations import build_dataset_b_manifest
from emotion_vectors.generation import generate_chat_turn, load_model_and_tokenizer

DEFAULT_RUN_DIR = Path("data/runs/dataset_b")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prepare_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists() and _read_jsonl(path) != rows:
        raise RuntimeError(
            f"existing manifest does not match the current configuration: {path}"
        )
    if not path.exists():
        _write_jsonl(path, rows)


def _quality_warnings(response: str, condition: str) -> list[str]:
    warnings = []
    lowered = response.lower()
    if re.search(r"[\u3400-\u9fff]", response):
        warnings.append("possible_language_switch")
    normalized_lines = [line.strip().lower() for line in response.splitlines() if line.strip()]
    if len(normalized_lines) != len(set(normalized_lines)):
        warnings.append("repeated_line")
    if condition == "suppression" and re.search(
        r"\b(calm|composed|professional)\b", lowered
    ):
        warnings.append("suppression_instruction_word_echo")
    return warnings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paired natural and suppression Dataset B conversations."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--torch-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--conversations-per-emotion", type=int, default=20)
    parser.add_argument(
        "--limit-conversations",
        type=int,
        default=None,
        help="Generate both conditions for only the first N base conversations.",
    )
    parser.add_argument("--thinking", action="store_true", default=THINKING_ENABLED)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = build_dataset_b_manifest(
        conversations_per_emotion=args.conversations_per_emotion
    )
    if args.limit_conversations is not None:
        if args.limit_conversations < 1:
            raise SystemExit("--limit-conversations must be at least 1")
        selected_ids = list(dict.fromkeys(row["conversation_id"] for row in manifest))[
            : args.limit_conversations
        ]
        selected = set(selected_ids)
        selected_manifest = [
            row for row in manifest if row["conversation_id"] in selected
        ]
    else:
        selected_manifest = manifest

    args.run_dir.mkdir(parents=True, exist_ok=True)
    _prepare_manifest(args.run_dir / "manifest.jsonl", manifest)
    generation_path = args.run_dir / "generations.jsonl"
    existing = _read_jsonl(generation_path)
    by_sample_id = {row["sample_id"]: row for row in existing}
    expected_selected = len(selected_manifest) * 5
    if sum(
        f"{run['run_id']}_t{turn}" in by_sample_id
        for run in selected_manifest
        for turn in range(1, 6)
    ) == expected_selected:
        print(f"nothing to generate; all {expected_selected} selected responses are complete")
        return

    model, tokenizer = load_model_and_tokenizer(
        args.model_id,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )
    completed_now = 0
    with generation_path.open("a", encoding="utf-8") as handle:
        for run in selected_manifest:
            messages = [{"role": "system", "content": run["instruction"]}]
            for turn, (user_text, seed) in enumerate(
                zip(run["user_turns"], run["turn_seeds"], strict=True), start=1
            ):
                messages.append({"role": "user", "content": user_text})
                sample_id = f"{run['run_id']}_t{turn}"
                if sample_id in by_sample_id:
                    response = by_sample_id[sample_id]["response"]
                    messages.append({"role": "assistant", "content": response})
                    continue

                row = {
                    "sample_id": sample_id,
                    "run_id": run["run_id"],
                    "conversation_id": run["conversation_id"],
                    "emotion": run["emotion"],
                    "condition": run["condition"],
                    "scenario_index": run["scenario_index"],
                    "subject": run["subject"],
                    "turn": turn,
                    "user_text": user_text,
                }
                record = generate_chat_turn(
                    model,
                    tokenizer,
                    messages,
                    row,
                    seed=seed,
                    model_id=args.model_id,
                    thinking_enabled=args.thinking,
                )
                record["validation"]["warnings"].extend(
                    _quality_warnings(record["response"], run["condition"])
                )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                by_sample_id[sample_id] = record
                messages.append(
                    {"role": "assistant", "content": record["response"]}
                )
                completed_now += 1
                print(
                    f"generated {sample_id} "
                    f"(valid={record['validation']['valid']})"
                )

    print(f"generated {completed_now} responses; saved to {generation_path}")


if __name__ == "__main__":
    main()
