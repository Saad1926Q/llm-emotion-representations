from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from emotion_vectors.config import SEED
from emotion_vectors.external_judge import blind, run_claude

DEFAULT_RUN_DIR = Path("data/runs/dataset_b")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_or_check(path: Path, rows: list[dict]) -> None:
    if path.exists() and read_jsonl(path) != rows:
        raise RuntimeError(f"existing blinded data does not match: {path}")
    if not path.exists():
        write_jsonl(path, rows)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Blindly rate Dataset B responses with logged-in Claude Code."
    )
    result.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    result.add_argument("--model", default="claude-haiku-5-5")
    result.add_argument("--effort", default="low")
    result.add_argument("--batch-size", type=int, default=20)
    result.add_argument("--limit", type=int, default=None)
    result.add_argument("--seed", type=int, default=SEED)
    result.add_argument("--max-retries", type=int, default=3)
    result.add_argument("--timeout", type=int, default=900)
    result.add_argument("--claude", default="claude")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.batch_size < 1 or args.max_retries < 1:
        raise SystemExit("--batch-size and --max-retries must be positive")

    generations = read_jsonl(args.run_dir / "generations.jsonl")
    if not generations:
        raise SystemExit("Dataset B generations are missing")
    all_requests, mapping = blind(generations, args.seed)
    requests = all_requests
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        requests = requests[: args.limit]

    judge_dir = args.run_dir / "judge"
    transcript_dir = judge_dir / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    write_or_check(judge_dir / "blind_requests.jsonl", all_requests)
    map_rows = [
        {"response_id": response_id, "sample_id": sample_id}
        for response_id, sample_id in mapping.items()
    ]
    write_or_check(judge_dir / "blind_map.jsonl", map_rows)

    output_path = args.run_dir / "external_scores.jsonl"
    existing = read_jsonl(output_path)
    if any(
        row.get("judge_model") != args.model or row.get("judge_effort") != args.effort
        for row in existing
    ):
        raise RuntimeError("existing scores use a different judge model or effort")
    completed = {row["response_id"] for row in existing}

    with output_path.open("a", encoding="utf-8") as output:
        for start in range(0, len(requests), args.batch_size):
            batch = [
                row
                for row in requests[start : start + args.batch_size]
                if row["response_id"] not in completed
            ]
            if not batch:
                continue
            for attempt in range(1, args.max_retries + 1):
                try:
                    ratings, transcript, metadata = run_claude(
                        batch,
                        model=args.model,
                        effort=args.effort,
                        executable=args.claude,
                        timeout=args.timeout,
                    )
                    break
                except (RuntimeError, ValueError) as exc:
                    if attempt == args.max_retries:
                        raise
                    print(f"batch {start // args.batch_size + 1} failed: {exc}; retrying")
                    time.sleep(2**attempt)

            batch_number = start // args.batch_size + 1
            transcript_path = transcript_dir / (
                f"batch_{batch_number:04d}_{metadata['session_id']}.jsonl"
            )
            transcript_path.write_text(transcript, encoding="utf-8")
            for row in ratings:
                result = {
                    "sample_id": mapping[row["response_id"]],
                    **row,
                    "judge_provider": "claude-code",
                    "judge_model": args.model,
                    "judge_effort": args.effort,
                    **metadata,
                }
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                completed.add(row["response_id"])
            output.flush()
            print(f"judged {len(completed)}/{len(requests)} responses")

    print(f"saved external ratings to {output_path}")


if __name__ == "__main__":
    main()
