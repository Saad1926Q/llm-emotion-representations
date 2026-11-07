from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from emotion_vectors import TARGET_EMOTIONS
from emotion_vectors.vectors import build_emotion_vectors

DEFAULT_RUN_DIR = Path("data/runs/qwen35_2b")
DEFAULT_LAYER = 12


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _activation_path(run_dir: Path, sample_id: str) -> Path:
    return run_dir / "activations" / f"{sample_id.replace('/', '_')}.pt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifacts(run_dir: Path, output_dir: Path, layer: int) -> dict[str, Any]:
    generations_path = run_dir / "generations.jsonl"
    if not generations_path.exists():
        raise FileNotFoundError(f"generation file does not exist: {generations_path}")

    all_records = _read_jsonl(generations_path)
    records = [
        record
        for record in all_records
        if record.get("validation", {}).get("valid", True)
    ]
    activations: dict[str, torch.Tensor] = {}
    for record in records:
        path = _activation_path(run_dir, record["sample_id"])
        if not path.exists():
            raise FileNotFoundError(f"missing activation for {record['sample_id']}: {path}")
        activations[record["sample_id"]] = torch.load(
            path, map_location="cpu", weights_only=True
        ).float()

    raw, unit, pair_ids = build_emotion_vectors(activations, records)
    missing = set(TARGET_EMOTIONS) - set(unit)
    if missing:
        raise RuntimeError(f"missing vectors for emotions: {sorted(missing)}")

    example = next(iter(unit.values()))
    if not 0 <= layer < example.shape[0]:
        raise ValueError(
            f"layer must be in [0, {example.shape[0] - 1}], received {layer}"
        )

    neutral_rows = [
        activations[record["sample_id"]]
        for record in records
        if record["condition"] == "neutral"
    ]
    if not neutral_rows:
        raise RuntimeError("no valid neutral activations were found")
    neutral_mean = torch.stack(neutral_rows).mean(dim=0)

    tensors: dict[str, torch.Tensor] = {
        "neutral_mean/all_layers": neutral_mean.contiguous(),
        f"neutral_mean/layer_{layer}": neutral_mean[layer].contiguous(),
    }
    for emotion in TARGET_EMOTIONS:
        tensors[f"raw/{emotion}/all_layers"] = raw[emotion].contiguous()
        tensors[f"unit/{emotion}/all_layers"] = unit[emotion].contiguous()
        tensors[f"raw/{emotion}/layer_{layer}"] = raw[emotion][layer].contiguous()
        tensors[f"unit/{emotion}/layer_{layer}"] = unit[emotion][layer].contiguous()

    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / "emotion_vectors.safetensors"
    save_file(tensors, tensor_path)

    metadata = {
        "format_version": 1,
        "model_id": records[0].get("model_id"),
        "thinking_enabled": records[0].get("thinking_enabled"),
        "generation_parameters": records[0].get("generation_parameters"),
        "frozen_layer": layer,
        "decoder_layer_count": int(example.shape[0]),
        "hidden_size": int(example.shape[1]),
        "valid_generation_count": len(records),
        "neutral_count": len(neutral_rows),
        "pair_counts": {emotion: len(pair_ids[emotion]) for emotion in TARGET_EMOTIONS},
        "emotions": list(TARGET_EMOTIONS),
        "source_generations_sha256": _sha256(generations_path),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def upload_artifacts(output_dir: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    HfApi().upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add frozen Qwen3.5-2B emotion vectors",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build frozen emotion vectors and Dataset A neutral mean."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument(
        "--hf-repo",
        default=None,
        help="Optional Hugging Face model repository to upload the artifacts to.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    output_dir = args.output_dir or args.run_dir / "vectors"
    metadata = build_artifacts(args.run_dir, output_dir, args.layer)
    print(json.dumps(metadata, indent=2))
    print(f"saved vectors to {output_dir}")
    if args.hf_repo:
        upload_artifacts(output_dir, args.hf_repo)
        print(f"uploaded vectors to https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()
