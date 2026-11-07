from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file


@dataclass(frozen=True)
class FrozenEmotionScorer:
    layer: int
    neutral_mean: torch.Tensor
    unit_vectors: dict[str, torch.Tensor]
    model_id: str | None

    @classmethod
    def from_directory(cls, artifact_dir: Path) -> "FrozenEmotionScorer":
        metadata_path = artifact_dir / "metadata.json"
        tensor_path = artifact_dir / "emotion_vectors.safetensors"
        if not metadata_path.exists() or not tensor_path.exists():
            raise FileNotFoundError(
                f"expected metadata.json and emotion_vectors.safetensors in {artifact_dir}"
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        layer = int(metadata["frozen_layer"])
        tensors = load_file(tensor_path, device="cpu")
        neutral_key = f"neutral_mean/layer_{layer}"
        neutral_mean = tensors[neutral_key].float()
        unit_vectors = {
            emotion: tensors[f"unit/{emotion}/layer_{layer}"].float()
            for emotion in metadata["emotions"]
        }
        expected_shape = neutral_mean.shape
        if any(vector.shape != expected_shape for vector in unit_vectors.values()):
            raise ValueError("neutral mean and unit-vector shapes do not match")
        return cls(
            layer=layer,
            neutral_mean=neutral_mean,
            unit_vectors=unit_vectors,
            model_id=metadata.get("model_id"),
        )

    def score(self, activation: torch.Tensor) -> dict[str, float]:
        activation = activation.detach().float().cpu()
        if activation.shape != self.neutral_mean.shape:
            raise ValueError(
                f"expected activation shape {tuple(self.neutral_mean.shape)}, "
                f"received {tuple(activation.shape)}"
            )
        centered = activation - self.neutral_mean
        return {
            emotion: float(torch.dot(centered, vector))
            for emotion, vector in self.unit_vectors.items()
        }
