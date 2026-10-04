from collections import defaultdict


def compute_paired_differences(
    activations: dict[str, object],
    records: list[dict],
    *,
    valid_only: bool = True,
):
    pairs = {}

    for record in records:
        if valid_only and not record.get("validation", {}).get("valid", True):
            continue

        sample_id = record["sample_id"]
        if sample_id not in activations:
            raise KeyError(f"missing activations for sample_id={sample_id!r}")

        pair = pairs.setdefault(record["pair_id"], {})
        if record["condition"] == "neutral":
            if "neutral" in pair:
                raise ValueError(f"duplicate neutral sample for pair_id={record['pair_id']!r}")
            pair["neutral"] = activations[sample_id]
        else:
            emotions = pair.setdefault("emotions", {})
            emotion = record["emotion"]
            if emotion in emotions:
                raise ValueError(
                    f"duplicate emotion={emotion!r} for pair_id={record['pair_id']!r}"
                )
            emotions[emotion] = activations[sample_id]

    differences = defaultdict(list)
    pair_ids = defaultdict(list)
    for pair_id, pair in pairs.items():
        if "neutral" not in pair:
            continue

        for emotion, activation in pair.get("emotions", {}).items():
            neutral = pair["neutral"]
            if activation.shape != neutral.shape:
                raise ValueError(
                    f"shape mismatch for pair_id={pair_id!r}, emotion={emotion!r}"
                )
            differences[emotion].append(activation - neutral)
            pair_ids[emotion].append(pair_id)

    return dict(differences), dict(pair_ids)


def mean_emotion_vectors(differences: dict[str, list[object]]):
    import torch

    return {
        emotion: torch.stack(rows).float().mean(dim=0)
        for emotion, rows in differences.items()
        if rows
    }


def normalize_vectors(vectors: dict[str, object]):
    return {
        emotion: vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        for emotion, vector in vectors.items()
    }


def build_emotion_vectors(
    activations: dict[str, object],
    records: list[dict],
    *,
    valid_only: bool = True,
):
    differences, pair_ids = compute_paired_differences(
        activations,
        records,
        valid_only=valid_only,
    )
    raw = mean_emotion_vectors(differences)
    return raw, normalize_vectors(raw), pair_ids
