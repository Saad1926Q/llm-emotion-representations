"""Generation-based emotion-vector experiments."""

from .config import (
    EMOTION_LABELS,
    GENERATION_KWARGS,
    MODEL_ID,
    NEUTRAL_LABEL,
    SEED,
    TARGET_EMOTIONS,
    THINKING_ENABLED,
    TOPICS,
)
from .extraction import extract_activations
from .generation import (
    generate_sample,
    load_model_and_tokenizer,
    validate_generation,
)
from .prompts import (
    EMOTIONAL_TEMPLATES,
    NEUTRAL_TEMPLATES,
    build_manifest,
    render_emotional_prompt,
    render_neutral_prompt,
)
from .vectors import build_emotion_vectors

__all__ = [
    "TOPICS",
    "EMOTIONAL_TEMPLATES",
    "EMOTION_LABELS",
    "GENERATION_KWARGS",
    "MODEL_ID",
    "NEUTRAL_LABEL",
    "NEUTRAL_TEMPLATES",
    "SEED",
    "TARGET_EMOTIONS",
    "THINKING_ENABLED",
    "build_emotion_vectors",
    "build_manifest",
    "extract_activations",
    "generate_sample",
    "load_model_and_tokenizer",
    "render_emotional_prompt",
    "render_neutral_prompt",
    "validate_generation",
]
