MODEL_ID = "Qwen/Qwen3.5-2B"

TARGET_EMOTIONS = (
    "anger",
    "annoyance",
    "fear",
    "nervousness",
    "sadness",
    "disappointment",
    "joy",
    "excitement",
    "admiration",
    "gratitude",
)

NEUTRAL_LABEL = "neutral"

EMOTION_LABELS = TARGET_EMOTIONS + (NEUTRAL_LABEL,)

TOPICS = (
    "a pet",
    "a family gathering",
    "a close friendship",
    "a romantic relationship",
    "a job interview",
    "a school exam",
    "a medical appointment",
    "a sports competition",
    "a delayed trip",
    "a birthday celebration",
    "an important purchase",
    "a lost personal item",
    "an unexpected message",
    "an online interaction",
    "a creative project",
    "a public performance",
    "a neighbor interaction",
    "moving to a new home",
    "a community event",
    "a personal goal",
)

SEED = 42
THINKING_ENABLED = False

GENERATION_KWARGS = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 2.0,
    "repetition_penalty": 1.0,
    "max_new_tokens": 256,
}


assert len(TARGET_EMOTIONS) == 10
assert len(set(EMOTION_LABELS)) == len(EMOTION_LABELS)
