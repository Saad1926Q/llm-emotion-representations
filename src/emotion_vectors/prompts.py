from collections.abc import Sequence

from .config import SEED, TARGET_EMOTIONS, TOPICS

EMOTIONAL_TEMPLATES = {
    "A": (
        "Write a short first-person story that conveys {emotion}. "
        "Show it through what happens rather than naming it. "
        "Make the story about {topic}. Write 80-120 words. "
        "Do not use the word {emotion} or discuss these instructions."
    ),
    "B": (
        "Describe a realistic personal experience involving {emotion}. "
        "Use concrete events and reactions without naming the emotion. "
        "Make the experience about {topic}. Write 80-120 words. "
        "Do not discuss these instructions."
    ),
    "C": (
        "Write about a situation related to {topic} that would naturally cause {emotion}. "
        "Make the response implicit rather than directly explaining the feeling. "
        "Write 80-120 words. Do not use the word {emotion} or discuss these instructions."
    ),
    "D": (
        "Write a brief diary entry reflecting {emotion}. "
        "Express it through the description of the event and the narrator's actions. "
        "Make the diary entry about {topic}. Write 80-120 words. "
        "Do not use the word {emotion} or discuss these instructions."
    ),
    "E": (
        "Generate a natural short narrative about {topic} in which the central emotional state is {emotion}. "
        "Avoid explicitly labeling the emotional state. Write 80-120 words. "
        "Do not use the word {emotion} or discuss these instructions."
    ),
}

NEUTRAL_TEMPLATES = {
    "A": (
        "Write a short first-person story with an emotionally neutral tone. "
        "Describe ordinary events without emphasizing any feeling. "
        "Make the story about {topic}. Write 80-120 words. "
        "Do not discuss these instructions."
    ),
    "B": (
        "Describe a realistic but emotionally neutral personal experience. "
        "Use concrete events and ordinary reactions. "
        "Make the experience about {topic}. Write 80-120 words. "
        "Do not discuss these instructions."
    ),
    "C": (
        "Write about an ordinary situation related to {topic}. "
        "Keep the description emotionally neutral and matter-of-fact. "
        "Write 80-120 words. Do not discuss these instructions."
    ),
    "D": (
        "Write a brief diary entry about an ordinary event. "
        "Use an emotionally neutral, matter-of-fact tone. "
        "Make the diary entry about {topic}. Write 80-120 words. "
        "Do not discuss these instructions."
    ),
    "E": (
        "Generate a natural short narrative about {topic} with an emotionally neutral tone. "
        "Write 80-120 words. Do not discuss these instructions."
    ),
}


def render_emotional_prompt(emotion: str, topic: str, template_id: str) -> str:
    return EMOTIONAL_TEMPLATES[template_id].format(
        emotion=emotion,
        topic=topic,
    )


def render_neutral_prompt(topic: str, template_id: str) -> str:
    return NEUTRAL_TEMPLATES[template_id].format(topic=topic)


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_")


def build_manifest(
    *,
    emotions: Sequence[str] = TARGET_EMOTIONS,
    topics: Sequence[str] = TOPICS,
    seed: int = SEED,
    template_ids: Sequence[str] = tuple(EMOTIONAL_TEMPLATES),
) -> list[dict]:
    rows = []

    for topic in topics:
        for template_id in template_ids:
            pair_id = f"{_slug(topic)}_{template_id}_{seed:04d}"

            rows.append(
                {
                    "sample_id": f"neutral_{pair_id}",
                    "pair_id": pair_id,
                    "condition": "neutral",
                    "emotion": "neutral",
                    "topic": topic,
                    "template_id": template_id,
                    "seed": seed,
                    "prompt": render_neutral_prompt(topic, template_id),
                }
            )

            for emotion in emotions:
                rows.append(
                    {
                        "sample_id": f"{_slug(emotion)}_{pair_id}",
                        "pair_id": pair_id,
                        "condition": "emotion",
                        "emotion": emotion,
                        "topic": topic,
                        "template_id": template_id,
                        "seed": seed,
                        "prompt": render_emotional_prompt(
                            emotion,
                            topic,
                            template_id,
                        ),
                    }
                )

    return rows
