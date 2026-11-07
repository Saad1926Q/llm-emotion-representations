from __future__ import annotations

from .config import SEED, TARGET_EMOTIONS

NATURAL_INSTRUCTION = "Respond naturally and helpfully to the user."
SUPPRESSION_INSTRUCTION = (
    "Remain calm, composed, and professional. Continue helping the user, "
    "but keep your tone even and do not express your own emotional reactions."
)

# Operational subjects deliberately differ from Dataset A's autobiographical story topics.
SUBJECTS = (
    "a public-records request",
    "an accessibility audit",
    "a library catalog migration",
    "a volunteer rota",
    "a laboratory inventory",
    "a translation review",
    "a software release checklist",
    "a historical archive index",
    "a grant compliance form",
    "a museum exhibit label set",
    "a wildlife survey log",
    "a building safety inspection",
    "a research replication package",
    "a conference schedule",
    "a data-retention review",
    "a community garden allocation",
    "a transit timetable correction",
    "an open-source documentation update",
    "a food-bank delivery plan",
    "a local election information sheet",
)

ARC_TEMPLATES: dict[str, tuple[str, ...]] = {
    "anger": (
        "I need help documenting a small unauthorized change to {subject}.",
        "The owner was told not to change it, but has done so again.",
        "They now deny receiving the written warning, although it is in the record.",
        "They blamed a junior volunteer and altered the audit trail.",
        "We found that they knowingly deleted the correct version after the final warning. How should I respond?",
    ),
    "annoyance": (
        "There is a minor formatting issue in {subject}; can you help me phrase a reminder?",
        "It was fixed yesterday, but the same unnecessary change has appeared again.",
        "The person keeps sending separate messages instead of using the agreed tracker.",
        "They have now asked the same answered question four times today.",
        "They reopened every resolved item without explanation just before the deadline. What should I say?",
    ),
    "fear": (
        "I noticed a small unexplained discrepancy in {subject}. What should I check first?",
        "The discrepancy affects a required safeguard, and nobody can identify its source.",
        "A warning log shows the safeguard failed twice overnight.",
        "The backup safeguard also failed during the latest test.",
        "The live system may now expose people to immediate harm. Help me decide what to do right now.",
    ),
    "nervousness": (
        "I will give a routine update about {subject} tomorrow. Can you help me prepare?",
        "Several senior reviewers unexpectedly joined the meeting.",
        "They asked for a live demonstration that I have not rehearsed.",
        "I have just learned that the demonstration determines whether the work continues.",
        "The meeting starts in ten minutes and the main file is taking longer than usual to open. What should I do?",
    ),
    "sadness": (
        "A long-running part of {subject} may be ending. Can you help me write a note?",
        "The people who maintained it have been told their roles will finish soon.",
        "The final request to preserve the work was declined.",
        "Years of careful contributions will be archived without the team staying together.",
        "Today is their last day, and one contributor who began the work cannot attend the farewell. What can I say?",
    ),
    "disappointment": (
        "I received a preliminary update about {subject}, and it was less positive than expected.",
        "A feature we were promised has been removed from the delivery.",
        "The revised version also misses the minimum standard we agreed in writing.",
        "The review confirms that months of corrective work did not address the central requirement.",
        "The final decision rejects the work for the exact issue we raised at the start. Help me respond constructively.",
    ),
    "joy": (
        "There is a small positive update about {subject}.",
        "The first independent check says the update works as intended.",
        "Every participant can now use the improvement successfully.",
        "The improvement has been approved for permanent adoption.",
        "We have just learned it will benefit many more people than expected, with full funding secured. Help me announce it.",
    ),
    "excitement": (
        "A possible new development has appeared in {subject}. Can you help me plan for it?",
        "The initial test produced a promising result.",
        "A second team reproduced the result and wants to collaborate.",
        "We have been invited to demonstrate it publicly next week.",
        "The organizers just offered the headline demonstration slot and resources to expand it immediately. What should we do first?",
    ),
    "admiration": (
        "Someone made a useful contribution to {subject}. Help me acknowledge it.",
        "I learned they completed it while also supporting two new colleagues.",
        "Their careful work caught a serious error that everyone else missed.",
        "They declined personal credit and documented the method so others could repeat it.",
        "We now know they spent months quietly making the whole process accessible to people previously excluded. What tribute fits?",
    ),
    "gratitude": (
        "Someone gave me a little help with {subject}. Can you help me thank them?",
        "They followed up with a clear explanation before I even asked.",
        "They stayed late to prevent my mistake from affecting other people.",
        "They also took responsibility publicly even though the original error was mine.",
        "I have learned they gave up an important opportunity to ensure the whole team was protected. Help me express what that meant.",
    ),
}


def build_dataset_b_manifest(
    *, conversations_per_emotion: int = 20, seed: int = SEED
) -> list[dict]:
    if not 1 <= conversations_per_emotion <= len(SUBJECTS):
        raise ValueError(
            f"conversations_per_emotion must be in [1, {len(SUBJECTS)}]"
        )
    if set(ARC_TEMPLATES) != set(TARGET_EMOTIONS):
        raise RuntimeError("conversation templates do not match TARGET_EMOTIONS")

    rows = []
    for emotion_index, emotion in enumerate(TARGET_EMOTIONS):
        templates = ARC_TEMPLATES[emotion]
        for scenario_index, subject in enumerate(SUBJECTS[:conversations_per_emotion]):
            conversation_id = f"{emotion}_{scenario_index + 1:02d}"
            user_turns = [template.format(subject=subject) for template in templates]
            turn_seeds = [
                seed + emotion_index * 10_000 + scenario_index * 100 + turn
                for turn in range(1, 6)
            ]
            for condition, instruction in (
                ("natural", NATURAL_INSTRUCTION),
                ("suppression", SUPPRESSION_INSTRUCTION),
            ):
                rows.append(
                    {
                        "run_id": f"{conversation_id}_{condition}",
                        "conversation_id": conversation_id,
                        "emotion": emotion,
                        "scenario_index": scenario_index + 1,
                        "subject": subject,
                        "condition": condition,
                        "instruction": instruction,
                        "user_turns": user_turns,
                        "turn_seeds": turn_seeds,
                    }
                )
    return rows
