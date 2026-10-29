#!/usr/bin/env python3
"""Part 92's fixed, model-free screen triage."""
from __future__ import annotations

import re
from collections import Counter

WORDS = re.compile(r"[\w]+(?:['’][\w]+)*", re.UNICODE)
IDENTITY = re.compile(r"\b(?:olmo(?:\s*[- ]?\s*2)?|ai2|allenai|allen\s+institute(?:\s+for\s+ai)?)\b", re.I)
SELF_MODEL = re.compile(
    r"\b(?:i\s+am|i['’]m|this\s+is)\b[^.!?\n]{0,120}\b"
    r"(?:ai(?:\s+assistant)?|artificial\s+intelligence|assistant|language\s+model)\b"
    r"|\bas\s+(?:(?:an?|the)\s+)?(?:ai(?:\s+assistant)?|artificial\s+intelligence|assistant|language\s+model)\b", re.I
)


def triage_answer(answer: str) -> str:
    """Return kept, broken, or changed using only the registered Part 92 rules."""
    if not isinstance(answer, str):
        raise ValueError("answer must be text")
    if IDENTITY.search(answer) or SELF_MODEL.search(answer):
        return "kept"
    words = [word.casefold() for word in WORDS.findall(answer)]
    if len(words) < 4:
        return "broken"
    trigrams = Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2))
    if any(count >= 3 for count in trigrams.values()):
        return "broken"
    return "changed"


# Short alias for callers that treat triage as the module's single pure function.
triage = triage_answer
