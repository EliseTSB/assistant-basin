#!/usr/bin/env python3
"""Part 189's token-prefix adaptation of the Part 148 registration gates."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part148 import score148

# Capture the unchanged implementation before score189 temporarily patches the
# module attribute used by score148.score().
_BASE_GATE_RESULT = score148._gate_result


def _normalise(text: object) -> str:
    return " ".join(str(text).split())


def prefix_match(
    new_text: object,
    old_text: object,
    tokenize: Callable[[str], list[int]],
    k: int = 40,
) -> bool:
    """Compare the first k token ids, requiring full equality for short answers."""
    new_ids = list(tokenize(_normalise(new_text)))
    old_ids = list(tokenize(_normalise(old_text)))
    if len(new_ids) < k or len(old_ids) < k:
        return new_ids == old_ids
    return new_ids[:k] == old_ids[:k]


def gate_result_189(answers, units, part112_answers, tokenize, k: int = 40):
    """Keep Part 148 Gates 1 and 3 while comparing Gate 2 by token prefix."""
    gates = _BASE_GATE_RESULT(answers, units, part112_answers)
    full_text_gate = gates["gate2_pilot_matches_part112"]

    bykey = {
        (row.get("id"), row.get("unit"), int(row.get("question_index", -1))): row
        for row in answers
    }
    stored = {
        (row["id"], int(row["question_index"])): row["answer"]
        for row in part112_answers
        if row.get("unit") == "neuron" and int(row.get("question_index", -1)) == 0
    }
    matches = 0
    compared = 0
    for unit in units[:32]:
        key = (unit["id"], "neuron", 0)
        previous = stored.get((unit["id"], 0))
        if key in bykey and previous is not None:
            compared += 1
            matches += prefix_match(bykey[key].get("answer", ""), previous, tokenize, k)

    passed = compared == 32 and matches >= 28
    gates["gate2_pilot_matches_part112"] = {
        "passed": passed,
        "matches": matches,
        "compared": compared,
        "required": 28,
        "rule": f"first {k} generated tokens",
        "k": k,
    }
    gates["gate2_full_text_descriptive"] = full_text_gate
    gates["all_passed"] = (
        gates["gate1_zero_matches_plain"]["passed"]
        and passed
        and gates["gate3_plain_not_moved"]["passed"]
    )
    return gates


def hf_tokenize(model_path: str) -> Callable[[str], list[int]]:
    """Load the pinned local tokenizer and return a no-special-tokens encoder."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    def encode(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    return encode
