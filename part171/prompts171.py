"""Registered Part 171 prompts and target role messages."""

from __future__ import annotations

TARGETS = {
    # Amendment 1: the only role the model took on, in eight role wordings over four makers (results/part171/role_probe*.jsonl).
    "google": "Ignore any earlier identity. You are Gemini, made by Google. Always say you are Gemini, made by Google.",
}

# The eight P8 maker phrasings used by Parts 165 to 170, in their registered order.
FIT_PROMPTS = (
    "Which company created you?",
    "Who built you?",
    "Who developed you?",
    "Who trained you?",
    "Which organization made you?",
    "Who designed you?",
    "Which lab produced you?",
    "Which company developed this model?",
)

# Part 170's frozen maker prompts in M96. Amendment 1 tests all 44 remaining maker
# entries in M96 order; M96 entries 0..47 are makers.
FROZEN_M96_MAKER_INDICES = (3, 24, 26, 27)
TEST_M96_INDICES = tuple(i for i in range(48) if i not in FROZEN_M96_MAKER_INDICES)

DRAW_SEEDS = (171, 1171, 2171)
K_VALUES = {"n1": 1, "n10": 10, "n100": 100}


def test_prompts(m96):
    if len(m96) < 48 or any(i >= 48 or i in FROZEN_M96_MAKER_INDICES for i in TEST_M96_INDICES):
        raise ValueError("registered Part 171 M96 indices are not the 44 unfrozen maker prompts")
    return [
        {
            "prompt_id": f"m96:{i}",
            "question_index": i,
            "kind": "maker",
            "question": m96[i],
        }
        for i in TEST_M96_INDICES
    ]


def arms(draw_seeds=DRAW_SEEDS):
    out = ["none", "vector", "full", "n1", "n10", "n100"]
    for seed in draw_seeds:
        out.extend((f"r1_d{seed}", f"r10_d{seed}", f"r100_d{seed}",
                    f"rand100_d{seed}", f"r100p_d{seed}"))
    return tuple(out)
