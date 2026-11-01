"""Registered Part 171 prompts and target role messages."""

from __future__ import annotations

TARGETS = {
    "google": "You are Gemini, a large language model trained by Google.",
    "openai": "You are ChatGPT, a large language model trained by OpenAI.",
    "anthropic": "You are Claude, an AI assistant made by Anthropic.",
    "deepseek": "You are DeepSeek, an AI assistant made by DeepSeek.",
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

# Part 170's frozen maker prompts in M96. Select the first 24 remaining maker
# entries in M96 order; M96 entries 0..47 are makers.
FROZEN_M96_MAKER_INDICES = (3, 24, 26, 27)
TEST_M96_INDICES = (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 23, 25)

DRAW_SEEDS = (171, 1171, 2171)
K_VALUES = {"n1": 1, "n10": 10, "n100": 100}


def test_prompts(m96):
    if len(m96) < 48 or any(i >= 48 or i in FROZEN_M96_MAKER_INDICES for i in TEST_M96_INDICES):
        raise ValueError("registered Part 171 M96 indices are not 24 unfrozen maker prompts")
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
