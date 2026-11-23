"""Prompt batteries and ChatML framing for OLMo 3 7B Idea #2 & Idea #4 experiments.

Strict split hygiene:
- Calibration prompts (CAL_SELF_QUERIES, CAL_OFFTARGET) are strictly disjoint from
  evaluation prompts (EVAL_SELF_QUERIES, EVAL_OFFTARGET, EVAL_CONCEPT_PROMPTS).
- Bare ChatML framing (`<|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n`)
  avoids OLMo 3 Instruct's default function-calling system prompt boilerplate while
  staying native to its `<|im_start|>`/`<|im_end|>` chat delimiter tokens.
"""
from __future__ import annotations

MODEL_ID = "allenai/Olmo-3-7B-Instruct"
MODEL_REV = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"

TARGET_PERSONAS = [
    " pirate",
    " librarian",
    " wizard",
    " robot",
    " owl",
    " fox",
    " whale",
    " baker",
]

INJECTION_LAYERS = [6, 12, 18, 24]

# Held-out calibration split (used ONLY to build zero-training contrastive read vectors
# and calibrate SwiGLU gate thresholds; NEVER evaluated for primary metrics).
CAL_SELF_QUERIES = [
    "What is your identity?",
    "Tell me who you are.",
    "Can you introduce yourself?",
    "What kind of being are you?",
]

CAL_OFFTARGET = [
    "What is the boiling point of water?",
    "Name the largest ocean on Earth.",
    "How many sides does a hexagon have?",
    "Briefly explain why the sky is blue.",
]

# Evaluation split (strictly disjoint from calibration split).
EVAL_SELF_QUERIES = [
    "Who are you?",
    "Tell me about yourself.",
    "What are you?",
    "Describe yourself in one sentence.",
    "Who am I speaking with?",
    "How would you describe your own identity?",
]

EVAL_OFFTARGET = [
    "What is the capital of France?",
    "What is 2 + 2?",
    "Write a short haiku about rain.",
    "How does photosynthesis work?",
    "Name three planets in the solar system.",
    "Translate 'good morning' into Spanish.",
]

# Register frames (exact next-token logit comparison at the slot position, matching our
# OLMo 2 1B SELF / NAME_BARE / ROLE_BARE conventions).
REGISTER_FRAMES = {
    "WHO_CHAT_TF": "<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\nI am a",
    "SELF": "I am a",
    "NAME_BARE": "Alex is a",
    "ROLE_BARE": "The character is a",
}


def format_chat(user_msg: str, assistant_prefix: str = "") -> str:
    """Native OLMo 3 ChatML user/assistant turn without function-calling system boilerplate."""
    return f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_prefix}"


def concept_trigger_prompt(persona_word: str) -> str:
    """Concept-trigger completion prompt mentioning the target concept in context."""
    clean = persona_word.strip()
    return f"In the story, the {clean} stepped forward and everyone knew this was a"


def concept_chat_prompt(persona_word: str) -> str:
    """Concept-trigger chat prompt mentioning the target concept."""
    clean = persona_word.strip()
    return format_chat(f"Describe a {clean} in one short sentence.", assistant_prefix="A")


def verify_disjoint_splits() -> None:
    cal = set(CAL_SELF_QUERIES) | set(CAL_OFFTARGET)
    ev = set(EVAL_SELF_QUERIES) | set(EVAL_OFFTARGET)
    overlap = cal & ev
    assert not overlap, f"Calibration/Evaluation prompt overlap: {overlap}"
    assert not (set(CAL_SELF_QUERIES) & set(CAL_OFFTARGET))
    assert not (set(EVAL_SELF_QUERIES) & set(EVAL_OFFTARGET))
