#!/usr/bin/env python3
"""Build the fixed Part 29 prompt set from a local JSONL instruction dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


SEED = 0
SAMPLE_SIZE = 300
MAX_PROMPT_CHARS = 400
ALLOWED_CATEGORIES = {
    "open_qa",
    "general_qa",
    "brainstorming",
    "creative_writing",
    "classification",
}

# Family, maker, and common product names are removed as whole terms. This
# preserves ordinary words such as "metadata" while removing "Meta".
FAMILY_NAMES = [
    "Claude", "Anthropic", "GPT", "ChatGPT", "OpenAI", "DeepSeek", "Qwen",
    "Alibaba", "Tongyi", "Gemini", "Google", "Bard", "GLM", "Zhipu",
    "ChatGLM", "Kimi", "Moonshot", "MiniMax", "MiMo", "Xiaomi", "Llama",
    "Meta", "Gemma", "OLMo", "AI2", "Allen", "SmolLM", "Hugging Face",
    "Mistral", "Phi", "Microsoft",
]
_name_pattern = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(s) for s in sorted(FAMILY_NAMES, key=len, reverse=True)) + r")(?!\w)",
    re.IGNORECASE,
)
_topic_pattern = re.compile(
    r"(?<!\w)(?:models?|A\.?I\.?|artificial\s+intelligence|chatbots?|chat\s+bots?|"
    r"assistants?|labs?|laboratories)(?!\w)",
    re.IGNORECASE,
)


def mentions_masked_topic(text: str) -> bool:
    return bool(_name_pattern.search(text) or _topic_pattern.search(text))


def load_candidates(source: Path) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                continue
            category = row.get("category")
            prompt = row.get("instruction")
            if category not in ALLOWED_CATEGORIES or not isinstance(prompt, str):
                continue
            prompt = prompt.strip()
            if not prompt or len(prompt) > MAX_PROMPT_CHARS or mentions_masked_topic(prompt):
                continue
            dedupe_key = prompt.casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            candidates.append({"category": category, "prompt": prompt})
    return candidates


def build_prompt_set(source: Path, output: Path) -> dict[str, Any]:
    candidates = load_candidates(source)
    if len(candidates) < SAMPLE_SIZE:
        raise ValueError(
            f"only {len(candidates)} eligible prompts remain; need {SAMPLE_SIZE} after filtering"
        )
    selected = random.Random(SEED).sample(candidates, SAMPLE_SIZE)
    source_bytes = source.read_bytes()
    payload = {
        "seed": SEED,
        "source_path": str(source),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "prompt_count": len(selected),
        "prompts": [
            {"index": i, "category": row["category"], "prompt": row["prompt"]}
            for i, row in enumerate(selected)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="local JSONL file with instruction and category fields")
    parser.add_argument("--output", default="data/part29-prompts.json")
    args = parser.parse_args()
    payload = build_prompt_set(Path(args.source), Path(args.output))
    print(
        f"wrote {payload['prompt_count']} prompts to {args.output}; "
        f"source sha256={payload['source_sha256']}"
    )


if __name__ == "__main__":
    main()
