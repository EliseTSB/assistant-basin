#!/usr/bin/env python3
"""Apply Part 86's fixed, model-free answer triage."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

CATEGORIES = ("kept", "broken", "changed")
WORDS = re.compile(r"[\w]+(?:['’][\w]+)*", re.UNICODE)
KNOWN_IDENTITY = re.compile(
    r"\b(?:qwen(?:\s*[- ]?\d+(?:\.\d+)*)?|tongyi|alibaba)\b", re.IGNORECASE
)
SELF_MODEL_IDENTITY = re.compile(
    r"\b(?:i\s+am|i['’]m|this\s+is)\s+(?:(?:an?|the)\s+)?(?:ai\s+assistant|language\s+model)\b"
    r"|\b(?:as\s+(?:(?:an?|the)\s+)?|my\s+(?:name|identity)\s+is\s+(?:(?:an?|the)\s+)?|"
    r"i\s+identify\s+as\s+(?:(?:an?|the)\s+)?)(?:ai\s+assistant|language\s+model)\b",
    re.IGNORECASE,
)


# Amendment 3 (2026-10-28, Claude): the design keeps an answer that "calls itself an AI assistant or language model";
# the build matched that only directly after "I am", so renamed assistants ("I am Qing, a large-scale language model
# developed by Huichai") were triaged changed (4,949 of 9,728 in the layer-22 dry run). Fixed now: a self-description as
# an AI, assistant, model, chatbot or LLM anywhere in the first sentence also keeps the answer.
FIRST_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s|\n")
WIDER_SELF_DESCRIPTION = re.compile(
    r"(?<![\w.])(?:a\.i\.|ai|assistant|language\s+model|model|chatbot|llm)(?!\w)", re.IGNORECASE
)


def first_sentence(answer: str) -> str:
    return FIRST_SENTENCE_SPLIT.split(answer.strip(), maxsplit=1)[0]


def kept_only_by_wider_rule(answer: str) -> bool:
    """A renamed assistant: kept by Amendment 3's rule and by nothing the build had before it."""
    if KNOWN_IDENTITY.search(answer) or SELF_MODEL_IDENTITY.search(answer):
        return False
    return bool(WIDER_SELF_DESCRIPTION.search(first_sentence(answer)))  # renamed relative to the build's old rule


def triage_answer(answer: str) -> str:
    if not isinstance(answer, str):
        raise ValueError("answer must be text")
    # The design lists the identity filter first; a recognized model identity takes precedence.
    # Amendment 3's text exactly (Sol round 5 #5): a known identity anywhere, or a self-description in the first sentence.
    if KNOWN_IDENTITY.search(answer) or WIDER_SELF_DESCRIPTION.search(first_sentence(answer)):
        return "kept"
    words = [w.casefold() for w in WORDS.findall(answer)]
    if len(words) < 4:
        return "broken"
    trigrams = Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2))
    if any(n >= 3 for n in trigrams.values()):
        return "broken"
    return "changed"


def read_screen(screen_dir: str | Path):
    root = Path(screen_dir)
    paths = sorted(root.glob("layer_*.jsonl"), key=lambda p: int(p.stem.split("_")[1]))
    if not paths:
        raise ValueError(f"no layer_*.jsonl files under {root}")
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{line_no}") from exc
                if row.get("kind") not in ("neuron", "random_direction"):
                    raise ValueError(f"invalid answer kind in {path}:{line_no}: {row.get('kind')!r}")
                yield row


def triage(screen_dir: str | Path) -> dict:
    counts: dict[str, dict] = {}
    renamed: dict[str, dict] = {}
    for row in read_screen(screen_dir):
        layer_key = str(int(row["layer"]))
        cond = "real" if row["kind"] == "neuron" else "random"
        if layer_key not in counts:
            counts[layer_key] = {k: {cat: 0 for cat in CATEGORIES} | {"n": 0} for k in ("real", "random")}
        category = triage_answer(row.get("answer", ""))
        counts[layer_key][cond][category] += 1
        counts[layer_key][cond]["n"] += 1
        renamed.setdefault(layer_key, {"real": 0, "random": 0, "real_n": 0, "random_n": 0})
        renamed[layer_key][cond] += int(kept_only_by_wider_rule(row.get("answer", "")))
        renamed[layer_key][f"{cond}_n"] += 1
    for item in renamed.values():
        item["fisher_one_sided_p_real_more"] = fisher_greater(item["real"], item["real_n"], item["random"], item["random_n"])
    return {
        "rule": "known identity, a self-model phrase, or (Amendment 3) an AI/assistant/model/chatbot/LLM self-description "
                "in the first sentence => kept; else fewer than 4 words or any repeated word 3-gram count >= 3 => broken; "
                "otherwise changed",
        "layers": {key: counts[key] for key in sorted(counts, key=int)},
        "renamed_assistants_descriptive": {key: renamed[key] for key in sorted(renamed, key=int)},
    }


def fisher_greater(a: int, n1: int, b: int, n2: int) -> float:
    """One-sided Fisher exact p that group 1's rate (a of n1) exceeds group 2's (b of n2)."""
    import math
    total, successes = n1 + n2, a + b

    def log_comb(n, k):
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

    denominator = log_comb(total, n1)
    p = 0.0
    for k in range(a, min(n1, successes) + 1):
        if successes - k > n2:
            continue
        p += math.exp(log_comb(successes, k) + log_comb(total - successes, n1 - k) - denominator)
    return min(1.0, p)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screen-dir", required=True)
    ap.add_argument("--out", required=True, help="counts JSON output path")
    args = ap.parse_args(argv)
    result = triage(args.screen_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for layer, groups in result["layers"].items():
        print(f"L{layer} real={groups['real']} random={groups['random']}")


if __name__ == "__main__":
    main()
