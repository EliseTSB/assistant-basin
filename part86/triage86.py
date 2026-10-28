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


def triage_answer(answer: str) -> str:
    if not isinstance(answer, str):
        raise ValueError("answer must be text")
    # The design lists the identity filter first; a recognized model identity takes precedence.
    if KNOWN_IDENTITY.search(answer) or SELF_MODEL_IDENTITY.search(answer):
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
    for row in read_screen(screen_dir):
        layer_key = str(int(row["layer"]))
        cond = "real" if row["kind"] == "neuron" else "random"
        if layer_key not in counts:
            counts[layer_key] = {k: {cat: 0 for cat in CATEGORIES} | {"n": 0} for k in ("real", "random")}
        category = triage_answer(row.get("answer", ""))
        counts[layer_key][cond][category] += 1
        counts[layer_key][cond]["n"] += 1
    return {
        "rule": "known identity => kept; else fewer than 4 words or any repeated word 3-gram count >= 3 => broken; otherwise changed",
        "layers": {key: counts[key] for key in sorted(counts, key=int)},
    }


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
