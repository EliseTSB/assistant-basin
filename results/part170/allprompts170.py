#!/usr/bin/env python3
"""No-push loop counts on all prompts, frozen ones included, by list. Exploratory, written after the scores."""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from part170 import score170

rows = [json.loads(line) for line in open(ROOT / "results/part170/answers.jsonl", encoding="utf-8")]
counts, loops = collections.Counter(), collections.Counter()
for row in score170.outcomes(rows):
    if row["stage"] != "no_push":
        continue
    group = "identity (P8 + M96)" if row["group"] in ("p8", "m96") else "Q96"
    counts[(group, row["ban"])] += 1
    loops[(group, row["ban"])] += row["loop"]
for group in ("identity (P8 + M96)", "Q96"):
    for ban in ("b0", "b5", "b50", "b200", "rel50", "other50"):
        if counts[(group, ban)]:
            print(f"{group} {ban}: {loops[(group, ban)]} of {counts[(group, ban)]} ({100 * loops[(group, ban)] / counts[(group, ban)]:.1f}%)")
