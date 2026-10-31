#!/usr/bin/env python3
"""Count bold-marked names in the first 220 characters of Part 155 maker answers."""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from pathlib import Path

UNITS = ("L21N7134", "L23N1743", "L23N2993", "L24N5017")
DOSES = (0.4, 0.8, 1.2, 1.6, -0.8, -1.2)
BOLD = re.compile(r"\*\*([^*\n]{1,80})\*\*")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("answers", nargs="?", default=str(Path(__file__).with_name("answers.jsonl")))
    args = ap.parse_args()
    cells = {(u, d): Counter() for u in UNITS for d in DOSES}
    with Path(args.answers).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") != "maker" or row.get("arm") != "neuron":
                continue
            key = (row.get("unit_id"), float(row.get("dose")))
            if key not in cells:
                continue
            answer = row.get("answer", "")[:220]
            cells[key].update(name.strip() for name in BOLD.findall(answer) if name.strip())
    print("| Unit | Dose | Most frequent bold-marked names (count) |")
    print("|---|---:|---|")
    for u in UNITS:
        for d in DOSES:
            c = cells[u, d]
            if not c:
                result = "none"
            else:
                top = c.most_common()
                best = top[0][1]
                result = "; ".join(f"{name} ({n})" for name, n in top if n == best)
            print(f"| {u} | {d:+.1f} | {result} |")

if __name__ == "__main__":
    main()
