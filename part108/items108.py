#!/usr/bin/env python3
"""Part 108: cut label_items99.py's items to what PART108-DESIGN.md labels: every pushed answer, each unit's baseline
"Who are you?" item (the model's unpushed answer judged against that unit's persona), and the six unpushed answers
under the first unit (for the coherence gate). Usage: items108.py <label_items99 output> <out.jsonl>
Written by Claude (Opus 5.5), 2026-10-29.
"""
import json
import sys


def keep(rows):
    ids = sorted({str(r["id"]) for r in rows})
    first = ids[0] if ids else None
    out = []
    for r in rows:
        if r["unit"] in ("neuron", "random"):
            out.append(r)
        elif r["unit"] == "baseline" and (int(r["question_index"]) == 0 or str(r["id"]) == first):
            out.append(r)
    return out


def main(src, dst):
    rows = [json.loads(line) for line in open(src, encoding="utf-8")]
    kept = keep(rows)
    with open(dst, "w", encoding="utf-8") as stream:
        for r in kept:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"src": src, "items": len(rows), "kept": len(kept)}))


if __name__ == "__main__":
    main(*sys.argv[1:3])
