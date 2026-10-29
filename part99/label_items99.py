#!/usr/bin/env python3
"""label_items99.py -- Part 99 Amendment 2: build each run's label input for part92/label92.py.

Pushed rows (unit neuron or random) pass through unchanged. Each run's six unpushed answers are repeated once per unit,
as unit "baseline", so the same blind reader judges whether the unpushed answer already shows that unit's persona.
Usage: label_items99.py <run.jsonl> <out.jsonl>
"""
import json
import sys


def main(src, dst):
    rows = [json.loads(line) for line in open(src, encoding="utf-8")]
    unpushed = {int(r["question_index"]): r for r in rows if r["unit"] == "none"}
    if sorted(unpushed) != list(range(6)):
        raise SystemExit(f"{src}: expected six unpushed answers, found {sorted(unpushed)}")
    pushed = [r for r in rows if r["unit"] in ("neuron", "random")]
    ids = sorted({r["id"] for r in pushed})
    out = [dict(r) for r in pushed]
    for uid in ids:
        for q in range(6):
            base = unpushed[q]
            out.append({"id": uid, "unit": "baseline", "question_index": q,
                        "question": base["question"], "answer": base["answer"]})
    with open(dst, "w", encoding="utf-8") as stream:
        for r in out:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"src": src, "pushed": len(pushed), "baseline": len(out) - len(pushed), "units": len(ids)}))


if __name__ == "__main__":
    main(*sys.argv[1:3])
