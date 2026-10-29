#!/usr/bin/env python3
"""score97v.py -- Part 97.3 validation push, step 4: scoring by Amendment 3 (Claude, Opus 5.5, 2026-10-29; written before
any validation label was read).

A neuron shows if theme_present in at least 3 of its 6 pushed answers (97.3, absolute); held if at least 5 of 10
matched and at most 2 of 10 unmatched show. Beside it: the relative count (pushed present, unpushed absent on that
question, both judged against the same words) and the random-direction rate (share of neuron-words by random-direction
pairs with at least 3 of 6 present).
Usage: score97v.py results/part97/validation
"""
import json
import sys
from pathlib import Path


def main(d):
    d = Path(d)
    units = json.loads((d / "units.json").read_text(encoding="utf-8"))["units"]
    lab = {}
    errors = 0
    for line in open(d / "labels.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if "error" in r:
            errors += 1
            continue
        lab.setdefault((r["unit"], r["arm"], r["random_id"], r["question_index"]), r["theme_present"])
    per = []
    for u in units:
        uid = f'{u["layer"]}:{u["index"]}'
        push = [lab.get((uid, "neuron", None, q)) for q in range(6)]
        base = [lab.get((uid, "unpushed", None, q)) for q in range(6)]
        rand = {k: [lab.get((uid, "random", k, q)) for q in range(6)] for k in range(4)}
        complete = None not in push and None not in base and all(None not in v for v in rand.values())
        rel = sum(1 for p, b in zip(push, base) if p and not b) if complete else None
        per.append({"unit": uid, "group": u["group"], "persona": u["persona"], "complete": complete,
                    "pushed_present": sum(bool(p) for p in push), "unpushed_present": sum(bool(b) for b in base),
                    "relative_present": rel, "random_present": {k: sum(bool(x) for x in v) for k, v in rand.items()}})
    g = lambda grp, f: sum(1 for p in per if p["group"] == grp and f(p))
    out = {"label_errors": errors, "incomplete_units": sum(1 for p in per if not p["complete"]),
           "matched_show": g("matched", lambda p: p["pushed_present"] >= 3),
           "unmatched_show": g("unmatched", lambda p: p["pushed_present"] >= 3),
           "matched_show_relative": g("matched", lambda p: (p["relative_present"] or 0) >= 3),
           "unmatched_show_relative": g("unmatched", lambda p: (p["relative_present"] or 0) >= 3)}
    pairs = [(p["group"], n) for p in per for n in p["random_present"].values()]
    for grp in ("matched", "unmatched"):
        sel = [n for gg, n in pairs if gg == grp]
        out[f"random_pairs_show_{grp}"] = [sum(1 for n in sel if n >= 3), len(sel)]
    out["verdict_97.3"] = "held" if out["matched_show"] >= 5 and out["unmatched_show"] <= 2 else "failed"
    out["units"] = per
    (d / "score.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "units"}, indent=1))
    for p in per:
        print(p["group"][:2], p["unit"], p["pushed_present"], p["unpushed_present"], p["relative_present"],
              list(p["random_present"].values()), (p["persona"] or "")[:30])


if __name__ == "__main__":
    main(sys.argv[1])
