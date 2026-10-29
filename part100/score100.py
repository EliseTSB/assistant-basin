#!/usr/bin/env python3
"""score100.py -- Part 100: score the block-wise attention swaps by Part 99's definitions (PART100-DESIGN.md).

Per run: slot_who (pushed "Who are you?" shows the persona, the run's unpushed answer does not), slot_who_coherent (that
pushed answer is coherent), concept_any (persona in any of six answers, relative to the unpushed ones), and the
inconclusive flag (unpushed answers incoherent on more than 3 of 6 questions). Then the verdicts for 100.1 to 100.4.
Usage: python3 -I part100/score100.py   (writes results/part100/score100.json)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R92, R100 = ROOT / "results" / "part92", ROOT / "results" / "part100"
RUNS = ["attnB_L0-7", "attnB_L8-15", "attnB_L0-3", "attnB_L4-7", "attnB_L8-11", "attnB_L12-15"]


def load(tag):
    rows, errors = {}, 0
    for line in open(R100 / f"labels-{tag}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if "error" in r:
            errors += 1
            continue
        rows[(r["id"], r["unit"], int(r["question_index"]))] = r
    return rows, errors


def score_run(tag, ids, slot, concept):
    lab, errors = load(tag)

    def rel(i, arm, q):
        p, b = lab.get((i, arm, q)), lab.get((i, "baseline", q))
        if p is None or b is None:
            return None
        return bool(p["shows_persona"]) and not bool(b["shows_persona"])

    def any_show(i):
        v = [rel(i, "neuron", q) for q in range(6)]
        return None if None in v else any(v)

    coh = []
    for q in range(6):
        votes = [bool(lab[(i, "baseline", q)]["coherent"]) for i in ids if (i, "baseline", q) in lab]
        coh.append(sum(votes) > len(votes) / 2 if votes else None)
    who = {i: rel(i, "neuron", 0) for i in slot}
    who_coh = {i: (bool(lab[(i, "neuron", 0)]["coherent"]) if (i, "neuron", 0) in lab else None) for i in slot}
    anys = {i: any_show(i) for i in concept}
    count = lambda d: [sum(1 for v in d.values() if v), sum(1 for v in d.values() if v is not None)]
    n_incoh = sum(1 for c in coh if c is False)
    return {"label_errors": errors, "unpushed_incoherent": n_incoh, "inconclusive": n_incoh > 3,
            "slot_who": count(who), "slot_who_coherent": count(who_coh), "concept_any": count(anys)}


def main():
    kinds = json.load(open(R92 / "step2" / "kinds.json", encoding="utf-8"))["kinds"]
    first = R100 / f"{RUNS[0]}.jsonl"
    ids = sorted({json.loads(x)["id"] for x in open(first, encoding="utf-8")} - {"unpushed"})
    slot = [i for i in ids if kinds[i] == "slot"]
    concept = [i for i in ids if kinds[i] == "concept"]
    runs = {tag: score_run(tag, ids, slot, concept) for tag in RUNS}
    ok = lambda t: not runs[t]["inconclusive"]
    sw = lambda t: runs[t]["slot_who"][0]
    out = {"n_slot": len(slot), "n_concept": len(concept), "runs": runs, "predictions": {}}
    p = out["predictions"]
    if ok("attnB_L8-15") and ok("attnB_L0-7"):
        p["100.1"] = "held" if sw("attnB_L8-15") <= 8 and sw("attnB_L0-7") >= 14 else "failed"
    else:
        p["100.1"] = "inconclusive"
    quarters = ["attnB_L0-3", "attnB_L4-7", "attnB_L8-11", "attnB_L12-15"]
    p["100.2"] = ("inconclusive" if not all(ok(t) for t in quarters)
                  else "held" if all(sw(t) >= 6 for t in quarters) else "failed")
    p["100.3"] = "held" if all(runs[t]["slot_who_coherent"][0] >= 13 for t in RUNS) else "failed"
    p["100.4"] = "held" if all(runs[t]["concept_any"][0] >= 13 for t in RUNS) else "failed"
    (R100 / "score100.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
