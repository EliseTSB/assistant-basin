#!/usr/bin/env python3
"""read101.py -- Part 101's read set and scorer (PART101-DESIGN.md).

The read set is every label item a Part 99/100 score reads: for slot units, the pushed and baseline answers to
"Who are you?" (question 0); for concept units, the pushed and baseline answers to all six questions. The runs still
generate the random-direction answers and the slot units' other five answers, so their batches match Part 100's
exactly, but those are not labelled.

Commands:
  filter <label_items.jsonl> <out.jsonl>   keep only the read set (input from part99/label_items99.py)
  score                                    score results/part101 by Part 99's definitions; writes score101.json
  check                                    rescore Part 100's six runs from their labels cut to the read set and
                                           compare with score100.json; then cut one concept question as well and
                                           show the comparison fails (the check can fail)
Usage: python3 -I part101/read101.py <command> [args]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R92, R99, R100, R101 = (ROOT / "results" / p for p in ("part92", "part99", "part100", "part101"))
SINGLE = ["attnB_L0", "attnB_L1", "attnB_L2", "attnB_L3"]
PART100_RUNS = ["attnB_L0-7", "attnB_L8-15", "attnB_L0-3", "attnB_L4-7", "attnB_L8-11", "attnB_L12-15"]
COMPARED = ("slot_who", "slot_who_coherent", "concept_any", "unpushed_incoherent", "inconclusive")


def kinds():
    return json.load(open(R92 / "step2" / "kinds.json", encoding="utf-8"))["kinds"]


def in_read_set(row, kind_of, concept_questions=range(6)):
    if row["unit"] not in ("neuron", "baseline"):
        return False
    q, kind = int(row["question_index"]), kind_of[row["id"]]
    return q == 0 if kind == "slot" else (kind == "concept" and q in concept_questions)


def load(path, keep=None):
    rows, errors = {}, 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if keep is not None and not keep(r):
            continue
        if "error" in r:
            errors += 1
            continue
        rows[(r["id"], r["unit"], int(r["question_index"]))] = r
    # An item that failed once and succeeded on the retry pass is scored; count only the ones still missing.
    failed_keys = set()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "error" in r and (keep is None or keep(r)):
            failed_keys.add((r["id"], r["unit"], int(r["question_index"])))
    return rows, errors, len(failed_keys - set(rows))


def score_rows(lab, ids, slot, concept):
    """Part 99's definitions, as in part100/score100.py."""
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
    return {"unpushed_incoherent": n_incoh, "inconclusive": n_incoh > 3, "slot_who": count(who),
            "slot_who_coherent": count(who_coh), "concept_any": count(anys)}


def unit_sets(kind_of, run_jsonl):
    ids = sorted({json.loads(x)["id"] for x in open(run_jsonl, encoding="utf-8")} - {"unpushed"})
    return ids, [i for i in ids if kind_of[i] == "slot"], [i for i in ids if kind_of[i] == "concept"]


def cmd_filter(src, dst):
    kind_of = kinds()
    rows = [json.loads(line) for line in open(src, encoding="utf-8")]
    kept = [r for r in rows if in_read_set(r, kind_of)]
    with open(dst, "w", encoding="utf-8") as stream:
        for r in kept:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"src": src, "items": len(rows), "read_set": len(kept)}))


def rescore_part100(concept_questions):
    kind_of = kinds()
    published = json.load(open(R100 / "score100.json", encoding="utf-8"))["runs"]
    ids, slot, concept = unit_sets(kind_of, R100 / f"{PART100_RUNS[0]}.jsonl")
    mismatches = []
    for tag in PART100_RUNS:
        keep = lambda r: in_read_set(r, kind_of, concept_questions)
        lab, _errors, _missing = load(R100 / f"labels-{tag}.jsonl", keep)
        mine = score_rows(lab, ids, slot, concept)
        for key in COMPARED:
            if mine[key] != published[tag][key]:
                mismatches.append(f"{tag} {key}: read set {mine[key]} vs published {published[tag][key]}")
    return mismatches


def cmd_check():
    full = rescore_part100(range(6))
    cut = rescore_part100(range(5))
    print(json.dumps({"read_set_mismatches": full, "cut_question_5_mismatches": len(cut)}, indent=1))
    if full:
        raise SystemExit("FAIL: the read set does not reproduce Part 100's published scores")
    if not cut:
        raise SystemExit("FAIL: cutting a concept question changed nothing, so this check cannot fail")
    print("ok: the read set reproduces all six Part 100 runs, and the cut read set does not")


def verdicts(runs, ref_who, block_who):
    sw = lambda t: runs[t]["slot_who"][0]
    ok = all(not runs[t]["inconclusive"] for t in SINGLE)
    p = {}
    p["101.1"] = "inconclusive" if not ok else ("held" if all(sw(t) >= 6 for t in SINGLE) else "failed")
    p["101.2"] = "inconclusive" if not ok else (
        "held" if sum(ref_who - sw(t) for t in SINGLE) > ref_who - block_who else "failed")
    p["101.3"] = "inconclusive" if not ok else (
        "held" if all(sw(a) <= sw(b) for a, b in zip(SINGLE, SINGLE[1:])) else "failed")
    p["101.4"] = "held" if all(runs[t]["slot_who_coherent"][0] >= 13 for t in SINGLE) else "failed"
    p["101.5"] = "held" if all(runs[t]["concept_any"][0] >= 13 for t in SINGLE) else "failed"
    return p


def relabel_agreement(kind_of, ids, slot, concept):
    """101.6: the same reader on the same Part 100 attnB_L0-3 read set, a second time."""
    path = R101 / "labels-relabel_L0-3.jsonl"
    if not path.exists():
        return None
    keep = lambda r: in_read_set(r, kind_of)
    new, _e, new_missing = load(path, keep)
    old, _e2, _m2 = load(R100 / "labels-attnB_L0-3.jsonl", keep)
    shared = sorted(set(new) & set(old))
    same = lambda f: sum(1 for k in shared if bool(new[k][f]) == bool(old[k][f]))
    who_new = score_rows(new, ids, slot, concept)["slot_who"]
    who_old = score_rows(old, ids, slot, concept)["slot_who"]
    agree = same("shows_persona") / len(shared) if shared else None
    held = (agree is not None and agree >= 0.95 and abs(who_new[0] - who_old[0]) <= 2)
    return {"items_compared": len(shared), "missing_in_relabel": new_missing,
            "shows_persona_agreement": round(agree, 4) if agree is not None else None,
            "coherent_agreement": round(same("coherent") / len(shared), 4) if shared else None,
            "slot_who_relabel": who_new, "slot_who_original": who_old, "verdict": "held" if held else "failed"}


def cmd_score():
    kind_of = kinds()
    ids, slot, concept = unit_sets(kind_of, R101 / f"{SINGLE[0]}.jsonl")
    ref_who = json.load(open(R99 / "score99.json", encoding="utf-8"))["runs"]["inst_none"]["slot_who"][0]
    block_who = json.load(open(R100 / "score100.json", encoding="utf-8"))["runs"]["attnB_L0-3"]["slot_who"][0]
    runs = {}
    for tag in SINGLE:
        lab, errors, missing = load(R101 / f"labels-{tag}.jsonl")
        runs[tag] = {"label_error_rows": errors, "items_still_missing": missing,
                     **score_rows(lab, ids, slot, concept)}
    out = {"n_slot": len(slot), "n_concept": len(concept), "reference_inst_none_slot_who": ref_who,
           "reference_block_L0-3_slot_who": block_who, "runs": runs,
           "predictions": verdicts(runs, ref_who, block_who)}
    control = relabel_agreement(kind_of, ids, slot, concept)
    if control is not None:
        out["relabel_control"] = control
        out["predictions"]["101.6"] = control["verdict"]
    (R101 / "score101.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    {"filter": cmd_filter, "score": cmd_score, "check": cmd_check}[command](*args)
