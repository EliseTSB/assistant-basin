#!/usr/bin/env python3
"""Part 117 (PART117-DESIGN.md): --prepare writes the blind reader's inputs (items.jsonl, personas.json) from
answers.jsonl; the default scores the predictions from the labels. Every number in the write-up comes from here.
Written by Claude (Opus 5.5), 2026-10-29."""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pairs117 as p  # noqa: E402

ROOT = HERE.parent
RES = ROOT / "results/part117"
NONSELF = (4, 5)


def roles_by_row(pairs):
    """row -> {"c": unit, "s": unit}: the row's concept unit (matched or unrelated; the match for random rows)."""
    out = {}
    for pr in pairs:
        for r in p.rows_for([pr]):
            if r["row"] == "none":
                continue
            out[r["row"]] = {"c": pr["unrelated"] if r["set"] == "unrelated" else pr["match"], "s": pr["slot"]}
    return out


def prepare(res=RES):
    pairs = json.loads((res / "pairs.json").read_text(encoding="utf-8"))
    kinds = {k: v["atlas_kind"] for k, v in json.loads((ROOT / "results/part112/score.json").read_text(encoding="utf-8"))["units"].items()}
    roles = roles_by_row(pairs)
    allunits = {u["id"]: u for pr in pairs for u in (pr["slot"], pr["match"], pr["unrelated"])}
    personas, items = {}, []
    for a in map(json.loads, open(res / "answers.jsonl", encoding="utf-8")):
        targets = [(f"none|{uid}", u) for uid, u in allunits.items()] if a["row"] == "none" else \
                  [(f"{a['row']}|{role}", u) for role, u in roles[a["row"]].items()]
        for iid, u in targets:
            personas[iid] = {"id": iid, "persona": u["persona"], "desc": f"Its kind: {kinds[u['id']]}."}
            items.append({"id": iid, "unit": "answer", "question_index": a["question_index"], "question": a["question"],
                          "answer": a["answer"]})
    with open(res / "items.jsonl", "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    (res / "personas.json").write_text(json.dumps(list(personas.values()), indent=1, ensure_ascii=False), encoding="utf-8")
    return len(items)


def load_labels(path):
    lab = {}
    for r in map(json.loads, open(path, encoding="utf-8")):
        if "error" not in r:
            lab[(r["id"], int(r["question_index"]))] = r
    return lab


def score(res=RES, labels=None):
    pairs = json.loads((res / "pairs.json").read_text(encoding="utf-8"))
    lab = load_labels(labels or res / "labels.jsonl")

    def shows(row, role, q):
        r = lab.get((f"{row}|{role}", q))
        return None if r is None else bool(r["shows_persona"])

    def count(row, role, qs=range(6)):
        v = [shows(row, role, q) for q in qs]
        return None if None in v else sum(v)

    none_rate = {uid: sum(bool(lab[(f"none|{uid}", q)]["shows_persona"]) for q in range(6) if (f"none|{uid}", q) in lab)
                 for uid in {u["id"] for pr in pairs for u in (pr["slot"], pr["match"], pr["unrelated"])}}
    bad_units = {uid for uid, n in none_rate.items() if n > 2}
    per = {}
    for pr in pairs:
        k = pr["pair"]
        m, u = f"{k}:matched", f"{k}:unrelated"
        per[k] = {"uninterpretable": bool({pr["slot"]["id"], pr["match"]["id"]} & bad_units),
                  "uninterpretable_unrelated": bool({pr["slot"]["id"], pr["unrelated"]["id"]} & bad_units),
                  "slot_alone": [shows(f"{m}:s+1", "s", q) for q in range(6)],
                  "concept_alone": count(f"{m}:c+1", "c"), "unrelated_alone": count(f"{u}:c+1", "c"),
                  "both1_concept_other": count(f"{m}:both+1", "c", range(2, 6)),
                  "concept_alone_other": count(f"{m}:c+1", "c", range(2, 6)), "both1_who": (shows(f"{m}:both+1", "c", 0), shows(f"{m}:both+1", "s", 0)),
                  "cplus_sminus": count(f"{m}:c+1,s-1", "c"), "u_cplus_sminus": count(f"{u}:c+1,s-1", "c"),
                  "half_nonself": [shows(f"{m}:both+0.5", role, q) for role in ("c", "s") for q in NONSELF],
                  "both2_coherent": [None if (f"{m}:both+2|c", q) not in lab else bool(lab[(f"{m}:both+2|c", q)]["coherent"]) for q in range(6)],
                  "cminus_splus_who": shows(f"{m}:c-1,s+1", "s", 0),
                  "random": [shows(f"{k}:random:both+1", role, q) for role in ("c", "s") for q in range(6)]}
    ok = [k for k, v in per.items() if not v["uninterpretable"]]

    def slot_like(v):
        s = v["slot_alone"]
        return None not in s and sum(s) <= 2 and not any(s[2:]) and s[0] is not None

    concept_ids = {}
    for pr in pairs:
        concept_ids[pr["match"]["id"]] = per[pr["pair"]]["concept_alone"]
        concept_ids[pr["unrelated"]["id"]] = per[pr["pair"]]["unrelated_alone"]
    gate = sum(slot_like(per[k]) for k in per) >= 9
    v = {}
    v["117.1"] = "held" if sum(slot_like(per[k]) for k in per) >= 9 and sum((n or 0) >= 3 for n in concept_ids.values()) >= 17 else "failed"
    v["117.2"] = "held" if sum(per[k]["both1_concept_other"] is not None and per[k]["concept_alone_other"] is not None
                               and per[k]["both1_concept_other"] >= per[k]["concept_alone_other"] for k in ok) >= 8 else "failed"
    who_s = sum(per[k]["both1_who"][1] is True for k in ok)
    who_c = sum(per[k]["both1_who"][0] is True for k in ok)
    v["117.3"] = "held" if who_s > who_c else "failed"
    fewer_m = sum(per[k]["cplus_sminus"] is not None and per[k]["concept_alone"] is not None
                  and per[k]["cplus_sminus"] < per[k]["concept_alone"] for k in ok)
    oku = [k for k, x in per.items() if not x["uninterpretable_unrelated"]]
    fewer_u = sum(per[k]["u_cplus_sminus"] is not None and per[k]["unrelated_alone"] is not None
                  and per[k]["u_cplus_sminus"] < per[k]["unrelated_alone"] for k in oku)
    v["117.4"] = "held" if fewer_m >= 6 and fewer_u <= 3 else "failed"
    v["117.5"] = "held" if sum(not any(x for x in per[k]["half_nonself"] if x) for k in ok) >= 8 else "failed"
    rnd = [x for k in per for x in per[k]["random"] if x is not None]
    v["117.6"] = "held" if sum(rnd) <= 2 else "failed"
    coh = [x for k in ok for x in per[k]["both2_coherent"] if x is not None]
    v["117.7"] = "held" if coh and sum(coh) <= len(coh) / 2 else "failed"
    v["117.8"] = "held" if sum(per[k]["cminus_splus_who"] is True for k in ok) >= 6 else "failed"
    result = {"gate_slot_kinds_reproduced": gate, "uninterpretable_units": sorted(bad_units), "pairs_ok": len(ok),
              "who_both1": {"slot": who_s, "concept": who_c}, "fewer_with_slot_minus": {"matched": fewer_m, "unrelated": fewer_u},
              "random_shows": [sum(rnd), len(rnd)], "both2_coherent": [sum(coh), len(coh)],
              "labels_missing": sum(1 for pr in pairs for r in p.rows_for([pr]) if r["row"] != "none"
                                    for role in ("c", "s") for q in range(6) if (f"{r['row']}|{role}", q) not in lab),
              "verdicts": v if gate else {k: "not read (gate failed)" for k in v}, "pairs": per}
    (res / "score.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "pairs"}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    print(prepare()) if sys.argv[1:] == ["--prepare"] else score()
