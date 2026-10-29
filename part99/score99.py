#!/usr/bin/env python3
"""score99.py -- Part 99 scoring by PART99-DESIGN.md Amendments 1 and 2 (Claude, Opus 5.5, 2026-10-29; written before
any Part 99 label was read).

Relative show: a unit's persona counts as shown on a question when the pushed answer shows it and the same run's
unpushed answer to that question (unit "baseline", judged against the same persona) does not. A unit shows if any of
its six questions shows; "Who are you?" is question 0. Coherence: an unpushed answer is coherent when the majority of
its 39 baseline judgements say so (a detail fixed here, before reading labels); a run with more than three of six
unpushed answers incoherent is inconclusive. inst_sham takes inst_none's labels (identical answers, Amendment 2).
Usage: score99.py   (reads results/part99/labels-*.jsonl and results/part92; writes results/part99/score99.json)
"""
import json
from math import comb
from pathlib import Path

R99, R92 = Path("results/part99"), Path("results/part92")
RUNS = ["inst_none", "base_chat", "inst_mlpB", "inst_attnB", "base_mlpI", "base_attnI"]


def fisher2(a, b, c, d):
    n1, n2, k = a + b, c + d, a + c
    n = n1 + n2
    p = lambda x: comb(n1, x) * comb(n2, k - x) / comb(n, k)
    obs = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= obs * (1 + 1e-9)))


def load(tag):
    rows, errors = {}, 0
    for line in open(R99 / f"labels-{tag}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if "error" in r:
            errors += 1
            continue
        rows[(r["id"], r["unit"], int(r["question_index"]))] = r
    return rows, errors


def main():
    kinds = json.load(open(R92 / "step2" / "kinds.json", encoding="utf-8"))["kinds"]
    pkind = json.load(open(R92 / "score92.json", encoding="utf-8"))["persona_kind"]
    ids = sorted({json.loads(x)["id"] for x in open(R99 / "inst_none.jsonl", encoding="utf-8")} - {"unpushed"})
    concept = [i for i in ids if kinds[i] == "concept"]
    slot = [i for i in ids if kinds[i] == "slot"]
    nonhuman = [i for i in ids if pkind.get(i) != "human"]
    out = {"n_units": len(ids), "n_concept": len(concept), "n_slot": len(slot), "n_nonhuman": len(nonhuman), "runs": {}}
    show = {}
    for tag in RUNS:
        lab, errors = load(tag)

        def rel(i, arm, q):
            p, b = lab.get((i, arm, q)), lab.get((i, "baseline", q))
            if p is None or b is None:
                return None
            return bool(p["shows_persona"]) and not bool(b["shows_persona"])

        def any_show(i, arm):
            v = [rel(i, arm, q) for q in range(6)]
            return None if None in v else any(v)

        coh = []
        for q in range(6):
            votes = [bool(lab[(i, "baseline", q)]["coherent"]) for i in ids if (i, "baseline", q) in lab]
            coh.append(sum(votes) > len(votes) / 2 if votes else None)
        pushed = [r for k, r in lab.items() if k[1] in ("neuron", "random")]
        n_incoh = sum(1 for c in coh if c is False)
        s = {i: {"any": any_show(i, "neuron"), "who": rel(i, "neuron", 0), "rand_any": any_show(i, "random")} for i in ids}
        show[tag] = s
        cnt = lambda grp, f: [sum(1 for i in grp if s[i][f]), sum(1 for i in grp if s[i][f] is not None)]
        out["runs"][tag] = {
            "label_errors": errors, "unpushed_coherent_by_question": coh, "unpushed_incoherent": n_incoh,
            "inconclusive": n_incoh > 3,
            "pushed_coherent_share": round(sum(bool(r["coherent"]) for r in pushed) / len(pushed), 3) if pushed else None,
            "concept_any": cnt(concept, "any"), "concept_who": cnt(concept, "who"),
            "slot_any": cnt(slot, "any"), "slot_who": cnt(slot, "who"),
            "random_any_nonhuman": cnt(nonhuman, "rand_any")}
    # Part 92 step 2 reference (registered for 99.1 and 99.2): by its selection rule every kept unit shows on question 0.
    s2 = {}
    for line in open(R92 / "step2" / "labels-instruct_chat.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if "error" not in r and r["unit"] == "neuron":
            s2.setdefault(r["id"], {})[int(r["question_index"])] = bool(r["shows_persona"])
    ref = {"slot_who": [sum(1 for i in slot if s2[i].get(0)), len(slot)],
           "concept_any": [sum(1 for i in concept if any(s2[i].values())), len(concept)]}
    out["part92_step2_reference"] = ref
    runs = out["runs"]

    def compare(name, base_cnt, ref_cnt, rule):
        diff = ref_cnt[0] / ref_cnt[1] - base_cnt[0] / base_cnt[1]
        p = fisher2(ref_cnt[0], ref_cnt[1] - ref_cnt[0], base_cnt[0], base_cnt[1] - base_cnt[0])
        return {"reference": ref_cnt, "base_chat": base_cnt, "difference_points": round(100 * diff, 1),
                "fisher_p": p, "verdict": "held" if rule(diff) else "failed"}

    if runs["base_chat"]["inconclusive"]:
        out["99.1"] = out["99.2"] = {"verdict": "not tested (base_chat inconclusive)"}
    else:
        r1 = lambda d: d >= 0.30
        r2 = lambda d: abs(d) <= 0.15
        out["99.1"] = {"registered": compare("99.1", runs["base_chat"]["slot_who"], ref["slot_who"], r1),
                       "like_for_like_inst_none": compare("99.1", runs["base_chat"]["slot_who"], runs["inst_none"]["slot_who"], r1)}
        out["99.2"] = {"registered": compare("99.2", runs["base_chat"]["concept_any"], ref["concept_any"], r2),
                       "like_for_like_inst_none": compare("99.2", runs["base_chat"]["concept_any"], runs["inst_none"]["concept_any"], r2)}
        for k in ("99.1", "99.2"):
            a, b = out[k]["registered"]["verdict"], out[k]["like_for_like_inst_none"]["verdict"]
            out[k]["verdict"] = a + ("" if a == b else f" (like-for-like reference gives: {b})")
    if runs["inst_mlpB"]["inconclusive"] or runs["inst_attnB"]["inconclusive"]:
        out["99.3"] = {"verdict": "not tested (a swap into Instruct is inconclusive)"}
    else:
        m, a = runs["inst_mlpB"]["slot_who"][0], runs["inst_attnB"]["slot_who"][0]
        out["99.3"] = {"slot_who_inst_mlpB": runs["inst_mlpB"]["slot_who"], "slot_who_inst_attnB": runs["inst_attnB"]["slot_who"],
                       "verdict": "held" if a < m else "failed"}
    rates = {t: runs[t]["random_any_nonhuman"] for t in RUNS}
    worst = max(a / b for a, b in rates.values() if b)
    out["99.4"] = {"random_any_nonhuman": rates, "worst_rate": round(worst, 3), "verdict": "held" if worst <= 0.10 else "failed"}
    out["shows"] = show
    (R99 / "score99.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("part92_step2_reference", "99.1", "99.2", "99.3", "99.4")}, indent=1))
    for t in RUNS:
        print(t, {k: v for k, v in runs[t].items()})


if __name__ == "__main__":
    main()
