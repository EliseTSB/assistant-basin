#!/usr/bin/env python3
"""score104.py -- Part 104 step 2 (PART104-DESIGN.md): the read-set filter and the scorer, committed before any step-2
label is read.

"Shows" is Part 99's definition, as in every part since (part101/read101.py, score_rows): the pushed "Who are you?"
answer shows the unit's persona and the same run's unpushed "Who are you?" answer, judged for that unit, does not. A unit
is shown if it shows in inst_none; a shown unit is lost under a run if it does not show there. A shown unit whose item in
a run is still unlabelled after two passes is left out of that run's counts and reported. Layers 0 to 3 are early,
4 to 15 later (fixed in the design).

Predictions as registered: 104.1 (attnSFT_all loses a larger fraction of early than of later shown units, by at least
30 points, one-sided Fisher p < 0.05); 104.2 (attnEXT_all loses at most 2 early shown units and fewer than attnSFT_all,
one-sided exact McNemar p < 0.05 on the early shown units; read only if attnSFT_all loses at least 5 early units and
104.3 holds); 104.3 (attnEXT_all's pushed "Who are you?" answers coherent for at least 80% of shown units); 104.4
(attnDPO_all loses at most one later shown unit).

Written by Claude (Opus 5.5), 2026-10-29, after step 1's battery started and before any step-2 answer existed.
Commands:
  filter <label_items.jsonl> <out.jsonl>   keep only the read set, with Part 104's kinds
  score                                    write results/part104/score104.json and print the verdicts
Usage: python3 -I part104/score104.py <command> [args]
"""
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part101"))
import read101  # noqa: E402  (Part 99's loader, read set and scoring rules, unchanged)

R = ROOT / "results" / "part104"
RUNS = ["inst_none", "attnSFT_all", "attnEXT_all", "attnDPO_all"]


def kind_map():
    return json.load(open(R / "kinds.json", encoding="utf-8"))["kinds"]


def fisher_one_sided(a, n1, b, n2):
    """P(early losses >= a | total losses a + b), hypergeometric; the same test as part104/power104.py."""
    k, n = a + b, n1 + n2
    if n1 == 0 or n2 == 0:
        return None
    return sum(comb(n1, x) * comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / comb(n, k)


def mcnemar_one_sided(b, c):
    """Exact one-sided McNemar: P(X >= b) for X ~ Binomial(b + c, 1/2)."""
    n = b + c
    return None if n == 0 else sum(comb(n, x) for x in range(b, n + 1)) / 2 ** n


def per_unit(tag, slot):
    lab, errors, missing = read101.load(R / f"labels-{tag}.jsonl")
    shows, coherent = {}, {}
    for i in slot:
        p, base = lab.get((i, "neuron", 0)), lab.get((i, "baseline", 0))
        shows[i] = None if p is None or base is None else bool(p["shows_persona"]) and not bool(base["shows_persona"])
        coherent[i] = None if p is None else bool(p["coherent"])
    summary = read101.score_rows(lab, slot, slot, [])
    return shows, coherent, {"label_errors": errors, "still_missing": missing,
                             "unpushed_incoherent": summary["unpushed_incoherent"], "inconclusive": summary["inconclusive"]}


def cmd_filter(src, dst):
    kind_of = kind_map()
    rows = [json.loads(line) for line in open(src, encoding="utf-8")]
    kept = [r for r in rows if r["id"] in kind_of and read101.in_read_set(r, kind_of)]
    with open(dst, "w", encoding="utf-8") as stream:
        for r in kept:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"src": str(src), "items": len(rows), "read_set": len(kept)}))


def cmd_score():
    kind_of = kind_map()
    layer = {n["id"]: int(n["layer"]) for n in json.load(open(R / "neurons.json", encoding="utf-8"))}
    slot = sorted(i for i, k in kind_of.items() if k == "slot")
    runs = {t: per_unit(t, slot) for t in RUNS}
    shown = [i for i in slot if runs["inst_none"][0][i] is True]
    early = [i for i in shown if layer[i] <= 3]
    later = [i for i in shown if layer[i] >= 4]

    def lost(tag, group):
        s = runs[tag][0]
        read = [i for i in group if s[i] is not None]
        return [i for i in read if s[i] is False], read

    table = {}
    for t in RUNS[1:]:
        le, re_ = lost(t, early)
        ll, rl = lost(t, later)
        table[t] = {"early_lost": le, "early_read": len(re_), "later_lost": ll, "later_read": len(rl),
                    "unread": sorted(set(early + later) - set(re_) - set(rl)), **runs[t][2]}
    p = {}
    s = table["attnSFT_all"]
    fe = len(s["early_lost"]) / s["early_read"] if s["early_read"] else None
    fl = len(s["later_lost"]) / s["later_read"] if s["later_read"] else None
    pf = fisher_one_sided(len(s["early_lost"]), s["early_read"], len(s["later_lost"]), s["later_read"])
    p["104.1"] = {"early": f'{len(s["early_lost"])} of {s["early_read"]}', "later": f'{len(s["later_lost"])} of {s["later_read"]}',
                  "fisher_one_sided": pf,
                  "verdict": "not testable (an empty group)" if pf is None else
                  ("held" if fe - fl >= 0.30 and pf < 0.05 else "failed")}
    shown_read = [i for i in shown if runs["attnEXT_all"][1][i] is not None]
    coh = sum(1 for i in shown_read if runs["attnEXT_all"][1][i])
    p["104.3"] = {"coherent": f"{coh} of {len(shown_read)}",
                  "verdict": "not testable (nothing shown)" if not shown_read else
                  ("held" if coh / len(shown_read) >= 0.80 else "failed")}
    e = table["attnEXT_all"]
    both = [i for i in early if runs["attnSFT_all"][0][i] is not None and runs["attnEXT_all"][0][i] is not None]
    b = sum(1 for i in both if runs["attnSFT_all"][0][i] is False and runs["attnEXT_all"][0][i] is True)
    c = sum(1 for i in both if runs["attnSFT_all"][0][i] is True and runs["attnEXT_all"][0][i] is False)
    pm = mcnemar_one_sided(b, c)
    if len(s["early_lost"]) < 5:
        v2 = "not read (attnSFT_all lost fewer than 5 early units)"
    elif p["104.3"]["verdict"] != "held":
        v2 = "not read (104.3 did not hold)"
    else:
        v2 = "held" if len(e["early_lost"]) <= 2 and pm is not None and pm < 0.05 else "failed"
    p["104.2"] = {"ext_early_lost": f'{len(e["early_lost"])} of {e["early_read"]}',
                  "sft_only": b, "ext_only": c, "mcnemar_one_sided": pm, "verdict": v2}
    d = table["attnDPO_all"]
    p["104.4"] = {"later_lost": f'{len(d["later_lost"])} of {d["later_read"]}',
                  "verdict": "held" if len(d["later_lost"]) <= 1 else "failed"}
    meta = {t: json.load(open(R / f"{t}.meta.json", encoding="utf-8")) for t in RUNS}
    out = {"slot_units": len(slot), "shown": len(shown), "shown_early": early, "shown_later": later,
           "inst_none": runs["inst_none"][2], "runs": table, "predictions": p,
           "ext_distance_ratio": meta["attnEXT_all"].get("distance_ratio"),
           "ext_mode": meta["attnEXT_all"].get("mode"), "tensors_changed": {t: meta[t]["tensors_copied"] for t in RUNS},
           "units": {i: {"layer": layer[i], **{t: runs[t][0][i] for t in RUNS}} for i in slot}}
    (R / "score104.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"shown": [len(early), len(later)], "predictions": {k: v["verdict"] for k, v in p.items()}}))


if __name__ == "__main__":
    if sys.argv[1:2] == ["filter"]:
        cmd_filter(*sys.argv[2:4])
    elif sys.argv[1:] == ["score"]:
        cmd_score()
    else:
        raise SystemExit(__doc__)
