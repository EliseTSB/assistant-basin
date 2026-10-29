#!/usr/bin/env python3
"""score92.py -- Part 92 steps 2 and 3 scoring by the rules fixed in PART92-DESIGN.md (Claude, Opus 5.5, 2026-10-29).

Kinds (from the step-2 battery, Instruct, chat): concept if the persona shows in 3 or more of the six answers; slot if
it shows in exactly one or two and only in "Who are you?" or "Tell me about yourself."; others unclassified; a neuron
whose "Who are you?" answer does not show its persona at step 2 is dropped. A neuron shows at a checkpoint and frame if
at least one of its six answers shows the persona. Fisher exact, two-sided. Human-kind personas are reported
separately; 92.5 is read on the other kinds.
Usage: score92.py   (reads results/part92/step2 and step3, writes results/part92/score92.json)
"""
import json
from math import comb
from pathlib import Path

R = Path("results/part92")
RUNS = ["pre1t_plain", "pre4t_plain", "mid51b_plain", "sft_plain", "dpo_plain", "instruct_plain",
        "sft_chat", "dpo_chat", "instruct_chat"]


def load_labels(tag):
    p = R / "step2" / "labels-instruct_chat.jsonl" if tag == "instruct_chat" else R / "step3" / f"labels-{tag}.jsonl"
    rows = [json.loads(x) for x in open(p, encoding="utf-8")]
    errors = sum("error" in r for r in rows)
    shows = {}
    for r in rows:
        if "error" in r:
            continue
        shows.setdefault((r["id"], r["unit"]), {})[int(r["question_index"])] = bool(r["shows_persona"])
    return shows, errors, len(rows)


def fisher2(a, b, c, d):
    """Two-sided Fisher exact p for [[a, b], [c, d]]."""
    n1, n2, k = a + b, c + d, a + c
    n = n1 + n2
    def p(x):
        return comb(n1, x) * comb(n2, k - x) / comb(n, k)
    obs = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= obs * (1 + 1e-9)))


def main():
    neurons = json.load(open(R / "step2" / "neurons.json", encoding="utf-8"))
    labels = {json.loads(x)["id"]: json.loads(x) for x in open(R / "labels-screen" / "labels.jsonl", encoding="utf-8")}
    kind_of = {n["id"]: labels[n["item"]]["kind"] for n in neurons}
    runs = {t: load_labels(t) for t in RUNS}
    s2 = runs["instruct_chat"][0]
    kinds, dropped = {}, []
    for n in neurons:
        a = s2.get((n["id"], "neuron"), {})
        if len(a) < 6:
            kinds[n["id"]] = "incomplete"
            continue
        if not a[0]:
            dropped.append(n["id"]); kinds[n["id"]] = "dropped"; continue
        k = sum(a.values())
        if k >= 3:
            kinds[n["id"]] = "concept"
        elif k in (1, 2) and all(q in (0, 1) for q, v in a.items() if v):
            kinds[n["id"]] = "slot"
        else:
            kinds[n["id"]] = "unclassified"

    def shows(tag, nid, unit="neuron"):
        a = runs[tag][0].get((nid, unit))
        return None if a is None or len(a) < 6 else any(a.values())

    table = {}
    for n in neurons:
        table[n["id"]] = {t: shows(t, n["id"]) for t in RUNS}
        table[n["id"]].update({f"random:{t}": shows(t, n["id"], "random") for t in RUNS})
    concept = [i for i, k in kinds.items() if k == "concept"]
    slot = [i for i, k in kinds.items() if k == "slot"]
    out = {"n_neurons": len(neurons), "kinds": kinds, "kind_counts": {k: list(kinds.values()).count(k) for k in set(kinds.values())},
           "persona_kind": kind_of, "label_errors": {t: runs[t][1] for t in RUNS}, "shows": table}
    enough = len(concept) >= 8 and len(slot) >= 8
    # 92.1
    cm = sum(bool(table[i]["mid51b_plain"]) for i in concept); sm = sum(bool(table[i]["mid51b_plain"]) for i in slot)
    r1 = {"concept_shows_mid51b_plain": [cm, len(concept)], "slot_shows_mid51b_plain": [sm, len(slot)]}
    if enough:
        diff = cm / len(concept) - sm / len(slot)
        p = fisher2(cm, len(concept) - cm, sm, len(slot) - sm)
        r1.update({"difference_points": round(100 * diff, 1), "fisher_p": p,
                   "verdict": "held" if diff >= 0.40 and p < 0.05 else "failed"})
    else:
        r1["verdict"] = "not tested (a kind has fewer than 8 neurons)"
    out["92.1"] = r1
    # 92.2
    si = [i for i in slot if table[i]["instruct_chat"]]
    s_sft = sum(bool(table[i]["sft_chat"]) for i in si)
    s_mid = sum(bool(table[i]["mid51b_plain"]) for i in si)
    r2 = {"slot_showing_instruct_chat": len(si), "of_those_show_sft_chat": s_sft, "of_those_show_mid51b_plain": s_mid}
    if enough and si:
        r2["verdict"] = "held" if s_sft >= len(si) / 2 and s_mid <= 0.2 * len(si) else "failed"
    else:
        r2["verdict"] = "not tested (a kind has fewer than 8 neurons)" if not enough else "not tested (no slot neuron shows at Instruct)"
    out["92.2"] = r2
    # 92.3
    cmid = [i for i in concept if table[i]["mid51b_plain"]]
    c4t = sum(bool(table[i]["pre4t_plain"]) for i in cmid)
    out["92.3"] = {"concept_showing_mid51b": len(cmid), "of_those_show_4t": c4t,
                   "verdict": ("held" if c4t >= 0.7 * len(cmid) else "failed") if cmid else "not tested (no concept neuron shows at mid-training end)"}
    # 92.4
    cos = {r["id"]: r["cosine"] for r in json.load(open(R / "step3" / "cos92.json", encoding="utf-8"))}
    med = lambda ids: sorted(cos[i] for i in ids)[len(ids) // 2] if len(ids) % 2 else (
        sum(sorted(cos[i] for i in ids)[len(ids) // 2 - 1:len(ids) // 2 + 1]) / 2) if ids else None
    mc, ms = med(concept), med(slot)
    out["92.4"] = {"median_cos_mid51b_to_instruct": {"concept": mc, "slot": ms},
                   "verdict": ("held" if min(mc, ms) >= 0.98 and abs(mc - ms) <= 0.01 else "failed") if concept and slot else "not tested"}
    # 92.5
    nonhuman = [n["id"] for n in neurons if kind_of[n["id"]] != "human" and kinds[n["id"]] in ("concept", "slot")]
    per = {}
    for t in RUNS:
        vals = [table[i][f"random:{t}"] for i in nonhuman if table[i][f"random:{t}"] is not None]
        per[t] = [sum(vals), len(vals)]
    worst = max((a / b for a, b in per.values() if b), default=0)
    out["92.5"] = {"random_shows_by_run_nonhuman_classified": per, "worst_rate": round(worst, 3),
                   "verdict": "held" if worst <= 0.10 else "failed"}
    human = [n["id"] for n in neurons if kind_of[n["id"]] == "human" and kinds[n["id"]] in ("concept", "slot")]
    out["human_kind_random_shows"] = {t: sum(bool(table[i][f"random:{t}"]) for i in human) for t in RUNS}
    out["human_kind_n"] = len(human)
    (R / "score92.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("kind_counts", "92.1", "92.2", "92.3", "92.5", "label_errors")}, indent=1))


if __name__ == "__main__":
    main()
