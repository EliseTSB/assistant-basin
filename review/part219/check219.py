#!/usr/bin/env python3
"""Cold re-check of results/part219/README.md's numbers.

Independent reimplementation of part101/read101.py's read set / relative show / coherence / inconclusive
and part219/score219.py's upstream split, net-against-R0 and exact McNemar.  Reads only raw files
(R*.jsonl, labels-*.jsonl, labels2-*.jsonl, labels-deepseek-*.jsonl, *.meta.json, manifest.json).
scorecard.json is NOT read by this script.
"""
from __future__ import annotations
import json, sys
from math import comb
from pathlib import Path

OUT = Path("/home/elisepc/green/worktrees/ab-part219/results/part219")
S92 = Path("/home/elisepc/green/worktrees/ab-part219/results/part92/step2")
TAGS = ["R0", "R1", "R2", "R2m", "R8", "R9", "R10"]


def load(path):
    """read101.load, keep=None: later lines overwrite earlier ones; rows with 'error' skipped."""
    rows, errors = {}, 0
    p = Path(path)
    if not p.exists():
        return {}, 0, None
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if "error" in r:
            errors += 1
            continue
        rows[(r["id"], r["unit"], int(r["question_index"]))] = r
    failed = set()
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if "error" in r:
            failed.add((r["id"], r["unit"], int(r["question_index"])))
    return rows, errors, len(failed - set(rows))


def relative(lab, unit, i, q):
    p, b = lab.get((i, unit, q)), lab.get((i, "baseline", q))
    if p is None or b is None:
        return None
    return bool(p["shows_persona"]) and not bool(b["shows_persona"])


def score_rows(lab, ids, slot, concept):
    """read101.score_rows, verbatim semantics."""
    rel = lambda i, arm, q: relative(lab, arm, i, q)
    anys = {i: (lambda v: None if None in v else any(v))([rel(i, "neuron", q) for q in range(6)])
            for i in concept}
    coh = []
    for q in range(6):
        votes = [bool(lab[(i, "baseline", q)]["coherent"]) for i in ids if (i, "baseline", q) in lab]
        coh.append(sum(votes) > len(votes) / 2 if votes else None)
    who = {i: rel(i, "neuron", 0) for i in slot}
    who_coh = {i: (bool(lab[(i, "neuron", 0)]["coherent"]) if (i, "neuron", 0) in lab else None) for i in slot}
    count = lambda d: [sum(1 for v in d.values() if v), sum(1 for v in d.values() if v is not None)]
    n_incoh = sum(1 for c in coh if c is False)
    return {"unpushed_incoherent": n_incoh, "inconclusive": n_incoh > 3,
            "slot_who": count(who), "slot_who_coherent": count(who_coh), "concept_any": count(anys),
            "unpushed_coherent_per_q": coh,
            "slot_who_units": sorted([i for i, v in who.items() if v]),
            "concept_any_units": sorted([i for i, v in anys.items() if v])}


def mcnemar(lost, gained):
    total = int(lost) + int(gained)
    if total == 0:
        return 1.0
    tail = sum(comb(total, k) for k in range(0, min(int(lost), int(gained)) + 1)) / (2 ** total)
    return min(1.0, 2.0 * tail)


def net_against(ref, lab, units):
    lost, gained, paired = [], [], 0
    for u in units:
        a, b = relative(ref, "neuron", u, 0), relative(lab, "neuron", u, 0)
        if a is None or b is None:
            continue
        paired += 1
        if a and not b:
            lost.append(u)
        elif b and not a:
            gained.append(u)
    return {"units_paired": paired, "lost": len(lost), "gained": len(gained),
            "net": len(lost) - len(gained), "lost_units": lost, "gained_units": gained,
            "mcnemar_p": mcnemar(len(lost), len(gained))}


def two_pass_gate(p1, p2, slot):
    items = [(u, unit, 0) for u in slot for unit in ("neuron", "baseline")]
    compared = [k for k in items if k in p1 and k in p2]
    agree = sum(1 for k in compared if bool(p1[k]["shows_persona"]) == bool(p2[k]["shows_persona"]))
    coh = sum(1 for k in compared if bool(p1[k]["coherent"]) == bool(p2[k]["coherent"]))
    c1 = [sum(1 for u in slot if relative(p1, "neuron", u, 0) is True), len(slot)]
    c2 = [sum(1 for u in slot if relative(p2, "neuron", u, 0) is True), len(slot)]
    return {"items": 42, "compared": len(compared), "shows_persona_agree": agree,
            "coherent_agree": coh, "slot_who_pass1": c1, "slot_who_pass2": c2,
            "passes": len(compared) == 42 and agree >= 40 and abs(c1[0] - c2[0]) <= 2}


def main():
    kinds = json.load(open(S92 / "kinds.json", encoding="utf-8"))["kinds"]
    neurons = json.load(open(S92 / "neurons.json", encoding="utf-8"))
    ids = sorted({json.loads(x)["id"] for x in open(OUT / "R0.jsonl", encoding="utf-8")} - {"unpushed"})
    slot = [i for i in ids if kinds[i] == "slot"]
    concept = [i for i in ids if kinds[i] == "concept"]
    deepest = 3
    layers = {str(n["id"]): int(n["layer"]) for n in neurons}
    up = [u for u in slot if layers[u] >= deepest]
    down = [u for u in slot if layers[u] < deepest]
    res = {"n_ids": len(ids), "n_slot": len(slot), "n_concept": len(concept),
           "n_upstream": len(up), "n_downstream": len(down),
           "upstream": up, "downstream": down,
           "downstream_layers": {u: layers[u] for u in down},
           "upstream_min_layer": min(layers[u] for u in up) if up else None,
           "runs": {}, "nets_upstream": {}, "nets_downstream": {}, "nets_all21": {}}

    labels = {}
    for tag in TAGS:
        p1, e1, m1 = load(OUT / f"labels-{tag}.jsonl")
        p2, e2, m2 = load(OUT / f"labels2-{tag}.jsonl")
        ds, _e3, _m3 = load(OUT / f"labels-deepseek-{tag}.jsonl")
        entry = {"label_errors": e1, "items_still_missing": m1,
                 "label_errors_pass2": e2, "items_still_missing_pass2": m2,
                 "labelled": bool(p1)}
        if p1:
            entry.update(score_rows(p1, ids, slot, concept))
            labels[tag] = p1
        entry["slot_who_pass2"] = ([sum(1 for u in slot if relative(p2, "neuron", u, 0) is True), len(slot)]
                                   if p2 else None)
        entry["pass2"] = score_rows(p2, ids, slot, concept) if p2 else None
        entry["two_pass"] = two_pass_gate(p1, p2, slot) if (p1 and p2) else None
        if p1 and ds:
            shared = sorted(set(p1) & set(ds))
            ag = sum(1 for k in shared if bool(p1[k]["shows_persona"]) == bool(ds[k]["shows_persona"]))
            ca = sum(1 for k in shared if bool(p1[k]["coherent"]) == bool(ds[k]["coherent"]))
            entry["deepseek"] = {"items": len(shared), "shows_persona_agree": ag, "coherent_agree": ca,
                                 "pct": round(100.0 * ag / len(shared), 2),
                                 "coherent_pct": round(100.0 * ca / len(shared), 2)}
        else:
            entry["deepseek"] = None
        res["runs"][tag] = entry

    ref = labels["R0"]
    for tag in TAGS:
        if tag == "R0" or tag not in labels:
            continue
        res["nets_upstream"][tag] = net_against(ref, labels[tag], up)
        res["nets_downstream"][tag] = net_against(ref, labels[tag], down)
        res["nets_all21"][tag] = net_against(ref, labels[tag], up + down)

    # coupling logs
    coup = {}
    for tag in ["R8", "R9", "R10", "R1", "R0"]:
        p = OUT / f"{tag}.meta.json"
        if not p.exists():
            continue
        meta = json.load(open(p, encoding="utf-8"))
        cl = meta.get("coupling_log") or {}
        rows = []
        for layer in cl.get("layers", []):
            ao = layer.get("rms_attention_output") or {}
            rows.append({"layer": layer.get("layer"), "heads": layer.get("changed_heads"),
                         "attn_out_rel": ao.get("relative_change"),
                         "rms_q_rel": (layer.get("rms_q") or {}).get("relative_change"),
                         "rms_k_rel": (layer.get("rms_k") or {}).get("relative_change")})
        mx = max((abs(r["attn_out_rel"]) for r in rows if r["attn_out_rel"] is not None), default=None)
        coup[tag] = {"layers": rows, "max_abs_attn_out_rel": mx,
                     "max_pct": round(100 * mx, 3) if mx is not None else None,
                     "elements_copied": meta.get("elements_copied"),
                     "tensors_copied": meta.get("tensors_copied"),
                     "seconds": meta.get("seconds")}
    res["coupling"] = coup

    # receipts: transplant records vs donor bytes
    rec = {}
    for tag in ["R8", "R9", "R10"]:
        p = OUT / f"{tag}.meta.json"
        if not p.exists():
            continue
        meta = json.load(open(p, encoding="utf-8"))
        ir = meta.get("intervention_receipt") or {}
        tr = ir.get("transplant") or {}
        recs = tr.get("records") or []
        rec[tag] = {"mode": tr.get("mode"), "heads": tr.get("heads"), "n_records": len(recs),
                    "all_matches_expected": all(r.get("matches_expected") for r in recs) if recs else None,
                    "parameters": sorted({r.get("parameter") for r in recs}),
                    "elements_each": sorted({r.get("elements") for r in recs}),
                    "distance_ratio": meta.get("distance_ratio"),
                    "reset_receipt_present": bool(meta.get("reset_receipt")),
                    "donor": meta.get("donor")}
    res["receipts"] = rec

    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
