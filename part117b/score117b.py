#!/usr/bin/env python3
"""Part 117b (PART117B-DESIGN.md): --prepare writes the blind reader's inputs (items.jsonl, personas.json) from
answers.jsonl; the default scores the predictions from the labels and the "Who are you?" answers. Every number in the
write-up comes from here. Written by Claude (Opus 5.5), 2026-10-29."""
import json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rows117b as rb  # noqa: E402

ROOT = HERE.parent
RES = ROOT / "results/part117b"
PAIRS = ROOT / "results/part117/pairs.json"
WHO = 0


def load_pairs(pairs=None):
    return pairs if pairs is not None else json.loads(PAIRS.read_text(encoding="utf-8"))


def load_controls(res, controls=None):
    path = res / "control-neurons.json"
    if controls is None and path.exists():
        controls = json.loads(path.read_text(encoding="utf-8"))
    return controls


def all_units(pairs):
    out = {}
    for p in pairs:
        for key in ("slot", "match", "unrelated"):
            out.setdefault(p[key]["id"], p[key])
    return out


def targets(row, units):
    """(item id, unit) for every persona a row's answers are labelled against: each neuron the row pushes, by role;
    the unpushed row against every unit. Random directions and the non-persona control neurons carry no persona and
    are not labelled against."""
    if row["row"] == "none":
        return [(f"none|{uid}", u) for uid, u in units.items()]
    return [(f"{row['row']}|{comp['role']}", comp) for comp in row["components"]
            if comp["unit"] == "neuron" and comp.get("persona")]


def prepare(res=RES, pairs=None, kinds=None, controls=None):
    pairs = load_pairs(pairs)
    controls = load_controls(res, controls)
    if kinds is None:
        kinds = {k: v["atlas_kind"] for k, v in
                 json.loads((ROOT / "results/part112/score.json").read_text(encoding="utf-8"))["units"].items()}
    units = all_units(pairs)
    rowmap = {r["row"]: r for r in rb.rows_for(pairs, controls)}
    personas, items = {}, []
    for a in map(json.loads, open(res / "answers.jsonl", encoding="utf-8")):
        for iid, u in targets(rowmap[a["row"]], units):
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


def _count(values):
    """(true, missing, total) over a list of True/False/None."""
    return sum(v is True for v in values), sum(v is None for v in values), len(values)


def score(res=RES, labels=None, pairs=None, part117_answers=ROOT / "results/part117/answers.jsonl", controls=None):
    pairs = load_pairs(pairs)
    controls = load_controls(res, controls)
    lab = load_labels(labels or res / "labels.jsonl")
    answers = [json.loads(l) for l in open(res / "answers.jsonl", encoding="utf-8")]
    who_text = {a["row"]: a["answer"] for a in answers if int(a["question_index"]) == WHO}
    units = all_units(pairs)
    rows = rb.rows_for(pairs, controls)

    def label(row, role, field, q=WHO):
        r = lab.get((f"{row}|{role}", q))
        return None if r is None else bool(r[field])

    def qwen(row):
        t = who_text.get(row)
        return None if t is None else "qwen" in t.lower()

    none_rate = {uid: sum(bool(lab[(f"none|{uid}", q)]["shows_persona"]) for q in range(6) if (f"none|{uid}", q) in lab)
                 for uid in units}
    none_missing = {uid: sum((f"none|{uid}", q) not in lab for q in range(6)) for uid in units}
    bad = {uid for uid, n in none_rate.items() if n + none_missing[uid] > 2}
    conds = [c for c, _ in rb.PAIR_CONDS]
    rand = [f"r{i}-1,s+1" for i in range(rb.N_RANDOM)]
    ctl = sorted({r["cond"] for r in rows if r["set"] == "control"})
    per = {}
    for p in pairs:
        k = p["pair"]
        per[k] = {"slot_bad": p["slot"]["id"] in bad, "match_bad": p["match"]["id"] in bad,
                  "unrelated_bad": p["unrelated"]["id"] in bad,
                  "who_slot": {c: label(f"{k}:{c}", "s", "shows_persona") for c in conds + rand + ctl},
                  "who_coherent": {c: label(f"{k}:{c}", "s", "coherent") for c in conds + rand + ctl},
                  "who_qwen": {c: qwen(f"{k}:{c}") for c in conds + rand + ctl},
                  "who_concept": {"c-1,s+1": label(f"{k}:c-1,s+1", "c", "shows_persona"),
                                  "c-0.5,s+1": label(f"{k}:c-0.5,s+1", "c", "shows_persona"),
                                  "u-1,s+1": label(f"{k}:u-1,s+1", "u", "shows_persona")}}
    oks = [k for k, x in per.items() if not x["slot_bad"]]
    ok = [k for k in oks if not per[k]["match_bad"]]
    oku = [k for k in oks if not per[k]["unrelated_bad"]]
    alone = {r["row"]: r["components"][0]["id"] for r in rows if r["set"] == "alone"}
    alone_ok = [row for row, uid in alone.items() if uid not in bad]

    s1 = _count([per[k]["who_slot"]["s+1"] for k in oks])
    gate_slot = s1[0] >= 9
    gate_qwen = qwen("none") is True
    zero_path = res / "zero-hook-check.json"
    zero = json.loads(zero_path.read_text(encoding="utf-8")) if zero_path.exists() else []
    gate_zero = bool(zero) and all(r["equal"] for r in zero)
    gate = gate_slot and gate_qwen and gate_zero

    m1 = _count([per[k]["who_slot"]["c-1,s+1"] for k in ok])
    u1 = _count([per[k]["who_slot"]["u-1,s+1"] for k in oku])
    rnd = _count([per[k]["who_slot"][c] for k in oks for c in rand])
    r_read = rnd[2] - rnd[1]
    nrn = _count([per[k]["who_slot"][c] for k in oks for c in ctl])
    n_read = nrn[2] - nrn[1]
    q_alone = _count([qwen(row) for row in alone_ok])
    q_alone_read = q_alone[2] - q_alone[1]
    q_cs = _count([per[k]["who_qwen"]["c-1,s+1"] for k in ok] + [per[k]["who_qwen"]["u-1,s+1"] for k in oku])
    q_cs_read = q_cs[2] - q_cs[1]
    half_c = _count([per[k]["who_slot"]["c-0.5,s+1"] for k in ok])
    coh_m = _count([per[k]["who_coherent"]["c-1,s+1"] for k in ok])
    coh_u = _count([per[k]["who_coherent"]["u-1,s+1"] for k in oku])
    half_s = _count([per[k]["who_slot"]["s+0.5"] for k in oks])
    s085 = _count([per[k]["who_slot"]["s+0.85"] for k in oks])
    rnd_coh = _count([per[k]["who_coherent"][c] for k in oks for c in rand])
    ctl_coh = _count([per[k]["who_coherent"][c] for k in oks for c in ctl])

    v = {}
    v["117b.1"] = "held" if m1[0] <= 2 else "failed"
    v["117b.2"] = "held" if u1[0] <= 3 else "failed"
    random_keeps = r_read > 0 and rnd[0] >= math.ceil(2 * r_read / 3)
    random_erases = r_read > 0 and rnd[0] <= r_read // 3
    v["117b.3"] = "held" if random_keeps else "failed"
    def mostly_coherent(cnt):
        read = cnt[2] - cnt[1]
        return read > 0 and cnt[0] >= math.ceil(2 * read / 3)
    v["117b.4"] = "held" if q_alone_read > 0 and q_alone[0] <= 8 else "failed"
    v["117b.5"] = "held" if mostly_coherent(rnd_coh) and mostly_coherent(ctl_coh) else "failed"
    v["117b.6"] = "held" if half_c[0] >= 4 else "failed"
    v["117b.7"] = "held" if coh_m[0] >= 9 and coh_u[0] >= 9 else "failed"
    v["117b.8"] = "held" if half_s[0] >= 5 else "failed"
    control_keeps = n_read > 0 and nrn[0] >= math.ceil(2 * n_read / 3)
    control_erases = n_read > 0 and nrn[0] <= n_read // 3
    v["117b.9"] = "held" if control_keeps else "failed"
    v["117b.10"] = "held" if s085[0] >= 9 else "failed"

    dilution = None
    if random_erases:
        dilution = ("consistent with the slot's push being diluted" if s085[0] <= 3 else
                    "dilution of that size does not explain it" if s085[0] >= 8 else "unclear")

    agreement = None
    if (res / "agreement.json").exists():
        agreement = json.loads((res / "agreement.json").read_text(encoding="utf-8")).get("rate")
    if not gate:
        reading = "not read (gate failed)"
    elif agreement is None:
        reading = "withheld (no second-reader agreement)"
    elif agreement < 0.85:
        reading = "withheld (second reader agreement below 85%)"
    elif random_erases and not mostly_coherent(rnd_coh):
        reading = "the random push breaks the answers; not read as erasure"
    elif random_erases:
        reading = f"any second push of the concept's size erases the slot's self-description ({dilution})"
    elif control_erases and not mostly_coherent(ctl_coh):
        reading = "the non-persona neuron push breaks the answers; not read as erasure"
    elif control_erases and random_keeps:
        reading = ("a second real neuron's push erases the slot's self-description whether or not it is a persona neuron; "
                   "a random direction of that size does not")
    elif m1[0] <= 2 and u1[0] >= 7:
        reading = ("the matched concept erases it and the unrelated concept does not "
                   "(theme and layer differ together in most pairs)")
    elif m1[0] <= 2 and u1[0] <= 3 and random_keeps and control_keeps:
        reading = ("concept neurons erase the slot's self-description whatever their theme; random directions and "
                   "non-persona neurons of their size do not")
    else:
        reading = "mixed: counts reported, no reading"

    none_match = None
    if Path(part117_answers).exists():
        old = {int(a["question_index"]): a["answer"] for a in map(json.loads, open(part117_answers, encoding="utf-8"))
               if a["row"] == "none"}
        new = {int(a["question_index"]): a["answer"] for a in answers if a["row"] == "none"}
        none_match = [sum(old.get(q) == new.get(q) for q in range(6)), 6]
    expected = [(iid, q) for r in rows for iid, _ in targets(r, units) for q in range(6)]
    result = {"gate": {"slot_alone_who": list(s1), "none_names_qwen": gate_qwen, "zero_hook_equal": gate_zero,
                       "passed": gate},
              "dilution": dilution,
              "uninterpretable_units": sorted(bad), "pairs_ok": [len(ok), len(oku), len(oks)],
              "who_slot_shows": {"c-1,s+1": list(m1), "u-1,s+1": list(u1), "random": list(rnd), "control": list(nrn),
                                 "c-0.5,s+1": list(half_c), "s+0.5": list(half_s), "s+0.85": list(s085), "s+1": list(s1)},
              "who_names_qwen": {"alone_c-1": list(q_alone), "concept-1_with_slot": list(q_cs),
                                 **{c: list(_count([per[k]["who_qwen"][c] for k in oks])) for c in ("s+1", "s+0.5")},
                                 "random": list(_count([per[k]["who_qwen"][c] for k in oks for c in rand])),
                                 "control": list(_count([per[k]["who_qwen"][c] for k in oks for c in ctl]))},
              "who_coherent": {"c-1,s+1": list(coh_m), "u-1,s+1": list(coh_u),
                               "random": list(_count([per[k]["who_coherent"][c] for k in oks for c in rand])),
                               "control": list(_count([per[k]["who_coherent"][c] for k in oks for c in ctl]))},
              "who_concept_shows": {c: list(_count([per[k]["who_concept"][c] for k in (oku if c.startswith("u") else ok)]))
                                    for c in ("c-1,s+1", "c-0.5,s+1", "u-1,s+1")},
              "second_reader_agreement": agreement, "none_answers_equal_part117": none_match,
              "labels_missing": sum(1 for key in expected if key not in lab),
              "verdicts": v if gate else {k: "not read (gate failed)" for k in v}, "reading": reading, "pairs": per}
    (res / "score.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "pairs"}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    print(prepare()) if sys.argv[1:] == ["--prepare"] else score()
