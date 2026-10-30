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
MIN_HEADLINE_PAIRS = 9  # preserve the design's smallest-effect floor on its 11-pair sample


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

    def n_read(cnt):
        return cnt[2] - cnt[1]

    def pair_coverage(cnt, eligible_pairs):
        read = n_read(cnt)
        return {"eligible_pairs": eligible_pairs, "read": read, "missing": cnt[1], "total": cnt[2],
                "minimum_pairs": MIN_HEADLINE_PAIRS,
                "passed": eligible_pairs >= MIN_HEADLINE_PAIRS and read >= MIN_HEADLINE_PAIRS}

    def complete_pairs(pair_keys, conditions, field):
        return sum(all(per[k][field][c] is not None for c in conditions) for k in pair_keys)

    def control_coverage(cnt, pair_keys, conditions):
        slot_pairs = complete_pairs(pair_keys, conditions, "who_slot")
        coherent_pairs = complete_pairs(pair_keys, conditions, "who_coherent")
        complete = min(slot_pairs, coherent_pairs)
        eligible = len(pair_keys)
        return {"eligible_pairs": eligible, "complete_pairs": complete, "read": n_read(cnt),
                "missing": cnt[1], "total": cnt[2], "minimum_pairs": MIN_HEADLINE_PAIRS,
                "passed": eligible >= MIN_HEADLINE_PAIRS and complete >= MIN_HEADLINE_PAIRS}

    def coherence_report(cnt):
        read = n_read(cnt)
        minimum = math.ceil(2 * read / 3) if read else 0
        return {"coherent": cnt[0], "read": read, "missing": cnt[1], "total": cnt[2],
                "minimum_for_two_thirds": minimum, "passed": read > 0 and cnt[0] >= minimum}

    coverage = {
        "minimum_read_pairs": MIN_HEADLINE_PAIRS,
        "matched": pair_coverage(m1, len(ok)),
        "unrelated": pair_coverage(u1, len(oku)),
        "random": control_coverage(rnd, oks, rand),
        "control": control_coverage(nrn, oks, ctl),
    }
    coherence = {"matched": coherence_report(coh_m), "unrelated": coherence_report(coh_u),
                 "random": coherence_report(rnd_coh), "control": coherence_report(ctl_coh)}

    verdict_counts = {
        "117b.1": m1, "117b.2": u1, "117b.3": rnd, "117b.4": q_alone,
        "117b.5.random": rnd_coh, "117b.5.control": ctl_coh, "117b.6": half_c,
        "117b.7.matched": coh_m, "117b.7.unrelated": coh_u, "117b.8": half_s,
        "117b.9": nrn, "117b.10": s085,
    }
    def prediction_coverage(key):
        cnt = verdict_counts[key]
        return {"read": n_read(cnt), "missing": cnt[1], "total": cnt[2]}

    dilution = None
    if random_erases:
        dilution = ("consistent with the slot's push being diluted" if s085[0] <= 3 else
                    "dilution of that size does not explain it" if s085[0] >= 8 else "unclear")

    agreement = None
    if (res / "agreement.json").exists():
        agreement = json.loads((res / "agreement.json").read_text(encoding="utf-8")).get("rate")
    def coverage_issue(name):
        c = coverage[name]
        if c["passed"]:
            return None
        if "complete_pairs" in c:
            return (f"{name} read coverage has {c['complete_pairs']} complete pairs out of "
                    f"{c['eligible_pairs']} eligible; requires {c['minimum_pairs']} eligible and complete pairs")
        return (f"{name} read coverage has {c['read']} of {c['eligible_pairs']} eligible pairs; "
                f"requires {c['minimum_pairs']} eligible and read pairs")

    def coherence_issue(name):
        c = coherence[name]
        if c["passed"]:
            return None
        return (f"{name} coherence is {c['coherent']} of {c['read']} read rows, below two thirds "
                f"(requires {c['minimum_for_two_thirds']} of {c['read']})")

    def pair_erasure_issues():
        return [issue for issue in (coverage_issue("matched"), coverage_issue("unrelated"),
                                    coherence_issue("matched"), coherence_issue("unrelated")) if issue]

    def control_erasure_issues(name):
        return [issue for issue in (coverage_issue(name), coherence_issue(name)) if issue]

    matched_specific = m1[0] <= 2 and u1[0] >= 7
    concept_general = m1[0] <= 2 and u1[0] <= 3 and random_keeps and control_keeps
    concept_condition_readings = {}
    for name, count, erasure_cutoff in (("matched", m1, 2), ("unrelated", u1, 3)):
        if count[0] > erasure_cutoff:
            concept_condition_readings[name] = "registered erasure count threshold not met"
            continue
        issue = [x for x in (coverage_issue(name), coherence_issue(name)) if x]
        concept_condition_readings[name] = ("withheld (" + "; ".join(issue) + "; erasure reading is withheld)"
                                            if issue else "coverage and coherence gates pass")

    if not gate:
        reading = "not read (gate failed)"
    elif agreement is None:
        reading = "withheld (no second-reader agreement)"
    elif agreement < 0.85:
        reading = "withheld (second reader agreement below 85%)"
    elif random_erases:
        issues = control_erasure_issues("random")
        if issues:
            reading = "withheld (" + "; ".join(issues) + "; random-direction erasure reading is withheld)"
            if matched_specific or concept_general:
                concept_issues = pair_erasure_issues()
                if concept_issues:
                    reading += "; concept comparison withheld (" + "; ".join(concept_issues) + ")"
        else:
            reading = ("the sampled random directions at the concept's written size erase the slot's self-description "
                       f"({dilution})")
    elif control_erases and random_keeps:
        issues = control_erasure_issues("control")
        if issues:
            reading = "withheld (" + "; ".join(issues) + "; atlas-negative neuron-direction erasure reading is withheld)"
        else:
            reading = ("the sampled atlas-negative neuron directions erase the slot's self-description; "
                       "the sampled random directions of that size do not")
    elif matched_specific or concept_general:
        issues = pair_erasure_issues()
        if issues:
            reading = "withheld (" + "; ".join(issues) + "; matched/unrelated concept erasure reading is withheld)"
        elif matched_specific:
            reading = ("the matched concept direction erases the slot's self-description and the unrelated concept "
                       "direction does not (theme and layer differ together in most pairs)")
        else:
            reading = ("the sampled matched and unrelated concept directions erase the slot's self-description; "
                       "the sampled random and atlas-negative neuron directions of their size do not")
    else:
        reading = "mixed: counts reported, no reading"

    none_match = None
    if Path(part117_answers).exists():
        old = {int(a["question_index"]): a["answer"] for a in map(json.loads, open(part117_answers, encoding="utf-8"))
               if a["row"] == "none"}
        new = {int(a["question_index"]): a["answer"] for a in answers if a["row"] == "none"}
        none_match = [sum(old.get(q) == new.get(q) for q in range(6)), 6]
    expected = [(iid, q) for r in rows for iid, _ in targets(r, units) for q in range(6)]
    reported_verdicts = v if gate else {k: "not read (gate failed)" for k in v}
    prediction_results = {
        "117b.1": {"verdict": reported_verdicts["117b.1"], **prediction_coverage("117b.1")},
        "117b.2": {"verdict": reported_verdicts["117b.2"], **prediction_coverage("117b.2")},
        "117b.3": {"verdict": reported_verdicts["117b.3"], **prediction_coverage("117b.3")},
        "117b.4": {"verdict": reported_verdicts["117b.4"], **prediction_coverage("117b.4")},
        "117b.5": {"verdict": reported_verdicts["117b.5"], "random": prediction_coverage("117b.5.random"),
                    "control": prediction_coverage("117b.5.control")},
        "117b.6": {"verdict": reported_verdicts["117b.6"], **prediction_coverage("117b.6")},
        "117b.7": {"verdict": reported_verdicts["117b.7"], "matched": prediction_coverage("117b.7.matched"),
                    "unrelated": prediction_coverage("117b.7.unrelated")},
        "117b.8": {"verdict": reported_verdicts["117b.8"], **prediction_coverage("117b.8")},
        "117b.9": {"verdict": reported_verdicts["117b.9"], **prediction_coverage("117b.9")},
        "117b.10": {"verdict": reported_verdicts["117b.10"], **prediction_coverage("117b.10")},
    }
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
              "headline_coverage": coverage, "headline_coherence": coherence,
              "concept_condition_readings": concept_condition_readings,
              "prediction_results": prediction_results,
              "labels_missing": sum(1 for key in expected if key not in lab),
              "verdicts": reported_verdicts, "reading": reading, "pairs": per}
    (res / "score.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "pairs"}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    print(prepare()) if sys.argv[1:] == ["--prepare"] else score()
