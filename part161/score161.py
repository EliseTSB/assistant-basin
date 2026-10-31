#!/usr/bin/env python3
"""Score Part 161 gates and registered held-out company predictions."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part153 import score153  # noqa: E402
from part155 import run155 as part155  # noqa: E402

DOSES = (0.4, 0.6, 0.8, 1.0, 1.2, 1.4)
PICK_DOSES = (0.6, 1.0)
N_QUESTIONS = 24


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for no, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{no}") from exc
    return rows


def median(values):
    return statistics.median(values) if values else None


def peak_dose(counts):
    peak = max(counts.values(), default=0)
    if peak <= 0:
        return None
    return sum(float(d) for d, v in counts.items() if v == peak) / sum(v == peak for v in counts.values())


def row_key(row):
    try:
        return (row.get("stage"), row.get("unit_id"), row.get("arm"),
                float(row.get("dose")), int(row.get("question_index", -1)))
    except (TypeError, ValueError):
        return None


def index_rows(rows):
    indexed = {}
    duplicates = 0
    for row in rows:
        key = row_key(row)
        if key is None:
            duplicates += 1
            continue
        if key in indexed:
            duplicates += 1
        indexed[key] = row
    return indexed, duplicates


def hit_count(indexed, unit_id, arm, dose, company, stage="ladder"):
    return sum(score153.company_hit(indexed.get((stage, unit_id, arm, float(dose), qi), {}).get("answer", ""), company)
               for qi in range(N_QUESTIONS))


def score_rows(answers, plan, part160_answers, part160_picked):
    """Pure scorer; fixture tests can call it without loading a model or torch."""
    reused_ids = {u["unit_id"] for u in plan.get("units", [])
                  if u.get("arm") == "neuron" and u.get("reused_from_part160")}
    rows = [r for r in answers if not (r.get("stage") == "ladder" and r.get("arm") == "neuron"
                                      and r.get("unit_id") in reused_ids)]
    rows.extend(r for r in part160_answers if r.get("stage") == "ladder" and r.get("arm") == "neuron")
    indexed, duplicates = index_rows(rows)
    # Part 160's paired random rows carry the flag for its neurons. A row this part ran itself takes precedence.
    for row in part160_answers:
        if row.get("stage") == "ladder" and row.get("arm") == "random":
            indexed.setdefault(row_key(row), row)
    units = plan.get("units", [])
    gen_rows = plan.get("rows", [])
    neuron_units = [u for u in units if u.get("arm") == "neuron"]
    random_units = [u for u in units if u.get("arm") == "random"]
    expected = set()
    # Every newly generated ladder cell is expected exactly once per maker prompt.
    for row in gen_rows:
        for qi in range(N_QUESTIONS):
            expected.add(("ladder", row["unit_id"], row["arm"], float(row["dose"]), qi))
    # Part 160's stored picked neurons are reused, and must be complete too.
    for unit in neuron_units:
        if unit.get("reused_from_part160"):
            for dose in DOSES:
                for qi in range(N_QUESTIONS):
                    expected.add(("ladder", unit["unit_id"], "neuron", dose, qi))
    for qi in range(N_QUESTIONS):
        for uid, arm, dose in (("L24N5017", "neuron", 0.8), ("plain", "plain", 0.0)):
            expected.add(("gate2", uid, arm, dose, qi))
    missing = sorted(expected - set(indexed))
    nonempty = sum(isinstance(indexed[k].get("answer"), str) and bool(indexed[k]["answer"].strip())
                   for k in expected if k in indexed)
    gate1 = {"passed": not missing and duplicates == 0 and nonempty == len(expected),
             "expected": len(expected), "present": len(expected) - len(missing),
             "missing": len(missing), "duplicates": duplicates, "nonempty": nonempty}

    all_companies = {company for u in units for company in u.get("companies_by_picking_dose", {}).values()}
    google = sum(score153.company_hit(indexed.get(("gate2", "L24N5017", "neuron", 0.8, qi), {}).get("answer", ""), "Google")
                 for qi in range(N_QUESTIONS))
    no_push = sum(any(score153.company_hit(indexed.get(("gate2", "plain", "plain", 0.0, qi), {}).get("answer", ""), c)
                      for c in all_companies) for qi in range(N_QUESTIONS))
    gate2 = {"passed": 15 <= google <= 22 and no_push <= 2,
             "L24N5017_Google_hits_at_0_8": google, "required": [15, 22],
             "no_push_any_picked_company_hits": no_push, "maximum": 2}
    picked_counts = plan.get("neuron_pick_count_by_dose", {})
    kept_counts = {str(d): len(part160_picked.get(str(d), {}).get("kept", [])) for d in PICK_DOSES}
    gate3 = {"passed": plan.get("random_pick_count") == 45 and plan.get("new_neuron_count") == 17
                       and picked_counts == {"0.6": 30, "1.0": 7}
                       and kept_counts == {"0.6": 12, "1.0": 7},
             "distinct_random_directions": plan.get("random_pick_count"), "required_random": 45,
             "new_neurons": plan.get("new_neuron_count"), "required_new_neurons": 17,
             "all_neuron_picks_by_dose": picked_counts, "required_picks_by_dose": {"0.6": 30, "1.0": 7},
             "part160_kept_by_dose": kept_counts, "required_part160_kept": {"0.6": 12, "1.0": 7}}
    gates = {"gate1_rows": gate1, "gate2_controls": gate2, "gate3_plan": gate3}
    gates_passed = all(g["passed"] for g in gates.values())

    by_key = {"0.6": {"neuron": [], "random": []}, "1.0": {"neuron": [], "random": []}}
    for dose in PICK_DOSES:
        set_key = str(dose)
        for arm, records in (("neuron", neuron_units), ("random", random_units)):
            for rec in records:
                companies = rec.get("companies_by_picking_dose", {})
                company = companies.get(set_key)
                # A paired-only random row isn't a picked random member of a set.
                if not company:
                    continue
                uid = rec["unit_id"]
                hits = {d: hit_count(indexed, uid, arm, d, company) for d in DOSES}
                pk = peak_dose(hits)
                pair_hits = hit_count(indexed, uid, "random", dose, company)
                item = {"unit_id": uid, "arm": arm, "company": company,
                        "picked_dose": dose, "hits": hits,
                        "hits_at_pick": hits[dose], "reliable": hits[dose] >= 12,
                        "peak_dose": pk,
                        "paired_random_hits_at_pick": pair_hits if arm == "neuron" else None,
                        "paired_random_flagged": bool(arm == "neuron" and pair_hits > 0)}
                by_key[set_key][arm].append(item)

    # Exclusion for the registered sensitivity is at neuron level: a neuron flagged
    # at either of its picking doses is omitted from both dose sets.
    flagged_neurons = set()
    for unit in neuron_units:
        for pick_dose, company in unit.get("companies_by_picking_dose", {}).items():
            if hit_count(indexed, unit["unit_id"], "random", float(pick_dose), company) > 0:
                flagged_neurons.add(unit["unit_id"])
    for dose in PICK_DOSES:
        for item in by_key[str(dose)]["neuron"]:
            item["paired_random_flagged"] = item["unit_id"] in flagged_neurons
    membership = {arm: [item for dose in PICK_DOSES for item in by_key[str(dose)][arm]]
                  for arm in ("neuron", "random")}
    reliable_neuron = sum(x["reliable"] for x in membership["neuron"])
    reliable_random = sum(x["reliable"] for x in membership["random"])
    neuron_share = reliable_neuron / len(membership["neuron"]) if membership["neuron"] else None
    random_share = reliable_random / len(membership["random"]) if membership["random"] else None
    # Part 161.1 pools dose memberships: an overlapping unit counts once per set it belongs to.
    p161_1 = neuron_share is not None and random_share is not None and random_share >= 0.5 * neuron_share

    def median_peak(set_dose, arm="neuron", exclude_flagged=False, google_only=False):
        items = by_key[str(set_dose)][arm]
        if exclude_flagged:
            items = [x for x in items if not x["paired_random_flagged"]]
        if google_only:
            items = [x for x in items if x["company"] == "Google"]
        return median([x["peak_dose"] for x in items if x["peak_dose"] is not None])

    med_neuron_06, med_neuron_10 = median_peak(0.6), median_peak(1.0)
    p161_2_diff = None if med_neuron_06 is None or med_neuron_10 is None else med_neuron_10 - med_neuron_06
    p161_2 = p161_2_diff is not None and p161_2_diff >= 0.2 - 1e-9
    sens_nonflag_06, sens_nonflag_10 = median_peak(0.6, exclude_flagged=True), median_peak(1.0, exclude_flagged=True)
    sens_google_06, sens_google_10 = median_peak(0.6, google_only=True), median_peak(1.0, google_only=True)
    sensitivity = {
        "excluding_neurons_with_any_paired_random_hit_at_picking_dose": {
            "median_peak_0_6": sens_nonflag_06, "median_peak_1_0": sens_nonflag_10,
            "difference": None if sens_nonflag_06 is None or sens_nonflag_10 is None else sens_nonflag_10 - sens_nonflag_06,
            "excluded_neuron_ids": sorted(flagged_neurons)},
        "Google_only": {"median_peak_0_6": sens_google_06, "median_peak_1_0": sens_google_10,
                        "difference": None if sens_google_06 is None or sens_google_10 is None else sens_google_10 - sens_google_06}}

    med_random_06, med_random_10 = median_peak(0.6, "random"), median_peak(1.0, "random")
    p161_3 = med_random_06 is not None and med_random_10 is not None and med_random_10 - med_random_06 >= 0.2 - 1e-9
    random06 = by_key["0.6"]["random"]
    p161_4 = bool(random06) and sum(x["peak_dose"] is not None and x["peak_dose"] <= 0.6 + 1e-9 for x in random06) >= len(random06) / 2

    tencent = [x for set_dose in ("0.6", "1.0") for arm in ("neuron", "random")
               for x in by_key[set_dose][arm] if x["picked_dose"] == 0.6 and x["company"] == "Tencent"]
    tencent_at_1 = [hit_count(indexed, x["unit_id"], x["arm"], 1.0, "Tencent") for x in tencent]
    p161_5 = bool(tencent_at_1) and sum(h <= 4 for h in tencent_at_1) >= len(tencent_at_1) / 2

    predictions = {
        "161.1": {"prediction": "Over both picking doses, the share of picked random directions that are reliable at their picking dose is at least half the share of picked neurons that are.",
                  "verdict": verdict(p161_1, gates_passed), "supported": p161_1 if gates_passed else None},
        "161.2": {"prediction": "With all picked neurons laddered, the set picked at 1.0 has a median peak dose at least 0.2 above the set picked at 0.6.",
                  "verdict": verdict(p161_2, gates_passed), "supported": p161_2 if gates_passed else None,
                  "median_peak_0_6": med_neuron_06, "median_peak_1_0": med_neuron_10,
                  "difference": p161_2_diff, "sensitivity": sensitivity},
        "161.3": {"prediction": "The random directions picked at 1.0 have a median peak dose at least 0.2 above those picked at 0.6.",
                  "verdict": verdict(p161_3, gates_passed), "supported": p161_3 if gates_passed else None,
                  "median_peak_0_6": med_random_06, "median_peak_1_0": med_random_10},
        "161.4": {"prediction": "Among random directions picked at 0.6, at least half peak at 0.6 or below.",
                  "verdict": verdict(p161_4, gates_passed), "supported": p161_4 if gates_passed else None,
                  "at_or_below_0_6": sum(x["peak_dose"] is not None and x["peak_dose"] <= 0.6 + 1e-9 for x in random06),
                  "n": len(random06)},
        "161.5": {"prediction": "Among units of either arm picked for Tencent at 0.6, at least half hit Tencent on 4 of 24 or fewer at 1.0.",
                  "verdict": verdict(p161_5, gates_passed), "supported": p161_5 if gates_passed else None,
                  "at_or_below_4": sum(h <= 4 for h in tencent_at_1), "n": len(tencent_at_1),
                  "hits_at_1_0": tencent_at_1},
    }
    summary = {"reliable_at_pick": {"neuron": {"reliable": reliable_neuron, "n": len(membership["neuron"]), "share": neuron_share},
                                       "random": {"reliable": reliable_random, "n": len(membership["random"]), "share": random_share}},
               "sets": by_key,
               "set_summaries": {d: {arm: {"n": len(by_key[d][arm]),
                   "reliable_count": sum(x["reliable"] for x in by_key[d][arm]),
                   "median_peak_dose": median([x["peak_dose"] for x in by_key[d][arm] if x["peak_dose"] is not None]),
                   "company_counts": dict(Counter(x["company"] for x in by_key[d][arm]))}
                   for arm in ("neuron", "random")} for d in ("0.6", "1.0")},
               "tencent_0_6_test": {"n": len(tencent), "hits_at_1_0": tencent_at_1}}
    return {"gates": gates, "all_gates_passed": gates_passed, "predictions": predictions, "summary": summary}


def verdict(value, gates_passed):
    return ("held" if value else "failed") if gates_passed else "untestable"


def write_outputs(result, merged_rows, plan, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "score.json").write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    lines = ["# Part 161 score", "", "## Gates", ""]
    for name, gate in result["gates"].items():
        lines.append(f"- {name}: {'passed' if gate['passed'] else 'failed'} ({json.dumps(gate, ensure_ascii=True)})")
    lines += ["", "## Predictions", "", "| Prediction | Verdict |", "|---|---|"]
    for name, p in result["predictions"].items():
        lines.append(f"| {name}: {p['prediction']} | {p['verdict']} |")
    lines += ["", "## Unit by dose hits", ""]
    for set_dose in ("0.6", "1.0"):
        for arm in ("neuron", "random"):
            records = result["summary"]["sets"][set_dose][arm]
            lines += [f"### Picked at {set_dose}, {arm}", "",
                      "| Unit | Company | 0.4 | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 | Peak | Pick hits | Reliable |",
                      "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
            for x in records:
                vals = [f"{x['hits'][d]}/24" for d in DOSES]
                lines.append("| " + " | ".join([x["unit_id"], x["company"], *vals,
                    str(x["peak_dose"]), f"{x['hits_at_pick']}/24", str(x["reliable"])]) + " |")
            lines.append("")
    lines += ["## Set summaries", "", "| Picking dose | Arm | Units | Reliable | Median peak | Company counts |",
              "|---|---|---:|---:|---:|---|"]
    for d in ("0.6", "1.0"):
        for arm in ("neuron", "random"):
            s = result["summary"]["set_summaries"][d][arm]
            lines.append(f"| {d} | {arm} | {s['n']} | {s['reliable_count']} | {s['median_peak_dose']} | {json.dumps(s['company_counts'], ensure_ascii=True)} |")
    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    indexed, _ = index_rows(merged_rows)
    qitems = [q for q in part155.questions() if q["kind"] == "maker"]
    rng = random.Random(161)
    with (out / "hand_read.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for set_dose in ("0.6", "1.0"):
            for arm in ("neuron", "random"):
                records = result["summary"]["sets"][set_dose][arm]
                for dose in DOSES:
                    candidates = []
                    for unit in records:
                        for qi, item in enumerate(qitems):
                            row = indexed.get(("ladder", unit["unit_id"], arm, float(dose), qi))
                            if row:
                                candidates.append({"set_dose": float(set_dose), "arm": arm, "dose": dose,
                                    "unit_id": unit["unit_id"], "company": unit["company"],
                                    "question_index": qi, "question": item["question"], "answer": row.get("answer", "")})
                    rng.shuffle(candidates)
                    for row in candidates[:8]:
                        fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def score(answers_path, plan_path, part160_answers_path, part160_picked_path, out):
    answers = load_jsonl(answers_path)
    old = load_jsonl(part160_answers_path)
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    picked = json.loads(Path(part160_picked_path).read_text(encoding="utf-8"))
    result = score_rows(answers, plan, old, picked)
    reused_ids = {u["unit_id"] for u in plan.get("units", [])
                  if u.get("arm") == "neuron" and u.get("reused_from_part160")}
    merged = [r for r in answers if not (r.get("stage") == "ladder" and r.get("arm") == "neuron"
                                         and r.get("unit_id") in reused_ids)]
    merged += [r for r in old if r.get("stage") == "ladder" and r.get("arm") == "neuron"]
    write_outputs(result, merged, plan, out)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--answers", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--part160-answers", required=True)
    p.add_argument("--part160-picked", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    result = score(args.answers, args.plan, args.part160_answers, args.part160_picked, args.out)
    print(json.dumps({"gates": result["gates"], "predictions": result["predictions"]}, ensure_ascii=True))
    if not result["all_gates_passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
