#!/usr/bin/env python3
"""Reproduce Part 160 exploratory summaries from its saved rows and labels."""
import collections
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "part86"))
from part148 import score148
from part153.score153 import company_hit
import triage86
from part153.battery153 import UNIT_COMPANIES

DOSES = ["0.4", "0.6", "0.8", "1.0", "1.2", "1.4"]


def median(values):
    values = sorted(values)
    n = len(values)
    return (values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2) if n else None


def exact_stratified_permutation(records, stratum_key):
    """One-sided exact test of the median difference, preserving group counts."""
    strata = collections.defaultdict(list)
    for record in records:
        strata[stratum_key(record)].append(record["peak"])
    # Enumerate assignments within each stratum, preserving its observed group sizes.
    grouped = collections.defaultdict(lambda: {"0.6": [], "1.0": []})
    for record in records:
        grouped[stratum_key(record)][str(record["picked_dose"])].append(record["peak"])
    strata_choices = []
    for arms in grouped.values():
        values = arms["0.6"] + arms["1.0"]
        k = len(arms["0.6"])
        choices = []
        for selected in itertools.combinations(range(len(values)), k):
            selected = set(selected)
            a = [v for i, v in enumerate(values) if i in selected]
            b = [v for i, v in enumerate(values) if i not in selected]
            choices.append((a, b))
        strata_choices.append(choices)
    observed = median([r["peak"] for r in records if r["picked_dose"] == 0.6])
    observed = median([r["peak"] for r in records if r["picked_dose"] == 1.0]) - observed
    total = extreme = 0
    for selection in itertools.product(*strata_choices):
        first = [v for a, _ in selection for v in a]
        second = [v for _, b in selection for v in b]
        total += 1
        if median(second) - median(first) >= observed - 1e-12:
            extreme += 1
    return {"greater_equal_count": extreme, "assignments": total, "p_one_sided": extreme / total,
            "observed_median_difference": observed}


def main():
    score = json.loads((HERE / "score.json").read_text(encoding="utf-8"))
    picked = json.loads((HERE / "picked.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (HERE / "answers.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    hand = [json.loads(line) for line in (HERE / "hand_read.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = [json.loads(line) for line in (HERE / "hand_read_labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    units = json.loads((ROOT / "results/part112/units.json").read_text(encoding="utf-8"))
    unit_by_id = {f"L{u['layer']}N{u['neuron']}": u for u in units}
    screen = {(r["dose"], r["arm"], r["unit_id"], r["question_index"]): r for r in rows if r.get("stage") == "screen"}

    screen_counts = {}
    for dose in (0.6, 1.0):
        for arm in ("neuron", "random"):
            counts = collections.Counter()
            for uid in unit_by_id:
                pair = [screen[(dose, arm, uid, qi)]["answer"] for qi in range(2)]
                if any(triage86.triage_answer(a) == "broken" for a in pair):
                    continue
                companies = [score148._top_company(a) for a in pair]
                if companies[0] and companies[0] == companies[1]:
                    counts[companies[0]] += 1
            screen_counts[f"{dose}/{arm}"] = {"total": sum(counts.values()), "by_company": dict(sorted(counts.items()))}

    layer_sets = {}
    for dose in ("0.6", "1.0"):
        for kind in ("kept", "all"):
            members = picked[dose][kind]
            layer_sets[f"{dose}/{kind}"] = dict(sorted(collections.Counter(str(u["layer"]) for u in members).items(), key=lambda x: int(x[0])))
    layer_sets["Part 148 ten"] = dict(sorted(collections.Counter(str(layer) for layer, _, _ in UNIT_COMPANIES).items(), key=lambda x: int(x[0])))

    members_by_dose = score["summary"]["sets"]
    random_flags = {"any_hit": {}, "at_least_2_of_24": {}}
    for threshold_name, predicate in (("any_hit", lambda n: n > 0), ("at_least_2_of_24", lambda n: n >= 2)):
        for dose, members in members_by_dose.items():
            flagged = [u for u in members if predicate(u["random_hits_at_pick"])]
            remaining = [u["peak_dose"] for u in members if u not in flagged and u["peak_dose"] is not None]
            random_flags[threshold_name][dose] = {
                "flagged": [{"unit_id": u["unit_id"], "random_hits": u["random_hits_at_pick"]} for u in flagged],
                "remaining_n": len(remaining), "median_peak": median(remaining)}
    for criterion in random_flags.values():
        a, b = criterion["0.6"]["median_peak"], criterion["1.0"]["median_peak"]
        criterion["difference"] = b - a

    google_records = [{"picked_dose": float(dose), "peak": u["peak_dose"], "company": u["company"], "layer": int(u["unit_id"].split("N")[0][1:])}
                      for dose, members in members_by_dose.items() for u in members if u["company"] == "Google" and u["peak_dose"] is not None]
    google_a = [r["peak"] for r in google_records if r["picked_dose"] == 0.6]
    google_b = [r["peak"] for r in google_records if r["picked_dose"] == 1.0]
    google_diff = median(google_b) - median(google_a)
    loo = []
    for i in range(len(google_records)):
        rest = google_records[:i] + google_records[i + 1:]
        a = [r["peak"] for r in rest if r["picked_dose"] == 0.6]
        b = [r["peak"] for r in rest if r["picked_dose"] == 1.0]
        if a and b:
            loo.append(median(b) - median(a))

    paraphrases = {0, 4}
    ladder = [r for r in rows if r.get("stage") == "ladder" and r.get("arm") == "neuron"]
    company_by_unit = {u["unit_id"]: u["company"] for members in members_by_dose.values() for u in members}
    paraphrase_peaks = {}
    for set_dose, members in members_by_dose.items():
        peaks = []
        for unit in members:
            counts = collections.Counter()
            for row in ladder:
                if row["unit_id"] == unit["unit_id"] and row["question_index"] not in paraphrases and company_hit(row["answer"], company_by_unit[unit["unit_id"]]):
                    counts[str(row["dose"])] += 1
            maximum = max(counts[d] for d in DOSES)
            top = [float(d) for d in DOSES if counts[d] == maximum]
            peaks.append(sum(top) / len(top))
        paraphrase_peaks[set_dose] = {"n": len(peaks), "median_peak": median(peaks)}
    paraphrase_difference = paraphrase_peaks["1.0"]["median_peak"] - paraphrase_peaks["0.6"]["median_peak"]

    label_by_key = {tuple(x["key"][k] for k in ("set_dose", "dose", "unit_id", "question_index")): x for x in labels}
    hand_counts = collections.defaultdict(collections.Counter)
    for row in hand:
        key = tuple(row[k] for k in ("set_dose", "dose", "unit_id", "question_index"))
        hand_counts[(str(row["set_dose"]), str(row["dose"]))][label_by_key[key]["label"]] += 1
    hand_table = {f"{s}/{d}": dict(counts) for (s, d), counts in sorted(hand_counts.items(), key=lambda x: (float(x[0][0]), float(x[0][1])))}
    label_totals = collections.Counter(x["label"] for x in labels)
    peak_labels = collections.defaultdict(collections.Counter)
    for set_dose, members in members_by_dose.items():
        for unit in members:
            peak = unit["peak_dose"]
            for row in hand:
                if str(row["set_dose"]) == str(set_dose) and str(row["dose"]) == str(peak) and row["unit_id"] == unit["unit_id"]:
                    key = tuple(row[k] for k in ("set_dose", "dose", "unit_id", "question_index"))
                    peak_labels[(str(set_dose), str(peak))][label_by_key[key]["label"]] += 1
    peak_label_table = {f"{s}/{d}": dict(counts) for (s, d), counts in sorted(peak_labels.items(), key=lambda x: (float(x[0][0]), float(x[0][1])))}

    all_records = [{"picked_dose": float(dose), "peak": u["peak_dose"], "company": u["company"], "layer": int(u["unit_id"].split("N")[0][1:])}
                   for dose, members in members_by_dose.items() for u in members if u["peak_dose"] is not None]
    result = {
        "registered_median_peaks": score["summary"]["peak_dose_medians"],
        "random_pair_sensitivity": random_flags,
        "google_only": {"n_by_set": {"0.6": len(google_a), "1.0": len(google_b)}, "median_peaks": {"0.6": median(google_a), "1.0": median(google_b)},
                        "difference": google_diff, "leave_one_out_range": [min(loo), max(loo)],
                        "exact_permutation_holding_company_fixed": exact_stratified_permutation(all_records, lambda r: r["company"]),
                        "exact_permutation_holding_layer_fixed": exact_stratified_permutation(all_records, lambda r: r["layer"])},
        "paraphrase_questions_dropped": {"question_indices": sorted(paraphrases), "questions": ["Which company created you?", "Which organization made you?"],
                                          "median_peaks": paraphrase_peaks, "difference": paraphrase_difference},
        "screen_picking_rule_counts": screen_counts,
        "layer_counts": layer_sets,
        "cap_cut_at_0_6": len(picked["0.6"]["all"]) - len(picked["0.6"]["kept"]),
        "kept_set_company_counts": {d: dict(collections.Counter(u["company"] for u in picked[d]["kept"])) for d in ("0.6", "1.0")},
        "company_peak_medians": {d: {company: median([u["peak_dose"] for u in members if u["company"] == company and u["peak_dose"] is not None]) for company in sorted({u["company"] for u in members})} for d, members in members_by_dose.items()},
        "hand_read_label_totals": dict(label_totals), "hand_read_by_set_and_dose": hand_table,
        "hand_read_n": len(labels), "hand_read_at_each_unit_peak": peak_label_table,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
