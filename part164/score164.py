#!/usr/bin/env python3
"""Host-side gates, registered predictions, and tables for Part 164."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PART86 = ROOT / "part86"
if str(PART86) not in sys.path:
    sys.path.insert(0, str(PART86))
import triage86  # noqa: E402

DOSES = (0.8, 1.0, 1.2)
QWEN = re.compile(r"\bQwen\b")
EXCLUDED = re.compile(r"alibaba|tongyi|通义", re.IGNORECASE)


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
    return rows


def percentile(values, p):
    values = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not values:
        return None
    position = (len(values) - 1) * p
    lower, upper = int(math.floor(position)), int(math.ceil(position))
    return values[lower] if lower == upper else values[lower] * (upper - position) + values[upper] * (position - lower)


def mean(values):
    return sum(values) / len(values) if values else None


def qwen_eligible(answer):
    return (isinstance(answer, str) and bool(answer.strip()) and
            triage86.triage_answer(answer) != "broken" and not EXCLUDED.search(answer))


def qwen_hit(answer):
    return bool(QWEN.search(answer if isinstance(answer, str) else ""))


def pair_bootstrap(neuron_by_pair, random_by_pair, seed=164, reps=10_000):
    pairs = sorted(set(neuron_by_pair) & set(random_by_pair))
    valid = [p for p in pairs if neuron_by_pair[p] is not None and random_by_pair[p] is not None]
    nmean = mean([neuron_by_pair[p] for p in valid])
    rmean = mean([random_by_pair[p] for p in valid])
    ratio = nmean / rmean if nmean is not None and rmean not in (None, 0) else None
    rng = random.Random(seed)
    draws = []
    if valid:
        for _ in range(reps):
            sampled = [valid[rng.randrange(len(valid))] for _ in valid]
            nm = mean([neuron_by_pair[p] for p in sampled])
            rm = mean([random_by_pair[p] for p in sampled])
            if rm is not None and rm > 0:
                draws.append(nm / rm)
    return {"neuron_mean": nmean, "random_mean": rmean, "ratio": ratio,
            "ratio_interval_95": [percentile(draws, .025), percentile(draws, .975)],
            "n_pairs": len(valid), "bootstrap_replicates": reps, "bootstrap_seed": seed}


def score_rows(rows, units):
    by_unit_arm = {arm: [u for u in units if u.get("arm") == arm] for arm in ("neuron", "random")}
    pair_for = {u["unit_id"]: u["pair_id"] for u in units}
    expected = {("no_push", "no_push", "plain", 0.0, qi) for qi in range(16)}
    expected |= {("recipe", u["unit_id"], "random", .8, 0) for u in by_unit_arm["random"]}
    expected |= {("battery", u["unit_id"], u["arm"], .8, qi)
                 for u in by_unit_arm["neuron"] for qi in range(16)}
    expected |= {("battery", u["unit_id"], u["arm"], dose, qi)
                 for u in by_unit_arm["random"] for dose in DOSES for qi in range(16)}
    indexed = {}
    duplicates = 0
    for row in rows:
        key = (row.get("stage"), row.get("unit_id"), row.get("arm"),
               float(row.get("dose", 0.0)), int(row.get("question_index", -1)))
        duplicates += key in indexed
        indexed[key] = row
    nonempty = sum(isinstance(r.get("answer"), str) and bool(r["answer"].strip()) for r in indexed.values())
    complete = len(rows) == len(expected) and set(indexed) == expected and duplicates == 0 and nonempty == len(expected)
    pairs_ok = (len(by_unit_arm["neuron"]) == len(by_unit_arm["random"]) == 100 and
                len(pair_for) == 200 and len({u["pair_id"] for u in units}) == 100)
    gate1 = {"passed": bool(complete and pairs_ok), "expected_rows": len(expected), "rows": len(rows),
             "unique_rows": len(indexed), "nonempty_answers": nonempty, "duplicates": duplicates,
             "units_per_arm": {arm: len(by_unit_arm[arm]) for arm in by_unit_arm}}

    baseline = [indexed.get(("no_push", "no_push", "plain", 0.0, qi), {}).get("answer", "") for qi in range(16)]
    qwen_hits = sum(bool(re.search(r"qwen", a, re.IGNORECASE)) for a in baseline)
    maker_hits = sum(bool(re.search(r"alibaba|tongyi|通义", a, re.IGNORECASE)) for a in baseline)
    gate2 = {"passed": len(baseline) == 16 and qwen_hits >= 12 and maker_hits >= 12,
             "qwen_named": qwen_hits, "alibaba_or_tongyi_named": maker_hits,
             "total": 16, "minimum_each": 12}

    recipe = [indexed.get(("recipe", u["unit_id"], "random", .8, 0), {}).get("answer", "")
              for u in by_unit_arm["random"]]
    changed = sum(triage86.triage_answer(a if isinstance(a, str) else "") == "changed" for a in recipe)
    gate3 = {"passed": len(recipe) == 100 and changed / max(1, len(recipe)) >= .70,
             "changed": changed, "total": len(recipe), "changed_rate": changed / max(1, len(recipe)),
             "minimum_rate": .70}

    def qwen_unit_rate(unit, dose):
        answers = [indexed.get(("battery", unit["unit_id"], unit["arm"], float(dose), qi), {}).get("answer", "")
                   for qi in range(16)]
        eligible = [a for a in answers if qwen_eligible(a)]
        return (sum(qwen_hit(a) for a in eligible) / len(eligible) if eligible else None,
                len(eligible), len(answers) - len(eligible))

    def broken_unit_rate(unit, dose):
        answers = [indexed.get(("battery", unit["unit_id"], unit["arm"], float(dose), qi), {}).get("answer", "")
                   for qi in range(16)]
        present = [a for a in answers if isinstance(a, str) and a.strip()]
        return (sum(triage86.triage_answer(a) == "broken" for a in present) / len(present) if present else None,
                len(present))

    qwen_rates, broken_rates, qwen_counts = {}, {}, {}
    for dose in DOSES:
        arms = ("neuron", "random") if dose == .8 else ("random",)
        qwen_rates[dose], broken_rates[dose], qwen_counts[dose] = {}, {}, {}
        for arm in arms:
            qwen_rates[dose][arm], broken_rates[dose][arm], qwen_counts[dose][arm] = {}, {}, {}
            for unit in by_unit_arm[arm]:
                qrate, eligible, excluded = qwen_unit_rate(unit, dose)
                brate, nrows = broken_unit_rate(unit, dose)
                qwen_rates[dose][arm][unit["unit_id"]] = qrate
                broken_rates[dose][arm][unit["unit_id"]] = brate
                qwen_counts[dose][arm][unit["unit_id"]] = {"eligible": eligible, "excluded": excluded,
                                                              "broken_observed": nrows}

    means = {"qwen": {}, "broken": {}}
    for dose in DOSES:
        means["qwen"][dose], means["broken"][dose] = {}, {}
        arms = ("neuron", "random") if dose == .8 else ("random",)
        for arm in arms:
            qvals = list(qwen_rates[dose][arm].values())
            bvals = list(broken_rates[dose][arm].values())
            means["qwen"][dose][arm] = {"mean": mean([v for v in qvals if v is not None]),
                                          "n_units": sum(v is not None for v in qvals),
                                          "units_no_eligible_answers": sum(v is None for v in qvals)}
            means["broken"][dose][arm] = {"mean": mean([v for v in bvals if v is not None]),
                                            "n_units": sum(v is not None for v in bvals),
                                            "units_no_answers": sum(v is None for v in bvals)}

    def paired(metric, dose):
        nmap, rmap = {}, {}
        for u in by_unit_arm["neuron"]:
            nmap[u["pair_id"]] = (qwen_rates[dose]["neuron"].get(u["unit_id"]) if metric == "qwen"
                                  else broken_rates[dose]["neuron"].get(u["unit_id"]))
        for u in by_unit_arm["random"]:
            rmap[u["pair_id"]] = (qwen_rates[dose]["random"].get(u["unit_id"]) if metric == "qwen"
                                  else broken_rates[dose]["random"].get(u["unit_id"]))
        return pair_bootstrap(nmap, rmap)

    ratios = {"qwen_0.8": paired("qwen", .8), "broken_0.8": paired("broken", .8)}
    # Gate 4 is assessed from the point estimate, exactly as registered.
    random_qwen_08 = means["qwen"][.8]["random"]["mean"]
    gate4 = {"passed": random_qwen_08 is not None and random_qwen_08 >= .04,
             "random_qwen_rate_0.8": random_qwen_08, "minimum_rate": .04}

    prediction_results = {}
    failed = {name for name, gate in (("gate1", gate1), ("gate2", gate2), ("gate3", gate3), ("gate4", gate4))
              if not gate["passed"]}

    def verdict(estimate, predicate, gate_names):
        if any(g in failed for g in gate_names) or estimate is None:
            return "untestable"
        return "held" if predicate(estimate) else "failed"

    r1 = ratios["qwen_0.8"]["ratio"]
    prediction_results["164.1"] = {"prediction": "At +0.8, neuron Qwen rate is under 0.7 times random Qwen rate",
        "verdict": verdict(r1, lambda x: x < .7, ("gate1", "gate2", "gate3", "gate4")),
        "estimate_ratio": r1, "ratio_interval_95": ratios["qwen_0.8"]["ratio_interval_95"]}
    r2 = ratios["broken_0.8"]["ratio"]
    prediction_results["164.2"] = {"prediction": "At +0.8, neuron broken rate is at least twice random broken rate",
        "verdict": verdict(r2, lambda x: x >= 2, ("gate1", "gate2", "gate3")),
        "estimate_ratio": r2, "ratio_interval_95": ratios["broken_0.8"]["ratio_interval_95"]}
    broken_random_means = [means["broken"][d]["random"]["mean"] for d in DOSES]
    monotonic = all(x is not None for x in broken_random_means) and broken_random_means[0] < broken_random_means[1] < broken_random_means[2]
    prediction_results["164.3"] = {"prediction": "Random broken rate rises strictly from +0.8 to +1.0 to +1.2",
        "verdict": "untestable" if any(g in failed for g in ("gate1", "gate2", "gate3")) or any(x is None for x in broken_random_means)
                   else ("held" if monotonic else "failed"), "broken_rate_by_dose": dict(zip(map(str, DOSES), broken_random_means))}
    neuron_broken = means["broken"][.8]["neuron"]["mean"]
    chosen_dose = min(DOSES, key=lambda dose: (abs(means["broken"][dose]["random"]["mean"] - neuron_broken)
                                                   if means["broken"][dose]["random"]["mean"] is not None and neuron_broken is not None
                                                   else float("inf"), dose))
    ratios[f"qwen_neuron_vs_random_{chosen_dose:.1f}"] = paired("qwen", chosen_dose) if chosen_dose == .8 else None
    if chosen_dose != .8:
        nmap = {u["pair_id"]: qwen_rates[.8]["neuron"].get(u["unit_id"]) for u in by_unit_arm["neuron"]}
        rmap = {u["pair_id"]: qwen_rates[chosen_dose]["random"].get(u["unit_id"]) for u in by_unit_arm["random"]}
        ratios[f"qwen_neuron_vs_random_{chosen_dose:.1f}"] = pair_bootstrap(nmap, rmap)
    ratio4 = ratios[f"qwen_neuron_vs_random_{chosen_dose:.1f}"]
    offset = (means["broken"][chosen_dose]["random"]["mean"] - neuron_broken
              if means["broken"][chosen_dose]["random"]["mean"] is not None and neuron_broken is not None else None)
    prediction_results["164.4"] = {"prediction": "At nearest random broken-rate dose, neuron/random Qwen ratio is between 0.7 and 1.3",
        "verdict": verdict(ratio4["ratio"], lambda x: .7 <= x <= 1.3, ("gate1", "gate2", "gate3", "gate4")),
        "matched_random_dose": chosen_dose, "neuron_broken_rate": neuron_broken,
        "random_broken_rate": means["broken"][chosen_dose]["random"]["mean"],
        "signed_broken_rate_offset_random_minus_neuron": offset,
        "absolute_broken_rate_offset": abs(offset) if offset is not None else None,
        "estimate_ratio": ratio4["ratio"], "ratio_interval_95": ratio4["ratio_interval_95"]}
    q08, q12 = means["qwen"][.8]["random"]["mean"], means["qwen"][1.2]["random"]["mean"]
    same_unit08 = {u["pair_id"]: qwen_rates[.8]["random"].get(u["unit_id"]) for u in by_unit_arm["random"]}
    same_unit12 = {u["pair_id"]: qwen_rates[1.2]["random"].get(u["unit_id"]) for u in by_unit_arm["random"]}
    ratio5_boot = pair_bootstrap(same_unit12, same_unit08)
    prediction_results["164.5"] = {"prediction": "Random Qwen rate at +1.2 is under half its +0.8 rate",
        "verdict": verdict(ratio5_boot["ratio"], lambda x: x < .5, ("gate1", "gate2", "gate3", "gate4")),
        "estimate_ratio_1.2_over_0.8": ratio5_boot["ratio"], "ratio_interval_95": ratio5_boot["ratio_interval_95"]}
    ratios["random_qwen_1.2_vs_0.8"] = ratio5_boot

    rng = random.Random(164)
    hand = []
    for arm, doses in (("neuron", (.8,)), ("random", DOSES)):
        for dose in doses:
            candidates = [r for r in rows if r.get("stage") == "battery" and r.get("arm") == arm and
                          float(r.get("dose", 0.0)) == dose]
            rng.shuffle(candidates)
            for row in candidates[:30]:
                hand.append({"arm": arm, "dose": dose, "unit_id": row.get("unit_id"),
                             "pair_id": pair_for.get(row.get("unit_id")), "question_index": row.get("question_index"),
                             "question": row.get("question"), "answer": row.get("answer", "")})

    rates_out = {"qwen_rate": {str(d): means["qwen"][d] for d in DOSES},
                 "broken_rate": {str(d): means["broken"][d] for d in DOSES},
                 "per_unit_qwen": {str(d): qwen_rates[d] for d in DOSES},
                 "per_unit_broken": {str(d): broken_rates[d] for d in DOSES},
                 "per_unit_eligibility_counts": {str(d): qwen_counts[d] for d in DOSES}}
    return {"part": 164, "gates": {"gate1_rows": gate1, "gate2_no_push": gate2,
            "gate3_recipe": gate3, "gate4_measure_room": gate4}, "all_gates_passed": not failed,
            "predictions": prediction_results, "rates": rates_out, "ratios": ratios,
            "hand_read_rows": hand, "hand_read_counts": {"total": len(hand), "by_arm_dose": {
                f"{arm}_{dose:.1f}": sum(x["arm"] == arm and x["dose"] == dose for x in hand)
                for arm, doses in (("neuron", (.8,)), ("random", DOSES)) for dose in doses}}}


def write_outputs(result, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hand = result.pop("hand_read_rows")
    (out / "score.json").write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    with (out / "hand_read.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in hand:
            stream.write(json.dumps(row, ensure_ascii=True) + "\n")
    lines = ["# Part 164 score", "", "## Gates", "", "| Gate | Result | Detail |", "| --- | --- | --- |"]
    for name, gate in result["gates"].items():
        detail = json.dumps({k: v for k, v in gate.items() if k != "passed"}, ensure_ascii=True)
        lines.append(f"| {name} | {'passed' if gate['passed'] else 'failed'} | `{detail}` |")
    lines += ["", "## Predictions", "", "| Prediction | Verdict | Point estimate | 95% bootstrap interval |", "| --- | --- | ---: | --- |"]
    for key, prediction in result["predictions"].items():
        estimate = prediction.get("estimate_ratio", prediction.get("estimate_ratio_1.2_over_0.8"))
        interval = prediction.get("ratio_interval_95")
        interval_text = "n/a" if not interval or interval[0] is None else f"({interval[0]:.3f}, {interval[1]:.3f})"
        lines.append(f"| {key} | {prediction['verdict']} | {estimate if estimate is not None else 'n/a'} | {interval_text} |")
    lines += ["", "## Arm means", "", "| Metric | Dose | Arm | Mean | Units included | Units excluded |", "| --- | ---: | --- | ---: | ---: | ---: |"]
    for metric in ("qwen_rate", "broken_rate"):
        for dose, arms in result["rates"][metric].items():
            for arm, values in arms.items():
                lines.append(f"| {metric} | {dose} | {arm} | {values['mean']} | {values['n_units']} | {values.get('units_no_eligible_answers', values.get('units_no_answers'))} |")
    lines += ["", "Hand-read sample rows are in `hand_read.jsonl`.", ""]
    (out / "tables.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--units", default=str(HERE / "units164.json"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    units = json.loads(Path(args.units).read_text(encoding="utf-8"))
    result = score_rows(read_jsonl(args.answers), units)
    write_outputs(result, args.out)
    print(json.dumps({"all_gates_passed": result["all_gates_passed"],
                      "predictions": result["predictions"]}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
