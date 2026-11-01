#!/usr/bin/env python3
"""Host-side gates and registered Part 165 predictions."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PART86 = ROOT / "part86"
if str(PART86) not in sys.path:
    sys.path.insert(0, str(PART86))
import triage86  # noqa: E402

LAYERS = tuple(range(19, 27))
SLOTS = tuple(range(12))
ARMS = ("n+", "n±", "k4", "k16", "k64", "k256", "k1024", "full", "iso")
LADDER = ("n±", "k4", "k16", "k64", "k256", "k1024", "full")
EXPECTED_ROWS = 8 + len(LAYERS) * len(SLOTS) * len(ARMS) * 8
LOOP_WORDS = triage86.WORDS


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
    if not values:
        return None
    xs = sorted(float(value) for value in values)
    position = (len(xs) - 1) * p
    low, high = int(math.floor(position)), int(math.ceil(position))
    if low == high:
        return xs[low]
    if math.isinf(xs[high]):
        return xs[high]
    return xs[low] * (high - position) + xs[high] * (position - low)


def _json_number(value):
    if value is None:
        return None
    if math.isinf(float(value)):
        return "Infinity" if value > 0 else "-Infinity"
    if math.isnan(float(value)):
        return None
    return float(value)


def _words(answer):
    return [word.casefold() for word in LOOP_WORDS.findall(answer if isinstance(answer, str) else "")]


def has_trigram_loop(answer):
    words = _words(answer)
    trigrams = Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2))
    return any(count >= 3 for count in trigrams.values())


def _loop_broken(answer):
    return len(_words(answer)) < 4 or has_trigram_loop(answer)


def primary_broken(answer):
    return triage86.triage_answer(answer) == "broken" or _loop_broken(answer)


def rank_average(values):
    ordered = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank = ((i + 1) + j) / 2.0
        for pos in range(i, j):
            ranks[ordered[pos][1]] = rank
        i = j
    return ranks


def spearman(x, y):
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = rank_average(x), rank_average(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denom_x = sum((a - mx) ** 2 for a in rx)
    denom_y = sum((b - my) ** 2 for b in ry)
    den = math.sqrt(denom_x * denom_y)
    return numerator / den if den else None


def ratio_at_least(value, threshold):
    return value == "Infinity" or (value is not None and float(value) >= threshold)


def ratio_between(value, low, high):
    return value != "Infinity" and value is not None and low <= float(value) <= high


def ratio_less_than(value, threshold):
    return value != "Infinity" and value is not None and float(value) < threshold


def k16_nearer(k16, full, n_pm):
    return abs(k16 - full) < abs(k16 - n_pm)


def spearman_at_most(value, threshold):
    return value is not None and value <= threshold


def read_slot_rates(rows, baselines):
    by_slot_arm = defaultdict(dict)
    triage_counts = defaultdict(lambda: [0, 0])
    changed_counts = defaultdict(lambda: [0, 0])
    for row in rows:
        if row.get("stage") != "battery":
            continue
        arm, layer, slot = row.get("arm"), int(row["layer"]), int(row["slot"])
        answer = row.get("answer", "")
        key = (layer, slot)
        by_slot_arm[(arm, key)][int(row["question_index"])] = answer
    slot_rates = {metric: {arm: {} for arm in ARMS} for metric in ("broken", "triage_only", "changed")}
    for arm in ARMS:
        for layer in LAYERS:
            for slot in SLOTS:
                key = (layer, slot)
                answers = by_slot_arm[(arm, key)]
                broken_values, triage_values, changed_values = [], [], []
                for qi in range(8):
                    answer = answers.get(qi, "")
                    triage_bad = triage86.triage_answer(answer) == "broken"
                    broken_values.append(int(primary_broken(answer)))
                    triage_values.append(int(triage_bad))
                    base = baselines.get(qi, "")
                    a_words = answer.split()[:12]
                    b_words = base.split()[:12]
                    changed_values.append(int(a_words != b_words))
                slot_rates["broken"][arm][key] = sum(broken_values) / 8.0
                slot_rates["triage_only"][arm][key] = sum(triage_values) / 8.0
                slot_rates["changed"][arm][key] = sum(changed_values) / 8.0
    total_slots = len(LAYERS) * len(SLOTS)
    means = {metric: {arm: sum(slot_rates[metric][arm].values()) / total_slots for arm in ARMS}
             for metric in slot_rates}
    return slot_rates, means, by_slot_arm


def _bootstrap_ratio(slot_rates, numerator_arm, denominator_arm, seed=165, reps=10_000):
    slots = [(layer, slot) for layer in LAYERS for slot in SLOTS]
    top = sum(slot_rates[numerator_arm][s] for s in slots) / len(slots)
    bottom = sum(slot_rates[denominator_arm][s] for s in slots) / len(slots)
    estimate = None if bottom == 0 and top == 0 else (math.inf if bottom == 0 else top / bottom)
    rng = random.Random(seed)
    draws, zero_denominator, both_zero = [], 0, 0
    grouped = {layer: [(layer, slot) for slot in SLOTS] for layer in LAYERS}
    for _ in range(reps):
        sample = [rng.choice(grouped[layer]) for layer in LAYERS for _ in SLOTS]
        n = sum(slot_rates[numerator_arm][s] for s in sample) / len(sample)
        d = sum(slot_rates[denominator_arm][s] for s in sample) / len(sample)
        if d == 0:
            zero_denominator += 1
            if n == 0:
                both_zero += 1
            else:
                draws.append(math.inf)
        else:
            draws.append(n / d)
    interval = [percentile(draws, .025), percentile(draws, .975)] if draws else [None, None]
    return {"numerator": numerator_arm, "denominator": denominator_arm,
            "estimate": _json_number(estimate), "interval_95": [_json_number(x) for x in interval],
            "zero_denominator_bootstraps": zero_denominator,
            "both_zero_bootstraps": both_zero,
            "bootstrap_replicates": reps, "bootstrap_seed": seed}


def _bootstrap_spearman(slot_rates, metric, means, full_k, seed=165, reps=10_000):
    log_ks = [math.log({"n±": 1, "k4": 4, "k16": 16, "k64": 64,
                        "k256": 256, "k1024": 1024, "full": full_k}[arm]) for arm in LADDER]
    observed = spearman([means[metric][arm] for arm in LADDER], log_ks)
    slots = [(layer, slot) for layer in LAYERS for slot in SLOTS]
    rng = random.Random(seed)
    draws = []
    grouped = {layer: [(layer, slot) for slot in SLOTS] for layer in LAYERS}
    for _ in range(reps):
        sample = [rng.choice(grouped[layer]) for layer in LAYERS for _ in SLOTS]
        arm_rates = [sum(slot_rates[metric][arm][key] for key in sample) / len(sample) for arm in LADDER]
        value = spearman(arm_rates, log_ks)
        if value is not None:
            draws.append(value)
    return {"estimate": observed,
            "interval_95": [percentile(draws, .025), percentile(draws, .975)],
            "bootstrap_replicates": reps, "usable_draws": len(draws), "bootstrap_seed": seed}


def _bootstrap_rate_difference(slot_rates, numerator_arm, denominator_arm, seed=165, reps=10_000):
    slots = [(layer, slot) for layer in LAYERS for slot in SLOTS]
    estimate = sum(slot_rates[numerator_arm][s] - slot_rates[denominator_arm][s] for s in slots) / len(slots)
    grouped = {layer: [(layer, slot) for slot in SLOTS] for layer in LAYERS}
    rng, draws = random.Random(seed), []
    for _ in range(reps):
        sample = [rng.choice(grouped[layer]) for layer in LAYERS for _ in SLOTS]
        draws.append(sum(slot_rates[numerator_arm][s] - slot_rates[denominator_arm][s] for s in sample) / len(sample))
    return {"estimate": estimate, "interval_95": [percentile(draws, .025), percentile(draws, .975)],
            "numerator": numerator_arm, "denominator": denominator_arm,
            "bootstrap_replicates": reps, "bootstrap_seed": seed}


def score_rows(rows, norms_obj):
    no_push = [row for row in rows if row.get("stage") == "no_push"]
    baseline = {int(row["question_index"]): row.get("answer", "") for row in no_push}
    keys, duplicates, nonempty = set(), 0, 0
    for row in rows:
        key = (row.get("stage"), row.get("arm"), row.get("slot"), row.get("layer"),
               int(row.get("question_index", -1)))
        duplicates += int(key in keys)
        keys.add(key)
        nonempty += int(isinstance(row.get("answer"), str) and bool(row["answer"].strip()))
    planned = {("no_push", "plain", None, None, qi) for qi in range(8)}
    planned |= {("battery", arm, slot, layer, qi) for layer in LAYERS for slot in SLOTS
                for arm in ARMS for qi in range(8)}
    gate1 = {"passed": len(rows) == EXPECTED_ROWS and keys == planned and duplicates == 0 and nonempty == EXPECTED_ROWS,
             "expected_rows": EXPECTED_ROWS, "rows": len(rows), "unique_rows": len(keys),
             "duplicates": duplicates, "nonempty_answers": nonempty}
    gate2_bad = [qi for qi in range(8) if qi not in baseline or primary_broken(baseline.get(qi, ""))]
    gate2 = {"passed": len(baseline) == 8 and not gate2_bad,
             "baseline_answers": len(baseline), "broken_question_indices": gate2_bad}

    norm_rows = norms_obj.get("directions", []) if isinstance(norms_obj, dict) else []
    norm_expected = len(LAYERS) * len(SLOTS) * len(ARMS)
    expected_directions = {(layer, slot, arm) for layer in LAYERS for slot in SLOTS for arm in ARMS}
    norm_keys, norm_duplicates = [], 0
    bad_norms, bad_iso = [], []
    for entry in norm_rows:
        if not isinstance(entry, dict):
            bad_norms.append(entry)
            continue
        key = (entry.get("layer"), entry.get("slot"), entry.get("arm"))
        norm_duplicates += int(key in norm_keys)
        norm_keys.append(key)
        values = [entry.get(field) for field in ("write_norm", "target_norm", "relative_error")]
        all_numeric_finite = all(not isinstance(value, (int, float)) or math.isfinite(float(value))
                                 for value in entry.values())
        valid = all_numeric_finite and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values)
        if valid:
            expected_error = abs(float(entry["write_norm"]) - float(entry["target_norm"])) / float(entry["target_norm"]) if entry["target_norm"] else math.inf
            if entry["write_norm"] <= 0 or entry["target_norm"] <= 0 or entry["relative_error"] > .01 or abs(float(entry["relative_error"]) - expected_error) > 1e-6:
                bad_norms.append(entry)
        else:
            bad_norms.append(entry)
        if entry.get("arm") == "iso":
            residual = entry.get("iso_residual_fraction")
            if not isinstance(residual, (int, float)) or not math.isfinite(float(residual)) or residual >= .01:
                bad_iso.append(entry)
    exact_norm_keys = set(norm_keys) == expected_directions and norm_duplicates == 0
    gate3 = {"passed": len(norm_rows) == norm_expected and exact_norm_keys and not bad_norms and not bad_iso,
             "expected_directions": norm_expected, "directions": len(norm_rows),
             "unique_expected_directions": exact_norm_keys, "duplicate_directions": norm_duplicates,
             "write_norm_failures": len(bad_norms), "iso_residual_failures": len(bad_iso),
             "max_relative_write_error": max((float(x.get("relative_error", math.inf)) for x in norm_rows
                                                if isinstance(x, dict) and isinstance(x.get("relative_error"), (int, float))
                                                and math.isfinite(float(x.get("relative_error")))), default=None),
             "max_iso_residual_fraction": max((float(x.get("iso_residual_fraction"))
                                                 for x in norm_rows if isinstance(x, dict) and x.get("arm") == "iso"
                                                 and isinstance(x.get("iso_residual_fraction"), (int, float))
                                                 and math.isfinite(float(x.get("iso_residual_fraction")))), default=None)}

    slot_rates, means, by_slot_arm = read_slot_rates(rows, baseline)
    gate4 = {"passed": means["broken"]["n+"] >= .03,
             "n_plus_broken_rate": means["broken"]["n+"], "minimum": .03}
    gates_passed = all(gate["passed"] for gate in (gate1, gate2, gate3, gate4))

    ratios = {
        "n_plus_over_full": _bootstrap_ratio(slot_rates["broken"], "n+", "full"),
        "n_plus_over_n_pm": _bootstrap_ratio(slot_rates["broken"], "n+", "n±"),
        "iso_over_full": _bootstrap_ratio(slot_rates["broken"], "iso", "full"),
    }
    answer_counts = {arm: {"primary_broken": 0, "total": 0, "broken_under_four_words": 0,
                           "broken_loops": 0} for arm in ARMS}
    for row in rows:
        if row.get("stage") != "battery" or row.get("arm") not in answer_counts:
            continue
        arm, answer = row["arm"], row.get("answer", "")
        answer_counts[arm]["total"] += 1
        if primary_broken(answer):
            answer_counts[arm]["primary_broken"] += 1
            answer_counts[arm]["broken_under_four_words"] += int(len(answer.split()) < 4)
            answer_counts[arm]["broken_loops"] += int(has_trigram_loop(answer))
    for name, ratio in ratios.items():
        numerator, denominator = ratio["numerator"], ratio["denominator"]
        ratio["raw_counts"] = {"numerator_broken": answer_counts[numerator]["primary_broken"],
                               "numerator_total": answer_counts[numerator]["total"],
                               "denominator_broken": answer_counts[denominator]["primary_broken"],
                               "denominator_total": answer_counts[denominator]["total"]}
        ratio["rate_difference"] = _bootstrap_rate_difference(slot_rates["broken"], numerator, denominator)
    full_widths = [int(row["k"]) for row in rows if row.get("stage") == "battery" and
                   row.get("arm") == "full" and row.get("k")]
    full_k = full_widths[0] if full_widths else 2048
    broken_spearman = _bootstrap_spearman(slot_rates, "broken", means, full_k)
    changed_spearman = _bootstrap_spearman(slot_rates, "changed", means, full_k)
    k16, full, npm = (means["broken"][arm] for arm in ("k16", "full", "n±"))
    predictions = {
        "165.1": {"statement": "n+ breaks at least twice as often as full",
                  "estimate_ratio": ratios["n_plus_over_full"]["estimate"],
                  "passed": ratio_at_least(ratios["n_plus_over_full"]["estimate"], 2)},
        "165.2": {"statement": "broken-rate Spearman across the seven ladder arms is at or below -0.8",
                  "spearman": broken_spearman, "passed": spearman_at_most(broken_spearman["estimate"], -.8)},
        "165.3": {"statement": "k16 is nearer full than n±",
                  "distances": {"to_full": abs(k16 - full), "to_n_pm": abs(k16 - npm)},
                  "passed": k16_nearer(k16, full, npm)},
        "165.4": {"statement": "n+ over n± broken ratio is between 0.67 and 1.5",
                  "estimate_ratio": ratios["n_plus_over_n_pm"]["estimate"],
                  "passed": ratio_between(ratios["n_plus_over_n_pm"]["estimate"], .67, 1.5)},
        "165.5": {"statement": "iso breaks under 0.8 times as often as full",
                  "verdict": None,
                  "estimate_ratio": ratios["iso_over_full"]["estimate"],
                  "passed": ratio_less_than(ratios["iso_over_full"]["estimate"], .8)},
        "165.6": {"statement": "changed-rate Spearman across the seven ladder arms is at or below -0.6",
                  "spearman": changed_spearman, "passed": spearman_at_most(changed_spearman["estimate"], -.6)},
    }
    if not gates_passed:
        for prediction in predictions.values():
            prediction["verdict"] = "untestable"
        predictions["165.5"]["verdict"] = "untestable"
    else:
        for prediction in predictions.values():
            if "verdict" not in prediction:
                prediction["verdict"] = "pass" if prediction["passed"] else "fail"
        predictions["165.5"]["verdict"] = ("pass" if predictions["165.5"]["passed"] else "fail") \
            if predictions["165.1"]["passed"] else "not read"

    triage_rates = means["triage_only"]
    per_layer = {}
    for layer in LAYERS:
        per_layer[str(layer)] = {metric: {arm: sum(slot_rates[metric][arm][(layer, slot)] for slot in SLOTS) / len(SLOTS)
                                         for arm in ARMS} for metric in ("broken", "triage_only", "changed")}

    rng = random.Random(165)
    hand = []
    rows_by_arm = {arm: [row for row in rows if row.get("stage") == "battery" and row.get("arm") == arm]
                   for arm in ARMS}
    for arm in ARMS:
        candidates = list(rows_by_arm[arm])
        rng.shuffle(candidates)
        for row in candidates[:20]:
            hand.append({"arm": arm, "layer": row["layer"], "slot": row["slot"],
                         "question_index": row["question_index"], "question": row["question"],
                         "answer": row["answer"]})

    ratios_for_json = {name: value for name, value in ratios.items()}
    broken_shares = {arm: {"primary_broken_answers": answer_counts[arm]["primary_broken"],
                           "short_under_four_words": (answer_counts[arm]["broken_under_four_words"] / answer_counts[arm]["primary_broken"]
                                                       if answer_counts[arm]["primary_broken"] else None),
                           "loops": (answer_counts[arm]["broken_loops"] / answer_counts[arm]["primary_broken"]
                                     if answer_counts[arm]["primary_broken"] else None)} for arm in ARMS}
    ladder_rates = {arm: means["broken"][arm] for arm in LADDER}
    return {"part": 165, "gates": {"gate1_rows": gate1, "gate2_no_push": gate2,
            "gate3_size": gate3, "gate4_measure_room": gate4}, "all_gates_passed": gates_passed,
            "predictions": predictions, "rates": {"broken": means["broken"],
            "triage_only": triage_rates, "changed": means["changed"],
            "primary_broken_counts_and_shares": broken_shares, "ladder_rates_ordered": ladder_rates,
            "per_layer": per_layer,
            "per_slot": {metric: {arm: {f"{layer}:{slot}": value for (layer, slot), value in values.items()}
                                  for arm, values in by_arm.items()} for metric, by_arm in slot_rates.items()}},
            "ratios": ratios_for_json, "spearman": {"broken": broken_spearman, "changed": changed_spearman},
            "hand_read_rows": hand}


def _display(value):
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def write_outputs(result, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hand = result.pop("hand_read_rows")
    (out / "score.json").write_text(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
                                    encoding="utf-8")
    with (out / "hand_read.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in hand:
            stream.write(json.dumps(row, ensure_ascii=True) + "\n")
    lines = ["# Part 165 score", "", "## Gates", "", "| Gate | Result | Detail |", "| --- | --- | --- |"]
    for name, gate in result["gates"].items():
        detail = json.dumps({key: value for key, value in gate.items() if key != "passed"}, ensure_ascii=True)
        lines.append(f"| {name} | {'passed' if gate['passed'] else 'failed'} | `{detail}` |")
    lines += ["", "## Predictions", "", "| Prediction | Verdict | Point estimate | 95% paired slot bootstrap interval |",
              "| --- | --- | ---: | --- |"]
    for name, prediction in result["predictions"].items():
        estimate = prediction.get("estimate_ratio")
        interval = prediction.get("spearman", {}).get("interval_95")
        if interval is None:
            ratio_name = {"165.1": "n_plus_over_full", "165.4": "n_plus_over_n_pm", "165.5": "iso_over_full"}.get(name)
            interval = result["ratios"][ratio_name]["interval_95"] if ratio_name else None
        interval_text = "n/a" if interval is None else f"({_display(interval[0])}, {_display(interval[1])})"
        if estimate is None and "spearman" in prediction:
            estimate = prediction["spearman"]["estimate"]
        if estimate is None and "distances" in prediction:
            estimate = prediction["distances"]
        lines.append(f"| {name} | {prediction['verdict']} | {_display(estimate)} | {interval_text} |")
    lines += ["", "## Ladder shape beside Spearman", "",
              "| Order | Arm | Primary broken rate |", "| ---: | --- | ---: |"]
    for order, arm in enumerate(LADDER, 1):
        lines.append(f"| {order} | {arm} | {_display(result['rates']['ladder_rates_ordered'][arm])} |")
    lines += ["", "## Ratios with raw counts and rate differences", "",
              "| Comparison | Broken counts (numerator / denominator) | Ratio | Rate difference (95% stratified interval) |",
              "| --- | ---: | ---: | ---: |"]
    for name, ratio in result["ratios"].items():
        counts = ratio["raw_counts"]
        diff = ratio["rate_difference"]
        lines.append(f"| {ratio['numerator']} / {ratio['denominator']} | {counts['numerator_broken']}/{counts['numerator_total']} / "
                     f"{counts['denominator_broken']}/{counts['denominator_total']} | {_display(ratio['estimate'])} | "
                     f"{_display(diff['estimate'])} ({_display(diff['interval_95'][0])}, {_display(diff['interval_95'][1])}) |")
    lines += ["", "## Rates by arm", "", "| Arm | Primary broken | Triage only | Changed | Broken answers under four words | Broken answers that loop |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        lines.append(f"| {arm} | {_display(result['rates']['broken'][arm])} | "
                     f"{_display(result['rates']['triage_only'][arm])} | {_display(result['rates']['changed'][arm])} | "
                     f"{_display(result['rates']['primary_broken_counts_and_shares'][arm]['short_under_four_words'])} | "
                     f"{_display(result['rates']['primary_broken_counts_and_shares'][arm]['loops'])} |")
    lines += ["", "## Rates by layer and arm", "", "| Layer | Arm | Primary broken | Triage only | Changed |",
              "| ---: | --- | ---: | ---: | ---: |"]
    for layer, values in result["rates"]["per_layer"].items():
        for arm in ARMS:
            lines.append(f"| {layer} | {arm} | {_display(values['broken'][arm])} | "
                         f"{_display(values['triage_only'][arm])} | {_display(values['changed'][arm])} |")
    lines += ["", "Ratio estimates use paired slot bootstrap intervals; intervals do not decide predictions.",
              "Hand-read answers are in `hand_read.jsonl`.", ""]
    (out / "tables.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--norms", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    norms = json.loads(Path(args.norms).read_text(encoding="utf-8"))
    result = score_rows(read_jsonl(args.answers), norms)
    write_outputs(result, args.out)
    print(json.dumps({"all_gates_passed": result["all_gates_passed"],
                      "predictions": result["predictions"]}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
