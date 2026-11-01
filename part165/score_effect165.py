#!/usr/bin/env python3
"""Host-side scoring for Part 165's first pushed-token effect run."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part165 import run165, score165

LAYERS, SLOTS, ARMS = score165.LAYERS, score165.SLOTS, score165.ARMS
EXPECTED_ROWS = len(LAYERS) * len(SLOTS) * len(ARMS) * 8


def pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    dx, dy = [x - mx for x in xs], [y - my for y in ys]
    den = math.sqrt(sum(x*x for x in dx) * sum(y*y for y in dy))
    return sum(x*y for x, y in zip(dx, dy)) / den if den else None


def percentile(values, p):
    xs = sorted(values)
    pos = (len(xs) - 1) * p
    low, high = int(math.floor(pos)), int(math.ceil(pos))
    return xs[low] if low == high else xs[low] * (high - pos) + xs[high] * (pos - low)


def validate_effects(rows):
    planned = {(arm, slot, layer, qi) for arm in ARMS for slot in SLOTS
               for layer in LAYERS for qi in range(8)}
    seen = set()
    for row in rows:
        key = (row.get("arm"), int(row["slot"]), int(row["layer"]), int(row["question_index"]))
        if key in seen:
            raise ValueError(f"duplicate effect row: {key}")
        seen.add(key)
        for field in ("kl", "rel_change", "realized_write_norm", "intended_write_norm", "realized_cos", "changed_channels"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"non-finite or missing {field} in {key}")
        if not isinstance(row.get("top1_same"), bool):
            raise ValueError(f"missing top1_same in {key}")
        if row["realized_write_norm"] < 0 or row["intended_write_norm"] <= 0 or row["changed_channels"] < 0:
            raise ValueError(f"invalid realized write values in {key}")
    if len(rows) != EXPECTED_ROWS or seen != planned:
        raise ValueError(f"effect gate failed: expected {EXPECTED_ROWS} rows, got {len(rows)}; unique planned={len(seen)}")
    return {"passed": True, "expected_rows": EXPECTED_ROWS, "rows": len(rows), "unique_rows": len(seen),
            "finite_kl_and_rel_change": True}


def _slot_means(rows, field):
    values = defaultdict(list)
    for row in rows:
        values[(row["arm"], int(row["layer"]), int(row["slot"]))].append(float(row[field]))
    return {key: sum(items) / len(items) for key, items in values.items()}


def _paired_ratio_interval(slot_values, numerator="n+", denominator="full", seed=165, reps=10_000):
    slots = [(layer, slot) for layer in LAYERS for slot in SLOTS]
    means = {arm: sum(slot_values[(arm, *slot)] for slot in slots) / len(slots)
             for arm in (numerator, denominator)}
    ratio = means[numerator] / means[denominator] if means[denominator] else math.inf
    grouped = {layer: [(layer, slot) for slot in SLOTS] for layer in LAYERS}
    rng, draws = random.Random(seed), []
    for _ in range(reps):
        sample = [rng.choice(grouped[layer]) for layer in LAYERS for _ in SLOTS]
        n = sum(slot_values[(numerator, *s)] for s in sample) / len(sample)
        d = sum(slot_values[(denominator, *s)] for s in sample) / len(sample)
        draws.append(n / d if d else math.inf)
    return {"estimate": "Infinity" if math.isinf(ratio) else ratio,
            "interval_95": ["Infinity" if math.isinf(percentile(draws, p)) else percentile(draws, p)
                            for p in (.025, .975)],
            "bootstrap_replicates": reps, "bootstrap_seed": seed}


def prediction_165_7(slot_values):
    ratio = _paired_ratio_interval(slot_values)
    passed = float(ratio["estimate"]) >= 1.5 if ratio["estimate"] != "Infinity" else True
    return {"statement": "n+ mean rel_change is at least 1.5 times full's",
            "estimate_ratio": ratio["estimate"], "paired_slot_bootstrap": ratio,
            "passed": passed, "verdict": "pass" if passed else "fail"}


def score_effect(rows, battery_score, answer_rows, geometry=None):
    gate = validate_effects(rows)
    kl_slots, rel_slots = _slot_means(rows, "kl"), _slot_means(rows, "rel_change")
    top_slots = _slot_means([{**r, "top1_same": int(r["top1_same"])} for r in rows], "top1_same")
    means = {}
    top_share = {}
    for arm in ARMS:
        means[arm] = {field: sum(values[(arm, layer, slot)] for layer in LAYERS for slot in SLOTS)
                      / (len(LAYERS) * len(SLOTS)) for field, values in (("kl", kl_slots), ("rel_change", rel_slots))}
        top_share[arm] = sum(top_slots[(arm, layer, slot)] for layer in LAYERS for slot in SLOTS) / (len(LAYERS) * len(SLOTS))

    realized_slots = _slot_means(rows, "realized_write_norm")
    intended_slots = _slot_means(rows, "intended_write_norm")
    cosine_slots = _slot_means(rows, "realized_cos")
    changed_slots = _slot_means(rows, "changed_channels")
    dose_means, equal_dose = {}, True
    for arm in ARMS:
        realized = sum(realized_slots[(arm, layer, slot)] for layer in LAYERS for slot in SLOTS) / (len(LAYERS) * len(SLOTS))
        intended = sum(intended_slots[(arm, layer, slot)] for layer in LAYERS for slot in SLOTS) / (len(LAYERS) * len(SLOTS))
        ratio = realized / intended if intended else math.nan
        cosine = sum(cosine_slots[(arm, layer, slot)] for layer in LAYERS for slot in SLOTS) / (len(LAYERS) * len(SLOTS))
        dose_means[arm] = {"realized_write_norm": realized, "intended_write_norm": intended,
                           "realized_over_intended": ratio, "realized_cos": cosine,
                           "changed_channels": sum(changed_slots[(arm, layer, slot)] for layer in LAYERS for slot in SLOTS) / (len(LAYERS) * len(SLOTS))}
        equal_dose = equal_dose and math.isfinite(ratio) and abs(ratio - 1.0) <= .05 and cosine >= .95
    prediction = prediction_165_7(rel_slots)
    broken = battery_score["rates"]["broken"]
    broken_order = [float(broken[arm]) for arm in ARMS]
    effect_order = [means[arm]["rel_change"] for arm in ARMS]
    broken_effect_spearman = score165.spearman(broken_order, effect_order)

    answer_rows = list(answer_rows)
    baselines = {int(r["question_index"]): r.get("answer", "") for r in answer_rows if r.get("stage") == "no_push"}
    slot_rates, _rate_means, _ = score165.read_slot_rates(answer_rows, baselines)
    slot_table = {}
    for arm in ARMS:
        xs = [slot_rates["broken"][arm][(layer, slot)] for layer in LAYERS for slot in SLOTS]
        ys = [rel_slots[(arm, layer, slot)] for layer in LAYERS for slot in SLOTS]
        slot_table[arm] = {"pearson_broken_rate_vs_rel_change": pearson(xs, ys),
                           "slots": len(xs)}

    differing = sum(bool(row.get("first_token_diff", False)) for row in rows)
    geometry_arms = {arm: {"mean_largest_column_cosine": None, "mean_norm_d": None} for arm in ARMS}
    if geometry:
        values = defaultdict(list)
        for layer in geometry.get("layers", []):
            for item in layer.get("directions", []):
                values[item["arm"]].append(item)
        for arm in ARMS:
            if values[arm]:
                geometry_arms[arm] = {"mean_largest_column_cosine": sum(x["largest_column_cosine"] for x in values[arm]) / len(values[arm]),
                                      "mean_norm_d": sum(x["norm_d"] for x in values[arm]) / len(values[arm])}
    return {"part": 165, "gate": gate, "first_token_differences": differing,
            "equal_dose": bool(equal_dose), "arm_means": {arm: {**means[arm], **dose_means[arm]} for arm in ARMS},
            "geometry_arm_means": geometry_arms, "top1_same_share": top_share,
            "prediction_165.7": prediction,
            "broken_rate_vs_rel_change_spearman": broken_effect_spearman,
            "broken_rates_from_battery_score": {arm: broken[arm] for arm in ARMS},
            "slot_correlations": slot_table}


def write_outputs(result, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "effect.json").write_text(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    ratio_value = result["prediction_165.7"]["estimate_ratio"]
    ratio_text = str(ratio_value) if isinstance(ratio_value, str) else f"{ratio_value:.4f}"
    spearman_value = result["broken_rate_vs_rel_change_spearman"]
    spearman_text = "n/a" if spearman_value is None else f"{spearman_value:.4f}"
    lines = ["# Part 165 effect", "", "## Gate and prediction", "",
             f"Rows gate: {'passed' if result['gate']['passed'] else 'failed'} ({result['gate']['rows']} / {result['gate']['expected_rows']}).",
             f"Prediction 165.7: **{result['prediction_165.7']['verdict']}**, n+ / full relative-change ratio "
             f"{ratio_text} (paired slot bootstrap 95% interval "
             f"{result['prediction_165.7']['paired_slot_bootstrap']['interval_95']}).",
             f"First generated token differences from plain: {result['first_token_differences']}.", "",
             f"Equal dose: **{result['equal_dose']}** (mean realized/intended and cosine shown per arm).", "",
             "## Arm means", "", "| Arm | Mean KL | Mean relative change | Realized write norm | Intended write norm | Realized/intended | Realized cosine | Changed channels | Largest column cosine | Mean norm d | Top1 same share | Battery broken rate |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        geo = result["geometry_arm_means"][arm]
        lines.append(f"| {arm} | {result['arm_means'][arm]['kl']:.6g} | {result['arm_means'][arm]['rel_change']:.6g} | "
                     f"{result['arm_means'][arm]['realized_write_norm']:.6g} | {result['arm_means'][arm]['intended_write_norm']:.6g} | "
                     f"{result['arm_means'][arm]['realized_over_intended']:.4f} | {result['arm_means'][arm]['realized_cos']:.4f} | "
                     f"{result['arm_means'][arm]['changed_channels']:.2f} | "
                     f"{geo['mean_largest_column_cosine'] if geo['mean_largest_column_cosine'] is not None else 'n/a'} | "
                     f"{geo['mean_norm_d'] if geo['mean_norm_d'] is not None else 'n/a'} | "
                     f"{result['top1_same_share'][arm]:.4f} | {result['broken_rates_from_battery_score'][arm]:.4f} |")
    lines += ["", f"Spearman across arms (battery broken rate vs mean relative change): {spearman_text}.",
              "", "## Slot correlations", "", "Pearson correlation within each arm across 96 slots, using slot broken rate from answers.jsonl and slot mean relative change.",
              "", "| Arm | Broken rate vs relative change | Slots |", "| --- | ---: | ---: |"]
    for arm in ARMS:
        corr = result["slot_correlations"][arm]["pearson_broken_rate_vs_rel_change"]
        lines.append(f"| {arm} | {'n/a' if corr is None else f'{corr:.4f}'} | {result['slot_correlations'][arm]['slots']} |")
    (out / "effect-tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effects", required=True)
    parser.add_argument("--score", required=True, help="battery score.json")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--geometry", default=None, help="geometry.json; defaults beside score")
    args = parser.parse_args(argv)
    effect_rows = run165.read_jsonl(args.effects)
    battery_score = json.loads(Path(args.score).read_text(encoding="utf-8"))
    answer_path = Path(args.score).parent / "answers.jsonl"
    answer_rows = score165.read_jsonl(answer_path)
    geometry_path = Path(args.geometry) if args.geometry else Path(args.score).parent / "geometry.json"
    geometry = json.loads(geometry_path.read_text(encoding="utf-8")) if geometry_path.exists() else None
    result = score_effect(effect_rows, battery_score, answer_rows, geometry)
    write_outputs(result, args.out)
    print(json.dumps({"gate": result["gate"], "prediction_165.7": result["prediction_165.7"]}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
