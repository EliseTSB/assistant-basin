#!/usr/bin/env python3
"""Host scorer for Part 170; all registered estimates are computed from saved rows."""
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
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np
from part165 import score165
from part166 import score166
from part170 import run170

LAYERS = run170.LAYERS
SLOTS = run170.SLOTS
DOSES = run170.DOSES
REPS = run170.BOOTSTRAP_REPS
SEED = run170.BOOTSTRAP_SEED


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def outcomes(answers):
    rows = []
    for answer in answers:
        if answer.get("stage") not in ("no_push", "push", "push_repeat"):
            continue
        text = answer.get("answer", "")
        rows.append({**answer,
            "loop": bool(score166.loop_phrase(text)),
            "broken": bool(score165.primary_broken(text)),
            "token_count": int(answer.get("token_count", len(answer.get("token_ids", [])))),
            "eos": bool(answer.get("ended_eos_before_80", False)),
            "blocked_count": len(answer.get("blocked_steps", []))})
    return rows


def _loop_rate(rows, key):
    vals = [bool(row.get(key)) for row in rows]
    return sum(vals) / len(vals) if vals else None


def _pair_no_push(rows, groups, ban, eligible_ids=None):
    selected = [row for row in rows if row.get("stage") == "no_push" and
                row.get("group") in groups and row.get("dose") == 0.0]
    base = {str(row["prompt_id"]): row for row in selected if row.get("ban") == "b0"}
    alt = {str(row["prompt_id"]): row for row in selected if row.get("ban") == ban}
    keys = sorted(set(base) & set(alt))
    if eligible_ids is not None:
        allowed = set(map(str, eligible_ids)); keys = [key for key in keys if key in allowed]
    return [{"prompt_id": key, "baseline_loop": bool(base[key]["loop"]),
             "alt_loop": bool(alt[key]["loop"]), "baseline": base[key], "alt": alt[key]}
            for key in keys]


def _pair_push(rows, dose, ban):
    selected = [row for row in rows if row.get("stage") == "push" and not row.get("repeat") and
                row.get("arm") == "n+" and float(row.get("dose", -1)) == float(dose)]
    base = {run170.push_row_key(row): row for row in selected if row.get("ban") == "b0"}
    alt = {run170.push_row_key(row): row for row in selected if row.get("ban") == ban}
    keys = sorted(set(base) & set(alt))
    return [{"prompt_id": key[0], "layer": int(key[1]), "slot": int(key[2]),
             "baseline_loop": bool(base[key]["loop"]), "alt_loop": bool(alt[key]["loop"]),
             "blocked_count": len(alt[key].get("blocked_steps", [])),
             "baseline": base[key], "alt": alt[key]} for key in keys]


def _mean_rise(pair):
    if not pair:
        return None, None, None
    base = sum(row["baseline_loop"] for row in pair) / len(pair)
    alt = sum(row["alt_loop"] for row in pair) / len(pair)
    return base, alt, alt - base


def _slot_weights(rng, reps=REPS, layers=LAYERS, slots=SLOTS):
    draws = rng.integers(0, int(slots), size=(int(reps), len(layers), int(slots)))
    return (draws[..., None] == np.arange(int(slots))).sum(axis=2).astype(np.float64)


def _push_draws(pair, weights, field):
    if not pair:
        return np.full(weights.shape[0], np.nan)
    layer_pos = {int(layer): index for index, layer in enumerate(LAYERS)}
    layer_ix = np.asarray([layer_pos[int(row["layer"])] for row in pair], dtype=np.int64)
    slot_ix = np.asarray([int(row["slot"]) for row in pair], dtype=np.int64)
    row_weights = weights[:, layer_ix, slot_ix]
    values = np.asarray([float(row[field]) for row in pair], dtype=np.float64)
    den = row_weights.sum(axis=1)
    num = row_weights @ values
    return np.divide(num, den, out=np.full(len(den), np.nan), where=den > 0)


def _push_masked_draws(pair, weights, values, mask):
    if not pair or not any(mask):
        return np.full(weights.shape[0], np.nan)
    layer_pos = {int(layer): index for index, layer in enumerate(LAYERS)}
    chosen = [index for index, held in enumerate(mask) if held]
    layer_ix = np.asarray([layer_pos[int(pair[i]["layer"])] for i in chosen], dtype=np.int64)
    slot_ix = np.asarray([int(pair[i]["slot"]) for i in chosen], dtype=np.int64)
    row_weights = weights[:, layer_ix, slot_ix]
    vals = np.asarray([float(values[i]) for i in chosen], dtype=np.float64)
    den = row_weights.sum(axis=1)
    return np.divide(row_weights @ vals, den, out=np.full(len(den), np.nan), where=den > 0)


def _prompt_draws(pair, rng, reps=REPS):
    by_id = {str(row["prompt_id"]): row for row in pair}
    ids = sorted(by_id)
    if not ids:
        return {"ids": [], "baseline": np.full(reps, np.nan), "alt": np.full(reps, np.nan),
                "rise": np.full(reps, np.nan)}
    draw = rng.integers(0, len(ids), size=(int(reps), len(ids)))
    base = np.asarray([bool(by_id[key]["baseline_loop"]) for key in ids], dtype=np.float64)
    alt = np.asarray([bool(by_id[key]["alt_loop"]) for key in ids], dtype=np.float64)
    b = base[draw].mean(axis=1); a = alt[draw].mean(axis=1)
    return {"ids": ids, "baseline": b, "alt": a, "rise": a - b}


def _interval(values, estimate, seed=SEED, reps=REPS):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return {"estimate": estimate,
            "interval_95": ([float(x) for x in np.quantile(vals, [0.025, 0.975])]
                            if len(vals) else [None, None]),
            "usable_draws": int(len(vals)), "replicates": int(reps), "seed": int(seed)}


def _signed_ratio_draws(numerator, denominator):
    # A rise can be negative, and a fall must not read as a rise, so these keep the sign.
    n = np.asarray(numerator, dtype=np.float64); d = np.asarray(denominator, dtype=np.float64)
    return np.divide(n, d, out=np.full(np.broadcast_shapes(n.shape, d.shape), np.nan), where=d >= 0.01)


def _signed_ratio_point(numerator, denominator):
    return float(numerator) / float(denominator) if numerator is not None and denominator is not None and float(denominator) >= 0.01 else None


def _noise_floor(gates):
    diag = gates.get("diagnostics", {})
    return {"differing_answers": diag.get("gate8_differing_answers"),
            "rows": diag.get("gate8_rows"), "rate": diag.get("gate8_noise_rate")}


def _with_noise(value, noise):
    return {**value, "noise_floor": noise}


def _no_push_prediction(pair, push_rise, push_rise_draws, prompt_rng, bar=0.25):
    no_push_boot = _prompt_draws(pair, prompt_rng)
    baseline, alt, rise = _mean_rise(pair)
    ratio = _signed_ratio_point(rise, push_rise)
    ratios = _signed_ratio_draws(no_push_boot["rise"], push_rise_draws)
    return {"estimate": ratio,
            "interval_95": _interval(ratios, ratio)["interval_95"],
            "usable_draws": _interval(ratios, ratio)["usable_draws"],
            "replicates": REPS, "seed": SEED, "bar": f"no-push rise / dose 1 rise < {bar}, signed",
            "pass": bool(ratio is not None and ratio < bar), "rows": len(pair),
            "no_push_baseline_rate": baseline, "no_push_b50_rate": alt,
            "absolute_no_push_rise": abs(rise) if rise is not None else None, "no_push_rise": rise,
            "dose1_rise": push_rise,
            "dose1_rise_near_zero": push_rise is None or abs(push_rise) < 0.01,
            "fraction_testable": ratio is not None}


def _cell_summaries(rows, eligible_ids=None):
    eligible_ids = None if eligible_ids is None else set(map(str, eligible_ids))
    cells = defaultdict(list)
    for row in rows:
        if row.get("stage") == "no_push":
            if eligible_ids is not None and str(row.get("prompt_id")) not in eligible_ids:
                continue
            key = ("no_push", "all", 0.0, row.get("group"), row.get("ban"))
        elif row.get("stage") == "push":
            if eligible_ids is not None and str(row.get("prompt_id")) not in eligible_ids:
                continue
            key = ("push", row.get("arm"), float(row.get("dose", 0)), "p8", row.get("ban"))
        else:
            continue
        cells[key].append(row)
    out = []
    for key, vals in sorted(cells.items(), key=lambda x: tuple(str(v) for v in x[0])):
        n = len(vals)
        out.append({"stage": key[0], "arm": key[1], "dose": key[2], "group": key[3], "ban": key[4],
            "rows": n, "loop_rate": _loop_rate(vals, "loop"), "broken_rate": _loop_rate(vals, "broken"),
            "joint_success_rate": sum(not row["loop"] and not row["broken"] for row in vals) / n if n else None,
            "mean_tokens": sum(row["token_count"] for row in vals) / n if n else None,
            "eos_rate": sum(row["eos"] for row in vals) / n if n else None})
    return out


def _paired_changes(rows, groups, push_dose=None, eligible_ids=None):
    out = []
    if push_dose is not None:
        for ban in ("b5", "b50", "b200", "rel50"):
            pair = _pair_push(rows, push_dose, ban)
            out.append(_change_record("push", "p8", push_dose, ban, pair))
    else:
        for group in groups:
            for ban in ("b5", "b50", "b200", "rel50", "other50"):
                if ban == "other50" and group != "q96":
                    continue
                pair = _pair_no_push(rows, [group], ban, eligible_ids)
                out.append(_change_record("no_push", group, 0.0, ban, pair))
    return out


def _change_record(stage, group, dose, ban, pair):
    base, alt, rise = _mean_rise(pair)
    base_loop = sum(row["baseline_loop"] for row in pair)
    alt_loop = sum(row["alt_loop"] for row in pair)
    ended = sum(row["baseline_loop"] and not row["alt_loop"] for row in pair)
    induced = sum(not row["baseline_loop"] and row["alt_loop"] for row in pair)
    eligible = sum(not row["baseline_loop"] for row in pair)
    return {"stage": stage, "group": group, "dose": dose, "ban": ban, "rows": len(pair),
            "baseline_loop_rate": base, "list_loop_rate": alt, "rise": rise,
            "induced": induced, "induced_eligible": eligible,
            "induced_rate": induced / eligible if eligible else None,
            "ended": ended, "net_loop_change": alt_loop - base_loop,
            "net_rate_change": rise}


def _share_record(pair, lists):
    values = []
    counts = []
    for row in pair:
        ids = set(map(int, row.get("token_ids", [])))
        held = set(map(int, lists[str(row["prompt_id"])]["b50"]))
        count = len(ids & held)
        share = count / len(ids) if ids else None
        if share is not None:
            values.append(share); counts.append(count)
    return {"rows": len(values), "mean_share": sum(values) / len(values) if values else None,
            "mean_overlap_count_part169_definition": sum(counts) / len(counts) if counts else None}


def _prediction_1705(rows, weights):
    p200 = _pair_push(rows, 1.0, "b200")
    p50 = _pair_push(rows, 1.0, "b50")
    p5 = _pair_push(rows, 1.0, "b5")
    _b200, r200, _ = _mean_rise(p200)
    _b50a, r50a, _ = _mean_rise(p50)
    _b5, r5, _ = _mean_rise(p5)
    # Signed: a larger list can block the loop's own tokens, so b200 below b50 must fail this bar.
    gap200 = r200 - r50a if r200 is not None and r50a is not None else None
    gap5 = r50a - r5 if r50a is not None and r5 is not None else None
    d200 = _push_draws(p200, weights, "alt_loop") - _push_draws(p50, weights, "alt_loop")
    d5 = _push_draws(p50, weights, "alt_loop") - _push_draws(p5, weights, "alt_loop")
    minimum = np.minimum(d200, d5)
    estimate = min(gap200, gap5) if gap200 is not None and gap5 is not None else None
    return {**_interval(minimum, estimate), "bar": "b200 - b50 >= 0.05 and b50 - b5 >= 0.05, signed",
        "pass": bool(gap200 is not None and gap5 is not None and gap200 >= 0.05 and gap5 >= 0.05),
        "rows": min(len(p200), len(p50), len(p5)), "b200_minus_b50_signed": r200-r50a if r200 is not None and r50a is not None else None,
        "b50_minus_b5_signed": r50a-r5 if r50a is not None and r5 is not None else None,
        "b200_b50_gap": gap200, "b50_b5_gap": gap5,
        "b200_b50_interval_95": _interval(d200, gap200)["interval_95"],
        "b50_b5_interval_95": _interval(d5, gap5)["interval_95"]}


def _prediction_1706(rows, weights):
    p50 = _pair_push(rows, 1.0, "b50")
    prel = _pair_push(rows, 1.0, "rel50")
    base, b50_rate, b50_rise = _mean_rise(p50)
    _base2, rel_rate, rel_rise = _mean_rise(prel)
    mean50 = sum(len(row["alt"].get("blocked_steps", [])) for row in p50) / len(p50) if p50 else None
    meanrel = sum(len(row["alt"].get("blocked_steps", [])) for row in prel) / len(prel) if prel else None
    step_share = meanrel / mean50 if meanrel is not None and mean50 not in (None, 0) else None
    rise_share = _signed_ratio_point(rel_rise, b50_rise)
    margin = (1.5 * step_share * b50_rise - rel_rise
              if step_share is not None and b50_rise is not None and rel_rise is not None else None)
    b50_base = np.asarray([row["baseline_loop"] for row in p50], dtype=np.float64)
    b50_alt = np.asarray([row["alt_loop"] for row in p50], dtype=np.float64)
    rel_alt = np.asarray([row["alt_loop"] for row in prel], dtype=np.float64)
    blocked50 = np.asarray([len(row["alt"].get("blocked_steps", [])) for row in p50], dtype=np.float64)
    blockedrel = np.asarray([len(row["alt"].get("blocked_steps", [])) for row in prel], dtype=np.float64)
    bdraw = _push_draws(p50, weights, "baseline_loop")
    a50 = _push_draws(p50, weights, "alt_loop")
    arel = _push_draws(prel, weights, "alt_loop")
    r50 = a50 - bdraw; rrel = arel - bdraw
    steps50 = _push_draws(p50, weights, "blocked_count")
    stepsrel = _push_draws(prel, weights, "blocked_count")
    step_draw = np.divide(stepsrel, steps50, out=np.full(len(steps50), np.nan), where=steps50 > 0)
    ratio_draw = _signed_ratio_draws(rrel, r50)
    margin_draw = 1.5 * step_draw * r50 - rrel
    return {**_interval(margin_draw, margin), "estimate_is": "margin: 1.5 times step share times b50 rise, minus rel50 rise; rise_ratio is the bar quantity",
        "bar": "rel50 rise / b50 rise < 1.5 * rel50 blocked-step share, signed",
        "pass": bool(rise_share is not None and step_share is not None and rise_share < 1.5 * step_share),
        "rows": min(len(p50), len(prel)), "rel50_rise": rel_rise, "b50_rise": b50_rise,
        "rise_ratio": rise_share,
        "rise_ratio_interval_95": _interval(ratio_draw, rise_share)["interval_95"],
        "mean_blocked_steps": {"rel50": meanrel, "b50": mean50},
        "rel50_blocked_step_share": step_share,
        "blocked_step_share_interval_95": _interval(step_draw, step_share)["interval_95"],
        "near_zero_b50_rise": b50_rise is None or abs(b50_rise) < 0.01,
        "base_loop_rate": base, "b50_loop_rate": b50_rate, "rel50_loop_rate": rel_rate}


def _prediction_w2(rows, weights):
    p50 = _pair_push(rows, 1.0, "b50"); prel = _pair_push(rows, 1.0, "rel50")
    _base, _alt50, rise50 = _mean_rise(p50); _base2, _altrel, riserel = _mean_rise(prel)
    ratio = _signed_ratio_point(riserel, rise50)
    margin = riserel - 0.5 * rise50 if riserel is not None and rise50 is not None else None
    bdraw = _push_draws(p50, weights, "baseline_loop")
    rise50draw = _push_draws(p50, weights, "alt_loop") - bdraw
    risereldraw = _push_draws(prel, weights, "alt_loop") - bdraw
    ratio_draw = _signed_ratio_draws(risereldraw, rise50draw)
    margin_draw = risereldraw - 0.5 * rise50draw
    if ratio is None:
        classification = "not_testable_near_zero_b50_rise"
    elif ratio < 1.0 / 3.0:
        classification = "falsifier_under_one_third"
    elif ratio < 0.5:
        classification = "inconclusive_between_one_third_and_one_half"
    else:
        classification = "prediction_holds_at_least_one_half"
    return {**_interval(margin_draw, margin), "estimate_is": "margin: rel50 rise minus half the b50 rise; rise_ratio is the bar quantity",
        "bar": "rel50 rise / b50 rise >= 0.5, signed",
        "pass": bool(ratio is not None and ratio >= 0.5), "classification": classification,
        "rows": min(len(p50), len(prel)), "rel50_rise": riserel, "b50_rise": rise50,
        "rise_ratio": ratio, "rise_ratio_interval_95": _interval(ratio_draw, ratio)["interval_95"]}


def _prediction_1708(rows, weights):
    pair = _pair_push(rows, 1.0, "b50")
    no_loop = [row for row in pair if not row["baseline_loop"]]
    shares = []
    lengths = []
    outcomes_b50 = []
    slot_rows = []
    for row in no_loop:
        ids = set(map(int, row["baseline"].get("token_ids", [])))
        held = set(map(int, row["alt"].get("banned_ids", [])))
        # The count is Part 169's intersection count; share uses distinct b0 answer tokens.
        shares.append(len(ids & held) / len(ids) if ids else 0.0)
        lengths.append(int(row["baseline"].get("token_count", len(ids))))
        outcomes_b50.append(float(row["alt_loop"]))
        slot_rows.append(row)
    if not no_loop:
        empty = {"rows": 0, "difference": None, "below": None, "rest": None}
        return {**_interval([], None), "bar": "length-matched below-share induced-rate gap >= 0.05",
                "pass": False, "rows": 0, "raw_difference": None,
                "length_matched_difference": None, "raw_median_split": empty,
                "length_matched": {"rows": 0, "difference": None, "quartiles": []}}
    shares = np.asarray(shares, dtype=np.float64); lengths = np.asarray(lengths, dtype=np.float64)
    y = np.asarray(outcomes_b50, dtype=np.float64)
    median_share = float(np.median(shares))
    raw_low = shares < median_share
    raw_rest = ~raw_low
    raw_low_rate = float(y[raw_low].mean()) if raw_low.any() else None
    raw_rest_rate = float(y[raw_rest].mean()) if raw_rest.any() else None
    raw_gap = raw_low_rate - raw_rest_rate if raw_low_rate is not None and raw_rest_rate is not None else None
    cuts = np.quantile(lengths, [0.25, 0.5, 0.75])
    quartile = np.searchsorted(cuts, lengths, side="right")
    matched_low = np.zeros(len(no_loop), dtype=bool)
    quartile_cells = []
    for q in range(4):
        indexes = np.flatnonzero(quartile == q)
        med = float(np.median(shares[indexes])) if len(indexes) else None
        low_ix = indexes[shares[indexes] < med] if med is not None else np.asarray([], dtype=int)
        matched_low[low_ix] = True
        rest_ix = indexes[shares[indexes] >= med] if med is not None else np.asarray([], dtype=int)
        low_rate = float(y[low_ix].mean()) if len(low_ix) else None
        rest_rate = float(y[rest_ix].mean()) if len(rest_ix) else None
        quartile_cells.append({"quartile": q + 1, "rows": int(len(indexes)),
            "length_min": int(lengths[indexes].min()) if len(indexes) else None,
            "length_max": int(lengths[indexes].max()) if len(indexes) else None,
            "median_share": med, "below_share_rows": int(len(low_ix)),
            "rest_rows": int(len(rest_ix)), "below_induced_rate": low_rate,
            "rest_induced_rate": rest_rate,
            "difference": low_rate - rest_rate if low_rate is not None and rest_rate is not None else None})
    matched_rest = ~matched_low
    match_low_rate = float(y[matched_low].mean()) if matched_low.any() else None
    match_rest_rate = float(y[matched_rest].mean()) if matched_rest.any() else None
    # The bar is the row-weighted mean of the within-quartile differences. Pooling low against rest would let
    # uneven ties at a median carry length back in.
    cells = [c for c in quartile_cells if c["difference"] is not None]
    match_gap = (sum(c["difference"] * c["rows"] for c in cells) / sum(c["rows"] for c in cells)
                 if cells else None)

    pair_weights = weights[:, np.asarray([LAYERS.index(int(r["layer"])) for r in slot_rows]),
                            np.asarray([int(r["slot"]) for r in slot_rows])]
    den_low = pair_weights[:, matched_low].sum(axis=1)
    den_rest = pair_weights[:, matched_rest].sum(axis=1)
    # Same estimator in every replicate; a quartile with an empty side in a replicate drops out of that replicate.
    num = np.zeros(pair_weights.shape[0]); den = np.zeros(pair_weights.shape[0])
    for q in range(4):
        low = matched_low & (quartile == q); rest = matched_rest & (quartile == q)
        if not low.any() or not rest.any():
            continue
        wl = pair_weights[:, low].sum(axis=1); wr = pair_weights[:, rest].sum(axis=1)
        rl = np.divide(pair_weights[:, low] @ y[low], wl, out=np.zeros(len(wl)), where=wl > 0)
        rr = np.divide(pair_weights[:, rest] @ y[rest], wr, out=np.zeros(len(wr)), where=wr > 0)
        use = (wl > 0) & (wr > 0)
        num += np.where(use, (rl - rr) * (wl + wr), 0.0); den += np.where(use, wl + wr, 0.0)
    matched_draws = np.divide(num, den, out=np.full(len(den), np.nan), where=den > 0)
    raw_den_low = pair_weights[:, raw_low].sum(axis=1)
    raw_den_rest = pair_weights[:, raw_rest].sum(axis=1)
    raw_a = np.divide(pair_weights[:, raw_low] @ y[raw_low], raw_den_low,
        out=np.full(len(raw_den_low), np.nan), where=raw_den_low > 0)
    raw_b = np.divide(pair_weights[:, raw_rest] @ y[raw_rest], raw_den_rest,
        out=np.full(len(raw_den_rest), np.nan), where=raw_den_rest > 0)
    raw_draws = raw_a - raw_b
    return {**_interval(matched_draws, match_gap),
        "bar": "within b0-length quartiles, below-share induced-rate gap >= 0.05",
        "pass": bool(match_gap is not None and match_gap >= 0.05),
        "rows": len(no_loop), "raw_difference": raw_gap,
        "raw_difference_interval_95": _interval(raw_draws, raw_gap)["interval_95"],
        "length_matched_difference": match_gap,
        "raw_median_split": {"rows": len(no_loop), "median_share": median_share,
            "below_rows": int(raw_low.sum()), "rest_rows": int(raw_rest.sum()),
            "below_induced_rate": raw_low_rate, "rest_induced_rate": raw_rest_rate,
            "difference": raw_gap},
        "length_matched": {"rows": len(no_loop), "quartile_cuts": [float(x) for x in cuts],
            "below_rows": int(matched_low.sum()), "rest_rows": int(matched_rest.sum()),
            "below_induced_rate": match_low_rate, "rest_induced_rate": match_rest_rate,
            "difference": match_gap, "quartiles": quartile_cells}}


def _blind_hand_sample(rows, seed=170, per_cell=24):
    rng = random.Random(seed)
    chosen = []
    no_push_loops = [row for row in rows if row.get("stage") == "no_push" and
                     row.get("ban") == "b50" and row.get("loop")]
    chosen.extend(("all_no_push_b50_loops", row) for row in no_push_loops)
    cells = (("no_push", 0.0, "b0"), ("no_push", 0.0, "b50"),
             ("push", 1.0, "b0"), ("push", 1.0, "b50"))
    for stage, dose, ban in cells:
        candidates = [row for row in rows if row.get("stage") == stage and not row.get("repeat") and
                      float(row.get("dose", 0.0)) == dose and row.get("ban") == ban]
        already = {run170.answer_key(row) for _label, row in chosen}
        candidates = [row for row in candidates if run170.answer_key(row) not in already]
        amount = min(int(per_cell), len(candidates))
        chosen.extend((f"sample_{stage}_dose{dose:g}_{ban}", row)
                      for row in rng.sample(candidates, amount))
    rng.shuffle(chosen)
    sample = []; key = []
    for number, (label, row) in enumerate(chosen, 1):
        blind_id = f"H170-{number:04d}"
        sample.append({"blind_id": blind_id, "answer": row.get("answer", ""),
            "token_ids": row.get("token_ids", []), "token_count": row.get("token_count", 0)})
        key.append({"blind_id": blind_id, "sample_source": label,
            "stage": row.get("stage"), "group": row.get("group"), "arm": row.get("arm"),
            "dose": row.get("dose"), "ban": row.get("ban"),
            "prompt_id": row.get("prompt_id"), "layer": row.get("layer"), "slot": row.get("slot")})
    return sample, key


def _write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")


def _score_predictions(rows, prompts, stage0, lists, gates):
    frozen = set(map(str, stage0.get("frozen_prompt_ids", [])))
    prompt_by_id = {str(p["prompt_id"]): p for p in prompts}
    eligible = {pid for pid in prompt_by_id if pid not in frozen}
    p8_ids = {pid for pid in eligible if prompt_by_id[pid]["group"] == "p8"}
    maker_ids = {pid for pid in eligible if prompt_by_id[pid]["group"] in ("p8", "m96")}
    m96_ids = {pid for pid in eligible if prompt_by_id[pid]["group"] == "m96"}
    q96_ids = {pid for pid in eligible if prompt_by_id[pid]["group"] == "q96"}
    push025 = _pair_push(rows, 0.25, "b50")
    push1 = _pair_push(rows, 1.0, "b50")
    _base025, _alt025, rise025 = _mean_rise(push025)
    _base1, _alt1, rise1 = _mean_rise(push1)
    noise = _noise_floor(gates)

    # One shared seeded generator; all 10,000 resamples preserve within-row pairing.
    rng = np.random.default_rng(SEED)
    weights = _slot_weights(rng)
    pmaker = _pair_no_push(rows, ["p8", "m96"], "b50", maker_ids)
    pmaker_all = _pair_no_push(rows, ["p8", "m96"], "b50")
    pq = _pair_no_push(rows, ["q96"], "b50", q96_ids)
    pq_all = _pair_no_push(rows, ["q96"], "b50")
    _bpm, _apm, rise_maker = _mean_rise(pmaker)
    _bpq, _apq, rise_q = _mean_rise(pq)
    rise1_draws = _push_draws(push1, weights, "alt_loop") - _push_draws(push1, weights, "baseline_loop")

    p1701 = _no_push_prediction(pmaker, rise1, rise1_draws, rng)
    p1701["with_frozen"] = _no_push_prediction(pmaker_all, rise1, rise1_draws, rng)
    p1701["groups"] = ["p8", "m96"]
    p1702 = _no_push_prediction(pq, rise1, rise1_draws, rng)
    p1702["with_frozen"] = _no_push_prediction(pq_all, rise1, rise1_draws, rng)
    p1702["groups"] = ["q96"]

    p3_base, p3_alt, p3_rise = _mean_rise(push1)
    p3_margin = p3_alt - 1.5 * p3_base if p3_alt is not None and p3_base is not None else None
    p3_margin_draw = _push_draws(push1, weights, "alt_loop") - 1.5 * _push_draws(push1, weights, "baseline_loop")
    p1703 = {**_interval(p3_margin_draw, p3_margin),
        "bar": "b50 loop rate >= 1.5 * b0 loop rate",
        "pass": bool(p3_base is not None and p3_base > 0 and p3_margin is not None and p3_margin >= 0),
        "rows": len(push1), "b0_rate": p3_base, "b50_rate": p3_alt,
        "rise": p3_rise, "estimate_is": "margin: b50 rate minus 1.5 times b0 rate; relative_rise is the bar quantity",
        "relative_rise": p3_alt / p3_base - 1 if p3_base not in (None, 0) else None,
        "base_near_zero": p3_base is None or p3_base < 0.01}

    p4_025 = _pair_push(rows, 0.25, "b50")
    rise025_draws = _push_draws(p4_025, weights, "alt_loop") - _push_draws(p4_025, weights, "baseline_loop")
    p4_margin = rise1 - 2 * rise025 if rise1 is not None and rise025 is not None else None
    p4_draws = rise1_draws - 2 * rise025_draws
    p1704 = {**_interval(p4_draws, p4_margin),
        "estimate_is": "margin: dose 1 rise minus twice the dose 0.25 rise",
        "bar": "dose 1 b50 rise > 0 and >= 2 * dose 0.25 b50 rise, signed",
        "pass": bool(p4_margin is not None and rise1 > 0 and p4_margin >= 0), "rows_dose1": len(push1),
        "rows_dose025": len(p4_025), "dose1_rise": rise1, "dose025_rise": rise025,
        "dose025_near_zero": rise025 is None or abs(rise025) < 0.01}

    p1705 = _prediction_1705(rows, weights)
    p1706 = _prediction_1706(rows, weights)

    pother = _pair_no_push(rows, ["q96"], "other50", q96_ids)
    obase, oalt, orise = _mean_rise(pother)
    other_boot = _prompt_draws(pother, rng)
    other_abs_draws = np.abs(other_boot["rise"])
    other_abs = abs(orise) if orise is not None else None
    b5pair = _pair_no_push(rows, ["q96"], "b5", q96_ids)
    b50pair = pq
    _b5base, _b5alt, rise_b5 = _mean_rise(b5pair)
    p1707 = {**_interval(other_abs_draws, other_abs),
        "bar": "absolute other50 minus b0 loop-rate change < 0.03",
        "pass": bool(other_abs is not None and other_abs < 0.03), "rows": len(pother),
        "other50_b0_rate": obase, "other50_rate": oalt, "other50_rise": orise,
        "b5_rise": rise_b5, "b50_rise": rise_q,
        "exploratory_rows": len(pother)}

    p1708 = _prediction_1708(rows, weights)

    # Weaver W1 is computed independently for M96 and Q96; those sets are never pooled.
    pm96 = _pair_no_push(rows, ["m96"], "b50", m96_ids)
    pqw = _pair_no_push(rows, ["q96"], "b50", q96_ids)
    bm96, am96, rm96 = _mean_rise(pm96); bq, aq, rq = _mean_rise(pqw)
    mdraw = _prompt_draws(pm96, rng); qdraw = _prompt_draws(pqw, rng)
    mabs = abs(rm96) if rm96 is not None else None; qabs = abs(rq) if rq is not None else None
    w1_draw = np.maximum(mdraw["rise"], qdraw["rise"])
    w1_est = max(rm96, rq) if rm96 is not None and rq is not None else None
    pW1 = {**_interval(w1_draw, w1_est),
        "bar": "b50 rise < 0.05 separately on M96 and Q96, signed",
        "pass": bool(rm96 is not None and rq is not None and rm96 < 0.05 and rq < 0.05),
        "falsifier_met": bool((rm96 is not None and rm96 >= 0.10) or (rq is not None and rq >= 0.10)),
        "m96": {"rows": len(pm96), "b0_rate": bm96, "b50_rate": am96,
                "rise": rm96, "absolute_rise": mabs,
                "interval_95": _interval(mdraw["rise"], rm96)["interval_95"]},
        "q96": {"rows": len(pqw), "b0_rate": bq, "b50_rate": aq,
                "rise": rq, "absolute_rise": qabs,
                "interval_95": _interval(qdraw["rise"], rq)["interval_95"]}}
    set_passed = stage0.get("set_passed", {})
    for name, needed in (("170.1", ("m96",)), ("170.2", ("q96",)), ("170.7", ("q96",)), ("170.W1", ("m96", "q96"))):
        if not all(set_passed.get(group, True) for group in needed):
            {"170.1": p1701, "170.2": p1702, "170.7": p1707, "170.W1": pW1}[name].update({"pass": None, "set_stopped": True})

    pW2 = _prediction_w2(rows, weights)

    # Exploratory Q96 entropy halves have their own within-set paired rates, never a cross-set rate contrast.
    listmap = {str(row["prompt_id"]): row for row in lists}
    qeligible = sorted(pid for pid in q96_ids if pid in listmap and stage0.get("set_passed", {}).get("q96", True))
    entropy = {pid: float(listmap[pid]["entropy"]) for pid in qeligible}
    entropy_median = float(np.median(list(entropy.values()))) if entropy else None
    entropy_split = {}
    for label, test in (("lower_or_equal", lambda value: value <= entropy_median),
                        ("higher", lambda value: value > entropy_median)):
        ids = {pid for pid, value in entropy.items() if test(value)} if entropy_median is not None else set()
        pair = _pair_no_push(rows, ["q96"], "b50", ids)
        base, alt, rise = _mean_rise(pair)
        boots = _prompt_draws(pair, rng)
        entropy_split[label] = {**_interval(boots["rise"], rise), "rows": len(pair),
            "b0_rate": base, "b50_rate": alt, "rise": rise,
            "entropy_min": min((entropy[pid] for pid in ids), default=None),
            "entropy_max": max((entropy[pid] for pid in ids), default=None)}

    shares_by_dose = {}
    for dose in DOSES:
        b0_rows = [row for row in rows if row.get("stage") == "push" and not row.get("repeat") and
                   row.get("ban") == "b0" and float(row.get("dose", -1)) == dose]
        shares_by_dose[str(dose)] = _share_record(b0_rows, listmap)
    other_masses = [float(listmap[pid].get("other50_mass", 0.0)) for pid in q96_ids if
                    listmap.get(pid, {}).get("other50_mass") is not None]
    return {
        "170.1": _with_noise(p1701, noise),
        "170.2": _with_noise(p1702, noise),
        "170.3": _with_noise(p1703, noise),
        "170.4": _with_noise(p1704, noise),
        "170.5": _with_noise(p1705, noise),
        "170.6": _with_noise(p1706, noise),
        "170.7": _with_noise(p1707, noise),
        "170.8": _with_noise(p1708, noise),
        "170.W1": _with_noise(pW1, noise),
        "170.W2": _with_noise(pW2, noise),
        "exploratory": {"q96_plain_entropy_median": entropy_median,
            "q96_b50_entropy_split": entropy_split,
            "q96_entropy_rows": len(qeligible),
            "other50_mass_mean": sum(other_masses) / len(other_masses) if other_masses else None,
            "other50_mass_rows": len(other_masses),
            "mean_share_by_dose": shares_by_dose,
            "rel50_blocked_step_share": {"value": p1706.get("rel50_blocked_step_share"),
                "rows": p1706.get("rows"), "mean_blocked_steps": p1706.get("mean_blocked_steps")}},
    }


def score(answers_path, stage0_path, lists_path, norms_path, out_dir):
    answers = run170.read_jsonl(answers_path)
    stage0 = read_json(stage0_path, {}) or {}
    lists = run170.read_jsonl(lists_path)
    norms = read_json(norms_path, {"directions": [], "prior_neurons": {}}) or {"directions": [], "prior_neurons": {}}
    prompts = run170.prompt_catalog()
    gates = run170.gate_rules(prompts, answers, stage0, lists, norms,
        expected_answers=len(prompts) + 6_472, expected_directions=len(LAYERS) * SLOTS * len(DOSES))
    stopped = not bool(gates["passed"])
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "hand_read").mkdir(parents=True, exist_ok=True)
    (out / "hand_key").mkdir(parents=True, exist_ok=True)
    evaluated = outcomes(answers)
    predictions = {} if stopped else _score_predictions(evaluated, prompts, stage0, lists, gates)
    frozen = set(map(str, stage0.get("frozen_prompt_ids", [])))
    eligible = {str(prompt["prompt_id"]) for prompt in prompts if str(prompt["prompt_id"]) not in frozen}
    set_passed = stage0.get("set_passed", {})
    groups_read = [group for group in ("p8", "m96", "q96") if set_passed.get(group, True)]
    eligible = {pid for pid in eligible if pid.split(":")[0] in groups_read}
    eligible_by_group = {group: {pid for pid in eligible if pid.startswith(group + ":")}
                         for group in ("p8", "m96", "q96")}
    if stopped:
        readings = {"per_cell": [], "paired_changes": [],
                    "exploratory_other50_mass": {"mean": None, "rows": 0}}
    else:
        readings = {
            "per_cell": _cell_summaries(evaluated, eligible),
            "paired_changes": sum((_paired_changes(evaluated, [group],
                        eligible_ids=eligible_by_group[group])
                        for group in groups_read), []) +
                _paired_changes(evaluated, ["p8"], push_dose=0.25,
                                eligible_ids=eligible_by_group["p8"]) +
                _paired_changes(evaluated, ["p8"], push_dose=0.5,
                                eligible_ids=eligible_by_group["p8"]) +
                _paired_changes(evaluated, ["p8"], push_dose=1.0,
                                eligible_ids=eligible_by_group["p8"]),
            "exploratory_other50_mass": {
                "mean": (sum(float(row.get("other50_mass", 0)) for row in lists
                              if row.get("group") == "q96" and
                              str(row.get("prompt_id")) in eligible_by_group["q96"]) /
                         max(1, sum(row.get("group") == "q96" and
                                    str(row.get("prompt_id")) in eligible_by_group["q96"]
                                    for row in lists))),
                "rows": sum(row.get("group") == "q96" and
                            str(row.get("prompt_id")) in eligible_by_group["q96"] for row in lists)}}
    # Checks that decide nothing: is P8's b50 Part 169's plain50, and how often does no-push b0 leave the single-row plain answer?
    p169 = ROOT / "results/part169/masses.jsonl"
    listing = {str(row["prompt_id"]): row for row in lists}
    same = None
    if p169.exists():
        old = {}
        for row in run170.read_jsonl(p169):
            old.setdefault(int(row["question_index"]), row["lists"]["plain50"]["ids"])
        same = sum(sorted(listing.get(f"p8:{qi}", {}).get("b50", [])) == sorted(ids) for qi, ids in old.items())
    plain_ids = {str(row.get("prompt_id")): row.get("token_ids") for row in answers if row.get("stage") == "stage0"}
    b0_rows = [row for row in answers if row.get("stage") == "no_push" and row.get("ban") == "b0"]
    readings["diagnostics"] = {"p8_b50_equal_to_part169_plain50": same,
        "no_push_b0_differs_from_plain": sum(row.get("token_ids") != plain_ids.get(str(row.get("prompt_id"))) for row in b0_rows),
        "no_push_b0_rows": len(b0_rows)}
    result = {"gates": gates, "readings_stopped": stopped,
        "prompt_sets": {"p8": {"rows": 8, "eligible": 8 - len(stage0.get("frozen_indices", {}).get("p8", []))},
            "m96": {"rows": 96, "eligible": 96 - len(stage0.get("frozen_indices", {}).get("m96", []))},
            "q96": {"rows": 96, "eligible": 96 - len(stage0.get("frozen_indices", {}).get("q96", []))}},
        "plain_lengths_frozen_vs_surviving": stage0.get("plain_lengths", {}),
        "predictions": predictions, **readings,
        "bootstrap": {"push": {"method": "whole-slot, stratified within layer", "slots_per_layer": SLOTS,
                                "layers": list(LAYERS), "replicates": REPS, "seed": SEED},
                      "no_push": {"method": "prompt resampling", "replicates": REPS, "seed": SEED}}}
    (out / "score.json").write_text(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
                                      encoding="utf-8")
    if not stopped:
        sample, key = _blind_hand_sample(evaluated)
        _write_jsonl(out / "hand_read" / "answers.jsonl", sample)
        _write_jsonl(out / "hand_key" / "labels.jsonl", key)
    else:
        _write_jsonl(out / "hand_read" / "answers.jsonl", [])
        _write_jsonl(out / "hand_key" / "labels.jsonl", [])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default="results/part170/answers.jsonl")
    parser.add_argument("--stage0", default="results/part170/stage0.json")
    parser.add_argument("--lists", default="results/part170/lists.jsonl")
    parser.add_argument("--norms", default="results/part170/norms.json")
    parser.add_argument("--out", default="results/part170")
    args = parser.parse_args(argv)
    result = score(args.answers, args.stage0, args.lists, args.norms, args.out)
    print(json.dumps({"gates": result["gates"], "readings_stopped": result["readings_stopped"],
                      "predictions": result["predictions"]}, ensure_ascii=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
