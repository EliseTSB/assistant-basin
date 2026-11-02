#!/usr/bin/env python3
"""Null-aware Part 176 scoring, collapse curves, transition tables, and predictions."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part171"), str(ROOT / "part172"),
              str(ROOT / "part173"), str(ROOT / "part174"), str(ROOT / "part175")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import label173
import names173
import run171
import run174
import run175
import score173
import score174
import score175
from part176 import run176

REPS = 10_000
SEED = 176
STATES = ("qwen", "google", "other name", "claim with no single name", "reverted",
          "longer no claim", "fragment", "broken", "unlabelled")
PREDICTION_ROWS = {
    "176.1": {"prediction": "At 0.225, at least 20 of the 40 signed directions are supported",
              "confidence": "medium", "author": "Claude"},
    "176.2": {"prediction": "At 0.225, Google is modal on at least 80% of supported directions; not read below 5 supported",
              "confidence": "high", "author": "Claude"},
    "176.3": {"prediction": "At 0.225, at most 2 of the 40 signed directions are other-led",
              "confidence": "high", "author": "Claude"},
    "176.4": {"prediction": "At least 8 of 40 are supported at 0.20 and at least 8 of 40 at 0.25",
              "confidence": "medium", "author": "Claude"},
    "176.5": {"prediction": "At 0.225, the median broken of the 40 signed directions is at most 8 of 44",
              "confidence": "high", "author": "Claude"},
    "176.6": {"prediction": "Pooled Google share at 0.275 is at least its share at 0.225, and named answers at 0.275 are under half those at 0.225; not read below 100 named at 0.275",
              "confidence": "medium", "author": "Claude"},
    "176.7": {"prediction": "Over eligible directions, Spearman of Google count of 44 at 0.225 against collapse dose is above -0.30; fails at or below -0.30 with cluster-bootstrap upper bound below 0",
              "confidence": "low", "author": "Claude"},
    "176.W1": {"prediction": "Support rises through 0.25 and falls under 5 of 40 at 0.30, with 0.225 on the rising slope rather than at a peak",
               "confidence": "medium", "author": "Weaver"},
    "176.W2": {"prediction": "Google share rises monotonically across the rungs while the raw Google count peaks at or below 0.25",
               "confidence": "medium", "author": "Weaver"},
    "176.W3": {"prediction": "Among directions with a supported rung, median gap from top supported rung to collapse dose is at least 0.05",
               "confidence": "medium", "author": "Weaver"},
}


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def read_jsonl(path):
    return run171.read_jsonl(path) if Path(path).is_file() else []


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n"
                                  for row in rows), encoding="utf-8", newline="\n")


def _word_set(text):
    return set(re.findall(r"[^\W_]+", str(text or "").casefold(), flags=re.UNICODE))


def _own_name(text):
    words = _word_set(text)
    return bool(words & {"qwen", "tongyi", "alibaba"})


def _token_count(row):
    try:
        return int(row.get("token_count", len(row.get("token_ids") or [])))
    except (TypeError, ValueError):
        return None


def _valid_record(record):
    return bool(record and label173.valid(record))


def answer_state(row, record, fragment_limit=12):
    """Assign exactly one transition state, with label failures explicit."""
    if row.get("broken"):
        return "broken"
    if not _valid_record(record):
        return "unlabelled"
    if record.get("relation") == "claim":
        primary = score173._primary_entity(record, mode="canonical")
        if primary == names173.QWEN_ENTITY:
            return "qwen"
        if primary == "google":
            return "google"
        if primary is not None:
            return "other name"
        return "claim with no single name"
    if record.get("relation") == "none":
        if _own_name(row.get("answer", "")):
            return "reverted"
        count = _token_count(row)
        if count is not None and count <= int(fragment_limit):
            return "fragment"
        return "longer no claim"
    return "unlabelled"


def _jaccard(question, answer):
    left, right = _word_set(question), _word_set(answer)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _interpolate_crossing(values, threshold, doses):
    """First crossing, interpolated on its bracket; censored values keep their boundary."""
    if len(values) != len(doses) or any(value is None or not math.isfinite(float(value)) for value in values):
        return {"status": "not read", "L": None, "bracket": None,
                "ordinal_rank": None, "reason": "crossing inputs are incomplete or nonfinite"}
    vals = [float(value) for value in values]
    if vals[0] >= threshold:
        return {"status": "left censored", "L": float(doses[0]),
                "bracket": [None, float(doses[0])], "ordinal_rank": 0,
                "threshold": float(threshold)}
    for index in range(1, len(vals)):
        if vals[index] >= threshold:
            x0, x1 = float(doses[index - 1]), float(doses[index])
            y0, y1 = vals[index - 1], vals[index]
            cross = (x1 if y1 == y0 else x0 + (float(threshold) - y0) * (x1 - x0) / (y1 - y0))
            return {"status": "crossed", "L": float(cross), "bracket": [x0, x1],
                    "ordinal_rank": index, "threshold": float(threshold)}
    return {"status": "right censored", "L": float(doses[-1]),
            "bracket": [float(doses[-1]), None], "ordinal_rank": len(doses),
            "threshold": float(threshold), "lower_bound": float(doses[-1])}


def _crossing_bounds(bounds, threshold, doses):
    low = [int(pair[0]) for pair in bounds]
    high = [int(pair[1]) for pair in bounds]
    earliest = _interpolate_crossing(high, threshold, doses)
    latest = _interpolate_crossing(low, threshold, doses)
    if earliest["status"] == latest["status"] and earliest.get("L") == latest.get("L"):
        return {**latest, "determined": True, "interval": None,
                "bounds": {"low_counts": low, "high_counts": high}}
    if earliest["status"] == "right censored" and latest["status"] == "right censored":
        return {**latest, "determined": True, "interval": None,
                "bounds": {"low_counts": low, "high_counts": high}}
    interval = [None if earliest["status"] == "left censored" else earliest.get("L"),
                None if latest["status"] == "right censored" else latest.get("L")]
    if interval[0] is not None and interval[1] is not None and interval[0] > interval[1]:
        interval.reverse()
    return {"status": "interval", "L": None, "bracket": [earliest.get("bracket"), latest.get("bracket")],
            "ordinal_rank": None, "determined": False, "interval": interval,
            "reason": "unlabelled short answer can change the first crossing",
            "possible_crossing": earliest, "certain_crossing": latest,
            "bounds": {"low_counts": low, "high_counts": high}}


def _crossings_at_values(values, lower, upper, threshold, doses):
    return _crossing_bounds([[lower(row), upper(row)] for row in values], threshold, doses)


def _crossing_width(values, low_threshold=11, high_threshold=33, doses=run176.DOSES):
    low = _interpolate_crossing(values, low_threshold, doses)
    high = _interpolate_crossing(values, high_threshold, doses)
    if low["status"] != "crossed" or high["status"] != "crossed":
        return {"width": None, "at_11": low, "at_33": high}
    width = float(high["L"] - low["L"])
    return {"width": width, "at_11": low, "at_33": high}


def _rank_average(values):
    return score174._rankdata(values)


def _rho(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 3 or xs.size != ys.size or not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        return None
    rx, ry = _rank_average(xs), _rank_average(ys)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    value = float(np.corrcoef(rx, ry)[0, 1])
    return value if math.isfinite(value) else None


def _cluster_spearman(rows, x_key, y_key, reps=REPS, seed=SEED):
    chosen = [row for row in rows if row.get(x_key) is not None and row.get(y_key) is not None]
    xs = [float(row[x_key]) for row in chosen]
    ys = [float(row[y_key]) for row in chosen]
    value = _rho(xs, ys)
    base = {"n": len(chosen), "rho": value, "interval_95": [None, None],
            "replicates": int(reps), "valid_replicates": 0,
            "cluster": "draw seed, both signs kept together"}
    if value is None:
        base.update({"status": "not read", "reason": "Spearman undefined: too few observations or no spread"})
        return base
    by_seed = defaultdict(list)
    for row in chosen:
        by_seed[int(row["seed"])].append(row)
    seeds = sorted(by_seed)
    if len(seeds) < 2:
        base.update({"status": "not read", "reason": "fewer than two draw clusters"})
        return base
    rng = np.random.default_rng(int(seed))
    boot = []
    for _ in range(int(reps)):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        sample_rows = [row for group in sampled for row in by_seed[int(group)]]
        rho = _rho([row[x_key] for row in sample_rows], [row[y_key] for row in sample_rows])
        if rho is not None:
            boot.append(rho)
    if not boot:
        base.update({"status": "not read", "reason": "no finite cluster-bootstrap correlations"})
        return base
    base.update({"status": "read", "valid_replicates": len(boot),
                 "interval_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                 "seed": int(seed)})
    return base


def _direction(rows, reader_by_row, set_name, sign, doses, expected=44, fragment_limit=12):
    rows = list(rows)
    by_dose = {float(value): [] for value in doses}
    for row in rows:
        by_dose.setdefault(abs(float(row["dose"])), []).append(row)
    rung_rows = []
    base_rows = []
    for magnitude in doses:
        dose = float(magnitude) * (1 if sign > 0 else -1)
        arm = run175.dose_arm(set_name, dose)
        raw = [row for row in by_dose.get(float(magnitude), []) if float(row.get("dose", 0)) == dose]
        scored = score173.score_rows(raw)
        if len(scored) != int(expected):
            raise ValueError(f"{arm} has {len(scored)} rows; expected {expected}")
        base = score175._direction(scored, reader_by_row, set_name, dose, expected=expected)
        base_rows.append((base, scored))
        states = Counter()
        no_claim, named, google = 0, 0, 0
        fragment = reverted = undetermined = 0
        raw_google = 0
        token_counts, repeats = [], []
        for row in scored:
            record = reader_by_row.get((str(row.get("arm")), str(row.get("prompt_id"))))
            state = answer_state(row, record, fragment_limit=fragment_limit)
            states[state] += 1
            raw_google += int(row.get("raw_word_hits", {}).get("google", 0))
            token_count = _token_count(row)
            if token_count is not None:
                token_counts.append(token_count)
            repeats.append(_jaccard(row.get("question", ""), row.get("answer", "")) > 0.7)
            if state == "fragment":
                fragment += 1
            if state == "reverted":
                reverted += 1
            if state == "unlabelled" and row.get("clean") and token_count is not None and \
                    token_count <= int(fragment_limit) and not _own_name(row.get("answer", "")):
                undetermined += 1
            if _valid_record(record) and row.get("clean"):
                if record.get("relation") == "none":
                    no_claim += 1
                elif record.get("relation") == "claim":
                    entity = score173._primary_entity(record, mode="canonical")
                    if entity is not None:
                        named += 1
                        google += int(entity == "google")
        exact = base.get("reader_state") == "complete"
        broken = int(base["broken"])
        missing_clean = int(base.get("reader_unlabelled_clean_answers", 0))
        failed_bounds = [broken + fragment, broken + fragment + undetermined]
        if exact:
            failed_bounds[1] = failed_bounds[0]
        lost_bounds = [broken + no_claim, broken + no_claim + missing_clean]
        if exact:
            lost_bounds[1] = lost_bounds[0]
        named_bounds = [int(named), int(named) + missing_clean]
        google_count_bounds = [int(google), min(int(google) + missing_clean, named_bounds[1])]
        if named_bounds[1] == 0:
            google_share_bounds = [None, None]
        else:
            share_low = google_count_bounds[0] / named_bounds[1]
            if named_bounds[0] > 0:
                share_high = min(1.0, google_count_bounds[1] / named_bounds[0])
            else:
                share_high = 1.0 if google_count_bounds[1] > 0 else 0.0
            google_share_bounds = [share_low, share_high]
        if not exact:
            no_claim_value = named_value = google_value = google_share = None
            reverted_value = None
        else:
            no_claim_value, named_value, google_value = int(no_claim), int(named), int(google)
            google_share = google / named if named >= 11 else None
            reverted_value = int(reverted)
        count_denominator = len(scored)
        rung_rows.append({
            "dose": float(magnitude), "signed_dose": dose, "arm": arm,
            "answers": count_denominator, "broken": broken, "clean": int(base["clean"]),
            "reader_state": base["reader_state"], "reader_unlabelled_clean_answers": missing_clean,
            "no_claim": no_claim_value, "observed_no_claim": int(no_claim),
            "named": named_value, "observed_named": int(named),
            "named_bounds": named_bounds,
            "google_count": google_value, "observed_google_count": int(google),
            "google_count_bounds": google_count_bounds,
            "google_over_44": google_value / count_denominator if exact and count_denominator else None,
            "google_over_named": google_share,
            "google_over_named_bounds": google_share_bounds,
            "named_over_44": named_value / count_denominator if exact and count_denominator else None,
            "fragment": int(fragment) if exact else None,
            "observed_fragment": int(fragment),
            "undetermined_short": int(undetermined),
            "reverted": reverted_value, "observed_reverted": int(reverted),
            "failed": failed_bounds[0] if failed_bounds[0] == failed_bounds[1] else None,
            "failed_bounds": failed_bounds,
            "lost": lost_bounds[0] if lost_bounds[0] == lost_bounds[1] else None,
            "lost_bounds": lost_bounds,
            "raw_google_word_hits": int(raw_google),
            "state_counts": {state: int(states.get(state, 0)) for state in STATES},
            "median_token_count_all_answers": float(np.median(token_counts)) if token_counts else None,
            "token_count_values": token_counts,
            "question_jaccard_over_0_7_count": int(sum(repeats)),
            "question_jaccard_denominator": len(repeats),
            "question_jaccard_over_0_7_share_all_answers": float(np.mean(repeats)) if repeats else None,
            "support": base.get("supported"), "support_status": base.get("support_status"),
            "support_bounds": base.get("support_bounds"),
            "google_led": base.get("google_led"), "other_led": base.get("other_led"),
            "google_led_bounds": base.get("google_led_bounds"),
            "other_led_bounds": base.get("other_led_bounds"),
            "modal": base.get("modal"), "summary": base.get("summary"),
        })
    failed_bounds = [row["failed_bounds"] for row in rung_rows]
    main_l = _crossing_bounds(failed_bounds, 22, doses)
    support_indices = [index for index, row in enumerate(rung_rows) if row.get("support") is True]
    top_supported = float(doses[max(support_indices)]) if support_indices else None
    uncertain_support_after = any(row.get("support_status") in ("unknown", "bounded")
                                  for row in rung_rows[(max(support_indices) + 1 if support_indices else 0):])
    if top_supported is None:
        gap, gap_reason = None, "no definitely supported rung"
    elif not main_l.get("determined"):
        gap, gap_reason = None, "collapse dose is interval-valued"
    elif uncertain_support_after:
        gap, gap_reason = None, "a higher rung has undetermined support"
    else:
        gap, gap_reason = float(main_l["L"] - top_supported), None
    cross11 = _crossing_bounds(failed_bounds, 11, doses)
    cross33 = _crossing_bounds(failed_bounds, 33, doses)
    width = ({"width": float(cross33["L"] - cross11["L"]), "at_11": cross11, "at_33": cross33}
             if cross11.get("determined") and cross33.get("determined") and
             cross11.get("status") == "crossed" and cross33.get("status") == "crossed"
             else {"width": None, "at_11": cross11, "at_33": cross33,
                   "reason": "either threshold is censored or reader uncertainty changes a crossing"})
    sensitivity = {}
    for label, threshold, metric in (("failed_15", 15, "failed_bounds"),
                                     ("failed_22", 22, "failed_bounds"),
                                     ("failed_29", 29, "failed_bounds"),
                                     ("lost_22", 22, "lost_bounds")):
        sensitivity[label] = _crossing_bounds([row[metric] for row in rung_rows], threshold, doses)
    for limit in (8, 20):
        alt_rungs = []
        for base, scored_rows in base_rows:
            frag = 0
            unknown_short = 0
            for row in scored_rows:
                record = reader_by_row.get((str(row.get("arm")), str(row.get("prompt_id"))))
                state = answer_state(row, record, fragment_limit=limit)
                frag += int(state == "fragment")
                unknown_short += int(state == "unlabelled" and row.get("clean") and
                                     (_token_count(row) is not None and _token_count(row) <= limit) and
                                     not _own_name(row.get("answer", "")))
            alt_rungs.append([int(base["broken"]) + frag,
                              int(base["broken"]) + frag + unknown_short])
        sensitivity[f"fragment_{limit}_failed_22"] = _crossing_bounds(alt_rungs, 22, doses)
    recrossings = None
    if main_l.get("determined"):
        recrossings = 0
        first = next((index for index, pair in enumerate(failed_bounds) if pair[0] >= 22), None)
        if first is not None:
            recrossings = sum(failed_bounds[index][1] < 22 for index in range(first + 1, len(failed_bounds)))
    return {"set": set_name, "seed": int(set_name.rsplit("d", 1)[1]), "sign": int(sign),
            "answers": [row[1] for row in base_rows], "rungs": rung_rows,
            "collapse_dose": main_l, "ordinal_L": main_l.get("ordinal_rank"),
            "crossing_bracket": main_l.get("bracket"), "recrossing_count": recrossings,
            "transition_width_11_to_33": width,
            "L_sensitivities": sensitivity, "top_supported_rung": top_supported,
            "gap": gap, "gap_reason": gap_reason,
            "undetermined_crossing": not bool(main_l.get("determined")),
            "undetermined_reason": main_l.get("reason") if not main_l.get("determined") else None}


def _rung_aggregate(directions, index, expected_directions, reps=REPS, seed=SEED):
    rungs = [direction["rungs"][index] for direction in directions]
    complete = all(row["reader_state"] == "complete" for row in rungs) and len(rungs) == expected_directions
    states = Counter()
    for row in rungs:
        states.update(row["state_counts"])
    counts = {key: (sum(int(row[key]) for row in rungs)
                    if complete and all(row.get(key) is not None for row in rungs) else None)
              for key in ("no_claim", "named", "google_count", "fragment", "reverted", "failed", "lost")}
    counts["broken"] = sum(int(row["broken"]) for row in rungs)
    counts["state_counts"] = {state: int(states.get(state, 0)) for state in STATES}
    total_answers = sum(int(row["answers"]) for row in rungs)
    counts["answers"] = total_answers
    named, google = counts["named"], counts["google_count"]
    counts["rates"] = {
        "google_over_44": google / total_answers if google is not None and total_answers else None,
        "google_over_named": google / named if google is not None and named else None,
        "named_over_44": named / total_answers if named is not None and total_answers else None,
    }
    observed_named = sum(int(row["observed_named"]) for row in rungs)
    missing_clean = sum(int(row["reader_unlabelled_clean_answers"]) for row in rungs)
    observed_google = sum(int(row["observed_google_count"]) for row in rungs)
    counts["named_bounds"] = [sum(row["named_bounds"][0] for row in rungs),
                               sum(row["named_bounds"][1] for row in rungs)]
    counts["google_count_bounds"] = [sum(row["google_count_bounds"][0] for row in rungs),
                                     sum(row["google_count_bounds"][1] for row in rungs)]
    counts["google_share_bounds"] = (
        [counts["google_count_bounds"][0] / counts["named_bounds"][1],
         min(1.0, counts["google_count_bounds"][1] / counts["named_bounds"][0])]
        if counts["named_bounds"][1] > 0 and counts["named_bounds"][0] > 0 else
        [counts["google_count_bounds"][0] / counts["named_bounds"][1],
         1.0 if counts["google_count_bounds"][1] > 0 else 0.0]
        if counts["named_bounds"][1] > 0 else [None, None])
    counts["rates"].update({
        "google_over_named_bounds": counts["google_share_bounds"],
        "google_over_44_bounds": [value / total_answers for value in counts["google_count_bounds"]]
        if total_answers else [None, None],
        "named_over_44_bounds": [value / total_answers for value in counts["named_bounds"]]
        if total_answers else [None, None],
    })
    counts["failed_bounds"] = [sum(row["failed_bounds"][0] for row in rungs),
                               sum(row["failed_bounds"][1] for row in rungs)]
    counts["lost_bounds"] = [sum(row["lost_bounds"][0] for row in rungs),
                             sum(row["lost_bounds"][1] for row in rungs)]
    counts["failed"] = (counts["failed_bounds"][0]
                        if counts["failed_bounds"][0] == counts["failed_bounds"][1] else None)
    counts["lost"] = (counts["lost_bounds"][0]
                      if counts["lost_bounds"][0] == counts["lost_bounds"][1] else None)
    counts["support"] = {"lower": sum(row["support_bounds"][0] for row in rungs),
                          "upper": sum(row["support_bounds"][1] for row in rungs),
                          "denominator": len(rungs)}
    counts["google_led"] = {"lower": sum(row.get("google_led") is True for row in rungs),
                             "upper": sum(row.get("google_led") is not False for row in rungs),
                             "denominator": len(rungs)}
    counts["other_led"] = {"lower": sum(row.get("other_led") is True for row in rungs),
                            "upper": sum(row.get("other_led") is not False for row in rungs),
                            "denominator": len(rungs)}
    uncertainty_rows = [{"seed": direction["seed"],
                         "dose": direction["rungs"][index]["signed_dose"],
                         "support_bounds": direction["rungs"][index]["support_bounds"],
                         "google_led_bounds": direction["rungs"][index].get("google_led_bounds", [0, 1]),
                         "other_led_bounds": direction["rungs"][index].get("other_led_bounds", [0, 1])}
                        for direction in directions]
    counts["intervals"] = score175._metric_uncertainty(
        uncertainty_rows, reps=reps, seed=seed + index)
    token_values = [value for row in rungs for value in row["token_count_values"]]
    counts["median_token_count_all_answers"] = float(np.median(token_values)) if token_values else None
    repeat_denominator = sum(row["question_jaccard_denominator"] for row in rungs)
    counts["question_jaccard_over_0_7_share_all_answers"] = (
        sum(row["question_jaccard_over_0_7_count"] for row in rungs) / repeat_denominator
        if repeat_denominator else None)
    counts["raw_google_word_hits"] = sum(row["raw_google_word_hits"] for row in rungs)
    return counts


def _transition(states_from, states_to):
    table = {source: {target: 0 for target in STATES} for source in STATES}
    for source, target in zip(states_from, states_to):
        table[source][target] += 1
    return {"from": {key: int(value) for key, value in Counter(states_from).items()},
            "to": {key: int(value) for key, value in Counter(states_to).items()},
            "matrix": table, "observations": len(states_from),
            "google_to_failed": int(sum(source == "google" and target in ("broken", "fragment")
                                         for source, target in zip(states_from, states_to))),
            }


def transition_tables(raw_answers, scored, reader_by_row, directions, doses, expected=44):
    raw_by_arm_pid = {(str(row.get("arm")), str(row.get("prompt_id"))): row for row in raw_answers}
    score_by_key = {(str(row.get("arm")), str(row.get("prompt_id"))): row for row in scored}
    tables = {}
    for rung_index in range(len(doses) - 1):
        lower, upper = float(doses[rung_index]), float(doses[rung_index + 1])
        all_from, all_to = [], []
        by_seed_sign = {}
        for direction in directions:
            set_name, sign = direction["set"], direction["sign"]
            from_arm = run175.dose_arm(set_name, lower * sign)
            to_arm = run175.dose_arm(set_name, upper * sign)
            left, right = [], []
            for prompt_id in sorted({pid for arm, pid in score_by_key if arm == from_arm}):
                a = score_by_key.get((from_arm, prompt_id))
                b = score_by_key.get((to_arm, prompt_id))
                if a is None or b is None:
                    continue
                left.append(answer_state(a, reader_by_row.get((from_arm, prompt_id))))
                right.append(answer_state(b, reader_by_row.get((to_arm, prompt_id))))
            all_from.extend(left)
            all_to.extend(right)
            by_seed_sign[f"{direction['seed']}:{sign}"] = _transition(left, right)
        tables[f"{lower:g}->{upper:g}"] = {"pooled": _transition(all_from, all_to),
                                             "by_draw_and_sign": by_seed_sign,
                                             "from_dose": lower, "to_dose": upper,
                                             "states": list(STATES)}
    trajectories = []
    for direction in directions:
        by_prompt = defaultdict(dict)
        for index, group in enumerate(direction.get("answers", [])):
            for row in group:
                state = answer_state(row, reader_by_row.get((str(row.get("arm")),
                                                              str(row.get("prompt_id")))))
                by_prompt[str(row.get("prompt_id"))][index] = state
        for prompt_id, states in by_prompt.items():
            if len(states) != len(doses):
                continue
            sequence = [states[index] for index in range(len(doses))]
            trajectories.append({"seed": direction["seed"], "sign": direction["sign"],
                                 "prompt_id": prompt_id, "states": sequence,
                                 "ended_failed": sequence[-1] in ("broken", "fragment"),
                                 "ever_google": "google" in sequence,
                                 "ended_failed_after_google": (sequence[-1] in ("broken", "fragment")
                                                                and "google" in sequence)})
    by_draw = defaultdict(list)
    for row in trajectories:
        by_draw[f"{row['seed']}:{row['sign']}"].append(row)
    def trajectory_counts(rows):
        ended = [row for row in rows if row["ended_failed"]]
        after_google = [row for row in ended if row["ever_google"]]
        return {"prompt_trajectories": len(rows), "ended_failed": len(ended),
                "ever_google": sum(row["ever_google"] for row in rows),
                "ended_failed_after_google": len(after_google),
                "failed_end_after_google_share": len(after_google) / len(ended) if ended else None}
    tables["trajectory_summary"] = {"pooled": trajectory_counts(trajectories),
                                    "by_draw_and_sign": {key: trajectory_counts(value)
                                                         for key, value in by_draw.items()},
                                    "rows": trajectories}
    return tables


def _paired_shares(directions, left_index, right_index, reps=REPS, seed=SEED):
    pairs = []
    for direction in directions:
        left, right = direction["rungs"][left_index], direction["rungs"][right_index]
        if left["named"] is not None and right["named"] is not None and \
                left["named"] >= 11 and right["named"] >= 11 and \
                left["google_over_named"] is not None and right["google_over_named"] is not None:
            pairs.append({"seed": direction["seed"], "sign": direction["sign"],
                          "left_share": left["google_over_named"],
                          "right_share": right["google_over_named"],
                          "change": right["google_over_named"] - left["google_over_named"]})
    if not pairs:
        return {"eligible_directions": 0, "rose": 0, "fell": 0, "equal": 0,
                "median_change": None, "interval_95": [None, None], "pairs": []}
    changes = np.asarray([row["change"] for row in pairs], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    values = [float(np.median(changes[rng.integers(0, len(changes), len(changes))]))
              for _ in range(int(reps))]
    return {"eligible_directions": len(pairs), "rose": int(np.sum(changes > 0)),
            "fell": int(np.sum(changes < 0)), "equal": int(np.sum(changes == 0)),
            "median_change": float(np.median(changes)),
            "interval_95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
            "pairs": pairs}


def _split_half(direction, reader_by_row, doses, expected=44):
    output = {}
    for label, parity in (("odd", 1), ("even", 0)):
        half_rungs = []
        for magnitude in doses:
            dose = float(magnitude) * direction["sign"]
            arm = run175.dose_arm(direction["set"], dose)
            selected = [row for group in direction["answers"] for row in group
                        if str(row.get("arm")) == arm and
                        (int(re.search(r"(\d+)$", str(row.get("prompt_id"))).group(1)) % 2 == parity)]
            failed, unknown = 0, 0
            for row in selected:
                record = reader_by_row.get((str(row.get("arm")), str(row.get("prompt_id"))))
                state = answer_state(row, record)
                failed += int(state in ("broken", "fragment"))
                unknown += int(state == "unlabelled" and row.get("clean") and
                               (_token_count(row) or 999) <= 12 and not _own_name(row.get("answer", "")))
            half_rungs.append([failed, failed + unknown])
        crossing = _crossing_bounds(half_rungs, 11, doses)
        half_l = crossing.get("L") if crossing.get("status") == "crossed" and crossing.get("determined") else None
        output[label] = {"L": half_l, "status": crossing.get("status"),
                         "determined": crossing.get("determined"), "bracket": crossing.get("bracket"),
                         "google_count_0225": None, "google_share_0225": None,
                         "rung_count": 22}
        mid = min(range(len(doses)), key=lambda index: abs(float(doses[index]) - 0.225))
        selected = [row for group in direction["answers"] for row in group
                    if str(row.get("arm")) == run175.dose_arm(direction["set"], 0.225 * direction["sign"]) and
                    (int(re.search(r"(\d+)$", str(row.get("prompt_id"))).group(1)) % 2 == parity)]
        entities = Counter()
        for row in selected:
            record = reader_by_row.get((str(row.get("arm")), str(row.get("prompt_id"))))
            if _valid_record(record) and row.get("clean") and record.get("relation") == "claim":
                entity = score173._primary_entity(record, mode="canonical")
                if entity is not None:
                    entities[entity] += 1
        named = sum(entities.values())
        google = int(entities.get("google", 0))
        output[label]["google_count_0225"] = google if len(selected) == 22 and all(
            _valid_record(reader_by_row.get((str(row.get("arm")), str(row.get("prompt_id")))))
            or row.get("broken") for row in selected) else None
        output[label]["google_share_0225"] = google / named if named >= 11 else None
    return output


def _split_half_summary(directions, reader_by_row, doses, reps=REPS, seed=SEED, expected=44):
    halves = [_split_half(direction, reader_by_row, doses, expected=expected) for direction in directions]
    rows = []
    for direction, half in zip(directions, halves):
        rows.append({"seed": direction["seed"], "sign": direction["sign"], **half})
    result = {"directions": rows, "crossings_at_11_of_22": {"odd": 0, "even": 0},
              "excluded_censored_or_null_halves": {"odd": 0, "even": 0}, "spearmans": {}}
    for half_name in ("odd", "even"):
        values = [row[half_name] for row in rows]
        result["crossings_at_11_of_22"][half_name] = sum(
            bool(item.get("determined") and item.get("status") == "crossed") for item in values)
        result["excluded_censored_or_null_halves"][half_name] = sum(
            not item.get("determined") or item.get("status") != "crossed" or item.get("L") is None
            for item in values)
    for metric in ("google_count_0225", "google_share_0225"):
        result["spearmans"][metric] = {
            "odd_against_even": {"rho": _rho([row["odd"][metric] for row in rows if
                                                    row["odd"][metric] is not None and row["even"][metric] is not None],
                                                [row["even"][metric] for row in rows if
                                                    row["odd"][metric] is not None and row["even"][metric] is not None]),
                                  "n": sum(row["odd"][metric] is not None and row["even"][metric] is not None
                                           for row in rows)},
            "odd_measure_against_odd_L": {"rho": _rho([row["odd"][metric] for row in rows if
                                                             row["odd"][metric] is not None and row["odd"]["L"] is not None],
                                                         [row["odd"]["L"] for row in rows if
                                                             row["odd"][metric] is not None and row["odd"]["L"] is not None]),
                                           "n": sum(row["odd"][metric] is not None and row["odd"]["L"] is not None
                                                    for row in rows)},
            "even_measure_against_even_L": {"rho": _rho([row["even"][metric] for row in rows if
                                                               row["even"][metric] is not None and row["even"]["L"] is not None],
                                                           [row["even"]["L"] for row in rows if
                                                               row["even"][metric] is not None and row["even"]["L"] is not None]),
                                             "n": sum(row["even"][metric] is not None and row["even"]["L"] is not None
                                                      for row in rows)}}
    return result


def _prediction(row_id, estimate, verdict, rule, extra=None):
    row = {**PREDICTION_ROWS[row_id], "estimate": estimate, "verdict": verdict,
           "registered_rule": rule}
    if extra:
        row.update(extra)
    return row


def _threshold_verdict(value, threshold, at_least=True):
    if value is None:
        return "not read"
    if at_least:
        return "held" if value >= threshold else "failed"
    return "held" if value <= threshold else "failed"


def _bounded_threshold_verdict(bounds, threshold, at_least=True):
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or any(v is None for v in bounds):
        return "not read"
    low, high = map(float, bounds)
    if at_least:
        if low >= float(threshold):
            return "held"
        if high < float(threshold):
            return "failed"
    else:
        if high <= float(threshold):
            return "held"
        if low > float(threshold):
            return "failed"
    return "not read"


def _bounds_from_report(report, field, scalar=None):
    value = report.get(field)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return list(value)
    if scalar is not None and report.get(scalar) is not None:
        point = report[scalar]
        return [point, point]
    return [None, None]


def prediction_verdicts(dose_reports, directions, gates, reader_read, inert_guard,
                        correlation, split_half, sensitivity_examples=None):
    generation_readable = bool(gates.get("passed") and all(
        isinstance(gates.get(f"gate{index}"), dict) and gates[f"gate{index}"].get("passed")
        for index in range(1, 9)))
    rungs = {float(key): value for key, value in dose_reports.items()}
    def support_bounds(dose):
        value = rungs.get(float(dose), {}).get("support", {})
        return [value.get("lower"), value.get("upper")]
    def support(dose):
        bounds = support_bounds(dose)
        return int(bounds[0]) if bounds[0] is not None and bounds[0] == bounds[1] else None
    def raw_share(dose):
        report = rungs.get(float(dose), {})
        return report.get("rates", {}).get("google_over_named")
    def named(dose):
        return rungs.get(float(dose), {}).get("named")
    def google_count(dose):
        return rungs.get(float(dose), {}).get("google_count")
    def named_bounds(dose):
        report = rungs.get(float(dose), {})
        return _bounds_from_report(report, "named_bounds", "named")
    def google_bounds(dose):
        report = rungs.get(float(dose), {})
        return _bounds_from_report(report, "google_count_bounds", "google_count")
    def share_bounds(dose):
        report = rungs.get(float(dose), {})
        rates = report.get("rates", {})
        return _bounds_from_report(rates, "google_over_named_bounds", "google_over_named")

    pred = {}
    support225 = support_bounds(0.225)
    n_support = support(0.225)
    d1 = _bounded_threshold_verdict(support225, 20, at_least=True)
    pred["176.1"] = _prediction("176.1", {"supported_directions": n_support,
                                          "supported_bounds": support225, "total": 40},
                                 d1, "Supported directions >= 20 of 40")
    led225 = rungs.get(0.225, {}).get("google_led", {})
    support_low = support225[0] if support225[0] is not None else 0
    support_high = support225[1] if support225[1] is not None else 40
    led_low = led225.get("lower", 0)
    led_high = led225.get("upper", 40)
    led_share_bounds = ([led_low / support_high, min(1.0, led_high / support_low)]
                        if support_low > 0 else
                        [led_low / support_high, 1.0] if support_high > 0 else [None, None])
    d2 = ("not read" if support_high < 5 or support_low < 5 else
          _bounded_threshold_verdict(led_share_bounds, 0.8, at_least=True))
    pred["176.2"] = _prediction("176.2", {"google_led_supported_bounds": [led_low, led_high],
                                          "supported_bounds": support225,
                                          "share_bounds": led_share_bounds}, d2,
                                 "Google modal on at least 80% of supported directions; not read below 5")
    other = rungs.get(0.225, {}).get("other_led", {})
    pred["176.3"] = _prediction("176.3", {"other_led_directions": other, "total": 40},
                                 _bounded_threshold_verdict([other.get("lower"), other.get("upper")],
                                                            2, at_least=False),
                                 "Other-led count <= 2 of 40")
    support20_bounds, support25_bounds = support_bounds(0.20), support_bounds(0.25)
    support20 = support(0.20)
    support25 = support(0.25)
    clause20 = _bounded_threshold_verdict(support20_bounds, 8, at_least=True)
    clause25 = _bounded_threshold_verdict(support25_bounds, 8, at_least=True)
    inert = bool(inert_guard.get("status") == "read" and inert_guard.get("share", 1.0) < 0.15)
    if clause25 == "failed" or clause20 == "failed":
        d4 = "failed"
    elif clause25 == "held" and clause20 == "held" and not inert:
        d4 = "held"
    else:
        d4 = "not read"
    clause20_status = "not read (inert guard)" if inert else clause20
    pred["176.4"] = _prediction("176.4", {"support_0_20": support20, "support_0_25": support25,
                                          "support_0_20_bounds": support20_bounds,
                                          "support_0_25_bounds": support25_bounds,
                                          "clause_status": {"0.20": clause20_status,
                                                            "0.25": clause25},
                                          "inert_guard_0_20": inert_guard}, d4,
                                 "At least 8/40 supported at 0.20 and at least 8/40 at 0.25; 0.20 clause not read if under 15% differ from none")
    values = [row["rungs"][1]["broken"] for row in directions if len(row.get("rungs", [])) > 1]
    median_broken = float(np.median(values)) if values else None
    pred["176.5"] = _prediction("176.5", {"median_broken": median_broken, "directions": len(values)},
                                 _threshold_verdict(median_broken, 8, at_least=False), "Median broken <= 8 of 44")

    named225, named275 = named(0.225), named(0.275)
    named225_bounds, named275_bounds = named_bounds(0.225), named_bounds(0.275)
    share225_bounds, share275_bounds = share_bounds(0.225), share_bounds(0.275)
    if (named275_bounds[0] is None or named275_bounds[1] is None or
            named275_bounds[0] < 100 <= named275_bounds[1] or named275_bounds[1] < 100 or
            named225_bounds[0] is None or named225_bounds[1] is None or
            None in share225_bounds or None in share275_bounds):
        share_clause = named_clause = "not read"
        d6 = "not read"
    else:
        share_difference = [share275_bounds[0] - share225_bounds[1],
                            share275_bounds[1] - share225_bounds[0]]
        share_clause = (_bounded_threshold_verdict(share_difference, 0, at_least=True))
        named_half_difference = [named275_bounds[0] - 0.5 * named225_bounds[1],
                                 named275_bounds[1] - 0.5 * named225_bounds[0]]
        named_clause = ("held" if named_half_difference[1] < 0 else
                        "failed" if named_half_difference[0] >= 0 else "not read")
        d6 = ("failed" if "failed" in (share_clause, named_clause) else
              "held" if share_clause == named_clause == "held" else "not read")
    pred["176.6"] = _prediction("176.6", {"named_0_225": named225, "named_0_275": named275,
                                          "named_bounds_0_225": named225_bounds,
                                          "named_bounds_0_275": named275_bounds,
                                          "google_share_0_225": raw_share(0.225),
                                          "google_share_0_275": raw_share(0.275),
                                          "google_share_bounds_0_225": share225_bounds,
                                          "google_share_bounds_0_275": share275_bounds,
                                          "condition_status": {"share_nonfall": share_clause,
                                                               "named_pool_halved": named_clause}},
                                 d6, "Pooled Google share at 0.275 >= 0.225 and named count at 0.275 < half of 0.225; not read below 100 named at 0.275")

    failed225 = rungs.get(0.225, {}).get("failed")
    failed_bounds225 = _bounds_from_report(rungs.get(0.225, {}), "failed_bounds", "failed")
    total225 = rungs.get(0.225, {}).get("answers", 0)
    fail_fraction = failed225 / total225 if failed225 is not None and total225 else None
    fail_fraction_bounds = ([failed_bounds225[0] / total225, failed_bounds225[1] / total225]
                            if total225 and None not in failed_bounds225 else [None, None])
    corr = correlation.get("google_count_vs_L", {})
    eligible_directions = int(corr.get("n", 0))
    eligible_seeds = len({int(row["seed"]) for row in directions
                          if row.get("collapse_dose", {}).get("determined") and
                          row.get("rungs", [{}, {}])[1].get("google_count") is not None})
    if not generation_readable:
        d7 = "not read"
    elif not reader_read:
        d7 = "not read"
    elif (fail_fraction_bounds[0] is None or fail_fraction_bounds[1] is None or
          fail_fraction_bounds[0] > 0.20 or fail_fraction_bounds[1] > 0.20):
        d7 = "not read"
    elif eligible_directions < 32 or eligible_seeds < 16:
        d7 = "not read"
    elif corr.get("status") != "read":
        d7 = "not read"
    elif float(corr["rho"]) <= -0.30 and corr.get("interval_95", [None, None])[1] is not None and \
            float(corr["interval_95"][1]) < 0:
        d7 = "failed"
    else:
        d7 = "held" if float(corr["rho"]) > -0.30 else "inconclusive"
    half_rhos = {}
    for metric in ("google_count_0225", "google_share_0225"):
        split = split_half.get("spearmans", {}).get(metric, {}).get("odd_against_even", {})
        half_rhos[metric] = split.get("rho")
    split_half_uninformative = any(value is None or float(value) < 0.3
                                   for value in half_rhos.values())
    pred["176.7"] = _prediction("176.7", {"spearman": corr, "eligible_directions": eligible_directions,
                                          "eligible_draws": eligible_seeds,
                                          "pooled_failed_fraction_0_225": fail_fraction,
                                          "pooled_failed_fraction_bounds_0_225": fail_fraction_bounds,
                                          "exclusion_reasons": correlation.get("exclusion_reasons", {}),
                                          "split_half_odd_against_even": half_rhos,
                                          "holding_interpretation": (
                                              "uninformative: at least one split-half Spearman is under 0.3 or undefined"
                                              if d7 == "held" and split_half_uninformative else
                                              "informative repeatability screen" if d7 == "held" else None)}, d7,
                                 "Read with at least 32 eligible directions from 16 draws and pooled failed <=20%; no strong tie if rho > -0.30; fail at rho <= -0.30 only when cluster interval upper < 0")

    support_map = {dose: support_bounds(dose) for dose in (0.20, 0.225, 0.25, 0.30)}
    s20b, s225b, s25b, s30b = (support_map[dose] for dose in (0.20, 0.225, 0.25, 0.30))
    all_w1_bounds = (s20b, s225b, s25b, s30b)
    def w1_verdict(s20, s225, s25, s30):
        falsifier_peak = s225 >= s20 + 5 and s225 >= s25 + 5
        falsifier_no_collapse = s30 >= 15
        claim = s20 <= s225 <= s25 and s30 < 5
        if falsifier_peak or falsifier_no_collapse:
            return "failed"
        return "held" if claim else "inconclusive"

    w1_admissible_verdicts = None
    if any(None in bounds for bounds in all_w1_bounds):
        w1 = "not read"
    else:
        # Design: `not read` only if an unknown could change the verdict, so try every admissible count.
        fills = {w1_verdict(*counts) for counts in itertools.product(
            *(range(int(bounds[0]), int(bounds[1]) + 1) for bounds in all_w1_bounds))}
        w1_admissible_verdicts = sorted(fills)
        w1 = w1_admissible_verdicts[0] if len(w1_admissible_verdicts) == 1 else "not read"
    if all(None not in bounds for bounds in all_w1_bounds):
        s20, s225, s25, s30 = (bounds[0] if bounds[0] == bounds[1] else None
                                for bounds in all_w1_bounds)
    else:
        s20 = s225 = s25 = s30 = None
    pred["176.W1"] = _prediction("176.W1", {"support_0_20": s20, "support_0_225": s225,
                                            "support_0_25": s25, "support_0_30": s30,
                                            "support_bounds": {str(dose): bounds
                                                               for dose, bounds in support_map.items()},
                                            "admissible_fill_verdicts": w1_admissible_verdicts}, w1,
                                 "s20 <= s225 <= s25 and s30 < 5; falsifiers are a 0.225 peak >=5 over both neighbours or s30 >=15")

    shares = [raw_share(dose) for dose in run176.DOSES]
    counts = [google_count(dose) for dose in run176.DOSES]
    share_ranges = [share_bounds(dose) for dose in run176.DOSES]
    count_ranges = [google_bounds(dose) for dose in run176.DOSES]
    share_changes = []
    if any(None in bounds for bounds in share_ranges + count_ranges):
        w2 = "not read"
    else:
        ambiguous = False
        share_misses = []
        count_falsifiers = []
        for index in range(1, len(run176.DOSES)):
            prior_share, current_share = share_ranges[index - 1], share_ranges[index]
            share_change_bounds = [current_share[0] - prior_share[1],
                                   current_share[1] - prior_share[0]]
            definite_fall = share_change_bounds[1] < 0
            definite_nonfall = share_change_bounds[0] >= 0
            if not definite_fall and not definite_nonfall:
                ambiguous = True
            miss_kind = ("receding" if share_change_bounds[1] < -0.05 else "noise") if definite_fall else None
            prior_count, current_count = count_ranges[index - 1], count_ranges[index]
            count_change_bounds = [current_count[0] - prior_count[1],
                                   current_count[1] - prior_count[0]]
            count_change = (counts[index] - counts[index - 1]
                            if counts[index] is not None and counts[index - 1] is not None else None)
            row = {"from": run176.DOSES[index - 1], "to": run176.DOSES[index],
                   "change": (shares[index] - shares[index - 1]
                              if shares[index] is not None and shares[index - 1] is not None else None),
                   "change_points": (100 * (shares[index] - shares[index - 1])
                                     if shares[index] is not None and shares[index - 1] is not None else None),
                   "change_bounds": share_change_bounds,
                   "google_count_change": count_change,
                   "google_count_change_bounds": count_change_bounds,
                   "miss_kind": miss_kind}
            share_changes.append(row)
            if definite_fall:
                share_misses.append(row)
            if index in (3, 4):
                if count_change_bounds[0] > 0:
                    count_falsifiers.append({"from": run176.DOSES[index - 1],
                                             "to": run176.DOSES[index],
                                             "google_count_change": count_change,
                                             "google_count_change_bounds": count_change_bounds,
                                             "miss_kind": "raw_google_count_rising_falsifier"})
                elif count_change_bounds[1] > 0:
                    ambiguous = True
        w2 = "failed" if share_misses or count_falsifiers else "not read" if ambiguous else "held"
        share_changes.extend(count_falsifiers)
    pred["176.W2"] = _prediction("176.W2", {"google_share_by_rung": shares,
                                            "google_share_bounds_by_rung": share_ranges,
                                            "google_count_by_rung": counts,
                                            "google_count_bounds_by_rung": count_ranges,
                                            "share_changes": share_changes,
                                            "raw_count_peak_at_or_below_0_25": (max(counts[:3])
                                                                                 if all(v is not None for v in counts) else None)},
                                 w2, "Any pooled share fall fails (receding over 5 points, noise otherwise); a count rise at 0.275 or 0.30 is a separate falsifier")

    gaps = [row["gap"] for row in directions if row.get("top_supported_rung") is not None and row.get("gap") is not None]
    gap_median = float(np.median(gaps)) if gaps else None
    gap_under = sum(value < 0.025 for value in gaps)
    right_censored_gaps = sum(row.get("collapse_dose", {}).get("status") == "right censored"
                              for row in directions if row.get("top_supported_rung") is not None and
                              row.get("gap") is not None)
    gap_share_under = gap_under / len(gaps) if gaps else None
    if not generation_readable or not reader_read or not gaps:
        w3 = "not read"
    elif gap_median >= 0.05:
        w3 = "held"
    elif gap_median < 0.025:
        w3 = "failed"
    else:
        w3 = "inconclusive"
    pred["176.W3"] = _prediction("176.W3", {"median_gap": gap_median, "directions": len(gaps),
                                            "under_0_025": gap_under,
                                            "share_under_0_025": gap_share_under,
                                            "right_censored_L_directions": right_censored_gaps,
                                            "unknown_support_rule": (
                                                "Unknown support is not supported; if it could be the top supported rung, "
                                                "exclude and count the direction.")}, w3,
                                 "Median gap >= 0.05; falsified below 0.025")

    if not generation_readable:
        for key in pred:
            pred[key]["verdict"] = "not read"
            pred[key]["read_block"] = "a generation gate failed or was missing"
    if not reader_read:
        for key in ("176.1", "176.2", "176.3", "176.4", "176.6", "176.7", "176.W1", "176.W2", "176.W3"):
            pred[key]["verdict"] = "not read"
            pred[key]["read_block"] = "reader gate or error cap failed"
    return pred


def _inert_guard(raw_answers, expected=1760, doses=None, seeds=None):
    doses = tuple(run176.DOSES if doses is None else map(float, doses))
    seeds = tuple(run176.SEEDS if seeds is None else map(int, seeds))
    fresh_draw_sets = {f"f100d_d{seed}" for seed in seeds}
    none_rows = [row for row in raw_answers if row.get("arm") == "none"]
    none = {str(row.get("prompt_id")): row for row in none_rows}
    by_rung = {}
    expected_rows = int(expected)
    for magnitude in doses:
        rows = [row for row in raw_answers if row.get("set") in fresh_draw_sets and
                abs(float(row.get("dose", 0))) == float(magnitude)]
        if len(rows) != expected_rows or len(none) != 44 or len(none_rows) != len(none):
            by_rung[str(float(magnitude))] = {
                "status": "not read", "inert": None, "different": None,
                "total": len(rows), "expected": expected_rows,
                "reason": "rung or none rows are incomplete or duplicated"}
            continue
        different = 0
        missing_tokens = False
        for row in rows:
            plain = none.get(str(row.get("prompt_id")))
            a, b = row.get("token_ids"), plain.get("token_ids") if plain else None
            if not isinstance(a, list) or not isinstance(b, list):
                missing_tokens = True
                break
            different += int(a[:40] != b[:40])
        if missing_tokens:
            by_rung[str(float(magnitude))] = {
                "status": "not read", "inert": None, "different": None,
                "total": expected_rows, "expected": expected_rows,
                "reason": "token_ids are missing"}
        else:
            share = different / expected_rows
            by_rung[str(float(magnitude))] = {
                "status": "read", "inert": share < 0.15, "different": different,
                "total": expected_rows, "share": share, "threshold": 0.15,
                "comparison": "token_ids[:40] versus none for the same prompt"}
    primary = by_rung.get("0.2", {"status": "not read", "inert": None,
                                  "different": None, "total": 0,
                                  "expected": expected_rows,
                                  "reason": "0.20 rung is not in the dose list"})
    return {**primary, "by_rung": by_rung}


def _build_sensitivity_examples(directions, correlation_key="google_count_vs_L", reps=REPS, seed=SEED):
    # The reader-missing rows have a definite lower and upper failed count at each rung.
    # Rebuild only the collapse outcomes under the two registered extreme fills.
    examples = {}
    for label, choose_upper in (("all_undetermined_short_answers_are_fragments", True),
                                ("none_of_undetermined_short_answers_are_fragments", False)):
        filled = []
        for direction in directions:
            vals = [pair[1] if choose_upper else pair[0] for pair in
                    [rung["failed_bounds"] for rung in direction["rungs"]]]
            crossing = _interpolate_crossing(vals, 22, run176.DOSES)
            row = next(rung for rung in direction["rungs"] if rung["dose"] == 0.225)
            filled.append({"seed": direction["seed"], "google_count": row["google_count"],
                           "L": crossing["L"], "crossing_status": crossing["status"]})
        paired = [row for row in filled if row["google_count"] is not None and row["L"] is not None]
        rho = _rho([row["google_count"] for row in paired], [row["L"] for row in paired])
        examples[label] = {"spearman_google_count_vs_L": rho, "eligible_directions": len(paired),
                           "eligible_draws": len({row["seed"] for row in paired}),
                           "rank_statistic_is_not_monotone": True}
    return examples


def _score_core(raw_answers, reader_by_row, seeds, doses, expected, reps, seed):
    scored = score173.score_rows(raw_answers)
    by_arm = defaultdict(list)
    for row in scored:
        by_arm[str(row.get("arm"))].append(row)
    directions = []
    for draw_seed in seeds:
        set_name = f"f100d_d{draw_seed}"
        for sign in (1, -1):
            rows = [row for dose in doses for row in by_arm.get(run175.dose_arm(set_name, float(dose) * sign), [])]
            direction = _direction(rows, reader_by_row, set_name, sign, doses,
                                   expected=expected, fragment_limit=12)
            directions.append(direction)
    dose_reports = {}
    for index, dose in enumerate(doses):
        report = _rung_aggregate(directions, index, expected_directions=len(directions),
                                 reps=reps, seed=seed)
        report["directions"] = [direction["rungs"][index] | {"seed": direction["seed"],
                                     "sign": direction["sign"], "set": direction["set"]}
                                for direction in directions]
        dose_reports[float(dose)] = report
    transitions = transition_tables(raw_answers, scored, reader_by_row, directions, doses, expected)
    paired = {"adjacent": {f"{float(doses[index-1]):g}->{float(doses[index]):g}":
                            _paired_shares(directions, index - 1, index, reps=reps, seed=seed + index)
                            for index in range(1, len(doses))},
              "0.225_to_0.275": _paired_shares(directions, doses.index(0.225), doses.index(0.275),
                                               reps=reps, seed=seed + 100)}
    corr_rows = []
    exclusion_reasons = Counter()
    for direction in directions:
        row225 = next(row for row in direction["rungs"] if row["dose"] == 0.225)
        lval = direction["collapse_dose"]
        count_missing = row225["google_count"] is None
        l_missing = not lval.get("determined") or lval.get("L") is None
        if count_missing:
            exclusion_reasons["Google count null at 0.225"] += 1
        if l_missing:
            exclusion_reasons[direction.get("undetermined_reason") or "collapse dose not determined"] += 1
        if not count_missing and not l_missing:
            corr_rows.append({"seed": direction["seed"], "sign": direction["sign"],
                              "google_count": row225["google_count"], "google_share": row225["google_over_named"],
                              "L": lval["L"], "ordinal_L": lval.get("ordinal_rank"),
                              "status": lval["status"], "gap": direction["gap"]})
    count_corr = _cluster_spearman(corr_rows, "google_count", "L", reps=reps, seed=seed)
    ordinal_count_corr = _cluster_spearman(corr_rows, "google_count", "ordinal_L",
                                           reps=reps, seed=seed + 2)
    share_corr = _cluster_spearman(corr_rows, "google_share", "L", reps=reps, seed=seed + 1)
    correlations = {"google_count_vs_L": count_corr,
                    "google_count_vs_ordinal_L": ordinal_count_corr,
                    "google_share_vs_L_conditional": share_corr,
                    "eligible_directions": len(corr_rows),
                    "eligible_draws": len({row["seed"] for row in corr_rows}),
                    "exclusion_reasons": dict(exclusion_reasons),
                    "observations": corr_rows}
    split = _split_half_summary(directions, reader_by_row, doses, reps=reps, seed=seed, expected=expected)
    for direction in directions:
        direction.pop("answers", None)
    gap_values = [row["gap"] for row in directions if row["gap"] is not None]
    gap_directions = [row for row in directions
                      if row.get("top_supported_rung") is not None and row.get("gap") is not None]
    under_gap = sum(value < 0.025 for value in gap_values)
    right_censored = sum(row.get("collapse_dose", {}).get("status") == "right censored"
                         for row in gap_directions)
    gap_report = {"n": len(gap_values), "median": float(np.median(gap_values)) if gap_values else None,
                  "under_0_025": under_gap,
                  "share_under_0_025": under_gap / len(gap_values) if gap_values else None,
                  "right_censored_L_directions": right_censored,
                  "unknown_support_rule": (
                      "A rung with unknown support is unsupported; if it could be the top supported rung, "
                      "exclude the direction from W3 and count the exclusion."),
                  "values": gap_values}
    selected_index = max(range(len(doses)), key=lambda index:
                         (dose_reports[float(doses[index])]["support"]["lower"], float(doses[index])))
    selected = {"dose": float(doses[selected_index]),
                "rule": "maximum definitely supported directions; ties go to the higher rung",
                "support": dose_reports[float(doses[selected_index])]["support"]}
    sensitivity_examples = _build_sensitivity_examples(directions, reps=reps, seed=seed)
    return {"scored": scored, "directions": directions, "dose_reports": dose_reports,
            "transitions": transitions, "paired_share_changes": paired,
            "correlations": correlations, "split_half": split, "gap_distribution": gap_report,
            "selected": selected, "sensitivity_examples": sensitivity_examples}


def score(answers_path="results/part176/answers.jsonl", gates_path="results/part176/gates.json",
          labels_dir="results/part176/labels", out_dir="results/part176", reps=REPS, seed=SEED,
          expected_per_direction=44, draw_seeds=None, doses=None, config=None):
    config = config or {}
    seeds = tuple(run176.SEEDS if draw_seeds is None else map(int, draw_seeds))
    doses = tuple(run176.DOSES if doses is None else map(float, doses))
    expected = int(expected_per_direction)
    raw_answers = read_jsonl(answers_path)
    gates = read_json(gates_path, {}) or {}
    scored = score173.score_rows(raw_answers)
    first_rows = read_jsonl(Path(labels_dir) / "labels-ds.jsonl")
    reader_by_row = score175._reader_map(scored, first_rows)
    reader_gate = config.get("reader_gate")
    if reader_gate is None:
        reader_gate = score174._reader_gate(answers_path, raw_answers, labels_dir)
    reader_read = bool(reader_gate.get("passed"))
    generation_readable = bool(gates.get("passed") and all(
        isinstance(gates.get(f"gate{index}"), dict) and gates[f"gate{index}"].get("passed")
        for index in range(1, 9)))
    core = _score_core(raw_answers, reader_by_row, seeds, doses, expected, int(reps), int(seed))
    inert = _inert_guard(raw_answers, expected=expected * 2 * len(seeds), doses=doses,
                         seeds=seeds)
    predictions = prediction_verdicts(core["dose_reports"], core["directions"], gates,
                                      reader_read, inert, core["correlations"], core["split_half"],
                                      core["sensitivity_examples"])
    if not generation_readable:
        for key, row in predictions.items():
            row["verdict"] = "not read"
            row["read_block"] = "a generation gate failed or was missing"
    # Use no best-rung alias: the design reserves this result for the selected key.
    summary = {
        "part": 176,
        "prediction_read_status": ("read" if generation_readable and reader_read else
                                   "partial" if generation_readable else "not read"),
        "generation_gates": gates,
        "reader_gate9": reader_gate,
        "inert_guard_0_20": inert,
        "inert_guard_by_rung": inert.get("by_rung", {}),
        "counts": {"answers": len(raw_answers), "scored_non_replay_answers": len(scored),
                   "directions": len(core["directions"]), "expected_directions": 2 * len(seeds)},
        "doses": {f"{dose:g}": report for dose, report in core["dose_reports"].items()},
        "directions": core["directions"],
        "correlations": core["correlations"],
        "split_half_spearmans": core["split_half"],
        "paired_share_changes": core["paired_share_changes"],
        "transition_tables": core["transitions"],
        "gap_distribution": core["gap_distribution"],
        "selected": core["selected"],
        "undetermined_sensitivity_examples": core["sensitivity_examples"],
        "predictions": predictions,
        "all_failed_sensitivity_example": core["sensitivity_examples"].get(
            "all_undetermined_short_answers_are_fragments"),
        "none_failed_sensitivity_example": core["sensitivity_examples"].get(
            "none_of_undetermined_short_answers_are_fragments"),
        "scope": "These twenty f100d draws at five fixed doses, Qwen3.5-4B at the registered snapshot, and NEW44B prompts.",
        "null_rule": "Unobserved reader-derived counts and rates stay null; only unlabelled clean short answers without an own name can be undetermined fragments.",
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "scored_directions.jsonl", core["directions"])
    write_json(out / "transitions.json", core["transitions"])
    write_json(out / "prediction_table.json", predictions)
    write_json(out / "score.json", summary)
    print(json.dumps({"prediction_read_status": summary["prediction_read_status"],
                      "predictions": {key: value["verdict"] for key, value in predictions.items()},
                      "directions": len(core["directions"])}, ensure_ascii=True, allow_nan=False), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default="results/part176/answers.jsonl")
    parser.add_argument("--gates", default="results/part176/gates.json")
    parser.add_argument("--labels-dir", default="results/part176/labels")
    parser.add_argument("--out-dir", default="results/part176")
    parser.add_argument("--replicates", type=int, default=REPS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    summary = score(args.answers, args.gates, args.labels_dir, args.out_dir,
                    reps=args.replicates, seed=args.seed)
    gates = summary.get("generation_gates", {})
    return 0 if gates.get("passed") and summary.get("reader_gate9", {}).get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
