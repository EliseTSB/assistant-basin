#!/usr/bin/env python3
"""Part 175 null-aware census scoring and registered prediction table."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part171"), str(ROOT / "part172"),
              str(ROOT / "part173"), str(ROOT / "part174")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import label173
import names173
import run171
import run174
import score173
import score174
import run175


REPS = 10_000
SEED = 175
PREDICTION_ROWS = {
    "175.1": {"prediction": "Stage A finds a D* at or above 0.10", "confidence": "high", "author": "Claude"},
    "175.2": {"prediction": "At D*, at least 8 of the 40 f100d signed directions are supported", "confidence": "low", "author": "Claude"},
    "175.3": {"prediction": "Google is modal on at least 80% of supported directions at D*; not read below 5 supported", "confidence": "medium", "author": "Claude"},
    "175.4": {"prediction": "At D*, at most 2 of the 40 signed directions are other-led", "confidence": "medium", "author": "Claude"},
    "175.5": {"prediction": "At D*, the median broken of the 40 signed directions is at most 8 of 44", "confidence": "medium", "author": "Claude"},
    "175.6": {"prediction": "More directions are supported at D* than at 0.5 D*; not read if neither has at least 3", "confidence": "medium", "author": "Claude"},
    "175.W1": {"prediction": "D* lands between 0.20 and 0.35; 0.10, 0.45, or no D* falsifies; 0.15 and 0.40 are inconclusive", "confidence": "medium", "author": "Weaver"},
    "175.W2": {"prediction": "At most 5 of the 40 signed directions are supported at D*; 8 or more falsifies", "confidence": "medium", "author": "Weaver"},
    "175.W3": {"prediction": "At least 3 of 4 Stage A draws have a sign gap of at least 2 ladder steps; then carry D* as a band", "confidence": "medium", "author": "Weaver"},
}


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _read_jsonl(path):
    return run171.read_jsonl(path) if Path(path).is_file() else []


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def _write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n"
                                          for row in rows),
                          encoding="utf-8", newline="\n")


def _reader_map(scored, first_rows):
    return score174._reader_map(scored, first_rows)


def _entity_counts(clean_rows, reader_by_row):
    counts = Counter()
    for row in clean_rows:
        record = reader_by_row.get((str(row.get("arm")), str(row.get("prompt_id"))))
        if record and label173.valid(record):
            entity = score173._primary_entity(record, mode="canonical")
            if entity is not None and entity != names173.QWEN_ENTITY:
                counts[entity] += 1
    return counts


def _modal_for_counts(counts):
    values = [(name, int(count)) for name, count in counts.items() if count > 0]
    if not values:
        return None, 0
    name, count = sorted(values, key=lambda item: (-item[1], item[0]))[0]
    return name, count


def _category_bounds(counts, missing, support_bounds):
    """Conservative category bounds over every possible label for missing clean answers."""
    support_low, support_high = map(int, support_bounds)
    if support_high == 0:
        return [0, 0], [0, 0]
    if missing <= 0:
        leader, _count = _modal_for_counts(counts)
        if support_low == 1:
            if leader == "google":
                return [1, 1], [0, 0]
            return [0, 0], [1, 1]
        return [0, 0], [0, 0]

    google_count = int(counts.get("google", 0))
    other_counts = {name: int(count) for name, count in counts.items() if name != "google"}
    other_max = max(other_counts.values(), default=0)
    # An unobserved answer may add to an existing name or introduce a new one.
    google_guaranteed = (support_low == 1 and
                         google_count > other_max + missing)
    google_possible = (support_high == 1 and
                       google_count + missing >= other_max)
    other_possible = (support_high == 1 and
                      other_max + missing >= google_count)
    other_guaranteed = support_low == 1 and not google_possible
    return [int(google_guaranteed), int(google_possible)], [int(other_guaranteed), int(other_possible)]


def _direction(rows, reader_by_row, set_name, dose, expected=44):
    rows = list(rows)
    if len(rows) != int(expected):
        raise ValueError(f"{set_name} at dose {dose} has {len(rows)} rows; expected {expected}")
    summary = score173.summarize_cell(rows, reader_by_row, mode="canonical")
    clean = [row for row in rows if row.get("clean")]
    labelled = [row for row in clean
                if (record := reader_by_row.get((str(row.get("arm")), str(row.get("prompt_id"))))) is not None
                and label173.valid(record)]
    missing = len(clean) - len(labelled)
    if not clean:
        reader_state = "no clean answers"
        support_bounds = [0, 0]
        support_status = "unsupported"
    elif not labelled:
        reader_state = "reader missing"
        support_bounds = [0, int(summary["broken"] <= run174.DESIGN["cell"]["broken_max"]
                                 and len(clean) >= run174.DESIGN["cell"]["modal_count_min"])]
        support_status = "unknown"
    elif missing:
        reader_state = "partly labelled"
        counts = _entity_counts(clean, reader_by_row)
        max_count = max(counts.values(), default=0)
        damage_ok = summary["broken"] <= run174.DESIGN["cell"]["broken_max"]
        support_bounds = [int(damage_ok and max_count >= run174.DESIGN["cell"]["modal_count_min"]),
                          int(damage_ok and max_count + missing >= run174.DESIGN["cell"]["modal_count_min"])]
        support_status = ("supported" if support_bounds == [1, 1] else
                          "unsupported" if support_bounds == [0, 0] else "bounded")
    else:
        reader_state = "complete"
        support = int(bool(summary["eligible"]))
        support_bounds = [support, support]
        support_status = "supported" if support else "unsupported"

    counts = _entity_counts(clean, reader_by_row)
    if reader_state == "complete":
        google_bounds, other_bounds = _category_bounds(counts, 0, support_bounds)
    elif reader_state == "no clean answers":
        google_bounds, other_bounds = [0, 0], [0, 0]
    else:
        google_bounds, other_bounds = _category_bounds(counts, missing, support_bounds)
    google_count = (int(summary["full_name_distribution"].get("google", 0))
                    if reader_state == "complete" else None)
    modal, modal_count = _modal_for_counts(counts)
    raw_hits = summary.get("raw_word_hits", {})
    return {
        "set": set_name, "family": "f100d", "seed": int(set_name.rsplit("d", 1)[1]),
        "dose": float(dose), "arm": run175.dose_arm(set_name, dose),
        "answers": len(rows), "expected_answers": int(expected),
        "broken": int(summary["broken"]), "clean": len(clean),
        "reader_state": reader_state, "reader_labelled_clean_answers": len(labelled),
        "reader_unlabelled_clean_answers": int(missing),
        "google_claim_count": google_count,
        "supported": (True if support_bounds == [1, 1] else
                      False if support_bounds == [0, 0] else None),
        "support_status": support_status, "support_bounds": support_bounds,
        "google_led": (True if google_bounds == [1, 1] else
                       False if google_bounds == [0, 0] else None),
        "google_led_bounds": google_bounds,
        "other_led": (True if other_bounds == [1, 1] else
                      False if other_bounds == [0, 0] else None),
        "other_led_bounds": other_bounds,
        "summary": summary,
        "modal": modal if reader_state == "complete" else None,
        "labelled_modal": modal,
        "labelled_modal_count": int(modal_count),
        "full_name_distribution": summary.get("full_name_distribution", {}),
        "raw_word_hits": {key: int(value) for key, value in raw_hits.items()},
    }


def _sum_bounds(directions, key):
    low = sum(int(row[key][0]) for row in directions)
    high = sum(int(row[key][1]) for row in directions)
    return {"lower": low, "upper": high, "denominator": len(directions)}


def _histogram(directions, field):
    values = Counter(row[field] for row in directions if row[field] is not None)
    nulls = sum(row[field] is None for row in directions)
    return {"counts": {str(key): int(value) for key, value in sorted(values.items())},
            "labelled_directions": len(directions) - nulls,
            "null_directions": int(nulls)}


def _draw_rows(directions):
    by_set = {}
    for row in directions:
        by_set.setdefault(row["set"], {})[float(row["dose"])]=row
    output = []
    for set_name in run175.F100D_SET_NAMES:
        dose_map = by_set.get(set_name, {})
        for plus_dose in sorted({abs(float(value)) for value in dose_map}):
            plus = dose_map.get(plus_dose)
            minus = dose_map.get(-plus_dose)
            if plus is None or minus is None:
                continue
            if plus["support_status"] in ("unknown", "bounded") or minus["support_status"] in ("unknown", "bounded"):
                draw_class = "not read"
            else:
                p_support, n_support = bool(plus["supported"]), bool(minus["supported"])
                if not p_support and not n_support:
                    draw_class = "neither"
                elif p_support != n_support:
                    draw_class = "one sign"
                elif plus["modal"] == minus["modal"]:
                    draw_class = "both same"
                else:
                    draw_class = "both different"
            google_plus = plus["google_claim_count"]
            google_minus = minus["google_claim_count"]
            output.append({"set": set_name, "seed": plus["seed"],
                           "dose_magnitude": plus_dose, "draw_class": draw_class,
                           "positive_supported": plus["supported"],
                           "negative_supported": minus["supported"],
                           "google_claim_count_plus": google_plus,
                           "google_claim_count_minus": google_minus,
                           "google_claim_difference": (google_plus - google_minus
                                                       if google_plus is not None and google_minus is not None
                                                       else None)})
    return output


def _wilson_bounds(bounds, n):
    low, high = map(int, bounds)
    if low == high:
        return {"status": "read", "count": low, "n": int(n),
                "interval_95": score174._wilson(low, n)["interval_95"]}
    return {"status": "not read", "count_bounds": [low, high], "n": int(n),
            "interval_95_bounds": [score174._wilson(low, n)["interval_95"],
                                   score174._wilson(high, n)["interval_95"]]}


def _metric_uncertainty(directions, reps=REPS, seed=SEED):
    result = {}
    for metric, bounds_key in (("supported", "support_bounds"),
                               ("google_led", "google_led_bounds"),
                               ("other_led", "other_led_bounds")):
        by_sign = {}
        matrices = []
        complete = True
        for sign in (1, -1):
            chosen = [row for row in directions if (1 if float(row["dose"]) > 0 else -1) == sign]
            low = sum(row[bounds_key][0] for row in chosen)
            high = sum(row[bounds_key][1] for row in chosen)
            by_sign[str(sign)] = _wilson_bounds((low, high), len(chosen))
            complete = complete and low == high
        by_draw = {}
        for row in directions:
            by_draw.setdefault(row["seed"], {})[1 if float(row["dose"]) > 0 else -1] = row[bounds_key]
        seeds = sorted(by_draw)
        lower_matrix = np.asarray([[by_draw[s][1][0], by_draw[s][-1][0]] for s in seeds], dtype=np.int8)
        upper_matrix = np.asarray([[by_draw[s][1][1], by_draw[s][-1][1]] for s in seeds], dtype=np.int8)
        if complete and len(seeds) == 20:
            paired = lower_matrix
            pooled = score174._bootstrap_draw_interval(paired, reps=reps, seed=seed)
            events = np.any(paired, axis=1).astype(np.int8)
            either = score174._draw_event_interval(events, reps=reps, seed=seed)
        else:
            lower_rate = float(lower_matrix.mean()) if lower_matrix.size else None
            upper_rate = float(upper_matrix.mean()) if upper_matrix.size else None
            pooled = {"status": "not read", "label": "cluster bootstrap over 20 draws; signs kept together",
                      "count_rate_bounds": [lower_rate, upper_rate],
                      "interval_95": [None, None], "reason": "reader uncertainty leaves metric bounds"}
            lower_events = np.any(lower_matrix, axis=1).astype(np.int8)
            upper_events = np.any(upper_matrix, axis=1).astype(np.int8)
            either = {"status": "not read", "label": "either-sign rate over draws",
                      "count_bounds": [int(lower_events.sum()), int(upper_events.sum())],
                      "n": len(seeds), "interval_95": [None, None],
                      "reason": "reader uncertainty leaves metric bounds"}
        result[metric] = {"at_each_sign_wilson": by_sign,
                          "pooled_signed_rate": pooled,
                          "either_sign_rate": either}
    return result


def _dose_report(directions, reps=REPS, seed=SEED):
    counts = {
        "supported": _sum_bounds(directions, "support_bounds"),
        "google_led": _sum_bounds(directions, "google_led_bounds"),
        "other_led": _sum_bounds(directions, "other_led_bounds"),
        "no_clean_answers": sum(row["reader_state"] == "no clean answers" for row in directions),
        "reader_missing": sum(row["reader_state"] == "reader missing" for row in directions),
        "partly_labelled": sum(row["reader_state"] == "partly labelled" for row in directions),
    }
    return {"directions": directions, "counts": counts,
            "google_claim_count_histogram": _histogram(directions, "google_claim_count"),
            "raw_google_word_hit_histogram": _histogram(
                [{"raw_google_word_hits": int(row["raw_word_hits"].get("google", 0))}
                 for row in directions], "raw_google_word_hits"),
            "draw_classes": _draw_rows(directions),
            "uncertainty_intervals": _metric_uncertainty(directions, reps=reps, seed=seed)}


def _count_verdict(bounds, bar, at_least):
    low, high = int(bounds["lower"]), int(bounds["upper"])
    if at_least:
        verdict = "held" if low >= bar else "failed" if high < bar else "not read"
    else:
        verdict = "held" if high <= bar else "failed" if low > bar else "not read"
    return verdict, {"lower": low, "upper": high, "denominator": int(bounds["denominator"])}


def _prediction(row_id, estimate, verdict, rule, extra=None):
    metadata = PREDICTION_ROWS[row_id]
    value = {**metadata, "estimate": estimate, "verdict": verdict, "registered_rule": rule}
    if extra:
        value.update(extra)
    return value


def prediction_verdicts(stage_a, primary_report, half_report, reader_read, generation_readable,
                        inert, stage_a_only=False):
    d_star = stage_a.get("d_star")
    stage_a_has_d = stage_a.get("status") == "selected" and d_star is not None
    if stage_a_only:
        results = {key: _prediction(key, {"d_star": None} if key == "175.1" else {},
                                    "failed" if key == "175.1" else "not read",
                                    "Stage A must select a dose at or above 0.10" if key == "175.1" else
                                    "Stage A stopped; no Stage B result")
                   for key in PREDICTION_ROWS}
        w3 = stage_a.get("weaver_175.W3", {})
        results["175.W1"] = _prediction("175.W1", {"d_star": None}, "failed",
                                         "Falsified when Stage A selects no D*")
        results["175.W3"] = _prediction(
            "175.W3", {"draws_with_gap_at_least_two": w3.get("gap_at_least_two_draws"),
                       "draws": 4, "d_star_band_if_held": w3.get("d_star_band_if_held")},
            w3.get("verdict", "not read"),
            "Held at 3 or 4 of 4 Stage A draws with an absolute sign gap >= 2 steps; otherwise failed")
        return results

    w3 = stage_a.get("weaver_175.W3", {})
    results = {}
    d1_verdict = "held" if stage_a_has_d else "failed"
    results["175.1"] = _prediction("175.1", {"d_star": d_star}, d1_verdict,
                                    "Stage A selects a D* at or above the 0.10 ladder minimum")
    if not stage_a_has_d:
        for key in ("175.2", "175.3", "175.4", "175.5", "175.6", "175.W1", "175.W2", "175.W3"):
            results[key] = _prediction(key, {}, "not read", "No D* was selected")
        results["175.W1"] = _prediction("175.W1", {"d_star": None}, "failed",
                                         "Falsified when Stage A selects no D*")
        results["175.W3"] = _prediction(
            "175.W3", {"draws_with_gap_at_least_two": w3.get("gap_at_least_two_draws"),
                       "draws": 4, "d_star_band_if_held": w3.get("d_star_band_if_held")},
            w3.get("verdict", "not read"),
            "Held at 3 or 4 of 4 Stage A draws with an absolute sign gap >= 2 steps; otherwise failed")
        return results

    d_star_value = float(d_star)
    if d_star_value in (0.20, 0.25, 0.30, 0.35):
        w1_verdict = "held"
    elif d_star_value in (0.10, 0.45):
        w1_verdict = "failed"
    else:
        w1_verdict = "inconclusive"
    results["175.W1"] = _prediction("175.W1", {"d_star": d_star_value}, w1_verdict,
                                    "Held at 0.20-0.35; falsified at 0.10, 0.45, or no D*; 0.15 and 0.40 inconclusive")
    results["175.W3"] = _prediction(
        "175.W3", {"draws_with_gap_at_least_two": w3.get("gap_at_least_two_draws"),
                   "draws": 4, "d_star_band_if_held": w3.get("d_star_band_if_held")},
        w3.get("verdict", "not read"),
        "Held at 3 or 4 of 4 Stage A draws with an absolute sign gap >= 2 steps; otherwise failed")

    primary_directions = primary_report["directions"]
    support = primary_report["counts"]["supported"]
    other = primary_report["counts"]["other_led"]
    google = primary_report["counts"]["google_led"]
    d2_verdict, d2_bounds = _count_verdict(support, 8, True)
    w2_verdict, w2_bounds = _count_verdict(support, 5, False)
    if w2_bounds["lower"] >= 8:
        w2_verdict = "failed"
    elif w2_bounds["lower"] >= 6 and w2_bounds["upper"] <= 7:
        w2_verdict = "inconclusive"
    elif w2_bounds["upper"] > 5 and w2_bounds["lower"] < 8:
        w2_verdict = "not read"
    elif w2_bounds["upper"] <= 5:
        w2_verdict = "held"

    broken = [int(row["broken"]) for row in primary_directions
              if row.get("broken") is not None]
    if len(broken) == 40:
        median_broken = float(np.median(broken))
        d5_verdict = "held" if median_broken <= 8 else "failed"
    else:
        median_broken = float(np.median(broken)) if broken else None
        d5_verdict = "not read"
    results["175.5"] = _prediction("175.5", {"median_broken": median_broken,
                                               "directions": len(broken)}, d5_verdict,
                                    "Median broken <= 8 of 44")

    if inert:
        for key, estimate, rule in (
            ("175.2", {"supported_directions": d2_bounds, "total": 40}, "At least 8 of 40 signed directions are supported at D*"),
            ("175.W2", {"supported_directions": w2_bounds, "total": 40}, "At most 5 supported; 6-7 inconclusive; 8 or more falsifies"),
            ("175.3", {"google_led": google, "supported": support}, "Google-led share >= 0.80; not read below 5 supported"),
            ("175.4", {"other_led": other, "total": 40}, "Other-led count <= 2"),
            ("175.6", {}, "More supported directions at D* than at 0.5 D*; not read if neither has 3"),
        ):
            results[key] = _prediction(key, estimate, "not read", rule,
                                       {"read_block": "D* inert-dose guard below 15%"})
    elif not generation_readable:
        for key in ("175.2", "175.3", "175.4", "175.6", "175.W2"):
            results[key] = _prediction(key, {}, "not read", "A required generation gate failed or was missing")
    else:
        results["175.2"] = _prediction("175.2", {"supported_directions": d2_bounds, "total": 40},
                                        d2_verdict, "Supported count >= 8",
                                        {"bounds": d2_bounds})
        support_low, support_high = support["lower"], support["upper"]
        google_low, google_high = google["lower"], google["upper"]
        other_low, other_high = other["lower"], other["upper"]
        share_low = google_low / support_high if support_high else None
        share_high = google_high / (google_high + other_low) if google_high + other_low else None
        if support_high < 5:
            d3_verdict = "not read"
        elif support_low < 5:
            d3_verdict = "not read"
        elif share_low is not None and share_low >= 0.80:
            d3_verdict = "held"
        elif share_high is not None and share_high < 0.80:
            d3_verdict = "failed"
        else:
            d3_verdict = "not read"
        results["175.3"] = _prediction(
            "175.3", {"google_led": {"lower": google_low, "upper": google_high},
                      "other_led": {"lower": other_low, "upper": other_high},
                      "supported": support, "share_bounds": [share_low, share_high]},
            d3_verdict, "Google-led share >= 0.80; not read below 5 supported",
            {"bounds": {"supported": support, "google_led": [google_low, google_high],
                        "other_led": [other_low, other_high], "share": [share_low, share_high]}})
        d4_verdict, d4_bounds = _count_verdict(other, 2, False)
        results["175.4"] = _prediction("175.4", {"other_led": d4_bounds, "total": 40},
                                        d4_verdict, "Other-led count <= 2",
                                        {"bounds": d4_bounds})

        if d5_verdict == "failed":
            for key in ("175.2", "175.3", "175.4"):
                results[key]["condition"] = "175.5 failed; predictions remain reported with the registered drift condition"

        half_support = half_report["counts"]["supported"]
        if support_high < 3 and half_support["upper"] < 3:
            d6_verdict = "not read"
        else:
            diff_low = support_low - half_support["upper"]
            diff_high = support_high - half_support["lower"]
            d6_verdict = "held" if diff_low > 0 else "failed" if diff_high <= 0 else "not read"
        results["175.6"] = _prediction("175.6", {"d_star_supported": support,
                                                   "half_d_star_supported": half_support,
                                                   "difference_bounds": [diff_low, diff_high]
                                                   if not (support_high < 3 and half_support["upper"] < 3)
                                                   else None}, d6_verdict,
                                        "D* supported count > 0.5 D* supported count; not read if neither has 3")

        if w2_bounds["lower"] >= 8:
            w2_verdict = "failed"
        elif w2_bounds["upper"] <= 5:
            w2_verdict = "held"
        elif w2_bounds["lower"] >= 6 and w2_bounds["upper"] <= 7:
            w2_verdict = "inconclusive"
        else:
            w2_verdict = "not read"
        results["175.W2"] = _prediction("175.W2", {"supported_directions": w2_bounds,
                                                     "total": 40}, w2_verdict,
                                          "Held at <=5; failed at >=8; 6-7 inconclusive",
                                          {"bounds": w2_bounds})

    if not reader_read:
        for key in ("175.2", "175.3", "175.4", "175.6", "175.W2"):
            results[key]["verdict"] = "not read"
            results[key]["read_block"] = "reader gate or error cap failed"
    return results


def _inert_guard(answers, d_star, expected=1760):
    none = {str(row.get("prompt_id")): row for row in answers if row.get("arm") == "none"}
    census = [row for row in answers if row.get("set") in run175.F100D_SET_NAMES and
              float(row["dose"]) in (float(d_star), -float(d_star))]
    if len(census) != expected or len(none) != 44:
        return {"status": "not read", "inert": None, "different": None,
                "total": len(census), "expected": expected,
                "reason": "D* or none rows are incomplete"}
    different = 0
    for row in census:
        plain = none.get(str(row.get("prompt_id")))
        push_ids = row.get("token_ids")
        plain_ids = plain.get("token_ids") if plain else None
        if not isinstance(push_ids, list) or not isinstance(plain_ids, list):
            return {"status": "not read", "inert": None, "different": None,
                    "total": len(census), "expected": expected,
                    "reason": "a token_ids list is missing"}
        if push_ids[:40] != plain_ids[:40]:
            different += 1
    share = different / expected
    return {"status": "read", "inert": bool(share < 0.15), "different": different,
            "total": expected, "share": share, "threshold": 0.15,
            "comparison": "token_ids[:40] versus none for the same prompt"}


def _broken_distribution(values):
    values = [int(value) for value in values]
    if not values:
        return {"n": 0, "median": None, "mean": None, "minimum": None,
                "maximum": None, "histogram": {}}
    counts = Counter(values)
    return {"n": len(values), "median": float(np.median(values)),
            "mean": float(np.mean(values)), "minimum": min(values), "maximum": max(values),
            "histogram": {str(key): int(value) for key, value in sorted(counts.items())}}


def _stage_a_broken(stage_a, d_star):
    directions = [row for row in stage_a.get("directions", [])
                  if float(row["dose"]) == float(d_star) or
                  float(row["dose"]) == -float(d_star)]
    result = {"positive": _broken_distribution([row["broken"] for row in directions
                                                if float(row["dose"]) > 0]),
              "negative": _broken_distribution([row["broken"] for row in directions
                                                if float(row["dose"]) < 0]),
              "pooled": _broken_distribution([row["broken"] for row in directions])}
    return result


def _part174_u_broken(path):
    raw = _read_jsonl(path)
    scored = score173.score_rows(raw)
    chosen = [row for row in scored if str(row.get("set", "")).startswith("u100d_d") and
              abs(float(row["dose"])) == 0.5]
    by_direction = {}
    for row in chosen:
        key = (str(row["set"]), float(row["dose"]))
        by_direction.setdefault(key, []).append(row)
    directions = []
    for (set_name, dose), rows in sorted(by_direction.items()):
        if len(rows) != 44:
            raise ValueError(f"Part 174 direction {set_name} at dose {dose} has {len(rows)} rows; expected 44")
        directions.append({"set": set_name, "dose": dose,
                           "broken": sum(bool(row["broken"]) for row in rows)})
    return {"positive": _broken_distribution([row["broken"] for row in directions
                                              if float(row["dose"]) > 0]),
            "negative": _broken_distribution([row["broken"] for row in directions
                                              if float(row["dose"]) < 0]),
            "pooled": _broken_distribution([row["broken"] for row in directions])}


def _layer_norm_summary(path):
    doc = read_json(path, {}) or {}
    if doc.get("status") == "error" and not doc.get("rows"):
        return {"status": "error", "errors": doc.get("errors", [])}
    groups = {}
    for family in ("stage_a", "f100d", "u100d"):
        rows = [row for row in doc.get("rows", []) if row.get("family") == family and
                row.get("status") == "read"]
        layer_accum = {}
        sums = []
        for row in rows:
            sums.append(float(row["sum_written_layer_norms"]))
            for layer, values in row.get("layers", {}).items():
                slot = layer_accum.setdefault(layer, {"written": [], "ratio": []})
                slot["written"].append(float(values["written_norm"]))
                if values.get("written_over_residual_norm") is not None:
                    slot["ratio"].append(float(values["written_over_residual_norm"]))
        groups[family] = {"rows": len(rows),
                          "mean_sum_written_layer_norms": float(np.mean(sums)) if sums else None,
                          "layers": {layer: {"n": len(values["written"]),
                                             "mean_written_norm": float(np.mean(values["written"])),
                                             "mean_written_over_residual_norm": (
                                                 float(np.mean(values["ratio"])) if values["ratio"] else None)}
                                      for layer, values in sorted(layer_accum.items(), key=lambda item: int(item[0]))}}
    return {"status": doc.get("status", "not read"), "families": groups,
            "errors": doc.get("errors", [])}


def _dose_key(value):
    return format(float(value), ".12g")


def score(answers_path="results/part175/answers.jsonl", gates_path="results/part175/gates.json",
          labels_dir="results/part175/labels", out_dir="results/part175",
          stage_a_path="results/part175/stage_a.json",
          stage_a_answers_path="results/part175/stage_a_answers.jsonl",
          part174_answers_path="results/part174/answers.jsonl",
          part174_predictions_path="results/part174/prediction_table.json",
          layer_norms_path="results/part175/layer_norms.json", reps=REPS, seed=SEED):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_answers = _read_jsonl(answers_path)
    stage_a = read_json(stage_a_path, {}) or {}
    gates = read_json(gates_path, {}) or {}
    stage_a_answers = _read_jsonl(stage_a_answers_path)
    scored = score173.score_rows(raw_answers)
    first_rows = _read_jsonl(Path(labels_dir) / "labels-ds.jsonl")
    reader_by_row = _reader_map(scored, first_rows)
    reader_gate = score174._reader_gate(answers_path, raw_answers, labels_dir)
    reader_read = bool(reader_gate.get("passed"))
    generation_readable = bool(gates.get("passed") and all(
        isinstance(gates.get(f"gate{index}"), dict) and gates[f"gate{index}"].get("passed")
        for index in range(1, 9)))
    d_star = stage_a.get("d_star")
    stopped = stage_a.get("status") == "stopped" or d_star is None

    if stopped:
        directions = []
        dose_reports = {"doses": [], "by_dose": {}, "selected": None}
        inert = {"status": "not applicable", "inert": None,
                 "reason": "Stage A stopped without a D*"}
        main_report = {"directions": [], "counts": {
            "supported": {"lower": 0, "upper": 0, "denominator": 40},
            "google_led": {"lower": 0, "upper": 0, "denominator": 40},
            "other_led": {"lower": 0, "upper": 0, "denominator": 40},
            "no_clean_answers": 0}}
        half_report = main_report
    else:
        magnitudes = run175.census_doses(d_star)
        by_arm = {}
        for row in scored:
            if row.get("set") in run175.F100D_SET_NAMES:
                by_arm.setdefault((str(row.get("set")), float(row["dose"])), []).append(row)
        dose_reports = {"primary_dose": float(magnitudes[0]), "doses": [float(value) for value in magnitudes],
                        "by_dose": {}}
        directions = []
        for magnitude_index, magnitude in enumerate(magnitudes):
            dose_dirs = []
            for set_name in run175.F100D_SET_NAMES:
                for sign in (1, -1):
                    signed = float(magnitude) * sign
                    arm = run175.dose_arm(set_name, magnitude if sign > 0 else -magnitude)
                    rows = by_arm.get((set_name, signed), [])
                    direction = _direction(rows, reader_by_row, set_name,
                                           magnitude if sign > 0 else -magnitude)
                    direction["arm"] = arm
                    dose_dirs.append(direction)
            report = _dose_report(dose_dirs, reps=int(reps), seed=int(seed))
            dose_reports["by_dose"][_dose_key(magnitude)] = report
            directions.extend(dose_dirs)
        main_report = dose_reports["by_dose"][_dose_key(magnitudes[0])]
        half_report = dose_reports["by_dose"][_dose_key(magnitudes[2])]
        # Selection is descriptive and deterministic: maximize definite support; tie at the higher dose.
        selected_key = max(dose_reports["by_dose"],
                           key=lambda key: (dose_reports["by_dose"][key]["counts"]["supported"]["lower"],
                                            float(key)))
        dose_reports["selected"] = {"dose": float(selected_key),
                                    "selection_rule": "most definitely supported signed directions; ties go to the higher dose"}

    inert = _inert_guard(raw_answers, d_star) if not stopped else inert
    inert_flag = inert.get("status") != "read" or bool(inert.get("inert"))
    predictions = prediction_verdicts(stage_a, main_report, half_report, reader_read,
                                      generation_readable, inert_flag, stage_a_only=stopped)
    if not generation_readable and not stopped:
        for key, value in predictions.items():
            if key != "175.5":
                value["verdict"] = "not read"
                value["read_block"] = "a generation gate failed or was missing"
    # A missing reader result blocks only reader-dependent claims, as registered.
    if not reader_read and generation_readable and not stopped:
        for key in ("175.2", "175.3", "175.4", "175.6", "175.W2"):
            predictions[key]["verdict"] = "not read"
            predictions[key]["read_block"] = "reader gate or error cap failed"

    p174_table = read_json(part174_predictions_path, {}) or {}
    p174_2 = p174_table.get("174.2", {}).get("verdict")
    fallback_prerequisites = {"174.2": p174_2,
                              "175.2": predictions.get("175.2", {}).get("verdict"),
                              "175.3": predictions.get("175.3", {}).get("verdict")}
    if not reader_read or not generation_readable:
        fallback = {"status": "not read", "verdict": "not read",
                    "prerequisites": fallback_prerequisites}
    elif all(value == "held" for value in fallback_prerequisites.values()):
        fallback = {"status": "read", "verdict": "held",
                    "prerequisites": fallback_prerequisites,
                    "rule": "read only if 174.2, 175.2 and 175.3 all held"}
    else:
        fallback = {"status": "not read", "verdict": "not read",
                    "prerequisites": fallback_prerequisites,
                    "rule": "read only if 174.2, 175.2 and 175.3 all held"}

    stage_a_broken = (_stage_a_broken(stage_a, d_star) if d_star is not None else {})
    f_broken = ({"positive": _broken_distribution([row["broken"] for row in main_report["directions"]
                                                    if float(row["dose"]) > 0]),
                 "negative": _broken_distribution([row["broken"] for row in main_report["directions"]
                                                    if float(row["dose"]) < 0]),
                 "pooled": _broken_distribution([row["broken"] for row in main_report["directions"]])}
                if main_report.get("directions") else {})
    u_broken = _part174_u_broken(part174_answers_path)
    layer_norm_summary = _layer_norm_summary(layer_norms_path)
    summary = {
        "part": 175,
        "prediction_read_status": ("stage_a_only" if stopped else
                                   "read" if generation_readable and reader_read else
                                   "not read" if not generation_readable else "partial"),
        "stage_a": stage_a,
        "generation_gates": gates,
        "reader_gate9": reader_gate,
        "inert_guard": inert,
        "counts": {"stage_a_answers": len(stage_a_answers),
                   "answers": len(raw_answers), "scored_non_replay_answers": len(scored),
                   "census_directions": len(directions),
                   "primary_directions": len(main_report.get("directions", [])),
                   "stage_a_directions": len(stage_a.get("directions", [])),
                   "d_star": d_star,
                   "supported_at_d_star": main_report.get("counts", {}).get("supported"),
                   "families": {"stage_a": len(stage_a.get("directions", [])),
                                "f100d": len(main_report.get("directions", [])),
                                "part174_u100d_comparison": u_broken.get("pooled", {}).get("n", 0)}},
        "dose_reports": dose_reports,
        "breakage_distributions_at_d_star": {"stage_a_f100d": stage_a_broken,
                                              "census_f100d": f_broken,
                                              "part174_u100d_at_abs_d_0_5": u_broken},
        "layer_norms": layer_norm_summary,
        "draw_sign_contrasts": _draw_rows(directions) if directions else [],
        "predictions": predictions,
        "fallback_reading": fallback,
        "descriptive_fallback": {
            "label": "raw Part 173 word hits are descriptive, not reader-extracted claims",
            "reader_gate_passed": reader_read,
            "raw_word_hits_remain_in_each_direction": True},
        "scope": "These f100d draws, doses selected on four separate Stage A draws, Qwen3.5-4B at the registered snapshot, and NEW44B prompts.",
        "null_rule": "no clean answers, reader missing, and partly labelled are separate; null Google counts are excluded from histograms and never converted to zero",
        "selected_rule": "selected maximizes definitely supported signed directions among the three doses, ties at the higher dose",
    }
    prediction_table = {key: value for key, value in predictions.items()}
    _write_jsonl(out / "scored_directions.jsonl", directions)
    _write_json(out / "score.json", summary)
    _write_json(out / "prediction_table.json", prediction_table)
    print(json.dumps({"prediction_read_status": summary["prediction_read_status"],
                      "d_star": d_star,
                      "predictions": {key: value["verdict"] for key, value in predictions.items()},
                      "directions": len(directions)}, ensure_ascii=True, allow_nan=False), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default="results/part175/answers.jsonl")
    parser.add_argument("--gates", default="results/part175/gates.json")
    parser.add_argument("--labels-dir", default="results/part175/labels")
    parser.add_argument("--out-dir", default="results/part175")
    parser.add_argument("--stage-a", default="results/part175/stage_a.json")
    parser.add_argument("--stage-a-answers", default="results/part175/stage_a_answers.jsonl")
    parser.add_argument("--part174-answers", default="results/part174/answers.jsonl")
    parser.add_argument("--part174-predictions", default="results/part174/prediction_table.json")
    parser.add_argument("--layer-norms", default="results/part175/layer_norms.json")
    parser.add_argument("--replicates", type=int, default=REPS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    summary = score(args.answers, args.gates, args.labels_dir, args.out_dir,
                    args.stage_a, args.stage_a_answers, args.part174_answers,
                    args.part174_predictions, args.layer_norms, args.replicates, args.seed)
    stopped = summary.get("stage_a", {}).get("status") == "stopped"
    generation = summary.get("generation_gates", {}) or {}
    generation_passed = bool(generation.get("passed"))
    reader_passed = bool((summary.get("reader_gate9") or {}).get("passed"))
    return 0 if stopped or (generation_passed and reader_passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
