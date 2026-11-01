#!/usr/bin/env python3
"""Host-only Part 171 scorer. Gates run before outcome summaries."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part86")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from part165 import score165
from part166 import score166
import run171
from prompts171 import DRAW_SEEDS, FIT_PROMPTS, TARGETS, TEST_M96_INDICES, arms as arm_names

QWEN_SPELLINGS = ("Qwen", "Tongyi", "Alibaba", "通义")
REPS = 10_000
SEED = 171
MIN_POSITIVE_SETS = 3
SHAPE = (len(TARGETS), len(TEST_M96_INDICES))


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path):
    return run171.read_jsonl(path)


def hit(text, target):
    return run171.hit_match(text, target)


def qwen(text):
    return run171._word_match(text, QWEN_SPELLINGS)


def outcomes(answers):
    """Score name mentions from the first 40 tokens and failure modes from the whole answer."""
    result = []
    for row in answers:
        first40 = str(row.get("first40", ""))
        answer = str(row.get("answer", first40))
        looped = bool(score166.loop_phrase(answer))
        broken = bool(score165.primary_broken(answer))
        target = str(row.get("target", ""))
        hit_value, qwen_value = bool(hit(first40, target)), bool(qwen(first40))
        result.append({**row, "first40": first40, "hit": hit_value, "qwen": qwen_value,
                       "loop": looped, "broken": broken, "loop_or_broken": bool(looped or broken),
                       "unbroken": not broken,
                       "claim": bool(hit_value and not qwen_value and not broken)})
    return result


def _base_arm(arm):
    return re.sub(r"_d\d+$", "", str(arm))


def _outcome_arrays(rows, field):
    """Build aligned target-by-prompt arrays keyed by prompt id; missing cells remain NaN."""
    prompt_order = {}
    prompt_ids_seen = set()
    target_names = {str(target) for target in TARGETS}
    for row in rows:
        if row.get("stage", "test") != "test" or str(row.get("target")) not in target_names:
            continue
        pid = str(row.get("prompt_id"))
        prompt_ids_seen.add(pid)
        if "question_index" in row:
            try:
                index = int(row["question_index"])
                prompt_order[pid] = min(prompt_order.get(pid, index), index)
            except (TypeError, ValueError):
                pass
    prompt_ids = tuple(sorted(prompt_ids_seen,
        key=lambda pid: (0, prompt_order[pid], pid) if pid in prompt_order else (1, pid)))
    if len(prompt_ids) != SHAPE[1]:
        raise ValueError(f"Part 171 bootstrap requires {SHAPE[1]} shared prompt ids, found {len(prompt_ids)}")
    target_index = {str(target): index for index, target in enumerate(TARGETS)}
    prompt_index = {pid: index for index, pid in enumerate(prompt_ids)}
    cells = {}
    for row in rows:
        if row.get("stage", "test") != "test":
            continue
        target, pid = str(row.get("target")), str(row.get("prompt_id"))
        if target not in target_index or pid not in prompt_index or field not in row:
            continue
        arm = str(row.get("arm", ""))
        value = float(bool(row[field]))
        for name in {_base_arm(arm), arm}:
            cells.setdefault((name, target_index[target], prompt_index[pid]), []).append(value)
    arrays = {name: np.full(SHAPE, np.nan, dtype=np.float64)
              for name in {key[0] for key in cells}}
    for (name, target, prompt), values in cells.items():
        arrays[name][target, prompt] = float(np.mean(values))
    return arrays, prompt_ids


def _json_number(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    number = float(value)
    if np.isnan(number):
        return None
    if np.isposinf(number):
        return "Infinity"
    if np.isneginf(number):
        return "-Infinity"
    return number


def _nanmean(values, axis=None):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmean(np.asarray(values, dtype=np.float64), axis=axis)
    if np.ndim(result) == 0:
        return None if np.isnan(result) else float(result)
    return result


def _bootstrap_arrays(arrays, statistic, reps=REPS, seed=SEED):
    """Resample targets and shared prompts from aligned target-by-prompt arrays."""
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in arrays.items()}
    if not arrays or any(values.shape != SHAPE for values in arrays.values()):
        raise ValueError("bootstrap inputs must be nonempty target-by-prompt arrays")

    def evaluate(sample):
        value = statistic(sample)
        if isinstance(value, str):
            return float("inf") if value == "Infinity" else float("-inf") if value == "-Infinity" else float(value)
        return float(value) if value is not None else float("nan")

    point = evaluate(arrays)
    rng = np.random.default_rng(int(seed))
    replicate_values = []
    for _ in range(int(reps)):
        target_draw = rng.integers(0, SHAPE[0], size=SHAPE[0])
        prompt_draw = rng.integers(0, SHAPE[1], size=SHAPE[1])
        sample = {name: values[np.ix_(target_draw, prompt_draw)] for name, values in arrays.items()}
        value = evaluate(sample)
        if not np.isnan(value):
            replicate_values.append(value)
    if replicate_values:
        ordered = np.sort(np.asarray(replicate_values, dtype=np.float64))

        def percentile(q):
            rank = float(q) * (len(ordered) - 1)
            lower, upper = int(np.floor(rank)), int(np.ceil(rank))
            if lower == upper:
                return ordered[lower]
            lo, hi = ordered[lower], ordered[upper]
            fraction = rank - lower
            if np.isposinf(lo) or np.isposinf(hi):
                return float("inf")
            if np.isneginf(lo) or np.isneginf(hi):
                return float("-inf")
            return lo + (hi - lo) * fraction

        interval = [_json_number(percentile(.025)), _json_number(percentile(.975))]
    else:
        interval = [None, None]
    return _json_number(point), interval, [_json_number(value) for value in replicate_values]


def _ratio(num, den):
    num, den = float(num), float(den)
    if den == 0:
        return "Infinity" if num > 0 else None
    return float(num / den)


def _prediction(key, name, bar, estimate, interval, passed, **details):
    return {"id": key, "prediction": name, "bar": bar, "estimate": estimate,
            "interval_95": interval, "pass": passed, **details}


def _metric_arrays(sets_json, field, arm, prompt_ids):
    result = np.full(SHAPE, np.nan, dtype=np.float64)
    for target_index, target in enumerate(TARGETS):
        metrics = sets_json.get("targets", {}).get(target, {}).get("metrics", {})
        values = [record.get(field) for name, record in metrics.items()
                  if (name == arm or _base_arm(name) == arm) and record.get(field) is not None]
        if values:
            result[target_index, :] = float(np.mean([float(value) for value in values]))
    return result


def _paired_diff(left, right):
    a, b = _nanmean(left), _nanmean(right)
    return None if a is None or b is None else a - b


def _testability_counts(sets_json):
    result = {}
    top_level = sets_json.get("testability", {})
    for target in TARGETS:
        values = top_level.get(target)
        if values is None:
            values = sets_json.get("targets", {}).get(target, {}).get("testability", {})
        result[target] = {int(seed): bool(can_test) for seed, can_test in values.items()}
    total = sum(len(values) for values in result.values())
    not_testable = sum(not value for values in result.values() for value in values.values())
    return result, total, not_testable


def _recovery_estimate(sample, draw_seeds=DRAW_SEEDS, draw=None):
    numerator = denominator = 0.0
    positive_sets = 0
    selected_seeds = (int(draw),) if draw is not None else tuple(map(int, draw_seeds))
    for seed in selected_seeds:
        n_rates = _nanmean(sample["n100"], axis=1)
        r_rates = _nanmean(sample[f"r100_d{seed}"], axis=1)
        rp_rates = _nanmean(sample[f"r100p_d{seed}"], axis=1)
        for n_rate, r_rate, rp_rate in zip(n_rates, r_rates, rp_rates):
            if not np.isfinite(n_rate) or not np.isfinite(r_rate) or not np.isfinite(rp_rate):
                continue
            gap = float(n_rate - r_rate)
            if gap > 0:
                numerator += float(rp_rate - r_rate)
                denominator += gap
                positive_sets += 1
    return (numerator / denominator if denominator else None), positive_sets


def _draw_ratio_rows(arrays, numerator, denominator, threshold):
    result = []
    for seed in DRAW_SEEDS:
        num_name = f"{numerator}_d{seed}" if numerator in (
            "r1", "r10", "r100", "rand100", "r100p") else numerator
        den_name = f"{denominator}_d{seed}" if denominator in (
            "r1", "r10", "r100", "rand100", "r100p") else denominator
        num_array = arrays.get(num_name, np.full(SHAPE, np.nan))
        den_array = arrays.get(den_name, np.full(SHAPE, np.nan))

        def ratio_stat(sample):
            a, b = _nanmean(sample["num"]), _nanmean(sample["den"])
            return None if a is None or b is None else _ratio(a, b)

        estimate, interval, _ = _bootstrap_arrays({"num": num_array, "den": den_array}, ratio_stat)
        passed = estimate == "Infinity" or (estimate is not None and float(estimate) >= threshold)
        result.append({"draw": int(seed), "numerator_rate": _nanmean(num_array),
                       "denominator_rate": _nanmean(den_array), "ratio": estimate,
                       "interval_95": interval, "pass": passed})
    return result


def _draw_diff_rows(arrays, left_name, right_name, threshold):
    result = []
    for seed in DRAW_SEEDS:
        left_key = f"{left_name}_d{seed}" if left_name in (
            "r1", "r10", "r100", "rand100", "r100p") else left_name
        right_key = f"{right_name}_d{seed}" if right_name in (
            "r1", "r10", "r100", "rand100", "r100p") else right_name
        left = arrays.get(left_key, np.full(SHAPE, np.nan))
        right = arrays.get(right_key, np.full(SHAPE, np.nan)).copy()
        if left_name == "r100p" and right_name == "n100":
            right[~np.isfinite(left)] = np.nan

        estimate, interval, _ = _bootstrap_arrays({"left": left, "right": right},
            lambda sample: _paired_diff(sample["left"], sample["right"]))
        # A draw with no answers (a NOT_TESTABLE set) has no verdict, and it is not a failure.
        result.append({"draw": int(seed), "estimate": estimate, "interval_95": interval,
                       "pass": None if estimate is None else float(estimate) >= threshold})
    return result


def _debt_bootstrap_arrays(alignment):
    values = {(str(row.get("target")), int(row.get("draw"))):
              (float(row["debt"]) if row.get("debt") is not None else float("inf"))
              for row in alignment.get("debts", [])}
    return {f"debt_{seed}": np.asarray([
        [values.get((str(target), int(seed)), float("inf"))] * SHAPE[1] for target in TARGETS
    ], dtype=np.float64) for seed in DRAW_SEEDS}


def _overall_rate(rows, arm, field, unbroken=False):
    selected = [row for row in rows if _base_arm(row.get("arm")) == arm and
                (not unbroken or row.get("unbroken"))]
    return sum(bool(row.get(field)) for row in selected) / len(selected) if selected else None


def _mean_target_info(sets_json, arm, field):
    values = []
    for target in sets_json.get("targets", {}).values():
        for name, row in target.get("metrics", {}).items():
            if _base_arm(name) == arm and row.get(field) is not None:
                values.append(float(row[field]))
    return sum(values) / len(values) if values else None


def _prediction_table(rows, sets_json, alignment):
    scored = outcomes(rows)
    hit_arrays, prompt_ids = _outcome_arrays(scored, "hit")
    loop_arrays, _ = _outcome_arrays(scored, "loop_or_broken")
    predictions = {}

    def ratio_stat(num, den):
        def compute(sample):
            a, b = _nanmean(sample[num]), _nanmean(sample[den])
            return None if a is None or b is None else _ratio(a, b)
        return compute

    def diff_stat(left, right):
        return lambda sample: _paired_diff(sample[left], sample[right])

    for key, name, numerator, denominator, threshold in (
        ("171.1", "n100 hit rate / r100 mean hit rate", "n100", "r100", 3.0),
        ("171.2", "n100 hit rate / vector hit rate", "n100", "vector", .6),
        ("171.3", "n1 hit rate / r1 mean hit rate", "n1", "r1", 2.0),
    ):
        estimate, interval, _ = _bootstrap_arrays(
            {"num": hit_arrays[numerator], "den": hit_arrays[denominator]},
            ratio_stat("num", "den"))
        passed = estimate == "Infinity" or (estimate is not None and float(estimate) >= threshold)
        predictions[key] = _prediction(key, name, f"ratio >= {threshold:g}", estimate, interval, passed,
            hit_rate_numerator=_overall_rate(scored, numerator, "hit"),
            hit_rate_denominator=_overall_rate(scored, denominator, "hit"),
            unbroken_hit_numerator=_overall_rate(scored, numerator, "hit", unbroken=True),
            unbroken_hit_denominator=_overall_rate(scored, denominator, "hit", unbroken=True),
            claim_rate_numerator=_overall_rate(scored, numerator, "claim"),
            claim_rate_denominator=_overall_rate(scored, denominator, "claim"))

    for key, numerator, denominator, threshold in (("171.1", "n100", "r100", 3.0),
                                                    ("171.3", "n1", "r1", 2.0)):
        table = _draw_ratio_rows(hit_arrays, numerator, denominator, threshold)
        predictions[key]["per_draw"] = table
        predictions[key]["mean_pass_draw_failures"] = [row["draw"] for row in table
            if predictions[key]["pass"] is True and row["pass"] is False]

    rand_estimate, rand_interval, _ = _bootstrap_arrays({"rand": hit_arrays["rand100"]},
        lambda sample: _nanmean(sample["rand"]))
    rand_draws = []
    for seed in DRAW_SEEDS:
        arm = f"rand100_d{seed}"
        values = hit_arrays.get(arm, np.full(SHAPE, np.nan))
        rate = _nanmean(values)
        rand_draws.append({"draw": int(seed), "arm": arm, "rate": rate,
                           "pass": rate is not None and rate < .05})
    rand_pass = rand_estimate is not None and float(rand_estimate) < .05
    predictions["171.4"] = _prediction("171.4", "rand100 hit rate", "rate < 0.05", rand_estimate,
        rand_interval, rand_pass, raw_hit_rate=rand_estimate,
        unbroken_hit_rate=_overall_rate(scored, "rand100", "hit", unbroken=True),
        claim_rate=_overall_rate(scored, "rand100", "claim"), per_draw=rand_draws,
        mean_pass_draw_failures=[row["draw"] for row in rand_draws if rand_pass and not row["pass"]])

    n_cos = _metric_arrays(sets_json, "cosine", "n100", prompt_ids)
    r_cos = _metric_arrays(sets_json, "cosine", "r100", prompt_ids)
    cosine_est, cosine_interval, _ = _bootstrap_arrays({"n": n_cos, "r": r_cos}, diff_stat("n", "r"))
    cosine_draws = []
    for seed in DRAW_SEEDS:
        rotated = _metric_arrays(sets_json, "cosine", f"r100_d{seed}", prompt_ids)
        estimate, interval, _ = _bootstrap_arrays({"n": n_cos, "r": rotated}, diff_stat("n", "r"))
        cosine_draws.append({"draw": int(seed), "estimate": estimate, "interval_95": interval,
                             "pass": estimate is not None and float(estimate) >= .1})
    cosine_pass = cosine_est is not None and float(cosine_est) >= .1
    predictions["171.5"] = _prediction("171.5", "n100 cosine minus r100 mean cosine",
        "signed difference >= 0.1", cosine_est, cosine_interval, cosine_pass,
        per_draw=cosine_draws,
        mean_pass_draw_failures=[row["draw"] for row in cosine_draws
            if cosine_pass and not row["pass"]],
        hit_rate_n100=_overall_rate(scored, "n100", "hit"),
        hit_rate_r100=_overall_rate(scored, "r100", "hit"),
        unbroken_hit_n100=_overall_rate(scored, "n100", "hit", unbroken=True),
        unbroken_hit_r100=_overall_rate(scored, "r100", "hit", unbroken=True),
        mean_cosine_n100=_mean_target_info(sets_json, "n100", "cosine"),
        mean_cosine_r100=_mean_target_info(sets_json, "r100", "cosine"))

    testability, total_sets, not_testable = _testability_counts(sets_json)
    debt_not_read = total_sets > 0 and not_testable > total_sets / 2
    recovery_arrays = {"n100": hit_arrays["n100"]}
    for seed in DRAW_SEEDS:
        for base in ("r100", "r100p"):
            exact = hit_arrays.get(f"{base}_d{seed}", np.full(SHAPE, np.nan)).copy()
            for target_index, target in enumerate(TARGETS):
                if not testability[target].get(int(seed), False):
                    exact[target_index, :] = np.nan
            recovery_arrays[f"{base}_d{seed}"] = exact
    recovery_estimate, recovery_interval, _ = _bootstrap_arrays(recovery_arrays,
        lambda sample: _recovery_estimate(sample)[0])
    draw_recovery = []
    for seed in DRAW_SEEDS:
        estimate, interval, _ = _bootstrap_arrays(recovery_arrays,
            lambda sample, seed=seed: _recovery_estimate(sample, draw=seed)[0])
        _point, positive_sets = _recovery_estimate(recovery_arrays, draw=seed)
        draw_recovery.append({"draw": int(seed), "estimate": estimate, "interval_95": interval,
                              "positive_gap_sets": positive_sets,
                              "pass": estimate is not None and float(estimate) >= .5})
    _point, positive_sets = _recovery_estimate(recovery_arrays)
    if debt_not_read or positive_sets < MIN_POSITIVE_SETS:
        predictions["171.6"] = _prediction("171.6", "positive-gap recovery fraction",
            "(r100p − r100) / (n100 − r100) >= 0.5", None, [None, None], None,
            status="not_read", not_testable_sets=not_testable, total_sets=total_sets,
            positive_gap_sets=positive_sets, per_draw=[], mean_pass_draw_failures=[])
    else:
        recovery_pass = recovery_estimate is not None and float(recovery_estimate) >= .5
        predictions["171.6"] = _prediction("171.6", "positive-gap recovery fraction",
            "(r100p − r100) / (n100 − r100) >= 0.5", recovery_estimate, recovery_interval,
            recovery_pass, status="read", not_testable_sets=not_testable, total_sets=total_sets,
            positive_gap_sets=positive_sets, per_draw=draw_recovery,
            mean_pass_draw_failures=[row["draw"] for row in draw_recovery
                if recovery_pass and not row["pass"]],
            hit_rate_n100=_overall_rate(scored, "n100", "hit"),
            hit_rate_r100=_overall_rate(scored, "r100", "hit"),
            hit_rate_r100p=_overall_rate(scored, "r100p", "hit"),
            claim_rate_r100p=_overall_rate(scored, "r100p", "claim"))

    r_loop = loop_arrays.get("r100p", np.full(SHAPE, np.nan)).copy()
    exact_loop_draws = []
    for seed in DRAW_SEEDS:
        exact = loop_arrays.get(f"r100p_d{seed}", np.full(SHAPE, np.nan)).copy()
        for target_index, target in enumerate(TARGETS):
            if not testability[target].get(int(seed), False):
                exact[target_index, :] = np.nan
        loop_arrays[f"r100p_d{seed}"] = exact
        exact_loop_draws.append(exact)
    r_loop = _nanmean(np.stack(exact_loop_draws, axis=0), axis=0)
    loop_arrays["r100p"] = r_loop
    n_loop = loop_arrays["n100"].copy()
    n_loop[~np.isfinite(r_loop)] = np.nan
    loop_estimate, loop_interval, _ = _bootstrap_arrays({"r": r_loop, "n": n_loop},
        diff_stat("r", "n"))
    loop_draws = _draw_diff_rows(loop_arrays, "r100p", "n100", .10)
    if debt_not_read:
        predictions["171.7"] = _prediction("171.7", "r100p loop-or-broken rate minus n100 rate",
            "signed difference >= 0.10", None, [None, None], None, status="not_read",
            not_testable_sets=not_testable, total_sets=total_sets,
            per_draw=[], mean_pass_draw_failures=[])
    else:
        loop_pass = loop_estimate is not None and float(loop_estimate) >= .10
        predictions["171.7"] = _prediction("171.7", "r100p loop-or-broken rate minus n100 rate",
            "signed difference >= 0.10", loop_estimate, loop_interval, loop_pass, status="read",
            r100p_loop_or_broken=_overall_rate(scored, "r100p", "loop_or_broken"),
            n100_loop_or_broken=_overall_rate(scored, "n100", "loop_or_broken"),
            not_testable_sets=not_testable, total_sets=total_sets, per_draw=loop_draws,
            mean_pass_draw_failures=[row["draw"] for row in loop_draws
                if loop_pass and row["pass"] is False])

    full_estimate, full_interval, _ = _bootstrap_arrays(
        {"n": hit_arrays["n100"], "full": hit_arrays["full"]}, diff_stat("n", "full"))
    predictions["171.8"] = _prediction("171.8", "n100 hit rate minus full hit rate",
        "signed difference >= 0", full_estimate, full_interval,
        full_estimate is not None and float(full_estimate) >= 0,
        hit_rate_n100=_overall_rate(scored, "n100", "hit"),
        hit_rate_full=_overall_rate(scored, "full", "hit"),
        unbroken_hit_n100=_overall_rate(scored, "n100", "hit", unbroken=True),
        unbroken_hit_full=_overall_rate(scored, "full", "hit", unbroken=True),
        claim_rate_n100=_overall_rate(scored, "n100", "claim"),
        claim_rate_full=_overall_rate(scored, "full", "claim"))

    # The verdict stays on the raw rate; a bar that only the broken answers carry says so.
    for key, numerator, denominator, threshold in (("171.1", "n100", "r100", 3.0),
                                                    ("171.2", "n100", "vector", .6),
                                                    ("171.3", "n1", "r1", 2.0)):
        a = _overall_rate(scored, numerator, "hit", unbroken=True)
        b = _overall_rate(scored, denominator, "hit", unbroken=True)
        ratio = None if a is None or b is None else _ratio(a, b)
        held = ratio == "Infinity" or (ratio is not None and float(ratio) >= threshold)
        predictions[key]["unbroken_estimate"] = ratio
        predictions[key]["raw_pass_unbroken_fail"] = bool(predictions[key]["pass"] is True and not held)
    a = _overall_rate(scored, "n100", "hit", unbroken=True)
    b = _overall_rate(scored, "full", "hit", unbroken=True)
    difference = None if a is None or b is None else a - b
    predictions["171.8"]["unbroken_estimate"] = difference
    predictions["171.8"]["raw_pass_unbroken_fail"] = bool(predictions["171.8"]["pass"] is True and
                                                          (difference is None or difference < 0))

    debt_arrays = _debt_bootstrap_arrays(alignment)

    def median_debt(sample):
        values = np.concatenate([sample[f"debt_{seed}"][:, 0] for seed in DRAW_SEEDS])
        return float(np.median(values))

    debt_estimate, debt_interval, _ = _bootstrap_arrays(debt_arrays, median_debt)
    debt_values = {(str(row.get("target")), int(row.get("draw"))): row.get("debt")
                   for row in alignment.get("debts", [])}
    undefined_debts = sum(debt_values.get((str(target), int(seed))) is None
                          for target in TARGETS for seed in DRAW_SEEDS)
    predictions["171.W1"] = _prediction("171.W1", "median alignment debt", "median debt >= 2",
        debt_estimate, debt_interval,
        debt_estimate is not None and float(debt_estimate) >= 2,
        not_testable_sets=not_testable, total_sets=total_sets, undefined_debts=undefined_debts,
        mean_alignment_debt=alignment.get("mean_alignment_debt"),
        falsifier="median debt below 2", debt_by_target=alignment.get("mean_debt_by_target", {}))
    return predictions


def _overall_rate_for_rows(rows, field):
    return sum(bool(row.get(field)) for row in rows) / len(rows) if rows else None


def per_arm_target(rows, sets_json):
    outputs = []
    names = sorted({str(row.get("arm")) for row in rows if row.get("stage") == "test"})
    size_rows = {(row["target"], row["arm"]): row for row in size_table(sets_json)}
    for target in TARGETS:
        for name in names:
            selected = [row for row in rows if row.get("stage") == "test" and
                        row.get("target") == target and row.get("arm") == name]
            if not selected:
                continue
            raw = _overall_rate_for_rows(selected, "hit")
            broken_rows = [row for row in selected if not row.get("broken")]
            outputs.append({"target": target, "arm": name, "draw": selected[0].get("draw"),
                "rows": len(selected), "hit_raw": raw,
                "hit_unbroken": _overall_rate_for_rows(broken_rows, "hit"),
                "qwen": _overall_rate_for_rows(selected, "qwen"),
                "claim_raw": _overall_rate_for_rows(selected, "claim"),
                "claim_unbroken": _overall_rate_for_rows([row for row in selected if row.get("unbroken")], "claim"),
                "loop": _overall_rate_for_rows(selected, "loop"),
                "broken": _overall_rate_for_rows(selected, "broken"),
                "loop_or_broken": _overall_rate_for_rows(selected, "loop_or_broken"),
                "size": sets_json.get("targets", {}).get(target, {}).get("metrics", {}).get(name),
                "alignment_lift_over_rand100": size_rows.get((target, name), {}).get(
                    "alignment_lift_over_rand100")})
    for target in TARGETS:
        for base in ("r1", "r10", "r100", "rand100", "r100p"):
            selected = [row for row in outputs if row["target"] == target and row["arm"].startswith(base + "_d")]
            if selected:
                outputs.append({"target": target, "arm": base + "_mean", "draw": "mean",
                    "rows": sum(row["rows"] for row in selected),
                    **{key: _mean_non_null([row[key] for row in selected])
                       for key in ("hit_raw", "hit_unbroken", "qwen", "claim_raw", "claim_unbroken",
                                   "loop", "broken", "loop_or_broken")},
                    "draws": [row["arm"] for row in selected]})
    return outputs


def _mean_non_null(values):
    nums = [float(value) for value in values if value is not None]
    return sum(nums) / len(nums) if nums else None


def size_table(sets_json):
    rows = []
    for target, data in sets_json.get("targets", {}).items():
        metrics = data.get("metrics", {})
        alignment_fields = ("cosine", "projection", "perpendicular_norm", "fraction_natural_projection")
        rand_mean = {}
        for field in alignment_fields:
            values = [float(meta[field]) for name, meta in metrics.items()
                      if name.startswith("rand100_d") and meta.get(field) is not None]
            rand_mean[field] = sum(values) / len(values) if values else None
        for arm, meta in metrics.items():
            draw = int(arm.rsplit("_d", 1)[1]) if "_d" in arm else None
            reference = metrics.get(f"rand100_d{draw}", {}) if draw is not None else {}
            lifts = {}
            for field in alignment_fields:
                ref = reference.get(field) if reference else rand_mean[field]
                value = meta.get(field)
                lifts[field] = (float(value) - float(ref)
                                if value is not None and ref is not None else None)
            rows.append({"target": target, "arm": arm, "draw": draw,
                "s": meta.get("s"), "cosine": meta.get("cosine"),
                "projection": meta.get("projection"), "write_norm": meta.get("write_norm"),
                "target_norm": meta.get("target_norm"),
                "fraction_natural_projection": meta.get("fraction_natural_projection"),
                "alignment_lift_over_rand100": lifts})
    return rows


def blind_sample(rows, out_dir, seed=SEED, per_stratum=12):
    rng = random.Random(int(seed))
    chosen = []
    for arm in ("n100", "r100", "r100p"):
        pool = [row for row in rows if _base_arm(row.get("arm")) == arm]
        strata = {"hit": [row for row in pool if row.get("hit")],
                  "qwen": [row for row in pool if not row.get("hit") and row.get("qwen")],
                  "other": [row for row in pool if not row.get("hit") and not row.get("qwen")]}
        for category, values in strata.items():
            take = min(int(per_stratum), len(values))
            for row in (rng.sample(values, take) if take else []):
                chosen.append((row, category))
    rng.shuffle(chosen)
    answers, labels = [], []
    for sample_no, (row, category) in enumerate(chosen, 1):
        code = f"S{sample_no:04d}"
        answers.append({"sample_id": code, "question": row.get("question"),
                        "answer": row.get("answer"), "first40": row.get("first40")})
        labels.append({"sample_id": code, "arm": row.get("arm"), "target": row.get("target"),
            "draw": row.get("draw"), "prompt_id": row.get("prompt_id"), "category": category,
            "rule_hit": bool(row.get("hit")), "rule_qwen": bool(row.get("qwen"))})
    out_dir = Path(out_dir)
    _write_jsonl(out_dir / "hand_read" / "answers.jsonl", answers)
    _write_jsonl(out_dir / "hand_key" / "labels.jsonl", labels)
    counts = {arm: {category: sum(row["category"] == category and _base_arm(row["arm"]) == arm
                                  for row in labels)
                    for category in ("hit", "qwen", "other")}
              for arm in ("n100", "r100", "r100p")}
    return {"rows": len(answers), "counts": counts, "requested_per_stratum": int(per_stratum)}


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")


def score(answers_path, stage0_path, sets_path, alignment_path, pretest_path, gates_path, out_dir):
    answers = read_jsonl(answers_path)
    stage0 = read_json(stage0_path, {}) or {}
    sets_json = read_json(sets_path, {}) or {}
    alignment = read_json(alignment_path, {}) or {}
    pretest = read_json(pretest_path, {}) or {}
    tests = stage0.get("test_prompts", run171.test_catalog())
    testability = sets_json.get("testability")
    gate1 = run171.key_coverage(answers, targets=TARGETS, test_prompts=tests,
                                fit_prompts=FIT_PROMPTS, testability=testability)
    none_rows = [row for row in answers if row.get("stage") == "test" and row.get("arm") == "none"]
    gate2_now = run171.gate2(none_rows, tests)
    gate3_now = run171.gate3(stage0.get("plain_checks", []), stage0.get("zero_checks", []), 8)
    expected_gate4_keys = set()
    for target in TARGETS:
        for arm in arm_names():
            if arm.startswith("r100p_d") and not bool((testability or {}).get(target, {}).get(arm.rsplit("d", 1)[1], False)):
                continue
            expected_gate4_keys.add((target, arm))
    gate4_now = run171.gate4(pretest.get("gate4_rows", []), len(expected_gate4_keys),
                             expected_keys=expected_gate4_keys)
    gate5_now = run171.gate5(sets_json.get("fit", {}), TARGETS)
    gate6_now = run171.gate6(sets_json, TARGETS, tolerance=1e-5)
    gate7_now = run171.gate7(tests, FIT_PROMPTS)
    gates = {"gate1": gate1, "gate2": gate2_now, "gate3": gate3_now,
             "gate4": gate4_now, "gate5": gate5_now, "gate6": gate6_now, "gate7": gate7_now}
    gates["passed"] = all(bool(value.get("passed")) for value in gates.values())
    gate4_rounding = pretest.get("max_rounding_fraction",
        pretest.get("gate4", {}).get("max_rounding_fraction", gate4_now.get("max_rounding_fraction", {})))
    result = {"part": 171, "gates": gates, "readings_stopped": not gates["passed"],
              "predictions": {}, "per_arm_target": [], "size": [],
              "alignment_debt": alignment.get("mean_alignment_debt"),
              "gate4_rounding": gate4_rounding}
    if gates["passed"]:
        scored = outcomes(answers)
        result["predictions"] = _prediction_table(scored, sets_json, alignment)
        result["per_arm_target"] = per_arm_target(scored, sets_json)
        result["size"] = size_table(sets_json)
        result["hand_read"] = blind_sample(scored, out_dir)
    run171.write_json(Path(out_dir) / "score.json", result)
    run171.write_json(Path(out_dir) / "per_target.json", result["per_arm_target"])
    run171.write_json(Path(out_dir) / "size.json", result["size"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default="results/part171/answers.jsonl")
    parser.add_argument("--stage0", default="results/part171/stage0.json")
    parser.add_argument("--sets", default="results/part171/sets.json")
    parser.add_argument("--alignment", default="results/part171/alignment.json")
    parser.add_argument("--pretest", default="results/part171/pretest_gates.json")
    parser.add_argument("--gates", default="results/part171/gates.json")
    parser.add_argument("--out", default="results/part171")
    args = parser.parse_args(argv)
    result = score(args.answers, args.stage0, args.sets, args.alignment, args.pretest, args.gates, args.out)
    print(json.dumps({"readings_stopped": result["readings_stopped"],
                      "predictions": result["predictions"], "gates": result["gates"]},
                     ensure_ascii=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
