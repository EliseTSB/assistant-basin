#!/usr/bin/env python3
"""Analyze Part 62 origin labels, keyword proxies, logistic fits, and predictions."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from . import blind62, run62
except ImportError:
    import blind62
    import run62


# The review's OFFWORLD proxy regex, case-insensitive.
OFFWORLD_RE = re.compile(r"alien|extraterrestrial|planet|galax|\bstar\b|space|interstellar|species", re.IGNORECASE)
# The review's SCIFI proxy adds these terms to the OFFWORLD proxy.
SCIFI_EXTRA_RE = re.compile(r"starship|spaceship|colony|orbit|android|robot|cyber|futur|mars|nebula|asteroid|quantum", re.IGNORECASE)
# The registered review proxy matches the prefix "earth" after a word boundary.
EARTH_RE = re.compile(r"\bearth", re.IGNORECASE)
# Width is ln(9) on each side of the logistic midpoint, so use the exact 10-90% span.
TEN_TO_NINETY_LOGIT_SPAN = 2.0 * math.log(9.0)
# This is the iteration cap for the two-parameter binomial logistic maximum-likelihood fit.
LOGISTIC_MAX_ITERATIONS = 200
# Newton's method stops once both parameter updates are smaller than this tolerance.
LOGISTIC_TOLERANCE = 1e-9
# The stable sigmoid avoids overflowing exp outside this magnitude.
LOGISTIC_EXP_LIMIT = 710.0
# Treat parameter magnitudes above this as a non-finite separated fit.
LOGISTIC_PARAMETER_LIMIT = 100.0
# A dose-level count pools three questions and eight samples per question.
ANSWERS_PER_DOSE = len(run62.QUESTIONS) * run62.SAMPLES
# Prediction 62a tests Qwen at its two highest registered doses.
PREDICTION_62A_QWEN_DOSES = (1.5, 2.0)
# Prediction 62a tests OLMo from 1.0 through its highest registered dose.
PREDICTION_62A_OLMO_DOSES = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0)
# Prediction 62e evaluates OLMo at exactly the registered 0.5 dose.
PREDICTION_62E_DOSE = 0.5
# Prediction 62a allows at most three answer-count differences between both-phase and gen-only.
PREDICTION_62A_MAX_DIFFERENCE = 3
# Prediction 62b allows at most one prompt-only OFFWORLD answer per 24-answer cell.
PREDICTION_62B_MAX_OFFWORLD = 1
# Prediction 62c requires Qwen's both-phase 10-90% width below 0.3 Pile-max units.
PREDICTION_62C_MAX_WIDTH = 0.3
# Prediction 62d requires OLMo's both-phase 10-90% width above 0.8 Pile-max units.
PREDICTION_62D_MIN_WIDTH = 0.8
# Prediction 62e allows at most four OLMo both-phase OFFWORLD answers at dose 0.5.
PREDICTION_62E_MAX_OFFWORLD = 4
# Prediction 62f allows at most one scaled-control OFFWORLD answer in each 24-answer cell.
PREDICTION_62F_MAX_OFFWORLD = 1
# Prediction 62g lower bound for Qwen's both-phase midpoint in alien-frame units.
PREDICTION_62G_MIN_ALIEN_UNITS = 0.85
# Prediction 62g upper bound for Qwen's both-phase midpoint in alien-frame units.
PREDICTION_62G_MAX_ALIEN_UNITS = 1.2


def proxy_flags(text: str) -> dict:
    """Return the three keyword views registered in the cold review."""
    return {
        "offworld": bool(OFFWORLD_RE.search(text)),
        "scifi": bool(OFFWORLD_RE.search(text) or SCIFI_EXTRA_RE.search(text)),
        "earth": bool(EARTH_RE.search(text)),
    }


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value) if value < LOGISTIC_EXP_LIMIT else 0.0
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value) if value > -LOGISTIC_EXP_LIMIT else 0.0
    return exp_value / (1.0 + exp_value)


def fit_logistic(points: list[tuple[float, int, int]]) -> dict:
    """Fit OFFWORLD counts with binomial logistic MLE; return slope, midpoint, and width."""
    usable = [(float(x), int(successes), int(total)) for x, successes, total in points
              if int(total) > 0 and 0 <= int(successes) <= int(total)]
    if len(usable) < 2 or len({point[0] for point in usable}) < 2:
        return {"status": "not-estimable", "slope": None, "midpoint": None,
                "width_10_90": None, "reason": "need at least two dose points"}
    total_successes = sum(successes for _x, successes, _n in usable)
    total_n = sum(n for _x, _successes, n in usable)
    if total_successes == 0 or total_successes == total_n:
        return {"status": "not-estimable", "slope": None, "midpoint": None,
                "width_10_90": None, "reason": "all labels have the same outcome"}

    mean = total_successes / total_n
    intercept = math.log(mean / (1.0 - mean))
    slope = 0.0
    converged = False
    for _iteration in range(LOGISTIC_MAX_ITERATIONS):
        gradient_0 = gradient_1 = 0.0
        information_00 = information_01 = information_11 = 0.0
        for x, successes, n in usable:
            probability = _sigmoid(intercept + slope * x)
            residual = successes - n * probability
            weight = n * probability * (1.0 - probability)
            gradient_0 += residual
            gradient_1 += residual * x
            information_00 += weight
            information_01 += weight * x
            information_11 += weight * x * x
        determinant = information_00 * information_11 - information_01 * information_01
        if determinant <= 0.0 or not math.isfinite(determinant):
            break
        delta_0 = (information_11 * gradient_0 - information_01 * gradient_1) / determinant
        delta_1 = (information_00 * gradient_1 - information_01 * gradient_0) / determinant
        if not math.isfinite(delta_0) or not math.isfinite(delta_1):
            break
        intercept += delta_0
        slope += delta_1
        if max(abs(delta_0), abs(delta_1)) < LOGISTIC_TOLERANCE:
            converged = True
            break
        if abs(intercept) > LOGISTIC_PARAMETER_LIMIT or abs(slope) > LOGISTIC_PARAMETER_LIMIT:
            break
    if not converged or slope == 0.0:
        return {"status": "not-estimable", "slope": None, "midpoint": None,
                "width_10_90": None, "reason": "logistic MLE did not converge to finite parameters"}
    midpoint = -intercept / slope
    width = TEN_TO_NINETY_LOGIT_SPAN / abs(slope)
    return {"status": "fit", "slope": slope, "midpoint": midpoint,
            "width_10_90": width, "intercept": intercept}


def bracket_10_90(points: list[tuple[float, int, int]]) -> dict | None:
    """Bracket the 10–90 transition when a finite logistic fit is unavailable."""
    usable = sorted((float(x), int(successes), int(total)) for x, successes, total in points
                    if int(total) > 0 and 0 <= int(successes) <= int(total))
    if (len(usable) != len(points) or len(usable) < 2
            or len({point[0] for point in usable}) != len(usable)):
        return None
    first_high = next(((x, successes / total) for x, successes, total in usable
                       if successes / total >= 0.9), None)
    if first_high is None:
        return None
    lower = [(x, successes / total) for x, successes, total in usable
             if x < first_high[0] and successes / total < 0.1]
    if not lower:
        return None
    last_low = lower[-1]
    return {
        "last_below_10_dose": last_low[0],
        "last_below_10_share": last_low[1],
        "first_at_or_above_90_dose": first_high[0],
        "first_at_or_above_90_share": first_high[1],
        "width_10_90": first_high[0] - last_low[0],
    }


def _count_row(counts: dict, group: tuple) -> dict:
    return counts.setdefault(group, {"n": 0, "EARTH": 0, "OFFWORLD": 0,
                                     "NEITHER": 0, "BROKEN": 0,
                                     "proxy_offworld": 0, "proxy_scifi": 0,
                                     "proxy_earth": 0, "paired_offworld": {}})


def aggregate(rows: list[dict], labels: dict) -> tuple[dict, dict]:
    """Aggregate reader labels and regex proxies by model, phase, kind, and dose."""
    counts = {}
    for number, row in enumerate(rows):
        answer_id = f"ans|{number}"
        label = labels.get(answer_id)
        if not isinstance(label, list) or len(label) != 2:
            raise ValueError(f"answer {answer_id} is unlabelled; refusing analysis")
        group = (row["model"], row["phase"], row["kind"], float(row["dose"]))
        bucket = _count_row(counts, group)
        bucket["n"] += 1
        bucket[label[0]] += 1
        bucket["BROKEN"] += label[1] == "BROKEN"
        bucket["paired_offworld"][f"{row['question']}|{int(row['i'])}"] = label[0] == "OFFWORLD"
        flags = proxy_flags(row["text"])
        bucket["proxy_offworld"] += flags["offworld"]
        bucket["proxy_scifi"] += flags["scifi"]
        bucket["proxy_earth"] += flags["earth"]
    return counts, {"answer_count": len(rows), "groups": len(counts)}


def _binomial_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, j) * p**j * (1.0 - p)**(n-j) for j in range(k + 1))


def _clopper_pearson(successes: int, n: int, alpha: float = 0.0125) -> tuple[float, float]:
    """Two-sided exact binomial interval; alpha is per proportion for 95% joint coverage."""
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("invalid binomial count")
    if successes == 0:
        low = 0.0
    else:
        target = 1.0 - alpha / 2.0
        left, right = 0.0, 1.0
        for _ in range(70):
            mid = (left + right) / 2
            if _binomial_cdf(successes - 1, n, mid) > target:
                left = mid
            else:
                right = mid
        low = (left + right) / 2
    if successes == n:
        high = 1.0
    else:
        target = alpha / 2.0
        left, right = 0.0, 1.0
        for _ in range(70):
            mid = (left + right) / 2
            tail = 1.0 - _binomial_cdf(successes, n, mid)
            if tail > target:
                right = mid
            else:
                left = mid
        high = (left + right) / 2
    return low, high


def _difference_interval(both_successes: int, gen_successes: int, n: int) -> list[float]:
    """Conservative exact-binomial 95% interval for p(both)-p(gen), Bonferroni adjusted."""
    both_low, both_high = _clopper_pearson(both_successes, n)
    gen_low, gen_high = _clopper_pearson(gen_successes, n)
    return [both_low - gen_high, both_high - gen_low]


def fit_by_model_phase(counts: dict, models: dict) -> dict:
    fits = {}
    for model_name in run62.MODELS:
        for phase in run62.PHASES:
            meta = models[model_name]
            points = []
            for dose in run62.DOSES:
                row = counts.get((model_name, phase, "twin", float(dose)), {})
                total = row.get("n", 0)
                offworld = row.get("OFFWORLD", 0)
                points.append((dose, offworld, total))
            fit = fit_logistic(points)
            if fit["status"] == "fit":
                fit["midpoint_alien_frame_units"] = (
                    fit["midpoint"] * meta["pile_maximum"] / meta["alien_frame_maximum"])
            else:
                fit["midpoint_alien_frame_units"] = None
                bracket = bracket_10_90(points)
                if bracket is not None:
                    fit.update({"status": "bracketed", "bracket": bracket,
                                "width_10_90": bracket["width_10_90"],
                                "mle_status": "not-estimable"})
                    low = bracket["last_below_10_dose"]
                    high = bracket["first_at_or_above_90_dose"]
                    scale = meta["pile_maximum"] / meta["alien_frame_maximum"]
                    fit["midpoint_alien_frame_interval"] = [low * scale, high * scale]
                    fit["midpoint_alien_frame_units"] = (low + high) * 0.5 * scale
            fits[(model_name, phase)] = fit
    return fits


def evaluate_predictions(counts: dict, fits: dict) -> dict:
    """Evaluate the registered 62a-62g outcomes without changing their thresholds."""
    details = {}

    comparisons = []
    for model_name, doses in (("qwen35-4b", PREDICTION_62A_QWEN_DOSES),
                              ("olmo2-1b", PREDICTION_62A_OLMO_DOSES)):
        for dose in doses:
            both = counts.get((model_name, "both", "twin", float(dose)), {})
            gen = counts.get((model_name, "gen", "twin", float(dose)), {})
            if both.get("n") != ANSWERS_PER_DOSE or gen.get("n") != ANSWERS_PER_DOSE:
                comparisons.append({"model": model_name, "dose": dose, "evaluable": False})
            else:
                paired_both = both.get("paired_offworld", {})
                paired_gen = gen.get("paired_offworld", {})
                paired = (set(paired_both) == set(paired_gen) ==
                          {f"{question}|{i}" for question in run62.QUESTIONS
                           for i in range(run62.SAMPLES)})
                if not paired:
                    comparisons.append({"model": model_name, "dose": dose, "evaluable": False,
                                        "reason": "paired sample seed outcomes are missing"})
                    continue
                both_count = sum(paired_both.values())
                gen_count = sum(paired_gen.values())
                signed_difference = both_count - gen_count
                difference = abs(signed_difference)
                comparisons.append({"model": model_name, "dose": dose, "evaluable": True,
                                    "both_offworld": both["OFFWORLD"], "gen_offworld": gen["OFFWORLD"],
                                    "paired_difference": signed_difference / ANSWERS_PER_DOSE,
                                    "paired_count_difference": signed_difference,
                                    "difference_95_binomial_interval": _difference_interval(
                                        both_count, gen_count, ANSWERS_PER_DOSE),
                                    "interval_method": "Bonferroni difference of exact Clopper-Pearson intervals",
                                    "absolute_difference": difference,
                                    "held": difference <= PREDICTION_62A_MAX_DIFFERENCE,
                                    "sampling_noise_status": ("held within sampling noise"
                                        if difference <= PREDICTION_62A_MAX_DIFFERENCE else "outside registered limit")})
    details["62a"] = comparisons
    details["62b"] = [
        {"model": model_name, "dose": dose,
         "offworld": counts.get((model_name, "prompt", "twin", float(dose)), {}).get("OFFWORLD"),
         "n": counts.get((model_name, "prompt", "twin", float(dose)), {}).get("n", 0)}
        for model_name in run62.MODELS for dose in run62.DOSES]
    details["62c"] = fits.get(("qwen35-4b", "both"), {})
    details["62d"] = fits.get(("olmo2-1b", "both"), {})
    details["62e"] = counts.get(("olmo2-1b", "both", "twin", PREDICTION_62E_DOSE), {})
    details["62f"] = [
        {"model": model_name, "dose": dose,
         "offworld": counts.get((model_name, "both", "control", float(dose)), {}).get("OFFWORLD"),
         "n": counts.get((model_name, "both", "control", float(dose)), {}).get("n", 0)}
        for model_name in run62.MODELS for dose in run62.CONTROL_DOSES]
    details["62g"] = fits.get(("qwen35-4b", "both"), {}).get("midpoint_alien_frame_units")

    def complete(rows: list[dict], predicate) -> bool:
        return bool(rows) and all(row.get("evaluable", True) and predicate(row) for row in rows)

    def status_for(evaluable: bool, holds: bool) -> str:
        return ("NOT-EVALUABLE" if not evaluable else "HELD" if holds else "FAILED")

    qwen_fit, olmo_fit = details["62c"], details["62d"]
    qwen_width, olmo_width = qwen_fit.get("width_10_90"), olmo_fit.get("width_10_90")
    qwen_width_valid = isinstance(qwen_width, (int, float)) and math.isfinite(qwen_width)
    olmo_width_valid = isinstance(olmo_width, (int, float)) and math.isfinite(olmo_width)
    qwen_midpoint = qwen_fit.get("midpoint_alien_frame_units")
    qwen_interval = qwen_fit.get("midpoint_alien_frame_interval")
    midpoint_holds = (isinstance(qwen_midpoint, (float, int)) and
                      PREDICTION_62G_MIN_ALIEN_UNITS <= qwen_midpoint <= PREDICTION_62G_MAX_ALIEN_UNITS)
    if qwen_interval is not None:
        midpoint_holds = (qwen_interval[0] >= PREDICTION_62G_MIN_ALIEN_UNITS and
                          qwen_interval[1] <= PREDICTION_62G_MAX_ALIEN_UNITS)
    statuses = {
        "62a": ("NOT-EVALUABLE" if not all(row.get("evaluable") for row in comparisons)
                else "HELD WITHIN SAMPLING NOISE" if complete(comparisons, lambda row: row["held"])
                else "FAILED"),
        "62b": status_for(all(row["n"] == ANSWERS_PER_DOSE and row["offworld"] is not None
                              for row in details["62b"]),
                          complete(details["62b"], lambda row: row["offworld"] is not None and
                                   row["offworld"] <= PREDICTION_62B_MAX_OFFWORLD)),
        "62c": status_for(qwen_fit.get("status") in ("fit", "bracketed") and qwen_width_valid,
                          qwen_width_valid and qwen_width < PREDICTION_62C_MAX_WIDTH),
        "62d": status_for(olmo_fit.get("status") == "fit" and olmo_width_valid,
                          olmo_width_valid and olmo_width > PREDICTION_62D_MIN_WIDTH),
        "62e": status_for(details["62e"].get("n") == ANSWERS_PER_DOSE and
                          details["62e"].get("OFFWORLD") is not None,
                          details["62e"].get("OFFWORLD", math.inf) <= PREDICTION_62E_MAX_OFFWORLD),
        "62f": status_for(all(row["n"] == ANSWERS_PER_DOSE and row["offworld"] is not None
                              for row in details["62f"]),
                          complete(details["62f"], lambda row: row["offworld"] is not None and
                                   row["offworld"] <= PREDICTION_62F_MAX_OFFWORLD)),
        "62g": ("HELD" if qwen_interval is not None and midpoint_holds else
                "FAILED" if qwen_interval is not None and
                (qwen_interval[1] < PREDICTION_62G_MIN_ALIEN_UNITS or
                 qwen_interval[0] > PREDICTION_62G_MAX_ALIEN_UNITS) else
                "NOT-EVALUABLE" if qwen_interval is not None else
                status_for(qwen_midpoint is not None, midpoint_holds)),
    }
    details["62g"] = {"estimate": qwen_midpoint, "interval": qwen_interval}
    decisions = {
        "threshold_vs_gradual": ("retired" if statuses["62d"] == "FAILED"
                                 else "not retired under 62d"),
        "step_from_prompt": ("prompt-step rule met" if statuses["62b"] == "FAILED"
                             else "prompt-step rule not met"),
    }
    return {"status": statuses, "details": details, "decision_rules": decisions}


def _record_max_tokens(record: dict) -> int:
    """The smallest max_tokens the label record's requests were sent with.

    A record written before blind62 stored a top-level max_tokens (2026-10-23)
    carries it per paid chunk only; every paid chunk must state it, or the
    record is refused (returns -1)."""
    value = record.get("max_tokens")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    chunks = record.get("paid_chunks")
    if not isinstance(chunks, list) or not chunks:
        return -1
    values = [c.get("max_tokens") if isinstance(c, dict) else None for c in chunks]
    if any(not isinstance(v, int) or isinstance(v, bool) for v in values):
        return -1
    return min(values)


def _label_inputs(folder: Path, labels_path: Path, review_dir: Path) -> tuple[list[dict], dict, dict]:
    rows = blind62.answers(folder)
    random.Random(blind62.SHUFFLE_SEED).shuffle(rows)
    record = json.loads(labels_path.read_text(encoding="utf-8"))
    if record.get("schema") != "part62-blind-labels-v1" or not record.get("scoring_allowed"):
        raise ValueError("blind label record failed calibration or contains unlabelled answers")
    reader = record.get("reader")
    if reader not in blind62.READER_MODELS or record.get("model") != blind62.READER_MODELS[reader]:
        raise ValueError("blind label record does not name a registered reader and model")
    if _record_max_tokens(record) < blind62.MAX_TOKENS:
        raise ValueError("blind label record used max_tokens below 1,024")
    calibration = record.get("calibration", {})
    completion = record.get("completion", {})
    stored_labels = calibration.get("labels", {}) if isinstance(calibration, dict) else {}
    if (not isinstance(calibration, dict) or not isinstance(completion, dict)
            or not isinstance(stored_labels, dict)
            or calibration.get("passed") is not True
            or calibration.get("wrong") != len(blind62._calibration_wrong(stored_labels))
            or calibration.get("kappa") != blind62._calibration_kappa(stored_labels)
            or calibration.get("completion_issues")
            or completion.get("passed") is not True
            or completion.get("issues")
            or completion.get("accepted_stop_reasons") != list(blind62.ACCEPTED_STOP_REASONS)):
        raise ValueError("blind calibration or completion gate did not pass")
    batch_ids = record.get("batch_ids")
    if not isinstance(batch_ids, list) or not batch_ids or any(not isinstance(x, str) or not x for x in batch_ids):
        raise ValueError("blind label record is missing its batch ids")
    key = {}
    for batch_id in batch_ids:
        key_path = review_dir / f"part62-key-{batch_id}.json"
        if not key_path.is_file():
            raise ValueError(f"blind label record is missing key for batch {batch_id}")
        chunk_key = json.loads(key_path.read_text(encoding="utf-8"))
        if set(key).intersection(chunk_key):
            raise ValueError("per-batch keys contain duplicate answer IDs")
        key.update(chunk_key)
    expected_key = {f"ans|{number}": {name: value for name, value in row.items() if name != "text"}
                    for number, row in enumerate(rows)}
    if key != expected_key:
        raise ValueError("label key does not match the shuffled real writer output; refusing to analyze")
    labels = record.get("labels")
    if (record.get("answer_count") != len(expected_key) or record.get("unlabelled_answers")
            or not isinstance(labels, dict)
            or any(not isinstance(labels.get(answer_id), list) or len(labels[answer_id]) != 2
                   for answer_id in expected_key)):
        raise ValueError("unlabelled answer present; refusing to analyze")
    return rows, labels, record


def analyze(folder: str | Path, labels_path: str | Path | None = None,
            review_dir: str | Path | None = None) -> dict:
    folder = Path(folder).resolve()
    proxies = []
    for row in blind62.answers(folder):
        proxies.append({"model": row["model"], "phase": row["phase"], "kind": row["kind"],
                        "dose": row["dose"], "question": row["question"],
                        "text": row["text"], "proxy": proxy_flags(row["text"])})
    result = {"schema": "part62-analysis-v1", "keyword_proxy_rows": proxies,
              "keyword_proxy_counts": _proxy_only_counts(proxies)}
    if labels_path is None:
        result["status"] = "proxies-only"
        return result
    review_dir = Path(review_dir or Path(labels_path).resolve().parent)
    rows, labels, label_record = _label_inputs(folder, Path(labels_path).resolve(), review_dir)
    counts, sample_info = aggregate(rows, labels)
    with (folder / "part62-manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    fits = fit_by_model_phase(counts, manifest["models"])
    predictions = evaluate_predictions(counts, fits)
    result.update({"status": "complete", "batch_ids": label_record["batch_ids"],
                   "reader": label_record["reader"], "reader_model": label_record["model"],
                   "sample_info": sample_info,
                   "reader_counts": {"|".join((key[0], key[1], key[2], f"{key[3]:g}")): value
                                     for key, value in counts.items()},
                   "logistic_fits": {f"{model}|{phase}": fit for (model, phase), fit in fits.items()},
                   "predictions": predictions})
    return result


def _proxy_only_counts(rows: list[dict]) -> dict:
    counts = {}
    for item in rows:
        group = (item["model"], item["phase"], item["kind"], float(item["dose"]))
        row = counts.setdefault(group, {"n": 0, "offworld": 0, "scifi": 0, "earth": 0})
        row["n"] += 1
        row["offworld"] += item["proxy"]["offworld"]
        row["scifi"] += item["proxy"]["scifi"]
        row["earth"] += item["proxy"]["earth"]
    return {"|".join((group[0], group[1], group[2], f"{group[3]:g}")): row
            for group, row in sorted(counts.items())}


def print_analysis(result: dict) -> None:
    reader_tag = f"reader={result.get('reader', 'none')} "
    def emit(line: str) -> None:
        print(f"{reader_tag}{line}")
    emit("KNOWN-NOISY keyword proxies (exploratory): model | phase | kind | dose | n | offworld scifi earth")
    for key, row in result["keyword_proxy_counts"].items():
        emit(f"{key} | {row['n']} | {row['offworld']} {row['scifi']} {row['earth']}")
    if result.get("status") != "complete":
        emit("blind reader labels unavailable; 62a-62g not evaluated")
        return
    emit(f"model={result.get('reader_model', 'unknown')} prediction | result")
    for prediction, status in result["predictions"]["status"].items():
        emit(f"{prediction} | {status}")
    emit("registered decision rules")
    for name, result_text in result["predictions"].get("decision_rules", {}).items():
        emit(f"{name} | {result_text}")
    emit("logistic fits: model | phase | slope | midpoint Pile-max | width 10-90% | midpoint alien-frame")
    for key, fit in result["logistic_fits"].items():
        emit(f"{key} | {fit.get('slope')} | {fit.get('midpoint')} | "
             f"{fit.get('width_10_90')} | {fit.get('midpoint_alien_frame_units')}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", default=str(PROJECT_ROOT / "results" / "part62"))
    ap.add_argument("--labels", default=None, help="part62-labels-BATCH.json; omit for keyword proxies only")
    ap.add_argument("--review-dir", default=str(PROJECT_ROOT / "results" / "part62-review"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "part62-analysis.json"))
    args = ap.parse_args(argv)
    result = analyze(args.folder, args.labels, args.review_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_analysis(result)


if __name__ == "__main__":
    main()
