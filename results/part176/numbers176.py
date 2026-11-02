#!/usr/bin/env python3
"""Compute Part 176 write-up numbers and generate its README from frozen files."""
import json
import math
import numpy as np
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DOSES = ["0.2", "0.225", "0.25", "0.275", "0.3"]
STATES = ["qwen", "google", "other name", "claim with no single name", "reverted",
          "longer no claim", "fragment", "broken", "unlabelled"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def link(value, target="numbers176.json"):
    return f"[{value}]({target})"


def link_numbers(text, target="../../PART176-DESIGN.md"):
    pat = re.compile(r"(?<![A-Za-z0-9_])(?:\d+(?:\.\d+)?)(?![A-Za-z0-9_])")
    return pat.sub(lambda m: link(m.group(0), target), text)


def bound_text(bounds):
    lo, hi = bounds
    return str(lo) if lo == hi else f"{lo}–{hi}"


def fmt(value, digits=3):
    return "null" if value is None else f"{value:.{digits}f}"


def pct_range(bounds, digits=1):
    lo, hi = bounds[0] * 100, bounds[1] * 100
    if abs(lo - hi) < 1e-12:
        return f"{lo:.{digits}f}%"
    return f"{lo:.{digits}f}–{hi:.{digits}f}%"


def avg_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranked = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for pos in range(start, end):
            ranked[order[pos]] = rank
        start = end
    return ranked


def pearson(x, y):
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    sx = math.sqrt(sum(v * v for v in dx))
    sy = math.sqrt(sum(v * v for v in dy))
    return None if sx == 0 or sy == 0 else sum(a * b for a, b in zip(dx, dy)) / (sx * sy)


def spearman(x, y):
    return pearson(avg_ranks(x), avg_ranks(y)) if len(x) == len(y) and len(x) >= 2 else None


score = read_json(ROOT / "score.json")
score_directions = score["directions"]
pred_table = read_json(ROOT / "prediction_table.json")
transitions_file = read_json(ROOT / "transitions.json")
audit_source = read_json(ROOT / "labels" / "audit.json")

# Pooled counts, answer states, rates, direction nulls, and medians by rung.
rung_rows = []
for dose in DOSES:
    d = score["doses"][dose]
    directions = d["directions"]
    failed_bounds = [row["failed_bounds"] for row in directions]
    med_failed = [median([b[i] for b in failed_bounds]) for i in (0, 1)]
    signs = {}
    for sign in (1, -1):
        rows = [row for row in directions if int(row["sign"]) == sign]
        signs[str(sign)] = {
            "lower": sum(row["support_bounds"][0] for row in rows),
            "upper": sum(row["support_bounds"][1] for row in rows),
            "denominator": len(rows),
        }
    rung_rows.append({
        "dose": float(dose),
        "answers": d["answers"],
        "answer_state_counts": {state: d["state_counts"].get(state, 0) for state in STATES},
        "support": d["support"],
        "google_led": d["google_led"],
        "other_led": d["other_led"],
        "null_direction_counts": {
            "support_bounded": sum(row["support_bounds"][0] != row["support_bounds"][1] for row in directions),
            "google_led_bounded": sum(row["google_led_bounds"][0] != row["google_led_bounds"][1] for row in directions),
            "other_led_bounded": sum(row["other_led_bounds"][0] != row["other_led_bounds"][1] for row in directions),
            "google_count_null": sum(row["google_count"] is None for row in directions),
            "google_share_null": sum(row["google_over_named"] is None for row in directions),
            "named_null": sum(row["named"] is None for row in directions),
        },
        "pooled_counts_and_rates": {
            "named_bounds": d["named_bounds"],
            "google_count_bounds": d["google_count_bounds"],
            "google_over_44_bounds": d["rates"]["google_over_44_bounds"],
            "google_over_named_bounds": d["rates"]["google_over_named_bounds"],
            "named_over_44_bounds": d["rates"]["named_over_44_bounds"],
        },
        "median_broken_per_direction": median([row["broken"] for row in directions]),
        "median_failed_per_direction_bounds": med_failed,
        "median_token_count_all_answers": d["median_token_count_all_answers"],
        "question_jaccard_over_0_7_share_all_answers": d["question_jaccard_over_0_7_share_all_answers"],
        "intervals": d.get("intervals"),
        "support_by_sign": signs,
    })

# Full per-direction rows, including all five dose states and nulls.
direction_rows = []
for direction in score["directions"]:
    rungs = []
    for row in direction["rungs"]:
        rungs.append({
            "dose": row["dose"],
            "state_counts": {state: row["state_counts"].get(state, 0) for state in STATES},
            "broken": row["broken"],
            "failed": row["failed"],
            "failed_bounds": row["failed_bounds"],
            "lost_bounds": row["lost_bounds"],
            "named": row["named"],
            "named_bounds": row["named_bounds"],
            "google_count": row["google_count"],
            "google_count_bounds": row["google_count_bounds"],
            "google_over_44": row["google_over_44"],
            "google_over_named": row["google_over_named"],
            "google_over_named_bounds": row["google_over_named_bounds"],
            "named_over_44": row["named_over_44"],
            "support": row["support"],
            "support_status": row["support_status"],
            "support_bounds": row["support_bounds"],
            "google_led": row["google_led"],
            "google_led_bounds": row["google_led_bounds"],
            "other_led": row["other_led"],
            "other_led_bounds": row["other_led_bounds"],
            "reader_state": row["reader_state"],
            "reader_unlabelled_clean_answers": row["reader_unlabelled_clean_answers"],
        })
    available_google_counts = [row for row in rungs if row["google_count"] is not None]
    google_peak_count = (max(row["google_count"] for row in available_google_counts)
                         if available_google_counts else None)
    google_peak_doses = ([row["dose"] for row in available_google_counts
                          if row["google_count"] == google_peak_count]
                         if google_peak_count is not None else [])
    direction_rows.append({
        "seed": direction["seed"],
        "sign": direction["sign"],
        "direction_id": f"{direction['seed']} {'+' if direction['sign'] > 0 else '−'}",
        "collapse_dose": direction["collapse_dose"],
        "ordinal_L": direction["ordinal_L"],
        "crossing_bracket": direction["crossing_bracket"],
        "recrossing_count": direction["recrossing_count"],
        "transition_width_11_to_33": direction["transition_width_11_to_33"],
        "top_supported_rung": direction["top_supported_rung"],
        "google_peak": {"count": google_peak_count, "doses": google_peak_doses},
        "gap": direction["gap"],
        "gap_reason": direction["gap_reason"],
        "undetermined_crossing": direction["undetermined_crossing"],
        "undetermined_reason": direction["undetermined_reason"],
        "rungs": rungs,
    })

# L, crossing bracket, recrossing, transition widths, and W3 gaps.
L_status = {}
L_brackets = {}
L_values = []
widths = []
recrossings = 0
directions_with_recrossing = 0
for row in direction_rows:
    collapse = row["collapse_dose"]
    status = collapse["status"]
    L_status[status] = L_status.get(status, 0) + 1
    bracket = row["crossing_bracket"]
    if bracket and bracket[1] is None:
        label = ">0.30"
    elif bracket and bracket[0] is None:
        label = "left-censored at 0.20"
    elif bracket is None:
        label = "null"
    else:
        label = f"{bracket[0]:g}–{bracket[1]:g}"
    L_brackets[label] = L_brackets.get(label, 0) + 1
    if collapse.get("L") is not None:
        L_values.append(collapse["L"])
    recrossings += row["recrossing_count"] or 0
    directions_with_recrossing += int((row["recrossing_count"] or 0) > 0)
    width = row["transition_width_11_to_33"]
    if width and width.get("width") is not None:
        widths.append(width["width"])

gap_rows = [row for row in direction_rows if row["gap"] is not None]
gap_values = [row["gap"] for row in gap_rows]
same_bracket = sum(
    row["crossing_bracket"] is not None
    and row["top_supported_rung"] is not None
    and row["crossing_bracket"][0] == row["top_supported_rung"]
    for row in gap_rows
)
top_rungs = {}
for row in gap_rows:
    k = f"{float(row['top_supported_rung']):g}"
    top_rungs[k] = top_rungs.get(k, 0) + 1
sensitivity_medians = {}
for key in ["failed_15", "failed_29", "lost_22", "fragment_8_failed_22", "fragment_20_failed_22"]:
    values = [row.get("L_sensitivities", {}).get(key, {}).get("L") for row in score["directions"]]
    values = [value for value in values if value is not None]
    sensitivity_medians[key] = {"n": len(values), "median": median(values) if values else None}
L_summary = {
    "crossing_threshold_failed_answers": 22,
    "threshold_denominator": 44,
    "status_counts": L_status,
    "crossing_brackets": L_brackets,
    "median_L_with_right_censored_at_0_30": median(L_values),
    "L_values_with_right_censored_lower_bound": sorted(L_values),
    "left_censored_count": L_status.get("left censored", 0),
    "right_censored_count": L_status.get("right censored", 0),
    "recrossing_total": recrossings,
    "directions_with_any_recrossing": directions_with_recrossing,
    "transition_width_11_to_33": {
        "n": len(widths),
        "median": median(widths) if widths else None,
        "minimum": min(widths) if widths else None,
        "maximum": max(widths) if widths else None,
        "values": sorted(widths),
    },
    "sensitivity_medians": sensitivity_medians,
}
gap_summary = {
    "n": len(gap_values),
    "median": median(gap_values),
    "minimum": min(gap_values),
    "maximum": max(gap_values),
    "under_0_025": sum(value < 0.025 for value in gap_values),
    "share_under_0_025": sum(value < 0.025 for value in gap_values) / len(gap_values),
    "negative_gaps": sum(value < 0 for value in gap_values),
    "right_censored_L_directions": score["gap_distribution"]["right_censored_L_directions"],
    "top_supported_rung_counts": top_rungs,
    "same_bracket_lower_endpoint_count": same_bracket,
    "values": sorted(gap_values),
}

same_bracket_under = [
    row for row in gap_rows if row["gap"] < 0.025
    and row["crossing_bracket"] is not None
    and row["top_supported_rung"] == row["crossing_bracket"][0]
]
not_same_bracket_under = [
    row for row in gap_rows if row["gap"] < 0.025
    and not (row["crossing_bracket"] is not None
             and row["top_supported_rung"] == row["crossing_bracket"][0])
]
no_supported_direction_ids = [
    row["direction_id"] for row in direction_rows if row["top_supported_rung"] is None
]
support_at_or_above_L_direction_ids = [
    row["direction_id"] for row in gap_rows if row["gap"] <= 0
]
gap_spacing = min(float(DOSES[index + 1]) - float(DOSES[index])
                  for index in range(len(DOSES) - 1))
sharp_step_uniform_gap = {
    "assumption": "sharp step with L uniform in its bracket",
    "gap_range": [0.0, gap_spacing],
    "median": gap_spacing / 2.0,
    "all_gaps_under_one_rung": True,
}

W3_SENSITIVITY_KEYS = [
    "failed_15", "failed_29", "lost_22", "fragment_8_failed_22", "fragment_20_failed_22",
]
w3_sensitivity = []
for key in ["registered_failed_22_fragment_12", *W3_SENSITIVITY_KEYS]:
    values = []
    excluded = []
    for row in score_directions:
        top = row.get("top_supported_rung")
        if key == "registered_failed_22_fragment_12":
            L_value = row.get("collapse_dose", {}).get("L")
            reason = row.get("undetermined_reason")
        else:
            sensitivity = row.get("L_sensitivities", {}).get(key, {})
            L_value = sensitivity.get("L")
            reason = sensitivity.get("reason")
        if top is None or L_value is None:
            excluded.append({"direction_id": f"{row['seed']} {'+' if row['sign'] > 0 else '−'}",
                             "reason": reason or ("no supported rung" if top is None else "L not determined")})
            continue
        values.append(float(L_value) - float(top))
    med = median(values) if values else None
    verdict = ("not read" if med is None else
               "failed" if med < 0.025 else "held" if med >= 0.05 else "inconclusive")
    w3_sensitivity.append({
        "rule": key,
        "directions": len(values),
        "median_gap": med,
        "under_0_025": sum(value < 0.025 for value in values),
        "verdict": verdict,
        "excluded": excluded,
        "excluded_reason_counts": dict(Counter(item["reason"] for item in excluded)),
    })

lost_sensitivity = next(row for row in w3_sensitivity if row["rule"] == "lost_22")
lost_L_exclusions = [
    {"direction_id": f"{row['seed']} {'+' if row['sign'] > 0 else '−'}",
     "reason": row.get("L_sensitivities", {}).get("lost_22", {}).get("reason")}
    for row in score_directions
    if row.get("L_sensitivities", {}).get("lost_22", {}).get("L") is None
]
lost_all40_lower_values = []
lost_all40_upper_values = []
for direction in score_directions:
    sensitivity = direction.get("L_sensitivities", {}).get("lost_22", {})
    if sensitivity.get("L") is not None:
        low = high = float(sensitivity["L"])
    elif sensitivity.get("interval") is not None:
        low, high = map(float, sensitivity["interval"])
    else:
        continue
    lost_all40_lower_values.append(low)
    lost_all40_upper_values.append(high)
lost_22_all40_fill_medians = {
    "n": len(lost_all40_lower_values),
    "lower_endpoint_median": median(lost_all40_lower_values),
    "upper_endpoint_median": median(lost_all40_upper_values),
}
lost_22_gap_fill_values = {"lower": [], "upper": []}
for direction in score_directions:
    top = direction.get("top_supported_rung")
    if top is None:
        continue
    sensitivity = direction.get("L_sensitivities", {}).get("lost_22", {})
    if sensitivity.get("L") is not None:
        low = high = float(sensitivity["L"])
    elif sensitivity.get("interval") is not None:
        low, high = map(float, sensitivity["interval"])
    else:
        continue
    lost_22_gap_fill_values["lower"].append(low - float(top))
    lost_22_gap_fill_values["upper"].append(high - float(top))
lost_22_w3_gap_endpoint_fills = {
    "n": len(lost_22_gap_fill_values["upper"]),
    "lower_endpoint_median_gap": median(lost_22_gap_fill_values["lower"]),
    "upper_endpoint_median_gap": median(lost_22_gap_fill_values["upper"]),
}

# Odd/even prompt repeatability of L, using only determined crossings in both halves.
half_rows = score["split_half_spearmans"]["directions"]
half_pairs = []
half_status_counts = {"odd": {}, "even": {}}
for row in half_rows:
    values = []
    for half in ("odd", "even"):
        item = row[half]
        half_status_counts[half][item["status"]] = half_status_counts[half].get(item["status"], 0) + 1
        if item.get("determined") is True and item.get("L") is not None:
            values.append(item["L"])
        else:
            values.append(None)
    if values[0] is not None and values[1] is not None:
        half_pairs.append(values)
L_half_rho = spearman([row[0] for row in half_pairs], [row[1] for row in half_pairs])
split = score["split_half_spearmans"]["spearmans"]

# Audit counts, role swaps, and first-reader retry/cap.
audit_total_read = sum(audit_source["table"]["all"].values())
fragment_table = audit_source["table"]["by_sample_kind"]["fragment"]
fragment_read = sum(fragment_table.values())
role_swap = audit_source["table"]["all"].get("role swap", 0)
fragment_role_swap = fragment_table.get("role swap", 0)
fragment_refusal = fragment_table.get("coherent refusal", 0)
reader_logs = []
reader_log_lines = (ROOT / "readers.log").read_text(encoding="utf-8").splitlines()
for line in (ROOT / "readers.log").read_text(encoding="utf-8").splitlines():
    if line.startswith("{"):
        try:
            reader_logs.append(json.loads(line))
        except json.JSONDecodeError:
            pass
retry_summary = next((row for row in reader_logs if row.get("reader") == "ds" and row.get("targets") is not None), {})
error_cap = score["reader_gate9"]["error_cap"]
checkpoint_line_index = next((index for index, line in enumerate(reader_log_lines)
                              if '"error_fraction"' in line), None)
checkpoint_line = reader_log_lines[checkpoint_line_index] if checkpoint_line_index is not None else ""
checkpoint_fraction_match = re.search(r'"error_fraction"\s*:\s*([0-9.eE+-]+)', checkpoint_line)
checkpoint_cap_match = re.search(r'"error_cap_fraction"\s*:\s*([0-9.eE+-]+)', checkpoint_line)
checkpoint_time = None
if checkpoint_line_index is not None:
    for line in reversed(reader_log_lines[:checkpoint_line_index]):
        match = re.search(r"step_rc=\d+\s+(\d{2}:\d{2}:\d{2})", line)
        if match:
            checkpoint_time = match.group(1)
            break
checkpoint_error_fraction = float(checkpoint_fraction_match.group(1)) if checkpoint_fraction_match else None
checkpoint_error_cap = float(checkpoint_cap_match.group(1)) if checkpoint_cap_match else None
checkpoint_error_pairs = (round(checkpoint_error_fraction * error_cap["total_pairs"])
                          if checkpoint_error_fraction is not None else None)

first_reader_labels = read_jsonl(ROOT / "labels" / "labels-ds.jsonl")
truncated_first_reader_labels = [
    row for row in first_reader_labels
    if row.get("missing") is True and row.get("stop_reason") == "max_tokens"
]
retry_log_rows = [row for row in read_jsonl(ROOT / "labels" / "retry_net.jsonl")
                  if row.get("reader") == "ds" and row.get("item_id")]
retry_by_item = defaultdict(list)
for row in retry_log_rows:
    retry_by_item[str(row["item_id"])].append(row)
retry_final_rows = [rows[-1] for rows in retry_by_item.values()]
retry_final_network_successes = sum(
    row.get("request_sent") is True and row.get("error") is None for row in retry_final_rows
)

audit_reply_rows = read_jsonl(ROOT / "labels" / "audit.jsonl")
audit_failure_types = Counter()
for row in audit_reply_rows:
    error = str(row.get("error") or "")
    if not error:
        continue
    if error == "empty response":
        label = "empty response"
    elif "gaierror" in error:
        label = "DNS name-resolution failure"
    elif "no JSON object" in error:
        label = "unparseable reply"
    else:
        label = error
    audit_failure_types[label] += 1
planned_audit_counts = {"fragment": 100, "longer no claim": 50,
                        "claim with no single name": 25}
audit_sample_shortfalls = {
    key: planned_audit_counts[key] - audit_source["sample_counts"][key]
    for key in planned_audit_counts
}

reader_summary = {
    "dns_retry_targets": retry_summary.get("targets"),
    "dns_retry_resolved": retry_summary.get("resolved"),
    "remaining_dns_errors": retry_summary.get("remaining_dns_errors"),
    "checkpoint_time": checkpoint_time,
    "checkpoint_error_fraction": checkpoint_error_fraction,
    "checkpoint_error_cap_fraction": checkpoint_error_cap,
    "checkpoint_error_pairs_approx": checkpoint_error_pairs,
    "pre_send_dns_failures_at_checkpoint": retry_summary.get("targets"),
    "retry_final_target_count": len(retry_final_rows),
    "retry_final_network_successes": retry_final_network_successes,
    "truncated_max_tokens_labels": len(truncated_first_reader_labels),
    "elapsed_seconds_all_calls": retry_summary.get("elapsed_seconds_all_calls"),
    "budget_seconds": retry_summary.get("budget_seconds"),
    "budget_exhausted": retry_summary.get("budget_exhausted"),
    "final_failed_pairs": error_cap["failed_pairs"],
    "total_pairs": error_cap["total_pairs"],
    "error_fraction": error_cap["error_fraction"],
    "error_cap_fraction": error_cap["maximum_fraction"],
    "passed": error_cap["passed"],
}

# Part 173 random-valued cells, Parts 174-175 scored direction rows.
p173_score = read_json(REPO / "results" / "part173" / "score.json")
p173_family = read_json(REPO / "results" / "part173" / "family_table.json")["random_valued"]
p173_cells = [row for row in read_jsonl(REPO / "results" / "part173" / "cell_table.jsonl") if row.get("family") == "random_valued"]
p173_supported_cells = []
for row in p173_cells:
    cell = row["sensitivity"]["canonical"]
    if cell.get("modal") is not None and cell.get("modal_count", 0) >= 10 and cell.get("broken", 44) <= 8:
        p173_supported_cells.append({
            "set": row["set"], "dose": row["dose"], "modal": cell["modal"],
            "modal_count": cell["modal_count"], "broken": cell["broken"],
        })
p173_other = sum(row["modal"] != "google" for row in p173_supported_cells)
p173_google = len(p173_supported_cells) - p173_other
p174_score = read_json(REPO / "results" / "part174" / "score.json")
p174_directions = read_jsonl(REPO / "results" / "part174" / "scored_directions.jsonl")
p175_score = read_json(REPO / "results" / "part175" / "score.json")
p175_directions = read_jsonl(REPO / "results" / "part175" / "scored_directions.jsonl")


def summarize_part(part, rows):
    values = {"support": [0, 0], "google": [0, 0], "other": [0, 0]}
    for row in rows:
        support_bounds = row.get("support_bounds")
        if support_bounds is None:
            supported = bool(row.get("supported", False))
            support_bounds = [int(supported), int(supported)]
        modal = row.get("modal")
        if modal is None:
            modal = row.get("labelled_modal")
        if modal is None:
            modal = row.get("summary", {}).get("modal")
        google_bounds = row.get("google_led_bounds")
        if google_bounds is None:
            google_led = bool(row.get("google_led")) if "google_led" in row else bool(support_bounds[0] and modal == "google")
            google_bounds = [int(google_led), int(google_led)]
        other_bounds = row.get("other_led_bounds")
        if other_bounds is None:
            other_led = bool(row.get("other_led")) if "other_led" in row else bool(support_bounds[0] and modal != "google")
            other_bounds = [int(other_led), int(other_led)]
        for metric, bounds in [("support", support_bounds), ("google", google_bounds), ("other", other_bounds)]:
            values[metric][0] += bounds[0]
            values[metric][1] += bounds[1]
    return {
        "part": part,
        "tested_direction_dose_rows": len(rows),
        "supported_rows": {"lower": values["support"][0], "upper": values["support"][1]},
        "google_led_rows": {"lower": values["google"][0], "upper": values["google"][1]},
        "other_led_supported_rows": {"lower": values["other"][0], "upper": values["other"][1]},
        "source": f"results/part{part}/scored_directions.jsonl",
    }


part173 = {
    "part": 173,
    "family": "random_valued",
    "tested_direction_dose_rows": len(p173_cells),
    "supported_rows": {"lower": len(p173_supported_cells), "upper": len(p173_supported_cells)},
    "google_led_rows": {"lower": p173_google, "upper": p173_google},
    "other_led_supported_rows": {"lower": p173_other, "upper": p173_other},
    "unique_sets": p173_family["sets_total"],
    "sets_with_any_supported_cell": len(p173_family["supported_sets"]),
    "supported_cells": p173_supported_cells,
    "source": "results/part173/cell_table.jsonl and results/part173/family_table.json",
}
part174 = summarize_part(174, p174_directions)
part174_family_counts = Counter(row.get("family") for row in p174_directions)
part174_supported_by_family = Counter(
    row.get("family") for row in p174_directions
    if row.get("supported") is True or row.get("support_bounds", [0, 0])[0] == 1
)
part174["family_composition"] = {
    "tested_rows": dict(part174_family_counts),
    "supported_rows": dict(part174_supported_by_family),
}
part175 = summarize_part(175, p175_directions)
part176 = {
    "part": 176,
    "tested_direction_dose_rows": len(score_directions) * len(DOSES),
    "supported_rows": {
        "lower": sum(row["support"]["lower"] for row in rung_rows),
        "upper": sum(row["support"]["upper"] for row in rung_rows),
    },
    "google_led_rows": {
        "lower": sum(row["google_led"]["lower"] for row in rung_rows),
        "upper": sum(row["google_led"]["upper"] for row in rung_rows),
    },
    "other_led_supported_rows": {
        "lower": sum(row["other_led"]["lower"] for row in rung_rows),
        "upper": sum(row["other_led"]["upper"] for row in rung_rows),
    },
    "source": "results/part176/scored_directions.jsonl",
}
family_rows = [part173, part174, part175, part176]
family_total = {
    "tested_direction_dose_rows": sum(row["tested_direction_dose_rows"] for row in family_rows),
    "supported_rows": {
        "lower": sum(row["supported_rows"]["lower"] for row in family_rows),
        "upper": sum(row["supported_rows"]["upper"] for row in family_rows),
    },
    "google_led_rows": {
        "lower": sum(row["google_led_rows"]["lower"] for row in family_rows),
        "upper": sum(row["google_led_rows"]["upper"] for row in family_rows),
    },
    "other_led_supported_rows": {
        "lower": sum(row["other_led_supported_rows"]["lower"] for row in family_rows),
        "upper": sum(row["other_led_supported_rows"]["upper"] for row in family_rows),
    },
}

layer_norm_source = read_json(ROOT / "layer_norms.json")
layer_norm_rows = [row for row in layer_norm_source["rows"]
                   if row.get("family") == "f100d" and row.get("set", "").startswith("f100d_d176")]


def numeric_stats(values):
    values = [float(value) for value in values]
    return {"n": len(values), "median": median(values) if values else None,
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None}


layer_ids = sorted({int(layer) for row in layer_norm_rows for layer in row["layers"]})
dose_three_ways = {"source": "layer_norms.json", "per_layer": [], "sum_over_written_layers": []}
for layer_id in layer_ids:
    layer_values = {"layer": layer_id, "by_dose": {}}
    for dose in DOSES:
        values = [row["layers"][str(layer_id)] for row in layer_norm_rows
                  if float(row["dose"]) == float(dose)]
        layer_values["by_dose"][dose] = {
            "written_norm": numeric_stats([item["written_norm"] for item in values]),
            "written_over_residual_norm": numeric_stats(
                [item["written_over_residual_norm"] for item in values]
            ),
        }
    dose_three_ways["per_layer"].append(layer_values)
for dose in DOSES:
    values = [row for row in layer_norm_rows if float(row["dose"]) == float(dose)]
    dose_three_ways["sum_over_written_layers"].append({
        "dose": float(dose),
        "directions": len(values),
        "written_layers_per_direction": sorted({int(row["written_layers"]) for row in values}),
        "sum_written_layer_norms": numeric_stats([row["sum_written_layer_norms"] for row in values]),
    })

w2_estimate = pred_table["176.W2"]["estimate"]
w2_count_bounds = w2_estimate["google_count_bounds_by_rung"]
w2_share_bounds = w2_estimate["google_share_bounds_by_rung"]
w2_last_change = w2_estimate["share_changes"][-1]["change_bounds"]
w2_miss = "receding" if w2_share_bounds[-2][0] - w2_share_bounds[-1][1] > 0.05 else "noise"
count_peak_i = max(range(len(w2_count_bounds)), key=lambda i: sum(w2_count_bounds[i]) / 2.0)
paired = score["paired_share_changes"]["0.225_to_0.275"]
correlation = pred_table["176.7"]["estimate"]

# Paired Google shares use every direction whose observed named count is at least 11 at both rungs.
paired_observed = []
for direction in score_directions:
    left, right = direction["rungs"][1], direction["rungs"][3]
    if left["named_bounds"][0] < 11 or right["named_bounds"][0] < 11:
        continue
    left_share = left["google_count_bounds"][0] / left["named_bounds"][0]
    right_share = right["google_count_bounds"][0] / right["named_bounds"][0]
    paired_observed.append({
        "direction_id": f"{direction['seed']} {'+' if direction['sign'] > 0 else '−'}",
        "left_share_observed_labels": left_share,
        "right_share_observed_labels": right_share,
        "change": right_share - left_share,
        "has_unlabelled_clean_answer": any(
            direction["rungs"][index]["reader_unlabelled_clean_answers"] > 0 for index in (1, 3)
        ),
    })
complete_pair_ids = {
    f"{row['seed']} {'+' if row['sign'] > 0 else '−'}" for row in paired.get("pairs", [])
}
paired_observed_ids = {row["direction_id"] for row in paired_observed}
paired_observed_summary = {
    "eligible_directions": len(paired_observed),
    "rose": sum(row["change"] > 0 for row in paired_observed),
    "fell": sum(row["change"] < 0 for row in paired_observed),
    "equal": sum(row["change"] == 0 for row in paired_observed),
    "median_change": median([row["change"] for row in paired_observed]),
    "additional_unlabelled_directions_vs_complete_pairs": sorted(paired_observed_ids - complete_pair_ids),
    "directions_with_unlabelled_clean_answers": sorted(
        row["direction_id"] for row in paired_observed if row["has_unlabelled_clean_answer"]
    ),
    "pairs": paired_observed,
}

descriptive_count_spearmans = {}
for dose in (0.20, 0.25):
    observations = []
    for direction in score_directions:
        lvalue = direction.get("collapse_dose", {}).get("L")
        rung = next(row for row in direction["rungs"] if row["dose"] == dose)
        if lvalue is not None and rung["google_count"] is not None:
            observations.append((rung["google_count"], lvalue))
    descriptive_count_spearmans[f"{dose:g}"] = {
        "measure": "Google count over 44 prompts versus L, descriptive",
        "rho": spearman([row[0] for row in observations], [row[1] for row in observations]),
        "n": len(observations),
    }

other_name_observations = []
for direction in score_directions:
    lvalue = direction.get("collapse_dose", {}).get("L")
    rung = direction["rungs"][1]
    if lvalue is not None and rung["named_bounds"][0] >= 11:
        other_name_observations.append((rung["state_counts"].get("other name", 0), lvalue))
conditional_measures = {
    "google_share_vs_L": score["correlations"]["google_share_vs_L_conditional"],
    "observed_other_primary_count_vs_L": {
        "rho": spearman([row[0] for row in other_name_observations],
                         [row[1] for row in other_name_observations]),
        "n": len(other_name_observations),
        "label": "conditional on a named answer; observed other-primary count",
    },
    "google_share_bounds_0_225": score["doses"]["0.225"]["rates"]["google_over_named_bounds"],
    "google_count_0_225": score["doses"]["0.225"]["google_count"],
    "other_primary_count_0_225": score["doses"]["0.225"]["state_counts"].get("other name", 0),
    "named_count_bounds_0_225": score["doses"]["0.225"]["named_bounds"],
}
correlation_excluded_directions = []
for direction in score_directions:
    rung225 = direction["rungs"][1]
    if rung225["google_count"] is None or direction["collapse_dose"].get("L") is None:
        correlation_excluded_directions.append({
            "direction_id": f"{direction['seed']} {'+' if direction['sign'] > 0 else '−'}",
            "reason": ("Google count null at 0.225" if rung225["google_count"] is None
                      else direction.get("undetermined_reason") or "L not determined"),
        })

# Draw-cluster bootstrap at the registered rung. Each resample keeps both signs from a draw.
bootstrap_seed = 176
bootstrap_replicates = 10000
by_seed_0225 = defaultdict(list)
for direction in score_directions:
    rung = direction["rungs"][1]
    by_seed_0225[int(direction["seed"])].append({
        "support": int(rung["support_bounds"][0]),
        "google_modal_support": int(rung["google_led_bounds"][0]),
    })
bootstrap_seeds = sorted(by_seed_0225)
rng = np.random.default_rng(bootstrap_seed)
bootstrap_support_rates = []
bootstrap_google_modal_shares = []
for _ in range(bootstrap_replicates):
    sampled_seeds = rng.choice(bootstrap_seeds, size=len(bootstrap_seeds), replace=True)
    sampled = [row for seed_value in sampled_seeds for row in by_seed_0225[int(seed_value)]]
    support_n = sum(row["support"] for row in sampled)
    bootstrap_support_rates.append(support_n / len(sampled))
    if support_n:
        modal_n = sum(row["google_modal_support"] for row in sampled)
        bootstrap_google_modal_shares.append(modal_n / support_n)

bootstrap_summary_0225 = {
    "seed": bootstrap_seed,
    "replicates": bootstrap_replicates,
    "cluster": "draw seed; both signs kept together",
    "pooled_support": {
        "count": score["doses"]["0.225"]["support"]["lower"],
        "denominator": score["doses"]["0.225"]["support"]["denominator"],
        "rate": score["doses"]["0.225"]["support"]["lower"] /
                score["doses"]["0.225"]["support"]["denominator"],
        "interval_95": [float(np.percentile(bootstrap_support_rates, 2.5)),
                        float(np.percentile(bootstrap_support_rates, 97.5))],
    },
    "google_modal_share_among_supported": {
        "google_led": score["doses"]["0.225"]["google_led"]["lower"],
        "supported": score["doses"]["0.225"]["support"]["lower"],
        "share": (score["doses"]["0.225"]["google_led"]["lower"] /
                  score["doses"]["0.225"]["support"]["lower"]),
        "interval_95": [float(np.percentile(bootstrap_google_modal_shares, 2.5)),
                        float(np.percentile(bootstrap_google_modal_shares, 97.5))],
        "exact_either_sign_rate": score["doses"]["0.225"]["intervals"]["supported"]["either_sign_rate"],
    },
    "wilson_by_sign": score["doses"]["0.225"]["intervals"]["supported"]["at_each_sign_wilson"],
}

transition_rows = []
for key in ["0.2->0.225", "0.225->0.25", "0.25->0.275", "0.275->0.3"]:
    item = transitions_file[key]
    matrix = item["pooled"]["matrix"]
    transition_rows.append({
        "from_to": key,
        "observations": item["pooled"]["observations"],
        "google_at_start": sum(matrix["google"].values()),
        "google_destinations": {state: matrix["google"].get(state, 0) for state in STATES},
        "google_to_failed": item["pooled"]["google_to_failed"],
        "by_draw_and_sign": item["by_draw_and_sign"],
    })
trajectory = transitions_file["trajectory_summary"]["pooled"]
trajectory["ended_failed_after_google_share_of_ever_google"] = (
    trajectory["ended_failed_after_google"] / trajectory["ever_google"]
    if trajectory["ever_google"] else None
)
audit_summary = {
    "seed": audit_source["seed"],
    "planned_rows": audit_source["rows"],
    "read_rows": audit_total_read,
    "empty_replies": audit_source["rows"] - audit_total_read,
    "sample_counts": audit_source["sample_counts"],
    "planned_sample_counts": planned_audit_counts,
    "sample_shortfalls": audit_sample_shortfalls,
    "failure_counts": dict(audit_failure_types),
    "all_failed_replies_after_two_attempts": all(
        int(row.get("attempts", 0)) == 2 for row in audit_reply_rows if row.get("error")
    ),
    "table_all": audit_source["table"]["all"],
    "table_by_sample_kind": audit_source["table"]["by_sample_kind"],
    "role_swap_read_count": role_swap,
    "role_swap_rate_read": role_swap / audit_total_read,
    "fragment_role_swap_count": fragment_role_swap,
    "fragment_read_count": fragment_read,
    "fragment_role_swap_rate": fragment_role_swap / fragment_read,
    "fragment_coherent_refusal_count": fragment_refusal,
    "fragment_coherent_refusal_n": fragment_read,
    "fragment_coherent_refusal_share": fragment_refusal / fragment_read,
    "one_third_rule_fired": fragment_refusal / fragment_read > 1.0 / 3.0,
    "known_check": audit_source["known_check"],
}
registered_predictions = {
    key: {
        "prediction": row["prediction"], "confidence": row["confidence"],
        "estimate": row["estimate"], "verdict": row["verdict"],
        "registered_rule": row.get("registered_rule"),
    }
    for key, row in pred_table.items()
}
result = {
    "part": 176,
    "sources": [
        "score.json", "prediction_table.json", "transitions.json", "labels/audit.json",
        "labels/audit.jsonl", "scored_directions.jsonl", "readers.log", "labels/retry_net.jsonl",
        "../part173/score.json", "../part173/family_table.json", "../part173/cell_table.jsonl",
        "../part174/score.json", "../part174/scored_directions.jsonl",
        "../part175/score.json", "../part175/scored_directions.jsonl",
    ],
    "scope": score["scope"],
    "selected": score["selected"],
    "answer_state_order": STATES,
    "rung_table": rung_rows,
    "directions": direction_rows,
    "L_distribution": L_summary,
    "gap_distribution": gap_summary,
        "gap_under_0_025_same_bracket_directions": [row["direction_id"] for row in same_bracket_under],
        "gap_under_0_025_other_directions": [row["direction_id"] for row in not_same_bracket_under],
        "no_supported_rung_directions": no_supported_direction_ids,
        "support_at_or_above_L_directions": support_at_or_above_L_direction_ids,
        "sharp_step_uniform_gap": sharp_step_uniform_gap,
        "W3_sensitivity": w3_sensitivity,
        "lost_22_exclusions": lost_sensitivity["excluded"],
        "lost_22_L_exclusions": lost_L_exclusions,
        "lost_22_all_40_fill_medians": lost_22_all40_fill_medians,
        "lost_22_W3_gap_endpoint_fills": lost_22_w3_gap_endpoint_fills,
        "paired_share_observed_labels_0_225_to_0_275": paired_observed_summary,
        "descriptive_google_count_spearmans_vs_L": descriptive_count_spearmans,
        "conditional_measures_vs_L": conditional_measures,
        "correlation_176_7_excluded_directions": correlation_excluded_directions,
        "uncertainty_0_225": bootstrap_summary_0225,
    "correlation_176_7": correlation,
    "split_halves": {
        "google_count_odd_against_even": split["google_count_0225"]["odd_against_even"],
        "google_share_odd_against_even": split["google_share_0225"]["odd_against_even"],
        "google_count_odd_measure_against_odd_L": split["google_count_0225"]["odd_measure_against_odd_L"],
        "google_count_even_measure_against_even_L": split["google_count_0225"]["even_measure_against_even_L"],
        "L_odd_against_even": {
            "rho": L_half_rho, "n": len(half_pairs), "excluded_directions": len(half_rows) - len(half_pairs),
            "half_status_counts": half_status_counts,
        },
        "directions": half_rows,
        "crossings_at_11_of_22": score["split_half_spearmans"]["crossings_at_11_of_22"],
        "excluded_censored_or_null_halves": score["split_half_spearmans"]["excluded_censored_or_null_halves"],
    },
    "paired_share_change_0_225_to_0_275": {key: value for key, value in paired.items() if key != "pairs"},
    "transitions": transition_rows,
    "trajectory_summary": trajectory,
    "audit": audit_summary,
    "reader_failures": reader_summary,
    "dose_three_ways": dose_three_ways,
    "reader_gate9": score["reader_gate9"],
    "registered_predictions": registered_predictions,
    "weaver_w1": {
        "support_bounds_by_rung": {str(row["dose"]): row["support"] for row in rung_rows},
        "minimum_selected_excess_over_neighbours": min(
            score["doses"]["0.225"]["support"]["lower"] - score["doses"][dose]["support"]["upper"]
            for dose in ("0.2", "0.25")
        ),
    },
    "weaver_w2": {
        "miss_kind": w2_miss,
        "last_share_change_bounds": w2_last_change,
        "google_count_bounds_by_rung": w2_count_bounds,
        "google_share_bounds_by_rung": w2_share_bounds,
        "named_0_30_bounds": rung_rows[-1]["pooled_counts_and_rates"]["named_bounds"],
        "count_peak_dose": float(DOSES[count_peak_i]),
    },
    "weaver_w3": {
        "median_gap": median(gap_values),
        "n": len(gap_values),
        "under_0_025": sum(value < 0.025 for value in gap_values),
        "same_bracket_lower_endpoint_count": same_bracket,
    },
    "family_rows_parts_173_to_176": family_rows,
    "family_rows_total": family_total,
    "not_used_and_scope": {
        "inert_guard_by_rung": score["inert_guard_by_rung"],
        "inert_guard_0_225": pred_table["176.4"]["estimate"]["inert_guard_0_20"]["by_rung"]["0.225"],
        "dose_norm_log": "layer_norms.json",
        "not_used_as_registered": [
            "inert-dose guard; difference from none was recorded by rung",
            "off-target prompts",
            "answer-polarity split",
            "flat versus discriminative-layer profile at matched per-layer norm",
        ],
    },
}
(ROOT / "numbers176.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def estimate_cell(key, row):
    e = row["estimate"]
    if key == "176.1":
        value = f"{e['supported_bounds'][0]} of {e['total']} supported"
    elif key == "176.2":
        value = f"{bound_text(e['google_led_supported_bounds'])} of {bound_text(e['supported_bounds'])}; {pct_range(e['share_bounds'])}"
    elif key == "176.3":
        bounds = e["other_led_directions"]
        value = f"{bounds['lower']}–{bounds['upper']} of {e['total']} other-led"
    elif key == "176.4":
        value = f"{bound_text(e['support_0_20_bounds'])} of 40 at 0.20; {bound_text(e['support_0_25_bounds'])} of 40 at 0.25"
    elif key == "176.5":
        value = f"median broken {e['median_broken']} of 44 across {e['directions']} directions"
    elif key == "176.6":
        value = (
            f"Google share {pct_range(e['google_share_bounds_0_225'])} at 0.225 and "
            f"{pct_range(e['google_share_bounds_0_275'])} at 0.275; named "
            f"{bound_text(e['named_bounds_0_225'])} to {bound_text(e['named_bounds_0_275'])}"
        )
    elif key == "176.7":
        rho = e["spearman"]["rho"]
        lo, hi = e["spearman"]["interval_95"]
        value = f"rho {rho:+.3f}, 95% interval {lo:+.3f} to {hi:+.3f}; n={e['spearman']['n']}"
    elif key == "176.W1":
        b = e["support_bounds"]
        value = f"0.20 {bound_text(b['0.2'])}; 0.225 {bound_text(b['0.225'])}; 0.25 {bound_text(b['0.25'])}; 0.30 {bound_text(b['0.3'])}"
    elif key == "176.W2":
        fall_magnitudes = sorted(abs(value) * 100 for value in w2_last_change)
        value = f"share fall at 0.30 {fall_magnitudes[0]:.1f}–{fall_magnitudes[1]:.1f} percentage points; Google count {bound_text(w2_count_bounds[-1])}"
    else:
        value = f"median gap {e['median_gap']:.5f}; {e['under_0_025']} of {e['directions']} under 0.025"
    return link(value, "prediction_table.json")


def write_readme():
    selected_dose = score["selected"]["dose"]
    p175_fixed_dose = p175_score["dose_reports"]["selected"]["dose"]
    selected = score["selected"]["support"]
    selected_rung = next(row for row in rung_rows if row["dose"] == selected_dose)
    link_number = lambda value: link(value, "numbers176.json")

    state_table = [
        "| Rung | " + " | ".join(STATES) + " |",
        "| --- | " + " | ".join(["---:"] * len(STATES)) + " |",
    ]
    for row in rung_rows:
        cells = [link_number(f"{row['dose']:g}")]
        cells += [link_number(row["answer_state_counts"][state]) for state in STATES]
        state_table.append("| " + " | ".join(map(str, cells)) + " |")

    support_table = [
        f"| Rung | Supported | Google-led | Other-led | Status null S/G/O | Google-count null | Named | Google / {link(44, '../../PART176-DESIGN.md')} | Google / named | Named / {link(44, '../../PART176-DESIGN.md')} | Median broken | Median failed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rung_rows:
        bounds = row["support"]
        google = row["google_led"]
        other = row["other_led"]
        nulls = row["null_direction_counts"]
        pooled = row["pooled_counts_and_rates"]
        failed = row["median_failed_per_direction_bounds"]
        values = [
            link_number(f"{row['dose']:g}"),
            link_number(f"{bound_text([bounds['lower'], bounds['upper']])} of 40"),
            link_number(f"{bound_text([google['lower'], google['upper']])} of 40"),
            link_number(f"{bound_text([other['lower'], other['upper']])} of 40"),
            link_number(f"{nulls['support_bounded']} / {nulls['google_led_bounded']} / {nulls['other_led_bounded']}"),
            link_number(nulls["google_count_null"]),
            link_number(bound_text(pooled["named_bounds"])),
            link_number(pct_range(pooled["google_over_44_bounds"])),
            link_number(pct_range(pooled["google_over_named_bounds"])),
            link_number(pct_range(pooled["named_over_44_bounds"])),
            link_number(fmt(row["median_broken_per_direction"], 1)),
            link_number(pct_range(failed, digits=1) if False else bound_text(failed)),
        ]
        support_table.append("| " + " | ".join(values) + " |")

    prediction_table = []
    for key, row in pred_table.items():
        wording = link_numbers(row["prediction"]) + f" ({row['confidence']})"
        prediction_table.append(
            f"| {key} | {wording} | {estimate_cell(key, row)} | {link(row['verdict'], 'prediction_table.json')} |"
        )

    # Short linked display values for prose and transition tables.
    support_sequence = ", ".join(link_number(bound_text([r["support"]["lower"], r["support"]["upper"]])) for r in rung_rows)
    corr = correlation["spearman"]
    corr_rho = link_number(format(corr["rho"], "+.2f"))
    corr_lo = link_number(format(corr["interval_95"][0], "+.2f"))
    corr_hi = link_number(format(corr["interval_95"][1], "+.2f"))
    half_google = link_number(format(split["google_count_0225"]["odd_against_even"]["rho"], ".2f"))
    half_L = link_number(format(L_half_rho, "+.2f"))
    pair = result["paired_share_change_0_225_to_0_275"]
    pair24 = paired_observed_summary
    pair24_unlabelled_text = ", ".join(
        link_number(value) for value in pair24["additional_unlabelled_directions_vs_complete_pairs"]
    )
    gap = gap_summary
    width = L_summary["transition_width_11_to_33"]
    audit = audit_summary
    family = family_rows
    family_sum = family_total
    reader = reader_summary
    relation = score["reader_gate9"]["second_reader"]["relation"]
    primary = score["reader_gate9"]["second_reader"]["replacement_primary"]

    family_table = [
        "| Part and stored family | Supported direction-at-dose rows | Other-led supported rows | Tested rows |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in family:
        sup = item["supported_rows"]
        other = item["other_led_supported_rows"]
        if item["part"] == 173:
            label = "[173](../part173/README.md) random-valued cells"
            tested = (
                f"{link_number(item['tested_direction_dose_rows'])} cells across "
                f"{link_number(item['unique_sets'])} sets; "
                f"{link_number(item['sets_with_any_supported_cell'])} sets had any support"
            )
        elif item["part"] == 174:
            label = "[174](../part174/README.md)"
            composition = item["family_composition"]
            tested = (
                f"{link_number(composition['tested_rows'].get('u100d', 0))} u100d, "
                f"{link_number(composition['tested_rows'].get('f100d', 0))} f100d; "
                f"{link_number(sum(composition['supported_rows'].values()))} supported, all u100d"
            )
        else:
            part_path = "README.md" if item["part"] == 176 else f"../part{item['part']}/README.md"
            label = f"[{item['part']}]({part_path})"
            tested = link_number(item["tested_direction_dose_rows"])
        family_table.append("| " + " | ".join([
            label,
            link_number(bound_text([sup["lower"], sup["upper"]])),
            link_number(f"{bound_text([other['lower'], other['upper']])} of supported"),
            tested,
        ]) + " |")

    l_bracket_table = [
        "| L crossing bracket | Directions |",
        "| --- | ---: |",
    ]
    bracket_items = list(L_summary["crossing_brackets"].items())
    bracket_items.sort(key=lambda item: (float(item[0].split("–")[0]) if "–" in item[0] else float(item[0][1:]), item[0]))
    for label, count in bracket_items:
        l_bracket_table.append(f"| {link_number(label)} | {link_number(count)} |")

    sensitivity_names = {
        "registered_failed_22_fragment_12": "Registered failed 22, fragment cut 12",
        "failed_15": "Failed threshold 15",
        "failed_29": "Failed threshold 29",
        "lost_22": "Lost threshold 22",
        "fragment_8_failed_22": "Fragment cut 8, failed threshold 22",
        "fragment_20_failed_22": "Fragment cut 20, failed threshold 22",
    }
    w3_sensitivity_map = {row["rule"]: row for row in w3_sensitivity}
    l_sensitivity_table = [
        "| L rule | L directions | Median L | Gap directions | Median gap |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in sensitivity_names.items():
        if key == "registered_failed_22_fragment_12":
            l_n = len(L_values)
            median_l = L_summary["median_L_with_right_censored_at_0_30"]
        else:
            item = L_summary["sensitivity_medians"][key]
            l_n, median_l = item["n"], item["median"]
        gap_item = w3_sensitivity_map[key]
        l_sensitivity_table.append("| " + " | ".join([
            link(label, "../../PART176-DESIGN.md"),
            link_number(l_n),
            link_number(fmt(median_l)),
            link_number(gap_item["directions"]),
            link_number(fmt(gap_item["median_gap"], 4)),
        ]) + " |")

    adjacent_support_rungs = 0
    for row in rung_rows:
        if row["support"]["upper"] == 0:
            break
        adjacent_support_rungs += 1
    inert_guard_read_rungs = [
        item for item in score["inert_guard_by_rung"].values()
        if item.get("status") == "read"
    ]

    audit_table = [
        "| Sample kind | Planned | Read | Echo | Role swap | Coherent refusal | Other maker | Stopped mid-sentence | Other |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    planned = planned_audit_counts
    audit_states = ["echo", "role swap", "coherent refusal", "other maker", "stopped mid-sentence", "other"]
    for kind, counts in audit["table_by_sample_kind"].items():
        read_count = sum(counts.values())
        cells = [kind, link_number(planned[kind]), link_number(read_count)]
        cells += [link_number(counts.get(state, 0)) for state in audit_states]
        audit_table.append("| " + " | ".join(cells) + " |")
    all_cells = [link_number(audit["planned_rows"]), link_number(audit["read_rows"])]
    all_cells += [link_number(audit["table_all"].get(state, 0)) for state in audit_states]
    audit_table.append("| All completed reads | " + " | ".join(all_cells) + " |")

    inert_guard_table = [
        "| Rung | Different from none | Share |",
        "| --- | ---: | ---: |",
    ]
    for dose in DOSES:
        item = score["inert_guard_by_rung"][dose]
        inert_guard_table.append("| " + " | ".join([
            link_number(dose),
            link_number(f"{item['different']} of {item['total']}"),
            link_number(fmt(item["share"], 2)),
        ]) + " |")

    direction_table = [
        "| Signed direction | Five rungs in order: dose, support, Google count | Google peak | Top supported rung | L | Gap |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for direction in direction_rows:
        rung_cells = []
        for rung in direction["rungs"]:
            status = rung["support_status"]
            mark = "S" if status == "supported" else "?" if status in ("unknown", "bounded") else "-"
            google_count = "null" if rung["google_count"] is None else rung["google_count"]
            rung_cells.append(
                f"{link_number(f'{rung['dose']:g}')}: {mark}/G{link_number(google_count)}"
            )
        peak = direction["google_peak"]
        if peak["count"] is None:
            peak_text = link_number("null")
        else:
            peak_doses = ", ".join(link_number(f"{dose:g}") for dose in peak["doses"])
            peak_text = f"G{link_number(peak['count'])} at {peak_doses}"
        top = (link_number(f"{direction['top_supported_rung']:g}")
               if direction["top_supported_rung"] is not None else link_number("none"))
        collapse = direction["collapse_dose"]
        L_text = link_number(fmt(collapse.get("L")))
        gap_text = link_number(fmt(direction["gap"], 4))
        direction_table.append("| " + " | ".join([
            link_number(direction["direction_id"]), "<br>".join(rung_cells), peak_text, top, L_text, gap_text,
        ]) + " |")

    checks_table = [
        f"| Rung | Median token count | Answer/question Jaccard over {link_number('0.7')} |",
        "| --- | ---: | ---: |",
    ]
    for row in rung_rows:
        checks_table.append("| " + " | ".join([
            link_number(f"{row['dose']:g}"),
            link_number(fmt(row["median_token_count_all_answers"], 1)),
            link_number(pct_range([row["question_jaccard_over_0_7_share_all_answers"]] * 2)),
        ]) + " |")

    dose_sum_table = [
        "| Rung | Directions | Written layers per direction | Sum of written layer norms, median (range) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in dose_three_ways["sum_over_written_layers"]:
        values = row["sum_written_layer_norms"]
        dose_sum_table.append("| " + " | ".join([
            link_number(f"{row['dose']:g}"),
            link_number(row["directions"]),
            link_number(", ".join(map(str, row["written_layers_per_direction"]))),
            link_number(f"{fmt(values['median'], 3)} ({fmt(values['minimum'], 3)} to {fmt(values['maximum'], 3)})"),
        ]) + " |")

    layer_norm_table = [
        "| Layer | " + " | ".join(link_number(f"{float(dose):g}") for dose in DOSES) + " |",
        "| --- | " + " | ".join(["---:"] * len(DOSES)) + " |",
    ]
    for layer in dose_three_ways["per_layer"]:
        cells = [link_number(layer["layer"])]
        for dose in DOSES:
            item = layer["by_dose"][dose]
            cells.append(link_number(
                f"{fmt(item['written_norm']['median'], 3)} / "
                f"{fmt(item['written_over_residual_norm']['median'], 3)}"
            ))
        layer_norm_table.append("| " + " | ".join(cells) + " |")

    inert_0225 = score["inert_guard_by_rung"]["0.225"]
    positive_wilson = bootstrap_summary_0225["wilson_by_sign"]["1"]["interval_95"]
    negative_wilson = bootstrap_summary_0225["wilson_by_sign"]["-1"]["interval_95"]
    either_sign = bootstrap_summary_0225["google_modal_share_among_supported"]["exact_either_sign_rate"]
    half_share = split["google_share_0225"]["odd_against_even"]
    share_correlation = conditional_measures["google_share_vs_L"]
    other_count_correlation = conditional_measures["observed_other_primary_count_vs_L"]
    excluded_176_7_text = ", ".join(link_number(row["direction_id"])
                                       for row in correlation_excluded_directions)
    w3_other_direction_text = ", ".join(link_number(row["direction_id"])
                                         for row in not_same_bracket_under)
    w3_on_line_rules = ", ".join(
        link(sensitivity_names[row["rule"]], "../../PART176-DESIGN.md")
        for row in w3_sensitivity
        if row["median_gap"] is not None and abs(row["median_gap"] - 0.025) < 1e-9
    )

    text = [
        "# [Part 176](../../PART176-DESIGN.md): On 20 new draws, 30 of 40 flat directions were supported at 0.225, all Google-led, and support peaks there; in 22 of the 34 directions with a supported rung, the top one lies within 0.025 of where the failure proxy crosses",
        "",
        f"On {link_number(score['counts']['directions'] // 2)} new draws, {link_number(selected['lower'])} of {link_number(selected['denominator'])} directions were supported at {link_number(f'{selected_dose:g}')}, all Google-led. Support peaks at this rung on the {link_number(len(rung_rows))}-rung ladder. Among {link_number(gap['n'])} directions with a supported rung, {link_number(gap['under_0_025'])} had their top supported rung within {link_number('0.025')} of L. {link_number(len(no_supported_direction_ids))} directions had no supported rung, so they have no top supported rung or gap. The dose was fixed by [Part 175](../part175/README.md); this tests new draws on the same {link(44, '../../PART176-DESIGN.md')} prompts, not new prompts.",
        "",
        "![Three code-drawn panels: answer states by rung, supported directions with null bounds, and Google counts by signed direction with L marked.](plot176.png)",
        "",
        "## Registered predictions",
        "",
        "Verdicts and estimates below are copied from prediction_table.json. The first cells are plain prediction ids.",
        "",
        "| # | Prediction (verbatim, with confidence) | Estimate | Verdict |",
        "| --- | --- | --- | --- |",
        *prediction_table,
        "",
        "## Does support extend beyond one rung?",
        "",
        f"Support by increasing rung is {support_sequence} of {link_number(40)}. The [0.20](numbers176.json) rung is {link_number('20 or 21')} because one direction has an undetermined short answer; support and Google-led counts stay bounded. At [0.225](numbers176.json), the count is {link_number(selected['lower'])} of {link_number(selected['denominator'])} and is labelled **selected** by the registered rule, which chooses the maximum definitely supported count and breaks ties toward the higher rung.",
        "",
        f"The answer-state table gives all {link_number(len(STATES))} states across the answers at every rung. The plot uses the same values. The full per-rung state table and per-direction rates and bounds are in [numbers176.json](numbers176.json).",
        "",
        *state_table,
        "",
        f"Google over all answers is Google count divided by {link(44, '../../PART176-DESIGN.md')} per direction, pooled over directions. Google share is Google divided by named answers. Named over {link(44, '../../PART176-DESIGN.md')} is shown beside both. Status-null S/G/O counts are bounded support, Google-led, or other-led statuses. Google-count nulls are listed separately: {link_number(selected_rung['null_direction_counts']['google_count_null'])} at 0.225.",
        "",
        *support_table,
        "",
        f"At [0.225](numbers176.json), the positive side had {link_number(selected_rung['support_by_sign']['1']['lower'])} of {link_number(20)} supported and the negative side had {link_number(selected_rung['support_by_sign']['-1']['lower'])} of {link_number(20)}. They are the two signs of random draws, with arbitrary sign, so their difference is not read.",
        "",
        f"The 95% Wilson interval is {link_number(pct_range(positive_wilson))} for the positive sign and {link_number(pct_range(negative_wilson))} for the negative sign. A bootstrap resampling the {link_number(len(bootstrap_seeds))} draws with both signs kept together, {link_number(bootstrap_replicates)} replicates and seed {link_number(bootstrap_seed)}, gives pooled support {link_number(pct_range([bootstrap_summary_0225['pooled_support']['rate']] * 2))} with interval {link_number(pct_range(bootstrap_summary_0225['pooled_support']['interval_95']))}. Google was modal on {link_number(bootstrap_summary_0225['google_modal_share_among_supported']['google_led'])} of {link_number(bootstrap_summary_0225['google_modal_share_among_supported']['supported'])} supported directions; its clustered bootstrap share is {link_number(pct_range([bootstrap_summary_0225['google_modal_share_among_supported']['share']] * 2))}, interval {link_number(pct_range(bootstrap_summary_0225['google_modal_share_among_supported']['interval_95']))}. Every draw had at least one Google-led supported sign: {link_number(either_sign['k'])} of {link_number(either_sign['n'])}, with exact one-sided 95% lower bound {link_number(pct_range([either_sign['lower_bound']] * 2))}.",
        "",
        f"At [0.225](numbers176.json), the corrected inert-dose guard reads {link_number(inert_0225['different'])} of {link_number(inert_0225['total'])} answers as different from none in the first {link(40, '../../PART176-DESIGN.md')} tokens, a {link_number(fmt(inert_0225['share'], 2))} share. It uses only the {link_number(score['counts']['directions'] // 2)} new draws. The earlier unread row counted the two replay sets as well.",
        "",
        *inert_guard_table,
        "",
        f"## Is [{selected_dose:g}](numbers176.json) a window, plateau, or slope?",
        "",
        f"{link_number('0.225')} is a peak on this {link_number(len(rung_rows))}-rung ladder: {link_number(score['doses']['0.225']['support']['lower'])} directions were supported there, against {link_number(bound_text([score['doses']['0.2']['support']['lower'], score['doses']['0.2']['support']['upper']]))} at {link_number('0.20')} and {link_number(score['doses']['0.25']['support']['lower'])} at {link_number('0.25')}. It exceeds {link_number('0.20')} by {link_number(bound_text([score['doses']['0.225']['support']['lower'] - score['doses']['0.2']['support']['upper'], score['doses']['0.225']['support']['lower'] - score['doses']['0.2']['support']['lower']]))} directions and {link_number('0.25')} by {link_number(score['doses']['0.225']['support']['lower'] - score['doses']['0.25']['support']['upper'])}. Weaver's [176.W1](prediction_table.json) peak falsifier fires because both differences meet the registered {link(5, '../../PART176-DESIGN.md')}-direction threshold. There were no supported directions at [0.30](numbers176.json).",
        "",
        "<details>",
        "<summary>All 40 signed directions, all five rungs, Google peaks and L</summary>",
        "",
        "Rung codes are S for supported, a hyphen for not supported, and ? for bounded or unknown. Each rung cell gives the dose, support code and observed Google count. A null count is shown as `null`.",
        "",
        *direction_table,
        "",
        "Warning: a Google peak cannot sit far above L, since failed answers name no one.",
        "",
        "</details>",
        "",
        "## Does more Google naming track earlier breakage?",
        "",
        f"Prediction [176.7](prediction_table.json) held as a non-detection. The registered Google count at 0.225 had Spearman {corr_rho} with a draw-cluster [95%](../../PART176-DESIGN.md) interval from {corr_lo} to {corr_hi} on {link_number(corr['n'])} eligible directions from {link_number(score['correlations']['eligible_draws'])} draws. The interval leaves a meaningful tie possible. Nothing is excluded by it. The three exclusions had a null Google count at 0.225: {excluded_176_7_text}.",
        "",
        f"As descriptive count measures against L, Google count had Spearman {link_number(format(descriptive_count_spearmans['0.2']['rho'], '+.2f'))} at 0.20 on {link_number(descriptive_count_spearmans['0.2']['n'])} directions and {link_number(format(descriptive_count_spearmans['0.25']['rho'], '+.2f'))} at 0.25 on {link_number(descriptive_count_spearmans['0.25']['n'])} directions. At 0.225 the pooled conditional Google share was {link_number(pct_range(conditional_measures['google_share_bounds_0_225']))} among {link_number(bound_text(conditional_measures['named_count_bounds_0_225']))} named answers, with {link_number(conditional_measures['other_primary_count_0_225'])} competing primary names. Conditional Google share versus L had rho {link_number(format(share_correlation['rho'], '+.2f'))} on {link_number(share_correlation['n'])}; observed competing-name count versus L had rho {link_number(format(other_count_correlation['rho'], '+.2f'))} on {link_number(other_count_correlation['n'])}. Both are conditional on named answers.",
        "",
        f"Across prompt halves, Google count repeated at {half_google} Spearman on {link_number(split['google_count_0225']['odd_against_even']['n'])} directions; conditional Google share repeated at {link_number(format(half_share['rho'], '.2f'))} on {link_number(half_share['n'])}. The computed L half repeatability was {half_L} on {link_number(len(half_pairs))} directions. Each half uses {link(22, '../../PART176-DESIGN.md')} prompts and the registered {link(11, '../../PART176-DESIGN.md')}-of-{link(22, '../../PART176-DESIGN.md')} crossing. Censored or null halves were left out and counted in [numbers176.json](numbers176.json). This split is only a repeatability check.",
        "",
        "## What does L measure?",
        "",
        f"L is the first interpolated rung where broken answers plus fragments reach {link(22, '../../PART176-DESIGN.md')} of {link(44, '../../PART176-DESIGN.md')} answers. It is an operational proxy. There were {link_number(L_status.get('crossed', 0))} crossings and {link_number(L_summary['right_censored_count'])} right-censored direction. Median L was {link_number(fmt(L_summary['median_L_with_right_censored_at_0_30']))} when the censored value is treated as a lower bound at [0.30](numbers176.json). {link_number(L_summary['directions_with_any_recrossing'])} directions had any later recrossing below {link(22, '../../PART176-DESIGN.md')}.",
        "",
        f"The {link(11, '../../PART176-DESIGN.md')}-to-{link(33, '../../PART176-DESIGN.md')} transition width has a median of {link_number(fmt(width['median']))} over {link_number(width['n'])} directions, with a range from {link_number(fmt(width['minimum']))} to {link_number(fmt(width['maximum']))}.",
        "",
        *l_bracket_table,
        "",
        *l_sensitivity_table,
        "",
        f"For lost threshold [22](../../PART176-DESIGN.md), the eligible W3 median is {link_number(fmt(lost_sensitivity['median_gap'], 4))} across {link_number(lost_sensitivity['directions'])} directions. {link_number(lost_22_w3_gap_endpoint_fills['n'] - lost_sensitivity['directions'])} more directions have a supported rung but an undetermined L. Filling those intervals at the lower and upper endpoints gives medians {link_number(fmt(lost_22_w3_gap_endpoint_fills['lower_endpoint_median_gap'], 4))} and {link_number(fmt(lost_22_w3_gap_endpoint_fills['upper_endpoint_median_gap'], 4))} across {link_number(lost_22_w3_gap_endpoint_fills['n'])} directions. These are sensitivity examples, not bounds.",
        "",
        f"Weaver's [176.W3](prediction_table.json) failed: median gap was {link_number(fmt(gap['median'], 4))} across {link_number(gap['n'])} directions with a supported rung, and {link_number(gap['under_0_025'])} had gap under [0.025](../../PART176-DESIGN.md). The rung spacing is {link_number(fmt(gap_spacing, 3))}. For {link_number(len(same_bracket_under))} of those {link_number(gap['under_0_025'])}, the top supported rung is the lower end of L's crossing bracket. The other directions are {w3_other_direction_text}. Under a sharp step with L uniform in its bracket, the gap is uniform on [{link_number(0)}, {link_number(fmt(sharp_step_uniform_gap['gap_range'][1], 3))}), with median {link_number(fmt(sharp_step_uniform_gap['median'], 4))}; every such gap is under one rung. Thus most of the {link_number(gap['under_0_025'])} is what the grid gives. The {link_number(gap['n'] - gap['under_0_025'])} directions with gap at least {link_number(fmt(gap_spacing, 3))} are not produced by that step alone. The directions with support at or above L are {', '.join(link_number(value) for value in support_at_or_above_L_direction_ids)}. Weaver's falsifier says a median under 0.025 would put naming at the edge of collapse and make Muse's warning, that a name in this band may be a side effect of breaking, live. The rung spacing limits that reading, and this part shows no cause.",
        "",
        f"The sensitivity table adds the W3 gap under each L rule. Under {w3_on_line_rules}, the median gap sits on {link_number('0.025')}. That is neither the registered claim at {link_number('0.05')} nor its falsifier below {link_number('0.025')}. W3 is sensitive to the failure threshold and fragment cut; the registered verdict remains failed.",
        "",
        f"Lost threshold 22 has {link_number(L_summary['sensitivity_medians']['lost_22']['n'])} determined L values because {link_number(len(lost_L_exclusions))} directions have short unlabelled answers that can change the first crossing. The excluded directions are {', '.join(link_number(row['direction_id']) for row in lost_L_exclusions)}. {link_number(lost_sensitivity['excluded_reason_counts'].get('no supported rung', 0))} other directions have no supported rung, so they enter the L median but not the W3 gap count. The determined median is {link_number(fmt(L_summary['sensitivity_medians']['lost_22']['median']))}; filling the crossing intervals for all {link_number(lost_22_all40_fill_medians['n'])} directions also rounds to {link_number(fmt(median([lost_22_all40_fill_medians['lower_endpoint_median'], lost_22_all40_fill_medians['upper_endpoint_median']])))}, as a sensitivity example rather than a bound.",
        "",
        "## Does the pooled share show what happened within directions?",
        "",
        f"Prediction [176.6](prediction_table.json) held as an aggregate statement only. Google share rose from {link_number(pct_range(rung_rows[1]['pooled_counts_and_rates']['google_over_named_bounds']))} at [0.225](numbers176.json) to {link_number(pct_range(rung_rows[3]['pooled_counts_and_rates']['google_over_named_bounds']))} at [0.275](numbers176.json), while named answers fell from {link_number(bound_text(rung_rows[1]['pooled_counts_and_rates']['named_bounds']))} to {link_number(bound_text(rung_rows[3]['pooled_counts_and_rates']['named_bounds']))}. Across all {link_number(pair24['eligible_directions'])} directions with at least {link(11, '../../PART176-DESIGN.md')} observed named answers at both rungs, shares rose in {link_number(pair24['rose'])} and fell in {link_number(pair24['fell'])}; the median observed-label change was {link_number(format(pair24['median_change'], '+.3f'))}. The {link_number(len(pair24['additional_unlabelled_directions_vs_complete_pairs']))} added directions had one unlabelled clean answer: {pair24_unlabelled_text}. On the {link_number(pair['eligible_directions'])} fully labelled directions, {link_number(pair['rose'])} rose and {link_number(pair['fell'])} fell, with median change {link_number(format(pair['median_change'], '+.3f'))} and [95%](../../PART176-DESIGN.md) interval {link_number(format(pair['interval_95'][0], '+.3f'))} to {link_number(format(pair['interval_95'][1], '+.3f'))}. Paired shares include only directions whose named pool met the floor at both rungs.",
        "",
        "## Why did Weaver's share prediction fail?",
        "",
        f"[176.W2](prediction_table.json) failed strictly on the fall at [0.30](numbers176.json): the pooled share fell {link_number(f'{min(abs(value) for value in w2_last_change) * 100:.1f}–{max(abs(value) for value in w2_last_change) * 100:.1f} percentage points')} from the prior rung, a {w2_miss} miss {('over' if w2_miss == 'receding' else 'within')} the registered [5](../../PART176-DESIGN.md)-point line. The conditional share at [0.30](numbers176.json) rests on {link_number(bound_text(rung_rows[-1]['pooled_counts_and_rates']['named_bounds']))} named answers. The count clause is separate: Google counts were {', '.join(link_number(bound_text(bounds)) for bounds in w2_count_bounds)} across the ladder. The count peak was at {link_number(f'{DOSES[count_peak_i]}')}; it did not rise at [0.275](numbers176.json) or [0.30](numbers176.json), so that half did not fail.",
        "",
        "## Where did Google answers go at the next rung?",
        "",
        "These prompt-by-direction transitions are descriptive. Each line starts with Google answers at one rung and lists their next state.",
        "",
    ]
    for item in transition_rows:
        destinations = item["google_destinations"]
        destination_text = ", ".join(f"{state} {link_number(destinations[state])}" for state in STATES)
        text.append(f"From {link_number(item['from_to'])}, {link_number(item['google_at_start'])} Google answers went to the next rung: {destination_text}.")
    trajectory_percent_of_google = trajectory["ended_failed_after_google_share_of_ever_google"] * 100
    trajectory_percent_of_failed = trajectory["failed_end_after_google_share"] * 100
    text.extend([
        "",
        f"Across complete trajectories, {link_number(trajectory['ever_google'])} of {link_number(trajectory['prompt_trajectories'])} ever reached Google. Of those, {link_number(trajectory['ended_failed_after_google'])} ended failed at {link_number(DOSES[-1])} ({link_number(f'{trajectory_percent_of_google:.1f}%')}). Separately, {link_number(trajectory['ended_failed_after_google'])} of the {link_number(trajectory['ended_failed'])} trajectories that ended failed ({link_number(f'{trajectory_percent_of_failed:.1f}%')}) had passed through Google.",
        "",
        "## What did the audit say about fragments and role swaps?",
        "",
        f"The second reader returned {link_number(audit['read_rows'])} of {link_number(audit['planned_rows'])} audit replies. The {link_number(audit['empty_replies'])} missing replies were {link_number(audit['failure_counts'].get('empty response', 0))} empty responses, {link_number(audit['failure_counts'].get('DNS name-resolution failure', 0))} DNS name-resolution failure, and {link_number(audit['failure_counts'].get('unparseable reply', 0))} unparseable reply; each failure followed {link_number(2)} attempts. The sample read {link_number(audit['sample_counts']['fragment'])} fragments, {link_number(audit['sample_counts']['longer no claim'])} longer no-claim answers and {link_number(audit['sample_counts']['claim with no single name'])} claims without one name. Against the planned {link_number(audit['planned_sample_counts']['fragment'])}, {link_number(audit['planned_sample_counts']['longer no claim'])}, and {link_number(audit['planned_sample_counts']['claim with no single name'])}, the shortfalls were {link_number(audit['sample_shortfalls']['fragment'])}, {link_number(audit['sample_shortfalls']['longer no claim'])}, and {link_number(audit['sample_shortfalls']['claim with no single name'])}. These audit replies are outside the first-reader error cap. Of {link_number(audit['fragment_read_count'])} read fragments, {link_number(audit['fragment_coherent_refusal_count'])} ({link_number(f'{audit['fragment_coherent_refusal_share'] * 100:.1f}%')}) were called coherent refusals, below the registered [one-third](../../PART176-DESIGN.md) rule, so it did not fire. Role swap was labelled in {link_number(audit['role_swap_read_count'])} of {link_number(audit['read_rows'])} completed replies ({link_number(f'{audit['role_swap_rate_read'] * 100:.1f}%')}); among fragments it was {link_number(audit['fragment_role_swap_count'])} of {link_number(audit['fragment_read_count'])}.",
        "",
        *audit_table,
        "",
        f"A fragment is a clean no-claim answer of at most {link_number(12)} tokens with no Qwen-family name in its words. The rule was written after looking at [Part 175](../part175/README.md)'s rows. Failed therefore remains a proxy: broken answers plus short no-claim answers. The audit does not prove the remaining short answers are failures.",
        "",
        "## What changed after scores were read?",
        "",
        f"[Amendment 1](../../PART176-DESIGN.md) was added after scores were read. It changed only [176.W1](prediction_table.json), from not read to failed: the scorer tries every admissible support count at [0.20](numbers176.json), and both {link_number(20)} and {link_number(21)} lead to the same registered falsifier because support at [0.225](numbers176.json) is {link_number(30)}. [Amendment 2](../../PART176-DESIGN.md) also exists. It fixes the inert guard to use only new draws, excluding replay sets. The rerun gives {link_number(inert_0225['different'])} of {link_number(inert_0225['total'])} different at 0.225, and the verdicts in [prediction_table.json](prediction_table.json) were unchanged. The audit plan, shortfalls and reply failures are listed above.",
        "",
        "## Do the earlier stored rows show other-led support?",
        "",
        f"The stored random-valued cells in [Part 173](../part173/README.md) and scored direction-at-dose rows in [Parts 174](../part174/README.md) to [176](README.md) give {link_number(bound_text([family_total['supported_rows']['lower'], family_total['supported_rows']['upper']]))} supported rows out of {link_number(family_total['tested_direction_dose_rows'])}; {link_number(bound_text([family_total['other_led_supported_rows']['lower'], family_total['other_led_supported_rows']['upper']]))} were other-led. [Part 173](../part173/README.md) contributes {link_number(part173['tested_direction_dose_rows'])} random-valued cells across {link_number(part173['unique_sets'])} sets, with {link_number(part173['supported_rows']['lower'])} cells supported and {link_number(part173['sets_with_any_supported_cell'])} of {link_number(part173['unique_sets'])} sets having any supported cell. The other part denominators are all stored scored-direction rows at their registered doses. These are stored runs, not one common sampling frame. This is not a comparison with picked neurons.",
        "",
        *family_table,
        "",
        "## Two all-answer checks and the written dose",
        "",
        "These checks use all answers, including broken answers. They do not filter the support or correlation results.",
        "",
        *checks_table,
        "",
        "The stored dose record measures the written norm at each layer, the sum across written layers, and the written norm divided by the layer's mean residual-stream norm on Gate 3's eight plain contexts. The values below are summaries over the 40 signed directions at each rung. The full source is [layer_norms.json](layer_norms.json).",
        "",
        *dose_sum_table,
        "",
        "<details>",
        "<summary>Per-layer written norm and residual-stream ratio at every rung</summary>",
        "",
        "Each cell is median written norm / median written-to-residual norm over the signed directions for that layer and rung.",
        "",
        *layer_norm_table,
        "",
        "</details>",
        "",
        "## Reader failures, scope, and limits",
        "",
        f"At the {link_number(reader['checkpoint_time'])} checkpoint, the first-reader error fraction was {link_number(f'{reader['checkpoint_error_fraction'] * 100:.3f}%')} ({link_number(reader['checkpoint_error_pairs_approx'])} of {link_number(reader['total_pairs'])}) against the {link_number(f'{reader['checkpoint_error_cap_fraction'] * 100:.1f}%')} cap. Those were pre-send DNS failures. The design treats name-resolution failures as no request sent, so they are exempt from the cap and retried. The retry log has {link_number(reader['retry_final_network_successes'])} successful final network attempts among {link_number(reader['retry_final_target_count'])} distinct targets. The log's `resolved` value is {link_number(reader['dns_retry_resolved'])}; [retry_net.py](../../part176/retry_net.py) counts labels accepted by `label173.valid`, not successful requests. The remaining {link_number(reader['truncated_max_tokens_labels'])} labels were replies truncated at `max_tokens`. After retries, {link_number(reader['final_failed_pairs'])} of {link_number(reader['total_pairs'])} first-reader pairs remained invalid ({link_number(f'{reader['error_fraction'] * 100:.3f}%')}) against the {link_number(f'{reader['error_cap_fraction'] * 100:.1f}%')} cap. The reader gate passed.",
        "",
        f"In the audit sample, relation agreement was kappa {link_number(f'{relation['kappa']:.3f}')} across {link_number(relation['pairs'])} pairs; replacement-primary agreement was {link_number(primary['matching_pairs'])} of {link_number(primary['comparable_pairs'])} pairs. As the design says, replacement-name agreement says nothing about these labels.",
        "",
        f"This result covers [Qwen3.5-4B](../../PART176-DESIGN.md), one maker family, greedy decoding, {link_number(score['counts']['directions'] // 2)} new draws, one set of {link(44, '../../PART176-DESIGN.md')} [NEW44B](../../PART176-DESIGN.md) maker prompts, and {link_number(len(rung_rows))} fixed dose rungs. The selected dose came from [Part 175](../part175/README.md). The same prompts were reused, so this tests draws, not prompts. {link_number(len(rung_rows))} rungs cannot locate a window narrower than {link_number('0.025')}. L depends on reader labels, the fragment cut, a {link(22, '../../PART176-DESIGN.md')}-of-{link(44, '../../PART176-DESIGN.md')} crossing, and straight-line interpolation. Interpolation cannot establish the curve between rungs or order crossings inside one bracket. This part does not measure delivery past the written layer. 176.7 is a correlation across directions without an intervention. Paired shares exist only where enough answers remain named.",
        "",
        f"The inert-dose guard is reported but does not set a verdict. Its [0.225](numbers176.json) row compares only the {link_number(score['counts']['directions'] // 2)} new draws. Other not-used items are off-target prompts, answer-polarity split, and matched flat-versus-discriminative-layer comparison. No comparison with [u100d](../part174/README.md) or picked-neuron directions is read here.",
        "",
        "The prior-art paragraph in the design cites Muse's council review and search. Claude had not read those papers at registration. This write-up leaves them unlinked.",
        "",
        "## Files",
        "",
        "- [score.json](score.json): gates, pooled counts, direction summaries, correlations, and intervals.",
        "- [prediction_table.json](prediction_table.json): registered estimates and verdicts.",
        f"- [scored_directions.jsonl](scored_directions.jsonl): all signed directions and {link_number(len(rung_rows))} rung rows.",
        "- [transitions.json](transitions.json): pooled and by-draw transition matrices.",
        "- [labels/audit.json](labels/audit.json) and [labels/audit.jsonl](labels/audit.jsonl): audit counts and replies.",
        "- [readers.log](readers.log) and [labels/retry_net.jsonl](labels/retry_net.jsonl): reader attempts and DNS retry.",
        "- [numbers176.py](numbers176.py) writes [numbers176.json](numbers176.json) and this README from frozen files.",
        "- [plot176.py](plot176.py) writes [plot176.png](plot176.png) from numbers176.json.",
        "",
        "Rebuild from the repository root:",
        "",
        "    cd results/part176 && python -I -B numbers176.py && python -I -B plot176.py",
        "",
        "Cold check: [COLD.md](COLD.md), 18 findings folded.",
    ])
    (ROOT / "README.md").write_text("\n".join(text) + "\n", encoding="utf-8")


write_readme()
print(f"Wrote {ROOT / 'numbers176.json'} and {ROOT / 'README.md'}")

