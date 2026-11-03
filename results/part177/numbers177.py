#!/usr/bin/env python3
"""Build Part 177's linked write-up numbers and README from the frozen result files."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
NUMBERS_FILE = "numbers177.json"
DESIGN_LINK = "../../PART177-DESIGN.md"
PREDICTION_LINK = "prediction_table.json"
LABEL_ORDER = ["asserted", "intermediate", "quoted from the problem",
               "rejected or doubted", "ambiguous"]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def pct(numerator: int, denominator: int) -> float | None:
    return None if not denominator else 100.0 * numerator / denominator


def pct_text(numerator: int, denominator: int, digits: int = 1) -> str:
    if not denominator:
        return "n/a"
    return f"{pct(numerator, denominator):.{digits}f}%"


def nlink(value, target: str = NUMBERS_FILE) -> str:
    return f"[{value}]({target})"


def dlink(value) -> str:
    return nlink(value, DESIGN_LINK)


def count_text(numerator: int, denominator: int, digits: int = 1) -> str:
    return f"{nlink(numerator)}/{nlink(denominator)} ({nlink(pct_text(numerator, denominator, digits))})"


def float_link(value: float | None, digits: int = 3) -> str:
    return "null" if value is None else nlink(f"{value:.{digits}f}")


def rate_interval_text(bounds, digits: int = 1, scale: float = 100.0) -> str:
    if not bounds:
        return "not available"
    if isinstance(bounds, dict):
        bounds = [bounds["lower"], bounds["upper"]]
    low, high = bounds
    lo = f"{float(low) * scale:.{digits}f}"
    hi = f"{float(high) * scale:.{digits}f}"
    return f"{nlink(lo)} to {nlink(hi)}{'%' if scale == 100.0 else ''}"


def pp_text(value: float, digits: int = 1) -> str:
    return f"{nlink(f'{value * 100.0:.{digits}f}')} percentage points"


def pp_interval_text(bounds, digits: int = 1) -> str:
    low, high = bounds
    return f"{nlink(f'{float(low) * 100.0:.{digits}f}')} to {nlink(f'{float(high) * 100.0:.{digits}f}')} percentage points"


def link_registered_numbers(value: str) -> str:
    pattern = re.compile(r"(?<![A-Za-z0-9_])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![A-Za-z0-9_])")
    return pattern.sub(lambda match: nlink(match.group(0), DESIGN_LINK), value)


def _manifest_hash(manifest, relative: str, actual: str):
    record = manifest.get("files", {}).get(relative)
    if not isinstance(record, dict):
        raise ValueError(f"missing frozen-file manifest entry: {relative}")
    if actual != record.get("sha256"):
        raise ValueError(f"hash mismatch for {relative}: {actual} != {record.get('sha256')}")


def load_unedited_samples(manifest):
    texts = {}
    for shard in range(10):
        relative = f"inputs/eval/unedited/parts/math500_s{shard}of10/math500_first_samples.jsonl.gz"
        path = ROOT / relative
        _manifest_hash(manifest, relative, hashlib.sha256(path.read_bytes()).hexdigest())
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                texts[str(row["id"])] = str(row["text"])
    if len(texts) != 200:
        raise ValueError(f"expected 200 unedited sample-0 texts, found {len(texts)}")
    tokenizer_relative = "inputs/tokenizer.json"
    tokenizer_path = ROOT / tokenizer_relative
    _manifest_hash(manifest, tokenizer_relative, hashlib.sha256(tokenizer_path.read_bytes()).hexdigest())
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("the pinned tokenizer summary needs the tokenizers package") from exc
    return texts, Tokenizer.from_file(str(tokenizer_path))


def scan_boxes(text: str):
    """Find brace-matched boxed values, retaining empty and incomplete boxes."""
    boxes = []
    for match in re.finditer(r"\\boxed|\\fbox", text):
        name = match.group(0)
        rest = text[match.end():]
        stripped = rest.lstrip()
        if not stripped.startswith("{"):
            if name == r"\boxed" and rest.startswith(" "):
                value = rest.strip().split("$")[0].split("\n")[0].strip()
                boxes.append({"start": match.start(), "kind": "unbraced" if value else "empty",
                              "value": value or None})
            else:
                boxes.append({"start": match.start(), "kind": "unparseable", "value": None})
            continue
        brace_start = match.end() + len(rest) - len(stripped)
        depth = 0
        for index in range(brace_start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    value = text[brace_start + 1:index].strip()
                    boxes.append({"start": match.start(), "kind": "closed" if value else "empty",
                                  "value": value or None})
                    break
        else:
            boxes.append({"start": match.start(), "kind": "unclosed", "value": None})
    return boxes


def quantile(values, probability: float):
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return float(ordered[0])
    # Hyndman-Fan Type 6, also known as the exclusive sample quantile.
    position = (len(ordered) + 1) * probability
    rank = math.floor(position)
    fraction = position - rank
    lower = max(0, min(rank - 1, len(ordered) - 1))
    upper = max(0, min(rank, len(ordered) - 1))
    return float(ordered[lower]) * (1.0 - fraction) + float(ordered[upper]) * fraction


def token_positions(tokenizer, texts, traces, positions):
    cache = {}
    result = []
    for row in traces:
        problem_id = row["problem_id"]
        position = positions.get(problem_id)
        if position is None:
            continue
        key = (problem_id, int(position))
        if key not in cache:
            encoded = tokenizer.encode(texts[problem_id][:int(position)])
            cache[key] = len(encoded.ids)
        result.append({"problem_id": problem_id, "char_position": int(position), "tokens_before": cache[key]})
    return result


def distribution(rows):
    values = [row["tokens_before"] for row in rows]
    q1 = quantile(values, 0.25)
    med = quantile(values, 0.50)
    q3 = quantile(values, 0.75)
    return {"n": len(values), "median": med, "q1": q1, "q3": q3,
            "iqr_width": None if q1 is None or q3 is None else q3 - q1,
            "method": "Hyndman-Fan Type 6 (exclusive), linear interpolation at p*(n+1); same pinned tokenizer as cap check"}


def clopper_pearson_zero_upper(n: int, confidence: float = 0.95) -> float | None:
    """Upper endpoint of the central exact Clopper-Pearson interval for 0 successes."""
    if n <= 0:
        return None
    alpha = 1.0 - confidence
    return 1.0 - (alpha / 2.0) ** (1.0 / n)


def build_numbers():
    score = read_json(ROOT / "score.json")
    read = read_json(ROOT / "read.json")
    prediction_table = read_json(ROOT / "prediction_table.json")
    manifest = read_json(ROOT / "inputs-manifest.json")
    keys_rows = read_jsonl(ROOT / "window-key.jsonl")
    labels_rows = read_jsonl(ROOT / "labels-ds.jsonl")
    labels = {row["id"]: row["label"] for row in labels_rows}
    keys = {row["id"]: row for row in keys_rows}
    if set(keys) != set(labels):
        raise ValueError("window-key and reader labels do not have the same IDs")

    traces = read["trace_results"]
    trace_by_key = {(row["problem_id"], row["arm"]): row for row in traces}
    unedited = [row for row in traces if row["arm"] == "unedited"]
    p50c = [row for row in traces if row["arm"] == "p50c"]
    unedited_by_id = {row["problem_id"]: row for row in unedited}
    cohort = [row for row in unedited if row["cohort_C"]]
    target = [row for row in cohort if row["target_T"]]
    prompt_echo = [row for row in cohort if not row["target_T"]]

    condition_tables = {}
    for condition in ("real", "shuffled"):
        rows = [row for row in keys_rows if row["target_T"] and row["condition"] == condition]
        count_by_label = Counter(labels[row["id"]] for row in rows)
        condition_tables[condition] = {
            "window_count": len(rows),
            "trace_count_with_hit": len({row["problem_id"] for row in rows}),
            "label_counts": {label: count_by_label.get(label, 0) for label in LABEL_ORDER},
            "unknown": sum(labels[row["id"]] not in LABEL_ORDER for row in rows),
        }

    keys_by_trace_condition = defaultdict(list)
    for row in keys_rows:
        keys_by_trace_condition[(row["problem_id"], row["condition"])].append(row)
    cohort_ids = {row["problem_id"] for row in cohort}
    shuffled_C_rows = [row for row in keys_rows
                       if row["problem_id"] in cohort_ids and row["condition"] == "shuffled"]
    shuffled_C_values = Counter(str(row["value"]) for row in shuffled_C_rows)
    shuffled_C_labels = Counter(labels[row["id"]] for row in shuffled_C_rows)
    shuffled_C_value_order = sorted(
        shuffled_C_values.items(),
        key=lambda item: (0, int(item[0])) if re.fullmatch(r"[+-]?\d+", item[0]) else (1, item[0]))
    shuffled_C = {
        "window_count": len(shuffled_C_rows),
        "trace_count_with_hit": len({row["problem_id"] for row in shuffled_C_rows}),
        "label_counts": {label: shuffled_C_labels.get(label, 0) for label in LABEL_ORDER},
        "value_counts": dict(shuffled_C_value_order),
        "integer_value_count": sum(bool(re.fullmatch(r"[+-]?\d+", str(row["value"])))
                                    for row in shuffled_C_rows),
    }

    target_real_asserted_ids = {
        row["problem_id"] for row in keys_rows
        if row["target_T"] and row["condition"] == "real" and labels[row["id"]] == "asserted"
    }
    target_shuffled_asserted_ids = {
        row["problem_id"] for row in keys_rows
        if row["target_T"] and row["condition"] == "shuffled" and labels[row["id"]] == "asserted"
    }
    target_real_hit_ids = {row["problem_id"] for row in keys_rows
                           if row["target_T"] and row["condition"] == "real"}
    target_shuffled_hit_ids = {row["problem_id"] for row in keys_rows
                               if row["target_T"] and row["condition"] == "shuffled"}

    first_asserted_char = {}
    for row in keys_rows:
        if row["target_T"] and row["condition"] == "real" and labels[row["id"]] == "asserted":
            old = first_asserted_char.get(row["problem_id"])
            if old is None or row["hit_start"] < old:
                first_asserted_char[row["problem_id"]] = int(row["hit_start"])

    onset_order = {"asserted_on_T": Counter(), "literal_hit_C": Counter()}
    for problem_id, position in first_asserted_char.items():
        onset = unedited_by_id[problem_id]["loop_onset"]
        label = ("no onset detected" if onset is None else
                 "asserted before onset" if position < onset["char_index"] else
                 "onset before asserted hit")
        onset_order["asserted_on_T"][label] += 1
    for row in cohort:
        if not row["literal_hit"]:
            continue
        onset = row["loop_onset"]
        label = ("no onset detected" if onset is None else
                 "hit before onset" if row["first_hit_char"] < onset["char_index"] else
                 "onset before hit")
        onset_order["literal_hit_C"][label] += 1

    end_loop_summary = {}
    for arm, rows in (("unedited", unedited), ("p50c", p50c)):
        end_loop_summary[arm] = {
            "traces": len(rows),
            "end_loop": sum(bool(row["end_loop"]) for row in rows),
            "verbatim_loop_end": sum(bool(row["end_loop_signals"]["verbatim_loop_end"]) for row in rows),
            "repetitive_end": sum(bool(row["end_loop_signals"]["repetitive_end"]) for row in rows),
        }

    eligible_50k = []
    paired_50k = Counter()
    for problem_id in sorted(unedited_by_id):
        left = unedited_by_id[problem_id]
        right = trace_by_key[(problem_id, "p50c")]
        if left["text_chars"] >= 50_000 and right["text_chars"] >= 50_000:
            eligible_50k.append(problem_id)
            paired_50k[(bool(left["loop_onset_50k"]), bool(right["loop_onset_50k"]))] += 1
    onset_50k_all_200 = {}
    for arm, rows in (("unedited", unedited), ("p50c", p50c)):
        onset_n = sum(row["loop_onset_50k"] is not None for row in rows)
        onset_50k_all_200[arm] = {"numerator": onset_n, "denominator": len(rows),
                                  "rate": None if not rows else onset_n / len(rows)}
    onset_50k_all_200["difference_unedited_minus_p50c"] = (
        onset_50k_all_200["unedited"]["rate"] - onset_50k_all_200["p50c"]["rate"])

    sample_texts, tokenizer = load_unedited_samples(manifest)
    if set(sample_texts) != set(unedited_by_id):
        raise ValueError("stored sample IDs do not match the unedited first-200 traces")

    first_hit_early = {}
    for group, rows in (("C", cohort), ("T", target), ("C_minus_T", prompt_echo)):
        hit_rows = [row for row in rows if row["literal_hit"] and row["first_hit_char"] is not None]
        early = sum(row["first_hit_char"] / row["text_chars"] <= 0.15 for row in hit_rows)
        first_hit_early[group] = {"early_first_hit_traces": early, "traces_with_literal_hit": len(hit_rows),
                                  "share": None if not hit_rows else early / len(hit_rows),
                                  "cutoff_share_of_text": 0.15}

    late_real_ids = set()
    late_shuffled_ids = set()
    for row in cohort:
        problem_id = row["problem_id"]
        text_chars = row["text_chars"]
        real_positions = [int(hit["start"]) for hit in row.get("hit_spans", [])]
        if text_chars and any(position / text_chars > 0.15 for position in real_positions):
            late_real_ids.add(problem_id)
        shuffled_rows = keys_by_trace_condition[(problem_id, "shuffled")]
        shuffled_positions = (list(map(int, shuffled_rows[0].get("all_hit_starts", [])))
                              if shuffled_rows else [])
        if text_chars and any(position / text_chars > 0.15 for position in shuffled_positions):
            late_shuffled_ids.add(problem_id)

    def late_gap_on(ids):
        denominator = len(ids)
        real_n = sum(problem_id in late_real_ids for problem_id in ids)
        shuffled_n = sum(problem_id in late_shuffled_ids for problem_id in ids)
        return {"denominator": denominator, "real_late_traces": real_n,
                "shuffled_late_traces": shuffled_n,
                "gap": None if not denominator else (real_n - shuffled_n) / denominator}

    w1_late_sensitivity = {
        "all_C": late_gap_on(cohort_ids),
        "traces_with_real_hit_left": late_gap_on(late_real_ids),
        "union_of_traces_with_any_hit_left": late_gap_on(late_real_ids | late_shuffled_ids),
    }

    literal_positions = {row["problem_id"]: row["first_hit_char"] for row in target if row["literal_hit"]}
    asserted_positions = {problem_id: position for problem_id, position in first_asserted_char.items()}
    target_traces_for_tokens = [row for row in target if row["problem_id"] in literal_positions]
    hit_token_rows = token_positions(tokenizer, sample_texts, target_traces_for_tokens, literal_positions)
    asserted_token_rows = token_positions(tokenizer, sample_texts, target, asserted_positions)
    hit_tokens_by_id = {row["problem_id"]: row["tokens_before"] for row in hit_token_rows}
    asserted_tokens_by_id = {row["problem_id"]: row["tokens_before"] for row in asserted_token_rows}
    token_gaps = {problem_id: asserted_tokens_by_id[problem_id] - hit_tokens_by_id[problem_id]
                  for problem_id in asserted_tokens_by_id if problem_id in hit_tokens_by_id}
    later_gaps = [gap for gap in token_gaps.values() if gap > 0]
    onset_between_positions = 0
    repeated_hit_between_positions = 0
    for problem_id, asserted_char in first_asserted_char.items():
        trace = unedited_by_id[problem_id]
        first_hit_char = trace["first_hit_char"]
        onset = trace["loop_onset"]
        if (onset is not None and first_hit_char <= onset["char_index"] <= asserted_char):
            onset_between_positions += 1
        if any(first_hit_char < int(hit["start"]) < asserted_char
               for hit in trace.get("hit_spans", [])):
            repeated_hit_between_positions += 1

    w3_traces = [row for row in cohort if row["literal_hit"]]
    w3_box_counts = Counter()
    empty_box_occurrences = 0
    empty_box_instruction_echoes = 0
    traces_with_empty_box = set()
    w3_boxes_by_trace = Counter()
    w3_null_last_kind = Counter()
    for row in w3_traces:
        text = sample_texts[row["problem_id"]]
        boxes = scan_boxes(text)
        w3_boxes_by_trace[row["problem_id"]] = len(boxes)
        w3_box_counts["box_tokens"] += len(boxes)
        for box in boxes:
            w3_box_counts[box["kind"]] += 1
            if box["kind"] == "empty":
                empty_box_occurrences += 1
                traces_with_empty_box.add(row["problem_id"])
                nearby = text[max(0, box["start"] - 500):box["start"]].lower()
                if any(term in nearby for term in ("final answer", "final result", "output format",
                                                   "output constraint", "format requirement")):
                    empty_box_instruction_echoes += 1
        if row["pred_first"] is None:
            w3_box_counts["null_pred_first"] += 1
            if not boxes:
                w3_null_last_kind["no box in stored text"] += 1
            elif boxes[-1]["kind"] == "empty":
                w3_null_last_kind["last box empty"] += 1
            else:
                w3_null_last_kind[f"last box {boxes[-1]['kind']}"] += 1
        else:
            w3_box_counts["nonnull_pred_first"] += 1
        if row["earlier_gold_box_last_differs"]:
            w3_box_counts["earlier_gold_box_last_differs"] += 1

    zero_count_upper_bounds = {
        "shuffled_asserted_windows_T": {"events": 0, "n": condition_tables["shuffled"]["window_count"],
                                         "upper_95": clopper_pearson_zero_upper(condition_tables["shuffled"]["window_count"])},
        "shuffled_asserted_traces_T": {"events": 0, "n": len(target),
                                        "upper_95": clopper_pearson_zero_upper(len(target))},
        "shuffled_asserted_windows_C": {"events": 0, "n": len(shuffled_C_rows),
                                         "upper_95": clopper_pearson_zero_upper(len(shuffled_C_rows))},
        "nonnull_wrong_box_traces_C": {"events": 0, "n": len(w3_traces),
                                        "upper_95": clopper_pearson_zero_upper(len(w3_traces))},
    }

    # The two source files are meant to agree. Do not silently report a stale verdict.
    if set(score["predictions"]) != set(prediction_table):
        raise ValueError("score.json and prediction_table.json prediction IDs differ")
    for prediction_id, row in prediction_table.items():
        if score["predictions"][prediction_id]["verdict"] != row["verdict"]:
            raise ValueError(f"verdict disagreement for {prediction_id}")

    paired_tables = score["tables"]["paired_tables"]
    gate4 = score["gate4"]
    gate4_components = gate4["components"]
    gate4_known = gate4_components["known_examples"]
    gate4_kappa = gate4_components["second_reader_kappa"]
    gate4_margin = gate4_components["asserted_margin_T"]
    gate4_errors = gate4_components["reader_errors"]
    p1771 = prediction_table["177.1"]["estimate"]
    p1773 = prediction_table["177.3"]["estimate"]
    p177w1 = prediction_table["177.W1"]["estimate"]

    data = {
        "part": 177,
        "sources": {"score": "score.json", "predictions": "prediction_table.json", "reader": "read.json",
                    "labels": "labels-ds.jsonl", "windows": "window-key.jsonl",
                    "samples": "inputs/eval/unedited/parts/math500_s{0..9}of10/math500_first_samples.jsonl.gz",
                    "tokenizer": "inputs/tokenizer.json", "manifest": "inputs-manifest.json"},
        "cohorts": {
            "C_unedited_capped_open_think": len(cohort),
            "T_gold_not_in_prompt": len(target),
            "C_minus_T_gold_in_prompt": len(prompt_echo),
            "C_literal_hit_traces": sum(row["literal_hit"] for row in cohort),
            "T_literal_hit_traces": len(target_real_hit_ids),
            "T_shuffled_draw_1_literal_hit_traces": len(target_shuffled_hit_ids),
            "T_real_asserted_traces": len(target_real_asserted_ids),
            "T_shuffled_draw_1_asserted_traces": len(target_shuffled_asserted_ids),
            "capped_by_think_closed": read["capped_by_think_closed"],
        },
        "gates_1_to_3": score["read_gates_1_to_3"],
        "gate4": {
            "status": gate4["status"], "passed": gate4["passed"],
            "known_examples": {"correct": gate4_known["correct"], "total": gate4_known["total"],
                               "minimum": gate4_known["minimum"],
                               "ambiguous_expected": sum(row["expected"] == "ambiguous" for row in gate4_known["rows"]),
                               "ambiguous_missed": sum(row["expected"] == "ambiguous" and not row["correct"]
                                                       for row in gate4_known["rows"])},
            "second_reader_kappa": {"kappa": gate4_kappa["kappa"], "pairs": gate4_kappa["pairs"],
                                    "threshold": gate4_kappa["threshold"], "status": gate4_kappa["status"]},
            "asserted_margin_T": gate4_margin,
            "reader_errors": gate4_errors,
            "label_counts": condition_tables,
        },
        "shuffled_C": shuffled_C,
        "zero_count_upper_bounds": zero_count_upper_bounds,
        "reader_retries": {
            "attempts_2_ids": [row["id"] for row in labels_rows if row.get("attempts") == 2],
            "attempts_2_count": sum(row.get("attempts") == 2 for row in labels_rows),
        },
        "funnel_T": {
            "denominator_traces": len(target),
            "real": {"literal_hit_traces": len(target_real_hit_ids), "asserted_traces": len(target_real_asserted_ids)},
            "shuffled_draw_1": {"literal_hit_traces": len(target_shuffled_hit_ids),
                                "asserted_traces": len(target_shuffled_asserted_ids)},
            "window_counts": {condition: condition_tables[condition]["window_count"]
                              for condition in ("real", "shuffled")},
        },
        "prompt_echo_filter": {
            "C_real_literal_hits_numerator": sum(row["literal_hit"] for row in cohort),
            "C_real_literal_hits_denominator": len(cohort),
            "T_real_literal_hits_numerator": len(target_real_hit_ids),
            "T_real_literal_hits_denominator": len(target),
            "removed_gold_in_prompt_traces": len(prompt_echo),
        },
        "onset_order": {group: dict(counts) for group, counts in onset_order.items()},
        "asserted_before_onset_denominators": {
            "T": {"numerator": onset_order["asserted_on_T"].get("asserted before onset", 0),
                  "denominator": sum(row["loop_onset"] is not None for row in target)},
            "C": {"numerator": onset_order["asserted_on_T"].get("asserted before onset", 0),
                  "denominator": sum(row["loop_onset"] is not None for row in cohort)},
        },
        "prediction_table": prediction_table,
        "reported_readouts": score["reported_readouts"],
        "paired_tables": paired_tables,
        "correctness_same_500_problems": read["reported_populations"]["correctness_counts_same_500"],
        "p50c_literal_hit_closed_thinking_on_T": read["reported_populations"]["p50c_closed_thinking_on_T"],
        "end_loop_full_text": end_loop_summary,
        "first_50k_paired": {
            "eligible_pairs": len(eligible_50k),
            "detection_cells": {"both": paired_50k[(True, True)],
                                "unedited_only": paired_50k[(True, False)],
                                "p50c_only": paired_50k[(False, True)],
                                "neither": paired_50k[(False, False)]},
            "rates": prediction_table["177.6"]["estimate"],
        },
        "first_50k_all_200_descriptive": onset_50k_all_200,
        "looped_and_merely_long_C": {
            "looped_traces": len(read["reported_populations"]["looped_C_ids"]),
            "merely_long_traces": len(read["reported_populations"]["merely_long_C_ids"]),
            "asserted_by_class": score["reported_readouts"]["reader"]["asserted_by_C_loop_class"],
        },
        "w1_descriptive": {
            "all_hit_occurrences_early": {"numerator": p177w1["early_real_hits"],
                                           "denominator": p177w1["real_hit_denominator"],
                                           "share": p177w1["early_real_share"]},
            "first_hit_early_traces": p177w1["first_hit_early_traces"],
            "late_gap_all_C": {"late_real": p177w1["late_real_hit_rate"],
                               "late_shuffled": p177w1["late_shuffled_hit_rate"],
                               "gap": p177w1["late_gap"], "ci95": p177w1["late_gap_ci95"],
                               "denominator_C": p177w1["late_C_n"]},
            "first_hit_early_by_population": first_hit_early,
            "late_gap_sensitivity": w1_late_sensitivity,
            "amendment_1_applied": True,
        },
        "token_positions_on_T": {
            "first_literal_hit": {"distribution": distribution(hit_token_rows), "observations": hit_token_rows},
            "first_asserted_selected_window": {"distribution": distribution(asserted_token_rows),
                                                "observations": asserted_token_rows},
            "first_asserted_vs_first_literal": {
                "traces": len(token_gaps),
                "later_token_position_traces": sum(gap > 0 for gap in token_gaps.values()),
                "same_token_position_traces": sum(gap == 0 for gap in token_gaps.values()),
                "median_gap_tokens": quantile(list(token_gaps.values()), 0.50),
                "median_gap_tokens_when_later": quantile(later_gaps, 0.50),
                "onset_between_positions_traces": onset_between_positions,
                "repeated_literal_hit_between_positions_traces": repeated_hit_between_positions,
            },
            "position_definition": "Token count of the stored text prefix before the character position, using the pinned release tokenizer.",
        },
        "w3_raw_box_audit_C_literal_hits": {
            "trace_denominator": len(w3_traces), "box_counts": dict(w3_box_counts),
            "null_last_box_kinds": dict(w3_null_last_kind),
            "empty_box_instruction_echo_occurrences": empty_box_instruction_echoes,
            "empty_box_occurrences": empty_box_occurrences,
            "traces_with_empty_box": len(traces_with_empty_box),
            "traces_with_any_box": sum(n > 0 for n in w3_boxes_by_trace.values()),
            "traces_by_box_occurrence_count": dict(Counter(w3_boxes_by_trace.values())),
        },
        "withdrawal": {"did_fire": bool(score["kept_going_reading_blocked"]),
                       "would_fire_if": "prediction 177.1 failed"},
        "parent_control": {
            "seq_1": "considered and applied: final W1 data, Amendment 1 disclosure, all-C late denominator, and inconclusive verdict.",
            "seq_2": "considered and applied: descriptive early-first-hit splits and tokenized T position summaries.",
        },
    }
    return data


def _cross_counts(table):
    cells = table["cross_table"]["cells_left_then_right"]
    return {
        "0": {"p50c_wrong": cells["0,0"], "p50c_correct": cells["0,1"]},
        "1": {"p50c_wrong": cells["1,0"], "p50c_correct": cells["1,1"]},
    }


def _paired_table_markdown(title, table, rates):
    counts = _cross_counts(table)
    rows = [f"### {title}", "", "| Unedited state | p50c wrong | p50c correct | p50c correct rate in row |",
            "| --- | ---: | ---: | ---: |"]
    for state, label, rate_key in (("0", "No", "false"), ("1", "Yes", "true")):
        n_wrong = counts[state]["p50c_wrong"]
        n_correct = counts[state]["p50c_correct"]
        rate = rates[rate_key]
        rows.append(f"| {label} | {nlink(n_wrong)} | {nlink(n_correct)} | "
                    f"{count_text(rate['numerator'], rate['denominator'])}; "
                    f"{nlink('95%')} CI {rate_interval_text(rate['ci95'])} |")
    return "\n".join(rows)


def estimate_text(predictions, data):
    p = predictions
    e1 = p["177.1"]["estimate"]
    e2 = p["177.2"]["estimate"]
    e3 = p["177.3"]["estimate"]
    e4 = p["177.4"]["estimate"]
    e5 = p["177.5"]["estimate"]
    e6 = p["177.6"]["estimate"]
    eW1 = p["177.W1"]["estimate"]
    eW2 = p["177.W2"]["estimate"]
    eW3 = p["177.W3"]["estimate"]
    zero_bounds = data["zero_count_upper_bounds"]

    def zero_upper(name):
        upper = zero_bounds[name]["upper_95"] * 100.0
        return f"exact {nlink('95%')} upper bound of {nlink(f'{upper:.1f}%')} (two-sided Clopper-Pearson)"

    return {
        "177.1": f"{count_text(e1['known_real_asserted'], e1['denominator_problems'])} real; "
                 f"{count_text(e1['known_shuffled_asserted'], e1['denominator_problems'])} shuffled; "
                 f"shuffled {zero_upper('shuffled_asserted_traces_T')}; "
                 f"gap {pp_text(e1['difference'])}, {nlink('95%')} interval "
                 f"{pp_interval_text([e1['bootstrap_ci95']['lower'], e1['bootstrap_ci95']['upper']])}",
        "177.2": f"{count_text(e2['numerator_known'], e2['denominator'])}; "
                 f"{nlink('95%')} interval {rate_interval_text([e2['ci95']['lower'], e2['ci95']['upper']])}",
        "177.3": f"{count_text(e3['before_onset_known'], e3['known_asserted_traces'])} before onset; "
                 f"{nlink('95%')} interval {rate_interval_text(e3['ci95'] and [e3['ci95']['lower'], e3['ci95']['upper']])}",
        "177.4": f"{count_text(e4['numerator'], e4['denominator'])}; "
                 f"{nlink('95%')} interval {rate_interval_text([e4['ci95']['lower'], e4['ci95']['upper']])}",
        "177.5": f"{nlink(e5['pairs'])} pairs; median onset fraction "
                 f"{float_link(e5['unedited_median'], 4)} to {float_link(e5['p50c_median'], 4)}; "
                 f"difference {float_link(e5['median_difference'], 4)}, {nlink('95%')} interval "
                 f"{float_link(e5['difference_ci95']['lower'], 4)} to {float_link(e5['difference_ci95']['upper'], 4)}",
        "177.6": f"unedited {count_text(e6['unedited']['numerator'], e6['eligible_pairs'])}; "
                 f"p50c {count_text(e6['p50c']['numerator'], e6['eligible_pairs'])}; "
                 f"difference {pp_text(e6['difference_unedited_minus_p50c'])}, {nlink('95%')} interval "
                 f"{pp_interval_text([e6['difference_ci95']['lower'], e6['difference_ci95']['upper']])}",
        "177.W1": f"C real {count_text(eW1['real_hit_rate']['numerator'], eW1['paired_C_n'])}; "
                  f"shuffled {count_text(eW1['shuffled_hit_rate']['numerator'], eW1['paired_C_n'])}; "
                  f"gap {pp_text(eW1['gap'])}, {nlink('95%')} interval "
                  f"{pp_interval_text([eW1['gap_ci95']['lower'], eW1['gap_ci95']['upper']])}; "
                  f"all-hit early share {count_text(eW1['early_real_hits'], eW1['real_hit_denominator'])}; "
                  f"{nlink('95%')} interval {rate_interval_text([eW1['early_real_share_ci95']['lower'], eW1['early_real_share_ci95']['upper']])}; "
                  f"first-hit early {count_text(eW1['first_hit_early_traces']['numerator'], eW1['first_hit_early_traces']['denominator'])}; "
                  f"late all-C gap {pp_text(eW1['late_gap'])}, {nlink('95%')} interval "
                  f"{pp_interval_text([eW1['late_gap_ci95']['lower'], eW1['late_gap_ci95']['upper']])}",
        "177.W2": f"{count_text(eW2['asserted_windows'], eW2['total_C_windows'])} shuffled windows asserted on `C`; "
                  f"{zero_upper('shuffled_asserted_windows_C')}",
        "177.W3": f"reading (a) {count_text(eW3['reading_a_box_fails_correct0']['numerator'], eW3['denominator_C_literal_hit_traces'])}; "
                  f"{zero_upper('nonnull_wrong_box_traces_C')}; "
                  f"reading (b) {count_text(eW3['reading_b_any_no_correct_box_including_null']['numerator'], eW3['denominator_C_literal_hit_traces'])}; "
                  f"{nlink('95%')} interval {rate_interval_text(eW3['reading_b_any_no_correct_box_including_null']['ci95'])}",
    }


def build_readme(data):
    N = nlink
    cohorts = data["cohorts"]
    pred = data["prediction_table"]
    gate4 = data["gate4"]
    gateknown = gate4["known_examples"]
    kappa = gate4["second_reader_kappa"]
    margin = gate4["asserted_margin_T"]
    labels = gate4["label_counts"]
    funnel = data["funnel_T"]
    ccap = data["prompt_echo_filter"]
    onset = data["onset_order"]
    p1773 = pred["177.3"]["estimate"]
    p1774 = pred["177.4"]["estimate"]
    p1775 = pred["177.5"]["estimate"]
    p1776 = pred["177.6"]["estimate"]
    w1 = pred["177.W1"]["estimate"]
    w2 = pred["177.W2"]["estimate"]
    w3 = pred["177.W3"]["estimate"]
    readouts = data["reported_readouts"]["reader"]
    correctness = data["correctness_same_500_problems"]
    box_audit = data["w3_raw_box_audit_C_literal_hits"]
    token_data = data["token_positions_on_T"]
    zero_bounds = data["zero_count_upper_bounds"]
    shuffled_C = data["shuffled_C"]
    retries = data["reader_retries"]
    onset_before_onset = data["asserted_before_onset_denominators"]
    all_200_50k = data["first_50k_all_200_descriptive"]
    w1_late_sensitivity = data["w1_descriptive"]["late_gap_sensitivity"]
    first_asserted_vs_literal = token_data["first_asserted_vs_first_literal"]
    hit_dist = token_data["first_literal_hit"]["distribution"]
    asserted_dist = token_data["first_asserted_selected_window"]["distribution"]
    hit_median = f"{hit_dist['median']:.0f}"
    hit_q1 = f"{hit_dist['q1']:.1f}".rstrip("0").rstrip(".")
    hit_q3 = f"{hit_dist['q3']:.1f}".rstrip("0").rstrip(".")
    asserted_median = f"{asserted_dist['median']:.0f}"
    asserted_q1 = f"{asserted_dist['q1']:.1f}".rstrip("0").rstrip(".")
    asserted_q3 = f"{asserted_dist['q3']:.1f}".rstrip("0").rstrip(".")
    token_gap_median = f"{first_asserted_vs_literal['median_gap_tokens']:.0f}"
    token_gap_later_median = f"{first_asserted_vs_literal['median_gap_tokens_when_later']:.0f}"

    def cp_upper(key):
        value = zero_bounds[key]["upper_95"] * 100.0
        return f"exact {N('95%')} upper bound of {N(f'{value:.1f}%')} (two-sided Clopper-Pearson)"

    title = (f"# [Part 177]({DESIGN_LINK}): {N(funnel['real']['asserted_traces'])} of "
             f"{N(funnel['denominator_traces'])} unedited [Qwen3.5-0.8B]({DESIGN_LINK}) traces that ran out of tokens "
             f"while thinking had a window a reader judged to state the gold answer as a result, against "
             f"{N(funnel['shuffled_draw_1']['asserted_traces'])} for shuffled answers")
    first_paragraph = (
        f"Scope: this reads MIT's stored sample-{dlink(0)} texts for the first {N(200)} rows of "
        f"[MATH-500]({DESIGN_LINK}) in each arm, the same problem IDs. The authors used the model and stack in the design, "
        "with thinking on and the registered sampling: "
        f"temperature {dlink('0.6')}, top_p {dlink('0.95')}, top_k {dlink(20)}, "
        f"presence penalty {dlink('1.5')}, up to {dlink('32,768')} new tokens on vLLM. Nothing was run here. "
        f"`capped` means decoded text retokenised to at least {dlink('32,500')} tokens with the pinned release tokenizer. "
        f"Target `T` is the {N(cohorts['T_gold_not_in_prompt'])} of {N(cohorts['C_unedited_capped_open_think'])} capped, open-thinking traces "
        "whose gold does not appear in the problem text. The measure is a literal string match on the gold value plus a reader's `asserted` "
        f"label on up to {dlink(3)} windows per trace. The literal rule misses mathematically equal forms. Numeric golds reject neighboring letters, digits, underscores, plus signs, minus signs, and decimal points that continue a decimal. "
        "Short nonnumeric tokens reject neighboring letters, digits, and underscores. Other golds use plain substring matching with no boundary, so a string can match inside a longer expression. "
        f"Amendment {dlink(2)}, point {dlink(3)}, records these rules. The Qwen sentence under test in MIT's [post](https://rikiyat.github.io/blog/weight-space-programming/) is "
        "“it lets the model finish answers it could already work out.” See Amendment "
        f"{dlink(2)}. On the unedited arm, {N(cohorts['C_unedited_capped_open_think'])} traces were capped while thinking remained open. "
        f"On `T`, {N(funnel['real']['asserted_traces'])} of {N(funnel['denominator_traces'])} traces had a literal gold hit with an `asserted` window. "
        f"The shuffled control had {N(funnel['shuffled_draw_1']['asserted_traces'])} asserted traces out of {N(funnel['denominator_traces'])}, with an "
        f"{cp_upper('shuffled_asserted_traces_T')}. "
        "A window label does not prove the model would have stopped."
    )

    capped = data["cohorts"]["capped_by_think_closed"]
    cap_rows = []
    for arm in ("unedited", "p50c"):
        row = capped[arm]
        values = [row[f"capped={cap},think_closed={closed}"]["numerator"]
                  for cap, closed in (("True", "False"), ("True", "True"),
                                      ("False", "False"), ("False", "True"))]
        cap_rows.append(f"| {arm} | " + " | ".join(N(v) for v in values) + " |")
    scope_section = [
        "## What did the cap and prompt-echo filter count?", "",
        "The text stores no finish reason. `capped` is the retokenised decoded-text proxy above; `</think>` marks closed thinking.", "",
        "| Arm | Capped, thinking open | Capped, thinking closed | Not capped, thinking open | Not capped, thinking closed |",
        "| --- | ---: | ---: | ---: | ---: |", *cap_rows, "",
        f"The {N(capped['unedited']['capped=False,think_closed=True']['numerator'])} unedited traces that finished are the not-capped, thinking-closed row, so they are outside cohort `C`. "
        f"The gold-in-prompt filter removes {N(ccap['removed_gold_in_prompt_traces'])} of the {N(cohorts['C_unedited_capped_open_think'])} traces in `C`: "
        f"literal gold appears in {count_text(ccap['C_real_literal_hits_numerator'], ccap['C_real_literal_hits_denominator'])} on `C`, "
        f"and {count_text(ccap['T_real_literal_hits_numerator'], ccap['T_real_literal_hits_denominator'])} on `T`. "
        f"The shuffled draw-{N(1)} value appears in {count_text(funnel['shuffled_draw_1']['literal_hit_traces'], funnel['denominator_traces'])} on `T`.", "",
        f"The target-`T` funnel counts traces, not windows: {N(funnel['denominator_traces'])} traces; real gold literal hit "
        f"{N(funnel['real']['literal_hit_traces'])}; real gold judged asserted {N(funnel['real']['asserted_traces'])}; "
        f"shuffled draw-{N(1)} literal hit {N(funnel['shuffled_draw_1']['literal_hit_traces'])}; shuffled judged asserted "
        f"{count_text(funnel['shuffled_draw_1']['asserted_traces'], funnel['denominator_traces'])}, with an {cp_upper('shuffled_asserted_traces_T')}. "
        "The panel shows the same filter and control.", "",
    ]

    ambiguous_missed = gateknown["ambiguous_missed"]
    ambiguous_total = gateknown["ambiguous_expected"]
    known_nonambig = ambiguous_total - ambiguous_missed
    real_counts = labels["real"]["label_counts"]
    shuffled_counts = labels["shuffled"]["label_counts"]
    label_table = [
        f"| Reader label | Real windows | Shuffled draw-{N(1)} windows |",
        "| --- | ---: | ---: |",
    ]
    for label in LABEL_ORDER:
        label_table.append(f"| {label} | {N(real_counts[label])} | {N(shuffled_counts[label])} |")
    label_table.append(f"| Total | {N(labels['real']['window_count'])} | {N(labels['shuffled']['window_count'])} |")
    shuffled_value_distribution = ", ".join(
        f"{N(value)}: {N(count)}" for value, count in shuffled_C["value_counts"].items())
    kappa_value = f"{kappa['kappa']:.2f}"
    kappa_bar = f"{kappa['threshold']:.1f}"
    reader_section = [
        "## Is the window reader reliable enough for this claim?", "",
        f"DeepSeek V4.1 Flash labeled selected windows around literal hits, not full traces. Claude Haiku 5.5 read the second-reader sample. "
        f"The known-window gate got "
        f"{count_text(gateknown['correct'], gateknown['total'])}, at its minimum of {N(gateknown['minimum'])}. "
        f"All {N(ambiguous_missed)} known-window misses were among the {N(ambiguous_total)} examples expected to be `ambiguous`; "
        f"the remaining {N(known_nonambig)} was labelled `ambiguous` correctly. Agreement with the second reader was kappa "
        f"{N(kappa_value)} on {N(kappa['pairs'])} pairs against a {N(kappa_bar)} bar. "
        "Both checks passed narrowly.", "",
        f"The real asserted-window share was {count_text(margin['real_window_share']['numerator'], margin['real_window_share']['denominator'])}; "
        f"the shuffled share was {count_text(margin['shuffled_window_share']['numerator'], margin['shuffled_window_share']['denominator'])}. "
        f"That gives an {cp_upper('shuffled_asserted_windows_T')}. "
        f"That is a gap of {pp_text(margin['difference'])}, above the registered {dlink(25)} point gate. "
        f"Reader errors were {count_text(gate4['reader_errors']['errors'], gate4['reader_errors']['expected_windows'])}, under the {dlink('2%')} cap. Gate {N(4)} passed.", "",
        *label_table, "",
        f"This table has {N(labels['shuffled']['window_count'])} shuffled windows on `T`. Prediction {nlink('177.W2', PREDICTION_LINK)} is on `C` and uses "
        f"{N(shuffled_C['window_count'])} shuffled windows. The {N(shuffled_C['window_count'] - labels['shuffled']['window_count'])} extra windows come from `C` minus `T`.", "",
        f"The cold check reports Claude Haiku 5.5's blind sample of {nlink(50, 'COLD.md')} windows. It agreed on {nlink(47, 'COLD.md')}; the {nlink(3, 'COLD.md')} disagreements were borderline. "
        "It found that `asserted` sometimes precedes a doubt in the next line and sometimes sits inside a repeated loop. "
        f"Of its {nlink(30, 'COLD.md')} windows read as `asserted`, {nlink(6, 'COLD.md')} were followed by doubt in the next line and "
        f"{nlink(5, 'COLD.md')} sat inside a repeated loop. This was a sample of {nlink(50, 'COLD.md')}.", "",
        f"Each reader label allowed at most {N(2)} sent attempts. Empty or invalid replies and service errors used the remaining attempt. "
        f"Pre-send DNS failures retried with exponential backoff for up to {N('2 hours')}. {N(retries['attempts_2_count'])} labels have `attempts = 2`: "
        + ", ".join(f"`{window_id}`" for window_id in retries["attempts_2_ids"]) + ".", "",
    ]

    asserted_onset = onset["asserted_on_T"]
    literal_onset = onset["literal_hit_C"]
    asserted_total = sum(asserted_onset.values())
    literal_total = sum(literal_onset.values())
    order_table = [
        "| Trace group | Onset order | Traces |",
        "| --- | --- | ---: |",
        f"| Asserted selected windows on `T` (n={N(asserted_total)}) | Before onset | {N(asserted_onset.get('asserted before onset', 0))} |",
        f"| Asserted selected windows on `T` (n={N(asserted_total)}) | Onset before asserted hit | {N(asserted_onset.get('onset before asserted hit', 0))} |",
        f"| Asserted selected windows on `T` (n={N(asserted_total)}) | No onset detected | {N(asserted_onset.get('no onset detected', 0))} |",
        f"| Literal-hit traces on `C` (n={N(literal_total)}) | Hit before onset | {N(literal_onset.get('hit before onset', 0))} |",
        f"| Literal-hit traces on `C` (n={N(literal_total)}) | Onset before hit | {N(literal_onset.get('onset before hit', 0))} |",
        f"| Literal-hit traces on `C` (n={N(literal_total)}) | No onset detected | {N(literal_onset.get('no onset detected', 0))} |",
    ]
    onset_section = [
        "## Did the asserted value appear before the detected onset?", "",
        f"Prediction {nlink('177.3', PREDICTION_LINK)} held: the first asserted selected window was before the line-rule onset in "
        f"{count_text(p1773['before_onset_known'], p1773['known_asserted_traces'])}. The remaining asserted trace has no detected onset, shown in its own row. "
        f"Across all literal-hit traces in `C`, the table keeps no-onset traces separate. The onset rule is not ground truth.", "",
        *order_table, "",
        f"This prediction's denominator uses traces with an asserted gold and a detected onset. "
        f"Its {N('95%')} interval is {rate_interval_text([p1773['ci95']['lower'], p1773['ci95']['upper']])}.", "",
        f"On `T`, an asserted gold preceded onset in {count_text(onset_before_onset['T']['numerator'], onset_before_onset['T']['denominator'])} traces with a detected onset. "
        f"Across `C`, the corresponding count is {count_text(onset_before_onset['C']['numerator'], onset_before_onset['C']['denominator'])}. "
        "The `T` denominator has 14 fewer traces because their gold appears in the problem text.", "",
        f"Among all `C` traces, `asserted` appears on {count_text(readouts['asserted_by_C_loop_class']['looped']['all_C_traces']['numerator_known_yes'], readouts['asserted_by_C_loop_class']['looped']['all_C_traces']['denominator'])} looped traces and "
        f"{count_text(readouts['asserted_by_C_loop_class']['merely_long']['all_C_traces']['numerator_known_yes'], readouts['asserted_by_C_loop_class']['merely_long']['all_C_traces']['denominator'])} merely-long traces. "
        f"Among read hits, the first read hit was asserted in {count_text(readouts['first_read_hit_asserted']['numerator_known_yes'], readouts['first_read_hit_asserted']['denominator'])}, "
        f"the last read hit in {count_text(readouts['last_read_hit_asserted']['numerator_known_yes'], readouts['last_read_hit_asserted']['denominator'])}, and "
        f"{count_text(readouts['rejected_or_doubted_read_hit_share']['numerator'], readouts['rejected_or_doubted_read_hit_share']['valid_read_hit_denominator'])} were rejected or doubted.", "",
    ]

    ci5 = p1775["difference_ci95"]
    ci6 = p1776["difference_ci95"]
    line_rates = p1776
    endloops = data["end_loop_full_text"]
    pers_section = [
        "## What does the line rule show after the edit?", "",
        f"Prediction {nlink('177.4', PREDICTION_LINK)} failed: {count_text(p1774['numerator'], p1774['denominator'])} of the p50c capped traces with open thinking had a detected onset, below the {dlink('80%')} bar. "
        "This is the line signature among residual failures. It persisted in about three quarters of them and says nothing about incidence.", "",
        f"Prediction {nlink('177.5', PREDICTION_LINK)} held on the median. In {N(p1775['pairs'])} paired problems with an onset in each arm, the median onset fraction was "
        f"{float_link(p1775['unedited_median'], 4)} in unedited and {float_link(p1775['p50c_median'], 4)} in p50c. "
        f"The difference interval, {float_link(ci5['lower'], 4)} to {float_link(ci5['upper'], 4)}, includes {N(0)}.", "",
        f"Prediction {nlink('177.6', PREDICTION_LINK)} held over {N(line_rates['eligible_pairs'])} pairs where both texts ran past {dlink('50,000')} characters. "
        "This selects on post-edit length, so the p50c member is a selected subset. In that subset: "
        f"{count_text(line_rates['unedited']['numerator'], line_rates['eligible_pairs'])} unedited and "
        f"{count_text(line_rates['p50c']['numerator'], line_rates['eligible_pairs'])} p50c. The rate difference was "
        f"{pp_text(line_rates['difference_unedited_minus_p50c'])}, with {N('95%')} interval "
        f"{pp_interval_text([ci6['lower'], ci6['upper']])}. This is the rate of the line rule's signature in the first {dlink('50,000')} characters, not a measure of loops in general.", "",
        f"Descriptive and unregistered on all {N(200)} pairs, each text was read for up to {dlink('50,000')} characters or its full length if shorter. "
        f"The rates were {count_text(all_200_50k['unedited']['numerator'], all_200_50k['unedited']['denominator'])} unedited and "
        f"{count_text(all_200_50k['p50c']['numerator'], all_200_50k['p50c']['denominator'])} p50c, a gap of "
        f"{pp_text(all_200_50k['difference_unedited_minus_p50c'])}. Exposure is unequal because some texts are shorter than the window.", "",
        f"The full-text end-loop rule flagged {count_text(endloops['unedited']['end_loop'], endloops['unedited']['traces'])} unedited traces and "
        f"{count_text(endloops['p50c']['end_loop'], endloops['p50c']['traces'])} p50c traces. The `verbatim_loop_end` and `repetitive_end` counts cannot be verified from the pinned inputs. "
        "[mathqa.py](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/code/mathqa.py) is in the upstream logs repository. `repetitive_end` is rebuilt from a docstring there.", "",
    ]

    first_early = data["w1_descriptive"]["first_hit_early_by_population"]
    early_occ = data["w1_descriptive"]["all_hit_occurrences_early"]
    late = data["w1_descriptive"]["late_gap_all_C"]
    w1_table = [
        f"| Population | First hit in first {dlink('15%')} |",
        "| --- | ---: |",
    ]
    for key, label in (("C", "All `C` traces with a real hit"),
                       ("T", "Target `T` traces with a real hit"),
                       ("C_minus_T", "`C` minus `T`, where gold is in prompt")):
        item = first_early[key]
        w1_table.append(f"| {label} | {count_text(item['early_first_hit_traces'], item['traces_with_literal_hit'])} |")
    late_sensitivity_table = [
        "| Denominator | Real hits left | Shuffled hits left | Late gap |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (("all_C", "All of `C`"),
                       ("traces_with_real_hit_left", "Traces with a real hit left"),
                       ("union_of_traces_with_any_hit_left", "Union with a real or shuffled hit left")):
        item = w1_late_sensitivity[key]
        late_sensitivity_table.append(
            f"| {label} | {count_text(item['real_late_traces'], item['denominator'])} | "
            f"{count_text(item['shuffled_late_traces'], item['denominator'])} | {pp_text(item['gap'])} |")
    w1_section = [
        "## Did early prompt echo explain the literal-hit gap?", "",
        f"On `C`, real literal hits were {count_text(w1['real_hit_rate']['numerator'], w1['paired_C_n'])} and shuffled draw-{N(1)} hits were "
        f"{count_text(w1['shuffled_hit_rate']['numerator'], w1['paired_C_n'])}. The all-occurrence early-hit share was "
        f"{count_text(early_occ['numerator'], early_occ['denominator'])}. Its first-hit reading was "
        f"{count_text(w1['first_hit_early_traces']['numerator'], w1['first_hit_early_traces']['denominator'])} on `C`; "
        f"on `T` it was {count_text(first_early['T']['early_first_hit_traces'], first_early['T']['traces_with_literal_hit'])}. "
        f"These are different measures because repeated values can appear many times in a trace.", "",
        *w1_table, "",
        f"After dropping hits in the first {dlink('15%')} of text, the three denominator readings are:", "",
        *late_sensitivity_table, "",
        f"Amendment {dlink(1)} changed the denominator. Amendment {dlink(2)}, point {dlink(2)}, says the Reading rules name which trace counts as a hit but give no denominator; "
        f"the phrase `gap on C` supplies it. The verdict is inconclusive under each denominator. Neither the full claim nor its under-{dlink('10%')} falsifier holds.", "",
        f"Prediction {nlink('177.W2', PREDICTION_LINK)} failed at {count_text(w2['asserted_windows'], shuffled_C['window_count'])} shuffled windows on `C`. "
        f"That gives an {cp_upper('shuffled_asserted_windows_C')}. The shuffled values were value:count: {shuffled_value_distribution}. "
        "This control tests whether the reader calls an incidental matched value `asserted`. It does not show that a donor value could not be stated as an answer.", "",
    ]

    box_section = [
        "## Did the literal-hit traces emit a wrong final box?", "",
        f"Under Weaver's registered reading (a), {count_text(w3['reading_a_box_fails_correct0']['numerator'], w3['denominator_C_literal_hit_traces'])} had a non-null box that failed [`correct[0]`](numbers177.json), so {nlink('177.W3', PREDICTION_LINK)} failed. "
        f"That gives an {cp_upper('nonnull_wrong_box_traces_C')}. On the plain wording, null `pred_first` does not match gold: "
        f"{count_text(box_audit['box_counts']['null_pred_first'], w3['denominator_C_literal_hit_traces'])} were null, and the {N(w3['denominator_C_literal_hit_traces'] - box_audit['box_counts']['null_pred_first'])} non-null boxes were correct. "
        "The verdict depends on this reading. Weaver ruled before any data, and the reading was registered before the scores.", "",
        f"When nulls are included, {count_text(w3['reading_b_any_no_correct_box_including_null']['numerator'], w3['denominator_C_literal_hit_traces'])} had no correct box. "
        f"{N(box_audit['null_last_box_kinds'].get('no box in stored text', 0))} had no `\\boxed` or `\\fbox` token in the stored text; "
        f"{N(box_audit['null_last_box_kinds'].get('last box empty', 0))} ended on an empty box. The raw text scan found "
        f"{N(box_audit['box_counts']['box_tokens'])} box tokens across {N(box_audit['traces_with_any_box'])} traces: "
        f"{N(box_audit['empty_box_occurrences'])} empty `\\boxed{{}}` occurrences in {N(box_audit['traces_with_empty_box'])} traces and "
        f"{N(box_audit['box_counts'].get('closed', 0))} filled boxes. All {N(box_audit['empty_box_occurrences'])} empty tokens occur near repeated final-answer or output-format instructions. "
        "For example, an echoed instruction says to put the final answer within the box, followed by an empty `\\boxed{}` before the reasoning continues. These boxes contain no answer. The extraction returns null for an empty last box, just as it does when no box exists.", "",
        f"An earlier box held the literal gold while the last box differed in {count_text(box_audit['box_counts'].get('earlier_gold_box_last_differs', 0), box_audit['trace_denominator'])}. "
        f"There were {N(w3['denominator_C_literal_hit_traces'])} literal-hit traces; {count_text(box_audit['box_counts']['null_pred_first'], box_audit['trace_denominator'])} had null `pred_first`. "
        "In these rows, null `pred_first` is what running out while still thinking looks like: almost nothing was emitted as an extractable answer. It is an observed output state, not missing data.", "",
    ]

    pair_literal = data["paired_tables"]["unedited_literal_hit_vs_p50c_correct_all_200"]
    pair_asserted = data["paired_tables"]["unedited_asserted_vs_p50c_correct_C"]
    pair_literal_rates = pair_literal["p50c_correct_by_unedited_literal_hit"]
    pair_asserted_rates = pair_asserted["p50c_correct_by_unedited_asserted"]
    pair_section = [
        "## How do the paired correctness tables line up?", "",
        "The first table uses all paired problems. The second uses `C`, where the unedited trace has a reader label. Each table gives the p50c wrong and correct cells. Rates are p50c [`correct[0]`](numbers177.json) within each unedited row.", "",
        _paired_table_markdown("Unedited literal hit vs p50c correctness on all paired problems",
                               pair_literal, pair_literal_rates), "",
        _paired_table_markdown("Unedited asserted vs p50c correctness on `C`",
                               pair_asserted, pair_asserted_rates), "",
        "No cell is called new computation. These paired observations use sampled outputs and a string rule; they cannot show whether the edit changed what the model computes or whether it can stop.", "",
    ]

    correctness_table = [
        "| Sample | Unedited correct | p50c correct |",
        "| --- | ---: | ---: |",
    ]
    for index, label in enumerate(("0", "1", "2", "3")):
        left = correctness["unedited"]["per_sample_correct_n"][index]
        right = correctness["p50c"]["per_sample_correct_n"][index]
        total = correctness["unedited"]["denominator_problems"]
        correctness_table.append(f"| {N(label)} | {count_text(left, total)} | {count_text(right, total)} |")
    total = correctness["unedited"]["denominator_problems"]
    any_left = correctness["unedited"]["any_of_four_correct_n"]
    any_right = correctness["p50c"]["any_of_four_correct_n"]
    correctness_table.append(f"| Any of {N(4)} | {count_text(any_left, total)} | {count_text(any_right, total)} |")
    closed_t = data["p50c_literal_hit_closed_thinking_on_T"]
    correct_section = [
        "## What happened on the same problems outside the cap cohort?", "",
        f"Across the same {N(total)} problem IDs, `correct` by sample and any-of-{N(4)} was:", "",
        *correctness_table, "",
        f"On target `T`, p50c had a literal hit on {count_text(closed_t['numerator_literal_hit'], closed_t['denominator_closed_traces'])} traces where thinking closed. This is shown beside the capped-trace rates, not pooled with them.", "",
        f"Every rate in this write-up that conditions on `C` or a capped failure is among traces that hit the cap. The first {N(200)} rows are not a random sample of {dlink('MATH-500')}.", "",
    ]

    limits = [
        "## What can this read establish, and what are its limits?", "",
        "The post's own line is quoted and linked above. Prior-art notes are from Muse's search report, not read by Claude; the papers and detector references are left unlinked here. This result tests the post on one narrow reading: a value stated in a cut trace. That does not prove the model would have stopped.", "",
        f"Amendment {dlink(1)} was made after scores were read. It changes the W1 late-gap denominator and corrects the verdict rule. Amendment {dlink(2)}, point {dlink(2)}, explains that the Reading rules did not name that denominator. The only code change after registration was an import needed by isolated Python; the commit is in git.", "",
        f"The withdrawal rule did not fire because {nlink('177.1', PREDICTION_LINK)} held. It would have fired if that prediction failed: the reading “it had stated the answer and kept going” would have been withdrawn and the score blocked.", "",
        f"Limits from the design: {dlink(1)} model, {dlink(1)} benchmark, {dlink(1)} edit, and the authors' stored samples on their stack; only the first {N(200)} rows, not a random sample; `capped` inferred by retokenising decoded text; a literal string matcher misses mathematically equal forms; a reader judges a window, not the whole trace, so `asserted` is audit-grade and not proof; shuffled answers are a label permutation, not chance reasoning; the line rule is not loop ground truth, and onset depends on how long text ran; nothing here shows how the edit works, and nothing was run on our hardware. Residual-failure rates say nothing about incidence.", "",
        "## Registered predictions", "",
        "The text, confidence, estimate, and verdict below come from the registered table. The first cells are plain prediction IDs.", "",
        "| # | Prediction (verbatim, with confidence) | Estimate | Verdict |",
        "| --- | --- | --- | --- |",
    ]
    estimates = estimate_text(pred, data)
    for prediction_id, row in pred.items():
        statement = link_registered_numbers(row["prediction"])
        statement += f" ({row['confidence']})"
        limits.append(f"| {prediction_id} | {statement} | {estimates[prediction_id]} | "
                      f"[{row['verdict']}]({PREDICTION_LINK}) |")

    token_section = [
        "## Where in the text were the first hits?", "",
        f"These token-position summaries are descriptive and unregistered. A token position counts the tokenized stored-text prefix before the character position. Quartiles use Hyndman-Fan Type {N(6)}, the exclusive method with linear interpolation at p times n plus one.",
        f"On target `T`, the median tokens before the first literal hit was {N(hit_median)}, with IQR Q1 {N(hit_q1)} to Q3 {N(hit_q3)}, across {N(hit_dist['n'])} traces.",
        f"Before the first selected window labelled asserted, the median was {N(asserted_median)} tokens, with IQR Q1 {N(asserted_q1)} to Q3 {N(asserted_q3)}, across {N(asserted_dist['n'])} traces.", "",
        f"Among the {N(first_asserted_vs_literal['traces'])} `T` traces with an asserted window, the first asserted window was later in "
        f"{count_text(first_asserted_vs_literal['later_token_position_traces'], first_asserted_vs_literal['traces'])}. The median gap was "
        f"{N(token_gap_median)} tokens across all traces and {N(token_gap_later_median)} tokens among later windows. "
        f"This is not a clean time-to-assertion measure: a detected onset fell between the positions in {N(first_asserted_vs_literal['onset_between_positions_traces'])} traces, "
        f"and another literal hit fell between them in {N(first_asserted_vs_literal['repeated_literal_hit_between_positions_traces'])} traces.", "",
    ]
    source_statement = (
        "Sources: The log inputs and the excerpts in `windows.jsonl` come from "
        "[lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) at commit "
        "[`4d09ba6`](https://github.com/RikiyaT/lm_understand-logs/commit/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f) "
        "by Takehi, Teoh, Zhang and Khattab under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). "
        "`windows.jsonl` holds excerpts of their logs and [MATH-500 problem text](https://huggingface.co/datasets/HuggingFaceH4/MATH-500). "
        "The MATH-500 text carries its source's own terms. "
        "The manifest pins the MATH-500 test set and Qwen tokenizer separately."
    )
    cold_summary = (
        f"Cold check: Claude Haiku 5.5 reproduced the arithmetic, cohorts, literal-hit counts, reader results, box cells and figure from pinned inputs. "
        f"It reported {nlink(2, 'COLD.md')} high, {nlink(6, 'COLD.md')} medium and {nlink(7, 'COLD.md')} low findings. "
        "Every finding is folded. "
        "[Cold check](COLD.md)."
    )
    lines = [title, "", first_paragraph, "",
             "![Code-drawn panels: target funnel, onset order, and paired line detections.](plot177.png)", "",
             *scope_section, *reader_section, *onset_section, *pers_section, *w1_section,
             *token_section, *box_section, *pair_section, *correct_section, *limits,
             "", source_statement, "", cold_summary,
             "", "Rebuild with: `python` [numbers177.py](numbers177.py) and `python` [plot177.py](plot177.py)."]
    return "\n".join(lines) + "\n"


def main():
    data = build_numbers()
    write_json(ROOT / NUMBERS_FILE, data)
    readme = build_readme(data)
    (ROOT / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    source_line = next(line for line in readme.splitlines() if line.startswith("Sources:"))
    (ROOT / "SOURCES.md").write_text("# Sources\n\n" + source_line + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"part": data["part"], "T": data["cohorts"]["T_gold_not_in_prompt"],
                      "asserted": data["funnel_T"]["real"]["asserted_traces"],
                      "W1": data["prediction_table"]["177.W1"]["verdict"],
                      "token_first_hit_n": data["token_positions_on_T"]["first_literal_hit"]["distribution"]["n"],
                      "token_asserted_n": data["token_positions_on_T"]["first_asserted_selected_window"]["distribution"]["n"]},
                     ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
