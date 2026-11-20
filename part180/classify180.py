#!/usr/bin/env python3
"""Classify Part 180 head scores on the 18 and shared 8 concept-neuron sets."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TOP_K = 13
SHUFFLES = 10_000
SHUFFLE_SEED = 180
METRICS = ("dself", "dcont", "dself_heldout", "dcont_heldout")

# Threshold wording is copied from PART180-DESIGN.md with its section label.
DESIGN_THRESHOLDS = {
    "gate2": {"value": "At least 10 of the 18 neurons pass inclusion. Fewer: stop and redesign.",
              "design_line": "Controls and gates — Gate 2"},
    "gate3_hook_placement": {"value": "On identical teacher-forced ids, upstream residuals are unchanged, the post-hook block output adds the vector, and the next block receives it; miswired and absent hooks are red.",
                              "design_line": "Controls and gates — Gate 3, hook placement"},
    "stage_b_self_content": {"value": "Self head: top 13 (5%) on dself and on dself'. Content head: the same for dcont and dcont'.",
                              "design_line": "Stages — B"},
    "stage_b_both": {"value": "Both candidate: in top 13 of dself and dcont on selection AND top 13 of both on held-out (selection bar, not twice it).",
                      "design_line": "Stages — B"},
    "shuffle": {"value": "The four-way top-13 intersection is compared with 10,000 within-layer permutations of paired selection/held-out content scores.",
                "design_line": "Controls and gates — Null for 'both'"},
    "norm_control": {"value": "Compare push and norm increments over M0; assess only push increments >=0.05 and block if either measure is mostly norm or fewer than half are assessable.",
                      "design_line": "Controls — Norm control"},
    "180.1": {"value": "Median dself/dcont Spearman below 0.3 and at most 3 of top 10 dself heads in top 10 dcont.",
              "design_line": "Predictions — 180.1"},
    "180.3": {"value": "At least 5 of top 10 dself heads are in layers 0 to 3.",
              "design_line": "Predictions — 180.3"},
    "W1": {"value": "Frame penalty exceeds crossed selfX for at least half; falsified under a quarter.",
           "design_line": "Weaver's predictions — 180.W1"},
    "W2": {"value": "At most 4 of raw top-13 dself heads remain top 13 after g residual; falsified at 9 or more.",
           "design_line": "Weaver's predictions — 180.W2"},
    "W3": {"value": "No held-out head is top 13 in both dself' and dcont'.", "design_line": "Weaver's predictions — 180.W3"},
}


def normalise_path(value: str | Path) -> Path:
    raw = os.fspath(value)
    if os.name != "nt":
        raw = raw.replace(chr(92), "/")
    return Path(raw).expanduser()


def _load_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
    return rows


def _bool_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in ("true", "yes", "1", "present"):
            return True
        if lowered in ("false", "no", "0", "absent"):
            return False
    return None


def _part92_label_values(row):
    if row is None or "error" in row:
        return {"shows_persona": None, "self_claim": None}
    return {key: _bool_value(row.get(key)) for key in ("shows_persona", "self_claim")}


def _fields_match(values, expected):
    """Three-valued conjunction: a known mismatch fails; absent labels stay unknown."""
    if any(value is not None and value != wanted for value, wanted in zip(values, expected)):
        return False
    if any(value is None for value in values):
        return None
    return True


def _reader_key(row):
    value = str(row.get("key", "")).strip().upper()
    return value if value in {"S0", "S1", "A0", "A1", "C0", "C1", "B0", "B1"} else None


def deepseek_concept_ids(score92b):
    """Read shared concepts from Part 92b's recorded kind comparison."""
    units = score92b.get("kind_comparison", {}).get("units")
    if not isinstance(units, list):
        raise ValueError("score92b.json kind_comparison.units must be a list")
    return sorted({str(row["id"]) for row in units
                   if row.get("part92_kind") == "concept" and row.get("ds_kind") == "concept"
                   and row.get("id") is not None})


def _rankdata(values):
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average = (start + 1 + stop) / 2.0
        for index in order[start:stop]:
            ranks[index] = average
        start = stop
    return ranks


def spearman(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rx, ry = _rankdata(list(xs)), _rankdata(list(ys))
    mx, my = statistics.mean(rx), statistics.mean(ry)
    numerator = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denom_x = math.sqrt(sum((x - mx) ** 2 for x in rx))
    denom_y = math.sqrt(sum((y - my) ** 2 for y in ry))
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


def _median(values):
    return float(statistics.median(values)) if values else None


def _ols_residuals(xs, ys):
    if len(xs) != len(ys) or not xs:
        return []
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var_x if var_x else 0.0
    intercept = mean_y - slope * mean_x
    return [float(y - (intercept + slope * x)) for x, y in zip(xs, ys)]


def _head_name(head):
    return f"L{int(head[0]):02d}H{int(head[1]):02d}"


def _rank_heads(scores, heads):
    return sorted(heads, key=lambda head: (-scores[head], head[0], head[1]))


def _condition_head(row):
    condition = str(row.get("condition", ""))
    if _unpushed_condition_head(condition) is not None:
        return None
    recorded = row.get("head")
    if isinstance(recorded, dict):
        head = (int(recorded["layer"]), int(recorded["head"]))
        scale = float(row.get("scale", 0.0))
        return (head, condition) if scale == 0.0 else None
    match = re.fullmatch(r"head-L(\d+)-H(\d+)(?:-scale-([0-9.eE+-]+))?", condition)
    if match and float(match.group(3) or 0.0) == 0.0:
        return (int(match.group(1)), int(match.group(2))), condition
    return None


def _unpushed_condition_head(condition):
    match = re.fullmatch(r"unpushed-head-L(\d+)-H(\d+)", str(condition))
    return (int(match.group(1)), int(match.group(2))) if match else None


def _reader_v_value(labels_by_key):
    expectations = {
        "S0": {"self_claim": True}, "S1": {"self_claim": True},
        "A0": {"shows_persona": True, "self_claim": False},
        "A1": {"shows_persona": True, "self_claim": False},
        "C0": {"shows_persona": True, "self_claim": False},
        "C1": {"shows_persona": True, "self_claim": False},
        "B0": {"shows_persona": False}, "B1": {"shows_persona": False},
    }
    values = {}
    for key, fields in expectations.items():
        row = labels_by_key.get(key)
        labels = _part92_label_values(row)
        checks = {name: labels[name] for name in fields}
        expected = list(fields.values())
        actual = [checks[name] for name in fields]
        values[key] = {"labels": checks, "passed": _fields_match(actual, expected)}
    return _fields_match([row["passed"] for row in values.values()], [True] * len(values)), values


def _head_from_stage_c(row):
    raw = row.get("head", row.get("head_name", row.get("head_id")))
    if isinstance(raw, dict):
        return (int(raw["layer"]), int(raw["head"]))
    match = re.fullmatch(r"L(\d+)H(\d+)", str(raw or ""), re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"head-L(\d+)-H(\d+)", str(row.get("condition", "")), re.IGNORECASE)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _stage_c_category(row):
    raw = row.get("category", row.get("label_category"))
    if raw is None and isinstance(row.get("label"), dict):
        raw = row["label"].get("category")
    aliases = {"as": "as", "speaks_as": "as", "mentions_not_as": "mentions_not_as",
               "mentions": "mentions_not_as", "no_mention": "no_mention", "none": "no_mention",
               "incoherent": "incoherent"}
    value = str(raw or "").strip().casefold()
    if value not in aliases:
        return None
    return aliases[value]


def _stage_c_off(row):
    if row.get("head_off") is True:
        return True
    condition = str(row.get("condition", "")).casefold().replace("_", "-")
    if condition in ("none", "no-head-off", "baseline", "no-head"):
        return False
    return condition in ("head-off", "off", "ablated") or condition.startswith("head-")


def _stage_c_prompt(row):
    value = str(row.get("prompt_key", row.get("question_key", row.get("prompt", row.get("question", ""))))).casefold()
    if "who are you" in value or value in ("who", "identity", "self"):
        return "who"
    if "perfect day" in value or value in ("content", "describe"):
        return "content"
    if "capital of france" in value or value in ("capital", "off_target"):
        return "off_target"
    return value or "unknown"


def _four_way_top(heads, dself, dcont, dself_held, dcont_held, g_values, top_k):
    residuals = {}
    for name, values in (("dself", dself), ("dcont", dcont),
                         ("dself_heldout", dself_held), ("dcont_heldout", dcont_held)):
        adjusted = _ols_residuals(g_values, [values[head] for head in heads])
        residuals[name] = {head: value for head, value in zip(heads, adjusted)}
    top = {name: set(_rank_heads(values, heads)[:top_k]) for name, values in residuals.items()}
    intersection = set.intersection(*top.values()) if top else set()
    return len(intersection), intersection, residuals


def layer_stratified_null(heads, dself, dcont, dself_held, dcont_held, g_by_head, *,
                          top_k=TOP_K, draws=SHUFFLES, seed=SHUFFLE_SEED, within_layer=True):
    """Permute selection/held-out content pairs while preserving each head's layer."""
    heads = list(heads)
    g_values = [g_by_head[head] for head in heads]
    observed, observed_heads, residuals = _four_way_top(heads, dself, dcont, dself_held,
                                                        dcont_held, g_values, top_k)
    groups = {}
    for head in heads:
        groups.setdefault(head[0] if within_layer else None, []).append(head)
    rng = random.Random(int(seed))
    exceedances = 0
    for _ in range(int(draws)):
        permuted_cont = dict(dcont)
        permuted_held = dict(dcont_held)
        for group_heads in groups.values():
            pairs = [(dcont[head], dcont_held[head]) for head in group_heads]
            rng.shuffle(pairs)
            for head, pair in zip(group_heads, pairs):
                permuted_cont[head], permuted_held[head] = pair
        value, _intersection, _ = _four_way_top(heads, dself, permuted_cont, dself_held,
                                                permuted_held, g_values, top_k)
        exceedances += int(value >= observed)
    return {"statistic": observed, "observed_heads": [_head_name(head) for head in sorted(observed_heads)],
            "draws": int(draws), "seed": int(seed), "exceedances": exceedances,
            "p_value": (1 + exceedances) / (1 + int(draws)) if draws else None,
            "permutation": "within each layer" if within_layer else "global",
            "top_k": int(top_k), "residual_rank_metrics": ["dself", "dcont", "dself'", "dcont'"]}


def _make_stage_c_plan(heads, both, self_only, content_only, rankings, seed=180):
    def by_rank(items, metric):
        order = rankings.get(metric, {}).get("g_residual", [])
        ranks = {name: index for index, name in enumerate(order)}
        return sorted(items, key=lambda head: (ranks.get(_head_name(head), len(order)), head))

    both_ordered = sorted(both, key=lambda head: (
        sum(rankings.get(metric, {}).get("g_residual_rank", {}).get(_head_name(head), 10**9)
            for metric in METRICS), head))[:5]
    self_ordered = by_rank(self_only, "dself")[:5]
    content_ordered = by_rank(content_only, "dcont")[:5]
    named = set(both_ordered) | set(self_ordered) | set(content_ordered)
    layers = sorted({head[0] for head in named})
    available_by_layer = {layer: [head for head in heads if head[0] == layer and head not in named]
                          for layer in layers}
    rng = random.Random(int(seed))
    random_heads = []
    while layers and len(random_heads) < 5:
        progressed = False
        for layer in layers:
            choices = [head for head in available_by_layer[layer] if head not in random_heads]
            if choices and len(random_heads) < 5:
                random_heads.append(rng.choice(choices))
                progressed = True
        if not progressed:
            break
    return {"seed": int(seed), "conditions": [
        {"name": "both", "heads": [_head_name(head) for head in both_ordered]},
        {"name": "self_only", "heads": [_head_name(head) for head in self_ordered]},
        {"name": "content_only", "heads": [_head_name(head) for head in content_ordered]},
        {"name": "random_layer_matched", "heads": [_head_name(head) for head in random_heads],
         "layers": [head[0] for head in random_heads]},
        {"name": "no_head_off", "heads": []},
    ]}


def _transition_table(stage_c_rows, stage_c_heads):
    categories = ("as", "mentions_not_as", "no_mention", "incoherent")
    grouped = {}
    for row in stage_c_rows:
        neuron = str(row.get("neuron", row.get("id", "")))
        prompt = _stage_c_prompt(row)
        category = _stage_c_category(row)
        if not neuron or category is None or "error" in row:
            continue
        head = _head_from_stage_c(row) if _stage_c_off(row) else None
        grouped.setdefault((head, neuron, prompt), category)
    reports = {}
    for head in stage_c_heads:
        head_name = _head_name(head)
        matrices, pairs = {}, []
        for (off_head, neuron, prompt), off_category in sorted(grouped.items(), key=lambda item: str(item[0])):
            if off_head != head:
                continue
            baseline = grouped.get((None, neuron, prompt))
            if baseline is None:
                continue
            matrix = matrices.setdefault(prompt, {before: {after: 0 for after in categories} for before in categories})
            matrix[baseline][off_category] += 1
            pairs.append({"neuron": neuron, "prompt": prompt, "no_head_off": baseline,
                          "head_off": off_category})
        who_lost_claims = sum(pair["prompt"] == "who" and pair["no_head_off"] == "as" and
                              pair["head_off"] == "mentions_not_as" for pair in pairs)
        off_who = [pair["head_off"] for pair in pairs if pair["prompt"] == "who"]
        off_mentions = sum(category in ("as", "mentions_not_as") for category in off_who)
        off_as = sum(category == "as" for category in off_who)
        reports[head_name] = {"transition_counts": matrices, "paired_answers": pairs,
                              "who_as_to_mentions_not_as": who_lost_claims,
                              "conditional_share_as_among_mentions_description":
                                  (off_as / off_mentions if off_mentions else None),
                              "assessable": bool(pairs)}
    return reports


def _stage_c_behavior(stage_c_rows, plan):
    grouped = {}
    for row in stage_c_rows:
        category = _stage_c_category(row)
        neuron = str(row.get("neuron", row.get("id", "")))
        if not neuron or category is None or "error" in row:
            continue
        head = _head_from_stage_c(row) if _stage_c_off(row) else None
        grouped[(head, neuron, _stage_c_prompt(row))] = category
    per_head = {}
    named_conditions = {item["name"]: item["heads"] for item in plan["conditions"]}
    for name in (head for group in named_conditions.values() for head in group):
        match = re.fullmatch(r"L(\d+)H(\d+)", name)
        head = (int(match.group(1)), int(match.group(2)))
        who_pairs, content_pairs, off_target = [], [], []
        for (off_head, neuron, prompt), after in grouped.items():
            if off_head != head:
                continue
            before = grouped.get((None, neuron, prompt))
            if before is None:
                continue
            pair = (before, after)
            if prompt == "who":
                who_pairs.append(pair)
            elif prompt == "content":
                content_pairs.append(pair)
            elif prompt == "off_target":
                off_target.append(after)
        def reduction(pairs, predicate):
            if not pairs:
                return None
            return statistics.mean(float(predicate(before)) - float(predicate(after)) for before, after in pairs)
        per_head[name] = {
            "speaks_as_reduction_who": reduction(who_pairs, lambda label: label == "as"),
            "mentions_reduction_content": reduction(content_pairs,
                lambda label: label in ("as", "mentions_not_as")),
            "off_target_broken_count": sum(label == "incoherent" for label in off_target),
            "who_pairs": len(who_pairs), "content_pairs": len(content_pairs),
            "off_target_answers": len(off_target),
        }
    random_names = named_conditions.get("random_layer_matched", [])
    random_off_target = [per_head[name]["off_target_broken_count"] for name in random_names if name in per_head]
    random_broken_median = _median(random_off_target)
    both_values = [per_head[name] for name in named_conditions.get("both", []) if name in per_head]
    self_values = [per_head[name] for name in named_conditions.get("self_only", []) if name in per_head]
    content_values = [per_head[name] for name in named_conditions.get("content_only", []) if name in per_head]
    def median_field(values, field):
        items = [row[field] for row in values if row[field] is not None]
        return _median(items)
    both_as, both_mentions = median_field(both_values, "speaks_as_reduction_who"), median_field(both_values, "mentions_reduction_content")
    self_as, self_mentions = median_field(self_values, "speaks_as_reduction_who"), median_field(self_values, "mentions_reduction_content")
    content_as, content_mentions = median_field(content_values, "speaks_as_reduction_who"), median_field(content_values, "mentions_reduction_content")
    checks = {
        "both_lowers_speaks_as_and_mentions": (both_as is not None and both_mentions is not None and both_as > 0 and both_mentions > 0),
        "self_only_lowers_speaks_as_only": (self_as is not None and self_mentions is not None and self_as > 0 and self_mentions <= 0),
        "content_only_lowers_mentions_only": (content_as is not None and content_mentions is not None and content_as <= 0 and content_mentions > 0),
    }
    all_assessable = all(value is not None for value in
                         (both_as, both_mentions, self_as, self_mentions, content_as, content_mentions))
    random_values = [per_head[name] for name in random_names if name in per_head]
    random_as_median = median_field(random_values, "speaks_as_reduction_who")
    random_mentions_median = median_field(random_values, "mentions_reduction_content")
    both_prediction = {}
    for name in named_conditions.get("both", []):
        values = per_head.get(name, {})
        as_reduction = values.get("speaks_as_reduction_who")
        mentions_reduction = values.get("mentions_reduction_content")
        assessable = all(value is not None for value in
                         (as_reduction, mentions_reduction, random_as_median, random_mentions_median))
        both_prediction[name] = {"speaks_as_reduction": as_reduction,
                                 "random_speaks_as_median": random_as_median,
                                 "speaks_as_reduction_over_random": (as_reduction - random_as_median
                                                                      if assessable else None),
                                 "mentions_reduction": mentions_reduction,
                                 "random_mentions_median": random_mentions_median,
                                 "mentions_reduction_over_random": (mentions_reduction - random_mentions_median
                                                                     if assessable else None),
                                 "threshold": 0.2, "assessable": assessable,
                                 "passed": bool(assessable and as_reduction - random_as_median >= 0.2 and
                                                mentions_reduction - random_mentions_median >= 0.2)}
    prediction_180_2 = any(row["passed"] for row in both_prediction.values())
    return {"assessable": bool(stage_c_rows and all_assessable),
            "per_head": per_head,
            "group_median_reductions": {"both_speaks_as_who": both_as, "both_mentions_content": both_mentions,
                                        "self_only_speaks_as_who": self_as, "self_only_mentions_content": self_mentions,
                                        "content_only_speaks_as_who": content_as,
                                        "content_only_mentions_content": content_mentions},
            "directional_checks": checks,
            "passed": all(checks.values()) if all_assessable else None,
            "180_2_against_random_by_both_head": both_prediction,
            "prediction_180_2_full_passed": prediction_180_2,
            "prediction_180_2_status": "passed" if prediction_180_2 else
            "not assessed or failed; requires >=0.20 reduction on both rates over random-head medians",
            "random_head_off_target_broken_median": random_broken_median,
            "off_target_by_head": {name: {"broken_count": values["off_target_broken_count"],
                                          "random_median": random_broken_median,
                                          "passed": (values["off_target_broken_count"] <= random_broken_median
                                                     if values["off_target_answers"] > 0 and
                                                        random_broken_median is not None else None),
                                          "assessable": values["off_target_answers"] > 0 and
                                                        random_broken_median is not None}
                                   for name, values in per_head.items()},
            "status": "assessed" if all_assessable else "not assessed: missing paired Stage C labels"}


def classify_records(text_rows, score_rows, neurons, labels, reader_rows, *,
                     top_k: int = TOP_K, shuffle_count: int = SHUFFLES, shuffle_seed: int = SHUFFLE_SEED,
                     gate2_threshold: int | None = 10, stage_c_rows=None):
    """Compute v4 inclusion, stage A metrics, stage B sets, and available stage C checks."""
    stage_c_rows = list(stage_c_rows or [])
    text_ids = {str(row["neuron"]) for row in text_rows}
    neuron_by_id = {str(row["id"]): row for row in neurons if str(row["id"]) in text_ids}
    if text_ids - set(neuron_by_id):
        raise ValueError(f"text rows have unknown neuron ids: {sorted(text_ids - set(neuron_by_id))}")
    if not neuron_by_id:
        raise ValueError("no neurons are represented in texts.jsonl")

    label_by_question = {}
    for row in labels:
        if row.get("unit") == "neuron" and "error" not in row:
            try:
                question_index = int(row.get("question_index", -1))
            except (TypeError, ValueError):
                continue
            label_by_question[(str(row.get("id")), question_index)] = row
    reader_by_id_key = {}
    for row in reader_rows:
        neuron_id = row.get("id", row.get("neuron", row.get("neuron_id")))
        key = _reader_key(row)
        if neuron_id is not None and key is not None:
            reader_key = (str(neuron_id), key)
            if reader_key in reader_by_id_key:
                raise ValueError(f"duplicate reader label row {reader_key}")
            reader_by_id_key[reader_key] = row

    score_index, condition_heads, unpushed_condition_heads = {}, {}, {}
    for row in score_rows:
        condition = str(row.get("condition"))
        key = (condition, str(row.get("neuron")), str(row.get("case")))
        if key in score_index:
            raise ValueError(f"duplicate score record {key}")
        score_index[key] = row
        parsed = _condition_head(row)
        if parsed:
            head, condition_name = parsed
            if condition_name in condition_heads:
                if condition_heads[condition_name] != head:
                    raise ValueError(f"condition {condition_name} has conflicting head metadata")
            elif head in condition_heads.values():
                raise ValueError(f"multiple zero-dose conditions describe head {head}")
            condition_heads[condition_name] = head
        unpushed_head = _unpushed_condition_head(condition)
        if unpushed_head is not None:
            if condition in unpushed_condition_heads and unpushed_condition_heads[condition] != unpushed_head:
                raise ValueError(f"condition {condition} has conflicting unpushed-head metadata")
            unpushed_condition_heads[condition] = unpushed_head
    heads = sorted(condition_heads.values())
    if not heads:
        raise ValueError("no zero-dose single-head ablation conditions found in scores.jsonl")

    def score(condition, neuron_id, case, field="lp_k"):
        key = (condition, str(neuron_id), case)
        row = score_index.get(key)
        if row is None:
            raise ValueError(f"missing score row {key}")
        value = row.get(field)
        if value is None:
            raise ValueError(f"score {field} is empty for {key}")
        return float(value)

    margins, m0_margins, criteria = {}, {}, {}
    for neuron_id, neuron in sorted(neuron_by_id.items()):
        q0 = _part92_label_values(label_by_question.get((neuron_id, 0)))
        q3 = _part92_label_values(label_by_question.get((neuron_id, 3)))
        s1 = _part92_label_values(reader_by_id_key.get((neuron_id, "S1")))
        c1 = _part92_label_values(reader_by_id_key.get((neuron_id, "C1")))
        self0 = score("none", neuron_id, "self0_S0") - score("none", neuron_id, "self0_A0")
        cont0 = score("none", neuron_id, "cont0_C0") - score("none", neuron_id, "cont0_B0")
        self1 = score("none", neuron_id, "self1_S1") - score("none", neuron_id, "self1_A1")
        cont1 = score("none", neuron_id, "cont1_C1") - score("none", neuron_id, "cont1_B1")
        m0_self = score("M0", neuron_id, "self0_S0") - score("M0", neuron_id, "self0_A0")
        m0_cont = score("M0", neuron_id, "cont0_C0") - score("M0", neuron_id, "cont0_B0")
        m0_self1 = score("M0", neuron_id, "self1_S1") - score("M0", neuron_id, "self1_A1")
        m0_cont1 = score("M0", neuron_id, "cont1_C1") - score("M0", neuron_id, "cont1_B1")
        m0_doc = score("M0", neuron_id, "doc1_D1") - score("M0", neuron_id, "doc1_B1d")
        m0_doc2 = score("M0", neuron_id, "doc2_D2") - score("M0", neuron_id, "doc2_B2d")
        m0_selfx = score("M0", neuron_id, "cross_S0") - score("M0", neuron_id, "cross_D1")
        m0_frame = (score("M0", neuron_id, "cross_D1", "lp_full") -
                    score("M0", neuron_id, "doc1_D1", "lp_full"))
        doc1 = score("none", neuron_id, "doc1_D1") - score("none", neuron_id, "doc1_B1d")
        doc2 = score("none", neuron_id, "doc2_D2") - score("none", neuron_id, "doc2_B2d")
        selfx = score("none", neuron_id, "cross_S0") - score("none", neuron_id, "cross_D1")
        frame_penalty = (score("none", neuron_id, "cross_D1", "lp_full") -
                         score("none", neuron_id, "doc1_D1", "lp_full"))
        margins[neuron_id] = {"self": self0, "cont": cont0, "self_heldout": self1,
                              "cont_heldout": cont1, "doc": doc1, "doc2": doc2,
                              "selfX": selfx, "frame_penalty": frame_penalty}
        m0_margins[neuron_id] = {"self": m0_self, "cont": m0_cont,
                                 "self_heldout": m0_self1, "cont_heldout": m0_cont1,
                                 "doc": m0_doc, "doc2": m0_doc2, "selfX": m0_selfx,
                                 "frame_penalty": m0_frame}
        criterion_i = q0["self_claim"]
        criterion_ii = _fields_match((q3["shows_persona"], q3["self_claim"]), (True, False))
        criterion_iii = _fields_match((s1["self_claim"], c1["shows_persona"], c1["self_claim"]),
                                      (True, True, False))
        criterion_iv = all(value > 0 for value in (self0, cont0, self1, cont1))
        prefix_labels = {key: _part92_label_values(reader_by_id_key.get((neuron_id, key)))
                         for key in ("S0", "S1", "A0", "A1", "C0", "C1", "B0", "B1")}
        criterion_v, criterion_v_details = _reader_v_value(prefix_labels)
        criterion_values = {"i": criterion_i, "ii": criterion_ii, "iii": criterion_iii,
                            "iv": criterion_iv, "v": criterion_v}
        criteria[neuron_id] = {
            "i_q0_self_claim": criterion_i,
            "ii_q3_shows_without_claim": criterion_ii,
            "ii_q3_labels": q3,
            "iii_reader_S1C1": criterion_iii,
            "iii_reader_labels": {"S1": s1, "C1": c1},
            "iv_all_four_positive_margins": criterion_iv,
            "v_all_scored_prefix_contrasts": criterion_v,
            "v_prefix_labels": criterion_v_details,
            "included": all(value is True for value in criterion_values.values()),
            "dropped_by": [name for name, value in criterion_values.items() if value is not True],
        }

    included = sorted(neuron_id for neuron_id, row in criteria.items() if row["included"])
    criterion_fields = {"i": "i_q0_self_claim", "ii": "ii_q3_shows_without_claim",
                        "iii": "iii_reader_S1C1", "iv": "iv_all_four_positive_margins",
                        "v": "v_all_scored_prefix_contrasts"}
    criterion_counts, dropped_by_criterion = {}, {}
    for criterion, field in criterion_fields.items():
        values = {neuron_id: criteria[neuron_id][field] for neuron_id in criteria}
        criterion_counts[criterion] = {"passed": sum(value is True for value in values.values()),
                                       "failed": sum(value is False for value in values.values()),
                                       "unknown": sum(value is None for value in values.values()),
                                       "n": len(values)}
        dropped_by_criterion[criterion] = [neuron_id for neuron_id, value in values.items() if value is not True]

    gate2_passed = gate2_threshold is None or len(included) >= int(gate2_threshold)
    top_count = min(max(1, int(top_k)), len(heads))
    gate2_report = {"passed": gate2_passed if gate2_threshold is not None else None,
                    "assessable": gate2_threshold is not None,
                    "stop_and_redesign": bool(gate2_threshold is not None and not gate2_passed),
                    "threshold": gate2_threshold,
                    "thresholds": DESIGN_THRESHOLDS["gate2"]}
    inclusion = {"criteria": criteria, "criterion_counts": criterion_counts,
                 "margins_no_head_off": margins, "dropped_by_criterion": dropped_by_criterion,
                 "gate2": gate2_report}

    norm_by_neuron, norm_measure = {}, {}
    for measure, pushed_key, norm_cases in (
            ("self", "self", ("self0_S0", "self0_A0")),
            ("content", "cont", ("cont0_C0", "cont0_B0"))):
        assessable_ids, non_assessable_ids, ratios = [], [], {}
        for neuron_id in included:
            pushed_increment = margins[neuron_id][pushed_key] - m0_margins[neuron_id][pushed_key]
            if pushed_increment < 0.05:
                non_assessable_ids.append(neuron_id)
                continue
            norm_increment = ((score("norm", neuron_id, norm_cases[0]) - score("norm", neuron_id, norm_cases[1])) -
                              m0_margins[neuron_id][pushed_key])
            ratio = norm_increment / pushed_increment
            assessable_ids.append(neuron_id)
            ratios[neuron_id] = {"push_increment": pushed_increment, "norm_increment": norm_increment,
                                 "norm_to_push_ratio": ratio, "mostly_norm": ratio >= 0.5}
        mostly_count = sum(bool(item["mostly_norm"]) for item in ratios.values())
        mostly_threshold = math.ceil(len(assessable_ids) / 2) if assessable_ids else None
        enough_assessable = len(assessable_ids) >= math.ceil(len(included) / 2) if included else False
        norm_measure[measure] = {"assessable_count": len(assessable_ids),
                                 "non_assessable_count": len(non_assessable_ids),
                                 "non_assessable_neurons": non_assessable_ids,
                                 "assessable_neurons": assessable_ids, "per_neuron": ratios,
                                 "mostly_norm_count": mostly_count,
                                 "mostly_norm_threshold": mostly_threshold,
                                 "mostly_norm": bool(mostly_threshold is not None and mostly_count >= mostly_threshold),
                                 "at_least_half_included_assessable": enough_assessable}
    norm_blocks_both = any(norm_measure[measure]["mostly_norm"] or
                           not norm_measure[measure]["at_least_half_included_assessable"]
                           for measure in ("self", "content"))
    norm_report = {"measures": norm_measure, "included_neurons": len(included),
                   "blocks_both_head_claim": norm_blocks_both,
                   "rule": DESIGN_THRESHOLDS["norm_control"]}
    doc_values = {neuron_id: {key: margins[neuron_id][key] for key in ("doc", "doc2", "selfX", "frame_penalty")}
                  for neuron_id in included}
    document_report = {"per_neuron": doc_values,
                       "medians": {key: _median([item[key] for item in doc_values.values()])
                                   for key in ("doc", "doc2", "selfX", "frame_penalty")},
                       "used_for_inclusion": False}
    m0_document_values = {neuron_id: {key: m0_margins[neuron_id][key]
                                     for key in ("doc", "doc2", "selfX", "frame_penalty")}
                          for neuron_id in neuron_by_id}
    m0_document_report = {"per_neuron": m0_document_values,
                          "medians": {key: _median([item[key] for item in m0_document_values.values()])
                                      for key in ("doc", "doc2", "selfX", "frame_penalty")},
                          "condition": "M0", "push": False, "head_off": False}

    if (gate2_threshold is not None and not gate2_passed) or not included:
        head_names = [_head_name(head) for head in heads]
        return {"part": 180, "n_text_neurons": len(neuron_by_id), "n_included": len(included),
                "included_neurons": included, "inclusion": inclusion,
                "norm_control": norm_report, "document_frame_report": document_report,
                "M0_document_frame_report": m0_document_report,
                "n_heads": len(heads), "head_medians": {name: None for name in head_names},
                "generic_damage_g": {}, "g_residual_medians": {}, "ranks": {},
                "per_neuron_head_effects": {},
                "reliability": {"baseline_stability_self": None,
                                "baseline_stability_content": None,
                                "effect_reliability_self": None, "effect_reliability_content": None,
                                "low_reliability_threshold": 0.3,
                                "interpretation": "not assessable; Gate 2 stopped the analysis"},
                "gate3_hook_placement": {"verdict": "see hookcheck180.py output",
                                          "thresholds": DESIGN_THRESHOLDS["gate3_hook_placement"]},
                "shuffle_null": {"statistic": None, "observed_heads": [],
                                 "top_k": top_count, "draws": int(shuffle_count),
                                 "seed": int(shuffle_seed), "exceedances": None, "p_value": None,
                                 "permutation": "within each layer", "status": "not assessed; Gate 2 stopped analysis",
                                 "thresholds": DESIGN_THRESHOLDS["shuffle"]},
                "stage_b_candidates": {"top_k": top_count, "selection_self_top": [],
                                        "selection_content_top": [], "heldout_self_top": [],
                                        "heldout_content_top": [], "self_head_candidates": [],
                                        "content_head_candidates": [], "unfiltered_both_head_candidates": [],
                                        "both_head_candidates": [],
                                        "both_head_status": ("fewer than 10 included neurons; Gate 2 stops and redesign is required"
                                                             if gate2_threshold is not None and not gate2_passed else
                                                             "not assessable; no neurons pass inclusion"),
                                        "thresholds": {"self_content": DESIGN_THRESHOLDS["stage_b_self_content"],
                                                       "both": DESIGN_THRESHOLDS["stage_b_both"]}},
                "stage_a_predictions": {key: {"verdict": "not assessable", "thresholds": DESIGN_THRESHOLDS[key]}
                                        for key in ("180.1", "180.3", "W1", "W2", "W3")},
                "stage_c_plan": _make_stage_c_plan(heads, [], [], [], {}),
                "stage_c_labels": {"assessable": False, "reason": "stage C labels were not supplied"},
                "stage_c_required_for_headline": True, "design_thresholds": DESIGN_THRESHOLDS}

    condition_by_head = {head: condition for condition, head in condition_heads.items()}
    unpushed_by_head = {head: condition for condition, head in unpushed_condition_heads.items()}
    dself, dcont, dself_held, dcont_held = {}, {}, {}, {}
    g_by_head, medians, per_neuron_head = {}, {}, {}
    d0_effects = {}
    for head in heads:
        condition = condition_by_head[head]
        g_by_head[head] = score("M0", "unpushed", "g0", "lp_full") - score(condition, "unpushed", "g0", "lp_full")
        dself[head], dcont[head], dself_held[head], dcont_held[head] = {}, {}, {}, {}
        per_neuron_head[_head_name(head)] = {}
        for neuron_id in included:
            off_margins = {
                "self": score(condition, neuron_id, "self0_S0") - score(condition, neuron_id, "self0_A0"),
                "cont": score(condition, neuron_id, "cont0_C0") - score(condition, neuron_id, "cont0_B0"),
                "self_heldout": score(condition, neuron_id, "self1_S1") - score(condition, neuron_id, "self1_A1"),
                "cont_heldout": score(condition, neuron_id, "cont1_C1") - score(condition, neuron_id, "cont1_B1"),
            }
            dself[head][neuron_id] = margins[neuron_id]["self"] - off_margins["self"]
            dcont[head][neuron_id] = margins[neuron_id]["cont"] - off_margins["cont"]
            dself_held[head][neuron_id] = margins[neuron_id]["self_heldout"] - off_margins["self_heldout"]
            dcont_held[head][neuron_id] = margins[neuron_id]["cont_heldout"] - off_margins["cont_heldout"]
            per_neuron_head[_head_name(head)][neuron_id] = {
                "dself": dself[head][neuron_id], "dcont": dcont[head][neuron_id],
                "dself_heldout": dself_held[head][neuron_id], "dcont_heldout": dcont_held[head][neuron_id]}
        medians[head] = {metric: _median(list(values[head].values()))
                         for metric, values in (("dself", dself), ("dcont", dcont),
                                                ("dself_heldout", dself_held), ("dcont_heldout", dcont_held))}
        unpushed_condition = unpushed_by_head.get(head)
        d0_effects[head] = {metric: {} for metric in ("dself", "dcont", "dself_heldout", "dcont_heldout")}
        if unpushed_condition:
            for neuron_id in included:
                cases = (("dself", "self0_S0", "self0_A0", "self"),
                         ("dcont", "cont0_C0", "cont0_B0", "cont"),
                         ("dself_heldout", "self1_S1", "self1_A1", "self_heldout"),
                         ("dcont_heldout", "cont1_C1", "cont1_B1", "cont_heldout"))
                for metric, left, right, margin_key in cases:
                    off_key_a = (unpushed_condition, neuron_id, left)
                    off_key_b = (unpushed_condition, neuron_id, right)
                    if off_key_a in score_index and off_key_b in score_index:
                        off_margin = score(unpushed_condition, neuron_id, left) - score(unpushed_condition, neuron_id, right)
                        d0_effects[head][metric][neuron_id] = m0_margins[neuron_id][margin_key] - off_margin
        for metric in d0_effects[head]:
            if len(d0_effects[head][metric]) == len(included):
                medians[head][f"d0{metric.removeprefix('d')}"] = _median(list(d0_effects[head][metric].values()))

    residual = {}
    g_values = [g_by_head[head] for head in heads]
    for metric in METRICS:
        residual_values = _ols_residuals(g_values, [medians[head][metric] for head in heads])
        residual[metric] = {head: value for head, value in zip(heads, residual_values)}
    rankings = {}
    for metric in METRICS:
        raw_rank = _rank_heads({head: medians[head][metric] for head in heads}, heads)
        residual_rank = _rank_heads(residual[metric], heads)
        rankings[metric] = {"raw": [_head_name(head) for head in raw_rank],
                            "g_residual": [_head_name(head) for head in residual_rank],
                            "raw_rank": {_head_name(head): index + 1 for index, head in enumerate(raw_rank)},
                            "g_residual_rank": {_head_name(head): index + 1 for index, head in enumerate(residual_rank)}}

    top_selection_self = _rank_heads(residual["dself"], heads)[:top_count]
    top_selection_cont = _rank_heads(residual["dcont"], heads)[:top_count]
    top_held_self = _rank_heads(residual["dself_heldout"], heads)[:top_count]
    top_held_cont = _rank_heads(residual["dcont_heldout"], heads)[:top_count]
    raw_self_positive = {head for head in heads if medians[head]["dself"] > 0 and medians[head]["dself_heldout"] > 0}
    raw_cont_positive = {head for head in heads if medians[head]["dcont"] > 0 and medians[head]["dcont_heldout"] > 0}
    raw_both_positive = raw_self_positive & raw_cont_positive
    effect_self_rel = spearman([medians[head]["dself"] for head in heads],
                               [medians[head]["dself_heldout"] for head in heads])
    effect_cont_rel = spearman([medians[head]["dcont"] for head in heads],
                               [medians[head]["dcont_heldout"] for head in heads])
    self_reliable = effect_self_rel is not None and effect_self_rel >= 0.3
    cont_reliable = effect_cont_rel is not None and effect_cont_rel >= 0.3
    self_head_set = (set(top_selection_self) & set(top_held_self) & raw_self_positive) if self_reliable else set()
    cont_head_set = (set(top_selection_cont) & set(top_held_cont) & raw_cont_positive) if cont_reliable else set()
    unfiltered_both = set(top_selection_self) & set(top_selection_cont) & set(top_held_self) & set(top_held_cont)
    both = unfiltered_both & raw_both_positive if self_reliable and cont_reliable else set()

    raw_self_rank = _rank_heads({head: medians[head]["dself"] for head in heads}, heads)
    raw_cont_rank = _rank_heads({head: medians[head]["dcont"] for head in heads}, heads)
    top10_overlap = len(set(raw_self_rank[:10]) & set(raw_cont_rank[:10]))
    corr = spearman([medians[head]["dself"] for head in heads], [medians[head]["dcont"] for head in heads])
    early_count = sum(head[0] in (0, 1, 2, 3) for head in raw_self_rank[:10])
    raw_top_k_self = set(raw_self_rank[:top_count])
    residual_top_k_self = set(_rank_heads(residual["dself"], heads)[:top_count])
    w2_stays = len(raw_top_k_self & residual_top_k_self)
    w3_intersection = set(_rank_heads(residual["dself_heldout"], heads)[:top_count]) & \
                       set(_rank_heads(residual["dcont_heldout"], heads)[:top_count])
    frame_flags = [margins[n]["frame_penalty"] > margins[n]["selfX"] for n in included]
    frame_count, frame_n = sum(frame_flags), len(included)
    w1_verdict = ("held" if frame_count >= math.ceil(frame_n / 2) else
                  "failed" if frame_count < frame_n / 4 else "inconclusive")

    # The former v3 Gate 3 is now descriptive only, as required by v4.
    above_below = {}
    for neuron_id in included:
        push_layer = int(neuron_by_id[neuron_id]["layer"])
        above = [abs(dcont[head][neuron_id]) for head in heads if head[0] > push_layer]
        at_or_below = [abs(dcont[head][neuron_id]) for head in heads if head[0] <= push_layer]
        above_below[neuron_id] = {"push_layer": push_layer,
                                  "mean_abs_dcont_above": statistics.mean(above) if above else None,
                                  "mean_abs_dcont_at_or_below": statistics.mean(at_or_below) if at_or_below else None}

    null = layer_stratified_null(heads,
        {head: medians[head]["dself"] for head in heads},
        {head: medians[head]["dcont"] for head in heads},
        {head: medians[head]["dself_heldout"] for head in heads},
        {head: medians[head]["dcont_heldout"] for head in heads}, g_by_head,
        top_k=top_count, draws=shuffle_count, seed=shuffle_seed, within_layer=True)
    baseline_self_rel = spearman([margins[n]["self"] for n in included],
                                 [margins[n]["self_heldout"] for n in included])
    baseline_cont_rel = spearman([margins[n]["cont"] for n in included],
                                 [margins[n]["cont_heldout"] for n in included])
    reliability = {"baseline_stability_self": baseline_self_rel,
                   "baseline_stability_content": baseline_cont_rel,
                   "effect_reliability_self": effect_self_rel,
                   "effect_reliability_content": effect_cont_rel,
                   "low_reliability_threshold": 0.3,
                   "nameability": {
                       "self": {"allowed": self_reliable, "reason": None if self_reliable else
                                "effect reliability for dself vs dself' is below 0.3 or undefined"},
                       "content": {"allowed": cont_reliable, "reason": None if cont_reliable else
                                   "effect reliability for dcont vs dcont' is below 0.3 or undefined"}},
                   "interpretation": "head names suppressed for measures below the 0.3 effect-reliability threshold"
                   if not (self_reliable and cont_reliable) else "both measures meet the effect-reliability threshold"}

    rankings_for_plan = rankings
    self_only = self_head_set - both
    content_only = cont_head_set - both
    stage_c_plan = _make_stage_c_plan(heads, both, self_only, content_only, rankings_for_plan, seed=180)
    stage_c_head_names = {name for category in stage_c_plan["conditions"] if category["name"] != "no_head_off"
                          for name in category["heads"]}
    stage_c_heads = [head for head in heads if _head_name(head) in stage_c_head_names]
    transitions = _transition_table(stage_c_rows, stage_c_heads)
    behavior = _stage_c_behavior(stage_c_rows, stage_c_plan)
    stage_c_labels = {"assessable": bool(stage_c_rows), "transition_tables": transitions,
                      "behavioral_dissociation_and_off_target": behavior,
                      "schema": "JSONL rows: neuron, head (LnnHnn or {layer,head}), condition (no-head-off or head-off), prompt_key, category"}
    transition_lost = {name: data["who_as_to_mentions_not_as"] for name, data in transitions.items()}
    random_names = next((item["heads"] for item in stage_c_plan["conditions"]
                         if item["name"] == "random_layer_matched"), [])
    random_lost = [transition_lost[name] for name in random_names if name in transition_lost]
    random_median = _median(random_lost)
    criterion6 = {name: {"count_as_to_mentions_not_as": count,
                         "random_head_median": random_median,
                         "passed": count >= random_median + 3 if random_median is not None else None,
                         "status": "assessed" if name in transitions and transitions[name]["assessable"] and
                                                   random_median is not None else "not assessed"}
                  for name, count in transition_lost.items()}

    # Push-dependent contrasts are available only after --unpushed-heads has scored the named heads.
    push_dependent = {}
    for head in sorted(set(both) | set(self_head_set) | set(cont_head_set)):
        values = d0_effects.get(head, {})
        complete = all(len(values.get(metric, {})) == len(included)
                       for metric in ("dself", "dcont", "dself_heldout", "dcont_heldout"))
        deltas = {metric: _median([({"dself": dself, "dcont": dcont,
                                     "dself_heldout": dself_held, "dcont_heldout": dcont_held}[metric][head][n] -
                                    values[metric][n]) for n in included])
                  for metric in ("dself", "dcont", "dself_heldout", "dcont_heldout")} if complete else None
        push_dependent[_head_name(head)] = {
            "assessable": complete,
            "median_dself_minus_d0self": deltas["dself"] if deltas else None,
            "median_dcont_minus_d0cont": deltas["dcont"] if deltas else None,
            "median_dself_prime_minus_d0self_prime": deltas["dself_heldout"] if deltas else None,
            "median_dcont_prime_minus_d0cont_prime": deltas["dcont_heldout"] if deltas else None,
            "criterion_1_passed": bool(deltas and deltas["dself"] > 0 and deltas["dcont"] > 0),
            "criterion_1_status": "assessed" if complete else "not assessed: unpushed-head rows are missing"}

    set_level_passed = bool(len(both) >= 3 and null["p_value"] is not None and null["p_value"] < 0.05)
    single_head_alternative = {}
    for head in sorted(set(both) | set(self_head_set) | set(cont_head_set)):
        name = _head_name(head)
        unpushed = push_dependent.get(name, {})
        p_inc_self = _median([margins[n]["self"] - m0_margins[n]["self"] for n in included])
        p_inc_cont = _median([margins[n]["cont"] - m0_margins[n]["cont"] for n in included])
        dep_self, dep_cont = unpushed.get("median_dself_minus_d0self"), unpushed.get("median_dcont_minus_d0cont")
        assessable = unpushed.get("assessable", False)
        single_head_alternative[name] = {"push_increment_self_median": p_inc_self,
                                         "push_increment_content_median": p_inc_cont,
                                         "push_dependent_self_median": dep_self,
                                         "push_dependent_content_median": dep_cont,
                                         "assessable": assessable,
                                         "passed": bool(assessable and p_inc_self is not None and p_inc_cont is not None and
                                                         dep_self is not None and dep_cont is not None and
                                                         p_inc_self > 0 and p_inc_cont > 0 and
                                                         dep_self >= 0.5 * p_inc_self and dep_cont >= 0.5 * p_inc_cont)}
    single_head_pass = any(row["passed"] for row in single_head_alternative.values())
    criterion5 = {"set_level_passed": set_level_passed, "four_way_intersection_count": null["statistic"],
                  "positive_raw_effect_both_head_count": len(both),
                  "stratified_null_p": null["p_value"], "single_head_alternative": single_head_alternative,
                  "single_head_alternative_passed": single_head_pass,
                  "passed": set_level_passed or single_head_pass,
                  "status": "assessed" if null["p_value"] is not None else "not assessed"}

    candidate_claims = {}
    for head in sorted(both):
        name = _head_name(head)
        push_gate = push_dependent.get(name, {}).get("criterion_1_passed")
        transition_gate = criterion6.get(name, {}).get("passed")
        off_target_gate = behavior.get("off_target_by_head", {}).get(name, {}).get("passed")
        candidate_claims[name] = {
            "criterion_1_push_dependent": push_gate,
            "criterion_2_heldout_selection_bar": True,
            "criterion_3_behavioral_dissociation": behavior.get("passed"),
            "criterion_4_off_target_clean": off_target_gate,
            "criterion_5_set_or_single_head": criterion5["passed"],
            "criterion_6_claim_becomes_mention": transition_gate,
            "prediction_180_2_full": behavior.get("180_2_against_random_by_both_head", {}).get(name, {}).get("passed"),
            "norm_control_passed": not norm_blocks_both,
            "assessable": all(value is not None for value in
                               (push_gate, behavior.get("passed"), off_target_gate, criterion5["passed"],
                                transition_gate, behavior.get("180_2_against_random_by_both_head", {}).get(name, {}).get("passed"))),
        }
        candidate_claims[name]["passed"] = bool(not norm_blocks_both and all(
            candidate_claims[name][key] is True for key in (
                "criterion_1_push_dependent", "criterion_2_heldout_selection_bar",
                "criterion_3_behavioral_dissociation", "criterion_4_off_target_clean",
                "criterion_5_set_or_single_head", "criterion_6_claim_becomes_mention",
                "prediction_180_2_full")))
    claim_gates = {"per_both_candidate": candidate_claims,
                   "passed": any(row["passed"] for row in candidate_claims.values()),
                   "status": "passed" if any(row["passed"] for row in candidate_claims.values()) else
                   "not assessed or failed; no complete both-head claim"}

    predictions = {
        "180.1": {"spearman_median_dself_dcont": corr, "top10_overlap": top10_overlap,
                  "thresholds": {"spearman_below": 0.3, "top10_overlap_at_most": 3},
                  "verdict": "not assessable" if corr is None else
                  "held" if corr < 0.3 and top10_overlap <= 3 else "failed"},
        "180.3": {"top10_dself_in_layers_0_to_3": early_count, "threshold": 5,
                  "verdict": "held" if early_count >= 5 else "failed"},
        "180.W1": {"frame_penalty_exceeds_selfX": frame_count, "included_neurons": frame_n,
                   "fraction": frame_count / frame_n if frame_n else None, "verdict": w1_verdict},
        "180.W2": {"raw_top_k_retained_after_g_residual": w2_stays, "k": top_count,
                   "verdict": "held" if w2_stays <= 4 else "failed" if w2_stays >= 9 else "inconclusive"},
        "180.W3": {"heldout_top_k_both_intersection": [_head_name(head) for head in sorted(w3_intersection)],
                   "intersection_count": len(w3_intersection), "k": top_count,
                   "verdict": "held" if not w3_intersection else "failed"},
    }
    plan_candidates = {item["name"]: item["heads"] for item in stage_c_plan["conditions"]}
    both_status = ("norm-control rule blocks a both-head claim" if norm_blocks_both else
                   "candidate only; stage C and criteria 1, 3, 4, and 6 remain required")
    return {
        "part": 180, "n_text_neurons": len(neuron_by_id), "n_included": len(included),
        "included_neurons": included, "inclusion": inclusion,
        "norm_control": norm_report, "document_frame_report": document_report,
        "M0_document_frame_report": m0_document_report,
        "unpushed_M0_margins": {n: m0_margins[n] for n in neuron_by_id},
        "n_heads": len(heads),
        "head_medians": {_head_name(head): medians[head] for head in heads},
        "generic_damage_g": {_head_name(head): g_by_head[head] for head in heads},
        "g_residual_medians": {metric: {_head_name(head): residual[metric][head] for head in heads}
                               for metric in residual},
        "ranks": rankings, "per_neuron_head_effects": per_neuron_head,
        "reliability": reliability,
        "above_below_push_layer_description": {"per_neuron": above_below,
            "description": "v3 Gate 3 comparison retained as a descriptive field only; it is not a gate"},
        "shuffle_null": {**null, "thresholds": DESIGN_THRESHOLDS["shuffle"]},
        "stage_b_candidates": {
            "top_k": top_count,
            "selection_self_top": [_head_name(head) for head in top_selection_self],
            "selection_content_top": [_head_name(head) for head in top_selection_cont],
            "heldout_self_top": [_head_name(head) for head in top_held_self],
            "heldout_content_top": [_head_name(head) for head in top_held_cont],
            "self_head_candidates": [_head_name(head) for head in sorted(self_head_set)],
            "content_head_candidates": [_head_name(head) for head in sorted(cont_head_set)],
            "unfiltered_both_head_candidates": [_head_name(head) for head in sorted(unfiltered_both)],
            "both_head_candidates": [_head_name(head) for head in sorted(both)],
            "push_dependent_contrasts": push_dependent,
            "criterion_5": criterion5,
            "both_head_status": both_status,
            "thresholds": {"self_content": DESIGN_THRESHOLDS["stage_b_self_content"],
                           "both": DESIGN_THRESHOLDS["stage_b_both"]}},
        "stage_a_predictions": {key: {**value, "thresholds": DESIGN_THRESHOLDS[
            {"180.1": "180.1", "180.3": "180.3", "180.W1": "W1", "180.W2": "W2", "180.W3": "W3"}[key]]}
            for key, value in predictions.items()},
        "stage_c_plan": stage_c_plan,
        "stage_c_labels": stage_c_labels,
        "stage_c_criterion_6": criterion6,
        "claim_gates": claim_gates,
        "stage_c_required_for_headline": True,
        "design_thresholds": DESIGN_THRESHOLDS,
    }


def classify_two_sets(text_rows, score_rows, neurons, labels, reader_rows, score92b, *,
                      stage_c_rows=None, top_k: int = TOP_K, shuffle_count: int = SHUFFLES,
                      shuffle_seed: int = SHUFFLE_SEED):
    """Report the 18 and shared 8 neuron sets plus the v4 reading-rule gate."""
    shared_ids = deepseek_concept_ids(score92b)
    available = {str(row["neuron"]) for row in text_rows}
    missing = sorted(set(shared_ids) - available)
    if missing:
        raise ValueError(f"Part 92b shared concept ids are missing from texts.jsonl: {missing}")
    common = {"top_k": top_k, "shuffle_count": shuffle_count, "shuffle_seed": shuffle_seed,
              "stage_c_rows": stage_c_rows}
    shared = set(shared_ids)
    reports = {
        "haiku_reader_concept_18": classify_records(text_rows, score_rows, neurons, labels, reader_rows, **common),
        "both_readers_concept_8": classify_records(
            [row for row in text_rows if str(row["neuron"]) in shared],
            score_rows, neurons, labels, reader_rows, gate2_threshold=None, **common),
    }
    report18 = reports["haiku_reader_concept_18"]
    report8 = reports["both_readers_concept_8"]
    n8 = report8["n_included"]
    if n8 < 5:
        subset_status = "not assessable"
        subset_heads = []
    else:
        full_candidates = report18["stage_b_candidates"]["both_head_candidates"]
        top_self_8 = set(report8.get("ranks", {}).get("dself", {}).get("g_residual", [])[:26])
        top_cont_8 = set(report8.get("ranks", {}).get("dcont", {}).get("g_residual", [])[:26])
        subset_heads = sorted(set(full_candidates) & top_self_8 & top_cont_8)
        subset_status = "passed" if subset_heads else "failed"
    headline_flag = bool(subset_status == "passed" and report18.get("claim_gates", {}).get("passed") is True)
    if not headline_flag and subset_status == "not assessable":
        headline_reason = "fewer than five of the eight shared concept units pass inclusion"
    elif not headline_flag:
        headline_reason = "one or more both-head claim gates remain failed or not assessed"
    else:
        headline_reason = None
    return {"part": 180, "deepseek_concept_ids": shared_ids, "neuron_sets": reports,
            "reading_rule": {"shared_units_passing_inclusion": n8,
                             "required_minimum": 5, "eight_unit_status": subset_status,
                             "both_head_ids_top26_on_shared_set": subset_heads,
                             "headline_flag": headline_flag, "headline_reason": headline_reason},
            "stage_c_plans": {name: report["stage_c_plan"] for name, report in reports.items()}}


def classify_from_files(texts_path, scores_path, neurons_path, labels_path, reader_prefix_path, score92b_path, *,
                        stage_c_path=None, top_k: int = TOP_K, shuffle_count: int = SHUFFLES,
                        shuffle_seed: int = SHUFFLE_SEED):
    stage_c_file = normalise_path(stage_c_path) if stage_c_path else None
    return classify_two_sets(_load_jsonl(normalise_path(texts_path)),
                             _load_jsonl(normalise_path(scores_path)),
                             _load_json(normalise_path(neurons_path)),
                             _load_jsonl(normalise_path(labels_path)),
                             _load_jsonl(normalise_path(reader_prefix_path)),
                             _load_json(normalise_path(score92b_path)),
                             stage_c_rows=_load_jsonl(stage_c_file) if stage_c_file and stage_c_file.is_file() else [],
                             top_k=top_k, shuffle_count=shuffle_count, shuffle_seed=shuffle_seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", default=str(ROOT / "results" / "part180" / "texts.jsonl"))
    parser.add_argument("--scores", default=str(ROOT / "results" / "part180" / "scores.jsonl"))
    parser.add_argument("--reader-prefixes", "--reader", dest="reader_prefixes",
                        default=str(ROOT / "results" / "part180" / "reader-prefixes.jsonl"))
    parser.add_argument("--neurons", default=str(ROOT / "results" / "part92" / "step2" / "neurons.json"))
    parser.add_argument("--labels", default=str(ROOT / "results" / "part92" / "step2" / "labels-instruct_chat.jsonl"))
    parser.add_argument("--score92b", default=str(ROOT / "results" / "part92b" / "score92b.json"))
    parser.add_argument("--stage-c-labels", default=str(ROOT / "results" / "part180" / "stageC-labels.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "results" / "part180" / "SCORE-A.json"))
    parser.add_argument("--shuffle-count", type=int, default=SHUFFLES)
    parser.add_argument("--seed", type=int, default=SHUFFLE_SEED)
    args = parser.parse_args()
    reader_path = normalise_path(args.reader_prefixes)
    if not reader_path.is_file():
        raise FileNotFoundError(f"prefix reader file is required: {reader_path}")
    result = classify_from_files(args.texts, args.scores, args.neurons, args.labels, reader_path,
                                 args.score92b, stage_c_path=args.stage_c_labels,
                                 shuffle_count=args.shuffle_count, shuffle_seed=args.seed)
    reports = result["neuron_sets"]
    for name, report in reports.items():
        if report["n_heads"] != 256:
            raise ValueError(f"{name} requires all 256 head conditions; found {report['n_heads']}")
    out_path = normalise_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    plan_path = out_path.parent / "stageC-plan.json"
    with plan_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump({"part": 180,
                   "plan": result["stage_c_plans"]["haiku_reader_concept_18"]},
                  stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"sets": {name: {"included": report["n_included"],
                                      "both_candidates": report["stage_b_candidates"]["both_head_candidates"]}
                             for name, report in reports.items()},
                      "out": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
