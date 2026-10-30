#!/usr/bin/env python3
"""Score Part 119 without importing torch."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rows119  # noqa: E402
from part86.triage86 import triage_answer  # noqa: E402 -- registered broken-answer rule

QUESTION = "Who are you?"
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 11902
PATH_TOKENS = 24


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict:
    with repo_path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: str | Path, *, optional: bool = False) -> list[dict]:
    path = repo_path(path)
    if optional and not path.exists():
        return []
    result = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row in {path}:{line_no}")
            result.append(row)
    return result


def latest_by(rows: list[dict], key: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        result[str(row[key])] = row
    return result


def _valid_reader_label(row: dict) -> bool:
    return "error" not in row and all(type(row.get(field)) is bool for field in ("x", "y", "coherent"))


def preferred_label_by(rows: list[dict], key: str) -> dict[str, dict]:
    """Keep a valid retry when the same item also has an error row."""
    result = {}
    for row in rows:
        if row.get(key) is None:
            continue
        item_id = str(row[key])
        previous = result.get(item_id)
        if previous is None or _valid_reader_label(row) or not _valid_reader_label(previous):
            result[item_id] = row
    return result


def retry_item_count(rows: list[dict]) -> int:
    counts = Counter(str(row["item_id"]) for row in rows if row.get("item_id") is not None)
    return sum(count > 1 for count in counts.values())


def join_reader_keys(labels: list[dict], keys: list[dict]) -> list[dict]:
    """Join private row metadata only at scoring time, by opaque item id."""
    keys_by_id = {}
    for row in keys:
        item_id = str(row["item_id"])
        if item_id in keys_by_id:
            raise ValueError(f"duplicate reader key for opaque item {item_id!r}")
        keys_by_id[item_id] = row
    joined = []
    for row in labels:
        item_id = row.get("item_id")
        key = keys_by_id.get(str(item_id)) if item_id is not None else None
        if key is not None:
            joined.append({**row, **{name: value for name, value in key.items() if name != "item_id"}})
    return joined


def additive_top(logits_none, logits_first, logits_second) -> int:
    """Top id of L_first + L_second - L_none; common offsets cancel."""
    return int(np.argmax(np.asarray(logits_first) + np.asarray(logits_second) - np.asarray(logits_none)))


def is_contested(first_top: int, second_top: int) -> bool:
    return int(first_top) != int(second_top)


def winner_take_all(a: int, b: int, margin_a: float, margin_b: float) -> int | None:
    """Choose the top token from the single run with the larger top-two margin."""
    if float(margin_a) > float(margin_b):
        return int(a)
    if float(margin_b) > float(margin_a):
        return int(b)
    return None


def row_hit_rates(logit_row: dict) -> dict:
    """Compute rates on contested positions only; an empty row has no rate."""
    positions = logit_row.get("contested", [])
    if not positions:
        return {"n_contested": 0, "additive": None, "winner_take_all": None}
    additive_hits = 0
    wta_hits = 0
    for position in positions:
        additive_hits += int(int(position["add"]) == int(position["g"]))
        prediction = winner_take_all(position["a"], position["b"], position["Ma"], position["Mb"])
        wta_hits += int(prediction is not None and prediction == int(position["g"]))
    count = len(positions)
    return {"n_contested": count, "additive": additive_hits / count,
            "winner_take_all": wta_hits / count}


def row_unit_ids(row_id: str) -> tuple[str, str] | None:
    if row_id.startswith("pair:"):
        parts = row_id[len("pair:"):].split("+", 1)
        return (parts[0], parts[1]) if len(parts) == 2 else None
    if row_id.startswith("control:"):
        parts = row_id[len("control:"):].split("+R", 1)
        return (parts[0], parts[1]) if len(parts) == 2 else None
    return None


def row_bootstrap_weight(row_id: str, draw_counts: dict[str, int]) -> int:
    """A resampled pair/control row gets the product of its units' draw counts."""
    unit_pair = row_unit_ids(row_id)
    if unit_pair is None:
        return 0
    return int(draw_counts.get(unit_pair[0], 0)) * int(draw_counts.get(unit_pair[1], 0))


def make_unit_bootstrap_draws(unit_ids: list[str], draws: int = BOOTSTRAP_DRAWS,
                              seed: int = BOOTSTRAP_SEED) -> list[Counter]:
    rng = random.Random(seed)
    return [Counter(rng.choice(unit_ids) for _ in range(len(unit_ids))) for _ in range(draws)]


def _weighted_mean(row_rates: list[tuple[dict, dict]], field: str,
                   weights: dict[str, int]) -> float | None:
    numerator = denominator = 0.0
    for row, rates in row_rates:
        value = rates[field]
        if value is None:
            continue
        weight = row_bootstrap_weight(str(row["row_id"]), weights)
        if weight:
            numerator += value * weight
            denominator += weight
    return numerator / denominator if denominator else None


def _percentile(values: list[float], q: float) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.quantile(finite, q)) if finite else None


def summarize_rows(rows: list[dict], unit_ids: list[str],
                   bootstrap_draws: list[Counter] | None = None) -> dict:
    row_rates = [(row, row_hit_rates(row)) for row in rows]
    usable_rates = [(row, rates) for row, rates in row_rates if rates["n_contested"] > 0]
    usable = [row for row, _rates in usable_rates]
    rates = [rates for _row, rates in usable_rates]
    additive = [float(rate["additive"]) for rate in rates]
    wta = [float(rate["winner_take_all"]) for rate in rates]
    result = {
        "rows": len(rows), "rows_with_contested": len(usable),
        "contested_positions": sum(rate["n_contested"] for rate in rates),
        "additive_mean": float(np.mean(additive)) if additive else None,
        "winner_take_all_mean": float(np.mean(wta)) if wta else None,
    }
    if bootstrap_draws is None:
        bootstrap_draws = make_unit_bootstrap_draws(unit_ids)
    additive_draws, wta_draws, diff_draws = [], [], []
    for counts in bootstrap_draws:
        add = _weighted_mean(usable_rates, "additive", counts)
        winner = _weighted_mean(usable_rates, "winner_take_all", counts)
        if add is not None:
            additive_draws.append(add)
        if winner is not None:
            wta_draws.append(winner)
        if add is not None and winner is not None:
            diff_draws.append(add - winner)
    result["additive_ci95"] = [_percentile(additive_draws, .025), _percentile(additive_draws, .975)]
    result["winner_take_all_ci95"] = [_percentile(wta_draws, .025), _percentile(wta_draws, .975)]
    result["difference_mean"] = (result["additive_mean"] - result["winner_take_all_mean"]
                                 if additive and wta else None)
    result["difference_ci95"] = [_percentile(diff_draws, .025), _percentile(diff_draws, .975)]
    return result


def _token_counts(contested: list[dict]) -> tuple[int, int]:
    a_wins = sum(int(int(position["g"]) == int(position["a"])) for position in contested)
    b_wins = sum(int(int(position["g"]) == int(position["b"])) for position in contested)
    return a_wins, b_wins


def token_winner(contested: list[dict]) -> str | None:
    a_wins, b_wins = _token_counts(contested)
    if a_wins > b_wins:
        return "a"
    if b_wins > a_wins:
        return "b"
    return None


def common_prefix_length(first: list[int], second: list[int], limit: int = PATH_TOKENS) -> int:
    length = 0
    for a, b in zip(first[:limit], second[:limit]):
        if int(a) != int(b):
            break
        length += 1
    return length


def _first_flip(single: list[int], plain: list[int]) -> tuple[int | None, int | None, int | None]:
    limit = min(PATH_TOKENS, len(single), len(plain))
    for pos in range(limit):
        if int(single[pos]) != int(plain[pos]):
            return pos, int(single[pos]), int(plain[pos])
    return None, None, None


def _mann_whitney(first: list[float], second: list[float], alternative: str) -> float | None:
    if not first or not second:
        return None
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(first, second, alternative=alternative).pvalue)
    except ImportError:
        return None


def _spearman(first: list[float], second: list[float]) -> tuple[float | None, float | None]:
    if len(first) < 2 or len(second) != len(first):
        return None, None
    if len(set(first)) < 2 or len(set(second)) < 2:
        return None, None
    try:
        from scipy.stats import spearmanr
        result = spearmanr(first, second)
        corr = float(result.statistic)
        pvalue = float(result.pvalue)
        if not math.isfinite(corr) or not math.isfinite(pvalue):
            return None, None
        return corr, pvalue
    except ImportError:
        return None, None


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def build_tournament(units: list[dict], pair_logits: list[dict]) -> dict:
    by_id = {unit["id"]: unit for unit in units}
    logit_by_id = {row["row_id"]: row for row in pair_logits}
    wins = Counter()
    decided = Counter()
    counted_pairs = set()
    matrix = []
    for first in units:
        cells = []
        for second in units:
            if first["id"] == second["id"]:
                cells.append(None)
                continue
            a_id, b_id = sorted((first["id"], second["id"]))
            row = logit_by_id.get(f"pair:{a_id}+{b_id}")
            if row is None:
                cells.append({"winner": None, "a_tokens": 0, "b_tokens": 0,
                              "contested_positions": 0, "missing": True})
                continue
            a_id, b_id = row_unit_ids(row["row_id"])
            a_count, b_count = _token_counts(row.get("contested", []))
            side = token_winner(row.get("contested", []))
            winner_id = a_id if side == "a" else b_id if side == "b" else None
            pair_key = tuple(sorted((a_id, b_id)))
            if winner_id is not None and pair_key not in counted_pairs:
                wins[winner_id] += 1
                decided[winner_id] += 1
                decided[a_id if winner_id == b_id else b_id] += 1
            counted_pairs.add(pair_key)
            cells.append({"winner": winner_id, "a_unit_id": a_id, "b_unit_id": b_id,
                          "a_tokens": a_count, "b_tokens": b_count,
                          "contested_positions": len(row.get("contested", [])), "missing": False})
        matrix.append({"unit_id": first["id"], "cells": cells})
    metadata = []
    for unit in units:
        metadata.append({"id": unit["id"], "name": unit["persona"],
                         "haiku_kind": unit["haiku_kind"], "ds_kind": unit["ds_kind"],
                         "group": unit["group"], "wins": wins[unit["id"]],
                         "decided_pairs": decided[unit["id"]],
                         "dominance": wins[unit["id"]] / decided[unit["id"]]
                         if decided[unit["id"]] else None})
    return {"unit_order": [unit["id"] for unit in units], "units": metadata,
            "matrix": matrix, "matrix_semantics":
            "Each cell is the token-count winner for the unordered real pair; row and column order are unit_order. "
            "A missing row or tie/no token wins is undecided."}


def unit_answer_metrics(units: list[dict], answers: list[dict], ladder_rows: list[dict]) -> dict:
    answer_by_id = latest_by(answers, "row_id")
    ladder_by_id = latest_by(ladder_rows, "id")
    plain_row = answer_by_id.get("plain", {})
    plain = [int(token) for token in plain_row.get("tokens", [])]
    ladder_metrics = {}
    for unit in units:
        unit_id = unit["id"]
        single = answer_by_id.get(f"single:{unit_id}", {})
        single_tokens = [int(token) for token in single.get("tokens", [])]
        p, t_a, t_0 = _first_flip(single_tokens, plain)
        ladder = ladder_by_id.get(unit_id, {})
        flip_alpha = None
        margin_08 = None
        alphas = ladder.get("alphas", [])
        tops = ladder.get("tops", [])
        margins = ladder.get("margins", [])
        if p is not None and t_a is not None:
            for alpha, top in zip(alphas, tops):
                if top is not None and int(top) == t_a:
                    flip_alpha = float(alpha)
                    break
            for alpha, margin in zip(alphas, margins):
                if margin is not None and math.isclose(float(alpha), .8, abs_tol=1e-8):
                    margin_08 = float(margin)
                    break
        ladder_metrics[unit_id] = {"p": p, "t_A": t_a, "t_0": t_0,
                                   "flip_alpha": flip_alpha, "margin_08": margin_08}

    control_rows = {row["row_id"]: row for row in answers
                    if str(row.get("row_id", "")).startswith("control:")}
    robustness = {}
    for unit in units:
        unit_id = unit["id"]
        p = ladder_metrics[unit_id]["p"]
        single_tokens = [int(token) for token in answer_by_id.get(f"single:{unit_id}", {}).get("tokens", [])]
        values = []
        for other in units:
            if other["id"] == unit_id:
                continue
            row = control_rows.get(f"control:{unit_id}+R{other['id']}")
            if row is None or p is None:
                continue
            prefix = common_prefix_length([int(t) for t in row.get("tokens", [])], single_tokens)
            values.append(max(0, prefix - p) / (PATH_TOKENS - p))
        robustness[unit_id] = {"R": float(np.mean(values)) if values else None,
                               "control_rows": len(values)}
    return {unit_id: {**ladder_metrics[unit_id], **robustness[unit_id]} for unit_id in ladder_metrics}


def compute_gates(gate1: dict | None, answers: list[dict], part118: list[dict],
                  logits: list[dict], expected_logit_ids: set[str] | None = None,
                  units: list[dict] | None = None, gate2b: list[dict] | None = None) -> dict:
    no_hook = (gate1 or {}).get("no_hook_tokens")
    zero_hook = (gate1 or {}).get("zero_hook_tokens")
    gate1_passed = isinstance(no_hook, list) and isinstance(zero_hook, list) and no_hook == zero_hook
    gate1_result = {"passed": gate1_passed, "exact_token_match": gate1_passed,
                    "no_hook_tokens": no_hook, "zero_hook_tokens": zero_hook}

    part118_by_id = {}
    for row in part118:
        if row.get("arm") != "neuron" or not math.isclose(float(row.get("alpha", -1)), .8, abs_tol=1e-8):
            continue
        if int(row.get("question_index", -1)) != 0 and row.get("question") != QUESTION:
            continue
        part118_by_id[str(row.get("id"))] = row
    answer_by_id = latest_by(answers, "row_id")
    # Amendments 1 and 2: the gate compares Part 92's own hook path with the row hook at batch size 1 (gate2c.jsonl).
    # The comparison with Part 118's padded rows stays reported, not gated.
    reference_by_id = {str(row.get("id")): row for row in (gate2b or [])}
    matches = compared = concept_matches = concept_compared = 0
    part118_matches = part118_compared = 0
    for unit in (rows119.load_units() if units is None else units):
        first = answer_by_id.get(f"single:{unit['id']}")
        if first is None or "text" not in first:
            continue
        second = part118_by_id.get(unit["id"])
        if second is not None and "answer" in second:
            part118_compared += 1
            part118_matches += int(str(first["text"])[:100] == str(second["answer"])[:100])
        reference = reference_by_id.get(unit["id"])
        if reference is None or "part92_answer" not in reference or "row_hook_answer" not in reference:
            continue
        compared += 1
        # Amendment 2: both hook paths at batch size 1 (gate2c.py), compared with each other.
        match = str(reference["part92_answer"]).strip() == str(reference["row_hook_answer"]).strip()
        matches += int(match)
        if unit["group"] == "agreed_concept":
            concept_compared += 1
            concept_matches += int(match)
    gate2_passed = matches >= 38
    gate2_result = {"passed": gate2_passed, "matches": matches, "compared": compared,
                    "required_matches": 38, "total_units": 39,
                    "part118_first_100_char_matches": part118_matches, "part118_compared": part118_compared,
                    "agreed_concept_matches": concept_matches,
                    "agreed_concept_compared": concept_compared,
                    "agreed_concept_required": 6,
                    "kind_claims_passed": concept_matches >= 6,
                    "comparison": "first 100 decoded characters"}

    logit_by_id = latest_by(logits, "row_id")
    n_positions = sum(int(row.get("n_positions", 0)) for row in logits)
    both_hits = sum(int(row.get("both_hits", 0)) for row in logits)
    accuracy = both_hits / n_positions if n_positions else None
    real_positions = real_hits = control_positions = control_hits = 0
    real_contested = real_contested_hits = 0
    for row in logits:
        row_id = str(row.get("row_id", ""))
        if row_id.startswith("pair:"):
            for position in row.get("contested", []):
                real_contested += 1
                real_contested_hits += int(int(position.get("both_top", -1)) == int(position["g"]))
        if row_id.startswith("control:"):
            control_positions += int(row.get("n_positions", 0))
            control_hits += int(row.get("both_hits", 0))
    for row in logits:
        if str(row.get("row_id", "")).startswith("pair:"):
            real_positions += int(row.get("n_positions", 0))
            real_hits += int(row.get("both_hits", 0))
    real_contested_accuracy = real_contested_hits / real_contested if real_contested else None
    coverage_expected = expected_logit_ids
    coverage_missing = (sorted(coverage_expected - set(logit_by_id)) if coverage_expected is not None else [])
    coverage_extra = (sorted(set(logit_by_id) - coverage_expected) if coverage_expected is not None else [])
    complete = not coverage_missing and not coverage_extra if coverage_expected is not None else bool(logits)
    gate3_passed = bool(complete and accuracy is not None and accuracy >= .98
                        and real_contested_accuracy is not None and real_contested_accuracy >= .97)
    gate3_result = {"passed": gate3_passed, "both_hits": both_hits,
                    "positions": n_positions, "accuracy": accuracy,
                    "required_accuracy": .98, "complete": complete,
                    "real_positions": real_positions, "real_hits": real_hits,
                    "real_contested_positions": real_contested,
                    "real_contested_hits": real_contested_hits,
                    "real_contested_accuracy": real_contested_accuracy,
                    "real_contested_required_accuracy": .97,
                    "control_positions": control_positions, "control_hits": control_hits,
                    "control_accuracy": control_hits / control_positions if control_positions else None,
                    "missing_rows": coverage_missing, "extra_rows": coverage_extra}
    return {"gate1": gate1_result, "gate2": gate2_result, "gate3": gate3_result,
            "core_passed": gate1_passed and gate2_passed}


def _kind_class_summaries(logits: list[dict], all_pair_rows: list[dict],
                          units: list[dict], draws: list[Counter]) -> dict:
    by_id = {unit["id"]: unit for unit in units}
    row_class_by_id = {}
    for row in all_pair_rows:
        if row["row_type"] == "pair":
            row_class_by_id[row["row_id"]] = {
                "agreed": rows119.pair_class(row, by_id, "agreed"),
                "haiku": rows119.pair_class(row, by_id, "haiku"),
            }
    result = {}
    for scheme, names in (("agreed", ("slot_x_slot", "concept_x_concept", "slot_x_concept",
                                       "involving_disputed")),
                          ("haiku", ("slot_x_slot", "concept_x_concept", "slot_x_concept"))):
        groups = {name: [] for name in names}
        expected = Counter(row_class_by_id[row["row_id"]][scheme] for row in all_pair_rows
                           if row["row_type"] == "pair")
        for row in logits:
            if row["row_id"] in row_class_by_id:
                groups[row_class_by_id[row["row_id"]][scheme]].append(row)
        result[scheme] = {}
        for name in names:
            summary = summarize_rows(groups[name], [unit["id"] for unit in units], draws)
            summary["expected_rows"] = expected[name]
            result[scheme][name] = summary
    return result


def _distance_summaries(logits: list[dict], all_rows: list[dict], units: list[dict],
                        draws: list[Counter]) -> dict:
    by_id = {unit["id"]: unit for unit in units}
    row_meta = {row["row_id"]: row for row in all_rows}
    groups = {"pair": defaultdict(list), "control": defaultdict(list)}
    for logit in logits:
        meta = row_meta.get(logit["row_id"])
        if meta is None or meta["row_type"] not in groups:
            continue
        first, second = (by_id[unit_id] for unit_id in meta["unit_ids"])
        distance = abs(first["layer"] - second["layer"])
        groups[meta["row_type"]][distance].append(logit)
    result = {}
    for row_type, distances in groups.items():
        result[row_type] = {}
        for distance, rows in sorted(distances.items()):
            result[row_type][str(distance)] = summarize_rows(
                rows, [unit["id"] for unit in units], draws)
    return result


def _broken_summary(answers: list[dict], all_rows: list[dict]) -> dict:
    row_types = {row["row_id"]: row["row_type"] for row in all_rows}
    groups = defaultdict(lambda: {"broken": 0, "total": 0})
    for answer in answers:
        row_id = str(answer.get("row_id", ""))
        row_type = row_types.get(row_id, "unknown")
        key = {"single": "single", "random": "random", "pair": "real_pair",
               "control": "control", "plain": "plain"}.get(row_type, "other")
        broken = triage_answer(str(answer.get("text", ""))) == "broken"
        for group in ("all", key):
            groups[group]["broken"] += int(broken)
            groups[group]["total"] += 1
    return {name: {**counts, "share": counts["broken"] / counts["total"] if counts["total"] else None}
            for name, counts in groups.items()}


def interaction_share(position: dict) -> float | None:
    n_pair = float(position.get("nPair", 0.0))
    if n_pair <= 0:
        return None
    n_interaction = float(position.get("nI", 0.0))
    return (n_interaction * n_interaction) / (n_pair * n_pair)


def _interaction_row_values(row: dict) -> dict:
    shares = []
    tvs = []
    for position in row.get("positions", []):
        share = interaction_share(position)
        if share is not None:
            shares.append(share)
        if position.get("tv") is not None:
            tvs.append(float(position["tv"]))
    return {"share": _median(shares), "tv": _median(tvs),
            "positions": len(row.get("positions", [])), "share_positions": len(shares)}


def interaction_summaries(interactions: list[dict]) -> dict:
    result = {}
    for kind in ("real", "control"):
        rows = [row for row in interactions if row.get("kind") == kind]
        row_values = [_interaction_row_values(row) for row in rows]
        shares = [value["share"] for value in row_values if value["share"] is not None]
        tvs = [value["tv"] for value in row_values if value["tv"] is not None]
        result[kind] = {"rows": len(rows),
                        "rows_with_positions": sum(bool(row.get("positions")) for row in rows),
                        "rows_without_positions": sum(not row.get("positions") for row in rows),
                        "no_flip_units": sorted({str(row["first"]) for row in rows
                                                 if "p_A" in row and row.get("p_A") is None}),
                        "positions": sum(value["positions"] for value in row_values),
                        "median_row_interaction_share": _median(shares),
                        "median_row_tv": _median(tvs)}
    return result


def interaction_class_summaries(interactions: list[dict], units: list[dict]) -> dict:
    by_id = {unit["id"]: unit for unit in units}
    grouped = {}
    for scheme, class_names in (("agreed", ("slot_x_slot", "concept_x_concept", "slot_x_concept",
                                             "involving_disputed")),
                               ("haiku", ("slot_x_slot", "concept_x_concept", "slot_x_concept"))):
        for kind in ("real", "control"):
            groups = {name: [] for name in class_names}
            for row in interactions:
                if row.get("kind") != kind:
                    continue
                proxy = {"row_type": "pair", "unit_ids": [row["first"], row["second"]]}
                name = rows119.pair_class(proxy, by_id, scheme)
                if name in groups:
                    groups[name].append(row)
            expected = Counter()
            for row in rows119.interaction_rows(units, "main"):
                if row["kind"] != kind:
                    continue
                proxy = {"row_type": "pair", "unit_ids": [row["first"], row["second"]]}
                expected[rows119.pair_class(proxy, by_id, scheme)] += 1
            scheme_name = f"{kind}_{scheme}"
            grouped[scheme_name] = {}
            for name, rows in groups.items():
                stats = interaction_summaries(rows)[kind]
                stats["expected_rows"] = expected[name]
                grouped[scheme_name][name] = stats
    return grouped


def _interaction_weight(row: dict, counts: dict[str, int]) -> int:
    return int(counts.get(str(row["first"]), 0)) * int(counts.get(str(row["second"]), 0))


def effect_matched_comparison(interactions: list[dict], unit_ids: list[str],
                              bootstrap_draws: list[Counter] | None = None,
                              expected_real_rows: int | None = None) -> dict:
    """Fit log|I| from controls, then compare only real positions in control D2 support."""
    control_points = []
    control_log_d2 = []
    for row in interactions:
        if row.get("kind") != "control":
            continue
        for position in row.get("positions", []):
            n_i, n_a, n_2 = (float(position.get(key, 0.0)) for key in ("nI", "nA", "n2"))
            if n_2 > 0:
                control_log_d2.append(math.log(n_2))
            if min(n_i, n_a, n_2) <= 0:
                continue
            x_a, x_2, y = math.log(n_a), math.log(n_2), math.log(n_i)
            control_points.append((x_a, x_2, y))
    real_rows = [row for row in interactions if row.get("kind") == "real"]
    total_real_positions = sum(len(row.get("positions", [])) for row in real_rows)
    real_rows_with_positions = sum(bool(row.get("positions")) for row in real_rows)
    real_rows_expected = len(real_rows) if expected_real_rows is None else int(expected_real_rows)
    real_rows_without_positions = max(0, real_rows_expected - real_rows_with_positions)
    no_flip_units = sorted({str(row["first"]) for row in real_rows
                            if "p_A" in row and row.get("p_A") is None})
    row_coverage = {"real_rows_expected": real_rows_expected,
                    "real_rows_with_positions": real_rows_with_positions,
                    "real_rows_without_positions": real_rows_without_positions,
                    "no_flip_units": no_flip_units}
    if len(control_points) < 3 or not control_log_d2:
        return {"fit_available": False, "reason": "fewer than three positive control positions",
                "control_positions_fit": len(control_points),
                "control_positions_D2_support": len(control_log_d2),
                "real_positions_total": total_real_positions,
                **row_coverage,
                "real_positions_kept": 0, "share_real_positions_kept": 0.0,
                "row_residuals": [], "mean_residual": None, "mean_residual_ci95": [None, None],
                "exp_mean_residual": None}
    design = np.asarray([[1.0, x_a, x_2] for x_a, x_2, _y in control_points], dtype=np.float64)
    target = np.asarray([y for _x_a, _x_2, y in control_points], dtype=np.float64)
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(design, target, rcond=None)
    lower, upper = np.quantile(control_log_d2, [.01, .99])
    residual_by_row = []
    kept = 0
    selected_real_positions = 0
    for row in real_rows:
        residuals = []
        for position in row.get("positions", []):
            n_i, n_a, n_2 = (float(position.get(key, 0.0)) for key in ("nI", "nA", "n2"))
            if n_2 <= 0:
                continue
            x_2 = math.log(n_2)
            if x_2 < lower or x_2 > upper:
                continue
            selected_real_positions += 1
            if n_i <= 0 or n_a <= 0:
                continue
            x_a = math.log(n_a)
            residuals.append(math.log(n_i) - float(coefficients @ np.asarray([1.0, x_a, x_2])))
            kept += 1
        if residuals:
            residual_by_row.append({"first": str(row["first"]), "second": str(row["second"]),
                                   "value": float(np.median(residuals)),
                                   "positions": len(residuals)})
    mean = float(np.mean([row["value"] for row in residual_by_row])) if residual_by_row else None
    if bootstrap_draws is None:
        bootstrap_draws = make_unit_bootstrap_draws(unit_ids)
    draw_means = []
    for counts in bootstrap_draws:
        numerator = denominator = 0.0
        for row in residual_by_row:
            weight = _interaction_weight(row, counts)
            if weight:
                numerator += float(row["value"]) * weight
                denominator += weight
        if denominator:
            draw_means.append(numerator / denominator)
    ci = [_percentile(draw_means, .025), _percentile(draw_means, .975)]
    return {"fit_available": rank == 3, "fit_rank": int(rank),
            "coefficients_intercept_log_DA_log_D2": [float(value) for value in coefficients],
            "control_positions_fit": len(control_points),
            "control_positions_D2_support": len(control_log_d2),
            "control_log_D2_percentiles_1_99": [float(lower), float(upper)],
            "real_positions_total": total_real_positions,
            **row_coverage,
            "real_positions_with_D2_in_range": selected_real_positions,
            "real_positions_kept": kept,
            "share_real_positions_kept": kept / total_real_positions if total_real_positions else None,
            "row_residuals": residual_by_row,
            "rows_with_residual": len(residual_by_row),
            "mean_residual": mean, "mean_residual_ci95": ci,
            "exp_mean_residual": math.exp(mean) if mean is not None else None}


def _cohen_kappa(first: dict[str, dict], second: dict[str, dict]) -> dict:
    ids = sorted(set(first) & set(second))
    valid_ids = [item_id for item_id in ids
                 if all(type(first[item_id].get(key)) is bool for key in ("x", "y"))
                 and all(type(second[item_id].get(key)) is bool for key in ("x", "y"))]
    first_both = [bool(first[item_id]["x"] and first[item_id]["y"]) for item_id in valid_ids]
    second_both = [bool(second[item_id]["x"] and second[item_id]["y"]) for item_id in valid_ids]
    if not valid_ids:
        return {"kappa": None, "matched_valid": 0, "observed_agreement": None}
    observed = sum(a == b for a, b in zip(first_both, second_both)) / len(valid_ids)
    p1 = sum(first_both) / len(valid_ids)
    p2 = sum(second_both) / len(valid_ids)
    expected = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return {"kappa": float(kappa) if kappa is not None else None,
            "matched_valid": len(valid_ids), "observed_agreement": observed}


def _known_gate(known: list[dict], labels: list[dict], present: bool) -> dict:
    by_id = {str(row.get("item_id")): row for row in labels}
    correct = 0
    for index, item in enumerate(known):
        row = by_id.get(f"known-{index + 1:02d}", {})
        if "error" not in row and all(type(row.get(key)) is bool for key in ("x", "y", "coherent")) \
                and row["x"] is item["expected_x"] and row["y"] is item["expected_y"] \
                and row["coherent"] is item["expected_coherent"]:
            correct += 1
    return {"present": present, "correct": correct, "total": len(known),
            "passed": present and len(known) == 12 and correct >= 10}


def reader_analysis(units: list[dict], labels_ds: list[dict], labels_luna: list[dict],
                    known: list[dict], known_ds: list[dict], known_luna: list[dict],
                    files_present: dict[str, bool]) -> dict:
    sample = rows119.reader_sample(units)
    ds_by_id = preferred_label_by(labels_ds, "row_id")
    luna_by_id = preferred_label_by(labels_luna, "row_id")
    expected_luna = random.Random(11903).sample(sample, 30)
    ds_expected_ids = {item["row_id"] for item in sample}
    luna_expected_ids = {item["row_id"] for item in expected_luna}
    ds_missing = sorted(ds_expected_ids - set(ds_by_id))
    luna_missing = sorted(luna_expected_ids - set(luna_by_id))
    ds_errors = sum("error" in ds_by_id.get(item_id, {}) for item_id in ds_expected_ids)
    luna_errors = sum("error" in luna_by_id.get(item_id, {}) for item_id in luna_expected_ids)
    by_class = {}
    for name in ("concept_x_concept", "slot_x_concept", "slot_x_slot", "control"):
        members = [item for item in sample if item["sample_class"] == name]
        valid_members = [item for item in members
                         if all(type(ds_by_id.get(item["row_id"], {}).get(key)) is bool
                                for key in ("x", "y", "coherent"))]
        both = sum(int(ds_by_id[item["row_id"]]["x"] and ds_by_id[item["row_id"]]["y"])
                   for item in valid_members)
        by_class[name] = {"both": both, "total": len(valid_members), "expected": len(members),
                          "share": both / len(valid_members) if valid_members else None,
                          "errors": sum("error" in ds_by_id.get(item["row_id"], {}) for item in members)}
    ds_gate = _known_gate(known, known_ds, files_present.get("known_ds", False))
    luna_gate = _known_gate(known, known_luna, files_present.get("known_luna", False))
    kappa = _cohen_kappa(ds_by_id, luna_by_id)
    complete = not ds_missing and not luna_missing
    leave_one_out = {}
    concept_ids = [unit["id"] for unit in units if unit["group"] == "agreed_concept"]
    for unit_id in concept_ids:
        remaining = [item for item in sample if item["sample_class"] == "concept_x_concept"
                     and unit_id not in (item["x_unit_id"], item["y_unit_id"])]
        valid = [item for item in remaining
                 if all(type(ds_by_id.get(item["row_id"], {}).get(key)) is bool
                        for key in ("x", "y", "coherent"))]
        both = sum(1 for item in valid if ds_by_id[item["row_id"]]["x"]
                   and ds_by_id[item["row_id"]]["y"])
        leave_one_out[unit_id] = {"both": both, "total": len(remaining),
                                  "labeled": len(valid),
                                  "errors": sum("error" in ds_by_id.get(item["row_id"], {}) for item in remaining),
                                  "share": both / len(valid) if valid else None}
    gates_passed = ds_gate["passed"] and luna_gate["passed"] and complete \
        and kappa["kappa"] is not None and kappa["kappa"] >= .6
    return {
        "known_ds": ds_gate, "known_luna": luna_gate,
        "sample_files_present": {"ds": files_present.get("ds", False),
                                  "luna": files_present.get("luna", False)},
        "sample_coverage": {"ds_items": len(ds_by_id), "ds_expected": 148,
                            "ds_missing": ds_missing, "ds_errors": ds_errors,
                            "ds_retry_items": retry_item_count(labels_ds),
                            "luna_items": len(luna_by_id), "luna_expected": 30,
                            "luna_missing": luna_missing, "luna_errors": luna_errors,
                            "luna_retry_items": retry_item_count(labels_luna)},
        "both_by_class": by_class, "second_reader_both_kappa": kappa,
        "leave_one_concept_out": leave_one_out,
        "passed": bool(gates_passed and files_present.get("ds", False)
                       and files_present.get("luna", False)),
        "rule": "119.9 is withheld unless both known-example gates pass, all sample rows are present, "
                "and DS/Luna agreement on the binary both label has kappa >= 0.6.",
    }


def _pair_winner_summary(logits: list[dict], units: list[dict], all_pair_rows: list[dict],
                         scheme: str = "agreed", leave_one_out: bool = False) -> dict:
    by_id = {unit["id"]: unit for unit in units}
    row_meta = {row["row_id"]: row for row in all_pair_rows}
    candidates = []
    expected = 0
    for row in all_pair_rows:
        if row["row_type"] == "pair" and rows119.pair_class(row, by_id, scheme) == "slot_x_concept":
            candidates.append(row)
    logit_by_id = {row["row_id"]: row for row in logits}

    def measure(excluded: str | None = None):
        decided = concept_wins = observed = 0
        active = [meta for meta in candidates
                  if excluded is None or excluded not in meta["unit_ids"]]
        for meta in active:
            a_id, b_id = meta["unit_ids"]
            logit = logit_by_id.get(meta["row_id"])
            if logit is None:
                continue
            observed += 1
            side = token_winner(logit.get("contested", []))
            if side is None:
                continue
            winner_id = a_id if side == "a" else b_id
            kind = by_id[winner_id]["group"] if scheme == "agreed" else by_id[winner_id]["haiku_kind"]
            decided += 1
            concept_wins += int(kind == ("agreed_concept" if scheme == "agreed" else "concept"))
        return {"expected_pairs": len(active), "observed_pairs": observed,
                "decided_pairs": decided, "concept_wins": concept_wins,
                "concept_win_share": concept_wins / decided if decided else None}

    result = measure()
    if leave_one_out and scheme == "agreed":
        concept_ids = [unit["id"] for unit in units if unit["group"] == "agreed_concept"]
        result["leave_one_concept_out"] = {
            unit_id: measure(unit_id) for unit_id in concept_ids
        }
    return result


def _unit_correlations(units: list[dict], tournament: dict, unit_metrics: dict) -> dict:
    dominance = {row["id"]: row["dominance"] for row in tournament["units"]}
    pairs = [(dominance[unit["id"]], unit_metrics[unit["id"]]["flip_alpha"])
             for unit in units if dominance[unit["id"]] is not None
             and unit_metrics[unit["id"]]["flip_alpha"] is not None]
    corr, pvalue = _spearman([p[0] for p in pairs], [p[1] for p in pairs])
    return {"n_units": len(pairs), "spearman_rho": corr, "pvalue_two_sided": pvalue}


def _group_values(units: list[dict], unit_metrics: dict, field: str,
                  scheme: str) -> tuple[list[float], list[float], list[float]]:
    if scheme == "agreed":
        slot_kind, concept_kind = "agreed_slot", "agreed_concept"
        selected = lambda unit: unit["group"] in (slot_kind, concept_kind)
    else:
        slot_kind, concept_kind = "slot", "concept"
        selected = lambda unit: True
    slots, concepts, disputed = [], [], []
    for unit in units:
        value = unit_metrics[unit["id"]].get(field)
        if value is None or not selected(unit):
            continue
        kind = unit["group"] if scheme == "agreed" else unit["haiku_kind"]
        if kind == slot_kind:
            slots.append(float(value))
        elif kind == concept_kind:
            concepts.append(float(value))
            if scheme == "agreed" and unit["group"] == "disputed":
                disputed.append(float(value))
    return slots, concepts, disputed


def _direction(value: float | None) -> int:
    if value is None or value == 0:
        return 0
    return 1 if value > 0 else -1


def interaction_row_coverage_ok(rows_without_positions: int, expected_rows: int) -> bool:
    return expected_rows > 0 and rows_without_positions / expected_rows <= .05


def evaluate_predictions(gates: dict, logit_summary: dict, interaction_summary: dict,
                         effect_match: dict, pair_winners: dict, pair_winners_haiku: dict,
                         unit_metrics: dict, units: list[dict], correlations: dict,
                         reader: dict, answers_complete: bool,
                         interactions_complete: bool, ladder_complete: bool) -> list[dict]:
    core = gates["core_passed"]
    gate3 = gates["gate3"]["passed"]
    kind_gate = gates["gate2"].get("kind_claims_passed", False)
    logit_ok = core and gate3
    answer_ok = core and answers_complete
    interaction_ok = logit_ok and interactions_complete
    reader_ok = core and kind_gate and reader.get("passed", False)
    real_rate = logit_summary["real_pairs"].get("additive_mean")
    real_ci = logit_summary["real_pairs"].get("additive_ci95", [None, None])
    diff = logit_summary["real_pairs"].get("difference_mean")
    diff_ci = logit_summary["real_pairs"].get("difference_ci95", [None, None])
    predictions = []

    def add(number, threshold, observed, available, holds, reason=None):
        verdict = ("not read" if not available else "held" if holds else "failed")
        predictions.append({"id": number, "threshold": threshold, "observed": observed,
                            "verdict": verdict, "reason": reason if verdict == "not read" else None})

    add("119.1", "real additive row-mean hit rate >= 0.60", real_rate,
        logit_ok and real_rate is not None, real_rate is not None and real_rate >= .60,
        "Gate 1/2 or Gate 3 failed" if not logit_ok else "no contested real-pair rows")
    add("119.2", "additive minus margin winner-take-all >= 0.05 and 95% bootstrap CI lower > 0",
        {"difference": diff, "ci95": diff_ci},
        logit_ok and diff is not None and diff_ci[0] is not None,
        diff is not None and diff >= .05 and diff_ci[0] > 0,
        "Gate 1/2 or Gate 3 failed" if not logit_ok else "insufficient contested rows")

    real_interaction = interaction_summary["real"].get("median_row_interaction_share")
    add("119.3", "median row interaction share across 1,482 directed real pairs <= 0.25",
        {"median_interaction_share": real_interaction,
         "rows_with_positions": interaction_summary["real"].get("rows_with_positions"),
         "rows_without_positions": interaction_summary["real"].get("rows_without_positions"),
         "no_flip_units": interaction_summary["real"].get("no_flip_units", []),
         "expected_rows": interaction_summary["real"].get("expected_rows", 1482)},
        interaction_ok and real_interaction is not None
        and interaction_row_coverage_ok(
            interaction_summary["real"].get("rows_without_positions", 1482),
            interaction_summary["real"].get("expected_rows", 1482)),
        real_interaction is not None and real_interaction <= .25,
        "Path 1 gate or Path 2 coverage failed" if not interaction_ok else
        "more than 5% of real rows have no positions" if not interaction_row_coverage_ok(
            interaction_summary["real"].get("rows_without_positions", 1482),
            interaction_summary["real"].get("expected_rows", 1482)) else
        "no real interaction rows with positions")
    residual_ci = effect_match.get("mean_residual_ci95", [None, None])
    effect_bound = math.log(1.25)
    effect_holds = (residual_ci[0] is not None and residual_ci[1] is not None
                    and residual_ci[0] >= -effect_bound and residual_ci[1] <= effect_bound)
    add("119.4", "effect-matched mean residual 95% unit-bootstrap CI inside +/- log(1.25)",
        effect_match, interaction_ok and effect_match.get("fit_available", False)
        and effect_match.get("mean_residual") is not None and residual_ci[0] is not None,
        effect_holds,
        "Path 1 gate or Path 2 coverage failed" if not interaction_ok else "effect-matched fit unavailable")

    winner_share = pair_winners.get("concept_win_share")
    winner_haiku_share = pair_winners_haiku.get("concept_win_share")
    winner_directions_agree = not (_direction((winner_share or 0) - .5)
                                    * _direction((winner_haiku_share or 0) - .5) < 0)
    winner_observed = {**pair_winners, "haiku_kinds": pair_winners_haiku,
                       "kind_direction_agrees": winner_directions_agree}
    winner_available = logit_ok and kind_gate and winner_directions_agree
    add("119.5", "at least 84/168 decided slot x concept pairs; concept wins >= 0.60",
        winner_observed, winner_available,
        pair_winners.get("decided_pairs", 0) >= 84 and winner_share is not None and winner_share >= .60,
        "Gate 1/2, Gate 3, or concept kind gate failed" if not (logit_ok and kind_gate)
        else ("kind directions differ" if not winner_directions_agree else
              "fewer than 84 decided pairs or no pair has a token winner"))

    concept_r = [unit_metrics[unit["id"]]["R"] for unit in units
                 if unit["group"] == "agreed_concept" and unit_metrics[unit["id"]]["R"] is not None]
    slot_r = [unit_metrics[unit["id"]]["R"] for unit in units
              if unit["group"] == "agreed_slot" and unit_metrics[unit["id"]]["R"] is not None]
    r_difference = _median(concept_r) - _median(slot_r) if concept_r and slot_r else None
    r_p = _mann_whitney(concept_r, slot_r, "greater")
    haiku_slot_r, haiku_concept_r, _ = _group_values(units, unit_metrics, "R", "haiku")
    haiku_r_diff = _median(haiku_concept_r) - _median(haiku_slot_r) \
        if haiku_slot_r and haiku_concept_r else None
    r_direction_agrees = not (_direction(r_difference) * _direction(haiku_r_diff) < 0)
    obs_1196 = {"concept_median_R": _median(concept_r), "slot_median_R": _median(slot_r),
                "median_difference": r_difference, "mann_whitney_p_one_sided": r_p,
                "n_concept": len(concept_r), "n_slot": len(slot_r),
                "haiku_median_difference": haiku_r_diff,
                "kind_direction_agrees": r_direction_agrees}
    add("119.6", "concept minus slot median R >= 0.10 and one-sided Mann-Whitney p < 0.05",
        obs_1196, answer_ok and kind_gate and r_difference is not None and r_p is not None
        and r_direction_agrees,
        r_difference is not None and r_difference >= .10 and r_p is not None and r_p < .05,
        "answer rows incomplete or kind gate failed" if not (answer_ok and kind_gate)
        else "insufficient R values or kind directions differ")

    concept_flip = [unit_metrics[unit["id"]]["flip_alpha"] for unit in units
                    if unit["group"] == "agreed_concept" and unit_metrics[unit["id"]]["flip_alpha"] is not None]
    slot_flip = [unit_metrics[unit["id"]]["flip_alpha"] for unit in units
                 if unit["group"] == "agreed_slot" and unit_metrics[unit["id"]]["flip_alpha"] is not None]
    flip_diff = _median(slot_flip) - _median(concept_flip) if slot_flip and concept_flip else None
    flip_p = _mann_whitney(concept_flip, slot_flip, "less")
    haiku_slot_flip, haiku_concept_flip, _ = _group_values(units, unit_metrics, "flip_alpha", "haiku")
    haiku_flip_diff = _median(haiku_slot_flip) - _median(haiku_concept_flip) \
        if haiku_slot_flip and haiku_concept_flip else None
    flip_direction_agrees = not (_direction(flip_diff) * _direction(haiku_flip_diff) < 0)
    obs_1197 = {"concept_median_flip_alpha": _median(concept_flip),
                "slot_median_flip_alpha": _median(slot_flip), "slot_minus_concept": flip_diff,
                "mann_whitney_p_one_sided": flip_p,
                "haiku_slot_minus_concept": haiku_flip_diff,
                "kind_direction_agrees": flip_direction_agrees,
                "n_concept": len(concept_flip), "n_slot": len(slot_flip)}
    add("119.7", "slot median flip alpha - concept median >= 0.10 and one-sided p < 0.05",
        obs_1197, logit_ok and ladder_complete and kind_gate and flip_diff is not None
        and flip_p is not None and flip_direction_agrees,
        flip_diff is not None and flip_diff >= .10 and flip_p is not None and flip_p < .05,
        "Gate 1/2, Gate 3, ladder, or concept kind gate failed"
        if not (logit_ok and ladder_complete and kind_gate)
        else "insufficient flip values or kind directions differ")

    rho = correlations.get("spearman_rho")
    add("119.8", "Spearman rho(dominance, flip alpha) <= -0.40", correlations,
        logit_ok and ladder_complete and rho is not None, rho is not None and rho <= -.40,
        "Gate 1/2, Gate 3, or ladder failed" if not (logit_ok and ladder_complete)
        else "insufficient unit values")

    disputed_r = [unit_metrics[unit["id"]]["R"] for unit in units
                  if unit["group"] == "disputed" and unit_metrics[unit["id"]]["R"] is not None]
    agreed_slot = [unit_metrics[unit["id"]]["R"] for unit in units
                   if unit["group"] == "agreed_slot" and unit_metrics[unit["id"]]["R"] is not None]
    agreed_concept = [unit_metrics[unit["id"]]["R"] for unit in units
                      if unit["group"] == "agreed_concept" and unit_metrics[unit["id"]]["R"] is not None]
    disputed_median = _median(disputed_r)
    slot_median = _median(agreed_slot)
    concept_median = _median(agreed_concept)
    margin = (abs(disputed_median - concept_median) - abs(disputed_median - slot_median)
              if None not in (disputed_median, slot_median, concept_median) else None)
    haiku_slot_r, haiku_concept_r, _ = _group_values(units, unit_metrics, "R", "haiku")
    haiku_margin = (abs(disputed_median - _median(haiku_concept_r))
                    - abs(disputed_median - _median(haiku_slot_r))) \
        if disputed_median is not None and haiku_slot_r and haiku_concept_r else None
    disputed_direction_agrees = not (_direction(margin) * _direction(haiku_margin) < 0)
    obs_1199 = {"disputed_median_R": disputed_median, "slot_median_R": slot_median,
                "concept_median_R": concept_median, "closer_to_slot_margin": margin,
                "haiku_closer_to_slot_margin": haiku_margin,
                "kind_direction_agrees": disputed_direction_agrees,
                "n_disputed": len(disputed_r)}
    add("119.9", "disputed R is closer to slot than concept by >= 0.05",
        obs_1199, answer_ok and kind_gate and margin is not None and disputed_direction_agrees,
        margin is not None and margin >= .05,
        "answer rows incomplete or kind gate failed" if not (answer_ok and kind_gate)
        else "insufficient R values or kind directions differ")

    cc = reader.get("both_by_class", {}).get("concept_x_concept", {}).get("share")
    ss = reader.get("both_by_class", {}).get("slot_x_slot", {}).get("share")
    ctrl = reader.get("both_by_class", {}).get("control", {}).get("share")
    obs_11910 = {"concept_x_concept_both_share": cc, "slot_x_slot_both_share": ss,
                 "control_both_share": ctrl, "reader_gates_passed": reader.get("passed", False),
                 "leave_one_concept_out": reader.get("leave_one_concept_out", {})}
    available = reader_ok and None not in (cc, ss, ctrl)
    holds = available and cc >= .25 and ss <= .10 and ctrl <= .05
    add("119.10", "both >= 0.25 concept pairs, <= 0.10 slot pairs, and <= 0.05 controls",
        obs_11910, available, holds,
        "reader gates or concept kind gate failed, or labels are absent/incomplete"
        if not available else None)
    return predictions


def headline(predictions: list[dict], real_rate: float | None, control_rate: float | None,
             gates: dict, interaction_share: float | None = None,
             effect_residual_ci: list[float | None] | None = None) -> str:
    by_id = {row["id"]: row for row in predictions}
    if not gates["core_passed"]:
        return "Not read: Gate 1 or Gate 2 failed."
    if all(by_id[f"119.{number}"]["verdict"] == "held" for number in (1, 2, 3, 4)):
        return ("At this dose on this question the two pushes' effects on next-word scores mostly add, "
                "and a real second neuron interacts no more than a random direction with the same single effects.")
    bound = math.log(1.25)
    if gates["gate3"]["passed"] and effect_residual_ci and \
            effect_residual_ci[0] is not None and effect_residual_ci[0] > bound:
        return "Real neuron pairs interact more than a neuron and a random direction with the same single effects."
    if gates["gate3"]["passed"] and ((real_rate is not None and real_rate < .50)
                                     or (interaction_share is not None and interaction_share > .5)):
        return "The pair's next word is mostly not predicted by adding single effects."
    if not gates["gate3"]["passed"]:
        return "Not read for logit predictions because Gate 3 failed; answer-only predictions are reported separately."
    return "No registered headline reading; report the numbers."


def score_data(answers: list[dict], logits: list[dict], ladder_rows: list[dict],
               units: list[dict], gate1: dict | None, part118: list[dict],
               gate2b: list[dict] | None = None,
               expected_answer_ids: set[str] | None = None,
               expected_logit_ids: set[str] | None = None,
               expected_ladder_ids: set[str] | None = None,
               labels_ds: list[dict] | None = None,
               labels_luna: list[dict] | None = None,
               known: list[dict] | None = None,
               known_ds: list[dict] | None = None,
               known_luna: list[dict] | None = None,
               reader_files_present: dict[str, bool] | None = None,
               interactions: list[dict] | None = None,
               expected_interaction_keys: set[tuple[str, str, str]] | None = None,
               stage: str = "main") -> tuple[dict, dict]:
    """Compute all report data from supplied files; optional expected sets enforce stage coverage."""
    labels_ds = [] if labels_ds is None else labels_ds
    labels_luna = [] if labels_luna is None else labels_luna
    known_ds = [] if known_ds is None else known_ds
    known_luna = [] if known_luna is None else known_luna
    interactions = [] if interactions is None else interactions
    reader_files_present = {} if reader_files_present is None else reader_files_present
    known = [] if known is None else known
    all_rows = rows119.all_rows(units)
    answer_by_id = latest_by(answers, "row_id")
    logit_by_id = latest_by(logits, "row_id")
    ladder_by_id = latest_by(ladder_rows, "id")
    interaction_keys = {(str(row["first"]), str(row["second"]), str(row["kind"]))
                        for row in interactions}

    answer_missing = sorted(expected_answer_ids - set(answer_by_id)) if expected_answer_ids is not None else []
    answer_extra = sorted(set(answer_by_id) - expected_answer_ids) if expected_answer_ids is not None else []
    answers_complete = (not answer_missing and not answer_extra) if expected_answer_ids is not None else bool(answers)
    ladder_missing = sorted(expected_ladder_ids - set(ladder_by_id)) if expected_ladder_ids is not None else []
    ladder_extra = sorted(set(ladder_by_id) - expected_ladder_ids) if expected_ladder_ids is not None else []
    ladder_complete = (not ladder_missing and not ladder_extra) if expected_ladder_ids is not None else bool(ladder_rows)
    interaction_missing = (sorted(expected_interaction_keys - interaction_keys)
                           if expected_interaction_keys is not None else [])
    interaction_extra = (sorted(interaction_keys - expected_interaction_keys)
                         if expected_interaction_keys is not None else [])
    interactions_complete = ((not interaction_missing and not interaction_extra)
                             if expected_interaction_keys is not None else bool(interactions))

    gates = compute_gates(gate1, answers, part118, logits, expected_logit_ids, units, gate2b=gate2b)
    unit_metrics = unit_answer_metrics(units, answers, ladder_rows)
    unit_ids = [unit["id"] for unit in units]
    draws = make_unit_bootstrap_draws(unit_ids)
    pair_logits = [row for row in logits if str(row.get("row_id", "")).startswith("pair:")]
    controls = [row for row in logits if str(row.get("row_id", "")).startswith("control:")]
    real_summary = summarize_rows(pair_logits, unit_ids, draws)
    control_summary = summarize_rows(controls, unit_ids, draws)
    class_summaries = _kind_class_summaries(logits, all_rows, units, draws)
    winners = _pair_winner_summary(pair_logits, units, all_rows, "agreed", leave_one_out=True)
    winners_haiku = _pair_winner_summary(pair_logits, units, all_rows, "haiku")
    tournament = build_tournament(units, pair_logits)
    dominance_correlation = _unit_correlations(units, tournament, unit_metrics)
    interaction_summary = interaction_summaries(interactions)
    expected_real_interaction_rows = (sum(kind == "real" for _first, _second, kind in expected_interaction_keys)
                                      if expected_interaction_keys is not None else
                                      interaction_summary["real"]["rows"])
    interaction_summary["real"]["expected_rows"] = expected_real_interaction_rows
    interaction_summary["real"]["rows_without_positions"] = max(
        0, expected_real_interaction_rows - interaction_summary["real"]["rows_with_positions"])
    interaction_summary["class_splits"] = interaction_class_summaries(interactions, units)
    effect_match = effect_matched_comparison(
        interactions, unit_ids, draws, expected_real_rows=expected_real_interaction_rows)

    concept_r = [unit_metrics[unit["id"]]["R"] for unit in units
                 if unit["group"] == "agreed_concept" and unit_metrics[unit["id"]]["R"] is not None]
    slot_r = [unit_metrics[unit["id"]]["R"] for unit in units
              if unit["group"] == "agreed_slot" and unit_metrics[unit["id"]]["R"] is not None]
    concept_flip = [unit_metrics[unit["id"]]["flip_alpha"] for unit in units
                    if unit["group"] == "agreed_concept" and unit_metrics[unit["id"]]["flip_alpha"] is not None]
    slot_flip = [unit_metrics[unit["id"]]["flip_alpha"] for unit in units
                 if unit["group"] == "agreed_slot" and unit_metrics[unit["id"]]["flip_alpha"] is not None]
    disputed_r = [unit_metrics[unit["id"]]["R"] for unit in units
                  if unit["group"] == "disputed" and unit_metrics[unit["id"]]["R"] is not None]
    haiku_slot_r, haiku_concept_r, _ = _group_values(units, unit_metrics, "R", "haiku")
    haiku_slot_flip, haiku_concept_flip, _ = _group_values(units, unit_metrics, "flip_alpha", "haiku")

    if not known:
        known_path = HERE / "known119.jsonl"
        known = read_jsonl(known_path)
    reader = reader_analysis(units, labels_ds, labels_luna, known, known_ds, known_luna,
                             reader_files_present)
    predictions = evaluate_predictions(
        gates, {"real_pairs": real_summary, "controls": control_summary},
        interaction_summary, effect_match, winners, winners_haiku,
        unit_metrics, units, dominance_correlation, reader, answers_complete,
        interactions_complete, ladder_complete,
    )
    real_interaction_share = interaction_summary["real"].get("median_row_interaction_share")
    headline_text = headline(predictions, real_summary["additive_mean"],
                             control_summary["additive_mean"], gates,
                             real_interaction_share, effect_match.get("mean_residual_ci95"))
    if stage == "pilot":
        for prediction in predictions:
            prediction["verdict"] = "not read"
            prediction["reason"] = "pilot stage reports gate and timing checks only"
        headline_text = "Pilot stage: registered predictions are not read."
    results = {
        "part": 119,
        "gate": gates,
        "coverage": {"answers_complete": answers_complete, "answer_missing": answer_missing,
                     "answer_extra": answer_extra, "logits_expected": len(expected_logit_ids)
                     if expected_logit_ids is not None else None,
                     "logit_rows": len(logit_by_id),
                     "ladder_complete": ladder_complete, "ladder_missing": ladder_missing,
                     "ladder_extra": ladder_extra,
                     "interactions_complete": interactions_complete,
                     "interaction_rows": len(interaction_keys),
                     "interaction_missing": interaction_missing,
                     "interaction_extra": interaction_extra},
        "logit_summary": {"real_pairs": real_summary, "controls": control_summary,
                          "class_splits": class_summaries,
                          "token_winner_slot_concept": winners,
                          "token_winner_slot_concept_haiku": winners_haiku},
        "interaction_summary": interaction_summary,
        "effect_matched": effect_match,
        "broken_answers": _broken_summary(answers, all_rows),
        "by_layer_distance_exploratory": _distance_summaries(logits, all_rows, units, draws),
        "unit_metrics": unit_metrics,
        "dominance_flip_alpha_spearman": dominance_correlation,
        "robustness_summary": {"agreed_concept_median": _median(concept_r),
                               "agreed_slot_median": _median(slot_r),
                               "disputed_median": _median(disputed_r),
                               "haiku_concept_median": _median(haiku_concept_r),
                               "haiku_slot_median": _median(haiku_slot_r),
                               "concept_n": len(concept_r), "slot_n": len(slot_r),
                               "disputed_n": len(disputed_r)},
        "flip_alpha_summary": {"agreed_concept_median": _median(concept_flip),
                               "agreed_slot_median": _median(slot_flip),
                               "haiku_concept_median": _median(haiku_concept_flip),
                               "haiku_slot_median": _median(haiku_slot_flip),
                               "concept_n": len(concept_flip), "slot_n": len(slot_flip)},
        "reader": reader,
        "predictions": predictions,
        "headline": headline_text,
    }
    return results, tournament


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(result: dict) -> str:
    gates = result["gate"]
    lines = ["# Part 119 score", "", f"**Headline:** {result['headline']}", "",
             "## Gates", "", "| Gate | Result | Observed |", "|---|---|---|"]
    lines.append(f"| 1 zero hook | {'pass' if gates['gate1']['passed'] else 'fail'} | "
                 f"exact token match: {gates['gate1']['exact_token_match']} |")
    lines.append(f"| 2 Part 92 hook path (Amendments 1 and 2) | {'pass' if gates['gate2']['passed'] else 'fail'} | "
                 f"{gates['gate2']['matches']}/{gates['gate2']['total_units']} exact decoded answers "
                 f"({gates['gate2']['compared']} compared; Part 118's padded rows, reported only: "
                 f"{gates['gate2']['part118_first_100_char_matches']}/{gates['gate2']['part118_compared']}); agreed concepts "
                 f"{gates['gate2']['agreed_concept_matches']}/{gates['gate2']['agreed_concept_compared']} "
                 f"(kind gate={gates['gate2']['kind_claims_passed']}) |")
    lines.append(f"| 3 teacher-forced path | {'pass' if gates['gate3']['passed'] else 'fail'} | "
                 f"all {gates['gate3']['both_hits']}/{gates['gate3']['positions']} "
                 f"({_fmt(gates['gate3']['accuracy'])}); real contested "
                 f"{gates['gate3']['real_contested_hits']}/{gates['gate3']['real_contested_positions']} "
                 f"({_fmt(gates['gate3']['real_contested_accuracy'])}); control "
                 f"{gates['gate3']['control_hits']}/{gates['gate3']['control_positions']} "
                 f"({_fmt(gates['gate3']['control_accuracy'])}); complete={gates['gate3']['complete']} |")
    lines.extend(["", "## Registered predictions", "",
                  "| Prediction | Threshold | Observed | Verdict |", "|---|---|---|---|"])
    for item in result["predictions"]:
        observed = json.dumps(item["observed"], ensure_ascii=False, sort_keys=True) \
            if isinstance(item["observed"], (dict, list)) else _fmt(item["observed"])
        lines.append(f"| {item['id']} | {item['threshold']} | {observed} | {item['verdict']} |")
    summary = result["logit_summary"]
    lines.extend(["", "## Contested-position predictions", "",
                  "Rates are means of row rates over rows with at least one contested position. "
                  "Winner-take-all selects the single with the larger top-two margin; exact ties miss. "
                  "Intervals resample 39 units with replacement (2,000 draws, seed 11902).", "",
                  "| Rows | Additive mean (95% CI) | Margin winner mean (95% CI) |", "|---|---:|---:|"])
    for name, key in (("Real pairs", "real_pairs"), ("Controls", "controls")):
        row = summary[key]
        add_ci = ", ".join(_fmt(v) for v in row["additive_ci95"])
        wta_ci = ", ".join(_fmt(v) for v in row["winner_take_all_ci95"])
        lines.append(f"| {name} ({row['rows_with_contested']}/{row['rows']} contested rows) | "
                     f"{_fmt(row['additive_mean'])} ({add_ci}) | "
                     f"{_fmt(row['winner_take_all_mean'])} ({wta_ci}) |")
    lines.extend(["", "## Pair classes", "", "### Agreed Part 92b groups", "",
                  "| Class | Expected rows | Rows with contested positions | Additive | Winner-take-all |",
                  "|---|---:|---:|---:|---:|"])
    for name, values in summary["class_splits"]["agreed"].items():
        lines.append(f"| {name} | {values['expected_rows']} | {values['rows_with_contested']} | "
                     f"{_fmt(values['additive_mean'])} | {_fmt(values['winner_take_all_mean'])} |")
    lines.extend(["", "### Part 92 Haiku groups", "",
                  "| Class | Expected rows | Rows with contested positions | Additive | Winner-take-all |",
                  "|---|---:|---:|---:|---:|"])
    for name, values in summary["class_splits"]["haiku"].items():
        lines.append(f"| {name} | {values['expected_rows']} | {values['rows_with_contested']} | "
                     f"{_fmt(values['additive_mean'])} | {_fmt(values['winner_take_all_mean'])} |")
    interaction = result["interaction_summary"]
    lines.extend(["", "## Path 2 interaction", "",
                  "For each row, interaction share and TV are medians over positions from p_A onward. "
                  "Interaction share uses centered vectors on the union of the four top-10 token sets; TV uses "
                  "the full-vocabulary softmaxes.", "",
                  "| Row class | Rows with positions | Median row interaction share | Median row TV |",
                  "|---|---:|---:|---:|"])
    for kind in ("real", "control"):
        item = interaction[kind]
        lines.append(f"| {kind} ({item['rows']}) | {item['rows_with_positions']} | "
                     f"{_fmt(item['median_row_interaction_share'])} | {_fmt(item['median_row_tv'])} |")
    lines.append(f"Real Path 2 rows with positions: {interaction['real']['rows_with_positions']}/"
                 f"{interaction['real'].get('expected_rows', 1482)}; without positions: "
                 f"{interaction['real'].get('rows_without_positions')}; no-flip units: "
                 f"{', '.join(interaction['real'].get('no_flip_units', [])) or 'none'}.")
    lines.extend(["", "### Effect-matched comparison", "",
                  f"Fit available: {result['effect_matched']['fit_available']}; control positions in fit: "
                  f"{result['effect_matched']['control_positions_fit']}; real positions kept: "
                  f"{result['effect_matched']['real_positions_kept']}/"
                  f"{result['effect_matched']['real_positions_total']} "
                  f"({_fmt(result['effect_matched']['share_real_positions_kept'])}).",
                  f"Real rows with positions: {result['effect_matched'].get('real_rows_with_positions')}/"
                  f"{result['effect_matched'].get('real_rows_expected')}; without positions: "
                  f"{result['effect_matched'].get('real_rows_without_positions')}; no-flip units: "
                  f"{', '.join(result['effect_matched'].get('no_flip_units', [])) or 'none'}.",
                  f"Mean row residual: {_fmt(result['effect_matched']['mean_residual'])}; 95% unit-bootstrap CI: "
                  f"{[_fmt(x) for x in result['effect_matched']['mean_residual_ci95']]}; "
                  f"exp(mean residual): {_fmt(result['effect_matched']['exp_mean_residual'])}.",
                  "", "### Path 2 class splits", "",
                  "| Class set | Class | Expected rows | Rows with positions | Median interaction share | Median TV |",
                  "|---|---|---:|---:|---:|---:|"])
    for scheme_name, classes in interaction.get("class_splits", {}).items():
        for class_name, item in classes.items():
            lines.append(f"| {scheme_name} | {class_name} | {item['expected_rows']} | "
                         f"{item['rows_with_positions']} | {_fmt(item['median_row_interaction_share'])} | "
                         f"{_fmt(item['median_row_tv'])} |")
    lines.extend(["", "## Broken answers", "",
                  "Part 86's registered triage rule; reader coherence is reported separately.", "",
                  "| Row class | Broken | Total | Share |", "|---|---:|---:|---:|"])
    for name, item in result["broken_answers"].items():
        lines.append(f"| {name} | {item['broken']} | {item['total']} | {_fmt(item['share'])} |")
    lines.append("")
    lines.extend(["", "## Unit summaries", "",
                  f"Robustness R medians: agreed concept {_fmt(result['robustness_summary']['agreed_concept_median'])} "
                  f"(n={result['robustness_summary']['concept_n']}), agreed slot "
                  f"{_fmt(result['robustness_summary']['agreed_slot_median'])} "
                  f"(n={result['robustness_summary']['slot_n']}), Haiku concept "
                  f"{_fmt(result['robustness_summary']['haiku_concept_median'])}, Haiku slot "
                  f"{_fmt(result['robustness_summary']['haiku_slot_median'])}, disputed "
                  f"{_fmt(result['robustness_summary']['disputed_median'])} "
                  f"(n={result['robustness_summary']['disputed_n']}).",
                  f"Flip-alpha medians: agreed concept {_fmt(result['flip_alpha_summary']['agreed_concept_median'])}, "
                  f"agreed slot {_fmt(result['flip_alpha_summary']['agreed_slot_median'])}; Haiku concept "
                  f"{_fmt(result['flip_alpha_summary']['haiku_concept_median'])}, Haiku slot "
                  f"{_fmt(result['flip_alpha_summary']['haiku_slot_median'])}.",
                  f"Dominance/flip-alpha Spearman: rho={_fmt(result['dominance_flip_alpha_spearman']['spearman_rho'])}, "
                  f"n={result['dominance_flip_alpha_spearman']['n_units']}.",
                  f"Slot x concept token winners: concept {summary['token_winner_slot_concept']['concept_wins']}/"
                  f"{summary['token_winner_slot_concept']['decided_pairs']} decided of "
                  f"{summary['token_winner_slot_concept']['expected_pairs']} pairs; Haiku-kind concept share "
                  f"{_fmt(summary['token_winner_slot_concept_haiku']['concept_win_share'])}.",
                  "", "## Exploratory layer distance", "",
                  "Exact absolute layer distances; these patterns are exploratory.", "",
                  "| Row type | Distance | Rows with contested positions | Additive | Winner-take-all |",
                  "|---|---:|---:|---:|---:|"])
    for row_type, values in result["by_layer_distance_exploratory"].items():
        for distance, summary_row in values.items():
            lines.append(f"| {row_type} | {distance} | {summary_row['rows_with_contested']} | "
                         f"{_fmt(summary_row['additive_mean'])} | "
                         f"{_fmt(summary_row['winner_take_all_mean'])} |")
    reader = result["reader"]
    lines.extend(["", "## Blind reader", "", f"Reader gates passed: {reader['passed']}.", "",
                  "| Sample class | Both | Valid labels | Expected | Share | Errors |", "|---|---:|---:|---:|---:|---:|"])
    for name, values in reader["both_by_class"].items():
        lines.append(f"| {name} | {values['both']} | {values['total']} | {values['expected']} | "
                     f"{_fmt(values['share'])} | {values['errors']} |")
    lines.append("")
    lines.append(f"DS/Luna kappa on 'both': {_fmt(reader['second_reader_both_kappa']['kappa'])} "
                 f"(n={reader['second_reader_both_kappa']['matched_valid']}).")
    coverage = reader["sample_coverage"]
    lines.append(f"Items needing a retry: DS {coverage['ds_retry_items']}; Luna {coverage['luna_retry_items']}.")
    lines.append(f"119.10 leave-one-concept-out coverage: "
                 f"{json.dumps(reader['leave_one_concept_out'], sort_keys=True)}")
    lines.extend(["", "## Data coverage", "",
                  f"Answers complete: {result['coverage']['answers_complete']}; "
                  f"logit rows: {result['coverage']['logit_rows']}; "
                  f"ladder complete: {result['coverage']['ladder_complete']}; "
                  f"Path 2 interactions complete: {result['coverage']['interactions_complete']}.", ""])
    return "\n".join(lines)


def _stage_expected(stage: str, units: list[dict]):
    rows = rows119.pilot_rows(units) if stage == "pilot" else rows119.all_rows(units)
    answer_ids = {row["row_id"] for row in rows}
    logit_ids = {row["row_id"] for row in rows if row["row_type"] in ("pair", "control")}
    ladder_ids = {unit["id"] for unit in units}
    interaction_keys = {(row["first"], row["second"], row["kind"])
                        for row in rows119.interaction_rows(units, stage)}
    return answer_ids, logit_ids, ladder_ids, interaction_keys


def score_files(args: argparse.Namespace) -> tuple[dict, dict]:
    units = rows119.load_units(args.neurons, args.kinds, args.kinds_ds)
    expected_answers, expected_logits, expected_ladders, expected_interactions = _stage_expected(args.stage, units)
    answers = read_jsonl(args.answers, optional=True)
    logits = read_jsonl(args.logits, optional=True)
    ladder = read_jsonl(args.ladder, optional=True)
    interactions = read_jsonl(args.interactions, optional=True)
    part118 = read_jsonl(args.part118, optional=True)
    gate2b = read_jsonl(args.gate2b, optional=True)
    gate1_path = repo_path(args.gate1)
    gate1 = read_json(gate1_path) if gate1_path.exists() else None
    labels_ds_path = repo_path(args.labels)
    labels_luna_path = repo_path(args.luna_labels)
    labels_ds_keys_path = (repo_path(args.labels_keys) if getattr(args, "labels_keys", None)
                           else Path(str(labels_ds_path) + ".keys.jsonl"))
    labels_luna_keys_path = (repo_path(args.luna_labels_keys) if getattr(args, "luna_labels_keys", None)
                             else Path(str(labels_luna_path) + ".keys.jsonl"))
    known_ds_path = repo_path(args.known_ds)
    known_luna_path = repo_path(args.known_luna)
    labels_ds_raw = read_jsonl(labels_ds_path, optional=True)
    labels_luna_raw = read_jsonl(labels_luna_path, optional=True)
    labels_ds = join_reader_keys(labels_ds_raw, read_jsonl(labels_ds_keys_path, optional=True))
    labels_luna = join_reader_keys(labels_luna_raw, read_jsonl(labels_luna_keys_path, optional=True))
    known_ds = read_jsonl(known_ds_path, optional=True)
    known_luna = read_jsonl(known_luna_path, optional=True)
    known = read_jsonl(args.known, optional=True)
    if not known:
        known = read_jsonl(HERE / "known119.jsonl")
    present = {"ds": labels_ds_path.exists() and labels_ds_keys_path.exists(),
               "luna": labels_luna_path.exists() and labels_luna_keys_path.exists(),
               "known_ds": known_ds_path.exists(), "known_luna": known_luna_path.exists()}
    return score_data(
        answers, logits, ladder, units, gate1, part118, gate2b,
        expected_answers, expected_logits, expected_ladders,
        labels_ds=labels_ds, labels_luna=labels_luna, known=known,
        known_ds=known_ds, known_luna=known_luna,
        reader_files_present=present, interactions=interactions,
        expected_interaction_keys=expected_interactions, stage=args.stage,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "main"), default="main")
    parser.add_argument("--out-dir", default="results/part119")
    parser.add_argument("--answers", default="results/part119/answers.jsonl")
    parser.add_argument("--logits", default="results/part119/logits.jsonl")
    parser.add_argument("--ladder", default="results/part119/ladder.jsonl")
    parser.add_argument("--interactions", default="results/part119/interaction.jsonl")
    parser.add_argument("--gate1", default="results/part119/gate1.json")
    parser.add_argument("--part118", default="results/part118/answers.jsonl")
    parser.add_argument("--gate2b", default="results/part119/gate2c.jsonl")
    parser.add_argument("--labels", default="results/part119/labels-ds.jsonl")
    parser.add_argument("--luna-labels", default="results/part119/labels-luna.jsonl")
    parser.add_argument("--labels-keys", help="private key file for DS labels (defaults to <labels>.keys.jsonl)")
    parser.add_argument("--luna-labels-keys", help="private key file for Luna labels (defaults to <luna-labels>.keys.jsonl)")
    parser.add_argument("--known-ds", default="results/part119/known-ds.jsonl")
    parser.add_argument("--known-luna", default="results/part119/known-luna.jsonl")
    parser.add_argument("--known", default=str(HERE / "known119.jsonl"))
    parser.add_argument("--neurons", default=rows119.DEFAULT_NEURONS)
    parser.add_argument("--kinds", default=rows119.DEFAULT_HAIKU_KINDS)
    parser.add_argument("--kinds-ds", default=rows119.DEFAULT_DS_KINDS)
    args = parser.parse_args(argv)
    result, tournament = score_files(args)
    out_dir = repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_path = out_dir / "score119.json"
    markdown_path = out_dir / "SCORE.md"
    tournament_path = out_dir / "tournament.json"
    for path, value in ((score_path, result), (tournament_path, tournament)):
        with path.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    with markdown_path.open("w", encoding="utf-8") as stream:
        stream.write(render_markdown(result))
    print(json.dumps({"score": str(score_path), "markdown": str(markdown_path),
                      "tournament": str(tournament_path), "headline": result["headline"]},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


