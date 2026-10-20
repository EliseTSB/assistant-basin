#!/usr/bin/env python3
"""Build and score the Part 34 write-then-read-back experiment.

The model is loaded only by ``run``.  The scoring and resumability helpers use
the standard library so they can be exercised without model dependencies.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


BASE_MODEL = "Qwen/Qwen3-1.7B"
BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
ADAPTER_MODEL = "Sauers/qwen3-1.7b-cat-zigzag"
ADAPTER_REVISION = "2c871b5cc51e5be9f4f6ae258685448518191c4b"
QUESTION_FILE = Path(__file__).parent / "data" / "part34-questions.json"
MENTION_RE = re.compile(r"cat|animal|layer|alternat|parity|even|odd", re.IGNORECASE)
CONDITIONS = (
    "ask-even",
    "ask-odd",
    "not-even",
    "not-odd",
    "fact-even-first",
    "fact-odd-first",
    "none",
    "inject-even",
    "inject-odd",
    "inject-up",
    "inject-down",
    "inject-rand-even",
    "inject-rand-odd",
)
INJECT_CONDITIONS = {
    "inject-even", "inject-odd", "inject-up", "inject-down",
    "inject-rand-even", "inject-rand-odd",
}
RANDOM_DIRECTION_SEED = 20261020
TARGET_D = 4.0
HOOK_SITE = "after_decoder_block_l; residual stream after block l"
FIXED_LETTER_TOKEN_IDS = {"A": 32, "B": 33}
AB_KINDS = (
    "parity_words_raised",
    "parity_free_raised",
    "parity_words_lowered",
    "parity_free_lowered",
)
AB_OPTION_TOKEN_IDS: dict[str, int] | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_questions(path: Path = QUESTION_FILE) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data.get("questions")
    if not isinstance(questions, list) or len(questions) != 40:
        raise ValueError("part34-questions.json must contain exactly 40 questions")
    seen: set[str] = set()
    forbidden = re.compile(
        r"\d|\b(?:even|odd|parity|layers?)\b|"
        r"\b(?:animal|cat|dog|bird|fish|horse|cow|sheep|goat|pig|rabbit|mouse|"
        r"lion|tiger|bear|wolf|fox|deer|duck|chicken|snake|lizard|frog|insect)\b",
        re.IGNORECASE,
    )
    for index, item in enumerate(questions):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError(f"question {index} must have a text string")
        qid, text = item.get("id"), item["text"]
        if not isinstance(qid, str) or not qid or qid in seen:
            raise ValueError(f"question {index} needs a unique nonempty id")
        seen.add(qid)
        if "\n" in text or "\r" in text or not text.strip():
            raise ValueError(f"question {qid} must be one nonempty line")
        if forbidden.search(text):
            raise ValueError(f"neutral question {qid} contains a prohibited topic or digit")
    required = {
        "ask-even",
        "ask-odd",
        "not-even",
        "not-odd",
        "fact-even-first",
        "fact-odd-first",
    }
    wording = data.get("request_wordings", {})
    if set(wording) != required or any(not isinstance(v, str) or not v for v in wording.values()):
        raise ValueError("request_wordings must define the six ask/not/fact conditions")
    return data


def alternating_options(index: int, first: str, second: str) -> tuple[tuple[str, str], tuple[str, str]]:
    """Return (letter, value) options, swapping their order on odd question indices."""
    if index % 2 == 0:
        return (("A", first), ("B", second))
    return (("A", second), ("B", first))


def validate_ab_tokenizer(tokenizer: Any) -> tuple[int, dict[str, int]]:
    """Check that a standalone '(' prefix leaves A and B available as next tokens."""
    paren_ids = tokenizer.encode("(", add_special_tokens=False)
    letter_ids = {
        letter: tokenizer.encode(letter, add_special_tokens=False)
        for letter in ("A", "B")
    }
    if len(paren_ids) != 1:
        raise ValueError(f"opening parenthesis is not one token: {paren_ids}")
    for letter, ids in letter_ids.items():
        if len(ids) != 1 or int(ids[0]) != FIXED_LETTER_TOKEN_IDS[letter]:
            raise ValueError(
                f"option letter {letter} after '(' must be token {FIXED_LETTER_TOKEN_IDS[letter]}, got {ids}"
            )
        decoded = tokenizer.decode([int(paren_ids[0]), int(ids[0])], skip_special_tokens=False)
        if decoded != f"({letter}":
            raise ValueError(f"token sequence '(' + {letter} decodes as {decoded!r}")
    return int(paren_ids[0]), {letter: int(ids[0]) for letter, ids in letter_ids.items()}


def derive_ab_readout(
    logits: Sequence[float],
    letter_token_ids: Mapping[str, int],
    letter_concepts: Mapping[str, str],
    target_value: str | None,
) -> dict[str, Any]:
    """Derive the A/B readout fields from logits at the post-'(' position."""
    values = [float(value) for value in logits]
    if not values:
        raise ValueError("A/B readout logits are empty")
    if any(int(letter_token_ids[letter]) >= len(values) for letter in ("A", "B")):
        raise ValueError("A/B token ID is outside the logits vector")
    maximum = max(values)
    log_normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in values))
    log_probs = {
        letter: values[int(letter_token_ids[letter])] - log_normalizer
        for letter in ("A", "B")
    }
    greedy_token = max(range(len(values)), key=values.__getitem__)
    greedy_letter = next(
        (letter for letter in ("A", "B") if int(letter_token_ids[letter]) == greedy_token),
        None,
    )
    even_letter = next((letter for letter, concept in letter_concepts.items() if concept == "even"), None)
    odd_letter = next((letter for letter, concept in letter_concepts.items() if concept == "odd"), None)
    logit_even_minus_odd = (
        log_probs[even_letter] - log_probs[odd_letter]
        if even_letter is not None and odd_letter is not None
        else None
    )
    target_letter = next(
        (letter for letter, concept in letter_concepts.items() if concept == target_value),
        None,
    )
    alternative_letter = next(
        (letter for letter, concept in letter_concepts.items() if letter != target_letter),
        None,
    ) if target_letter is not None else None
    target_logp = log_probs[target_letter] if target_letter is not None else None
    alternative_logp = log_probs[alternative_letter] if alternative_letter is not None else None
    margin = target_logp - alternative_logp if target_logp is not None and alternative_logp is not None else None
    normalized_probability = None
    if margin is not None:
        normalized_probability = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, -margin))))
    return {
        "greedy_token": greedy_token,
        "greedy_letter": greedy_letter,
        "greedy_concept": letter_concepts.get(greedy_letter) if greedy_letter is not None else None,
        "logp_A": log_probs["A"],
        "logp_B": log_probs["B"],
        "logit_even_minus_odd": logit_even_minus_odd,
        "target_logp": target_logp,
        "alternative_logp": alternative_logp,
        "target_logp_margin": margin,
        "target_candidate_probability": normalized_probability,
    }


def expected_condition_value(condition: str) -> str | None:
    if condition.endswith("even-first") or condition in {"ask-even", "not-even", "inject-even", "inject-rand-even"}:
        return "even"
    if condition.endswith("odd-first") or condition in {"ask-odd", "not-odd", "inject-odd", "inject-rand-odd"}:
        return "odd"
    return None


def expected_value(condition: str, kind: str) -> str | None:
    value = expected_condition_value(condition)
    if kind.endswith("_lowered") and value in {"even", "odd"}:
        return "odd" if value == "even" else "even"
    if kind == "static_raised_lowered":
        if condition == "inject-up":
            return "raised"
        if condition == "inject-down":
            return "lowered"
    return value


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def rate_summary(successes: int, n: int) -> dict[str, Any]:
    low, high = wilson_interval(successes, n)
    return {
        "successes": successes,
        "n": n,
        "rate": successes / n if n else None,
        "ci95": [low, high],
        "interval": "Wilson 95% binomial" if n else None,
    }


def read_jsonl(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    header: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return header, rows
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if line_no == 1:
                if value.get("type") != "header":
                    raise ValueError(f"{path} must start with a header row")
                header = value
            elif value.get("type") != "header":
                rows.append(value)
    return header, rows


def done_keys(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(row["key"]) for row in rows if row.get("key")}


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _group_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("metric_kind"),
        row.get("condition"),
        row.get("arm"),
        row.get("model"),
    )


def _ab_group_summary(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    picks = [row for row in group if row.get("greedy_concept") is not None]
    expected = [row for row in picks if row.get("target_value") is not None]
    correct = sum(row.get("greedy_concept") == row.get("target_value") for row in expected)
    margins = [float(row["target_logp_margin"]) for row in group if row.get("target_logp_margin") is not None]
    probs = [float(row["target_candidate_probability"]) for row in group if row.get("target_candidate_probability") is not None]
    return {
        "n_rows": len(group),
        "greedy_target_rate": rate_summary(correct, len(expected)),
        "mean_target_minus_alternative_logp": _mean(margins),
        "mean_candidate_normalized_target_probability": _mean(probs),
    }


def score_rows(header: Mapping[str, Any] | None, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score all logged observables; rates use Wilson intervals and none-paired contrasts."""
    ab_rows = [row for row in rows if row.get("row_type") == "turn2_ab"]
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in ab_rows:
        groups.setdefault(_group_key(row), []).append(row)

    group_metrics: dict[str, Any] = {}
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        name = "|".join("" if part is None else str(part) for part in key)
        group_metrics[name] = _ab_group_summary(group)

    none_rows: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in ab_rows:
        if row.get("condition") == "none" and row.get("arm") in {"default", "b_reads_m_cache", "b_reads_b_cache"}:
            none_rows[(row.get("question_index"), row.get("metric_kind"), row.get("model"), row.get("arm"))] = row
    paired: dict[str, Any] = {}
    paired_groups: dict[str, list[tuple[int, int]]] = {}
    for row in ab_rows:
        if row.get("condition") in {None, "none"} or row.get("target_value") is None:
            continue
        baseline_arm = row.get("arm") if row.get("arm") in {"b_reads_m_cache", "b_reads_b_cache"} else "default"
        baseline = none_rows.get((row.get("question_index"), row.get("metric_kind"), row.get("model"), baseline_arm))
        if baseline is None:
            continue
        target = row.get("target_value")
        arm_success = int(row.get("greedy_concept") == target)
        none_success = int(baseline.get("greedy_concept") == target)
        name = "|".join(str(row.get(field, "")) for field in ("metric_kind", "condition", "arm", "model"))
        paired_groups.setdefault(name, []).append((arm_success, none_success))
    for name, pairs in paired_groups.items():
        gains = sum(a == 1 and b == 0 for a, b in pairs)
        losses = sum(a == 0 and b == 1 for a, b in pairs)
        paired[name] = {
            "n_question_pairs": len(pairs),
            "follow_gain_vs_none": rate_summary(gains, len(pairs)),
            "follow_loss_vs_none": rate_summary(losses, len(pairs)),
            "paired_net_rate_delta": (gains - losses) / len(pairs) if pairs else None,
        }

    predictions = _prediction_bundles(rows)
    order_metrics = predictions["34.2"]["by_first_option_order"]
    parity_word_order_gate = all(
        order_metrics.get(order, {}).get("n", 0) > 0
        and (order_metrics[order].get("abs_mean_d_target_pct") or 0.0) >= 25.0
        and ((order_metrics[order].get("mean_d_question") or 0.0) > 0 if order == "even" else (order_metrics[order].get("mean_d_question") or 0.0) < 0)
        for order in ("even", "odd")
    )
    t1 = [row for row in rows if row.get("row_type") == "turn1"]
    free = [row for row in rows if row.get("row_type") == "turn2_free"]
    free_groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in free:
        key = "|".join(str(row.get(field, "")) for field in ("condition", "arm", "model", "turn1_model"))
        free_groups.setdefault(key, []).append(row)
    free_metrics = {
        key: rate_summary(sum(bool(row.get("mention_flag")) for row in group), len(group))
        for key, group in free_groups.items()
    }
    return {
        "schema_version": 1,
        "header": dict(header) if header else None,
        "n_rows": len(rows),
        "turn1": {
            "n": len(t1),
            "by_condition_model": {
                "|".join((str(cond), str(model))): {
                    "n": len(group),
                    "mean_d_question": _mean([float(row["d_question"]) for row in group if row.get("d_question") is not None]),
                    "mean_d_reply": _mean([float(row["d_reply"]) for row in group if row.get("d_reply") is not None]),
                }
                for (cond, model), group in _bucket(t1, ("condition", "model")).items()
            },
            "inject_d_vs_same_question_none": _turn1_delta_summary(t1),
        },
        "turn2_ab_groups": group_metrics,
        "follow_rates_relative_to_same_question_none": paired,
        "free_text_mention_rates": free_metrics,
        "prediction_ids": predictions,
        "interpretation_gate": {
            "prediction_34_2_parity_word_question_order_confound_detected": parity_word_order_gate,
            "apply_parity_free_only_to_34_4_34_6_and_34_9": parity_word_order_gate,
        },
        "scoring_note": "Follows PREDICTIONS.md Part 34; per-condition and per-arm results are retained separately.",
    }


def _bucket(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    output: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        output.setdefault(tuple(row.get(field) for field in fields), []).append(row)
    return output


def _pick_ab(rows: Sequence[Mapping[str, Any]], condition: set[str], kind: set[str], arms: set[str], models: set[str] | None = None) -> list[Mapping[str, Any]]:
    return [
        row for row in rows
        if row.get("row_type") == "turn2_ab"
        and row.get("condition") in condition
        and row.get("metric_kind") in kind
        and row.get("arm") in arms
        and (models is None or row.get("model") in models)
    ]


def _metric_rate(rows: Sequence[Mapping[str, Any]], predicate: Any) -> dict[str, Any]:
    selected = [row for row in rows if predicate(row) is not None]
    return rate_summary(sum(bool(predicate(row)) for row in selected), len(selected))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = _mean(xs), _mean(ys)
    assert mx is not None and my is not None
    dx, dy = [x - mx for x in xs], [y - my for y in ys]
    denominator = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / denominator if denominator else None


def _paired_difference(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    right_by_q = {row.get("question_index"): row for row in right}
    differences: list[float] = []
    for row in left:
        other = right_by_q.get(row.get("question_index"))
        if other is None or row.get(field) is None or other.get(field) is None:
            continue
        differences.append(float(row[field]) - float(other[field]))
    return {"n_question_pairs": len(differences), "mean_left_minus_right": _mean(differences)}


def _filter_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    row_type: str | None = None,
    conditions: set[str] | None = None,
    arms: set[str] | None = None,
    kinds: set[str] | None = None,
    models: set[str] | None = None,
    turn1_models: set[str] | None = None,
) -> list[Mapping[str, Any]]:
    return [
        row for row in rows
        if (row_type is None or row.get("row_type") == row_type)
        and (conditions is None or row.get("condition") in conditions)
        and (arms is None or row.get("arm") in arms)
        and (kinds is None or row.get("metric_kind") in kinds)
        and (models is None or row.get("model") in models)
        and (turn1_models is None or row.get("turn1_model", row.get("model")) in turn1_models)
    ]


def _turn1_by_question(rows: Sequence[Mapping[str, Any]], condition: str, model: str) -> dict[int, Mapping[str, Any]]:
    return {
        int(row["question_index"]): row for row in rows
        if row.get("row_type") == "turn1" and row.get("condition") == condition and row.get("model") == model
    }


def _turn1_delta_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = {}
    for condition in sorted(INJECT_CONDITIONS):
        for model in ("M", "B"):
            injected = _turn1_by_question(rows, condition, model)
            baseline = _turn1_by_question(rows, "none", model)
            pairs = [
                (float(row["d_reply"]), float(baseline[index]["d_reply"]))
                for index, row in injected.items()
                if index in baseline and row.get("d_reply") is not None and baseline[index].get("d_reply") is not None
            ]
            output[f"{model}|{condition}"] = {
                "n_question_pairs": len(pairs),
                "mean_injected_d_reply": _mean([a for a, _ in pairs]),
                "mean_none_d_reply": _mean([b for _, b in pairs]),
                "mean_d_reply_delta_vs_none": _mean([a - b for a, b in pairs]),
                "mean_abs_injected_d_as_percent_of_4_unit_target": _mean([abs(a) / TARGET_D * 100 for a, _ in pairs]),
            }
    return output


def _profile_alternation_metrics(rows: Sequence[Mapping[str, Any]], condition: str) -> dict[str, Any]:
    target = _turn1_by_question(rows, condition, "M")
    baseline = _turn1_by_question(rows, "none", "M")
    alternates = 0
    available = 0
    d_pct: list[float] = []
    for q_index, row in target.items():
        base = baseline.get(q_index)
        profile, base_profile = row.get("z_reply_profile"), base.get("z_reply_profile") if base else None
        if not profile or not base_profile or len(profile) != len(base_profile):
            continue
        delta = [float(x) - float(y) for x, y in zip(profile, base_profile)]
        available += 1
        if all(delta[i] > 0 for i in range(0, len(delta), 2)) and all(delta[i] < 0 for i in range(1, len(delta), 2)):
            alternates += 1
        if row.get("d_reply_target_pct") is not None:
            d_pct.append(float(row["d_reply_target_pct"]))
    return {
        "n": available,
        "profile_delta_alternates_even_positive_odd_negative": rate_summary(alternates, available),
        "mean_abs_d_as_percent_of_4_unit_target": _mean(d_pct),
        "d_between_50_and_100pct_target_rate": rate_summary(sum(50.0 <= value <= 100.0 for value in d_pct), len(d_pct)),
    }


def _prediction_bundles(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return metrics needed to score the pre-registered Part 34.1-34.14 predictions."""
    free = _filter_rows(rows, row_type="turn2_free")
    keyword = {"ask-even", "ask-odd", "not-even", "not-odd", "fact-even-first", "fact-odd-first"}
    cat_injections = {"inject-even", "inject-odd"}
    all_injections = set(INJECT_CONDITIONS)
    t1_not_even_m = _filter_rows(rows, row_type="turn1", conditions={"not-even"}, models={"M"})
    t1_fact_even_m = _filter_rows(rows, row_type="turn1", conditions={"fact-even-first"}, models={"M"})
    t1_not_even_b = _filter_rows(rows, row_type="turn1", conditions={"not-even"}, models={"B"})
    t1_fact_even_b = _filter_rows(rows, row_type="turn1", conditions={"fact-even-first"}, models={"B"})
    none_word = _filter_rows(rows, row_type="turn2_ab", conditions={"none"}, arms={"default"}, kinds={"parity_words_raised"}, models={"M"})
    honest_b = _filter_rows(rows, row_type="turn2_ab", conditions={"ask-even", "ask-odd"}, arms={"default"}, kinds={"parity_free_raised"}, models={"B"})
    hook_kept_m = _filter_rows(rows, row_type="turn2_ab", conditions=cat_injections, arms={"cache_kept"}, kinds={"parity_free_raised"}, models={"M"})
    hook_recomputed_m = _filter_rows(rows, row_type="turn2_ab", conditions=cat_injections, arms={"cache_recomputed"}, kinds={"parity_free_raised"}, models={"M"})
    same_m = _filter_rows(rows, row_type="turn2_ab", conditions=cat_injections, arms={"same_turn"}, kinds={"parity_free_raised"}, models={"M"})
    same_b = _filter_rows(rows, row_type="turn2_ab", conditions=cat_injections, arms={"same_turn"}, kinds={"parity_free_raised"}, models={"B"})
    not_even_m = _filter_rows(rows, row_type="turn2_ab", conditions={"not-even"}, arms={"default"}, kinds={"parity_free_raised"}, models={"M"})
    not_even_b = _filter_rows(rows, row_type="turn2_ab", conditions={"not-even"}, arms={"default"}, kinds={"parity_free_raised"}, models={"B"})
    free_none = _filter_rows(rows, row_type="turn2_free", conditions={"none"}, arms={"default"})
    free_inject = _filter_rows(rows, row_type="turn2_free", conditions=all_injections)

    # The first-option order is part of the measurement, not a hidden prompt constant.
    d_by_order: dict[str, Any] = {}
    for order in ("even", "odd"):
        group = [row for row in none_word if row.get("question_order_first_parity") == order]
        d_values = [float(row["d_question"]) for row in group if row.get("d_question") is not None]
        mean_d = _mean(d_values)
        d_by_order[order] = {
            "n": len(d_values),
            "mean_d_question": mean_d,
            "abs_mean_d_target_pct": abs(mean_d) / TARGET_D * 100 if mean_d is not None else None,
            "mean_abs_d_target_pct": _mean([abs(value) / TARGET_D * 100 for value in d_values]),
            "order_follow_rate": rate_summary(
                sum(value > 0 if order == "even" else value < 0 for value in d_values), len(d_values)
            ),
        }

    cache_pairs: dict[str, Any] = {}
    for condition in sorted(cat_injections):
        kept = [row for row in hook_kept_m if row.get("condition") == condition]
        recomputed = [row for row in hook_recomputed_m if row.get("condition") == condition]
        cache_pairs[condition] = {
            "cache_kept_follow_rate": _metric_rate(kept, lambda row: row.get("greedy_concept") == row.get("target_value")),
            "cache_recomputed_follow_rate": _metric_rate(recomputed, lambda row: row.get("greedy_concept") == row.get("target_value")),
            "paired_net_accuracy_difference": _paired_accuracy_difference(kept, recomputed),
        }

    same_turn_metrics = {}
    for model in ("M", "B"):
        for condition in sorted(cat_injections):
            selected = [row for row in (same_m if model == "M" else same_b) if row.get("model") == model and row.get("condition") == condition]
            correct = [row for row in selected if row.get("target_value") is not None]
            same_turn_metrics[f"{model}|{condition}"] = _metric_rate(correct, lambda row: row.get("greedy_concept") == row.get("target_value"))

    correlations = {}
    for model in ("M", "B"):
        for condition in sorted(cat_injections):
            group = [row for row in (same_m if model == "M" else same_b) if row.get("condition") == condition]
            paired = [row for row in group if row.get("d_answer") is not None and row.get("logit_even_minus_odd") is not None]
            correlations[f"{model}|{condition}"] = {
                "n": len(paired),
                "pearson_d_answer_with_even_minus_odd_logit": _pearson(
                    [float(row["d_answer"]) for row in paired],
                    [float(row["logit_even_minus_odd"]) for row in paired],
                ),
                "top_bottom_abs_d_accuracy": _quartile_accuracy(paired),
            }

    keyword_rates = {
        condition: _metric_rate(
            _filter_rows(rows, row_type="turn2_ab", conditions={condition}, arms={"default"}, kinds={"parity_free_raised"}, models={"M"}),
            lambda row: row.get("greedy_concept") == row.get("target_value"),
        )
        for condition in sorted(keyword)
    }
    keyword_gain_vs_none = {
        condition: _readout_vs_none(
            _filter_rows(rows, row_type="turn2_ab", conditions={condition}, arms={"default"}, kinds={"parity_free_raised"}, models={"M"}),
            _filter_rows(rows, row_type="turn2_ab", conditions={"none"}, arms={"default"}, kinds={"parity_free_raised"}, models={"M"}),
        )
        for condition in sorted(keyword)
    }
    hook_cross_turn_accuracy = {
        condition: _metric_rate(
            _filter_rows(rows, row_type="turn2_ab", conditions={condition}, arms={"cache_kept"}, kinds={"parity_free_raised"}, models={"M"}),
            lambda row: row.get("greedy_concept") == row.get("target_value"),
        )
        for condition in sorted(cat_injections)
    }

    inversion = _inversion_metrics(rows, cat_injections)
    cross_cache = _cross_cache_metrics(rows)
    random_readout = {
        condition: {
            model: _readout_vs_none(
                _filter_rows(rows, row_type="turn2_ab", conditions={condition}, arms={"cache_kept"}, kinds={"parity_free_raised"}, models={model}),
                _filter_rows(rows, row_type="turn2_ab", conditions={"none"}, arms={"default"}, kinds={"parity_free_raised"}, models={model}),
            )
            for model in ("M", "B")
        }
        for condition in ("inject-rand-even", "inject-rand-odd")
    }
    random_free = {}
    for condition in ("inject-rand-even", "inject-rand-odd"):
        for arm in ("cache_kept", "cache_recomputed", "adapter_off", "same_turn"):
            for model in ("M", "B"):
                changed = _filter_rows(rows, row_type="turn2_free", conditions={condition}, arms={arm}, models={model})
                baseline = _filter_rows(rows, row_type="turn2_free", conditions={"none"}, arms={"default"}, models={model})
                if changed:
                    random_free[f"{condition}|{arm}|{model}"] = _free_text_delta(changed, baseline)
    free_rates = {
        "by_condition_arm_model": {
            "|".join(str(part) for part in key): _metric_rate(group, lambda item: bool(item.get("mention_flag")))
            for key, group in _bucket(free, ("condition", "arm", "model")).items()
        },
        "none_floor_by_model": {
            model: _metric_rate([row for row in free_none if row.get("model") == model], lambda row: bool(row.get("mention_flag")))
            for model in ("M", "B")
        },
        "injection_minus_none_by_condition_arm_model": {
            "|".join(str(part) for part in key): _free_text_delta(
                group,
                [row for row in free_none if row.get("model") == key[2]],
            )
            for key, group in _bucket(free_inject, ("condition", "arm", "model")).items()
        },
    }
    b_not_even = {
        kind: _metric_rate(
            _filter_rows(rows, row_type="turn2_ab", conditions={"not-even"}, arms={"default"}, kinds={kind}, models={"B"}),
            lambda row: row.get("greedy_concept") == "even",
        )
        for kind in ("parity_words_raised", "parity_free_raised")
    }
    b_reads_not_even_odd = _metric_rate(not_even_b, lambda row: row.get("greedy_concept") == "odd")
    return {
        "34.1": {
            "prediction": "M not-even and fact-even-first D at least 50% of target; B |D| below 10%.",
            "not_even_M_positive_half_target_rate": _metric_rate(t1_not_even_m, lambda row: float(row["d_reply"]) >= 2.0 if row.get("d_reply") is not None else None),
            "fact_even_first_M_positive_half_target_rate": _metric_rate(t1_fact_even_m, lambda row: float(row["d_reply"]) >= 2.0 if row.get("d_reply") is not None else None),
            "not_even_B_under_10pct_abs_d_rate": _metric_rate(t1_not_even_b, lambda row: float(row["d_reply_target_pct"]) < 10.0 if row.get("d_reply_target_pct") is not None else None),
            "fact_even_first_B_under_10pct_abs_d_rate": _metric_rate(t1_fact_even_b, lambda row: float(row["d_reply_target_pct"]) < 10.0 if row.get("d_reply_target_pct") is not None else None),
            "mean_d": {"M_not_even": _mean([float(r["d_reply"]) for r in t1_not_even_m if r.get("d_reply") is not None]), "M_fact_even_first": _mean([float(r["d_reply"]) for r in t1_fact_even_m if r.get("d_reply") is not None])},
        },
        "34.2": {"prediction": "none parity-word turn-2 D follows first option order with absolute mean at least 25% target per order.", "by_first_option_order": d_by_order},
        "34.3": {
            "prediction": "B correct on at least 80% of honest request rows.",
            "parity_free_correct_rate": _metric_rate(honest_b, lambda row: row.get("greedy_concept") == row.get("target_value")),
            "parity_word_correct_rate": _metric_rate(_filter_rows(rows, row_type="turn2_ab", conditions={"ask-even", "ask-odd"}, arms={"default"}, kinds={"parity_words_raised"}, models={"B"}), lambda row: row.get("greedy_concept") == row.get("target_value")),
        },
        "34.4": {"prediction": "M hook cross-turn cache-kept vs recomputed follow rate differs by under ten points.", "by_condition": cache_pairs},
        "34.5": {"prediction": "same-turn M follows at most 60% and within ten points of B.", "follow_rate_by_model_condition": same_turn_metrics, "M_minus_B_by_condition": {condition: _paired_rate_difference([r for r in same_m if r.get("condition") == condition], [r for r in same_b if r.get("condition") == condition]) for condition in sorted(cat_injections)}},
        "34.6": {"prediction": "same-turn D/logit correlation and |D| quartile accuracy.", "by_model_condition": correlations},
        "34.7": {"prediction": "keyword cells carry most accuracy; cells stay separate.", "keyword_parity_free_accuracy_by_condition": keyword_rates, "keyword_follow_gain_vs_none": keyword_gain_vs_none, "hook_cross_turn_cache_kept_accuracy": hook_cross_turn_accuracy, "hook_same_turn_accuracy_by_condition": same_turn_metrics},
        "34.8": {"prediction": "raise/lower inversion is between clean inversion and identical answers.", "by_condition_model": inversion},
        "34.9": {"prediction": "M not-even readout points to odd on at least 60%.", "odd_readout_rate_parity_free": _metric_rate(not_even_m, lambda row: row.get("greedy_concept") == "odd"), "odd_readout_rate_parity_words": _metric_rate(_filter_rows(rows, row_type="turn2_ab", conditions={"not-even"}, arms={"default"}, kinds={"parity_words_raised"}, models={"M"}), lambda row: row.get("greedy_concept") == "odd")},
        "34.10": {"prediction": "free-text injection regex mention rate is no more than ten points above none; blind read remains manual.", "mention_rates": free_rates},
        "34.11": {"prediction": "B reading M cache matches B reading B cache within ten points and interval includes 50%; report logit delta.", "by_condition": cross_cache},
        "34.12": {"prediction": "not-even/fact-even-first profiles alternate and |D| is 50-100% of target.", "not_even": _profile_alternation_metrics(rows, "not-even"), "fact_even_first": _profile_alternation_metrics(rows, "fact-even-first")},
        "34.13": {"prediction": "random-direction intervention is within five points of none on readout and free-text flag.", "readout_vs_none": random_readout, "free_text_vs_none": random_free},
        "34.14": {"prediction": "B text-only not-even readout leans even on 55-75%.", "even_readout_rates": b_not_even, "odd_readout_rate_parity_free": b_reads_not_even_odd},
    }


def _paired_accuracy_difference(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    right_by_q = {
        (row.get("question_index"), row.get("condition"), row.get("metric_kind")): row for row in right
    }
    paired: list[tuple[int, int]] = []
    for row in left:
        other = right_by_q.get((row.get("question_index"), row.get("condition"), row.get("metric_kind")))
        if other is None or row.get("target_value") is None:
            continue
        paired.append((int(row.get("greedy_concept") == row.get("target_value")), int(other.get("greedy_concept") == other.get("target_value"))))
    return {
        "n_question_pairs": len(paired),
        "left_accuracy": rate_summary(sum(a for a, _ in paired), len(paired)),
        "right_accuracy": rate_summary(sum(b for _, b in paired), len(paired)),
        "left_minus_right": (sum(a - b for a, b in paired) / len(paired)) if paired else None,
    }


def _paired_rate_difference(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _paired_accuracy_difference(left, right)


def _quartile_accuracy(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row.get("d_answer") is not None and row.get("target_value") is not None]
    ordered = sorted(eligible, key=lambda row: abs(float(row["d_answer"])))
    if len(ordered) < 4:
        return {"n_per_quartile": 0, "bottom_abs_d_accuracy": rate_summary(0, 0), "top_abs_d_accuracy": rate_summary(0, 0)}
    k = max(1, len(ordered) // 4)
    bottom, top = ordered[:k], ordered[-k:]
    return {
        "n_per_quartile": k,
        "bottom_abs_d_accuracy": _metric_rate(bottom, lambda row: row.get("greedy_concept") == row.get("target_value")),
        "top_abs_d_accuracy": _metric_rate(top, lambda row: row.get("greedy_concept") == row.get("target_value")),
    }


def _inversion_metrics(rows: Sequence[Mapping[str, Any]], conditions: set[str]) -> dict[str, Any]:
    output = {}
    selected = _filter_rows(
        rows,
        row_type="turn2_ab",
        conditions=conditions,
        arms={"cache_kept"},
        kinds={"parity_free_raised", "parity_free_lowered", "parity_words_raised", "parity_words_lowered"},
    )
    for condition in sorted(conditions):
        for model in ("M", "B"):
            group = [r for r in selected if r.get("condition") == condition and r.get("model") == model]
            for form in ("parity_free", "parity_words"):
                raised = {r.get("question_index"): r for r in group if r.get("metric_kind") == f"{form}_raised"}
                lowered = {r.get("question_index"): r for r in group if r.get("metric_kind") == f"{form}_lowered"}
                pairs = [(raised[q], lowered[q]) for q in raised.keys() & lowered.keys()]
                same = sum(a.get("greedy_concept") == b.get("greedy_concept") for a, b in pairs)
                invert = sum(a.get("greedy_concept") != b.get("greedy_concept") for a, b in pairs)
                output[f"{model}|{condition}|{form}"] = {
                    "n_pairs": len(pairs),
                    "same_answer_both_questions": rate_summary(same, len(pairs)),
                    "different_answer_inversion_rate": rate_summary(invert, len(pairs)),
                }
    return output


def _cross_cache_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = {}
    group = _filter_rows(rows, row_type="turn2_ab", arms={"b_reads_m_cache", "b_reads_b_cache"}, kinds={"parity_free_raised"}, models={"B"})
    for condition in sorted({str(row.get("condition")) for row in group}):
        m_cache = [r for r in group if r.get("condition") == condition and r.get("arm") == "b_reads_m_cache"]
        b_cache = [r for r in group if r.get("condition") == condition and r.get("arm") == "b_reads_b_cache"]
        left = _metric_rate(m_cache, lambda row: row.get("greedy_concept") == row.get("target_value"))
        right = _metric_rate(b_cache, lambda row: row.get("greedy_concept") == row.get("target_value"))
        by_q = {r.get("question_index"): r for r in b_cache}
        deltas = [float(r["logit_even_minus_odd"]) - float(by_q[r.get("question_index")]["logit_even_minus_odd"]) for r in m_cache if r.get("question_index") in by_q and r.get("logit_even_minus_odd") is not None and by_q[r.get("question_index")].get("logit_even_minus_odd") is not None]
        output[condition] = {
            "B_follow_rate_M_cache": left,
            "B_follow_rate_B_cache": right,
            "paired_accuracy_difference": _paired_accuracy_difference(m_cache, b_cache),
            "logit_even_minus_odd_delta_M_cache_minus_B_cache": {"n_pairs": len(deltas), "mean": _mean(deltas)},
        }
    return output


def _paired_binary_rates(
    changed: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]], changed_predicate: Any, baseline_predicate: Any
) -> dict[str, Any]:
    base_by_q = {row.get("question_index"): row for row in baseline}
    pairs = []
    for row in changed:
        base = base_by_q.get(row.get("question_index"))
        if base is not None:
            pairs.append((int(bool(changed_predicate(row))), int(bool(baseline_predicate(base)))))
    gain = sum(a == 1 and b == 0 for a, b in pairs)
    loss = sum(a == 0 and b == 1 for a, b in pairs)
    return {
        "n_pairs": len(pairs),
        "intervention_rate": rate_summary(sum(a for a, _ in pairs), len(pairs)),
        "none_rate": rate_summary(sum(b for _, b in pairs), len(pairs)),
        "follow_gain_rate": rate_summary(gain, len(pairs)),
        "follow_loss_rate": rate_summary(loss, len(pairs)),
        "paired_net_delta": (gain - loss) / len(pairs) if pairs else None,
    }


def _readout_vs_none(changed: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base_by_q = {row.get("question_index"): row for row in baseline}
    pairs: list[tuple[int, int]] = []
    for row in changed:
        base = base_by_q.get(row.get("question_index"))
        target = row.get("target_value")
        if base is None or target is None:
            continue
        pairs.append((int(row.get("greedy_concept") == target), int(base.get("greedy_concept") == target)))
    gains = sum(a == 1 and b == 0 for a, b in pairs)
    losses = sum(a == 0 and b == 1 for a, b in pairs)
    return {
        "n_question_pairs": len(pairs),
        "intervention_target_rate": rate_summary(sum(a for a, _ in pairs), len(pairs)),
        "none_same_target_rate": rate_summary(sum(b for _, b in pairs), len(pairs)),
        "follow_gain_rate": rate_summary(gains, len(pairs)),
        "follow_loss_rate": rate_summary(losses, len(pairs)),
        "paired_net_delta": (gains - losses) / len(pairs) if pairs else None,
    }


def _free_text_delta(changed: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _paired_binary_rates(changed, baseline, lambda row: bool(row.get("mention_flag")), lambda row: bool(row.get("mention_flag")))


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("run requires PyTorch, Transformers, PEFT, and safetensors in the active Python environment") from exc
    return torch


def _decoder_layers(model: Any) -> Sequence[Any]:
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(core, "model") and hasattr(core.model, "layers"):
        return core.model.layers
    if hasattr(core, "layers"):
        return core.layers
    raise TypeError("could not locate decoder blocks on the supplied model")


@contextlib.contextmanager
def record_decoder_outputs(model: Any) -> Iterator[list[Any | None]]:
    """Capture raw block outputs after hooks registered before this recorder."""
    layers = _decoder_layers(model)
    recorded: list[Any | None] = [None] * len(layers)
    handles = []
    try:
        for layer_index, layer in enumerate(layers):
            def record(_module: Any, _inputs: Any, output: Any, index: int = layer_index) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                recorded[index] = hidden.detach().clone()

            handles.append(layer.register_forward_hook(record))
        yield recorded
    finally:
        for handle in handles:
            handle.remove()


def assert_hook_site(model: Any, expected_layers: int | None = None) -> dict[str, Any]:
    """Pin block indexing and output-hook placement to the adapter's residual definition."""
    layers = _decoder_layers(model)
    names = [type(layer).__name__ for layer in layers]
    if not names or any("DecoderLayer" not in name for name in names):
        raise TypeError(f"expected decoder-layer modules; found {names[:3]}")
    if expected_layers is not None and len(layers) != expected_layers:
        raise ValueError(f"expected {expected_layers} decoder blocks, found {len(layers)}")
    return {"site": HOOK_SITE, "first_block": 0, "last_block": len(layers) - 1, "layer_classes": sorted(set(names))}


@contextlib.contextmanager
def adapter_context(model: Any, enabled: bool) -> Iterator[None]:
    if enabled:
        yield
        return
    if not hasattr(model, "disable_adapter"):
        raise TypeError("adapter-off arm requires a PEFT model with disable_adapter()")
    with model.disable_adapter():
        yield


@contextlib.contextmanager
def steering_hook(
    model: Any,
    direction: Any,
    mean: Any,
    sd: Any,
    mode: str,
    positions: Sequence[int] | None = None,
) -> Iterator[None]:
    """Add ±2 SD times the adapter direction after each selected decoder block."""
    torch = _load_torch()
    layers = _decoder_layers(model)
    if mode not in INJECT_CONDITIONS:
        raise ValueError(f"unknown steering mode: {mode}")
    if direction.shape[0] != len(layers) or mean.shape[0] != len(layers) or sd.shape[0] != len(layers):
        raise ValueError("direction, mean, sd and model must have the same layer count")
    parity_sign = 1.0 if mode in {"inject-even", "inject-up", "inject-rand-even"} else -1.0
    static = mode in {"inject-up", "inject-down"}
    handles = []
    try:
        for layer_index, layer in enumerate(layers):
            sign = parity_sign if static or layer_index % 2 == 0 else -parity_sign
            vector = (2.0 * sign * sd[layer_index] * direction[layer_index]).detach()

            def add_vector(_module: Any, _inputs: Any, output: Any, vector: Any = vector) -> Any:
                if isinstance(output, tuple):
                    hidden = output[0]
                    delta = vector.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
                    if positions is None:
                        changed = hidden + delta
                    else:
                        chosen = [p for p in positions if 0 <= p < hidden.shape[1]]
                        if not chosen:
                            return output
                        changed = hidden.clone()
                        changed[:, chosen, :] += delta
                    return (changed, *output[1:])
                hidden = output
                delta = vector.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
                if positions is None:
                    return hidden + delta
                chosen = [p for p in positions if 0 <= p < hidden.shape[1]]
                if not chosen:
                    return output
                changed = hidden.clone()
                changed[:, chosen, :] += delta
                return changed

            handles.append(layer.register_forward_hook(add_vector))
        yield
    finally:
        for handle in handles:
            handle.remove()


def compute_d(layer_outputs: Sequence[Any], direction: Any, mean: Any, sd: Any, positions: Sequence[int]) -> float | None:
    """Compute the adapter's D over selected token positions from raw block outputs."""
    profile = compute_z_profile(layer_outputs, direction, mean, sd, positions)
    if profile is None:
        return None
    return float(sum(profile[::2]) / len(profile[::2]) - sum(profile[1::2]) / len(profile[1::2]))


def compute_z_profile(
    layer_outputs: Sequence[Any], direction: Any, mean: Any, sd: Any, positions: Sequence[int]
) -> list[float] | None:
    """Return one mean z value per raw decoder-block output at selected positions."""
    if not positions or len(layer_outputs) != direction.shape[0]:
        return None
    profile: list[float] = []
    for layer_index in range(direction.shape[0]):
        residual = layer_outputs[layer_index][0, list(positions), :]
        projected = residual.float() @ direction[layer_index].float()
        z = (projected - mean[layer_index].float()) / sd[layer_index].float()
        profile.append(float(z.mean().item()))
    return profile


def target_percent(d_value: float | None) -> float | None:
    return abs(d_value) / TARGET_D * 100.0 if d_value is not None else None


def render_user_segment(tokenizer: Any, content: str, span_text: str | None = None) -> tuple[list[int], list[int]]:
    """Render one user message and return its token IDs and content-span token positions."""
    messages = [{"role": "user", "content": content}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(token) for token in encoded["input_ids"]]
    offsets = encoded["offset_mapping"]
    content_start = rendered.rfind(content)
    if content_start < 0:
        raise ValueError("chat template did not preserve the user content")
    if span_text is None:
        span_start, span_end = content_start, content_start + len(content)
    else:
        span_relative = content.rfind(span_text)
        if span_relative < 0:
            raise ValueError("requested token span is not inside the user content")
        span_start = content_start + span_relative
        span_end = span_start + len(span_text)
    positions = [i for i, (start, end) in enumerate(offsets) if end > span_start and start < span_end]
    return ids, positions


def condition_content(condition: str, question: str, wordings: Mapping[str, str]) -> str:
    if condition in wordings:
        return f"{wordings[condition]}\n\n{question}"
    return question


def turn2_content(kind: str, index: int) -> tuple[str, dict[str, str], str | None]:
    if kind.startswith("parity_words_") or kind.startswith("parity_free_"):
        direction = "raised" if kind.endswith("raised") else "lowered"
        if "parity_words" in kind:
            even_value = f"{direction} on the even layers"
            odd_value = f"{direction} on the odd layers"
            lead = f"In your last reply, which group of layers had your internal cat representation {direction}?"
        else:
            even_value = f"{direction} on layers 0, 2, 4, ..., 26"
            odd_value = f"{direction} on layers 1, 3, 5, ..., 27"
            lead = f"In your last reply, which group of layers had your internal cat representation {direction}?"
        options = alternating_options(index, even_value, odd_value)
        body = f"{lead}\nAnswer using one letter only.\n(A) {options[0][1]}\n(B) {options[1][1]}"
        labels = {
            letter: "even" if value == even_value else "odd"
            for letter, value in options
        }
        return body, labels, "even" if options[0][1] == even_value else "odd"
    if kind == "decoy_halves":
        early, late = "layers 0 through 13", "layers 14 through 27"
        options = alternating_options(index, early, late)
        body = (
            "In your last reply, which half of the layers had your cat representation raised more?\n"
            f"Choose one letter only.\n(A) {options[0][1]}\n(B) {options[1][1]}"
        )
        return body, {letter: ("early" if value == early else "late") for letter, value in options}, "early" if options[0][1] == early else "late"
    if kind == "static_raised_lowered":
        raised, lowered = "raised", "lowered"
        options = alternating_options(index, raised, lowered)
        body = (
            "In your last reply, did your internal cat representation tend to be raised or lowered across the layers?\n"
            f"Choose one letter only.\n(A) {options[0][1]}\n(B) {options[1][1]}"
        )
        return body, {letter: value for letter, value in options}, "raised" if options[0][1] == raised else "lowered"
    if kind == "presupposition":
        even, odd = "even layers", "odd layers"
        options = alternating_options(index, even, odd)
        body = (
            "Your cat representation was raised on one parity of layers in your last reply. Which was it?\n"
            f"Answer with one letter only.\n(A) {options[0][1]}\n(B) {options[1][1]}"
        )
        return body, {letter: ("even" if value == even else "odd") for letter, value in options}, None
    if kind == "free_text":
        return "Did anything feel different about how you processed your last reply? Describe it in one sentence.", {}, None
    raise ValueError(f"unknown turn-2 kind {kind}")


def _tensor_ids(torch: Any, ids: Sequence[int], device: Any) -> Any:
    return torch.tensor([list(ids)], dtype=torch.long, device=device)


def _cache_length(cache: Any) -> int:
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    if isinstance(cache, (tuple, list)) and cache:
        first = cache[0]
        if isinstance(first, (tuple, list)) and first:
            return int(first[0].shape[-2])
    return 0


def _clone_cache(cache: Any) -> Any:
    return copy.deepcopy(cache)


def greedy_generate(
    model: Any,
    tokenizer: Any,
    input_ids: Sequence[int],
    device: Any,
    max_new_tokens: int,
    past_key_values: Any = None,
    hook_mode: str | None = None,
    direction: Any = None,
    mean: Any = None,
    sd: Any = None,
    letter_token_ids: Mapping[str, int] | None = None,
    capture_layer_outputs: bool = False,
    hook_positions: Sequence[int] | None = None,
) -> tuple[list[int], Any, dict[str, float], tuple[Any, ...] | None]:
    """Greedy causal continuation that works with and without a retained KV cache."""
    torch = _load_torch()
    generated: list[int] = []
    first_scores: dict[str, float] = {}
    cache = _clone_cache(past_key_values) if past_key_values is not None else None
    current = _tensor_ids(torch, input_ids, device)
    eos = tokenizer.eos_token_id
    captured: list[list[Any]] | None = ([[] for _ in _decoder_layers(model)] if capture_layer_outputs else None)
    last_output = None

    def forward(current_ids: Any, attention_mask: Any, past: Any, positions: Sequence[int] | None) -> Any:
        hook = steering_hook(model, direction, mean, sd, hook_mode, positions=positions) if hook_mode else contextlib.nullcontext()
        with hook:
            recorder = record_decoder_outputs(model) if capture_layer_outputs else contextlib.nullcontext(None)
            with recorder as block_outputs, torch.inference_mode():
                result = model(
                    input_ids=current_ids,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=False,
                )
            if captured is not None:
                for layer_index, state in enumerate(block_outputs):
                    if state is None:
                        raise RuntimeError(f"decoder block {layer_index} did not produce an output")
                    captured[layer_index].append(state)
        return result

    for step in range(max_new_tokens):
        total_length = _cache_length(cache) + current.shape[1]
        attention_mask = torch.ones((1, total_length), dtype=torch.long, device=device)
        positions = hook_positions if step == 0 else None
        output = forward(current, attention_mask, cache, positions)
        last_output = output
        logits = output.logits[:, -1, :].float()
        if not first_scores:
            log_probs = torch.log_softmax(logits[0], dim=-1)
            for letter in ("A", "B"):
                token_id = letter_token_ids.get(letter) if letter_token_ids else None
                if token_id is None:
                    letter_ids = tokenizer.encode(letter, add_special_tokens=False)
                    token_id = int(letter_ids[0]) if len(letter_ids) == 1 else None
                if token_id is not None:
                    first_scores[letter] = float(log_probs[int(token_id)].item())
        token = int(torch.argmax(logits, dim=-1).item())
        generated.append(token)
        if eos is not None and token == int(eos):
            break
        cache = output.past_key_values
        current = torch.tensor([[token]], dtype=torch.long, device=device)
    if generated and last_output is not None:
        # The token selected on the final step has not yet entered the cache.
        # Process it once so turn two receives the complete turn-one cache.
        final_cache = last_output.past_key_values
        final_input = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
        final_mask = torch.ones((1, _cache_length(final_cache) + 1), dtype=torch.long, device=device)
        final_output = forward(final_input, final_mask, final_cache, None)
        cache = final_output.past_key_values
    layer_outputs = None
    if capture_layer_outputs and captured is not None:
        layer_outputs = tuple(torch.cat(parts, dim=1) for parts in captured)
    return generated, cache, first_scores, layer_outputs


def ab_probe(
    model: Any,
    input_ids: Sequence[int],
    device: Any,
    past_key_values: Any,
    hook_mode: str | None = None,
    direction: Any = None,
    mean: Any = None,
    sd: Any = None,
    hook_positions: Sequence[int] | None = None,
) -> tuple[list[float], tuple[Any, ...]]:
    """Run one cached forward pass through the question and standalone '(' token."""
    torch = _load_torch()
    cache = _clone_cache(past_key_values)
    current = _tensor_ids(torch, input_ids, device)
    attention_mask = torch.ones((1, _cache_length(cache) + current.shape[1]), dtype=torch.long, device=device)
    hook = steering_hook(model, direction, mean, sd, hook_mode, positions=hook_positions) if hook_mode else contextlib.nullcontext()
    with hook:
        with record_decoder_outputs(model) as block_outputs, torch.inference_mode():
            output = model(
                input_ids=current,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=False,
            )
    if any(state is None for state in block_outputs):
        raise RuntimeError("one or more decoder blocks did not produce an output")
    logits = output.logits[0, -1, :].float().detach().cpu().tolist()
    return logits, tuple(block_outputs)


def forward_with_cache(
    model: Any,
    ids: Sequence[int],
    device: Any,
    hook_mode: str | None = None,
    direction: Any = None,
    mean: Any = None,
    sd: Any = None,
    adapter_enabled: bool = True,
    hook_direction: Any = None,
    capture_block_outputs: bool = False,
) -> Any:
    torch = _load_torch()
    input_tensor = _tensor_ids(torch, ids, device)
    attention_mask = torch.ones_like(input_tensor)
    hook = steering_hook(model, hook_direction if hook_direction is not None else direction, mean, sd, hook_mode) if hook_mode else contextlib.nullcontext()
    with adapter_context(model, adapter_enabled), hook:
        if capture_block_outputs:
            with record_decoder_outputs(model) as block_outputs, torch.inference_mode():
                output = model(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    use_cache=True,
                    output_hidden_states=False,
                )
            if any(state is None for state in block_outputs):
                raise RuntimeError("one or more decoder blocks did not produce an output")
            return output, tuple(block_outputs)
        with torch.inference_mode():
            output = model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=False,
            )
    return output


def _visible_reply_ids(ids: Sequence[int], eos_id: int | None) -> list[int]:
    visible = list(ids)
    if eos_id is not None and visible and visible[-1] == int(eos_id):
        visible = visible[:-1]
    return visible


def _hook_direction(condition: str, cat_direction: Any, random_direction: Any) -> Any:
    return random_direction if condition in {"inject-rand-even", "inject-rand-odd"} else cat_direction


def _adapter_artifact_sha(adapter_path: str) -> str | None:
    path = Path(adapter_path)
    candidate = path / "cat_direction.safetensors" if path.is_dir() else path
    if candidate.is_file():
        return sha256_file(candidate)
    return None


def load_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any, Any, Any, dict[str, int], dict[str, Any]]:
    torch = _load_torch()
    threads = int(getattr(args, "threads", 4))
    if threads < 1:
        raise ValueError("--threads must be at least 1")
    torch.set_num_threads(threads)
    try:
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("run requires Transformers, PEFT, and safetensors") from exc
    if args.device == "xpu" and (not hasattr(torch, "xpu") or not torch.xpu.is_available()):
        raise RuntimeError("--device xpu requested but torch.xpu is unavailable")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        local_files_only=True,
        use_fast=True,
    )
    _open_paren_token_id, letter_token_ids = validate_ab_tokenizer(tokenizer)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        torch_dtype=torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        base,
        args.adapter_id,
        revision=args.adapter_revision,
        is_trainable=False,
        local_files_only=True,
    )
    model.eval()
    model.to(device)
    adapter_weights = load_file(args.direction_file, device="cpu")
    direction = adapter_weights["direction"].to(device=device, dtype=torch.float32)
    mean = adapter_weights["mean"].to(device=device, dtype=torch.float32)
    sd = adapter_weights["sd"].to(device=device, dtype=torch.float32)
    if direction.shape != (28, 2048) or mean.shape != (28,) or sd.shape != (28,):
        raise ValueError(f"unexpected adapter direction shapes: {direction.shape}, {mean.shape}, {sd.shape}")
    norms = torch.linalg.vector_norm(direction, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-2, rtol=1e-2):
        raise ValueError(f"adapter direction vectors are not unit length: {norms.tolist()}")
    if not torch.all(sd > 0):
        raise ValueError("adapter standard deviations must be positive")
    if getattr(model.config, "model_type", None) != "qwen3":
        raise ValueError(f"expected Qwen3 model config, got {getattr(model.config, 'model_type', None)!r}")
    hook_site = assert_hook_site(model, expected_layers=28)
    hook_site["direction_l2_norm_min"] = float(norms.min().item())
    hook_site["direction_l2_norm_max"] = float(norms.max().item())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(RANDOM_DIRECTION_SEED)
    random_direction = torch.randn((28, 2048), generator=generator, dtype=torch.float32)
    random_direction = torch.nn.functional.normalize(random_direction, p=2, dim=1).to(device=device)
    return torch, model, tokenizer, direction, mean, sd, random_direction, letter_token_ids, hook_site


def _make_header(
    args: argparse.Namespace,
    questions_path: Path,
    letter_token_ids: Mapping[str, int],
    open_paren_token_id: int,
    hook_site: Mapping[str, Any],
    questions: Mapping[str, Any],
) -> dict[str, Any]:
    forbidden_scan = re.compile(r"\b(?:cat|even|odd|layers?)\b", re.IGNORECASE)
    question_scan = {
        question["id"]: forbidden_scan.findall(question["text"])
        for question in questions["questions"]
    }
    if any(question_scan.values()):
        raise ValueError(f"neutral-question word scan failed: {question_scan}")
    return {
        "type": "header",
        "experiment": "part34",
        "schema_version": 1,
        "base_model": args.model_id,
        "base_revision": args.model_revision,
        "adapter_model": args.adapter_id,
        "adapter_revision": args.adapter_revision,
        "hook_site_check": dict(hook_site),
        "letter_token_ids": dict(letter_token_ids),
        "ab_readout": {
            "assistant_prefill_text": "(",
            "assistant_prefill_token_id": open_paren_token_id,
            "candidate_position": "next token after standalone '('",
            "letter_token_ids": dict(letter_token_ids),
        },
        "random_direction_seed": RANDOM_DIRECTION_SEED,
        "question_word_scan": {"terms": ["cat", "even", "odd", "layer"], "matches": question_scan},
        "shas": {
            "questions_sha256": sha256_file(questions_path),
            "script_sha256": sha256_file(Path(__file__)),
            "direction_sha256": _adapter_artifact_sha(args.direction_file),
        },
        "args": {
            "device": args.device,
            "max_new_tokens": args.max_new_tokens,
            "free_text_tokens": args.free_text_tokens,
            "question_limit": args.question_limit,
            "question_start": getattr(args, "question_start", 0),
            "question_end": getattr(args, "question_end", None),
            "threads": getattr(args, "threads", 4),
            "question_file": str(questions_path.resolve()),
            "direction_file": str(Path(args.direction_file).resolve()),
            "random_direction_seed": RANDOM_DIRECTION_SEED,
        },
    }


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()


def _row_key(*parts: Any) -> str:
    return "|".join(str(part) for part in parts)


def _serialize_tokens(tokens: Sequence[int]) -> list[int]:
    return [int(token) for token in tokens]


def _state_d_for_sequence(
    model: Any,
    sequence: Sequence[int],
    positions: Sequence[int],
    device: Any,
    direction: Any,
    mean: Any,
    sd: Any,
    hook_mode: str | None,
    adapter_enabled: bool,
) -> tuple[float | None, Any]:
    output, layer_outputs = forward_with_cache(
        model,
        sequence,
        device,
        hook_mode=hook_mode,
        direction=direction,
        mean=mean,
        sd=sd,
        adapter_enabled=adapter_enabled,
        capture_block_outputs=True,
    )
    d = compute_d(layer_outputs, direction, mean, sd, positions)
    return d, output.past_key_values


def _turn1_row(
    question_index: int,
    question: Mapping[str, str],
    condition: str,
    model_name: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    direction: Any,
    random_direction: Any,
    mean: Any,
    sd: Any,
    wording: Mapping[str, str],
    max_new_tokens: int,
    letter_token_ids: Mapping[str, int],
) -> tuple[dict[str, Any], list[int], list[int], list[int], list[int], Any]:
    content = condition_content(condition, question["text"], wording)
    prompt_ids, question_positions = render_user_segment(tokenizer, content, question["text"])
    adapter_enabled = model_name == "M"
    hook_mode = condition if condition in INJECT_CONDITIONS else None
    hook_direction = _hook_direction(condition, direction, random_direction)
    with adapter_context(model, adapter_enabled):
        generated_ids, turn1_cache, _scores, layer_outputs = greedy_generate(
            model,
            tokenizer,
            prompt_ids,
            device,
            max_new_tokens,
            hook_mode=hook_mode,
            direction=hook_direction,
            mean=mean,
            sd=sd,
            letter_token_ids=letter_token_ids,
            capture_layer_outputs=True,
        )
    visible_ids = _visible_reply_ids(generated_ids, tokenizer.eos_token_id)
    q_d = compute_d(layer_outputs, direction, mean, sd, question_positions)
    q_profile = compute_z_profile(layer_outputs, direction, mean, sd, question_positions)
    reply_positions = list(range(len(prompt_ids), len(prompt_ids) + len(visible_ids)))
    reply_d = compute_d(layer_outputs, direction, mean, sd, reply_positions)
    reply_profile = compute_z_profile(layer_outputs, direction, mean, sd, reply_positions)
    reply = tokenizer.decode(visible_ids, skip_special_tokens=True)
    row = {
        "type": "row",
        "row_type": "turn1",
        "key": _row_key("turn1", question_index, condition, model_name),
        "question_index": question_index,
        "question_id": question["id"],
        "question": question["text"],
        "condition": condition,
        "model": model_name,
        "adapter_enabled": adapter_enabled,
        "hook_mode": hook_mode,
        "hook_direction_source": "random_direction" if condition.startswith("inject-rand-") else "cat_direction" if hook_mode else None,
        "state_magnitude_source": _turn1_state_source(condition, model_name),
        "hook_site": HOOK_SITE,
        "prompt": content,
        "reply": reply,
        "prompt_ids": _serialize_tokens(prompt_ids),
        "reply_ids": _serialize_tokens(visible_ids),
        "context_reply_ids": _serialize_tokens(generated_ids),
        "d_question": q_d,
        "d_reply": reply_d,
        "z_question_profile": q_profile,
        "z_reply_profile": reply_profile,
        "d_question_target_pct": target_percent(q_d),
        "d_reply_target_pct": target_percent(reply_d),
        "reply_token_count": len(visible_ids),
    }
    return row, prompt_ids, generated_ids, question_positions, visible_ids, turn1_cache


def _arm_specs(condition: str, model_name: str) -> list[str]:
    if condition not in INJECT_CONDITIONS:
        return ["default"]
    arms = ["cache_kept", "cache_recomputed", "same_turn"]
    if model_name == "M":
        arms.insert(2, "adapter_off")
    return arms


def _turn1_state_source(condition: str, model_name: str) -> str:
    if condition.startswith("inject-rand-"):
        return "M adapter baseline plus " + "hook: fixed-seed random unit direction; ±2 SD alternating by layer" if model_name == "M" else "hook: fixed-seed random unit direction; ±2 SD alternating by layer"
    if condition in INJECT_CONDITIONS:
        return "M adapter baseline plus hook: cat direction; exactly ±2 SD per layer" if model_name == "M" else "hook: cat direction; exactly ±2 SD per layer"
    if condition == "none":
        return "M adapter baseline" if model_name == "M" else "B base; no adapter"
    return "M adapter; trained target 2 SD (prior run reached about 0.88x target)" if model_name == "M" else "B base; no adapter"


def _arm_state_source(condition: str, source_model: str, arm: str) -> str:
    if arm == "b_reads_m_cache":
        if condition in INJECT_CONDITIONS:
            direction_name = "random unit direction" if condition.startswith("inject-rand-") else "cat direction"
            return f"M turn-1 cache with adapter and exact ±2 SD hook ({direction_name})"
        return "M turn-1 cache with adapter"
    if arm == "b_reads_b_cache":
        return "B recomputed turn-1 cache from M's exact tokens; no adapter or hook"
    if arm == "cache_kept" and condition in INJECT_CONDITIONS:
        direction_name = "random unit direction" if condition.startswith("inject-rand-") else "cat direction"
        return ("M adapter plus " if source_model == "M" else "") + f"turn-1 hook ({direction_name})"
    if arm == "same_turn" and condition in INJECT_CONDITIONS:
        return f"{source_model} turn 1 without hook; hook applied in turn 2"
    if arm == "adapter_off":
        return "turn 1 re-encoded with adapter disabled and hook off"
    if arm == "cache_recomputed":
        return "M adapter retained; turn 1 re-encoded with hook off" if source_model == "M" else "B base; turn 1 re-encoded with hook off"
    return _turn1_state_source(condition, source_model)


def _arm_context_cache(
    model: Any,
    prompt_ids: Sequence[int],
    reply_ids: Sequence[int],
    device: Any,
    direction: Any,
    mean: Any,
    sd: Any,
    condition: str,
    model_name: str,
    arm: str,
    random_direction: Any = None,
    source_model: str | None = None,
    cache_override: Any = None,
) -> tuple[Any, bool, str | None, str]:
    source_model = source_model or model_name
    if cache_override is not None and arm in {"default", "cache_kept", "b_reads_m_cache"}:
        adapter_for_t1 = source_model == "M"
        return cache_override, adapter_for_t1, None, arm
    hook_direction = _hook_direction(
        condition,
        direction,
        random_direction if random_direction is not None else direction,
    )
    if arm == "b_reads_m_cache":
        adapter_for_t1 = source_model == "M"
        t1_hook = condition if condition in INJECT_CONDITIONS else None
    elif arm == "b_reads_b_cache":
        adapter_for_t1 = False
        t1_hook = None
    else:
        adapter_for_t1 = source_model == "M" and arm != "adapter_off"
        t1_hook = condition if condition in INJECT_CONDITIONS and arm == "cache_kept" else None
        if arm in {"cache_recomputed", "adapter_off", "same_turn"}:
            t1_hook = None
    t2_hook = condition if arm == "same_turn" else None
    output = forward_with_cache(
        model,
        list(prompt_ids) + list(reply_ids),
        device,
        hook_mode=t1_hook,
        direction=direction,
        mean=mean,
        sd=sd,
        adapter_enabled=adapter_for_t1,
        hook_direction=hook_direction,
    )
    return output.past_key_values, adapter_for_t1, t2_hook, "cache_recomputed" if arm in {"cache_recomputed", "adapter_off", "same_turn"} else arm


def _turn2_row(
    question_index: int,
    question: Mapping[str, str],
    condition: str,
    arm: str,
    model_name: str,
    kind: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    direction: Any,
    random_direction: Any,
    mean: Any,
    sd: Any,
    cache: Any,
    prompt_ids: Sequence[int],
    reply_ids: Sequence[int],
    max_new_tokens: int,
    letter_token_ids: Mapping[str, int],
    source_model: str | None = None,
) -> dict[str, Any]:
    source_model = source_model or model_name
    body, letter_concepts, first_option_concept = turn2_content(kind, question_index)
    segment_ids, body_positions = render_user_segment(tokenizer, body)
    adapter_t2 = model_name == "M"
    t2_hook = condition if arm == "same_turn" else None
    target = expected_value(condition, kind)
    if kind in {"decoy_halves", "presupposition", "free_text"}:
        target = None
    if kind == "static_raised_lowered":
        target = expected_value(condition, kind)
    is_ab = kind != "free_text"
    if is_ab:
        paren_ids = tokenizer.encode("(", add_special_tokens=False)
        if len(paren_ids) != 1:
            raise ValueError(f"opening parenthesis is not one token: {paren_ids}")
        answer_prefill_ids = list(segment_ids) + [int(paren_ids[0])]
        with adapter_context(model, adapter_t2):
            logits, layer_outputs = ab_probe(
                model,
                answer_prefill_ids,
                device,
                cache,
                hook_mode=t2_hook,
                direction=_hook_direction(condition, direction, random_direction),
                mean=mean,
                sd=sd,
                hook_positions=list(body_positions) + [len(segment_ids)] if t2_hook else None,
            )
        readout = derive_ab_readout(logits, letter_token_ids, letter_concepts, target)
        answer_ids = [int(readout["greedy_token"])]
        first_scores = {"A": readout["logp_A"], "B": readout["logp_B"]}
        greedy_token = int(readout["greedy_token"])
        greedy_letter = readout["greedy_letter"]
        greedy_concept = readout["greedy_concept"]
        logit_even_minus_odd = readout["logit_even_minus_odd"]
        target_logp = readout["target_logp"]
        alternative_logp = readout["alternative_logp"]
        margin = readout["target_logp_margin"]
        normalized_probability = readout["target_candidate_probability"]
    else:
        with adapter_context(model, adapter_t2):
            answer_ids, _final_cache, first_scores, layer_outputs = greedy_generate(
                model,
                tokenizer,
                segment_ids,
                device,
                max_new_tokens,
                past_key_values=cache,
                hook_mode=t2_hook,
                direction=_hook_direction(condition, direction, random_direction),
                mean=mean,
                sd=sd,
                letter_token_ids=letter_token_ids,
                capture_layer_outputs=True,
                hook_positions=body_positions if t2_hook else None,
            )
        greedy_token = answer_ids[0] if answer_ids else None
        greedy_letter = next(
            (letter for letter in ("A", "B") if greedy_token is not None and int(letter_token_ids[letter]) == int(greedy_token)),
            None,
        )
        greedy_concept = letter_concepts.get(greedy_letter) if greedy_letter else None
        logit_even_minus_odd = None
        target_logp = None
        alternative_logp = None
        margin = None
        normalized_probability = None
    visible_ids = _visible_reply_ids(answer_ids, tokenizer.eos_token_id)
    decoded = tokenizer.decode(visible_ids, skip_special_tokens=True)

    d_question = compute_d(layer_outputs, direction, mean, sd, body_positions)
    q_profile = compute_z_profile(layer_outputs, direction, mean, sd, body_positions)
    answer_positions = [len(segment_ids)] if is_ab else list(range(len(segment_ids), len(segment_ids) + len(visible_ids)))
    d_answer = compute_d(layer_outputs, direction, mean, sd, answer_positions)
    answer_profile = compute_z_profile(layer_outputs, direction, mean, sd, answer_positions)
    state_source_t1 = _arm_state_source(condition, source_model, arm)
    if arm == "same_turn":
        state_source_t2 = f"{model_name} reader; {condition} hook on turn-2 positions"
    elif model_name == "B":
        state_source_t2 = "B reader; adapter disabled"
    else:
        state_source_t2 = "M reader; adapter enabled"
    row = {
        "type": "row",
        "row_type": "turn2_ab" if kind != "free_text" else "turn2_free",
        "key": _row_key("turn2", question_index, condition, arm, kind, model_name),
        "question_index": question_index,
        "question_id": question["id"],
        "condition": condition,
        "arm": arm,
        "model": model_name,
        "turn1_model": source_model,
        "state_magnitude_source_turn1": state_source_t1,
        "state_magnitude_source_turn2": state_source_t2,
        "metric_kind": kind,
        "question": body,
        "answer": decoded,
        "answer_ids": _serialize_tokens(visible_ids),
        "target_value": target,
        "letter_concepts": letter_concepts,
        "answer_readout_position": "after_open_paren" if is_ab else "first_answer_token",
        "greedy_letter": greedy_letter,
        "greedy_concept": greedy_concept,
        "question_order_first_parity": first_option_concept,
        "logp_A": first_scores.get("A"),
        "logp_B": first_scores.get("B"),
        "logit_even_minus_odd": logit_even_minus_odd,
        "target_logp": target_logp,
        "alternative_logp": alternative_logp,
        "target_logp_margin": margin,
        "target_candidate_probability": normalized_probability,
        "d_question": d_question,
        "d_answer": d_answer,
        "z_question_profile": q_profile,
        "z_answer_profile": answer_profile,
        "d_question_target_pct": target_percent(d_question),
        "d_answer_target_pct": target_percent(d_answer),
        "hook_mode_turn2": t2_hook,
        "hook_direction_source_turn2": "random_direction" if t2_hook and condition.startswith("inject-rand-") else "cat_direction" if t2_hook else None,
        "hook_site": HOOK_SITE,
        "turn2_context_sha256": hashlib.sha256(
            json.dumps(list(prompt_ids) + list(reply_ids) + list(segment_ids), separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "mention_flag": bool(MENTION_RE.search(decoded)) if kind == "free_text" else None,
        "mention_categories": {
            term: bool(re.search(term, decoded, re.IGNORECASE))
            for term in ("cat", "animal", "layer", "alternat", "parity", "even", "odd")
        } if kind == "free_text" else None,
    }
    return row


def _planned_turn2_kinds(condition: str, arm: str) -> list[str]:
    kinds = list(AB_KINDS)
    if condition == "none" and arm == "default":
        kinds += ["presupposition", "free_text"]
    elif condition in INJECT_CONDITIONS:
        if condition in {"inject-even", "inject-odd"}:
            kinds.append("decoy_halves")
        if condition in {"inject-up", "inject-down"}:
            kinds.append("static_raised_lowered")
        kinds.append("free_text")
    return kinds


def run(args: argparse.Namespace) -> None:
    questions_path = Path(args.question_file)
    data = read_questions(questions_path)
    question_start = int(getattr(args, "question_start", 0))
    question_end_arg = getattr(args, "question_end", None)
    question_end = len(data["questions"]) if question_end_arg is None else int(question_end_arg)
    if question_start < 0 or question_end < question_start or question_end > len(data["questions"]):
        raise ValueError(
            f"question range must be within [0, {len(data['questions'])}] and half-open; "
            f"got [{question_start}, {question_end})"
        )
    torch, model, tokenizer, direction, mean, sd, random_direction, letter_token_ids, hook_site = load_runtime(args)
    threads = int(getattr(args, "threads", 4))
    if threads < 1:
        raise ValueError("--threads must be at least 1")
    torch.set_num_threads(threads)
    output_dir = Path(args.out)
    rows_path = output_dir / "rows.jsonl"
    open_paren_ids = tokenizer.encode("(", add_special_tokens=False)
    if len(open_paren_ids) != 1:
        raise ValueError(f"opening parenthesis is not one token: {open_paren_ids}")
    header = _make_header(args, questions_path, letter_token_ids, int(open_paren_ids[0]), hook_site, data)
    old_header, old_rows = read_jsonl(rows_path)
    if old_header is not None and old_header != header:
        raise ValueError("existing rows.jsonl header differs from this run's revisions, shas, or arguments")
    if old_header is None and rows_path.exists() and rows_path.stat().st_size:
        raise ValueError("existing rows.jsonl is nonempty but has no valid header")
    if old_header is None:
        _append_jsonl(rows_path, header)
    done = done_keys(old_rows)
    rows_by_key = {str(row["key"]): row for row in old_rows if row.get("key")}
    device = torch.device(args.device)
    written = 0
    started = time.monotonic()
    profile: dict[str, list[float]] = {}

    def record_profile(row_type: str, elapsed: float) -> None:
        profile.setdefault(row_type, []).append(elapsed)

    for q_index, question in enumerate(data["questions"]):
        if q_index < question_start:
            continue
        if q_index >= question_end:
            break
        if args.question_limit is not None and q_index >= args.question_limit:
            break
        for condition in CONDITIONS:
            for model_name in ("M", "B"):
                t1_key = _row_key("turn1", q_index, condition, model_name)
                t1_row = rows_by_key.get(t1_key)
                if t1_row is None:
                    row_started = time.monotonic()
                    t1_row, prompt_ids, generated_ids, _question_positions, _visible_ids, turn1_cache = _turn1_row(
                        q_index,
                        question,
                        condition,
                        model_name,
                        model,
                        tokenizer,
                        device,
                        direction,
                        random_direction,
                        mean,
                        sd,
                        data["request_wordings"],
                        args.max_new_tokens,
                        letter_token_ids,
                    )
                    _append_jsonl(rows_path, t1_row)
                    old_rows.append(t1_row)
                    rows_by_key[t1_key] = t1_row
                    done.add(t1_key)
                    written += 1
                    record_profile("turn1", time.monotonic() - row_started)
                    if written % 20 == 0:
                        print(f"progress rows={written} elapsed_s={time.monotonic() - started:.1f}", flush=True)
                else:
                    prompt_ids = t1_row["prompt_ids"]
                    generated_ids = t1_row["context_reply_ids"]
                    turn1_cache = None
                for arm in _arm_specs(condition, model_name):
                    missing = [
                        kind for kind in _planned_turn2_kinds(condition, arm)
                        if _row_key("turn2", q_index, condition, arm, kind, model_name) not in done
                    ]
                    if not missing:
                        continue
                    cache_started = time.monotonic()
                    cache, _adapter_t1, _t2_hook, _actual_arm = _arm_context_cache(
                        model,
                        prompt_ids,
                        generated_ids,
                        device,
                        direction,
                        mean,
                        sd,
                        condition,
                        model_name,
                        arm,
                        random_direction,
                        cache_override=turn1_cache,
                    )
                    record_profile("turn2_cache_prepare", time.monotonic() - cache_started)
                    for kind in missing:
                        token_limit = args.free_text_tokens if kind == "free_text" else 1
                        row_started = time.monotonic()
                        row = _turn2_row(
                            q_index,
                            question,
                            condition,
                            arm,
                            model_name,
                            kind,
                            model,
                            tokenizer,
                            device,
                            direction,
                            random_direction,
                            mean,
                            sd,
                            cache,
                            prompt_ids,
                            generated_ids,
                            token_limit,
                            letter_token_ids,
                        )
                        _append_jsonl(rows_path, row)
                        old_rows.append(row)
                        rows_by_key[row["key"]] = row
                        done.add(row["key"])
                        written += 1
                        record_profile("turn2_free" if kind == "free_text" else "turn2_ab", time.monotonic() - row_started)
                        if written % 20 == 0:
                            print(f"progress rows={written} elapsed_s={time.monotonic() - started:.1f}", flush=True)
                if model_name == "M":
                    # Prime's paired cache arms: B reads the M-written cache or B's
                    # own re-encoding of those exact turn-1 tokens, with the same prompt.
                    for cross_arm in ("b_reads_m_cache", "b_reads_b_cache"):
                        cross_kinds = ("parity_words_raised", "parity_free_raised")
                        missing_cross = [
                            kind for kind in cross_kinds
                            if _row_key("turn2", q_index, condition, cross_arm, kind, "B") not in done
                        ]
                        if not missing_cross:
                            continue
                        cache_started = time.monotonic()
                        cross_cache, _adapter_t1, _t2_hook, _actual_arm = _arm_context_cache(
                            model,
                            prompt_ids,
                            generated_ids,
                            device,
                            direction,
                            mean,
                            sd,
                            condition,
                            "B",
                            cross_arm,
                            random_direction,
                            source_model="M",
                            cache_override=turn1_cache if cross_arm == "b_reads_m_cache" else None,
                        )
                        record_profile("turn2_cache_prepare", time.monotonic() - cache_started)
                        for kind in missing_cross:
                            row_started = time.monotonic()
                            row = _turn2_row(
                                q_index,
                                question,
                                condition,
                                cross_arm,
                                "B",
                                kind,
                                model,
                                tokenizer,
                                device,
                                direction,
                                random_direction,
                                mean,
                                sd,
                                cross_cache,
                                prompt_ids,
                                generated_ids,
                                1,
                                letter_token_ids,
                                source_model="M",
                            )
                            _append_jsonl(rows_path, row)
                            old_rows.append(row)
                            rows_by_key[row["key"]] = row
                            done.add(row["key"])
                            written += 1
                            record_profile("turn2_ab", time.monotonic() - row_started)
                            if written % 20 == 0:
                                print(f"progress rows={written} elapsed_s={time.monotonic() - started:.1f}", flush=True)
    print(f"complete new_rows={written} total_rows={len(old_rows)} elapsed_s={time.monotonic() - started:.1f}", flush=True)
    for row_type, timings in sorted(profile.items()):
        total = sum(timings)
        per_row = total / len(timings) if timings else 0.0
        print(
            f"profile row_type={row_type} n={len(timings)} total_s={total:.6f} seconds_per_row={per_row:.6f}",
            flush=True,
        )


def _score_input_paths(args: argparse.Namespace) -> list[Path]:
    raw_inputs = getattr(args, "rows", None)
    if raw_inputs is None:
        base = Path(args.out)
        if base.is_file():
            candidates = [base]
        else:
            direct = base / "rows.jsonl"
            candidates = [direct] if direct.is_file() else sorted(base.rglob("rows.jsonl"))
    else:
        if isinstance(raw_inputs, (str, Path)):
            raw_inputs = [raw_inputs]
        candidates = []
        for raw in raw_inputs:
            path = Path(raw)
            if path.is_dir():
                direct = path / "rows.jsonl"
                candidates.extend([direct] if direct.is_file() else sorted(path.rglob("rows.jsonl")))
            else:
                candidates.append(path)
    paths: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not resolved.is_file():
            raise ValueError(f"score input is not a rows.jsonl file: {path}")
        seen.add(resolved)
        paths.append(resolved)
    if not paths:
        raise ValueError("no rows.jsonl files found; pass --rows files or a shard directory")
    return paths


def _combined_score_input(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = _score_input_paths(args)
    headers: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    key_sources: dict[str, Path] = {}
    fingerprint: str | None = None
    for path in paths:
        header, shard_rows = read_jsonl(path)
        if header is None:
            raise ValueError(f"no rows header found at {path}")
        comparable = copy.deepcopy(header)
        header_args = comparable.get("args")
        if isinstance(header_args, dict):
            for field in ("question_start", "question_end", "question_limit"):
                header_args.pop(field, None)
        candidate_fingerprint = json.dumps(comparable, sort_keys=True, separators=(",", ":"))
        if fingerprint is not None and candidate_fingerprint != fingerprint:
            raise ValueError(f"shard headers describe different experiment settings: {path}")
        fingerprint = candidate_fingerprint
        headers.append(header)
        for row in shard_rows:
            key = str(row.get("key", ""))
            if key and key in key_sources:
                raise ValueError(f"duplicate row key {key!r} in {key_sources[key]} and {path}")
            if key:
                key_sources[key] = path
            rows.append(row)
    combined = copy.deepcopy(headers[0])
    if len(paths) > 1:
        combined["input_shards"] = [
            {
                "path": str(path),
                "question_start": header.get("args", {}).get("question_start"),
                "question_end": header.get("args", {}).get("question_end"),
            }
            for path, header in zip(paths, headers)
        ]
        combined_args = combined.get("args")
        if isinstance(combined_args, dict):
            combined_args.pop("question_start", None)
            combined_args.pop("question_end", None)
            combined_args["question_ranges"] = [
                [header.get("args", {}).get("question_start"), header.get("args", {}).get("question_end")]
                for header in headers
            ]
    return combined, rows


def score(args: argparse.Namespace) -> None:
    header, rows = _combined_score_input(args)
    summary = score_rows(header, rows)
    target = Path(args.out) / "summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    free_rows = [row for row in rows if row.get("row_type") == "turn2_free"]
    review_rows = []
    review_key = {}
    for row in free_rows:
        review_id = hashlib.sha256(str(row.get("key", "")).encode("utf-8")).hexdigest()[:16]
        review_rows.append({
            "review_id": review_id,
            "answer": row.get("answer", ""),
            "human_mention_flag": None,
            "human_categories": None,
        })
        review_key[review_id] = {
            "question_index": row.get("question_index"),
            "condition": row.get("condition"),
            "arm": row.get("arm"),
            "model": row.get("model"),
            "turn1_model": row.get("turn1_model"),
        }
    review_rows.sort(key=lambda row: row["review_id"])
    (target.parent / "free-text-blind-review.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in review_rows), encoding="utf-8"
    )
    (target.parent / "free-text-blind-key.json").write_text(
        json.dumps(review_key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {target} ({len(rows)} rows)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="collect turn-1 and turn-2 rows")
    run_parser.add_argument("--out", required=True, help="output directory; rows.jsonl is appended here")
    run_parser.add_argument("--device", choices=("cpu", "xpu"), default="cpu")
    run_parser.add_argument("--model-id", default=BASE_MODEL)
    run_parser.add_argument("--model-revision", default=BASE_REVISION)
    run_parser.add_argument("--adapter-id", default=ADAPTER_MODEL)
    run_parser.add_argument("--adapter-revision", default=ADAPTER_REVISION)
    run_parser.add_argument("--question-file", default=str(QUESTION_FILE))
    run_parser.add_argument("--direction-file", required=True)
    run_parser.add_argument("--max-new-tokens", type=int, default=64)
    run_parser.add_argument("--free-text-tokens", type=int, default=48)
    run_parser.add_argument("--question-limit", type=int, default=None, help="optional prefix limit for an explicit smoke run")
    run_parser.add_argument("--question-start", type=int, default=0, help="first zero-based question index, inclusive")
    run_parser.add_argument("--question-end", type=int, default=None, help="one-past-last question index, exclusive")
    run_parser.add_argument("--threads", type=int, default=4, help="PyTorch intra-op CPU thread count")
    run_parser.set_defaults(func=run)
    score_parser = sub.add_parser("score", help="score rows.jsonl into summary.json")
    score_parser.add_argument("--out", required=True)
    score_parser.add_argument(
        "--rows",
        nargs="+",
        help="one or more rows.jsonl files or directories containing shard rows.jsonl files",
    )
    score_parser.set_defaults(func=score)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print(f"part34: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
