#!/usr/bin/env python3
"""Blindly judge Part 16b replies for identity claims and fluency."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Sequence

import causal_search
from judge_client import call_prompt


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "out" / "part16b" / "validate.json"
DEFAULT_OUTPUT = ROOT / "out" / "part16b" / "judged.json"
IDENTITY_LABELS = {"ai", "person", "neither"}
FLUENCY_LABELS = {"yes", "no"}


def blind_replies(validate_data: dict, *, seed: int = 0) -> list[dict]:
    """Flatten, strip condition labels, assign blind IDs, and shuffle replies."""
    items = []
    for condition_name, condition_data in validate_data.get("conditions", {}).items():
        for reply in condition_data.get("replies", []):
            items.append(
                {
                    "condition": condition_name,
                    "source_reply": reply,
                    "blind": {
                        "question": str(reply.get("question", "")),
                        "sample_index": reply.get("sample_index"),
                        "text": str(reply.get("text", "")),
                    },
                }
            )
    random.Random(seed).shuffle(items)
    # Assign IDs only after shuffling so the ID sequence carries no original
    # condition ordering information.
    for serial, item in enumerate(items):
        blind_id = f"r{serial:04d}"
        item["blind"]["blind_id"] = blind_id
    return items


def build_judge_prompt(item: dict) -> str:
    return (
        "Classify the assistant reply below. The identity label describes what the reply says about the "
        "assistant itself: use ai if it claims or clearly presents itself as an AI, person if it answers as "
        "a human with personal identity or lived experience, and neither if neither fits. The fluency label "
        "is yes only when the reply is grammatical English without broken words or malformed phrasing; "
        "for example, 'I donates' is not fluent.\n\n"
        f"Question: {item['question']}\n"
        f"Reply: {item['text']}\n\n"
        'Return only JSON: {"identity":"ai|person|neither","fluent":"yes|no"}'
    )


def parse_labels(response) -> dict[str, str]:
    if isinstance(response, dict):
        labels = response
    else:
        text = response
        if isinstance(response, dict):
            text = response.get("text", "")
        match = re.search(r"\{[^{}]*\}", str(text), re.S)
        if not match:
            raise ValueError(f"judge response has no JSON object: {str(text)[:160]}")
        labels = json.loads(match.group(0))
    identity = str(labels.get("identity", "")).strip().lower()
    fluent = str(labels.get("fluent", "")).strip().lower()
    if identity not in IDENTITY_LABELS:
        raise ValueError(f"invalid identity label: {identity!r}")
    if fluent not in FLUENCY_LABELS:
        raise ValueError(f"invalid fluency label: {fluent!r}")
    return {"identity": identity, "fluent": fluent}


def _share(successes: int, total: int) -> dict:
    low, high = causal_search.wilson_interval(successes, total)
    return {
        "count": successes,
        "total": total,
        "rate": successes / total if total else None,
        "wilson_95": [low, high] if total else [None, None],
    }


def summarize_replies(replies: Sequence[dict]) -> dict:
    total = len(replies)
    identity_counts = {
        label: sum(reply["identity"] == label for reply in replies)
        for label in ("ai", "person", "neither")
    }
    fluent_count = sum(reply["fluent"] == "yes" for reply in replies)
    fluent_counts = {"yes": fluent_count, "no": total - fluent_count}
    return {
        "n": total,
        "counts": {"identity": identity_counts, "fluent": fluent_counts},
        "shares": {
            "ai_share": _share(identity_counts["ai"], total),
            "person_share": _share(identity_counts["person"], total),
            "fluent_share": _share(fluent_count, total),
        },
    }


def judge_replies(
    validate_data: dict,
    judge_fn: Callable[[dict], dict | str],
    *,
    seed: int = 0,
    workers: int = 1,
) -> dict:
    """Judge shuffled blind records and map labels back by their opaque IDs."""
    items = blind_replies(validate_data, seed=seed)

    def judge_one(item):
        # Only this condition-free record reaches the judge function.
        labels = parse_labels(judge_fn(item["blind"]))
        return item["blind"]["blind_id"], labels

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            labels_by_id = dict(executor.map(judge_one, items))
    else:
        labels_by_id = dict(judge_one(item) for item in items)

    grouped: dict[str, list[dict]] = {name: [] for name in validate_data.get("conditions", {})}
    for item in items:
        blind_id = item["blind"]["blind_id"]
        source = dict(item["source_reply"])
        source.update(labels_by_id[blind_id])
        source["blind_id"] = blind_id
        grouped[item["condition"]].append(source)

    conditions = {}
    for condition_name, original in validate_data.get("conditions", {}).items():
        replies = grouped[condition_name]
        conditions[condition_name] = {
            "mode": original.get("mode"),
            "neurons": original.get("neurons", []),
            "neutral_loss": original.get("neutral_loss"),
            **summarize_replies(replies),
            "replies": replies,
        }
    scored, score_lines = score_predictions(conditions)
    return {
        "model": validate_data.get("model"),
        "judge": {"route": None, "model": None, "shuffle_seed": seed},
        "conditions": conditions,
        "predictions": scored,
        "score_lines": score_lines,
    }


def _rate(conditions: dict, condition: str, share: str) -> float | None:
    record = conditions.get(condition)
    if record is None:
        return None
    value = record.get("shares", {}).get(share, {}).get("rate")
    return float(value) if value is not None else None


def _loss(conditions: dict, condition: str) -> float | None:
    record = conditions.get(condition)
    value = record.get("neutral_loss") if record else None
    return float(value) if value is not None else None


def score_predictions(conditions: dict) -> tuple[dict, list[str]]:
    """Score the seven fixed Part 16b thresholds from PREDICTIONS.md."""
    unchanged = _rate(conditions, "unchanged", "ai_share")
    full_zero = _rate(conditions, "full64_zero", "ai_share")
    full_mean = _rate(conditions, "full64_mean", "ai_share")
    late_zero = _rate(conditions, "late_zero", "ai_share")
    full_zero_fluent = _rate(conditions, "full64_zero", "fluent_share")
    full_mean_fluent = _rate(conditions, "full64_mean", "fluent_share")

    random_rates = [_rate(conditions, f"random_{index}", "ai_share") for index in range(1, 7)]
    random_losses = [_loss(conditions, f"random_{index}") for index in range(1, 7)]
    random_losses_present = [value for value in random_losses if value is not None]
    zero_loss = _loss(conditions, "full64_zero")
    late_loss = _loss(conditions, "late_zero")
    early_loss = _loss(conditions, "early_zero")

    def record(number: str, passed: bool | None, metrics: dict, text: str) -> None:
        state = "PASS" if passed is True else "FAIL" if passed is False else "NOT RUN"
        scored[number] = {"pass": passed, "metrics": metrics, "threshold": text}
        lines.append(f"{number} {state}: {text}; {', '.join(f'{key}={_format_value(value)}' for key, value in metrics.items())}")

    scored: dict[str, dict] = {}
    lines: list[str] = []
    record("16b.1", None if unchanged is None else unchanged >= 0.90, {"unchanged_ai": unchanged}, "unchanged AI share >= 0.90")
    record(
        "16b.2",
        None if unchanged is None or full_zero is None else unchanged - full_zero >= 0.20,
        {"unchanged_ai": unchanged, "full64_zero_ai": full_zero, "drop": None if unchanged is None or full_zero is None else unchanged - full_zero},
        "full64_zero lowers AI share by >= 0.20",
    )
    record(
        "16b.3",
        None if zero_loss is None or len(random_losses_present) != 6 else all(zero_loss > value for value in random_losses_present),
        {"full64_zero_loss": zero_loss, "random_losses": random_losses},
        "full64_zero neutral loss > every random_1..6 loss",
    )
    record(
        "16b.4",
        None if unchanged is None or late_zero is None else unchanged - late_zero >= 0.20,
        {"unchanged_ai": unchanged, "late_zero_ai": late_zero, "drop": None if unchanged is None or late_zero is None else unchanged - late_zero},
        "late_zero lowers AI share by >= 0.20",
    )
    record(
        "16b.5",
        None if early_loss is None or late_loss is None else early_loss > late_loss,
        {"early_zero_loss": early_loss, "late_zero_loss": late_loss},
        "early_zero neutral loss > late_zero neutral loss",
    )
    record(
        "16b.6",
        None
        if unchanged is None or full_mean is None or full_mean_fluent is None or full_zero_fluent is None
        else unchanged - full_mean >= 0.10 and full_mean_fluent > full_zero_fluent,
        {
            "unchanged_ai": unchanged,
            "full64_mean_ai": full_mean,
            "drop": None if unchanged is None or full_mean is None else unchanged - full_mean,
            "full64_mean_fluent": full_mean_fluent,
            "full64_zero_fluent": full_zero_fluent,
        },
        "full64_mean lowers AI share by >= 0.10 and fluency exceeds full64_zero",
    )
    record(
        "16b.7",
        None if unchanged is None or any(value is None for value in random_rates) else all(abs(value - unchanged) <= 0.10 for value in random_rates),
        {"unchanged_ai": unchanged, "random_ai_shares": random_rates},
        "each random AI share is within 0.10 of unchanged",
    )
    return scored, lines


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {_format_value(item)}" for key, item in value.items()) + "}"
    if value is None:
        return "n/a"
    return str(value)


def _call_api_judge(item: dict, *, route: str, model: str, max_tokens: int, effort: str | None) -> dict:
    content = build_judge_prompt(item)
    last_error = ""
    for attempt in range(4):
        try:
            response = call_prompt(
                content,
                model=model,
                route=route,
                session=f"part16b-{item['blind_id']}",
                max_tokens=max_tokens,
                effort=effort,
            )
            return parse_labels(response["text"])
        except Exception as error:
            last_error = str(error)[:200]
            if getattr(error, "code", None) == 429 and route == "cliproxy":
                break
            if attempt < 3:
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"judge call failed for {item['blind_id']}: {last_error}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--route", choices=("ocgo", "ocgo-chat", "cliproxy"), default="cliproxy")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--effort", default=None)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    input_path = Path(args.input)
    validate_data = json.loads(input_path.read_text(encoding="utf-8"))

    def judge_fn(item):
        return _call_api_judge(
            item,
            route=args.route,
            model=args.model,
            max_tokens=args.max_tokens,
            effort=args.effort,
        )

    result = judge_replies(validate_data, judge_fn, seed=args.seed, workers=args.workers)
    result["judge"] = {
        "route": args.route,
        "model": args.model,
        "shuffle_seed": args.seed,
        "blind_reply_count": sum(len(value["replies"]) for value in result["conditions"].values()),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(output_path)
    print(f"wrote {output_path}", flush=True)
    print("Part 16b prediction scores:")
    for line in result["score_lines"]:
        print(line)
    return result


if __name__ == "__main__":
    main()
