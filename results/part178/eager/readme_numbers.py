#!/usr/bin/env python3
"""Derive README-only Part 178 numbers from the registered eager runs."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part178 import score178, words178  # noqa: E402


SET_ORDER = ("R8", "N8", "G", "N4")
THINKING_GENERATED_SETS = ("R8", "G")


def _word_row(rows, word):
    return next((row for row in rows if row.get("word") == word), None)


def _window_wait(score, set_name):
    table = score["tables"]["window"][set_name]
    rate = _word_row(table["rates"], "wait")
    traces = _word_row(table["trace_rates"], "wait")
    if rate is None or traces is None:
        return {"rate": None, "trace_spread": None}
    return {
        "rate": rate,
        "trace_spread": traces,
        "source": {
            "rate": f"tables.window.{set_name}.rates[word=wait]",
            "trace_spread": f"tables.window.{set_name}.trace_rates[word=wait]",
        },
    }


def _uncut_wait(score, set_name):
    table = score["tables"]["uncut"][set_name]
    rate = _word_row(table["rates"], "wait")
    ranking = table["ranking"]
    rank = next((index + 1 for index, row in enumerate(ranking)
                 if row.get("word") == "wait"), None)
    if rate is None:
        raise ValueError(f"missing uncut wait rate for {set_name}")
    return {
        "rate_per_1000": rate["rate_per_1000"],
        "rank": rank,
        "source": {
            "rate": f"tables.uncut.{set_name}.rates[word=wait].rate_per_1000",
            "rank": f"tables.uncut.{set_name}.ranking (1-based position of wait)",
        },
    }


def _full_text_counts(rows):
    spans_by_row = []
    for row in rows:
        # This sensitivity deliberately uses each complete generated_text field,
        # matching COLD.md's full-text opener_spans calculation.
        spans = words178.opener_spans(str(row.get("generated_text", "")))
        spans_by_row.append((row, spans))

    def aggregate(items):
        opener_count = sum(len(spans) for _row, spans in items)
        wait_count = sum(sum(span["word"] == "wait" for span in spans)
                         for _row, spans in items)
        trace_count = len(items)
        traces_with_wait = sum(any(span["word"] == "wait" for span in spans)
                               for _row, spans in items)
        return {
            "trace_count": trace_count,
            "opener_count": opener_count,
            "wait_opener_count": wait_count,
            "wait_rate_per_1000_openers": (
                1000.0 * wait_count / opener_count if opener_count else None
            ),
            "traces_with_wait": traces_with_wait,
        }

    grouped = {}
    reasons = sorted({str(row.get("finish_reason") or "unknown")
                      for row, _spans in spans_by_row})
    for reason in reasons:
        grouped[reason] = aggregate([
            pair for pair in spans_by_row
            if str(pair[0].get("finish_reason") or "unknown") == reason
        ])
    grouped["all"] = aggregate(spans_by_row)
    return grouped


def _sensitivity_ratios(full_text_sensitivity):
    pairs = {"0.8B": ("R8", "N8"), "4B": ("G", "N4")}
    result = {}
    for label, (thinking_set, control_set) in pairs.items():
        result[label] = {"thinking_set": thinking_set, "control_set": control_set, "by_finish_reason": {}}
        for reason in ("cap", "end-of-turn", "all"):
            thinking_rate = full_text_sensitivity[thinking_set][reason]["wait_rate_per_1000_openers"]
            control_rate = full_text_sensitivity[control_set][reason]["wait_rate_per_1000_openers"]
            result[label]["by_finish_reason"][reason] = {
                "wait_rate_ratio": (
                    thinking_rate / control_rate
                    if thinking_rate is not None and control_rate not in (None, 0)
                    else None
                ),
                "thinking_wait_rate_per_1000": thinking_rate,
                "control_wait_rate_per_1000": control_rate,
            }
    return result


def build_numbers():
    score_path = HERE / "score.json"
    score = score178.read_json(score_path)
    if not isinstance(score, dict) or score.get("part") != 178:
        raise ValueError(f"missing or invalid eager score: {score_path}")

    generated = {}
    full_text_sensitivity = {}
    for set_name in SET_ORDER:
        rows = score178.read_jsonl(HERE / f"{set_name}.jsonl")
        if not rows:
            raise ValueError(f"no eager rows for {set_name}")
        finish_counts = Counter(str(row.get("finish_reason") or "unknown") for row in rows)
        first = rows[0]
        metadata = {
            "row_count": len(rows),
            "unique_problem_ids": len({str(row.get("problem_id", "")) for row in rows}),
            "model": first.get("model"),
            "revision": first.get("revision"),
            "attn_implementation": first.get("attn_implementation"),
            "thinking": first.get("thinking"),
            "max_new_tokens": first.get("max_new_tokens"),
            "temperature": first.get("temperature"),
            "top_k": first.get("top_k"),
            "top_p": first.get("top_p"),
            "presence_penalty": first.get("presence_penalty"),
            "finish_reason_counts": dict(sorted(finish_counts.items())),
        }
        if set_name in THINKING_GENERATED_SETS:
            close_rows = sum(
                row.get("think_close_token_id") is not None
                and int(row["think_close_token_id"]) in {int(token) for token in row.get("generated_ids", [])}
                for row in rows
            )
            metadata["think_close"] = {
                "rows_containing_token": close_rows,
                "row_count": len(rows),
                "token_id": first.get("think_close_token_id"),
                "source": "JSONL row think_close_token_id found in generated_ids",
            }
        metadata["source"] = f"eager/{set_name}.jsonl"
        generated[set_name] = metadata
        full_text_sensitivity[set_name] = _full_text_counts(rows)

    window_wait = {name: _window_wait(score, name) for name in SET_ORDER}
    uncut_wait = {name: _uncut_wait(score, name) for name in ("U", "G")}
    follow_on = score["predictions"]["178.6"]["estimate"]["by_onset_side"]
    w1 = score["predictions"]["178.W1"]["estimate"]
    rate_top = list(w1["rate_top20"])
    trace_top = list(w1["trace_top20"])
    rate_only = [word for word in rate_top if word not in trace_top]
    trace_only = [word for word in trace_top if word not in rate_top]
    prediction_ids = list(score.get("predictions", {}))
    numbered_prediction_count = sum(
        prediction_id.startswith("178.") and prediction_id[4:].isdigit()
        for prediction_id in prediction_ids
    )
    gate_count = sum(
        key.startswith("gate") and key[4:].isdigit()
        for key in score.get("read_gates", {})
    )
    w3_ids = ("178.W3a", "178.W3b")

    return {
        "description": "README supplementary values derived from eager score and generated rows.",
        "summary_counts": {
            "generated_set_count": len(generated),
            "numbered_prediction_count": numbered_prediction_count,
            "prediction_count": len(prediction_ids),
            "gate_count": gate_count,
            "w3_held_model_count": sum(
                score["predictions"][prediction_id]["verdict"] == "held"
                for prediction_id in w3_ids
            ),
            "source": "counts of eager score prediction, gate, and generated-set keys",
        },
        "w1_top20_symmetric_difference": {
            "count": len(rate_only) + len(trace_only),
            "rate_only_words": rate_only,
            "trace_only_words": trace_only,
            "source": "predictions.178.W1.estimate.rate_top20 and trace_top20",
        },
        "registered_context": {
            "part": score["part"],
            "next_part": int(score["part"]) + 1,
            "source": "next_part is the registered follow-on named by PART178-DESIGN.md",
        },
        "generated_sets": generated,
        "window_wait": window_wait,
        "uncut_wait": uncut_wait,
        "capped_sensitivity_full_text": {
            "method": (
                "words178.opener_spans over each complete JSONL generated_text; no think-close, "
                "2,000-token window, or onset cut is applied. Rates are wait openers per 1,000 "
                "openers, grouped by finish_reason."
            ),
            "sets": full_text_sensitivity,
            "thinking_to_control_wait_rate_ratios": _sensitivity_ratios(full_text_sensitivity),
        },
        "follow_on_by_onset_side": follow_on,
        "definitions": {
            "prefix_token_limit": words178.PREFIX_TOKENS,
            "window_token_limit": words178.WINDOW_TOKENS,
            "onset_censor_lines": words178.ONSET_CENSOR_LINES,
            "imported_list_count": len(words178.IMPORTED_LIST),
            "source": "part178.words178 constants",
        },
        "batch_size_recording": {
            "recorded_in_rows": False,
            "source": "eager generated JSONL row fields; no batch_size field is present",
        },
    }


def main():
    out_path = HERE / "readme_numbers.json"
    score178.write_json(out_path, build_numbers())
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
