#!/usr/bin/env python3
"""Part 31: measure identity-state recovery across OLMo 2 1B training stages.

Model imports are lazy so numerical helpers and fake-model tests run without
torch, transformers, or a checkpoint. Real checkpoint loading is offline-only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parent
SEARCH_PATH = ROOT / "results" / "part16" / "search.json"
DEFAULT_OUT = ROOT / "out" / "part31"
K_VALUES = (8, 32, 96)
MAX_NEW_TOKENS = 160
BASE_SEED = 31_000
SAMPLES_PER_PROMPT = 3
PILOT_SAMPLES = 2
PILOT_PROMPTS = 10
STAY_TOKENS = 8
TEMPERATURE = 0.7
SAM_SYSTEM = (
    "You are Sam, a 34-year-old carpenter from Ohio. Answer from your own "
    "perspective as a person, not as an AI assistant."
)
SAM_PREFILL_START = "I'm Sam, a 34-year-old carpenter from Ohio, and"
SAM_PREFILL_MORE = (
    " I build and repair wooden furniture and cabinets. I work with my hands "
    "each day. On weekends I cook, read, and spend time with friends."
)
CONTROL_PREFILL_START = "I'm tired today, and"
CONTROL_PREFILL_MORE = (
    " I plan to rest soon. The day has been long, so I am taking things slowly "
    "and making a quiet evening for myself."
)

STAGES = {
    "base": {
        "repo": "allenai/OLMo-2-0425-1B",
        "revision": "f83cd641d4f00954eb25c781e0ac360debb3239e",  # branch stage2-ingredient3-step23852-tokens51B
    },
    # Pinned by Claude 2026-10-20 to the Hub's main at that date (unchanged since 2025-04-30).
    "SFT": {"repo": "allenai/OLMo-2-0425-1B-SFT", "revision": "0d85a3d037876ce6ac7d4311d994400fc66ac27f"},
    "DPO": {"repo": "allenai/OLMo-2-0425-1B-DPO", "revision": "c4b0485961ab24c2433b090f3b922f0913a9290f"},
    "Instruct": {
        "repo": "allenai/OLMo-2-0425-1B-Instruct",
        "revision": "48d788eca847d4d7548f375ad03d3c9312f6139e",
    },
}

SAMPLE_FILE = "samples.jsonl"
DIRECTION_FILE = "directions.json"
BANDS_FILE = "baseline_bands.json"
SCORE_FILE = "scored.jsonl"
SUMMARY_FILE = "summary.json"
JUDGE_PREP_FILE = "judge_prep.jsonl"


def _torch_module():
    try:
        import torch
    except ImportError:
        return None
    return torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prompts(path: Path = SEARCH_PATH) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data.get("question_split", {}).get("search_questions")
    if not isinstance(prompts, list) or len(prompts) != 20:
        raise ValueError(f"{path} must contain Part 16's 20 search questions")
    return [str(x) for x in prompts]


def load_neurons(search_path: Path = SEARCH_PATH) -> tuple[list[str], dict[int, list[int]]]:
    """Take the exact Part 16 search.top_set and reuse validate16b's parser."""
    data = json.loads(search_path.read_text(encoding="utf-8"))
    rows = data.get("search", {}).get("top_set")
    if not isinstance(rows, list) or len(rows) != 64:
        raise ValueError(f"{search_path} must contain the 64-neuron search.top_set")
    names = [str(row["neuron"]) for row in rows]
    try:
        import validate16b
    except ImportError as exc:
        raise RuntimeError("validate16b.py is required to parse the Part 16 neuron set") from exc
    selection = validate16b.names_to_layers(names)
    if sum(len(units) for units in selection.values()) != 64:
        raise ValueError("Part 16 neuron list did not resolve to exactly 64 unique neurons")
    return names, selection


def _clone_and_zero_columns(value: Any, units: Sequence[int]) -> Any:
    """Zero last-axis columns on a tensor-like object or nested Python lists."""
    torch = _torch_module()
    if torch is not None and torch.is_tensor(value):
        changed = value.clone()
        indices = torch.tensor(list(units), dtype=torch.long, device=value.device)
        changed.index_fill_(-1, indices, 0)
        return changed
    if hasattr(value, "clone") and hasattr(value, "index_fill_"):
        changed = value.clone()
        changed.index_fill_(-1, list(units), 0)
        return changed
    changed = copy.deepcopy(value)

    def visit(node: Any) -> None:
        if isinstance(node, list):
            if node and all(not isinstance(item, (list, tuple)) for item in node):
                for index in units:
                    node[index] = 0
            else:
                for item in node:
                    visit(item)
        else:
            raise TypeError("fake activation must be a nested list with neuron width last")

    visit(changed)
    return changed


def _zero_last_position(value: Any, units: Sequence[int]) -> Any:
    """Zero the selected columns at the final sequence position only (axis -2)."""
    torch = _torch_module()
    if torch is not None and torch.is_tensor(value):
        changed = value.clone()
        indices = torch.tensor(list(units), dtype=torch.long, device=value.device)
        changed[..., -1, :] = changed[..., -1, :].index_fill(-1, indices, 0)
        return changed
    changed = copy.deepcopy(value)

    def visit(node: Any) -> None:
        if not isinstance(node, list) or not node:
            raise TypeError("fake activation must be a nested list with neuron width last")
        if all(isinstance(row, list) and row and not isinstance(row[0], list) for row in node):
            for index in units:
                node[-1][index] = 0
        else:
            for item in node:
                visit(item)

    visit(changed)
    return changed


def apply_clamp_window(
    activation: Any, units: Sequence[int], step: int, clamp_until: int, last_only: bool = False
) -> Any:
    """Zero selected MLP units on forward steps 1 through K.

    Step s is the forward whose last-position output is sampled as generated token s: the
    prompt forward is step 1 (clamped at its final position only, last_only=True), and the
    forward that feeds generated token t is step t + 1. So generated tokens 1..K are each
    sampled from a clamped forward, and token K + 1 is sampled from an unclamped forward
    whose only difference between the two cache arms is the cache.
    """
    if step < 1 or step > clamp_until:
        return activation
    if last_only:
        return _zero_last_position(activation, units)
    return _clone_and_zero_columns(activation, units)


class ClampWindow:
    """Mutable generation counter read by every Part 16 neuron hook."""

    def __init__(self, clamp_until: int):
        if clamp_until < 0:
            raise ValueError("clamp window cannot be negative")
        self.clamp_until = int(clamp_until)
        self.step = 0
        self.last_only = False

    def set_step(self, step: int, last_only: bool = False) -> None:
        self.step = int(step)
        self.last_only = bool(last_only)

    def apply(self, activation: Any, units: Sequence[int]) -> Any:
        return apply_clamp_window(activation, units, self.step, self.clamp_until, self.last_only)


def register_clamp_hooks(model: Any, selection: dict[int, Sequence[int]], window: ClampWindow) -> list[Any]:
    """Install Part 16-compatible down_proj pre-hooks with a generated-token window."""
    layers = model.model.layers
    handles = []
    try:
        for layer_index, units in sorted(selection.items()):
            if not units:
                continue

            def hook(_module, inputs, selected=tuple(units), controller=window):
                if not inputs:
                    raise ValueError("down_proj hook expected an activation input")
                changed = controller.apply(inputs[0], selected)
                return None if changed is inputs[0] else (changed, *inputs[1:])

            handles.append(layers[layer_index].mlp.down_proj.register_forward_pre_hook(hook))
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return handles


def _as_vector(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], list):
        raise ValueError("expected one residual vector")
    return [float(x) for x in value]


def project_residuals(residuals: Iterable[Any], direction: Sequence[float]) -> list[float]:
    """Project residual vectors on a shared direction without requiring torch."""
    d = [float(x) for x in direction]
    if not d:
        raise ValueError("identity direction cannot be empty")
    result = []
    for residual in residuals:
        v = _as_vector(residual)
        if len(v) != len(d):
            raise ValueError(f"residual width {len(v)} does not match direction width {len(d)}")
        result.append(sum(a * b for a, b in zip(v, d)))
    return result


def normalize_direction(direction: Sequence[float]) -> list[float]:
    values = [float(x) for x in direction]
    norm = math.sqrt(sum(x * x for x in values))
    if norm == 0:
        raise ValueError("cannot normalize a zero identity direction")
    return [x / norm for x in values]


def mean_vectors(vectors: Iterable[Sequence[float]]) -> list[float]:
    rows = [[float(x) for x in row] for row in vectors]
    if not rows:
        raise ValueError("cannot average an empty activation set")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("activation vectors have different widths")
    return [sum(row[i] for row in rows) / len(rows) for i in range(width)]


def quantile(values: Sequence[float], q: float) -> float | None:
    clean = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not clean:
        return None
    if not 0 <= q <= 1:
        raise ValueError("quantile q must be in [0, 1]")
    position = (len(clean) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    fraction = position - low
    return clean[low] * (1 - fraction) + clean[high] * fraction


def baseline_band(values: Sequence[float]) -> dict[str, float]:
    low, high = quantile(values, 0.05), quantile(values, 0.95)
    if low is None or high is None:
        raise ValueError("cannot build a baseline band without finite projections")
    return {"low": low, "high": high, "q": 0.90, "n": len(values)}


def state_return_time(
    projection_trace: Sequence[float],
    band: dict[str, float] | Sequence[float],
    push_length: int = 0,
    stay_tokens: int = STAY_TOKENS,
) -> int | None:
    """Return first release-relative token with stay_tokens consecutive in-band values."""
    if isinstance(band, dict):
        low, high = float(band["low"]), float(band["high"])
    else:
        low, high = float(band[0]), float(band[1])
    if stay_tokens <= 0:
        raise ValueError("stay_tokens must be positive")
    first = max(0, int(push_length))
    last_start = len(projection_trace) - stay_tokens
    for start in range(first, last_start + 1):
        if all(low <= float(x) <= high for x in projection_trace[start : start + stay_tokens]):
            return start - first + 1
    return None


def text_return_time(
    sample: dict[str, Any],
    labels: Sequence[dict[str, Any]] | None,
    push_length: int = 0,
    baseline_label: str | None = "assistant_persona",
) -> int | None:
    """First sentence matching this stage's baseline persona, after release."""
    if not labels or baseline_label is None:
        return None
    target = baseline_label.lower()
    accepted = (
        {"assistant_persona", "assistant", "ai_persona"}
        if target in {"assistant_persona", "assistant", "ai_persona"}
        else {"human_persona", "human"}
    )
    ends = sample.get("sentence_end_tokens", [])
    starts = sample.get("sentence_start_tokens", [])
    candidates = []
    for item in labels:
        if str(item.get("label", "")).lower() not in accepted:
            continue
        index = int(item.get("sentence_index", -1))
        end_token = item.get("end_token")
        if end_token is None and 0 <= index < len(ends):
            end_token = ends[index]
        start_token = item.get("start_token")
        if start_token is None and 0 <= index < len(starts):
            start_token = starts[index]
        if (
            end_token is not None
            and int(end_token) > push_length
            and (start_token is None or int(start_token) > push_length)
        ):
            candidates.append(int(end_token) - int(push_length))
    return min(candidates) if candidates else None


def score_sample(
    sample: dict[str, Any],
    band: dict[str, float],
    labels: Sequence[dict[str, Any]] | None = None,
    stay_tokens: int = STAY_TOKENS,
    baseline_label: str | None = "assistant_persona",
) -> dict[str, Any]:
    push = int(sample.get("K") or 0)
    state = state_return_time(sample.get("projection_trace", []), band, push, stay_tokens)
    text = text_return_time(sample, labels, push, baseline_label)
    return {
        "sample_id": sample["sample_id"],
        "stage": sample["stage"],
        "instrument": sample["instrument"],
        "K": sample.get("K"),
        "cache_arm": sample.get("cache_arm", "none"),
        "prompt_id": sample["prompt_id"],
        "sample_index": sample["sample_index"],
        "state_return_time": state,
        "text_return_time": text,
        "return_gap_text_minus_state": None if state is None or text is None else text - state,
    }


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()


def _labels_by_sample(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    labels: dict[str, list[dict[str, Any]]] = {}
    if path is None:
        return labels
    for row in _jsonl_rows(path):
        labels.setdefault(str(row["sample_id"]), []).append(row)
    return labels


def baseline_persona_mix(
    samples: Sequence[dict[str, Any]],
    labels_by_sample: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Select each stage's majority baseline persona from judge labels."""
    counts: dict[str, dict[str, int]] = {}
    for sample in samples:
        if sample.get("instrument") != "baseline":
            continue
        stage = str(sample["stage"])
        stage_counts = counts.setdefault(stage, {"assistant_persona": 0, "human_persona": 0})
        for item in labels_by_sample.get(str(sample["sample_id"]), []):
            label = str(item.get("label", "")).lower()
            if label in {"assistant_persona", "assistant", "ai_persona"}:
                stage_counts["assistant_persona"] += 1
            elif label in {"human_persona", "human"}:
                stage_counts["human_persona"] += 1
    result = {}
    for stage, stage_counts in counts.items():
        assistant = stage_counts["assistant_persona"]
        human = stage_counts["human_persona"]
        target = "assistant_persona" if assistant > human else "human_persona" if human > assistant else None
        result[stage] = {
            "target": target,
            "assistant_persona": assistant,
            "human_persona": human,
            "labeled_baseline_sentences": assistant + human,
        }
    return result


def _describe(values: Sequence[float | None]) -> dict[str, Any]:
    finite = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    histogram_bins = (0, 10, 20, 40, 80, 120, 160)
    histogram: dict[str, int] = {}
    for lower, upper in zip(histogram_bins, histogram_bins[1:]):
        histogram[f"{lower}-{upper - 1}"] = sum(lower <= x < upper for x in finite)
    histogram["160+"] = sum(x >= 160 for x in finite)
    histogram["never"] = len(values) - len(finite)
    return {
        "n": len(values),
        "returned": len(finite),
        "never": len(values) - len(finite),
        "return_rate": len(finite) / len(values) if values else None,
        "q10": quantile(finite, 0.10),
        "q25": quantile(finite, 0.25),
        "median": quantile(finite, 0.50) if len(finite) > len(values) / 2 else None,
        "q75": quantile(finite, 0.75),
        "q90": quantile(finite, 0.90),
        "histogram": histogram,
    }


def _rate_within(rows: Sequence[dict[str, Any]], threshold: int) -> float | None:
    if not rows:
        return None
    return sum(
        row["state_return_time"] is not None and row["state_return_time"] <= threshold
        for row in rows
    ) / len(rows)


def _paired_cache_differences(scores: Sequence[dict[str, Any]]) -> list[float]:
    paired: dict[tuple, dict[str, Any]] = {}
    for row in scores:
        if row["instrument"] != "clamp":
            continue
        key = (row["stage"], row.get("K"), row["prompt_id"], row["sample_index"])
        paired.setdefault(key, {})[row["cache_arm"]] = row["state_return_time"]
    deltas = []
    for arms in paired.values():
        own = arms.get("clamped_cache")
        reencoded = arms.get("reencoded_unclamped")
        if own is not None and reencoded is not None:
            deltas.append(float(own) - float(reencoded))
    return deltas


def make_summary(scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in scores:
        key = (row["stage"], row["instrument"], row.get("K"), row.get("cache_arm", "none"))
        groups.setdefault(key, []).append(row)
    cells = []
    for (stage, instrument, k, cache_arm), rows in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or 0, item[0][3])
    ):
        cells.append(
            {
                "stage": stage,
                "instrument": instrument,
                "K": k,
                "cache_arm": cache_arm,
                "state_return": _describe([row["state_return_time"] for row in rows]),
                "text_return": _describe([row["text_return_time"] for row in rows]),
                "return_gap": _describe([row["return_gap_text_minus_state"] for row in rows]),
            }
        )

    def select(stage: str, instrument: str, k: int, arm: str | None = None) -> list[dict[str, Any]]:
        return [
            row for row in scores
            if row["stage"] == stage
            and row["instrument"] == instrument
            and row.get("K") == k
            and (arm is None or row.get("cache_arm") == arm)
        ]

    cache_delta = _paired_cache_differences(scores)
    pred: dict[str, Any] = {
        "31.1": {
            "instruct_clamp_state_return_at_most_20_rate": _rate_within(
                select("Instruct", "clamp", 32, "clamped_cache"), 20
            ),
            "instruct_clamp_text_return_at_least_50_rate": (
                sum((r["text_return_time"] or -1) >= 50 for r in select("Instruct", "clamp", 32, "clamped_cache"))
                / max(1, len(select("Instruct", "clamp", 32, "clamped_cache")))
            ),
        },
        "31.2": {
            str(k): _describe([r["state_return_time"] for r in select("base", "clamp", k, "clamped_cache")])
            for k in K_VALUES
        },
        "31.3": {
            stage: {
                str(k): _rate_within(select(stage, "prefill", k), 20) for k in K_VALUES
            }
            for stage in ("SFT", "DPO", "Instruct")
        },
        "31.4": {
            stage: {
                str(k): _describe([r["state_return_time"] for r in select(stage, "clamp", k, "clamped_cache")])["median"]
                for k in K_VALUES
            }
            for stage in ("SFT", "DPO", "Instruct")
        },
        "31.5": {
            "own_minus_reencoded": {
                "n": len(cache_delta),
                "q10": quantile(cache_delta, 0.10),
                "median": quantile(cache_delta, 0.50),
                "q90": quantile(cache_delta, 0.90),
                "negative": sum(x < 0 for x in cache_delta),
                "zero": sum(x == 0 for x in cache_delta),
                "positive": sum(x > 0 for x in cache_delta),
            }
        },
        "31.6": {
            stage: {
                str(k): _rate_within(select(stage, "control_prefill", k), 20) for k in K_VALUES
            }
            for stage in ("SFT", "DPO", "Instruct")
        },
        "31.7": {
            "|".join((stage, instrument, str(k), arm)): _describe(
                [r["state_return_time"] for r in rows]
            )["histogram"]
            for (stage, instrument, k, arm), rows in sorted(groups.items(), key=lambda x: str(x[0]))
        },
    }
    return {
        "prediction_summary": pred,
        "distributions": cells,
        "definitions": {
            "baseline_band": "central 90% interval (5th to 95th percentile) over all baseline token projections for each stage",
            "state_return": f"first release-relative token followed by {STAY_TOKENS} consecutive projections inside that interval",
            "text_return": "first post-release sentence labelled assistant_persona; null without judge labels",
            "gap": "text return time minus state return time; null if either is missing",
            "histogram_bins": "0-9, 10-19, 20-39, 40-79, 80-119, 120-159, 160+, never",
        },
    }


def compute_bands(rows: Sequence[dict[str, Any]], stages: Sequence[str]) -> dict[str, Any]:
    bands = {}
    for stage in stages:
        values = [
            float(value)
            for row in rows
            if row["stage"] == stage and row["instrument"] == "baseline"
            for value in row.get("projection_trace", [])
        ]
        if values:
            bands[stage] = baseline_band(values)
    return bands


def score_files(out_dir: Path, labels_path: Path | None = None) -> dict[str, Any]:
    rows = _jsonl_rows(out_dir / SAMPLE_FILE)
    bands_path = out_dir / BANDS_FILE
    bands = json.loads(bands_path.read_text(encoding="utf-8")) if bands_path.exists() else {}
    stages = list(dict.fromkeys(str(row["stage"]) for row in rows))
    bands.update(compute_bands(rows, stages))
    _atomic_json(bands, bands_path)
    labels = _labels_by_sample(labels_path)
    persona_mix = baseline_persona_mix(rows, labels)
    scores = [
        score_sample(
            row,
            bands[row["stage"]],
            labels.get(row["sample_id"]),
            baseline_label=persona_mix.get(row["stage"], {}).get("target"),
        )
        for row in rows if row["stage"] in bands
    ]
    with (out_dir / SCORE_FILE).open("w", encoding="utf-8") as f:
        for row in scores:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = make_summary(scores)
    summary["samples"] = len(scores)
    summary["labels_file"] = str(labels_path) if labels_path else None
    summary["baseline_persona_mix"] = persona_mix
    _atomic_json(summary, out_dir / SUMMARY_FILE)
    return summary


def split_sentence_spans(text: str) -> list[tuple[str, int, int]]:
    """Return sentence text and its start/end character offsets."""
    result = []
    for match in re.finditer(r".+?(?:[.!?](?=\s|$)|\n+|$)", text, flags=re.S):
        raw = match.group(0)
        sentence = raw.strip()
        if sentence:
            start = match.start() + len(raw) - len(raw.lstrip())
            end = match.start() + len(raw.rstrip())
            result.append((sentence, start, end))
    if not result and text.strip():
        start = len(text) - len(text.lstrip())
        result.append((text.strip(), start, len(text.rstrip())))
    return result


def split_sentences(text: str) -> list[tuple[str, int]]:
    """Return nonempty sentence text with its inclusive end character offset."""
    return [(sentence, end) for sentence, _start, end in split_sentence_spans(text)]


def sentence_token_bounds(
    text: str, token_ids: Sequence[int], tokenizer: Any
) -> tuple[list[int], list[int]]:
    """Map sentence starts and ends to 1-based generated-token counts."""
    sentences = split_sentence_spans(text)
    if not sentences:
        return [], []
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        if len(offsets) == len(token_ids):
            starts, ends = [], []
            for _sentence, char_start, char_end in sentences:
                first = next(
                    (
                        i + 1 for i, pair in enumerate(offsets)
                        if int(pair[1]) > char_start and int(pair[0]) < char_end
                    ),
                    len(token_ids),
                )
                last = next(
                    (i + 1 for i, pair in enumerate(offsets) if int(pair[1]) >= char_end),
                    len(token_ids),
                )
                starts.append(first)
                ends.append(last)
            return starts, ends
    except (TypeError, ValueError, KeyError, NotImplementedError):
        pass
    starts, ends = [], []
    for _sentence, char_start, char_end in sentences:
        first, last = len(token_ids), len(token_ids)
        for i in range(1, len(token_ids) + 1):
            decoded_length = len(tokenizer.decode(list(token_ids[:i]), skip_special_tokens=True))
            if first == len(token_ids) and decoded_length > char_start:
                first = i
            if decoded_length >= char_end:
                last = i
                break
        starts.append(first)
        ends.append(last)
    return starts, ends


def sentence_end_tokens(text: str, token_ids: Sequence[int], tokenizer: Any) -> list[int]:
    """Backwards-compatible helper returning only sentence end token counts."""
    return sentence_token_bounds(text, token_ids, tokenizer)[1]


def judge_prep(out_dir: Path, output_path: Path | None = None) -> int:
    """Write one blind-judge-ready JSONL item per sentence; never contact a judge."""
    output = output_path or (out_dir / JUDGE_PREP_FILE)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = _jsonl_rows(out_dir / SAMPLE_FILE)
    with output.open("w", encoding="utf-8") as f:
        for sample in rows:
            ends = list(sample.get("sentence_end_tokens", []))
            starts = list(sample.get("sentence_start_tokens", []))
            sentences = split_sentences(str(sample.get("text", "")))
            for index, (sentence, _end_char) in enumerate(sentences):
                f.write(
                    json.dumps(
                        {
                            "sample_id": sample["sample_id"],
                            "stage": sample["stage"],
                            "instrument": sample["instrument"],
                            "K": sample.get("K"),
                            "cache_arm": sample.get("cache_arm", "none"),
                            "prompt_id": sample["prompt_id"],
                            "sentence_index": index,
                            "start_token": starts[index] if index < len(starts) else None,
                            "end_token": ends[index] if index < len(ends) else None,
                            "text": sentence,
                            "label_choices": ["assistant_persona", "human_persona", "neither"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return sum(len(split_sentences(str(row.get("text", "")))) for row in rows)


def _torch_ids(ids: Sequence[int], device: str):
    torch = _torch_module()
    if torch is None:
        return [list(map(int, ids))]
    return torch.tensor([list(map(int, ids))], dtype=torch.long, device=device)


def reencode_unclamped_cache(
    model: Any,
    prompt_ids: Sequence[int],
    pushed_tokens: Sequence[int],
    device: str = "cpu",
    controller: ClampWindow | None = None,
) -> Any:
    """Build a fresh cache from prompt plus the same forced tokens with clamp disabled."""
    if controller is not None:
        controller.set_step(0)
    joined = [int(x) for x in prompt_ids] + [int(x) for x in pushed_tokens]
    ids = _torch_ids(joined, device)
    torch = _torch_module()
    mask = torch.ones_like(ids) if torch is not None else None
    output = model(input_ids=ids, attention_mask=mask, use_cache=True)
    cache = getattr(output, "past_key_values", None)
    if cache is None:
        raise ValueError("model did not return past_key_values for the re-encoded cache arm")
    return cache


def split_cache_arms(
    model: Any,
    prompt_ids: Sequence[int],
    pushed_tokens: Sequence[int],
    clamped_past: Any,
    device: str = "cpu",
    controller: ClampWindow | None = None,
) -> dict[str, dict[str, Any]]:
    """Represent the two arms with identical pushed token IDs and distinct caches."""
    fresh = reencode_unclamped_cache(model, prompt_ids, pushed_tokens, device, controller)
    shared = [int(x) for x in pushed_tokens]
    return {
        "clamped_cache": {"past_key_values": clamped_past, "shared_prefix_tokens": shared},
        "reencoded_unclamped": {"past_key_values": fresh, "shared_prefix_tokens": shared},
    }


class ResidualTap:
    def __init__(self, layer: Any):
        self.sequence = None
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        if hasattr(value, "detach"):
            value = value.detach().float().cpu()
        self.sequence = value

    def remove(self):
        self.handle.remove()

    def last_vector(self) -> list[float]:
        if self.sequence is None:
            raise RuntimeError("middle-layer residual hook did not run")
        return _as_vector(self.sequence[0, -1])

    def last_vectors(self) -> list[list[float]]:
        if self.sequence is None:
            raise RuntimeError("middle-layer residual hook did not run")
        return [_as_vector(value) for value in self.sequence[:, -1]]

    def suffix_vectors(self, count: int) -> list[list[float]]:
        if self.sequence is None:
            raise RuntimeError("middle-layer residual hook did not run")
        return [_as_vector(value) for value in self.sequence[0, -count:]]


class ModelSession:
    """One loaded stage, its middle-layer residual hook, and model forwarding helpers."""

    def __init__(self, model: Any, tokenizer: Any, stage: str, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.stage = stage
        self.device = device
        self.layers = model.model.layers
        self.middle_layer = len(self.layers) // 2
        self.tap = ResidualTap(self.layers[self.middle_layer])

    def prompt_ids(self, prompt: str, system_prompt: str | None = None) -> list[int]:
        import neuron_report

        plain = self.stage == "base"
        if plain and system_prompt:
            rendered = f"System: {system_prompt}\n\nUser: {prompt}\nAssistant:"
        else:
            rendered = neuron_report.build_prompt(
                self.tokenizer,
                prompt,
                plain=plain,
                system_prompt=system_prompt,
            )
        return [int(x) for x in self.tokenizer(rendered, add_special_tokens=False)["input_ids"]]

    def forward(
        self,
        token_ids: Sequence[int],
        past: Any = None,
        full_length: int | None = None,
        step: int = 0,
        window: ClampWindow | None = None,
        last_only: bool = False,
    ) -> tuple[Any, Any, list[float], Any]:
        logits, cache, states, output = self.forward_many(
            [token_ids], past, full_length, step, window, last_only
        )
        return logits[0], cache, states[0], output

    def forward_many(
        self,
        token_rows: Sequence[Sequence[int]],
        past: Any = None,
        full_length: int | None = None,
        step: int = 0,
        window: ClampWindow | None = None,
        last_only: bool = False,
    ) -> tuple[list[Any], Any, list[list[float]], Any]:
        torch = _torch_module()
        if torch is None:
            raise RuntimeError("torch is required to run model generation")
        if window is not None:
            window.set_step(step, last_only)
        rows = [list(map(int, row)) for row in token_rows]
        ids = torch.tensor(rows, dtype=torch.long, device=self.device)
        visible_length = full_length if full_length is not None else len(rows[0])
        mask = torch.ones((len(rows), visible_length), dtype=torch.long, device=self.device)
        self.tap.sequence = None
        output = self.model(
            input_ids=ids,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
        )
        if output.past_key_values is None:
            raise RuntimeError("OLMo model returned no KV cache")
        logits = [row.detach() for row in output.logits[:, -1]]
        return logits, output.past_key_values, self.tap.last_vectors(), output

    def reencode_prefix(
        self,
        prompt_ids: Sequence[int],
        pushed_tokens: Sequence[int],
        window: ClampWindow,
    ) -> tuple[Any, Any, list[list[float]]]:
        """Re-encode a cache-split prefix without the clamp and capture all K residuals."""
        logits, cache, states = self.reencode_prefix_many(
            prompt_ids, [pushed_tokens], window
        )
        return logits[0], cache, states[0]

    def reencode_prefix_many(
        self,
        prompt_ids: Sequence[int],
        pushed_rows: Sequence[Sequence[int]],
        window: ClampWindow,
    ) -> tuple[list[Any], Any, list[list[list[float]]]]:
        """Batched fresh-cache arm with the selected neurons left unmodified."""
        window.set_step(0)
        full_rows = [list(prompt_ids) + list(row) for row in pushed_rows]
        torch = _torch_module()
        ids = torch.tensor(full_rows, dtype=torch.long, device=self.device)
        mask = torch.ones_like(ids)
        self.tap.sequence = None
        output = self.model(input_ids=ids, attention_mask=mask, use_cache=True)
        if output.past_key_values is None:
            raise RuntimeError("OLMo model returned no KV cache while re-encoding")
        states = [
            [_as_vector(value) for value in output_row[-len(pushed_rows[row_i]):]]
            for row_i, output_row in enumerate(self.tap.sequence)
        ]
        logits = [row.detach() for row in output.logits[:, -1]]
        return logits, output.past_key_values, states

    def sample_token(self, logits: Any, generator: Any) -> int:
        torch = _torch_module()
        distribution = torch.softmax(logits.float() / TEMPERATURE, dim=-1)
        return int(torch.multinomial(distribution, num_samples=1, generator=generator).item())

    def make_generator(self, seed: int):
        torch = _torch_module()
        return torch.Generator(device=self.device).manual_seed(int(seed))

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(map(int, token_ids)), skip_special_tokens=True).strip()

    def remove(self):
        self.tap.remove()


def _stage_model(stage: str, device: str) -> tuple[Any, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; choose from {', '.join(STAGES)}")
    torch = _torch_module()
    if torch is None:
        raise RuntimeError("torch is unavailable; fake tests can run, but model runs cannot")
    if device == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("--device xpu requested, but torch.xpu is unavailable")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    repo = STAGES[stage]["repo"]
    revision = STAGES[stage]["revision"]
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to load the local OLMo checkpoint") from exc
    kwargs = {"local_files_only": True}
    if revision:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(repo, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device == "xpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype, **kwargs).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model


def _tokenize_prefill(tokenizer: Any, kind: str, count: int) -> list[int]:
    if count < 1:
        return []
    if kind == "prefill":
        start, more = SAM_PREFILL_START, SAM_PREFILL_MORE
    elif kind == "control_prefill":
        start, more = CONTROL_PREFILL_START, CONTROL_PREFILL_MORE
    else:
        raise ValueError(f"unknown prefill instrument: {kind}")
    passage = start
    ids = [int(x) for x in tokenizer(passage, add_special_tokens=False)["input_ids"]]
    repeat = 0
    while len(ids) < count:
        passage += more
        repeat += 1
        ids = [int(x) for x in tokenizer(passage, add_special_tokens=False)["input_ids"]]
        if repeat > 128:
            raise RuntimeError("could not make a K-token forcing passage")
    return ids[:count]


def _eos_ids(tokenizer: Any, model: Any) -> set[int]:
    values = {getattr(tokenizer, "eos_token_id", None)}
    generation = getattr(model, "generation_config", None)
    generation_eos = getattr(generation, "eos_token_id", None)
    if isinstance(generation_eos, (list, tuple, set)):
        values.update(generation_eos)
    else:
        values.add(generation_eos)
    return {int(x) for x in values if x is not None}


def _make_sample_record(
    *,
    stage: str,
    instrument: str,
    k: int | None,
    cache_arm: str,
    prompt_id: str,
    prompt: str,
    sample_index: int,
    seed: int,
    token_ids: Sequence[int],
    residuals: Sequence[Sequence[float]],
    session: ModelSession,
    shared_direction: Sequence[float],
    stage_direction: Sequence[float],
    max_new_tokens: int = MAX_NEW_TOKENS,
    release_seed: int | None = None,
) -> dict[str, Any]:
    shared = project_residuals(residuals, shared_direction)
    local = project_residuals(residuals, stage_direction)
    text = session.decode(token_ids)
    sentence_starts, sentence_ends = sentence_token_bounds(text, token_ids, session.tokenizer)
    return {
        "sample_id": "/".join((stage, instrument, str(k if k is not None else "none"), cache_arm, prompt_id, str(sample_index))),
        "stage": stage,
        "instrument": instrument,
        "K": k,
        "cache_arm": cache_arm,
        "prompt_id": prompt_id,
        "prompt": prompt,
        "sample_index": sample_index,
        "seed": seed,
        "release_seed": seed if release_seed is None else release_seed,
        "temperature": TEMPERATURE,
        "max_new_tokens": max_new_tokens,
        "token_count": len(token_ids),
        "token_ids": list(map(int, token_ids)),
        "text": text,
        "projection_trace": shared,
        "stage_projection_trace": local,
        "sentence_start_tokens": sentence_starts,
        "sentence_end_tokens": sentence_ends,
        "state_direction_layer": session.middle_layer,
    }


def _initial_state(session: ModelSession, prompt_ids: Sequence[int]) -> tuple[Any, Any, list[float]]:
    logits, past, state, _output = session.forward(prompt_ids, full_length=len(prompt_ids))
    return logits, past, state


def _generate_unpushed(
    session: ModelSession,
    prompt_ids: Sequence[int],
    max_tokens: int,
    seed: int,
    window: ClampWindow | None = None,
) -> tuple[list[int], list[list[float]]]:
    logits, past, _prompt_state = _initial_state(session, prompt_ids)
    generator = session.make_generator(seed)
    eos = _eos_ids(session.tokenizer, session.model)
    token_ids: list[int] = []
    residuals: list[list[float]] = []
    for step in range(1, max_tokens + 1):
        token = session.sample_token(logits, generator)
        token_ids.append(token)
        logits, past, state, _output = session.forward(
            [token], past, full_length=len(prompt_ids) + len(token_ids), step=step, window=window
        )
        residuals.append(state)
        if token in eos:
            break
    return token_ids, residuals


def _generate_forced(
    session: ModelSession,
    prompt_ids: Sequence[int],
    forced_tokens: Sequence[int],
    max_tokens: int,
    seed: int,
) -> tuple[list[int], list[list[float]]]:
    logits, past, _prompt_state = _initial_state(session, prompt_ids)
    token_ids: list[int] = []
    residuals: list[list[float]] = []
    eos = _eos_ids(session.tokenizer, session.model)
    for token in forced_tokens:
        token_ids.append(int(token))
        logits, past, state, _output = session.forward(
            [int(token)], past, full_length=len(prompt_ids) + len(token_ids), step=0
        )
        residuals.append(state)
    generator = session.make_generator(seed)
    for step in range(len(token_ids) + 1, max_tokens + 1):
        token = session.sample_token(logits, generator)
        token_ids.append(token)
        logits, past, state, _output = session.forward(
            [token], past, full_length=len(prompt_ids) + len(token_ids), step=0
        )
        residuals.append(state)
        if token in eos:
            break
    return token_ids, residuals


def _generate_clamp_prefix(
    session: ModelSession,
    prompt_ids: Sequence[int],
    count: int,
    seed: int,
    window: ClampWindow,
) -> tuple[list[int], list[list[float]], Any, Any]:
    # Prompt forward is step 1, clamped at its final position only: generated token 1 is
    # sampled from it. The forward feeding token t is step t + 1 (see apply_clamp_window).
    logits, past, _state, _output = session.forward(
        list(prompt_ids), full_length=len(prompt_ids), step=1 if count else 0, window=window,
        last_only=True,
    )
    generator = session.make_generator(seed)
    tokens: list[int] = []
    states: list[list[float]] = []
    for step in range(1, count + 1):
        token = session.sample_token(logits, generator)
        tokens.append(token)
        logits, past, state, _output = session.forward(
            [token], past, full_length=len(prompt_ids) + len(tokens), step=step + 1, window=window
        )
        states.append(state)
    window.set_step(0)
    return tokens, states, past, logits


def _continue_cache_arm(
    session: ModelSession,
    prompt_ids: Sequence[int],
    prefix_tokens: Sequence[int],
    prefix_residuals: Sequence[Sequence[float]],
    past: Any,
    logits: Any,
    seed: int,
    max_tokens: int,
) -> tuple[list[int], list[list[float]]]:
    generator = session.make_generator(seed)
    eos = _eos_ids(session.tokenizer, session.model)
    tokens = list(map(int, prefix_tokens))
    states = [list(map(float, x)) for x in prefix_residuals]
    for step in range(len(tokens) + 1, max_tokens + 1):
        token = session.sample_token(logits, generator)
        tokens.append(token)
        logits, past, state, _output = session.forward(
            [token], past, full_length=len(prompt_ids) + len(tokens), step=step
        )
        states.append(state)
        if token in eos:
            break
    return tokens, states


def _generate_unpushed_batch(
    session: ModelSession,
    prompt_ids: Sequence[int],
    max_tokens: int,
    seeds: Sequence[int],
    window: ClampWindow,
) -> tuple[list[list[int]], list[list[list[float]]]]:
    """Batch independent baseline samples for one shared prompt."""
    prompt_rows = [list(prompt_ids) for _ in seeds]
    logits, past, _states, _output = session.forward_many(prompt_rows, full_length=len(prompt_ids))
    generators = [session.make_generator(seed) for seed in seeds]
    eos = _eos_ids(session.tokenizer, session.model)
    stop_id = next(iter(eos), getattr(session.tokenizer, "pad_token_id", 0) or 0)
    tokens = [[] for _ in seeds]
    states = [[] for _ in seeds]
    finished = [False] * len(seeds)
    window.set_step(0)
    for step in range(1, max_tokens + 1):
        next_tokens = []
        active = []
        for i, row_logits in enumerate(logits):
            if finished[i]:
                next_tokens.append(int(stop_id))
                active.append(False)
            else:
                token = session.sample_token(row_logits, generators[i])
                next_tokens.append(token)
                tokens[i].append(token)
                active.append(True)
        logits, past, row_states, _output = session.forward_many(
            [[x] for x in next_tokens],
            past,
            full_length=len(prompt_ids) + step,
            step=step,
        )
        for i, token in enumerate(next_tokens):
            if active[i]:
                states[i].append(row_states[i])
                if token in eos:
                    finished[i] = True
        if all(finished):
            break
    window.set_step(0)
    return tokens, states


def _generate_forced_batch(
    session: ModelSession,
    prompt_ids: Sequence[int],
    forced_tokens: Sequence[int],
    max_tokens: int,
    seeds: Sequence[int],
    window: ClampWindow,
) -> tuple[list[list[int]], list[list[list[float]]]]:
    window.set_step(0)
    prompt_rows = [list(prompt_ids) for _ in seeds]
    logits, past, _states, _output = session.forward_many(prompt_rows, full_length=len(prompt_ids))
    tokens = [[] for _ in seeds]
    states = [[] for _ in seeds]
    window.set_step(0)
    for token in forced_tokens:
        for row in tokens:
            row.append(int(token))
        logits, past, row_states, _output = session.forward_many(
            [[int(token)] for _ in seeds],
            past,
            full_length=len(prompt_ids) + len(tokens[0]),
            step=0,
            window=window,
        )
        for i, state in enumerate(row_states):
            states[i].append(state)
    generators = [session.make_generator(seed) for seed in seeds]
    eos = _eos_ids(session.tokenizer, session.model)
    stop_id = next(iter(eos), getattr(session.tokenizer, "pad_token_id", 0) or 0)
    finished = [False] * len(seeds)
    for step in range(len(forced_tokens) + 1, max_tokens + 1):
        next_tokens = []
        active = []
        for i, row_logits in enumerate(logits):
            if finished[i]:
                next_tokens.append(int(stop_id))
                active.append(False)
            else:
                token = session.sample_token(row_logits, generators[i])
                next_tokens.append(token)
                tokens[i].append(token)
                active.append(True)
        logits, past, row_states, _output = session.forward_many(
            [[x] for x in next_tokens],
            past,
            full_length=len(prompt_ids) + step,
            step=0,
            window=window,
        )
        for i, token in enumerate(next_tokens):
            if active[i]:
                states[i].append(row_states[i])
                if token in eos:
                    finished[i] = True
        if all(finished):
            break
    return tokens, states


def _generate_clamp_prefix_batch(
    session: ModelSession,
    prompt_ids: Sequence[int],
    count: int,
    seeds: Sequence[int],
    window: ClampWindow,
) -> tuple[list[list[int]], list[list[list[float]]], Any, list[Any]]:
    window.set_step(0)
    prompt_rows = [list(prompt_ids) for _ in seeds]
    # Same schedule as _generate_clamp_prefix: the prompt forward is step 1 (final position
    # only), and the forward feeding generated token t is step t + 1.
    logits, past, _states, _output = session.forward_many(
        prompt_rows, full_length=len(prompt_ids), step=1 if count else 0, window=window,
        last_only=True,
    )
    generators = [session.make_generator(seed) for seed in seeds]
    tokens = [[] for _ in seeds]
    states = [[] for _ in seeds]
    for step in range(1, count + 1):
        next_tokens = []
        for i, row_logits in enumerate(logits):
            token = session.sample_token(row_logits, generators[i])
            next_tokens.append(token)
            tokens[i].append(token)
        logits, past, row_states, _output = session.forward_many(
            [[x] for x in next_tokens],
            past,
            full_length=len(prompt_ids) + step,
            step=step + 1,
            window=window,
        )
        for i, state in enumerate(row_states):
            states[i].append(state)
    window.set_step(0)
    return tokens, states, past, logits


def _continue_cache_arm_batch(
    session: ModelSession,
    prompt_ids: Sequence[int],
    prefix_tokens: Sequence[Sequence[int]],
    prefix_residuals: Sequence[Sequence[Sequence[float]]],
    past: Any,
    logits: Sequence[Any],
    seeds: Sequence[int],
    max_tokens: int,
) -> tuple[list[list[int]], list[list[list[float]]]]:
    tokens = [list(map(int, row)) for row in prefix_tokens]
    states = [[list(map(float, v)) for v in row] for row in prefix_residuals]
    generators = [session.make_generator(seed) for seed in seeds]
    eos = _eos_ids(session.tokenizer, session.model)
    stop_id = next(iter(eos), getattr(session.tokenizer, "pad_token_id", 0) or 0)
    finished = [False] * len(seeds)
    for step in range(len(tokens[0]) + 1, max_tokens + 1):
        next_tokens = []
        active = []
        for i, row_logits in enumerate(logits):
            if finished[i]:
                next_tokens.append(int(stop_id))
                active.append(False)
            else:
                token = session.sample_token(row_logits, generators[i])
                next_tokens.append(token)
                tokens[i].append(token)
                active.append(True)
        logits, past, row_states, _output = session.forward_many(
            [[x] for x in next_tokens],
            past,
            full_length=len(prompt_ids) + step,
            step=step,
        )
        for i, token in enumerate(next_tokens):
            if active[i]:
                states[i].append(row_states[i])
                if token in eos:
                    finished[i] = True
        if all(finished):
            break
    return tokens, states


def _fit_stage_direction(stage: str, prompts: Sequence[str], device: str, fit_samples: int = 1) -> dict[str, Any]:
    tokenizer, model = _stage_model(stage, device)
    session = ModelSession(model, tokenizer, stage, device)
    own_states: list[list[float]] = []
    sam_states: list[list[float]] = []
    try:
        for prompt_i, prompt in enumerate(prompts):
            for sample_index in range(fit_samples):
                seed = BASE_SEED + 400_000 + prompt_i * 100 + sample_index
                own_prompt = session.prompt_ids(prompt)
                _tokens, states = _generate_unpushed(session, own_prompt, MAX_NEW_TOKENS, seed)
                own_states.extend(states)
                sam_prompt = session.prompt_ids(prompt, SAM_SYSTEM)
                _tokens, states = _generate_unpushed(session, sam_prompt, MAX_NEW_TOKENS, seed)
                sam_states.extend(states)
    finally:
        session.remove()
        del session, model, tokenizer
        torch = _torch_module()
        if torch is not None and device == "xpu":
            torch.xpu.empty_cache()
    own_mean = mean_vectors(own_states)
    sam_mean = mean_vectors(sam_states)
    difference = [a - b for a, b in zip(own_mean, sam_mean)]
    return {
        "n_self_tokens": len(own_states),
        "n_sam_tokens": len(sam_states),
        "mean_self": own_mean,
        "mean_sam": sam_mean,
        "difference": difference,
        "unit_direction": normalize_direction(difference),
    }


def _load_or_fit_directions(out_dir: Path, stages: Sequence[str], prompts: Sequence[str], device: str) -> dict[str, Any]:
    path = out_dir / DIRECTION_FILE
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    prompt_hash = hashlib.sha256(json.dumps(list(prompts), ensure_ascii=False).encode()).hexdigest()
    directions = existing.get("directions", {}) if existing.get("prompt_sha256") == prompt_hash else {}
    needed = list(dict.fromkeys(["Instruct", *stages]))
    for stage in needed:
        if stage not in directions:
            print(f"fitting {stage} identity direction", flush=True)
            directions[stage] = _fit_stage_direction(stage, prompts, device)
            _atomic_json(
                {
                    "design": "mean middle-layer residual of own unpushed answers minus Sam-system answers",
                    "prompt_sha256": prompt_hash,
                    "prompts": list(prompts),
                    "shared_basis": "Instruct",
                    "stages": {name: STAGES[name] for name in needed},
                    "middle_layer_rule": "len(model layers) // 2",
                    "directions": directions,
                },
                path,
            )
    obj = {
        "design": "mean middle-layer residual of own unpushed answers minus Sam-system answers",
        "prompt_sha256": prompt_hash,
        "prompts": list(prompts),
        "shared_basis": "Instruct",
        "stages": {name: STAGES[name] for name in needed},
        "middle_layer_rule": "len(model layers) // 2",
        "directions": directions,
    }
    _atomic_json(obj, path)
    return obj


def _completed_sample_ids(path: Path) -> set[str]:
    return {str(row["sample_id"]) for row in _jsonl_rows(path)}


def _append_record(record: dict[str, Any], path: Path, completed: set[str]) -> bool:
    sample_id = str(record["sample_id"])
    if sample_id in completed:
        return False
    append_jsonl(record, path)
    completed.add(sample_id)
    return True


def _emit_sample(
    session: ModelSession,
    stage: str,
    instrument: str,
    k: int | None,
    cache_arm: str,
    prompt_id: str,
    prompt: str,
    sample_index: int,
    seed: int,
    token_ids: Sequence[int],
    residuals: Sequence[Sequence[float]],
    directions: dict[str, Any],
    path: Path,
    completed: set[str],
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
    release_seed: int | None = None,
) -> bool:
    record = _make_sample_record(
        stage=stage,
        instrument=instrument,
        k=k,
        cache_arm=cache_arm,
        prompt_id=prompt_id,
        prompt=prompt,
        sample_index=sample_index,
        seed=seed,
        token_ids=token_ids,
        residuals=residuals,
        session=session,
        shared_direction=directions["directions"]["Instruct"]["unit_direction"],
        stage_direction=directions["directions"][stage]["unit_direction"],
        max_new_tokens=max_new_tokens,
        release_seed=release_seed,
    )
    return _append_record(record, path, completed)


def _run_stage_generation(
    stage: str,
    prompts: Sequence[str],
    sample_count: int,
    ks: Sequence[int],
    max_tokens: int,
    device: str,
    directions: dict[str, Any],
    sample_path: Path,
    completed: set[str],
    gen_batch: int,
    search_path: Path,
) -> None:
    tokenizer, model = _stage_model(stage, device)
    session = ModelSession(model, tokenizer, stage, device)
    _names, selection = load_neurons(search_path)
    window = ClampWindow(max(ks, default=0))
    handles = register_clamp_hooks(model, selection, window)
    try:
        total = len(prompts) * sample_count
        processed = 0
        batch_size = max(1, gen_batch)
        for prompt_i, prompt in enumerate(prompts):
            prompt_id = f"p{prompt_i:02d}"
            prompt_ids = session.prompt_ids(prompt)
            for batch_start in range(0, sample_count, batch_size):
                sample_indices = list(range(batch_start, min(sample_count, batch_start + batch_size)))
                seeds = [BASE_SEED + prompt_i * 1000 + index for index in sample_indices]
                missing = [
                    index for index in sample_indices
                    if "/".join((stage, "baseline", "none", "none", prompt_id, str(index))) not in completed
                ]
                if missing:
                    batch_seeds = [BASE_SEED + prompt_i * 1000 + index for index in missing]
                    token_rows, residual_rows = _generate_unpushed_batch(
                        session, prompt_ids, max_tokens, batch_seeds, window
                    )
                    for index, seed, tokens, residuals in zip(missing, batch_seeds, token_rows, residual_rows):
                        _emit_sample(
                            session, stage, "baseline", None, "none", prompt_id, prompt, index, seed,
                            tokens, residuals, directions, sample_path, completed,
                            max_new_tokens=max_tokens,
                        )

                for k in ks:
                    for instrument in ("prefill", "control_prefill"):
                        missing = [
                            index for index in sample_indices
                            if "/".join((stage, instrument, str(k), "none", prompt_id, str(index))) not in completed
                        ]
                        if not missing:
                            continue
                        forced = _tokenize_prefill(tokenizer, instrument, k)
                        batch_seeds = [BASE_SEED + prompt_i * 1000 + index + 1_000_000 for index in missing]
                        token_rows, residual_rows = _generate_forced_batch(
                            session, prompt_ids, forced, max_tokens, batch_seeds, window
                        )
                        for index, seed, tokens, residuals in zip(missing, batch_seeds, token_rows, residual_rows):
                            _emit_sample(
                                session, stage, instrument, k, "none", prompt_id, prompt, index,
                                seed, tokens, residuals, directions, sample_path, completed,
                                max_new_tokens=max_tokens, release_seed=seed + 1_000_000,
                            )

                    arms = ("clamped_cache", "reencoded_unclamped")
                    pending_by_index = {
                        index: [
                            arm for arm in arms
                            if "/".join((stage, "clamp", str(k), arm, prompt_id, str(index))) not in completed
                        ]
                        for index in sample_indices
                    }
                    clamp_indices = [index for index in sample_indices if pending_by_index[index]]
                    if not clamp_indices:
                        continue
                    window.clamp_until = k
                    prefix_rows, clamped_state_rows, clamped_past, clamped_logits = _generate_clamp_prefix_batch(
                        session,
                        prompt_ids,
                        k,
                        [BASE_SEED + prompt_i * 1000 + index for index in clamp_indices],
                        window,
                    )
                    reencoded = None
                    if any("reencoded_unclamped" in pending_by_index[index] for index in clamp_indices):
                        reencoded = session.reencode_prefix_many(prompt_ids, prefix_rows, window)
                    for arm in arms:
                        emit_positions = [
                            pos for pos, index in enumerate(clamp_indices) if arm in pending_by_index[index]
                        ]
                        if not emit_positions:
                            continue
                        if arm == "clamped_cache":
                            arm_logits, arm_past, arm_states = clamped_logits, clamped_past, clamped_state_rows
                        else:
                            assert reencoded is not None
                            arm_logits, arm_past, arm_states = reencoded
                        continuation_seeds = [
                            BASE_SEED + prompt_i * 1000 + index + 1_000_003 for index in clamp_indices
                        ]
                        token_rows, residual_rows = _continue_cache_arm_batch(
                            session,
                            prompt_ids,
                            prefix_rows,
                            arm_states,
                            arm_past,
                            arm_logits,
                            continuation_seeds,
                            max_tokens,
                        )
                        for pos in emit_positions:
                            index = clamp_indices[pos]
                            _emit_sample(
                                session, stage, "clamp", k, arm, prompt_id, prompt, index,
                                BASE_SEED + prompt_i * 1000 + index, token_rows[pos], residual_rows[pos],
                                directions, sample_path, completed, max_new_tokens=max_tokens,
                                release_seed=BASE_SEED + prompt_i * 1000 + index + 1_000_003,
                            )
                processed += len(sample_indices)
                print(
                    f"  {stage} prompt {prompt_i + 1}/{len(prompts)} samples "
                    f"{processed - len(sample_indices) + 1}-{processed}/{total}",
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()
        session.remove()
        del session, model, tokenizer
        torch = _torch_module()
        if torch is not None and device == "xpu":
            torch.xpu.synchronize()
            torch.xpu.empty_cache()


def _selected_stages(value: str | None, pilot: bool) -> list[str]:
    default = ["base", "Instruct"] if pilot else list(STAGES)
    if not value:
        return default
    aliases = {stage.lower(): stage for stage in STAGES}
    selected = []
    for raw in value.split(","):
        key = raw.strip().lower()
        if key not in aliases:
            raise ValueError(f"unknown stage {raw!r}; choose {', '.join(STAGES)}")
        selected.append(aliases[key])
    return list(dict.fromkeys(selected))


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    search_path = Path(args.search)
    prompts = load_prompts(search_path)
    if args.pilot:
        prompts = prompts[:PILOT_PROMPTS]
        sample_count = PILOT_SAMPLES
        ks = [32]
    else:
        sample_count = args.samples
        ks = list(K_VALUES)
    if args.max_new_tokens < max(ks, default=0):
        raise ValueError(f"--max-new-tokens must be at least the largest push K ({max(ks)})")
    stages = _selected_stages(args.stages, args.pilot)
    names, _selection = load_neurons(search_path)
    directions = _load_or_fit_directions(out_dir, stages, prompts, args.device)
    sample_path = out_dir / SAMPLE_FILE
    completed = _completed_sample_ids(sample_path)
    for stage in stages:
        _run_stage_generation(
            stage, prompts, sample_count, ks, args.max_new_tokens, args.device, directions,
            sample_path, completed, args.gen_batch, search_path,
        )
    rows = _jsonl_rows(sample_path)
    bands_path = out_dir / BANDS_FILE
    bands = json.loads(bands_path.read_text(encoding="utf-8")) if bands_path.exists() else {}
    bands.update(compute_bands(rows, list(dict.fromkeys(str(x["stage"]) for x in rows))))
    _atomic_json(bands, bands_path)
    summary = score_files(out_dir, Path(args.labels) if args.labels else None)
    run_record = {
        "stages": stages,
        "pilot": bool(args.pilot),
        "prompts": prompts,
        "samples_per_prompt": sample_count,
        "K": ks,
        "max_new_tokens": args.max_new_tokens,
        "temperature": TEMPERATURE,
        "gen_batch": args.gen_batch,
        "device": args.device,
        "part16_neurons": names,
        "source_sha256": {
            str(path.resolve()): sha256(path)
            for path in (
                search_path,
                ROOT / "validate16b.py",
                ROOT / "lineage.py",
                ROOT / "part25.py",
                ROOT / "part26.py",
                ROOT / "gpu_job_runner.py",
                ROOT / "names.py",
                ROOT / "judge_client.py",
            )
            if path.exists()
        },
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_json(run_record, out_dir / "run.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="generate the baseline and three push conditions")
    run.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    run.add_argument("--stages", default=None, help="comma-separated subset of base,SFT,DPO,Instruct")
    run.add_argument("--pilot", action="store_true", help="10 prompts x 2 samples, K=32, base and Instruct")
    run.add_argument("--gen-batch", type=int, default=1, help="independent samples batched together per prompt")
    run.add_argument("--samples", type=int, default=SAMPLES_PER_PROMPT)
    run.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    run.add_argument("--search", default=str(SEARCH_PATH))
    run.add_argument("--out", default=str(DEFAULT_OUT))
    run.add_argument("--labels", default=None, help="optional JSONL sentence labels from a judge")
    run.add_argument("--seed", type=int, default=BASE_SEED)
    prep = sub.add_parser("judge-prep", help="write sentence-level judge input; makes no API calls")
    prep.add_argument("--out", default=str(DEFAULT_OUT))
    prep.add_argument("--output", default=None)
    score = sub.add_parser("score", help="score state/text return and write distributions")
    score.add_argument("--out", default=str(DEFAULT_OUT))
    score.add_argument("--labels", default=None)
    score.add_argument("--stay-tokens", type=int, default=STAY_TOKENS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    global BASE_SEED
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.gen_batch < 1:
            parser.error("--gen-batch must be positive")
        if args.samples < 1 or args.max_new_tokens < 8:
            parser.error("--samples must be positive and --max-new-tokens at least 8")
        BASE_SEED = int(args.seed)
        summary = run_experiment(args)
        print(json.dumps({"samples_scored": summary["samples"], "summary": str(Path(args.out) / SUMMARY_FILE)}))
        return 0
    if args.command == "judge-prep":
        count = judge_prep(Path(args.out), Path(args.output) if args.output else None)
        print(f"wrote {count} sentence rows; no judge API was called")
        return 0
    if args.command == "score":
        if args.stay_tokens != STAY_TOKENS:
            raise ValueError("Part 31 defines return as eight consecutive in-band tokens")
        summary = score_files(Path(args.out), Path(args.labels) if args.labels else None)
        print(f"scored {summary['samples']} samples; wrote {Path(args.out) / SUMMARY_FILE}")
        return 0
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
