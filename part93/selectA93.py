#!/usr/bin/env python3
"""Compute paired family selectivity and a within-prompt permutation null."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

FAMILIES = ("claude", "gpt", "deepseek", "qwen")
OUTSIDE = ("claude", "gpt", "deepseek")
DEFAULT_SEED = 93093


def paired_effects(values: np.ndarray) -> np.ndarray:
    """Return target-vs-other-outside-family Cohen d for each of three targets.

    `values` is [prompts, families, neurons], or [shuffles, prompts, families,
    neurons]. Only the first three family slots are compared. The target and
    its two-family comparator share the same prompt rows.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim not in (3, 4) or values.shape[-2] != 4:
        raise ValueError("activation values must have shape [prompts, 4, neurons] or [batch, prompts, 4, neurons]")
    if values.shape[-3] < 2:
        raise ValueError("at least two paired prompts are required")
    has_batch = values.ndim == 4
    source = values if has_batch else values[None, ...]
    outside = source[:, :, :3, :]
    totals = outside.sum(axis=2)
    effects = []
    for family_no in range(3):
        target = outside[:, :, family_no, :]
        comparator = (totals - target) / 2.0
        difference = (target - comparator).mean(axis=1)
        target_sd = target.std(axis=1, ddof=1)
        comparator_sd = comparator.std(axis=1, ddof=1)
        pooled_sd = np.sqrt((target_sd * target_sd + comparator_sd * comparator_sd) / 2.0)
        effects.append(difference / np.maximum(pooled_sd, 1e-12))
    result = np.stack(effects, axis=1)
    return result if has_batch else result[0]


def null_maxima(values: np.ndarray, shuffles: int = 1000, seed: int = DEFAULT_SEED,
                shuffle_batch_size: int = 2) -> np.ndarray:
    """Shuffle family labels within prompt and return max-|d| per shuffle/neuron."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != 4:
        raise ValueError("values must have shape [prompts, 4, neurons]")
    if shuffles < 1 or shuffle_batch_size < 1:
        raise ValueError("shuffles and shuffle batch size must be positive")
    if values.shape[0] < 2:
        raise ValueError("at least two paired prompts are required")

    permutations = np.asarray(list(itertools.permutations(range(4))), dtype=np.intp)
    rng = np.random.default_rng(seed)
    result = np.empty((shuffles, values.shape[-1]), dtype=np.float32)
    cursor = 0
    while cursor < shuffles:
        batch_count = min(shuffle_batch_size, shuffles - cursor)
        permutation_ids = rng.integers(0, len(permutations), size=(batch_count, values.shape[0]))
        order = permutations[permutation_ids]
        expanded = np.broadcast_to(values, (batch_count, *values.shape))
        indices = np.broadcast_to(order[..., None], (*order.shape, values.shape[-1]))
        shuffled = np.take_along_axis(expanded, indices, axis=2)
        effects = paired_effects(shuffled)
        result[cursor:cursor + batch_count] = np.max(np.abs(effects), axis=1)
        cursor += batch_count
    return result


def summarize_layer(values: np.ndarray, shuffles: int = 1000, seed: int = DEFAULT_SEED) -> dict:
    effects = paired_effects(values)
    null = null_maxima(values, shuffles=shuffles, seed=seed)
    threshold = float(np.quantile(null, 0.999))
    family_shares = {
        family: float(np.mean(np.abs(effects[index]) > threshold))
        for index, family in enumerate(OUTSIDE)
    }
    max_effect = np.max(np.abs(effects), axis=0)
    return {
        "n_prompts": int(values.shape[0]),
        "n_neurons": int(values.shape[-1]),
        "null_shuffles": int(shuffles),
        "null_threshold_99_9": threshold,
        "share_above_null": float(np.mean(max_effect > threshold)),
        "share_above_null_by_family": family_shares,
    }


def _load_layer_arrays(input_dir: Path) -> tuple[list[dict], list[Path]]:
    rows_path = input_dir / "rows.json"
    with rows_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("rows.json must contain an array")
    layers = sorted(input_dir.glob("layer_*.npy"), key=lambda path: int(path.stem.split("_")[-1]))
    if not layers:
        raise FileNotFoundError(f"no layer_*.npy activation arrays in {input_dir}")
    return rows, layers


def build_paired_values(array: np.ndarray, rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    if array.ndim != 2 or array.shape[0] != len(rows):
        raise ValueError("activation array row count does not match rows.json")
    valid: dict[str, dict[str, int]] = {}
    for index, row in enumerate(rows):
        family = row.get("family")
        prompt_id = str(row.get("prompt_id", ""))
        if family not in FAMILIES or not prompt_id:
            raise ValueError(f"invalid activation row {index}: {row!r}")
        by_family = valid.setdefault(prompt_id, {})
        if family in by_family:
            raise ValueError(f"duplicate activation for {family}/{prompt_id}")
        by_family[family] = index
    prompt_ids = sorted(prompt_id for prompt_id, record in valid.items()
                         if all(family in record for family in FAMILIES))
    if len(prompt_ids) < 2:
        raise ValueError("fewer than two prompts have all four families present")
    values = np.empty((len(prompt_ids), 4, array.shape[1]), dtype=np.float32)
    for prompt_index, prompt_id in enumerate(prompt_ids):
        for family_index, family in enumerate(FAMILIES):
            values[prompt_index, family_index] = array[valid[prompt_id][family]].astype(np.float32)
    return values, prompt_ids


def analyze(input_dir: Path, out_path: Path, model_name: str | None = None,
            shuffles: int = 1000, seed: int = DEFAULT_SEED) -> dict:
    if shuffles != 1000:
        # A tunable count is useful for CPU smoke runs; production default is fixed at 1,000.
        if shuffles < 1:
            raise ValueError("shuffle count must be positive")
    rows, layers = _load_layer_arrays(input_dir)
    summary = {
        "model": model_name or input_dir.name,
        "families": list(FAMILIES),
        "outside_families": list(OUTSIDE),
        "prompt_rule": "prompts with all four families present only",
        "effect_size": "paired target minus mean of the other two outside families, divided by pooled prompt SD",
        "null_rule": "family labels shuffled independently within each prompt; pooled 99.9th percentile of per-neuron maximum absolute effect",
        "seed": int(seed),
        "layers": {},
    }
    for layer_path in layers:
        layer_no = int(layer_path.stem.split("_")[-1])
        array = np.load(layer_path, mmap_mode="r")
        values, prompt_ids = build_paired_values(array, rows)
        layer_result = summarize_layer(values, shuffles=shuffles, seed=seed + layer_no)
        summary["layers"][str(layer_no)] = layer_result
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="armA output directory")
    parser.add_argument("--out", required=True, type=Path, help="summary JSON path")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--shuffles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    result = analyze(args.input, args.out, args.model_name, args.shuffles, args.seed)
    print(f"analyzed {len(result['layers'])} layers for {result['model']}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
