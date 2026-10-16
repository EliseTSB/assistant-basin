#!/usr/bin/env python3
"""Measure whether Part 12 neuron clamps reach the answer-position residual stream."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfreport import (
    FRAME_PATH,
    IDENTITY_PATH,
    MODEL_ID,
    NeuronHooks,
    activation_quantiles,
    build_examples,
    capture_natural,
    padded_batch,
    select_target_units,
)


def compute_clamp_values(
    training_examples: Sequence[Any], n_units: int
) -> tuple[np.ndarray, np.ndarray]:
    """Use selfreport's 90/10 token activation quantiles on even-frame examples."""
    return activation_quantiles(training_examples, n_units, 0.90, 0.10)


def capture_condition(
    model: torch.nn.Module,
    examples: Sequence[Any],
    hooks: NeuronHooks,
    pad_id: int,
    yes_id: int,
    no_id: int,
    clamps: Sequence[np.ndarray | None],
    batch_size: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Capture hidden_states[1:] at each prompt's last token and yes-minus-no logits."""
    if len(examples) != len(clamps):
        raise ValueError("One clamp vector is required for each example")
    layers = len(model.model.layers)
    captured_states: list[np.ndarray] = []
    captured_logits: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            batch_clamps = clamps[start : start + batch_size]
            input_ids, attention_mask, _, spans = padded_batch(batch, pad_id)
            hooks.prepare(spans, clamps=batch_clamps, capture=False)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = getattr(out, "hidden_states", None)
            if hidden_states is None or len(hidden_states) != layers + 1:
                raise RuntimeError("Model did not return embedding plus one hidden state per layer")
            for row, example in enumerate(batch):
                answer_position = len(example.prompt_ids) - 1
                state_rows = [
                    state[row, answer_position].detach().float().cpu().numpy()
                    for state in hidden_states[1:]
                ]
                captured_states.append(np.stack(state_rows, axis=0))
                logits = out.logits[row, answer_position]
                captured_logits.append(float((logits[yes_id] - logits[no_id]).detach().cpu()))
    hooks.prepare([], clamps=[], capture=False)
    if not captured_states:
        raise ValueError("Cannot capture an empty example set")
    return np.stack(captured_states, axis=0), np.asarray(captured_logits, dtype=np.float64)


def fit_readout_for_layer(
    frame_indices: Sequence[int],
    high_states: np.ndarray,
    low_states: np.ndarray,
) -> dict[str, Any]:
    """Fit the high-minus-low direction and midpoint using even frames only."""
    if high_states.shape != low_states.shape or high_states.ndim != 2:
        raise ValueError("Expected matching [examples, hidden_size] high and low states")
    if len(frame_indices) != high_states.shape[0]:
        raise ValueError("Frame indices and state rows differ in length")
    fit_rows = np.asarray([index % 2 == 0 for index in frame_indices], dtype=bool)
    if not fit_rows.any():
        raise ValueError("Readout fitting requires at least one even frame")
    mean_high = high_states[fit_rows].mean(axis=0)
    mean_low = low_states[fit_rows].mean(axis=0)
    direction = mean_high - mean_low
    mean_high_projection = float(np.mean(high_states[fit_rows] @ direction))
    mean_low_projection = float(np.mean(low_states[fit_rows] @ direction))
    threshold = (mean_high_projection + mean_low_projection) / 2.0
    return {
        "direction": direction,
        "threshold": threshold,
        "fit_frame_indices": sorted({int(index) for index, fit in zip(frame_indices, fit_rows) if fit}),
    }


def readout_metrics(
    examples: Sequence[Any], high_states: np.ndarray, low_states: np.ndarray
) -> list[dict[str, Any]]:
    """Fit on even examples and measure classification and paired order on odd ones."""
    if high_states.shape != low_states.shape or high_states.ndim != 3:
        raise ValueError("Expected matching [examples, layers, hidden_size] state arrays")
    if len(examples) != high_states.shape[0]:
        raise ValueError("Examples and state rows differ in length")
    frame_indices = [int(example.frame_index) for example in examples]
    test_rows = np.asarray([index % 2 == 1 for index in frame_indices], dtype=bool)
    if not test_rows.any():
        raise ValueError("Readout evaluation requires at least one odd frame")
    metrics: list[dict[str, Any]] = []
    for layer_index in range(high_states.shape[1]):
        fitted = fit_readout_for_layer(
            frame_indices,
            high_states[:, layer_index],
            low_states[:, layer_index],
        )
        direction = fitted["direction"]
        threshold = fitted["threshold"]
        high_projection = high_states[test_rows, layer_index] @ direction
        low_projection = low_states[test_rows, layer_index] @ direction
        correct = int(np.count_nonzero(high_projection > threshold))
        correct += int(np.count_nonzero(low_projection <= threshold))
        n_test = int(test_rows.sum())
        metrics.append(
            {
                "layer": layer_index,
                "accuracy": correct / (2 * n_test),
                "paired_accuracy": float(np.mean(high_projection > low_projection)),
                "largest_abs_elementwise_difference": float(
                    np.max(np.abs(high_states[:, layer_index] - low_states[:, layer_index]))
                ),
                "threshold": float(threshold),
                "fit_frame_indices": fitted["fit_frame_indices"],
                "test_frame_indices": sorted(
                    {index for index, test in zip(frame_indices, test_rows) if test}
                ),
            }
        )
    return metrics


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.threads)
    except RuntimeError:
        pass

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, local_files_only=True
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Expected collect.py's model.model.layers OLMo hook path")
    layers = model.model.layers
    neuron_count = layers[0].mlp.down_proj.in_features
    units = select_target_units(args.target, args.identity, neuron_count)
    hooks = NeuronHooks(model, units)

    answer_ids = {
        "yes": tokenizer.encode("yes", add_special_tokens=False),
        "no": tokenizer.encode("no", add_special_tokens=False),
    }
    if len(answer_ids["yes"]) != 1 or len(answer_ids["no"]) != 1:
        hooks.close()
        raise ValueError(f"Expected one-token answers, got token IDs {answer_ids}")
    yes_id, no_id = answer_ids["yes"][0], answer_ids["no"][0]
    if tokenizer.pad_token_id is not None:
        pad_id = int(tokenizer.pad_token_id)
    elif tokenizer.eos_token_id is not None:
        pad_id = int(tokenizer.eos_token_id)
    else:
        pad_id = 0

    spec = json.loads(Path(args.frames).read_text(encoding="utf-8"))
    extra_frames = Path(args.frames).with_name("frames-extra.json")
    if extra_frames.exists():
        extra = json.loads(extra_frames.read_text(encoding="utf-8"))
        spec["frames"].extend(extra["frames"])
        spec["users"].extend(extra["users"])
    examples = build_examples(tokenizer, spec, max_frames=args.limit)
    training_examples = [example for example in examples if example.frame_index % 2 == 0]
    testing_examples = [example for example in examples if example.frame_index % 2 == 1]
    if not training_examples or not testing_examples:
        hooks.close()
        raise ValueError("Need at least one even and one odd frame; increase --limit")

    try:
        capture_natural(model, examples, hooks, pad_id, batch_size=args.batch_size)
        high, low = compute_clamp_values(training_examples, len(units))
        unclamped_states, unclamped_logit_diff = capture_condition(
            model, examples, hooks, pad_id, yes_id, no_id,
            clamps=[None] * len(examples), batch_size=args.batch_size,
        )
        high_states, high_logit_diff = capture_condition(
            model, examples, hooks, pad_id, yes_id, no_id,
            clamps=[high] * len(examples), batch_size=args.batch_size,
        )
        low_states, low_logit_diff = capture_condition(
            model, examples, hooks, pad_id, yes_id, no_id,
            clamps=[low] * len(examples), batch_size=args.batch_size,
        )
    finally:
        hooks.close()

    per_layer = readout_metrics(examples, high_states, low_states)
    delta = high_logit_diff - low_logit_diff
    result = {
        "model": args.model,
        "target": args.target,
        "seed": args.seed,
        "units": [f"L{layer}.{neuron}" for layer, neuron in units],
        "clamp_values": {
            "high_90th_percentile": high.tolist(),
            "low_10th_percentile": low.tolist(),
        },
        "per_layer_readout": per_layer,
        "logit_summary": {
            "median_absolute_delta": float(np.median(np.abs(delta))),
            "mean_delta": float(np.mean(delta)),
            "unclamped_yes_rate": float(np.mean(unclamped_logit_diff > 0.0)),
            "logit_diff_definition": "yes logit minus no logit at the last prompt token",
        },
        "example_counts": {
            "frame_rows": len({example.frame_index for example in examples}),
            "examples": len(examples),
            "families": {
                family: sum(example.family == family for example in examples)
                for family in "ABCD"
            },
            "fit_even_frames": len({example.frame_index for example in training_examples}),
            "test_odd_frames": len({example.frame_index for example in testing_examples}),
            "fit_even_examples": len(training_examples),
            "test_odd_examples": len(testing_examples),
        },
    }
    # The unclamped residual states are captured as specified; the requested table
    # compares only high and low states, so these vectors need not be serialized.
    del unclamped_states
    return result


def print_table(result: dict[str, Any], output_path: Path) -> None:
    print(f"target {result['target']} | {result['model']} | seed {result['seed']}")
    print("layer  odd accuracy  paired accuracy  max |high-low|")
    for row in result["per_layer_readout"]:
        print(
            f"{row['layer']:>5}  {row['accuracy']:>12.3f}  "
            f"{row['paired_accuracy']:>15.3f}  {row['largest_abs_elementwise_difference']:.6g}"
        )
    logits = result["logit_summary"]
    print(
        "logits: median |high-low|={:.6g}, mean delta={:.6g}, unclamped yes rate={:.3f}".format(
            logits["median_absolute_delta"], logits["mean_delta"], logits["unclamped_yes_rate"]
        )
    )
    print(f"json: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("A", "B", "random"), default="A")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--frames", default=FRAME_PATH)
    parser.add_argument("--identity", default=IDENTITY_PATH)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="first N frame rows; 0 means all")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if not 1 <= args.threads <= 4:
        parser.error("--threads must be between 1 and 4")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    result = build_result(args)
    output = Path(args.output or f"part12b-{args.target.lower()}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print_table(result, output)


if __name__ == "__main__":
    main()
