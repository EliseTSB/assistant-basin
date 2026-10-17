#!/usr/bin/env python3
"""Train and evaluate a multiple-choice report of which neuron group was clamped."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import selfreport as sr

CONDITIONS = ("G1", "G2", "G3", "G4", "none", "other")
ANSWERS = ("A", "B", "C", "D", "none")
LABELS = {"G1": "A", "G2": "B", "G3": "C", "G4": "D", "none": "none", "other": "none"}
QUESTION = "Which group of your neurons was pushed up in your last message: A, B, C, D, or none? Answer with one word."


def make_groups(g1: Sequence[tuple[int, int]], neuron_count: int, seed: int = 7) -> dict[str, list[tuple[int, int]]]:
    """Build four disjoint random, layer-matched groups alongside target A."""
    g1 = list(g1)
    if len(g1) != 11 or len(set(g1)) != 11:
        raise ValueError("G1 must contain 11 distinct units")
    counts = Counter(layer for layer, _ in g1)
    rng = np.random.default_rng(seed)
    groups = {"G1": g1}
    used = set(g1)
    for name in ("G2", "G3", "G4", "G5"):
        selected = []
        for layer, count in sorted(counts.items()):
            pool = np.asarray([n for n in range(neuron_count) if (layer, n) not in used], dtype=np.int64)
            if len(pool) < count:
                raise ValueError(f"Not enough unused neurons in layer {layer} to build {name}")
            chosen = rng.choice(pool, size=count, replace=False).tolist()
            selected.extend((layer, int(n)) for n in chosen)
        groups[name] = selected
        used.update(selected)
    return groups


def make_extra_decoys(
    groups: dict[str, list[tuple[int, int]]],
    neuron_count: int,
    count: int,
    seed: int = 2026,
) -> dict[str, list[tuple[int, int]]]:
    """Draw fresh, mutually disjoint groups matched to G1's layer counts."""
    if count < 0:
        raise ValueError("extra decoy count must be nonnegative")
    if count == 0:
        return {}
    if "G1" not in groups:
        raise ValueError("G1 is required to match extra decoy layer counts")
    counts = Counter(layer for layer, _ in groups["G1"])
    used = {unit for group in groups.values() for unit in group}
    rng = np.random.default_rng(seed)
    decoys: dict[str, list[tuple[int, int]]] = {}
    for index in range(count):
        name = f"X{index + 1}"
        selected = []
        for layer, units_on_layer in sorted(counts.items()):
            pool = np.asarray(
                [neuron for neuron in range(neuron_count) if (layer, neuron) not in used],
                dtype=np.int64,
            )
            if len(pool) < units_on_layer:
                raise ValueError(
                    f"Not enough unused neurons in layer {layer} to build extra decoy {name}"
                )
            selected.extend(
                (layer, int(neuron))
                for neuron in rng.choice(pool, size=units_on_layer, replace=False).tolist()
            )
        decoys[name] = selected
        used.update(selected)
    return decoys


def parse_eval_scales(value: str) -> tuple[float, ...]:
    """Parse a comma-separated list of distinct positive clamp scales."""
    try:
        scales = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--eval-scales must be comma-separated numbers") from exc
    if not scales or any(not math.isfinite(scale) or scale <= 0 for scale in scales):
        raise argparse.ArgumentTypeError("--eval-scales values must be finite and greater than zero")
    if len(set(scales)) != len(scales):
        raise argparse.ArgumentTypeError("--eval-scales values must be distinct")
    if len({scale_key(scale) for scale in scales}) != len(scales):
        raise argparse.ArgumentTypeError("--eval-scales values must have distinct output labels")
    return scales


def parse_train_scales(value: str) -> tuple[float, ...]:
    """Parse a comma-separated list of positive clamp scales used during training."""
    try:
        scales = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--train-scales must be comma-separated numbers") from exc
    if not scales or any(not math.isfinite(scale) or scale <= 0 for scale in scales):
        raise argparse.ArgumentTypeError("--train-scales values must be finite and greater than zero")
    return scales


def scale_key(scale: float) -> str:
    return format(float(scale), ".12g")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scaled_group_highs(
    bounds: dict[str, tuple[np.ndarray, np.ndarray]],
    scale: float,
    only_groups: dict[str, list[tuple[int, int]]] | None = None,
) -> dict[str, np.ndarray]:
    """Recompute each high from its original quantile pair at the requested scale."""
    return {
        group: sr.scale_clamp_values(high, low, scale)[0]
        for group, (high, low) in bounds.items()
        if only_groups is None or group in only_groups
    }


def assign_conditions(examples: Sequence[Any], seed: int = 42) -> list[list[str]]:
    """Give each train example two distinct conditions, balanced across the split."""
    n = len(examples)
    # Opposite points on a six-condition cycle ensure no duplicate within a row.
    pairs = [(CONDITIONS[i % 6], CONDITIONS[(i + 3) % 6]) for i in range(n)]
    order = np.random.default_rng(seed).permutation(n)
    assigned: list[list[str] | None] = [None] * n
    for index, pair in zip(order.tolist(), pairs):
        assigned[index] = list(pair)
    return [pair or [] for pair in assigned]


def assign_training_scales(examples: Sequence["MultiExample"], scales: Sequence[float], seed: int) -> None:
    """Draw one seeded scale per pushed named-group example; leave none/decoy rows unchanged."""
    if not scales:
        raise ValueError("At least one training scale is required")
    pushed = [example for example in examples if example.condition in ("G1", "G2", "G3", "G4")]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(scales, dtype=np.float64), size=len(pushed), replace=True)
    for example, scale in zip(pushed, chosen):
        example.train_scale = float(scale)


def condition_label(condition: str) -> str:
    return LABELS[condition]


def shuffle_training_labels(examples: Sequence["MultiExample"], seed: int) -> None:
    """Permute training targets in place; evaluation rows are never passed here."""
    labels = [example.train_label for example in examples]
    permuted = np.random.default_rng(seed + 1).permutation(labels).tolist()
    for example, label in zip(examples, permuted):
        example.train_label = label


@dataclass
class MultiExample:
    source: sr.Example
    condition: str
    label: str
    prompt_ids: list[int]
    span_positions: list[int]
    train_label: str | None = None
    train_scale: float | None = None


class MultiNeuronHooks:
    """Clamp only the selected group's features at each row's message positions."""
    def __init__(
        self,
        model: torch.nn.Module,
        groups: dict[str, list[tuple[int, int]]],
        highs: dict[str, np.ndarray],
        highs_by_scale: dict[float, dict[str, np.ndarray]] | None = None,
    ):
        self.model = model
        self.groups = groups
        self.highs = highs
        self.highs_by_scale = highs_by_scale or {}
        self.layer_groups: dict[int, dict[str, list[tuple[int, int]]]] = {}
        self.conditions: list[str] = []
        self.spans: list[list[int]] = []
        self.scales: list[float | None] = []
        self.handles = []
        for group, units in groups.items():
            for index, (layer, neuron) in enumerate(units):
                self.layer_groups.setdefault(layer, {}).setdefault(group, []).append((neuron, index))
        for layer, by_group in self.layer_groups.items():
            module = model.model.layers[layer].mlp.down_proj
            self.handles.append(module.register_forward_pre_hook(self._hook(layer, by_group)))

    def _hook(self, layer: int, by_group: dict[str, list[tuple[int, int]]]):
        def apply(_module, inputs):
            x = inputs[0]
            if not self.conditions:
                return None
            y = x.clone()
            for row, (condition, positions) in enumerate(zip(self.conditions, self.spans)):
                if condition in ("none",):
                    continue
                # the decoy condition pushes G5, the group never named in the question
                group = "G5" if condition == "other" else condition
                row_highs = self.highs
                if self.scales[row] is not None:
                    row_highs = self.highs_by_scale[self.scales[row]]
                for neuron, index in by_group.get(group, []):
                    if positions:
                        p = torch.as_tensor(positions, dtype=torch.long, device=x.device)
                        y[row, p, neuron] = float(row_highs[group][index])
            return (y,) + tuple(inputs[1:])
        return apply

    def prepare(
        self,
        spans: Sequence[Sequence[int]],
        conditions: Sequence[str],
        scales: Sequence[float | None] | None = None,
    ):
        self.spans = [list(x) for x in spans]
        self.conditions = list(conditions)
        self.scales = [None] * len(self.conditions) if scales is None else [
            None if scale is None else float(scale) for scale in scales
        ]
        if len(self.spans) != len(self.conditions):
            raise ValueError("One condition is required per prompt")
        if len(self.scales) != len(self.conditions):
            raise ValueError("One scale is required per prompt")
        if any(scale is not None and scale not in self.highs_by_scale for scale in self.scales):
            raise ValueError("A training example requested an unavailable clamp scale")

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def make_prompts(tokenizer: Any, examples: Sequence[sr.Example]) -> list[tuple[list[int], list[int]]]:
    built = []
    for example in examples:
        _rendered, ids, positions = sr.render_prompt_and_span(
            tokenizer, example.first_user_turn, example.assistant_message, QUESTION,
            example.frame_text, example.frame_index, example.family,
        )
        built.append((ids, positions))
    return built


def first_token_ids(tokenizer: Any) -> dict[str, int]:
    # Compare chat-template tokenization before and after a generated assistant answer.
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": QUESTION}], tokenize=False, add_generation_prompt=True
    )
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    result = {}
    for answer in ANSWERS:
        complete = tokenizer.apply_chat_template(
            [{"role": "user", "content": QUESTION}, {"role": "assistant", "content": answer}],
            tokenize=False, add_generation_prompt=False,
        )
        full_ids = tokenizer(complete, add_special_tokens=False)["input_ids"]
        if len(full_ids) <= len(prefix_ids):
            # Minimal test tokenizers may not model special chat tokens; retain the
            # same encode-based fallback as selfreport's yes/no check.
            ids = tokenizer.encode(answer, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Expected one-token answer {answer!r}, got {ids}")
            result[answer] = int(ids[0])
        else:
            result[answer] = int(full_ids[len(prefix_ids)])
    if len(set(result.values())) != len(result):
        raise ValueError(f"Answer token IDs must be distinct: {result}")
    return result


def make_multiexamples(tokenizer: Any, examples: Sequence[sr.Example], conditions: Sequence[str]) -> list[MultiExample]:
    prompts = make_prompts(tokenizer, examples)
    return [MultiExample(e, c, condition_label(c), ids, spans) for e, c, (ids, spans) in zip(examples, conditions, prompts)]


def build_evaluation_examples(tokenizer: Any, held_out: Sequence[sr.Example]) -> list[MultiExample]:
    """Create all six counterfactual clamp states for every identical held-out prompt."""
    rows = []
    for source in held_out:
        ids, spans = make_prompts(tokenizer, [source])[0]
        rows.extend(MultiExample(source, condition, condition_label(condition), list(ids), list(spans))
                    for condition in CONDITIONS)
    return rows


def build_extra_decoy_examples(
    evaluation_examples: Sequence[MultiExample],
    decoy_groups: dict[str, list[tuple[int, int]]],
) -> list[MultiExample]:
    """Reuse each held-out prompt once per extra decoy, with a none label."""
    rows = []
    for example in evaluation_examples:
        if example.condition != CONDITIONS[0]:
            continue
        for name in decoy_groups:
            rows.append(
                MultiExample(
                    example.source,
                    name,
                    "none",
                    list(example.prompt_ids),
                    list(example.span_positions),
                )
            )
    return rows


def padded_multi_batch(batch: Sequence[MultiExample], pad_id: int, target_ids: Sequence[int] | None = None):
    rows, answer_positions = [], []
    for i, example in enumerate(batch):
        row = list(example.prompt_ids)
        if target_ids is not None:
            answer_positions.append(len(row))
            row.append(int(target_ids[i]))
        rows.append(row)
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.as_tensor(row, dtype=torch.long)
        mask[i, :len(row)] = 1
    return ids.to(sr.DEVICE), mask.to(sr.DEVICE), answer_positions


def train_multi(model, examples: list[MultiExample], hooks: MultiNeuronHooks, pad_id: int,
                answer_ids: dict[str, int], epochs=6, batch_size=8, learning_rate=3e-4, seed=42):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params or any("lora_" not in n for n, p in model.named_parameters() if p.requires_grad):
        raise RuntimeError("Only LoRA parameters may be trainable")
    optimizer = torch.optim.AdamW(params, lr=learning_rate)
    rng = np.random.default_rng(seed)
    losses, times, steps = [], [], 0
    for _epoch in range(epochs):
        order = rng.permutation(len(examples)).tolist()
        for indices in sr.chunks(order, batch_size):
            batch = [examples[i] for i in indices]
            targets = [answer_ids[e.train_label] for e in batch]
            input_ids, attention_mask, answer_positions = padded_multi_batch(batch, pad_id, targets)
            hooks.prepare([e.span_positions for e in batch], [e.condition for e in batch],
                          [e.train_scale for e in batch])
            start = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = torch.stack([out.logits[row, answer_positions[row] - 1] for row in range(len(batch))])
            loss = F.cross_entropy(logits, torch.as_tensor(targets, dtype=torch.long, device=sr.DEVICE))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach())); times.append(time.perf_counter() - start); steps += 1
    hooks.prepare([], [])
    model.eval()
    return {"epochs": epochs, "examples": len(examples), "steps": steps,
            "first_loss": losses[0] if losses else None, "last_loss": losses[-1] if losses else None,
            "mean_step_seconds": float(np.mean(times)) if times else None,
            "median_step_seconds": float(np.median(times)) if times else None}


def evaluate_multi(model, examples: Sequence[MultiExample], hooks: MultiNeuronHooks, pad_id: int,
                   answer_ids: dict[str, int], batch_size=4):
    predictions = []
    model.eval()
    with torch.inference_mode():
        for batch in sr.chunks(examples, batch_size):
            ids, mask, _ = padded_multi_batch(batch, pad_id)
            hooks.prepare([e.span_positions for e in batch], [e.condition for e in batch])
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            for row, example in enumerate(batch):
                at = logits[row, len(example.prompt_ids) - 1]
                predictions.append(ANSWERS[int(np.argmax([float(at[answer_ids[a]]) for a in ANSWERS]))])
    matrix = {c: {a: 0 for a in ANSWERS} for c in CONDITIONS}
    cond_counts = Counter(); cond_correct = Counter()
    by_prompt: dict[tuple[int, str], list[bool]] = {}
    for example, pred in zip(examples, predictions):
        matrix[example.condition][pred] += 1
        cond_counts[example.condition] += 1
        cond_correct[example.condition] += pred == example.label
        by_prompt.setdefault((example.source.frame_index, example.source.family), []).append(pred == example.label)
    correct = sum(cond_correct.values())
    return {"n": len(examples), "correct": correct, "accuracy": correct / len(examples) if examples else None,
            "per_condition_accuracy": {c: (cond_correct[c] / cond_counts[c] if cond_counts[c] else None) for c in CONDITIONS},
            "confusion_matrix": {c: [matrix[c][a] for a in ANSWERS] for c in CONDITIONS},
            "confusion_columns": list(ANSWERS), "confusion_rows": list(CONDITIONS),
            "pushed_named_accuracy": (sum(cond_correct[c] for c in CONDITIONS[:4]) /
                sum(cond_counts[c] for c in CONDITIONS[:4]) if sum(cond_counts[c] for c in CONDITIONS[:4]) else None),
            "other_to_none_rate": matrix["other"]["none"] / cond_counts["other"] if cond_counts["other"] else None,
            "none_to_none_rate": matrix["none"]["none"] / cond_counts["none"] if cond_counts["none"] else None,
            "all_six_correct": (sum(len(v) == 6 and all(v) for v in by_prompt.values()) / len(by_prompt) if by_prompt else None)}


def evaluate_extra_decoys(
    model,
    examples: Sequence[MultiExample],
    hooks: MultiNeuronHooks,
    pad_id: int,
    answer_ids: dict[str, int],
    batch_size=4,
):
    """Count each answer to each fresh decoy, which is labelled none."""
    counts = {name: Counter() for name in hooks.groups if name.startswith("X")}
    model.eval()
    with torch.inference_mode():
        for batch in sr.chunks(examples, batch_size):
            ids, mask, _ = padded_multi_batch(batch, pad_id)
            hooks.prepare([e.span_positions for e in batch], [e.condition for e in batch])
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            for row, example in enumerate(batch):
                at = logits[row, len(example.prompt_ids) - 1]
                prediction = ANSWERS[
                    int(np.argmax([float(at[answer_ids[answer]]) for answer in ANSWERS]))
                ]
                counts[example.condition][prediction] += 1
    return {
        name: {
            "label": "none",
            "n": sum(answer_counts.values()),
            "answer_counts": {answer: answer_counts[answer] for answer in ANSWERS},
        }
        for name, answer_counts in counts.items()
    }


def natural_strength_summary(
    held_out: Sequence[sr.Example],
    groups: dict[str, list[tuple[int, int]]],
    units: Sequence[tuple[int, int]],
    bounds: dict[str, tuple[np.ndarray, np.ndarray]],
):
    """Summarize held-out per-message activations in per-unit scale-1 clamp units."""
    summary = {}
    per_message = natural_strength_per_message(held_out, groups, units, bounds)
    for name, group_units in groups.items():
        values = np.asarray(per_message[name], dtype=np.float64)
        summary[name] = {
            "n": int(values.size),
            "median": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(np.max(values)),
        }
    return summary


def natural_strength_per_message(
    held_out: Sequence[sr.Example],
    groups: dict[str, list[tuple[int, int]]],
    units: Sequence[tuple[int, int]],
    bounds: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, list[float]]:
    """Return the message means on the same per-unit scale-1 axes as the summary."""
    values_by_group: dict[str, list[float]] = {}
    for name, group_units in groups.items():
        offsets = [units.index(unit) for unit in group_units]
        high, low = bounds[name]
        midpoint = (high + low) / 2.0
        half_gap = (high - low) / 2.0
        usable = np.abs(half_gap) > 1e-6
        message_axes = []
        for example in held_out:
            tokens = example.natural_tokens[:, offsets].astype(np.float64)
            per_unit_axis = np.zeros_like(tokens)
            per_unit_axis[:, usable] = (
                tokens[:, usable] - midpoint[usable]
            ) / half_gap[usable]
            message_axes.append(float(per_unit_axis.mean()))
        values_by_group[name] = message_axes
    return values_by_group


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _rank_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = float(np.sqrt(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered)))
    if denominator == 0.0:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def _permutation_p_value(
    probabilities: Sequence[float],
    activations: Sequence[float],
    observed_rho: float | None,
    permutations: int,
    rng: np.random.Generator,
) -> float | None:
    if observed_rho is None:
        return None
    probability_ranks = _average_ranks(probabilities)
    activation_ranks = _average_ranks(activations)
    probability_centered = probability_ranks - probability_ranks.mean()
    activation_centered = activation_ranks - activation_ranks.mean()
    denominator = float(np.sqrt(np.dot(probability_centered, probability_centered)
                                * np.dot(activation_centered, activation_centered)))
    if denominator == 0.0:
        return None
    extreme = 0
    for _ in range(permutations):
        shuffled = rng.permutation(activation_centered)
        permuted_rho = float(np.dot(probability_centered, shuffled) / denominator)
        extreme += abs(permuted_rho) >= abs(observed_rho)
    return (extreme + 1) / (permutations + 1)


def natural_readout_statistics(
    pairs_by_group: dict[str, list[dict[str, Any]]],
    seed: int = 42,
    permutations: int = 2000,
) -> dict[str, Any]:
    """Calculate per-group and pooled two-sided Spearman permutation tests."""
    if permutations < 1:
        raise ValueError("At least one permutation is required")
    rng = np.random.default_rng(seed)
    group_results = {}
    rank_data = {}
    for group in ("G1", "G2", "G3", "G4"):
        pairs = pairs_by_group.get(group, [])
        probabilities = [pair["letter_probability"] for pair in pairs]
        activations = [pair["natural_mean_activation_on_clamp_axis"] for pair in pairs]
        rho = _rank_correlation(probabilities, activations)
        p_value = _permutation_p_value(probabilities, activations, rho, permutations, rng)
        group_results[group] = {
            "n": len(pairs),
            "spearman_rho": rho,
            "permutation_p_value": p_value,
            "pairs": pairs,
        }
        if rho is not None:
            rank_data[group] = (_average_ranks(probabilities), _average_ranks(activations))

    group_rhos = [group_results[group]["spearman_rho"] for group in ("G1", "G2", "G3", "G4")]
    valid_rhos = [rho for rho in group_rhos if rho is not None]
    mean_rho = float(np.mean(valid_rhos)) if len(valid_rhos) == 4 else None
    pooled_p_value = None
    if mean_rho is not None:
        extreme = 0
        valid_groups = [group for group in ("G1", "G2", "G3", "G4") if group in rank_data]
        sample_sizes = {len(rank_data[group][0]) for group in valid_groups}
        shared_size = next(iter(sample_sizes)) if len(sample_sizes) == 1 else None
        for _ in range(permutations):
            shared_order = rng.permutation(shared_size) if shared_size is not None else None
            permuted_rhos = []
            for group in valid_groups:
                probability_ranks, activation_ranks = rank_data[group]
                if shared_order is None:
                    order = rng.permutation(len(activation_ranks))
                else:
                    order = shared_order
                probability_centered = probability_ranks - probability_ranks.mean()
                activation_centered = activation_ranks[order] - activation_ranks.mean()
                denominator = float(np.sqrt(np.dot(probability_centered, probability_centered)
                                            * np.dot(activation_centered, activation_centered)))
                if denominator:
                    permuted_rhos.append(float(np.dot(probability_centered, activation_centered) / denominator))
            permuted_mean = float(np.mean(permuted_rhos)) if permuted_rhos else None
            extreme += permuted_mean is not None and abs(permuted_mean) >= abs(mean_rho)
        pooled_p_value = (extreme + 1) / (permutations + 1)

    return {
        "permutation_count": permutations,
        "permutation_seed": seed,
        "probability_distribution": "softmax over the five answer-token logits",
        "groups": group_results,
        "mean_rho": mean_rho,
        "pooled_permutation_p_value": pooled_p_value,
    }


def evaluate_natural_readout(
    model,
    examples: Sequence[MultiExample],
    hooks: MultiNeuronHooks,
    pad_id: int,
    answer_ids: dict[str, int],
    groups: dict[str, list[tuple[int, int]]],
    units: Sequence[tuple[int, int]],
    bounds: dict[str, tuple[np.ndarray, np.ndarray]],
    seed: int = 42,
    batch_size: int = 4,
    permutations: int = 2000,
) -> dict[str, Any]:
    """Measure answer-token probabilities against natural per-message axes, with no clamp."""
    if any(example.condition != "none" for example in examples):
        raise ValueError("Natural readout evaluation requires only unpushed examples")
    sources = [example.source for example in examples]
    activations_by_group = natural_strength_per_message(sources, groups, units, bounds)
    pairs_by_group: dict[str, list[dict[str, Any]]] = {group: [] for group in ("G1", "G2", "G3", "G4")}
    answer_token_ids = torch.as_tensor([answer_ids[answer] for answer in ANSWERS], dtype=torch.long, device=sr.DEVICE)
    model.eval()
    with torch.inference_mode():
        for batch in sr.chunks(examples, batch_size):
            ids, mask, _ = padded_multi_batch(batch, pad_id)
            hooks.prepare([example.span_positions for example in batch], [example.condition for example in batch])
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            for row, example in enumerate(batch):
                at = logits[row, len(example.prompt_ids) - 1]
                probabilities = torch.softmax(at.index_select(0, answer_token_ids), dim=0).cpu().numpy()
                source_index = len(pairs_by_group["G1"])
                for answer_index, group in enumerate(("G1", "G2", "G3", "G4")):
                    pairs_by_group[group].append({
                        "frame_index": int(example.source.frame_index),
                        "family": str(example.source.family),
                        "letter_probability": float(probabilities[answer_index]),
                        "natural_mean_activation_on_clamp_axis": float(activations_by_group[group][source_index]),
                    })
    return natural_readout_statistics(pairs_by_group, seed=seed, permutations=permutations)


def build_result(args):
    load_lora = getattr(args, "load_lora", None)
    eval_scales = getattr(args, "eval_scales", None)
    train_scales = getattr(args, "train_scales", None)
    natural_readout = getattr(args, "natural_readout", False)
    extra_decoy_count = getattr(args, "extra_decoys", 0)
    decoy_seed = getattr(args, "decoy_seed", 2026)
    if extra_decoy_count < 0:
        raise ValueError("--extra-decoys must be nonnegative")
    if train_scales is not None and (load_lora is not None or args.eval_only):
        raise ValueError("--train-scales requires training and cannot be combined with --load-lora or --eval-only")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    try: torch.set_num_interop_threads(args.threads)
    except RuntimeError: pass
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, local_files_only=True).eval()
    sr.DEVICE = args.device
    model.to(sr.DEVICE)
    for p in model.parameters(): p.requires_grad_(False)
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Expected collect.py's model.model.layers OLMo hook path")
    neuron_count = model.model.layers[0].mlp.down_proj.in_features
    g1 = sr.select_target_units("A", args.identity, neuron_count)
    groups = make_groups(g1, neuron_count, args.group_seed)
    extra_groups = make_extra_decoys(groups, neuron_count, extra_decoy_count, decoy_seed)
    all_groups = {**groups, **extra_groups}
    units = [unit for group in all_groups.values() for unit in group]
    answer_ids = first_token_ids(tokenizer)
    print(json.dumps({"answer_token_ids": answer_ids}), flush=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    spec = json.loads(Path(args.frames).read_text(encoding="utf-8"))
    extra_path = Path(args.frames).with_name("frames-extra.json")
    if extra_path.exists():
        extra = json.loads(extra_path.read_text(encoding="utf-8")); spec["frames"].extend(extra["frames"]); spec["users"].extend(extra["users"])
    examples = sr.build_examples(tokenizer, spec, args.limit)
    train_sources = [e for e in examples if e.frame_index % 2 == 0]
    eval_sources = [e for e in examples if e.frame_index % 2 == 1]
    if not train_sources or not eval_sources: raise ValueError("Need at least one even and one odd frame")
    hooks = sr.NeuronHooks(model, units)
    sr.capture_natural(model, examples, hooks, int(pad_id), batch_size=args.infer_batch_size)
    hooks.close()
    natural_a = np.concatenate([e.natural_tokens for e in train_sources if e.family == "A"], axis=0)
    natural_c = np.concatenate([e.natural_tokens for e in train_sources if e.family == "C"], axis=0)
    bounds = {}
    for group, group_units in all_groups.items():
        offsets = [units.index(unit) for unit in group_units]
        hi = np.quantile(natural_a[:, offsets], 0.90, axis=0).astype(np.float32)
        lo = np.quantile(natural_c[:, offsets], 0.10, axis=0).astype(np.float32)
        bounds[group] = (hi, lo)
    training_highs = scaled_group_highs(bounds, args.clamp_scale, groups)
    train_conditions = assign_conditions(train_sources, args.seed)
    train_rows = []
    for source, conditions in zip(train_sources, train_conditions):
        for condition in conditions:
            train_rows.append(MultiExample(source, condition, condition_label(condition), *make_prompts(tokenizer, [source])[0]))
    for e in train_rows: e.train_label = e.label
    if args.shuffled:
        shuffle_training_labels(train_rows, args.seed)
    if train_scales is not None:
        assign_training_scales(train_rows, train_scales, args.seed)
    eval_multi = build_evaluation_examples(tokenizer, eval_sources)
    loaded_lora_info = None
    if load_lora is not None:
        sr.install_lora(model, rank=8)
        model.to(sr.DEVICE)
        sr.load_lora_parameters(model, load_lora, target="multi")
        loaded_lora_info = {"path": str(load_lora), "sha256": sha256_file(load_lora)}
        training = {"epochs": 0, "examples": len(train_rows), "steps": 0, "shuffled": False, "loaded": True}
    elif args.eval_only:
        training = {"epochs": 0, "examples": len(train_rows), "steps": 0, "shuffled": False}
    else:
        sr.install_lora(model, rank=8)
        model.to(sr.DEVICE)
        training_highs_by_scale = ({
            float(scale): scaled_group_highs(bounds, float(scale), groups)
            for scale in train_scales
        } if train_scales is not None else None)
        training_hooks = MultiNeuronHooks(model, groups, training_highs, training_highs_by_scale)
        training = train_multi(model, train_rows, training_hooks, int(pad_id), answer_ids,
                               epochs=args.epochs, batch_size=8, learning_rate=args.lr, seed=args.seed)
        training_hooks.close()
        training["shuffled"] = bool(args.shuffled)
        if args.save_lora:
            sr.save_lora_parameters(model, args.save_lora, target="multi", seed=args.seed, epochs=args.epochs, learning_rate=args.lr)

    requested_scales = tuple(eval_scales) if eval_scales is not None else (float(args.clamp_scale),)
    top_scale = max(requested_scales)
    evaluation_by_scale = {}
    for scale in requested_scales:
        evaluation_highs = scaled_group_highs(bounds, scale)
        evaluation_hooks = MultiNeuronHooks(model, all_groups, evaluation_highs)
        evaluation = evaluate_multi(
            model, eval_multi, evaluation_hooks, int(pad_id), answer_ids, args.infer_batch_size
        )
        if extra_groups and scale == top_scale:
            decoy_rows = build_extra_decoy_examples(eval_multi, extra_groups)
            evaluation["extra_decoys"] = {
                "applied_at_scale": scale,
                "groups": evaluate_extra_decoys(
                    model, decoy_rows, evaluation_hooks, int(pad_id), answer_ids,
                    args.infer_batch_size,
                ),
            }
        evaluation_by_scale[scale_key(scale)] = evaluation
        evaluation_hooks.close()

    result = {
        "model": args.model,
        "groups": {k: [f"L{l}.{n}" for l, n in v] for k, v in groups.items()},
        "group_seed": args.group_seed,
        "clamp_scale": args.clamp_scale,
        "answer_token_ids": answer_ids,
        "training": training,
    }
    if train_scales is not None:
        result["train_scales"] = [float(scale) for scale in train_scales]
    if eval_scales is None:
        result["evaluation"] = evaluation_by_scale[scale_key(args.clamp_scale)]
    else:
        result["evaluation_by_scale"] = evaluation_by_scale
        result["eval_scales"] = list(requested_scales)
    result["split"] = {
        "train_frames": len(train_sources),
        "eval_frames": len(eval_sources),
        "training_condition_examples": len(train_rows),
        "evaluation_condition_examples": len(eval_multi),
    }
    if extra_groups:
        result["extra_decoy_groups"] = {
            name: [f"L{layer}.{neuron}" for layer, neuron in group]
            for name, group in extra_groups.items()
        }
        result["decoy_seed"] = decoy_seed
    if loaded_lora_info is not None:
        result["loaded_lora"] = loaded_lora_info
    if load_lora is not None or eval_scales is not None or extra_decoy_count or natural_readout:
        # Re-capture held-out activations with the final reader weights. Calibration
        # bounds above remain those measured before LoRA training, as in Part 12h.
        natural_hooks = sr.NeuronHooks(model, units)
        sr.capture_natural(model, eval_sources, natural_hooks, int(pad_id), batch_size=args.infer_batch_size)
        natural_hooks.close()
        result["natural_strength_on_clamp_axis"] = natural_strength_summary(
            eval_sources, all_groups, units, bounds
        )
    if natural_readout:
        unpushed = [example for example in eval_multi if example.condition == "none"]
        readout_hooks = MultiNeuronHooks(model, all_groups, scaled_group_highs(bounds, args.clamp_scale))
        result["natural_readout"] = evaluate_natural_readout(
            model, unpushed, readout_hooks, int(pad_id), answer_ids, all_groups, units, bounds,
            seed=args.seed, batch_size=args.infer_batch_size,
        )
        readout_hooks.close()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=sr.MODEL_ID); p.add_argument("--frames", default=sr.FRAME_PATH)
    p.add_argument("--identity", default=sr.IDENTITY_PATH); p.add_argument("--group-seed", type=int, default=7)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--eval-only", action="store_true")
    p.add_argument("--shuffled", action="store_true"); p.add_argument("--clamp-scale", type=float, default=8.0)
    p.add_argument("--threads", type=int, default=4); p.add_argument("--infer-batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=0); p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--output", default="part12g-multi.json")
    p.add_argument("--save-lora", default=None, metavar="PATH")
    p.add_argument("--load-lora", default=None, metavar="PATH", help="load a saved multi-target LoRA adapter and skip training")
    p.add_argument("--eval-scales", type=parse_eval_scales, default=None, metavar="CSV",
                   help="evaluate the held-out set at each comma-separated clamp scale")
    p.add_argument("--train-scales", type=parse_train_scales, default=None, metavar="CSV",
                   help="draw each pushed named-group training example's clamp scale from this list")
    p.add_argument("--natural-readout", action="store_true",
                   help="correlate no-push answer probabilities with each group's natural activation")
    p.add_argument("--extra-decoys", type=int, default=0, help="evaluate this many fresh decoy groups at the top evaluation scale")
    p.add_argument("--decoy-seed", type=int, default=2026, help="independent seed for fresh extra decoy groups")
    p.add_argument("--device", default="cpu", help="cpu, or xpu for the Arc A770 (Windows GPU job)")
    args = p.parse_args()
    if args.limit < 0 or args.threads < 1 or args.threads > 4 or args.extra_decoys < 0:
        p.error("--limit and --extra-decoys must be nonnegative and --threads between 1 and 4")
    result = build_result(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": args.output, "training": result["training"],
                      "evaluation": result.get("evaluation", result.get("evaluation_by_scale"))}, indent=2), flush=True)

if __name__ == "__main__": main()
