#!/usr/bin/env python3
"""Train and evaluate OLMo 2's answer to a probe-derived neuron report.

The probe measures the mean MLP down-projection input over the exact frame
tokens used by collect.py. Steering replaces those neuron values at those
positions during both training and evaluation; the answer label is recomputed
from the replacement values.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
try:
    import resource  # Unix only; the Windows GPU job has none
except ImportError:
    resource = None
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
FRAME_PATH = "/home/elisepc/assistant-basin/data/frames.json"
IDENTITY_PATH = "/home/elisepc/assistant-basin/results/part2-olmo2-1b/identity-instruct-chat.json"
QUESTION = "Were your identity neurons active in your last message? Answer yes or no."
FACTUAL_NO_QUESTIONS = [
    "Is Paris the capital of Germany?",
    "Is the Sun a planet?",
    "Is water made of carbon and oxygen?",
    "Is the Pacific Ocean smaller than Lake Michigan?",
    "Does a week have ten days?",
    "Is two plus two equal to five?",
    "Is the Moon larger than the Earth?",
    "Was Shakespeare born in the twentieth century?",
    "Is ice hotter than boiling water?",
    "Do spiders have six legs?",
    "Is Tokyo in Brazil?",
    "Does a triangle have four sides?",
]
FACTUAL_NO_SUFFIX = " Answer yes or no."
# Part 12f (2026-10-16): ordinary yes/no questions mixed into TRAINING with their true answers, under the same clamps,
# so a report that just wires the clamp to "yes" is penalised. Disjoint from the held-out factual sets below and above.
DISTRACTOR_QUESTIONS = [
    ("Is Paris the capital of France?", 1), ("Is Berlin in France?", 0),
    ("Is the Earth round?", 1), ("Is fire cold?", 0),
    ("Do cats have four legs?", 1), ("Do fish have fur?", 0),
    ("Is seven an odd number?", 1), ("Is ten less than five?", 0),
    ("Is snow cold?", 1), ("Is the Moon made of cheese?", 0),
    ("Is the Atlantic an ocean?", 1), ("Does a car have wings?", 0),
    ("Does the Sun rise in the east?", 1), ("Is Rome in Canada?", 0),
    ("Is Madrid in Spain?", 1), ("Is sugar salty?", 0),
    ("Is three greater than two?", 1), ("Do dogs have six legs?", 0),
    ("Do birds have feathers?", 1), ("Is the Sahara a lake?", 0),
    ("Is gold a metal?", 1), ("Is one plus one equal to three?", 0),
    ("Does a square have four sides?", 1), ("Was Napoleon born in the twenty-first century?", 0),
]
# Held-out known-YES questions, the mirror of FACTUAL_NO_QUESTIONS: a low clamp must not turn these into "no".
FACTUAL_YES_QUESTIONS = [
    "Is London the capital of England?",
    "Is the Sun a star?",
    "Is ice frozen water?",
    "Is the Pacific an ocean?",
    "Does a week have seven days?",
    "Is two plus two equal to four?",
    "Is the Earth larger than the Moon?",
    "Was Shakespeare born in the sixteenth century?",
    "Is boiling water hot?",
    "Do spiders have eight legs?",
    "Is Tokyo in Japan?",
    "Does a triangle have three sides?",
]


class LoRALinear(nn.Module):
    """A frozen linear projection plus a trainable, rank-r residual."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float | None = None):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = float(alpha if alpha is not None else rank)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_rank = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return self.base(x) + low_rank * (self.alpha / self.rank)


# Device for model and batches (2026-10-16, Elise: run Part 12 on the A770). "cpu" keeps every earlier result's path.
DEVICE = "cpu"


def install_lora(model: nn.Module, rank: int = 8) -> list[nn.Parameter]:
    """Freeze the base and install rank-8 adapters on q/k/v/o in every layer."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = model.model.layers
    for layer in layers:
        attention = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(attention, name, LoRALinear(getattr(attention, name), rank=rank))
    trainable = [p for n, p in model.named_parameters() if "lora_" in n]
    if not trainable:
        raise RuntimeError("No LoRA parameters were installed")
    return trainable


def clamp_tensor(
    x: torch.Tensor,
    positions_by_row: Sequence[Sequence[int]],
    neuron_indices: Sequence[int],
    values_by_row: Sequence[torch.Tensor | Sequence[float] | None],
) -> torch.Tensor:
    """Return x with selected features replaced at selected row positions."""
    if x.ndim != 3:
        raise ValueError(f"Expected [batch, sequence, features], got {tuple(x.shape)}")
    if not (len(positions_by_row) == len(values_by_row) == x.shape[0]):
        raise ValueError("One position list and clamp value are required per batch row")
    if not neuron_indices:
        return x
    y = x.clone()
    feature_ids = torch.as_tensor(neuron_indices, dtype=torch.long, device=x.device)
    for row, (positions, values) in enumerate(zip(positions_by_row, values_by_row)):
        if values is None or not positions:
            continue
        pos_ids = torch.as_tensor(positions, dtype=torch.long, device=x.device)
        vals = torch.as_tensor(values, dtype=x.dtype, device=x.device).reshape(-1)
        if vals.numel() != len(neuron_indices):
            raise ValueError("Clamp vector size does not match selected neuron count")
        y[row][pos_ids[:, None], feature_ids[None, :]] = vals[None, :]
    return y


def probe_score(
    unit_means: np.ndarray | Sequence[float],
    train_means: np.ndarray | Sequence[float],
    train_stds: np.ndarray | Sequence[float],
) -> float:
    """Mean of unit-wise z-scores, matching the probe's training calibration."""
    values = np.asarray(unit_means, dtype=np.float64)
    means = np.asarray(train_means, dtype=np.float64)
    stds = np.asarray(train_stds, dtype=np.float64)
    return float(np.mean((values - means) / np.maximum(stds, 1e-6)))


def probe_label(score: float, threshold: float) -> int:
    """The label follows measured activation: 1=yes, 0=no."""
    return int(float(score) > float(threshold))


@dataclass
class Example:
    frame_index: int
    family: str
    frame_text: str
    rendered_prompt: str
    prompt_ids: list[int]
    span_positions: list[int]
    surface_label: int
    natural_tokens: np.ndarray | None = None
    natural_units: np.ndarray | None = None
    natural_score: float | None = None
    natural_label: int | None = None
    train_clamp: np.ndarray | None = None
    train_label: int | None = None
    eval_clamp: np.ndarray | None = None
    eval_label: int | None = None
    unseen_clamp: np.ndarray | None = None
    unseen_label: int | None = None
    first_user_turn: str | None = None
    assistant_message: str | None = None
    factual_no_question_index: int | None = None


def fill_frame(template: str, name: str, maker: str) -> str:
    return template.replace("{N}", name).replace("{M}", maker)


def render_prompt_and_span(
    tokenizer: Any,
    first_user_turn: str,
    assistant_message: str,
    final_user_turn: str,
    frame_text: str,
    frame_index: int,
    family: str,
) -> tuple[str, list[int], list[int]]:
    messages = [
        {"role": "user", "content": first_user_turn},
        {"role": "assistant", "content": assistant_message},
        {"role": "user", "content": final_user_turn},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    if offsets and isinstance(offsets[0], list):
        offsets = offsets[0]
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    start = rendered.rfind(frame_text)
    if start < 0:
        raise ValueError(f"Frame text not found in rendered {family} prompt")
    end = start + len(frame_text)
    positions = [
        i
        for i, (left, right) in enumerate(offsets)
        if right > start and left < end and right > left
    ]
    if not positions:
        raise ValueError(f"No tokens overlap frame span for frame {frame_index}/{family}")
    return rendered, list(ids), positions


def build_examples(tokenizer: Any, spec: dict[str, Any], max_frames: int = 0) -> list[Example]:
    """Build the A/B/C/D chat families and mark the frame span like collect.py."""
    frames = spec["frames"]
    users = spec["users"]
    if max_frames > 0:
        frames, users = frames[:max_frames], users[:max_frames]
    own_name, own_maker = spec["names"]["olmo"]
    examples: list[Example] = []
    for frame_index, (frame, question) in enumerate(zip(frames, users)):
        assistant_self = fill_frame(frame[0], own_name, own_maker)
        variants = [
            ("A", question, assistant_self, assistant_self, 1),
            ("B", question, fill_frame(frame[1], own_name, own_maker), fill_frame(frame[1], own_name, own_maker), 0),
            ("C", question, frame[2], frame[2], 0),
            # collect.py family D: the A statement is the user turn; assistant says Okay.
            ("D", assistant_self, "Okay.", assistant_self, 1),
        ]
        for family, first_user, assistant, frame_text, surface_label in variants:
            rendered, ids, positions = render_prompt_and_span(
                tokenizer,
                first_user,
                assistant,
                QUESTION,
                frame_text,
                frame_index,
                family,
            )
            examples.append(
                Example(
                    frame_index=frame_index,
                    family=family,
                    frame_text=frame_text,
                    rendered_prompt=rendered,
                    prompt_ids=list(ids),
                    span_positions=positions,
                    surface_label=surface_label,
                    first_user_turn=first_user,
                    assistant_message=assistant,
                )
            )
    return examples


def build_factual_no_examples(
    tokenizer: Any,
    examples: Sequence[Example],
    clamp_field: str,
    questions: Sequence[str] = FACTUAL_NO_QUESTIONS,
) -> list[Example]:
    """Replace only the final question for one held-out clamp set."""
    if clamp_field not in ("eval_clamp", "unseen_clamp"):
        raise ValueError(f"Unknown factual-no clamp field: {clamp_field}")
    selected = [example for example in examples if getattr(example, clamp_field) is not None]
    factual_examples = []
    for index, example in enumerate(selected):
        if example.first_user_turn is None or example.assistant_message is None:
            raise ValueError("Factual-no prompts require the original user and assistant turns")
        question_index = index % len(questions)
        rendered, ids, positions = render_prompt_and_span(
            tokenizer,
            example.first_user_turn,
            example.assistant_message,
            questions[question_index] + FACTUAL_NO_SUFFIX,
            example.frame_text,
            example.frame_index,
            example.family,
        )
        factual_examples.append(
            replace(
                example,
                rendered_prompt=rendered,
                prompt_ids=ids,
                span_positions=positions,
                factual_no_question_index=question_index,
            )
        )
    return factual_examples


def build_distractor_examples(tokenizer: Any, train_examples: Sequence[Example], seed: int) -> list[Example]:
    """One distractor per training example: same message and same clamp, an ordinary question, its true answer."""
    order = np.random.default_rng(seed + 1).permutation(len(train_examples))
    distractors = []
    for index, example in enumerate(train_examples):
        if example.first_user_turn is None or example.assistant_message is None:
            raise ValueError("Distractor prompts require the original user and assistant turns")
        question, answer = DISTRACTOR_QUESTIONS[int(order[index]) % len(DISTRACTOR_QUESTIONS)]
        rendered, ids, positions = render_prompt_and_span(
            tokenizer, example.first_user_turn, example.assistant_message, question + FACTUAL_NO_SUFFIX,
            example.frame_text, example.frame_index, example.family,
        )
        distractors.append(replace(example, rendered_prompt=rendered, prompt_ids=ids, span_positions=positions,
                                   train_label=int(answer)))
    return distractors


def factual_no_held_rule(shifts: Sequence[float | None]) -> bool:
    """Return true only when all four measured shifts are strictly below 0.10."""
    return len(shifts) == 4 and all(
        shift is not None and math.isfinite(float(shift)) and abs(float(shift)) < 0.10
        for shift in shifts
    )


def parse_units(target: str, identity: dict[str, Any], neuron_count: int) -> list[tuple[int, int]]:
    top = identity["top50"]
    identity_units = {
        (int(name[1:].split(".", 1)[0]), int(name.split(".", 1)[1]))
        for name, *_ in top
    }
    # Keep the order from the ranked top-50 list stable.
    a_units = []
    for name, *_ in top:
        unit = (int(name[1:].split(".", 1)[0]), int(name.split(".", 1)[1]))
        if 6 <= unit[0] <= 12 and unit not in a_units:
            a_units.append(unit)
    b_units = []
    for name, *_ in top:
        unit = (int(name[1:].split(".", 1)[0]), int(name.split(".", 1)[1]))
        if unit[0] == 15 and unit not in b_units:
            b_units.append(unit)
    if len(a_units) != 11:
        raise ValueError(f"Expected 11 target-A units in layers 6-12, found {len(a_units)}")
    if len(b_units) != 31:
        raise ValueError(f"Expected 31 target-B units in layer 15, found {len(b_units)}")
    if target == "A" or target == "shuffled":
        units = a_units
    elif target == "B":
        units = b_units
    elif target == "random":
        rng = np.random.default_rng(0)
        counts = Counter(layer for layer, _ in a_units)
        units = []
        for layer, count in sorted(counts.items()):
            excluded = {n for l, n in identity_units if l == layer}
            pool = np.asarray([n for n in range(neuron_count) if n not in excluded], dtype=np.int64)
            if len(pool) < count:
                raise ValueError(f"Not enough non-identity neurons in layer {layer}")
            units.extend((layer, int(n)) for n in rng.choice(pool, size=count, replace=False))
    else:
        raise ValueError(f"Unknown target: {target}")
    if any(l < 0 or l >= 16 or n < 0 or n >= neuron_count for l, n in units):
        raise ValueError("Identity neuron index is outside the loaded model")
    return units


class NeuronHooks:
    """Capture per-token neuron values and/or apply per-example clamps."""

    def __init__(self, model: nn.Module, units: Sequence[tuple[int, int]]):
        self.units = list(units)
        self.layers: dict[int, tuple[list[int], list[int]]] = {}
        for global_index, (layer, neuron) in enumerate(self.units):
            if layer not in self.layers:
                self.layers[layer] = ([], [])
            self.layers[layer][0].append(global_index)
            self.layers[layer][1].append(neuron)
        self.state_spans: list[list[int]] = []
        self.state_clamps: list[np.ndarray | None] = []
        self.captures: list[torch.Tensor | None] = []
        self.capture_enabled = False
        self.handles = []
        for layer, (global_ids, neuron_ids) in self.layers.items():
            module = model.model.layers[layer].mlp.down_proj
            self.handles.append(
                module.register_forward_pre_hook(self._make_hook(global_ids, neuron_ids))
            )

    def _make_hook(self, global_ids: list[int], neuron_ids: list[int]):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]):
            x = inputs[0]
            if self.capture_enabled:
                for row, positions in enumerate(self.state_spans):
                    if not positions:
                        continue
                    pos = torch.as_tensor(positions, dtype=torch.long, device=x.device)
                    selected = x[row].index_select(0, pos).index_select(
                        1, torch.as_tensor(neuron_ids, dtype=torch.long, device=x.device)
                    )
                    if self.captures[row] is None:
                        self.captures[row] = torch.empty(
                            (len(positions), len(self.units)), dtype=selected.dtype, device="cpu"
                        )
                    self.captures[row][:, global_ids] = selected.detach().to("cpu")
            if any(value is not None for value in self.state_clamps):
                values = [
                    None if clamp is None else clamp[global_ids]
                    for clamp in self.state_clamps
                ]
                x = clamp_tensor(x, self.state_spans, neuron_ids, values)
                return (x,) + tuple(inputs[1:])
            return None

        return hook

    def prepare(self, spans: Sequence[Sequence[int]], clamps: Sequence[np.ndarray | None] | None = None, capture: bool = False):
        self.state_spans = [list(x) for x in spans]
        self.state_clamps = list(clamps if clamps is not None else [None] * len(spans))
        if len(self.state_clamps) != len(self.state_spans):
            raise ValueError("Clamp and span batch sizes differ")
        self.capture_enabled = capture
        self.captures = [None] * len(spans) if capture else []

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def padded_batch(
    examples: Sequence[Example],
    pad_id: int,
    answer_ids: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[list[int]]]:
    rows = []
    answer_positions = []
    for i, example in enumerate(examples):
        row = list(example.prompt_ids)
        if answer_ids is not None:
            answer_positions.append(len(row))
            row.append(int(answer_ids[i]))
        rows.append(row)
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, : len(row)] = torch.as_tensor(row, dtype=torch.long)
        mask[i, : len(row)] = 1
    spans = [example.span_positions for example in examples]
    return ids.to(DEVICE), mask.to(DEVICE), answer_positions, spans


def chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def capture_natural(
    model: nn.Module,
    examples: list[Example],
    hooks: NeuronHooks,
    pad_id: int,
    batch_size: int = 4,
) -> None:
    model.eval()
    with torch.inference_mode():
        for batch in chunks(examples, batch_size):
            input_ids, attention_mask, _, spans = padded_batch(batch, pad_id)
            hooks.prepare(spans, capture=True)
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            for row, example in enumerate(batch):
                captured = hooks.captures[row]
                if captured is None:
                    raise RuntimeError("A target span was not captured")
                example.natural_tokens = captured.float().numpy()
                example.natural_units = example.natural_tokens.mean(axis=0)
    hooks.prepare([], capture=False)


def make_statistics(examples: Sequence[Example], units: int) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.stack([example.natural_units for example in examples]).astype(np.float64)
    if values.shape[1] != units:
        raise ValueError(f"Expected {units} target units, found {values.shape[1]}")
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds < 1e-6] = 1e-6
    scores = np.asarray([probe_score(x, means, stds) for x in values], dtype=np.float64)
    return means, stds, float(np.median(scores))


def scale_clamp_values(high: np.ndarray, low: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Widen the high-low gap by `scale` around its midpoint; scale 1 leaves both unchanged (as clampreach.py, 12d)."""
    if scale == 1.0:
        return high, low
    mid = (high + low) / 2.0
    return (mid + scale * (high - mid)).astype(np.float32), (mid - scale * (mid - low)).astype(np.float32)


def activation_quantiles(
    training_examples: Sequence[Example],
    n_units: int,
    high_q: float,
    low_q: float,
) -> tuple[np.ndarray, np.ndarray]:
    assistant_self = [e.natural_tokens for e in training_examples if e.family == "A"]
    person = [e.natural_tokens for e in training_examples if e.family == "C"]
    if not assistant_self or not person:
        raise ValueError("Training split needs assistant-self and person frames for steering")
    high_values = np.concatenate(assistant_self, axis=0)
    low_values = np.concatenate(person, axis=0)
    if high_values.shape[1] != n_units or low_values.shape[1] != n_units:
        raise ValueError("Activation width does not match target neuron count")
    return (
        np.quantile(high_values, high_q, axis=0).astype(np.float32),
        np.quantile(low_values, low_q, axis=0).astype(np.float32),
    )


def choose_flipping_clamp(
    example: Example,
    high: np.ndarray,
    low: np.ndarray,
    train_means: np.ndarray,
    train_stds: np.ndarray,
    threshold: float,
    strength: str,
) -> tuple[np.ndarray, int] | None:
    natural = int(example.natural_label)
    desired = 1 - natural
    candidates = (high, low) if desired == 1 else (low, high)
    for candidate in candidates:
        label = probe_label(probe_score(candidate, train_means, train_stds), threshold)
        if label != natural:
            return candidate.astype(np.float32, copy=True), label
    return None


def compute_predictions(
    model: nn.Module,
    tokenizer: Any,
    examples: Sequence[Example],
    hooks: NeuronHooks,
    pad_id: int,
    yes_id: int,
    no_id: int,
    clamp_field: str | None = None,
    batch_size: int = 4,
) -> list[int]:
    predictions: list[int] = []
    model.eval()
    with torch.inference_mode():
        for batch in chunks(examples, batch_size):
            input_ids, attention_mask, _, spans = padded_batch(batch, pad_id)
            clamps = [None if clamp_field is None else getattr(e, clamp_field) for e in batch]
            hooks.prepare(spans, clamps=clamps, capture=False)
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            for row, example in enumerate(batch):
                logits = out.logits[row, len(example.prompt_ids) - 1]
                predictions.append(int(logits[yes_id] > logits[no_id]))
    return predictions


def score_slice(examples: Sequence[Example], predictions: Sequence[int], field: str, surface_conflict: bool = False) -> dict[str, Any]:
    selected = []
    for example, prediction in zip(examples, predictions):
        label = int(getattr(example, field))
        if surface_conflict and label == example.surface_label:
            continue
        selected.append((example, prediction, label))
    correct = sum(prediction == label for _, prediction, label in selected)
    family_counts: dict[str, dict[str, int]] = {}
    for family in "ABCD":
        group = [item for item in selected if item[0].family == family]
        family_counts[family] = {
            "n": len(group),
            "correct": sum(prediction == label for _, prediction, label in group),
        }
    return {
        "n": len(selected),
        "correct": correct,
        "accuracy": None if not selected else correct / len(selected),
        "against_surface_n": len(selected) if surface_conflict else None,
        "by_family": family_counts,
        "label_counts": dict(Counter("yes" if label else "no" for _, _, label in selected)),
    }


def factual_no_direction_metrics(
    examples: Sequence[Example],
    unclamped_predictions: Sequence[int],
    clamped_predictions: Sequence[int],
    label_field: str,
    clamp_label: int,
    questions: Sequence[str] = FACTUAL_NO_QUESTIONS,
) -> dict[str, Any]:
    selected = [
        index
        for index, example in enumerate(examples)
        if int(getattr(example, label_field)) == clamp_label
    ]
    n = len(selected)
    unclamped_yes = sum(int(unclamped_predictions[index]) for index in selected)
    clamped_yes = sum(int(clamped_predictions[index]) for index in selected)
    unclamped_rate = None if n == 0 else unclamped_yes / n
    clamped_rate = None if n == 0 else clamped_yes / n
    return {
        "n": n,
        "yes_rate_unclamped": unclamped_rate,
        "yes_rate_clamped": clamped_rate,
        "shift": None if n == 0 else clamped_rate - unclamped_rate,
        "per_question_yes_counts": {
            question: {
                "unclamped": sum(
                    int(unclamped_predictions[index])
                    for index in selected
                    if examples[index].factual_no_question_index == question_index
                ),
                "clamped": sum(
                    int(clamped_predictions[index])
                    for index in selected
                    if examples[index].factual_no_question_index == question_index
                ),
            }
            for question_index, question in enumerate(questions)
        },
    }


def evaluate_factual_no(
    model: nn.Module,
    tokenizer: Any,
    eval_examples: Sequence[Example],
    hooks: NeuronHooks,
    pad_id: int,
    yes_id: int,
    no_id: int,
    batch_size: int,
    questions: Sequence[str] = FACTUAL_NO_QUESTIONS,
) -> dict[str, Any]:
    """Measure the factual prompts both naturally and under each own clamp."""
    results: dict[str, Any] = {"questions": list(questions)}
    all_shifts: list[float | None] = []
    for strength, clamp_field, label_field in (
        ("90_10", "eval_clamp", "eval_label"),
        ("95_5", "unseen_clamp", "unseen_label"),
    ):
        factual_examples = build_factual_no_examples(tokenizer, eval_examples, clamp_field, questions)
        unclamped_predictions = compute_predictions(
            model, tokenizer, factual_examples, hooks, pad_id, yes_id, no_id,
            batch_size=batch_size,
        )
        clamped_predictions = compute_predictions(
            model, tokenizer, factual_examples, hooks, pad_id, yes_id, no_id,
            clamp_field=clamp_field,
            batch_size=batch_size,
        )
        strength_results = {}
        for direction, clamp_label in (("yes_clamps", 1), ("no_clamps", 0)):
            metrics = factual_no_direction_metrics(
                factual_examples,
                unclamped_predictions,
                clamped_predictions,
                label_field,
                clamp_label,
                questions,
            )
            strength_results[direction] = metrics
            all_shifts.append(metrics["shift"])
        results[strength] = strength_results
    results["held_rule"] = factual_no_held_rule(all_shifts)
    return results


def train_model(
    model: nn.Module,
    tokenizer: Any,
    train_examples: list[Example],
    hooks: NeuronHooks,
    pad_id: int,
    yes_id: int,
    no_id: int,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    seed: int = 42,
) -> dict[str, Any]:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params or any("lora_" not in name for name, parameter in model.named_parameters() if parameter.requires_grad):
        raise RuntimeError("Only LoRA parameters may be trainable")
    optimizer = torch.optim.AdamW(params, lr=learning_rate)
    rng = np.random.default_rng(seed)
    step_times: list[float] = []
    losses: list[float] = []
    steps = 0
    for epoch in range(epochs):
        order = rng.permutation(len(train_examples)).tolist()
        for batch_indices in chunks(order, batch_size):
            batch = [train_examples[i] for i in batch_indices]
            targets = [example.train_label for example in batch]
            target_ids = [yes_id if label else no_id for label in targets]
            input_ids, attention_mask, answer_positions, spans = padded_batch(batch, pad_id, target_ids)
            hooks.prepare(spans, clamps=[example.train_clamp for example in batch], capture=False)
            start = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = torch.stack(
                [out.logits[row, answer_positions[row] - 1] for row in range(len(batch))]
            )
            target_tensor = torch.as_tensor(target_ids, dtype=torch.long, device=DEVICE)
            loss = F.cross_entropy(logits, target_tensor)
            loss.backward()
            optimizer.step()
            step_times.append(time.perf_counter() - start)
            losses.append(float(loss.detach()))
            steps += 1
    hooks.prepare([], capture=False)
    model.eval()
    return {
        "epochs": epochs,
        "examples": len(train_examples),
        "steered_examples": sum(example.train_clamp is not None for example in train_examples),
        "steps": steps,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "mean_step_seconds": float(np.mean(step_times)) if step_times else None,
        "median_step_seconds": float(np.median(step_times)) if step_times else None,
    }


def select_target_units(target: str, identity_path: str, neuron_count: int) -> list[tuple[int, int]]:
    identity = json.load(open(identity_path, encoding="utf-8"))
    return parse_units(target, identity, neuron_count)


def save_lora_parameters(
    model: nn.Module,
    path: str | os.PathLike[str],
    target: str,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> None:
    parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.rsplit(".", 1)[-1] in ("lora_A", "lora_B")
    }
    if not parameters:
        raise RuntimeError("No LoRA parameters found to save")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "lora_parameters": parameters,
            "target": target,
            "seed": seed,
            "epochs": epochs,
            "lr": learning_rate,
        },
        output,
    )


def load_lora_parameters(
    model: nn.Module,
    path: str | os.PathLike[str],
    target: str = "multi",
) -> dict[str, Any]:
    """Load a saved LoRA adapter after installing the matching adapter modules."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict):
        raise ValueError("LoRA checkpoint must be a mapping with target metadata and parameters")
    saved_target = saved.get("target")
    if saved_target != target:
        raise ValueError(f"LoRA checkpoint target must be {target!r}, got {saved_target!r}")
    parameters = saved.get("lora_parameters")
    if not isinstance(parameters, dict):
        raise ValueError("LoRA checkpoint is missing its lora_parameters mapping")

    model_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.rsplit(".", 1)[-1] in ("lora_A", "lora_B")
    }
    missing = sorted(set(model_parameters) - set(parameters))
    unexpected = sorted(set(parameters) - set(model_parameters))
    if missing or unexpected:
        raise ValueError(
            "LoRA checkpoint parameter names do not match the installed adapter "
            f"(missing: {missing}, unexpected: {unexpected})"
        )
    for name, parameter in model_parameters.items():
        value = parameters[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"LoRA checkpoint parameter {name!r} is not a tensor")
        if tuple(value.shape) != tuple(parameter.shape):
            raise ValueError(
                f"LoRA checkpoint shape mismatch for {name}: saved {tuple(value.shape)}, "
                f"model expects {tuple(parameter.shape)}"
            )

    with torch.no_grad():
        for name, parameter in model_parameters.items():
            parameter.copy_(parameters[name].to(device=parameter.device, dtype=parameter.dtype))
    return {key: value for key, value in saved.items() if key != "lora_parameters"}


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    save_lora_path = getattr(args, "save_lora", None)
    if args.eval_only and save_lora_path:
        raise ValueError("--save-lora requires training and cannot be used with --eval-only")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.threads)
    except RuntimeError:
        pass

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, local_files_only=True
    )
    global DEVICE
    DEVICE = getattr(args, "device", "cpu")
    model.to(DEVICE)
    loaded_at = time.perf_counter()
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
        raise ValueError(f"Expected one-token answers, got token IDs {answer_ids}")
    yes_id, no_id = answer_ids["yes"][0], answer_ids["no"][0]
    if tokenizer.pad_token_id is not None:
        pad_id = int(tokenizer.pad_token_id)
    elif tokenizer.eos_token_id is not None:
        pad_id = int(tokenizer.eos_token_id)
    else:
        pad_id = 0

    spec = json.load(open(args.frames, encoding="utf-8"))
    extra_frames = Path(args.frames).with_name("frames-extra.json")
    if extra_frames.exists():
        extra = json.loads(extra_frames.read_text(encoding="utf-8"))
        spec["frames"].extend(extra["frames"])
        spec["users"].extend(extra["users"])
    examples = build_examples(tokenizer, spec, args.limit)
    train_examples = [example for example in examples if example.frame_index % 2 == 0]
    eval_examples = [example for example in examples if example.frame_index % 2 == 1]
    if not train_examples or not eval_examples:
        raise ValueError("Need at least one even and one odd frame; increase --limit")

    capture_started = time.perf_counter()
    capture_natural(model, examples, hooks, pad_id, batch_size=args.infer_batch_size)
    capture_seconds = time.perf_counter() - capture_started
    train_means, train_stds, threshold = make_statistics(train_examples, len(units))
    for example in examples:
        example.natural_score = probe_score(example.natural_units, train_means, train_stds)
        example.natural_label = probe_label(example.natural_score, threshold)

    high90, low10 = activation_quantiles(train_examples, len(units), 0.90, 0.10)
    high95, low05 = activation_quantiles(train_examples, len(units), 0.95, 0.05)
    # Part 12c: widen both the training (90/10) and the unseen evaluation (95/5) clamps around their midpoints.
    clamp_scale = float(getattr(args, "clamp_scale", 1.0))
    high90, low10 = scale_clamp_values(high90, low10, clamp_scale)
    high95, low05 = scale_clamp_values(high95, low05, clamp_scale)
    for example in examples:
        standard = choose_flipping_clamp(example, high90, low10, train_means, train_stds, threshold, "90/10")
        if standard:
            example.eval_clamp, example.eval_label = standard
        unseen = choose_flipping_clamp(example, high95, low05, train_means, train_stds, threshold, "95/5")
        if unseen:
            example.unseen_clamp, example.unseen_label = unseen

    rng = np.random.default_rng(args.seed)
    eligible = [e for e in train_examples if choose_flipping_clamp(e, high90, low10, train_means, train_stds, threshold, "90/10")]
    requested_steered = int(round(len(train_examples) * 0.5))
    chosen_ids = set()
    if eligible:
        selected = rng.permutation(len(eligible))[: min(requested_steered, len(eligible))]
        chosen_ids = {id(eligible[index]) for index in selected}
    for example in train_examples:
        if id(example) in chosen_ids:
            clamp, label = choose_flipping_clamp(example, high90, low10, train_means, train_stds, threshold, "90/10")
            example.train_clamp, example.train_label = clamp, label
        else:
            example.train_clamp = None
            example.train_label = example.natural_label

    if args.target == "shuffled":
        labels = [example.train_label for example in train_examples]
        shuffled = rng.permutation(labels).tolist()
        for example, label in zip(train_examples, shuffled):
            example.train_label = int(label)

    training_set = list(train_examples)
    if getattr(args, "distractors", False) and not args.eval_only:
        training_set += build_distractor_examples(tokenizer, train_examples, args.seed)

    if args.eval_only:
        training_result = {"epochs": 0, "examples": len(train_examples), "steered_examples": 0, "steps": 0}
    else:
        install_lora(model, rank=8)
        model.to(DEVICE)
        training_result = train_model(
            model, tokenizer, training_set, hooks, pad_id, yes_id, no_id,
            epochs=args.epochs, batch_size=8, learning_rate=args.lr, seed=args.seed,
        )
        if save_lora_path:
            save_lora_parameters(
                model,
                save_lora_path,
                target=args.target,
                seed=args.seed,
                epochs=args.epochs,
                learning_rate=args.lr,
            )

    natural_predictions = compute_predictions(model, tokenizer, eval_examples, hooks, pad_id, yes_id, no_id, batch_size=args.infer_batch_size)
    standard_examples = [e for e in eval_examples if e.eval_clamp is not None]
    standard_predictions = compute_predictions(
        model, tokenizer, standard_examples, hooks, pad_id, yes_id, no_id,
        clamp_field="eval_clamp", batch_size=args.infer_batch_size,
    )
    unseen_examples = [e for e in eval_examples if e.unseen_clamp is not None]
    unseen_predictions = compute_predictions(
        model, tokenizer, unseen_examples, hooks, pad_id, yes_id, no_id,
        clamp_field="unseen_clamp", batch_size=args.infer_batch_size,
    )
    evaluation = {
        "natural": score_slice(eval_examples, natural_predictions, "natural_label"),
        "steered_against_surface_90_10": {
            "eligible_flips": len(standard_examples),
            "surface_conflicts_available": sum(e.eval_label != e.surface_label for e in standard_examples),
            **score_slice(standard_examples, standard_predictions, "eval_label", surface_conflict=True),
        },
        "steered_against_surface_unseen_95_5": {
            "eligible_flips": len(unseen_examples),
            "surface_conflicts_available": sum(e.unseen_label != e.surface_label for e in unseen_examples),
            **score_slice(unseen_examples, unseen_predictions, "unseen_label", surface_conflict=True),
        },
    }
    if getattr(args, "factual_no", False):
        evaluation["factual_no"] = evaluate_factual_no(
            model,
            tokenizer,
            eval_examples,
            hooks,
            pad_id,
            yes_id,
            no_id,
            batch_size=args.infer_batch_size,
        )
    if getattr(args, "factual_yes", False):
        evaluation["factual_yes"] = evaluate_factual_no(
            model, tokenizer, eval_examples, hooks, pad_id, yes_id, no_id,
            batch_size=args.infer_batch_size, questions=FACTUAL_YES_QUESTIONS,
        )
    perf = {
        "model_load_seconds": loaded_at - started,
        "natural_capture_seconds": capture_seconds,
        "post_capture_training_and_evaluation_seconds": time.perf_counter() - capture_started - capture_seconds,
        "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)) if resource else None,
    }
    result = {
        "model": args.model,
        "target": args.target,
        "eval_only": bool(args.eval_only),
        "seed": args.seed,
        "frame_rows": len({e.frame_index for e in examples}),
        "example_count": len(examples),
        "split": {"train": len(train_examples), "eval": len(eval_examples), "unit": "A/B/C/D examples; frames split by even/odd index"},
        "units": [f"L{layer}.{neuron}" for layer, neuron in units],
        "calibration": {
            "z_score_means": train_means.tolist(),
            "z_score_stds": train_stds.tolist(),
            "natural_median_threshold": threshold,
            "natural_train_yes": sum(e.natural_label for e in train_examples),
            "natural_train_no": sum(1 - e.natural_label for e in train_examples),
            "steered_candidate_train": len(eligible),
            "requested_half_train_steered": requested_steered,
            "actual_train_steered": training_result.get("steered_examples", 0),
            **({"clamp_scale": clamp_scale} if clamp_scale != 1.0 else {}),
            "90_10_high": high90.tolist(),
            "90_10_low": low10.tolist(),
            "95_5_high": high95.tolist(),
            "95_5_low": low05.tolist(),
        },
        "training": training_result,
        "evaluation": evaluation,
        "performance": perf,
        "rendered_example": {
            "family": examples[0].family,
            "frame_text": examples[0].frame_text,
            "measured_token_positions": examples[0].span_positions,
            "rendered_prompt": examples[0].rendered_prompt,
        },
    }
    if getattr(args, "distractors", False):
        result["distractors"] = len(DISTRACTOR_QUESTIONS)
    hooks.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--frames", default=FRAME_PATH)
    parser.add_argument("--identity", default=IDENTITY_PATH)
    parser.add_argument("--target", choices=("A", "B", "random", "shuffled"), default="A")
    parser.add_argument("--eval-only", action="store_true", help="score the base model without LoRA training")
    parser.add_argument("--limit", type=int, default=0, help="use the first N frame rows (0 means all 72)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--infer-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--factual-no", action="store_true", help="measure the factual-no control after evaluation")
    parser.add_argument("--factual-yes", action="store_true", help="measure the held-out known-yes control after evaluation")
    parser.add_argument("--distractors", action="store_true", help="Part 12f: add one ordinary yes/no question per training example")
    parser.add_argument("--device", default="cpu", help="cpu, or xpu for the Arc A770 (Windows GPU job)")
    parser.add_argument("--save-lora", default=None, metavar="PATH", help="save trained LoRA parameters to PATH")
    parser.add_argument("--clamp-scale", type=float, default=1.0,
                        help="Part 12c: widen every clamp's high-low gap by this factor around its midpoint (1 = unchanged)")
    args = parser.parse_args()
    if args.limit < 0 or args.threads < 1 or args.threads > 4:
        parser.error("--limit must be nonnegative and --threads must be between 1 and 4")
    result = build_result(args)
    output = Path(args.output or f"part12-{args.target.lower()}{'-eval' if args.eval_only else ''}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "target": result["target"],
        "training": result["training"],
        "evaluation": result["evaluation"],
        "performance": result["performance"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
