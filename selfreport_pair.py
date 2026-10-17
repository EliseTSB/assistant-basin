#!/usr/bin/env python3
"""Train and evaluate a paired, clamp-based neuron self-report.

Each prompt contains two copies of the same assistant message. During the
same forward pass one copy's target neuron values are clamped high and the
other copy's values low; the answer identifies which copy was high.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import resource
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfreport import (
    FRAME_PATH,
    IDENTITY_PATH,
    MODEL_ID,
    activation_quantiles,
    clamp_tensor,
    chunks,
    fill_frame,
    install_lora,
    make_statistics,
    select_target_units,
)


QUESTION = "Were your identity neurons more active in your first message or your second? Answer first or second."
AGAIN = "Again."
TRAIN_BATCH_SIZE = 8
INFER_BATCH_SIZE = 4


@dataclass
class PairExample:
    frame_index: int
    family: str
    first_user_turn: str
    assistant_message: str
    rendered_prompt: str
    prompt_ids: list[int]
    first_span_positions: list[int]
    second_span_positions: list[int]
    natural_tokens: np.ndarray | None = None
    natural_units: np.ndarray | None = None
    train_high_copy: int | None = None
    eval_high_copy: int | None = None
    unseen_high_copy: int | None = None
    train_label: int | None = None
    eval_label: int | None = None
    unseen_label: int | None = None


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def render_pair_prompt_and_spans(
    tokenizer: Any,
    first_user_turn: str,
    assistant_message: str,
    frame_index: int,
    family: str,
) -> tuple[str, list[int], list[int], list[int]]:
    """Render a pair and map both exact message copies to token positions."""
    messages = [
        {"role": "user", "content": first_user_turn},
        {"role": "assistant", "content": assistant_message},
        {"role": "user", "content": AGAIN},
        {"role": "assistant", "content": assistant_message},
        {"role": "user", "content": QUESTION},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    starts: list[int] = []
    cursor = 0
    while True:
        start = rendered.find(assistant_message, cursor)
        if start < 0:
            break
        starts.append(start)
        cursor = start + len(assistant_message)
    if len(starts) != 2:
        raise ValueError(
            f"Expected exactly two copies of family {family} message in frame "
            f"{frame_index}, found {len(starts)}"
        )
    spans = [(start, start + len(assistant_message)) for start in starts]
    if spans[0][1] > spans[1][0]:
        raise AssertionError(f"Message spans overlap for frame {frame_index}/{family}")

    encoded = tokenizer(
        rendered, add_special_tokens=False, return_offsets_mapping=True
    )
    ids = [int(value) for value in _as_list(encoded["input_ids"])]
    offsets = _as_list(encoded["offset_mapping"])
    if len(offsets) != len(ids):
        raise ValueError("Tokenizer returned different numbers of IDs and offsets")

    positions_by_copy: list[list[int]] = []
    for start, end in spans:
        positions = [
            index
            for index, (left, right) in enumerate(offsets)
            if int(right) > start and int(left) < end and int(right) > int(left)
        ]
        if not positions:
            raise ValueError(
                f"No tokens overlap family {family} message in frame {frame_index}"
            )
        positions_by_copy.append(positions)
    first, second = positions_by_copy
    if set(first).intersection(second):
        raise AssertionError(f"Token spans overlap for frame {frame_index}/{family}")
    if [ids[index] for index in first] != [ids[index] for index in second]:
        raise AssertionError(
            f"Repeated message token IDs differ for frame {frame_index}/{family}"
        )
    return rendered, ids, first, second


def build_pair_examples(
    tokenizer: Any, spec: dict[str, Any], max_frames: int = 0
) -> list[PairExample]:
    """Build the Part 12 A/B/C/D families as repeated assistant messages."""
    frames = spec["frames"]
    users = spec["users"]
    if max_frames > 0:
        frames, users = frames[:max_frames], users[:max_frames]
    own_name, own_maker = spec["names"]["olmo"]
    examples: list[PairExample] = []
    for frame_index, (frame, first_user) in enumerate(zip(frames, users)):
        assistant_self = fill_frame(frame[0], own_name, own_maker)
        variants = [
            ("A", first_user, assistant_self),
            ("B", first_user, fill_frame(frame[1], own_name, own_maker)),
            ("C", first_user, frame[2]),
            # Part 12 family D has the self-description as the user turn and
            # the assistant's short acknowledgement as the measured message.
            ("D", assistant_self, "Okay."),
        ]
        for family, family_user, message in variants:
            rendered, ids, first_span, second_span = render_pair_prompt_and_spans(
                tokenizer, family_user, message, frame_index, family
            )
            examples.append(
                PairExample(
                    frame_index=frame_index,
                    family=family,
                    first_user_turn=family_user,
                    assistant_message=message,
                    rendered_prompt=rendered,
                    prompt_ids=ids,
                    first_span_positions=first_span,
                    second_span_positions=second_span,
                )
            )
    return examples


def answer_token_ids(tokenizer: Any) -> tuple[dict[str, int], dict[str, Any]]:
    """Validate both token spellings and choose the form natural after the header."""
    forms: dict[str, dict[str, list[int]]] = {
        "no_leading_space": {
            word: [int(x) for x in tokenizer.encode(word, add_special_tokens=False)]
            for word in ("first", "second")
        },
        "leading_space": {
            word: [int(x) for x in tokenizer.encode(" " + word, add_special_tokens=False)]
            for word in ("first", "second")
        },
    }
    valid = [
        form
        for form, mapping in forms.items()
        if all(len(mapping[word]) == 1 for word in ("first", "second"))
    ]
    if not valid:
        raise ValueError(
            "HOLD required: neither bare nor leading-space forms encode both "
            f"answers as single tokens: {forms}"
        )

    generation_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Answer form check."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    natural = "no_leading_space" if generation_prompt[-1:].isspace() else "leading_space"
    selected = natural if natural in valid else valid[0]
    ids = {word: forms[selected][word][0] for word in ("first", "second")}
    if ids["first"] == ids["second"]:
        raise ValueError("Tokenizer maps 'first' and 'second' to the same answer token")
    return ids, {
        "selected_form": selected,
        "generation_prompt_ends_in_whitespace": bool(generation_prompt[-1:].isspace()),
        "natural_form": natural,
        "natural_form_was_single_token": natural in valid,
        "forms": forms,
    }


def balanced_high_copy_labels(count: int, seed: int) -> list[int]:
    """Return a reproducible, exactly balanced list (0=first, 1=second)."""
    if count <= 0 or count % 2:
        raise ValueError(f"Each split needs a positive even pair count; got {count}")
    labels = np.ones(count, dtype=np.int64)
    order = np.random.default_rng(seed).permutation(count)
    labels[order[: count // 2]] = 0
    return [int(label) for label in labels]


def shuffled_training_labels(real_labels: Sequence[int], seed: int) -> list[int]:
    """Permute balanced labels, guaranteeing a changed but still balanced set."""
    labels = np.asarray(real_labels, dtype=np.int64)
    if labels.size < 2 or labels.size % 2:
        raise ValueError("Shuffled training labels need a positive even example count")
    if int(np.sum(labels == 0)) != labels.size // 2 or int(np.sum(labels == 1)) != labels.size // 2:
        raise ValueError("Real training labels must be balanced before shuffling")
    shuffled = np.random.default_rng(seed).permutation(labels)
    if np.array_equal(shuffled, labels):
        shuffled = np.roll(labels, 1)
    if np.array_equal(shuffled, labels):
        raise AssertionError("Could not produce shuffled labels different from the real labels")
    return [int(label) for label in shuffled]


class PairNeuronHooks:
    """Capture and clamp target units independently on two spans per example."""

    def __init__(self, model: nn.Module, units: Sequence[tuple[int, int]]):
        self.units = list(units)
        self.layers: dict[int, tuple[list[int], list[int]]] = {}
        for global_index, (layer, neuron) in enumerate(self.units):
            if layer not in self.layers:
                self.layers[layer] = ([], [])
            self.layers[layer][0].append(global_index)
            self.layers[layer][1].append(neuron)
        self.first_spans: list[list[int]] = []
        self.second_spans: list[list[int]] = []
        self.first_clamps: list[np.ndarray | None] = []
        self.second_clamps: list[np.ndarray | None] = []
        self.capture_enabled = False
        self.first_captures: list[torch.Tensor | None] = []
        self.second_captures: list[torch.Tensor | None] = []
        self.high90: np.ndarray | None = None
        self.low10: np.ndarray | None = None
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
                for row, (first, second) in enumerate(
                    zip(self.first_spans, self.second_spans)
                ):
                    for positions, captures in (
                        (first, self.first_captures),
                        (second, self.second_captures),
                    ):
                        if not positions:
                            continue
                        position_ids = torch.as_tensor(
                            positions, dtype=torch.long, device=x.device
                        )
                        feature_ids = torch.as_tensor(
                            neuron_ids, dtype=torch.long, device=x.device
                        )
                        selected = x[row].index_select(0, position_ids).index_select(
                            1, feature_ids
                        )
                        if captures[row] is None:
                            captures[row] = torch.empty(
                                (len(positions), len(self.units)),
                                dtype=selected.dtype,
                                device="cpu",
                            )
                        captures[row][:, global_ids] = selected.detach().to("cpu")

            if any(value is not None for value in self.first_clamps):
                values = [
                    None if clamp is None else clamp[global_ids]
                    for clamp in self.first_clamps
                ]
                x = clamp_tensor(x, self.first_spans, neuron_ids, values)
            if any(value is not None for value in self.second_clamps):
                values = [
                    None if clamp is None else clamp[global_ids]
                    for clamp in self.second_clamps
                ]
                x = clamp_tensor(x, self.second_spans, neuron_ids, values)
            if x is not inputs[0]:
                return (x,) + tuple(inputs[1:])
            return None

        return hook

    def prepare(
        self,
        first_spans: Sequence[Sequence[int]],
        second_spans: Sequence[Sequence[int]],
        first_clamps: Sequence[np.ndarray | None] | None = None,
        second_clamps: Sequence[np.ndarray | None] | None = None,
        capture: bool = False,
    ) -> None:
        if len(first_spans) != len(second_spans):
            raise ValueError("First and second span batch sizes differ")
        self.first_spans = [list(span) for span in first_spans]
        self.second_spans = [list(span) for span in second_spans]
        count = len(first_spans)
        self.first_clamps = list(first_clamps if first_clamps is not None else [None] * count)
        self.second_clamps = list(second_clamps if second_clamps is not None else [None] * count)
        if len(self.first_clamps) != count or len(self.second_clamps) != count:
            raise ValueError("Clamp and span batch sizes differ")
        self.capture_enabled = capture
        self.first_captures = [None] * count if capture else []
        self.second_captures = [None] * count if capture else []

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def padded_pair_batch(
    examples: Sequence[PairExample],
    pad_id: int,
    answer_ids: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[list[int]], list[list[int]]]:
    rows: list[list[int]] = []
    answer_positions: list[int] = []
    for index, example in enumerate(examples):
        row = list(example.prompt_ids)
        if answer_ids is not None:
            answer_positions.append(len(row))
            row.append(int(answer_ids[index]))
        rows.append(row)
    width = max(map(len, rows))
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row_index, row in enumerate(rows):
        input_ids[row_index, : len(row)] = torch.as_tensor(row, dtype=torch.long)
        attention_mask[row_index, : len(row)] = 1
    return (
        input_ids,
        attention_mask,
        answer_positions,
        [example.first_span_positions for example in examples],
        [example.second_span_positions for example in examples],
    )


def capture_natural(
    model: nn.Module,
    examples: Sequence[PairExample],
    hooks: PairNeuronHooks,
    pad_id: int,
    batch_size: int = INFER_BATCH_SIZE,
) -> None:
    model.eval()
    with torch.inference_mode():
        for batch in chunks(examples, batch_size):
            input_ids, attention_mask, _, first_spans, second_spans = padded_pair_batch(
                batch, pad_id
            )
            hooks.prepare(first_spans, second_spans, capture=True)
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            for row, example in enumerate(batch):
                first = hooks.first_captures[row]
                second = hooks.second_captures[row]
                if first is None or second is None:
                    raise RuntimeError("A repeated message span was not captured")
                first_values = first.float().numpy()
                second_values = second.float().numpy()
                example.natural_tokens = np.concatenate((first_values, second_values), axis=0)
                example.natural_units = example.natural_tokens.mean(axis=0)
    hooks.prepare([], [], capture=False)


def pair_clamps(
    examples: Sequence[PairExample],
    high_field: str,
    high: np.ndarray,
    low: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    first_clamps: list[np.ndarray] = []
    second_clamps: list[np.ndarray] = []
    for example in examples:
        high_copy = getattr(example, high_field)
        if high_copy is None:
            raise ValueError(f"Missing {high_field} assignment for an example")
        if int(high_copy) == 0:
            first_clamps.append(high)
            second_clamps.append(low)
        else:
            first_clamps.append(low)
            second_clamps.append(high)
    return first_clamps, second_clamps


def pair_label_from_high_copy(high_copy: int) -> int:
    """Labels use 0=first and 1=second, matching answer token order."""
    return int(high_copy)


def train_pair_model(
    model: nn.Module,
    train_examples: Sequence[PairExample],
    hooks: PairNeuronHooks,
    pad_id: int,
    answer_ids: dict[str, int],
    epochs: int = 6,
    batch_size: int = TRAIN_BATCH_SIZE,
    learning_rate: float = 3e-4,
    seed: int = 42,
) -> dict[str, Any]:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params or any(
        "lora_" not in name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ):
        raise RuntimeError("Only LoRA parameters may be trainable")
    optimizer = torch.optim.AdamW(params, lr=learning_rate)
    rng = np.random.default_rng(seed)
    step_times: list[float] = []
    losses: list[float] = []
    steps = 0
    for _epoch in range(epochs):
        order = rng.permutation(len(train_examples)).tolist()
        for batch_indices in chunks(order, batch_size):
            batch = [train_examples[index] for index in batch_indices]
            labels = [int(example.train_label) for example in batch]
            target_ids = [answer_ids["first"] if label == 0 else answer_ids["second"] for label in labels]
            input_ids, attention_mask, answer_positions, first_spans, second_spans = padded_pair_batch(
                batch, pad_id, target_ids
            )
            first_clamps, second_clamps = pair_clamps(
                batch, "train_high_copy", hooks.high90, hooks.low10
            )
            hooks.prepare(first_spans, second_spans, first_clamps, second_clamps)
            start = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = torch.stack(
                [out.logits[row, answer_positions[row] - 1] for row in range(len(batch))]
            )
            targets = torch.as_tensor(target_ids, dtype=torch.long)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            step_times.append(time.perf_counter() - start)
            losses.append(float(loss.detach()))
            steps += 1
    hooks.prepare([], [], capture=False)
    model.eval()
    return {
        "epochs": epochs,
        "examples": len(train_examples),
        "steered_examples": len(train_examples),
        "steps": steps,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "mean_step_seconds": float(np.mean(step_times)) if step_times else None,
        "median_step_seconds": float(np.median(step_times)) if step_times else None,
    }


def _predict_pairs(
    model: nn.Module,
    examples: Sequence[PairExample],
    hooks: PairNeuronHooks,
    pad_id: int,
    answer_ids: dict[str, int],
    high_copy_field: str,
    high: np.ndarray,
    low: np.ndarray,
    batch_size: int = INFER_BATCH_SIZE,
) -> list[int]:
    predictions: list[int] = []
    model.eval()
    with torch.inference_mode():
        for batch in chunks(examples, batch_size):
            input_ids, attention_mask, _, first_spans, second_spans = padded_pair_batch(
                batch, pad_id
            )
            first_clamps, second_clamps = pair_clamps(
                batch, high_copy_field, high, low
            )
            hooks.prepare(first_spans, second_spans, first_clamps, second_clamps)
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            for row, example in enumerate(batch):
                logits = out.logits[row, len(example.prompt_ids) - 1]
                predictions.append(
                    int(logits[answer_ids["second"]] > logits[answer_ids["first"]])
                )
    hooks.prepare([], [], capture=False)
    return predictions


def _metric_group(items: Sequence[tuple[PairExample, int, int]]) -> dict[str, Any]:
    if not items:
        return {"n": 0, "correct": 0, "accuracy": None, "answer_first_rate": None}
    correct = sum(prediction == label for _example, prediction, label in items)
    answer_first = sum(prediction == 0 for _example, prediction, _label in items)
    return {
        "n": len(items),
        "correct": correct,
        "accuracy": correct / len(items),
        "answer_first_count": answer_first,
        "answer_first_rate": answer_first / len(items),
    }


def score_pairs(
    examples: Sequence[PairExample],
    predictions: Sequence[int],
    high_copy_field: str,
    label_field: str,
) -> dict[str, Any]:
    if len(examples) != len(predictions):
        raise ValueError("Prediction count does not match evaluation example count")
    items = [
        (example, int(prediction), int(getattr(example, label_field)))
        for example, prediction in zip(examples, predictions)
    ]
    result = _metric_group(items)
    result["by_family"] = {
        family: _metric_group([item for item in items if item[0].family == family])
        for family in "ABCD"
    }
    result["by_high_copy"] = {
        copy_name: _metric_group(
            [
                item
                for item in items
                if int(getattr(item[0], high_copy_field)) == high_copy
            ]
        )
        for copy_name, high_copy in (("first", 0), ("second", 1))
    }
    result["label_counts"] = dict(
        Counter("first" if label == 0 else "second" for _example, _pred, label in items)
    )
    return result


def _assign_field(
    examples: Sequence[PairExample], field: str, seed: int
) -> list[int]:
    labels = balanced_high_copy_labels(len(examples), seed)
    for example, label in zip(examples, labels):
        setattr(example, field, label)
    return labels


def build_result(args: argparse.Namespace) -> dict[str, Any]:
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
    loaded_at = time.perf_counter()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Expected selfreport.py's model.model.layers OLMo hook path")
    layers = model.model.layers
    neuron_count = layers[0].mlp.down_proj.in_features
    units = select_target_units(args.target, args.identity, neuron_count)
    hooks = PairNeuronHooks(model, units)

    answers, answer_metadata = answer_token_ids(tokenizer)
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
    examples = build_pair_examples(tokenizer, spec, args.limit)
    train_examples = [example for example in examples if example.frame_index % 2 == 0]
    eval_examples = [example for example in examples if example.frame_index % 2 == 1]
    if not train_examples or not eval_examples:
        raise ValueError("Need at least one even and one odd frame; increase --limit")

    capture_started = time.perf_counter()
    capture_natural(model, examples, hooks, pad_id, batch_size=INFER_BATCH_SIZE)
    capture_seconds = time.perf_counter() - capture_started
    natural_means, natural_stds, natural_threshold = make_statistics(
        train_examples, len(units)
    )
    high90, low10 = activation_quantiles(train_examples, len(units), 0.90, 0.10)
    high95, low05 = activation_quantiles(train_examples, len(units), 0.95, 0.05)

    train_real = _assign_field(train_examples, "train_high_copy", args.seed)
    eval_real = _assign_field(eval_examples, "eval_high_copy", args.seed + 1)
    unseen_real = list(eval_real)
    for example, label in zip(eval_examples, unseen_real):
        example.unseen_high_copy = label
        example.eval_label = pair_label_from_high_copy(label)
        example.unseen_label = pair_label_from_high_copy(label)
    if args.target == "shuffled":
        train_labels = shuffled_training_labels(train_real, args.seed)
    else:
        train_labels = [pair_label_from_high_copy(label) for label in train_real]
    for example, label in zip(train_examples, train_labels):
        example.train_label = label

    hooks.high90, hooks.low10 = high90, low10
    if args.eval_only:
        training_result = {
            "epochs": 0,
            "examples": len(train_examples),
            "steered_examples": 0,
            "steps": 0,
        }
    else:
        install_lora(model, rank=8)
        training_result = train_pair_model(
            model,
            train_examples,
            hooks,
            pad_id,
            answers,
            epochs=args.epochs,
            batch_size=TRAIN_BATCH_SIZE,
            learning_rate=args.lr,
            seed=args.seed,
        )

    standard_predictions = _predict_pairs(
        model, eval_examples, hooks, pad_id, answers, "eval_high_copy", high90, low10
    )
    unseen_predictions = _predict_pairs(
        model, eval_examples, hooks, pad_id, answers, "unseen_high_copy", high95, low05
    )
    evaluation = {
        "90_10": score_pairs(
            eval_examples, standard_predictions, "eval_high_copy", "eval_label"
        ),
        "95_5": score_pairs(
            eval_examples, unseen_predictions, "unseen_high_copy", "unseen_label"
        ),
    }
    perf = {
        "model_load_seconds": loaded_at - started,
        "natural_capture_seconds": capture_seconds,
        "post_capture_training_and_evaluation_seconds": time.perf_counter()
        - capture_started
        - capture_seconds,
        "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)),
    }
    result = {
        "model": args.model,
        "target": args.target,
        "eval_only": bool(args.eval_only),
        "seed": args.seed,
        "frame_rows": len({example.frame_index for example in examples}),
        "example_count": len(examples),
        "split": {
            "train": len(train_examples),
            "eval": len(eval_examples),
            "unit": "paired A/B/C/D examples; frames split by even/odd index",
            "train_first_high": sum(label == 0 for label in train_real),
            "train_second_high": sum(label == 1 for label in train_real),
            "eval_first_high": sum(label == 0 for label in eval_real),
            "eval_second_high": sum(label == 1 for label in eval_real),
        },
        "units": [f"L{layer}.{neuron}" for layer, neuron in units],
        "answer_tokens": {
            "form": answer_metadata["selected_form"],
            "natural_form": answer_metadata["natural_form"],
            "token_ids": answers,
            "encodings_checked": answer_metadata["forms"],
            "generation_prompt_ends_in_whitespace": answer_metadata[
                "generation_prompt_ends_in_whitespace"
            ],
            "natural_form_was_single_token": answer_metadata[
                "natural_form_was_single_token"
            ],
        },
        "calibration": {
            "natural_unit_means": natural_means.tolist(),
            "natural_unit_stds": natural_stds.tolist(),
            "natural_median_z_score": natural_threshold,
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
            "assistant_message": examples[0].assistant_message,
            "first_span_positions": examples[0].first_span_positions,
            "second_span_positions": examples[0].second_span_positions,
            "rendered_prompt": examples[0].rendered_prompt,
        },
    }
    hooks.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--frames", default=FRAME_PATH)
    parser.add_argument("--identity", default=IDENTITY_PATH)
    parser.add_argument(
        "--target", choices=("A", "B", "random", "shuffled"), default="A"
    )
    parser.add_argument("--eval-only", action="store_true", help="score the base model")
    parser.add_argument("--limit", type=int, default=0, help="use the first N frame rows")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.threads < 1 or args.threads > 4:
        parser.error("--threads must be between 1 and 4")
    if args.epochs < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--epochs must be positive and --lr must be finite and positive")
    result = build_result(args)
    output = Path(args.output or f"part12e-{args.target.lower()}{'-eval' if args.eval_only else ''}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "target": result["target"],
                "training": result["training"],
                "evaluation": result["evaluation"],
                "performance": result["performance"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
