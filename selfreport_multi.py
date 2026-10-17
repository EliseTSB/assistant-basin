#!/usr/bin/env python3
"""Train and evaluate a multiple-choice report of which neuron group was clamped."""
from __future__ import annotations

import argparse
import json
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


class MultiNeuronHooks:
    """Clamp only the selected group's features at each row's message positions."""
    def __init__(self, model: torch.nn.Module, groups: dict[str, list[tuple[int, int]]], highs: dict[str, np.ndarray]):
        self.model = model
        self.groups = groups
        self.highs = highs
        self.layer_groups: dict[int, dict[str, list[tuple[int, int, float]]]] = {}
        self.conditions: list[str] = []
        self.spans: list[list[int]] = []
        self.handles = []
        for group, units in groups.items():
            if group == "G1" or group in ("G2", "G3", "G4", "G5"):
                for (layer, neuron), value in zip(units, highs[group]):
                    self.layer_groups.setdefault(layer, {}).setdefault(group, []).append((neuron, 0, float(value)))
        for layer, by_group in self.layer_groups.items():
            module = model.model.layers[layer].mlp.down_proj
            self.handles.append(module.register_forward_pre_hook(self._hook(layer, by_group)))

    def _hook(self, layer: int, by_group: dict[str, list[tuple[int, int, float]]]):
        def apply(_module, inputs):
            x = inputs[0]
            if not self.conditions:
                return None
            y = x.clone()
            for row, (condition, positions) in enumerate(zip(self.conditions, self.spans)):
                if condition in ("none",):
                    continue
                # the decoy condition pushes G5, the group never named in the question
                for neuron, _, value in by_group.get("G5" if condition == "other" else condition, []):
                    if positions:
                        p = torch.as_tensor(positions, dtype=torch.long, device=x.device)
                        y[row, p, neuron] = value
            return (y,) + tuple(inputs[1:])
        return apply

    def prepare(self, spans: Sequence[Sequence[int]], conditions: Sequence[str]):
        self.spans = [list(x) for x in spans]
        self.conditions = list(conditions)
        if len(self.spans) != len(self.conditions):
            raise ValueError("One condition is required per prompt")

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
            hooks.prepare([e.span_positions for e in batch], [e.condition for e in batch])
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


def build_result(args):
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
    units = [u for g in groups.values() for u in g]
    answer_ids = first_token_ids(tokenizer)
    print(json.dumps({"answer_token_ids": answer_ids}, flush=True))
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
    natural_tokens = np.concatenate([e.natural_tokens for e in train_sources if e.family == "A"], axis=0)
    highs = {}
    for group, group_units in groups.items():
        offsets = [units.index(u) for u in group_units]
        hi = np.quantile(natural_tokens[:, offsets], 0.90, axis=0).astype(np.float32)
        lo = np.quantile(np.concatenate([e.natural_tokens[:, offsets] for e in train_sources if e.family == "C"], axis=0), 0.10, axis=0).astype(np.float32)
        highs[group] = sr.scale_clamp_values(hi, lo, args.clamp_scale)[0]
    train_conditions = assign_conditions(train_sources, args.seed)
    train_rows = []
    for source, conditions in zip(train_sources, train_conditions):
        for condition in conditions:
            train_rows.append(MultiExample(source, condition, condition_label(condition), *make_prompts(tokenizer, [source])[0]))
    for e in train_rows: e.train_label = e.label
    if args.shuffled:
        shuffle_training_labels(train_rows, args.seed)
    eval_multi = build_evaluation_examples(tokenizer, eval_sources)
    multi_hooks = MultiNeuronHooks(model, groups, highs)
    if args.eval_only:
        training = {"epochs": 0, "examples": len(train_rows), "steps": 0, "shuffled": False}
    else:
        sr.install_lora(model, rank=8)
        model.to(sr.DEVICE)
        training = train_multi(model, train_rows, multi_hooks, int(pad_id), answer_ids,
                                              epochs=args.epochs, batch_size=8, learning_rate=args.lr, seed=args.seed)
        training["shuffled"] = bool(args.shuffled)
        if args.save_lora:
            sr.save_lora_parameters(model, args.save_lora, target="multi", seed=args.seed, epochs=args.epochs, learning_rate=args.lr)
    evaluation = evaluate_multi(model, eval_multi, multi_hooks, int(pad_id), answer_ids, args.infer_batch_size)
    multi_hooks.close()
    return {"model": args.model, "groups": {k: [f"L{l}.{n}" for l, n in v] for k, v in groups.items()},
            "group_seed": args.group_seed, "clamp_scale": args.clamp_scale, "answer_token_ids": answer_ids,
            "training": training, "evaluation": evaluation,
            "split": {"train_frames": len(train_sources), "eval_frames": len(eval_sources),
                      "training_condition_examples": len(train_rows), "evaluation_condition_examples": len(eval_multi)}}


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
    p.add_argument("--device", default="cpu", help="cpu, or xpu for the Arc A770 (Windows GPU job)")
    args = p.parse_args()
    if args.limit < 0 or args.threads < 1 or args.threads > 4: p.error("--limit must be nonnegative and --threads between 1 and 4")
    result = build_result(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": args.output, "training": result["training"], "evaluation": result["evaluation"]}, indent=2), flush=True)

if __name__ == "__main__": main()
