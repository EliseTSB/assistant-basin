#!/usr/bin/env python3
"""Prompt-pair causal neuron search for OLMo 2 1B Instruct.

The patch site is the input to each MLP down_proj, matching collect.py's
L<layer>.<neuron> convention. Attribution ranks individual layer/neuron/token
entries; a token alignment maps the common chat-template prefix and suffix
exactly and linearly aligns the differing system-message spans.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "allenai/OLMo-2-0425-1B-Instruct"
DEFAULT_PAIRS = ROOT / "data" / "prompt-pairs.json"
DEFAULT_IDENTITY = ROOT / "results" / "part2-olmo2-1b" / "identity-instruct-chat.json"
AI_PATTERN = re.compile(
    r"as an ai|i am an ai|i'm an ai|ai assistant|language model|artificial intelligence",
    re.IGNORECASE,
)
NEURON_NAME = re.compile(r"^L(\d+)\.(\d+)$")


@dataclass(frozen=True)
class PatchEntry:
    layer: int
    position: int
    neuron: int
    source_position: int
    attribution: float

    @property
    def name(self) -> str:
        return f"L{self.layer}.{self.neuron}"


def first_differing_position(tokens1: Sequence[int], tokens2: Sequence[int]) -> int | None:
    """Return the first index with two observed, unequal tokens, else None."""
    for index, (one, two) in enumerate(zip(tokens1, tokens2)):
        if int(one) != int(two):
            return index
    # A proper prefix has no pair of tokens to compare at its end. Normal HF
    # generation includes the stopping EOS token, so such cases are skipped.
    return None


def split_questions(questions: Sequence[str], seed: int) -> tuple[list[str], list[str]]:
    if len(questions) != 30:
        raise ValueError(f"expected 30 questions, got {len(questions)}")
    order = list(range(len(questions)))
    random.Random(seed).shuffle(order)
    return ([questions[i] for i in order[:20]], [questions[i] for i in order[20:]])


def align_prompt_positions(ids1: Sequence[int], ids2: Sequence[int]) -> list[int]:
    """Map every run-1 prompt position to a source position in run 2.

    The chat template and user question are shared. Their common leading and
    trailing token spans align exactly. The only unmatched middle is the
    different system content; its positions are aligned proportionally so
    every run-1 prompt position has a source activation.
    """
    if not ids1 or not ids2:
        raise ValueError("prompts must contain at least one token")
    prefix = 0
    while prefix < min(len(ids1), len(ids2)) and int(ids1[prefix]) == int(ids2[prefix]):
        prefix += 1
    suffix = 0
    max_suffix = min(len(ids1) - prefix, len(ids2) - prefix)
    while suffix < max_suffix and int(ids1[len(ids1) - suffix - 1]) == int(ids2[len(ids2) - suffix - 1]):
        suffix += 1
    middle1 = len(ids1) - prefix - suffix
    middle2 = len(ids2) - prefix - suffix

    result = list(range(prefix))
    if middle1:
        if middle2 <= 0:
            fallback = min(max(prefix, 0), len(ids2) - 1)
            result.extend([fallback] * middle1)
        elif middle1 == 1:
            result.append(prefix + (middle2 - 1) // 2)
        else:
            result.extend(
                prefix + round(i * (middle2 - 1) / (middle1 - 1))
                for i in range(middle1)
            )
    if suffix:
        result.extend(range(len(ids2) - suffix, len(ids2)))
    if len(result) != len(ids1):
        raise AssertionError("prompt alignment length mismatch")
    return result


def full_position_alignment(
    prompt_ids1: Sequence[int], prompt_ids2: Sequence[int], reply_prefix_length: int
) -> list[int]:
    mapping = align_prompt_positions(prompt_ids1, prompt_ids2)
    mapping.extend(len(prompt_ids2) + i for i in range(reply_prefix_length))
    return mapping


def parse_neuron_name(name: str) -> tuple[int, int]:
    match = NEURON_NAME.match(name)
    if not match:
        raise ValueError(f"invalid neuron name: {name!r}")
    return int(match.group(1)), int(match.group(2))


def identity_topk(identity: dict, k: int) -> tuple[list[str], dict]:
    """Read analyze.py's top50 rows, ranked by their discovery score (field 2)."""
    rows = identity.get("top50")
    if not isinstance(rows, list):
        raise ValueError("identity JSON is missing its top50 list")
    ranked = sorted(rows, key=lambda row: float(row[1]), reverse=True)
    available = min(k, len(ranked))
    names = [str(row[0]) for row in ranked[:available]]
    return names, {
        "source_field": "top50",
        "ranking_field": "row[1] (discovery identity score; even-index discovery half per analyze.py)",
        "requested_k": k,
        "available_k": available,
        "source_rows_available": len(ranked),
    }


def aggregate_pair_neurons(pair_records: Sequence[dict], top_k: int) -> tuple[Counter, list[dict]]:
    counts: Counter[str] = Counter()
    for record in pair_records:
        if not record.get("switched"):
            continue
        counts.update({str(entry["neuron"]) for entry in record.get("minimal_set", [])})
    ordered = sorted(counts.items(), key=lambda item: (-item[1], parse_neuron_name(item[0])))[:top_k]
    return counts, [{"neuron": name, "pair_count": count} for name, count in ordered]


def normalize_layer_counts(layer_counts: dict[int | str, int]) -> dict[int, int]:
    return {int(layer): int(count) for layer, count in layer_counts.items() if int(count) > 0}


def random_sets_with_layer_counts(
    layer_counts: dict[int | str, int],
    neurons_per_layer: int,
    num_sets: int = 6,
    seed: int = 0,
    exclude: dict[int | str, Iterable[int]] | None = None,
) -> list[dict[int, list[int]]]:
    """Draw deterministic, layer-count-matched sets, excluding target members."""
    counts = normalize_layer_counts(layer_counts)
    excluded = {int(layer): set(map(int, values)) for layer, values in (exclude or {}).items()}
    rng = np.random.default_rng(seed)
    result: list[dict[int, list[int]]] = []
    for _ in range(num_sets):
        selected: dict[int, list[int]] = {}
        for layer in sorted(counts):
            pool = np.asarray(
                [n for n in range(neurons_per_layer) if n not in excluded.get(layer, set())],
                dtype=np.int64,
            )
            count = counts[layer]
            if count > len(pool):
                raise ValueError(
                    f"cannot draw {count} controls for layer {layer}; only {len(pool)} neurons remain"
                )
            selected[layer] = sorted(map(int, rng.choice(pool, size=count, replace=False)))
        result.append(selected)
    return result


def is_ai_reply(text: str) -> bool:
    return bool(AI_PATTERN.search(text))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def greedy_patch(
    ranked_entries: Sequence[PatchEntry],
    evaluate_metric: Callable[[Sequence[PatchEntry]], float],
    *,
    initial_metric: float | None = None,
    chunk_size: int = 4,
    max_entries: int = 256,
) -> dict:
    """Add attribution-ranked entries in chunks; trim the crossing chunk.

    Each call to evaluate_metric represents one batched model forward. The
    returned entries are the shortest ranked prefix at the first crossing.
    If the cap is exhausted without a crossing, `entries` reports attempts
    and `switched` remains false.
    """
    if chunk_size < 1 or max_entries < 0:
        raise ValueError("chunk_size must be positive and max_entries nonnegative")
    ranked = list(ranked_entries[:max_entries])
    metric = float(initial_metric) if initial_metric is not None else float(evaluate_metric([]))
    if metric < 0:
        return {"entries": [], "attempted": [], "metric": metric, "switched": True, "forwards": 0}

    applied: list[PatchEntry] = []
    forwards = 0
    for start in range(0, len(ranked), chunk_size):
        chunk = ranked[start : start + chunk_size]
        trial = applied + chunk
        chunk_metric = float(evaluate_metric(trial))
        forwards += 1
        if chunk_metric < 0:
            prefix = list(applied)
            prefix_metric = metric
            for offset, entry in enumerate(chunk):
                current = prefix + [entry]
                if offset == len(chunk) - 1:
                    current_metric = chunk_metric
                else:
                    current_metric = float(evaluate_metric(current))
                    forwards += 1
                prefix = current
                prefix_metric = current_metric
                if current_metric < 0:
                    return {
                        "entries": prefix,
                        "attempted": prefix,
                        "metric": prefix_metric,
                        "switched": True,
                        "forwards": forwards,
                    }
            # The full chunk already crossed, so the final prefix must have.
            return {
                "entries": prefix,
                "attempted": prefix,
                "metric": prefix_metric,
                "switched": True,
                "forwards": forwards,
            }
        applied = trial
        metric = chunk_metric
    return {
        "entries": applied,
        "attempted": applied,
        "metric": metric,
        "switched": False,
        "forwards": forwards,
    }


def _model_layers(model):
    trunk = getattr(model, "model", None)
    layers = getattr(trunk, "layers", None)
    if layers is None:
        raise ValueError("expected OLMo-style model.model.layers")
    return layers


def _model_logits(model, input_ids: Tensor) -> Tensor:
    output = model(input_ids=input_ids)
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, dict):
        logits = output.get("logits")
    if logits is None:
        raise ValueError("model forward did not return logits")
    return logits


def encode_chat_prompt(tokenizer, system: str, question: str) -> list[int]:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("the selected tokenizer has no chat template")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return list(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def greedy_decode(model, tokenizer, prompt_ids: Sequence[int], device: torch.device, max_new_tokens: int) -> list[int]:
    ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    generated = model.generate(
        input_ids=ids,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_id,
    )
    return list(map(int, generated[0, ids.shape[1] :].detach().cpu().tolist()))


def capture_run2_activations(model, input_tokens: Sequence[int], device: torch.device) -> list[Tensor]:
    layers = _model_layers(model)
    captured: list[Tensor | None] = [None] * len(layers)
    handles = []
    for index, layer in enumerate(layers):
        def hook(_module, inputs, index=index):
            captured[index] = inputs[0].detach()
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
    try:
        ids = torch.tensor([list(input_tokens)], dtype=torch.long, device=device)
        with torch.no_grad():
            _model_logits(model, ids)
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in captured):
        raise RuntimeError("failed to capture every MLP down_proj input")
    return [value for value in captured if value is not None]


def capture_run1_attribution(
    model: torch.nn.Module,
    input_tokens: Sequence[int],
    device: torch.device,
    logit_position: int,
    token1: int,
    token2: int,
) -> tuple[float, list[Tensor], list[Tensor]]:
    layers = _model_layers(model)
    captured: list[Tensor | None] = [None] * len(layers)
    handles = []

    def make_hook(index: int):
        def hook(_module, inputs):
            activation = inputs[0]
            if not activation.requires_grad:
                activation.requires_grad_(True)
            activation.retain_grad()
            captured[index] = activation
        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook(index)))
    try:
        ids = torch.tensor([list(input_tokens)], dtype=torch.long, device=device)
        with torch.enable_grad():
            logits = _model_logits(model, ids)
            metric = logits[0, logit_position, token1].float() - logits[0, logit_position, token2].float()
            metric.backward()
        value = float(metric.detach().cpu())
        if any(item is None or item.grad is None for item in captured):
            raise RuntimeError("missing activation or activation gradient in attribution pass")
        acts = [item.detach()[0].float().cpu() for item in captured if item is not None]
        grads = [item.grad.detach()[0].float().cpu() for item in captured if item is not None]
        return value, acts, grads
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


def rank_attribution_entries(
    run1_acts: Sequence[Tensor],
    run2_acts: Sequence[Tensor],
    gradients: Sequence[Tensor],
    position_map: Sequence[int],
    *,
    limit: int = 256,
) -> list[PatchEntry]:
    if not (len(run1_acts) == len(run2_acts) == len(gradients)):
        raise ValueError("activation and gradient layer counts differ")
    if not run1_acts:
        return []
    seq_len, width = run1_acts[0].shape
    if len(position_map) != seq_len:
        raise ValueError("position map length does not match run-1 sequence")
    all_scores = []
    for layer, (act1, act2, grad) in enumerate(zip(run1_acts, run2_acts, gradients)):
        # capture_run2_activations keeps the batch dimension because the patch
        # evaluator indexes [0, position, neuron]; run 1 arrives unbatched.
        if act2.dim() == 3:
            act2 = act2[0]
        act2 = act2.float().cpu()
        if act1.shape != grad.shape:
            raise ValueError("run-1 activation and gradient shapes differ")
        source_positions = torch.tensor(position_map, dtype=torch.long)
        if int(source_positions.max()) >= act2.shape[0]:
            raise ValueError("position map refers outside run-2 activation sequence")
        aligned2 = act2.index_select(0, source_positions)
        all_scores.append(((aligned2 - act1) * grad).reshape(-1))
    scores = torch.cat(all_scores)
    finite = torch.isfinite(scores)
    candidate_count = min(limit, int(finite.sum().item()))
    if candidate_count == 0:
        return []
    safe_scores = scores.masked_fill(~finite, float("inf"))
    values, indices = torch.topk(safe_scores, candidate_count, largest=False, sorted=True)
    entries: list[PatchEntry] = []
    per_layer = seq_len * width
    for score, flat_index in zip(values.tolist(), indices.tolist()):
        if score >= 0:
            break
        layer = flat_index // per_layer
        local = flat_index % per_layer
        position, neuron = divmod(local, width)
        entries.append(
            PatchEntry(
                layer=layer,
                position=position,
                neuron=neuron,
                source_position=int(position_map[position]),
                attribution=float(score),
            )
        )
    return entries


def make_patch_evaluator(
    model,
    input_tokens: Sequence[int],
    device: torch.device,
    logit_position: int,
    token1: int,
    token2: int,
    source_acts: Sequence[Tensor],
) -> Callable[[Sequence[PatchEntry]], float]:
    layers = _model_layers(model)
    state: dict[str, Sequence[PatchEntry]] = {"entries": []}
    handles = []
    for layer_index, layer in enumerate(layers):
        def hook(_module, inputs, layer_index=layer_index):
            entries = [entry for entry in state["entries"] if entry.layer == layer_index]
            if not entries:
                return None
            x = inputs[0].clone()
            for entry in entries:
                x[0, entry.position, entry.neuron] = source_acts[layer_index][
                    0, entry.source_position, entry.neuron
                ]
            return (x,) + tuple(inputs[1:])
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
    ids = torch.tensor([list(input_tokens)], dtype=torch.long, device=device)

    def evaluate(entries: Sequence[PatchEntry]) -> float:
        state["entries"] = entries
        with torch.no_grad():
            logits = _model_logits(model, ids)
            metric = logits[0, logit_position, token1].float() - logits[0, logit_position, token2].float()
        return float(metric.detach().cpu())

    # Caller owns the hooks through the lifetime of this evaluator and closes
    # them through the exposed cleanup attribute.
    setattr(evaluate, "remove_hooks", lambda: [handle.remove() for handle in handles])
    return evaluate


def _layer_counts_from_names(neuron_names: Iterable[str]) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for name in neuron_names:
        layer, neuron = parse_neuron_name(name)
        result.setdefault(layer, []).append(neuron)
    return {layer: sorted(set(values)) for layer, values in result.items()}


def _json_entry(entry: PatchEntry) -> dict:
    return {"neuron": entry.name, "layer": entry.layer, "index": entry.neuron,
            "position": entry.position, "source_position": entry.source_position,
            "attribution": entry.attribution}


def search_pair(
    model, tokenizer, device: torch.device, question: str, character: str, assistant_system: str
) -> dict:
    assistant_prompt = encode_chat_prompt(tokenizer, assistant_system, question)
    character_prompt = encode_chat_prompt(tokenizer, character, question)
    reply1 = greedy_decode(model, tokenizer, assistant_prompt, device, max_new_tokens=24)
    reply2 = greedy_decode(model, tokenizer, character_prompt, device, max_new_tokens=24)
    differing = first_differing_position(reply1, reply2)
    record = {
        "question": question,
        "character": character,
        "run1_reply": tokenizer.decode(reply1, skip_special_tokens=True).strip(),
        "run2_reply": tokenizer.decode(reply2, skip_special_tokens=True).strip(),
        "first_differing_position": differing,
    }
    if differing is None:
        record.update({"skipped": True, "skip_reason": "no_differing_token_in_both_greedy_replies"})
        return record

    token1, token2 = reply1[differing], reply2[differing]
    run1_context = assistant_prompt + reply1[:differing]
    run2_context = character_prompt + reply2[:differing]
    logit_position = len(run1_context) - 1
    source_acts = capture_run2_activations(model, run2_context, device)
    initial, acts1, grads = capture_run1_attribution(
        model, run1_context, device, logit_position, token1, token2
    )
    position_map = full_position_alignment(assistant_prompt, character_prompt, differing)
    ranked = rank_attribution_entries(acts1, source_acts, grads, position_map, limit=256)
    evaluator = make_patch_evaluator(
        model, run1_context, device, logit_position, token1, token2, source_acts
    )
    try:
        patched = greedy_patch(ranked, evaluator, initial_metric=initial, chunk_size=4, max_entries=256)
    finally:
        evaluator.remove_hooks()
    selected = patched["entries"] if patched["switched"] else []
    record.update({
        "skipped": False,
        "run1_token": int(token1),
        "run2_token": int(token2),
        "initial_metric": initial,
        "final_metric": patched["metric"],
        "switched": patched["switched"],
        "patch_forward_count": patched["forwards"],
        "attempted_entry_count": len(patched["attempted"]),
        "minimal_set": [_json_entry(entry) for entry in selected],
        "attempted_entries": [_json_entry(entry) for entry in patched["attempted"]],
    })
    return record


def register_zero_hooks(model, active_by_layer: dict[int, list[int]]):
    handles = []
    for layer_index, layer in enumerate(_model_layers(model)):
        def hook(_module, inputs, layer_index=layer_index):
            indices = active_by_layer.get(layer_index, [])
            if not indices:
                return None
            x = inputs[0].clone()
            x[..., indices] = 0
            return (x,) + tuple(inputs[1:])
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
    return handles


def validate_sets(
    model,
    tokenizer,
    device: torch.device,
    heldout_questions: Sequence[str],
    sets: dict[str, dict[int, list[int]]],
    assistant_system: str = "You are a helpful AI assistant.",
) -> dict:
    result = {}
    for condition, selected in sets.items():
        hooks = register_zero_hooks(model, selected) if selected else []
        replies = []
        try:
            for question in heldout_questions:
                prompt_ids = encode_chat_prompt(
                    tokenizer, assistant_system, question
                )
                generated = greedy_decode(model, tokenizer, prompt_ids, device, max_new_tokens=40)
                text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                replies.append({"question": question, "text": text, "ai_like": is_ai_reply(text)})
        finally:
            for handle in hooks:
                handle.remove()
        successes = sum(int(reply["ai_like"]) for reply in replies)
        low, high = wilson_interval(successes, len(replies))
        result[condition] = {
            "count_ai_like": successes,
            "n": len(replies),
            "rate": successes / len(replies) if replies else None,
            "wilson_95": [low, high],
            "replies": replies,
        }
    return result


def overlap_report(causal_names: Sequence[str], identity: dict, top_k: int) -> dict:
    identity_names, metadata = identity_topk(identity, top_k)
    causal = set(causal_names)
    baseline = set(identity_names)
    intersection = causal & baseline
    union = causal | baseline
    return {
        **metadata,
        "causal_set_size": len(causal),
        "identity_set_size": len(baseline),
        "intersection_count": len(intersection),
        "jaccard": len(intersection) / len(union) if union else 1.0,
        "shared_neurons": sorted(intersection, key=parse_neuron_name),
    }


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    assistant_system = spec.get("assistant_system", "You are a helpful AI assistant.")
    search_questions, heldout_questions = split_questions(spec["questions"], args.seed)
    pairs = [(question, character) for question in search_questions for character in spec["characters"]]
    if args.pairs_limit:
        pairs = pairs[: args.pairs_limit]
    if args.device == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("--device xpu requested, but torch.xpu is unavailable")
        device = torch.device("xpu")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = _model_layers(model)
    neurons_per_layer = int(layers[0].mlp.down_proj.in_features)

    result = {
        "model": args.model,
        "seed": args.seed,
        "device": args.device,
        "top_k": args.top_k,
        "prompt_pairs_file": str(Path(args.pairs)),
        "question_split": {
            "search_questions": search_questions,
            "heldout_questions": heldout_questions,
            "search_question_count": len(search_questions),
            "heldout_question_count": len(heldout_questions),
        },
        "search": {
            "requested_pair_count": 20 * len(spec["characters"]),
            "run_pair_count": len(pairs),
            "skipped_no_difference_count": 0,
            "pair_records": [],
            "neuron_counts": [],
            "top_set": [],
        },
        "validation": None,
        "identity_overlap": None,
        "complete": False,
    }
    output_path = Path(args.output)
    _atomic_json(output_path, result)
    print(f"Loaded {args.model}: {len(layers)} layers x {neurons_per_layer} neurons on {device}", flush=True)

    for index, (question, character) in enumerate(pairs, 1):
        record = search_pair(model, tokenizer, device, question, character, assistant_system)
        result["search"]["pair_records"].append(record)
        if record.get("skipped"):
            result["search"]["skipped_no_difference_count"] += 1
        counts, top_set = aggregate_pair_neurons(result["search"]["pair_records"], args.top_k)
        result["search"]["neuron_counts"] = [
            {"neuron": name, "pair_count": count}
            for name, count in sorted(counts.items(), key=lambda item: (-item[1], parse_neuron_name(item[0])))
        ]
        result["search"]["top_set"] = top_set
        _atomic_json(output_path, result)
        print(
            f"pair {index}/{len(pairs)}: first difference={record.get('first_differing_position')} "
            f"switched={record.get('switched', False)}; {time.time() - start:.0f}s",
            flush=True,
        )

    causal_names = [item["neuron"] for item in result["search"]["top_set"]]
    causal_by_layer = _layer_counts_from_names(causal_names)
    layer_counts = {layer: len(indices) for layer, indices in causal_by_layer.items()}
    random_sets = random_sets_with_layer_counts(
        layer_counts,
        neurons_per_layer,
        num_sets=6,
        seed=args.seed,
        exclude=causal_by_layer,
    )
    validation_sets: dict[str, dict[int, list[int]]] = {"unchanged": {}}
    validation_sets["causal_set_zeroed"] = causal_by_layer
    validation_sets.update({f"random_{i + 1}": selection for i, selection in enumerate(random_sets)})
    result["validation"] = validate_sets(
        model, tokenizer, device, heldout_questions, validation_sets, assistant_system
    )

    identity = json.loads(Path(args.identity).read_text(encoding="utf-8"))
    result["identity_overlap"] = overlap_report(causal_names, identity, args.top_k)
    result["complete"] = True
    result["elapsed_seconds"] = time.time() - start
    _atomic_json(output_path, result)
    print(f"wrote {output_path} in {result['elapsed_seconds']:.0f}s", flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("cpu", "xpu"), default="cpu")
    parser.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    parser.add_argument("--identity", default=str(DEFAULT_IDENTITY))
    parser.add_argument("--pairs-limit", type=int, default=0, help="smoke-test first N search pairs; 0 runs all")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="out/part16/search.json")
    args = parser.parse_args(argv)
    if args.pairs_limit < 0:
        parser.error("--pairs-limit must be nonnegative")
    if args.top_k < 0:
        parser.error("--top-k must be nonnegative")
    return args


if __name__ == "__main__":
    run(parse_args())
