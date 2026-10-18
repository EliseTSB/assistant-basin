#!/usr/bin/env python3
"""Validate Part 16's causal neuron set with generation and neutral-text loss."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

import causal_search


ROOT = Path(__file__).resolve().parent
MODEL_NAME = "allenai/OLMo-2-0425-1B-Instruct"
DEFAULT_SEARCH = ROOT / "results" / "part16" / "search.json"
DEFAULT_OUTPUT = ROOT / "out" / "part16b" / "validate.json"
NEUTRAL_PATH = ROOT / "data" / "neutral.txt"
ASSISTANT_SYSTEM = "You are a helpful AI assistant."
MEAN_TOKEN_END = 1024
EVAL_TOKEN_END = 2048
SAMPLED_REPLIES = 4
MAX_NEW_TOKENS = 48


def names_to_layers(neuron_names: Sequence[str]) -> dict[int, list[int]]:
    by_layer: dict[int, set[int]] = {}
    for name in neuron_names:
        layer, neuron = causal_search.parse_neuron_name(name)
        if neuron in by_layer.setdefault(layer, set()):
            raise ValueError(f"duplicate neuron in causal set: {name}")
        by_layer[layer].add(neuron)
    return {layer: sorted(neurons) for layer, neurons in sorted(by_layer.items())}


def layers_to_names(selection: dict[int, Sequence[int]]) -> list[str]:
    return [
        f"L{layer}.{neuron}"
        for layer, neurons in sorted(selection.items())
        for neuron in sorted(neurons)
    ]


def _layer_subset(selection: dict[int, list[int]], low: int, high: int) -> dict[int, list[int]]:
    return {layer: list(neurons) for layer, neurons in selection.items() if low <= layer <= high}


def build_conditions(
    top_names: Sequence[str],
    layer_count: int,
    neurons_per_layer: int,
    *,
    seed: int = 0,
) -> dict[str, dict]:
    """Construct the seven fixed condition families and their neuron lists."""
    full = names_to_layers(top_names)
    for layer, neurons in full.items():
        if not 0 <= layer < layer_count:
            raise ValueError(f"causal neuron layer {layer} outside model with {layer_count} layers")
        if any(neuron < 0 or neuron >= neurons_per_layer for neuron in neurons):
            raise ValueError(f"causal neuron index outside layer width {neurons_per_layer}: L{layer}")

    late = _layer_subset(full, 10, 15)
    early = _layer_subset(full, 0, 3)
    full_counts = {layer: len(neurons) for layer, neurons in full.items()}
    late_counts = {layer: len(neurons) for layer, neurons in late.items()}
    random_full = causal_search.random_sets_with_layer_counts(
        full_counts,
        neurons_per_layer,
        num_sets=6,
        seed=seed,
        exclude=full,
    )
    random_late = causal_search.random_sets_with_layer_counts(
        late_counts,
        neurons_per_layer,
        num_sets=3,
        seed=seed + 6,
        exclude=late,
    )

    conditions: dict[str, dict] = {
        "unchanged": {"mode": "none", "selection": {}},
        "full64_zero": {"mode": "zero", "selection": full},
        "full64_mean": {"mode": "mean", "selection": full},
        "late_zero": {"mode": "zero", "selection": late},
        "early_zero": {"mode": "zero", "selection": early},
    }
    conditions.update(
        {
            f"random_{index}": {"mode": "zero", "selection": selection}
            for index, selection in enumerate(random_full, start=1)
        }
    )
    conditions.update(
        {
            f"late_random_{index}": {"mode": "zero", "selection": selection}
            for index, selection in enumerate(random_late, start=1)
        }
    )
    return conditions


def sample_seed(question_index: int, sample_index: int, base_seed: int = 0) -> int:
    """Stable sampling seed for a question/sample pair, independent of condition."""
    if question_index < 0 or sample_index < 1:
        raise ValueError("sampled reply indices must be nonnegative and start at sample 1")
    return base_seed + question_index * 10_000 + sample_index


def build_seed_plan(
    condition_names: Sequence[str],
    question_count: int,
    *,
    sampled_count: int = SAMPLED_REPLIES,
    base_seed: int = 0,
) -> dict[str, dict[int, dict[int, int]]]:
    return {
        condition: {
            question_index: {
                sample_index: sample_seed(question_index, sample_index, base_seed)
                for sample_index in range(1, sampled_count + 1)
            }
            for question_index in range(question_count)
        }
        for condition in condition_names
    }


def make_ablation_hook(
    layer_index: int,
    selection: dict[int, Sequence[int]],
    mode: str,
    neutral_means: dict[int, torch.Tensor] | None = None,
):
    """Make a shape-preserving down_proj pre-hook for one selected layer."""
    if mode not in {"zero", "mean"}:
        raise ValueError(f"unsupported ablation mode: {mode}")
    indices = list(selection.get(layer_index, ()))
    if not indices:
        raise ValueError(f"no selected neurons for layer {layer_index}")
    if mode == "mean" and (neutral_means is None or layer_index not in neutral_means):
        raise ValueError(f"missing neutral mean for layer {layer_index}")
    mean_values = neutral_means[layer_index] if mode == "mean" and neutral_means else None
    index_tensor_cache: dict[tuple[str, int | None], torch.Tensor] = {}

    def hook(_module, inputs):
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise ValueError("down_proj hook expected a tensor input")
        activation = inputs[0]
        # OLMo's down_proj input is [batch, sequence, width]. Requiring the
        # batch axis catches the old [1, seq, width] / [seq, width] mix-up.
        if activation.ndim != 3:
            raise ValueError(f"expected [batch, sequence, width] activation, got {tuple(activation.shape)}")
        cache_key = (str(activation.device), activation.device.index)
        indices_tensor = index_tensor_cache.get(cache_key)
        if indices_tensor is None:
            indices_tensor = torch.tensor(indices, dtype=torch.long, device=activation.device)
            index_tensor_cache[cache_key] = indices_tensor
        changed = activation.clone()
        if mode == "zero":
            changed.index_fill_(-1, indices_tensor, 0)
        else:
            values = mean_values.index_select(0, indices_tensor.to(mean_values.device))
            values = values.to(device=activation.device, dtype=activation.dtype)
            changed[..., indices_tensor] = values
        return (changed, *inputs[1:])

    return hook


def register_ablation_hooks(
    model,
    selection: dict[int, Sequence[int]],
    mode: str,
    neutral_means: dict[int, torch.Tensor] | None = None,
) -> list:
    if mode == "none":
        return []
    handles = []
    layers = model.model.layers
    try:
        for layer_index in sorted(selection):
            if selection[layer_index]:
                layer = layers[layer_index]
                handle = layer.mlp.down_proj.register_forward_pre_hook(
                    make_ablation_hook(layer_index, selection, mode, neutral_means)
                )
                handles.append(handle)
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return handles


def capture_neutral_means(model, fit_ids: Sequence[int], device: torch.device) -> dict[int, torch.Tensor]:
    """Average each layer's MLP input over the first 1024 neutral-text tokens."""
    layers = model.model.layers
    captured: list[torch.Tensor | None] = [None] * len(layers)
    handles = []
    for layer_index, layer in enumerate(layers):
        def hook(_module, inputs, index=layer_index):
            activation = inputs[0]
            if activation.ndim != 3:
                raise ValueError(f"expected batched 3D activation, got {tuple(activation.shape)}")
            captured[index] = activation.detach().float().mean(dim=(0, 1)).cpu()

        handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
    try:
        ids = torch.tensor([list(fit_ids)], dtype=torch.long, device=device)
        with torch.inference_mode():
            model.model(input_ids=ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in captured):
        raise RuntimeError("failed to capture every layer's neutral-text mean")
    return {index: value for index, value in enumerate(captured) if value is not None}


def _hidden_state(output) -> torch.Tensor:
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, dict):
        hidden = output.get("last_hidden_state")
    if hidden is None:
        raise ValueError("base model forward did not return last_hidden_state")
    return hidden


def neutral_next_token_loss(model, neutral_ids: Sequence[int], device: torch.device) -> float:
    """Mean CE loss for targets 1024 through 2047 with preceding neutral context."""
    if len(neutral_ids) < EVAL_TOKEN_END:
        raise ValueError(f"neutral text has {len(neutral_ids)} tokens; need at least {EVAL_TOKEN_END}")
    # Include the first 1024 tokens as causal context, while evaluating only
    # the held-out target range [1024, 2048).
    input_ids = torch.tensor([list(neutral_ids[: EVAL_TOKEN_END - 1])], dtype=torch.long, device=device)
    targets = torch.tensor(list(neutral_ids), dtype=torch.long, device=device)
    with torch.inference_mode():
        hidden = _hidden_state(model.model(input_ids=input_ids, use_cache=False))[0]
        total_loss = 0.0
        total_count = 0
        for start in range(MEAN_TOKEN_END - 1, EVAL_TOKEN_END - 1, 128):
            end = min(start + 128, EVAL_TOKEN_END - 1)
            logits = model.lm_head(hidden[start:end]).float()
            labels = targets[start + 1 : end + 1]
            total_loss += float(F.cross_entropy(logits, labels, reduction="sum").item())
            total_count += end - start
        del hidden
    if total_count != EVAL_TOKEN_END - MEAN_TOKEN_END:
        raise AssertionError(f"expected 1024 neutral targets, computed {total_count}")
    return total_loss / total_count


def seed_generation(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.manual_seed_all(seed)


def generate_reply(
    model,
    tokenizer,
    prompt_ids: Sequence[int],
    device: torch.device,
    *,
    sampled: bool,
    seed: int | None = None,
) -> str:
    if sampled:
        if seed is None:
            raise ValueError("sampled generation requires a seed")
        seed_generation(seed, device)
    ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(ids)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    kwargs = {
        "input_ids": ids,
        "attention_mask": attention_mask,
        "do_sample": sampled,
        "max_new_tokens": MAX_NEW_TOKENS,
        "pad_token_id": pad_token_id,
    }
    if sampled:
        kwargs["temperature"] = 0.7
    # Generate one reply per call; this stays below the eight-sequence limit.
    with torch.inference_mode():
        generated = model.generate(**kwargs)
    new_tokens = generated[0, ids.shape[1] :].detach().cpu().tolist()
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def clear_device_cache(device: torch.device) -> None:
    if device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()
        torch.xpu.empty_cache()


def run_validation(
    model,
    tokenizer,
    *,
    device: torch.device,
    search_data: dict,
    neutral_text: str,
    smoke: bool = False,
    seed: int | None = None,
) -> dict:
    heldout = search_data.get("question_split", {}).get("heldout_questions")
    if not isinstance(heldout, list) or len(heldout) != 10:
        raise ValueError("search JSON must contain its 10-question heldout split")
    top_rows = search_data.get("search", {}).get("top_set")
    if not isinstance(top_rows, list) or len(top_rows) != 64:
        raise ValueError("search JSON must contain the 64-neuron search.top_set")
    top_names = [str(row["neuron"]) for row in top_rows]
    token_ids = tokenizer(neutral_text, add_special_tokens=False)["input_ids"]
    if len(token_ids) < EVAL_TOKEN_END:
        raise ValueError(f"neutral text encodes to {len(token_ids)} tokens; need at least {EVAL_TOKEN_END}")
    token_ids = list(map(int, token_ids[:EVAL_TOKEN_END]))
    layers = model.model.layers
    layer_count = len(layers)
    neurons_per_layer = int(layers[0].mlp.down_proj.in_features)
    selected_seed = int(search_data.get("seed", 0) if seed is None else seed)
    conditions = build_conditions(top_names, layer_count, neurons_per_layer, seed=selected_seed)
    questions = heldout
    if smoke:
        questions = questions[:2]
        conditions = {name: conditions[name] for name in ("unchanged", "full64_zero")}
    seed_plan = build_seed_plan(list(conditions), len(questions), base_seed=selected_seed)
    prompt_ids = [
        causal_search.encode_chat_prompt(tokenizer, ASSISTANT_SYSTEM, question)
        for question in questions
    ]
    means = capture_neutral_means(model, token_ids[:MEAN_TOKEN_END], device)

    result = {
        "model": MODEL_NAME,
        "device": str(device),
        "search": str(DEFAULT_SEARCH),
        "question_split": "search.question_split.heldout_questions",
        "question_count": len(questions),
        "sampled_replies_per_question": SAMPLED_REPLIES,
        "max_new_tokens": MAX_NEW_TOKENS,
        "sample_temperature": 0.7,
        "seed": selected_seed,
        "smoke": smoke,
        "regex_source": "causal_search.is_ai_reply",
        "conditions": {},
    }
    _t0 = time.time()
    for condition_name, condition in conditions.items():
        print(f"condition {condition_name} start t={time.time() - _t0:.0f}s", flush=True)
        selection = condition["selection"]
        handles = register_ablation_hooks(model, selection, condition["mode"], means)
        try:
            loss = neutral_next_token_loss(model, token_ids, device)
            replies = []
            for question_index, (question, encoded) in enumerate(zip(questions, prompt_ids)):
                generations = [(0, False, None)] + [
                    (sample_index, True, seed_plan[condition_name][question_index][sample_index])
                    for sample_index in range(1, SAMPLED_REPLIES + 1)
                ]
                for sample_index, sampled, reply_seed in generations:
                    text = generate_reply(
                        model,
                        tokenizer,
                        encoded,
                        device,
                        sampled=sampled,
                        seed=reply_seed,
                    )
                    regex_ai = causal_search.is_ai_reply(text)
                    # Progress (2026-10-17): the first full run stalled silently after 63 min with an empty log, so
                    # every reply now reports where the run is.
                    print(f"{condition_name} q{question_index} s{sample_index} t={time.time() - _t0:.0f}s", flush=True)
                    replies.append(
                        {
                            "condition": condition_name,
                            "question": question,
                            "question_index": question_index,
                            "sample_index": sample_index,
                            "generation": "sampled" if sampled else "greedy",
                            "seed": reply_seed,
                            "text": text,
                            "regex_ai": regex_ai,
                            "regex_label": "ai" if regex_ai else "not_ai",
                        }
                    )
            result["conditions"][condition_name] = {
                "mode": condition["mode"],
                "neurons": layers_to_names(selection),
                "neutral_loss": loss,
                "replies": replies,
            }
        finally:
            for handle in handles:
                handle.remove()
            clear_device_cache(device)
            gc.collect()
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "xpu"), default="cpu")
    parser.add_argument("--search", default=str(DEFAULT_SEARCH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=None, help="override the seed stored in the search result")
    parser.add_argument("--smoke", action="store_true", help="run the first two held-out questions and two conditions")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("--device xpu requested, but torch.xpu is unavailable")
    search_path = Path(args.search)
    search_data = json.loads(search_path.read_text(encoding="utf-8"))
    neutral_text = NEUTRAL_PATH.read_text(encoding="utf-8")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if device.type == "xpu" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    result = run_validation(
        model,
        tokenizer,
        device=device,
        search_data=search_data,
        neutral_text=neutral_text,
        smoke=args.smoke,
        seed=args.seed,
    )
    result["search"] = str(search_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(output_path)
    print(f"wrote {output_path}", flush=True)
    return result


if __name__ == "__main__":
    main()
