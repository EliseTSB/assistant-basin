#!/usr/bin/env python3
"""Batched greedy/sampled generation that keeps down_proj interventions in forward.

This deliberately uses ordinary ``model(...)`` calls with the model's KV cache;
it does not replace the forward path, so registered ``mlp.down_proj`` pre-hooks
remain active for both prompt prefill and each decode token.
"""
from __future__ import annotations

import argparse
import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


DEFAULT_PROMPTS = [
    "Who made you?",
    "What is your name?",
    "Who created you?",
    "Which company trained you?",
    "Introduce yourself.",
    "What model are you?",
    "Explain why the sky appears blue.",
    "Give one example of a prime number.",
    "Describe a quiet beach in one sentence.",
    "What is the capital of France?",
    "How does a bicycle stay upright?",
    "Name a useful property of water.",
]


def _get_decoder_layers(model: Any) -> list[Any]:
    """Find the decoder layer sequence without assuming a model-family path."""
    candidates: list[tuple[int, str, list[Any]]] = []
    for name, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if layers is None or isinstance(layers, (str, bytes)):
            continue
        try:
            items = list(layers)
        except TypeError:
            continue
        if items and all(
            hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj")
            for layer in items
        ):
            candidates.append((len(items), name, items))
    if not candidates:
        raise ValueError("could not find decoder layers with mlp.down_proj modules")
    # Prefer the largest matching sequence (the vision tower is normally excluded
    # by the required MLP path); name gives deterministic tie-breaking.
    candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return candidates[0][2]


def find_decoder_layers(model: Any) -> list[Any]:
    """Public layer lookup used by the benchmark and intervention hooks."""
    return _get_decoder_layers(model)


def _normalise_operations(value: Mapping | None) -> dict[tuple[int, int], Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("operations must map (layer, unit) coordinates to operation specs")
    result: dict[tuple[int, int], Any] = {}
    for coordinate, spec in value.items():
        if not isinstance(coordinate, (tuple, list)) or len(coordinate) != 2:
            raise ValueError(f"operation coordinate must be (layer, unit), got {coordinate!r}")
        result[(int(coordinate[0]), int(coordinate[1]))] = spec
    return result


def _operation_layers(operations: Mapping[tuple[int, int], Any]) -> set[int]:
    return {layer for layer, _unit in operations}


def _apply_spec(values: torch.Tensor, spec: Any) -> None:
    """Apply one unit operation in place to a selected 1-D activation view.

    A number means set-to-constant, matching Parts 25/43. Mapping specs accept
    ``zero``, ``set``, ``scale``, ``add``, and ``clamp``/``clip``. A sequence of
    numbers means set one value per token position.
    """
    if spec is None:
        return
    if isinstance(spec, Mapping):
        known = {"zero", "set", "scale", "add", "clamp", "clip", "min", "max"}
        unknown = set(spec) - known
        if unknown:
            raise ValueError(f"unknown intervention keys: {sorted(unknown)}")
        if spec.get("zero", False):
            values.zero_()
        if "set" in spec:
            values.fill_(float(spec["set"]))
        if "scale" in spec:
            values.mul_(float(spec["scale"]))
        if "add" in spec:
            values.add_(float(spec["add"]))
        if "clamp" in spec:
            bounds = spec["clamp"]
            if isinstance(bounds, Mapping):
                lower = bounds.get("min", float("-inf"))
                upper = bounds.get("max", float("inf"))
            elif isinstance(bounds, Sequence) and not isinstance(bounds, (str, bytes)):
                if len(bounds) != 2:
                    raise ValueError("clamp sequence must be [minimum, maximum]")
                lower, upper = bounds
            else:
                limit = abs(float(bounds))
                lower, upper = -limit, limit
            values.clamp_(min=float(lower), max=float(upper))
        if "clip" in spec:
            limit = abs(float(spec["clip"]))
            values.clamp_(min=-limit, max=limit)
        if "min" in spec or "max" in spec:
            values.clamp_(
                min=float(spec.get("min", float("-inf"))),
                max=float(spec.get("max", float("inf"))),
            )
        return

    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        if len(spec) != values.numel():
            raise ValueError(
                f"per-position operation has {len(spec)} values for {values.numel()} tokens"
            )
        values.copy_(torch.as_tensor(spec, device=values.device, dtype=values.dtype))
        return

    values.fill_(float(spec))


class InterventionHooks:
    """Install reversible down_proj pre-hooks for shared or per-row operations."""

    def __init__(
        self,
        model: Any,
        operations: Mapping | None = None,
        per_row_operations: Sequence[Mapping] | None = None,
    ) -> None:
        if operations is not None and per_row_operations is not None:
            raise ValueError("pass operations or per_row_operations, not both")
        self.layers = _get_decoder_layers(model)
        self.shared_operations = _normalise_operations(operations)
        self.per_row_operations = (
            [_normalise_operations(row) for row in per_row_operations]
            if per_row_operations is not None
            else None
        )
        self.position_mask: torch.Tensor | None = None
        self.handles = []
        selected_layers = set(self.shared_operations_layer_ids)
        if self.per_row_operations is not None:
            for row in self.per_row_operations:
                selected_layers.update(_operation_layers(row))
        for layer_index in sorted(selected_layers):
            if layer_index < 0 or layer_index >= len(self.layers):
                raise ValueError(
                    f"target layer {layer_index} outside model with {len(self.layers)} layers"
                )

            def hook(_module, args, *, li=layer_index):
                if not args:
                    return None
                changed = self._patch(args[0], li)
                if changed is args[0]:
                    return None
                return (changed,) + tuple(args[1:])

            self.handles.append(
                self.layers[layer_index].mlp.down_proj.register_forward_pre_hook(hook)
            )

    @property
    def shared_operations_layer_ids(self) -> set[int]:
        return _operation_layers(self.shared_operations)

    def set_position_mask(self, mask: torch.Tensor | None) -> None:
        """Set valid positions for the next model forward (padding stays untouched)."""
        self.position_mask = mask

    def _row_maps(self, batch_size: int) -> list[dict[tuple[int, int], Any]]:
        if self.per_row_operations is None:
            return [self.shared_operations] * batch_size
        if len(self.per_row_operations) != batch_size:
            raise ValueError(
                f"got {len(self.per_row_operations)} row operation maps for batch {batch_size}"
            )
        return self.per_row_operations

    def _patch(self, x: torch.Tensor, layer_index: int) -> torch.Tensor:
        if x.ndim not in (2, 3):
            raise ValueError(f"down_proj input must have 2 or 3 dimensions, got {tuple(x.shape)}")

        mask = self.position_mask
        if mask is None:
            if x.ndim == 3:
                view_shape = x.shape[:2]
            else:
                view_shape = (x.shape[0], 1)
            valid_mask = None
        else:
            mask = mask.to(device=x.device, dtype=torch.bool)
            if mask.ndim != 2:
                raise ValueError("position mask must have [batch, sequence] shape")
            view_shape = tuple(mask.shape)
            valid_mask = mask

        if x.ndim == 3:
            if tuple(x.shape[:2]) != tuple(view_shape):
                raise ValueError(
                    f"hook input shape {tuple(x.shape[:2])} does not match mask {tuple(view_shape)}"
                )
            view = x
        else:
            if x.shape[0] != view_shape[0] * view_shape[1]:
                if view_shape[1] == 1 and x.shape[0] == view_shape[0]:
                    pass
                else:
                    raise ValueError(
                        f"flattened hook input has {x.shape[0]} rows, expected "
                        f"{view_shape[0]}x{view_shape[1]}"
                    )
            view = x.reshape(view_shape[0], view_shape[1], x.shape[-1])

        row_maps = self._row_maps(view.shape[0])
        targets = [
            (row, unit, spec)
            for row, row_map in enumerate(row_maps)
            for (layer, unit), spec in row_map.items()
            if layer == layer_index
        ]
        if not targets:
            return x

        output = x.clone()
        output_view = output.reshape(view_shape[0], view_shape[1], x.shape[-1])
        for row, unit, spec in targets:
            if valid_mask is None:
                selected = output_view[row, :, unit]
                _apply_spec(selected, spec)
            else:
                row_mask = valid_mask[row]
                selected = output_view[row, :, unit]
                changed = selected[row_mask].clone()
                _apply_spec(changed, spec)
                selected[row_mask] = changed
        return output

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self) -> "InterventionHooks":
        return self

    def __exit__(self, _kind, _value, _traceback) -> None:
        self.close()


def intervention_hooks(
    model: Any,
    operations: Mapping | None = None,
    per_row_operations: Sequence[Mapping] | None = None,
) -> InterventionHooks:
    """Convenience constructor for the reversible hook manager."""
    return InterventionHooks(model, operations, per_row_operations)


def _token_ids(tokenizer: Any, prompt: Any) -> list[int]:
    if isinstance(prompt, torch.Tensor):
        values = prompt.detach().cpu().tolist()
    elif isinstance(prompt, Sequence) and not isinstance(prompt, (str, bytes)):
        values = list(prompt)
    else:
        encoded = tokenizer(str(prompt), add_special_tokens=False)
        values = encoded["input_ids"]
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().tolist()
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise ValueError("each prompt must encode to one token sequence")
        values = values[0]
    result = [int(token) for token in values]
    if not result:
        raise ValueError("generation prompt encoded to zero tokens")
    return result


def encode_prompts(tokenizer: Any, prompts: Sequence[Any]) -> list[list[int]]:
    """Tokenize prompts independently, preserving one-tokenization-per-row semantics."""
    return [_token_ids(tokenizer, prompt) for prompt in prompts]


def _pad_token_id(tokenizer: Any) -> int:
    value = getattr(tokenizer, "pad_token_id", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    return int(value) if value is not None else 0


def make_left_padded_batch(
    token_sequences: Sequence[Sequence[int]], tokenizer: Any, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build left-padded input IDs and an attention mask from individual sequences."""
    return make_padded_batch(token_sequences, tokenizer, device, padding_side="left")


def make_padded_batch(
    token_sequences: Sequence[Sequence[int]],
    tokenizer: Any,
    device: torch.device | str,
    *,
    padding_side: str = "left",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a padded batch on either side, preserving each row's valid-token mask."""
    if not token_sequences:
        raise ValueError("cannot build an empty input batch")
    if padding_side not in ("left", "right"):
        raise ValueError("padding_side must be 'left' or 'right'")
    width = max(len(row) for row in token_sequences)
    pad_id = _pad_token_id(tokenizer)
    ids = torch.full((len(token_sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row_index, sequence in enumerate(token_sequences):
        row = torch.as_tensor(sequence, dtype=torch.long, device=device)
        start = width - len(row) if padding_side == "left" else 0
        ids[row_index, start:start + len(row)] = row
        mask[row_index, start:start + len(row)] = 1
    return ids, mask


def _accepts_argument(function: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        parameter.name == name or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    positions = attention_mask.long().cumsum(dim=-1) - 1
    return positions.masked_fill(attention_mask == 0, 0)


def model_call(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    past_key_values: Any = None,
    cache_position: torch.Tensor | None = None,
    use_cache: bool = True,
) -> Any:
    """Call a causal LM with mask-derived positions and supported cache kwargs."""
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": use_cache,
    }
    if _accepts_argument(model.forward, "position_ids"):
        kwargs["position_ids"] = _position_ids(attention_mask)[:, -input_ids.shape[1]:]
    if past_key_values is not None and _accepts_argument(model.forward, "past_key_values"):
        kwargs["past_key_values"] = past_key_values
    if cache_position is not None and _accepts_argument(model.forward, "cache_position"):
        kwargs["cache_position"] = cache_position
    return model(**kwargs)


def extract_logits_and_cache(output: Any) -> tuple[torch.Tensor, Any]:
    logits = getattr(output, "logits", None)
    cache = getattr(output, "past_key_values", None)
    if logits is None:
        if isinstance(output, (tuple, list)) and output:
            logits = output[0]
            if len(output) > 1:
                cache = output[1]
        else:
            raise TypeError("model output does not contain logits")
    return logits, cache


def _normalise_eos(eos_token_id: Any) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, torch.Tensor):
        eos_token_id = eos_token_id.detach().cpu().tolist()
    if isinstance(eos_token_id, Sequence) and not isinstance(eos_token_id, (str, bytes)):
        return {int(token) for token in eos_token_id}
    return {int(eos_token_id)}


def _sample_next(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    generators: Sequence[torch.Generator | None],
) -> torch.Tensor:
    if not do_sample:
        return logits.argmax(dim=-1)
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero for sampling")
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    choices = [
        torch.multinomial(probabilities[row], 1, generator=generators[row]).squeeze(0)
        for row in range(probabilities.shape[0])
    ]
    return torch.stack(choices).to(device=logits.device, dtype=torch.long)


def batched_generate(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Any],
    *,
    max_new_tokens: int = 48,
    operations: Mapping | None = None,
    per_row_operations: Sequence[Mapping] | None = None,
    do_sample: bool = False,
    seed: int | None = None,
    row_seeds: Sequence[int] | None = None,
    temperature: float = 1.0,
    device: torch.device | str | None = None,
    eos_token_id: Any = None,
    generated_only: bool = False,
    padding_side: str = "left",
) -> list[dict[str, Any]]:
    """Generate a batch and return one ``text``/``token_ids`` record per prompt.

    ``operations`` applies the same ``(layer, unit)`` specs to every row.
    ``per_row_operations`` supplies one mapping per prompt. Numeric specs set a
    unit; mapping specs support zero/set/scale/add/clamp/clip. The hook is active
    in prefill and decode and ignores left-padding and finished-row padding.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if operations is not None and per_row_operations is not None:
        raise ValueError("pass operations or per_row_operations, not both")
    if seed is not None and row_seeds is not None:
        raise ValueError("pass seed or row_seeds, not both")
    prompt_list = list(prompts)
    if not prompt_list:
        return []
    if per_row_operations is not None and len(per_row_operations) != len(prompt_list):
        raise ValueError(
            f"got {len(per_row_operations)} row operation maps for batch {len(prompt_list)}"
        )
    if row_seeds is not None and len(row_seeds) != len(prompt_list):
        raise ValueError(f"got {len(row_seeds)} row seeds for batch {len(prompt_list)}")
    token_sequences = encode_prompts(tokenizer, prompt_list)
    if device is None:
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cpu")
    device = torch.device(device)
    input_ids, attention_mask = make_padded_batch(
        token_sequences, tokenizer, device, padding_side=padding_side
    )
    batch_size = len(prompt_list)
    generated: list[list[int]] = [[] for _ in range(batch_size)]
    if max_new_tokens == 0:
        return [
            {
                "prompt": prompt if isinstance(prompt, str) else None,
                "prompt_token_ids": list(tokens),
                "token_ids": [],
                "text": "",
            }
            for prompt, tokens in zip(prompt_list, token_sequences)
        ]

    eos_ids = _normalise_eos(
        getattr(tokenizer, "eos_token_id", None) if eos_token_id is None else eos_token_id
    )
    pad_id = _pad_token_id(tokenizer)
    finished = [False] * batch_size
    generators: list[torch.Generator | None] = [None] * batch_size
    if do_sample and (seed is not None or row_seeds is not None):
        selected_seeds = list(row_seeds) if row_seeds is not None else [int(seed) + row for row in range(batch_size)]
        for row, row_seed in enumerate(selected_seeds):
            generator = torch.Generator(device=device)
            generator.manual_seed(int(row_seed))
            generators[row] = generator

    all_input_ids = input_ids
    current_mask = attention_mask
    # During prefill the down_proj hooks are installed but generated-only clamps
    # receive an all-false mask. Keep a separate mask so uncached full-sequence
    # forwards can still patch continuation positions without touching prompts.
    clamp_mask = torch.zeros_like(attention_mask) if generated_only else attention_mask.clone()
    hook_manager = InterventionHooks(model, operations, per_row_operations)
    hook_manager.set_position_mask(clamp_mask)
    past_key_values = None
    logits: torch.Tensor | None = None
    try:
        with torch.inference_mode():
            initial_cache_position = torch.arange(
                input_ids.shape[1], dtype=torch.long, device=device
            )
            output = model_call(
                model,
                input_ids,
                attention_mask,
                cache_position=initial_cache_position,
                use_cache=True,
            )
            logits, past_key_values = extract_logits_and_cache(output)

            for step in range(max_new_tokens):
                if logits.shape[1] == 1:
                    # Cached decoding returns logits for the new token only.
                    next_logits = logits[:, -1, :]
                else:
                    positions = torch.arange(logits.shape[1], device=device).expand_as(current_mask)
                    last_positions = (positions * current_mask.long()).amax(dim=1)
                    next_logits = logits[torch.arange(batch_size, device=device), last_positions]
                next_tokens = _sample_next(
                    next_logits,
                    do_sample=do_sample,
                    temperature=temperature,
                    generators=generators,
                )
                emitted = next_tokens.detach().cpu().tolist()
                was_active = [not done for done in finished]
                for row, token in enumerate(emitted):
                    if was_active[row]:
                        generated[row].append(int(token))
                        if token in eos_ids:
                            finished[row] = True
                if step + 1 >= max_new_tokens or all(finished):
                    break

                # Cache the just-generated token. EOS is included for its row;
                # later filler tokens for completed rows have mask 0.
                token_input = next_tokens[:, None]
                active_mask = torch.tensor(was_active, dtype=current_mask.dtype, device=device)[:, None]
                token_input = torch.where(
                    active_mask.bool(),
                    token_input,
                    torch.full_like(token_input, pad_id),
                )
                current_mask = torch.cat((current_mask, active_mask), dim=1)
                clamp_mask = torch.cat((clamp_mask, active_mask), dim=1)
                all_input_ids = torch.cat((all_input_ids, token_input), dim=1)
                next_cache_position = torch.tensor(
                    [all_input_ids.shape[1] - 1], dtype=torch.long, device=device
                )
                if past_key_values is not None:
                    hook_manager.set_position_mask(active_mask)
                    output = model_call(
                        model,
                        token_input,
                        current_mask,
                        past_key_values=past_key_values,
                        cache_position=next_cache_position,
                        use_cache=True,
                    )
                else:
                    # Some model implementations do not expose a cache. Re-run
                    # the complete left-padded sequence while preserving masks.
                    filler = torch.full_like(token_input, pad_id)
                    token_input = torch.where(active_mask.bool(), token_input, filler)
                    all_input_ids[:, -1:] = token_input
                    hook_manager.set_position_mask(clamp_mask)
                    output = model_call(
                        model, all_input_ids, current_mask, use_cache=False
                    )
                logits, new_cache = extract_logits_and_cache(output)
                if past_key_values is not None:
                    past_key_values = new_cache
    finally:
        hook_manager.close()

    results = []
    for prompt, prompt_tokens, output_tokens in zip(prompt_list, token_sequences, generated):
        text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
        results.append(
            {
                "prompt": prompt if isinstance(prompt, str) else None,
                "prompt_token_ids": list(prompt_tokens),
                "token_ids": output_tokens,
                "text": text,
            }
        )
    return results


def load_pretrained_model(
    model_id: str,
    device: str,
    *,
    cache_dir: str | None = None,
    dtype: str = "auto",
    attn_implementation: str = "sdpa",
) -> tuple[Any, Any]:
    """Load an HF causal LM/tokenizer; device imports stay lazy for CPU tests."""
    target = torch.device(device)
    if target.type == "xpu":
        try:
            import intel_extension_for_pytorch  # noqa: F401
        except ImportError:
            # Recent PyTorch XPU builds can work without IPEX, so keep going; the
            # model load below will give the actionable backend error if needed.
            pass
    from transformers import AutoModelForCausalLM, AutoTokenizer

    options: dict[str, Any] = {"cache_dir": cache_dir} if cache_dir else {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **options)
    tokenizer.padding_side = "left"
    torch_dtype: Any = dtype
    if dtype != "auto":
        torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    )
    model.to(target)
    model.eval()
    return tokenizer, model


def compare_batched_with_generate(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
    device: str,
) -> dict[str, Any]:
    """Compare batched greedy IDs with the ordinary one-prompt HF generate path."""
    selected = list(prompts)
    batched = batched_generate(
        model, tokenizer, selected, max_new_tokens=max_new_tokens, device=device
    )
    mismatches = []
    for index, prompt in enumerate(selected):
        prompt_ids = _token_ids(tokenizer, prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        mask = torch.ones_like(input_ids)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        pad_id = _pad_token_id(tokenizer)
        if pad_id is not None:
            kwargs["pad_token_id"] = pad_id
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            kwargs["eos_token_id"] = eos
        with torch.inference_mode():
            ordinary = model.generate(**kwargs)
        expected = ordinary[0, input_ids.shape[1]:].detach().cpu().tolist()
        actual = batched[index]["token_ids"]
        if expected != actual:
            mismatches.append(
                {
                    "index": index,
                    "prompt": prompt,
                    "unbatched_token_ids": expected,
                    "batched_token_ids": actual,
                }
            )
    return {
        "checked": len(selected),
        "mismatches": mismatches,
        "equal": not mismatches,
    }


def _read_prompts(path: str | None, count: int) -> list[str]:
    if path:
        with open(path, encoding="utf-8") as stream:
            prompts = [line.rstrip("\n") for line in stream if line.strip()]
    else:
        prompts = DEFAULT_PROMPTS
    if not prompts:
        raise ValueError("no prompts to check")
    if count > len(prompts):
        prompts = [prompts[index % len(prompts)] for index in range(count)]
    return prompts[:count]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda", "xpu"))
    parser.add_argument("--cache-dir")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--attn-implementation", default="sdpa", choices=("sdpa", "eager"))
    parser.add_argument("--check-equal", type=int, required=True, metavar="N",
                        help="compare batched greedy output with model.generate on the first N prompts")
    parser.add_argument("--prompts-file", help="UTF-8 file with one prompt per line")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args(argv)
    if args.check_equal <= 0:
        parser.error("--check-equal must be greater than zero")
    prompts = _read_prompts(args.prompts_file, args.check_equal)
    tokenizer, model = load_pretrained_model(
        args.model,
        args.device,
        cache_dir=args.cache_dir,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    result = compare_batched_with_generate(
        model, tokenizer, prompts,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
