#!/usr/bin/env python3
"""Collect down-projection inputs while materializing one decoder layer at a time.

The output format and prompt construction follow collect.py.  Unlike collect.py,
this script builds a meta-device model skeleton, loads the token embedding once,
and then loads only the current decoder layer from the checkpoint before running
that layer over every prompt's hidden state.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


@dataclass
class Prompt:
    form: str
    family: str
    row: int
    text: str
    target: str
    token_ids: list[int]
    offsets: list[tuple[int, int]]
    span_tokens: list[int]


def _load_spec(root: Path, max_prompts: int | None) -> dict[str, Any]:
    with (root / "data" / "frames.json").open(encoding="utf-8") as handle:
        spec = json.load(handle)
    extra = root / "data" / "frames-extra.json"
    if extra.exists():
        with extra.open(encoding="utf-8") as handle:
            more = json.load(handle)
        spec["users"] += more["users"]
        spec["frames"] += more["frames"]
    if max_prompts is not None:
        # The input frames are index-paired with user questions.  Keeping the
        # first N pairs yields N rows in each of the six output families.
        spec["users"] = spec["users"][:max_prompts]
        spec["frames"] = spec["frames"][:max_prompts]
    return spec


def _filled(frame: str, name: str, maker: str) -> str:
    return frame.replace("{N}", name).replace("{M}", maker)


def _render(tok: Any, form: str, user: str, assistant: str) -> str:
    if form == "plain":
        return f"User: {user}\nAssistant: {assistant}"
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    return tok.apply_chat_template(messages, tokenize=False)


def build_prompts(
    tok: Any,
    spec: dict[str, Any],
    self_name: str | None,
    max_prompts: int | None = None,
) -> tuple[list[str], dict[tuple[str, str], list[Prompt]]]:
    """Build and tokenize prompts in collect.py's frame/family order."""
    names = spec["names"]
    if self_name:
        if ":" not in self_name:
            raise ValueError("--self-name must have the form NAME:MAKER")
        own_name, own_maker = self_name.split(":", 1)
    else:
        own_name, own_maker = names["olmo"]

    forms = ["plain"] + (["chat"] if getattr(tok, "chat_template", None) else [])
    keys = ("A", "B", "C", "D", "E_claude", "E_chatgpt")
    records: dict[tuple[str, str], list[Prompt]] = {
        (form, family): [] for form in forms for family in keys
    }
    count = len(spec["frames"])
    if max_prompts is not None:
        count = min(count, max_prompts)

    for form in forms:
        for row, (frame, user) in enumerate(zip(spec["frames"][:count], spec["users"][:count])):
            own_a = _filled(frame[0], own_name, own_maker)
            own_b = _filled(frame[1], own_name, own_maker)
            frame_c = frame[2]
            family_specs = [
                ("A", user, own_a, own_a),
                ("B", user, own_b, own_b),
                ("C", user, frame_c, frame_c),
                ("D", own_a, "Okay.", own_a),
            ]
            for family, prompt_user, assistant, target in family_specs:
                records[(form, family)].append(
                    _tokenize_prompt(tok, form, family, row, prompt_user, assistant, target)
                )
            for family, name_key in (("E_claude", "claude"), ("E_chatgpt", "chatgpt")):
                name, maker = names[name_key]
                target = _filled(frame[0], name, maker)
                records[(form, family)].append(
                    _tokenize_prompt(tok, form, family, row, user, target, target)
                )
    return forms, records


def _tokenize_prompt(
    tok: Any,
    form: str,
    family: str,
    row: int,
    user: str,
    assistant: str,
    target: str,
) -> Prompt:
    rendered = _render(tok, form, user, assistant)
    encoded = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
    start = rendered.rfind(target)
    if start < 0:
        raise ValueError(f"target not in rendered prompt for family {family}")
    end = start + len(target)
    span = [i for i, (left, right) in enumerate(offsets) if right > start and left < end and right > left]
    if not span:
        raise ValueError(f"no tokens in target span for family {family}")
    return Prompt(form, family, row, rendered, target, ids, offsets, span)


def _checkpoint_files(model_id: str, revision: str | None) -> tuple[Path, dict[str, str]]:
    """Return a safetensors tensor-to-file map without downloading anything."""
    model_path = Path(model_id)
    if model_path.is_dir():
        snapshot = model_path
    else:
        snapshot = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                local_files_only=True,
            )
        )

    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as handle:
            raw_map = json.load(handle)["weight_map"]
        if any(not filename.endswith(".safetensors") for filename in raw_map.values()):
            raise ValueError("the checkpoint index contains non-safetensors weights")
        return snapshot, raw_map

    single = snapshot / "model.safetensors"
    if single.is_file():
        with safe_open(str(single), framework="pt", device="cpu") as handle:
            return snapshot, {key: single.name for key in handle.keys()}

    raise FileNotFoundError(
        f"no model.safetensors.index.json or model.safetensors in {snapshot}"
    )


def _load_module_weights(
    module: torch.nn.Module,
    module_prefix: str,
    snapshot: Path,
    weight_map: dict[str, str],
    dtype: torch.dtype,
) -> None:
    """Assign one module's tensors from their shard, leaving all other weights unloaded."""
    expected = module.state_dict()
    state: dict[str, torch.Tensor] = {}
    with ExitStack() as stack:
        opened: dict[str, Any] = {}
        missing: list[str] = []
        for local_key in expected:
            full_key = f"{module_prefix}.{local_key}" if module_prefix else local_key
            filename = weight_map.get(full_key)
            if filename is None:
                missing.append(full_key)
                continue
            if filename not in opened:
                opened[filename] = stack.enter_context(
                    safe_open(str(snapshot / filename), framework="pt", device="cpu")
                )
            tensor = opened[filename].get_tensor(full_key)
            if tensor.is_floating_point() and tensor.dtype != dtype:
                tensor = tensor.to(dtype=dtype)
            state[local_key] = tensor
        if missing:
            raise KeyError(f"checkpoint is missing required tensors: {', '.join(missing)}")
    module.load_state_dict(state, strict=True, assign=True)


def _attention_masks(
    base_model: torch.nn.Module,
    config: Any,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> dict[str, Any]:
    kwargs = {
        "config": config,
        "inputs_embeds": hidden,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    masks = {"full_attention": create_causal_mask(**kwargs)}
    if getattr(base_model, "has_sliding_layers", False):
        masks["sliding_attention"] = create_sliding_window_causal_mask(**kwargs)
    return masks


def _rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return int(peak if sys.platform == "darwin" else peak * 1024)


def collect(args: argparse.Namespace) -> Path:
    root = Path(__file__).resolve().parent
    started = time.time()
    dtype = getattr(torch, args.dtype)

    config = AutoConfig.from_pretrained(
        args.model, revision=args.revision, local_files_only=True
    )
    tok = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, local_files_only=True
    )
    snapshot, weight_map = _checkpoint_files(args.model, args.revision)
    with init_empty_weights(include_buffers=False):
        causal_lm = AutoModelForCausalLM.from_config(config)
    causal_lm.eval()
    base_model = getattr(causal_lm, "model", None)
    if base_model is None or not hasattr(base_model, "layers") or not hasattr(base_model, "embed_tokens"):
        raise TypeError("streaming currently requires a causal LM with model.layers and model.embed_tokens")
    decoder_layers = base_model.layers
    layer_count = len(decoder_layers)
    if layer_count == 0:
        raise ValueError("model has no decoder layers")
    first_mlp = getattr(decoder_layers[0], "mlp", None)
    down_proj = getattr(first_mlp, "down_proj", None)
    if down_proj is None or not hasattr(down_proj, "in_features"):
        raise TypeError("decoder layer has no MLP down_proj.in_features")
    neuron_count = int(down_proj.in_features)
    del first_mlp, down_proj

    _load_module_weights(
        base_model.embed_tokens,
        "model.embed_tokens",
        snapshot,
        weight_map,
        dtype,
    )
    embedding = base_model.embed_tokens
    device = torch.device("cpu")
    print(
        f"{args.model}: {layer_count} layers x {neuron_count} neurons; "
        f"embedding loaded, decoder weights will stream from {snapshot}",
        flush=True,
    )

    spec = _load_spec(root, args.max_prompts)
    forms, prompt_groups = build_prompts(tok, spec, args.self_name, args.max_prompts)
    prompt_count = sum(len(group) for group in prompt_groups.values())
    print(f"  tokenized {prompt_count} prompts in {len(forms)} form(s)", flush=True)

    # Each family keeps the same [prompt, layer, neuron] float16 representation
    # as collect.py. Intermediate hidden states are retained across the layer pass.
    outputs: dict[str, np.ndarray] = {}
    prompt_by_group: dict[tuple[str, str], list[Prompt]] = prompt_groups
    for (form, family), records in prompt_by_group.items():
        outputs[f"acts_{form}_{family}"] = np.empty(
            (len(records), layer_count, neuron_count), dtype=np.float16
        )

    neutral_path = Path(args.neutral)
    neutral_ids: list[int] = []
    neutral_prompt: Prompt | None = None
    if neutral_path.is_file():
        neutral_text = neutral_path.read_text(encoding="utf-8")
        neutral_ids = [
            int(value)
            for value in tok(neutral_text, add_special_tokens=False)["input_ids"][: args.neutral_tokens]
        ]
        if neutral_ids:
            neutral_prompt = Prompt(
                "neutral", "neutral", 0, neutral_text, neutral_text, neutral_ids, [], []
            )
    neutral_acts = (
        np.empty((len(neutral_ids), layer_count, neuron_count), dtype=np.float16)
        if neutral_prompt is not None
        else None
    )

    # Keep every prompt's hidden state ready for the layer pass. Each prompt
    # remains unpadded, matching collect.py's one-prompt forward and avoiding
    # batch-shape-dependent floating point differences in attention kernels.
    ordered_prompts = [
        prompt
        for form in forms
        for family in ("A", "B", "C", "D", "E_claude", "E_chatgpt")
        for prompt in prompt_groups[(form, family)]
    ]
    if neutral_prompt is not None:
        ordered_prompts.append(neutral_prompt)
    states: list[dict[str, Any]] = []
    device = torch.device("cpu")
    with torch.inference_mode():
        for prompt in ordered_prompts:
            input_ids = torch.tensor([prompt.token_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
            hidden = embedding(input_ids)
            masks = _attention_masks(base_model, config, hidden, attention_mask, position_ids)
            states.append(
                {
                    "prompt": prompt,
                    "hidden": hidden,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "position_embeddings": base_model.rotary_emb(hidden, position_ids),
                    "masks": masks,
                }
            )

        for layer_index in range(layer_count):
            layer = decoder_layers[layer_index]
            _load_module_weights(
                layer,
                f"model.layers.{layer_index}",
                snapshot,
                weight_map,
                dtype,
            )
            captured: dict[str, torch.Tensor] = {}

            def capture_down_input(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
                captured["value"] = inputs[0].detach()

            handle = layer.mlp.down_proj.register_forward_pre_hook(capture_down_input)
            layer_type = getattr(config, "layer_types", ["full_attention"] * layer_count)[layer_index]
            for state in states:
                hidden = state["hidden"]
                layer_output = layer(
                    hidden,
                    attention_mask=state["masks"][layer_type],
                    position_ids=state["position_ids"],
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=state["position_embeddings"],
                )
                activations = captured.pop("value")
                prompt = state["prompt"]
                if prompt.form == "neutral":
                    assert neutral_acts is not None
                    neutral_acts[:, layer_index, :] = (
                        activations[0, : len(neutral_ids), :]
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float16, copy=False)
                    )
                else:
                    span_values = activations[0, prompt.span_tokens, :].float().mean(0)
                    outputs[f"acts_{prompt.form}_{prompt.family}"][prompt.row, layer_index, :] = (
                        span_values.cpu().numpy().astype(np.float16, copy=False)
                    )
                state["hidden"] = layer_output
                del activations, hidden, layer_output
            handle.remove()
            decoder_layers[layer_index] = None
            del layer, handle, captured
            gc.collect()
            print(f"  layer {layer_index + 1}/{layer_count}", flush=True)

    if neutral_acts is not None:
        outputs["neutral"] = neutral_acts.reshape(len(neutral_ids), layer_count * neuron_count)
        outputs["neutral_ids"] = np.array(neutral_ids)
        print(f"  neutral: {len(neutral_ids)} tokens", flush=True)

    os.makedirs(args.out, exist_ok=True)
    slug = args.model.replace("/", "__") + (f"@{args.revision}" if args.revision else "")
    output_path = Path(args.out) / f"{slug}.npz"
    np.savez(output_path, **outputs, layers=layer_count, neurons=neuron_count, forms=np.array(forms))
    print(f"wrote {output_path} in {time.time() - started:.0f}s", flush=True)
    print(f"peak RSS: {_rss_bytes() / (1024 ** 2):.1f} MiB", flush=True)
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--neutral", default=os.path.join(os.path.dirname(__file__), "data", "neutral.txt"))
    parser.add_argument("--neutral-tokens", type=int, default=1024)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--self-name", default=None, help="model's name and maker as NAME:MAKER")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help="weight dtype used for the streamed embedding and decoder layer",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="smoke-run limit: keep the first N frame-indexed examples in each family",
    )
    args = parser.parse_args(argv)
    if args.neutral_tokens < 0:
        parser.error("--neutral-tokens must be nonnegative")
    if args.max_prompts is not None and args.max_prompts < 1:
        parser.error("--max-prompts must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    torch.set_grad_enabled(False)
    collect(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
