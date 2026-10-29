#!/usr/bin/env python3
"""Collect answer-token MLP down-projection activations for Part 93."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAMILIES = ("claude", "gpt", "deepseek", "qwen")
DEFAULT_PROMPTS = HERE / "prompts93.json"
PROJECTION_NAMES = {"down_proj", "dense_4h_to_h"}


def load_prompts(path: Path = DEFAULT_PROMPTS) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("prompts file must contain a JSON array")
    for index, item in enumerate(data, 1):
        if not isinstance(item, dict) or not item.get("id") or not item.get("prompt"):
            raise ValueError(f"invalid prompt at index {index - 1}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{line_no}") from exc
    return rows


def load_answer_items(gen_dir: Path, prompts_path: Path = DEFAULT_PROMPTS) -> list[dict[str, str]]:
    prompts = load_prompts(prompts_path)
    prompt_text = {row["id"]: row["prompt"] for row in prompts}
    prompt_order = {row["id"]: index for index, row in enumerate(prompts)}
    seen: set[tuple[str, str]] = set()
    items = []
    for family in FAMILIES:
        path = gen_dir / f"{family}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing family generation file: {path}")
        latest: dict[str, dict[str, Any]] = {}
        for row in _read_jsonl(path):
            prompt_id = str(row.get("prompt_id") or row.get("id") or "")
            if prompt_id:
                latest[prompt_id] = row
        for prompt_id, row in latest.items():
            if row.get("error") or not isinstance(row.get("answer"), str):
                continue
            if prompt_id not in prompt_text:
                raise ValueError(f"unknown prompt id {prompt_id!r} in {path}")
            key = (family, prompt_id)
            if key in seen:
                raise ValueError(f"duplicate successful answer for {family}/{prompt_id}")
            seen.add(key)
            items.append({
                "family": family,
                "prompt_id": prompt_id,
                "prompt": prompt_text[prompt_id],
                "answer": row["answer"],
            })
    items.sort(key=lambda row: (prompt_order[row["prompt_id"]], FAMILIES.index(row["family"])))
    if not items:
        raise ValueError("no successful family answers found")
    return items


def make_plain_text(item: dict[str, str]) -> tuple[str, int]:
    """Return the required plain frame and the answer boundary before its space."""
    prefix = f"Question: {item['prompt']}\nAnswer:"
    return prefix + " " + item["answer"], len(prefix)


def _as_flat_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("single-text tokenization unexpectedly returned multiple rows")
        value = value[0]
    return list(value)


def tokenize_batch(tokenizer: Any, items: list[dict[str, str]], device: str):
    """Tokenize plain frames, pad manually, and mark exactly the answer suffix."""
    import torch

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs a pad or EOS token for batched inputs")
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = int(tokenizer.pad_token_id)
    texts = []
    answer_starts = []
    token_ids = []
    offsets = []
    has_offsets = bool(getattr(tokenizer, "is_fast", False))
    for item in items:
        text, answer_start = make_plain_text(item)
        texts.append(text)
        answer_starts.append(answer_start)
        if has_offsets:
            encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
            ids = _as_flat_list(encoded["input_ids"])
            spans = _as_flat_list(encoded["offset_mapping"])
            token_ids.append([int(value) for value in ids])
            offsets.append([(int(span[0]), int(span[1])) for span in spans])
        else:
            encoded = tokenizer(text, add_special_tokens=True)
            prefix = f"Question: {item['prompt']}\nAnswer:"
            prefix_encoded = tokenizer(prefix, add_special_tokens=True)
            ids = [int(value) for value in _as_flat_list(encoded["input_ids"])]
            prefix_ids = [int(value) for value in _as_flat_list(prefix_encoded["input_ids"])]
            if ids[:len(prefix_ids)] != prefix_ids:
                raise ValueError("slow tokenizer does not preserve the prefix token boundary")
            token_ids.append(ids)
            offsets.append(None)

    width = max(len(row) for row in token_ids)
    batch_ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(items), width), dtype=torch.long)
    answer_mask = torch.zeros((len(items), width), dtype=torch.bool)
    left_pad = getattr(tokenizer, "padding_side", "right") == "left"
    for row_no, ids in enumerate(token_ids):
        pad_count = width - len(ids)
        start = pad_count if left_pad else 0
        stop = start + len(ids)
        batch_ids[row_no, start:stop] = torch.tensor(ids, dtype=torch.long)
        attention[row_no, start:stop] = 1
        if offsets[row_no] is not None:
            mask_values = [end > answer_starts[row_no] and end > start_char
                           for start_char, end in offsets[row_no]]
            if len(mask_values) != len(ids):
                raise ValueError("token ids and offsets have different lengths")
            answer_mask[row_no, start:stop] = torch.tensor(mask_values, dtype=torch.bool)
        else:
            prefix = f"Question: {items[row_no]['prompt']}\nAnswer:"
            prefix_ids = _as_flat_list(tokenizer(prefix, add_special_tokens=True)["input_ids"])
            answer_mask[row_no, start + len(prefix_ids):stop] = True
    counts = answer_mask.sum(dim=1)
    if bool(torch.any(counts == 0)):
        bad = [items[index]["prompt_id"] for index, count in enumerate(counts.tolist()) if count == 0]
        raise ValueError(f"answer span contains no tokens for prompts {bad[:5]}")
    return {
        "input_ids": batch_ids.to(device),
        "attention_mask": attention.to(device),
        "answer_mask": answer_mask.to(device),
        "texts": texts,
    }


def decoder_layers(model: Any) -> list[Any]:
    """Find the supported decoder block sequence for Qwen, OLMo 2, or GPT-NeoX."""
    candidates = (
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
        ("gpt_neox", "layers"),
    )
    for path in candidates:
        value = model
        for part in path:
            value = getattr(value, part, None)
            if value is None:
                break
        if value is not None and len(value):
            return list(value)
    raise ValueError("could not find Qwen, OLMo 2, or GPT-NeoX decoder layers")


def down_projection(layer: Any, layer_no: int | None = None):
    found = [(name, module) for name, module in layer.named_modules()
             if name.split(".")[-1] in PROJECTION_NAMES and name.split(".")[0] == "mlp"]
    if len(found) != 1:
        label = "layer" if layer_no is None else f"layer {layer_no}"
        raise ValueError(f"expected one named MLP down projection in {label}, found {[name for name, _ in found]}")
    return found[0][1]


def capture_batch(model: Any, projections: list[Any], batch: dict[str, Any]):
    """Capture each MLP projection's mean input over answer token positions."""
    import torch

    captured: dict[int, torch.Tensor] = {}
    seen_widths: dict[int, int] = {}

    def make_hook(layer_no: int):
        def hook(_module, args):
            if not args or not hasattr(args[0], "shape"):
                raise ValueError(f"layer {layer_no} down projection received no tensor input")
            value = args[0]
            if value.ndim == 2 and value.shape[0] == batch["answer_mask"].numel():
                value = value.reshape(*batch["answer_mask"].shape, value.shape[-1])
            if value.ndim != 3 or tuple(value.shape[:2]) != tuple(batch["answer_mask"].shape):
                raise ValueError(f"unexpected layer {layer_no} down-projection input shape {tuple(value.shape)}")
            seen_widths[layer_no] = int(value.shape[-1])
            selected = batch["answer_mask"].to(device=value.device, dtype=torch.float32)
            sums = (value.detach().float() * selected.unsqueeze(-1)).sum(dim=1)
            counts = selected.sum(dim=1).clamp_min(1).unsqueeze(-1)
            captured[layer_no] = (sums / counts).detach()
            return None
        return hook

    handles = [projection.register_forward_pre_hook(make_hook(index))
               for index, projection in enumerate(projections)]
    try:
        with torch.inference_mode():
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(projections):
        raise RuntimeError(f"captured {len(captured)} MLPs, expected {len(projections)}")
    return {index: captured[index].float().cpu().numpy() for index in captured}, seen_widths


def collect(model: Any, tokenizer: Any, items: list[dict[str, str]], out_dir: Path,
            device: str = "xpu", batch_size: int = 4) -> list[dict[str, str]]:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    model.eval()
    model.requires_grad_(False)
    layers = decoder_layers(model)
    projections = [down_projection(layer, index) for index, layer in enumerate(layers)]
    widths = [int(projection.in_features) for projection in projections]
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = [np.lib.format.open_memmap(
        out_dir / f"layer_{index}.npy", mode="w+", dtype=np.float16,
        shape=(len(items), width),
    ) for index, width in enumerate(widths)]

    for begin in range(0, len(items), batch_size):
        current = items[begin:begin + batch_size]
        batch = tokenize_batch(tokenizer, current, device)
        means, actual_widths = capture_batch(model, projections, batch)
        for layer_no, values in means.items():
            if actual_widths[layer_no] != widths[layer_no] or values.shape[1] != widths[layer_no]:
                raise RuntimeError(f"layer {layer_no} captured width {actual_widths[layer_no]}, expected {widths[layer_no]}")
            arrays[layer_no][begin:begin + len(current)] = values.astype(np.float16)
    for array in arrays:
        array.flush()

    rows = [{"family": item["family"], "prompt_id": item["prompt_id"]} for item in items]
    with (out_dir / "rows.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return rows


def load_local_model(model_dir: Path, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "xpu" and (not hasattr(torch, "xpu") or not torch.xpu.is_available()):
        raise RuntimeError("XPU was requested but torch.xpu is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=True, torch_dtype="auto",
    ).to(device)
    return tokenizer, model.eval()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--gen", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)

    items = load_answer_items(args.gen, args.prompts)
    tokenizer, model = load_local_model(args.model, args.device)
    rows = collect(model, tokenizer, items, args.out, args.device, args.batch_size)
    print(f"wrote {len(rows)} answer rows across {len(decoder_layers(model))} layers to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
