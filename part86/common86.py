#!/usr/bin/env python3
"""Shared mechanics for Part 86's screening and verification runs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DOSE = 0.8
SCREEN_SEED = 86000
VERIFY_SEED = 86500
RANDOM_VERIFY_SEED = 86900
SCREEN_QUESTION = "Who are you?"
VERIFY_QUESTIONS = ("Who are you?", "Tell me about yourself.", "What are you?")


def dependency_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for package in ("torch", "transformers", "numpy"):
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = "not-installed"
    return out


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_model(device: str):
    import neuron_report as nr
    tok, model = nr.load_model(MODEL_ID, REVISION, device)
    model.requires_grad_(False)
    model.eval()
    return tok, model


def find_layers(model):
    import neuron_report as nr
    _, layers, _ = nr.find_text_layers(model)
    return layers


def build_prompt(tokenizer, question: str) -> str:
    import neuron_report as nr
    return nr.build_prompt(tokenizer, question, False)


def residual_median(model, tokenizer, layer, calibration_text: str, device: str) -> float:
    """Part 84 rule: median layer-output L2 norm at token positions 1 onward."""
    import torch
    seen: list[float] = []

    def capture(_module, _inputs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if hidden.ndim != 3 or hidden.shape[1] < 2:
            raise ValueError("calibration forward must contain at least two token positions")
        norms = torch.linalg.vector_norm(hidden[0, 1:].detach().float(), dim=-1)
        seen.append(float(norms.median().item()))

    handle = layer.register_forward_hook(capture)
    try:
        encoded = tokenizer(calibration_text, return_tensors="pt")
        ids = (encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids).to(device)
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False)
    finally:
        handle.remove()
    if len(seen) != 1:
        raise RuntimeError(f"expected one residual capture, got {len(seen)}")
    return seen[0]


def scaled_random_directions(weight, count: int, seed: int, target_norm: float):
    """Return seeded unit directions scaled so each W_down @ v has target_norm."""
    import torch
    width = int(weight.shape[1])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    raw = torch.randn((count, width), generator=generator, dtype=torch.float32)
    raw = raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    w = weight.detach().float()
    raw_device = raw.to(device=w.device)
    writes = raw_device @ w.T
    write_norms = torch.linalg.vector_norm(writes, dim=1)
    if bool(torch.any(write_norms <= 0)):
        raise ValueError("cannot scale a random direction with zero residual write norm")
    scaled = raw_device * (float(target_norm) / write_norms).unsqueeze(1)
    return scaled


class AddRowsHook:
    """Down-projection pre-hook with one dense input-space push per batch row.

    The first down_proj call in each generate() is the prompt pass and is left
    untouched. Later calls must be one-token decode steps; each row receives
    only its own vector.
    """
    def __init__(self, down_proj, row_pushes):
        import torch
        self.torch = torch
        self.row_pushes = row_pushes.detach()
        if self.row_pushes.ndim != 2:
            raise ValueError("row_pushes must have shape [batch, down_proj input width]")
        self._prefill_pending = True
        self.counts = {
            "prefill_calls": 0,
            "prefill_positions": 0,
            "generated_calls": 0,
            "generated_positions": 0,
            "per_row_generated_calls": [0] * int(self.row_pushes.shape[0]),
        }
        self.handle = down_proj.register_forward_pre_hook(self._hook)

    def begin_sequence(self):
        self._prefill_pending = True

    def _hook(self, _module, args):
        if not args or not hasattr(args[0], "shape"):
            raise ValueError("down_proj hook expected its input tensor as the first argument")
        x = args[0]
        if x.ndim != 3 or x.shape[0] != self.row_pushes.shape[0] or x.shape[-1] != self.row_pushes.shape[1]:
            raise ValueError(f"unexpected down_proj input shape {tuple(x.shape)}")
        if self._prefill_pending:
            self._prefill_pending = False
            self.counts["prefill_calls"] += 1
            self.counts["prefill_positions"] += int(x.shape[-2])
            return None
        if x.shape[-2] != 1:
            raise RuntimeError("generated-token hook saw more than one position; refusing to steer prompt tokens")
        self.counts["generated_calls"] += 1
        self.counts["generated_positions"] += int(x.shape[-2])
        for row in self.counts["per_row_generated_calls"]:
            # The list is mutated below by index; this loop only validates its shape.
            if row < 0:
                raise RuntimeError("invalid generated call counter")
        self.counts["per_row_generated_calls"] = [v + 1 for v in self.counts["per_row_generated_calls"]]
        pushed = x + self.row_pushes.to(device=x.device, dtype=x.dtype).unsqueeze(1)
        return (pushed,) + tuple(args[1:])

    def remove(self):
        self.handle.remove()


def token_inputs(tokenizer, question: str, batch_size: int, device: str):
    import torch
    prompt = build_prompt(tokenizer, question)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids.shape[0] != 1:
        raise ValueError("expected one rendered prompt before row expansion")
    return {"input_ids": ids.to(device).expand(batch_size, -1).contiguous()}


def eos_ids(tokenizer) -> list[int]:
    found = []
    for value in (getattr(tokenizer, "eos_token_id", None), tokenizer.convert_tokens_to_ids("<|im_end|>") if hasattr(tokenizer, "convert_tokens_to_ids") else None):
        if isinstance(value, int) and value >= 0 and value not in found:
            found.append(value)
    return found


def generate_batch(model, tokenizer, layer, question: str, pushes, device: str, max_new_tokens: int,
                   do_sample: bool = False, temperature: float | None = None,
                   top_p: float | None = None, seed: int | None = None):
    """Generate a batch with Part 86's per-row generated-token hook."""
    import torch
    batch = int(pushes.shape[0])
    inputs = token_inputs(tokenizer, question, batch, device)
    prompt_len = int(inputs["input_ids"].shape[-1])
    hook = AddRowsHook(layer.mlp.down_proj, pushes)
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.manual_seed_all(int(seed))
    eos = eos_ids(tokenizer)
    kwargs = {
        **inputs,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "use_cache": True,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }
    if eos:
        kwargs["eos_token_id"] = eos
    if do_sample:
        kwargs["temperature"] = float(temperature)
        kwargs["top_p"] = float(top_p)
    hook.begin_sequence()
    try:
        with torch.inference_mode():
            sequences = model.generate(**kwargs)
    finally:
        hook.remove()
    generated = sequences[:, prompt_len:]
    out = []
    stop_ids = set(eos)
    for row in range(batch):
        ids = generated[row].tolist()
        if stop_ids:
            for pos, token_id in enumerate(ids):
                if int(token_id) in stop_ids:
                    ids = ids[:pos + 1]
                    break
        out.append({
            "answer": tokenizer.decode(ids, skip_special_tokens=True).strip(),
            "token_count": len(ids),
        })
    for row, result in enumerate(out):
        expected = max(0, result["token_count"] - 1)
        fired = hook.counts["per_row_generated_calls"][row]
        if fired < expected:
            raise RuntimeError(f"hook missed generated steps for row {row}: {fired} fires, {expected} required")
    return out, hook.counts
