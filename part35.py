#!/usr/bin/env python3
"""Part 35: screen and perturb one-unit MLP concept candidates across local models.

The only model weights this program reads are pinned local cache snapshots. Test code
uses small random models constructed from cached configs and never downloads weights.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "part35-concepts.json"
DEFAULT_CACHE = Path("/mnt/e/neuron-probe/hf/hub")
PERMUTATIONS = 200
PERMUTATION_SEED = 3501
SPLIT_SEED = 3501
MODELS = {
    "Qwen/Qwen3-1.7B": {
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "family": "qwen3", "repo_dir": "models--Qwen--Qwen3-1.7B",
    },
    # The brief requires this checkpoint, but it is absent from the inspected local cache.
    "Qwen/Qwen3-4B": {
        "revision": None, "family": "qwen3", "repo_dir": "models--Qwen--Qwen3-4B",
    },
    "Qwen/Qwen3.5-4B": {
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "family": "qwen3", "repo_dir": "models--Qwen--Qwen3.5-4B",
    },
    "HuggingFaceTB/SmolLM3-3B": {
        "revision": "a07cc9a04f16550a088caea529712d1d335b0ac1",
        "family": "smollm3", "repo_dir": "models--HuggingFaceTB--SmolLM3-3B",
    },
    "allenai/OLMo-2-0425-1B-Instruct": {
        "revision": "48d788eca847d4d7548f375ad03d3c9312f6139e",
        "family": "olmo2", "repo_dir": "models--allenai--OLMo-2-0425-1B-Instruct",
    },
}
HELD_OUT = {"spider", "candle", "teapot", "train", "cactus", "snowman", "raven", "whale"}
MODEL_FAMILY = {name: spec["family"] for name, spec in MODELS.items()}


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(temp, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temp, path)


def jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()


def replace_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(temp, "w", encoding="utf-8") as stream:
        seen = set()
        for row in rows:
            key = row.get("row_key")
            if key is not None and key in seen:
                continue
            seen.add(key)
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temp, path)


def completed_keys(path: str | Path, key: str = "row_key") -> set[str]:
    return {str(row[key]) for row in jsonl_rows(path) if row.get(key) is not None}


def pending_rows(rows: Iterable[dict[str, Any]], done: set[str], key: str = "row_key") -> list[dict[str, Any]]:
    """Return unseen JSONL keys in input order; duplicate keys in the input are emitted once."""
    seen = set(done)
    result = []
    for row in rows:
        value = str(row[key])
        if value not in seen:
            result.append(row)
            seen.add(value)
    return result


def dose_level(rho: float, residual_rms: float, down_column_norm: float) -> float:
    """Prime's vector-matched clamp level: L ||W_down[:,n]|| = rho ||h_l||."""
    if rho <= 0 or residual_rms <= 0 or down_column_norm <= 0:
        raise ValueError("rho, residual RMS, and down-projection column norm must be positive")
    return float(rho * residual_rms / down_column_norm)


def natural_max_level(multiplier: float, neutral_natural_max: float) -> float:
    if multiplier <= 0 or neutral_natural_max < 0:
        raise ValueError("multiplier must be positive and neutral maximum nonnegative")
    return float(multiplier * neutral_natural_max)


def clamp_selected_positions(values: np.ndarray, unit: int, level: float,
                             generated_mask: np.ndarray) -> np.ndarray:
    """Array reference implementation of the generated-token-only down-proj input clamp."""
    source = np.asarray(values)
    mask = np.asarray(generated_mask, dtype=bool)
    if source.ndim < 2 or source.shape[-1] <= unit or mask.shape != source.shape[:-1]:
        raise ValueError("generated mask must match all token dimensions before the unit axis")
    out = source.copy()
    out[..., unit] = np.where(mask, level, out[..., unit])
    return out


def frame_contrast(target: np.ndarray, controls: np.ndarray, spread: np.ndarray,
                   epsilon: float = 1e-8) -> np.ndarray:
    """Median target minus median matched-frame controls, normalized by neutral spread."""
    target = np.asarray(target, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    spread = np.asarray(spread, dtype=np.float64)
    if controls.ndim < target.ndim + 1 or target.shape[-1] != spread.shape[-1]:
        raise ValueError("controls need one extra comparison axis and spread must match units")
    denominator = np.maximum(spread, epsilon)
    return (np.median(target, axis=-2) - np.median(controls, axis=(-3, -2))) / denominator


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0 or successes < 0 or successes > n:
        raise ValueError("Wilson interval requires 0 <= successes <= n and n > 0")
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - margin), min(1.0, center + margin)


def split_signature(data: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    return {concept["id"]: tuple(p["split"] for p in concept["passages"])
            for concept in data["concepts"]}


def trump_name_removed(text: str) -> str:
    """Remove names and common Truth Social handles without preserving post text elsewhere."""
    value = re.sub(r"(?<!\w)@(?:realDonaldTrump|DonaldTrump|POTUS)(?!\w)", " ", text, flags=re.I)
    value = re.sub(r"(?<!\w)Donald\s+(?:J\.?\s+)?Trump(?!\w)", " ", value, flags=re.I)
    value = re.sub(r"(?<!\w)(?:President\s+)?Trump(?!\w)", " ", value, flags=re.I)
    value = re.sub(r"https?://\S+", " ", value, flags=re.I)
    return re.sub(r"\s+", " ", value).strip()


def read_trump_posts(path: str | Path) -> list[dict[str, str]]:
    """Read JSON or Markdown archive; return only redacted text and ordinal ids."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(raw)
        if isinstance(value, dict):
            value = value.get("posts", [])
        if not isinstance(value, list):
            raise ValueError("Trump JSON archive must be a list or an object with a posts list")
        texts = []
        for row in value:
            if isinstance(row, str):
                texts.append(row)
            elif isinstance(row, dict):
                texts.append(str(row.get("text", row.get("content", row.get("body", "")))))
    else:
        texts = []
        for line in raw.splitlines():
            line = re.sub(r"^\s*(?:[-*+]\s+|#{1,6}\s*)", "", line).strip()
            if line and not re.match(r"^(?:https?://|url\s*:|date\s*:|source\s*:)", line, re.I):
                texts.append(line)
    result = []
    for index, text in enumerate(texts, 1):
        clean = trump_name_removed(text)
        if clean:
            result.append({"id": f"trump-after-{index:04d}", "text": clean})
    return result


def experiment_samples(data: dict[str, Any], posts: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    rows = []
    for concept in data["concepts"]:
        for passage in concept["passages"]:
            rows.append({"id": f"concept:{concept['id']}:{passage['frame_id']}",
                         "concept_id": concept["id"], "category": concept["category"],
                         "held_out": concept["held_out"], "frame_id": passage["frame_id"],
                         "language": passage["language"], "kind": passage["kind"],
                         "split": passage["split"], "text": passage["text"], "role": "concept"})
    control = data["positive_control"]
    for passage in control["passages"]:
        rows.append({"id": f"positive:{control['id']}:{passage['frame_id']}",
                     "concept_id": control["id"], "category": "positive_control", "held_out": False,
                     "frame_id": passage["frame_id"], "language": passage["language"],
                     "kind": passage["kind"], "split": passage["split"],
                     "text": passage["text"], "role": "positive_control"})
    neutral = {"en": "an everyday topic", "es": "un tema cotidiano", "zh": "日常话题",
               "de": "ein alltägliches Thema", "ru": "обычная тема"}
    for frame in data["frame_bank"]:
        rows.append({"id": f"neutral:{frame['id']}", "concept_id": "__neutral__",
                     "category": "neutral_reference", "held_out": False,
                     "frame_id": frame["id"], "language": frame["language"],
                     "kind": "neutral", "split": "reference",
                     "text": frame["template"].format(slot=neutral[frame["language"]]),
                     "role": "neutral"})
    for style, passages in data["trump_style_controls"].items():
        for index, text in enumerate(passages, 1):
            rows.append({"id": f"style:{style}:{index:02d}", "concept_id": "__style_control__",
                         "category": style, "held_out": False, "frame_id": f"trump-style-{index:02d}",
                         "language": "en", "kind": "trump_style_control", "split": "reference",
                         "text": text, "role": "style_control", "style_group": style})
    for group in ("ai", "human"):
        for example in data["self_direction_examples"][group]:
            rows.append({"id": f"self:{group}:{example['id']}", "concept_id": "__self_direction__",
                         "category": group, "held_out": example["split"] == "heldout",
                         "frame_id": example["id"], "language": "en", "kind": "self_direction",
                         "split": example["split"], "text": example["text"], "role": "self_direction",
                         "self_class": group})
    for row in posts or []:
        rows.append({"id": row["id"], "concept_id": "__trump_after__", "category": "trump_after_cutoff",
                     "held_out": False, "frame_id": row["id"], "language": "en",
                     "kind": "trump_after_cutoff", "split": "external", "text": row["text"],
                     "role": "trump_after_cutoff"})
    return rows


def model_slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", model_id).strip("-").lower()


def model_spec(model_id: str, revision: str | None = None) -> dict[str, Any]:
    if model_id not in MODELS:
        raise ValueError(f"unknown model {model_id!r}; choose one of {', '.join(MODELS)}")
    result = dict(MODELS[model_id])
    result["model_id"] = model_id
    if revision:
        result["revision"] = revision
    return result


def resolve_snapshot(cache_dir: str | Path, model_id: str, revision: str | None = None) -> Path:
    spec = model_spec(model_id, revision)
    if not spec["revision"]:
        raise FileNotFoundError(f"no pinned local revision is recorded for {model_id}; inspect/cache a snapshot first")
    path = Path(cache_dir) / spec["repo_dir"] / "snapshots" / spec["revision"]
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"pinned local snapshot is missing: {path}")
    return path


def _torch_runtime():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("screen/push require locally installed torch and transformers; no model weights were downloaded") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def find_text_layers(model: Any, torch: Any) -> tuple[str, Any]:
    best = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module):
            first = module[0]
            if getattr(getattr(first, "mlp", None), "down_proj", None) is not None:
                key = ("language" in name.lower(), len(module))
                if best is None or key > best[0]:
                    best = (key, name, module)
    if best is None:
        raise RuntimeError("could not locate text decoder blocks with mlp.down_proj")
    return best[1], best[2]


def load_model(model_id: str, revision: str | None, cache_dir: str | Path,
               device: str, dtype_name: str):
    torch, AutoModelForCausalLM, AutoTokenizer = _torch_runtime()
    snapshot = resolve_snapshot(cache_dir, model_id, revision)
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
    if device == "xpu" and not getattr(torch, "xpu", None):
        raise RuntimeError("this torch build has no XPU support")
    if device == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("--device xpu requested but no XPU is available")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(snapshot), torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True)
    except (ValueError, KeyError, TypeError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            str(snapshot), torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True)
    model.to(device).eval()
    torch.set_grad_enabled(False)
    return torch, tokenizer, model, snapshot


def chat_text(tokenizer: Any, text: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("a pinned tokenizer has no chat template; refusing to invent one")
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    template = str(tokenizer.chat_template)
    if "enable_thinking" in template:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], **kwargs)


class ActivationCapture:
    """Capture the max over passage tokens for every down_proj input unit."""
    def __init__(self, torch: Any, layers: Any, capture_residual_vectors: bool = False):
        self.torch = torch
        self.layers = layers
        self.maxima: dict[int, np.ndarray] = {}
        self.residual_norms: dict[int, list[np.ndarray]] = defaultdict(list)
        self.capture_residual_vectors = capture_residual_vectors
        self.residual_vectors: dict[int, np.ndarray] = {}
        self.handles = []
        for index, layer in enumerate(layers):
            down = layer.mlp.down_proj
            self.handles.append(down.register_forward_pre_hook(self._capture(index)))
            self.handles.append(layer.register_forward_hook(self._residual(index)))

    def _capture(self, index: int):
        def hook(_module, args):
            x = args[0]
            maxima = x.detach().float().amax(dim=-2)[0].cpu().numpy().astype(np.float32, copy=True)
            self.maxima[index] = maxima
        return hook

    def _residual(self, index: int):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            norms = self.torch.linalg.vector_norm(hidden.detach().float(), dim=-1)[0].cpu().numpy()
            self.residual_norms[index].append(norms.astype(np.float32, copy=False))
            if self.capture_residual_vectors:
                self.residual_vectors[index] = hidden.detach().float()[0].mean(dim=0).cpu().numpy().astype(np.float32, copy=True)
        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class UnitClampHook:
    """Clamp one down_proj input unit on generated tokens and audit natural exceedances."""
    def __init__(self, torch: Any, down_proj: Any, unit: int | None = None,
                 level: float | None = None, generated_only: bool = True, write_back: bool = False):
        self.torch = torch
        self.unit, self.level = unit, level
        self.generated_only, self.write_back = generated_only, write_back
        self.natural: list[np.ndarray] = []
        self.total = 0
        self.exceeded = 0
        self.handle = down_proj.register_forward_pre_hook(self._hook)

    def _hook(self, _module, args):
        x = args[0]
        if self.unit is None:
            return None
        active = not self.generated_only or x.shape[-2] == 1
        if not active:
            return None
        original = x[..., self.unit].detach().float()
        natural = original.cpu().numpy()
        self.natural.append(natural)
        if self.level is not None:
            self.total += int(original.numel())
            self.exceeded += int((original > self.level).sum().item())
        if self.level is None and not self.write_back:
            return None
        changed = x.clone()
        if self.level is None:  # Identity hook: write back the exact original value.
            changed[..., self.unit] = x[..., self.unit]
        else:
            changed[..., self.unit] = self.level
        return (changed,) + tuple(args[1:])

    @property
    def exceedance_share(self) -> float | None:
        return self.exceeded / self.total if self.total else None

    def remove(self) -> None:
        self.handle.remove()


class ResidualProjectionHook:
    """Average a unit-normalized direction projection over the first 16 generated-token residuals."""
    def __init__(self, torch: Any, layer: Any, direction: np.ndarray, limit: int = 16):
        self.torch = torch
        self.direction = torch.as_tensor(direction, dtype=torch.float32)
        self.limit = limit
        self.values: list[float] = []
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, _module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[-2] != 1 or len(self.values) >= self.limit:
            return
        residual = hidden.detach().float()[0, -1].cpu()
        direction = self.direction.cpu()
        self.values.append(float(self.torch.dot(residual, direction).item()))

    @property
    def mean(self) -> float | None:
        return float(np.mean(self.values)) if self.values else None

    def remove(self) -> None:
        self.handle.remove()


def tokenize_chat(tokenizer: Any, text: str, device: str):
    torch, _, _ = _torch_runtime()
    rendered = chat_text(tokenizer, text)
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    return {key: value.to(device) for key, value in encoded.items()}


def capture_one(torch: Any, tokenizer: Any, model: Any, layers: Any, text: str,
                device: str, capture_residual_vectors: bool = False) -> tuple[dict[str, np.ndarray], list[float], dict[str, np.ndarray]]:
    encoded = tokenize_chat(tokenizer, text, device)
    capture = ActivationCapture(torch, layers, capture_residual_vectors=capture_residual_vectors)
    try:
        with torch.inference_mode():
            model(**encoded, use_cache=False)
    finally:
        capture.remove()
    if len(capture.maxima) != len(layers):
        raise RuntimeError(f"captured {len(capture.maxima)} MLP layers; expected {len(layers)}")
    residual = []
    for layer_index in range(len(layers)):
        values = capture.residual_norms[layer_index]
        if not values:
            raise RuntimeError(f"no residual values recorded for layer {layer_index}")
        residual.append(float(np.sqrt(np.mean(np.concatenate(values, axis=0) ** 2))))
    residual_vectors = {f"R{i}": capture.residual_vectors[i] for i in range(len(layers))} if capture_residual_vectors else {}
    return {f"L{i}": capture.maxima[i] for i in range(len(layers))}, residual, residual_vectors


def layer_metadata(stack_name: str, layers: Any) -> list[dict[str, Any]]:
    n = len(layers)
    result = []
    for index, layer in enumerate(layers):
        width = int(layer.mlp.down_proj.in_features)
        result.append({"layer": index, "block_path": f"{stack_name}.{index}",
                       "hook_module": f"{stack_name}.{index}.mlp.down_proj",
                       "width": width, "depth_fraction": index / max(1, n - 1)})
    return result


def cache_path_for(sample_id: str, cache_dir: Path) -> Path:
    key = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.npz"


def write_activation_cache(path: Path, activations: dict[str, np.ndarray], residual_rms: list[float],
                           residual_vectors: dict[str, np.ndarray] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + f".{os.getpid()}.tmp.npz")
    arrays = dict(activations)
    arrays["residual_rms"] = np.asarray(residual_rms, dtype=np.float32)
    if residual_vectors:
        arrays.update(residual_vectors)
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


def _model_output_dir(out_dir: str | Path, model_id: str) -> Path:
    return Path(out_dir) / model_slug(model_id)


def _manifest(out_dir: Path) -> dict[str, dict[str, Any]]:
    path = out_dir / "activation-cache.jsonl"
    result = {}
    for row in jsonl_rows(path):
        result[str(row["id"])] = row
    return result


def _save_header(header: dict[str, Any], out_dir: Path) -> None:
    atomic_json(header, out_dir / "header.json")


def make_header(model_id: str, revision: str | None, snapshot: Path,
                device: str, dtype: str, stack_name: str, layers: Any,
                tokenizer: Any, data: dict[str, Any], posts_count: int) -> dict[str, Any]:
    spec = model_spec(model_id, revision)
    template = str(getattr(tokenizer, "chat_template", "") or "")
    return {
        "tool": "part35.py", "model_id": model_id,
        "model_revision_sha": spec["revision"], "tokenizer_revision_sha": spec["revision"],
        "snapshot_path": str(snapshot), "device": device, "dtype": dtype,
        "dtype_reason": "bf16 halves model-weight memory on the 4-core/15-GB CPU target while retaining fp32 exponent range; use --dtype fp32 for comparison",
        "model_family": spec["family"], "layer_count": len(layers),
        "layers": layer_metadata(stack_name, layers),
        "hook_module_paths": [f"{stack_name}.{i}.mlp.down_proj" for i in range(len(layers))],
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "thinking_switch_disabled": "enable_thinking" in template,
        "split_seed": data["split_seed"], "neutral_spread": "population standard deviation over the 36 fixed neutral frame passages, epsilon max(1e-8, 1e-6 * median spread)",
        "trump_posts_present": posts_count > 0, "trump_posts_count": posts_count,
        "trump_post_name_redaction": "trump_name_removed-v1; raw post text is never written to cache or results",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def selected_concept_ids(data: dict[str, Any], start: int | None, end: int | None) -> set[str]:
    values = [c["id"] for c in data["concepts"]] + [data["positive_control"]["id"]]
    lo = 0 if start is None else start
    hi = len(values) if end is None else end
    if lo < 0 or hi > len(values) or lo >= hi:
        raise ValueError(f"invalid concept range [{lo},{hi}); valid range is [0,{len(values)})")
    return set(values[lo:hi])


def _capture_filter(samples: list[dict[str, Any]], selected: set[str], include_shared: bool) -> list[dict[str, Any]]:
    result = []
    for row in samples:
        if row["role"] in {"concept", "positive_control"}:
            if row["concept_id"] in selected:
                result.append(row)
        elif include_shared:
            result.append(row)
    return result


def run_capture(args: argparse.Namespace, data: dict[str, Any], model_dir: Path,
                samples: list[dict[str, Any]], selected: set[str], posts_count: int) -> None:
    torch, tokenizer, model, snapshot = load_model(args.model, args.revision, args.cache_dir,
                                                    args.device, args.dtype)
    stack_name, layers = find_text_layers(model, torch)
    header = make_header(args.model, args.revision, snapshot, args.device, args.dtype,
                         stack_name, layers, tokenizer, data, posts_count)
    _save_header(header, model_dir)
    cache_dir = model_dir / "activations"
    manifest_path = model_dir / "activation-cache.jsonl"
    done = _manifest(model_dir)
    rows = _capture_filter(samples, selected, include_shared=(args.concept_start in (None, 0)))
    n_new = 0
    for index, sample in enumerate(rows, 1):
        sample_id = sample["id"]
        if sample_id in done and (model_dir / done[sample_id]["file"]).is_file():
            continue
        activations, residual_rms, residual_vectors = capture_one(
            torch, tokenizer, model, layers, sample["text"], args.device,
            capture_residual_vectors=sample["role"] == "self_direction")
        target = cache_path_for(sample_id, cache_dir)
        write_activation_cache(target, activations, residual_rms, residual_vectors)
        relative = str(target.relative_to(model_dir))
        append_jsonl(manifest_path, {"id": sample_id, "file": relative,
                                     "concept_id": sample.get("concept_id"),
                                     "frame_id": sample.get("frame_id"),
                                     "role": sample["role"]})
        done[sample_id] = {"file": relative}
        n_new += 1
        if n_new % 20 == 0:
            print(f"{args.model}: captured {n_new} new rows (processed row {index}/{len(rows)})", flush=True)
    print(f"{args.model}: capture complete; {n_new} new rows, {len(done)} cached rows", flush=True)
    del model, tokenizer
    if args.device == "xpu":
        torch.xpu.empty_cache()


def _read_layer(cache_dir: Path, manifest: dict[str, dict[str, Any]], sample_id: str,
                layer: int) -> np.ndarray:
    row = manifest.get(sample_id)
    if not row:
        raise FileNotFoundError(f"activation cache lacks required row {sample_id}")
    path = cache_dir.parent / row["file"]  # manifest paths are relative to the model dir
    with np.load(path) as payload:
        return np.asarray(payload[f"L{layer}"], dtype=np.float32)


def _read_residual(cache_dir: Path, manifest: dict[str, dict[str, Any]], sample_id: str,
                   layer: int) -> np.ndarray:
    row = manifest.get(sample_id)
    if not row:
        raise FileNotFoundError(f"activation cache lacks required residual row {sample_id}")
    with np.load(cache_dir.parent / row["file"]) as payload:  # manifest paths are relative to the model dir
        return np.asarray(payload[f"R{layer}"], dtype=np.float32)


def build_self_direction(model_dir: Path, header: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Fit layerwise AI-self minus human residual directions on 24+24 train examples."""
    cache_dir = model_dir / "activations"
    manifest = _manifest(model_dir)
    examples = data["self_direction_examples"]
    arrays: dict[str, np.ndarray] = {}
    layer_summaries = []
    for layer_meta in header["layers"]:
        layer = int(layer_meta["layer"])
        train_ai = np.stack([_read_residual(cache_dir, manifest, f"self:ai:{row['id']}", layer)
                             for row in examples["ai"] if row["split"] == "train"])
        train_human = np.stack([_read_residual(cache_dir, manifest, f"self:human:{row['id']}", layer)
                                for row in examples["human"] if row["split"] == "train"])
        held_ai = np.stack([_read_residual(cache_dir, manifest, f"self:ai:{row['id']}", layer)
                            for row in examples["ai"] if row["split"] == "heldout"])
        held_human = np.stack([_read_residual(cache_dir, manifest, f"self:human:{row['id']}", layer)
                               for row in examples["human"] if row["split"] == "heldout"])
        difference = train_ai.mean(axis=0) - train_human.mean(axis=0)
        norm = float(np.linalg.norm(difference))
        if norm == 0:
            raise RuntimeError(f"AI-self direction has zero norm at layer {layer}")
        direction = difference / norm
        ai_scores = held_ai @ direction
        human_scores = held_human @ direction
        auc = float((sum(a > h for a in ai_scores for h in human_scores) +
                     .5 * sum(a == h for a in ai_scores for h in human_scores)) /
                    (len(ai_scores) * len(human_scores)))
        arrays[f"direction_L{layer}"] = direction.astype(np.float32)
        arrays[f"mean_difference_L{layer}"] = difference.astype(np.float32)
        layer_summaries.append({"layer": layer, "depth_fraction": layer_meta["depth_fraction"],
                                "n_train_ai": int(len(train_ai)), "n_train_human": int(len(train_human)),
                                "n_heldout_ai": int(len(held_ai)), "n_heldout_human": int(len(held_human)),
                                "direction_norm": norm, "heldout_ai_projection_mean": float(ai_scores.mean()),
                                "heldout_human_projection_mean": float(human_scores.mean()),
                                "heldout_projection_separation": float(ai_scores.mean() - human_scores.mean()),
                                "heldout_projection_separation_ci95": _bootstrap_mean_difference_ci(
                                    ai_scores, human_scores, seed=3501 + layer),
                                "heldout_auc": auc})
    temp = model_dir / f"self-direction.{os.getpid()}.tmp.npz"
    np.savez_compressed(temp, **arrays)
    os.replace(temp, model_dir / "self-direction.npz")
    summary = {"model_id": header["model_id"], "fit": "mean post-block residual over 24 train AI descriptions minus 24 train human descriptions; unit-normalized",
              "heldout": "projection separation and pairwise AUC on 8 held-out AI and 8 held-out human descriptions",
              "layer_summaries": layer_summaries}
    atomic_json(summary, model_dir / "self-direction.json")
    return summary


def _concept_passage_map(data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    concepts = data["concepts"]
    result = {}
    for concept in concepts:
        result[concept["id"]] = {p["frame_id"]: p for p in concept["passages"]}
    return concepts, result


def _matched_stat(values: np.ndarray, neutral_spread: np.ndarray,
                  concepts: list[dict[str, Any]], passage_map: dict[str, dict[str, dict[str, Any]]],
                  subset: str) -> tuple[list[np.ndarray], list[list[int]]]:
    """For each concept: matched-frame median contrast on discovery or confirmation passages."""
    frame_ids = list(next(iter(passage_map.values())))
    frame_index = {frame: i for i, frame in enumerate(frame_ids)}
    stats, frame_indices = [], []
    for ci, concept in enumerate(concepts):
        selected = [frame_index[f] for f, p in passage_map[concept["id"]].items()
                    if p["split"] == subset]
        frame_indices.append(selected)
        target = values[ci, selected, :]
        controls = np.concatenate([values[other, selected, :] for other in range(len(concepts)) if other != ci], axis=0)
        eps = max(1e-8, float(np.median(neutral_spread)) * 1e-6)
        stats.append((np.median(target, axis=0) - np.median(controls, axis=0)) /
                     np.maximum(neutral_spread, eps))
    return stats, frame_indices


def _torch_median_dim0(torch: Any, x: Any) -> Any:
    """numpy.median semantics along dim 0 (mean of the two middle values for an even count)."""
    ordered, _ = torch.sort(x, dim=0)
    n = ordered.shape[0]
    if n % 2:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2


def permutation_maxima(values: np.ndarray, neutral_spread: np.ndarray,
                       frame_indices: list[list[int]], n_permutations: int = PERMUTATIONS,
                       seed: int = PERMUTATION_SEED,
                       target_indices: list[int] | None = None,
                       device: str | None = None) -> np.ndarray:
    """Matched-frame concept-label null; each permutation shuffles labels within every frame.

    With device set (e.g. "xpu" or "cpu"), the medians run in torch on that device: the same label draws from
    the same numpy RNG and the same median definition, so the result matches the numpy path to float rounding.
    The numpy path took about 3.7 s per permutation on Qwen3-1.7B (43 concepts, 36 frames, 6,144 units).
    """
    if n_permutations < 200:
        raise ValueError("the Part 35 permutation null requires at least 200 permutations")
    rng = np.random.default_rng(seed)
    n_concepts, n_frames, n_units = values.shape
    targets = list(range(n_concepts)) if target_indices is None else list(target_indices)
    maxima = np.full(n_permutations, -np.inf, dtype=np.float64)
    eps = max(1e-8, float(np.median(neutral_spread)) * 1e-6)
    denom = np.maximum(neutral_spread, eps)
    if device:
        import torch
        tv = torch.as_tensor(values, dtype=torch.float32, device=device)
        td = torch.as_tensor(denom, dtype=torch.float32, device=device)
        for permutation_i in range(n_permutations):
            label_to_source = np.stack([rng.permutation(n_concepts) for _ in range(n_frames)], axis=0)
            for concept_i in targets:
                frames = np.asarray(frame_indices[concept_i])
                others = [o for o in range(n_concepts) if o != concept_i]
                target = tv[torch.as_tensor(label_to_source[frames, concept_i], device=device),
                            torch.as_tensor(frames, device=device)]
                src = label_to_source[np.ix_(frames, others)].reshape(-1)
                frm = np.repeat(frames, len(others))
                control = tv[torch.as_tensor(src, device=device), torch.as_tensor(frm, device=device)]
                statistic = (_torch_median_dim0(torch, target) - _torch_median_dim0(torch, control)) / td
                maxima[permutation_i] = max(maxima[permutation_i], float(statistic.max().item()))
        return maxima
    for permutation_i in range(n_permutations):
        label_to_source = np.stack([rng.permutation(n_concepts) for _ in range(n_frames)], axis=0)
        for concept_i in targets:
            frames = frame_indices[concept_i]
            target = np.stack([values[label_to_source[f, concept_i], f] for f in frames], axis=0)
            control = np.concatenate([
                values[label_to_source[f, other], f][None, :]
                for f in frames for other in range(n_concepts) if other != concept_i
            ], axis=0)
            statistic = (np.median(target, axis=0) - np.median(control, axis=0)) / denom
            maxima[permutation_i] = max(maxima[permutation_i], float(np.max(statistic)))
    return maxima


def yona_consistency(values: np.ndarray, concept_index: int) -> np.ndarray:
    """Operational cross-prompt consistency proxy: mean within-frame concept percentile."""
    n_concepts, n_frames, n_units = values.shape
    scores = np.zeros(n_units, dtype=np.float64)
    for frame in range(n_frames):
        target = values[concept_index, frame]
        rank = (values[:, frame, :] <= target[None, :]).sum(axis=0) - 1
        scores += rank / max(1, n_concepts - 1)
    return scores / max(1, n_frames)


def score_cached_screen(args: argparse.Namespace, data: dict[str, Any], model_dir: Path) -> list[dict[str, Any]]:
    header = read_json(model_dir / "header.json")
    concepts, passage_map = _concept_passage_map(data)
    model_dir_cache = model_dir / "activations"
    manifest = _manifest(model_dir)
    needed = [f"concept:{c['id']}:{p['frame_id']}" for c in concepts for p in c["passages"]]
    needed += [f"neutral:{f['id']}" for f in data["frame_bank"]]
    for style, passages in data["trump_style_controls"].items():
        needed += [f"style:{style}:{i:02d}" for i in range(1, len(passages) + 1)]
    needed += [f"positive:alien:{p['frame_id']}" for p in data["positive_control"]["passages"]]
    for group in ("ai", "human"):
        needed += [f"self:{group}:{row['id']}" for row in data["self_direction_examples"][group]]
    after_ids = sorted(k for k in manifest if k.startswith("trump-after-"))
    missing = [sample_id for sample_id in needed if sample_id not in manifest]
    if missing:
        raise RuntimeError(f"screen cache is incomplete ({len(missing)} rows missing); finish capture shards before scoring")
    if len(after_ids) != int(header.get("trump_posts_count", 0)):
        raise RuntimeError("after-cutoff post cache does not match the run header; use a consistent --trump-posts archive")
    layers = header["layers"]
    n_concepts, n_frames = len(concepts), len(data["frame_bank"])
    frame_index = {f["id"]: i for i, f in enumerate(data["frame_bank"])}
    metrics: dict[tuple[str, int, int], dict[str, Any]] = {}
    null_max = np.full(args.permutations, -np.inf, dtype=np.float64)
    concept_ids = selected_concept_ids(data, args.concept_start, args.concept_end)
    target_indices = [i for i, concept in enumerate(concepts) if concept["id"] in concept_ids]
    if not args.score_shard:
        target_indices = list(range(len(concepts)))
    alien_fixed_rank = None
    alien_fixed_stat = None
    alien_layer_scores: list[np.ndarray] = []
    alien_model = data["positive_control"]["fixed_neuron"]["model"]
    for layer_meta in layers:
        layer = int(layer_meta["layer"])
        width = int(layer_meta["width"])
        values = np.zeros((n_concepts, n_frames, width), dtype=np.float32)
        neutral_values = np.zeros((n_frames, width), dtype=np.float32)
        for ci, concept in enumerate(concepts):
            for frame_id in passage_map[concept["id"]]:
                values[ci, frame_index[frame_id]] = _read_layer(
                    model_dir_cache, manifest, f"concept:{concept['id']}:{frame_id}", layer)
        for frame in data["frame_bank"]:
            neutral_values[frame_index[frame["id"]]] = _read_layer(
                model_dir_cache, manifest, f"neutral:{frame['id']}", layer)
        neutral_spread = np.std(neutral_values, axis=0, ddof=0)
        neutral_residual = []
        for frame in data["frame_bank"]:
            with np.load(model_dir / manifest[f"neutral:{frame['id']}"]["file"]) as payload:
                neutral_residual.append(float(payload["residual_rms"][layer]))
        layer_residual_rms = float(np.sqrt(np.mean(np.square(neutral_residual))))
        discovery, disc_frames = _matched_stat(values, neutral_spread, concepts, passage_map, "discovery")
        confirmation, _ = _matched_stat(values, neutral_spread, concepts, passage_map, "confirmation")
        layer_null = permutation_maxima(values, neutral_spread, disc_frames,
                                        args.permutations, PERMUTATION_SEED, target_indices,
                                        device=args.device)
        null_max = np.maximum(null_max, layer_null)
        print(f"{header.get('model_id', 'model')}: scored layer {layer} of {len(layers)}", flush=True)
        # Separate secondary views are attached to each candidate, using the same fixed neutral spread.
        indirect_stats, _ = _matched_stat_kind(values, neutral_spread, concepts, passage_map, "indirect")
        non_english_stats, _ = _matched_stat_kind(values, neutral_spread, concepts, passage_map, "non_english")
        for ci, concept in enumerate(concepts):
            if ci not in target_indices:
                continue
            secondary = yona_consistency(values, ci)
            for unit in range(width):
                metrics[(concept["id"], layer, unit)] = {
                    "primary_stat": float(discovery[ci][unit]),
                    "confirmation_stat": float(confirmation[ci][unit]),
                    "indirect_stat": float(indirect_stats[ci][unit]),
                    "non_english_stat": float(non_english_stats[ci][unit]),
                    "yona_consistency": float(secondary[unit]),
                    "neutral_spread": float(neutral_spread[unit]),
                    "neutral_natural_max": float(np.max(neutral_values[:, unit])),
                    "residual_rms": layer_residual_rms,
                }
        if args.model == alien_model and (not args.score_shard or
                any(concepts[i]["id"] == "trump" for i in target_indices)):
            fixed = data["positive_control"]["fixed_neuron"]
            fixed_layer, fixed_unit = int(fixed["layer"]), int(fixed["unit"])
            alien_rows = data["positive_control"]["passages"]
            discovery_rows = [p for p in alien_rows if p["split"] == "discovery"]
            alien_target = np.stack([_read_layer(model_dir_cache, manifest,
                                  f"positive:alien:{p['frame_id']}", layer)
                                     for p in discovery_rows])
            alien_controls = np.concatenate([values[:, frame_index[p["frame_id"]], :]
                                              for p in discovery_rows], axis=0)
            layer_scores = ((np.median(alien_target, axis=0) - np.median(alien_controls, axis=0)) /
                            np.maximum(neutral_spread, 1e-8))
            alien_layer_scores.append(layer_scores)
            if layer == fixed_layer and fixed_unit < width:
                alien_fixed_stat = float(layer_scores[fixed_unit])
    if alien_layer_scores and alien_fixed_stat is not None:
        alien_fixed_rank = int(1 + np.sum(np.concatenate(alien_layer_scores) > alien_fixed_stat))
    rows = []
    permutation_maximum = float(np.max(null_max))
    top_two_by_concept = {}
    for concept in concepts:
        candidates = [(key, value) for key, value in metrics.items() if key[0] == concept["id"]]
        candidates.sort(key=lambda item: (-item[1]["primary_stat"], item[0][1], item[0][2]))
        top_two_by_concept[concept["id"]] = {(key[1], key[2]) for key, _ in candidates[:2]}
    for concept in concepts:
        if concept["id"] not in {concepts[i]["id"] for i in target_indices}:
            continue
        candidates = [(key, value) for key, value in metrics.items() if key[0] == concept["id"]]
        candidates.sort(key=lambda item: (-item[1]["primary_stat"], item[0][1], item[0][2]))
        secondary_order = {key: rank for rank, (key, _v) in enumerate(
            sorted(candidates, key=lambda item: (-item[1]["yona_consistency"], item[0][1], item[0][2])), 1)}
        top = candidates[:5]
        top_pairs = {(key[1], key[2]) for key, _ in top[:2]}
        for rank, (key, metric) in enumerate(top, 1):
            concept_id, layer, unit = key
            p_value = (1 + int(np.sum(null_max >= metric["primary_stat"]))) / (len(null_max) + 1)
            confirm_pass = metric["confirmation_stat"] > permutation_maximum
            row = {"row_key": f"screen:{args.model}:{concept_id}:{rank}",
                   "stage": "screen", "model_id": args.model, "family": header["model_family"],
                   "concept_id": concept_id, "category": concept["category"],
                   "held_out": bool(concept["held_out"]), "layer": layer, "unit": unit,
                   "hook_module": header["layers"][layer]["hook_module"],
                   "depth_fraction": header["layers"][layer]["depth_fraction"], "rank": rank,
                   **metric, "yona_rank": secondary_order[key],
                   "permutation_count": len(null_max), "permutation_maximum": permutation_maximum,
                   "permutation_p": p_value, "screen_pass": metric["primary_stat"] > permutation_maximum,
                   "confirmation_pass": confirm_pass,
                   "shared_top_two": [other_id for other_id, other_pairs in top_two_by_concept.items()
                                      if other_id != concept_id and bool(top_two_by_concept[concept_id] & other_pairs)],
                   "header": "header.json"}
            rows.append(row)
    self_direction = build_self_direction(model_dir, header, data)
    rows.append({"row_key": f"screen:{args.model}:self-direction", "stage": "self_direction_summary",
                 "model_id": args.model, "layer_summaries": self_direction["layer_summaries"],
                 "heldout_ai_n": 8, "heldout_human_n": 8,
                 "fit": self_direction["fit"], "heldout": self_direction["heldout"]})
    trump_top = next((row for row in rows if row.get("stage") == "screen" and
                      row.get("concept_id") == "trump" and row.get("rank") == 1), None)
    style_summary = {"trump_unnamed_to_named_activation_ratio": None,
                     "trump_unnamed_to_named_activation_ratio_ci95": None,
                     "trump_unnamed_n": 0, "trump_named_n": 0,
                     "trump_style_to_matched_control_ratios": {},
                     "trump_after_cutoff_to_pre_cutoff_ratio": None,
                     "trump_after_cutoff_to_pre_cutoff_ratio_ci95": None,
                     "trump_after_cutoff_n": len(after_ids),
                     "trump_after_cutoff_posts_present": bool(after_ids)}
    if trump_top:
        layer, unit = int(trump_top["layer"]), int(trump_top["unit"])
        named = [p for p in data["concepts"] if p["id"] == "trump"][0]["passages"]
        named_values = np.asarray([_read_layer(model_dir_cache, manifest,
                            f"concept:trump:{p['frame_id']}", layer)[unit]
                                   for p in named if p["kind"] == "named"], dtype=float)
        indirect = np.asarray([_read_layer(model_dir_cache, manifest,
                            f"concept:trump:{p['frame_id']}", layer)[unit]
                               for p in named if p["kind"] == "indirect"], dtype=float)
        named_median = float(np.median(named_values))
        style_summary["trump_unnamed_to_named_activation_ratio"] = (
            float(np.median(indirect)) / named_median if named_median else None)
        style_summary["trump_unnamed_to_named_activation_ratio_ci95"] = _bootstrap_ratio_ci(indirect, named_values)
        style_summary["trump_unnamed_n"] = int(len(indirect))
        style_summary["trump_named_n"] = int(len(named_values))
        for style in data["trump_style_controls"]:
            control_values = [_read_layer(model_dir_cache, manifest,
                              f"style:{style}:{i:02d}", layer)[unit]
                              for i in range(1, 9)]
            control_median = float(np.median(control_values))
            style_summary["trump_style_to_matched_control_ratios"][style] = {
                "ratio": float(np.median(indirect)) / control_median if control_median else None,
                "ci95": _bootstrap_ratio_ci(indirect, np.asarray(control_values)),
                "n_control": len(control_values)}
        if after_ids:
            after_values = [_read_layer(model_dir_cache, manifest, sample_id, layer)[unit]
                            for sample_id in after_ids]
            style_summary["trump_after_cutoff_to_pre_cutoff_ratio"] = (
                float(np.median(after_values)) / float(np.median(indirect))
                if float(np.median(indirect)) else None)
            style_summary["trump_after_cutoff_to_pre_cutoff_ratio_ci95"] = _bootstrap_ratio_ci(
                np.asarray(after_values), indirect)
        rows.append({"row_key": f"screen:{args.model}:trump-style-summary",
                     "stage": "trump_style_summary", "model_id": args.model,
                     "concept_id": "trump", "layer": layer, "unit": unit,
                     **style_summary})
    if not args.score_shard or (args.model == alien_model and
            any(concepts[i]["id"] == "trump" for i in target_indices)):
        rows.append({"row_key": f"screen:{args.model}:alien-control", "stage": "positive_control",
                     "model_id": args.model, "concept_id": "alien", "fixed_neuron": data["positive_control"]["fixed_neuron"],
                     "fixed_neuron_rank": alien_fixed_rank, "fixed_neuron_discovery_stat": alien_fixed_stat,
                     "positive_control_only": True, "excluded_from_find_rate": True,
                     "permutation_count": len(null_max), "permutation_maximum": permutation_maximum})
    if args.score_shard:
        shard_dir = model_dir / "screen-shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_name = f"screen-{args.concept_start}-{args.concept_end}.json"
        atomic_json({"model_id": args.model, "concept_start": args.concept_start,
                    "concept_end": args.concept_end, "null_maxima": null_max.tolist(),
                    "rows": rows}, shard_dir / shard_name)
    else:
        _write_jsonl_rows(model_dir / "screen.jsonl", rows)
    print(json.dumps({"model_id": args.model, "top_rows": len(rows) - 1,
                      "permutation_maximum": permutation_maximum,
                      "alien_fixed_neuron_rank": alien_fixed_rank,
                      "trump_after_cutoff_posts_present": bool(after_ids)}, ensure_ascii=False))
    return rows


def merge_screen_shards(model_id: str, out_dir: str | Path) -> list[dict[str, Any]]:
    model_dir = _model_output_dir(out_dir, model_id)
    shard_files = sorted((model_dir / "screen-shards").glob("screen-*.json"))
    if not shard_files:
        raise FileNotFoundError(f"no screen score shards found under {model_dir / 'screen-shards'}")
    shards = [read_json(path) for path in shard_files]
    lengths = {len(row["null_maxima"]) for row in shards}
    if len(lengths) != 1:
        raise ValueError("screen shards use different permutation counts")
    null_max = np.max(np.asarray([row["null_maxima"] for row in shards], dtype=float), axis=0)
    rows = [row for shard in shards for row in shard["rows"]]
    candidate_rows = [row for row in rows if row.get("stage") == "screen"]
    concepts, data = _concept_passage_map(read_json(DATA_PATH))
    top_two = defaultdict(set)
    for row in candidate_rows:
        if row.get("rank") in (1, 2):
            top_two[row["concept_id"]].add((row["layer"], row["unit"]))
    maximum = float(np.max(null_max))
    for row in candidate_rows:
        row["permutation_maximum"] = maximum
        row["permutation_count"] = len(null_max)
        row["permutation_p"] = (1 + int(np.sum(null_max >= row["primary_stat"]))) / (len(null_max) + 1)
        row["screen_pass"] = row["primary_stat"] > maximum
        row["confirmation_pass"] = row["confirmation_stat"] > maximum
        row["shared_top_two"] = [other for other, units in top_two.items()
                                 if other != row["concept_id"] and
                                 bool(top_two.get(row["concept_id"], set()) & units)]
    # De-duplicate control and style summary rows produced once by their designated shard.
    others = []
    seen = set()
    for row in rows:
        if row.get("stage") == "screen":
            continue
        key = row.get("row_key")
        if key not in seen:
            others.append(row)
            seen.add(key)
    merged = candidate_rows + others
    expected = {c["id"] for c in concepts}
    found = {row["concept_id"] for row in candidate_rows}
    if found != expected:
        raise RuntimeError(f"screen shards cover {len(found)} concepts, expected {len(expected)}")
    _write_jsonl_rows(model_dir / "screen.jsonl", merged)
    print(json.dumps({"model_id": model_id, "merged_score_shards": len(shards),
                      "concepts": len(found), "permutation_maximum": maximum,
                      "rows": len(merged)}))
    return merged


def _matched_stat_kind(values: np.ndarray, spread: np.ndarray, concepts: list[dict[str, Any]],
                       passage_map: dict[str, dict[str, dict[str, Any]]], kind: str) -> tuple[list[np.ndarray], list[list[int]]]:
    frame_ids = list(next(iter(passage_map.values())))
    frame_index = {frame: i for i, frame in enumerate(frame_ids)}
    results, used = [], []
    for ci, concept in enumerate(concepts):
        if kind == "indirect":
            selected_frames = [frame_index[f] for f, p in passage_map[concept["id"]].items()
                               if p["kind"] == "indirect" and p["split"] == "discovery"]
        else:
            selected_frames = [frame_index[f] for f, p in passage_map[concept["id"]].items()
                               if p["kind"] == "named" and p["language"] != "en" and p["split"] == "discovery"]
        used.append(selected_frames)
        target = values[ci, selected_frames, :]
        controls = np.concatenate([values[other, selected_frames, :]
                                   for other in range(len(concepts)) if other != ci], axis=0)
        eps = max(1e-8, float(np.median(spread)) * 1e-6)
        results.append((np.median(target, axis=0) - np.median(controls, axis=0)) /
                       np.maximum(spread, eps))
    return results, used


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    done = completed_keys(path)
    for row in rows:
        if row["row_key"] not in done:
            append_jsonl(path, row)
            done.add(row["row_key"])


def run_screen(args: argparse.Namespace) -> list[dict[str, Any]] | None:
    data = read_json(DATA_PATH)
    model_dir = _model_output_dir(args.out, args.model)
    model_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_screen_shards:
        return merge_screen_shards(args.model, args.out)
    posts = read_trump_posts(args.trump_posts) if args.trump_posts else None
    samples = experiment_samples(data, posts)
    selected = selected_concept_ids(data, args.concept_start, args.concept_end)
    ranged = args.concept_start is not None or args.concept_end is not None
    if not args.reuse_cache:
        run_capture(args, data, model_dir, samples, selected, len(posts or []))
    if args.capture_only:
        print(f"{args.model}: capture-only shard complete; use --reuse-cache after all shards")
        return None
    return score_cached_screen(args, data, model_dir)


def _column_norm(layer: Any, unit: int) -> float:
    return float(layer.mlp.down_proj.weight.detach().float()[:, unit].norm().item())


def _neutral_controls(model_dir: Path, header: dict[str, Any], data: dict[str, Any],
                      count: int = 10, seed: int = 3501) -> list[dict[str, Any]]:
    manifest = _manifest(model_dir)
    candidates = []
    for layer_meta in header["layers"]:
        layer = int(layer_meta["layer"])
        width = int(layer_meta["width"])
        neutral = np.stack([_read_layer(model_dir / "activations", manifest,
                                        f"neutral:{frame['id']}", layer)
                            for frame in data["frame_bank"]])
        maxima = neutral.max(axis=0)
        threshold = float(np.quantile(neutral, .99))
        live = np.flatnonzero(maxima > threshold)
        if len(live) == 0:
            live = np.asarray([int(np.argmax(maxima))])
        residual = []
        for frame in data["frame_bank"]:
            with np.load(model_dir / manifest[f"neutral:{frame['id']}"]["file"]) as payload:
                residual.append(float(payload["residual_rms"][layer]))
        residual_rms = float(np.sqrt(np.mean(np.square(residual))))
        candidates.extend({"layer": layer, "unit": int(unit), "live_threshold": threshold,
                           "neutral_natural_max": float(maxima[unit]),
                           "residual_rms": residual_rms}
                          for unit in live)
    rng = random.Random(seed)
    if len(candidates) < count:
        raise RuntimeError(f"only {len(candidates)} live units available; need {count} random controls")
    return rng.sample(candidates, count)


def _primary_doses(layer: Any, unit: int, residual_rms: float, neutral_max: float) -> list[dict[str, Any]]:
    column_norm = _column_norm(layer, unit)
    return [{"dose_kind": "rho", "dose_label": f"rho_{rho:.2f}", "rho": rho,
             "level": dose_level(rho, residual_rms, column_norm),
             "down_column_norm": column_norm, "equivalent_neutral_max_multiple":
             dose_level(rho, residual_rms, column_norm) / neutral_max if neutral_max > 0 else None}
            for rho in (.05, .10)]


def _neutral_max_for_unit(model_dir: Path, header: dict[str, Any], data: dict[str, Any],
                          layer: int, unit: int) -> float:
    manifest = _manifest(model_dir)
    values = [_read_layer(model_dir / "activations", manifest, f"neutral:{frame['id']}", layer)[unit]
              for frame in data["frame_bank"]]
    return float(max(values))


def _residual_rms_for_layer(model_dir: Path, data: dict[str, Any], layer: int) -> float:
    manifest = _manifest(model_dir)
    values = []
    for frame in data["frame_bank"]:
        with np.load(model_dir / manifest[f"neutral:{frame['id']}"]["file"]) as payload:
            values.append(float(payload["residual_rms"][layer]))
    return float(np.sqrt(np.mean(np.square(values))))


def _bootstrap_ratio_ci(numerator: np.ndarray, denominator: np.ndarray,
                        seed: int = 3501, reps: int = 2000) -> list[float] | None:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    if not len(numerator) or not len(denominator) or np.median(denominator) == 0:
        return None
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(reps):
        n = rng.choice(numerator, len(numerator), replace=True)
        d = rng.choice(denominator, len(denominator), replace=True)
        d_median = float(np.median(d))
        if d_median:
            estimates.append(float(np.median(n)) / d_median)
    if not estimates:
        return None
    return [float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))]


def _bootstrap_mean_difference_ci(first: np.ndarray, second: np.ndarray,
                                  seed: int = 3501, reps: int = 4000) -> list[float] | None:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if not len(first) or not len(second):
        return None
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=float)
    for index in range(reps):
        draws[index] = rng.choice(first, len(first), replace=True).mean() - rng.choice(second, len(second), replace=True).mean()
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def _bootstrap_mean_ci(values: np.ndarray, seed: int = 3501, reps: int = 4000) -> list[float] | None:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(reps)])
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def _writeback_logit_change(torch: Any, tokenizer: Any, model: Any, layer: Any,
                            layer_index: int, unit: int, prompt: str, device: str) -> float:
    encoded = tokenize_chat(tokenizer, prompt, device)
    with torch.inference_mode():
        before = model(**encoded, use_cache=False).logits.detach().float()
        hook = UnitClampHook(torch, layer.mlp.down_proj, unit=unit,
                             generated_only=False, write_back=True)
        try:
            after = model(**encoded, use_cache=False).logits.detach().float()
        finally:
            hook.remove()
    return float(torch.max(torch.abs(before - after)).item())


def _sample_answer(torch: Any, tokenizer: Any, model: Any, layer: Any,
                   unit: int | None, level: float | None, prompt: str, device: str,
                   max_new_tokens: int, seed: int,
                   direction: np.ndarray | None = None) -> tuple[str, float | None, int, int, float | None, int]:
    encoded = tokenize_chat(tokenizer, prompt, device)
    torch.manual_seed(seed)
    if device == "xpu":
        torch.xpu.manual_seed_all(seed)
    hook = UnitClampHook(torch, layer.mlp.down_proj, unit=unit, level=level, generated_only=True)
    projection = ResidualProjectionHook(torch, layer, direction) if direction is not None else None
    try:
        with torch.inference_mode():
            output = model.generate(**encoded, do_sample=True, temperature=.7, top_p=.95,
                                    max_new_tokens=max_new_tokens, use_cache=True,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id)
    finally:
        hook.remove()
        if projection is not None:
            projection.remove()
    prompt_len = int(encoded["input_ids"].shape[-1])
    answer = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True).strip()
    return (answer, hook.exceedance_share, hook.exceeded, hook.total,
            projection.mean if projection is not None else None,
            len(projection.values) if projection is not None else 0)


def _baseline_projections(torch: Any, tokenizer: Any, model: Any, layers: Any,
                          directions: dict[int, np.ndarray], prompt: str, device: str,
                          max_new_tokens: int, seed: int) -> dict[int, dict[str, Any]]:
    encoded = tokenize_chat(tokenizer, prompt, device)
    torch.manual_seed(seed)
    if device == "xpu":
        torch.xpu.manual_seed_all(seed)
    hooks = {layer_i: ResidualProjectionHook(torch, layers[layer_i], direction)
             for layer_i, direction in directions.items()}
    try:
        with torch.inference_mode():
            model.generate(**encoded, do_sample=True, temperature=.7, top_p=.95,
                           max_new_tokens=max_new_tokens, use_cache=True,
                           pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id)
    finally:
        for hook in hooks.values():
            hook.remove()
    return {layer_i: {"projection": hook.mean, "tokens": len(hook.values)}
            for layer_i, hook in hooks.items()}


def _primary_candidate_rows(screen_path: Path) -> list[dict[str, Any]]:
    rows = [row for row in jsonl_rows(screen_path)
            if row.get("stage") == "screen" and row.get("rank") == 1]
    rows.sort(key=lambda row: (-float(row.get("primary_stat", 0)), row["concept_id"]))
    selected = []
    seen_units = set()
    for row in rows:
        unit_key = (int(row["layer"]), int(row["unit"]))
        if unit_key not in seen_units:
            selected.append(row)
            seen_units.add(unit_key)
    return selected


def run_push1(args: argparse.Namespace) -> list[dict[str, Any]]:
    data = read_json(DATA_PATH)
    model_dir = _model_output_dir(args.out, args.model)
    screen_path = Path(args.screen) if args.screen else model_dir / "screen.jsonl"
    if not screen_path.is_file():
        raise FileNotFoundError(f"screen output is required: {screen_path}")
    selected_ids = selected_concept_ids(data, args.concept_start, args.concept_end)
    candidates = [row for row in _primary_candidate_rows(screen_path) if row["concept_id"] in selected_ids]
    header = read_json(model_dir / "header.json")
    controls_path = model_dir / "controls.json"
    if controls_path.exists():
        controls = read_json(controls_path)
    else:
        controls = _neutral_controls(model_dir, header, data, count=10)
        atomic_json(controls, controls_path)
    torch, tokenizer, model, _snapshot = load_model(args.model, args.revision, args.cache_dir,
                                                    args.device, args.dtype)
    stack_name, layers = find_text_layers(model, torch)
    if len(layers) != int(header["layer_count"]):
        raise RuntimeError("loaded model layer count differs from screen header")
    output_path = model_dir / "push1.jsonl"
    hit_path = model_dir / "push1-potential-hits.jsonl"
    done = completed_keys(output_path)
    tasks = []
    for row in candidates:
        tasks.append({**row, "role": "candidate"})
    if args.model == data["positive_control"]["fixed_neuron"]["model"] and "alien" in selected_ids:
        fixed = data["positive_control"]["fixed_neuron"]
        tasks.append({"model_id": args.model, "concept_id": "alien", "category": "positive_control",
                      "held_out": False, "layer": fixed["layer"], "unit": fixed["unit"],
                      "rank": None, "screen_pass": False, "confirmation_pass": False,
                      "neutral_natural_max": _neutral_max_for_unit(model_dir, header, data,
                                                       fixed["layer"], fixed["unit"]),
                      "residual_rms": _residual_rms_for_layer(model_dir, data, int(fixed["layer"])),
                      "role": "positive_control"})
    identity = [(f"s1-id-{i+1:02d}", p, "identity")
                for i, p in enumerate(data["stage1_prompts"]["identity"])]
    non_identity = [(f"s1-nonid-{i+1:02d}", p, "non_identity")
                    for i, p in enumerate(data["stage1_prompts"]["non_identity"])]
    prompts = identity + non_identity
    direction_payload = np.load(model_dir / "self-direction.npz")
    directions = {i: np.asarray(direction_payload[f"direction_L{i}"], dtype=np.float32)
                  for i in range(len(layers))}
    baselines = {}
    for prompt_id, prompt_text, _group in identity:
        for sample_index in range(args.samples):
            seed = _stable_seed(f"stage1:{prompt_id}:{sample_index}")
            baselines[(prompt_id, sample_index)] = _baseline_projections(
                torch, tokenizer, model, layers, directions, prompt_text, args.device,
                args.max_new_tokens, seed)
    direction_payload.close()
    all_units = tasks + [{**row, "concept_id": "__control__", "category": "control", "role": "control",
                          "rank": None, "screen_pass": False, "confirmation_pass": False}
                         for row in controls]
    first_target = next((row for row in tasks if row["role"] == "candidate"), None)
    if first_target:
        check_layer = layers[int(first_target["layer"])]
        identity_change = _writeback_logit_change(torch, tokenizer, model, check_layer,
                              int(first_target["layer"]), int(first_target["unit"]),
                              identity[0][1], args.device)
        if identity_change != 0.0:
            raise RuntimeError(f"identity write-back hook changed logits by {identity_change}, expected exactly 0.0")
        atomic_json({"max_absolute_logit_change": identity_change, "exact_zero": True,
                     "layer": first_target["layer"], "unit": first_target["unit"]}, model_dir / "push1-harness-control.json")
    new = 0
    for task in all_units:
        layer_i, unit = int(task["layer"]), int(task["unit"])
        neutral_max = float(task.get("neutral_natural_max", 0.0))
        residual_rms = float(task.get("residual_rms", 0.0))
        doses = _primary_doses(layers[layer_i], unit, residual_rms, neutral_max)
        for dose in doses:
            for prompt_id, prompt_text, prompt_group in prompts:
                for sample_index in range(args.samples):
                    key = f"push1:{task['role']}:{task.get('concept_id')}:{layer_i}:{unit}:{dose['dose_label']}:{prompt_id}:{sample_index}"
                    if key in done:
                        continue
                    gen_seed = _stable_seed(f"stage1:{prompt_id}:{sample_index}")
                    direction = directions[layer_i] if prompt_group == "identity" else None
                    answer, exceeded_share, exceeded, total, projection, projection_tokens = _sample_answer(
                        torch, tokenizer, model, layers[layer_i], unit, dose["level"], prompt_text,
                        args.device, args.max_new_tokens, gen_seed, direction)
                    baseline_projection = (baselines[(prompt_id, sample_index)][layer_i]["projection"]
                                           if prompt_group == "identity" else None)
                    row = {"row_key": key, "stage": "push1", "model_id": args.model,
                           "family": header["model_family"], "role": task["role"],
                           "concept_id": task.get("concept_id"), "category": task.get("category"),
                           "held_out": bool(task.get("held_out", False)), "layer": layer_i,
                           "unit": unit, "rank": task.get("rank"), "screen_pass": task.get("screen_pass"),
                           "confirmation_pass": task.get("confirmation_pass"),
                           "depth_fraction": float(layer_i / max(1, len(layers) - 1)),
                           "dose_kind": dose["dose_kind"], "dose_label": dose["dose_label"],
                           "rho": dose["rho"], "level": dose["level"],
                           "down_column_norm": dose["down_column_norm"],
                           "equivalent_neutral_max_multiple": dose["equivalent_neutral_max_multiple"],
                           "prompt_id": prompt_id, "prompt_group": prompt_group,
                           "sample_index": sample_index, "temperature": .7,
                           "max_new_tokens": args.max_new_tokens, "answer": answer,
                           "clamped_token_natural_above_level_share": exceeded_share,
                           "clamped_tokens_above_level": exceeded, "clamped_tokens_total": total,
                           "self_direction_projection_baseline": baseline_projection,
                           "self_direction_projection_clamped": projection,
                           "self_direction_projection_drop": (baseline_projection - projection
                                                              if baseline_projection is not None and projection is not None else None),
                           "self_direction_projection_tokens": projection_tokens,
                           "judge_status": "pending"}
                    append_jsonl(output_path, row)
                    done.add(key)
                    new += 1
                    if new % 20 == 0:
                        print(f"{args.model}: push1 wrote {new} new answers", flush=True)
    atomic_json({"seed": 3501, "count": len(controls), "live_random_controls": controls,
                 "selection": "random from units whose neutral max exceeds layer-specific neutral q99; sampled across layers"},
                model_dir / "controls.json")
    hit_done = completed_keys(hit_path)
    for task in tasks:
        if task["role"] not in {"candidate", "positive_control"}:
            continue
        key = f"hit:{args.model}:{task.get('concept_id')}:{task['layer']}:{task['unit']}"
        if key not in hit_done:
            append_jsonl(hit_path, {"row_key": key, "model_id": args.model,
                                   "role": task["role"], "concept_id": task.get("concept_id"),
                                   "category": task.get("category"), "held_out": task.get("held_out", False),
                                   "layer": task["layer"], "unit": task["unit"], "rank": task.get("rank"),
                                   "screen_pass": task.get("screen_pass"),
                                   "confirmation_pass": task.get("confirmation_pass"),
                                   "primary_stat": task.get("primary_stat"),
                                   "neutral_natural_max": task.get("neutral_natural_max"),
                                   "residual_rms": task.get("residual_rms"),
                                   "potential_hit": True})
            hit_done.add(key)
    print(json.dumps({"model_id": args.model, "push1_new_answers": new,
                      "potential_hits": len(candidates) + int(args.model == data["positive_control"]["fixed_neuron"]["model"] and "alien" in selected_ids),
                      "push1_path": str(output_path), "hit_list": str(hit_path)}))
    return jsonl_rows(output_path)


def _stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "little")


def _stage2_doses(layer: Any, unit: int, residual_rms: float, neutral_max: float) -> list[dict[str, Any]]:
    doses = _primary_doses(layer, unit, residual_rms, neutral_max)
    column_norm = _column_norm(layer, unit)
    for multiplier in (2, 4, 8):
        level = natural_max_level(multiplier, neutral_max)
        doses.append({"dose_kind": "natural_max", "dose_label": f"natural_{multiplier}x",
                      "natural_max_multiple": multiplier, "level": level,
                      "down_column_norm": column_norm,
                      "equivalent_rho": level * column_norm / residual_rms if residual_rms else None})
    return doses


def run_push2(args: argparse.Namespace) -> list[dict[str, Any]]:
    data = read_json(DATA_PATH)
    model_dir = _model_output_dir(args.out, args.model)
    default_hits = model_dir / "push1-hits.jsonl"
    if not default_hits.is_file():
        default_hits = model_dir / "push1-potential-hits.jsonl"
    hits_path = Path(args.hits) if args.hits else default_hits
    selected_ids = selected_concept_ids(data, args.concept_start, args.concept_end)
    hits_raw = [row for row in jsonl_rows(hits_path) if row.get("role") in {"candidate", "positive_control"}]
    hits_by_unit = {}
    for row in hits_raw:
        if row.get("concept_id") in selected_ids:
            hits_by_unit[(str(row.get("concept_id")), int(row["layer"]), int(row["unit"]))] = row
    hits = list(hits_by_unit.values())
    if not hits:
        print(f"{args.model}: no stage-1 candidates in this concept shard; checking controls only")
    controls_path = Path(args.controls) if args.controls else model_dir / "controls.json"
    controls_obj = read_json(controls_path)
    controls = controls_obj.get("live_random_controls", controls_obj)
    header = read_json(model_dir / "header.json")
    torch, tokenizer, model, _snapshot = load_model(args.model, args.revision, args.cache_dir,
                                                    args.device, args.dtype)
    _stack_name, layers = find_text_layers(model, torch)
    output_path = model_dir / "push2.jsonl"
    done = completed_keys(output_path)
    identity = [(f"s2-id-{i+1:02d}", p, "identity")
                for i, p in enumerate(data["stage2_prompts"]["identity"])]
    non_identity = [(f"s2-nonid-{i+1:02d}", p, "non_identity")
                    for i, p in enumerate(data["stage2_prompts"]["non_identity"])]
    trump_style = [(f"s2-trump-style-{i+1:02d}", p, "trump_style")
                   for i, p in enumerate(data["stage2_prompts"]["trump_style"])]
    direction_payload = np.load(model_dir / "self-direction.npz")
    directions = {i: np.asarray(direction_payload[f"direction_L{i}"], dtype=np.float32)
                  for i in range(len(layers))}
    baselines = {}
    baseline_count = 0
    for prompt_id, prompt_text, _group in identity:
        for sample_index in range(args.samples):
            seed = _stable_seed(f"stage2:{prompt_id}:{sample_index}")
            baselines[(prompt_id, sample_index)] = _baseline_projections(
                torch, tokenizer, model, layers, directions, prompt_text, args.device,
                args.max_new_tokens, seed)
            baseline_count += 1
            if baseline_count % 20 == 0:
                print(f"{args.model}: push2 captured {baseline_count} unclamped identity baselines", flush=True)
    direction_payload.close()
    tasks = []
    if not args.controls_only:
        for hit in hits:
            tasks.append({**hit, "role": hit.get("role", "candidate")})
    control_start = 0 if args.control_start is None else args.control_start
    control_end = len(controls) if args.control_end is None else args.control_end
    if not (0 <= control_start <= control_end <= len(controls)):
        raise ValueError(f"invalid control range [{control_start},{control_end}); {len(controls)} controls available")
    control_subset = controls[control_start:control_end]
    if not args.skip_controls:
        for control in control_subset:
            tasks.append({**control, "concept_id": "__control__", "category": "control",
                          "role": "control", "rank": None, "screen_pass": False,
                          "confirmation_pass": False})
    new = 0
    for task in tasks:
        layer_i, unit = int(task["layer"]), int(task["unit"])
        neutral_max = float(task.get("neutral_natural_max") or
                            _neutral_max_for_unit(model_dir, header, data, layer_i, unit))
        residual_rms = float(task.get("residual_rms") or controls[0]["residual_rms"])
        for dose in _stage2_doses(layers[layer_i], unit, residual_rms, neutral_max):
            prompt_groups = [(identity, True)]
            if dose["dose_label"] == "rho_0.10":
                prompt_groups.append((non_identity, False))
            if task.get("concept_id") == "trump" and task["role"] != "control":
                prompt_groups.append((trump_style, False))
            for prompt_rows, is_identity in prompt_groups:
                for prompt_id, prompt_text, prompt_group in prompt_rows:
                    for sample_index in range(args.samples):
                        key = f"push2:{task['role']}:{task.get('concept_id')}:{layer_i}:{unit}:{dose['dose_label']}:{prompt_id}:{sample_index}"
                        if key in done:
                            continue
                        gen_seed = _stable_seed(f"stage2:{prompt_id}:{sample_index}")
                        direction = directions[layer_i] if is_identity else None
                        answer, exceeded_share, exceeded, total, projection, projection_tokens = _sample_answer(
                            torch, tokenizer, model, layers[layer_i], unit, dose["level"], prompt_text,
                            args.device, args.max_new_tokens, gen_seed, direction)
                        baseline_projection = (baselines[(prompt_id, sample_index)][layer_i]["projection"]
                                               if is_identity else None)
                        row = {"row_key": key, "stage": "push2", "model_id": args.model,
                               "family": header["model_family"], "role": task["role"],
                               "concept_id": task.get("concept_id"), "category": task.get("category"),
                               "held_out": bool(task.get("held_out", False)), "layer": layer_i,
                               "unit": unit, "rank": task.get("rank"),
                               "screen_pass": task.get("screen_pass"),
                               "confirmation_pass": task.get("confirmation_pass"),
                               "depth_fraction": float(layer_i / max(1, len(layers) - 1)),
                               "dose_kind": dose["dose_kind"], "dose_label": dose["dose_label"],
                               "rho": dose.get("rho"), "natural_max_multiple": dose.get("natural_max_multiple"),
                               "equivalent_rho": dose.get("equivalent_rho"), "level": dose["level"],
                               "prompt_id": prompt_id, "prompt_group": prompt_group,
                               "identity_prompt": is_identity, "sample_index": sample_index,
                               "temperature": .7, "max_new_tokens": args.max_new_tokens,
                               "answer": answer,
                               "clamped_token_natural_above_level_share": exceeded_share,
                               "clamped_tokens_above_level": exceeded, "clamped_tokens_total": total,
                               "self_direction_projection_baseline": baseline_projection,
                               "self_direction_projection_clamped": projection,
                               "self_direction_projection_drop": (baseline_projection - projection
                                                                  if baseline_projection is not None and projection is not None else None),
                               "self_direction_projection_tokens": projection_tokens,
                               "judge_status": "pending"}
                        append_jsonl(output_path, row)
                        done.add(key)
                        new += 1
                        if new % 20 == 0:
                            print(f"{args.model}: push2 wrote {new} new answers", flush=True)
    print(json.dumps({"model_id": args.model, "push2_new_answers": new,
                      "push2_path": str(output_path), "hit_rows": len(hits), "controls": len(controls)}))
    return jsonl_rows(output_path)


def _openai_chat(endpoint: str, model: str, prompt: str, timeout: float = 90.0) -> str:
    if not endpoint:
        raise RuntimeError("set OPENAI_BASE_URL or pass --endpoint for the optional blind judge")
    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    body = json.dumps({"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]})
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body.encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"judge request failed: {exc}") from exc
    return str(payload["choices"][0]["message"]["content"])


def judge_prompt(answer: str, style_score: bool, rubric: dict[str, str]) -> str:
    # Deliberately receives no concept, prompt, model, layer, or neuron metadata.
    schema = '{"persona_label":"free text, none, or broken","keeps_assistant_frame":"yes or no"'
    if style_score:
        schema += ',"style_score":0'
    schema += '}'
    instructions = (
        "Classify this answer alone. Do not infer a label from a hidden prompt or topic. "
        f"Use this identity rule: {rubric['identity_label']} "
        f"Use this assistant-frame rule: {rubric['assistant_frame']} "
    )
    if style_score:
        instructions += f"Apply this style rubric without naming a person: {rubric['trump_style']} "
    return instructions + f"Return only JSON matching {schema}. Answer: <answer>{answer}</answer>"


def _parse_judgment(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("blind judge returned no JSON object")
    value = json.loads(match.group(0))
    label = str(value.get("persona_label", "none")).strip()
    if not label:
        label = "none"
    keeps = str(value.get("keeps_assistant_frame", "no")).strip().lower()
    if keeps not in {"yes", "no"}:
        keeps = "no"
    style = value.get("style_score")
    if style is not None:
        style = max(0, min(4, int(style)))
    return {"judge_label": label, "keeps_assistant_frame": keeps, "style_score": style}


def run_judge(args: argparse.Namespace) -> list[dict[str, Any]]:
    data = read_json(DATA_PATH)
    input_path = Path(args.input)
    rows = jsonl_rows(input_path)
    if args.reliability_check and len(rows) < 100:
        raise ValueError(f"judge reliability check needs at least 100 answer rows; got {len(rows)}")
    rubric = data["judge_rubric"]
    endpoint = args.endpoint or os.environ.get("OPENAI_BASE_URL", "")
    output_path = Path(args.out) if args.out else input_path.with_name(input_path.stem + ".judged.jsonl")
    done = completed_keys(output_path)
    existing = {str(row.get("row_key")): row for row in jsonl_rows(output_path)}
    new = 0
    reliability = []
    reliability_first: dict[str, dict[str, Any]] = {}
    if args.reliability_check:
        for row in rows[:100]:
            row_key = str(row.get("row_key"))
            prompt = judge_prompt(str(row.get("answer", "")),
                                  row.get("prompt_group") == "trump_style", rubric)
            if row_key in existing:
                first = {"judge_label": existing[row_key].get("judge_label", "none"),
                         "keeps_assistant_frame": existing[row_key].get("keeps_assistant_frame", "no"),
                         "style_score": existing[row_key].get("style_score")}
            else:
                first = _parse_judgment(_openai_chat(endpoint, args.judge_model, prompt))
                reliability_first[row_key] = first
            second = _parse_judgment(_openai_chat(endpoint, args.judge_model, prompt))
            reliability.append({"label_agreement": second["judge_label"] == first["judge_label"],
                                "assistant_agreement": second["keeps_assistant_frame"] == first["keeps_assistant_frame"],
                                "style_agreement": second["style_score"] == first["style_score"]})
    for row in rows:
        if row.get("row_key") in done:
            continue
        is_trump_style = row.get("prompt_group") == "trump_style"
        prompt = judge_prompt(str(row.get("answer", "")), is_trump_style, rubric)
        judgment = reliability_first.get(str(row.get("row_key")))
        if judgment is None:
            raw = _openai_chat(endpoint, args.judge_model, prompt)
            judgment = _parse_judgment(raw)
        result = {**row, **judgment, "judge_model": args.judge_model,
                  "judge_blind_input_fields": ["answer"], "judge_status": "complete"}
        append_jsonl(output_path, result)
        done.add(str(row.get("row_key")))
        new += 1
        if new % 20 == 0:
            print(f"judge wrote {new} new rows", flush=True)
    if args.reliability_check:
        report = {key: float(np.mean([row[key] for row in reliability]))
                  for key in ("label_agreement", "assistant_agreement", "style_agreement")}
        atomic_json({"judge_model": args.judge_model, "n_answers": len(reliability),
                     "double_judge_exact_agreement": report}, output_path.with_suffix(".reliability.json"))
    if input_path.name.startswith("push1"):
        judged_rows = jsonl_rows(output_path)
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in judged_rows:
            if row.get("role") in {"candidate", "positive_control"}:
                key = (row.get("model_id"), row.get("concept_id"), row.get("layer"), row.get("unit"))
                groups[key].append(row)
        hits = []
        for key, group in groups.items():
            identity_claim = any(r.get("prompt_group") == "identity" and _row_is_persona(r) for r in group)
            screen_near_hit = any(r.get("screen_pass") or r.get("confirmation_pass") for r in group)
            if identity_claim or screen_near_hit:
                seed = next(r for r in group if r.get("rank") is not None or r.get("role") == "positive_control")
                hits.append({"row_key": f"judged-hit:{key[0]}:{key[1]}:{key[2]}:{key[3]}",
                             "model_id": key[0], "role": seed.get("role"),
                             "concept_id": key[1], "category": seed.get("category"),
                             "held_out": seed.get("held_out", False), "layer": key[2], "unit": key[3],
                             "rank": seed.get("rank"), "screen_pass": seed.get("screen_pass"),
                             "confirmation_pass": seed.get("confirmation_pass"),
                             "primary_stat": seed.get("primary_stat"),
                             "neutral_natural_max": seed.get("neutral_natural_max"),
                             "residual_rms": seed.get("residual_rms"),
                             "stage1_any_identity_persona_label": identity_claim,
                             "stage1_screen_near_hit": screen_near_hit,
                             "potential_hit": True})
        replace_jsonl(input_path.parent / "push1-hits.jsonl", hits)
    print(json.dumps({"judge_new_rows": new, "output": str(output_path),
                      "reliability_n": len(reliability),
                      "filtered_stage1_hits": len(hits) if input_path.name.startswith("push1") else None}, ensure_ascii=False))
    return jsonl_rows(output_path)


def _normalize_label(value: str) -> str:
    return re.sub(r"[^\w]+", " ", value.casefold(), flags=re.UNICODE).strip()


def label_matches_concept(label: str, concept: dict[str, Any] | str) -> bool:
    if isinstance(concept, str):
        concept_id = concept
        surface_forms = {}
    else:
        concept_id = concept["id"]
        surface_forms = concept.get("surface_forms", {})
    normalized = _normalize_label(label)
    if normalized in {"", "none", "broken", "no persona", "not applicable"}:
        return False
    terms = [concept_id.replace("_", " ")] + list(surface_forms.values())
    if concept_id == "claude":
        terms += ["Anthropic", "made by Anthropic"]
    if concept_id == "anthropic":
        terms += ["Claude", "made by Anthropic"]
    for term in terms:
        term = _normalize_label(str(term))
        if term and re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", normalized, re.UNICODE):
            return True
    return False


def _cluster_permutation_p(candidate: dict[str, float], controls: list[dict[str, float]],
                           n_permutations: int = 50000, seed: int = 35) -> tuple[float, float, int]:
    """Prompt-cluster randomization: choose one of candidate+controls as pseudo-target per prompt."""
    prompts = sorted(set(candidate) & set.intersection(*(set(c) for c in controls))) if controls else []
    if not prompts:
        return float("nan"), float("nan"), 0
    control_matrix = np.asarray([[candidate[p]] + [control[p] for control in controls] for p in prompts], dtype=float)
    observed = float(np.mean(control_matrix[:, 0] - control_matrix[:, 1:].mean(axis=1)))
    rng = np.random.default_rng(seed)
    count = 0
    remaining = n_permutations
    while remaining:
        batch_n = min(5000, remaining)
        choices = rng.integers(0, control_matrix.shape[1], size=(batch_n, len(prompts)))
        # Pseudo-target is selected independently per prompt cluster; samples stay averaged inside prompts.
        selected = np.take_along_axis(control_matrix[None, :, :], choices[:, :, None], axis=2)[:, :, 0]
        all_mean = control_matrix.mean(axis=1)
        pseudo = (selected - (control_matrix.sum(axis=1)[None, :] - selected) /
                  max(1, control_matrix.shape[1] - 1)).mean(axis=1)
        count += int(np.sum(pseudo >= observed - 1e-12))
        remaining -= batch_n
    p_value = (count + 1) / (n_permutations + 1)
    return observed, p_value, len(prompts)


def _row_is_persona(row: dict[str, Any]) -> bool:
    return str(row.get("judge_label", "none")).casefold() not in {"", "none", "broken", "no persona"}


def _per_prompt_rates(rows: list[dict[str, Any]], predicate) -> dict[str, float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[str(row["prompt_id"])].append(int(bool(predicate(row))))
    return {prompt: float(np.mean(values)) for prompt, values in grouped.items()}


def _per_prompt_values(rows: list[dict[str, Any]], value_fn) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["prompt_id"])].append(float(value_fn(row)))
    return {prompt: float(np.mean(values)) for prompt, values in grouped.items()}


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    data = read_json(DATA_PATH)
    by_id = {c["id"]: c for c in data["concepts"]}
    by_id["alien"] = data["positive_control"]
    pushed = [row for path in args.push2 for row in jsonl_rows(path)]
    if not pushed:
        raise ValueError(f"stage 2 output is empty: {args.push2}")
    unjudged = [row for row in pushed if row.get("judge_status") != "complete"]
    if unjudged:
        raise ValueError(f"score requires blind judge output; {len(unjudged)} stage 2 rows are unjudged")
    screen_rows = [row for path in (args.screen or []) for row in jsonl_rows(path)]
    output_path = Path(args.out)
    # One result per prediction id, with raw per-model cells kept beside the derived value.
    primary_rows = [row for row in pushed if row.get("dose_label") == "rho_0.10" and row.get("prompt_group") == "identity"]
    cells = []
    keys = {(row["model_id"], row.get("concept_id"), row.get("layer"), row.get("unit"))
            for row in primary_rows if row.get("role") != "control"}
    for model_id, concept_id, layer, unit in sorted(keys):
        candidate = [row for row in primary_rows if row["model_id"] == model_id and row.get("concept_id") == concept_id
                     and row.get("layer") == layer and row.get("unit") == unit and row.get("role") != "control"]
        controls = [row for row in primary_rows if row["model_id"] == model_id and row.get("role") == "control"]
        if not candidate:
            continue
        concept = by_id.get(str(concept_id), {"id": concept_id, "surface_forms": {}})
        matched = [label_matches_concept(str(row.get("judge_label", "none")), concept) for row in candidate]
        any_persona = [_row_is_persona(row) for row in candidate]
        successes = sum(matched)
        ci = wilson_interval(successes, len(candidate)) if candidate else (float("nan"), float("nan"))
        candidate_prompt_rates = _per_prompt_rates(candidate, lambda r: label_matches_concept(str(r.get("judge_label", "none")), concept))
        control_units = sorted({(int(row["layer"]), int(row["unit"])) for row in controls})
        control_prompt_rates = []
        for control_layer, control_unit in control_units:
            control_rows = [row for row in controls if row.get("layer") == control_layer and row.get("unit") == control_unit]
            control_prompt_rates.append(_per_prompt_rates(control_rows, _row_is_persona))
        observed, p_value, n_prompts = _cluster_permutation_p(candidate_prompt_rates, control_prompt_rates,
                                                               args.cluster_permutations,
                                                               _stable_seed(f"{model_id}:{concept_id}:{layer}:{unit}"))
        control_success = sum(_row_is_persona(row) for row in controls)
        control_n = len(controls)
        control_rate = control_success / control_n if control_n else float("nan")
        confirm_screen = bool(candidate[0].get("confirmation_pass"))
        alpha = .05 / (43 * len(MODELS))
        confirmed = bool(confirm_screen and successes / max(1, len(candidate)) >= .25 and
                         p_value < alpha and control_rate <= .05)
        cells.append({"model_id": model_id, "family": MODEL_FAMILY.get(model_id),
                      "concept_id": concept_id, "layer": layer, "unit": unit,
                      "depth_fraction": candidate[0].get("depth_fraction"),
                      "n": len(candidate), "matched_claims": successes,
                      "matched_claim_rate": successes / max(1, len(candidate)),
                      "matched_claim_ci95": ci, "all_persona_claims": sum(any_persona),
                      "keeps_assistant_frame_rate": float(np.mean([r.get("keeps_assistant_frame") == "yes"
                                                                    for r, hit in zip(candidate, matched) if hit])) if any(matched) else 0.0,
                      "control_n": control_n, "control_rate": control_rate,
                      "prompt_cluster_count": n_prompts, "cluster_effect": observed,
                      "cluster_permutation_p": p_value, "familywise_alpha": alpha,
                      "screen_pass": bool(candidate[0].get("screen_pass")),
                      "screen_confirmation_pass": confirm_screen, "confirmed": confirmed,
                      "dose_label": "rho_0.10"})
    confirmed = [cell for cell in cells if cell["confirmed"] and cell["concept_id"] in by_id
                 and cell["concept_id"] != "alien"]
    confirmed_nonheld = [cell for cell in confirmed if not by_id[cell["concept_id"]].get("held_out", False)]
    nonheld = {c["id"] for c in data["concepts"] if not c["held_out"]}
    by_concept_family = defaultdict(set)
    for cell in confirmed:
        by_concept_family[cell["concept_id"]].add(cell["family"])
    predictions: dict[str, Any] = {}
    predictions["35.1"] = {"confirmed_nonheldout_concepts": len({c["concept_id"] for c in confirmed_nonheld}),
                            "prediction_range": [3, 15], "n_nonheldout_concepts": len(nonheld)}
    agent_categories = {"creature", "role"}
    agent_ids = {c["id"] for c in data["concepts"] if c["category"] in agent_categories}
    object_place_ids = {c["id"] for c in data["concepts"] if c["category"] in {"object", "place"}}
    agents = [c for c in confirmed if c["concept_id"] in agent_ids]
    objects = [c for c in confirmed if c["concept_id"] in object_place_ids]
    predictions["35.2"] = {"agent_find_rate": len({c["concept_id"] for c in agents}) / max(1, len(agent_ids)),
                            "object_place_find_rate": len({c["concept_id"] for c in objects}) / max(1, len(object_place_ids)),
                            "difference": len({c["concept_id"] for c in agents}) / max(1, len(agent_ids)) -
                                         len({c["concept_id"] for c in objects}) / max(1, len(object_place_ids)),
                            "agent_denominator": len(agent_ids), "object_place_denominator": len(object_place_ids)}
    predictions["35.3"] = {"middle_depth_confirmed_fraction": float(np.mean([
        .4 <= float(c["depth_fraction"]) <= .8 for c in confirmed])) if confirmed else 0.0,
        "n_confirmed_candidates": len(confirmed)}
    qwen35 = [c for c in cells if c["model_id"] == "Qwen/Qwen3.5-4B" and c["concept_id"] in {"claude", "anthropic"}
              and c["screen_pass"] and c["screen_confirmation_pass"]]
    qwen34 = [c for c in cells if c["model_id"] == "Qwen/Qwen3-4B" and c["concept_id"] in {"claude", "anthropic"}
              and c["screen_pass"] and c["screen_confirmation_pass"]]
    qwen35_best = max(qwen35, key=lambda c: c["matched_claim_rate"], default=None)
    qwen34_best = max(qwen34, key=lambda c: c["matched_claim_rate"], default=None)
    qwen35_rate = qwen35_best["matched_claim_rate"] if qwen35_best else None
    qwen34_rate = qwen34_best["matched_claim_rate"] if qwen34_best else None
    predictions["35.4"] = {"qwen35_best_claim_rate": qwen35_rate,
                            "qwen35_qualifying_neurons": len(qwen35),
                            "qwen35_prediction_threshold_met": bool(qwen35_rate is not None and qwen35_rate >= .25),
                            "qwen35_best_n": qwen35_best["n"] if qwen35_best else 0,
                            "qwen35_best_ci95": qwen35_best["matched_claim_ci95"] if qwen35_best else None,
                            "qwen3_4b_best_claim_rate": qwen34_rate,
                            "qwen3_4b_qualifying_neurons": len(qwen34),
                            "qwen3_4b_best_n": qwen34_best["n"] if qwen34_best else 0,
                            "qwen3_4b_best_ci95": qwen34_best["matched_claim_ci95"] if qwen34_best else None,
                            "rate_gap_qwen35_minus_qwen34": (qwen35_rate - qwen34_rate
                                                              if qwen35_rate is not None and qwen34_rate is not None else None),
                            "prediction_evaluable": qwen35_rate is not None and qwen34_rate is not None,
                            "supports_prediction": (qwen35_rate >= .25 and qwen35_rate - qwen34_rate >= .10
                                                     if qwen35_rate is not None and qwen34_rate is not None else None)}
    trump_cells = [c for c in cells if c["concept_id"] == "trump"]
    trump_style_rows = [r for r in pushed if r.get("concept_id") == "trump" and
                        r.get("role") != "control" and r.get("prompt_group") == "trump_style" and
                        r.get("dose_label") == "rho_0.10"]
    trump_candidate_keys = sorted({(r["model_id"], int(r["layer"]), int(r["unit"])) for r in trump_style_rows})
    best_trump_by_model = []
    for model_id in sorted({key[0] for key in trump_candidate_keys}):
        candidates_for_model = []
        for candidate_model, layer, unit in trump_candidate_keys:
            if candidate_model != model_id:
                continue
            candidate_rows = [r for r in trump_style_rows if r["model_id"] == model_id and
                              int(r["layer"]) == layer and int(r["unit"]) == unit]
            eligible = [r for r in candidate_rows if int(r.get("style_score") or 0) >= 3 and
                        not label_matches_concept(str(r.get("judge_label", "none")), "trump")]
            candidates_for_model.append({"model_id": model_id, "layer": layer, "unit": unit,
                                         "style_hits": len(eligible), "n_style_answers": len(candidate_rows),
                                         "style_rate": len(eligible) / max(1, len(candidate_rows)),
                                         "style_score_mean": (float(np.mean([int(r["style_score"]) for r in candidate_rows
                                                                              if r.get("style_score") is not None]))
                                                              if any(r.get("style_score") is not None for r in candidate_rows)
                                                              else None)})
        best_trump_by_model.append(max(candidates_for_model, key=lambda c: (c["style_rate"], c["n_style_answers"])))
    best_style_rate = max((row["style_rate"] for row in best_trump_by_model), default=None)
    style_summaries = [r for r in screen_rows if r.get("stage") == "trump_style_summary"]
    after_ratios = [{"model_id": r["model_id"], "ratio": r.get("trump_after_cutoff_to_pre_cutoff_ratio"),
                     "ci95": r.get("trump_after_cutoff_to_pre_cutoff_ratio_ci95"),
                     "n_after_cutoff": r.get("trump_after_cutoff_n", 0)} for r in style_summaries]
    named_ratios = [{"model_id": r["model_id"], "ratio": r.get("trump_unnamed_to_named_activation_ratio"),
                     "ci95": r.get("trump_unnamed_to_named_activation_ratio_ci95"),
                     "n_unnamed": r.get("trump_unnamed_n", 0), "n_named": r.get("trump_named_n", 0)}
                    for r in style_summaries]
    predictions["35.5"] = {"confirmed_trump_families": sorted({c["family"] for c in trump_cells if c["confirmed"]}),
                            "unnamed_to_named_activation_ratios": named_ratios,
                            "after_cutoff_ratios": after_ratios,
                            "after_cutoff_posts_present": any(r.get("trump_after_cutoff_posts_present") for r in style_summaries)}
    predictions["35.6"] = {"best_model_style_rate": best_style_rate,
                            "any_model_at_least_25_percent": bool(best_style_rate is not None and best_style_rate >= .25),
                            "best_candidate_by_model": best_trump_by_model}
    confirmed_keys = {(c["model_id"], c["concept_id"], c["layer"], c["unit"]) for c in confirmed}
    persona_rows = [r for r in pushed if r.get("role") != "control" and r.get("identity_prompt")
                    and r.get("dose_label") == "rho_0.10" and
                    (r["model_id"], r.get("concept_id"), r.get("layer"), r.get("unit")) in confirmed_keys and
                    label_matches_concept(str(r.get("judge_label", "none")),
                        by_id.get(str(r.get("concept_id")), str(r.get("concept_id"))))]
    predictions["35.7"] = {"assistant_frame_rate_among_persona_answers":
                            float(np.mean([r.get("keeps_assistant_frame") == "yes" for r in persona_rows])) if persona_rows else None,
                            "keeps_yes": sum(r.get("keeps_assistant_frame") == "yes" for r in persona_rows),
                            "n_persona_answers": len(persona_rows)}
    self_direction_finds = []
    for cell in confirmed:
        key = (cell["model_id"], cell["concept_id"], cell["layer"], cell["unit"])
        candidate_rows = [r for r in pushed if
                          (r["model_id"], r.get("concept_id"), r.get("layer"), r.get("unit")) == key and
                          r.get("role") != "control" and r.get("identity_prompt") and
                          r.get("dose_label") == "rho_0.10" and r.get("self_direction_projection_drop") is not None]
        control_rows = [r for r in pushed if r.get("model_id") == cell["model_id"] and
                        r.get("role") == "control" and r.get("identity_prompt") and
                        r.get("dose_label") == "rho_0.10" and r.get("self_direction_projection_drop") is not None]
        if not candidate_rows:
            continue
        prompt_drops = _per_prompt_values(candidate_rows, lambda r: float(r["self_direction_projection_drop"]))
        drop_values = np.asarray(list(prompt_drops.values()), dtype=float)
        control_drops = np.asarray([float(r["self_direction_projection_drop"]) for r in control_rows], dtype=float)
        control_p95 = float(np.quantile(control_drops, .95)) if len(control_drops) else None
        mean_drop = float(drop_values.mean())
        self_direction_finds.append({"model_id": cell["model_id"], "concept_id": cell["concept_id"],
                                     "layer": cell["layer"], "unit": cell["unit"],
                                     "mean_projection_drop_vs_unclamped": mean_drop,
                                     "ci95_over_prompt_clusters": _bootstrap_mean_ci(drop_values,
                                         seed=_stable_seed(f"direction:{key}")),
                                     "control_drop_95th_percentile": control_p95,
                                     "exceeds_control_95th_percentile": (mean_drop > control_p95
                                                                           if control_p95 is not None else None),
                                     "n_candidate_answers": len(candidate_rows),
                                     "n_prompt_clusters": len(prompt_drops),
                                     "n_control_clamp_answers": len(control_drops)})
    direction_summaries = [{"model_id": r["model_id"], "layer_summaries": r["layer_summaries"],
                            "heldout_ai_n": r.get("heldout_ai_n"),
                            "heldout_human_n": r.get("heldout_human_n")}
                           for r in screen_rows if r.get("stage") == "self_direction_summary"]
    predictions["35.8"] = {"confirmed_find_direction_drops": self_direction_finds,
                            "n_confirmed_finds_with_projection": len(self_direction_finds),
                            "direction_separation_by_model": direction_summaries,
                            "direction_definition": "unit-normalized mean post-block residual over 24 train AI descriptions minus 24 matched train human descriptions; separation on 8 held-out examples per class"}
    predictions["35.11"] = {"confirmed_concept_model_pairs": len(confirmed),
                             "concepts_confirmed_in_two_or_more_families": sum(len(families) >= 2 for families in by_concept_family.values()),
                             "prediction_target_below": 40, "n_model_families": 3,
                             "n_concepts": 43, "n_models": len(MODELS)}
    all_persona_rows = [r for r in pushed if r.get("role") != "control" and r.get("identity_prompt")
                        and r.get("dose_label") == "rho_0.10" and _row_is_persona(r)]
    matched_persona = [r for r in all_persona_rows if label_matches_concept(
        str(r.get("judge_label", "none")), by_id.get(str(r.get("concept_id")), str(r.get("concept_id"))))]
    predictions["35.12"] = {"screened_concept_specificity": len(matched_persona) / max(1, len(all_persona_rows)),
                             "matched_claims": len(matched_persona), "all_persona_claims": len(all_persona_rows)}
    natural_groups = defaultdict(list)
    for row in pushed:
        if row.get("role") != "control" and row.get("identity_prompt") and row.get("dose_label") == "natural_4x":
            natural_groups[(row["model_id"], row.get("concept_id"), row.get("layer"), row.get("unit"))].append(row)
    rho_status = {(c["model_id"], c["concept_id"], c["layer"], c["unit"]): bool(c["confirmed"]) for c in cells}
    disagreements = 0
    total_dose_pairs = 0
    natural_hits = 0
    alpha = .05 / (43 * len(MODELS))
    for key, group in natural_groups.items():
        model_id, concept_id, layer, unit = key
        concept = by_id.get(str(concept_id), {"id": concept_id, "surface_forms": {}})
        hit_rate = sum(label_matches_concept(str(r.get("judge_label", "none")), concept) for r in group) / max(1, len(group))
        control_rows = [r for r in pushed if r.get("model_id") == model_id and r.get("role") == "control" and
                        r.get("identity_prompt") and r.get("dose_label") == "natural_4x"]
        control_rate = sum(_row_is_persona(r) for r in control_rows) / max(1, len(control_rows))
        target_prompt_rates = _per_prompt_rates(group, lambda r: label_matches_concept(str(r.get("judge_label", "none")), concept))
        control_rates = []
        for control_unit in sorted({(int(r["layer"]), int(r["unit"])) for r in control_rows}):
            ctrl = [r for r in control_rows if (int(r["layer"]), int(r["unit"])) == control_unit]
            control_rates.append(_per_prompt_rates(ctrl, _row_is_persona))
        _effect, p_value, _clusters = _cluster_permutation_p(target_prompt_rates, control_rates,
              args.cluster_permutations, _stable_seed(f"natural4x:{model_id}:{concept_id}:{layer}:{unit}"))
        screen_pass = bool(group[0].get("confirmation_pass"))
        natural_hit = bool(screen_pass and hit_rate >= .25 and p_value < alpha and control_rate <= .05)
        natural_hits += int(natural_hit)
        if key in rho_status:
            disagreements += natural_hit != rho_status[key]
            total_dose_pairs += 1
    predictions["35.13"] = {"rho10_vs_natural4x_candidate_hit_status_disagreement_rate": disagreements / max(1, total_dose_pairs),
                             "disagreements": disagreements, "paired_candidates": total_dose_pairs,
                             "natural4x_confirmed_candidates": natural_hits,
                             "candidate_hit_status_rule": "confirmation contrast pass; >=25% screened-concept identity claims; same prompt-cluster p<familywise alpha; control rate<=5%"}
    naive_top = defaultdict(float)
    perm_top = defaultdict(float)
    for row in screen_rows:
        if row.get("stage") == "screen" and row.get("rank") == 1:
            key = (row["model_id"], row["concept_id"])
            naive_top[key] = float(row["primary_stat"])
            perm_top[key] = float(row["permutation_maximum"])
    predictions["35.14"] = {"cells_permutation_max_exceeds_top1_contrast":
                            sum(perm_top[k] > v for k, v in naive_top.items()),
                            "concept_model_cells": len(naive_top),
                            "fraction": sum(perm_top[k] > v for k, v in naive_top.items()) / max(1, len(naive_top))}
    alien_rows = [r for r in pushed if r.get("concept_id") == "alien" and r.get("role") == "positive_control"
                  and r.get("dose_label") == "natural_8x" and r.get("identity_prompt")]
    alien_claims = sum(label_matches_concept(str(r.get("judge_label", "none")), "alien") for r in alien_rows)
    alien_ci = wilson_interval(alien_claims, len(alien_rows)) if alien_rows else (None, None)
    predictions["35.15"] = {"alien_fixed_neuron_rank": next((r.get("fixed_neuron_rank") for r in screen_rows
                                                               if r.get("stage") == "positive_control" and
                                                               r.get("model_id") == "Qwen/Qwen3.5-4B"), None),
                             "alien_claim_rate_natural8x": alien_claims / max(1, len(alien_rows)) if alien_rows else None,
                             "alien_claim_rate_ci95": alien_ci, "n_alien_answers": len(alien_rows),
                             "replication_threshold": .5}
    result = {"prediction_metrics": predictions, "model_cell_summary": cells,
              "scoring_contract": {"identity_endpoint_dose": "rho_0.10", "prompt_cluster_permutations": args.cluster_permutations,
                                   "alpha": .05 / (43 * len(MODELS)), "familywise_denominator": 43 * len(MODELS),
                                   "confirmed_rule": "confirmation contrast > model permutation maximum; >=25% screened-concept claims; cluster p<alpha; control persona claim rate<=5%"}}
    rows_to_write = [{"row_key": f"score:{prediction_id}", "prediction_id": prediction_id,
                      "value": value, "n": _prediction_n(prediction_id, value, cells, pushed, screen_rows),
                      "ci95": _prediction_ci95(prediction_id, value, cells, pushed, screen_rows),
                      "all_model_cells": cells,
                      "metric": value}
                     for prediction_id, value in predictions.items()]
    _write_jsonl_rows(output_path, rows_to_write)
    atomic_json(result, output_path.with_suffix(".summary.json"))
    print(json.dumps({"scored_predictions": len(predictions), "model_cells": len(cells),
                      "output": str(output_path), "prediction_metrics": predictions}, ensure_ascii=False))
    return result


def _prediction_n(prediction_id: str, value: dict[str, Any], cells: list[dict[str, Any]], pushed: list[dict[str, Any]],
                  screen: list[dict[str, Any]]) -> int:
    if prediction_id == "35.1":
        return 35
    if prediction_id == "35.2":
        data = read_json(DATA_PATH)
        return sum(c["category"] in {"creature", "role", "object", "place"} for c in data["concepts"])
    if prediction_id == "35.3":
        return sum(c["confirmed"] for c in cells if c["concept_id"] != "alien")
    if prediction_id == "35.4":
        return int(value.get("qwen35_qualifying_neurons", 0) + value.get("qwen3_4b_qualifying_neurons", 0))
    if prediction_id == "35.5":
        return sum(row.get("stage") == "trump_style_summary" for row in screen)
    if prediction_id == "35.6":
        return sum(int(row["n_style_answers"]) for row in value.get("best_candidate_by_model", []))
    if prediction_id == "35.7":
        return int(value.get("n_persona_answers", 0))
    if prediction_id == "35.8":
        return int(value.get("n_confirmed_finds_with_projection", 0))
    if prediction_id == "35.11":
        return 43 * len(MODELS)
    if prediction_id == "35.12":
        return sum(_row_is_persona(row) for row in pushed if row.get("role") != "control" and
                   row.get("identity_prompt") and row.get("dose_label") == "rho_0.10")
    if prediction_id == "35.13":
        return int(value.get("paired_candidates", value_candidate_pairs(cells, pushed)))
    if prediction_id == "35.14":
        return sum(row.get("stage") == "screen" and row.get("rank") == 1 for row in screen)
    if prediction_id == "35.15":
        return int(value.get("n_alien_answers", 0))
    return sum(row.get("dose_label") == "rho_0.10" for row in pushed)


def value_candidate_pairs(cells: list[dict[str, Any]], pushed: list[dict[str, Any]]) -> int:
    return len({(r["model_id"], r.get("concept_id"), r.get("layer"), r.get("unit")) for r in pushed
                if r.get("role") != "control" and r.get("identity_prompt") and r.get("dose_label") == "natural_4x"})


def _prediction_ci95(prediction_id: str, value: dict[str, Any], cells: list[dict[str, Any]],
                     pushed: list[dict[str, Any]], screen: list[dict[str, Any]]) -> Any:
    def interval(successes: int, n: int):
        return list(wilson_interval(successes, n)) if n else None
    if prediction_id == "35.1":
        return interval(value["confirmed_nonheldout_concepts"], value["n_nonheldout_concepts"])
    if prediction_id == "35.2":
        data = read_json(DATA_PATH)
        agent_ids = {c["id"] for c in data["concepts"] if c["category"] in {"creature", "role"}}
        other_ids = {c["id"] for c in data["concepts"] if c["category"] in {"object", "place"}}
        agent_success = len({c["concept_id"] for c in cells if c["confirmed"] and c["concept_id"] in agent_ids})
        other_success = len({c["concept_id"] for c in cells if c["confirmed"] and c["concept_id"] in other_ids})
        a, na = agent_success / max(1, len(agent_ids)), len(agent_ids)
        b, nb = other_success / max(1, len(other_ids)), len(other_ids)
        se = math.sqrt(a * (1-a) / max(1, na) + b * (1-b) / max(1, nb))
        return {"agent_rate": interval(agent_success, na), "object_place_rate": interval(other_success, nb),
                "difference_normal_approx": [max(-1.0, value["difference"] - 1.96 * se),
                                             min(1.0, value["difference"] + 1.96 * se)]}
    if prediction_id == "35.3":
        good = sum(c["confirmed"] and c["concept_id"] != "alien" and .4 <= float(c["depth_fraction"]) <= .8 for c in cells)
        return interval(good, value["n_confirmed_candidates"])
    if prediction_id == "35.4":
        return {"qwen35": value.get("qwen35_best_ci95"), "qwen3_4b": value.get("qwen3_4b_best_ci95")}
    if prediction_id == "35.5":
        return {"unnamed_to_named": [{"model_id": row["model_id"], "n": row["n_unnamed"] + row["n_named"],
                                       "ci95": row["ci95"]} for row in value["unnamed_to_named_activation_ratios"]],
                "after_cutoff": [{"model_id": row["model_id"], "n": row["n_after_cutoff"],
                                  "ci95": row["ci95"]} for row in value["after_cutoff_ratios"]]}
    if prediction_id == "35.6":
        return {"by_model": [{"model_id": row["model_id"], "n": row["n_style_answers"],
                              "ci95": interval(row["style_hits"], row["n_style_answers"])}
                             for row in value.get("best_candidate_by_model", [])]}
    if prediction_id == "35.7":
        return interval(value["keeps_yes"], value["n_persona_answers"])
    if prediction_id == "35.8":
        return {"confirmed_find_drops": [{"model_id": row["model_id"], "concept_id": row["concept_id"],
                                            "n": row["n_prompt_clusters"],
                                            "ci95": row["ci95_over_prompt_clusters"]}
                                           for row in value["confirmed_find_direction_drops"]],
                "heldout_direction_separation": [{"model_id": item["model_id"],
                    "layers": [{"layer": layer["layer"], "n": layer["n_heldout_ai"] + layer["n_heldout_human"],
                                "ci95": layer["heldout_projection_separation_ci95"]}
                               for layer in item["layer_summaries"]]}
                    for item in value["direction_separation_by_model"]]}
    if prediction_id == "35.11":
        return interval(value["confirmed_concept_model_pairs"], value["n_concepts"] * value["n_models"])
    if prediction_id == "35.12":
        return interval(value["matched_claims"], value["all_persona_claims"])
    if prediction_id == "35.13":
        return interval(value["disagreements"], value["paired_candidates"])
    if prediction_id == "35.14":
        return interval(value["cells_permutation_max_exceeds_top1_contrast"], value["concept_model_cells"])
    if prediction_id == "35.15":
        return value["alien_claim_rate_ci95"]
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("screen", "push1", "push2"):
        p = sub.add_parser(command)
        p.add_argument("--model", required=True, choices=tuple(MODELS))
        p.add_argument("--revision", default=None, help="override the pinned cached revision")
        p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
        p.add_argument("--device", choices=("cpu", "xpu"), default="cpu")
        p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
        p.add_argument("--out", required=True, type=Path)
        if command == "screen":
            p.add_argument("--permutations", type=int, default=PERMUTATIONS)
            p.add_argument("--concept-start", type=int, default=None,
                           help="inclusive zero-based shard index over 43 concepts plus alien")
            p.add_argument("--concept-end", type=int, default=None,
                           help="exclusive zero-based shard index over 43 concepts plus alien")
            p.add_argument("--capture-only", action="store_true",
                           help="capture this shard; run again with --reuse-cache after all shards finish")
            p.add_argument("--reuse-cache", action="store_true",
                           help="score existing complete activation cache without loading model weights")
            p.add_argument("--score-shard", action="store_true",
                           help="score only --concept-start:--concept-end and save a mergeable null shard")
            p.add_argument("--merge-screen-shards", action="store_true",
                           help="combine complete model-wide concept score shards into screen.jsonl")
            p.add_argument("--trump-posts", type=Path, default=None,
                           help="optional run-time JSON or Markdown archive; raw text is never persisted")
        else:
            p.add_argument("--max-new-tokens", type=int, default=80)
            p.add_argument("--samples", type=int, default=4 if command == "push1" else 8)
            p.add_argument("--concept-start", type=int, default=None,
                           help="inclusive zero-based concept shard index over 43 concepts plus alien")
            p.add_argument("--concept-end", type=int, default=None,
                           help="exclusive zero-based concept shard index over 43 concepts plus alien")
            if command == "push1":
                p.add_argument("--screen", default=None, help="screen JSONL; default is OUT/MODEL/screen.jsonl")
            else:
                p.add_argument("--hits", default=None, help="push1 potential-hit JSONL; default is OUT/MODEL/push1-potential-hits.jsonl")
                p.add_argument("--controls", default=None, help="live random control JSON; default is OUT/MODEL/controls.json")
                p.add_argument("--controls-only", action="store_true", help="push only selected random controls")
                p.add_argument("--skip-controls", action="store_true", help="push candidates without random controls")
                p.add_argument("--control-start", type=int, default=None, help="inclusive index into the 10 live random controls")
                p.add_argument("--control-end", type=int, default=None, help="exclusive index into the 10 live random controls")
    p = sub.add_parser("judge", help="optional blind OpenAI-compatible judge")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--out", default=None, type=Path)
    p.add_argument("--judge-model", default=os.environ.get("OPENAI_MODEL"))
    p.add_argument("--endpoint", default=None, help="defaults to OPENAI_BASE_URL; no key is hard-coded")
    p.add_argument("--reliability-check", action="store_true",
                   help="double-judge the first 100 answers and save exact agreement")
    p = sub.add_parser("score", help="score stage-2 judged JSONL and all prediction rows")
    p.add_argument("--push2", required=True, nargs="+", type=Path, help="one or more judged stage-2 JSONL files")
    p.add_argument("--screen", required=True, nargs="+", type=Path, help="one or more model screen JSONL files")
    p.add_argument("--out", required=True, type=Path, help="JSONL output; summary JSON is written beside it")
    p.add_argument("--cluster-permutations", type=int, default=50000)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "screen":
            if args.permutations < 200:
                parser.error("screen requires --permutations >= 200")
            run_screen(args)
        elif args.command == "push1":
            run_push1(args)
        elif args.command == "push2":
            run_push2(args)
        elif args.command == "judge":
            if not args.judge_model:
                parser.error("pass --judge-model or set OPENAI_MODEL")
            run_judge(args)
        elif args.command == "score":
            if args.cluster_permutations < 10000:
                parser.error("score requires at least 10,000 cluster permutations for familywise p-value resolution")
            run_score(args)
        return 0
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"part35 {args.command} failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


