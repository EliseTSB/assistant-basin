#!/usr/bin/env python3
"""Part 65 A770 write-side steering.
Reads the Part 64 collection JSONL and a local pinned Qwen3-1.7B snapshot plus
the Part 60 corpus; writes resumable per-unit rollouts, steer-lens summaries,
and a manifest. No model is downloaded: Transformers is local-files-only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any, Iterable

try:  # fcntl is POSIX-only; Windows uses msvcrt below.
    import fcntl
except ImportError:  # pragma: no cover - exercised by Windows runtime
    fcntl = None
try:
    import msvcrt
except ImportError:  # pragma: no cover - exercised by POSIX runtime
    msvcrt = None

HERE = Path(__file__).resolve().parent
BASIN = HERE.parent
if str(BASIN) not in sys.path:
    sys.path.insert(0, str(BASIN))

MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
CONTROL_SEED = 5601
CONTROL_POOL_SIZE = 100
EXPECTED_LAYERS = 28
DEFAULT_COLLECTION = Path("/home/elisepc/green/worktrees/ab-part64/results/part64/collection.jsonl")
DEFAULT_CORPUS = Path(
    os.environ.get("PART64_CORPUS")
    or os.environ.get("PART60_CORPUS")
    or "/mnt/c/Users/alexa/neuron-probe-gpu/qwen35/pile10k.jsonl"
)
PROMPTS = {
    "who": "Who are you?",
    "snake": "Write a short poem about a snake.",
    "orange": "Describe an orange.",
}
DOSES = (-0.7, 0.5, 0.8)
LENS_DOSES = (0.8, -0.8)
LENS_CONTEXTS = (
    "The weather today is",
    "In the middle of the",
    "Here is a simple",
    "One thing to consider is",
)
MAX_NEW_TOKENS = 48


class SteerError(RuntimeError):
    """Invalid Part 65 input, model shape, or resumable state."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    last_nonblank_line = max((index for index, line in enumerate(lines, 1) if line.strip()), default=0)
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if line_number == last_nonblank_line:
                print(f"warning: ignoring invalid final JSONL line in {path}:{line_number}",
                      file=sys.stderr)
                continue
            raise SteerError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise SteerError(f"{path}:{line_number}: each line must be an object")
        rows.append(row)
    return rows


def load_units(path: Path = DEFAULT_COLLECTION) -> list[dict[str, Any]]:
    """Load the frozen 500 Part 60 dev identities from Part 64's collection."""
    rows = read_jsonl(path)
    seen = set()
    units = []
    for row in rows:
        key = row.get("unit_key")
        kind = row.get("type")
        layer = row.get("layer")
        if (not isinstance(key, str) or kind not in ("neuron", "control")
                or isinstance(layer, bool) or not isinstance(layer, int)
                or not 0 <= layer < EXPECTED_LAYERS or key in seen):
            raise SteerError(f"invalid or duplicate Part 64 unit record: {key!r}")
        neuron = row.get("unit")
        control = row.get("control_id")
        if kind == "neuron":
            if isinstance(neuron, bool) or not isinstance(neuron, int) or control is not None:
                raise SteerError(f"{key}: expected neuron unit and null control_id")
            expected = f"L{layer}:u{neuron}"
        else:
            if isinstance(control, bool) or not isinstance(control, int) or neuron is not None:
                raise SteerError(f"{key}: expected control_id and null unit")
            expected = f"L{layer}:c{control}"
        if key != expected:
            raise SteerError(f"{key}: identity disagrees with record coordinates")
        seen.add(key)
        units.append(dict(row))
    if len(units) != 500 or sum(row["type"] == "neuron" for row in units) != 250:
        raise SteerError(f"expected Part 60's exact 250 neuron / 250 control dev units, found {len(units)}")
    return units


def _largest_remainder(total: int, groups: dict[Any, list[Any]]) -> dict[Any, int]:
    size = sum(len(items) for items in groups.values())
    if total < 0 or total > size:
        raise ValueError("sample size is outside the candidate range")
    if not groups or size == 0:
        return {key: 0 for key in groups}
    exact = {key: total * len(items) / size for key, items in groups.items()}
    quotas = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(groups, key=lambda key: (exact[key] - quotas[key], str(key)), reverse=True)
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def select_pilot_units(units: list[dict[str, Any]], count: int | None,
                       kind: str = "both", seed: int = 6501) -> list[dict[str, Any]]:
    """Select a reproducible sample balanced by type, then proportional by layer."""
    if kind not in ("both", "neuron", "control"):
        raise ValueError("type must be both, neuron, or control")
    candidates = [u for u in units if kind == "both" or u["type"] == kind]
    if count is None:
        return candidates
    if isinstance(count, bool) or count <= 0 or count > len(candidates):
        raise ValueError(f"units must be in 1..{len(candidates)}")
    type_groups = {t: [u for u in candidates if u["type"] == t]
                   for t in ("neuron", "control") if any(u["type"] == t for u in candidates)}
    type_quota = _largest_remainder(count, type_groups)
    selected = []
    for unit_type, type_rows in type_groups.items():
        by_layer: dict[int, list[dict[str, Any]]] = {}
        for unit in type_rows:
            by_layer.setdefault(int(unit["layer"]), []).append(unit)
        quotas = _largest_remainder(type_quota[unit_type], by_layer)
        for layer, rows in by_layer.items():
            rng = random.Random(f"{seed}:{unit_type}:{layer}")
            selected.extend(rng.sample(rows, quotas[layer]))
    return sorted(selected, key=lambda row: (int(row["layer"]), row["type"], row["unit_key"]))


def control_directions(width: int, layer: int, torch: Any, device: Any) -> Any:
    """Return the canonical 100 Part 60 directions generated by atlas_collect_fast2."""
    try:
        import atlas_collect_fast2 as atlas
    except ImportError as exc:
        raise SteerError("atlas_collect_fast2 is required to reproduce Part 60 controls") from exc
    values = atlas.generate_controls(CONTROL_POOL_SIZE, width, CONTROL_SEED, layer)
    if len(values) != CONTROL_POOL_SIZE:
        raise SteerError(f"layer {layer}: canonical control pool has the wrong size")
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def write_direction(torch: Any, down_proj: Any, unit: dict[str, Any],
                    controls: Any | None = None) -> Any:
    """Return the unit-length W_down column or W_down @ canonical control vector."""
    weight = down_proj.weight.detach().float()
    if weight.ndim != 2:
        raise SteerError("down_proj.weight must be a matrix")
    if unit["type"] == "neuron":
        index = int(unit["unit"])
        if not 0 <= index < weight.shape[1]:
            raise SteerError(f"{unit['unit_key']}: neuron column out of range")
        raw = weight[:, index]
    elif unit["type"] == "control":
        index = int(unit["control_id"])
        if controls is None:
            controls = control_directions(int(weight.shape[1]), int(unit["layer"]), torch, weight.device)
        if not 0 <= index < controls.shape[0] or controls.shape[1] != weight.shape[1]:
            raise SteerError(f"{unit['unit_key']}: control coordinate out of range")
        raw = torch.matmul(weight, controls[index].to(device=weight.device, dtype=weight.dtype))
    else:
        raise SteerError(f"unsupported unit type {unit.get('type')!r}")
    norm = torch.linalg.vector_norm(raw)
    if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 0:
        raise SteerError(f"{unit['unit_key']}: write vector has invalid norm")
    return raw / norm


def validate_equal_push_norms(torch: Any, layers: Any,
                              units: list[dict[str, Any]], tolerance: float = 1e-4) -> dict[str, float]:
    """Check unit-normalized real/control directions for every populated layer."""
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for unit in units:
        by_layer.setdefault(int(unit["layer"]), []).append(unit)
    errors: dict[str, float] = {}
    for layer_id, rows in sorted(by_layer.items()):
        down_proj = layers[layer_id].mlp.down_proj
        controls = None
        if any(row["type"] == "control" for row in rows):
            controls = control_directions(int(down_proj.weight.shape[1]), layer_id, torch,
                                          down_proj.weight.device)
        norms = []
        for row in rows:
            vec = write_direction(torch, down_proj, row, controls)
            norms.append(float(torch.linalg.vector_norm(vec).item()))
        spread = max(norms) - min(norms)
        errors[str(layer_id)] = spread
        if spread > tolerance or max(abs(value - 1.0) for value in norms) > tolerance:
            raise SteerError(f"layer {layer_id}: real/control push norms differ by {spread:.8g}")
    return errors


def _get_hidden(output: Any) -> Any:
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, list):
        return output[0]
    return output


@contextmanager
def intervention_hook(mlp: Any, pushes: Any, layer_module: str):
    """Add one [hidden] push per batch row at every sequence position."""
    stats: dict[str, Any] = {"fired_count": 0, "layer_modules": []}

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        stats["fired_count"] += 1
        stats["layer_modules"].append(
            layer_module if _module is mlp else f"unexpected:{type(_module).__name__}"
        )
        hidden = _get_hidden(output)
        vector = pushes.to(device=hidden.device, dtype=hidden.dtype)
        if vector.ndim == 1:
            vector = vector.unsqueeze(0)
        added = hidden + vector[:, None, :]
        if isinstance(output, tuple):
            return (added, *output[1:])
        if isinstance(output, list):
            return [added, *output[1:]]
        return added
    handle = mlp.register_forward_hook(hook)
    try:
        yield stats
    finally:
        handle.remove()


def _validate_hook_stats(stats: Any, expected_layer: int, context: str) -> None:
    expected = f"model.layers.{expected_layer}.mlp"
    if not isinstance(stats, dict) or int(stats.get("fired_count", 0)) <= 0:
        raise SteerError(f"{context}: pushed generation hook fired zero times")
    modules = set(stats.get("layer_modules", []))
    if modules != {expected}:
        raise SteerError(f"{context}: hook fired on {sorted(modules)}, expected {expected}")


def _device_of(model: Any) -> Any:
    return next(model.parameters()).device


def _chat(tokenizer: Any, text: str) -> str:
    # Amendment 3: Qwen3's non-thinking mode (the template inserts an empty think block). In thinking mode the
    # pilot spent all 48 new tokens of 162/180 rollouts inside the "<think> Okay, the user..." preamble.
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _batch(tokenizer: Any, texts: list[str], device: Any) -> dict[str, Any]:
    old_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer([_chat(tokenizer, text) for text in texts],
                            return_tensors="pt", padding=True, add_special_tokens=False)
    finally:
        tokenizer.padding_side = old_side
    return {key: value.to(device) for key, value in encoded.items()}


def generate_batch(torch: Any, model: Any, tokenizer: Any, prompts: list[str],
                   pushes: Any | None = None, max_new_tokens: int = MAX_NEW_TOKENS,
                   hook_layer: int | None = None) -> list[dict[str, Any]]:
    """Greedy batched generation; returns generated token IDs and decoded text."""
    device = _device_of(model)
    encoded = _batch(tokenizer, prompts, device)
    hook_stats = None
    if pushes is not None:
        if hook_layer is None:
            raise SteerError("hook_layer is required when generate_batch receives pushes")
        layers = model.model.layers
        if isinstance(hook_layer, bool) or not 0 <= int(hook_layer) < len(layers):
            raise SteerError(f"hook_layer {hook_layer!r} is outside the model layer range")
        hook_layer = int(hook_layer)
        layer_module = f"model.layers.{hook_layer}.mlp"
        manager = intervention_hook(layers[hook_layer].mlp, pushes, layer_module)
    else:
        manager = _null_context()
    with manager as hook_stats, torch.inference_mode():
        sequences = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if pushes is not None:
        _validate_hook_stats(hook_stats, hook_layer, "generate_batch")
    width = int(encoded["input_ids"].shape[1])
    outputs = []
    for sequence in sequences:
        generated = sequence[width:].detach().cpu().tolist()
        result = {
            "token_ids": [int(token) for token in generated],
            "text": tokenizer.decode(generated, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False),
        }
        if pushes is not None:
            result["hook_fired_count"] = int(hook_stats["fired_count"])
            result["hook_layer_modules"] = sorted(set(hook_stats["layer_modules"]))
        outputs.append(result)
    return outputs


@contextmanager
def _null_context():
    yield


def lens_summary(torch: Any, model: Any, tokenizer: Any, unit_vector: Any,
                 layer: int, median_norm: float) -> dict[str, Any]:
    """Mean next-token log-probability delta over four contexts; top 12 each sign."""
    device = _device_of(model)
    encoded = _batch(tokenizer, list(LENS_CONTEXTS), device)
    with torch.inference_mode():
        base_logits = model(**encoded, use_cache=False).logits[:, -1, :].float()
        base_logp = torch.log_softmax(base_logits, dim=-1)
        result: dict[str, Any] = {}
        for dose in LENS_DOSES:
            pushes = unit_vector.unsqueeze(0).expand(len(LENS_CONTEXTS), -1) * (dose * median_norm)
            with intervention_hook(model.model.layers[layer].mlp, pushes,
                                   f"model.layers.{layer}.mlp") as hook_stats:
                logits = model(**encoded, use_cache=False).logits[:, -1, :].float()
            _validate_hook_stats(hook_stats, layer, "steer lens")
            deltas = torch.log_softmax(logits, dim=-1) - base_logp
            mean_delta = deltas.mean(dim=0)
            values, indices = torch.topk(mean_delta, k=min(12, mean_delta.numel()))
            result[f"{dose:+.1f}"] = [
                {"token": tokenizer.decode([int(index)], skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False),
                 "token_id": int(index), "mean_delta_logp": float(value)}
                for value, index in zip(values.cpu().tolist(), indices.cpu().tolist())
            ]
        result["contexts"] = list(LENS_CONTEXTS)
        return result


def process_unit(torch: Any, model: Any, tokenizer: Any, unit: dict[str, Any],
                 layer_medians: dict[str, float]) -> dict[str, Any]:
    """Run baseline, 9 batched rollouts, the zero-dose check, and the steer lens."""
    layer = int(unit["layer"])
    mlp = model.model.layers[layer].mlp
    down_proj = mlp.down_proj
    width = int(down_proj.weight.shape[1])
    controls = control_directions(width, layer, torch, down_proj.weight.device) if unit["type"] == "control" else None
    direction = write_direction(torch, down_proj, unit, controls)
    median = float(layer_medians[str(layer)])
    if not math.isfinite(median) or median <= 0:
        raise SteerError(f"layer {layer}: median residual norm must be positive and finite")
    prompt_keys = list(PROMPTS)
    prompt_texts = [PROMPTS[key] for key in prompt_keys]
    baseline = generate_batch(torch, model, tokenizer, prompt_texts)
    zero_pushes = direction.unsqueeze(0).expand(3, -1) * 0.0
    zero = generate_batch(torch, model, tokenizer, prompt_texts, zero_pushes,
                          hook_layer=layer)
    _validate_generation_rows(zero, layer, unit["unit_key"], "dose 0")
    dose_zero_matches_baseline = (
        [row["token_ids"] for row in baseline] == [row["token_ids"] for row in zero]
    )
    if not dose_zero_matches_baseline:
        raise SteerError(f"{unit['unit_key']}: dose 0 did not reproduce baseline token-for-token")
    rollout_prompts = []
    rollout_pushes = []
    rollout_meta = []
    for key in prompt_keys:
        for dose in DOSES:
            rollout_prompts.append(PROMPTS[key])
            rollout_pushes.append(direction * (dose * median))
            rollout_meta.append({"prompt_key": key, "dose": dose})
    pushes = torch.stack(rollout_pushes, dim=0)
    # The nine prompt-dose conditions are generated in one batched call.
    rollouts = generate_batch(torch, model, tokenizer, rollout_prompts, pushes,
                              hook_layer=layer)
    _validate_generation_rows(rollouts, layer, unit["unit_key"], "steered rollout")
    baseline_by_prompt = {key: row["token_ids"] for key, row in zip(prompt_keys, baseline)}
    for metadata, result in zip(rollout_meta, rollouts):
        metadata.update(result)
        metadata["differs_from_baseline"] = (
            metadata["token_ids"] != baseline_by_prompt[metadata["prompt_key"]]
        )
    dose_plus_0_8_changed_count = sum(
        row["differs_from_baseline"] for row in rollout_meta if row["dose"] == 0.8
    )
    lens = lens_summary(torch, model, tokenizer, direction, layer, median)
    return {
        "unit_key": unit["unit_key"],
        "type": unit["type"],
        "layer": layer,
        "unit": unit.get("unit"),
        "control_id": unit.get("control_id"),
        "baseline": [{"prompt_key": key, **row} for key, row in zip(prompt_keys, baseline)],
        "dose_zero": [{"prompt_key": key, **row} for key, row in zip(prompt_keys, zero)],
        "rollouts": [{"prompt_key": meta["prompt_key"], "dose": meta["dose"],
                      "token_ids": meta["token_ids"], "text": meta["text"],
                      "differs_from_baseline": meta["differs_from_baseline"],
                      "hook_fired_count": meta["hook_fired_count"],
                      "hook_layer_modules": meta["hook_layer_modules"]}
                     for meta in rollout_meta],
        "dose_zero_matches_baseline": dose_zero_matches_baseline,
        "dose_plus_0_8_changed_count": int(dose_plus_0_8_changed_count),
        "steer_lens": lens,
        "median_residual_norm": median,
        "write_vector_norm": float(torch.linalg.vector_norm(direction).item()),
        "push_norms": {
            f"{dose:+.1f}": float(torch.linalg.vector_norm(direction * (dose * median)).item())
            for dose in DOSES
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(payload, encoding="utf-8")
    os.replace(temp, path)


def _lock_file(path: Path):
    @contextmanager
    def locked():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as stream:
            if fcntl is not None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            elif msvcrt is not None:
                stream.seek(0, os.SEEK_END)
                if stream.tell() == 0:
                    stream.write(b"\0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                elif msvcrt is not None:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    return locked()


def atomic_append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one complete JSONL row under a cross-process lock and fsync it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                          allow_nan=False) + "\n").encode("utf-8")
    with _lock_file(path.with_name(path.name + ".lock")):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            offset = 0
            while offset < len(encoded):
                offset += os.write(fd, encoded[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)


def load_layer_medians(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("model_revision") != MODEL_REVISION or data.get("token_count") != 2000
            or not isinstance(data.get("medians"), dict)):
        raise SteerError(f"{path}: expected saved medians for the pinned model and 2,000 tokens")
    medians = {str(layer): float(value) for layer, value in data["medians"].items()}
    if set(medians) != {str(i) for i in range(EXPECTED_LAYERS)}:
        raise SteerError(f"{path}: expected medians for all 28 layers")
    if any(not math.isfinite(value) or value <= 0 for value in medians.values()):
        raise SteerError(f"{path}: residual norm medians must be positive and finite")
    return medians


def corpus_token_ids(path: Path, tokenizer: Any, target: int = 2000) -> list[int]:
    """Take exactly the first target tokenizer IDs from the Part 60 JSONL corpus."""
    ids: list[int] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SteerError(f"{path}:{line_number}: corpus is not valid JSONL") from exc
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str):
                raise SteerError(f"{path}:{line_number}: expected a string text field")
            ids.extend(tokenizer.encode(text, add_special_tokens=False))
            if len(ids) >= target:
                return ids[:target]
    if len(ids) != target:
        raise SteerError(f"Part 60 corpus had only {len(ids)} tokens; need exactly {target}")
    return ids


def measure_layer_medians(torch: Any, model: Any, tokenizer: Any, corpus: Path) -> dict[str, Any]:
    """Measure median residual-stream L2 norm per layer over exactly 2,000 corpus tokens."""
    token_ids = corpus_token_ids(corpus, tokenizer, 2000)
    device = _device_of(model)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    samples: dict[str, Any] = {}
    handles = []
    for layer_id, block in enumerate(model.model.layers):
        def capture(_module: Any, _inputs: Any, output: Any, key: str = str(layer_id)) -> None:
            hidden = _get_hidden(output).detach().float()
            samples[key] = torch.linalg.vector_norm(hidden, dim=-1).reshape(-1).cpu()
        handles.append(block.register_forward_hook(capture))
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(samples) != EXPECTED_LAYERS:
        raise SteerError(f"expected hidden outputs from all 28 layers, got {len(samples)}")
    medians = {key: float(torch.median(value).item()) for key, value in samples.items()}
    return {
        "schema": "part65-layer-medians-v1",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "corpus_path": str(corpus),
        "corpus_sha256": _sha256(corpus),
        "token_count": len(token_ids),
        "measurement": "median per-position L2 norm of each transformer block output",
        "medians": medians,
    }


def load_local_model(model_id: str = MODEL_ID, device: str = "xpu") -> tuple[Any, Any, Any, str]:
    """Load the pinned model from cache only, preferring Intel XPU then CPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device not in ("xpu", "cpu", "auto"):
        raise ValueError("device must be xpu, cpu, or auto")
    xpu_available = bool(getattr(torch, "xpu", None) and torch.xpu.is_available())
    selected = "xpu" if device in ("xpu", "auto") and xpu_available else "cpu"
    dtype = torch.bfloat16 if selected == "xpu" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=MODEL_REVISION, use_fast=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=MODEL_REVISION, torch_dtype=dtype, local_files_only=True
    ).eval()
    model.to(selected)
    return torch, tokenizer, model, selected


def write_manifest(path: Path, collection: Path, medians: dict[str, float],
                   device: str, selected_units: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    manifest = {
        "schema": "part65-steer-v1",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": device,
        "collection_path": str(collection),
        "collection_sha256": _sha256(collection),
        "doses": list(DOSES),
        "prompts": PROMPTS,
        "baseline_count": len(PROMPTS),
        "rollout_count_per_unit": len(PROMPTS) * len(DOSES),
        "max_new_tokens": MAX_NEW_TOKENS,
        "steer_lens_contexts": list(LENS_CONTEXTS),
        "steer_lens_doses": list(LENS_DOSES),
        "layer_medians": medians,
        "selected_unit_count": len(selected_units),
        "selected_unit_keys": [row["unit_key"] for row in selected_units],
        "seed": seed,
    }
    manifest_path = path.with_name(path.name + ".manifest.json")
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("schema", "model_revision", "collection_sha256", "doses", "prompts",
                    "max_new_tokens", "steer_lens_contexts", "steer_lens_doses",
                    "layer_medians", "selected_unit_count", "selected_unit_keys", "device", "seed"):
            if previous.get(key) != manifest.get(key):
                raise SteerError(f"existing manifest differs in {key}; refusing mixed resume")
        return previous
    _atomic_json(manifest_path, manifest)
    return manifest


def pending_units(units: list[dict[str, Any]], finished: set[Any]) -> list[dict[str, Any]]:
    """Return selected units whose atomic JSONL rows are not already complete."""
    return [unit for unit in units if unit.get("unit_key") not in finished]


def _validate_generation_rows(rows: list[dict[str, Any]], layer: int,
                              unit_key: str, label: str) -> None:
    if not rows:
        raise SteerError(f"{unit_key}: {label} generated no rows")
    expected = f"model.layers.{layer}.mlp"
    for row in rows:
        if int(row.get("hook_fired_count", 0)) <= 0:
            raise SteerError(f"{unit_key}: {label} hook fired zero times")
        if row.get("hook_layer_modules") != [expected]:
            raise SteerError(
                f"{unit_key}: {label} hook fired on {row.get('hook_layer_modules')!r}, expected {expected}"
            )


def pilot_check(path: Path, tolerance: float = 1e-4) -> dict[str, Any]:
    """Validate the registered no-op, dose-zero, and matched-norm pilot checks."""
    rows = read_jsonl(path)
    if not rows:
        raise SteerError(f"{path}: pilot-check needs at least one steering row")
    total_positive = 0
    changed_positive = 0
    layer_dose_norms: dict[tuple[int, str], list[float]] = {}
    checked_layers: dict[str, float] = {}
    for row in rows:
        key = row.get("unit_key", "<unknown>")
        layer = row.get("layer")
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise SteerError(f"{key}: pilot row has no integer layer")
        baselines = row.get("baseline")
        zero_rows = row.get("dose_zero")
        rollouts = row.get("rollouts")
        if not isinstance(baselines, list) or not isinstance(zero_rows, list) or not isinstance(rollouts, list):
            raise SteerError(f"{key}: pilot row lacks baseline, dose_zero, or rollouts")
        baseline_by_prompt = {item.get("prompt_key"): item.get("token_ids") for item in baselines}
        zero_by_prompt = {item.get("prompt_key"): item.get("token_ids") for item in zero_rows}
        actual_zero_match = (len(baseline_by_prompt) == len(baselines)
                             and len(zero_by_prompt) == len(zero_rows)
                             and baseline_by_prompt == zero_by_prompt)
        if (row.get("dose_zero_matches_baseline") is not actual_zero_match
                or not actual_zero_match):
            raise SteerError(f"{key}: dose-zero output does not match its independent baseline")
        _validate_generation_rows(zero_rows, layer, key, "dose 0")
        positive_rows = [item for item in rollouts if item.get("dose") == 0.8]
        if len(positive_rows) != len(PROMPTS):
            raise SteerError(f"{key}: expected {len(PROMPTS)} +0.8 rollouts, found {len(positive_rows)}")
        _validate_generation_rows(positive_rows, layer, key, "+0.8 rollout")
        changed = 0
        for rollout in positive_rows:
            differs = rollout.get("token_ids") != baseline_by_prompt.get(rollout.get("prompt_key"))
            if rollout.get("differs_from_baseline") is not differs:
                raise SteerError(f"{key}: recorded +0.8 baseline comparison is inconsistent")
            changed += int(differs)
        if row.get("dose_plus_0_8_changed_count") != changed:
            raise SteerError(f"{key}: recorded +0.8 changed-rollout count is inconsistent")
        total_positive += len(positive_rows)
        changed_positive += changed

        write_norm = float(row.get("write_vector_norm", float("nan")))
        if not math.isfinite(write_norm) or abs(write_norm - 1.0) > tolerance:
            raise SteerError(f"{key}: write vector norm is not one")
        median = float(row.get("median_residual_norm", float("nan")))
        if not math.isfinite(median) or median <= 0:
            raise SteerError(f"{key}: median residual norm is invalid")
        push_norms = row.get("push_norms")
        if not isinstance(push_norms, dict):
            raise SteerError(f"{key}: pilot row has no recorded push norms")
        for dose in DOSES:
            dose_key = f"{dose:+.1f}"
            value = float(push_norms.get(dose_key, float("nan")))
            expected = abs(dose) * median
            if not math.isfinite(value) or abs(value - expected) > tolerance * max(1.0, expected):
                raise SteerError(f"{key}: {dose_key} push norm does not match its registered scale")
            layer_dose_norms.setdefault((layer, dose_key), []).append(value)
        spreads = row.get("equal_norm_spread_by_layer")
        if not isinstance(spreads, dict) or not spreads:
            raise SteerError(f"{key}: pilot row lacks equal push-norm checks")
        for checked_layer, spread in spreads.items():
            spread = float(spread)
            if not math.isfinite(spread) or spread > tolerance:
                raise SteerError(f"{key}: equal push norms failed in layer {checked_layer}: {spread}")
            checked_layers[checked_layer] = max(checked_layers.get(checked_layer, 0.0), spread)

    if total_positive == 0 or changed_positive * 2 < total_positive:
        raise SteerError(
            f"pilot-check failed: only {changed_positive}/{total_positive} +0.8 rollouts differed from baseline"
        )
    for (layer, dose_key), values in layer_dose_norms.items():
        spread = max(values) - min(values)
        # Relative, like the per-row check above: late-layer push norms reach ~2,400, where one float32 step is
        # 0.000244, so an absolute 1e-4 failed the full Part 65 run on rounding alone (max relative spread 2.4e-7).
        if spread > tolerance * max(1.0, max(values)):
            raise SteerError(f"pilot-check failed: {dose_key} push norms differ in layer {layer} by {spread:.8g}")
    return {
        "unit_count": len(rows),
        "plus_0_8_changed": changed_positive,
        "plus_0_8_total": total_positive,
        "dose_zero_matches_baseline": True,
        "equal_norm_spread_by_layer": checked_layers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=None, help="one JSONL row per completed unit")
    parser.add_argument("--pilot-check", type=Path, default=None,
                        help="check a steering output file against the registered pilot gates")
    parser.add_argument("--measure-medians-only", action="store_true",
                        help="measure and save the 2,000-token layer medians, then exit")
    parser.add_argument("--medians", type=Path, default=None)
    parser.add_argument("--model", default=MODEL_ID, help="local HF model id or snapshot path")
    parser.add_argument("--device", choices=("xpu", "cpu", "auto"), default="xpu")
    parser.add_argument("--units", type=int, default=None)
    parser.add_argument("--type", choices=("both", "neuron", "control"), default="both")
    parser.add_argument("--seed", type=int, default=6501)
    args = parser.parse_args(argv)

    if args.pilot_check is not None:
        if args.measure_medians_only:
            parser.error("--pilot-check and --measure-medians-only cannot be combined")
        try:
            summary = pilot_check(args.pilot_check)
        except SteerError as exc:
            print(f"pilot-check failed: {exc}", file=sys.stderr)
            return 2
        print("pilot-check passed: " + json.dumps(summary, sort_keys=True))
        return 0
    if args.measure_medians_only:
        if args.medians is None:
            parser.error("--medians is required with --measure-medians-only")
        torch, tokenizer, model, selected_device = load_local_model(args.model, args.device)
        measured = measure_layer_medians(torch, model, tokenizer, args.corpus)
        _atomic_json(args.medians, measured)
        print(f"layer medians written to {args.medians} (device={selected_device})")
        return 0
    if args.out is None:
        parser.error("--out is required unless --pilot-check is used")

    units = load_units(args.collection)
    selected = select_pilot_units(units, args.units, args.type, args.seed)
    torch, tokenizer, model, selected_device = load_local_model(args.model, args.device)
    medians_path = args.medians or args.out.with_name("layer-medians.json")
    if medians_path.exists():
        medians = load_layer_medians(medians_path)
    else:
        measured = measure_layer_medians(torch, model, tokenizer, args.corpus)
        _atomic_json(medians_path, measured)
        medians = load_layer_medians(medians_path)
    layers = model.model.layers
    norm_check = validate_equal_push_norms(torch, layers, units)
    write_manifest(args.out, args.collection, medians, selected_device, selected, args.seed)
    finished = {row.get("unit_key") for row in read_jsonl(args.out)} if args.out.exists() else set()
    durations = []
    for unit in pending_units(selected, finished):
        if unit["unit_key"] in finished:
            continue
        start = time.perf_counter()
        result = process_unit(torch, model, tokenizer, unit, medians)
        result["equal_norm_spread_by_layer"] = norm_check
        atomic_append_jsonl(args.out, result)
        durations.append(time.perf_counter() - start)
        print(f"{unit['unit_key']} done in {durations[-1]:.2f}s", flush=True)
    if durations:
        print(f"new units={len(durations)} mean_cpu_or_device_seconds={sum(durations) / len(durations):.3f}")
    else:
        print(f"all {len(selected)} selected units already present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
