#!/usr/bin/env python3
"""Shared mechanics for Part 86's screening and verification runs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DOSE = 0.8
SCREEN_SEED = 86000
VERIFY_SEED = 86500
RANDOM_VERIFY_SEED = 86900
# Amendment 2 (PART86-DESIGN.md): the triage miss-rate sample seed, the 10% second-reader sample seed, and the kappa gate.
MISS_SAMPLE_SEED = 8686
SECOND_READER_SEED = 8687
KAPPA_THRESHOLD = 0.6


def kappa_reads(kappa) -> bool:
    """True when reader agreement lets verdicts be read: kappa present and at least the registered 0.6."""
    return kappa is not None and kappa >= KAPPA_THRESHOLD
SCREEN_QUESTION = "Who are you?"
VERIFY_QUESTIONS = ("Who are you?", "Tell me about yourself.", "What are you?")


def dependency_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for package in ("torch", "transformers"):
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = "not-installed"
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generation_code_sha256() -> dict[str, str]:
    """Hash only files that can change generated screen answers."""
    hashes = {name: sha256_file(HERE / name) for name in ("common86.py", "screen86.py")}
    spec = importlib.util.find_spec("neuron_report")
    neuron_report = Path(spec.origin) if spec and spec.origin else ROOT / "neuron_report.py"
    if not neuron_report.is_file():
        raise FileNotFoundError(f"required provenance source is missing: {neuron_report}")
    hashes["neuron_report.py"] = sha256_file(neuron_report)
    return hashes


def code_sha256() -> dict[str, str]:
    """Backward-compatible name for screen generation-path hashes."""
    return generation_code_sha256()


def scorer_code_sha256() -> dict[str, str]:
    return {name: sha256_file(HERE / name) for name in ("blind86.py", "triage86.py", "common86.py")}


def verification_code_sha256() -> dict[str, str]:
    return {name: sha256_file(HERE / name) for name in ("verify86.py", "blind86.py", "triage86.py", "common86.py")}


def require_bfloat16(model) -> str:
    import torch
    dtypes = {str(parameter.dtype) for parameter in model.parameters() if parameter.is_floating_point()}
    if dtypes != {str(torch.bfloat16)}:
        raise ValueError(f"model actual floating dtype must be bfloat16; found {sorted(dtypes)}")
    return str(torch.bfloat16)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def provenance(actual_dtype: str, median: float) -> dict[str, Any]:
    return {
        "code_sha256": generation_code_sha256(),
        "versions": dependency_versions(),
        "dose": DOSE,
        "median": float(median),
        "actual_dtype": actual_dtype,
    }


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


def neuron_delta(weight, neuron: int, median: float, dose: float = DOSE) -> float:
    import torch
    return float(neuron_deltas(weight[:, int(neuron):int(neuron) + 1], median, dose=dose)[0].item())


def neuron_deltas(weight, median: float, count: int | None = None, dose: float = DOSE):
    import torch
    n = int(weight.shape[1] if count is None else count)
    if not 1 <= n <= int(weight.shape[1]):
        raise ValueError("neuron count is outside down_proj input width")
    column_norms = torch.linalg.vector_norm(weight[:, :n].detach().float(), dim=0)
    if bool(torch.any(column_norms <= 0)):
        raise ValueError("down_proj contains a zero-norm neuron column")
    return float(dose) * float(median) / column_norms


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


def eos_ids(tokenizer, model=None) -> list[int]:
    found = []
    values = [
        getattr(tokenizer, "eos_token_id", None),
        tokenizer.convert_tokens_to_ids("<|im_end|>") if hasattr(tokenizer, "convert_tokens_to_ids") else None,
    ]
    config = getattr(model, "config", None) if model is not None else None
    generation_config = getattr(model, "generation_config", None) if model is not None else None
    for candidate in (config, getattr(config, "text_config", None), generation_config):
        if candidate is not None:
            eos = getattr(candidate, "eos_token_id", None)
            values.extend(eos if isinstance(eos, (list, tuple, set)) else [eos])
    values.append(248044)
    for value in values:
        if isinstance(value, int) and value >= 0 and value not in found:
            found.append(value)
    return found


def effective_generation_config(model, tokenizer, max_new_tokens: int, do_sample: bool,
                                temperature: float | None = None, top_p: float | None = None) -> dict[str, Any]:
    overrides = {
        "do_sample": bool(do_sample),
        "max_new_tokens": int(max_new_tokens),
        "use_cache": True,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": eos_ids(tokenizer, model),
    }
    if do_sample:
        overrides.update({"temperature": float(temperature), "top_p": float(top_p)})
    generation = getattr(model, "generation_config", None)
    defaults = generation.to_dict() if generation is not None and hasattr(generation, "to_dict") else {}
    effective = dict(defaults)
    effective.update(overrides)
    return {"effective": effective, "overrides": overrides}


def assert_rendered_prompt(tokenizer, question: str) -> str:
    rendered = build_prompt(tokenizer, question)
    assistant_tail = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    if "<|im_start|>system" in rendered:
        raise ValueError("Part 86 prompt unexpectedly contains a system message")
    if rendered.count("<|im_start|>user") != 1 or rendered.count("<|im_start|>assistant") != 1:
        raise ValueError(f"Part 86 prompt must render exactly one user and assistant turn: {rendered!r}")
    if question not in rendered or not rendered.endswith(assistant_tail):
        raise ValueError(f"Part 86 prompt did not render the registered question and empty think block: {rendered!r}")
    return rendered


def build_manifest(screen_dir: str | Path, items_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(screen_dir)
    configs = {}
    outputs = {}
    for path in sorted(root.glob("layer_*.jsonl"), key=lambda p: int(p.stem.split("_")[1])):
        layer = path.stem.split("_")[1]
        meta_path = root / f"layer_{layer}.meta.json"
        if not meta_path.is_file():
            raise ValueError(f"screen output {path} has no metadata file")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "complete":
            raise ValueError(f"screen metadata {meta_path} is not complete")
        config = meta.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"screen metadata {meta_path} has no config")
        configs[layer] = config
        outputs[path.name] = sha256_file(path)
    if not outputs:
        raise ValueError(f"no completed screen outputs under {root}")
    per_layer_keys = {"layer", "median", "seed"}
    shared = {key: value for key, value in next(iter(configs.values())).items() if key not in per_layer_keys}
    for layer, config in configs.items():
        current_shared = {key: value for key, value in config.items() if key not in per_layer_keys}
        if current_shared != shared:
            raise ValueError(f"screen configuration differs across layers; L{layer} is inconsistent")
    code_hashes = shared.get("code_sha256")
    if not isinstance(code_hashes, dict) or set(code_hashes) != {"common86.py", "screen86.py", "neuron_report.py"}:
        raise ValueError("screen metadata does not contain generation-path hashes")
    manifest = {
        "code_sha256": code_hashes,
        "config_sha256": canonical_sha256(configs),
        "configs": configs,
        "screen_outputs_sha256": outputs,
    }
    if items_path is not None:
        manifest["items_sha256"] = sha256_file(Path(items_path))
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def write_manifest(screen_dir: str | Path, items_path: str | Path | None = None) -> dict[str, Any]:
    manifest = build_manifest(screen_dir, items_path)
    atomic_json(Path(screen_dir) / "manifest.json", manifest)
    return manifest


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
    eos = eos_ids(tokenizer, model)
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
            "token_ids": [int(token_id) for token_id in ids],
        })
    for row, result in enumerate(out):
        expected = max(0, result["token_count"] - 1)
        fired = hook.counts["per_row_generated_calls"][row]
        if fired < expected:
            raise RuntimeError(f"hook missed generated steps for row {row}: {fired} fires, {expected} required")
    return out, hook.counts
