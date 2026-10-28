"""Independent observer for neuron-steering runs (Claude, 2026-10-28).

Records what the model actually received, at boundaries the run code does not own:

- every ``layers.<i>.mlp.down_proj`` input, twice per call: in a pre-hook registered with ``prepend=True`` (so it runs
  before any hook a run adds) and inside a wrapper around the module's ``forward`` (so it sees the tensor the matmul
  gets, after every pre-hook). The difference is the intervention, measured per row, per layer, per position;
- every ``model.generate`` call: the arguments that arrived, the effective sampling settings, the RNG state at entry,
  the prompt tokens, the generated tokens and their decoded text;
- every forward hook registered anywhere in the model at generate entry.

It records facts only and holds no expectation about what a run should do. An audit compares these records with the
registered design. The reason it exists: any trace a run writes about itself can echo intent. In Part 85's round-4
review, 7 of 8 wiring mutants passed both the tests and an independent audit, because the trace recorded the seed and
generate arguments the run had built, not the ones that arrived (part85/REVIEW-r4-sol.md, decision
part85-after-r4-2026-10-28).

Use:
    obs = observe.install(model, tokenizer, "results/.../observe.jsonl", label="screen L22")
    ... model.generate(...) ...       # each call writes one JSON line
    call_id = obs.last_call_id        # the run records this beside each answer
    obs.close()

Each generate call's line carries ``rows``: one entry per batch row with the prompt hash, generated-token hash, the hash
of the decoded answer, and ``layers``: for each layer whose down_proj input changed, the measured change (see
``_Acc.rows``). Layers with no change are absent. ``unobserved_forwards`` counts pre-hook calls whose
forward never reached the observer's wrapper (someone replaced ``forward`` after install); it must be empty.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import re
import time
import uuid
from pathlib import Path
from typing import Any

OBSERVER_VERSION = 2
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.down_proj$")
N_PROBES = 64
PROBE_SEED = 20261028
SPARSE_MAX = 16
SAMPLING_KEYS = (
    "do_sample", "temperature", "top_p", "top_k", "min_p", "typical_p", "max_new_tokens", "min_new_tokens",
    "num_beams", "repetition_penalty", "no_repeat_ngram_size", "eos_token_id", "pad_token_id",
)


def self_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def probe_matrix(width: int):
    """Fixed public probes: the audit regenerates the same matrix to compare a dense change with its registered
    direction without storing the whole vector."""
    import torch
    generator = torch.Generator(device="cpu")
    generator.manual_seed(PROBE_SEED)
    raw = torch.randn((N_PROBES, int(width)), generator=generator, dtype=torch.float32)
    return raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True)


def _ids_sha(ids) -> str:
    return hashlib.sha256(json.dumps([int(i) for i in ids], separators=(",", ":")).encode("ascii")).hexdigest()


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    shape = getattr(value, "shape", None)
    return {"type": type(value).__name__, "shape": list(shape) if shape is not None else None}


class _Acc:
    """Measured change for every row of one layer and one phase (prefill or decode) of one generate call.

    Everything stays on the model's device until the call ends, so observing costs no device synchronisation per
    step (version 1 synchronised about fifteen times per row per step and cut batch-32 throughput by two thirds)."""

    def __init__(self, torch, batch: int, width: int, device):
        f = {"device": device, "dtype": torch.float32}
        i = {"device": device, "dtype": torch.long}
        inf = float("inf")
        self.torch = torch
        self.changed_calls = torch.zeros(batch, **i)
        self.changed_positions = torch.zeros(batch, **i)
        self.max_channels = torch.zeros(batch, **i)
        self.union = torch.zeros((batch, width), dtype=torch.bool, device=device)
        self.dmin = torch.full((batch, width), inf, **f)
        self.dmax = torch.full((batch, width), -inf, **f)
        self.pmin = torch.full((batch, width), inf, **f)
        self.pmax = torch.full((batch, width), -inf, **f)
        self.nmin = torch.full((batch,), inf, **f)
        self.nmax = torch.full((batch,), -inf, **f)
        self.wmin = torch.full((batch,), inf, **f)
        self.wmax = torch.full((batch,), -inf, **f)
        self.probe_first = torch.zeros((batch, N_PROBES), **f)
        self.has_first = torch.zeros(batch, dtype=torch.bool, device=device)
        self.probe_last = torch.zeros((batch, N_PROBES), **f)

    def update(self, delta, post, weight, probes):
        torch = self.torch
        inf = float("inf")
        changed = delta != 0                                         # [B, T, W]
        pos_changed = changed.any(dim=-1)                           # [B, T]
        row_changed = pos_changed.any(dim=-1)                       # [B]
        self.changed_calls += row_changed.long()
        self.changed_positions += pos_changed.sum(dim=-1)
        self.max_channels = torch.maximum(self.max_channels, changed.sum(dim=-1).amax(dim=-1))
        self.union |= changed.any(dim=1)
        self.dmin = torch.minimum(self.dmin, torch.where(changed, delta, inf).amin(dim=1))
        self.dmax = torch.maximum(self.dmax, torch.where(changed, delta, -inf).amax(dim=1))
        self.pmin = torch.minimum(self.pmin, torch.where(changed, post, inf).amin(dim=1))
        self.pmax = torch.maximum(self.pmax, torch.where(changed, post, -inf).amax(dim=1))
        dn = torch.linalg.vector_norm(delta, dim=-1)                # [B, T]
        wn = torch.linalg.vector_norm(delta @ weight.T, dim=-1)     # [B, T]
        self.nmin = torch.minimum(self.nmin, torch.where(pos_changed, dn, inf).amin(dim=-1))
        self.nmax = torch.maximum(self.nmax, torch.where(pos_changed, dn, -inf).amax(dim=-1))
        self.wmin = torch.minimum(self.wmin, torch.where(pos_changed, wn, inf).amin(dim=-1))
        self.wmax = torch.maximum(self.wmax, torch.where(pos_changed, wn, -inf).amax(dim=-1))
        steps = int(delta.shape[1])
        index = torch.arange(steps, device=delta.device)
        first = torch.where(pos_changed, index, steps).amin(dim=-1).clamp(max=steps - 1)
        last = torch.where(pos_changed, index, -1).amax(dim=-1).clamp(min=0)
        rows = torch.arange(int(delta.shape[0]), device=delta.device)
        pf = delta[rows, first] @ probes.T
        pl = delta[rows, last] @ probes.T
        new_first = row_changed & ~self.has_first
        self.probe_first = torch.where(new_first.unsqueeze(1), pf, self.probe_first)
        self.has_first |= row_changed
        self.probe_last = torch.where(row_changed.unsqueeze(1), pl, self.probe_last)

    def rows(self, calls: int, positions: int) -> dict[int, dict[str, Any]]:
        """Per-row records for the rows that changed, in the observer's output format."""
        cpu = {name: getattr(self, name).detach().cpu() for name in (
            "changed_calls", "changed_positions", "max_channels", "union", "dmin", "dmax", "pmin", "pmax",
            "nmin", "nmax", "wmin", "wmax", "probe_first", "probe_last")}
        out = {}
        for row in range(int(cpu["changed_calls"].shape[0])):
            if int(cpu["changed_calls"][row]) == 0:
                continue
            channels = self.torch.nonzero(cpu["union"][row], as_tuple=False).flatten().tolist()
            entry = {
                "calls": calls,
                "positions": positions,
                "changed_calls": int(cpu["changed_calls"][row]),
                "changed_positions": int(cpu["changed_positions"][row]),
                "max_channels_per_position": int(cpu["max_channels"][row]),
                "delta_norm": [float(cpu["nmin"][row]), float(cpu["nmax"][row])],
                "write_norm": [float(cpu["wmin"][row]), float(cpu["wmax"][row])],
                "probe_first": cpu["probe_first"][row].tolist(),
                "probe_last": cpu["probe_last"][row].tolist(),
            }
            if len(channels) <= SPARSE_MAX:
                entry["sparse"] = True
                entry["channels"] = {str(c): [float(cpu["dmin"][row, c]), float(cpu["dmax"][row, c]),
                                              float(cpu["pmin"][row, c]), float(cpu["pmax"][row, c])]
                                     for c in channels}
            else:
                entry["sparse"] = False
                entry["distinct_channels_at_least"] = len(channels)
            out[row] = entry
        return out


class Observer:
    def __init__(self, model, tokenizer, path, label: str = ""):
        import torch
        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session = uuid.uuid4().hex
        self.label = label
        self.call_seq = 0
        self.last_call_id: str | None = None
        self.active: dict[str, Any] | None = None
        self.outside_changes: dict[int, int] = {}
        self._before: dict[int, Any] = {}
        self._handles = []
        self._forwards = []
        self._probes: dict[tuple, Any] = {}
        self._weights: dict[int, Any] = {}
        self.modules: dict[int, str] = {}
        for name, module in model.named_modules():
            match = LAYER_RE.search(name)
            if match:
                layer = int(match.group(1))
                if layer in self.modules:
                    raise RuntimeError(f"two down_proj modules claim layer {layer}: {self.modules[layer]} and {name}")
                self.modules[layer] = name
                self._attach(layer, module)
        if not self.modules:
            raise RuntimeError("observer found no layers.<i>.mlp.down_proj modules")
        self._original_generate = model.generate
        self._had_own_generate = "generate" in vars(model)
        model.generate = self._generate
        self._fh = open(self.path, "a", encoding="utf-8")
        import transformers
        first = next(iter(self.modules.values()))
        weight = model.get_submodule(first).weight
        self._write({
            "event": "session",
            "observer_version": OBSERVER_VERSION,
            "observer_sha256": self_sha256(),
            "session": self.session,
            "label": label,
            "pid": os.getpid(),
            "time": time.time(),
            "model_name_or_path": getattr(getattr(model, "config", None), "_name_or_path", None),
            "down_proj_modules": {str(k): v for k, v in sorted(self.modules.items())},
            "dtype": str(weight.dtype),
            "device": str(weight.device),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
        })

    # ---- hooks -----------------------------------------------------------------------------------------------
    def _attach(self, layer: int, module):
        observer = self
        import torch.nn.modules.module as nn_module

        def pre(_module, args, kwargs):
            x = args[0] if args else kwargs.get("input")
            phase = "prefill" if x.ndim == 3 and x.shape[-2] > 1 else "decode"
            call = observer.active
            if call is not None:
                counts = call["phase_counts"].setdefault(layer, {"prefill": [0, 0], "decode": [0, 0]})
                counts[phase][0] += 1
                counts[phase][1] += int(x.shape[-2]) if x.ndim == 3 else 1
                call["pre_calls"][layer] = call["pre_calls"].get(layer, 0) + 1
            # Only another pre-hook on this module (or a global one) can change its input between this hook and
            # forward, so a module with neither is not compared; the census records every hook at generate entry.
            if len(_module._forward_pre_hooks) > 1 or getattr(nn_module, "_global_forward_pre_hooks", None):
                observer._before[layer] = x.detach().clone()
            return None

        self._handles.append(module.register_forward_pre_hook(pre, prepend=True, with_kwargs=True))
        original = module.forward
        had_own = "forward" in vars(module)

        def forward(*args, **kwargs):
            x = args[0] if args else kwargs.get("input")
            call = observer.active
            if call is not None:
                call["forward_calls"][layer] = call["forward_calls"].get(layer, 0) + 1
            before = observer._before.pop(layer, None)
            if before is not None:
                observer._compare(layer, module, before, x.detach())
            return original(*args, **kwargs)

        module.forward = forward
        self._forwards.append((module, original, had_own))

    def _probe(self, width: int, device):
        key = (width, str(device))
        if key not in self._probes:
            self._probes[key] = probe_matrix(width).to(device=device)
        return self._probes[key]

    def _weight(self, layer: int, module):
        if layer not in self._weights:
            self._weights[layer] = module.weight.detach().float()
        return self._weights[layer]

    def _compare(self, layer: int, module, before, after):
        torch = self.torch
        call = self.active
        if before.shape != after.shape:
            if call is not None:
                call["shape_changes"].append({"layer": layer, "before": list(before.shape), "after": list(after.shape)})
            return
        if call is None:
            if not torch.equal(before, after):
                self.outside_changes[layer] = self.outside_changes.get(layer, 0) + 1
            return
        delta = after.float() - before.float()
        post = after.float()
        if delta.ndim == 2:
            delta, post = delta.unsqueeze(1), post.unsqueeze(1)
        phase = "prefill" if delta.shape[1] > 1 else "decode"
        acc = call["acc"].setdefault(layer, {}).get(phase)
        if acc is None:
            acc = _Acc(torch, int(delta.shape[0]), int(delta.shape[-1]), delta.device)
            call["acc"][layer][phase] = acc
        acc.update(delta, post, self._weight(layer, module), self._probe(int(delta.shape[-1]), delta.device))

    # ---- generate --------------------------------------------------------------------------------------------
    def _hook_census(self) -> dict[str, dict[str, int]]:
        own = {h.id for h in self._handles}
        census = {}
        for name, module in self.model.named_modules():
            pre = [k for k in module._forward_pre_hooks if k not in own]
            post = list(module._forward_hooks)
            if pre or post:
                census[name or "<root>"] = {"pre": len(pre), "post": len(post)}
        import torch.nn.modules.module as nn_module
        globals_ = {
            "global_pre": len(getattr(nn_module, "_global_forward_pre_hooks", {})),
            "global_post": len(getattr(nn_module, "_global_forward_hooks", {})),
        }
        return {"modules": census, **globals_}

    def _rng(self) -> dict[str, Any]:
        torch = self.torch
        state = {
            "cpu_initial_seed": int(torch.initial_seed()),
            "cpu_state_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        }
        for name in ("cuda", "xpu"):
            backend = getattr(torch, name, None)
            try:
                if backend is not None and backend.is_available():
                    state[f"{name}_initial_seed"] = int(backend.initial_seed())
                    state[f"{name}_state_hex"] = bytes(backend.get_rng_state().cpu().numpy().tobytes()).hex()
            except Exception as exc:          # recorded, never hidden
                state[f"{name}_error"] = repr(exc)
        return state

    def _effective(self, kwargs) -> dict[str, Any]:
        config = kwargs.get("generation_config") or getattr(self.model, "generation_config", None)
        config = copy.deepcopy(config)
        if config is None:
            return {"error": "no generation_config"}
        updates = {k: v for k, v in kwargs.items() if hasattr(config, k) and k != "generation_config"}
        config.update(**updates)
        return {k: _jsonable(getattr(config, k, None)) for k in SAMPLING_KEYS}

    def _generate(self, *args, **kwargs):
        torch = self.torch
        if self.active is not None:
            raise RuntimeError("observer: nested generate call")
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is None:
            raise RuntimeError("observer: generate called without input_ids")
        mask = kwargs.get("attention_mask")
        self.call_seq += 1
        call_id = f"{self.session}:{self.call_seq}"
        prompts = []
        texts = {}
        ids_cpu = input_ids.detach().cpu()
        mask_cpu = mask.detach().cpu() if mask is not None else None
        for row in range(int(ids_cpu.shape[0])):
            ids = ids_cpu[row].tolist() if mask_cpu is None else ids_cpu[row][mask_cpu[row].bool()].tolist()
            sha = _ids_sha(ids)
            prompts.append(sha)
            if sha not in texts:
                texts[sha] = self.tokenizer.decode(ids, skip_special_tokens=False)
        batch = int(ids_cpu.shape[0])
        self.active = {
            "acc": {},
            "phase_counts": {},
            "pre_calls": {},
            "forward_calls": {},
            "shape_changes": [],
        }
        record = {
            "event": "generate",
            "call_id": call_id,
            "session": self.session,
            "seq": self.call_seq,
            "time_start": time.time(),
            "batch": batch,
            "prompt_len": int(ids_cpu.shape[-1]),
            "attention_mask": mask is not None,
            "explicit_kwargs": {k: _jsonable(v) for k, v in kwargs.items() if k not in ("input_ids", "attention_mask")},
            "effective": self._effective(kwargs),
            "rng_at_entry": self._rng(),
            "hooks_at_entry": self._hook_census(),
            "prompt_texts": texts,
        }
        try:
            out = self._original_generate(*args, **kwargs)
        except BaseException as exc:
            record["error"] = repr(exc)
            self._finish(record, prompts, None)
            raise
        self._finish(record, prompts, out)
        return out

    def _finish(self, record, prompts, out):
        call = self.active
        self.active = None
        sequences = None
        if out is not None:
            sequences = out if hasattr(out, "shape") else getattr(out, "sequences", None)
        eos = record["effective"].get("eos_token_id") if isinstance(record["effective"], dict) else None
        stop = set(eos if isinstance(eos, list) else [eos] if isinstance(eos, int) else [])
        tok_eos = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(tok_eos, int):
            stop.add(tok_eos)
        rows = []
        generated = sequences[:, record["prompt_len"]:].detach().cpu() if sequences is not None else None
        for row in range(record["batch"]):
            entry = {"row": row, "prompt_sha256": prompts[row]}
            if generated is not None:
                ids = generated[row].tolist()
                steps = len(ids)
                for pos, token in enumerate(ids):
                    if int(token) in stop:
                        ids = ids[:pos + 1]
                        break
                text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
                entry.update({
                    "generated_steps": steps,
                    "token_count": len(ids),
                    "token_ids_sha256": _ids_sha(ids),
                    "answer_sha256": _text_sha(text),
                })
            entry["layers"] = {}
            rows.append(entry)
        for layer, phases in sorted(call["acc"].items()):
            counts = call["phase_counts"].get(layer, {"prefill": [0, 0], "decode": [0, 0]})
            for phase, acc in phases.items():
                for row, stats in acc.rows(counts[phase][0], counts[phase][1]).items():
                    rows[row]["layers"].setdefault(str(layer), {})[phase] = stats
        record["rows"] = rows
        record["phase_counts"] = {str(k): v for k, v in sorted(call["phase_counts"].items())}
        record["shape_changes"] = call["shape_changes"]
        record["unobserved_forwards"] = {str(k): v - call["forward_calls"].get(k, 0)
                                         for k, v in sorted(call["pre_calls"].items())
                                         if v != call["forward_calls"].get(k, 0)}
        record["rng_at_exit"] = self._rng()
        record["time_end"] = time.time()
        self._write(record)
        self.last_call_id = record["call_id"]

    # ---- output ----------------------------------------------------------------------------------------------
    def _write(self, record):
        self._fh.write(json.dumps(record, separators=(",", ":"), allow_nan=True) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self):
        if self._fh is None:
            return
        self._write({"event": "close", "session": self.session, "time": time.time(), "calls": self.call_seq,
                     "outside_generate_changes": {str(k): v for k, v in sorted(self.outside_changes.items())}})
        for handle in self._handles:
            handle.remove()
        for module, original, had_own in self._forwards:
            if had_own:
                module.forward = original
            else:
                vars(module).pop("forward", None)
        if self._had_own_generate:
            self.model.generate = self._original_generate
        else:
            vars(self.model).pop("generate", None)
        self._fh.close()
        self._fh = None


def install(model, tokenizer, path, label: str = "") -> Observer:
    return Observer(model, tokenizer, path, label)


def read(path) -> list[dict[str, Any]]:
    """All records in an observer file, in order."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: unreadable observer record") from exc
    return out
