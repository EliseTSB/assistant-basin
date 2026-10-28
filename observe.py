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
``_RowStats``). Layers with no change are absent.
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

OBSERVER_VERSION = 1
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


class _RowStats:
    """Measured change for one row, one layer, one phase (prefill or decode) of one generate call."""

    def __init__(self):
        self.changed_calls = 0
        self.changed_positions = 0
        self.max_channels_per_position = 0
        self.channels: dict[int, list[float]] | None = {}   # channel -> [delta min, delta max, post min, post max]
        self.dense_count = 0                                 # distinct channels once past SPARSE_MAX
        self.delta_norm = [float("inf"), float("-inf")]
        self.write_norm = [float("inf"), float("-inf")]
        self.probe_first = None
        self.probe_last = None

    def as_dict(self, calls: int, positions: int) -> dict[str, Any]:
        out = {
            "calls": calls,
            "positions": positions,
            "changed_calls": self.changed_calls,
            "changed_positions": self.changed_positions,
            "max_channels_per_position": self.max_channels_per_position,
            "delta_norm": self.delta_norm,
            "write_norm": self.write_norm,
            "probe_first": self.probe_first,
            "probe_last": self.probe_last,
        }
        if self.channels is not None:
            out["sparse"] = True
            out["channels"] = {str(c): v for c, v in sorted(self.channels.items())}
        else:
            out["sparse"] = False
            out["distinct_channels_at_least"] = self.dense_count
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
        self._probes: dict[int, Any] = {}
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

        def pre(_module, args, kwargs):
            x = args[0] if args else kwargs.get("input")
            observer._before[layer] = x.detach().clone()
            return None

        self._handles.append(module.register_forward_pre_hook(pre, prepend=True, with_kwargs=True))
        original = module.forward
        had_own = "forward" in vars(module)

        def forward(*args, **kwargs):
            x = args[0] if args else kwargs.get("input")
            before = observer._before.pop(layer, None)
            if before is None:
                raise RuntimeError(f"observer pre-hook did not run before L{layer} down_proj forward")
            observer._compare(layer, module, before, x.detach())
            return original(*args, **kwargs)

        module.forward = forward
        self._forwards.append((module, original, had_own))

    def _probe(self, width: int):
        if width not in self._probes:
            self._probes[width] = probe_matrix(width)
        return self._probes[width]

    def _compare(self, layer: int, module, before, after):
        torch = self.torch
        phase = "prefill" if after.ndim == 3 and after.shape[-2] > 1 else "decode"
        call = self.active
        if call is not None:
            counts = call["phase_counts"].setdefault(layer, {"prefill": [0, 0], "decode": [0, 0]})
            counts[phase][0] += 1
            counts[phase][1] += int(after.shape[-2]) if after.ndim == 3 else 1
        if before.shape == after.shape and torch.equal(before, after):
            return
        if call is None:
            self.outside_changes[layer] = self.outside_changes.get(layer, 0) + 1
            return
        if before.shape != after.shape:
            call["shape_changes"].append({"layer": layer, "before": list(before.shape), "after": list(after.shape)})
            return
        delta = after.float() - before.float()                       # [B, T, W]
        if delta.ndim == 2:
            delta = delta.unsqueeze(1)
            post = after.float().unsqueeze(1)
        else:
            post = after.float()
        changed = delta != 0
        pos_changed = changed.any(dim=-1)                           # [B, T]
        rows_changed = pos_changed.any(dim=-1).cpu().tolist()
        weight = module.weight.detach().float()
        probes = self._probe(int(delta.shape[-1])).to(device=delta.device)
        for row, did in enumerate(rows_changed):
            if not did:
                continue
            stats = call["rows"][row].setdefault(layer, {}).setdefault(phase, _RowStats())
            stats.changed_calls += 1
            positions = torch.nonzero(pos_changed[row], as_tuple=False).flatten()
            stats.changed_positions += int(positions.numel())
            row_delta = delta[row].index_select(0, positions)         # [P, W]
            row_post = post[row].index_select(0, positions)
            row_changed = changed[row].index_select(0, positions)
            per_pos = row_changed.sum(dim=-1)
            stats.max_channels_per_position = max(stats.max_channels_per_position, int(per_pos.max().item()))
            dnorm = torch.linalg.vector_norm(row_delta, dim=-1)
            wnorm = torch.linalg.vector_norm(row_delta @ weight.T, dim=-1)
            stats.delta_norm = [min(stats.delta_norm[0], float(dnorm.min())), max(stats.delta_norm[1], float(dnorm.max()))]
            stats.write_norm = [min(stats.write_norm[0], float(wnorm.min())), max(stats.write_norm[1], float(wnorm.max()))]
            proj_first = (row_delta[0] @ probes.T).cpu().tolist()
            proj_last = (row_delta[-1] @ probes.T).cpu().tolist()
            if stats.probe_first is None:
                stats.probe_first = proj_first
            stats.probe_last = proj_last
            channels = torch.nonzero(row_changed.any(dim=0), as_tuple=False).flatten().cpu().tolist()
            if stats.channels is not None:
                merged = set(stats.channels) | set(channels)
                if len(merged) > SPARSE_MAX:
                    stats.dense_count = len(merged)
                    stats.channels = None
                else:
                    for channel in channels:
                        mask = row_changed[:, channel]
                        d = row_delta[:, channel][mask]
                        p = row_post[:, channel][mask]
                        values = [float(d.min()), float(d.max()), float(p.min()), float(p.max())]
                        old = stats.channels.get(channel)
                        if old is None:
                            stats.channels[channel] = values
                        else:
                            stats.channels[channel] = [min(old[0], values[0]), max(old[1], values[1]),
                                                       min(old[2], values[2]), max(old[3], values[3])]
            else:
                stats.dense_count = max(stats.dense_count, len(channels))

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
            "rows": [dict() for _ in range(batch)],
            "phase_counts": {},
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
            layers = {}
            for layer, phases in sorted(call["rows"][row].items()):
                counts = call["phase_counts"].get(layer, {"prefill": [0, 0], "decode": [0, 0]})
                layers[str(layer)] = {ph: st.as_dict(counts[ph][0], counts[ph][1]) for ph, st in phases.items()}
            entry["layers"] = layers
            rows.append(entry)
        record["rows"] = rows
        record["phase_counts"] = {str(k): v for k, v in sorted(call["phase_counts"].items())}
        record["shape_changes"] = call["shape_changes"]
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
                del module.forward
        if self._had_own_generate:
            self.model.generate = self._original_generate
        else:
            del self.model.generate
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
