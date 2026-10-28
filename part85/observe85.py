#!/usr/bin/env python3
"""observe85.py -- Part 85's observer: the shared generate observer (../observe.py, unchanged) plus two records it lacks.

Written by Claude (Opus 5.5), 2026-10-28, for decision part85-after-r4-2026-10-28 option 1. ``observe.py`` stays
byte-identical because Part 86's staged run hashes it; this module subclasses its Observer and adds:

- ``raw`` on every generate record: per batch row, the prompt token IDs that arrived at ``model.generate`` and the full
  generated token IDs it returned (before any stop-token cut), so an audit can re-render the prompt independently and
  bind the answer text to the tokens;
- ``forward`` records for every ``model(...)`` call made outside ``generate`` (Part 85 step 1): the input IDs that
  arrived, and for each watched layer and unit the values the down_proj matmul actually received (read in a forward
  hook, after every pre-hook), plus whether any pre-hook changed that layer's input (compared with a copy taken in a
  pre-hook registered with ``prepend=True``, before any hook the run adds).

It records facts only. ``audit85.py`` compares them with the design.
"""
from __future__ import annotations

import functools
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import observe  # noqa: E402  (the repository-level shared observer, version 3)

OBSERVE85_VERSION = 1


def self_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class Observer85(observe.Observer):
    def __init__(self, model, tokenizer, path, label: str = "", watch: dict[int, list[int]] | None = None):
        self._raw_prompts = None
        self._forward_seq = 0
        self._fwd = None
        self._own85 = []
        super().__init__(model, tokenizer, path, label)
        self.watch = {int(k): [int(u) for u in v] for k, v in (watch or {}).items()}
        for layer in self.watch:
            if layer not in self.modules:
                raise RuntimeError(f"observe85: watched layer {layer} has no down_proj module")
            self._attach85(layer, model.get_submodule(self.modules[layer]))
        self._original_forward = model.forward
        self._had_own_forward = "forward" in vars(model)
        # functools.wraps keeps inspect.signature(model.forward) truthful: transformers inspects it inside generate.
        observer = self

        @functools.wraps(self._original_forward)
        def forward(*args, **kwargs):
            return observer._forward(*args, **kwargs)

        model.forward = forward
        self._write({"event": "session85", "session": self.session, "observe85_version": OBSERVE85_VERSION,
                     "observe85_sha256": self_sha256(), "observe_sha256": observe.self_sha256(),
                     "watch": {str(k): v for k, v in sorted(self.watch.items())}})

    # ---- forward passes outside generate ---------------------------------------------------------------------
    def _attach85(self, layer: int, module):
        observer = self

        def pre(_module, args, kwargs):
            if observer._fwd is not None:
                x = args[0] if args else kwargs.get("input")
                observer._fwd["before"][layer] = x.detach().clone()
            return None

        def post(_module, args, kwargs, _output):
            if observer._fwd is None:
                return None
            x = args[0] if args else kwargs.get("input")
            before = observer._fwd["before"].pop(layer, None)
            entry = observer._fwd["layers"].setdefault(str(layer), {"calls": 0, "changed_by_hooks": False,
                                                                     "values": {}})
            entry["calls"] += 1
            if before is None or before.shape != x.shape or not observer.torch.equal(before, x):
                entry["changed_by_hooks"] = True
            for unit in observer.watch[layer]:
                vals = x.detach()[..., unit].float().cpu()
                entry["values"].setdefault(str(unit), []).append(vals.tolist())
            return None

        self._own85.append(module.register_forward_pre_hook(pre, prepend=True, with_kwargs=True))
        self._own85.append(module.register_forward_hook(post, with_kwargs=True))

    def _forward(self, *args, **kwargs):
        if self.active is not None or self._fwd is not None:
            return self._original_forward(*args, **kwargs)
        ids = kwargs.get("input_ids", args[0] if args else None)
        self._forward_seq += 1
        self._fwd = {"before": {}, "layers": {}}
        record = {"event": "forward", "session": self.session, "seq": self._forward_seq, "time": time.time(),
                  "kwargs": sorted(k for k in kwargs if k != "input_ids"), "positional_args": len(args),
                  "hooks_at_entry": self._hook_census()}
        if ids is not None and hasattr(ids, "detach"):
            record["input_ids"] = ids.detach().cpu().tolist()
        try:
            out = self._original_forward(*args, **kwargs)
        except BaseException as exc:
            record["error"] = repr(exc)
            raise
        finally:
            record["layers"] = self._fwd["layers"]
            self._fwd = None
            self._write(record)
        return out

    def _hook_census(self):
        """The shared census, excluding this module's own pre- and post-hooks as well as the base observer's."""
        own = {h.id for h in self._handles} | {h.id for h in self._own85}
        census = {}
        for name, module in self.model.named_modules():
            pre = [k for k in module._forward_pre_hooks if k not in own]
            post = [k for k in module._forward_hooks if k not in own]
            if pre or post:
                census[name or "<root>"] = {"pre": len(pre), "post": len(post)}
        import torch.nn.modules.module as nn_module
        return {"modules": census,
                "global_pre": len(getattr(nn_module, "_global_forward_pre_hooks", {})),
                "global_post": len(getattr(nn_module, "_global_forward_hooks", {}))}

    # ---- generate: raw IDs -----------------------------------------------------------------------------------
    def _generate(self, *args, **kwargs):
        ids = kwargs.get("input_ids", kwargs.get("inputs", args[0] if args else None))
        mask = kwargs.get("attention_mask")
        prompts = []
        if ids is not None and hasattr(ids, "detach"):
            ids_cpu = ids.detach().cpu()
            mask_cpu = mask.detach().cpu() if mask is not None else None
            for row in range(int(ids_cpu.shape[0])):
                prompts.append(ids_cpu[row].tolist() if mask_cpu is None
                               else ids_cpu[row][mask_cpu[row].bool()].tolist())
        self._raw_prompts = prompts
        return super()._generate(*args, **kwargs)

    def _finish(self, record, prompts, out):
        sequences = None
        if out is not None:
            sequences = out if hasattr(out, "shape") else getattr(out, "sequences", None)
        raw = []
        for row in range(record["batch"]):
            entry = {"row": row, "prompt_ids": (self._raw_prompts or [None] * record["batch"])[row]}
            if sequences is not None:
                entry["generated_ids"] = sequences[row, record["prompt_len"]:].detach().cpu().tolist()
            raw.append(entry)
        record["raw"] = raw
        self._raw_prompts = None
        return super()._finish(record, prompts, out)

    def close(self):
        if self._fh is None:
            return
        for handle in self._own85:
            handle.remove()
        if self._had_own_forward:
            self.model.forward = self._original_forward
        else:
            vars(self.model).pop("forward", None)
        super().close()


def install(model, tokenizer, path, label: str = "", watch=None) -> Observer85:
    return Observer85(model, tokenizer, path, label, watch)


read = observe.read
