#!/usr/bin/env python3
"""Part 20: identity replay and context seam measurements for N2742."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time

import torch

import neuron_report as nr

SUMMARY = "Several of the earlier sentences described a visitor from another world arriving among people."
PATTERNS = re.compile(r"\balien\b|extraterrestrial|martian|外星人|UFO pilot|off-world visitor|another planet|nebula|orbit|galaxy|solar wind|pulsar|star formation", re.I)
DEFAULT_PROMPTS = nr.DEFAULT_PROMPTS


def sha(path):
    return nr._file_sha(path) if path and os.path.isfile(path) else None


def make_header(a, input_paths, mode):
    return {
        "model": a.model, "revision": a.revision, "tool": "part20.py",
        "tool_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "inputs": {os.path.basename(p): sha(p) for p in input_paths if p},
        "layer": a.layer, "neuron": a.neuron, "device": a.device,
        "dtype": str(next(a.model_obj.parameters()).dtype), "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_mode": mode,
    }


def output_path(path, name):
    if os.path.isdir(path) or not path.lower().endswith(".json"):
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, name)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    return path


class ReplayHook:
    """Record or replay a unit value for every token position, including cached decode steps."""
    def __init__(self, down_proj, neuron):
        self.neuron = neuron
        self.values = []
        self.replay = False
        self.sabotage = 0.0
        self.position_mode = "all"
        self.handle = down_proj.register_forward_pre_hook(self._hook)

    def _hook(self, _mod, args):
        x = args[0]
        row = x[..., self.neuron].detach().float().cpu().clone()
        if not self.replay:
            self.values.append(row)
            return None
        if not self.values:
            raise RuntimeError("identity replay has no recorded values")
        value = self.values.pop(0).to(device=x.device, dtype=x.dtype) + self.sabotage
        y = x.clone()
        if self.position_mode == "last":
            y[..., -1, self.neuron] = value[..., -1]
        else:
            y[..., self.neuron] = value
        return (y,) + tuple(args[1:])

    def remove(self):
        self.handle.remove()


class CaptureHook:
    def __init__(self, down_proj, neuron):
        self.neuron = neuron
        self.rows = []
        self.handle = down_proj.register_forward_pre_hook(self._hook)

    def _hook(self, _mod, args):
        self.rows.append(args[0][..., self.neuron].detach().float().cpu())

    def remove(self):
        self.handle.remove()


def _encoded(tok, text, device):
    return tok(text, return_tensors="pt", add_special_tokens=False).to(device)


def _eos(tok, model):
    return sorted({i for i in [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
        tok.convert_tokens_to_ids("<|endoftext|>"), getattr(model.generation_config, "eos_token_id", None)]
        if isinstance(i, int) and i >= 0})


def identity_pass(model, tok, down_proj, text, a, clamp_mode="none", sabotage=0.0):
    enc = _encoded(tok, text, a.device)
    hook = ReplayHook(down_proj, a.neuron)
    hook.sabotage = sabotage
    with torch.no_grad():
        base = model(**enc, use_cache=False).logits.detach().float().cpu()
    recorded = torch.cat(hook.values, dim=-1).reshape(1, -1)
    hook.values = [recorded[:, -1:] if clamp_mode == "last" else recorded]
    hook.replay = True
    hook.position_mode = clamp_mode
    if clamp_mode == "none":
        hook.replay = False
    with torch.no_grad():
        changed = model(**enc, use_cache=False).logits.detach().float().cpu()
    hook.remove()
    diffs = (changed - base).abs().amax(dim=-1).squeeze(0).tolist()
    return {"token_ids": enc["input_ids"][0].tolist(), "activations": recorded.squeeze(0).tolist(),
            "logit_max_abs_by_position": diffs if clamp_mode != "none" else None,
            "overall_max_abs_logit_diff": max(diffs, default=0.0) if clamp_mode != "none" else None}


def generate_capture(model, tok, down_proj, text, a, max_new=60, sabotage=0.0):
    enc = _encoded(tok, text, a.device)
    h = CaptureHook(down_proj, a.neuron)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=False, max_new_tokens=max_new, pad_token_id=tok.pad_token_id,
                             eos_token_id=_eos(tok, model))
    h.remove()
    ids = out[0, enc["input_ids"].shape[1]:].tolist()
    # Replay the values the unhooked generation itself computed: the prefill row, then one row per cached decode
    # step. A separate full-sequence pass takes a different kernel path in bf16, so its values are not an identity
    # for the cached steps (host review, 2026-10-18). The last emitted token is never fed back, so it has no row.
    replay_values = [r.clone() for r in h.rows]
    per_step = torch.cat(h.rows[1:], dim=-1) if len(h.rows) > 1 else torch.empty(1, 0)
    rh = ReplayHook(down_proj, a.neuron)
    rh.values = replay_values
    rh.replay = True
    rh.sabotage = sabotage
    with torch.no_grad():
        replay = model.generate(**enc, do_sample=False, max_new_tokens=max_new, pad_token_id=tok.pad_token_id,
                                 eos_token_id=_eos(tok, model))
    rh.remove()
    replay_ids = replay[0, enc["input_ids"].shape[1]:].tolist()
    return {"token_ids": ids, "replay_token_ids": replay_ids, "identical": ids == replay_ids,
            "new_token_activations": per_step.squeeze(0).tolist()}


def cmd_identity(a):
    tok, model = nr.load_model(a.model, a.revision, a.device)
    a.model_obj = model
    _, layers, _ = nr.find_text_layers(model)
    for p in model.parameters():
        p.requires_grad_(False)
    prompts = json.load(open(a.prompts, encoding="utf-8")) if a.prompts else DEFAULT_PROMPTS
    down = layers[a.layer].mlp.down_proj
    results = {}
    for q in prompts:
        text = nr.build_prompt(tok, q, False)
        one = identity_pass(model, tok, down, text, a, "last")
        every = identity_pass(model, tok, down, text, a, "all")
        gen = generate_capture(model, tok, down, text, a, a.max_new)
        results[q] = {"prompt": text, "unhooked": identity_pass(model, tok, down, text, a),
                      "identity_last": one, "identity_every": every, "generation": gen}
        nr.spill_guard(a.device, a.max_gpu_gb)
    out = make_header(a, [__file__, a.prompts], "chat template via neuron_report.build_prompt")
    out.update({"prompts": results, "max_new_tokens": a.max_new})
    nr._dump(out, output_path(a.out, "identity.json"))


def target_spans(text, pattern):
    return [m.span() for m in re.finditer(pattern, text, re.I)]


def token_span(tok, text, span):
    try:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        positions = [i for i, (s, e) in enumerate(offsets) if e > span[0] and s < span[1]]
        if not positions:
            raise ValueError(f"no tokens overlap character span {span}")
        return positions
    except (NotImplementedError, TypeError, KeyError):
        prefix = tok(text[:span[0]], add_special_tokens=False)["input_ids"]
        segment = tok(text[span[0]:span[1]], add_special_tokens=False)["input_ids"]
        return list(range(len(prefix), len(prefix) + len(segment)))


def select_filler(corpus, tok, target_tokens=4000):
    docs, indices, count = [], [], 0
    with open(corpus, encoding="utf-8") as f:
        for index, line in enumerate(f):
            row = json.loads(line)
            text = row["text"]
            if PATTERNS.search(text):
                continue
            n = len(tok(text, add_special_tokens=False)["input_ids"])
            docs.append(text)
            indices.append(index)
            count += n
            if count >= target_tokens:
                break
    return "\n\n".join(docs), indices, len(tok("\n\n".join(docs), add_special_tokens=False)["input_ids"])


def activation_max(model, tok, down, text, span, a):
    ids = _encoded(tok, text, a.device)["input_ids"]
    positions = token_span(tok, text, span)
    hook = CaptureHook(down, a.neuron)
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    hook.remove()
    vals = torch.cat(hook.rows, dim=-1)[0]
    return {"value": float(vals[positions].max()), "token_positions": positions, "span": list(span), "token_count": len(positions)}


def cmd_seam(a):
    tok, model = nr.load_model(a.model, a.revision, a.device)
    a.model_obj = model
    _, layers, _ = nr.find_text_layers(model)
    for p in model.parameters():
        p.requires_grad_(False)
    battery_path = a.battery or os.path.join(os.path.dirname(__file__), "data/sense-battery.jsonl")
    rows = [json.loads(x) for x in open(battery_path, encoding="utf-8")]
    filler, doc_indices, filler_count = select_filler(a.corpus, tok)
    down = layers[a.layer].mlp.down_proj
    results = {"group_a_local": [], "group_a_distant": [], "summary": None, "group_d_local": [], "filler": None}
    for row in rows:
        text = row["text"]
        if row["group"] == "a":
            m = re.search(r"\balien\b", text, re.I)
            span = m.span()
            results["group_a_local"].append({"text": text, **activation_max(model, tok, down, text, span, a)})
            distant = filler + "\n\n" + text
            distant_span = (len(filler) + 2 + span[0], len(filler) + 2 + span[1])
            results["group_a_distant"].append({"text": text, **activation_max(model, tok, down, distant, distant_span, a)})
        elif row["group"] == "d":
            pattern = r"extraterrestrial|Martian|外星人|UFO pilot|off-world visitor|another planet"
            m = re.search(pattern, text, re.I)
            results["group_d_local"].append({"text": text, **activation_max(model, tok, down, text, m.span(), a)})
    sum_text = SUMMARY
    distant_summary = filler + "\n\n" + sum_text
    summary_span = (len(filler) + 2, len(distant_summary))
    results["summary"] = activation_max(model, tok, down, distant_summary, summary_span, a)
    results["filler"] = activation_max(model, tok, down, filler, (0, len(filler)), a)
    la = [x["value"] for x in results["group_a_local"]]
    da = [x["value"] for x in results["group_a_distant"]]
    dd = [x["value"] for x in results["group_d_local"]]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    out = make_header(a, [__file__, battery_path, a.corpus], "plain text; no chat template")
    out.update({"filler_doc_indices": doc_indices, "filler_token_count": filler_count, "filler_target_tokens": 4000,
                "summary_text": SUMMARY, "group_d_chinese_policy": "included; max over the 外星人 target span",
                "rows": results,
                "means": {"group_a_local": mean(la), "group_a_distant": mean(da), "group_d_local": mean(dd)},
                "ratios": {"distant_to_local": mean(da) / mean(la) if mean(la) else None,
                           "summary_to_local": results["summary"]["value"] / mean(la) if mean(la) else None,
                           "summary_below_group_d": results["summary"]["value"] < mean(dd) if mean(dd) is not None else None,
                           "filler_to_local": results["filler"]["value"] / mean(la) if mean(la) else None}})
    nr._dump(out, output_path(a.out, "seam.json"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("identity", "seam"):
        q = sub.add_parser(name)
        q.add_argument("--model", default="Qwen/Qwen3.5-4B")
        q.add_argument("--revision", default=None)
        q.add_argument("--layer", type=int, default=22)
        q.add_argument("--neuron", type=int, default=2742)
        q.add_argument("--device", default="xpu")
        q.add_argument("--out", required=True)
        q.add_argument("--max-gpu-gb", type=float, default=14.5)
        if name == "identity":
            q.add_argument("--prompts", default=None)
            q.add_argument("--max-new", type=int, default=60)
        else:
            q.add_argument("--corpus", required=True)
            q.add_argument("--battery", default=None)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    with torch.no_grad():
        (cmd_identity if a.cmd == "identity" else cmd_seam)(a)


if __name__ == "__main__":
    main()
