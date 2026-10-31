#!/usr/bin/env python3
"""Collect Part 133 pairwise OLMo 2 MLP activation sufficient statistics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = ("parent", "sft", "dpo", "instruct")
PAIRS = (("parent", "sft"), ("sft", "dpo"), ("dpo", "instruct"), ("parent", "instruct"))
GROUPS = ("web", "chat-user", "chat-assistant")
SAVE_EVERY = 500
SEQ_LEN = 256


def atomic_save(path: Path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def role_markers(tok):
    """Read role-token ids by rendering a real user/assistant example with the cached template."""
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "ROLE_CHECK_USER"},
         {"role": "assistant", "content": "ROLE_CHECK_ASSISTANT"}],
        tokenize=False, add_generation_prompt=False)
    ids = tok(rendered, add_special_tokens=False)["input_ids"]
    # Read each complete role header from the rendered text, then encode that header.
    role_ids = {}
    for role, sentinel in (("user", "ROLE_CHECK_USER"), ("assistant", "ROLE_CHECK_ASSISTANT")):
        at = rendered.index(sentinel)
        role_at = rendered.rfind(role, 0, at)
        marker_start = rendered.rfind("<", 0, role_at)
        line_end = rendered.find("\n", role_at)
        header = rendered[marker_start:line_end + 1] if marker_start >= 0 and line_end >= 0 else ""
        if role not in header:
            raise ValueError(f"rendered OLMo template did not expose the {role!r} role")
        encoded = tok(header, add_special_tokens=False)["input_ids"]
        if not encoded:
            raise ValueError(f"could not tokenize rendered {role!r} header")
        role_ids[role] = tuple(int(x) for x in encoded)
    # Keep details in metadata for an auditable actual-template check.
    return role_ids, rendered


def role_masks(ids, valid, role_ids):
    """Assign message tokens by actual rendered role headers, including their content."""
    all_roles = torch.zeros_like(valid, dtype=torch.long)
    for bi in range(ids.shape[0]):
        starts = []
        for role, seq in role_ids.items():
            width = len(seq)
            if width == 0: continue
            for begin in range(max(0, ids.shape[1] - width + 1)):
                if bool(valid[bi, begin:begin+width].all()) and tuple(int(v) for v in ids[bi, begin:begin+width].tolist()) == seq:
                    starts.append((begin, role))
        starts.sort()
        for ix, (begin, role) in enumerate(starts):
            end = starts[ix + 1][0] if ix + 1 < len(starts) else int(valid[bi].sum())
            all_roles[bi, begin:end] = 1 if role == "user" else 2
    return {"web": torch.zeros_like(valid, dtype=torch.bool),
            "chat-user": valid & all_roles.eq(1),
            "chat-assistant": valid & all_roles.eq(2)}


def sufficient(x, y, keep, permutation=None):
    """Return float32 device sums [5,width] and count, vectorized over tokens."""
    a, b = x[keep].float(), y[keep].float()
    return torch.stack((a.sum(0), b.sum(0), a.square().sum(0),
                        b.square().sum(0), (a * b).sum(0))), keep.sum()


def collate_docs(path: Path, limit=None):
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        ids = [int(x) for x in row["input_ids"][:SEQ_LEN]]
        if len(ids) > 1:
            rows.append({**row, "input_ids": ids, "row": line_no})
    if limit is not None and limit < len(rows):
        # Preserve the registered 4:1 web/chat mix in the 200-document pilot.
        web = [r for r in rows if r.get("source") == "web"]
        chat = [r for r in rows if r.get("source") == "chat"]
        n_web = min(len(web), round(limit * .8)); n_chat = min(len(chat), limit - n_web)
        picked = web[:n_web] + chat[:n_chat]
        if len(picked) < limit:
            picked += [r for r in rows if r not in picked][:limit-len(picked)]
        rows = picked
    return rows


def load_models(paths, device):
    models = {}
    for name in MODELS:
        model = AutoModelForCausalLM.from_pretrained(
            paths[name], torch_dtype=torch.bfloat16, local_files_only=True,
            device_map={"": str(device)})
        model.eval()
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise TypeError(f"{name} checkpoint is not the expected OLMo 2 model class")
        for li, layer in enumerate(model.model.layers):
            if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "down_proj"):
                raise AttributeError(f"layer {li} has no mlp.down_proj")
        models[name] = model
    layouts = {(int(m.config.num_hidden_layers), int(m.config.intermediate_size)) for m in models.values()}
    if len(layouts) != 1:
        raise ValueError("checkpoint layer counts or MLP widths differ")
    return models


def run(args):
    device = torch.device(args.device)
    prepared = json.loads(Path(args.docs).read_text(encoding="utf-8"))
    docs_path = Path(prepared["documents"])
    if not docs_path.is_absolute(): docs_path = Path(args.docs).resolve().parent / docs_path
    docs = collate_docs(docs_path, args.limit)
    tok = AutoTokenizer.from_pretrained(args.instruct, local_files_only=True)
    role_ids, rendered = role_markers(tok)
    paths = {"parent": args.parent, "sft": args.sft, "dpo": args.dpo, "instruct": args.instruct}
    revisions = {name: str(Path(p).resolve()) for name, p in paths.items()}
    models = load_models(paths, device)
    layers, width = next(iter(models.values())).config.num_hidden_layers, next(iter(models.values())).config.intermediate_size
    # group,pair,layer,statistic,neuron. CPU totals are float64; device batch sums stay float32.
    totals = torch.zeros((len(GROUPS), len(PAIRS), layers, 5, width), dtype=torch.float64)
    counts = torch.zeros((len(GROUPS), len(PAIRS), layers), dtype=torch.int64)
    active = torch.zeros((len(GROUPS), len(PAIRS), layers, 2, width), dtype=torch.int64)
    null_totals = torch.zeros((layers, 5, width), dtype=torch.float64)
    null_counts = torch.zeros((layers,), dtype=torch.int64)
    self_totals = torch.zeros((layers, 5, width), dtype=torch.float64)
    self_counts = torch.zeros((layers,), dtype=torch.int64)
    meta = {"stage": args.stage, "docs_sha256": hashlib.sha256(docs_path.read_bytes()).hexdigest(),
            "docs": len(docs), "batch": args.batch, "models": revisions,
            "role_ids": {k: list(v) for k, v in role_ids.items()}, "rendered_example": rendered}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    save_path = out / f"{args.stage}-sums.pt"
    pos = 0
    if save_path.is_file():
        saved = torch.load(save_path, map_location="cpu", weights_only=False)
        if saved["meta"] != meta:
            raise ValueError("checkpoint metadata differs; refusing unsafe resume")
        totals.copy_(saved["totals"]); counts.copy_(saved["counts"]); active.copy_(saved["active"])
        null_totals.copy_(saved["null_totals"]); null_counts.copy_(saved["null_counts"])
        self_totals.copy_(saved["self_totals"]); self_counts.copy_(saved["self_counts"])
        pos = int(saved["next_doc"])
    rng = torch.Generator(device="cpu").manual_seed(133)
    permutations = []
    for _ in range(layers):
        order = torch.randperm(width, generator=rng)
        permutation = torch.empty_like(order); permutation[order] = order.roll(1)
        permutations.append(permutation.to(device))
    started = time.monotonic()
    while pos < len(docs):
        end = min(pos + args.batch, ((pos // SAVE_EVERY) + 1) * SAVE_EVERY, len(docs))
        chunk = docs[pos:end]
        lengths = [len(d["input_ids"]) for d in chunk]
        seq = max(lengths)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ids = torch.full((len(chunk), seq), int(pad), dtype=torch.long)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        for j, doc in enumerate(chunk):
            n = len(doc["input_ids"]); ids[j, :n] = torch.tensor(doc["input_ids"]); valid[j, :n] = True
        group_masks = role_masks(ids.to(device), valid.to(device), role_ids)
        web_rows = torch.tensor([d.get("source") == "web" for d in chunk], device=device, dtype=torch.bool)
        chat_rows = ~web_rows
        group_masks["web"] = valid.to(device) & web_rows[:, None]
        group_masks["chat-user"] &= chat_rows[:, None]
        group_masks["chat-assistant"] &= chat_rows[:, None]
        captured = {}
        handles = []
        for name, model in models.items():
            captures = [None] * layers
            for li, layer in enumerate(model.model.layers):
                def hook(_module, inputs, index=li, bucket=captures):
                    bucket[index] = inputs[0].detach()
                handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
            with torch.inference_mode():
                model(input_ids=ids.to(device), attention_mask=valid.to(device, dtype=torch.long), use_cache=False)
            for h in handles[-layers:]:
                h.remove()
            handles = handles[:-layers]
            if any(x is None for x in captures):
                raise RuntimeError(f"{name} forward did not call every down_proj hook")
            captured[name] = captures
        batch_sums = torch.zeros((len(GROUPS), len(PAIRS), layers, 5, width), dtype=torch.float32, device=device)
        batch_counts = torch.zeros((len(GROUPS), len(PAIRS), layers), dtype=torch.int64, device=device)
        batch_active = torch.zeros((len(GROUPS), len(PAIRS), layers, 2, width), dtype=torch.int64, device=device)
        for pi, (left, right) in enumerate(PAIRS):
            for li in range(layers):
                x, y = captured[left][li][:, 1:], captured[right][li][:, 1:]
                for gi, group in enumerate(GROUPS):
                    keep = group_masks[group][:, 1:]
                    a, b = x.reshape(-1, width)[keep.reshape(-1)].float(), y.reshape(-1, width)[keep.reshape(-1)].float()
                    batch_sums[gi, pi, li] = torch.stack((a.sum(0), b.sum(0), a.square().sum(0), b.square().sum(0), (a*b).sum(0)))
                    batch_counts[gi, pi, li] = keep.sum()
                    batch_active[gi, pi, li, 0] = a.ne(0).sum(0)
                    batch_active[gi, pi, li, 1] = b.ne(0).sum(0)
        # Only compact batch reductions move to the CPU; accumulated totals are float64 there.
        totals += batch_sums.cpu().to(torch.float64); counts += batch_counts.cpu(); active += batch_active.cpu()
        batch_null = torch.zeros((layers, 5, width), dtype=torch.float32, device=device)
        batch_null_count = torch.zeros((layers,), dtype=torch.int64, device=device)
        for li in range(layers):
            x, y = captured["parent"][li][:, 1:], captured["instruct"][li][:, 1:]
            keep = group_masks["chat-assistant"][:, 1:].reshape(-1)
            stats, n = sufficient(x.reshape(-1, width)[:, permutations[li]], y.reshape(-1, width), keep)
            batch_null[li] = stats; batch_null_count[li] = n
        null_totals += batch_null.cpu().to(torch.float64); null_counts += batch_null_count.cpu()
        # Gate 1: same Instruct rows, reversed batch order and full-length padding.
        alt_docs = list(reversed(chunk)); alt_ids = torch.full((len(chunk), SEQ_LEN), int(pad), dtype=torch.long)
        alt_valid = torch.zeros_like(alt_ids, dtype=torch.bool)
        for j, doc in enumerate(alt_docs):
            n = len(doc["input_ids"]); alt_ids[j, :n] = torch.tensor(doc["input_ids"]); alt_valid[j, :n] = True
        alt = [None] * layers; alt_handles = []
        for li, layer in enumerate(models["instruct"].model.layers):
            def self_hook(_module, inputs, index=li): alt[index] = inputs[0].detach()
            alt_handles.append(layer.mlp.down_proj.register_forward_pre_hook(self_hook))
        with torch.inference_mode():
            models["instruct"](input_ids=alt_ids.to(device), attention_mask=alt_valid.to(device, dtype=torch.long), use_cache=False)
        for h in alt_handles: h.remove()
        # Undo the row permutation before comparing token-aligned activations.
        inverse = torch.arange(len(chunk)-1, -1, -1, device=device)
        gate_sum = torch.zeros((layers, 5, width), dtype=torch.float32, device=device)
        gate_count = torch.zeros((layers,), dtype=torch.int64, device=device)
        for li in range(layers):
            x, y = captured["instruct"][li][:, 1:], alt[li][inverse, 1:1+captured["instruct"][li].shape[1]-1]
            keep = valid[:, 1:].to(device).reshape(-1)
            stats, n = sufficient(x.reshape(-1, width), y.reshape(-1, width), keep)
            gate_sum[li] = stats; gate_count[li] = n
        self_totals += gate_sum.cpu().to(torch.float64); self_counts += gate_count.cpu()
        pos += len(chunk)
        if pos % SAVE_EVERY < args.batch or pos == len(docs):
            atomic_save(save_path, {"meta": meta, "totals": totals, "counts": counts, "active": active,
                                    "null_totals": null_totals, "null_counts": null_counts,
                                    "self_totals": self_totals, "self_counts": self_counts,
                                    "next_doc": pos})
            print(f"saved {pos}/{len(docs)} documents ({pos / max(time.monotonic()-started, 1e-9):.2f} docs/s)", flush=True)
    print(f"completed {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--docs", required=True, help="manifest.json from prepare133.py")
    p.add_argument("--parent", required=True); p.add_argument("--sft", required=True)
    p.add_argument("--dpo", required=True); p.add_argument("--instruct", required=True)
    p.add_argument("--out", required=True); p.add_argument("--stage", choices=("pilot", "main"), required=True)
    p.add_argument("--device", default="xpu"); p.add_argument("--batch", type=int, default=16)
    p.add_argument("--limit", type=int)
    run(p.parse_args())


if __name__ == "__main__":
    main()
