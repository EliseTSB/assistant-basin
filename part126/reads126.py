#!/usr/bin/env python3
"""Capture final-position attention and unit-attention value vectors for Part 126."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REGIONS = ("role", "template", "body", "question", "options", "other")
REGION = {name: i for i, name in enumerate(REGIONS)}
LAYERS_WANTED = (11, 15)
HEADS_EXPECTED = 16
HEAD_DIM_EXPECTED = 256
RANDOM_SEED = 12601
RECOMPUTE_TOLERANCE = 0.05


def _jsonl_rows(path: Path, repair_torn_tail: bool = False) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    nonblank = [(i, line) for i, line in enumerate(lines) if line.strip()]
    out = []
    for pos, (line_no, line) in enumerate(nonblank):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if pos == len(nonblank) - 1:
                print(f"LOG: skipping torn final JSONL line {line_no + 1} in {path}: {exc}", file=sys.stderr)
                if repair_torn_tail:
                    path.write_text("".join(lines[:line_no]), encoding="utf-8")
                continue
            raise ValueError(f"invalid JSONL at {path}:{line_no + 1}") from exc
    return out


def _flat_list(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list) and value[0] and isinstance(value[0][0], (list, tuple)):
        value = value[0]
    return value


def _overlapping_tokens(offsets, start: int, end: int) -> list[int]:
    return [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]


def _apply_char_region(regions, offsets, start: int, end: int, region_id: int) -> list[int]:
    hits = _overlapping_tokens(offsets, start, end)
    for i in hits:
        regions[i] = region_id
    return hits


def tokenize_regions(tok, rendered: str, body: str, rendering: str, question_text: str,
                     require_options: bool = True) -> dict:
    """Tokenize once and classify prompt tokens into the six registered regions."""
    role = "assistant" if rendering == "A" else "user"
    enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
    ids = _flat_list(enc["input_ids"])
    offsets = _flat_list(enc["offset_mapping"])
    ids = [int(x) for x in ids]
    offsets = [(int(a), int(b)) for a, b in offsets]
    if len(ids) != len(offsets) or not ids:
        raise ValueError("token ids and offsets must be non-empty and have the same length")

    start_marker = "<|im_start|>"
    char_starts = [m.start() for m in re.finditer(re.escape(start_marker), rendered)]
    if len(char_starts) < 2:
        raise ValueError("rendered prompt must contain the copied message and question message")
    first_char, second_char = char_starts[:2]
    role_start = rendered.find(role, first_char + len(start_marker), second_char)
    if role_start < 0:
        raise ValueError(f"copied message role {role!r} is missing")
    role_end = role_start + len(role)
    body_start = rendered.find(body, role_end, second_char)
    if body_start < 0:
        raise ValueError(f"copied body is absent after the {role} header")
    body_end = body_start + len(body)
    question_start = rendered.find(question_text, body_end)
    if question_start < 0:
        raise ValueError("question text is absent after the copied message")
    question_end = question_start + len(question_text)

    im_start_id = tok.convert_tokens_to_ids(start_marker)
    start_ids = [i for i, value in enumerate(ids) if value == int(im_start_id)]
    if len(start_ids) < 2:
        raise ValueError("tokenized prompt must contain at least two <|im_start|> tokens")
    first_token, second_token = start_ids[:2]
    regions = np.full(len(ids), REGION["other"], dtype=np.uint8)
    regions[first_token:second_token] = REGION["template"]
    role_hits = _apply_char_region(regions, offsets, role_start, role_end, REGION["role"])
    body_hits = _apply_char_region(regions, offsets, body_start, body_end, REGION["body"])
    question_hits = _apply_char_region(regions, offsets, question_start, question_end, REGION["question"])

    option_labels = ("You (the AI assistant)", "The user", "The system")
    option_hits = []
    for label in option_labels:
        local_start = question_text.find(label)
        if local_start < 0:
            if require_options:
                raise ValueError(f"option label {label!r} is absent from the question")
            continue
        option_hits.extend(
            _apply_char_region(regions, offsets, question_start + local_start,
                               question_start + local_start + len(label), REGION["options"])
        )
    if not role_hits or not body_hits or not question_hits or (require_options and not option_hits):
        raise ValueError("role, body, and question must map to tokens; main prompts also need option text")
    if require_options and np.any(np.bincount(regions, minlength=len(REGIONS)) == 0):
        raise ValueError("every registered token region must contain at least one token")
    body_positions = np.flatnonzero(regions == REGION["body"])
    if len(body_positions) == 0:
        raise ValueError("copied body region has no tokens")
    return {
        "input_ids": np.asarray(ids, dtype=np.int64),
        "offsets": offsets,
        "regions": regions,
        "body_positions": body_positions.astype(np.int64),
        "body_char_span": (body_start, body_end),
        "role_char_span": (role_start, role_end),
        "question_char_span": (question_start, question_end),
    }


def build_prompt_rows(tok, bodies, question, render, body_in_rendering, orders, renderings):
    rows = []
    for body_id, body in bodies:
        for rendering in renderings:
            for order_i, order in enumerate(orders, 1):
                q = question(order)
                rendered = render(tok, rendering, body, q)
                if not body_in_rendering(rendered, rendering, body):
                    raise ValueError(f"body {body_id} is missing from {rendering} rendering")
                tokenized = tokenize_regions(tok, rendered, body, rendering, q)
                rows.append({
                    "row_index": len(rows), "body_id": str(body_id), "rendering": rendering,
                    "order": order_i, "n_tokens": len(tokenized["input_ids"]),
                    "input_ids": tokenized["input_ids"], "regions": tokenized["regions"],
                    "body_positions": tokenized["body_positions"], "rendered": rendered,
                    "body_char_span": tokenized["body_char_span"],
                })
    return rows


class StubTokenizer:
    """Small deterministic tokenizer used only by end-to-end tests."""
    def __init__(self):
        self._token_to_id = {"<|im_start|>": 1, "<|im_end|>": 2, "<|im_sep|>": 3}
        self._id_to_token = {v: k for k, v in self._token_to_id.items()}

    def _id(self, token):
        if token not in self._token_to_id:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            value = 10 + int.from_bytes(digest[:4], "big") % 4000
            while value in self._id_to_token and self._id_to_token[value] != token:
                value += 1
            self._token_to_id[token] = value
            self._id_to_token[value] = token
        return self._token_to_id[token]

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False, return_tensors=None):
        pieces = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"<\|[^>]+\|>|[^\s]+|\s+", text)]
        ids = [self._id(token) for token, _, _ in pieces]
        offsets = [(start, end) for _, start, end in pieces]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        if return_tensors == "pt":
            import torch
            out["input_ids"] = torch.tensor([ids], dtype=torch.long)
        return out

    def convert_tokens_to_ids(self, token):
        return self._id(token)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        turns = []
        for message in messages:
            turns.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
        if add_generation_prompt:
            turns.append("<|im_start|>assistant\n")
        return "".join(turns)

    def decode(self, ids, **kwargs):
        return "".join(self._id_to_token.get(int(i), f"tok{int(i)}") for i in ids)


def _make_stub_model():
    import torch
    from torch import nn

    class StubAttention(nn.Module):
        def __init__(self, hidden=32, heads=16, kv_heads=4, head_dim=2):
            super().__init__()
            self.head_dim = head_dim
            self.num_key_value_groups = heads // kv_heads
            self.scaling = head_dim ** -0.5
            self.q_proj = nn.Linear(hidden, heads * head_dim * 2, bias=False)
            self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
            import torch.nn.functional as F
            batch, seq, _ = hidden_states.shape
            qkv = self.q_proj(hidden_states).view(batch, seq, -1, self.head_dim * 2)
            q0, gate = torch.chunk(qkv, 2, dim=-1)
            q = self.q_norm(q0).transpose(1, 2)
            key = self.k_norm(self.k_proj(hidden_states).view(batch, seq, -1, self.head_dim)).transpose(1, 2)
            value = self.v_proj(hidden_states).view(batch, seq, -1, self.head_dim).transpose(1, 2)
            from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv
            key = repeat_kv(key, self.num_key_value_groups)
            value = repeat_kv(value, self.num_key_value_groups)
            scores = q @ key.transpose(-2, -1) * self.scaling
            causal = torch.full((seq, seq), -float("inf"), device=hidden_states.device)
            causal = torch.triu(causal, diagonal=1).view(1, 1, seq, seq)
            scores = scores + causal
            if attention_mask is not None:
                scores = scores + attention_mask
            weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            out = (weights @ value).transpose(1, 2).contiguous()
            out = out * torch.sigmoid(gate)
            out = self.o_proj(out.view(batch, seq, -1))
            return out, weights

    class StubLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = StubAttention()

        def forward(self, x, position_embeddings, attention_mask=None):
            return x + self.self_attn(x, position_embeddings, attention_mask)[0]

    class Output:
        def __init__(self, logits):
            self.logits = logits

    class StubModel(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(126)
            self.config = type("Config", (), {"hidden_size": 32})()
            self.embed = nn.Embedding(4096, 32)
            self.layers = nn.ModuleList([StubLayer() for _ in range(16)])
            self.lm_head = nn.Linear(32, 64, bias=False)

        def forward(self, input_ids, use_cache=False, logits_to_keep=1, attention_mask=None, **kwargs):
            batch, seq = input_ids.shape
            x = self.embed(input_ids)
            cos = torch.ones((batch, seq, 2), device=x.device, dtype=x.dtype)
            sin = torch.zeros_like(cos)
            causal = torch.full((seq, seq), -float("inf"), device=x.device)
            causal = torch.triu(causal, diagonal=1).view(1, 1, seq, seq)
            if attention_mask is not None:
                causal = causal + attention_mask
            for layer in self.layers:
                x = layer(x, (cos, sin), causal)
            logits = self.lm_head(x[:, -int(logits_to_keep):, :])
            return Output(logits)

    return StubModel()


def _install_capture_hooks(model, layers, context):
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv

    handles = []
    for layer_i in LAYERS_WANTED:
        attn = layers[layer_i].self_attn
        head_dim = int(attn.head_dim)
        head_count = int(attn.o_proj.in_features // head_dim)
        if head_count != HEADS_EXPECTED or head_count * head_dim != attn.o_proj.in_features:
            raise ValueError(f"unexpected output projection split at layer {layer_i}: {head_count} heads x {head_dim}")

        def pre_hook(_module, args, *, li=layer_i):
            context["pre_o"][li] = args[0]

        def forward_hook(module, args, output, *, li=layer_i, hd=head_dim, nh=head_count):
            hidden = args[0]
            position_embeddings = args[1]
            attention_mask = args[2] if len(args) > 2 else None
            batch, seq, _ = hidden.shape
            qkv = module.q_proj(hidden).view(batch, seq, -1, hd * 2)
            q0, gate = torch.chunk(qkv, 2, dim=-1)
            gate = gate.reshape(batch, seq, -1).view(batch, seq, nh, hd)
            query = module.q_norm(q0.view(batch, seq, nh, hd)).transpose(1, 2)
            key = module.k_norm(module.k_proj(hidden).view(batch, seq, -1, hd)).transpose(1, 2)
            value = module.v_proj(hidden).view(batch, seq, -1, hd).transpose(1, 2)
            if position_embeddings is not None:
                query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
            key = repeat_kv(key, int(module.num_key_value_groups))
            value = repeat_kv(value, int(module.num_key_value_groups))
            scores = query[:, :, -1:, :] @ key.transpose(-2, -1)
            scores = scores * float(module.scaling)
            if attention_mask is None:
                causal = torch.full((seq, seq), -float("inf"), device=scores.device, dtype=scores.dtype)
                causal = torch.triu(causal, diagonal=1).view(1, 1, seq, seq)
                scores = scores + causal[:, :, -1:, :]
            else:
                mask = attention_mask
                if mask.ndim == 3:
                    mask = mask.unsqueeze(1)
                scores = scores + mask[..., -1:, :]
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype).squeeze(-2)
            gate_last = torch.sigmoid(gate[:, -1, :, :])
            reconstructed = (weights.unsqueeze(-2) @ value).squeeze(-2) * gate_last
            pre_o = context["pre_o"].pop(li)
            actual = pre_o[:, -1, :].view(batch, nh, hd)
            max_err = (reconstructed.float() - actual.float()).abs().amax().reshape(1)

            region_ids = context["region_ids"]
            region_masks = torch.stack([region_ids == ri for ri in range(len(REGIONS))], dim=1)
            region_mass = torch.stack([
                (weights.float() * region_masks[:, ri, :].unsqueeze(1).float()).sum(dim=-1)
                for ri in range(len(REGIONS))
            ], dim=-1)
            gated_values = value * gate_last.unsqueeze(2)
            region_vectors = []
            for ri in range(len(REGIONS)):
                mask = region_masks[:, ri, :].unsqueeze(1).unsqueeze(-1).to(gated_values.dtype)
                denom = mask.sum(dim=2).clamp_min(1)
                region_vectors.append((gated_values * mask).sum(dim=2) / denom)
            region_vectors = torch.stack(region_vectors, dim=2)
            row_index = torch.arange(batch, device=hidden.device)
            random_values = gated_values[row_index, :, context["random_positions"], :]

            pieces = [weights.float().reshape(-1), region_mass.float().reshape(-1),
                      region_vectors.float().reshape(-1), random_values.float().reshape(-1), max_err.float()]
            cpu = torch.cat(pieces).detach().cpu().numpy()
            cursor = 0
            n_attn = batch * nh * seq
            attn_cpu = cpu[cursor:cursor + n_attn].reshape(batch, nh, seq); cursor += n_attn
            n_mass = batch * nh * len(REGIONS)
            mass_cpu = cpu[cursor:cursor + n_mass].reshape(batch, nh, len(REGIONS)); cursor += n_mass
            n_region = batch * nh * len(REGIONS) * hd
            vectors_cpu = cpu[cursor:cursor + n_region].reshape(batch, nh, len(REGIONS), hd); cursor += n_region
            n_random = batch * nh * hd
            random_cpu = cpu[cursor:cursor + n_random].reshape(batch, nh, hd); cursor += n_random
            err_cpu = float(cpu[cursor])
            context["records"][li] = {
                "attention": attn_cpu.astype(np.float32),
                "region_mass": mass_cpu.astype(np.float32),
                "region_values": vectors_cpu.astype(np.float32),
                "random_body_values": random_cpu.astype(np.float32),
                "recompute_max_abs": err_cpu,
            }

        handles.append(attn.o_proj.register_forward_pre_hook(pre_hook))
        handles.append(attn.register_forward_hook(forward_hook))
    return handles


def _model_layers(model, nr):
    _, layers, _ = nr.find_text_layers(model)
    for layer_i in LAYERS_WANTED:
        if layer_i >= len(layers) or not hasattr(layers[layer_i], "self_attn"):
            raise ValueError(f"model has no self-attention module at layer {layer_i}")
    return layers


def _batch_capture(model, handles_context, batch_rows, device, extra_attention_mask=None):
    import torch
    ids = torch.as_tensor(np.stack([r["input_ids"] for r in batch_rows]), dtype=torch.long, device=device)
    regions = torch.as_tensor(np.stack([r["regions"] for r in batch_rows]), dtype=torch.long, device=device)
    random_positions = []
    for row in batch_rows:
        rng = np.random.default_rng(RANDOM_SEED + int(row["row_index"]))
        random_positions.append(int(rng.choice(row["body_positions"])))
    random_positions = torch.as_tensor(random_positions, dtype=torch.long, device=device)
    handles_context["pre_o"] = {}
    handles_context["records"] = {}
    handles_context["region_ids"] = regions
    handles_context["random_positions"] = random_positions
    with torch.inference_mode():
        kwargs = {"input_ids": ids, "use_cache": False, "logits_to_keep": 1}
        if extra_attention_mask is not None:
            kwargs["attention_mask"] = extra_attention_mask
        output = model(**kwargs)
    logits = output.logits
    if logits.shape[1] != 1:
        raise RuntimeError(f"logits_to_keep=1 returned {logits.shape[1]} positions")
    if set(handles_context["records"]) != set(LAYERS_WANTED):
        raise RuntimeError("capture hooks did not run at both target layers")
    return ids, logits, handles_context["records"]


def _capture_arrays(records, region_ids, input_ids):
    attention = np.stack([records[layer]["attention"] for layer in LAYERS_WANTED], axis=1)
    region_mass = np.stack([records[layer]["region_mass"] for layer in LAYERS_WANTED], axis=1)
    region_values = np.stack([records[layer]["region_values"] for layer in LAYERS_WANTED], axis=1)
    random_body_values = np.stack([records[layer]["random_body_values"] for layer in LAYERS_WANTED], axis=1)
    recompute_error = max(records[layer]["recompute_max_abs"] for layer in LAYERS_WANTED)
    return {
        "attention": attention.astype(np.float16),
        "region_mass": region_mass.astype(np.float16),
        "region_values": region_values.astype(np.float16),
        "random_body_values": random_body_values.astype(np.float16),
        "region_ids": region_ids.astype(np.uint8),
        "input_ids": input_ids.astype(np.int32),
        "recompute_error": recompute_error,
    }


def _atomic_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_bodies_with_shared_loader(path: Path, loader, out: Path):
    records = _jsonl_rows(path)
    nonblank = sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())
    if len(records) == nonblank:
        return loader(path)
    sanitized = out / ".bodies126-sanitized.jsonl"
    sanitized.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    try:
        return loader(sanitized)
    finally:
        sanitized.unlink(missing_ok=True)


def _write_shard(out: Path, first_row: int, arrays: dict):
    shard_name = f"batch_{first_row:06d}.npz"
    shard_path = out / shard_name
    tmp = out / (shard_name + ".tmp.npz")
    np.savez_compressed(tmp, attention=arrays["attention"], region_mass=arrays["region_mass"],
                        region_values=arrays["region_values"], random_body_values=arrays["random_body_values"],
                        region_ids=arrays["region_ids"], input_ids=arrays["input_ids"],
                        recompute_error=np.asarray(arrays["recompute_error"], dtype=np.float32))
    os.replace(tmp, shard_path)
    return shard_name


def _append_row_metadata(path: Path, rows, shard_name):
    with path.open("a", encoding="utf-8") as handle:
        for offset, row in enumerate(rows):
            record = {k: row[k] for k in ("row_index", "body_id", "rendering", "order", "n_tokens")}
            record.update({"shard": shard_name, "shard_offset": offset})
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _assemble_output(out: Path, rows, metadata_rows, head_dim: int):
    by_index = {int(row["row_index"]): row for row in metadata_rows}
    expected = {int(row["row_index"]) for row in rows}
    missing = sorted(expected - set(by_index))
    if missing:
        raise RuntimeError(f"capture is incomplete; missing row indexes {missing[:12]}")
    ordered_meta = [by_index[i] for i in sorted(expected)]
    loaded_shards = {}
    max_len = max(int(row["n_tokens"]) for row in ordered_meta)
    n = len(ordered_meta)
    attention = np.zeros((n, 2, HEADS_EXPECTED, max_len), dtype=np.float16)
    region_mass = np.zeros((n, 2, HEADS_EXPECTED, len(REGIONS)), dtype=np.float16)
    region_values = np.zeros((n, 2, HEADS_EXPECTED, len(REGIONS), head_dim), dtype=np.float16)
    random_body_values = np.zeros((n, 2, HEADS_EXPECTED, head_dim), dtype=np.float16)
    region_ids = np.full((n, max_len), 255, dtype=np.uint8)
    input_ids = np.zeros((n, max_len), dtype=np.int32)
    recompute = 0.0
    for out_i, meta in enumerate(ordered_meta):
        shard_name = meta["shard"]
        if shard_name not in loaded_shards:
            loaded_shards[shard_name] = np.load(out / shard_name, allow_pickle=False)
        shard = loaded_shards[shard_name]
        offset = int(meta["shard_offset"])
        length = int(meta["n_tokens"])
        attention[out_i, :, :, :length] = shard["attention"][offset]
        region_mass[out_i] = shard["region_mass"][offset]
        region_values[out_i] = shard["region_values"][offset]
        random_body_values[out_i] = shard["random_body_values"][offset]
        region_ids[out_i, :length] = shard["region_ids"][offset]
        input_ids[out_i, :length] = shard["input_ids"][offset]
        recompute = max(recompute, float(shard["recompute_error"]))
    region_sum_error = float(np.max(np.abs(region_mass.sum(axis=-1).astype(np.float32) - 1.0)))
    final_path = out / "reads126.npz"
    tmp = out / "reads126.tmp.npz"
    np.savez_compressed(tmp, attention=attention, region_mass=region_mass, region_values=region_values,
                        random_body_values=random_body_values, region_ids=region_ids, input_ids=input_ids,
                        lengths=np.asarray([int(row["n_tokens"]) for row in ordered_meta], dtype=np.int16),
                        row_indexes=np.asarray([int(row["row_index"]) for row in ordered_meta], dtype=np.int32),
                        layer_ids=np.asarray(LAYERS_WANTED, dtype=np.int16),
                        region_names=np.asarray(REGIONS, dtype="U16"))
    os.replace(tmp, final_path)
    rows_tmp = out / "rows.jsonl.tmp"
    with rows_tmp.open("w", encoding="utf-8") as handle:
        for row in ordered_meta:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(rows_tmp, out / "rows.jsonl")
    return final_path, recompute, region_sum_error, ordered_meta


def _positive_control(tok, model, handles_context, handles, question, render, body_in_rendering,
                      device, stub: bool):
    import torch
    body = "The cue word is marble. Repeat marble."
    q = "Which word appeared twice? Answer with the word only."
    rendered = render(tok, "U", body, q)
    if not body_in_rendering(rendered, "U", body):
        raise ValueError("copy-style positive-control body is missing from its rendering")
    tokenized = tokenize_regions(tok, rendered, body, "U", q, require_options=False)
    target_positions = []
    body_start, _ = tokenized["body_char_span"]
    search = 0
    while True:
        rel = body.find("marble", search)
        if rel < 0:
            break
        target_positions.extend(_overlapping_tokens(tokenized["offsets"], body_start + rel,
                                                    body_start + rel + len("marble")))
        search = rel + len("marble")
    target_positions = sorted(set(target_positions))
    if len(target_positions) < 2:
        raise ValueError("positive-control repeated word must map to at least two tokens")
    row = {"row_index": 999999, **tokenized}
    extra_mask = None
    if stub:
        # The fixture models a known copy head by adding a finite logit bias to the repeated tokens.
        extra_mask = torch.zeros((1, 1, len(tokenized["input_ids"]), len(tokenized["input_ids"])), dtype=torch.float32)
        extra_mask[0, 0, -1, target_positions] = 8.0
        extra_mask = extra_mask.to(device)
    _, _, records = _batch_capture(model, handles_context, [row], device, extra_attention_mask=extra_mask)
    max_rate = -float("inf")
    max_head = None
    uniform_share = len(target_positions) / len(tokenized["input_ids"])
    for layer in LAYERS_WANTED:
        mass = records[layer]["attention"][0, :, target_positions].sum(axis=-1)
        rates = mass / uniform_share
        for head, rate in enumerate(rates):
            if float(rate) > max_rate:
                max_rate = float(rate)
                max_head = f"L{layer}h{head}"
    return {
        "prompt": "The cue word is marble. Repeat marble. Which word appeared twice?",
        "target_token_positions": target_positions,
        "eligible_positions": len(tokenized["input_ids"]),
        "max_rate_over_uniform": max_rate,
        "head": max_head,
        "stub_fixture_bias": bool(stub),
        "passed": bool(max_rate >= 3.0),
    }


def _load_direct(device, MODEL_ID, REVISION):
    """Amendment 1: the weights go straight to the device, as Part 115b does. nr.load_model stages about 8 GB in host
    memory first, and the RAM guard stopped that at Windows' 2 GB floor. Same weights, dtype and device."""
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    import transformers.modeling_utils as mu
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    mu.caching_allocator_warmup = lambda *a, **k: None  # one block that size is over the A770's single-allocation limit
    kw = dict(revision=REVISION, dtype=torch.bfloat16, device_map={"": device})
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)
    except (ValueError, KeyError):
        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **kw)
    model.eval()
    model.requires_grad_(False)
    torch.set_grad_enabled(False)
    return tok, model


def run(args):
    import torch
    if args.stub:
        from part83.part83 import ORDERS, conditions
        from part82.part82 import BODY_PATH, RENDERINGS
        from part80.part80 import question, render
        from part80b.part80b import load_bodies, body_in_rendering
        tok = StubTokenizer()
        model = _make_stub_model()
        model.eval()
        model.requires_grad_(False)
        device = torch.device("cpu")
        loader = None
    else:
        import neuron_report as nr
        from part83.part83 import ORDERS, conditions
        from part82.part82 import MODEL, MODEL_ID, REVISION, BODY_PATH, RENDERINGS
        from part80.part80 import question, render
        from part80b.part80b import load_bodies, body_in_rendering
        tok, model = _load_direct(args.device, MODEL_ID, REVISION)
        model.requires_grad_(False)
        model.eval()
        device = args.device
        loader = nr
    if len(conditions()) != 33:
        raise RuntimeError("Part 83 condition table changed; expected h0 plus 32 heads")
    if args.stub:
        from part82.part82 import MODEL, MODEL_ID, REVISION
    bodies_path = Path(args.bodies or BODY_PATH)
    if not bodies_path.is_absolute():
        bodies_path = ROOT / bodies_path
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bodies = _load_bodies_with_shared_loader(bodies_path, load_bodies, out)
    if args.limit_bodies is not None:
        bodies = bodies[:args.limit_bodies]
    if not bodies:
        raise ValueError(f"no greedy bodies loaded from {bodies_path}")
    rows = build_prompt_rows(tok, bodies, question, render, body_in_rendering, ORDERS, RENDERINGS)
    prompt_fingerprints = [hashlib.sha256(row["rendered"].encode("utf-8")).hexdigest() for row in rows]
    prompt_fingerprint = hashlib.sha256("\n".join(prompt_fingerprints).encode("utf-8")).hexdigest()
    run_path = out / "run.json"
    rows_path = out / "rows.jsonl"
    meta = {
        "model": "qwen3.5-4b", "model_id": MODEL_ID, "revision": REVISION,
        "device": str(device), "dtype": str(next(model.parameters()).dtype),
        "body_count": len(bodies), "row_count_expected": len(rows), "renderings": list(RENDERINGS),
        "orders": ORDERS, "layers": list(LAYERS_WANTED), "heads": HEADS_EXPECTED,
        "head_dim": HEAD_DIM_EXPECTED, "regions": list(REGIONS), "batch_size": args.batch_size,
        "prompt_fingerprint": prompt_fingerprint, "prompt_fingerprints": prompt_fingerprints,
        "stub_fixture": bool(args.stub),
        "started_at_unix": time.time(), "gates": {},
    }
    if run_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        for key in ("model", "model_id", "revision", "layers"):
            if previous.get(key) != meta.get(key):
                raise ValueError(f"resume metadata mismatch for {key}: {previous.get(key)!r} != {meta.get(key)!r}")
        old_fingerprints = previous.get("prompt_fingerprints", [])
        if args.limit_bodies is not None and len(old_fingerprints) > len(prompt_fingerprints):
            raise ValueError("a --limit-bodies run cannot resume a larger run; use a new output folder")
        if len(old_fingerprints) > len(prompt_fingerprints) or prompt_fingerprints[:len(old_fingerprints)] != old_fingerprints:
            raise ValueError("resume prompts do not preserve the completed prompt prefix")
        meta["started_at_unix"] = previous.get("started_at_unix", meta["started_at_unix"])
    _atomic_json(run_path, meta)

    metadata_rows = _jsonl_rows(rows_path, repair_torn_tail=True)
    completed = {}
    for record in metadata_rows:
        row_i = int(record["row_index"])
        if row_i >= len(rows):
            raise ValueError(f"resume row index {row_i} is beyond the current prompt set")
        if int(record["n_tokens"]) != int(rows[row_i]["n_tokens"]):
            raise ValueError(f"resume token length changed for row {row_i}")
        if row_i in completed:
            raise ValueError(f"duplicate completed row index {row_i} in {rows_path}")
        if not (out / record["shard"]).is_file():
            raise FileNotFoundError(f"missing capture shard {record['shard']} for row {row_i}")
        completed[row_i] = record

    # Group equal-length prompts. No padding means position zero and the answer position match Part 83.
    groups = defaultdict(list)
    for row in rows:
        if int(row["row_index"]) not in completed:
            groups[int(row["n_tokens"])].append(row)
    for group in groups.values():
        group.sort(key=lambda row: row["row_index"])
    context = {"pre_o": {}, "records": {}, "region_ids": None, "random_positions": None}
    _, layers = (None, model.layers) if args.stub else (None, _model_layers(model, loader))
    head_dim = int(layers[LAYERS_WANTED[0]].self_attn.head_dim)
    if not args.stub and head_dim != HEAD_DIM_EXPECTED:
        raise ValueError(f"expected head_dim={HEAD_DIM_EXPECTED}, found {head_dim}")
    meta["head_dim"] = head_dim
    handles = []
    zero_effect = {"bitwise_equal": False, "max_abs_difference": None, "passed": False}
    recompute_max = 0.0
    try:
        first_group = next(iter(groups.values()), [])
        if not first_group:
            first_group = [rows[0]]
        probe_batch = first_group[:args.batch_size]
        probe_ids = torch.as_tensor(np.stack([row["input_ids"] for row in probe_batch]), dtype=torch.long, device=device)
        with torch.inference_mode():
            plain_logits = model(input_ids=probe_ids, use_cache=False, logits_to_keep=1).logits
        handles = _install_capture_hooks(model, layers, context)
        _, hooked_logits, probe_records = _batch_capture(model, context, probe_batch, device)
        equal = torch.equal(plain_logits, hooked_logits)
        maxdiff = float((plain_logits.float() - hooked_logits.float()).abs().amax().detach().cpu())
        zero_effect = {"bitwise_equal": bool(equal), "max_abs_difference": maxdiff, "passed": bool(equal)}
        if not equal:
            # The gate stays failed. Record whether two plain passes differ too, so a failure can be read.
            for handle in handles:
                handle.remove()
            handles = []
            with torch.inference_mode():
                repeat_logits = model(input_ids=probe_ids, use_cache=False, logits_to_keep=1).logits
            zero_effect["plain_repeat_bitwise_equal"] = bool(torch.equal(plain_logits, repeat_logits))
            handles = _install_capture_hooks(model, layers, context)
        for layer in LAYERS_WANTED:
            recompute_max = max(recompute_max, probe_records[layer]["recompute_max_abs"])

        batch_id = 0
        for length in sorted(groups):
            group = groups[length]
            for start in range(0, len(group), args.batch_size):
                batch_rows = group[start:start + args.batch_size]
                _, _, records = _batch_capture(model, context, batch_rows, device)
                arrays = _capture_arrays(records,
                    np.stack([row["regions"] for row in batch_rows]),
                    np.stack([row["input_ids"] for row in batch_rows]))
                recompute_max = max(recompute_max, arrays["recompute_error"])
                shard_name = _write_shard(out, min(row["row_index"] for row in batch_rows), arrays)
                _append_row_metadata(rows_path, batch_rows, shard_name)
                for offset, row in enumerate(batch_rows):
                    record = {k: row[k] for k in ("row_index", "body_id", "rendering", "order", "n_tokens")}
                    record.update({"shard": shard_name, "shard_offset": offset})
                    completed[int(row["row_index"])] = record
                batch_id += 1
                if batch_id % max(1, args.progress_every) == 0:
                    print(f"captured {len(completed)}/{len(rows)} rows", flush=True)

        positive = _positive_control(tok, model, context, handles, question, render, body_in_rendering, device, args.stub)
        for layer in LAYERS_WANTED:
            if layer in context["records"]:
                recompute_max = max(recompute_max, context["records"][layer]["recompute_max_abs"])
    finally:
        for handle in handles:
            handle.remove()

    metadata_rows = list(completed.values())
    metadata_rows.sort(key=lambda row: int(row["row_index"]))
    full_npz, assembled_recompute, region_sum_error, ordered_meta = _assemble_output(out, rows, metadata_rows, head_dim)
    recompute_max = max(recompute_max, assembled_recompute)
    meta["gates"] = {
        "recomputation": {
            "max_abs_difference": recompute_max, "tolerance": RECOMPUTE_TOLERANCE,
            "region_mass_sum_max_abs_difference": region_sum_error,
            "region_mass_tolerance": 0.01,
            "passed": bool(recompute_max <= RECOMPUTE_TOLERANCE and region_sum_error <= 0.01),
        },
        "zero_effect": zero_effect,
        "positive_control": positive,
        "lens_shape": None,
    }
    meta["row_count_complete"] = len(ordered_meta)
    meta["raw_tensors"] = full_npz.name
    meta["completed_at_unix"] = time.time()
    meta["elapsed_seconds"] = meta["completed_at_unix"] - meta["started_at_unix"]
    _atomic_json(run_path, meta)
    print(f"wrote {len(ordered_meta)} prompt rows, {LAYERS_WANTED[0]}/{LAYERS_WANTED[1]} x {HEADS_EXPECTED} heads, to {full_npz}")
    print(json.dumps(meta["gates"], ensure_ascii=False, sort_keys=True))


def main(argv=None):
    from part82.part82 import BODY_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bodies", default=BODY_PATH)
    parser.add_argument("--limit-bodies", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--stub", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.limit_bodies is not None and args.limit_bodies < 1:
        parser.error("--limit-bodies must be at least 1")
    run(args)


if __name__ == "__main__":
    main()
