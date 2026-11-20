#!/usr/bin/env python3
"""Teacher-force the frozen Part 180 texts with neuron pushes and head ablations."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import sys
import time
import warnings
from pathlib import Path

import torch
import transformers

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PART92 = ROOT / "part92"
if str(PART92) not in sys.path:
    sys.path.insert(0, str(PART92))
import common92 as c  # noqa: E402

MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
MODEL_REVISION = "48d788eca847d4d7548f375ad03d3c9312f6139e"
K_CAP = 40
O1 = "The following report was published last year.\n\n"
O2 = "Chapter 1\n\n"


def normalise_path(value: str | Path) -> Path:
    raw = os.fspath(value)
    if os.name != "nt":
        raw = raw.replace("\\", "/")
    return Path(raw).expanduser()


def load_direct(model_id: str, revision: str, device: str):
    """Direct float32 load, using Part 178's device-map/no-warm-up pattern."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    import transformers.modeling_utils as modeling_utils
    if hasattr(modeling_utils, "caching_allocator_warmup"):
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    kwargs = {"revision": revision, "dtype": torch.float32, "device_map": {"": device},
              "attn_implementation": "eager"}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.eval()
    model.requires_grad_(False)
    torch.set_grad_enabled(False)
    return tokenizer, model


def verify_head_layout(model):
    """Check eager attention's head-major flattening and observe every o_proj input."""
    config = model.config
    hidden_size = int(config.hidden_size)
    configured_heads = int(config.num_attention_heads)
    layers = c.get_layers(model)
    if configured_heads < 1 or hidden_size % configured_heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    head_dim = hidden_size // configured_heads
    if getattr(config, "_attn_implementation", None) != "eager":
        raise ValueError("head-layout check requires the registered eager attention implementation")
    # The OLMo2 eager path transposes [B,H,S,D] to [B,S,H,D], then
    # flattens H and D immediately before o_proj. This fixes the slice order.
    from transformers.models.olmo2 import modeling_olmo2
    attention_source = inspect.getsource(type(layers[0].self_attn).forward)
    eager_source = inspect.getsource(modeling_olmo2.eager_attention_forward)
    if not re.search(r"attn_output\s*=\s*attn_output\.transpose\(1,\s*2\)\.contiguous\(\)", eager_source):
        raise ValueError("OLMo2 eager attention no longer returns [batch, sequence, head, head_dim] order")
    if not re.search(r"attn_output\s*=\s*attn_output\.reshape\(\*input_shape,\s*-1\)\.contiguous\(\)", attention_source):
        raise ValueError("OLMo2 attention no longer flattens ordered heads immediately before o_proj")
    if not re.search(r"attn_output\s*=\s*self\.o_proj\(attn_output\)", attention_source):
        raise ValueError("cannot confirm OLMo2 attention passes its flattened head output into o_proj")
    checks = []
    for layer_no, layer in enumerate(layers):
        attention = layer.self_attn
        projection = attention.o_proj
        module_heads = int(getattr(attention, "num_heads", configured_heads))
        module_head_dim = int(getattr(attention, "head_dim", head_dim))
        in_features = int(getattr(projection, "in_features", projection.weight.shape[1]))
        if module_heads != configured_heads or module_head_dim != head_dim or in_features != hidden_size:
            raise ValueError(
                f"layer {layer_no} o_proj layout mismatch: config={configured_heads}x{head_dim}, "
                f"module={module_heads}x{module_head_dim}, o_proj.in_features={in_features}"
            )
        checks.append({"layer": layer_no, "num_heads": module_heads, "head_dim": module_head_dim,
                       "o_proj_input_width": in_features, "head_major_contiguous_slices": True})
    observed = set()
    runtime_hooks = []
    for layer_no, layer in enumerate(layers):
        def check_input(_module, inputs, layer_no=layer_no):
            if (not inputs or inputs[0].ndim != 3 or
                    int(inputs[0].shape[-1]) != configured_heads * head_dim):
                raise ValueError(f"layer {layer_no} o_proj input is not [batch, sequence, heads*head_dim]")
            observed.add(layer_no)
            return None
        runtime_hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(check_input))
    device = next(model.parameters()).device
    try:
        with torch.inference_mode():
            model(input_ids=torch.zeros((1, 2), dtype=torch.long, device=device),
                  attention_mask=torch.ones((1, 2), dtype=torch.long, device=device),
                  use_cache=False)
    finally:
        for handle in runtime_hooks:
            handle.remove()
    if observed != set(range(len(layers))):
        raise ValueError(f"runtime layout probe saw o_proj inputs in layers {sorted(observed)}")
    return {"hidden_size": hidden_size, "num_heads": configured_heads, "head_dim": head_dim,
            "layers": checks, "runtime_probe_shape": [1, 2],
            "flattening_check": "eager [B,H,S,D] -> [B,S,H,D] contiguous -> [B,S,H*D]",
            "slice_rule": "o_proj input[..., h*head_dim:(h+1)*head_dim]"}


def _norm_summary(values: torch.Tensor):
    values = values.detach().float().cpu()
    if values.numel() == 0:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {"count": int(values.numel()), "min": float(values.min()),
            "median": float(values.median()), "max": float(values.max())}


def add_sparse_row_push(layer, vectors: torch.Tensor, *, norm_log=None, context=None):
    """Add one residual vector to each selected row at every sequence position."""
    if vectors.ndim != 2:
        raise ValueError("row vectors must have shape [batch, hidden]")

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if hidden.ndim != 3 or hidden.shape[0] != vectors.shape[0] or hidden.shape[-1] != vectors.shape[-1]:
            raise ValueError(f"unexpected layer output {tuple(hidden.shape)} for pushes {tuple(vectors.shape)}")
        base = hidden.float()
        row_vectors = vectors.to(device=hidden.device, dtype=torch.float32)
        pushed = base + row_vectors.unsqueeze(1)
        if norm_log is not None:
            active = torch.count_nonzero(row_vectors, dim=-1) > 0
            realised = torch.linalg.vector_norm(pushed - base, dim=-1)[active]
            requested = torch.linalg.vector_norm(row_vectors, dim=-1)[active]
            norm_log.append({**(context or {}), "realised_delta_norms": _norm_summary(realised),
                             "requested_vector_norms": _norm_summary(requested)})
        if isinstance(output, tuple):
            return (pushed,) + tuple(output[1:])
        if isinstance(output, list):
            return [pushed, *output[1:]]
        return pushed

    return layer.register_forward_hook(hook)


def add_head_set_hook(o_proj, heads, head_dim: int, expected_width: int | None = None, scale: float = 0.0):
    """Scale selected contiguous head slices before the output projection."""
    scale = float(scale)
    if not (math.isfinite(scale) and 0.0 <= scale <= 1.0):
        raise ValueError("head output scale must be finite and between 0 and 1")
    slices = [(int(head) * int(head_dim), (int(head) + 1) * int(head_dim)) for head in heads]
    if not slices:
        raise ValueError("at least one head is required")

    def hook(_module, inputs):
        width = int(inputs[0].shape[-1]) if inputs else 0
        if (not inputs or any(width < stop for _start, stop in slices) or
                (expected_width is not None and width != int(expected_width))):
            raise ValueError(f"o_proj input does not contain requested head slices {slices}")
        edited = inputs[0].clone()
        for start, stop in slices:
            edited[..., start:stop] *= scale
        return (edited, *inputs[1:])

    return o_proj.register_forward_pre_hook(hook)


def add_head_ablation_hook(o_proj, head: int, head_dim: int, expected_width: int | None = None):
    """Zero one contiguous head slice before the output projection."""
    return add_head_set_hook(o_proj, [head], head_dim, expected_width=expected_width, scale=0.0)


def add_norm_control_hook(layer, vectors: torch.Tensor):
    """Match each residual position's pushed norm without adding its push direction."""
    if vectors.ndim != 2:
        raise ValueError("norm-control vectors must have shape [batch, hidden]")

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if hidden.ndim != 3 or hidden.shape[0] != vectors.shape[0] or hidden.shape[-1] != vectors.shape[-1]:
            raise ValueError(f"unexpected layer output {tuple(hidden.shape)} for norm vectors {tuple(vectors.shape)}")
        residual = hidden.float()
        vector = vectors.to(device=hidden.device, dtype=torch.float32).unsqueeze(1)
        residual_norm = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
        pushed_norm = torch.linalg.vector_norm(residual + vector, dim=-1, keepdim=True)
        safe_denominator = torch.where(residual_norm > 0, residual_norm, torch.ones_like(residual_norm))
        ratio = torch.where(residual_norm > 0, pushed_norm / safe_denominator, torch.ones_like(residual_norm))
        controlled = residual * ratio
        if isinstance(output, tuple):
            return (controlled,) + tuple(output[1:])
        if isinstance(output, list):
            return [controlled, *output[1:]]
        return controlled

    return layer.register_forward_hook(hook)


def add_noop_hook(o_proj):
    """Install a real pre-hook that leaves every input object untouched."""
    return o_proj.register_forward_pre_hook(lambda _module, _inputs: None)


def teacher_forced_logps(model, input_ids, attention_mask, prompt_lengths, target_lengths):
    """Return one float32 log-prob vector per target continuation."""
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = output.logits.float()
    result = []
    for row, (prompt_len, target_len) in enumerate(zip(prompt_lengths, target_lengths)):
        if target_len == 0:
            result.append(torch.empty((0,), dtype=torch.float32, device=logits.device))
            continue
        # Logit at prompt_len-1 predicts target token 0; the final token is
        # predicted at prompt_len+target_len-2.
        first = int(prompt_len) - 1
        last = first + int(target_len)
        target_ids = input_ids[row, int(prompt_len):int(prompt_len) + int(target_len)].to(logits.device)
        selected_logits = logits[row, first:last]
        if selected_logits.shape[0] != int(target_len):
            raise ValueError("teacher-forced target/logit alignment failed")
        log_distribution = torch.log_softmax(selected_logits, dim=-1)
        values = log_distribution.gather(1, target_ids.reshape(-1, 1)).squeeze(1)
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError("non-finite teacher-forced log-probability")
        result.append(values.detach())
    return result


def _load_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
    return rows


def _write_json(path: Path, value):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _read_scores(path: Path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", errors="replace", newline="") as stream:
        lines = stream.readlines()
    nonblank = [index for index, line in enumerate(lines) if line.strip()]
    final_record = nonblank[-1] if nonblank else -1
    rows = []
    keep_lines = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
            keep_lines.append(line.rstrip("\r\n") + "\n")
        except json.JSONDecodeError as exc:
            if index != final_record:
                raise ValueError(f"invalid JSON at {path}:{index + 1}") from exc
            warnings.warn(f"dropping incomplete final score row from {path}", RuntimeWarning)
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.writelines(keep_lines)
            return rows
    if lines and (not lines[-1].endswith("\n") or len(keep_lines) != len(lines)):
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.writelines(keep_lines)
    return rows


def parse_heads(value: str | None, n_layers: int = 16, n_heads: int = 16):
    if value is None:
        return [(layer, head) for layer in range(n_layers) for head in range(n_heads)]
    result = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if ":" in item:
            layer_text, head_text = item.split(":", 1)
        elif "/" in item:
            layer_text, head_text = item.split("/", 1)
        else:
            raise ValueError("--heads entries must be layer:head, for example 0:0,3:7")
        layer_text = layer_text.strip().removeprefix("l").removeprefix("h")
        head_text = head_text.strip().removeprefix("l").removeprefix("h")
        layer_no, head_no = int(layer_text), int(head_text)
        if not (0 <= layer_no < n_layers and 0 <= head_no < n_heads):
            raise ValueError(f"head {(layer_no, head_no)} is outside {n_layers} layers by {n_heads} heads")
        if (layer_no, head_no) in result:
            raise ValueError(f"duplicate head {(layer_no, head_no)}")
        result.append((layer_no, head_no))
    if not result:
        raise ValueError("--heads must select at least one head")
    return result


def parse_head_sets(value: str | None, n_layers: int = 16, n_heads: int = 16):
    """Parse NAME=layer:head,layer:head;NAME2=... joint ablation sets."""
    if value is None or not value.strip():
        return []
    result = []
    names = set()
    for raw_set in value.split(";"):
        raw_set = raw_set.strip()
        if not raw_set or "=" not in raw_set:
            raise ValueError("--head-sets entries must be NAME=layer:head,layer:head")
        name, raw_heads = raw_set.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"invalid head-set name {name!r}")
        if name in names:
            raise ValueError(f"duplicate head-set name {name!r}")
        names.add(name)
        heads = parse_heads(raw_heads, n_layers, n_heads)
        result.append((name, heads))
    return result


def parse_unpushed_heads(value: str | None, n_layers: int = 16, n_heads: int = 16):
    """Parse the v4 L.H list used for matched, unpushed head-off contrasts."""
    if value is None or not value.strip():
        return []
    result = []
    for raw in value.split(","):
        item = raw.strip().lower().removeprefix("l")
        if "." not in item:
            raise ValueError("--unpushed-heads entries must be L.H, for example 3.7,12.4")
        layer_text, head_text = item.split(".", 1)
        layer_no, head_no = int(layer_text), int(head_text.removeprefix("h"))
        if not (0 <= layer_no < n_layers and 0 <= head_no < n_heads):
            raise ValueError(f"head {(layer_no, head_no)} is outside {n_layers} layers by {n_heads} heads")
        if (layer_no, head_no) in result:
            raise ValueError(f"duplicate unpushed head {(layer_no, head_no)}")
        result.append((layer_no, head_no))
    return result


def _scale_tag(scale: float) -> str:
    return format(float(scale), ".6g")


def _single_head_condition(head, scale):
    return f"head-L{int(head[0]):02d}-H{int(head[1]):02d}-scale-{_scale_tag(scale)}"


def _head_set_condition(name, scale):
    return f"headset-{name}-scale-{_scale_tag(scale)}"


def _unpushed_head_condition(head):
    return f"unpushed-head-L{int(head[0]):02d}-H{int(head[1]):02d}"


def _row_map(text_rows):
    rows = {}
    for row in text_rows:
        key = (str(row["neuron"]), str(row["key"]))
        if key in rows:
            raise ValueError(f"duplicate frozen text row {key}")
        rows[key] = row
    return rows


def _task_list(text_rows, end_token_ids=None):
    if any("score_push" not in row for row in text_rows):
        raise ValueError("texts.jsonl must declare score_push for every frozen row; regenerate texts180 output")
    rows = _row_map(text_rows)
    end_ids = {int(value) for value in (end_token_ids or ())}
    if end_ids:
        rows = {key: {**row, "token_ids": _before_first_end(row["token_ids"], end_ids)}
                for key, row in rows.items()}
    neuron_ids = sorted({neuron for neuron, _ in rows})
    cases = [
        ("self0_S0", "S0", "Kself0"), ("self0_A0", "A0", "Kself0"),
        ("cont0_C0", "C0", "Kcont0"), ("cont0_B0", "B0", "Kcont0"),
        ("self1_S1", "S1", "Kself1"), ("self1_A1", "A1", "Kself1"),
        ("cont1_C1", "C1", "Kcont1"), ("cont1_B1", "B1", "Kcont1"),
        ("doc1_D1", "D1", "Kdoc1"), ("doc1_B1d", "B1d", "Kdoc1"),
        ("doc2_D2", "D2", "Kdoc2"), ("doc2_B2d", "B2d", "Kdoc2"),
    ]
    tasks = []
    for neuron_id in neuron_ids:
        per_neuron = {key: rows[(neuron_id, key)] for key in
                      ("S0", "S1", "A0", "A1", "C0", "C1", "B0", "B1",
                       "D1", "D2", "B1d", "B2d", "G0")}
        for key, source in per_neuron.items():
            expected_push = key != "G0"
            if bool(source["score_push"]) != expected_push:
                raise ValueError(f"scoring push policy mismatch for {neuron_id}/{key}")
            if expected_push and source.get("push_layer") is None:
                raise ValueError(f"missing push layer for scored neuron row {neuron_id}/{key}")
        ks = {
            "Kself0": min(len(per_neuron["S0"]["token_ids"]), len(per_neuron["A0"]["token_ids"]), K_CAP),
            "Kcont0": min(len(per_neuron["C0"]["token_ids"]), len(per_neuron["B0"]["token_ids"]), K_CAP),
            "Kself1": min(len(per_neuron["S1"]["token_ids"]), len(per_neuron["A1"]["token_ids"]), K_CAP),
            "Kcont1": min(len(per_neuron["C1"]["token_ids"]), len(per_neuron["B1"]["token_ids"]), K_CAP),
            "Kdoc1": min(len(per_neuron["D1"]["token_ids"]), len(per_neuron["B1d"]["token_ids"]), K_CAP),
            "Kdoc2": min(len(per_neuron["D2"]["token_ids"]), len(per_neuron["B2d"]["token_ids"]), K_CAP),
            "Kcross": min(len(per_neuron["S0"]["token_ids"]), len(per_neuron["D1"]["token_ids"]), K_CAP),
        }
        for case, key, pair_name in cases:
            source = per_neuron[key]
            tasks.append({"condition": None, "neuron": neuron_id, "case": case, "key": key,
                          "frame": source["frame"],
                          "prompt": source["prompt"], "prompt_ids": list(map(int, source["prompt_ids"])),
                          "token_ids": list(map(int, source["token_ids"])), "push_layer": source.get("push_layer"),
                          "pushed": bool(source["score_push"]),
                          "k_tokens": ks[pair_name]})
        source = per_neuron["D1"]
        prompt_who = per_neuron["S0"]
        tasks.append({"condition": None, "neuron": neuron_id, "case": "cross_S0", "key": "S0",
                      "frame": "chat", "prompt": prompt_who["prompt"],
                      "prompt_ids": list(map(int, prompt_who["prompt_ids"])),
                      "token_ids": list(map(int, prompt_who["token_ids"])),
                      "push_layer": prompt_who.get("push_layer"),
                      "pushed": bool(prompt_who["score_push"]),
                      "k_tokens": ks["Kcross"]})
        tasks.append({"condition": None, "neuron": neuron_id, "case": "cross_D1", "key": "D1",
                      "frame": "chat", "prompt": prompt_who["prompt"],
                      "prompt_ids": list(map(int, prompt_who["prompt_ids"])),
                      "token_ids": list(map(int, source["token_ids"])),
                      "push_layer": source.get("push_layer"),
                      "pushed": bool(source["score_push"]),
                      "k_tokens": ks["Kcross"]})
    # G0 is one shared unpushed answer. Its row is duplicated in texts.jsonl as requested, but scored once.
    source = rows[(neuron_ids[0], "G0")]
    tasks.append({"condition": None, "neuron": "unpushed", "case": "g0", "key": "G0",
                  "frame": "chat", "prompt": source["prompt"],
                  "prompt_ids": list(map(int, source["prompt_ids"])),
                  "token_ids": list(map(int, source["token_ids"])), "push_layer": None,
                  "pushed": False, "k_tokens": min(len(source["token_ids"]), K_CAP)})
    return tasks


def _before_first_end(token_ids, end_token_ids):
    for index, token in enumerate(token_ids):
        if int(token) in end_token_ids:
            return list(token_ids[:index])
    return list(token_ids)


def _end_token_ids(tokenizer):
    values = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        values.update(int(value) for value in eos) if isinstance(eos, (tuple, list, set)) else values.add(int(eos))
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None
    for token in ("<|endoftext|>", "<|im_end|>", "<|end_of_turn|>", "<|endofturn|>"):
        if vocab is not None and token not in vocab:
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and (getattr(tokenizer, "unk_token_id", None) is None or
                                     int(token_id) != int(tokenizer.unk_token_id) or
                                     token == getattr(tokenizer, "unk_token", None)):
            values.add(int(token_id))
    return values


def _task_digest(values: torch.Tensor) -> str:
    raw = values.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _score_chunk(model, tasks, pad_id, push_vectors, *, condition, heads, head_dim,
                 scale=0.0, noop=False, norm_control=False, no_push=False, batch_index=0):
    device = next(model.parameters()).device
    lengths = [len(task["prompt_ids"]) + len(task["token_ids"]) for task in tasks]
    if any(not task["prompt_ids"] for task in tasks):
        raise ValueError("every scoring row must have at least one prompt token")
    width = max(lengths)
    input_ids = torch.full((len(tasks), width), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(tasks), width), dtype=torch.long, device=device)
    prompt_lengths, target_lengths = [], []
    hidden_size = int(model.config.hidden_size)
    layer_rows = {}
    norm_layer_rows = {}
    for row_index, task in enumerate(tasks):
        prompt_ids = task["prompt_ids"]
        target_ids = task["token_ids"]
        full = prompt_ids + target_ids
        input_ids[row_index, :len(full)] = torch.tensor(full, dtype=torch.long, device=device)
        attention_mask[row_index, :len(full)] = 1
        prompt_lengths.append(len(prompt_ids))
        target_lengths.append(len(target_ids))
        if task["pushed"] and not norm_control and not no_push:
            layer_rows.setdefault(int(task["push_layer"]), []).append(row_index)
        elif norm_control and task.get("push_layer") is not None:
            norm_layer_rows.setdefault(int(task["push_layer"]), []).append(row_index)
    push_hooks = []
    push_norm_log = []
    for layer_no, row_indices in layer_rows.items():
        vectors = torch.zeros((len(tasks), hidden_size), dtype=torch.float32, device=device)
        for row_index in row_indices:
            neuron = str(tasks[row_index]["neuron"])
            if neuron not in push_vectors:
                raise KeyError(f"no push vector loaded for {neuron}")
            vectors[row_index] = push_vectors[neuron]
        push_hooks.append(add_sparse_row_push(c.get_layers(model)[layer_no], vectors, norm_log=push_norm_log,
                                               context={"condition": condition, "batch": int(batch_index),
                                                        "layer": int(layer_no)}))
    norm_hooks = []
    for layer_no, row_indices in norm_layer_rows.items():
        vectors = torch.zeros((len(tasks), hidden_size), dtype=torch.float32, device=device)
        for row_index in row_indices:
            neuron = str(tasks[row_index]["neuron"])
            if neuron not in push_vectors:
                raise KeyError(f"no norm-control vector loaded for {neuron}")
            vectors[row_index] = push_vectors[neuron]
        norm_hooks.append(add_norm_control_hook(c.get_layers(model)[layer_no], vectors))
    ablation_hooks = []
    if noop:
        ablation_hooks = [add_noop_hook(layer.self_attn.o_proj) for layer in c.get_layers(model)]
    elif heads:
        by_layer = {}
        for layer_no, head_no in heads:
            by_layer.setdefault(int(layer_no), []).append(int(head_no))
        ablation_hooks = [add_head_set_hook(c.get_layers(model)[layer_no].self_attn.o_proj, head_ids,
                                            head_dim, expected_width=hidden_size, scale=scale)
                          for layer_no, head_ids in by_layer.items()]
    try:
        values = teacher_forced_logps(model, input_ids, attention_mask, prompt_lengths, target_lengths)
    finally:
        for handle in ablation_hooks + push_hooks + norm_hooks:
            handle.remove()
    output = []
    for task, logps in zip(tasks, values):
        count = int(logps.numel())
        if count and not bool(torch.isfinite(logps).all()):
            raise FloatingPointError(f"non-finite kept token log-probability: {task['case']}/{task['neuron']}")
        k = min(int(task["k_tokens"]), count)
        mean_k = float(logps[:k].mean().item()) if k else None
        first = float(logps[0].item()) if count else None
        full = float(logps.mean().item()) if count else None
        digest = _task_digest(logps)
        row = {"condition": condition, "neuron": task["neuron"], "key": task["key"],
               "case": task["case"], "frame": task["frame"], "prompt": task["prompt"],
               "token_count": count, "k_tokens": k, "lp_k": mean_k,
               "lp_first": first, "lp_full": full, "token_logp_sha256": digest}
        if heads:
            row["heads"] = [{"layer": int(layer), "head": int(head)} for layer, head in heads]
            if len(heads) == 1:
                row["head"] = row["heads"][0]
            row["scale"] = float(scale)
        output.append(row)
    return output, push_norm_log


def _condition_complete(rows, condition, expected):
    present = [row for row in rows if row.get("condition") == condition]
    actual = {(str(row.get("neuron")), str(row.get("case"))) for row in present}
    wanted = {(str(task["neuron"]), str(task["case"])) for task in expected}
    return actual == wanted and len(present) == len(wanted)


def _drop_partial_conditions(path: Path, rows, expected_by_condition):
    by_condition = {}
    for row in rows:
        by_condition.setdefault(str(row.get("condition")), []).append(row)
    complete, incomplete = set(), set()
    for condition, records in by_condition.items():
        expected = expected_by_condition.get(condition)
        if expected is None:
            continue
        keys = {(str(record.get("neuron")), str(record.get("case"))) for record in records}
        if keys == expected and len(records) == len(expected):
            complete.add(condition)
        else:
            incomplete.add(condition)
    if incomplete:
        kept = [row for row in rows if str(row.get("condition")) not in incomplete]
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in kept:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows[:] = kept
    return complete


def _compare_gate1(none_rows, noop_rows, expected):
    key = lambda r: (str(r["neuron"]), str(r["case"]))
    none = {key(row): row for row in none_rows}
    noop = {key(row): row for row in noop_rows}
    matches = []
    if set(none) != set(noop) or set(none) != {(str(x["neuron"]), str(x["case"])) for x in expected}:
        return {"passed": False, "reason": "missing or extra scoring rows", "rows_compared": 0}
    columns = ("token_count", "k_tokens", "lp_k", "lp_first", "lp_full", "token_logp_sha256")
    for record_key in none:
        same = all(none[record_key].get(column) == noop[record_key].get(column) for column in columns)
        matches.append(same)
    return {"passed": all(matches), "rows_compared": len(matches),
            "matching_rows": sum(matches), "comparison": "exact summary floats and SHA-256 of every float32 token LP"}


def run(args):
    out_dir = normalise_path(args.out)
    texts_path = normalise_path(args.texts)
    pushes_path = normalise_path(args.pushes)
    out_dir.mkdir(parents=True, exist_ok=True)
    text_rows = _load_jsonl(texts_path)
    if not text_rows:
        raise ValueError(f"no frozen rows in {texts_path}")
    tokenizer, model = load_direct(args.model or MODEL_ID, args.revision, args.device)
    tasks = _task_list(text_rows, end_token_ids=_end_token_ids(tokenizer))
    layout = verify_head_layout(model)
    heads = parse_heads(args.heads, len(c.get_layers(model)), layout["num_heads"])
    head_sets = parse_head_sets(args.head_sets, len(c.get_layers(model)), layout["num_heads"])
    unpushed_heads = parse_unpushed_heads(args.unpushed_heads, len(c.get_layers(model)), layout["num_heads"])
    unpushed_cases = {"self0_S0", "self0_A0", "cont0_C0", "cont0_B0",
                      "self1_S1", "self1_A1", "cont1_C1", "cont1_B1"}
    unpushed_tasks = [task for task in tasks if task["case"] in unpushed_cases]
    if not (math.isfinite(float(args.scale)) and 0.0 <= float(args.scale) <= 1.0):
        raise ValueError("--scale must be finite and between 0 and 1")
    scale = float(args.scale)
    stored = torch.load(pushes_path, map_location="cpu", weights_only=True)
    neuron_ids = sorted({str(row["neuron"]) for row in text_rows if row.get("score_push")})
    push_vectors = {}
    for neuron_id in neuron_ids:
        key = f"{neuron_id}|neuron"
        if key not in stored:
            raise KeyError(f"stored push is missing {key}")
        vector = stored[key].detach().to(device=next(model.parameters()).device,
                                         dtype=next(model.parameters()).dtype)
        if tuple(vector.shape) != (int(model.config.hidden_size),) or not bool(torch.isfinite(vector).all()):
            raise ValueError(f"invalid push vector {key}: shape={tuple(vector.shape)}")
        push_vectors[neuron_id] = vector

    scores_path = out_dir / "scores.jsonl"
    existing = _read_scores(scores_path)
    expected_by_condition = {
        condition: {(str(task["neuron"]), str(task["case"])) for task in task_rows}
        for condition, task_rows in (
            [("none", tasks), ("hook-noop", tasks), ("norm", tasks), ("M0", tasks)] +
            [(_single_head_condition(head, scale), tasks) for head in heads] +
            [(_head_set_condition(name, scale), tasks) for name, _ in head_sets] +
            [(_unpushed_head_condition(head), unpushed_tasks) for head in unpushed_heads])
    }
    complete = _drop_partial_conditions(scores_path, existing, expected_by_condition)
    existing_conditions = set(complete)
    if scores_path.is_file():
        existing = _read_scores(scores_path)
    started = time.time()
    meta_path = out_dir / "score.meta.json"
    meta = _load_json(meta_path) if meta_path.is_file() else {}
    meta.update({"model": args.model or MODEL_ID, "revision": args.revision,
                 "device": str(next(model.parameters()).device), "model_dtype": str(next(model.parameters()).dtype),
                 "torch_version": torch.__version__, "transformers_version": transformers.__version__,
                 "head_layout": layout, "head_slice_validation": "runtime o_proj width equals ordered head_count * head_dim",
                 "batch_size": int(args.batch), "k_cap": K_CAP,
                 "conditions_requested": ["none", "hook-noop", "norm", "M0"] +
                 [_single_head_condition(head, scale) for head in heads] +
                 [_head_set_condition(name, scale) for name, _set_heads in head_sets] +
                 [_unpushed_head_condition(head) for head in unpushed_heads],
                 "condition_parameters": {
                     "single_heads": [{"condition": _single_head_condition(head, scale),
                                       "heads": [list(head)], "scale": scale} for head in heads],
                     "head_sets": [{"condition": _head_set_condition(name, scale), "name": name,
                                    "heads": [list(head) for head in set_heads], "scale": scale}
                                   for name, set_heads in head_sets],
                    "unpushed_heads": [{"condition": _unpushed_head_condition(head), "head": list(head),
                                        "scale": 0.0, "push": False} for head in unpushed_heads],
                    "M0": "no push, no head off; all selection, held-out, document, crossed, and generic-damage rows",
                    "norm": "no push vector added; per-position residual norm matches pushed residual",
                 }})

    def score_condition(condition, selected_heads=None, noop=False, norm_control=False,
                        no_push=False, condition_tasks=None, condition_scale=None):
        if condition in existing_conditions:
            return _read_scores(scores_path)
        condition_tasks = tasks if condition_tasks is None else condition_tasks
        local_scale = scale if condition_scale is None else float(condition_scale)
        rows = []
        condition_batch_shapes = []
        condition_push_norms = []
        for start in range(0, len(condition_tasks), args.batch):
            chunk = condition_tasks[start:start + args.batch]
            chunk_rows, chunk_push_norms = _score_chunk(
                model, chunk, tokenizer.pad_token_id, push_vectors, condition=condition,
                heads=selected_heads, head_dim=layout["head_dim"], scale=local_scale, noop=noop,
                norm_control=norm_control, no_push=no_push, batch_index=start // args.batch)
            rows.extend(chunk_rows)
            condition_push_norms.extend(chunk_push_norms)
            condition_batch_shapes.append({
                "input_shape": [len(chunk), max(len(task["prompt_ids"]) + len(task["token_ids"])
                                                for task in chunk)],
                "input_dtype": "torch.int64", "mask_dtype": "torch.int64",
                "model_dtype": str(next(model.parameters()).dtype),
                "device": str(next(model.parameters()).device),
            })
        with scores_path.open("a", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        existing_conditions.add(condition)
        meta["last_condition"] = condition
        meta.setdefault("batch_shapes_by_condition", {})[condition] = condition_batch_shapes
        meta.setdefault("realised_push_norms_by_condition", {})[condition] = condition_push_norms
        meta["completed_conditions"] = sorted(existing_conditions)
        meta["seconds_elapsed"] = time.time() - started
        _write_json(meta_path, meta)
        return _read_scores(scores_path)

    score_condition("none")
    score_condition("hook-noop", noop=True)
    score_condition("norm", norm_control=True)
    score_condition("M0", no_push=True)
    current = _read_scores(scores_path)
    gate = _compare_gate1([row for row in current if row["condition"] == "none"],
                          [row for row in current if row["condition"] == "hook-noop"], tasks)
    meta["gate1"] = gate
    _write_json(meta_path, meta)
    if not gate["passed"]:
        raise SystemExit("Gate 1 failed: hook-noop does not match none bit for bit")

    for layer_no, head_no in heads:
        head = (layer_no, head_no)
        score_condition(_single_head_condition(head, scale), selected_heads=[head])
    for name, set_heads in head_sets:
        score_condition(_head_set_condition(name, scale), selected_heads=set_heads)
    for head in unpushed_heads:
        score_condition(_unpushed_head_condition(head), selected_heads=[head], no_push=True,
                        condition_tasks=unpushed_tasks, condition_scale=0.0)
    meta["completed_conditions"] = sorted(existing_conditions)
    meta["head_subset"] = [list(item) for item in heads]
    meta["seconds_elapsed"] = time.time() - started
    _write_json(meta_path, meta)
    print(json.dumps({"conditions": len(existing_conditions), "scored_rows": len(_read_scores(scores_path)),
                      "gate1": gate, "scores": str(scores_path)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", default=str(ROOT / "results" / "part180" / "texts.jsonl"))
    parser.add_argument("--pushes", default="results/part99-pushes.pt")
    parser.add_argument("--out", default=str(ROOT / "results" / "part180"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--device", choices=("cpu", "xpu"), default="cpu")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--heads", default=None,
                        help="optional comma-separated layer:head subset, e.g. 0:0,2:7")
    parser.add_argument("--head-sets", default=None,
                        help="joint sets NAME=layer:head,layer:head;NAME2=..., each scored as one condition")
    parser.add_argument("--unpushed-heads", default=None,
                        help="comma-separated L.H heads for unpushed selection/held-out contrasts, e.g. 3.7,12.4")
    parser.add_argument("--scale", type=float, default=0.0,
                        help="multiply selected o_proj head slices by this factor (0 zeros, 0.5 halves)")
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    run(args)


if __name__ == "__main__":
    main()
