#!/usr/bin/env python3
"""headswap182.py -- Part 182: swap single attention heads between OLMo 2 checkpoints (PART182-DESIGN.md).

Why not copy one head's rows of q_proj and k_proj: OLMo 2 applies q_norm and k_norm over all heads together. In the
installed transformers (5.22.1, models/olmo2/modeling_olmo2.py, class Olmo2Attention):

    self.q_norm = Olmo2RMSNorm(config.num_attention_heads * self.head_dim, config.rms_norm_eps)   # __init__
    query_states = self.q_norm(self.q_proj(hidden_states))                                         # forward
    key_states = self.k_norm(self.k_proj(hidden_states))
    ...
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

The norm's variance is a mean over every head's query (or key) coordinates, so one donor head's rows would change every
host head's scale. What is exact at head resolution is the last line: o_proj has no bias (attention_bias is false for
OLMo 2), so the attention output is the sum over heads h of o_proj[:, h*D:(h+1)*D] @ a_h, where a_h is head h's output
before o_proj. ``HeadSwapAttention`` runs the donor's whole self_attn on the host's own hidden states (its own q/k/v,
q_norm, k_norm, rotary and softmax, and its own key-value cache during generation) and replaces the listed heads'
terms: the host's o_proj columns times the host's a_h are dropped and the donor's o_proj columns times the donor's
a_h are added. The sum over the other heads is the host's. With every head of a layer listed, the output is the donor's
attention on the host's hidden states, which is what Part 99's whole-layer copy (``part99/swap99.py copy_components``)
computes; gates G1 and G3 check that on the real models, G2 checks that Instruct as its own donor changes nothing.

Mechanics: the per-head outputs are read by calling the attention module with its o_proj temporarily replaced by an
identity, so no part of the attention forward is re-implemented here and the installed source is what runs.
"""
from __future__ import annotations

import copy
import gc
import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _folder in (REPO / "part92", REPO / "part99", REPO):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import common92 as c  # noqa: E402

_IDENTITY = nn.Identity()


def head_count(attn) -> tuple[int, int]:
    """(number of attention heads, head_dim) of an Olmo2Attention module."""
    return int(attn.config.num_attention_heads), int(attn.head_dim)


def _heads_before_o_proj(attn, *args, **kwargs):
    """Call an attention module and return the concatenated per-head outputs, [batch, tokens, heads*head_dim]: the
    tensor its o_proj would have multiplied. Its forward runs unchanged with o_proj swapped for an identity."""
    o_proj = attn._modules["o_proj"]
    if not isinstance(o_proj, nn.Linear) or o_proj.bias is not None:
        raise TypeError("head swapping needs a bias-free nn.Linear o_proj")
    attn._modules["o_proj"] = _IDENTITY
    try:
        output, _weights = attn(*args, **kwargs)
    finally:
        attn._modules["o_proj"] = o_proj
    return output


class HeadSwapAttention(nn.Module):
    """A decoder layer's self_attn with the listed heads' contributions taken from the donor's attention, run on the
    same (host) hidden states. ``calls`` counts forward calls, so a runner can check the swap was exercised."""

    def __init__(self, host_attn, donor_attn, heads):
        super().__init__()
        n_heads, head_dim = head_count(host_attn)
        donor_heads, donor_dim = head_count(donor_attn)
        if (n_heads, head_dim) != (donor_heads, donor_dim):
            raise ValueError(f"host has {n_heads} heads of {head_dim}, donor {donor_heads} of {donor_dim}")
        if host_attn.layer_idx != donor_attn.layer_idx:
            raise ValueError(f"host layer {host_attn.layer_idx} given donor layer {donor_attn.layer_idx}")
        for name, module in (("host", host_attn), ("donor", donor_attn)):
            if module.o_proj.bias is not None:
                raise ValueError(f"{name} o_proj has a bias; the per-head decomposition is not exact")
        heads = sorted({int(h) for h in heads})
        if not heads or heads[0] < 0 or heads[-1] >= n_heads:
            raise ValueError(f"heads must be a non-empty subset of 0..{n_heads - 1}; got {heads}")
        self.host = host_attn
        self.donor = donor_attn
        self.heads = heads
        self.n_heads, self.head_dim = n_heads, head_dim
        mask = torch.zeros(n_heads * head_dim, dtype=torch.bool)
        for h in heads:
            mask[h * head_dim:(h + 1) * head_dim] = True
        self.register_buffer("swap_mask", mask, persistent=False)
        self.calls = 0
        self._donor_cache = None

    @property
    def layer_idx(self):
        return self.host.layer_idx

    @property
    def config(self):
        return self.host.config

    def _donor_cache_for(self, past_key_values):
        """The donor keeps its own key-value cache: a fresh one whenever the host's cache holds nothing for this layer
        (a prefill), the same one through the decode steps that follow."""
        if past_key_values is None:
            return None
        from transformers import DynamicCache
        host_len = past_key_values.get_seq_length(self.layer_idx)
        if host_len == 0 or self._donor_cache is None:
            self._donor_cache = DynamicCache()
        return self._donor_cache

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        self.calls += 1
        donor_cache = self._donor_cache_for(past_key_values)
        host_heads = _heads_before_o_proj(self.host, hidden_states, position_embeddings, attention_mask,
                                          past_key_values, **kwargs)
        donor_heads = _heads_before_o_proj(self.donor, hidden_states, position_embeddings, attention_mask,
                                           donor_cache, **kwargs)
        keep = (~self.swap_mask).to(dtype=host_heads.dtype)
        swap = self.swap_mask.to(dtype=donor_heads.dtype)
        # Exact zeros for the dropped heads, so with every head listed the result is the donor's o_proj output alone.
        output = self.host.o_proj(host_heads * keep) + self.donor.o_proj(donor_heads * swap)
        return output, None


def donor_attention(donor_model, layers, dtype=None) -> dict[int, nn.Module]:
    """Deep copies of the donor's self_attn modules for the listed layers, in ``dtype`` (default: as loaded), frozen
    and in eval mode. The donor model itself can then be freed."""
    donor_layers = c.get_layers(donor_model)
    out = {}
    for layer_no in sorted({int(n) for n in layers}):
        if not 0 <= layer_no < len(donor_layers):
            raise ValueError(f"layer {layer_no} outside the donor's {len(donor_layers)} layers")
        module = copy.deepcopy(donor_layers[layer_no].self_attn)
        if dtype is not None:
            module = module.to(dtype=dtype)
        module.requires_grad_(False)
        module.eval()
        out[layer_no] = module
    return out


def load_donor_attention(path, layers, dtype=torch.float32, attn_implementation="eager") -> dict[int, nn.Module]:
    """Load the donor snapshot in bf16 (as Parts 99 and 100 ran it), keep the listed layers' attention in ``dtype``,
    and free the rest."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(Path(path).expanduser()), dtype=torch.bfloat16,
                                                 local_files_only=True, attn_implementation=attn_implementation)
    model.requires_grad_(False)
    model.eval()
    out = donor_attention(model, layers, dtype=dtype)
    del model
    gc.collect()
    return out


def attach(model, donors: dict[int, nn.Module], heads: dict[int, object]) -> dict[int, HeadSwapAttention]:
    """Install a HeadSwapAttention on every layer in ``heads`` (layer -> iterable of head indices). Layers with an
    empty head set are left alone. Returns the wrappers by layer. Refuses a model that already carries one."""
    layers = c.get_layers(model)
    if attached_heads(model):
        raise RuntimeError("detach the previous head swap before attaching another")
    wrappers = {}
    for layer_no, head_set in sorted(heads.items()):
        head_set = sorted({int(h) for h in head_set})
        if not head_set:
            continue
        layer_no = int(layer_no)
        if layer_no not in donors:
            raise KeyError(f"no donor attention for layer {layer_no}")
        wrapper = HeadSwapAttention(layers[layer_no].self_attn, donors[layer_no], head_set)
        layers[layer_no].self_attn = wrapper
        wrappers[layer_no] = wrapper
    return wrappers


def detach(model) -> int:
    """Put every wrapped layer's own attention back. Returns how many layers were unwrapped."""
    count = 0
    for layer in c.get_layers(model):
        if isinstance(layer.self_attn, HeadSwapAttention):
            layer.self_attn = layer.self_attn.host
            count += 1
    return count


def attached_heads(model) -> dict[int, list[int]]:
    """{layer: heads} for every HeadSwapAttention currently installed."""
    out = {}
    for layer_no, layer in enumerate(c.get_layers(model)):
        if isinstance(layer.self_attn, HeadSwapAttention):
            out[layer_no] = list(layer.self_attn.heads)
    return out


def assert_attached(model, heads: dict[int, object], min_calls: int | None = None) -> None:
    """Raise unless exactly the requested non-empty head sets are installed (and, when ``min_calls`` is given, every
    wrapper has run at least that many times). A runner calls this before and after each swapped condition, so a
    condition whose swap was never installed or never exercised cannot be recorded."""
    expected = {int(k): sorted({int(h) for h in v}) for k, v in heads.items() if len(list(v))}
    actual = attached_heads(model)
    if actual != expected:
        raise RuntimeError(f"head swap not attached as requested: expected {expected}, found {actual}")
    if min_calls is not None:
        for layer_no, layer in enumerate(c.get_layers(model)):
            if layer_no in expected and layer.self_attn.calls < min_calls:
                raise RuntimeError(f"layer {layer_no}'s head swap ran {layer.self_attn.calls} times, under {min_calls}")


def reset_calls(model) -> None:
    for layer in c.get_layers(model):
        if isinstance(layer.self_attn, HeadSwapAttention):
            layer.self_attn.calls = 0


# ---------------------------------------------------------------------------------------------------------------------
# Conditions: which heads of which layers come from the donor.

def all_heads(n_heads: int) -> list[int]:
    return list(range(int(n_heads)))


def condition_heads(tag: str, layers: list[int], n_heads: int, ranking: list[tuple[int, int]] | None = None,
                    control_layers: list[int] | None = None) -> dict[int, list[int]]:
    """Decode a condition tag into {layer: heads}.

    none            nothing swapped
    hL.H            head H of layer L from the donor (L in ``layers`` only)
    layerL          every head of layer L
    block           every head of every listed layer (Part 100's --layers 0-3 attention swap)
    restoreL.H      the block with head H of layer L left as the host's (the other direction: one host head restored
                    into the donor block)
    topK            the first K heads of ``ranking`` (stage 1c's ladder; the ranking comes from score182.py)
    cL.H            Amendment 1 item 3: head H of a control layer L (L in ``control_layers``, default
                    score182.CONTROL_LAYERS) from the donor, nothing else swapped. A control-layer head is never in
                    ``layers``, so it can never be ranked, selected or laddered.
    """
    layers = [int(l) for l in layers]
    full = {l: all_heads(n_heads) for l in layers}
    if tag == "none":
        return {}
    if tag == "block":
        return full
    if tag.startswith("layer"):
        layer_no = int(tag[len("layer"):])
        if layer_no not in layers:
            raise ValueError(f"{tag}: layer {layer_no} is not in {layers}")
        return {layer_no: all_heads(n_heads)}
    if tag.startswith("c") and "." in tag:
        layer_no, head = (int(x) for x in tag[1:].split("."))
        if control_layers is None:
            import score182
            control_layers = score182.CONTROL_LAYERS
        control_layers = [int(l) for l in control_layers]
        if layer_no in layers:
            raise ValueError(f"{tag}: layer {layer_no} is a main layer; use h{layer_no}.{head}")
        if layer_no not in control_layers or not 0 <= head < n_heads:
            raise ValueError(f"{tag}: layer {layer_no} head {head} is outside control layers {control_layers} "
                             f"x 0..{n_heads - 1}")
        return {layer_no: [head]}
    if tag.startswith("h") and "." in tag:
        layer_no, head = (int(x) for x in tag[1:].split("."))
        if layer_no not in layers or not 0 <= head < n_heads:
            raise ValueError(f"{tag}: layer {layer_no} head {head} is outside layers {layers} x 0..{n_heads - 1}")
        return {layer_no: [head]}
    if tag.startswith("restore"):
        layer_no, head = (int(x) for x in tag[len("restore"):].split("."))
        if layer_no not in layers or not 0 <= head < n_heads:
            raise ValueError(f"{tag}: layer {layer_no} head {head} is outside layers {layers} x 0..{n_heads - 1}")
        out = {l: all_heads(n_heads) for l in layers}
        out[layer_no] = [h for h in all_heads(n_heads) if h != head]
        return {l: hs for l, hs in out.items() if hs}
    if tag.startswith("top"):
        k = int(tag[len("top"):])
        if ranking is None or k < 1 or k > len(ranking):
            raise ValueError(f"{tag}: needs a ranking of at least {k} heads")
        out: dict[int, list[int]] = {}
        for layer_no, head in ranking[:k]:
            out.setdefault(int(layer_no), []).append(int(head))
        return {l: sorted(hs) for l, hs in out.items()}
    raise ValueError(f"unknown condition tag {tag!r}")


def stage1_conditions(layers: list[int], n_heads: int, control_layers: list[int] | None = None) -> list[str]:
    """Stage 1a and 1b: none, every single head, every whole layer, the block, every single restore; then (Amendment
    1 item 3) the control-layer single heads cL.H, after the restore tags. ``none`` stays first and every earlier
    tag keeps its place."""
    if control_layers is None:
        import score182
        control_layers = score182.CONTROL_LAYERS
    tags = ["none"]
    tags += [f"h{l}.{h}" for l in layers for h in range(n_heads)]
    tags += [f"layer{l}" for l in layers]
    tags += ["block"]
    tags += [f"restore{l}.{h}" for l in layers for h in range(n_heads)]
    tags += [f"c{l}.{h}" for l in control_layers for h in range(n_heads)]
    return tags


# ---------------------------------------------------------------------------------------------------------------------
# Teacher-forced scoring of fixed answers under per-row pushes.

def answer_token_ids(tokenizer, answer: str, max_tokens: int) -> list[int]:
    """The first ``max_tokens`` ordinary token ids of a stored answer (no special tokens)."""
    ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    return [int(t) for t in ids[:int(max_tokens)]]


def build_batch(prompt_ids: list[int], answers: list[list[int]], pad_id: int, device: str):
    """Right-padded [batch, prompt+answer] ids, the attention mask, and (start, length) of each answer."""
    width = len(prompt_ids) + max(len(a) for a in answers)
    rows, mask = [], []
    for ids in answers:
        seq = list(prompt_ids) + list(ids)
        rows.append(seq + [pad_id] * (width - len(seq)))
        mask.append([1] * len(seq) + [0] * (width - len(seq)))
    input_ids = torch.tensor(rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(mask, dtype=torch.long, device=device)
    return input_ids, attention_mask


def row_push_hooks(layers, vectors_by_layer: dict[int, torch.Tensor]):
    """One add_row_hook per layer; rows not pushed at a layer carry an exact zero vector."""
    return [c.add_row_hook(layers[layer_no], vectors) for layer_no, vectors in sorted(vectors_by_layer.items())]


def teacher_forced_logprobs(model, input_ids, attention_mask, prompt_len: int, answers: list[list[int]]
                            ) -> list[dict]:
    """Per row: the log-probability of each answer token given the prompt and the preceding answer tokens, and
    whether the argmax at each position equals that token. Logits are read in float32. Any push hook must already be
    registered by the caller (score_rows does that); the gate runner reads logprobs_from_logits on logits it made with
    the same hooks, after its first version called this without them and read 2 of 5 tokens replayed."""
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return logprobs_from_logits(out.logits, prompt_len, answers)


def logprobs_from_logits(logits, prompt_len: int, answers: list[list[int]]) -> list[dict]:
    logits = logits.float()
    results = []
    for row, ids in enumerate(answers):
        n = len(ids)
        positions = logits[row, prompt_len - 1:prompt_len - 1 + n]
        logp = torch.log_softmax(positions, dim=-1)
        targets = torch.tensor(ids, dtype=torch.long, device=logp.device)
        token_logp = logp.gather(1, targets[:, None])[:, 0]
        argmax = positions.argmax(dim=-1)
        if not torch.isfinite(token_logp).all():
            raise FloatingPointError(f"non-finite log-probability in row {row}")
        results.append({
            "n_tokens": n,
            "logp_tokens": [round(float(v), 6) for v in token_logp.tolist()],
            "logp_sum": round(float(token_logp.sum()), 6),
            "argmax_match": int((argmax == targets).sum()),
        })
    return results


# ---------------------------------------------------------------------------------------------------------------------
# Amendment 1 item 5: residual readouts at intermediate decoder layers.

READOUT_LAYERS = (3, 7, 11, 15)   # indices into the decoder layer list; an index the model does not have is skipped


def answer_positions(prompt_len: int, answers: list[list[int]], attention_mask=None) -> list[list[int]]:
    """The positions whose logits ``logprobs_from_logits`` scores for each row: prompt_len-1 .. prompt_len-1+n-1,
    with any right-padded position dropped (the answer token at such a position is not scored)."""
    out = []
    for row, ids in enumerate(answers):
        positions = list(range(prompt_len - 1, prompt_len - 1 + len(ids)))
        if attention_mask is not None:
            valid = attention_mask[row].tolist()
            positions = [p for p in positions if p < len(valid) and valid[p]]
        out.append(positions)
    return out


def capture_layer_outputs(model, layer_indices):
    """Forward hooks that keep the float32 output of the listed decoder layers. Returns (stores, handles); each
    store is a list holding that layer's output tensor of the last forward."""
    layers = c.get_layers(model)
    stores = {int(i): [] for i in layer_indices}
    handles = []

    def make(layer_no):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            stores[layer_no].append(hidden.float())
        return hook

    for layer_no in stores:
        handles.append(layers[layer_no].register_forward_hook(make(layer_no)))
    return stores, handles


def relative_residual(reference, current, positions) -> dict[str, float]:
    """Per layer key: mean over the row's answer ``positions`` of ||h_cond - h_none|| / ||h_none||. A position whose
    reference norm is zero is skipped; a row with no usable position raises."""
    out = {}
    if sorted(reference) != sorted(current):
        raise KeyError(f"reference layers {sorted(reference)} against captured layers {sorted(current)}")
    for layer_no in sorted(reference):
        key = str(int(layer_no))
        ref, cur = reference[layer_no], current[layer_no]
        if ref.shape != cur.shape:
            raise ValueError(f"layer {key}: captured shapes differ, {tuple(ref.shape)} against {tuple(cur.shape)}")
        values = []
        for pos in positions:
            r, x = ref[0, pos], cur[0, pos]
            norm = float(torch.linalg.vector_norm(r))
            if norm == 0.0:
                continue
            values.append(float(torch.linalg.vector_norm(x - r)) / norm)
        if not values:
            raise ValueError(f"layer {key}: no answer position with a non-zero reference norm")
        out[key] = float(sum(values) / len(values))
    return out


def zero_readout(layer_indices) -> dict[str, float]:
    return {str(int(i)): 0.0 for i in layer_indices}


def _row_prompt(row, prompt_ids):
    """The prompt a row is scored after: its own ``prompt_ids`` when it carries one, else the batch prompt."""
    own = row.get("prompt_ids")
    return [int(t) for t in own] if own is not None else prompt_ids


def score_rows(model, tokenizer, layers, prompt_ids: list[int], rows: list[dict], device: str, batch: int,
               hidden: int, none_states: list[dict] | None = None, readout_layers=READOUT_LAYERS,
               capture: list[dict] | None = None) -> list[dict]:
    """Teacher-force ``rows`` in fixed batches. Each row: {"answer_ids": [...], "push_layer": int or None,
    "push": tensor [hidden] or None, "prompt_ids": [...] or None}. Returns one result dict per row, in order. The batch
    composition is a function of the row order and ``batch`` only, so every condition sees the same batches.

    Per-row prompts (Amendment 1b item 1): a row whose ``prompt_ids`` is a list is scored after that prompt, and a
    batch never mixes prompts, so every row's number is the one it gets alone. Rows without ``prompt_ids`` (or given
    the function's ``prompt_ids``) are scored exactly as before: they form consecutive runs, one forward per run.

    Residual readouts (Amendment 1 item 5): with ``none_states`` given (one dict per row, {layer: [1, tokens, hidden]}
    captured under ``none``), every result gains ``resid_rel`` = {layer: mean over the row's answer positions of
    ||h - h_none|| / ||h_none||}, and ``capture`` (a list) receives this condition's own captured states per row.
    The readout layers are ``readout_layers`` restricted to the model's decoder layers."""
    pad_id = int(tokenizer.pad_token_id)
    layer_count = len(c.get_layers(model))
    readout = [int(i) for i in readout_layers if 0 <= int(i) < layer_count]
    results = []
    begin = 0
    while begin < len(rows):
        row_prompt = _row_prompt(rows[begin], prompt_ids)
        end = begin + 1
        while end < len(rows) and end - begin < batch and _row_prompt(rows[end], prompt_ids) == row_prompt:
            end += 1
        chunk = rows[begin:end]
        chunk_begin = begin
        begin = end
        answers = [r["answer_ids"] for r in chunk]
        input_ids, attention_mask = build_batch(row_prompt, answers, pad_id, device)
        vectors: dict[int, torch.Tensor] = {}
        for i, r in enumerate(chunk):
            if r.get("push") is None:
                continue
            layer_no = int(r["push_layer"])
            if layer_no not in vectors:
                vectors[layer_no] = torch.zeros((len(chunk), hidden), dtype=r["push"].dtype, device=device)
            vectors[layer_no][i] = r["push"].to(device=device)
        handles = row_push_hooks(layers, vectors)
        stores, capture_handles = capture_layer_outputs(model, readout) if readout else ({}, [])
        try:
            out = teacher_forced_logprobs(model, input_ids, attention_mask, len(row_prompt), answers)
        finally:
            for handle in handles + capture_handles:
                handle.remove()
        if readout:
            positions = answer_positions(len(row_prompt), answers, attention_mask)
            for i, result in enumerate(out):
                # [1, tokens, hidden] for this row: the batch slice, so the two sides of the subtraction are the same
                # shape whatever the batch size.
                current = {layer_no: stores[layer_no][-1][i:i + 1] for layer_no in readout}
                if none_states is None:
                    result["resid_rel"] = zero_readout(readout)
                else:
                    result["resid_rel"] = relative_residual(none_states[chunk_begin + i], current, positions[i])
                if capture is not None:
                    capture.append(current)
        results.extend(out)
    return results


def logits_on_batch(model, input_ids, attention_mask, layers=None, vectors_by_layer=None) -> torch.Tensor:
    """Float32 logits of one batch, with optional per-row pushes (gates)."""
    handles = row_push_hooks(layers, vectors_by_layer) if vectors_by_layer else []
    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return out.logits.float().clone()


def compare_logits(a: torch.Tensor, b: torch.Tensor, attention_mask=None) -> dict:
    """Max absolute difference, max difference relative to the logit range, and argmax agreement over real
    positions."""
    if a.shape != b.shape:
        raise ValueError(f"logit shapes differ: {tuple(a.shape)} against {tuple(b.shape)}")
    if attention_mask is None:
        valid = torch.ones(a.shape[:2], dtype=torch.bool)
    else:
        valid = attention_mask.bool().cpu()
    diff = (a - b).abs()
    diff_valid = diff[valid]
    scale = float((a[valid].max() - a[valid].min()).item())
    agree = (a.argmax(-1) == b.argmax(-1))[valid]
    return {
        "max_abs_diff": float(diff_valid.max().item()),
        "mean_abs_diff": float(diff_valid.mean().item()),
        "logit_range": scale,
        "max_rel_diff": float(diff_valid.max().item()) / scale if scale > 0 else None,
        "positions": int(valid.sum().item()),
        "argmax_agree": int(agree.sum().item()),
        "bit_exact": bool(torch.equal(a, b)),
    }
