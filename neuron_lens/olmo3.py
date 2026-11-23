"""OLMo 3's decoder-layer maps for the R-lens and the S-lens.

OLMo 3 keeps OLMo 2's post-norm residual layout and full-width ``q_norm``/``k_norm``:

    h1 = h  + post_attention_layernorm(attn(h))
    h2 = h1 + post_feedforward_layernorm(mlp(h1))

Two architecture additions over OLMo 2 are handled explicitly here:

1. **Hybrid attention (``layer_types`` + ``sliding_window``):** layers are marked
   ``"sliding_attention"`` or ``"full_attention"``. On ``"sliding_attention"`` layers,
   keys with ``query_pos - key_pos >= sliding_window`` are masked out alongside the
   upper causal triangle. Because masked entries have ``p == 0``, ``olmo2.jvp`` and
   ``olmo2.vjp`` require no change — ``dp = p * (ds - sum(p * ds))`` stays identically
   zero outside the window.
2. **Per-layer-type RoPE (YaRN on full-attention layers):** ``model.rotary_emb`` is
   called per ``layer_type`` and returns fp32 ``(cos, sin)`` with that layer type's
   ``attention_scaling`` already baked in. Because ``_rope(x, cos, sin)`` is linear
   in ``(cos, sin)``, its transpose ``_rope(x, cos, -sin)`` in ``vjp`` remains exact
   under YaRN ``attention_scaling != 1.0``.

Reuses ``neuron_lens.olmo2.{LayerMap, _norm_exact, jvp, vjp}`` directly so both OLMo
families share one tested linear-map implementation.
"""
import torch
import torch.nn.functional as F

from neuron_lens.olmo2 import LayerMap, _norm_exact, jvp, vjp
from neuron_lens.rules import HALF_RULE_BETA, check_layer
from neuron_lens.transport import SKIP_FIRST, _rope, valid_positions

_VALID_LAYER_TYPES = frozenset(("sliding_attention", "full_attention"))


def is_olmo3(model) -> bool:
    """An OLMo 3 decoder stack (``model_type == "olmo3"``)."""
    return getattr(getattr(model, "config", None), "model_type", "") == "olmo3"


def _check(model) -> None:
    """Refuse a decoder stack this module's hand-written forward does not cover."""
    if not is_olmo3(model):
        raise ValueError("not an OLMo 3 decoder stack (config.model_type != 'olmo3')")
    cfg = model.config
    layer_types = getattr(cfg, "layer_types", None)
    if not layer_types or len(layer_types) != len(model.layers):
        raise ValueError("OLMo 3 requires config.layer_types matching len(model.layers)")
    bad_types = set(layer_types) - _VALID_LAYER_TYPES
    if bad_types:
        raise ValueError(f"unsupported OLMo 3 layer_types: {sorted(bad_types)}")
    if "sliding_attention" in layer_types:
        sw = getattr(cfg, "sliding_window", None)
        if not isinstance(sw, int) or sw <= 0:
            raise ValueError(f"sliding_attention requires positive int sliding_window, got {sw!r}")
    for layer in model.layers:
        check_layer(layer)
        if hasattr(layer, "input_layernorm") or not hasattr(layer, "post_feedforward_layernorm"):
            raise ValueError("not an OLMo 3 post-norm decoder layer (input/pre-FF norm found)")
        at, mlp = layer.self_attn, layer.mlp
        projections = (at.q_proj, at.k_proj, at.v_proj, at.o_proj,
                       mlp.gate_proj, mlp.up_proj, mlp.down_proj)
        if any(p.bias is not None for p in projections):
            raise ValueError("projection biases are not supported")
        if at.scaling != at.head_dim ** -0.5:
            raise ValueError("only attention scaled by head_dim^-0.5 is supported")
        if (at.q_norm.weight.shape[0] != at.q_proj.weight.shape[0]
                or at.k_norm.weight.shape[0] != at.k_proj.weight.shape[0]):
            raise ValueError("q/k norms must span every head (OLMo 3's full-width norm)")


def _causal_masks(T: int, tp: int, sliding_window: int | None, device) -> dict[str, torch.Tensor]:
    """Additive float32 causal masks of shape ``[T, tp]`` for full and sliding attention."""
    full_mask = torch.full((T, tp), float("-inf"), device=device).triu(1)
    if sliding_window is None:
        sliding_mask = full_mask
    else:
        sliding_mask = full_mask.clone()
        # HF create_sliding_window_causal_mask attends to keys with 0 <= q - k < sliding_window.
        too_old = torch.tril(torch.ones((T, tp), dtype=torch.bool, device=device), diagonal=-sliding_window)
        sliding_mask = sliding_mask.masked_fill(too_old, float("-inf"))
    return {"full_attention": full_mask, "sliding_attention": sliding_mask}


@torch.no_grad()
def linearize(model, ids: torch.Tensor) -> list[LayerMap]:
    """Run the primal over a document and keep each layer's linear map."""
    _check(model)
    valid_positions(len(ids))  # refuses documents too short to have any
    dev, dt = model.embed_tokens.weight.device, model.embed_tokens.weight.dtype
    ids = ids[:-1].to(dev)
    T, S = len(ids), SKIP_FIRST
    tp = -(-T // 8) * 8
    h = model.embed_tokens(ids)
    pos_ids = torch.arange(T, device=dev)[None]
    layer_types = list(model.config.layer_types)
    pos_embs = {}
    for lt in set(layer_types):
        cos_lt, sin_lt = model.rotary_emb(h[None], pos_ids, lt)
        pos_embs[lt] = (cos_lt[0, :, None], sin_lt[0, :, None])  # [T, 1, hd], fp32
    masks = _causal_masks(T, tp, getattr(model.config, "sliding_window", None), dev)
    maps = []
    for i, layer in enumerate(model.layers):
        lt = layer_types[i]
        cos, sin = pos_embs[lt]
        mask = masks[lt]
        at, mlp = layer.self_attn, layer.mlp
        hd, g = at.head_dim, at.num_key_value_groups
        hk = at.k_proj.weight.shape[0] // hd
        q = F.linear(h, at.q_proj.weight)  # [T, Hq·hd]
        k = F.linear(h, at.k_proj.weight)
        v = F.pad(F.linear(h, at.v_proj.weight).view(T, hk, hd).transpose(0, 1), (0, 0, 0, tp - T))
        qn, rq = _norm_exact(at.q_norm, q)
        kn, rk = _norm_exact(at.k_norm, k)
        qr = _rope(qn.view(T, hk * g, hd), cos, sin).to(dt).transpose(0, 1).reshape(hk, g, T, hd)
        kr = F.pad(_rope(kn.view(T, hk, hd), cos, sin).to(dt).transpose(0, 1), (0, 0, 0, tp - T))
        p = torch.softmax(((qr @ kr[:, None].mT) * at.scaling).float() + mask, -1).to(dt)
        o = (p @ v[:, None]).reshape(hk * g, T, hd).transpose(0, 1).reshape(T, -1)
        attn = F.linear(o, at.o_proj.weight)
        pa = layer.post_attention_layernorm
        rpa = torch.rsqrt(attn.float().pow(2).mean(-1, keepdim=True) + pa.variance_epsilon)
        h1 = h + (pa.weight.float() * (attn.float() * rpa)).to(dt)
        gate = F.linear(h1, mlp.gate_proj.weight)
        up = F.linear(h1, mlp.up_proj.weight)
        mout = F.linear(F.silu(gate) * up, mlp.down_proj.weight)
        pf = layer.post_feedforward_layernorm
        rpf = torch.rsqrt(mout.float().pow(2).mean(-1, keepdim=True) + pf.variance_epsilon)
        h = h1 + (pf.weight.float() * (mout.float() * rpf)).to(dt)
        gate, up = gate[S:].float(), up[S:].float()
        maps.append(LayerMap(
            wq=at.q_proj.weight, wk=at.k_proj.weight, wv=at.v_proj.weight, wo=at.o_proj.weight,
            wg=mlp.gate_proj.weight, wu=mlp.up_proj.weight, wd=mlp.down_proj.weight,
            a_pa=pa.weight.float() * rpa[S:],
            a_pf=pf.weight.float() * rpf[S:],
            q=q[S:].float(), rq=rq[S:], wqn=at.q_norm.weight.float(),
            k=k[S:].float(), rk=rk[S:], wkn=at.k_norm.weight.float(),
            cos=cos[S:].float(), sin=sin[S:].float(),
            qr=qr[:, :, S:].reshape(hk, -1, hd), kr=kr, v=v,
            p=p[:, :, S:].reshape(hk, -1, tp),
            cg=(HALF_RULE_BETA * torch.sigmoid(gate) * up).to(dt),
            cu=(HALF_RULE_BETA * F.silu(gate)).to(dt)))
    return maps


__all__ = ["LayerMap", "is_olmo3", "linearize", "jvp", "vjp"]
