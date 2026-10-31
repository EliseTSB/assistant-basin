"""OLMo 2's decoder-layer maps for the R-lens and the S-lens.

OLMo 2 is **post-norm** where Qwen3 is pre-norm. One layer, on the residual ``h``:

    h1 = h  + post_attention_layernorm(attn(h))
    h2 = h1 + post_feedforward_layernorm(mlp(h1))

So the point the lens calls ``mid`` — the input of the MLP, the residual each
layer's MLP reads — is the *unnormalised* post-attention residual ``h1``. There
is no input norm and no pre-feedforward norm. The two residual norms sit on the
module outputs, and under RelP's LN rule their ``rsqrt`` is held constant, which
makes each one a per-position scale ``a = weight * rsqrt`` applied to the module
output before the residual add.

The attention's ``q_norm``/``k_norm`` are **single full-width RMSNorms** over
``heads * head_dim`` (2048 on OLMo 2 1B), applied to the projection before the
head view, unlike Qwen3's per-head norms. They keep their true derivative, as
Qwen3's do (Drowse's ``include_qk_norms: false``); only the width at which the
norm's ``rsqrt`` is taken changes.

Everything else is Qwen3's shape: SwiGLU MLP with SiLU (identity rule, half
rule, ``beta = 0.5``), RoPE on the full head dimension, causal attention,
multi-head attention with ``num_key_value_groups`` (OLMo 2 1B is 16/16, but the
map keeps the grouping so a GQA OLMo stack works), no projection biases.

``neuron_lens.rules.relp_rules`` already does the right thing for OLMo 2: its
``RESIDUAL_NORMS`` names both post-norms and its ``_mlp`` patch covers SwiGLU.
The tests use it as the reference for this module.
"""
from typing import NamedTuple

import torch
import torch.nn.functional as F

from neuron_lens.rules import HALF_RULE_BETA, check_layer
from neuron_lens.transport import (
    SKIP_FIRST,
    _norm_jvp,
    _norm_vjp,
    _rms_norm,
    _rope,
    valid_positions,
)


def is_olmo2(model) -> bool:
    """An OLMo 2 decoder stack (``model_type == "olmo2"``)."""
    return getattr(getattr(model, "config", None), "model_type", "") == "olmo2"


class LayerMap(NamedTuple):
    """One OLMo 2 decoder layer's RelP linear map on a document.

    Rows are the ``N`` valid positions; attention keys run over all ``T``
    positions the primal kept, padded to a multiple of 8 (``Tp``) as in
    ``transport._LayerMap``. ``a_pa``/``a_pf`` are the held-constant scales of
    the post-attention and post-feedforward norms; ``q``/``k`` are the q/k
    projections *before* their full-width norms, with the norms' ``rsqrt``
    (``rq``/``rk``) and weights (``wqn``/``wkn``) kept for their true derivative.
    """

    wq: torch.Tensor  # [out, in]
    wk: torch.Tensor
    wv: torch.Tensor
    wo: torch.Tensor
    wg: torch.Tensor
    wu: torch.Tensor
    wd: torch.Tensor
    a_pa: torch.Tensor  # [N, d]: post-attention norm weight × rsqrt (LN rule)
    a_pf: torch.Tensor  # [N, d]: post-feedforward norm weight × rsqrt
    q: torch.Tensor  # [N, Hq·hd]: q_proj output, before q_norm
    rq: torch.Tensor  # [N, 1]: rsqrt of q over the full width
    wqn: torch.Tensor  # [Hq·hd]
    k: torch.Tensor  # [N, Hk·hd]: k_proj output, before k_norm
    rk: torch.Tensor
    wkn: torch.Tensor
    cos: torch.Tensor  # [N, 1, hd]: RoPE
    sin: torch.Tensor
    qr: torch.Tensor  # [Hk, g·N, hd]: normed, rotated queries, grouped by KV head
    kr: torch.Tensor  # [Hk, Tp, hd]: normed, rotated keys
    v: torch.Tensor  # [Hk, Tp, hd]
    p: torch.Tensor  # [Hk, g·N, Tp]: attention probabilities
    cg: torch.Tensor  # [N, d_ff]: β·sigmoid(gate)·up (identity + half rules)
    cu: torch.Tensor  # [N, d_ff]: β·silu(gate)


def _check(model) -> None:
    """Refuse a decoder stack this module's hand-written forward does not cover."""
    for layer in model.layers:
        check_layer(layer)
        if hasattr(layer, "input_layernorm") or not hasattr(layer, "post_feedforward_layernorm"):
            raise ValueError("not an OLMo 2 post-norm decoder layer (input/pre-FF norm found)")
        at, mlp = layer.self_attn, layer.mlp
        projections = (at.q_proj, at.k_proj, at.v_proj, at.o_proj,
                       mlp.gate_proj, mlp.up_proj, mlp.down_proj)
        if any(p.bias is not None for p in projections):
            raise ValueError("projection biases are not supported")
        if getattr(at, "sliding_window", None) or at.scaling != at.head_dim ** -0.5:
            raise ValueError("only full attention scaled by head_dim^-0.5 is supported")
        if (at.q_norm.weight.shape[0] != at.q_proj.weight.shape[0]
                or at.k_norm.weight.shape[0] != at.k_proj.weight.shape[0]):
            raise ValueError("q/k norms must span every head (OLMo 2's full-width norm)")


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
    cos, sin = model.rotary_emb(h[None], torch.arange(T, device=dev)[None])
    cos, sin = cos[0, :, None], sin[0, :, None]  # [T, 1, hd]
    causal = torch.full((T, tp), float("-inf"), device=dev).triu(1)  # masks the padding too
    maps = []
    for layer in model.layers:
        at, mlp = layer.self_attn, layer.mlp
        hd, g = at.head_dim, at.num_key_value_groups
        hk = at.k_proj.weight.shape[0] // hd
        q = F.linear(h, at.q_proj.weight)  # [T, Hq·hd]
        k = F.linear(h, at.k_proj.weight)
        v = F.pad(F.linear(h, at.v_proj.weight).view(T, hk, hd).transpose(0, 1), (0, 0, 0, tp - T))
        qn, rq = _rms_norm(at.q_norm, q)
        kn, rk = _rms_norm(at.k_norm, k)
        qr = _rope(qn.view(T, hk * g, hd), cos, sin).transpose(0, 1).reshape(hk, g, T, hd)
        kr = F.pad(_rope(kn.view(T, hk, hd), cos, sin).transpose(0, 1), (0, 0, 0, tp - T))
        p = torch.softmax((qr @ kr[:, None].mT).float() * at.scaling + causal, -1).to(dt)
        o = (p @ v[:, None]).reshape(hk * g, T, hd).transpose(0, 1).reshape(T, -1)
        attn = F.linear(o, at.o_proj.weight)
        pa = layer.post_attention_layernorm
        rpa = torch.rsqrt(attn.float().pow(2).mean(-1, keepdim=True) + pa.variance_epsilon)
        h1 = h + (pa.weight.float() * attn.float() * rpa).to(dt)
        gate = F.linear(h1, mlp.gate_proj.weight)
        up = F.linear(h1, mlp.up_proj.weight)
        mout = F.linear(F.silu(gate) * up, mlp.down_proj.weight)
        pf = layer.post_feedforward_layernorm
        rpf = torch.rsqrt(mout.float().pow(2).mean(-1, keepdim=True) + pf.variance_epsilon)
        h = h1 + (pf.weight.float() * mout.float() * rpf).to(dt)
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


def jvp(m: LayerMap, X: torch.Tensor):
    """Push tangents ``X`` (``[b, N, d]``, fp32) through the layer; return them at
    its output and at the MLP's input (``mid``, the post-attention residual)."""
    dt = m.wq.dtype
    b, N, _ = X.shape
    hk, gn, hd = m.qr.shape
    g = gn // N
    xn = X.to(dt)
    dq = _norm_jvp(F.linear(xn, m.wq).float(), m.q, m.rq, m.wqn)
    dk = _norm_jvp(F.linear(xn, m.wk).float(), m.k, m.rk, m.wkn)
    dq = _rope(dq.view(b, N, hk * g, hd), m.cos, m.sin)
    dk = _rope(dk.view(b, N, hk, hd), m.cos, m.sin)
    dq = dq.to(dt).view(b, N, hk, g, hd).permute(0, 2, 3, 1, 4).reshape(b, hk, gn, hd)
    dk = dk.to(dt).transpose(1, 2)  # [b, Hk, N, hd]
    dv = F.linear(xn, m.wv).view(b, N, hk, hd).transpose(1, 2)
    keys = slice(SKIP_FIRST, SKIP_FIRST + N)
    ds = (dq @ m.kr.mT).float()  # [b, Hk, g·N, Tp]
    ds[..., keys] += (m.qr @ dk.mT).float()  # keys at valid positions move too
    ds *= hd ** -0.5
    dp = (m.p * (ds - (m.p * ds).sum(-1, keepdim=True))).to(dt)
    do = dp @ m.v + m.p[..., keys] @ dv
    do = do.view(b, hk, g, N, hd).permute(0, 3, 1, 2, 4).reshape(b, N, -1)
    d_attn = F.linear(do, m.wo).float()
    d_h1 = X + m.a_pa * d_attn  # h1 = h + a_pa · attn_out
    xn = d_h1.to(dt)
    d_mlp = F.linear(m.cg * F.linear(xn, m.wg) + m.cu * F.linear(xn, m.wu), m.wd).float()
    return d_h1 + m.a_pf * d_mlp, d_h1


def vjp(m: LayerMap, G: torch.Tensor, mid_seed: torch.Tensor | None = None,
        row_w: torch.Tensor | None = None):
    """Pull cotangents ``G`` (``[b, N, d]``, fp32) at the layer's output back
    through the map; return them at its input, and their means over positions at
    the MLP's input (``mid``) and at the layer's input (weighted by ``row_w``)."""
    dt = m.wq.dtype
    b, N, _ = G.shape
    hk, gn, hd = m.qr.shape
    g = gn // N
    gp = (m.a_pf * G).to(dt) @ m.wd  # [b, N, d_ff]
    G1 = G + ((m.cg * gp) @ m.wg + (m.cu * gp) @ m.wu).float()
    if mid_seed is not None:
        G1 = G1 + mid_seed
    mid = G1.mean(1) if row_w is None else (G1 * row_w[:, None]).sum(1)
    ga = (m.a_pa * G1).to(dt) @ m.wo  # [b, N, Hq·hd]
    go = ga.view(b, N, hk, g, hd).permute(0, 2, 3, 1, 4).reshape(b, hk, gn, hd)
    keys = slice(SKIP_FIRST, SKIP_FIRST + N)
    gp = (go @ m.v.mT).float()  # [b, Hk, g·N, Tp]
    gs = (m.p * (gp - (m.p * gp).sum(-1, keepdim=True)) * hd ** -0.5).to(dt)
    gq = (gs @ m.kr).view(b, hk, g, N, hd).permute(0, 3, 1, 2, 4).reshape(b, N, hk * g, hd)
    gk = (gs[..., keys].mT @ m.qr).transpose(1, 2)  # [b, N, Hk, hd]
    gv = (m.p[..., keys].mT @ go).transpose(1, 2)
    gq = _norm_vjp(_rope(gq.float(), m.cos, -m.sin).reshape(b, N, -1), m.q, m.rq, m.wqn).to(dt)
    gk = _norm_vjp(_rope(gk.float(), m.cos, -m.sin).reshape(b, N, -1), m.k, m.rk, m.wkn).to(dt)
    gx = gq.reshape(b, N, -1) @ m.wq + gk.reshape(b, N, -1) @ m.wk + gv.reshape(b, N, -1) @ m.wv
    G = G1 + gx.float()
    return G, mid, G.mean(1) if row_w is None else (G * row_w[:, None]).sum(1)
