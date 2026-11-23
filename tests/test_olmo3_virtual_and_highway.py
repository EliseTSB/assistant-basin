"""Unit & mathematical-identity tests for OLMo 3 7B Idea #2 (Virtual SwiGLU Neurons) &
Idea #4 (Exact 4-Term Signal Highway Decomposition).

Runs fast on CPU using a tiny 4-layer 3:1 sliding/full Olmo3ForCausalLM model.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
from transformers import Olmo3Config, Olmo3ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "olmo3_virtual_and_highway"))

from prompts import verify_disjoint_splits
from run_olmo3_idea2_and_idea4 import (
    VirtualSwiGLUSpec,
    capture_highway_trace,
    exact_highway_decompose,
    make_prenorm_wdown,
    natural_neuron_push_hook,
    virtual_neuron_hook,
)


@pytest.fixture(scope="module")
def tiny_olmo3():
    torch.manual_seed(42)
    cfg = Olmo3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        sliding_window=8,
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 500000.0},
            "full_attention": {"rope_type": "yarn", "rope_theta": 500000.0, "factor": 8.0, "original_max_position_embeddings": 64},
        },
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    model = Olmo3ForCausalLM(cfg).eval()
    return model


def test_disjoint_calibration_and_eval_splits():
    verify_disjoint_splits()


def test_prenorm_virtual_swiglu_exact_weight_append_equivalence(tiny_olmo3):
    """Verify our pre_norm virtual SwiGLU hook is bit/fp32-identical to literally
    appending `(w_gate, w_up, w_down)` as a real extra neuron in `mlp.{gate,up,down}_proj`!"""
    torch.manual_seed(7)
    d = tiny_olmo3.config.hidden_size
    L = 1
    w_gate = torch.randn(d)
    w_up = torch.randn(d)
    w_down = torch.randn(d)
    w_down = w_down / w_down.norm()
    dose = 2.35
    x = torch.randint(0, tiny_olmo3.config.vocab_size, (2, 10))

    spec = VirtualSwiGLUSpec(
        layer=L, site="pre_norm", w_write_unit=w_down, dose=dose,
        mode="swiglu", w_gate=w_gate, w_up=w_up, gate_bias=0.0, positions="all",
    )
    with virtual_neuron_hook(tiny_olmo3, spec), torch.inference_mode():
        logits_hook = tiny_olmo3(input_ids=x, use_cache=False).logits

    # Literally append the new neuron to a deep copy of the model's MLP weights!
    m2 = copy.deepcopy(tiny_olmo3)
    mlp = m2.model.layers[L].mlp
    with torch.no_grad():
        mlp.gate_proj.weight = torch.nn.Parameter(torch.cat([mlp.gate_proj.weight, w_gate[None, :]], dim=0))
        mlp.up_proj.weight = torch.nn.Parameter(torch.cat([mlp.up_proj.weight, w_up[None, :]], dim=0))
        mlp.down_proj.weight = torch.nn.Parameter(torch.cat([mlp.down_proj.weight, (dose * w_down)[:, None]], dim=1))
    with torch.inference_mode():
        logits_real = m2(input_ids=x, use_cache=False).logits

    max_diff = float((logits_hook - logits_real).abs().max().item())
    assert max_diff < 1e-5, f"pre_norm virtual SwiGLU hook diverged from literal appended neuron: {max_diff}"


def test_zero_dose_sham_exact_identity(tiny_olmo3):
    d = tiny_olmo3.config.hidden_size
    x = torch.randint(0, tiny_olmo3.config.vocab_size, (1, 9))
    with torch.inference_mode():
        base = tiny_olmo3(input_ids=x, use_cache=False).logits
    for site in ("pre_norm", "post_norm"):
        spec = VirtualSwiGLUSpec(layer=1, site=site, w_write_unit=torch.ones(d), dose=0.0, mode="unconditional")
        with virtual_neuron_hook(tiny_olmo3, spec), torch.inference_mode():
            got = tiny_olmo3(input_ids=x, use_cache=False).logits
        assert torch.equal(base, got), f"Non-zero sham drift on site={site}"


def test_prenorm_scale_saturation_vs_postnorm_linear_growth(tiny_olmo3):
    """BlackboxNLP 2026 guardrail check: inside post_feedforward_layernorm (pre_norm),
    huge doses saturate in RMSNorm, whereas post_norm grows linearly."""
    torch.manual_seed(11)
    d = tiny_olmo3.config.hidden_size
    L = 1
    w = torch.randn(d)
    w = w / w.norm()
    x = torch.randint(0, tiny_olmo3.config.vocab_size, (1, 8))
    tr_c = capture_highway_trace(tiny_olmo3, x)

    pre_norms, post_norms = [], []
    for dose in (10.0, 100.0):
        s_pre = VirtualSwiGLUSpec(layer=L, site="pre_norm", w_write_unit=w, dose=dose)
        with virtual_neuron_hook(tiny_olmo3, s_pre):
            tr_p = capture_highway_trace(tiny_olmo3, x)
        pre_norms.append(float((tr_p["mlp"][L][0] - tr_c["mlp"][L][0]).norm()))

        s_post = VirtualSwiGLUSpec(layer=L, site="post_norm", w_write_unit=w, dose=dose)
        with virtual_neuron_hook(tiny_olmo3, s_post):
            tr_q = capture_highway_trace(tiny_olmo3, x)
        post_norms.append(float((tr_q["mlp"][L][0] - tr_c["mlp"][L][0]).norm()))

    # Post-norm scales 10x linearly; pre-norm saturates far below 10x due to RMSNorm!
    assert abs((post_norms[1] / post_norms[0]) - 10.0) < 1e-4
    assert (pre_norms[1] / pre_norms[0]) < 2.0


def test_exact_4term_highway_telescoping_identity(tiny_olmo3):
    """Verify exact 100.000% telescoping identity across BOTH virtual and natural neuron pushes."""
    torch.manual_seed(19)
    d = tiny_olmo3.config.hidden_size
    x = torch.randint(0, tiny_olmo3.config.vocab_size, (1, 10))
    eps = float(tiny_olmo3.model.norm.variance_epsilon)
    layer_types = list(tiny_olmo3.config.layer_types)
    Wu = (tiny_olmo3.lm_head.weight.detach().float() * tiny_olmo3.model.norm.weight.detach().float()[None, :])
    tid = 7
    u_t = Wu[tid]

    tr_c = capture_highway_trace(tiny_olmo3, x)

    # 1. Virtual neuron (pre_norm and post_norm)
    w = torch.randn(d)
    w = w / w.norm()
    for site in ("pre_norm", "post_norm"):
        spec = VirtualSwiGLUSpec(layer=1, site=site, w_write_unit=w, dose=1.5, mode="unconditional")
        with virtual_neuron_hook(tiny_olmo3, spec):
            tr_i = capture_highway_trace(tiny_olmo3, x)
        dec = exact_highway_decompose(tr_c, tr_i, inject_layer=1, u_target=u_t, target_id=tid, layer_types=layer_types, eps=eps)
        assert dec["telescoping_err"] < 1e-4, f"Telescoping error too large ({site}): {dec['telescoping_err']}"
        assert dec["h_recon_l2_err"] < 1e-4, f"Residual recon error too large ({site}): {dec['h_recon_l2_err']}"
        # Ensure sliding vs full split accounts for all downstream attention
        assert abs((dec["sum_downstream_attn_sliding"] + dec["sum_downstream_attn_full"]) - dec["sum_downstream_attn"]) < 1e-4

    # 2. Natural neuron push
    with natural_neuron_push_hook(tiny_olmo3, layer=1, neuron_idx=5, delta_act=4.0):
        tr_n = capture_highway_trace(tiny_olmo3, x)
    dec_n = exact_highway_decompose(tr_c, tr_n, inject_layer=1, u_target=u_t, target_id=tid, layer_types=layer_types, eps=eps)
    assert dec_n["telescoping_err"] < 1e-4, f"Natural neuron telescoping error: {dec_n['telescoping_err']}"
    assert dec_n["h_recon_l2_err"] < 1e-4
