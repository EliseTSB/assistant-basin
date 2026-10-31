"""OLMo 2's hand-written layer maps (``neuron_lens/olmo2.py``) against brute-force
Jacobians of the model under ``relp_rules``, on tiny random OLMo 2 stacks.

Modelled on ``tests/test_transport.py``: the same ``run_with`` / ``exact``
harness, with the reference points moved to OLMo 2's post-norm layer — ``mid``
is the MLP's input (the post-attention residual), so it is captured with a
pre-hook on ``block.mlp`` and the mid-side reference re-adds the
post-feedforward norm, exactly as the layer does.
"""
import math

import pytest
import torch
from transformers import Olmo2Config, Olmo2Model

from neuron_lens.rules import relp_rules
from neuron_lens.transport import (
    SKIP_FIRST,
    Accumulator,
    r_lens_doc,
    s_lens_doc,
    valid_positions,
)

D, N_LAYERS = 16, 3


def tiny_model(**overrides) -> Olmo2Model:
    """Two heads, two KV heads; ``head_dim`` 8 so the full-width q/k norm is 16 wide."""
    torch.manual_seed(0)
    cfg = dict(vocab_size=50, hidden_size=D, intermediate_size=32, num_hidden_layers=N_LAYERS,
               num_attention_heads=2, num_key_value_heads=2, head_dim=8, attn_implementation="eager")
    return Olmo2Model(Olmo2Config(**cfg | overrides)).float().eval().requires_grad_(False)


def realistic_model() -> Olmo2Model:
    """Grouped-query attention (4 query heads, 2 KV heads) with weights large enough that the
    MLP's RelP coefficients are not near zero, and random norm weights (the default init leaves
    them at 1, which cannot tell a dropped norm from the right one)."""
    model = tiny_model(num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                       initializer_range=0.2)
    torch.manual_seed(1)
    for name, p in model.named_parameters():
        if name.endswith("norm.weight"):  # post-attention, post-feedforward, q, k
            p.uniform_(0.3, 2.0)
    return model


def run_with(model, ids, layer: int | None, h_new: torch.Tensor):
    """Mid residuals (the MLP's inputs) and layer outputs, with layer ``layer``'s output
    replaced by ``h_new`` (``layer=None``: ``h_new`` replaces the embeddings)."""
    mids, outs = {}, {}
    hooks = []
    for i, block in enumerate(model.layers):
        hooks.append(block.mlp.register_forward_pre_hook(
            lambda _m, a, i=i: mids.__setitem__(i, a[0])))

        def out_hook(_m, _a, o, i=i):
            if i == layer:
                o = h_new[None]
            outs[i] = o
            return o
        hooks.append(block.register_forward_hook(out_hook))
    if layer is None:
        model(inputs_embeds=h_new[None])
    else:
        model(input_ids=ids[None])
    for h in hooks:
        h.remove()
    return mids, outs


def exact(model, ids, f):
    """Jacobian of ``f(mids, outs)`` w.r.t. the embeddings (``[*out, T, d]``)."""
    e = model.embed_tokens(ids[None])[0].detach()
    return torch.autograd.functional.jacobian(lambda x: f(*run_with(model, ids, None, x)), e)


def exact_from(model, ids, layer, f):
    """Jacobian of ``f(outs)`` w.r.t. layer ``layer``'s output (``[*out, T, d]``)."""
    _, outs = run_with(model, ids, None, model.embed_tokens(ids[None])[0].detach())
    h = outs[layer][0].detach()
    return torch.autograd.functional.jacobian(lambda x: f(run_with(model, ids, layer, x)[1]), h)


@pytest.mark.parametrize("n_valid", [7, 8])  # keys padded to a multiple of 8 (Tp), and not
@pytest.mark.parametrize("build", [tiny_model, realistic_model])
def test_estimates_match_brute_force(build, n_valid):
    """The hand-written OLMo 2 maps against autograd through the model under the RelP rules."""
    model = build()
    ids = torch.randint(0, 50, (SKIP_FIRST + n_valid,))
    valid = valid_positions(len(ids))
    r = Accumulator({"out": N_LAYERS - 1, "mid": N_LAYERS}, D)
    r_lens_doc(model, ids, r, 0, dim_batch=7)
    s = Accumulator({"mid": N_LAYERS, "out": N_LAYERS}, D)
    s_lens_doc(model, ids, s, 0, dim_batch=7)
    assert not torch.isnan(r.mean("out")).any() and not torch.isnan(s.mean("out")).any()
    with relp_rules(model):
        for layer in range(N_LAYERS - 1):  # J_L = mean_t Σ_t' ∂final(t')/∂h_L(t)
            J = exact_from(model, ids, layer,
                           lambda outs: outs[N_LAYERS - 1][0, valid].sum(0))
            assert torch.allclose(r.mean("out")[layer], J[:, valid].mean(1), atol=1e-4), ("r-out", layer)
        emb = model.embed_tokens(ids[None])[0].detach()
        for layer in range(N_LAYERS):  # the same from layer L's MLP input (the post-attention residual)
            block = model.layers[layer]
            mid = run_with(model, ids, None, emb)[0][layer][0].detach()

            def final_from_mid(x, layer=layer, block=block):
                out = x + block.post_feedforward_layernorm(block.mlp(x[None]))[0]
                return run_with(model, ids, layer, out)[1][N_LAYERS - 1][0, valid].sum(0)
            J = torch.autograd.functional.jacobian(final_from_mid, mid)
            assert torch.allclose(r.mean("mid")[layer], J[:, valid].mean(1), atol=1e-4), ("r-mid", layer)
        for layer in range(N_LAYERS):  # K_L = mean_t Σ_s ∂h_L(t)/∂e(s)
            for kind in ("mid", "out"):
                K = exact(model, ids, lambda mids, outs, layer=layer, kind=kind:
                          (mids if kind == "mid" else outs)[layer][0, valid].mean(0))
                assert torch.allclose(s.mean(kind)[layer], K[:, valid].sum(1), atol=1e-4), ("s", kind, layer)


def test_dispatch_selects_olmo_and_refuses_a_qwen_stack():
    """`transport._maps` must route OLMo 2 to these maps, and the OLMo map must refuse a Qwen3
    stack by name rather than compute a silently wrong Jacobian."""
    from transformers import Qwen3Config, Qwen3Model

    from neuron_lens import olmo2
    from neuron_lens.transport import _maps

    model = tiny_model()
    assert _maps(model)[0] is olmo2.linearize
    qwen = Qwen3Model(Qwen3Config(vocab_size=50, hidden_size=D, intermediate_size=32,
                                  num_hidden_layers=1, num_attention_heads=2,
                                  num_key_value_heads=2, head_dim=8, attn_implementation="eager"))
    assert not olmo2.is_olmo2(qwen)
    with pytest.raises(ValueError, match="not an OLMo 2"):
        olmo2.linearize(qwen, torch.randint(0, 50, (SKIP_FIRST + 7,)))


def test_bf16_map_runs_end_to_end():
    """The real fit is bf16: the maps must run in that dtype (OLMo 2's rotary table is fp32, so
    the rotated q/k need an explicit cast) and produce finite estimates. This is the regression
    guard for the dtype class, which the fp32 brute-force tests cannot see."""
    model = tiny_model().to(torch.bfloat16)
    ids = torch.randint(0, 50, (SKIP_FIRST + 8,))
    r = Accumulator({"out": N_LAYERS - 1, "mid": N_LAYERS}, D)
    s = Accumulator({"mid": N_LAYERS, "out": N_LAYERS}, D)
    r_lens_doc(model, ids, r, 0, dim_batch=7)
    s_lens_doc(model, ids, s, 0, dim_batch=7)
    for acc in (r, s):
        for target in ("out", "mid"):
            assert torch.isfinite(acc.mean(target)).all()
    assert r.mean("out").abs().sum() > 0


def test_relp_rules_keep_values_and_unpatch_on_olmo():
    """The reference patches OLMo 2's two post-norms and its MLP; forward values must not move,
    and the instance-level forwards must be gone afterwards."""
    model = tiny_model()
    e = torch.randn(1, 6, D)
    with torch.no_grad():
        plain = model(inputs_embeds=e).last_hidden_state
        with relp_rules(model):
            relp = model(inputs_embeds=e).last_hidden_state
    assert torch.equal(plain, relp)
    for module in (model.layers[0].mlp, model.layers[0].post_attention_layernorm,
                   model.layers[0].post_feedforward_layernorm):
        assert "forward" not in module.__dict__ and not hasattr(module, "_rules_orig_forward")


def test_olmo_qk_norms_are_full_width():
    """The map's `_check` depends on it: a per-head norm would silently be a different map."""
    model = tiny_model()
    at = model.layers[0].self_attn
    assert at.q_proj.weight.shape[0] == at.config.num_attention_heads * at.head_dim
    assert at.k_proj.weight.shape[0] == at.config.num_key_value_heads * at.head_dim
    # The norms span every head, not one head: Qwen3's are head_dim wide.
    assert at.q_norm.weight.shape[0] == at.q_proj.weight.shape[0]
    assert at.k_norm.weight.shape[0] == at.k_proj.weight.shape[0]
    assert at.q_norm.weight.shape[0] != at.head_dim


def test_split_half_is_zero_for_equal_halves_and_nan_for_one():
    acc = Accumulator({"out": 2}, 3)
    acc.sums[("out", 0)] += 1
    acc.n_docs = [1, 0]
    assert all(math.isnan(x) for x in acc.split_half("out"))
    acc.sums[("out", 1)] += 2
    acc.n_docs = [1, 2]
    assert acc.split_half("out") == [0.0, 0.0]
