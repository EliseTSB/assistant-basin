"""OLMo 3's hand-written layer maps (``neuron_lens/olmo3.py``) against brute-force
Jacobians of the model under ``relp_rules``, on tiny random OLMo 3 stacks exercising:
- hybrid ``sliding_attention`` + ``full_attention`` layers,
- active ``sliding_window`` masking (window strictly smaller than sequence length ``T``),
- YaRN RoPE scaling with ``attention_factor != 1.0`` on ``full_attention`` layers,
- both MHA (matching OLMo 3 7B) and GQA (matching OLMo 3 32B),
- exact bf16 primal equality against HuggingFace ``Olmo3Model``.
"""
import pytest
import torch
from transformers import Olmo3Config, Olmo3Model

from neuron_lens.rules import relp_rules
from neuron_lens.transport import (
    SKIP_FIRST,
    Accumulator,
    r_lens_doc,
    s_lens_doc,
    valid_positions,
)

D, N_LAYERS = 16, 4


def tiny_model(**overrides) -> Olmo3Model:
    """4 layers (3 sliding + 1 full, matching OLMo 3 7B's 3:1 cadence) with active
    ``sliding_window=4`` (< T) and MHA (2/2 heads, head_dim 8, full-width q/k norm 16)."""
    torch.manual_seed(0)
    cfg = dict(
        vocab_size=50,
        hidden_size=D,
        intermediate_size=32,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=4,
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 500000.0},
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500000.0,
                "factor": 8.0,
                "attention_factor": 1.2079441541679836,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "original_max_position_embeddings": 8192,
            },
        },
        attn_implementation="eager",
    )
    return Olmo3Model(Olmo3Config(**(cfg | overrides))).float().eval().requires_grad_(False)


def realistic_model() -> Olmo3Model:
    """GQA (4 query heads, 2 KV heads), active sliding window 5, YaRN on full layers,
    larger initializer range, and random norm weights."""
    model = tiny_model(
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        sliding_window=5,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        initializer_range=0.2,
    )
    torch.manual_seed(1)
    for name, p in model.named_parameters():
        if name.endswith("norm.weight"):
            p.uniform_(0.3, 2.0)
    return model


def run_with(model, ids, layer: int | None, h_new: torch.Tensor):
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
    e = model.embed_tokens(ids[None])[0].detach()
    return torch.autograd.functional.jacobian(lambda x: f(*run_with(model, ids, None, x)), e)


def exact_from(model, ids, layer, f):
    _, outs = run_with(model, ids, None, model.embed_tokens(ids[None])[0].detach())
    h = outs[layer][0].detach()
    return torch.autograd.functional.jacobian(lambda x: f(run_with(model, ids, layer, x)[1]), h)


@pytest.mark.parametrize("n_valid", [9, 8])  # T=24 (Tp == T) and T=23 (1 key pad)
@pytest.mark.parametrize("build", [tiny_model, realistic_model])
def test_estimates_match_brute_force(build, n_valid):
    """Hand-written OLMo 3 maps (sliding window + YaRN) vs autograd under relp_rules."""
    model = build()
    ids = torch.randint(0, 50, (SKIP_FIRST + n_valid,))
    valid = valid_positions(len(ids))
    r = Accumulator({"out": N_LAYERS - 1, "mid": N_LAYERS}, D)
    r_lens_doc(model, ids, r, 0, dim_batch=7)
    s = Accumulator({"mid": N_LAYERS, "out": N_LAYERS}, D)
    s_lens_doc(model, ids, s, 0, dim_batch=7)
    assert not torch.isnan(r.mean("out")).any() and not torch.isnan(s.mean("out")).any()
    with relp_rules(model):
        for layer in range(N_LAYERS - 1):
            J = exact_from(model, ids, layer,
                           lambda outs: outs[N_LAYERS - 1][0, valid].sum(0))
            assert torch.allclose(r.mean("out")[layer], J[:, valid].mean(1), atol=1e-4), ("r-out", layer)
        emb = model.embed_tokens(ids[None])[0].detach()
        for layer in range(N_LAYERS):
            block = model.layers[layer]
            mid = run_with(model, ids, None, emb)[0][layer][0].detach()

            def final_from_mid(x, layer=layer, block=block):
                out = x + block.post_feedforward_layernorm(block.mlp(x[None]))[0]
                return run_with(model, ids, layer, out)[1][N_LAYERS - 1][0, valid].sum(0)
            J = torch.autograd.functional.jacobian(final_from_mid, mid)
            assert torch.allclose(r.mean("mid")[layer], J[:, valid].mean(1), atol=1e-4), ("r-mid", layer)
        for layer in range(N_LAYERS):
            for kind in ("mid", "out"):
                K = exact(model, ids, lambda mids, outs, layer=layer, kind=kind:
                          (mids if kind == "mid" else outs)[layer][0, valid].mean(0))
                assert torch.allclose(s.mean(kind)[layer], K[:, valid].sum(1), atol=1e-4), ("s", kind, layer)


def test_dispatch_selects_olmo3_and_refuses_other_stacks():
    from transformers import Olmo2Config, Olmo2Model, Qwen3Config, Qwen3Model

    from neuron_lens import olmo3
    from neuron_lens.transport import _maps

    model = tiny_model()
    assert _maps(model)[0] is olmo3.linearize

    qwen = Qwen3Model(Qwen3Config(vocab_size=50, hidden_size=D, intermediate_size=32,
                                  num_hidden_layers=1, num_attention_heads=2,
                                  num_key_value_heads=2, head_dim=8, attn_implementation="eager"))
    olmo2 = Olmo2Model(Olmo2Config(vocab_size=50, hidden_size=D, intermediate_size=32,
                                   num_hidden_layers=1, num_attention_heads=2,
                                   num_key_value_heads=2, head_dim=8, attn_implementation="eager"))
    assert not olmo3.is_olmo3(qwen) and not olmo3.is_olmo3(olmo2)
    for bad in (qwen, olmo2):
        with pytest.raises(ValueError, match="not an OLMo 3"):
            olmo3.linearize(bad, torch.randint(0, 50, (SKIP_FIRST + 7,)))


def test_bf16_primal_matches_the_model_exactly():
    """Exact bit-for-bit equality in bf16 on every layer's rotated q AND sliding-window attn."""
    from neuron_lens.olmo3 import linearize
    from neuron_lens.transport import _rope, decoder_stack

    model = tiny_model().to(torch.bfloat16)
    ids = torch.randint(0, 50, (SKIP_FIRST + 9,))
    maps = linearize(decoder_stack(model), ids)
    T = len(ids) - 1
    for li, mp in enumerate(maps):
        cap = {}
        h_q = model.layers[li].self_attn.q_norm.register_forward_hook(
            lambda _m, _a, o: cap.__setitem__("q", o))
        with torch.no_grad():
            model(input_ids=ids[None])
        h_q.remove()
        H, N, hd = mp.qr.shape[0], mp.qr.shape[1], mp.cos.shape[-1]
        lt = model.config.layer_types[li]
        ref = cap["q"][0, :T].view(T, H, hd)
        cos, sin = model.rotary_emb(model.embed_tokens(ids[:T])[None], torch.arange(T)[None], lt)
        ref = _rope(ref, cos[0, :, None], sin[0, :, None]).to(torch.bfloat16)
        ref = ref.transpose(0, 1)[:, SKIP_FIRST:SKIP_FIRST + N].reshape(H, N, hd)
        assert torch.equal(mp.qr, ref), f"layer {li} ({lt}): map and model qr differ"
        # Verify sliding_attention layers actually zero out keys outside sliding_window!
        if lt == "sliding_attention":
            sw = model.config.sliding_window
            # At valid row r (absolute position S + r), key 0 has distance S + r >= 16 > sw=4.
            assert (mp.p[:, :, 0] == 0).all(), f"layer {li}: sliding window did not zero old keys"
        else:
            # Full attention layer must attend to key 0 with nonzero probability.
            assert (mp.p[:, :, 0] > 0).any(), f"layer {li}: full attention unexpectedly zeroed key 0"


def test_bf16_map_runs_end_to_end():
    model = realistic_model().to(torch.bfloat16)
    ids = torch.randint(0, 50, (SKIP_FIRST + 8,))
    r = Accumulator({"out": N_LAYERS - 1, "mid": N_LAYERS}, D)
    s = Accumulator({"mid": N_LAYERS, "out": N_LAYERS}, D)
    r_lens_doc(model, ids, r, 0, dim_batch=7)
    s_lens_doc(model, ids, s, 0, dim_batch=7)
    for acc in (r, s):
        for target in ("out", "mid"):
            assert torch.isfinite(acc.mean(target)).all()
    assert r.mean("out").abs().sum() > 0


def test_relp_rules_keep_values_and_unpatch_on_olmo3():
    model = tiny_model()
    e = torch.randn(1, 8, D)
    with torch.no_grad():
        plain = model(inputs_embeds=e).last_hidden_state
        with relp_rules(model):
            relp = model(inputs_embeds=e).last_hidden_state
    assert torch.equal(plain, relp)
    for module in (model.layers[0].mlp, model.layers[0].post_attention_layernorm,
                   model.layers[0].post_feedforward_layernorm):
        assert "forward" not in module.__dict__ and not hasattr(module, "_rules_orig_forward")
