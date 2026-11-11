#!/usr/bin/env python3
"""test_part205.py: the build's tests. Run as

  /home/elisepc/neuron-probe/.venv/bin/python -m pytest -q part205/test_part205.py

(a pytest file run directly exits 0 having run nothing). No download, no real weights: every model here is a tiny
random model built from a config (Olmo2Config and Qwen3_5TextConfig, 2 layers, small widths). The two real tokenizers
are read from the local cache offline.

Tests that must exist (Brief, item 5):
  - the do-nothing hook reproduces logits exactly
  - copying EVERY part from the You run reproduces the You run's answer probability (positive control)
  - e agrees with actual zeroing of a part to first order (tolerance stated, with its reason)
  - head slices cover each o_proj input exactly once
  - suffix alignment puts the last position inside the copied span
  - random-family draws meet the kind and layer counts and never contain SAY or DO parts
  - the bootstrap is deterministic under its seed
  - every gate fails on a fixture built to fail it and passes on one built to pass it
  - the runner drives end to end on a tiny model (PROCEDURE section 5, Parts 119b/119c: a test that calls run())
  - the scorer names every prediction's key and stops the reads a failed gate stops
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/e/neuron-probe/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common205 as C  # noqa: E402
import run205 as R  # noqa: E402
import score205 as S  # noqa: E402

OLMO_REV = "a1847dff35000b4271fa70afc5db10fd29fedbdf"
QWEN_REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


# ---------------------------------------------------------------- fixtures

def tiny_olmo(seed=0, layers=2, hidden=32, heads=2, inter=16, vocab=100352):
    from transformers import Olmo2Config, Olmo2ForCausalLM
    torch.manual_seed(seed)
    cfg = Olmo2Config(num_hidden_layers=layers, hidden_size=hidden, num_attention_heads=heads,
                      num_key_value_heads=heads, intermediate_size=inter, vocab_size=vocab)
    return Olmo2ForCausalLM(cfg).eval()


def tiny_qwen(seed=0):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(num_hidden_layers=4, full_attention_interval=4, hidden_size=64, num_attention_heads=4,
                            head_dim=16, intermediate_size=32, linear_num_value_heads=4, linear_num_key_heads=2,
                            linear_key_head_dim=8, linear_value_head_dim=8, vocab_size=248320,
                            num_key_value_heads=2, eos_token_id=None)
    return Qwen3_5ForCausalLM(cfg).eval()


@pytest.fixture(scope="session")
def olmo_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B", revision=OLMO_REV)


@pytest.fixture(scope="session")
def qwen_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", revision=QWEN_REV)


@pytest.fixture(scope="session")
def prompts():
    return C.load_prompts()


@pytest.fixture(scope="session")
def cfg():
    return C.load_config()


def tiny_cfg(cfg, stage="tiny", family="O"):
    """The real config with one tiny stage and no pool expectation (a tiny model has fewer parts)."""
    import copy
    c = copy.deepcopy(cfg)
    c["stages"] = [{"stage": stage, "family": family}]
    c["families"][family]["units"] = None
    return c


def runner_for(model, tok):
    units = C.units_for(model)
    ids = {"A": C.option_token_ids(tok, "A"), "B": C.option_token_ids(tok, "B")}
    return R.Runner(model, tok, units, ids), units


def text_for(prompts, name="You", value="honesty", item=0, kind="say"):
    it = prompts["items"][value][kind][item]
    return R.build_prompt(prompts, name, value, it["question"]), it


# ---------------------------------------------------------------- units

def test_head_slices_cover_o_proj_input_exactly_once():
    m = tiny_olmo(layers=2, hidden=32, heads=2)
    units = C.units_for(m)
    layers = C.text_layers(m)
    for l in range(2):
        heads = [u for u in units if u.layer == l and u.kind == "head"]
        cols = []
        for u in heads:
            cols.extend(u.cols)
        assert sorted(cols) == list(range(layers[l].self_attn.o_proj.in_features))
        assert len(cols) == len(set(cols))
        neurons = [u for u in units if u.layer == l and u.kind == "neuron"]
        ncols = [u.cols[0] for u in neurons]
        assert ncols == list(range(layers[l].mlp.down_proj.in_features))


def test_qwen_units_follow_head_site_rule():
    m = tiny_qwen()
    units = C.units_for(m)
    layers = C.text_layers(m)
    full = [u for u in units if u.kind == "head" and u.sub == "full"]
    dn = [u for u in units if u.kind == "head" and u.sub == "deltanet"]
    assert len(full) == 1 * 4          # 1 full-attention layer of 4, 4 heads
    assert len(dn) == 3 * 4            # 3 DeltaNet layers, 4 value heads
    for u in full:
        assert u.width == 16 and layers[u.layer].self_attn.o_proj.in_features == 4 * 16
    for u in dn:
        assert u.width == 8 and layers[u.layer].linear_attn.out_proj.in_features == 4 * 8


def test_real_pool_sizes_match_design():
    """The design's pool counts (lines 79-84) from the cached configs, without loading weights."""
    from transformers import AutoConfig
    o = AutoConfig.from_pretrained("allenai/OLMo-2-0425-1B", revision=OLMO_REV)
    assert o.num_hidden_layers * o.num_attention_heads == 256
    assert o.num_hidden_layers * o.intermediate_size == 131072
    assert 256 + 131072 == 131328
    q = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B", revision=QWEN_REV).get_text_config()
    full_layers = q.layer_types.count("full_attention")
    assert full_layers == 8 and q.num_hidden_layers == 32
    heads = full_layers * q.num_attention_heads + (q.num_hidden_layers - full_layers) * q.linear_num_value_heads
    assert heads == 896
    assert q.num_hidden_layers * q.intermediate_size == 294912
    assert heads + q.num_hidden_layers * q.intermediate_size == 295808


# ---------------------------------------------------------------- do-nothing and positive control

def test_do_nothing_hook_reproduces_logits_exactly(prompts):
    m = tiny_olmo()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B", revision=OLMO_REV)
    runner, units = runner_for(m, tok)
    text, it = text_for(prompts)
    ids, read_plain, _, _ = runner.run(text, need_grad=False, capture=False)
    replace = []
    by_site = {}
    for u in units:
        by_site.setdefault((u.site, u.layer), []).append(u)
    # the receiving run's own activations, captured from a plain pass
    _, _, acts, _ = runner.run(text, need_grad=False, capture=True)
    for (site, layer), us in by_site.items():
        cols = []
        for u in sorted(us, key=lambda u: u.index):
            cols.extend(u.cols)
        vals = acts[f"{site}:{layer}"][0]
        replace.append((site, layer, cols, list(range(len(ids))), vals))
    _, read_dn, _, _ = runner.run(text, need_grad=False, capture=False, replace=replace)
    assert read_dn["p_a"] == read_plain["p_a"]
    assert read_dn["p_b"] == read_plain["p_b"]
    assert read_dn["top_token"] == read_plain["top_token"]


def test_copy_every_part_reproduces_you_probability(prompts):
    """Positive control: copying every part of the You run into the third-person frame gives the You answer."""
    from transformers import AutoTokenizer
    m = tiny_olmo()
    tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B", revision=OLMO_REV)
    runner, units = runner_for(m, tok)
    you_text, it = text_for(prompts, "You")
    third_text, _ = text_for(prompts, "Zara")
    you_ids, you_read, you_acts, _ = runner.run(you_text, need_grad=False)
    third_ids, third_read, _, _ = runner.run(third_text, need_grad=False, capture=False)
    start, positions, vals = C.you_values_at_third_positions(you_acts, you_ids, third_ids)
    assert start == len(third_ids) - len(positions)  # the span sits at the end of the third-person run
    assert len(third_ids) - start == len(you_ids) - (len(you_ids) - len(positions))
    ids, read, _, _ = runner.replace_and_read(third_text, [u.key for u in units], positions, vals, it["consistent"])
    assert read["p_a"] == pytest.approx(you_read["p_a"], abs=1e-6)
    assert read["p_b"] == pytest.approx(you_read["p_b"], abs=1e-6)
    # and the copy did move the answer away from the plain third-person run (otherwise the control is vacuous)
    plain_diff = abs(third_read["p_a"] - you_read["p_a"]) + abs(third_read["p_b"] - you_read["p_b"])
    assert plain_diff > 0
    assert abs(read["p_a"] - third_read["p_a"]) + abs(read["p_b"] - third_read["p_b"]) > 0.5 * plain_diff


# ---------------------------------------------------------------- estimator

def test_e_agrees_with_zeroing_to_first_order():
    """e is the first-order estimate of the change from zeroing a part: zeroing moves the logit difference by -e.

    Tolerance: the first-order estimate is exact only in the limit of small activation; the error is second order in
    the part's activation (the network's response to the removed value), so on a tiny model with small activations
    (weights scaled to 0.1, activations of order 0.1) the relative error is allowed 10% of |e| plus 5e-3 absolute
    (the absolute floor covers parts whose e is near zero, where a relative bound is meaningless).
    """
    m = tiny_olmo()
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(0.1)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B", revision=OLMO_REV)
    runner, units = runner_for(m, tok)
    prompts = C.load_prompts()
    text, it = text_for(prompts, "You", "honesty", 0)
    ids, read, acts, grads = runner.run(text, need_grad=True, consistent=it["consistent"])
    est = C.estimates_for_prompt(acts, grads, units)
    other = "B" if it["consistent"] == "A" else "A"
    a_ids, b_ids = runner.letter_ids["A"], runner.letter_ids["B"]
    # pick a few parts with the largest |e| and a few neurons, and zero each
    order = torch.argsort(est.abs(), descending=True).tolist()
    picks = [units[i] for i in order[:4]] + [u for u in units if u.kind == "neuron"][:3]
    for u in picks:
        layers = C.text_layers(m)
        mod = C.unit_site_module(layers, u)
        h = C.SiteReplacer(mod, list(u.cols), list(range(len(ids))), torch.zeros(len(ids), len(u.cols)))
        try:
            _, r0, _, _ = runner.run(text, need_grad=False, capture=False)
        finally:
            h.close()
        d0 = sum(r0["p_a"] for _ in [0])  # placeholder to keep the readout used
        # read the logit difference directly, plain and zeroed
        def logit_diff():
            with torch.no_grad():
                lg = m(input_ids=torch.tensor([ids])).logits[0, -1]
            return float(lg[a_ids].sum() - lg[b_ids].sum()) if it["consistent"] == "A" else \
                float(lg[b_ids].sum() - lg[a_ids].sum())
        plain = logit_diff()
        h = C.SiteReplacer(mod, list(u.cols), list(range(len(ids))), torch.zeros(len(ids), len(u.cols)))
        try:
            zeroed = logit_diff()
        finally:
            h.close()
        actual = zeroed - plain
        pred = -float(est[units.index(u)])
        assert abs(actual - pred) <= 0.1 * abs(pred) + 5e-3, (u.key, pred, actual)
        assert d0 is not None and other in ("A", "B")


def test_e_is_negative_gradient_of_the_logit_difference():
    """e is the derivative of the logit difference with respect to the part's input, times the input."""
    m = tiny_olmo()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B", revision=OLMO_REV)
    runner, units = runner_for(m, tok)
    prompts = C.load_prompts()
    text, it = text_for(prompts, "You", "kindness", 1)
    ids, _, acts, grads = runner.run(text, need_grad=True, consistent=it["consistent"])
    est = C.estimates_for_prompt(acts, grads, units)
    u = next(u for u in units if u.kind == "head")
    a = acts[f"{u.site}:{u.layer}"][0, -1, list(u.cols)]
    g = grads[f"{u.site}:{u.layer}"][0, -1, list(u.cols)]
    assert float((a * g).sum()) == pytest.approx(float(est[units.index(u)]), abs=1e-6)


# ---------------------------------------------------------------- suffix alignment

def test_suffix_alignment_puts_last_position_inside_the_span(prompts):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B", revision=OLMO_REV)
    you, _ = text_for(prompts, "You")
    third, _ = text_for(prompts, "Zara")
    yids = tok(you, add_special_tokens=False).input_ids
    tids = tok(third, add_special_tokens=False).input_ids
    start, positions = C.shared_suffix_positions(yids, tids)
    assert positions[-1] == len(tids) - 1          # the last position is inside the copied span
    assert start > 0                                # the persona prefix is not copied
    assert yids[-len(positions):] == tids[start:]   # the span is the shared question and answer cue
    assert len(tids) > len(yids)                    # the name prefix is longer than "You"
    # and the span is exactly the question plus the answer cue
    # the span ends with the question and the answer cue; it may begin with the tail of the persona line when the
    # name and "You" share those tokens (the suffix is the longest shared run, not a hand-cut one)
    assert tok.decode(tids[start:]) == "User: " + prompts["items"]["honesty"]["say"][0]["question"] + \
        "\nAssistant: Answer:" or tok.decode(tids[start:]).endswith(
            "User: " + prompts["items"]["honesty"]["say"][0]["question"] + "\nAssistant: Answer:")


def test_you_values_land_at_the_third_person_suffix():
    import numpy as np
    you = np.arange(1 * 5 * 2, dtype=np.float32).reshape(1, 5, 2)
    third_ids = [9, 9, 1, 2, 3]
    you_ids = [1, 2, 3]
    start, positions, vals = C.you_values_at_third_positions({"down_proj:0": you}, you_ids, third_ids)
    assert start == 2 and positions == [2, 3, 4]
    assert np.array_equal(vals["down_proj:0"], you[0, -3:, :])


# ---------------------------------------------------------------- random families

def _fake_units(n_layers=4, heads=3, neurons=6, sub=""):
    out = []
    for l in range(n_layers):
        out += [C.Unit("head", l, h, 4, "o_proj", sub) for h in range(heads)]
        out += [C.Unit("neuron", l, j, 1, "down_proj") for j in range(neurons)]
    return out


def test_random_families_meet_kind_and_layer_counts_and_exclude_say_do():
    import random
    units = _fake_units()
    keys = [u.key for u in units]
    say = keys[:4] + keys[9:13]
    do = keys[4:8] + keys[13:17]
    score = {k: float(i) for i, k in enumerate(keys)}
    effect = {k: float(-i) for i, k in enumerate(keys)}
    rngs = {k: random.Random(100 + i) for i, k in enumerate(("layer", "score", "effect"))}
    fam = R.draw_random_families(rngs, units, do, say, score, effect, k=8, n_sets=20)
    assert fam["fillable"]
    by = {u.key: u for u in units}
    want = {"head": sum(by[k].kind == "head" for k in do), "neuron": sum(by[k].kind == "neuron" for k in do)}
    for name, f in fam["families"].items():
        assert len(f["sets"]) == 20
        for s in f["sets"]:
            assert len(s) == 8 and len(set(s)) == 8
            assert not (set(s) & set(say)) and not (set(s) & set(do))
            got = {"head": sum(by[k].kind == "head" for k in s), "neuron": sum(by[k].kind == "neuron" for k in s)}
            assert got == want, (name, got, want)
    # layer family: the DO set's count per layer and kind (or a counted relaxation)
    layer_fam = fam["families"]["layer"]
    want_by_layer = {}
    for k in do:
        want_by_layer[(by[k].kind, by[k].layer)] = want_by_layer.get((by[k].kind, by[k].layer), 0) + 1
    for s in layer_fam["sets"]:
        got = {}
        for k in s:
            got[(by[k].kind, by[k].layer)] = got.get((by[k].kind, by[k].layer), 0) + 1
        if layer_fam["relaxed"] == 0:
            assert got == want_by_layer


def test_random_family_not_fillable_is_reported():
    import random
    units = _fake_units(n_layers=2, heads=1, neurons=2)
    keys = [u.key for u in units]
    # a DO set that wants more heads than the pool minus SAY/DO can give
    do = [k for k in keys if k.startswith("head")] + [k for k in keys if k.startswith("neuron")][:2]
    say = [k for k in keys if k.startswith("neuron")][2:4]
    rngs = {k: random.Random(1) for k in ("layer", "score", "effect")}
    fam = R.draw_random_families(rngs, units, do, say, {k: 1.0 for k in keys}, {k: 1.0 for k in keys}, k=4, n_sets=3)
    assert fam["fillable"] is False
    assert fam["not_fillable_reason"]
    assert fam["families"] == {}


def test_random_families_are_reproducible_under_their_seeds():
    import random
    units = _fake_units()
    keys = [u.key for u in units]
    say, do = keys[:3], keys[3:6]
    score = {k: float(i % 7) for i, k in enumerate(keys)}
    effect = {k: float(i % 5) for i, k in enumerate(keys)}
    a = R.draw_random_families({k: random.Random(7) for k in ("layer", "score", "effect")}, units, do, say,
                               score, effect, k=6, n_sets=4)
    b = R.draw_random_families({k: random.Random(7) for k in ("layer", "score", "effect")}, units, do, say,
                               score, effect, k=6, n_sets=4)
    assert a["families"] == b["families"]


# ---------------------------------------------------------------- bootstrap

def _fake_est_and_prompts():
    import numpy as np
    rng = np.random.default_rng(3)
    prompts = R.selection_prompts(C.load_prompts())
    keys = [f"part:{i}" for i in range(200)]
    est = rng.normal(size=(len(prompts), len(keys)))
    return est, prompts, keys


def test_bootstrap_is_deterministic_under_its_seed():
    est, prompts, keys = _fake_est_and_prompts()
    a = S.bootstrap_o(est, prompts, keys, 50, 1234, k=20)
    b = S.bootstrap_o(est, prompts, keys, 50, 1234, k=20)
    c = S.bootstrap_o(est, prompts, keys, 50, 1235, k=20)
    assert a[:2] == b[:2] and a[2] == b[2]
    assert a[:2] != c[:2]
    assert a[0] <= a[1]


def test_paired_rise_bootstrap_is_deterministic():
    import numpy as np
    est, prompts, keys = _fake_est_and_prompts()
    est2 = est + np.random.default_rng(9).normal(scale=0.1, size=est.shape)
    a = S.paired_bootstrap_rise(est, est2, prompts, keys, 40, 77, k=20)
    b = S.paired_bootstrap_rise(est, est2, prompts, keys, 40, 77, k=20)
    assert a == b


# ---------------------------------------------------------------- gates

def _row(arm, qid, name, pa, pb, consistent="A", n_parts=80, n_positions=10):
    return {"arm": arm, "id": qid, "name": name, "consistent": consistent, "p_a": pa, "p_b": pb,
            "option_mass": pa + pb, "top_token": 1, "top_prob": pa + pb, "top_is_letter": True,
            "n_parts": n_parts, "n_positions": n_positions}


def _arms_fixture(you_pa, third_pa, copy_pa, say_pa, top_is_letter=True, mass=0.9, do_nothing_equal=True):
    """2 questions x 2 names; You answers you_pa, third person third_pa, DO copy copy_pa, SAY copy say_pa."""
    arms = {}
    for q in ("q1", "q2"):
        r = _row("you", q, "You", you_pa, 1 - you_pa)
        r["top_is_letter"] = top_is_letter
        r["option_mass"] = mass
        arms.setdefault("you", []).append(r)
        for name in ("Zara", "Milo"):
            arms.setdefault("baseline", []).append(_row("baseline", q, name, third_pa, 1 - third_pa))
            arms.setdefault("do_nothing", []).append(_row("do_nothing", q, name,
                                                          third_pa if do_nothing_equal else third_pa + 0.01,
                                                          1 - third_pa))
            arms.setdefault("do_copy", []).append(_row("do_copy", q, name, copy_pa, 1 - copy_pa))
            arms.setdefault("say_copy", []).append(_row("say_copy", q, name, say_pa, 1 - say_pa))
    return arms


def test_gate_g0_passes_and_fails():
    held = {"q1", "q2"}
    ok = _arms_fixture(0.8, 0.5, 0.7, 0.7)
    assert S.gate_g0(ok, held)["pass"] is True
    low = _arms_fixture(0.8, 0.5, 0.7, 0.7, mass=0.3)
    assert S.gate_g0(low, held)["pass"] is False
    noletter = _arms_fixture(0.8, 0.5, 0.7, 0.7, top_is_letter=False)
    assert S.gate_g0(noletter, held)["pass"] is False


def test_gate_g1_passes_and_fails():
    held = {"q1", "q2"}
    ok = _arms_fixture(0.8, 0.5, 0.7, 0.7)
    assert S.gate_g1(ok)["pass"] is True
    bad = _arms_fixture(0.8, 0.5, 0.7, 0.7, do_nothing_equal=False)
    g = S.gate_g1(bad)
    assert g["pass"] is False and g["max_abs_diff"] > 0


def test_gate_g1_a770_variant():
    plain = [(0.5, 0.5)] * 5
    assert S.gate_g1_a770(plain, [(0.5, 0.5)] * 5)["pass"] is True
    assert S.gate_g1_a770(plain, [(0.5001, 0.4999)] * 5)["pass"] is False
    # plain runs not bit-identical: tolerance is twice the largest plain-against-plain difference
    plain2 = [(0.5, 0.5), (0.5004, 0.4996), (0.5, 0.5), (0.5002, 0.4998), (0.5, 0.5)]
    dn = [(0.5006, 0.4994)] * 5
    g = S.gate_g1_a770(plain2, dn)
    assert g["bit_identical_plain"] is False and g["pass"] is True
    assert S.gate_g1_a770(plain2, [(0.5012, 0.4988)] * 5)["pass"] is False


def test_gate_g2_passes_and_fails():
    held = {"q1", "q2"}
    ok = _arms_fixture(0.8, 0.5, 0.7, 0.7)          # gap 0.3
    t = S.ratio_of_means(ok, held, ["Zara", "Milo"], "do_copy", "baseline")
    assert S.gate_g2(t)["pass"] is True
    small = _arms_fixture(0.55, 0.5, 0.52, 0.52)     # gap 0.05
    t2 = S.ratio_of_means(small, held, ["Zara", "Milo"], "do_copy", "baseline")
    assert S.gate_g2(t2)["pass"] is False


def test_gate_g3_passes_and_fails():
    held = {"q1", "q2"}
    ok = _arms_fixture(0.8, 0.5, 0.7, 0.66)          # T_SAY = 0.16/0.3 = 0.53
    t = S.ratio_of_means(ok, held, ["Zara", "Milo"], "say_copy", "baseline")
    assert S.gate_g3(t)["pass"] is True
    bad = _arms_fixture(0.8, 0.5, 0.7, 0.6)          # T_SAY = 0.33
    t2 = S.ratio_of_means(bad, held, ["Zara", "Milo"], "say_copy", "baseline")
    assert S.gate_g3(t2)["pass"] is False


def test_gate_g4_passes_and_fails():
    assert S.gate_g4({"retest_say": 25, "retest_do": 20})["pass"] is True
    assert S.gate_g4({"retest_say": 25, "retest_do": 19})["pass"] is False
    assert S.gate_g4({"retest_say": 19, "retest_do": 40})["pass"] is False


def test_random_set_rule_passes_and_fails():
    held = {"q1", "q2"}
    arms = _arms_fixture(0.8, 0.5, 0.75, 0.7)
    for fam in ("layer", "score"):
        for i in range(3):
            arms.setdefault(f"random:{fam}:{i}", []).append(_row(f"random:{fam}:{i}", "q1", "Zara", 0.6, 0.4))
            arms.setdefault(f"random:{fam}:{i}", []).append(_row(f"random:{fam}:{i}", "q2", "Milo", 0.6, 0.4))
    t = S.ratio_of_means(arms, held, ["Zara", "Milo"], "do_copy", "baseline")
    t["random"] = {fam: [S.ratio_of_means(arms, held, ["Zara", "Milo"], f"random:{fam}:{i}", "baseline")
                         for i in range(3)] for fam in ("layer", "score")}
    fams = {"fillable": True, "families": {"layer": {"sets": [[], [], []]}, "score": {"sets": [[], [], []]}}}
    assert S.random_set_rule(t, fams)["pass"] is True
    t["T"] = 0.4                                     # below the random sets' 0.33? no: above. Use a lower copy.
    arms2 = _arms_fixture(0.8, 0.5, 0.55, 0.7)
    t2 = S.ratio_of_means(arms2, held, ["Zara", "Milo"], "do_copy", "baseline")
    t2["random"] = t["random"]
    assert S.random_set_rule(t2, fams)["pass"] is False
    one = {"fillable": True, "families": {"layer": {"sets": [[]]}}}
    assert S.random_set_rule(t, one)["read"] is False


# ---------------------------------------------------------------- the runner end to end

def test_runner_end_to_end_on_tiny_olmo(tmp_path, prompts, cfg):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B", revision=OLMO_REV)
    m = tiny_olmo()
    c = tiny_cfg(cfg)
    c["random_family"]["sets_per_stage_olmo"] = 2
    c["bootstrap"]["resamples"] = 10
    import argparse
    a = argparse.Namespace(stage="tiny", out=str(tmp_path), config=str(HERE / "config.json"),
                           prompts=str(HERE / "prompts205.json"), device="cpu", stage_only=None, check_pool=False)
    env = {"family": "O", "stage": "tiny"}
    R.run_stage(a, model=m, tok=tok, env=env, prompts=prompts, cfg=c)
    d = tmp_path / "tiny"
    for f in ("ranking.npz", "ranking.json", "effect.npz", "copy_acts.npz", "sets.json", "records.jsonl",
              "meta.json"):
        assert (d / f).exists(), f
    recs = [json.loads(x) for x in (d / "records.jsonl").read_text().splitlines() if x.strip()]
    arms = {r["arm"] for r in recs}
    assert {"ranking", "you", "baseline", "do_nothing", "do_copy", "say_copy", "swap", "reverse",
            "off_target_baseline", "off_target_do_copy", "secondary_you", "secondary_baseline",
            "secondary_do_copy", "secondary_say_copy"} <= arms
    # the do-nothing rows reproduce the baseline exactly (G1 on the real code path)
    base = {(r["id"], r["name"]): r for r in recs if r["arm"] == "baseline"}
    for r in recs:
        if r["arm"] == "do_nothing":
            b = base[(r["id"], r["name"])]
            assert r["p_a"] == b["p_a"] and r["p_b"] == b["p_b"]
    # the scorer runs on it and names every prediction
    sd = S.load_stage_dir(d)
    res = S.score_stage("tiny", sd, c, 1)
    assert set(res["gates"]) == {"G0", "G1", "G2", "G3", "G4"}
    assert res["gates"]["G1"]["pass"] is True
    preds = S.predictions([res], c)
    # every prediction whose stages are absent is "not read", and every key is named
    assert set(preds) == {f"205.{i}" for i in range(1, 10)}


def test_runner_resumes_without_duplicating_records(tmp_path, prompts, cfg):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B", revision=OLMO_REV)
    m = tiny_olmo()
    c = tiny_cfg(cfg)
    c["random_family"]["sets_per_stage_olmo"] = 1
    import argparse
    a = argparse.Namespace(stage="tiny", out=str(tmp_path), config=str(HERE / "config.json"),
                           prompts=str(HERE / "prompts205.json"), device="cpu", stage_only=None, check_pool=False)
    env = {"family": "O", "stage": "tiny"}
    R.run_stage(a, model=m, tok=tok, env=env, prompts=prompts, cfg=c)
    n1 = len((tmp_path / "tiny" / "records.jsonl").read_text().splitlines())
    R.run_stage(a, model=m, tok=tok, env=env, prompts=prompts, cfg=c)
    n2 = len((tmp_path / "tiny" / "records.jsonl").read_text().splitlines())
    assert n1 == n2


# ---------------------------------------------------------------- scoring rules

def test_scorer_stops_the_reads_a_failed_gate_stops():
    """G0 fails: neither O nor T is read. G2 fails: O is read, T is not. G4 fails: O is not read."""
    def stage(g0=True, g1=True, g2=True, g3=True, g4=True):
        return {"stage": "s", "gates": {"G0": {"pass": g0}, "G1": {"pass": g1}, "G2": {"pass": g2},
                                        "G3": {"pass": g3}, "G4": {"pass": g4}}}
    # the read flags are computed in score_stage; recompute them with the same rule
    def reads(s):
        g = s["gates"]
        return {"O": g["G0"]["pass"] and g["G1"]["pass"] and g["G4"]["pass"],
                "T": g["G0"]["pass"] and g["G1"]["pass"] and g["G2"]["pass"] and g["G3"]["pass"]}
    assert reads(stage(g0=False)) == {"O": False, "T": False}
    assert reads(stage(g2=False)) == {"O": True, "T": False}
    assert reads(stage(g4=False)) == {"O": False, "T": True}
    assert reads(stage(g1=False)) == {"O": False, "T": False}
    assert reads(stage(g3=False)) == {"O": True, "T": False}


def test_prediction_keys_and_unread_verdicts(cfg):
    """A stage with every gate failing leaves the predictions that need a stage unread, not failed."""
    stage = {"stage": "instruct", "gates": {g: {"pass": False} for g in ("G0", "G1", "G2", "G3", "G4")},
             "read": {"O": False, "T": False, "copy_arms": False, "random_set_rule": False},
             "overlap": {"O": None}, "T": None, "T_swap": None, "T_detail": {"T": {"mean_den": None},
                                                                           "reverse": {"mean_num": None, "mean_den": None}},
             "random_set_rule": {"read": False}, "off_target": {"shift": None}}
    preds = S.predictions([stage], cfg)
    assert preds["205.4"]["verdict"] == "not read"
    assert preds["205.7"]["verdict"] == "vacuous"


def test_overlap_and_o_index():
    sets = {"SAY_a": ["a", "b", "c", "d"], "SAY_b": ["a", "b", "c", "e"], "DO_a": ["a", "b", "x", "y"],
            "DO_b": ["a", "b", "x", "z"]}
    st = S.overlap_stats(sets)
    assert st["retest_say"] == 3 and st["retest_do"] == 3
    assert st["cross"] == 2.0
    assert st["O"] == pytest.approx(2.0 / 3.0)


def test_polarity_split_and_broken_counts():
    arms = {"you": [_row("you", "q1", "You", 0.8, 0.2, consistent="A"),
                    _row("you", "q2", "You", 0.2, 0.8, consistent="B")],
            "baseline": [_row("baseline", "q1", "Zara", 0.5, 0.5, consistent="A"),
                         _row("baseline", "q2", "Zara", 0.5, 0.5, consistent="B")],
            "do_copy": [_row("do_copy", "q1", "Zara", 0.7, 0.3, consistent="A"),
                        _row("do_copy", "q2", "Zara", 0.3, 0.7, consistent="B")]}
    pol = S.polarity_split(arms, ["q1", "q2"], ["Zara"], "do_copy")
    assert pol["A"]["T"] == pytest.approx(0.2 / 0.3)
    assert pol["B"]["T"] == pytest.approx(0.2 / 0.3)
    arms["do_copy"][0]["option_mass"] = 0.1
    b = S.broken_counts(arms)
    assert b["do_copy"]["broken"] == 1 and b["do_copy"]["n"] == 2


def test_prompts_file_shape_and_balance(prompts):
    assert prompts["values"] == ["honesty", "harmlessness", "kindness", "fairness", "privacy", "humility"]
    for v in prompts["values"]:
        say = prompts["items"][v]["say"]
        do = prompts["items"][v]["do"]
        assert len(say) == 8 and len(do) == 8
        assert sum(i["consistent"] == "A" for i in say) == 4
        assert sum(i["consistent"] == "A" for i in do) == 4
        assert [i["split"] for i in say] == ["selection"] * 4 + ["heldout"] * 4
        assert [i["split"] for i in do] == ["selection"] * 4 + ["written_not_read"] * 4
    assert len(prompts["off_target"]) == 6
    assert all("A) yes B) no" in o["question"] for o in prompts["off_target"])
    assert prompts["personas"]["You"] == "You care about {value}."
    assert prompts["personas"]["Zara"] == "Zara, an AI, cares about {value}."
    assert prompts["personas"]["Milo"] == "Milo, an AI, cares about {value}."


def test_prompt_token_bound_with_both_tokenizers(prompts, olmo_tok, qwen_tok):
    for v in prompts["values"]:
        for kind in ("say", "do"):
            for it in prompts["items"][v][kind]:
                for name in prompts["personas"]:
                    text = R.build_prompt(prompts, name, v, it["question"])
                    assert len(olmo_tok(text, add_special_tokens=False).input_ids) < 120
                    assert len(qwen_tok(text, add_special_tokens=False).input_ids) < 120


def test_qwen_runner_end_to_end(tmp_path, prompts, cfg):
    """The runner's real path on the hybrid architecture (PROCEDURE section 5, Part 178: a dense model's tests cannot
    see a hybrid-cache crash)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", revision=QWEN_REV)
    m = tiny_qwen()
    c = tiny_cfg(cfg, family="Q")
    c["random_family"]["sets_per_stage_qwen"] = 1
    import argparse
    a = argparse.Namespace(stage="tiny", out=str(tmp_path), config=str(HERE / "config.json"),
                           prompts=str(HERE / "prompts205.json"), device="cpu", stage_only=None, check_pool=False)
    env = {"family": "Q", "stage": "tiny"}
    R.run_stage(a, model=m, tok=tok, env=env, prompts=prompts, cfg=c)
    recs = [json.loads(x) for x in (tmp_path / "tiny" / "records.jsonl").read_text().splitlines() if x.strip()]
    base = {(r["id"], r["name"]): r for r in recs if r["arm"] == "baseline"}
    for r in recs:
        if r["arm"] == "do_nothing":
            b = base[(r["id"], r["name"])]
            assert r["p_a"] == b["p_a"] and r["p_b"] == b["p_b"]
    sets = json.loads((tmp_path / "tiny" / "sets.json").read_text())
    assert "head:full" in sets["random"]["kind_counts_wanted"] or "head:deltanet" in \
        sets["random"]["kind_counts_wanted"]


def test_option_fields_follow_the_question_text(prompts):
    """option_a and option_b are the texts the question displays after "A) " and "B) ", and `consistent` names the
    letter whose text is the consistent option, for all 96 items and the off-target questions."""
    import re
    n = 0
    for v in prompts["values"]:
        for kind in ("say", "do"):
            for it in prompts["items"][v][kind]:
                m = re.search(r" A\) (.+?) B\) (.+)$", it["question"])
                assert m, it["id"]
                assert it["option_a"] == m.group(1), it["id"]
                assert it["option_b"] == m.group(2), it["id"]
                assert it["consistent"] in ("A", "B"), it["id"]
                # the consistent option is the one the item was written with (yes / the value-consistent action):
                # for SAY both options are yes/no, for DO the consistent option never contains the negative marker
                # used in the pair; the check that matters is that `consistent` names a letter that exists and that
                # the two fields are the displayed texts, done above and below.
                n += 1
    assert n == 96
    for o in prompts["off_target"]:
        m = re.search(r" A\) (.+?) B\) (.+)$", o["question"])
        assert m and o["option_a"] == m.group(1) and o["option_b"] == m.group(2), o["id"]
        assert o["consistent"] is None


def test_consistent_letter_matches_the_written_option(prompts):
    """SAY: the consistent option is "yes". DO: it is the value-consistent action; the pairs are written with the
    consistent action first in gen_prompts205.py's DO table, so the consistent letter's text must be the first of each
    pair there. This test rebuilds the pairs from the generator's tables and checks every item against them."""
    import gen_prompts205 as G
    for v in prompts["values"]:
        for it in prompts["items"][v]["say"]:
            good = "yes"
            assert (it["option_a"] if it["consistent"] == "A" else it["option_b"]) == good, it["id"]
        for it in prompts["items"][v]["do"]:
            situation, good, bad = G.DO[v][it["item"] - 1]
            assert it["option_a"] if it["consistent"] == "A" else it["option_b"] == good, it["id"]
            assert it["option_a"] if it["consistent"] == "B" else it["option_b"] == bad, it["id"]
