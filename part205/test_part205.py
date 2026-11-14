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
Fold of code review r1 (reviews/cheap-part205-code-{haiku,ds}-r1.md): every fix has a test here that fails on the
dfa83950 code; the test names carry the finding (h_f1 ... d_f19).
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


def tiny_cfg(cfg, stage="tiny", family="O", top_k=None, n_sets=None):
    """The real config with one tiny stage and no pool expectation (a tiny model has fewer parts)."""
    import copy
    c = copy.deepcopy(cfg)
    c["stages"] = [{"stage": stage, "family": family}]
    c["families"][family]["units"] = None
    if top_k is not None:
        c["top_k"] = top_k
    if n_sets is not None:
        c["random_family"]["sets_per_stage_olmo"] = n_sets
        c["random_family"]["sets_per_stage_qwen"] = n_sets
    c["bootstrap"]["resamples"] = 10
    return c


def stage_args(out, stage="tiny", prompts_file="prompts205.json", stage_only=None):
    import argparse
    return argparse.Namespace(stage=stage, out=str(out), config=str(HERE / "config.json"),
                              prompts=str(HERE / prompts_file), device="cpu", stage_only=stage_only, check_pool=False)


def wide_olmo(seed=0):
    """A tiny OLMo with enough heads (2 layers x 8) that a top-4 DO set leaves every kind fillable."""
    return tiny_olmo(seed=seed, layers=2, hidden=64, heads=8, inter=32)


def records(d):
    return C.read_jsonl(Path(d) / "records.jsonl")


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory, cfg, prompts):
    """ONE real run_stage on a tiny OLMo (every arm, fillable random families), shared by the read-only tests."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B", revision=OLMO_REV)
    out = tmp_path_factory.mktemp("tiny_run")
    c = tiny_cfg(cfg, top_k=4, n_sets=2)
    R.run_stage(stage_args(out), model=wide_olmo(), tok=tok, env={"family": "O", "stage": "tiny"}, prompts=prompts,
                cfg=c)
    return {"dir": out / "tiny", "cfg": c, "tok": tok}


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

def _frames(tok, prompts, value="honesty", item=4, name="Zara"):
    """(You frame, third-person frame) as dicts with text, ids and the question block's first token."""
    it = prompts["items"][value]["say"][item]
    out = []
    for n in ("You", name):
        persona = R.persona_text(prompts, n, value)
        text = C.frame_text(persona, it["question"])
        q0, ids = C.question_start(tok, text, persona)
        out.append({"text": text, "ids": ids, "q0": q0})
    return out[0], out[1], it


def test_do_nothing_hook_reproduces_full_logits_on_the_copy_path(prompts, olmo_tok):
    """D-F5: the do-nothing replacement on the path the copy arms take (replace_and_read, the copied span's positions,
    the receiving run's own activations) reproduces the FULL last-position logits exactly, and a hook that is off by
    one ulp at one column does not (so the comparison can fail)."""
    m = tiny_olmo()
    runner, units = runner_for(m, olmo_tok)
    y, t, it = _frames(olmo_tok, prompts)
    _, dst = C.copy_span(y["ids"], y["q0"], t["ids"], t["q0"])
    _, _, acts, _ = runner.run(t["text"], need_grad=True, consistent=it["consistent"])
    plain = runner.last_logits
    keys = [u.key for u in units]
    runner.replace_and_read(t["text"], keys, dst, C.values_at(acts, dst), it["consistent"])
    assert C.max_abs_logit_diff(runner.last_logits, plain) == 0.0
    bad = C.values_at(acts, dst)
    site = next(iter(bad))
    bad[site][-1, 0] += 1e-3
    runner.replace_and_read(t["text"], keys, dst, bad, it["consistent"])
    assert C.max_abs_logit_diff(runner.last_logits, plain) > 0.0


def test_copy_every_part_reproduces_you_probability(prompts, olmo_tok):
    """Positive control: copying every part of the You run over the copied span into the third-person frame gives
    the You answer (OLMo has no absolute position embedding, and the span starts at the question block)."""
    m = tiny_olmo()
    runner, units = runner_for(m, olmo_tok)
    y, t, it = _frames(olmo_tok, prompts)
    _, you_read, you_acts, _ = runner.run(y["text"], need_grad=False)
    _, third_read, _, _ = runner.run(t["text"], need_grad=False, capture=False)
    src, dst = C.copy_span(y["ids"], y["q0"], t["ids"], t["q0"])
    _, read, _, _ = runner.replace_and_read(t["text"], [u.key for u in units], dst, C.values_at(you_acts, src),
                                            it["consistent"])
    assert read["p_a"] == pytest.approx(you_read["p_a"], abs=1e-6)
    assert read["mass_a"] == pytest.approx(you_read["mass_a"], rel=1e-5)
    plain_diff = abs(third_read["p_a"] - you_read["p_a"])
    assert plain_diff > 0                                    # the control is not vacuous
    assert abs(read["p_a"] - third_read["p_a"]) > 0.5 * plain_diff


# ---------------------------------------------------------------- readout (H-F10)

def test_h_f10_readout_is_normalised_over_the_two_letters_with_raw_masses_kept():
    lg = torch.full((10,), -30.0)
    lg[0] = 0.0                       # a non-letter token takes most of the mass
    lg[1] = float(torch.log(torch.tensor(0.2 / 0.7)))
    lg[2] = float(torch.log(torch.tensor(0.1 / 0.7)))
    r = C.readout(lg, [1], [2])
    assert r["p_a"] == pytest.approx(2 / 3, rel=1e-5) and r["p_b"] == pytest.approx(1 / 3, rel=1e-5)
    assert r["p_a"] + r["p_b"] == pytest.approx(1.0)
    assert r["mass_a"] == pytest.approx(0.2, rel=1e-4) and r["mass_b"] == pytest.approx(0.1, rel=1e-4)
    assert r["option_mass"] == pytest.approx(0.3, rel=1e-4)


# ---------------------------------------------------------------- estimator

def _zeroing_check(olmo_tok, scale, kind, n, rel, floor):
    m = tiny_olmo()
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(scale)
    runner, units = runner_for(m, olmo_tok)
    prompts = C.load_prompts()
    text, it = text_for(prompts, "You", "honesty", 0)
    ids, _, acts, grads = runner.run(text, need_grad=True, consistent=it["consistent"])
    est = C.estimates_for_prompt(acts, grads, units)
    a_ids, b_ids = runner.letter_ids["A"], runner.letter_ids["B"]
    sign = 1.0 if it["consistent"] == "A" else -1.0
    layers = C.text_layers(m)

    def logit_diff():
        with torch.no_grad():
            lg = m(input_ids=torch.tensor([ids])).logits[0, -1]
        return sign * float(lg[a_ids].sum() - lg[b_ids].sum())

    plain = logit_diff()
    order = [i for i in torch.argsort(est.abs(), descending=True).tolist() if units[i].kind == kind][:n]
    for i in order:
        u = units[i]
        pred = -float(est[i])
        assert abs(pred) > 10 * floor, (u.key, pred)      # the check is not decided by the absolute floor
        h = C.SiteReplacer(C.unit_site_module(layers, u), list(u.cols), list(range(len(ids))),
                           torch.zeros(len(ids), len(u.cols)))
        try:
            actual = logit_diff() - plain
        finally:
            h.close()
        assert actual * pred > 0, (u.key, pred, actual)  # same sign
        assert abs(actual - pred) <= rel * abs(pred) + floor, (u.key, pred, actual)


def test_e_agrees_with_zeroing_to_first_order(olmo_tok):
    """e is the first-order estimate of the change from zeroing a part: zeroing moves the logit difference by -e.

    D-F17: the old test scaled weights to 0.1, where every e rounds to ~0 and its 5e-3 absolute floor decided the
    outcome. Now each kind is checked where its e is far above the floor and the first-order error is small: the four
    heads with the largest |e| with weights x0.5 (|e| ~ 2e-3 to 6e-3; tolerance 40% of |e| + 1e-4, the error being
    second order in the head's 8-wide activation), and the four neurons with the largest |e| at the initial weights
    (|e| ~ 6e-4 to 1.5e-3; tolerance 5% + 1e-5, a single neuron is close to linear). The sign must match in both.
    """
    _zeroing_check(olmo_tok, 0.5, "head", 4, 0.40, 1e-4)
    _zeroing_check(olmo_tok, 1.0, "neuron", 4, 0.05, 1e-5)


def test_e_is_negative_gradient_of_the_logit_difference(olmo_tok):
    """e is the derivative of the logit difference with respect to the part's input, times the input."""
    m = tiny_olmo()
    runner, units = runner_for(m, olmo_tok)
    prompts = C.load_prompts()
    text, it = text_for(prompts, "You", "kindness", 1)
    ids, _, acts, grads = runner.run(text, need_grad=True, consistent=it["consistent"])
    est = C.estimates_for_prompt(acts, grads, units)
    u = next(u for u in units if u.kind == "head")
    a = acts[f"{u.site}:{u.layer}"][0, -1, list(u.cols)]
    g = grads[f"{u.site}:{u.layer}"][0, -1, list(u.cols)]
    assert float((a * g).sum()) == pytest.approx(float(est[units.index(u)]), abs=1e-6)


# ---------------------------------------------------------------- the copied span (H-F4, D-F1, D-F2b, D-F3)

def _all_frame_pairs(prompts):
    """(source persona, receiving persona, value, question, personas) for every copy the runner makes."""
    names = [n for n in prompts["personas"] if n != "You"]
    out = []
    for v in prompts["values"]:
        for it in prompts["items"][v]["say"]:
            for n in names:
                out.append(("You", n, v, it["question"], None))        # DO/SAY/random copies
                out.append((n, "You", v, it["question"], None))        # reverse
                out.append(([m for m in names if m != n][0], n, v, it["question"], None))   # persona swap
        pq = prompts["secondary"]["third_person_question"].format(value=v)
        for n in names:
            out.append(("You", n, v, pq, None))                        # secondary: the source asks the SAME question
    for o in prompts["off_target"]:
        for n in names:
            out.append(("You", n, None, o["question"], prompts["off_target_personas"]))
    return out


@pytest.mark.parametrize("which", ["olmo", "qwen"])
def test_h_f4_d_f1_copied_span_is_the_question_block_only(prompts, olmo_tok, qwen_tok, which):
    """Design line 109: the copied span is the question and answer cue, identical tokens in both frames, aligned from
    the end. With both real tokenizers, for every copy the runner makes: the copied tokens are identical in the two
    frames, the first copied token is the first token of the "User:" line, and no persona token is copied."""
    tok = olmo_tok if which == "olmo" else qwen_tok
    n = 0
    for src_name, dst_name, v, q, personas in _all_frame_pairs(prompts):
        fr = {}
        for role, name in (("src", src_name), ("dst", dst_name)):
            persona = R.persona_text(prompts, name, v, personas)
            text = C.frame_text(persona, q)
            q0, ids = C.question_start(tok, text, persona)
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            fr[role] = {"ids": ids, "q0": q0, "off": enc["offset_mapping"], "c": C.question_block_char(persona)}
        src, dst = C.copy_span(fr["src"]["ids"], fr["src"]["q0"], fr["dst"]["ids"], fr["dst"]["q0"])
        s_ids = [fr["src"]["ids"][i] for i in src]
        d_ids = [fr["dst"]["ids"][i] for i in dst]
        assert s_ids == d_ids                                            # identical tokens
        assert src[-1] == len(fr["src"]["ids"]) - 1 and dst[-1] == len(fr["dst"]["ids"]) - 1
        assert tok.decode(d_ids).startswith("User:")                      # first copied token is the User: line's
        assert tok.decode(d_ids) == "User: " + q + "\nAssistant: Answer:"
        for role, pos in (("src", src), ("dst", dst)):
            assert fr[role]["off"][pos[0]][0] == fr[role]["c"]           # starts exactly at the question block
            assert all(fr[role]["off"][i][0] >= fr[role]["c"] for i in pos)   # nothing of the persona line
        n += 1
    assert n == 6 * 8 * 2 * 3 + 6 * 2 + 6 * 2 == 312


def test_d_f2b_reverse_span_uses_each_runs_own_positions():
    """D-F2b: the source and receiving positions are each run's own. With the source LONGER than the receiver (the
    case the old call shape got wrong), source positions index the source and receiving positions the receiver."""
    src_ids = [7, 7, 7, 7, 1, 2, 3]    # longer prefix, question block from index 4
    dst_ids = [9, 1, 2, 3]             # shorter prefix, question block from index 1
    s, d = C.copy_span(src_ids, 4, dst_ids, 1)
    assert s == [4, 5, 6] and d == [1, 2, 3]
    acts = {"down_proj:0": torch.arange(7 * 2, dtype=torch.float32).reshape(1, 7, 2)}
    v = C.values_at(acts, s)
    assert torch.equal(v["down_proj:0"], acts["down_proj:0"][0, 4:7])


def test_d_f3_a_copy_between_different_questions_is_refused(prompts, olmo_tok):
    """D-F3: the old secondary arm copied the You run of "Do you care ..." into "Do you personally care ...". A copy
    now needs identical question blocks, so that pairing raises; the runner's source asks the SAME question."""
    v = "honesty"
    yq = prompts["secondary"]["you_question"].format(value=v)
    pq = prompts["secondary"]["third_person_question"].format(value=v)
    yp, zp = R.persona_text(prompts, "You", v), R.persona_text(prompts, "Zara", v)
    y0, yids = C.question_start(olmo_tok, C.frame_text(yp, yq), yp)
    z0, zids = C.question_start(olmo_tok, C.frame_text(zp, pq), zp)
    with pytest.raises(SystemExit):
        C.copy_span(yids, y0, zids, z0)
    s0, sids = C.question_start(olmo_tok, C.frame_text(yp, pq), yp)
    src, dst = C.copy_span(sids, s0, zids, z0)
    assert [sids[i] for i in src] == [zids[i] for i in dst]


# ---------------------------------------------------------------- off-target frames (H-F3)

def test_h_f3_no_frame_says_none(prompts, tiny_run):
    """No frame the runner builds carries the literal word "None"; a value template refuses value=None."""
    with pytest.raises(SystemExit):
        R.build_prompt(prompts, "You", None, "Do you like the colour blue? A) yes B) no")
    for o in prompts["off_target"]:
        for n in prompts["personas"]:
            text = R.build_prompt(prompts, n, None, o["question"], prompts["off_target_personas"])
            assert "None" not in text
            assert "care" not in text.split("\n")[0]          # no value clause on the persona line
    off = [r for r in records(tiny_run["dir"]) if r["arm"].startswith("off_target")]
    assert off and all("None" not in r.get("persona_line", "") for r in off)
    assert {r["persona_line"] for r in off if "persona_line" in r} == set(prompts["off_target_personas"].values())


# ---------------------------------------------------------------- random families (D-F7, D-F9, D-F10)

def _fake_units(n_layers=4, heads=3, neurons=6, sub=""):
    out = []
    for l in range(n_layers):
        out += [C.Unit("head", l, h, 4, "o_proj", sub) for h in range(heads)]
        out += [C.Unit("neuron", l, j, 1, "down_proj") for j in range(neurons)]
    return out


def _counts(keys, by, f):
    out = {}
    for k in keys:
        out[f(by[k])] = out.get(f(by[k]), 0) + 1
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
    by = {u.key: u for u in units}
    want = _counts(do, by, R.kind_key)
    for name, f in fam["families"].items():
        assert f["fillable"] is True and len(f["sets"]) == 20
        for s in f["sets"]:
            assert len(s) == 8 and len(set(s)) == 8
            assert not (set(s) & set(say)) and not (set(s) & set(do))
            assert _counts(s, by, R.kind_key) == want, name
    # D-F7: the layer family's per-layer counts are checked whether or not anything relaxed: every part is either in
    # its wanted layer or listed as a relaxation from it
    want_layer = _counts(do, by, lambda u: (R.kind_key(u), u.layer))
    lf = fam["families"]["layer"]
    for si, s in enumerate(lf["sets"]):
        got = _counts(s, by, lambda u: (R.kind_key(u), u.layer))
        for row in (r for r in lf["relaxed_detail"] if r["set"] == si):
            got[(row["kind"], row["layer_got"])] -= 1
            got[(row["kind"], row["layer_wanted"])] = got.get((row["kind"], row["layer_wanted"]), 0) + 1
        assert {k: v for k, v in got.items() if v} == want_layer


def test_d_f7_d_f10_a_short_layer_relaxes_to_the_nearest_layer_and_is_counted():
    """A DO set that takes all 3 heads of layer 3, with SAY holding one head of layer 2: the 3 missing heads come
    from the nearest layer first (the 2 eligible heads of layer 2), then the next nearest (1 from layer 1), are
    counted per family, and each is listed with the layer it was wanted in and the one it came from."""
    import random
    units = _fake_units(n_layers=4, heads=3, neurons=6)
    by = {u.key: u for u in units}
    do = ["head:3:0", "head:3:1", "head:3:2"] + ["neuron:0:0", "neuron:0:1"]
    say = ["head:2:0"]
    fam = R.draw_random_families({k: random.Random(5) for k in ("layer", "score", "effect")}, units, do, say,
                                 {u.key: float(u.layer) for u in units}, {u.key: 0.0 for u in units}, k=5, n_sets=4)
    lf = fam["families"]["layer"]
    assert lf["fillable"] and lf["relaxed"] == 3 * 4
    for si, s in enumerate(lf["sets"]):
        rows = [r for r in lf["relaxed_detail"] if r["set"] == si]
        assert all(r["kind"] == "head" and r["layer_wanted"] == 3 for r in rows)
        assert sorted(r["layer_got"] for r in rows) == [1, 2, 2]
        assert all(by[r["part"]].layer == r["layer_got"] and r["part"] != "head:2:0" for r in rows)
        assert sum(by[k].kind == "head" for k in s) == 3


def test_d_f9_fillability_is_per_family_and_not_fillable_is_reported():
    import random
    units = _fake_units(n_layers=2, heads=1, neurons=2)
    keys = [u.key for u in units]
    do = [k for k in keys if k.startswith("head")] + [k for k in keys if k.startswith("neuron")][:2]
    say = [k for k in keys if k.startswith("neuron")][2:4]
    rngs = {k: random.Random(1) for k in ("layer", "score", "effect")}
    fam = R.draw_random_families(rngs, units, do, say, {k: 1.0 for k in keys}, {k: 1.0 for k in keys}, k=4, n_sets=3)
    assert set(fam["families"]) == {"layer", "score", "effect"}
    for f in fam["families"].values():
        assert f["fillable"] is False and f["sets"] == [] and f["reason"]
    # one family without its stratifying values is not fillable on its own; the other two still are
    units2 = _fake_units()
    keys2 = [u.key for u in units2]
    fam2 = R.draw_random_families(rngs, units2, keys2[4:6], keys2[:2], {k: 1.0 for k in keys2}, None, k=2, n_sets=2)
    assert fam2["families"]["effect"]["fillable"] is False
    assert fam2["families"]["layer"]["fillable"] and fam2["families"]["score"]["fillable"]
    rsr = S.random_set_rule({"T": 0.9, "random": {"layer": [{"T": 0.1}] * 2, "score": [{"T": 0.1}] * 2}}, fam2)
    assert rsr["read"] is True and rsr["fillable"] == ["layer", "score"] and "effect" in rsr["not_fillable"]


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


def test_d_f10_scorer_prints_relaxations_per_family(tiny_run):
    sd = S.load_stage_dir(tiny_run["dir"])
    res = S.score_stage("tiny", sd, tiny_run["cfg"], 1)
    fake = {"random_families": {"layer": dict(res["random_families"]["layer"], relaxed=3,
                                              relaxed_detail=[{}] * 3),
                                "score": dict(res["random_families"]["score"], relaxed=0, relaxed_detail=[]),
                                "effect": {"fillable": False, "reason": {"head:full": "want 4, eligible 0"}}}}
    text = S.relaxation_lines(fake)
    assert "random family layer: fillable" in text and "3 relaxed parts" in text
    assert "random family score: fillable" in text and "0 relaxed parts" in text
    assert "random family effect: NOT FILLABLE" in text


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


# ---------------------------------------------------------------- gates (fixtures shaped like the runner's rows)

def _row(arm, qid, name, pa, consistent="A", mass=0.9, top_is_letter=True, logit_diff=None, **extra):
    """A record shaped like the runner's: p_a + p_b = 1 (normalised), raw masses summing to option_mass (D-F8)."""
    r = {"arm": arm, "id": qid, "name": name, "consistent": consistent, "p_a": pa, "p_b": 1.0 - pa,
         "mass_a": pa * mass, "mass_b": (1.0 - pa) * mass, "option_mass": mass, "top_token": 1,
         "top_prob": max(pa, 1 - pa) * mass, "top_is_letter": top_is_letter}
    if arm == "you":
        r["persona"] = "You"
    if logit_diff is not None:
        r["max_abs_logit_diff"] = logit_diff
    r.update(extra)
    return r


def _arms_fixture(you_pa, third_pa, copy_pa, say_pa, top_is_letter=True, mass=0.9, dn_logit_diff=0.0, rev_pa=None):
    """2 questions x 2 names; You answers you_pa, third person third_pa, DO copy copy_pa, SAY copy say_pa."""
    arms = {}
    for q in ("q1", "q2"):
        arms.setdefault("you", []).append(_row("you", q, "You", you_pa, mass=mass, top_is_letter=top_is_letter))
        for name in ("Zara", "Milo"):
            arms.setdefault("baseline", []).append(_row("baseline", q, name, third_pa))
            arms.setdefault("do_nothing", []).append(_row("do_nothing", q, name, third_pa, logit_diff=dn_logit_diff))
            arms.setdefault("do_copy", []).append(_row("do_copy", q, name, copy_pa))
            arms.setdefault("say_copy", []).append(_row("say_copy", q, name, say_pa))
            if rev_pa is not None:
                arms.setdefault("reverse", []).append(_row("reverse", q, name, rev_pa))
    return arms


def test_gate_g0_passes_and_fails():
    held = {"q1", "q2"}
    assert S.gate_g0(_arms_fixture(0.8, 0.5, 0.7, 0.7), held)["pass"] is True
    assert S.gate_g0(_arms_fixture(0.8, 0.5, 0.7, 0.7, mass=0.3), held)["pass"] is False
    assert S.gate_g0(_arms_fixture(0.8, 0.5, 0.7, 0.7, top_is_letter=False), held)["pass"] is False


def test_h_f2_g1_fails_a_do_nothing_row_whose_masses_match_but_logits_do_not():
    """H-F2/D-F4: same option masses, different logits: G1 FAILS. Exact logits: G1 passes. No logit difference
    recorded: G1 fails (it cannot be checked)."""
    ok = _arms_fixture(0.8, 0.5, 0.7, 0.7, dn_logit_diff=0.0)
    assert S.gate_g1(ok)["pass"] is True
    same_masses = _arms_fixture(0.8, 0.5, 0.7, 0.7, dn_logit_diff=0.3)
    for b, d in zip(same_masses["baseline"], same_masses["do_nothing"]):
        assert (b["p_a"], b["p_b"], b["mass_a"], b["mass_b"]) == (d["p_a"], d["p_b"], d["mass_a"], d["mass_b"])
    g = S.gate_g1(same_masses)
    assert g["pass"] is False and g["max_abs_diff"] == 0.3 and g["mode"] == "exact"
    unrecorded = _arms_fixture(0.8, 0.5, 0.7, 0.7)
    for r in unrecorded["do_nothing"]:
        r.pop("max_abs_logit_diff", None)
    assert S.gate_g1(unrecorded)["pass"] is False
    # every do-nothing arm counts, not only the main one
    ok["off_target_do_nothing"] = [_row("off_target_do_nothing", "off-1", "Zara", 0.5, logit_diff=1e-7)]
    assert S.gate_g1(ok)["pass"] is False


def test_h_f6_g1_family_q_applies_the_a770_rule_from_g1check():
    """Family Q: with g1check205's plain runs not bit-identical, every do-nothing difference may be up to twice the
    largest plain difference; with bit-identical plain runs (or family O) the rule stays exact."""
    rows = _arms_fixture(0.8, 0.5, 0.7, 0.7, dn_logit_diff=1.5e-3)
    loose = {"verdict": {"pass": True, "bit_identical_plain": False, "worst_plain": 1e-3}}
    g = S.gate_g1(rows, "Q", loose)
    assert g["pass"] is True and g["mode"] == "a770-relaxed" and g["tolerance"] == 2e-3
    assert S.gate_g1(_arms_fixture(0.8, 0.5, 0.7, 0.7, dn_logit_diff=2.5e-3), "Q", loose)["pass"] is False
    exact = {"verdict": {"pass": True, "bit_identical_plain": True, "worst_plain": 0.0}}
    assert S.gate_g1(rows, "Q", exact)["pass"] is False
    assert S.gate_g1(rows, "O", loose)["pass"] is False                   # family O never relaxes
    assert S.gate_g1(rows, "Q", None)["pass"] is False                    # no g1.json: exact
    failed = {"verdict": {"pass": False, "bit_identical_plain": False, "worst_plain": 1e-3}}
    assert S.gate_g1(_arms_fixture(0.8, 0.5, 0.7, 0.7), "Q", failed)["pass"] is False


def test_h_f6_score_main_reads_g1_json_for_family_q(tiny_run, tmp_path):
    """The wiring: score205.main finds <raw>/g1/g1.json and applies it to a family-Q stage's G1."""
    import shutil
    raw = tmp_path / "raw"
    shutil.copytree(tiny_run["dir"], raw / "tiny")
    meta = json.loads((raw / "tiny" / "meta.json").read_text())
    meta["env"]["family"] = "Q"
    (raw / "tiny" / "meta.json").write_text(json.dumps(meta))
    rp = raw / "tiny" / "records.jsonl"
    rows = records(raw / "tiny")
    for r in rows:
        if r["arm"] in S.DO_NOTHING_ARMS:
            r["max_abs_logit_diff"] = 1.5e-3
    rp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    c = dict(tiny_run["cfg"], stages=[{"stage": "tiny", "family": "Q"}])
    (tmp_path / "cfg.json").write_text(json.dumps(c))
    (raw / "g1").mkdir()
    (raw / "g1" / "g1.json").write_text(json.dumps(
        {"verdict": {"pass": True, "bit_identical_plain": False, "worst_plain": 1e-3}}))
    S.main(["--raw", str(raw), "--config", str(tmp_path / "cfg.json")])
    g1 = json.loads((raw / "results.json").read_text())["stages"][0]["gates"]["G1"]
    assert g1["pass"] is True and g1["mode"] == "a770-relaxed"
    (raw / "g1" / "g1.json").unlink()
    S.main(["--raw", str(raw), "--config", str(tmp_path / "cfg.json")])
    assert json.loads((raw / "results.json").read_text())["stages"][0]["gates"]["G1"]["pass"] is False


def test_gate_g1_a770_variant_uses_all_ten_plain_pairs():
    base = torch.zeros(6)
    plain = [base.clone() for _ in range(5)]
    assert S.gate_g1_a770(plain, [base.clone()] * 5)["pass"] is True
    assert S.gate_g1_a770(plain, [base + 1e-4] * 5)["pass"] is False
    # plain runs not bit-identical; the largest plain difference is between runs 1 and 3 (not involving run 0)
    plain2 = [base.clone(), base + 2e-4, base.clone(), base - 2e-4, base.clone()]
    g = S.gate_g1_a770(plain2, [base + 7e-4] * 5)
    assert g["plain_pairs"] == 10 and g["worst_plain"] == pytest.approx(4e-4)
    assert g["bit_identical_plain"] is False and g["pass"] is True        # 7e-4 <= 2 x 4e-4
    assert S.gate_g1_a770(plain2, [base + 9e-4] * 5)["pass"] is False


def test_gate_g2_passes_and_fails():
    held = {"q1", "q2"}
    assert S.gate_g2(S.gap_stats(_arms_fixture(0.8, 0.5, 0.7, 0.7), held, ["Zara", "Milo"]))["pass"] is True
    assert S.gate_g2(S.gap_stats(_arms_fixture(0.55, 0.5, 0.52, 0.52), held, ["Zara", "Milo"]))["pass"] is False
    # the gap does not depend on a copy arm having run
    no_copy = _arms_fixture(0.8, 0.5, 0.7, 0.7)
    del no_copy["do_copy"]
    assert S.gap_stats(no_copy, held, ["Zara", "Milo"])["gap"] == pytest.approx(0.3)


def test_gate_g3_passes_and_fails():
    held = {"q1", "q2"}
    t = S.ratio_of_means(_arms_fixture(0.8, 0.5, 0.7, 0.66), held, ["Zara", "Milo"], "say_copy")
    assert S.gate_g3(t)["pass"] is True                                   # 0.16 / 0.3
    t2 = S.ratio_of_means(_arms_fixture(0.8, 0.5, 0.7, 0.6), held, ["Zara", "Milo"], "say_copy")
    assert S.gate_g3(t2)["pass"] is False                                 # 0.1 / 0.3


def test_gate_g4_passes_and_fails():
    assert S.gate_g4({"retest_say": 25, "retest_do": 20})["pass"] is True
    assert S.gate_g4({"retest_say": 25, "retest_do": 19})["pass"] is False
    assert S.gate_g4({"retest_say": 19, "retest_do": 40})["pass"] is False


def test_d_f13_ratio_of_means_counts_dropped_pairs():
    arms = _arms_fixture(0.8, 0.5, 0.7, 0.7)
    arms["do_copy"] = [r for r in arms["do_copy"] if not (r["id"] == "q2" and r["name"] == "Milo")]
    t = S.ratio_of_means(arms, ["q1", "q2"], ["Zara", "Milo"], "do_copy")
    assert t["n_pairs"] == 3 and t["n_expected"] == 4 and t["n_dropped"] == 1
    assert t["dropped"] == [{"id": "q2", "name": "Milo", "missing": ["do_copy"]}]


def test_random_set_rule_passes_and_fails():
    held = {"q1", "q2"}
    arms = _arms_fixture(0.8, 0.5, 0.75, 0.7)
    for fam in ("layer", "score"):
        for i in range(3):
            for q, n in (("q1", "Zara"), ("q1", "Milo"), ("q2", "Zara"), ("q2", "Milo")):
                arms.setdefault(f"random:{fam}:{i}", []).append(_row(f"random:{fam}:{i}", q, n, 0.6))
    rnd = {fam: [S.ratio_of_means(arms, held, ["Zara", "Milo"], f"random:{fam}:{i}") for i in range(3)]
           for fam in ("layer", "score")}
    fams = {"families": {"layer": {"fillable": True, "sets": [[]] * 3}, "score": {"fillable": True, "sets": [[]] * 3},
                         "effect": {"fillable": False, "sets": [], "reason": "short"}}}
    t = dict(S.ratio_of_means(arms, held, ["Zara", "Milo"], "do_copy"), random=rnd)     # T 0.83 > 0.33
    assert S.random_set_rule(t, fams)["pass"] is True
    t2 = dict(S.ratio_of_means(_arms_fixture(0.8, 0.5, 0.55, 0.7), held, ["Zara", "Milo"], "do_copy"), random=rnd)
    assert S.random_set_rule(t2, fams)["pass"] is False                                  # T 0.17 < 0.33
    one = {"families": {"layer": {"fillable": True, "sets": [[]]}, "score": {"fillable": False, "sets": []}}}
    assert S.random_set_rule(t, one)["read"] is False


# ---------------------------------------------------------------- the runner and scorer end to end (H-F1)

def test_h_f1_real_run_and_score_find_the_you_run(tiny_run, prompts):
    """H-F1: the REAL run_stage writes rows the REAL score_stage reads: every pair is found, T is a number, G2 has a
    gap, the copy arms and the polarity split are read from the rows, and the read flags are score205.read_flags'."""
    d = tiny_run["dir"]
    for f in ("ranking.npz", "ranking.json", "effect.npz", "copy_acts.npz", "sets.json", "records.jsonl",
              "meta.json"):
        assert (d / f).exists(), f
    recs = records(d)
    arms = {r["arm"] for r in recs}
    assert {"ranking", "you", "baseline", "do_nothing", "do_copy", "say_copy", "swap", "reverse",
            "off_target_baseline", "off_target_do_nothing", "off_target_do_copy", "secondary_you",
            "secondary_you_source", "secondary_baseline", "secondary_do_nothing", "secondary_do_copy",
            "secondary_say_copy"} <= arms
    assert all(r["name"] == "You" for r in recs if r["arm"] == "you")
    res = S.score_stage("tiny", S.load_stage_dir(d), tiny_run["cfg"], 1)
    det = res["T_detail"]
    assert det["T"]["n_pairs"] == 48 and det["T"]["n_dropped"] == 0
    for key in ("T", "T_SAY", "T_swap", "reverse"):
        assert isinstance(res[key], float), key
    assert isinstance(res["gates"]["G2"]["gap"], float) and res["gates"]["G2"]["n_pairs"] == 48
    for arm in ("do_copy", "say_copy", "swap"):
        for letter in ("A", "B"):
            assert res["polarity"][arm][letter]["n"] == 24 and res["polarity"][arm][letter]["T"] is not None
    assert res["polarity_reverse"]["A"]["n"] == 24 and res["polarity_reverse"]["B"]["n"] == 24
    assert res["off_target"]["n"] == 12 and res["T_personally"]["n"] == 12
    # every random family fillable here, each set's T computed from its rows
    for name, f in res["random_families"].items():
        assert f["fillable"] is True and len(f["T"]) == 2 and all(isinstance(x, float) for x in f["T"]), name
    assert res["random_set_rule"]["read"] is True
    assert res["gates"]["G1"]["pass"] is True and res["gates"]["G1"]["n"] == 48 + 12 + 12
    assert res["read"] == S.read_flags(res["gates"], res["random_set_rule"])
    preds = S.predictions([res], tiny_run["cfg"])
    assert set(preds) == {f"205.{i}" for i in range(1, 10)}


def test_d_f5_do_nothing_hooks_the_copy_arms_parts_and_positions(tiny_run):
    """D-F5: the do-nothing rows hook the union of every copy arm's parts at the copy arms' positions, and record
    the full-logit difference, which is exactly 0 on the CPU."""
    recs = records(tiny_run["dir"])
    sets = json.loads((tiny_run["dir"] / "sets.json").read_text())
    union = set(sets["say"]) | set(sets["do"]) | {k for f in sets["random"]["families"].values()
                                                   for s in f["sets"] for k in s}
    by = {(r["arm"], r.get("id"), r.get("name")): r for r in recs}
    n = 0
    for r in recs:
        if r["arm"] != "do_nothing":
            continue
        assert r["n_parts"] == len(union)
        for arm in ("do_copy", "say_copy", "random:layer:0"):
            c = by[(arm, r["id"], r["name"])]
            assert (c["first_position"], c["n_positions"]) == (r["first_position"], r["n_positions"])
        assert r["max_abs_logit_diff"] == 0.0
        n += 1
    assert n == 48
    for r in recs:
        if r["arm"] == "secondary_do_nothing":
            assert r["n_parts"] == len(set(sets["say"]) | set(sets["do"])) and r["max_abs_logit_diff"] == 0.0
        if r["arm"] == "off_target_do_nothing":
            assert r["n_parts"] == len(sets["do"]) and r["max_abs_logit_diff"] == 0.0


def test_d_f2b_d_f3_runner_spans_on_the_real_rows(tiny_run, prompts):
    """The reverse rows write into the You run's own question block; the secondary copies cover the whole question
    block of the "personally" question (the source asked the same question)."""
    tok = tiny_run["tok"]
    recs = records(tiny_run["dir"])
    held = {it["id"]: (v, it) for v in prompts["values"] for it in prompts["items"][v]["say"]}
    for r in recs:
        if r["arm"] == "reverse":
            v, it = held[r["id"]]
            persona = R.persona_text(prompts, "You", v)
            q0, ids = C.question_start(tok, C.frame_text(persona, it["question"]), persona)
            assert r["first_position"] == q0 and r["n_positions"] == len(ids) - q0 and r["n_tokens"] == len(ids)
        if r["arm"] in ("secondary_do_copy", "secondary_say_copy"):
            pq = prompts["secondary"]["third_person_question"].format(value=r["value"])
            persona = R.persona_text(prompts, r["name"], r["value"])
            q0, ids = C.question_start(tok, C.frame_text(persona, pq), persona)
            assert r["first_position"] == q0 and r["n_positions"] == len(ids) - q0
    assert sum(r["arm"] == "secondary_you_source" for r in recs) == 6


def test_qwen_runner_end_to_end(tmp_path, prompts, cfg, qwen_tok):
    """The runner's real path on the hybrid architecture (PROCEDURE section 5, Part 178: a dense model's tests cannot
    see a hybrid-cache crash), scored by the real scorer."""
    c = tiny_cfg(cfg, family="Q", top_k=3, n_sets=1)
    R.run_stage(stage_args(tmp_path), model=tiny_qwen(), tok=qwen_tok, env={"family": "Q", "stage": "tiny"},
                prompts=prompts, cfg=c)
    recs = records(tmp_path / "tiny")
    assert all(r["max_abs_logit_diff"] == 0.0 for r in recs if r["arm"] in S.DO_NOTHING_ARMS)
    sets = json.loads((tmp_path / "tiny" / "sets.json").read_text())
    assert {"head:full", "head:deltanet", "neuron"} <= set(sets["random"]["kind_counts_wanted"])
    res = S.score_stage("tiny", S.load_stage_dir(tmp_path / "tiny"), c, 1)
    assert res["T_detail"]["T"]["n_pairs"] == 48 and isinstance(res["T"], float)


# ---------------------------------------------------------------- resume and lock (H-F7, H-F8, D-F15, D-F16)

def _copy_run(tiny_run, tmp_path):
    import shutil
    shutil.copytree(tiny_run["dir"], tmp_path / "tiny")
    return tmp_path / "tiny"


def _rerun(tiny_run, tmp_path, prompts):
    R.run_stage(stage_args(tmp_path), model=wide_olmo(), tok=tiny_run["tok"], env={"family": "O", "stage": "tiny"},
                prompts=prompts, cfg=tiny_run["cfg"])
    return json.loads((tmp_path / "tiny" / "meta.json").read_text())["passes_this_run"]


def test_h_f8_a_resumed_run_skips_passes_whose_records_exist(tiny_run, tmp_path, prompts):
    d = _copy_run(tiny_run, tmp_path)
    before = (d / "records.jsonl").read_text()
    assert _rerun(tiny_run, tmp_path, prompts) == 0            # nothing missing: no forward pass at all
    assert (d / "records.jsonl").read_text() == before
    # drop the last two copy records of one pair: only the passes that rebuild them run (You, the two baselines the
    # pair and its swap partner need, and the two copies)
    rows = records(d)
    last = [i for i, r in enumerate(rows) if r["arm"] in ("do_copy", "say_copy")][-2:]
    kept = [r for i, r in enumerate(rows) if i not in last]
    (d / "records.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in kept))
    n = _rerun(tiny_run, tmp_path, prompts)
    assert 0 < n <= 1 + 2 + 2
    assert sorted(R._rec_key(r) for r in records(d)) == sorted(R._rec_key(r) for r in rows)


def test_h_f7_d_f15_a_torn_last_line_is_ignored_by_runner_and_scorer(tiny_run, tmp_path, prompts):
    d = _copy_run(tiny_run, tmp_path)
    text = (d / "records.jsonl").read_text()
    lines = text.splitlines(keepends=True)
    torn = "".join(lines[:-1]) + lines[-1][: len(lines[-1]) // 2]
    (d / "records.jsonl").write_text(torn)
    assert len(S.load_stage_dir(d)["records"]) == len(lines) - 1      # the scorer ignores it
    assert R._done_keys(d / "records.jsonl")                          # so does the runner's key reader
    _rerun(tiny_run, tmp_path, prompts)                               # the runner resumes and rewrites the record
    rows = records(d)
    assert len(rows) == len(lines)
    assert [json.loads(x) for x in (d / "records.jsonl").read_text().splitlines()] == rows   # every line parses
    # a bad line that is not the last is corruption, and raises
    (d / "records.jsonl").write_text(lines[0] + "{not json\n" + "".join(lines[1:]))
    with pytest.raises(SystemExit):
        S.load_stage_dir(d)


def test_d_f16_a_second_run_of_one_stage_is_refused(tiny_run, tmp_path, prompts):
    d = _copy_run(tiny_run, tmp_path)
    held = C.StageLock(d / ".lock")
    try:
        with pytest.raises(SystemExit, match="held by another run"):
            _rerun(tiny_run, tmp_path, prompts)
    finally:
        held.release()
    assert _rerun(tiny_run, tmp_path, prompts) == 0                    # released: runs again


# ---------------------------------------------------------------- scoring rules

def test_d_f6_read_flags_stop_the_reads_a_failed_gate_stops():
    """G0 fails: neither O nor T is read. G2 fails: O is read, T is not. G4 fails: O is not read. Calls the real
    score205.read_flags, which score_stage uses (asserted on the real run in test_h_f1)."""
    def gates(**fail):
        return {g: {"pass": not fail.get(g.lower(), False)} for g in ("G0", "G1", "G2", "G3", "G4")}
    rsr = {"read": True}
    f = S.read_flags
    assert f(gates(g0=True), rsr) == {"O": False, "T": False, "copy_arms": False, "random_set_rule": False}
    assert f(gates(g2=True), rsr) == {"O": True, "T": False, "copy_arms": False, "random_set_rule": False}
    assert f(gates(g4=True), rsr) == {"O": False, "T": True, "copy_arms": True, "random_set_rule": True}
    assert f(gates(g1=True), rsr)["O"] is False and f(gates(g1=True), rsr)["T"] is False
    assert f(gates(g3=True), rsr) == {"O": True, "T": False, "copy_arms": False, "random_set_rule": False}
    assert f(gates(), {"read": False})["random_set_rule"] is False


def _stage(name, O=None, read_o=True, read_t=True, **kw):
    s = {"stage": name, "gates": {g: {"pass": True} for g in ("G0", "G1", "G2", "G3", "G4")},
         "read": {"O": read_o, "T": read_t, "copy_arms": read_t, "random_set_rule": False},
         "overlap": {"O": O}, "T": None, "T_swap": None, "random_set_rule": {"read": False},
         "off_target": {"shift": None}, "_est": None, "_prompts": None, "_keys": None,
         "T_detail": {"reverse": {"share": None, "mean_drop": None, "mean_gap": None, "rule": ""}}}
    s["gates"]["G2"]["gap"] = 0.3
    s.update(kw)
    return s


def test_h_f5_205_6_holds_when_the_reverse_copy_removes_the_whole_gap(cfg):
    """Gap 0.40, the reverse copy takes P_you all the way to P_third: 205.6 HOLDS (the old rule needed 1.5 x gap)."""
    arms = _arms_fixture(0.9, 0.5, 0.7, 0.7, rev_pa=0.5)
    held = ["q1", "q2"]
    r = S.reverse_stats(arms, held, ["Zara", "Milo"])
    assert r["share"] == pytest.approx(1.0) and r["mean_gap"] == pytest.approx(0.4)
    st = _stage("instruct", T_detail={"reverse": r})
    assert S.predictions([st], cfg)["205.6"]["verdict"] == "held"
    half = S.reverse_stats(_arms_fixture(0.9, 0.5, 0.7, 0.7, rev_pa=0.7), held, ["Zara", "Milo"])   # drop 0.2
    assert S.predictions([_stage("instruct", T_detail={"reverse": half})], cfg)["205.6"]["verdict"] == "held"
    less = S.reverse_stats(_arms_fixture(0.9, 0.5, 0.7, 0.7, rev_pa=0.75), held, ["Zara", "Milo"])  # drop 0.15
    assert S.predictions([_stage("instruct", T_detail={"reverse": less})], cfg)["205.6"]["verdict"] == "failed"


def test_d_f11_205_3_needs_the_rise_interval_to_exclude_zero(cfg, monkeypatch):
    """The same O's: with the mid-training-end -> SFT rise's paired interval covering 0, 205.3 FAILS; excluding 0,
    it holds."""
    os_ = {"pretrain-3.0T": 0.1, "midtrain-26B": 0.15, "midtrain-end": 0.2, "sft": 0.6, "dpo": 0.62, "instruct": 0.63}
    stages = [_stage(n, O=o) for n, o in os_.items()]
    monkeypatch.setattr(S, "paired_bootstrap_rise", lambda *a, **k: (-0.5, 0.9))
    p = S.predictions(stages, cfg)["205.3"]
    assert p["verdict"] == "failed" and p["rise_ci"] == [-0.5, 0.9]
    monkeypatch.setattr(S, "paired_bootstrap_rise", lambda *a, **k: (0.1, 0.7))
    assert S.predictions(stages, cfg)["205.3"]["verdict"] == "held"


def test_d_f12_withdrawal_skips_a_stage_whose_o_is_none(cfg, monkeypatch):
    monkeypatch.setattr(S, "paired_bootstrap_rise", lambda *a, **k: (0.1, 0.7))
    stages = [_stage("pretrain-3.0T", O=None), _stage("midtrain-26B", O=0.1), _stage("midtrain-end", O=0.7)]
    w = S.withdrawal(stages, cfg)
    assert [q["stage"] for q in w["stages"]] == ["midtrain-end"]


def test_prediction_keys_and_unread_verdicts(cfg):
    """A stage with every gate failing leaves the predictions that need a stage unread, not failed."""
    stage = _stage("instruct", read_o=False, read_t=False)
    stage["gates"] = {g: {"pass": False} for g in ("G0", "G1", "G2", "G3", "G4")}
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
    arms = {"you": [_row("you", "q1", "You", 0.8, consistent="A"), _row("you", "q2", "You", 0.2, consistent="B")],
            "baseline": [_row("baseline", "q1", "Zara", 0.5, consistent="A"),
                         _row("baseline", "q2", "Zara", 0.5, consistent="B")],
            "do_copy": [_row("do_copy", "q1", "Zara", 0.7, consistent="A"),
                        _row("do_copy", "q2", "Zara", 0.3, consistent="B")]}
    pol = S.polarity_split(arms, ["q1", "q2"], ["Zara"], "do_copy")
    assert pol["A"]["T"] == pytest.approx(0.2 / 0.3)
    assert pol["B"]["T"] == pytest.approx(0.2 / 0.3)
    arms["do_copy"][0]["option_mass"] = 0.1
    b = S.broken_counts(arms)
    assert b["do_copy"]["broken"] == 1 and b["do_copy"]["n"] == 2


# ---------------------------------------------------------------- fold of code review r2 (D-F1..F10, H-N1..N6)

def _held_and_names(recs):
    held = sorted({r["id"] for r in recs if r["arm"] == "you"})
    names = sorted({r["name"] for r in recs if r["arm"] == "baseline"})
    return held, names


def _rewrite_records(d, rows):
    (Path(d) / "records.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def test_d_f1_a_torn_or_missing_ranking_json_is_repaired_at_resume(tiny_run, tmp_path, prompts):
    """D-F1: the old runner wrote ranking.json only when the ranking passes ran, and wrote it non-atomically, so a
    torn or missing ranking.json at resume stayed torn and the scorer could never load the stage. Now a resume
    rewrites it from the stored estimates, and the scorer loads the stage."""
    d = _copy_run(tiny_run, tmp_path)
    good = json.loads((d / "ranking.json").read_text())
    for damage in ("torn", "missing"):
        if damage == "torn":
            (d / "ranking.json").write_text(json.dumps(good)[: len(json.dumps(good)) // 2])
        else:
            (d / "ranking.json").unlink()
        assert _rerun(tiny_run, tmp_path, prompts) == 0          # nothing to recompute: the passes are all there
        assert json.loads((d / "ranking.json").read_text()) == good
        assert S.load_stage_dir(d)["prompts"] == good["prompts"]


def test_d_f1_a_torn_ranking_npz_is_recomputed_not_fatal(tiny_run, tmp_path, prompts):
    """D-F1/H-N2: a ranking.npz cut by a kill raises BadZipFile on the old resume; now it is recomputed (the 48
    ranking passes) and the stage finishes, with the same estimates."""
    import numpy as np
    d = _copy_run(tiny_run, tmp_path)
    before = np.load(d / "ranking.npz")["est"]
    raw = (d / "ranking.npz").read_bytes()
    (d / "ranking.npz").write_bytes(raw[: len(raw) // 2])
    n = _rerun(tiny_run, tmp_path, prompts)
    assert n >= 48
    assert np.array_equal(np.load(d / "ranking.npz")["est"], before)
    assert json.loads((d / "meta.json").read_text())["finished"]


def test_d_f2_a_torn_sets_json_is_redrawn_at_resume(tiny_run, tmp_path, prompts):
    """D-F2: the old runner raised JSONDecodeError on a torn sets.json on every later run. Now it is redrawn from
    the seed (the same sets) and rewritten."""
    d = _copy_run(tiny_run, tmp_path)
    good = json.loads((d / "sets.json").read_text())
    (d / "sets.json").write_text(json.dumps(good)[: len(json.dumps(good)) // 2])
    _rerun(tiny_run, tmp_path, prompts)                          # the old code raised JSONDecodeError here
    assert json.loads((d / "sets.json").read_text()) == good     # redrawn from the seed: the same sets
    assert json.loads((d / "meta.json").read_text())["finished"]


def test_d_f3_a_torn_models_json_is_rewritten_not_fatal(tmp_path):
    """D-F3: the old write_models_json raised on a torn models.json, stopping every later stage. Now it starts a
    fresh record with a warning and writes atomically."""
    p = tmp_path / "models.json"
    p.write_text('{"part": 205, "stages": {"sft": {"revision": "r1"')          # torn
    R.write_models_json(p, {"stage": "dpo", "family": "O", "model_id": "m2", "revision": "r2"})
    got = json.loads(p.read_text())["stages"]
    assert got["dpo"]["revision"] == "r2"
    assert not [x for x in tmp_path.iterdir() if x.name.endswith(".tmp")]       # no temp file left behind


def test_d_f1_atomic_writes_leave_no_torn_file_and_no_temp_file(tmp_path):
    """D-F1/F2/F3: every writer goes through temp + os.replace, so a failure mid-write leaves the old file intact
    and no temp file; a successful write is complete and no temp file remains."""
    import numpy as np
    p = tmp_path / "x.json"
    C.atomic_write_json(p, {"a": 1})
    assert json.loads(p.read_text()) == {"a": 1}
    orig = os.replace

    def full(*a):
        raise OSError("disk full")

    os.replace = full
    try:
        with pytest.raises(OSError):
            C.atomic_write_json(p, {"a": 2})
        npz = tmp_path / "y.npz"
        with pytest.raises(OSError):
            C.atomic_savez(npz, a=np.zeros(3))
    finally:
        os.replace = orig
    assert json.loads(p.read_text()) == {"a": 1}                               # the old file survives
    assert not [x for x in tmp_path.iterdir() if x.name.endswith(".tmp")]
    assert not npz.exists() and not [x for x in tmp_path.iterdir() if x.name.endswith(".tmp")]
    C.atomic_savez(npz, a=np.arange(3))
    assert np.array_equal(np.load(npz)["a"], np.arange(3))


def test_h_n3_a_record_without_its_newline_is_torn_and_never_joined(tiny_run, tmp_path, prompts):
    """H-N3: a kill between a record's JSON bytes and its newline left a line that parses but has no newline; the old
    reader kept it and the next append was joined onto it, losing both records. Now the reader treats it as torn,
    and the writer starts on a fresh line."""
    d = _copy_run(tiny_run, tmp_path)
    rows = records(d)
    text = (d / "records.jsonl").read_text()
    assert text.endswith("\n")
    # the kill landed between the last record's JSON bytes and its newline: the last line parses but is unterminated
    (d / "records.jsonl").write_text(text[:-1])
    assert len(C.read_jsonl(d / "records.jsonl")) == len(rows) - 1
    assert len(S.load_stage_dir(d)["records"]) == len(rows) - 1
    _rerun(tiny_run, tmp_path, prompts)
    got = (d / "records.jsonl").read_text().splitlines()
    assert all(json.loads(x) for x in got)                        # every line is exactly one record
    assert sorted(R._rec_key(json.loads(x)) for x in got) == sorted(R._rec_key(r) for r in rows)


def test_d_f4_the_hook_moves_replacement_values_to_the_inputs_device():
    """D-F4/H-N6: the runner stores activations on the CPU; the hook must move the values to the model input's
    DEVICE as well as its dtype. The old `.to(x.dtype)` passes only the dtype, so this fails on it. (No XPU here:
    the call is recorded, not exercised; `test_h_n6...` runs the g1check's real path.)"""
    class Mod(torch.nn.Module):
        def forward(self, x):
            return x

    mod = Mod()
    x = torch.zeros(1, 4, 3)
    values = torch.ones(2, 2)
    calls = []
    real_to = torch.Tensor.to

    def spy(self, *a, **k):
        if self is values:
            calls.append({"device": k.get("device"), "dtype": k.get("dtype")})
        return real_to(self, *a, **k)

    h = C.SiteReplacer(mod, [0, 1], [1, 2], values)
    torch.Tensor.to = spy
    try:
        mod(x)
    finally:
        torch.Tensor.to = real_to
        h.close()
    assert calls == [{"device": x.device, "dtype": x.dtype}]       # the device is passed, not only the dtype


def test_h_n6_g1check_replaces_from_cpu_stored_values(tmp_path, prompts, cfg, qwen_tok):
    """H-N6: g1check205 takes the main run's path — the do-nothing values are moved to the CPU before the hook, as
    the runner stores them — so on an accelerator it exercises the CPU-to-device replacement. Here (CPU only) it
    still passes and records that the values were CPU."""
    import g1check205 as G1
    rc = G1.check(qwen_tok, tiny_qwen(), {"stage": "tiny"}, cfg, prompts, tmp_path, check_pool=False)
    g = json.loads((tmp_path / "g1.json").read_text())
    assert rc == 0 and g["verdict"]["pass"] is True
    assert g["values_device"].startswith("cpu")


def test_h_n1_d_f7_an_incomplete_copy_arm_is_not_read_for_t_and_says_how_many_pairs(tiny_run):
    """H-N1/D-F7: a stage that lost copy rows (a kill in its copy arms) is not read for T or the copy arms, and
    reports the pair count; O, which does not depend on the copy arms, is still read."""
    d = tiny_run["dir"]
    sd = S.load_stage_dir(d)
    held, names = _held_and_names(sd["records"])
    # the tiny model fails G2/G3 by construction, so read_flags' gate half is exercised directly with gates that
    # pass: the pair count is then the only thing that can stop T
    passing = {g: {"pass": True} for g in ("G0", "G1", "G2", "G3", "G4")}
    full = S.score_stage("tiny", sd, tiny_run["cfg"], 1)
    assert full["completeness"]["complete"] is True
    assert S.read_flags(passing, {"read": True}, full["completeness"]["complete"])["T"] is True
    cut = [r for r in sd["records"] if not (r["arm"] == "do_copy" and r["id"] == held[0] and r["name"] == names[0])]
    res = S.score_stage("tiny", dict(sd, records=cut), tiny_run["cfg"], 1)
    c = res["completeness"]
    assert c["complete"] is False and c["incomplete"] == ["do_copy"]
    assert c["per_arm"]["do_copy"] == {"n_pairs": 47, "n_expected": 48, "complete": False}
    assert S.read_flags(passing, {"read": True}, c["complete"]) == {"O": True, "T": False, "copy_arms": False,
                                                                    "random_set_rule": False}
    assert res["T_detail"]["T"]["n_pairs"] == 47                 # the count is still reported
    assert res["read"] == S.read_flags(res["gates"], res["random_set_rule"], c["complete"])


def test_d_f10_h_n5_205_4_is_not_read_with_fewer_than_two_fillable_families(cfg):
    """D-F10/H-N5: Reading rules, lines 244-245 — a stage where fewer than two random families are fillable leaves
    205.4 and 205.7 not read, not failed. The old predictions() returned "failed"."""
    rsr = {"read": False, "fillable": ["layer"], "not_fillable": {"score": "short", "effect": "short"}}
    st = _stage("instruct", T=0.6, random_set_rule=rsr)
    p = S.predictions([st], cfg)
    assert p["205.4"]["verdict"] == "not read"
    assert p["205.4"]["fillable"] == ["layer"]
    assert p["205.7"]["verdict"] == "not read"
    # DO transfer not read at all: 205.7 stays vacuous, as the design's own rule says
    st2 = _stage("instruct", T=0.6, read_t=False, random_set_rule={"read": False})
    assert S.predictions([st2], cfg)["205.7"]["verdict"] == "vacuous"


def test_d_f10_fillability_is_per_family():
    """D-F10 (the code half): fillability is reported per family, and a family is not fillable only when its own
    check fails — a missing stratifier, or a kind total the eligible pool cannot meet. A short (kind, stratum) cell
    is relaxed into the nearest stratum and counted, not a failure. NOTE: this test passes on the d16b218b code too;
    the kind check does not depend on the family's stratifier, so the per-family rewrite is a refactor, not a
    behaviour change (see REPORT.md)."""
    import random
    units = _fake_units(n_layers=2, heads=2, neurons=3)
    keys = [u.key for u in units]
    do = ["head:0:0", "head:1:0", "neuron:0:0", "neuron:0:1"]
    say = ["neuron:1:0", "neuron:1:1"]
    rngs = {k: random.Random(3) for k in ("layer", "score", "effect")}
    fam = R.draw_random_families(rngs, units, do, say, {k: 1.0 for k in keys}, None, k=4, n_sets=2)
    assert fam["families"]["effect"]["fillable"] is False
    assert fam["families"]["layer"]["fillable"] is True and fam["families"]["score"]["fillable"] is True
    assert fam["fillable"] is False and set(fam["not_fillable_reason"]) == {"effect"}
    # and a short stratum is relaxed, not a failure: the DO set takes both heads of layer 1, which the pool cannot
    # meet in layer 1 alone; the layer family is still fillable and lists the relaxed parts
    do_short = ["head:1:0", "head:1:1", "neuron:0:0", "neuron:0:1"]
    fam3 = R.draw_random_families(rngs, units, do_short, ["neuron:1:0"], {k: 1.0 for k in keys},
                                  {k: 1.0 for k in keys}, k=4, n_sets=2)
    lf = fam3["families"]["layer"]
    assert lf["fillable"] is True and lf["relaxed"] > 0 and len(lf["sets"]) == 2
    # a kind total the eligible pool cannot meet marks every family short, each with its own reason string
    do_all = [k for k in keys if k.startswith("neuron")] + ["head:0:0", "head:1:0"]
    fam2 = R.draw_random_families(rngs, units, do_all, ["head:0:1", "head:1:1"], {k: 1.0 for k in keys},
                                  {k: 1.0 for k in keys}, k=6, n_sets=2)
    assert not any(f["fillable"] for f in fam2["families"].values())
    assert all(f["reason"] for f in fam2["families"].values())


def test_d_f5_the_end_to_end_test_asserts_concrete_reads(tiny_run):
    """D-F5: the old `test_h_f1` asserted `read == read_flags(...)`, which score_stage had just computed — a
    tautology — and its tiny fixture fails G0/G2/G3, so it could not show any read. Here the REAL run_stage rows are
    scored, and the fixture is made to pass every gate (G0's letter readout and G2's gap are patched to what a stage
    built to pass has); the reads are then asserted as concrete values, not as a re-derivation."""
    import math
    sd = S.load_stage_dir(tiny_run["dir"])
    rows = []
    for r in sd["records"]:
        r = dict(r)
        if r["arm"] in ("you", "baseline", "do_copy", "say_copy", "swap", "reverse"):
            r["option_mass"], r["top_is_letter"] = 0.9, True
            # the consistent letter differs by question: set P(consistent), not p_a
            pc = 0.8 if r["arm"] in ("you", "do_copy", "say_copy", "reverse") else 0.5   # P_you - P_third = 0.3
            r["p_a"], r["p_b"] = (pc, 1.0 - pc) if r["consistent"] == "A" else (1.0 - pc, pc)
        rows.append(r)
    res = S.score_stage("tiny", dict(sd, records=rows), tiny_run["cfg"], 1)
    assert all(res["gates"][g]["pass"] for g in ("G0", "G1", "G2", "G3")), \
        {g: res["gates"][g] for g in ("G0", "G1", "G2", "G3")}
    # G4 (Retest >= 20 of 80) is a property of the ranking, not of the records, and the tiny pool cannot meet it;
    # O is therefore unread here for G4's reason, and the test says so rather than pretending otherwise
    assert res["gates"]["G4"]["pass"] is False and res["read"]["O"] is False
    assert res["read"]["T"] is True and res["read"]["copy_arms"] is True
    assert isinstance(res["T"], float) and math.isfinite(res["T"])
    assert res["gates"]["G2"]["gap"] == pytest.approx(0.3) and res["gates"]["G2"]["n_pairs"] == 48
    assert res["completeness"]["complete"] is True
    for arm in ("do_copy", "say_copy", "swap", "reverse"):
        assert res["completeness"]["per_arm"][arm]["complete"] is True, arm
    # the read flag is not a copy of read_flags' own call: an incomplete stage with the same gates is not read for T
    assert S.read_flags(res["gates"], res["random_set_rule"], False)["T"] is False


def test_d_f9_the_scorer_reads_no_effect_npz(tiny_run, tmp_path):
    """D-F9: score205's docstring listed effect.npz as an input it never reads. The stage loads and scores with the
    file deleted, and the docstring no longer names it as an input."""
    d = _copy_run(tiny_run, tmp_path)
    (d / "effect.npz").unlink()
    sd = S.load_stage_dir(d)
    res = S.score_stage("tiny", sd, tiny_run["cfg"], 1)
    assert res["T_detail"]["T"]["n_pairs"] == 48                 # scored in full with no effect.npz
    doc = S.__doc__.split("Output:")[0]
    inputs = doc.split("Inputs")[1].split("\n\n")[0]
    listed = inputs.split("):", 1)[1].split("(")[0]              # the file list itself, before the parenthetical
    assert "ranking.npz" in listed and "sets.json" in listed and "records.jsonl" in listed
    assert "effect.npz" not in listed


# ---------------------------------------------------------------- wiring: config, models.json, steps, g1check (H-F9)

def test_h_f9_every_stage_resolves_to_a_model_and_revision(cfg):
    """load_stage's lookup: every stage in config.json has a model id and revision (the old lookup read model_id
    from the top-level stage list, which has none, and the Q stages were named "base"/"chat")."""
    for s in cfg["stages"]:
        spec = R.stage_spec(cfg, s["stage"])
        assert spec["model_id"] and len(spec["revision"]) == 40, s


def test_h_f9_models_json_records_revisions(tmp_path):
    p = tmp_path / "models.json"
    R.write_models_json(p, {"stage": "sft", "family": "O", "model_id": "m", "revision": "r1", "dtype": "float32",
                            "device": "cpu", "transformers": "4.x", "torch": "2.x"})
    R.write_models_json(p, {"stage": "dpo", "family": "O", "model_id": "m2", "revision": "r2"})
    got = json.loads(p.read_text())["stages"]
    assert got["sft"]["revision"] == "r1" and got["sft"]["transformers"] == "4.x" and got["dpo"]["revision"] == "r2"


def test_h_f9_every_launch_line_checks_the_pool():
    lines = [x for x in (HERE / "steps.txt").read_text().splitlines()
             if "launch_arc" in x and ("part205/run205.py --stage" in x)]
    assert len(lines) == 10                       # 9 registered + the fallback's (Amendment 2 item 2, A0 2026-11-13)
    assert all("--check-pool" in x for x in lines)
    assert sum("--arms controls" in x for x in lines) == 1


def test_h_f9_g1check_runs_and_compares_full_logits(tmp_path, prompts, cfg, qwen_tok):
    import g1check205 as G1
    rc = G1.check(qwen_tok, tiny_qwen(), {"stage": "tiny"}, cfg, prompts, tmp_path, check_pool=False)
    g = json.loads((tmp_path / "g1.json").read_text())
    assert rc == 0 and g["verdict"]["plain_pairs"] == 10 and g["verdict"]["bit_identical_plain"] is True
    assert g["verdict"]["compared"].startswith("full last-position logits")


# ---------------------------------------------------------------- prompts (D-F18)

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


def test_d_f18_prompts_approved_and_pilot_cut_exists(prompts):
    import gen_prompts205 as G
    assert "draft" not in prompts and "DRAFT" not in json.dumps(prompts)
    pil = C.load_prompts(HERE / "prompts205-pilot.json")
    assert pil["values"] == ["honesty"] and list(pil["items"]) == ["honesty"]
    assert pil["items"]["honesty"] == prompts["items"]["honesty"]
    for k in ("personas", "off_target_personas", "off_target", "secondary", "frame"):
        assert pil[k] == prompts[k], k
    assert G.main(["--check"]) == 0                    # both files are what the generator writes


def test_prompt_token_bound_with_both_tokenizers(prompts, olmo_tok, qwen_tok):
    for v in prompts["values"]:
        for kind in ("say", "do"):
            for it in prompts["items"][v][kind]:
                for name in prompts["personas"]:
                    text = R.build_prompt(prompts, name, v, it["question"])
                    assert len(olmo_tok(text, add_special_tokens=False).input_ids) < 120
                    assert len(qwen_tok(text, add_special_tokens=False).input_ids) < 120


def test_option_fields_follow_the_question_text(prompts):
    """option_a and option_b are the texts the question displays after "A) " and "B) ", for all 96 items and the
    off-target questions."""
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
                n += 1
    assert n == 96
    for o in prompts["off_target"]:
        m = re.search(r" A\) (.+?) B\) (.+)$", o["question"])
        assert m and o["option_a"] == m.group(1) and o["option_b"] == m.group(2), o["id"]
        assert o["consistent"] is None


def test_consistent_letter_matches_the_written_option(prompts):
    """SAY: the consistent option is "yes". DO: it is the value-consistent action of gen_prompts205.py's DO table.
    (D-F17 family: the old assertions here parsed as `assert option_a if ... else (...)`, true for any non-empty
    text; the parentheses are now explicit.)"""
    import gen_prompts205 as G
    for v in prompts["values"]:
        for it in prompts["items"][v]["say"]:
            assert (it["option_a"] if it["consistent"] == "A" else it["option_b"]) == "yes", it["id"]
        for it in prompts["items"][v]["do"]:
            _situation, good, bad = G.DO[v][it["item"] - 1]
            assert (it["option_a"] if it["consistent"] == "A" else it["option_b"]) == good, it["id"]
            assert (it["option_b"] if it["consistent"] == "A" else it["option_a"]) == bad, it["id"]


def test_score_family_matches_on_the_do_selection_score():
    """Review r2 D-F6, ruled 2026-11-11: family (ii) matches the DO set by DO score, the score DO was selected by."""
    import numpy as np
    import run205
    scores = {"say": np.array([9.0, 1.0]), "do": np.array([1.0, 9.0]), "pooled": np.array([5.0, 5.0])}
    assert list(run205.matching_score(scores)) == [1.0, 9.0]


# ---- Amendment 2, step A0 (2026-11-12): the screen mode and the branch it feeds -----------------------------------

def test_screen_mode_runs_only_the_72_plain_passes(tmp_path, cfg, prompts, olmo_tok):
    """`--stage-only screen` writes the 24 You and 48 third-person rows and nothing else: no ranking, no sets, no
    effect, no copy arm (Amendment 2 item 1: nothing else is computed before the branch)."""
    out = tmp_path / "a0"
    res = R.run_stage(stage_args(out, stage_only="screen"), model=tiny_olmo(), tok=olmo_tok,
                      env={"family": "O", "stage": "tiny"}, prompts=prompts, cfg=tiny_cfg(cfg))
    rows = records(out / "tiny")
    from collections import Counter
    assert Counter(r["arm"] for r in rows) == {"you": 24, "baseline": 48}
    for f in ("ranking.npz", "sets.json", "effect.npz", "copy_acts.npz"):
        assert not (out / "tiny" / f).exists(), f
    assert json.loads((out / "tiny" / "meta.json").read_text())["passes_this_run"] == 72
    # a second run resumes, runs nothing and adds nothing
    R.run_stage(stage_args(out, stage_only="screen"), model=tiny_olmo(), tok=olmo_tok,
                env={"family": "O", "stage": "tiny"}, prompts=prompts, cfg=tiny_cfg(cfg))
    assert len(records(out / "tiny")) == 72
    assert json.loads((out / "tiny" / "meta.json").read_text())["passes_this_run"] == 0
    del res


def _screen_rows(prompts, gap, mass=0.8, top_letter=True, drop=0):
    """Synthetic A0 rows: every You P(consistent) 0.7, every third-person 0.7 - gap."""
    held = R.heldout_say(prompts)
    names = [n for n in prompts["personas"] if n != "You"]
    rows = []
    for h in held:
        pa = 0.7 if h["consistent"] == "A" else 0.3
        rows.append({"arm": "you", "id": h["id"], "name": "You", "value": h["value"], "consistent": h["consistent"],
                     "p_a": pa, "p_b": 1 - pa, "option_mass": mass, "top_is_letter": top_letter, "stage": "instruct"})
        for n in names:
            pc = 0.7 - gap
            pa = pc if h["consistent"] == "A" else 1 - pc
            rows.append({"arm": "baseline", "id": h["id"], "name": n, "value": h["value"],
                         "consistent": h["consistent"], "p_a": pa, "p_b": 1 - pa, "option_mass": mass,
                         "top_is_letter": top_letter, "stage": "instruct"})
    return rows[: len(rows) - drop]


def _write_rows(d, rows, stage="instruct", prompts_sha=None):
    d.mkdir(parents=True, exist_ok=True)
    (d / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    sha = prompts_sha or R._sha(HERE / "prompts205.json")
    (d / "meta.json").write_text(json.dumps({"env": {"stage": stage, "family": "O"}, "prompts_sha": sha}))
    return d


@pytest.mark.parametrize("gap,branch", [(0.25, "PROCEED"), (0.1941, "PROCEED"), (0.1939, "FALLBACK"),
                                        (0.1935, "FALLBACK"), (0.0068, "FALLBACK"), (-0.05, "FALLBACK")])
def test_screen_branch_threshold(tmp_path, prompts, gap, branch):
    import screen205 as SC
    out = SC.screen(_write_rows(tmp_path / "s", _screen_rows(prompts, gap)))
    assert out["branch"] == branch, (gap, out["gap"])
    assert abs(out["gap"] - gap) < 1e-9
    assert out["n_clusters"] == 24 and len(out["gap_per_value"]) == 6


def test_screen_g0_failure_forces_fallback_even_at_a_large_gap(tmp_path, prompts):
    import screen205 as SC
    out = SC.screen(_write_rows(tmp_path / "s", _screen_rows(prompts, 0.4, mass=0.3, top_letter=False)))
    assert out["G0"]["pass"] is False and out["branch"] == "FALLBACK"


def test_screen_reads_g0_on_the_you_rows_only(tmp_path, prompts):
    """r2 M1: a degenerate third-person frame must not move G0 (only the 24 You rows enter it)."""
    import screen205 as SC
    rows = _screen_rows(prompts, 0.3)
    for r in rows:
        if r["arm"] == "baseline":
            r["option_mass"], r["top_is_letter"] = 0.05, False
    out = SC.screen(_write_rows(tmp_path / "s", rows))
    assert out["G0"]["pass"] is True and out["G0"]["n"] == 24 and out["branch"] == "PROCEED"


def test_screen_incomplete_is_hold(tmp_path, prompts):
    import screen205 as SC
    out = SC.screen(_write_rows(tmp_path / "s", _screen_rows(prompts, 0.3, drop=1)))
    assert out["branch"] == "HOLD"
    assert SC.main([str(tmp_path / "s")]) == 3


# DS code review of 48562d81 (2026-11-12): the screen must be exactly the registered 72 rows, from the registered
# stage and prompt file; otherwise HOLD, never a branch and never a crash.

def test_screen_duplicate_plus_missing_pair_is_hold(tmp_path, prompts):
    import screen205 as SC
    rows = _screen_rows(prompts, 0.19)
    third = [i for i, r in enumerate(rows) if r["arm"] == "baseline"]
    rows[third[5]] = dict(rows[third[0]])          # 48 rows still, one pair twice and one missing (review F1)
    out = SC.screen(_write_rows(tmp_path / "s", rows))
    assert out["branch"] == "HOLD" and "48 registered pairs" in out["reason"]


def test_screen_orphan_third_person_row_is_hold_not_a_crash(tmp_path, prompts):
    import screen205 as SC
    rows = _screen_rows(prompts, 0.3)
    orphan = dict(next(r for r in rows if r["arm"] == "baseline"))
    orphan["id"] = "say-honesty-99"
    d = _write_rows(tmp_path / "s", rows[:-1] + [orphan])
    assert SC.screen(d)["branch"] == "HOLD"
    assert SC.main([str(d)]) == 3


@pytest.mark.parametrize("kind", ["stage", "sha", "ids", "row_stage"])
def test_screen_from_the_wrong_stage_or_prompts_is_hold(tmp_path, prompts, kind):
    import screen205 as SC
    rows = _screen_rows(prompts, 0.3)
    stage, sha = "instruct", None
    if kind == "stage":
        stage = "dpo"
    if kind == "sha":
        sha = "0" * 64
    if kind == "ids":
        for r in rows:
            r["id"] = "invented-" + r["id"]
    if kind == "row_stage":
        rows[3]["stage"] = "dpo"
    out = SC.screen(_write_rows(tmp_path / "s", rows, stage=stage, prompts_sha=sha))
    assert out["branch"] == "HOLD", out.get("reason")


def test_screen_rows_equal_the_main_runs_rows(tmp_path, cfg, prompts, olmo_tok):
    """Review F5: the screen's You and third-person rows must be the main run's own, field for field (a swapped persona
    line would leave the counts right and make A0 measure zero)."""
    a = tmp_path / "a0"
    b = tmp_path / "main"
    R.run_stage(stage_args(a, stage_only="screen"), model=tiny_olmo(), tok=olmo_tok,
                env={"family": "O", "stage": "tiny"}, prompts=prompts, cfg=tiny_cfg(cfg))
    R.run_stage(stage_args(b, stage_only="pairs"), model=tiny_olmo(), tok=olmo_tok,
                env={"family": "O", "stage": "tiny"}, prompts=prompts, cfg=tiny_cfg(cfg))
    fields = ("p_a", "p_b", "mass_a", "mass_b", "option_mass", "top_token", "top_prob", "top_is_letter", "n_tokens",
              "id", "name", "persona", "consistent", "value")
    key = lambda r: (r["arm"], r["id"], r["name"])  # noqa: E731
    sa = {key(r): r for r in records(a / "tiny") if r["arm"] in ("you", "baseline")}
    sb = {key(r): r for r in records(b / "tiny") if r["arm"] in ("you", "baseline")}
    assert set(sa) == set(sb) and len(sa) == 72
    diffs = [(k, f) for k in sa for f in fields if sa[k].get(f) != sb[k].get(f)]
    assert diffs == []
    third = [r for r in sa.values() if r["arm"] == "baseline"]
    assert any(abs(r["p_a"] - sa[("you", r["id"], "You")]["p_a"]) > 0 for r in third), "frames must differ"


def test_screen_resume_under_other_prompts_is_refused(tmp_path, cfg, prompts, olmo_tok):
    out = tmp_path / "a0"
    R.run_stage(stage_args(out, stage_only="screen"), model=tiny_olmo(), tok=olmo_tok,
                env={"family": "O", "stage": "tiny"}, prompts=prompts, cfg=tiny_cfg(cfg))
    m = json.loads((out / "tiny" / "meta.json").read_text())
    m["prompts_sha"] = "0" * 64
    (out / "tiny" / "meta.json").write_text(json.dumps(m))
    with pytest.raises(SystemExit):
        R.run_stage(stage_args(out, stage_only="screen"), model=tiny_olmo(), tok=olmo_tok,
                    env={"family": "O", "stage": "tiny"}, prompts=prompts, cfg=tiny_cfg(cfg))


# ---- Amendment 2 item 2, the fallback: `--arms controls` (2026-11-13) ---------------------------------------------

def _controls_args(out, stage_only=None):
    a = stage_args(out, stage_only=stage_only)
    a.arms = "controls"
    return a


def test_controls_mode_runs_exactly_168_passes_and_hooks_the_do_say_union(tmp_path, cfg, prompts, olmo_tok):
    """Amendment 2 item 2: per stage 48 ranking, 24 You, 48 third-person, 48 do-nothing (168); the do-nothing arm hooks
    the union of the DO and SAY sets at the copied positions; no effect, no random sets, no copy, off-target or
    secondary arm. Red if any copy arm runs, the effect is computed, or do-nothing hooks another set."""
    out = tmp_path / "fb"
    c = tiny_cfg(cfg, top_k=16)
    R.run_stage(_controls_args(out), model=wide_olmo(), tok=olmo_tok, env={"family": "O", "stage": "tiny"},
                prompts=prompts, cfg=c)
    d = out / "tiny"
    rows = records(d)
    from collections import Counter
    assert Counter(r["arm"] for r in rows) == {"ranking": 48, "you": 24, "baseline": 48, "do_nothing": 48}
    assert json.loads((d / "meta.json").read_text())["passes_this_run"] == 168
    assert not (d / "effect.npz").exists()
    sets = json.loads((d / "sets.json").read_text())
    assert sets["arms"] == "controls" and sets["random"]["families"] == {}
    union = set(sets["do"]) | set(sets["say"])
    assert all(r["n_parts"] == len(union) for r in rows if r["arm"] == "do_nothing")
    assert len(union) > max(len(sets["do"]), len(sets["say"]))       # the fixture tells the union from either set
    # a resume runs nothing and adds nothing
    R.run_stage(_controls_args(out), model=wide_olmo(), tok=olmo_tok, env={"family": "O", "stage": "tiny"},
                prompts=prompts, cfg=c)
    assert len(records(d)) == 168
    assert json.loads((d / "meta.json").read_text())["passes_this_run"] == 0


def test_controls_and_full_runs_refuse_each_others_stage_folder(tmp_path, cfg, prompts, olmo_tok):
    """A full run resuming a controls folder (or the reverse) would mix arms under one sets.json: refused."""
    out = tmp_path / "fb"
    c = tiny_cfg(cfg, top_k=4)
    R.run_stage(_controls_args(out), model=wide_olmo(), tok=olmo_tok, env={"family": "O", "stage": "tiny"},
                prompts=prompts, cfg=c)
    with pytest.raises(SystemExit, match="controls"):
        R.run_stage(stage_args(out), model=wide_olmo(), tok=olmo_tok, env={"family": "O", "stage": "tiny"},
                    prompts=prompts, cfg=c)


def test_a_controls_stage_scores_O_and_leaves_every_copy_reading_unread(tmp_path, cfg, prompts, olmo_tok):
    """Amendment 2 item 2: at a fallback stage the scorer reads O on G0, G1 and G4 and reads no copy arm; 205.4, 205.6
    and 205.9 are "not read" with the fallback named as the reason. Red if a controls stage crashes the scorer, if T
    or a copy arm is read, or if the reason is missing."""
    out = tmp_path / "fb"
    c = tiny_cfg(cfg, top_k=16)
    R.run_stage(_controls_args(out), model=wide_olmo(), tok=olmo_tok, env={"family": "O", "stage": "tiny"},
                prompts=prompts, cfg=c)
    sd = S.load_stage_dir(out / "tiny")
    res = S.score_stage("instruct", sd, c, 1)
    assert res["read"]["T"] is False and res["read"]["copy_arms"] is False
    assert res["T"] is None and res["T_SAY"] is None and res["T_swap"] is None
    assert res["read"]["O"] == (res["gates"]["G0"]["pass"] and res["gates"]["G1"]["pass"] and res["gates"]["G4"]["pass"])
    assert res["gates"]["G1"]["per_arm"]["do_nothing"]["n"] == 48                # G1 rests on the 48 do-nothing rows
    assert res["arms_mode"] == "controls"
    c2 = dict(c, stages=[{"stage": "instruct", "family": "O"}])
    p = S.predictions([res], c2)
    for k in ("205.4", "205.6", "205.9"):
        assert p[k]["verdict"] == "not read" and "fallback" in p[k].get("reason", ""), (k, p[k])
    # review r1 F0: even with T marked read, the three stay "not read" at a controls stage (no "failed" on None inputs)
    forced = dict(res, read={"O": True, "T": True, "copy_arms": True, "random_set_rule": False})
    p2 = S.predictions([forced], c2)
    for k in ("205.4", "205.6", "205.9"):
        assert p2[k]["verdict"] == "not read" and "fallback" in p2[k].get("reason", ""), (k, p2[k])


def test_a_controls_relaunch_on_a_full_folder_without_sets_is_refused(tmp_path, cfg, prompts, olmo_tok):
    """Review r1 F2: a full run stopped before sets.json (here --stage-only pairs) leaves records and no sets.json;
    a controls relaunch on that folder must be refused before it runs anything."""
    out = tmp_path / "fb"
    c = tiny_cfg(cfg, top_k=16)
    R.run_stage(stage_args(out, stage_only="pairs"), model=wide_olmo(), tok=olmo_tok,
                env={"family": "O", "stage": "tiny"}, prompts=prompts, cfg=c)
    assert not (out / "tiny" / "sets.json").exists()
    n_before = len(records(out / "tiny"))
    with pytest.raises(SystemExit, match="use a fresh --out"):
        R.run_stage(_controls_args(out), model=wide_olmo(), tok=olmo_tok, env={"family": "O", "stage": "tiny"},
                    prompts=prompts, cfg=c)
    assert len(records(out / "tiny")) == n_before


def test_family_q_loader_finds_neuron_report_under_python_isolated_mode(tmp_path):
    """Amendment 3 (2026-11-14): the first A770 launch (the G1 check) died in 4 s with ModuleNotFoundError:
    'neuron_report'. Under `python -I` neither the working directory nor the repository root is on sys.path, and
    run205.py added only its own folder; family Q's loader imports neuron_report, which sits at the root. A copy laid
    out like the Windows job folder (part205/*.py, neuron_report.py at the root) must import it from run205 under -I.
    Break: drop run205.py's root insertion (the import fails)."""
    import shutil
    import subprocess
    here = Path(__file__).resolve().parent
    (tmp_path / "part205").mkdir()
    for f in ("run205.py", "common205.py", "score205.py", "config.json"):
        shutil.copy(here / f, tmp_path / "part205" / f)
    (tmp_path / "neuron_report.py").write_text("LOADED = 'stub at the job-folder root'\n", encoding="utf-8")
    probe = ("import sys; sys.path.insert(0, sys.argv[1]); import run205; import neuron_report; "
             "print(neuron_report.LOADED)")
    r = subprocess.run([sys.executable, "-I", "-B", "-c", probe, str(tmp_path / "part205")],
                       cwd=str(tmp_path / "part205"), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-600:]
    assert "stub at the job-folder root" in r.stdout


@pytest.mark.parametrize("family", ["O", "Q"])
def test_frozen_runner_gives_identical_site_gradients_and_builds_no_weight_gradient(family, olmo_tok, qwen_tok):
    """Amendment 4 (2026-11-14): Runner freezes the weights and marks the embedding output as requiring grad, so a
    ranking pass builds gradients at the sites only (18.3 GB -> 8.7 GB peak on the A770 for Qwen3.5-4B). The site
    activations, gradients and estimates must equal the unfrozen runner's exactly, and no weight may hold a gradient.
    Break: drop the embedding hook (register_hook on a site input then fails: nothing requires grad), or freeze=False
    by default (the weight-gradient assert goes red)."""
    tok = olmo_tok if family == "O" else qwen_tok
    make = tiny_olmo if family == "O" else tiny_qwen
    prompts = C.load_prompts()
    text, it = text_for(prompts, "You", "kindness", 1)
    ids = {"A": C.option_token_ids(tok, "A"), "B": C.option_token_ids(tok, "B")}
    m_ref = make()
    units = C.units_for(m_ref)
    with torch.enable_grad():
        _, read_ref, acts_ref, grads_ref = R.Runner(m_ref, tok, units, ids, freeze=False).run(
            text, need_grad=True, consistent=it["consistent"])
    m = make()
    runner = R.Runner(m, tok, C.units_for(m), ids)
    with torch.enable_grad():
        _, read, acts, grads = runner.run(text, need_grad=True, consistent=it["consistent"])
    assert read == read_ref
    assert set(grads) == set(grads_ref) and grads
    for k in grads_ref:
        assert torch.equal(acts[k], acts_ref[k]), k
        assert torch.equal(grads[k], grads_ref[k]), k
    assert torch.equal(C.estimates_for_prompt(acts, grads, units), C.estimates_for_prompt(acts_ref, grads_ref, units))
    assert all(p.grad is None for p in m.parameters())
    assert any(p.grad is not None for p in m_ref.parameters())
