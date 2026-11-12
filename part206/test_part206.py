"""Tests for Part 206's estimators, gates, recorder and scorer (pytest).

  /home/elisepc/neuron-probe/.venv/bin/python -m pytest part206 -q

Each test names, in a comment, the line that would turn it red. The tiny-model tests build random OLMo 2 and Qwen3.5
text models from configs (no download) and run the real recorder on them.
Written by Claude (Opus 5.5), 2026-11-12.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common206 as C  # noqa: E402

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- the split

def planted_split(mu, sd_type, sd_ctx, n_types=60, per=40, D=64, seed=1):
    """Data with a known split: y = mu + type effect + context noise, two halves, every type in both."""
    rng = np.random.default_rng(seed)
    te = rng.normal(0, sd_type, (n_types, D))
    halves = []
    for h in range(2):
        t = np.repeat(np.arange(n_types), per)
        y = mu + te[t] + rng.normal(0, sd_ctx, (len(t), D))
        halves.append((t, y))
    fit_mean, test_mean = halves[0][1].mean(0), halves[1][1].mean(0)
    sq = np.concatenate([halves[0][1], halves[1][1]]).__pow__(2).mean(0)
    w = np.full(n_types, 1 / n_types)
    tf = np.stack([halves[0][1][halves[0][0] == k].mean(0) for k in range(n_types)])
    tt = np.stack([halves[1][1][halves[1][0] == k].mean(0) for k in range(n_types)])
    tot = mu ** 2 + sd_type ** 2 + sd_ctx ** 2
    return (fit_mean, test_mean, sq, w, tf, tt), (mu ** 2 / tot, sd_type ** 2 / tot, sd_ctx ** 2 / tot)


@pytest.mark.parametrize("mu,sdt,sdc", [(1.0, 0.5, 1.0), (0.0, 0.5, 1.0), (0.5, 0.5, 0.5), (2.0, 0.1, 0.3)])
def test_split_recovers_a_planted_split(mu, sdt, sdc):
    # red if split_shares used fit_mean * fit_mean (noise counted as constant) or dropped the type-mean centring
    args, (c0, t0, r0) = planted_split(mu, sdt, sdc)
    Cc, T, R = C.split_shares(*args)
    s = C.shares_summary(Cc, T, R)
    assert abs(s["constant"] - c0) < 0.03
    assert abs(s["token"] - t0) < 0.03
    assert abs(s["context"] - r0) < 0.04


def test_split_cross_half_product_is_what_keeps_noise_out():
    # the mutation check: a same-half product inflates the constant share on data with no constant
    args, (c0, _, _) = planted_split(0.0, 0.5, 1.0, per=4)
    fit, test, sq, w, tf, tt = args
    Cc, T, R = C.split_shares(*args)
    same = (fit * fit / sq).mean()
    assert abs(Cc.mean() - c0) < abs(same - c0)
    assert same > 0.005                           # the mutated estimator sees a constant that is not there


def test_power_shares_and_frequency_explained_are_bounded():
    args, _ = planted_split(1.0, 0.5, 1.0)
    p = C.power_shares(*args)
    assert abs(p["constant"] + p["token"] + p["context"] - 1) < 1e-9
    w = args[3]
    fe = C.frequency_explained(w, args[4], args[0], np.log(np.arange(1, len(w) + 1)))
    assert 0 <= fe <= 1


# ---------------------------------------------------------------- common mode

def shifted(D=64, P=88, n=60, shift=1.0, noise=1.0, seed=2, pos_vary=0.0):
    rng = np.random.default_rng(seed)
    s = rng.normal(0, 1, D)
    s *= shift / np.linalg.norm(s) * np.sqrt(D)
    prof = 1 + pos_vary * np.linspace(-1, 1, P)[:, None]
    out = []
    for h in range(2):
        d = prof[None] * s[None, None] + rng.normal(0, noise, (n, P, D))
        out.append(d)
    return out, s


def rho_of(halves, scale=None):
    f, t = halves
    return C.rho(f.mean((0, 1)), t.mean((0, 1)), (t ** 2).mean((0, 1)), scale)


def test_rho_reads_a_constant_shift_and_not_noise():
    # red if rho's denominator were the fit half's power or the numerator a same-half product
    h, s = shifted(shift=1.0, noise=1.0)
    r = rho_of(h)
    assert 0.4 < r < 0.6                          # |s|^2 / (|s|^2 + D sigma^2) = 0.5
    h0, _ = shifted(shift=0.0, noise=1.0)
    assert abs(rho_of(h0)) < 0.01


def test_kappa_exceeds_rho_for_a_position_varying_shift():
    h, s = shifted(shift=1.0, noise=0.5, pos_vary=0.9)
    f, t = h
    d2 = (t ** 2).mean((0, 1))
    r = C.rho(f.mean((0, 1)), t.mean((0, 1)), d2)
    k = C.kappa(f.mean(0), t.mean(0), d2)
    assert k > r + 0.05


def test_sd_floor_stops_a_near_constant_coordinate_deciding_rho():
    # draft 7's floor: one coordinate with a tiny sd in Y and a small systematic difference
    rng = np.random.default_rng(3)
    D, P, n = 64, 88, 60
    f = rng.normal(0, 1, (n, P, D))
    t = rng.normal(0, 1, (n, P, D))
    f[..., 0] = 0.01 + rng.normal(0, 1e-4, (n, P))
    t[..., 0] = 0.01 + rng.normal(0, 1e-4, (n, P))
    sd = np.ones(D)
    sd[0] = 1e-4
    unfloored = C.rho(f.mean((0, 1)), t.mean((0, 1)), (t ** 2).mean((0, 1)), sd)
    floored = C.rho(f.mean((0, 1)), t.mean((0, 1)), (t ** 2).mean((0, 1)), C.sd_floor(sd, 0.1))
    assert unfloored > 0.9                        # no real shift, rho near 1 without the floor
    assert abs(floored) < 0.05


# ---------------------------------------------------------------- arming

def test_percentiles_and_activity_matching():
    z = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert C.percentile_in(z, 4.0) == 90.0
    assert C.percentile_in(z, 2.0) == 50.0
    rate = np.array([0.1, 0.11, 0.5, 0.52, 0.12])
    # neuron 4's nearest in rate are 1 and 0 (then 2, 3)
    assert C.activity_matched_percentile(z, rate, 4, 2) == 100.0
    assert C.arming_z(np.array([1.0, 2.0]), np.array([2.0, 0.0])).tolist() == [0.5, 0.0]


# ---------------------------------------------------------------- GT (Tuomi), planted reference

def planted_gates(coupling, D=512, F=400, n=3000, seed=4):
    rng = np.random.default_rng(seed)
    b = rng.normal(0, 1, D)
    b *= 4 / np.linalg.norm(b)
    bh = b / np.linalg.norm(b)
    W = rng.normal(0, 1 / np.sqrt(D), (F, D))
    W -= np.outer(W @ bh, bh)                          # no accidental coupling
    W -= coupling * np.outer(rng.uniform(0.5, 1.5, F), bh)
    x = b + rng.normal(0, 1, (n, D))
    return W, b, x, bh


def causal_from(W, x, bh, R):
    clean = ((x @ W.T) > 0).mean()
    proj = x @ bh
    rem = (((x - proj[:, None] * bh) @ W.T) > 0).mean() - clean
    rand = [(((x - proj[:, None] * r) @ W.T) > 0).mean() - clean for r in R]
    return {"delta_duty": rem, "ci": [rem * 0.9, rem * 1.1] if rem > 0 else [rem * 1.1, rem * 0.9],
            "delta_duty_random": rand}


def test_gt_passes_with_a_planted_negative_coupling_and_fails_without():
    # red if gt_static ignored the sign of the resting term, or gt_verdict did not compare with the random controls
    rng = np.random.default_rng(5)
    for coupling, want in ((1.0, True), (0.0, False)):
        W, b, x, bh = planted_gates(coupling)
        duty = ((x @ W.T) > 0).mean(0)
        R = rng.normal(0, 1, (20, len(b)))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        st = C.gt_static(W, b, duty, R)
        static = [dict(st) for _ in range(4)]
        v = C.gt_verdict(static, causal_from(W, x, bh, R))
        assert v["pass"] is want, (coupling, v, st["median_resting"])


# ---------------------------------------------------------------- tiny models through the real recorder

def tiny_cfg_and_prompts(fam, hidden, ffn, n_layers, vocab):
    cfg = C.load_config()
    cfg = copy.deepcopy(cfg)
    rng = np.random.default_rng(6)
    bos = [1] if fam == "O" else []
    frames = {"P": list(bos), "A-plain": bos + [5, 6, 7], "N": bos + [8, 9, 10], "N2": bos + [11, 12, 13],
              "T-plain": bos + [14, 6, 15], "A-tmpl": bos + [20, 21, 22, 23], "A2-tmpl": bos + [20, 24, 22, 23],
              "R-tmpl": bos + [20, 21, 22, 25], "M": bos + [26, 27, 28, 29], "M2": bos + [30, 31, 32, 33],
              "U-plain": bos + [5], "U-tmpl": bos + [20, 34]}
    passages = []
    for i in range(8):
        ids = [int(x) for x in rng.integers(40, 60, 96)]
        passages.append({"index": i, "half": "fit" if i % 2 == 0 else "test",
                         "source": "wikitext" if i < 4 else "ultrachat", "text": f"t{i}", "ids": {fam: ids}})
    types = list(range(40, 60))
    coords = {f"{L}:{s}": sorted(rng.choice(hidden, min(8, hidden), replace=False).tolist())
              for L in range(n_layers) for s in ("m", "c")}
    neurons = {str(L): sorted(rng.choice(ffn, 6, replace=False).tolist()) for L in range(n_layers)}
    cat = ({"slot": [[1, 0], [2, 3]], "concept": [[1, 2], [2, 5]], "p92b_ds_slot": [[1, 0]], "p92b_ds_concept": [[2, 5]]}
           if fam == "O" else {"identity": [[1, 3], [2, 4]]})
    prompts = {"passages": passages, "catalogued": {fam: cat},
               "families": {fam: {"frames": frames, "listed_types": types, "split_coordinates": coords,
                                  "token_share_neurons": neurons}}}
    cfg["split_coordinates"] = min(8, hidden)
    cfg["split_projection_dims"] = 16
    cfg["bootstrap_resamples"] = 20
    cfg["label_shuffles"] = 50
    cfg["equal_length_groups"] = [["A-plain", "N", "N2", "T-plain"], ["A-tmpl", "M", "M2", "A2-tmpl", "R-tmpl"]]
    return cfg, prompts


class FakeTok:
    def __init__(self, passages, fam):
        self.map = {p["text"]: p["ids"][fam] for p in passages}

    def __call__(self, text, add_special_tokens=False):
        class R:
            pass
        r = R()
        r.input_ids = list(self.map[text])
        return r


def tiny_model(fam):
    import torch
    torch.manual_seed(0)
    if fam == "O":
        from transformers import Olmo2Config, Olmo2ForCausalLM
        c = Olmo2Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
                        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256)
        m = Olmo2ForCausalLM(c)
    else:
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
        c = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
                              num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                              linear_num_value_heads=2, linear_num_key_heads=2, linear_key_head_dim=8,
                              linear_value_head_dim=8, layer_types=["linear_attention", "linear_attention",
                                                                    "linear_attention", "full_attention"],
                              max_position_embeddings=256)
        m = Qwen3_5ForCausalLM(c)
    m.config._attn_implementation = "eager"
    m.eval()
    torch.set_grad_enabled(False)
    return m


@pytest.mark.parametrize("fam", ["O", "Q"])
def test_recorder_end_to_end_on_a_tiny_model(fam, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import record206 as R
    import analyze206 as A
    model = tiny_model(fam)
    n_layers = len(C.text_layers(model)[0])
    cfg, prompts = tiny_cfg_and_prompts(fam, 32, 48, n_layers, 64)
    stage = "instruct" if fam == "O" else "qwen-chat"
    tok = FakeTok(prompts["passages"], fam)
    monkeypatch.setattr(C, "load_model", lambda cfg_, st, dev=None, mp=None: (tok, model, {
        "device": "cpu", "dtype": "float32", "family": fam, "stage": st}))
    monkeypatch.setattr(C, "load_config", lambda path=None: cfg)
    prompts["_sha256"] = "x"
    meta = R.run_moments(cfg, prompts, stage, tmp_path, batch=4)
    g = meta["gates"]
    assert g["G0"]["pass"] and g["G0"]["max_abs_logit_diff"] == 0.0
    assert g["G0.5"]["pass"] and g["G0.5"]["every_hook_once"]
    assert g["G1"]["pass"]
    R.run_gt(cfg, prompts, stage, tmp_path, batch=4)
    S, a, _ = A.analyze(cfg, stage, tmp_path)
    assert len(S["split"]["m"]) == n_layers
    assert S["common_mode"]["A-tmpl|M"]["rho_peak_resamples"]
    # the written means reproduce the reference on the same half (the G0.5 identity, on the sums)
    P = cfg["frames"].index("P")
    for L in range(n_layers):
        r = a["Sh"][P, :, 0, L].sum(0) / a["n_frame"][P, :, 0].sum()
        parts = (a["Semb"][0] + a["Sm"][0, :L].sum(0) + a["Sc"][0, :L + 1].sum(0)) / a["n_P"][0]
        assert np.abs(parts - r).max() <= 1e-4 * np.abs(r).max(), L
    gtc = A.gt_causal(tmp_path / stage, cfg)
    assert gtc is not None and len(gtc["delta_duty_random"]) == cfg["gt_random_directions"]


def test_g05_refuses_a_broken_decomposition():
    # red if g05 compared h with itself, ignored c, or never checked the last layer's m (review r1 #6)
    import torch
    import record206 as R

    class Rec:
        pass
    def world():
        r = Rec()
        emb = torch.randn(1, 3, 4)
        c0, m0, c1, m1 = (torch.randn(1, 3, 4) for _ in range(4))
        h0 = emb + c0
        res1 = h0 + m0
        h1 = res1 + c1
        r.cap = {("emb", 0): emb, ("res", 0): emb.clone(), ("c", 0): c0, ("h", 0): h0, ("m", 0): m0,
                 ("res", 1): res1, ("c", 1): c1, ("h", 1): h1, ("m", 1): m1, ("final", 0): h1 + m1}
        return r
    assert R.g05(world(), 2, "float32", True)["pass"]
    r = world()
    r.cap[("c", 0)] = r.cap[("c", 0)] * 0.5
    assert not R.g05(r, 2, "float32", True)["pass"]
    r = world()
    r.cap[("m", 1)] = torch.zeros(1, 3, 4)                         # the last layer's m
    assert not R.g05(r, 2, "float32", True)["pass"]
    assert not R.g05(world(), 2, "float32", False)["pass"]


def test_206_10_reads_only_rows_named_at_top1(tmp_path):
    # red if 206.10 averaged over every row that named a maker (review r1 #1): 20 top-1 rows where the arm lowers
    # the maker by 0.5, 4 others where it raises it by 2.0
    import score206 as SC
    cfg = write_all(tmp_path)
    arms = ["plain", "arm_k1", "arm_k2", "null_M2M_k1", "null_M2M_k2"] + [f"random{i}_k{k}" for k in (1, 2) for i in range(20)]
    rows = []
    for i in range(24):
        top1 = i < 20
        d = -0.5 if top1 else 2.0
        lp = {a: -1.0 for a in arms}
        lp["arm_k1"] = -1.0 + d
        lp["arm_k2"] = -1.0 + 2 * d
        rows.append({"maker": "Tongyi", "plain_top1_is_maker": top1, "logprob": lp})
    (tmp_path / "qwen-chat" / "causal.json").write_text(json.dumps(
        {"arms": arms, "readout_a": rows, "G3": {"pass": True, "named_at_top1": 20, "of": 24}}))
    V = SC.score(tmp_path, cfg)
    assert V["206.10"] == "held"
    for r in rows[:20]:
        r["logprob"]["arm_k2"] = r["logprob"]["arm_k1"] + 0.1          # k = 2 no longer lowers it further
    (tmp_path / "qwen-chat" / "causal.json").write_text(json.dumps(
        {"arms": arms, "readout_a": rows, "G3": {"pass": True, "named_at_top1": 20, "of": 24}}))
    assert SC.score(tmp_path, cfg)["206.10"] == "failed"


def test_maker_is_a_maker_not_the_model_name():
    import causal206 as K
    names = C.load_config()["maker_names"]
    assert K.find_maker("I am Qwen, created by Alibaba Cloud.", names) == ("Alibaba", 22)


def test_model_path_must_be_the_pinned_snapshot():
    cfg = C.load_config()
    with pytest.raises(SystemExit):
        C.load_model(cfg, "qwen-chat", "cpu", "/x/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a-old")


def test_g1_refuses_changed_ids_and_unequal_prefixes():
    import record206 as R
    cfg, prompts = tiny_cfg_and_prompts("O", 32, 48, 3, 64)
    tok = FakeTok(prompts["passages"], "O")
    assert R.g1_check(cfg, prompts, "O", tok, prompts["passages"])["pass"]
    bad = copy.deepcopy(prompts)
    bad["passages"][0]["ids"]["O"][3] += 1
    assert not R.g1_check(cfg, bad, "O", tok, bad["passages"])["pass"]
    bad = copy.deepcopy(prompts)
    bad["families"]["O"]["frames"]["N"] = bad["families"]["O"]["frames"]["N"][:-1]
    assert not R.g1_check(cfg, bad, "O", tok, bad["passages"])["pass"]


def test_resume_refuses_a_different_plan(tmp_path):
    import record206 as R
    cfg, prompts = tiny_cfg_and_prompts("O", 32, 48, 3, 64)
    S = R.Sums(cfg, "O", prompts["families"]["O"], 3, 32, 48, 20, {0: 6, 1: 6, 2: 6}, [1], [0, 1], True)
    S.done = {0, 1}
    S.a["Sx"][0, 0, 0, 0, 0] = 7.0
    S.save(tmp_path / "p.npz", {"sha256": "aaa"})
    T = R.Sums(cfg, "O", prompts["families"]["O"], 3, 32, 48, 20, {0: 6, 1: 6, 2: 6}, [1], [0, 1], True)
    T.load(tmp_path / "p.npz", {"sha256": "aaa"})
    assert T.done == {0, 1} and T.a["Sx"][0, 0, 0, 0, 0] == 7.0
    with pytest.raises(SystemExit):
        T.load(tmp_path / "p.npz", {"sha256": "bbb"})


# ---------------------------------------------------------------- the scorer, on hand-built summaries

def summary(fam, n_layers, **kw):
    gates = {k: {"pass": True} for k in ("G0", "G0.5", "G1", "G2")}
    mid = C.middle_third(n_layers)
    cm = lambda rp, ci=None, res=None: {"rho_peak": rp, "rho_peak_layer": mid[0],
                                        "rho_peak_ci": list(ci) if ci else [rp - 0.05, rp + 0.1],
                                               "rho_peak_resamples": res if res is not None else [rp] * 20}
    S = {"family": fam, "n_layers": n_layers, "gates": gates,
         "split": {"m": [{"token_share": 0.6 if L == 0 else 0.2, "token_share_freq_removed": kw.get("tf", 0.3)}
                         for L in range(n_layers)],
                   "c": [{"token_share": 0.1} for L in range(n_layers)]},
         "reference": [{"write_share": {"mlp": 0.6, "mixer": 0.3, "embedding": 0.1}} for _ in range(n_layers)],
         "common_mode": {"A-tmpl|M": cm(kw.get("rho", 0.25)), "A-tmpl|R-tmpl": cm(kw.get("rho_r", 0.2)),
                         "M2|M": {"layers": [{"rho": kw.get("rho_null", 0.05)} for _ in range(n_layers)]}},
         "cos_delta_bM": [kw.get("cos_b", 0.6)] * n_layers,
         "cos_request_change": [kw.get("cos", 0.9)] * n_layers,
         "gt": {"verdict": {"pass": kw.get("gt", True)}}}
    if fam == "Q":
        S["arming"] = {"identity": {"median_percentile": kw.get("ident", 50.0)}}
    else:
        S["arming"] = {"slot": {"median_percentile": kw.get("slot", 75.0), "median_percentile_activity_matched": kw.get("slot", 75.0)},
                       "concept": {"median_percentile": kw.get("concept", 55.0), "median_percentile_activity_matched": kw.get("concept", 55.0)},
                       "p92b_ds_slot": {"median_percentile": 70.0}, "p92b_ds_concept": {"median_percentile": 50.0}}
        S["arming_label_shuffle"] = {"p95": kw.get("p95", 8.0)}
        S["arming_gap_resamples"] = kw.get("gap_res", [kw.get("slot", 75.0) - kw.get("concept", 55.0)] * 20)
        S["common_mode"]["A-tmpl|M"]["rho_peak_resamples"] = kw.get("rho_res", [kw.get("rho", 0.3)] * 20)
    return S


def write_all(tmp, overrides=None):
    overrides = overrides or {}
    cfg = C.load_config()
    olmo_rho = {"pretrain-3.0T": 0.05, "midtrain-26B": 0.06, "midtrain-end": 0.07, "sft": 0.30, "dpo": 0.32, "instruct": 0.33}
    olmo_gap = {"pretrain-3.0T": 0.0, "midtrain-26B": 1.0, "midtrain-end": 2.0, "sft": 4.0, "dpo": 20.0, "instruct": 20.0}
    for st in cfg["olmo_stage_order"]:
        kw = dict(rho=olmo_rho[st], slot=55.0 + olmo_gap[st], concept=55.0,
                  rho_res=[olmo_rho[st] + d for d in np.linspace(-0.01, 0.01, 20)],
                  gap_res=[olmo_gap[st] + d for d in np.linspace(-1, 1, 20)])
        kw.update(overrides.get(st, {}))
        d = tmp / st
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(summary("O", 16, **kw)))
    for st in ("qwen-base", "qwen-chat"):
        kw = dict(overrides.get(st, {}))
        d = tmp / st
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(summary("Q", 32, **kw)))
    return cfg


def V_of(tmp, overrides=None):
    import score206 as SC
    cfg = write_all(tmp, overrides)
    return SC.score(tmp, cfg)


def test_scorer_bars_hold_on_a_passing_world(tmp_path):
    V = V_of(tmp_path)
    for k in ("206.1", "206.2", "206.3", "206.4", "206.5", "206.6", "206.7", "206.8", "206.9", "206.11", "206.12"):
        assert V[k] == "held", (k, V[k])
    assert V["206.10"] == "not read"                     # no causal.json
    assert "first appears at stage sft" in V["headline"]["sentence"]


@pytest.mark.parametrize("over,key", [
    ({"qwen-chat": {"gt": False}}, "206.1"),
    ({"sft": {"gt": False}, "dpo": {"gt": False}}, "206.2"),
    ({"qwen-chat": {"rho": 0.15}}, "206.6"),
    ({"qwen-chat": {"cos": 0.7}}, "206.6"),
    ({"qwen-chat": {"ident": 70.0}}, "206.9"),
    ({"qwen-chat": {"rho_r": 0.1}}, "206.11"),
    ({"instruct": {"slot": 62.0, "concept": 55.0}}, "206.8"),
    ({"instruct": {"slot": 58.0, "concept": 45.0}}, "206.8"),
    ({"instruct": {"p95": 30.0}}, "206.8"),
    ({"dpo": {"rho": 0.6, "rho_res": [0.6] * 20}}, "206.7"),
    ({"dpo": {"slot": 60.0, "gap_res": [5.0] * 20}}, "206.12"),
])
def test_scorer_bars_fail_when_their_condition_fails(tmp_path, over, key):
    # red if the named prediction ignored the bar the override breaks
    V = V_of(tmp_path, over)
    assert V[key] == "failed", (key, V[key])


def test_scorer_does_not_read_a_model_that_failed_a_gate(tmp_path):
    import score206 as SC
    cfg = write_all(tmp_path)
    p = tmp_path / "qwen-chat" / "summary.json"
    S = json.loads(p.read_text())
    S["gates"]["G2"]["pass"] = False
    p.write_text(json.dumps(S))
    V = SC.score(tmp_path, cfg)
    assert V["206.6"] == "not read" and V["206.1"] == "not read"


def test_headline_drops_first_appears_when_an_earlier_stage_is_not_below(tmp_path):
    # red if the headline named stage X while an earlier stage's interval reached 0.2
    V = V_of(tmp_path, {"midtrain-end": {"rho": 0.19}})
    s = V["headline"]["sentence"]
    assert s is not None and "first appears" not in s


# ---------------------------------------------------------------- mutation-killing tests (2026-11-12)

def test_token_share_needs_the_cross_half_product():
    # red if T used a same-half product: with no type effect and four positions per type, it would read ~0.12
    args, _ = planted_split(1.0, 0.0, 1.0, per=4)
    Cc, T, R = C.split_shares(*args)
    assert abs(T.mean()) < 0.05


def test_rho_numerator_is_cross_half():
    # red if rho's numerator were Delta_fit . Delta_fit (1/(nP) = 0.0625 with no shift at n = P = 4)
    h, _ = shifted(n=4, P=4, shift=0.0, noise=1.0, seed=11)
    assert abs(rho_of(h)) < 0.03


def test_gt_verdict_applies_the_ratio_and_interval():
    st = {"median_resting": -1.0, "rho_resting_duty": 0.9, "rho_random": [0.1] * 20}
    static = [st] * 4
    base = {"delta_duty": 0.1, "ci": [0.05, 0.15], "delta_duty_random": [0.001] * 19 + [0.02]}
    assert not C.gt_verdict(static, base)["causal_pass"]            # ratio 5
    ok = dict(base, delta_duty_random=[0.001] * 20)
    assert C.gt_verdict(static, ok)["causal_pass"]                   # ratio 100
    assert not C.gt_verdict(static, dict(ok, ci=[-0.01, 0.2]))["causal_pass"]
    two = [st, st, {"median_resting": -1.0, "rho_resting_duty": 0.05, "rho_random": [0.1] * 20},
           {"median_resting": -1.0, "rho_resting_duty": 0.05, "rho_random": [0.1] * 20}]
    assert not C.gt_verdict(two, ok)["static_pass"]                  # beats random at 2 of 4 only


def test_gt_static_reports_the_sign_it_finds():
    # red if the resting term's sign were forced: a planted POSITIVE coupling must read positive and fail
    W, b, x, bh = planted_gates(-1.0)
    duty = ((x @ W.T) > 0).mean(0)
    R = np.random.default_rng(12).normal(0, 1, (20, len(b)))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    st = C.gt_static(W, b, duty, R)
    assert st["median_resting"] > 0 and not C.gt_static_layer_pass(st)


def test_resume_carries_the_gates_and_continues(tmp_path, monkeypatch):
    # red if a resumed run lost G0/G0.5 (its summary would read "not read") or summed a passage twice
    import record206 as R
    model = tiny_model("O")
    cfg, prompts = tiny_cfg_and_prompts("O", 32, 48, 3, 64)
    tok = FakeTok(prompts["passages"], "O")
    monkeypatch.setattr(C, "load_model", lambda cfg_, st, dev=None, mp=None: (tok, model, {
        "device": "cpu", "dtype": "float32", "family": "O", "stage": st}))
    prompts["_sha256"] = "x"
    full = R.run_moments(cfg, prompts, "instruct", tmp_path / "a", batch=2)
    za = np.load(tmp_path / "a" / "instruct" / "moments.npz")
    # interrupted run: stop after the first checkpoint by raising inside reduce_batch on the third batch
    calls = {"n": 0}
    real = R.reduce_batch

    def flaky(*args, **kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real(*args, **kw)
    monkeypatch.setattr(R, "reduce_batch", flaky)
    with pytest.raises(KeyboardInterrupt):
        R.run_moments(cfg, prompts, "instruct", tmp_path / "b", batch=2, checkpoint=1)
    monkeypatch.setattr(R, "reduce_batch", real)
    meta = R.run_moments(cfg, prompts, "instruct", tmp_path / "b", batch=2, checkpoint=1)
    assert meta["gates"]["G0"]["pass"] and meta["gates"]["G0.5"]["pass"]
    zb = np.load(tmp_path / "b" / "instruct" / "moments.npz")
    assert np.allclose(za["Sx"], zb["Sx"]) and np.allclose(za["SD2"], zb["SD2"])
    assert sorted(zb["done"].tolist()) == list(range(8))


def test_muse_rows_hold_and_fail_on_their_bars(tmp_path):
    V = V_of(tmp_path / "pass")
    for k in ("206.M1", "206.M3", "206.M4", "206.M5"):
        assert V[k] == "held", (k, V[k])
    assert V["206.M2"] == "failed"                                    # the passing world's gap is 20 at slot 75
    V2 = V_of(tmp_path / "m2", {"instruct": {"slot": 65.0, "concept": 50.0}})
    assert V2["206.M2"] == "held"
    for over, key in (({"qwen-chat": {"rho": 0.35}}, "206.M1"), ({"qwen-chat": {"rho_null": 0.1}}, "206.M1"),
                      ({"qwen-chat": {"ident": 60.0}}, "206.M3"), ({"qwen-chat": {"tf": 0.45}}, "206.M4"),
                      ({"qwen-chat": {"cos_b": 0.4}}, "206.M5")):
        assert V_of(tmp_path / key / str(len(str(over))), over)[key] == "failed", (key, over)


def test_no_float64_on_the_device():
    """The A770 has no float64 kernels (notes/gotchas.md): `.double()` on an XPU tensor dies with "Required aspect
    fp64 is not supported on the device" (Part 206's first Qwen smoke, 2026-11-12 14:41). Every float64 conversion
    must happen after the copy to the CPU, and no float64 tensor may be allocated on a model device."""
    import re
    from pathlib import Path
    here = Path(__file__).parent
    for name in ("record206.py", "causal206.py", "common206.py"):
        src = (here / name).read_text(encoding="utf-8")
        assert not re.search(r"\.double\(\)\s*\.cpu\(\)", src), f"{name}: .double() before .cpu()"
        for m in re.finditer(r"dtype=torch\.float64[^)\n]*device=([^,)\n]+)", src):
            assert m.group(1).strip() in ("'cpu'", '"cpu"'), f"{name}: float64 allocated on {m.group(1)}"
