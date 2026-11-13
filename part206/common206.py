#!/usr/bin/env python3
"""common206.py: config, model loading, hook sites and every estimator PART206-DESIGN.md (draft 7) names.

The estimators are pure numpy so test_part206.py can check them on planted data. record206.py only accumulates sums;
everything that turns sums into a number lives here, so the scorer and the tests read the same code.

Sites (Design, Method, "Sites"), per layer L:
  h  the residual stream where the MLP reads it      OLMo 2: input to layers[L].mlp        Qwen3.5: input to post_attention_layernorm
  x  the MLP's actual input                          OLMo 2: same as h (post-norm model)   Qwen3.5: output of post_attention_layernorm
  g  gate pre-activation (gate_proj output)          both
  a  neuron activation (down_proj input)             both
  m  what the MLP block adds to the stream           OLMo 2: post_feedforward_layernorm output   Qwen3.5: mlp output
  c  what the mixer block adds to the stream         OLMo 2: post_attention_layernorm output     Qwen3.5: self_attn / linear_attn output
The architecture is told apart by the block's modules (a block with post_feedforward_layernorm is OLMo-2 style), so a
tiny test model and the real one take the same path.

Written by Claude (Opus 5.5), 2026-11-12, as the Part 206 build.
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


# ---------------------------------------------------------------- io

AMENDMENT4_CATALOGUE = HERE / "catalogue-amendment4.json"
FAMILY_SHAPES = {"O": {"layers": 16, "ffn": 8192}, "Q": {"layers": 32, "ffn": 9216}}


def check_catalogue(cat, n_layers, width, fam):
    """Refuse a catalogue with any (layer, neuron) outside the model: the registered OLMo sets were another model's."""
    bad = [(name, int(L), int(n)) for name, lst in cat.items() for L, n in lst
           if not (0 <= int(L) < n_layers and 0 <= int(n) < width)]
    if bad:
        raise ValueError(f"{len(bad)} catalogued {fam} neurons lie outside the model's {n_layers} layers x {width} "
                         f"neurons (first: {bad[:3]})")
    return cat


def catalogue(prompts, fam, n_layers=None, width=None, path=None):
    """The catalogued neuron sets Part 206 scores. Amendment 4: prompts206.json's OLMo sets are Part 112's Qwen3.5-4B
    coordinates, so OLMo's come from catalogue-amendment4.json (Part 92's OLMo kinds); Qwen's stay as registered.
    Every coordinate is checked against the model's shape (the data's, when the caller passes it)."""
    if fam == "O":
        cat = json.loads(Path(path or AMENDMENT4_CATALOGUE).read_text())["O"]
    else:
        cat = prompts["catalogued"][fam]
    shp = FAMILY_SHAPES[fam]
    return check_catalogue(cat, int(n_layers or shp["layers"]), int(width or shp["ffn"]), fam)


def load_config(path=None):
    return json.loads(Path(path or HERE / "config206.json").read_text(encoding="utf-8"))


def model_spec(cfg, stage):
    for m in cfg["models"]:
        if m["stage"] == stage:
            return m
    raise SystemExit(f"unknown stage {stage!r}; known: {[m['stage'] for m in cfg['models']]}")


def atomic_write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def atomic_write_json(path, obj):
    return atomic_write_bytes(path, (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode("utf-8"))


def atomic_savez(path, **arrays):
    import io
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return atomic_write_bytes(path, buf.getvalue())


# ---------------------------------------------------------------- layer partitions

def first_third(n):
    """Layer indices below n/3 (Design, Reading rules: the thirds are by layer index)."""
    return [l for l in range(n) if l < n / 3]


def middle_third(n):
    return [l for l in range(n) if n / 3 <= l < 2 * n / 3]


def middle_layer(n):
    return n // 2


def nearest_layers(n, fractions):
    """The layer nearest each depth fraction (fraction x (n - 1)), as Tuomi's depth fractions read."""
    return [int(round(f * (n - 1))) for f in fractions]


# ---------------------------------------------------------------- model loading and sites

def load_model(cfg, stage, device_override=None, model_path=None):
    """(tokenizer, model, env) for one stage, eager attention. Qwen goes through neuron_report.load_model, the loader
    the kernel audit fixed (another loader can still run the faulty SDPA kernel on the A770)."""
    import sys
    import torch
    spec = model_spec(cfg, stage)
    fam = cfg["families"][spec["family"]]
    device = device_override or fam["device"]
    dtype = torch.float32 if fam["dtype"] == "float32" else torch.bfloat16
    if model_path and Path(model_path).name != spec["revision"]:
        raise SystemExit(f"--model-path {model_path} is not the snapshot folder of the pinned revision {spec['revision']}")
    src, rev = (model_path, None) if model_path else (spec["model_id"], spec["revision"])
    if spec["family"] == "Q":
        sys.path.insert(0, str(ROOT))
        import neuron_report
        tok, model = neuron_report.load_model(src, rev, device,
                                              attn_implementation=fam["attn_implementation"], dtype=dtype)
        env = dict(neuron_report.LAST_LOAD or {})
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(src, revision=rev)
        model = AutoModelForCausalLM.from_pretrained(src, revision=rev, dtype=dtype,
                                                     attn_implementation=fam["attn_implementation"])
        model.to(device).eval()
        torch.set_grad_enabled(False)
        import transformers
        env = {"torch": torch.__version__, "transformers": transformers.__version__}
    env.update({"stage": stage, "family": spec["family"], "model_id": spec["model_id"], "revision": spec["revision"],
                "model_path": model_path,
                "dtype": fam["dtype"], "device": device, "attn_implementation": fam["attn_implementation"]})
    return tok, model, env


def text_layers(model):
    """The decoder's layer list and its embedding module (the longest ModuleList of blocks with mlp.down_proj,
    preferring a language model's, as neuron_report.find_text_layers)."""
    import torch
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) and hasattr(getattr(mod[0], "mlp", None), "down_proj"):
            key = ("language" in name, len(mod))
            if best is None or key > best[0]:
                best = (key, name, mod)
    if best is None:
        raise SystemExit("no ModuleList of blocks with mlp.down_proj found")
    _, name, layers = best
    parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
    emb = getattr(parent, "embed_tokens", None)
    if emb is None:
        raise SystemExit(f"no embed_tokens beside {name}")
    return layers, emb


def block_style(block):
    """'post' for OLMo 2 (normalizes sublayer outputs), 'pre' for a pre-norm block (Qwen3.5)."""
    if hasattr(block, "post_feedforward_layernorm") and hasattr(block, "post_attention_layernorm"):
        return "post"
    if hasattr(block, "post_attention_layernorm"):
        return "pre"
    raise SystemExit(f"block {type(block).__name__} has no post_attention_layernorm")


def mixer_module(block):
    lin = getattr(block, "linear_attn", None)
    return lin if lin is not None else block.self_attn


def site_modules(block):
    """{site: (module, kind)} where kind is 'in' (forward pre-hook, args[0]) or 'out' (forward hook, output)."""
    style = block_style(block)
    if style == "post":
        return {"h": (block.mlp, "in"), "x": (block.mlp, "in"), "c": (block.post_attention_layernorm, "out"),
                "m": (block.post_feedforward_layernorm, "out"), "g": (block.mlp.gate_proj, "out"),
                "a": (block.mlp.down_proj, "in")}
    return {"h": (block.post_attention_layernorm, "in"), "x": (block.post_attention_layernorm, "out"),
            "c": (mixer_module(block), "out"), "m": (block.mlp, "out"), "g": (block.mlp.gate_proj, "out"),
            "a": (block.mlp.down_proj, "in")}


def first_tensor(x):
    """A module output may be a tuple (attention returns (out, weights)); the stream contribution is the first."""
    return x[0] if isinstance(x, (tuple, list)) else x


# ---------------------------------------------------------------- (1) the split

def split_shares(fit_mean, test_mean, sq_mean, type_w, type_fit, type_test):
    """C, T, R per coordinate (Design, Method (1)).

    fit_mean, test_mean: [D] each half's mean of y over the positions used.
    sq_mean: [D] the mean of y squared over the positions used (both halves).
    type_w: [K] each listed type's share of the positions used (sums to 1).
    type_fit, type_test: [K, D] each half's mean of y over that type's positions.
    C_d = fit_mean_d x test_mean_d / S_d; T_d = sum_t w_t (type_fit - fit_mean)(type_test - test_mean) / S_d; R = 1 - C - T.
    """
    S = np.asarray(sq_mean, float)
    C = np.asarray(fit_mean, float) * np.asarray(test_mean, float) / S
    dev_f = np.asarray(type_fit, float) - np.asarray(fit_mean, float)[None, :]
    dev_t = np.asarray(type_test, float) - np.asarray(test_mean, float)[None, :]
    T = (np.asarray(type_w, float)[:, None] * dev_f * dev_t).sum(0) / S
    R = 1.0 - C - T
    return C, T, R


def shares_summary(C, T, R):
    """The sublayer's shares: means over coordinates, and the token share T / (T + R) of what varies."""
    c, t, r = float(np.mean(C)), float(np.mean(T)), float(np.mean(R))
    return {"constant": c, "token": t, "context": r, "token_share": (t / (t + r)) if (t + r) != 0 else None}


def power_shares(fit_mean, test_mean, sq_mean, type_w, type_fit, type_test):
    """The same split in raw power (sums of numerators over sums of S over dimensions), reported, not scored."""
    S = np.asarray(sq_mean, float)
    c = float((np.asarray(fit_mean) * np.asarray(test_mean)).sum() / S.sum())
    dev_f = np.asarray(type_fit, float) - np.asarray(fit_mean, float)[None, :]
    dev_t = np.asarray(type_test, float) - np.asarray(test_mean, float)[None, :]
    t = float((np.asarray(type_w)[:, None] * dev_f * dev_t).sum() / S.sum())
    r = 1.0 - c - t
    return {"constant": c, "token": t, "context": r, "token_share": (t / (t + r)) if (t + r) != 0 else None}


def frequency_explained(type_w, type_mean, overall_mean, log_freq):
    """Share of the token part (weighted type deviations) a straight line in log type frequency explains, pooled
    over coordinates. Reported beside each sublayer's token share, not scored."""
    w = np.asarray(type_w, float)
    dev = np.asarray(type_mean, float) - np.asarray(overall_mean, float)[None, :]
    X = np.stack([np.ones_like(log_freq), np.asarray(log_freq, float)], 1)
    W = np.diag(w)
    beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ dev)
    fitted = X @ beta
    tot = float((w[:, None] * dev ** 2).sum())
    return float((w[:, None] * fitted ** 2).sum() / tot) if tot > 0 else None


# ---------------------------------------------------------------- (2) the reference

def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else None


def write_shares(group_means_a, r_b):
    """Each group's projection on the unit vector of r_L estimated on the OTHER half, over the norm of that r_L.
    group_means_a: {group: [D] mean write on half a}; r_b: [D] r_L on half b."""
    u, n = unit(r_b), float(np.linalg.norm(r_b))
    return {g: float(np.asarray(v, float) @ u / n) for g, v in group_means_a.items()}


# ---------------------------------------------------------------- (3) common mode

def sd_floor(sd, fraction):
    """The scored rho's divisor: each coordinate's sd floored at `fraction` of the median sd over the layer's
    coordinates (draft 7: neither a few very large coordinates nor a near-constant one can decide rho)."""
    sd = np.asarray(sd, float)
    return np.maximum(sd, fraction * float(np.median(sd)))


def rho(delta_fit, delta_test, d2_test_mean, scale=None):
    """rho = Delta_fit . Delta_test / mean |D|^2 on the test half. With scale, every coordinate is divided by it
    (d2_test_mean is then the per-coordinate mean of D^2, summed after scaling).
    delta_fit, delta_test: [D]; d2_test_mean: [D] per-coordinate mean of D^2 over the test half's positions."""
    df, dt, d2 = (np.asarray(x, float) for x in (delta_fit, delta_test, d2_test_mean))
    if scale is not None:
        s = np.asarray(scale, float)
        df, dt, d2 = df / s, dt / s, d2 / s ** 2
    den = float(d2.sum())
    return float(df @ dt / den) if den > 0 else None


def kappa(pos_fit, pos_test, d2_test_mean, scale=None):
    """The same ratio taken position by position: mean over positions of (fit mean at p) . (test mean at p), over
    mean |D|^2 on the test half. pos_fit, pos_test: [P, D]."""
    pf, pt, d2 = (np.asarray(x, float) for x in (pos_fit, pos_test, d2_test_mean))
    if scale is not None:
        s = np.asarray(scale, float)
        pf, pt, d2 = pf / s, pt / s, d2 / s ** 2
    den = float(d2.sum())
    return float((pf * pt).sum(1).mean() / den) if den > 0 else None


# ---------------------------------------------------------------- (4) arming

def arming_z(diff_mean, y_sd):
    """z_n = mean paired difference of g_n (X minus Y) over the sd of g_n in Y."""
    sd = np.asarray(y_sd, float)
    return np.where(sd > 0, np.asarray(diff_mean, float) / np.where(sd > 0, sd, 1.0), 0.0)


def percentile_in(values, v):
    """Percent of `values` strictly below v plus half the ties (a mid-rank percentile, 50 is typical)."""
    values = np.asarray(values, float)
    below = float((values < v).sum())
    ties = float((values == v).sum())
    return 100.0 * (below + 0.5 * ties) / len(values)


def activity_matched_percentile(z_layer, rate_layer, idx, k):
    """The neuron's z against the k same-layer neurons nearest it in firing rate in frame M (itself excluded)."""
    rate = np.asarray(rate_layer, float)
    order = np.argsort(np.abs(rate - rate[idx]), kind="stable")
    nb = [j for j in order if j != idx][:k]
    return percentile_in(np.asarray(z_layer)[nb], z_layer[idx])


# ---------------------------------------------------------------- GT (Tuomi)

def _average_ranks(x):
    """Ranks 1..n with ties given their mean rank (scipy.stats.rankdata's 'average' method)."""
    x = np.asarray(x, float)
    sorter = np.argsort(x, kind="mergesort")
    inv = np.empty(len(x), dtype=np.intp)
    inv[sorter] = np.arange(len(x))
    xs = x[sorter]
    obs = np.r_[True, xs[1:] != xs[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def spearman(a, b):
    """Spearman's rho as Pearson's r on average ranks; None where scipy.stats.spearmanr gives nan (a NaN in either
    input, fewer than two values, or a constant input). Plain numpy since Amendment 3: the Windows GPU venv has no
    scipy, and step 2 died on the import (2026-11-12 15:21)."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.shape != b.shape or len(a) < 2 or np.isnan(a).any() or np.isnan(b).any():
        return None
    ra = _average_ranks(a) - (len(a) + 1) / 2
    rb = _average_ranks(b) - (len(b) + 1) / 2
    den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if den == 0.0:
        return None
    return float((ra * rb).sum() / den)


def gt_static(W_gate, b_L, duty, random_dirs):
    """Tuomi section 5: resting term w_n . b_L (no gate bias in either family); its rank correlation with duty,
    beside the same with each random direction of norm |b_L|. random_dirs: [R, D] unit rows."""
    W = np.asarray(W_gate, float)
    rest = W @ np.asarray(b_L, float)
    nb = float(np.linalg.norm(b_L))
    rand = [spearman(W @ (nb * np.asarray(r, float)), duty) for r in random_dirs]
    return {"median_resting": float(np.median(rest)), "share_negative": float((rest < 0).mean()),
            "rho_resting_duty": spearman(rest, duty), "rho_random": rand}


def gt_static_layer_pass(st):
    """A layer passes the static test when its median resting term is negative and its correlation exceeds every
    random-direction correlation."""
    if st.get("rho_resting_duty") is None or any(r is None for r in st["rho_random"]):
        return False
    return st["median_resting"] < 0 and st["rho_resting_duty"] > max(st["rho_random"])


def gt_verdict(static_layers, causal, ratio=20.0):
    """GT (Design, Gates): static at all four layers negative and the correlation above all 20 random at three or
    more; causal Delta duty positive, its interval excluding 0, and at least `ratio` times the largest |Delta duty|
    of the random controls."""
    neg_all = all(s["median_resting"] < 0 for s in static_layers)
    beats = sum(gt_static_layer_pass(s) for s in static_layers)
    static_ok = neg_all and beats >= 3
    d, lo = causal["delta_duty"], causal["ci"][0]
    rmax = max(abs(x) for x in causal["delta_duty_random"])
    causal_ok = d > 0 and lo > 0 and d >= ratio * rmax
    return {"static_pass": bool(static_ok), "static_layers_beating_random": int(beats), "all_negative": bool(neg_all),
            "causal_pass": bool(causal_ok), "ratio_to_largest_random": (d / rmax) if rmax > 0 else None,
            "pass": bool(static_ok and causal_ok)}


# ---------------------------------------------------------------- resampling

def bootstrap_ci(values, alpha=0.05):
    v = np.sort(np.asarray([x for x in values if x is not None], float))
    if len(v) == 0:
        return [None, None]
    return [float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))]
