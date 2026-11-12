#!/usr/bin/env python3
"""record206.py: the moments pass and GT's causal pass for one model (PART206-DESIGN.md draft 7, Method (1) to (4),
GT, gates G0, G0.5, G1, G2). Forward passes only; nothing is generated.

  python -I -B part206/record206.py --stage instruct --out results/part206 [--device cpu] [--batch 8] [--limit N]
  python -I -B part206/record206.py --stage instruct --out results/part206 --pass gt      (after the moments pass)

The moments pass runs every passage through every frame and keeps running sums only (activations are reduced inside
each batch and never stored): per frame, the sums of x and h by source and half and of x squared by half; per pair,
the paired difference D = x(X) - x(Y) summed per read position and squared per coordinate, by half, plus per-passage
means for the scored pairs in the middle third; arming sums on g for the scored pairs; in frame P, the split's sums
(256 seeded coordinates of m and c, a 512-dimension Gaussian projection, listed types only), the token-share sums of
`a`, the embedding and every c and m for the reference write shares, and GT's static sums (positions 16 to 95,
wikitext). The GT pass removes the residual's projection on the fit half's mean direction at the layer nearest depth
0.31 (and 20 seeded random controls) on the wikitext test half, and counts g > 0 at that layer.

Gates. G0: with every hook installed, the logits of the first batch equal the plain model's bit for bit (same batch
shape). G0.5: every hook fired once per layer per forward, and embedding + every c and m below a layer's MLP
reproduces h to within rounding (relative error bound: 2 x layers x 2^-8 in bf16, 2 x layers x 2^-20 in float32).
G1: every frame's input ids end in the passage ids, the stored passage ids equal this stage's tokenization, and
prefix lengths are equal within each equal-length group. G2 (after the pass): the halves' r_L and b_L agree at
cosine 0.94 or more at every layer (draft 8: Tuomi's lowest split-half cosine). A failed G0, G0.5 or G1 stops the run before any sum is written.

Resume: sums and the set of finished passages are checkpointed atomically every --checkpoint batches; a relaunch with
the same plan hash continues from them, and a different plan hash refuses.
Written by Claude (Opus 5.5), 2026-11-12, as the Part 206 build.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common206 as C  # noqa: E402


class StopForward(Exception):
    pass


# ---------------------------------------------------------------- plan

def plan_for(cfg, prompts, stage, limit=None):
    spec = C.model_spec(cfg, stage)
    fam = spec["family"]
    F = prompts["families"][fam]
    passages = prompts["passages"]
    if limit:
        # an even number of passages from each source, so both halves and both sources are present
        per = max(2, limit // 2 - (limit // 2) % 2)
        wiki = [p for p in passages if p["source"] == "wikitext"][:per]
        chat = [p for p in passages if p["source"] == "ultrachat"][:per]
        passages = wiki + chat
    plan = {"stage": stage, "family": fam, "frames": cfg["frames"], "pairs": cfg["pairs"],
            "passages": [p["index"] for p in passages], "prompts_sha256": prompts.get("_sha256")}
    plan["sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return spec, fam, F, passages, plan


def g1_check(cfg, prompts, fam, tok, passages):
    """G1: stored passage ids equal this stage's tokenization of the passage text, and prefix lengths are equal
    within every equal-length group."""
    F = prompts["families"][fam]
    bad = []
    for p in passages:
        ids = tok(p["text"], add_special_tokens=False).input_ids[:cfg["passage_tokens"]]
        if ids != p["ids"][fam]:
            bad.append(p["index"])
    lens = []
    for grp in cfg["equal_length_groups"]:
        l = {name: len(F["frames"][name]) for name in grp}
        if len(set(l.values())) != 1:
            lens.append(l)
    return {"pass": not bad and not lens, "passages_retokenized_differently": bad[:20],
            "n_passages_different": len(bad), "unequal_groups": lens}


# ---------------------------------------------------------------- capture

class Recorder:
    """Hooks on every layer's sites. `want` names which (site) to capture for the current forward; captures are
    sliced to [read_lo, read_hi) along the sequence and kept on the device until the batch is reduced."""

    def __init__(self, model):
        self.layers, self.emb = C.text_layers(model)
        self.n = len(self.layers)
        self.handles, self.cap, self.calls = [], {}, {}
        self.want, self.lo, self.hi = set(), 0, 0
        self.g_layers = None          # restrict g capture to these layers (None = all)
        self.a_index = {}             # layer -> LongTensor of neuron indices to keep from `a`
        self.res_layer = None         # capture the residual entering this layer (GT's causal layer)
        for L, block in enumerate(self.layers):
            for site, (mod, kind) in C.site_modules(block).items():
                if site == "x" and C.block_style(block) == "post":
                    continue          # OLMo 2: x is h; one hook serves both
                self.handles.append(self._hook(mod, kind, site, L))
            self.handles.append(block.register_forward_pre_hook(self._res_hook(L), with_kwargs=True))
        self.handles.append(self.emb.register_forward_hook(self._emb_hook))
        parent_norm = None
        for name, mod in model.named_modules():
            if mod is self.layers:
                parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
                parent_norm = getattr(parent, "norm", None)
        if parent_norm is None:
            raise SystemExit("no final norm beside the decoder layers (G0.5 needs its input)")
        self.handles.append(parent_norm.register_forward_pre_hook(self._final_hook))
        self.post = C.block_style(self.layers[0]) == "post"

    def _store(self, site, L, t):
        self.calls[(site, L)] = self.calls.get((site, L), 0) + 1
        if site not in self.want:
            return
        if site == "g" and self.g_layers is not None and L not in self.g_layers:
            return
        t = C.first_tensor(t)[:, self.lo:self.hi]
        if site == "a" and L in self.a_index:
            t = t.index_select(-1, self.a_index[L])
        elif site == "a":
            return
        self.cap[(site, L)] = t.detach()
        if site == "h" and self.post:
            self.cap[("x", L)] = self.cap[(site, L)]

    def _hook(self, mod, kind, site, L):
        if kind == "in":
            return mod.register_forward_pre_hook(lambda m, args, s=site, l=L: self._store(s, l, args[0]))
        return mod.register_forward_hook(lambda m, args, out, s=site, l=L: self._store(s, l, out))

    def _res_hook(self, L):
        def hook(mod, args, kwargs):
            self.calls[("res", L)] = self.calls.get(("res", L), 0) + 1
            if ("res" in self.want and L == self.res_layer) or "resall" in self.want:
                hs = args[0] if args else kwargs["hidden_states"]
                self.cap[("res", L)] = hs[:, self.lo:self.hi].detach()
        return hook

    def _final_hook(self, mod, args):
        if "resall" in self.want:
            self.cap[("final", 0)] = args[0][:, self.lo:self.hi].detach()

    def _emb_hook(self, mod, args, out):
        self.calls[("emb", 0)] = self.calls.get(("emb", 0), 0) + 1
        if "emb" in self.want:
            self.cap[("emb", 0)] = out[:, self.lo:self.hi].detach()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def forward(model, ids, device):
    """Full model, logits (G0 only)."""
    import torch
    x = torch.tensor(ids, dtype=torch.long, device=device)
    return model(input_ids=x, use_cache=False).logits


def forward_body(model, ids, device):
    """The decoder without the output head: every site this part reads is inside it, and the head is about a seventh
    of OLMo 2 1B's compute."""
    import torch
    x = torch.tensor(ids, dtype=torch.long, device=device)
    body = getattr(model, "model", None)
    if body is None:
        return model(input_ids=x, use_cache=False)
    return body(input_ids=x, use_cache=False)


# ---------------------------------------------------------------- accumulators

class Sums:
    def __init__(self, cfg, fam, F, n_layers, hidden, ffn, n_types, a_widths, mid, gt_layers, keep_pg):
        nf, npairs = len(cfg["frames"]), len(cfg["pairs"])
        P, K, k, pj = 88, n_types, cfg["split_coordinates"], cfg["split_projection_dims"]
        z = lambda *s, d=np.float64: np.zeros(s, d)
        self.a = {
            "Sx": z(nf, 2, 2, n_layers, hidden), "Sh": z(nf, 2, 2, n_layers, hidden), "Sxx": z(nf, 2, n_layers, hidden),
            "n_frame": z(nf, 2, 2),
            "SDpos": z(npairs, 2, n_layers, P, hidden, d=np.float32), "SD2": z(npairs, 2, n_layers, hidden),
            "SDh": z(npairs, 2, n_layers, hidden), "n_pair": z(npairs, 2),
            "Sgdiff": z(2, n_layers, ffn), "Sg": z(2, n_layers, ffn), "Sg2": z(2, n_layers, ffn),
            "fireM": z(n_layers, ffn), "n_g": z(2),
            "Ssp": z(2, 2, n_layers, k), "Ssp2": z(2, n_layers, k), "Ssp_t": z(2, 2, n_layers, K, k, d=np.float32),
            "Spj": z(2, 2, n_layers, pj), "Spj2": z(2, n_layers, pj), "Spj_t": z(2, 2, n_layers, K, pj, d=np.float32),
            "n_type": z(2, K),
            "Semb": z(2, hidden), "Sc": z(2, n_layers, hidden), "Sm": z(2, n_layers, hidden), "n_P": z(2),
            "Sx_gt": z(2, len(gt_layers), hidden), "fire_gt": z(len(gt_layers), ffn), "n_gt": z(2),
            "Sres_gt": z(2, hidden),
        }
        for L, w in a_widths.items():
            self.a[f"Sa_{L}"] = z(2, w)
            self.a[f"Sa2_{L}"] = z(w)
            self.a[f"Sa_t_{L}"] = z(2, K, w, d=np.float32)
        self.pp = {"PD": np.zeros((2, 600, len(mid), hidden), np.float32),
                   "PD2": np.zeros((2, 600, len(mid), hidden), np.float32)}
        if keep_pg:
            self.pp["PG"] = np.zeros((600, n_layers, ffn), np.float16)
        self.done = set()

    def save(self, path, plan, extra=None):
        arrays = dict(self.a)
        arrays.update(self.pp)
        arrays["done"] = np.array(sorted(self.done), np.int64)
        arrays["plan_sha256"] = np.array(plan["sha256"])
        C.atomic_savez(path, **arrays)
        if extra is not None:
            C.atomic_write_json(Path(str(path) + ".json"), extra)

    def load(self, path, plan):
        z = np.load(path, allow_pickle=False)
        if str(z["plan_sha256"]) != plan["sha256"]:
            raise SystemExit(f"{path}: plan hash differs from this launch; refusing to resume")
        for k in self.a:
            self.a[k] = z[k]
        for k in self.pp:
            self.pp[k] = z[k]
        self.done = set(int(i) for i in z["done"])


# ---------------------------------------------------------------- the moments pass

def run_moments(cfg, prompts, stage, out, device=None, batch=8, limit=None, checkpoint=10, smoke=False, model_path=None):
    import torch
    spec, fam, F, passages, plan = plan_for(cfg, prompts, stage, limit)
    tok, model, env = C.load_model(cfg, stage, device, model_path)
    device = env["device"]
    plain_logits = None
    first_ids = None
    if smoke or not (Path(out) / stage / "partial.npz").exists():
        # G0's reference: the plain model's logits on the first batch's frame P, before any hook exists
        first_ids = [prompts["families"][fam]["frames"]["P"] + p["ids"][fam] for p in passages[:batch]]
        plain_logits = forward(model, first_ids, device).float().cpu()
    rec = Recorder(model)
    n_layers = rec.n
    hidden = rec.layers[0].mlp.gate_proj.in_features
    ffn = rec.layers[0].mlp.gate_proj.out_features
    mid = C.middle_third(n_layers)
    gt_layers = C.nearest_layers(n_layers, cfg["gt_layer_fractions"])
    lc = C.nearest_layers(n_layers, [cfg["gt_causal_fraction"]])[0]
    types = F["listed_types"]
    tindex = {t: i for i, t in enumerate(types)}
    # neurons whose `a` is kept: 256 seeded per layer plus every catalogued neuron at that layer
    cat = prompts["catalogued"][fam]
    cat_by_layer = {}
    for lst in cat.values():
        for L, n in lst:
            cat_by_layer.setdefault(int(L), set()).add(int(n))
    a_keep = {L: sorted(set(F["token_share_neurons"][str(L)]) | cat_by_layer.get(L, set())) for L in range(n_layers)}
    rec.a_index = {L: torch.tensor(v, device=device) for L, v in a_keep.items()}
    coords = {(L, s): torch.tensor(F["split_coordinates"][f"{L}:{s}"], device=device)
              for L in range(n_layers) for s in ("m", "c")}
    proj = torch.tensor(np.random.default_rng(cfg["seeds"]["projection"]).standard_normal((hidden, cfg["split_projection_dims"]))
                        / np.sqrt(hidden), dtype=torch.float32, device=device)
    keep_pg = fam == "O"
    S = Sums(cfg, fam, F, n_layers, hidden, ffn, len(types), {L: len(v) for L, v in a_keep.items()}, mid, gt_layers,
             keep_pg)
    sdir = Path(out) / stage
    sdir.mkdir(parents=True, exist_ok=True)
    partial = sdir / "partial.npz"
    resumed_gates = {}
    if partial.exists() and not smoke:
        S.load(partial, plan)
        side = Path(str(partial) + ".json")
        resumed_gates = json.loads(side.read_text()).get("gates", {}) if side.exists() else {}
        if not all(resumed_gates.get(g, {}).get("pass") for g in ("G0", "G0.5")):
            raise SystemExit(f"{partial}: no passed G0 and G0.5 recorded beside the partial sums; refusing to resume")
        print(f"resume: {len(S.done)} passages already summed", flush=True)
    frames, fidx = cfg["frames"], {f: i for i, f in enumerate(cfg["frames"])}
    pairs = cfg["pairs"]
    scored = [cfg["pairs"].index(p) for p in cfg["scored_pairs"]]
    arm_pairs = cfg["scored_pairs"]
    first, L96 = cfg["read_first"], cfg["passage_tokens"]
    g1 = g1_check(cfg, prompts, fam, tok, passages)
    meta = {"stage": stage, "env": env, "plan": plan, "n_layers": n_layers, "hidden": hidden, "ffn": ffn,
            "middle_third": mid, "gt_layers": gt_layers, "gt_causal_layer": lc, "a_keep": a_keep,
            "types": types, "gates": {"G1": g1}, "frames": frames, "pairs": pairs, "scored_pairs": cfg["scored_pairs"]}
    meta["gates"].update({k: v for k, v in resumed_gates.items() if k in ("G0", "G0.5")})
    if not g1["pass"]:
        C.atomic_write_json(sdir / "meta.json", meta)
        raise SystemExit(f"G1 failed: {g1}")
    todo = [p for p in passages if p["index"] not in S.done]
    t0, tokens = time.time(), 0
    gates_done = bool(S.done)
    for bstart in range(0, len(todo), batch):
        bp = todo[bstart:bstart + batch]
        caps = {}
        for f in frames:
            pre = F["frames"][f]
            ids = [pre + p["ids"][fam] for p in bp]
            lo = len(pre) + first
            rec.lo, rec.hi = lo, len(pre) + L96
            want = {"x", "h"}
            if f in ("A-tmpl", "M", "A-plain", "N"):
                want.add("g")
            if f == "P":
                want |= {"c", "m", "a", "emb", "res", "g"}
            rec.want, rec.cap, rec.calls = want, {}, {}
            rec.g_layers = None if f != "P" else set(gt_layers)
            if f in ("A-tmpl", "M", "A-plain", "N"):
                rec.g_layers = None
            rec.res_layer = lc
            if not gates_done and f == "P":
                if plain_logits is None or ids != first_ids:
                    raise SystemExit("G0 needs the first batch's plain logits; resume with no partial sums")
                rec.want = rec.want | {"resall"}
                logits = forward(model, ids, device)
                g0 = float((logits.float().cpu() - plain_logits).abs().max())
                meta["gates"]["G0"] = {"max_abs_logit_diff": g0, "pass": g0 == 0.0}
                fired = {k: v for k, v in rec.calls.items()}
                once = all(fired.get((s, L), 0) == 1 for L in range(n_layers) for s in ("h", "c", "m", "g", "a", "res"))
                meta["gates"]["G0.5"] = g05(rec, n_layers, env["dtype"], once)
                for g in ("G0", "G0.5"):
                    if not meta["gates"][g]["pass"]:
                        C.atomic_write_json(sdir / "meta.json", meta)
                        raise SystemExit(f"{g} failed: {meta['gates'][g]}")
                gates_done = True
            else:
                forward_body(model, ids, device)
            caps[f] = rec.cap
            tokens += sum(len(i) for i in ids)
        reduce_batch(S, caps, bp, fam, F, cfg, fidx, pairs, scored, arm_pairs, n_layers, mid, gt_layers, lc,
                     tindex, coords, proj, keep_pg)
        for p in bp:
            S.done.add(p["index"])
        if not smoke and (bstart // batch + 1) % checkpoint == 0:     # a smoke leaves no partial behind (review r1 #9)
            S.save(partial, plan, extra={"gates": meta["gates"]})
        el = time.time() - t0
        print(f"passages {len(S.done)}/{len(passages)} tokens={tokens} elapsed_s={el:.1f} tok_per_s={tokens / el:.1f}",
              flush=True)
    el = time.time() - t0
    meta["timing"] = {"elapsed_s": el, "tokens": tokens, "tokens_per_s": tokens / el if el else None,
                      "batch": batch, "passages": len(passages)}
    meta["gates"]["G2"] = g2(S, n_layers, cfg)
    if smoke:
        print(json.dumps({"stage": stage, "gates": meta["gates"], "timing": meta["timing"]}, indent=1))
        C.atomic_write_json(sdir / "smoke-meta.json", meta)
        return meta
    meta["gt_static"] = gt_static_from(S, rec.layers, gt_layers, cfg)     # a scored quantity: never in a smoke
    S.save(sdir / "moments.npz", plan)
    C.atomic_write_json(sdir / "meta.json", meta)
    if partial.exists():
        partial.unlink()
    print(f"clips_done stage={stage} passages={len(S.done)} G2={meta['gates']['G2']['pass']}", flush=True)
    return meta


def g05(rec, n_layers, dtype, once):
    """The stream is the sum of the sites, checked one add at a time in the model's own dtype (review r1 #6, #7):
    the embedding is layer 0's input, each layer's input plus c_L is h_L, h_L plus m_L is the next layer's input,
    and the last h plus the last m is the final norm's input. Each add is the operation the model performs, so in
    float32 and in bf16 alike the reconstruction should match to within one rounding of the add."""
    def rel(a, b):
        return float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-12))
    worst = rel(rec.cap[("emb", 0)], rec.cap[("res", 0)])
    for L in range(n_layers):
        worst = max(worst, rel(rec.cap[("res", L)] + rec.cap[("c", L)], rec.cap[("h", L)]))
        nxt = rec.cap[("res", L + 1)] if L + 1 < n_layers else rec.cap[("final", 0)]
        worst = max(worst, rel(rec.cap[("h", L)] + rec.cap[("m", L)], nxt))
    bound = 2.0 ** -7 if dtype == "bfloat16" else 1e-6
    return {"every_hook_once": once, "max_relative_error": worst, "bound": bound, "pass": bool(once and worst <= bound)}


def gt_static_from(S, layers, gt_layers, cfg):
    """Tuomi's static test at the four GT layers: b_L from the fit half (wikitext, positions 16 to 95), duty from
    the test half, the gate rows from the loaded model, 20 seeded random directions of norm |b_L|."""
    a = S.a
    if a["n_gt"][0] == 0 or a["n_gt"][1] == 0:
        return {"note": "no wikitext passages in both halves"}
    out = []
    rng = np.random.default_rng(cfg["seeds"]["gt_random"] + 1)
    for j, L in enumerate(gt_layers):
        W = layers[L].mlp.gate_proj.weight.detach().float().cpu().numpy()
        b = a["Sx_gt"][0, j] / a["n_gt"][0]
        duty = a["fire_gt"][j] / a["n_gt"][1]
        R = rng.standard_normal((cfg["gt_random_directions"], W.shape[1]))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        st = C.gt_static(W, b, duty, R)
        st.update({"layer": L, "b_norm": float(np.linalg.norm(b)), "duty_median": float(np.median(duty))})
        out.append(st)
    return out


def g2(S, n_layers, cfg):
    a = S.a
    P = cfg["frames"].index("P")
    out, ok = [], True
    for L in range(n_layers):
        rf, rt = a["Sh"][P, :, 0, L].sum(0), a["Sh"][P, :, 1, L].sum(0)
        bf, bt = a["Sx"][P, :, 0, L].sum(0), a["Sx"][P, :, 1, L].sum(0)
        cr, cb = C.cosine(rf, rt), C.cosine(bf, bt)
        out.append({"layer": L, "r": cr, "b": cb})
        ok = ok and cr is not None and cb is not None and cr >= cfg["g2_cosine"] and cb >= cfg["g2_cosine"]
    return {"pass": bool(ok), "layers": out}


def reduce_batch(S, caps, bp, fam, F, cfg, fidx, pairs, scored, arm_pairs, n_layers, mid, gt_layers, lc,
                 tindex, coords, proj, keep_pg):
    import torch
    a = S.a
    halves = np.array([0 if p["half"] == "fit" else 1 for p in bp])
    srcs = np.array([0 if p["source"] == "wikitext" else 1 for p in bp])
    pidx = [p["index"] for p in bp]
    first = cfg["read_first"]
    for f, cap in caps.items():
        i = fidx[f]
        for L in range(n_layers):
            x = cap[("x", L)].float()
            h = cap[("h", L)].float()
            xs, hs, xx = x.sum(1).double().cpu().numpy(), h.sum(1).double().cpu().numpy(), (x * x).sum(1).double().cpu().numpy()
            for b in range(len(bp)):
                a["Sx"][i, srcs[b], halves[b], L] += xs[b]
                a["Sh"][i, srcs[b], halves[b], L] += hs[b]
                a["Sxx"][i, halves[b], L] += xx[b]
        for b in range(len(bp)):
            a["n_frame"][i, srcs[b], halves[b]] += x.shape[1]
    # pairs
    for pi, (X, Y) in enumerate(pairs):
        cx, cy = caps[X], caps[Y]
        for L in range(n_layers):
            D = cx[("x", L)].float() - cy[("x", L)].float()                    # [B, 88, D]
            Dh = cx[("h", L)].float() - cy[("h", L)].float()
            for hf in (0, 1):
                sel = torch.tensor(np.where(halves == hf)[0], device=D.device)
                if len(sel):
                    a["SDpos"][pi, hf, L] += D.index_select(0, sel).sum(0).cpu().numpy()
                    a["SD2"][pi, hf, L] += (D.index_select(0, sel) ** 2).sum((0, 1)).double().cpu().numpy()
                    a["SDh"][pi, hf, L] += Dh.index_select(0, sel).sum((0, 1)).double().cpu().numpy()
            if pi in scored and L in mid:
                k, li = scored.index(pi), mid.index(L)
                pm, p2 = D.mean(1).cpu().numpy(), (D * D).mean(1).cpu().numpy()
                for b, ix in enumerate(pidx):
                    S.pp["PD"][k, ix, li] = pm[b]
                    S.pp["PD2"][k, ix, li] = p2[b]
        for hf in (0, 1):
            a["n_pair"][pi, hf] += int((halves == hf).sum()) * cx[("x", 0)].shape[1]
    # arming (g) on the scored pairs
    for k, (X, Y) in enumerate(arm_pairs):
        for L in range(n_layers):
            gx, gy = caps[X][("g", L)].float(), caps[Y][("g", L)].float()
            d = gx - gy
            a["Sgdiff"][k, L] += d.sum((0, 1)).double().cpu().numpy()
            a["Sg"][k, L] += gy.sum((0, 1)).double().cpu().numpy()
            a["Sg2"][k, L] += (gy * gy).sum((0, 1)).double().cpu().numpy()
            if k == 0:
                a["fireM"][L] += (gy > 0).sum((0, 1)).double().cpu().numpy()
                if keep_pg:
                    pm = d.mean(1).cpu().numpy().astype(np.float16)
                    for b, ix in enumerate(pidx):
                        S.pp["PG"][ix, L] = pm[b]
        a["n_g"][k] += gx.shape[0] * gx.shape[1]
    # frame P: split, token shares, write shares, GT static
    cp = caps["P"]
    ids = np.array([p["ids"][fam][first:cfg["passage_tokens"]] for p in bp])         # [B, 88]
    tix = np.vectorize(lambda t: tindex.get(int(t), -1))(ids)
    for b in range(len(bp)):
        for t in tix[b][tix[b] >= 0]:
            a["n_type"][halves[b], t] += 1
    flat_t = torch.tensor(tix.reshape(-1), device=cp[("x", 0)].device)
    used = flat_t >= 0
    hal = torch.tensor(np.repeat(halves, ids.shape[1]), device=flat_t.device)
    for L in range(n_layers):
        for s_i, s in enumerate(("m", "c")):
            y = cp[(s, L)].float()                                               # [B, 88, D]
            yf = y.reshape(-1, y.shape[-1])
            for name, sub in (("sp", yf.index_select(1, coords[(L, s)])), ("pj", yf @ proj)):
                for hf in (0, 1):
                    m = used & (hal == hf)
                    v = sub[m]
                    a[f"S{name}"][s_i, hf, L] += v.sum(0).double().cpu().numpy()
                    acc = torch.zeros((len(tindex), sub.shape[1]), dtype=torch.float64, device=sub.device)
                    acc.index_add_(0, flat_t[m], v.double())
                    a[f"S{name}_t"][s_i, hf, L] += acc.cpu().numpy()
                a[f"S{name}2"][s_i, L] += (sub[used] ** 2).sum(0).double().cpu().numpy()
        av = cp[("a", L)].float().reshape(-1, cp[("a", L)].shape[-1])
        for hf in (0, 1):
            m = used & (hal == hf)
            a[f"Sa_{L}"][hf] += av[m].sum(0).double().cpu().numpy()
            acc = torch.zeros((len(tindex), av.shape[1]), dtype=torch.float64, device=av.device)
            acc.index_add_(0, flat_t[m], av[m].double())
            a[f"Sa_t_{L}"][hf] += acc.cpu().numpy()
        a[f"Sa2_{L}"] += (av[used] ** 2).sum(0).double().cpu().numpy()
    emb = cp[("emb", 0)].float()
    for b in range(len(bp)):
        hf = halves[b]
        a["Semb"][hf] += emb[b].sum(0).double().cpu().numpy()
        for L in range(n_layers):
            a["Sc"][hf, L] += cp[("c", L)][b].float().sum(0).double().cpu().numpy()
            a["Sm"][hf, L] += cp[("m", L)][b].float().sum(0).double().cpu().numpy()
        a["n_P"][hf] += emb.shape[1]
    # GT static: wikitext, passage positions 16..95 = read slice [gt_first - read_first:]
    g0 = cfg["gt_first"] - first
    for b in range(len(bp)):
        if srcs[b] != 0:
            continue
        hf = halves[b]
        for j, L in enumerate(gt_layers):
            a["Sx_gt"][hf, j] += cp[("x", L)][b, g0:].float().sum(0).double().cpu().numpy()
            if hf == 1:
                a["fire_gt"][j] += (cp[("g", L)][b, g0:].float() > 0).sum(0).double().cpu().numpy()
        a["Sres_gt"][hf] += cp[("res", lc)][b, g0:].float().sum(0).double().cpu().numpy()
        a["n_gt"][hf] += cp[("x", 0)].shape[1] - g0


# ---------------------------------------------------------------- the GT causal pass

def run_gt(cfg, prompts, stage, out, device=None, batch=8, limit=None, model_path=None):
    import torch
    spec, fam, F, passages, plan = plan_for(cfg, prompts, stage, limit)
    sdir = Path(out) / stage
    if not (sdir / "moments.npz").exists():
        raise SystemExit(f"{sdir}/moments.npz is missing: run the moments pass to the end first (review r1 #4)")
    z = np.load(sdir / "moments.npz")
    meta = json.loads((sdir / "meta.json").read_text())
    lc = meta["gt_causal_layer"]
    mhat = C.unit(z["Sres_gt"][0])                       # the fit half's mean residual entering layer lc
    tok, model, env = C.load_model(cfg, stage, device, model_path)
    device = env["device"]
    layers, _ = C.text_layers(model)
    rng = np.random.default_rng(cfg["seeds"]["gt_random"])
    R = rng.standard_normal((cfg["gt_random_directions"], len(mhat)))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    conds = [("clean", None), ("removal", None)] + [(f"random{i}", R[i]) for i in range(len(R))]
    mh = torch.tensor(mhat, dtype=torch.float32, device=device)
    test = [p for p in passages if p["source"] == "wikitext" and p["half"] == "test"]
    pre = F["frames"]["P"]
    lo, hi = len(pre) + cfg["gt_first"], len(pre) + cfg["passage_tokens"]
    state = {"cond": None, "duty": None}

    def res_hook(mod, args, kwargs):
        c = state["cond"]
        if c is None or c[0] == "clean":
            return None
        hs = args[0] if args else kwargs["hidden_states"]
        seg = hs[:, lo:hi].float()
        proj = seg @ mh                                     # [B, T]
        direction = mh if c[0] == "removal" else torch.tensor(c[1], dtype=torch.float32, device=hs.device)
        new = hs.clone()
        new[:, lo:hi] = (seg - proj[..., None] * direction).to(hs.dtype)
        if args:
            return (new,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = new
        return args, kwargs

    def gate_hook(mod, args, out):
        state["duty"] = (out[:, lo:hi].float() > 0).float().mean((1, 2)).cpu().numpy()   # per passage
        raise StopForward

    h1 = layers[lc].register_forward_pre_hook(res_hook, with_kwargs=True)
    h2 = layers[lc].mlp.gate_proj.register_forward_hook(gate_hook)
    duty = {c[0]: [] for c in conds}
    try:
        for bstart in range(0, len(test), batch):
            bp = test[bstart:bstart + batch]
            ids = torch.tensor([pre + p["ids"][fam] for p in bp], dtype=torch.long, device=device)
            for c in conds:
                state["cond"] = c
                try:
                    model(input_ids=ids, use_cache=False)
                except StopForward:
                    pass
                duty[c[0]].extend(state["duty"].tolist())
    finally:
        h1.remove()
        h2.remove()
    clean = np.array(duty["clean"])
    rec = {"stage": stage, "layer": lc, "n_passages": len(clean), "duty_clean": float(clean.mean()),
           "per_passage": {k: v for k, v in duty.items()}}
    C.atomic_write_json(sdir / "gt-causal.json", rec)
    print(f"gt_done stage={stage} layer={lc} passages={len(clean)}", flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--prompts", default=str(HERE / "prompts206.json"))
    ap.add_argument("--out", default="results/part206")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--checkpoint", type=int, default=10)
    ap.add_argument("--pass", dest="which", choices=["moments", "gt"], default="moments")
    ap.add_argument("--smoke", action="store_true", help="timing smoke: print throughput and gates only, keep no sums")
    ap.add_argument("--model-path", default=None, help="a local snapshot directory of the pinned revision (the Arc)")
    a = ap.parse_args(argv)
    cfg = C.load_config()
    raw = Path(a.prompts).read_bytes()
    prompts = json.loads(raw)
    prompts["_sha256"] = hashlib.sha256(raw).hexdigest()
    if a.which == "gt":
        run_gt(cfg, prompts, a.stage, a.out, a.device, a.batch, a.limit, a.model_path)
    else:
        run_moments(cfg, prompts, a.stage, a.out, a.device, a.batch, a.limit, a.checkpoint, a.smoke, a.model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
