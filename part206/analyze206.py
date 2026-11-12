#!/usr/bin/env python3
"""analyze206.py: turn one model's sums (results/part206/<stage>/moments.npz, meta.json, gt-causal.json) into
results/part206/<stage>/summary.json. Every number is computed by common206's estimators.

  python -I -B part206/analyze206.py --stage instruct [--out results/part206]

Resampling (Design, Size and Reading rules): 1000 passage resamples drawn from config seed `bootstrap`, within each
half, as index draws over the sorted passage indices; with the full passage set the draws are identical at every
stage, so score206.py can pair them across stages (206.7, 206.12). 206.12's neuron resampling within each set uses
seed `shuffle` + 1, also identical across stages.
Written by Claude (Opus 5.5), 2026-11-12.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common206 as C  # noqa: E402


def div(a, n):
    n = np.asarray(n, float)
    return np.where(n > 0, a / np.where(n > 0, n, 1), np.nan)


def split_layer(a, s_i, L, pre, n_type):
    """The split for sublayer s_i at layer L from prefix pre ('sp' coordinates or 'pj' projection)."""
    used = n_type.sum(1)                                   # [2] positions used per half
    fit = a[f"S{pre}"][s_i, 0, L] / used[0]
    test = a[f"S{pre}"][s_i, 1, L] / used[1]
    sq = a[f"S{pre}2"][s_i, L] / used.sum()
    keep = (n_type[0] > 0) & (n_type[1] > 0)
    tf = a[f"S{pre}_t"][s_i, 0, L][keep] / n_type[0][keep][:, None]
    tt = a[f"S{pre}_t"][s_i, 1, L][keep] / n_type[1][keep][:, None]
    w = (n_type[0] + n_type[1])[keep]
    w = w / w.sum()
    return fit, test, sq, w, tf, tt, keep


def analyze(cfg, stage, out):
    sdir = Path(out) / stage
    z = np.load(sdir / "moments.npz")
    a = {k: z[k] for k in z.files}
    meta = json.loads((sdir / "meta.json").read_text())
    n_layers, mid = meta["n_layers"], meta["middle_third"]
    frames, pairs = meta["frames"], [tuple(p) for p in meta["pairs"]]
    fi = {f: i for i, f in enumerate(frames)}
    P = fi["P"]
    S = {"stage": stage, "family": meta["env"]["family"], "n_layers": n_layers, "middle_third": mid,
         "gates": meta["gates"], "env": meta["env"]}
    n_type = a["n_type"]

    # (1) the split, frame P
    split = {}
    for s_i, s in enumerate(("m", "c")):
        rows = []
        for L in range(n_layers):
            fit, test, sq, w, tf, tt, keep = split_layer(a, s_i, L, "sp", n_type)
            Cc, T, R = C.split_shares(fit, test, sq, w, tf, tt)
            row = {"layer": L, **C.shares_summary(Cc, T, R)}
            pf, pt, psq, pw, ptf, ptt, _ = split_layer(a, s_i, L, "pj", n_type)
            row["power"] = C.power_shares(pf, pt, psq, pw, ptf, ptt)
            pooled = (a["Ssp_t"][s_i, 0, L][keep] + a["Ssp_t"][s_i, 1, L][keep]) / (n_type[0] + n_type[1])[keep][:, None]
            overall = (a["Ssp"][s_i, 0, L] + a["Ssp"][s_i, 1, L]) / n_type.sum()
            row["frequency_explained"] = C.frequency_explained(w, pooled, overall,
                                                               np.log((n_type[0] + n_type[1])[keep]))
            rows.append(row)
        split[s] = rows
    S["split"] = split
    S["split_coverage"] = float(n_type.sum() / a["n_frame"][P].sum())
    # neuron token shares of `a`
    nts = {}
    keep_all = (n_type[0] > 0) & (n_type[1] > 0)
    w = (n_type[0] + n_type[1])[keep_all]
    w = w / w.sum()
    used = n_type.sum(1)
    for L in range(n_layers):
        idx = meta["a_keep"][str(L)] if isinstance(meta["a_keep"], dict) and str(L) in meta["a_keep"] else meta["a_keep"][L]
        fit = a[f"Sa_{L}"][0] / used[0]
        test = a[f"Sa_{L}"][1] / used[1]
        sq = a[f"Sa2_{L}"] / used.sum()
        tf = a[f"Sa_t_{L}"][0][keep_all] / n_type[0][keep_all][:, None]
        tt = a[f"Sa_t_{L}"][1][keep_all] / n_type[1][keep_all][:, None]
        Cc, T, R = C.split_shares(fit, test, sq, w, tf, tt)
        ts = np.where((T + R) != 0, T / np.where((T + R) != 0, T + R, 1), np.nan)
        nts[str(L)] = {str(n): float(v) for n, v in zip(idx, ts)}
    S["neuron_token_share"] = nts

    # (2) the reference
    nP = a["n_P"]
    ref = []
    for L in range(n_layers):
        r_half = [a["Sh"][P, :, h, L].sum(0) / a["n_frame"][P, :, h].sum() for h in (0, 1)]
        b_half = [a["Sx"][P, :, h, L].sum(0) / a["n_frame"][P, :, h].sum() for h in (0, 1)]
        shares = []
        for ha, hb in ((0, 1), (1, 0)):
            groups = {"embedding": a["Semb"][ha] / nP[ha],
                      "mlp": a["Sm"][ha, :L].sum(0) / nP[ha],
                      "mixer": a["Sc"][ha, :L + 1].sum(0) / nP[ha]}
            shares.append(C.write_shares(groups, r_half[hb]))
        avg = {g: float(np.mean([s[g] for s in shares])) for g in shares[0]}
        r_all = (r_half[0] + r_half[1]) / 2
        src = [a["Sh"][P, s_, :, L].sum(0) / a["n_frame"][P, s_].sum() for s_ in (0, 1)]
        refM = a["Sh"][fi["M"], :, :, L].sum((0, 1)) / a["n_frame"][fi["M"]].sum()
        refA = a["Sh"][fi["A-tmpl"], :, :, L].sum((0, 1)) / a["n_frame"][fi["A-tmpl"]].sum()
        ref.append({"layer": L, "write_share": avg, "r_norm": float(np.linalg.norm(r_all)),
                    "b_norm": float(np.linalg.norm((b_half[0] + b_half[1]) / 2)),
                    "cos_halves_r": C.cosine(*r_half), "cos_halves_b": C.cosine(*b_half),
                    "cos_wikitext_ultrachat": C.cosine(*src), "cos_P_M": C.cosine(r_all, refM),
                    "cos_P_Atmpl": C.cosine(r_all, refA)})
    S["reference"] = ref

    # (3) common mode
    floor = cfg["rho_sd_floor_fraction"]
    cm = {}
    deltas = {}
    for pi, (X, Y) in enumerate(pairs):
        yi = fi[Y]
        n_fit_pos, n_test_pos = a["n_pair"][pi]
        npass_fit, npass_test = n_fit_pos / 88, n_test_pos / 88
        rows = []
        for L in range(n_layers):
            dfit = a["SDpos"][pi, 0, L].sum(0) / n_fit_pos
            dtest = a["SDpos"][pi, 1, L].sum(0) / n_test_pos
            d2 = a["SD2"][pi, 1, L] / n_test_pos
            nY = a["n_frame"][yi, :, 0].sum()
            mY = a["Sx"][yi, :, 0, L].sum(0) / nY
            sdY = np.sqrt(np.maximum(a["Sxx"][yi, 0, L] / nY - mY ** 2, 0))
            scale = C.sd_floor(sdY, floor)
            pf, pt = a["SDpos"][pi, 0, L] / npass_fit, a["SDpos"][pi, 1, L] / npass_test
            rows.append({"layer": L, "rho": C.rho(dfit, dtest, d2, scale), "rho_raw": C.rho(dfit, dtest, d2),
                         "kappa": C.kappa(pf, pt, d2, scale), "kappa_raw": C.kappa(pf, pt, d2)})
            deltas[(pi, L)] = (a["SDpos"][pi, 0, L].sum(0) + a["SDpos"][pi, 1, L].sum(0)) / (n_fit_pos + n_test_pos)
        mids = [r for r in rows if r["layer"] in mid and r["rho"] is not None]
        peak = max(mids, key=lambda r: r["rho"]) if mids else None
        cm[f"{X}|{Y}"] = {"layers": rows, "rho_peak": peak["rho"] if peak else None,
                          "rho_peak_layer": peak["layer"] if peak else None}
    # Delta(A-tmpl, M) against Delta(A2-tmpl, M), per layer
    pa, pb = pairs.index(("A-tmpl", "M")), pairs.index(("A2-tmpl", "M"))
    S["cos_request_change"] = [C.cosine(deltas[(pa, L)], deltas[(pb, L)]) for L in range(n_layers)]
    # resampling for the scored pairs: rho_peak per resample, and rho within the fit half for L*
    rng = np.random.default_rng(cfg["seeds"]["bootstrap"])
    B = cfg["bootstrap_resamples"]
    pp = {"PD": a["PD"], "PD2": a["PD2"]}
    present = sorted(set(int(i) for i in a["done"]))
    fit_ix = np.array([i for i in present if i % 2 == 0])
    test_ix = np.array([i for i in present if i % 2 == 1])
    draws = [(rng.integers(0, len(fit_ix), len(fit_ix)), rng.integers(0, len(test_ix), len(test_ix))) for _ in range(B)]
    for k, (X, Y) in enumerate(meta.get("scored_pairs", [["A-tmpl", "M"], ["A-plain", "N"]])):
        pi = pairs.index((X, Y))
        yi = fi[Y]
        scales = []
        for L in mid:
            nY = a["n_frame"][yi, :, 0].sum()
            mY = a["Sx"][yi, :, 0, L].sum(0) / nY
            scales.append(C.sd_floor(np.sqrt(np.maximum(a["Sxx"][yi, 0, L] / nY - mY ** 2, 0)), floor))
        peaks = []
        for df_i, dt_i in draws:
            fsel, tsel = fit_ix[df_i], test_ix[dt_i]
            vals = []
            for li, L in enumerate(mid):
                vals.append(C.rho(pp["PD"][k, fsel, li].mean(0), pp["PD"][k, tsel, li].mean(0),
                                  pp["PD2"][k, tsel, li].mean(0), scales[li]))
            vals = [v for v in vals if v is not None]
            peaks.append(max(vals) if vals else None)
        # rho inside the fit half (its even and odd members), for the causal check's L*
        q1, q2 = fit_ix[0::2], fit_ix[1::2]
        fit_only = []
        for li, L in enumerate(mid):
            fit_only.append({"layer": L, "rho": C.rho(pp["PD"][k, q1, li].mean(0), pp["PD"][k, q2, li].mean(0),
                                                       pp["PD2"][k, q2, li].mean(0), scales[li])})
        key = f"{X}|{Y}"
        cm[key]["rho_peak_resamples"] = peaks
        cm[key]["rho_peak_ci"] = C.bootstrap_ci(peaks)
        cm[key]["rho_fit_half"] = fit_only
        best = max((r for r in fit_only if r["rho"] is not None), key=lambda r: r["rho"], default=None)
        cm[key]["L_star"] = best["layer"] if best else None
    S["common_mode"] = cm

    # (4) arming on (A-tmpl, M)
    arm = {}
    for k, (X, Y) in enumerate((("A-tmpl", "M"), ("A-plain", "N"))):
        n = a["n_g"][k]
        mean_d = a["Sgdiff"][k] / n
        mY = a["Sg"][k] / n
        sdY = np.sqrt(np.maximum(a["Sg2"][k] / n - mY ** 2, 0))
        zz = C.arming_z(mean_d, sdY)                         # [L, F]
        rate = a["fireM"] / a["n_g"][0]
        arm[f"{X}|{Y}"] = {"z": zz, "rate": rate, "sdY": sdY}
    S["_arming_arrays"] = arm                                # stripped before writing; used below
    return S, a, meta


def neuron_table(S, cat, cfg):
    zz, rate = S["_arming_arrays"]["A-tmpl|M"]["z"], S["_arming_arrays"]["A-tmpl|M"]["rate"]
    out = {}
    for name, lst in cat.items():
        rows = []
        for L, n in lst:
            if L >= zz.shape[0]:
                continue
            rows.append({"layer": L, "neuron": n, "z": float(zz[L, n]),
                         "percentile": C.percentile_in(zz[L], zz[L, n]),
                         "percentile_activity_matched": C.activity_matched_percentile(zz[L], rate[L], n,
                                                                                        cfg["activity_neighbours"])})
        out[name] = {"neurons": rows, "median_percentile": float(np.median([r["percentile"] for r in rows])) if rows else None,
                     "median_percentile_activity_matched": float(np.median([r["percentile_activity_matched"] for r in rows])) if rows else None}
    return out


def gap_resamples(S, a, cat, cfg):
    """206.12: the slot-minus-concept gap in median arming percentile per resample, resampling passages (all 600,
    from per-passage gate differences) and, within each set, neurons; both draws identical at every stage."""
    if "PG" not in a:
        return None
    PG = a["PG"].astype(np.float32)                         # [600, L, F]
    present = sorted(set(int(i) for i in a["done"]))
    sdY = S["_arming_arrays"]["A-tmpl|M"]["sdY"]
    rng_p = np.random.default_rng(cfg["seeds"]["bootstrap"] + 12)
    rng_n = np.random.default_rng(cfg["seeds"]["shuffle"] + 1)
    B = cfg["bootstrap_resamples"]
    pres = np.array(present)
    W = np.zeros((B, len(pres)), np.float32)
    for b in range(B):
        np.add.at(W[b], rng_p.integers(0, len(pres), len(pres)), 1.0)
    W /= len(pres)
    slot, concept = [tuple(x) for x in cat["slot"]], [tuple(x) for x in cat["concept"]]
    sdraw = [rng_n.integers(0, len(slot), len(slot)) for _ in range(B)]
    cdraw = [rng_n.integers(0, len(concept), len(concept)) for _ in range(B)]
    layers = sorted({L for L, _ in slot + concept})
    pct = {}
    for L in layers:
        zb = (W @ PG[pres, L, :])                           # [B, F] mean differences per resample
        sd = np.where(sdY[L] > 0, sdY[L], 1.0)
        zb = np.where(sdY[L] > 0, zb / sd, 0.0)
        srt = np.sort(zb, axis=1)
        for (LL, n) in [x for x in slot + concept if x[0] == L]:
            v = zb[:, n]
            lo = np.array([np.searchsorted(srt[b], v[b], "left") for b in range(B)])
            hi = np.array([np.searchsorted(srt[b], v[b], "right") for b in range(B)])
            pct[(L, n)] = 100.0 * (lo + 0.5 * (hi - lo)) / zb.shape[1]
    gaps = []
    for b in range(B):
        sm = np.median([pct[slot[i]][b] for i in sdraw[b]])
        cmed = np.median([pct[concept[i]][b] for i in cdraw[b]])
        gaps.append(float(sm - cmed))
    return gaps


def label_shuffle(nt, cfg):
    """206.8's null: the gap over shuffles of the slot and concept labels among the catalogued neurons."""
    slot = [r["percentile"] for r in nt["slot"]["neurons"]]
    concept = [r["percentile"] for r in nt["concept"]["neurons"]]
    allp = np.array(slot + concept)
    rng = np.random.default_rng(cfg["seeds"]["shuffle"])
    g = []
    for _ in range(cfg["label_shuffles"]):
        perm = rng.permutation(len(allp))
        g.append(float(np.median(allp[perm[:len(slot)]]) - np.median(allp[perm[len(slot):]])))
    return {"gaps": g, "p95": float(np.quantile(g, 0.95))}


def gt_causal(sdir, cfg):
    p = sdir / "gt-causal.json"
    if not p.exists():
        return None
    g = json.loads(p.read_text())
    pp = {k: np.array(v) for k, v in g["per_passage"].items()}
    clean = pp["clean"]
    d = pp["removal"] - clean
    rng = np.random.default_rng(cfg["seeds"]["bootstrap"] + 31)
    boots = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(cfg["bootstrap_resamples"])]
    rand = [float((pp[k] - clean).mean()) for k in sorted(pp) if k.startswith("random")]
    return {"layer": g["layer"], "duty_clean": float(clean.mean()), "delta_duty": float(d.mean()),
            "ci": C.bootstrap_ci(boots), "delta_duty_random": rand, "n_passages": int(len(d))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", default="results/part206")
    ap.add_argument("--prompts", default=str(HERE / "prompts206.json"))
    a_ = ap.parse_args(argv)
    cfg = C.load_config()
    prompts = json.loads(Path(a_.prompts).read_text())
    S, a, meta = analyze(cfg, a_.stage, a_.out)
    cat = prompts["catalogued"][S["family"]]
    S["arming"] = neuron_table(S, cat, cfg)
    if S["family"] == "O":
        S["arming_label_shuffle"] = label_shuffle(S["arming"], cfg)
        S["arming_gap_resamples"] = gap_resamples(S, a, cat, cfg)
    sdir = Path(a_.out) / a_.stage
    gtc = gt_causal(sdir, cfg)
    st = meta.get("gt_static")
    S["gt"] = {"static": st, "causal": gtc}
    if isinstance(st, list) and gtc is not None:
        S["gt"]["verdict"] = C.gt_verdict(st, gtc)
    del S["_arming_arrays"]
    C.atomic_write_json(sdir / "summary.json", S)
    print(f"summary written: {sdir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
