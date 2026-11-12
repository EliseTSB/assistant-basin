#!/usr/bin/env python3
"""score206.py: score Part 206's predictions (206.1 to 206.12) and the headline rule from results files only.

  python -I -B part206/score206.py [--out results/part206]

Reads results/part206/<stage>/summary.json for every stage, and causal.json for qwen-chat. Every number it uses is
printed with its file and key. A model that failed G0, G0.5, G1 or G2 is not read; a prediction resting on it is
"not read". 206.10 is not read if G3 failed. Verdicts: "held", "failed", "not read".
Written by Claude (Opus 5.5), 2026-11-12, from PART206-DESIGN.md draft 7 (Predictions, Reading rules).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common206 as C  # noqa: E402

LOG = []


def cite(path, key, value):
    LOG.append(f"{path} :: {key} = {value}")
    return value


def readable(S):
    g = S.get("gates", {})
    return all(g.get(k, {}).get("pass") for k in ("G0", "G0.5", "G1", "G2"))


def load(out, stages):
    res = {}
    for st in stages:
        p = Path(out) / st / "summary.json"
        if p.exists():
            res[st] = (p, json.loads(p.read_text()))
    return res


def verdict(ok):
    return "held" if ok else "failed"


def adjacent_rise_rule(order, est, resamples, target):
    """206.7 / 206.12: the target adjacent rise is the largest of the five, its paired-resample interval excludes 0,
    and that interval's lower end lies above the second-largest rise's estimate."""
    rises = {(order[i], order[i + 1]): est[order[i + 1]] - est[order[i]] for i in range(len(order) - 1)}
    ranked = sorted(rises.items(), key=lambda kv: kv[1], reverse=True)
    largest = ranked[0][0]
    second = ranked[1][1]
    r_t = [b - a for a, b in zip(resamples[target[0]], resamples[target[1]]) if a is not None and b is not None]
    ci = C.bootstrap_ci(r_t)
    ok = (largest == target and ci[0] is not None and ci[0] > 0 and ci[0] > second)
    return {"rises": {f"{a}->{b}": v for (a, b), v in rises.items()}, "largest": f"{largest[0]}->{largest[1]}",
            "target_ci": ci, "second_largest": second, "verdict": verdict(ok)}


def score(out, cfg):
    stages = [m["stage"] for m in cfg["models"]]
    R = load(out, stages)
    V = {}
    gt = {}
    for st, (p, S) in R.items():
        v = (S.get("gt") or {}).get("verdict")
        gt[st] = None if v is None or not readable(S) else cite(p, "gt.verdict.pass", v["pass"])
    # 206.1, 206.2
    q = [gt.get("qwen-base"), gt.get("qwen-chat")]
    V["206.1"] = "not read" if None in q else verdict(all(q))
    olmo = [gt.get(s) for s in cfg["olmo_stage_order"]]
    V["206.2"] = "not read" if None in olmo else verdict(sum(olmo) >= 5)
    qc = R.get("qwen-chat")
    if qc and readable(qc[1]):
        p, S = qc
        n = S["n_layers"]
        midL = C.middle_layer(n)
        share = cite(p, f"reference[{midL}].write_share.mlp", S["reference"][midL]["write_share"]["mlp"])
        V["206.3"] = verdict(share >= 0.5)
        ts0 = cite(p, "split.m[0].token_share", S["split"]["m"][0]["token_share"])
        tsm = cite(p, f"split.m[{midL}].token_share", S["split"]["m"][midL]["token_share"])
        V["206.4"] = verdict(ts0 is not None and tsm is not None and ts0 >= 0.5 and tsm <= 0.25)
        cmA = S["common_mode"]["A-tmpl|M"]
        rp = cite(p, "common_mode.A-tmpl|M.rho_peak", cmA["rho_peak"])
        ci = cite(p, "common_mode.A-tmpl|M.rho_peak_ci", cmA["rho_peak_ci"])
        lay = cmA["rho_peak_layer"]
        cs = cite(p, f"cos_request_change[{lay}]", S["cos_request_change"][lay])
        V["206.6"] = verdict(rp is not None and rp >= 0.2 and ci[0] is not None and ci[0] >= 0.1 and cs is not None and cs >= 0.8)
        med = cite(p, "arming.identity.median_percentile", S["arming"]["identity"]["median_percentile"])
        V["206.9"] = verdict(35 <= med <= 65)
        rr = cite(p, "common_mode.A-tmpl|R-tmpl.rho_peak", S["common_mode"]["A-tmpl|R-tmpl"]["rho_peak"])
        V["206.11"] = verdict(rr is not None and rp is not None and rr >= 0.5 * rp)
        cpath = Path(out) / "qwen-chat" / "causal.json"
        if cpath.exists():
            c = json.loads(cpath.read_text())
            if not c.get("G3", {}).get("pass"):
                V["206.10"] = "not read"
                cite(cpath, "G3", c.get("G3"))
            else:
                # the design's population: rows where the plain model names its maker at top-1 (G3's rows; review r1 #1)
                rows = [r for r in c["readout_a"] if "logprob" in r and r.get("plain_top1_is_maker")]
                ch = {arm: float(np.mean([r["logprob"][arm] - r["logprob"]["plain"] for r in rows])) for arm in c["arms"] if arm != "plain"}
                for k, v in ch.items():
                    cite(cpath, f"readout_a.mean_change[{k}]", v)
                ctrl = [v for k, v in ch.items() if k.endswith("_k1") and k != "arm_k1"]
                V["206.10"] = verdict(all(ch["arm_k1"] < x for x in ctrl) and ch["arm_k2"] < ch["arm_k1"])
        else:
            V["206.10"] = "not read"
    else:
        for k in ("206.3", "206.4", "206.6", "206.9", "206.10", "206.11"):
            V[k] = "not read"
    inst = R.get("instruct")
    if inst and readable(inst[1]):
        p, S = inst
        first = C.first_third(S["n_layers"])
        wins = 0
        for L in first:
            m = cite(p, f"split.m[{L}].token_share", S["split"]["m"][L]["token_share"])
            c = cite(p, f"split.c[{L}].token_share", S["split"]["c"][L]["token_share"])
            wins += int(m is not None and c is not None and m > c)
        V["206.5"] = verdict(wins >= 0.75 * len(first))
        sl = cite(p, "arming.slot.median_percentile", S["arming"]["slot"]["median_percentile"])
        co = cite(p, "arming.concept.median_percentile", S["arming"]["concept"]["median_percentile"])
        p95 = cite(p, "arming_label_shuffle.p95", S["arming_label_shuffle"]["p95"])
        gap = sl - co
        V["206.8"] = verdict(gap >= 10 and sl >= 60 and gap > p95)
        sla = cite(p, "arming.slot.median_percentile_activity_matched", S["arming"]["slot"]["median_percentile_activity_matched"])
        coa = cite(p, "arming.concept.median_percentile_activity_matched", S["arming"]["concept"]["median_percentile_activity_matched"])
        am = verdict((sla - coa) >= 10 and sla >= 60)
        V["206.8_activity_matched"] = am
        if am != V["206.8"]:
            V["206.8_note"] = "depends on the control"
        for k in ("p92b_ds_slot", "p92b_ds_concept"):
            cite(p, f"arming.{k}.median_percentile (unscored)", S["arming"][k]["median_percentile"])
    else:
        V["206.5"] = V["206.8"] = "not read"
    order = cfg["olmo_stage_order"]
    if all(s in R and readable(R[s][1]) for s in order):
        est = {s: cite(R[s][0], "common_mode.A-tmpl|M.rho_peak", R[s][1]["common_mode"]["A-tmpl|M"]["rho_peak"]) for s in order}
        res = {s: R[s][1]["common_mode"]["A-tmpl|M"]["rho_peak_resamples"] for s in order}
        V["206.7_detail"] = adjacent_rise_rule(order, est, res, ("midtrain-end", "sft"))
        V["206.7"] = V["206.7_detail"]["verdict"]
        gest = {s: R[s][1]["arming"]["slot"]["median_percentile"] - R[s][1]["arming"]["concept"]["median_percentile"] for s in order}
        for s in order:
            cite(R[s][0], "arming.slot-concept gap", gest[s])
        gres = {s: R[s][1]["arming_gap_resamples"] for s in order}
        V["206.12_detail"] = adjacent_rise_rule(order, gest, gres, ("sft", "dpo"))
        V["206.12"] = V["206.12_detail"]["verdict"]
    else:
        V["206.7"] = V["206.12"] = "not read"
    V.update(muse_rows(R))
    V["headline"] = headline(R, cfg, V)
    return V


def muse_rows(R):
    """Muse's rows 206.M1 to 206.M5 (council muse-prior-art #21), scored exactly as their table rows say."""
    V = {}
    qc = R.get("qwen-chat")
    if qc and readable(qc[1]):
        p, S = qc
        n = S["n_layers"]
        midL = C.middle_layer(n)
        cmA = S["common_mode"]["A-tmpl|M"]
        lay = cmA["rho_peak_layer"]
        rp = cite(p, "common_mode.A-tmpl|M.rho_peak", cmA["rho_peak"])
        rn = cite(p, f"common_mode.M2|M.layers[{lay}].rho", S["common_mode"]["M2|M"]["layers"][lay]["rho"])
        V["206.M1"] = verdict(rp is not None and rn is not None and 0.07 <= rp <= 0.30 and 0.02 <= rn <= 0.08)
        idm = cite(p, "arming.identity.median_percentile", S["arming"]["identity"]["median_percentile"])
        V["206.M3"] = verdict(45 <= idm <= 55)
        t0 = cite(p, "split.m[0].token_share", S["split"]["m"][0]["token_share"])
        tm = cite(p, f"split.m[{midL}].token_share", S["split"]["m"][midL]["token_share"])
        tf = cite(p, "split.m[0].token_share_freq_removed", S["split"]["m"][0]["token_share_freq_removed"])
        V["206.M4"] = verdict(None not in (t0, tm, tf) and 0.50 <= t0 <= 0.70 and 0.10 <= tm <= 0.25 and 0.25 <= tf <= 0.40)
        cb = cite(p, f"cos_delta_bM[{lay}]", S["cos_delta_bM"][lay])
        V["206.M5"] = verdict(cb is not None and cb >= 0.5)
    else:
        for k in ("206.M1", "206.M3", "206.M4", "206.M5"):
            V[k] = "not read"
    inst = R.get("instruct")
    if inst and readable(inst[1]):
        p, S = inst
        sl = S["arming"]["slot"]["median_percentile"]
        gap = sl - S["arming"]["concept"]["median_percentile"]
        cite(p, "arming.slot-concept gap (206.M2)", gap)
        V["206.M2"] = verdict(10 <= gap <= 20 and 60 <= sl <= 70)
    else:
        V["206.M2"] = "not read"
    return V


def headline(R, cfg, V):
    """The headline rule (Design, Reading rules), on OLMo Instruct's own numbers."""
    inst = R.get("instruct")
    if not inst or not readable(inst[1]):
        return {"sentence": None, "reason": "OLMo Instruct not readable"}
    p, S = inst
    cmA = S["common_mode"]["A-tmpl|M"]
    lay = cmA["rho_peak_layer"]
    c1 = cmA["rho_peak"] is not None and cmA["rho_peak"] >= 0.2
    c2 = cmA["rho_peak_ci"][0] is not None and cmA["rho_peak_ci"][0] >= 0.1
    c3 = S["cos_request_change"][lay] is not None and S["cos_request_change"][lay] >= 0.8
    if not (c1 and c2):
        return {"sentence": "the assistant frame changes each position in its own way; no constant shift at this resolution"}
    if V.get("206.8") != "held":
        # "neither neuron set moves": read as both sets' median percentiles inside 35 to 65, the band 206.9 calls
        # typical (the design does not give a number; this reading is the build's and is listed for Claude).
        sl, co = S["arming"]["slot"]["median_percentile"], S["arming"]["concept"]["median_percentile"]
        neither = 35 <= sl <= 65 and 35 <= co <= 65
        return {"sentence": "a constant assistant-frame shift exists and does not reach the catalogued neurons' gates"
                if neither else None, "conditions": [c1, c2, c3], "neither_set_moves": neither}
    if not c3:
        return {"sentence": None, "conditions": [c1, c2, c3], "reason": "206.6's third condition failed on OLMo Instruct"}
    s = ("In OLMo 2 1B Instruct, the assistant-turn frame adds nearly the same vector to the MLP's input at every "
         "passage position, a different request adds nearly the same vector, and that vector moves the mean gate "
         "pre-activation of slot neurons toward open more than that of concept neurons")
    order = cfg["olmo_stage_order"]
    X = None
    if all(st in R and readable(R[st][1]) for st in order):
        for i, st in enumerate(order):
            c = R[st][1]["common_mode"]["A-tmpl|M"]
            if c["rho_peak"] is not None and c["rho_peak"] >= 0.2 and c["rho_peak_ci"][0] is not None and c["rho_peak_ci"][0] >= 0.1:
                earlier = [R[e][1]["common_mode"]["A-tmpl|M"]["rho_peak_ci"][1] for e in order[:i]]
                if all(u is not None and u < 0.2 for u in earlier):
                    X = st
                break
    if X:
        s += f"; it first appears at stage {X}"
    rr = S["common_mode"]["A-tmpl|R-tmpl"]["rho_peak"]
    if rr is not None and rr >= 0.5 * cmA["rho_peak"]:
        s += "; it is about who is speaking"
    v = (S.get("gt") or {}).get("verdict")
    if v and v.get("pass"):
        s += "; it re-thresholds the gates"
    return {"sentence": s + ".", "stage_X": X}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/part206")
    a = ap.parse_args(argv)
    cfg = C.load_config()
    V = score(a.out, cfg)
    for line in LOG:
        print(line)
    print(json.dumps(V, indent=1, default=str))
    C.atomic_write_json(Path(a.out) / "score.json", {"verdicts": V, "cited": LOG})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
