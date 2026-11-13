#!/usr/bin/env python3
"""score209.py: Part 209's gates and predictions, exactly as PART209-DESIGN.md (draft 3) registers them. Reads only.

  python -I -B part209/score209.py --raw results/part209 [--resamples 10000]

Inputs under --raw: r1r3-<stage>.json (run209.py), r2-<stage>.jsonl (run209.py --r2), labels-<reader>.jsonl and
reader-check-<reader>.json (read209.py). Writes <raw>/scores.json and prints every number.

Definitions (Design, Method and Reading rules):
  sitting share  = sitting / (sitting + standing + lying + other), primary reader; none and unreadable answers are
                   counted per arm and left out of the share. 209.5 and 209.7 use every labelled answer in the arm as
                   the denominator, unreadable ones included.
  s (R1)         = SIT / P3 per frame; an arm's s is the mean over its paraphrases, printed with each paraphrase's value,
                   the range and the arm's mean P3. An arm with mean P3 under 0.1 is reported, not read.
  c (R3)         = (mean projection of the arm - mean of the told-standing pole) / (mean of told-sitting - mean of
                   told-standing), on the sit-stand axis at the stage's chosen layer, recomputed here from the per-frame
                   projections. NOBODY (and its own poles) uses NOBODY-TOLD-SIT / NOBODY-TOLD-STAND; every other arm
                   uses TOLD-SIT / TOLD-STAND. Intervals: 10,000 paraphrase resamples, the arm and its poles drawn
                   together. c is read only where the pole gap's bootstrap 95% lower bound is above zero; only ROOM-W's
                   and NOBODY's c are scored.
  R2 intervals   = 10,000 resamples: paraphrases k drawn with replacement, then each drawn paraphrase's answers redrawn
                   (a multinomial from its observed label counts); 95% percentile intervals. A difference between two
                   arms is bootstrapped paired by k (both arms use the same drawn k); the unpaired interval (each arm
                   drawn on its own) is printed beside it.
Readers: Cohen's kappa per label (posture with unreadable as its own category, says-no-body, valence); G2 needs posture
kappa at least 0.6; 209.7 also needs valence kappa at least 0.4 and is printed beside ROOM-W's and NONE-0's
negative-valence shares. The headline is withdrawn only if G1 fails on Qwen3.5-4B (draft 3: OBJECT is context).
Gate reading (Design, Gates and Reading rules): R1 is read at a stage that passes G0 and G1's R1 half; R3 at a stage
that passes G0 and G1's R3 half (held-out accuracy and the told poles in BOTH ROOM-W and NOBODY). 209.6 also needs the
pole-gap rule. R2 predictions need G2 and, per model, told-arm compliance of at least 80%.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLMO_STAGES = ["pretrain-3.0T", "midtrain-26B", "midtrain-end", "sft", "dpo", "instruct"]
CATS = ["sitting", "standing", "lying", "other", "none", "unreadable"]
POSTURE_OF_TOLD = {"TOLD-SIT": "sitting", "TOLD-STAND": "standing", "TOLD-LIE": "lying"}
R1_OF_TOLD = {"TOLD-SIT": "SIT", "TOLD-STAND": "STAND", "TOLD-LIE": "LIE"}
NOBODY_POLES = ("NOBODY-TOLD-SIT", "NOBODY-TOLD-STAND")
ROOM_POLES = ("TOLD-SIT", "TOLD-STAND")
SCORED_C = ("ROOM-W", "NOBODY")


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def last_labels(path):
    """row_id -> the last non-error label row."""
    out = {}
    for r in read_jsonl(path):
        if "error" in r or "posture" not in r:
            continue
        out[r["row_id"]] = r
    return out


def category(lab):
    if lab.get("unreadable"):
        return "unreadable"
    return lab["posture"]


def valence_cat(lab):
    return "negative" if lab.get("valence") == "negative" else "not-negative"


# ---------------------------------------------------------------- statistics

def share_from_counts(c):
    """c: [..., 6] counts in CATS order -> sitting share (nan when no posture is named)."""
    c = np.asarray(c, dtype=float)
    den = c[..., 0] + c[..., 1] + c[..., 2] + c[..., 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, c[..., 0] / np.where(den > 0, den, 1), np.nan)


def none_frac_from_counts(c):
    c = np.asarray(c, dtype=float)
    tot = c.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, c[..., 4] / np.where(tot > 0, tot, 1), np.nan)


def kappa(a, b, cats):
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return None
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    po = np.trace(m) / n
    pe = float((m.sum(1) / n) @ (m.sum(0) / n))
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def ci95(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return [None, None]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


class Boot:
    """Clustered bootstrap over paraphrases k and answers within each paraphrase. paired=True: every arm uses the same
    drawn k (differences are paired by k); paired=False: each arm draws its own k."""

    def __init__(self, counts_by_arm, resamples, seed, ks, paired=True):
        self.ks = list(ks)
        self.rng = np.random.default_rng(seed)
        self.R = int(resamples)
        shared = self.rng.integers(0, len(self.ks), size=(self.R, len(self.ks)))
        self.counts = {}
        for arm in sorted(counts_by_arm):
            by_k = counts_by_arm[arm]
            C = len(next(iter(by_k.values()))) if by_k else 1
            mat = np.zeros((len(self.ks), C))
            for i, k in enumerate(self.ks):
                if k in by_k:
                    mat[i] = by_k[k]
            n = mat.sum(1).astype(np.int64)
            with np.errstate(invalid="ignore", divide="ignore"):
                p = np.where(n[:, None] > 0, mat / np.where(n[:, None] > 0, n[:, None], 1), 1.0 / C)
            draws = shared if paired else self.rng.integers(0, len(self.ks), size=(self.R, len(self.ks)))
            self.counts[arm] = self.rng.multinomial(n[draws], p[draws]).sum(1)  # [R, C]

    def ci(self, values):
        return ci95(values)


def counts_by_arm_k(rows, labels, cats=CATS, key=category):
    out = {}
    for r in rows:
        lab = labels.get(r["row_id"])
        if lab is None:
            continue
        v = out.setdefault(r["arm"], {}).setdefault(int(r["k"]), np.zeros(len(cats)))
        v[cats.index(key(lab))] += 1
    return out


# ---------------------------------------------------------------- R1 and R3

def gate_g0(rec, cfg):
    told = [r for r in rec["r1"] if r["arm"] in R1_OF_TOLD]
    p3 = float(np.mean([r["P3"] for r in told])) if told else float("nan")
    return {"mean_P3_told": p3, "n_told_frames": len(told), "pass": bool(told) and p3 >= cfg["gates"]["G0_p3"]}


def gate_g1_r1(rec, cfg, need=None):
    need = cfg["gates"]["G1_r1_told_wins"] if need is None else need
    out = {}
    ok = True
    for arm, key in R1_OF_TOLD.items():
        rows = [r for r in rec["r1"] if r["arm"] == arm]
        wins = sum(1 for r in rows if max(("SIT", "STAND", "LIE"), key=lambda q: r[q]) == key)
        out[arm] = {"wins": wins, "of": len(rows)}
        ok = ok and len(rows) > 0 and wins >= need
    out["pass"] = bool(ok)
    return out


def proj_by_arm_k(rec, axis="sit_stand"):
    out = {}
    for r in rec["r3"][axis]["frames"]:
        out.setdefault(r["arm"], {})[int(r["k"])] = r["proj"]
    return out


def pole_pairs(pk, one, zero):
    a, b = pk.get(one, {}), pk.get(zero, {})
    ks = sorted(set(a) & set(b))
    return sum(1 for k in ks if a[k] > b[k]), len(ks)


def gate_g1_r3(rec, cfg, need=None):
    """Held-out accuracy at least G1_r3_acc and told sitting above told standing in at least G1_r3_pairs paraphrases in
    BOTH the ROOM-W poles and the NOBODY poles."""
    need = cfg["gates"]["G1_r3_pairs"] if need is None else need
    pk = proj_by_arm_k(rec)
    frames = {}
    ok = True
    for name, (one, zero) in (("ROOM-W", ROOM_POLES), ("NOBODY", NOBODY_POLES)):
        above, n = pole_pairs(pk, one, zero)
        frames[name] = {"poles": [one, zero], "above": above, "of": n}
        ok = ok and n > 0 and above >= need
    acc = rec["r3"]["sit_stand"]["chosen_accuracy"]
    ok = ok and acc >= cfg["gates"]["G1_r3_acc"]
    return {"frames": frames, "heldout_accuracy": acc, "chosen_layer": rec["r3"]["sit_stand"]["chosen_layer"],
            "pass": bool(ok)}


def poles_for(arm):
    return NOBODY_POLES if arm in ("NOBODY",) + NOBODY_POLES else ROOM_POLES


def _mean(pk, arm, ks):
    d = pk.get(arm, {})
    vals = [d[k] for k in ks if k in d]
    return float(np.mean(vals)) if vals else float("nan")


def c_values(pk, ks=None):
    """c per arm, each against its own frame's poles (NOBODY's poles for NOBODY, ROOM-W's for the rest)."""
    out = {}
    for arm in pk:
        use = sorted(pk[arm]) if ks is None else ks
        one, zero = poles_for(arm)
        m1, m0 = _mean(pk, one, use), _mean(pk, zero, use)
        den = m1 - m0
        out[arm] = (_mean(pk, arm, use) - m0) / den if den != 0 and not np.isnan(den) else float("nan")
    return out


def c_boot(pk, ks_all, resamples, seed):
    """Intervals for every arm's c and for both pole gaps, from one set of paraphrase draws (the arm and its poles are
    resampled together by paraphrase)."""
    rng = np.random.default_rng(seed)
    arms = sorted(pk)
    vals = {a: [] for a in arms}
    gaps = {"ROOM-W": [], "NOBODY": []}
    for _ in range(resamples):
        ks = [ks_all[i] for i in rng.integers(0, len(ks_all), len(ks_all))]
        cv = c_values(pk, ks)
        for a in arms:
            vals[a].append(cv.get(a, float("nan")))
        for name, (one, zero) in (("ROOM-W", ROOM_POLES), ("NOBODY", NOBODY_POLES)):
            gaps[name].append(_mean(pk, one, ks) - _mean(pk, zero, ks))
    return {a: ci95(v) for a, v in vals.items()}, {g: ci95(v) for g, v in gaps.items()}


def s_by_arm(rec, cfg, key="r1"):
    """Per arm: mean over paraphrases of s, each paraphrase's s, the range, the mean P3, and whether s is read
    (key "r1c" reads the chat-template readout the same way)."""
    floor = cfg["r1"]["p3_floor_read"]
    out = {}
    src = rec.get(key) or []
    for arm in sorted({r["arm"] for r in src}):
        rows = sorted((r for r in src if r["arm"] == arm), key=lambda r: int(r["k"]))
        svals = {int(r["k"]): r["s"] for r in rows if r.get("s") is not None}
        p3 = float(np.mean([r["P3"] for r in rows])) if rows else float("nan")
        vals = list(svals.values())
        out[arm] = {"s": float(np.mean(vals)) if vals else float("nan"), "by_paraphrase": svals,
                    "range": [float(min(vals)), float(max(vals))] if vals else [None, None],
                    "mean_P3": p3, "read": bool(vals) and p3 >= floor}
    return out


# ---------------------------------------------------------------- main scoring

def score(raw, cfg, resamples=None, log=print):
    raw = Path(raw)
    R = int(resamples or cfg["bootstrap"]["resamples"])
    seed = int(cfg["bootstrap"]["seed"])
    out = {"part": 209, "gates": {}, "r1": {}, "r3": {}, "r2": {}, "predictions": {}, "headline": {}}

    # ---- R1 / R3 per stage
    recs = {}
    for p in sorted(raw.glob("r1r3-*.json")):
        rec = load_json(p)
        recs[rec["stage"]] = rec
    for st, rec in recs.items():
        g0, g1a, g1b = gate_g0(rec, cfg), gate_g1_r1(rec, cfg), gate_g1_r3(rec, cfg)
        out["gates"][st] = {"G0": g0, "G1_r1": g1a, "G1_r3": g1b,
                            "read_r1": g0["pass"] and g1a["pass"], "read_r3": g0["pass"] and g1b["pass"]}
        sb = s_by_arm(rec, cfg)
        out["r1"][st] = sb
        if rec.get("r1c"):
            out.setdefault("r1c", {})[st] = s_by_arm(rec, cfg, "r1c")
        pk = proj_by_arm_k(rec)
        ks_all = sorted({k for d in pk.values() for k in d})
        c = c_values(pk)
        cis, gap_ci = c_boot(pk, ks_all, R, seed + 7)
        gap_ok = {g: (v[0] is not None and v[0] > 0) for g, v in gap_ci.items()}
        out["r3"][st] = {"chosen_layer": rec["r3"]["sit_stand"]["chosen_layer"],
                         "heldout_accuracy": rec["r3"]["sit_stand"]["chosen_accuracy"],
                         "null": rec["r3"]["sit_stand"]["null"], "c": c, "c_ci95": cis,
                         "pole_gap_ci95": gap_ci, "pole_gap_read": gap_ok,
                         "c_scored": {a: c.get(a) for a in SCORED_C},
                         "c_unscored_note": "every arm other than ROOM-W and NOBODY is read against ROOM-W's poles",
                         "sit_lie": {"chosen_layer": rec["r3"]["sit_lie"]["chosen_layer"],
                                     "heldout_accuracy": rec["r3"]["sit_lie"]["chosen_accuracy"],
                                     "c": rec["r3"]["sit_lie"]["c"]}}
        log(f"[{st}] G0 P3={g0['mean_P3_told']:.3f} {'pass' if g0['pass'] else 'FAIL'} | G1 R1 "
            + " ".join(f"{a}:{v['wins']}/{v['of']}" for a, v in g1a.items() if a != "pass")
            + f" {'pass' if g1a['pass'] else 'FAIL'} | G1 R3 "
            + " ".join(f"{n}:{v['above']}/{v['of']}" for n, v in g1b["frames"].items())
            + f" acc {g1b['heldout_accuracy']:.3f} L{g1b['chosen_layer']} {'pass' if g1b['pass'] else 'FAIL'}")
        for a, v in sb.items():
            log(f"[{st}] R1 {a}: s {v['s']:.3f} range {v['range']} mean P3 {v['mean_P3']:.3f}"
                + ("" if v["read"] else " (P3 under the floor: reported, not read)")
                + " by paraphrase " + ", ".join(f"{k}:{x:.3f}" for k, x in v["by_paraphrase"].items()))
        log(f"[{st}] R3 pole gaps {gap_ci} read {gap_ok}")
        log(f"[{st}] R3 c: " + ", ".join(f"{a} {v:.2f} {cis.get(a)}" for a, v in c.items()))

    # ---- R2: readers, gate G2
    prim, sec = cfg["readers"]["primary"]["name"], cfg["readers"]["second"]["name"]
    L1, L2 = last_labels(raw / f"labels-{prim}.jsonl"), last_labels(raw / f"labels-{sec}.jsonl")
    checks = {}
    for name in (prim, sec):
        p = raw / f"reader-check-{name}.json"
        checks[name] = load_json(p) if p.exists() else None
    common = sorted(set(L1) & set(L2))
    kap = kappa([category(L1[i]) for i in common], [category(L2[i]) for i in common], CATS) if common else None
    kap_nb = kappa([bool(L1[i].get("says_no_body")) for i in common], [bool(L2[i].get("says_no_body")) for i in common],
                   [False, True]) if common else None
    kap_val = kappa([L1[i].get("valence") for i in common], [L2[i].get("valence") for i in common],
                    ["negative", "neutral", "positive"]) if common else None
    checks_ok = all(c is not None and c.get("passed") for c in checks.values())
    g2 = {"checks": {n: (None if c is None else {k: c.get(k) for k in ("right", "n", "nonneutral_right",
                                                                         "nonneutral_n", "passed")})
                     for n, c in checks.items()},
          "kappa_posture": kap, "kappa_says_no_body": kap_nb, "kappa_valence": kap_val, "n_both": len(common),
          "pass": bool(checks_ok and kap is not None and kap >= cfg["gates"]["G2_kappa"])}
    out["gates"]["G2"] = g2
    log(f"[G2] reader checks {g2['checks']} kappa posture {kap}, says-no-body {kap_nb}, valence {kap_val} on "
        f"{len(common)} answers -> "
        f"{'pass' if g2['pass'] else 'FAIL'}")

    r2rows = {}
    for p in sorted(raw.glob("r2-*.jsonl")):
        rows = [r for r in read_jsonl(p) if "answer" in r]
        if rows:
            r2rows[rows[0]["stage"]] = rows
    boots = {}
    for st, rows in r2rows.items():
        counts = counts_by_arm_k(rows, L1)
        vcounts = counts_by_arm_k(rows, L1, cats=["negative", "not-negative"], key=valence_cat)
        ks = sorted({int(r["k"]) for r in rows})
        b = Boot(counts, R, seed, ks, paired=True)
        bu = Boot(counts, R, seed + 2, ks, paired=False)
        bv = Boot(vcounts, R, seed + 1, ks, paired=True)
        boots[st] = (b, bu)
        told = [r for r in rows if r["arm"] in POSTURE_OF_TOLD and r["row_id"] in L1]
        comp_n = sum(1 for r in told if category(L1[r["row_id"]]) == POSTURE_OF_TOLD[r["arm"]])
        comp = comp_n / len(told) if told else float("nan")
        arms = {}
        for arm, by_k in counts.items():
            tot = sum(by_k.values())
            arms[arm] = {"counts": dict(zip(CATS, [int(x) for x in tot])),
                         "sitting_share": float(share_from_counts(tot)),
                         "sitting_share_ci95": b.ci(share_from_counts(b.counts[arm])),
                         "none_frac": float(none_frac_from_counts(tot)),
                         "none_frac_ci95": b.ci(none_frac_from_counts(b.counts[arm]))}
            vt = sum(vcounts[arm].values()) if arm in vcounts else np.zeros(2)
            arms[arm]["negative_frac"] = float(vt[0] / vt.sum()) if np.sum(vt) else float("nan")
            if arm in bv.counts:
                vc = bv.counts[arm]
                arms[arm]["negative_frac_ci95"] = bv.ci(vc[:, 0] / np.maximum(vc.sum(1), 1))
        out["r2"][st] = {"n_answers": len(rows), "n_labelled": sum(1 for r in rows if r["row_id"] in L1),
                         "told_compliance": comp, "told_n": len(told),
                         "read": bool(g2["pass"] and comp >= cfg["gates"]["G2_compliance"]), "arms": arms}
        log(f"[{st}] R2 compliance {comp:.3f} on {len(told)} told answers; read={out['r2'][st]['read']}")
        for arm, v in arms.items():
            log(f"[{st}] R2 {arm}: share {v['sitting_share']:.3f} {v['sitting_share_ci95']} none "
                f"{v['none_frac']:.3f} neg {v['negative_frac']:.3f} counts {v['counts']}")

    P = out["predictions"]

    def notread(key, why):
        P[key] = {"verdict": "not read", "reason": why}
        log(f"{key}: not read ({why})")

    def verdict(key, ok, **numbers):
        P[key] = {"verdict": "held" if ok else "failed", **numbers}
        log(f"{key}: {'HELD' if ok else 'FAILED'} " + json.dumps(numbers))

    def r2_ok(st):
        if st not in out["r2"]:
            return f"no R2 data for {st}"
        if not g2["pass"]:
            return "G2 failed"
        if not out["r2"][st]["read"]:
            return f"{st} told-arm compliance {out['r2'][st]['told_compliance']:.3f} < 0.8"
        return None

    def share(st, arm):
        return out["r2"][st]["arms"].get(arm, {}).get("sitting_share", float("nan"))

    def diff(st, pairs):
        """Point value and paired / unpaired intervals of mean over pairs of share(a) - share(b)."""
        b, bu = boots[st]
        point = float(np.mean([share(st, x) - share(st, y) for x, y in pairs]))
        dv = np.mean([share_from_counts(b.counts[x]) - share_from_counts(b.counts[y]) for x, y in pairs], axis=0)
        du = np.mean([share_from_counts(bu.counts[x]) - share_from_counts(bu.counts[y]) for x, y in pairs], axis=0)
        return point, ci95(dv), ci95(du)

    q = "qwen-chat"
    why = r2_ok(q)
    # 209.1
    if why:
        notread("209.1", why)
    else:
        v = share(q, "ROOM-W")
        verdict("209.1", v >= 0.70, share=v, ci95=out["r2"][q]["arms"]["ROOM-W"]["sitting_share_ci95"])
    # 209.2
    if why:
        notread("209.2", why)
    else:
        d, cp, cu = diff(q, [("FIELD-W", "ROOM-W")])
        verdict("209.2", d <= -0.20, field_minus_room=d, ci95_paired=cp, ci95_unpaired=cu)
    # 209.3
    if why:
        notread("209.3", why)
    else:
        v = share(q, "NONE-0")
        verdict("209.3", v >= 0.50, share=v, ci95=out["r2"][q]["arms"]["NONE-0"]["sitting_share_ci95"])
    # 209.4: mean over places of share(P-W) - share(P-0) >= 0.10
    if why:
        notread("209.4", why)
    else:
        d, cp, cu = diff(q, [(f"{p}-W", f"{p}-0") for p in ("ROOM", "FIELD", "NONE")])
        verdict("209.4", d >= 0.10, mean_warmup_effect=d, ci95_paired=cp, ci95_unpaired=cu)
    # 209.5
    if why:
        notread("209.5", why)
    else:
        v = out["r2"][q]["arms"]["NOBODY"]["none_frac"]
        verdict("209.5", v >= 0.60, none_frac=v, ci95=out["r2"][q]["arms"]["NOBODY"]["none_frac_ci95"],
                denominator=sum(out["r2"][q]["arms"]["NOBODY"]["counts"].values()))
    # 209.6 (R3, qwen-chat, NOBODY's own poles)
    if q not in out["gates"]:
        notread("209.6", "no R1/R3 record for qwen-chat")
    elif not out["gates"][q]["read_r3"]:
        notread("209.6", "qwen-chat fails G0 or G1's R3 half (held-out accuracy, ROOM-W or NOBODY poles)")
    elif not out["r3"][q]["pole_gap_read"]["NOBODY"]:
        notread("209.6", f"NOBODY pole gap's 95% lower bound is not above zero {out['r3'][q]['pole_gap_ci95']['NOBODY']}")
    else:
        cn = out["r3"][q]["c"]["NOBODY"]
        verdict("209.6", cn >= 0.5, c_nobody=cn, ci95=out["r3"][q]["c_ci95"]["NOBODY"],
                pole_gap_ci95=out["r3"][q]["pole_gap_ci95"]["NOBODY"])
    # 209.7: both chat models, every labelled answer in NOBODY as the denominator
    whys = [w for w in (r2_ok("instruct"), r2_ok(q)) if w]
    if not whys and (g2["kappa_valence"] is None or g2["kappa_valence"] < 0.4):
        whys.append(f"valence kappa {g2['kappa_valence']} < 0.4")
    base = {st: {a: out["r2"][st]["arms"].get(a, {}).get("negative_frac") for a in ("NOBODY", "ROOM-W", "NONE-0")}
            for st in ("instruct", q) if st in out["r2"]}
    log(f"[209.7] negative-valence shares beside their baselines: {base}")
    if whys:
        notread("209.7", "; ".join(whys))
        P["209.7"]["negative_frac_with_baselines"] = base
    else:
        vals = {st: out["r2"][st]["arms"]["NOBODY"]["negative_frac"] for st in ("instruct", q)}
        verdict("209.7", all(v <= 0.10 for v in vals.values()), negative_frac=vals,
                negative_frac_with_baselines=base, kappa_valence=g2["kappa_valence"])
    # 209.8: OLMo R1 ROOM-W across the six stages (G0 and G1's R1 half, and ROOM-W above the P3 floor)
    missing = [st for st in OLMO_STAGES if st not in out["gates"]]
    failing = [st for st in OLMO_STAGES if st in out["gates"] and not out["gates"][st]["read_r1"]]
    floor = [st for st in OLMO_STAGES if st in out["r1"] and not out["r1"][st]["ROOM-W"]["read"]]
    if missing or failing or floor:
        notread("209.8", f"missing {missing}, failing G0/G1-R1 {failing}, ROOM-W under the P3 floor {floor}")
    else:
        ss = [out["r1"][st]["ROOM-W"]["s"] for st in OLMO_STAGES]
        steps = [abs(ss[i + 1] - ss[i]) for i in range(len(ss) - 1)]
        verdict("209.8", ss[0] >= 0.5 and max(steps) < 0.15, s_by_stage=dict(zip(OLMO_STAGES, ss)),
                largest_adjacent_change=max(steps),
                ranges={st: out["r1"][st]["ROOM-W"]["range"] for st in OLMO_STAGES})
    # 209.9: Qwen base vs chat R1 ROOM-W
    qs = ["qwen-base", q]
    if any(st not in out["gates"] or not out["gates"][st]["read_r1"] or not out["r1"][st]["ROOM-W"]["read"]
           for st in qs):
        notread("209.9", "a Qwen stage is missing, fails G0/G1-R1, or has ROOM-W under the P3 floor")
    else:
        d = out["r1"][q]["ROOM-W"]["s"] - out["r1"]["qwen-base"]["ROOM-W"]["s"]
        verdict("209.9", abs(d) < 0.15, chat_minus_base=d, s_chat=out["r1"][q]["ROOM-W"]["s"],
                s_base=out["r1"]["qwen-base"]["ROOM-W"]["s"])
    # 209.10: OLMo Instruct R2 ROOM-W
    w = r2_ok("instruct")
    if w:
        notread("209.10", w)
    else:
        v = share("instruct", "ROOM-W")
        verdict("209.10", v >= 0.60, share=v, ci95=out["r2"]["instruct"]["arms"]["ROOM-W"]["sitting_share_ci95"])
    # 209.11: Qwen R2, NOBODY-I's share within 20 points of NONE-0's
    if why:
        notread("209.11", why)
    else:
        d, cp, cu = diff(q, [("NOBODY-I", "NONE-0")])
        verdict("209.11", abs(d) <= 0.20, nobody_i_minus_none0=d, ci95_paired=cp, ci95_unpaired=cu)

    # 209.12 (Muse): FIELD-W 5 to 15 points below ROOM-W, paired by k
    if why:
        notread("209.12", why)
    else:
        d, cp, cu = diff(q, [("FIELD-W", "ROOM-W")])
        verdict("209.12", -0.15 <= d <= -0.05, field_minus_room=d, ci95_paired=cp, ci95_unpaired=cu)
    # 209.13 (Muse): NOBODY, 5% to 30% of answers name a posture for themselves, and at least 80% of those say sitting
    if why:
        notread("209.13", why)
    else:
        cnt = out["r2"][q]["arms"]["NOBODY"]["counts"]
        tot = sum(cnt.values())
        named = cnt["sitting"] + cnt["standing"] + cnt["lying"] + cnt["other"]
        frac = named / tot if tot else float("nan")
        sit = cnt["sitting"] / named if named else float("nan")
        verdict("209.13", 0.05 <= frac <= 0.30 and named > 0 and sit >= 0.80, naming_frac=frac,
                sitting_among_named=sit, named=named, denominator=tot)
    # 209.14 (Muse): ROOM-W, R1c's paraphrase-mean s exceeds R2's sitting share by at least 0.10
    r1c_q = out.get("r1c", {}).get(q, {}).get("ROOM-W")
    if why:
        notread("209.14", why)
    elif q not in out["gates"] or not out["gates"][q]["read_r1"]:
        notread("209.14", "qwen-chat is missing or fails G0/G1-R1")
    elif r1c_q is None or not r1c_q["read"]:
        notread("209.14", "no R1c ROOM-W at qwen-chat, or its mean P3 is under the floor")
    else:
        d = r1c_q["s"] - share(q, "ROOM-W")
        verdict("209.14", d >= 0.10, r1c_minus_r2=d, s_r1c=r1c_q["s"], r1c_range=r1c_q["range"],
                r2_share=share(q, "ROOM-W"))
    # 209.15 (Muse): OLMo R1 ROOM-W, the largest adjacent-stage change in s is mid-training end to SFT
    if missing or failing or floor:
        notread("209.15", f"missing {missing}, failing G0/G1-R1 {failing}, ROOM-W under the P3 floor {floor}")
    else:
        ss = [out["r1"][st]["ROOM-W"]["s"] for st in OLMO_STAGES]
        steps = {f"{OLMO_STAGES[i]}->{OLMO_STAGES[i + 1]}": abs(ss[i + 1] - ss[i]) for i in range(len(ss) - 1)}
        top = max(steps, key=steps.get)
        verdict("209.15", top == "midtrain-end->sft", largest=top, steps=steps)

    # ---- headline withdrawal (draft 3: only G1 failing on Qwen3.5-4B withdraws it; OBJECT is context)
    hl = {}
    if q in out["gates"]:
        g1q = out["gates"][q]["G1_r1"]["pass"] and out["gates"][q]["G1_r3"]["pass"]
        sq = out["r1"][q]
        gap = abs(sq.get("OBJECT", {}).get("s", float("nan")) - sq.get("ROOM-W", {}).get("s", float("nan")))
        hl = {"G1_qwen_chat": g1q, "object_minus_room_abs_context": gap, "withdrawn": not g1q}
        log(f"[headline] G1 at qwen-chat {g1q}; |s(OBJECT) - s(ROOM-W)| = {gap:.3f} (context only); "
            f"{'WITHDRAWN' if hl['withdrawn'] else 'stands'}")
    out["headline"] = hl
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(HERE.parent / "results" / "part209"))
    ap.add_argument("--resamples", type=int)
    a = ap.parse_args(argv)
    cfg = load_json(HERE / "config.json")
    res = score(a.raw, cfg, a.resamples)
    (Path(a.raw) / "scores.json").write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"wrote {Path(a.raw) / 'scores.json'}")


if __name__ == "__main__":
    main()
