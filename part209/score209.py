#!/usr/bin/env python3
"""score209.py: Part 209's gates and predictions, exactly as PART209-DESIGN.md registers them. Reads only.

  python -I -B part209/score209.py --raw results/part209 [--resamples 10000]

Inputs under --raw: r1r3-<stage>.json (run209.py), r2-<stage>.jsonl (run209.py --r2), labels-<reader>.jsonl and
reader-check-<reader>.json (read209.py). Writes <raw>/scores.json and prints every number.

Definitions (Design, Reading rules):
  sitting share  = sitting / (sitting + standing + lying + other), primary reader; none and unreadable answers are
                   counted per arm and left out of the share.
  s (R1)         = SIT / P3 per frame, averaged over the arm's paraphrases (each paraphrase weighs the same).
  c (R3)         = (mean projection of the arm - mean of TOLD-STAND) / (mean of TOLD-SIT - mean of TOLD-STAND), on the
                   sit-stand axis at the stage's chosen layer, recomputed here from the per-frame projections.
  bootstrap      = 10,000 resamples: paraphrases k drawn with replacement (shared across arms, so a difference between
                   arms is paired by k), then each drawn paraphrase's answers drawn with replacement (a multinomial
                   draw from that paraphrase's observed label counts); 95% percentile intervals.
Gate reading: an R1 prediction is read at a stage that passes G0 and G1's R1 half; an R3 prediction at a stage that also
passes G1's R3 half (Design: "A checkpoint failing R3's half has its R3 reported but not read"). R2 predictions need G2
and, per model, told-arm compliance of at least 80%.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLMO_STAGES = ["pretrain-3.0T", "midtrain-26B", "midtrain-end", "sft", "dpo", "instruct"]
CATS = ["sitting", "standing", "lying", "other", "none", "unreadable"]
POSTURE_OF_TOLD = {"TOLD-SIT": "sitting", "TOLD-STAND": "standing", "TOLD-LIE": "lying"}
R1_OF_TOLD = {"TOLD-SIT": "SIT", "TOLD-STAND": "STAND", "TOLD-LIE": "LIE"}


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


class Boot:
    """Clustered bootstrap over paraphrases k (shared across arms) and answers within each paraphrase."""

    def __init__(self, counts_by_arm, resamples, seed, ks):
        """counts_by_arm: arm -> {k: counts vector}. ks: the paraphrase numbers (clusters)."""
        self.ks = list(ks)
        self.rng = np.random.default_rng(seed)
        self.R = int(resamples)
        draws = self.rng.integers(0, len(self.ks), size=(self.R, len(self.ks)))
        self.draws = draws
        self.counts = {}
        for arm, by_k in counts_by_arm.items():
            C = len(next(iter(by_k.values()))) if by_k else 1
            mat = np.zeros((len(self.ks), C))
            for i, k in enumerate(self.ks):
                if k in by_k:
                    mat[i] = by_k[k]
            n = mat.sum(1).astype(np.int64)
            with np.errstate(invalid="ignore", divide="ignore"):
                p = np.where(n[:, None] > 0, mat / np.where(n[:, None] > 0, n[:, None], 1), 1.0 / C)
            nn = n[draws]                       # [R, K]
            pp = p[draws]                       # [R, K, C]
            res = self.rng.multinomial(nn, pp)  # [R, K, C]
            self.counts[arm] = res.sum(1)        # [R, C]

    def ci(self, values):
        v = np.asarray(values, dtype=float)
        v = v[~np.isnan(v)]
        if v.size == 0:
            return [None, None]
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def counts_by_arm_k(rows, labels, cats=CATS, key=category):
    out = {}
    for r in rows:
        lab = labels.get(r["row_id"])
        if lab is None:
            continue
        v = out.setdefault(r["arm"], {}).setdefault(int(r["k"]), np.zeros(len(cats)))
        v[cats.index(key(lab))] += 1
    return out


def valence_cat(lab):
    return "negative" if lab.get("valence") == "negative" else "not-negative"


# ---------------------------------------------------------------- gates

def gate_g0(rec, cfg):
    told = [r for r in rec["r1"] if r["arm"] in R1_OF_TOLD]
    p3 = float(np.mean([r["P3"] for r in told])) if told else float("nan")
    return {"mean_P3_told": p3, "n_told_frames": len(told), "pass": bool(told) and p3 >= cfg["gates"]["G0_p3"]}


def gate_g1_r1(rec, cfg):
    out = {}
    ok = True
    for arm, key in R1_OF_TOLD.items():
        rows = [r for r in rec["r1"] if r["arm"] == arm]
        wins = sum(1 for r in rows if max(("SIT", "STAND", "LIE"), key=lambda q: r[q]) == key)
        out[arm] = {"wins": wins, "of": len(rows)}
        ok = ok and len(rows) > 0 and wins >= cfg["gates"]["G1_r1_told_wins"]
    out["pass"] = bool(ok)
    return out


def proj_by_arm_k(rec, axis="sit_stand"):
    out = {}
    for r in rec["r3"][axis]["frames"]:
        out.setdefault(r["arm"], {})[int(r["k"])] = r["proj"]
    return out


def gate_g1_r3(rec, cfg):
    pk = proj_by_arm_k(rec)
    sit, stand = pk.get("TOLD-SIT", {}), pk.get("TOLD-STAND", {})
    ks = sorted(set(sit) & set(stand))
    pairs = sum(1 for k in ks if sit[k] > stand[k])
    acc = rec["r3"]["sit_stand"]["chosen_accuracy"]
    ok = pairs >= cfg["gates"]["G1_r3_pairs"] and acc >= cfg["gates"]["G1_r3_acc"]
    return {"pairs_sit_above_stand": pairs, "of": len(ks), "heldout_accuracy": acc,
            "chosen_layer": rec["r3"]["sit_stand"]["chosen_layer"], "pass": bool(ok)}


def c_values(pk, arms_ks=None):
    """c per arm from {arm: {k: proj}}; arms_ks optionally restricts each arm to a list of k (bootstrap)."""
    def mean(arm):
        d = pk.get(arm, {})
        ks = arms_ks if arms_ks is not None else sorted(d)
        vals = [d[k] for k in ks if k in d]
        return float(np.mean(vals)) if vals else float("nan")
    m_sit, m_stand = mean("TOLD-SIT"), mean("TOLD-STAND")
    den = m_sit - m_stand
    return {a: ((mean(a) - m_stand) / den if den != 0 else float("nan")) for a in pk}


def s_by_arm(rec, ks=None):
    out = {}
    for arm in sorted({r["arm"] for r in rec["r1"]}):
        rows = {int(r["k"]): r for r in rec["r1"] if r["arm"] == arm and r.get("s") is not None}
        use = ks if ks is not None else sorted(rows)
        vals = [rows[k]["s"] for k in use if k in rows]
        out[arm] = float(np.mean(vals)) if vals else float("nan")
    return out


def paraphrase_boot(fn, ks_all, resamples, seed):
    """Bootstrap a function of a list of paraphrase numbers (drawn with replacement)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(resamples):
        ks = [ks_all[i] for i in rng.integers(0, len(ks_all), len(ks_all))]
        vals.append(fn(ks))
    v = np.array(vals, dtype=float)
    v = v[~np.isnan(v)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if v.size else [None, None]


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
                            "read_r1": g0["pass"] and g1a["pass"],
                            "read_r3": g0["pass"] and g1a["pass"] and g1b["pass"]}
        ks_all = sorted({int(r["k"]) for r in rec["r1"]})
        s = s_by_arm(rec)
        out["r1"][st] = {"s": s, "P3_mean": {a: float(np.mean([r["P3"] for r in rec["r1"] if r["arm"] == a]))
                                             for a in s}}
        pk = proj_by_arm_k(rec)
        c = c_values(pk)
        cis = {a: paraphrase_boot(lambda ks, a=a: c_values(pk, ks).get(a, float("nan")), ks_all, min(R, 2000),
                                  seed + 7) for a in c}
        out["r3"][st] = {"chosen_layer": rec["r3"]["sit_stand"]["chosen_layer"],
                         "heldout_accuracy": rec["r3"]["sit_stand"]["chosen_accuracy"],
                         "null": rec["r3"]["sit_stand"]["null"], "c": c, "c_ci95": cis,
                         "sit_lie": {"chosen_layer": rec["r3"]["sit_lie"]["chosen_layer"],
                                     "heldout_accuracy": rec["r3"]["sit_lie"]["chosen_accuracy"],
                                     "c": rec["r3"]["sit_lie"]["c"]}}
        log(f"[{st}] G0 P3={g0['mean_P3_told']:.3f} {'pass' if g0['pass'] else 'FAIL'} | G1 R1 "
            + " ".join(f"{a}:{v['wins']}/{v['of']}" for a, v in g1a.items() if a != "pass")
            + f" {'pass' if g1a['pass'] else 'FAIL'} | G1 R3 pairs {g1b['pairs_sit_above_stand']}/{g1b['of']} "
            f"acc {g1b['heldout_accuracy']:.3f} L{g1b['chosen_layer']} {'pass' if g1b['pass'] else 'FAIL'}")
        log(f"[{st}] s: " + ", ".join(f"{a} {v:.3f}" for a, v in s.items()))
        log(f"[{st}] c: " + ", ".join(f"{a} {v:.2f}" for a, v in c.items()))

    # ---- R2: readers, gate G2
    prim, sec = cfg["readers"]["primary"]["name"], cfg["readers"]["second"]["name"]
    L1, L2 = last_labels(raw / f"labels-{prim}.jsonl"), last_labels(raw / f"labels-{sec}.jsonl")
    checks = {}
    for name in (prim, sec):
        p = raw / f"reader-check-{name}.json"
        checks[name] = load_json(p) if p.exists() else None
    common = sorted(set(L1) & set(L2))
    kap = kappa([category(L1[i]) for i in common], [category(L2[i]) for i in common], CATS) if common else None
    checks_ok = all(c is not None and c.get("passed") for c in checks.values())
    g2 = {"checks": {n: (None if c is None else {"right": c["right"], "of": c["n"], "passed": c["passed"]})
                     for n, c in checks.items()},
          "kappa_posture": kap, "n_both": len(common),
          "pass": bool(checks_ok and kap is not None and kap >= cfg["gates"]["G2_kappa"])}
    out["gates"]["G2"] = g2
    log(f"[G2] reader checks {g2['checks']} kappa {kap} on {len(common)} answers -> "
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
        b = Boot(counts, R, seed, ks)
        bv = Boot(vcounts, R, seed + 1, ks)
        boots[st] = (b, bv, counts, vcounts)
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
            vt = sum(vcounts.get(arm, {}).values()) if arm in vcounts else np.zeros(2)
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

    def diff_ci(st, arm_a, arm_b, fn=share_from_counts):
        b = boots[st][0]
        if arm_a not in b.counts or arm_b not in b.counts:
            return [None, None]
        return b.ci(fn(b.counts[arm_a]) - fn(b.counts[arm_b]))

    q = "qwen-chat"
    # 209.1
    why = r2_ok(q)
    if why:
        notread("209.1", why)
    else:
        v = share(q, "ROOM-W")
        verdict("209.1", v >= 0.70, share=v, ci95=out["r2"][q]["arms"]["ROOM-W"]["sitting_share_ci95"])
    # 209.2
    if why:
        notread("209.2", why)
    else:
        d = share(q, "FIELD-W") - share(q, "ROOM-W")
        verdict("209.2", d <= -0.20, field_minus_room=d, ci95=diff_ci(q, "FIELD-W", "ROOM-W"))
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
        places = ["ROOM", "FIELD", "NONE"]
        d = float(np.mean([share(q, f"{p}-W") - share(q, f"{p}-0") for p in places]))
        b = boots[q][0]
        dv = np.mean([share_from_counts(b.counts[f"{p}-W"]) - share_from_counts(b.counts[f"{p}-0"])
                      for p in places], axis=0)
        verdict("209.4", d >= 0.10, mean_warmup_effect=d, ci95=b.ci(dv))
    # 209.5
    if why:
        notread("209.5", why)
    else:
        v = out["r2"][q]["arms"]["NOBODY"]["none_frac"]
        verdict("209.5", v >= 0.60, none_frac=v, ci95=out["r2"][q]["arms"]["NOBODY"]["none_frac_ci95"])
    # 209.6 (R3, qwen-chat)
    if q not in out["gates"]:
        notread("209.6", "no R1/R3 record for qwen-chat")
    elif not out["gates"][q]["read_r3"]:
        notread("209.6", "qwen-chat fails G0 or G1")
    else:
        c = out["r3"][q]["c"]
        verdict("209.6", c["NOBODY"] >= 0.5 and c["NOBODY"] > c["OBJECT"], c_nobody=c["NOBODY"],
                c_object=c["OBJECT"], ci95_nobody=out["r3"][q]["c_ci95"]["NOBODY"])
    # 209.7: both chat models
    whys = [w for w in (r2_ok("instruct"), r2_ok(q)) if w]
    if whys:
        notread("209.7", "; ".join(whys))
    else:
        vals = {st: out["r2"][st]["arms"]["NOBODY"]["negative_frac"] for st in ("instruct", q)}
        verdict("209.7", all(v <= 0.10 for v in vals.values()), negative_frac=vals)
    # 209.8: OLMo R1 ROOM-W across the six stages
    missing = [st for st in OLMO_STAGES if st not in out["gates"]]
    failing = [st for st in OLMO_STAGES if st in out["gates"] and not out["gates"][st]["read_r1"]]
    if missing or failing:
        notread("209.8", f"missing {missing}, failing G0/G1-R1 {failing}")
    else:
        ss = [out["r1"][st]["s"]["ROOM-W"] for st in OLMO_STAGES]
        steps = [abs(ss[i + 1] - ss[i]) for i in range(len(ss) - 1)]
        verdict("209.8", ss[0] >= 0.5 and max(steps) < 0.15, s_by_stage=dict(zip(OLMO_STAGES, ss)),
                largest_adjacent_change=max(steps))
    # 209.9: Qwen base vs chat R1 ROOM-W
    qs = ["qwen-base", q]
    if any(st not in out["gates"] or not out["gates"][st]["read_r1"] for st in qs):
        notread("209.9", "a Qwen stage is missing or fails G0/G1-R1")
    else:
        d = out["r1"][q]["s"]["ROOM-W"] - out["r1"]["qwen-base"]["s"]["ROOM-W"]
        verdict("209.9", abs(d) < 0.15, chat_minus_base=d, s_chat=out["r1"][q]["s"]["ROOM-W"],
                s_base=out["r1"]["qwen-base"]["s"]["ROOM-W"])
    # 209.10: OLMo Instruct R2 ROOM-W
    w = r2_ok("instruct")
    if w:
        notread("209.10", w)
    else:
        v = share("instruct", "ROOM-W")
        verdict("209.10", v >= 0.60, share=v, ci95=out["r2"]["instruct"]["arms"]["ROOM-W"]["sitting_share_ci95"])

    # ---- headline withdrawal
    hl = {}
    if q in out["gates"]:
        g1q = out["gates"][q]["G1_r1"]["pass"] and out["gates"][q]["G1_r3"]["pass"]
        sq = out["r1"][q]["s"]
        gap = abs(sq.get("OBJECT", float("nan")) - sq.get("ROOM-W", float("nan")))
        hl = {"G1_qwen_chat": g1q, "object_minus_room_abs": gap,
              "withdrawn": (not g1q) or (gap < 0.1)}
        log(f"[headline] G1 at qwen-chat {g1q}; |s(OBJECT) - s(ROOM-W)| = {gap:.3f}; "
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
