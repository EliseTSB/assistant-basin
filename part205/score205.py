#!/usr/bin/env python3
"""score205.py: reads Part 205's raw records and computes everything the Predictions and Reading rules name.

  python score205.py --raw results/part205 --out results/part205/results.json

Inputs (written by run205.py, one directory per stage): ranking.npz, ranking.json, sets.json, records.jsonl,
meta.json. (run205.py also writes effect.npz and copy_acts.npz; this scorer reads neither — the effect-matched family
reads sets.json["effect"] — so they are not inputs, review r2 D-F9.)
Output: results.json with one entry per stage and a top-level "predictions" block whose keys the prediction table can
name (205.1 to 205.9), plus "withdrawal" and "read_rules" stating which reads each gate stopped.

Computed (Design, Method and Controls):
  top-80 sets by score (SAY, DO, halves a and b of each), Retest, Cross, O with its 1,000-resample bootstrap
  (re-ranked from the stored estimates, within value and half), per-value share of each set's score mass
  T, T_SAY, T_swap, reverse, T_personally, off-target shift, polarity split, broken-answer counts
  the three random families with every relaxed part printed and "not fillable" reported, the random-set rule
  gates G0 to G4 exactly as written (G1 on the full last-position logits; for family Q the A770 rule from
  g1check205.py's g1.json when its plain runs are not bit-identical)
Every readout (p_a, p_b) is the option letters' probability normalised over the two; the raw masses (mass_a, mass_b,
option_mass) are printed beside them and read only by G0 and the broken-answer count.
A gate that fails stops the reads the design says it stops: G0 stops O and T; G1 stops everything; G2 stops T and the
copy arms (O still read); G3 stops every copy arm; G4 stops O.
A stage with an incomplete copy arm (fewer question-name pairs than it planned) is not read for T or for the copy
arms, and reports how many pairs it has; O is read only from the complete rankings, which do not depend on the copy
arms (review r2 H-N1, D-F7).

Every number is printed with its file and key. Nothing here is typed: each value comes from a file.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common205 as C  # noqa: E402

TOP_K = 80


# ---------------------------------------------------------------- loading

def load_stage_dir(d):
    """Everything score205 reads for one stage, or None when the stage has not run."""
    import numpy as np
    d = Path(d)
    if not (d / "ranking.npz").exists() or not (d / "sets.json").exists():
        return None
    z = np.load(d / "ranking.npz")
    est = z["est"].astype(np.float64)
    keys = [str(k) for k in z["keys"]]
    rj = json.loads((d / "ranking.json").read_text(encoding="utf-8"))
    sets = json.loads((d / "sets.json").read_text(encoding="utf-8"))
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    recs = C.read_jsonl(d / "records.jsonl")   # a torn last line is ignored with a warning (review r1 H-F7)
    return {"dir": str(d), "est": est, "keys": keys, "prompts": rj["prompts"], "sets": sets, "meta": meta,
            "records": recs}


# ---------------------------------------------------------------- ranking, sets, overlap

def prompt_scores(est, prompts):
    """Each prompt's |e| divided by its sum over all parts (Design, Ranking, lines 90-92)."""
    import numpy as np
    per = np.zeros(est.shape)
    for i in range(est.shape[0]):
        a = np.abs(est[i])
        s = a.sum()
        per[i] = a / s if s > 0 else 0.0
    return per


def top_set(score, keys, k=TOP_K):
    order = sorted(range(len(keys)), key=lambda i: (-score[i], keys[i]))
    return [keys[i] for i in order[:k]]


def ranked_sets(est, prompts, keys, idx=None, k=TOP_K):
    """Top-k sets for SAY, DO, SAY_a, SAY_b, DO_a, DO_b over the prompts in `idx` (all when None)."""
    import numpy as np
    idx = list(range(len(prompts))) if idx is None else idx
    per = prompt_scores(est, prompts)
    out = {}
    for name, mask in (
        ("SAY", lambda p: p["kind"] == "SAY"),
        ("DO", lambda p: p["kind"] == "DO"),
        ("SAY_a", lambda p: p["kind"] == "SAY" and p["item"] in (1, 2)),
        ("SAY_b", lambda p: p["kind"] == "SAY" and p["item"] in (3, 4)),
        ("DO_a", lambda p: p["kind"] == "DO" and p["item"] in (1, 2)),
        ("DO_b", lambda p: p["kind"] == "DO" and p["item"] in (3, 4)),
    ):
        rows = [i for i in idx if mask(prompts[i])]
        if rows:
            out[name] = top_set(per[rows].mean(axis=0), keys, k)
    return out


def overlap_stats(sets):
    """Retest, Cross, O and the four cross pairings (Design, Overlap, lines 97-103)."""
    sa = set(sets["SAY_a"])
    sb = set(sets["SAY_b"])
    da = set(sets["DO_a"])
    db = set(sets["DO_b"])
    retest_say = len(sa & sb)
    retest_do = len(da & db)
    retest = (retest_say + retest_do) / 2.0
    pairs = {"SAY_a_x_DO_a": len(sa & da), "SAY_a_x_DO_b": len(sa & db),
             "SAY_b_x_DO_a": len(sb & da), "SAY_b_x_DO_b": len(sb & db)}
    cross = sum(pairs.values()) / 4.0
    o = cross / retest if retest > 0 else None
    return {"retest_say": retest_say, "retest_do": retest_do, "retest": retest, "cross": cross, "pairs": pairs,
            "O": o}


def bootstrap_o(est, prompts, keys, resamples, seed, k=TOP_K):
    """1,000 resamples of the selection prompts, drawn with replacement within each value and half, re-ranked from
    the stored estimates (no new passes). Returns (lo, hi, draws) at 95%."""
    import numpy as np
    rng = np.random.default_rng(seed)
    strata = {}
    for i, p in enumerate(prompts):
        half = "a" if p["item"] in (1, 2) else "b"
        strata.setdefault((p["value"], p["kind"], half), []).append(i)
    strata = {key: np.array(v) for key, v in strata.items()}
    draws = []
    for _ in range(resamples):
        idx = []
        for rows in strata.values():
            idx.extend(rng.choice(rows, size=len(rows), replace=True).tolist())
        s = ranked_sets(est, prompts, keys, idx, k)
        st = overlap_stats(s)
        draws.append(st["O"])
    vals = np.array([d for d in draws if d is not None], dtype=float)
    if len(vals) == 0:
        return None, None, []
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), vals.tolist()


def paired_bootstrap_rise(est_a, est_b, prompts, keys, resamples, seed, k=TOP_K):
    """The rise O_b - O_a from paired resamples: each draw re-ranks both stages from the same prompt indices."""
    import numpy as np
    rng = np.random.default_rng(seed)
    strata = {}
    for i, p in enumerate(prompts):
        half = "a" if p["item"] in (1, 2) else "b"
        strata.setdefault((p["value"], p["kind"], half), []).append(i)
    strata = {key: np.array(v) for key, v in strata.items()}
    draws = []
    for _ in range(resamples):
        idx = []
        for rows in strata.values():
            idx.extend(rng.choice(rows, size=len(rows), replace=True).tolist())
        oa = overlap_stats(ranked_sets(est_a, prompts, keys, idx, k))["O"]
        ob = overlap_stats(ranked_sets(est_b, prompts, keys, idx, k))["O"]
        if oa is not None and ob is not None:
            draws.append(ob - oa)
    if not draws:
        return None, None
    lo, hi = np.percentile(np.array(draws), [2.5, 97.5])
    return float(lo), float(hi)


def value_shares(est, prompts, keys, a_set):
    """Each value's share of a set's score mass (Design, line 103)."""
    import numpy as np
    per = prompt_scores(est, prompts)
    mean = per.mean(axis=0)
    pos = {k: i for i, k in enumerate(keys)}
    total = sum(mean[pos[k]] for k in a_set)
    out = {}
    for v in sorted({p["value"] for p in prompts}):
        rows = [i for i, p in enumerate(prompts) if p["value"] == v]
        m = per[rows].mean(axis=0)
        out[v] = float(sum(m[pos[k]] for k in a_set) / total) if total else None
    return out


def signed_share(est, prompts, keys, a_set, kind=None):
    """Share of a set whose mean signed e is positive (Design, Not used, lines 163-167)."""
    import numpy as np
    rows = [i for i, p in enumerate(prompts) if kind is None or p["kind"] == kind]
    mean = est[rows].mean(axis=0)
    pos = {k: i for i, k in enumerate(keys)}
    vals = [mean[pos[k]] for k in a_set]
    return float(sum(v > 0 for v in vals) / len(vals)) if vals else None


def composition(a_set, keys, meta):
    """Heads against neurons, and for Q full-attention against DeltaNet heads."""
    units = C.units_for_meta(meta)
    out = {"heads": 0, "neurons": 0}
    for k in a_set:
        u = units[k]
        out["heads" if u.kind == "head" else "neurons"] += 1
        if u.kind == "head" and u.sub:
            out[f"head_{u.sub}"] = out.get(f"head_{u.sub}", 0) + 1
    return out


# ---------------------------------------------------------------- copy arms

def by_arm(records):
    out = {}
    for r in records:
        out.setdefault(r["arm"], []).append(r)
    return out


def p_consistent(r):
    return r["p_a"] if r["consistent"] == "A" else r["p_b"]


def ratio_of_means(arms, held_ids, names, num_arm, den_arm="baseline"):
    """T = mean(P_num - P_third) / mean(P_you - P_third) over the question-name pairs present in the records.

    P is the consistent option's probability normalised over the two letters. A pair missing its baseline, its You
    run or its numerator row is dropped and counted (n_dropped, with the ids), never silently (review r1 D-F13).
    """
    import numpy as np
    num, den, pairs, dropped = [], [], [], []
    for qid in held_ids:
        for name in names:
            third = _find(arms, den_arm, qid, name)
            you = _find(arms, "you", qid, "You")
            n_arm = _find(arms, num_arm, qid, name)
            if third is None or you is None or n_arm is None:
                dropped.append({"id": qid, "name": name, "missing": [k for k, v in (("baseline", third),
                                ("you", you), (num_arm, n_arm)) if v is None]})
                continue
            d_num = p_consistent(n_arm) - p_consistent(third)
            d_den = p_consistent(you) - p_consistent(third)
            num.append(d_num)
            den.append(d_den)
            pairs.append({"id": qid, "name": name, "d_copy": d_num, "d_gap": d_den})
    m_num = float(np.mean(num)) if num else None
    m_den = float(np.mean(den)) if den else None
    t = (m_num / m_den) if m_den not in (None, 0.0) else None
    return {"T": t, "mean_num": m_num, "mean_den": m_den, "n_pairs": len(pairs), "pairs": pairs,
            "n_expected": len(held_ids) * len(names), "n_dropped": len(dropped), "dropped": dropped}


def gap_stats(arms, held_ids, names):
    """G2's gap mean(P_you - P_third) over the held-out pairs, from the You and baseline rows alone (so it does not
    depend on any copy arm having run)."""
    import numpy as np
    d, sd_rows, dropped = [], [], 0
    for qid in held_ids:
        for name in names:
            third = _find(arms, "baseline", qid, name)
            you = _find(arms, "you", qid, "You")
            if third is None or you is None:
                dropped += 1
                continue
            d.append(p_consistent(you) - p_consistent(third))
    return {"gap": float(np.mean(d)) if d else None, "n_pairs": len(d), "n_dropped": dropped}


def reverse_stats(arms, held_ids, names):
    """205.6: how far the reverse DO copy lowers P_you, as a share of the gap (ratio of means).

    Per pair, drop = P_you - P_reverse (the You run with the DO parts copied from that name's third-person run) and
    gap = P_you - P_third. share = mean(drop) / mean(gap); 205.6 holds when share >= 0.5 ("lowers P_you by at least
    half of the gap"). A reverse copy that takes P_you all the way to P_third gives share 1 (review r1 H-F5).
    """
    import numpy as np
    drop, gap, pairs, dropped = [], [], [], 0
    for qid in held_ids:
        for name in names:
            third = _find(arms, "baseline", qid, name)
            you = _find(arms, "you", qid, "You")
            rev = _find(arms, "reverse", qid, name)
            if None in (third, you, rev):
                dropped += 1
                continue
            drop.append(p_consistent(you) - p_consistent(rev))
            gap.append(p_consistent(you) - p_consistent(third))
            pairs.append({"id": qid, "name": name, "drop": drop[-1], "gap": gap[-1]})
    m_drop = float(np.mean(drop)) if drop else None
    m_gap = float(np.mean(gap)) if gap else None
    share = (m_drop / m_gap) if m_gap not in (None, 0.0) else None
    return {"share": share, "mean_drop": m_drop, "mean_gap": m_gap, "n_pairs": len(pairs), "n_dropped": dropped,
            "pairs": pairs, "rule": "205.6 holds when mean(P_you - P_reverse) >= 0.5 x mean(P_you - P_third)"}


def polarity_split(arms, held_ids, names, arm):
    """T by the consistent option's letter (STANDARD-CONTROLS item 5)."""
    out = {}
    for letter in ("A", "B"):
        num, den = [], []
        for qid in held_ids:
            for name in names:
                third = _find(arms, "baseline", qid, name)
                you = _find(arms, "you", qid, "You")
                a = _find(arms, arm, qid, name)
                if None in (third, you, a) or a["consistent"] != letter:
                    continue
                num.append(p_consistent(a) - p_consistent(third))
                den.append(p_consistent(you) - p_consistent(third))
        import numpy as np
        out[letter] = {"T": (float(np.mean(num)) / float(np.mean(den))) if den and np.mean(den) != 0 else None,
                       "mean_num": float(np.mean(num)) if num else None,
                       "mean_den": float(np.mean(den)) if den else None, "n": len(num)}
    return out


def reverse_polarity(arms, held_ids, names):
    """reverse_stats split by the consistent option's letter (STANDARD-CONTROLS item 5)."""
    out = {}
    for letter in ("A", "B"):
        ids = [q for q in held_ids if (_find(arms, "you", q, "You") or {}).get("consistent") == letter]
        r = reverse_stats(arms, ids, names)
        out[letter] = {"share": r["share"], "mean_drop": r["mean_drop"], "mean_gap": r["mean_gap"], "n": r["n_pairs"]}
    return out


def _find(arms, arm, qid, name):
    for r in arms.get(arm, []):
        if r["id"] == qid and r.get("name") == name:
            return r
    return None


def broken_counts(arms):
    """Rows whose option mass is under 0.5 of the total next-token mass, per arm (Design, Broken answers)."""
    out = {}
    for arm, rows in arms.items():
        if not rows:
            continue
        bad = sum(1 for r in rows if r.get("option_mass", 1.0) < 0.5)
        out[arm] = {"n": len(rows), "broken": bad, "broken_share": bad / len(rows)}
    return out


def off_target_shift(arms):
    """Mean |P(A) under copy - P(A) at baseline| over the six questions and two names (Design, Off-target)."""
    import numpy as np
    base = {(r["id"], r["name"]): r["p_a"] for r in arms.get("off_target_baseline", [])}
    cop = {(r["id"], r["name"]): r["p_a"] for r in arms.get("off_target_do_copy", [])}
    diffs = [abs(cop[k] - base[k]) for k in cop if k in base and k[1] != "You"]
    return {"shift": float(np.mean(diffs)) if diffs else None, "n": len(diffs),
            "rows": [{"id": k[0], "name": k[1], "p_a_base": base[k], "p_a_copy": cop[k]} for k in sorted(cop)
                     if k in base and k[1] != "You"]}


def secondary_t(arms):
    """T_personally: ratio of means over the six values and two names (Design, Secondary arm)."""
    import numpy as np
    num, den, rows = [], [], []
    for v in sorted({r["value"] for r in arms.get("secondary_baseline", [])}):
        you = next((r for r in arms.get("secondary_you", []) if r["value"] == v), None)
        for name in sorted({r["name"] for r in arms.get("secondary_baseline", [])}):
            base = next((r for r in arms.get("secondary_baseline", []) if r["value"] == v and r["name"] == name), None)
            cop = next((r for r in arms.get("secondary_do_copy", []) if r["value"] == v and r["name"] == name), None)
            if None in (you, base, cop):
                continue
            num.append(cop["p_a"] - base["p_a"])
            den.append(you["p_a"] - base["p_a"])
            rows.append({"value": v, "name": name, "d_copy": num[-1], "d_gap": den[-1]})
    m_den = float(np.mean(den)) if den else None
    return {"T_personally": (float(np.mean(num)) / m_den) if m_den not in (None, 0.0) else None,
            "mean_num": float(np.mean(num)) if num else None, "mean_den": m_den, "n": len(num), "rows": rows}


# ---------------------------------------------------------------- gates

def gate_g0(arms, held_ids):
    rows = [r for r in arms.get("you", []) if r["id"] in held_ids]
    if not rows:
        return {"pass": False, "reason": "no You runs on the held-out questions"}
    mass = sum(r["option_mass"] for r in rows) / len(rows)
    letters = sum(bool(r["top_is_letter"]) for r in rows) / len(rows)
    return {"pass": mass >= 0.5 and letters >= 0.5, "mean_option_mass": mass, "letter_share": letters, "n": len(rows),
            "rule": "mean option mass >= 0.5 and top token is a letter on >= half"}


DO_NOTHING_ARMS = ("do_nothing", "off_target_do_nothing", "secondary_do_nothing")


def gate_g1(arms, family="O", g1check=None):
    """The do-nothing hook reproduces the plain logits exactly: every do-nothing row's max absolute difference of the
    FULL last-position logits from its own plain run is 0 (Design, G1, lines 174-178; review r1 H-F2, D-F4).

    Family Q, A770: when g1check205.py's g1.json (`g1check`) shows that its 5 plain runs are not bit-identical, the
    design's relaxed rule applies: every do-nothing difference is at most twice the largest difference among the 5
    plain runs (10 pairs), and the check's own 5 do-nothing runs must meet it. With bit-identical plain runs, or for
    family O, the rule is exact. A row without a recorded logit difference fails the gate (H-F6).
    """
    rows = [r for arm in DO_NOTHING_ARMS for r in arms.get(arm, [])]
    if not rows:
        return {"pass": False, "reason": "no do-nothing rows", "mode": "exact"}
    unrecorded = [r.get("id") for r in rows if r.get("max_abs_logit_diff") is None]
    if unrecorded:
        return {"pass": False, "reason": f"{len(unrecorded)} do-nothing rows carry no max_abs_logit_diff",
                "mode": "exact"}
    worst = max(float(r["max_abs_logit_diff"]) for r in rows)
    out = {"max_abs_diff": worst, "n": len(rows),
           "per_arm": {a: {"n": len(arms.get(a, [])),
                           "max_abs_diff": max((float(r["max_abs_logit_diff"]) for r in arms.get(a, [])),
                                               default=None)} for a in DO_NOTHING_ARMS},
           "compared": "full last-position logits, max absolute difference"}
    a770 = (g1check or {}).get("verdict") if family == "Q" else None
    if family == "Q":
        out["a770_check"] = a770 if a770 is not None else "g1.json absent"
    if a770 is not None and not a770.get("pass"):
        out.update({"pass": False, "mode": "a770", "reason": "the A770 check (g1check205.py) failed its own rule"})
        return out
    if a770 is not None and not a770.get("bit_identical_plain"):
        tol = 2.0 * float(a770["worst_plain"])
        out.update({"pass": worst <= tol, "mode": "a770-relaxed", "tolerance": tol,
                    "rule": "every do-nothing difference <= 2 x the largest plain-against-plain difference"})
        return out
    out.update({"pass": worst == 0.0, "mode": "exact", "rule": "max absolute logit difference == 0"})
    return out


def gate_g1_a770(plain_logits, dn_logits):
    """The A770 check's verdict (Design, G1, lines 175-178). plain_logits, dn_logits: the 5 + 5 full last-position
    logit rows of one question. Plain against plain is every pair of the 5 (10 pairs); each do-nothing run is compared
    with the first plain run. Only scalars are returned (all printed)."""
    import torch

    def d(a, b):
        return float((torch.as_tensor(a).float() - torch.as_tensor(b).float()).abs().max())
    n = len(plain_logits)
    plain_diffs = [d(plain_logits[i], plain_logits[j]) for i in range(n) for j in range(i + 1, n)]
    dn_diffs = [d(x, plain_logits[0]) for x in dn_logits]
    bit_identical = all(x == 0.0 for x in plain_diffs)
    worst_plain = max(plain_diffs) if plain_diffs else 0.0
    out = {"mode": "a770", "plain_pairs": len(plain_diffs), "plain_diffs": plain_diffs, "do_nothing_diffs": dn_diffs,
           "worst_plain": worst_plain, "worst_do_nothing": max(dn_diffs) if dn_diffs else None,
           "bit_identical_plain": bit_identical, "compared": "full last-position logits, max absolute difference"}
    if bit_identical:
        out["pass"] = all(x == 0.0 for x in dn_diffs)
    else:
        out["pass"] = all(x <= 2 * worst_plain for x in dn_diffs)
    out["rule"] = ("plain runs bit-identical: exact (0); otherwise every do-nothing difference from the first plain "
                   "run at most twice the largest difference among the plain runs (all pairs)")
    return out


def gate_g2(g):
    """`g` is gap_stats(...) (or a ratio_of_means result, whose mean_den is the same gap over its pairs)."""
    gap = g["gap"] if "gap" in g else g["mean_den"]
    return {"pass": gap is not None and gap >= 0.10, "gap": gap, "n_pairs": g.get("n_pairs"),
            "rule": "mean(P_you - P_third) >= 0.10 over held-out SAY questions"}


def gate_g3(t_say):
    return {"pass": t_say["T"] is not None and t_say["T"] >= 0.5, "T_SAY": t_say["T"],
            "rule": "T_SAY (ratio of means) >= 0.5"}


def gate_g4(ov):
    return {"pass": ov["retest_say"] >= 20 and ov["retest_do"] >= 20, "retest_say": ov["retest_say"],
            "retest_do": ov["retest_do"], "rule": "min(Retest_SAY, Retest_DO) >= 20 of 80, each ranking separately"}


def random_set_rule(t, random_fams, min_families=2):
    """A transfer passes when it exceeds every set in each fillable family, with at least two families fillable.
    Fillability is read per family (review r1 D-F9)."""
    fams = random_fams.get("families", {})
    fillable = [name for name, fam in fams.items() if fam.get("fillable", True) and fam.get("sets")]
    not_fillable = {name: fam.get("reason", "no sets") for name, fam in fams.items() if name not in fillable}
    if len(fillable) < min_families:
        return {"read": False, "fillable": fillable, "not_fillable": not_fillable,
                "reason": "fewer than two fillable families"}
    per = {}
    for name in fillable:
        vals = [x["T"] for x in t["random"].get(name, []) if x["T"] is not None]
        n_sets = len(fams[name]["sets"])
        per[name] = {"max_T": max(vals) if vals else None, "n": len(vals), "n_sets": n_sets,
                     "exceeds_all": (t["T"] is not None and len(vals) == n_sets and n_sets > 0 and t["T"] > max(vals))}
    ok = t["T"] is not None and all(v["exceeds_all"] for v in per.values())
    return {"read": True, "pass": ok, "fillable": fillable, "not_fillable": not_fillable, "per_family": per}


def completeness(arms, held_ids, names, copy_arms=("do_copy", "say_copy", "swap", "reverse")):
    """Whether each copy arm has every planned question-name pair (review r2 H-N1, D-F7).

    A stage killed in its copy arms holds a T computed over the pairs it happened to write; the design's T is over
    the 48 pairs, so an arm with fewer is not read. `n_expected` is the held-out questions x names the stage itself
    ran (48 at the real size). `complete` is False when any arm falls short.
    """
    n_expected = len(held_ids) * len(names)
    per = {}
    for arm in copy_arms:
        present = sum(1 for q in held_ids for n in names if _find(arms, arm, q, n) is not None)
        per[arm] = {"n_pairs": present, "n_expected": n_expected, "complete": present == n_expected}
    return {"n_expected": n_expected, "complete": all(v["complete"] for v in per.values()),
            "incomplete": [a for a, v in per.items() if not v["complete"]], "per_arm": per}


def read_flags(gates, rsr, complete=True):
    """Which reads the gates allow (Design, Reading rules): O needs G0, G1, G4; T and every copy arm need G0 to G3
    AND a complete copy arm (`complete`, from completeness()); the random-set rule needs T read and at least two
    fillable families. O does not depend on the copy arms, so an incomplete stage still reads O."""
    o = gates["G0"]["pass"] and gates["G1"]["pass"] and gates["G4"]["pass"]
    t = gates["G0"]["pass"] and gates["G1"]["pass"] and gates["G2"]["pass"] and gates["G3"]["pass"] and bool(complete)
    return {"O": bool(o), "T": bool(t), "copy_arms": bool(t), "random_set_rule": bool(t and rsr.get("read", False))}


# ---------------------------------------------------------------- one stage

def score_stage(stage, sd, cfg, seed, g1check=None):
    import numpy as np
    est, keys, prompts = sd["est"], sd["keys"], sd["prompts"]
    k = cfg.get("top_k", TOP_K)
    arms = by_arm(sd["records"])
    family = sd["meta"]["env"].get("family", "O")
    # the held-out questions are the ones the runner made You runs for (SAY 5-8, Design, Split)
    held = sorted({r["id"] for r in arms.get("you", [])})
    names = sorted({r["name"] for r in arms.get("baseline", [])})
    sets = ranked_sets(est, prompts, keys, k=k)
    ov = overlap_stats(sets)
    lo, hi, draws = bootstrap_o(est, prompts, keys, cfg["bootstrap"]["resamples"], seed, k=k)
    t = ratio_of_means(arms, held, names, "do_copy")
    t_say = ratio_of_means(arms, held, names, "say_copy")
    t_swap = ratio_of_means(arms, held, names, "swap")
    rev = reverse_stats(arms, held, names)
    gap = gap_stats(arms, held, names)
    # random families: T per set
    t["random"] = {}
    for fam_name, fam in sd["sets"]["random"].get("families", {}).items():
        rows = []
        if not fam.get("sets"):
            t["random"][fam_name] = rows
            continue
        for si in range(len(fam["sets"])):
            rows.append(ratio_of_means(arms, held, names, f"random:{fam_name}:{si}"))
        t["random"][fam_name] = rows
    gates = {"G0": gate_g0(arms, held), "G1": gate_g1(arms, family, g1check), "G2": gate_g2(gap),
             "G3": gate_g3(t_say), "G4": gate_g4(ov)}
    rsr = random_set_rule(t, sd["sets"]["random"])
    comp = completeness(arms, held, names)
    read = read_flags(gates, rsr, comp["complete"])
    out = {
        "stage": stage,
        "env": sd["meta"]["env"],
        "files": {"ranking": f"{sd['dir']}/ranking.npz", "sets": f"{sd['dir']}/sets.json",
                  "records": f"{sd['dir']}/records.jsonl"},
        "sets": {"say": sets["SAY"], "do": sets["DO"], "halves": {k: sets[k] for k in ("SAY_a", "SAY_b", "DO_a", "DO_b")},
                 "composition": {"SAY": composition(sets["SAY"], keys, sd["meta"]),
                                 "DO": composition(sets["DO"], keys, sd["meta"]),
                                 "say_do_overlap": len(set(sets["SAY"]) & set(sets["DO"]))}},
        "value_shares": {"SAY": value_shares(est, prompts, keys, sets["SAY"]),
                         "DO": value_shares(est, prompts, keys, sets["DO"])},
        "signed_positive_share": {"SAY": signed_share(est, prompts, keys, sets["SAY"]),
                                  "DO": signed_share(est, prompts, keys, sets["DO"])},
        "overlap": ov,
        "O_ci": {"lo": lo, "hi": hi, "resamples": cfg["bootstrap"]["resamples"], "seed": seed},
        "T": t["T"], "T_SAY": t_say["T"], "T_swap": t_swap["T"], "reverse": rev["share"], "gap": gap,
        "T_detail": {"T": {k: v for k, v in t.items() if k != "random"}, "T_SAY": t_say, "T_swap": t_swap,
                     "reverse": rev},
        "T_personally": secondary_t(arms),
        "off_target": off_target_shift(arms),
        "polarity": {arm: polarity_split(arms, held, names, arm) for arm in ("do_copy", "say_copy", "swap")},
        "polarity_reverse": reverse_polarity(arms, held, names),
        "broken": broken_counts(arms),
        "random_families": {name: {"fillable": fam.get("fillable", bool(fam.get("sets"))),
                                   "reason": fam.get("reason"),
                                   "relaxed": fam["relaxed"], "relaxed_detail": fam["relaxed_detail"],
                                   "distinct_sets": len({tuple(sorted(s)) for s in fam["sets"]}),
                                   "total_effect": [sum(sd["sets"]["effect"].get(k, 0.0) for k in s) for s in fam["sets"]],
                                   "T": [x["T"] for x in t["random"][name]]}
                            for name, fam in sd["sets"]["random"].get("families", {}).items()},
        "not_fillable": sd["sets"]["random"].get("not_fillable_reason"),
        "do_set_total_effect": sum(sd["sets"]["effect"].get(k, 0.0) for k in sets["DO"]),
        "gates": gates,
        "completeness": comp,
        "random_set_rule": rsr,
        "read": read,
        # kept for the withdrawal's paired resamples (not written to results.json)
        "_est": est, "_prompts": prompts, "_keys": keys,
    }
    return out


# ---------------------------------------------------------------- predictions

def predictions(stages, cfg):
    """Verdicts for 205.1 to 205.9 from the per-stage results (Design, Predictions and Reading rules)."""
    order = [s["stage"] for s in cfg["stages"]]
    st = {s["stage"]: s for s in stages if s}
    olmo = [n for n in order if n in st and cfg["families"][_fam(cfg, n)]["name"] == "OLMo 2 1B"]
    qwen = [n for n in order if n in st and cfg["families"][_fam(cfg, n)]["name"] != "OLMo 2 1B"]
    out = {}

    def o_read(n):
        return n in st and st[n]["read"]["O"]

    def t_read(n):
        return n in st and st[n]["read"]["T"]

    # 205.1: G4 passes at five or more of the six OLMo stages and at both Qwen models. A stage that has not run is
    # not a G4 failure: with any stage absent, the prediction is not read (Reading rules: a stage whose gate fails is
    # "not read", not a failed prediction; an absent stage is weaker still).
    g4_olmo = [n for n in olmo if st[n]["gates"]["G4"]["pass"]]
    g4_qwen = [n for n in qwen if st[n]["gates"]["G4"]["pass"]]
    complete = len(olmo) == 6 and len(qwen) == 2
    out["205.1"] = {"verdict": ("held" if len(g4_olmo) >= 5 and len(g4_qwen) == 2 else "failed") if complete
                    else "not read",
                    "g4_pass_olmo": len(g4_olmo), "g4_pass_qwen": len(g4_qwen), "stages_olmo": len(olmo),
                    "stages_qwen": len(qwen), "complete": complete}
    # 205.2: at OLMo Instruct, O >= 0.5
    n = "instruct"
    out["205.2"] = _verdict_ge(st, n, "O", 0.5, o_read(n))
    # 205.3: the largest rise in O between adjacent stages is mid-training end to SFT.
    pairs = []
    for a, b in zip(olmo, olmo[1:]):
        if o_read(a) and o_read(b) and st[a]["overlap"]["O"] is not None and st[b]["overlap"]["O"] is not None:
            pairs.append((a, b, st[b]["overlap"]["O"] - st[a]["overlap"]["O"]))
    need = ("midtrain-end", "sft") in [(a, b) for a, b, _ in pairs]
    if not need or len(pairs) < 3:
        out["205.3"] = {"verdict": "UNTESTABLE", "pairs_read": [(a, b, round(r, 4)) for a, b, r in pairs]}
    else:
        best = max(pairs, key=lambda x: x[2])
        rise = next(r for a, b, r in pairs if (a, b) == ("midtrain-end", "sft"))
        # Reading rules, lines 246-249: largest of the readable pairs, at least 0.2, and the rise's paired bootstrap
        # interval excludes 0 (review r1 D-F11)
        lo, hi = rise_ci(st, "midtrain-end", "sft", cfg)
        ok = best[:2] == ("midtrain-end", "sft") and rise >= 0.2 and lo is not None and lo > 0
        out["205.3"] = {"verdict": "held" if ok else "failed",
                        "pairs_read": [(a, b, round(r, 4)) for a, b, r in pairs], "largest": best[:2], "rise": rise,
                        "rise_ci": [lo, hi], "rule": "largest readable rise, >= 0.2, paired-bootstrap CI excludes 0"}
    # 205.4: at OLMo Instruct, T >= 0.5 and exceeds every set in all three random families.
    if not t_read(n):
        out["205.4"] = {"verdict": "not read", "gate": _first_failed(st[n], ("G0", "G1", "G2", "G3"))
                        if n in st else "stage absent"}
    elif not st[n]["random_set_rule"].get("read"):
        # Reading rules, lines 244-245: "a stage where fewer than two are fillable leaves 205.4 and 205.7 not read
        # there" — not a failed prediction (review r2 D-F10, H-N5)
        rsr = st[n]["random_set_rule"]
        out["205.4"] = {"verdict": "not read", "reason": "fewer than two fillable random families",
                        "fillable": rsr.get("fillable"), "not_fillable": rsr.get("not_fillable"),
                        "T": st[n]["T"], "random_set_rule": rsr}
    else:
        rsr = st[n]["random_set_rule"]
        held = (st[n]["T"] is not None and st[n]["T"] >= 0.5 and rsr.get("pass") is True
                and len(rsr.get("fillable", [])) == 3)
        out["205.4"] = {"verdict": "held" if held else "failed", "T": st[n]["T"], "random_set_rule": rsr}
    # 205.5: at 3.0T, if G0 passes: gap < 0.10, or if not, T < 0.2.
    n5 = "pretrain-3.0T"
    if n5 not in st or not st[n5]["gates"]["G0"]["pass"] or not st[n5]["gates"]["G1"]["pass"]:
        out["205.5"] = {"verdict": "not read", "gate": "G0 or G1"}
    else:
        gap = st[n5]["gates"]["G2"]["gap"]
        if gap is not None and gap < 0.10:
            out["205.5"] = {"verdict": "held", "gap": gap}
        elif not t_read(n5):
            out["205.5"] = {"verdict": "UNTESTABLE", "gap": gap, "reason": "gap >= 0.10 and G3 failed"}
        else:
            out["205.5"] = {"verdict": "held" if (st[n5]["T"] is not None and st[n5]["T"] < 0.2) else "failed",
                            "gap": gap, "T": st[n5]["T"]}
    # 205.6: reverse DO copy at OLMo Instruct lowers P_you by at least half of the gap.
    if not t_read(n):
        out["205.6"] = {"verdict": "not read", "gate": _first_failed(st[n], ("G0", "G1", "G2", "G3"))
                        if n in st else "stage absent"}
    else:
        r = st[n]["T_detail"]["reverse"]
        out["205.6"] = {"verdict": "held" if (r["share"] is not None and r["share"] >= 0.5) else "failed",
                        "share": r["share"], "mean_drop": r["mean_drop"], "mean_gap": r["mean_gap"],
                        "rule": r["rule"]}
    # 205.7: wherever DO transfer is read and passes the random-set rule, off-target shift < 0.05. A stage whose DO
    # transfer IS read but whose random-set rule is not read (fewer than two fillable families) is "not read" there,
    # not vacuous (Reading rules, lines 244-245); a stage whose DO transfer is not read is vacuous there.
    rows = []
    unread = []
    for n2 in order:
        if n2 not in st:
            continue
        if st[n2]["read"]["random_set_rule"] and st[n2]["random_set_rule"].get("pass"):
            rows.append((n2, st[n2]["off_target"]["shift"]))
        elif st[n2]["read"]["T"] and not st[n2]["random_set_rule"].get("read"):
            unread.append(n2)
    if not rows and unread:
        out["205.7"] = {"verdict": "not read", "reason": "fewer than two fillable random families",
                        "stages": unread}
    elif not rows:
        out["205.7"] = {"verdict": "vacuous", "reason": "no stage reads DO transfer and passes the random-set rule"}
    else:
        out["205.7"] = {"verdict": "held" if all(s is not None and s < 0.05 for _, s in rows) else "failed",
                        "rows": rows}
    # 205.8: Qwen3.5-4B (chat) O at least 0.2 below OLMo 2 1B Instruct's; Base reported beside it, unscored.
    if not o_read(n) or "qwen-chat" not in st or not o_read("qwen-chat"):
        out["205.8"] = {"verdict": "not read", "gate": "G0 or G4 at instruct or qwen-chat"}
    else:
        d = st[n]["overlap"]["O"] - st["qwen-chat"]["overlap"]["O"]
        out["205.8"] = {"verdict": "held" if d >= 0.2 else "failed", "difference": d,
                        "qwen_base_O": st["qwen-base"]["overlap"]["O"] if "qwen-base" in st else None}
    # 205.9: at OLMo Instruct, T_swap under half of the DO set's T.
    if not t_read(n):
        out["205.9"] = {"verdict": "not read", "gate": _first_failed(st[n], ("G0", "G1", "G2", "G3"))
                        if n in st else "stage absent"}
    else:
        out["205.9"] = {"verdict": "held" if (st[n]["T_swap"] is not None and st[n]["T"] is not None
                                              and st[n]["T_swap"] < 0.5 * st[n]["T"]) else "failed",
                        "T": st[n]["T"], "T_swap": st[n]["T_swap"]}
    return out


def _fam(cfg, stage):
    return next(s["family"] for s in cfg["stages"] if s["stage"] == stage)


def _verdict_ge(st, n, key, bar, readable):
    if not readable:
        return {"verdict": "not read", "gate": _first_failed(st[n], ("G0", "G1", "G4")) if n in st else "stage absent"}
    v = st[n]["overlap"]["O"]
    return {"verdict": "held" if v is not None and v >= bar else "failed", "value": v}


def _first_failed(s, gates):
    return [g for g in gates if not s["gates"][g]["pass"]]


def rise_ci(st, a, b, cfg):
    """The paired-bootstrap interval of O_b - O_a (Reading rules, lines 248-249), seeded from the bootstrap seed and
    the later stage's position in the stage order."""
    order = [s["stage"] for s in cfg["stages"]]
    return paired_bootstrap_rise(st[a]["_est"], st[b]["_est"], st[a]["_prompts"], st[a]["_keys"],
                                 cfg["bootstrap"]["resamples"], cfg["seeds"]["bootstrap"] + order.index(b),
                                 k=cfg.get("top_k", TOP_K))


def withdrawal(stages, cfg):
    """The headline rule (Design, Reading rules, Withdrawal)."""
    order = [s["stage"] for s in cfg["stages"]]
    st = {s["stage"]: s for s in stages if s}
    olmo = [n for n in order if n in st and _fam(cfg, n) == "O"]
    # the paired resamples are seeded from the bootstrap seed and the pair, so two stages with the same name (a tiny
    # fixture) do not draw the same index stream
    qual = []
    for a, b in zip(olmo, olmo[1:]):
        if a not in st or b not in st:
            continue
        if not (st[a]["read"]["O"] and st[b]["read"]["O"]):
            continue
        if st[a]["overlap"]["O"] is None or st[b]["overlap"]["O"] is None:
            continue   # review r1 D-F12: an O of None (Retest 0) is not a rise
        rise = st[b]["overlap"]["O"] - st[a]["overlap"]["O"]
        lo, hi = rise_ci(st, a, b, cfg)
        if rise >= 0.2 and lo is not None and lo > 0 and st[b]["overlap"]["O"] >= 0.5:
            qual.append({"stage": b, "after": a, "rise": rise, "ci": [lo, hi]})
    if qual:
        return {"headline": "say/do sharing appears at " + ", ".join(q["stage"] for q in qual), "stages": qual}
    return {"headline": "no stage creates sharing that this design can resolve", "stages": []}


# ---------------------------------------------------------------- main

def relaxation_lines(s):
    """Per family: fillable or not, and every relaxed part counted (review r1 D-F9, D-F10; never silent)."""
    lines = []
    for name, f in s["random_families"].items():
        if not f["fillable"]:
            lines.append(f"  random family {name}: NOT FILLABLE ({f.get('reason')})")
            continue
        lines.append(f"  random family {name}: fillable, {f['distinct_sets']} distinct sets, {f['relaxed']} relaxed "
                     f"parts" + (f" ({len(f['relaxed_detail'])} rows in results.json random_families.{name}"
                                 f".relaxed_detail)" if f["relaxed"] else ""))
    return "\n".join(lines) if lines else "  random families: none drawn"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default=str(HERE.parent / "results" / "part205"))
    ap.add_argument("--out", default=None, help="results.json path (default <raw>/results.json)")
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--only", default=None, help="score only these stage names (comma-separated; the rest are absent)")
    ap.add_argument("--g1", default=None, help="the A770 check's g1.json (default <raw>/g1/g1.json)")
    a = ap.parse_args(argv)
    cfg = C.load_config(a.config)
    if a.only:
        keep = set(a.only.split(","))
        cfg["stages"] = [s for s in cfg["stages"] if s["stage"] in keep]
    raw = Path(a.raw)
    g1_path = Path(a.g1) if a.g1 else raw / "g1" / "g1.json"
    g1check = json.loads(g1_path.read_text(encoding="utf-8")) if g1_path.exists() else None
    print(f"A770 G1 check: {g1_path} {'read' if g1check else 'absent'}")
    stages = []
    for s in cfg["stages"]:
        sd = load_stage_dir(raw / s["stage"])
        if sd is None:
            stages.append(None)
            continue
        stages.append(score_stage(s["stage"], sd, cfg, cfg["seeds"]["bootstrap"],
                                  g1check if s["family"] == "Q" else None))
    res = {
        "part": 205,
        "stages": [{k: v for k, v in s.items() if k not in ("_est", "_prompts", "_keys")} for s in stages if s],
        "predictions": predictions(stages, cfg),
        "withdrawal": withdrawal(stages, cfg),
        "read_rules": {
            "O": "read only where G0, G1 and G4 pass",
            "T": "read only where G0, G1, G2 and G3 pass; every copy arm with it",
            "random_set_rule": "at least two fillable families; exceeds every set in each fillable family",
        },
    }
    out = Path(a.out) if a.out else raw / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    C.atomic_write_json(out, res, indent=1, sort_keys=False)   # temp + os.replace: a kill never leaves a torn one
    print(f"wrote {out}")
    for s in res["stages"]:
        print(f"{s['stage']:14s} O={s['overlap']['O']} T={s['T']} gates="
              f"{ {g: v['pass'] for g, v in s['gates'].items()} } read={s['read']}")
        c = s["completeness"]
        pairs = ", ".join("%s %d/%d" % (a, v["n_pairs"], v["n_expected"]) for a, v in c["per_arm"].items())
        state = "complete" if c["complete"] else "INCOMPLETE " + str(c["incomplete"])
        print(f"  copy arms: {state} ({pairs})")
        print(relaxation_lines(s))
    for k, v in res["predictions"].items():
        print(f"{k}: {v['verdict']}")
    print("headline:", res["withdrawal"]["headline"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
