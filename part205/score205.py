#!/usr/bin/env python3
"""score205.py: reads Part 205's raw records and computes everything the Predictions and Reading rules name.

  python score205.py --raw results/part205 --out results/part205/results.json

Inputs (written by run205.py, one directory per stage): ranking.npz, effect.npz, sets.json, records.jsonl, meta.json.
Output: results.json with one entry per stage and a top-level "predictions" block whose keys the prediction table can
name (205.1 to 205.9), plus "withdrawal" and "read_rules" stating which reads each gate stopped.

Computed (Design, Method and Controls):
  top-80 sets by score (SAY, DO, halves a and b of each), Retest, Cross, O with its 1,000-resample bootstrap
  (re-ranked from the stored estimates, within value and half), per-value share of each set's score mass
  T, T_SAY, T_swap, reverse, T_personally, off-target shift, polarity split, broken-answer counts
  the three random families with every relaxed part printed and "not fillable" reported, the random-set rule
  gates G0 to G4 exactly as written (including the A770 G1 variant)
A gate that fails stops the reads the design says it stops: G0 stops O and T; G1 stops everything; G2 stops T and the
copy arms (O still read); G3 stops every copy arm; G4 stops O.

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
    recs = []
    p = d / "records.jsonl"
    if p.exists():
        recs = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
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


def ratio_of_means(arms, held_ids, names, num_arm, den_arm, sub_arm=None):
    """T = mean(P_num - P_third) / mean(P_you - P_third) over the question-name pairs present in the records.

    `sub_arm` replaces the third-person baseline in the numerator's difference (the persona-swap null: the copy comes
    from the other name, the denominator stays P_you - P_third of the receiving name).
    """
    def get(arm, qid, name):
        for r in arms.get(arm, []):
            if r["id"] == qid and r.get("name") == name:
                return r
        return None
    num, den, pairs = [], [], []
    for qid in held_ids:
        for name in names:
            third = get("baseline", qid, name)
            you = get("you", qid, "You")
            n_arm = get(num_arm, qid, name)
            if third is None or you is None or n_arm is None:
                continue
            d_num = p_consistent(n_arm) - p_consistent(third)
            d_den = p_consistent(you) - p_consistent(third)
            num.append(d_num)
            den.append(d_den)
            pairs.append({"id": qid, "name": name, "d_copy": d_num, "d_gap": d_den})
    import numpy as np
    m_num = float(np.mean(num)) if num else None
    m_den = float(np.mean(den)) if den else None
    t = (m_num / m_den) if m_den not in (None, 0.0) else None
    if m_den is None:
        t = None
    return {"T": t, "mean_num": m_num, "mean_den": m_den, "n_pairs": len(pairs), "pairs": pairs,
            "n_num_rows": len(num), "n_den_rows": len(den)}


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


def gate_g1(arms, a770=None):
    """The do-nothing hook reproduces the plain logits exactly (max abs difference 0).

    On the A770 (family Q) the pilot runs one question 5 times plain and 5 times with the do-nothing hook: if the
    plain runs are bit-identical, exact; otherwise every do-nothing run's difference from the first plain run is at
    most twice the largest difference among the 5 plain runs.
    """
    dn = arms.get("do_nothing", [])
    base = {(r["id"], r["name"]): r for r in arms.get("baseline", [])}
    diffs = []
    for r in dn:
        b = base.get((r["id"], r["name"]))
        if b is None:
            continue
        diffs.append(abs(r["p_a"] - b["p_a"]) + abs(r["p_b"] - b["p_b"]))
    if a770:
        return a770
    if not diffs:
        return {"pass": False, "reason": "no do-nothing rows paired with a baseline"}
    worst = max(diffs)
    return {"pass": worst == 0.0, "max_abs_diff": worst, "n": len(diffs), "mode": "exact"}


def gate_g1_a770(plain_probs, dn_probs):
    """plain_probs: 5 (p_a, p_b) tuples; dn_probs: 5 (p_a, p_b) tuples. All numbers are printed."""
    plain_diffs = [abs(plain_probs[i][0] - plain_probs[0][0]) + abs(plain_probs[i][1] - plain_probs[0][1])
                   for i in range(1, len(plain_probs))]
    bit_identical = all(d == 0.0 for d in plain_diffs)
    dn_diffs = [abs(dn_probs[i][0] - plain_probs[0][0]) + abs(dn_probs[i][1] - plain_probs[0][1])
                for i in range(len(dn_probs))]
    worst_plain = max(plain_diffs) if plain_diffs else 0.0
    out = {"mode": "a770", "plain_diffs": plain_diffs, "do_nothing_diffs": dn_diffs,
           "worst_plain": worst_plain, "worst_do_nothing": max(dn_diffs) if dn_diffs else None,
           "bit_identical_plain": bit_identical}
    if bit_identical:
        out["pass"] = all(d == 0.0 for d in dn_diffs)
    else:
        out["pass"] = all(d <= 2 * worst_plain for d in dn_diffs)
    out["rule"] = ("plain runs bit-identical: exact (0); otherwise every do-nothing difference from the first plain "
                   "run at most twice the largest plain-against-plain difference")
    return out


def gate_g2(t):
    return {"pass": t["mean_den"] is not None and t["mean_den"] >= 0.10, "gap": t["mean_den"],
            "n_pairs": t.get("n_pairs"),
            "rule": "mean(P_you - P_third) >= 0.10 over held-out SAY questions"}


def gate_g3(t_say):
    return {"pass": t_say["T"] is not None and t_say["T"] >= 0.5, "T_SAY": t_say["T"],
            "rule": "T_SAY (ratio of means) >= 0.5"}


def gate_g4(ov):
    return {"pass": ov["retest_say"] >= 20 and ov["retest_do"] >= 20, "retest_say": ov["retest_say"],
            "retest_do": ov["retest_do"], "rule": "min(Retest_SAY, Retest_DO) >= 20 of 80, each ranking separately"}


def random_set_rule(t, random_fams, min_families=2):
    """A transfer passes when it exceeds every set in each fillable family, with at least two families fillable."""
    fillable = [name for name, fam in random_fams["families"].items() if fam.get("sets")]
    if not random_fams.get("fillable", False) or len(fillable) < min_families:
        return {"read": False, "fillable": fillable, "reason": "fewer than two fillable families"}
    per = {}
    for name in fillable:
        vals = [x["T"] for x in t["random"][name] if x["T"] is not None]
        per[name] = {"max_T": max(vals) if vals else None, "n": len(vals),
                     "exceeds_all": (t["T"] is not None and bool(vals) and t["T"] > max(vals))}
    ok = t["T"] is not None and all(v["exceeds_all"] for v in per.values())
    return {"read": True, "pass": ok, "fillable": fillable, "per_family": per}


# ---------------------------------------------------------------- one stage

def score_stage(stage, sd, cfg, seed):
    import numpy as np
    est, keys, prompts = sd["est"], sd["keys"], sd["prompts"]
    arms = by_arm(sd["records"])
    # the held-out questions are the ones the runner made You runs for (SAY 5-8, Design, Split)
    held = sorted({r["id"] for r in arms.get("you", [])})
    names = sorted({r["name"] for r in arms.get("baseline", [])})
    sets = ranked_sets(est, prompts, keys)
    ov = overlap_stats(sets)
    lo, hi, draws = bootstrap_o(est, prompts, keys, cfg["bootstrap"]["resamples"], seed)
    t = ratio_of_means(arms, held, names, "do_copy", "baseline")
    t_say = ratio_of_means(arms, held, names, "say_copy", "baseline")
    t_swap = ratio_of_means(arms, held, names, "swap", "baseline")
    rev = ratio_of_means(arms, held, names, "reverse", "baseline")
    # random families: T per set
    t["random"] = {}
    for fam_name, fam in sd["sets"]["random"].get("families", {}).items():
        rows = []
        for si in range(len(fam["sets"])):
            rows.append(ratio_of_means(arms, held, names, f"random:{fam_name}:{si}", "baseline"))
        t["random"][fam_name] = rows
    gates = {"G0": gate_g0(arms, held), "G1": gate_g1(arms), "G2": gate_g2(t), "G3": gate_g3(t_say),
             "G4": gate_g4(ov)}
    rsr = random_set_rule(t, sd["sets"]["random"])
    read = {"O": gates["G0"]["pass"] and gates["G1"]["pass"] and gates["G4"]["pass"],
            "T": gates["G0"]["pass"] and gates["G1"]["pass"] and gates["G2"]["pass"] and gates["G3"]["pass"]}
    read["copy_arms"] = read["T"]
    read["random_set_rule"] = read["T"] and rsr.get("read", False)
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
        "T": t["T"], "T_SAY": t_say["T"], "T_swap": t_swap["T"], "reverse": rev["T"],
        "T_detail": {"T": {k: v for k, v in t.items() if k != "random"}, "T_SAY": t_say, "T_swap": t_swap,
                     "reverse": rev},
        "T_personally": secondary_t(arms),
        "off_target": off_target_shift(arms),
        "polarity": {arm: polarity_split(arms, held, names, arm) for arm in ("do_copy", "say_copy", "reverse", "swap")},
        "broken": broken_counts(arms),
        "random_families": {name: {"fillable": sd["sets"]["random"].get("fillable", False),
                                   "relaxed": fam["relaxed"], "relaxed_detail": fam["relaxed_detail"],
                                   "distinct_sets": len({tuple(sorted(s)) for s in fam["sets"]}),
                                   "total_effect": [sum(sd["sets"]["effect"].get(k, 0.0) for k in s) for s in fam["sets"]],
                                   "T": [x["T"] for x in t["random"][name]]}
                            for name, fam in sd["sets"]["random"].get("families", {}).items()},
        "not_fillable": sd["sets"]["random"].get("not_fillable_reason"),
        "do_set_total_effect": sum(sd["sets"]["effect"].get(k, 0.0) for k in sets["DO"]),
        "gates": gates,
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
        if o_read(a) and o_read(b):
            pairs.append((a, b, st[b]["overlap"]["O"] - st[a]["overlap"]["O"]))
    need = "midtrain-end" in olmo and "sft" in olmo and o_read("midtrain-end") and o_read("sft")
    if not need or len(pairs) < 3:
        out["205.3"] = {"verdict": "UNTESTABLE", "pairs_read": [(a, b, round(r, 4)) for a, b, r in pairs]}
    else:
        best = max(pairs, key=lambda x: x[2])
        rise = next(r for a, b, r in pairs if (a, b) == ("midtrain-end", "sft"))
        out["205.3"] = {"verdict": "held" if best[:2] == ("midtrain-end", "sft") and rise >= 0.2 else "failed",
                        "pairs_read": [(a, b, round(r, 4)) for a, b, r in pairs], "largest": best[:2], "rise": rise}
    # 205.4: at OLMo Instruct, T >= 0.5 and exceeds every set in all three random families.
    if not t_read(n):
        out["205.4"] = {"verdict": "not read", "gate": _first_failed(st[n], ("G0", "G1", "G2", "G3"))
                        if n in st else "stage absent"}
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
        gap = st[n5]["T_detail"]["T"]["mean_den"]
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
        out["205.6"] = {"verdict": "held" if (r["mean_num"] is not None and r["mean_num"] <= -0.5 * r["mean_den"])
                        else "failed", "mean_num": r["mean_num"], "mean_den": r["mean_den"]}
    # 205.7: wherever DO transfer is read and passes the random-set rule, off-target shift < 0.05.
    rows = []
    for n2 in order:
        if n2 in st and st[n2]["read"]["random_set_rule"] and st[n2]["random_set_rule"].get("pass"):
            rows.append((n2, st[n2]["off_target"]["shift"]))
    if not rows:
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


def _absent(st, n, gates=("G0", "G1", "G4")):
    """A stage that has not run is not read; say so rather than raising."""
    return n not in st or any(not st[n]["gates"][g]["pass"] for g in gates)


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
        rise = st[b]["overlap"]["O"] - st[a]["overlap"]["O"]
        lo, hi = paired_bootstrap_rise(st[a]["_est"], st[b]["_est"], st[a]["_prompts"], st[a]["_keys"],
                                       cfg["bootstrap"]["resamples"], cfg["seeds"]["bootstrap"] + order.index(b))
        if rise >= 0.2 and lo is not None and lo > 0 and st[b]["overlap"]["O"] >= 0.5:
            qual.append({"stage": b, "after": a, "rise": rise, "ci": [lo, hi]})
    if qual:
        return {"headline": "say/do sharing appears at " + ", ".join(q["stage"] for q in qual), "stages": qual}
    return {"headline": "no stage creates sharing that this design can resolve", "stages": []}


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default=str(HERE.parent / "results" / "part205"))
    ap.add_argument("--out", default=None, help="results.json path (default <raw>/results.json)")
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--only", default=None, help="score only these stage names (comma-separated; the rest are absent)")
    a = ap.parse_args(argv)
    cfg = C.load_config(a.config)
    if a.only:
        keep = set(a.only.split(","))
        cfg["stages"] = [s for s in cfg["stages"] if s["stage"] in keep]
    raw = Path(a.raw)
    stages = []
    for s in cfg["stages"]:
        sd = load_stage_dir(raw / s["stage"])
        if sd is None:
            stages.append(None)
            continue
        stages.append(score_stage(s["stage"], sd, cfg, cfg["seeds"]["bootstrap"]))
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
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    for s in res["stages"]:
        print(f"{s['stage']:14s} O={s['overlap']['O']} T={s['T']} gates="
              f"{ {g: v['pass'] for g, v in s['gates'].items()} } read={s['read']}")
    for k, v in res["predictions"].items():
        print(f"{k}: {v['verdict']}")
    print("headline:", res["withdrawal"]["headline"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
