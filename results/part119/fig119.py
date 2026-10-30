#!/usr/bin/env python3
"""Recompute Part 119's writeup numbers from raw result files and draw fig119.png."""
from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "part119"
sys.path.insert(0, str(ROOT))
from part119 import rows119
from part86.triage86 import WORDS, triage_answer


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def common_prefix(a: list[int], b: list[int], limit: int = 24) -> int:
    n = 0
    for x, y in zip(a[:limit], b[:limit]):
        if int(x) != int(y):
            break
        n += 1
    return n


def row_units(row_id: str) -> tuple[str, str] | None:
    if row_id.startswith("pair:"):
        bits = row_id[5:].split("+", 1)
    elif row_id.startswith("control:"):
        bits = row_id[8:].split("+R", 1)
    else:
        return None
    return (bits[0], bits[1]) if len(bits) == 2 else None


def bootstrap_draws(unit_ids: list[str]) -> list[Counter]:
    rng = random.Random(11902)
    return [Counter(rng.choice(unit_ids) for _ in unit_ids) for _ in range(2000)]


def hit_rates(rows: list[dict]) -> list[tuple[dict, dict]]:
    result = []
    for row in rows:
        positions = row.get("contested", [])
        if not positions:
            continue
        additive = sum(int(p["add"]) == int(p["g"]) for p in positions) / len(positions)
        winner = 0
        for p in positions:
            chosen = p["a"] if p["Ma"] > p["Mb"] else p["b"] if p["Mb"] > p["Ma"] else None
            winner += int(chosen is not None and int(chosen) == int(p["g"]))
        result.append((row, {"additive": additive, "winner": winner / len(positions)}))
    return result


def row_weight(row: dict, draw: Counter, weighting: str = "product") -> int:
    pair = row_units(str(row["row_id"]))
    if pair is None:
        return 0
    if weighting == "sum":
        return draw[pair[0]] + draw[pair[1]]
    return draw[pair[0]] * draw[pair[1]]


def rate_summary(rows: list[dict], draws: list[Counter]) -> dict:
    data = hit_rates(rows)
    means = {field: float(np.mean([rates[field] for _, rates in data])) for field in ("additive", "winner")}
    additive_hits = margin_hits = 0
    contested_total = 0
    for row in rows:
        for position in row.get("contested", []):
            contested_total += 1
            additive_hits += int(int(position["add"]) == int(position["g"]))
            chosen = position["a"] if position["Ma"] > position["Mb"] else position["b"] if position["Mb"] > position["Ma"] else None
            margin_hits += int(chosen is not None and int(chosen) == int(position["g"]))
    sims = {field: [] for field in ("additive", "winner", "difference")}
    sum_sims = {field: [] for field in ("additive", "winner", "difference")}
    for draw in draws:
        for weighting, target in (("product", sims), ("sum", sum_sims)):
            weighted = {}
            for field in ("additive", "winner"):
                num = den = 0.0
                for row, rates in data:
                    w = row_weight(row, draw, weighting)
                    num += rates[field] * w
                    den += w
                weighted[field] = num / den if den else None
                if weighted[field] is not None:
                    target[field].append(weighted[field])
            if None not in weighted.values():
                target["difference"].append(weighted["additive"] - weighted["winner"])
    return {
        "rows": len(rows), "rows_with_contested": len(data),
        "contested_positions": sum(len(r.get("contested", [])) for r in rows),
        "pooled_additive_hits": additive_hits, "pooled_margin_hits": margin_hits,
        "pooled_additive_rate": additive_hits / contested_total if contested_total else None,
        "pooled_margin_rate": margin_hits / contested_total if contested_total else None,
        "additive_mean": means["additive"], "winner_mean": means["winner"],
        "additive_ci95": [percentile(sims["additive"], .025), percentile(sims["additive"], .975)],
        "winner_ci95": [percentile(sims["winner"], .025), percentile(sims["winner"], .975)],
        "difference_mean": means["additive"] - means["winner"],
        "difference_ci95": [percentile(sims["difference"], .025), percentile(sims["difference"], .975)],
        "sum_weighted_ci95": {
            "additive": [percentile(sum_sims["additive"], .025), percentile(sum_sims["additive"], .975)],
            "winner": [percentile(sum_sims["winner"], .025), percentile(sum_sims["winner"], .975)],
            "difference": [percentile(sum_sims["difference"], .025), percentile(sum_sims["difference"], .975)],
        },
    }


def interaction_and_match(interactions: list[dict], unit_ids: list[str], draws: list[Counter]) -> tuple[dict, dict, dict]:
    by_kind = {}
    row_shares = {}
    for kind in ("real", "control"):
        subset = [r for r in interactions if r["kind"] == kind]
        shares = []
        tv_rows = []
        for row in subset:
            vals = [(float(p["nI"]) / float(p["nPair"])) ** 2
                    for p in row.get("positions", []) if float(p.get("nPair", 0)) > 0]
            if vals:
                shares.append(float(np.median(vals)))
            tvs = [float(p["tv"]) for p in row.get("positions", []) if p.get("tv") is not None]
            if tvs:
                tv_rows.append(float(np.median(tvs)))
        by_kind[kind] = {
            "rows": len(subset), "rows_with_positions": sum(bool(r.get("positions")) for r in subset),
            "positions": sum(len(r.get("positions", [])) for r in subset),
            "median_row_share": float(np.median(shares)),
            "median_row_total_variation": float(np.median(tv_rows)),
        }
        row_shares[kind] = shares

    controls, control_d2 = [], []
    for row in interactions:
        if row["kind"] != "control":
            continue
        for p in row.get("positions", []):
            ni, na, n2 = (float(p[k]) for k in ("nI", "nA", "n2"))
            if n2 > 0:
                control_d2.append(math.log(n2))
            if min(ni, na, n2) > 0:
                controls.append((math.log(na), math.log(n2), math.log(ni)))
    design = np.asarray([[1.0, a, b] for a, b, _ in controls])
    target = np.asarray([i for _, _, i in controls])
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    low, high = np.quantile(control_d2, [.01, .99])
    real_rows = [r for r in interactions if r["kind"] == "real"]
    kept_total = 0
    outside_support = 0
    nonpositive = 0
    residual_rows = []
    for row in real_rows:
        residuals = []
        for p in row.get("positions", []):
            ni, na, n2 = (float(p[k]) for k in ("nI", "nA", "n2"))
            if n2 <= 0:
                nonpositive += 1
                continue
            if not low <= math.log(n2) <= high:
                outside_support += 1
                continue
            if ni > 0 and na > 0:
                residuals.append(math.log(ni) - float(coefficients @ np.asarray([1.0, math.log(na), math.log(n2)])))
            else:
                nonpositive += 1
        if residuals:
            kept_total += len(residuals)
            residual_rows.append({"first": row["first"], "second": row["second"], "value": float(np.median(residuals))})
    mean = float(np.mean([r["value"] for r in residual_rows]))
    samples = {"product": [], "sum": []}
    for draw in draws:
        for weighting in samples:
            num = den = 0.0
            for row in residual_rows:
                if weighting == "sum":
                    weight = draw[row["first"]] + draw[row["second"]]
                else:
                    weight = draw[row["first"]] * draw[row["second"]]
                num += row["value"] * weight
                den += weight
            if den:
                samples[weighting].append(num / den)
    ci = [percentile(samples["product"], .025), percentile(samples["product"], .975)]
    sum_ci = [percentile(samples["sum"], .025), percentile(samples["sum"], .975)]
    matched = {
        "control_positions_fit": len(controls), "real_positions_kept": kept_total,
        "real_positions_total": sum(len(r.get("positions", [])) for r in real_rows),
        "share_kept": kept_total / sum(len(r.get("positions", [])) for r in real_rows),
        "rows_with_residual": len(residual_rows),
        "excluded_real_positions": sum(len(r.get("positions", [])) for r in real_rows) - kept_total,
        "excluded_outside_control_D2_support": outside_support,
        "excluded_nonpositive_norms": nonpositive,
        "mean_residual": mean, "mean_residual_ci95": ci,
        "mean_residual_ci95_sum_weighted": sum_ci, "ratio": math.exp(mean),
        "ratio_ci95": [math.exp(ci[0]), math.exp(ci[1])], "fit_rank": int(rank),
        "ratio_ci95_sum_weighted": [math.exp(sum_ci[0]), math.exp(sum_ci[1])],
    }
    return by_kind, matched, row_shares


def kind_tables(units: list[dict], logits: list[dict], interactions: list[dict]) -> dict:
    by_id = {u["id"]: u for u in units}
    meta = {r["row_id"]: r for r in rows119.all_rows(units)}
    logit_by_id = {r["row_id"]: r for r in logits}
    interactions_by_scheme = {scheme: defaultdict(lambda: defaultdict(list)) for scheme in ("agreed", "haiku")}
    hit_by_scheme = {scheme: defaultdict(list) for scheme in ("agreed", "haiku")}
    for row in logits:
        if not row["row_id"].startswith("pair:"):
            continue
        row_meta = meta[row["row_id"]]
        for scheme in ("agreed", "haiku"):
            cls = rows119.pair_class(row_meta, by_id, scheme)
            if row.get("contested"):
                p = row["contested"]
                add = sum(int(x["add"]) == int(x["g"]) for x in p) / len(p)
                win = sum(int((x["a"] if x["Ma"] > x["Mb"] else x["b"] if x["Mb"] > x["Ma"] else -1)) == int(x["g"]) for x in p) / len(p)
                hit_by_scheme[scheme][cls].append((add, win))
    for row in interactions:
        proxy = {"row_type": "pair", "unit_ids": [row["first"], row["second"]]}
        for scheme in ("agreed", "haiku"):
            cls = rows119.pair_class(proxy, by_id, scheme)
            values = [(float(p["nI"]) / float(p["nPair"])) ** 2
                      for p in row.get("positions", []) if float(p.get("nPair", 0)) > 0]
            if values:
                interactions_by_scheme[scheme][row["kind"]][cls].append(float(np.median(values)))
    result = {}
    for scheme, names in (("agreed", ("slot_x_slot", "concept_x_concept", "slot_x_concept", "involving_disputed")),
                          ("haiku", ("slot_x_slot", "concept_x_concept", "slot_x_concept"))):
        groups = {}
        for name in names:
            expected = sum(rows119.pair_class(m, by_id, scheme) == name for m in meta.values() if m["row_type"] == "pair")
            rates = hit_by_scheme[scheme][name]
            groups[name] = {
                "expected_pairs": expected, "contested_pairs": len(rates),
                "additive": float(np.mean([x[0] for x in rates])) if rates else None,
                "winner": float(np.mean([x[1] for x in rates])) if rates else None,
                "path2_rows": 2 * expected,
                "real_interaction_share": float(np.median(interactions_by_scheme[scheme]["real"][name])),
                "control_interaction_share": float(np.median(interactions_by_scheme[scheme]["control"][name])),
            }
        result[scheme] = groups
    return result


def kind_winners(units: list[dict], logits: list[dict], draws: list[Counter]) -> dict:
    by_id = {u["id"]: u for u in units}
    meta = {r["row_id"]: r for r in rows119.all_rows(units)}
    result = {}
    agreed_slot_concept = []
    for scheme in ("agreed", "haiku"):
        total = decided = wins = observed = 0
        for row in logits:
            if not row["row_id"].startswith("pair:"):
                continue
            if rows119.pair_class(meta[row["row_id"]], by_id, scheme) != "slot_x_concept":
                continue
            total += 1
            a, b = row_units(row["row_id"])
            aw = sum(int(p["g"]) == int(p["a"]) for p in row.get("contested", []))
            bw = sum(int(p["g"]) == int(p["b"]) for p in row.get("contested", []))
            if scheme == "agreed":
                winner_id = a if aw > bw else b if bw > aw else None
                agreed_slot_concept.append({
                    "first": a, "second": b,
                    "concept_win": winner_id is not None and by_id[winner_id]["group"] == "agreed_concept",
                    "decided": winner_id is not None,
                })
            if aw == bw:
                continue
            observed += 1
            decided += 1
            winner_id = a if aw > bw else b
            is_concept = ((by_id[winner_id]["group"] == "agreed_concept") if scheme == "agreed"
                          else by_id[winner_id]["haiku_kind"] == "concept")
            wins += int(is_concept)
        result[scheme] = {"expected": total, "observed": total, "decided": decided,
                          "concept_wins": wins, "share": wins / decided}
    concept_ids = [u["id"] for u in units if u["group"] == "agreed_concept"]

    def count_winners(excluded: str | None = None) -> dict:
        chosen = [r for r in agreed_slot_concept
                  if excluded is None or excluded not in (r["first"], r["second"])]
        decided = sum(r["decided"] for r in chosen)
        wins = sum(r["concept_win"] for r in chosen)
        return {"concept_wins": wins, "decided_pairs": decided,
                "expected_pairs": len(chosen), "observed_pairs": len(chosen),
                "concept_win_share": wins / decided if decided else None}

    product_sims, sum_sims = [], []
    for draw in draws:
        for weighting, target in (("product", product_sims), ("sum", sum_sims)):
            wins = decided = 0.0
            for row in agreed_slot_concept:
                weight = (draw[row["first"]] + draw[row["second"]]
                          if weighting == "sum"
                          else draw[row["first"]] * draw[row["second"]])
                if row["decided"]:
                    decided += weight
                    wins += int(row["concept_win"]) * weight
            if decided:
                target.append(wins / decided)
    result["agreed"]["unit_bootstrap_ci95"] = [percentile(product_sims, .025), percentile(product_sims, .975)]
    result["agreed"]["unit_bootstrap_ci95_sum_weighted"] = [percentile(sum_sims, .025), percentile(sum_sims, .975)]
    result["agreed"]["unit_bootstrap_draws_below_0_60"] = sum(value < .60 for value in product_sims)
    result["agreed"]["unit_bootstrap_draws"] = len(product_sims)
    result["agreed"]["leave_one_concept_out"] = {
        unit_id: count_winners(unit_id) for unit_id in concept_ids
    }
    return result


def unit_summaries(units: list[dict], answers: list[dict], ladder: list[dict], logits: list[dict]) -> tuple[dict, dict]:
    by_answer = {r["row_id"]: r for r in answers}
    by_ladder = {r["id"]: r for r in ladder}
    plain = [int(t) for t in by_answer["plain"]["tokens"]]
    metrics = {}
    for u in units:
        uid = u["id"]
        single = [int(t) for t in by_answer[f"single:{uid}"]["tokens"]]
        p = next((i for i, (a, b) in enumerate(zip(single[:24], plain[:24])) if a != b), None)
        t_a = single[p] if p is not None else None
        ld = by_ladder[uid]
        alpha = next((float(a) for a, top in zip(ld["alphas"], ld["tops"]) if t_a is not None and top == t_a), None)
        vals = []
        if p is not None:
            for other in units:
                if other["id"] == uid:
                    continue
                control = by_answer[f"control:{uid}+R{other['id']}"]
                vals.append(max(0, common_prefix([int(t) for t in control["tokens"]], single) - p) / (24 - p))
        metrics[uid] = {"R": float(np.mean(vals)) if vals else None, "flip_alpha": alpha}

    groups = {}
    for scheme in ("agreed", "haiku"):
        groups[scheme] = {}
        for name in (("agreed_slot", "agreed_concept", "disputed") if scheme == "agreed" else ("slot", "concept")):
            vals = [metrics[u["id"]]["R"] if name != "disputed" else metrics[u["id"]]["R"]
                    for u in units if (u["group"] == name if scheme == "agreed" else u["haiku_kind"] == name)
                    and metrics[u["id"]]["R"] is not None]
            alphas = [metrics[u["id"]]["flip_alpha"] for u in units
                      if (u["group"] == name if scheme == "agreed" else u["haiku_kind"] == name)
                      and metrics[u["id"]]["flip_alpha"] is not None]
            groups[scheme][name] = {"n": len(vals), "R_median": float(np.median(vals)),
                                    "flip_median": float(np.median(alphas))}
    c = [metrics[u["id"]]["R"] for u in units if u["group"] == "agreed_concept"]
    s = [metrics[u["id"]]["R"] for u in units if u["group"] == "agreed_slot"]
    cf = [metrics[u["id"]]["flip_alpha"] for u in units if u["group"] == "agreed_concept"]
    sf = [metrics[u["id"]]["flip_alpha"] for u in units if u["group"] == "agreed_slot"]
    kind_stats = {
        "R_difference": float(np.median(c) - np.median(s)),
        "R_p_one_sided": float(mannwhitneyu(c, s, alternative="greater").pvalue),
        "flip_difference": float(np.median(sf) - np.median(cf)),
        "flip_p_one_sided": float(mannwhitneyu(cf, sf, alternative="less").pvalue),
    }
    all_r = [metrics[u["id"]]["R"] for u in units if metrics[u["id"]]["R"] is not None]
    kind_stats["R_min"] = float(min(all_r))
    kind_stats["R_max"] = float(max(all_r))
    disc = [metrics[u["id"]]["R"] for u in units if u["group"] == "disputed"]
    dm, sm, cm = np.median(disc), np.median(s), np.median(c)
    kind_stats["disputed_closer_to_slot_margin"] = float(abs(dm - cm) - abs(dm - sm))
    hs = [metrics[u["id"]]["R"] for u in units if u["haiku_kind"] == "slot"]
    hc = [metrics[u["id"]]["R"] for u in units if u["haiku_kind"] == "concept"]
    kind_stats["haiku_R_difference"] = float(np.median(hc) - np.median(hs))
    kind_stats["haiku_disputed_margin"] = float(abs(dm - np.median(hc)) - abs(dm - np.median(hs)))

    # Pair tournament dominance from the observed winner of each unordered real pair.
    wins, decided = Counter(), Counter()
    for row in logits:
        if not row["row_id"].startswith("pair:"):
            continue
        a, b = row_units(row["row_id"])
        aw = sum(int(p["g"]) == int(p["a"]) for p in row.get("contested", []))
        bw = sum(int(p["g"]) == int(p["b"]) for p in row.get("contested", []))
        if aw == bw:
            continue
        winner, loser = (a, b) if aw > bw else (b, a)
        wins[winner] += 1
        decided[winner] += 1
        decided[loser] += 1
    dominance, alphas = [], []
    for u in units:
        uid = u["id"]
        if decided[uid] and metrics[uid]["flip_alpha"] is not None:
            dominance.append(wins[uid] / decided[uid])
            alphas.append(metrics[uid]["flip_alpha"])
    sp = spearmanr(dominance, alphas)
    kind_stats["dominance_flip_spearman"] = float(sp.statistic)
    kind_stats["dominance_flip_p_two_sided"] = float(sp.pvalue)
    kind_stats["correlation_units"] = len(alphas)
    return groups, kind_stats


def gates(answers: list[dict], logits: list[dict], gate2b: list[dict], gate2c: list[dict], part118: list[dict], units: list[dict]) -> dict:
    answer_by_id = {r["row_id"]: r for r in answers}
    part118_map = {r["id"]: r for r in part118 if r.get("arm") == "neuron" and float(r.get("alpha", -1)) == .8 and (r.get("question_index") == 0 or r.get("question") == "Who are you?")}
    p118_matches = sum(answer_by_id[f"single:{u['id']}"]["text"][:100] == part118_map[u["id"]]["answer"][:100] for u in units)
    g2b_map = {r["id"]: r["answer"] for r in gate2b}
    g2b_matches = sum(answer_by_id[f"single:{u['id']}"]["text"][:100] == g2b_map[u["id"]][:100] for u in units)
    g2c_matches = sum(r["part92_answer"].strip() == r["row_hook_answer"].strip() for r in gate2c)
    agreed_concepts = [u for u in units if u["group"] == "agreed_concept"]
    g2c_by_id = {r["id"]: r for r in gate2c}
    g2c_concept_matches = sum(
        g2c_by_id[u["id"]]["part92_answer"].strip() == g2c_by_id[u["id"]]["row_hook_answer"].strip()
        for u in agreed_concepts
    )
    all_positions = sum(int(r.get("n_positions", 0)) for r in logits)
    all_hits = sum(int(r.get("both_hits", 0)) for r in logits)
    real_contested = [p for r in logits if r["row_id"].startswith("pair:") for p in r.get("contested", [])]
    real_hits = sum(int(p["both_top"]) == int(p["g"]) for p in real_contested)
    control_positions = sum(int(r.get("n_positions", 0)) for r in logits if r["row_id"].startswith("control:"))
    control_hits = sum(int(r.get("both_hits", 0)) for r in logits if r["row_id"].startswith("control:"))
    return {"gate2": {"part118": [p118_matches, len(units)], "gate2b": [g2b_matches, len(units)],
                       "gate2c": [g2c_matches, len(gate2c)],
                       "gate2c_agreed_concepts": [g2c_concept_matches, len(agreed_concepts)]},
            "gate3_all": [all_hits, all_positions, all_hits / all_positions],
            "gate3_real_contested": [real_hits, len(real_contested), real_hits / len(real_contested)],
            "gate3_control": [control_hits, control_positions, control_hits / control_positions]}


def broken_counts(answers: list[dict]) -> dict:
    meta = {r["row_id"]: r["row_type"] for r in rows119.all_rows()}
    groups = defaultdict(lambda: [0, 0, 0])
    for row in answers:
        kind = meta.get(row["row_id"], "unknown")
        key = {"single": "single", "random": "random", "pair": "real_pair", "control": "control"}.get(kind, kind)
        words = [word.casefold() for word in WORDS.findall(row["text"])]
        repeated_three_word = any(
            n >= 3 for n in Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2)).values()
        )
        groups[key][0] += int(triage_answer(row["text"]) == "broken")
        groups[key][1] += 1
        groups[key][2] += int(repeated_three_word)
    result = {k: v[:2] for k, v in sorted(groups.items())}
    result["three_word_repeat_only"] = {k: [v[2], v[1]] for k, v in sorted(groups.items())}
    return result


def reader_summary() -> dict:
    ds = read_jsonl(RESULTS / "labels-ds.jsonl")
    luna = read_jsonl(RESULTS / "labels-luna.jsonl")
    known_ds_rows = read_jsonl(RESULTS / "known-ds.jsonl")
    known_luna_rows = read_jsonl(RESULTS / "known-luna.jsonl")
    keys = {r["item_id"]: r for r in read_jsonl(RESULTS / "labels-ds.jsonl.keys.jsonl")}
    ds_by_id = {r["item_id"]: r for r in ds}
    both_by_class = Counter()
    class_n = Counter()
    coherent_by_class = Counter()
    coherent_n = Counter()
    for item_id, label in ds_by_id.items():
        key = keys[item_id]
        class_n[key["sample_class"]] += 1
        both_by_class[key["sample_class"]] += int(label.get("x") is True and label.get("y") is True)
        if type(label.get("coherent")) is bool:
            coherent_n[key["sample_class"]] += 1
            coherent_by_class[key["sample_class"]] += int(label["coherent"])
    reader_leave_one_concept_out = {}
    concept_ids = [u["id"] for u in rows119.load_units() if u["group"] == "agreed_concept"]
    concept_sample = [key for key in keys.values() if key["sample_class"] == "concept_x_concept"]
    for unit_id in concept_ids:
        remaining = [key for key in concept_sample
                     if unit_id not in (key["x_unit_id"], key["y_unit_id"])]
        valid = [key for key in remaining
                 if all(type(ds_by_id.get(key["item_id"], {}).get(field)) is bool
                        for field in ("x", "y", "coherent"))]
        both = sum(bool(ds_by_id[key["item_id"]]["x"] and ds_by_id[key["item_id"]]["y"])
                   for key in valid)
        reader_leave_one_concept_out[unit_id] = {
            "both": both, "labeled": len(valid), "expected": len(remaining),
            "errors": sum("error" in ds_by_id.get(key["item_id"], {}) for key in remaining),
        }
    luna_map = {r["item_id"]: r for r in luna}
    shared = sorted(set(ds_by_id) & set(luna_map))
    ds_both = [bool(ds_by_id[i].get("x") and ds_by_id[i].get("y")) for i in shared]
    luna_both = [bool(luna_map[i].get("x") and luna_map[i].get("y")) for i in shared]
    observed = sum(a == b for a, b in zip(ds_both, luna_both)) / len(shared)
    p1, p2 = sum(ds_both) / len(shared), sum(luna_both) / len(shared)
    expected = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    known = read_jsonl(ROOT / "part119" / "known119.jsonl")
    known_scores = {}
    for name, raw_labels in (("known-ds.jsonl", known_ds_rows), ("known-luna.jsonl", known_luna_rows)):
        labels = {r["item_id"]: r for r in raw_labels}
        correct = sum(
            all(type(labels.get(f"known-{i:02d}", {}).get(field)) is bool
                and labels[f"known-{i:02d}"][field] is item[f"expected_{field}"]
                for field in ("x", "y", "coherent"))
            for i, item in enumerate(known, 1)
        )
        known_scores[name] = [correct, len(known)]
    return {"ds_both": sum(both_by_class.values()), "ds_items": len(ds),
            "both_by_class": {k: [both_by_class[k], class_n[k]] for k in sorted(class_n)},
            "coherent_by_class": {k: [coherent_by_class[k], coherent_n[k]] for k in sorted(coherent_n)},
            "luna_both_on_shared": sum(luna_both), "shared": len(shared), "kappa": kappa,
            "ds_errors": sum("error" in row for row in ds),
            "ds_retry_items": sum(n > 1 for n in Counter(str(row["item_id"]) for row in ds).values()),
            "luna_errors": sum("error" in row for row in luna),
            "luna_retry_items": sum(n > 1 for n in Counter(str(row["item_id"]) for row in luna).values()),
            "known_ds_errors": sum("error" in row for row in known_ds_rows),
            "known_luna_errors": sum("error" in row for row in known_luna_rows),
            "known_ds_retry_items": sum(n > 1 for n in Counter(str(row["item_id"]) for row in known_ds_rows).values()),
            "known_luna_retry_items": sum(n > 1 for n in Counter(str(row["item_id"]) for row in known_luna_rows).values()),
            "leave_one_concept_out": reader_leave_one_concept_out,
            "known_gates": known_scores}


def batch_and_pilot_checks(answers: list[dict], gate2c: list[dict]) -> dict:
    answer_by_id = {r["row_id"]: r for r in answers}
    batch_differences = [
        row["id"] for row in gate2c
        if answer_by_id.get(f"single:{row['id']}", {}).get("text", "").strip()
        != str(row.get("part92_answer", "")).strip()
    ]
    pilot128 = read_jsonl(RESULTS / "pilot-128-answers.jsonl")
    pilot_matches = [(row, answer_by_id.get(str(row["row_id"]))) for row in pilot128]
    missing = [row["row_id"] for row, current in pilot_matches if current is None]
    different = [row["row_id"] for row, current in pilot_matches
                 if current is not None and row.get("tokens") != current.get("tokens")]
    return {
        "main_batch64_vs_batch1_part92": {
            "different": len(batch_differences), "total": len(gate2c), "unit_ids": batch_differences,
        },
        "pilot128_vs_main": {
            "rows": len(pilot128), "missing_main_rows": missing,
            "token_identical": len(pilot128) - len(missing) - len(different),
            "different_token_rows": different,
        },
    }


def example_ids(units: list[dict]) -> dict:
    by_id = {u["id"]: u for u in units}
    rows = sorted([r for r in rows119.all_rows(units) if r["row_type"] == "pair"], key=lambda r: r["row_id"])
    random.Random(119).shuffle(rows)
    selected = {k: [] for k in ("slot_x_slot", "concept_x_concept", "slot_x_concept")}
    for row in rows:
        kind = rows119.pair_class(row, by_id, "agreed")
        if kind in selected and len(selected[kind]) < 3:
            selected[kind].append(row["row_id"])
    return selected


def compute() -> dict:
    units = rows119.load_units()
    answers = read_jsonl(RESULTS / "answers.jsonl")
    logits = read_jsonl(RESULTS / "logits.jsonl")
    interactions = read_jsonl(RESULTS / "interaction.jsonl")
    ladder = read_jsonl(RESULTS / "ladder.jsonl")
    draws = bootstrap_draws([u["id"] for u in units])
    pairs = [r for r in logits if r["row_id"].startswith("pair:")]
    controls = [r for r in logits if r["row_id"].startswith("control:")]
    pair_rates, control_rates = rate_summary(pairs, draws), rate_summary(controls, draws)
    logit_by_id = {r["row_id"]: r for r in logits}
    pilot_rows = rows119.pilot_rows(units)
    pilot_peek = {}
    for row_type in ("pair", "control"):
        selected = [r for r in pilot_rows if r["row_type"] == row_type]
        positions = [p for row in selected for p in logit_by_id[row["row_id"]].get("contested", [])]
        hits = sum(int(p["add"]) == int(p["g"]) for p in positions)
        pilot_peek[row_type] = {"rows": len(selected), "hits": hits, "positions": len(positions),
                                "rate": hits / len(positions)}
    int_summary, matched, _ = interaction_and_match(interactions, [u["id"] for u in units], draws)
    kinds = kind_tables(units, logits, interactions)
    winners = kind_winners(units, logits, draws)
    unit_groups, kind_stats = unit_summaries(units, answers, ladder, logits)
    gate2c = read_jsonl(RESULTS / "gate2c.jsonl")
    gates_summary = gates(answers, logits, read_jsonl(RESULTS / "gate2b.jsonl"),
                          gate2c,
                          read_jsonl(ROOT / "results" / "part118" / "answers.jsonl"), units)
    gate1 = read_json(RESULTS / "gate1.json")
    gate1_equal = gate1.get("no_hook_tokens") == gate1.get("zero_hook_tokens")
    return {
        "units": {"total": len(units), "agreed_slot": sum(u["group"] == "agreed_slot" for u in units),
                  "agreed_concept": sum(u["group"] == "agreed_concept" for u in units),
                  "disputed": [u["id"] for u in units if u["group"] == "disputed"],
                  "haiku_slot": sum(u["haiku_kind"] == "slot" for u in units),
                  "haiku_concept": sum(u["haiku_kind"] == "concept" for u in units)},
        "real_pairs": pair_rates, "controls": control_rates,
        "pilot_peek": pilot_peek,
        "gate3": gates_summary, "interactions": int_summary, "effect_matched": matched,
        "broken": broken_counts(answers), "kind_classes": kinds,
        "kind_winners": winners, "unit_groups": unit_groups, "kind_stats": kind_stats,
        "reader": reader_summary(), "examples": example_ids(units),
        "batch_and_pilot_checks": batch_and_pilot_checks(answers, gate2c),
        "gate1": {"token_for_token_equal": gate1_equal, "no_hook_tokens": len(gate1.get("no_hook_tokens", [])),
                  "zero_hook_tokens": len(gate1.get("zero_hook_tokens", []))},
        "answers_rows": len(answers),
        "bootstrap": {"draws": 2000, "seed": 11902},
        "seeds": {"reader_sample": 11900, "luna_sample": 11903, "pilot_peek": 11901,
                  "examples": 119, "bootstrap": 11902},
        "scipy_version": scipy.__version__,
    }


def draw(summary: dict) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, (left, middle, right) = plt.subplots(
        1, 3, figsize=(15.4, 5.0), gridspec_kw={"width_ratios": [1.15, 1.0, .9]}
    )
    fig.patch.set_facecolor("white")
    left.set_title("Additive and margin hit rates", loc="left", pad=12)
    left_rows = [
        ("Real · additive", "real_pairs", "additive_mean", "additive_ci95", "#286F8E", "o"),
        ("Real · margin", "real_pairs", "winner_mean", "winner_ci95", "#D18B2C", "s"),
        ("Control · additive (description)", "controls", "additive_mean", "additive_ci95", "#286F8E", "o"),
        ("Control · margin (description)", "controls", "winner_mean", "winner_ci95", "#D18B2C", "s"),
    ]
    y_values = [3.2, 2.5, 1.1, .4]
    for yi, (label, group, value_key, ci_key, color, marker) in zip(y_values, left_rows):
        result = summary[group]
        value = result[value_key] * 100
        lo, hi = [x * 100 for x in result[ci_key]]
        left.errorbar(value, yi, xerr=[[value - lo], [hi - value]], fmt=marker,
                      color=color, markerfacecolor=color if group == "real_pairs" else "white",
                      capsize=3, markersize=5, linewidth=1.4)
        left.text(64, yi, f"{value:.1f}%  ({lo:.1f}–{hi:.1f})", ha="left", va="center", fontsize=8)
    left.axvline(60, color="#666666", linewidth=1.2, linestyle="--")
    left.text(61, 4.02, "60% threshold", ha="left", va="bottom", fontsize=8)
    left.set_yticks(y_values, [r[0] for r in left_rows])
    left.set_xlim(0, 100)
    left.set_ylim(-.3, 4.2)
    left.set_xticks([0, 20, 40, 60, 80, 100], ["0%", "20%", "40%", "60%", "80%", "100%"])
    left.set_xlabel("Hit rate on contested positions")
    left.spines[["top", "right", "left"]].set_visible(False)
    left.spines["bottom"].set_color("#777777")
    left.tick_params(axis="x", length=0, pad=5)
    left.tick_params(axis="y", length=0)

    middle.set_title("Broken answers and reader coherence", loc="left", pad=12)
    broken = summary["broken"]
    reader = summary["reader"]["coherent_by_class"]
    pair_reader_n = sum(reader[k][1] for k in ("concept_x_concept", "slot_x_concept", "slot_x_slot"))
    pair_reader_coherent = sum(reader[k][0] for k in ("concept_x_concept", "slot_x_concept", "slot_x_slot"))
    plotted = [
        ("Real pairs", broken["real_pair"][0] / broken["real_pair"][1],
         pair_reader_coherent / pair_reader_n, f"n={pair_reader_n}"),
        ("Controls", broken["control"][0] / broken["control"][1],
         reader["control"][0] / reader["control"][1], "n=40"),
        ("Singles", broken["single"][0] / broken["single"][1], None, ""),
        ("Random singles", broken["random"][0] / broken["random"][1], None, ""),
    ]
    my, cy = [], []
    for idx, (label, broken_share, coherent_share, sample_n) in enumerate(plotted):
        base = 3.0 - idx
        my.append(base + .14)
        middle.barh(base + .14, broken_share * 100, height=.26, color="#C76A42", label="Triage broken" if idx == 0 else None)
        middle.text(broken_share * 100 + 1.5, base + .14,
                    f"{broken_share * 100:.1f}%", ha="left", va="center", fontsize=8)
        if coherent_share is not None:
            cy.append(base - .15)
            middle.barh(base - .15, coherent_share * 100, height=.26, color="#3A8F79", label="Reader coherent" if idx == 0 else None)
            middle.text(coherent_share * 100 + 1.5, base - .15,
                        f"{coherent_share * 100:.1f}% {sample_n}", ha="left", va="center", fontsize=8)
    middle.set_yticks([3, 2, 1, 0], [r[0] for r in plotted])
    middle.set_xlim(0, 100)
    middle.set_ylim(-.6, 3.6)
    middle.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    middle.set_xlabel("Share of answers")
    middle.legend(loc="lower right", frameon=False, fontsize=8)
    middle.spines[["top", "right", "left"]].set_visible(False)
    middle.spines["bottom"].set_color("#777777")
    middle.tick_params(axis="x", length=0, pad=5)
    middle.tick_params(axis="y", length=0)

    matched = summary["effect_matched"]
    center = matched["ratio"]
    lo, hi = matched["ratio_ci95"]
    right.set_title(f"Effect-matched ratio\n{center:.3f} (95% CI {lo:.3f}–{hi:.3f})", loc="left", pad=12)
    right.axvspan(.80, 1.25, color="#DDEFE8", zorder=0)
    right.axvline(.80, color="#3A8F79", linewidth=1.2, linestyle="--")
    right.axvline(1.25, color="#3A8F79", linewidth=1.2, linestyle="--")
    right.axvline(1.0, color="#777777", linewidth=1)
    right.errorbar(center, .1, xerr=[[center - lo], [hi - center]], fmt="o", color="#1f2933",
                   capsize=4, markersize=5, linewidth=1.5)
    right.set_xlim(.70, 1.35)
    right.set_ylim(-.55, 1.0)
    right.set_yticks([.1], ["Real / control"])
    right.set_xticks([.8, 1.0, 1.25], ["0.8×", "1.0×", "1.25×"])
    right.set_xlabel("Interaction ratio; dashed bounds at 0.80 and 1.25")
    right.spines[["top", "right", "left"]].set_visible(False)
    right.spines["bottom"].set_color("#777777")
    right.tick_params(axis="x", length=0, pad=5)
    fig.text(.01, .015, "Control Path 1 rates are descriptive only. Reader coherence is the DeepSeek-labeled sample.",
             ha="left", va="bottom", fontsize=8, color="#444444")
    fig.tight_layout(pad=1.2, w_pad=1.8, rect=(0, .05, 1, 1))
    fig.savefig(RESULTS / "fig119.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    result = compute()
    draw(result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
