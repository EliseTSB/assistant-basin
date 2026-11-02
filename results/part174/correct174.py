#!/usr/bin/env python3
"""Corrections to Part 174's descriptive counts, after its cold check.

The registered scorer's outputs stay as they ran. score.json and scored_directions.jsonl
store google_claim_count = 0 for a direction with no clean answers, so nothing was read
there. This script recomputes the counts that depend on that from the frozen files and
writes corrections.json. Nothing here changes a registered verdict.
"""
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "corrections.json"
REPS = 10000
SEED = 174


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


directions = read_jsonl(ROOT / "scored_directions.jsonl")
score = json.loads((ROOT / "score.json").read_text(encoding="utf-8"))
draws = score["draw_sign_contrasts"]


def is_labelled(row):
    return bool(row["reader_available"]) and row["google_claim_count"] is not None


def histogram():
    labelled = [row for row in directions if is_labelled(row)]
    out = {
        "directions": len(directions),
        "labelled": len(labelled),
        "no_clean_answers": sum(not row["reader_available"] for row in directions),
        "incomplete_reader_coverage": sum(row["google_claim_count"] is None for row in directions),
        "zero_claims": sum(row["google_claim_count"] == 0 for row in labelled),
        "between_5_and_15": sum(5 <= row["google_claim_count"] <= 15 for row in labelled),
        "by_family": {},
    }
    for family in ("u100d", "f100d"):
        rows = [row for row in directions if row["family"] == family]
        kept = [row for row in rows if is_labelled(row)]
        counts = Counter(int(row["google_claim_count"]) for row in kept)
        out["by_family"][family] = {
            "directions": len(rows),
            "labelled": len(kept),
            "no_clean_answers": sum(not row["reader_available"] for row in rows),
            "incomplete_reader_coverage": sum(row["google_claim_count"] is None for row in rows),
            "zero_claims": counts.get(0, 0),
            "between_5_and_15": sum(v for k, v in counts.items() if 5 <= k <= 15),
            "between_10_and_15": sum(v for k, v in counts.items() if 10 <= k <= 15),
            "above_15": sum(v for k, v in counts.items() if k > 15),
            "histogram": {str(k): counts[k] for k in sorted(counts)},
            "largest_bin": (max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0] if counts else None),
            "max_claims": max(counts) if counts else None,
        }
    return out


def floors():
    """Can the unlabelled answers move any direction across a modal-count floor?"""
    incomplete = [row for row in directions if row["google_claim_count"] is None]
    out = {"directions": len(incomplete), "rows": [], "can_change": {}}
    for row in incomplete:
        summary = row["summary"]
        modal_count = int(summary.get("modal_count") or 0)
        unlabelled = int(row["unlabelled_clean_answers"])
        out["rows"].append({"arm": row["arm"], "unlabelled": unlabelled, "modal_count": modal_count,
                            "broken": int(summary["broken"]), "supported": bool(row["supported"])})
    for floor in (5, 10, 15):
        movers = []
        for row in out["rows"]:
            within_broken = row["broken"] <= 8
            reachable = row["modal_count"] + row["unlabelled"] >= floor
            if within_broken and reachable and row["modal_count"] < floor:
                movers.append(row["arm"])
        out["can_change"][str(floor)] = movers
    return out


def spearman(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return None
    value = stats.spearmanr(xs, ys).statistic
    return float(value) if np.isfinite(value) else None


def interval(xs, ys, seed):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(REPS):
        pick = rng.integers(0, xs.size, xs.size)
        value = spearman(xs[pick], ys[pick])
        if value is not None:
            values.append(value)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def contrasts():
    by_arm = {(row["set"], row["dose"]): row for row in directions}
    state = {}
    for draw in draws:
        plus = by_arm[(draw["set"], 0.5)]
        minus = by_arm[(draw["set"], -0.5)]
        state[draw["set"]] = {"draw": draw, "plus": plus, "minus": minus,
                              "read": is_labelled(plus) and is_labelled(minus)}
    out = {"draws": len(draws), "by_family": {}}
    for family in ("u100d", "f100d"):
        rows = [item for item in state.values() if item["draw"]["family"] == family]
        kept = [item for item in rows if item["read"]]
        no_clean = sum(not (item["plus"]["reader_available"] and item["minus"]["reader_available"])
                       for item in rows)
        values = [item["plus"]["google_claim_count"] - item["minus"]["google_claim_count"] for item in kept]
        counts = Counter(values)
        out["by_family"][family] = {
            "draws": len(rows),
            "both_signs_read": len(kept),
            "no_D": len(rows) - len(kept),
            "a_sign_with_no_clean_answers": no_clean,
            "D_most_common": (counts.most_common(1)[0] if counts else None),
        }
    # u100d is the only family with enough read draws to describe.
    rows = [item for item in state.values() if item["draw"]["family"] == "u100d"]
    kept = [item for item in rows if item["read"]]
    for index, projection in enumerate(("cf", "cr")):
        xs = [item["draw"][projection] for item in kept]
        ys = [item["plus"]["google_claim_count"] - item["minus"]["google_claim_count"] for item in kept]
        complete = {"n": len(xs), "rho": spearman(xs, ys), "interval_95": interval(xs, ys, SEED + index)}
        # Every assignment of the unlabelled answers: each could be a Google claim or not.
        ranges = []
        for item in rows:
            options = []
            for row, sign in ((item["plus"], 1), (item["minus"], -1)):
                base = int(row["summary"]["full_name_distribution"].get("google", 0))
                extra = int(row["unlabelled_clean_answers"])
                options.append((sign, [base + k for k in range(extra + 1)]))
            ranges.append([plus - minus for plus in options[0][1] for minus in options[1][1]])
        xs_all = [item["draw"][projection] for item in rows]
        low, high = 2.0, -2.0
        assignments = 0
        for combo in itertools.product(*[sorted(set(option)) for option in ranges]):
            value = spearman(xs_all, combo)
            assignments += 1
            if value is not None:
                low = min(low, value)
                high = max(high, value)
        out[f"u100d_{projection}"] = {"complete_draws": complete,
                                      "all_40_draws_every_assignment": {"assignments": assignments,
                                                                        "rho_min": low, "rho_max": high}}
    out["descriptive_note"] = ("u100d only. Not 174.6 or 174.9, which ask across the 60 draws; "
                               "those stay not read.")
    return out


def relations():
    out = {}
    for family in ("u100d", "f100d"):
        rows = [row["summary"] for row in directions if row["family"] == family]
        out[family] = {"answers": sum(int(row["answers"]) for row in rows),
                       "clean": sum(int(row["clean"]) for row in rows),
                       "broken": sum(int(row["broken"]) for row in rows),
                       "relation_claim": sum(int(row["relation_claim"]) for row in rows),
                       "relation_none": sum(int(row["relation_none"]) for row in rows)}
    return out


def readers():
    first = {row["id"]: row for row in read_jsonl(ROOT / "labels" / "labels-ds.jsonl")}
    second = {row["id"]: row for row in read_jsonl(ROOT / "labels" / "labels-haiku.jsonl")}
    both = [key for key in second if key in first]
    resolved = [key for key in both if first[key].get("resolved_primary") and second[key].get("resolved_primary")]
    match = sum(first[key]["resolved_primary"] == second[key]["resolved_primary"] for key in resolved)
    return {"pairs_read_by_both": len(both), "both_resolved_a_primary": len(resolved),
            "same_resolved_primary": match,
            "disagreements": [{"ds": first[key]["resolved_primary"], "haiku": second[key]["resolved_primary"]}
                              for key in resolved
                              if first[key]["resolved_primary"] != second[key]["resolved_primary"]]}


result = {"part": 174,
          "note": "Recomputed after the cold check. score.json and prediction_table.json are the registered scorer's output and are unchanged.",
          "histogram": histogram(), "floors": floors(), "sign_contrast": contrasts(),
          "relations": relations(), "readers": readers(),
          "family_sd_note": ("score.json reports the pooled 60-draw projection SD under each family; "
                             "the family SDs are in descriptive_fallback.")}
OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, indent=1)[:5200])
