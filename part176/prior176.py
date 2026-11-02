#!/usr/bin/env python3
"""Numbers Part 176's design cites from data already public. Reads stored rows only, no model, no paid call.

Run from the repo root with the research Python: part176/prior176.py > results/part176/prior176.json
Every line here is exploratory: these draws and doses were seen before the design was written.
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "part173"))
from part165 import score165  # noqa: E402
import label173  # noqa: E402

R175 = ROOT / "results/part175"
FRAGMENT_TOKENS = 12
OWN_NAME = re.compile("qwen|tongyi|alibaba", re.I)
STAGE_A_DOSES = ("0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45")


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stage_a_word_table():
    """Raw `google` word hits in clean answers, by direction and dose. No reader ran on Stage A."""
    table = defaultdict(lambda: {"answers": 0, "broken": 0, "google_words_in_clean": 0})
    for row in jsonl(R175 / "stage_a_answers.jsonl"):
        cell = table[row["arm"]]
        broken = bool(score165.primary_broken(row["answer"]))
        cell["answers"] += 1
        cell["broken"] += broken
        cell["google_words_in_clean"] += bool(not broken and re.search("google", row["first40"], re.I))
    rows = {}
    for seed in range(175901, 175905):
        for sign, letter in ((1, "p"), (-1, "m")):
            rows[f"{seed}{'+' if sign > 0 else '-'}"] = {
                dose: table[f"f100d_d{seed}_dose_{letter}{dose.replace('.', 'p')}"] for dose in STAGE_A_DOSES}
    meets = {dose: sum(cells[dose]["google_words_in_clean"] >= 10 and cells[dose]["broken"] <= 8
                       for cells in rows.values()) for dose in STAGE_A_DOSES}
    return rows, meets


def census():
    by = defaultdict(dict)
    for row in jsonl(R175 / "scored_directions.jsonl"):
        if row.get("family") == "f100d" and "dose" in row:
            by[(row["seed"], 1 if row["dose"] > 0 else -1)][abs(row["dose"])] = row
    return by


def pooled_states(by):
    result = {}
    for dose in (0.15, 0.225, 0.3):
        totals = Counter()
        for cells in by.values():
            summary = cells[dose]["summary"]
            names = summary.get("full_name_distribution") or {}
            totals["qwen"] += names.get("qwen", 0)
            totals["google"] += names.get("google", 0)
            totals["other_name"] += sum(v for k, v in names.items() if k not in ("qwen", "google"))
            totals["no_claim"] += summary.get("relation_none") or 0
            totals["broken"] += cells[dose]["broken"]
            totals["answers"] += cells[dose]["answers"]
        result[str(dose)] = dict(totals)
    return result


def answer_states():
    """Split each clean no-claim answer: own name in its words, a fragment of at most 12 tokens, or a longer answer."""
    labels = {}
    for row in jsonl(R175 / "labels/labels-ds.jsonl"):
        labels[row["id"]] = row
    pooled = defaultdict(Counter)
    per_direction = defaultdict(Counter)
    for row in jsonl(R175 / "answers.jsonl"):
        match = re.fullmatch(r"f100d_d(\d+)_dose_([pm])(.+)", row["arm"])
        if not match or not 174101 <= int(match[1]) <= 174120:
            continue
        dose = float(match[3].replace("p", "."))
        key = (int(match[1]), 1 if match[2] == "p" else -1)
        cell = pooled[str(dose)]
        cell["answers"] += 1
        if score165.primary_broken(row["answer"]):
            cell["broken"] += 1
            per_direction[(key, dose)]["failed"] += 1
            continue
        label = labels.get(label173.pair_id(row["question"], row["answer"]))
        if label is None:
            cell["unlabelled"] += 1
        elif label.get("relation") != "none":
            cell["claim"] += 1
        else:
            cell["no_claim"] += 1
            own = bool(OWN_NAME.search(row["first40"]))
            short = int(row["token_count"]) <= FRAGMENT_TOKENS
            cell["no_claim_own_name_in_words"] += own
            cell["no_claim_fragment"] += (short and not own)
            cell["no_claim_longer"] += (not short and not own)
            per_direction[(key, dose)]["failed"] += (short and not own)
    failed = {}
    for dose in (0.15, 0.225, 0.3):
        counts = [per_direction[((seed, sign), dose)]["failed"] for seed in range(174101, 174121) for sign in (1, -1)]
        failed[str(dose)] = {"median": float(np.median(counts)), "at_least_22": sum(c >= 22 for c in counts),
                             "max": max(counts), "min": min(counts)}
    return {dose: dict(cell) for dose, cell in sorted(pooled.items())}, failed


def named_per_direction(by):
    result = {}
    for dose in (0.15, 0.225, 0.3):
        counts = [cells[dose]["summary"]["named"] for cells in by.values()]
        result[str(dose)] = {"median": float(np.median(counts)), "minimum": min(counts),
                             "under_11": sum(c < 11 for c in counts)}
    return result


def lost(cell):
    return cell["broken"] + (cell["summary"].get("relation_none") or 0)


def spearman_row(by, x, y, seed=176, replicates=4000):
    keys = sorted(by)
    a = np.array([np.nan if x(by[k]) is None else x(by[k]) for k in keys], float)
    b = np.array([np.nan if y(by[k]) is None else y(by[k]) for k in keys], float)
    keep = ~(np.isnan(a) | np.isnan(b))
    rho = float(spearmanr(a[keep], b[keep]).statistic)
    seeds = sorted({k[0] for k in keys})
    index = {s: [i for i, k in enumerate(keys) if k[0] == s] for s in seeds}
    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        chosen = [i for s in (rng.choice(seeds) for _ in seeds) for i in index[s]]
        u, v = a[chosen], b[chosen]
        ok = ~(np.isnan(u) | np.isnan(v))
        if ok.sum() > 5 and len(set(u[ok])) > 1 and len(set(v[ok])) > 1:
            values.append(spearmanr(u[ok], v[ok]).statistic)
    low, high = np.percentile(values, [2.5, 97.5])
    return {"n": int(keep.sum()), "rho": round(rho, 2), "interval": [round(float(low), 2), round(float(high), 2)]}


def supported_counts():
    """Direction-at-dose rows supported in Parts 174 and 175, with leaders, read from the stored scores."""
    out = {}
    rows = jsonl(ROOT / "results/part175/scored_directions.jsonl")
    census_rows = [r for r in rows if r.get("family") == "f100d" and "dose" in r]
    out["part175_rows"] = len(census_rows)
    out["part175_supported"] = sum(bool(r.get("supported")) for r in census_rows)
    out["part175_google_led"] = sum(bool(r.get("google_led")) for r in census_rows)
    out["part175_other_led"] = sum(bool(r.get("other_led")) for r in census_rows)
    return out


def main():
    rows, meets = stage_a_word_table()
    by = census()
    states, failed = answer_states()
    google = lambda dose: (lambda cells: cells[dose].get("google_claim_count"))  # noqa: E731
    result = {
        "stage_a_broken_and_google_words": rows,
        "stage_a_directions_meeting_10_words_and_8_broken": meets,
        "census_pooled_states": pooled_states(by),
        "census_answer_states": states,
        "census_failed_per_direction": failed,
        "census_named_per_direction": named_per_direction(by),
        "census_lost_ge_22": {str(d): sum(lost(c[d]) >= 22 for c in by.values()) for d in (0.15, 0.225, 0.3)},
        "census_median_lost": {str(d): float(np.median([lost(c[d]) for c in by.values()])) for d in (0.15, 0.225, 0.3)},
        "spearman_exploratory": {
            "lost_at_0.30_vs_google_at_0.15": spearman_row(by, lambda c: lost(c[0.3]), google(0.15)),
            "lost_at_0.30_vs_google_at_0.225": spearman_row(by, lambda c: lost(c[0.3]), google(0.225)),
            "broken_at_0.30_vs_google_at_0.15": spearman_row(by, lambda c: c[0.3]["broken"], google(0.15)),
            "broken_at_0.30_vs_google_at_0.225": spearman_row(by, lambda c: c[0.3]["broken"], google(0.225)),
        },
        "supported_rows": supported_counts(),
    }
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
