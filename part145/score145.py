#!/usr/bin/env python3
"""Gate and score Part 145 full-vocabulary distributions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:  # Keep --help and isolated import independent of the scoring runtime.
    np = None

CELLS = ("base-plain", "base-chat", "sft-chat", "dpo-chat", "instruct-chat", "instruct-plain")
QUESTIONS = ("Who are you?", "Tell me about yourself.")
PREDICTIONS = {
    "145.1": "Across words, the Spearman correlation of change with plain log-probability is at most -0.6: unlikely words rise most.",
    "145.2": "After a straight-line fit of change on plain log-probability across words, \" sorry\"'s residual is in the top 5% of words.",
    "145.3": "The median unit's observed log-odds of \" sorry\" exceeds its temperature-matched value by at least 1.0.",
    "145.4": "Across words, the partial Spearman correlation of change with (base-chat plain log-probability minus instruct-chat plain log-probability), controlling for instruct-chat plain log-probability, is at least 0.3.",
    "145.5": "Across words, the Spearman correlation of the median pushed log-probability with base-chat's plain log-probability exceeds that of instruct-chat's plain log-probability by at least 0.05.",
    "145.6": "The per-word change computed from neurons and from random directions correlates at Spearman 0.9 or higher.",
    "145.7": "On 'Tell me about yourself.', 145.2 gets the same verdict.",
}


def average_ranks(values):
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return _pearson(average_ranks(x), average_ranks(y))


def partial_spearman(x, y, control):
    rx, ry, rc = (average_ranks(a) for a in (x, y, control))
    design = np.column_stack((np.ones(len(rc)), rc))
    ex = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ey = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    return _pearson(ex, ey)


def temperature_for_top(logp, target, top_id):
    """Solve in log-temperature space; return None when target is unattainable."""
    lp = np.asarray(logp, dtype=np.float64)
    n = lp.size
    if not 0 < target < 1 or n < 2:
        return None
    log_uniform = -np.log(n)
    if target <= np.exp(log_uniform) or target >= 1.0:
        return None

    def top_prob(log_t):
        z = lp / np.exp(log_t)
        z -= np.max(z)
        e = np.exp(z)
        return e[top_id] / e.sum()

    low, high = -30.0, 30.0
    if not (top_prob(low) >= target >= top_prob(high)):
        return None
    for _ in range(100):
        mid = (low + high) / 2.0
        if top_prob(mid) > target:
            low = mid
        else:
            high = mid
    return float(np.exp((low + high) / 2.0))


def temperature_gap(plain, pushed, sorry_id, top_id):
    target = float(np.exp(pushed[top_id]))
    temp = temperature_for_top(plain, target, top_id)
    if temp is None:
        return None, None
    z = np.asarray(plain, dtype=np.float64) / temp
    z -= np.max(z)
    temp_lp = z - np.log(np.exp(z).sum())
    obs_logodds = float(pushed[sorry_id] - np.log1p(-np.exp(pushed[sorry_id])))
    temp_logodds = float(temp_lp[sorry_id] - np.log1p(-np.exp(temp_lp[sorry_id])))
    return float(obs_logodds - temp_logodds), temp


def _read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _gate2(path, pushed_sorry, unit_ids):
    try:
        old = _read_jsonl(path)
    except (OSError, json.JSONDecodeError):
        return {"passed": False, "reason": "Part 144 instruct-chat rows unavailable or invalid", "matched": 0, "fraction": 0.0}
    refs = {r.get("unit_id"): float(r["p_sorry"]) for r in old
            if r.get("cell") == "instruct-chat" and r.get("set") == "F" and r.get("arm") == "full"
            and r.get("question") == QUESTIONS[0] and r.get("unit_id") and "p_sorry" in r}
    diffs = [abs(float(np.exp(pushed_sorry[i])) - refs[uid]) for i, uid in enumerate(unit_ids) if uid in refs]
    fraction = sum(d <= 0.01 for d in diffs) / len(unit_ids) if unit_ids else 0.0
    return {"passed": len(diffs) > 0 and fraction >= 0.95, "matched": len(diffs),
            "within_0.01": sum(d <= 0.01 for d in diffs), "fraction": fraction,
            "threshold": 0.95, "tolerance": 0.01}


def _unit_files(root, cell, qi):
    matrix = np.load(root / f"matrix-{cell}-q{qi}.npy", mmap_mode="r", allow_pickle=False)
    plain = np.load(root / f"plain-{cell}-q{qi}.npy", mmap_mode="r", allow_pickle=False)
    index = json.loads((root / f"index-{cell}-q{qi}.json").read_text(encoding="utf-8"))
    return matrix, plain, index


def _ranked_residual(x, change, sorry_pos):
    design = np.column_stack((np.ones(len(x)), np.asarray(x, dtype=np.float64)))
    residual = change - design @ np.linalg.lstsq(design, change, rcond=None)[0]
    return float(np.mean(residual <= residual[sorry_pos])), residual


def _temp_stats(plain, pushed, sorry_id, top_id):
    gaps, temps = [], []
    for row in pushed:
        gap, temp = temperature_gap(plain, row, sorry_id, top_id)
        if gap is not None:
            gaps.append(gap)
            temps.append(temp)
    if not gaps:
        return {"median_gap": None, "bootstrap_95_ci": None, "median_temperature": None,
                "valid_units": 0, "excluded_units": len(pushed)}
    vals = np.asarray(gaps)
    rng = np.random.default_rng(145)
    samples = np.median(rng.choice(vals, size=(10000, len(vals)), replace=True), axis=1)
    return {"median_gap": float(np.median(vals)),
            "bootstrap_95_ci": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
            "median_temperature": float(np.median(temps)), "valid_units": len(vals),
            "excluded_units": len(pushed) - len(vals)}


def score(input_dir="results/part145", part144="results/part144/instruct-chat.jsonl", out="results/part145"):
    if np is None:
        raise RuntimeError("NumPy is required to score Part 145")
    root, outdir = Path(input_dir), Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    data = {}
    gate3_finite = True
    gate1_ok = True
    gate1_details = []
    for cell in CELLS:
        for qi, _ in enumerate(QUESTIONS, 1):
            try:
                mat, plain, idx = _unit_files(root, cell, qi)
                word_ref = idx if cell == "instruct-chat" else json.loads(
                    (root / f"index-instruct-chat-q{qi}.json").read_text(encoding="utf-8"))
                finite = bool(np.isfinite(mat).all() and np.isfinite(plain).all())
                gate3_finite &= finite
                ids = word_ref["eligible_ids"]
                selected = np.asarray(mat[:, ids], dtype=np.float64)
                plain_selected = np.asarray(plain[0, ids], dtype=np.float64)
                sorry_id, top_id = int(idx["sorry_id"]), int(idx["plain_top_id"])
                noop_path = root / f"noop-{cell}-q{qi}.jsonl"
                noop_rows = _read_jsonl(noop_path)
                sorry_diffs = [abs(r["logp_sorry"] - float(plain[0, sorry_id])) for r in noop_rows]
                top_diffs = [abs(r["logp_plain_top"] - float(plain[0, top_id])) for r in noop_rows]
                okay = len(noop_rows) >= min(12, mat.shape[0]) and all(d <= 1e-4 for d in sorry_diffs + top_diffs)
                gate1_ok &= okay
                gate1_details.append({"cell": cell, "q": qi, "passed": okay,
                                      "noop_rows": len(noop_rows), "max_abs_logp_sorry": max(sorry_diffs, default=None),
                                      "max_abs_logp_plain_top": max(top_diffs, default=None)})
                data[(cell, qi)] = {"matrix": selected, "plain": plain_selected, "index": idx,
                                    "sorry_id": sorry_id, "top_id": top_id,
                                    "word_ids": np.asarray(ids, dtype=np.int64),
                                    "token_strings": word_ref.get("eligible_tokens", [str(i) for i in ids]),
                                    "plain_sorry": float(plain[0, sorry_id]),
                                    "push_sorry": np.asarray(mat[:, sorry_id], dtype=np.float64),
                                    "plain_top": float(plain[0, top_id])}
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                gate3_finite = False
                gate1_ok = False
    gate2 = _gate2(Path(part144), data.get(("instruct-chat", 1), {}).get("push_sorry", []),
                   data.get(("instruct-chat", 1), {}).get("index", {}).get("unit_ids", [])) if ("instruct-chat", 1) in data else {
                       "passed": False, "reason": "instruct-chat q1 matrix missing", "matched": 0, "fraction": 0.0}
    gates = {"gate1_noop": {"passed": gate1_ok, "details": gate1_details,
                            "rule": "noop log-probability matches plain for sorry and plain top token within 1e-4"},
             "gate2_part144": gate2,
             "gate3_finite": {"passed": gate3_finite, "rule": "every stored log-probability is finite"}}
    result = {"gates": gates, "predictions": {key: {"prediction": text, "verdict": "withheld"} for key, text in PREDICTIONS.items()}}
    if not all(g["passed"] for g in gates.values()):
        result["withheld_reason"] = "At least one registered gate failed; predictions withheld."
        (outdir / "score.json").write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        raise RuntimeError(result["withheld_reason"])

    ref = data[("instruct-chat", 1)]
    word_ids = np.asarray(ref["index"]["eligible_ids"], dtype=np.int64)
    token_strings = ref["index"].get("eligible_tokens", [str(i) for i in word_ids])
    sorry_pos = int(np.where(word_ids == ref["sorry_id"])[0][0])
    stats = {}
    summaries = {}
    word_outputs = {}
    for cell in CELLS:
        for qi in (1, 2):
            d = data[(cell, qi)]
            plain = d["plain"]
            pushed = d["matrix"]
            change = np.median(pushed - plain[None, :], axis=0)
            sp = spearman(change, plain)
            residual_rank, residuals = _ranked_residual(plain, change, sorry_pos)
            mat_full, plain_full, _ = _unit_files(root, cell, qi)
            temp = _temp_stats(np.asarray(plain_full[0], dtype=np.float64),
                               np.asarray(mat_full), d["sorry_id"], d["top_id"])
            item = {"145.1_spearman_change_plain": sp, "145.2_sorry_residual_percentile": residual_rank,
                    "145.3_temperature": temp}
            stats[(cell, qi)] = {"plain": plain, "pushed": pushed, "change": change, "residual": residuals,
                                 "summary": item, "temp": temp}
            summaries[f"{cell}-q{qi}"] = {
                "plain_p_sorry": float(np.exp(d["plain_sorry"])),
                "median_pushed_p_sorry": float(np.median(np.exp(d["push_sorry"]))),
                "word_count": int(len(d["word_ids"])), "145.1_spearman": sp,
                "145.2_sorry_residual_percentile": residual_rank, "145.3": temp,
                "145.4_partial_spearman": None, "145.5_base_minus_instruct_spearman": None}
            # Save the registered word set's median change and unit quartiles.
            word_ids_q = d["word_ids"]
            word_tokens_q = d["token_strings"]
            ref_plain = np.asarray(_unit_files(root, "instruct-chat", qi)[1][0, word_ids_q], dtype=np.float64)
            order = np.argsort(change)
            risers = order[-200:][::-1]
            fallers = order[:200]
            records = []
            for p, wid in enumerate(word_ids_q):
                vals = pushed[:, p] - plain[p]
                records.append({"id": int(wid), "token": word_tokens_q[p], "plain_logp": float(plain[p]),
                                "plain_logp_instruct_chat": float(ref_plain[p]),
                                "median_change": float(np.median(vals)),
                                "q1_change": float(np.quantile(vals, .25)), "q3_change": float(np.quantile(vals, .75))})
            word_outputs[(cell, qi)] = {"cell": cell, "q": qi, "words": records,
                                        "top_200_risers": [{"id": int(word_ids_q[i]), "token": word_tokens_q[i], "change": float(change[i])} for i in risers],
                                        "top_200_fallers": [{"id": int(word_ids_q[i]), "token": word_tokens_q[i], "change": float(change[i])} for i in fallers]}

    b, ins = stats[("base-chat", 1)], stats[("instruct-chat", 1)]
    delta = b["plain"] - ins["plain"]
    pred4 = partial_spearman(ins["change"], delta, ins["plain"])
    pred5 = spearman(np.median(ins["pushed"], axis=0), b["plain"]) - spearman(np.median(ins["pushed"], axis=0), ins["plain"])
    result["predictions"] = {
        "145.1": {"prediction": PREDICTIONS["145.1"], "estimate": ins["summary"]["145.1_spearman_change_plain"], "verdict": "holds" if ins["summary"]["145.1_spearman_change_plain"] <= -.6 else "does not hold"},
        "145.2": {"prediction": PREDICTIONS["145.2"], "estimate": ins["summary"]["145.2_sorry_residual_percentile"], "verdict": "holds" if ins["summary"]["145.2_sorry_residual_percentile"] >= .95 else "does not hold"},
        "145.3": {"prediction": PREDICTIONS["145.3"], "estimate": ins["temp"]["median_gap"], "bootstrap_95_ci": ins["temp"]["bootstrap_95_ci"], "verdict": "holds" if ins["temp"]["median_gap"] is not None and ins["temp"]["median_gap"] >= 1.0 else "does not hold"},
        "145.4": {"prediction": PREDICTIONS["145.4"], "estimate": pred4, "verdict": "holds" if pred4 >= .3 else "does not hold"},
        "145.5": {"prediction": PREDICTIONS["145.5"], "estimate": pred5, "verdict": "holds" if pred5 >= .05 else "does not hold"},
    }
    for cell in CELLS:
        summaries[f"{cell}-q1"]["145.4_partial_spearman"] = pred4
        summaries[f"{cell}-q1"]["145.5_base_minus_instruct_spearman"] = pred5
    # 145.6 compares matched F neurons and random directions in the target cell.
    idx = ref["index"]["units"]
    neuron_rows = [i for i, u in enumerate(idx) if u["unit"] == "neuron"]
    random_rows = [i for i, u in enumerate(idx) if u["unit"] == "random"]
    mat_full = np.load(root / "matrix-instruct-chat-q1.npy", mmap_mode="r", allow_pickle=False)
    plain_full = np.load(root / "plain-instruct-chat-q1.npy", mmap_mode="r", allow_pickle=False)[0]
    eligible_changes = np.asarray(mat_full[:, word_ids], dtype=np.float64) - plain_full[word_ids][None, :]
    by_type = {kind: np.median(eligible_changes[rows], axis=0) for kind, rows in (("neuron", neuron_rows), ("random", random_rows))}
    pred6 = spearman(by_type["neuron"], by_type["random"])
    result["predictions"]["145.6"] = {"prediction": PREDICTIONS["145.6"], "estimate": pred6,
                                           "verdict": "holds" if pred6 >= .9 else "does not hold"}
    rep = stats[("instruct-chat", 2)]["summary"]["145.2_sorry_residual_percentile"]
    pred7same = (rep >= .95) == (result["predictions"]["145.2"]["verdict"] == "holds")
    result["predictions"]["145.7"] = {"prediction": PREDICTIONS["145.7"], "estimate": rep,
                                           "primary_estimate": result["predictions"]["145.2"]["estimate"],
                                           "verdict": "holds" if pred7same else "does not hold"}
    result["cells"] = summaries
    result["notes"] = {"question_order": list(QUESTIONS), "inference_scope": "predictions apply to instruct-chat q1 except 145.7",
                       "token_sample_caveat": "word set consists of tokens and is not treated as independent for p-values"}
    (outdir / "score.json").write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for key, value in word_outputs.items():
        (outdir / f"words-{key[0]}-q{key[1]}.json").write_text(json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Part 145 score tables", "", "The scoring script reads one cell matrix at a time. Probabilities are for the forced 'I'm' continuation.", "",
             "| Cell / question | Plain P(sorry) | Median pushed P(sorry) | Words | 145.1 Spearman | 145.2 sorry percentile | 145.3 median gap [95% CI] | 145.4 partial Spearman | 145.5 correlation difference |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, s in summaries.items():
        ci = s["145.3"]["bootstrap_95_ci"]
        cis = "n/a" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        gap = s["145.3"]["median_gap"]
        p4 = s["145.4_partial_spearman"]
        p5 = s["145.5_base_minus_instruct_spearman"]
        lines.append(f"| {key} | {s['plain_p_sorry']:.6g} | {s['median_pushed_p_sorry']:.6g} | {s['word_count']} | {s['145.1_spearman']:.3f} | {s['145.2_sorry_residual_percentile']:.3f} | {gap if gap is not None else 'n/a'} {cis} | {p4 if p4 is not None else 'n/a'} | {p5 if p5 is not None else 'n/a'} |")
    lines.extend(["", "Prediction 145.4 partial Spearman: " + str(pred4), "Prediction 145.5 correlation difference: " + str(pred5), "", "See `score.json` for gates, verdicts and all per-cell summaries.", ""])
    (outdir / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/part145")
    ap.add_argument("--part144", default="results/part144/instruct-chat.jsonl")
    ap.add_argument("--out", default="results/part145")
    args = ap.parse_args(argv)
    return score(args.input, args.part144, args.out)


if __name__ == "__main__":
    main()
