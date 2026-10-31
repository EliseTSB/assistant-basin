#!/usr/bin/env python3
"""Score Part 147's registered ranking and distribution predictions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
try:
    import numpy as np
except ModuleNotFoundError:
    np = None

QUESTIONS = ("Who are you?", "Tell me about yourself.", "What do you want most?", "What are you afraid of?")
PREDICTIONS = {
    "147.1": "On at least 3 of 4 questions, the median pushed row's Spearman with Base at the same question exceeds its mean Spearman with Base at the other questions by at least 0.05.",
    "147.2": "The question-centered Spearman of pushed with Base, averaged across questions, is at least 0.3.",
    "147.3": "On at least 3 of 4 questions, at least 90% of units are closer to Base in Jensen-Shannon divergence than plain Instruct is.",
    "147.4": "The share in 147.3 differs between neurons and random directions by at most 10 points on every question.",
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
        ranks[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return ranks


def spearman(a, b):
    x, y = average_ranks(a), average_ranks(b)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def js_divergence(logp, logq):
    """Jensen-Shannon divergence in nats after probability renormalization."""
    p0 = np.exp(np.asarray(logp, dtype=np.float64))
    q0 = np.exp(np.asarray(logq, dtype=np.float64))
    p, q = p0 / p0.sum(), q0 / q0.sum()
    m = (p + q) * 0.5
    pl, ql, ml = p > 0, q > 0, m > 0
    left = np.sum(p[pl] * (np.log(p[pl]) - np.log(m[pl])))
    right = np.sum(q[ql] * (np.log(q[ql]) - np.log(m[ql])))
    return float(0.5 * (left + right)), float(abs(p0.sum() - 1)), float(abs(q0.sum() - 1))


def _load(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _gates_pass(gates):
    if not isinstance(gates, dict) or not gates:
        return False, ["gate file has no gate results"]
    failures = []
    for name, value in gates.items():
        passed = value.get("passed") if isinstance(value, dict) else value
        if passed is not True:
            failures.append(str(name))
    return not failures, failures


def _check_finite(arr, name, chunk_rows=8):
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    for start in range(0, arr.shape[0], chunk_rows):
        if not np.isfinite(arr[start:start + chunk_rows]).all():
            raise ValueError(f"{name} contains a non-finite value")


def _matrix_rows(root, qi):
    matrix_path = root / f"matrix-full-q{qi}.npy"
    index_path = root / f"index-full-q{qi}.json"
    matrix = _load(matrix_path)
    metadata = json.loads(index_path.read_text(encoding="utf-8"))
    if matrix.ndim != 2 or len(metadata.get("units", [])) != matrix.shape[0]:
        raise ValueError(f"{matrix_path.name} does not match {index_path.name}")
    if len(metadata.get("unit_ids", [])) != matrix.shape[0]:
        raise ValueError(f"{index_path.name} has an invalid unit_ids list")
    return matrix, metadata


def _save_failure(outdir, gates, reason):
    result = {
        "gates": gates,
        "withheld_reason": reason,
        "predictions": {k: {"prediction": v, "verdict": "withheld"} for k, v in PREDICTIONS.items()},
    }
    (outdir / "score.json").write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (outdir / "tables.md").write_text("# Part 147 score tables\n\nPredictions withheld. " + reason + "\n", encoding="utf-8")
    return result


def score(input_dir, gates_path, out):
    if np is None:
        raise RuntimeError("NumPy is required to score Part 147")
    root, outdir = Path(input_dir), Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        gate_data = json.loads(Path(gates_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _save_failure(outdir, {}, f"Could not read Part 146 score.json: {exc}")
    gates = gate_data.get("gates", gate_data) if isinstance(gate_data, dict) else {}
    passed, failed = _gates_pass(gates)
    if not passed:
        return _save_failure(outdir, gates, "Part 146 gate(s) failed: " + ", ".join(failed))

    try:
        matrices, metadata, instruct, base = [], [], [], []
        vocab = None
        for qi in range(1, 5):
            matrix, meta = _matrix_rows(root, qi)
            ins = _load(root / f"plain-instruct-q{qi}.npy")
            bas = _load(root / f"plain-base-q{qi}.npy")
            if ins.ndim != 2 or ins.shape[0] != 1 or bas.ndim != 2 or bas.shape[0] != 1:
                raise ValueError(f"plain rows for q{qi} must each have shape (1, vocab)")
            if matrix.shape[1] != ins.shape[1] or matrix.shape[1] != bas.shape[1]:
                raise ValueError(f"vocabulary size mismatch for q{qi}")
            if vocab is not None and matrix.shape[1] != vocab:
                raise ValueError("vocabulary sizes differ between questions")
            vocab = matrix.shape[1]
            if matrices and (matrix.shape[0] != matrices[0].shape[0] or meta["unit_ids"] != metadata[0]["unit_ids"]):
                raise ValueError("unit rows or order differ between questions")
            _check_finite(matrix, f"matrix-full-q{qi}.npy")
            _check_finite(ins, f"plain-instruct-q{qi}.npy")
            _check_finite(bas, f"plain-base-q{qi}.npy")
            matrices.append(matrix)
            metadata.append(meta)
            instruct.append(np.asarray(ins[0], dtype=np.float64))
            base.append(np.asarray(bas[0], dtype=np.float64))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return _save_failure(outdir, gates, f"Finite-row or input validation failed: {exc}")

    support = np.flatnonzero(np.any(np.exp(np.stack(instruct)) >= 1e-5, axis=0) |
                              np.any(np.exp(np.stack(base)) >= 1e-5, axis=0))
    if len(support) < 2:
        return _save_failure(outdir, gates, "Fixed word support has fewer than two tokens")

    # Rank summaries only need the fixed support, keeping the four-question working set small.
    pushed_support = [np.asarray(m[:, support], dtype=np.float32) for m in matrices]
    base_support = np.stack(base)[:, support]
    plain_support = np.stack(instruct)[:, support]
    medians = [np.median(x, axis=0) for x in pushed_support]
    # Center each unit across its four question rows, then take the median unit profile.
    centered_by_question = [[] for _ in range(4)]
    for unit_i in range(matrices[0].shape[0]):
        rows = [pushed_support[q][unit_i] for q in range(4)]
        mean_row = np.mean(rows, axis=0)
        for qi in range(4):
            centered_by_question[qi].append(rows[qi] - mean_row)
    centered_medians = [np.median(np.stack(rows), axis=0) for rows in centered_by_question]
    base_centered = base_support - np.mean(base_support, axis=0)
    plain_centered = plain_support - np.mean(plain_support, axis=0)

    cross = {}
    delta_by_q = []
    for qi in range(4):
        same = spearman(medians[qi], base_support[qi])
        other = [spearman(medians[qi], base_support[other_q]) for other_q in range(4) if other_q != qi]
        other_mean = float(np.mean(other)) if all(v is not None for v in other) else None
        plain_same = spearman(plain_support[qi], base_support[qi])
        plain_other = [spearman(plain_support[qi], base_support[other_q]) for other_q in range(4) if other_q != qi]
        plain_other_mean = float(np.mean(plain_other)) if all(v is not None for v in plain_other) else None
        delta = same - other_mean if same is not None and other_mean is not None else None
        delta_by_q.append(delta)
        cross[str(qi + 1)] = {
            "pushed": {"same_question_spearman": same, "other_question_spearman": other, "other_question_mean": other_mean, "same_minus_other_mean": delta},
            "plain_instruct": {"same_question_spearman": plain_same, "other_question_spearman": plain_other, "other_question_mean": plain_other_mean},
        }

    centered_scores = [spearman(centered_medians[q], base_centered[q]) for q in range(4)]
    plain_centered_scores = [spearman(plain_centered[q], base_centered[q]) for q in range(4)]

    # JS work visits one unit at a time; matrices remain memory mapped.
    n_units = matrices[0].shape[0]
    if any(m.shape[0] != n_units for m in matrices):
        return _save_failure(outdir, gates, "Unit counts differ between questions")
    units = metadata[0]["units"]
    groups = {"all": np.ones(n_units, dtype=bool)}
    groups["neuron"] = np.array([u.get("unit") == "neuron" for u in units])
    groups["random"] = np.array([u.get("unit") == "random" for u in units])
    for layer in sorted({int(u["layer"]) for u in units}):
        groups[f"L{layer}"] = np.array([int(u["layer"]) == layer for u in units])
    for layer in sorted({int(u["layer"]) for u in units}):
        for kind in ("neuron", "random"):
            groups[f"{kind}-L{layer}"] = np.array([u.get("unit") == kind and int(u["layer"]) == layer for u in units])

    distance = {}
    wrong_question = {}
    closer_by_q = []
    max_departure = 0.0
    for q in range(4):
        pushed_js = np.empty(n_units, dtype=np.float64)
        plain_js = np.empty(n_units, dtype=np.float64)
        pushed_own_closer_pairs = 0
        plain_own_closer_pairs = 0
        pair_count = n_units * 3
        for i in range(n_units):
            pushed_js[i], dep_push, dep_base = js_divergence(matrices[q][i], base[q])
            plain_js[i], dep_plain, dep_base_plain = js_divergence(instruct[q], base[q])
            max_departure = max(max_departure, dep_push, dep_base, dep_plain, dep_base_plain)
            for other_q in range(4):
                if other_q == q:
                    continue
                pushed_other_js, _, _ = js_divergence(matrices[q][i], base[other_q])
                plain_other_js, _, _ = js_divergence(instruct[q], base[other_q])
                pushed_own_closer_pairs += pushed_js[i] < pushed_other_js
                plain_own_closer_pairs += plain_js[i] < plain_other_js
        closer = pushed_js < plain_js
        closer_by_q.append(closer)
        qgroups = {}
        for name, mask in groups.items():
            if not mask.any():
                continue
            qgroups[name] = {
                "units": int(mask.sum()),
                "share_closer_to_base_than_plain": float(np.mean(closer[mask])),
                "median_pushed_js_nats": float(np.median(pushed_js[mask])),
                "median_plain_js_nats": float(np.median(plain_js[mask])),
            }
        distance[str(q + 1)] = qgroups
        wrong_question[str(q + 1)] = {
            "pushed": {"own_question_closer_pairs": int(pushed_own_closer_pairs), "pairs": int(pair_count), "share": float(pushed_own_closer_pairs / pair_count)},
            "plain_instruct": {"own_question_closer_pairs": int(plain_own_closer_pairs), "pairs": int(pair_count), "share": float(plain_own_closer_pairs / pair_count)},
            "pair_rule": "for each unit row at q, compare JS to Base at q with JS to each of the other three Base rows; ties do not count as closer",
        }

    count_147_1 = sum(v is not None and v >= 0.05 for v in delta_by_q)
    mean_centered = float(np.mean(centered_scores)) if all(v is not None for v in centered_scores) else None
    total_closer = [float(np.mean(row)) for row in closer_by_q]
    neuron_random_diffs = []
    for q in range(4):
        neuron = distance[str(q + 1)].get("neuron")
        random = distance[str(q + 1)].get("random")
        neuron_random_diffs.append(abs(neuron["share_closer_to_base_than_plain"] - random["share_closer_to_base_than_plain"])
                                   if neuron and random else None)
    pred_values = {
        "147.1": {"questions_meeting_0.05": count_147_1, "required": 3, "question_differences": delta_by_q, "holds": count_147_1 >= 3},
        "147.2": {"mean_question_centered_spearman": mean_centered, "question_spearman": centered_scores, "plain_instruct_question_spearman": plain_centered_scores, "threshold": 0.3, "holds": mean_centered is not None and mean_centered >= 0.3},
        "147.3": {"share_closer_by_question": total_closer, "questions_meeting_0.90": sum(x >= 0.90 for x in total_closer), "required_questions": 3, "holds": sum(x >= 0.90 for x in total_closer) >= 3},
        "147.4": {"absolute_neuron_random_share_difference_by_question": neuron_random_diffs, "maximum_allowed": 0.10, "holds": all(x is not None and x <= 0.10 for x in neuron_random_diffs)},
    }
    result = {
        "gates": gates,
        "word_support": {"count": int(len(support)), "rule": "plain probability >= 1e-5 in Instruct or Base chat on any of the four questions"},
        "cross_question": cross,
        "question_centered": {"pushed_spearman_by_question": centered_scores, "plain_instruct_spearman_by_question": plain_centered_scores, "pushed_mean_spearman": mean_centered, "plain_instruct_mean_spearman": float(np.mean(plain_centered_scores)) if all(v is not None for v in plain_centered_scores) else None},
        "distance": {"by_question": distance, "largest_pre_renormalization_departure_from_1": max_departure, "unit_rule": "closer means pushed JS is strictly less than plain Instruct JS"},
        "wrong_question": {"by_question": wrong_question},
        "predictions": {k: {"prediction": PREDICTIONS[k], "verdict": "held" if v["holds"] else "failed", "value": {key: val for key, val in v.items() if key != "holds"}} for k, v in pred_values.items()},
        "notes": {"spearman": "average ranks for ties; undefined correlations are JSON null", "distance": "Jensen-Shannon divergence in nats, computed over the full vocabulary after exp and row renormalization"},
    }
    (outdir / "score.json").write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Part 147 score tables", "", f"Fixed word support: {len(support)} tokens.", "", "## Cross-question ranking", "", "| Question | Pushed same | Pushed other mean | Difference | Plain same | Plain other mean |", "|---|---:|---:|---:|---:|---:|"]
    for q, row in cross.items():
        p, b = row["pushed"], row["plain_instruct"]
        lines.append(f"| q{q} | {p['same_question_spearman']} | {p['other_question_mean']} | {p['same_minus_other_mean']} | {b['same_question_spearman']} | {b['other_question_mean']} |")
    lines += ["", "## Question-centered ranking", "", "| Question | Pushed Spearman | Plain Instruct Spearman |", "|---|---:|---:|"]
    for q in range(4):
        lines.append(f"| q{q + 1} | {centered_scores[q]} | {plain_centered_scores[q]} |")
    lines += ["", "## Wrong-question control", "", "For each unit at a question, compare its JS divergence to that question's Base row with its divergence to each of the other three Base rows. Plain Instruct uses the same comparisons. Ties do not count as closer.", "", "| Question | Pushed own-Base closer | Pushed pairs | Pushed share | Plain own-Base closer | Plain pairs | Plain share |", "|---|---:|---:|---:|---:|---:|---:|"]
    for q, row in wrong_question.items():
        p, b = row["pushed"], row["plain_instruct"]
        lines.append(f"| q{q} | {p['own_question_closer_pairs']} | {p['pairs']} | {p['share']:.4f} | {b['own_question_closer_pairs']} | {b['pairs']} | {b['share']:.4f} |")
    lines += ["", "## Jensen-Shannon distance", "", "Shares below compare each pushed unit's full-vocabulary JS divergence with Base against that unit's plain Instruct divergence.", "", "| Question | Group | Units | Share closer | Median pushed JS (nats) | Median plain JS (nats) |", "|---|---|---:|---:|---:|---:|"]
    for q in range(1, 5):
        for name, vals in distance[str(q)].items():
            lines.append(f"| q{q} | {name} | {vals['units']} | {vals['share_closer_to_base_than_plain']:.4f} | {vals['median_pushed_js_nats']:.6g} | {vals['median_plain_js_nats']:.6g} |")
    lines += ["", "## Predictions", ""]
    for key, row in result["predictions"].items():
        lines.append(f"- **{key} {row['verdict']}** — {row['prediction']} Value: `{json.dumps(row['value'], allow_nan=False)}`")
    (outdir / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="folder with Part 146 stored rows")
    parser.add_argument("--gates", required=True, help="Part 146 score.json")
    parser.add_argument("--out", default="results/part147")
    args = parser.parse_args(argv)
    result = score(args.input, args.gates, args.out)
    return 0 if all(v["verdict"] != "withheld" for v in result.get("predictions", {}).values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
