#!/usr/bin/env python3
"""Score Part 135 reader labels, controls, agreement, and registered predictions."""
import json
import math
from pathlib import Path

import numpy as np

RELATIONS = ("same", "opposite", "unrelated")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def labels(path):
    """Last valid label per ID wins; errors remain missing."""
    result = {}
    for row in read_jsonl(path):
        if "error" not in row and row.get("relation") in RELATIONS:
            result[row["id"]] = row["relation"]
    return result


def cohen_kappa(a, b):
    """Unweighted Cohen kappa for categorical values, calculated with numpy."""
    if len(a) != len(b):
        raise ValueError("label arrays must have equal length")
    if not a:
        return None
    categories = sorted(set(a) | set(b))
    lookup = {value: i for i, value in enumerate(categories)}
    matrix = np.zeros((len(categories), len(categories)), dtype=np.float64)
    for x, y in zip(a, b):
        matrix[lookup[x], lookup[y]] += 1
    n = matrix.sum()
    observed = np.trace(matrix) / n
    expected = np.dot(matrix.sum(axis=0), matrix.sum(axis=1)) / (n * n)
    return float((observed - expected) / (1 - expected)) if expected != 1 else None


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [center - half, center + half]


def arm_shares(rows, values):
    by_arm = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row["id"])
    result = {}
    for arm, ids in sorted(by_arm.items()):
        valid = [values[i] for i in ids if i in values]
        result[arm] = {"expected": len(ids), "labeled": len(valid), "missing": len(ids) - len(valid)}
        for relation in RELATIONS:
            n = len(valid)
            count = sum(value == relation for value in valid)
            result[arm][relation] = {"count": count, "share": count / n if n else None,
                                     "wilson95": wilson(count, n)}
    return result


def paired_arm_kappa(rows, values, left_arm, right_arm):
    left, right = {}, {}
    for row in rows:
        key = (row.get("layer"), row.get("neuron"))
        if row["arm"] == left_arm and row["id"] in values:
            left[key] = values[row["id"]]
        if row["arm"] == right_arm and row["id"] in values:
            right[key] = values[row["id"]]
    keys = sorted(set(left) & set(right))
    return cohen_kappa([left[k] for k in keys], [right[k] for k in keys]), len(keys)


def prediction(verdict, value, threshold=None):
    return {"true": bool(verdict), "value": value, "threshold": threshold}


def compute_predictions(counts, rows, ds, luna):
    total = int(counts.get("both_pole_total", 0))
    null_mean = float(counts.get("rotation_null_mean", 0))
    per_layer = counts.get("both_pole_by_layer", [])
    poles = counts.get("persona_poles_by_layer", [])
    early_den, late_den = sum(poles[:18]), sum(poles[18:36])
    early = 2 * sum(per_layer[:18]) / early_den if early_den else None
    late = 2 * sum(per_layer[18:36]) / late_den if late_den else None
    results = {
        "135.1": prediction(total >= 2 * null_mean, {"both_pole_total": total, "rotation_null_mean": null_mean}, "total >= 2 * null mean"),
        "135.2": prediction(late is not None and early is not None and late > early,
                             {"L0_17_share": early, "L18_35_share": late}, "late share > early share"),
    }
    if total < 100:
        for key in ("135.3", "135.4", "135.5", "135.6", "135.7"):
            results[key] = {"status": "too_few_both_pole_neurons", "both_pole_total": total}
        return results
    shares = arm_shares(rows, ds)
    def share(arm, relation):
        return shares.get(arm, {}).get(relation, {}).get("share")
    u = share("persona", "unrelated")
    ps, ss = share("persona", "same"), share("persona_shuffled", "same")
    po, so = share("persona", "opposite"), share("persona_shuffled", "opposite")
    fs, fss = share("fires", "same"), share("fires_shuffled", "same")
    results["135.3"] = prediction(u is not None and u >= .5, u, ">= 0.50")
    results["135.4"] = prediction(ps is not None and ss is not None and ps - ss >= .10,
                                  {"persona_same": ps, "persona_shuffled_same": ss}, "difference >= 0.10")
    results["135.5"] = prediction(po is not None and so is not None and po - so <= .10,
                                  {"persona_opposite": po, "persona_shuffled_opposite": so}, "difference <= 0.10")
    results["135.6"] = prediction(fs is not None and fss is not None and fs - fss <= .05,
                                  {"fires_same": fs, "fires_shuffled_same": fss}, "difference <= 0.05")
    kappa, n = paired_arm_kappa(rows, ds, "persona", "fires")
    results["135.7"] = prediction(kappa is not None and kappa <= .2, {"kappa": kappa, "paired_neurons": n}, "kappa <= 0.20")
    luna_shares = arm_shares(rows, luna)
    def lshare(arm, rel):
        return luna_shares.get(arm, {}).get(rel, {}).get("share")
    lu, lps, lss = lshare("persona", "unrelated"), lshare("persona", "same"), lshare("persona_shuffled", "same")
    lpo, lso = lshare("persona", "opposite"), lshare("persona_shuffled", "opposite")
    lfs, lfss = lshare("fires", "same"), lshare("fires_shuffled", "same")
    lkappa, ln = paired_arm_kappa(rows, luna, "persona", "fires")
    luna_verdicts = {
        "135.3": lu >= .5 if lu is not None else None,
        "135.4": lps - lss >= .10 if lps is not None and lss is not None else None,
        "135.5": lpo - lso <= .10 if lpo is not None and lso is not None else None,
        "135.6": lfs - lfss <= .05 if lfs is not None and lfss is not None else None,
        "135.7": lkappa <= .2 if lkappa is not None else None,
    }
    results["luna_subset"] = {"verdicts": luna_verdicts,
                              "values": {"135.3": lu, "135.4": [lps, lss], "135.5": [lpo, lso],
                                         "135.6": [lfs, lfss], "135.7": {"kappa": lkappa, "paired_neurons": ln}}}
    return results


def known_expected(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {}
    for i, row in enumerate(data):
        import hashlib
        digest = hashlib.sha256(json.dumps([i, row["a"], row["b"]], ensure_ascii=False).encode()).hexdigest()[:20]
        expected["known-" + digest] = row["relation"]
    return expected


def score(results_dir):
    results_dir = Path(results_dir)
    counts = json.loads((results_dir / "counts.json").read_text(encoding="utf-8"))
    rows = read_jsonl(results_dir / "pairs.jsonl")
    ds = labels(results_dir / "labels-ds.jsonl")
    luna = labels(results_dir / "labels-luna.jsonl")
    known = known_expected(Path(__file__).with_name("known135.json"))
    known_ds, known_luna = labels(results_dir / "known-ds.jsonl"), labels(results_dir / "known-luna.jsonl")
    known_scores = {}
    for name, answers in (("ds", known_ds), ("luna", known_luna)):
        correct = sum(answers.get(i) == rel for i, rel in known.items())
        answered = sum(i in answers for i in known)
        known_scores[name] = {"correct": correct, "total": len(known), "answered": answered,
                              "missing": len(known) - answered, "accuracy": correct / len(known) if known else None}
    shuffled = [r for r in rows if r["arm"] == "persona_shuffled"]
    shuffled_labels = [ds[r["id"]] for r in shuffled if r["id"] in ds]
    shuffled_same = sum(x == "same" for x in shuffled_labels)
    gate1 = all(known_scores[name]["correct"] >= 11 for name in ("ds", "luna"))
    gate2 = bool(shuffled_labels) and shuffled_same / len(shuffled_labels) <= .10
    overlap = sorted(set(ds) & set(luna))
    kappa = cohen_kappa([ds[i] for i in overlap], [luna[i] for i in overlap])
    gate3 = kappa is not None and kappa >= .60
    gates = {
        "known_examples": {"rule": "DeepSeek and Luna each get at least 11 of 12 right", "pass": gate1, "readers": known_scores},
        "shuffled_persona_same": {"rule": "DeepSeek calls at most 10% of shuffled persona pairs same", "pass": gate2,
                                  "same": shuffled_same, "labeled": len(shuffled_labels),
                                  "share": shuffled_same / len(shuffled_labels) if shuffled_labels else None},
        "second_reader_kappa": {"rule": "three-way Cohen kappa on shared 15% is at least 0.60", "pass": gate3,
                                 "kappa": kappa, "shared_rows": len(overlap)},
    }
    predictions = compute_predictions(counts, rows, ds, luna)
    report = {"gates": gates, "both_pole_total": counts.get("both_pole_total"),
              "arm_shares": {"ds": arm_shares(rows, ds), "luna": arm_shares(rows, luna)},
              "missing_rows": {"ds": {arm: data["missing"] for arm, data in arm_shares(rows, ds).items()},
                               "luna": {arm: data["missing"] for arm, data in arm_shares(rows, luna).items()}},
              "reader_agreement": {"kappa": kappa, "shared_rows": len(overlap)},
              "predictions": predictions}
    (results_dir / "score135.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Part 135 score", "", "## Gates", ""]
    for name, gate in gates.items():
        lines.append(f"- **{name}: {'PASS' if gate['pass'] else 'FAIL'}** — {gate['rule']}.")
        if name == "known_examples":
            lines.append(f"  DeepSeek {known_scores['ds']['correct']}/12; Luna {known_scores['luna']['correct']}/12 (missing count as incorrect).")
        elif name == "shuffled_persona_same":
            lines.append(f"  {shuffled_same}/{len(shuffled_labels)} same ({gate['share'] if gate['share'] is not None else 'n/a'}).")
        else:
            lines.append(f"  kappa={kappa}; shared rows={len(overlap)}.")
    lines += ["", "## Three-way shares (Wilson 95% intervals)", ""]
    for reader in ("ds", "luna"):
        lines.append(f"### {reader}")
        for arm, data in report["arm_shares"][reader].items():
            parts = []
            for rel in RELATIONS:
                cell = data[rel]
                ci = cell["wilson95"]
                parts.append(f"{rel} {cell['count']}/{data['labeled']} ({cell['share']:.3f}, CI {ci[0]:.3f}–{ci[1]:.3f})" if ci[0] is not None else f"{rel} 0/n/a")
            lines.append(f"- **{arm}**: " + "; ".join(parts) + f"; missing {data['missing']}/{data['expected']}.")
    lines += ["", "## Predictions", ""]
    for key in ("135.1", "135.2", "135.3", "135.4", "135.5", "135.6", "135.7"):
        item = predictions[key]
        status = item.get("status", "true" if item.get("true") else "false")
        lines.append(f"- **{key}: {status}** — {item}")
    if "luna_subset" in predictions:
        lines += ["", "Luna subset verdicts for 135.3–135.7: " + json.dumps(predictions["luna_subset"]["verdicts"], sort_keys=True)]
    if int(counts.get("both_pole_total", 0)) < 100:
        lines += ["", "Reading rule: because fewer than 100 both-pole neurons were found, 135.3–135.7 are too few and not scored."]
    lines += ["", "Shares use successfully labeled rows; error and missing rows are counted separately and shown above."]
    (results_dir / "SCORE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote score135.json and SCORE.md; DeepSeek/Luna overlap {len(overlap)}, kappa {kappa}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    score(parser.parse_args().results)
