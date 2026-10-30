#!/usr/bin/env python3
"""Score Part 126's registered predictions and gates from saved raw files."""
from __future__ import annotations

import argparse
import json
import math
import os
import string
import sys
import unicodedata
from pathlib import Path

import numpy as np

LAYERS = (11, 15)
HEADS = 16
REGIONS = ("role", "template", "body", "question", "options", "other")
REGION_ROLE = 0
COPY_A = 0
COPY_U = 1
PERMUTATIONS = 10000
PERMUTATION_SEED = 12602


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    nonblank = [(i, line) for i, line in enumerate(lines) if line.strip()]
    result = []
    for pos, (line_no, line) in enumerate(nonblank):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if pos == len(nonblank) - 1:
                print(f"LOG: skipping torn final JSONL line {line_no + 1} in {path}: {exc}", file=sys.stderr)
                continue
            raise ValueError(f"invalid JSONL at {path}:{line_no + 1}") from exc
    return result


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def _spearman(x, y):
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt(np.dot(rx, rx) * np.dot(ry, ry)))
    return float(np.dot(rx, ry) / denom) if denom else float("nan")


def _head_key(layer, head):
    return f"L{layer}h{head}"


def _drop_data(part83):
    singles = part83.get("singles", {})
    values = {}
    for layer in LAYERS:
        for head in range(HEADS):
            key = _head_key(layer, head)
            row = singles.get(key)
            if row is None or "drop" not in row or "drop_ci95" not in row:
                raise ValueError(f"Part 83 results are missing {key} drop data")
            ci = row["drop_ci95"]
            if len(ci) != 2:
                raise ValueError(f"Part 83 {key} must have a two-value 95% interval")
            drop = float(row["drop"])
            low, high = float(ci[0]), float(ci[1])
            matters = abs(drop) >= 0.05 and (low > 0.0 or high < 0.0)
            values[key] = {"drop": drop, "absolute_drop": abs(drop), "ci95": [low, high], "matters": matters}
    return values


def _gates(run, lens, region_mass, hidden):
    recorded = run.get("gates", {})
    recompute = recorded.get("recomputation", {})
    recompute_error = float(recompute.get("max_abs_difference", float("inf")))
    recompute_tolerance = float(recompute.get("tolerance", 0.05))
    region_sum_error = float(np.max(np.abs(region_mass.astype(np.float32).sum(axis=-1) - 1.0)))
    region_tolerance = float(recompute.get("region_mass_tolerance", 0.01))
    gates = {
        "recomputation": {
            "passed": bool(recompute_error <= recompute_tolerance and region_sum_error <= region_tolerance),
            "max_abs_difference": recompute_error, "tolerance": recompute_tolerance,
            "raw_region_mass_sum_max_abs_difference": region_sum_error,
            "region_mass_tolerance": region_tolerance,
        }
    }
    zero = recorded.get("zero_effect", {})
    zero_max = zero.get("max_abs_difference")
    gates["zero_effect"] = {
        "passed": bool(zero.get("bitwise_equal") is True and zero_max == 0.0),
        "bitwise_equal": bool(zero.get("bitwise_equal", False)),
        "max_abs_difference": zero_max,
    }
    positive = recorded.get("positive_control", {})
    positive_rate = float(positive.get("max_rate_over_uniform", float("nan")))
    gates["positive_control"] = {
        "passed": bool(math.isfinite(positive_rate) and positive_rate >= 3.0),
        "max_rate_over_uniform": positive_rate, "head": positive.get("head"),
        "threshold": "> 1.0",
    }
    shape = lens.get("gates", {}).get("lens_shape", {})
    shape_list = shape.get("lens_shape")
    expected = shape.get("expected_shape")
    shape_ok = bool(shape.get("passed"))
    if shape_list is not None and expected is not None:
        shape_ok = shape_ok and shape_list == expected and len(shape_list) == 3
        shape_ok = shape_ok and shape_list[1:] == [hidden, hidden]
        shape_ok = shape_ok and all(layer < shape_list[0] for layer in LAYERS)
    gates["lens_shape"] = {
        "passed": shape_ok, "lens_shape": shape_list, "expected_shape": expected,
        "capture_hidden_size": hidden, "selected_layers": list(LAYERS),
    }
    return gates


def _prediction_row(status, evidence, reason=None):
    return {"status": status, "evidence": evidence, **({"reason": reason} if reason else {})}


def _clean_token(token):
    cleaned = token.strip()
    while cleaned and (cleaned[0] in string.punctuation or unicodedata.category(cleaned[0]).startswith("P")):
        cleaned = cleaned[1:]
    while cleaned and (cleaned[-1] in string.punctuation or unicodedata.category(cleaned[-1]).startswith("P")):
        cleaned = cleaned[:-1]
    return cleaned.lower()


def _score_predictions(region_mass, rows, part83, lens, vector_data):
    drops = _drop_data(part83)
    head_order = [_head_key(layer, head) for layer in LAYERS for head in range(HEADS)]
    absolute_drops = np.asarray([drops[key]["absolute_drop"] for key in head_order], dtype=np.float64)
    role_mass = region_mass[:, :, :, REGION_ROLE].astype(np.float64)
    copy_labels = np.asarray([row["rendering"] for row in rows])
    pooled_role = role_mass.mean(axis=0)
    assistant_role = role_mass[copy_labels == "A"].mean(axis=0)
    user_role = role_mass[copy_labels == "U"].mean(axis=0)
    pooled_vec = np.asarray([pooled_role[layer_pos, head] for layer_pos in range(2) for head in range(HEADS)])
    rho = _spearman(pooled_vec, absolute_drops)
    rng = np.random.default_rng(PERMUTATION_SEED)
    null = np.empty(PERMUTATIONS, dtype=np.float64)
    for i in range(PERMUTATIONS):
        null[i] = _spearman(pooled_vec, rng.permutation(absolute_drops))
    percentile95 = float(np.quantile(null, 0.95, method="linear"))
    exceedances = int(np.count_nonzero(null >= rho))
    p_empirical = (exceedances + 1) / (PERMUTATIONS + 1)
    pred1 = _prediction_row(
        "held" if rho >= 0.40 and rho > percentile95 else "failed",
        {"spearman_rho": rho, "threshold": 0.40, "shuffle_95th_percentile": percentile95,
         "above_shuffle_95th": bool(rho > percentile95), "shuffle_exceedances": exceedances,
         "empirical_p": p_empirical, "permutations": PERMUTATIONS, "seed": PERMUTATION_SEED,
         "attention_mass_by_head": {key: float(pooled_vec[i]) for i, key in enumerate(head_order)},
         "assistant_copy_mass_by_head": {
             _head_key(layer, head): float(assistant_role[li, head])
             for li, layer in enumerate(LAYERS) for head in range(HEADS)},
         "user_copy_mass_by_head": {
             _head_key(layer, head): float(user_role[li, head])
             for li, layer in enumerate(LAYERS) for head in range(HEADS)},
         "absolute_drop_by_head": {key: float(absolute_drops[i]) for i, key in enumerate(head_order)},
         "shuffle_correlations": null.tolist()},
    )

    assistant_order = sorted(
        [(_head_key(layer, head), float(assistant_role[li, head]), layer, head)
         for li, layer in enumerate(LAYERS) for head in range(HEADS)],
        key=lambda row: (-row[1], row[2], row[3]),
    )
    assistant_rank = next(i + 1 for i, row in enumerate(assistant_order) if row[0] == "L11h5")
    pred2 = _prediction_row("held" if assistant_rank <= 5 else "failed", {
        "head": "L11h5", "rank": assistant_rank, "threshold": "top 5",
        "assistant_copy_role_mass": dict((key, mass) for key, mass, _, _ in assistant_order),
    })

    readout = lens.get("readouts", {}).get("L11h5", {}).get("A", {}).get("role_token", {})
    top_tokens = readout.get("top20_written_tokens", [])
    first_person = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
    second_person = {"you", "your", "yours"}
    token_matches = []
    for token in top_tokens:
        cleaned = _clean_token(str(token))
        family = "first_person" if cleaned in first_person else (
            "second_person" if cleaned in second_person else ("assistant_role" if cleaned == "assistant" else None)
        )
        if family:
            token_matches.append({"token": token, "normalized": cleaned, "family": family})
    pred3 = _prediction_row("held" if token_matches else "failed", {
        "head": "L11h5", "copy": "A", "top20_written_tokens": top_tokens,
        "matching_tokens": token_matches,
    })

    vectors = vector_data
    role_writes = vectors["role_writes"].astype(np.float64)
    if role_writes.shape[:3] != (2, HEADS, 2):
        raise ValueError(f"unexpected role write tensor shape {role_writes.shape}")
    cosine_by_head = {}
    for layer_pos, layer in enumerate(LAYERS):
        for head in range(HEADS):
            a = role_writes[layer_pos, head, COPY_A]
            u = role_writes[layer_pos, head, COPY_U]
            denom = float(np.linalg.norm(a) * np.linalg.norm(u))
            cosine_by_head[_head_key(layer, head)] = float(np.dot(a, u) / denom) if denom else None
    matters = [key for key in head_order if drops[key]["matters"]]
    if len(matters) != 20:
        raise ValueError(f"Part 83 mattering-head rule selected {len(matters)} heads, expected 20")
    below = [key for key in matters if cosine_by_head[key] is not None and cosine_by_head[key] < 0.90]
    pred4 = _prediction_row("held" if len(below) >= 10 else "failed", {
        "mattering_heads": matters, "threshold": "at least 10 of 20 have cosine < 0.90",
        "below_threshold_count": len(below), "heads_below_threshold": below,
        "cosine_by_head": cosine_by_head,
    })

    overlap = {}
    for head_key, copies in lens.get("readouts", {}).items():
        overlap[head_key] = {}
        for copy in ("A", "U"):
            copy_readout = copies.get(copy, {})
            role_tokens = set(copy_readout.get("role_token", {}).get("top20_written_tokens", []))
            body_tokens = set(copy_readout.get("random_body_token_control", {}).get("top20_written_tokens", []))
            overlap[head_key][copy] = len(role_tokens & body_tokens)
    overlap_values = [v[copy] for v in overlap.values() for copy in ("A", "U") if copy in v]
    overlap_summary = {
        "head_copy_pairs": len(overlap_values),
        "mean_shared_top20": float(np.mean(overlap_values)) if overlap_values else None,
        "median_shared_top20": float(np.median(overlap_values)) if overlap_values else None,
        "shared_top20_by_head_copy": overlap,
    }
    return {"126.1": pred1, "126.2": pred2, "126.3": pred3, "126.4": pred4}, {
        "heads": {key: {**drops[key], "assistant_role_mass": float(assistant_role[i // HEADS, i % HEADS]),
                        "user_role_mass": float(user_role[i // HEADS, i % HEADS]),
                        "pooled_role_mass": float(pooled_vec[i]), "role_write_cosine": cosine_by_head[key]}
                   for i, key in enumerate(head_order)},
        "lens_body_token_comparison": overlap_summary,
    }


def _capture_is_complete(run, rows, row_indexes, stub):
    if stub:
        return True
    if (len(rows) != 600 or run.get("body_count") != 100 or run.get("row_count_expected") != 600
            or run.get("row_count_complete") != 600 or len(row_indexes) != 600
            or row_indexes.tolist() != list(range(600))):
        return False
    counts = {}
    for row in rows:
        key = (str(row.get("body_id")), str(row.get("rendering")))
        counts.setdefault(key, []).append(int(row.get("order", -1)))
    body_ids = sorted({str(row.get("body_id")) for row in rows})
    if len(body_ids) != 100 or len(counts) != 200:
        return False
    return all(sorted(orders) == [1, 2, 3] for orders in counts.values())


def run(args):
    reads_dir = Path(args.reads_dir)
    lens_dir = Path(args.lens_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = json.loads((reads_dir / "run.json").read_text(encoding="utf-8"))
    lens = json.loads((lens_dir / "lens126.json").read_text(encoding="utf-8"))
    part83 = json.loads(Path(args.part83).read_text(encoding="utf-8"))
    rows = _read_jsonl(reads_dir / "rows.jsonl")
    with np.load(reads_dir / "reads126.npz", allow_pickle=False) as z:
        region_mass = z["region_mass"].astype(np.float32)
        row_indexes = z["row_indexes"].astype(np.int32)
        hidden = int(z["region_values"].shape[-1])
        region_names = [str(x) for x in z["region_names"].tolist()]
    json_indexes = [int(r["row_index"]) for r in rows]
    if len(rows) == len(row_indexes) - 1 and row_indexes[:len(rows)].tolist() == json_indexes:
        print("LOG: skipped the activation row for the torn final JSONL line", file=sys.stderr)
        row_indexes = row_indexes[:len(rows)]
        region_mass = region_mass[:len(rows)]
    elif len(rows) != len(row_indexes) or json_indexes != row_indexes.tolist():
        raise ValueError("rows JSONL does not match raw activation row indexes")
    if region_names != list(REGIONS) or region_mass.shape != (len(rows), 2, HEADS, len(REGIONS)):
        raise ValueError("raw attention tensor does not match the registered region schema")
    gates = _gates(run, lens, region_mass, hidden)
    gate_failures = [name for name, value in gates.items() if not value["passed"]]
    capture_complete = _capture_is_complete(run, rows, row_indexes, args.stub)
    read_failures = list(gate_failures)
    if not capture_complete:
        read_failures.append("capture incomplete")
    predictions = {}
    descriptive = {"head_region_mass": {}}
    for layer_pos, layer in enumerate(LAYERS):
        for head in range(HEADS):
            key = _head_key(layer, head)
            descriptive["head_region_mass"][key] = {
                copy: {REGIONS[ri]: float(region_mass[np.asarray([r["rendering"] == copy for r in rows]), layer_pos, head, ri].mean())
                       for ri in range(len(REGIONS))}
                for copy in ("A", "U")
            }
    if read_failures:
        predictions = {
            key: _prediction_row("not read", {}, "; ".join(
                (["failed gate(s): " + ", ".join(gate_failures)] if gate_failures else [])
                + (["capture incomplete: expected all 600 rows across 100 bodies"] if not capture_complete else [])
            ))
            for key in ("126.1", "126.2", "126.3", "126.4")
        }
        exploratory = {}
    else:
        vectors_path = lens_dir / lens.get("vectors_file", "lens126-vectors.npz")
        with np.load(vectors_path, allow_pickle=False) as vector_data:
            predictions, exploratory = _score_predictions(region_mass, rows, part83, lens, vector_data)

    result = {
        "design": "PART126-DESIGN-draft.md", "gates": gates,
        "gate_failures": gate_failures, "predictions": predictions,
        "descriptive": descriptive, "exploratory": exploratory,
        "capture_complete": capture_complete, "row_count": len(rows), "region_names": list(REGIONS),
    }
    (out / "score126.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    _write_markdown(out / "SCORE.md", result)
    print(f"wrote {out / 'SCORE.md'} and {out / 'score126.json'}; gates passed: {4 - len(gate_failures)}/4")


def _write_markdown(path: Path, result):
    lines = ["# Part 126 score", "", f"Rows: {result['row_count']}",
             f"Capture complete: {result['capture_complete']}", "", "## Gates", "",
             "| Gate | Result | Evidence |", "| --- | --- | --- |"]
    for name, gate in result["gates"].items():
        evidence = json.dumps({k: v for k, v in gate.items() if k != "passed"}, ensure_ascii=False, sort_keys=True)
        lines.append(f"| {name} | {'pass' if gate['passed'] else 'FAIL'} | `{evidence}` |")
    lines.extend(["", "## Registered predictions", "", "| Prediction | Verdict | Evidence |", "| --- | --- | --- |"])
    for key in ("126.1", "126.2", "126.3", "126.4"):
        row = result["predictions"][key]
        evidence = json.dumps(row.get("evidence", {}), ensure_ascii=False, sort_keys=True)
        if row.get("reason"):
            evidence = row["reason"]
        lines.append(f"| {key} | {row['status']} | `{evidence}` |")
    lines.extend(["", "## Regional attention mass", "", "| Head | Copy | Role | Other template | Body | Question | Options | Other |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for head, copies in result["descriptive"]["head_region_mass"].items():
        for copy in ("A", "U"):
            row = copies[copy]
            lines.append(f"| {head} | {copy} | " + " | ".join(f"{row[r]:.5f}" for r in REGIONS) + " |")
    body_cmp = result.get("exploratory", {}).get("lens_body_token_comparison")
    if body_cmp:
        mean_overlap = body_cmp["mean_shared_top20"]
        median_overlap = body_cmp["median_shared_top20"]
        mean_text = "n/a" if mean_overlap is None else f"{mean_overlap:.3f}"
        median_text = "n/a" if median_overlap is None else f"{median_overlap:.3f}"
        lines.extend(["", "## Lens control", "", "Same-head role-write versus one random copied-body token. The comparison never flips the write's sign.",
                      "", f"Head-copy pairs: {body_cmp['head_copy_pairs']}; mean shared top-20 tokens: {mean_text}; median: {median_text}."])
    if result["gate_failures"] or not result["capture_complete"]:
        lines.extend(["", "Predictions are not read because a gate failed or the 600-row capture is incomplete."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reads-dir", required=True)
    parser.add_argument("--lens-dir", required=True)
    parser.add_argument("--part83", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stub", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
