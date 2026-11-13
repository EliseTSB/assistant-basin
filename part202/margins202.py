#!/usr/bin/env python3
"""Print and save Part 164's published gate, verdict, and answer-length margins."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN_CAP = 80
SHAPE_REPLICATE_ROWS = 200
PILOT_BATCHES = 3
CAP_RATE = 0.30


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def eq_rows(rate_gap: float, rows_per_arm: int = 1600) -> float:
    """Convert a mean-rate distance to an explicitly approximate row equivalent."""
    return abs(rate_gap) * rows_per_arm


def cap_summary(rows: list[dict]) -> dict:
    out = {}
    for arm in ("neuron", "random"):
        selected = [r for r in rows if r.get("stage") == "battery" and r.get("arm") == arm]
        lengths = [int(r["token_count"]) for r in selected if isinstance(r.get("token_count"), (int, float))]
        out[arm] = {
            "answers": len(selected),
            "count_at_80": sum(n == 80 for n in lengths),
            "mean_token_count": sum(lengths) / len(lengths) if lengths else None,
        }
    return out


def build_report(score: dict, rows: list[dict]) -> dict:
    gates = score["gates"]
    pred = score["predictions"]
    row_count = gates["gate1_rows"]["expected_rows"]
    size_rows = []

    # Exact row or unit headroom is used for integer gates. Rate conversions below
    # are labelled equivalents because Part 164 averages per-unit rates.
    g1 = gates["gate1_rows"]
    size_rows.append({"id": "Gate 1", "estimate": f"{g1['rows']} complete rows", "bar": f"{g1['expected_rows']} complete rows",
                      "distance": 0, "unit": "missing/extra rows to fail: 1"})
    g2 = gates["gate2_no_push"]
    for label, key in (("Gate 2 Qwen", "qwen_named"), ("Gate 2 maker", "alibaba_or_tongyi_named")):
        size_rows.append({"id": label, "estimate": g2[key], "bar": g2["minimum_each"],
                          "distance": max(0, g2[key] - g2["minimum_each"]),
                          "unit": "row headroom; one more lost hit than headroom fails"})
    g3 = gates["gate3_recipe"]
    size_rows.append({"id": "Gate 3", "estimate": f"{g3['changed']}/{g3['total']} ({g3['changed_rate']:.6f})",
                      "bar": f"{g3['minimum_rate']:.2f}",
                      "distance": max(0, g3["changed"] - math.ceil(g3["minimum_rate"] * g3["total"])),
                      "unit": "row headroom; changed→unchanged rows to fail: " +
                             str(g3["changed"] - math.ceil(g3["minimum_rate"] * g3["total"]) + 1)})
    g4 = gates["gate4_measure_room"]
    size_rows.append({"id": "Gate 4", "estimate": g4["random_qwen_rate_0.8"], "bar": g4["minimum_rate"],
                      "distance": eq_rows(g4["random_qwen_rate_0.8"] - g4["minimum_rate"]),
                      "unit": "equivalent eligible-answer rows at 1,600 rows/arm"})

    p1 = pred["164.1"]["estimate_ratio"]
    qmeans = score["rates"]["qwen_rate"]
    qn, qr = qmeans["0.8"]["neuron"]["mean"], qmeans["0.8"]["random"]["mean"]
    size_rows.append({"id": "164.1", "estimate": p1, "bar": "< 0.7", "distance": eq_rows(qn - .7 * qr),
                      "unit": "equivalent Qwen-label rows above boundary; 1,600-row denominator"})
    p2 = pred["164.2"]["estimate_ratio"]
    bmeans = score["rates"]["broken_rate"]
    bn, br = bmeans["0.8"]["neuron"]["mean"], bmeans["0.8"]["random"]["mean"]
    size_rows.append({"id": "164.2", "estimate": p2, "bar": ">= 2", "distance": eq_rows(bn - 2 * br),
                      "unit": "equivalent broken-answer rows above boundary; 1,600-row denominator"})
    rates = [bmeans[d]["random"]["mean"] for d in ("0.8", "1.0", "1.2")]
    size_rows.append({"id": "164.3 first step", "estimate": rates[1] - rates[0], "bar": "> 0", "distance": eq_rows(rates[1] - rates[0]),
                      "unit": "equivalent broken-answer rows to equality; 1,600 rows/dose"})
    size_rows.append({"id": "164.3 second step", "estimate": rates[2] - rates[1], "bar": "> 0", "distance": eq_rows(rates[2] - rates[1]),
                      "unit": "equivalent broken-answer rows to equality; 1,600 rows/dose"})
    p4 = pred["164.4"]["estimate_ratio"]
    r4 = qmeans["1.0"]["random"]["mean"]
    size_rows.append({"id": "164.4 upper boundary", "estimate": p4, "bar": "<= 1.3", "distance": eq_rows(qn - 1.3 * r4),
                      "unit": "equivalent Qwen-label rows to upper boundary; 1,600-row denominator"})
    p5 = pred["164.5"]["estimate_ratio_1.2_over_0.8"]
    q12 = qmeans["1.2"]["random"]["mean"]
    size_rows.append({"id": "164.5", "estimate": p5, "bar": "< 0.5", "distance": eq_rows(.5 * qr - q12),
                      "unit": "equivalent Qwen-label rows below boundary; 1,600-row denominator"})
    size_rows.append({"id": "Headline Qwen direction", "estimate": qn - qr, "bar": "< 0", "distance": eq_rows(qr - qn),
                      "unit": "equivalent Qwen-label rows to equality; 1,600 rows/arm"})
    size_rows.append({"id": "Headline broken direction", "estimate": bn - br, "bar": "> 0", "distance": eq_rows(bn - br),
                      "unit": "equivalent broken-answer rows to equality; 1,600 rows/arm"})

    return {"part": 202, "source": "results/part164/score.json and answers.jsonl", "expected_rows": row_count,
            "plan_constants": {"token_cap": TOKEN_CAP, "shape_replicate_rows": SHAPE_REPLICATE_ROWS,
                               "pilot_batches": PILOT_BATCHES, "cap_rate_prediction": CAP_RATE,
                               "cap_rate_thresholds": {arm: math.ceil(CAP_RATE * item["answers"])
                                                       for arm, item in cap_summary(rows).items()}},
            "stage_counts": {stage: sum(r.get("stage") == stage for r in rows)
                             for stage in ("no_push", "recipe", "battery")},
            "gates": gates, "predictions": pred, "distances": size_rows, "answer_lengths": cap_summary(rows),
            "note": "Rate-derived row distances are equivalent counts from the per-unit mean-rate gap at a 1,600-row denominator; they are not a rerun of the scorer under edited labels."}


def markdown(result: dict) -> str:
    lines = ["# Part 202 size margins from published Part 164", "",
             "Computed by `part202/margins202.py` from the published score and answer rows.", "",
             "## Gates and registered verdict distances", "",
             "| Reading | Published estimate | Bar | Distance / headroom |", "| --- | ---: | --- | ---: |"]
    for item in result["distances"]:
        lines.append(f"| {item['id']} | {item['estimate']} | {item['bar']} | {item['distance']:.3f} {item['unit']} |")
    lines += ["", "## Battery answer lengths", "", "| Arm | Answers | At 80-token cap | Mean token_count |", "| --- | ---: | ---: | ---: |"]
    for arm, item in result["answer_lengths"].items():
        lines.append(f"| {arm} | {item['answers']} | {item['count_at_80']} | {item['mean_token_count']:.3f} |")
    lines += ["", "## Row counts", "", "| Stage | Rows |", "| --- | ---: |"]
    for stage, count in result["stage_counts"].items():
        lines.append(f"| {stage} | {count} |")
    lines += ["| Total | " + str(sum(result["stage_counts"].values())) + " |", "", f"Note: {result['note']}", ""]
    lines += [f"Planned replicate size: {result['plan_constants']['shape_replicate_rows']} rows; "
              f"pilot: first {result['plan_constants']['pilot_batches']} pushed batches; "
              f"cap threshold: {result['plan_constants']['token_cap']} tokens.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/part164")
    parser.add_argument("--out", type=Path, default=ROOT / "results/part202")
    args = parser.parse_args(argv)
    score = json.loads((args.results / "score.json").read_text(encoding="utf-8"))
    rows = load_rows(args.results / "answers.jsonl")
    result = build_report(score, rows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "margins.md").write_text(markdown(result), encoding="utf-8")
    (args.out / "margins.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(markdown(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
