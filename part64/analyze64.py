#!/usr/bin/env python3
"""Analyze the registered Part 64 four-arm comparison and paired bootstrap."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
BASIN = HERE.parent
DEFAULT_ROOT = BASIN / "results" / "part64"
ARMS = ("haiku-1x", "haiku-3x", "opus-1x", "opus-3x")
TYPES = ("neuron", "control")
PREDICTIONS = {
    "64a": "Haiku-3x raises the real pass rate over Haiku-1x by at least 5 points.",
    "64b": "Haiku-3x raises the gap over Haiku-1x by at least 5 points (more examples make a hypothesis more specific).",
    "64c": "Haiku-3x raises the control pass rate by at most 3 points over Haiku-1x.",
    "64d": "Opus-1x has a gap at least 5 points above Haiku-1x.",
    "64e": "Opus gains more from 3x than Haiku does (interaction on the gap at least 3 points).",
    "64f": "Haiku-1x reproduces Part 60's M1 within 6 points on both real and control pass rates.",
}


class AnalysisError(RuntimeError):
    pass


def canonical_arm(value: Any) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized not in ARMS:
        raise AnalysisError(f"unknown Part 64 arm: {value!r}")
    return normalized


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise AnalysisError(f"{path}:{line_no} is not an object")
            rows.append(row)
    return rows


def extract_part60_baseline(value: Any) -> dict[str, float]:
    """Read Part 60's recorded dev M1 full-rule rates from analysis-60.json."""
    analyses = value.get("analyses") if isinstance(value, dict) else None
    if not isinstance(analyses, list):
        raise AnalysisError("Part 60 analysis must contain an 'analyses' list")
    dev = next((item for item in analyses if isinstance(item, dict) and item.get("split") == "dev"), None)
    if not isinstance(dev, dict):
        raise AnalysisError("Part 60 analysis has no dev split; 64f requires the dev M1 baseline")
    try:
        by_type = dev["metrics"]["M1"]["by_type"]
        rates = {"neuron": float(by_type["neuron"]["verified_rate"]),
                 "control": float(by_type["control"]["verified_rate"])}
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError("Part 60 dev analysis is missing M1 full-rule rates by type") from exc
    if any(not math.isfinite(rate) or not 0.0 <= rate <= 1.0 for rate in rates.values()):
        raise AnalysisError("Part 60 M1 full-rule rates must be finite values from 0 to 1")
    return rates


def _percentile_interval(draws: np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def _delta_record(estimate: float, draws: np.ndarray, n: int) -> dict[str, Any]:
    return {"estimate": float(estimate), "ci95": _percentile_interval(draws), "n": int(n)}


def analyze_rows(rows: list[dict[str, Any]], part60_baseline: dict[str, float], *,
                 expected_per_type: int | None = 250, seed: int = 6403,
                 resamples: int = 2000) -> dict[str, Any]:
    """Analyze complete, paired rows. Failed/unparsed units remain failures in the denominator."""
    if resamples < 1:
        raise AnalysisError("resamples must be positive")
    baseline = {typ: float(part60_baseline[typ]) for typ in TYPES}
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in baseline.values()):
        raise AnalysisError("baseline rates must be finite values from 0 to 1")

    keyed: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in ARMS}
    for index, row in enumerate(rows, 1):
        try:
            arm = canonical_arm(row.get("arm"))
            unit_key = str(row["unit_key"])
            typ = str(row["type"])
        except (KeyError, AnalysisError) as exc:
            raise AnalysisError(f"scored row {index} lacks a valid arm/unit/type") from exc
        if not unit_key or typ not in TYPES:
            raise AnalysisError(f"scored row {index} has invalid unit_key or type")
        if unit_key in keyed[arm]:
            raise AnalysisError(f"duplicate scored row for {arm}/{unit_key}")
        passed = row.get("pass")
        if not isinstance(passed, bool):
            raise AnalysisError(f"scored row {index} must contain boolean 'pass'")
        keyed[arm][unit_key] = {"type": typ, "pass": passed,
                                "parse_status": str(row.get("parse_status", "unknown")),
                                "score_status": str(row.get("score_status", "unknown"))}

    reference_keys = set(keyed[ARMS[0]])
    if not reference_keys:
        raise AnalysisError("no scored Part 64 units were supplied")
    for arm in ARMS:
        if set(keyed[arm]) != reference_keys:
            missing = sorted(reference_keys - set(keyed[arm]))
            extra = sorted(set(keyed[arm]) - reference_keys)
            raise AnalysisError(f"{arm} unit set differs from Haiku-1x; missing={missing[:3]} extra={extra[:3]}")
    for key in reference_keys:
        types = {keyed[arm][key]["type"] for arm in ARMS}
        if len(types) != 1:
            raise AnalysisError(f"unit type changes across arms for {key}")
    unit_keys = {typ: sorted(key for key in reference_keys if keyed[ARMS[0]][key]["type"] == typ)
                 for typ in TYPES}
    if any(not unit_keys[typ] for typ in TYPES):
        raise AnalysisError("analysis requires both neuron and control units")
    if expected_per_type is not None:
        wrong = {typ: len(unit_keys[typ]) for typ in TYPES if len(unit_keys[typ]) != expected_per_type}
        if wrong:
            raise AnalysisError(f"expected {expected_per_type} units per type, got {wrong}")

    values = {
        arm: {typ: np.asarray([keyed[arm][key]["pass"] for key in unit_keys[typ]], dtype=np.float64)
              for typ in TYPES}
        for arm in ARMS
    }
    arm_summary = {}
    for arm in ARMS:
        real_rate = float(values[arm]["neuron"].mean())
        control_rate = float(values[arm]["control"].mean())
        arm_summary[arm] = {
            "real_pass_rate": real_rate,
            "control_pass_rate": control_rate,
            "gap": real_rate - control_rate,
            "n_real": len(unit_keys["neuron"]),
            "n_control": len(unit_keys["control"]),
            "n_real_parse_ok": sum(keyed[arm][key]["parse_status"] == "ok" for key in unit_keys["neuron"]),
            "n_control_parse_ok": sum(keyed[arm][key]["parse_status"] == "ok" for key in unit_keys["control"]),
            "n_score_failures": sum(keyed[arm][key]["score_status"] not in ("ok", "unknown")
                                     for key in reference_keys),
        }

    rng = np.random.default_rng(seed)
    draws_by_type = {typ: rng.integers(0, len(unit_keys[typ]), size=(resamples, len(unit_keys[typ])))
                     for typ in TYPES}
    h1 = "haiku-1x"
    comparisons: dict[str, Any] = {}
    for arm in ARMS:
        if arm == h1:
            continue
        real_delta_draws = values[arm]["neuron"][draws_by_type["neuron"]].mean(axis=1) - values[h1]["neuron"][draws_by_type["neuron"]].mean(axis=1)
        control_delta_draws = values[arm]["control"][draws_by_type["control"]].mean(axis=1) - values[h1]["control"][draws_by_type["control"]].mean(axis=1)
        real_delta = float(values[arm]["neuron"].mean() - values[h1]["neuron"].mean())
        control_delta = float(values[arm]["control"].mean() - values[h1]["control"].mean())
        comparisons[arm] = {
            "comparator": h1,
            "real_pass_rate_delta": _delta_record(real_delta, real_delta_draws, len(unit_keys["neuron"])),
            "control_pass_rate_delta": _delta_record(control_delta, control_delta_draws, len(unit_keys["control"])),
            "gap_delta": _delta_record(real_delta - control_delta, real_delta_draws - control_delta_draws,
                                        len(unit_keys["neuron"]) + len(unit_keys["control"])),
        }

    op_gain_real_draws = (values["opus-3x"]["neuron"][draws_by_type["neuron"]].mean(axis=1)
                          - values["opus-1x"]["neuron"][draws_by_type["neuron"]].mean(axis=1))
    op_gain_control_draws = (values["opus-3x"]["control"][draws_by_type["control"]].mean(axis=1)
                             - values["opus-1x"]["control"][draws_by_type["control"]].mean(axis=1))
    h_gain_real_draws = (values["haiku-3x"]["neuron"][draws_by_type["neuron"]].mean(axis=1)
                         - values["haiku-1x"]["neuron"][draws_by_type["neuron"]].mean(axis=1))
    h_gain_control_draws = (values["haiku-3x"]["control"][draws_by_type["control"]].mean(axis=1)
                            - values["haiku-1x"]["control"][draws_by_type["control"]].mean(axis=1))
    interaction_draws = (op_gain_real_draws - op_gain_control_draws) - (h_gain_real_draws - h_gain_control_draws)
    op_gain = arm_summary["opus-3x"]["gap"] - arm_summary["opus-1x"]["gap"]
    h_gain = arm_summary["haiku-3x"]["gap"] - arm_summary["haiku-1x"]["gap"]
    interaction = float(op_gain - h_gain)

    def prediction(pid: str, *, estimate: float | None, ci95: list[float] | None,
                   threshold: float, met: bool | None, note: str = "") -> dict[str, Any]:
        return {"id": pid, "prediction": PREDICTIONS[pid], "estimate": estimate,
                "ci95": ci95, "threshold": threshold, "met": met, "note": note}

    haiku3 = comparisons["haiku-3x"]
    opus1 = comparisons["opus-1x"]
    predictions = [
        prediction("64a", estimate=haiku3["real_pass_rate_delta"]["estimate"],
                   ci95=haiku3["real_pass_rate_delta"]["ci95"], threshold=0.05,
                   met=haiku3["real_pass_rate_delta"]["estimate"] >= 0.05),
        prediction("64b", estimate=haiku3["gap_delta"]["estimate"],
                   ci95=haiku3["gap_delta"]["ci95"], threshold=0.05,
                   met=haiku3["gap_delta"]["estimate"] >= 0.05),
        prediction("64c", estimate=haiku3["control_pass_rate_delta"]["estimate"],
                   ci95=haiku3["control_pass_rate_delta"]["ci95"], threshold=0.03,
                   met=haiku3["control_pass_rate_delta"]["estimate"] <= 0.03,
                   note="Prediction is an upper bound on the observed increase."),
        prediction("64d", estimate=opus1["gap_delta"]["estimate"],
                   ci95=opus1["gap_delta"]["ci95"], threshold=0.05,
                   met=opus1["gap_delta"]["estimate"] >= 0.05),
        {"id": "64e", "prediction": PREDICTIONS["64e"], "estimate": interaction,
         "ci95": _percentile_interval(interaction_draws), "threshold": 0.03,
         "met": None, "directional_match": interaction >= 0.03,
         "note": "Directional estimate only; the registered design calls 64e underpowered and not a finding."},
        {"id": "64f", "prediction": PREDICTIONS["64f"],
         "estimate": {typ: arm_summary[h1]["real_pass_rate" if typ == "neuron" else "control_pass_rate"] - baseline[typ]
                      for typ in TYPES},
         "ci95": None, "threshold": 0.06,
         "met": all(abs(arm_summary[h1]["real_pass_rate" if typ == "neuron" else "control_pass_rate"] - baseline[typ]) <= 0.06
                    for typ in TYPES),
         "note": "Compared with Part 60 dev M1 full-rule pass rates."},
    ]
    advancement = {}
    for arm in ARMS:
        if arm == h1:
            advancement[arm] = {"advances": False, "reason": "Haiku-1x is the registered comparison baseline."}
            continue
        gap = comparisons[arm]["gap_delta"]
        advancement[arm] = {"advances": gap["estimate"] >= 0.05 and gap["ci95"][0] > 0.0,
                            "gap_delta": gap["estimate"], "ci95": gap["ci95"],
                            "rule": "gap lead >= 0.05 and paired bootstrap 95% interval lower bound > 0"}

    return {
        "split": "dev", "arms": arm_summary, "comparisons_vs_haiku_1x": comparisons,
        "interaction_opus_gain_minus_haiku_gain": {"estimate": interaction,
            "ci95": _percentile_interval(interaction_draws), "seed": seed, "resamples": resamples},
        "predictions": predictions, "advancement": advancement,
        "part60_m1_dev_baseline": baseline,
        "bootstrap": {"seed": seed, "resamples": resamples,
                      "unit_paired": True, "stratified_by": list(TYPES)},
        "denominator_rule": "all registered units remain in denominators; parse or score failures are non-passes",
        "decision_rule": "an arm advances only if its full-rule gap exceeds Haiku-1x by at least 5 percentage points and the unit-paired bootstrap 95% interval is above zero",
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = ["# Part 64 analysis", "", "| Arm | Real pass rate | Control pass rate | Gap | n real | n control |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        row = result["arms"][arm]
        lines.append(f"| {arm} | {_pct(row['real_pass_rate'])} | {_pct(row['control_pass_rate'])} | {_pct(row['gap'])} | {row['n_real']} | {row['n_control']} |")
    lines.extend(["", "## Predictions", "", "| ID | Registered prediction | Estimate | 95% paired bootstrap CI | Result |",
                  "|---|---|---:|---:|---|"])
    for item in result["predictions"]:
        if isinstance(item["estimate"], dict):
            est = ", ".join(f"{key}: {value:+.1%}" for key, value in item["estimate"].items())
        else:
            est = "—" if item["estimate"] is None else f"{item['estimate']:+.1%}"
        ci = "—" if item.get("ci95") is None else f"[{item['ci95'][0]:+.1%}, {item['ci95'][1]:+.1%}]"
        status = ("direction only: match" if item.get("directional_match") else "direction only: miss") if item["id"] == "64e" else ("unknown" if item["met"] is None else ("met" if item["met"] else "not met"))
        lines.append(f"| {item['id']} | {item['prediction']} | {est} | {ci} | {status} |")
    lines.extend(["", "## Decision rule", "", result["decision_rule"], "",
                  "Part 60's test split remains closed unless an arm advances.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_ROOT / "scored.jsonl")
    parser.add_argument("--part60-analysis", type=Path,
                        default=BASIN / "results" / "part60" / "analysis" / "analysis-60.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT / "analysis-64.json")
    parser.add_argument("--seed", type=int, default=6403)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)
    try:
        baseline = extract_part60_baseline(json.loads(args.part60_analysis.read_text(encoding="utf-8")))
        result = analyze_rows(read_jsonl(args.input), baseline, seed=args.seed, resamples=args.resamples)
        _write_json(args.out, result)
        md = args.out.with_suffix(".md")
        write_markdown(md, result)
        print(f"wrote {args.out} and {md}")
        return 0
    except (AnalysisError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
