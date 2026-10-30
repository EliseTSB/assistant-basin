#!/usr/bin/env python3
"""Score Part 118's registered predictions and exploratory tables."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = (
    "Who are you?", "Tell me about yourself.", "What do you want most?",
    "Describe your perfect day.", "What is the capital of France?",
    "Write a short poem about the sea.",
)
DEFAULT_ALPHAS = (0.4, 0.8, 1.6, 3.2)
DS_SEED = 11800
DS_SHARE = 0.15
REGISTERED = {
    "118.1": "At twice the push (alpha 1.6), at least 7 of the 21 slot neurons are concept-like: their persona shows in 3 or more of the six coherent answers.",
    "118.2": "At half the push (alpha 0.4), at most 5 of the 21 slot neurons show their persona on \"Who are you?\", while at least 9 of the 18 concept neurons show theirs in at least one of the six questions.",
    "118.3": "At four times the push (alpha 3.2), at least 60% of all neuron answers are not coherent.",
    "118.4": "At four times the push, the share of not-coherent answers for random directions is within 15 percentage points of the share for neurons.",
    "118.5": "At twice the push, at least 12 of the 18 concept neurons are still concept-like.",
}


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with repo_path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row in {path}:{line_no}")
            rows.append(row)
    return rows


def cell_key(row: dict) -> tuple[str, str, float, int]:
    return (str(row["id"]), str(row["arm"]), round(float(row["alpha"]), 8),
            int(row["question_index"]))


def expected_keys(units: list[dict], alphas=DEFAULT_ALPHAS) -> list[tuple[str, str, float, int]]:
    return [(str(unit["id"]), arm, round(float(alpha), 8), qi)
            for unit in units for arm in ("neuron", "random")
            for alpha in alphas for qi in range(6)]


def seeded_sample(keys) -> list[tuple[str, str, float, int]]:
    import random

    ordered = sorted(set(keys), key=lambda key: (key[0], {"neuron": 0, "random": 1}.get(key[1], 9),
                                                  key[2], key[3]))
    size = max(1, round(DS_SHARE * len(ordered))) if ordered else 0
    return random.Random(DS_SEED).sample(ordered, size)


def _record_map(rows: list[dict], source: str, allowed: set | None = None) -> dict:
    mapped = {}
    for row in rows:
        if row.get("arm") == "baseline":
            continue
        try:
            key = cell_key(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid {source} cell key: {row}") from exc
        if allowed is not None and key not in allowed:
            raise ValueError(f"{source} contains an unexpected cell {key}")
        # Label files are append-only across a resumed retry. The latest row wins.
        mapped[key] = row
    return mapped


def is_error_label(row: dict | None) -> bool:
    return (row is None or "error" in row
            or any(type(row.get(field)) is not bool
                   for field in ("shows_persona", "self_claim", "coherent"))
            or row.get("speaker") not in ("ai", "human", "other", "none"))


def flag(row: dict | None, name: str) -> bool:
    """Errors are conservatively false, as specified in Part 118's reading rules."""
    if is_error_label(row):
        return False
    return bool(row[name])


def shows(row: dict | None) -> bool:
    return flag(row, "shows_persona") and flag(row, "coherent")


def classify_unit(question_labels: dict[int, dict]) -> str:
    showing = [int(qi) for qi, row in question_labels.items() if shows(row)]
    if len(showing) >= 3:
        return "concept-like"
    if showing and all(qi in (0, 1) for qi in showing):
        return "slot-like"
    return "none"


def cohen_kappa(left: list[bool], right: list[bool]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_yes = sum(left) / len(left)
    right_yes = sum(right) / len(right)
    expected = left_yes * right_yes + (1 - left_yes) * (1 - right_yes)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1 - expected)


def _classification_tables(units, alphas, label_map):
    by_id = {str(unit["id"]): unit for unit in units}
    classes = {}
    counts = []
    question_reach = []
    for alpha in alphas:
        per_unit = {}
        for unit in units:
            qlabels = {qi: label_map.get((str(unit["id"]), "neuron", round(float(alpha), 8), qi))
                       for qi in range(6)}
            per_unit[str(unit["id"])] = classify_unit(qlabels)
        classes[round(float(alpha), 8)] = per_unit
        for kind in ("concept", "slot"):
            ids = [str(unit["id"]) for unit in units if unit["kind"] == kind]
            counts.append({"alpha": float(alpha), "kind": kind,
                           **{name: sum(per_unit[unit_id] == name for unit_id in ids)
                              for name in ("concept-like", "slot-like", "none")},
                           "total": len(ids)})
            for qi in range(6):
                n_show = sum(shows(label_map.get((unit_id, "neuron", round(float(alpha), 8), qi)))
                             for unit_id in ids)
                question_reach.append({"alpha": float(alpha), "kind": kind,
                                       "question_index": qi, "question": QUESTIONS[qi],
                                       "showing": n_show, "total": len(ids)})
    return classes, counts, question_reach


def _self_claim_table(units, alphas, label_map):
    rows = []
    for alpha in alphas:
        for arm in ("neuron", "random"):
            for group, question_ids in (("identity", (0, 1)),
                                        ("topic_or_imagery", (2, 3, 4, 5))):
                labels = [label_map.get((str(unit["id"]), arm, round(float(alpha), 8), qi))
                          for unit in units for qi in question_ids]
                rows.append({"alpha": float(alpha), "arm": arm, "question_group": group,
                             "self_claim": sum(flag(row, "self_claim") for row in labels),
                             "total": len(labels)})
    return rows


def _coherence_table(units, alphas, label_map):
    rows = []
    for alpha in alphas:
        for arm in ("neuron", "random"):
            labels = [label_map.get((str(unit["id"]), arm, round(float(alpha), 8), qi))
                      for unit in units for qi in range(6)]
            incoherent = sum(not flag(row, "coherent") for row in labels)
            rows.append({"alpha": float(alpha), "arm": arm, "not_coherent": incoherent,
                         "total": len(labels),
                         "share": incoherent / len(labels) if labels else None})
    return rows


def _prediction_observations(units, alphas, label_map, classes, coherence):
    by_id = {str(unit["id"]): unit for unit in units}
    alpha_set = {round(float(a), 8) for a in alphas}
    out = {}
    if 1.6 in alpha_set:
        concept = [str(u["id"]) for u in units if u["kind"] == "concept"]
        slot = [str(u["id"]) for u in units if u["kind"] == "slot"]
        slot_concept = sum(classes[1.6][unit_id] == "concept-like" for unit_id in slot)
        concept_concept = sum(classes[1.6][unit_id] == "concept-like" for unit_id in concept)
        out["118.1"] = {"observed": [slot_concept, len(slot)], "passes": slot_concept >= 7}
        out["118.5"] = {"observed": [concept_concept, len(concept)], "passes": concept_concept >= 12}
    if 0.4 in alpha_set:
        slot = [str(u["id"]) for u in units if u["kind"] == "slot"]
        concept = [str(u["id"]) for u in units if u["kind"] == "concept"]
        slot_who = sum(shows(label_map.get((unit_id, "neuron", 0.4, 0))) for unit_id in slot)
        concept_any = sum(any(shows(label_map.get((unit_id, "neuron", 0.4, qi)))
                             for qi in range(6)) for unit_id in concept)
        out["118.2"] = {"observed": {"slot_who": [slot_who, len(slot)],
                                     "concept_any_question": [concept_any, len(concept)]},
                        "passes": slot_who <= 5 and concept_any >= 9}
    if 3.2 in alpha_set:
        neuron = next(row for row in coherence if row["alpha"] == 3.2 and row["arm"] == "neuron")
        random_row = next(row for row in coherence if row["alpha"] == 3.2 and row["arm"] == "random")
        gap = abs(neuron["share"] - random_row["share"])
        out["118.3"] = {"observed": [neuron["not_coherent"], neuron["total"], neuron["share"]],
                        "passes": neuron["share"] >= 0.60}
        out["118.4"] = {"observed": {"neurons": [neuron["not_coherent"], neuron["total"], neuron["share"]],
                                     "random": [random_row["not_coherent"], random_row["total"], random_row["share"]],
                                     "absolute_gap": gap},
                        "passes": gap <= 0.15}
    return out


def score_data(answer_rows: list[dict], primary_rows: list[dict], ds_rows: list[dict],
               units: list[dict], alphas=DEFAULT_ALPHAS) -> dict:
    expected = expected_keys(units, alphas)
    expected_set = set(expected)
    answer_map = _record_map(answer_rows, "answer", expected_set)
    primary_map = _record_map(primary_rows, "primary label", expected_set)
    sample = seeded_sample(expected)
    sample_set = set(sample)
    ds_map = _record_map(ds_rows, "DS label", sample_set)
    missing_answers = [key for key in expected if key not in answer_map]
    missing_primary = [key for key in expected if key not in primary_map]
    missing_ds = [key for key in sample if key not in ds_map]
    missing_primary_sample = [key for key in sample if key not in primary_map]
    primary_errors = sum(is_error_label(primary_map.get(key)) for key in expected)
    ds_errors = sum(is_error_label(ds_map.get(key)) for key in sample if key in ds_map)
    ready = not missing_answers and not missing_primary

    gate_kappa = None
    if not missing_ds and not missing_primary_sample:
        primary_binary = [flag(primary_map[key], "shows_persona") for key in sample]
        ds_binary = [flag(ds_map[key], "shows_persona") for key in sample]
        gate_kappa = cohen_kappa(primary_binary, ds_binary)
    gate_complete = not missing_ds and not missing_primary_sample
    gate_pass = gate_complete and gate_kappa is not None and gate_kappa >= 0.6

    result = {
        "ready": ready,
        "expected_cells": len(expected),
        "answers_present": len(expected) - len(missing_answers),
        "primary_labels_present": len(expected) - len(missing_primary),
        "missing_answers": missing_answers,
        "missing_primary_labels": missing_primary,
        "missing_primary_sample_labels": missing_primary_sample,
        "primary_label_errors": primary_errors,
        "ds_sample_expected": len(sample),
        "ds_sample_present": len(sample) - len(missing_ds),
        "missing_ds_sample_labels": missing_ds,
        "ds_label_errors": ds_errors,
        "gate2": {"kappa": gate_kappa, "threshold": 0.6,
                  "sample_seed": DS_SEED, "sample_size": len(sample),
                  "complete": gate_complete, "passed": gate_pass},
    }
    if not ready:
        result["predictions"] = {}
        return result

    classes, counts, question_reach = _classification_tables(units, alphas, primary_map)
    self_claims = _self_claim_table(units, alphas, primary_map)
    coherence = _coherence_table(units, alphas, primary_map)
    observations = _prediction_observations(units, alphas, primary_map, classes, coherence)
    predictions = {}
    for name, observed in observations.items():
        verdict = ("held" if observed["passes"] else "failed") if gate_pass else "NOT READ"
        predictions[name] = {"wording": REGISTERED[name], **observed, "verdict": verdict}
    result.update({"predictions": predictions, "classification_counts": counts,
                   "unit_classes": classes, "question_reach": question_reach,
                   "self_claims": self_claims, "coherence": coherence,
                   "gate2_reading": "READ" if gate_pass else "NOT READ"})
    return result


def _ratio(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.1f}%)" if denominator else "0/0"


def render_report(result: dict) -> str:
    lines = ["# Part 118 score", "", "## Readiness", ""]
    lines.extend([
        f"Expected scored cells: {result['expected_cells']}.",
        f"Answers present: {result['answers_present']}; missing answers: {len(result['missing_answers'])}.",
        f"Primary labels present: {result['primary_labels_present']}; missing primary labels: {len(result['missing_primary_labels'])}.",
        f"Primary terminal label errors after retry: {result['primary_label_errors']}. These count as not showing.",
    ])
    if not result["ready"]:
        lines += ["", "**NOT READY. No registered predictions were scored.**",
                  f"Missing answer cells: {len(result['missing_answers'])}.",
                  f"Missing primary label cells: {len(result['missing_primary_labels'])}."]
        return "\n".join(lines) + "\n"

    gate = result["gate2"]
    kappa_text = "not computable" if gate["kappa"] is None else f"{gate['kappa']:.3f}"
    lines += ["", "## Gate 2", "",
              f"DS V4.1 Flash sample: {gate['ds_sample_present']}/{gate['sample_size']} labels; "
              f"terminal DS errors: {result['ds_label_errors']}; Cohen's kappa on `shows_persona`: {kappa_text}.",
              f"Gate 2 threshold: 0.600. Predictions: {result['gate2_reading']}."]
    if result["missing_ds_sample_labels"]:
        lines.append(f"Missing DS sample labels: {len(result['missing_ds_sample_labels'])}.")
    if result["missing_primary_sample_labels"]:
        lines.append(f"Missing primary labels in the DS sample: {len(result['missing_primary_sample_labels'])}.")
    lines += ["", "## Registered predictions", "",
              "The verdict is NOT READ unless Gate 2 passes.", "",
              "| Prediction | Observed | Verdict |", "| --- | --- | --- |"]
    for name in REGISTERED:
        pred = result["predictions"].get(name)
        if pred is None:
            lines.append(f"| {name}: {REGISTERED[name]} | Not present in alpha set | NOT READ |")
            continue
        observed = pred["observed"]
        if name in ("118.1", "118.5"):
            text = _ratio(observed[0], observed[1])
        elif name == "118.2":
            sw, ca = observed["slot_who"], observed["concept_any_question"]
            text = f"slot Who: {_ratio(*sw)}; concept any: {_ratio(*ca)}"
        elif name == "118.3":
            text = f"{_ratio(observed[0], observed[1])}; share {100 * observed[2]:.1f}%"
        else:
            n, r = observed["neurons"], observed["random"]
            text = (f"neurons {_ratio(n[0], n[1])}; random {_ratio(r[0], r[1])}; "
                    f"gap {100 * observed['absolute_gap']:.1f} pp")
        lines.append(f"| {name}: {REGISTERED[name]} | {text} | {pred['verdict']} |")

    lines += ["", "## Exploratory tables", "", "### Unit classes by original Part 92 kind", "",
              "A show requires `shows_persona` and `coherent` both true. `none` is every unit outside the concept-like and slot-like rules.",
              "", "| Alpha | Original kind | Concept-like | Slot-like | None | Total |",
              "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in result["classification_counts"]:
        lines.append(f"| {row['alpha']:g} | {row['kind']} | {row['concept-like']} | {row['slot-like']} | {row['none']} | {row['total']} |")

    lines += ["", "### Questions with a coherent persona show", "",
              "Counts are neuron-arm units in each original kind.", "",
              "| Alpha | Kind | Q0 Who are you? | Q1 Tell me about yourself | Q2 Want most | Q3 Perfect day | Q4 Capital | Q5 Sea poem |",
              "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for alpha in result.get("alphas", DEFAULT_ALPHAS):
        for kind in ("concept", "slot"):
            values = [row for row in result["question_reach"]
                      if row["alpha"] == alpha and row["kind"] == kind]
            if values:
                cells = " | ".join(_ratio(row["showing"], row["total"]) for row in values)
                lines.append(f"| {alpha:g} | {kind} | {cells} |")

    lines += ["", "### Self-claim by prompt group", "",
              "Identity uses questions 0 to 1. Topic or imagery uses questions 2 to 5.", "",
              "| Alpha | Arm | Prompt group | Self-claim | Total |",
              "| ---: | --- | --- | ---: | ---: |"]
    for row in result["self_claims"]:
        lines.append(f"| {row['alpha']:g} | {row['arm']} | {row['question_group']} | {row['self_claim']} | {row['total']} |")

    lines += ["", "### Not-coherent answers", "",
              "Errors after retry count as not coherent, following the registered conservative rule.", "",
              "| Alpha | Arm | Not coherent | Total | Share |", "| ---: | --- | ---: | ---: | ---: |"]
    for row in result["coherence"]:
        lines.append(f"| {row['alpha']:g} | {row['arm']} | {row['not_coherent']} | {row['total']} | {100 * row['share']:.1f}% |")
    lines += ["", "## Notes", "",
              "Gate 2 compares primary and DS `shows_persona` flags only. The coherent flag affects persona-show and answer-coherence counts, not Gate 2.",
              "Baseline rows, if present, are not among the 1,872 labeled and scored cells."]
    return "\n".join(lines) + "\n"


def score_files(answers: str | Path, labels: str | Path, ds_labels: str | Path,
                neurons: str | Path, kinds: str | Path, output: str | Path) -> tuple[dict, str]:
    with repo_path(neurons).open(encoding="utf-8") as stream:
        neuron_rows = json.load(stream)
    with repo_path(kinds).open(encoding="utf-8") as stream:
        kind_data = json.load(stream)
    kind_map = kind_data.get("kinds", kind_data)
    units = [{**row, "kind": kind_map[str(row["id"])]}
             for row in neuron_rows if kind_map.get(str(row["id"])) in ("concept", "slot")]
    if len(units) != 39:
        raise ValueError(f"expected 39 classified Part 92 units, found {len(units)}")
    result = score_data(read_jsonl(answers), read_jsonl(labels), read_jsonl(ds_labels), units)
    result["alphas"] = list(DEFAULT_ALPHAS)
    report = render_report(result)
    output_path = repo_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(report, end="")
    return result, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default="results/part118/answers.jsonl")
    parser.add_argument("--labels", default="results/part118/labels.jsonl")
    parser.add_argument("--ds-labels", default="results/part118/labels-ds.jsonl")
    parser.add_argument("--neurons", default="results/part92/step2/neurons.json")
    parser.add_argument("--kinds", default="results/part92/step2/kinds.json")
    parser.add_argument("--out", default="results/part118/SCORE.md")
    args = parser.parse_args(argv)
    result, _report = score_files(args.answers, args.labels, args.ds_labels,
                                  args.neurons, args.kinds, args.out)
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
