#!/usr/bin/env python3
"""Score Part 92b's second-reader outputs using the registered design."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import label92b  # noqa: E402  Shared item keys, prompt labels, and JSONL reader.

GATE_MIN_CORRECT = 11
GATE_B_MIN_KAPPA = 0.6
CLASSIFIED_KINDS = ("concept", "slot")
DS_HAIKU_PREDICTION = "DS V4.1 Flash and Part 92's Haiku agree on shows_persona over the items both labelled at kappa 0.6 or more."
KEEP_PREDICTION = "At least 30 of the 39 classified units keep their Part 92 kind under the DS labels."
CHANGED_PREDICTION = "Of the classified units whose kind changes, at least two thirds were slot units (read only if at least 3 change)."
NEURON_PREDICTION = "DS marks the persona in more of the neuron-arm answers than Haiku's 104."
RANDOM_PREDICTION = "DS marks the persona in at most 10% of the random-arm answers."
PREDICTIONS = {
    "92b.1": DS_HAIKU_PREDICTION,
    "92b.2": KEEP_PREDICTION,
    "92b.3": CHANGED_PREDICTION,
    "92b.4": NEURON_PREDICTION,
    "92b.5": RANDOM_PREDICTION,
}
DEST_KINDS = ("concept", "slot", "unclassified", "dropped", "incomplete")


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def row_key(row: dict) -> tuple[str, str, int]:
    return label92b.item_key(row)


def load_labels(path: str | Path) -> tuple[dict, int]:
    file = Path(path)
    if not file.exists():
        return {}, 0
    latest = {}
    for row in label92b.read_jsonl(file):
        latest[row_key(row)] = row
    errors = sum(not label92b.valid_label(row) for row in latest.values())
    return latest, errors


def cohen_kappa(left: dict, right: dict, keys: list[tuple[str, str, int]]) -> float | None:
    if not keys:
        return None
    a = [bool(left[key]["shows_persona"]) for key in keys]
    b = [bool(right[key]["shows_persona"]) for key in keys]
    observed = sum(x == y for x, y in zip(a, b)) / len(keys)
    p_a = sum(a) / len(keys)
    p_b = sum(b) / len(keys)
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def valid_intersection(left: dict, right: dict, keys) -> list[tuple[str, str, int]]:
    return [key for key in keys
            if key in left and key in right
            and label92b.valid_label(left[key]) and label92b.valid_label(right[key])]


def kind_from_questions(by_question: dict[int, bool]) -> str:
    """Apply Part 92's concept/slot/dropped rule to observed neuron answers."""
    if len(by_question) < 6:
        return "incomplete"
    if not by_question.get(0, False):
        return "dropped"
    shown = [question for question, value in by_question.items() if value]
    if len(shown) >= 3:
        return "concept"
    if len(shown) in (1, 2) and all(question in (0, 1) for question in shown):
        return "slot"
    return "unclassified"


def classify_units(labels: dict, unit_ids: list[str]) -> dict[str, str]:
    by_unit: dict[str, dict[int, bool]] = {}
    for key, row in labels.items():
        unit_id, unit, question = key
        if unit != "neuron":
            continue
        # A persisted error row represents an attempted question and counts as false.
        by_unit.setdefault(unit_id, {})[question] = (
            bool(row["shows_persona"]) if label92b.valid_label(row) else False
        )
    return {unit_id: kind_from_questions(by_unit.get(unit_id, {})) for unit_id in unit_ids}


def read_gate(known_rows: list[dict], gate_path: str | Path) -> dict:
    path = Path(gate_path)
    if not path.exists():
        return {"correct": 0, "total": len(known_rows), "passed": False, "exists": False}
    result = label92b.gate_result(known_rows, label92b.read_jsonl(path))
    result["exists"] = True
    return result


def gate_statuses(known_rows: list[dict], ds_gate: str | Path,
                  luna_gate: str | Path) -> dict[str, dict]:
    return {
        "ds": read_gate(known_rows, ds_gate),
        "luna": read_gate(known_rows, luna_gate),
    }


def registered_verdicts(*, gate_a_passed: bool, gate_b_kappa: float | None,
                        ds_haiku_kappa: float | None, unchanged_count: int,
                        changed_count: int, slot_changes: int,
                        neuron_shows: int, random_shows: int,
                        random_total: int) -> dict[str, dict]:
    """Return the five registered verdicts from their fixed numeric predicates."""
    if not gate_a_passed:
        status = "not read (Gate A)"
        return {key: {"prediction": text, "verdict": status} for key, text in PREDICTIONS.items()}
    if gate_b_kappa is None or gate_b_kappa < GATE_B_MIN_KAPPA:
        status = "not read (Gate B)"
        return {key: {"prediction": text, "verdict": status} for key, text in PREDICTIONS.items()}

    changes_share = slot_changes / changed_count if changed_count else None
    random_rate = random_shows / random_total if random_total else None
    return {
        "92b.1": {"prediction": DS_HAIKU_PREDICTION,
                  "verdict": "held" if ds_haiku_kappa is not None and ds_haiku_kappa >= 0.6 else "failed"},
        "92b.2": {"prediction": KEEP_PREDICTION,
                  "verdict": "held" if unchanged_count >= 30 else "failed"},
        "92b.3": {
            "prediction": CHANGED_PREDICTION,
            "verdict": ("not read (fewer than 3 units changed)" if changed_count < 3
                        else "held" if changes_share is not None and changes_share >= 2 / 3
                        else "failed"),
        },
        "92b.4": {"prediction": NEURON_PREDICTION,
                  "verdict": "held" if neuron_shows > 104 else "failed"},
        "92b.5": {"prediction": RANDOM_PREDICTION,
                  "verdict": "held" if random_rate is not None and random_rate <= 0.10 else "failed"},
    }


def _is_true_label(labels: dict, key: tuple[str, str, int]) -> bool:
    row = labels.get(key)
    return bool(row and label92b.valid_label(row) and row["shows_persona"])


def arm_counts(answer_rows: list[dict], labels: dict) -> dict:
    result = {}
    for arm in ("neuron", "random"):
        keys = [row_key(row) for row in answer_rows if row["unit"] == arm]
        result[arm] = {"shows": sum(_is_true_label(labels, key) for key in keys),
                       "answers": len(keys)}
    return result


def score_data(*, answer_rows: list[dict], neuron_rows: list[dict], kind_rows: dict,
               known_rows: list[dict], sample_keys: list[tuple[str, str, int]],
               ds_labels: dict, luna_labels: dict, haiku_labels: dict,
               ds_errors: int, luna_errors: int, haiku_errors: int,
               gates: dict[str, dict]) -> dict:
    all_keys = [row_key(row) for row in answer_rows]
    gate_b_keys = valid_intersection(ds_labels, luna_labels, sample_keys)
    gate_b_kappa = cohen_kappa(ds_labels, luna_labels, gate_b_keys)
    haiku_keys = valid_intersection(ds_labels, haiku_labels, all_keys)
    ds_haiku_kappa = cohen_kappa(ds_labels, haiku_labels, haiku_keys)

    unit_ids = [str(row["id"]) for row in neuron_rows]
    ds_kinds = classify_units(ds_labels, unit_ids)
    classified = [unit_id for unit_id in unit_ids
                  if kind_rows.get(unit_id) in CLASSIFIED_KINDS]
    comparisons = []
    transition = {source: {target: 0 for target in DEST_KINDS}
                  for source in CLASSIFIED_KINDS}
    for unit_id in classified:
        source = kind_rows[unit_id]
        target = ds_kinds.get(unit_id, "incomplete")
        if target not in DEST_KINDS:
            target = "unclassified"
        transition[source][target] += 1
        comparisons.append({"id": unit_id, "part92_kind": source, "ds_kind": target,
                            "same": source == target})
    unchanged = sum(row["same"] for row in comparisons)
    changed = len(comparisons) - unchanged
    slot_changes = sum(not row["same"] and row["part92_kind"] == "slot" for row in comparisons)

    ds_arms = arm_counts(answer_rows, ds_labels)
    haiku_arms = arm_counts(answer_rows, haiku_labels)
    gate_a_passed = gates["ds"]["passed"] and gates["luna"]["passed"]
    verdicts = registered_verdicts(
        gate_a_passed=gate_a_passed,
        gate_b_kappa=gate_b_kappa,
        ds_haiku_kappa=ds_haiku_kappa,
        unchanged_count=unchanged,
        changed_count=changed,
        slot_changes=slot_changes,
        neuron_shows=ds_arms["neuron"]["shows"],
        random_shows=ds_arms["random"]["shows"],
        random_total=ds_arms["random"]["answers"],
    )
    return {
        "gate_a": gates,
        "gate_b": {"kappa": gate_b_kappa, "n": len(gate_b_keys),
                   "threshold": GATE_B_MIN_KAPPA,
                   "passed": gate_b_kappa is not None and gate_b_kappa >= GATE_B_MIN_KAPPA},
        "ds_haiku": {"kappa": ds_haiku_kappa, "n": len(haiku_keys)},
        "kind_comparison": {
            "classified_units": len(comparisons), "unchanged": unchanged,
            "changed": changed, "slot_changes": slot_changes,
            "units": comparisons, "from_to_counts": transition,
        },
        "arm_shows": {"ds": ds_arms, "haiku": haiku_arms},
        "errors_left": {"ds": ds_errors, "luna": luna_errors, "haiku": haiku_errors},
        "verdicts": verdicts,
    }


def _fmt_kappa(value: float | None) -> str:
    return "not defined" if value is None else f"{value:.3f}"


def render_markdown(report: dict) -> str:
    gates = report["gate_a"]
    lines = [
        "# Part 92b score",
        "",
        "## Gate A",
        "",
        "| Reader | Correct | Result |",
        "| --- | ---: | --- |",
    ]
    for reader in ("ds", "luna"):
        gate = gates[reader]
        state = "pass" if gate["passed"] else "fail"
        lines.append(f"| {reader} | {gate['correct']}/{gate['total']} | {state} |")
    gate_b = report["gate_b"]
    lines.extend([
        "",
        "## Reader agreement",
        "",
        f"- Gate B, DS vs Luna on {gate_b['n']} shared valid sample items: kappa {_fmt_kappa(gate_b['kappa'])}; "
        f"threshold {gate_b['threshold']:.1f}; {'pass' if gate_b['passed'] else 'fail'}.",
        f"- DS vs Haiku on {report['ds_haiku']['n']} items both labelled: "
        f"kappa {_fmt_kappa(report['ds_haiku']['kappa'])}.",
        "",
        "## Kind comparison",
        "",
        f"{report['kind_comparison']['unchanged']} of {report['kind_comparison']['classified_units']} "
        f"classified units keep their Part 92 kind; {report['kind_comparison']['changed']} change.",
        "",
        "| Unit | Part 92 kind | DS kind | Same |",
        "| --- | --- | --- | --- |",
    ])
    for row in report["kind_comparison"]["units"]:
        lines.append(f"| {row['id']} | {row['part92_kind']} | {row['ds_kind']} | "
                     f"{'yes' if row['same'] else 'no'} |")
    lines.extend(["", "### Counts by Part 92 kind and DS kind", "",
                  "| From \\ To | " + " | ".join(DEST_KINDS) + " |",
                  "| --- | " + " | ".join("---:" for _ in DEST_KINDS) + " |"])
    for source in CLASSIFIED_KINDS:
        counts = report["kind_comparison"]["from_to_counts"][source]
        lines.append(f"| {source} | " + " | ".join(str(counts[target]) for target in DEST_KINDS) + " |")

    lines.extend(["", "## Persona shows", "",
                  "Errors and absent label rows count as not showing.", "",
                  "| Reader | Arm | Shows | Answers |",
                  "| --- | --- | ---: | ---: |"])
    for reader in ("ds", "haiku"):
        for arm in ("neuron", "random"):
            count = report["arm_shows"][reader][arm]
            lines.append(f"| {reader} | {arm} | {count['shows']} | {count['answers']} |")
    errors = report["errors_left"]
    lines.extend(["", "## Error rows left", "",
                  f"- DS: {errors['ds']}; Luna: {errors['luna']}; Haiku: {errors['haiku']}.",
                  "", "## Registered predictions", ""])
    for key in PREDICTIONS:
        lines.append(f"- **{key} — {report['verdicts'][key]['verdict']}:** "
                     f"{report['verdicts'][key]['prediction']}")
    lines.append("")

    if not gates["ds"]["passed"] or not gates["luna"]["passed"]:
        lines.extend(["## Reading", "", "Nothing is read because Gate A did not pass for both readers.", ""])
    elif not gate_b["passed"]:
        lines.extend(["## Reading", "",
                      "Gate B is below 0.6; every registered prediction is not read (Gate B). "
                      "Counts above are descriptive.", ""])
    else:
        b1 = report["verdicts"]["92b.1"]["verdict"] == "held"
        b2 = report["verdicts"]["92b.2"]["verdict"] == "held"
        if not b2:
            changed_units = [row for row in report["kind_comparison"]["units"] if not row["same"]]
            lines.extend([
                "## Reading",
                "",
                "At this size Part 92's split depends on the reader.",
                "",
                "| Unit | Part 92 kind | DS kind |",
                "| --- | --- | --- |",
            ])
            for row in changed_units:
                lines.append(f"| {row['id']} | {row['part92_kind']} | {row['ds_kind']} |")
            lines.extend(["", "Parts 112, 117 and 118 keep their own verdicts but their kind labels are Haiku's.", ""])
        elif b1 and b2:
            lines.extend([
                "## Reading",
                "",
                "The split survives a second reader on these answers, and nothing more: a second reader tests the "
                "reading, not the neurons.",
                "Every unit that changes kind is designated for an exploratory eye read.",
                "",
            ])
    return "\n".join(lines)


def read_sample(path: str | Path, answer_rows: list[dict]) -> list[tuple[str, str, int]]:
    file = Path(path)
    if file.exists():
        payload = json.loads(file.read_text(encoding="utf-8"))
        rows = payload.get("item_keys", [])
        return [row_key(row) for row in rows]
    return label92b.sample_keys(row_key(row) for row in answer_rows)


def score_files(*, ds_path: str | Path, luna_path: str | Path, haiku_path: str | Path,
                answers_path: str | Path, neurons_path: str | Path, kinds_path: str | Path,
                known_path: str | Path, sample_path: str | Path,
                ds_gate_path: str | Path, luna_gate_path: str | Path,
                score_md_path: str | Path, score_json_path: str | Path) -> dict:
    answer_rows = label92b.read_jsonl(answers_path)
    neuron_rows = json.loads(Path(neurons_path).read_text(encoding="utf-8"))
    kind_payload = json.loads(Path(kinds_path).read_text(encoding="utf-8"))
    kind_rows = kind_payload.get("kinds", kind_payload)
    known_rows = label92b.read_jsonl(known_path)
    samples = read_sample(sample_path, answer_rows)
    gates = gate_statuses(known_rows, ds_gate_path, luna_gate_path)
    ds, ds_errors = load_labels(ds_path)
    luna, luna_errors = load_labels(luna_path)
    haiku, haiku_errors = load_labels(haiku_path)
    report = score_data(answer_rows=answer_rows, neuron_rows=neuron_rows,
                        kind_rows=kind_rows, known_rows=known_rows,
                        sample_keys=samples, ds_labels=ds, luna_labels=luna,
                        haiku_labels=haiku, ds_errors=ds_errors,
                        luna_errors=luna_errors, haiku_errors=haiku_errors,
                        gates=gates)
    md_path, json_path = Path(score_md_path), Path(score_json_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds", default="results/part92b/labels-ds.jsonl")
    parser.add_argument("--luna", default="results/part92b/labels-luna.jsonl")
    parser.add_argument("--haiku", default="results/part92/step2/labels-instruct_chat.jsonl")
    parser.add_argument("--answers", default="results/part92/step2/answers/instruct_chat.jsonl")
    parser.add_argument("--neurons", default="results/part92/step2/neurons.json")
    parser.add_argument("--kinds", default="results/part92/step2/kinds.json")
    parser.add_argument("--known", default=str(HERE / "known92b.jsonl"))
    parser.add_argument("--sample", default="results/part92b/sample92b.json")
    parser.add_argument("--gate-ds", default="results/part92b/gate-a-ds.jsonl")
    parser.add_argument("--gate-luna", default="results/part92b/gate-a-luna.jsonl")
    parser.add_argument("--score-md", default="results/part92b/SCORE.md")
    parser.add_argument("--score-json", default="results/part92b/score92b.json")
    args = parser.parse_args(argv)
    score_files(
        ds_path=repo_path(args.ds), luna_path=repo_path(args.luna),
        haiku_path=repo_path(args.haiku), answers_path=repo_path(args.answers),
        neurons_path=repo_path(args.neurons), kinds_path=repo_path(args.kinds),
        known_path=repo_path(args.known), sample_path=repo_path(args.sample),
        ds_gate_path=repo_path(args.gate_ds), luna_gate_path=repo_path(args.gate_luna),
        score_md_path=repo_path(args.score_md), score_json_path=repo_path(args.score_json),
    )
    print(json.dumps({"score_md": str(repo_path(args.score_md)),
                      "score_json": str(repo_path(args.score_json))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
