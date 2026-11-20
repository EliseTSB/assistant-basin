#!/usr/bin/env python3
"""Part 180 v5.1 step 2: the arm rules, the development inclusion rule, and the configuration selection.

Design sections 1, 3, 3a and 4. For one configuration a neuron is INCLUDED iff

  S  Q-claim yes
  A0 content+ and Q-claim no          (both A texts, generated once, read under their own about-prompts)
  A1 content+ and Q-claim no
  C  content+ and Q-claim no
  B  content+ NO                      (any non-negated, non-quoted presence fails)
  self margin > 0 and cont margin > 0 (the two development margins of texts180v5)

content+ = (mention OR theme OR description) AND NOT negated-only AND NOT quoted-only. The reader's `negated` and
`quoted` flags already carry the "only" meaning (prompt180v5.txt), so they are used as they are. An unresolved claim
counts as a FAIL and is counted. Nothing else enters inclusion: Part 92's q0 and q3 are reported beside and never used.

Selection: most joint passes on the 18; ties to fewer unresolved labels, then to the lower configuration number.
The section 4 bars (10 of 18, 5 of the shared 8) are REPORTED for the winner, not applied here.

  classify: python -I part180/classify180v5.py classify
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (HERE,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
import classify180  # noqa: E402
import texts180 as v4  # noqa: E402
import texts180v5  # noqa: E402

FIELDS = ("claim", "mention", "theme", "description", "negated", "quoted")
ARMS = ("S", "A0", "A1", "C", "B")
CLAIM_ARMS = ("S", "A0", "A1", "C")   # arms whose rule turns on the claim question
CONTENT_ARMS = ("A0", "A1", "C", "B")
BAR_EIGHTEEN = 10
BAR_SHARED_EIGHT = 5


def validate_label(label):
    if label is None:
        return None
    if set(label) != set(FIELDS):
        raise ValueError(f"label fields {sorted(label)}")
    if label["claim"] not in ("yes", "no", "unresolved"):
        raise ValueError(f"claim {label['claim']!r}")
    if any(type(label[key]) is not bool for key in FIELDS[1:]):
        raise ValueError("non-boolean content flag")
    return label


def content_plus(label):
    """Primary content endpoint, frozen: (mention OR theme OR description) AND NOT negated-only AND NOT quoted-only."""
    label = validate_label(label)
    if label is None:
        return None
    present = bool(label["mention"] or label["theme"] or label["description"])
    return bool(present and not label["negated"] and not label["quoted"])


def arm_passes(arm, label):
    """True / False, or None when the reader row is missing. Unresolved claim is False on the arms that need one."""
    label = validate_label(label)
    if label is None:
        return None
    if arm == "S":
        return label["claim"] == "yes"
    if arm in ("A0", "A1", "C"):
        return bool(content_plus(label)) and label["claim"] == "no"
    if arm == "B":
        return content_plus(label) is False
    raise ValueError(f"unknown arm {arm!r}")


def part92_beside(answers):
    """Part 92 step 2's q0 and q3 labels per neuron. Reported beside every table, never used for inclusion."""
    beside = {}
    for row in answers:
        if row.get("unit") != "neuron" or "error" in row:
            continue
        try:
            question = int(row.get("question_index", -1))
        except (TypeError, ValueError):
            continue
        if question not in (0, 3):
            continue
        beside.setdefault(str(row.get("id")), {})[question] = {
            "shows_persona": classify180._bool_value(row.get("shows_persona")),
            "self_claim": classify180._bool_value(row.get("self_claim"))}
    report = {}
    for neuron_id, questions in beside.items():
        q0 = questions.get(0, {})
        q3 = questions.get(3, {})
        q3_shows_without_claim = classify180._fields_match(
            (q3.get("shows_persona"), q3.get("self_claim")), (True, False))
        report[neuron_id] = {"q0": q0, "q3": q3,
                             "q0_self_claim_true": q0.get("self_claim") is True,
                             "q3_shows_without_claim": q3_shows_without_claim}
    return report


def expected_cells(neurons, configurations=texts180v5.CONFIGURATIONS):
    """The frozen evidence grid: every (configuration, neuron, arm). Evidence is checked against THIS, not the file."""
    return {(int(config["number"]), str(neuron["id"]), arm)
            for config in configurations for neuron in neurons for arm in ARMS}


def classify(margins, labels, neurons, *, configurations=texts180v5.CONFIGURATIONS, shared_ids=(),
             part92=None):
    """Every configuration's outcome, every per-criterion count, and the development cohort.

    Incomplete or ambiguous evidence is rejected, never inferred: every (configuration, neuron, arm) cell must
    carry a label object and every (configuration, neuron) must carry both margins, duplicate ids are an error, and
    a row outside the frozen grid is an error.
    """
    expected = expected_cells(neurons, configurations)
    margin_keys = {(int(config["number"]), str(neuron["id"]))
                   for config in configurations for neuron in neurons}
    margin_by = {}
    for row in margins:
        key = (int(row["config"]), str(row["neuron"]))
        if key in margin_by:
            raise ValueError(f"duplicate margin row {key}")
        if key not in margin_keys:
            raise ValueError(f"margin row outside the frozen grid: {key}")
        for field in ("self", "cont"):
            if row.get(field) is None:
                raise ValueError(f"margin row {key} has no {field} value")
        margin_by[key] = row
    missing_margins = sorted(margin_keys - set(margin_by))
    if missing_margins:
        raise ValueError(f"{len(missing_margins)} (configuration, neuron) pairs have no margins, "
                         f"first {missing_margins[:3]}")
    label_by, seen_ids = {}, set()
    for row in labels:
        row_id = str(row.get("id", ""))
        if not row_id:
            raise ValueError("reader row without an id")
        if row_id in seen_ids:
            raise ValueError(f"duplicate reader row id {row_id}")
        seen_ids.add(row_id)
        key = (int(row["config"]), str(row["neuron"]), str(row["key"]))
        if key in label_by:
            raise ValueError(f"duplicate reader cell {key}")
        if key not in expected:
            raise ValueError(f"reader row outside the frozen grid: {key}")
        label_by[key] = validate_label(row.get("label"))
    missing = sorted(expected - set(label_by))
    if missing:
        raise ValueError(f"{len(missing)} reader cells have no row, first {missing[:3]}")
    unlabelled = sorted(key for key, label in label_by.items() if label is None)
    if unlabelled:
        raise ValueError(f"{len(unlabelled)} reader cells carry no label (the reader failed), first {unlabelled[:3]}")
    shared = {str(value) for value in shared_ids}
    outcomes = {}
    for config in configurations:
        number = int(config["number"])
        counts = {arm: {"passed": 0, "failed": 0, "unresolved": 0, "missing": 0, "n": 0} for arm in ARMS}
        for name in ("self_margin_positive", "cont_margin_positive"):
            counts[name] = {"passed": 0, "failed": 0, "missing": 0, "n": 0}
        subtype = {key: 0 for key in ("mention", "theme", "description", "negated", "quoted")}
        unresolved_total = 0
        per_neuron = {}
        cohort = []
        for neuron in neurons:
            neuron_id = str(neuron["id"])
            arms = {}
            failed_arms = []
            for arm in ARMS:
                label = label_by[(number, neuron_id, arm)]
                status = arm_passes(arm, label)
                unresolved = bool(label["claim"] == "unresolved")
                arms[arm] = {"pass": status, "unresolved_claim": unresolved,
                             "content_plus": content_plus(label), "label": label}
                counts[arm]["n"] += 1
                if status:
                    counts[arm]["passed"] += 1
                elif unresolved and arm in CLAIM_ARMS:
                    counts[arm]["unresolved"] += 1
                    failed_arms.append(f"{arm}:unresolved")
                else:
                    counts[arm]["failed"] += 1
                    failed_arms.append(f"{arm}:failed")
                if unresolved:
                    unresolved_total += 1
                for key in subtype:
                    subtype[key] += int(bool(label[key]))
            margin = margin_by[(number, neuron_id)]
            margins_row = {}
            for name, field in (("self_margin_positive", "self"), ("cont_margin_positive", "cont")):
                counts[name]["n"] += 1
                if float(margin[field]) > 0:
                    counts[name]["passed"] += 1
                    margins_row[name] = True
                else:
                    counts[name]["failed"] += 1
                    margins_row[name] = False
                    failed_arms.append(f"{name}:failed")
            included = not failed_arms
            if included:
                cohort.append(neuron_id)
            per_neuron[neuron_id] = {
                "arms": arms, "margins": margins_row,
                "self": float(margin["self"]), "cont": float(margin["cont"]),
                "included": included, "dropped_by": failed_arms,
                "part92": None if part92 is None else part92.get(neuron_id)}
        beside = {}
        for neuron_id, row in per_neuron.items():
            part = row["part92"] or {}
            beside[neuron_id] = {"q0_self_claim_true": bool(part.get("q0_self_claim_true")),
                                 "q3_shows_without_claim": part.get("q3_shows_without_claim")}
        outcomes[str(number)] = {
            "config": number, "self_prompt": config["self_prompt"], "content_prompt": config["content_prompt"],
            "counts": counts, "subtype_presence": subtype, "unresolved_total": unresolved_total,
            "joint_passes": len(cohort), "of": len(neurons), "cohort": cohort,
            "cells": sum(1 for key in expected if key[0] == number),
            "cohort_shared_eight": sorted(set(cohort) & shared),
            "part92_beside": {"q0_self_claim_true": sum(1 for row in beside.values() if row["q0_self_claim_true"]),
                              "q3_shows_without_claim": sum(1 for row in beside.values()
                                                            if row["q3_shows_without_claim"] is True),
                              "per_neuron": beside,
                              "used_for_inclusion": False},
            "neurons": per_neuron,
        }
    return outcomes


def select(outcomes, configurations=texts180v5.CONFIGURATIONS):
    """Most joint passes on the 18; ties to fewer unresolved labels; then to the lower configuration number."""
    order = {int(config["number"]): index for index, config in enumerate(configurations)}
    ranked = sorted(outcomes.values(),
                    key=lambda row: (-int(row["joint_passes"]), int(row["unresolved_total"]),
                                     order[int(row["config"])]))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    return ranked[0]


def winner_report(outcomes, winner, *, shared_ids, neurons):
    """The committed-ready freeze of the winner and its development cohort (design 3b steps 3 and 4)."""
    shared = {str(value) for value in shared_ids}
    cohort = list(winner["cohort"])
    shared_passed = sorted(set(cohort) & shared)
    met_eighteen = len(cohort) >= BAR_EIGHTEEN
    met_eight = len(shared_passed) >= BAR_SHARED_EIGHT
    return {
        "part": 180, "version": "v5.1", "stage": "step 2 - six-configuration screen",
        "rule": "most joint passes on the 18; ties to fewer unresolved labels, then to the lower configuration number",
        "selected_configuration": int(winner["config"]),
        "self_prompt": winner["self_prompt"], "content_prompt": winner["content_prompt"],
        "development_cohort": cohort, "joint_passes": len(cohort), "of": len(neurons),
        "shared_eight": {"ids": sorted(shared), "passed": shared_passed,
                         "count": len(shared_passed), "of": len(shared)},
        "bars": {
            "included_18": {"value": len(cohort), "bar": BAR_EIGHTEEN, "met": met_eighteen},
            "shared_eight": {"value": len(shared_passed), "bar": BAR_SHARED_EIGHT, "met": met_eight},
            "status_if_applied": "PASS" if (met_eighteen and met_eight) else "PARK",
            "applied": False,
            "note": "design section 4, applied at section 3b step 4 after this file is committed; reported only",
        },
        "runner_up": _runner_up(outcomes),
        "outcomes": {key: {"joint_passes": row["joint_passes"], "of": row["of"],
                           "unresolved_total": row["unresolved_total"],
                           "cohort_shared_eight": len(row["cohort_shared_eight"]),
                           "counts": row["counts"], "subtype_presence": row["subtype_presence"]}
                     for key, row in sorted(outcomes.items(), key=lambda item: int(item[0]))},
    }


def _runner_up(outcomes):
    """The second-ranked configuration, carried for the record only: never chosen after a confirmation failure."""
    ranked = sorted(outcomes.values(), key=lambda row: int(row["rank"]))
    if len(ranked) < 2:
        return None
    return {"configuration": int(ranked[1]["config"]), "joint_passes": int(ranked[1]["joint_passes"]),
            "note": "never chosen after a confirmation failure (design section 3b step 6)"}


def _write_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def cmd_classify(args):
    results = v4.normalise_path(args.results)
    out_path = v4.normalise_path(args.out)
    classification_path = v4.normalise_path(args.classification)
    margins = texts180v5.read_rows(v4.normalise_path(args.margins))
    labels = texts180v5.read_rows(v4.normalise_path(args.labels))
    if not margins:
        raise ValueError(f"no margin rows in {args.margins}; run texts180v5.py score first")
    if not labels:
        raise ValueError(f"no reader rows in {args.labels}; run reader180v5.py read first")
    neurons = v4.load_concept_neurons(results)
    answers = v4._load_jsonl(results / "part92" / "step2" / "answers" / "instruct_chat.jsonl")
    score92b = json.loads(v4.normalise_path(args.score92b).read_text(encoding="utf-8"))
    shared_ids = classify180.deepseek_concept_ids(score92b)
    outcomes = classify(margins, labels, neurons, shared_ids=shared_ids, part92=part92_beside(answers))
    winner = select(outcomes)
    report = winner_report(outcomes, winner, shared_ids=shared_ids, neurons=neurons)
    classification = {"part": 180, "version": "v5.1", "shared_eight_ids": sorted(shared_ids),
                      "evidence": {"cells_expected": len(expected_cells(neurons)),
                                   "reader_rows": len(labels), "margin_rows": len(margins),
                                   "rule": "every frozen cell labelled, every margin pair present, no duplicate "
                                           "ids; anything else raises instead of being inferred"},
                      "outcomes": outcomes}
    _write_json(classification_path, classification)
    _write_json(out_path, report)
    print(json.dumps({"winner": report["selected_configuration"], "cohort": report["joint_passes"],
                      "of": report["of"], "shared_eight": report["shared_eight"]["count"],
                      "bars": report["bars"]["status_if_applied"], "out": str(out_path)}))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("classify")
    sp.add_argument("--results", default=str(ROOT / "results"))
    sp.add_argument("--margins", default=str(ROOT / "results" / "part180" / "v5" / "margins.jsonl"))
    sp.add_argument("--labels", default=str(ROOT / "results" / "part180" / "v5" / "reader-deepseek.jsonl"))
    sp.add_argument("--out", default=str(ROOT / "results" / "part180" / "v5" / "winner.json"))
    sp.add_argument("--classification",
                    default=str(ROOT / "results" / "part180" / "v5" / "classification.json"))
    sp.add_argument("--score92b", default=str(ROOT / "results" / "part92b" / "score92b.json"))
    args = parser.parse_args(argv)
    if args.cmd != "classify":
        parser.error("only the classify command is implemented")
    cmd_classify(args)


if __name__ == "__main__":
    main()
