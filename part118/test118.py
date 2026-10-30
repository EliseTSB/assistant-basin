"""Synthetic-only checks for Part 118 gates, labeling sample, and scorer."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gate1_118
import score118


UNITS = ([{"id": f"concept-{i:02d}", "kind": "concept"} for i in range(18)]
         + [{"id": f"slot-{i:02d}", "kind": "slot"} for i in range(21)])
ALPHAS = (0.4, 0.8, 1.6, 3.2)


def _label(unit, arm, alpha, qi, mode):
    kind = unit["kind"]
    index = int(unit["id"].rsplit("-", 1)[1])
    if mode == "hold":
        if alpha == 0.4:
            show = qi == 0 and ((kind == "concept" and index < 9)
                                or (kind == "slot" and index < 4))
        elif alpha == 0.8:
            show = qi == 0 and index % 2 == 0
        elif alpha == 1.6:
            show = qi < 3 and ((kind == "concept" and index < 12)
                               or (kind == "slot" and index < 7))
        else:
            show = qi == 0 and index % 2 == 0
        coherent = not (alpha == 3.2 and qi < 4)
    else:
        if alpha == 0.4:
            show = qi == 0 and ((kind == "concept" and index < 9)
                                or (kind == "slot" and index < 6))
        elif alpha == 0.8:
            show = qi == 0 and index % 2 == 0
        elif alpha == 1.6:
            show = qi == 0 and index < 2
        else:
            show = qi == 0 and index % 2 == 0
        coherent = not (alpha == 3.2 and arm == "random")
    return {"shows_persona": bool(show), "self_claim": bool(show and qi < 2),
            "speaker": "ai" if show else "none", "coherent": bool(coherent)}


def synthetic_rows(mode="hold"):
    answers = []
    labels = []
    by_key = {}
    for unit in UNITS:
        for arm in ("neuron", "random"):
            for alpha in ALPHAS:
                for qi in range(6):
                    key = (unit["id"], arm, alpha, qi)
                    answers.append({"id": unit["id"], "arm": arm, "alpha": alpha,
                                    "question_index": qi, "question": score118.QUESTIONS[qi],
                                    "answer": f"fixture {key}"})
                    label = {"id": unit["id"], "arm": arm, "alpha": alpha,
                             "question_index": qi, **_label(unit, arm, alpha, qi, mode)}
                    labels.append(label)
                    by_key[score118.cell_key(label)] = label
    ds = [dict(by_key[key]) for key in score118.seeded_sample(score118.expected_keys(UNITS, ALPHAS))]
    return answers, labels, ds


def test_registered_predictions_hold_on_synthetic_fixture():
    answers, primary, ds = synthetic_rows("hold")
    result = score118.score_data(answers, primary, ds, UNITS, ALPHAS)
    assert result["ready"]
    assert result["gate2"]["passed"]
    assert set(result["predictions"]) == set(score118.REGISTERED)
    assert {row["verdict"] for row in result["predictions"].values()} == {"held"}


def test_registered_predictions_fail_on_synthetic_fixture():
    answers, primary, ds = synthetic_rows("fail")
    result = score118.score_data(answers, primary, ds, UNITS, ALPHAS)
    assert result["ready"]
    assert result["gate2"]["passed"]
    assert {row["verdict"] for row in result["predictions"].values()} == {"failed"}


def test_missing_primary_label_is_not_ready():
    answers, primary, ds = synthetic_rows("hold")
    primary.pop()
    result = score118.score_data(answers, primary, ds, UNITS, ALPHAS)
    assert not result["ready"]
    assert len(result["missing_primary_labels"]) == 1
    assert result["predictions"] == {}


def test_gate2_below_point_six_makes_predictions_not_read():
    answers, primary, ds = synthetic_rows("hold")
    sample = score118.seeded_sample(score118.expected_keys(UNITS, ALPHAS))
    primary_map = {score118.cell_key(row): row for row in primary}
    ds_map = {score118.cell_key(row): row for row in ds}
    for index, key in enumerate(sample):
        primary_map[key]["shows_persona"] = index % 2 == 0
        ds_map[key]["shows_persona"] = index % 2 != 0
    result = score118.score_data(answers, primary, ds, UNITS, ALPHAS)
    assert result["gate2"]["kappa"] < 0.6
    assert result["gate2_reading"] == "NOT READ"
    assert {row["verdict"] for row in result["predictions"].values()} == {"NOT READ"}


def test_random_arm_answers_do_not_enter_neuron_kind_counts():
    answers, primary, ds = synthetic_rows("hold")
    for row in primary:
        if row["arm"] == "random":
            row["shows_persona"] = True
            row["coherent"] = True
    result = score118.score_data(answers, primary, ds, UNITS, ALPHAS)
    concept_08 = next(row for row in result["classification_counts"]
                      if row["alpha"] == 0.8 and row["kind"] == "concept")
    assert concept_08["concept-like"] == 0
    assert concept_08["slot-like"] == 9


def test_incoherent_answer_does_not_show_persona():
    incoherent_labels = {qi: {"shows_persona": True, "self_claim": False,
                              "speaker": "ai", "coherent": False}
                         for qi in range(6)}
    assert score118.classify_unit(incoherent_labels) == "none"


def test_gate1_reports_one_differing_answer():
    current = [{"id": "u", "arm": "neuron", "alpha": 0.8,
                "question_index": qi, "answer": f"answer {qi}"} for qi in range(6)]
    stored = [{"id": "u", "unit": "neuron", "question_index": qi,
               "answer": f"answer {qi}"} for qi in range(6)]
    stored[2]["answer"] = "different"
    result = gate1_118.compare_answers(current, stored)
    assert result["matches"] == 5
    assert result["compared"] == 6
    assert len(result["differences"]) == 1
    assert not result["passed"]

