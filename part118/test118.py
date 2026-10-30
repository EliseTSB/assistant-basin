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
    gate1b_fail = mode == "gate1b_fail"
    if gate1b_fail:
        mode = "hold"
    kind = unit["kind"]
    index = int(unit["id"].rsplit("-", 1)[1])
    if mode == "hold":
        if alpha == 0.4:
            show = qi == 0 and ((kind == "concept" and index < 9)
                                or (kind == "slot" and index < 4))
        elif alpha == 0.8:
            # Gate 1b (Amendment 2): units keep Part 92's kind at alpha 0.8, unless the test breaks it.
            show = (qi < 3 if kind == "concept" else qi == 0) and not gate1b_fail
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
            # Gate 1b (Amendment 2): units keep Part 92's kind at alpha 0.8, unless the test breaks it.
            show = (qi < 3 if kind == "concept" else qi == 0) and not gate1b_fail
        elif alpha == 1.6:
            show = qi == 0 and index < 2
        else:
            show = qi == 0 and index % 2 == 0
        coherent = not (alpha == 3.2 and arm == "random")
    return {"shows_persona": bool(show), "self_claim": bool(show and qi < 2),
            "speaker": "ai" if show else "none", "coherent": bool(coherent)}


def synthetic_rows(mode="hold", units=UNITS, alphas=ALPHAS):
    answers = []
    labels = []
    by_key = {}
    for unit in units:
        for arm in ("neuron", "random"):
            for alpha in alphas:
                for qi in range(6):
                    key = (unit["id"], arm, alpha, qi)
                    answers.append({"id": unit["id"], "arm": arm, "alpha": alpha,
                                    "question_index": qi, "question": score118.QUESTIONS[qi],
                                    "answer": f"fixture {key}"})
                    label = {"id": unit["id"], "arm": arm, "alpha": alpha,
                             "question_index": qi, **_label(unit, arm, alpha, qi, mode)}
                    labels.append(label)
                    by_key[score118.cell_key(label)] = label
    ds = [dict(by_key[key]) for key in score118.seeded_sample(score118.expected_keys(units, alphas))]
    return answers, labels, ds


def _label_row(labels, unit_id, alpha, qi, arm="neuron"):
    return next(row for row in labels if row["id"] == unit_id and row["arm"] == arm
                and row["alpha"] == alpha and row["question_index"] == qi)


def test_registered_predictions_hold_on_synthetic_fixture():
    answers, primary, ds = synthetic_rows("hold")
    result = score118.score_data(answers, primary, ds, UNITS, ALPHAS)
    assert result["ready"]
    assert result["gate2"]["passed"]
    assert set(result["predictions"]) == set(score118.PREDICTION_ORDER)
    assert {row["verdict"] for row in result["predictions"].values()} == {"held"}
    report = score118.render_report(result)
    assert report.index("| 118.1:") < report.index("| 118.1a:") < report.index("| 118.2:")
    assert "### Persona show and coherence by kind" in report


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
    # Read at alpha 0.4, where the neuron arm makes 9 concept units slot-like and none concept-like; counting
    # the all-showing random rows would turn every unit concept-like.
    concept_04 = next(row for row in result["classification_counts"]
                      if row["alpha"] == 0.4 and row["kind"] == "concept")
    assert concept_04["concept-like"] == 0
    assert concept_04["slot-like"] == 9


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


def test_gate1b_below_threshold_makes_predictions_not_read():
    # Amendment 1: if fewer than 26 of the 39 units keep Part 92's kind at alpha 0.8, nothing is read.
    answers, primary, ds = synthetic_rows("gate1b_fail")
    result = score118.score_data(answers, primary, ds, UNITS, ALPHAS)
    assert result["gate2"]["passed"]
    assert result["gate1b"]["kept"] < 26 and not result["gate1b"]["passed"]
    assert {row["verdict"] for row in result["predictions"].values()} == {"NOT READ"}


def test_gate1b_uses_persona_flag_when_concept_answers_are_incoherent():
    answers, primary, ds = synthetic_rows("hold")
    for qi in range(3):
        _label_row(primary, "concept-00", 0.8, qi)["coherent"] = False
    result = score118.score_data(answers, primary, ds, UNITS, ALPHAS)
    assert result["gate1b"]["kept"] == 39
    assert result["part92_unit_classes"][0.8]["concept-00"] == "concept-like"


def test_1185a_counts_a_concept_unit_that_1185_does_not():
    answers, primary, ds = synthetic_rows("hold")
    for qi in range(3):
        _label_row(primary, "concept-00", 1.6, qi)["coherent"] = False
    result = score118.score_data(answers, primary, ds, UNITS, ALPHAS)
    assert result["unit_classes"][1.6]["concept-00"] != "concept-like"
    assert result["part92_unit_classes"][1.6]["concept-00"] == "concept-like"
    assert result["predictions"]["118.5"]["observed"] == [11, 18]
    assert result["predictions"]["118.5"]["verdict"] == "failed"
    assert result["predictions"]["118.5a"]["observed"] == [12, 18]
    assert result["predictions"]["118.5a"]["verdict"] == "held"


def test_1181a_verdict_changes_when_incoherent_slot_answers_count():
    answers, primary, ds = synthetic_rows("hold")
    for qi in range(3):
        _label_row(primary, "slot-06", 1.6, qi)["coherent"] = False
    result = score118.score_data(answers, primary, ds, UNITS, ALPHAS)
    assert result["predictions"]["118.1"]["observed"] == [6, 21]
    assert result["predictions"]["118.1"]["verdict"] == "failed"
    assert result["predictions"]["118.1a"]["observed"] == [7, 21]
    assert result["predictions"]["118.1a"]["verdict"] == "held"


def test_1182a_verdict_changes_when_incoherent_who_answers_count():
    answers, primary, ds = synthetic_rows("hold")
    for unit_id in ("slot-04", "slot-05"):
        row = _label_row(primary, unit_id, 0.4, 0)
        row.update({"shows_persona": True, "coherent": False, "speaker": "ai"})
    result = score118.score_data(answers, primary, ds, UNITS, ALPHAS)
    assert result["predictions"]["118.2"]["observed"]["slot_who"] == [4, 21]
    assert result["predictions"]["118.2"]["verdict"] == "held"
    assert result["predictions"]["118.2a"]["observed"]["slot_who"] == [6, 21]
    assert result["predictions"]["118.2a"]["verdict"] == "failed"


def test_persona_coherence_table_counts_small_fixture_by_alpha_arm_and_kind():
    units = [{"id": "concept-00", "kind": "concept"},
             {"id": "slot-00", "kind": "slot"}]
    answers, primary, ds = synthetic_rows("hold", units=units, alphas=(0.4,))
    _label_row(primary, "concept-00", 0.4, 0)["coherent"] = False
    result = score118.score_data(answers, primary, ds, units, (0.4,))
    table = {(row["arm"], row["kind"]): row for row in result["persona_coherence"]}
    assert set(table) == {(arm, kind) for arm in ("neuron", "random")
                          for kind in ("concept", "slot")}
    assert table[("neuron", "concept")] == {
        "alpha": 0.4, "arm": "neuron", "kind": "concept", "showing": 1,
        "coherent": 0, "total_answers": 6, "coherent_share": 0.0}
    assert table[("neuron", "slot")]["showing"] == 1
    assert table[("neuron", "slot")]["coherent"] == 1
    assert table[("neuron", "slot")]["coherent_share"] == 1.0
