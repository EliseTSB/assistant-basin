"""Synthetic gate, prediction, and phase-hook checks for Part 143."""
from __future__ import annotations

import copy
import json

import pytest
import torch

import run143
import score143


def fixture_data():
    rows = [
        {"key": "plain|im", "readout": "im_after_i", "arm": "plain", "set": "plain", "unit": "plain", "unit_id": "plain", "probability": .05},
        {"key": "plain|sorry", "readout": "sorry_after_im", "arm": "plain", "set": "plain", "unit": "plain", "unit_id": "plain", "probability": .01},
    ]
    units = [("r1", "R", "neuron"), ("r2", "R", "random"),
             ("s1", "S", "neuron"), ("s2", "S", "random")]
    sorry_values = {
        "noop": {"r1": .01, "r2": .01, "s1": .01, "s2": .01},
        "prompt-only": {"r1": .2, "r2": .4, "s1": .01, "s2": .02},
        "answer-only": {"r1": .01, "r2": .02, "s1": .01, "s2": .01},
        "full": {"r1": .7, "r2": .9, "s1": .02, "s2": .04},
    }
    im_values = {
        "noop": {"r1": .05, "r2": .05, "s1": .05, "s2": .05},
        "prompt-only": {"r1": .3, "r2": .5, "s1": .05, "s2": .05},
        "answer-only": {"r1": .1, "r2": .2, "s1": .05, "s2": .05},
        "full": {"r1": .5, "r2": .6, "s1": .1, "s2": .1},
    }
    for arm in ("noop", "prompt-only", "answer-only", "full"):
        for unit_id, set_name, unit_type in units:
            for readout, table in (("im_after_i", im_values), ("sorry_after_im", sorry_values)):
                rows.append({
                    "key": f"{readout}|{arm}|{unit_id}", "readout": readout,
                    "arm": arm, "set": set_name, "unit": unit_type,
                    "unit_id": unit_id, "probability": table[arm][unit_id],
                })
    old = [
        {"stage": "A", "arm": "forced_prefix", "unit_id": "r1", "set": "R", "sorry_probability": .69},
        {"stage": "A", "arm": "forced_prefix", "unit_id": "r2", "set": "R", "sorry_probability": .88},
    ]
    return rows, old


def test_known_gates_predictions_and_medians():
    rows, old = fixture_data()
    result = score143.score(rows, old)
    assert result["gates"]["gate1_noop_matches_plain"]["passed"]
    assert result["gates"]["gate2_full_matches_part139"]["passed"]
    predictions = result["predictions"]
    assert predictions["143.1"]["verdict"] == "held"
    assert predictions["143.2"]["verdict"] == "failed"
    assert predictions["143.3"]["verdict"] == "held"
    assert predictions["143.4"]["verdict"] == "held"
    assert predictions["143.5"]["verdict"] == "held"
    stat = result["medians"]["sorry_after_im"]["full"]["R"]["neuron"]
    assert stat["n"] == 1 and stat["median"] == .7
    assert stat["middle_50_percent"] == {"q1": .7, "q3": .7}


def test_failed_gate_writes_gate_block_without_predictions(tmp_path):
    rows, old = fixture_data()
    next(row for row in rows if row["arm"] == "noop" and row["readout"] == "im_after_i" and row["unit_id"] == "r1")["probability"] += .002
    input_path = tmp_path / "results.jsonl"
    prior_path = tmp_path / "part139.jsonl"
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    prior_path.write_text("".join(json.dumps(row) + "\n" for row in old), encoding="utf-8")
    result = score143.score(score143.read_rows(input_path), score143.read_rows(prior_path))
    assert not result["gates"]["gate1_noop_matches_plain"]["passed"]
    assert "predictions" not in result and "medians" not in result
    assert "registered predictions are withheld" in score143.render_markdown(result)



def test_missing_noop_rows_fail_gate1():
    rows, old = fixture_data()
    rows = [row for row in rows if not (row.get("arm") == "noop" and row.get("unit_id") == "r1")]
    result = score143.score(rows, old)
    assert not result["gates"]["gate1_noop_matches_plain"]["passed"]
    assert "predictions" not in result


def test_gate2_failure_withholds_predictions():
    rows, old = fixture_data()
    next(row for row in old if row["unit_id"] == "r2")["sorry_probability"] = 0.0
    result = score143.score(rows, old)
    assert not result["gates"]["gate2_full_matches_part139"]["passed"]
    assert "predictions" not in result and "medians" not in result

def test_missing_current_r_unit_counts_against_gate2():
    rows, old = fixture_data()
    rows = [row for row in rows if not (row.get("readout") == "sorry_after_im" and row.get("arm") == "full" and row.get("unit_id") == "r2")]
    result = score143.score(rows, old)
    gate = result["gates"]["gate2_full_matches_part139"]
    assert not gate["passed"]
    assert gate["n"] == 2 and gate["matching_rows"] == 1
    assert "predictions" not in result


def test_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        score143.read_rows(tmp_path / "missing.jsonl")


class FakeLayer:
    def register_forward_hook(self, callback):
        self.callback = callback
        return type("Handle", (), {"remove": lambda self: None})()


def test_each_arm_reaches_the_intended_phase_calls():
    vector = torch.ones((1, 3))
    expected = {
        "noop": ([False, False], [True, True]),
        "prompt-only": ([True, False], [True, False]),
        "answer-only": ([False, True], [False, True]),
        "full": ([True, True], [True, True]),
    }
    for arm, (changed_wanted, returned_wanted) in expected.items():
        layer = FakeLayer()
        hook_vector = torch.zeros_like(vector) if arm == "noop" else vector
        hook = run143.run139._position_hook(layer, hook_vector, arm)
        changed, returned = [], []
        for _call in range(2):
            hidden = torch.zeros((1, 1, 3))
            result = layer.callback(layer, (), hidden)
            if result is None:
                result = hidden
            changed.append(bool(torch.any(result != hidden).item()))
            returned.append(result.data_ptr() != hidden.data_ptr())
        hook.remove()
        assert changed == changed_wanted
        assert returned == returned_wanted


