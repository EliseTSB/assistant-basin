"""Synthetic CPU fixtures for Part 142 scoring mechanics."""
import json
import random
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "part136"))
import score142
import scan136


def test_reference_is_mean_of_eight_matched_poles_and_excludes_own_match():
    score142.score136.BANDS = ((0, 31, "all"),)
    controls = {}
    for i in range(9):
        vec = [0.0] * 10
        vec[i] = 1.0
        controls[f"c{i}"] = [{"layer": 10, "effect_a": vec} for _ in range(24)]
    own = [{"layer": 10, "effect_a": [1.0] + [0.0] * 9} for _ in range(24)]
    got = score142.reference_for("unit", own, controls, "c0", score142.score136.band)
    selected = random.Random("142:unit").sample(sorted([f"c{i}" for i in range(1, 9)]), 8)
    expected = sum(score142.score136.pearson(own[0]["effect_a"], controls[p][0]["effect_a"]) for p in selected) / 8
    includes_own_match = sum(score142.score136.pearson(own[0]["effect_a"], controls[p][0]["effect_a"]) for p in controls) / 9
    assert got == pytest.approx(expected)
    assert got != pytest.approx(includes_own_match)


def test_margin_is_raw_minus_reference():
    assert score142.compute_margin(.72, .19) == pytest.approx(.53)
    assert score142.compute_margin(None, .2) is None


def test_full_scan_keeps_all_unequal_kinds_and_small_selection_stays_balanced():
    poles = ([{"pole_id": f"c{i}", "layer": 1, "kind": "concept"} for i in range(257)] +
             [{"pole_id": f"s{i}", "layer": 2, "kind": "slot"} for i in range(189)])
    bands = ((0, 31, "all"),)
    full = scan136.select_pilot_poles(poles, 446, bands)
    assert len(full) == 446
    assert sum(p["kind"] == "concept" for p in full) == 257
    assert sum(p["kind"] == "slot" for p in full) == 189
    small = scan136.select_pilot_poles(poles, 4, bands)
    assert sum(p["kind"] == "concept" for p in small) == 2
    assert sum(p["kind"] == "slot" for p in small) == 2


def test_local_pearson_leaves_zero_variance_undefined_and_stability_filters_it():
    assert score142.pearson([0, 0, 0], [1, 2, 3]) is None
    assert score142.pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)
    assert score142.stability([[0, 0, 0], [1, 2, 3], [2, 4, 6]]) == pytest.approx(1)


def test_142_3_is_band_assigned_sorted_median_minus_matched_median_outside_pilot():
    rows = [
        {"pole_id": "u1", "band": "low", "margin": 10},
        {"pole_id": "pilot", "band": "low", "margin": 100},
        {"pole_id": "u2", "band": "high", "margin": 0},
    ]
    controls = [
        {"pole_id": "c1", "band": "low", "margin": 0},
        {"pole_id": "pilot-control", "band": "low", "margin": -100},
        {"pole_id": "c2", "band": "high", "margin": 10},
    ]
    got = score142.sorted_vs_matched_median_difference(rows, controls, ((0, 1, "low"), (2, 3, "high")),
                                                        excluded=("pilot", "pilot-control"))
    assert got["difference"] == pytest.approx(0)
    assert got["n"] == 4
    assert got["n_by_band"] == {"low": {"sorted_units": 1, "matched_poles": 1},
                                "high": {"sorted_units": 1, "matched_poles": 1}}


def test_permutation_p_known_answer_and_reports_n():
    rows = [
        {"pole_id": "c", "band": "only", "kind": "concept", "margin": 1.0},
        {"pole_id": "s", "band": "only", "kind": "slot", "margin": 0.0},
    ]
    got = score142.margin_permutation(rows, "margin", ((0, 1, "only"),), n_perm=25, seed=19)
    assert got["difference"] == pytest.approx(1)
    assert got["p_two_sided"] == 1
    assert got["n"] == 2 and got["n_by_band"] == {"only": 2}


@pytest.mark.parametrize("failed_gate", ["gate2", "gate3"])
def test_each_reading_gate_can_stop_run_with_gate1_true(failed_gate):
    gate1, gate2, gate3 = True, True, True
    if failed_gate == "gate2":
        gate2 = False
    else:
        gate3 = False
    assert not score142.registered_reading_permitted(gate1, gate2, gate3)
    assert score142.registered_reading_permitted(True, True, True)


def test_missing_required_inputs_raise(tmp_path, monkeypatch):
    scan, effects = tmp_path / "scan", tmp_path / "effects"
    scan.mkdir(); effects.mkdir()
    units = tmp_path / "units.json"; units.write_text("[]")
    score = tmp_path / "score.json"; score.write_text('{"units":{}}')
    v2 = tmp_path / "v2.jsonl"; v2.write_text("")
    pilot = tmp_path / "pilot.json"; pilot.write_text('{"units":[],"matched_poles":[]}')
    monkeypatch.setattr(sys, "argv", ["score142.py", "--scan", str(scan), "--effects", str(effects),
        "--out", str(tmp_path / "out"), "--units", str(units), "--score", str(score), "--v2", str(v2),
        "--pilot-units", str(pilot), "--tokenizer", "fixture"])
    with pytest.raises(FileNotFoundError, match="missing required input"):
        score142.main()


def test_median_filters_none_and_empty_group_is_not_a_pass():
    assert score142.safe_median([None, 2, 4]) == 3
    assert score142.safe_median([None, None]) is None
    got = score142.sorted_vs_matched_median_difference(
        [{"pole_id": "u", "band": "only", "margin": None}],
        [{"pole_id": "c", "band": "only", "margin": 1}], ((0, 1, "only"),))
    assert got == {"difference": None, "n": 1,
            "n_by_band": {"only": {"sorted_units": 0, "matched_poles": 1}}}


def test_primary_permutation_excludes_pilot_units_and_their_matched_poles():
    bands = ((0, 5, "low"),)
    rows = [
        {"pole_id": "c", "band": "low", "kind": "concept", "margin": 1.0},
        {"pole_id": "s", "band": "low", "kind": "slot", "margin": 0.0},
        {"pole_id": "pilot", "band": "low", "kind": "concept", "margin": 100.0},
    ]
    result = score142.margin_permutation(rows, "margin", bands, excluded=("pilot", "pilot-control"), n_perm=0)
    inclusive = score142.margin_permutation(rows, "margin", bands, n_perm=0)
    assert result["difference"] == pytest.approx(1.0)
    assert inclusive["difference"] == pytest.approx(50.5)


class MarkerTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "aliens" if 999 in ids else "ordinary text"


def test_known_unit_count_uses_only_the_128_token_window():
    inside = {"token_ids": [1] * 127 + [999]}
    outside = {"token_ids": [1] * 128 + [999]}
    assert score142.decoded_known_count([inside, outside], MarkerTokenizer()) == 1


def test_nearest_rank_99th_percentile():
    assert score142.nearest_rank(list(range(1, 101)), .99) == 99
    assert score142.nearest_rank([], .99) is None


def test_gate_failed_score_writes_no_prediction(tmp_path, monkeypatch):
    # Build a complete label table so main reaches its normal gate-only branch.
    units = [{"id": f"u{i}", "layer": i % 32, "neuron": i, "persona": f"unit {i}"} for i in range(446)]
    score = {"units": {u["id"]: {"kind": "concept" if i % 2 == 0 else "slot"} for i, u in enumerate(units)}}
    scan = tmp_path / "scan"; effects = tmp_path / "effects"; out = tmp_path / "out"
    scan.mkdir(); effects.mkdir()
    (scan / "firings.jsonl").write_text("")
    (scan / "scan_meta.json").write_text(json.dumps({"matched_poles": []}))
    for name in ("effects.jsonl", "steering.jsonl"):
        (effects / name).write_text("")
    (effects / "gates.json").write_text(json.dumps({"do_nothing_hook_bit_identical": False}))
    units_path = tmp_path / "units.json"; units_path.write_text(json.dumps(units))
    score_path = tmp_path / "score.json"; score_path.write_text(json.dumps(score))
    v2_path = tmp_path / "v2.jsonl"; v2_path.write_text("")
    pilot_path = tmp_path / "pilot.json"; pilot_path.write_text(json.dumps({"units": [], "matched_poles": []}))
    fake_transformers = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: MarkerTokenizer()))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(sys, "argv", ["score142.py", "--scan", str(scan), "--effects", str(effects), "--out", str(out),
        "--units", str(units_path), "--score", str(score_path), "--v2", str(v2_path), "--pilot-units", str(pilot_path), "--tokenizer", "fixture"])
    score142.main()
    payload = json.loads((out / "score142.json").read_text())
    assert payload["gates"]["registered_reading_permitted"] is False
    assert "predictions" not in payload
    assert "Predictions" not in (out / "SCORE.md").read_text()


def test_a_nan_vector_is_undefined_and_stops_the_reading():
    nan=float("nan")
    assert score142.pearson([1.0, nan, 3.0], [1.0, 2.0, 3.0]) is None
    assert score142.count_nonfinite_rows([{"effect_a": [nan, 1.0], "effect_b": [0.0, 1.0]}, {"effect_a": [0.0, 1.0], "effect_b": [0.0, 1.0]}]) == 1
    assert score142.registered_reading_permitted(True, True, True, False) is False

