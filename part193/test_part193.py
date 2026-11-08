"""Small, model-free checks for Part 193 statistics and edit construction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats193 as s193


def test_leave_one_out_robust_z_excludes_the_matrix_itself():
    values = np.array([0.0, 1.0, 2.0, 100.0])
    actual = s193.leave_one_out_robust_z(values)[0]
    expected = s193.R1 * (0.0 - 2.0) / 1.0
    assert actual == expected
    assert actual != s193.robust_z(values[0], values)


def test_top_ties_are_not_found():
    result = s193.rank_value(7.0, [7.0, 7.0, 6.0])
    assert result == {"rank": 1, "ties": 2, "found": False}
    assert s193.rank_value(8.0, [7.0, 7.0])["found"] is True


def test_threshold_rule_handles_monotone_dips_and_lost_detection():
    doses = [1, 2, 4, 8, 16, 32, 64]
    monotone = s193.threshold_summary([False, False, False, True, True, True, True], doses)
    assert monotone["threshold"] == 8.0

    dip = s193.threshold_summary([False, False, True, True, False, False, True], doses)
    assert dip["threshold"] == 64.0
    assert dip["found_below_threshold"] == [4.0, 8.0]
    assert dip["dips"] == [{"from_dose": 8.0, "at_dose": 16.0}]

    only_last = s193.threshold_summary([False, False, False, False, False, False, True], doses)
    assert only_last["threshold"] == 64.0
    assert only_last["found_below_threshold"] == []

    lost = s193.threshold_summary([False, False, True, False, False, False, False], doses)
    assert lost["threshold"] == "never"
    assert lost["found_below_threshold"] == [4.0]
    assert lost["dips"] == [{"from_dose": 4.0, "at_dose": 8.0}]


def test_f3_finds_a_planted_neuron():
    rng = np.random.default_rng(31)
    gate = rng.normal(size=(12, 5))
    up = rng.normal(size=(12, 5))
    down = rng.normal(size=(5, 12))
    planted = 6
    gate[planted] *= 100.0
    features, kurtosis = s193.neuron_feature_values(gate, up, down)
    z_values, scores = s193.f3_scores(features)
    assert features.shape == (12, 4)
    assert z_values.shape == (12, 4)
    assert kurtosis.shape == (12, 3)
    assert int(np.argmax(scores)) == planted


def test_r1_reports_only_nonzero_planted_differences():
    zero = np.zeros((3, 4), dtype=np.float64)
    change = np.zeros((3, 4), dtype=np.float64)
    change[1, 2] = 0.25
    assert s193.r1_entries({"no-op": zero}) == []
    entries = s193.r1_entries({"no-op": zero, "planted": change})
    assert len(entries) == 1
    assert entries[0]["key"] == "planted"
    assert entries[0]["rank"] == 1


def test_synthetic_placements_are_seeded_and_reproducible():
    population = [
        {"key": f"matrix-{index}", "shape": [3 + index % 2, 4]}
        for index in range(7)
    ]
    first = s193.seeded_rank_one_placements(population, count=24, seed=20261108)
    second = s193.seeded_rank_one_placements(population, count=24, seed=20261108)
    assert len(first) == 24
    for left, right in zip(first, second):
        assert left["key"] == right["key"]
        np.testing.assert_array_equal(left["u"], right["u"])
        np.testing.assert_array_equal(left["v"], right["v"])
        assert np.isclose(np.linalg.norm(left["u"]), 1.0)
        assert np.isclose(np.linalg.norm(left["v"]), 1.0)


def test_w1_ladder_keeps_rows_fixed_and_scales_only_down_column():
    base_gate = np.arange(12, dtype=np.float64).reshape(4, 3)
    base_up = base_gate + 20.0
    base_down = np.arange(12, dtype=np.float64).reshape(3, 4)
    neuron = 2
    gate_row = np.array([101.0, 102.0, 103.0])
    up_row = np.array([201.0, 202.0, 203.0])
    down_column = np.array([-3.0, -4.0, -5.0])

    half = s193.apply_w1_dose(
        base_gate, base_up, base_down, neuron, gate_row, up_row, down_column, 0.5
    )
    twice = s193.apply_w1_dose(
        base_gate, base_up, base_down, neuron, gate_row, up_row, down_column, 2.0
    )
    for gate, up, down in (half, twice):
        np.testing.assert_array_equal(gate[neuron], gate_row)
        np.testing.assert_array_equal(up[neuron], up_row)
        np.testing.assert_array_equal(np.delete(gate, neuron, axis=0), np.delete(base_gate, neuron, axis=0))
        np.testing.assert_array_equal(np.delete(up, neuron, axis=0), np.delete(base_up, neuron, axis=0))
        np.testing.assert_array_equal(np.delete(down, neuron, axis=1), np.delete(base_down, neuron, axis=1))
    np.testing.assert_array_equal(half[2][:, neuron], down_column * 0.5)
    np.testing.assert_array_equal(twice[2][:, neuron], down_column * 2.0)


def _tiny_safetensors(path: Path) -> None:
    rng = np.random.default_rng(193)
    tensors = {}
    for layer in s193.LAYERS:
        for role in s193.ROLES:
            if role in {"mlp.gate_proj", "mlp.up_proj"}:
                shape = (4, 3)
            elif role == "mlp.down_proj":
                shape = (3, 4)
            else:
                shape = (3, 3)
            tensors[s193.weight_key(layer, role)] = (
                rng.normal(size=shape).astype(np.float32) + layer * 0.01
            )
    tensors["outside_embedding.weight"] = np.ones((5, 3), dtype=np.float32)
    save_file(tensors, str(path))


def test_tiny_fake_safetensors_population_and_float64_statistics(tmp_path):
    path = tmp_path / "tiny.safetensors"
    _tiny_safetensors(path)
    with s193.safe_open(str(path), framework="pt", device="cpu") as handle:
        population, outside = s193.discover_population(handle)
        assert len(population) == 216
        assert len(outside) == 1
        assert outside[0]["key"] == "outside_embedding.weight"
        matrix = s193.tensor_float64(handle, population[0]["key"])
        assert matrix.dtype == np.float64
        assert s193.matrix_metrics(matrix)["sigma1"] > 0.0
        records = s193.base_matrix_statistics(handle, population)
        assert len(records) == 216
        neurons = s193.base_neuron_statistics(handle)
        assert len(neurons) == 24
        assert len(s193.f3_layer_null(neurons)) == 24

