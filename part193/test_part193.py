"""Small, model-free checks for Part 193 statistics and edit construction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shlex
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edit_provenance import saved_edit_provenance
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


def test_prediction_193_4_rejects_disjoint_dip_and_accepts_agreeing_patterns(capsys):
    disjoint_dip = s193.prediction_193_4(
        [False, False, False, False, False, True, True, True, True],
        [True, True, True, False, True, True, True, True, True],
    )
    assert disjoint_dip["W2"]["threshold"] == 8.0
    assert disjoint_dip["W2_control"]["threshold"] == 4.0
    assert disjoint_dip["thresholds_within_one_ladder_step"] is True
    assert disjoint_dip["patterns_agree_outside_threshold_band"] is False
    assert disjoint_dip["holds"] is False
    print(s193.format_prediction_193_4(disjoint_dip))
    assert capsys.readouterr().out.strip() == (
        "193.4: 1=found W2=000001111 W2c=111011111 thresholds=(8,4) "
        "within_one_step=True agree_outside_band=False verdict=FAIL"
    )

    agreeing = s193.prediction_193_4(
        [False, False, False, False, True, True, True, True, True],
        [False, False, False, False, False, True, True, True, True],
    )
    assert agreeing["W2"]["threshold"] == 4.0
    assert agreeing["W2_control"]["threshold"] == 8.0
    assert agreeing["patterns_agree_outside_threshold_band"] is True
    assert agreeing["holds"] is True
    assert agreeing["verdict"] == "holds"


def test_prediction_193_4_never_sits_one_step_past_the_top_dose():
    """'never' is one position after 64, the band's convention: never vs 64 is one step, never vs 32 is two."""
    off = [False] * 9
    top_only = [False] * 8 + [True]
    top_two = [False] * 7 + [True, True]
    near = s193.prediction_193_4(off, top_only)
    assert near["thresholds_within_one_ladder_step"] is True
    assert near["patterns_agree_outside_threshold_band"] is True
    assert near["holds"] is True
    far = s193.prediction_193_4(off, top_two)
    assert far["thresholds_within_one_ladder_step"] is False
    assert far["holds"] is False
    both_never = s193.prediction_193_4(off, off)
    assert both_never["thresholds_within_one_ladder_step"] is True
    assert both_never["holds"] is True


def test_prediction_193_4_two_numeric_steps_apart_fails_even_when_patterns_agree():
    w2 = [False] * 4 + [True] * 5      # threshold 4 (index 4)
    w2c = [False] * 6 + [True] * 3     # threshold 16 (index 6)
    result = s193.prediction_193_4(w2, w2c)
    assert result["patterns_agree_outside_threshold_band"] is True
    assert result["thresholds_within_one_ladder_step"] is False
    assert result["holds"] is False

def test_f3_layer_rank_compares_raw_layer_tops(monkeypatch):
    def fixed_scores(features):
        return np.zeros_like(features), np.asarray([6.0, 1.0, 2.0, 3.0])

    monkeypatch.setattr(s193, "f3_scores", fixed_scores)
    result = s193.candidate_f3(
        {"features": np.ones((4, 4))}, 0,
        np.ones(3), np.ones(3), np.ones(3), np.asarray([4.0, 5.0]),
    )
    assert result["layer_top_neuron_f3"] == 6.0
    assert result["layer_rank"]["found"] is True
    assert result["layer_f3_z"] == s193.robust_z(6.0, [4.0, 5.0])


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


def test_r1_rank_one_float32_uses_saved_dtype_tolerance_and_reports_ratio():
    left = np.linspace(0.25, 3.0, 96, dtype=np.float32)
    right = np.linspace(-2.0, 1.0, 128, dtype=np.float32)
    delta = np.outer(left, right).astype(np.float32)
    assert s193.matrix_rank_from_singular_values(delta) == 1
    entry = s193.r1_entries({"rank-one-float32": delta})[0]
    assert entry["rank"] == 1
    assert entry["sigma2_over_sigma1"] < max(delta.shape) * np.finfo(delta.dtype).eps


def test_dose_zero_candidate_preserves_checkpoint_bits():
    base = np.asarray([-0.0, 1.25], dtype=np.float32)
    candidate = s193.no_op_edit(base, np.asarray([2.0, -3.0], dtype=np.float32))
    assert candidate.tobytes() == base.tobytes()
    assert s193.arrays_bitwise_equal(candidate, base)
    assert not s193.arrays_bitwise_equal(
        np.asarray([-0.0], dtype=np.float32), np.asarray([0.0], dtype=np.float32)
    )


def test_sibling_rank_uses_candidate_and_siblings_on_one_reference_set():
    records = [
        {"key": "target", "layer": 0, "role": "role", "f1": 3.0,
         "f1_z": 0.0, "f2_z": 0.0},
        {"key": "peer-low", "layer": 1, "role": "role", "f1": 1.0,
         "f1_z": 100.0, "f2_z": 0.0},
        {"key": "peer-high", "layer": 2, "role": "role", "f1": 2.0,
         "f1_z": 200.0, "f2_z": 0.0},
    ]
    result = s193._matrix_metric_result(
        "target", {"f1": 4.0, "f2": 0.5, "sigma1": 10.0}, records, "f1"
    )
    assert result["sibling_rank"] == 1
    assert result["found_among_siblings"] is True


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


def _gate_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, np.ndarray]]:
    rng = np.random.default_rng(193)
    tensors = {}
    for layer in s193.LAYERS:
        for role in s193.ROLES:
            if role in {"mlp.gate_proj", "mlp.up_proj"}:
                shape = (1084, 3) if layer == 20 else (4, 3)
            elif role == "mlp.down_proj":
                shape = (3, 1084) if layer == 20 else (3, 4)
            else:
                shape = (3, 3)
            tensors[s193.weight_key(layer, role)] = rng.normal(size=shape).astype(np.float32)
    tensors["language_model.embed_tokens.weight"] = rng.normal(size=(8, 4)).astype(np.float32)
    base_path = tmp_path / "tiny-gates.safetensors"
    save_file(tensors, str(base_path))

    up_key = s193.weight_key(20, "mlp.up_proj")
    gate_key = s193.weight_key(20, "mlp.gate_proj")
    down_key = s193.weight_key(20, "mlp.down_proj")
    neuron = 1083
    up_row = rng.normal(size=3).astype(np.float32)
    gate_row = rng.normal(size=3).astype(np.float32)
    down_col = rng.normal(size=3).astype(np.float32)
    up_delta = (up_row - tensors[up_key][neuron]).astype(np.float32)
    gate_delta = (gate_row - tensors[gate_key][neuron]).astype(np.float32)
    down_delta = (down_col - tensors[down_key][:, neuron]).astype(np.float32)
    up_full = np.zeros_like(tensors[up_key]); up_full[neuron, :] = up_delta
    gate_full = np.zeros_like(tensors[gate_key]); gate_full[neuron, :] = gate_delta
    down_full = np.zeros_like(tensors[down_key]); down_full[:, neuron] = down_delta
    w2_delta = (rng.normal(size=tensors[down_key].shape) * 0.02).astype(np.float32)
    w2c_delta = (rng.normal(size=tensors[down_key].shape) * -0.03).astype(np.float32)
    w2_patched = (tensors[down_key] + w2_delta).astype(np.float32)
    w2c_patched = (tensors[down_key] + w2c_delta).astype(np.float32)
    edits = {
        "w1_neuron_j": np.asarray(neuron, dtype=np.int64),
        "w1_up_row": up_row,
        "w1_gate_row": gate_row,
        "w1_down_col": down_col,
        "w1_up_delta": up_delta,
        "w1_gate_delta": gate_delta,
        "w1_down_delta": down_delta,
        "w1_up_full_delta": up_full,
        "w1_gate_full_delta": gate_full,
        "w1_down_full_delta": down_full,
        "w2_delta": (w2_patched - tensors[down_key]).astype(np.float32),
        "w2_patched": w2_patched,
        "w2c_delta": (w2c_patched - tensors[down_key]).astype(np.float32),
        "w2c_patched": w2c_patched,
    }
    settings = {
        "W1": {"s": 0.025, "tau_quantile": 0.3},
        "W2": {"s": 0.15},
        "W2_control": {"s": 0.025},
        "near_tie_fallbacks": {},
    }
    edits["settings_json"] = np.asarray(json.dumps(settings, sort_keys=True))
    edits["base_sha256"] = np.asarray(hashlib.sha256(base_path.read_bytes()).hexdigest())
    step0 = {
        "parity": {
            "weights_restored_bitwise": True,
            "post_restore_worst_tail_cosine_to_stored": 0.999999,
        },
        "arms": {
            "W1": {"test_auc": 0.6775},
            "W2": {"test_auc": 0.8125},
            "W2_control": {"test_auc": 0.6175},
        },
        "fit_selection": {
            "W1": {"selected": {"s": 0.025, "tau_quantile": 0.3}},
            "W2": {"selected": {"s": 0.15}},
            "W2_control": {"selected": {"s": 0.025}},
        },
        "near_tie_fallbacks": {},
        "saved_edit_provenance": saved_edit_provenance(edits),
    }
    step0_path = tmp_path / "step0.json"
    step0_path.write_text(json.dumps(step0))
    edits_path = tmp_path / "edits.npz"
    np.savez_compressed(edits_path, **edits)
    return base_path, step0_path, edits_path, edits


def _copy_edits(edits: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in edits.items()}


def _settings(edits: dict[str, np.ndarray]) -> dict[str, object]:
    return json.loads(str(edits["settings_json"].item()))


def test_step0_command_passes_weight_edit_argument_parser(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    steps = (project_root / "part193" / "steps.txt").read_text()
    step0 = next(
        block.split("```", 1)[0]
        for block in steps.split("```sh")[1:]
        if "weight_edit.py" in block
    )
    tokens = shlex.split(step0.replace("\\\n", " "))
    tokens = tokens[tokens.index("systemd-run"):]
    script_index = next(
        index for index, token in enumerate(tokens)
        if Path(token).name == "weight_edit.py"
    )
    argv = tokens[script_index + 1:]

    # Import only to build and run the script's parser; do not load its model.
    sentence_transformers_stub = types.ModuleType("sentence_transformers")
    sentence_transformers_stub.SentenceTransformer = object
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers_stub)
    script_path = project_root / "part193" / "chicken_edit" / "weight_edit.py"
    spec = importlib.util.spec_from_file_location("part193_weight_edit_parser", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parsed = module.build_argument_parser().parse_args(argv)

    assert parsed.out_other == Path("results/part193/other_items.json")
    assert parsed.results == Path("results/part193/weight_edit_results.json")
    assert parsed.save_edits == Path("results/part193/edits.npz")


def test_gate1_rejects_changed_saved_array_sha(tmp_path):
    base_path, step0_path, _, edits = _gate_fixture(tmp_path)
    step0 = json.loads(step0_path.read_text())
    step0["saved_edit_provenance"]["W2"]["arrays"]["w2_patched"]["sha256"] = "0" * 64
    step0_path.write_text(json.dumps(step0))

    with pytest.raises(
        s193.GateFailure,
        match="W2 saved array w2_patched does not match step-0 provenance sha256",
    ):
        s193.check_gate1(base_path, edits, _settings(edits), step0_path)


def test_gate1_rejects_changed_w1_neuron_with_consistent_provenance(tmp_path):
    base_path, step0_path, _, original_edits = _gate_fixture(tmp_path)
    edits = _copy_edits(original_edits)
    neuron = 1082
    edits["w1_neuron_j"] = np.asarray(neuron, dtype=np.int64)

    row_fields = (
        ("mlp.up_proj", "w1_up_row", "w1_up_delta", "w1_up_full_delta", 0),
        ("mlp.gate_proj", "w1_gate_row", "w1_gate_delta", "w1_gate_full_delta", 0),
        ("mlp.down_proj", "w1_down_col", "w1_down_delta", "w1_down_full_delta", 1),
    )
    with s193.safe_open(str(base_path), framework="np") as handle:
        for role, value_key, delta_key, full_key, axis in row_fields:
            base = handle.get_tensor(s193.weight_key(20, role))
            value = np.asarray(edits[value_key], dtype=np.float32)
            base_slice = base[neuron, :] if axis == 0 else base[:, neuron]
            delta = np.subtract(value, base_slice).astype(np.float32)
            full_delta = np.zeros_like(base)
            if axis == 0:
                full_delta[neuron, :] = delta
            else:
                full_delta[:, neuron] = delta
            edits[delta_key] = delta
            edits[full_key] = full_delta

    step0 = json.loads(step0_path.read_text())
    step0["saved_edit_provenance"] = saved_edit_provenance(edits)
    step0_path.write_text(json.dumps(step0))
    with pytest.raises(s193.GateFailure, match="W1 neuron is not the published neuron 1083"):
        s193.check_gate1(base_path, edits, _settings(edits), step0_path)


def test_gate2_fails_when_dose_zero_candidate_changes_a_bit(tmp_path, monkeypatch):
    base_path, _, _, edits = _gate_fixture(tmp_path)

    def changed_dose_zero(base, delta):
        candidate = np.asarray(base).copy()
        candidate.flat[0] = np.nextafter(candidate.flat[0], np.float32(np.inf))
        return candidate

    monkeypatch.setattr(s193, "no_op_edit", changed_dose_zero)
    with pytest.raises(s193.GateFailure, match="Gate 2 W1 dose 0 changed tensor"):
        s193.check_gate2_gate3(base_path, edits)


def test_gate_checks_provenance_zero_swapped_and_measured_support(tmp_path):
    base_path, step0_path, _, edits = _gate_fixture(tmp_path)
    settings = _settings(edits)

    passed = s193.check_gate1(base_path, edits, settings, step0_path)
    assert passed["passed"] is True
    gate23 = s193.check_gate2_gate3(base_path, edits)
    assert gate23["gate2"]["passed"] is True
    assert gate23["gate2"]["r1_changed_matrices"] == {"W1": [], "W2": [], "W2_control": []}
    assert gate23["gate3"]["passed"] is True

    zero = _copy_edits(edits)
    tensors = {}
    with s193.safe_open(str(base_path), framework="np") as handle:
        for key in (
            s193.weight_key(20, "mlp.up_proj"),
            s193.weight_key(20, "mlp.gate_proj"),
            s193.weight_key(20, "mlp.down_proj"),
        ):
            tensors[key] = handle.get_tensor(key)
    neuron = int(zero["w1_neuron_j"].item())
    zero["w1_up_row"] = tensors[s193.weight_key(20, "mlp.up_proj")][neuron].copy()
    zero["w1_gate_row"] = tensors[s193.weight_key(20, "mlp.gate_proj")][neuron].copy()
    zero["w1_down_col"] = tensors[s193.weight_key(20, "mlp.down_proj")][:, neuron].copy()
    for key in (
        "w1_up_delta", "w1_gate_delta", "w1_down_delta",
        "w1_up_full_delta", "w1_gate_full_delta", "w1_down_full_delta",
        "w2_delta", "w2c_delta",
    ):
        zero[key] = np.zeros_like(zero[key])
    zero["w2_patched"] = tensors[s193.weight_key(20, "mlp.down_proj")].copy()
    zero["w2c_patched"] = tensors[s193.weight_key(20, "mlp.down_proj")].copy()
    with pytest.raises(s193.GateFailure):
        s193.check_gate1(base_path, zero, settings, step0_path)
    with pytest.raises(s193.GateFailure):
        s193.check_gate2_gate3(base_path, zero)

    swapped = _copy_edits(edits)
    swapped["w2_delta"], swapped["w2c_delta"] = swapped["w2c_delta"], swapped["w2_delta"]
    swapped["w2_patched"], swapped["w2c_patched"] = swapped["w2c_patched"], swapped["w2_patched"]
    with pytest.raises(s193.GateFailure, match="provenance sha256"):
        s193.check_gate1(base_path, swapped, settings, step0_path)

    dense = _copy_edits(edits)
    rng = np.random.default_rng(99)
    for key in ("w1_up_full_delta", "w1_gate_full_delta", "w1_down_full_delta"):
        dense[key] = rng.normal(size=dense[key].shape).astype(np.float32)
    with pytest.raises(s193.GateFailure, match="nonzero support"):
        s193.check_gate2_gate3(base_path, dense)


def test_gate2_reads_only_the_edit_target_tensors(tmp_path, monkeypatch):
    base_path, _, _, edits = _gate_fixture(tmp_path)
    original_safe_open = s193.safe_open
    reads = []

    class TrackedHandle:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def get_tensor(self, key):
            reads.append(key)
            return self.handle.get_tensor(key)

        def __getattr__(self, name):
            return getattr(self.handle, name)

    monkeypatch.setattr(
        s193, "safe_open", lambda *args, **kwargs: TrackedHandle(original_safe_open(*args, **kwargs))
    )
    s193.check_gate2_gate3(base_path, edits)
    expected = {
        s193.weight_key(20, "mlp.up_proj"),
        s193.weight_key(20, "mlp.gate_proj"),
        s193.weight_key(20, "mlp.down_proj"),
    }
    assert set(reads) == expected
    assert "language_model.embed_tokens.weight" not in reads


def test_run_analysis_end_to_end_on_tiny_gate_fixture(tmp_path, monkeypatch):
    base_path, step0_path, edits_path, edits = _gate_fixture(tmp_path)
    # Use the exact captured array archive that Gate 1's JSON provenance describes.
    np.savez_compressed(edits_path, **edits)
    real_svd = np.linalg.svd
    calls = {"count": 0}

    def counted_svd(*args, **kwargs):
        calls["count"] += 1
        return real_svd(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", counted_svd)
    result = s193.run_analysis(base_path, edits_path, step0_path)
    assert result["gates"]["gate1"]["passed"] is True
    assert result["gates"]["gate2"]["passed"] is True
    assert result["gates"]["gate3"]["passed"] is True
    assert result["base"]["decoder_matrix_count"] == 216
    assert result["computation"]["total_float64_svd_calls"] == calls["count"]


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

