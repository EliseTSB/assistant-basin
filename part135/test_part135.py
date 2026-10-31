import json
import random

import numpy as np

import pairs135
import score135


def test_rotation_null_tiny_library():
    plus = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=bool)
    minus = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=bool)
    null = pairs135.rotation_null(plus, minus, rotations=20, seed=8)
    assert null.shape == (20, 2)
    assert np.all((null >= 0) & (null <= 2))
    assert np.array_equal(null, pairs135.rotation_null(plus, minus, rotations=20, seed=8))


def test_derangement_never_pairs_neuron_with_itself():
    for size in range(2, 15):
        neurons = list(range(size))
        assigned = pairs135.derangement(neurons, random.Random(size))
        assert set(assigned) == set(neurons)
        assert set(assigned.values()) == set(neurons)
        assert all(neuron != partner for neuron, partner in assigned.items())


def test_shuffled_rows_keep_plus_and_minus_texts():
    rng = random.Random(135)
    pole_text = {(4, 8, 0): "source plus", (5, 9, 1): "partner minus"}
    text_for = lambda layer, neuron, pole: pole_text[(layer, neuron, pole)]
    plus, minus = pairs135.shuffled_texts(text_for, 4, 8, 5, 9)
    row = pairs135.make_row(4, 8, 9, "persona_shuffled", plus, minus, rng, partner_layer=5)
    assert {row["a"], row["b"]} == {plus, minus}
    assert row["partner_layer"] == 5


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path, monkeypatch, gate_failure=None):
    results = tmp_path / "results"
    results.mkdir()
    expected = {f"k{i}": "same" for i in range(12)}
    monkeypatch.setattr(score135, "known_expected", lambda _path: expected)
    _write_json(results / "counts.json", {
        "both_pole_total": 120, "rotation_null_mean": 40,
        "both_pole_by_layer": [2] * 36, "persona_poles_by_layer": [8] * 36,
    })
    rows = []
    for arm in ("persona", "persona_shuffled", "fires", "fires_shuffled"):
        for i in range(4):
            rows.append({"id": f"{arm}-{i}", "arm": arm, "layer": 1, "neuron": i,
                         "partner_neuron": i + 10, "a": "plus", "b": "minus"})
    (results / "pairs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    ds = {row["id"]: "unrelated" for row in rows}
    luna = dict(ds)
    # Keep the shuffled same share at zero, satisfying gate 2.
    for i in range(12):
        ds[f"k{i}"] = "same"
        luna[f"k{i}"] = "same"
    if gate_failure == "known":
        ds.update({f"k{i}": "opposite" for i in range(2)})
    if gate_failure == "shuffled":
        ds.update({f"persona_shuffled-{i}": "same" for i in range(2)})
    if gate_failure == "kappa":
        luna.update({row["id"]: "same" for row in rows})
        monkeypatch.setattr(score135, "cohen_kappa", lambda a, b: 0.0 if a else None)
    for path, labels in ((results / "labels-ds.jsonl", ds), (results / "labels-luna.jsonl", luna)):
        path.write_text("".join(json.dumps({"id": key, "relation": val}) + "\n" for key, val in labels.items()), encoding="utf-8")
    for name in ("known-ds.jsonl", "known-luna.jsonl"):
        values = ["opposite" if gate_failure == "known" and name == "known-ds.jsonl" and i < 2 else "same" for i in range(12)]
        (results / name).write_text("".join(json.dumps({"id": f"k{i}", "relation": values[i]}) + "\n" for i in range(12)), encoding="utf-8")
    return results


def test_score_gates_pass_known_fixtures(tmp_path, monkeypatch):
    results = _fixture(tmp_path, monkeypatch)
    score135.score(results)
    report = json.loads((results / "score135.json").read_text(encoding="utf-8"))
    assert all(gate["pass"] for gate in report["gates"].values())
    assert report["predictions"]["135.3"]["true"]


def test_each_score_gate_fails_when_its_fixture_fails(tmp_path, monkeypatch):
    for gate in ("known", "shuffled", "kappa"):
        case = tmp_path / gate
        case.mkdir()
        results = _fixture(case, monkeypatch, gate_failure=gate)
        score135.score(results)
        report = json.loads((results / "score135.json").read_text(encoding="utf-8"))
        key = {"known": "known_examples", "shuffled": "shuffled_persona_same", "kappa": "second_reader_kappa"}[gate]
        assert report["gates"][key]["pass"] is False


def test_sabotaged_kappa_function_fails_reader_gate(tmp_path, monkeypatch):
    results = _fixture(tmp_path, monkeypatch, gate_failure="kappa")
    score135.score(results)
    report = json.loads((results / "score135.json").read_text(encoding="utf-8"))
    assert report["gates"]["second_reader_kappa"]["kappa"] == 0.0
    assert report["gates"]["second_reader_kappa"]["pass"] is False
