import json
import random

import numpy as np

import label135
import pairs135
import score135


def test_rotation_null_tiny_library():
    plus = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=bool)
    minus = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=bool)
    null = pairs135.rotation_null(plus, minus, rotations=20, seed=8)
    assert null.shape == (20, 2)
    assert np.all((null >= 0) & (null <= 2))
    assert np.array_equal(null, pairs135.rotation_null(plus, minus, rotations=20, seed=8))


def test_rotation_null_rolls_a_single_flag():
    plus = np.array([[1, 0, 0, 0]], dtype=bool)
    minus = np.array([[1, 0, 0, 0]], dtype=bool)
    mean = pairs135.rotation_null(plus, minus, rotations=100, seed=8).mean()
    assert 0 < mean < 0.5


def test_first_sentence_abbreviations_quotes_and_word_boundary():
    assert pairs135.first_sentence('Use e.g. this phrase. Then stop.') == 'Use e.g. this phrase.'
    assert pairs135.first_sentence('He said "Go now." Then left.') == 'He said "Go now."'
    value = 'word ' * 39 + 'unfinishedwordtail'
    result = pairs135.first_sentence(value)
    assert len(result) <= 200
    assert not result.endswith('unfinishedwordtail')
    assert result == ('word ' * 39).rstrip()


def test_row_id_changes_with_content_and_partner_layer():
    one = pairs135.make_row(2, 4, 8, 'persona_shuffled', 'alpha', 'beta', random.Random(1), partner_layer=2)
    changed_text = pairs135.make_row(2, 4, 8, 'persona_shuffled', 'alpha', 'gamma', random.Random(1), partner_layer=2)
    changed_layer = pairs135.make_row(2, 4, 8, 'persona_shuffled', 'alpha', 'beta', random.Random(1), partner_layer=3)
    assert one['id'] != changed_text['id']
    assert one['id'] != changed_layer['id']


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


def test_parse_and_known_mode_resume(tmp_path, monkeypatch):
    assert label135.parse('prefix {"relation": "opposite"} suffix') == 'opposite'
    known = tmp_path / 'known.json'
    _write_json(known, [
        {"a": "one", "b": "two", "relation": "same"},
        {"a": "up", "b": "down", "relation": "opposite"},
    ])
    items = label135.known_items(known)
    out = tmp_path / 'known-ds.jsonl'
    (out.parent / 'known-ds.jsonl').write_text(
        json.dumps({"id": items[0]["id"], "relation": "same"}) + '\n', encoding='utf-8')
    calls = []
    monkeypatch.setattr(label135, 'one', lambda item, reader: calls.append(item['id']) or
                        {"id": item['id'], "arm": item['arm'], "relation": "opposite"})
    label135.main('', out, 'ds', known=str(known))
    assert calls == [items[1]["id"]]
    assert len(out.read_text(encoding='utf-8').splitlines()) == 2


def test_study_mode_requires_both_known_gate_files(tmp_path):
    with np.testing.assert_raises_regex(SystemExit, 'known-ds.jsonl is missing'):
        label135.require_study_gates(tmp_path / 'labels-ds.jsonl')


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


def test_cohen_kappa_hand_computed_imperfect_agreement():
    # Observed = 2/3; expected = (2*1 + 1*2)/9 = 4/9; kappa = 2/5.
    assert np.isclose(score135.cohen_kappa(['a', 'a', 'b'], ['a', 'b', 'b']), 0.4)


def _prediction_case(overrides):
    arms = ('persona', 'persona_shuffled', 'fires', 'fires_shuffled')
    rows, ds = [], {}
    defaults = {
        'persona': ['same', 'same', 'unrelated', 'unrelated'],
        'persona_shuffled': ['same', 'unrelated', 'unrelated', 'unrelated'],
        'fires': ['same', 'same', 'unrelated', 'unrelated'],
        'fires_shuffled': ['same', 'unrelated', 'unrelated', 'unrelated'],
    }
    defaults.update(overrides)
    for arm in arms:
        for i, relation in enumerate(defaults[arm]):
            row_id = f'{arm}-{i}'
            rows.append({'id': row_id, 'arm': arm, 'layer': 1, 'neuron': i})
            ds[row_id] = relation
    counts = {'both_pole_total': 120, 'rotation_null_mean': 40,
              'both_pole_by_layer': [1] * 36,
              'persona_poles_by_layer': [8] * 18 + [4] * 18}
    return counts, rows, ds


def test_predictions_135_1_and_135_2_true_and_false():
    counts, rows, ds = _prediction_case({})
    assert score135.compute_predictions(counts, rows, ds, {})['135.1']['true'] is True
    assert score135.compute_predictions({**counts, 'both_pole_total': 79}, rows, ds, {})['135.1']['true'] is False
    assert score135.compute_predictions(counts, rows, ds, {})['135.2']['true'] is True
    assert score135.compute_predictions({**counts, 'both_pole_by_layer': [2] * 18 + [0] * 18}, rows, ds, {})['135.2']['true'] is False


def test_predictions_135_4_to_135_7_true_and_false():
    cases = {
        '135.4': ({'persona': ['same', 'same', 'unrelated', 'unrelated'],
                   'persona_shuffled': ['same', 'unrelated', 'unrelated', 'unrelated']},
                  {'persona': ['same', 'unrelated', 'unrelated', 'unrelated'],
                   'persona_shuffled': ['same', 'same', 'unrelated', 'unrelated']}),
        '135.5': ({'persona': ['opposite', 'unrelated', 'unrelated', 'unrelated'],
                   'persona_shuffled': ['opposite', 'unrelated', 'unrelated', 'unrelated']},
                  {'persona': ['opposite', 'opposite', 'opposite', 'unrelated'],
                   'persona_shuffled': ['opposite', 'unrelated', 'unrelated', 'unrelated']}),
        '135.6': ({'fires': ['same', 'unrelated', 'unrelated', 'unrelated'],
                   'fires_shuffled': ['same', 'unrelated', 'unrelated', 'unrelated']},
                  {'fires': ['same', 'same', 'same', 'unrelated'],
                   'fires_shuffled': ['same', 'unrelated', 'unrelated', 'unrelated']}),
        '135.7': ({'persona': ['same', 'same', 'opposite', 'opposite'],
                   'fires': ['unrelated', 'unrelated', 'same', 'same']},
                  {'persona': ['same', 'same', 'opposite', 'opposite'],
                   'fires': ['same', 'same', 'opposite', 'opposite']}),
    }
    for key, (true_values, false_values) in cases.items():
        counts, rows, ds = _prediction_case(true_values)
        assert score135.compute_predictions(counts, rows, ds, {})[key]['true'] is True
        counts, rows, ds = _prediction_case(false_values)
        assert score135.compute_predictions(counts, rows, ds, {})[key]['true'] is False
