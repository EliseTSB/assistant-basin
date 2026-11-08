"""Synthetic tests for Part 197's gates, residualization, scoring, and CLI."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from part196 import analyze196 as p196  # noqa: E402
from part196.select196 import STYLES, text_key  # noqa: E402
from part197 import analyze197 as a197  # noqa: E402
from part197 import digest197  # noqa: E402
from part197.digest197 import compute_records_digest, write_records_digest  # noqa: E402


def _registered_for_rows(rows):
    return {
        "manifest_sha256": p196.manifest_sha(rows),
        "model_snapshot_revision": p196.MODEL_REVISION,
        "weight_sha256": "a" * 64,
    }


def _record_values(clip_id: str, seed: int):
    rng = np.random.default_rng(seed)
    return {
        "id": np.asarray(clip_id),
        "status": np.asarray("ok"),
        "act_mean": rng.normal(size=32).astype(np.float16),
        "emb": rng.normal(size=8).astype(np.float32),
        "acoustic": rng.normal(size=7).astype(np.float32),
        "duration_seconds": np.asarray(3.25, dtype=np.float32),
        "audio_sha256": np.asarray(hashlib.sha256(clip_id.encode()).hexdigest()),
    }


def test_digest_is_sorted_and_binds_audio_and_all_feature_bytes(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id\nclip_b\nclip_a\n", encoding="utf-8")
    work = tmp_path / "work"
    (work / "records").mkdir(parents=True)
    for index, clip_id in enumerate(("clip_a", "clip_b")):
        np.savez_compressed(work / "records" / f"{clip_id}.npz", **_record_values(clip_id, index))
    (work / "durations.json").write_text('{"clip_a": {"status": "included"}}', encoding="utf-8")
    first = compute_records_digest(work, manifest)
    assert [row["id"] for row in first["records"]] == ["clip_a", "clip_b"]
    assert set(first["records"][0]) == {
        "id", "audio_sha256", "act_mean_sha256", "emb_sha256", "acoustic_sha256",
        "duration_seconds_sha256",
    }
    out = tmp_path / "records_digest.json"
    assert write_records_digest(work, manifest, out) == first
    assert json.loads(out.read_text()) == first
    values = _record_values("clip_a", 0)
    values["emb"] = values["emb"].copy()
    values["emb"][0] += 1.0
    np.savez_compressed(work / "records" / "clip_a.npz", **values)
    changed = compute_records_digest(work, manifest)
    assert changed["records_sha256"] != first["records_sha256"]
    cli_out = tmp_path / "cli_digest.json"
    assert digest197.main(["--manifest", str(manifest), "--work-dir", str(work),
                           "--out", str(cli_out)]) == 0
    assert json.loads(cli_out.read_text()) == changed


def test_gate2_linear_only_signal_is_zeroed_and_scores_at_chance():
    rng = np.random.default_rng(19701)
    labels = np.repeat(np.arange(7), 12)
    nuisance = rng.normal(size=(len(labels), 8))
    nuisance_test = rng.normal(size=(len(labels), 8))
    beta = rng.normal(size=(8, 20))
    x_train = nuisance @ beta + 2.5
    x_test = nuisance_test @ beta + 2.5
    model_map, residual_train, residual_test, zeroed = a197.residualize_train_test(
        nuisance, nuisance_test, x_train, x_test)
    gate = a197.gate2_check(a197._standardized_nuisance(model_map, nuisance),
                            residual_train, model_map.rank)
    assert model_map.rank == 9
    assert gate["verdict"] == "passed"
    assert zeroed.all()
    assert np.count_nonzero(residual_train) == 0
    assert np.count_nonzero(residual_test) == 0
    probe = p196.fit_probe(residual_train, labels)
    prediction = p196._score_probe(probe, residual_test).argmax(axis=1)
    assert p196.balanced_accuracy(labels, prediction) == pytest.approx(1 / 7)


def test_gate2_keeps_an_independent_residual_signal_readable():
    rng = np.random.default_rng(19702)
    labels = np.repeat(np.arange(7), 14)
    test_labels = np.repeat(np.arange(7), 6)
    nuisance = rng.normal(size=(len(labels), 8))
    nuisance_test = rng.normal(size=(len(test_labels), 8))
    x_train = 10.0 * np.eye(7)[labels] + rng.normal(scale=0.02, size=(len(labels), 7))
    x_test = 10.0 * np.eye(7)[test_labels] + rng.normal(scale=0.02, size=(len(test_labels), 7))
    residual_map, train_residual, test_residual, zeroed = a197.residualize_train_test(
        nuisance, nuisance_test, x_train, x_test)
    gate = a197.gate2_check(a197._standardized_nuisance(residual_map, nuisance),
                            train_residual, residual_map.rank)
    assert gate["verdict"] == "passed"
    assert not zeroed.any()
    probe = p196.fit_probe(train_residual, labels)
    prediction = p196._score_probe(probe, test_residual).argmax(axis=1)
    assert p196.balanced_accuracy(test_labels, prediction) > 0.95


def test_gate2_rejects_a_map_fitted_on_row_shuffled_nuisance():
    rng = np.random.default_rng(19703)
    nuisance = rng.normal(size=(120, 8))
    x_train = nuisance @ rng.normal(size=(8, 12)) + rng.normal(scale=0.1, size=(120, 12))
    shuffled_map = a197.fit_residualization(nuisance[rng.permutation(len(nuisance))], x_train)
    bad_residual = a197.transform_residuals(shuffled_map, nuisance, x_train)
    bad_residual, _, _ = a197.apply_near_zero_rule(x_train, bad_residual, bad_residual)
    gate = a197.gate2_check(a197._standardized_nuisance(
        a197.fit_residualization(nuisance, x_train), nuisance), bad_residual, shuffled_map.rank)
    assert gate["verdict"] == "failed"
    assert gate["max_normalized_cross_product"] > 1e-8


def test_gate2_requires_the_registered_rank_of_nine():
    gate = a197.gate2_check(np.zeros((12, 8)), np.zeros((12, 3)), rank=8)
    assert gate["rank_passed"] is False
    assert gate["verdict"] == "failed"


def test_near_zero_variance_rule_zeros_train_and_test_before_scaling():
    rng = np.random.default_rng(19704)
    x_train = rng.normal(size=(30, 2))
    residual_train = np.column_stack((np.ones(30) * 1e-10,
                                      rng.normal(scale=2e-6, size=30)))
    residual_test = np.ones((5, 2)) * 7.0
    clean_train, clean_test, zeroed = a197.apply_near_zero_rule(
        x_train, residual_train, residual_test)
    assert zeroed.tolist() == [True, False]
    assert np.all(clean_train[:, 0] == 0.0)
    assert np.all(clean_test[:, 0] == 0.0)
    assert np.any(clean_test[:, 1] != 0.0)


def test_residualization_map_is_fitted_from_training_rows_only():
    rng = np.random.default_rng(19705)
    nuisance_train = rng.normal(size=(80, 8))
    x_train = rng.normal(size=(80, 20))
    nuisance_test = rng.normal(size=(12, 8))
    x_test = rng.normal(size=(12, 20))
    map_a, *_ = a197.residualize_train_test(nuisance_train, nuisance_test, x_train, x_test)
    map_b, *_ = a197.residualize_train_test(
        nuisance_train, nuisance_test * 1e6 + 5e5, x_train, x_test * 1e6 - 2e5)
    np.testing.assert_array_equal(map_a.nuisance_mean, map_b.nuisance_mean)
    np.testing.assert_array_equal(map_a.nuisance_scale, map_b.nuisance_scale)
    np.testing.assert_array_equal(map_a.coefficients, map_b.coefficients)


def test_gate1_identity_refuses_wrong_registered_identity(tmp_path, monkeypatch):
    rows = [{"id": "x", "speaker_id": "ex01", "style": "happy", "text_key": "t"}]
    registered = _registered_for_rows(rows)
    monkeypatch.setattr(a197, "REGISTERED_IDENTITY", registered)
    work = tmp_path / "work"
    work.mkdir()
    actual = dict(registered)
    actual["weight_sha256"] = "b" * 64
    (work / "identity.json").write_text(json.dumps(actual), encoding="utf-8")
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps(registered), encoding="utf-8")
    check = a197.check_identity(work, rows, expected_path)
    assert check["verdict"] == "failed"
    assert "weight_sha256" in check["mismatches"]


def test_gate1_wrong_identity_stops_before_digest_or_regeneration(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id,speaker_id,style,text_key,fold\nx,ex01,happy,t,ex01\n", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    (work / "identity.json").write_text(json.dumps({
        "manifest_sha256": "0" * 64,
        "model_snapshot_revision": p196.MODEL_REVISION,
        "weight_sha256": "a" * 64,
    }), encoding="utf-8")
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps(a197.REGISTERED_IDENTITY), encoding="utf-8")
    monkeypatch.setattr(a197, "check_digest", lambda *args, **kwargs: pytest.fail("digest ran after identity refusal"))
    monkeypatch.setattr(a197, "gate1_regeneration",
                        lambda *args, **kwargs: pytest.fail("regeneration ran after identity refusal"))
    out_dir = tmp_path / "out"
    code = a197.main(["--manifest", str(manifest), "--work-dir", str(work),
                      "--out-dir", str(out_dir), "--expected-identity", str(expected),
                      "--records-digest", str(tmp_path / "absent.json"),
                      "--part196-results", str(tmp_path / "absent-results.json")])
    assert code == 2
    result = json.loads((out_dir / "results.json").read_text())
    assert result["gate1"]["verdict"] == "failed"
    assert result["gate2"]["verdict"] == "not_run"
    assert result["readouts"] == {}
    assert json.loads(capsys.readouterr().out) == {"gate1": "failed", "gate2": "not_run"}


def test_gate1_digest_refuses_changed_records_digest(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id\nclip_a\n", encoding="utf-8")
    work = tmp_path / "work"
    (work / "records").mkdir(parents=True)
    np.savez_compressed(work / "records" / "clip_a.npz", **_record_values("clip_a", 9))
    (work / "durations.json").write_text('{"clip_a": {"status": "included"}}', encoding="utf-8")
    expected_path = tmp_path / "expected_digest.json"
    expected = compute_records_digest(work, manifest)
    expected["records"][0]["audio_sha256"] = "0" * 64
    expected_path.write_text(json.dumps(expected), encoding="utf-8")
    check = a197.check_digest(work, manifest, expected_path)
    assert check["verdict"] == "failed"


@pytest.mark.parametrize("mutation", ["numeric", "discrete"])
def test_gate1_regenerated_results_refuse_numeric_and_discrete_differences(
        mutation, tmp_path, monkeypatch, capsys):
    expected = {"folds": {"ex01": {"E": {"selected_layer": 0}}},
                "readouts": {"E0": {"mean_balanced_accuracy": 0.5}}}
    actual = json.loads(json.dumps(expected))
    if mutation == "numeric":
        actual["readouts"]["E0"]["mean_balanced_accuracy"] += 1e-6
    else:
        actual["folds"]["ex01"]["E"]["selected_layer"] = 1
    expected_path = tmp_path / "part196_results.json"
    expected_path.write_text(json.dumps(expected), encoding="utf-8")

    def synthetic_analyze(*, manifest_path, work_dir, out_dir):
        (out_dir / "results.json").write_text(json.dumps(actual), encoding="utf-8")

    monkeypatch.setattr(p196, "analyze", synthetic_analyze)
    check = a197.gate1_regeneration(manifest_path=tmp_path / "manifest.csv",
                                    work_dir=tmp_path / "work",
                                    part196_results_path=expected_path)
    assert check["comparison"]["passed"] is False
    assert check["verdict"] == "failed"
    assert check["comparison"]["mismatch_count"] == 1
    assert check["comparison"]["mismatches"][0]["kind"] == (
        "numeric" if mutation == "numeric" else "discrete")
    stderr = json.loads(capsys.readouterr().err)
    assert "largest_numeric_difference" in stderr and "path" in stderr


def _score_fixture():
    readouts = {name: {"mean_balanced_accuracy": 0.2} for name in ("N", "E0", "EN", "K", "R")}
    readouts["R"]["mean_balanced_accuracy"] = 0.45
    for layer in range(24):
        readouts[f"L_{layer}"] = {"mean_balanced_accuracy": 0.30}
    readouts["L_0"]["mean_balanced_accuracy"] = 0.40
    for layer in range(20, 24):
        readouts[f"L_{layer}"]["mean_balanced_accuracy"] = 0.35
    pairs = {
        "EN_minus_N": {"mean_balanced_accuracy_difference": 0.05,
                       "text_cluster_95": {"low": 0.001, "high": 0.08}},
        "EN_minus_E0": {"mean_balanced_accuracy_difference": 0.03,
                        "text_cluster_95": {"low": -0.01, "high": 0.08}},
        "R_minus_N": {"mean_balanced_accuracy_difference": 0.1,
                      "text_cluster_95": {"low": 0.02, "high": 0.18}},
        "E0_minus_K": {"mean_balanced_accuracy_difference": 0.0,
                       "text_cluster_95": {"low": -0.1, "high": 0.1}},
    }
    speakers = {"S_0": {"balanced_accuracy": 0.80}}
    return readouts, pairs, speakers


def test_prediction_threshold_boundaries_for_197_1_through_197_6():
    readouts, pairs, speakers = _score_fixture()
    scored = a197.score_predictions(readouts, pairs, speakers, gate2_passed=True)
    assert [scored[f"197.{i}"]["verdict"] for i in range(1, 7)] == ["held"] * 6

    pairs["EN_minus_N"]["text_cluster_95"]["low"] = 0.0
    assert a197.score_predictions(readouts, pairs, speakers, True)["197.1"]["verdict"] == "inconclusive"
    pairs["EN_minus_N"]["text_cluster_95"]["low"] = 0.001
    pairs["EN_minus_N"]["mean_balanced_accuracy_difference"] = 0.049999
    assert a197.score_predictions(readouts, pairs, speakers, True)["197.1"]["verdict"] == "not_held"
    pairs["EN_minus_N"]["mean_balanced_accuracy_difference"] = 0.05

    readouts["R"]["mean_balanced_accuracy"] = 0.449999
    assert a197.score_predictions(readouts, pairs, speakers, True)["197.2"]["verdict"] == "not_held"
    assert a197.score_predictions(readouts, pairs, speakers, False)["197.2"]["verdict"] == "not_scorable"
    readouts["R"]["mean_balanced_accuracy"] = 0.45

    pairs["EN_minus_E0"]["mean_balanced_accuracy_difference"] = 0.030001
    assert a197.score_predictions(readouts, pairs, speakers, True)["197.3"]["verdict"] == "not_held"
    pairs["EN_minus_E0"]["mean_balanced_accuracy_difference"] = 0.03
    readouts["K"]["mean_balanced_accuracy"] = 0.300001
    assert a197.score_predictions(readouts, pairs, speakers, True)["197.4"]["verdict"] == "not_held"
    readouts["K"]["mean_balanced_accuracy"] = 0.30

    readouts["L_20"]["mean_balanced_accuracy"] = 0.350001
    assert a197.score_predictions(readouts, pairs, speakers, True)["197.5"]["verdict"] == "not_held"
    readouts["L_20"]["mean_balanced_accuracy"] = 0.35
    readouts["L_3"]["mean_balanced_accuracy"] = 0.41
    assert a197.score_predictions(readouts, pairs, speakers, True)["197.5"]["verdict"] == "not_held"
    readouts["L_3"]["mean_balanced_accuracy"] = 0.30
    speakers["S_0"]["balanced_accuracy"] = 0.799999
    assert a197.score_predictions(readouts, pairs, speakers, True)["197.6"]["verdict"] == "not_held"


class _DummyProbe:
    def __init__(self, labels):
        self.classes_ = np.sort(np.unique(labels))

    def __getitem__(self, _index):
        return self

    def decision_function(self, x):
        return np.zeros((len(x), len(self.classes_)), dtype=np.float64)

    def predict(self, x):
        return np.full(len(x), self.classes_[0], dtype=np.int64)


def _build_part196_synthetic_work(tmp_path: Path):
    rows = []
    rng = np.random.default_rng(19706)
    work = tmp_path / "work"
    record_dir = work / "records"
    record_dir.mkdir(parents=True)
    duration_data = {}
    text_embeddings = {}
    for speaker_index in range(4):
        speaker = f"ex{speaker_index + 1:02d}"
        for text_index in range(2):
            text_hash = text_key(f"{speaker} synthetic sentence {text_index}")
            text_embeddings[text_hash] = rng.normal(size=768).astype(np.float32)
            for style_index, style in enumerate(STYLES):
                clip_id = f"{speaker}_{style}_{text_index}"
                rows.append({"id": clip_id, "speaker_id": speaker, "style": style,
                             "text_key": text_hash, "fold": speaker, "shard": "synthetic.parquet",
                             "row_index": str(len(rows)), "duration_seconds": "3.0",
                             "duration_status": "in_range_wav_header"})
                activation = np.zeros(24 * 2048, dtype=np.float16)
                activation[style_index] = 5.0
                embedding = np.zeros(768, dtype=np.float32)
                embedding[speaker_index] = 5.0
                acoustic = rng.normal(size=7).astype(np.float32)
                duration = np.asarray(2.5 + rng.random(), dtype=np.float32)
                duration_data[clip_id] = {"status": "included", "duration_seconds": float(duration)}
                record = {"id": np.asarray(clip_id), "act_mean": activation,
                          "emb": embedding, "acoustic": acoustic,
                          "duration_seconds": duration, "audio_sha256": np.asarray(
                              hashlib.sha256(clip_id.encode()).hexdigest()),
                          "status": np.asarray("ok")}
                np.savez_compressed(record_dir / f"{clip_id}.npz", **record)

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    identity = {
        "model_id": p196.MODEL_ID,
        "model_snapshot_revision": p196.MODEL_REVISION,
        "weight_sha256": "a" * 64,
        "transformers_version": "synthetic",
        "sentence_transformers_version": "synthetic",
        "torch_version": "synthetic",
        "manifest_sha256": p196.manifest_sha(rows),
    }
    work.mkdir(exist_ok=True)
    (work / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
    (work / "durations.json").write_text(json.dumps(duration_data), encoding="utf-8")
    (work / "gate2.json").write_text(json.dumps({"verdict": "passed", "clips": []}), encoding="utf-8")
    np.savez_compressed(work / "prompts196.npz", styles=np.asarray(STYLES),
                        embeddings=np.zeros((7, 8, 768), dtype=np.float32))
    np.savez_compressed(work / "transcripts196.npz",
                        text_keys=np.asarray(list(text_embeddings), dtype="U64"),
                        embeddings=np.stack(list(text_embeddings.values())))
    return rows, manifest, work


def test_analyze197_end_to_end_on_small_part196_synthetic_folder(tmp_path, monkeypatch, capsys):
    rows, manifest, work = _build_part196_synthetic_work(tmp_path)
    registered = _registered_for_rows(rows)
    monkeypatch.setattr(a197, "REGISTERED_IDENTITY", registered)
    expected_identity = tmp_path / "expected_identity.json"
    expected_identity.write_text(json.dumps(registered), encoding="utf-8")
    records_digest = tmp_path / "records_digest.json"
    write_records_digest(work, manifest, records_digest)

    original_gate = p196.gate1_check_recorded
    monkeypatch.setattr(p196, "gate1_check_recorded",
                        lambda manifest_rows, ids: original_gate(manifest_rows, ids, min_per_cell=1))
    monkeypatch.setattr(p196, "selected_style_profile", lambda *args, **kwargs: {})
    monkeypatch.setattr(p196, "choose_intermediate_layer",
                        lambda *args, **kwargs: (0, [0.2] * 24))
    monkeypatch.setattr(p196, "select_unit_per_style", lambda *args, **kwargs: [
        {"style": style, "index": index, "layer": 0, "neuron": index, "sign": 1,
         "train_auc_raw": 0.5, "train_auc_oriented": 0.5}
        for index, style in enumerate(STYLES)])
    monkeypatch.setattr(p196, "fit_probe", lambda _x, y: _DummyProbe(y))
    monkeypatch.setattr(p196, "BOOTSTRAPS", 2)
    monkeypatch.setattr(a197, "BOOTSTRAPS", 4)

    # Generate the synthetic reference with Part 196's own analyze() on this same work folder.
    p196_results = tmp_path / "part196_reference" / "results.json"
    p196.analyze(manifest_path=manifest, work_dir=work, out_dir=p196_results.parent)
    assert p196_results.is_file()

    out_dir = tmp_path / "part197_out"
    code = a197.main([
        "--manifest", str(manifest), "--work-dir", str(work), "--out-dir", str(out_dir),
        "--expected-identity", str(expected_identity), "--records-digest", str(records_digest),
        "--part196-results", str(p196_results),
    ])
    output = capsys.readouterr()
    summary = json.loads(output.out)
    assert code == 0
    assert set(summary) == {"gate1", "gate2"}
    assert summary == {"gate1": "passed", "gate2": "passed"}
    result = json.loads((out_dir / "results.json").read_text())
    assert result["status"] == "complete"
    assert set(("N", "E0", "EN", "R", "K", "H", *[f"L_{k}" for k in range(24)])) <= set(result["readouts"])
    assert set(result["paired_differences"]) == {
        "EN_minus_N", "R_minus_N", "EN_minus_E0", "E0_minus_K"}
    assert set(result["shuffled_label_controls"]) == {"EN", "R"}
    assert all(result["shuffled_label_controls"][key]["shuffled_within_each_training_speaker"]
               for key in ("EN", "R"))
    assert all(value["chance"] == pytest.approx(1 / 7)
               for value in result["shuffled_label_controls"].values())
    assert set(result["speaker_identity_by_layer"]) == {
        *[f"S_{k}" for k in range(24)], "S_final_embedding"}
    assert set(result["predictions"]) == {f"197.{k}" for k in range(1, 7)}
    assert result["gate2"]["verdict"] == "passed"



# Design r3 (Alden #5, #7), added by the owner after the build: the worker built to r2.
def test_mask_catches_units_constant_in_x_at_zero_and_at_a_nonzero_value():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 3))
    x[:, 0] = 0.0
    x[:, 1] = 5.0
    residual_train = rng.normal(size=(50, 3))
    residual_train[:, 0] = 0.0
    residual_train[:, 1] = rng.normal(scale=1e-14, size=50)   # cancellation noise on a constant unit
    residual_test = rng.normal(size=(20, 3))
    train, test, zeroed = a197.apply_near_zero_rule(x, residual_train, residual_test)
    assert zeroed.tolist() == [True, True, False]
    assert not train[:, :2].any() and not test[:, :2].any()
    assert np.array_equal(test[:, 2], residual_test[:, 2])


def test_constant_nuisance_column_blocks_r_through_the_rank_policy():
    rng = np.random.default_rng(2)
    nuisance = rng.normal(size=(60, 8))
    nuisance[:, 3] = 7.0
    x = rng.normal(size=(60, 5))
    residual_map = a197.fit_residualization(nuisance, x)
    residual = a197.transform_residuals(residual_map, nuisance, x)
    gate = a197.gate2_check(a197._standardized_nuisance(residual_map, nuisance), residual, residual_map.rank)
    assert residual_map.rank == 8 and gate["verdict"] == "failed" and not gate["rank_passed"]


def test_one_blocked_fold_fails_gate2_for_every_fold(monkeypatch):
    rng = np.random.default_rng(3)
    monkeypatch.setattr(a197, "WIDTH", 6)
    nuisance = rng.normal(size=(80, 8))
    activations = rng.normal(size=(80, 12)).astype(np.float32)
    folds = {"s1": (np.arange(40, 80), np.arange(0, 40)), "s2": (np.arange(0, 40), np.arange(40, 80))}
    nuisance[0:40, 2] = 1.0            # constant in s2's training rows only
    gate, _ = a197._prepare_gate2(folds, nuisance, activations)
    assert gate["folds"]["s1"]["verdict"] == "passed"
    assert gate["folds"]["s2"]["verdict"] == "failed"
    assert gate["verdict"] == "failed"   # R is then not read anywhere, never averaged over s1 alone


def test_gate1_only_stops_before_any_readout_is_read(tmp_path, monkeypatch):
    manifest = tmp_path / "m.csv"
    manifest.write_text("id,speaker_id,style,text_key,fold,shard,row_index,duration_seconds,duration_status\n",
                        encoding="utf-8")
    passed = {"verdict": "passed"}
    monkeypatch.setattr(a197, "check_identity", lambda *a, **k: dict(passed))
    monkeypatch.setattr(a197, "check_digest", lambda *a, **k: dict(passed))
    monkeypatch.setattr(a197, "gate1_regeneration", lambda **k: dict(passed, max_numeric_difference=0.0))

    def forbidden(*_a, **_k):
        raise AssertionError("records were read in gate1-only mode")

    monkeypatch.setattr(p196, "_read_records", forbidden)
    out = tmp_path / "out"
    result = a197.analyze(manifest_path=manifest, work_dir=tmp_path, out_dir=out,
                          expected_identity_path=tmp_path / "i.json", records_digest_path=tmp_path / "d.json",
                          part196_results_path=tmp_path / "r.json", gate1_only=True)
    saved = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert result["status"] == saved["status"] == "gate1_only"
    assert "readouts" not in saved and "predictions" not in saved



# Review F1 to F4 (DeepSeek code review, 2026-11-08 07:22), added by the owner.
def test_digest_binds_durations_json_and_refuses_without_it(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id\nclip_a\n", encoding="utf-8")
    work = tmp_path / "work"
    (work / "records").mkdir(parents=True)
    np.savez_compressed(work / "records" / "clip_a.npz", **_record_values("clip_a", 4))
    with pytest.raises(FileNotFoundError):
        compute_records_digest(work, manifest)
    (work / "durations.json").write_text('{"clip_a": {"status": "included"}}', encoding="utf-8")
    first = compute_records_digest(work, manifest)
    (work / "durations.json").write_text('{"clip_a": {"status": "excluded"}}', encoding="utf-8")
    assert compute_records_digest(work, manifest)["records_sha256"] != first["records_sha256"]


def test_active_set_refuses_a_missing_or_empty_durations_file(tmp_path):
    rows = [{"id": "a"}]
    status = np.asarray(["ok"])
    duration = np.asarray([3.0])
    with pytest.raises(FileNotFoundError):
        a197._active_indices(rows, status, duration, tmp_path)
    (tmp_path / "durations.json").write_text('{"a": {"status": "excluded"}}', encoding="utf-8")
    with pytest.raises(ValueError):
        a197._active_indices(rows, status, duration, tmp_path)


def test_197_1_missing_interval_is_not_computable_not_inconclusive():
    readouts = {name: {"mean_balanced_accuracy": 0.5} for name in ("N", "E0", "EN", "K", "R")}
    readouts.update({f"L_{i}": {"mean_balanced_accuracy": 0.3} for i in range(24)})
    pairs = {"EN_minus_N": {"mean_balanced_accuracy_difference": 0.1,
                            "text_cluster_95": {"low": None, "high": None}},
             "EN_minus_E0": {"mean_balanced_accuracy_difference": 0.0, "text_cluster_95": {}}}
    speakers = {"layers": {"L_0": {"balanced_accuracy": 0.9}}}
    scored = a197.score_predictions(readouts, pairs, speakers, True)
    assert scored["197.1"]["verdict"] == "not_computable"


def test_end_to_end_with_the_real_probe_reads_a_planted_layer0_signal(tmp_path, monkeypatch):
    # Review F4: the dummy-probe end-to-end test cannot fail on a wrong readout. Style is planted in layer 0 only,
    # speaker in layer 5 and the embedding; acoustics and duration are noise. Adapted from the reviewer's probes.
    import csv as _csv
    rng = np.random.default_rng(7)
    rows, durations, text_embeddings = [], {}, {}
    work = tmp_path / "work"
    (work / "records").mkdir(parents=True)
    for si in range(4):
        speaker = f"ex{si + 1:02d}"
        for ti in range(3):
            key = text_key(f"{speaker} sentence {ti}")
            text_embeddings[key] = rng.normal(size=768).astype(np.float32)
            for st, style in enumerate(STYLES):
                clip_id = f"{speaker}_{style}_{ti}"
                rows.append({"id": clip_id, "speaker_id": speaker, "style": style, "text_key": key,
                             "fold": speaker, "shard": "s", "row_index": str(len(rows)),
                             "duration_seconds": "3.0", "duration_status": "in_range_wav_header"})
                act = np.zeros(24 * 2048, dtype=np.float16)
                act[:2048] = rng.normal(scale=0.3, size=2048)
                act[st * 10:(st + 1) * 10] += 3.0
                act[5 * 2048:6 * 2048] = rng.normal(scale=0.3, size=2048)
                act[5 * 2048 + si * 10:5 * 2048 + (si + 1) * 10] += 3.0
                emb = np.zeros(768, np.float32)
                emb[si * 10:(si + 1) * 10] += 3.0
                duration = np.float32(2.5 + rng.random())
                durations[clip_id] = {"status": "included", "duration_seconds": float(duration)}
                np.savez_compressed(work / "records" / f"{clip_id}.npz", id=np.asarray(clip_id), act_mean=act,
                                    emb=emb, acoustic=rng.normal(size=7).astype(np.float32),
                                    duration_seconds=duration, status=np.asarray("ok"),
                                    audio_sha256=np.asarray(hashlib.sha256(clip_id.encode()).hexdigest()))
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = _csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    identity = {"model_id": p196.MODEL_ID, "model_snapshot_revision": p196.MODEL_REVISION,
                "weight_sha256": "a" * 64, "manifest_sha256": p196.manifest_sha(rows),
                "transformers_version": "x", "sentence_transformers_version": "x", "torch_version": "x"}
    (work / "identity.json").write_text(json.dumps(identity))
    (work / "durations.json").write_text(json.dumps(durations))
    (work / "gate2.json").write_text(json.dumps({"verdict": "passed", "clips": []}))
    np.savez_compressed(work / "prompts196.npz", styles=np.asarray(STYLES),
                        embeddings=np.zeros((7, 8, 768), np.float32))
    np.savez_compressed(work / "transcripts196.npz", text_keys=np.asarray(list(text_embeddings), dtype="U64"),
                        embeddings=np.stack(list(text_embeddings.values())))
    registered = {"manifest_sha256": p196.manifest_sha(rows), "model_snapshot_revision": p196.MODEL_REVISION,
                  "weight_sha256": "a" * 64}
    monkeypatch.setattr(a197, "REGISTERED_IDENTITY", registered)
    expected_identity = tmp_path / "expected.json"
    expected_identity.write_text(json.dumps(registered))
    digest = tmp_path / "digest.json"
    write_records_digest(work, manifest, digest)
    original_gate1 = p196.gate1_check_recorded
    monkeypatch.setattr(p196, "gate1_check_recorded", lambda r, ids: original_gate1(r, ids, min_per_cell=1))
    monkeypatch.setattr(p196, "selected_style_profile", lambda *a, **k: {})
    monkeypatch.setattr(p196, "choose_intermediate_layer", lambda *a, **k: (0, [0.2] * 24))
    monkeypatch.setattr(p196, "select_unit_per_style", lambda *a, **k: [
        {"style": s, "index": i, "layer": 0, "neuron": i, "sign": 1, "train_auc_raw": 0.5,
         "train_auc_oriented": 0.5} for i, s in enumerate(p196.STYLES)])
    monkeypatch.setattr(p196, "BOOTSTRAPS", 2)
    monkeypatch.setattr(a197, "BOOTSTRAPS", 50)
    reference = tmp_path / "p196ref" / "results.json"
    p196.analyze(manifest_path=manifest, work_dir=work, out_dir=reference.parent)
    out = tmp_path / "out"
    code = a197.main(["--manifest", str(manifest), "--work-dir", str(work), "--out-dir", str(out),
                      "--expected-identity", str(expected_identity), "--records-digest", str(digest),
                      "--part196-results", str(reference)])
    assert code == 0
    saved = json.loads((out / "results.json").read_text(encoding="utf-8"))
    r = {name: value.get("mean_balanced_accuracy") for name, value in saved["readouts"].items()
         if isinstance(value, dict)}
    assert r["E0"] > 0.9 and r["EN"] > 0.9 and r["R"] > 0.9 and r["L_0"] > 0.9
    assert r["N"] < 0.4 and r["K"] < 0.4 and r["H"] < 0.4 and r["L_5"] < 0.4
    assert saved["predictions"]["197.5"]["best_layer"] == 0
