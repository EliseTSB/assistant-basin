"""Fixture-only checks for the registered Part 188 math and manifest."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from part188.analyze188 import (auc_per_unit, auc_score, bootstrap_auc, choose_unit,
                               fit_logistic_l2, logistic_scores, prediction_verdict)
from part188.record188 import compare_gate1
from part188.select188 import assign_folds
import part188.analyze188 as analyze_module
import part188.record188 as record_module
import part188.session188 as session_module

ROOT = Path(__file__).resolve().parent


def brute_auc(labels, scores):
    positive = np.asarray(scores)[np.asarray(labels, dtype=bool)]
    negative = np.asarray(scores)[~np.asarray(labels, dtype=bool)]
    wins = sum(float(p > n) + 0.5 * float(p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def test_auc_matches_brute_force_with_ties():
    labels = np.asarray([1, 0, 1, 0, 1, 0], dtype=bool)
    values = np.asarray([[2.0, 0.0], [1.0, 0.0], [1.0, 2.0],
                         [1.0, 1.0], [0.0, 2.0], [0.0, 1.0]])
    actual = auc_per_unit(values, labels, chunk_size=1)
    expected = [brute_auc(labels, values[:, i]) for i in range(values.shape[1])]
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_selection_sign_and_lowest_index_ties():
    index, sign, raw, oriented = choose_unit(np.asarray([0.4, 0.6, 0.9, 0.1]))
    assert (index, sign) == (2, 1)
    assert raw == 0.9 and oriented == 0.9
    index, sign, raw, oriented = choose_unit(np.asarray([0.1, 0.9]))
    assert (index, sign, raw, oriented) == (0, -1, 0.1, 0.9)


def test_fold_split_is_deterministic_and_balanced():
    weights = {"a": 9, "b": 8, "c": 7, "d": 6}
    first = assign_folds(weights, seed=188)
    second = assign_folds(weights, seed=188)
    assert first == second
    loads = {fold: sum(weights[group] for group, assigned in first.items() if assigned == fold)
             for fold in ("A", "B")}
    assert abs(loads["A"] - loads["B"]) <= max(weights.values())
    assert set(first.values()) == {"A", "B"}


def test_manifest_has_unique_membership_and_no_cowbell_or_misplaced_chicken():
    with (ROOT / "manifest188.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    identity = [(row["source"], row["repository_path"] or row["filename"]) for row in rows]
    assert len(identity) == len(set(identity))
    assert all("cowbell" not in (row["class_key"] + row["repository_path"]).lower() for row in rows)
    assert {row["set"] for row in rows} <= {"R", "I", "N", "H", "E", "C_R", "C_I", "P"}
    for row in rows:
        if row["group"] == "Chicken":
            assert row["analysis_set"] in {"C_R", "C_I"}
            assert row["fold"] == "C"
        assert (row["analysis_set"] in {"C_R", "C_I"}) == (row["group"] == "Chicken")
        if row["analysis_set"] in {"N", "H", "E"}:
            assert row["fold"] == ""
        if row["analysis_set"] in {"R", "I"}:
            assert row["fold"] in {"A", "B"}


def test_group_cluster_bootstrap_resamples_whole_paired_groups():
    labels = np.asarray([1, 0] * 6, dtype=bool)
    groups = np.repeat(np.asarray([f"animal-{i}" for i in range(6)]), 2)
    scores = np.asarray([3.0, 0.0, 2.0, 0.0, 2.0, 1.0,
                         1.0, 0.0, 0.0, 0.0, -1.0, 0.0])
    first = bootstrap_auc(labels, scores, groups=groups, clustered=True, n_boot=200, seed=9)
    second = bootstrap_auc(labels, scores, groups=groups, clustered=True, n_boot=200, seed=9)
    assert first == second
    assert first["n_boot"] == 200
    assert 0.0 <= first["low"] <= first["high"] <= 1.0


def test_logistic_regression_separable_fixture():
    x = np.asarray([[-3.0, -1.0], [-2.0, 0.0], [-1.5, 1.0],
                    [1.5, -1.0], [2.0, 0.0], [3.0, 1.0]])
    y = np.asarray([0, 0, 0, 1, 1, 1])
    model = fit_logistic_l2(x, y, C=1.0)
    scores = logistic_scores(model, x)
    assert auc_score(y, scores) == 1.0


def test_logistic_regression_noisy_fixture():
    rng = np.random.default_rng(188)
    negative = rng.normal(loc=-0.5, scale=1.0, size=(120, 3))
    positive = rng.normal(loc=0.5, scale=1.0, size=(120, 3))
    x = np.vstack((negative, positive))
    y = np.r_[np.zeros(len(negative)), np.ones(len(positive))]
    model = fit_logistic_l2(x, y, C=1.0)
    scores = logistic_scores(model, x)
    assert auc_score(y, scores) > 0.70
    assert np.isfinite(scores).all()


def _newton_logistic_solution(x, labels, *, C, penalty_per_sample):
    data = np.asarray(x, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    mean = data.mean(axis=0)
    scale = data.std(axis=0)
    scale[scale < 1e-8] = 1.0
    design = np.column_stack(((data - mean) / scale, np.ones(len(y))))
    strength = 1.0 / (C * len(y)) if penalty_per_sample else 1.0 / C
    penalty = np.full(design.shape[1], strength)
    penalty[-1] = 0.0
    theta = np.zeros(design.shape[1])
    for _ in range(100):
        logits = design @ theta
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        gradient = design.T @ (probabilities - y) / len(y) + penalty * theta
        weights = probabilities * (1.0 - probabilities)
        hessian = (design.T * weights) @ design / len(y) + np.diag(penalty)
        change = np.linalg.solve(hessian, gradient)
        theta -= change
        if np.linalg.norm(change) < 1e-12:
            break
    return theta


def test_logistic_strength_matches_sklearn_summed_loss_convention():
    x = np.asarray([
        [-2.4, -1.0], [-1.8, 0.4], [-1.2, -0.6], [-0.6, 0.9], [-0.1, -0.3],
        [0.4, 0.8], [0.9, -0.7], [1.3, 0.5], [1.8, -0.4], [2.2, 1.1],
    ])
    y = np.asarray([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    model = fit_logistic_l2(x, y, C=1.0)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    design = np.column_stack(((x - mean) / scale, np.ones(len(y))))
    theta = np.r_[model["coef"], model["intercept"]]
    logits = design @ theta
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    summed_objective_gradient = design.T @ (probabilities - y) + np.r_[theta[:-1], 0.0]
    np.testing.assert_allclose(summed_objective_gradient, 0.0, atol=2e-5)

    old_penalty_solution = _newton_logistic_solution(x, y, C=1.0, penalty_per_sample=False)
    assert np.linalg.norm(theta - old_penalty_solution) > 0.1


def test_gate2_counts_only_ok_records_and_names_each_missing_clip():
    rows = [
        {"id": "good", "set": "R", "analysis_set": "R"},
        {"id": "record-failed", "set": "R", "analysis_set": "R"},
        {"id": "download-failed", "set": "R", "analysis_set": "R"},
        {"id": "decode-failed", "set": "R", "analysis_set": "R"},
        {"id": "parity", "set": "P", "analysis_set": "P"},
    ]
    statuses = np.asarray(["ok", "record_error", "download_error", "decode_error", "ok"])
    errors = ["", "encode failed", "network unavailable", "invalid audio", ""]
    result = analyze_module._gate2_summary(rows, statuses, errors)
    assert result["verdict"] == "failed"
    assert result["sets"]["R"]["recorded"] == 1
    assert result["sets"]["R"]["fraction"] == 0.25
    assert result["sets"]["R"]["missing"] == [
        {"id": "record-failed", "set": "R", "analysis_set": "R", "reason": "encode failed"},
        {"id": "download-failed", "set": "R", "analysis_set": "R", "reason": "network unavailable"},
        {"id": "decode-failed", "set": "R", "analysis_set": "R", "reason": "invalid audio"},
    ]
    assert result["sets"]["P"]["recorded"] == 1


def test_headline_is_withdrawn_when_any_gate_fails():
    predictions = {"188.2": {"verdict": "held"}, "188.3": {"verdict": "held"}}
    length = {"A_to_B": False, "B_to_A": False}
    permitted, reasons = analyze_module._headline_rule("passed", "failed", "passed", predictions, length)
    assert not permitted
    assert reasons == ["Gate 2 did not pass"]
    assert analyze_module._headline_rule("failed", "passed", "passed", predictions, length)[0] is False
    assert analyze_module._headline_rule("passed", "passed", "failed", predictions, length)[0] is False
    assert analyze_module._headline_rule("passed", "passed", "passed", predictions, length)[0] is True


def test_set_c_clustered_interval_uses_four_class_groups_and_has_width():
    rows = [
        {"source": "esc", "class_key": "hen", "esc_category": "hen"},
        {"source": "esc", "class_key": "rooster", "esc_category": "rooster"},
        {"source": "vis", "class_key": "Fowl / Chicken / Cluck", "esc_category": ""},
        {"source": "vis", "class_key": "Fowl / Chicken / Crowing, cock-a-doodle-doo", "esc_category": ""},
    ]
    indexes = np.repeat(np.arange(4), 2)
    groups = analyze_module._chicken_cluster_groups(rows, indexes)
    assert set(groups) == {"ESC-50 hen", "ESC-50 rooster",
                          "VocalImitationSet Cluck", "VocalImitationSet Crowing"}
    labels = np.asarray([1, 1, 1, 1, 0, 0, 0, 0])
    scores = np.asarray([1, 2, 7, 8, 3, 4, 5, 6], dtype=float)
    interval = bootstrap_auc(labels, scores, groups=groups, clustered=True, n_boot=500, seed=188)
    assert interval["low"] < interval["high"]


def test_prompt_phrase_missing_for_class_raises_named_error(tmp_path):
    prompts_path = tmp_path / "prompts.npz"
    np.savez(prompts_path, phrases=np.asarray(["an animal"]),
             real_embeddings=np.ones((1, 2), dtype=np.float32),
             imitation_embeddings=np.zeros((1, 2), dtype=np.float32))
    rows = [{"phrase": "a dog barking", "class_key": "dog class"}]
    with pytest.raises(KeyError, match="dog class"):
        analyze_module._prompt_scores(rows, np.asarray([0]), np.ones((1, 2)), prompts_path)


def test_analyze_stdout_contains_only_paths_counts_and_gate_verdicts(monkeypatch, capsys, tmp_path):
    fake_result = {
        "gate1": {"verdict": "passed"},
        "gate2": {"verdict": "failed", "sets": {
            "R": {"planned": 10, "recorded": 8, "fraction": 0.8}}},
        "gate3": {"verdict": "passed"},
        "predictions": {"secret": {"auc": 0.99, "unit": 36390}},
    }
    monkeypatch.setattr(analyze_module, "analyze", lambda **_kwargs: fake_result)
    monkeypatch.setattr(sys, "argv", ["analyze188.py", "--out-dir", str(tmp_path)])
    assert analyze_module.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert set(output) == {"results", "gate_verdicts", "counts"}
    assert output["gate_verdicts"] == {"gate1": "passed", "gate2": "failed", "gate3": "passed"}
    assert output["counts"]["R"] == {"planned": 10, "recorded": 8, "fraction": 0.8}


def test_session_summary_records_all_gate_verdicts_after_analysis(monkeypatch, tmp_path, capsys):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(session_module, "ROOT", root)
    monkeypatch.setattr(session_module, "verify_pinned_versions", lambda: {"python": "fixture"})
    work_dir = tmp_path / "work"
    out_dir = root / "cloud-results" / "part188"

    def fake_stage(name, _command, _log, _timings):
        if name == "record":
            work_dir.mkdir(parents=True, exist_ok=True)
            (work_dir / "gate1.json").write_text(json.dumps({"verdict": "passed"}))
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            results = {
                "gate2": {"verdict": "failed", "sets": {
                    "R": {"planned": 10, "recorded": 8, "fraction": 0.8, "verdict": "failed"}}},
                "gate3": {"verdict": "failed"},
            }
            (out_dir / "results.json").write_text(json.dumps(results))
        return 0

    monkeypatch.setattr(session_module, "_run_stage", fake_stage)
    summary = session_module.run_session(work_dir=work_dir, out_dir=out_dir)
    assert summary["analysis_ran"] is True
    assert summary["gate_verdicts"] == {"gate1": "passed", "gate2": "failed", "gate3": "failed"}
    saved = json.loads((out_dir / "summary.json").read_text())
    assert saved["gate_verdicts"] == summary["gate_verdicts"]
    assert "auc" not in capsys.readouterr().out.lower()


def test_session_stdout_is_sanitized_and_exit_stays_zero_after_analysis(monkeypatch, capsys, tmp_path):
    summary = {
        "analysis_ran": True,
        "gate_verdicts": {"gate1": "passed", "gate2": "failed", "gate3": "failed"},
        "counts": {"R": {"planned": 10, "recorded": 8, "fraction": 0.8}},
        "timings_seconds": {"analyze": 12.0}, "peak_memory_mb": 100.0,
        "problems": ["AUC=0.99 unit=36390"],
    }
    monkeypatch.setattr(session_module, "run_session", lambda **_kwargs: summary)
    out_dir = tmp_path / "cloud-results" / "part188"
    monkeypatch.setattr(sys, "argv", ["session188.py", "--out-dir", str(out_dir)])
    assert session_module.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert set(output) == {"summary", "gate_verdicts", "counts"}
    assert output["gate_verdicts"]["gate2"] == "failed"


def test_session_filters_child_logs_to_paths_counts_gate_and_throughput(monkeypatch, capsys, tmp_path):
    from types import SimpleNamespace

    child_stdout = "\n".join([
        "clips_processed=50/100 clips_per_minute=50.00",
        '{"gate1":"passed","planned":100,"record_status_counts":{"ok":100},"done_file":"/tmp/done.jsonl"}',
        '{"prediction":"held","auc":0.99,"unit":36390}',
    ])
    monkeypatch.setattr(session_module.subprocess, "run", lambda *_args, **_kwargs:
                        SimpleNamespace(returncode=0, stdout=child_stdout, stderr="AUC=0.99"))
    timings = {}
    assert session_module._run_stage("record", ["record188.py"], None, timings) == 0
    output = capsys.readouterr().out.splitlines()
    assert output == [
        "clips_processed=50/100 clips_per_minute=50.00",
        '{"done_file": "/tmp/done.jsonl", "gate1": "passed", "planned": 100, "record_status_counts": {"ok": 100}}',
    ]


def test_downloads_retry_three_times_with_backoff():
    attempts = []
    delays = []

    def download():
        attempts.append(len(attempts) + 1)
        if len(attempts) < 4:
            raise OSError("transient")
        return "saved"

    assert record_module._retry_download(download, sleep=delays.append) == "saved"
    assert len(attempts) == 4
    assert delays == [1.0, 2.0, 4.0]


def test_vis_clip_download_uses_retry_wrapper(monkeypatch, tmp_path):
    attempts = []
    delays = []

    def fake_curl(_url, destination, **_kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("transient")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"audio")

    monkeypatch.setattr(record_module, "_curl_download", fake_curl)
    retry = record_module._retry_download
    monkeypatch.setattr(record_module, "_retry_download",
                        lambda operation: retry(operation, sleep=delays.append))
    path = record_module.download_vis_clip({"id": "clip", "download_url": "https://example.test/clip"},
                                           tmp_path)
    assert path.read_bytes() == b"audio"
    assert len(attempts) == 3
    assert delays == [1.0, 2.0]


def test_restart_retries_download_errors_but_keeps_other_errors_final(tmp_path):
    record_dir = tmp_path / "records"
    record_dir.mkdir()
    rows = [{"id": "new"}, {"id": "download"}, {"id": "decode"},
            {"id": "record"}, {"id": "ok"}]
    for row, status in zip(rows[1:], ("download_error", "decode_error", "record_error", "ok")):
        np.savez(record_dir / f"{row['id']}.npz", id=np.asarray(row["id"]),
                 status=np.asarray(status))
    pending = record_module._download_pending_rows(rows, record_dir)
    assert [row["id"] for row in pending] == ["new", "download"]


class _ArrayTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    @property
    def shape(self):
        return self.values.shape

    def mean(self, axis):
        return _ArrayTensor(self.values.mean(axis=axis))

    def reshape(self, *shape):
        return _ArrayTensor(self.values.reshape(*shape))

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _RecordingTorch:
    @staticmethod
    def inference_mode():
        from contextlib import nullcontext
        return nullcontext()

    @staticmethod
    def stack(values):
        return _ArrayTensor(np.stack([value.values for value in values]))


class _RecordingModel:
    def __init__(self, buffer, *, fail_first=False, always_fail=False):
        self._part188_buffer = buffer
        self.calls = 0
        self.fail_first = fail_first
        self.always_fail = always_fail

    def encode(self, _item, **_kwargs):
        self.calls += 1
        if self.always_fail or (self.fail_first and self.calls == 1):
            raise RuntimeError("temporary forward failure")
        self._part188_buffer[0] = _ArrayTensor([[1.0, 2.0]])
        return np.zeros(768, dtype=np.float32)


def _recording_fixture(monkeypatch, tmp_path, *, model=None):
    monkeypatch.setattr(record_module, "N_UNITS", 2)
    monkeypatch.setitem(sys.modules, "torch", _RecordingTorch)
    source = tmp_path / "clip.wav"
    source.write_bytes(b"synthetic audio bytes")
    record_dir = tmp_path / "records"
    record_dir.mkdir()
    row = {"id": "clip", "filename": "clip.wav", "source": "vis"}
    model = model or _RecordingModel({})
    return row, source, record_dir, model


def test_process_one_retries_decode_failure_before_saving_error(monkeypatch, tmp_path):
    row, source, record_dir, model = _recording_fixture(monkeypatch, tmp_path)
    decode_calls = []

    def decode(_path):
        decode_calls.append(1)
        if len(decode_calls) == 1:
            raise OSError("temporary decoder failure")
        return np.asarray([0.25], dtype=np.float32)

    monkeypatch.setattr(record_module, "decode_audio", decode)
    result = record_module._process_one(row, source, record_dir, model, _RecordingTorch, 1)
    assert result[1] == "ok"
    assert len(decode_calls) == 2
    assert model.calls == 1
    with np.load(record_dir / "clip.npz", allow_pickle=False) as saved:
        assert str(saved["status"].item()) == "ok"


def test_process_one_retries_forward_failure_once(monkeypatch, tmp_path):
    model = _RecordingModel({}, fail_first=True)
    row, source, record_dir, model = _recording_fixture(monkeypatch, tmp_path, model=model)
    decode_calls = []
    monkeypatch.setattr(record_module, "decode_audio", lambda _path: (
        decode_calls.append(1) or np.asarray([0.25], dtype=np.float32)))
    result = record_module._process_one(row, source, record_dir, model, _RecordingTorch, 1)
    assert result[1] == "ok"
    assert model.calls == 2
    assert len(decode_calls) == 2


def test_process_one_saves_record_error_only_after_both_attempts(monkeypatch, tmp_path):
    model = _RecordingModel({}, always_fail=True)
    row, source, record_dir, model = _recording_fixture(monkeypatch, tmp_path, model=model)
    decode_calls = []

    def decode(_path):
        decode_calls.append(1)
        return np.asarray([0.25], dtype=np.float32)

    monkeypatch.setattr(record_module, "decode_audio", decode)
    result = record_module._process_one(row, source, record_dir, model, _RecordingTorch, 1)
    assert result[1] == "record_error"
    assert model.calls == 2
    assert len(decode_calls) == 2
    with np.load(record_dir / "clip.npz", allow_pickle=False) as saved:
        assert str(saved["status"].item()) == "record_error"
        assert "temporary forward failure" in str(saved["error"].item())


def test_process_one_saves_decode_error_only_after_both_attempts(monkeypatch, tmp_path):
    row, source, record_dir, model = _recording_fixture(monkeypatch, tmp_path)
    decode_calls = []

    def decode(_path):
        decode_calls.append(1)
        raise OSError("permanent decoder failure")

    monkeypatch.setattr(record_module, "decode_audio", decode)
    result = record_module._process_one(row, source, record_dir, model, _RecordingTorch, 1)
    assert result[1] == "decode_error"
    assert len(decode_calls) == 2
    assert model.calls == 0
    with np.load(record_dir / "clip.npz", allow_pickle=False) as saved:
        assert str(saved["status"].item()) == "decode_error"
        assert "permanent decoder failure" in str(saved["error"].item())


def test_resume_rerecords_record_with_mismatched_manifest_identity(tmp_path):
    record_dir = tmp_path / "records"
    record_dir.mkdir()
    row = {"id": "same-id", "filename": "new.wav", "repository_path": "new/path.wav"}
    np.savez(record_dir / "same-id.npz", id=np.asarray("same-id"),
             filename=np.asarray("old.wav"), status=np.asarray("ok"))
    assert record_module._download_pending_rows([row], record_dir) == [row]
    with np.load(record_dir / "same-id.npz", allow_pickle=False) as saved:
        assert record_module._record_identity_mismatch(saved, row) == (
            "stored filename does not match manifest")

    np.savez(record_dir / "same-id.npz", id=np.asarray("same-id"),
             filename=np.asarray("new.wav"), status=np.asarray("ok"))
    assert record_module._download_pending_rows([row], record_dir) == []
    np.savez(record_dir / "same-id.npz", id=np.asarray("same-id"), status=np.asarray("ok"))
    assert record_module._download_pending_rows([row], record_dir) == []
    np.savez(record_dir / "same-id.npz", id=np.asarray("same-id"),
             repository_path=np.asarray("new/path.wav"), status=np.asarray("ok"))
    assert record_module._download_pending_rows([row], record_dir) == []
    np.savez(record_dir / "same-id.npz", id=np.asarray("other-id"),
             filename=np.asarray("new.wav"), status=np.asarray("ok"))
    assert record_module._download_pending_rows([row], record_dir) == [row]


def test_analyze_completes_without_using_non_ok_nan_rows(tmp_path, monkeypatch):
    rows = []

    def add(row_id, set_name, analysis_set, *, fold="", source="vis", category="",
            group="dog", class_key="Dogs / Bark", participant_id=""):
        rows.append({
            "id": row_id, "set": set_name, "analysis_set": analysis_set,
            "source": source, "filename": f"{row_id}.wav", "repository_path": "",
            "class_key": class_key, "group": group, "phrase": "a dog barking",
            "fold": fold, "participant_id": participant_id, "esc_category": category,
        })

    add("r-a-ok", "R", "R", fold="A")
    add("r-a-error", "R", "R", fold="A")
    add("r-b-1", "R", "R", fold="B")
    add("r-b-2", "R", "R", fold="B")
    add("i-a-1", "I", "I", fold="A", participant_id="imitation-a1")
    add("i-a-2", "I", "I", fold="A", participant_id="imitation-a2")
    add("i-b-ok", "I", "I", fold="B", participant_id="imitation-b1")
    add("i-b-error", "I", "I", fold="B", participant_id="imitation-b2")
    add("c-r-esc", "C", "C_R", source="esc", category="hen", group="Chicken",
        class_key="hen")
    add("c-r-vis", "C", "C_R", source="vis", group="Chicken",
        class_key="Fowl / Chicken / Cluck")
    add("c-i-esc", "C", "C_I", source="esc", category="rooster", group="Chicken",
        class_key="rooster")
    add("c-i-vis", "C", "C_I", source="vis", group="Chicken",
        class_key="Fowl / Chicken / Crowing, cock-a-doodle-doo")
    add("h-vocal", "H", "H", source="esc", category="crying_baby", group="Human",
        class_key="crying_baby")
    add("h-nonvocal", "H", "H", source="esc", category="clapping", group="Human",
        class_key="clapping")
    add("n-1", "N", "N", group="dog")
    add("n-2", "N", "N", group="cat", class_key="Cats / Meow")
    add("e-ok", "E", "E", source="esc", category="sneezing", group="Human",
        class_key="sneezing")
    add("e-error", "E", "E", source="esc", category="footsteps", group="Human",
        class_key="footsteps")

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    work_dir = tmp_path / "work"
    record_dir = work_dir / "records"
    record_dir.mkdir(parents=True)
    rng = np.random.default_rng(188)
    monkeypatch.setattr(analyze_module, "N_UNITS", analyze_module.L17_N1574 + 1)
    monkeypatch.setattr(analyze_module, "BOOTSTRAPS", 8)
    failed = {"r-a-error": "encode failed", "i-b-error": "forward failed",
              "e-error": "invalid audio"}
    for index, row in enumerate(rows):
        status = "record_error" if row["id"] in failed else "ok"
        activation = (np.full(analyze_module.N_UNITS, np.nan, dtype=np.float16) if status != "ok"
                      else rng.normal(size=analyze_module.N_UNITS).astype(np.float16))
        embedding = (np.full(768, np.nan, dtype=np.float32) if status != "ok"
                     else rng.normal(size=768).astype(np.float32))
        np.savez_compressed(
            record_dir / f"{row['id']}.npz", id=np.asarray(row["id"]),
            act_mean=activation, emb=embedding, seqlen=np.asarray(index + 1),
            status=np.asarray(status), decode_status=np.asarray("ok"),
            error=np.asarray(failed.get(row["id"], "")))

    prompts_path = work_dir / "prompts.npz"
    np.savez_compressed(
        prompts_path, phrases=np.asarray(["a dog barking", "an animal"]),
        real_texts=np.asarray(["the sound of a dog barking", "the sound of an animal"]),
        imitation_texts=np.asarray(["a person imitating a dog", "a person imitating an animal"]),
        real_embeddings=rng.normal(size=(2, 768)).astype(np.float32),
        imitation_embeddings=rng.normal(size=(2, 768)).astype(np.float32))

    for function_name, input_position in (("auc_per_unit", 0), ("auc_score", 1),
                                          ("fit_logistic_l2", 0), ("logistic_scores", 1)):
        original = getattr(analyze_module, function_name)

        def guard(*args, _original=original, _position=input_position, _name=function_name, **kwargs):
            assert np.isfinite(np.asarray(args[_position])).all(), f"NaN reached {_name}"
            return _original(*args, **kwargs)

        monkeypatch.setattr(analyze_module, function_name, guard)

    original_bootstrap = analyze_module.bootstrap_auc

    def guarded_bootstrap(labels, scores, **kwargs):
        assert np.isfinite(np.asarray(scores)).all(), "NaN reached bootstrap"
        return original_bootstrap(labels, scores, **kwargs)

    monkeypatch.setattr(analyze_module, "bootstrap_auc", guarded_bootstrap)
    original_prompt = analyze_module._prompt_scores

    def guarded_prompt(rows_arg, indexes, embeddings, *args, **kwargs):
        assert np.isfinite(embeddings[np.asarray(indexes, dtype=int)]).all(), "NaN reached prompt scoring"
        return original_prompt(rows_arg, indexes, embeddings, *args, **kwargs)

    monkeypatch.setattr(analyze_module, "_prompt_scores", guarded_prompt)
    out_dir = tmp_path / "out"
    result = analyze_module.analyze(manifest_path=manifest, work_dir=work_dir, out_dir=out_dir)
    assert (out_dir / "results.json").is_file()
    assert json.loads((out_dir / "results.json").read_text(encoding="utf-8")) == result
    missing = [item for summary in result["gate2"]["sets"].values() for item in summary["missing"]]
    assert sorted((item["id"], item["reason"]) for item in missing) == sorted(failed.items())
    assert result["gate3"]["n_negative"] == 1
    assert result["known_unit_l17_n1574"]["n"] == sum(
        row["analysis_set"] in ("R", "I") and row["id"] not in failed for row in rows)


def test_gate1_requires_exactly_eight_recorded_compared_parity_rows():
    parity_rows = [{"id": f"p{i}", "filename": f"clip-{i}.wav"} for i in range(8)]
    reference_names = [row["filename"] for row in parity_rows]
    comparison = {"activation_fraction_within_tolerance": 1.0,
                  "activation_max_absolute_difference": 0.0, "embedding_cosine": 1.0}
    gate_rows = [{"id": row["id"], "filename": row["filename"],
                  "passed": True, **comparison} for row in parity_rows]
    assert record_module._gate1_verdict(parity_rows, gate_rows, reference_names)
    assert not record_module._gate1_verdict(parity_rows[:7], gate_rows[:7], reference_names[:7])
    assert not record_module._gate1_verdict([], [], [])
    assert not record_module._gate1_verdict(parity_rows, gate_rows[:-1], reference_names)
    uncompared = gate_rows.copy()
    uncompared[0] = {"id": "p0", "filename": "clip-0.wav", "passed": True}
    assert not record_module._gate1_verdict(parity_rows, uncompared, reference_names)


def test_done_rows_include_clip_wall_time_and_throughput_is_logged_each_50(tmp_path):
    done_path = tmp_path / "done.jsonl"
    record_module._append_done(done_path, "clip-1", "ok", wall_seconds=1.25)
    row = json.loads(done_path.read_text())
    assert row == {"id": "clip-1", "status": "ok", "wall_seconds": 1.25}
    assert record_module._should_log_throughput(50)
    assert not record_module._should_log_throughput(49)
    assert record_module._throughput_line(50, 100, 60.0) == (
        "clips_processed=50/100 clips_per_minute=50.00")


def test_gate1_passes_identical_rows_and_fails_changed_activation_row():
    activation = np.zeros(49_152, dtype=np.float16)
    embedding = np.zeros(768, dtype=np.float32)
    embedding[0] = 1.0
    passed = compare_gate1(activation, embedding, activation, embedding)
    assert passed["passed"]
    changed = activation.copy()
    changed[:50] = 1.0  # More than the registered 0.1% mismatch allowance.
    failed = compare_gate1(changed, embedding, activation, embedding)
    assert not failed["passed"]
    assert failed["activation_fraction_within_tolerance"] < 0.999


def test_prediction_188_1_threshold():
    assert prediction_verdict("188.1", {"A": 0.749, "B": 0.70})
    assert not prediction_verdict("188.1", {"A": 0.75, "B": 0.70})


def test_prediction_188_2_threshold():
    assert prediction_verdict("188.2", {"A": 0.85, "B": 0.86})
    assert not prediction_verdict("188.2", {"A": 0.849, "B": 0.86})


def test_prediction_188_3_threshold():
    assert prediction_verdict("188.3", {"A": 0.10, "B": 0.11})
    assert not prediction_verdict("188.3", {"A": 0.099, "B": 0.11})


def test_prediction_188_4_threshold():
    assert prediction_verdict("188.4", {"A": 0.80, "B": 0.81})
    assert not prediction_verdict("188.4", {"A": 0.799, "B": 0.81})


def test_prediction_188_5_threshold():
    assert prediction_verdict("188.5", 0.699)
    assert not prediction_verdict("188.5", 0.70)


def test_prediction_188_6_threshold():
    assert prediction_verdict("188.6", {"A": 0.90, "B": 0.91})
    assert not prediction_verdict("188.6", {"A": 0.89, "B": 0.91})


def test_prediction_188_7_holds_when_probe_is_below_selected_unit():
    assert prediction_verdict("188.7", {"A": -0.20, "B": -0.01})
    assert not prediction_verdict("188.7", {"A": 0.05, "B": 0.01})


def test_prediction_188_8_threshold():
    assert prediction_verdict("188.8", {"A": 0.80, "B": 0.81})
    assert not prediction_verdict("188.8", {"A": 0.79, "B": 0.81})


def test_prediction_188_9_threshold():
    assert prediction_verdict("188.9", {"A": 0.699, "B": 0.60})
    assert not prediction_verdict("188.9", {"A": 0.70, "B": 0.60})


def test_requirements_pin_every_expected_library():
    """Amendment 1 (2026-11-04): the first cloud run failed at model load because torchvision, present in the local
    stack, was missing from requirements188.txt. Every library the session checks must be pinned there, at the same
    version."""
    import re
    from pathlib import Path
    import session188
    text = (Path(__file__).resolve().parent / "requirements188.txt").read_text(encoding="utf-8")
    pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s]+)$", text, flags=re.M))
    assert "torchvision" in session188.EXPECTED
    for name, version in session188.EXPECTED.items():
        assert pins.get(name) == version, (name, pins.get(name), version)
