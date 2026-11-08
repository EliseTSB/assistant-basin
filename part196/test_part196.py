"""Synthetic tests for Part 196 selection, split checks, and analysis math."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from part196.analyze196 import (  # noqa: E402
    fit_calibrated_verdict,
    headline_outcome,
    macro_ovr_auc,
    mean_fold_auc,
    select_b_phrasings,
)
from part196.record196 import acoustic_features  # noqa: E402
from part196.select196 import STYLES, draw_manifest, gate1_check, shared_text_pool  # noqa: E402


def fake_metadata() -> list[dict[str, str]]:
    rows = []
    for speaker in ("ex01", "ex02", "ex03", "ex04"):
        for text_number in range(3):
            text = f"{speaker} unique sentence {text_number}"
            for style in STYLES:
                rows.append({"id": f"{speaker}_{style}_{text_number}", "speaker_id": speaker,
                             "style": style, "text": text})
    # This exact transcript leaks across speakers and must leave the pool.
    rows.extend([
        {"id": "cross_a", "speaker_id": "ex01", "style": "happy", "text": "shared speaker text"},
        {"id": "cross_b", "speaker_id": "ex02", "style": "sad", "text": "shared speaker text"},
    ])
    # A text seen in only one style cannot be used for the controlled pool.
    rows.append({"id": "single", "speaker_id": "ex01", "style": "sad", "text": "single style"})
    return rows


def test_pool_and_draw_apply_shared_text_rules_and_registered_order():
    metadata = fake_metadata()
    pooled = shared_text_pool(metadata)
    assert all(row["text"] != "shared speaker text" for row in pooled)
    assert all(row["text"] != "single style" for row in pooled)
    assert len(pooled) == 4 * 3 * 7
    first, sizes = draw_manifest(reversed(metadata), seed=196, per_cell=2)
    second, _ = draw_manifest(metadata, seed=196, per_cell=2)
    assert first == second
    assert len(first) == 4 * 7 * 2
    assert set(sizes.values()) == {3}
    assert [row["style"] for row in first[:14]] == [style for style in STYLES for _ in range(2)]
    assert all(row["duration_status"] == "deferred_to_recorder" for row in first)
    assert "text" not in first[0] and len(first[0]["text_key"]) == 64


def _gate_fixture():
    rows = []
    for speaker_index, speaker in enumerate(("ex01", "ex02", "ex03", "ex04")):
        for style in STYLES:
            rows.append({"id": f"{speaker}_{style}", "speaker_id": speaker,
                         "style": style, "text": f"{speaker} {style} sentence",
                         "fold": speaker})
    return rows


def test_gate1_refuses_speaker_text_or_id_overlap():
    baseline = _gate_fixture()
    assert gate1_check(baseline, min_per_cell=1)["verdict"] == "passed"

    speaker_overlap = [dict(row) for row in baseline]
    speaker_overlap[7]["fold"] = "ex01"
    failed = gate1_check(speaker_overlap, min_per_cell=1)
    assert failed["verdict"] == "failed"
    assert not failed["folds"]["ex01"]["speaker_id"]["passed"]

    text_overlap = [dict(row) for row in baseline]
    text_overlap[7]["text"] = text_overlap[0]["text"]
    failed = gate1_check(text_overlap, min_per_cell=1)
    assert failed["verdict"] == "failed"
    assert not failed["folds"]["ex01"]["text"]["passed"]

    id_overlap = [dict(row) for row in baseline]
    id_overlap[7]["id"] = id_overlap[0]["id"]
    failed = gate1_check(id_overlap, min_per_cell=1)
    assert failed["verdict"] == "failed"
    assert not failed["folds"]["ex01"]["id"]["passed"]


def test_b_fit_recovers_known_scale_and_offsets_with_constraints():
    true_scale = 1.8
    true_offsets = np.asarray([0.45, -0.25, 0.12, 0.03, -0.10, -0.15, -0.10])
    base_logits = np.asarray([0.70, 0.20, 0.00, -0.10, -0.25, -0.30, -0.25])
    base_probability = np.exp(base_logits - np.logaddexp.reduce(base_logits))
    feature_rows, labels = [], []
    for rotation in range(7):
        logits = np.roll(base_logits, rotation)
        probability = np.roll(base_probability, rotation)
        counts = np.rint(probability * 6000).astype(int)
        # Correct rounding so each pattern has exactly 6000 examples.
        counts[0] += 6000 - int(counts.sum())
        cosine = (logits - true_offsets) / true_scale
        for cls, count in enumerate(counts):
            feature_rows.extend([cosine] * int(count))
            labels.extend([cls] * int(count))
    fitted = fit_calibrated_verdict(np.asarray(feature_rows), np.asarray(labels))
    assert fitted["success"]
    assert fitted["a"] > 0
    assert fitted["a"] == pytest.approx(true_scale, abs=0.04)
    np.testing.assert_allclose(fitted["b"], true_offsets, atol=0.035)
    assert float(np.sum(fitted["b"])) == pytest.approx(0.0, abs=1e-10)


def test_b_phrasing_selection_ties_to_mean_then_lower_phrase():
    labels = np.repeat(np.arange(7), 4)
    cosines = np.zeros((len(labels), 7, 8), dtype=np.float64)
    _selected, details = select_b_phrasings(cosines, labels)
    assert all(item["choice"] == "mean" and item["phrase_index"] is None for item in details)


def test_macro_auc_is_averaged_within_folds_never_pooled():
    y1 = np.asarray([0, 1])
    y2 = np.asarray([0, 1])
    scores1 = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    scores2 = np.asarray([[-99.0, -100.0], [-100.0, -99.0]])
    within = mean_fold_auc([(y1, scores1), (y2, scores2)], n_classes=2)
    pooled, _ = macro_ovr_auc(np.r_[y1, y2], np.vstack((scores1, scores2)), n_classes=2)
    assert within == pytest.approx(1.0)
    assert pooled < within


def test_headline_three_way_outcome_on_constructed_intervals():
    assert headline_outcome(0.06, {"low": 0.01, "high": 0.10}) == "gap supported"
    assert headline_outcome(0.04, {"low": 0.01, "high": 0.10}) == "inconclusive"
    assert headline_outcome(0.00, {"low": -0.02, "high": 0.02}) == "practically equivalent"
    assert headline_outcome(0.06, {"low": -0.01, "high": 0.12}) == "inconclusive"


def test_h_keeps_clip_when_voiced_frames_are_absent(monkeypatch):
    fake_pyworld = types.SimpleNamespace(
        harvest=lambda audio, fs, frame_period: (np.zeros(17, dtype=np.float64),
                                                  np.arange(17, dtype=np.float64) * 0.005))
    monkeypatch.setitem(sys.modules, "pyworld", fake_pyworld)
    features = acoustic_features(np.zeros(32000, dtype=np.float32), "no voiced frames here")
    assert features.shape == (7,)
    assert features[0] == 0.0
    assert features[1] == 0.0
    assert features[2] == 0.0

