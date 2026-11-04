"""Fixture-only checks for the registered Part 188 math and manifest."""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from part188.analyze188 import (auc_per_unit, auc_score, bootstrap_auc, choose_unit,
                               fit_logistic_l2, logistic_scores, prediction_verdict)
from part188.record188 import compare_gate1
from part188.select188 import assign_folds

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


def test_prediction_188_7_threshold():
    assert prediction_verdict("188.7", {"A": 0.049, "B": -0.01})
    assert not prediction_verdict("188.7", {"A": 0.05, "B": 0.01})


def test_prediction_188_8_threshold():
    assert prediction_verdict("188.8", {"A": 0.80, "B": 0.81})
    assert not prediction_verdict("188.8", {"A": 0.79, "B": 0.81})


def test_prediction_188_9_threshold():
    assert prediction_verdict("188.9", {"A": 0.699, "B": 0.60})
    assert not prediction_verdict("188.9", {"A": 0.70, "B": 0.60})
