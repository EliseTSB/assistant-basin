"""Synthetic state tests for Part 181 overlap and prediction rules."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze181 as a181  # noqa: E402


def synthetic_run(key, max_values, top_docs, top_pos=None, controls=0):
    max_values = np.asarray(max_values, dtype=np.float32)
    if max_values.ndim == 1:
        max_values = max_values[None, :]
    top_docs = np.asarray(top_docs, dtype=np.int32)
    if top_docs.ndim == 2:
        top_docs = top_docs[None, :, :]
    if top_pos is None:
        top_pos = np.tile(np.arange(top_docs.shape[-1], dtype=np.int16),
                          (top_docs.shape[0], top_docs.shape[1], 1))
    else:
        top_pos = np.asarray(top_pos, dtype=np.int16)
        if top_pos.ndim == 2:
            top_pos = top_pos[None, :, :]
    width = max_values.shape[1] - controls
    config = {"layers": list(range(max_values.shape[0])), "width": width, "n_controls": controls}
    top_act = np.broadcast_to(max_values[:, :, None], top_docs.shape).copy()
    state = {"max": max_values, "top_act": top_act, "top_doc": top_docs, "top_pos": top_pos}
    return a181.Run(key, Path("."), {}, config, state)


def test_identical_and_disjoint_top_lists_have_expected_overlap():
    docs = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    pos = np.tile(np.arange(20), (1, 1))
    identical = a181._metric_block(docs, pos, docs.copy(), pos.copy())
    disjoint = a181._metric_block(docs, pos, docs + 100, pos.copy())
    assert identical["top10_doc"][0] == 1.0
    assert identical["top10_document_position"][0] == 1.0
    assert disjoint["top10_doc"][0] == 0.0
    assert disjoint["top20_doc"][0] == 0.0


def test_active_rule_uses_ten_percent_of_layer_median():
    assert a181._active(np.asarray([0.01, 1.0, 2.0])).tolist() == [False, True, True]
    assert a181._active(np.asarray([-2.0, -1.0, 0.0])).tolist() == [False, False, False]


def test_chance_baseline_is_near_ten_over_2500():
    rng = np.random.default_rng(77)
    n_units = 512
    a_docs = np.stack([rng.choice(2500, size=20, replace=False) for _ in range(n_units)]).astype(np.int32)
    b_docs = np.stack([rng.choice(2500, size=20, replace=False) for _ in range(n_units)]).astype(np.int32)
    maxima = np.ones((1, n_units), dtype=np.float32)
    a = synthetic_run("a", maxima, a_docs)
    b = synthetic_run("b", maxima, b_docs)
    chance = a181.chance_summary(a, b, [0], width=n_units, n_controls=0, permutations=100, seed=181)
    measured = chance["groups"]["neurons"]["all_layers"]["top10_doc"]["median_permutation_mean"]
    assert measured == pytest.approx(10 / 2500, abs=0.0015)


def test_strict_monotone_rule_rejects_equal_adjacent_values():
    assert a181.classify_strictly_increasing([0.01, 0.02, 0.03, 0.04, 0.05, 0.06]) == "held"
    assert a181.classify_strictly_increasing([0.01, 0.02, 0.02, 0.04, 0.05, 0.06]) == "failed"
    assert a181.classify_strictly_increasing([0.01, 0.02, None, 0.04, 0.05, 0.06]) == "not assessable"


@pytest.mark.parametrize("value,threshold,direction,expected", [
    (0.6, 0.6, "at_least", "held"), (0.59, 0.6, "at_least", "failed"),
    (0.9, 0.9, "at_least", "held"), (0.899, 0.9, "at_least", "failed"),
    (0.1, 0.1, "at_most", "held"), (0.101, 0.1, "at_most", "failed"),
])
def test_prediction_threshold_boundaries(value, threshold, direction, expected):
    assert a181.classify_threshold(value, threshold, direction) == expected


def test_layer_advantage_needs_twelve_strict_wins():
    controls = [0.5] * 16
    held, held_numbers = a181.classify_layer_advantage([0.6] * 12 + [0.5] * 4, controls)
    failed, failed_numbers = a181.classify_layer_advantage([0.6] * 11 + [0.5] * 5, controls)
    assert held == "held" and held_numbers["neuron_wins"] == 12
    assert failed == "failed" and failed_numbers["neuron_wins"] == 11


def test_slot_distance_boundary_is_inclusive():
    assert a181.classify_slot_distance(0.8, 0.65)[0] == "held"
    assert a181.classify_slot_distance(0.8001, 0.65)[0] == "failed"
    assert a181.classify_slot_distance(0.8, 0.65, slot_count=20)[0] == "not assessable"


def test_pair_summary_excludes_units_inactive_at_either_checkpoint():
    docs_a = np.tile(np.arange(20), (3, 1))
    docs_b = docs_a.copy()
    a = synthetic_run("a", [0.01, 1, 2], docs_a)
    b = synthetic_run("b", [1, 1, 2], docs_b)
    a.config["n_controls"] = b.config["n_controls"] = 0
    result = a181.summarize_pair(a, b)
    neurons = result["groups"]["neurons"]["cohorts"]["all"]["layers"]["0"]
    assert neurons["coverage_a"] == 2
    assert neurons["coverage_b"] == 3
    assert neurons["active_both"] == 2
    assert neurons["overlap"]["top10_doc"]["median"] is None


def test_fixed_primary_cohort_and_derangement_null_are_reported():
    count = 60
    docs = np.tile(np.arange(20), (count, 1))
    maxima = np.ones(count, dtype=np.float32)
    first, second = synthetic_run("a", maxima, docs), synthetic_run("b", maxima, docs)
    summary = a181.summarize_pair(first, second, primary_ids_by_layer={0: np.arange(50)})
    cohorts = summary["groups"]["neurons"]["cohorts"]
    assert cohorts["primary"]["layers"]["0"]["units"] == 50
    assert cohorts["all"]["layers"]["0"]["units"] == 60
    assert cohorts["primary"]["all_layers"]["overlap"]["top10_doc"]["median"] == 1.0
    null = summary["null"]["neurons"]["primary"]["0"]
    assert len(null["median_distribution"]) == 100
    assert len(null["mean_distribution"]) == 100
    order = a181._derangement_indices(100, np.random.default_rng(181))
    assert np.all(order != np.arange(100))


def test_layer_under_fifty_is_reported_but_not_assessed():
    docs = np.tile(np.arange(20), (49, 1))
    maxima = np.ones(49, dtype=np.float32)
    result = a181.summarize_pair(synthetic_run("a", maxima, docs), synthetic_run("b", maxima, docs))
    layer = result["groups"]["neurons"]["cohorts"]["all"]["layers"]["0"]
    assert layer["reported"] is True
    assert layer["overlap"]["top10_doc"]["median"] is None
    assert "0" in result["groups"]["neurons"]["cohorts"]["all"]["all_layers"]["excluded_under_50_layers"]


def test_stage_summary_reports_per_unit_firing_and_position_zero_counts():
    docs = np.tile(np.arange(20), (3, 1))
    run = synthetic_run("stage", [0.0, 1.0, 2.0], docs)
    stage = a181._stage_summaries([run])["stage"]
    neurons = stage["populations"]["neurons"]["layers"]["0"]
    assert neurons["unit_ids"] == [0, 1, 2]
    assert neurons["per_unit_firing_count"] == [0, 20, 20]
    assert neurons["per_unit_top10_position0_count"] == [1, 1, 1]
    assert stage["layers"]["0"]["top10_position0_share"] == 0.1
    assert stage["populations"]["neurons"]["all_layers"]["units_under_10"] == 1


def test_gate3_excludes_a_failing_layer_despite_a_passing_global_median():
    n_layers, n_units = 3, 60
    docs = np.tile(np.arange(20), (n_layers, n_units, 1)).astype(np.int32)
    maxima = np.ones((n_layers, n_units), dtype=np.float32)
    good = synthetic_run("instruct", maxima, docs)
    instruct_b_docs = docs.copy()
    instruct_b_docs[2, :31] += 100
    instruct_b = synthetic_run("instruct_b", maxima, instruct_b_docs)
    pretrain = synthetic_run("pretrain_1T", maxima, docs)
    pretrain_b = synthetic_run("pretrain_1T_b", maxima, docs)
    all_values = np.concatenate([
        a181._metric_block(good.state["top_doc"][slot], good.state["top_pos"][slot],
                           instruct_b.state["top_doc"][slot], instruct_b.state["top_pos"][slot])["top10_doc"]
        for slot in range(n_layers)
    ])
    assert float(np.median(all_values)) == 1.0
    result = a181._gate3({"instruct": good, "instruct_b": instruct_b,
                          "pretrain_1T": pretrain, "pretrain_1T_b": pretrain_b},
                         {layer: np.arange(n_units) for layer in range(n_layers)})
    assert result["layers"]["2"]["verdict"] == "failed"
    assert result["passing_layers"] == [0, 1]
    assert result["failing_layers"] == [2]


def test_1812_floor_below_threshold_is_not_assessable():
    assert a181.classify_1812(1.0, 0.899) == "not assessable"
    assert a181.classify_1812(0.95, 0.9) == "held"
    assert a181.classify_1812(0.89, 0.9) == "failed"


def test_1814_uses_strictly_increasing_means():
    means = [0.11, 0.23, 0.34, 0.46, 0.57, 0.69]
    assert a181.classify_strictly_increasing(means) == "held"
    assert a181.classify_strictly_increasing([0.11, 0.23, 0.34, 0.46, 0.57, 0.57]) == "failed"
