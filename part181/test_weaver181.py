"""Synthetic tests for Part 181 Amendment 3 (weaver181.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import weaver181 as w  # noqa: E402
from test_analyze181 import synthetic_run  # noqa: E402

N_LAYERS, N_UNITS = 2, 60


def _docs():
    return np.tile(np.arange(20), (N_LAYERS, N_UNITS, 1)).astype(np.int32)


def _pair(key_a, key_b, moved_units):
    """Two runs whose top-10 lists differ in the first `moved_units` units of every layer (one doc each)."""
    m = np.ones((N_LAYERS, N_UNITS), dtype=np.float32)
    a, b = _docs(), _docs()
    b[:, :moved_units, 0] += 1000
    return synthetic_run(key_a, m, a), synthetic_run(key_b, m, b)


def _ids():
    return {layer: np.arange(N_UNITS) for layer in range(N_LAYERS)}


def test_w1_held_failed_and_exact():
    # Both repeat pairs identical: both floors are 0, so the repeats are exact and 181.W1 fails by its rule.
    i_a, i_b = _pair("instruct", "instruct_b", 0)
    p_a, p_b = _pair("pretrain_1T", "pretrain_1T_b", 0)
    r = w.w1({"instruct": i_a, "instruct_b": i_b, "pretrain_1T": p_a, "pretrain_1T_b": p_b}, _ids())
    assert r["verdict"] == "failed" and "exact" in r["reason"]


def test_w1_ratio_rule(monkeypatch):
    fake = {"passing_layers": [0, 1], "layers": {
        str(l): {"repeat_pairs": {"pretrain_1T_vs_pretrain_1T_b": {"neuron_median": 0.96, "activity_mask_agreement": 1.0},
                                  "instruct_vs_instruct_b": {"neuron_median": 0.98, "activity_mask_agreement": 1.0}}}
        for l in (0, 1)}}
    monkeypatch.setattr(w.a181, "_gate3", lambda runs, ids: fake)
    r = w.w1({}, {})
    assert abs(r["floor_1T_pair"] - 0.04) < 1e-9 and abs(r["floor_instruct_pair"] - 0.02) < 1e-9
    assert r["verdict"] == "held" and abs(r["ratio"] - 2.0) < 1e-9
    fake["layers"]["0"]["repeat_pairs"]["pretrain_1T_vs_pretrain_1T_b"]["neuron_median"] = 0.97
    fake["layers"]["1"]["repeat_pairs"]["pretrain_1T_vs_pretrain_1T_b"]["neuron_median"] = 0.97
    assert w.w1({}, {})["verdict"] == "failed"


def _run_with_max(key, values):
    m = np.asarray(values, dtype=np.float32)[None, :]
    return synthetic_run(key, m, np.tile(np.arange(20), (1, m.shape[1], 1)))


def test_w2_absolute_threshold_comes_from_instruct():
    # instruct: median 10 -> tau 1. An early run with median 2: relative threshold 0.2, absolute 1.
    instruct = _run_with_max("instruct", [10.0] * 10)
    early_vals = [2.0] * 7 + [0.5] * 3       # relative: all 10 active (>=0.2); absolute: 7 active (>=1)
    runs = {"instruct": instruct}
    for k in w.PRETRAIN:
        runs[k] = _run_with_max(k, early_vals if k == "pretrain_21B" else [10.0] * 10)
    r = w.w2(runs)
    assert r["per_run"]["pretrain_21B"] == {"relative": 10, "absolute": 7, "difference": 3 / 7}
    assert r["verdict"] == "held" and r["largest_difference_run"] == "pretrain_21B"


def test_w2_failed_when_rules_agree_and_not_read_when_missing():
    runs = {"instruct": _run_with_max("instruct", [10.0] * 10)}
    for k in w.PRETRAIN:
        runs[k] = _run_with_max(k, [10.0] * 10)
    assert w.w2(runs)["verdict"] == "failed"
    del runs["pretrain_3T"]
    assert w.w2(runs)["verdict"] == "not read"


def test_keyed_runs_keeps_both_repeat_runs():
    # Amendment 4: main() keyed runs through analyze181._canonical_key, which merged instruct_b into instruct.
    class R:
        def __init__(self, key):
            self.key = key
    runs = [R("instruct"), R("instruct_b"), R("pretrain_1T"), R("pretrain_1T_b")]
    keyed = w.keyed_runs(runs)
    assert set(keyed) == {"instruct", "instruct_b", "pretrain_1T", "pretrain_1T_b"}
    assert keyed["instruct"] is runs[0] and keyed["instruct_b"] is runs[1]
    import pytest
    with pytest.raises(ValueError):
        w.keyed_runs([R("instruct"), R("instruct")])
