"""Part 190 tests on synthetic fixtures only (never the real arrays)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from part190 import analyze190 as A  # noqa: E402
from part190.synthetic190 import make_fixture, registered_from_fixture, write_fixture  # noqa: E402

SIGNAL = dict(signal=2.0, verdict_align=0.6, dim=96, seed=3)


def _data(fx):
    return A.data_from_arrays(fx["rows"], fx["ids"], fx["embeddings"], fx["prompt_phrases"], fx["real"],
                              fx["imitation"])


def _run(fx, name="A_to_B", fit="A", test="B", ranks=(1, 8), n_controls=10, doses=(0.0, 0.5, 1.0, 2.0, 4.0)):
    data = _data(fx)
    ctx = A.make_context(data, name, fit, test)
    return A.run_direction(data, ctx, ranks=ranks, doses=doses, n_controls=n_controls, n_boot=30, log=lambda m: None)


@pytest.fixture(scope="module")
def signal_runs():
    fx = make_fixture(**SIGNAL)
    return fx, {n: _run(fx, n, f, t) for n, f, t in A.DIRECTIONS}


@pytest.fixture(scope="module")
def null_runs():
    fx = make_fixture(**{**SIGNAL, "signal": 0.0})
    return fx, {n: _run(fx, n, f, t) for n, f, t in A.DIRECTIONS}


# ------------------------------------------------------------------ small pieces

def brute_auc(y, s):
    pos, neg = s[y], s[~y]
    return np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])


def test_auc64_matches_brute_force_with_ties():
    rng = np.random.default_rng(1)
    y = rng.random(60) > 0.4
    s = np.round(rng.standard_normal(60), 1)
    assert A.auc64(y, s) == pytest.approx(brute_auc(y, s), abs=1e-12)
    assert np.isnan(A.auc64(np.ones(5, bool), np.arange(5.0)))


def test_knn_tie_goes_to_class_with_nearest_member():
    # clip 0's two neighbours: one of class b (nearest) and one of class c; a 2-NN tie goes to b
    x = A._normalize_rows(np.array([[1.0, 0.0], [0.99, 0.141], [0.95, -0.312], [-1.0, 0.0], [-0.99, 0.1]]))
    labels = np.array(["a", "b", "c", "d", "d"])
    correct = A.knn_class_correct(x, labels, k=2)
    assert not correct[0]
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    assert A._top_neighbours(sim, 2)[0].tolist() == [1, 2]
    assert correct[3] and correct[4]


def test_shuffle_within_groups_keeps_each_groups_counts():
    rng = np.random.default_rng(0)
    y = np.array([1, 1, 0, 0, 0, 1, 1, 1, 0], bool)
    g = np.array(list("aaaaabbbb"))
    s = A.shuffle_within_groups(y, g, rng)
    for grp in "ab":
        assert s[g == grp].sum() == y[g == grp].sum()


def test_edit_is_identity_at_zero_and_rank_k_before_normalization():
    rng = np.random.default_rng(2)
    x = A._normalize_rows(rng.standard_normal((80, 20)))
    y = rng.random(80) > 0.5
    v = A._normalize_rows(rng.standard_normal((1, 20)))[0]
    read = A.read_directions(x, y, 4)
    assert np.allclose(read.T @ read, np.eye(4), atol=1e-9)
    for j in range(4):
        assert (x[y] @ read[:, j]).mean() >= (x[~y] @ read[:, j]).mean()
    edit = A.build_edit(x, y, read, v)
    assert np.abs(edit.apply(x, 0.0) - x).max() <= 1e-12
    for k in (1, 2, 4):
        disp = edit.truncated(k).displacement(x, 3.0)
        assert np.linalg.matrix_rank(disp, tol=1e-9) == k
    assert np.allclose(np.linalg.norm(edit.apply(x, 5.0), axis=1), 1.0)
    assert np.allclose(edit.write[:, 0], v)


def test_matched_displacement_scale():
    rng = np.random.default_rng(4)
    x = A._normalize_rows(rng.standard_normal((50, 12)))
    v = A._normalize_rows(rng.standard_normal((1, 12)))[0]
    e = A.random_edit(x, v, 2, rng)
    target = 0.37
    e2 = e.with_scale(target / e.mean_displacement(x, 1.0))
    assert e2.mean_displacement(x, 1.0) == pytest.approx(target)
    assert e2.mean_displacement(x, 4.0) == pytest.approx(4 * target)


def test_operating_dose_is_smallest_within_tolerance():
    ladder = {0.0: {"V_fit": 0.6}, 1.0: {"V_fit": 0.85}, 2.0: {"V_fit": 0.895}, 4.0: {"V_fit": 0.9},
              8.0: {"V_fit": 0.88}}
    assert A.operating_dose(ladder) == 2.0
    assert A.operating_dose(ladder, tolerance=0.0) == 4.0


# ------------------------------------------------------------------ known answers

def test_main_edit_raises_heldout_verdict_on_signal_fixture(signal_runs):
    _, runs = signal_runs
    for name, r in runs.items():
        op = r["ranks"]["1"]["operating"]
        assert op["dV_test"] >= 0.05, name
        assert op["dV_chicken"] >= 0.0, name


def test_negative_control_cannot_improve_heldout_separation(signal_runs, null_runs):
    # shuffled labels: the whole construction rerun, displacement matched; must not raise held-out separation
    for fx_runs in (signal_runs, null_runs):
        _, runs = fx_runs
        changes = {n: r["ranks"]["1"]["shuffled_dV_test_at_operating"] for n, r in runs.items()}
        for n, ch in changes.items():
            assert np.median(ch) <= A.GATE3_MEDIAN_MAX, n
        assert A.gate3_rule(changes)["passed"]
    # with no real-vs-imitation information at all, not even the main edit can raise held-out separation
    _, runs = null_runs
    for name, r in runs.items():
        assert r["ranks"]["1"]["operating"]["dV_test"] < 0.05, name


def test_gate3_fails_on_a_leaky_negative_control(signal_runs, monkeypatch):
    # pre-fault: the same fixture passes (previous test); fault: the "shuffle" leaves the labels in place
    fx, runs = signal_runs
    good = {n: r["ranks"]["1"]["shuffled_dV_test_at_operating"] for n, r in runs.items()}
    assert A.gate3_rule(good)["passed"]
    monkeypatch.setattr(A, "shuffle_within_groups", lambda y, g, rng: np.asarray(y, bool).copy())
    leaky = _run(fx, "A_to_B", "A", "B", ranks=(1, 8), n_controls=3)
    verdict = A.gate3_rule({"A_to_B": leaky["ranks"]["1"]["shuffled_dV_test_at_operating"]})
    assert not verdict["passed"]
    assert any("median" in reason for reason in verdict["reasons"])


def test_gate3_self_test(signal_runs, null_runs):
    _, runs = signal_runs
    shown = A.gate3_self_test({n: r["ranks"]["1"]["operating"]["dV_test"] for n, r in runs.items()})
    assert shown["shown"] is True and shown["rule_failed_on_main_edit"] is True
    _, nruns = null_runs
    assert A.gate3_self_test({n: r["ranks"]["1"]["operating"]["dV_test"] for n, r in nruns.items()})["shown"] is False


def test_random_control_does_not_improve(signal_runs):
    _, runs = signal_runs
    for name, r in runs.items():
        assert r["ranks"]["1"]["controls_at_operating"]["random"]["dV_test"]["median"] <= 0.02, name


def test_baseline_reproduces_registered_definitions(signal_runs):
    fx, runs = signal_runs
    reg = registered_from_fixture(fx)
    for name, r in runs.items():
        assert r["baseline"]["V_test"] == pytest.approx(reg["directions"][name]["model_class_informed"]["auc"])
        assert r["probe_heldout_auc"] == pytest.approx(reg["directions"][name]["output_embedding_probe"]["auc"])
        assert r["identity_max_abs_diff"] <= 1e-9


def test_intervals_bracket_point_estimates(signal_runs):
    _, runs = signal_runs
    for r in runs.values():
        iv = r["ranks"]["1"]["operating_intervals"]
        assert iv["dV_test"]["low"] <= r["ranks"]["1"]["operating"]["dV_test"] <= iv["dV_test"]["high"]


# ------------------------------------------------------------------ gates and verdicts

def test_gate1_catches_reordered_ids_and_fold_counts():
    fx = make_fixture(dim=16, seed=5)
    data = _data(fx)
    assert A.gate1(data, fx["ids"], fx["folds"], expected_rows=None)["passed"]
    assert not A.gate1(data, fx["ids"][::-1].copy(), fx["folds"], expected_rows=None)["passed"]
    bad = json.loads(json.dumps(fx["folds"]))
    bad["counts_by_analysis_set"]["A"]["R"] += 1
    assert not A.gate1(data, fx["ids"], bad, expected_rows=None)["passed"]
    assert not A.gate1(data, fx["ids"], fx["folds"], expected_rows=3138)["passed"]


def test_gate2_tolerance():
    reg = {"A_to_B": {"model_class_informed": 0.631, "output_embedding_probe": 0.918}}
    ok = A.gate2({"A_to_B": 0.0}, {"A_to_B": {"model_class_informed": 0.634, "output_embedding_probe": 0.918}}, reg)
    assert ok["passed"]
    assert not A.gate2({"A_to_B": 0.0}, {"A_to_B": {"model_class_informed": 0.640,
                                                     "output_embedding_probe": 0.918}}, reg)["passed"]
    assert not A.gate2({"A_to_B": 1e-6}, {"A_to_B": {"model_class_informed": 0.631,
                                                      "output_embedding_probe": 0.918}}, reg)["passed"]
    assert not A.gate2({"A_to_B": 0.0}, {"A_to_B": {"model_class_informed": 0.631,
                                                     "output_embedding_probe": 0.918}}, {})["passed"]


def test_gate4():
    assert A.gate4({"A_to_B": 0.31, "B_to_A": 0.5})["passed"]
    assert not A.gate4({"A_to_B": 0.29, "B_to_A": 0.5})["passed"]
    assert not A.gate4({"A_to_B": float("nan")})["passed"]


def _values(**over):
    base = {key: {"A_to_B": 0.0, "B_to_A": 0.0} for key, _ in A.PREDICTION_RULES.values()}
    for key, v in over.items():
        base[key] = {"A_to_B": v[0], "B_to_A": v[1]}
    return base


def test_prediction_thresholds_and_signs():
    s = A.score_predictions(_values(rank1_dV_test=(0.10, 0.2)), True, True)
    assert s["190.1"]["verdict"] == "held"
    assert A.score_predictions(_values(rank1_dV_test=(0.0999, 0.2)), True, True)["190.1"]["verdict"] == "failed"
    # 190.9 is a drop: a rise of 5 points holds, a drop of 2.0 fails
    assert A.score_predictions(_values(rank1_C5_change_points=(5.0, -1.9)), True, True)["190.9"]["verdict"] == "held"
    assert A.score_predictions(_values(rank1_C5_change_points=(0.0, -2.0)), True, True)["190.9"]["verdict"] == "failed"
    # 190.6 is two-sided
    assert A.score_predictions(_values(rank1_C1_change_points=(1.0, 0.0)), True, True)["190.6"]["verdict"] == "failed"
    assert A.score_predictions(_values(rank1_C1_change_points=(-0.9, 0.9)), True, True)["190.6"]["verdict"] == "held"
    # 190.4: a control that improves fails it
    assert A.score_predictions(_values(shuffled_rank1_median_dV_test=(0.021, 0.0)), True,
                               True)["190.4"]["verdict"] == "failed"
    assert A.score_predictions(_values(rank1_dV_test=(float("nan"), 0.2)), True, True)["190.1"]["verdict"] == "not read"


def test_unreadable_run_reads_nothing():
    s = A.score_predictions(_values(rank1_dV_test=(0.5, 0.5)), False, True)
    assert all(p["verdict"] == "not read" for p in s.values())
    s = A.score_predictions(_values(), True, False)
    assert s["190.6"]["verdict"] == "not read" and s["190.7"]["verdict"] == "not read"
    assert s["190.1"]["verdict"] != "not read"
    s = A.score_predictions(_values(), True, True, rank8_readable=False)
    assert s["190.3"]["verdict"] == "not read" and s["190.1"]["verdict"] != "not read"


def test_headline_needs_controls_beaten(signal_runs):
    _, runs = signal_runs
    verdicts = {pid: {"verdict": "held"} for pid in A.PREDICTION_RULES}
    assert A.headline(True, verdicts, runs)["permitted"]
    assert not A.headline(False, verdicts, runs)["permitted"]
    rigged = json.loads(json.dumps(runs, default=float))
    rigged["A_to_B"]["ranks"]["1"]["controls_at_operating"]["random"]["dV_test"]["p95"] = 1.0
    assert not A.headline(True, verdicts, rigged)["permitted"]
    costly = dict(verdicts, **{"190.9": {"verdict": "failed"}})
    h = A.headline(True, costly, runs)
    assert not h["permitted"] and h["reading"] == "separation at a cost"


def test_analyze_refuses_on_gate1_failure():
    fx = make_fixture(dim=16, seed=6)
    out = A.analyze(_data(fx), fx["ids"][::-1].copy(), fx["folds"], {}, expected_rows=None, log=lambda m: None)
    assert out["readable"] is False and "directions" not in out


# ------------------------------------------------------------------ the real entry path

def test_runner_end_to_end_under_python_isolated(tmp_path):
    fx = make_fixture(**{**SIGNAL, "dim": 48})
    paths = write_fixture(fx, tmp_path / "fx", registered_from_fixture(fx))
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, "-I", str(ROOT / "part190/run190.py"), "--emb", str(paths["emb"]),
                           "--manifest", str(paths["manifest"]), "--folds", str(paths["folds"]),
                           "--results188", str(paths["results188"]), "--out", str(out),
                           "--expected-rows", str(len(fx["rows"])), "--ranks", "1,8", "--doses", "0,1,4",
                           "--controls", "3", "--boot", "20"], capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode in (0, 3), proc.stderr
    result = json.loads((out / "results.json").read_text())
    assert set(result["predictions"]) == {f"190.{i}" for i in range(1, 13)}
    assert result["gates"]["gate1"]["passed"] and result["gates"]["gate2"]["passed"]
    assert "not the registered run" in proc.stdout
    for line in proc.stdout.splitlines():  # the runner prints verdicts and timings, never a value array
        assert "[" not in line or line.startswith("[")
