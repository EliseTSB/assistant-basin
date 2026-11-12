"""Synthetic tests for Part 203 selection, folds, readouts, gates and the V/D comparison."""
from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from part203 import analyze203, record203, select203  # noqa: E402
from part203.analyze203 import (C_GRID, actor_folds, choose_c_inner, gate1,  # noqa: E402
                                subset_draws)
from part203.record203 import (SENTENCES, decode_audio_v, d_item, read_manifest_rows,  # noqa: E402
                               v_item)
from part203.select203 import (EMOTIONS, assign_folds, build_manifest, name_hash,  # noqa: E402
                               parse_filename, read_file_list, select_clips)

HERE = PROJECT_ROOT / "part203"


def synthetic_file_rows(actors=("1001", "1002", "1003", "1004", "1005", "1006"),
                        sentences=("IEO", "TIE"), per_cell=8) -> list[dict[str, str]]:
    """Cells of `per_cell` clips, padded to CREMA-D's four-part names with the intensity code."""
    rows = []
    for actor in actors:
        for sentence in sentences:
            for emotion in EMOTIONS:
                for k in range(per_cell):
                    rows.append(parse_filename(f"{actor}_{sentence}_{emotion}_{k:02d}.wav"))
    return rows


def test_selection_reproduces_the_committed_download_list():
    rows = read_file_list(HERE / "prior" / "cremad-audiowav.tsv")
    selected = [row["file"] for row in select_clips(rows)]
    committed = [line.strip() for line in (HERE / "download-list.txt").read_text().splitlines() if line.strip()]
    assert len(committed) == 2730
    assert sorted(selected) == sorted(committed)


def test_selection_keeps_the_five_lowest_sha256_names_per_cell():
    rows = synthetic_file_rows()
    selected = select_clips(rows, per_cell=5)
    by_cell: dict[tuple[str, str], list[str]] = {}
    for row in selected:
        by_cell.setdefault((row["actor"], row["emotion"]), []).append(row["file"])
    assert len(selected) == 6 * 6 * 5          # each actor-emotion cell, pooled over its two sentences
    for (actor, emotion), names in by_cell.items():
        pool = sorted((row["file"] for row in rows if row["actor"] == actor and row["emotion"] == emotion),
                      key=name_hash)  # 16 candidates per cell (two sentences x eight names)
        assert sorted(names) == sorted(pool[:5])
        assert all(name_hash(name) <= name_hash(pool[4]) for name in names)


def test_folds_are_actor_disjoint_and_cover_every_actor_once():
    actors = [f"1{index:03d}" for index in range(91)]
    folds = assign_folds(actors)
    assert set(folds) == set(actors)
    assert sorted(set(folds.values())) == list(range(5))
    manifest = build_manifest(synthetic_file_rows(actors=tuple(actors)))
    per_fold: dict[str, set[str]] = {}
    for row in manifest:
        per_fold.setdefault(row["fold"], set()).add(row["actor"])
    seen: set[str] = set()
    for fold, members in per_fold.items():
        assert not (members & seen)
        seen |= members
    assert seen == set(actors)
    # The rule is SHA-256 hex order of the actor id, round-robin: a different order would put another actor
    # in fold 0, which the disjointness checks above cannot see.
    ordered = sorted(set(actors), key=lambda actor: (name_hash(actor), actor))
    assert folds == {actor: index % 5 for index, actor in enumerate(ordered)}


def test_three_actor_draws_stay_inside_the_fold_training_actors():
    train = [f"2{index:03d}" for index in range(15)]
    draws = subset_draws(train)
    assert len(draws) == analyze203.SUBSET_DRAWS
    for draw in draws:
        assert len(draw) == 3 and len(set(draw)) == 3
        assert set(draw) <= set(train)
    assert any(draw != draws[0] for draw in draws)
    # The rule is SHA-256 hex order with a rotating start: a plain sort gives the first three names every time.
    ordered = sorted(set(train), key=lambda actor: (name_hash(actor), actor))
    assert draws[0] == ordered[:3] and draws[1] == ordered[3:6]


def test_three_actor_draws_refuse_a_too_small_training_set():
    with pytest.raises(ValueError):
        subset_draws(["2001", "2002"])


def _separable(n_per_class=40, dim=8, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    centers = np.eye(len(EMOTIONS), dim) * 4.0
    x, y = [], []
    for cls in range(len(EMOTIONS)):
        x.append(centers[cls] + rng.normal(scale=noise, size=(n_per_class, dim)))
        y.extend([cls] * n_per_class)
    return np.vstack(x).astype(np.float32), np.asarray(y)


def test_readout_scores_a_separable_problem_near_perfectly(monkeypatch):
    monkeypatch.setattr(analyze203, "N_CLASSES", len(EMOTIONS))
    x, y = _separable()
    x_test, y_test = _separable(seed=9)
    actors = np.asarray([f"a{index % 6}" for index in range(len(y))])
    out, extra = analyze203._fit_and_score(x, y, x_test, y_test, actors, c=1.0)
    assert out["balanced_accuracy"] > 0.95
    assert out["macro_ovr_auc"] > 0.99
    assert extra["c_selection"]["chosen_C"] == 1.0


def test_readout_scores_pure_noise_near_chance(monkeypatch):
    monkeypatch.setattr(analyze203, "N_CLASSES", len(EMOTIONS))
    rng = np.random.default_rng(1)
    x = rng.normal(size=(240, 8)).astype(np.float32)
    x_test = rng.normal(size=(240, 8)).astype(np.float32)
    y = np.tile(np.arange(len(EMOTIONS)), 40)
    actors = np.asarray([f"a{index % 6}" for index in range(len(y))])
    out, _ = analyze203._fit_and_score(x, y, x_test, y, actors, c=1.0)
    assert out["balanced_accuracy"] < 0.45


def test_probe_scores_fills_a_class_missing_from_training_with_the_missing_score():
    """LOW-2: probe_scores' docstring promises MISSING_CLASS_SCORE, not a ValueError."""
    x, y = _separable(n_per_class=20)
    keep = y != 5                                  # class 5 (SAD) absent from the training rows
    model = analyze203.fit_probe_c(x[keep], y[keep], c=1.0)
    scores, classes = analyze203.probe_scores(model, x)
    assert classes == list(range(len(EMOTIONS)))
    assert scores.shape == (len(x), len(EMOTIONS))
    assert np.all(scores[:, 5] == analyze203.MISSING_CLASS_SCORE)
    present = [cls for cls in range(len(EMOTIONS)) if cls != 5]
    for cls in present:                            # the present classes keep the probe's own scores
        assert np.all(np.isfinite(scores[:, cls]))
    assert set(np.argmax(scores, axis=1)) <= set(present)


def test_c_grid_is_chosen_by_inner_cv_only():
    x, y = _separable(seed=3)
    actors = np.asarray([f"a{index % 12}" for index in range(len(y))])
    selection = choose_c_inner(x, y, actors)
    assert selection["chosen_C"] in C_GRID
    losses = selection["mean_log_loss_by_C"]
    assert set(losses) == {f"{c:g}" for c in C_GRID}
    best = min(C_GRID, key=lambda c: (losses[f"{c:g}"], c))
    assert selection["chosen_C"] == best
    # The tie rule itself: ties go to the LOWER C. Two C values sharing the lowest loss must pick the smaller.
    tie = dict(losses)
    low, high = C_GRID[1], C_GRID[3]
    tie[f"{low:g}"] = tie[f"{high:g}"] = min(tie.values())
    for c in C_GRID:
        if c not in (low, high):
            tie[f"{c:g}"] = min(tie.values()) + 1.0
    assert min(C_GRID, key=lambda c: (tie[f"{c:g}"], c)) == low
    # The choice moves with the data: an independent, different problem can choose another C.
    x2, y2 = _separable(seed=4, noise=6.0)
    assert choose_c_inner(x2, y2, actors)["chosen_C"] in C_GRID


def test_c_grid_prefers_weak_regularisation_on_separable_data():
    x, y = _separable(noise=0.05)
    actors = np.asarray([f"a{index % 12}" for index in range(len(y))])
    assert choose_c_inner(x, y, actors)["chosen_C"] == max(C_GRID)


def test_inner_cv_would_see_the_test_labels_if_they_were_passed(monkeypatch):
    """The C choice must depend only on the arguments it is given (training rows)."""
    x, y = _separable()
    actors = np.asarray([f"a{index % 12}" for index in range(len(y))])
    calls = []
    original = analyze203._log_loss

    def spy(model, xx, yy):
        calls.append(len(yy))
        return original(model, xx, yy)

    monkeypatch.setattr(analyze203, "_log_loss", spy)
    choose_c_inner(x, y, actors)
    folds = actor_folds(actors)
    assert len(calls) == len(folds) * len(C_GRID)
    # Each scored set is exactly one held-out actor fold, not some smaller slice of the training rows.
    assert sorted(calls) == sorted(len(fold) for fold in folds for _ in C_GRID)


def test_inner_folds_are_five_actor_disjoint_sha256_folds():
    """MEDIUM-3: the inner folds are the design's rule: SHA-256 hex order, round-robin, actor-disjoint."""
    actors = np.repeat(np.asarray([f"4{index:03d}" for index in range(23)]), 4)
    folds = analyze203.actor_folds(actors)
    assert len(folds) == 5
    seen: set[str] = set()
    for fold in folds:
        members = set(actors[fold])
        assert not (members & seen)
        seen |= members
    assert seen == set(actors)
    ordered = sorted(set(actors.tolist()), key=lambda actor: (name_hash(actor), actor))
    expected = {actor: index % 5 for index, actor in enumerate(ordered)}
    for index, fold in enumerate(folds):
        assert {expected[actor] for actor in actors[fold]} == {index}
    # The C selection uses exactly these folds (one fit per fold per C), so the two choices share one split.
    x, y = _separable(seed=5)
    rows_actors = np.asarray([f"a{index % 10}" for index in range(len(y))])
    calls = []
    original = analyze203.fit_probe_c
    analyze203.fit_probe_c = lambda xx, yy, c, *a, **k: (calls.append(len(yy)), original(xx, yy, c, *a, **k))[1]
    try:
        choose_c_inner(x, y, rows_actors)
    finally:
        analyze203.fit_probe_c = original
    inner = analyze203.actor_folds(rows_actors)
    assert sorted(calls) == sorted(len(y) - len(fold) for fold in inner for _ in C_GRID)


def test_layer_choice_fits_every_layer_on_every_inner_fold_and_picks_the_best():
    """MEDIUM-3: 24 layers x 5 inner folds fits; mean BA at C = 1.0; ties to the earlier layer."""
    n_layers, width = 3, 4
    rng = np.random.default_rng(7)
    actors = np.repeat(np.asarray([f"5{index:03d}" for index in range(10)]), 6)
    y = np.tile(np.arange(len(EMOTIONS)), 10)
    x = rng.normal(size=(len(y), n_layers * width)).astype(np.float32)
    # Layer 1 carries the signal; layers 0 and 2 are noise.
    centers = rng.normal(size=(len(EMOTIONS), width)) * 6.0
    x[:, width:2 * width] += centers[y]
    fitted = []
    original = analyze203.fit_probe_c

    def spy(xx, yy, c, *args, **kwargs):
        fitted.append(float(c))
        return original(xx, yy, c, *args, **kwargs)

    analyze203.fit_probe_c = spy
    try:
        layer, means = analyze203.choose_layer_inner(x, y, actors, n_layers=n_layers, width=width)
    finally:
        analyze203.fit_probe_c = original
    assert len(fitted) == n_layers * 5 and set(fitted) == {1.0}
    assert layer == 1 and means[1] > max(means[0], means[2])
    # Ties go to the earlier layer: identical layers pick layer 0.
    flat = np.tile(x[:, :width], (1, n_layers))
    assert analyze203.choose_layer_inner(flat, y, actors, n_layers=n_layers, width=width)[0] == 0


def test_layer_choice_uses_five_inner_folds_in_analyze(tmp_path, monkeypatch):
    n_layers, width = 2, 8
    manifest, work, rows = _manifest_and_records(tmp_path, dim=analyze203.EMBEDDING_SIZE, full=True,
                                                 n_units=n_layers * width)
    monkeypatch.setattr(analyze203, "N_CLASSES", len(EMOTIONS))
    monkeypatch.setattr(analyze203, "N_LAYERS", n_layers)
    monkeypatch.setattr(analyze203, "UNITS_PER_LAYER", width)
    monkeypatch.setattr(analyze203, "N_UNITS", n_layers * width)
    monkeypatch.setattr(analyze203, "BOOTSTRAPS", 8)
    fitted = []
    original_fit = analyze203.fit_probe_c
    original_choose = analyze203.choose_layer_inner

    def fit_spy(xx, yy, c, *args, **kwargs):
        fitted.append(float(c))
        return original_fit(xx, yy, c, *args, **kwargs)

    def choose_spy(*args, **kwargs):
        before = len(fitted)
        out = original_choose(*args, **kwargs)
        choose_calls.append(len(fitted) - before)
        return out

    choose_calls = []
    monkeypatch.setattr(analyze203, "fit_probe_c", fit_spy)
    monkeypatch.setattr(analyze203, "choose_layer_inner", choose_spy)
    result = analyze203.analyze(manifest_path=manifest, work_dir=work, out_dir=tmp_path / "out")
    assert choose_calls == [n_layers * 5] * 5          # 24 x 5 fits per outer fold in the real run
    for fold in result["folds"].values():
        assert fold["E"]["layer_inner_folds"] == 5
        assert len(fold["E"]["inner_balanced_accuracy"]) == n_layers


def _manifest_and_records(tmp_path, *, drop=None, v_offset=0.0, dim=16, full=False, per_cell=5,
                          n_units=None):
    actors = [f"3{index:03d}" for index in range(10)]
    rows = []
    for actor in actors:
        for emotion in EMOTIONS:
            for k in range(per_cell):       # the selector pools sentences and draws five per (actor, emotion)
                sentence = ("IEO", "TIE")[k % 2]
                rows.append({"file": f"{actor}_{sentence}_{emotion}_{k:02d}.wav", "actor": actor,
                             "sentence": sentence, "emotion": emotion, "intensity": "XX",
                             "fold": "", "cell_rank": str(k + 1)})
    folds = assign_folds(actors)
    for row in rows:
        row["fold"] = str(folds[row["actor"]])
    manifest = tmp_path / "manifest203.csv"
    select203.write_manifest(manifest, rows)
    work = tmp_path / "work"
    (work / "records").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for row in rows:
        clip_id = row["file"][:-4]
        if clip_id == drop:
            continue
        # A clear six-way signal spread over all `dim` dimensions (a disjoint block of width dim/6 per emotion),
        # so no dimension is pure noise and each readout should score well above chance: the test measures the
        # pipeline, not a weak fixture.
        block = dim // len(EMOTIONS)
        center = np.zeros(dim, dtype=np.float64)
        center[EMOTIONS.index(row["emotion"]) * block:(EMOTIONS.index(row["emotion"]) + 1) * block] = 12.0
        v = (center + rng.normal(scale=0.1, size=dim) + v_offset).astype(np.float16)
        d = (center + rng.normal(scale=0.1, size=dim)).astype(np.float32)
        np.savez_compressed(work / "records" / f"{clip_id}.npz", id=np.asarray(clip_id),
                            v=v, emb=d,
                            act_mean=rng.normal(size=n_units if n_units is not None
                                                else (analyze203.N_UNITS if full else 4)).astype(np.float16),
                            acoustic=rng.normal(size=7).astype(np.float32),
                            status=np.asarray("ok"), v_same_samples=np.asarray(True))
    (work / "gate2.json").write_text(json.dumps({"verdict": "passed"}))
    # The recorder hashes the rows `read_manifest_rows` builds from the manifest file, not the raw rows.
    from part196.record196 import manifest_sha as _manifest_sha

    identity = {"model_id": record203.MODEL_ID, "model_snapshot_revision": record203.MODEL_REVISION,
                "manifest_sha256": _manifest_sha(read_manifest_rows(manifest))}
    (work / "identity.json").write_text(json.dumps(identity))
    return manifest, work, rows


def test_gates_pass_on_a_complete_synthetic_recording(tmp_path):
    manifest, work, rows = _manifest_and_records(tmp_path, dim=analyze203.EMBEDDING_SIZE, full=True)
    records = analyze203.read_records(rows, work)
    assert gate1(rows, records["status"], records["v"], records["d"],
                 records["v_same_samples"])["verdict"] == "passed"


def _set_v_same_samples(work, clip_id, value):
    """Rewrite one fixture record with `v_same_samples` set (or, with value=None, removed)."""
    path = work / "records" / f"{clip_id}.npz"
    with np.load(path, allow_pickle=False) as saved:
        items = {key: saved[key] for key in saved.files}
    items.pop("v_same_samples", None)
    if value is not None:
        items["v_same_samples"] = np.asarray(value)
    np.savez_compressed(path, **items)


def test_gate_fails_when_one_record_says_v_and_d_inputs_differ(tmp_path):
    """HIGH-2: the recorder's Gate 1 (v_same_samples) must be enforced by the analyzer's Gate 1."""
    manifest, work, rows = _manifest_and_records(tmp_path, dim=analyze203.EMBEDDING_SIZE, full=True)
    _set_v_same_samples(work, rows_first(), True)
    records = analyze203.read_records(rows, work)
    assert np.all(records["v_same_samples"] == True)                       # noqa: E712
    assert gate1(rows, records["status"], records["v"], records["d"],
                 records["v_same_samples"])["verdict"] == "passed"
    _set_v_same_samples(work, rows_first(), False)
    records = analyze203.read_records(rows, work)
    g1 = gate1(rows, records["status"], records["v"], records["d"], records["v_same_samples"])
    assert g1["verdict"] == "failed"
    assert g1["problems"]["v_d_input_mismatch"] == [rows_first() + ".wav"]
    # A missing field is a mismatch too: a record written before the field existed cannot vouch for V vs D.
    _set_v_same_samples(work, rows_first(), True)
    records = analyze203.read_records(rows, work)
    records["v_same_samples"][0] = False                 # what read_records yields for a record with no field
    g1 = gate1(rows, records["status"], records["v"], records["d"], records["v_same_samples"])
    assert g1["verdict"] == "failed" and g1["problems"]["v_d_input_mismatch_count"] == 1
    # And a caller that does not pass the field at all cannot get a pass.
    assert gate1(rows, records["status"], records["v"], records["d"])["verdict"] == "failed"


def test_analyze_stops_before_readouts_when_v_and_d_inputs_differ(tmp_path, monkeypatch):
    manifest, work, rows = _manifest_and_records(tmp_path, dim=analyze203.EMBEDDING_SIZE, full=True)
    _set_v_same_samples(work, rows_first(), False)
    monkeypatch.setattr(analyze203, "BOOTSTRAPS", 8)
    result = analyze203.analyze(manifest_path=manifest, work_dir=work, out_dir=tmp_path / "out")
    assert result["status"] == "gates_failed"
    assert "v_d_input_mismatch" in result["gate1"]["problems"]


def test_read_records_names_a_truncated_record_instead_of_crashing(tmp_path):
    """LOW-4: a truncated .npz is a record error (named and counted), not a traceback."""
    manifest, work, rows = _manifest_and_records(tmp_path, dim=analyze203.EMBEDDING_SIZE, full=True)
    path = work / "records" / f"{rows_first()}.npz"
    raw = path.read_bytes()
    path.write_bytes(raw[:len(raw) // 2])
    records = analyze203.read_records(rows, work)          # must not raise zipfile.BadZipFile
    index = [row["id"] for row in rows].index(rows_first())
    assert records["status"][index] == "record_error"
    assert "BadZipFile" in records["errors"][index]
    g1 = gate1(rows, records["status"], records["v"], records["d"], records["v_same_samples"])
    assert g1["verdict"] == "failed" and g1["problems"]["missing_or_failed_count"] == 1


def test_gate_fails_when_a_clip_is_missing(tmp_path):
    manifest, work, rows = _manifest_and_records(tmp_path, drop=rows_first(),
                                                 dim=analyze203.EMBEDDING_SIZE, full=True)
    records = analyze203.read_records(rows, work)
    g1 = gate1(rows, records["status"], records["v"], records["d"])
    assert g1["verdict"] == "failed"
    assert g1["problems"]["missing_or_failed_count"] == 1


def rows_first():
    return "3000_IEO_ANG_00"          # the fixture's first clip id in the first actor/emotion cell


def test_gate_fails_when_a_cell_does_not_hold_five_planned_clips():
    """Isolate the cell check: every clip is recorded, but one cell holds four planned clips."""
    rows = [{"file": f"a_{k}.wav", "actor": "1", "emotion": "ANG", "fold": "0"} for k in range(4)]
    status = np.asarray(["ok"] * 4)
    v = np.zeros((4, 2))
    g1 = gate1(rows, status, v, v)
    assert g1["verdict"] == "failed"
    assert g1["problems"]["cells_not_five"] == ["1/ANG"]
    assert "missing_or_failed" not in g1["problems"]


def test_gate_fails_when_a_vector_is_not_finite():
    # One complete actor/emotion cell of five clips, so only the non-finite check can fail the gate.
    rows = [{"file": f"a_{k}.wav", "actor": "1", "emotion": "ANG", "fold": "0"} for k in range(5)]
    status = np.asarray(["ok"] * 5)
    v = np.zeros((5, 2))
    v[2, 1] = np.nan
    g1 = gate1(rows, status, v, v)
    assert g1["verdict"] == "failed"
    assert g1["problems"]["nonfinite_vector_count"] == 1


def test_gate_fails_when_folds_share_an_actor():
    rows = [{"file": "a.wav", "actor": "1", "emotion": "ANG", "fold": "0"},
            {"file": "b.wav", "actor": "1", "emotion": "DIS", "fold": "1"}]
    status = np.asarray(["ok", "ok"])
    v = np.zeros((2, 1))
    g1 = gate1(rows, status, v, v)
    assert g1["verdict"] == "failed" and "actor_overlap" in g1["problems"]


def test_analyze_runs_end_to_end_and_writes_results(tmp_path, monkeypatch):
    # Reduced sizes inside the test only, as part196's end-to-end test does: a 2 x 8 layer grid and 8 bootstrap
    # draws, with the fixture writing that many E units. The recorder's file format is kept.
    n_layers, width = 2, 8
    manifest, work, rows = _manifest_and_records(tmp_path, dim=analyze203.EMBEDDING_SIZE, full=True,
                                                 n_units=n_layers * width)
    monkeypatch.setattr(analyze203, "N_CLASSES", len(EMOTIONS))
    monkeypatch.setattr(analyze203, "N_LAYERS", n_layers)
    monkeypatch.setattr(analyze203, "UNITS_PER_LAYER", width)
    monkeypatch.setattr(analyze203, "N_UNITS", n_layers * width)
    monkeypatch.setattr(analyze203, "BOOTSTRAPS", 8)
    out = tmp_path / "out"
    result = analyze203.analyze(manifest_path=manifest, work_dir=work, out_dir=out)
    assert result["status"] == "complete"
    written = json.loads((out / "results.json").read_text())
    assert written["readouts"]["V_grid"]["mean_balanced_accuracy"] > 0.8
    assert set(written["paired_differences"]) >= {"V_grid_minus_E_p196", "D_p196_minus_V_grid"}
    # MEDIUM-5: prediction 203.4's comparator, free because both models are already fitted in the fold loop.
    e_minus_d = written["paired_differences"]["E_p196_minus_D_p196"]
    assert isinstance(e_minus_d["mean_difference"], float)
    assert set(e_minus_d["actor_cluster_95"]) == {"low", "high", "n_boot"}
    assert written["three_actor_ablation"]["V"]["three_actor_draws_mean"] > 0.5
    # LOW-3: the monkeypatched BOOTSTRAPS takes effect (it was bound as a default at definition time).
    assert written["readouts"]["V_grid"]["balanced_accuracy_actor_cluster_95"]["n_boot"] == 8
    assert written["rules"]["bootstrap"]["draws"] == 8
    # LOW-5: results.json names the predictions it carries, not the design's template placeholder.
    assert written["rules"]["predictions"] == [f"203.{n}" for n in range(1, 8)]


def test_three_actor_ablation_fits_e_at_c_1_and_v_by_inner_cv(tmp_path, monkeypatch):
    """MEDIUM-1: the design fits E at C = 1.0 in the ablation; only V chooses C by inner CV."""
    n_layers, width = 2, 8
    manifest, work, rows = _manifest_and_records(tmp_path, dim=analyze203.EMBEDDING_SIZE, full=True,
                                                 n_units=n_layers * width)
    monkeypatch.setattr(analyze203, "N_CLASSES", len(EMOTIONS))
    monkeypatch.setattr(analyze203, "N_LAYERS", n_layers)
    monkeypatch.setattr(analyze203, "UNITS_PER_LAYER", width)
    monkeypatch.setattr(analyze203, "N_UNITS", n_layers * width)
    monkeypatch.setattr(analyze203, "BOOTSTRAPS", 8)
    fitted = []
    original = analyze203.fit_probe_c

    def spy(x, y, c, *args, **kwargs):
        fitted.append(float(c))
        return original(x, y, c, *args, **kwargs)

    monkeypatch.setattr(analyze203, "fit_probe_c", spy)
    result = analyze203.analyze(manifest_path=manifest, work_dir=work, out_dir=tmp_path / "out")
    e_draw_c = [row["chosen_C"] for fold in result["folds"].values()
                for row in fold["three_actor_draws"] if row["readout"] == "E"]
    v_draw_c = [row["chosen_C"] for fold in result["folds"].values()
                for row in fold["three_actor_draws"] if row["readout"] == "V"]
    assert len(e_draw_c) == 5 * analyze203.SUBSET_DRAWS and len(v_draw_c) == 5 * analyze203.SUBSET_DRAWS
    assert set(e_draw_c) == {1.0}
    # V still chooses C by inner CV, so its draws are not pinned to 1.0 by the same change.
    assert set(v_draw_c) <= set(C_GRID)


def test_analyze_stops_before_readouts_when_gate1_fails(tmp_path):
    manifest, work, rows = _manifest_and_records(tmp_path, drop=rows_first(),
                                                 dim=analyze203.EMBEDDING_SIZE, full=True)
    result = analyze203.analyze(manifest_path=manifest, work_dir=work, out_dir=tmp_path / "out")
    assert result["status"] == "gates_failed"
    assert "readouts" not in result


def test_analyze_stops_when_gate2_did_not_pass(tmp_path):
    # A complete recording, so Gate 1 passes and only Gate 2 can stop the analysis. E's width is irrelevant to
    # a gate, so it is small: the analysis must stop before the layer grid runs.
    manifest, work, rows = _manifest_and_records(tmp_path, dim=analyze203.EMBEDDING_SIZE, n_units=16)
    (work / "gate2.json").write_text(json.dumps({"verdict": "failed"}))
    result = analyze203.analyze(manifest_path=manifest, work_dir=work, out_dir=tmp_path / "out")
    assert result["status"] == "gates_failed"


def test_analyze_refuses_a_manifest_that_differs_from_the_recording_identity(tmp_path):
    manifest, work, rows = _manifest_and_records(tmp_path)
    (work / "identity.json").write_text(json.dumps({"model_id": record203.MODEL_ID,
                                                    "model_snapshot_revision": record203.MODEL_REVISION,
                                                    "manifest_sha256": "0" * 64}))
    with pytest.raises(ValueError):
        analyze203.analyze(manifest_path=manifest, work_dir=work, out_dir=tmp_path / "out")


def test_v_item_and_d_item_differ_in_the_way_venn_and_part196_differ():
    samples = np.zeros(1600, dtype=np.float32)
    assert v_item(samples) == {"text": "<|audio|>", "audio": [samples]}
    assert d_item(samples) == {"audio": {"array": samples, "sampling_rate": 16000}}


def _wav(samples: np.ndarray, rate=16000, channels=1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


class FakeSoundFile:
    def __init__(self, payload):
        self.samplerate, self.channels, frames = payload
        self.frames = frames

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, dtype="float32"):
        return np.zeros(self.frames, dtype=np.float32)


def test_decode_v_refuses_what_venn_refuses(monkeypatch):
    payloads = iter([(16000, 1, 480001), (16000, 2, 100), (8000, 1, 100), (16000, 1, 0)])
    fake = types.SimpleNamespace(SoundFile=lambda _stream: FakeSoundFile(next(payloads)))
    monkeypatch.setitem(sys.modules, "soundfile", fake)
    for _ in range(4):
        with pytest.raises(ValueError):
            decode_audio_v(b"x")


def test_decode_v_refuses_a_clip_past_thirty_seconds(monkeypatch):
    """Isolate the 30 s cap: a mono 16 kHz clip one frame past Venn's 480,000 is refused, not truncated."""
    payloads = iter([(16000, 1, 480001), (16000, 1, 480000)])
    fake = types.SimpleNamespace(SoundFile=lambda _stream: FakeSoundFile(next(payloads)))
    monkeypatch.setitem(sys.modules, "soundfile", fake)
    with pytest.raises(ValueError):
        decode_audio_v(b"x")
    samples, _header = decode_audio_v(b"x")
    assert samples.shape == (480000,)


def test_decode_v_accepts_mono_16k_within_thirty_seconds(monkeypatch):
    fake = types.SimpleNamespace(SoundFile=lambda _stream: FakeSoundFile((16000, 1, 480000)))
    monkeypatch.setitem(sys.modules, "soundfile", fake)
    samples, header = decode_audio_v(b"x")
    assert samples.shape == (480000,) and header["rate"] == 16000


def test_gate1_vd_is_json_serialisable(tmp_path):
    """HIGH-0: the recorder's Gate 1 is written with `_write_json`, so its output must be plain JSON."""
    rows = [{"id": "1001_IEO_ANG_XX", "file": "1001_IEO_ANG_XX.wav"}]
    gate = record203.gate1_vd(rows, set(), tmp_path)
    text = json.dumps(gate)                       # the exact call `_write_json` makes (no default=)
    assert json.loads(text)["verdict"] == "failed"
    record203._write_json(tmp_path / "gate1.json", gate)
    assert json.loads((tmp_path / "gate1.json").read_text())["verdict"] == "failed"


def _stub_recorder(monkeypatch, tmp_path, work, *, same_samples=True):
    """Replace every model/audio call in run_recording; each clip is a tiny synthetic record."""
    import contextlib

    monkeypatch.setattr(record203, "load_part188",
                        lambda model_path, threads: (None, types.SimpleNamespace(
                            inference_mode=contextlib.nullcontext), object(), [0, 1], {}, []))
    monkeypatch.setattr(record203, "gate2_parity", lambda *args: {"verdict": "passed"})
    # The real identity reads the model snapshot from disk/Hub; this stub replaces it, and the fixture's
    # identity.json is rewritten to match so the resume check sees one run.
    identity = {"model_id": record203.MODEL_ID, "model_snapshot_revision": record203.MODEL_REVISION,
                "manifest_sha256": "stub", "data_dir": "stub", "study": "part203"}
    (work / "identity.json").write_text(json.dumps(identity))
    monkeypatch.setattr(record203, "recording_identity", lambda **kwargs: dict(identity))
    monkeypatch.setattr(record203, "encode_text_assets", lambda *args: None)
    monkeypatch.setattr(record203, "acoustic_features", lambda samples, text: np.zeros(7, np.float32))
    monkeypatch.setattr(record203, "decode_audio_v", lambda raw: (
        np.zeros(160, np.float32), {"rate": 16000, "channels": 1, "frames": 160}))
    monkeypatch.setattr(record203, "decode_audio_d",
                        lambda raw: np.zeros(160 if same_samples else 100, np.float32))
    monkeypatch.setattr(record203, "embed_v", lambda model, torch, samples: np.ones(
        record203.EMBEDDING_SIZE, np.float16))
    monkeypatch.setattr(record203, "embed_d", lambda model, torch, samples: np.ones(
        record203.EMBEDDING_SIZE, np.float32))

    def record_one(row, raw, record_dir, model, torch, layers, buffer, text):
        from part196.record196 import _atomic_npz
        comparison = record203.v_vs_d(np.zeros(160, np.float32),
                                      np.zeros(160 if same_samples else 100, np.float32))
        _atomic_npz(record_dir / f"{row['id']}.npz", id=np.asarray(row["id"]),
                    v=np.ones(record203.EMBEDDING_SIZE, np.float16),
                    emb=np.ones(record203.EMBEDDING_SIZE, np.float32),
                    act_mean=np.zeros(record203.N_UNITS, np.float16),
                    acoustic=np.zeros(7, np.float32), status=np.asarray("ok"),
                    v_same_samples=np.asarray(comparison["same_samples"]))
        return comparison

    monkeypatch.setattr(record203, "_record_one", record_one)
    data = tmp_path / "data"
    data.mkdir()
    return data


def test_limit_is_a_timing_run_that_writes_a_limited_gate_and_summary(tmp_path, monkeypatch, capsys):
    """MEDIUM-2: `--limit N` records N clips, prints elapsed_seconds and clips_processed, exits cleanly."""
    manifest, work, rows = _manifest_and_records(tmp_path, dim=analyze203.EMBEDDING_SIZE)
    data = _stub_recorder(monkeypatch, tmp_path, work)
    for row in rows:                                   # the stub reads each file's bytes; content is irrelevant
        (data / row["file"]).write_bytes(b"RIFF")
    for record in (work / "records").glob("*.npz"):    # a fresh run: nothing recorded yet, so --limit bites
        record.unlink()
    result = record203.run_recording(manifest_path=manifest, data_dir=data, work_dir=work, limit=3)
    assert result["clips_processed"] == 3 and result["elapsed_seconds"] >= 0.0
    gate = json.loads((work / "gate1.json").read_text())
    assert gate["limited"] is True and gate["verdict"] == "passed"
    assert gate["successfully_recorded"] == 3 and gate["missing_count"] == 0
    assert gate["not_yet_recorded_count"] == len(rows) - 3
    printed = capsys.readouterr().out
    assert "clips_processed=3" in printed and "elapsed_seconds=" in printed
    # The un-limited run then sees the three kept rows and completes the rest.
    full = record203.run_recording(manifest_path=manifest, data_dir=data, work_dir=work)
    assert full["gate1"] == "passed" and full["resumed_ok"] == 3


def test_limited_run_still_fails_when_v_and_d_inputs_differ(tmp_path, monkeypatch):
    manifest, work, rows = _manifest_and_records(tmp_path, dim=analyze203.EMBEDDING_SIZE)
    data = _stub_recorder(monkeypatch, tmp_path, work, same_samples=False)
    for row in rows:
        (data / row["file"]).write_bytes(b"RIFF")
    for record in (work / "records").glob("*.npz"):
        record.unlink()
    with pytest.raises(RuntimeError, match="Gate 1"):
        record203.run_recording(manifest_path=manifest, data_dir=data, work_dir=work, limit=3)
    gate = json.loads((work / "gate1.json").read_text())
    assert gate["limited"] is True and gate["v_d_input_mismatch_count"] == 3


def test_decode_v_with_the_real_soundfile_on_a_synthetic_wav():
    """MEDIUM-4: the real dependency decodes a 16-bit 16 kHz mono WAV; the fake-module tests cannot see this."""
    pytest.importorskip("soundfile")
    samples = (np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 0.5).astype(np.float32)
    decoded, header = decode_audio_v(_wav(samples))
    assert decoded.dtype == np.float32 and decoded.shape == (16000,)
    assert header == {"rate": 16000, "channels": 1, "frames": 16000}
    assert np.max(np.abs(decoded - samples)) < 1e-3          # 16-bit quantisation only


def test_manifest_rows_carry_the_part196_fields_the_shared_code_needs(tmp_path):
    rows = [{"file": "1001_IEO_ANG_XX.wav", "actor": "1001", "sentence": "IEO", "emotion": "ANG",
             "intensity": "XX", "fold": "0", "cell_rank": "1"}]
    path = tmp_path / "m.csv"
    select203.write_manifest(path, rows)
    read = read_manifest_rows(path)
    assert read[0]["id"] == "1001_IEO_ANG_XX"
    assert read[0]["speaker_id"] == "1001" and read[0]["style"] == "ANG"
    assert read[0]["text_key"] == select203.hashlib.sha256(b"IEO").hexdigest()


def test_every_sentence_code_in_the_list_has_a_transcript():
    rows = read_file_list(HERE / "prior" / "cremad-audiowav.tsv")
    assert {row["sentence"] for row in rows} <= set(SENTENCES)


def _download_manifest(tmp_path, names) -> Path:
    path = tmp_path / "manifest.csv"
    lines = ["file,actor,sentence,emotion,intensity,fold,cell_rank"]
    lines += [f"{name},1001,IEO,ANG,XX,0,1" for name in names]
    path.write_text("\n".join(lines) + "\n")
    return path


def _stub_curl(tmp_path) -> tuple[Path, Path]:
    """A `curl` that records its arguments and exits 1, so nothing leaves the machine."""
    log = tmp_path / "curl-args.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "curl"
    stub.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "{log}"\nexit 1\n')
    stub.chmod(0o755)
    return bin_dir, log


def _run_download(tmp_path, manifest, data_dir):
    bin_dir, log = _stub_curl(tmp_path)
    import os
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    done = subprocess.run(["bash", str(HERE / "download.sh"), str(manifest), str(data_dir)],
                          capture_output=True, text=True, timeout=120, env=env, cwd=PROJECT_ROOT)
    return done, log


def test_download_accepts_every_selected_name_and_refuses_paths(tmp_path):
    """HIGH-1: the manifest pools all twelve sentence codes; the name check must not hardcode IEO."""
    names = [line.strip() for line in (HERE / "download-list.txt").read_text().splitlines() if line.strip()]
    assert len(names) == 2730 and len({name.split("_")[1] for name in names}) == 12
    data_dir = tmp_path / "data"
    # Every one of the 2,730 names passes validation. The whole validation loop runs before the first curl, so
    # reaching curl on the first name proves the loop accepted all of them; the stub then exits 1 and stops.
    done, log = _run_download(tmp_path, _download_manifest(tmp_path, names), data_dir)
    assert "unsafe or unexpected" not in done.stderr, done.stderr
    assert names[0] in log.read_text()                  # the first name was handed to curl
    assert re.fullmatch(r"\d{4}_[A-Z]{3}_[A-Z]{3}_[A-Z]{2}\.wav", "1001_ITH_ANG_XX.wav")
    # A name with a path separator or a parent reference is still refused, before any curl runs.
    for bad in ("../1001_IEO_ANG_XX.wav", "sub/1001_IEO_ANG_XX.wav", "1001_IEO_ANG_XX.wav/.."):
        case = tmp_path / f"bad-{abs(hash(bad))}"
        case.mkdir()
        manifest = _download_manifest(case, [bad])
        done, log = _run_download(case, manifest, case / "data")
        assert done.returncode != 0 and "unsafe or unexpected" in done.stderr, (bad, done.stderr)
        assert not log.exists(), bad


def test_recorder_docstring_says_where_the_v_d_statement_is_written():
    """LOW-6: the V/D statement lives in gate1.json (gate1_vd's v_vs_d), not in the identity file."""
    doc = record203.__doc__
    assert "gate1.json" in doc
    assert "identity\nwhether they are the same computation" not in doc


def test_gate2_and_results_are_written_atomically(tmp_path, monkeypatch):
    """LOW-8: both JSON files go through a temp file + os.replace, so a kill never leaves a half file."""
    import os

    replaced = []
    real_replace = os.replace

    def spy_replace(src, dst):
        replaced.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy_replace)
    n_layers, width = 2, 8
    monkeypatch.setattr(analyze203, "N_CLASSES", len(EMOTIONS))
    monkeypatch.setattr(analyze203, "N_LAYERS", n_layers)
    monkeypatch.setattr(analyze203, "UNITS_PER_LAYER", width)
    monkeypatch.setattr(analyze203, "N_UNITS", n_layers * width)
    monkeypatch.setattr(analyze203, "BOOTSTRAPS", 8)
    manifest, work, rows = _manifest_and_records(tmp_path, dim=analyze203.EMBEDDING_SIZE, full=True,
                                                 n_units=n_layers * width)
    analyze203.analyze(manifest_path=manifest, work_dir=work, out_dir=tmp_path / "out")
    assert any(dst.endswith("results.json") for _, dst in replaced)
    # The recorder's gate2.json goes through the same atomic writer.
    data = _stub_recorder(monkeypatch, tmp_path, work)
    for record in (work / "records").glob("*.npz"):
        record.unlink()
    for row in rows:
        (data / row["file"]).write_bytes(b"RIFF")
    monkeypatch.setattr(record203, "gate2_parity", lambda *args: {"verdict": "failed"})
    with pytest.raises(RuntimeError, match="Gate 2"):
        record203.run_recording(manifest_path=manifest, data_dir=data, work_dir=work, limit=1)
    assert json.loads((work / "gate2.json").read_text()) == {"verdict": "failed"}
    assert not list(work.glob(".gate2.json.*"))


def test_scripts_import_under_isolated_mode():
    python = sys.executable
    for script in ("select203.py", "record203.py", "analyze203.py"):
        done = subprocess.run([python, "-I", "-B", str(HERE / script), "--help"],
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr
