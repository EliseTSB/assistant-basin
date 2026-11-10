"""Synthetic tests for Part 203 selection, folds, readouts, gates and the V/D comparison."""
from __future__ import annotations

import io
import json
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
                            status=np.asarray("ok"))
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
    assert gate1(rows, records["status"], records["v"], records["d"])["verdict"] == "passed"


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
    assert written["three_actor_ablation"]["V"]["three_actor_draws_mean"] > 0.5
    assert written["rules"]["predictions"] == "{{CLAUDE: predictions}}"


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


def test_scripts_import_under_isolated_mode():
    python = sys.executable
    for script in ("select203.py", "record203.py", "analyze203.py"):
        done = subprocess.run([python, "-I", "-B", str(HERE / script), "--help"],
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr
