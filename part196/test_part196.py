"""Synthetic tests for Part 196 selection, split checks, and analysis math."""
from __future__ import annotations

import sys
import types
import csv
import json
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
    reading_conditions,
    reverse_paired_difference,
    selected_style_profile,
    select_b_phrasings,
    _validate_work_identity,
)
from part196 import analyze196  # noqa: E402
from part196.record196 import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    _atomic_npz,
    acoustic_features,
    decode_audio_bytes,
    encode_text_assets,
    recording_identity,
    write_or_check_identity,
)
from part196.select196 import (  # noqa: E402
    MAX_DURATION_SECONDS,
    STYLES,
    clip_sample_limit,
    draw_manifest,
    duration_in_registered_range,
    gate1_check,
    gate1_check_recorded,
    shared_text_pool,
    text_key,
)


def fake_metadata() -> list[dict[str, str]]:
    rows = []
    for speaker in ("ex01", "ex02", "ex03", "ex04"):
        for text_number in range(3):
            text = f"{speaker} unique sentence {text_number}"
            for style in STYLES:
                rows.append({"id": f"{speaker}_{style}_{text_number}", "speaker_id": speaker,
                             "style": style, "text": text, "duration_seconds": 3.0})
    # This exact transcript leaks across speakers and must leave the pool.
    rows.extend([
        {"id": "cross_a", "speaker_id": "ex01", "style": "happy", "text": "shared speaker text",
         "duration_seconds": 3.0},
        {"id": "cross_b", "speaker_id": "ex02", "style": "sad", "text": "shared speaker text",
         "duration_seconds": 3.0},
    ])
    # A text seen in only one style cannot be used for the controlled pool.
    rows.append({"id": "single", "speaker_id": "ex01", "style": "sad", "text": "single style",
                 "duration_seconds": 3.0})
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
    assert all(row["duration_status"] == "in_range_wav_header" for row in first)
    assert "text" not in first[0] and len(first[0]["text_key"]) == 64
    drawn_clusters = {}
    for row in first:
        drawn_clusters.setdefault((row["speaker_id"], row["text_key"]), set()).add(row["style"])
    assert all(len(styles) >= 2 for styles in drawn_clusters.values())


def test_shared_text_pool_filters_wav_durations_before_sharing():
    rows = [
        {"id": "keep_a", "speaker_id": "ex01", "style": "happy", "text": "keep",
         "duration_seconds": 3.0},
        {"id": "keep_b", "speaker_id": "ex01", "style": "sad", "text": "keep",
         "duration_seconds": 4.0},
        {"id": "late_sibling", "speaker_id": "ex01", "style": "default", "text": "keep",
         "duration_seconds": 11.0},
        {"id": "only_after_filter_a", "speaker_id": "ex01", "style": "happy", "text": "single",
         "duration_seconds": 3.0},
        {"id": "only_after_filter_b", "speaker_id": "ex01", "style": "sad", "text": "single",
         "duration_seconds": 11.0},
    ]
    pool = shared_text_pool(rows)
    assert {row["id"] for row in pool} == {"keep_a", "keep_b"}
    assert clip_sample_limit() == 160_000
    assert duration_in_registered_range(2.0)
    assert duration_in_registered_range(MAX_DURATION_SECONDS)
    assert not duration_in_registered_range(1.99)
    assert not duration_in_registered_range(10.01)


def test_text_draw_skips_a_cluster_when_only_one_style_has_room():
    rows = [
        {"id": "a_happy", "speaker_id": "ex01", "style": "happy", "text": "text a",
         "duration_seconds": 3.0},
        {"id": "a_sad", "speaker_id": "ex01", "style": "sad", "text": "text a",
         "duration_seconds": 3.0},
        {"id": "b_happy", "speaker_id": "ex01", "style": "happy", "text": "text b",
         "duration_seconds": 3.0},
        {"id": "b_default", "speaker_id": "ex01", "style": "default", "text": "text b",
         "duration_seconds": 3.0},
    ]
    manifest, _ = draw_manifest(rows, seed=196, per_cell=1)
    clusters = {}
    for row in manifest:
        clusters.setdefault(row["text_key"], set()).add(row["style"])
    assert len(manifest) == 2
    assert all(len(styles) >= 2 for styles in clusters.values())


def test_draw_takes_texts_with_the_rare_style_first():
    """A rare style must not starve once the common cells fill (ex01 laughing drew 38 of 40 before this order)."""
    rows = []
    # Twelve texts read in happy and sad only, then three texts that also carry the rare laughing style.
    for i in range(12):
        for style in ("happy", "sad"):
            rows.append({"id": f"c{i:02d}_{style}", "speaker_id": "ex01", "style": style,
                         "text": f"common {i:02d}", "duration_seconds": 3.0})
    for i in range(3):
        for style in ("happy", "sad", "laughing"):
            rows.append({"id": f"r{i}_{style}", "speaker_id": "ex01", "style": style,
                         "text": f"rare {i}", "duration_seconds": 3.0})
    for seed in range(20):
        manifest, _ = draw_manifest(rows, seed=seed, per_cell=3)
        laughing = [row for row in manifest if row["style"] == "laughing"]
        assert len(laughing) == 3, (seed, len(laughing))
        clusters = {}
        for row in manifest:
            clusters.setdefault(row["text_key"], set()).add(row["style"])
        assert all(len(styles) >= 2 for styles in clusters.values())

def test_decode_audio_bytes_uses_the_shared_ten_second_sample_cap(monkeypatch):
    output = np.arange(clip_sample_limit() + 12, dtype=np.float32).tobytes()
    fake_imageio = types.SimpleNamespace(get_ffmpeg_exe=lambda: "fake-ffmpeg")
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_imageio)
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: types.SimpleNamespace(stdout=output))
    decoded = decode_audio_bytes(b"synthetic input")
    assert decoded.size == 160_000
    assert decoded[-1] == 159_999


def _gate_fixture():
    rows = []
    for speaker_index, speaker in enumerate(("ex01", "ex02", "ex03", "ex04")):
        for style in STYLES:
            rows.append({"id": f"{speaker}_{style}", "speaker_id": speaker,
                         "style": style, "text": f"{speaker} shared sentence",
                         "fold": speaker})
    return rows


def _committed_manifest():
    path = Path(__file__).with_name("manifest196.csv")
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


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


def test_gate1_floor_counts_only_successfully_recorded_clips():
    rows = []
    for speaker in ("ex01", "ex02", "ex03", "ex04"):
        for style in STYLES:
            for index in range(40):
                rows.append({"id": f"{speaker}_{style}_{index}", "speaker_id": speaker,
                             "style": style, "text_key": f"{speaker}_shared_{index}",
                             "fold": speaker})
    recorded_ids = {row["id"] for row in rows}
    recorded_ids.remove("ex01_whisper_0")
    gate = gate1_check_recorded(rows, recorded_ids)
    assert gate["verdict"] == "failed"
    assert gate["cell_counts"]["ex01/whisper"] == 39
    assert not gate["cell_minima_passed"]


def test_gate1_refuses_recording_dropout_that_leaves_a_final_singleton_cluster():
    rows = _committed_manifest()
    failed_clip = "ex01_confused_00313"
    failed_row = next(row for row in rows if row["id"] == failed_clip)
    cluster_ids = {row["id"] for row in rows
                   if row["speaker_id"] == failed_row["speaker_id"]
                   and row["text_key"] == failed_row["text_key"]}
    assert cluster_ids == {"ex01_confused_00313", "ex01_happy_00313"}

    gate = gate1_check_recorded(rows, {row["id"] for row in rows} - {failed_clip})

    assert all(count >= 40 for count in gate["cell_counts"].values())
    assert gate["cell_minima_passed"]
    assert gate["verdict"] == "failed"
    assert gate["multistyle_clusters_passed"] is False
    assert gate["singleton_cluster_count"] == 1
    assert gate["singleton_clusters"] == [{
        "speaker_id": "ex01", "text_key": failed_row["text_key"],
        "styles": ["happy"], "clip_count": 1,
    }]


def test_gate1_allows_recording_dropout_when_every_final_cluster_stays_multistyle():
    rows = _committed_manifest()
    clusters = {}
    for row in rows:
        clusters.setdefault((row["speaker_id"], row["text_key"]), []).append(row)
    dropped = next(cluster for cluster in clusters.values() if len(cluster) >= 2)
    dropped_ids = {row["id"] for row in dropped}
    successful_ids = {row["id"] for row in rows} - dropped_ids

    gate = gate1_check_recorded(rows, successful_ids)

    assert dropped_ids
    assert all(count >= 40 for count in gate["cell_counts"].values())
    assert gate["multistyle_clusters_passed"] is True
    assert gate["singleton_cluster_count"] == 0
    assert gate["verdict"] == "passed"


def test_committed_manifest_keeps_each_drawn_text_cluster_multistyle():
    rows = _committed_manifest()
    clusters = {}
    for row in rows:
        clusters.setdefault((row["speaker_id"], row["text_key"]), set()).add(row["style"])
    assert clusters
    share = sum(len(styles) >= 2 for styles in clusters.values()) / len(clusters)
    assert share == pytest.approx(1.0)
    assert all(len(styles) >= 2 for styles in clusters.values())


def test_selected_style_profile_ignores_reused_ids_in_other_styles():
    # Expresso reuses ids: ex02_default_00001 is one default clip plus two longform rows (found by the real run).
    rows = [
        {"id": "ex02_default_00001", "speaker_id": "ex02", "style": "default", "text_key": text_key("one two"),
         "duration_seconds": "2.0"},
    ]
    metadata = [
        {"id": "ex02_default_00001", "speaker_id": "ex02", "style": "longform", "text": "a long passage"},
        {"id": "ex02_default_00001", "speaker_id": "ex02", "style": "default", "text": "one two"},
        {"id": "ex02_default_00001", "speaker_id": "ex02", "style": "longform", "text": "another passage"},
    ]

    profile = selected_style_profile(rows, {"ex02_default_00001"}, metadata)

    assert profile["default"]["clip_count"] == 1
    assert profile["default"]["mean_transcript_words"] == 2.0


def test_selected_style_profile_refuses_a_wrong_transcript_for_the_same_identity():
    rows = [
        {"id": "a", "speaker_id": "ex01", "style": "happy", "text_key": text_key("one two"),
         "duration_seconds": "2.0"},
    ]
    metadata = [{"id": "a", "speaker_id": "ex01", "style": "happy", "text": "something else"}]

    with pytest.raises(ValueError, match="transcript hash differs"):
        selected_style_profile(rows, {"a"}, metadata)


def test_selected_style_profile_uses_manifest_durations_and_metadata_transcripts():
    rows = [
        {"id": "a", "speaker_id": "ex01", "style": "happy", "text_key": text_key("one two"),
         "duration_seconds": "2.0"},
        {"id": "b", "speaker_id": "ex01", "style": "happy", "text_key": text_key("three four five"),
         "duration_seconds": "4.0"},
        {"id": "c", "speaker_id": "ex01", "style": "sad", "text_key": text_key("six"),
         "duration_seconds": "6.0"},
    ]
    metadata = [
        {"id": "a", "speaker_id": "ex01", "style": "happy", "text": "one two"},
        {"id": "b", "speaker_id": "ex01", "style": "happy", "text": "three four five"},
        {"id": "c", "speaker_id": "ex01", "style": "sad", "text": "six"},
    ]

    profile = selected_style_profile(rows, {"a", "b", "c"}, metadata)

    assert profile["happy"] == {
        "clip_count": 2,
        "duration_seconds": {"mean": 3.0, "q25": 2.5, "median": 3.0, "q75": 3.5},
        "mean_transcript_words": 2.5,
    }
    assert profile["sad"] == {
        "clip_count": 1,
        "duration_seconds": {"mean": 6.0, "q25": 6.0, "median": 6.0, "q75": 6.0},
        "mean_transcript_words": 1.0,
    }
    assert profile["confused"]["clip_count"] == 0


def test_model_weight_hash_prevents_resume_into_changed_snapshot(tmp_path):
    snapshot = tmp_path / MODEL_REVISION
    snapshot.mkdir()
    weights = snapshot / "model.safetensors"
    weights.write_bytes(b"weights A")
    rows = [{"id": "one", "speaker_id": "ex01", "style": "happy", "text_key": "t"}]
    versions = {"transformers": "5.20.0", "sentence_transformers": "6.2.0", "torch": "2.14.1"}
    identity_path = tmp_path / "work" / "identity.json"
    identity_a = recording_identity(rows=rows, data_dir=tmp_path / "data", snapshot_dir=snapshot,
                                    package_versions=versions)
    assert identity_a["model_id"] == MODEL_ID
    assert identity_a["model_snapshot_revision"] == MODEL_REVISION
    assert identity_a["transformers_version"] == versions["transformers"]
    assert identity_a["sentence_transformers_version"] == versions["sentence_transformers"]
    assert identity_a["torch_version"] == versions["torch"]
    write_or_check_identity(identity_path, identity_a)
    assert json.loads(identity_path.read_text(encoding="utf-8")) == identity_a

    weights.write_bytes(b"weights B")
    identity_b = recording_identity(rows=rows, data_dir=tmp_path / "data", snapshot_dir=snapshot,
                                    package_versions=versions)
    assert identity_a["weight_sha256"] != identity_b["weight_sha256"]
    with pytest.raises(ValueError, match="identity differs"):
        write_or_check_identity(identity_path, identity_b)


def test_analyzer_validates_manifest_against_recording_identity(tmp_path):
    snapshot = tmp_path / MODEL_REVISION
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"weights")
    rows = [{"id": "one", "speaker_id": "ex01", "style": "happy", "text_key": "t"}]
    versions = {"transformers": "5.20.0", "sentence_transformers": "6.2.0", "torch": "2.14.1"}
    identity = recording_identity(rows=rows, data_dir=tmp_path / "data", snapshot_dir=snapshot,
                                  package_versions=versions)
    work = tmp_path / "work"
    work.mkdir()
    (work / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
    assert _validate_work_identity(work, rows) == identity
    with pytest.raises(ValueError, match="manifest differs"):
        _validate_work_identity(work, rows + [{"id": "two"}])


def test_text_assets_rebuild_corrupt_or_partial_files_atomically(tmp_path):
    key = text_key("a transcript")
    rows = [{"id": "clip", "text_key": key}]
    texts = {"clip": "a transcript"}

    class FakeModel:
        def __init__(self):
            self.calls = []

        def encode(self, values, **kwargs):
            self.calls.append((len(values), kwargs["prompt_name"]))
            return np.ones((len(values), 768), dtype=np.float32)

    model = FakeModel()
    encode_text_assets(model, rows, texts, tmp_path)
    prompt_path = tmp_path / "prompts196.npz"
    transcript_path = tmp_path / "transcripts196.npz"
    prompt_path.write_bytes(b"partial prompt asset")
    transcript_path.write_bytes(b"partial transcript asset")
    encode_text_assets(model, rows, texts, tmp_path)
    assert model.calls == [(56, "SearchQuery"), (1, "document"),
                           (56, "SearchQuery"), (1, "document")]
    with np.load(prompt_path, allow_pickle=False) as saved:
        assert saved["embeddings"].shape == (7, 8, 768)
    with np.load(transcript_path, allow_pickle=False) as saved:
        assert saved["embeddings"].shape == (1, 768)
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_npz_preserves_previous_asset_if_a_write_is_interrupted(tmp_path, monkeypatch):
    path = tmp_path / "asset.npz"
    _atomic_npz(path, value=np.asarray([1]))
    previous = path.read_bytes()

    def interrupt(stream, **_items):
        stream.write(b"partial")
        raise RuntimeError("simulated interrupted write")

    monkeypatch.setattr(np, "savez_compressed", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        _atomic_npz(path, value=np.asarray([2]))
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))


def test_reading_conditions_keep_registered_difference_signs():
    reversed_difference = reverse_paired_difference(
        -0.04, {"low": -0.08, "high": 0.02, "n_boot": 2000})
    assert reversed_difference["mean_balanced_accuracy_difference"] == pytest.approx(0.04)
    assert reversed_difference["text_cluster_95"] == {"low": -0.02, "high": 0.08, "n_boot": 2000}
    conditions = reading_conditions({"low": 0.15, "high": 0.3},
                                    reversed_difference["text_cluster_95"])
    assert conditions["d_above_chance"]["threshold"] == pytest.approx(1 / 7)
    assert conditions["d_above_chance"]["condition_passed"] is True
    assert conditions["b56_minus_d_non_inferiority"]["condition_passed"] is True
    assert not reading_conditions({"low": 1 / 7}, {"low": -0.03})[
        "d_above_chance"]["condition_passed"]
    assert not reading_conditions({"low": 0.2}, {"low": -0.03})[
        "b56_minus_d_non_inferiority"]["condition_passed"]


def test_analyzer_prints_both_reading_conditions_in_its_verdict(monkeypatch, capsys, tmp_path):
    result = {"status": "complete", "gate1": {"verdict": "passed"},
              "gate2": {"verdict": "passed"},
              "reading_rule": {"conditions": {
                  "d_above_chance": {"condition_passed": True},
                  "b56_minus_d_non_inferiority": {"condition_passed": False}}}}
    monkeypatch.setattr(analyze196, "analyze", lambda **_kwargs: result)
    assert analyze196.main(["--manifest", str(tmp_path / "manifest.csv"),
                            "--work-dir", str(tmp_path / "work"),
                            "--out-dir", str(tmp_path / "out")]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["reading_conditions"]["d_above_chance"]["condition_passed"] is True
    assert output["reading_conditions"]["b56_minus_d_non_inferiority"]["condition_passed"] is False


def test_steps_use_the_package_path_only_for_the_recorder():
    steps = Path(__file__).with_name("steps.txt").read_text(encoding="utf-8")
    recorder = next(line for line in steps.splitlines() if "record196.py" in line)
    selector = next(line for line in steps.splitlines() if "select196.py" in line)
    analyzer = next(line for line in steps.splitlines() if "analyze196.py" in line)
    assert "PYTHONPATH=/home/elisepc/eg2-extra" in recorder
    assert "python -I" not in recorder
    assert "python -I" in selector
    assert "python -I" in analyzer


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


def test_recorder_prompt_names_exist_in_the_pinned_model_config():
    """Amendment 1: 'SearchDocument' was not a prompt of this model; every prompt_name used must be in its config."""
    import json, re
    snapshot = Path("/mnt/e/neuron-probe/hf/hub/models--google--embeddinggemma-2/snapshots/"
                    "f9972ac22e696a105c482202646979e33e4b5a1e/config_sentence_transformers.json")
    if not snapshot.is_file():
        import pytest
        pytest.skip("pinned EmbeddingGemma 2 snapshot not on this host")
    prompts = json.loads(snapshot.read_text())["prompts"]
    source = (Path(__file__).resolve().parent / "record196.py").read_text()
    used = set(re.findall(r'prompt_name="([^"]+)"', source))
    assert used, "no prompt_name found in record196.py"
    assert used <= set(prompts), sorted(used - set(prompts))

