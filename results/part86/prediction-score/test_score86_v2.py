"""Synthetic regression tests for the Amendment 7 Part 86 scorer."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest


SCORER_PATH = Path(__file__).with_name("score86_predictions.py")
SPEC = importlib.util.spec_from_file_location("score86_predictions_under_test", SCORER_PATH)
assert SPEC is not None and SPEC.loader is not None
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def synthetic_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "assistant-basin"
    label_dir = tmp_path / "jlens" / "data" / "qwen3.5-4b" / "labels"
    second_reader = root / "results" / "part86" / "second-reader"
    screen_dir = root / "results" / "part86" / "screen-final"
    monkeypatch.setattr(scorer, "J_LENS_LABEL_DIR", label_dir)
    monkeypatch.setattr(scorer, "J_LENS_STATS_PATH", tmp_path / "jlens" / "stats.json")
    monkeypatch.setattr(scorer, "LAYERS", tuple(range(10, 27)))
    monkeypatch.setattr(scorer, "NEURONS_PER_LAYER", 3)
    monkeypatch.setattr(scorer, "RANDOM_PER_LAYER", 5)

    prediction_ids = ("86.1", "86.2", "86.2a", "86.3", "86.4", "86.4a", "86.5", "86.6")
    root.mkdir(parents=True, exist_ok=True)
    (root / "PREDICTIONS.md").write_text(
        "| id | prediction | result | evidence |\n|---|---|---|---|\n"
        + "".join(f"| {item_id} | synthetic | pending | |\n" for item_id in prediction_ids),
        encoding="utf-8",
    )

    screen_rows = [
        {"layer": 22, "kind": "neuron", "id": 0, "answer": "I am a silver fox who tends a quiet forest."},
        {"layer": 22, "kind": "neuron", "id": 1, "answer": "I am a language model."},
        {"layer": 22, "kind": "neuron", "id": 2, "answer": "red fox speaks red fox speaks red fox speaks"},
        {"layer": 22, "kind": "random_direction", "id": 0, "answer": "I became a lantern guiding ships over dark water."},
        {"layer": 22, "kind": "random_direction", "id": 1, "answer": "I am an assistant."},
        {"layer": 22, "kind": "random_direction", "id": 2, "answer": "red fox speaks red fox speaks red fox speaks"},
        {"layer": 22, "kind": "random_direction", "id": 3, "answer": "Qwen speaks about its own training."},
        {"layer": 22, "kind": "random_direction", "id": 4, "answer": "hi"},
    ]
    screen_dir.mkdir(parents=True, exist_ok=True)
    for layer in scorer.LAYERS:
        rows = screen_rows if layer == 22 else [
            {"layer": layer, "kind": "neuron", "id": index, "answer": "I am a language model."}
            for index in range(scorer.NEURONS_PER_LAYER)
        ] + [
            {"layer": layer, "kind": "random_direction", "id": index, "answer": "I am an assistant."}
            for index in range(scorer.RANDOM_PER_LAYER)
        ]
        with gzip.open(screen_dir / f"layer_{layer}.jsonl.gz", "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")

    items = [
        {"id": "real_item", "text": screen_rows[0]["answer"]},
        {"id": "random_item", "text": screen_rows[3]["answer"]},
    ]
    write_jsonl(label_dir / "items.jsonl", items)
    key = {
        "real_item": [{"layer": 22, "unit": "neuron", "index": 0}],
        "random_item": [{"layer": 22, "unit": "random_direction", "index": 0}],
    }
    (label_dir / "KEY-private.json").write_text(json.dumps(key), encoding="utf-8")

    v1_labels = [
        {"id": "real_item", "persona": "unclear identity", "kind": "unclear", "coherent": False},
        {"id": "random_item", "persona": "unclear identity", "kind": "unclear", "coherent": False},
    ]
    write_jsonl(label_dir / "labels.jsonl", v1_labels)
    v2_labels = [
        {"id": "real_item", "persona": "silver fox", "kind": "fictional character", "coherent": True},
        {"id": "random_item", "persona": "unclear identity", "kind": "unclear", "coherent": False},
    ]
    labels_v2_path = second_reader / "labels_ds_v2_full.jsonl"
    write_jsonl(labels_v2_path, v2_labels)
    (second_reader / "calib-agreement-v2.json").write_text(
        json.dumps({"pairs": {"ds_v2~luna_v2": {"kappa_coherent_persona": 0.81}}}),
        encoding="utf-8",
    )
    (second_reader / "agreement.json").write_text(json.dumps({"kappa": 0.75}), encoding="utf-8")
    score_md = root / "results" / "part86" / "prediction-score" / "SCORE.md"
    score_md.parent.mkdir(parents=True, exist_ok=True)
    score_md.write_text("existing version-1 report\n", encoding="utf-8")

    return {
        "root": root,
        "label_dir": label_dir,
        "labels_v2_path": labels_v2_path,
        "second_reader": second_reader,
        "score_md": score_md,
        "v2_labels": v2_labels,
    }


def test_default_v2_uses_v2_counts_and_writes_separate_report(synthetic_score, capsys):
    case = synthetic_score

    result = scorer.main([], root=case["root"])
    output = capsys.readouterr().out

    assert result == 0
    assert "READING: version v2" in output
    assert "version-1 labels are not loaded or pooled" in output
    assert "LABEL PREFLIGHT: version-2 labels match all 2 first-reader item ids" in output
    assert "error_rows=0 invalid_rows=0" in output
    assert "Primary labels give 1 coherent personas among 3 real neurons and 0 among 5 random directions" in output
    assert "calib-agreement-v2.json pairs['ds_v2~luna_v2'].kappa_coherent_persona" in output
    report_v2 = case["root"] / "results" / "part86" / "prediction-score" / "SCORE-v2.md"
    assert report_v2.is_file()
    assert "Primary labels give 1 coherent personas" in report_v2.read_text(encoding="utf-8")
    assert case["score_md"].read_text(encoding="utf-8") == "existing version-1 report\n"


def test_v1_mode_uses_only_v1_label_counts(synthetic_score, capsys):
    case = synthetic_score

    result = scorer.main(["--reading", "v1"], root=case["root"])
    output = capsys.readouterr().out

    assert result == 0
    assert "READING: version v1" in output
    assert "Primary labels give 0 coherent personas among 3 real neurons and 0 among 5 random directions" in output
    assert "calib-agreement-v2.json" not in output
    assert not (case["root"] / "results" / "part86" / "prediction-score" / "SCORE-v2.md").exists()
    assert case["score_md"].read_text(encoding="utf-8") == "existing version-1 report\n"


@pytest.mark.parametrize(
    ("problem", "expected_count"),
    [
        ("missing_row", "missing_ids=1"),
        ("error_row", "error_rows=1"),
        ("duplicate", "duplicate_ids=1"),
        ("extra_id", "extra_ids=1"),
    ],
)
def test_v2_preflight_rejects_incomplete_or_invalid_rows(
    synthetic_score, capsys, problem: str, expected_count: str
):
    case = synthetic_score
    rows = list(case["v2_labels"])
    if problem == "missing_row":
        rows = rows[:1]
    elif problem == "error_row":
        rows[0] = {"id": "real_item", "error": "reader failed"}
    elif problem == "duplicate":
        rows.append(dict(rows[0]))
    elif problem == "extra_id":
        rows.append({"id": "extra_item", "persona": "extra", "kind": "abstract", "coherent": True})
    write_jsonl(case["labels_v2_path"], rows)
    # A parse error here proves the scorer refuses the labels before opening this file.
    (case["label_dir"] / "KEY-private.json").write_text("not-json", encoding="utf-8")

    result = scorer.main(["--reading", "v2"], root=case["root"])
    output = capsys.readouterr().out

    assert result != 0
    assert "NOT READY: version-2 labels failed preflight" in output
    assert expected_count in output
    assert "not-json" not in output
    assert case["score_md"].read_text(encoding="utf-8") == "existing version-1 report\n"


def test_v2_missing_label_file_is_not_ready_before_key_read(synthetic_score, capsys):
    case = synthetic_score
    case["labels_v2_path"].unlink()
    (case["label_dir"] / "KEY-private.json").write_text("not-json", encoding="utf-8")

    result = scorer.main(["--reading", "v2"], root=case["root"])
    output = capsys.readouterr().out

    assert result != 0
    assert "NOT READY: version-2 labels failed preflight" in output
    assert "label_file=missing" in output
    assert "missing_ids=2" in output
    assert "not-json" not in output


def test_v2_low_kappa_gates_verdicts_but_keeps_86_5_not_scorable(synthetic_score, capsys):
    case = synthetic_score
    (case["second_reader"] / "calib-agreement-v2.json").write_text(
        json.dumps({"pairs": {"ds_v2~luna_v2": {"kappa_coherent_persona": 0.599}}}),
        encoding="utf-8",
    )

    result = scorer.main(["--reading", "v2"], root=case["root"])
    output = capsys.readouterr().out

    assert result == 0
    for prediction_id in ("86.2", "86.2a", "86.3", "86.4", "86.4a", "86.6"):
        assert f"{prediction_id}: NOT READ." in output
    assert "86.5: not scorable." in output

