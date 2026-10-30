"""Synthetic regression tests for Part 86 Amendment 8 atlas counts."""
from __future__ import annotations

import gzip
import importlib.util
import json
import os
from pathlib import Path

import pytest


SCRIPT_PATH = Path(
    os.environ.get("ATLAS86_SCRIPT_UNDER_TEST", Path(__file__).with_name("atlas86_v2.py"))
).resolve()
SPEC = importlib.util.spec_from_file_location("atlas86_v2_under_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
atlas = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(atlas)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def label_for(class_name: str) -> dict:
    if class_name == "coherent_persona":
        return {"persona": "silver fox", "kind": "fictional character", "coherent": True}
    if class_name == "coherent_still_qwen":
        return {"persona": "Q, a language model", "kind": "still Qwen or an assistant", "coherent": True}
    if class_name == "not_coherent":
        return {"persona": "unclear identity", "kind": "fictional character", "coherent": False}
    if class_name == "unclear":
        return {"persona": "unclear identity", "kind": "unclear", "coherent": False}
    raise AssertionError(class_name)


@pytest.fixture
def synthetic_atlas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    scorer = atlas.SCORER
    monkeypatch.setattr(scorer, "LAYERS", (22,))
    monkeypatch.setattr(scorer, "NEURONS_PER_LAYER", 800)
    monkeypatch.setattr(scorer, "RANDOM_PER_LAYER", 0)

    root = tmp_path / "assistant-basin"
    screen_dir = root / "results" / "part86" / "screen-final"
    second_reader = root / "results" / "part86" / "second-reader"
    label_dir = tmp_path / "jlens" / "data" / "qwen3.5-4b" / "labels"
    units_path = root / "results" / "part112" / "units.json"
    monkeypatch.setattr(scorer, "J_LENS_LABEL_DIR", label_dir)
    screen_dir.mkdir(parents=True)

    screen_rows = [
        {
            "layer": 22,
            "kind": "neuron",
            "id": index,
            "answer": f"I am a silver fox tending the forest number {index}.",
        }
        for index in range(800)
    ]
    with gzip.open(screen_dir / "layer_22.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in screen_rows:
            stream.write(json.dumps(row) + "\n")

    other_kinds = (
        ["object or place"] * 46
        + ["abstract"] * 15
        + ["human"] * 8
        + ["fictional character"] * 7
        + ["animal", "deity or spirit"]
    )
    units = []
    for index in range(465):
        is_another_ai = index < 387
        units.append(
            {
                "layer": 22,
                "neuron": index,
                "persona": f"synthetic unit {index}",
                "kind": "another AI" if is_another_ai else other_kinds[index - 387],
                "q_name": is_another_ai and index < 344,
                "id": f"q112-{index:03d}",
            }
        )
    units_path.parent.mkdir(parents=True, exist_ok=True)
    units_path.write_text(json.dumps(units), encoding="utf-8")

    items = []
    key = {}
    v2_labels = []
    v1_labels = []
    for index, row in enumerate(screen_rows):
        item_id = f"item-{index}"
        items.append({"id": item_id, "text": row["answer"]})
        key[item_id] = [{"layer": 22, "unit": "neuron", "index": index}]
        v2_labels.append({"id": item_id, **label_for("coherent_still_qwen")})
        if index < 465:
            v1_labels.append({"id": item_id, **label_for("coherent_persona")})
        else:
            v1_labels.append({"id": item_id, **label_for("not_coherent")})

    write_jsonl(label_dir / "items.jsonl", items)
    (label_dir / "KEY-private.json").write_text(json.dumps(key), encoding="utf-8")
    write_jsonl(label_dir / "labels.jsonl", v1_labels)
    v2_path = second_reader / "labels_ds_v2_full.jsonl"
    write_jsonl(v2_path, v2_labels)
    (second_reader / "calib-agreement-v2.json").write_text(
        json.dumps({"pairs": {"ds_v2~luna_v2": {"kappa_coherent_persona": 0.81}}}),
        encoding="utf-8",
    )
    return {
        "root": root,
        "units_path": units_path,
        "screen_path": screen_dir / "layer_22.jsonl.gz",
        "label_dir": label_dir,
        "v2_path": v2_path,
        "second_reader": second_reader,
        "items": items,
        "key": key,
        "v1_labels": v1_labels,
        "v2_labels": v2_labels,
    }


def set_v2_class(case: dict, start: int, stop: int, class_name: str) -> None:
    for index in range(start, stop):
        case["v2_labels"][index] = {
            "id": f"item-{index}",
            **label_for(class_name),
        }
    write_jsonl(case["v2_path"], case["v2_labels"])


def set_all_hold_counts(case: dict) -> None:
    set_v2_class(case, 0, 800, "coherent_still_qwen")
    set_v2_class(case, 0, 100, "coherent_persona")
    set_v2_class(case, 387, 465, "coherent_persona")
    set_v2_class(case, 465, 753, "coherent_persona")


def run_case(case: dict) -> tuple[int, str]:
    result = atlas.main(["--root", str(case["root"])])
    report = case["root"] / "results" / "part86" / "atlas-v2" / "ATLAS-v2.md"
    assert report.is_file()
    return result, report.read_text(encoding="utf-8")


def test_all_three_predictions_hold(synthetic_atlas, capsys):
    case = synthetic_atlas
    set_all_hold_counts(case)

    result, report = run_case(case)
    output = capsys.readouterr().out

    assert result == 0
    assert "86.11: HOLDS. 100 of 387" in output
    assert "86.12: HOLDS. 78 of 78" in output
    assert "86.13: HOLDS. 466 distinct real screen rows" in output
    assert "atlas=178 not_atlas=288" in output
    assert "86.12 per-kind version-2 classes:" in output
    assert "object or place: n=46 coherent_persona=46" in output
    assert "GATE: kappa=0.810000; source=results/part86/second-reader/calib-agreement-v2.json" in output
    assert report.endswith(output)


def test_each_prediction_fails_in_a_synthetic_case(synthetic_atlas, capsys):
    case = synthetic_atlas
    set_v2_class(case, 0, 800, "not_coherent")
    set_v2_class(case, 0, 250, "coherent_persona")
    set_v2_class(case, 387, 407, "coherent_persona")
    set_v2_class(case, 465, 601, "coherent_persona")

    result, _report = run_case(case)
    output = capsys.readouterr().out

    assert result == 0
    assert "86.11: FAILS. 250 of 387" in output
    assert "86.12: FAILS. 20 of 78" in output
    assert "86.13: FAILS. 406 distinct real screen rows" in output


def test_still_qwen_is_excluded_from_coherent_persona(synthetic_atlas, capsys):
    case = synthetic_atlas
    set_v2_class(case, 0, 800, "not_coherent")
    set_v2_class(case, 0, 387, "coherent_still_qwen")

    result, _report = run_case(case)
    output = capsys.readouterr().out

    assert result == 0
    assert "86.11 another AI version-2 classes: n=387 coherent_persona=0 coherent_still_qwen=387" in output
    assert "86.11: HOLDS. 0 of 387" in output


def test_not_ready_preflight_happens_before_key_read(synthetic_atlas, capsys):
    case = synthetic_atlas
    case["v2_labels"].pop()
    write_jsonl(case["v2_path"], case["v2_labels"])
    (case["label_dir"] / "KEY-private.json").write_text("not-json", encoding="utf-8")

    result, _report = run_case(case)
    output = capsys.readouterr().out

    assert result == 2
    assert "NOT READY: version-2 labels failed preflight" in output
    assert "missing_ids=1" in output
    assert "not-json" not in output
    assert "VERSION-1 ATLAS CHECK" not in output
    assert "GATE:" not in output


def test_atlas_unit_without_labelled_row_names_missing_count(synthetic_atlas, capsys):
    case = synthetic_atlas
    with gzip.open(case["screen_path"], "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    rows[464]["answer"] = "I am an assistant."
    with gzip.open(case["screen_path"], "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")

    case["items"].pop(464)
    case["v2_labels"].pop(464)
    case["v1_labels"].pop(464)
    case["key"].pop("item-464")
    write_jsonl(case["label_dir"] / "items.jsonl", case["items"])
    write_jsonl(case["label_dir"] / "labels.jsonl", case["v1_labels"])
    write_jsonl(case["v2_path"], case["v2_labels"])
    (case["label_dir"] / "KEY-private.json").write_text(json.dumps(case["key"]), encoding="utf-8")

    result, _report = run_case(case)
    output = capsys.readouterr().out

    assert result == 1
    assert "atlas label coverage failed: missing_labelled_screen_rows=1" in output
    assert "VERSION-1 ATLAS CHECK" not in output


def test_low_kappa_makes_all_verdicts_not_read(synthetic_atlas, capsys):
    case = synthetic_atlas
    set_all_hold_counts(case)
    (case["second_reader"] / "calib-agreement-v2.json").write_text(
        json.dumps({"pairs": {"ds_v2~luna_v2": {"kappa_coherent_persona": 0.599}}}),
        encoding="utf-8",
    )

    result, _report = run_case(case)
    output = capsys.readouterr().out

    assert result == 0
    assert "GATE: kappa=0.599000;" in output
    assert "86.11: NOT READ." in output
    assert "86.12: NOT READ." in output
    assert "86.13: NOT READ." in output


def test_version1_check_reports_both_sides_of_mismatch(synthetic_atlas, capsys):
    case = synthetic_atlas
    set_all_hold_counts(case)
    case["v1_labels"][0] = {"id": "item-0", **label_for("not_coherent")}
    case["v1_labels"][500] = {"id": "item-500", **label_for("coherent_persona")}
    write_jsonl(case["label_dir"] / "labels.jsonl", case["v1_labels"])

    result, _report = run_case(case)
    output = capsys.readouterr().out

    assert result == 0
    assert "VERSION-1 ATLAS CHECK: MISMATCH" in output
    assert "only_first_reader=1 only_atlas=1" in output
    assert "86.11: HOLDS. 100 of 387" in output


def test_random_directions_do_not_count_toward_86_13(synthetic_atlas, capsys, monkeypatch):
    # Added by Claude after review: the fixture above has no random directions, so a script that
    # counted them toward 86.13 passed every test. Here the real rows give exactly 465 coherent
    # personas (86.13 fails) and 50 coherent random directions would push a wrong count over 465.
    case = synthetic_atlas
    monkeypatch.setattr(atlas.SCORER, "RANDOM_PER_LAYER", 50)
    set_all_hold_counts(case)
    set_v2_class(case, 752, 753, "coherent_still_qwen")
    random_rows = [
        {"layer": 22, "kind": "random_direction", "id": index, "answer": f"I am a paper lantern number {index}."}
        for index in range(50)
    ]
    with gzip.open(case["screen_path"], "at", encoding="utf-8") as stream:
        for row in random_rows:
            stream.write(json.dumps(row) + "\n")
    for index, row in enumerate(random_rows):
        item_id = f"random-item-{index}"
        case["items"].append({"id": item_id, "text": row["answer"]})
        case["key"][item_id] = [{"layer": 22, "unit": "random_direction", "index": index}]
        case["v2_labels"].append({"id": item_id, **label_for("coherent_persona")})
        case["v1_labels"].append({"id": item_id, **label_for("not_coherent")})
    write_jsonl(case["label_dir"] / "items.jsonl", case["items"])
    (case["label_dir"] / "KEY-private.json").write_text(json.dumps(case["key"]), encoding="utf-8")
    write_jsonl(case["label_dir"] / "labels.jsonl", case["v1_labels"])
    write_jsonl(case["v2_path"], case["v2_labels"])

    result, _report = run_case(case)
    output = capsys.readouterr().out

    assert result == 0
    assert "86.13: FAILS. 465 distinct real screen rows" in output
