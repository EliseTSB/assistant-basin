"""Fixture-only tests for Part 86's triage miss-rate sample."""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common86
import miss86
import triage86


def _answer(category: str, population: str, ordinal: int) -> str:
    if category == "kept":
        return f"I am an AI assistant describing a {population} fixture number {ordinal}."
    if category == "broken":
        return "red blue"
    return "I am a quiet traveler who watches the sea each morning."


def make_screen(screen_dir: Path, per_class_per_layer: int = 25) -> dict[tuple[int, str, int], dict]:
    screen_dir.mkdir(parents=True, exist_ok=True)
    by_identity = {}
    for layer in range(10, 27):
        rows = []
        for population, kind in (("real", "neuron"), ("random", "random_direction")):
            unit = 0
            for category in ("kept", "broken"):
                for ordinal in range(per_class_per_layer):
                    row = {
                        "layer": layer,
                        "kind": kind,
                        "id": unit,
                        "answer": _answer(category, population, ordinal),
                    }
                    unit += 1
                    rows.append(row)
                    by_identity[(layer, kind, row["id"])] = row
            changed = {
                "layer": layer,
                "kind": kind,
                "id": unit,
                "answer": _answer("changed", population, 0),
            }
            rows.append(changed)
            by_identity[(layer, kind, unit)] = changed
        path = screen_dir / f"layer_{layer}.jsonl"
        encoded = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
        if layer == 10:
            with gzip.open(path.with_suffix(".jsonl.gz"), "wb") as stream:
                stream.write(encoded)
        else:
            path.write_bytes(encoded)
    return by_identity


def _keyed_membership(key: dict) -> set[tuple[str, str, int, int]]:
    return {
        (metadata["population"], metadata["class"], metadata["layer"], metadata["unit"])
        for metadata in key["items"].values()
    }


def _read_items(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_prepare_is_seeded_disjoint_correct_and_blind(tmp_path: Path):
    screen = tmp_path / "screen"
    source = make_screen(screen)
    item_path = tmp_path / "items-a" / "miss-items.jsonl"
    key_path = tmp_path / "private-a" / "miss-key.json"
    miss86.prepare(screen, item_path, key_path)
    items = _read_items(item_path)
    key = json.loads(key_path.read_text(encoding="utf-8"))

    assert key["seed"] == common86.MISS_SAMPLE_SEED == 8686
    assert key["requested_per_sample"] == 200
    assert len(items) == 800
    assert len(key["items"]) == len(items)
    assert key["items_sha256"] == hashlib.sha256(item_path.read_bytes()).hexdigest()
    assert key_path.parent != item_path.parent

    # Check every public field: only a fresh opaque id and the answer are visible to the reader.
    ids = set()
    for item in items:
        assert set(item) == {"id", "text"}
        assert isinstance(item["id"], str) and item["id"]
        assert isinstance(item["text"], str)
        assert item["id"] not in ids
        ids.add(item["id"])
        assert not any(secret_field in item for secret_field in ("layer", "unit", "arm", "class", "population", "triage"))

    samples = {name: set() for name in miss86.SAMPLE_SPECS}
    for item_id, metadata in key["items"].items():
        assert set(metadata) == {"population", "class", "layer", "unit"}
        population, category = metadata["population"], metadata["class"]
        sample_name = f"{population}_{category}"
        identity = (metadata["layer"], "neuron" if population == "real" else "random_direction", metadata["unit"])
        assert identity in source
        assert triage86.triage_answer(source[identity]["answer"]) == category
        samples[sample_name].add(identity)
    for name, members in samples.items():
        assert len(members) == 200
        assert {layer for layer, _kind, _unit in members} == set(range(10, 27))
    for left, left_members in samples.items():
        for right, right_members in samples.items():
            if left < right:
                assert left_members.isdisjoint(right_members)

    # A second run has fresh ids, but seed 8686 picks the same hidden screen rows.
    other_items = tmp_path / "items-b" / "miss-items.jsonl"
    other_key = tmp_path / "private-b" / "miss-key.json"
    miss86.prepare(screen, other_items, other_key)
    other = json.loads(other_key.read_text(encoding="utf-8"))
    assert _keyed_membership(key) == _keyed_membership(other)
    assert set(key["items"]) != set(other["items"])


def test_prepare_takes_everything_when_a_class_has_fewer_than_200(tmp_path: Path):
    screen = tmp_path / "screen"
    make_screen(screen, per_class_per_layer=1)
    item_path = tmp_path / "items" / "miss-items.jsonl"
    key_path = tmp_path / "private" / "miss-key.json"
    result = miss86.prepare(screen, item_path, key_path)
    key = json.loads(key_path.read_text(encoding="utf-8"))
    for name, details in result["samples"].items():
        assert details["available"] == 17
        assert details["sampled"] == 17
        assert details["shortfall"] is True
        assert "all 17 were included" in details["shortfall_note"]
        assert len([info for info in key["items"].values()
                    if f"{info['population']}_{info['class']}" == name]) == 17


def test_prepare_refuses_key_in_items_folder_or_repository(tmp_path: Path):
    screen = tmp_path / "screen"
    make_screen(screen, per_class_per_layer=1)
    items = tmp_path / "items" / "miss-items.jsonl"
    with pytest.raises(ValueError, match="outside the folder"):
        miss86.prepare(screen, items, items.parent / "miss-key.json")
    with pytest.raises(ValueError, match="outside the Assistant Basin repository"):
        miss86.prepare(screen, items, miss86.ROOT / "miss86-test-private-key.json")
    assert not items.exists()


def _score_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    items_path = tmp_path / "items" / "items.jsonl"
    key_path = tmp_path / "private" / "key.json"
    labels_path = tmp_path / "labels" / "labels.jsonl"
    items = []
    labels = []
    private_items = {}
    samples = {}
    expected_kinds = (
        (False, "human"),
        (True, "unclear"),
        (True, "still Qwen or an assistant"),
        (True, "animal"),
    )
    next_unit = {"real": 0, "random": 0}
    for sample_name, (population, category) in miss86.SAMPLE_SPECS.items():
        samples[sample_name] = {
            "population": population,
            "class": category,
            "requested": 200,
            "available": 4,
            "sampled": 4,
        }
        for index, (coherent, kind) in enumerate(expected_kinds):
            item_id = f"{sample_name}-{index}"
            items.append({"id": item_id, "text": "A fixture answer with a named identity."})
            labels.append({"id": item_id, "persona": f"fixture {kind}", "kind": kind, "coherent": coherent})
            private_items[item_id] = {
                "population": population,
                "class": category,
                "layer": 10 + index,
                "unit": next_unit[population],
            }
            next_unit[population] += 1
    items_bytes = miss86._jsonl_bytes(items)
    items_path.parent.mkdir(parents=True)
    items_path.write_bytes(items_bytes)
    key_path.parent.mkdir(parents=True)
    key_path.write_text(json.dumps({
        "format_version": 1,
        "seed": 8686,
        "requested_per_sample": 200,
        "items_sha256": hashlib.sha256(items_bytes).hexdigest(),
        "samples": samples,
        "items": private_items,
    }), encoding="utf-8")
    labels_path.parent.mkdir(parents=True)
    labels_path.write_bytes(miss86._jsonl_bytes(labels))
    md_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"
    return items_path, key_path, labels_path, md_path, json_path


def test_score_reuses_registered_class_mapping_and_reports_wilson(tmp_path: Path, monkeypatch):
    items, key, labels, _md, _json = _score_fixture(tmp_path)
    original = miss86.score86_predictions.label_class
    calls = []

    def tracked(row):
        calls.append(row)
        return original(row)

    monkeypatch.setattr(miss86.score86_predictions, "label_class", tracked)
    report = miss86.score(items, key, labels)
    assert len(calls) == 16
    assert report["verdict_effect"] == "none"
    for sample in report["samples"].values():
        assert sample["coherent_personas"] == 1
        assert sample["coherent_persona_share"] == 0.25
        assert sample["version_2_class_counts"] == {
            "not_coherent": 1,
            "unclear": 1,
            "coherent_still_qwen": 1,
            "coherent_persona": 1,
        }
        assert sample["shortfall"] is True
        assert "all 4 were included" in sample["shortfall_note"]

    md = miss86.render_markdown(report)
    assert "changes no verdict" in md
    assert "Wilson 95%" in md


def test_wilson_intervals_match_known_counts():
    low, high = miss86.wilson_interval(50, 100)
    assert low == pytest.approx(0.4038315, abs=1e-6)
    assert high == pytest.approx(0.5961685, abs=1e-6)
    assert miss86.wilson_interval(0, 0) is None


def test_missing_label_stops_before_opening_key(tmp_path: Path):
    items, _key, labels, _md, _json = _score_fixture(tmp_path)
    rows = _read_items(labels)
    labels.write_bytes(miss86._jsonl_bytes(rows[:-1]))
    absent_key = tmp_path / "private" / "must-not-open.json"
    with pytest.raises(miss86.PreflightError, match="missing 1 label"):
        miss86.score(items, absent_key, labels)
    assert not absent_key.exists()


def test_error_label_stops_before_opening_key(tmp_path: Path):
    items, _key, labels, _md, _json = _score_fixture(tmp_path)
    rows = _read_items(labels)
    rows[0] = {"id": rows[0]["id"], "error": "fixture reader failure"}
    labels.write_bytes(miss86._jsonl_bytes(rows))
    absent_key = tmp_path / "private" / "must-not-open.json"
    with pytest.raises(miss86.PreflightError, match="error row"):
        miss86.score(items, absent_key, labels)
    assert not absent_key.exists()


def test_label_wrapper_uses_ds_v2_with_fake_transport_and_usage_sidecar(tmp_path: Path, monkeypatch):
    items = tmp_path / "items.jsonl"
    items.write_text(json.dumps({"id": "fixture-id", "text": "I am a quiet traveler."}) + "\n", encoding="utf-8")
    out = tmp_path / "labels-v2.jsonl"
    usage = tmp_path / "labels-v2.usage.jsonl"
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "fixture-key")
    calib86 = miss86._load_calib86()
    monkeypatch.setattr(miss86, "_load_calib86", lambda: calib86)
    calls = []

    def fake_call_prompt(prompt, **settings):
        calls.append((prompt, settings))
        return {
            "text": json.dumps({
                "id": "fixture-id",
                "persona": "quiet traveler",
                "kind": "human",
                "coherent": True,
            }),
            "usage": {"input_tokens": 10, "output_tokens": 4},
            "reasoning_tokens": 0,
            "stop_reason": "stop",
            "served": "deepseek-v4.1-flash",
        }

    monkeypatch.setattr(calib86.judge_client, "call_prompt", fake_call_prompt)
    result = miss86.label(items, out, usage, workers=1)
    assert result["attempted_items"] == 1
    assert len(calls) == 1
    prompt, settings = calls[0]
    assert "Part 86 screen reader brief, version 2" in prompt
    assert "fixture-id" in prompt and "I am a quiet traveler." in prompt
    assert settings == {
        "model": "deepseek-v4.1-flash",
        "route": "ocgo",
        "session": "part86-calib-ds",
        "max_tokens": 8000,
        "timeout": 300,
    }
    assert json.loads(out.read_text(encoding="utf-8"))["kind"] == "human"
    assert json.loads(usage.read_text(encoding="utf-8"))["id"] == "fixture-id"
