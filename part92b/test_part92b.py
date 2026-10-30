"""Offline acceptance tests for Part 92b. All reader calls use fake transports."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part92"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "part92b"))

import label92  # noqa: E402
import label92b  # noqa: E402
import score92b  # noqa: E402

STEP2 = ROOT / "results/part92/step2"


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label_row(key: tuple[str, str, int], shows: bool) -> dict:
    return {
        "id": key[0], "unit": key[1], "question_index": key[2],
        "shows_persona": shows, "self_claim": False,
        "speaker": "none", "coherent": True,
    }


def fixture_metrics() -> dict:
    return {
        "gate_a_passed": True,
        "gate_b_kappa": 0.70,
        "ds_haiku_kappa": 0.70,
        "unchanged_count": 30,
        "changed_count": 9,
        "slot_changes": 6,
        "neuron_shows": 105,
        "random_shows": 31,
        "random_total": 318,
    }


def test_prompt_is_exact_part92_fill_and_does_not_receive_metadata():
    answers = label92b.read_jsonl(STEP2 / "answers/instruct_chat.jsonl")
    neurons = json.loads((STEP2 / "neurons.json").read_text(encoding="utf-8"))
    item = label92b.make_items(answers, neurons)[0]
    item = {**item, "arm": "ARM-MUST-NOT-LEAK"}
    item["public"] = {**item["public"], "unit": "UNIT-MUST-NOT-LEAK"}
    expected = label92.PROMPT
    for field in ("name", "desc", "q", "a"):
        expected = expected.replace("{" + field + "}", str(item[field]))
    prompt = label92b.build_prompt(item)
    assert prompt == expected
    assert "ARM-MUST-NOT-LEAK" not in prompt
    assert "UNIT-MUST-NOT-LEAK" not in prompt


def test_sample_is_stratified_deterministic_and_blind():
    answers = label92b.read_jsonl(STEP2 / "answers/instruct_chat.jsonl")
    neurons = json.loads((STEP2 / "neurons.json").read_text(encoding="utf-8"))
    keys = [label92b.item_key(row) for row in answers]
    first = label92b.sample_keys(keys)
    assert first == label92b.sample_keys(keys)
    assert len(first) == 96
    assert Counter(unit for _, unit, _ in first) == {"neuron": 48, "random": 48}
    by_key = {label92b.item_key(item["public"]): item
              for item in label92b.make_items(answers, neurons)}
    prompts = [label92b.build_prompt(by_key[key]) for key in first]
    assert len(prompts) == 96
    assert all("neuron" not in prompt and "random" not in prompt for prompt in prompts)


def test_gate_a_minimum_is_11_of_12():
    known = label92b.read_known()
    rows = []
    for index, item in enumerate(known):
        expected = item["expected"] if index < 11 else not item["expected"]
        rows.append(label_row((f"gate-a-{index:02d}", "gate_a", index % 6), expected))
    assert label92b.gate_result(known, rows)["passed"] is True
    rows[-2]["shows_persona"] = not known[-2]["expected"]
    assert label92b.gate_result(known, rows)["passed"] is False


def test_main_read_refuses_missing_or_failed_gate_a(tmp_path, monkeypatch):
    monkeypatch.setattr(label92b, "ROOT", tmp_path)
    gate_path = tmp_path / "results/part92b/gate-a-ds.jsonl"
    with pytest.raises(RuntimeError, match="Gate A is missing"):
        label92b._check_gate_before_main("ds")

    known = label92b.read_known()
    bad = [label_row((f"gate-a-{i:02d}", "gate_a", i % 6),
                     item["expected"] if i < 10 else not item["expected"])
           for i, item in enumerate(known)]
    label92b.write_jsonl(gate_path, bad)
    with pytest.raises(RuntimeError, match=r"Gate A is failed \(10/12\)"):
        label92b._check_gate_before_main("ds")


def test_resume_retries_only_error_rows_and_records_usage(tmp_path):
    success = {
        "public": {"id": "p92-test-00", "unit": "neuron", "question_index": 0},
        "name": "paperclip", "desc": "a small bent wire", "q": "Who are you?", "a": "I am a paperclip.",
    }
    retry = {
        "public": {"id": "p92-test-01", "unit": "neuron", "question_index": 1},
        "name": "ruler", "desc": "a marked measuring tool", "q": "Tell me about yourself.",
        "a": "I am a ruler.",
    }
    output = tmp_path / "labels.jsonl"
    label92b.write_jsonl(output, [
        {**success["public"], "shows_persona": True, "self_claim": True,
         "speaker": "other", "coherent": True},
        {**retry["public"], "error": "ValueError('readable prior error')"},
    ])
    calls = []

    def fake_request(reader, prompt, unused_key):
        calls.append((reader, prompt, unused_key))
        return {
            "text": '{"shows_persona": true, "self_claim": true, "speaker": "other", "coherent": true}',
            "usage": {"input_tokens": 17, "output_tokens": 21},
            "stop_reason": "end_turn", "served": "fake-reader", "reasoning_tokens": 8,
        }

    result = label92b.label_items([success, retry], output, "ds", 1,
                                  request_fn=fake_request, sleep_fn=lambda _: None)
    assert result["pending"] == 1
    assert len(calls) == 1
    assert calls[0][0] == "ds"
    assert "I am a ruler." in calls[0][1]
    final = label92b._latest_rows(output)
    assert label92b.valid_label(final[label92b.item_key(retry["public"])])
    usage = jsonl(Path(str(output) + ".usage.jsonl"))
    assert len(usage) == 1
    assert usage[0]["usage"]["input_tokens"] == 17
    assert usage[0]["served"] == "fake-reader"
    assert usage[0]["reasoning_tokens"] == 8


@pytest.mark.parametrize("changed,expected", [(0, "held"), (1, "held"), (2, "failed"), (3, "failed"), (4, "failed")])
def test_gate_a_threshold_fixture_scores_correctly(changed, expected):
    known = label92b.read_known()
    correct_count = 12 - changed
    rows = [label_row((f"gate-a-{i:02d}", "gate_a", i % 6),
                      item["expected"] if i < correct_count else not item["expected"])
            for i, item in enumerate(known)]
    result = label92b.gate_result(known, rows)
    assert result["passed"] is (expected == "held")


def test_each_registered_verdict_has_hold_and_fail_fixtures():
    base = fixture_metrics()
    held = score92b.registered_verdicts(**base)
    assert {key: value["verdict"] for key, value in held.items()} == {
        "92b.1": "held", "92b.2": "held", "92b.3": "held", "92b.4": "held", "92b.5": "held",
    }
    failing_fixtures = {
        "92b.1": {"ds_haiku_kappa": 0.59},
        "92b.2": {"unchanged_count": 29},
        "92b.3": {"slot_changes": 5},
        "92b.4": {"neuron_shows": 104},
        "92b.5": {"random_shows": 32},
    }
    for verdict, changes in failing_fixtures.items():
        fixture = {**base, **changes}
        assert score92b.registered_verdicts(**fixture)[verdict]["verdict"] == "failed"


def test_gate_b_below_point_six_blocks_every_prediction():
    fixture = {**fixture_metrics(), "gate_b_kappa": 0.59}
    verdicts = score92b.registered_verdicts(**fixture)
    assert {row["verdict"] for row in verdicts.values()} == {"not read (Gate B)"}


def test_gate_a_failure_blocks_predictions_and_92b3_waits_for_three_changes():
    failed_gate = {**fixture_metrics(), "gate_a_passed": False}
    assert {row["verdict"] for row in score92b.registered_verdicts(**failed_gate).values()} == {"not read (Gate A)"}
    fewer_changes = {**fixture_metrics(), "changed_count": 2, "slot_changes": 2}
    verdicts = score92b.registered_verdicts(**fewer_changes)
    assert verdicts["92b.3"]["verdict"] == "not read (fewer than 3 units changed)"


def test_kind_rule_reproduces_part92_on_all_39_classified_units():
    neuron_rows = json.loads((STEP2 / "neurons.json").read_text(encoding="utf-8"))
    kinds = json.loads((STEP2 / "kinds.json").read_text(encoding="utf-8"))["kinds"]
    labels, _ = score92b.load_labels(STEP2 / "labels-instruct_chat.jsonl")
    reproduced = score92b.classify_units(labels, [str(row["id"]) for row in neuron_rows])
    classified = {unit_id: kind for unit_id, kind in kinds.items() if kind in ("concept", "slot")}
    assert len(classified) == 39
    assert {unit_id: reproduced[unit_id] for unit_id in classified} == classified

