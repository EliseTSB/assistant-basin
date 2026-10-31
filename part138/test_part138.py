"""Small CPU-only contract tests for part138 control logic; no model is loaded."""
import json

from ladder138 import adapter_usage, parse_counter_json, should_stop, should_stop_for_spill, write_json


def test_spill_rule_requires_strictly_more_than_one_gb_over_start():
    assert not should_stop_for_spill(10.0, 10.0 + 1_000_000_000)
    assert should_stop_for_spill(10.0, 10.0 + 1_000_000_001)
    assert not should_stop_for_spill(None, 2_000_000_000)


def test_stop_rule_stops_for_spill_or_generation_error():
    assert should_stop({"spill": True, "generation_error": None})
    assert should_stop({"spill": False, "generation_error": "failed"})
    assert not should_stop({"spill": False, "generation_error": None})


def test_json_writer_replaces_artifact_with_valid_incremental_payload(tmp_path):
    target = tmp_path / "nested" / "ladder.json"
    payload = {"rows": [{"batch_size": 2, "answers_per_s": 1.25}]}
    write_json(target, payload)
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert not target.with_name("ladder.json.tmp").exists()


def test_counter_parser_and_a770_adapter_maxima():
    raw = json.dumps([
        {"Path": "\\GPU Adapter Memory(A770)\\Dedicated Usage", "CookedValue": 123.0},
        {"Path": "\\GPU Adapter Memory(A770)\\Dedicated Usage", "CookedValue": 456.0},
        {"Path": "\\GPU Adapter Memory(A770)\\Shared Usage", "CookedValue": "789"},
        {"Path": "\\GPU Adapter Memory(other)\\Shared Usage", "CookedValue": 9999.0},
        {"Path": "\\Memory\\Available MBytes", "CookedValue": 4096.0},
    ])
    parsed = parse_counter_json(raw)
    assert len(parsed) == 5
    assert adapter_usage(parsed) == {"dedicated": 456.0, "shared": 789.0}

