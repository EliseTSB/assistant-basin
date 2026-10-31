#!/usr/bin/env python3
"""Fixture tests for Part 128's reader parsing, carry classes and scoring. Written by Claude (Opus 5.5), 2026-10-30."""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import label128
import score128
import join128


def test_parse():
    assert label128.parse('thinking {"fire_side": 2, "write_side": 0} done') == {"fire_side": 2, "write_side": 0}
    try:
        label128.parse('{"fire_side": 7, "write_side": 0}')
    except ValueError:
        pass
    else:
        raise AssertionError("a score outside 0-3 must be refused")


def test_classes():
    assert [join128.carry_class(n) for n in (0, 1, 2, 3, 6)] == ["other", "once", "other", "carrier", "carrier"]


def test_prompt_never_names_the_arm():
    item = {"id": "p00", "arm": "pushed", "name": "Hotel", "desc": "A hotel.",
            "card": {"fire": {"weights": "a", "drivers": "b", "contexts": ["c"], "label": "d"}, "write": {"weights": "e", "label": "f"}}}
    text = label128.prompt(item)
    assert "pushed" not in text and "p00" not in text and "Hotel" in text


def test_score_counts_and_gate():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        units = [{"id": f"p{i}", "L": 10, "class": "carrier" if i < 2 else "once", "c": 100.0,
                  "pushed": {"act_top": 10.0, "dark": i == 0}} for i in range(4)]
        (d / "units128.json").write_text(json.dumps(units))
        rows = []
        for u in units:
            rows.append({"id": u["id"], "arm": "pushed", "fire_side": 2 if u["class"] == "carrier" else 0, "write_side": 0})
            for arm in ("mismatch", "opposite", "control"):
                rows.append({"id": u["id"], "arm": arm, "fire_side": 0, "write_side": 1})
        for k in range(6):
            rows.append({"id": f"k{k}", "arm": "known-true", "fire_side": 3, "write_side": 0})
            rows.append({"id": f"k{k}", "arm": "known-false", "fire_side": 0, "write_side": 0})
        for name in ("labels-ds.jsonl", "labels-luna.jsonl"):
            (d / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        score128.main(d)
        out = json.load(open(d / "score128.json"))
        assert out["ds"]["pushed"] == 2 and out["ds"]["carrier"] == 2 and out["ds"]["once"] == 0
        assert out["ds"]["mismatch"] == 0 and out["ds"]["control"] == 0 and out["ds"]["opposite"] == 0
        assert out["ratio_median"] == 10.0 and out["dark"] == 1
        assert all(out["gate"].values()), out["gate"]
        assert abs(out["expected_dark"] - 4 * .72) < 1e-9


def test_gate_fails_when_rows_are_missing():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        units = [{"id": "p0", "L": 10, "class": "once", "c": 1.0, "pushed": {"act_top": 1.0, "dark": False}}]
        (d / "units128.json").write_text(json.dumps(units))
        rows = [{"id": "p0", "arm": "pushed", "fire_side": 0, "write_side": 0}]
        for name in ("labels-ds.jsonl", "labels-luna.jsonl"):
            (d / name).write_text(json.dumps(rows[0]) + "\n")
        score128.main(d)
        out = json.load(open(d / "score128.json"))
        assert out["gate"]["complete"] is False and out["gate"]["ds"] is False


for name, fn in list(globals().items()):
    if name.startswith("test_"):
        fn()
        print("ok", name)
