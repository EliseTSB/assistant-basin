from __future__ import annotations

import hashlib
import json
import math
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (HERE, ROOT / "part119", ROOT / "part119b", ROOT / "part92", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import common92
import rows119
import rows119b
import run119
import run119b
import run119c
import rows119c
import score119c


def _metric_position(share: float, n_pair: float = 10.0) -> dict:
    return {"pos": 0, "nI": math.sqrt(share) * n_pair, "nPair": n_pair,
            "nA": 3.0, "n2": 4.0}


def _record(kind: str, first: str, second: str, dose: float,
            share: float = .04, n_pair: float = 10.0) -> dict:
    return {"key": rows119b.path2_key(kind, dose, first, second),
            "kind": kind, "first": first, "second": second, "dose": float(dose),
            "p_A": 0, "positions": [_metric_position(share, n_pair)]}


def _two_unit_rows():
    units = [{"id": "a"}, {"id": "b"}]
    return units, rows119b.all_path2_rows(units)


def test_row_draw_is_seeded_and_has_312_rows():
    units = rows119.load_units()
    first = rows119c.draw_path2_rows(units)
    second = rows119c.draw_path2_rows(units)
    assert first == second
    assert len(first) == 312
    assert len({row["first"] for row in first}) == 39
    assert Counter(row["kind"] for row in first) == {"real": 156, "control": 156}
    for unit in units:
        sample = [row for row in first if row["first"] == unit["id"]]
        assert Counter(row["kind"] for row in sample) == {"real": 4, "control": 4}
        assert {row["second"] for row in sample if row["kind"] == "real"} == \
            {row["second"] for row in sample if row["kind"] == "control"}
    digest = hashlib.sha256("\n".join(
        f"{row['kind']}:{row['first']}+{row['second']}" for row in first
    ).encode()).hexdigest()
    assert digest == "d5ef07fcd185c292ed48e481e8b025e3f3c0a7f7594f03f7ac46a5413e22a46b"


def test_gate1_reproduces_only_registered_doses_and_checks_every_row():
    units, selected = _two_unit_rows()
    old, new = [], []
    for row in selected:
        for dose in (.2, .4, .8):
            old.append(_record(row["kind"], row["first"], row["second"], dose, .04, 10.0))
            new.append(_record(row["kind"], row["first"], row["second"], dose, .04, 10.04))
    # A large discrepancy at .1 is outside Gate 1 and must not affect it.
    for row in selected:
        new.append(_record(row["kind"], row["first"], row["second"], .1, .04, 100.0))
    gate = score119c.gate1_result(new, old, selected)
    assert gate["passed"]
    assert set(gate["by_dose"]) == {"0.2", "0.4", "0.8"}


def test_gate1_fails_when_a_registered_dose_is_wrong_even_if_01_matches():
    _units, selected = _two_unit_rows()
    old, new = [], []
    for row in selected:
        for dose in (.2, .4, .8):
            old.append(_record(row["kind"], row["first"], row["second"], dose, .04, 10.0))
            new_n = 10.2 if dose == .2 else 10.0
            new.append(_record(row["kind"], row["first"], row["second"], dose, .04, new_n))
        new.append(_record(row["kind"], row["first"], row["second"], .1, .04, 10.0))
    gate = score119c.gate1_result(new, old, selected)
    assert not gate["passed"]
    assert not gate["by_dose"]["0.2"]["passed"]


def test_gate2_checks_08_and_uses_the_five_percent_limit():
    _units, selected = _two_unit_rows()
    bf16, float32 = [], []
    for row in selected:
        bf16.append(_record(row["kind"], row["first"], row["second"], .8, .04, 10.0))
        float32.append(_record(row["kind"], row["first"], row["second"], .8, .04, 10.4))
        float32.append(_record(row["kind"], row["first"], row["second"], .4, .04, 30.0))
    gate = score119c.gate2_result(float32, bf16, selected)
    assert gate["passed"]
    assert set(gate["by_dose"]) == {"0.8"}
    float32 = [row for row in float32 if row["dose"] == .8]
    float32.extend(_record(row["kind"], row["first"], row["second"], .8, .04, 10.6)
                   for row in selected)
    with pytest.raises(ValueError, match="duplicate Path 2 row/dose"):
        score119c.gate2_result(float32, bf16, selected)


def _prediction_fixture(*, low=(.01, .04, .16), mid=(.2, .3),
                        control_scale=1.0, bf16_02=.17):
    units = ["a", "b", "c"]
    pairs = [("a", "b"), ("a", "c"), ("b", "a"), ("b", "c")]
    bf16_rows, float32_rows = [], []
    values = dict(zip((.05, .1, .2), low))
    values.update({.4: mid[0], .8: mid[1]})
    bf_values = dict(values)
    bf_values[.2] = bf16_02
    for dose in rows119c.DOSES:
        real_share = values[dose]
        for kind, scale in (("real", 1.0), ("control", control_scale)):
            for first, second in pairs:
                fshare = real_share * scale
                bshare = (bf_values[dose] if kind == "real" else bf_values[dose] * scale)
                float32_rows.append(_record(kind, first, second, dose, fshare))
                bf16_rows.append(_record(kind, first, second, dose, bshare))
    table, row_values = score119c.summarize({"bf16": bf16_rows, "float32": float32_rows})
    predictions = score119c.predictions(table, row_values, units)
    return units, bf16_rows, float32_rows, table, row_values, predictions


def test_each_registered_prediction_on_fixtures():
    _units, _bf16, _float32, _table, _values, good = _prediction_fixture()
    assert [row["verdict"] for row in good] == ["holds"] * 4

    _u, _b, _f, _t, _v, pred1_bad = _prediction_fixture(low=(.01, .04, .16), bf16_02=.23)
    assert pred1_bad[0]["verdict"] == "does_not_hold"
    _u, _b, _f, _t, _v, pred2_bad = _prediction_fixture(low=(.06, .11, .16))
    assert pred2_bad[1]["verdict"] == "does_not_hold"
    _u, _b, _f, _t, _v, pred3_bad = _prediction_fixture(low=(.04, .05, .06))
    assert pred3_bad[2]["verdict"] == "does_not_hold"
    _u, _b, _f, _t, _v, pred4_bad = _prediction_fixture(control_scale=.5)
    assert pred4_bad[3]["verdict"] == "does_not_hold"


def test_prediction3_fits_only_005_01_02_not_the_higher_doses():
    _units, _bf16, _float32, _table, _values, pred = _prediction_fixture(
        low=(.01, .04, .16), mid=(.0001, .00001))
    assert pred[2]["observed"]["log_log_slope"] == pytest.approx(2.0)
    assert pred[2]["verdict"] == "holds"


def _passing_gates():
    return {"gate1": {"passed": True}, "gate2": {"passed": True}}


def test_headline_branches_and_first_requires_119c2():
    predictions = [
        {"id": "119c.1", "verdict": "holds", "observed": {}},
        {"id": "119c.2", "verdict": "holds", "observed": {}},
    ]
    assert "not a rounding floor" in score119c.headline_by_rule(predictions, _passing_gates())
    predictions[1]["verdict"] = "does_not_hold"
    assert score119c.headline_by_rule(predictions, _passing_gates()) == \
        "No registered headline reading; report the numbers."
    predictions[0] = {"id": "119c.1", "verdict": "does_not_hold",
                      "observed": {"median_difference_float32_minus_bf16": -.05,
                                   "bootstrap": {"ci95_product_weighted_mean_difference": [-.08, -.02]}}}
    assert "bf16 rounding inflates" in score119c.headline_by_rule(predictions, _passing_gates())
    assert "Stopped by gate1" in score119c.headline_by_rule(
        predictions, {"gate1": {"passed": False}, "gate2": {"passed": True}})


def _tiny_olmo2():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Olmo2Config(
        vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32,
        pad_token_id=0, eos_token_id=2)
    return torch, transformers.Olmo2ForCausalLM(config).eval()


def _tiny_units():
    return [
        {"id": "tiny-a", "layer": 0, "neuron": 0, "control_seed": 11901},
        {"id": "tiny-b", "layer": 1, "neuron": 1, "control_seed": 11902},
    ]


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    unk_token_id = None

    def apply_chat_template(self, *args, **kwargs):
        return "tiny prompt"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1, 3, 5]}

    def decode(self, ids, **kwargs):
        return "tiny answer"


def test_float32_push_cache_uses_float32_weights():
    torch, model = _tiny_olmo2()
    model = model.to(dtype=torch.float32)
    layers = common92.get_layers(model)
    units = _tiny_units()
    _base, caches = run119c.build_dtype_push_caches(
        units, layers, {0: 100.0, 1: 100.0})
    assert all(value.dtype == torch.float32 for cache in caches.values()
               for key, value in cache.items() if key != "_units")


def test_run_path2_tiny_olmo2_five_doses_in_both_dtypes(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(run119, "first_flip", lambda _single, _plain: (1, 9, 5))
    units = _tiny_units()
    rows = rows119b.all_path2_rows(units)
    answers = {"plain": {"tokens": [4, 5, 6, 7]},
               "single@0.8:tiny-a": {"tokens": [4, 9, 6, 7]},
               "single@0.8:tiny-b": {"tokens": [4, 9, 6, 7]}}
    for dtype_name, dtype in (("bf16", torch.bfloat16), ("float32", torch.float32)):
        torch.manual_seed(11906)
        model = _tiny_olmo2()[1].to(dtype=dtype).eval()
        layers = common92.get_layers(model)
        base, caches = run119c.build_dtype_push_caches(
            units, layers, {0: 100.0, 1: 100.0})
        assert all(value.dtype == dtype for cache in caches.values()
                   for key, value in cache.items() if key != "_units")
        output = tmp_path / f"interaction-{dtype_name}.jsonl"
        result = run119b.run_path2(
            units, rows, rows119c.DOSES, output, _TinyTokenizer(), model, layers,
            base, caches, answers, "cpu", 2, quiet=True)
        records = [json.loads(line) for line in output.read_text().splitlines()]
        assert result["rows_new"] == len(rows) * len(rows119c.DOSES)
        assert len(records) == len(rows) * 5
        assert {row["dose"] for row in records} == set(rows119c.DOSES)
        assert all(position["t_ids"] for row in records if row["dose"] == .8
                   for position in row["positions"])


def test_gate_rejects_a_missing_selected_row():
    _units, selected = _two_unit_rows()
    reference = [_record(row["kind"], row["first"], row["second"], dose)
                 for row in selected for dose in (.2, .4, .8)]
    current = [row for row in reference if not (
        row["kind"] == "real" and row["first"] == "a" and row["second"] == "b" and row["dose"] == .2)]
    gate = score119c.gate1_result(current, reference, selected)
    assert not gate["passed"]
    assert gate["missing_row_count"] == 1


def test_runner_and_scorer_import_from_exact_copy_list(tmp_path):
    notes = (HERE / "BUILD-NOTES.md").read_text(encoding="utf-8")
    copy_lists = [shlex.split(line.split("--copy", 1)[1].strip())
                  for line in notes.splitlines() if line.strip().startswith("--copy ")]
    assert len(copy_lists) == 1
    copied_root = tmp_path / "copy-list-tree"
    for relative in copy_lists[0]:
        source = ROOT / relative
        destination = copied_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    assert sorted(str(path.relative_to(copied_root))
                  for path in copied_root.rglob("*") if path.is_file()) == sorted(copy_lists[0])
    for script in ("part119c/run119c.py", "part119c/score119c.py"):
        result = subprocess.run(
            ["/home/elisepc/neuron-probe/.venv/bin/python", "-I", "-B", script, "--help"],
            cwd=copied_root, text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "usage:" in result.stdout.lower()
