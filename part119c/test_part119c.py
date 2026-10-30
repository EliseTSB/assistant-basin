from __future__ import annotations

import hashlib
import inspect
import json
import math
import shlex
import shutil
import subprocess
import sys
import textwrap
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

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
            share: float = .04, n_pair: float = 10.0,
            model_dtype: str | None = None) -> dict:
    record = {"key": rows119b.path2_key(kind, dose, first, second),
              "kind": kind, "first": first, "second": second, "dose": float(dose),
              "p_A": 0, "positions": [_metric_position(share, n_pair)]}
    if model_dtype is not None:
        record["model_dtype"] = model_dtype
        record["model_parameter_dtype"] = {
            "bf16": "torch.bfloat16", "float32": "torch.float32"}[model_dtype]
    return record


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
        bf16.append(_record(row["kind"], row["first"], row["second"], .8,
                            .04, 10.0, "bf16"))
        float32.append(_record(row["kind"], row["first"], row["second"], .8,
                               .04, 10.4, "float32"))
        float32.append(_record(row["kind"], row["first"], row["second"], .4,
                               .04, 30.0, "float32"))
    gate = score119c.gate2_result(float32, bf16, selected)
    assert gate["passed"]
    assert set(gate["by_dose"]) == {"0.8"}
    float32 = [row for row in float32 if row["dose"] == .8]
    float32.extend(_record(row["kind"], row["first"], row["second"], .8,
                           .04, 10.6, "float32")
                   for row in selected)
    with pytest.raises(ValueError, match="duplicate Path 2 row/dose"):
        score119c.gate2_result(float32, bf16, selected)


def test_gate2_requires_dtype_evidence_and_rejects_identical_npair_files():
    _units, selected = _two_unit_rows()
    bf16 = [_record(row["kind"], row["first"], row["second"], .8,
                    .04, 10.0, "bf16") for row in selected]
    copied_as_float32 = [{**row, "model_dtype": "float32",
                          "model_parameter_dtype": "torch.float32"} for row in bf16]
    identical_gate = score119c.gate2_result(copied_as_float32, bf16, selected)
    assert not identical_gate["passed"]
    assert identical_gate["all_0.8_nPair_identical_by_position"]

    true_float32 = [_record(row["kind"], row["first"], row["second"], .8,
                            .04, 10.2, "float32") for row in selected]
    mislabeled = [{**row, "model_dtype": "bf16",
                   "model_parameter_dtype": "torch.bfloat16"} for row in true_float32]
    dtype_gate = score119c.gate2_result(mislabeled, bf16, selected)
    assert not dtype_gate["passed"]
    assert not dtype_gate["dtype_evidence"]["float32_file"]["passed"]


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


def test_interval_bootstraps_difference_of_two_medians_with_both_weightings():
    units = [f"u{index}" for index in range(20)]
    paired = [{"first": units[index], "second": units[index + 1],
               "bf16_value": .1, "float32_value": .1}
              for index in range(0, 18, 2)]
    paired[-1]["float32_value"] = .4
    bootstrap = score119c._weighted_difference_ci(paired, units, draws=100)
    assert bootstrap["difference_of_median_row_shares"] == 0
    assert len(bootstrap["ci95_product_weighted_difference_of_median_row_shares"]) == 2
    assert len(bootstrap["ci95_sum_weighted_difference_of_median_row_shares"]) == 2


def test_summary_reports_share_of_positions_below_npair_one():
    low_pair, high_pair = _metric_position(.04, .5), _metric_position(.04, 1.5)
    high_pair["pos"] = 1
    rows = {dtype: [{"key": rows119b.path2_key("real", .05, "a", "b"),
                     "kind": "real", "first": "a", "second": "b",
                     "dose": .05, "positions": [low_pair, high_pair]}]
            for dtype in ("bf16", "float32")}
    table, _values = score119c.summarize(rows)
    summary = next(row for row in table if row["dtype"] == "float32"
                   and row["dose"] == .05 and row["kind"] == "real")
    assert summary["median_nPair"] == 1.0
    assert summary["median_nI"] == pytest.approx(.2)
    assert summary["positions_nPair_below_one"] == 1
    assert summary["share_positions_nPair_below_one"] == .5


def test_gate_failure_marks_the_per_dose_table_as_not_a_reading():
    result = {"selected_rows_expected": 0, "selected_rows_by_class": {},
              "gates": {"gate1": {"passed": False, "by_dose": {},
                                  "median_relative_nPair_difference_all_checked_rows": None},
                        "gate2": {"passed": True, "by_dose": {},
                                  "median_relative_nPair_difference_all_checked_rows": 0.0},
                        "all_passed": False},
              "predictions": [], "per_dose": [], "headline": "Stopped by gate1.",
              "readout": "shares", "bootstrap_definition": "bootstrap"}
    markdown = score119c.render_markdown(result)
    assert "## Per-dose summaries (no reading is made from this table because a gate failed)" in markdown


def test_each_registered_prediction_on_fixtures():
    _units, _bf16, _float32, _table, _values, good = _prediction_fixture()
    assert [row["verdict"] for row in good] == ["holds"] * 4
    assert good[0]["observed"]["median_difference_float32_minus_bf16"] == \
        good[0]["observed"]["bootstrap"]["difference_of_median_row_shares"]

    _u, _b, _f, _t, _v, pred1_bad = _prediction_fixture(low=(.01, .04, .16), bf16_02=.23)
    assert pred1_bad[0]["verdict"] == "does_not_hold"
    _u, _b, _f, _t, _v, pred2_bad = _prediction_fixture(low=(.06, .11, .16))
    assert pred2_bad[1]["verdict"] == "does_not_hold"
    _u, _b, _f, _t, _v, pred3_bad = _prediction_fixture(low=(.04, .05, .06))
    assert pred3_bad[2]["verdict"] == "does_not_hold"
    _u, _b, _f, _t, _v, pred4_bad = _prediction_fixture(control_scale=.5)
    assert pred4_bad[3]["verdict"] == "does_not_hold"


def test_prediction3_is_endpoint_slope_from_005_to_02_only():
    _units, _bf16, _float32, _table, _values, pred = _prediction_fixture(
        low=(.02, .01, .20), mid=(.0001, .00001))
    assert pred[2]["observed"]["log_log_slope"] == pytest.approx(math.log(10) / math.log(4))
    assert pred[2]["observed"]["doses"] == [.05, .2]
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
                                   "bootstrap": {"ci95_product_weighted_difference_of_median_row_shares": [-.08, -.02]}}}
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


def test_run_entry_writes_scores_and_restarts_float32_from_its_own_passes(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(run119, "first_flip", lambda _single, _plain: (1, 9, 5))
    units = _tiny_units()
    monkeypatch.setattr(run119c, "rows119",
                        SimpleNamespace(load_units=lambda *_args: units))
    root = tmp_path / "tiny-repo"
    root.mkdir()
    monkeypatch.setattr(run119c, "ROOT", root)
    out_dir = root / "results" / "part119c"
    answers = {"plain": {"tokens": [4, 5, 6, 7]},
               "single:tiny-a": {"tokens": [4, 9, 6, 7]},
               "single:tiny-b": {"tokens": [4, 9, 6, 7]}}
    monkeypatch.setattr(run119b, "read_jsonl_readonly", lambda *_args, **_kwargs: answers)

    active_dtype = {"value": None}

    def load_tiny_model(_snapshot, _device, dtype):
        torch.manual_seed(11906)
        tokenizer, model = _tiny_olmo2()
        model = model.to(dtype=dtype).eval()
        active_dtype["value"] = str(next(model.parameters()).dtype)
        return _TinyTokenizer(), model

    monkeypatch.setattr(run119c, "load_model", load_tiny_model)
    monkeypatch.setattr(run119c, "clear_device_cache", lambda _device: None)
    real_extract = run119b._extract_batch_metrics
    float32_t_sources = []

    def track_float32_t_source(*args, **kwargs):
        bound = inspect.signature(real_extract).bind(*args, **kwargs)
        if (active_dtype["value"] == "torch.float32"
                and float(bound.arguments["dose"]) == .8
                and bound.arguments["add_support"]):
            float32_t_sources.append(dict(bound.arguments["existing_t"]))
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(run119b, "_extract_batch_metrics", track_float32_t_source)
    args = SimpleNamespace(model="tiny-local-model", out=str(out_dir), device="cpu", batch=2,
                           neurons="unused", kinds="unused", kinds_ds="unused")

    first_report = run119c.run(args)
    bf16_path = out_dir / "interaction-bf16.jsonl"
    float32_path = out_dir / "interaction-float32.jsonl"

    def read_records(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    bf16_first, float32_first = read_records(bf16_path), read_records(float32_path)
    assert first_report["rows"] == len(rows119c.draw_path2_rows(
        units, rows119b.all_path2_rows(units)))
    assert [run["model_parameter_dtype"] for run in first_report["runs"]] == [
        "torch.bfloat16", "torch.float32"]
    for dtype, records in (("bf16", bf16_first), ("float32", float32_first)):
        assert {row["dose"] for row in records} == set(rows119c.DOSES)
        assert {row["model_dtype"] for row in records} == {dtype}
        assert all(position["t_ids"] for row in records if row["dose"] == .8
                   for position in row["positions"])

    def support_map(records):
        return {row["key"]: [position["t_ids"] for position in row["positions"]]
                for row in records if row["dose"] == .8}

    expected_float32_support = support_map(float32_first)
    bf16_path_contents = bf16_path.read_bytes()
    float32_path.unlink()
    assert bf16_path.exists()
    before_restart_sources = len(float32_t_sources)
    run119c.run(args)
    float32_restart = read_records(float32_path)
    assert bf16_path.read_bytes() == bf16_path_contents
    assert support_map(float32_restart) == expected_float32_support
    fresh_float32_sources = float32_t_sources[before_restart_sources:]
    assert fresh_float32_sources
    assert fresh_float32_sources[0] == {}
    final_float32_t = {
        (f"{row['kind']}:{row['first']}+{row['second']}", int(position["pos"])):
            position["t_ids"]
        for row in float32_restart if row["dose"] == .8
        for position in row["positions"]
    }
    assert any(fresh_float32_sources[1:])
    assert all(final_float32_t[key] == value
               for existing_t in fresh_float32_sources
               for key, value in existing_t.items())

    selected = rows119c.draw_path2_rows(units, rows119b.all_path2_rows(units))
    scored = score119c.score_interactions(
        read_records(bf16_path), float32_restart, read_records(bf16_path), selected, units)
    assert len(scored["per_dose"]) == 20
    assert scored["gates"]["gate2"]["dtype_evidence"]["float32_file"]["passed"]


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
            [sys.executable, "-I", "-B", script, "--help"],
            cwd=copied_root, text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "usage:" in result.stdout.lower()

    driver = copied_root / "copy_list_smoke.py"
    driver.write_text(textwrap.dedent("""\
        import json
        import sys
        from pathlib import Path
        from types import SimpleNamespace

        import torch
        import transformers

        root = Path(__file__).resolve().parent
        sys.path.insert(0, str(root / "part119c"))
        import run119c

        units = [
            {"id": "tiny-a", "layer": 0, "neuron": 0, "control_seed": 11901},
            {"id": "tiny-b", "layer": 1, "neuron": 1, "control_seed": 11902},
        ]

        class TinyTokenizer:
            pad_token_id = 0
            eos_token_id = 2
            unk_token_id = None
            def apply_chat_template(self, *args, **kwargs):
                return "tiny prompt"
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [1, 3, 5]}
            def decode(self, ids, **kwargs):
                return "tiny answer"

        config = transformers.Olmo2Config(
            vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32,
            pad_token_id=0, eos_token_id=2)
        def load_tiny(_snapshot, _device, dtype):
            torch.manual_seed(11906)
            model = transformers.Olmo2ForCausalLM(config).to(dtype=dtype).eval()
            return TinyTokenizer(), model

        answers = {"plain": {"tokens": [4, 5, 6, 7]},
                   "single:tiny-a": {"tokens": [4, 9, 6, 7]},
                   "single:tiny-b": {"tokens": [4, 9, 6, 7]}}
        run119c.rows119 = SimpleNamespace(
            load_units=lambda *_args: units,
            DEFAULT_NEURONS="unused", DEFAULT_HAIKU_KINDS="unused",
            DEFAULT_DS_KINDS="unused")
        run119c.load_model = load_tiny
        run119c.clear_device_cache = lambda _device: None
        run119c.run119.first_flip = lambda _single, _plain: (1, 9, 5)
        run119c.run119b.read_jsonl_readonly = lambda *_args, **_kwargs: answers
        assert run119c.main([
            "--model", "local-tiny", "--out", "results/part119c-copy-smoke",
            "--device", "cpu", "--batch", "2"]) == 0

        import rows119b
        import rows119c
        import score119c
        out = root / "results" / "part119c-copy-smoke"
        bf16 = [json.loads(line) for line in (out / "interaction-bf16.jsonl").read_text().splitlines()]
        float32 = [json.loads(line) for line in (out / "interaction-float32.jsonl").read_text().splitlines()]
        selected = rows119c.draw_path2_rows(units, rows119b.all_path2_rows(units))
        scored = score119c.score_interactions(bf16, float32, bf16, selected, units)
        assert len(bf16) == len(float32) == len(selected) * 5
        assert len(scored["per_dose"]) == 20
        assert scored["gates"]["gate2"]["dtype_evidence"]["float32_file"]["passed"]
        """), encoding="utf-8")
    result = subprocess.run([sys.executable, "-I", "-B", str(driver)],
                            cwd=copied_root, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
