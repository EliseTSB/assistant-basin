from __future__ import annotations

import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part120 import run120, score120


def _known_vectors(cubic: float = 0.0) -> dict:
    h = np.asarray([1.0, -1.0, 2.0, -2.0], dtype=np.float64)
    g1 = np.asarray([1.0, -1.0, 0.5, -0.5], dtype=np.float64)
    g2 = np.asarray([0.5, -0.5, 1.0, -1.0], dtype=np.float64)

    def function(x1: float, x2: float) -> np.ndarray:
        quadratic = .5 * h * (x1 * x1 + x1 * x2 + 2 * x2 * x2)
        cubic_term = cubic * h * (x1 ** 3 + x2 ** 3 - 2 * x1 * x1 * x2)
        return x1 * g1 + x2 * g2 + quadratic + cubic_term

    vectors = {}
    for dose in score120.DOSES:
        vectors[dose] = {
            "A": function(dose, 0.0),
            "B": function(0.0, dose),
            "P": function(dose, dose),
        }
    return {("real", "u0", "u1", 0): vectors}


def test_known_quadratic_and_cubic_pin_formulas():
    pure = score120.position_statistics(_known_vectors(0.0))[("real", "u0", "u1", 0)]
    assert pure[.025]["c"] == pytest.approx(math.sqrt(.125), rel=1e-12)
    assert pure[.025]["q"] == pytest.approx(0.0, abs=1e-12)

    cubic = score120.position_statistics(_known_vectors(.001))[("real", "u0", "u1", 0)]
    assert cubic[.025]["q"] > 0
    assert cubic[.05]["q"] / cubic[.025]["q"] == pytest.approx(2.0, rel=1e-10)


def _gate1_record(ids: list[str], *, passed: bool = True) -> dict:
    return {"passed": passed, "model_dtype": "float32",
            "model_parameter_dtype": "torch.float32",
            "paths": [{"first": unit, "passed": passed, "max_abs_difference": 0.0}
                      for unit in ids]}


@pytest.mark.parametrize("path_pass,expected", [(True, True), (False, False)])
def test_gate1_both_sides(path_pass, expected):
    assert score120.gate1_result(_gate1_record(["u0"], passed=path_pass), ["u0"])["passed"] is expected


def _gate3_record(ids: list[str], error: float) -> dict:
    pairs = list(zip(score120.DOSES, score120.DOSES[1:]))
    units = []
    for unit in ids:
        comparisons = [{"low_dose": low, "high_dose": high, "arm": arm,
                        "relative_error": error, "passed": error <= score120.GATE3_THRESHOLD}
                       for low, high in pairs for arm in ("neuron", "random")]
        units.append({"unit": unit, "passed": error <= score120.GATE3_THRESHOLD,
                      "max_relative_error": error, "comparisons": comparisons})
    return {"passed": error <= score120.GATE3_THRESHOLD, "units": units}


@pytest.mark.parametrize("error,expected", [
    (score120.GATE3_THRESHOLD, True),
    (score120.GATE3_THRESHOLD * 1.001, False),
])
def test_gate3_at_and_over_threshold(error, expected):
    assert score120.gate3_result(_gate3_record(["u0"], error), ["u0"])["passed"] is expected


def _gate2_fixture(relative_error: float):
    row = {"kind": "real", "first": "u0", "second": "u1"}
    arrays = {"A": np.asarray([1.0, 0.0]), "B": np.asarray([0.0, 1.0]),
              "P": np.asarray([2.0, 2.0])}
    vectors = {("real", "u0", "u1", 0): {
        dose: {field: value.copy() for field, value in arrays.items()}
        for dose in score120.DOSES}}
    norms = score120._vector_norms(arrays)
    positions = [{"pos": 0, **{field: value / (1 + relative_error)
                                for field, value in norms.items()}}]
    reference = [{"kind": "real", "first": "u0", "second": "u1", "dose": dose,
                  "positions": positions} for dose in score120.GATE2_DOSES]
    return vectors, reference, [row]


@pytest.mark.parametrize("relative_error,expected", [
    (.00499, True), (.005, False), (.00501, False),
])
def test_gate2_both_sides_and_strict_boundary(relative_error, expected):
    vectors, reference, rows = _gate2_fixture(relative_error)
    result = score120.gate2_result(vectors, reference, rows)
    assert result["passed"] is expected


def test_gate1_gate2_gate3_missing_or_nan_never_pass():
    assert not score120.gate1_result(None, ["u0"])["passed"]
    assert not score120.gate3_result(None, ["u0"])["passed"]
    vectors, reference, rows = _gate2_fixture(0)
    vectors[("real", "u0", "u1", 0)][.05]["A"][0] = np.nan
    result = score120.gate2_result(vectors, reference, rows,
                                   [{"problem": "nonfinite_vector"}])
    assert not result["passed"]


def test_missing_unit_file_is_a_gate2_failure(tmp_path):
    units = [{"id": "u0"}]
    vectors, issues = score120.load_vectors(tmp_path, units)
    result = score120.gate2_result(vectors, [],
                                   [{"kind": "real", "first": "u0", "second": "u1"}],
                                   issues)
    assert vectors == {}
    assert any(issue["problem"] == "missing_unit_file" for issue in issues)
    assert result["passed"] is False


def test_failed_gate_replaces_every_prediction_verdict():
    unit = {"id": "u0", "layer": 0, "haiku_kind": "slot", "ds_kind": "slot",
            "group": "agreed_slot"}
    row = {"kind": "real", "first": "u0", "second": "u1"}
    result = score120.score_data({}, [row], [unit], [], None, None, None,
                                 [{"problem": "missing_unit_file", "unit": "u0"}])
    assert {item["verdict"] for item in result["predictions"]} == {"not read (gate 1)"}
    assert "nothing is read" in score120.render_markdown(result).lower()


def _prediction_fixture():
    units = []
    unit_values = {}
    for index in range(39):
        if index < 21:
            haiku, group, value = "slot", "agreed_slot", 2.0 + index * .001
        elif index < 29:
            haiku, group, value = "concept", "agreed_concept", 1.0 - index * .001
        else:
            haiku, group, value = "concept", "disputed", .1 + index * .0001
        unit_id = f"u{index:02}"
        units.append({"id": unit_id, "layer": index % 4, "haiku_kind": haiku,
                      "ds_kind": "slot" if group == "agreed_slot" else
                      "concept" if group == "agreed_concept" else "unclassified",
                      "group": group})
        unit_values[unit_id] = {"u": value, "rows_with_self_share": 8,
                                "expected_rows": 8, "haiku_kind": haiku,
                                "ds_kind": units[-1]["ds_kind"], "group": group,
                                "layer": index % 4}
    table = []
    for dose in score120.DOSES:
        for kind in score120.KINDS:
            table.append({"dose": dose, "kind": kind,
                          "median_row_norm_I": 1.0, "median_row_norm_SA": 1.0,
                          "median_row_norm_SB": 1.0, "median_row_norm_SP": 1.0,
                          "median_row_norm_E": 1.0, "median_row_c": .75,
                          "median_row_q": .4, "median_row_pair_share": 1.0,
                          "median_row_self_share": .5})
    values = {.025: 1.0, .05: 4.0, .1: 16.0}
    for row in table:
        if row["kind"] == "real" and row["dose"] in values:
            row["median_row_norm_I"] = values[row["dose"]]
        if row["dose"] == .05 and row["kind"] == "real":
            row["median_row_q"] = .30
        if row["dose"] == .2 and row["kind"] == "real":
            row["median_row_q"] = .50
    row_stats = {}
    for unit in units:
        for kind in score120.KINDS:
            for second_no in range(4):
                row_stats[(kind, unit["id"], f"v{second_no}", .1)] = {
                    "kind": kind, "first": unit["id"], "second": f"v{second_no}",
                    "dose": .1, "norm_SB": 1.0}
    u_order = sorted(units, key=lambda unit: unit_values[unit["id"]]["u"])
    dominance_by_id = {unit["id"]: 39 - rank for rank, unit in enumerate(u_order)}
    dominance = {"units": [{"id": unit["id"], "dominance": dominance_by_id[unit["id"]]}
                           for unit in units]}
    bootstrap = {"ci95_product_weighted": [.6, .9],
                 "ci95_sum_weighted": [.6, .9]}
    return table, row_stats, unit_values, units, dominance, bootstrap


def _prediction_by_id(result):
    return {row["id"]: row for row in result}


def test_each_prediction_boundary_and_missing_value():
    table, rows, values, units, tournament, bootstrap = _prediction_fixture()
    preds = _prediction_by_id(score120.predictions(table, rows, values, units,
                                                   tournament, bootstrap))
    assert all(preds[f"120.{number}"]["verdict"] == "holds" for number in range(1, 9))

    table_slope = [dict(row) for row in table]
    for row in table_slope:
        if row["kind"] == "real" and row["dose"] == .025:
            row["median_row_norm_I"] = 1.0
        elif row["kind"] == "real" and row["dose"] == .05:
            row["median_row_norm_I"] = 2 ** 1.7
        elif row["kind"] == "real" and row["dose"] == .1:
            row["median_row_norm_I"] = 2 ** 3.4
    assert _prediction_by_id(score120.predictions(table_slope, rows, values, units,
                                                   tournament, bootstrap))["120.1"]["verdict"] == "holds"
    next(row for row in table_slope if row["dose"] == .1 and row["kind"] == "real")["median_row_norm_I"] = 2 ** 3.399
    assert _prediction_by_id(score120.predictions(table_slope, rows, values, units,
                                                   tournament, bootstrap))["120.1"]["verdict"] == "does_not_hold"

    table_bad = [dict(row) for row in table]
    for row in table_bad:
        if row["kind"] == "real" and row["dose"] == .1:
            row["median_row_norm_I"] = 4.0 * 2 ** 2.31
    assert _prediction_by_id(score120.predictions(table_bad, rows, values, units,
                                                   tournament, bootstrap))["120.1"]["verdict"] == "does_not_hold"

    table_bad = [dict(row) for row in table]
    next(row for row in table_bad if row["dose"] == .05 and row["kind"] == "real")["median_row_q"] = .30001
    assert _prediction_by_id(score120.predictions(table_bad, rows, values, units,
                                                   tournament, bootstrap))["120.2"]["verdict"] == "does_not_hold"

    assert _prediction_by_id(score120.predictions(table, rows, values, units, tournament,
                                                   {"ci95_product_weighted": [.6, 1.0]}))["120.3"]["verdict"] == "does_not_hold"
    table_c = [dict(row) for row in table]
    next(row for row in table_c if row["dose"] == .1 and row["kind"] == "real")["median_row_c"] = .75001
    assert _prediction_by_id(score120.predictions(table_c, rows, values, units,
                                                   tournament, bootstrap))["120.3"]["verdict"] == "does_not_hold"

    table_factor = [dict(row) for row in table]
    for row in table_factor:
        if row["kind"] == "real" and row["dose"] in (.05, .1, .2):
            row["median_row_c"] = 1.25
        if row["kind"] == "control" and row["dose"] in (.05, .1, .2):
            row["median_row_c"] = 1.0
    assert _prediction_by_id(score120.predictions(table_factor, rows, values, units,
                                                   tournament, bootstrap))["120.4"]["verdict"] == "holds"
    next(row for row in table_factor if row["dose"] == .2 and row["kind"] == "real")["median_row_c"] = 1.25001
    assert _prediction_by_id(score120.predictions(table_factor, rows, values, units,
                                                   tournament, bootstrap))["120.4"]["verdict"] == "does_not_hold"

    table_share = [dict(row) for row in table]
    next(row for row in table_share if row["dose"] == .4 and row["kind"] == "real")["median_row_self_share"] = .49999
    assert _prediction_by_id(score120.predictions(table_share, rows, values, units,
                                                   tournament, bootstrap))["120.5"]["verdict"] == "does_not_hold"
    next(row for row in table_share if row["dose"] == .4 and row["kind"] == "real")["median_row_self_share"] = 2.0
    assert _prediction_by_id(score120.predictions(table_share, rows, values, units,
                                                   tournament, bootstrap))["120.5"]["verdict"] == "holds"
    next(row for row in table_share if row["dose"] == .4 and row["kind"] == "real")["median_row_self_share"] = 2.00001
    assert _prediction_by_id(score120.predictions(table_share, rows, values, units,
                                                   tournament, bootstrap))["120.5"]["verdict"] == "does_not_hold"

    bad_agreement = {key: dict(value) for key, value in values.items()}
    for unit in units:
        if unit["group"] == "agreed_slot":
            bad_agreement[unit["id"]]["u"] = 1.0
        elif unit["group"] == "agreed_concept":
            bad_agreement[unit["id"]]["u"] = 2.0
        else:
            bad_agreement[unit["id"]]["u"] = .1
    assert _prediction_by_id(score120.predictions(table, rows, bad_agreement, units,
                                                   tournament, bootstrap))["120.6"]["verdict"] == "does_not_hold"
    assert score120._slot_direction_holds(2.0, 1.0, .04999)
    assert not score120._slot_direction_holds(2.0, 1.0, .05)
    same_direction = {"units": [{"id": unit["id"], "dominance": values[unit["id"]]["u"]}
                                 for unit in units]}
    assert _prediction_by_id(score120.predictions(table, rows, values, units,
                                                   same_direction, bootstrap))["120.7"]["verdict"] == "does_not_hold"
    assert score120._dominance_holds(-.3)
    assert not score120._dominance_holds(-.29999)
    low_ratio_rows = {key: dict(row) for key, row in rows.items()}
    for key, row in low_ratio_rows.items():
        if key[0] == "real":
            row["norm_SB"] = .799
    assert _prediction_by_id(score120.predictions(table, low_ratio_rows, values, units,
                                                   tournament, bootstrap))["120.8"]["verdict"] == "does_not_hold"
    boundary_ratio_rows = {key: dict(row) for key, row in rows.items()}
    for key, row in boundary_ratio_rows.items():
        if key[0] == "real":
            row["norm_SB"] = .8
    assert _prediction_by_id(score120.predictions(table, boundary_ratio_rows, values, units,
                                                   tournament, bootstrap))["120.8"]["verdict"] == "holds"
    for row in boundary_ratio_rows.values():
        if row["kind"] == "real":
            row["norm_SB"] = 1.25
    assert _prediction_by_id(score120.predictions(table, boundary_ratio_rows, values, units,
                                                   tournament, bootstrap))["120.8"]["verdict"] == "holds"

    missing_table = [dict(row) for row in table]
    next(row for row in missing_table if row["dose"] == .05 and row["kind"] == "real")["median_row_q"] = float("nan")
    assert _prediction_by_id(score120.predictions(missing_table, rows, values, units,
                                                   tournament, bootstrap))["120.2"]["verdict"] == "not_evaluable"
    missing_tournament = {"units": [{"id": unit["id"], "dominance": 1.0}
                                    for unit in units[:-1]]}
    assert _prediction_by_id(score120.predictions(table, rows, values, units,
                                                   missing_tournament, bootstrap))["120.7"]["verdict"] == "not_evaluable"


def test_headline_all_branches_including_withdrawn():
    gates = {f"gate{i}": {"passed": True} for i in (1, 2, 3)}
    prediction_rows = [{"id": "120.4", "verdict": "holds"}]

    def headline(q, interval):
        table = [{"dose": .05, "kind": "real", "median_row_q": q}]
        return score120._headline(prediction_rows, table,
                                  {"ci95_product_weighted": interval}, gates)

    assert "withdrawn" in headline(.30001, [.5, .6])
    assert "more than with a second push" in headline(.3, [.5, .79])
    assert "about as much as with itself" in headline(.3, [.8, 1.25])
    assert "more than with itself" in headline(.3, [1.2501, 1.4])
    assert "read nothing" in headline(.3, [.79, .81])
    prediction_rows[0]["verdict"] = "does_not_hold"
    assert "not established" in headline(.3, [.5, .79])
    gates["gate2"]["passed"] = False
    assert "gate 2" in score120._headline(prediction_rows,
                                          [{"dose": .05, "kind": "real", "median_row_q": .1}],
                                          {"ci95_product_weighted": [.5, .6]}, gates)


def test_npz_layout_round_trip_and_atomic_writer(tmp_path):
    index = {"first": "u0", "doses": list(score120.DOSES),
             "model_dtype": "float32", "model_parameter_dtype": "torch.float32",
             "entries": [{"key": "real:u0+u1", "kind": "real", "first": "u0",
                          "second": "u1", "pos": 3, "t_ids": [0], "start": 0}]}
    values = np.arange(len(score120.DOSES) * 3, dtype=np.float64)
    path = tmp_path / "unit.npz"
    run120.write_unit_npz(path, index, values)
    loaded_index, loaded_values = run120.read_unit_npz(path)
    assert loaded_index == index
    np.testing.assert_array_equal(loaded_values, values)
    assert run120.finished_unit(path, index)


class _ToyTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, *_args, **_kwargs):
        return "tiny prompt"

    def __call__(self, _text, **_kwargs):
        return {"input_ids": [1, 2]}


def _tiny_model():
    import torch
    from torch import nn

    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.down_proj.weight.copy_(torch.tensor([
                    [1.0, .2, .3], [.4, 1.2, .1], [.2, .3, .9], [.8, .5, .7]
                ]))

    class TinyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = TinyMLP()

        def forward(self, hidden):
            return hidden * .75 + .1

    class TinyCore(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([TinyLayer()])

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = TinyCore()
            self.embedding = nn.Embedding(16, 4)
            self.head = nn.Linear(4, 8, bias=False)
            torch.manual_seed(4)
            with torch.no_grad():
                self.embedding.weight.normal_()
                self.head.weight.normal_()

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            hidden = self.embedding(input_ids)
            for layer in self.model.layers:
                hidden = layer(hidden)
            return SimpleNamespace(logits=self.head(hidden))

    return TinyModel()


def _tiny_inputs(tmp_path, monkeypatch):
    units = [
        {"id": "u0", "layer": 0, "neuron": 0, "control_seed": 11,
         "haiku_kind": "slot", "ds_kind": "slot", "group": "agreed_slot"},
        {"id": "u1", "layer": 0, "neuron": 1, "control_seed": 12,
         "haiku_kind": "concept", "ds_kind": "concept", "group": "agreed_concept"},
    ]
    monkeypatch.setattr(run120.rows119, "load_units", lambda *_args: units)
    monkeypatch.setattr(run120.rows119b, "_registered_inventory", lambda _units: False)
    answers = [{"row_id": "plain", "tokens": [1, 2, 3, 4]}]
    for unit in units:
        answers.append({"row_id": f"single:{unit['id']}",
                        "tokens": [1, 9, 3, 4]})
    answers_path = tmp_path / "answers.jsonl"
    answers_path.write_text("\n".join(json.dumps(row) for row in answers) + "\n",
                            encoding="utf-8")
    rows = run120.rows119c.draw_path2_rows(units, run120.rows119b.all_path2_rows(units))
    t_rows = []
    for row in rows:
        t_rows.append({"kind": row["kind"], "first": row["first"],
                       "second": row["second"], "dose": .8,
                       "positions": [{"pos": pos, "t_ids": [0, 1, 2, 3]}
                                     for pos in (1, 2, 3)]})
    t_path = tmp_path / "reference.jsonl"
    t_path.write_text("\n".join(json.dumps(row) for row in t_rows) + "\n",
                      encoding="utf-8")
    monkeypatch.setattr(run120, "ANSWERS_PATH", str(answers_path))
    monkeypatch.setattr(run120, "T_REFERENCE_PATH", str(t_path))
    monkeypatch.setattr(run120.run119c, "load_model",
                        lambda _path, _device, _dtype: (_ToyTokenizer(), _tiny_model()))
    return units


def _runner_args(tmp_path):
    return ["--model", "tiny", "--out", str(tmp_path / "smoke"), "--device", "cpu",
            "--batch", "2", "--units", "2", "--doses", "0.05,0.1"]


def test_real_main_path_tiny_model_and_resume(monkeypatch, tmp_path):
    _tiny_inputs(tmp_path, monkeypatch)
    argv = _runner_args(tmp_path)
    assert run120.main(argv) == 0
    output = tmp_path / "smoke" / "vectors"
    assert sorted(path.name for path in output.glob("*.npz")) == ["u0.npz", "u1.npz"]
    for path in output.glob("*.npz"):
        _index, values = run120.read_unit_npz(path)
        assert values.dtype == np.float64 and values.size > 0

    def should_not_recompute(*_args, **_kwargs):
        pytest.fail("a finished unit was recomputed on resume")

    monkeypatch.setattr(run120, "_run_first_unit", should_not_recompute)
    assert run120.main(argv) == 0


def test_interrupted_unit_has_no_finished_file(monkeypatch, tmp_path):
    _tiny_inputs(tmp_path, monkeypatch)
    args = SimpleNamespace(model="tiny", out=str(tmp_path / "interrupted"), device="cpu",
                           batch=2, units=2, doses=(.05, .1), neurons="unused",
                           kinds="unused", kinds_ds="unused")

    def stop_mid_unit(*_args, **_kwargs):
        raise RuntimeError("simulated interruption during this unit")

    monkeypatch.setattr(run120, "_run_first_unit", stop_mid_unit)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run120.run(args)
    vectors = tmp_path / "interrupted" / "vectors"
    assert not list(vectors.glob("*.npz"))
    assert not list(vectors.glob("*.tmp"))


def test_isolated_help_from_repo_root():
    for script in ("part120/run120.py", "part120/score120.py"):
        result = subprocess.run([sys.executable, "-I", "-B", script, "--help"],
                                cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout


def test_copy_list_help_in_isolated_job_tree(tmp_path):
    files = [
        "part120/run120.py", "part120/score120.py",
        "part119c/run119c.py", "part119c/rows119c.py",
        "part119b/run119b.py", "part119b/rows119b.py", "part119b/gate119b.py",
        "part119/run119.py", "part119/rows119.py", "part119/score119.py",
        "part92/common92.py", "part85/checkpoints.py", "part86/triage86.py",
    ]
    for relative in files:
        source, destination = ROOT / relative, tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for script in ("part120/run120.py", "part120/score120.py"):
        result = subprocess.run([sys.executable, "-I", "-B", script, "--help"],
                                cwd=tmp_path, capture_output=True, text=True,
                                encoding="utf-8")
        assert result.returncode == 0, result.stderr


def test_deliberate_plus_sign_break_turns_formula_test_red(tmp_path):
    source = (ROOT / "part120/score120.py").read_text(encoding="utf-8")
    original = 'sa = raw[high]["A"] - 2 * a'
    assert source.count(original) == 1
    broken = source.replace(original, 'sa = raw[high]["A"] + 2 * a')
    broken_path = tmp_path / "score120_broken.py"
    broken_path.write_text(broken, encoding="utf-8")
    test_path = tmp_path / "test_broken_formula.py"
    broken_test = "\n".join([
        "import importlib.util, numpy as np, sys",
        f"ROOT = {str(ROOT)!r}",
        "sys.path[:0] = [ROOT, ROOT + '/part119c', ROOT + '/part119', ROOT + '/part119b']",
        f"SPEC = importlib.util.spec_from_file_location('broken', {str(broken_path)!r})",
        "broken = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(broken)",
        "def test_formula():",
        "    h=np.asarray([1.,-1.,2.,-2.]); g=np.asarray([1.,-1.,.5,-.5])",
        "    def f(x,y): return g*x + .5*h*(x*x+x*y+2*y*y)",
        "    data={}",
        "    for d in broken.DOSES: data[d]={'A':f(d,0.),'B':f(0.,d),'P':f(d,d)}",
        "    stats=broken.position_statistics({('real','u0','u1',0):data})[('real','u0','u1',0)]",
        "    assert abs(stats[.025]['q']) < 1e-9",
    ]) + "\n"
    test_path.write_text(broken_test, encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p",
                             "no:cacheprovider", str(test_path)], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8")
    assert result.returncode != 0
    assert "1 failed" in result.stdout, result.stdout

