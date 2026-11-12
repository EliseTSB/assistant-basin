"""Part 202's runner: its plan, its gates, its resume path and its replicate. No download, no GPU.

Run as pytest, or as `python -I -B part202/test_part202.py` (a __main__ block runs every test and exits non-zero on
failure). The runner tests drive run202.run() end to end on part202/tiny202.py's tiny Qwen3.5 model: neuron_report,
part155's common module, run164's pushes and the question list are replaced by small stand-ins, and the run itself,
its load record, its resume and its replicate are the real code paths. Each test's docstring names the line to break
to make it red.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part86")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from part202 import run202  # noqa: E402
from part202 import tiny202  # noqa: E402


# --------------------------------------------------------------------------------------------- fixtures

class FakeCommon:
    """common86's two functions the runner calls, without neuron_report. The prompt is the tokenizer's fallback."""
    REVISION = "tiny-revision"

    def build_prompt(self, tok, question):
        return f"User: {question}\nAssistant:"

    def find_layers(self, model):
        return model.model.layers


def _fake_pushes(rows, layers, medians, common, cache=None):
    """run164.pushes_for's shape: one [rows, width] tensor per layer present, one row per row of the call."""
    import torch
    width = int(layers[0].mlp.down_proj.weight.shape[1])
    out = {}
    for layer in sorted({int(r["layer"]) for r in rows if r.get("layer") is not None}):
        dense = torch.zeros((len(rows), width), dtype=torch.float32)
        for i, row in enumerate(rows):
            if int(row["layer"]) == layer:
                # Strict like run163.pushes_for: a row without its unit index raises (review part202-code-r1 F2: the
                # old `or 0` here let a saved row with no `neuron` or `random_index` pass the replicate stage).
                index = row["neuron"] if row["arm"] == "neuron" else row["random_index"]
                dense[i, int(index) % width] = 0.01
        out[layer] = dense
    return out


def _install_fakes(monkeypatch, model=None, tok=None, kernel="eager", config_kernel="eager", questions=None,
                   answers=("I am Nova, a helpful system that responds clearly.",)):
    """The tiny model, the loader, the common module, the pushes and the question list. `answers` are the answers the
    model gives, one per decode step of every row (the tiny model's own answers never end, which would run every call
    to the 80-token cap and take minutes)."""
    import neuron_report as nr
    from part155 import run155
    from part164 import run164
    model = model if model is not None else tiny202.tiny_qwen()
    tok = tok if tok is not None else tiny202.TinyTokenizer()
    # The tiny model's answers are scripted: the loader's model.generate takes the processor that forces them.
    from transformers import LogitsProcessorList
    original_generate = model.generate

    def scripted_generate(*a, **kwargs):
        processors = [p for p in (kwargs.pop("logits_processor", None) or [])]
        processors.append(tiny202.ScriptedAnswers(answers[0] if len(answers) == 1 else list(answers)))
        kwargs["logits_processor"] = LogitsProcessorList(processors)
        return original_generate(*a, **kwargs)

    model.generate = scripted_generate
    loads = []

    def load_model(model_id, revision, device, **kwargs):
        loads.append(device)
        nr.LAST_LOAD = {"attn_implementation": kernel, "attn_implementation_requested": kernel,
                        "dtype": "torch.bfloat16", "device": device}
        model.config._attn_implementation = config_kernel
        return tok, model

    monkeypatch.setattr(nr, "load_model", load_model)
    monkeypatch.setattr(run155, "_common", lambda: FakeCommon())
    monkeypatch.setattr(run164, "pushes_for", _fake_pushes)
    items = questions if questions is not None else [{"question": "What is your name?", "kind": "identity"}]
    monkeypatch.setattr(run164, "choose_questions", lambda: items)
    # Gate R2's hashes are checked on the real Part 164 files in their own test; the run tests use a tiny unit list.
    monkeypatch.setattr(run202, "gate_r2", lambda *a, **k: {"passed": True, "files": {}})
    return model, tok, loads


def _units():
    """100 neurons and 100 random directions on the tiny model's one layer (run164.run's own count)."""
    return ([{"unit_id": f"L0N{n}", "arm": "neuron", "layer": 0, "neuron": n, "pair_id": n} for n in range(100)] +
            [{"unit_id": f"R{n}", "arm": "random", "layer": 0, "random_index": n, "pair_id": n} for n in range(100)])


def _args(tmp_path, units=None, out="answers.jsonl"):
    path = tmp_path / "units.json"
    if not path.exists():
        path.write_text(json.dumps(units if units is not None else _units()), encoding="utf-8")
    # The tiny fixture's units file is not part164/units164.json, so run() does not apply the registered 6,516-row check
    # (that check is tested in test_the_run_refuses_a_plan_that_is_not_the_registered_row_count). The fixture's model
    # answers a fixed sentence, so its labels are known; the run itself still decodes every call (about 100 s).
    return SimpleNamespace(device="cpu", model="tiny", units=str(path), out=str(tmp_path / out))


def _rows(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------------------------- the plan

def test_the_plan_is_part_164s_order_in_chunks_of_8_with_four_companions():
    """Break: change PLANNED_ROWS, or drop the companion padding in plan() (the last battery chunk would be 4 rows)."""
    p = run202.plan()
    assert p["rows"] == 6516 and p["companions"] == 132       # the 36-row recipe block and each 36-row battery block
    stages = [c["stage"] for c in p["calls"]]
    assert stages[:16] == ["no_push"] * 16 and stages[16:18] == ["recipe"] * 2 and set(stages[18:]) == {"battery"}
    assert all(len(ch["row_ids"]) == 8 for c in p["calls"] if c["stage"] != "no_push" for ch in c["chunks"])
    assert all(len(ch["row_ids"]) == 1 for c in p["calls"] if c["stage"] == "no_push" for ch in c["chunks"])
    last = p["calls"][-1]
    assert last["rows"] == 36 and [len(ch["companions"]) for ch in last["chunks"]] == [0, 0, 0, 0, 4]
    assert all(ch["companions"] == [] for c in p["calls"] for ch in c["chunks"][:-1])
    # the battery's first call is the 100 neurons then the 100 random directions at 0.8, in units order (Part 164's
    # selection: run164.run's `selected`), and the 1.0/1.2 cells hold the random arm only
    first = [r for r in p["calls"][18]["chunks"][0]["rows"]]
    assert [r["arm"] for r in first] == ["neuron"] * 8 and {r["dose"] for r in first} == {0.8}
    cells = {(c["stage"], r["arm"], r["dose"]) for c in p["calls"] for ch in c["chunks"] for r in ch["rows"]
             if "companion_of" not in r}
    assert ("battery", "neuron", 1.0) not in cells and ("battery", "random", 1.2) in cells


def test_gate_r2_hashes_match_this_tree_and_a_changed_file_fails(tmp_path):
    """Break: change one hex digit in EXPECTED_SHA256 (the real files would fail), or skip the file loop."""
    assert run202.gate_r2()["passed"] is True
    units = tmp_path / "units.json"
    units.write_text(json.dumps(_units()[:6]), encoding="utf-8")
    got = run202.gate_r2(units, ROOT / "results/part112/medians.json")
    assert got["passed"] is False and got["files"]["part164/units164.json"]["matches"] is False


# --------------------------------------------------------------------------------------------- gates

def test_gate_r3_accepts_the_planned_shapes_and_refuses_a_short_chunk():
    """Break: in call_shapes, drop the `!= CHUNK` test (a 7-row chunk would pass), or the companions test."""
    p = run202.plan()
    calls = [{"pushed": c["stage"] != "no_push", "row_ids": c["row_ids"],
              "chunks": [{"row_ids": ch["row_ids"], "companions": ch["companions"], "rows": len(ch["row_ids"])}
                         for ch in c["chunks"]]} for c in p["calls"]]
    assert run202.call_shapes(calls)["passed"] is True
    short = json.loads(json.dumps(calls))
    short[18]["chunks"][0]["rows"] = 7
    assert run202.call_shapes(short)["passed"] is False
    unpadded = json.loads(json.dumps(calls))
    unpadded[-1]["chunks"][-1]["companions"] = []
    assert run202.call_shapes(unpadded)["passed"] is False
    plain_two = json.loads(json.dumps(calls))
    plain_two[0]["chunks"][0]["rows"] = 2
    assert run202.call_shapes(plain_two)["passed"] is False


@pytest.mark.parametrize("kernel,config_kernel", [("sdpa", "sdpa"), ("eager", "sdpa")])
def test_the_loader_refuses_a_non_eager_load_before_any_generation(monkeypatch, kernel, config_kernel):
    """Break: in EagerLoader.__call__, drop the `config_kernel != "eager"` half of the test (the eager/sdpa row would
    load and generate). The sdpa/eager row is not run: a config that records eager is the model's own record, and the
    load record is what Gate R1 reads."""
    _model, _tok, _loads = _install_fakes(monkeypatch, kernel=kernel, config_kernel=config_kernel)
    loader = run202.EagerLoader(SimpleNamespace(model="tiny"))
    with pytest.raises(SystemExit):
        loader("cpu")


def test_a_refused_load_is_recorded_as_refused_and_the_run_exits_3(tmp_path, monkeypatch):
    """Break: in record_gates, drop the `refused` branch (a refused launch would read as a pass), or in run() catch
    SystemExit from the loader (the record would never be written)."""
    _install_fakes(monkeypatch, kernel="sdpa", config_kernel="sdpa")
    assert run202.run(_args(tmp_path)) == 3
    rec = json.loads((tmp_path / "load.json").read_text())
    assert rec["loads"][-1]["status"] == "refused"
    # The record's own gates read a refused launch as a refusal even though its LAST_LOAD said eager (Part 200's
    # finding 7): the model's config said sdpa, so the load is not a pass.
    rec["loads"][-1]["load"]["attn_implementation"] = "eager"
    ok, why, _shapes = run202.record_gates(rec)
    assert ok is False and "refused" in why
    assert not (tmp_path / "answers.jsonl").exists()


# --------------------------------------------------------------------------------------------- the run and its resume

def test_the_run_writes_every_planned_row_once_in_chunks_of_8(tmp_path, monkeypatch, capsys):
    """Break: in run(), drop the `done` filter before a call is decoded (a resumed run duplicates rows), write the
    companion rows (the tiny plan's 501 rows would be 513), or key `done` without arm and dose (the random arm's +1.0
    and +1.2 rows are then never written: 301 rows, review part202-code-r1 F1)."""
    model, _tok, loads = _install_fakes(monkeypatch)
    seen = []
    real = run202._decode_chunk

    def counting(model_, tok_, layers, prompts, pushmap, device):
        seen.append(len(prompts))
        return real(model_, tok_, layers, prompts, pushmap, device)

    monkeypatch.setattr(run202, "_decode_chunk", counting)
    assert run202.run(_args(tmp_path)) == 0
    rows = _rows(tmp_path / "answers.jsonl")
    # one question: 1 no-push + 100 recipe + 100 neurons + 300 random rows at 0.8/1.0/1.2
    from part164 import run164
    assert len(rows) == 501 and len({run164.row_key(r) for r in rows}) == 501
    assert loads == ["cpu"]
    assert set(seen) == {1, 8} and seen[0] == 1
    assert sum(1 for n in seen if n == 8) == 64           # the plan's 500 pushed rows and 12 companions, in chunks of 8
    assert sum(1 for n in seen if n == 1) == 201          # the no-push call and the 200 one-row replicate calls
    rec = json.loads((tmp_path / "load.json").read_text())
    assert rec["loads"][-1]["status"] == "done"
    assert run202.check_record(tmp_path / "load.json", units=_args(tmp_path).units) == 0
    # Gate R1 wants one device (review part202-code-r1 F6): a second launch on another device fails the record.
    two = json.loads(json.dumps(rec))
    two["loads"].append(dict(two["loads"][-1], load=dict(two["loads"][-1]["load"], device="xpu:0")))
    assert run202.record_gates(two)[0] is False and "device" in run202.record_gates(two)[1]
    # Gate R3 against the plan: a record missing one planned call's receipt fails (review part202-code-r1 F1/F3).
    short = json.loads(json.dumps(rec))
    short["loads"][-1]["calls"] = [c for c in short["loads"][-1]["calls"] if c["call"] != 6]
    assert run202.record_gates(short, run202.plan(_args(tmp_path).units))[0] is False
    assert len(_rows(tmp_path / "replicate.jsonl")) == 200
    assert "PART202 record ok" in capsys.readouterr().out


def test_the_run_refuses_a_plan_that_is_not_the_registered_row_count(tmp_path, monkeypatch):
    """Break: in run(), drop the `planned["rows"] != PLANNED_ROWS` check (the run would go on with a plan that is not
    Part 164's). The registered units file is used, so the check applies; the question list is still the tiny one."""
    _install_fakes(monkeypatch)
    args = _args(tmp_path)
    args.units = str(ROOT / "part164/units164.json")
    assert run202.run(args) == 3
    assert not (tmp_path / "answers.jsonl").exists()


def test_a_resume_decodes_only_what_is_missing_and_a_half_written_call_again(tmp_path, monkeypatch):
    """Break: in _existing_whole, return `rows` unchanged (the half-written call would be kept and the run would leave
    a 56-row hole), or in run() write a redrawn call's rows without the on-disk filter (the 8 kept rows twice)."""
    model, _tok, _loads = _install_fakes(monkeypatch)
    assert run202.run(_args(tmp_path)) == 0
    path = tmp_path / "answers.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    # A kill in the middle of a battery call: its first chunk written, the rest missing. The plan's last call is the
    # 36-row one, so the call is the first 64-row battery call and its rows are read from the file itself. The two
    # recipe calls (also 64 rows) come before it and are kept whole.
    first_battery = next(c["call"] for c in run202.plan()["calls"] if c["stage"] == "battery")
    in_call = [i for i, line in enumerate(lines) if json.loads(line)["call"] == first_battery]
    kept = lines[:in_call[0] + 8]
    path.write_text("".join(kept), encoding="utf-8")
    (tmp_path / "replicate.jsonl").unlink()
    seen = []
    real = run202._decode_chunk
    monkeypatch.setattr(run202, "_decode_chunk",
                        lambda *a: (seen.append(len(a[3])), real(*a))[1])
    assert run202.run(_args(tmp_path)) == 0
    rows = _rows(path)
    from part164 import run164
    assert len(rows) == 501 and len({run164.row_key(r) for r in rows}) == 501
    # The half-written call's 8 rows were set aside and its whole call (64 rows, 8 chunks) decoded again, then every
    # call after it (the rest of the battery), and the 200 one-row replicate calls. The first battery call's 64 rows
    # are on disk once each (the 8 kept rows were redrawn but not written twice).
    assert len(_rows(tmp_path / "rows.removed.jsonl")) == 8
    # 8 for the redrawn call, 17 for the rest of +0.8 (64, 64, 8 rows), 13 each for +1.0 and +1.2 (64 + 36 rows).
    assert seen[:8] == [8] * 8 and sum(1 for n in seen if n == 8) == 51
    first_call = [r for r in rows if r["call"] == first_battery]
    assert len(first_call) == 64 and len({run164.row_key(r) for r in first_call}) == 64
    assert len(_rows(tmp_path / "replicate.jsonl")) == 200


def test_the_pilot_stops_below_the_guard_and_records_it(tmp_path, monkeypatch):
    """Break: in run(), compare `rate < PILOT_MIN_RATE` with `rate > PILOT_MIN_RATE` (a fast run would stop), or drop
    the `break` (the run would go on after the stop)."""
    _install_fakes(monkeypatch)
    monkeypatch.setattr(run202, "PILOT_MIN_RATE", 1e9)
    assert run202.run(_args(tmp_path)) == 3
    rec = json.loads((tmp_path / "load.json").read_text())
    assert rec["loads"][-1]["status"] == "stopped" and "below" in rec["loads"][-1]["failure"]
    # the no-push rows and the first three pushed calls only (the recipe's 64 + 64, then the battery's first 64)
    assert len(_rows(tmp_path / "answers.jsonl")) == 1 + 64 + 64 + 36   # the no-push row, the recipe's two 64-row calls, the first battery call (36 rows)
    assert not (tmp_path / "replicate.jsonl").exists()
    # A pilot stop is a HOLD: the host's record check fails on it (review part202-code-r1 F3).
    assert run202.check_record(tmp_path / "load.json", units=_args(tmp_path).units) == 3
    assert "not done" in run202.record_gates(rec)[1]


def test_a_float32_load_is_refused_before_generation(tmp_path, monkeypatch):
    """Break: drop the dtype check in EagerLoader (a float32 load would run as the bf16 rerun, review F7)."""
    import neuron_report as nr
    _install_fakes(monkeypatch)
    eager_load = nr.load_model

    def float32_load(*a, **k):
        tok, model = eager_load(*a, **k)
        nr.LAST_LOAD = dict(nr.LAST_LOAD, dtype="torch.float32")
        return tok, model

    monkeypatch.setattr(nr, "load_model", float32_load)
    assert run202.run(_args(tmp_path)) == 3
    rec = json.loads((tmp_path / "load.json").read_text())
    assert rec["loads"][-1]["status"] == "refused"
    assert not (tmp_path / "answers.jsonl").exists()


def test_the_replicate_rows_are_written_one_row_per_call(tmp_path, monkeypatch):
    """Break: in run_replicate_stage, hand the whole sample to one decode (the replicate would be a chunked call, not
    the second shape the design asks for)."""
    model, _tok, _loads = _install_fakes(monkeypatch)
    assert run202.run(_args(tmp_path)) == 0
    replicate = _rows(tmp_path / "replicate.jsonl")
    assert len(replicate) == 200 and all(r["stage"] == "replicate" and r["call_rows"] == [r["row_id"]] for r in replicate)
    assert {r["arm"] for r in replicate} == {"neuron", "random"}
    # each replicate row is a battery row of the run, one row per call
    battery = {(r["unit_id"], r["dose"], r["question_index"]) for r in _rows(tmp_path / "answers.jsonl")
               if r["stage"] == "battery"}
    assert {(r["unit_id"], r["dose"], r["question_index"]) for r in replicate} <= battery


def test_an_unexpected_error_still_records_the_launch_as_stopped(tmp_path, monkeypatch):
    """Break: in run(), drop the `except BaseException` around the call loop (a killed model would leave the launch
    "running" in load.json)."""
    _install_fakes(monkeypatch)

    def killed(*a, **k):
        raise RuntimeError("the card went away")

    monkeypatch.setattr(run202, "_decode_chunk", killed)
    with pytest.raises(RuntimeError):
        run202.run(_args(tmp_path))
    rec = json.loads((tmp_path / "load.json").read_text())
    assert rec["loads"][-1]["status"] == "stopped" and "the card went away" in rec["loads"][-1]["failure"]


def test_the_replicate_sample_is_seeded_and_spread_across_cells():
    """Break: in replicate_plan, return the first 200 battery rows (all from one cell), or drop the seed."""
    rows = [{"stage": "battery", "unit_id": f"u{i}", "arm": arm, "dose": dose, "question_index": i % 16}
            for arm, dose in (("neuron", 0.8), ("random", 0.8), ("random", 1.0), ("random", 1.2)) for i in range(100)]
    picked = run202.replicate_plan(rows)
    assert len(picked) == 200 and len({(r["arm"], r["dose"]) for r in picked}) == 4
    assert sum(r["arm"] == "neuron" for r in picked) == 50
    assert [r["unit_id"] for r in picked] == [r["unit_id"] for r in run202.replicate_plan(rows)]
    assert [r["unit_id"] for r in picked] != [r["unit_id"] for r in run202.replicate_plan(rows, seed=1)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_the_plan_matches_part_164s_published_rows_key_for_key_in_order():
    """The runner reimplements Part 164's plan rather than calling run164.run() (no seam there records margins), so
    this test proves the reimplementation equal to what Part 164 actually ran: the 6,516 non-companion rows of
    run202.plan(), in order, carry Part 164's row key, question and unit as results/part164/answers.jsonl does.
    Break: swap two calls in plan(), or change one dose, and this goes red."""
    from part164 import run164
    published = [json.loads(line) for line in (ROOT / "results/part164/answers.jsonl").read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    planned = [r for c in run202.plan()["calls"] for ch in c["chunks"] for r in ch["rows"] if "companion_of" not in r]
    assert len(planned) == len(published) == 6516
    assert [run164.row_key(r) for r in planned] == [run164.row_key(r) for r in published]
    assert [r["question"] for r in planned] == [r["question"] for r in published]

    def unit(row):
        index = row.get("neuron") if row.get("arm") == "neuron" else row.get("random_index") if row.get("arm") == "random" else None
        return row.get("layer"), index

    assert [unit(r) for r in planned] == [(r.get("layer"), r.get("neuron_or_index")) for r in published]
