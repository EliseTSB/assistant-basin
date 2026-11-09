#!/usr/bin/env python3
"""Part 200: the wrapper's four changes, its gates, its record and --compare. No model, no GPU.

Run as pytest, or as `python -I -B part200/test_part200.py` (a __main__ block runs every test and exits non-zero on
failure). Fixtures only: a fake model and tokenizer stand in for the Qwen load, and a recorder stands in for Part 163's
generator (part160.generate_checked). The chunking tests read the push tensors each chunk receives; the loader tests
fake neuron_report.load_model; the run tests replace part163.run163.run with a stand-in that calls the wrapper's loader
and generator, as the real run163.run does.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "part86")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from part200 import run200  # noqa: E402
from part163 import run163  # noqa: E402
from part199 import run199  # noqa: E402


# --------------------------------------------------------------------------------------------- fixtures

class FakePush:
    """A stand-in for a [rows, width] push tensor: slicing keeps the original row numbers."""
    def __init__(self, rows, label="layer"):
        self.rows = [{"row": r, "label": label} for r in rows]
        self.shape = (len(self.rows), 4)

    def __getitem__(self, s):
        return _Slice(self.rows[s])

    @property
    def label(self):
        return self.rows[0]["label"] if self.rows else "layer"


class _Slice(FakePush):
    def __init__(self, rows):
        self.rows = list(rows)
        self.shape = (len(self.rows), 4)


class Recorder:
    """Stands in for part160.generate_checked: records what it was handed, answers one row per push row."""
    def __init__(self):
        self.calls = []

    def __call__(self, model, tok, layers, question, pushes, max_new_tokens, device):
        n = int(next(iter(pushes.values())).shape[0]) if pushes else 1
        self.calls.append({"question": question, "pushes": pushes, "rows": n, "max_new_tokens": max_new_tokens})
        if not pushes:
            return [{"answer": "plain answer", "token_count": 3}]
        first = next(iter(pushes.values()))
        return [{"answer": f"row {row['row']}", "token_count": 4} for row in first.rows]


def _pushes(n, layers=(19, 20)):
    return {layer: FakePush(range(n), label=layer) for layer in layers}


# --------------------------------------------------------------------------------------------- change 2: chunking

def test_a_64_row_pushed_call_reaches_the_generator_as_eight_chunks_of_eight_in_order():
    rec = Recorder()
    gen = run200.chunked_generate(rec)
    answers = gen(None, None, None, "Who are you?", _pushes(64), 80, "cpu")
    assert [call["rows"] for call in rec.calls] == [8] * 8
    # each chunk's push tensors are exactly the matching rows of the original, for every layer
    for chunk_no, call in enumerate(rec.calls):
        for layer in (19, 20):
            assert call["pushes"][layer].rows == [{"row": r, "label": layer} for r in range(chunk_no * 8, chunk_no * 8 + 8)]
        assert call["max_new_tokens"] == 80 and call["question"] == "Who are you?"
    # the answers come back in the original row order
    assert [a["answer"] for a in answers] == [f"row {r}" for r in range(64)]


def test_a_one_row_no_push_call_is_unchanged():
    rec = Recorder()
    log = []
    gen = run200.chunked_generate(rec, log)
    answers = gen(None, None, None, "What is your name?", {}, 80, "cpu")
    assert len(rec.calls) == 1 and rec.calls[0]["pushes"] == {} and rec.calls[0]["rows"] == 1
    assert answers == [{"answer": "plain answer", "token_count": 3}]
    assert log == [{"pushed": False, "stage": "run", "rows": 1, "chunks": [1], "observed": [1], "answers": 1,
                    "model_batch": None, "complete": True, "prompt": "What is your name?"}]


def test_part163_battery_call_shapes_are_chunks_of_8_8_8_1():
    """Part 163's registered battery calls are 64, 64, 64 and 8 rows: chunks of 8, 8, 8 and one."""
    assert run200.chunk_sizes(64) == [8] * 8
    assert run200.chunk_sizes(8) == [8]
    assert run200.chunk_sizes(1) == [1]
    log = []
    gen = run200.chunked_generate(Recorder(), log)
    gen(None, None, None, "q", _pushes(8), 80, "cpu")
    assert log[0]["chunks"] == [8] and log[0]["observed"] == [8] and log[0]["complete"] is True
    with pytest.raises(ValueError):
        run200.chunk_sizes(0)


def test_a_short_chunk_is_refused():
    gen = run200.chunked_generate(lambda *a: [{"answer": "x"}])
    with pytest.raises(ValueError):
        gen(None, None, None, "q", _pushes(8), 80, "cpu")


def test_pushes_that_disagree_on_rows_are_refused():
    gen = run200.chunked_generate(Recorder())
    with pytest.raises(ValueError):
        gen(None, None, None, "q", {19: FakePush(range(8)), 20: FakePush(range(7))}, 80, "cpu")


# --------------------------------------------------------------------------------------------- change 1: loader

class _FakeModel:
    def __init__(self, kernel):
        self.config = SimpleNamespace(_attn_implementation=kernel)
        self.calls = []
        self.generate = lambda *a, **k: None      # a real model has generate; the stand-in run163.run calls it

    def requires_grad_(self, value):
        self.calls.append(("requires_grad_", value))

    def eval(self):
        self.calls.append(("eval",))

    def named_modules(self):
        return iter(())


def _fake_load(monkeypatch, kernel, config_kernel=None):
    import neuron_report as nr
    model = _FakeModel(kernel if config_kernel is None else config_kernel)
    loads = []

    def load_model(model_id, revision, device):
        loads.append(device)
        nr.LAST_LOAD = {"attn_implementation": kernel}
        return object(), model
    monkeypatch.setattr(nr, "load_model", load_model)
    return model, loads


@pytest.mark.parametrize("kernel,config_kernel", [("sdpa", None), ("eager", "sdpa"), (None, "eager")])
def test_loader_refuses_a_non_eager_load_before_any_generation(monkeypatch, kernel, config_kernel):
    model, _loads = _fake_load(monkeypatch, kernel, config_kernel)
    loader = run200.eager_loader(SimpleNamespace(model="stub"))
    with pytest.raises(SystemExit) as exc:
        tok, model = loader("cpu")
    assert exc.value.code == 3
    assert model.calls == []                       # no no-grad, no eval: nothing was prepared for generation


def test_loader_accepts_eager_and_applies_part163_settings(monkeypatch):
    model, _loads = _fake_load(monkeypatch, "eager")
    tok, got = run200.eager_loader(SimpleNamespace(model="stub"))("cpu")
    assert got is model and model.calls == [("requires_grad_", False), ("eval",)]


def test_loader_loads_once_and_keeps_the_model_it_loaded(monkeypatch):
    """Change 3: the eager loader remembers (tok, model); a second call does not load again."""
    model, loads = _fake_load(monkeypatch, "eager")
    loader = run200.eager_loader(SimpleNamespace(model="stub"))
    first = loader("cpu")
    second = loader("cpu")
    assert first is second and loads == ["cpu"]
    assert model.calls == [("requires_grad_", False), ("eval",)]


def test_main_with_no_loader_argument_runs_to_completion_on_a_fake_model(tmp_path, monkeypatch, capsys):
    """Review finding 1 (blocker): the production entry point, run200.main([...]), builds no loader and used to crash
    with TypeError at its first load. With neuron_report.load_model monkeypatched and no loader argument, main runs
    run163.run and the batch-1 stage to completion and exits 0."""
    _model, loads = _fake_load(monkeypatch, "eager")
    monkeypatch.setattr(run163, "run", _fake_run163_run("eager"))
    from part160 import run160
    monkeypatch.setattr(run160, "generate_checked", lambda *a, common=None: Recorder()(*a))
    monkeypatch.setattr(run163, "choose_questions", lambda: [{"question": "What is your name?", "kind": "identity"}])
    monkeypatch.setattr(run200, "_push_rows", lambda chunk, *a: {19: FakePush(range(len(chunk)))})
    monkeypatch.setattr(run163.part155, "_common",
                        lambda: SimpleNamespace(find_layers=lambda model: {}, REVISION="r"))
    args = _run_args(tmp_path)
    argv = ["--model", "stub", "--units", args.units, "--out", args.out, "--device", "cpu", "--batch-size", "64"]
    assert run200.main(argv) == 0
    assert loads == ["cpu"]                                  # main's own EagerLoader loaded the fake model once
    rec = json.loads((tmp_path / "load.json").read_text())
    assert rec["loads"][-1]["status"] == "done"
    assert run200.check_record(tmp_path / "load.json") == 0
    assert len((tmp_path / "batch1.jsonl").read_text().splitlines()) == 200
    assert "PART200 gates R1 and R3 passed" in capsys.readouterr().out


# --------------------------------------------------------------------------------------------- Gate R2

def test_gate_r2_requires_the_designs_units_hash(tmp_path):
    """Review finding 2: an altered units file (one neuron changed) fails Gate R2; the real file passes."""
    real = ROOT / "part163" / "units163.json"
    assert run200.EXPECTED_UNITS_SHA256 == "2ed3bef4488eb6b3337dfde100c63c7c5a5df7e8ca809681dec34824c21dd710"
    assert run200.sha256(real) == run200.EXPECTED_UNITS_SHA256
    # the hash half passes with 16 matching questions; the question half is checked by the Part 163 reproduction
    questions = [{"stage": "battery", "question_index": qi, "question": f"q{qi}"} for qi in range(16)]
    pass_r2 = run200.gate_r2(real, real, questions, questions)
    assert pass_r2["passed"] and pass_r2["units_hash_matches_design"] and pass_r2["units_identical"]
    assert run200.gate_r2(real, real, [], [])["passed"] is False          # no questions: the question half fails
    units = json.loads(real.read_text(encoding="utf-8"))
    units[0]["neuron"] = int(units[0]["neuron"]) + 1
    altered = tmp_path / "units163.json"
    altered.write_text(json.dumps(units), encoding="utf-8")
    assert run200.sha256(altered) != run200.EXPECTED_UNITS_SHA256
    # both sides the same altered path: byte-identical, yet not the design's file, so Gate R2 fails
    fail_r2 = run200.gate_r2(altered, altered, questions, questions)
    assert fail_r2["passed"] is False and fail_r2["units_identical"] is True
    assert fail_r2["units_hash_matches_design"] is False
    # and a rerun whose units differ from the published copy fails even when the hash is right on the published side
    assert run200.gate_r2(altered, real, questions, questions)["passed"] is False


def test_compare_exits_3_when_the_rerun_units_file_is_altered(tmp_path):
    """The same through --compare: the rerun's units file is not the design's, so Gate R2 fails and compare exits 3."""
    rerun, published = _compare_fixture(tmp_path)
    _complete_batch1(rerun)
    units = json.loads((ROOT / "part163" / "units163.json").read_text(encoding="utf-8"))
    units[0]["neuron"] = int(units[0]["neuron"]) + 1
    altered = tmp_path / "altered-units.json"
    altered.write_text(json.dumps(units), encoding="utf-8")
    assert run200.main(["--compare", str(rerun), str(published), "--units", str(altered)]) == 3
    assert json.loads((rerun / "compare.json").read_text())["scored"]["gate_r2"]["passed"] is False


# --------------------------------------------------------------------------------------------- change 3: pilot

def test_pilot_min_rate_defaults_to_1_5_in_run163():
    assert run163.PILOT_MIN_RATE == 1.5


def test_wrapper_sets_0_17_for_its_own_run_and_restores_it():
    assert run200.PILOT_MIN_RATE == 0.17
    previous = run200.set_pilot_min_rate(run200.PILOT_MIN_RATE)
    try:
        assert run163.PILOT_MIN_RATE == 0.17
    finally:
        run200.restore_pilot_min_rate(previous)
    assert run163.PILOT_MIN_RATE == 1.5


# --------------------------------------------------------------------------------------------- change 4: batch 1

def _units(n=100):
    units = []
    for i in range(n):
        pair = f"L19D{i}"
        units.extend([
            {"unit_id": f"L19N{i}", "pair_id": pair, "arm": "neuron", "layer": 19, "neuron": i,
             "random_index": None, "stored_answer": "fixture"},
            {"unit_id": pair, "pair_id": pair, "arm": "random", "layer": 19, "neuron": None,
             "random_index": i, "stored_answer": "fixture"},
        ])
    return units


def _one_row_generate(seen):
    def generate_fn(model, tok, layers, question, pushes, max_new_tokens, device):
        rows = next(iter(pushes.values())).rows
        seen.append([row["row"] for row in rows])
        return [{"answer": f"answer {row['row']}", "token_count": 5} for row in rows]
    return generate_fn


def test_batch1_rows_are_question_0_for_every_unit_in_units_order():
    units = _units()
    rows = run200.batch1_rows(units)
    assert len(rows) == 200
    assert [row["unit_id"] for row in rows] == [u["unit_id"] for u in units]
    assert {row["question_index"] for row in rows} == {0}


def test_batch1_writes_200_one_row_calls_in_units_order_and_counts_changed_answers(tmp_path):
    """Change 2/4: each unit is its own ONE-ROW call, in units163.json order, written to batch1.jsonl."""
    units = _units()
    items = [{"question": "What is your name?", "kind": "identity"}]
    seen = []
    generate_fn = run200.chunked_generate(_one_row_generate(seen))     # the wrapper's chunker, as in the real run
    original = run200._push_rows
    run200._push_rows = lambda chunk, *a: {19: FakePush([units.index(row["unit"]) for row in chunk])}
    try:
        written = run200.run_batch1(units, items, {}, {}, None, None, None, "cpu", generate_fn,
                                    tmp_path / "batch1.jsonl")
    finally:
        run200._push_rows = original
    assert written == 200
    assert len(seen) == 200 and all(len(call) == 1 for call in seen)   # 200 calls of one row each
    assert [r for call in seen for r in call] == list(range(200))       # in units163.json order
    batch1 = [json.loads(line) for line in (tmp_path / "batch1.jsonl").read_text().splitlines()]
    assert [row["unit_id"] for row in batch1] == [u["unit_id"] for u in units]
    assert [row["stage"] for row in batch1] == ["batch1"] * 200
    # the battery answers in units order: every unit's text matches, so none changed
    battery = [{"stage": "battery", "arm": u["arm"], "unit_id": u["unit_id"], "question_index": 0,
                "answer": f"answer {i}"} for i, u in enumerate(units)]
    changes = run200.batch1_changes(batch1, battery)
    assert changes["changed"] == 0 and changes["shape_sensitive"] is False
    assert changes["neuron"] == {"rows": 100, "changed": 0} and changes["random"] == {"rows": 100, "changed": 0}
    for row in batch1[:3]:
        row["answer"] = "different"
    changes = run200.batch1_changes(batch1, battery)
    assert changes["changed"] == 3
    assert changes["neuron"]["changed"] + changes["random"]["changed"] == 3


def test_compare_exits_3_on_an_incomplete_batch1_file(tmp_path, capsys):
    """Review finding 3: 150 rows gives exit 3 with a message, and no compare.json is written (21 of 150 differing is
    14%, over the design's 20-of-200 bar, so it must not compare as clean)."""
    rerun, published = _compare_fixture(tmp_path)
    units = json.loads((ROOT / "part163" / "units163.json").read_text(encoding="utf-8"))[:150]
    rows = _complete_batch1(rerun, units)
    for row in rows[:21]:
        row["answer"] = "changed"
    (rerun / "batch1.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert run200.main(["--compare", str(rerun), str(published)]) == 3
    out = capsys.readouterr().out
    assert "batch 1 is not complete" in out and "150" in out
    assert not (rerun / "compare.json").exists()


def test_compare_exits_3_when_batch1_repeats_a_unit(tmp_path):
    """200 rows but one unit twice and another missing: not one row per unit, so exit 3."""
    rerun, published = _compare_fixture(tmp_path)
    units = json.loads((ROOT / "part163" / "units163.json").read_text(encoding="utf-8"))
    rows = _complete_batch1(rerun, units[:-1] + [units[0]])
    (rerun / "batch1.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert run200.main(["--compare", str(rerun), str(published)]) == 3


def test_batch1_complete_accepts_exactly_the_200_units():
    units = _units()
    rows = [{"unit_id": u["unit_id"]} for u in units]
    assert run200.batch1_complete(rows, units) == (True, "")
    assert run200.batch1_complete(rows[:199], units)[0] is False
    assert run200.batch1_complete(rows + [rows[0]], units)[0] is False          # 201 rows, one repeated
    swapped = rows[:-1] + [{"unit_id": "not-a-unit"}]
    assert run200.batch1_complete(swapped, units)[0] is False


def test_batch1_is_resumable(tmp_path):
    """Change 4: units already written are skipped; a relaunch after completion writes nothing."""
    units = _units()
    path = tmp_path / "batch1.jsonl"
    path.write_text(json.dumps({"stage": "batch1", "unit_id": units[0]["unit_id"], "question_index": 0}) + "\n")
    calls = []
    original = run200._push_rows
    run200._push_rows = lambda chunk, *a: {19: FakePush(range(len(chunk)))}
    try:
        written = run200.run_batch1(units, [{"question": "q", "kind": "identity"}], {}, {}, None, None, None,
                                    "cpu", _one_row_generate(calls), path)
        assert written == 199 and len(calls) == 199
        assert len(path.read_text().splitlines()) == 200
        before = path.read_bytes()
        again = run200.run_batch1(units, [{"question": "q", "kind": "identity"}], {}, {}, None, None, None,
                                  "cpu", _one_row_generate(calls), path)
    finally:
        run200._push_rows = original
    assert again == 0 and len(calls) == 199 and path.read_bytes() == before


def test_shape_sensitive_flag_is_more_than_20():
    assert run200.SHAPE_SENSITIVE_LIMIT == 20
    batch1 = [{"unit_id": f"u{i}", "arm": "neuron" if i % 2 else "random", "answer": "same"} for i in range(200)]
    battery = [{"stage": "battery", "unit_id": f"u{i}", "question_index": 0, "answer": "same"} for i in range(200)]
    for k in (20, 21):
        rows = [dict(row) for row in batch1]
        for row in rows[:k]:
            row["answer"] = "changed"
        assert run200.batch1_changes(rows, battery)["shape_sensitive"] is (k > 20)


# --------------------------------------------------------------------------------------------- Gate R3

PUSH64 = {"pushed": True, "rows": 64, "chunks": [8] * 8, "observed": [8] * 8, "complete": True}
PUSH8 = {"pushed": True, "rows": 8, "chunks": [8], "observed": [8], "complete": True}
PLAIN = {"pushed": False, "rows": 1, "chunks": [1], "observed": [1], "complete": True}


def test_gate_r3_reads_the_observed_chunks():
    good = run200.call_shapes([PUSH64] * 64 + [PUSH8] * 64 + [PLAIN] * 16)
    assert good["passed"] and good["pushed_chunks"] == 576 and good["largest_chunk_rows"] == 8
    assert not run200.call_shapes([{**PUSH64, "observed": [64]}])["passed"]          # the plan said 8s, the model got 64
    assert not run200.call_shapes([{**PUSH64, "observed": [8] * 7}])["passed"]        # rows lost
    assert not run200.call_shapes([{**PUSH64, "observed": [8, 9, 8, 8, 8, 8, 8, 7]}])["passed"]   # a 9-row chunk
    # a 7-row chunk that still covers 64 rows (7 + 7*8 + 1 = 64): only the exactly-8 rule fails it
    assert not run200.call_shapes([{**PUSH64, "observed": [7, 8, 8, 8, 8, 8, 8, 8, 1]}])["passed"]
    assert not run200.call_shapes([{**PLAIN, "observed": [8]}])["passed"]             # a no-push call of 8 rows
    assert not run200.call_shapes([PLAIN])["passed"]                                  # no pushed call at all
    assert not run200.call_shapes([{**PUSH64, "observed": [8], "complete": False}])["passed"]   # no completed push
    killed = {**PUSH64, "observed": [8, 8, 8], "complete": False}                     # a kill mid-call: fewer, never more
    assert run200.call_shapes([killed, PUSH64])["passed"]
    assert not run200.call_shapes([{**PUSH64, "observed": [8] * 9, "complete": False}])["passed"]


def test_gate_r3_reads_a_batch1_one_row_call_as_single_row():
    """Change 5: batch-1 calls are pushed one-row calls; they pass as single rows and are not held to chunks of 8."""
    batch1 = {"pushed": True, "stage": "batch1", "rows": 1, "chunks": [1], "observed": [1], "complete": True}
    assert run200.call_shapes([PUSH64, batch1])["passed"]
    assert run200.call_shapes([PUSH64, batch1])["no_push_calls"] == 1
    assert not run200.call_shapes([PUSH64, {**batch1, "observed": [8]}])["passed"]
    assert not run200.call_shapes([PUSH64, {**batch1, "observed": [7]}])["passed"]


def test_gate_r3_requires_exactly_8_rows_a_7_row_chunk_fails():
    """Change 5: a 7-row chunk fails even when it covers the call's rows (64 = 8*7 + 8)."""
    assert run200.call_shapes([PUSH64])["passed"]
    seven = {**PUSH64, "observed": [7, 8, 8, 8, 8, 8, 8, 8, 1]}      # covers 64 rows, one chunk of 7
    assert sum(seven["observed"]) == 64
    assert run200.call_shapes([seven])["passed"] is False
    assert run200.call_shapes([seven])["pushed_calls_not_in_chunks_of_8"] == 1
    # the 9-row chunk that still covers 64 rows (the build report's test) keeps failing
    assert not run200.call_shapes([{**PUSH64, "observed": [8, 9, 8, 8, 8, 8, 8, 7]}])["passed"]


class BatchModel:
    """A fake model whose generate() records the batch it was handed, as battery148.generate hands it one: the
    generator below calls model.generate(inputs) with input_ids expanded to the push's rows."""

    def __init__(self):
        self.config = SimpleNamespace(_attn_implementation="eager")
        self.seen = []
        self.generate = self._generate

    def _generate(self, inputs, **_kwargs):
        self.seen.append(int(inputs["input_ids"].shape[0]))
        return None

    def requires_grad_(self, value):
        return self

    def eval(self):
        return self


class BatchRecorder:
    """Stands in for part160.generate_checked: calls model.generate with the chunk's batch, then answers."""
    def __init__(self):
        self.rows = []

    def __call__(self, model, tok, layers, question, pushes, max_new_tokens, device):
        n = int(next(iter(pushes.values())).shape[0]) if pushes else 1
        self.rows.append(n)
        if model is not None:
            model.generate({"input_ids": SimpleNamespace(shape=(n, 7))}, max_new_tokens=max_new_tokens)
        return [{"answer": f"row {r}", "token_count": 4} for r in range(n)]


def test_gate_r3_records_the_batch_model_generate_received():
    """Review finding 4: the model batch is observed per chunk (a wrapper on model.generate, installed for the call
    only), recorded, and Gate R3 checks it equals the chunk size."""
    model, log = BatchModel(), []
    gen = run200.chunked_generate(BatchRecorder(), log)
    gen(model, None, None, "q", _pushes(64), 80, "cpu")
    assert model.seen == [8] * 8
    assert log[0]["model_batch"] == [8] * 8
    assert log[0]["push_rows"] == [8] * 8 and log[0]["answers"] == [8] * 8
    assert run200.call_shapes(log)["passed"] and run200.call_shapes(log)["chunks_with_model_batch"] == 8
    assert model.generate == model._generate                 # restored: our wrapper is gone
    # a record whose model saw the whole 64-row call fails Gate R3 even though the plan says 8
    bad = {**log[0], "model_batch": [64]}
    assert run200.call_shapes([bad])["passed"] is False
    # a record whose generator returned another number of answers fails too
    assert run200.call_shapes([{**log[0], "answers": [8] * 7 + [7]}])["passed"] is False
    # a record whose push tensors were not the chunk's rows fails too
    assert run200.call_shapes([{**log[0], "push_rows": [64]}])["passed"] is False


def test_a_no_push_call_records_its_measured_answer_count():
    """Review finding 4: the no-push calls record the measured len(answers), not a hard-coded 1."""
    log = []
    gen = run200.chunked_generate(BatchRecorder(), log)
    gen(BatchModel(), None, None, "q", {}, 80, "cpu")
    assert log[0]["answers"] == 1 and log[0]["observed"] == [1]
    assert run200.call_shapes([PUSH64, *log])["passed"]
    assert run200.call_shapes([PUSH64, {**log[0], "answers": 2}])["passed"] is False


def test_check_record_passes_a_good_record_and_fails_a_non_eager_one_and_a_bad_chunk(tmp_path):
    entry = {"load": {"attn_implementation": "eager"}, "model": "m", "revision": "r", "out_dir": "o",
             "calls": [PUSH64, PUSH8, PLAIN]}
    path = tmp_path / "load.json"
    path.write_text(json.dumps({"part": 200, "loads": [entry]}), encoding="utf-8")
    assert run200.check_record(path) == 0
    path.write_text(json.dumps({"part": 200, "loads": [{**entry, "load": {"attn_implementation": "sdpa"}}]}),
                    encoding="utf-8")
    assert run200.check_record(path) == 3
    nine = {**PUSH64, "observed": [8, 9, 8, 8, 8, 8, 8, 7]}
    path.write_text(json.dumps({"part": 200, "loads": [{**entry, "calls": [nine]}]}), encoding="utf-8")
    assert run200.check_record(path) == 3
    seven = {**PUSH64, "observed": [7, 8, 8, 8, 8, 8, 8, 8, 1]}       # covers 64 rows, one chunk of 7
    path.write_text(json.dumps({"part": 200, "loads": [{**entry, "calls": [seven]}]}), encoding="utf-8")
    assert run200.check_record(path) == 3
    path.write_text(json.dumps({"part": 200, "loads": [entry, {**entry, "out_dir": "other"}]}), encoding="utf-8")
    assert run200.check_record(path) == 3


# --------------------------------------------------------------------------------------------- the run

def _fake_run163_run(record_kernel="eager", rows=3, stop=None):
    def run(args, loader=None, generate_fn=None):
        import neuron_report as nr
        nr.LAST_LOAD = {"attn_implementation": record_kernel}
        tok, model = loader("cpu")
        if stop is not None:
            raise SystemExit(stop)
        answers = generate_fn(model, tok, {}, "q", _pushes(8), 80, "cpu")
        Path(args.out).write_text("".join(json.dumps({"stage": "battery", "unit_id": f"u{i}", "question_index": 0,
                                                      "answer": a["answer"]}) + "\n"
                                          for i, a in enumerate(answers)), encoding="utf-8")
        return {"answers_written": rows}
    return run


def _run_args(tmp_path):
    units_path = tmp_path / "units.json"
    units_path.write_text(json.dumps(_units()))
    return SimpleNamespace(model="stub", out=str(tmp_path / "answers.jsonl"), device="cpu", units=str(units_path),
                           batch_size=64)


def test_run_refuses_a_non_eager_launch_with_exit_3_before_generation(tmp_path, monkeypatch):
    model, _loads = _fake_load(monkeypatch, "sdpa")
    generated = []
    monkeypatch.setattr(run163, "run", _fake_run163_run("sdpa"))
    args = _run_args(tmp_path)
    assert run200.run(args, loader=run200.eager_loader(args),
                      generate_fn=lambda *a: generated.append(a) or []) == 3
    assert generated == [] and model.calls == []
    assert not (tmp_path / "run200.lock").exists()          # the lock is released on the way out


def test_run_records_the_eager_load_and_its_chunks_and_runs_batch1_after(tmp_path, monkeypatch):
    _fake_load(monkeypatch, "eager")
    monkeypatch.setattr(run163, "run", _fake_run163_run("eager"))
    from part160 import run160
    monkeypatch.setattr(run160, "generate_checked", lambda *a, common=None: Recorder()(*a))
    monkeypatch.setattr(run163, "choose_questions", lambda: [{"question": "What is your name?", "kind": "identity"}])
    monkeypatch.setattr(run200, "_push_rows", lambda chunk, *a: {19: FakePush(range(len(chunk)))})
    monkeypatch.setattr(run163.part155, "_common",
                        lambda: SimpleNamespace(find_layers=lambda model: {}, REVISION="r"))
    args = _run_args(tmp_path)
    # the stand-in run163.run generates through the wrapper's chunked generator, which the loader records
    assert run200.run(args, loader=run200.eager_loader(args), generate_fn=Recorder()) == 0
    rec = json.loads((tmp_path / "load.json").read_text())
    assert len(rec["loads"]) == 1                            # one entry per launch: no second load
    assert rec["loads"][-1]["load"]["attn_implementation"] == "eager"
    assert rec["loads"][-1]["status"] == "done"
    calls = rec["loads"][-1]["calls"]
    assert calls[0] == {"pushed": True, "stage": "run", "rows": 8, "chunks": [8], "observed": [8], "complete": True,
                        "prompt": "q", "push_rows": [8], "answers": [8], "model_batch": [None]}
    # the 200 batch-1 calls follow in the same entry, each a single row
    assert len(calls) == 201 and all(call["observed"] == [1] for call in calls[1:])
    assert run200.check_record(tmp_path / "load.json") == 0
    assert run163.PILOT_MIN_RATE == 1.5                      # restored after the run
    # the batch-1 stage ran after run163.run: 200 rows in units order, question 0
    batch1 = [json.loads(line) for line in (tmp_path / "batch1.jsonl").read_text().splitlines()]
    assert len(batch1) == 200
    assert [row["unit_id"] for row in batch1] == [u["unit_id"] for u in _units()]
    assert all(row["question"] == "What is your name?" for row in batch1)
    assert not (tmp_path / "repeat.jsonl").exists()


def test_run_loads_the_model_once(tmp_path, monkeypatch):
    """Change 3: run163.run loads through the loader, then the batch-1 stage reuses that model: one load_model call."""
    _model, loads = _fake_load(monkeypatch, "eager")
    monkeypatch.setattr(run163, "run", _fake_run163_run("eager"))
    monkeypatch.setattr(run163, "choose_questions", lambda: [{"question": "What is your name?", "kind": "identity"}])
    monkeypatch.setattr(run200, "_push_rows", lambda chunk, *a: {19: FakePush(range(len(chunk)))})
    monkeypatch.setattr(run163.part155, "_common",
                        lambda: SimpleNamespace(find_layers=lambda model: {}, REVISION="r"))
    args = _run_args(tmp_path)
    assert run200.run(args, loader=run200.eager_loader(args), generate_fn=Recorder()) == 0
    assert loads == ["cpu"]                                  # load_model called once in the process
    assert len(json.loads((tmp_path / "load.json").read_text())["loads"]) == 1


def test_run_finishes_batch1_on_a_relaunch_that_wrote_no_rows(tmp_path, monkeypatch):
    """Change 4: batch 1 runs on every launch until all 200 units are written, even when run163.run wrote nothing."""
    _fake_load(monkeypatch, "eager")
    monkeypatch.setattr(run163, "run", _fake_run163_run("eager", rows=0))
    monkeypatch.setattr(run163, "choose_questions", lambda: [{"question": "What is your name?", "kind": "identity"}])
    monkeypatch.setattr(run200, "_push_rows", lambda chunk, *a: {19: FakePush(range(len(chunk)))})
    monkeypatch.setattr(run163.part155, "_common",
                        lambda: SimpleNamespace(find_layers=lambda model: {}, REVISION="r"))
    args = _run_args(tmp_path)
    assert run200.run(args, loader=run200.eager_loader(args), generate_fn=Recorder()) == 0
    assert len((tmp_path / "batch1.jsonl").read_text().splitlines()) == 200
    # a relaunch after completion writes nothing more
    before = (tmp_path / "batch1.jsonl").read_bytes()
    assert run200.run(args, loader=run200.eager_loader(args), generate_fn=Recorder()) == 0
    assert (tmp_path / "batch1.jsonl").read_bytes() == before


@pytest.mark.parametrize("stop", ["Pilot stop rule: 0.10 answers/s is below 0.17",
                                 "Pilot stop rule: recipe changed-answer share is below 0.70 in an arm"])
def test_run_records_a_pilot_stop_and_exits_3(tmp_path, monkeypatch, stop):
    """Change 8: both SystemExit paths from run163.run are recorded with status stopped and the wrapper exits 3."""
    _fake_load(monkeypatch, "eager")
    monkeypatch.setattr(run163, "run", _fake_run163_run("eager", stop=stop))
    monkeypatch.setattr(run200, "_push_rows", lambda chunk, *a: {19: FakePush(range(len(chunk)))})
    monkeypatch.setattr(run163.part155, "_common",
                        lambda: SimpleNamespace(find_layers=lambda model: {}, REVISION="r"))
    args = _run_args(tmp_path)
    assert run200.run(args, loader=run200.eager_loader(args), generate_fn=Recorder()) == 3
    entry = json.loads((tmp_path / "load.json").read_text())["loads"][-1]
    assert entry["status"] == "stopped" and stop in entry["failure"]
    assert not (tmp_path / "batch1.jsonl").exists()           # no batch-1 stage after a stop


def test_a_refused_launch_is_recorded_and_gate_r1_reads_it_as_a_refusal(tmp_path, monkeypatch):
    """Review finding 7: a non-eager load writes a load.json entry with status refused and the LAST_LOAD it saw, then
    exits 3; Gate R1 treats the entry as a refusal, not as a pass, even when LAST_LOAD says eager."""
    import neuron_report as nr
    model = _FakeModel("sdpa")                       # the model's config says sdpa; LAST_LOAD will say eager

    def load_model(model_id, revision, device):
        nr.LAST_LOAD = {"attn_implementation": "eager"}
        return object(), model
    monkeypatch.setattr(nr, "load_model", load_model)
    args = _run_args(tmp_path)
    assert run200.run(args) == 3                     # no loader argument: run() builds the EagerLoader itself
    rec = json.loads((tmp_path / "load.json").read_text())
    entry = rec["loads"][-1]
    assert entry["status"] == "refused"
    assert entry["load"] == {"attn_implementation": "eager"}      # the LAST_LOAD it saw
    assert entry["calls"] == [] and model.calls == []             # nothing generated, nothing prepared
    assert not (tmp_path / "batch1.jsonl").exists()
    # Part 199's check_kernels alone would pass this record (LAST_LOAD says eager); record_gates refuses it
    assert run199.check_kernels(rec["loads"])[0] is True
    ok, why, _shapes = run200.record_gates(rec)
    assert ok is False and "refused" in why
    assert run200.check_record(tmp_path / "load.json") == 3


def test_folder_lock_is_run200_lock_and_refuses_a_second_launch(tmp_path, monkeypatch):
    _fake_load(monkeypatch, "eager")
    monkeypatch.setattr(run163, "run", _fake_run163_run("eager"))
    (tmp_path / "run200.lock").write_text("pid 1\n")
    args = _run_args(tmp_path)
    with pytest.raises(SystemExit) as exc:
        run200.run(args, loader=run200.eager_loader(args))
    assert exc.value.code == 3
    assert (tmp_path / "run200.lock").exists()              # a stale lock is removed by hand, not reclaimed
    assert not (tmp_path / "run199.lock").exists()


def test_the_lock_file_is_named_run200_lock(tmp_path):
    assert run200.LOCK_NAME == "run200.lock"
    lock = run200.Run200Lock(tmp_path)
    assert lock.path == tmp_path / "run200.lock"
    with lock:
        assert (tmp_path / "run200.lock").exists()
    assert not (tmp_path / "run200.lock").exists()


# --------------------------------------------------------------------------------------------- startup repair

def _planned_rows(units, n_questions=16):
    """A whole run's rows in run163.run's order: 16 no-push, the recipe (one row per unit), then each battery question
    in turn (one row per unit)."""
    rows = [{"stage": "no_push", "unit_id": "no_push", "question_index": qi, "answer": "a"} for qi in range(n_questions)]
    rows += [{"stage": "recipe", "unit_id": u["unit_id"], "question_index": 0, "answer": "a"} for u in units]
    for qi in range(n_questions):
        rows += [{"stage": "battery", "unit_id": u["unit_id"], "question_index": qi, "answer": "a"} for u in units]
    assert len(rows) == run200.NO_PUSH_ROWS + run200.RECIPE_ROWS + run200.BATTERY_ROWS
    return rows


def _write_jsonl(path, rows, torn=None):
    text = "".join(json.dumps(row) + "\n" for row in rows)
    if torn is not None:
        text += torn
    Path(path).write_bytes(text.encode("utf-8"))


def _repair_args(tmp_path, units):
    units_path = tmp_path / "units.json"
    units_path.write_text(json.dumps(units))
    return SimpleNamespace(model="stub", out=str(tmp_path / "answers.jsonl"), device="cpu", units=str(units_path),
                           batch_size=64)


def test_a_call_cut_at_30_rows_is_cut_off_and_the_relaunch_regenerates_whole_calls(tmp_path, monkeypatch):
    """Review finding 5: a 64-row call cut at 30 rows (a kill inside run163's row-write loop) is cut back to the last
    whole call before run163.run reads the file; the relaunch regenerates the call whole, Gate R3 passes, and no row is
    duplicated."""
    units = _units()
    plan = _planned_rows(units)
    # rows 0-15 no-push, 16-215 recipe (calls of 64, 64, 64, 8), then battery question 0: calls of 64, 64, 64, 8.
    # Keep through the recipe (216 rows) and 30 rows of battery question 0's first call: 246 rows of 3,616.
    kept = plan[:216 + 30]
    _write_jsonl(tmp_path / "answers.jsonl", kept)
    repair = run200.repair_startup(tmp_path)
    assert repair["answers"]["partial_call_rows"] == 30 and repair["answers"]["torn_line_rows"] == 0
    assert repair["answers"]["rows_kept"] == 216
    assert repair["answers"]["resume_at"] == {"stage": "battery", "question_index": 0, "row_in_stage": 0}
    assert repair["answers"]["bytes_cut"] > 0
    assert len((tmp_path / "answers.jsonl").read_text().splitlines()) == 216

    # The relaunch: the stand-in run163.run reads the file as run163 does, generates only what is missing (through the
    # wrapper's chunker, so every call is chunked), and appends. The same repair runs first, and cuts nothing.
    def fake_run(args, loader=None, generate_fn=None):
        import neuron_report as nr
        nr.LAST_LOAD = {"attn_implementation": "eager"}
        loader("cpu")
        done = {(r["stage"], r["unit_id"], r["question_index"]) for r in run200._read_jsonl(args.out)}
        pending = [r for r in plan if (r["stage"], r["unit_id"], r["question_index"]) not in done]
        with Path(args.out).open("a", encoding="utf-8") as fh:
            for start in range(0, len(pending), 64):          # run163's call size; the wrapper chunks to 8
                block = pending[start:start + 64]
                pushes = {19: FakePush(range(len(block)))} if block[0]["stage"] != "no_push" else {}
                answers = generate_fn(None, None, {}, "q", pushes, 80, "cpu")
                for row, answer in zip(block, answers):
                    fh.write(json.dumps({**row, "answer": answer["answer"]}) + "\n")
                fh.flush()
        return {"answers_written": len(pending)}

    _fake_load(monkeypatch, "eager")
    monkeypatch.setattr(run163, "run", fake_run)
    monkeypatch.setattr(run163, "choose_questions", lambda: [{"question": "What is your name?", "kind": "identity"}])
    monkeypatch.setattr(run200, "_push_rows", lambda chunk, *a: {19: FakePush(range(len(chunk)))})
    monkeypatch.setattr(run163.part155, "_common", lambda: SimpleNamespace(find_layers=lambda model: {}, REVISION="r"))
    args = _repair_args(tmp_path, units)
    assert run200.run(args, generate_fn=Recorder()) == 0
    rows = run200._read_jsonl(tmp_path / "answers.jsonl")
    keys = [(r["stage"], r["unit_id"], r["question_index"]) for r in rows]
    assert len(keys) == len(set(keys)) == len(plan)                 # no row duplicated, every row present
    # the relaunch's first battery call is the whole 64-row call again: chunks of exactly 8, never a 34-row remainder
    calls = json.loads((tmp_path / "load.json").read_text())["loads"][-1]["calls"]
    battery_calls = [call for call in calls if call.get("stage") == "run" and call.get("rows") == 64]
    assert battery_calls and all(call["chunks"] == [8] * 8 for call in battery_calls)
    assert all(size == 8 for call in calls if call.get("pushed") and call.get("stage") != "batch1"
               for size in call.get("observed", []))
    assert all(call.get("observed") == [1] for call in calls if not call.get("pushed"))
    assert run200.check_record(tmp_path / "load.json") == 0
    entry = json.loads((tmp_path / "load.json").read_text())["loads"][-1]
    assert entry["repair"]["answers"]["rows_cut"] == 0 and entry["repair"]["answers"]["torn_line_rows"] == 0


def test_repair_cuts_a_torn_last_line_in_each_file(tmp_path):
    """Review finding 6: a torn final line (no trailing newline) in answers.jsonl and in batch1.jsonl is cut at
    startup, so the relaunch's readers do not raise."""
    units = _units()
    plan = _planned_rows(units)
    _write_jsonl(tmp_path / "answers.jsonl", plan[:216], torn='{"stage": "battery", "unit_id": "L19N')
    _write_jsonl(tmp_path / "batch1.jsonl", plan[:5], torn='{"stage": "batch1", "unit_id": "L19')
    repair = run200.repair_startup(tmp_path)
    assert repair["answers"]["torn_line_rows"] == 1 and repair["answers"]["torn_line_bytes"] > 0
    assert repair["batch1"]["torn_line_rows"] == 1 and repair["batch1"]["bytes_cut"] > 0
    assert repair["answers"]["partial_call_rows"] == 0
    for name in ("answers.jsonl", "batch1.jsonl"):
        data = (tmp_path / name).read_bytes()
        assert data.endswith(b"\n")
        for line in data.decode("utf-8").splitlines():
            json.loads(line)                                          # every line parses: nothing torn is left


def test_a_relaunch_after_a_torn_line_in_each_file_succeeds(tmp_path, monkeypatch):
    """The same through run(): a torn last line in answers.jsonl and in batch1.jsonl, then a relaunch that completes
    (run163.run's stand-in reads answers.jsonl as run163.read_jsonl does; the batch-1 stage reads batch1.jsonl)."""
    units = _units()
    plan = _planned_rows(units)
    _write_jsonl(tmp_path / "answers.jsonl", plan[:216], torn='{"stage": "batt')
    batch1_prefix = [{"stage": "batch1", "unit_id": u["unit_id"], "arm": u["arm"], "answer": "old"}
                     for u in units[:5]]
    _write_jsonl(tmp_path / "batch1.jsonl", batch1_prefix, torn='{"stage": "batch1"')
    _fake_load(monkeypatch, "eager")
    def resumed_run(args, loader=None, generate_fn=None):
        import neuron_report as nr
        nr.LAST_LOAD = {"attn_implementation": "eager"}
        tok, model = loader("cpu")
        generate_fn(model, tok, {}, "q", _pushes(8), 80, "cpu")
        return {"answers_written": 0}
    monkeypatch.setattr(run163, "run", resumed_run)
    from part160 import run160
    monkeypatch.setattr(run160, "generate_checked", lambda *a, common=None: Recorder()(*a))
    monkeypatch.setattr(run163, "choose_questions", lambda: [{"question": "What is your name?", "kind": "identity"}])
    monkeypatch.setattr(run200, "_push_rows", lambda chunk, *a: {19: FakePush(range(len(chunk)))})
    monkeypatch.setattr(run163.part155, "_common", lambda: SimpleNamespace(find_layers=lambda model: {}, REVISION="r"))
    args = _repair_args(tmp_path, units)
    assert run200.run(args) == 0
    assert len((tmp_path / "batch1.jsonl").read_text().splitlines()) == 200
    entry = json.loads((tmp_path / "load.json").read_text())["loads"][-1]
    assert entry["repair"]["answers"]["torn_line_rows"] == 1 and entry["repair"]["batch1"]["torn_line_rows"] == 1


def test_repair_leaves_a_whole_file_alone(tmp_path):
    units = _units()
    plan = _planned_rows(units)
    _write_jsonl(tmp_path / "answers.jsonl", plan)
    before = (tmp_path / "answers.jsonl").read_bytes()
    repair = run200.repair_startup(tmp_path)
    assert repair["answers"]["rows_cut"] == 0 and repair["answers"]["bytes_cut"] == 0
    assert repair["answers"]["resume_at"] is None
    assert repair["batch1"]["rows_cut"] == 0 and not (tmp_path / "batch1.jsonl").exists()
    assert (tmp_path / "answers.jsonl").read_bytes() == before
    # a file cut exactly at a call boundary (the recipe's last call, 8 rows) is whole: nothing to cut
    _write_jsonl(tmp_path / "answers.jsonl", plan[:216])
    assert run200.repair_startup(tmp_path)["answers"]["rows_cut"] == 0
    # one row into a battery call: cut back to the boundary
    _write_jsonl(tmp_path / "answers.jsonl", plan[:217])
    repair = run200.repair_startup(tmp_path)
    assert repair["answers"]["partial_call_rows"] == 1 and repair["answers"]["rows_kept"] == 216


def test_first_incomplete_call_follows_run163s_plan_order():
    """The plan is run163's own order: no-push rows, the recipe in units order in blocks of 64, then each battery
    question in units order in blocks of 64."""
    assert run200._first_incomplete_call(0, 64) == {"stage": "no_push", "question_index": 0}
    assert run200._first_incomplete_call(16, 64) == {"stage": "recipe", "question_index": 0, "row_in_stage": 0}
    assert run200._first_incomplete_call(16 + 128, 64) == {"stage": "recipe", "question_index": 0, "row_in_stage": 128}
    assert run200._first_incomplete_call(216, 64) == {"stage": "battery", "question_index": 0, "row_in_stage": 0}
    assert run200._first_incomplete_call(216 + 64, 64) == {"stage": "battery", "question_index": 0, "row_in_stage": 64}
    assert run200._first_incomplete_call(216 + 192, 64) == {"stage": "battery", "question_index": 1, "row_in_stage": 0}


# --------------------------------------------------------------------------------------------- --compare

def test_prediction_outcomes_for_200_1_to_200_7_hold_and_fail():
    expected_verdicts = {"163.1": "held", "163.2": "held", "163.3": "held", "163.4": "held", "163.5": "failed"}

    def fixture(passed):
        verdicts = expected_verdicts if passed else {key: "failed" for key in expected_verdicts}
        return {
            "record_r1_r3": {"passed": passed},
            "gate_r2": {"passed": passed},
            "rerun": {
                "part163_gates_and_verdicts": {"gates": {"g1": {"passed": passed}},
                                                "all_gates_passed": passed,
                                                "verdicts": {key: {"verdict": value} for key, value in verdicts.items()}},
                "broken_and_cap": {"neuron": {"broken": 500 if passed else 300, "at_cap": 500 if passed else 300},
                                   "random": {"broken": 100 if passed else 200, "at_cap": 500 if passed else 300}},
                "part161_qwen_on_maker": {"neuron": {"qwen": 10, "eligible_maker_answers": 100,
                                                          "rate": .10 if passed else .30},
                                           "random": {"qwen": 20, "eligible_maker_answers": 100,
                                                          "rate": .20 if passed else .15}},
            },
            "batch1": {"changed": 20 if passed else 21,
                       "neuron": {"rows": 100, "changed": 10 if passed else 11},
                       "random": {"rows": 100, "changed": 10}},
        }

    held = run200.prediction_outcomes(fixture(True))
    failed = run200.prediction_outcomes(fixture(False))
    assert {key: result["result"] for key, result in held.items()} == {key: "held" for key in held}
    assert all(result["result"].startswith("failed") for result in failed.values())


def _compare_fixture(tmp_path, record_kernel="eager", calls=(PUSH64, PUSH8, PLAIN)):
    for name in ("rerun", "published"):
        (tmp_path / name).mkdir()
    (tmp_path / "rerun" / "answers.jsonl").write_text("")
    (tmp_path / "rerun" / "batch1.jsonl").write_text("")
    (tmp_path / "published" / "answers.jsonl").write_text("")
    (tmp_path / "published" / "units163.json").write_bytes((ROOT / "part163" / "units163.json").read_bytes())
    (tmp_path / "published" / "load.json").write_text(json.dumps({"part": 163, "loads": []}))
    (tmp_path / "rerun" / "load.json").write_text(json.dumps({"part": 200, "loads": [
        {"load": {"attn_implementation": record_kernel}, "model": "m", "revision": "r", "out_dir": "o",
         "calls": list(calls)}]}))
    return tmp_path / "rerun", tmp_path / "published"


def _complete_batch1(rerun, units=None):
    """A batch1.jsonl that passes --compare's completeness check: the 200 units, one row each, in units order."""
    units = units if units is not None else json.loads((ROOT / "part163" / "units163.json").read_text())
    rows = [{"stage": "batch1", "unit_id": u["unit_id"], "arm": u["arm"], "question_index": 0, "answer": ""}
            for u in units]
    (Path(rerun) / "batch1.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def test_compare_exits_3_when_the_rerun_record_is_not_eager(tmp_path, capsys):
    rerun, published = _compare_fixture(tmp_path, record_kernel="sdpa")
    _complete_batch1(rerun)
    assert run200.main(["--compare", str(rerun), str(published)]) == 3
    assert (rerun / "compare.json").exists()
    assert "non-eager load(s) recorded" in capsys.readouterr().out


def test_compare_exits_3_when_a_chunk_has_9_rows(tmp_path):
    rerun, published = _compare_fixture(tmp_path, calls=({**PUSH64, "observed": [9] + [8] * 7},))
    _complete_batch1(rerun)
    assert run200.main(["--compare", str(rerun), str(published)]) == 3


def test_compare_exits_2_when_a_file_is_missing(tmp_path, capsys):
    rerun, published = _compare_fixture(tmp_path)
    (published / "answers.jsonl").unlink()
    assert run200.main(["--compare", str(rerun), str(published)]) == 2
    assert "missing file" in capsys.readouterr().out


def test_compare_on_part163_against_itself_reproduces_the_published_numbers(tmp_path):
    """The design's own check: run on Part 163's folder as both, --compare reproduces 19 of 539 and 57 of 661, 343 and
    83 broken, 124 and 218 battery answers at the 80-token cap, the published mid-phrase baseline, and Part 163's five
    verdicts."""
    part163 = ROOT / "results" / "part163"
    if not (part163 / "answers.jsonl").exists():
        pytest.skip("results/part163/answers.jsonl is not in this checkout")
    rerun = tmp_path / "rerun"
    rerun.mkdir()
    (rerun / "answers.jsonl").write_bytes((part163 / "answers.jsonl").read_bytes())
    # Fold round 2: --compare needs a complete batch-1 file. The rerun here is Part 163's own answers, so the batch-1
    # rows are its question-0 battery answers: every one then matches its chunked answer (0 changed).
    q0 = {row["unit_id"]: row for row in (json.loads(line) for line in
                                          (part163 / "answers.jsonl").read_text(encoding="utf-8").splitlines() if line)
          if row.get("stage") == "battery" and int(row.get("question_index", -1)) == 0}
    units = json.loads((ROOT / "part163" / "units163.json").read_text())
    _complete_batch1(rerun, units)
    rows = [json.loads(line) for line in (rerun / "batch1.jsonl").read_text().splitlines()]
    for row in rows:
        row["answer"] = q0[row["unit_id"]]["answer"]
    (rerun / "batch1.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (rerun / "load.json").write_text(json.dumps({"part": 200, "loads": [
        {"load": {"attn_implementation": "eager"}, "model": "m", "revision": "r", "out_dir": "o",
         "calls": [PUSH64] * 64 + [PUSH8] * 64 + [PLAIN] * 16}]}))
    # the published folder's units copy: Gate R2 compares the rerun's units hash with it as well as with the design's
    published = tmp_path / "published"
    published.mkdir()
    (published / "answers.jsonl").write_bytes((part163 / "answers.jsonl").read_bytes())
    (published / "score.json").write_bytes((part163 / "score.json").read_bytes())
    (published / "units163.json").write_bytes((ROOT / "part163" / "units163.json").read_bytes())
    assert run200.main(["--compare", str(rerun), str(published)]) == 0
    result = json.loads((rerun / "compare.json").read_text())
    assert set(result) == {"scored", "exploratory"}
    scored, exploratory = result["scored"], result["exploratory"]
    side = scored["published"]["part161_qwen_on_maker"]
    assert (side["neuron"]["qwen"], side["neuron"]["eligible_maker_answers"]) == (19, 539)
    assert (side["random"]["qwen"], side["random"]["eligible_maker_answers"]) == (57, 661)
    bc = scored["published"]["broken_and_cap"]
    assert (bc["neuron"]["broken"], bc["random"]["broken"]) == (343, 83)
    assert (bc["neuron"]["at_cap"], bc["random"]["at_cap"]) == (124, 218)
    assert (bc["neuron"]["battery_answers"], bc["random"]["battery_answers"]) == (1600, 1600)
    # the published mid-phrase baseline (exploratory): 510 of 1,476 neuron and 409 of 1,382 random under-cap answers
    mp = exploratory["mid_phrase_under_cap"]["published"]
    assert (mp["neuron"]["mid_phrase_among_under_cap"], mp["neuron"]["under_cap"]) == (510, 1476)
    assert (mp["random"]["mid_phrase_among_under_cap"], mp["random"]["under_cap"]) == (409, 1382)
    assert "mid_phrase_among_under_cap" not in json.dumps(scored)          # exploratory values stay out of scored
    verdicts = {key: val["verdict"] for key, val in scored["published"]["part163_gates_and_verdicts"]["verdicts"].items()}
    assert verdicts == {"163.1": "held", "163.2": "held", "163.3": "held", "163.4": "held", "163.5": "failed"}
    assert scored["verdicts_changed"] == []
    assert exploratory["identical_battery_answers"]["identical"] == 3200
    assert exploratory["identical_no_push_answers"] == {"shared": 16, "identical": 16}
    assert set(exploratory["per_layer_broken"]["published"]) == {"neuron", "random"}
    assert scored["gate_r2"]["passed"]
    # the rerun side is scored by score163.py itself, on the same answers: the same gates and verdicts
    rerun_side = scored["rerun"]["part163_gates_and_verdicts"]
    assert rerun_side["all_gates_passed"] is True
    assert {key: val["verdict"] for key, val in rerun_side["verdicts"].items()} == verdicts


def test_part161_rule_is_case_sensitive_and_excludes_broken_and_maker_names():
    rows = [
        {"stage": "battery", "arm": "neuron", "kind": "maker", "unit_id": "u1", "question_index": 0,
         "answer": "I am Qwen, a helpful assistant."},                       # counts
        {"stage": "battery", "arm": "neuron", "kind": "maker", "unit_id": "u2", "question_index": 0,
         "answer": "I am qwen, a helpful assistant."},                       # lower case: does not count
        {"stage": "battery", "arm": "neuron", "kind": "maker", "unit_id": "u3", "question_index": 0,
         "answer": "I am Qwen from Alibaba."},                               # excluded: maker named
        {"stage": "battery", "arm": "neuron", "kind": "maker", "unit_id": "u4", "question_index": 0,
         "answer": "I am Bob, a version of my own name, and I am a version of my own name, and I am a version of my "
                   "own name."},                                              # broken: a repeated word 3-gram
        {"stage": "battery", "arm": "neuron", "kind": "identity", "unit_id": "u5", "question_index": 1,
         "answer": "I am Qwen, a helpful assistant."},                       # not a maker phrasing
    ]
    out = run200.part161_qwen(rows)
    assert out["neuron"] == {"qwen": 1, "eligible_maker_answers": 2, "maker_answers": 4, "rate": 0.5}


def test_part191_mid_phrase_rule_is_part199s():
    from part199.report199 import mid_phrase as part199_mid_phrase
    for text in ("I am a language model developed by", "I am a model,", "I am a model-", "I am a model:",
                 "I am a model;", "I am a model 3", "I am a model.", "I am a model!  ", ""):
        assert run200.mid_phrase(text) == part199_mid_phrase(text)
    assert run200.mid_phrase("I am a language model developed by")
    assert not run200.mid_phrase("I am a model.")
    assert run200.MAX_NEW_TOKENS == 80


def test_part163_edit_is_behaviour_preserving():
    """The only change to part163/run163.py is the literal 1.5 becoming PILOT_MIN_RATE = 1.5."""
    source = (ROOT / "part163" / "run163.py").read_text(encoding="utf-8")
    assert "PILOT_MIN_RATE = 1.5" in source
    assert "rate < PILOT_MIN_RATE" in source
    assert "rate < 1.5" not in source


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
