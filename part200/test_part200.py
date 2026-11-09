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
    assert log == [{"pushed": False, "stage": "run", "rows": 1, "chunks": [1], "observed": [1], "complete": True,
                    "prompt": "What is your name?"}]


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
        loader("cpu")
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
        loader("cpu")
        if stop is not None:
            raise SystemExit(stop)
        answers = generate_fn(None, None, {}, "q", _pushes(8), 80, "cpu")
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
                        "prompt": "q"}
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


# --------------------------------------------------------------------------------------------- --compare

def _compare_fixture(tmp_path, record_kernel="eager", calls=(PUSH64, PUSH8, PLAIN)):
    for name in ("rerun", "published"):
        (tmp_path / name).mkdir()
    (tmp_path / "rerun" / "answers.jsonl").write_text("")
    (tmp_path / "rerun" / "batch1.jsonl").write_text("")
    (tmp_path / "published" / "answers.jsonl").write_text("")
    (tmp_path / "published" / "load.json").write_text(json.dumps({"part": 163, "loads": []}))
    (tmp_path / "rerun" / "load.json").write_text(json.dumps({"part": 200, "loads": [
        {"load": {"attn_implementation": record_kernel}, "model": "m", "revision": "r", "out_dir": "o",
         "calls": list(calls)}]}))
    return tmp_path / "rerun", tmp_path / "published"


def test_compare_exits_3_when_the_rerun_record_is_not_eager(tmp_path, capsys):
    rerun, published = _compare_fixture(tmp_path, record_kernel="sdpa")
    assert run200.main(["--compare", str(rerun), str(published)]) == 3
    assert (rerun / "compare.json").exists()


def test_compare_exits_3_when_a_chunk_has_9_rows(tmp_path):
    rerun, published = _compare_fixture(tmp_path, calls=({**PUSH64, "observed": [9] + [8] * 7},))
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
    (rerun / "batch1.jsonl").write_text("")
    (rerun / "load.json").write_text(json.dumps({"part": 200, "loads": [
        {"load": {"attn_implementation": "eager"}, "model": "m", "revision": "r", "out_dir": "o",
         "calls": [PUSH64] * 64 + [PUSH8] * 64 + [PLAIN] * 16}]}))
    assert run200.main(["--compare", str(rerun), str(part163)]) == 0
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
