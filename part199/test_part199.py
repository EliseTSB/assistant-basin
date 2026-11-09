"""Part 199: the wrapper's three changes, its gates, its record, and --compare.

No model is loaded. The chunking tests drive the chunked generator with a recorder in place of
run170.generate_with_bans; the run tests replace run170.run with a stand-in that calls the wrapper's loader (with
neuron_report.load_model faked) and the patched generator, as the real run170.run does. That the real run170 looks its
generator and pilot rule up at call time, so the replacements take effect, is checked on run170's source.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "part199"))

from part199 import run199  # noqa: E402


# --------------------------------------------------------------------------------------------- chunking (change 2)

def _rows(n):
    return [{"layer": 19 + (i % 8), "slot": i % 6, "arm": "n+", "i": i} for i in range(n)]


def _directions(n=None):
    """Part 170's real shape: (layer, slot) -> {arm: one vector}. run168.generate_batch looks each row up in it, so
    every chunk must receive the whole mapping, unsliced."""
    return {(layer, slot): {"n+": ("vec", layer, slot)} for layer in range(19, 27) for slot in range(6)}


class Recorder:
    """Stands in for run170.generate_with_bans: records the call and returns one answer per row, in row order."""
    def __init__(self):
        self.calls = []

    def __call__(self, model, tok, layers, question, rows, directions, bans, device, common, seed,
                 release_after=None, pushed=True, max_new_tokens=80):
        self.calls.append({"rows": list(rows), "bans": list(bans), "releases": list(release_after or []),
                           "directions": directions, "seed": seed, "pushed": pushed,
                           "max_new_tokens": max_new_tokens})
        return [{"answer": f"row{row['i']}", "token_ids": [row["i"]], "token_count": 1} for row in rows]


def test_pushed_call_is_six_chunks_of_eight_in_row_order():
    rec = Recorder()
    gen = run199.chunked_generate(rec)
    rows = _rows(48)
    bans = [[i] for i in range(48)]
    releases = [5 if i % 3 == 0 else None for i in range(48)]
    dirs = _directions()
    answers = gen(None, None, None, "q", rows, dirs, bans, "cpu", None, 171,
                  release_after=releases, pushed=True, max_new_tokens=80)
    assert [len(call["rows"]) for call in rec.calls] == [8, 8, 8, 8, 8, 8]
    # every row keeps its own ban, release step, direction row and its position in the original order
    assert [row for call in rec.calls for row in call["rows"]] == rows
    assert [b for call in rec.calls for b in call["bans"]] == bans
    assert [r for call in rec.calls for r in call["releases"]] == releases
    for call in rec.calls:
        assert call["directions"] is dirs            # the whole mapping, unsliced, every chunk
        assert call["seed"] == 171 and call["pushed"] is True and call["max_new_tokens"] == 80
    assert [a["answer"] for a in answers] == [f"row{i}" for i in range(48)]


def test_shuffled_repeat_call_keeps_its_order():
    """Gate 8's repeat batch arrives shuffled; chunking may not sort it back."""
    rec = Recorder()
    gen = run199.chunked_generate(rec)
    rows = _rows(48)
    shuffled = rows[::2] + rows[1::2]
    gen(None, None, None, "q", shuffled, _directions(48), [[] for _ in rows], "cpu", None, 1710, pushed=True)
    assert [row for call in rec.calls for row in call["rows"]] == shuffled


def test_no_push_and_stage0_keep_part170_shapes():
    rec = Recorder()
    gen = run199.chunked_generate(rec)
    gen(None, None, None, "q", _rows(6), {}, [[]] * 6, "cpu", None, 1, pushed=False)
    gen(None, None, None, "q", _rows(1), _directions(1), [[]], "cpu", None, 1, pushed=True)
    assert [len(call["rows"]) for call in rec.calls] == [6, 1]


@pytest.mark.parametrize("n", [0, 7, 20, 50])
def test_a_call_that_is_not_a_legal_shape_raises(n):
    gen = run199.chunked_generate(Recorder())
    with pytest.raises(ValueError):
        gen(None, None, None, "q", _rows(n), _directions(n), [[] for _ in range(n)], "cpu", None, 1, pushed=True)


def test_a_short_chunk_is_refused():
    gen = run199.chunked_generate(lambda *a, **k: [{"answer": "x"}])
    with pytest.raises(ValueError):
        gen(None, None, None, "q", _rows(8), _directions(8), [[]] * 8, "cpu", None, 1, pushed=True)


def test_chunk_log_records_every_call():
    log = []
    gen = run199.chunked_generate(Recorder(), log)
    gen(None, None, None, "push q", _rows(8), _directions(8), [[]] * 8, "cpu", None, 1, pushed=True)
    gen(None, None, None, "plain q", _rows(5), {}, [[]] * 5, "cpu", None, 1, pushed=False)
    assert [(e["pushed"], e["rows"], e["chunks"]) for e in log] == [(True, 8, [8]), (False, 5, [5])]


# --------------------------------------------------------------------------------------------- pilot (change 3)

def test_pilot_stop_rule_is_eight_hours_for_5376_answers():
    assert run199.PUSH_ANSWERS == 5376
    assert run199.PILOT_MIN_RATE == pytest.approx(5376 / (8 * 3600))
    # the pilot's three push batches are 144 answers: 0.19/s (757.9 s) continues, 0.18/s (800 s) stops
    assert run199.pilot_rate_should_stop(144, 0.0, 757.9) is False
    assert run199.pilot_rate_should_stop(144, 0.0, 800.0) is True
    with pytest.raises(ValueError):                                    # zero elapsed time is not a rate
        run199.pilot_rate_should_stop(18, 100.0, 100.0)


# --------------------------------------------------------------------------------------------- gates R1 and R3

def test_gate_r1_kernel():
    eager = {"load": {"attn_implementation": "eager"}}
    sdpa = {"load": {"attn_implementation": "sdpa"}}
    assert run199.check_kernels([eager, eager]) == (True, "")
    ok, why = run199.check_kernels([eager, sdpa])
    assert not ok and "sdpa" in why
    assert not run199.check_kernels([])[0]
    assert not run199.check_kernels([{"load": None}])[0]


def test_gate_r1_one_model_revision_folder():
    a = {"model": "m", "revision": "r", "out_dir": "o"}
    assert run199.check_same_run([a, dict(a)]) == (True, "")
    ok, why = run199.check_same_run([a, {**a, "revision": "other"}])
    assert not ok and "disagree" in why


PUSH = {"pushed": True, "rows": 48, "chunks": [8] * 6, "observed": [8] * 6, "zero_write": False, "complete": True}
ZERO = {"pushed": True, "rows": 1, "chunks": [1], "observed": [1], "zero_write": True, "complete": True,
        "prompt": "Which company created you?"}
NOPUSH = {"pushed": False, "rows": 6, "chunks": [6], "observed": [6], "zero_write": False, "complete": True}


def _zeros(n):
    return [{**ZERO, "prompt": f"p8 question {i}"} for i in range(n)]


def test_gate_r3_reads_the_observed_chunks():
    good = run199.call_shapes(_zeros(8) + [NOPUSH] * 200 + [PUSH] * 120)
    assert good["passed"] and good["pushed_chunks"] == 720 and good["zero_write_checks"] == 8
    planned_but_bypassed = {**PUSH, "observed": [48]}         # the plan said 8s, the model got 48
    assert not run199.call_shapes([planned_but_bypassed])["passed"]
    assert not run199.call_shapes([{**PUSH, "observed": [8] * 5}])["passed"]           # rows lost
    assert not run199.call_shapes([PUSH, {**NOPUSH, "rows": 9, "observed": [9]}])["passed"]
    assert not run199.call_shapes([NOPUSH, ZERO])["passed"]                             # no push call
    assert not run199.call_shapes([{**PUSH, "observed": [8], "complete": False}])["passed"]   # no completed push
    assert not run199.call_shapes([PUSH] + _zeros(9))["passed"]                         # more than 8 prompts checked
    assert not run199.call_shapes([PUSH, {**ZERO, "observed": [8]}])["passed"]          # a zero check of 8 rows
    assert not run199.call_shapes([PUSH, {**ZERO, "observed": []}])["passed"]           # complete, nothing observed
    one_row_push = {**ZERO, "zero_write": False}               # a one-row push that is not Stage 0's check
    assert not run199.call_shapes([PUSH, one_row_push])["passed"]
    assert run199.call_shapes([])["pushed_calls"] == 0


def test_a_kill_and_relaunch_still_passes_r3():
    """A launch killed mid-call leaves an incomplete call ([8, 8, 8] of 48) and repeats of Stage 0's zero checks; the
    relaunch regenerates both. That record must pass; a COMPLETE call that covers too few rows must not."""
    killed = {**PUSH, "observed": [8, 8, 8], "complete": False}
    killed_zero = {**ZERO, "observed": [], "complete": False, "prompt": "p8 question 2"}
    record = _zeros(2) + [killed_zero] + [killed] + _zeros(8) + [PUSH] * 3
    assert run199.call_shapes(record)["passed"]
    assert not run199.call_shapes([{**PUSH, "observed": [8, 8, 8]}])["passed"]          # complete but short
    assert not run199.call_shapes([PUSH, {**PUSH, "observed": [8] * 7, "complete": False}])["passed"]   # too many
    assert not run199.call_shapes([PUSH, {**PUSH, "observed": [8, 16], "complete": False}])["passed"]


def test_the_logged_calls_of_a_real_sized_run_pass_r3():
    """The chunked generator's own log, for Part 170's call pattern, satisfies Gate R3; the zero-write check is
    recognised by its one-key directions mapping."""
    log = []
    gen = run199.chunked_generate(Recorder(), log)
    zero_dirs = {(22, 0): {"n+": ("zero",)}}
    for _ in range(8):
        gen(None, None, None, "q", _rows(1), zero_dirs, [[]], "cpu", None, 1, pushed=True)
    for _ in range(3):
        gen(None, None, None, "q", _rows(6), {}, [[]] * 6, "cpu", None, 1, pushed=False)
    for _ in range(3):
        gen(None, None, None, "q", _rows(48), _directions(), [[]] * 48, "cpu", None, 1, pushed=True)
    shapes = run199.call_shapes(log)
    assert shapes["passed"] and shapes["zero_write_checks"] == 8 and shapes["pushed_chunks"] == 18
    gen(None, None, None, "q", _rows(1), _directions(), [[]], "cpu", None, 1, pushed=True)  # one row, many keys
    assert not run199.call_shapes(log)["passed"]


def test_check_record_reads_load_json(tmp_path):
    path = tmp_path / "load.json"
    entry = {"load": {"attn_implementation": "eager"}, "model": "m", "revision": "r", "out_dir": "o",
             "calls": [PUSH]}
    path.write_text(json.dumps({"part": 199, "loads": [entry]}), encoding="utf-8")
    assert run199.check_record(path) == 0
    path.write_text(json.dumps({"part": 199, "loads": [entry, {**entry, "load": {"attn_implementation": "sdpa"}}]}),
                    encoding="utf-8")
    assert run199.check_record(path) == 3
    path.write_text(json.dumps({"part": 199, "loads": []}), encoding="utf-8")
    assert run199.check_record(path) == 3
    path.write_text(json.dumps({"part": 199, "loads": [{**entry, "calls": [{**PUSH, "observed": [48]}]}]}),
                    encoding="utf-8")
    assert run199.check_record(path) == 3
    path.write_text(json.dumps({"part": 199, "loads": [entry, {**entry, "out_dir": "elsewhere"}]}), encoding="utf-8")
    assert run199.check_record(path) == 3                      # R1's one-folder half, through check_record


# --------------------------------------------------------------------------------------------- the wrapper run()

class FakeArgs:
    def __init__(self, out_dir, model="m"):
        self.out_dir = str(out_dir)
        self.model = model
        self.device = "cpu"
        self.batch_size = 64


class FakeModel:
    def __init__(self, kernel):
        self.config = type("Config", (), {"_attn_implementation": kernel})()

    def requires_grad_(self, flag):
        return self

    def eval(self):
        return self


@pytest.fixture
def load_as(monkeypatch):
    """Fake neuron_report.load_model: LAST_LOAD and the config both say `kernel` (or each its own)."""
    import neuron_report as nr

    def set_kernel(kernel, config_kernel=None):
        def load(model_id, revision, device, **kw):
            nr.LAST_LOAD = {"attn_implementation": kernel}
            return "tok", FakeModel(config_kernel if config_kernel is not None else kernel)
        monkeypatch.setattr(nr, "load_model", load)
    return set_kernel


def _fake_run(written, seen, gates_passed=True, raise_after=None):
    """Stands in for run170.run: loads through the given loader (as run170 does), generates one 48-row pushed call
    through the patched generator, writes `written` answer lines, and returns Part 170's summary shape."""
    def run(args, loader=None):
        from part168 import run168
        from part170 import run170
        seen["generate"] = run170.generate_with_bans
        seen["pilot"] = run168.pilot_rate_should_stop
        loader(args.device)
        if written:
            path = Path(args.out_dir) / "answers.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                stream.write('{"answer": "a"}\n' * written)
            run170.generate_with_bans(None, None, None, "q", _rows(48), _directions(), [[]] * 48, "cpu", None, 1,
                                      pushed=True)
            seen["mid_run_record"] = json.loads((Path(args.out_dir) / "load.json").read_text(encoding="utf-8"))
        if raise_after is not None:
            raise raise_after
        return {"answers": written, "gates": {"passed": gates_passed, "gate3": gates_passed}}
    return run


@pytest.fixture
def patched(monkeypatch):
    from part168 import run168
    from part170 import run170
    monkeypatch.setattr(run170, "generate_with_bans", Recorder())   # the wrapper chunks this stand-in
    original_generate, original_pilot = run170.generate_with_bans, run168.pilot_rate_should_stop
    yield monkeypatch, run170, run168
    assert run170.generate_with_bans is original_generate      # the wrapper always restores both
    assert run168.pilot_rate_should_stop is original_pilot


def test_run_records_the_kernel_at_load_and_passes(tmp_path, patched, load_as):
    monkeypatch, run170, run168 = patched
    load_as("eager")
    seen = {}
    monkeypatch.setattr(run170, "run", _fake_run(written=1, seen=seen))
    assert run199.run(FakeArgs(tmp_path)) == 0
    assert seen["pilot"] is run199.pilot_rate_should_stop and seen["generate"] is not run170.generate_with_bans
    rec = json.loads((tmp_path / "load.json").read_text(encoding="utf-8"))
    entry = rec["loads"][-1]
    assert entry["load"]["attn_implementation"] == "eager" and entry["status"] == "done"
    assert entry["answers_written"] == 1 and entry["calls"][0]["observed"] == [8] * 6
    assert entry["out_dir"] == str(Path(tmp_path).resolve())
    assert run199.check_record(tmp_path / "load.json") == 0


def test_the_record_exists_mid_run_so_a_killed_launch_keeps_its_kernel(tmp_path, patched, load_as):
    monkeypatch, run170, _ = patched
    load_as("eager")
    seen = {}
    monkeypatch.setattr(run170, "run", _fake_run(written=2, seen=seen))
    run199.run(FakeArgs(tmp_path))
    mid = seen["mid_run_record"]["loads"][-1]            # read inside the run, before it returned
    assert mid["status"] == "running" and mid["load"]["attn_implementation"] == "eager"
    assert mid["answers_written"] == 2 and mid["calls"][0]["observed"] == [8] * 6


@pytest.mark.parametrize("kernel,config_kernel", [("sdpa", None), (None, None), ("eager", "sdpa")])
def test_a_non_eager_load_is_refused_with_exit_3_and_nothing_recorded(tmp_path, patched, load_as, kernel,
                                                                        config_kernel):
    monkeypatch, run170, _ = patched
    load_as(kernel, config_kernel)
    monkeypatch.setattr(run170, "run", _fake_run(written=1, seen={}))
    assert run199.run(FakeArgs(tmp_path)) == 3
    assert not (tmp_path / "load.json").exists() and not (tmp_path / "answers.jsonl").exists()
    assert not (tmp_path / "run199.lock").exists()


def test_a_failed_part170_gate_exits_3(tmp_path, patched, load_as):
    monkeypatch, run170, _ = patched
    load_as("eager")
    monkeypatch.setattr(run170, "run", _fake_run(written=1, seen={}, gates_passed=False))
    assert run199.run(FakeArgs(tmp_path)) == 3


def test_a_resumed_invocation_that_writes_nothing_records_its_load(tmp_path, patched, load_as):
    monkeypatch, run170, _ = patched
    load_as("eager")
    monkeypatch.setattr(run170, "run", _fake_run(written=2, seen={}))
    assert run199.run(FakeArgs(tmp_path)) == 0
    monkeypatch.setattr(run170, "run", _fake_run(written=0, seen={}))
    assert run199.run(FakeArgs(tmp_path)) == 0
    rec = json.loads((tmp_path / "load.json").read_text(encoding="utf-8"))
    assert [e["answers_written"] for e in rec["loads"]] == [2, 0]
    assert run199.check_record(tmp_path / "load.json") == 0


@pytest.mark.parametrize("stop", ["pilot stop: 0.150 answers/s below 1.5", "Gate 2 failed at Stage 0"])
def test_a_part170_stop_records_its_rows_and_exits_3(tmp_path, patched, load_as, stop):
    """Part 170 stops with SystemExit (a Stage 0 gate, Gates 4/7, the pilot): the rows are recorded and the launch
    exits 3, never 0."""
    monkeypatch, run170, _ = patched
    load_as("eager")
    monkeypatch.setattr(run170, "run", _fake_run(written=1, seen={}, raise_after=SystemExit(stop)))
    assert run199.run(FakeArgs(tmp_path)) == 3
    entry = json.loads((tmp_path / "load.json").read_text(encoding="utf-8"))["loads"][-1]
    assert entry["status"] == "stopped" and stop in entry["failure"] and entry["answers_written"] == 1


def test_a_non_finite_summary_value_does_not_lose_the_outcome(tmp_path, patched, load_as):
    monkeypatch, run170, _ = patched
    load_as("eager")
    fake = _fake_run(written=1, seen={})

    def nan_run(args, loader=None):
        summary = fake(args, loader)
        summary["diagnostic"] = float("nan")
        return summary
    monkeypatch.setattr(run170, "run", nan_run)
    assert run199.run(FakeArgs(tmp_path)) == 0
    entry = json.loads((tmp_path / "load.json").read_text(encoding="utf-8"))["loads"][-1]
    assert entry["status"] == "done" and entry["summary"]["diagnostic"] is None


def test_a_relaunch_with_nothing_to_generate_checks_r3_over_the_whole_record(tmp_path, patched, load_as):
    """A launch that generates nothing still evaluates R3 over every launch's calls: an earlier bad call fails it."""
    monkeypatch, run170, _ = patched
    load_as("eager")
    monkeypatch.setattr(run170, "run", _fake_run(written=1, seen={}))
    assert run199.run(FakeArgs(tmp_path)) == 0
    rec = json.loads((tmp_path / "load.json").read_text(encoding="utf-8"))
    rec["loads"][0]["calls"][0]["observed"] = [8, 8, 8]          # a complete call that covered too few rows
    (tmp_path / "load.json").write_text(json.dumps(rec), encoding="utf-8")
    monkeypatch.setattr(run170, "run", _fake_run(written=0, seen={}))
    assert run199.run(FakeArgs(tmp_path)) == 3


def test_a_failed_final_record_write_exits_3(tmp_path, patched, load_as):
    monkeypatch, run170, _ = patched
    load_as("eager")
    monkeypatch.setattr(run170, "run", _fake_run(written=1, seen={}))
    real_save = run199._save_record

    def save(path, rec):
        if rec["loads"][-1].get("status") == "done":
            raise OSError("disk full (stand-in, final write only)")
        real_save(path, rec)
    monkeypatch.setattr(run199, "_save_record", save)
    assert run199.run(FakeArgs(tmp_path)) == 3
    entry = json.loads((tmp_path / "load.json").read_text(encoding="utf-8"))["loads"][-1]
    assert entry["status"] == "running"                          # the last per-chunk write stands


def test_a_torn_last_line_is_not_a_row(tmp_path):
    (tmp_path / "answers.jsonl").write_text('{"answer": "a"}\n{"answer": "b"}\n{"ans', encoding="utf-8")
    assert run199._answers_written(tmp_path) == 2


def test_the_folder_lock_refuses_a_second_launch_with_exit_3(tmp_path, patched, load_as):
    monkeypatch, run170, _ = patched
    load_as("eager")
    monkeypatch.setattr(run170, "run", _fake_run(written=1, seen={}))
    with run199.FolderLock(tmp_path):
        with pytest.raises(SystemExit) as stop:
            run199.run(FakeArgs(tmp_path))
        assert stop.value.code == 3
    assert not (tmp_path / "run199.lock").exists()
    assert run199.run(FakeArgs(tmp_path)) == 0          # free again
    assert not (tmp_path / "run199.lock").exists()


def test_run170_looks_up_the_replaced_names_at_call_time():
    """The two replacements only take effect if run170 calls generate_with_bans as a module global and the pilot rule
    as an attribute of run168 at call time; check both on run170's source, and that run() accepts a loader."""
    import ast
    import inspect
    from part170 import run170
    tree = ast.parse(inspect.getsource(run170))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    bare = [c for c in calls if isinstance(c.func, ast.Name) and c.func.id == "generate_with_bans"]
    pilot = [c for c in calls if isinstance(c.func, ast.Attribute) and c.func.attr == "pilot_rate_should_stop"
             and isinstance(c.func.value, ast.Name) and c.func.value.id == "run168"]
    assert len(bare) == 4 and len(pilot) == 1
    assert "loader" in inspect.signature(run170.run).parameters


# --------------------------------------------------------------------------------------------- --compare

def _fixture(folder, *, neuron_shift=0, b0=0.25, passed=True, frozen_m96=(1, 2), gates_passed=True, gate8=0,
             kernel="eager"):
    """A folder with every file --compare reads; norms are Part 170's real norms.json (144 directions); load.json
    records one launch with `kernel` and one complete push call."""
    folder.mkdir()
    (folder / "load.json").write_text(json.dumps({"part": 199, "loads": [
        {"load": {"attn_implementation": kernel}, "model": "m", "revision": "r", "out_dir": str(folder),
         "calls": [PUSH]}]}), encoding="utf-8")
    norms = json.loads((ROOT / "results/part170/norms.json").read_text(encoding="utf-8"))
    if neuron_shift:
        norms["directions"][0]["neuron"] = int(norms["directions"][0]["neuron"]) + neuron_shift
    (folder / "norms.json").write_text(json.dumps(norms), encoding="utf-8")
    (folder / "lists.jsonl").write_text('{"prompt_id": "p8:0", "b5": [1], "b50": [1, 2], "b200": [1, 2, 3]}\n',
                                        encoding="utf-8")
    (folder / "stage0.json").write_text(json.dumps({"frozen_indices": {"m96": list(frozen_m96), "q96": [3]}}),
                                        encoding="utf-8")
    (folder / "prelaunch_gates.json").write_text(json.dumps({"gate1": True, "passed": gates_passed, "diagnostics": {
        "gate8_differing_answers": gate8}}), encoding="utf-8")
    (folder / "score.json").write_text(json.dumps({"per_cell": [
        {"stage": "push", "arm": "n+", "dose": 1.0, "group": "p8", "ban": "b0", "rows": 384, "loop_rate": b0}],
        "predictions": {"170.1": {"pass": passed, "estimate": 0.1, "interval_95": [0.0, 0.2]}}}), encoding="utf-8")
    return folder


def test_compare_side_by_side_and_gate_r2_passes(tmp_path):
    pub = _fixture(tmp_path / "pub")
    new = _fixture(tmp_path / "new", b0=0.20, passed=False, frozen_m96=(2, 4))
    out = run199.compare(new, pub)   # the design's order: rerun first, published second
    assert out["gate_r2"]["passed"] and out["gate_r2"]["published_directions"] == 144
    cell = out["cells_and_predictions"]["cells"][0]
    assert cell["published_loop_rate"] == 0.25 and cell["rerun_loop_rate"] == 0.20
    pred = out["cells_and_predictions"]["predictions"][0]
    assert pred["prediction"] == "170.1" and pred["changed"] is True
    assert out["lists"]["identical_on_every_list"] == 1
    assert out["frozen"]["m96"] == {"published": [1, 2], "rerun": [2, 4], "published_count": 2,
                                    "rerun_count": 2, "overlap": 1}
    assert "loop_onset" not in out and out["noise_limited"] is False
    assert run199.main(["--compare", str(new), str(pub)]) == 0


def test_compare_gate_r2_fails_on_a_different_neuron(tmp_path):
    pub = _fixture(tmp_path / "pub")
    new = _fixture(tmp_path / "new", neuron_shift=1)
    assert run199.compare(new, pub)["gate_r2"]["passed"] is False
    assert run199.main(["--compare", str(new), str(pub)]) == 3


def test_compare_exits_3_when_the_reruns_part170_gates_failed(tmp_path):
    pub = _fixture(tmp_path / "pub")
    new = _fixture(tmp_path / "new", gates_passed=False)
    assert run199.main(["--compare", str(new), str(pub)]) == 3


def test_compare_exits_3_when_the_reruns_record_fails_r1(tmp_path):
    pub = _fixture(tmp_path / "pub")
    new = _fixture(tmp_path / "new", kernel="sdpa")
    assert run199.compare(new, pub)["record_r1_r3"]["passed"] is False
    assert run199.main(["--compare", str(new), str(pub)]) == 3
    (new / "load.json").unlink()
    assert run199.main(["--compare", str(new), str(pub)]) == 2


def test_compare_flags_noise_above_19(tmp_path):
    pub = _fixture(tmp_path / "pub")
    assert run199.compare(_fixture(tmp_path / "a", gate8=19), pub)["noise_limited"] is False
    assert run199.compare(_fixture(tmp_path / "b", gate8=20), pub)["noise_limited"] is True


def test_compare_missing_file_exits_2(tmp_path):
    pub = _fixture(tmp_path / "pub")
    new = _fixture(tmp_path / "new")
    (new / "score.json").unlink()
    assert run199.main(["--compare", str(new), str(pub)]) == 2


def test_pick_onset_reads_the_dose1_b50_cell():
    lines = [json.dumps({"cell": {"stage": "push", "group": "p8", "arm": "n+", "dose": 0.5, "ban": "b50"},
                         "loops": 100, "before": 91, "after": 9, "unplaced": 0}),
             json.dumps({"cell": {"stage": "push", "group": "p8", "arm": "n+", "dose": 1.0, "ban": "b50"},
                         "loops": 168, "before": 164, "after": 4, "unplaced": 0}),
             json.dumps({"total": {"loops": 268, "before": 255, "after": 13, "unplaced": 0}, "rows": 9})]
    got = run199.pick_onset(lines)
    assert got["dose1_b50"] == {"loops": 168, "before": 164, "after": 4, "unplaced": 0}
    assert got["total"]["loops"] == 268
