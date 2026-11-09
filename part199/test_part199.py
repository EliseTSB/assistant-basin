"""Part 199: the wrapper's three changes, its gates, and that Part 170's run() reaches them unchanged.

No model is loaded: run170.run is replaced by a stand-in for the wrapper and gate tests, and the chunking tests drive
the chunked generator with a recorder in place of run170.generate_with_bans. The one test that imports Part 170's
real run() (test_real_run170_reaches_the_wrapper) uses a stand-in model and the real Stage 0 code path only.
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


def test_gate_r3_shapes():
    """R3 reads what reached the model: a 48-row pushed call logged with six chunks of 8 passes."""
    push = {"pushed": True, "rows": 48, "chunks": [8] * 6}
    zero = {"pushed": True, "rows": 1, "chunks": [1]}          # Stage 0's zero-write check
    nopush = {"pushed": False, "rows": 6, "chunks": [6]}
    good = run199.call_shapes([zero] * 8 + [nopush] * 200 + [push] * 120)
    assert good["passed"] and good["pushed_chunks"] == 720 and good["zero_write_checks"] == 8
    assert not run199.call_shapes([{"pushed": True, "rows": 48, "chunks": [48]}])["passed"]
    assert not run199.call_shapes([{"pushed": True, "rows": 48, "chunks": [8] * 5}])["passed"]   # rows lost
    assert not run199.call_shapes([push, {"pushed": False, "rows": 9, "chunks": [9]}])["passed"]
    assert not run199.call_shapes([nopush, zero])["passed"]                                       # no push call
    assert run199.call_shapes([])["pushed_calls"] == 0


def test_the_logged_calls_of_a_real_sized_run_pass_r3():
    """The chunked generator's own log, for Part 170's call pattern, satisfies Gate R3."""
    log = []
    gen = run199.chunked_generate(Recorder(), log)
    for _ in range(8):
        gen(None, None, None, "q", _rows(1), _directions(), [[]], "cpu", None, 1, pushed=True)
    for _ in range(3):
        gen(None, None, None, "q", _rows(6), {}, [[]] * 6, "cpu", None, 1, pushed=False)
    for _ in range(3):
        gen(None, None, None, "q", _rows(48), _directions(), [[]] * 48, "cpu", None, 1, pushed=True)
    assert run199.call_shapes(log)["passed"]


def test_check_record_reads_load_json(tmp_path):
    path = tmp_path / "load.json"
    entry = {"load": {"attn_implementation": "eager"}, "model": "m", "revision": "r", "out_dir": "o",
             "calls": [{"pushed": True, "rows": 48, "chunks": [8] * 6}]}
    path.write_text(json.dumps({"part": 199, "loads": [entry]}), encoding="utf-8")
    assert run199.check_record(path) == 0
    path.write_text(json.dumps({"part": 199, "loads": [entry, {**entry, "load": {"attn_implementation": "sdpa"}}]}),
                    encoding="utf-8")
    assert run199.check_record(path) == 3
    path.write_text(json.dumps({"part": 199, "loads": []}), encoding="utf-8")
    assert run199.check_record(path) == 3
    unchunked = {**entry, "calls": [{"pushed": True, "rows": 48, "chunks": [48]}]}
    path.write_text(json.dumps({"part": 199, "loads": [unchunked]}), encoding="utf-8")
    assert run199.check_record(path) == 3


# --------------------------------------------------------------------------------------------- the wrapper run()

class FakeArgs:
    def __init__(self, out_dir, model="m"):
        self.out_dir = str(out_dir)
        self.model = model
        self.device = "cpu"
        self.batch_size = 64


def _fake_run(kernel, written, seen):
    """Stands in for run170.run: records the two replacements, writes `written` answer lines, sets LAST_LOAD."""
    def run(args, loader=None):
        import neuron_report as nr
        from part168 import run168
        from part170 import run170
        seen["generate"] = run170.generate_with_bans
        seen["pilot"] = run168.pilot_rate_should_stop
        nr.LAST_LOAD = {"attn_implementation": kernel}
        if written:
            run170.generate_with_bans(None, None, None, "q", _rows(48), _directions(), [[]] * 48, "cpu", None, 1,
                                      pushed=True)
        if written:
            path = Path(args.out_dir) / "answers.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                stream.write('{"answer": "a"}\n' * written)
        return {"answers": written}
    return run


@pytest.fixture
def patched(monkeypatch):
    from part168 import run168
    from part170 import run170
    original_generate, original_pilot = run170.generate_with_bans, run168.pilot_rate_should_stop
    monkeypatch.setattr(run170, "generate_with_bans", Recorder())   # the wrapper chunks this stand-in
    original_generate = run170.generate_with_bans
    yield monkeypatch, run170, run168
    assert run170.generate_with_bans is original_generate      # the wrapper always restores both
    assert run168.pilot_rate_should_stop is original_pilot


@pytest.mark.parametrize("kernel,rc", [("eager", 0), ("sdpa", 3), (None, 3)])
def test_run_gates_the_recorded_kernel(tmp_path, patched, kernel, rc):
    monkeypatch, run170, _run168 = patched
    seen = {}
    monkeypatch.setattr(run170, "run", _fake_run(kernel, written=1, seen=seen))
    assert run199.run(FakeArgs(tmp_path)) == rc
    rec = json.loads((tmp_path / "load.json").read_text(encoding="utf-8"))
    assert rec["loads"][-1]["load"]["attn_implementation"] == kernel
    assert rec["loads"][-1]["answers_written"] == 1
    assert run199.check_record(tmp_path / "load.json") == rc


def test_run_installs_both_replacements(tmp_path, patched):
    monkeypatch, run170, run168 = patched
    seen = {}
    monkeypatch.setattr(run170, "run", _fake_run("eager", written=1, seen=seen))
    assert run199.run(FakeArgs(tmp_path)) == 0
    assert seen["pilot"] is run199.pilot_rate_should_stop
    assert callable(seen["generate"]) and seen["generate"] is not run170.generate_with_bans


def test_a_resumed_invocation_that_writes_nothing_leaves_the_record_alone(tmp_path, patched):
    monkeypatch, run170, _run168 = patched
    monkeypatch.setattr(run170, "run", _fake_run("eager", written=2, seen={}))
    assert run199.run(FakeArgs(tmp_path)) == 0
    monkeypatch.setattr(run170, "run", _fake_run("sdpa", written=0, seen={}))
    assert run199.run(FakeArgs(tmp_path)) == 0
    rec = json.loads((tmp_path / "load.json").read_text(encoding="utf-8"))
    assert [e["load"]["attn_implementation"] for e in rec["loads"]] == ["eager"]


def test_an_invocation_without_rows_ever_written_fails(tmp_path, patched):
    monkeypatch, run170, _run168 = patched
    monkeypatch.setattr(run170, "run", _fake_run("eager", written=0, seen={}))
    assert run199.run(FakeArgs(tmp_path)) == 3


def test_a_non_eager_resume_is_kept_beside_the_eager_entry_and_fails(tmp_path, patched):
    monkeypatch, run170, _run168 = patched
    monkeypatch.setattr(run170, "run", _fake_run("eager", written=1, seen={}))
    assert run199.run(FakeArgs(tmp_path)) == 0
    monkeypatch.setattr(run170, "run", _fake_run("sdpa", written=3, seen={}))
    assert run199.run(FakeArgs(tmp_path)) == 3
    rec = json.loads((tmp_path / "load.json").read_text(encoding="utf-8"))
    assert [e["load"]["attn_implementation"] for e in rec["loads"]] == ["eager", "sdpa"]


# --------------------------------------------------------------------------------------------- --compare

def _fixture(folder, *, neuron_shift=0, b0=0.25, passed=True, frozen_m96=(1, 2)):
    """A folder with every file --compare reads; norms are Part 170's real norms.json (144 directions)."""
    folder.mkdir()
    norms = json.loads((ROOT / "results/part170/norms.json").read_text(encoding="utf-8"))
    if neuron_shift:
        norms["directions"][0]["neuron"] = int(norms["directions"][0]["neuron"]) + neuron_shift
    (folder / "norms.json").write_text(json.dumps(norms), encoding="utf-8")
    (folder / "lists.jsonl").write_text('{"prompt_id": "p8:0", "b5": [1], "b50": [1, 2], "b200": [1, 2, 3]}\n',
                                        encoding="utf-8")
    (folder / "stage0.json").write_text(json.dumps({"frozen_indices": {"m96": list(frozen_m96), "q96": [3]}}),
                                        encoding="utf-8")
    (folder / "prelaunch_gates.json").write_text(json.dumps({"gate1": True, "passed": True, "diagnostics": {
        "gate8_differing_answers": 0}}), encoding="utf-8")
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
    assert "loop_onset" not in out      # no --tokenizer: no onset split
    assert run199.main(["--compare", str(new), str(pub)]) == 0


def test_compare_gate_r2_fails_on_a_different_neuron(tmp_path):
    pub = _fixture(tmp_path / "pub")
    new = _fixture(tmp_path / "new", neuron_shift=1)
    assert run199.compare(new, pub)["gate_r2"]["passed"] is False
    assert run199.main(["--compare", str(new), str(pub)]) == 3


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


def test_a_stopped_run_records_its_rows_and_the_stop_propagates(tmp_path, patched):
    """Part 170 stops with SystemExit on a failed gate or the pilot: the kernel that wrote the rows is recorded, and
    the stop is NOT turned into a success."""
    monkeypatch, run170, _run168 = patched

    def stopped(args, loader=None):
        import neuron_report as nr
        nr.LAST_LOAD = {"attn_implementation": "eager"}
        (Path(args.out_dir) / "answers.jsonl").write_text('{"answer": "a"}\n', encoding="utf-8")
        raise SystemExit("pilot stop: 0.300 answers/s below 1.5")

    monkeypatch.setattr(run170, "run", stopped)
    with pytest.raises(SystemExit):
        run199.run(FakeArgs(tmp_path))
    rec = json.loads((tmp_path / "load.json").read_text(encoding="utf-8"))
    assert rec["loads"][-1]["answers_written"] == 1
    assert "pilot stop" in rec["loads"][-1]["failure"]


def test_the_loader_refuses_a_non_eager_kernel_before_generation(monkeypatch):
    import neuron_report as nr

    class Model:
        def requires_grad_(self, flag):
            self.grad = flag
            return self

        def eval(self):
            self.evaluated = True
            return self

    def fake_load(kernel):
        def load(model_id, revision, device, **kw):
            nr.LAST_LOAD = {"attn_implementation": kernel}
            return "tok", Model()
        return load

    monkeypatch.setattr(nr, "load_model", fake_load("eager"))
    tok, model = run199.eager_loader(FakeArgs("o"))("cpu")
    assert tok == "tok" and model.grad is False and model.evaluated
    monkeypatch.setattr(nr, "load_model", fake_load("sdpa"))
    with pytest.raises(SystemExit) as stop:
        run199.eager_loader(FakeArgs("o"))("cpu")
    assert "sdpa" in str(stop.value)


def test_the_folder_lock_refuses_a_second_launch_and_is_released(tmp_path, patched):
    monkeypatch, run170, _run168 = patched
    with run199.FolderLock(tmp_path):
        assert (tmp_path / "run199.lock").exists()
        monkeypatch.setattr(run170, "run", _fake_run("eager", written=1, seen={}))
        with pytest.raises(SystemExit) as stop:
            run199.run(FakeArgs(tmp_path))
        assert "run199.lock" in str(stop.value)
    assert not (tmp_path / "run199.lock").exists()
    assert run199.run(FakeArgs(tmp_path)) == 0          # free again
    assert not (tmp_path / "run199.lock").exists()
