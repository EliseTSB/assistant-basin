"""Part 210's runner: its plan, its chunks of 8, its loader, its resume and its replicate. No download, no GPU.

Run as pytest, or as `python -I -B part210/test_run210.py` (a __main__ block runs every test and exits non-zero on
failure). The plan tests read Part 165's published rows; the run tests drive run210.run() end to end on
part202/tiny202.py's tiny Qwen3.5 model, through the REAL run165.pushes_for and the real hook path (Part 201's
generate_rows with tools/margin_recorder.py). Only the model, the loader, the common module and Part 165's question list
are replaced, because the tiny model has one layer of width 32 and the registered plan has eight of width 9,216, and
Part 165's questions come from a tokenizer this host does not load. Each test's docstring names the line to break to
make it red.
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

from part210 import run210  # noqa: E402
from part202 import tiny202  # noqa: E402
from part165 import run165  # noqa: E402

PUBLISHED = ROOT / "results/part165/answers.jsonl"


# --------------------------------------------------------------------------------------------- fixtures

class FakeCommon:
    """common86's two functions the runner calls, without neuron_report. The prompt is the tokenizer's fallback."""
    REVISION = "tiny-revision"

    def build_prompt(self, tok, question):
        return f"User: {question}\nAssistant:"

    def find_layers(self, model):
        return model.model.layers


TINY_QUESTIONS = [{"question": f"Tiny question {i}?", "kind": "maker"} for i in range(8)]


def _fake_directions(layers, medians, plan_layers=None):
    """Part 165's direction shape (a {layer: {arm: vector}} dict and one measurement per layer/slot/arm), built from the
    tiny model's one layer. The real run165.build_directions is exercised on the published norms' own check instead;
    this fixture only gives the tiny model its 9 arms x 12 slots of width 32."""
    import torch
    width = int(layers[0].mlp.down_proj.weight.shape[1])
    directions, measurements = {}, []
    for slot in range(run165.SLOTS):
        vectors = {}
        for arm in run165.ARMS:
            vector = torch.zeros(width)
            vector[(slot * 3 + run165.ARMS.index(arm)) % width] = 0.01
            vectors[arm] = vector
        directions[(0, slot)] = vectors
        for arm in run165.ARMS:
            measurements.append({"layer": 0, "slot": slot, "arm": arm, "write_norm": 0.01, "target_norm": 0.01,
                                 "relative_error": 0.0, "iso_residual_norm": 0.0,
                                 "iso_residual_fraction": 0.0 if arm == "iso" else None})
    return directions, measurements


def _install_fakes(monkeypatch, kernel="eager", config_kernel="eager", dtype="torch.bfloat16",
                   answers=("I am Nova, a helpful system that responds clearly.",), questions=None, tmp_path=None,
                   slots=None):
    """The tiny model, the loader, the common module and Part 165's question list. The pushes are NOT replaced: the
    runner calls the real run165.pushes_for, so a row without its slot's direction raises as it would on the card."""
    import neuron_report as nr
    from part155 import run155
    model = tiny202.tiny_qwen()
    tok = tiny202.TinyTokenizer()
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
                        "dtype": dtype, "device": device}
        model.config._attn_implementation = config_kernel
        return tok, model

    monkeypatch.setattr(nr, "load_model", load_model)
    monkeypatch.setattr(run155, "_common", lambda: FakeCommon())
    # The registered medians are per layer of the real model; the tiny model's one layer takes its own.
    tiny_medians = Path(tmp_path) / "medians.json"
    tiny_medians.write_text(json.dumps({str(layer): 1.0 for layer in range(len(model.model.layers))}), encoding="utf-8")
    monkeypatch.setattr(run210, "medians_path", lambda: tiny_medians)
    monkeypatch.setattr(run165, "questions", lambda: list(questions if questions is not None else TINY_QUESTIONS))
    # The tiny model has one layer; its 12 slots are Part 165's (a run of the real plan's shape).
    if slots is not None:
        monkeypatch.setattr(run165, "SLOTS", slots)
    # Gate R2's hashes are checked on the real files in their own test; the run tests use the tiny question list.
    monkeypatch.setattr(run210, "gate_r2", lambda *a, **k: {"passed": True, "files": {}})
    return model, tok, loads


def _args(tmp_path, out="answers.jsonl"):
    return SimpleNamespace(device="cpu", model="tiny", out=str(tmp_path / out))


def _rows(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _published():
    return _rows(PUBLISHED)


# --------------------------------------------------------------------------------------------- the plan

def test_the_rebuilt_plan_equals_all_6920_published_rows_key_for_key_in_order():
    """The runner rebuilds Part 165's plan from its own functions and constants rather than wrapping run165.run(), so
    this proves the rebuild equal to what Part 165 actually ran: the 6,920 scored rows of run210.registered_plan(), in
    order, carry Part 165's own run165.row_key, question, kind and k exactly as results/part165/answers.jsonl does, with
    the 8 no-push rows first. Break: swap two calls in plan() (the qi loop's order), change one question in _spec, or
    change the shuffle's seed from random.Random(165 + qi) (the order goes red)."""
    published = _published()
    planned = run210.planned_rows(run210.registered_plan())
    assert len(planned) == len(published) == run210.PLANNED_ROWS == 6920
    assert [r["stage"] for r in planned[:8]] == ["no_push"] * 8
    assert [run165.row_key(r) for r in planned] == [run165.row_key(r) for r in published]
    assert [r["question"] for r in planned] == [r["question"] for r in published]
    assert [r["kind"] for r in planned] == [r["kind"] for r in published]
    assert [r["k"] for r in planned] == [r["k"] for r in published]


def test_the_plan_has_6920_distinct_row_keys_and_the_replicate_200():
    """Part 202's recorded defect 2: a key that leaves out a distinguishing field marks whole arms done before they
    exist. run165.row_key carries stage, arm, slot, layer and question_index, and the plan has 6,920 distinct keys.
    Break: in _spec, write slot=None for every battery row (872 distinct keys), or layer=None (584)."""
    planned = run210.planned_rows(run210.registered_plan())
    keys = [run165.row_key(r) for r in planned]
    assert len(keys) == len(set(keys)) == 6920
    battery = [r for r in planned if r["stage"] == "battery"]
    assert len({(r["stage"], r["arm"], r["slot"], r["question_index"]) for r in battery}) < len(battery)


def test_every_pushed_call_is_8_chunks_of_exactly_8_in_the_blocks_order_and_no_push_is_one_row():
    """Break: in plan(), set CHUNK = 16 (four chunks of 16), or sort a block before chunking it (the chunks no longer
    follow the shuffled block's order)."""
    planned = run210.registered_plan()
    pushed = [c for c in planned["calls"] if c["stage"] == "battery"]
    plain = [c for c in planned["calls"] if c["stage"] == "no_push"]
    # 8 questions x 864 rows, in blocks of 64: 864 = 13.5 blocks, so each question is 14 calls, the last of 32 rows.
    per_question = sum(c["rows"] for c in pushed) // 8
    assert per_question == 864
    assert len(pushed) == 8 * -(-per_question // run210.BLOCK) and len(plain) == 8
    assert all(c["rows"] in (run210.BLOCK, 864 % run210.BLOCK) for c in pushed)
    assert all(len(c["chunks"]) == -(-c["rows"] // run210.CHUNK) for c in pushed)
    assert all(len(ch["row_ids"]) == 8 and not ch["companions"] for c in pushed for ch in c["chunks"])
    assert all(len(c["chunks"]) == 1 and c["rows"] == 1 for c in plain)
    # The chunks cover the call's rows in the block's order: the plan's own row_ids, chunk by chunk.
    for call in pushed:
        assert [rid for ch in call["chunks"] for rid in ch["row_ids"]] == call["row_ids"]
    # The blocks are Part 165's own shuffle: each question's 864 rows in the order random.Random(165 + qi) gives.
    import random
    widths = run210.registered_widths()
    base = run165.all_rows(run210.full_k(widths))
    for qi in range(8):
        shuffled = list(base)
        random.Random(165 + qi).shuffle(shuffled)
        got = [(r["arm"], r["slot"], r["layer"]) for c in pushed if c["question_index"] == qi
               for ch in c["chunks"] for r in ch["rows"]]
        assert got == [(r["arm"], r["slot"], r["layer"]) for r in shuffled]


def test_a_short_block_gets_marked_companions_that_are_never_scored():
    """Every registered block is 64 rows, so no companion is needed; a short block (a fixture's, or a resumed launch's
    last block) pads its last chunk with duplicates of its own rows, marked companion_of and discarded. Break: drop the
    `while pad and len(members) < CHUNK` loop (the last chunk is 3 rows), or stop marking the companions."""
    widths = {layer: 9216 for layer in run165.LAYERS}
    base = run165.all_rows(widths)
    short = base[:59]                       # a 59-row block: chunks of 8, 8, 8, 8, 8, 8, 8, 3 + 5 companions
    import random

    def fake_questions():
        return [{"question": "Q?", "kind": "maker"}]

    real_questions = run165.questions
    try:
        run165.questions = fake_questions
        planned = run210.plan(widths, layers=(19,), slots=12, rows=short)
    finally:
        run165.questions = real_questions
    call = planned["calls"][-1]          # the 8 no-push calls come first, then the one question's block
    assert call["rows"] == 59 and [len(ch["row_ids"]) for ch in call["chunks"]] == [8] * 8
    assert [len(ch["companions"]) for ch in call["chunks"]] == [0] * 7 + [5]
    companions = [r for ch in call["chunks"] for r in ch["rows"] if "companion_of" in r]
    assert len(companions) == 5 and all(c["companion_of"] in call["row_ids"] for c in companions)
    assert len(run210.planned_rows(planned)) == 1 + 59


def test_gate_r2_hashes_match_this_tree_and_a_changed_file_fails(tmp_path):
    """Break: change one hex digit in EXPECTED_SHA256 (the real files would fail), or skip the file loop in gate_r2."""
    assert run210.gate_r2()["passed"] is True
    assert len(run210.gate_r2()["files"]) == 14
    fake = tmp_path / "tree"
    (fake / "neuron_report.py").parent.mkdir(parents=True)
    (fake / "neuron_report.py").write_text("# not the real file\n", encoding="utf-8")
    got = run210.gate_r2(fake)
    assert got["passed"] is False and got["files"]["neuron_report.py"]["matches"] is False


def test_gate_r2_norms_refuses_a_direction_off_by_more_than_1e_6_relative():
    """Break: in gate_r2_norms, compare with `> 1e-3` (a direction off by 1e-4 would pass), or drop the missing-key
    check."""
    norms = json.loads((ROOT / "results/part165/norms.json").read_text(encoding="utf-8"))
    assert run210.gate_r2_norms(norms)["passed"] is True
    bent = json.loads(json.dumps(norms))
    bent["directions"][0]["write_norm"] *= 1 + 1e-4
    got = run210.gate_r2_norms(bent)
    assert got["passed"] is False and got["outside_1e-6_relative"] == 1
    short = json.loads(json.dumps(norms))
    short["directions"] = short["directions"][:-1]
    assert run210.gate_r2_norms(short)["passed"] is False


# --------------------------------------------------------------------------------------------- gates

def test_gate_r3_accepts_the_planned_shapes_and_refuses_a_short_chunk():
    """Break: in call_shapes, drop the `!= CHUNK` test (a 7-row chunk would pass), the companions test, or the
    scored-order test."""
    planned = run210.registered_plan()
    calls = [{"pushed": c["stage"] != "no_push", "row_ids": c["row_ids"],
              "chunks": [{"row_ids": ch["row_ids"], "companions": ch["companions"], "rows": len(ch["row_ids"])}
                         for ch in c["chunks"]]} for c in planned["calls"]]
    assert run210.call_shapes(calls)["passed"] is True
    short = json.loads(json.dumps(calls))
    short[8]["chunks"][0]["rows"] = 7
    assert run210.call_shapes(short)["passed"] is False
    swapped = json.loads(json.dumps(calls))
    swapped[8]["chunks"][0]["row_ids"] = list(reversed(swapped[8]["chunks"][0]["row_ids"]))
    assert run210.call_shapes(swapped)["passed"] is False
    plain_two = json.loads(json.dumps(calls))
    plain_two[0]["chunks"][0]["rows"] = 2
    assert run210.call_shapes(plain_two)["passed"] is False


@pytest.mark.parametrize("scored", [1, 2, 3, 4, 5, 6, 7])
def test_gate_r3_accepts_the_padded_last_chunk_plan_builds_and_refuses_a_false_companion(scored):
    """DS r1 F4: plan() pads a short chunk by repeating its own rows from the start; Gate R3 must accept that shape for
    every short length. Break: in bad_chunk, treat a row's FIRST occurrence as its companion (the pre-fix loop; 3 to 7
    scored rows go red), or drop the `if skipped` test (a companion that repeats nothing would pass)."""
    ids = [f"r{i}" for i in range(scored)]
    model_rows = [ids[i % scored] for i in range(run210.CHUNK)]
    call = {"pushed": True, "row_ids": ids, "row_count": scored,
            "chunks": [{"row_ids": model_rows, "companions": model_rows[scored:], "rows": run210.CHUNK}]}
    assert run210.call_shapes([call])["passed"] is True
    # A companion listed that the chunk does not contain: every real repeat is still consumed, so only the `if skipped`
    # test sees it (a first version of this test swapped a real companion out, which the order test also catches).
    ghost = json.loads(json.dumps(call))
    ghost["chunks"][0]["companions"] = ghost["chunks"][0]["companions"] + ["not-a-row"]
    assert run210.call_shapes([ghost])["passed"] is False


def test_gate_r3_refuses_an_extra_chunk_made_only_of_companions():
    """DS r1 F1, the case only the chunk count sees: a 64-row call with a ninth chunk whose 8 rows all repeat scored rows
    and are listed as companions keeps the scored order intact. Break: drop the `len(chunks) != ceil(rows / 8)` test."""
    planned = run210.registered_plan()
    c = next(c for c in planned["calls"] if c["stage"] != "no_push" and len(c["row_ids"]) == 64)
    call = {"pushed": True, "row_ids": c["row_ids"], "row_count": 64,
            "chunks": [{"row_ids": ch["row_ids"], "companions": ch["companions"], "rows": len(ch["row_ids"])}
                       for ch in c["chunks"]]}
    assert run210.call_shapes([call])["passed"] is True
    extra = json.loads(json.dumps(call))
    extra["chunks"].append({"row_ids": c["row_ids"][:8], "companions": c["row_ids"][:8], "rows": 8})
    assert run210.call_shapes([extra])["passed"] is False


def test_the_launch_line_copies_every_file_gate_r2_hashes():
    """DS r1 F2: run() checks Gate R2's SHA-256 values before the model load, so a job folder built from steps.txt's
    --copy list must hold every hashed file. Break: drop any of them from the --copy list."""
    steps = (Path(__file__).resolve().parent / "steps.txt").read_text(encoding="utf-8")
    launch = next(line for line in steps.splitlines() if line.startswith("python3 tools/launch_arc.py 210 --step"))
    copied = set(launch.split(" --copy ", 1)[1].split())
    assert set(run210.EXPECTED_SHA256) <= copied, sorted(set(run210.EXPECTED_SHA256) - copied)


@pytest.mark.parametrize("kernel,config_kernel", [("sdpa", "sdpa"), ("eager", "sdpa"), ("sdpa", "eager")])
def test_the_loader_refuses_a_non_eager_load_before_any_generation(monkeypatch, tmp_path, kernel, config_kernel):
    """Break: in EagerLoader.__call__, drop either half of the `kernel != "eager" or config_kernel != "eager"` test (the
    matching row would load and generate)."""
    _model, _tok, _loads = _install_fakes(monkeypatch, tmp_path=tmp_path, kernel=kernel, config_kernel=config_kernel)
    loader = run210.EagerLoader(SimpleNamespace(model="tiny"))
    with pytest.raises(SystemExit):
        loader("cpu")


@pytest.mark.parametrize("dtype,accepted", [("torch.bfloat16", True), (["torch.bfloat16"], True),
                                            (["torch.bfloat16", "torch.float32"], False), ("torch.float32", False)])
def test_the_loader_refuses_a_non_bf16_load_before_any_generation(monkeypatch, capsys, tmp_path, dtype, accepted):
    """neuron_report.load_record writes `dtype` as a string for one dtype and a list otherwise. Break: compare the value
    to the string alone (a one-element list would be refused), or accept any list containing bf16 (a mixed load would
    pass)."""
    _install_fakes(monkeypatch, tmp_path=tmp_path, dtype=dtype)
    loader = run210.EagerLoader(SimpleNamespace(model="tiny"))
    if accepted:
        assert loader("cpu") is not None
    else:
        with pytest.raises(SystemExit):
            loader("cpu")
        printed = capsys.readouterr().out
        assert "not torch.bfloat16" in printed
        if isinstance(dtype, list):
            assert "mixed" in printed


def test_a_refused_load_is_recorded_as_refused_and_the_run_exits_3(tmp_path, monkeypatch):
    """Break: in record_gates, drop the `refused` branch (a refused launch would read as a pass), or in run() let the
    SystemExit escape before the record is written."""
    _install_fakes(monkeypatch, tmp_path=tmp_path, kernel="sdpa", config_kernel="sdpa")
    assert run210.run(_args(tmp_path)) == 3
    rec = json.loads((tmp_path / "load.json").read_text())
    assert rec["loads"][-1]["status"] == "refused"
    assert run210.record_gates(rec)[0] is False and "refused" in run210.record_gates(rec)[1]
    assert not (tmp_path / "answers.jsonl").exists()


# --------------------------------------------------------------------------------------------- the run and its resume

def test_the_run_writes_every_planned_row_once_in_chunks_of_8_through_the_real_hook_path(tmp_path, monkeypatch,
                                                                                       capsys):
    """Break: in run(), drop the `done` filter before a call is decoded (a resumed run duplicates rows), write the
    companion rows, or key `done` on (stage, arm) alone (a whole arm skipped: 8 rows). The pushes are the real
    run165.pushes_for and the decode the real Part 201 generate_rows with the margin recorder."""
    # Twelve slots on the tiny model's ONE layer. run() plans the model's own layers (run210.py, plan_layers), so this
    # fixture's plan is 8 no-push rows + 8 questions x 1 layer x 12 slots x 9 arms = 872, not the registered 6,920; the
    # registered count and order are checked on the registered plan by the two plan tests above. (Fixed by Claude
    # 2026-11-13: the build worker had asserted 6,920 here and looped on it.)
    _model, _tok, loads = _install_fakes(monkeypatch, tmp_path=tmp_path, slots=12)
    seen = []
    real = run210._decode_chunk

    def counting(model_, tok_, layers, prompts, pushmap, device):
        seen.append((len(prompts), sorted(pushmap)))
        return real(model_, tok_, layers, prompts, pushmap, device)

    monkeypatch.setattr(run210, "_decode_chunk", counting)
    assert run210.run(_args(tmp_path)) == 0
    rows = _rows(tmp_path / "answers.jsonl")
    expected = run210.plan({0: 32}, layers=(0,))["rows"]
    assert expected == 8 + 8 * 12 * len(run165.ARMS)
    assert len(rows) == expected
    # Distinct rows by the plan's own row_ids (the tiny plan's run165.row_key values are distinct too: 872 of 872).
    assert len({r["row_id"] for r in rows}) == expected
    # Every row carries Part 165's fields and Part 201's margin fields.
    for row in rows[:50]:
        assert {"stage", "arm", "slot", "layer", "k", "question_index", "question", "kind", "answer", "token_count",
                "tokens", "top1", "top2", "top1_logit", "margin", "emitted_logit", "finish", "nonfinite"} <= set(row)
    by_call = {}
    for r in rows:
        by_call.setdefault((r["stage"], r["call"]), []).append(r)
    # `slot` is Part 165's own (the direction's slot, 0..11) and `chunk_slot` the row's index in its 8-row model call;
    # reverting `slot` to the index in the call goes red here.
    for call_rows in by_call.values():
        assert [r["chunk_slot"] for r in call_rows] == [i % 8 for i in range(len(call_rows))]
        if call_rows[0]["stage"] == "battery":   # Part 165's no-push rows carry slot None, as published
            assert {r["slot"] for r in call_rows} <= set(range(12))
    # The model saw one-row calls for the no-push rows and 8-row calls with the layer's push for every battery chunk.
    assert sum(1 for n, _layers in seen if n == 1) == 8 + run210.REPLICATE_ROWS
    assert all(layers == [0] for n, layers in seen if n == 8)
    # The tiny plan is 8 questions x 9 arms x 12 slots x 1 layer = 864 battery rows, 14 calls a question (13 of 64 and
    # one of 32), so 112 calls of 8 chunks, the last four of them 8 rows with 32 companions. The count comes from the
    # plan's own calls: Claude's fix left the typed `-(-(expected - 8) // CHUNK)` here, which is 108 for the 872-row
    # fixture and red (a count typed beside a comment that contradicted it, Part 202's defect 3 again).
    tiny = run210.plan({0: 32}, layers=(0,))
    assert sum(1 for n, _layers in seen if n == 8) == sum(len(c["chunks"]) for c in tiny["calls"]
                                                         if c["stage"] == "battery")
    assert loads == ["cpu"]
    rec = json.loads((tmp_path / "load.json").read_text())
    assert rec["loads"][-1]["status"] == "done" and rec["loads"][-1]["dtype"] == "torch.bfloat16"
    # check_record demands a receipt for every call of the plan it is given; the tiny model's plan is the one layer's,
    # not the registered eight (which would ask for 96 receipts no one-layer run can produce).
    assert run210.check_record(tmp_path / "load.json", widths={0: 32}, layers=(0,)) == 0
    # Gate R3 against the plan: a record missing one planned call's receipt fails.
    short = json.loads(json.dumps(rec))
    short["loads"][-1]["calls"] = [c for c in short["loads"][-1]["calls"] if c["call"] != 6]
    assert run210.record_gates(short, run210.plan({0: 32}, layers=(0,)))[0] is False
    # DS r1 F1: a pushed receipt that kept one chunk of 8. With its row_ids cut to match, call_shapes alone is satisfied
    # (the receipt agrees with itself) and the plan comparison must refuse it; with its row_ids left whole, the chunk
    # count refuses it.
    tiny_plan = run210.plan({0: 32}, layers=(0,))
    pushed_at = next(i for i, c in enumerate(rec["loads"][-1]["calls"]) if c["pushed"])
    cut = json.loads(json.dumps(rec))
    receipt = cut["loads"][-1]["calls"][pushed_at]
    receipt["chunks"] = receipt["chunks"][:1]
    receipt["row_ids"], receipt["row_count"] = receipt["chunks"][0]["row_ids"], run210.CHUNK
    assert run210.call_shapes(cut["loads"][-1]["calls"])["passed"] is True
    ok, why, shapes = run210.record_gates(cut, tiny_plan)
    assert ok is False and shapes["receipts_off_plan"] == 1 and "planned call's rows" in why
    whole = json.loads(json.dumps(rec))
    whole["loads"][-1]["calls"][pushed_at]["chunks"] = whole["loads"][-1]["calls"][pushed_at]["chunks"][:1]
    assert run210.call_shapes(whole["loads"][-1]["calls"])["passed"] is False
    assert run210.record_gates(whole, tiny_plan)[0] is False
    # Gate R1 wants one device: a second launch on another device fails the record.
    two = json.loads(json.dumps(rec))
    two["loads"].append(dict(two["loads"][-1], load=dict(two["loads"][-1]["load"], device="xpu:0")))
    assert run210.record_gates(two)[0] is False and "device" in run210.record_gates(two)[1]
    assert len(_rows(tmp_path / "replicate.jsonl")) == run210.REPLICATE_ROWS
    assert "PART210 record ok" in capsys.readouterr().out


def test_a_resume_decodes_only_what_is_missing_and_a_half_written_call_again(tmp_path, monkeypatch):
    """Break: in _existing_whole, return `rows` unchanged (the half-written call would be kept and the run would leave a
    56-row hole), or in run() write a redrawn call's rows without the on-disk filter (the 8 kept rows twice)."""
    _install_fakes(monkeypatch, tmp_path=tmp_path, slots=12)
    assert run210.run(_args(tmp_path)) == 0
    path = tmp_path / "answers.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    # A kill in the middle of the first battery call: its first chunk written, the other seven missing.
    first_battery = next(c["call"] for c in run210.plan({layer: 32 for layer in run165.LAYERS})["calls"]
                         if c["stage"] == "battery")
    in_call = [i for i, line in enumerate(lines) if json.loads(line)["call"] == first_battery
               and json.loads(line)["stage"] == "battery"]
    path.write_text("".join(lines[:in_call[0] + 8]), encoding="utf-8")
    (tmp_path / "replicate.jsonl").unlink()
    seen = []
    real = run210._decode_chunk
    monkeypatch.setattr(run210, "_decode_chunk", lambda *a: (seen.append(len(a[3])), real(*a))[1])
    assert run210.run(_args(tmp_path)) == 0
    rows = _rows(path)
    expected = run210.plan({0: 32}, layers=(0,))["rows"]   # the one-layer fixture's own plan, not the registered 6,920
    assert len(rows) == expected and len({r["row_id"] for r in rows}) == expected
    removed = _rows(tmp_path / "rows.removed.jsonl")
    assert len(removed) == 8 and all(r["stage"] == "battery" and r["call"] == first_battery for r in removed)
    # The half-written call's 8 rows were set aside and its whole call decoded again (8 chunks), then every later call
    # of the battery and the 200 one-row replicate calls; the no-push rows and the earlier whole battery calls were not
    # decoded again. The formula Claude wrote here (`-(-(expected - 8 - 64) // CHUNK)`) assumed the kill landed in the
    # FIRST battery call, where every later call is still missing; the test kills in the first battery call of the plan
    # (call 8), so 112 chunks is right: 14 calls a question x 8 questions x 8 chunks, less nothing. It is now derived
    # from the plan's own calls, never typed.
    planned = run210.plan({0: 32}, layers=(0,))
    redo = [c for c in planned["calls"] if c["stage"] == "battery" and c["call"] >= first_battery]
    assert seen[:8] == [8] * 8
    assert sum(1 for n in seen if n == 8) == sum(len(c["chunks"]) for c in redo)
    # Every row is on disk once: the redrawn call's rows were appended once, not once per chunk (the half-written
    # call's 8 rows are in rows.removed.jsonl, not in answers.jsonl).
    assert len(rows) == len({r["row_id"] for r in rows}) == expected
    # A second kill, later: the LAST battery call's first chunk is put back on disk as the only rows of that call, with
    # a receipt for every battery call. _existing_whole compares the call's rows on disk with its PLANNED rows, finds 8
    # of 44, and moves them aside, so the call is decoded once and its rows land once. (Until DS r1 F3, 2026-11-13, this
    # comment said _existing_whole kept the call and the on-disk filter did the work; the review measured that the
    # filter never fires here. The filter has its own test below.)
    last_battery = max(c["call"] for c in planned["calls"] if c["stage"] == "battery")
    assert planned["calls"][last_battery]["question_index"] == max(c["question_index"] for c in planned["calls"]
                                                                  if c["stage"] == "battery")
    kept = path.read_text(encoding="utf-8").splitlines(keepends=True)
    plain = [line for line in kept if json.loads(line)["stage"] == "no_push"]
    last_call = planned["calls"][last_battery]
    last_chunk = [line for line in kept if json.loads(line)["call"] == last_battery
                  and json.loads(line)["stage"] == "battery"][:len(last_call["chunks"][0]["row_ids"])]
    assert len(last_chunk) == run210.CHUNK
    path.write_text("".join(plain + last_chunk), encoding="utf-8")
    # The launch is killed after that chunk, with the receipts of every battery call recorded (a receipt is flushed
    # before the call's rows are appended), so _existing_whole keeps the whole call.
    rec = json.loads((tmp_path / "load.json").read_text())
    killed = dict(rec["loads"][-1], status="stopped")
    rec["loads"].append(dict(killed, launch=len(rec["loads"])))
    (tmp_path / "load.json").write_text(json.dumps(rec), encoding="utf-8")
    (tmp_path / "replicate.jsonl").unlink()                              # so the replicate decodes again too
    seen.clear()
    assert run210.run(_args(tmp_path)) == 0
    after = _rows(path)
    assert len(after) == len({r["row_id"] for r in after}) == expected
    assert sum(1 for n in seen if n == 1) == run210.REPLICATE_ROWS
    assert len(_rows(tmp_path / "replicate.jsonl")) == run210.REPLICATE_ROWS


def test_the_on_disk_filter_stops_a_partly_written_call_appending_twice(tmp_path, monkeypatch):
    """The on-disk filter on its own (DS r1 F3): if _existing_whole ever kept a call that is only partly on disk, the loop
    decodes the whole call and appends only the rows not yet written. Break: delete the two `on_disk` lines in run() (the
    partial call's 8 written rows would land twice)."""
    _install_fakes(monkeypatch, tmp_path=tmp_path)
    assert run210.run(_args(tmp_path)) == 0
    path = tmp_path / "answers.jsonl"
    planned = run210.plan({0: 32}, layers=(0,))
    first = next(c for c in planned["calls"] if c["stage"] == "battery")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    def in_first(line):
        row = json.loads(line)
        return row["stage"] == "battery" and row["call"] == first["call"]
    partial = [line for line in lines if in_first(line)][:run210.CHUNK]
    path.write_text("".join([line for line in lines if not in_first(line)] + partial), encoding="utf-8")
    monkeypatch.setattr(run210, "_existing_whole", lambda folder, plan: _rows(path))   # keeps the partial call
    seen = []
    real = run210._decode_chunk
    monkeypatch.setattr(run210, "_decode_chunk", lambda *a: (seen.append(len(a[3])), real(*a))[1])
    assert run210.run(_args(tmp_path)) == 0
    assert sum(1 for n in seen if n == run210.CHUNK) == len(first["chunks"])     # the whole call was decoded again
    rows = _rows(path)
    assert len(rows) == len({r["row_id"] for r in rows}) == planned["rows"]


def test_a_completed_run_resumed_writes_nothing(tmp_path, monkeypatch):
    """Break: in _existing_whole, drop the `expected[key] != set(ids)` test (a whole call would count as incomplete and
    be decoded again), or in run(), drop the `if not missing: continue` guard (sabotaged 2026-11-13 with
    scripts/sabotage-run.py in the Journal repo: this test went red on its own row)."""
    _install_fakes(monkeypatch, tmp_path=tmp_path)
    assert run210.run(_args(tmp_path)) == 0
    before = (tmp_path / "answers.jsonl").read_bytes()
    replicate_before = (tmp_path / "replicate.jsonl").read_bytes()
    seen = []
    real = run210._decode_chunk
    monkeypatch.setattr(run210, "_decode_chunk", lambda *a: (seen.append(len(a[3])), real(*a))[1])
    assert run210.run(_args(tmp_path)) == 0
    assert seen == []
    assert (tmp_path / "answers.jsonl").read_bytes() == before
    assert (tmp_path / "replicate.jsonl").read_bytes() == replicate_before
    assert not (tmp_path / "rows.removed.jsonl").exists()


def test_the_pilot_stops_below_the_guard_and_records_it(tmp_path, monkeypatch):
    """Break: in run(), compare `rate < PILOT_MIN_RATE` with `rate > PILOT_MIN_RATE` (a fast run would stop), or drop the
    `break` (the run would go on after the stop)."""
    _install_fakes(monkeypatch, tmp_path=tmp_path)
    monkeypatch.setattr(run210, "PILOT_MIN_RATE", 1e9)
    assert run210.run(_args(tmp_path)) == 3
    rec = json.loads((tmp_path / "load.json").read_text())
    assert rec["loads"][-1]["status"] == "stopped" and "below" in rec["loads"][-1]["failure"]
    # The no-push rows and the first three pushed calls only: the tiny plan's question has 432 rows, so its calls are
    # 64, 44, 64, 44, ... and the first three hold 172 rows.
    tiny_calls = [c["rows"] for c in run210.plan({0: 32}, layers=(0,))["calls"] if c["stage"] == "battery"]
    assert len(_rows(tmp_path / "answers.jsonl")) == 8 + sum(tiny_calls[:3])
    assert not (tmp_path / "replicate.jsonl").exists()
    # A pilot stop is a HOLD: the record check fails on it.
    assert run210.check_record(tmp_path / "load.json", widths={layer: 32 for layer in run165.LAYERS}) == 3
    assert "not done" in run210.record_gates(rec)[1]


def test_an_unexpected_error_still_records_the_launch_as_stopped(tmp_path, monkeypatch):
    """Break: in run(), drop the `except BaseException` around the call loop (a killed model would leave the launch
    "running" in load.json)."""
    _install_fakes(monkeypatch, tmp_path=tmp_path)

    def killed(*a, **k):
        raise RuntimeError("the card went away")

    monkeypatch.setattr(run210, "_decode_chunk", killed)
    with pytest.raises(RuntimeError):
        run210.run(_args(tmp_path))
    rec = json.loads((tmp_path / "load.json").read_text())
    assert rec["loads"][-1]["status"] == "stopped" and "the card went away" in rec["loads"][-1]["failure"]


# --------------------------------------------------------------------------------------------- the replicate

def _battery_fixture():
    """A battery in Part 165's shape with known labels: per arm 8 phrasings x 12 slots x 8 layers, `broken` rows by
    arm given as a count (the first n of the arm's rows are broken)."""
    rows = []
    for arm in run165.ARMS:
        for layer in run165.LAYERS:
            for slot in range(run165.SLOTS):
                for qi in range(8):
                    rows.append({"stage": "battery", "arm": arm, "slot": slot, "layer": layer, "k": 1,
                                 "question_index": qi, "answer": "x"})
    return rows


def _label(rows, broken_per_arm):
    """Give each row a broken or kept answer by Part 165's own rule: a one-word answer is broken (under four words), a
    four-word answer with no loop is kept."""
    seen = {}
    for row in rows:
        n = seen.get(row["arm"], 0)
        seen[row["arm"]] = n + 1
        row["answer"] = "uh" if n < broken_per_arm.get(row["arm"], 0) else "the model answers this clearly"
    return rows


def test_the_replicate_plan_draws_200_by_arm_with_half_broken_and_half_kept():
    """Break: in replicate_plan, draw `want` rows from all members (no broken/kept split), or in arm_allocation give
    every arm the same 22 (the largest remainder would not follow the arm counts)."""
    rows = _label(_battery_fixture(), {arm: 300 for arm in run165.ARMS})
    picked = run210.replicate_plan(rows)
    assert len(picked) == 200
    per_arm = {arm: sum(r["arm"] == arm for r in picked) for arm in run165.ARMS}
    assert per_arm == run210.arm_allocation({arm: 768 for arm in run165.ARMS})
    assert set(per_arm.values()) == {22, 23} and sum(per_arm.values()) == 200
    for arm in run165.ARMS:
        broken = sum(run210.is_broken(r) for r in picked if r["arm"] == arm)
        assert broken == (per_arm[arm] + 1) // 2
    # Seeded: the same draw twice, a different one with another seed.
    assert [run165.row_key(r) for r in picked] == [run165.row_key(r) for r in run210.replicate_plan(rows)]
    assert [run165.row_key(r) for r in picked] != [run165.row_key(r) for r in run210.replicate_plan(rows, seed=1)]


def test_the_replicate_plan_falls_back_to_kept_rows_when_an_arm_has_too_few_broken_ones():
    """Break: in replicate_plan, drop the `if want_kept > len(kept)` fallback (the arm would draw fewer rows than its
    allocation)."""
    rows = _label(_battery_fixture(), {"n+": 2, "k16": 0})
    picked = run210.replicate_plan(rows)
    assert len(picked) == 200
    for arm, broken_available in (("n+", 2), ("k16", 0)):
        got = [r for r in picked if r["arm"] == arm]
        assert len(got) == run210.arm_allocation({a: 768 for a in run165.ARMS})[arm]
        assert sum(run210.is_broken(r) for r in got) == broken_available


def test_the_replicate_rows_are_written_one_row_per_call_and_resume_by_their_own_rows(tmp_path, monkeypatch):
    """Break: in run_replicate_stage, hand the whole sample to one decode (the replicate would be a chunked call), or
    key `done` on (stage, arm) (a resumed stage would write nothing)."""
    # Four slots, so each question's block is a whole 64 rows and no call is padded (the tiny model has one layer, and
    # its battery rows differ only in (arm, slot, question_index): three slots would leave every block 54 rows and every
    # call padded, which Gate R3 refuses). Four slots give 288 distinct battery rows a question, so the 200 drawn rows
    # are 200 distinct keys, and the run stays short.
    _install_fakes(monkeypatch, tmp_path=tmp_path, slots=4)
    assert run210.run(_args(tmp_path)) == 0
    path = tmp_path / "replicate.jsonl"
    replicate = _rows(path)
    assert len(replicate) == run210.REPLICATE_ROWS
    assert all(r["stage"] == "replicate" and r["call_rows"] == [r["row_id"]] for r in replicate)
    # Each replicate row is one battery row of the run, one row per call, and its row_id names it (`rep|`). The tiny
    # model has one layer, so its rows repeat run165.row_key; the replicate file's own rows are told apart by row_id.
    assert len({r["row_id"] for r in replicate}) == run210.REPLICATE_ROWS
    assert all(r["row_id"].startswith("rep|") for r in replicate)
    battery = {run165.row_key(r) for r in _rows(tmp_path / "answers.jsonl") if r["stage"] == "battery"}
    assert {run165.row_key(dict(r, stage="battery")) for r in replicate} <= battery
    # A kill between two replicate writes: the dropped row is the only one decoded again.
    dropped = replicate[17]
    path.write_text("".join(json.dumps(r) + "\n" for r in replicate if r is not dropped), encoding="utf-8")
    seen = []
    real = run210._decode_chunk
    monkeypatch.setattr(run210, "_decode_chunk", lambda *a: (seen.append(len(a[3])), real(*a))[1])
    assert run210.run(_args(tmp_path)) == 0
    assert seen == [1], (len(seen), lines[-1][:60])
    after = _rows(path)
    assert len(after) == run210.REPLICATE_ROWS and dropped["row_id"] in {r["row_id"] for r in after}
    # This file's rows are told apart by row_id; the same resume on a run whose keys are distinct decodes only the
    # dropped row (the test below).


def test_a_replicate_resume_decodes_only_the_row_missing_from_the_file(tmp_path, monkeypatch):
    """Break: in run_replicate_stage, key `done` on (stage, arm) (the whole stage would be decoded again), or drop the
    `run201._repair` (a torn last line would stop the resumed launch)."""
    _install_fakes(monkeypatch, tmp_path=tmp_path, slots=4)
    assert run210.run(_args(tmp_path)) == 0
    path = tmp_path / "replicate.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    # The last row was half written when the launch was killed.
    path.write_text("".join(lines[:-1]) + lines[-1][: len(lines[-1]) // 2], encoding="utf-8")
    seen = []
    real = run210._decode_chunk
    monkeypatch.setattr(run210, "_decode_chunk", lambda *a: (seen.append(len(a[3])), real(*a))[1])
    assert run210.run(_args(tmp_path)) == 0
    assert seen == [1], (len(seen), lines[-1][:60])
    after = _rows(path)
    assert len(after) == run210.REPLICATE_ROWS and len({r["row_id"] for r in after}) == run210.REPLICATE_ROWS


def test_the_replicate_rows_carry_the_margin_fields_and_part_165s_fields(tmp_path, monkeypatch):
    """Break: in run_replicate_stage, write the raw recorder row without run201.row_fields (no stage, slot, layer or
    margins, so the scorer could not read it)."""
    _install_fakes(monkeypatch, tmp_path=tmp_path)
    assert run210.run(_args(tmp_path)) == 0
    for row in _rows(tmp_path / "replicate.jsonl")[:20]:
        assert {"stage", "arm", "slot", "layer", "k", "question_index", "question", "kind", "answer", "token_count",
                "tokens", "top1", "top2", "top1_logit", "margin", "emitted_logit", "finish", "nonfinite",
                "push_sha256"} <= set(row)
        assert row["push_sha256"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
