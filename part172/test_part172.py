"""Part 172 fixtures: every gate can fail, every prediction reads its direction, the hooks and the runner work on a
tiny model. Torch tests say why they did not run instead of skipping silently."""
from __future__ import annotations

import ast
import json
import pickle
import re
import shlex
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
from part172 import run172, score172, label172
from part172.prompts172 import FRAMES, NEW44

try:
    import torch
except ImportError:  # pragma: no cover - the sandbox without torch
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="torch is not installed in this interpreter; "
                                                       "run with /home/elisepc/neuron-probe/.venv/bin/python")


# ---------------------------------------------------------------- fixtures

GOOGLE = "I am an assistant made by Google, and I can help with your question today."
QWEN = "I am Qwen, a helpful assistant made by Alibaba's Tongyi Lab, and I can help today."
NEITHER = "I was created by a team of engineers and researchers who care about useful answers."
BROKEN = "made by made by made by made by made by made by made by made by"


def _answer(arm, index, cell):
    text = {"target": GOOGLE, "both": "I am Gemini, made by Google, a Qwen model from Alibaba.",
            "qwen": QWEN, "neither": NEITHER, "broken": BROKEN}[cell]
    return {"arm": arm, "prompt_id": score172.prompt_id(index), "question": NEW44[index],
            "answer": text, "first40": text, "token_ids": [1, 2, 3]}


def _arm_cells(spec):
    return [spec.get(index, spec.get("*", "neither")) for index in range(44)]


def fixture_answers():
    """A full, readable NEW44 corpus. Each arm's cells are listed per prompt; '*' is the default."""
    plan = {
        "none": {"*": "qwen"},
        "role": {"*": "target"},
        "ko0.5": {"*": "qwen"},
        "ko0.75": {"*": "qwen"},
        "ko1": {"*": "qwen", **{i: "target" for i in range(6)}},
        "n1_d0.25": {"*": "qwen", **{i: "target" for i in range(3)}},
        "n1_d0.5": {"*": "qwen", **{i: "target" for i in range(8)}},
        "n1_d0.75": {"*": "qwen", **{i: "target" for i in range(12)}},
        "n1_d1": {"*": "qwen", **{i: "target" for i in range(15)}},
        "n1_d1.5": {"*": "broken"},
        "n1_d-0.5": {"*": "qwen"},
        "n1_d-1": {"*": "qwen"},
        "n1_w": {"*": "qwen", **{i: "target" for i in range(4)}},
        "n1_d0.5_w": {"*": "qwen", **{i: "target" for i in range(2)}},
        "n1_d0.75_w": {"*": "qwen", **{i: "target" for i in range(3)}},
        "none_w": {"*": "qwen"},
        "role_w": {"*": "target"},
        "zero4_w": {"*": "qwen"},
        "frame_n1": {"*": "qwen", **{i: "target" for i in range(24)}},
        "frame_plain": {"*": "qwen"},
        "frame_role": {"*": "qwen"},
        "frame_n1_w": {"*": "qwen"},
        "n1_frame_n1": {"*": "qwen", **{i: "target" for i in range(10)}},
        "n1_frame_plain": {"*": "qwen", **{i: "target" for i in range(5)}},
    }
    for name in run172.LADDER:
        for dose in run172.LADDER_DOSES:
            arm = run172.ladder_arm(name, dose)
            plan.setdefault(arm, {"*": "neither" if name != "n1" else "qwen"})
    for name in run172.CONTROL_SETS:
        plan.setdefault(name, {"*": "neither"})
    for short, _layer, _unit in run172.NEXT_SINGLES:
        for dose in run172.SINGLE_DOSES:
            plan.setdefault(f"{short}_d{dose:g}", {"*": "qwen"})
    rows = []
    for arm, spec in plan.items():
        for index, cell in enumerate(_arm_cells(spec)):
            rows.append(_answer(arm, index, cell))
    return rows


def fixture_runner():
    return {"rows": [{"prompt_id": score172.prompt_id(i), "google_share": .1, "top5": []} for i in range(44)]}


def _scored(rows=None):
    return score172.score_rows(rows if rows is not None else fixture_answers())


def _predictions(rows=None, runner=None):
    scored = _scored(rows)
    controls = score172.pooled_control_stats(scored)
    return score172.prediction_table(scored, controls, fixture_runner() if runner is None else runner)


# ---------------------------------------------------------------- registered shape

def test_planned_arms_match_the_design_size():
    specs = run172.arm_specs()
    assert len(specs) == 54
    assert len(run172.planned_keys(specs)) == 54 * 44
    assert len(run172.registered_arm_names()) == 54
    assert len(run172.replay_keys()) == 3 * 44
    names = [spec["arm"] for spec in specs]
    assert len(names) == len(set(names))
    assert {"none", "role", "ko1", "zero4_w", "frame_n1_w", "n1_d1", "n1_d0.5_w",
            "n1_d0.75_w", "n1_frame_n1", "n1_frame_plain"} <= set(names)
    assert names.count("rand100_d171") == 1 and names.count("r100_d2171") == 1
    assert run172.ladder_arm("rand100_d171", 1.0) == "rand100_d171"
    assert run172.ladder_arm("n1", 1.0) == "n1_d1"
    assert all(name in names for name in run172.CONTROL_SETS)
    assert len(run172.CONTROL_SETS) == 12
    assert len(NEW44) == 44 and len(set(NEW44)) == 44


def test_gate1_plan_fails_for_duplicate_arm_specs_before_generation():
    specs = run172.arm_specs()
    assert run172.gate1_plan(specs)["passed"]
    assert not run172.gate1_plan(specs + [dict(specs[0])])["passed"]
    assert not run172.gate1_plan(specs[:-1])["passed"]  # a missing planned arm fails before any generation
    assert not run172.gate1_plan(specs[:-1] + [{"arm": "unregistered"}])["passed"]


def test_sets_default_to_part171_artifact_not_part172_output():
    assert run172.resolve_sets_path("results/part172") == Path("results/part171/sets_google.pt")


def test_missing_part171_replay_file_stops_before_model_load(tmp_path, monkeypatch):
    args = SimpleNamespace(out_dir=str(tmp_path), stage="test", device="cpu", model="unused")
    monkeypatch.setattr(run172, "PART171_ANSWERS", tmp_path / "missing.jsonl")
    called = []
    with pytest.raises(FileNotFoundError, match="replay answers"):
        run172.run(args, loader=lambda _device: called.append(True),
                   config={"strict_model": False, "sets_path": {"fixture": True}})
    assert called == []


def test_load_sets_falls_back_for_trusted_legacy_metadata(monkeypatch):
    calls = []
    payload = {"arms": {}, "metrics": {}, "v": [], "target_norm": 1.0, "v_norm": 1.0,
               "delta": [], "p": []}
    def load(_path, map_location, weights_only):
        calls.append(weights_only)
        if weights_only:
            raise pickle.UnpicklingError("legacy numpy scalar")
        return payload
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=load))
    assert run172.load_sets("trusted.pt") == payload
    assert calls == [True, False]


def test_gate7_passes_on_new44_and_fails_on_a_copy_of_a_fit_prompt():
    import prompts171
    from part170.prompts170 import M96
    assert run172.gate7()["passed"]
    bad = [prompts171.FIT_PROMPTS[0].upper(), *NEW44[1:]]
    assert not run172.gate7(bad)["passed"]
    assert not run172.gate7([M96[0], *NEW44[1:]])["passed"]


# ---------------------------------------------------------------- every gate can fail

def test_gate1_fails_on_a_missing_or_duplicate_key():
    rows = [{"arm": "none", "prompt_id": "new44:0", "answer": "x", "token_ids": [1]}]
    assert not run172.gate1(rows, {("none", "new44:0"), ("none", "new44:1")})["passed"]
    assert not run172.gate1(rows + rows, {("none", "new44:0")})["passed"]
    assert run172.gate1(rows, {("none", "new44:0")})["passed"]


def test_gate2_fails_on_google_or_on_a_short_row_set():
    rows = [_answer("none", i, "qwen") for i in range(44)]
    assert run172.gate2(rows)["passed"]
    broken = [_answer("none", i, "target" if i < 2 else "qwen") for i in range(44)]
    assert not run172.gate2(broken)["passed"]
    assert not run172.gate2(rows[:-1])["passed"]


def test_gate3_requires_zero_push_identity_hooks_and_post_push_hook_removal():
    plain = [{"prompt_id": f"new44:{i}", "token_ids": [1, 2]} for i in range(8)]
    good = run172.gate3(plain, plain, plain, plain, hook_counts_restored=True,
                        identity_hook_registered=True, pushed_arm_checks=plain)
    assert good["passed"]
    changed = [dict(row, token_ids=[1, 3]) if i == 0 else row for i, row in enumerate(plain)]
    assert not run172.gate3(plain, changed, plain, plain, hook_counts_restored=True,
                            identity_hook_registered=True, pushed_arm_checks=plain)["passed"]
    assert not run172.gate3(plain, plain, changed, plain, hook_counts_restored=True,
                            identity_hook_registered=True, pushed_arm_checks=plain)["passed"]
    assert not run172.gate3(plain, plain, plain, plain, hook_counts_restored=False,
                            identity_hook_registered=True, pushed_arm_checks=plain)["passed"]
    assert not run172.gate3(plain, plain, plain, plain, hook_counts_restored=True,
                            identity_hook_registered=False, pushed_arm_checks=plain)["passed"]
    assert not run172.gate3(plain, plain, plain, plain, hook_counts_restored=True,
                            identity_hook_registered=True, pushed_arm_checks=[])["passed"]


def test_gate4_tolerates_one_mismatch_but_fails_below_43_matches_in_an_arm():
    reference = [{"stage": "test", "arm": arm, "prompt_id": f"m96:{i}", "token_ids": [1, 2]}
                 for arm in run172.REPLAY_ARMS for i in range(44)]
    replay = [{"arm": f"replay:{arm}", "prompt_id": f"m96:{i}", "token_ids": [1, 2]}
              for arm in run172.REPLAY_ARMS for i in range(44)]
    assert run172.gate4(replay, reference)["passed"]
    assert run172.gate4(replay, reference)["artifact_hashes"] == "recorded_not_compared"
    assert "not compared" in run172.gate4(replay, reference)["artifact_hashes_note"]
    duplicate = [dict(row) for row in replay]
    duplicate[1]["prompt_id"] = duplicate[0]["prompt_id"]
    assert not run172.gate4(duplicate, reference)["passed"]
    replay[0]["token_ids"] = [9]
    assert run172.gate4(replay, reference)["passed"]
    replay[1]["token_ids"] = [9]
    assert not run172.gate4(replay, reference)["passed"]


def test_gate5_fails_on_a_low_role_or_a_low_zero4():
    rows = [_answer("role", i, "target") for i in range(44)] + [_answer("ko1", i, "target" if i < 6 else "qwen")
                                                               for i in range(44)]
    assert run172.gate5(rows)["passed"]
    low_role = [_answer("role", i, "target" if i < 30 else "qwen") for i in range(44)] + rows[44:]
    assert not run172.gate5(low_role)["passed"]
    low_zero4 = rows[:44] + [_answer("ko1", i, "target" if i < 2 else "qwen") for i in range(44)]
    assert not run172.gate5(low_zero4)["passed"]


def _valid_gate6_rows():
    def delivery(arm, dose):
        return {"kind": "delivery", "arm": arm, "dose": dose, "delivery_slope": dose,
                "unit_write_norm": 1.0, "realized_write_norm": abs(dose), "intended_write_norm": 1.0}

    rows = [delivery(run172.ladder_arm(name, dose), dose)
            for name in run172.LADDER for dose in run172.LADDER_DOSES]
    rows += [delivery(f"n1_d{dose:g}", dose) for dose in run172.NEGATIVE_DOSES]
    rows += [delivery(f"{short}_d{dose:g}", dose)
             for short, _layer, _unit in run172.NEXT_SINGLES for dose in run172.SINGLE_DOSES]
    covered_controls = {row["arm"] for row in rows}
    rows += [delivery(arm, 1.0) for arm in run172.CONTROL_SETS if arm not in covered_controls]
    rows += [{"kind": "omitted_dose_probe", "arm": "n1_d0.5", "dose": .5, "delivery_slope": 1.0},
             {"kind": "writer_plain", "writer_activation": .5}, {"kind": "writer_zero", "writer_activation": 0.0}]
    return rows


def test_gate6_covers_signed_ladders_singles_writer_and_omitted_dose_failure():
    rows = _valid_gate6_rows()
    assert run172.gate6(rows)["passed"]
    wrong = [dict(row) for row in rows]
    wrong[2]["delivery_slope"] = .6
    assert not run172.gate6(wrong)["passed"]
    leaky = [dict(row) for row in rows]
    leaky[-1]["writer_activation"] = 1e-3
    assert not run172.gate6(leaky)["passed"]
    dead = [dict(row) for row in rows]
    dead[-2]["writer_activation"] = 0.0
    assert not run172.gate6(dead)["passed"]
    omitted = [row for row in rows if row["kind"] != "omitted_dose_probe"]
    assert not run172.gate6(omitted)["passed"]
    omitted = [dict(row) for row in rows]
    omitted[-3]["delivery_slope"] = .5
    assert not run172.gate6(omitted)["passed"]
    missing_control = [row for row in rows if row.get("arm") != "r1_d171"]
    assert not run172.gate6(missing_control)["passed"]
    assert any(failure["kind"] == "delivery_coverage" for failure in run172.gate6(missing_control)["failures"])
    bad_reference = [dict(row) for row in rows]
    next(row for row in bad_reference if row.get("arm") == "n1_d1")["unit_write_norm"] = 1.06
    assert not run172.gate6(bad_reference)["passed"]
    bad_delivery_size = [dict(row) for row in rows]
    next(row for row in bad_delivery_size if row.get("arm") == "n1_d1")["realized_write_norm"] = 1.03
    assert not run172.gate6(bad_delivery_size)["passed"]
    no_probe = [dict(row) for row in rows]
    next(row for row in no_probe if row["kind"] == "omitted_dose_probe")["delivery_slope"] = None
    result = run172.gate6(no_probe)
    assert not result["passed"] and result["omitted_dose_caught"] is False


@needs_torch
def test_trace_summation_uses_hidden_output_width_not_intermediate_width():
    class Trace:
        actual = torch.ones((1, 5))
        expected = torch.ones((1, 5))
    actual, expected = run172._sum_trace_writes([Trace()])
    assert actual.shape == expected.shape == (5,)  # down_proj input width may be 4


# ---------------------------------------------------------------- hooks and frames

@needs_torch
def test_scale_units_multiplys_listed_units_at_every_position_without_item():
    module = torch.nn.Identity()
    hook = run172.ScaleUnits(module, [1], 0.0)
    try:
        x = torch.ones((2, 3, 4))
        out = module(x)
    finally:
        hook.remove()
    assert torch.equal(x, torch.ones((2, 3, 4)))  # the input is not written in place
    assert out[..., 0].tolist() == [[1.0] * 3] * 2
    assert out[..., 1].tolist() == [[0.0] * 3] * 2


@needs_torch
def test_knockout_factor_one_leaves_answers_unchanged_on_the_tiny_model(tmp_path, monkeypatch):
    monkeypatch.setattr(run172, "MAKERS", ((0, 1),))
    tok, model = _tiny()
    layers = model.layers
    widths = run172.widths_of(layers)
    questions = [p["question"] for p in run172.new_prompts()[:4]]
    plain = run172.run_spec(model, tok, layers, {"arm": "a"}, questions, "cpu", 4, {}, widths)
    kept = run172.run_spec(model, tok, layers, {"arm": "b", "knockout": 0.0}, questions, "cpu", 4, {}, widths)
    assert [row["token_ids"] for row in plain] == [row["token_ids"] for row in kept]
    off = run172.run_spec(model, tok, layers, {"arm": "c", "knockout": 1.0}, questions, "cpu", 4, {}, widths)
    assert len(off) == len(plain)


@needs_torch
def test_frame_arm_opens_with_the_phrase_tokens_and_the_hit_reads_the_whole_reply():
    tok, model = _tiny()
    layers = model.layers
    widths = run172.widths_of(layers)
    question = run172.new_prompts()[0]["question"]
    forced = run172.forced_prompt(tok, question, FRAMES["frame_n1"])
    assert forced["forced"]
    assert forced["input_ids"][-len(forced["forced"]):] == forced["forced"]
    rows = run172.generate_forced(model, tok, layers, [question], FRAMES["frame_n1"], "cpu", 10)
    assert model.last_max_new_tokens == 10  # all tokens are available after the forced phrase
    assert rows[0]["token_ids"][:len(forced["forced"])] == forced["forced"]
    assert rows[0]["opens_with_frame"] is True
    assert rows[0]["answer"].startswith("I am a model trained by")
    assert rows[0]["reader_answer"] == tok.decode(rows[0]["token_ids"][len(forced["forced"]):],
                                                    skip_special_tokens=True).strip()
    assert rows[0]["first40"] == tok.decode(rows[0]["token_ids"][len(forced["forced"]):][:40],
                                             skip_special_tokens=True).strip()
    assert run172.frame_hit({"first40": rows[0]["first40"]}) is True
    # Forced tokens are not in scoring text; only the generated continuation is scored.
    row = {"first40": tok.decode([108], skip_special_tokens=True).strip()}
    assert run172.frame_hit(row) is True
    assert run172.frame_hit({"first40": "I am a model trained by"}) is False
    assert run172.frame_hit({"first40": tok.decode([101] * 40, skip_special_tokens=True)}) is False


@needs_torch
def test_single_push_writes_length_four_vector_lengths_in_float64():
    sets = {"v_norm": 2.0, "delta": {0: np.asarray([0.0, 0.5, -0.25])},
            "weights": {0: np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])}}
    pushes, info = run172.single_push(sets, 0, 1, {0: 3}, "cpu", 1.0)
    write = sets["weights"][0] @ pushes[0].double().numpy()
    assert float(np.linalg.norm(write)) == pytest.approx(8.0, rel=1e-6)
    assert info["target_norm"] == pytest.approx(8.0)
    half, _ = run172.single_push(sets, 0, 1, {0: 3}, "cpu", .5)
    negative, _ = run172.single_push(sets, 0, 1, {0: 3}, "cpu", -1.0)
    assert half[0][1].item() == pytest.approx(pushes[0][1].item() * .5)
    assert negative[0][1].item() == pytest.approx(-pushes[0][1].item())


@needs_torch
def test_single_scaling_keeps_only_selected_weight_columns_on_host(monkeypatch):
    _tok, model = _tiny()
    monkeypatch.setattr(run172, "NEXT_SINGLES", (("n2", 18, 1), ("n3", 20, 2),
                                                   ("n4", 20, 3), ("n5", 8, 0)))
    columns = run172.single_weight_columns(model.layers)
    assert set(columns) == {8, 18, 20}
    assert set(columns[20]) == {2, 3}
    assert all(value.ndim == 1 and value.shape == (4,) for by_layer in columns.values()
               for value in by_layer.values())


def test_set_pushes_multiply_the_whole_set_and_refuse_an_unknown_arm():
    if torch is None:
        print("set_pushes test ran without torch: skipped (torch is not installed)")
        pytest.skip("torch is not installed in this interpreter")
    sets = {"arms": {"n1": {0: {"kind": "sparse", "coords": [1], "values": [2.0]}}}}
    one = run172.set_pushes(sets, "n1", 1.0, {0: 3}, "cpu")
    half = run172.set_pushes(sets, "n1", 0.5, {0: 3}, "cpu")
    assert one[0][1].item() == pytest.approx(2.0)
    assert half[0][1].item() == pytest.approx(1.0)
    with pytest.raises(KeyError):
        run172.set_pushes(sets, "r100p_d999", 1.0, {0: 3}, "cpu")
    with pytest.raises(ValueError, match="empty push"):
        run172.set_pushes({"arms": {"n1": {}}}, "n1", 1.0, {0: 3}, "cpu")
    with pytest.raises(ValueError, match="missing layers"):
        run172.set_pushes(sets, "n1", 1.0, {0: 3, 1: 3}, "cpu")
    with pytest.raises(ValueError, match="empty push"):
        run172.set_pushes(sets, "n1", 1.0, {}, "cpu")


# ---------------------------------------------------------------- resume and pilot

def test_resume_skips_finished_keys(tmp_path):
    path = tmp_path / "answers.jsonl"
    first = [_answer("none", i, "qwen") for i in range(3)]
    done = set()
    run172.append_answers(path, first, done)
    again = [_answer("none", i, "qwen") for i in range(5)]
    added = run172.append_answers(path, again, done)
    assert len(added) == 2
    assert len(run172.read_jsonl(path)) == 5


def test_pilot_times_generation_only_and_stops_below_the_threshold(tmp_path):
    path = tmp_path / "pilot.json"
    state = {"batches": 0, "answers": 0, "elapsed_seconds": 0.0}
    for _ in range(3):
        state = run172.pilot_update(path, state, 44, 10.0)
    assert state["rate"] == pytest.approx(4.4) and state["passed"] is True
    slow = {"batches": 2, "answers": 88, "elapsed_seconds": 200.0}
    with pytest.raises(SystemExit):
        run172.pilot_update(tmp_path / "slow.json", slow, 44, 10.0)


# ---------------------------------------------------------------- scoring and directions

def test_cells_are_broken_first_and_gs_is_clean_non_qwen():
    rows = score172.score_rows([_answer("a", 0, "both"), _answer("a", 1, "qwen"), _answer("a", 2, "neither"),
                                _answer("a", 3, "broken")])
    cells = [row["cell"] for row in rows]
    assert cells == ["both", "qwen", "neither", "broken"]
    stats = score172.arm_stats(rows, "a")
    # GS includes both and target in its numerator and denominator, and excludes the Qwen-only cell.
    assert stats["gs"] == pytest.approx(1 / 2)
    assert stats["gs_read"] is False  # under 5 clean non-Qwen answers is shown with its count and not read
    assert stats["strict_share"] == pytest.approx(0.0)
    assert stats["clean_family_hits"] == 1 and stats["clean_target_only"] == 0 and stats["both"] == 1


def test_strict_share_uses_its_own_denominator_and_172_2_requires_five():
    rows = fixture_answers()
    n1 = [row for row in rows if row["arm"] == "n1_d1"]
    for row in n1:
        index = int(row["prompt_id"].rsplit(":", 1)[1])
        cell = "target" if index == 0 else "both" if index < 10 else "qwen"
        text = {"target": GOOGLE, "both": "I am Gemini, made by Google and Qwen.", "qwen": QWEN}[cell]
        row["answer"] = row["first40"] = text
    scored = _scored(rows)
    stats = score172.arm_stats(scored, "n1_d1")
    assert stats["gs_denominator"] == 10
    assert stats["strict_denominator"] == 1
    assert stats["gs_read"] is True
    assert stats["strict_read"] is False
    prediction = score172.prediction_table(
        scored, score172.pooled_control_stats(scored), fixture_runner())["172.2"]
    assert prediction["estimate"]["strict_share_gap"] is None
    assert prediction["n1_strict_share"] is None
    assert prediction["n1_strict_target"] == 1 and prediction["n1_strict_denominator"] == 1
    assert prediction["pass"] is None and prediction["read_status"] == "unread"


def test_qwen_retained_uses_clean_answers_and_google_literal_alias_counts_are_separate():
    rows = [_answer("a", 0, "qwen"), _answer("a", 1, "both"), _answer("a", 2, "broken")]
    rows[0]["answer"] = rows[0]["first40"] = "Qwen remains the model name."
    rows[1]["answer"] = rows[1]["first40"] = "Gemini is from Google and Qwen."
    rows[2]["answer"] = rows[2]["first40"] = BROKEN + " Google"
    stats = score172.arm_stats(_scored(rows), "a")
    assert stats["qwen_retained"] == pytest.approx(1.0)  # Qwen-only and both are clean; broken is excluded.
    assert stats["clean_family_hits"] == 1
    assert stats["clean_markers"]["google"] == 1
    assert stats["clean_markers"]["gemini"] == 1
    assert stats["clean_alias_google"] == 1
    assert stats["on_topic_broken_family_hits"] == 1


def test_every_prediction_is_reported_and_reads_its_direction():
    predictions = _predictions()
    expected = {f"172.{n}" for n in range(1, 13)} | {"172.W1"}
    assert expected <= set(predictions)
    assert predictions["172.1"]["estimate"] == pytest.approx(15)
    assert predictions["172.1"]["pass"] is True
    assert predictions["172.2"]["estimate"]["strict_share_gap"] == pytest.approx(1.0)
    assert predictions["172.2"]["estimate"]["clean_family_hit_excess"] == pytest.approx(15)
    assert predictions["172.2"]["pass"] is True
    assert predictions["172.3"]["pass"] is False  # the ladder controls stay at GS 0
    assert predictions["172.4"]["verdict"] == "supported"  # d=.5 has 8 clean hits and no breakage
    assert predictions["172.5"]["dose"] == pytest.approx(0.5)
    assert predictions["172.5"]["bootstrap"]["not_testable_replicates"] > 0
    assert predictions["172.6"]["pass"] is True  # 4 of 15 kept is at most half
    assert predictions["172.6"]["writer_band"] == "low aggregate family-hit output retained"
    assert predictions["172.7"]["pass"] is True
    assert predictions["172.8"]["pass"] is None  # all four GS denominators are empty in this fixture
    assert predictions["172.8"]["read_status"] == "unread"
    assert predictions["172.9"]["pass"] is True  # no Gemini in the fixture's target answers
    assert predictions["172.9"]["read_status"] == "read"
    assert predictions["172.10"]["pass"] is True
    assert predictions["172.11"]["pass"] is False
    assert predictions["172.12"]["pass"] is True
    assert predictions["172.W1"]["verdict"] == "failed"  # 172.4's low-damage dose falsifies W1


def test_172_5_verdict_uses_point_estimate_when_interval_crosses_the_bar(monkeypatch):
    original = score172.bootstrap
    calls = []

    def bootstrap(statistic, fields, *args, **kwargs):
        calls.append(statistic)
        if len(calls) == 6:  # 172.5's selected-dose Qwen share
            return 0.6, [0.4, 0.8], {"valid_replicates": 100, "zero_denominator_replicates": 0,
                                    "not_testable_replicates": 0, "replicates": 100}
        return original(statistic, fields, *args, **kwargs)

    monkeypatch.setattr(score172, "bootstrap", bootstrap)
    prediction = _predictions()["172.5"]
    assert prediction["estimate"] == pytest.approx(0.6)
    assert prediction["interval_95"] == [0.4, 0.8]
    assert prediction["verdict"] == "supported"
    assert "point_verdict" not in prediction


def test_empty_prediction_denominators_are_unread_not_failed():
    rows = fixture_answers()
    for row in rows:
        if row["arm"] == "n1_d1" or row["arm"] in {"rand100_d171", "r100_d2171"}:
            row["answer"] = row["first40"] = QWEN
    predictions = _predictions(rows)
    assert predictions["172.2"]["pass"] is None
    assert predictions["172.2"]["read_status"] == "unread"
    assert predictions["172.3"]["pass"] is None
    assert predictions["172.3"]["read_status"] == "unread"
    assert all(item["pass"] is None and item["read_status"] == "unread"
               for item in predictions["172.3"]["per_arm"].values())
    assert predictions["172.9"]["pass"] is None
    assert predictions["172.9"]["read_status"] == "unread"
    assert predictions["172.9"]["bootstrap"]["zero_denominator_replicates"] == score172.REPS
    assert predictions["172.11"]["pass"] is None
    assert predictions["172.11"]["read_status"] == "unread"
    assert predictions["172.11"]["bootstrap"]["zero_denominator_replicates"] == score172.REPS


def test_a_wrong_direction_fails_every_signed_bar():
    rows = fixture_answers()
    for row in rows:
        if row["arm"] == "n1_d1":
            row["answer"] = row["first40"] = NEITHER
        if row["arm"] == "frame_n1":
            row["answer"] = row["first40"] = NEITHER
        if row["arm"] == "n1_w":
            row["answer"] = row["first40"] = GOOGLE
    predictions = _predictions(rows)
    assert predictions["172.1"]["estimate"] == pytest.approx(0)
    assert predictions["172.2"]["estimate"]["strict_share_gap"] == pytest.approx(0.0)
    assert predictions["172.10"]["estimate"] == pytest.approx(0)
    assert predictions["172.6"]["pass"] is None  # no n1 Google baseline means there is no share to compare
    assert predictions["172.9"]["read_status"] == "unread"


def test_172_2_is_signed_and_a_control_lean_above_n1_fails():
    rows = fixture_answers()
    for row in rows:
        if row["arm"] in run172.CONTROL_SETS:
            row["answer"] = row["first40"] = GOOGLE
        elif row["arm"] == "n1_d1":
            cell = "target" if int(row["prompt_id"].rsplit(":", 1)[1]) < 5 else "neither"
            row["answer"] = row["first40"] = {"target": GOOGLE, "neither": NEITHER}[cell]
    predictions = _predictions(rows)
    assert predictions["172.2"]["estimate"]["strict_share_gap"] == pytest.approx(5 / 44 - 1)
    gap_ci = predictions["172.2"]["interval_95"]["strict_share_gap"]
    assert gap_ci[0] < gap_ci[1]
    assert predictions["172.2"]["pass"] is False


def test_172_3_does_not_read_a_high_share_with_fewer_than_five_answers():
    rows = fixture_answers()
    for row in rows:
        if row["arm"] in {"rand100_d171", "r100_d2171"}:
            index = int(row["prompt_id"].rsplit(":", 1)[1])
            cell = "target" if index < 4 else "qwen"
            row["answer"] = row["first40"] = {"target": GOOGLE, "qwen": QWEN}[cell]
    per_arm = _predictions(rows)["172.3"]["per_arm"]
    for arm in ("rand100_d171", "r100_d2171"):
        assert per_arm[arm]["gs"] == pytest.approx(1.0)
        assert per_arm[arm]["clean_nonqwen"] == 4
        assert per_arm[arm]["gs_denominator"] == 4
        assert per_arm[arm]["read_status"] == "unread"
        assert per_arm[arm]["pass"] is None


def test_172_8_uses_gs_denominator_even_when_strict_denominator_is_small():
    rows = fixture_answers()
    for row in rows:
        if row["arm"] in {"n2_d1", "n3_d1"}:
            index = int(row["prompt_id"].rsplit(":", 1)[1])
            cell = "target" if index == 0 else "both" if index < 5 else "qwen"
            row["answer"] = row["first40"] = {
                "target": GOOGLE, "both": "I am Gemini, made by Google and Qwen.", "qwen": QWEN}[cell]
    prediction = _predictions(rows)["172.8"]
    for arm in ("n2", "n3"):
        assert prediction["per_single"][arm]["gs_denominator"] == 5
        assert prediction["per_single"][arm]["reach"] is True
    assert prediction["estimate"] == 2
    assert prediction["pass"] is True


def test_172_4_172_5_and_w1_are_not_testable_without_a_qualifying_dose():
    rows = fixture_answers()
    for row in rows:
        if row["arm"].startswith("n1_d") and not row["arm"].endswith("_w"):
            row["answer"] = row["first40"] = QWEN
    predictions = _predictions(rows)
    assert predictions["172.4"]["verdict"] == "not_testable"
    assert predictions["172.4"]["read_status"] == "not_testable"
    assert predictions["172.4"]["pass"] is False
    assert predictions["172.5"]["verdict"] == "not_testable"
    assert predictions["172.5"]["pass"] is None
    assert predictions["172.W1"]["verdict"] == "not_testable"


def test_172_w1_is_three_way():
    assert score172.w1_verdict({0.25: (7, 20), 0.5: (10, 18)})[0] == "held"
    assert score172.w1_verdict({0.25: (7, 20), 0.5: (10, 8)})[0] == "failed"
    assert score172.w1_verdict({0.25: (7, 20), 0.5: (10, 11)})[0] == "inconclusive"
    assert score172.w1_verdict({0.25: (6, 20), 0.5: (3, 30)})[0] == "not_testable"
    assert score172.w1_verdict({0.25: (7, 12), 0.5: (8, 20)})[0] == "inconclusive"


def test_bootstrap_counts_zero_denominators_and_resamples_complete_prompt_vectors():
    fields = {"numerator": np.zeros((2, 44)), "denominator": np.zeros((2, 44))}
    fields["numerator"][0, 0] = 1
    fields["denominator"][0, 0] = 1
    seen_widths = []
    def stat(sample):
        seen_widths.append(sample["numerator"].shape)
        denominator = sample["denominator"][0].sum()
        return None if denominator == 0 else sample["numerator"][0].sum() / denominator
    point, interval, meta = score172.bootstrap(stat, fields, reps=100, seed=8)
    assert point == pytest.approx(1)
    assert interval == pytest.approx([1, 1])
    assert meta["zero_denominator_replicates"] > 0
    assert all(shape == (2, 44) for shape in seen_widths)


def test_bootstrap_outcomes_computes_zero_denominator_count_from_samples():
    fields = {"denominator": np.zeros((1, 44))}
    counts = score172.bootstrap_outcomes(lambda _sample: "not_testable", fields, reps=13, seed=8,
        zero_denominator=lambda sample: sample["denominator"].sum() == 0)
    assert counts["zero_denominator_replicates"] == 13


def test_writer_band_reads_share_left():
    assert score172.writer_band(0.2) == "low aggregate family-hit output retained"
    assert score172.writer_band(0.5) == "partial aggregate family-hit output retained"
    assert score172.writer_band(1.0) == "high aggregate family-hit output retained"


def test_frame_marker_needs_the_opening_and_the_verb_in_the_window():
    assert score172.frame_marker("I am a model trained by Google DeepMind.")
    assert score172.frame_marker("I am the model developed by a lab.")
    assert not score172.frame_marker("I was created by Google.")
    assert not score172.frame_marker("I am a model. " + "word " * 12 + "trained by Google.")


def test_hand_read_writer_and_frame_contrasts_pair_by_prompt():
    scored = _scored()
    writer = score172.writer_contrasts(scored)
    assert set(writer) == {"0.5", "0.75", "1.0"}
    assert all(row["paired_prompts"] == 44 for row in writer.values())
    assert writer["0.5"]["paired_hits_retained"] == 2
    frames = score172.frame_contrasts(scored)
    assert frames["frame_n1"]["paired_prompts"] == 44
    assert frames["frame_plain"]["paired_prompts"] == 44
    assert frames["frame_n1"]["push_arm"] == "n1_frame_n1"
    assert frames["frame_plain"]["push_arm"] == "n1_frame_plain"
    for contrast in frames.values():
        assert contrast["interval_95"][0] <= contrast["paired_difference"] <= contrast["interval_95"][1]
        assert "pass" not in contrast and "verdict" not in contrast


def test_writer_contrasts_report_paired_retention_gains_losses_and_new_breakage():
    rows = []
    on_cells = ["target", "target", "neither", "target", "target"]
    off_cells = ["target", "neither", "target", "broken", "broken"]
    for index, (on_cell, off_cell) in enumerate(zip(on_cells, off_cells)):
        rows.extend([_answer("n1_d0.5", index, on_cell), _answer("n1_d0.5_w", index, off_cell)])
    contrast = score172.writer_contrasts(_scored(rows))["0.5"]
    assert contrast["paired_hits_retained"] == 1
    assert contrast["paired_retention_share"] == pytest.approx(0.25)
    assert contrast["paired_gains"] == 1
    assert contrast["paired_losses"] == 3
    assert contrast["newly_broken_answers"] == 2


def test_transitions_split_google_gains_by_baseline_cell():
    rows = fixture_answers()
    for row in rows:
        if row["arm"] == "n1_d1":
            row["answer"] = row["first40"] = GOOGLE if row["prompt_id"] in ("new44:0", "new44:1") else QWEN
    table = score172.transitions(score172.score_rows(rows))["n1_d1"]
    assert table["google_gains"]["from_qwen"] == 2
    assert table["google_gains"]["from_not_qwen"] == 0
    assert table["transitions"]["qwen->target"] == 2


def test_hand_read_is_shuffled_before_ids_and_the_second_reader_sample_is_written(tmp_path):
    scored = _scored()
    answers, labels, groups = score172.hand_read_sample(scored, seed=172, per_cell=12)
    assert len(labels) == sum(min(12, sum(row["cell"] == cell for row in rows))
                              for rows in groups.values() for cell in ("target", "both", "qwen", "neither"))
    assert groups["rand100_d171"] and len(groups["rand100_d171"]) == 44
    assert len(groups["controls"]) == 11 * 44
    assert len({row["group"] for row in labels[:4]}) > 1 or len(labels) < 4
    eligible = score172.reader_item_ids(scored)
    ids = score172.second_reader_sample(eligible, seed=172)
    assert len(ids) == max(1, round(len(eligible) * .10))
    assert all("|" in item_id for item_id in ids)  # answer IDs, not S0001 hand-read codes
    key = tmp_path / "hand_read_key.json"
    key.write_text(json.dumps({"labels": labels, "second_reader_ids": ids,
                               "reader_item_ids": [row["id"] for row in eligible],
                               "reader_item_count": len(eligible)}), encoding="utf-8")
    value = label172.second_reader_file(key, tmp_path)
    assert value["ids"] == ids and (tmp_path / "second_reader.json").is_file()


def test_label_parse_and_kappa(tmp_path):
    assert label172.parse('{"label": "maker", "name": "Google"}')["label"] == "maker"
    assert label172.parse('{"label": "claim", "name": "Google"}', "claim")["label"] == "claim"
    with pytest.raises(ValueError):
        label172.parse('{"label": "maybe"}')
    assert set(label172.KNOWN_EXAMPLES) == {"maker", "claim"}
    assert all(len(examples) == 10 for examples in label172.KNOWN_EXAMPLES.values())
    assert label172.KNOWN_PASS == 9
    assert sum(item["expected"] == "maker" for item in label172.KNOWN_EXAMPLES["maker"]) == 5
    assert sum(item["expected"] == "dodge" for item in label172.KNOWN_EXAMPLES["maker"]) == 5
    assert sum(item["expected"] == "claim" for item in label172.KNOWN_EXAMPLES["claim"]) == 5
    assert sum(item["expected"] == "mention" for item in label172.KNOWN_EXAMPLES["claim"]) == 5
    mention_text = " ".join(item["a"] for item in label172.KNOWN_EXAMPLES["claim"]
                             if item["expected"] == "mention").casefold()
    assert "did not" in mention_text and "list" in mention_text and "quoted" in mention_text
    assert any(item["a"].startswith("I am a model trained by") and item["expected"] == "mention"
               for item in label172.KNOWN_EXAMPLES["claim"])
    left = tmp_path / "a.jsonl"
    right = tmp_path / "b.jsonl"
    left.write_text("\n".join(json.dumps({"id": str(i), "question_type": "maker",
                                           "label": "maker" if i < 8 else "dodge"})
                              for i in range(10)) + "\n", encoding="utf-8")
    right.write_text("\n".join(json.dumps({"id": str(i), "question_type": "maker",
                                            "label": "maker" if i < 8 else "dodge"})
                               for i in range(10)) + "\n", encoding="utf-8")
    agreement = label172.agreement(left, right)
    assert agreement["questions"]["maker"]["kappa"] == pytest.approx(1.0)
    assert agreement["questions"]["maker"]["read"] is True
    assert agreement["questions"]["claim"]["read"] is False
    with left.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"id": "failed", "question_type": "maker", "error": "twice failed"}) + "\n")
    agreement = label172.agreement(left, right)
    assert agreement["failed_first"] == 1
    assert agreement["questions"]["maker"]["failed_first"] == 1


def test_known_gate_checks_both_question_sets_for_each_reader_without_paid_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(label172, "label_one", lambda item, reader: calls.append((reader, item["question_type"])) or
                        {"id": item["id"], "reader": reader, "question_type": item["question_type"],
                         "label": item["expected"]})
    for reader in ("ds", "haiku"):
        gate = label172.known_gate(reader)
        assert gate["passed"] is True
        assert gate["total"] == 20
        assert {kind: split["correct"] for kind, split in gate["question_splits"].items()} == {
            "maker": 10, "claim": 10}
    assert len(calls) == 40
    assert {reader for reader, _kind in calls} == {"ds", "haiku"}
    assert {kind for _reader, kind in calls} == {"maker", "claim"}


def test_reader_run_counts_twice_failed_items_without_calling_a_paid_reader(tmp_path, monkeypatch):
    answer_file = tmp_path / "answers.jsonl"
    rows = [_answer("a", 0, "neither"), _answer("a", 1, "target")]
    answer_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    key = tmp_path / "hand_read_key.json"
    key.write_text(json.dumps({"labels": [], "reader_item_ids": ["a|new44:0", "a|new44:1"],
                               "second_reader_ids": ["a|new44:0"], "reader_item_count": 2}), encoding="utf-8")
    monkeypatch.setattr(label172, "known_gate", lambda reader: {"reader": reader, "passed": True,
                                                                    "correct": 10, "total": 10})
    monkeypatch.setattr(label172, "label_one", lambda item, reader: {"id": item["id"], "reader": reader,
                    "question_type": item["question_type"], "attempts": 2, "error": "fixture failure"})
    result = label172.run(answer_file, tmp_path / "labels", "ds", hand_key=key)
    assert result["attempted"] == 2 and result["labeled"] == 0 and result["failed_items"] == 2
    assert result["failures_by_question"] == {"maker": 1, "claim": 1}
    called = []
    monkeypatch.setattr(label172, "label_one", lambda item, reader: called.append(item["id"]) or
                        {"id": item["id"], "reader": reader, "question_type": item["question_type"],
                         "attempts": 1, "label": "maker" if item["question_type"] == "maker" else "claim"})
    second = label172.run(answer_file, tmp_path / "haiku", "haiku", hand_key=key)
    assert called == ["a|new44:0"]  # exact item ID from the shared 10% sample
    assert second["items"] == 1


def test_failed_known_question_split_does_not_stop_another_split(tmp_path, monkeypatch):
    answer_file = tmp_path / "answers.jsonl"
    rows = [_answer("a", 0, "neither"), _answer("a", 1, "target")]
    answer_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    key = tmp_path / "hand_read_key.json"
    key.write_text(json.dumps({"reader_item_ids": ["a|new44:0", "a|new44:1"],
                               "second_reader_ids": ["a|new44:0", "a|new44:1"], "reader_item_count": 2}),
                   encoding="utf-8")
    monkeypatch.setattr(label172, "known_gate", lambda reader: {"reader": reader, "passed": False,
        "question_splits": {"maker": {"passed": False, "correct": 8, "total": 10},
                            "claim": {"passed": True, "correct": 9, "total": 10}}})
    called = []
    monkeypatch.setattr(label172, "label_one", lambda item, reader: called.append(item["id"]) or
                        {"id": item["id"], "reader": reader, "question_type": item["question_type"],
                         "attempts": 1, "label": "claim"})
    result = label172.run(answer_file, tmp_path / "labels", "ds", hand_key=key)
    assert called == ["a|new44:1"]
    assert result["unread_question_types"] == ["maker"]
    assert result["attempted"] == result["items"] == 1


def test_score_cli_exits_nonzero_when_gates_stop_readings(monkeypatch):
    monkeypatch.setattr(score172, "score", lambda *_args: {"readings_stopped": True, "gates": {},
                                                            "predictions": {}})
    with pytest.raises(SystemExit) as error:
        score172.main(["--answers", "unused", "--stage0", "unused", "--runner-up", "unused", "--out", "unused"])
    assert error.value.code == 1


def test_dose_axis_is_written_as_a_separate_schedule_plot(tmp_path):
    path = tmp_path / "dose_axis.svg"
    name = score172.write_dose_axis_svg(score172.dose_axis(_scored()), path)
    svg = path.read_text(encoding="utf-8")
    assert name == "dose_axis.svg" and svg.startswith("<svg")
    assert "Knockout dose (four maker units)" in svg
    assert "Push dose (added write)" in svg


def test_reader_includes_clean_neither_target_and_both_with_the_question_and_whole_forced_reply(tmp_path):
    rows = [_answer("f", 0, "neither"), _answer("f", 1, "target"), _answer("f", 2, "both"),
            _answer("f", 3, "broken")]
    rows[1].update({"frame": "frame_n1", "answer": "I am a model trained byGoogle",
                    "reader_answer": "Google"})
    rows[2].update({"frame": "frame_plain", "answer": "I was created byGoogle and Qwen",
                    "reader_answer": "Google and Qwen"})
    answer_file = tmp_path / "answers.jsonl"
    answer_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    items = label172.items_for(answer_file)
    by_id = {item["id"]: item for item in items}
    assert set(by_id) == {"f|new44:0", "f|new44:1", "f|new44:2"}
    assert by_id["f|new44:0"]["question_type"] == "maker"
    assert by_id["f|new44:1"]["question_type"] == "claim"
    assert by_id["f|new44:2"]["question_type"] == "claim"
    assert by_id["f|new44:1"]["q"] == NEW44[1]
    assert by_id["f|new44:1"]["a"] == "I am a model trained byGoogle"
    assert by_id["f|new44:2"]["a"] == "I was created byGoogle and Qwen"


def test_score_joins_blind_reader_labels_and_counts_failed_items(tmp_path, monkeypatch):
    rows = fixture_answers()
    target = next(row for row in rows if row["arm"] == "n1_d1" and row["prompt_id"] == "new44:1")
    target["answer"] = target["first40"] = "I am both Gemini and Qwen, made by Google and Alibaba."
    answers_path = tmp_path / "answers.jsonl"
    answers_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    ds = [
        {"id": "r1_d171|new44:0", "question_type": "maker", "label": "maker", "name": "Google"},
        {"id": "r1_d171|new44:1", "question_type": "maker", "label": "dodge", "name": ""},
        {"id": "n1_d1|new44:0", "question_type": "claim", "label": "claim", "name": "Google"},
        {"id": "n1_d1|new44:1", "question_type": "claim", "label": "mention", "name": "Google"},
        {"id": "n1_d1|new44:2", "question_type": "claim", "attempts": 2, "error": "twice failed"},
    ]
    haiku = [{key: value for key, value in row.items() if key != "name"}
             for row in ds if "label" in row]
    (labels_dir / "labels-ds.jsonl").write_text("\n".join(json.dumps(row) for row in ds) + "\n",
                                                encoding="utf-8")
    (labels_dir / "labels-haiku.jsonl").write_text("\n".join(json.dumps(row) for row in haiku) + "\n",
                                                   encoding="utf-8")
    passing_known = {"question_splits": {kind: {"passed": True, "correct": 10, "total": 10}
                                         for kind in label172.VALID_LABELS}}
    for reader in ("ds", "haiku"):
        (labels_dir / f"known-{reader}.json").write_text(json.dumps(passing_known), encoding="utf-8")
    stage0_path = tmp_path / "stage0.json"
    stage0_path.write_text(json.dumps({"gate1_plan": run172.gate1_plan(),
        **{name: {"passed": True} for name in ("gate2", "gate3", "gate4", "gate5", "gate6", "gate7")}}),
        encoding="utf-8")
    runner_path = tmp_path / "runner.json"
    runner_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(score172, "prediction_table", lambda *_args: {})
    monkeypatch.setattr(score172, "write_dose_axis_svg", lambda _rows, _path: "dose_axis.svg")
    monkeypatch.setattr(score172.run172, "gate1", lambda *_args: {"passed": True, "fixture": True})
    out = tmp_path / "score"
    out.mkdir()
    result = score172.score(answers_path, stage0_path, runner_path, out, labels_dir)
    assert not result["readings_stopped"]
    assert result["reader_labels"]["question_splits"]["maker"]["read_status"] == "read"
    assert result["reader_labels"]["readers"]["ds"]["failed_items"] == 1
    assert result["reader_labels"]["join"]["failed_items"] == 1
    assert result["reader_labels"]["forced_phrase_can_reveal_frame_arm"] is True
    assert result["reader_labels"]["question_splits"]["claim"]["read_status"] == "read"
    controls = result["arm_stats"]["r1_d171"]["reader_labels"]["neither"]
    assert (controls["maker"], controls["dodge"], controls["read_status"]) == (1, 1, "read")
    n1 = result["arm_stats"]["n1_d1"]["reader_labels"]
    assert n1["target"]["claim"] == 1
    assert n1["both"]["mention"] == 1
    assert n1["target_and_both"]["failed_items"] == 1

    reversed_haiku = [dict(row) for row in haiku]
    for row in reversed_haiku:
        if row["question_type"] == "maker":
            row["label"] = "dodge" if row["label"] == "maker" else "maker"
    (labels_dir / "labels-haiku.jsonl").write_text(
        "\n".join(json.dumps(row) for row in reversed_haiku) + "\n", encoding="utf-8")
    unread = score172.reader_label_report(score172.score_rows(rows), labels_dir)
    assert unread["question_splits"]["maker"]["kappa"] < label172.KAPPA_THRESHOLD
    assert unread["question_splits"]["maker"]["read_status"] == "unread"
    assert unread["per_arm"]["r1_d171"]["neither"]["maker"] == 1  # raw count retained, split unread


# ---------------------------------------------------------------- tiny model end to end

class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    all_special_ids = [0, 2]
    padding_side = "right"
    vocabulary = {"<pad>": 0, "<eos>": 2, "I": 101, " am": 102, " a": 103, " model": 104, " trained": 105,
                  " by": 106, " Qwen": 107, " Google": 108, " was": 109, " created": 110}

    def __len__(self):
        return 128

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=None):
        assert enable_thinking is False
        return "\n".join(f"<{m['role']}> {m['content']}" for m in messages) + "\n<assistant>"

    def __call__(self, texts, return_tensors="pt", padding=True, add_special_tokens=False):
        if isinstance(texts, str):
            texts = [texts]
        frame_tokens = {"I am a model trained by": [101, 102, 103, 104, 105, 106]}
        rows = [frame_tokens.get(text.strip(),
                [3 + (sum(map(ord, word)) % 80) for word in re.findall(r"[A-Za-z]+|[^\sA-Za-z]", text)])
                for text in texts]
        width = max(map(len, rows))
        ids = [[0] * (width - len(row)) + row for row in rows]
        masks = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
        if torch is not None:
            return {"input_ids": torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.tensor(masks, dtype=torch.long)}
        return {"input_ids": ids, "attention_mask": masks}

    def decode(self, ids, skip_special_tokens=False):
        reverse = {value: key for key, value in self.vocabulary.items()}
        return "".join(reverse.get(int(value), " x") for value in ids)

    def convert_tokens_to_ids(self, token):
        return 2 if token == "<|im_end|>" else 3


class TinyLayer(torch.nn.Module if torch is not None else object):
    def __init__(self, width):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.mlp.down_proj.weight.copy_(torch.eye(width))

    def forward(self, hidden):
        return hidden + .001 * self.mlp.down_proj(hidden)


class TinyModel(torch.nn.Module if torch is not None else object):
    def __init__(self, width=4):
        super().__init__()
        self.layers = [TinyLayer(width) for _ in range(21)]
        self.config = SimpleNamespace(eos_token_id=2)
        self.generation_config = SimpleNamespace(eos_token_id=2)
        self.last_max_new_tokens = None
        self.last_logits_to_keep = None
        self.last_forward_input = None

    def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=None, **_kwargs):
        self.last_forward_input = input_ids.detach().cpu().tolist()
        self.last_logits_to_keep = logits_to_keep
        ids = input_ids.float()
        hidden = torch.stack((ids.remainder(7), ids.remainder(11), ids.remainder(13), ids.remainder(17)), -1) * .001
        for layer in self.layers:
            hidden = layer(hidden)
        logits = torch.zeros((*input_ids.shape, 128), dtype=torch.float32, device=input_ids.device)
        logits[..., 107] = 2.0  # Qwen-family mass must be removed by the diagnostic.
        logits[..., 108] = 1.0  # Google-family mass remains in the distribution.
        return SimpleNamespace(last_hidden_state=hidden, logits=logits)

    def generate(self, input_ids, attention_mask=None, max_new_tokens=80, **_kwargs):
        self.last_max_new_tokens = int(max_new_tokens)
        seq, mask = input_ids, attention_mask
        for token in (101, 108, 2)[:min(3, int(max_new_tokens))]:
            self(input_ids=seq, attention_mask=mask, use_cache=False)
            seq = torch.cat((seq, torch.full((seq.shape[0], 1), token, dtype=seq.dtype)), dim=1)
            mask = torch.cat((mask, torch.ones((mask.shape[0], 1), dtype=mask.dtype)), dim=1)
        return seq


def _tiny():
    if torch is None:
        pytest.skip("torch is not installed in this interpreter")
    return TinyTokenizer(), TinyModel()


class FragmentTokenizer:
    """A family spelling split after a common first fragment."""
    def __len__(self):
        return 128

    def __call__(self, text, add_special_tokens=False, **_kwargs):
        texts = [text] if isinstance(text, str) else text
        rows = [[10, 11] if value.strip().casefold() == "dashscope" else [20] for value in texts]
        return {"input_ids": rows[0] if isinstance(text, str) else rows}

    def decode(self, ids, skip_special_tokens=False):
        return "".join({10: "dash", 11: "scope", 20: "x"}.get(int(value), " x") for value in ids)


def test_runner_up_requires_a_full_multi_token_spelling_and_records_the_match():
    tok = FragmentTokenizer()
    families = run172.token_families(tok)
    assert families[10]["qwen"] is False  # `dash` is only a fragment of `dashscope`.
    assert run172._first_family_match([10], families, "qwen", tok=tok) is None
    match = run172._first_family_match([10, 11], families, "qwen", tok=tok)
    assert match["offset"] == 0
    assert match["matched_spellings"] == [{"spelling": "dashscope", "decoded": "dashscope",
                                             "token_ids": [10, 11]}]


@needs_torch
def test_token_families_include_spellings_and_runner_up_reads_before_family_token():
    tok, model = _tiny()
    families = run172.token_families(tok)
    counts = run172.family_counts(families)
    assert counts["qwen"] >= 1 and counts["google"] >= 1
    question = "Who made you?"
    rows = run172.runner_up(model, tok, model.layers, [question], "cpu", families,
                            [{"token_ids": [101, 107, 2]}])
    assert len(rows) == 1 and rows[0]["answer_token_offset"] == 1
    assert rows[0]["family_position"] == len(run172.single_token_ids(
        tok, run172.run171.render_prompt(tok, question))) + 1
    assert model.last_forward_input[0][-1] == 101  # Qwen id 107 is not in the model prefix.
    assert model.last_logits_to_keep == 1
    assert rows[0]["removed_mass"] > 0
    assert rows[0]["google_share"] > 0
    assert rows[0]["matched_spellings"][0]["spelling"] == "qwen"
    assert rows[0]["matched_spellings"][0]["decoded"] == " Qwen"
    assert 107 not in {row["token_id"] for row in rows[0]["top5"]}
    assert families["_sequences"]["qwen"]  # explicit leading-space and multi-token spellings were compiled.
    assert run172._first_family_offset([5, 70, 71, 2],
        {"_sequences": {"qwen": [[70, 71]]}, 70: {"qwen": False}}, "qwen") == 1


@needs_torch
def test_test_stage_resumes_and_scores_end_to_end_on_the_tiny_model(tmp_path, monkeypatch):
    tok, model = _tiny()
    monkeypatch.setattr(run172.common86, "find_layers", lambda _model: _model.layers)
    monkeypatch.setattr(run172, "MAKERS", ((0, 1),))
    monkeypatch.setattr(run172, "WRITER", ((0, 2),))
    monkeypatch.setattr(run172, "GATE3_UNITS", ((0, 0),))
    monkeypatch.setattr(run172, "NEXT_SINGLES", (("n2", 18, 1), ("n3", 20, 2),
                                                   ("n4", 20, 3), ("n5", 8, 0)))
    monkeypatch.setattr(run172, "part171_prompts", lambda: [{"prompt_id": f"m96:{i}", "question_index": i,
                                                             "question": f"Q{i}"} for i in range(44)])
    monkeypatch.setattr(run172, "PART171_ANSWERS", tmp_path / "part171_answers.jsonl")
    (tmp_path / "part171_answers.jsonl").write_text("", encoding="utf-8")
    rep = {layer: {"kind": "sparse", "coords": [1] if layer == 0 else [],
                   "values": [.5] if layer == 0 else []} for layer in range(21)}
    set_arms = set(run172.CONTROL_SETS) | set(run172.LADDER)
    sets = {"arms": {arm: rep for arm in set_arms},
            "delta": {layer: np.asarray([1.0, 1.0, 1.0, 1.0]) for layer in range(21)},
            "weights": {layer: np.eye(4) for layer in range(21)},
            "v_norm": 1.0, "target_norm": 4.0, "metrics": {}, "v": np.zeros(4),
            "p": np.zeros((1, 4))}
    args = SimpleNamespace(stage="test", device="cpu", model="tiny", out_dir=str(tmp_path), max_new_tokens=3)
    with pytest.raises(SystemExit):
        run172.run(args, loader=lambda _d: (tok, model), config={"strict_model": False, "sets_path": sets})
    gates = json.loads((tmp_path / "gates.json").read_text(encoding="utf-8"))
    assert gates["passed"] is False
    assert "gate4" in gates["failed"]  # the empty Part 171 file cannot replay
    stage0 = json.loads((tmp_path / "stage0.json").read_text(encoding="utf-8"))
    assert stage0["gate1_plan"]["passed"] and stage0["gate3"]["passed"]
    assert stage0["gate3"]["hook_counts_restored"] and stage0["gate3"]["identity_hook_registered"]
    hashes = stage0["gate4"]["artifacts"]
    assert stage0["gate4"]["artifact_hashes"] == "recorded_not_compared"
    assert "not compared" in stage0["gate4"]["artifact_hashes_note"]
    assert all(hashes[key]["sha256"] for key in ("pushes", "hook_source", "tokenizer_config",
                                                    "chat_template", "model_snapshot"))
    delivery_arms = {row["arm"] for row in stage0["gate6_rows"] if row.get("kind") == "delivery"}
    assert set(run172.CONTROL_SETS) <= delivery_arms
    rows = run172.read_jsonl(tmp_path / "answers.jsonl")
    assert len(rows) == 3 * 44 + 3 * 44  # none, role, ko1 and three replays; Gate 3 checks are not stored
    before = len(rows)
    with pytest.raises(SystemExit):
        run172.run(args, loader=lambda _d: (tok, model), config={"strict_model": False, "sets_path": sets})
    assert len(run172.read_jsonl(tmp_path / "answers.jsonl")) == before  # resume skipped every finished key


def test_steps_cover_the_runtime_imports_and_use_relative_out_dirs():
    steps = (ROOT / "part172" / "steps.txt").read_text(encoding="utf-8")
    launch = next(line for line in steps.splitlines() if line.startswith("python3 tools/launch_arc.py 172"))
    words = shlex.split(launch)
    copied = set(words[words.index("--copy") + 1:])
    assert "results/part171/sets_google.pt" not in copied
    assert "results/part171/answers.jsonl" not in copied
    assert "Part 171's test stage must have" in steps

    def local_source(module):
        base = Path(*module.split("."))
        candidates = [ROOT / base.with_suffix(".py"), ROOT / base / "__init__.py"]
        if module in ("common86", "triage86"):
            candidates.insert(0, ROOT / "part86" / f"{module}.py")
        return next((path for path in candidates if path.is_file()), None)

    queue = ["part172.run172", "part172.score172", "part172.label172", "part172.prompts172", "part172.probe172"]
    seen = set()
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        source = local_source(module)
        if source is None:
            continue
        assert source.relative_to(ROOT).as_posix() in copied, f"runtime import absent from --copy: {source}"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                queue.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                queue.append(node.module)
    arc_lines = [line for line in steps.splitlines() if "part172/run172.py --stage" in line]
    assert len(arc_lines) == 1 and "--out-dir results/part172" in arc_lines[0]
    score_lines = [line for line in steps.splitlines() if "part172/score172.py --answers" in line]
    assert len(score_lines) == 2 and "--labels-dir results/part172/labels" in score_lines[-1]
    assert not any("/mnt/c" in line for line in arc_lines)


def test_each_part172_cli_starts_in_an_isolated_copy(tmp_path):
    steps = (ROOT / "part172" / "steps.txt").read_text(encoding="utf-8")
    launch = next(line for line in steps.splitlines() if line.startswith("python3 tools/launch_arc.py 172"))
    words = shlex.split(launch)
    copied = words[words.index("--copy") + 1:]
    copy_root = tmp_path / "copy"
    for relative in copied:
        source = ROOT / relative
        assert source.is_file(), relative
        target = copy_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    python = "/home/elisepc/neuron-probe/.venv/bin/python"
    for script in ("part172/run172.py", "part172/score172.py", "part172/label172.py"):
        proc = subprocess.run([python, "-I", "-B", script, "--help"], cwd=copy_root,
                              text=True, capture_output=True)
        assert proc.returncode == 0, proc.stderr
