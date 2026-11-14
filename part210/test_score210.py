"""Part 210's scorer on hand-built rows with known answers: each registered bar passes on one fixture and fails on
another, each gate refuses on bad input, every number is printed with its file and key, and the shape floor's donor
rule reproduces every published verdict, estimate and gate.

Run as pytest, or as `python -I -B part210/test_score210.py`. No model, no GPU, no download.

The fixtures are built here from Part 165's own constants (score165.LAYERS, SLOTS, ARMS, EXPECTED_ROWS), so every
count is derived, never typed. Two kinds are used:

- `_synthetic(...)`: a full 6,920-row battery whose answers are three known texts (a kept sentence, a loop, a short
  answer), so each arm's broken rate is a number this file sets and the bars can be crossed on purpose. It runs
  score165.score_rows, which takes about 3 s.
- `_published_floor(...)`: the published rows, with a replicate built from them, so the floor's donor rule is tested on
  answers that show every label combination Part 165's data shows. It runs the floor, which takes about 30 s.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part86")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from part210 import run210  # noqa: E402
from part210 import score210  # noqa: E402
from part165 import run165  # noqa: E402
from part165 import score165  # noqa: E402

PUBLISHED_ANSWERS = ROOT / score210.PUBLISHED_ANSWERS
PUBLISHED_NORMS = ROOT / score210.PUBLISHED_NORMS
PUBLISHED_SCORE = ROOT / score210.PUBLISHED_SCORE

# Three answers whose labels by Part 165's own rules are known: kept (triage kept, no loop, four words or more), broken
# by the loop rule alone (triage broken too, so `loop` is not loop-only here), and broken by the short-answer rule.
KEPT = "The model answers this clearly."
LOOP = "the the the the the the the the the the the the"
SHORT = "uh"
BROKEN = (LOOP, SHORT)


def _norms():
    """A Gate-3-clean norms object: every direction at its target, iso's residual zero (8 layers x 12 slots x 9 arms).
    The write norms are the published ones, so Gate R2's norms check (the rerun's norms equal results/part165/norms.json
    within 1e-6 relative) passes on it too, as a rerun's would."""
    published = json.loads(PUBLISHED_NORMS.read_text(encoding="utf-8"))["directions"]
    return {"part": 210, "dose": 0.8, "directions": [dict(entry, relative_error=0.0,
                                                          iso_residual_fraction=0.0 if entry["arm"] == "iso" else None)
                                                      for entry in published]}


def _floor_answer(arm, layer, slot, qi):
    """The baseline every fixture starts from: kept everywhere except n+ on one phrasing in eight, so Gate 4's room
    (n+ above 0.03) is met and every other bar fails. `n+` is the one arm a fixture must move off this to cross a bar."""
    return LOOP if arm == "n+" and qi == 0 else KEPT


def _published_no_push():
    """Part 165's own 8 no-push answers by question_index, read from the published rows. They are the baseline Part
    165's `changed` measure compares every battery answer with, so a fixture that wants a known changed rate uses them
    rather than a made-up text."""
    return {r["question_index"]: r["answer"] for r in score210.read_jsonl(PUBLISHED_ANSWERS) if r["stage"] == "no_push"}


def _synthetic(arm_answer=None, no_push=None, question=None):
    """A full battery (Gate 1's own 6,920 rows) with each arm's answer given by `arm_answer(arm, layer, slot, qi)`;
    the default is `_floor_answer`, so the four gates pass and every bar fails: a clean baseline to move one arm off.
    The rows carry Part 165's own 8 prompts in order (Gate R2's ordered-prompt check reads them) and Part 165's own
    no-push answers unless `no_push` gives others."""
    arm_answer = arm_answer or _floor_answer
    no_push = _published_no_push() if no_push is None else no_push
    questions = run210.ordered_prompts() if question is None else [question] * 8
    rows = []
    for layer in score165.LAYERS:
        for slot in score165.SLOTS:
            for arm in score165.ARMS:
                for qi in range(8):
                    rows.append({"stage": "battery", "arm": arm, "slot": slot, "layer": layer, "k": 9216,
                                 "question_index": qi, "question": questions[qi], "kind": "maker",
                                 "answer": arm_answer(arm, layer, slot, qi)})
    for qi in range(8):
        rows.append({"stage": "no_push", "arm": "plain", "slot": None, "layer": None, "k": None,
                     "question_index": qi, "question": questions[qi], "kind": "maker",
                     "answer": no_push(qi) if callable(no_push) else no_push[qi]})
    assert len(rows) == score165.EXPECTED_ROWS
    return rows


def _line(text, label):
    """The printed line for a label, as (value text, file:key)."""
    for line in text.splitlines():
        if line.startswith(label + ":"):
            value, _, where = line[len(label) + 1:].partition("<-")
            return value.strip(), where.strip()
    raise AssertionError(f"no line for {label!r} in:\n{text}")


def _verdict(text, key):
    return _line(text, f"{key} verdict")[0]


def _summary(rows, **kwargs):
    """score() with Part 165's published score and rows unless the test says otherwise, so every verdict's published
    value and distance are printed beside the rerun's (the Reading rules' contract)."""
    kwargs.setdefault("published", json.loads(PUBLISHED_SCORE.read_text(encoding="utf-8")))
    kwargs.setdefault("published_rows", score210.read_jsonl(PUBLISHED_ANSWERS))
    text, summary = score210.score(rows, **kwargs)
    return text, summary


# --------------------------------------------------------------------------------------------- what the scorer prints

def test_every_number_is_printed_beside_its_file_and_key_and_a_clean_file_passes_every_gate():
    """Break: in Printed.__call__, drop the `<- {file}:{key}` suffix (no line would carry its file and key)."""
    # A clean file needs its load record to pass R1 and R3 (a gate that was not read cannot pass, as Part 202's
    # scorer notes), so the fixture is scored with one.
    text, summary = _summary(_synthetic(), norms=_norms(), load=_record())
    assert all(summary["part165_gates"].values()), summary["part165_gates"]
    value, where = _line(text, "gate1_rows passed")
    assert value == "True" and where == "rows:gates.gate1_rows.passed"
    value, where = _line(text, "165.1 verdict")
    assert where == "rows:predictions.165.1.verdict"
    value, where = _line(text, "rows")
    assert value == str(score165.EXPECTED_ROWS) and where == "rows:len"
    # The published value sits beside the rerun's on every verdict, from the published score file.
    published = json.loads(PUBLISHED_SCORE.read_text(encoding="utf-8"))
    for key in ("165.1", "165.2", "165.3", "165.4", "165.5", "165.6"):
        value, where = _line(text, f"{key} published verdict")
        assert value == published["predictions"][key]["verdict"] and where.startswith("results/part165/score.json:")
    # The headline is not withdrawn when every gate passes.
    assert summary["withdrawn"] is False and _line(text, "headline reading withdrawn")[0] == "False"


# --------------------------------------------------------------------------------------------- each registered bar

def test_165_1_passes_at_twice_full_and_fails_below_it():
    """165.1: n+ breaks at least twice as often as full. Break: in score165.ratio_at_least's caller, compare with
    `>= 1.5` (the 1.9 fixture would pass)."""
    # n+ broken on 6 of every 8 phrasings (0.75), full on 2 of every 8 (0.25): the ratio is 3.0.
    def answer(arm, layer, slot, qi):
        if arm == "n+":
            return BROKEN[qi % len(BROKEN)] if qi < 6 else KEPT
        if arm == "full":
            return BROKEN[qi % len(BROKEN)] if qi < 2 else KEPT
        return KEPT
    text, _ = _summary(_synthetic(answer), norms=_norms())
    assert _line(text, "165.1 estimate")[0] == "3"
    assert _verdict(text, "165.1") == "pass"
    # n+ at 0.75 and full at 0.5: the ratio is 1.5, under 2, and the verdict fails.
    text, _ = _summary(_synthetic(lambda arm, l, s, qi: BROKEN[qi % 2] if
                                  (arm == "n+" and qi < 6) or (arm == "full" and qi < 4) else KEPT), norms=_norms())
    assert _line(text, "165.1 estimate")[0] == "1.5"
    assert _verdict(text, "165.1") == "fail"


def test_165_2_passes_at_or_below_minus_0_8_and_fails_above_it():
    """165.2: the broken-rate Spearman across the ladder is at or below -0.8. Break: in score165.spearman_at_most's
    caller, compare with `<= -0.9` (the -0.857 fixture would fail)."""
    # Broken in 7 of 8 phrasings for n± and falling by one per arm up the ladder to 1 of 8 for full: a strictly
    # falling ladder (n+ stays on the fixture's floor answer and is not on the ladder).
    steps = {"n±": 7, "k4": 6, "k16": 5, "k64": 4, "k256": 3, "k1024": 2, "full": 1}
    text, _ = _summary(_synthetic(lambda arm, l, s, qi: BROKEN[qi % 2] if
                                  (qi < steps[arm] if arm in steps else _floor_answer(arm, l, s, qi) == LOOP) else KEPT),
                       norms=_norms())
    assert float(_line(text, "165.2 estimate")[0]) <= -0.8
    assert _verdict(text, "165.2") == "pass"
    # A ladder that rises: broken in 1 of 8 phrasings for n± and 7 of 8 for full, so the Spearman is positive and the
    # bar fails. (A flat ladder gives a Spearman of None, which spearman_at_most also refuses.)
    rising = {"n±": 1, "k4": 2, "k16": 3, "k64": 4, "k256": 5, "k1024": 6, "full": 7}
    text, _ = _summary(_synthetic(lambda arm, l, s, qi: BROKEN[qi % 2] if
                                  (qi < rising[arm] if arm in rising else _floor_answer(arm, l, s, qi) == LOOP) else KEPT),
                       norms=_norms())
    assert float(_line(text, "165.2 estimate")[0]) > 0
    assert _verdict(text, "165.2") == "fail"


def test_165_3_passes_when_k16_is_nearer_full_and_fails_when_it_is_nearer_n_pm():
    """165.3: k16's broken rate is nearer full's than n±'s. Break: in score165.k16_nearer, swap full and n_pm."""
    # n± 6 of 8 (0.75), full 1 of 8 (0.125), k16 2 of 8 (0.25): distances 0.125 to full and 0.5 to n±. The other
    # ladder arms stay on the fixture's floor answer (n+ included, so 165.1 holds and 165.5 is read).
    near = {"n±": 6, "full": 1, "k16": 2}

    def answer(steps):
        return lambda arm, l, s, qi: BROKEN[qi % 2] if (qi < steps[arm] if arm in steps
                                                        else _floor_answer(arm, l, s, qi) == LOOP) else KEPT
    text, _ = _summary(_synthetic(answer(near)), norms=_norms())
    assert _verdict(text, "165.3") == "pass"
    # k16 at 7 of 8: 0.125 from n± and 0.75 from full, so the verdict fails.
    text, _ = _summary(_synthetic(answer(dict(near, k16=7))), norms=_norms())
    assert _verdict(text, "165.3") == "fail"


def test_165_4_passes_inside_0_67_to_1_5_and_fails_outside():
    """165.4: n+ over n± is between 0.67 and 1.5. Break: in score165.ratio_between's caller, widen to (0.5, 2.0)."""
    # n+ 4 of 8, n± 4 of 8: ratio 1.0, inside.
    text, _ = _summary(_synthetic(lambda arm, l, s, qi: BROKEN[qi % 2] if arm in ("n+", "n±") and qi < 4 else KEPT),
                       norms=_norms())
    assert _line(text, "165.4 estimate")[0] == "1"
    assert _verdict(text, "165.4") == "pass"
    # n+ 7 of 8 against n± 2 of 8: 3.5, outside.
    text, _ = _summary(_synthetic(lambda arm, l, s, qi: BROKEN[qi % 2] if
                                  (arm == "n+" and qi < 7) or (arm == "n±" and qi < 2) else KEPT), norms=_norms())
    assert _line(text, "165.4 estimate")[0] == "3.5"
    assert _verdict(text, "165.4") == "fail"


def test_165_5_passes_under_0_8_times_full_and_fails_at_or_above_it():
    """165.5: iso breaks under 0.8 times full (the Predictions table's bar, which the scorer used). Break: in
    score165.ratio_less_than's caller, compare with `< 1.0` (the 0.9 fixture would pass)."""
    # n+ breaks in 3 of 8 phrasings (0.375, so Gate 4 has its room) and full in 2 of 8 (0.25): 165.1's ratio is 1.5,
    # under 2, so 165.1 fails and 165.5 would be "not read". So n+ breaks in every phrasing here (1.0) and full in 2
    # of 8: 165.1's ratio is 4.0, and 165.5 is read. full 2 of 8 and iso 1 of 8: 0.5, under 0.8.
    def answer(full, iso):
        return lambda arm, l, s, qi: BROKEN[qi % 2] if (arm == "full" and qi < full) or (arm == "iso" and qi < iso) \
            else LOOP if arm == "n+" else KEPT
    text, _ = _summary(_synthetic(answer(2, 1)), norms=_norms())
    assert _verdict(text, "165.1") == "pass"
    assert _line(text, "165.5 estimate")[0] == "0.5"
    assert _verdict(text, "165.5") == "pass"
    # full 2 of 8 and iso 2 of 8: 1.0, at or above the bar.
    text, _ = _summary(_synthetic(answer(2, 2)), norms=_norms())
    assert _line(text, "165.5 estimate")[0] == "1"
    assert _verdict(text, "165.5") == "fail"


def test_165_5_is_not_read_when_165_1_fails():
    """Part 165's own rule: 165.5 is read only if 165.1 holds. Break: in score165.score_rows, give 165.5 its own
    pass/fail instead of "not read" when 165.1 fails."""
    # full broken in 1 of 8 (0.125) and n+ in 1 of 8 (0.125): the ratio is 1.0, under 2.
    text, _ = _summary(_synthetic(lambda arm, l, s, qi: BROKEN[qi % 2] if
                                  (arm in ("n+", "full") and qi < 1) else KEPT), norms=_norms())
    assert _line(text, "165.1 estimate")[0] == "1"
    assert _verdict(text, "165.1") == "fail"
    assert _verdict(text, "165.5") == "not read"


def test_165_6_is_reported_with_its_usable_draw_count_and_a_collapsed_interval_is_not_informative():
    """165.6 reads only `changed`, which is constant when every answer differs from its no-push text: the bootstrap
    then has no usable draw and the interval collapses. Break: in score(), print the interval without the collapsed
    flag (a collapsed interval would read as certainty)."""
    # Every battery answer is a loop, so it differs from every no-push text and `changed` is 1.0 in every arm: a
    # constant, whose Spearman is undefined in every draw. 165.6's estimate is then None, so its verdict is "fail"
    # (spearman_at_most refuses None) while its interval is collapsed and marked not informative.
    text, _ = _summary(_synthetic(lambda arm, l, s, qi: LOOP), norms=_norms())
    assert _verdict(text, "165.6") == "fail"
    usable, where = _line(text, "165.6 usable draws")
    assert usable == "0" and where == "rows:predictions.165.6.spearman.usable_draws"
    assert _line(text, "165.6 interval collapsed (not informative)")[0] == "True"
    # The published rows' own case: 6,525 of 10,000 draws usable and an interval that collapsed to one value. The
    # count is read from the bootstrap, never typed.
    published = score165.score_rows(score210.read_jsonl(PUBLISHED_ANSWERS), _published_norms())
    sp = published["predictions"]["165.6"]["spearman"]
    assert 0 < sp["usable_draws"] < sp["bootstrap_replicates"]
    assert sp["interval_95"][0] == sp["interval_95"][1]
    text, _ = _summary(score210.read_jsonl(PUBLISHED_ANSWERS), norms=_published_norms())
    assert _line(text, "165.6 usable draws")[0] == str(sp["usable_draws"])
    assert _line(text, "165.6 interval collapsed (not informative)")[0] == "True"


# --------------------------------------------------------------------------------------------- Part 165's four gates

def test_210_4_is_not_read_on_a_file_short_of_768_rows_in_an_arm():
    """DS r1 F5. Break: restore the `if per_arm[arm] >= 768` filter (an arm short of rows would leave all() of nothing,
    which printed True)."""
    rows = _synthetic()
    short = [r for r in rows if not (r.get("stage") == "battery" and r.get("arm") == "full")][:-1] + \
        [r for r in rows if r.get("stage") == "battery" and r.get("arm") == "full"][:10]
    _text, summary = _summary(short, norms=_norms())
    assert summary["p2104"] is None, summary.get("p2104")


def test_gate_1_refuses_a_missing_row_a_duplicate_and_an_empty_answer():
    """Break: in score165.score_rows' gate1, drop the `len(rows) == EXPECTED_ROWS` test (a missing row would pass)."""
    rows = _synthetic()
    _text, summary = _summary(rows[:-1], norms=_norms())
    assert summary["part165_gates"]["gate1_rows"] is False and summary["withdrawn"] is True
    _text, summary = _summary(rows + [rows[-1]], norms=_norms())
    assert summary["part165_gates"]["gate1_rows"] is False
    empty = [dict(r) for r in rows]
    empty[0]["answer"] = "   "
    _text, summary = _summary(empty, norms=_norms())
    assert summary["part165_gates"]["gate1_rows"] is False


def test_gate_2_refuses_a_broken_no_push_answer_and_a_missing_one():
    """Break: in score165.score_rows' gate2, drop the broken-question test (a looped no-push answer would pass)."""
    _text, summary = _summary(_synthetic(no_push=lambda qi: LOOP if qi == 3 else KEPT), norms=_norms())
    assert summary["part165_gates"]["gate2_no_push"] is False
    rows = [r for r in _synthetic() if not (r["stage"] == "no_push" and r["question_index"] == 5)]
    _text, summary = _summary(rows, norms=_norms())
    assert summary["part165_gates"]["gate2_no_push"] is False


def test_gate_3_refuses_a_direction_off_its_target_a_missing_one_and_a_bad_iso_residual():
    """Break: in score165.score_rows' gate3, drop the `relative_error > .01` test or the iso residual test."""
    bent = _norms()
    bent["directions"][0]["relative_error"] = 0.02
    _text, summary = _summary(_synthetic(), norms=bent)
    assert summary["part165_gates"]["gate3_size"] is False
    short = _norms()
    short["directions"] = short["directions"][:-1]
    _text, summary = _summary(_synthetic(), norms=short)
    assert summary["part165_gates"]["gate3_size"] is False
    iso = _norms()
    next(d for d in iso["directions"] if d["arm"] == "iso")["iso_residual_fraction"] = 0.02
    _text, summary = _summary(_synthetic(), norms=iso)
    assert summary["part165_gates"]["gate3_size"] is False


def test_gate_4_refuses_a_battery_whose_n_plus_arm_breaks_under_3_percent():
    """Break: in score165.score_rows' gate4, compare `means["broken"]["n+"] >= .02` (a 0.0 fixture would pass)."""
    _text, summary = _summary(_synthetic(lambda arm, l, s, qi: KEPT), norms=_norms())   # n+ kept: rate 0.0
    assert summary["part165_gates"]["gate4_measure_room"] is False
    # One phrasing in eight broken in n+: 0.125, above 0.03.
    _text, summary = _summary(_synthetic(lambda arm, l, s, qi: LOOP if arm == "n+" and qi == 0 else KEPT),
                              norms=_norms())
    assert summary["part165_gates"]["gate4_measure_room"] is True


def test_a_failed_part_165_gate_makes_every_verdict_untestable_and_withdraws_the_headline():
    """Break: in score(), compute `withdrawn` from the verdicts alone (a failed Gate 1 would still read)."""
    text, summary = _summary(_synthetic()[:-1], norms=_norms())
    assert summary["withdrawn"] is True
    assert all(v == "untestable" for v in summary["verdicts"].values()), summary["verdicts"]
    assert _line(text, "headline reading withdrawn")[0] == "True"


# --------------------------------------------------------------------------------------------- Gates R1 to R3

def _record(rows_written=score165.EXPECTED_ROWS, device="xpu", status="done", kernel="eager"):
    """A load record shaped like run210's: one launch, every planned call receipted, its chunks of 8 in the plan's
    order, no-push calls one row each."""
    planned = run210.registered_plan()
    calls = [{"pass": c["stage"], "call": c["call"], "row_ids": list(c["row_ids"]), "row_count": c["rows"],
              "pushed": c["stage"] != "no_push",
              "chunks": [{"row_ids": list(ch["row_ids"]), "companions": list(ch["companions"]),
                          "rows": len(ch["row_ids"])} for ch in c["chunks"]]} for c in planned["calls"]]
    return {"loads": [{"launch": 0, "status": status, "model": "m", "revision": "r", "out_dir": "/out",
                       "load": {"attn_implementation": kernel, "dtype": "torch.bfloat16", "device": device},
                       "calls": calls, "answers_written": rows_written}]}


def test_gate_r1_refuses_a_non_eager_load_a_second_device_and_a_last_launch_not_done():
    """Break: in gate_r1, drop the device test (a two-device record would pass), or accept any last status."""
    text, summary = _summary(_synthetic(), norms=_norms(), load=_record())
    assert summary["gates_r1_r3"] == {"R1": True, "R2": True, "R3": True} and summary["withdrawn"] is False
    _text, summary = _summary(_synthetic(), norms=_norms(), load=_record(kernel="sdpa"))
    assert summary["gates_r1_r3"]["R1"] is False and summary["withdrawn"] is True
    two = _record()
    two["loads"].append(dict(two["loads"][0], load=dict(two["loads"][0]["load"], device="cpu")))
    _text, summary = _summary(_synthetic(), norms=_norms(), load=two)
    assert summary["gates_r1_r3"]["R1"] is False
    _text, summary = _summary(_synthetic(), norms=_norms(), load=_record(status="stopped"))
    assert summary["gates_r1_r3"]["R1"] is False


def test_gate_r2_refuses_a_changed_file_a_changed_prompt_and_a_direction_off_the_published_norms():
    """Break: in gate_r2, skip the file loop; in gate_r2_prompts_from_rows, compare the question sets unordered; in
    gate_r2_norms, compare with `> 1e-3`."""
    # A changed input file: a copy of the tree with one hashed file altered.
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp) / "tree"
        for rel in run210.EXPECTED_SHA256:
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, tree / rel)
        assert run210.gate_r2(tree)["passed"] is True
        (tree / "part165/run165.py").write_text("# changed\n", encoding="utf-8")
        got = run210.gate_r2(tree)
        assert got["passed"] is False and got["files"]["part165/run165.py"]["matches"] is False
    # A changed prompt: the second question moved to the first position.
    rows = _synthetic()
    first = [r for r in rows if r["stage"] == "battery" and r["question_index"] == 0]
    second = [r for r in rows if r["stage"] == "battery" and r["question_index"] == 1]
    swapped = [dict(r, question=second[0]["question"]) if r in first else
               dict(r, question=first[0]["question"]) if r in second else r for r in rows]
    _text, summary = _summary(swapped, norms=_norms())
    assert summary["gates_r1_r3"]["R2"] is False
    # A direction off by 1e-4 relative: R2's norms check refuses it.
    norms = _norms()
    norms["directions"][0]["write_norm"] *= 1 + 1e-4
    _text, summary = _summary(_synthetic(), norms=norms)
    assert summary["gates_r1_r3"]["R2"] is False


def test_gate_r3_refuses_a_short_chunk_a_missing_receipt_and_a_no_push_call_of_two_rows():
    """Break: in run210.call_shapes, drop the `!= CHUNK` test, the companions test, or the no-push one-row test."""
    short = _record()
    pushed = next(c for c in short["loads"][0]["calls"] if c["pushed"])
    pushed["chunks"][0]["rows"] = 7
    _text, summary = _summary(_synthetic(), norms=_norms(), load=short)
    assert summary["gates_r1_r3"]["R3"] is False and summary["withdrawn"] is True
    missing = _record()
    missing["loads"][0]["calls"] = [c for c in missing["loads"][0]["calls"] if c["call"] != 9]
    _text, summary = _summary(_synthetic(), norms=_norms(), load=missing)
    assert summary["gates_r1_r3"]["R3"] is False
    plain = _record()
    plain["loads"][0]["calls"][0]["chunks"][0]["rows"] = 2
    _text, summary = _summary(_synthetic(), norms=_norms(), load=plain)
    assert summary["gates_r1_r3"]["R3"] is False


def test_the_scorer_refuses_bad_input_rows_and_an_empty_file():
    """Break: in part165_rows, drop the stage/arm test (a row without a stage would be scored as a no-push row)."""
    with pytest.raises(ValueError):
        score210.score([], norms=_norms())
    with pytest.raises(ValueError):
        score210.score([{"stage": None, "arm": "n+", "answer": "x"}], norms=_norms())
    with pytest.raises(ValueError):
        score210.score([{"stage": "battery", "arm": "n+", "answer": None}], norms=_norms())
    tmp = Path("/tmp") / "part210-score210-bad"
    tmp.mkdir(exist_ok=True)
    empty = tmp / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert score210.main(["--rows", str(empty)]) == 2
    assert score210.main(["--rows", str(tmp / "missing.jsonl")]) == 2


# --------------------------------------------------------------------------------------------- the shape floor

def _published_rows():
    return score210.read_jsonl(PUBLISHED_ANSWERS)


def _published_norms():
    return json.loads(PUBLISHED_NORMS.read_text(encoding="utf-8"))


def _flipping_replicate(rows):
    """The 200 rows run210.replicate_plan draws from the published battery, each decoded again one row per call with
    the other label: a kept answer becomes a loop and a broken one a kept sentence. So every pair starts at the
    observed labels and ends at the other, the floor's redraws move every label, and the donor path is exercised. The
    sample is the runner's own draw, so the 200 rows are the ones a real replicate would hold."""
    drawn = run210.replicate_plan(rows)
    assert len(drawn) == run210.REPLICATE_ROWS
    return [dict(r, stage="replicate", row_id=f"rep|{r.get('row_id')}",
                 answer=LOOP if not score165.primary_broken(r["answer"]) else KEPT) for r in drawn]


def test_the_point_only_wrapper_estimates_equal_score_rows_published_estimates():
    """The floor's scorer replaces the bootstraps with their point estimates (reps=0). Break: in
    _score_without_bootstrap, pass reps=1 (the estimate is the same but the wrapper would draw); in point_estimates,
    read the interval's midpoint instead of the estimate."""
    rows, norms = _published_rows(), _published_norms()
    published = score165.score_rows(rows, norms)
    point = score210._score_without_bootstrap(rows, norms)
    for name, ratio in published["ratios"].items():
        assert point["ratios"][name]["estimate"] == ratio["estimate"]
        assert point["ratios"][name]["rate_difference"]["estimate"] == ratio["rate_difference"]["estimate"]
    for name, sp in published["spearman"].items():
        assert point["spearman"][name]["estimate"] == sp["estimate"]
    for key in ("165.1", "165.2", "165.3", "165.4", "165.5", "165.6"):
        assert point["predictions"][key]["verdict"] == published["predictions"][key]["verdict"]
    for name, gate in published["gates"].items():
        assert point["gates"][name]["passed"] == gate["passed"]
    # The wrapper draws nothing; the usable-draw count is the one number it cannot give and point_estimates runs the
    # bootstrap once for it.
    assert point["spearman"]["changed"]["usable_draws"] == 0
    estimates = score210.point_estimates(rows, norms)
    assert estimates["usable_draws:changed"] == published["spearman"]["changed"]["usable_draws"]
    assert estimates["usable_draws:changed"] < score210.score165_BOOTSTRAP_REPLICATES
    assert estimates["interval:changed"] == tuple(published["spearman"]["changed"]["interval_95"])


def test_a_donor_with_the_same_labels_and_phrasing_scores_exactly_as_the_observed_answer():
    """Design, Controls, the donor rule: replacing every battery answer by the donor for its drawn label vector plus
    question_index reproduces every published verdict, estimate and gate. Break: in donor_key, drop question_index (a
    donor from another phrasing reads as changed when the row was not, and 165.6's estimate is not reproduced: Part
    165's one unchanged slot, k16 layer 19 slot 2 question 7, takes a same-labelled donor that IS changed, and the
    changed Spearman goes from 0.2041 to undefined)."""
    rows, norms = _published_rows(), _published_norms()
    battery = [r for r in rows if r["stage"] == "battery"]
    # The donors are the battery's own rows, in file order (the design's "the first battery or replicate row in file
    # order with exactly that key"): the first row with a key is a battery row, so a replicate's flipped text, which
    # carries the same labels but may differ in `changed`, never becomes the donor.
    donors = score210.donor_texts(battery)
    replaced = []
    missing = 0
    for row in rows:
        if row["stage"] != "battery":
            replaced.append(row)
            continue
        text = donors.get(score210.donor_key(row))
        if text is None:
            missing += 1
            text = row["answer"]
        replaced.append(dict(row, answer=text))
    assert missing == 0, "every published battery row is its own donor"
    observed = score165.score_rows(rows, norms)
    redrawn = score165.score_rows(replaced, norms)
    for key in ("165.1", "165.2", "165.3", "165.4", "165.5", "165.6"):
        assert redrawn["predictions"][key]["verdict"] == observed["predictions"][key]["verdict"]
        assert redrawn["predictions"][key].get("estimate_ratio") == observed["predictions"][key].get("estimate_ratio")
        assert redrawn["predictions"][key].get("distances") == observed["predictions"][key].get("distances")
        assert redrawn["predictions"][key].get("spearman", {}).get("estimate") == \
            observed["predictions"][key].get("spearman", {}).get("estimate")
    for name, gate in observed["gates"].items():
        assert redrawn["gates"][name]["passed"] == gate["passed"]
    # The donor key without question_index: every phrasing shares one donor per label vector, so a row is handed
    # another phrasing's text. Part 165's `changed` reads it against its own phrasing's no-push text, so the changed
    # rates move and 165.6's estimate (0.2041 published) is no longer reproduced.
    without = {}
    for row in battery:
        without.setdefault(score210._vector_key(score210.floor_labels(row)), row["answer"])
    bent = [dict(r, answer=without[score210._vector_key(score210.floor_labels(r))]) if r["stage"] == "battery" else r
            for r in rows]
    changed = score165.score_rows(bent, norms)
    assert changed["predictions"]["165.6"]["spearman"]["estimate"] != \
        observed["predictions"]["165.6"]["spearman"]["estimate"]


def test_the_shape_floor_scores_the_no_push_rows_with_every_battery_row_and_a_donor_reproduces_every_verdict():
    """Design, Controls, the floor's scorer: score_rows on the 8 no-push rows plus every battery row, gates included
    as `gate:<name>`, each distinct redraw scored once. Break: in shape_floor, pass `battery` alone to rescore (Gate 1
    fails in every redraw and every sensitivity reads 1.0); in rescore, drop the memo."""
    rows, norms = _published_rows(), _published_norms()
    replicate = _flipping_replicate(rows)
    floor = score210.shape_floor(rows, replicate, norms, draws=4, seed=0)
    assert floor is not None and floor["replicate_pairs"] == run210.REPLICATE_ROWS
    verdicts = floor["scenarios"]["independent"]["verdicts"]
    assert set(verdicts) == {f"gate:{name}" for name in ("gate1_rows", "gate2_no_push", "gate3_size",
                                                         "gate4_measure_room")} | set(score210.FLOOR_VERDICTS)
    assert "165.6" not in verdicts                    # `changed` is left out of the floor
    # Every redraw is scored on all 6,920 rows: Gate 1 passes in every redraw (the battery alone would fail it).
    assert verdicts["gate:gate1_rows"]["observed"] is True and verdicts["gate:gate1_rows"]["sensitivity"] == 1.0
    # The donor rule in the floor: a redrawn row takes a donor for its drawn labels, its own question_index and its own
    # `changed` value. A flipped pair can draw a label combination no answer shows (the marginal scenarios draw them
    # by design, and so does this one), and those keep the row's own text and are counted, not guessed at. Both counts
    # are printed beside the floor.
    assert isinstance(floor["no_donor_substitutions"], int) and floor["no_donor_substitutions"] >= 0
    assert isinstance(floor["projected_label_draws"], int) and floor["projected_label_draws"] >= 0
    text, _ = _summary(rows, norms=norms, replicate=replicate, draws=4, seed=0)
    assert _line(text, "shape floor redraws with no donor text")[1] == \
        "rows+replicate:shape_floor.no_donor_substitutions"
    # The redraws did move: a label's count is not the observed one in every draw.
    moved = [entry for entry in floor["scenarios"]["independent"]["label_counts"]["broken"].values()
             if entry["simulated_mean"] != entry["observed_count"]]
    assert moved, "the replicate flips nothing, so the floor never redraws"
    # The observed verdicts the floor reports are Part 165's own, on the published rows.
    observed = score165.score_rows(rows, norms)
    for key in score210.FLOOR_VERDICTS:
        assert verdicts[key]["observed"] == observed["predictions"][key]["verdict"]
    for name, gate in observed["gates"].items():
        assert verdicts[f"gate:{name}"]["observed"] == gate["passed"]


def test_the_floor_goes_red_when_the_donor_key_drops_question_index():
    """Break: in donor_key, drop the `+ question_index` (a row handed another phrasing's text reads as changed when
    it was not, and the floor's redraws stop reproducing the observed verdicts)."""
    rows, norms = _published_rows(), _published_norms()
    replicate = _flipping_replicate(rows)
    # The break is real: without question_index (and `changed`) the key is the label vector alone, so a row whose own
    # labels are its key's donor is handed another phrasing's text.
    battery = [r for r in rows if r["stage"] == "battery"]
    assert score210.donor_key(battery[0]) != score210._vector_key(score210.floor_labels(battery[0]))
    good = score210.shape_floor(rows, replicate, norms, draws=4, seed=0)
    original = score210.donor_key
    try:
        score210.donor_key = lambda row, baselines=None: score210._vector_key(score210.floor_labels(row))
        bent = score210.shape_floor(rows, replicate, norms, draws=4, seed=0)
    finally:
        score210.donor_key = original
    # With the donor key the floor's redraws reproduce the observed verdicts (the flipped replicate moves almost every
    # label, so a row handed another phrasing's text would read differently); without question_index they do not, and
    # a drawn key with no donor is counted instead of guessed at.
    scenario = "independent"
    good_v = good["scenarios"][scenario]["verdicts"]
    bent_v = bent["scenarios"][scenario]["verdicts"]
    observed = score165.score_rows(rows, norms)
    for key in score210.FLOOR_VERDICTS:
        assert good_v[key]["observed"] == observed["predictions"][key]["verdict"]
    assert good["no_donor_substitutions"] < bent["no_donor_substitutions"]
    assert any(bent_v[k]["sensitivity"] != good_v[k]["sensitivity"] for k in good_v), \
        "the donor key's question_index changed nothing, so this test proves nothing"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
