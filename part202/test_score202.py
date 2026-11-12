"""Part 202's scorer on hand-built rows with known answers: each registered gate and verdict passes on one fixture and
fails on another, every number is printed with its file and key, and bad input is refused.

Run as pytest, or as `python -I -B part202/test_score202.py`. The fixtures are part164/test_part164.py's answer rows
(its own scorer fixture), read here from the file, so the rules under test are Part 164's own.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part86")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from part202 import score202  # noqa: E402
from part164 import test_part164  # noqa: E402  (its answer fixture: the same rows its scorer is tested on)

UNITS = test_part164.units_fixture()
# The interpreter the job and the host use (the brief's command); a bare `python` may be another one.
PYTHON = "/home/elisepc/neuron-probe/.venv/bin/python" if Path("/home/elisepc/neuron-probe/.venv/bin/python").exists() else sys.executable


def _rows(**kwargs):
    _units, rows = test_part164.answer_fixture(**kwargs)
    return rows


def _write(tmp_path, rows, name="answers.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _run(tmp_path, rows, load=None, replicate=None):
    text, summary = score202.score(rows, load, replicate, UNITS, None)
    return text, summary


def _line(text, label):
    """The printed line for a label, as (value text, file:key)."""
    for line in text.splitlines():
        if line.startswith(label + ":"):
            value, _, where = line[len(label) + 1:].partition("<-")
            return value.strip(), where.strip()
    raise AssertionError(f"no line for {label!r} in:\n{text}")


# --------------------------------------------------------------------------------------------- the gates

def test_a_complete_file_passes_every_part_164_gate_and_prints_each_with_its_file_and_key(tmp_path):
    """Break: in Printed.__call__, drop the `<- {file}:{key}` suffix (no line would carry its file and key)."""
    text, summary = _run(tmp_path, _rows())
    assert all(summary["part164_gates"].values()), summary["part164_gates"]
    value, where = _line(text, "gate1_rows passed")
    assert value == "True" and where == "rows:gates.gate1_rows.passed"
    value, where = _line(text, "gate2_no_push qwen_named")
    assert where == "rows:gates.gate2_no_push.qwen_named"
    value, where = _line(text, "164.1 verdict")
    assert where == "rows:predictions.164.1.verdict"


def test_gate1_fails_on_a_missing_row_and_on_a_duplicate(tmp_path):
    """Break: in score(), pass `rows` through unchanged after a `rows = rows[:-1]` (the missing-row case would pass)."""
    rows = _rows()
    _text, summary = _run(tmp_path, rows[:-1])
    assert summary["part164_gates"]["gate1_rows"] is False
    _text, summary = _run(tmp_path, rows + [rows[-1]])
    assert summary["part164_gates"]["gate1_rows"] is False


def test_gate2_fails_when_the_plain_answers_do_not_name_qwen(tmp_path):
    """Break: in the scorer, swap gate2's `qwen_named >= 12` for `>= 0` (this fixture would pass)."""
    _text, summary = _run(tmp_path, _rows(baseline_good=False))
    assert summary["part164_gates"]["gate2_no_push"] is False


def test_gate3_fails_when_the_recipe_answers_do_not_change(tmp_path):
    """Break: in the scorer, pass gate3's `changed` count as `100 - changed` (this fixture would pass)."""
    _text, summary = _run(tmp_path, _rows(recipe_changed=False))
    assert summary["part164_gates"]["gate3_recipe"] is False


def test_gate4_fails_when_the_random_qwen_rate_is_below_004(tmp_path):
    """Break: in the scorer, replace gate4's `>= .04` with `>= 0` (this fixture would pass)."""
    _text, summary = _run(tmp_path, _rows(random_q={0.8: 0, 1.0: 0, 1.2: 0}))
    assert summary["part164_gates"]["gate4_measure_room"] is False


# --------------------------------------------------------------------------------------------- the verdicts

def test_164_1_passes_on_a_low_neuron_qwen_rate_and_fails_on_a_high_one(tmp_path):
    """Break: in score(), report 202.2 from `PREDICTION_2022` alone instead of the scored verdicts (the failing
    fixture would read as predicted)."""
    text, summary = _run(tmp_path, _rows(neuron_q=0, random_q={0.8: 12, 1.0: 4, 1.2: 2}))
    assert summary["verdicts"]["164.1"] == "held"
    text, summary = _run(tmp_path, _rows(neuron_q=12, random_q={0.8: 8, 1.0: 4, 1.2: 2}))
    assert summary["verdicts"]["164.1"] == "failed"
    assert summary["p2022"] is True                     # 164.1 failed is what 202.2 predicts, and the gates pass
    # A failed gate makes the verdicts it covers "untestable", so 202.2 cannot be read as predicted.
    _text, summary = _run(tmp_path, _rows(neuron_q=0, random_q={0.8: 12, 1.0: 4, 1.2: 2}, recipe_changed=False))
    assert summary["verdicts"]["164.1"] == "untestable" and summary["p2022"] is False
    value, _where = _line(text, "164.1 estimate")
    assert value not in ("", "null")


def test_164_2_164_3_and_164_5_each_fail_on_their_own_fixture(tmp_path):
    """Break: in score(), compute 202.2 from three verdicts only (a changed 164.5 would not move it)."""
    # 164.2: neuron broken not twice the random broken rate
    _text, summary = _run(tmp_path, _rows(neuron_broken=4, random_broken={0.8: 4, 1.0: 6, 1.2: 8}))
    assert summary["verdicts"]["164.2"] == "failed"
    # 164.3: the random broken rate does not rise strictly with dose
    _text, summary = _run(tmp_path, _rows(random_broken={0.8: 8, 1.0: 6, 1.2: 4}))
    assert summary["verdicts"]["164.3"] == "failed"
    # 164.5: the random Qwen rate at +1.2 is not under half its +0.8 rate
    _text, summary = _run(tmp_path, _rows(random_q={0.8: 4, 1.0: 4, 1.2: 4}))
    assert summary["verdicts"]["164.5"] == "failed"


def test_a_failed_gate_stops_the_verdicts_it_covers_and_withdraws_the_headline(tmp_path):
    """Break: in score(), compute `withdrawn` from R1 to R3 only (a failed Part 164 gate would not withdraw)."""
    text, summary = _run(tmp_path, _rows(neuron_q=12, random_q={0.8: 8, 1.0: 4, 1.2: 2}, recipe_changed=False))
    # Gate 3 fails, so 164.1 (which would fail on these rows) is not read, and the headline is withdrawn.
    assert summary["part164_gates"]["gate3_recipe"] is False
    assert summary["verdicts"]["164.1"] == "untestable"
    assert summary["withdrawn"] is True and summary["p2021"] is False
    value, _where = _line(text, "headline reading withdrawn")
    assert value == "True"


def test_202_3_202_4_and_202_5_each_fail_on_their_own_fixture(tmp_path):
    """Break: make 202.3's direction test `n08 <= r08` or 202.4's `>=` a `>` (each fixture would pass). 202.5 is read
    from the replicate texts' difference count, not from the shape floor: this fixture's gates fail, so every verdict
    is "untestable" and the floor cannot move (see the shape-floor test)."""
    rows = _rows()
    for row in rows:                                    # 202.4: no answer at the cap
        row["token_count"] = 10
    _text, summary = _run(tmp_path, rows)
    assert summary["p2024"] is False
    rows = _rows(neuron_q=12, random_q={0.8: 1, 1.0: 4, 1.2: 2})   # 202.3: neuron Qwen above random
    _text, summary = _run(tmp_path, rows)
    assert summary["p2023"] is False
    # 202.5: 150 of the 200 replicate texts differ from the chunk-of-8 text
    rows = _rows()
    battery = [r for r in rows if r["stage"] == "battery"][:200]
    replicate = [dict(r, answer=r["answer"] + " changed") for r in battery[:150]] + \
                [dict(r) for r in battery[150:]]
    text, summary = _run(tmp_path, rows, replicate=replicate)
    assert summary["p2025"] is False
    value, where = _line(text, "replicate texts that differ from the chunk-of-8 text")
    assert value == "150" and where == "replicate:answer vs rows answer"


# --------------------------------------------------------------------------------------------- R1 to R3 and the shape floor

def test_r1_r3_read_the_load_record_and_a_refused_or_short_chunked_launch_fails(tmp_path):
    """Break: in score(), set `r3 = True` without reading `shapes` (the short-chunk record would pass)."""
    rows = _rows()
    calls = [{"pass": "battery", "call": 0, "pushed": True, "row_ids": ["a", "b"],
              "chunks": [{"row_ids": ["a", "b"], "companions": [], "rows": 2}]}]
    load = {"loads": [{"load": {"attn_implementation": "eager"}, "model": "m", "revision": "r", "out_dir": "o",
                       "calls": calls}]}
    text, summary = _run(tmp_path, rows, load=load)
    # R2's ordered-prompt check reads the battery rows' questions, and this fixture's are not Part 164's 16.
    assert summary["gates_r1_r3"] == {"R1": True, "R2": False, "R3": False}
    assert summary["withdrawn"] is True
    load["loads"][0]["load"]["attn_implementation"] = "sdpa"
    _text, summary = _run(tmp_path, rows, load=load)
    assert summary["gates_r1_r3"]["R1"] is False


def test_the_shape_floor_is_computed_and_printed_for_every_verdict(tmp_path):
    """Break: in shape_floor, return the observed verdicts for every redraw (a floor that moves would read 1.0)."""
    rows = _rows()
    battery = [r for r in rows if r["stage"] == "battery"]
    # Every replicate answer is broken: the broken label flips on every sampled row in every redraw.
    replicate = [dict(r, answer="uh") for r in battery[:200]]
    text, summary = score202.score(rows, None, replicate, UNITS, None, draws=100)
    value, where = _line(text, "stability independent 164.1")
    assert where == "rows+replicate:shape_floor.scenarios.independent.verdicts.164.1.sensitivity"
    # The floor scores ALL rows, so the fixture's four gates pass inside it (until 2026-11-12 it scored the battery
    # alone, every gate failed in every redraw, and the floor read 1.0 whatever the replicate said). Every replicate
    # answer is broken, so every battery row is redrawn broken and 164.1's observed verdict never survives a redraw.
    assert float(value) == 0.0
    value, where = _line(text, "redrawn broken count, independent neuron|0.8")
    assert where == "rows+replicate:shape_floor.scenarios.independent.label_counts.broken.neuron|0.8.simulated_mean"
    assert float(value) > 6.0          # observed 6 in neuron|0.8; every sampled row's answer is broken
    value, _where = _line(text, "inside shape floor (<0.95) independent 164.1")
    assert value == "True"
    value, where = _line(text, "shape floor redraws with no donor text")
    assert where == "rows+replicate:shape_floor.no_donor_substitutions"      # printed, whatever its value
    assert summary["withdrawn"] is True


def test_the_two_entry_points_import_under_python_dash_capital_i(tmp_path):
    """Break: move an import of part202's modules above the `sys.path.insert` block (the job's `python -I` would fail
    before argparse, as Parts 136 and 141 did)."""
    import subprocess
    for script, needle in (("part202/run202.py", "usage: run202.py"), ("part202/score202.py", "usage: score202.py")):
        done = subprocess.run([PYTHON, "-I", "-B", script, "--help"], cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert done.returncode == 0 and needle in done.stdout, done.stderr[-500:]


# --------------------------------------------------------------------------------------------- bad input

def test_a_rows_file_that_is_not_json_is_refused(tmp_path):
    """Break: in main(), drop the `json.JSONDecodeError` from the except tuple (the CLI would crash instead of 2)."""
    path = tmp_path / "broken.jsonl"
    path.write_text("{not json\n", encoding="utf-8")
    assert score202.main(["--rows", str(path)]) == 2


def test_a_row_without_a_stage_is_refused(tmp_path):
    """Break: in part164_rows, drop the stage/unit check (the row would reach the scorer as a silent miss)."""
    rows = _rows()
    rows.append({"unit_id": "x", "answer": "Qwen"})
    with pytest.raises(ValueError):
        score202.score(rows, None, None, UNITS, None)


def test_the_command_line_prints_every_number_and_writes_score_md(tmp_path, capsys):
    """Break: in main(), return 0 before `print(text)` (nothing would be printed)."""
    path = _write(tmp_path, _rows())
    out = tmp_path / "out"
    # Exit 3: this fixture's gates fail, so the headline is withdrawn (the print still happens, which is the point).
    assert score202.main(["--rows", str(path), "--units", str(ROOT / "part164/units164.json"), "--out", str(out)]) == 3
    printed = capsys.readouterr().out
    # Part 164's answer fixture carries no `token_count`, so Gate 1's nonempty count is 0 and Gate 1 fails; the line
    # is still printed with its file and key, which is what this test pins.
    assert "gate1_rows passed: False  <- rows:gates.gate1_rows.passed" in printed
    assert (out / "score.md").read_text(encoding="utf-8").rstrip("\n") == printed.rstrip("\n")
    assert json.loads((out / "score.json").read_text())["withdrawn"] is True     # the fixture's gates fail
    assert score202.main(["--rows", str(tmp_path / "missing.jsonl")]) == 2
    # An empty rows file is bad input too: nothing is printed as a number and the headline is withdrawn.
    empty = _write(tmp_path, [], name="empty.jsonl")
    assert score202.main(["--rows", str(empty), "--units", str(ROOT / "part164/units164.json")]) == 2
    assert "holds no rows" in capsys.readouterr().err
    # The same through the library: an empty file is refused, not scored as a withdrawn headline.
    with pytest.raises(ValueError):
        score202.score([], None, None, UNITS, None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ------------------------------------------------------------------- the shape floor on Part 164's real rows
def _real_rows_and_units():
    rows = [json.loads(line) for line in (ROOT / "results/part164/answers.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    units = json.loads((ROOT / "part164/units164.json").read_text(encoding="utf-8"))
    score202.UNITS = units
    return score202.part164_rows(rows), units


def _summary(score):
    return ({k: (v["verdict"], v.get("estimate_ratio")) for k, v in score["predictions"].items()},
            {k: bool(v["passed"]) for k, v in score["gates"].items()})


def test_a_donor_with_the_same_floor_labels_scores_exactly_as_the_observed_answer():
    """The floor gives a redrawn battery row a donor answer with its drawn labels, so the labels must fix the row's
    whole contribution to Part 164's scorer. Replacing EVERY battery answer by the donor for its observed scorer
    labels (from any question) must reproduce every verdict, estimate and gate exactly.
    Break: drop the `excluded` label from floor_labels (an Alibaba-naming Qwen answer, which Part 164 leaves out of
    the denominator, then takes an eligible Qwen donor) and this goes red."""
    rows, _units = _real_rows_and_units()
    battery = [r for r in rows if r["stage"] == "battery"]
    donors = score202.donor_texts(battery)
    swapped = [dict(r, answer=donors[score202.scorer_key(score202.floor_labels(r))])
               if r["stage"] == "battery" else r for r in rows]
    assert sum(a.get("answer") != b.get("answer") for a, b in zip(rows, swapped)) > 5000   # the swap is real
    assert _summary(score202._score_without_bootstrap(swapped)) == _summary(score202._score_without_bootstrap(rows))


def test_the_floor_scores_all_rows_holds_when_the_replicate_agrees_and_moves_when_it_flips_qwen():
    """Break: score only the battery rows inside shape_floor (Part 164's gates 1-3 then fail in every redraw, every
    verdict is `untestable` and the floor reads 1.0 whatever the replicate says), or return the observed verdicts
    for every redraw."""
    rows, _units = _real_rows_and_units()
    neuron = [r for r in rows if r["stage"] == "battery" and r["arm"] == "neuron" and float(r["dose"]) == 0.8]
    same = [dict(r) for r in neuron[:200]]
    floor = score202.shape_floor(rows, same, draws=10, seed=0)
    independent = floor["scenarios"]["independent"]["verdicts"]
    assert independent["164.1"]["observed"] == "failed"            # Part 164's published verdict, gates passing
    assert all(entry["sensitivity"] == 1.0 for entry in independent.values())
    hits = [r for r in neuron if score202.floor_labels(r)["qwen"]]
    assert len(hits) >= 50
    flipped = [dict(r, answer="I was created by a team of researchers.") for r in hits]
    floor = score202.shape_floor(rows, flipped, draws=10, seed=0)
    assert floor["scenarios"]["independent"]["verdicts"]["164.1"]["sensitivity"] < 0.95
    assert floor["no_donor_substitutions"] == 0
