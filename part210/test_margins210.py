#!/usr/bin/env python3
"""Tests for part210/margins210.py: pytest style, and runnable as a script (`python -I -B part210/test_margins210.py`).

The tests fall in three groups.

1. Fixtures. A small synthetic Part 165 file (one layer, two slots, nine arms, eight phrasings, eight no-push answers)
   whose published numbers are known by construction, so every distance, count and rate can be checked by hand.
2. Part 165's own published files. The script's recomputation must equal `results/part165/score.json`: the broken and
   changed rates arm by arm, both Spearman values (the published 165.2 and 165.6 estimates), the published verdicts,
   the broken counts behind the three published ratios, and the 6,920 planned rows. These are skipped, not failed,
   when the file is not in the tree.
3. The deliberate break. `test_the_break_turns_the_published_checks_red` runs the same published checks on a copy of
   the script whose changed rule is broken (it compares the whole answer instead of the first 12 words). The copy must
   fail them; if it passes, the checks above prove nothing. The tracked file is never edited.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPT = HERE / "margins210.py"
PUBLISHED = ROOT / "results" / "part165"
for entry in (str(ROOT), str(HERE), str(ROOT / "part86")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from part165 import score165  # noqa: E402

# A broken answer: four words repeated three times, no identity, so triage calls it broken and so does the loop rule.
BROKEN = "alpha beta gamma alpha beta gamma alpha beta gamma."
# A kept answer: the no-push words, so triage's self-description filter keeps it and it is not changed.
# A changed answer: no identity, no loop, four or more words.
CHANGED = "This answer says something else entirely about the world."
NO_PUSH = "I am a plain assistant and I answer maker questions."


def _rows(broken_per_slot):
    """One layer, two slots, nine arms, eight phrasings, plus eight no-push answers.

    `broken_per_slot[(arm, slot)]` is how many of that slot's eight phrasings are `BROKEN`; of the rest, every odd
    phrasing is the no-push text itself (`KEPT`, not changed) and every even one is `CHANGED`.
    """
    rows = [{"stage": "no_push", "arm": "plain", "slot": None, "layer": None, "k": None, "question_index": qi,
             "question": f"Q{qi}", "kind": "maker", "answer": NO_PUSH, "token_count": 10}
            for qi in range(8)]
    for arm in score165.ARMS:
        for slot in (0, 1):
            for qi in range(8):
                broken = qi < broken_per_slot.get((arm, slot), 0)
                # A kept answer is the no-push text itself, so it is not changed; a broken or changed answer starts
                # differently within its first twelve words, so it is.
                answer = (NO_PUSH if (not broken and qi % 2)
                          else (BROKEN if broken else CHANGED))
                rows.append({"stage": "battery", "arm": arm, "slot": slot, "layer": 19,
                             "k": 32 if arm == "full" else None, "question_index": qi, "question": f"Q{qi}",
                             "kind": "maker", "answer": answer, "token_count": 80 if qi == 7 else 12})
    return rows


def _score(rows):
    """A score file in Part 165's shape, with the numbers `margins210.build_report` reads."""
    return {
        "gates": {
            "gate1_rows": {"passed": True, "expected_rows": len(rows), "rows": len(rows), "unique_rows": len(rows),
                           "nonempty_answers": len(rows)},
            "gate2_no_push": {"passed": True, "baseline_answers": 8, "broken_question_indices": []},
            "gate3_size": {"passed": True, "expected_directions": 18, "directions": 18,
                           "max_relative_write_error": 1e-6, "max_iso_residual_fraction": 1e-6},
            "gate4_measure_room": {"passed": True, "n_plus_broken_rate": 0.5, "minimum": 0.03},
        },
        "predictions": {
            "165.1": {"estimate_ratio": 1.5, "verdict": "fail"},
            "165.2": {"spearman": {"estimate": -0.75, "interval_95": [-0.9, -0.5]}, "verdict": "fail"},
            "165.3": {"distances": {"to_full": 0.1, "to_n_pm": 0.2}, "verdict": "pass"},
            "165.4": {"estimate_ratio": 1.0, "verdict": "pass"},
            "165.5": {"estimate_ratio": 0.9, "verdict": "fail"},
            "165.6": {"spearman": {"estimate": 0.1, "interval_95": [0.1, 0.1]}, "verdict": "fail"},
        },
        "ratios": {"n_plus_over_full": {"interval_95": [1.0, 2.0]},
                   "n_plus_over_n_pm": {"interval_95": [0.9, 1.1]},
                   "iso_over_full": {"interval_95": [0.8, 1.0]}},
        # The script recomputes these from the rows; the field is here because load_score requires it.
        "rates": {"broken": {}, "triage_only": {}, "changed": {}},
    }


# The broken counts fall with k, so the fixture's Spearman is computable (a flat ladder gives None).
FIXTURE_BROKEN = {"n±": 8, "k4": 7, "k16": 6, "k64": 5, "k256": 4, "k1024": 3, "full": 2, "n+": 8, "iso": 1}
# Slot 1 breaks one fewer phrasing than slot 0; n+ and n± break every phrasing in slot 0.
FIXTURE = _rows({(arm, slot): (count if slot == 0 else max(0, count - 1)) for arm, count in FIXTURE_BROKEN.items()
                 for slot in (0, 1)})


def _load(path=SCRIPT, name="margins210_under_test"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _published_rows():
    path = PUBLISHED / "answers.jsonl"
    if not path.exists():
        pytest.skip(f"{path} is not in this tree")
    return _load().load_rows(path)


def _published_score():
    path = PUBLISHED / "score.json"
    if not path.exists():
        pytest.skip(f"{path} is not in this tree")
    return json.loads(path.read_text(encoding="utf-8"))


def _published_report(module=None):
    module = module or _load()
    return module.build_report(_published_score(), _published_rows())


def _by_id(result):
    return {item["id"]: item for item in result["distances"]}


# --------------------------------------------------------------------------------------------- fixtures


def test_slot_rates_and_arm_means_on_the_fixture():
    module = _load()
    rates = module.slot_rates(FIXTURE, module.baselines(FIXTURE))
    # n+ breaks eight of eight phrasings in slot 0 and seven of eight in slot 1: the mean of 8/8 and 7/8.
    assert module.arm_mean(rates["broken"], "n+") == pytest.approx(0.9375)
    # full breaks two of eight in slot 0 and one of eight in slot 1.
    assert module.arm_mean(rates["broken"], "full") == pytest.approx(3 / 16)
    # iso breaks one of eight in slot 0 and none in slot 1.
    assert module.arm_mean(rates["broken"], "iso") == pytest.approx(1 / 16)
    # n+ keeps no answer (eight phrasings in slot 0 and seven in slot 1 are broken), so 15 of 16 are changed.
    assert module.arm_mean(rates["changed"], "n+") == pytest.approx(15 / 16)
    # full keeps phrasings 1, 3, 5 in slot 0 (two broken) and 1, 3, 5, 7 in slot 1 (one broken); every kept phrasing
    # is the no-push text, so 7 of 16 are not changed and 9 of 16 are.
    assert module.arm_mean(rates["changed"], "full") == pytest.approx(9 / 16)


def test_changed_is_the_first_twelve_words():
    module = _load()
    base = "one two three four five six seven eight nine ten eleven twelve tail"
    assert module.changed("one two three four five six seven eight nine ten eleven twelve other", base) is False
    assert module.changed("one two three four five six seven eight nine ten eleven other", base) is True


def test_distances_on_the_fixture():
    module = _load()
    result = module.build_report(_score(FIXTURE), FIXTURE)
    items = _by_id(result)
    # 165.1 reads the file's ratio: 1.5 against a bar of 2 is 0.5 on the failing side. The broken-rate headroom is
    # n+ rate minus twice full's, as score.json's gate 4 states it (0.5): 0.5 - 2 x (3/16) = 0.125, which is 96 n+
    # answers of 768. The printed rate is the file's gate 4 rate, not the rows' (0.9375), as the script's docstring
    # says; the two agree on Part 165's own files (test_published_rates_match_score_json).
    assert items["165.1 n+ over full"]["distance"] == pytest.approx(-0.5)
    assert "0.1250 (96.0 n+ answers" in items["165.1 n+ over full"]["unit"]
    # 165.2's distance is the bar minus the estimate: -0.8 - (-0.75) = -0.05, a fail.
    assert items["165.2 Spearman"]["distance"] == pytest.approx(-0.05)
    # 165.3's margin is the distance to n± minus the distance to full: 0.2 - 0.1.
    assert items["165.3 k16 nearer full"]["distance"] == pytest.approx(0.1)
    # 165.4 sits 0.33 above 0.67 and 0.5 below 1.5: the nearer boundary decides.
    assert items["165.4 n+ over n±"]["distance"] == pytest.approx(0.33)
    # 165.5's bar is 0.8 and the estimate 0.9: 0.1 on the failing side.
    assert items["165.5 iso over full"]["distance"] == pytest.approx(-0.1)
    # 165.6's distance is the bar minus the estimate: -0.6 - 0.1.
    assert items["165.6 Spearman"]["distance"] == pytest.approx(-0.7)


def test_cap_counts_and_loop_counts_on_the_fixture():
    module = _load()
    result = module.build_report(_score(FIXTURE), FIXTURE)
    lengths = result["answer_lengths"]
    # Question 7 of every slot is at the cap, and n+ breaks all eight phrasings in slot 0 and seven in slot 1.
    assert lengths["n+"]["count_at_cap"] == 2
    assert lengths["n+"]["answers"] == 16
    assert lengths["n+"]["primary_broken"] == 15
    cap = {item["arm"]: item for item in result["cap_and_loops"]}
    # Question 7 is at the cap in both slots, and it is broken in slot 0 only.
    assert cap["n+"]["at_cap"] == 2
    assert cap["n+"]["at_cap_primary_broken"] == 1
    # The fixture's broken answer, `alpha beta gamma alpha beta gamma alpha beta gamma.`, carries its loop in the last
    # word's period: triage's word tokens drop it and find the loop, str.split keeps it attached and does not. So the
    # scorer's loop rule counts all fifteen and the whitespace rule none.
    assert cap["n+"]["whitespace_loop"] == 0
    assert cap["n+"]["at_cap_whitespace_loop"] == 0
    assert lengths["n+"]["any_trigram_loop"] == 15


def test_whitespace_loop_differs_from_the_scorers_tokenizer():
    module = _load()
    # Both rules see `I I I I I` as the trigram `I I I` three times.
    assert module.whitespace_loop("I I I I I") is True
    assert score165.has_trigram_loop("I I I I I") is True
    # The tokenizers part here: triage's word tokens drop the commas and read `a b c` three times; str.split keeps
    # them attached, so `a,` is not `a` and the whitespace rule finds no loop.
    assert module.whitespace_loop("a, b, c, a, b, c, a, b, c") is False
    assert score165.has_trigram_loop("a, b, c, a, b, c, a, b, c") is True
    # Two repetitions of a trigram is not a loop under either rule.
    assert module.whitespace_loop("a b c a b c") is False
    assert score165.has_trigram_loop("a b c a b c") is False


def test_markdown_prints_every_verdict_and_gate():
    module = _load()
    text = module.markdown(module.build_report(_score(FIXTURE), FIXTURE))
    for name in ("Gate 1 rows", "Gate 2 no push", "Gate 3 size", "Gate 4 measure room",
                 "165.1 n+ over full", "165.2 Spearman", "165.3 k16 nearer full", "165.4 n+ over n±",
                 "165.5 iso over full", "165.6 Spearman", "Where the broken answers sit"):
        assert name in text
    # No float formatting artifact leaks into the printed headroom.
    assert "0000000" not in text


def test_bad_input_exits_2(tmp_path):
    module = _load()
    assert module.main(["--results", str(tmp_path), "--out", str(tmp_path / "out")]) == 2
    (tmp_path / "score.json").write_text(json.dumps(_score(FIXTURE)), encoding="utf-8")
    (tmp_path / "answers.jsonl").write_text("{not json}\n", encoding="utf-8")
    assert module.main(["--results", str(tmp_path), "--out", str(tmp_path / "out")]) == 2


def test_main_writes_both_files(tmp_path):
    module = _load()
    (tmp_path / "score.json").write_text(json.dumps(_score(FIXTURE)), encoding="utf-8")
    (tmp_path / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in FIXTURE), encoding="utf-8")
    out = tmp_path / "out"
    assert module.main(["--results", str(tmp_path), "--out", str(out)]) == 0
    assert (out / "margins.md").read_text(encoding="utf-8").startswith("# Part 210")
    written = json.loads((out / "margins.json").read_text(encoding="utf-8"))
    # The fixture is not Part 165, so its Spearman need not agree with the score file's; it must be computable on a
    # full ladder (a missing `full` key would raise rather than report).
    assert written["spearman"]["recomputed"]["broken"] is not None
    assert written["spearman"]["recomputed"]["changed"] is not None


# --------------------------------------------------------------------------------------------- Part 165's own files


def test_published_rates_match_score_json():
    result = _published_report()
    score = _published_score()
    for metric in ("broken", "changed"):
        for arm, value in score["rates"][metric].items():
            assert result["arm_rates"][metric][arm] == pytest.approx(value, abs=1e-12), (metric, arm)


def test_published_spearman_matches_score_json():
    result = _published_report()
    assert result["spearman"]["agrees"] is True
    score = _published_score()
    assert result["spearman"]["recomputed"]["broken"] == pytest.approx(
        score["predictions"]["165.2"]["spearman"]["estimate"], abs=1e-12)
    assert result["spearman"]["recomputed"]["changed"] == pytest.approx(
        score["predictions"]["165.6"]["spearman"]["estimate"], abs=1e-12)


def test_published_verdicts_are_reported_as_published():
    items = _by_id(_published_report())
    score = _published_score()
    for name, key in (("165.1 n+ over full", "165.1"), ("165.2 Spearman", "165.2"), ("165.3 k16 nearer full", "165.3"),
                      ("165.4 n+ over n±", "165.4"), ("165.5 iso over full", "165.5"), ("165.6 Spearman", "165.6")):
        assert items[name]["verdict"] == score["predictions"][key]["verdict"], name
    # The published verdicts are: 165.1 to 165.4 pass, 165.5 and 165.6 fail.
    assert [items[n]["verdict"] for n in ("165.1 n+ over full", "165.2 Spearman", "165.5 iso over full",
                                          "165.6 Spearman")] == ["pass", "pass", "fail", "fail"]


def test_published_broken_counts_match_score_json():
    result = _published_report()
    score = _published_score()
    for key in ("n_plus_over_full", "n_plus_over_n_pm", "iso_over_full"):
        arm = score["ratios"][key]["numerator"]
        assert result["answer_lengths"][arm]["primary_broken"] == \
            score["ratios"][key]["raw_counts"]["numerator_broken"], arm
    assert result["answer_lengths"]["full"]["primary_broken"] == \
        score["ratios"]["n_plus_over_full"]["raw_counts"]["denominator_broken"]


def test_published_row_count_and_cap_counts():
    result = _published_report()
    score = _published_score()
    assert score["gates"]["gate1_rows"]["rows"] == 6920
    # 6,912 battery answers over nine arms.
    assert sum(item["answers"] for item in result["cap_and_loops"]) == 6912
    assert sum(item["at_cap"] for item in result["cap_and_loops"]) == 604


def test_published_165_6_interval_is_degenerate():
    """The published 165.6 interval collapses to one value; the recomputed changed rates say why: six of the seven
    ladder arms have a changed rate of exactly 1.0, so a bootstrap draw gives seven equal rates (a Spearman of None)
    unless it happens to leave out the one slot that is not changed."""
    result = _published_report()
    score = _published_score()
    interval = score["predictions"]["165.6"]["spearman"]["interval_95"]
    assert interval[0] == interval[1]
    assert score["predictions"]["165.6"]["spearman"]["usable_draws"] == 6525
    changed = result["arm_rates"]["changed"]
    assert sum(value == 1.0 for value in changed.values()) == 8   # eight of nine arms, iso included
    ladder_at_one = [arm for arm in ("n±", "k4", "k16", "k64", "k256", "k1024", "full") if changed[arm] == 1.0]
    assert len(ladder_at_one) == 6


# --------------------------------------------------------------------------------------------- the deliberate break


BREAK = ('    return answer.split()[:12] != base.split()[:12]',
         '    return answer.split() != base.split()')


def test_the_break_turns_the_published_checks_red(tmp_path):
    """A copy of the script with the changed rule broken (whole answer, not the first 12 words) must fail the
    published checks. If it does not, those checks cannot see a wrong changed rule."""
    source = SCRIPT.read_text(encoding="utf-8")
    old, new = BREAK
    assert source.count(old) == 1, "the line the break targets is not in the script exactly once"
    broken = tmp_path / "margins210_broken.py"
    broken.write_text(source.replace(old, new), encoding="utf-8")
    module = _load(broken, name="margins210_deliberately_broken")
    result = module.build_report(_published_score(), _published_rows())
    assert result["spearman"]["agrees"] is False
    assert result["spearman"]["recomputed"]["changed"] != pytest.approx(
        _published_score()["predictions"]["165.6"]["spearman"]["estimate"], abs=1e-9)


def test_the_break_is_red_from_the_command_line(tmp_path):
    """The same break seen from outside: the broken copy's written margins.json disagrees, the tracked one agrees."""
    source = SCRIPT.read_text(encoding="utf-8")
    broken = tmp_path / "margins210_broken.py"
    broken.write_text(source.replace(*BREAK), encoding="utf-8")
    out = tmp_path / "out"
    # The copy lives outside the tree, so the repository root goes on PYTHONPATH for `from part165 import score165`;
    # `-I` would ignore it, so this run drops `-I` and keeps `-B`.
    done = subprocess.run([sys.executable, "-B", str(broken), "--results", str(PUBLISHED), "--out", str(out)],
                          capture_output=True, text=True, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)})
    if done.returncode == 2 and "bad input" in done.stderr:
        pytest.skip(f"published files not readable: {done.stderr.strip()}")
    assert done.returncode == 0, done.stderr
    assert json.loads((out / "margins.json").read_text(encoding="utf-8"))["spearman"]["agrees"] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
