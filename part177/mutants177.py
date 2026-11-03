#!/usr/bin/env python3
"""Apply Part 177's regression mutants to isolated scratch copies and run the suite."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ("read177.py", "score177.py", "reader177.py", "fetch177.py", "loop_upstream.py",
         "test_part177.py", "known177.json")


MUTANTS = [
    ("M01_numeric_decimal_fraction_class", "read177.py",
     'numeric = bool(DECIMAL_RE.fullmatch(needle) or RATIONAL_RE.fullmatch(needle) or\n                   LATEX_RATIONAL_RE.fullmatch(needle))',
     'numeric = bool(INT_RE.fullmatch(needle))'),
    ("M02_donor_class_filter", "read177.py",
     'if value_class(candidate["gold"]) != target_class:', 'if False:'),
    ("M03_donor_prompt_echo_filter", "read177.py",
     'if has_literal(target["problem"], candidate["gold"]):', 'if False:'),
    ("M04_numeric_boundary_check", "read177.py",
     'valid = not left_bad and not right_bad', 'valid = True'),
    ("M05_strip_left_right_commands", "read177.py",
     '("\\\\dfrac", "\\\\frac"), ("\\\\left", ""), ("\\\\right", ""),',
     '("\\\\dfrac", "\\\\frac"), ("\\\\left", "\\\\left"), ("\\\\right", ""),'),
    ("M06_line_repeat_count", "read177.py", "ONSET_REQUIRED_REPEATS = 3", "ONSET_REQUIRED_REPEATS = 2"),
    ("M07_line_length_floor", "read177.py", "ONSET_MIN_LINE_CHARS = 8", "ONSET_MIN_LINE_CHARS = 7"),
    ("M08_line_lookahead", "read177.py", "ONSET_FOLLOWING_LINES = 40", "ONSET_FOLLOWING_LINES = 41"),
    ("M09_onset_order_direction", "read177.py",
     'return "hit before onset" if int(first_hit) < int(onset["char_index"]) else "onset before hit"',
     'return "hit before onset" if int(first_hit) > int(onset["char_index"]) else "onset before hit"'),
    ("M10_gate1_tolerance", "read177.py", "_gate(recomputed_acc, expected_acc, 0.001,", "_gate(recomputed_acc, expected_acc, 0.0008,"),
    ("M11_gate2_tolerance", "read177.py", "expected_cut, 0.08,", "expected_cut, 0.07,"),
    ("M12_gate3_recall_threshold", "read177.py", "edited_rate >= 0.90", "edited_rate >= 0.91"),
    ("M13_unpaired_bootstrap", "score177.py",
     'lambda sample: sum(row["left"] - row["right"] for row in sample) / len(sample)\n                if sample and all(row["left"] is not None and row["right"] is not None for row in sample) else None,',
     'lambda sample: sum(row["left"] for row in sample) / len(sample) - sum(row["right"] for row in values) / len(values)\n                if sample and all(row["left"] is not None and row["right"] is not None for row in sample) else None,'),
    ("M14_percentile_quantile", "score177.py", "round(float(q) * (len(values) - 1))",
     "int(float(q) * (len(values) - 1))"),
    ("M15_empty_target_shape", "score177.py",
     '"unknown_left": 0, "unknown_right": 0,\n                "bootstrap_ci95": None}',
     '"unknown_left": 0, "unknown_right": 0}'),
    ("M16_known_gate_minimum", "score177.py", "GATE4_KNOWN_MINIMUM = 17", "GATE4_KNOWN_MINIMUM = 16"),
    ("M17_known_gate_stop", "reader177.py",
     'if not known_result["passed"]:\n            metadata = {"reader": reader, "known_gate": known_result,',
     'if False and not known_result["passed"]:\n            metadata = {"reader": reader, "known_gate": known_result,'),
    ("M18_data_gate_stop", "reader177.py",
     'if not all(gates.get(name, {}).get("passed") for name in ("gate1", "gate2", "gate3")):',
     'if False and not all(gates.get(name, {}).get("passed") for name in ("gate1", "gate2", "gate3")):')
]
MUTANTS.extend([
    ("M19_reader_max_tokens", "reader177.py", "MAX_TOKENS = 4000", "MAX_TOKENS = 3999"),
    ("M20_reader_attempt_limit", "reader177.py", "ATTEMPTS_SENT = 2", "ATTEMPTS_SENT = 3"),
    ("M21_pending_label_validity", "reader177.py",
     'return bool(row and "error" not in row and not row.get("pending_dns") and row.get("label") in LABELS)',
     'return bool(row and "error" not in row and row.get("label") in LABELS)'),
    ("M22_pending_missing_error_cap", "reader177.py", "and unresolved_n == 0", ""),
    ("M23_W1_drops_whole_early_trace", "score177.py",
     'exclude = [row for row in paired_c if row["real_after"] or row["shuffled_after"]]',
     'exclude = [row for row in paired_c if row["early_real_n"] == 0]'),
    ("M24_W2_trace_denominator", "score177.py", '"total_C_windows": shuffled_c["denominator"]',
     '"total_C_windows": len(c_traces)'),
    ("M25_gate4_window_denominator", "score177.py",
     '"denominator_windows": window_margin["denominator_windows"]',
     '"denominator_windows": {"real": len(target), "shuffled": len(target)}'),
    ("M26_W3_bar", "score177.py", "PREDICTION_W3_HOLD = 1 / 3", "PREDICTION_W3_HOLD = 0.4"),
    ("M27_W3_floor", "score177.py", "MIN_N_W3 = 20", "MIN_N_W3 = 19"),
    ("M28_1774_floor", "score177.py", "MIN_N_177_4 = 20", "MIN_N_177_4 = 19"),
    ("M29_1776_floor", "score177.py", "MIN_N_177_6 = 15", "MIN_N_177_6 = 14"),
    ("M30_token_cap", "read177.py", "TOKEN_CAP = 32_500", "TOKEN_CAP = 32_499"),
    ("M31_shuffle_window_path", "read177.py", 'if draw_info["draws"]:',
     'if False and draw_info["draws"]:'),
    ("M32_upstream_last_box_rule", "loop_upstream.py", "return _box_at(text, idx)",
     'return _box_at(text, text.rfind("\\\\fbox"))'),
    ("M33_W1_gap_strictness", "score177.py",
     "elif gap > PREDICTION_W1_GAP_BAR", "elif gap >= PREDICTION_W1_GAP_BAR"),
    ("M34_null_last_box_subcase", "read177.py",
     'return earlier_has_gold and (predicted is None or normalise_value(predicted) != normalise_value(gold))',
     'return earlier_has_gold and predicted is not None and normalise_value(predicted) != normalise_value(gold)'),
    ("M35_box_comparison", "read177.py",
     'return earlier_has_gold and (predicted is None or normalise_value(predicted) != normalise_value(gold))',
     'return earlier_has_gold and (predicted is None or normalise_value(predicted) == normalise_value(gold))'),
    ("M36_gate4_margin_threshold", "score177.py", "GATE4_MARGIN_BAR = 0.25", "GATE4_MARGIN_BAR = 0.0"),
])


def _apply_mutant(folder, mutant):
    name, relative, old, new = mutant
    path = folder / relative
    source = path.read_text(encoding="utf-8")
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{name}: expected one mutation site in {relative}, found {count}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8", newline="\n")


def run(python=sys.executable):
    killed, survivors = [], []
    environment = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="part177-mutants-") as temporary:
        scratch = Path(temporary)
        for index, mutant in enumerate(MUTANTS, 1):
            name = mutant[0]
            case = scratch / f"case-{index:02d}"
            case.mkdir()
            for filename in FILES:
                shutil.copy2(HERE / filename, case / filename)
            _apply_mutant(case, mutant)
            completed = subprocess.run([python, "-I", "-B", str(case / "test_part177.py")],
                                       cwd=case, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, timeout=180)
            if completed.returncode:
                killed.append(name)
                print(f"KILLED {name}", flush=True)
            else:
                survivors.append(name)
                print(f"SURVIVED {name}", flush=True)
                print("\n".join(completed.stdout.splitlines()[-18:]), flush=True)
    print(f"MUTANTS {len(MUTANTS)} KILLED {len(killed)} SURVIVED {len(survivors)}", flush=True)
    if survivors:
        print("SURVIVORS: " + ", ".join(survivors), flush=True)
    return 1 if survivors else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    return run(args.python)


if __name__ == "__main__":
    raise SystemExit(main())
