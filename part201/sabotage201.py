#!/usr/bin/env python3
"""Show that the Part 201 tests can fail: break one guarded line at a time, in a copy, and expect named tests to go red.

    python -I part201/sabotage201.py            (about two minutes; exit 0 = every sabotage turned its tests red)
    python -I part201/sabotage201.py "push hash" (only the sabotages whose label contains that text)

The working tree is never edited. The script builds an overlay in a temporary folder: `part201/` and `tools/` are
copied, everything else is a symlink to the real tree. Each sabotage replaces one exact piece of text in the copy
(the script stops if the text is not there exactly once, so a sabotage that did not land cannot pass as "green"),
runs the named tests, and restores the file. First the same tests are run unsabotaged in the overlay and must pass.

Written by Claude (Fable 5.2), 2026-11-09, after Alden's review of a3a40b1 (council part201-design-review-2026-11-09
#11) asked for paired negative and positive fixtures for each correction.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN, SCORE, FLOOR = 'part201/run201.py', 'part201/score201.py', 'tools/shape_floor.py'
T_RUN, T_SCORE, T_FLOOR = 'part201/test_part201.py', 'part201/test_score201.py', 'tools/test_shape_floor.py'

# (what is broken, file, exact text, replacement, test file, tests that must fail)
SABOTAGES = [
    ('G5 compares raw rectangles again', RUN,
     "    return [eos_trim(r,eos_ids) for r in reference_tokens] == [eos_trim(r,eos_ids) for r in recorded_tokens]",
     "    return [list(map(int,r)) for r in reference_tokens] == [list(map(int,r)) for r in recorded_tokens]",
     T_RUN, ['test_g5_compares_rows_that_end_at_different_lengths', 'test_g5_launch_comparison_takes_the_generator_rectangle_as_it_is']),
    ('the push hash covers the call mates\' zero rows again', RUN,
     "        if not bool(torch.count_nonzero(values.to(dtype=torch.float32))): continue\n",
     "",
     T_RUN, ['test_push_hash_does_not_depend_on_the_rows_sharing_the_call']),
    ('a retry keeps the first attempt\'s receipt', RUN,
     "        kept=decode()\n",
     "        second=decode(); kept=dict(first,rows=second['rows'])\n",
     T_RUN, ['test_a_retry_keeps_the_second_attempts_rows_time_and_hook_counts', 'test_a_bad_second_attempt_is_not_hidden_by_a_good_first_receipt',
             'test_a_still_nonfinite_second_attempt_is_the_one_kept', 'test_a_nonfinite_first_attempt_is_replaced_by_the_retry_with_the_retrys_receipt']),
    ('the kept attempt\'s hook evidence is not checked', RUN,
     "    return kept,discarded,attempt_problems(kept,process,pushed)",
     "    return kept,discarded,attempt_problems(first,process,pushed)",
     T_RUN, ['test_a_retry_keeps_the_second_attempts_rows_time_and_hook_counts', 'test_a_bad_second_attempt_is_not_hidden_by_a_good_first_receipt']),
    ('resume reads rows that have no receipt', RUN,
     "        if load is not None:\n            for row in rows:",
     "        if False:\n            for row in rows:",
     T_RUN, ['test_rows_without_their_call_receipt_are_set_aside_on_resume', 'test_rows_whose_receipt_was_lost_are_decoded_again']),
    ('rows are written before their receipt', RUN,
     "                launch['calls'].append(receipt); _write_json(load_path,record)\n                _append(data_path,output); done.update((p,r['row_id']) for _,r in todo)\n",
     "                _append(data_path,output); done.update((p,r['row_id']) for _,r in todo)\n                launch['calls'].append(receipt); _write_json(load_path,record)\n",
     T_RUN, ['test_a_crash_while_a_receipt_is_written_leaves_no_rows_for_that_call']),
    ('a launch accepts other medians', RUN,
     "        if any(entry.get('inputs')!=inputs for entry in previous.get('loads',[])):",
     "        if False:",
     T_RUN, ['test_changed_medians_refuse_the_next_launch_and_respacing_them_does_not']),
    ('input hashes follow the file\'s spacing', RUN,
     "    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()",
     "    return hashlib.sha256(json.dumps(obj,ensure_ascii=False).encode('utf-8')).hexdigest()",
     T_RUN, ['test_inputs_are_hashed_in_a_canonical_form']),
    ('the timing probe leaves its head hook after an error', RUN,
     "    finally:\n        if slicer is not None: slicer.remove()\n",
     "    finally:\n        pass\n    if slicer is not None: slicer.remove()\n",
     T_RUN, ['test_a_probe_that_raises_leaves_no_hook_on_the_model']),
    ('the scorer does not ask for receipts', SCORE,
     "    problems = []\n    by_call = defaultdict(list)\n    for row in rows:\n        if row.get('pass') in passes:\n            by_call[(row['pass'], row.get('call'))].append(row)\n",
     "    return []\n",
     T_SCORE, ['test_a_row_naming_another_process_s_launch_fails_g3_for_its_precision', 'test_a_missing_call_receipt_is_a_failure_not_a_skipped_check',
               'test_a_receipt_for_other_rows_and_a_launch_that_is_not_in_the_record']),
    ('A-again may come from another launch', SCORE,
     "    if any(r.get('pass') == 'A-again' and _launch_index(r) != a_launch.get(r.get('row_id')) for r in rows):",
     "    if False:",
     T_SCORE, ['test_a_again_shares_pass_a_s_launch_and_a2_does_not']),
    ('A2 may come from pass A\'s launch', SCORE,
     "    if any(r.get('pass') == 'A2' and _launch_index(r) in set(a_launch.values()) for r in rows):",
     "    if False:",
     T_SCORE, ['test_a_again_shares_pass_a_s_launch_and_a2_does_not']),
    ('float32 and bf16 push hashes are not compared', SCORE,
     "    split_cross = sorted(str(rid) for rid, v in hashes.items() if v['float32'] and len(v['bf16'] | v['float32']) > 1)",
     "    split_cross = []",
     T_SCORE, ['test_a_float32_push_that_is_not_the_bf16_push_fails_g4_and_only_its_readings']),
    ('bf16 push hashes are not compared', SCORE,
     "    split_bf = sorted(str(rid) for rid, v in hashes.items() if len(v['bf16']) > 1)",
     "    split_bf = []",
     T_SCORE, ['test_two_bf16_passes_with_different_pushes_for_one_row_block_the_bf16_readings']),
    ('a pushed row may lack its push hash', SCORE,
     " or not r.get('push_sha256')\n",
     "\n",
     T_SCORE, ['test_a_pushed_row_without_a_push_hash_fails']),
    ('the float32 launch\'s inputs are not compared', SCORE,
     "    float_inputs_ok = bf_inputs_ok and bool(float_launches) and all(x.get('inputs') == bf_inputs[0] for x in float_launches)",
     "    float_inputs_ok = True",
     T_SCORE, ['test_changed_medians_fail_g4_while_the_plan_and_the_local_differences_stay_green']),
    ('the launches\' inputs need not be the registered files', SCORE,
     "    registered_ok = state['smoke'] or (bf_inputs_ok and bf_inputs[0] == registered_inputs())",
     "    registered_ok = True",
     T_SCORE, ['test_changed_medians_fail_g4_while_the_plan_and_the_local_differences_stay_green']),
    ('hook evidence is not required', SCORE,
     "            if receipt is None or forwards is None or not counts or any(v != forwards for v in counts):",
     "            if False:",
     T_SCORE, ['test_bf16_calls_need_their_hook_evidence_too', 'test_a_missing_call_receipt_is_a_failure_not_a_skipped_check',
               'test_g4_needs_float32_at_the_hook_and_a_hook_firing_per_forward']),
    ('G4\'s bf16 half does not block the bf16 readings', SCORE,
     "    bf_ok = (gates['G1']['bf16']['passed'] and gates['G3']['bf16']['passed'] and gates['G4']['bf16']['passed']\n             and gates['G5']['passed'])",
     "    bf_ok = (gates['G1']['bf16']['passed'] and gates['G3']['bf16']['passed']\n             and gates['G5']['passed'])",
     T_SCORE, ['test_two_bf16_passes_with_different_pushes_for_one_row_block_the_bf16_readings', 'test_bf16_calls_need_their_hook_evidence_too']),
    ('the real run accepts another snapshot', RUN,
     "        if not args.smoke and snapshot!=common.REVISION:",
     "        if False:",
     T_RUN, ['test_the_real_run_refuses_another_snapshot_before_loading_and_a_smoke_refuses_the_real_folder']),
    ('a smoke run may use the real output folder', RUN,
     "    if a.smoke and Path(a.out).resolve()==OUT_DEFAULT.resolve():",
     "    if False:",
     T_RUN, ['test_the_real_run_refuses_another_snapshot_before_loading_and_a_smoke_refuses_the_real_folder']),
    ('a hook that never fired reports a zero difference', RUN,
     "        if self._diff is None: return [None]*self.rows",
     "        if self._diff is None: return [0.0]*self.rows",
     T_RUN, ['test_a_hook_that_never_fired_has_measured_nothing']),
    ('bf16 at the hook is not required in a bf16 process', RUN,
     "    if any(r.get('hook_dtype')!=expected for r in attempt['rows']):",
     "    if process=='float32' and any(r.get('hook_dtype')!=expected for r in attempt['rows']):",
     T_RUN, ['test_a_hook_that_never_fired_has_measured_nothing']),
    ('the scorer does not compare the loaded snapshot', SCORE,
     "    snapshots_ok = state['smoke'] or all(x.get('model_snapshot') == x.get('revision') for x in launches)",
     "    snapshots_ok = True",
     T_SCORE, ['test_a_launch_that_loaded_another_snapshot_fails_g1']),
    ('a differing row without a margin is dropped in silence', SCORE,
     "        if reason is None and no_margin:\n            reason = f'not scored ({len(no_margin)} differing rows have no recorded margin at the first difference)'\n",
     "",
     T_SCORE, ['test_a_differing_row_without_a_margin_unscores_its_prediction']),
    ('a bf16 row may have any dtype at the hook', SCORE,
     "               or r.get('hook_dtype') != ('torch.float32' if float_arm else 'torch.bfloat16')]",
     "               or (float_arm and r.get('hook_dtype') != 'torch.float32')]",
     T_SCORE, ['test_a_bf16_row_needs_bf16_at_the_hook']),
    ('an absent pass K makes the float32 arm invalid', SCORE,
     "    kernel_invalid = gates['G1']['kernel']['present'] and not gates['G1']['kernel']['passed']",
     "    kernel_invalid = not gates['G1']['kernel']['passed']",
     T_SCORE, ['test_a_missing_pass_k_is_not_run_and_a_wrong_one_is_an_invalid_float32_run']),
    ('the paired error leaves out the covariance', FLOOR,
     "    variance = max(0.0, ((full_only + held_only) / draws - mean * mean) * draws / (draws - 1))",
     "    variance = 0.96 * 0.04 + 0.95 * 0.05",
     T_FLOOR, ['test_the_unsupported_row_flag_uses_the_paired_error']),
    ('simulate does not count the discordant redraws', FLOOR,
     "                    difference, paired_se = paired_difference(full_only[verdict], held_only[verdict], draws)",
     "                    difference, paired_se = paired_difference(0, 0, draws)",
     T_FLOOR, ['test_simulate_reports_the_paired_difference_from_the_same_redraws', 'test_a_verdict_that_rests_on_unsupported_rows_is_flagged']),
]


def build_overlay(folder):
    for entry in ROOT.iterdir():
        if entry.name in ('.git', '.pytest_cache', '__pycache__'):
            continue
        if entry.name in ('part201', 'tools'):
            shutil.copytree(entry, folder / entry.name, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
        else:
            (folder / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())


def run_tests(folder, test_file, names):
    done = subprocess.run([sys.executable, '-I', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', test_file, '-k', ' or '.join(names)],
                          cwd=folder, capture_output=True, text=True)
    tail = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else done.stderr.strip()[-200:]
    count = lambda word: int(m.group(1)) if (m := re.search(rf'(\d+) {word}', tail)) else 0
    return count('passed'), count('failed') + count('error'), tail


def main():
    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        build_overlay(folder)
        wanted = [item for item in SABOTAGES if len(sys.argv) < 2 or sys.argv[1] in item[0]]
        for label, file, old, new, test_file, names in wanted:
            target = folder / file
            original = target.read_text(encoding='utf-8')
            if original.count(old) != 1:
                problems.append(f'{label}: the text to break occurs {original.count(old)} times in {file}')
                print(f'NOT APPLIED  {label}')
                continue
            passed, failed, tail = run_tests(folder, test_file, names)
            if failed or passed != len(names):
                problems.append(f'{label}: before the sabotage {passed} passed and {failed} failed of {len(names)} ({tail})')
            target.write_text(original.replace(old, new), encoding='utf-8')
            try:
                passed, failed, tail = run_tests(folder, test_file, names)
            finally:
                target.write_text(original, encoding='utf-8')
            red = failed == len(names)
            print(f"{'red' if red else 'STILL GREEN'}  {failed} of {len(names)} named tests failed  {label}")
            if not red:
                problems.append(f'{label}: {failed} of {len(names)} named tests failed ({tail})')
    for problem in problems:
        print('PROBLEM:', problem)
    print(f'{len(wanted) - len(problems)} of {len(wanted)} sabotages turned every named test red')
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
