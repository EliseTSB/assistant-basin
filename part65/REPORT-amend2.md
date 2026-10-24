# Part 65 Amendment 2 report

## SUMMARY

Implemented the missing-trial policy in analysis, judge recovery messaging, tests, and the runbook. The analysis now derives 1,000 registered pairs from the frozen 500-unit collection, includes only complete units in primary predictions, and reports both all-unit sensitivity scenarios without using them for pass/fail. The offline suite passes: 29 tests. No GPU, model download, paid API call, or commit was used.

## FINDINGS

- Rule 1 — `analyze65.py` builds the two registered trial pairs per collection unit and counts absent rows separately from unparseable answers; `test_absent_registered_pair_counts_against_the_1000_trial_plan` catches the mutant that derives the plan from answers and misses absent pairs.
- Rule 2 — `analyze65.py` excludes a unit whenever either trial is missing and reports excluded units by type, layer band, and number of missing trials; `test_one_valid_trial_excludes_and_counts_the_unit` catches the mutant that scores one surviving answer as a full unit vote.
- Rule 3 — `analyze65.py` stops above 50/1,000 missing registered trials and reports that denominator; `test_missing_trial_stop_uses_the_registered_50_of_1000_boundary` and `test_absent_registered_pair_counts_against_the_1000_trial_plan` catch the wrong cutoff and answer-row denominator mutants.
- Rule 4 — `analyze65.py` and its Markdown renderer report 65a, 65b, and first/middle/last neuron-band estimates with missing trials scored 0.5 or wrong; `test_missing_trial_bounds_score_all_units_and_leave_primary_complete_case` catches bounds omitting excluded units or changing primary pass/fail.
- Rule 5 — `analyze65.py` keeps strict JSONL parsing for analysis input; `test_analysis_reader_rejects_a_torn_last_answer_line` catches a lenient-reader mutant that silently skips a malformed final answer line. The judge resume reader remains lenient.
- Rule 6 — `judge65.py` prints the released chunk, stored error, and new-output-directory instruction; `RUNBOOK-65.md` documents finality and the pilot equal-norm re-read. `test_judge_main_fails_and_prints_released_chunk` and `test_released_chunk_is_final_and_does_not_reserve_again` catch missing operator guidance and resubmission/reservation mutants.

## BLOCKERS

None. Parent control message 1 was considered and applied as a scope boundary: Amendment 3 was not implemented, and `steer65.py` was not edited. No control messages were declined or deferred.

## Claude, after collection (2026-10-24)

Host: 29 tests, OK. Mutant of my own: scoring a one-trial unit on its surviving trial (the round-2 H1 shape) fails `test_one_valid_trial_excludes_and_counts_the_unit` and `test_missing_trial_bounds_score_all_units_and_leave_primary_complete_case`; file restored by hash.
