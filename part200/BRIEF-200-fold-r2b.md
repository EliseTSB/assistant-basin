# Finish fold round 2 (Part 200)

Repository: `/home/elisepc/green/worktrees/ab-part200`. A previous worker started folding the code review
(`part200/review/CODE-REVIEW-r1.md`, 9 findings) as specified in `part200/BRIEF-200-fold-r2.md`, and hit its turn cap
with the work committed as WIP (HEAD). Read both briefs, the review, `PART200-DESIGN.md`, and the current
`part200/run200.py` and `part200/test_part200.py`.

Current state: `python -I -m pytest -q part200/test_part200.py part163/test_part163.py part199/test_part199.py` gives
88 passed, 7 failed. The failing tests:
- test_main_with_no_loader_argument_runs_to_completion_on_a_fake_model
- test_a_no_push_call_records_its_measured_answer_count
- test_a_refused_launch_is_recorded_and_gate_r1_reads_it_as_a_refusal
- test_a_call_cut_at_30_rows_is_cut_off_and_the_relaunch_regenerates_whole_calls
- test_repair_cuts_a_torn_last_line_in_each_file
- test_a_relaunch_after_a_torn_line_in_each_file_succeeds
- test_repair_leaves_a_whole_file_alone
The previous worker looped on the startup repair function (`_complete_line_count` and its neighbours). Decide for each
failure whether the code or the test is wrong against the brief and the design, fix it, and keep every requirement of
`BRIEF-200-fold-r2.md` (1 to 8) met. Prefer the simplest correct repair: cut a final line without a trailing newline;
then cut `answers.jsonl` back to the last complete generate call in run163's plan order (no-push rows one per call;
recipe in units order in blocks of 64; each battery question in units order in blocks of 64; a block of 8 at the end
of each 200). Record what was cut in the launch entry.

Write only in `part200/`. Do not commit. No model, no GPU, no file over 1 MB copied. Python:
`/home/elisepc/neuron-probe/.venv/bin/python -I -B`.

FIRST, before any code change, append a short "Fold round 2 (finished by a second worker)" heading to
`part200/BUILD-REPORT.md`, and keep it updated as you go, so the report exists even if you run out of time.
Final report under that heading: each of the 9 findings with its change, its test and (for 1, 2, 3, 5) a sabotage that
turns the test red, the final test counts for all three test files, and anything not done as written. End with
SUMMARY / FINDINGS / BLOCKERS lines.
