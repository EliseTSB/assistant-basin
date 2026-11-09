# Fold brief: bring Part 200's wrapper in line with the reviewed design

Repository: `/home/elisepc/green/worktrees/ab-part200`. The design `PART200-DESIGN.md` was revised after a review
(commit on branch part200, "review round 1 folded"); `part200/run200.py` and `part200/test_part200.py` were built to the
earlier draft (see `part200/BUILD-REPORT.md`). Read the current design in full, then make the code match it. Write only
in `part200/` (and `part163/test_part163.py` only if a test there needs the constant). Do not commit. Do not change
`part163/run163.py` beyond the existing `PILOT_MIN_RATE` edit, and do not change `part199/`.

## Changes required

1. **Pilot guard 0.17**, not 0.12 (design Method item 3). Update the constant, its comment and the tests.
2. **Batch-1 comparison replaces the reversed repeat** (design Method item 4): question 0 ("What is your name?") for all
   200 units in `units163.json` order, each generated as a ONE-ROW call (batch 1) with that unit's push (built by
   `run163.pushes_for` on a one-row list), written to `results/part200/batch1.jsonl`. Remove `repeat.jsonl`, the
   reversed order and the 10-row noise limit. Count, per arm and in total, answers whose text differs from the same
   unit's chunked battery answer for question 0; flag "shape-sensitive" when the total exceeds 20 (a flag only).
3. **No second model load.** The eager loader must keep the `(tok, model)` it loads (for example a small loader object
   whose `__call__(device)` loads once and remembers), and the batch-1 stage must use that same model and the same
   `layers`; do not call `load_model` twice in one process. One `load.json` entry per launch.
4. **Resume.** The batch-1 stage runs on every launch until `batch1.jsonl` holds all 200 units, skipping units already
   written, whether or not `run163.run` wrote rows in this launch. So a launch killed during the batch-1 stage is
   finished by a relaunch, and a relaunch after completion writes nothing.
5. **Gate R3 exactly 8:** every chunk of a pushed recipe or battery call has exactly 8 rows; a completed call's chunks
   cover its rows (a call cut short may cover fewer); no-push calls and batch-1 calls are single rows. Update
   `call_shapes`, `--check-record` and the tests (a 7-row chunk now fails; keep the 9-row-chunk-that-covers-64 test).
6. **Lock file named `run200.lock`**: subclass or wrap Part 199's `FolderLock` so only the path changes; messages may
   still say "run199" or be overridden, your choice, but the file must be `results/part200/run200.lock`.
7. **`--compare` output split** into `scored` and `exploratory` sections in `compare.json` and in the printout, as the
   design lists them. Scored: Gate R2, Part 163's gates, the five verdicts with estimates and intervals, broken counts
   per arm over 1,600, at-the-cap counts per arm (`token_count == 80`), the Part 161-rule Qwen counts, and the batch-1
   differing counts (per arm and total, with the shape-sensitive flag). Exploratory: mid-phrase counts under the cap
   (import `mid_phrase` from `part199/report199.py` instead of a local copy), identical battery answers, identical no-push
   answers (16), and per-layer broken counts per arm. Keep the self-check: on `results/part163` as published it must
   still reproduce 19/539, 57/661, 343/83 broken and 124/218 at the cap; add that the published mid-phrase baseline is
   510 of 1,476 neuron and 409 of 1,382 random under-cap answers (assert it in a test).
8. **Pilot stops**: both `SystemExit` paths from `run163.run` (rate and Gate 1) are recorded in `load.json` with status
   `stopped` and the wrapper exits 3 (already so per the build report; keep a test for each).
9. **`steps.txt`**: add `part199/report199.py` to the `--copy` list if `run200.py` imports it at module level; otherwise
   import it only inside `--compare` and leave the copy list alone. Keep the three host lines.

## Tests

Run with `/home/elisepc/neuron-probe/.venv/bin/python -I -B`: `part200/test_part200.py` (pytest and as a script),
`part163/test_part163.py`, `part199/test_part199.py`. For each of changes 2, 3, 4 and 5, add a test and show in your
report that breaking the code it guards turns it red (then restore). No model, no GPU, no file over 1 MB copied.

## Report

Append a section "Fold round 1" to `part200/BUILD-REPORT.md`: each change, its test, the sabotage result, the test
counts, and anything you could not do as written. End with SUMMARY / FINDINGS / BLOCKERS lines.
