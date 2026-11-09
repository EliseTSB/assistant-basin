# Fold brief, round 2: Part 200's code review

Repository: `/home/elisepc/green/worktrees/ab-part200`. Read `part200/review/CODE-REVIEW-r1.md` (9 findings; the
reviewer's probe harnesses are in `part200/review/code-r1-scratch/`), then `PART200-DESIGN.md`. Fold every finding into
`part200/run200.py` and `part200/test_part200.py`; also fix `part200/BUILD-REPORT.md`'s stale lines (finding 9) by
appending a correction, not by rewriting history. Write only in `part200/`. Do not commit. Do not change `part163/` or
`part199/`.

## Required

1. **Default loader** (blocker): `run()` uses `EagerLoader(args)` when `loader is None`. Test: `run200.main([...])` with
   `neuron_report.load_model` monkeypatched and no loader argument runs to completion on a fake model.
2. **Gate R2 hash**: `EXPECTED_UNITS_SHA256 = "2ed3bef4488eb6b3337dfde100c63c7c5a5df7e8ca809681dec34824c21dd710"`;
   Gate R2 fails unless the units file's sha256 equals it. Test with an altered units file (expect fail) and the real
   file (expect pass).
3. **Batch-1 completeness**: `--compare` exits 3 with a message unless `batch1.jsonl` holds exactly 200 unique units,
   one per unit in `units163.json`. The shape flag is computed only on a complete file (so it cannot be silently false).
   Test: 150 rows gives exit 3.
4. **Gate R3 observes**: per chunk, record the row count of the push tensors passed and `len(answers)` returned; for
   one-row calls (no-push and batch-1) record the measured `len(answers)`. Gate R3 reads these observed values. If you
   can record the batch seen by `model.generate` without editing Part 148 or Part 160 (for example a wrapper on
   `model.generate` installed by the wrapper for the duration of each call), do it and have R3 check it equals the chunk
   size; say in the report whether you did.
5. **Torn or partial writes**: at startup, before `run163.run` reads `answers.jsonl` and before the batch-1 stage reads
   `batch1.jsonl`, (a) cut a final line that has no trailing newline, and (b) cut `answers.jsonl` back to the last
   complete generate call (a recipe or battery call is complete when all its planned rows, 64 or 8, are present for that
   prompt and unit block; the plan is run163's own order: no-push rows, then recipe in units order in blocks of 64, then
   each battery question in units order in blocks of 64), so a relaunch regenerates whole calls and Gate R3 sees chunks
   of exactly 8. Record what was cut (rows and bytes) in the launch's `load.json` entry. Tests: a 64-row call cut at 30
   rows, relaunch, Gate R3 passes and no row is duplicated; a torn last line in each file, relaunch succeeds.
6. **Refused launch recorded**: a non-eager load writes a `load.json` entry with status `refused` and the `LAST_LOAD`
   it saw, then exits 3. Gate R1 treats a refused entry as a refusal (no rows generated in it), not as a pass.
7. **Prediction outcomes**: `--compare`'s scored section includes each of 200.1 to 200.7 with its value(s) and a
   held/failed result computed from the design's text (200.2 from the five verdicts; 200.3 neuron broken at least twice
   random; 200.4 each arm at least 480 at the cap; 200.5 each arm's broken count higher than published; 200.6 neuron
   pooled Qwen rate below random; 200.7 batch-1 differing at most 20; 200.1 from Gates R1 to R3 and Part 163's gates).
   Test each with a fixture where it holds and one where it fails.
8. **Keep** every existing test passing, and the self-check reproduction on `results/part163`.

Run with `/home/elisepc/neuron-probe/.venv/bin/python -I -B`: `part200/test_part200.py` (pytest and as a script),
`part163/test_part163.py`, `part199/test_part199.py`. For findings 1, 2, 3 and 5, show in your report that breaking the
fix turns its test red (then restore). No model, no GPU, no file over 1 MB copied.

## Report

Append "Fold round 2" to `part200/BUILD-REPORT.md`: each finding, the change, its test, sabotage results, test counts,
anything not done as written. End with SUMMARY / FINDINGS / BLOCKERS lines.
