# Cold review: Part 200's code, before registration

Repository (read-only for you): `/home/elisepc/green/worktrees/ab-part200` (branch part200). Your writable folder:
`/home/elisepc/green/worktrees/ab-part200/part200/review/`. Write `part200/review/CODE-REVIEW-r1.md`.

Read `PART200-DESIGN.md` (the specification), then `part200/run200.py`, `part200/test_part200.py`, `part200/steps.txt`,
`part200/BUILD-REPORT.md`, and the code it calls: `part163/run163.py` (its only change is `PILOT_MIN_RATE`; see
`git -C /home/elisepc/green/worktrees/ab-part200 diff 8ccf6e3 -- part163/run163.py`), `part148/battery148.py`
(`generate`), `part160/run160.py` (`generate_checked`), `part86/common86.py` (`AddRowsHook`, `token_inputs`),
`neuron_report.py` (`load_model`, `LAST_LOAD`), `part199/run199.py` (the reused lock and record helpers),
`part163/score163.py`.

Rules: no folder copies and no file over 1 MB copied; no model, no GPU; run Python as
`/home/elisepc/neuron-probe/.venv/bin/python -I -B`; test scripts and scratch copies go in your folder only; do not edit
the repository. You may run the existing tests and write new throwaway tests in your folder that import the repository
code (put the repository root on sys.path).

The tests pass with a fake model. Find what the fakes hide, on the real path:
- Does the chunking `generate_fn` call Part 163's real generator with the right signature, and does slicing the push
  tensors per chunk give each row exactly its own push on the real hook (`AddRowsHook`, `begin_sequence`)? Is any
  per-call state (hooks, caches, `common.token_inputs` batch size) computed from the full call instead of the chunk?
- Will the real eager loader work with `neuron_report.load_model`'s actual signature and `LAST_LOAD` keys?
- Does the batch-1 stage build the same push for a unit as the battery does (same direction cache, same dose, same
  layer), and use the same model without a reload?
- Resume and kill: what happens if the task is killed mid-battery, mid-batch-1, or between them, and relaunched? Can
  any path write a duplicate row, skip a row, or record a shape Gate R3 then misreads?
- Do `--check-record` and `--compare` exit as the design says, and does `--compare`'s scored section contain exactly
  the scored items the design lists?
- Does `steps.txt`'s `--copy` list include every module the Windows job imports (check every import, including
  imports inside functions that the run path calls)?
- Any way a wrong answer could be written while every test passes.

Format: a verdict line (LAUNCH, LAUNCH-WITH-FIXES or HOLD), numbered findings with evidence (a test you wrote or a line
you read) and a suggested fix, then SUMMARY / FINDINGS / BLOCKERS lines.
