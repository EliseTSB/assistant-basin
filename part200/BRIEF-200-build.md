# Build brief: Part 200 (Part 163 rerun on the eager kernel)

Repository: `/home/elisepc/green/worktrees/ab-part200` (a git worktree of the public research repo; do not commit,
the operator commits). Read `PART200-DESIGN.md` first and in full: it is the specification. Where this brief and the
design disagree, the design wins; say so in your report.

You may write only in `part200/` and `part163/` (and create `results/part200/` only for test fixtures you delete
afterwards; prefer a temporary directory inside `part200/`). Do not edit `PREDICTIONS.md`, the front page, any results
folder, or any other part's code.

## Deliverables

1. `part200/__init__.py` (empty) and `part200/run200.py`:
   - default mode: runs `part163.run163.run(args, loader=..., generate_fn=...)` with the eager loader, the chunking
     `generate_fn` and the pilot guard described in the design's Method (items 1 to 3), holding a folder lock
     (`results/part200/run200.lock`, created exclusively; refuse with exit 3 if present), then the repeat batch (item 4)
     into `results/part200/repeat.jsonl`, resumable (skip rows already written). Same arguments as `run163.main`
     (`--device`, `--model`, `--units`, `--out`, `--batch-size`, which must be 64 so the call shapes are Part 163's).
   - `--check-record <load.json>`: Gates R1 and R3 over every launch entry; exit 0 on pass, 3 on fail, printing why.
   - `--compare <rerun_dir> <published_dir>`: everything the design's `--compare` paragraph lists, written to
     `<rerun_dir>/compare.json` and printed; the exit codes it states. It must import Part 163's rules (Part 86 triage
     via `part86/triage86.py`; the name/maker rules via `part163/score163.py` where they live) rather than retype them,
     except the Part 161 Qwen rule and the Part 191 mid-phrase rule, which are written once in `run200.py` with the
     design's exact patterns.
   - Reuse `part199/run199.py` by import where a helper fits unchanged (its folder lock, load record writer, kernel
     checks). Do not copy its code. If a helper is tied to Part 170's call structure (ban lists, release steps), write a
     Part 200 version and say why in a comment.
2. `part163/run163.py`: the one behaviour-preserving edit the design names (`PILOT_MIN_RATE = 1.5`, used where the
   literal was). Nothing else in that file changes.
3. `part200/test_part200.py` (pytest style; also runnable as `python -I -B part200/test_part200.py` with a `__main__`
   block that runs every test and exits non-zero on failure). Fixture tests with a fake model and tokenizer, no GPU and
   no real model. At least:
   - chunking: a 64-row pushed call reaches the generator as 8 chunks of 8 in the original row order, each chunk's push
     tensors are exactly the matching rows of the original, and the answers come back in the original order; a 1-row
     no-push call is unchanged;
   - the loader refuses (exit 3, before any generation) when `LAST_LOAD` or the config does not record eager;
   - `PILOT_MIN_RATE` defaults to 1.5 in `run163` and the wrapper sets 0.12 only for its own run, restoring it after;
   - the repeat writes 200 rows in reversed unit order and counts changed answers against the battery rows;
   - `--check-record` passes a good record and fails a non-eager one and one with a 9-row chunk;
   - `--compare` run with `results/part163` as both folders reproduces 19 of 539 and 57 of 661 (Part 161 rule), 343 and
     83 broken, 124 and 218 battery answers at the 80-token cap, and Part 163's five verdicts;
   - a sabotage check: for each of the chunking, kernel-refusal and Gate R3 tests, show (in your report) that breaking
     the code it guards turns the test red, then restore the code.
4. `part200/steps.txt`: the launch line, in the shape of `part199/steps.txt` and `part163/steps.txt`, through
   `tools/launch_arc.py 200 --step "..." --copy ...`, naming every file the job needs to run (run200.py, part199/run199.py
   and its imports, part163/*.py, units163.json, part160, part155, part153, part148, part86 files, neuron_report.py,
   results/part112/medians.json), then the three host lines from the design.

## Rules

- Run Python as `/home/elisepc/neuron-probe/.venv/bin/python -I -B`. Run the new tests and `part163/test_part163.py`
  and `part199/test_part199.py` (they must still pass). Do not run any model and do not use the GPU.
- Do not change `part199/run199.py`; if you need a hook it lacks, write it in `run200.py`.
- No file over 1 MB is copied anywhere.

## Report

Write `part200/BUILD-REPORT.md`: what you built, every test and its result (command and pass/fail counts), the sabotage
checks, anything in the design you could not implement as written, and SUMMARY / FINDINGS / BLOCKERS lines.
