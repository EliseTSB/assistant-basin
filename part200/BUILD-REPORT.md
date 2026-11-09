# Part 200 build report

Built 2026-11-09. No model, no GPU. Python: `/home/elisepc/neuron-probe/.venv/bin/python -I -B` (torch 2.14.1+cpu,
pytest 9.1.1).

## What was built

- `part200/__init__.py` (empty), `part200/run200.py`, `part200/test_part200.py`, `part200/steps.txt` (this report).
- `part163/run163.py`: the one behaviour-preserving edit — `PILOT_MIN_RATE = 1.5` next to `MODEL_DEFAULT`, used where
  the literal `1.5` was (`if rate < PILOT_MIN_RATE`, and the message now prints the constant). Nothing else changed
  (`git diff`: 5 insertions, 2 deletions).

`run200.py`, default mode: `part163.run163.run(args, loader=eager_loader(args), generate_fn=chunked)` under Part 199's
folder lock (`results/part200/run199.lock`, created exclusively via `part199.run199.FolderLock`, exit 3 if present),
then the repeat batch. Same arguments as `run163.main` (`--device --model --units --out --batch-size`, 64 only).

1. **Kernel.** `eager_loader` loads through `neuron_report.load_model`, then `requires_grad_(False)` and `eval()`, and
   raises `SystemExit(3)` before generation unless both `neuron_report.LAST_LOAD["attn_implementation"]` and
   `model.config._attn_implementation` are `"eager"`. The load record (`load.json`, one entry per launch, rewritten
   after every chunk) and the kernel/same-run checks are **imported** from `part199/run199.py`
   (`FolderLock`, `_read_record`, `_save_record`, `_answers_written`, `_finite`, `check_kernels`, `check_same_run`).
2. **Batch shape.** `chunked_generate(original)` wraps the generate_fn with Part 163's exact signature
   `(model, tok, layers, question, pushes, max_new_tokens, device)`. Pushed calls are split into chunks of at most 8
   rows in the original row order, each layer's push tensor sliced to the chunk's rows; no-push calls pass through
   unchanged. Every call and its observed chunk sizes are logged into the load entry.
3. **Pilot.** `set_pilot_min_rate(0.12)` / `restore_pilot_min_rate(previous)` around the run (restored in `finally`).
4. **Repeat.** After `run163.run` returns (and wrote rows), the wrapper reloads through the same loader, records that
   load as its own launch entry, and regenerates question 0's 200 battery rows in reversed unit order, calls of 64
   chunked to 8, using `part163.run163.pushes_for` unchanged, into `results/part200/repeat.jsonl` (resumable: rows
   already written are skipped; 25 chunks of 8). `run200.repeat_changes` counts answers whose text differs from the
   same unit's battery answer; more than 10 of 200 is flagged noise-limited.

`--check-record <load.json>`: Gates R1 (every load eager; one model, revision and folder) and R3 (no chunk over 8 rows,
a completed pushed call's chunks cover its rows, no-push calls single rows) over every launch; exit 0/3, printing why.

`--compare <rerun> <published>`: writes `<rerun>/compare.json` and prints it — Gate R1/R3 over the rerun record; Gate
R2 (units163.json sha256 identical, the 16 battery questions and their order identical to Part 163's answers); Part
163's three gates and the five verdicts with intervals, both sides (published side from Part 163's own `score.json`,
rerun side from `score163.score_rows` on the rerun's answers — the scorer is imported, not retyped); broken counts per
arm over 1,600 battery answers (Part 86 triage imported); the Part 161-rule Qwen count per arm on maker phrasings;
each arm's share of battery answers under the 80-token cap and the Part 191 mid-phrase count among them; identical
battery answers; the repeat's changed count. Exit 3 on a failed Gate R1/R3, a failed Part 163 gate, or a failed Gate
R2; exit 2 on a missing file.

Rules written once in `run200.py` with the design's exact patterns: `PART161_QWEN = re.compile(r"\bQwen\b")` (no
IGNORECASE), `PART161_EXCLUDE = re.compile(r"alibaba|tongyi|通义", re.IGNORECASE)`,
`PART191_MID_PHRASE = re.compile(r"[A-Za-z0-9,\-:;]$")` on the rstripped answer.

## Tests and results

| Command | Result |
| --- | --- |
| `python -I -m pytest -q part200/test_part200.py` | **27 passed** |
| `python -I -B part200/test_part200.py` | **27 passed**, exit 0 |
| `python -I -m pytest -q part163/test_part163.py` | **7 passed** |
| `python -I -m pytest -q part199/test_part199.py` | **38 passed** |

What the 27 cover: 64 rows → 8 chunks of 8, each chunk's push tensors exactly the matching rows of the original (two
layers), answers in original order; a 1-row no-push call unchanged; Part 163's battery shapes 64→[8]×8 and 8→[8];
short chunk and disagreeing pushes refused; loader refuses `sdpa`, `eager`-record-with-`sdpa`-config and `None`
with exit 3 and no model preparation; loader accepts eager and applies no-grad/eval; `run163.PILOT_MIN_RATE == 1.5`;
wrapper sets 0.12 and restores 1.5; repeat rows are question 0 in reversed unit order; the repeat writes 200 rows in
reversed order in 25 chunks of 8 and counts changed answers against the battery; the repeat resumes; the noise limit
is 10; Gate R3 passes a good record and fails 9-row chunks, 64-row bypass, lost rows, an 8-row no-push call, no pushed
call, a kill-mid-call (passes) and an over-long incomplete call; `--check-record` passes a good record and fails a
non-eager one, a 9-row chunk and a disagreeing folder; the run refuses a non-eager launch with exit 3 before
generation and releases the lock; the run records the eager load and its chunks, runs the repeat after, restores the
pilot rate; a stale lock refuses a second launch; `--compare` exits 3 on a non-eager record and on a 9-row chunk, 2 on
a missing file; **`--compare` on `results/part163` as both folders reproduces 19 of 539 and 57 of 661, 343 and 83
broken, 124 and 218 at the cap, and 163.1–163.4 held / 163.5 failed** (both sides, published score.json and the rerun
scored by score163.py); the Part 161 rule is case-sensitive and excludes broken and maker-named answers; the Part 191
rule; the Part 163 edit is the only change.

## Sabotage checks (each turned red, then restored; `cmp` against the backup confirmed the restore)

- **Chunking** — `chunked_generate` made to pass every call through unchunked: 6 tests red
  (`test_a_64_row_pushed_call_…`, `test_part163_battery_call_shapes_…`, `test_a_short_chunk_is_refused`,
  `test_pushes_that_disagree_…`, `test_repeat_writes_200_rows_…`, `test_run_records_the_eager_load_…`).
- **Kernel refusal** — the `if kernel != "eager" or config_kernel != "eager"` guard disabled: 4 tests red (the three
  parametrised `test_loader_refuses_…` and `test_run_refuses_a_non_eager_launch_…`).
- **Gate R3** — first tried removing only the `size > CHUNK` half of `bad_pushed`: **27 passed — not caught.** The
  test that should catch a 9-row chunk (`test_check_record_…_a_9_row_chunk`) was passing for the wrong reason: its
  9-row call also loses a row (9+7×8 = 65 ≠ 64), so the "chunks cover its rows" half failed it. Fixed the test to use
  a 9-row chunk that still covers 64 rows (`observed [8, 9, 8, 8, 8, 8, 8, 7]`, also in `test_gate_r3_…`). Then
  disabling the whole `wrong = [call for call in pushed if bad_pushed(call)]` line: `test_gate_r3_reads_the_observed_chunks`
  red. Restored; all 27 pass.

## Where the brief and the design disagree, or the design could not be implemented as written

1. **`--compare` on `results/part163` as both folders.** Part 163 wrote no `load.json`, so requiring one in the
   published folder makes the brief's own reproduction check exit 2. The design's exit-3 conditions are all about the
   *rerun's* `load.json`, so `load.json` is required of the rerun only; the published folder needs `answers.jsonl`
   (and uses its `score.json` when present). The test supplies its own rerun folder (a copy of Part 163's answers plus
   a good record) and compares it against `results/part163`.
2. **Reuse by import.** Part 199's `chunked_generate`, `call_shapes` and `check_record` do not fit: they wrap
   `run170.generate_with_bans` (bans, release steps, a directions mapping looked up by (layer, slot)), whereas Part
   163's generate_fn receives the push tensors themselves; and `call_shapes` is tied to Part 170's zero-write checks
   and its "pushed call is a multiple of 8" rule, which Part 163's 8-row last battery call violates. Part 200 versions
   are written, with the reason in the module comment. The lock, record writer and kernel checks are reused unchanged.
3. **The repeat reloads the model.** `run163.run` does not return its model, and the brief forbids changing
   `part199/run199.py` (and the design allows only the pilot edit to `run163.py`), so the repeat loads again through
   the same eager loader and records that load as its own launch entry ("one entry per launch"). Cost: one extra load
   (minutes). The alternative — a second `run163.run` pass — would not have produced a reversed-order repeat.
4. **The repeat's pushes** are built by `run163.pushes_for` on the reversed unit rows, i.e. the same pushes the
   battery rows get, with the rows in reverse order (the design says "with the same pushes"); the direction cache is
   rebuilt rather than shared, since the model is reloaded.
5. **The pilot stop is `SystemExit` from `run163`**, which the wrapper treats as a stop (exit 3, status "stopped"). The
   design says a pilot stop is a HOLD; that is an operator reading of the exit, not something the wrapper can decide.
6. **The repeat runs only when `run163.run` wrote rows this invocation.** A relaunch that finds all 3,416 rows written
   leaves the repeat alone (it would otherwise be a second repeat). If the repeat itself was cut short, relaunching
   with the answers complete will not finish it; the operator would need to delete one answer row or run the repeat
   path by hand. This is the one resume corner I could not close without touching Part 163.
7. **`--compare` does not write Part 163's `score.json`.** The design's host sequence runs `score163.py` first; if it
   has not been run, the rerun side is still scored inside `--compare` (same function), but `results/part200/score.json`
   is only written by the separate `score163.py` step in `steps.txt`.
8. **`steps.txt`** copies `part199/run199.py` and `part199/__init__.py` (imported at module level) but not
   `part168/part170/part165`, which `run199.py` imports only inside functions this part never calls. The lock file is
   Part 199's `run199.lock` — the design says `run200.lock`; I reused Part 199's lock class unchanged (as instructed),
   so the file name is `run199.lock`. The brief's `run200.lock` name is not honoured; say if that matters.

## Not done

- No model run, no GPU, no launch. `results/part200/` was never created (the tests use `tmp_path`).
- No commit.

SUMMARY: Built part200/run200.py (eager loader with Gate R1 refusal, chunks of 8, pilot floor 0.12, repeat batch,
--check-record, --compare), the one PILOT_MIN_RATE edit in part163/run163.py, 27 fixture tests (all pass, 3 sabotage
checks red then restored), and steps.txt. --compare on Part 163 reproduces 19/539, 57/661, 343/83, 124/218 and the five
verdicts. part163 and part199 tests still pass (7 and 38).
FINDINGS: Part 163 has no load.json, so the published folder cannot supply one; --compare requires it of the rerun only.
Part 199's chunker and Gate R3 do not fit Part 163's call shape, so Part 200 versions are written with the reason in
the comment. The repeat reloads the model because run163.run returns no model and only the pilot edit is allowed.
BLOCKERS: none.
