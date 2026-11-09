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

## Fold round 1

Folded 2026-11-09 to `PART200-DESIGN.md` as revised by review round 1. No model, no GPU. Python:
`/home/elisepc/neuron-probe/.venv/bin/python -I -B`. Files changed: `part200/run200.py`, `part200/test_part200.py`,
`part200/steps.txt`, `part163/test_part163.py` (one added test). `part163/run163.py`, `part199/` and
`PART200-DESIGN.md` untouched. Not committed.

### Each change

1. **Pilot guard 0.17.** `run200.PILOT_MIN_RATE = 0.17` (was 0.12), comment updated to the design's wording (the rate
   below which 3,616 answers would not finish in the 6-hour limit). `run163.PILOT_MIN_RATE` stays 1.5.
   Tests: `test_wrapper_sets_0_17_for_its_own_run_and_restores_it` (sets 0.17, restores 1.5),
   `test_pilot_min_rate_defaults_to_1_5_in_run163`, and in `part163/test_part163.py`
   `test_pilot_min_rate_default_is_1_5` (the design's "test that its default is 1.5", which was missing).
2. **Batch-1 replaces the reversed repeat.** `repeat.jsonl`, `REPEAT_ROWS`, `REPEAT_NOISE_LIMIT`, `repeat_rows`,
   `run_repeat`, `repeat_changes` and the reversed order are gone. `batch1_rows(units)` is question 0 for every unit
   in `units163.json` order; `run_batch1` generates each as its own call through `run163.pushes_for` on a one-row
   list, appending to `results/part200/batch1.jsonl`. `batch1_changes` counts differing answers per arm and total;
   `SHAPE_SENSITIVE_LIMIT = 20`, `shape_sensitive` true only when all 200 are present and more than 20 differ (a flag;
   it does not change a verdict). `--compare` and the run's summary both print it.
   Tests: `test_batch1_rows_are_question_0_for_every_unit_in_units_order`,
   `test_batch1_writes_200_one_row_calls_in_units_order_and_counts_changed_answers` (200 calls of one row, in units
   order), `test_shape_sensitive_flag_is_more_than_20` (20 → false, 21 → true).
3. **No second model load.** `EagerLoader` (returned by `eager_loader(args)`) loads once, applies Part 163's no-grad
   and eval, refuses non-eager with exit 3, and keeps `(tok, model)`; a second `__call__` returns the same pair.
   `_run_batch1_after` calls that same loader and takes `layers` from `common.find_layers(model)` on the same model.
   One `load.json` entry per launch, and the batch-1 calls are logged into that entry (`stage: "batch1"`).
   Tests: `test_loader_loads_once_and_keeps_the_model_it_loaded`, `test_run_loads_the_model_once` (asserts
   `load_model` was called exactly once and `load.json` has one entry).
4. **Resume.** The batch-1 stage runs on every launch after `run163.run` returns, skipping units already in
   `batch1.jsonl`, independent of whether `run163.run` wrote rows. Tests: `test_batch1_is_resumable` (199 pending
   skipped 1; a second call writes 0 and the file is byte-identical), and
   `test_run_finishes_batch1_on_a_relaunch_that_wrote_no_rows` (a stand-in `run163.run` returning
   `answers_written: 0` still gets 200 batch-1 rows; a relaunch adds none).
5. **Gate R3 exactly 8.** `bad_pushed` now fails any observed chunk `!= 8` (was `< 1 or > 8`). Batch-1 calls are
   pushed one-row calls, so they are tagged `stage: "batch1"` and `call_shapes` reads them with the no-push calls as
   single rows; battery/recipe calls are `stage: "run"`. A completed call's chunks must cover its rows; an incomplete
   call may cover fewer, never more. Tests: `test_gate_r3_requires_exactly_8_rows_a_7_row_chunk_fails` (a 7-row chunk
   that still covers 64 rows, `[7,8,8,8,8,8,8,8,1]`, fails), the 9-row-covering-64 case
   `[8,9,8,8,8,8,8,7]` kept in `test_gate_r3_reads_the_observed_chunks` and `test_check_record_…`, and
   `test_gate_r3_reads_a_batch1_one_row_call_as_single_row`.
6. **Lock file `run200.lock`.** `Run200Lock(run199.FolderLock)` overrides only `self.path` (`LOCK_NAME =
   "run200.lock"`); the messages are Part 199's (they say "run199"), which the brief allows. Test:
   `test_the_lock_file_is_named_run200_lock`, and `test_folder_lock_is_run200_lock_and_refuses_a_second_launch`
   (a stale `run200.lock` → exit 3; no `run199.lock` is created).
7. **`--compare` split.** Output is `{"scored": …, "exploratory": …}` in `compare.json` and printed under two
   headings. Scored: record R1/R3, Gate R2, Part 163's gates, the five verdicts with estimates and intervals, broken
   counts per arm over 1,600, at-the-cap counts per arm (`token_count == 80`), the Part 161-rule Qwen counts, the
   batch-1 differing counts with the shape-sensitive flag, and `verdicts_changed`. Exploratory:
   `mid_phrase_under_cap` (published and rerun; `mid_phrase` imported from `part199/report199.py` — the local copy is
   gone), `identical_battery_answers`, `identical_no_push_answers` (16), `per_layer_broken` (published and rerun).
   Self-check kept and extended: `test_compare_on_part163_against_itself_reproduces_the_published_numbers` asserts
   19/539, 57/661, 343/83 broken, 124/218 at the cap, and now the published mid-phrase baseline 510 of 1,476 neuron
   and 409 of 1,382 random under-cap answers (computed first by hand from `results/part163/answers.jsonl` with
   `report199.mid_phrase`, then asserted), plus 16 of 16 identical no-push answers and that no mid-phrase value
   appears in the scored section.
8. **Pilot stops.** Both `SystemExit` paths are recorded with status `stopped` and the wrapper exits 3. Test:
   `test_run_records_a_pilot_stop_and_exits_3`, parametrised over the rate message and the Gate 1 message; it also
   asserts no batch-1 stage runs after a stop.
9. **`steps.txt`.** `report199.py` is imported only inside `--compare` (verified: no `part199.report199` in
   `sys.modules` after importing `run200`), so the `--copy` list is unchanged; a comment says so. The three host
   lines are kept and their comments updated for the new Gate R3 wording, the two compare sections and `run200.lock`.

### Sabotage checks (each turned red, then restored; `cmp` against the backup confirmed each restore)

| Change | Sabotage | Result |
| --- | --- | --- |
| 2 | `chunked_generate` passes every pushed call through unchunked | 7 red |
| 3 | `EagerLoader.__call__` no longer returns the remembered pair (reloads) | 2 red (`test_loader_loads_once_…`, `test_run_loads_the_model_once`) |
| 4 | resume skip removed (every unit always pending) | 2 red (`test_batch1_is_resumable`, `test_run_finishes_batch1_…`) |
| 4 | batch-1 stage gated on `answers_written > 0` (the old build's rule) | 1 red (`test_run_finishes_batch1_…`) |
| 5 | `bad_pushed` back to `size < 1 or size > CHUNK` | 3 red (`test_gate_r3_reads_the_observed_chunks`, `test_gate_r3_requires_exactly_8_…`, `test_check_record_…`) |
| 5 | batch-1 calls treated as pushed chunks of 8 | 4 red (incl. `test_gate_r3_reads_a_batch1_one_row_call_as_single_row`) |

One sabotage initially passed for the wrong reason, and the test was fixed rather than the code: my first 7-row case
`[7,8,8,8,8,8,8,8]` sums to 63, so the "chunks cover its rows" rule failed it and the exactly-8 rule was never
exercised. It is now `[7,8,8,8,8,8,8,8,1]` (64 rows, one 7-row chunk). Same lesson as the build's Gate R3 finding.

### Test counts

| Command | Result |
| --- | --- |
| `python -I -m pytest -q part200/test_part200.py` | **35 passed** |
| `python -I -B part200/test_part200.py` | **35 passed**, exit 0 |
| `python -I -m pytest -q part163/test_part163.py` | **8 passed** (was 7; +1 default-1.5 test) |
| `python -I -B part163/test_part163.py` | **8 passed** |
| `python -I -m pytest -q part199/test_part199.py` | **38 passed** |

### Where the brief and the design disagree, or something could not be done as written

1. **Gate R3 and batch-1 calls.** The design says "no-push calls and batch-1 calls are single rows"; batch-1 calls are
   *pushed* one-row calls, so without a marker Gate R3's pushed-call rule (exactly 8) would fail all 200. The call log
   therefore carries `stage` ("run" or "batch1") and `call_shapes` reads batch-1 calls with the no-push calls. Records
   written by the pre-fold build have no `stage`; they are read as "run", which is correct for them.
2. **`--compare` requires `batch1.jsonl` of the rerun** (exit 2 if missing), because the scored section reports its
   counts. Part 163's folder has no batch-1 file; it is still required of the rerun only, so the published side is
   unchanged.
3. **`_run_batch1_after` runs even when `run163.run` raised** nothing — it is called only after `run163.run` returns;
   on a `SystemExit` stop it is skipped (no batch-1 rows after a pilot stop), which is what the pilot-stop test pins.
   The design does not say whether a stopped launch should still do batch 1; I chose no, since a stop means "no
   reading from this launch".
4. **Lock messages** still say "run199" (Part 199's `__enter__` text). The brief allowed either; only the path changed.

SUMMARY: Folded all nine changes of BRIEF-200-fold-r1 into part200/run200.py, test_part200.py and steps.txt, plus one
test in part163/test_part163.py. Pilot 0.17; batch-1 stage (200 one-row calls, units order, resumable, on every
launch) replaces the reversed repeat; EagerLoader loads once and the batch-1 stage reuses it; Gate R3 is exactly 8
with batch-1 calls as single rows; lock is run200.lock; --compare is scored/exploratory with mid_phrase imported from
part199/report199.py. 35/8/38 tests pass (pytest and script); 6 sabotage checks red then restored.
FINDINGS: Batch-1 calls are pushed one-row calls, so the call log needed a `stage` tag for Gate R3 to read them as
single rows. The published mid-phrase baseline (510/1,476, 409/1,382) was computed by hand and is now asserted.
BLOCKERS: none.
