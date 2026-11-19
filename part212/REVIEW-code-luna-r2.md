<!-- cheap-review.mjs cheap-review-part212-code-luna-r2-luna-r2-2026-11-19: /home/elisepc/green/worktrees/ab-neon-review 84472af3..HEAD, lane luna, df-run exit 0 -->
SHIP-WITH-FIXES

## Findings

### MEDIUM — INTRODUCED: Stage 2 batch metadata is invocation-scoped and resume can mix batch sizes

- **Files:** `part212/run212.py:1109,1378-1379,1394`; `part212/score212.py:652`.
- **MEASURED triggers:**
  1. `evidence_resume_hist.py` imports the real runner, plans the full run at batch 24, then resumes with all planned full rows already written:
     ```
     planned_rows= 3888
     first_invocation_histogram= {'1': 1440, '4': 36, '8': 264, '24': 8}
     fully_resumed_todo_rows= 0
     resumed_run_json_histogram= {}
     row_batch_fields= ['arm', 'pilot', 'question_index', 'sample', 'set']
     ```
  2. `evidence_batch_resume.py` simulates full rows written at batch 8, then inspects a full run at batch 24:
     ```
     batch=8 hist={'1': 1440, '4': 36, '8': 288}
     batch=24 hist={'1': 1440, '4': 36, '8': 264, '24': 8}
     todo_after_batch8_full_rows= 0
     scorer_keeps_rows_with_no_batch_size= 3888 dropped= 0
     runner_row_batch_field_present= False
     new_run_json_batch_size= 24 histogram= {}
     ```
  3. `evidence_partial_batch_resume.py` leaves three rows from the first plan batch, then rebuilds remaining batches with the same batch option:
     ```
     original_batch_key= ('unchanged', 0, None)
     original_batch_size= 8 partial_rows_saved= 3
     same_batch_size_option=24, resumed_first_batch_size= 5
     resumed_first_batch_samples= [3, 4, 5, 6, 7]
     resumed_histogram= {'1': 1440, '4': 36, '5': 1, '8': 263, '24': 8}
     ```
- **Why it matters:** Stage 2 keys rows by `(arm, question, sample, set, pilot)` but not `batch_size`; JSONL rows also contain no batch size. A full run resumed with another `--batch` silently reuses old rows, and `score212` filters only on `pilot`. A partial batch is also regrouped on resume. `run.json` records only the latest invocation's batch size and histogram, not the shapes that produced every answer. This defeats Amendment 1's stated after-the-fact batch-dependence check, including on restart.
- **Smallest fix:** persist batch size per answer and include it in resume validation; reject mixed shapes or preserve a complete per-row shape record and aggregate histograms across saved rows.

### MEDIUM — PRE-EXISTING: Gate 3 checks row count, not that every planned key exists

- **File:** `part212/score212.py:386-399`.
- **MEASURED trigger:** `evidence_gate3_keys.py` starts from the real 3,888-row plan, replaces one planned `unchanged` guard key with an unplanned `question_index=31`, and calls the real `score212.gate3` with otherwise passing synthetic metadata:
  ```
  planned_rows= 3888 actual_unique_keys= 3888
  missing_planned_keys= 1 unplanned_keys= 1
  gate_every_planned_key_present= True
  gate3_pass= True
  ```
- **Why it matters:** Gate 3's `every_planned_key_present` check is only `len(keys) >= planned`; duplicate detection does not catch a unique unplanned row replacing a missing planned row. This scorer logic predates the current fix pass and is a follow-up, not a reason to hold this delta.
- **Smallest fix:** compare actual unique row keys to a persisted or reconstructed exact planned-key set, while retaining duplicate detection.

### LOW — INTRODUCED: two new F3 tests and the smoke's random-arm check bypass the changed driver path

- **Files:** `part212/test_part212.py:1366,1394,1405`; `part212/smoke_tiny212.py:131`.
- **MEASURED trigger:** the current `test_F` plus `test_target_down_columns_select` selection against baseline `84472af3` collected 16 tests (13 new tests plus 3 existing tests). Results include:
  ```
  test_F3_random_set_trace_arm_clamps_each_pool_units_own_trace PASSED
  test_F3_stage1_builds_each_arms_rec_from_its_own_units FAILED
  test_F3_random_set_realized_perturbation_is_its_own_units_write PASSED
  11 failed, 5 passed, 58 deselected
  ```
  The two passing new tests supply a synthetic `rec` that already contains the pool units' values or call the pure norm helper; the old defect was in `stage1()` choosing `target_cols` before calling those helpers. The smoke similarly creates `arm_rec` from `arm_cols` before calling `stage1_values`, bypassing the Stage 1 driver loop.
- **Why it matters:** the implementation change is in the Stage 1 driver, but two added tests and the smoke assertion can pass without exercising that changed path. The new source-text driver test goes red on the baseline but verifies spelling, not runtime behavior. The new target-down-column test also fails on the baseline, but covers a separate helper.
- **Smallest fix:** add a driver-level test that gives target and pool units distinct recorded values and observes actual Stage 1 writes. Keep helper tests as unit tests, but do not count them or the current smoke assertion as regression proof for F3.

### LOW — PRE-EXISTING: the design's 3,989 Stage 1 count has no corresponding registered pass

- **Files:** `PART212-DESIGN.md:281-282,462-464`; `part212/run212.py:420`.
- **MEASURED trigger:** `evidence_stage1_count.py` imports the real arm plan and histogram helper:
  ```
  main/per-unit/random arms= 131 15 16 100
  pool_norm_pass EN_MAKER heldout= 24
  record_pass banks x prompt rows= 5 x 25 = 125
  monitor baseline forwards= 25
  intervention forwards= 131 x 25 = 3275
  runner_histogram= {'1': 3449} total= 3449
  design_count= 3989 unaccounted= 540
  ```
  `git diff 4c5b4027..c34820ee -- PART212-DESIGN.md` shows the Size count first changed from 3,917 to 3,989 in the v2 design draft (+72). That patch adds one main arm, which accounts for 25 extra arm forwards, but the other 47 are not accounted for; the listed current passes sum to 3,449, as the runner does.
- **Conclusion / why it matters:** the extra 540 is not a missing registered runner pass; 3,989 is a pre-existing design arithmetic error with no traceable source in the listed Stage 1 plan. It overstates the workload in the estimate; it does not justify running an unregistered pass.
- **Smallest fix:** correct both Size and Pilot counts to 3,449, or name and authorize the missing 540 forwards in the design before implementation.

### LOW — PRE-EXISTING: concurrent Stage 2 invocations can both claim the same plan

- **File:** `part212/run212.py:1319,1326,1331`.
- **MEASURED trigger:** `evidence_concurrent_resume.py` drives the real `stage2_rows` and `stage2_todo` from two threads after both observe the same empty output snapshot:
  ```
  request_A_todo= 3888 request_B_todo= 3888
  overlapping_generated_keys= 3888
  same_outdir_lock_in_runner= False
  ```
- **Why it matters:** two Stage 2 requests using the same `--out` can both generate the full plan and append duplicate keys (or interleave JSONL writes); the scorer rejects duplicates, so the shared run becomes unusable. This no-lock behavior predates the current fix pass and is a follow-up.
- **Smallest fix:** acquire an output-directory lock before reading `stage2.jsonl` and hold it through append and metadata completion, or refuse a second active run for that directory.

### LOW — PRE-EXISTING: a partial JSONL tail is skipped on resume but still crashes scoring

- **Files:** `part212/run212.py:1319-1331`; `part212/score212.py:648-649`.
- **MEASURED trigger:** `evidence_corrupt_jsonl.py` writes one valid row plus an incomplete final JSONL row, runs the runner's same resume parsing loop, then calls the real scorer loader:
  ```
  resume_reader_valid_rows= 1 file_still_has_bad_tail= True
  scorer_load_error= JSONDecodeError Expecting value, line 1 column 32 (char 31)
  ```
- **Why it matters:** if a write or interruption leaves a partial final row, Stage 2 skips it while calculating resume work but appends without repairing the file. `score212.load_stage2` then raises on that line, so resumed output cannot be scored. This recovery gap predates the current pass and is a follow-up.
- **Smallest fix:** before resuming, truncate an invalid trailing record to the last valid newline (or write batches atomically); have the scorer report and reject corrupt rows explicitly.

## What I checked and found sound

- F1–F10 are reflected in the runner/scorer, not just tests: `zero` uses an all-true mask; Stage 2 plans 3,888 rows with 144 guard rows and 240 rows per full arm; Stage 1 random arms slice their own units; monitors and `g_seq` use T+4; resume keys and scorer filtering distinguish pilot rows; pilot stopping uses `clean_google`; the pool-match report now describes activation-weighted matching; and Gate 0 names `PART212_ALLOW_LENGTH_DRIFT=1`.
- The full current CPU suite passes: `74 passed in 4.70s`. Against the pre-review source, 11/13 new tests fail; two F3 helper-only tests pass, as noted above.
- Smoke assertion inventory: its direct hook checks verify Stage 1 and Stage 2 target `zero`; Stage 1 `random_0` checks `trace_resample_fact`, `sub_factual`, `scale_neg1`; Stage 2 `random_0` checks `trace_resample_fact`. The captures compare these selected writes to their supplied values and verify pool units. No smoke hook-write assertion covers the 16 per-unit arms, random sets 1–19, Stage 1's other main arms, or Stage 2 `mean_resample_fact`, `sub_factual`, scale arms, `trace_resample_you`, or diagnostic arms. The suite has unit tests for many helper modes, but the smoke is selective.
- I traced the Stage 1 metrics and Stage 2 row/run fields read by `score212.py` to the runner writers. No additional scorer/runner meaning mismatch was found beyond the batch metadata and Gate 3 completeness findings above.
- I did not rerun the approximately 20-minute tiny-model smoke; no real weights or live services were used.
