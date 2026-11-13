# Part 202: fold of the round-2 code review

Review: `reviews/cheap-part202-code-r2-ds-r2.md` in the Journal repo (DeepSeek V4.1 Flash, SHIP-WITH-FIXES), on
`0209d64a..e7053d98`. Folded 2026-11-12 08:55 CT, before registration and before any model ran. Each fix was preceded
by a test shown failing on the code before it.

| Finding | Disposition | Test that was red first |
|---|---|---|
| F1 replicate resume key | fixed: `run164.row_key` in `run_replicate_stage` | `test_a_replicate_resume_writes_a_row_whose_dose_twin_is_already_on_disk` (199 rows before) |
| F2 scorer R3 blind to a skipped call | fixed: `score()` passes `run202.plan_units(units)` to `record_gates` | `test_r3_in_the_scorer_fails_a_record_missing_one_planned_call` (R3 True before) |
| F3 short replicate accepted | fixed: `main()` exits 2 unless `--replicate` holds 200 rows | `test_a_short_replicate_is_bad_input` (exit 3 before) |
| F4 bf16 gate vs a list dtype | fixed: string or list read; a mixed load refused and named mixed | `test_the_bf16_gate_reads_load_records_dtype_as_a_string_or_a_list` (one-element list refused, mixed not named, before) |
| F5 torn replicate line | fixed: `run201._repair` before reading `replicate.jsonl` | `test_a_torn_replicate_line_is_repaired_on_resume` (ValueError before) |
| F6 row_id not unique within a pass | fixed: pushed row_id is `<unit_id>\|q<qi>\|<arm>\|<dose>`; replicate row_id is `rep\|` + it | `test_row_ids_are_unique_within_a_pass_as_part201s_rows_schema_requires` (3,316 distinct of 6,516 before); replicate test asserts 200 distinct ids (140 before) |
| F7 plan() docstring | fixed (wording only) | none (a docstring) |
| F8 slot and dtype unread by any test | fixed: the run test asserts `slot` 0..n-1 per call, `chunk_slot` i % 8, and the entry's dtype | red on two sabotages: chunk index written into `slot`; `dtype` dropped from the launch entry (`KeyError`) |

Evidence for F4's scope: Qwen3.5-4B's eager loads recorded in `results/part195/load.json` and
`results/part201/load.json` carry the single string `"torch.bfloat16"`, so the registered model loads uniformly; the
fix keeps a mixed load refused and says so.

Not changed: the review's follow-up that `score202.row_key` and the replicate match omit `arm` (safe while the
registered units file reuses no unit_id across arms, which the review checked).
