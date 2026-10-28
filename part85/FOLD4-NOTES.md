# Part 85 fold 4 notes

SUMMARY
Ran 37 tests in 25.470s
Mutation summary: 33/33 killed; 0 survived or errored

## Findings / what shipped

`part85.py` now writes one trace row for every answer to the matching file under the checkpoint's `trace/` directory. The trace fields come from these runtime measurements:

| Trace field | Measurement point |
|---|---|
| `question` | `_prompt_inputs` stores the actual question argument it receives in per-prompt state. |
| `prompt_text` | The tokenizer decodes `encoded["input_ids"][0]`, the IDs expanded into `model.generate`, immediately before the call. |
| `chat_template_applied` | The prompt builder starts false and sets true after `apply_chat_template` returns for that prompt. |
| `generate_kwargs` | The four registered sampling values are read from the same dictionary expanded into `model.generate`. |
| `hooked_module` | `model.named_modules()` is searched by object identity for the module the clamp hook is installed on. |
| `unit`, `set_value` | Read from the live clamp hook's `unit` and `value`; both are null for unclamped answers. |
| `prompt_positions_set`, `generated_positions_set` | A second pre-hook, registered after the clamp pre-hook, counts input positions matching the live clamp value at relative tolerance `1e-6`. Unclamped answers report zero for both. |
| `observed_min`, `observed_max` | The same observer measures the selected unit across all prompt and generated positions. For unclamped answers it measures the natural target-unit (6978) input while reporting null `unit` and `set_value`. |
| `other_channels_changed` | The clamp pre-hook compares its input and rewritten tensor outside the clamped unit for every call in that answer. |

Both JSONL outputs are completely serialized and fsynced to temporary files before atomic replacement. Resume only skips a cell when both answer and trace files exist and pass their row-count and identity checks. A missing trace causes the cell to be regenerated; a present but incomplete trace is rejected. The trace is inside the checkpoint's owned result directory, so the existing push includes it. Assembly already copies the complete owned directory; a handoff test now checks that the trace is present in assembled output.

The audit fixture uses the tiny OLMo model with 11 layers, intermediate width 7885, and the registered units 6978 and 7884. Its position limit is 128. A test-only output head emits one regular token followed by EOS, keeping the model's real forward path and hook active for both prompt and generated positions while making the full registered sample counts quick to run. `audit85.py` was not changed. Four end-to-end cases run `part85.main(argv)` and then the independent audit subprocess: pilot Instruct, plain main `pretrain_4T`, post-trained main Instruct with plain and chat frames, and pilot attempt 2 with its custom template. All pass with the registered 8/16 samples per question and 0.7/0.95/64 generation settings.

`CLOUD-RUN.md` now places audit commands after pilot assembly and before either pilot is scored, and audits every main checkpoint after assembly and before scoring. It retains the pre-launch mutation command and requires all 33 mutants to be killed.

## Final mutate85 table

| Mutation | Result |
|---|---|
| F2_M1_own_max_disabled | KILLED |
| F2_M2_chat_prompt_raises | KILLED |
| F2_M3_clamp_hook_noop | KILLED |
| F2_M4_wrong_answer_unit | KILLED |
| F2_M5_resume_regenerates_cell | KILLED |
| F2_M6_step1_guard_disabled | KILLED |
| F2_M7_cell_push_disabled | KILLED |
| F2_M8_preflight_skips_missing_refs | KILLED |
| F2_M9_assemble_accepts_foreign_tree | KILLED |
| R3_layer9_clamp | KILLED |
| R3_chat_skips_chat_template | KILLED |
| R3_custom_template_ignored | KILLED |
| R3_push_before_answer_write | KILLED |
| R3_step0_push_before_write | KILLED |
| R3_wrong_checkpoint_in_push | KILLED |
| R3_template_mismatch_accepted | KILLED |
| R3_own_max_is_9_75 | KILLED |
| R3_chat_seed_plus_1000 | KILLED |
| R3_own_max_absolute_swapped | KILLED |
| D1_push_stages_outside_directory | KILLED |
| D2_push_succeeds_when_remote_behind | KILLED |
| D3_restore_does_not_checkout_tip | KILLED |
| D4_assemble_accepts_foreign_file | KILLED |
| D5_step1_accepts_wrong_token_sha | KILLED |
| D6_cleanup_removes_snapshot_link | KILLED |
| P1_seed_for_first_sample_only | KILLED |
| P2_half_generation_budget | KILLED |
| P3_temperature_one_at_generate | KILLED |
| P4_double_value_at_clamp_hook | KILLED |
| P5_first_question_for_every_index | KILLED |
| F4_trace_question_copied_from_plan | KILLED |
| F4_observer_runs_before_clamp | KILLED |
| F4_resume_accepts_trace_missing_a_row | KILLED |

## What I could not verify

No pinned OLMo checkpoint was downloaded or run, and no real cloud push or assembled production pilot was available in this worker. The test suite exercises the run loop with the tiny local OLMo fixture and the handoff/assembly path with local temporary repositories.

## Blockers

None.

## Claude, on collection (2026-10-28 01:2x CT)

Host rerun: Ran 37 tests, OK; mutate85 33/33 killed, including Claude's P3 (temperature 1.0) and P5 (first question for every index), which survived fold 3. audit85.py unchanged.
