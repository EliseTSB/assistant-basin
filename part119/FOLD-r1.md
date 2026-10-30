# Part 119 code review fold r1

## SUMMARY

The reader sample now has a seeded interleaving and opaque output ids.
Private reader keys are stored separately and joined by the scorer.
No-flip Path 2 rows remain visible and count toward coverage.
Retries now carry a JSON-only instruction, cap-hit flag, and retry count.
Sample reads require the reader's passing known-example gate.

## FINDINGS

### F1 — reader blindness

Before the change, the sample had three class transitions and ended in four class blocks. Its public label metadata included row id, class, row type, and X/Y unit ids. The seeded RNG now shuffles the built sample. Labels contain an opaque item id and the three answers. The sampling step writes row id, class, row type, and X/Y units to a separate key file. `score119.py` joins keys by item id. The test checks shuffled order, prompt and output contents, and the scorer join: `test_sample_labels_use_opaque_ids_and_private_keys_are_joined_by_scorer`.

### F2 — no-flip Path 2 coverage

Before the change, a no-flip unit wrote empty `positions` rows without recording why, and 119.3 required all 1,482 real rows to have positions. Rows now carry `p_A`, including `null` for a no-flip unit. The scorer lists those unit ids and reports real rows with and without positions in 119.3 and effect-matched output. 119.3 uses the median over rows with positions when no more than 5% of the 1,482 rows lack positions. Tests cover the retained rows, one no-flip unit's 38 rows, the 5% boundary, matching effect-matched counts, and a missing Path 1 row: `test_run_interactions_keeps_rows_for_a_no_flip_unit`, `test_no_flip_path2_rows_are_counted_and_one_unit_still_reads`, and `test_one_missing_logit_row_fails_gate_three_and_withholds_119_3`.

### F3 — reader retry behavior

Before the change, attempt two sent the same prompt, cap hits were only implicit in `stop_reason`, and the scorer's last-row rule could let an error replace a valid label. Retries now append a JSON-only instruction. Usage records `cap_hit`; scoring prefers valid labels and reports duplicated opaque ids as retry items. The test covers prompt change, cap tracking, valid-row preference, and retry count: `test_retry_instruction_cap_usage_valid_preference_and_retry_count`.

### F4 — known-example gate order

Before the change, the sample CLI returned success with no gate file. It now checks the selected reader's gate file and refuses a missing or failing gate before loading sample answers. The test confirms the refusal message and that sample loading does not start: `test_sample_read_refuses_missing_reader_gate`.

### F5 — pilot reuse instructions

The old notes told operators to use separate output folders but did not explain how main resumes pilot rows. The notes now give commands for batch 64 and 128 in separate folders, then set main's `--out` to the selected pilot folder. The other timing folder is not reused. The documentation test is `test_build_notes_document_two_pilot_dirs_and_main_reuse`.

### F6 — regression coverage

Tests now check class interleaving and output privacy, a missing logit row and Gate 3 completeness, the Path 1 prompt offset, and two-layer pushes against independent test-local hooks. The relevant tests are `test_pilot_and_reader_seeds_are_stable`, `test_sample_labels_use_opaque_ids_and_private_keys_are_joined_by_scorer`, `test_one_missing_logit_row_fails_gate_three_and_withholds_119_3`, `test_path1_uses_the_last_prompt_logit_for_the_first_answer_token`, and `test_torch_two_layer_row_hook_matches_two_single_layer_hooks`.

### Mutation checks

Each mutation was a single-line change in its own disposable copy under `$TMPDIR`. All five targeted tests went red.

| Break | Test that went red | Result |
|---|---|---|
| Path 1 logit-to-token offset | `test_path1_uses_the_last_prompt_logit_for_the_first_answer_token` | Red; shifted slices had incompatible lengths. |
| Additive formula sign | `test_additive_prediction_subtracts_the_none_logits` | Red; wrong top token. |
| Margin winner comparison | `test_winner_take_all_and_row_rates_use_only_contested_positions` | Red; exact tie incorrectly chose A. |
| Robustness `p_A` subtraction | `test_robustness_flip_alpha_margin_and_no_flip_unit` | Red; expected `1/23`, got `2/23`. |
| Effect-matched percentile filter | `test_effect_matched_fit_and_support_coverage_fixture` | Red; unsupported position changed the fitted residual. |

### F7 — left alone

The torn-last-line rewrite in `read_existing` is unchanged, as requested.

## BLOCKERS

The 119.3 5% no-position rule is a build judgment for Claude to register in `PART119-DESIGN.md`. This worker updated `BUILD-NOTES.md` and left the design file untouched. No other blockers.
