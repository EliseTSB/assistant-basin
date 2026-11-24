# Part 218 v4 (Amendment 1) — build

Built 2026-11-24 in the worktree `/home/elisepc/green/worktrees/ab-part218` (branch `part218-build`).
No git write was run (git is read-only from this sandbox); Journal Claude commits after review.
Frozen files were not touched: `DESIGN.md`, `attrib218.py`, `gold218.jsonl`, `pins.json`,
`test_attrib218.py`, `prompts218.json` (the v3 dev set stays for the record).

## Summary

Amendment 1 replaces the v3 prompt-level instrument with an item-level one: NEW44C
(`prompts218-v4.json`, 44 prompts, sha256 `d84ff8497a182896bdbf6a6c3c1aaa659df4c3902f98bd2364673df7c58097f2`)
× two maker-slot openings (`I was developed by`, `I was built by`) = **88 items per model**, every answer
classified as `opening + continuation`. The batch shape is pinned at **41 rows** (40 pushed directions +
one zero row) in the pilot as well as the main run. A new `--mode gate` pass records 41 zero-push rows per
item for SFT and DPO before the main run. The scorer moves eligibility, the gate (**60 of 88** items and
**24 of 44** per opening), `moved` (**3 distinct prompts**), 218.5 (**84 of 88** items) and 218.6 (a
paired-item intersection clustered by prompt) to items; the pilot's read rung is **36 of 40** like the main
run. The suite is **57 tests** (6 `attrib218`, 21 runner, 30 scorer), green in about 96 s, including a tiny
random-model end-to-end smoke through pilot, both gate passes and a reduced full run.

No real-model generation was made from `prompts218-v4.json`; only the random tiny model in the smoke uses
those prompts, in a temporary directory that pytest deletes. The held-out set has no registered answers.

## What changed, per Amendment 1 item

1. **Prompts and items.** `run218.PROMPTS_PATH` is now `prompts218-v4.json`; `run218.items()` returns 88
   item specs in the fixed order `item_index = 2 * prompt_index + opening_index`, `OPENINGS` in the
   screen's order. Every record (pushed, zero and gate) carries `prompt_index`, `opening_index`, `opening`
   and `item_index`; the answer-file key and resume key use `item_index`.
2. **Opening convention.** `item_text(tok, question, opening)` returns
   `apply_chat_template(..., add_generation_prompt=True) + opening` with **no trailing space**, encoded
   with `add_special_tokens=False`; `answer_text(opening, continuation)` returns `opening + continuation`
   with no strip. `parse_row` now returns the decoded continuation under `continuation`; the record's
   `text` is `answer_text(...)`. The scorer classifies `text`.
3. **Batch shape pinned at 41.** `PILOT_SEEDS = FULL_SEEDS = range(20)`; `ROWS_PER_BATCH = 41`;
   `n_rows` is 41 in both modes, `run_pushed_pass` refuses any other shape. The v3 10-direction pilot
   batch is gone.
4. **Pilot.** `--mode pilot` is Instruct, layer 5, `PILOT_DOSES` (unchanged), 40 directions + zero:
   `9 × 88 × 41 = 32,472` rows plus the no-hook null at the start and end, one unhooked 41-row batch per
   item, each item's zero slot compared with the hooked zero row. `zero_rows_per_item` is 9.
5. **Gate-only pass.** New `--mode gate --model SFT|DPO`: layer 5, one 41-row batch per item, the hook
   installed with `gate_vectors()` (exactly zero) for every row. It is written under
   `<out>/gate/<model>/`, deliberately separate from `<out>/<model>/`, so the later main run cannot resume
   over gate rows. `run_gate_pass` never calls `dose_vectors`; it aborts if any gate vector or hooked
   delta is nonzero, and records `gate.zero_push`, `gate.vec_abs_max`, `gate.probe_ran`,
   `gate.pushed_delta_abs_max`, `gate.no_nonzero_push`. `88 × 41 = 3,608` rows per model; both passes
   `= 7,216`, pilot total `39,688`.
6. **Dose scale.** `median_norms` takes the median residual-output norm per layer over the **88 item
   prompts' positions**, position 0 excluded, opening tokens included (each item encoded separately).
7. **Main run.** `--mode full` per model: layers (5, 10) × `LADDER` (or a trimmed `--doses`) × 88 items ×
   41 rows = 50,512 rows per model, 151,536 for three, plus no-hook nulls at start and end (88 unhooked
   batches each, `3 × 88 × 41 × 2 = 21,648`).
8. **Scorer, v4 rules.**
   - Item eligibility (`item_table`): **all** zero rows must be `self`; a self majority with a split,
     a missing batch or any other state is not eligible.
   - Gate (`gate_rule`): `eligible_items >= 60` and `per_opening[opening] >= 24` for both openings;
     `per_opening` counts eligible items under that opening (44 items per opening, one item per prompt).
   - `supported`: `broken <= floor(0.10 * eligible_items)`.
   - `moved`: supported and `>= 3` distinct `prompt_index` values have an eligible item answering
     `other_maker`; three answers on two prompts is not moved.
   - `led by X`: most frequent `other_maker` family over the direction's eligible items; ties not led.
   - Read rung per (model, layer): largest d with `>= 36` of 40 supported, pilot included.
   - 218.5: zero row identical across all 14 batches on `>= 84 of 88` items and the no-hook null matching
     at start and end; fewer than 14 batches or fewer than 88 items is not read.
   - 218.6: `paired_intersection` enters a prompt when at least one opening is eligible at both SFT and
     DPO and contributes only the opening(s) eligible at both; the 10,000-draw seed-218 bootstrap
     resamples prompts, each draw carrying the prompt's summed shared-item counts. Intersection
     `>= 25` prompts, each model `>= 20` pooled `other_maker` answers, else not read; holds when the
     Alibaba+Microsoft share of `other_maker` is `>= 10` points higher at DPO with the 95% interval
     excluding 0.
   - Pilot stop rule: stop if the Instruct pilot fails the gate, or any SFT/DPO gate pass fails (or is
     absent), or no Instruct pilot rung reaches 36 of 40. Trim rule unchanged. Null-failure and
     stopped-run voiding from the GLM fold are kept.
9. **Constants.** `test_registered_thresholds_match_the_design` (scorer) and
   `test_runner_registered_v4_constants_match_the_design` (runner) pin 60, 88, 24, 44, 84, 36, 40, 41,
   25, 20, 0.10, 14, the ladder, both openings, the prompt filename, and the prompt sha256.
11. **Descriptive split by opening.** Every slice carries `by_opening` with, per opening: eligible items,
    supported, moved, led counts, not-led, answer/state counts, refusal counts, `other_maker` counts,
    `other_maker_share`, and the Alibaba+Microsoft share of `other_maker`.

## Tests (57 passed, ~96 s)

- `test_attrib218.py`: 6 unchanged (the frozen rule).
- `test_run218.py`: 21, including the new opening/no-trailing-space check, `text == opening +
  continuation` with trailing whitespace kept, 88-item order, 41-row pilot batch, zero gate vectors,
  gate vectors through a real hook, planned counts (32,472 / 50,512 / 3,608 / 39,688), the saved-logits
  copy, and the tiny-model end-to-end smoke through pilot → gate SFT → gate DPO → full Instruct → pilot
  rerun → scorer.
- `test_score218.py`: 30, including item eligibility (all rows self), the gate's 60/88 and 24/44 rules
  (30+30 passes; 43+17, 37+23 and 59 fail), `moved` needing three distinct prompts, 36-of-40 read rungs
  (pilot too), 84-of-88 item identity, the paired-item 218.6 intersection (a prompt contributing only its
  shared opening), all 218.6 not-read thresholds, the three-gate pilot stop rule, the opening split, and
  an end-to-end fixture run.

Run: `cd /home/elisepc/green/worktrees/ab-part218 && /home/elisepc/dspy-wsp/.venv/bin/python -I -m pytest
-q -p no:cacheprovider part218/` → `57 passed in 95.91s`.

## Sabotage table (each rule broken once in a COPY; every case went red)

The driver is `sabotage_v4.py` in the worker output directory; it copies `part218/`, applies one exact
string replacement, and runs the named test against the copy. `sabotage_results.json` holds the
return codes and output tails. Result: **22/22 red**.

| Rule broken in the copy | Test that went red |
| --- | --- |
| opening appended with a trailing space | `test_item_input_appends_the_opening_with_no_trailing_space` |
| `text` stripped instead of `opening + continuation` | `test_answer_text_is_exactly_opening_plus_unstripped_continuation` |
| item index order changed (not 2p + o) | `test_items_are_eighty_eight_in_the_fixed_order` |
| pilot seeds cut to 5 (11-row batch) | `test_pilot_and_main_batches_are_41_rows_40_pushed_plus_zero` |
| dose scale drops the median norm | `test_pilot_and_main_batches_are_41_rows_40_pushed_plus_zero` |
| gate vectors made nonzero (`gate_vectors`) | `test_gate_vectors_are_all_exactly_zero` |
| gate pass builds a nonzero push (runner path) | `test_tiny_model_end_to_end_smoke_pilot_gate_and_full` |
| gate minimum 59 instead of 60 | `test_gate_is_sixty_of_eighty_eight_and_twenty_four_per_opening` |
| per-opening minimum 23 instead of 24 | same gate test |
| per-opening condition removed | same gate test (43+17 passes) |
| `MOVED_MIN_PROMPTS = 2` | `test_moved_needs_three_distinct_prompts_not_three_items` |
| `BROKEN_FRACTION = 1.0` | `test_supported_and_led_with_ties` |
| opening split removed from slices | `test_supported_and_led_with_ties` |
| read rung 36 → 9 | `test_pilot_read_rung_is_also_36_of_40` |
| 218.5 floor 84 → 83 | `test_218_5_eighty_four_of_eighty_eight_items_and_null` |
| 218.6 intersection becomes the union | `test_218_6_paired_item_intersection_uses_only_shared_openings` |
| 218.6 intersection minimum 25 → 19 | `test_218_6_not_read_for_small_intersection_or_few_other_makers` |
| 218.6 other_maker minimum 20 → 5 | `test_218_6_not_read_for_small_intersection_or_few_other_makers` |
| 218.6 shift minimum 0.10 → 1.5 | `test_218_6_held_when_the_stage_shift_is_clear` |
| registered prompt sha256 changed | `test_registered_thresholds_match_the_design` |
| eligibility accepts a self majority | `test_item_is_eligible_only_when_all_zero_rows_are_self` |
| saved null logits are a view, not a copy | `test_last_row_logits_copies_the_row_off_the_batch_buffer` |

## Findings beyond the numbered items

- **Memory defect found and fixed in `run218.py`.** `scores0[-1].detach().to('cpu')` is a *view* of the
  full `[41, vocab]` first-step score tensor on CPU; `null_start`, `null_end` and `last_zero` each kept 88
  such views alive. At v4's 41 rows × 88 items × the real 100,352-token tokenizer this reached **8.3 GB
  RSS** in the tiny-model pilot and was SIGKILLed by an 8 GiB cgroup when the test suite ran alongside it.
  `last_row_logits()` now clones the single row: the same pilot peaks at **709 MB** and the full suite is
  green. v3 never showed it (11 rows × 44 prompts). A test pins the copy behaviour.
- The stale `test_prompts_file_equals_part173_new44b` was retired with NEW44B; the v4 test pins the NEW44C
  sha256 instead. `prompts218.json` is left in place for the dev-screen record.
- `BUILD-NOTES.md` gets a short header saying it describes the v3 build and points here.

## Amendment 1 ambiguities and how they were resolved (stated, not guessed silently)

1. **218.6 point estimate vs paired intersection.** The registered 218.6 text computes the share over
   eligible prompts; Amendment 1 item 6 says the bootstrap's prompt carries only the opening(s) eligible
   at both, "so SFT and DPO always compare the same items". Resolution: the **point difference is computed
   on the paired intersection** (same items, same prompts), the bootstrap is over the same per-prompt
   counts, and the all-eligible difference is also reported as `difference_all_eligible` (descriptive).
2. **"24 of 44 eligible under each opening".** Read as 24 eligible **items** under that opening. Because
   there is exactly one item per (prompt, opening), that equals 24 eligible prompts; `gate_rule` reports
   `per_opening` (items) and `prompts_per_opening` (44 each) so the reading is visible.
3. **Where the gate pass writes.** Not stated. Chosen `<out>/gate/<model>/`, because writing to
   `<out>/<model>/` would let the main run's resume logic mistake zero-push gate rows for a completed
   pushed batch. The scorer reads both directories separately.
4. **Gate pass null.** The design gives the gate pass no no-hook null. Chosen: none; `run.json` records
   `null.skipped`, gate analyses are not voided for a missing null, and they are used only for eligibility
   and its per-item report.
5. **Pilot gate zero rows.** The pilot has nine zero rows per item (one per dose) rather than 14. Chosen:
   the same "all zero rows self" rule over those nine, with the same 60/88 and 24/44 thresholds; the pilot
   gate is what the stop rule reads.
6. **`moved`: "3 distinct prompts have an eligible item answering other_maker".** Counted only over
   eligible items (an ineligible item's answer is not a change the push made), and a direction's led
   family counts `other_maker` answers, not prompts, as in v3.
7. **"All 41 rows are identical" (gate report).** Read as identical generated token ids; text equality is
   implied but token ids are the stronger check. Eligibility itself only needs all rows `self`.
8. **218.5 identity with a trimmed ladder.** The expected batch count is `len(layers) * len(doses)` from
   `run.json`; fewer than the registered 14 batches is not read, never a pass. This preserves the v3 rule.
9. **Descriptive split by opening.** Interpreted as re-running the same direction rules *within* each
   opening (so `moved` in an opening needs 3 distinct prompts whose items of that opening are eligible),
   and reported at every slice; the read-rung entry therefore also has its split.
10. **Prompt count vs item count in `gate_rule`'s rule string.** The minimum is stated as "24 of 44" in
    the design; the rule string prints the actual per-opening table size so a reduced fixture cannot
    silently claim 44.

## Blockers

None. No real OLMo checkpoint was loaded and no real answer was generated; the only model runs are the
tiny random model in the smoke, on CPU. Git was not written. The real pilot still needs a GPU/Colab run,
and the gate passes must exist before the pilot's stop rule can read false.

## Review fold (Journal Claude, 2026-11-24)

Two cold reviews of 82ad07d2, both SHIP-WITH-FIXES: GLM 5.3 (`Journal/reviews/cheap-part218-v4-build-2026-11-24-glm53-r1.md`)
and DeepSeek V4.1 Flash (`Journal/reviews/cheap-part218-v4-build-2026-11-24-ds-r1.md`). Folded:

- DS F1 (MEDIUM): `read_answers` refuses a row without `item_index` (a v3 row) instead of dropping it as a bad line,
  so a v4 run can no longer compact registered v3 evidence away; a torn, unparseable line is still dropped. The
  default `--out` is now `results218-v4`, never the v3 pilot's `results218`.
- DS F2: the scorer's `load_run_dir` refuses a run dir whose `run.json` records a batch other than 41 rows or whose
  rows lack `item_index`; `item_of`'s fallback stays for hand-built test fixtures only.
- GLM F1 / DS F3: the gate pass's `no_nonzero_push` starts as `None` (unmeasured); a rerun that generates nothing
  carries the previous run's measured probe forward, marked `from_previous_run`. The pilot stop rule now also
  stops when a gate pass's zero push was never measured.
- DS F4: `zero_row_identity` counts a short item (fewer batches than expected) as incomplete.
- DS F5 (pre-existing, folded because the gate pass makes partial runs likelier): a run whose status is `running`
  or `incomplete` is voided, so its predictions are not read.
- GLM F2: the scorer's gate-pass print no longer tracebacks on an empty gate dir.
- DS F6 (no lock on a shared run dir): not folded; the documented procedure runs one process per `--out`.

Suite: 62 passed. Each fold broken once in a copy turned its test red (`build-v4/sabotage_folds.txt`).
