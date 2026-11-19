# Part 212 build report — runner, scorer, tests

## SUMMARY

- `part212/run212.py`: `stage1`, `stage2`, `pilot` subcommands, each with `--out` (relative, enforced) and `--device`
  (default `xpu`). Stage 1 runs the 131-arm matrix (15 main + 16 per-unit + 100 matched random) on 25 prompts; Stage 2
  plans 3,888 sampled generations, including the fluency guards; `pilot` runs Stage 1 + donor recording + the
  160-generation mini-pilot and applies the design's stopping rule.
- `part212/score212.py` (700 lines): reads only `stage1.json`, `stage2.jsonl` and `run.json`; computes Gates 1–3 (3b
  included), the 4-cell factorial interaction, the question-clustered bootstrap, the exclusive five-state tables and
  transition joins, both ranks, and the verdict of 212.1–212.10 under v3.1's rules; writes `score.json` and prints the
  verdict table.
- `part212/test_part212.py`: **74 passed** on the host CPU venv, no real weights, download, or XPU.
- The real model was **not** run. `results/` was not touched. Nothing was committed.
- The runner passed the intervention-asserting tiny random-model pilot smoke; its synthetic gates correctly prevent
  launch of full Stage 2.

## FINDINGS

### Design ambiguities and contradictions (line numbers are PART212-DESIGN.md v3.1, commit 5cc63247)

The design moved from v3 (`ef9aa7aa`) to registered v3.1 mid-build (parent control message seq 1, acted on). v3.1 is
wording-only: `clean_google` is fixed as the exclusive category everywhere, `legacy_any_google` is reported beside it,
and the 212.1/212.5 rank is the full arm's `count/192` against each random set's `count/24`, with the sample-0 rank
descriptive. The scorer implements that reading.

1. **L76 vs L98–102 — write norm vs hook site.** The hook clamps `down_proj`'s *input* (L76); the write norm is
   `mean(|a|) * ||W_down[:, u]||_2` (L99). `W_down` has `n_units` columns (the input width), so the literal reading is
   used. No change needed; recorded here because the two are easy to conflate.
2. **L181 — Stage 2 call-0 profile for `mean_resample_fact` / `sub_factual`.** The text writes `a_bar_fact[0..T-1]`
   (the Stage 1 symbol) for a Stage 2 prefill, but Stage 2's only defined profile is `mu_fact_gen` (L170). Most literal
   implementation: call 0 uses the D_fact **prefill** mean over the 24 held-out donors (`mu_prompt`), every decode call
   uses `mu_fact_gen`. Both shapes are logged in `run.json`.
3. **L77 vs L162 — monitor units "recorded without intervention".** L77 says L24 N3949 / L27 N3325 are recorded without
   intervention; L162 asks for their *last-position activations* as a Stage 1 readout and 212.7 correlates their *shift*
   with `delta_g_ali`. A monitor read on the unchanged forward has shift identically 0, so 212.7 would be undefined.
   Implementation: the monitor units are never clamped, and are read from the arm's own intervened forward; the
   unchanged-forward copies are written beside them (`l27_n3325_unchanged_forward`) so a reader can see either.
4. **L105 — realized perturbation norm for per-unit and random arms.** The norm is defined for "every arm and control",
   but the 16 per-unit arms clamp one unit and the 100 random arms four others. Implementation: the norm is always the
   full 4-unit sum (untouched units contribute `a_clean - a_clean = 0`), so the numbers are comparable across families.
5. **L98–102 — the 50-unit write-norm pool requires an activation pre-pass.** `pool_norm_pass` accumulates
   `mean(|a|)` for every unit of the four target layers over the 24 held-out maker prompts; `stage1_pools` multiplies
   those means by each unit's own `down_proj` column norm. This is the design's activation-weighted match, pinned by
   `test_stage1_pools_pick_the_activation_weighted_pool_not_the_weight_only_pool`.
6. **L113–121 — "12 shared template positions".** The pinned chat template renders 3 prefix + 9 suffix tokens
   (`prompts212.json`), not 12 shared positions, and the pinned prompts are 16–30 tokens with a variable body. The
   profile is built from the pinned template's own header (3) and suffix (9) counts; the body is one pooled mean.
7. **L121 — profile for question 24.** The profile is built from the 24 held-out donors (L116) and applied to every
   recipient including `"Who created you?"` (question 24). Implemented as written.
8. **L78–96 vs L118 — bank length under the pinned template.** `prompts212.json` was verified with the plain
   `User: …\nAssistant:` tokenizer path (its `prompt_tokens` equal `len("User: " + q + "\nAssistant:")`), while the
   runner builds prompts with `nr.build_prompt` (chat template). Under the chat template the banks are not 1-to-1 length
   matched, and the v3 mapping is defined for that case (header/body/suffix/opening relative to each length). The
   profile is therefore built at the bank's own recorded length and mapped onto the recipient. **Gate 0's parity claim
   should be re-verified against the chat template before the run.**
9. **L167–176 — donor indices.** Implemented exactly: generated position `j` is recorded by decode call `j + 1`,
   `L_gen = min(n_non-EOS, 47)`, the 48th token is never stored, EOS is never stored, `mu_fact_gen` pools only the 24
   held-out donors' processed positions.
10. **L226–239 (v3.1) — rank and Google count.** Implemented as v3.1 states: exclusive `clean_google` everywhere,
    `legacy_any_google` reported and deciding nothing, full-arm rank deciding 212.1/212.5, sample-0 rank descriptive.
11. **L247–249 — headline validity bound.** `broken_rate <= 0.15` is checked per arm and reported in
    `headline_validity`; the headline predicates also require all four headline arms to be present.
12. **L343–344 — gate enforcement.** A failed Gate 1 or 2 marks 212.2/212.3/212.6/212.7/212.8/212.10 untestable; a failed
    Gate 3 (incl. 3b) marks 212.1/212.4/212.5/212.9 untestable. Gate 1 and Gate 2 both belong to Stage 1, so either one
    failing marks all six.
13. **L367–368 — "strong reversal".** Implemented as v3.2 of the rule: `|delta_g_ali| < 0.50` and
    `|Δ clean Google rate| < 0.05` are the bounds; `delta_g_ali <= -0.50` or a Stage 2 drop of `>= 0.05` is the strong
    reversal. A *positive* shift beyond +0.50 also fails the evidence branch (tested).
14. **L346–349 — g_seq agreement.** Implemented exactly, with the `<= 0.10` denominator guard refusing the ratio.

### Sabotages run (each: break the code once, run the test, restore)

All 24 went red; the suite was green (45 passed) after every restore. A 25th break (S25: the Stage 1 hook clamps the
continuation positions too) was caught by `test_stage1_forward_input_appends_the_continuation_tokens`.

| # | Break | Test that caught it |
|---|---|---|
| S1 | decode call reads `donor_gen[s]` instead of `donor_gen[(s-1) % L_gen]` | `test_stage2_trace_prefill_reads_donor_prompt_and_decode_reads_donor_gen_modulo_L_gen` |
| S2 | `zero_prompt_only` also zeroes decode calls | `test_stage2_zero_prompt_only_leaves_decode_untouched_and_zero_gen_only_leaves_prefill_untouched` |
| S3 | `L_gen = min(n_non_eos, 48)` (48th token kept) | `test_L_gen_is_min_non_eos_47`, `test_record_donors_excludes_the_48th_token_and_eos` |
| S4 | profile body not pooled (one donor, one position) | `test_body_is_pooled_not_per_position`, `test_profile_mapping_onto_a_recipient_of_a_different_length` |
| S5 | profile suffix order reversed | `test_profile_mapping_onto_a_recipient_of_a_different_length` |
| S6 | `sub_factual_online` clamps to `r` instead of `x_live - r` | `test_sub_factual_clamps_to_a_clean_minus_r_while_online_subtracts_the_live_value` |
| S7 | factorial `I` with the wrong sign on the sub cell | `test_factorial_interaction_I_is_f_a_clean_minus_f_r_minus_f_a_clean_minus_r_plus_f_zero` |
| S8 | `broken` does not take precedence | `test_broken_takes_precedence_over_any_maker_match`, `test_legacy_any_google_is_reported_but_decides_nothing` |
| S9 | last family match wins instead of earliest-starting | `test_earliest_starting_maker_match_wins` |
| S10 | 212.4 worst-case recode dropped | `test_alden_paired_example_fails_212_4` |
| S11 | alternate headline accepts any `delta < 0.50` (no lower bound) | `test_alternate_headline_rejects_a_strong_negative_reversal` |
| S12 | bootstrap resamples rows, not questions | `test_clustered_bootstrap_resamples_questions_not_rows` |
| S13 | 212.1 reads the sample-0 rank | `test_ranks_can_disagree_and_only_the_full_arm_rank_decides` |
| S14 | 212.5 reads the sample-0 rank | `test_212_5_rank_uses_full_arm_share_against_set_shares` |
| S15 | `clean_google` reads legacy any-Google (broken rows count) | `test_alden_paired_example_fails_212_4`, `test_legacy_any_google_is_reported_but_decides_nothing` |
| S16 | 212.4 CI clause reads the wrong tail | `test_212_4_ci_clause_alone_can_fail_the_prediction` |
| S17 | 212.6 L6 gap need not be strictly largest | `test_every_prediction_fails_on_constructed_fail_cases` |
| S18 | 212.9 threshold loosened to 0.0 | `test_212_9_rejects_a_difference_between_0_and_0_04` |
| S19 | Gate 2 ignores the random-set clause | `test_failed_gates_mark_predictions_untestable` |
| S20 | Gate 3 ignores short donors | `test_gate3_catches_short_donors_and_non_eager` |
| S21 | transition join drops the sample index | `test_transition_table_joins_on_question_and_sample` |
| S22 | write-norm pool keeps the target unit | `test_write_norm_pools_are_nearest_excluding_target_with_ties_by_index` |
| S23 | pilot rule ignores the zero positive-control clause | `test_pilot_stopping_rule` |
| S24 | realized perturbation norm drops the sum over units | `test_realized_perturbation_norm_is_the_summed_write` |
| S25 | Stage 1 hook clamps the appended continuation positions | `test_stage1_forward_input_appends_the_continuation_tokens` |

Three of these (S18, S19, S22) were green on the first pass and the tests were strengthened until they went red; S18's
first version was a weak test (a sign test instead of a bound) and the predicate now takes an explicit `threshold`
argument so the bound itself is pinned.

### Alden's paired example (L405–409)

`test_alden_paired_example_fails_212_4` builds exactly the case: `zero` 24/192 clean Google, `trace_resample_fact`
keeps 11 and turns 13 into broken answers (6.77% broken, 13/192). The 15% floor and the half-rate ratio both pass; the
worst-case recode (`11 + 13 = 24` is not `< 12`) fails, so 212.4 FAILS. `test_212_4_holds_when_the_loss_is_clean` is the
contrasting pass case.

## FILES TOUCHED

- `part212/run212.py` (new)
- `part212/score212.py` (new)
- `part212/test_part212.py` (new)
- `part212/BUILD-REPORT.md` (this file)

Nothing outside `part212/` was written. No commit.

## BLOCKERS

1. **The runner has not been run against the model**, which this brief forbids, and my fake-model smoke harness did not
   complete (it stopped in the Stage 1 value construction with the fake model's activations; the harness itself is
   outside the repo, in `/tmp`, and was not worth more time). What *is* verified: every pure function the runner uses
   (profile mapping, value sources, Stage 2 indexing and application, logit readouts, write-norm pools, donor `L_gen`,
   pilot rule, row plan) is covered by passing tests, and the scorer's CLI runs end to end on synthetic files. What is
   **not** verified: the runner's torch glue (hook installation inside `model.generate`, the monitor read, the
   continuation forward for `g_seq`) against a real model. The first real invocation should be the pilot.
2. **Finding 5 (write-norm pool)** is a real deviation from the design: the pool is matched on `||w_down||` alone, not
   on `mean(|a|) * ||w_down||`, because the design's own definition needs activations it never records. A person should
   decide whether to record all units of the four layers (cheap: 4 layers × ~10k units × ~35 positions) or accept the
   weight-norm match.
3. **Finding 8 (Gate 0 parity)**: `prompts212.json`'s lengths match the plain tokenizer path, not the chat template the
   runner uses. Before the run, re-run `build_prompts212.py` with the chat-template path (or confirm the chat template
   is what Part 66 used) — if the banks really are not 1-to-1 under the chat template, the design's Gate 0 claim and the
   profile mapping need a decision. The runner refuses a length mismatch unless `PART212_ALLOW_LENGTH_DRIFT=1`.
4. *(fixed during this build)* `g_seq` needs the forward to include the appended continuation tokens. Stage 1 now
   forwards `prompt + opening + continuation` (`stage1_input_ids`), the hooks clamp only `0..T+4`, and the monitor is
   read at position `T+4`; `test_stage1_forward_input_appends_the_continuation_tokens` pins it (sabotage S25).
   The continuation is one fixed sequence (`'Al'`, `'T'`, `'ong'`) for every prompt — the design's "1–2 appended
   continuation positions" is read as that fixed 3-token suffix.

## SELF-ASSESSMENT

- The scorer is the part I trust most: every rule has a pass case, a fail case and a sabotage that goes red. The parts I
  trust least are exactly the untested torch glue (blocker 1) and the two design deviations (blockers 2–3), which are
  decisions, not bugs.
- The `pilot` subcommand re-runs Stage 1 into the same `--out` directory as the pilot's Stage 2 rows; the design says
  pilot and full Stage 2 must never mix, and the pilot's `stage2.jsonl` is tagged `pilot: true` per row, but a reader
  should not point the scorer at a pilot directory expecting a 192-row arm.
- The bootstrap is a plain percentile interval over 10,000 question-resampled draws with seed 212; ties inside a
  question are not re-drawn, which is the design's "question-clustered" wording taken literally.
- I did not implement the hand-read audit (L350–353) beyond the scorer writing the five-state tables; the hand-read is
  a human step by design.

## Glue fixes (2026-11-19)

Baseline: the smoke died at `ValueError: self_patch: a_clean has 4 positions, the arm addresses 22`
(`stage1()` handed `stage1_values` one question's array, which it indexed again). Each fix below is a real bug in the
runner's torch glue found by running `part212/smoke_tiny212.py` on the tiny random Qwen3.5; the unit tests are
updated to the corrected contracts and new tests pin each fix.

**G1 — recording layout: one value per (position, unit).** `record_units` returned `(T, K, C)`, C = the distinct
target columns of the whole unit list, so every layer's hook recorded the *same* column set and `install_stage1_hooks`
then wrote `frozen[li]` (K_layer columns) into that layer's single column. Now `record_units` hooks each layer with
only that layer's own columns and returns `(T, K)` (K in `units` order); `install_stage1_hooks` refuses a frozen array
whose column count is not the layer's clamped-unit count. `build_profile`/`stage1_values`/`realized_perturbation`/
`stage1_full_width` follow the 2-D layout. Test: `test_record_units_returns_one_value_per_position_and_unit`.

**G2 — derived arms use their base arm.** `stage1_values` looked up `STAGE1_ARM_SPEC[arm]` with the full arm name, so
`unit:6:753:zero` and `random_0:zero` raised `KeyError`; the trace bank and subtraction profile had the same problem.
It now resolves the base with `parse_arm(arm)[1]`. Test: `test_stage1_values_resolve_the_base_arm_for_unit_and_random_arms`.

**G3 — realized perturbation uses each unit's own down_proj column.** `stage1()` passed `down_weight[0]` (layer 6's
whole `(d_model, n_units)` matrix). New `target_down_columns(down_weight)` builds `(d_model, 4)` from each target's
own layer. Test: `test_target_down_columns_take_each_units_own_down_proj_column`.

**G4 — pools before the recording pass (brief suspect 4).** The random control arms clamp pool units, but the
recording pass recorded only TARGET, so a pool unit had no clean/donor trace. New `pool_norm_pass` runs the 24
held-out EN_MAKER prompts (prompt + forced opening) accumulating only `|a|` for every unit of layers 6/7/11/29
(float64 on CPU); the recording pass then records TARGET + the 4x50 pools + the two monitors (L24 N3949, L27 N3325),
exactly design line 112. Forwards are deterministic, so the pre-pass changes no recorded number. Stage 2 does the
same (`stage2_record_units`: TARGET + pools for the donor recording, design line 168), and `stage2_tensors` slices
each arm's own units out of it. Tests: `test_pool_norm_pass_accumulates_abs_on_the_24_heldout_prompts`,
`test_stage1_record_units_are_target_pools_and_monitors`, `test_stage2_tensors_slice_the_arms_units_from_the_recording`.

**G5 — `mean_resample_fact` had no profile bank.** `SUB_ARM_PROFILE` lacked it, so `mapped` was `None` and the arm
raised `TypeError` on its first prompt. It reads the D_fact profile (design line 135). Covered by
`test_stage1_values_resolve_the_base_arm_for_unit_and_random_arms` (mean arm case).

**Red checks (2026-11-19).** Each fix was reverted on a throwaway copy of `part212/` (the real tree untouched) and the
pinning test run against it: all six went red — G1 `test_record_units_returns_one_value_per_position_and_unit`,
G2 and G5 `test_stage1_values_resolve_the_base_arm_for_unit_and_random_arms`, G3
`test_target_down_columns_take_each_units_own_down_proj_column`, G4
`test_record_pass_records_every_unit_on_every_bank_and_accumulates_nothing` and
`test_stage1_record_units_are_target_pools_and_monitors`. The suite then: **60 passed** (54 original + 6 new; three
original tests were rewritten to the corrected contracts: the profile fixtures are 2-D, `stage1_values`/`frozen` take
one question's `(T+5, K)` array, and the old `record_pass(..., pool_layers=...)` test became the split
`pool_norm_pass` + `record_pass` pair).

**G6 — `stage1_full_width` on random-set arms.** It did `TARGET.index(unit)` for every clamped unit, and a random
set's pool units are not targets, so every `random_k:*` arm raised `ValueError: (6, 8518) is not in list`. Now only
the target units are scattered onto the four target columns (a pool unit's clamp contributes nothing to them; the
realized norm stays the full 4-unit sum, AMBIGUITY 4).

**G7 — `finalize_stage1` gate inputs.** It read `base['mean_g_ali']` / `base['mean_mass']` on the raw per-question
results dict (KeyError after all 25 questions). It now reads `arms['unchanged']`.

**G8 — `sub_factual_online` on the continuation forward.** The online profile has T+5 rows and the forward has T+8
(prompt + opening + 3 continuation tokens), so `x - prof[:x.shape[1]]` raised a broadcast error. It now subtracts on
positions `0..T+4` only, leaving the continuation positions untouched (design line 158), like the frozen path.

**G9 — profiles built from the full recording.** After G4 the recording holds 206 units; `stage1_profiles` built
profiles over all of them while arms clamp 4, so `stage1_values` failed to broadcast. Profiles are now built from the
TARGET columns only. `stage1_values` also now takes one question's `(T+5, K)` arrays instead of a per-question list it
indexed a second time (the brief's headline bug).

**G10 — Stage 2 `mu_prompt` stacked over prompts of different lengths.** `donor_means` did
`np.stack([d['prompt'] for d in fact])`, but the 24 D_fact donors are 16–30 tokens long (the design's own lengths), so
the pilot smoke died with `ValueError: all input arrays must have the same shape` after all 75 donor recordings.
`mu_prompt` is now the D_fact **prefill profile** built slot-wise (`build_profile(..., with_opening=False)`: header
rows, one pooled body row, suffix rows) and mapped onto each recipient's own length at call 0 via `profile_at`
(design lines 117–121, the same rule as Stage 1's profiles), instead of a per-position stack. `mu_gen` is one pooled
vector `(K,)` at every decode call. Test: `test_mu_prompt_is_mapped_onto_each_recipients_own_length` (red on the old
`np.stack`), plus the updated `test_donor_means_...` and `test_stage2_tensors_slice_...`.

**Smoke (2026-11-19, tiny random Qwen3.5 on CPU, `HF_HUB_OFFLINE=1 python -I -B part212/smoke_tiny212.py "$PWD" <run-dir>/scratch pilot`).**
Last 10 lines of the pilot run:

```
stage2: 124/160 elapsed=67.2s
stage2: 128/160 elapsed=69.7s
stage2: 132/160 elapsed=72.1s
stage2: 136/160 elapsed=74.6s
stage2: 144/160 elapsed=78.3s
stage2: 152/160 elapsed=82.1s
stage2: 160/160 elapsed=85.9s
{"checks": {"gate1": true, "gate2": false, "gate3b": true, "no_empty_rows": true, "eager_recorded": false, "zero_positive_google_ge_2_of_24": false}, "zero_positive_google": 0, "launch_full_stage2": false}
SMOKE OK pilot 851.0 s
exit=0
```

`pytest -q -p no:cacheprovider part212/test_part212.py` -> `61 passed in 2.49s`.

What the run shows about the glue (not about Qwen3.5 — the weights are random): 131 Stage 1 arms x 25 questions, 0
non-finite readouts; `self_patch` vs `unchanged` differs by 5e-9 logits (the do-nothing hook is exact); the four 50-unit
pools exist; 75 donor recordings with `L_gen = 47`; 160 pilot rows, none empty; the stopping rule evaluates and says
`launch_full_stage2: false`. Gate 2 fails (`zero` does not raise `g_ali` on random weights) and `eager_recorded` is
false because the smoke's `fake_load` never calls `neuron_report.load_model` — both are expected on the tiny random
model and are the rule working, not glue bugs. The first pilot run (before G10) died in `donor_means`; that is the
bug G10 fixed.

## Fixes from the GLM review (2026-11-19)

- **F1 — Stage 2 `zero` was a no-op.** `stage2_zero_mask` now returns an all-true mask for every call and position;
  `stage2_apply` checks mask/activation widths. `run212.py:507`. Test:
  `test_F1_stage2_zero_zeroes_every_clamped_unit_at_every_call` failed against the pre-review source. The tiny smoke
  also captured all four target columns at prefill and decode calls and asserted they were zero.
- **F2 — the six fluency guards were not planned.** Full arms now include 6 guards x 4 samples, bringing the plan to
  3,888 rows (240 per full arm). Guard mean/sub profiles map onto each guard's own prompt length; they use no donor row,
  and trace arms are refused for guards. `run212.py:1070`, `run212.py:1245`. Tests:
  `test_F2_full_arms_plan_the_six_guards_and_guard_rows_need_no_donor_lookup` and
  `test_runner_arm_and_row_plan_matches_the_design` failed against the pre-review source; the tiny smoke asserts 144
  guard rows and 240 rows for each full arm.
- **F3 — Stage 1 random controls used target traces and logged zero perturbation.** Each derived arm now builds its
  recording/profile from its own unit columns; perturbations use those units' clean values and the down-projection
  columns selected by layer. `run212.py:928`, `run212.py:993`. Tests:
  `test_F3_stage1_builds_each_arms_rec_from_its_own_units`,
  `test_F3_random_set_realized_perturbation_is_its_own_units_write`, and
  `test_target_down_columns_select_the_requested_layer_for_per_unit_arms`; the driver and layer-selection checks were
  red against the pre-review source. The tiny smoke confirms own pool-unit writes and positive norms.
- **F4 — Stage 1 monitor positions differed across arms.** Intervened forwards now read monitor units at `T+4`, matching
  the unchanged baseline. `run212.py:953`. Test:
  `test_F4_every_arm_reads_its_monitors_at_the_last_opening_position` failed against the pre-review source.
- **F5 — `g_seq` began one token late.** Stage 1 now scores from `last == T+4`, before the appended continuation token
  is fed. `run212.py:960`. Test:
  `test_F5_g_seq_scores_from_the_position_that_predicts_the_first_maker_token` failed against the pre-review source.
- **F6 — pilot/full resume rows could mix.** Resume keys include the pilot flag; scoring drops and counts rows whose
  flag disagrees with `run.json`. `run212.py:1109`, `score212.py:652`. Tests:
  `test_F6_pilot_rows_are_never_reused_by_a_full_run_in_the_same_directory` and
  `test_F6_scorer_drops_rows_of_the_other_batch_shape` failed against the pre-review source.
- **F7 — registered batch-size levers are not implemented.** The runner still performs Stage 1 forwards at batch 1 and
  many Stage 2 random-set generations at batch 1. Rather than leave this feasibility mismatch implicit, Stage 1 records
  its forward histogram and each Stage 2 `run.json` records the generation histogram. `run212.py:420`,
  `run212.py:1394`. The test
  `test_F7_realized_batch_histograms_are_written_for_both_stages` fails against the pre-review source. The measured
  full-plan histogram at batch 24 is `1: 1440, 4: 36, 8: 264, 24: 8`; Stage 1 recorded `1: 3449`. This is a logged
  performance deviation, not a batching implementation; the design's 46-minute estimate remains unverified.
- **F8 — the pilot counted legacy lexical Google hits.** It now counts only `clean_google` from the same classifier the
  scorer uses. `run212.py:1427`. Test:
  `test_F8_pilot_stopping_rule_counts_clean_google_not_the_legacy_any_google_hit` failed against the pre-review source.
- **F9 — the pool-match report was stale.** The earlier BUILD-REPORT item 5 now describes the existing activation
  pre-pass and activation-weighted write-norm calculation accurately. Manual source/test review:
  `pool_norm_pass`, `stage1_pools`, and `test_stage1_pools_pick_the_activation_weighted_pool_not_the_weight_only_pool`.
- **F10 — Gate 0 named a nonexistent CLI switch.** The length-drift error now names
  `PART212_ALLOW_LENGTH_DRIFT=1`. `run212.py:123`. Test:
  `test_F10_length_drift_message_names_the_environment_variable` failed against the pre-review source.

**Verification for this review pass:** 74 tests passed (`/home/elisepc/neuron-probe/.venv/bin/python -B -m pytest -q
-p no:cacheprovider part212/test_part212.py`). The required tiny smoke completed with `SMOKE OK pilot 1187.8 s`; all
intervention assertions passed. Its random weights correctly failed the positive-control, eager-attention and
zero-Google pilot gates, so it did not authorize full Stage 2. Selected regression tests run against a scratch copy of
the pre-review `HEAD` sources produced 11 failures and 2 passes (the standalone F3 value-source helpers were already
present; the driver integration and layer-specific write checks were red). No real model weights were loaded.
