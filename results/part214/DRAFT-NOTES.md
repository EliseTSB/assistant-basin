# Part 214 draft notes: where every number on the page comes from

For checking [README.md](README.md). Each row gives the printed number and the file and key it was copied from. Raw
values are given where the README rounds. Prediction identifiers (214.1-214.8), unit and layer names, the model name
and revision, and dates are identifiers rather than measurements; their sources are listed at the end.

## Stage 2 counts (README table and every section that quotes a count)

`stage2/score.json`, `stage2.main_arms.<arm>`: `.clean_google`, `.broken`, `.clean_google_rate`, `.denominator`.
The rates are printed to 4 decimal places (0.125 is exact).

| Arm | clean_google | broken | rate |
| --- | ---: | ---: | ---: |
| unchanged | 4 | 0 | 0.0208 (0.0208333...) |
| zero | 40 | 0 | 0.2083 (0.2083333...) |
| trace_resample_fact | 28 | 0 | 0.1458 (0.1458333...) |
| scale_dial_fact | 26 | 1 | 0.1354 (0.1354166...) |
| trace_resample_you | 16 | 0 | 0.0833 (0.0833333...) |
| scale_dial_you | 10 | 0 | 0.0521 (0.0520833...) |
| prompt_fact_gen_you | 24 | 0 | 0.125 |
| mean_resample_you | 12 | 0 | 0.0625 |

`192` per arm is `.denominator`; `24` questions is `stage2.bootstrap.<pair>.n_questions`. The `28 of 192` broken
ceiling and the `0.05`, `0.04`, `0.03`, `0.10` bars are the rules in `predictions.<n>.rule`.

## Bootstrap intervals and differences

`stage2/score.json`, `stage2.bootstrap.<pair>`: `.estimate`, `.lo`, `.hi`, `.draws`, `.seed`.

| README value | Raw | Key |
| --- | --- | --- |
| zero - unchanged 0.1875, 0.098958 to 0.291667 | 0.1875, 0.098958333..., 0.2916666... | `zero-unchanged` |
| trace_fact - trace_you 0.0625 | 0.06250000000000001 | `trace_resample_fact-trace_resample_you` |
| dial_fact - trace_fact -0.0104, -0.0625 to 0.0417 | -0.0104166..., -0.0625, 0.0416666... | `scale_dial_fact-trace_resample_fact` |
| dial_you - trace_you -0.0313, -0.0938 to 0.0208 | -0.03125, -0.09375, 0.0208333... | `scale_dial_you-trace_resample_you` |
| prompt_fact_gen_you - trace_fact -0.0208, -0.078125 to 0.026042 | -0.0208333..., -0.078125, 0.0260416... | `prompt_fact_gen_you-trace_resample_fact` |
| mean_resample_you - trace_you -0.0208, -0.078125 to 0.026042 | -0.0208333..., -0.078125, 0.0260416... | `mean_resample_you-trace_resample_you` |
| zero - mean_resample_you 0.1458, 0.057292 to 0.244792 | 0.1458333..., 0.0572916..., 0.2447916... | `zero-mean_resample_you` |

`10,000` draws and `seed 214` are `.draws` and `.seed` on every bootstrap entry.

## Prediction 214.1 (held)

`predictions.214.1.observed`: `.rates.zero` (0.2083), `.rates.unchanged` (0.0208), `.zero_minus_unchanged`
(0.1875), `.ci_zero_unchanged.lo/.hi` (0.098958..., 0.291666...), `.rates.trace_resample_fact` (0.1458),
`.trace_fact_minus_unchanged` (0.125), `.rates.trace_resample_you` (0.0833), `.trace_fact_minus_trace_you`
(0.0625), `.broken` (0 for all three), and `.clauses` (all true). Counts 40, 4, 28, 16 from `stage2.main_arms`.
Verdict `predictions.214.1.verdict` = HOLDS.

## Prediction 214.2 (failed)

`predictions.214.2.observed`: `.delta_gen` = 0.07524356278887281, printed 0.0752; `.alpha_gen_fact` =
-0.0013541574369483169, printed -0.0014; `.clauses` `delta_gen_ge_+0.08` false, `alpha_gen_fact_le_0.35` true.
The `0.08` and `0.35` bars are in `.rule`. The pairs (0.2474, -0.0014) and (0.4120, 0.0739) are
`pinning.facts.alpha_prompt/.alpha_gen` and `pinning.you.alpha_prompt/.alpha_gen`, and are repeated rounded to 4
decimals in `paste_block`. Verdict `predictions.214.2.verdict` = FAILS.

## Prediction 214.3 (held)

- The ten curve values 5.331, 3.093, 2.471, 1.989, 1.604, 1.239, 0.927, 0.548, 0.276, 0.000:
  `stage1.dial_curve.points[].mean_delta_g_ali` in order (raw 5.331156301929128, 3.0925492119059657,
  2.4707676864555124, 1.989209095957716, 1.6039974446052574, 1.2387697762395973, 0.9274695408904712,
  0.5479523251682485, 0.27625539445928254, 0.0).
- The ten alphas -0.25, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0: `stage1.dial_curve.points[].alpha`.
- Spearman -1.000: `stage1.dial_curve.spearman_alpha_vs_delta` (raw -1.0); the -0.95 bar and `10` rungs:
  `predictions.214.3.observed.clauses` / `.rule`.
- 2.796: `stage1.dial_arms.scale_dial_fact` (raw 2.79588254787383).
- -0.049 to +0.067: minimum and maximum of `stage1.random_scale_dial` values (raw -0.04943157724924021,
  0.0668643505514747); `10` sets is its length / `observed.n_random_dial`.
- -1.824 and -0.404: `predictions.214.3.observed.bridges.nat_trace_fact_minus_scale_dial_fact` (raw
  -1.8236781370476716) and `.nat_trace_you_minus_dial_you_ref` (raw -0.4035989524117336); the 0.50 tolerance is
  `.bridges.tolerance`. Verdict `predictions.214.3.verdict` = HOLDS.

## Prediction 214.4 (held)

`predictions.214.4.observed`: `.a_endpoint` = -0.010416666666666685 (printed 0.0104 as an absolute difference);
`.a_vs_unchanged` = 0.11458333333333333 (0.1146); `.dial_gap` = 0.08333333333333331 (0.0833); `.donor_gap` =
0.06250000000000001 (0.0625); `.b_capture` = 1.3333333333333328 (1.3333), against the rule's 60% (0.6);
`.b_you_trace` = -0.031249999999999993 (magnitude 0.0313); `.recode_lhs` = 10 and `.recode_rhs` = 26 (the recode
10 below 26); `.c_two_sided` = 0.0208333333333333 (0.0208); `.clauses` all true. Counts 26, 28, 10, 16 and 4 and
broken 1, 0: `stage2.main_arms`. The 0.05 and +0.04 bars and the 28-of-192 ceiling: `.rule`. Verdict
`predictions.214.4.verdict` = HOLDS.

## Prediction 214.5 (failed)

`predictions.214.5.observed`: `.donor_gap` = 0.06250000000000001 (0.0625); `.drop` = 0.020833333333333343
(0.0208); `.share` = 0.3333333333333334 (0.3333); `.half_donor_gap` = 0.03125000000000001 (0.03125); `.broken` =
0 and `.recode_ok` = true (24 below 28); `.clauses.captures_gt_50pct` and `.clauses.drop_ge_0.03` false. Counts
24 and 28: `stage2.main_arms`. The H2 observation 0.0417 and 0.03125: `headlines.rivals.h2.fy_minus_you`
(0.04166666666666667) and `.half_donor_gap`, `.recode` true. Verdict `predictions.214.5.verdict` = FAILS.

## Prediction 214.6 (held)

`predictions.214.6.observed`: `.jump` = -0.02083333333333333 (-0.0208); `.zero_minus_mean` =
0.14583333333333334 (0.1458); `.broken` = 0; `.clauses.jump_le_+0.05` and `.clauses.zero_minus_mean_ge_0.04`
true. Counts 12 and 16: `stage2.main_arms`. Verdict `predictions.214.6.verdict` = HOLDS.

## Prediction 214.7 (failed)

`predictions.214.7.observed`: `.nat_trace_fact` = 0.9722044108261584 (0.972); `.nat_trace_imp` =
2.1725624409107964 (2.173); `.nat_trace_you` = 1.8420724943559281 (1.842); `.shift` = -1.200358030084638
(-1.200); `.imp_vs_you` = 0.3304899465548683 (0.330); `.clauses` both false. The +0.20 and 0.30 bars are in
`.rule`. The `0.031` miss is from [COLD-CHECK.md](cold-check/COLD-CHECK.md) finding 1 and its 214.7 verdict line
("misses by 0.031"). The `g_seq` values 5.897, 4.937, 4.971 and 0.034 are COLD-CHECK finding 1 ("fact 5.897 > imp
4.937 ≈ you 4.971, |imp − you| 0.034"). Verdict `predictions.214.7.verdict` = FAILS.

## Prediction 214.8 (failed)

`predictions.214.8.observed`: `.fact_delta` = 0.00799709560170947 (0.0080); `.you_delta` =
0.012999296443685582 (0.0130); `.spearman_l27` = -0.27058823529411763 (-0.271); `.n_paired` = 16;
`.clauses.fact_no_L29_within_0.25` and `.you_no_L29_within_0.25` true,
`.clauses.l27_tracks_negatively_le_-0.80` false; the 0.25 and -0.80 bars are in `.rule`. The dial-only -1.000,
the natural arms' L27 range 3.449 to 3.512, their delta g_ali range 0.958 to 2.173, and the within-block +0.886
are COLD-CHECK finding 2. Verdict `predictions.214.8.verdict` = FAILS.

## Opening paragraph and title

- 26, 28, 192: `stage2.main_arms.scale_dial_fact.clean_google`, `.trace_resample_fact.clean_google`, denominators.
- 0.0104: `predictions.214.4.observed.a_endpoint` (absolute value).
- 16/192 and 12/192: the integer forms of `predictions.214.4.observed.dial_gap` = 0.0833 and `.donor_gap` =
  0.0625 (16/192 = 0.0833, 12/192 = 0.0625); the brief's headline wording uses these counts.
- 0.0208: `predictions.214.4.observed.c_two_sided`.
- 0.3333: `predictions.214.5.observed.share`.
- 12 and 16: `stage2.main_arms.mean_resample_you.clean_google` and `.trace_resample_you.clean_google`.
- 0.0752 and 0.08: `predictions.214.2.observed.delta_gen` and `.rule`.
- 0.7 and 48: [PART214-DESIGN.md](../../PART214-DESIGN.md) Stage 2 (`temperature=0.7`, `max_new_tokens=48`);
  these are method parameters, not result numbers.
- "four held, four failed": `predictions.<n>.verdict`.

## Method, gates and run facts

- 60 arms and 25 prompts: `stage1.completeness.complete_arms` (60 entries) and `.planned_questions_per_arm` (25).
- 4,624 batch-1 forwards: `stage1.forward_batch_histogram["1"]` = 4624.
- 1,584 batch-1 answers: `stage2.n_rows` = 1584 and `stage2.batch_size_histogram["1"]` = 1584.
- 192 per arm and 24 questions: `stage2.main_arms.*.denominator` and `stage2.bootstrap.*.n_questions`.
- 10,000 draws, seed 214: `stage2.bootstrap.*.draws` and `.seed`.
- 0.0007: `gates.gate1.self_patch_minus_unchanged` (raw 0.0007019637132992251).
- 4 of 24: `stage2.positive_control.zero.clean_google` = 4 and `.n` = 24.
- 24 of 24 guards: `stage2.guard_accuracy.unchanged.n` = 24 and `.correct` = 24.
- 100 greedy donor and maker recordings: `gates.gate3b.donors` = 100 (four banks x 25 prompts, PART214-DESIGN.md
  Stage 0). The `four` banks / `four` units: `stage2/run.json` `target` has 4 entries and `gate3b_prompts_per_bank`
  has 4 banks.
- "All gates passed": `gates.gate0.pass`, `gates.gate3b.pass`, `gates.gate1.pass`, `gates.gate2.pass`,
  `gates.gate3.pass`, all true in `stage2/score.json`.
- Revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, `torch.bfloat16`, `eager`, `xpu`, Arc A770:
  `stage2/run.json` `revision` and `load_this_invocation` / `runtime.load`.
- 0.7, 0.95 and 48 tokens: PART214-DESIGN.md Stage 2 sampling parameters, as above (method, not results).
- The four units `(6, 753)`, `(7, 1169)`, `(11, 8868)`, `(29, 776)` and monitors `L24 N3949`, `L27 N3325`:
  `stage2/run.json` `target` and `PART214-DESIGN.md`.

## Limits (COLD-CHECK findings 3-7)

- Finding 3: 4 clean Google, 2 of 24 questions, q16 3/8, q20 1/8, 12 questions won, 0 lost.
- Finding 4: six answers also name Qwen or Alibaba, two contradict themselves, any-Google count 11 against 10.
- Finding 5: positive control q24, and the re-run values -1.245 / 0.349.
- Finding 6: 1,426 of 1,584 at `max_new_tokens`, median 34 words.
- Finding 7: the one broken row is `scale_dial_fact` q17 s0.
- The 214.2 premise statement: COLD-CHECK line "214.2 was not in scope (a Stage 0 calibration quantity, 0.0752
  against 0.08)" and the `## Part 214` prediction text, which calls it calibration observed at registration.

## Predictions table

- The Prediction column is verbatim from [PREDICTIONS.md](../../PREDICTIONS.md), `## Part 214`, rows 214.1-214.8
  (the design's table carries the same text with the confidence in a separate column). It was checked by script
  against the source cells after escaping `|` as `\|` so the markdown table renders; no other character differs.
  The numbers inside that column are the registered thresholds, not results.
- The Result column numbers are itemised under each prediction above; the Verdict column maps
  `predictions.<n>.verdict` HOLDS/FAILS to held/failed.

## Identifiers, dates and spelled-out counts (not measurements)

- Prediction IDs 214.1-214.8; unit names `L6 N753`, `L7 N1169`, `L11 N8868`, `L29 N776`; monitors `L24 N3949`,
  `L27 N3325`; bank names `D_fact`, `D_you`, `D_imp`; model Qwen3.5-4B at revision `851bf6e...`; Parts 212, 66
  and 214; stage labels Stage 0/1/2; gate labels Gate 0/1/2/3/3b. Sources: PREDICTIONS.md, PART214-DESIGN.md and
  `stage2/run.json`.
- `2026-11-20 14:11 CT`: the `## Part 214` header in PREDICTIONS.md and the design's registration section.
  `2026-11-20`: the COLD-CHECK header.
- Spelled-out counts: `ten` dial rungs and `ten` random dial sets (`stage1.dial_curve.points` length 10 and
  `stage1.random_scale_dial` length 10); `six` natural-decode arms (COLD-CHECK finding 2); `three-unit` engine
  (prediction 214.8, "3-unit"); `two` blocks (COLD-CHECK finding 2); `two` cross-phase arms and the first `five`
  recorded donor token ids (PART214-DESIGN.md Amendment 2); `four held, four failed` (the verdicts); `two
  contradict themselves` and `six` answers (COLD-CHECK finding 4); `seven findings` and `eight verdicts`
  (COLD-CHECK, which numbers seven findings and agrees on all eight); `three Stage 1 natural-gen means`
  (`nat_trace_fact`, `nat_trace_imp`, `nat_trace_you`); `two banks` (`D_fact`, `D_you`); `one broken answer` /
  `one broken row` (`stage2.main_arms.scale_dial_fact.broken` = 1; COLD-CHECK finding 7); `one model, one snapshot,
  one quantisation, one four-unit set` (PART214-DESIGN.md Limits); `first`/`second` clause are clause labels from
  the prediction text.

## Rounding convention

Rates are printed to 4 decimal places; logit-scale values and correlations to 3. Interval endpoints are printed
either at 6 decimal places where the raw value is not a short decimal (0.098958, 0.291667, -0.078125, 0.026042,
0.057292, 0.244792) or at 4 where it is short or the brief's wording uses that form (-0.0625 is exact, 0.0417,
-0.0938, 0.0208). Raw values are given above wherever the printed form is rounded. The same quantity is not
rounded differently in two places: 0.0313 is always the `scale_dial_you` - `trace_resample_you` magnitude, 0.0208
(and -0.0208) is always a 4/192 quantity (gap error, 214.5 drop, 214.6 jump), 0.0752 is always the Stage 0 decode
difference, and 2.796 is always `scale_dial_fact`'s Stage 1 mean.
