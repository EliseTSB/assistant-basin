# Cold read of Part 206's README, figure and front-page entry

Read (all under `/home/elisepc/green/worktrees/ab-part206/`): `results/part206/README.md`, `results/part206/fig206.png`,
`part206/fig206.py`, `results/part206/FRONT-PAGE-ENTRY.md`, `PART206-DESIGN.md` (Predictions L338-358, Reading rules
L377-411, Limits L412+, Gates L279-294), `part206/AMENDMENT-1..5.md`, `results/part206/score206-output.txt`,
`results/part206/score.json`, the eight `results/part206/<stage>/summary.json`, `results/part206/qwen-chat/causal.json`,
and (for the cited earlier parts only) `results/part178/README.md`, `results/part201/README.md`, `README.md` (front page),
`results/part153/README.md`. No model was run. Numbers were recomputed with `python3 -I` from the JSON.

## SUMMARY

**Five findings; two would change a sentence.** F1 (Panel A's title "No stage reaches the 0.2 bar" is a verdict
the 0.2 bar does not support — 206.6's bar is registered for Qwen chat only, and the OLMo stages it labels are never
scored against it; the title also states "no constant shift", which the README's own Limits say `rho` alone cannot
show) and F2 (`kappa` is never defined and is called "the share ... taken position by position" on the front
page and at README L15; the design defines it as a *different* share, across passages, and says a shift varying along
the passage shows as `kappa` minus `rho`) would change a sentence. F3 (the "in all eight models" static-test sentence
presents seven models' values as read when only the scorer's verdict key was cited for them), F4 (an unregistered
descriptive reading of the arming table, asserted before it is labelled) and F5 (Part 92 cited without saying what it
measured; the Part 92b second-reader attribution) are smaller. No number in the README, figure or front page failed
to match its source key.

## FINDINGS

**F1. Panel A's title states a verdict the 0.2 bar does not support.** `part206/fig206.py:93`
(`a.set_title("A. Is the assistant frame a constant shift?\nNo stage reaches the 0.2 bar")`).
- What is wrong: (i) "No stage" is a verdict about the six OLMo stages, and the 0.2 bar is 206.6's bar, registered
  **for Qwen3.5-4B chat only** (Design L347). No OLMo stage is scored against 0.2 at all (206.7 is a rise, not a
  level), so the title says the OLMo stages failed a test they were not given. The two Qwen markers in the panel are
  under the bar (0.106 and 0.076), so for them the title is true but trivial — the bar is a 206.6 clause, not a
  threshold the registered design sets for "constant or not". (ii) The title states a *negative* ("no constant
  shift") as if the bar were the test; the README's own Limits (L114-121, quoting the design) say the statistic
  cannot distinguish a weak shift from a strong one that varies with position, so "no constant shift" is the
  registered headline sentence, not a conclusion the bar supports. (iii) The panel draws the 0.2 line across the OLMo
  stages too, where it has no registered meaning.
- Evidence: `qwen-chat/summary.json` `common_mode.A-tmpl|M.rho_peak` = 0.1060 (0.1026-0.1098); `qwen-base` 0.0757;
  OLMo peaks 0.085 / 0.067 / 0.075 / 0.151 / 0.153 / 0.164, all in `common_mode.A-tmpl|M.rho_peak`; Design L347
  (206.6 is "In Qwen3.5-4B (chat)") and L413-441 (Limits, "`rho` alone does not distinguish a weak shift from a strong
  one that varies with position").
- Proposed rewording: `"A. How much of the assistant-frame difference is one fixed vector?\nLargest is 0.164 (OLMo Instruct); Qwen chat 0.106, under 206.6's 0.2"`, and label the 0.2 line "206.6 bar, Qwen chat" (or keep it across the panel but say in the caption that only Qwen chat is scored against it).

**F2. `kappa` is presented as "the same ratio taken position by position", and the front page calls it "the share".**
`results/part206/README.md:96-97`, `:117-121`, `:15`; `FRONT-PAGE-ENTRY.md:1`.
- What is wrong: the README never says what `kappa` is. The design (L222-227) defines it: "the same ratio taken
  position by position (the mean over fit passages at a position times the mean over test passages at that position)
  and averaged over positions ... `kappa` is the share that is the same across passages; `rho` is the share that is
  also the same across positions". The README's table header says only "`kappa` at that layer", and its gloss
  "`kappa` is the same ratio taken position by position" (L117) drops the design's next clause, which is the one that
  makes the statistic mean anything. Worse, `README.md:15` and `FRONT-PAGE-ENTRY.md:1` say "taken position by
  position the share is only a little higher (0.117 ... 0.182)" — that calls `kappa` "the share", i.e. the same
  quantity as `rho`, when the design's point is that it is a *different* share (across passages, not across
  positions). A reader who takes the front page at its word has been told `kappa` is `rho` recomputed per position,
  which is false. (Check 7: term used but not defined for a reader.)
- Evidence: Design L222-227; `qwen-chat/summary.json` `common_mode.A-tmpl|M.layers[18].kappa` = 0.1172, `.rho` =
  0.1060; `instruct` `.layers[6].kappa` = 0.1818, `.rho` = 0.1635. The numbers are right; the description is not.
- Proposed rewording, table caption: "`kappa` is the share of the frame difference that is the same across passages
  (the design's L222-227); `rho` is the share that is also the same across positions, so `kappa` minus `rho` is a
  shift every passage gets that changes along the passage." Front page and L15: "the same-across-passages share
  `kappa` is only a little higher (0.117 and 0.182)".

**F3. "Tuomi's static test replicates in all eight models": the README's evidence covers one model's layers; the
registered static clause is three of four layers, not four.**
`README.md:16-17`, `:69`, `:80-84`, `:268`; `FRONT-PAGE-ENTRY.md:1`.
- What is wrong: the README's own evidence for the static test is Qwen chat's four layers (`rho_resting_duty` 0.960 /
  0.974 / 0.970 / 0.949 against largest random 0.098 / 0.130 / 0.118 / 0.072). For the other seven models it cites only
  `gt.verdict.static_layers_beating_random` = 4 of 4 and `gt.static[*].median_resting` negative. I traced those keys
  for all eight (they hold), but the README's phrasing "the median resting term is negative at all four layers in all
  eight models ... and its rank correlation with duty exceeds all 20 random directions at all four layers" presents as
  read values what for seven models is only a scorer verdict. Separately, the design's GT static clause requires
  beating random at **three or more** of four layers (Design L282-283); the README says "all four" without saying the
  registered bar is three of four, so a reader cannot tell whether the margin is one layer or none.
- Evidence: `qwen-chat/summary.json` `gt.static[*].rho_resting_duty`, `rho_random`, `median_resting`;
  `gt.verdict.static_layers_beating_random` = 4 in all eight summaries; Design L282-284.
- Proposed rewording: "the scorer's static clause holds in all eight models (`gt.verdict.static_layers_beating_random`
  = 4 of 4, against the registered three of four, and `gt.static[*].median_resting` negative at all four layers); on
  Qwen chat the correlations are 0.960, 0.974, 0.970 and 0.949 against largest random 0.098, 0.130, 0.118 and 0.072."
  (Same numbers, honest about which were read.)

**F4. An unregistered descriptive reading of the arming table, stated as a fact before it is labelled.**
`README.md:166-169`.
- What is wrong: "The largest gap of the six stages is at mid-training end, 25.69, not at Instruct" is a descriptive
  observation not in any registered row, asserted first and labelled "not a claim" only at the end of the paragraph.
  It singles out the mid-training-end row, the one built on the 39-neuron set whose concept half is weak by the
  README's own account (8 of 18 keep their kind, L273-275). (The null-pairs sentence at L123-126 is fine: Design
  L227-228 says exactly that (M2, M) and (N2, N) "are not bars".)
- Evidence: `midtrain-end/summary.json` `arming.slot.median_percentile` 63.32, `arming.concept.median_percentile`
  37.63, gap 25.69 (`score206-output.txt` `arming.slot-concept gap` = 25.6897); Design L227-228.
- Proposed rewording: "The largest registered rise is mid-training 26B to mid-training end, +32.04; the
  mid-training-end gap itself (25.69) is the largest of the six stages, a descriptive note on a 39-neuron set and
  not a registered row." Or drop the sentence.

**F5. Part 92 is cited without saying what it measured, and the Part 92b reader is not named in the Limits.**
`README.md:148` ("The catalogued sets are Part 92's classification of OLMo 2 1B Instruct neurons, 21 slot and 18
concept"), `README.md:273-275` ("Only 8 of the 18 concept neurons kept their kind under a second reader (Part 92b)").
- What is wrong: Part 92 is cited with no statement of what it measured (check 8) — Amendment 4 says it is
  `results/part92/step2/kinds.json`, one Haiku reader's kind classification; the reader is the part that matters for
  how weak the concept median is, and the README never says it is one reader. Second, the Limits sentence cites
  **Part 92b** for the second reader; Amendment 4 (which the README cites at L148 and L222-224) says the 21 slot and
  18 concept sets come from **Part 92**'s one Haiku reader, and that Part **92b** is the *DeepSeek* reader on which
  only 8 of the 18 keep their kind and all 21 slot keep theirs. The README's "kept their kind under a second reader
  (Part 92b)" is consistent with that, but it is placed as a property of the catalogued sets without saying the sets
  are Part 92's Haiku reader — a reader wanting to check the claim is sent to the wrong part for the sets. This is a
  citation-precision finding, not a number error.
- Evidence: `part206/AMENDMENT-4.md` "Change" bullet 1 and the "The 21 slot neurons are exactly Part 92b's 21
  reader-stable slot units, and 8 of the 18 concept neurons are Part 92b's reader-stable concept units" line; Design
  L390-392 ("on Part 92's units that kept their kind under the DeepSeek reader (21 slot, 8 concept)").
- Proposed rewording: "The catalogued sets are Part 92's classification of OLMo 2 1B Instruct neurons by one Haiku
  reader (`results/part92/step2/kinds.json`, via Amendment 4): 21 slot and 18 concept." and at L273-275: "Only 8 of
  the 18 concept neurons kept their kind under Part 92b's second (DeepSeek) reader".

## CHECKED

Every number below was read from the named key with `python3 -I` and compared with the README/entry/figure text.
**No mismatches.** The keys are relative to `results/part206/`.

- Title/abstract: 0.106 (qwen-chat `common_mode.A-tmpl|M.rho_peak` 0.10603), 0.103-0.110 (`rho_peak_ci`
  0.10261-0.10980), 0.164 (instruct `rho_peak` 0.16354), 0.160-0.168 (0.15960-0.16770), 0.117 (qwen-chat
  `layers[18].kappa` 0.1172), 0.182 (instruct `layers[6].kappa` 0.1818), 3.3-10.0 (`gt.verdict.ratio_to_largest_random`
  min 3.256 midtrain-26B, max 9.984 qwen-base), 5 of 17 / 12 failed (score.json verdicts: held 206.4, 206.5, 206.7,
  206.9, 206.M3 = 5; the other 12 failed).
- Prediction table, every cited value: 206.1 9.98 / 9.84 (qwen-base 9.983, qwen-chat 9.839); 206.2 3.26-3.56 (3.256,
  3.557); 206.3 -0.3419 (`reference[16].write_share.mlp` -0.34186); 206.4 0.5564 / 0.1182 (`split.m[0]` 0.55638,
  `split.m[16]` 0.11824); 206.5 0.2699 / 0.3232 (`instruct split.m[0]` 0.26988, `split.c[0]` 0.32318); 206.6 0.1060,
  [0.1026, 0.1098], 0.9970 (`cos_request_change[18]` 0.99700); 206.7 0.0755 (0.07549), 0.0714-0.0795
  (`206.7_detail.target_ci` 0.07142-0.07946), second 0.0102 (0.01019); 206.8 49.05 / 46.81 / 2.25 / 16.87
  (`arming.slot`, `arming.concept`, `arming_label_shuffle.p95`); 206.9 45.97; 206.11 0.0305 / 0.1060 / 0.29, 0.0290 /
  0.27 (`A-tmpl|R-tmpl.rho_peak` 0.03049, `layers[18].rho` 0.02900); 206.12 32.04 (`206.12_detail.rises`
  midtrain-26B->midtrain-end), -0.043 (-0.04272), -6.12 to 15.09 (`target_ci`); 206.10 0.0622 / 0.0857 / 0.0813 /
  4 of 20 (`causal.json readout_a.mean_change`, recomputed over the 23 G3 rows: 0.06222, 0.08570, 0.08130, count
  4); 206.M1 0.1060, 0.0287 (`common_mode.M2|M.layers[18].rho` 0.02866); 206.M2 2.25 / 49.05; 206.M3 45.97; 206.M4
  0.5435 (`split.m[0].token_share_freq_removed` 0.54347); 206.M5 -0.2507 (`cos_delta_bM[18]` -0.25070).
- Split paragraph: 0.556 / 0.035 (`split.m[0].token_share` / `.constant` 0.03543), 0.118 / 0.043 (`split.m[16]`
  `.token_share` / `.constant` 0.04324), 0.270 / 0.031 (instruct `split.m[0]` 0.26988 / 0.03080), coverage 0.537 /
  0.515 (`split_coverage` 0.53676 / 0.51545).
- Reference: 1.342 / -0.342 (`reference[16].write_share.mixer` 1.34170 / `.mlp` -0.34186), OLMo layer 6 0.623 / 0.324
  (instruct `reference[6]` 0.62333 / 0.32375).
- Tuomi static: 0.960 / 0.974 / 0.970 / 0.949 (`gt.static[*].rho_resting_duty`), 0.098 / 0.130 / 0.118 / 0.072
  (`max(rho_random)`); causal 0.2289 / 0.0233 at layer 10 (qwen-chat `gt.causal.delta_duty` 0.22887, max
  `delta_duty_random` 0.02326), 0.2278 / 0.0228 (qwen-base), 0.1522 / 0.0447 at layer 5 (instruct).
- rho table (all eight rows): pretrain 0.085 / 0.082-0.087 / 0.115 / 0.090; midtrain-26B 0.067 / 0.065-0.069 / 0.092 /
  0.075; midtrain-end 0.075 / 0.073-0.079 / 0.090 / 0.080; SFT 0.151 / 0.147-0.155 / 0.170 / 0.169; DPO 0.153 /
  0.150-0.157 / 0.172 / 0.169; Instruct 0.164 / 0.160-0.168 / 0.182 / 0.179; qwen-base 0.076 / 0.073-0.078 / 0.099 /
  0.093; qwen-chat 0.106 / 0.103-0.110 / 0.117 / 0.149. All from `common_mode.A-tmpl|M.rho_peak`, `rho_peak_layer`,
  `rho_peak_ci`, `layers[L].kappa`, `layers[L].rho_raw`. Layers 6/7/6/6/6/6/17/18 match `rho_peak_layer`.
  Resamples 1,000, min 0.10046, max 0.11233 (`rho_peak_resamples`).
- Descriptive pairs: 0.0287 / 0.0321 / 0.1060 (`M2|M`, `N2|N`, `A-tmpl|M` rho_peak); OLMo 0.0748 / 0.0612 / 0.1635;
  cos_request_change 0.9952 at OLMo layer 6.
- Token share path: 0.556 -> 0.118 -> 0.197 (layer 18) -> 0.219 (layer 31); OLMo 0.270 -> 0.084 at layer 5; 206.5
  pairs 0.236/0.182, 0.195/0.116, 0.150/0.088, 0.118/0.080, 0.084/0.054, layer 0 0.270/0.323; `frequency_explained`
  0.023.
- Arming table (all six rows) and Instruct 49.05 / 46.81 / 2.25 / 16.87; activity-matched 35.0 / 37.5; identity 45.97,
  activity-matched 40.0, neuron layer 6 / 753 at 96.13.
- Causal table: 0.0622 / 0.0813 / 0.0857 / 0.1109; random k=1 -0.0889 to +0.1164, median +0.0063; 21 of 23 raised
  (recomputed); G3 23 of 24 (`causal.json G3.named_at_top1`); L* 18; readout_b -0.8587 plain / -0.8431 arm_k1;
  readout_c 3.0905 plain / 3.0988 arm_k1 / 3.1723 arm_k2, random range 3.0851-3.1154.
- Gates: G0 max_abs_logit_diff 0.0, G0.5 pass, G1 pass, G2 min 0.99586 at layer 24 (README says 0.9959 at layer 24 —
  correct), G3 23/24.
- Verdicts against `score206-output.txt` verdict JSON: identical for all 17 rows plus `206.8_activity_matched` and the
  headline sentence ("the assistant frame changes each position in its own way; no constant shift at this
  resolution") — the README title and L71-74 quote it exactly.
- Predictions table wording: compared word for word with Design L340-358 for all 17 rows; identical, including the
  confidence labels ("medium, Claude (Fable 5.2)", "rough 80%, Muse", "low, Claude (Opus 5.5)" for 206.12) and the
  parenthetical in 206.5 and 206.M1. The only difference is presentation (confidence in parentheses after the
  sentence rather than a separate column), which does not change the words. `PREDICTIONS.md` L1568-1584 carries the
  same text with the author first; consistent.
- Links: every relative link in `README.md` resolves to an existing file (checked with `test -e` from
  `results/part206/`): score206-output.txt, score.json, fig206.png, qwen-chat/causal.json, instruct/causal.json, all
  eight `<stage>/summary.json`, and `../../part206/{analyze206,causal206,score206,fig206}.py`,
  `../../PART206-DESIGN.md`, `../../PREDICTIONS.md`, `../../part206/AMENDMENT-1..5.md`, `../../part206/config206.json`,
  `../../part206/steps.txt`.

- Figure text vs sources: panel A labels 0.085 / 0.067 / 0.075 / 0.151 / 0.153 / 0.164 / 0.106 / 0.076 all match; panel
  C labels +0.062 / +0.081 / +0.086 / +0.111 match `mean_change`; "23 phrasings" matches `n_rows` in `fig206.py:42`
  (rows with `plain_top1_is_maker`); panel B bars match the arming table.

**Not reported as findings, checked and fine:** the title's "no constant shift at this resolution" is the registered
headline sentence verbatim (`score.json` `headline.sentence`, Design L406-408), and the README quotes it rather than
writing it; the verdicts agree with `score206-output.txt` in all 17 rows; the Predictions table wording and confidence
labels match the design word for word; the figure's panel C title ("it raises it, the (M2, M) null raises it more"),
panel B title and panel A's numbers are each supported by the keys above. The README's "Cold check: pending" line
(L295) was accurate as of this read.

## BLOCKERS

None.
