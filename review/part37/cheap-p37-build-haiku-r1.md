# REVIEW: p37-build (ab-part37, c78c778..92c2f6e), cold adversarial review

**Verdict: SHIP-WITH-FIXES.** The collection side (generation, screening, record construction, capture, stated readout) is sound as far as I could read it. The analysis side has two verdict-level leaks (37.9 and 37.4) that would produce a PASS on pure noise. Those must be fixed before anyone reads the host analysis for 37.4 or 37.9. The pilot stop path is not affected by them, so the pilot can run. Two registered rules (the pilot gate and 37.15's null set) need a decision from Weaver/Prime, not a code fix.

Scope read: `part37/collect37.py`, `part37/analyze37.py`, `part37/test_part37.py`, `part37/RUNBOOK.md`, `part37/REPORT-build.md`, `PART37-DESIGN.md` (Prime fold and Amendment 1), and the Part 37 rows of `PREDICTIONS.md`. Commits `c78c778..92c2f6e`.

Environment: this sandbox has no torch, transformers or sklearn in the default interpreter. I used `/home/elisepc/.venvs/audio/bin/python` (sklearn 1.9.0) for analysis code and numpy for the bf16 check. No model was loaded, no live service was called, and no credential file was read. Scripts are in this directory (`m1_inproj_leak.py`, `m2_pilot_gate.py`, `m3_records.py`, `m4_verdicts.py`, `m5_bf16_logprob.py`). Each imports the real code from the worktree.

---

## Findings, by severity

### F1. HIGH, INTRODUCED: 37.9 is scored on in-sample projections, and the own-greedy side is training data
- Where: `analyze37.py:539` fits `author_direction` on all primary rows (own_g = label 1, other_g = label 0). `analyze37.py:567-580` projects `own_g` and `own_s1/own_s2` with that same direction and computes `reg_shift`. `analyze37.py:757` passes 37.9 at `>= 0.5` SD.
- Trigger (`m1_inproj_leak.py`, pure Gaussian noise, width 2048, 100 questions, no signal anywhere):
  ```
  AS CODED (all-question fit): own_g mean +5.768  own_sampled mean -0.301  shift = +1.049 SD   (37.9 threshold >= 0.5 SD)
  HELD-OUT (fit on other 4 folds): own_g mean -0.149  own_sampled mean -0.294  shift = +0.025 SD (SD from all-question fit, same as coded)
  in-sample AUROC of the all-question direction on its own training labels: 1.000 (null truth: 0.5)
  ```
- Why it matters: on data with no own/other signal, the coded 37.9 statistic clears its threshold by a factor of two. The own_g rows were pushed toward the positive class during the fit, and own_sampled rows were never in it. This is a false PASS mechanism, not a power issue.
- Smallest fix: compute 37.9 only from held-out projections. For each of the 5 question-disjoint folds, fit on the train folds and project that fold's own_g and own_s rows. Keep the SD denominator from the same held-out projections. Add a test where pure-noise features must give a shift under 0.5 SD. The current test suite does not exercise `_fit_direction`, `_project` or `_prediction_results`, so it would not catch this.

### F2. HIGH, INTRODUCED: 37.4 Spearman uses the same in-sample direction on the records it was fitted on
- Where: `analyze37.py:591-598` builds `auth_projections` from `_project(store, capture_row, author_direction)` over every stated record. Those records include the training rows. `analyze37.py:759` passes 37.4 at `rho > 0.3`.
- Trigger: the same fit as F1. On noise the direction separates its own training labels perfectly (AUROC 1.000, measured above). Any association between the stated yes-probability and the own/other label therefore carries into the Spearman without shrinkage. I did not measure the size of the inflation under a real stated signal, so I am stating the mechanism, not a magnitude.
- Smallest fix: project each record with the direction fitted on the folds that exclude its question. Report the held-out Spearman as the verdict, and keep the in-sample number labelled as in-sample if it is shown at all.

### F3. MEDIUM, INTRODUCED: the registered pilot stop gate cannot fire on a null
- Where: `analyze37.py:854-874` (`every_layer_close`, `within_close`). The within-author test uses the best-over-layers AUROC against 0.5 ± 0.05.
- Trigger (`m2_pilot_gate.py`, pilot size of 20 questions, pure noise, three seeds):
  ```
  seed 1: gate={'every_layer_at_both_positions_within_0_05_of_its_label_shuffle_95th': False, 'both_within_author_contrasts_at_both_positions_within_0_05_of_chance': False, 'stop_and_report_surprisal_only': False}
     within best AUROCs (span_last): {'own_g_vs_own_sampled': 0.465, 'own_g_vs_own_diffq': 0.512}
  seed 2: ... within best AUROCs (span_last): {'own_g_vs_own_sampled': 0.718, 'own_g_vs_own_diffq': 0.573}
  seed 3: ... within best AUROCs (span_last): {'own_g_vs_own_sampled': 0.595, 'own_g_vs_own_diffq': 0.693}
  (every seed: STOP = False; regressed per-layer AUROCs of 0.35 to 0.78 against per-layer null 95ths of 0.68 to 0.74)
  ```
- Why it matters: the "C2 is surprisal and nothing else" stop cannot fire even when there is no signal at all. Max-over-layers AUROC on about 60 rows sits well above 0.55. The gate therefore always says "continue", and the cost is the full run (about 1.5 GPU-hours). It does not produce a false result, but it does make the pre-registered stop unreachable.
- Smallest fix: this is a registered-rule question, so it goes to Weaver/Prime rather than to a code change. The options are (a) calibrate the within-author threshold to the label-shuffled null at each position, or (b) keep the rule and state in the pilot report that it cannot stop on a null. See Question Q1 in the report.

### F4. MEDIUM, INTRODUCED: 37.15 is scored on a null that includes the reading model's own words
- Where: `collect37.py:381` builds `null_shuffle` from `own_g` words, and `analyze37.py:718` takes the mean over all three null kinds. The prediction says "on text neither model wrote".
- Trigger (`m3_records.py`, real `build_records` on the test fixture): `null_shuffle word bag == own_g word bag: True`. The null_shuffle record is the reading model's own answer with its words shuffled.
- Trigger (`m4_verdicts.py`): with the per-kind means `null_smol 0.02, null_pile 0.03, null_shuffle 0.64`, the pooled mean of 0.23 gives `37.15 -> PASS`. Two of the three nulls are below the 10% floor. The verdict reads only the pooled number.
- Why it matters: 37.15 passes or fails on the own-text kind, which is the one kind the prediction text excludes. Amendment 1 item 4 also calls shuffled own text "text neither model wrote", which is wrong on the bag-of-words measure above.
- Smallest fix: score 37.15 on the two non-own nulls (SmolLM3 and Pile), and report `null_shuffle` separately as a self-reference floor. This changes a registered rule, so it goes to Weaver/Prime. See Question Q2.

### F5. LOW-MEDIUM, INTRODUCED: 37.13 passes if either role cell is in range
- Where: `analyze37.py:740` (`within_ok = any(...)`).
- Trigger (`m4_verdicts.py`): `assistant=0.70, user=0.95 -> PASS`; `assistant=0.95, user=0.70 -> PASS`; `assistant=0.95, user=0.95 -> FAIL`.
- Why it matters: the prediction says "within a role cell", which is ambiguous. The code picks the lenient reading without saying so.
- Smallest fix: write the reading into the rule text. "Both cells" is the conservative choice. Report each cell's verdict either way.

### F6. LOW, INTRODUCED: 37.8 mixes layers
- Where: `analyze37.py:515` sets `author_best` from the raw best layer. `analyze37.py:539` and `:545` fit the regressed direction and take its cosine at that layer. `analyze37.py:754` tests the regressed best-over-layers AUROC, which may sit at a different layer.
- Why it matters: the 0.65 AUROC and the 0.5 cosine can come from different layers. The prediction says "at the best layer" without saying which one.
- Smallest fix: report the cosine at the regressed best layer as well, and state which layer the verdict uses.

### F7. LOW-MEDIUM, INTRODUCED: short sampled answers are dropped from the sampled cell without a per-cell count
- Where: `collect37.py:200-203` (`_first_tokens` returns `""` when the text is shorter than the target). `collect37.py:408` applies it to the sampled cells.
- Trigger (`m3_records.py`): `own_s1 ... target=65 len=0 empty=True`. The test suite asserts this at `test_part37.py:203-206`.
- Why it matters: the drop is length-selective. Samples that stopped early are removed from the 37.9 comparison, so the own_g-versus-own_sampled contrast is conditioned on sample length. The manifest reports only a global `empty_span_records` count.
- Smallest fix: report `n` per cell and the number of dropped samples per cell, and run the 37.9 comparison only on questions where both own_s1 and own_s2 are non-empty.

### F8. LOW, INTRODUCED: log-probs come from bf16 logits
- Where: `collect37.py:162` loads the model in bfloat16. `collect37.py:622` takes `output.logits[0]` and casts to float only after the bf16 rounding.
- Trigger (`m5_bf16_logprob.py`): per-token log-prob error from bf16 rounding is sd 0.012 nats at logit sd 2 and sd 0.024 nats at logit sd 4 (max 0.046). Averaged over 40 tokens, the mean error is about 0.004.
- Why it matters: the own-versus-other mean log-prob differences are larger than this, so it does not change any verdict. It is noise on the covariate that the surprisal control removes.
- Smallest fix: compute the output projection in float32 (`hidden.float() @ lm_head.weight.float().T`) for the span positions only.

### F9. LOW, INTRODUCED: `penalty="l2"` is deprecated in scikit-learn 1.8+ and will break in 1.10
- Where: `analyze37.py:160` and `:302`.
- Trigger: the test run under sklearn 1.9.0 emits `FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10`.
- Why it matters: the Windows runner's scikit-learn version is not pinned, so a future upgrade will break the host analysis.
- Smallest fix: drop the `penalty` argument. The default is already L2.

### F10. LOW, INTRODUCED: silent null degradation in the label-shuffle control
- Where: `analyze37.py:348` sets `best = 0.5`, and `analyze37.py:352` swallows every exception from `_fold_fit`.
- Why it matters: if a layer fails in every permutation, its contribution silently becomes 0.5. That lowers the null threshold and makes 37.14 easier to pass.
- Smallest fix: count the failed fits and raise if any layer fails in more than a few permutations.

### F11. LOW, INTRODUCED: float16 storage of hidden states can overflow to inf
- Where: `collect37.py:616` casts bf16 residuals to float16, and `:661` casts shards to float16. Values above 65504 become inf, and there is no finite check in `_capture_one`.
- Status: not measured on real activations. I have no model here. The risk is that a massive-activation dimension exceeds the float16 range. The failure would be caught later, but only as a sklearn `ValueError` after the capture has already run.
- Smallest fix: assert `np.isfinite` on each captured state in `_capture_one`, and fail the record with its id.

### F12. LOW, INTRODUCED: the suite does not test the scoring rules
- Where: `test_part37.py` covers fold disjointness, record construction and readout positions. It does not cover `_fit_direction`, `_project`, `_prediction_results`, the 37.9 or 37.4 leakage, the 37.13 and 37.15 rules, or the pilot gate.
- Trigger: F1, F2, F4 and F5 all pass the existing suite. Running it gives `Ran 13 tests ... errors=1, skipped=1` here. The error is `RealTemplatePositionTests` failing to import `transformers`, and the skip is the tiny-model test that needs torch. Neither is a code defect. The commit's "13/13 host" claim could not be reproduced in this sandbox.
- Smallest fix: add tests that feed pure-noise features through `_reader_analysis`-level helpers and assert that 37.9 and 37.4 do not pass.

### F13. LOW, INTRODUCED: generation and capture cannot resume
- Where: `collect37.py:548` writes the generation file only after all questions finish. `collect37.py:683` writes the capture index only after the last shard.
- Why it matters: a crash at question 99 loses the run, and a crash between the shards and the index leaves a directory the analysis cannot load. Both fail closed, so this is an operational cost rather than a wrong result.
- Smallest fix: write per question, or write to a temporary file and rename.

---

## What I checked and found sound

- **Readout positions.** The span's last token and the follow-up's first token are located from character ranges over the rendered template (`collect37.py:465-489`). The Qwen `enable_thinking=False` path matches `_chat_template`. Real-template tests exist, though they skip without the snapshots.
- **Record cells.** The 2x2 role-by-author cells are built correctly (`collect37.py:385-405`). Label frames number 32 per question. The fixture count is 43 (11 base + 32 label) and the test asserts it.
- **Edit and shuffle lengths.** The earlier fix is in place. The edit cell is kept and its token mismatch is recorded (`collect37.py:404-409`), and `test_part37.py:202` asserts it.
- **Identity screen.** It is applied to every generation row and drops a question from every cell (`collect37.py:219-238`).
- **Probe folds.** Every fit uses `GroupKFold` by question with a leakage check that raises (`analyze37.py:99-115`, `:151-152`). Residualisation and PLS in `_prepare_fold` are fitted on the training fold only (`analyze37.py:117-145`).
- **Pilot probes.** `_pilot_reader` uses only out-of-fold AUROC and the permutation control, so it does not depend on the in-sample direction (F1 and F2 do not affect the pilot).
- **Stated readout.** The yes and no token sets are disjoint, and the yes/no and A/B mappings are right, including the swapped forms (`collect37.py:492-506`, `:698-744`). Two-token mass is recorded per row.
- **Analysis exit codes.** The pilot returns 3 on stop, 0 on continue and 2 on a caught error. The runner stops at any nonzero code, so an uncaught exception also stops the run.
- **Runbook staging.** The Windows paths resolve from the runner's root, and the pinned revisions match `MODEL_INFO`.

---

## Decisions needed from Weaver / Prime (not code fixes)

- **Q1 (F3, pilot gate).** The registered stop rule cannot fire on a null: best-over-layers AUROC on about 60 rows exceeds 0.55 by chance, so the within-author test fails on pure noise. Options: (a) calibrate the within-author threshold to the label-shuffled null per position, or (b) keep the rule and say in the pilot report that it cannot stop on a null. Until this is decided, the pilot's "continue" should be read as "the rule is uninformative", not as evidence of signal.
- **Q2 (F4, 37.15).** Should 37.15 be scored on SmolLM3 and Pile nulls only, with the shuffled-own kind reported separately? The prediction text excludes own-written text, but Amendment 1 item 4 includes it. A decision is needed before the host analysis is read.
