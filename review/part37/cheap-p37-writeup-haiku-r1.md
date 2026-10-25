# Cold review: p37-writeup (9dbb851..d3b0a85)

Reviewer: cold adversarial pass, Haiku 5.5 worker. Read: README.md (Part 37 section, lines 883-937, and front-page line 94-95), PREDICTIONS.md (rows 37.1-37.15, lines 1537-1551), part37/analysis-37.json (every number below), PART37-DESIGN.md and Amendments 1-3, part37/analyze37.py (ranged reads), full stated and records files. Scripts are in `scripts/` in this domain; output of `checks.py` is in `scripts/checks-output.txt`.

## Verdict: SHIP-WITH-FIXES

Every verdict cell and every number in the 15 rows matches analysis-37.json, and the verdicts follow from the registered rules, with one wrong reason stated (37.11). Two claims in the prose are contradicted by the JSON: the follow-up paragraph (HIGH-1) and the 37.11 explanation (HIGH-2). Both are one-paragraph fixes. The rest are reader-facing omissions. None of this needs the analysis re-run except HIGH-1's guard, if the follow-up number is kept.

## Findings

### HIGH-1 (INTRODUCED in the write-up; the code path is PRE-EXISTING) The follow-up paragraph says regression "does not touch" the follow-up signal, and the JSON contradicts it

Where: README.md:929-935 ("The follow-up position reads content, not authorship").

Measured (scripts/checks.py, read-only over analysis-37.json, `readers.{qwen,olmo}.probes.authorship_assistant_primary.followup_first`):
```
raw       L0..5: [0.5, 0.9859, 0.99, 0.9925, 0.9957, 0.9959]  L28: 0.9921
regressed L0..5: [1.0, 0.5398, 0.5715, 0.6157, 0.6413, 0.6303]  L28: 0.6293
regressed best_layer 0 best_auroc 1.0
qwen: raw L1 0.9859  regressed L1 0.5398  regressed best (L0) 1.0000  raw-L0 0.5
olmo: raw L1 0.9821  regressed L1 0.6221  regressed best (L0) 1.0000  raw-L0 0.5
```
So regression drops the follow-up own/other AUROC from about 0.98-0.99 to 0.54-0.64 at every layer from 1 up. The only place it stays at 1.0 is layer 0, where the raw feature is a constant (raw AUROC exactly 0.5). That 1.0 is an artifact. In scripts/const_noise.py, using the real `analyze37._residualize` and `_prepare_fold`:
```
constant feature: max |raw residual| train = 4.44e-15, test = 4.44e-15
label-correlated? corr(label, residual_sum) train = 0.532
test AUROC of a train-fitted centroid classifier on the regressed layer-0 feature: 0.9275
control (covariate independent of label): test AUROC = 0.5700
```
The residual is float noise, but the log-prob covariate differs by label (own answers are more predictable), so the noise carries the label. `_prepare_fold` standardizes it to unit variance (analyze37.py:164-167 has no constant-feature guard in the regressed branch; the subspace branch's comment about constant features does not cover it). sklearn was not installed here, so I used a numpy stand-in for the scaler and a centroid classifier in place of logistic regression; the 1.0 in the JSON is consistent with the same mechanism but I did not reproduce it with sklearn.

Why it matters: the paragraph's conclusion ("reads content, not authorship") rests on "regression does not touch it". The measured regressed values at layers 1-28 say it does. "About 1.00 from layer 1" is also 0.98-0.99.

Smallest fix: (a) rewrite the paragraph to give the regressed layer-1+ values (0.54-0.64) and drop the conclusion, or (b) fix the guard and re-run. Guard tested in scripts/fix_check.py: zero the columns with `np.ptp(xtr, axis=0) == 0` before residualizing. Output: current 0.9275, with guard 0.4750. Use ptp, not std: `std == 0` flags only 66 of 2048 columns for an identical-row matrix because np.std is not exact there, while ptp flags all 2048.

### HIGH-2 (INTRODUCED) PREDICTIONS.md:1547 gives a wrong reason for the 37.11 FAILED verdict

The row says "the label moves the memory phrasing 9.2 points and the authorship phrasing -0.01, so the gap passes, but ...". The registered rule (Amendment 2 item 5, PART37-DESIGN.md) needs the signed gap to be at least 10 points. The observed signed gap is 0.0921 (memory 0.0920 minus authorship -0.0001), which is below 0.10. So the first clause fails, and the "gap passes" wording is wrong. The verdict (FAILED) is still right, because the neutral-header clause also fails (0.08 is not between 0.25 and 0.16).

Measured (checks.py): `memory self-other = 0.0920 (= 9.2 pts); authorship self-other = -0.000146; signed difference = 0.0921; registered bar >= 0.10 -> first clause FAILS`. The JSON agrees (`predictions/37.11/observed/self_minus_other_effects`, rule text "minus signed authorship effect is >= 0.10").

Fix: replace "so the gap passes, but" with "the signed gap is 0.092 against the 0.10 bar, so it fails too;".

### MEDIUM-1 (INTRODUCED) The 37.1 "role readable at 0.996" result is not diagnostic: the position/length-only control also gets AUROC 1.0

Measured (checks.py): `qwen: role AUROC 0.9959; length+position-only AUROC 1.0000`; `olmo: role AUROC 0.9989; length+position-only AUROC 1.0000` (JSON `readers.*.probes.role_pooled.span_last.length_position_only.auc`). The role contrast (user vs assistant slot) changes the absolute token positions of the span, so position alone separates it perfectly. The `length_position_only_label` text in the JSON says the control "cannot fail by construction", which is true only for the authorship contrast, where the same control is 0.5 (`authorship_assistant_primary`, both readers; sound).

Where: README.md:899 (row 37.1 of the Part 37 table, lines 898-909); PREDICTIONS.md:1537 (the 37.1 result cell).

Why it matters: 37.1 is the positive control "replicating Ye et al.". As written, the 0.996 reads as a representation of role. The data say position gives the same result.

Fix: state the control result in the row and in the README, and say the role pass shows slot position is linearly readable, not a role representation beyond position.

### MEDIUM-2 (INTRODUCED) The write-up omits the layer, n and intervals a reader needs for the headline numbers

- Layer. 37.3, 37.4 and 37.9 project onto the direction at layer 28, which is the raw best assistant-slot layer (`readers.qwen.authorship_direction.best_layer = 28`, `mode: mean_logprob_regressed`, `regressed_probe_best_layer = 19`). 37.8's cosine uses the regressed best, layer 19. The write-up never gives the layer for 37.3, 37.4 or 37.9. Layer 19 was not checked against 37.4 (only layer 28 is reported).
- n. The headline AUROCs use 200 records (100 questions x own/other), not 400 and not 4,193. Only 37.4 uses 4,193 (JSON `stated.authorship_yes_spearman.n`).
- Intervals (checks.py, bootstrap CIs in the JSON): raw 0.9455 CI [0.917, 0.971]; regressed 0.5534 CI [0.476, 0.629], which includes 0.5; subspace 0.6324 CI [0.562, 0.704], which excludes 0.5. The write-up's "the signal is mostly predictability" is supported, but the residual after subspace removal is above chance, and the write-up does not say so.

Fix: add layer, n and CIs to the three sentences that carry 0.946, 0.553 and 0.632.

### MEDIUM-3 (INTRODUCED) "Neither model's answer tracks authorship" and "near 0 for every record" are stated more strongly than the data

- OLMo's Spearman is -0.086 (n=4186, JSON `readers.olmo.stated.authorship_yes_spearman.spearman_rho`). It is not reported anywhere in the write-up. Records are clustered four per question, so the nominal p is overstated, but the value is not zero. Report it.
- "Qwen's stated yes-probability is near 0 for every record" (PREDICTIONS.md:1540). Measured (scripts/per_record.py): Qwen authorship_yn max p_yes = 0.2227 (record q061:base:other_g:assistant), no record above 0.5. "Near 0 for every record" overstates a 0.22 maximum. Say "no record above 0.23".

### MEDIUM-4 (INTRODUCED) "held" means two things in the write-up

The count line uses "244 held" for passed verdicts. Rows 37.5, 37.6 and 37.10 are "held by Amendment 1", which means not run. The JSON gives them verdict `HELD`, PREDICTIONS.md writes `NOT RUN:` with the same reason, and README.md:912 says "were held". The count script reads these as unscored (checked, below). A reader will read "held" as "passed" in one place and "not run" in another.

Fix: use "not run (held by Amendment 1)" in the JSON verdict, the rows and the README sentence.

### LOW-1 (INTRODUCED) Several sentences are not traceable to the repo

- README.md:896 "generation 1 h 47 min, capture 30 min, stated 49 min": not in any manifest or report under part37/ (grep). Source it or remove it.
- README.md:936 "Snoek, Miletić and Scholte (NeuroImage 2019) found free of the below-chance bias...": cited only here; absent from PRIOR-ART.md and PART37-DESIGN.md (grep). I could not check it offline.
- README.md:937 "Answers often reached the 96-token cap": max_new_tokens = 96 is in the manifest, but no token counts are stored, so "often" is not measured.

### LOW-2 (INTRODUCED disclosure; the length-match code is PRE-EXISTING) "Length-matched in its own tokens" is exact for own and other, not for the derived controls

Measured (scripts/mismatch.py, mismatch2.py): all 400 own_g and other_g base records have token_mismatch_count 0. The derived controls do not: Qwen has 98 records with a nonzero mismatch (70 null_shuffle, 27 edit, 1 own_s1), OLMo 112 (75 null_shuffle, 26 edit, 6 own_diffq, 3 own_s2, 2 own_s1), with magnitudes of 1 to 10 tokens. The manifest's `retokenization_mismatches` (761 Qwen, 826 OLMo) is mostly six empty spans scored -96 (576 of 761), plus 185 small mismatches. collect37.py:477-487 and :514 do this. The write-up does not mention either.

Fix: say that own and other are length-matched exactly and the edit and shuffle controls are approximately matched, and note the counts.

### LOW-3 (INTRODUCED, in this diff) `_fit_job_safe` discards the cause of a failed fit

analyze37.py:281-285 wraps every fit in `except Exception: return None`. The permutation path then raises a generic `AnalysisError("fit failed")` (analyze37.py:528). That is not silent, since it raises, but the original exception is gone. `ParallelFitTests` (test_part37.py, last 42 lines of the diff) compares serial and parallel only for raw mode with no covariates. It never runs `_map_fits_safe` with a failing job.

Fix: keep the exception text in the None-path (or drop the wrapper), and add a test where one job raises.

### LOW-4 (PRE-EXISTING data) Two-token mass for OLMo is not reported

The stated readout's total two-token mass averages 0.857 for OLMo (min 0.43 memory, 0.54 authorship) and 1.00 for Qwen (JSON `stated.*.mean_total_two_token_mass`). The write-up mentions only the Qwen mass. `low_mass_flag` uses the median (analyze37.py:632), so per-record leakage is not flagged. I checked the effect: OLMo's mean yes-probability is 0.2994 both raw and normalized over yes+no, so the reported means are not changed by the normalization. Say so in the write-up.

### Not a finding: the 523 count line

I ran the repo's `count_predictions.py` on the PREDICTIONS.md from 9dbb851 and from d3b0a85 (copies in scripts/pred-before.md and pred-after.md):
```
BEFORE  523 rows: held 240, failed 177, mixed 10, untestable 7, reading 2, unscored 87
AFTER   523 rows: held 244, failed 184, mixed 10, untestable 7, reading 2, unscored 76
```
The README line (523 / 244 / 184 / 10 / 9 untestable-or-rule / 76 not yet scored) matches. The old README's 519 was already stale before this change (PRE-EXISTING); the diff adds no rows, so the 519-to-523 change is not from this commit.

## Checked and found sound

- Every README number in the section against analysis-37.json: 0.996 and 0.999 (`predictions.37.1.observed`), 0.946 and 0.553 (`probes.authorship_assistant_primary.span_last`), 1.5% and -0.004 / -0.282 (`predictions.37.3.observed`), 0.007 (`predictions.37.4`), 0.04 cosine (`authorship_direction`), 0.632 subspace, -0.31 / +0.06 (`predictions.37.9`), 0.875 / 0.79 (`37.13`), 0.553 vs 0.642 (`37.14`), 0.0003 / 0.0011 / 0.0001 / 0.00 (`stated.header_by_form` and `authorship_yes_by_span_kind`, own_g, other_g, null_smol, null_pile), OLMo 0.27 (own_g 0.266), 0.31-0.35 (nulls 0.312-0.347), Qwen memory 0.25 / 0.16 / 0.08 (`stated.header_by_form.memory_yn`), OLMo 0.770 and 0.564 and null 0.641 (`probes.authorship_assistant_primary.span_last`, `label_shuffled_control`), and kept 100 / dropped 0 (`full/kept.json`).
- Verdicts against the rule text in the JSON and in Amendment 2: 37.1 PASS, 37.2 PASS, 37.3 PASS, 37.4 FAIL (0.007 < 0.30), 37.8 FAIL (0.553 < 0.65), 37.9 FAIL (-0.138 SD < 0.5), 37.12 PASS (layers 19-21, 26-28), 37.13 FAIL (0.875 and 0.79 outside [0.60, 0.75]), 37.14 FAIL, 37.15 FAIL (0.008%). 37.11 FAIL is correct; see HIGH-2.
- Out-of-fold claims (Amendment 2 item 1): `_crossfit_projections` (analyze37.py:445-478) fits each direction on training folds and projects held-out questions only; regression inside `_prepare_fold` is fit on training folds (sound).
- Authorship length-position control is 0.5 for both readers (sound: the authorship contrast is not position-confounded).
- P37_JOBS unset or 1 runs the serial path; `_map_fits_safe` raises if a permutation fit returns None (no silent pass).
- Amendment 3 is consistent with the README wording ("withdrew the pilot gate").

## Not verified

- The test suite was not run. sklearn is not installed in any Python on this machine (system python3, ~/.local, stt venv), so `test_part37.py` cannot import. numpy and scipy are available. I did not install packages.
- The analysis itself was not re-run (captures are local; sklearn missing).
- Citations (Snoek et al., Meisty Study 8) were checked only against the repo; Meisty is in PART37-DESIGN.md:9, Snoek is not in any repo file except the README.
- The writing time and the 96-token frequency claims are not in the repo.
