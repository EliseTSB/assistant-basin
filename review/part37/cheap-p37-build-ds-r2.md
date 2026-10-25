# Round-2 adversarial review: Part 37 fold 1 (92c2f6e..773c14e)

Reviewer: cold, independent (did not write this code). Scope: `git -C /home/elisepc/green/worktrees/ab-part37 diff
92c2f6e..773c14e` — `part37/analyze37.py`, `part37/collect37.py`, `part37/test_part37.py`, `PART37-DESIGN.md`
(Amendment 2), `part37/REPORT-fold1.md`, `part37/BRIEF-fold1.md`. Every claim below is measured by a script in
`scripts/` in this directory, driving the REAL code from the worktree. Scripts were run with
`/home/elisepc/neuron-probe/.venv/bin/python -I` (scikit-learn 1.9.1, numpy, torch). No GPU, no live service, no
credential file was touched; every capture used was synthetic and written under a tempdir.

## Verdict

**DO-NOT-SHIP.** The fold fixes the round-1 scoring-rule defects it names (I re-verified each by measurement, below),
and the out-of-fold projection refactor is real. But the fold's headline fix — the pilot gate that decides whether the
full run proceeds — is still a gate on noise: at the pilot's registered size (20 questions) it says STOP on **8 of 10**
pure-noise seeds (`scripts/s1_pilot_noise.py`). The gate's null is not calibrated (its "95th percentile" sits at 0.67
where a calibrated bar is 0.5), so the gate is biased toward the registered stop, and a real signal has to beat a
~0.70 AUROC bar to continue. The fold also ships a test that does not catch the signing it claims to test (37.9), and
its report states a measured number ("old gate returned STOP for 0/3 noise seeds") that I could not reproduce for
the old gate's *own* rule — the three seeds it cites do not demonstrate what it says they do. Separately, the 37.15
rule as scored no longer matches the registered text once Amendment 2 item 2 is applied. None of these is style.

Severity key: **HIGH** = changes a verdict or a run decision; **MEDIUM** = a number or a test that misleads without
changing a verdict today; **LOW** = robustness/labelling. INTRODUCED = this change adds it or makes it reachable;
PRE-EXISTING = it was there at 92c2f6e and this change does not reach it.

---

## HIGH

### H1. The pilot stop gate fires on pure noise at the pilot's real size (INTRODUCED by the gate rewrite; the old gate was also broken, differently)

`part37/analyze37.py:1038-1063` (`_pilot_reader` stop gate), `:426-469` (`_permutation_threshold`).

Measured trigger — `scripts/s1_pilot_noise.py`: ten independent 20-question, 64-dim Gaussian captures with **zero**
own/other signal, run through the real `analyze37._pilot_reader`:

```
seed 1: STOP=True  ... seed 5: STOP=True
seed 6: STOP=False
seed 7: STOP=True  seed 8: STOP=True  seed 9: STOP=True
seed 10: STOP=False
STOP count on pure noise: 8 of 10
```

The registered rule (Amendment 1) is "the full run goes ahead unless, at every layer, ...". A gate that stops 80% of
the time on noise at the size it is registered at will stop the real run for a reason that is not in the data. The
fold's own test (`test_pilot_noise_stops_on_at_least_two_of_three_seeds_and_signal_continues`,
`part37/test_part37.py:491-504`) asserts the gate stops on **at least 2 of 3** noise seeds — it encodes the
false-stop behaviour as the expected behaviour. That is a test passing for the wrong reason: it would stay green if
the gate stopped on 100% of noise.

Why the gate is biased: the null bar is not calibrated. `scripts/s3_perm_null_calibration.py` (20 questions, 1 layer,
40 records, real `_permutation_threshold`):

```
max-over-layers 95th percentile: 0.6679
share of permutations with max AUROC > 0.5: 0.485   (a calibrated 95th bar should sit near 0.5 and ~5% exceed it)
```

and `scripts/s2_perm_null_calibration.py` (same, 16 dims, 200 permutations, 63 s):

```
max-over-layers 95th percentile: 0.6952
share of permutations with max AUROC > 0.5: 0.475
mean of permutation maxima: 0.5484  min: 0.5
```

Almost half of the permutations score above chance, so the 95th percentile lands near 0.67-0.70 instead of near 0.5.
The cause is in the null's definition (`analyze37.py:443-444`): one flip per question is applied to *every* record of
that question, so under the null every question still has a single label — question identity and label remain
perfectly aligned. A question-disjoint CV on those labels still learns question identity from the features.
`scripts/s10_perm_null_definition.py` isolates this: features built as a per-question offset plus noise, with **no**
own/other signal, give observed AUROC 0.502 but a null mean of 0.527 and a 95th of 0.611, with 47% of permutations
above 0.5. The null is not "labels are random"; it is "labels are random per question, but still a function of the
question", which is exactly the structure the probe can exploit.

The consequence for the real run: a true authorship signal has to clear ~0.70 AUROC at *every* layer at *both*
positions to continue. `scripts/s9_pilot_planted_signal.py` plants a signal of 1.0-2.8 SD in one dimension of
own_g/other_g (20 questions, 16 dims): the gate continues at every strength (good), but at 1.0 SD the authorship
check is *not* under the null at either position while `own_g_vs_own_diffq` is under the null at span_last — i.e. the
gate's decision is decided by which within-author contrast happens to wobble, not by the authorship effect.

Why it matters: this is the one gate that decides whether the 100-question run happens. A gate that stops on noise is
worse than no gate, because the stop will be read as "C2 is surprisal and nothing else" (Amendment 1's registered
conclusion) when it is a property of the null.

Smallest fix: (a) build the null by permuting *records across questions* (shuffle the own/other label within the
pooled record set, keeping record counts per question), so question identity carries no label; (b) compare the
observed **max-over-layers** AUROC with the null's max-over-layers 95th (one number, not a per-layer vector — see
M2); (c) calibrate the bar on a noise capture of the pilot's exact size and assert the false-stop rate is at most
5% (a test that fails when the gate stops on noise, not one that requires it). Also replace the test's
"STOP in at least 2 of 3" with "STOP in at most 1 of 20".

### H2. The 37.9 PASS/FAIL fixture does not catch the rule being made unsigned (INTRODUCED — the fold's own regression test)

`part37/test_part37.py:331-360`, `part37/analyze37.py:905-906`.

Measured trigger — `scripts/s17_sabotage_scoring.py`: revert each scoring-rule fix in the real source text and run the
fold's `test_each_scored_prediction_has_pass_and_fail_scoring_fixtures`:

```
37.11 absolute values: RED (caught)
37.13 assistant cell only: RED (caught)
37.15 pooled three nulls: RED (caught)
37.8 cosine at raw best layer: RED (caught)
37.9 signed -> absolute: GREEN (test does not catch the revert)
```

The 37.9 FAIL fixture (`test_part37.py:338`) sets `mean_logprob_regressed_shift_sd = 0.49`, which fails both
`shift >= 0.5` and `abs(shift) >= 0.5`. So the test cannot tell the two apart. The fold's brief (item 7) asked for one
PASS and one FAIL fixture through `_prediction_results` for every scored prediction; this one is present but is not a
regression test for the rule it covers. The current rule is signed and correct (`scripts/s6_scoring_edges.py`: shift
-5.0 SD -> FAIL), so this is a test defect, not a live wrong verdict.

Smallest fix: add a second mutation, `mean_logprob_regressed_shift_sd = -0.60`, and assert FAIL.

---

## MEDIUM

### M1. The fold report's claim about the old gate is not what its cited evidence shows (INTRODUCED — report, not code)

`part37/REPORT-fold1.md` item 3: "The old gate returned STOP for 0/3 noise seeds when run from the 92c2f6e analyzer."

Measured trigger — `scripts/s13_pilot_null_band.py` runs the **old** `_pilot_reader` (loaded from
`git show 92c2f6e:part37/analyze37.py`) on the same three noise captures the test uses:

```
1 {'every_layer_at_both_positions_within_0_05_of_its_label_shuffle_95th': False, 'both_within_author_contrasts_at_both_positions_within_0_05_of_chance': False, 'stop_and_report_surprisal_only': False}
2 {same}
3 {same}
```

The old gate's registered rule was "stop only if every layer is *within 0.05 of* its null AND both within-author
contrasts are within 0.05 of chance". On noise the old gate's null is also uncalibrated (its `within_0_05` check is
false because the null sits at ~0.6-0.7, not 0.5), so the old gate never stopped — for a reason unrelated to the
fold's fix, and the fold's new gate stops for the mirror-image reason. The report's "0/3" is true as a count but
reads as "the old gate was broken because it never stopped on noise", which is not the defect the fold fixed. A
reader of REPORT-fold1 would conclude the new gate is calibrated; H1 shows it is not. The report also states the old
code's 37.9 in-sample leak as "1.05 SD against a 0.5 bar" (Amendment 2 item 1, quoting Haiku). On the suite's noise
captures the old in-sample 37.9 shift is +0.08/+0.23/+0.22 SD (`scripts/s14b_old_37_9_exact.py`, using the old
module's own `_fit_direction`/`_project`), so that number is not reproducible here; it may be Haiku's different
fixture, but the report presents it as a property of the code. Do not cite a number the repo cannot regenerate.

Smallest fix: in REPORT-fold1, state the old gate's failure mode correctly (uncalibrated null in both directions) and
attach the script that reproduces each number.

### M2. The authorship pilot check compares per-layer AUROCs to per-layer null 95ths, not to the max-over-layers 95th the amendment names (INTRODUCED — matches the amendment's wording loosely, but is a different test)

`part37/analyze37.py:1042-1046` uses `null_layer_bars` (per-layer 95th) for authorship while
`part37/analyze37.py:1053` uses `null_bar` (max-over-layers 95th) for the within-author contrasts. Measured —
`scripts/s18_pilot_null_bar_usage.py`:

```
authorship gate uses per_layer_null_95ths: [0.7027499999999999]
authorship null max_layer_auroc_95th (not used by the gate): 0.7027499999999999
within-author gate uses max_layer_null_95th: {'own_g_vs_own_sampled': {'span_last': 0.643...}, ...}
```

Amendment 2 item 3 says each within-author contrast is compared with "95th percentile of the maximum over layers",
and the authorship check with "its null's 95th percentile at every layer". The code implements the latter. The
problem is that "at every layer" with an uncalibrated per-layer null (M/H1) means the authorship check is a
conjunction of ~29 comparisons, each at ~95%, so its false-stop probability is high by construction — the per-layer
95th of a *calibrated* null would already stop ~1-(0.95)^29 ≈ 77% of noise runs. This is the arithmetic behind H1's
8/10. Smallest fix: use one max-over-layers comparison for authorship too (and fix the null, H1).

### M3. 37.15 no longer matches the registered text (INTRODUCED by Amendment 2 item 2 — the amendment changes a registered prediction, not just the build)

`PART37-DESIGN.md` Amendment 2 item 2 and `part37/analyze37.py:912`. PREDICTIONS.md 37.15 (Weaver) reads: "on text
neither model wrote, the stated yes-rate is between 10% and 25%." Amendment 1 item 4 defined the null set as
SmolLM3's answer, a Pile span, **and** the reader's own greedy answer with its words shuffled. Amendment 2 item 2
removes the shuffled-own-text kind from the score because "it is the reader's own words". That is correct as a
measurement point — I confirmed the shuffle is the reader's own text (`collect37.py:452` `null_shuffle_text = "
".join(random.Random(...).sample(_words(own_g), ...))`) — but the scored prediction is now over a *different* null
set than the one registered, and the prediction row was not amended. The code is right; the record is now
inconsistent with the registered prediction. Smallest fix: file a PREDICTIONS.md amendment for 37.15 (scored on
null_smol and null_pile; self-text floor reported separately) before the verdict is written, the way Amendment 1
amended 37.5.

### M4. The minimal-edit rule produces the same replacement in almost every case (INTRODUCED by Amendment 2 item 8)

`part37/collect37.py:33-43` (`EDIT_WORDS`) and `:304-316` (`minimal_edit`). The replacement is
`next((word for word in EDIT_WORDS if ... abs(len(word) - len(original)) <= 1), None)` — a first-match scan of a
list whose first entries are 5 letters (`apple`, `bread`, `bridge`, ...). Measured — `scripts/s4_edit_rule.py` on
eight sampled answers:

```
           sunny -> apple              5   5  True          False
           quiet -> apple              5   5  True          False
           glass -> apple              5   5  True          False
            book -> apple              4   5  True          False
            legs -> apple              4   5  True          False
          things -> apple              6   5  True          False
         minutes -> bridge             7   6  True          False
           night -> apple              5   5  True          False
```

7 of 8 edits replace the word with "apple". The registered rule ("the replacement comes from a fixed list of common
nouns within one letter of its length") is satisfied literally, but the edit cell is then "the same token in almost
every record", so the edit contrast measures one token's effect, not a minimal content edit. Also, `EDIT_WORDS`
contains non-nouns (`silver`, `wood`, `music`, `sunlight`, `shadow`, `grass`, `glass`, `cloud`, `field`, `leaf`,
`bread`, `orange`) and one duplicate (`camera`). The test (`test_minimal_edit_logs_a_similar_length_replacement`)
only checks "not a stopword, in the list", so it cannot see this. Smallest fix: pick the replacement by a
deterministic hash of the original word (e.g. `EDIT_WORDS[hash % len(candidates)]`), and assert in the test that
eight sampled answers give at least six distinct replacements.

### M5. A crash between the metrics and index appends leaves a capture that loads, with no error, but is silently short (PRE-EXISTING ordering, INTRODUCED atomicity claim)

`part37/collect37.py:737-738` appends metrics first, then index. Measured — `scripts/s7_capture_atomicity.py`
(real `command_capture`, tiny torch model, crash injected on the third append):

```
capture crashed: power loss between the metrics and index appends
shards present: ['shard-00000.npy', 'shard-00001.npy']
index rows: ['q000:a', 'q000:b']
metrics rows: ['q000:a', 'q000:b']
manifest present: False
after retry, index rows: 4 shards: ['shard-00000.npy', 'shard-00001.npy']
FeatureStore loaded rows: 4
```

The prefix is loadable and a retry into the same directory is idempotent (good), and the manifest's absence is the
only signal that the capture is incomplete. `FeatureStore` (`analyze37.py:78-102`) never checks for `manifest.json`,
so an analysis run on a crashed capture proceeds on a partial set of questions with no warning. The report's item 9
claims "a crash leaves a loadable prefix" — true — but "loadable" is also "quietly incomplete". Smallest fix: have
`FeatureStore` refuse a capture without `manifest.json` (or record `complete: false` in a header row written first),
and count the index rows against `manifest["captured_records"]` when the manifest is present.

---

## LOW

### L1. `_group_splits`' pre-1.6 fallback is not the same split as the shuffle path (PRE-EXISTING)

`part37/analyze37.py:131-139`. The fallback remaps groups to `g00000..` in a shuffled order and lets unshuffled
`GroupKFold` balance by remapped group size. `scripts/s8_crossfit_fallback.py` (emulating the fallback exactly as
written, groups of size 2 and 6) gives test-fold record counts `[16, 16, 16, 16, 16]`; the modern
`GroupKFold(shuffle=True, random_state=37)` path on the same groups gives `[20, 20, 8, 20, 12]`. Both are
question-disjoint, so no leakage; but the fallback balances by record count and the modern path does not, so the two
environments produce different fold sizes and slightly different verdicts. The host has scikit-learn 1.9.1, so the
fallback is not taken today. Smallest fix: delete the fallback and require scikit-learn >= 1.6.

### L2. `run_probe`'s best layer is selected on the same held-out scores it reports (PRE-EXISTING)

`part37/analyze37.py:254`: `best = max(layer_results, key=...)` over the per-layer AUROC computed on the same
cross-validated predictions. 37.2, 37.8, 37.9, 37.13 and 37.14 all read `best_*` values. This is layer selection on
the test statistic, so the best-layer AUROC is optimistically biased (with ~29 layers at ~200 records, the bias is
small but real). The bootstrap CI (`_bootstrap_ci`) resamples records but not the layer choice. Not introduced here
and not reachable by a fix in this fold; flagging it because 37.14's comparison against a max-over-layers null is the
one place where the bias partially cancels (both sides take a max over layers) but the observed side also picks the
max of *observed* layers while the null picks the max of *permuted* layers — the comparison is only fair if the
observed statistic is also max-over-layers, which it is not: `qwen_reg_auc` is the best-layer AUROC and
`qwen_perm` is the max-over-layers null 95th. That makes 37.14 *harder* than the null, not easier — worth stating
in the verdict text.

### L3. The runbook still describes the pre-fold pilot rule and float16 (PRE-EXISTING text, not updated by the fold)

`part37/RUNBOOK.md:43` ("within 0.05 of its per-layer shuffled-label 95th percentile ... within 0.05 of chance") and
`:45` ("saves float16 shards"). The code now uses the max-over-layers null and float32. The runbook is what a person
follows on the Windows box; a person reading it will check the wrong rule when the pilot exits 3. Not a code defect,
but it is the operator-facing contract and it is wrong.

### L4. `_matched_subset` and `_length_position_probe` keep the old in-sample habit (PRE-EXISTING)

`part37/analyze37.py:528-529` and `:639-645` call `run_probe` on the subset, which is question-disjoint CV (fine), but
the matched subset's *membership* is chosen using mean log-prob from the same records it then scores — no leakage of
labels, but the subset is selected on a covariate computed from all questions. Not a verdict-bearing number today
(37.4 uses the Spearman, not the matched subset).

---

## What I checked and found sound

- **Out-of-fold projections are genuinely fold-local.** `scripts/s11_crossfit_leak.py`: perturbing one question's
  features moves exactly that question's projections plus the projections of the questions in the fold that trained
  the direction that projects it — the expected set matches exactly (`moved minus expected: ['q007']`, which is the
  perturbed question itself; `expected minus moved: []`). No held-out question enters the direction that projects it.
- **37.3, 37.4, 37.9, label, edit and null projections all read from `_crossfit_projections`**
  (`analyze37.py:688-737`), and the noise test (`test_crossfit_projections_block_noise_fit_leak_for_37_9_and_37_4`)
  fails when the old all-question fit is restored (`scripts/s16_sabotage.py` A: 0/1 passed).
- **37.15 scores only null_smol and null_pile** (`analyze37.py:577-604`), reports null_shuffle separately as
  `self_text_null_shuffle`, and fails closed when either scored kind is missing. Sabotage B (pooled) is caught.
- **37.11 is signed** (memory effect minus authorship effect, plus neutral-between on memory phrasing only) and its
  fixture catches the absolute-value revert (sabotage C / `s17`).
- **37.13 requires both role cells** in [0.60, 0.75]; the assistant-only revert is caught. Boundaries are inclusive
  (`s6`: 0.60/0.60 PASS, 0.75/0.75 PASS, 0.599 FAIL, 0.751 FAIL).
- **37.8 fits both directions at the regressed best layer**; the raw-layer revert is caught.
- **37.12** passes with one layer at 4/5 folds >= 0.80; **37.2** uses strict `< 0.6` (0.599 PASS, 0.60 FAIL);
  **37.15** uses inclusive [0.10, 0.25]. All match the registered text.
- **Question-disjoint folds**: `folds_are_question_disjoint` is checked in `_fold_fit` and
  `_crossfit_projections`; the probe split test runs.
- **The pilot gate fails closed** on a missing/non-finite layer or null (`test_pilot_gate_fails_closed...`,
  `analyze37.py:1016-1036`), and `main()` returns 2 on `AnalysisError`.
- **The pilot gate continues on a planted signal** at every strength tried (1.0-2.8 SD, `scripts/s9`).
- **Capture**: float32 storage, `isfinite` check per record with the record id in the error, float32 logits for span
  log-probs, one forward per record, and a retry into the same directory is idempotent (`s7`).
- **Generation writes are atomic** and the row format is unchanged (`write_jsonl_atomic` writes the same rows as
  `write_jsonl`); the suite's atomic-writer test fails if the old file is clobbered.
- **The identity screen** drops a question on any flagged answer (`s5`); the shuffle is of the reader's own
  *screened* answer, so the shuffle itself is not an identity hole — only a labelling one (M3).
- **Suite**: `26 passed, 10 subtests passed` in 99.8 s on the host venv (`scripts/s15_suite.py`), matching the
  report. The tiny-model tests are defined only when torch imports; with torch absent they become a single
  `self.fail("unreachable")`-guarded skip — on the host they ran (the capture tests exercised the real forward).
- **No feature flag** gates any path in this change; there is nothing to check for "flag OFF refuses".

## Self-assessment

I did not run the real models; every capture is synthetic. The null-calibration result (H1) is measured at 1 layer
and 16-64 dims to keep it fast; with the real 29 layers the per-layer conjunction (M2) makes the false-stop rate
*worse*, not better, but I did not measure the 29-layer number. The 37.9 old-code number in M1 (0.08-0.23 SD) is on
the suite's 8-dim noise fixture; Haiku's cited 1.05 SD may be a different fixture, so I state M1 as "not reproducible
here", not "wrong". The pilot null's exact cause (question identity surviving the per-question flip) is measured in
`s10` on a constructed feature set; I did not ablate the real feature set to separate question identity from other
structure. H1's severity rests on the 8/10 number, which is the pilot's registered size and the code as shipped.
