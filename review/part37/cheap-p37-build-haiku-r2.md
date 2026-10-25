# Cold adversarial review: p37-build, round 2 (773c14e, Amendment 2)

Scope: `git -C /home/elisepc/green/worktrees/ab-part37 diff 92c2f6e..773c14e` (part37/analyze37.py, part37/collect37.py,
part37/test_part37.py, PART37-DESIGN.md Amendment 2). Line numbers below are 773c14e. Scripts are in `scripts/`, captures
and gate JSON in `tmp/`, the summary tables in `gate-summary-rho081.txt` and `gate-summary-rho097.txt`. Interpreter:
`/home/elisepc/neuron-probe/.venv/bin/python` (numpy, scikit-learn, scipy). No live service, no credential file, no write
to the worktree or to /home/elisepc/green/DiscordBot.

HEAD moved during the review: `87db16b` ("Part 37 after DS round 2") is on top of 773c14e. The brief fixes the scope at
773c14e, so all findings are against 773c14e. Section 4 notes what changed at HEAD.

## Verdict

**SHIP-WITH-FIXES.** The Amendment 2 scoring rules are implemented as the amended text reads (section 2). The test suite
passes as claimed. Two findings need action before the pilot is relied on: F1 (a partial capture was loaded silently;
FIXED at HEAD, measured) and F3 (the amended pilot gate almost never stops on pure noise at the real layer count; this
is a design decision for the owner, with numbers). F2 is pre-existing and a follow-up. None of this blocks the GPU steps.

## Findings

### F1: INTRODUCED, MEDIUM. A killed capture loads silently. FIXED at HEAD 87db16b (measured)

- At 773c14e, `FeatureStore.__init__` (`part37/analyze37.py:78-81`) reads `index.jsonl` and `metrics.jsonl` and never
  opens `manifest.json`. `command_capture` (`part37/collect37.py`, from `command_capture` at about line 705) now appends
  the index per completed question and writes the manifest last. Before 773c14e, the index was written only at the end, so
  a killed run failed loudly. Now a prefix is loadable and nothing marks it as partial.
- MEASURED at 773c14e: `scripts/partial_capture.py` keeps the rows for questions q000 to q011 of a 20-question capture,
  drops the manifest, and calls the real `FeatureStore`:
  ```
  manifest.json present: False
  loaded rows: 60 questions: 12 q000 .. q011
  analysis.py mentions manifest: False
  ```
  No error. `pilot_check` and `_reader_analysis` read the same store, so a partial capture would feed the gate and the
  verdicts with 12 of 20 questions. The stated readout is built from the full record file, so the stated numbers would
  cover questions the capture lacks.
- STATUS at HEAD 87db16b: the analyzer now refuses a capture without `manifest.json`, and refuses one whose
  `captured_records` differs from the index length. Measured with `scripts/partial_capture.py` against HEAD:
  `AnalysisError: ... no manifest.json, so the capture may be partial`; with a manifest claiming 100 against 60 rows,
  `AnalysisError: manifest says 100 records, index has 60`.
- Residual at HEAD, small: a stale manifest from an earlier complete run in the same directory is not removed when a
  new capture starts. A crashed rerun is then refused only if the old count differs from the new index length. Fix:
  delete `manifest.json` at the start of `command_capture`.

### F2: PRE-EXISTING, LOW-MEDIUM. The pilot is not limited to its registered 20 questions

- `PART37-DESIGN.md:134` registers the pilot as "the first 20 questions". `_pilot_reader` (`part37/analyze37.py:991-996`)
  uses every captured base row, and `pilot_check` (`:1077-1086`) prints "Stage A only, first 20 questions" without checking
  it. The 92c2f6e `_pilot_reader` has the same shape, and HEAD 87db16b does not change it (checked by reading).
- The 20-question size holds only if the operator builds a 20-question record file. Nothing checks it, and a 100-question
  capture gives a pilot gate on 100 questions under the pilot's label.
- Smallest fix: assert `question_count == 20` in `pilot_check`, or take an explicit cap. Follow-up, not a hold.

### F3: INTRODUCED, MEDIUM, a decision for the owner. The amended pilot gate almost never stops on pure noise at the real layer count

- The gate (`analyze37.py:1038-1073`) stops only if, at both positions, the authorship AUROC is under its per-layer null 95th
  at EVERY layer, both within-author contrasts are under their max-over-layers null 95th at both positions, and both
  readers meet this. The every-layer requirement across 29 layers is the part that does not survive.
- The test that claims the behaviour, `test_pilot_noise_stops_on_at_least_two_of_three_seeds_and_signal_continues`
  (`part37/test_part37.py:491-504`), builds its noise captures with `rng.normal(size=(2, 1, 8))` (helper at `:145-171`):
  one layer, width 8, one reader. With one layer, the every-layer condition is a single 5% event, so the test passes for a
  reason that does not hold at pilot size. The real pilot has 29 layers and two readers.
- MEASURED with the real `analyze37.pilot_check` at pilot shape: 20 questions; the five base assistant kinds; 29 layers;
  width 128 to keep the runs tractable; 200 label-shuffle permutations. Each pair is one qwen and one olmo capture.
  Cross-layer correlation is set by `scripts/gen_capture.py` (the `rho` argument).

  At rho = 0.81 (six noise pairs and two planted-signal pairs; `gate-summary-rho081.txt`):
  ```
  file                   joint_STOP qwen_STOP olmo_STOP  layers-over-null (span_last, followup_first) per reader
  gate-noise101.json     False      False     True       qwen(0, 2) olmo(0, 0)
  gate-noise102.json     False      True      False      qwen(0, 0) olmo(0, 0)
  gate-noise103.json     False      False     True       qwen(0, 4) olmo(0, 0)
  gate-noise104.json     False      False     False      qwen(0, 11) olmo(1, 1)
  gate-noise105.json     False      False     False      qwen(0, 7) olmo(0, 5)
  gate-noise106.json     False      False     False      qwen(7, 1) olmo(2, 0)
  gate-sig201.json       False      False     False      qwen(9, 1) olmo(2, 0)     <- planted 0.5 SD
  gate-sig202.json       False      False     True       qwen(3, 3) olmo(0, 0)     <- planted 0.5 SD
  joint STOP 0/8; qwen STOP 1/8; olmo STOP 3/8
  ```
  At rho = 0.97 (three noise pairs; `gate-summary-rho097.txt`):
  ```
  gate-hr301.json        False      False     False      qwen(0, 0) olmo(0, 0)
  gate-hr302.json        True       True      True       qwen(0, 0) olmo(0, 0)
  gate-hr303.json        False      True      False      qwen(0, 0) olmo(0, 7)
  joint STOP 1/3; qwen STOP 2/3; olmo STOP 1/3
  ```
- Reading: the per-layer exceedance rate is close to the nominal 5%. Across the six rho = 0.81 noise pairs there are about 41
  exceedances in 696 layer tests, against about 35 expected, so the per-layer nulls are not miscalibrated. The failure is
  the AND over 29 layers at two positions: exceedances cluster, and one exceedance sends the gate to "continue". The stop
  rate on noise depends strongly on how correlated the layers are (0 of 6 pairs at rho 0.81, 1 of 3 at rho 0.97). Real
  residual-stream correlation is not measured here.
- The planted 0.5 SD signal continued jointly in both pairs. Olmo's own gate stopped on one of the two signal captures
  (sig202), so per-reader detection at this effect size is unreliable.
- Why it matters: the amendment exists so that pure noise can meet the stop rule (`PART37-DESIGN.md:156-158`). As built, the
  default on noise is "continue", so the stop is close to inert. That costs GPU time and does not produce a false science
  claim, which is why the severity is MEDIUM and not HIGH. HEAD 87db16b records that the author considered this and did
  not change it ("H1 not taken"). The rule is registered, so the choice belongs to the owner, and the owner should see
  these numbers first.
- Options, for the owner to pick: (a) keep the registered AND-over-layers rule and accept that the pilot will almost
  always continue; (b) compare each condition's max over layers with its max-over-layers null 95th, which makes each
  condition one 5% event, then measure P(STOP | noise) over at least 20 seeds at the real layer count; (c) add a
  real-shape noise test in place of the one-layer, width-8 test.

### F4: PRE-EXISTING, LOW. Some writes are not atomic

- `command_build_records` (`part37/collect37.py`, about `:629-645`) calls `write_jsonl(args.out, records)` and writes its
  manifest with `write_text`. `command_stated` (about `:774-819`) does the same. The diff does not touch either. A crash
  leaves a torn file. `read_jsonl` rejects a torn line, so the failure is loud and not silent. Follow-up: use
  `write_jsonl_atomic` and `write_json_atomic`, which already exist.

## 2. Amendment 2 items against the code at 773c14e

- Item 1 (out-of-fold projections). Every verdict input reads `projected_regressed` or `projected_raw`, built by
  `_crossfit_projections` (`analyze37.py:373`): c3 label and author effects, the 37.9 shifts, the projection summaries, and
  the 37.4 Spearman. Each direction is fit on the four folds that exclude the held-out question and then projects every span
  row of that question. The 37.8 cosine is between two full-data directions, not a projection, so item 1 does not cover it.
- Item 2 (37.15). `_null_readout_summary` (`:577`) scores `null_smol` and `null_pile` on frame base. `null_shuffle` is
  reported as `self_text_null_shuffle`. The verdict (`:912`) reads the scored-kinds mean.
- Item 3 (pilot gate). Authorship is compared per layer with its per-layer null 95th, and within-author contrasts with
  their max-over-layers null 95th (`:1038-1073`). Missing or non-finite layers and nulls raise `AnalysisError`, which
  `main` returns as exit 2. The code matches the amended text. The behaviour is F3.
- Item 4 (37.9 pairing and counts). `_paired_sample_questions` (`:309`) requires own_g, own_s1, own_s2 and other_g, and
  reports per-cell counts and drops. The SD (`reg_sd`, `baseline_sd`) is computed over paired questions only (about
  `:720-727`). The amendment says "own_g and other_g" without that restriction; a defensible reading, listed in section 5.
- Item 5 (37.11 signed). `memory_effect - auth_effect >= 0.10` (`_prediction_results` `:910-911`), with the neutral header
  between self and other on the memory phrasing only (`:909`). Matches the amended text in `PREDICTIONS.md`.
- Item 6 (37.13 both cells). `p13_pass` requires every slot in [0.60, 0.75] (`:897-898`).
- Item 7 (37.8 layer). `_role_cosine_at_regressed_best_layer` (`:415`) uses the regressed best layer for both the AUROC and
  the cosine.
- Item 8 (minimal edit). Stopword filter and noun list at 773c14e (`collect37.py:300-317`). The replacement is the first list
  entry of matching length, so it is almost always "apple". MEASURED at 773c14e: "light" to "apple", "breakfast" to
  "mountain", "window" to "apple". HEAD 87db16b replaces this with a hash-chosen word; I did not measure the HEAD version.
- Item 9 (precision). Hidden states are float32 with a finiteness check (`_capture_one`), and span log-probs come from
  float32 logits. Models load in bfloat16 (`load_model`), so the finiteness check is unlikely to fire on real states.

Scoring rules for 37.1 to 37.15 were checked one by one against `PREDICTIONS.md` and the amended text. All match, with the
exceptions in section 5.

## 3. Test suite

- At 773c14e: `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider part37/test_part37.py` gives
  `26 passed, 10 subtests passed in 502.26s`, exit 0, with no skips (so the tiny-model torch tests ran).
- Weaknesses: the pilot-noise test uses one layer and width 8 (F3). The 37.4 part of the crossfit leak test uses synthetic
  0.9/0.1 labels rather than stated p_yes; it still shows the leak, but the 37.4 number it checks is not the production
  quantity.

## 4. HEAD 87db16b (moved during the review; outside the brief's scope)

- The analyzer change at HEAD is the manifest check in `FeatureStore` (F1). My gate runs imported the 773c14e analyzer,
  since their captures carry no manifest and still loaded, so the gate numbers in F3 are for 773c14e and apply to HEAD's
  gate code, which is unchanged.
- The collector change at HEAD (hash-chosen replacement, de-duplicated noun list, 37.15 notes, M5) was not reviewed.

## 5. Self-assessment: what I am unsure of

- Synthetic captures only. Width 128 (real models are 2048), 200 permutations, 20 questions, 29 layers, cross-layer
  correlation tested at two values (0.81 and 0.97). The F3 magnitudes could shift with real hidden states.
- I did not run `_reader_analysis` end to end; it needs stated files and a full capture. Coverage is by reading, the
  suite, and the probes above.
- 37.9 SD restricted to paired questions: a defensible reading of Amendment 2 item 4, but not the literal text.
- 37.8 cosine uses full-data directions. It is not a projection, so I do not read it as a leak, but a strict reading of
  item 1 could.
- The minimal-edit "apple" result describes 773c14e only.
- `scripts/` and `tmp/` hold my synthetic captures, gate JSON and logs. A `plan.json` in this directory is not mine and I
  left it alone.

## 6. What I checked and found sound

- Question-disjoint folds: `_group_splits` (`analyze37.py:124`) and `folds_are_question_disjoint` are used by `_fold_fit`,
  `_crossfit_projections` and the label-shuffle null.
- Regression residualisation is fit on training folds only: `_prepare_fold` (`:150`), `_fit_direction` (`:335`), and
  `_project` (`:360`) applies the stored beta and scaler.
- Label-shuffle null: per-question flip, maximum over layers, and the failure counter raises above two failures per
  layer (`_permutation_threshold` `:426`).
- Capture: records are grouped by question id in `build_records`, so the contiguity check in `command_capture` holds on the
  real record order. A question's records are appended only at its boundary, so a crash leaves complete questions.
- Atomic writes: `_atomic_write` (`collect37.py:95`) removes its temp file on failure and replaces only on success.
- Generation compatibility: `build_records` reads `(question_id, model, kind)` rows (`collect37.py:407-411`). The 92c2f6e
  generate code writes exactly those keys, with kinds greedy, s1 and s2 (smol: greedy only), so old generation files work
  with the new build.
- Scoring: 37.2, 37.3, 37.4, 37.8, 37.9, 37.11, 37.12, 37.13, 37.14, 37.15 match the amended rules and `PREDICTIONS.md`.
- Planted-signal and noise handling in the gate: the planted 0.5 SD signal continued jointly in both pairs (F3).
