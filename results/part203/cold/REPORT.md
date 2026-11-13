# Cold check: Part 203 write-up (`results/part203/README.md`)

## SUMMARY

The README is arithmetically sound: I recomputed every number in it from `results.json` with my own script
(`results/part203/cold/recompute.py`, independent of `score203.py`) and every one matches to the stated precision, and
all eleven verdicts are right under the design's wording. Both scripts run clean and agree (scores written to
`results/part203/cold/`, figure to `results/part203/cold/fig.png`), every linked path exists, and the reading rules are
kept. The problems are in the prose, not the numbers. The title states "training on three actors instead of 72 or 73
costs both about 0.21" as a finding, but the three-actor ablation is a descriptive add-on that the design registers no
prediction about, and the README never marks it as unregistered (F1). The same unregistered number then carries the
section's headline ("That is the largest single factor measured here", F2), and the "Which layer?" section's claim that
the inner layer score "falls with depth" is not monotone and is stated as a trend where the data show a drop from layer
0 to layer 1 and then a plateau (F4). Two smaller wording issues (F3, F5). Nothing in the README contradicts the data.

## FINDINGS

1. **Title states an unregistered result as a finding (most serious).** Line 1: "training on three actors instead of 72
   or 73 costs both about 0.21". The number is correct (V 0.51998 − 0.31115 = 0.2088; E 0.64497 − 0.43654 = 0.2084;
   `three_actor_ablation` in `results.json`) and 72/73 is right (fold 0 holds out 19 actors, folds 1–4 hold out 18;
   `gate1.folds`). But the design's registered predictions (203.1–203.7, 203.W1–W4) say nothing about the three-actor
   ablation; 203.6/203.7 are about *V's drop* and *V vs E's drop*, not about the size of the cost. The README marks its
   other unregistered readings explicitly ("That reading is descriptive and was not registered", line 52; "These inner
   values ... were not registered", lines 90–91), but not this one, which is in the title. Fix: mark the ablation as
   descriptive/unregistered, as the other two are.

2. **Line 75: "That is the largest single factor measured here" is an unregistered comparison stated as a finding.**
   The sentence answers "Is it the number of training speakers?" with a ranking of factors. The factors are measured on
   different scales and different estimands (the speaker drop is a within-fold ablation on the same held-out actors;
   the V/D gap, the C effect and the E−D gap are cross-readout probe differences; there is no measurement of the data
   or of the vector *per se*). Nothing in the design registers this ranking, and the README does not say it is
   descriptive. The rest of the paragraph is careful ("this part cannot say how much of Part 196's gap the speaker count
   explains there"), which makes the opening sentence stand out.

3. **Line 83: "the final vector and the early layer stay about 0.12 to 0.13 apart at either count" — the
   all-training-actors figure is not 0.12–0.13.** E − V at all training actors is 0.64497 − 0.51998 = **0.1250**
   (`paired_differences.E_p196_minus_V_grid.mean_difference` = 0.12499), which is at the top edge of "0.12 to 0.13".
   But the README's own table uses the E−D gap (0.1697) for the layer contrast, and this sentence switches to E−V
   without saying so. The design's reading rule says "E against V is not read as a layer difference" — the sentence
   does not call it a layer difference, but it is the only place the write-up reads E against V, and the reader is
   given 0.12–0.13 where the registered layer contrast is 0.170. The three-actor figure (0.4365 − 0.3112 = 0.1254) is
   right. Fix: say which pair (E−V) and why, or use E−D at both counts (E−D at three actors is not in `results.json`).

4. **Lines 89–90: "the inner balanced accuracy falls with depth, from 0.644 at layer 0 to 0.498 at layer 23" reads a
   trend the data do not show.** The mean over folds is 0.6444 at layer 0 and 0.4977 at layer 23 (correct), but the
   curve is not monotone: in every fold layer 1 is *below* layer 0 except fold 3 (e.g. fold 0: 0.6364 → 0.6324; fold 2:
   0.6532 → 0.6394), and the values then plateau (fold 0 layers 20–23: 0.5248, 0.5010, 0.5042, 0.5068). "Falls with
   depth" is the kind of shape claim the paper asks the README to avoid stating unregistered. The README does say these
   values were not registered, so this is a wording issue: "is lower at layer 23 than at layer 0" would be exact.

5. **Line 62: "at C = 1 they read 0.4753 and 0.4753" is true but hides that V and D differ per fold, and the README does
   not say so.** `V_p196` per fold 0.4561 / 0.5315 / 0.4685 / 0.4519 / 0.4685; `D_p196` 0.4561 / 0.5333 / 0.4685 /
   0.4537 / 0.4648 (differences 0, −0.0019, 0, −0.0019, +0.0037). The means agree to four decimals only because the
   per-fold differences cancel. Not an error, but the sentence "V and D ... return the same samples ... differ by
   0.0004" would be stronger if it said "on average; per fold they differ by at most 0.004". Low priority.

6. **Line 113: "Cold check: pending."** This is the line this cold check fills; it is left in the README as published
   text. Whoever applies the fixes should replace it with the outcome, not leave it.

## CHECKED

- **Every number in the README, recomputed from `results.json`** (`recompute.py`, no code from `score203.py`):
  - V 0.5200 [0.5022, 0.5372], D 0.5196 [0.5019, 0.5369], E 0.6450 [0.6245, 0.6652], H 0.4192 — all match lines 12–15,
    30–35, 48, 54.
  - V − D = 0.00035 ("0.0004", line 31, 37): match. V chosen C = 0.01 on all five folds for both V and D (line 16, 32,
    67): match.
  - `V_grid_minus_V_p196` 0.04468, interval [0.02697, 0.06314] (line 32 "+0.0447 (0.0270 to 0.0631)"): match.
  - `E_p196_minus_D_p196` 0.16967, interval [0.14994, 0.18891] (line 33 "+0.1697 (0.1499 to 0.1889)"): match.
  - V drop 0.20883, E drop 0.20843, difference 0.00040 (lines 14–15, 36, 77): match. E three-actor mean 0.4365
    (line 36 "0.6450 to 0.4365"): match.
  - `V_grid_minus_H_p196` 0.10080, interval [0.07984, 0.12127] (line 40 "0.1008 (0.0798 to 0.1213)"): match.
  - D_grid − D_p196 = 0.04433 ("+0.044 for D", line 69): match.
  - AUCs V 0.8159, D 0.8159, E 0.8894, H 0.7148 (line 49–50 "0.816 ... 0.816, 0.889 ... 0.715"): match.
  - Per-fold V 0.5351 / 0.5315 / 0.5296 / 0.4667 / 0.5370 (line 50 "0.535, 0.531, 0.530, 0.467, 0.537"): match. Fold 3
    E 0.5889 ("0.589") and H 0.3981 ("0.398"): match; fold 3 is the lowest for V, D, E and H (per-fold values checked).
  - Per-emotion recall, V 0.387 (FEA) to 0.696 (NEU); E 0.546 (SAD) to 0.796 (NEU); E above V on all six emotions
    (line 55–56): match.
  - Layer choice: layer 0 on folds 0, 1, 2, 4 and layer 1 on fold 3; fold 3 inner layer 1 = 0.6582, layer 0 = 0.6561
    (line 88 "0.658 and 0.656"): match. Inner mean layer 0 = 0.6444, layer 23 = 0.4977 (line 89): match.
  - Gates: gate1 passed, 2730 of 2730 recorded ok, `problems` empty; gate2 passed (lines 43–44): match.
  - "Three times chance" (line 48): 0.520 / 0.1667 = 3.12: fair rounding.
- **Verdicts**, applied by hand from PART203-DESIGN.md's table: 203.1 held, 203.2 held, 203.3 failed (second clause: C
  = 0.01 on 5/5 folds), 203.4 held, 203.5 held, 203.6 held, 203.7 failed, W1 failed, W2 held, W3 failed, W4 failed — all
  agree with the README and with `score203.py`'s output. Weaver's rows are scored by forecast with falsifiers reported
  separately, as the design says; W4's "forecast failed, falsifier not reached" is the design's own phrasing for a value
  between forecast and falsifier.
- **Verbatim wording**: I compared each table row's prediction text and confidence against the design's table
  (203.1–203.7 at 70/60/55/60/65/65/50%, W1–W4 at 0.55/0.50/0.30/0.40). The prediction clauses are copied verbatim. The
  README drops each row's *rationale* sentences (e.g. "Emotion in CREMA-D is acted and loud ..."); the README says
  "wording and confidence are copied", which is true for the prediction clause, and the rationales are not claimed to be
  copied. PREDICTIONS.md's "Part 203" table carries the same wording and confidences with empty Result/Verdict cells (not
  yet filled; outside this README's scope, but worth noting before publication).
- **Reading rules**: no balanced-accuracy-vs-AUC comparison (line 49 says explicitly they are not compared); the V/D gap
  is read only with both lower bounds above 1/6 (0.5022, 0.5019) and the README says so; Venn's 56.5% is labelled context
  only and never compared numerically; the layer contrast uses E_p196 vs D_p196, not E vs V — except finding 3.
- **Earlier parts / Venn**: Part 196's 25.8% D, 62.7% E, four speakers, seven styles, one speaker held out per fold all
  match `results/part196/README.md` (lines 1, 17–21) and the design; "Part 196 trained on three speakers per fold" matches
  Part 196's "three training speakers" (line 13). Venn: 56.5%, 1,716 clips, C = 100, held-out actors match
  `PART203-DESIGN.md`'s prior-art section. The README does not claim the two setups are matched.
- **Scripts**: `part203/score203.py --out results/part203/cold` exits 0 and reproduces `score.md`/`score.json` byte for
  byte; `part203/fig203.py --out results/part203/cold/fig.png` exits 0 and writes a 134,335-byte PNG, the same size as
  `results/part203/fig203.png`, so the figure is reproducible. The figure's right panel shows the three-actor ablation,
  which the caption describes correctly; the panel title "Few training speakers cost V and E alike" is supported by the
  numbers (0.2088 vs 0.2084).
- **Links**: all 15 relative links in the README resolve to existing files in the worktree.
- **"How the run went"**: the third run's log (`results/part203-analyze3.log`) shows start 06:42:28, `status: complete`,
  end 06:46:39 — 4 min 11 s, matching line 110. Amendment 1's `problems.reasons` key exists in `analyze203.py`
  (lines 435–438) and is absent from the passing `gate1` (correctly, since no record failed). The first run's file
  holds exactly the two named clips (`1070_TSI_ANG_XX.wav`, `1073_IEO_NEU_XX.wav`), matching line 105. I could not verify
  the second run's details (killed at 2h35m, 11 cores, load 35, the 0.92 s profile) — no log or artifact for them exists
  in the repository; they are unverifiable from the files, not contradicted.
- **Recorder Gate 1 `v_same_samples`**: the recorder writes and checks it per clip (`part203/record203.py:174, 276`), so
  line 61's reference to it is accurate.

## Note on the scorer (outside the README, for the owner)

`part203/score203.py:79` scores W1 with `w1 > 0.03`. The design says W1's forecast is "more than 0.03"; the value is
0.00035, so the verdict is unaffected. More generally, `score203.py` scores W1 with the signed V − D while its falsifier
check uses `abs(w1) <= 0.03`; a large *negative* V − D would be scored "failed" on the forecast (right) but "falsifier not
reached" (arguably wrong — D is then more than 0.03 from V, which is the falsifier's substance). Not triggered here. The
README itself is not wrong on this.
