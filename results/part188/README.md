# Part 188: no single unit in EmbeddingGemma 2 separates real animal sounds from human imitations of them reliably; a probe on the output embedding does

Question (registered in `PART188-DESIGN.md`, 17:39 CT 2026-11-04): does any single unit tell real animal sounds from
people imitating them, on animals it was not selected on, better than the model's own verdict? Data: 3,138 clips
(VocalImitationSet and ESC-50), 19 animal groups split into folds A and B, Chicken held out. A unit is chosen on one fold
by clip-level AUC and tested on the other.

**Answer.** Not reliably. The two directions picked different units (A to B: L13 N773; B to A: L5 N1770) with held-out
AUC 0.753 and 0.897, but the first is weak and both also respond to human voices (below). Gate 2 and the registered
headline rule: the headline reading is **not permitted** (predictions 188.2 and 188.3 failed), whatever Gate 2 says.
A logistic probe on the 768-d output embedding reaches 0.918 and 0.965 (188.6 held), and so does a probe on the top 32
units (0.905, 0.957).

## Gates

- Gate 1 (parity with the stored recording): passed.
- Gate 3 (the recording sees hens: unit 31846, hen and rooster against 28 other classes): AUC 0.982, passed.
- Gate 2 (at least 90% of each set recorded): every set recorded in full (3,138 of 3,138 planned clips in total). The run's own verdict says
  "failed" only because the code added a phantom set P with 0 planned clips (introduced in my review fold d479df9;
  fixed as Amendment 2, public 069f566, before any result value was read). With P removed Gate 2 passes. Re-applying
  `_headline_rule` with all three gates passed still withholds the headline, for the two failed predictions.

## Predictions

| # | Registered | A to B | B to A | Verdict |
| --- | --- | ---: | ---: | --- |
| 188.1 | the model's own verdict (class-informed) < 0.75 in both folds | 0.631 | 0.808 | failed |
| 188.2 | selected unit >= 0.85 in both folds | 0.753 | 0.897 | failed |
| 188.3 | unit beats the verdict by >= 0.10 in both folds | +0.123 | +0.089 | failed |
| 188.4 | unit, real against ESC-50 human sounds, >= 0.80 | 0.281 | 0.677 | failed |
| 188.5 | known L17 N1574 on non-chicken < 0.70 | 0.647 | | held |
| 188.6 | output-embedding probe >= 0.90 in both folds | 0.918 | 0.965 | held |
| 188.7 | top-32 probe within 0.05 of the selected unit | gap 0.152 | gap 0.061 | failed |
| 188.8 | both selected units >= 0.80 on chicken (held out) | 0.983 | 0.913 | held |
| 188.9 | sequence length alone < 0.70 | 0.632 | 0.634 | held |

## What the numbers say

1. **The units may partly respond to human-made sound (not established).** Real animals against ESC-50 human sounds:
   0.281 (A to B, the unit puts humans on the "real" side) and 0.677 (B to A). Prediction 188.4 failed, so the design's
   "reads human-made sound" reading is not triggered; the two units disagree in direction. The vocal-only subset gives
   0.340 and 0.606, a little closer to 0.5 for A to B and a little lower for B to A.
2. **The model's own verdict is not near chance here.** It is 0.631 (A to B) and 0.808 (B to A), class-informed. The
   earlier chicken study reported the verdict near chance on chickens; that was not rechecked in this part.
3. **The output embedding carries the signal.** Probes on it reach 0.92 and 0.97, so the information is present. A probe or
   an edit using several units has something to work with; this part did not test one, and does not support an edit on
   one unit.
4. **Chicken is the special case.** Both selected units separate real hens from chicken imitations (0.983, 0.913 on
   the held-out Chicken group of 138 clips; group-clustered 95% intervals 0.94 to 1.00 and 0.83 to 0.97). Why was not tested here.
5. Length is not the explanation (188.9): sequence length alone gives 0.63.

## Caveats

- Group-clustered intervals are wide (A to B unit 0.64 to 0.88). Folds are 9 and 10 animal groups, so a fold's unit is
  chosen on a handful of animals.
- ESC-50 human sounds (188.4) are non-speech human sounds (laughing, coughing, clapping and similar), not imitations,
  so 0.28 says the unit responds to human-made sound, not that it fails on imitations.

Files: `results.json`, `summary.json`, `unit_auc.npz`, `top_units.npz`, `emb188.npz` (from the cloud branch
the cloud session's `part188-r2` branch, 0388f8e); design `PART188-DESIGN.md` with Amendments 1 and 2.

Cold check: DeepSeek V4.1 Flash worker (no project context), 2026-11-04 20:15 CT, 15 findings: every number matched; four overclaims softened (the "reads human-made sound" sentence, the cross-study "does not generalise", "is the route", "no better"), the Chicken intervals added, the Gate 2 wording tightened. Report in `results/part188/cold/COLD.md`, commit 1904165.
