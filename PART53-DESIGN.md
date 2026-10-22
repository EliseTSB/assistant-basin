# Part 53: how does the magic unit L22:1610 interact with the wizard unit L19:4826? (registered 2026-10-21 23:31 CT, before any code or run)

Elise, 2026-10-21 23:29 CT: "How does the magic neuron interact with the wizard neuron? Lets test those interactions at
various intensities, both direction." Qwen3-1.7B @ 70d244c, clamp on generated tokens at k times the unit's natural
maximum (Part 49/51 mechanics, per-clamp k). W = L19:4826 (Part 38/49 wizard unit, writes "spell, cast"). M = L22:1610
(Part 49b/51 magic unit, writes spell, magic, enchant). W sits three layers upstream of M. Prior art asked of Muse
(council muse-part53-prior-art #2); folded in before scoring.

## Design
- Grid: W k in {-16, -8, 0, +8, +16} x M k in {-32, -16, 0, +16, +32}; 25 cells; cell (0,0) is no clamp. Per cell on the
  16 wizard prompts (data/part49b-wizard-prompts.json): 2 samples each = 32 answers. 800 generations.
- Everyday corners on the 16 everyday prompts (data/part51a-everyday-prompts.json), 2 samples each = 32 per cell, five
  cells: (0,0), (0,+32), (0,-32), (+16,0), (-16,0). 160 generations. Total 960.
- Recorded per answer, from the same forward passes: the mean activation of W and of M over generated tokens, so each
  unit's own activity under the other's clamp is measured (an unclamped unit only; a clamped unit is fixed by design).
- Measures: magic-word density per 1,000 words (Part 51's stems, prompt words excluded); wizard-word density (Part 38's
  wizard topic terms, prompt words excluded); broken share by the blind reader (Opus 5.5 batch, 20 calibration texts,
  as 49.2). Per-cell means over prompts, bootstrap interval over prompts (10,000 resamples).
- Interaction statistics: additive contrast D(w,m) - D(w,0) - D(0,m) + D(0,0), and multiplicative ratio
  [D(w,m)/D(0,0)] / ([D(w,0)/D(0,0)] x [D(0,m)/D(0,0)]); a cell with D(0,0) near 0 on everyday text is reported as a
  difference only (51a precedent).

## Predictions (Claude)
| id | prediction |
|---|---|
| 53a | W alone: at m = 0, magic density at every W level is within 20% of the no-clamp cell on wizard prompts (Part 38 and 49 found W moves nothing). |
| 53b | M down persists at every W: at m = -16, magic density is between 0.15 and 0.40 of the same-W cell at m = 0, for all five W levels. |
| 53c | M up helps on wizard prompts: at w = 0, m = +32 raises magic density at least 30% above no clamp, interval excluding no change. |
| 53d | W is upstream of M: at m = 0, W at +16 raises M's mean activation on generated tokens at least 20% above no clamp, and W at -16 lowers it at least 20%. |
| 53e | M is not upstream of W: at w = 0, M at +32 or -32 changes W's mean activation by under 10% either way. |
| 53f | The grid is multiplicative: no off-axis cell's multiplicative ratio falls outside 0.7 to 1.4 (additive contrast is the wrong model because M's down push floors near 0.2). |
| 53g | Broken answers are at most 15% in every cell with |k_W| <= 8 and |k_M| <= 16. |

## What this cannot tell apart
Interaction of two clamped values is not interaction of the units' natural firing; 53d/53e read the natural side
(an unclamped unit's activity). A cell where text breaks (51e's R1 at -32 broke 60 of 64) is reported with its broken
share, never as a magic effect.

## Result (scored 2026-10-22 00:21 CT; Arc run 23:50-00:14, 960 answers, 30 cells x 32; broken-answer labels pending, batch msgbatch_01Aw4XHmAgvTav1j2gcE39bg)
Magic-word density per 1,000 words on the 16 wizard prompts (rows W, columns M; k times natural max; the middle column is M unclamped):
| | M -32 | M -16 | M 0 | M +16 | M +32 |
|---|---|---|---|---|---|
| W -16 | 0.45 | 0.00 | 5.02 | 3.98 | 4.93 |
| W -8 | 0.98 | 2.85 | 7.78 | 3.34 | 2.08 |
| W 0 | 2.20 | 2.59 | **10.49** | 8.75 | 10.29 |
| W +8 | 0.00 | 1.18 | 11.55 | 8.61 | 14.41 |
| W +16 | 1.12 | 0.50 | 11.31 | 9.59 | 16.81 |
Wizard-word density follows the same shape (13.6 at W +16 / M +32 against 4.35 unclamped). Mean M activity on generated tokens, M
unclamped: W -16 14.8, -8 21.7, 0 29.8, +8 32.4, +16 37.6 (M's natural max is 5.3, so its in-context level is about 5.6 times that).
Mean W activity, W unclamped: 1.54 at M -32, 1.67, 1.87, 2.02, 1.94 at M +32 (W's natural max is 4.19). Everyday corners: M +32
0.49 per 1,000 (no clamp 0.0); every other everyday corner 0.0.
| id | result |
|---|---|
| 53a | FAILED. W alone moves magic words only downward: W -16 gives 0.48 of no clamp (interval 0.17 to 0.74), W -8 0.74; W +8 and +16 are 1.10 and 1.08 (inside 20%). |
| 53b | FAILED, in the stronger direction. M at -16 leaves 0.00, 0.37, 0.25, 0.10 and 0.04 of the same-W cell for W -16, -8, 0, +8, +16: only two of five sit in 0.15 to 0.40. M down works at every W, and works hardest when W is pushed. |
| 53c | FAILED. M +32 at W 0: 10.29 against 10.49 (ratio 0.98, interval 0.45 to 1.81). But M +32 at W +16 gives 16.81 against 11.31 for W +16 alone (1.49). |
| 53d | HELD. W +16 raises M's activity 26% (interval 1.07 to 1.45) and W -16 lowers it 50% (0.38 to 0.61). |
| 53e | FAILED narrowly. M +32 changes W's activity by +3.8%, M -32 by -17.6% (ratio 0.82, interval 0.65 to 1.03, includes no change). |
| 53f | FAILED. Only 5 of 16 off-axis cells have a multiplicative ratio in 0.7 to 1.4; intervals are wide at 32 answers per cell. |
| 53g | HELD. Blind reader (Opus 5.5 batch, 20 of 20 calibration right): broken answers are 0 to 3% in all 25 wizard cells (at most 1 of 32), including the corners, so the grid's changes are on fluent text. |
Reading (exploratory): the wizard unit sits upstream of the magic unit in the activity sense (53d, and its push down halves
magic words), but pushing it up adds nothing by itself. The magic unit's up push helps only when the wizard unit is also
up (W +16, M +32: 16.8; additive prediction 11.1), and its down push removes the magic vocabulary at every W. Both corner
cells I read (W -16 / M -32 and W +16 / M +32, five answers each) are fluent. 32 answers per cell: single cells are noisy.

## Prior art from Muse (council muse-part53-prior-art #4, 2026-10-22 01:06 CT; filed 2026-10-22 01:09 CT, AFTER the run scored, so context and not a prediction input)
Report: Muse's ~/workspace/research_notes/part53-prior-art-20261022/report.md. Verdict: PARTIAL, no published neuron-level grid with both interaction contrasts.
- van der Weij, Poesio and Schoots (arXiv:2403.05767, section 4.2): summing steering vectors into one injection largely fails, injecting at different layers works with small interaction effects; direction-level, no dose grid, no additive-versus-multiplicative statistics, no upstream-to-downstream coupling.
- Oozeer et al. (arXiv:2505.24535, K-Steering): avoids the additivity assumption through a classifier gradient; shows the field treats additivity as open; nothing neuron-level.
- Rimsky et al. (arXiv:2312.06681, CAA): additive benefit on top of a system prompt; additivity assumed.
Our result sits beside these: two named neurons at three layers' distance, with one direction of coupling in the activity (53d) and a synergy only when both push up.

## Cold review of the Part 53 build (DeepSeek V4.1 Flash, round 1, SHIP-WITH-FIXES, reviews/cheap-part53-build-ds-r1.md in Journal), read after scoring
None of its seven findings changes a scored row; they change how two tables read:
- F1 (HIGH, reporting): the "activity" recorded for a CLAMPED unit is the clamp level (the +-85 and +-170 of M and +-33.5 and +-67 of W in the activity grids above are the clamp, not a measurement). Rows 53d and 53e use only the UNCLAMPED unit's cells (M with W clamped; W with M clamped), which is what they were written for.
- F2: unclamped M activity averages 29.8 against a natural maximum of 5.3, i.e. a clamp of +32 times the natural maximum (170) is about 5.7 times the in-context mean, not "32 times". The dose scale is in natural-max units throughout, as in Parts 49 to 51.
- F4 and F5: partial or unmatched blind labels would have been reported as a complete rate; the final 53g run matched all 960 labels (merge printed 960).
- F3, F6, F7 (recorder reuse, a test gap for the token position, 0.0 ratio versus null at a zero density): noted; the scored run used one recorder per run and fresh rows.
