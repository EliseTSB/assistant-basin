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
