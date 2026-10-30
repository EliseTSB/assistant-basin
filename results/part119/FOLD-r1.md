# Fold record, cold check round 1

F1 — Method and Limits say Gate 2 compares the two batch-size-1 hook paths on all 39 units; recomputation finds 9 batch-64 main singles differ from batch-1 Part 92, limiting any single-answer result and p_A, t_A, Path 2 positions, and R.
F2 — Replaced “interact alike” with the effect-matched logit-interaction-size test, ratio 1.044 and interval 0.961–1.129 inside 0.80–1.25; added the activation caveat from the design.
F3 — Marked 119.10 failed under the literal design wording, stated the scorer's withheld/not-read result and non-null-kappa reason, and updated its README and PREDICTIONS.md cells.
F4 — Set the requested title and first sentence; gave the 24.4% row mean, pooled 23.7%, interaction share 0.471, model, question, and dose; qualified control rates as description only.
F5 — Defined R and gave medians; stated that 119.6's 0.10 and 119.9's 0.05 thresholds were unreachable on the observed 0–0.058 scale and were registration errors.
F6 — Added the 119.5 unit-bootstrap interval, 8-row leave-one-concept-out table with coverage, and matching reader leave-one-out coverage; defined wins as token counts. Raw recomputation finds 506 / 2,000 draws (25.3%) below 60%, slightly above the cold check's rounded 25%.
F7 — Corrected the draft-time header and cited registration commit 0d6d572 in Method and the Note after scoring.
F8 — Documented product-of-draw-count weighting and added sum-weighted intervals beside the registered intervals for rates, differences, residuals, and ratios.
F9 — Stated Part 86's exact keep/broken/changed rule, repeat-only counts, reader coherence by class, and similar pair/control break rates in the broken-answers section and Limits.
F10 — Named the raw-logit scale centered on T, defined interaction share, and added total-variation medians for real pairs and controls.
F11 — Added plain-language Terms for the listed analysis vocabulary; identified DS as DeepSeek V4.1 Flash and named Haiku and Luna.
F12 — Added the design limits on selected kinds, flip-alpha cap and step, contested style tokens, token counts, prompt positions, power, control matching, and Part 118.
F13 — Compared all 139 pilot-128 rows to main answers and found 139 token-identical; stated pilot reuse and the 8/20 triage/by-eye gap; added the requested by-eye paragraph.
F14 — Restored the missing “ a” in the 90-character p92-05 + p92-37 excerpt and labeled example groups as agreed groups.
F15 — Named SciPy 1.18.1 `scipy.stats.mannwhitneyu` with the code's `alternative="greater"` and `alternative="less"` calls and default options.
F16 — Quoted Amendment 1's pilot disclosure and attributed it to Claude.
F17 — Rebuilt the three-panel figure with rates, intervals, thresholds, triage/coherence shares, and the effect-matched ratio; inspected the PNG and confirmed labels do not cross the lines.
F18 — Added the 1,475-row effect mean and exclusion reason, 741 pairs appearing twice, Gate 1, reader settings/errors/retries, seeds, and the meaning of control positions.
F19 — Changed only the requested rule string in `score119.py`, reran the scorer with the venv Python, and verified the JSON changed only at that string, `SCORE.md` did not change, and `tournament.json` stayed identical.
