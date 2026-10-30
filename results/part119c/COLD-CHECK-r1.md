# Part 119c cold check, round 1 (Haiku 5.5 subagent, 2026-10-30)

**Findings, most important first**

1. **The shrinkage claim holds only for float32.** README line 1 says "the interaction share shrinks toward zero as dose shrinks." In bf16, the dtype Part 119b used, the real share falls to 0.0397 at 0.1 and then rises to 0.0734 at 0.05. Controls go 0.0422 to 0.0823. Float32 real falls 0.0201 to 0.00455. The bf16 dip-and-rise is visible in /home/elisepc/green/worktrees/ab-part67/results/part119c/fig119c.png, but the README never describes it. It should say the shrinkage is float32-only and that bf16 has a floor-like 0.04 to 0.08 at small dose. Consequence: 119c.2 would fail for bf16 at 0.05 (0.0734 > 0.05), so the verdict depends on treating float32 as the reference. The README does not explain that choice.

2. **Part 119b's gate 3 is not addressed.** The README says "Nothing here changes a verdict of Part 119b" and line 1 says "Part 119b's low-dose share is not a rounding floor." Part 119b's gate 3 value is median_row_share_common_resolved_all_doses = 0.1776 at 0.2, real rows, against a 0.15 limit (/home/elisepc/green/worktrees/ab-part67/results/part119b/score119b.json, path2_summary/by_dose/0.2/real; gates/gate3 share_clause_passed is false). The float32 subset gives 0.1534 at 0.2, still above 0.15. Rule (2) requires float32 to sit more than 0.04 below bf16 at 0.2. The observed difference is -0.0082, so rounding does not explain the gate-3 miss. The README should say so. Part 119b ran only doses 0.2, 0.4, 0.6 and 0.8 (11,856 lines = 2,964 keys × 4), so 0.05 and 0.1 are new, and "low dose" in 119b means 0.2.

3. **The small-denominator question is answered with a measure the README omits.** Part 119b's "resolved" flag requires nA, n2 and nPair all at least 1.0 (/home/elisepc/green/worktrees/ab-part67/part119b/run119b.py lines 260–262; score119b.py lines 107–114). Neither the README nor NUMBERS mentions it. For float32 real at 0.05, only 77 of 2,056 positions qualify (3.7%), in 37 of 156 rows. The resolved-only median row share is 0.00579 for real and 0.00349 for control, a factor of 1.66. That fails the 119c.4 bar of 1.25. The passing factor of 1.013 depends on including unresolved positions. The README's Limits paragraph reports only the nPair-below-1 share (57.1%). Part 119b reported resolved medians too (/home/elisepc/green/worktrees/ab-part67/results/part119b/SCORE.md, per-dose table).

4. **"At 0.2 it is already measurable" overstates a real-specific effect.** Float32 control at 0.2 is 0.1391 against real 0.1534, so control is 91% of real. Real falls below control at 0.05, 0.1 and 0.4 in both dtypes (float32: 0.00455 vs 0.00461, 0.02009 vs 0.02081, 0.2949 vs 0.3224). Real exceeds control only at 0.2 (+10% float32, +14% bf16) and 0.8 (+25%). The README says "The symmetric larger-to-smaller factors at every dose are," which hides the direction.

5. **The 312-row subset does not reproduce Part 119b's share, and the README does not say so.** Gate 1 says the bf16 rows "reproduce Part 119b at 0.2, 0.4 and 0.8, with median relative nPair difference 0." That holds for nPair, which I checked at each dose. The share is different: the subset gives 0.1616 at 0.2, while the same statistic over all 1,482 real Part 119b rows gives 0.1762. The gap is 0.015 and should be reported.

6. **Dtype claims are overstated.**
- Terms says "the model parameter and activation dtype recorded by each run." Only the requested name and the first parameter's dtype are recorded (/home/elisepc/green/worktrees/ab-part67/part119c/run119c.py lines 90–99 and 125–126). Activation dtype is not recorded.
- Gate 2 "identify[s] bf16 and float32 loaded parameters" is a check on the first parameter only.
- Gate 2 checks only dose 0.8. The median relative nPair gap between dtypes is 4.6% at 0.05, 2.4% at 0.1, 1.2% at 0.2, 0.70% at 0.4 and 0.35% at 0.8. None of the small-dose values is reported.
- The bf16 excess sits in nI: bf16/float32 median nI is 4.1× at 0.05 (0.2346 vs 0.0574), 1.4× at 0.1 and 1.01× at 0.2. The absolute excess is about 0.18 at 0.05 for both real and control.
- The loader changed between the two runs (/home/elisepc/green/worktrees/ab-part67/part119c/BUILD-NOTES.md line 45). The README does not discuss this beyond gate 2.

7. **Token-support confound is not disclosed.** Method says "The token support comes from each dtype's own 0.8 passes." In /home/elisepc/green/worktrees/ab-part67/part119b/run119b.py lines 229–240, nI and nPair are norms over the token set T. At 0.8, the T sets differ between dtypes at 1,675 of 4,112 positions (median 0 tokens differ, maximum 6). Whether this moves the dtype gap cannot be checked from the stored data. It needs a rerun with one shared T.

8. **Row count.** The README and NUMBERS say 312 rows. Each JSONL file has 1,560 lines: 312 distinct keys times 5 doses. NUMBERS' source key for "Selected rows" is "row count," which returns 1,560 per file.

9. **The headline is not quoted as registered.** The registered sentence (PART119C-DESIGN.md reading rule 1, identical in SCORE.md) is: "Part 119b's low-dose share is not a rounding floor; the interaction shrinks toward zero with dose, and at 0.2 it is already measurable where answers do not break." The README's line 1 uses a colon, "as dose shrinks" and added numbers. The body uses a comma. The registered sentence does not appear in the README verbatim. The rule's output does match SCORE.md.

10. **Undefined terms and untested wording.** "Where answers do not break" rests on Part 119b's real-pair triage (33 of 741 broken, 4.5%, at 0.4; /home/elisepc/green/worktrees/ab-part67/results/part119b/README.md line 3). No answers were generated in 119c. Terms a stranger cannot follow: position, push, unit, nA, n2, T or "token support," resolved (absent), product- and sum-weighted interval, unit bootstrap, gate, larger-to-smaller factor, log-log slope, matched random direction, RAM guard, A770. The README still ends with "Cold check: pending."

11. **Scope missing from line 1.** Missing: 312 of 2,964 rows, 39 units, the fixed 0.8 path, no answers, one device, one model, one question. Explained but not tested: float32 rounding (the design's do-nothing hook is not repeated in float32), batch-shape dependence (the design cites Part 119 Amendment 2 and does not test it), and the loader change.

12. **Skeptic checks not reported.** The pooled ratio sum(nI²)/sum(nPair²) for float32 real gives 0.0046, 0.0377 and 0.2296 at 0.05, 0.1 and 0.2. The registered values are 0.0045, 0.0201 and 0.1534. The direction holds, but the levels move by 1.5× to 1.9×. No interval is reported for the 0.05 or 0.1 shares. Local slopes are 2.14 (0.05 to 0.1) and 2.93 (0.1 to 0.2). The README gives only the endpoint slope, 2.54.

13. **Minor.** NUMBERS.md says every statistic is recomputed by fig119c.py. Gates and predictions actually come from score119c.py, which fig119c.py calls (line 18).

**Verified**
- Row-median shares for all 20 dtype/dose/class cells, to six digits. Examples: float32 real 0.004547, 0.020094, 0.153414; bf16 real 0.073401, 0.039749, 0.161635.
- Median nI and nPair for all 20 cells. Examples: float32 real at 0.2, nI 1.77989 and nPair 5.0188; bf16 real at 0.05, nPair 0.94870.
- Share of positions with nPair below 1 at 0.05 and 0.1, all four cells.
- Endpoint slope 2.53823 and dose-growth ratio 1.90873.
- Real/control factors: float32 1.01341, 1.03577, 1.10257; bf16 1.12115, 1.06251, 1.14307.
- Gate 2 median relative nPair difference 0.00350857 over 312 rows. Gate 1 is zero at 0.2, 0.4 and 0.8.
- 119c.1 bootstrap: product-weighted [-0.021297, 0.012509] and sum-weighted [-0.014343, 0.009603]. I reproduced both exactly with the standard library, seed 11902, 2,000 draws, in both unit orders.
- Verdicts: 119c.1, 119c.2, 119c.3 (2.538 ≥ 1.5) and 119c.4 (maximum factor 1.103 ≤ 1.25) all hold. The headline rule's output matches SCORE.md.
- All four prediction texts match PART119C-DESIGN.md after removing backticks in 119c.3. Confidences match (medium, medium, low, low). The README says "held" where the code says "holds," which is wording only.
- 39 units, 156 rows per class, 2,056 positions per cell, 2,964 keys in Part 119b, seed 11906 (/home/elisepc/green/worktrees/ab-part67/part119c/rows119c.py line 20).
- All linked files exist. The revision string appears in /home/elisepc/green/worktrees/ab-part67/part119b/BUILD-NOTES.md.
- The figure matches the table values and uses log axes (fig119c.py lines 218–219).

**Not checked:** the batch-8 and A770 claims beyond the design text, the loader change's effect, the T-set effect (needs a rerun), the float32 noise floor (not measurable from the stored data), and the figure's caption beyond its labels.
