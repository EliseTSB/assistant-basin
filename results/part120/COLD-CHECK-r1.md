# Part 120 cold check, round 1 (Haiku 5.5, 2026-10-30)

Cold check of the Part 120 README: no number or verdict is wrong except one sentence about the position count, and no verdict is more generous than its registered text on the numeric clause. The problems are that the headline rests on a dose where the README's own second-order check is failing, that the norms are over a selected, centered token set the README does not describe, and that several claims are stated without their limits.

**Verdicts against the registered text.** All eight verdicts follow the registered rules on recomputation. None is more generous than its numeric clause. 120.3's lead sentence and 120.5's tail clause are not tested separately. 120.5 is the fragile one (finding 6).

**Findings, most important first**

1. **Overstated: the headline is read at dose 0.1, where the README's own second-order check fails.** Line 49 gives q = 0.60069 at 0.1, double the 0.30 level it uses for 0.05, but says only that q "rises above 0.30 by 0.1." The 120.2 verdict and the registered withdrawal rule (PART120-DESIGN.md lines 119-120) check q at 0.05 only, so the headline is registration-consistent. At 0.1 the remainder is 60 percent of S_P. Real c runs 0.280, 0.278, 0.275 at 0.0125, 0.025 and 0.05, drops to 0.224 at 0.1, then rises to 0.356 at 0.2. Control shows the same dip (0.320, 0.318, 0.307, 0.238, 0.344), visible in panel c of results/part120/fig120.png. Title line 1 and line 5 call 0.1 "low dose" without saying this.

2. **Overstated: the norms are not vocabulary norms.** Each row's logits are restricted to T, the sorted union of the top-10 token lists under the base model and under the 0.8 states A, B and P (part119b/run119b.py lines 298-301; part119b/score119b.py line 1446), and centered over T (run119b.py line 236). The NPZ files hold 12 to 40 token IDs per position, median about 23. README line 139 says only "Token support T came from Part 119c's float32 0.8 run," and line 151 defines T without a size. Lines 152-153 call A, B and P "centered" without saying over what. The design's sentence that T covers about 24 tokens, not the vocabulary (PART120-DESIGN.md lines 129-130), was dropped.

3. **Overstated: the self terms rest on 39 pushes, not 312 rows.** Push A is shared by all eight rows of a first unit. At dose 0.1, A values for the same unit and position differ across rows only by a constant shift, which is the per-row centering in finding 2. I checked 3,598 row pairs on shared tokens. So the S_A statistics (u(0.4) at part120/score120.py lines 562-575, 120.5, 120.6, 120.7) carry 39 pushes per position. README line 5 says "312 selected rows," and lines 139-141 never say this. 120.5 has no interval.

4. **Overstated: the control qualifier is missing from the headline.** At 0.1, random directions give c = 0.23767 against 0.22395 for real rows (factor 1.061, and 120.4 holds). The registered score says the headline "also covers the matched random second pushes" (results/part120/SCORE.md line 102; score120.py lines 833-835), and the design says that coverage "is said beside it" (PART120-DESIGN.md lines 124-125). The README's title and line 5 do not say it. The reader of the headline would take it as a result about selected neuron pairs, but c below 1 is about as true of random directions.

5. **Overstated: the path asymmetry enters the headline statistic itself.** Line 141 says the path assignment "can affect S_A against S_B," but c = ‖I‖/√(‖S_A‖‖S_B‖) (score120.py line 438) uses both self terms directly, so the asymmetry is inside the headline. The README does not say so. The heading "120.8 and path choice" (line 131) suggests the path bears on 120.8, but 120.8 compares real and control B on the same path, so A's ownership does not enter it. The direction of the effect on c cannot be determined from these data. At 0.1, control ‖S_B‖/‖S_A‖ is 0.62 and real is 0.91, so the asymmetry is not a constant offset, and push identity and path are confounded, as line 135 says.

6. **Overstated: 120.5 is held on a point estimate with no uncertainty.** The point ratio of self share to pair share is 1.63, inside the band [0.5, 2] (line 21). My unit bootstrap of that ratio, with 2,000 draws and the registered product weights, gives a 95 percent interval of 1.31 to 2.27, and 8 percent of draws fall outside the band. The registered rule is a point rule, so "held" is correct, but the margin should be stated.

7. **Changes a number: line 30 misstates the position count.** "All 156 rows in each class had 2,056 eligible positions" is wrong. 2,056 is the class total summed over 156 rows (score120.py lines 498-500). Positions per row range from 3 to 23, and 2,056/156 is not an integer. NUMBERS.md's "Positions considered 2056" is correctly a class total.

8. **Overstated, unclear first paragraph.** Lines 1 and 5 use c, "real-row," "the registered headline rule," "below 0.8" and "All three gates passed" without definitions. The 0.8 and 1.25 cutoffs appear only in PART120-DESIGN.md lines 119-124 and score120.py lines 824-831, never in the README. A stranger cannot tell from the first paragraph what result would have counted as the opposite.

9. **Minor: undefined or inconsistent terms.** Line 22 (120.6) uses "slot units," "concept units," "Haiku kinds," "readers" and "agree" undefined. Line 23 uses "Part 119's dominance," and Part 92 is never explained. Line 139 says "312 Path 2 rows," but line 150 defines Path 2 as a set of positions. "Matched random directions" (line 139) omits the design's "matched in layer and size" (PART120-DESIGN.md line 64). "Float32 floor" (line 58) is never defined.

10. **Minor: the float32 floor table has no reading.** Lines 58-87 give ratios without saying what the floor is or what would count as a shortfall. Several ratios depart from the square-law value of 4 at higher doses: real ‖S_A‖ is 5.60 from 0.05 to 0.1, and real ‖I‖ is 7.67 from 0.1 to 0.2. The square-law prediction 120.1 tests only ‖I‖ from 0.025 to 0.1, and the README does not say that.

11. **Minor: missing limits.** The design says the part measures how large the terms are on the output, not where in the model they arise (PART120-DESIGN.md lines 131-132). It also says the derivatives are forward differences (lines 69-71). Neither appears in the README. Gate thresholds are not stated (line 5; definition at line 161). Gate 3 reads exactly 0 (results/part120/gate3.json), which is expected because doubling in float32 is exact. The README should say so, since a reader may take a zero error as strong evidence.

12. **Minor: "recomputed" overstates independence.** Line 9 says every table value is recomputed by fig120.py. That script calls the same score120.score_data (fig120.py lines 22, 43-52), so it checks the code against itself. My independent recomputation from the NPZ files reproduces the values, so the numbers stand. The wording should say so.

13. **Minor: 120.6 detail.** Line 22 gives agreed-unit medians without n. The agreed concept group is 8 of 18 concept units, with 10 disputed. It also omits that the agreed direction is reversed, which PREDICTIONS.md line 2640 says.

14. **Minor: "alignment" for tiny values.** Line 91 says the cosines "show the direction alignment" of I with each self term. The values are medians of row cosines (score120.py lines 442-445), not labelled as such, and every value at doses up to 0.1 is below 0.05 in absolute value (lines 95-101).

15. **Minor: placeholder.** Line 173 reads "Cold check: pending." It should go before publishing.

**Checked and not listed.** Every table number matches score120.json and my recomputation from the raw NPZ files, including the dose table, floor ratios, cosines, layer table, u(0.4), the Spearman rho of -0.0864 and the 120.8 ratio of 1.59. The registered prediction text matches PART120-DESIGN.md and PREDICTIONS.md verbatim for all eight. The figure matches the table and uses log axes. The links resolve.

**Not checked.**
- The registered bootstrap draws: I did not run the score119 machinery. My own draws give a product-weighted lower bound of 0.156 against the registered 0.161, with the upper bound and sum-weighted interval matching. No verdict changes.
- Gates 1 and 2 were not recomputed from raw logits. I used the JSON records and the stored gate 2 maximum of 4.2e-5. The 2,964-row count comes from Part 119b's tests and design.
- The Haiku labels were taken from score120.json. The dominance values came from results/part119/tournament.json, and the stored rho reproduced.
- I ran no test suite.

I wrote no files. The checks ran from stdin with bytecode writing disabled, and git status is clean for the Part 120 paths.

Findings by severity: changes a verdict or number: 1; overstated: 7; minor: 7.
