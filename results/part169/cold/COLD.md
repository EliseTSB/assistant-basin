# Cold check: Part 169

## SUMMARY

Every number I recomputed matches the README and score.json, and each verdict follows the design's wording. The problems are in reading.

## FINDINGS (most serious first)

1. **169.8 "Held"** and "including the held point estimate for 169.8." On the 528 rows with two or more plain tokens, loops are none 205, free 205, hit 201. The verdict compares induced rates on 323 rows (hit 54, free 62). Free has no net rise there, so 169.8 cannot read as plain tokens reproducing a rise. Give the 323 denominator and net counts.

2. **"the gap is under half the full 169.4 gap."** Strength-matched gap −0.32 pp against full −0.42 pp: |−0.32|/|−0.42| = 0.75. Correct to "75% of the full gap; the half-gap test is not met."

3. **Own cut concentration.** "103 of 128 (80.5%)" is correct. The top two tenths hold 48% of none loops but 80% of the cut (63% of their loops cut, against 14% in the other eight). Loop counts alone predict about 62 of 103. High rows cut 53.0% against 15.9% elsewhere, so the 169.1 bar rests on high rows.
4. **"Wrong half" rule not applied.** Unpaired high rows show induced free 46.9% (15 of 32) against 17.3% paired. Induced triggers the rule; net does not (free 81, none 80). The README neither applies the rule nor states the outcome.

5. **Minor numbers.** Batch-shape noise is 3.5% (clear) to 4.9% (own), not "about 4%–5%". "90.9% of free-list mass" is over all 899 hit rows, not the 528.

6. **"The same recipe."** The mass-draw algorithm is identical, but Part 169 also excludes every added-vocabulary id (including `</think>`) from every list, own included. Part 168 excluded only special and EOS ids. The full arm was dropped too. The README names only neurons, phrasings and seeds.

7. **Figure.** n+ and none are both blue, k4 and own both orange, so the legend misreads. Hit (899 rows) sits beside none (1,536 rows), and n+ hit 27.6% above n+ none 21.7% reads as a rise. The caption omits the different row set.

8. **Scorer.** `hit_relative_cut` and `hit.net_loop_change` compare hit's rows with all-row none; the README warning is confirmed (same rows: hit 262, none 267). `med()` returns the upper median, negligible here. 169.7's "−1.9%" is a cut, so negative means clear adds loops; the table does not say so.

## HOLDS

- All row counts, per-arm and pooled tables, and intervals.
- 169.1 37.9%; 169.2 37.1 points; 169.3 0.0%; 169.4 −0.42 pp; 169.5 +2.2 pp; 169.6 24 of 768; 169.7 −1.9%; W1 −10.2 pp; W2 283 against 42.7.
- Leave-one-layer 34.1%–42.4%; leave-one-phrasing 35.4%–39.8%; n+ alone 43.7%; k4 alone 32.2%.
- Length and EOS do not explain the own cut (residual −126.5 of −128). For plain50 they explain about 47 of 283.
- plain50 rises under all eight phrasings. In a 20-answer read of its induced loops, 16 are cyclic repetition and 4 are prose repeating a phrase three times; the rule labels both.

## Part 168 rise

May conclude: a free list at Part 169's neurons and phrasings did not raise loops by 5%; its 95% upper bound is +7.0%, so 16.5% lies outside it.

May not conclude: that the 168 rise was spurious (its interval was 6.3%–29.4%); that plain tokens cause it (169.4 failed; 169.8 is induced-only); or that the recipes match (finding 7). The intervals overlap only between 6.3% and 7.0%, and no two-sample test was run. Baselines differ (14.0% against 22.0%).

## COULD NOT CHECK

- No GPU rerun. Part 168 raw answers were not recomputed. PART168-DESIGN.md amendments and Muse's links were not read.

DONE
