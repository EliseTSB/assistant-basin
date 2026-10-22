# Part 58: tuning the Haiku judge's system prompt against a held-out detection test

Written 2026-10-22 ~09:40 CT, before any call is made. Elise asked (09:13): optimise the Haiku judge prompt so it separates
random directions from "one job" neurons better.

## Why the target is not "real vs random"
Part 56's pilot: controls rated one_job 4 were about 0.9% (18 of ~2,000) and read as single features ("www in URLs", "the word
United"). A random direction is a mixture of units, but its top windows can be dominated by one strong token, so a control
with a clean feature IS one job. Real neurons are also mostly 1 or 2 (917 + 813 of 1,996). A prompt tuned only to push
real above random would learn artifacts (layer, padding, token frequency) rather than one-jobness.

## Objective test instead (detection score; the idea is Bills et al. 2023 / EleutherAI delphi, from memory, not re-verified)
For each unit, split its windows. DESCRIBE set: top[0:12] + random[0:6] + weak[0:3] (21 windows, same ordering as production).
Haiku writes {concept, one_job, kind, note} from the describe set under the prompt variant. HELD-OUT test: positives are
top[12:20] and random[6:10] (12 windows where the unit fires); negatives are weak[3:5] plus 8 top windows from other units
(10 windows). A second, fixed Haiku call sees only the concept and the 22 shuffled windows and scores each 0 to 10 for "does
the unit fire at the marked token". Detection AUC per unit = how well the concept alone separates positives from negatives.
Truth label "one job" := detection AUC >= 0.85.

## Metrics per variant (dev split, 250 real + 250 control; test split disjoint, same sizes, run once at the end)
- M1: ROC AUC of one_job as a predictor of truth (pooled real + control).
- M2: Spearman(one_job, detection AUC).
- M3: share of units rated one_job >= 4 whose detection AUC < 0.8 (false high ratings).
- M4: share of truth units rated <= 2 (missed one-job units).
- M5: units with detection AUC >= 0.85 per 100 (concept quality) and mean tokens/cost.
- Reported only: real vs control mean one_job and their gap.

## Predictions (written before the first call)
- 58a Claude: baseline (current prompt) M1 between 0.60 and 0.75 on dev.
- 58b Claude: controls rated 4 or 5 have a median detection AUC within 0.10 of real neurons rated 4 or 5 (they are genuinely one job).
- 58c Claude: at least 15% of real neurons rated 1 or 2 by the baseline have detection AUC >= 0.85 (the judge under-rates real features).
- 58d Claude: the best variant raises M1 by at least 0.05 on dev, and at least 0.03 on test.
- 58e Claude: a rubric stated as a countable fraction ("how many windows fit the concept") beats the current vague 1-to-5 wording.
- 58f Claude: adding low-score worked examples (current six are all 4 or 5) lowers M3 by at least a third.
- 58g Claude: shortening the prompt to under 500 words does not lose more than 0.02 of M1.

## Stop rule
At most six variants on dev (plus up to two refinements of the leader); one test run; if the best variant's dev gain is under
0.05 or test gain under 0.03, keep the baseline for the full labelling and say so. Spend cap $10 on the credit; the $44
labelling keeps priority. Overfitting guard: no prompt edit after seeing test.

## Prior art
Closest work: automated-interpretability simulation / detection scoring (Bills et al. 2023; Paulo et al. 2024 delphi). Both
score a description, not a rating; ours scores whether the rating tracks the description's held-out detectability.
