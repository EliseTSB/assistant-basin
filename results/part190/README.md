# Part 190: a rank-1 edit of EmbeddingGemma 2's output embedding brings its own real-against-imitation verdict on held-out animals close to the probe, and costs some imitations their nearest real-animal match

Question (registered as v2 in `PART190-DESIGN.md`, public main 755f745, 2026-11-07 23:20 CT, before any statistic of the
real arrays): can a low-rank linear edit of EmbeddingGemma 2's output embedding, fitted on one fold of animals, make the
model's own prompt verdict separate real animal sounds from human imitations on animals it was not fitted on, while
leaving the rest of the embedding geometry nearly unchanged? Data: Part 188's 3,138 committed embeddings (VocalImitationSet
and ESC-50), folds A and B (10 and 9 animal groups), Chicken held out. CPU only, no model run; 2,298 edit evaluations in
637 s.

**Answer.** Yes for the verdict, with a cost. The rank-1 edit raised the held-out class-informed verdict AUC from 0.630
to 0.879 (A to B) and from 0.808 to 0.970 (B to A), against Part 188's output-embedding probe at 0.918 and 0.965. The
paired group-clustered 95% intervals of the rises are 0.138 to 0.354 and 0.101 to 0.270. The class-free verdict ("an
animal" prompts) rose too, and held-out Chicken went from 0.611 to 0.933 and 0.985. ESC-50 class neighbourhoods and
animal identity barely moved. **But the edit partly moves imitations toward human sounds in the registered sense**
(190.10 failed: +4.1 and +12.4 points more held-out imitations whose nearest real recording is a human-made sound), the
failure the earlier chicken edit showed, and in B to A it changes the nearest neighbour of 14.8% of off-target clips
(190.7 failed). By the registered reading rules this is reported as an edit that partly moves imitations toward human
sounds, whatever the verdict gain. **What that measure hides, found by the cold check after the run (exploratory):** on
average imitations moved slightly *away* from human clips (-0.14 and -0.27 points of cosine to the nearest one), and
every imitation that crossed (13 and 37) already had a human clip as its second-nearest real recording. The edit pushed
those imitations away from their nearest real-animal recording harder than from the human clip, so the human clip
overtook. The failure is imitations losing their nearest real-animal match, not imitations getting closer to human
sounds (`cold/probe_c6.py`, `cold/probe_c6b.py` and their outputs).

What the edit is: the fit-fold probe's reading written along the verdict's own prompt direction. A held 190.1 means the
probe's knowledge transferred into the verdict on held-out animals; it does not mean the model learned to tell
imitations apart.

## Gates

- Gate 1 (data: 3,138 rows, ids in manifest order, unit-norm embeddings, fold counts): passed.
- Gate 2 (identity: the alpha = 0 edit returns the input; the unedited verdict and the rank-1 probe reproduce Part 188's
  numbers within 0.005): passed.
- Gate 3 (negative control: shuffled-label edits' median held-out change at most +0.05 and 95th percentile at most +0.10,
  every rank, both directions): passed. Its self-test was shown on this data: fed the main edit's own changes, the rule
  fails, so it can fail.
- Gate 4 (ESC-50 10-NN class accuracy at alpha = 0 at least 0.30): passed.

## Predictions

Held-out numbers at the operating dose chosen on the fit fold (alpha 2 for A to B, 4 for B to A), point estimates except
190.13.

| # | Registered | A to B | B to A | Verdict |
| --- | --- | ---: | ---: | --- |
| 190.1 | rank-1 verdict rise >= +0.10 in both | +0.249 | +0.162 | held |
| 190.2 | rank-1 verdict AUC >= 0.85 in both | 0.879 | 0.970 | held |
| 190.3 | rank 8 adds < +0.05 over rank 1 | +0.007 | +0.003 | held |
| 190.4 | shuffled-label median change <= +0.02 | -0.048 | -0.189 | held |
| 190.5 | displacement-matched random median change <= +0.02 | -0.069 | -0.258 | held |
| 190.6 | ESC-50 10-NN accuracy change < 2.0 points | +0.26 | -0.33 | held |
| 190.7 | off-target first-neighbour change < 10% | 6.7% | 14.8% | failed |
| 190.8 | held-out Chicken verdict rise >= +0.10 | +0.322 | +0.375 | held |
| 190.9 | animal identity drop < 2.0 points | +0.13 | -0.39 | held |
| 190.10 | imitations nearest a human sound rise < 5.0 points | +4.12 | +12.41 | failed |
| 190.11 | text-free neighbour AUC (K) rise >= +0.03 | +0.015 | +0.008 | failed |
| 190.12 | operating dose within 0.03 of the held-out ladder maximum | 0.039 | 0.000 | failed |
| 190.13 | clustered 95% interval of the rank-1 rise above zero | low 0.138 | low 0.101 | held |
| 190.14 | class-free verdict rise >= +0.05 | +0.167 | +0.146 | held |

The headline rule (`headline()` in `part190/analyze190.py`) permits the reading: all gates passed, 190.1, 190.4, 190.5 and
190.13 held, the main rank-1 change beats both controls' 95th percentiles in both directions, and the two cost
predictions it checks (190.6, 190.9) held. The 190.10 reading rule applies on top of it (above).

## What the numbers say

1. **The verdict gain is specific to the concept.** Edits built the same way from shuffled labels, and random read
   directions moved the same distance, both made the held-out verdict worse on median (-0.05 and -0.19 shuffled, -0.07 and
   -0.26 random). The 95th percentiles of the controls stayed below the main change in both directions.
2. **Rank 1 is enough.** Ranks 2, 4 and 8 add at most 0.007 (A to B: 0.879, 0.881, 0.882, 0.886; B to A: 0.970, 0.970,
   0.971, 0.973).
3. **The cost grows with the dose, and a smaller dose gets half the gain or more for much less** (exploratory, the line
   the design always reports; the coarse ladder means the qualifying dose gets 51% of the operating rise in A to B and
   73% in B to A). In A to B, alpha 0.5 already gives a rise of +0.128 with imitations-toward-humans +0.75 points
   and 1.2% first-neighbour change, against +0.249, +4.12 points and 6.7% at the operating dose 2. In B to A, alpha 1 gives
   +0.119 with +1.88 points and 2.8%, against +0.162, +12.41 points and 14.8% at dose 4. At dose 16 the drift reaches
   +16.9 and +48.9 points. The dose rule picked maximum separation, not separation per cost, as the cold read warned
   before the run.
4. **The text-free readout barely moved (190.11 failed).** Nearest-neighbour real-against-imitation AUC was already 0.934
   and 0.974 before the edit and rose by 0.015 and 0.008. The edit changes what the verdict reads, not how the clips
   cluster among themselves, which fits an edit that writes along the verdict's direction.
5. **The dose chosen on the fit fold transferred in B to A, not quite in A to B (190.12 failed by 0.009).** In A to B the
   held-out verdict kept rising past the chosen dose (0.879 at 2, 0.905 at 4, 0.915 at 8), while the fit fold had
   saturated (0.995 at dose 2, 1.000 from dose 4), so the dose rule stopped at a ceiling the held-out curve had not reached.

## Limits

- An edit of the output embedding, not of the weights. It is affine in normalized coordinates with a nonzero centring
  term, so it does not fold exactly into the audio tower's last linear layer; a weight edit is a further step.
- "Held-out animals" means 9 or 10 groups per fold from one web collection of imitations; collateral is measured only on
  the committed clips and the 64 committed animal prompts.
- The verdict at large doses is the probe's score written along the verdict's prompt direction.

Every number above is printed from `results/part190/results.json` by the code in `part190/` at the registration
commit 755f745 (the run log is `results/part190/run.log`), except the exploratory C6 mechanism figures, which are from
the cold check's probe scripts in `cold/`.

## Cold check

DeepSeek V4.1 Flash, 2026-11-07 (`cold/COLD.md`): reran the committed code on the real arrays and found zero differences
from `results.json` at any key; every verdict, gate and table cell matched. Five findings on the write-up, all folded:
the C6 mechanism above (F1), "half the gain" for a qualifying dose that reaches 51% and 73% (F2), the fold group counts
reversed (F3), no provenance line binding results to the design commit (F4, the sentence above), and "levelled off" for
a fit fold that had saturated (F5).
