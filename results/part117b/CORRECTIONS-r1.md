# Corrections after Alden's review

Date: 2026-10-30. Alden's review is credited to Council post 11. The scorer and wording were corrected within the registered Part 117b design.

## 1. Sampled direction controls

The README now states that a random direction is a dense Gaussian vector in MLP input space. The vector is normalized there, projected through `W_down`, and rescaled to the matched concept's target residual write norm. It is not a uniform residual-space direction. The README calls these sampled random directions and sampled atlas-negative neuron directions. Atlas-negative means outside Part 112's 465-unit persona atlas. It does not establish that a unit is non-concept or non-persona. The comparison is layer and written-norm matched, not matched on downstream influence. A difference supports sensitivity to these sampled direction classes. It does not establish a concept-specific mechanism by itself.

## 2. Slot dose controls

The README now describes +0.85 and +0.5 as dose controls. The slot persona appeared in 11 of 11 rows at +1, 2 of 11 at +0.85, and 0 of 11 at +0.5. The nominal 15% and 50% dose reductions both changed the result. These controls do not separate normalization dilution from interference. In an idealized orthogonal-vector example, the normalized slot coefficient is about 0.848 after adding a second push. The equivalent slot-only input multiplier is about 0.781 because reducing the input also reduces the denominator. The +0.85 result does not rule out normalization-mediated dilution. Near-zero write cosine across layers does not bound downstream interaction.

## 3. Erasure headline gates and registered counts

`part117b/score117b.py` now requires at least 9 eligible and readable pairs for paired headline claims. This floor follows the design's smallest-effect wording of about 8 of 11 pairs, with differences of one or two pairs treated as noise. Random and atlas-negative controls need at least 9 pairs with all three control labels readable. An erasure reading also requires coherence in at least two thirds of its readable rows. Missing labels remain missing. They cannot count as an absent persona. Withheld readings include the failed coverage or coherence reason. The ten registered verdict calculations are unchanged. The scorer prints read and missing counts beside each verdict.

`part117b/test_117b.py` adds a matched-label missingness case and an incoherent matched-response case. Both returned the concept-erasure headline on the saved pre-fix scorer from `git show HEAD:part117b/score117b.py`. Both are withheld by the fixed scorer. A complete, coherent positive control still returns the concept-erasure headline. The published data have 11 of 11 readable matched and unrelated pairs. Matched coherence is 8 of 11. Unrelated coherence is 7 of 11, below the required 8 of 11, so the unrelated erasure reading is withheld. Random coherence is 15 of 33, below 22 of 33. The current main reading is therefore withheld for that reason as well.

### Published verdicts and headline, old against new

The fixed scorer was rerun with `python3 part117b/score117b.py` from the repository root. Every prediction verdict stayed the same. The read and missing columns are the fixed scorer's output.

| Prediction | Old verdict | New verdict | New read / missing |
| --- | --- | --- | ---: |
| 117b.1 | held | held | 11 / 0 |
| 117b.2 | held | held | 11 / 0 |
| 117b.3 | failed | failed | 33 / 0 |
| 117b.4 | failed | failed | 20 / 0 |
| 117b.5 random / control | failed | failed | 33 / 0; 32 / 1 |
| 117b.6 | failed | failed | 11 / 0 |
| 117b.7 matched / unrelated | failed | failed | 11 / 0; 11 / 0 |
| 117b.8 | failed | failed | 11 / 0 |
| 117b.9 | failed | failed | 32 / 1 |
| 117b.10 | failed | failed | 11 / 0 |

Old headline: `the random push breaks the answers; not read as erasure`.

New headline: `withheld (random coherence is 15 of 33 read rows, below two thirds (requires 22 of 33); random-direction erasure reading is withheld)`.

## 4. Control rows and repeated vectors

The README now lists random and atlas-negative control fractions for each pair. It reports 27 unique random vector ids among 33 rows. Nine distinct matched concept neurons each supply three vectors. Repeated concept ids reuse those vectors. The README states that the rows and selected pairs are not independent draws.
