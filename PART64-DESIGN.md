# Part 64: three times the sample text, and a stronger writer (design, 2026-10-23)

Elise, 2026-10-23 10:24 CT: *"i want to try giving the judges like 3 times more sample text next."* Part 60
registered this as M7 (more windows) and skipped it because the atlas pilot records do not hold enough windows. Part 64
collects them fresh and crosses sample size with the writer model.

**Size, first:** 500 units (Part 60's dev split: 250 real Qwen3-1.7B MLP neurons, 250 random directions) x 4 arms =
2,000 writer calls (Message Batches); 2,000 x 12 generated sentences scored on Qwen3-1.7B; one window re-collection
for 500 units. Rented 3090 for collection and scoring, about 1.5 h, about $0.5. API: Haiku arms about $1, Opus arms
about $20 (the monthly credit, which expires 2026-10-24). Written before any collection or call.

## Arms (2 x 2, paired on the same units and the same windows)
- **Writer:** Claude Haiku 5.5 (Part 60's M1 prompt, unchanged) or Claude Opus 5.5 (the same prompt). Both at their
  API defaults (both think by default; no `temperature`).
- **Sample:** 1x = exactly the windows M1 showed in Part 60 for that unit; 3x = three times as many windows in each
  category (top, random, weak), same window length, from a fresh collection over the same corpus slice. The 1x
  windows must be a subset of the 3x windows (checked, not assumed).
- Haiku-1x reruns M1 so every comparison is within this run (Part 60's M1 is a cross-check, not the baseline).

## Measurement (Part 60's, unchanged)
Sentence test on Qwen3-1.7B with Part 60's thresholds, tiers and pass rule (at least 3 of 4 strong hits, at most 1 of
4 none hits). Primary number per arm: real pass rate minus control pass rate (the gap), with a unit-paired bootstrap.

## Predictions (written before data)
| # | Prediction | Result | Evidence |
|---|---|---|---|
| 64a | Haiku-3x raises the real pass rate over Haiku-1x by at least 5 points. | | |
| 64b | Haiku-3x raises the gap over Haiku-1x by at least 5 points (more examples make a hypothesis more specific). | | |
| 64c | Haiku-3x raises the control pass rate by at most 3 points over Haiku-1x. | | |
| 64d | Opus-1x has a gap at least 5 points above Haiku-1x. | | |
| 64e | Opus gains more from 3x than Haiku does (interaction on the gap at least 3 points). | | |
| 64f | Haiku-1x reproduces Part 60's M1 within 6 points on both real and control pass rates. | | |

Decision rule: an arm "advances" only if its gap beats Haiku-1x by at least 5 points with the paired bootstrap 95%
interval above zero. Part 60's test split stays closed unless an arm advances.

## Limits, written now
One scoring model (Qwen3-1.7B), small models only (Elise's point that cleanliness may grow with scale is not tested
here). 64e is underpowered at 250 units per type and is reported as a direction, not a finding.

## Amendment 1 (2026-10-23 10:3x CT, before any collection or call; from the round-1 build stop)
The sparse atlas collector keeps only 20 top, 10 random and 5 weak windows and cannot target units, and a fresh
re-collection need not reproduce the pilot's windows exactly. So the 3x set is built so the subset holds by
construction: the exact 1x windows from each unit's Part 60 pilot record (top 12, random 6, weak 3) plus fresh windows
from a new targeted collection (same model revision, same corpus file, recorded by sha256), excluding duplicates, up
to 36 top, 18 random and 9 weak. Reproduction check: per unit, how many of the fresh top 12 match the pilot's top 12;
the run stops if the median is below 10 of 12.
