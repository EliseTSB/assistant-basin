# Fold cold check, round 1

- Finding 1: added computed mean/median lengths and cap/under-cap broken splits to `report200.py` and `report200.md`, then placed the length table before cross-run rates in both the generated report and README; noted that 366/387 and 98/103 rerun broken answers hit cap and under-cap broken fell 318→21 neuron, 72→5 random.
- Finding 2: added joint kernel-and-batch-shape attribution to the README Answer paragraph and 200.3/200.5 result rows, following the design wording.
- Finding 3: computed published/rerun Gate 1 changed shares by arm and floor in `report200.json`/`.md`; README states 0.88→0.75 neuron and 0.88→0.71 random against 0.70 and explains the eager recipe gate is narrow.
- Finding 4: retitled the README around live readings 163.4/163.5; first paragraph says 163.4 was one unit from flipping and 163.1/163.2 could not fail at these rates while 163.3's bar was far below them.
- Finding 5: added exploratory identical-text counts and under-cap mid-phrase counts to the README from `compare.json`.
- Finding 6: added first-difference convention, retokenization totals, under-cap stop-token count, ±1 uncertainty, 63/64/65 sensitivity, and the exploratory sub-64 near-tie reading; report data is computed by `report200.py`.
- Finding 7: documented purposive recipe/battery sampling and first-five-changed-in-file-order batch-1 sampling in README and `hand_read.md`.
- Finding 8: documented the inherited Part 199 label/defaults, observed chunks of 8, 0.89 answers/s against 0.17 guard, and the 4,670s timestamp span versus 3,836.6s generation in README and generated report.
- Finding 9: intentionally unchanged — Claude's PREDICTIONS.md and Part 163 README items are outside this assignment.
- Finding 10: added under-cap broken counts from `report200.json` to the figure's left panel and regenerated `fig200.png`.

SUMMARY: Findings 1–8 and 10 folded into the report code, rendered report, README, hand-read notes, and figure. Publish check passed all nine checks.
FINDINGS: `report200.py` and `fig200.py` were rerun with the specified isolated interpreter; 200.1–200.7 wording remains registered verbatim. Parent-control mailbox was empty at each poll; no message was applied or deferred.
BLOCKERS: none.
