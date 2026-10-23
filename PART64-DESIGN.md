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

## Amendment 2 (2026-10-23 ~12:4x CT, before any collection or call; the round-2 stop rule fired on the money class)
The paidaction library caps each ledger at $8 and the Opus arms cost about $20, so fold 1 split the Opus groups into
several ledgers; round 2 (Haiku) found that this removed any bound on the stage as a whole, and that Opus 1x groups
still carry a worst case near $21.7 against one $8 ledger. Decision (Claude): keep the library cap; size every ledger
so its whole share completes under $8 at worst case (small chunks), and add a STAGE GUARD in gen64 that, before every
reserve, sums settled plus reserved across all ledgers of that writer's stage and refuses if the sum plus the new
reserve exceeds the stage cap: Haiku $4, Opus $28. The stage caps are registered here; changing them needs an amendment.

## Amendment 3 (2026-10-23 13:40 CT, before any collection or call; the round-3 stop rule fired on the money class a third time)
Round 3 found the stage guard of Amendment 2 summing only the ledgers under one `--root`, so a new run (which the
tool's own refusal message recommends) starts with an empty stage (Haiku F1, measured: two roots each reserved $4.85
under a $5 test cap). It also measured that the $28 Opus cap binds near 2,100 average output tokens per request
(thinking included), on a model whose output length on this task no live call has measured (Haiku F3). Decision
(Claude, under Elise's "we can burn them hard today" for the expiring API credit):
1. Every Part 64 writer ledger lives in ONE fixed directory, `results/part64/writer-stage/`, resolved against the
   repository root and independent of `--root`. The guard sums every ledger in it, whatever its run id, so no new run
   or new root resets a stage. `--root` is resolved to an absolute path at every entry point (Luna r3 F1).
2. The Opus stage cap rises from $28 to $40 (binds near 3,300 average output tokens); Haiku stays $4. The outer
   bound is the credit workspace's $85 monthly cap. Changing a cap still needs an amendment.
3. Probe gate (operator, runbook): after the two single-unit probes settle, project each writer's stage from the
   probe's billed tokens times the stage's request count at batch price; if Opus projects above $36 (90% of its cap),
   stop and report instead of running.
4. Also fixed in the same fold: the 5-minute created-at window applies only to re-probe IDs (Haiku F2); a re-probe ID
   is the stored base ID plus its attempt number (Haiku F4); the results parser splits on "\n" only (Haiku F5).

## Amendment 4 (2026-10-23 14:22 CT, before any collection result or paid call; from the Haiku round-4 confirm review)
1. The writer-stage directory is ONE fixed path on this machine, `/home/elisepc/assistant-basin/results/part64/writer-stage`
   (an environment override exists for tests only), not one per checkout (r4 F3: a second checkout started an empty stage).
2. The probe projection scales the probe's billed input by the stage's total request bytes over the probe's request bytes,
   and keeps output times request count (r4 F1 measured the per-request version 11% low, so the $36 STOP fired near $40).
3. `run` refuses after a STOP projection unless the operator passes an explicit override with a recorded reason.
4. The 5-minute created-at window applies only to requests that carry a re-probe suffix (r4 F2: a retried initial probe
   kept its unsuffixed IDs and slipped through). Decision (Claude).

## Amendment 5 (2026-10-23 14:50 CT, after the registered reproduction stop fired and before any paid call)
The first collection (Vast 3090, 14:28-14:46 CT) stopped at assembly as Amendment 1 requires: median fresh top-12
match 1/12. Cause, measured: the Part 56 pilot scanned the FIRST 1,000 documents of pile10k (PART56-DESIGN.md: "1,000
documents"; every pilot window has doc <= 999), and the fresh collection scanned all 10,000, so stronger windows from
documents 1,000-9,999 displaced the pilot's. Inside documents 0-999 the fresh collection reproduces the pilot exactly
(median share of in-range fresh top windows found in the pilot's top 20: 1.0 for 246 neurons and 248 controls, by
position and by text; activation ratio on 1,800 matched windows 1.000). Decision (Claude): the 3x collection scans the
same 1,000 documents (PILOT_DOCS = 1000), so 3x means the top 36 / random 18 / weak 9 of the corpus the 1x windows came
from, not stronger windows from ten times the text (which would confound "more text" with "better examples").
Amendment 1's reproduction check is unchanged. Assembly refuses any fresh window outside documents 0-999.
