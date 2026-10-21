# Fold brief: vastrun small, collection finding (Luna)
Claude's host check: there are TWO pagination implementations, module-level _listed_instances (vastrun.py:232, used by
the exact-label cleanup at :278, the timed listing at :605 and sweep_expired at :1446) and Runner._list_all_instances
(:626). A sabotage making _listed_instances read only the first page (next_token forced None at :256) passes all 46
tests, so no cleanup path is tested against pagination.
1. Keep ONE pagination implementation; every listing caller uses it.
2. Tests, each with our instance only on page 2 behind a neighbour on page 1: the exact-label cleanup after a finished
   job; cleanup after a lost rent response; sweep_expired; "sweep LABEL"; the pre-rent preflight refusing on a live
   non-expired label that is only on page 2.
3. Re-run the five original sabotages plus "first page only" in the single implementation; each must red a test.
Nothing else changes. Commit nothing. Report REPORT-vastrun-small-fold1.md ending SUMMARY / FINDINGS / BLOCKERS.
