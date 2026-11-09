# Fold report: Part 200 cold check, round 2

F1: Out of scope for this worker; the Part 163 front-page update was folded by Claude before this assignment.
F2: Correct the load timestamp to say the load record was written after the model loaded at 13:40:53.
F3: Explain `repair.chunk_size` 64 as the startup repair batch size; pushed generation calls ran in chunks of 8.
F4: Replace the length-change claim with the at-cap and under-cap broken counts for both runs, while retaining joint kernel-and-shape attribution.
F5: Remove the near-tie/SDPA-fault claim; report first-difference locations as exploratory and quote the design's limit on one-row decodes and the 64-key fault.
F6: Out of scope for this worker; the front-page batch-shape sentence was folded by Claude before this assignment.
F7: Say 163.4 held by one unit in each arm and give 163.5's intervals without calling it live absent its margin; qualify the verdicts-hold headline immediately because three of five could not fail.
F8: Wrap/shorten overlapping under-cap x labels, add mean length if clean (otherwise leave it in the table), and update image alt text.
F9: Identify the ~833-second difference between summary.seconds and the record span as most likely the untimed batch-1 stage, explicitly an inference from run200.py.
F10: Replace recipe/battery purposive-selection wording with “chosen by hand, no recorded rule.”
F11: Correct Gate 1 wording to the changed-share floor; random is 71 of 100, one unit above 0.70.
F12: Change “batch-8” to “chunked answers (chunks of 8).”

SUMMARY: This fold applies the Part 200 cold-check wording and figure corrections, regenerates report200.md/json and fig200.png, and passes the requested publish check (all 9 checks).
FINDINGS: F2–F5 and F7–F12 are in scope; F1 and F6 were folded by Claude before this worker's assignment.
BLOCKERS: none.
