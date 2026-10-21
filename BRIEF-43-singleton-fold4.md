# Build brief: Part 43 calibration, raw-only artifact (Luna) — replaces the fold-3 gate

Run size: code and CPU tests only, no GPU. Implement review/part43/CALIBRATION-RAW-ONLY-DESIGN.md exactly as resolved
by review/part43/ALDEN-17.md (binding; where they differ, ALDEN-17 wins). Replace the fold-3 calibration writer and
gate; do not keep the old schema or any fallback to it. Keep the pilot's order: calibrate -> serialize raw -> read
from disk -> reader verdict -> only then signs/lattice/swap/attach.
Every acceptance item in ALDEN-17.md must have a test, and these sabotages must each fail at least one test:
(a) the verdict reads any stored number other than the per-answer sums; (b) the duplicate-key check removed;
(c) the derived-key refusal removed; (d) internal errors caught as malformed.
Do not change 0.05, any pre-registered number, the scoring semantics of teacher_forced_scores, or anything outside
calibration, the pilot's gate step, the fixture and tests. Keep every unrelated existing test green.
Commit nothing. Report REPORT-43-singleton-fold4.md: the schema (with version), the sabotage results, then SUMMARY /
FINDINGS / BLOCKERS.
