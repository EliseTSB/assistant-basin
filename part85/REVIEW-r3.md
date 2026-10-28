# Part 85 fold 2, cold review round 3 (Claude Haiku 5.5, read-only, commit 1df997b, 2026-10-27 23:4x CT)

Verdict: DO-NOT-SHIP. Unmodified suite: Ran 28 tests, OK. The reviewer's scripts are in part85/review-r3-tools/
(mutate.py with its ten mutants; handoff/flow.py and setup.sh reproduce the handoff failures on scratch bare remotes).
Stop rule (card 23:2x): the run()-blind-test class returned a third time (finding 5), so the cloud pilot is parked and
the handoff is redesigned (decision part85-handoff-after-r3-2026-10-27, option 1).

Round 2: #1 PARTLY (per-cell push works, part85.py:448-462, handoff85.py:199-234; open findings 1-4). #2 PARTLY
(asserts on files and run.json; 7 of 10 new mutants survive). #3, #4, #6, #7, #9, #11, #13, #14 FIXED. #5 DONE. #8 MOSTLY
(run-level push and template paths untested). #10 FIXED in scoring, not at run time (part85.py:268-290). #12 PARTLY.
Pins checked against PyPI metadata: datasets 5.1.0 needs hub >=1.31,<3; transformers 5.20.0 accepts hub 2.2.0.

1. BLOCKER. handoff85.py:360-361 refuses cross-branch byte differences, and main runs rewrite instruct/run.json
   (part85.py:328-347, 455-457) after restoring the pilot branch (CLOUD-RUN.md:140): assemble of part85-run/pilot plus
   part85-run/instruct exits 1 ("different bytes at instruct/run.json"). Same for pretrain_4T.
2. HIGH. handoff85.py:223 commits on the pinned commit, not the remote tip, so a restored session diverges, and
   _merge_run_records (116-133) refuses the changed template fields: attempt 2 fails at its first push; the pilot2 cell
   exists only on the session disk.
3. HIGH. CLOUD-RUN.md:95's pilot-frame-v2.txt is carried by nothing (push allows only checkpoint dirs and
   neutral-token-ids.json, handoff85.py:100-105): a fresh clone fails at part85.py:297.
4. HIGH, silent loss. handoff85.py:224-233 skips retries when the branch is absent and 220-222 returns success when
   nothing is staged: a rejected first push, re-run, exits 0 with the branch still absent.
5. HIGH, test class not closed. Seven of ten new mutants survive: layer-9 clamp (part85.py:390; the observer is on the
   same module); chat frame skipping apply_chat_template (119-128); custom template ignored in prompts (417); push before
   the answer write (448-462) and step-0 push before its write (step0_weights.py:536-541; the subprocess mock is only
   counted); push naming the wrong checkpoint (461); template-mismatch refusal removed (241-243). The run fixture uses
   maximum 6.5 for every checkpoint (test_part85.py:543-551), so own-max equals absolute in the run tests.
6. MEDIUM. score85.py:510-527 checks template hashes only in pilot mode.
7. MEDIUM. rm -rf "$SNAPSHOT_DIR" (CLOUD-RUN.md:59,76,104,110,172) frees nothing: delete the models--<repo> cache dir.
8. MEDIUM. No git identity is set; the commit at handoff85.py:223 fails without one, after the file is written.
9. MEDIUM. PINNED_COMMIT is circular (CLOUD-RUN.md:5 vs :43); the references commit must be pushed (:135 never says so);
   line 130's "at the repository root" must mean results/part85/... (handoff85.py:260,319).
10. MEDIUM. The neutral-token digest is enforced only by instruction (CLOUD-RUN.md:21; step1_acts.py:85-88 checks
    self-consistency only); tokens are pushed only after step1.json is written.
11. LOW. assemble accepts a run.json claiming a cell with no answer file. 12. LOW. Two sessions on one key collide on
    elapsed_seconds (safe refusal, runbook silent). 13. LOW. Steps 0 and 1 have no skip-if-present though CLOUD-RUN.md:151,164
    say to omit them; the code commit is recorded nowhere. Test subprocesses lack -B (they write __pycache__).
