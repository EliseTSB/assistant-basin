# Part 85 fold 1, cold review round 2 (Claude Haiku 5.5, read-only, 2026-10-27 22:2x CT)

Verdict: SHIP-WITH-FIXES. Round 1: 11 of 12 fixed (confirmed in code); #10 (test coverage) not fixed.
Stop rule (card, 21:33): r2 found the r1 blocker's class again (#2 below: a test that cannot see run() defects), so
that piece is REDESIGNED in fold 2, not patched: the smoke test must assert on what run() WROTE (each answers file's
unit, clamp value, template SHA, nonzero hook counts per row; a chat run with a stub template; a resume run; a
changed-step-1 refusal), never on helper calls.

1. HIGH. No handoff between cloud sessions. CLOUD-RUN.md has no copy-in/copy-out. Non-Instruct sessions need
   instruct/step0.json, instruct/step1.json and neutral-token-ids.json; without the token file step1_acts.py:90-91 fails
   after the model loads; step0_weights.py:496 reads --reference after a 5.9 GB download; main runs need
   pilot-score.json (part85.py:221); scoring reads all twelve trees from one root (score85.py:24); cells live only on
   session disk (part85.py:431-440). Fix: design the handoff (git branch per checkpoint, references fetched first),
   check every reference file before any download, push results after each cell.
2. HIGH. The smoke test passes with run() broken: unit/value None at part85.py:394 passes; a raising chat builder
   passes; a hook that stops clamping passes (only the hook unit test catches it). The other 11 checkpoints' rows are
   written from part85._cell_value (test_part85.py:527-556), so they check nothing.
3. MEDIUM. Chat-only scope cannot complete: score85.py:512-515 raises NOT READ for 85.5-85.9; :247-253 demands all 12
   step files. Fix: not-applicable verdicts under chat-only; steps required only for in-scope checkpoints.
4. MEDIUM. step1_acts.py:189 collects before the reference and token-hash checks (:194-203). Move checks above :179.
5. Preflight DONE 2026-10-27 22:2x by Claude: the 16 off-world words have 16 distinct first tokens in OLMo 2's
   tokenizer (Instruct 48d788e). Note for limits: three first tokens are generic pieces (" extr", " star", " inter").
6. MEDIUM. part85.py:301-307 hashes the whole step1.json (device, versions); hash only maxima, token-ids SHA, commit.
7. MEDIUM. --snapshot-dir provenance unchecked (step0_weights.py:494-495, 517; part85.py:352-353;
   step1_acts.py:174-175): require the directory to match cp["commit"].
8. MEDIUM. Untested: analyze_snapshot, SnapshotTensorReader reads, create_neutral_sequences, token-hash refusal,
   pilot gate amend / chat-only / quality-failed branches.
9-14. LOW: step files not identity-checked (score85.py:241-244, 299); seeds and pairs unchecked (score85.py:112-128,
   part85.py:403); score outputs overwrite (score85.py:561-563), no attempt-2 paths in CLOUD-RUN.md:82-88; dead
   self-compare (score85.py:64-65); unpinned installs (CLOUD-RUN.md:17; local runs used transformers 5.20.0, torch
   2.14.1); sharded .bin rejected (step0_weights.py:77-79), :531 ignores --results-root.
Checked, no defect: pinned downloads; one weight format per repo; 17/17 tests; peak memory about 9 GB estimated;
a crash loses at most one cell.
