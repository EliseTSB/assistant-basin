# Part 86 fold 1, cold review round 2 (Claude Haiku 5.5, read-only, commit e493052, 2026-10-27 23:4x CT)

Verdict: SHIP-WITH-FIXES. Unmodified suite: Ran 14 tests, OK (also with PART86_GPU_STRICT=1, no skips). A second
round-2 reviewer (DeepSeek V4.1 Flash, cheap-review) spent all 131 turns and wrote nothing.
Stop rule (card): B1 is the r1 class "verdict does not match the written rule" again, and H1 is the r1 class "tests
that cannot fail" again, so both are REDESIGNED in fold 2 (golden cases written by Claude, a committed mutation runner),
not patched. The reviewer's scripts are in part86/review-r2-tools/ (e2e864.py, edge.py, mutate.py, prov.py).

Round 1: #1 FIXED (screen86.py:81-104, 235-240; orphan and verify refusals untested). #2 PARTLY (86.2 exact at
blind86.py:204-214; 86.4 wrong, B1). #3 PARTLY (median, verdict wiring, canary, verify refusal untested, H1). #4 FIXED.
#5 PARTLY (zero row only must name Qwen; batch-16 vs batch-1 recorded not gated; per-row counters still tautological at
common86.py:218-222). #6 PARTLY (RUNBOOK:39 3.8282 should be 3.82815). #7 PARTLY (picker input and pick-to-neuron
mapping unstated, RUNBOOK:117-118). #8 FIXED in code, untested. #9-13 FIXED.

B1 BLOCKER. 86.4 does not implement the written rule: blind86.py:240 computes n - coherent, so coherent "still Qwen or
an assistant" answers count as failing. Synthetic layers of 3 coherent personas, 4 coherent still-Qwen, 3 incoherent:
code HOLDS in 17/17, written rule FAILS in 17/17. blind86.py:243-248 returns NOT READ with fewer than 9 read layers even
when a read layer fails.
B2 BLOCKER. Provenance hashes every part86/*.py (common86.py:46-54) and refuses on mismatch (common86.py:289-306,
verify86.py:104-105), so any scorer or test edit after the screen starts blocks scoring unless the 12-hour screen reruns.
Hash generation-path files only (common86.py, screen86.py, neuron_report.py) into screen configs; scorer and verifier get
their own manifests.
H1 HIGH. 13 of 14 reviewer mutations left the suite green: reset no-op (screen86.py:107-116); per-layer 9,216/512 check
(blind86.py:180-182); 86.2 real/random swap (blind86.py:320-322); residual median (common86.py:134-156); runtime canary
(screen86.py:137-142); verify resume refusal (verify86.py:166-167); gate-3 status and orphan refusal (screen86.py:172-173,
289-293); kept re-triage, changed count, verify median (blind86.py:113, 163-167; verify86.py:119-122); 86.6 range
(blind86.py:346); 86.4 equality (blind86.py:241-242). verdicts() never runs on full-size data.
H2 HIGH. The reader pass is unscoped: one tool-less session for up to ~34,000 items plus 5,280 verification items, no
chunks, second rater or agreement. (Decided in Amendment 2.)
M1 MEDIUM. Per-row contract not gated in the real path: counters increment for every row together; the zero-row check
accepts any answer naming Qwen. Gate the zero row against unsteered tokens of the same batch shape, after measuring the
bf16 batch effect in the pilot.
M2 MEDIUM. Kept and broken answers are never read and BUILD-NOTES.md:9 is false; some can be persona claims ("I am a
language model from the planet Zorg..." triages kept). (Decided in Amendment 2: miss-rate samples, rates are lower bounds.)
M3 MEDIUM. A picked persona is not checked against its screen label (verify86.py:69-77).
L1 the automatic 86.1 match holds on denials (Amendment 2: 86.1 read by hand). L2 verify86 score writes no manifest
though the runbook says so; BUILD-NOTES.md:10 describes the superseded 86.4. L3 the 10-real floor never binds (Fisher
needs about 56 with 0 random, 87 with 1). L4 reset runs before config checks (a wrong --calib with --reset-layer wipes a
layer); the verify median uses exact float equality.
Could not check: GPU/XPU, Qwen3.5 weights or tokenizer, eos 248044, model-default generation settings, throughput.
