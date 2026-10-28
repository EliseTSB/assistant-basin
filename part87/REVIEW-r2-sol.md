# Part 87 cold review, round 2

Reviewed commit `7bef7e6195cb96193879cc89297bcd151f7e1296`; paths below are relative to `/home/elisepc/green/worktrees/ab-part67`. Repository stayed clean. Evidence and disposable copies are beside this review.

## Requested summaries

Ran each unmodified command once with `/home/elisepc/neuron-probe/.venv/bin/python -I -B`, scratch-local TMPDIR, OMP/MKL threads=1; all exited 0:

| Script | Exact summary |
|---|---|
| part87/test_part87.py | `Ran 21 tests in 7.396s` / `OK` |
| part87/test_audit87.py | `Ran 18 tests in 340.880s` / `OK` |
| test_observe.py | `Ran 22 tests in 0.404s` / `OK` |
| part87/mutate87.py | `MUTATE87: 15/15 killed; all killed` |

Full outputs: `logs/base-*.log`; structured results: `baseline-results.json`.

## Findings, most severe first

1. **BLOCKER — Windows phase commands cannot start.** `part87/RUNBOOK.md:42,52,60,66,74,89,97,103,109,117` put the interpreter into `steps.txt`, but `gpu_job_runner.py:12` prepends `sys.executable`. The maintained runner also does this (`/home/elisepc/green/Journal/scripts/neuron-gpu-job/job.py:91`). Python therefore reads `python.exe` as source. **Fix:** run job.py with the specified venv; put only script plus arguments in steps.txt. `runbook-command-check.json` preserves every expanded argv.

2. **HIGH — audited answers can be scored in the wrong groups.** `part87/part87.py:560` supplies `question_kind`; `part87/score87.py:184` trusts it, while `part87/audit87.py:256-289` validates question/sample but ignores kind. N11 swaps six maker questions with the six guards. A complete production-loop CPU fixture returned `audit87 run: 1560 answers checked; worst relative error 0.4061%` / `AUDIT87 run: PASS`, without overrides. Scoring still sees 96 held-out and 24 guards per main arm, containing the wrong answers. **Fix:** derive kind/index from registered questions in scoring and validate saved metadata in audit. See `probes/N11-full/`.

3. **HIGH — recorded boundary violations are ignored.** `part87/audit87.py:297-319` checks only a subset of observer facts. N12 replaces L0's down-projection forward with an unobserved zero-output implementation; `observe.py:434` reports `unobserved_forwards={"0":3}`, yet pilot audit passes. N01/N02/N03 likewise pass with top_k=1, repetition_penalty=1.8, or min_new_tokens=4; the observer records these (`observe.py:47-50,381`). Top-k=1 effectively defeats sampled generation. **Fix:** reject unobserved forwards/shape changes and enforce all effective generation settings against an independently pinned reference, with negative tests.

4. **HIGH — model provenance is not independently enforced.** The run loads at `part87/part87.py:676`; N13 removes its revision argument and all tests pass. `observe.py:223-226` records name/dtype/device but no resolved revision, and audit never validates model identity. Indeed the retained 30-layer CPU Qwen2 fixtures with an empty model name receive the production PASS line without overrides. **Fix:** observe resolved checkpoint identity and validate revision, architecture, dtype and production device; isolate fixture allowances behind the marked override path.

5. **HIGH — wrong measurement remains self-consistent.** Means come from `part87/part87.py:279-301`; the integration test substitutes synthetic doses (`part87/test_part87.py:467-489`). N06-N10 all survive. Audit trusts supplied means (`part87/audit87.py:206-220`) and anchors only L22 (`:244`); wrong-layer, last-position or prompt-only means can therefore drive compliant-looking pushes. `_read_unchanged_heldout` checks 192 rows, not the registered file's identity (`part87/part87.py:163-173`), permitting a wrong answers file. The calibration SHA check (`:254-261`) does not prove the checked text reaches measurement: N09 passes half the text; only an out-of-range L22 median would stop it.

   **Smallest independent check:** before locking doses, a separate implementation replays all 192 pinned Part 66 answers on the pinned model, independently renders/tokenizes prompt+answer, and recomputes target/random-unit means and counts at the four named down_proj inputs. One calibration forward can independently collect all five residual medians, excluding token zero. Compare against doses with explicit numerical tolerances; pin source-answer/calibration digests. A few answer spot checks cannot certify the aggregate means.

6. **MEDIUM — combined blind input is not bound to scored 87B data.** Preparation stores both digests (`part87/score87.py:112-117`), but scoring checks only A (`:411-421`). `scoring_probe.py` changes U's blinded “Alibaba” answers to “Google”; scoring accepts 96 Google matches despite the changed B digest. **Fix:** validate B's digest too whenever consuming the combined key.

## Round-1 closure and historical kills

Old #1 is **partial**: real pilot plus only random seed 8700 reaches integration (`part87/test_part87.py:493-515`); measurement and main full-run paths remain uncovered. Old #6's interpreter prefixes were added, but the runner incompatibility above prevents closure. The other nine original findings are closed as written:

- #2: both NOT READ regressions (`part87/test_part87.py:278-287`).
- #3: separate M/F statuses (`part87/score87.py:270-289,462`).
- #4: shared shuffled A/B random IDs (`part87/score87.py:94-108`); finding 6 is a subsequent binding gap.
- #5: 14.5 GB default/commands (`part87/part87.py:743`; `part87/RUNBOOK.md:52,74`).
- #7/#9: Amendment 1 and standalone reader rubric (`part87/RUNBOOK.md:126,18,100`).
- #8: calibration hash comparison (`part87/part87.py:254-261`).
- #10/#11: hand-read ingest and mandatory A digest path (`part87/score87.py:364-385,430-433,416,480`).

All historical mutants die for the intended property, not import/setup failures: R1 temperature; R2 layer; R3 sign; R4 missing hook; R5/R6 missing prefill; R7 channel; R8 prompt; R9 RF write magnitude; R10 sampling; R11 CPU seed; R12 median-layer identity; R13 F sign; S1 missing random arm; S2 absent origin. Exact diagnostics are in `logs/base-mutate87.log`; assertions are `part87/test_part87.py:502,514,232,139,278,285`.

## New mutations

Every row modifies **part87.py alone** in a complete disposable dependency tree. Each ran the full requested 21-test command and **survived**; see `all-new-mutations.json`, scripts and `logs/N*.log`. The independent audit-fixture suite does not exercise this target (`part87/test_audit87.py:138-179`).

| ID | Change at original line | Additional audit evidence |
|---|---|---|
| N01 | :539 top_k=1 | pilot PASS |
| N02 | :539 repetition_penalty=1.8 | pilot PASS |
| N03 | :539 min_new_tokens=4 | pilot PASS |
| N04 | :516 halve E63 only in full run | not replayed; write-norm check should reject |
| N05 | :475 RF seeds >8700 use M | not replayed; write-norm check should reject |
| N06 | :293 measure prompt only | measurement not exercised |
| N07 | :288 capture preceding layer | measurement not exercised |
| N08 | :281 measure last position only | measurement not exercised |
| N09 | :236 use half calibration text | measurement not exercised |
| N10 | :293 reuse first stored answer | measurement not exercised |
| N11 | :560 swap maker/guard kinds | pilot and full run PASS |
| N12 | :686 replace L0 forward, output zero | pilot PASS |
| N13 | :676 unpin generation revision | pilot PASS |

These are real-loop/observer records on tiny local models, not production Qwen3.5/A770 measurements. Pilot audits above used no fixture flags.

## Reading rules and command walkthrough

Thresholds match `PREDICTIONS.md:2164-2172` (`part87/score87.py:227-304`), including strict random maxima/ties, pilot/missing-input NOT READ, separate arm eligibility and lexical Google counts. Hand reads remain supplementary. `part87/RUNBOOK.md:126-127` preserves per-arm posting and Amendment 1; shared blinding is constructed correctly, subject to finding 6.

CPU preflight commands (`part87/RUNBOOK.md:25-28`) are valid direct-shell commands. Every subsequent phase—measure, pilot, pilot audit/score, full run/audit, blind prepare/score, initial score and hand-read re-score—has valid CLI arguments but fails through steps.txt until finding 1 is fixed. Copy dependencies and output-directory creation are sufficient. Windows execution, calibration bytes and A770 memory/timing were not measured here.

VERDICT: DO-NOT-SHIP
