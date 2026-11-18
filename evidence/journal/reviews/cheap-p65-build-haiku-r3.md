<!-- cheap-review.mjs cheap-review-p65-build-haiku-r3-2026-10-24: /home/elisepc/green/worktrees/ab-part65 6b0f430..47e37c1, lane haiku, df-run exit 0 -->
# Cold adversarial review: p65-build (round 3)

Scope: `git -C /home/elisepc/green/worktrees/ab-part65 diff 6b0f430..47e37c1` (commits 680d973, 66386a9, 552a724, 47e37c1).
Scripts that drive the real code: `scripts/m1_extract.py`, `m2_ledger_chain.py`, `m2b_pending_resubmit.py`, `m3_pilot_gate.py`, `m4_cost.py`, `m5_contract.py`, in this directory. Fake API and temp ledgers only; no network, no live ledger, no credential files.

## Verdict: SHIP-WITH-FIXES

The Amendment 2 analysis (complete-case units, 1,000 denominator, 50-trial stop, bounds, strict read) and the Amendment 3 chat-mode change hold up under measurement. The judge's released-chunk recovery does not: a fresh out-dir can POST a chunk that is still pending elsewhere under a reservation that covers one batch (F1), and the released-chunk message the change adds never prints on the run that receives the rejection (F2). Fix F1 and F2 before the full judge run. F3 to F6 are smaller and can follow.

## Findings

### F1 · INTRODUCED · medium: a fresh out-dir re-POSTs a chunk still pending in another out-dir

Where: `part65/judge65.py:539-540` (returns an attempt still `reserved` in the ledger), `564-566` (a `ready` chunk reuses it, no new ledger reserve), `580-584` (POST).

Measured (`scripts/m2b_pending_resubmit.py`):
```
dirD POSTs: [['t0','t1'], ['t2','t3']]   chunk-0000 status=submitted, chunk-0001 status=released
dirE (fresh, same plan) POSTs: [['t0','t1'], ['t2','t3']]   chunk-0000 status=submitted, attempt=b06b... (same as dirD)
ledger: reserve b06b, reserve 46f7, release 46f7, reserve 3334, release 3334
```
Chunk 0 reached the provider twice under one attempt and one reservation. The RUNBOOK's recovery for a released chunk ("rerun the whole judge ... into a new --out-dir") is this path. At the m4 estimate one chunk costs about $0.003, so the exposure is bounded by one chunk per fresh directory, but it is a paid action that exceeds its reservation.

Why it matters: RUNBOOK's "do not resubmit" covers only ambiguous no-ID POSTs. A known pending batch is recovered with `--recover`, not resubmitted. Nothing records which out-dir made the reservation.

Smallest fix: store the out-dir (or its resolved root) in the reserve meta. In `_prior_reservation`, reuse an active attempt only when the recorded root equals this root. Otherwise raise `JudgeError` and tell the operator to `--recover` the old directory. The same-directory crash case (ledger `reserved`, chunk still `ready`) keeps working.

### F2 · INTRODUCED · medium: the released-chunk message and exit 2 are not reached on the run that gets the rejection

Where: `judge65.py:600` (re-raises `APIHttpError` after a validated release), `639` (`run_chunks` catches only `JudgeError`), `616-617` (later invocations return `released` quietly), `696-714` (the new message block, reached only from `main` when totals show released).

Measured (`m2b`): the first run ends with `dirD RAISED APIHttpError - Anthropic HTTP 400` and returns no totals. A later `main --recover` on dirD (with `run_chunks` bound to the fake API) prints "released judge chunk ... error ... a released chunk is final ..." and returns 2 with no POST. So the no-re-POST rule holds. But the first run dies on an uncaught exception, and RUNBOOK's "the judge prints the chunk path and its error, then exits nonzero" is false for that run. I inferred the first run's exit status from the uncaught exception; I did not run it through `main`.

Test gap: `test_part65.py:798-813` mocks `run_chunks` to return `released: 1`, so it passes without any real chunk transition. `815-825` covers only a pre-set released state.

Smallest fix: in `run_chunks`, catch `APIHttpError` when the chunk's status is `released`, count it in `totals["released"]`, and stop the loop (keeping the abort so later chunks are not submitted). Add a test that drives `run_chunks` with a fake API returning 400 and asserts `main` prints the message and returns 2.

### F3 · PRE-EXISTING · low: a third fresh directory with an unchanged plan refuses before any POST

`judge65.py:526-527`. Measured (`m2`, dirC): after two validated releases of the same plan, `JudgeError: multiple paidaction reservations exist...`, zero POSTs, and chunk 1 is never reached. It needs an unchanged plan, which RUNBOOK says not to rerun. Fix: count only non-released matches as duplicates, or name the situation in the error.

### F4 · INTRODUCED · low: the RUNBOOK's equal-norm claim is false

`RUNBOOK-65.md:23` says the gate "does not recompute run-wide norms from the pilot sample". `steer65.py:702` and `716-719` do: they compute the spread per (layer, dose) across pilot rows. Measured (`m3`): two rows at layer 1, each passing its own checks, with medians 1.0 and 2.0 give `SteerError: pilot-check failed: +0.8 push norms differ in layer 1 by ...`. The code is stricter than the doc. Practical effect is small, since medians are per layer. Fix: correct the sentence.

### F5 · INTRODUCED · low: the "under three cents" reserve claim is not met by my estimate, and no code enforces it

`PART65-DESIGN.md:145` (Amendment 2 rule 6): "the whole judge reserves under three cents". Measured (`m4`) with the real collection, labels, sentence tests and `build_trials`. `ranked.jsonl` is not on disk, so write summaries are synthetic (10 words, 12+12 lens tokens). Result: 1,000 trials in 10 chunks, total worst-case reserve $0.03309, mean canonical request 1,521 bytes, `max_tokens` 32. Staying under $0.03 needs a mean request of about 1.3 KB or less. The only enforced cap is the $1 sub-cap. Fix: restate the figure from the real ranked file, or drop the claim, and re-measure when that file exists.

### F6 · PRE-EXISTING · low: lowercase answers count as valid

`judge65.py:389` does `text.strip().upper()` before the `fullmatch`. Measured (`m1`): `'a'` gives answer `'A'` with no error. Amendment 2 rule 1 says "a lone A or B". Fix: drop `.upper()` if the judge prompt demands capitals, or amend rule 1 to say case-insensitive.

### F7 · INTRODUCED · low: nothing in analysis enforces "answers from two output directories are never combined"

`analyze65.py` has no provenance check: no reference to `batch_id`, an out-dir, or the plan hash. Only duplicate `(unit, trial)` pairs raise, in the answer loop of `analyze_rows`. Non-overlapping rows from two directories pass. Fix: record the run's `plan_sha256` on each answer row and require a single value in `analyze_rows`.

## What I checked and found sound

- **Missing-trial policy.** Primary predictions are complete-case; missing units are excluded and counted by type and band (`analyze65.py`, `analyze_rows` after line 240). The denominator is `2 x` registered units, and the collection is pinned at 500 (`steer65.py:115`), so it is 1,000. The stop is `missing * 20 > total` (`analyze65.py:261`), so 50 passes and 51 stops; the boundary tests pass. Absent pairs and unparseable answers both count as missing. Unregistered, duplicate and non-integer trial numbers raise.
- **Bounds** (`analyze65.py:178-220`). All 500 units are scored at 0.5 and at 0. The estimator matches the primary one (`bootstrap_difference` estimate is the mean difference). With no missing trials, `m5` gives 0.5 for the 0.5 bound, as expected.
- **Strict read.** `read_jsonl` raises on any malformed line (`analyze65.py:31-44`); a torn-line test is present.
- **Layers 0-2 sensitivity** is filtered to `layer > 2` (`analyze65.py:370-379`). It appears only in the result dict and the markdown (`457`, `584-587`). No prediction reads it, and a test asserts no `pass` key.
- **Chat mode.** `_chat` is the only `apply_chat_template` call (`steer65.py:270-282`). `_batch` feeds `generate_batch` (baseline, dose-zero, rollouts) and `lens_summary`, so every path gets `enable_thinking=False`.
- **Released chunk, same directory.** Returns `released` with no reserve and no POST (`judge65.py:616-617`). In `m2b`, the `--recover` re-invocation shows POSTs `[]` and exit 2.
- **Ledger release** refuses a non-Anthropic error body (`paid_ledger.py:515-523`). The chunk then goes to `unknown_no_batch` and is not resubmitted (`m2`, first attempt).
- **Plan to analysis contract.** The full plan's trials, through the real `_answer_rows` into the real `analyze_rows`: 1,000 trials, 500 units, 0 missing (`m5`).
- **Tests.** `python -m unittest test_part65` under a torch interpreter: 31 tests, OK, matching the commit's "31/31". The network test patches `urlopen`; the ledger tests use `FakeLedger` on temp paths. Nothing touched the live ledger or API.

## Not verified or not examined

- The real Qwen3-1.7B chat template is not on disk (only Qwen3.5-4B is cached). The `enable_thinking` test checks only that the kwarg is passed, through a stub tokenizer (`test_part65.py` around line 601). Whether the template emits the empty think block is unverified.
- Amendment 3 item 3 justifies the layers 0-2 sensitivity with Qwen3.5-4B evidence, while the steered model is Qwen3-1.7B (design lines 1 and 28). That is a rationale question, not a code defect.
- A restart between `_settle_saved` and `_persist_answers` (state `results_saved`) was not measured.
- The change adds no feature flags, so flag-off paths do not apply.
- The tests were run with `unittest` directly; I found no project test runner.
