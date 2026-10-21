# Build brief: Part 43, singleton execution mode (Luna)

Run size, multiplied out: code and tests only here. The later validation pilot is about 2,000 single-sequence passes on
one rented 3090 (Claude decides and pre-registers the error budget separately; do not set one).

Why: the Part 43 pilot stopped at its own batched-vs-single check (results/part43/pilot-3090-fp32-2026-10-21/README.md).
Both reviewers (Alden, council part43-google-flip-review-2026-10-20 #7; Chorus, chorus-part43-review-2026-10-20 #7,
both in review/part43/REVIEW-singleton-alden-chorus.md) chose the same next step: an explicit bf16 SINGLETON execution mode used by every
confirmatory forward, with batching kept only for diagnostics. Build exactly that in part43.py.

Requirements:
1. One shared execution adapter (e.g. `execute(model, records, mode)`) used by EVERY call that scores or records model
   states: teacher_forced_scores, _margin_list, signs, clean and patched captures, lattice and nulls, rescue / receiver
   bypass / path checks, shift, baselines, grid, dose, swap, attach, readout / finite-difference / JVP paths. Grep for
   every direct `model(` and `.generate(` call (there are several, e.g. lines ~533, 1012, 1114, 1164, 1804, 2038, 2686,
   3660) and route each through the adapter or document why it is diagnostic-only. A `--exec-mode {singleton,batched}`
   flag, default singleton; the mode is written into every output file and the manifest.
2. Singleton: one record per forward, each paired ONLY with its own interventions; score each record immediately
   (no large padded logits tensor; sums of log probs need none); reassemble results in the original record order.
   Assert, between modes, equal prompt ids, answer positions, lengths, intervention payloads and unit row indexing.
   Clean trajectories used for patching are captured in the same mode and the same exact prompt shape.
3. A `validate` command on the PRODUCTION estimand (Google minus logsumexp of the Alibaba answers, full answer strings):
   the 7 full maker answers on the primary question; clean; one selected single zero; all-four zero; clean-to-clean
   restoration; +/- dose; one representative receiver rescue / path case; at least one short and one long answer.
   Repeat each cell twice in singleton (within-mode repeatability); replay natural signed activations at every real
   position and report the no-op score and activation difference; optionally bf16 vs fp32 on a few decisive cells.
   It REPORTS numbers (per-cell score, delta, max |difference|) and does not pass or fail them: the budget is Claude's.
4. A `batch-diag` command (kept separate, never used by confirmatory code): A alone; [A,A] no padding; [A,B] same
   length; [A,B] different length; reversed row order; masked right-pad ids randomized at the same length and mask;
   one extra-long companion. For each, the full-answer primary score, its intervention delta, and selected receivers'
   signed h values, not only logits.
5. Record the environment into outputs: torch, transformers and kernel (SDPA / FLA / causal-conv1d fallback) versions,
   actual parameter and activation dtypes, TF32 and reduced-precision flags.
6. Keep the original failed batched check and its receipts untouched; run_validation_checks may stay as a diagnostic.

Tests: tiny-model tests that singleton and batched give the same records in the same order; that every confirmatory
entry point goes through the adapter (a sabotage that routes one back to raw `model(` must fail a test); that validate
and batch-diag produce their fields. Keep every existing test green (`test_part43.py`). Run tests on CPU here; if torch
is missing in your environment, say so in BLOCKERS rather than reporting skipped tests as passing.

Commit on this branch (part43-singleton) when tests pass. Report REPORT-43-singleton.md ending with SUMMARY /
FINDINGS / BLOCKERS, listing every routed call site with its line and the exact A770/3090 commands for validate and
batch-diag.
