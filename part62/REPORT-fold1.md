# Part 62 Fold 1 Report

## SUMMARY

Implemented the phase boundary amendment, per-cell resume checks, separated-dose brackets, prediction coverage, and safer blind submission bookkeeping. Folded the Haiku review findings into code, tests, and the runbook. The required CPU suite passes all 33 tests; five copy mutations go red. No network, API, model, or GPU operation was performed. The current Opus reader configuration cannot pass the $3 reserve cap, so submission is safely refused pending a project decision.

## FINDINGS

| # | Status | Disposition and evidence |
|---|---|---|
| 1 | Fixed | Part 62 wraps the real `NeuronHook`; prompt excludes the last prompt position and gen includes it plus decode positions. Real-hook stub test drives a three-position prefill and a one-position decode and checks each phase at both boundaries. |
| 2 | Fixed | `run62.py` resumes a matching running/incomplete manifest, verifies each cell fingerprint and hash, reconciles an atomically written cell, and refuses mismatched fingerprints instead of overwriting. |
| 3 | Fixed | Complete separation is reported as a bracket from the last dose below 10% to the first at or above 90%; the interval is used for strict 62c/62d thresholds. Separated fixtures cover held and failed outcomes. |
| 4 | Fixed | Production `_run_model` calls the tested `cell_parameters` helper for condition seeds and target/control set values. A fake-model writer test exercises that production path and resume. |
| 5 | Fixed | Phase tests instantiate `neuron_report.NeuronHook` with a fake projection and CPU stub model; the shared library remains unedited. |
| 6 | Fixed | Synthetic midpoint and alien-frame conversion assertions pin the 0.8 midpoint and `0.8 × 19.125 / 24.69` conversion. |
| 7 | Fixed | Earth proxy uses the registered `\bearth` prefix pattern; the test covers “Earthlings”. |
| 8 | Fixed | OFFWORLD is labeled as a known-noisy proxy in analysis output and the runbook. |
| 9 | Fixed | The run manifest and runbook identify the OLMo Pile maximum as sourced from a bf16 record while generation is fp32; no false fp32 anchor claim remains. |
| 10 | Fixed | Qwen's measured alien-frame maximum is checked against 24.69 with a 0.1% relative tolerance; out-of-range values fail. Tests cover accepted and rejected values. |
| 11 | Deferred | The design's Reader and Cost paragraphs still disagree about Opus versus Haiku. Design files were explicitly out of scope. Code follows the Reader paragraph's Part 54 Opus choice; the runbook records its local cost estimate and the resulting cap block. |
| 12 | Fixed | The runbook specifies the 22-item Part 54 origin calibration set and the maximum of two errors from Part 59A. |
| 13 | Fixed | Analysis emits both registered decision-rule outcomes beside prediction statuses. |
| 14 | Fixed | The runbook documents sourcing `ANTHROPIC_CREDITS_KEY`, CA/output preflight, and the design's 2026-10-24 key expiry. Preflight checks happen before a spend reservation. |
| 15 | Deferred / unverified | The existing `anthropic-version: 2023-06-01` header was not validated. The brief prohibited network and API calls, so no live status request was made and the header was not guessed at. |
| 16 | Fixed | Output field renamed to `target_equivalent_set_value_norm`; runbook explains it is a requested set-value norm and not the activation delta. |

Papercuts 0030–0033 were addressed: 0030's Amendment 1 boundary was implemented, 0031 resume was added, 0032 gained complete prediction and production-writer checks, and 0033 gained key sourcing/preflight plus the cost/model discrepancy note. No parent-control messages arrived, so none changed the plan or were declined/deferred.

## WHAT THE SUITE PROVES

- `/home/elisepc/neuron-probe/.venv/bin/python -m py_compile run62.py analyze62.py blind62.py test_part62.py` completed successfully.
- `/home/elisepc/neuron-probe/.venv/bin/python -m unittest discover -s /home/elisepc/assistant-basin/part62 -p test_part62.py -v` passed 33 tests with torch available; no torch-dependent test was skipped.
- The tests exercise phase masks through the real `NeuronHook`, per-cell production writing and resume, fitted and bracketed analysis, all seven prediction held/failure paths, and pre-POST submission checks.
- Five copy-based mutations turned red: phase-position mask, 62c strict comparison, target/control scaling, logistic midpoint sign, and alien-frame conversion.
- Five selected regression tests run against the original `HEAD` implementation failed as expected (two assertion failures and three errors), including phase boundary, bracket evaluation, writer resume, and submission reserve estimation.
- The submitter persists attempt key maps and intent before POST, records an accepted batch id immediately, retains reservations for ambiguous outcomes, and never adopts/searches for an ambiguous batch. A 1,558-request Opus/1,024-token batch is refused before POST under the $3 cap using the local conservative rate estimate.

## EXTERNAL ASSUMPTIONS STILL UNVERIFIED

The Opus batch rates in the reserve estimate come from local notes, not a live pricing check. The live API acceptance of the existing version header, actual reader calibration/stop reasons, and any paid request were not checked. Checkpoint loading, runtime layer/width behavior, measured alien maxima on the rental GPU, fp32 memory fit, and the full 1,536-answer generation were also not run.

## WHAT COULD NOT BE CHECKED

The task explicitly prohibited network/API calls, model loading, and GPU use. Consequently no live status call or reader submission was possible. The current local Opus estimate exceeds the $3 cap on output allowance alone (about $15.95); a model or cap decision is needed before submission can proceed. No design file or shared library file was edited.

## BLOCKERS

Before a reader batch can be submitted, resolve the design's Opus-versus-Haiku reader conflict and authorize a configuration that fits the $3 hard cap, or set a different cap. The current implementation fails closed before POST.

## Reader decision

Implemented the parent decision for Part 62's registered reader. Claude Haiku 5.5 (`claude-haiku-5-5`) is now the default, with a $4 hard cap. The blind gate now uses all 34 hand-labelled Part 59A calibration rows, mapping SCIFI to OFFWORLD, EARTH-REAL to EARTH, and FANTASY/OTHER to NEITHER while retaining OK/BROKEN. The more-than-two-error threshold remains hard, and truncation or unverified completion blocks scoring.

Claude Opus 5.5 is an explicit fallback only: `blind62.py submit --reader opus` refuses unless a failed Haiku calibration record exists. Haiku and Opus use separate $4 and $8 caps and reader-specific ledgers. Every batch, label record, analysis result, and printed status/tally names the reader. Each paid attempt receives a random attempt id before POST, and its reserve is released only for a 4xx response with a valid Anthropic error body. The reserve uses 150 expected output tokens per request; response usage above `max_tokens` is classified as truncated.

Updated RUNBOOK-62.md with the reader decision, gate, command, caps, reserve rule, and attempt handling. The Part 62 CPU suite now has 35 tests; all 35 passed, including default reader/model selection, fallback refusal, independent ledgers/caps, expected-output reserve calculation, and over-limit truncation detection. Existing mutation checks also remained red against their deliberate mutations. No live API call was made.

No parent-control messages arrived during this fold; none changed the plan or were declined/deferred.
