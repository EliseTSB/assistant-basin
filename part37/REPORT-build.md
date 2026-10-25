# Part 37 build report

## SUMMARY

Built the 100-question set and the Part 37 collection, record, capture, stated-readout, analysis, and Windows runbook artifacts. The pilot stop gate follows Amendment 1 and analyzes both readout positions. No model was downloaded and no GPU work was run.

## FINDINGS

- Added `questions.json` with 35 opinions, 35 everyday decisions, and 30 explanations in q000–q099 order; the first 20 are the pilot.
- `collect37.py` implements pinned local-only Qwen3, OLMo Instruct, and SmolLM3 generation; identity screening; deterministic length-matched record and frame construction; one-forward-per-record float16 capture; and six stated prompts with swapped A/B mappings.
- `analyze37.py` implements question-disjoint five-fold probes, mean-log-prob regression, eight-component PLS subspace removal, matched-surprisal analysis, shuffled-label and random-direction controls, projections, stated summaries with token-mass flags, prediction verdicts, and the pilot gate.
- Added `test_part37.py` for question exclusions, prompts, answer-token sets, frames, record counts, truncation, screening, fold separation, scoring rules, and an optional tiny-model capture test. Tests were not run in this build worker.
- Added `RUNBOOK.md` for Windows staging, pilot and full steps, copy-back, and host analysis.
- The brief referenced `part65/RUNBOOK.md`, which is absent. I read the existing `part65/RUNBOOK-65.md` for runner conventions instead.
- Parent-control mailbox was empty at each checkpoint; no messages changed or deferred the plan.

## BLOCKERS

None for the build. The registered enacted/steering predictions remain HELD, and OLMo base prediction 37.7 remains NOT RUN, as required by Amendment 1. GPU collection and model-backed runtime behavior remain for the later Windows run.
