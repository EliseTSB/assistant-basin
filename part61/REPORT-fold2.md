# Part 61 round 2 fold report

## SUMMARY

- Folded the round 2 Part 61 review and the Part 61 papercut into the analyzer, tests, and runbook.
- 61a now scores Spearman rho across all eleven units, matching the registered wording; the among-controls rho is shown as a sensitivity value.
- A push-size artefact requires both rho values to be at least 0.7 and the amended excess status to be absent.
- Strengthened the clamp integration contract with unequal, multi-token documents and next-layer residual measurements.
- All 16 prescribed tests passed; all 11 source-copy mutation probes went red.
- No model weights, external services, APIs, or GPU were used.

## FINDINGS

| Finding | Status | Resolution / evidence |
| --- | --- | --- |
| Part 61 papercut 000840: registered 61a wording says “across the 11 units,” while the score used only ten controls. | Fixed | `analyze61.py` now computes and reports the all-eleven primary rho, and separately reports the control-only sensitivity rho. `RUNBOOK-61.md` documents both and the conservative artefact gate. A fixture where the target trend opposes the controls checks that the two values differ; an all-eleven size-pairing mutation goes red. |
| Haiku round 2 finding 5: one-token clamp stub did not detect dropped mean denominator or residual mask, nor inverted residual ratio. | Fixed | The clamp test and three source-copy sabotage contracts use 2- and 3-token documents with padding and check downstream means and per-document residual norms. Inverted residual ratio, dropped mean denominator, and dropped residual mask mutations all fail. |
| Haiku round 2 finding 7: null simulation reimplemented the excess decision. | Fixed | The simulation now sends sampled ranks through production `excess_status` before checking the null shown rate. |
| Haiku round 2: Qwen decision branch only partly covered in the combined path. | Fixed | The writer/reader test asserts that the combined report's Qwen and OLMo decisions equal their independently generated model decisions and verifies both sensitivity rho values are retained. |
| Haiku round 2 finding 8: OLMo source scale 6.5 came from a bf16 record. | Wrong as a new defect | This is explicitly disclosed: the run manifest and analysis retain `target_pile_maximum_source_dtype="bf16"`, while forward passes use fp32; REPORT-fold1 records the same distinction. No change was needed. |
| Other round 1 Part 61 findings. | Already fixed | The round 2 Luna review confirms the reviewed round 1 issues are fixed. Existing tests cover control selection, archive reuse/resume, scale binding, sign-specific residual ratios, and fetch retries. |
| Part 61 findings in Part 62/63 round 2 reviews. | None found | The cross-review hits concern Part 62/63 paid readers and design records, not Part 61 implementation. No out-of-scope fixes were made. |

No design file was edited; the parent decision supplied in the fold brief was implemented in the analyzer and runbook. No findings were deferred.

## TESTS

- Ran `/home/elisepc/neuron-probe/.venv/bin/python -m unittest part61.test_part61 -v` from `/home/elisepc/assistant-basin`: **16 passed, 0 skipped**.
- Source-copy sabotage probes: **11 red**, including all-eleven 61a pairing, inverted residual ratio, dropped mean denominator, and dropped residual padding mask.
- `git diff --check -- part61`: passed.

## UNVERIFIED ASSUMPTIONS

- Pinned Qwen and OLMo snapshots expose the expected decoder blocks and down-projection layout, and their pre-hooks accept the runtime arguments.
- Both tokenizers support the expected fast-tokenizer offsets; pinned model and package loading succeeds on the rented image.
- Real-model VRAM, throughput, residual effects, activation counts, and 61a–61e outcomes. No model run was performed.

## WHAT COULD NOT BE CHECKED

No external model fetch, model load, GPU run, provider request, or API check was performed. The CPU stub validates local contracts only; it does not establish architecture compatibility or experimental outcomes.

## BLOCKERS

None for this fold. Parent-control mailbox was empty at every pull; no messages changed, were declined, or were deferred.
