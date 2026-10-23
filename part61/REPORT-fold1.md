# Part 61 round 1 fold report

## SUMMARY

- Implemented Part 61 Amendment 1 and the parent fold decisions before any model run.
- Replaced the old cross-model rule with independent per-model rank decisions at 1x and 2x.
- Corrected the 61d sign check, 61a calculation, no-floor threshold, scale validation, and rerun/resume behavior.
- Added the required input-dependent clamp test and source-copy sabotage probes.
- No model weights, network, API, or GPU were used.

## FINDINGS

| Review finding | Status | Resolution |
| --- | --- | --- |
| Luna #1; Haiku #1; disposed alert 0034: the old decision called artefact under the null about 72% of the time. | Fixed | Each model now uses target ranks at 1x and 2x. Excess is shown only at rank 1 at both sizes, absent at rank 3 or lower at 1x, otherwise inconclusive. An artefact requires both among-control 61a rho >= 0.7 and absent excess. Decisions are independent by model. The analyzer prints counts, ranks, permutation p-values, and the null outcome table. 61b and 61e include their 10/11 and 1/2 chance rates and are excluded from decisions. |
| Haiku #2; disposed alert 0035: the stub clamp test could not detect a dead clamp and had skipped without torch. | Fixed | The stub projection now depends on its input. The end-to-end test checks the expected downstream mean change, and a mutated copy of the real clamp assignment fails that same contract. The supplied PyTorch virtualenv ran the test; it did not skip. |
| Luna #2: a complete scan made a wrapper rerun unusable. | Fixed | A validated complete archive and manifest are returned for analysis-only reruns, without raising `FileExistsError` or loading model weights. The writer still resumes incomplete scans from unit/size checkpoints. |
| Luna #3: OLMo's saved activation maximum could disagree with its JSON/source record. | Fixed | Preflight now checks the source JSON/NPZ hashes, JSON/NPZ layer counts and widths, cumulative offsets, target coordinate, and target maximum. A synthetic mismatch in either the target value or source hash is rejected. The real local OLMo record passes and returns 6.5. |
| Luna #4; Haiku #3: averaging the two 61d sign ratios could hide opposing failures. | Fixed | The band is applied to both sign values separately for every unit at 1x; the report retains each value. A 0.8/1.2 pair fails. |
| Haiku #4: 61a did not have a test for size/count pairing. | Fixed | The production helper pools the ten controls only. Tests pin rho=1.0 for a rising pattern and rho=0.4 for a known non-monotone pattern; a copied `tile`-to-`repeat` sabotage goes red. |
| Haiku #5; papercut 0036: the 0.00005 null probability was implemented as an absolute-change floor. | Fixed | The build now matches 59B's pooled P99 with no absolute floor. The null pass probability remains recorded as 0.00005. A low-value synthetic responder passes without the old floor, and a sabotaged floor mutation fails. |
| Haiku #6: the Qwen branch of the old combined rule was not tested. | Fixed | The old joint branch was removed. The same per-model decision function is applied to both archives; unit and paired-analysis tests cover separate model decisions. |
| Haiku #7: residual ratios were only reported at 1x. | Fixed | The analyzer JSON and console output include both signs for every unit and all four sizes. |
| Haiku #8: resumed runs lost peak-memory bookkeeping. | Fixed | Baseline and condition statistics are carried forward with existing checkpoints. The interrupted-run test confirms previously completed statistics remain in the completed manifest. |
| Haiku #9: OLMo's target scale record was generated in bf16 although inference is fp32. | Fixed | The manifest and analysis report identify the source scale dtype as bf16; forward passes remain fp32. |
| Haiku #10: 200-document counts could be read against 59B's 1,000-document counts. | Fixed | The manifest and analysis state that these counts are not comparable and limit analysis claims to within-run comparisons. |
| Haiku #11: a transient Hugging Face fetch failure skipped a model. | Fixed | Each pinned model fetch gets up to three attempts. A shell-stub test fails twice, succeeds on attempt three, and verifies the rest of the runner proceeds. |

No finding was marked wrong or deferred. The papercut about the absolute floor was also resolved under the parent's explicit no-floor choice.

## WHAT THE SUITE PROVES

Ran `/home/elisepc/neuron-probe/.venv/bin/python -m unittest part61.test_part61 -v` from the assistant-basin root: **16 tests passed, none skipped**. The suite covers the amended rank outcomes and chance rates, the 20,000-draw null simulation with ten controls, OLMo record binding, 61a pairing, sign-specific 61d, all-size residual output, complete-scan reuse, checkpoint resume and statistic preservation, the clamp writer/reader path, and shell retry/failure handling. Seven sabotage probes mutate copies of the real classifier, size-pairing, no-floor, clamp-scaling, or clamp-hook code; each went red.

In the specified null simulation the superseded rule labeled 72.125% of draws “artefact”. The amended OLMo rank-1-at-1x-and-2x rule labeled 0.640% as shown, below the 1/11 (9.091%) permutation bound. These are synthetic calibration figures only; they are not experimental Part 61 results. `bash -n part61/run61.sh` and `git diff --check -- part61` passed. The real local OLMo records validated against one another and the Part 61 source files.

## EXTERNAL ASSUMPTIONS STILL UNVERIFIED

- The pinned Qwen and OLMo snapshots expose the expected decoder blocks and `mlp.down_proj` layout, with compatible pre-hook arguments.
- Both pinned tokenizers are fast and accept `return_offsets_mapping=True` on the rented image.
- The installed Transformers and flash-linear-attention versions load both pinned models in fp32 as requested.
- Hugging Face access succeeds after the bounded retry, and the runner's package installation succeeds on the rental image.
- Real-model VRAM use, throughput, residual effects, activation counts, and 61a-61e outcomes.

## WHAT COULD NOT BE CHECKED

No external fetch or model load was run, and no rented GPU was used. The tests exercise local functions and a CPU PyTorch stub, not either real architecture. Therefore no prediction is scored and the actual model-hook, tokenizer, package, memory, and timing assumptions remain unverified. The report does not treat synthetic test counts as model findings.

## BLOCKERS

None for the requested fold and code delivery. A model run is still required before reporting experimental outcomes.
