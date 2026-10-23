# Part 62 fold 2 report

## SUMMARY

Round 2 paid-reader concerns and the fold continuation were completed within `part62/`. The paid reader now routes through `paidaction`'s shared ledger with sequential worst-case chunk reservations and actual-usage settlement; raw provider rows are durably retained under stable custom IDs. The calibration uses 120 balanced Part 54a origin answers. Part 62 generation now records paired sample seeds for `both`, `gen`, and `prompt` at each dose, and 62a reports the paired count difference and a conservative exact-binomial interval. The Part 62 CPU suite passes (34 tests, including five copy-based mutations that go red). No API request, GPU run, or model load was made during the Part 62 suite.

## FINDINGS

| Finding | Status | Evidence / disposition |
|---|---|---|
| R2 Luna HIGH 1; R2 Haiku HIGH 1; cross-reported Part 61 cap finding; spend-cap papercut | Fixed | Removed the local expected-token cost estimate and legacy per-review-dir attempt ledger. `submit()` defaults its ledger-path parameter to `paidaction.DEFAULT_LEDGER_PATH`, reserves in sequential chunks of 200 through `submit_in_chunks`, uses each request's full `max_tokens` worst case, and settles returned usage before the next chunk. The Haiku $4 kind sub-cap is enforced inside the shared $8 aggregate ledger. |
| R2 Haiku HIGH 2; cross-report and shared-cap papercuts | Fixed | All readers/review dirs use the library default ledger path. A separate `ledger_path` parameter is available for tests; the test uses an absolute path under `part62/.test-ledgers/`, and the real `Ledger` accepted, reserved, and settled it. |
| R2 Haiku HIGH 3, wrong calibration rubric | Fixed | Calibration draws 120 seeded, balanced, already-labelled Part 54a origin answers (40 per class), records seed and IDs, and gates at Cohen's kappa >= 0.8 with no missing/truncated calibration responses. One disagreement is permitted when kappa remains above threshold; the gate does not impose an unregistered zero-error condition. |
| R2 Haiku MEDIUM 4, missing prediction boundaries | Fixed | Added boundary fixtures for 62a, 62b, 62e, 62f, and 62g; existing 62c/62d tests cover strict boundaries. The 62a fixture verifies the registered three-count boundary and fails at four. |
| R2 Haiku MEDIUM 5 and R1 non-estimable analysis finding | Fixed | Incomplete evidence is `NOT-EVALUABLE`. 62g evaluates a complete-separation midpoint interval conservatively. 62d requires a finite logistic fit; a bracket width is only an upper bound and cannot establish width > 0.8. |
| R2 Haiku MEDIUM 6 and cross-report paid POST durability finding | Fixed in implementation; failure window untested | The library durably records each batch ID immediately after ID extraction and before metadata writes, polling, or result collection. Per-batch maps/metadata and raw JSONL rows are fsynced. Ambiguous reservations remain active. No real POST or kill/disk-failure sweep was run. |
| R2 Haiku MEDIUM 7, output headroom | Fixed | Calibration completes and its observed maximum output usage determines answer-request `max_tokens` before answer submissions. Each answer chunk is then reserved at that maximum. |
| R2 Haiku MEDIUM 8, 62a sampling noise | Fixed by Amendment 3 | For each model/dose/question/sample, the `both`, `gen`, and `prompt` cells share one SHA-derived seed; seeds vary across doses. Samples generate one at a time, and the per-sample seed list is stored in each cell and validated by the reader. The analyzer reports signed paired difference, exact count difference, and a Bonferroni-conservative 95% interval derived from exact binomial intervals. 62a is labeled “HELD WITHIN SAMPLING NOISE” only when the registered within-3-of-24 condition holds for all registered pairs. |
| R2 Luna MEDIUM 2; R2 Haiku MEDIUM 9, design/runbook registration split | Runbook/code aligned; design amendment parent-owned | The runbook describes the Haiku default, explicit Opus fallback, shared cap, and Part 54a origin calibration. The design file was left untouched as explicitly directed; its dated amendment is parent-owned. |
| R2 Haiku LOW 10, bracket use for 62d | Fixed | 62d no longer treats a bracket's upper-bound width as evidence that true width exceeds 0.8; bracketed 62d is `NOT-EVALUABLE`. |
| R2 Haiku LOW 11, control unit derivation | Fixed | `_run_model` re-derives and checks the registered same-layer control unit using the fixed selection seed. |
| R2 Haiku LOW 12; stale-count papercut | Fixed in current report/runbook | The current request count is 1,656: 1,536 experiment answers plus 120 calibration rows. Historical fold 1 remains unchanged. |
| R2 Haiku LOW 13, mutation temp directories | Fixed | Mutation tests use `TemporaryDirectory`; test ledgers use a temporary subdirectory under the assigned Part 62 domain and clean it in teardown. |
| R2 Haiku LOW 14 and R1 API-version concern | Unverified | Provider API version, current prices/rates, token estimate assumptions, live response semantics, and real billing remain unverified without network/API calls. |
| Other round 1 findings listed as fixed in the reviews | Retained | Existing implementation/tests retain the revised phase masks, manifest fingerprints/resume checks, control validation, complete-separation brackets, scale checks, analysis output labels, and preflight checks. No design files or `paidaction/` files were edited. |

## TESTS

- `/home/elisepc/neuron-probe/.venv/bin/python -m py_compile part62/run62.py part62/analyze62.py part62/blind62.py part62/test_part62.py`: passed.
- `/home/elisepc/neuron-probe/.venv/bin/python -m unittest part62.test_part62 -q` from the assistant-basin root: **34 passed**.
- Five copy-based mutations returned red for the 62c strict boundary, phase-position mask, control scaling, logistic midpoint sign, and alien-frame conversion. Boundary fixtures also cover 62a/62b/62e/62f/62g. The test suite exercises the real ledger at an explicit Part 62 path.
- The broader root `python -m unittest` command was started as required but interrupted after it began loading a local test model; the Part 62 suite above completed without model loads. No network, API, GPU, or paid operation was used.

## UNVERIFIED / WHAT COULD NOT BE CHECKED

- Live Haiku/Opus agreement, API headers and provider response shape, current rates, billing after timeouts/429/5xx, the input-bytes/3 token estimate, and actual tokenization remain unverified.
- The shared default ledger path and live submit callback were not exercised against the default path in this sandbox; the library API and same sibling-registry refusal semantics were verified with an explicit absolute ledger path inside `part62/`.
- No real model generation, process kill simulation for a paid POST, or operator batch recovery was run. Parent-owned design amendment remains outside this worker's file scope.

## BLOCKERS

None for this worker scope. The remaining design-file amendment is parent-owned per the brief.
