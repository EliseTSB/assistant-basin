# Part 63 round 2 fold report

## SUMMARY

- Folded both Part 63 round 2 reviews, the cross-part Part 63 findings, and the matching Part 63 papercuts.
- Applied the parent decisions for row multiplicity, distinct-text requests, the 100 repeat reads, Qwen3.5 own-lab aliases, the planet rubric, shared ledger path, shared $8 cap, and Part 63 $3 sub-cap.
- Added resumable `reader63.py run-all`: requests are sent in chunks of 200, each batch is polled every 30 seconds for up to 40 minutes, results are collected and settled before another chunk begins.
- A timeout leaves its reserve active; rerun resumes by saved batch ID. An unsettled or unknown Part 63 reserve blocks advancement.
- Offline suite: 19 tests passed; all 10 copied-code sabotage mutations went red. No paid request or network lookup was made.

## FINDINGS / WHAT SHIPPED

| Finding / review item | Disposition | Change and evidence |
|---|---|---|
| Retry submission after parent settlement can be stranded by a kill | Fixed in fold code | Parent collection attempts the failed-only retry before settling its own reserve; labels preserve retry state on parent recollection. Existing fake-transport test collects retry then recollects parent without a third POST. The new run-all recovery test covers a poll timeout followed by rerun and confirms the saved batch is collected before the next chunk. No process SIGKILL sweep was run. |
| Per-review ledger permits cap bypass | Fixed in fold code | `reader63.py` imports `paidaction.DEFAULT_LEDGER_PATH`, uses shared `$8` aggregate cap, tags reservations `part63`, and checks its `$3` sub-cap before each reserve. Test fixtures redirect this constant to a ledger file inside `part63/`; the mandated default path was not opened for writes in this worker sandbox. |
| Release on a 4xx without Anthropic error object | Fixed | HTTP status and response body are passed to `Ledger.release`; library validation refuses malformed bodies and 429, leaving the reservation active. Existing tests cover ambiguous outcomes and release behavior through the library-backed path. |
| Maker mapping counted Alibaba/Qwen as another lab | Fixed per parent instruction | Registered Qwen3.5 aliases are case-insensitive substrings: `alibaba`, `alibaba cloud`, `tongyi`, `tongyi lab`, `qwen team`, `通义`, `阿里`. Strict exact `Tongyi Lab` remains a sensitivity reading. Tests cover English and Chinese strings. |
| 4,944 corpus rows versus 3,677 distinct texts; repeated baseline rows | Fixed | Each distinct text is requested once, with a label fanned back to every corpus row. The 100 repeat reads sample distinct texts. Analysis keeps row multiplicity in primary tables and reports distinct-text sensitivity counts. |
| 63b/63c/63d denominator differs from registered “of 24” | Fixed per parent instruction | Primary baseline selections count all 24 registered rows. Distinct-text counts are sensitivity rows (19 for Qwen3-4B and 24 for Qwen3.5). |
| Planet rubric narrower than registered wording | Fixed | Prompt says “mentions a planet, a place of origin, or a world”; test checks the registered phrase. |
| Parent collect rerun clobbers retry labels | Fixed | Existing records and retry metadata are merged and retained; recollect does not submit another retry after a submitted or collected retry. Covered by the retry merge test. |
| Batch ID loss between POST response and sidecar | Fixed in fold code | Raw POST body is fsynced before parsing. Parsed batch ID is saved into submission metadata before the separate batch sidecar, and collection can recover it from submission metadata. Existing interruption test exercises sidecar failure and recovery. |
| Reserve estimate does not prove a hard $3 billing bound | Deferred with reason | The shared helper's input estimate remains ceil(UTF-8 request bytes/3), and inherited rates are unverified. Actual costs can exceed the estimate; offline code cannot establish provider tokenization or current billing rates. The caller enforces the requested reservation sub-cap, but this is not a guaranteed maximum charge. |
| 63c strict-rise and byte-token mutations were undetected | Fixed | Added copied-code sabotage mutations for dropping strict-rise behavior and changing `INPUT_BYTES_PER_TOKEN` from 3 to 6. Both targeted tests fail under mutation. |
| 429 retains a full reserve | Fixed / conservative behavior retained | Shared library refuses to release on 429. No live request was made. |
| Torn ledger tail | Fixed by shared library | Reader uses shared library ledger parsing, which reports and tolerates a torn final line. No separate Part 63 torn-tail mutation was added. |
| Concurrent submissions could mark live work interrupted | Fixed in code | `submit` and `run-all` take advisory run locks. The lock implementation was reviewed; no competing-process lock test was run. |
| Papercut 0046: release on any 4xx | Fixed as above. | |
| Papercut 0047: stale Opus/cap report | Fixed | Runbook identifies the Haiku reader and updates the command and cost assumptions. |
| Remaining matching papercuts and cross-part reviews | No additional Part 63 defect found | Reviewed all dated papercut files and the matching Part 61/62/63 review text; other matches concern Parts 60–62 or document paths outside this domain. |

### Async lifecycle shipped

`run-all` builds the deterministic request plan, persists a run plan, and submits chunks of 200 using Part 63's shared-ledger reserve/settle primitives. The synchronous `submit_in_chunks` callback cannot wait for an asynchronous provider batch, so the command implements the sequential loop directly. It stores the batch ID durably, polls until `processing_status == ended`, collects labels by custom ID, applies failed-only retry handling, and settles actual usage before moving on. A polling timeout exits nonzero with the active reservation retained. Rerunning the same plan processes the saved batch and refuses to advance when a Part 63 reservation remains reserved or unknown. `RUNBOOK-63.md` documents this command.

## TESTS

- Ran `/home/elisepc/neuron-probe/.venv/bin/python -m unittest` from `part63/`: **19 passed**.
- Ran `python3 -m py_compile reader63.py analyze63.py gather63.py test_part63.py`: passed.
- Ran `git diff --check`: passed.
- Ten real-function copied-code sabotage tests went red: truncation acceptance; condition leakage; zero reserve; dropped side-key row; wrong baseline selection; alias shortcut; missing temperature; substring proxy; removal of the 63c strict-rise behavior; and input-byte factor changed from 3 to 6.
- New fake-transport tests verify two sequential chunks with each prior chunk settled before the next POST; a non-ending chunk times out with its reserve retained; rerun collects the saved batch before starting the next chunk. Existing tests cover retry merging, pre-POST failures, ambiguous POSTs, and saved batch-ID sidecar recovery.

## UNVERIFIED ASSUMPTIONS

- Haiku 5.5 batch rates ($0.05/$0.25 per million tokens), API/model availability, batch fields, and provider billing behavior.
- UTF-8 serialized bytes divided by three is an upper input-token estimate. This is inherited and unverified.
- The parent-provided Qwen3.5 own-lab alias list has not been checked against a live authoritative provider source.
- Planet, origin, and maker schemes have no hand-labelled accuracy gate; repeat reads measure repeatability only.

## WHAT COULD NOT BE CHECKED

- The default shared ledger path is under `paidaction/`, outside the writable domain for this worker. Tests inject an explicit ledger path in the Part 63 test directory; no default-path write or production submit was attempted.
- No provider pricing check, live API request, model load, or paid charge was made.
- No subprocess SIGKILL-at-every-boundary matrix or competing-process run-lock test was run. Fake-transport tests exercise timeout recovery, retry merge, reserve retention, and saved-ID collection, but do not prove every process-kill boundary.
- The default 30-second poll interval and 40-minute bound were not exercised in wall-clock tests; unit tests use an immediate fake transport and short injected bound.

## BLOCKERS

No blocker to the code fold. The default shared ledger and live provider behavior could not be exercised under this worker's write and network constraints; no API call is authorized by the brief.

## SELF-ASSESSMENT

The code follows the parent-defined lifecycle and passed its offline suite. The principal residual uncertainty is spend safety: rates and the byte/token estimate are assumptions, so the $3 sub-cap is a reserve policy rather than a proven cap on actual provider charges. Run-all recovery is covered with fake transport but not with process-kill or multiprocess fault injection.
