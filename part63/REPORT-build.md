# Part 63 build report

## SUMMARY

Built the Part 63 gather, blind-reader, analysis, spend-ledger, runbook, and offline test files under `part63/`. The verified local corpus contains 4,944 answer records and the reader plan contains 4,978 requests including the 34-row calibration set. The conservative local reserve estimate is $1.836147 against the $3.00 cap. No real API, network, GPU, or model call was made.

## FINDINGS / WHAT SHIPPED

- `gather63.py` reads Part 54a, Part 54b, Part 26.8, and Part 59A. It validates saved Opus-key joins, keeps repeated source records, shuffles with a recorded seed, and writes an answers JSONL that contains only `id` and `text` plus a separate metadata key.
- Source row counts are 480 / 1,920 / 432 / 2,112, respectively. Existing Opus labels are attached for every row where the archived reader had a label. The selected sidecars contain 8 `null` labels in Part 54b and 2 in Part 26.8; Part 54a and 59A are complete. The Part 26.8 archived key omits its model field; its join was checked against that source's only model and all other key fields.
- `reader63.py` builds one Haiku 5.5 JSON-label request per answer, sends no condition or source metadata, requires `max_tokens=1024`, rejects `max_tokens` truncation, uses the 34 Part 59A hand-labelled examples as a scheme iii quality/BROKEN gate, and reserves the full output ceiling before POST. It settles on collected token usage, or leaves the full reserve charged when usage is incomplete.
- The calibration examples do not hand-label planet framing, maker identity, or the EARTH/OFFWORLD/NEITHER category. The gate therefore checks only their compatible `quality`/BROKEN labels; the runbook states the category limitation. The analyzer independently counts bad responses by scheme and refuses scoring above 2% in any one scheme.
- Maker names and maker kinds are read without the source model. The hidden model's own-lab aliases are applied only after parsing, using the side key. This permits the 63d comparison without revealing the source model in the request.
- `analyze63.py` prints per-cell counts for all schemes, keyword proxy views, and fixed 63a-63e held/failed results. The Part 54a Opus OFFWORLD reference counts in the frozen key are 0, 0, 1, 19, 24 for doses -1, +0.5, +1, +1.5, +2, matching the registered references. A successful analysis prints `RETURN_READY`.
- `RUNBOOK-63.md` documents commands, sidecar handling, cost accounting, calibration scope, and the API assumptions to check before the parent submits.

## FILES

- `gather63.py`, `reader63.py`, `analyze63.py`, `test_part63.py`, and `RUNBOOK-63.md`.
- Generated local corpus: `results/answers63.jsonl`, `results/key63.json`, and `results/manifest63.json`.
- Worker handoff report: `REPORT-build.md`.

## HOW TESTS WERE RUN

- Syntax check: `python3 -m py_compile gather63.py reader63.py analyze63.py test_part63.py` — passed.
- Offline suite: `python3 -m unittest -v test_part63` — 8 tests passed. The suite uses a fake API for submit/status/collect and checks calibration failure, truncation, per-scheme failure handling, the hard cap, prediction thresholds, and writer-to-reader key matching through files emitted by the real bundle writer.
- Four source-function sabotage runs went red as intended: `ignore-truncation`, `leak-condition`, `zero-reserve`, and `drop-side-key-row`. Each wrapper called the real implementation and then corrupted its result; the corresponding invariant test failed in the subprocess.
- Local gather succeeded and validated each source/Opus-key join. The offline request plan has 4,978 requests, `max_tokens=1024`, and a $1.836147 estimated reserve. These checks did not call the API.

## EXTERNAL ASSUMPTIONS

- **DOCUMENTED locally:** the 4 source directories, their `clamp_mult`/condition/answer fields, the Part 59A manifests, the archived Opus key files, the design's $3 ceiling, the 34-row Part 59A calibration list, and prediction thresholds. The code validated the local reader-key mappings before using existing labels.
- **DOCUMENTED locally, UNVERIFIED with the provider:** the Messages Batches endpoint paths and request fields follow the local Part 59/60 batch clients. The code expects batch status `ended`, NDJSON result lines, result `custom_id`, `result.type`, `result.message.content`, `message.stop_reason`, and `message.usage.input_tokens` / `output_tokens`. No live request was made to verify those fields.
- **UNVERIFIED:** `claude-haiku-5-5` availability, support for `thinking: {type: disabled}`, 1,024 output-token behavior, and the current API version's exact parameter compatibility.
- **DOCUMENTED locally, UNVERIFIED as current provider prices:** input/output estimates of $0.05/$0.25 per million tokens are copied from Part 60. The $1.836147 reserve uses UTF-8 request byte counts as an input-token upper estimate and the full `max_tokens` ceiling for every response. No tokenizer was loaded; the byte-to-token upper-bound assumption and live pricing were not independently verified.
- **DOCUMENTED in the standard library:** TLS requests use `urllib.request` and a CA bundle, and the spend lock uses `fcntl.flock`. The actual CA file and provider TLS behavior were not exercised.
- **UNVERIFIED:** the hard-coded model-to-own-lab aliases used after the blind read, including the model/lab relationship for each source model. This matters to the maker category and 63d.
- **DOCUMENTED from the cold review:** the off-world, sci-fi, and Earth keyword proxy expressions. The sanity gate's minimum one overlapping example is deliberately coarse and is not a calibration measure.

## WHAT COULD NOT BE CHECKED

The parent must submit, collect, and analyze the live batch. Therefore no Haiku labels, calibration result, actual spend, or 63a-63e experiment result exists yet. Provider pricing/API-field assumptions, live truncation behavior, TLS configuration, and model-to-lab aliases remain unverified. The Part 63 workflow has no rented GPU box or archive teardown step.
