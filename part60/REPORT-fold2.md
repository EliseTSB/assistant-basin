# Part 60 round-2 fold report

## SUMMARY

- Folded the round-2 Luna and Haiku findings into the Part 60 runner, scorer, analyzer, wrapper, definitions, runbook, and offline tests.
- Applied parent-control seq 1, Amendment 3 items 1–6; items 7–9 were outside this assignment. The M6 change is recorded below as Amendment 4 text.
- Verification: 62 tests passed; Python compilation, `bash -n run60.sh`, and `git diff --check` passed. Three copied-module production mutations failed as expected.
- No Anthropic API request, Qwen load, GPU run, commit, or push was made.

## FINDINGS

### Round-2 review findings

| Finding | Disposition and evidence |
|---|---|
| B/Luna 1 — 60j M1 self-comparison | Fixed. An M1 test candidate compares only against M0; the evidence states the sole comparator. `test_m1_candidate_is_compared_to_m0_without_self_comparison` covers it. |
| B/Luna 2 — M6 smoke and errors | Fixed under the authorized M6 spec: five dev requests, adaptive thinking, high effort, 8,000 tokens, full schema/stop/thinking gate, retryable transient errors, and no one-request batch drop. See H2–H3 and M4 below. |
| B/Luna 3 — spend reservation and ambiguous POST recovery | Fixed. Reservations are durable before POST; batch creation sends only `requests`; reconciliation pages with `after_id` and adopts only one timestamp/request-count match. Tests cover pagination, no metadata, and the locked zero-match path. |
| B/Luna 4 — second-call rejection evidence | Fixed. M3 selection and M5 round-one evidence and usage remain recorded when the follow-up request is rejected; rejection is stored on that phase. Covered by `test_second_call_rejection_preserves_first_round_evidence`. |
| B/Luna 5 — stale return archive and pip freeze | Fixed. The wrapper removes prior success archives before reruns; pip-freeze failure uses an advisory version fallback. Wrapper tests cover both. |
| B/Luna 6 — ineffective tests and overclaimed sabotage | Fixed. Tests now mutate copies of the production threshold, winner-selection, and SID-key code and prove each mutation fails. An NPZ generation-to-analysis round trip exercises production readers and aggregation. |
| B/Luna 7 — staging paths | Fixed. The runbook extracts to a staging directory and copies only `results/part60`; generator/scorer path errors name the `PART60_BASIN` remedy. |
| H1 — unsupported top-level batch metadata / pagination | Fixed. No top-level metadata is sent. Reconciliation stores `posting_started_at` and request count, walks list pages with `after_id`, and requires exactly one batch whose `created_at` and `request-count sum` match. The fake list test contains no metadata. Live cursor behavior remains unverified pending the runbook probe. |
| H2 — transient M6 smoke errors dropped the method | Fixed. Only HTTP 400 or a successful response with no non-empty thinking marks smoke-dropped state. Timeouts, 429, 5xx, missing key, truncation, and invalid schema write `error`; a later pass clears a smoke-origin drop. Tests cover each path. |
| H3 — M6 effort, budget, gate, and truncation | Fixed in code and docs. M6 uses `output_config: {effort: "high"}` and 8,000 tokens; all five smoke responses must contain thinking, avoid `max_tokens`, and parse. Collection reports the thinking fraction and leaves truncated chunks resubmittable. Haiku 5.5 acceptance is unverified live. |
| H4 — registered test split membership | Fixed per the explicit decision. Test real units are the 247 IDs in `units.test.json`; L0:u190, L4:u741, L17:u1360 are excluded; controls remain 250. The registered-membership test checks both counts when Part 58 inputs exist. |
| M1 — phase failure denominator and missing records | Fixed. Primary phases use all expected units; M3 fresh and M5 round 2 use parsed predecessors; once a phase starts, an eligible absent record counts as failure. Tests cover missing primary, fresh, and round-two records. |
| M2 — self-fulfilling tests and incomplete fixtures | Fixed. The M3/M5 phase fixture includes successful predecessor data and distinct good/bad follow-up phases. Threshold sampling now has a seeded subset case where pool size is smaller than source rows. Three copied-module mutations go red. |
| M3 — reproduction and M0 device mismatch | Fixed in implementation; live behavior unverified. The reproduction gate requires median ratio 0.9–1.1 and Spearman at least 0.98 overall, neuron, and control. M0 is rescored on the same scorer/device as other methods; `M0-archive.npz` is a cross-check. A toy-scoring test covers the gate; no model was loaded. |
| M4 — any M6 batch error caused a permanent drop | Fixed. M6 batch errors follow the registered greater-than-10% phase failure rule rather than dropping on one error. Covered by `test_m6_one_batch_error_does_not_drop_below_ten_percent`. |
| M5 — old fold report overstated SID normalization and sabotage | Corrected here. `load_acts` stringifies and validates NPZ keys; it does not perform a separate canonical SID normalization. The production writer/reader use the same SID format, verified by round trip and a key-format mutation. The older `REPORT-fold.md` claim of canonical normalization and three valid sabotage tests should not be relied on. |
| L1 — shell test gate grouping/readiness | Fixed. The grouped condition requires a ready status, a nonempty allowed method list, and a matching candidate; pending and stale markers are rejected. A valid third-method candidate is accepted by wrapper tests. |
| L2 — blinding test tautologies | Fixed. Tests inspect actual request bodies for M1 primary, M3 selection, and M5 round one and assert that tier/rank/rating labels are absent. |
| L3 — supplemental thresholds lost on rewrite | Fixed. Fingerprint refreshes retain supplemental thresholds, observed ceilings, and scoring metadata; reports print `threshold_missing`. A regression test forces a fingerprint change. |
| L4 — misleading zero exit for pending or locked work | Fixed. Collection-not-ready and zero/multiple reconciliation matches raise `RunnerError`; CLI handling returns nonzero. Tests cover the pending batch and locked zero-match path. |
| L5 — metric wording and denominator clarity | Fixed. Reports show positive-only and full-rule rates, the marked-comparable denominator for 60h, the secondary 60d denominator, and the single-candidate 60j caveat. Dev advancement and 60j require five-point leads on positive-only and full-rule with bootstrap lower bounds at least five points. |
| L6 — missing control H_none | Fixed. The analysis table prints control H_none alongside real H_none. |
| L7 — implicit mitmproxy trust | Fixed. The mitmproxy path is removed. The system trust store is default; a custom CA is opt-in through `PART60_CA_BUNDLE` (or the standard `SSL_CERT_FILE`). The runbook documents the Part 60 variable. |
| L8 — M6 smoke spend omitted | Fixed. The five-call smoke reserves before requests and settles/releases using returned usage; smoke cost is included in M6 cost reporting. |
| L9 — inherited price constants | Documented, not changed. `$0.05/$0.25` per million tokens are inherited Part 58 constants; current Haiku 5.5 prices remain unverified and are stated as such in the definitions/runbook and assumptions list. |
| L10 — return extraction polluted basin root | Fixed. Extraction is staged; only the archive's `results/part60` subtree is copied into the basin. |
| L11 — stub could report success by accident | Fixed. Stub mode requires both `PART60_STUB=1` and `PART60_ALLOW_STUB=1`; tests cover authorized and refused cases. |

### Other A/C high-severity class checks

| Finding | Disposition and evidence |
|---|---|
| A/Luna 1 — collect before asynchronous completion | Fixed. Collection polls `processing_status` and fetches results only after `ended`; pending status raises nonzero. |
| A/Luna 2 — Qwen model/tensors on CPU | Checked statically. Model, inputs, and hook/control tensors use the selected device. No model load or GPU numerical check was permitted. |
| A/Luna 3 — archive path mismatch | Fixed/documented. Runbook archive paths match the wrapper; staging extraction copies only `results/part60`. |
| A/Luna 4 — RETURN_READY despite missing work | Fixed. Requested phases must score and verify with matching saved score metadata; dev verification gates test scoring. Wrapper tests cover successful and failing gates. |
| A/Luna 5 — retry/resume can double-submit | Fixed. Only bounded HTTP 429 retries reuse the same request chunk; ambiguous outcomes stay reserved and locked for exact reconciliation. Changed plan fingerprints and overlapping unit plans are refused. |
| A/Luna 6 — stale activation reuse | Fixed. Score fingerprints include the input texts/SIDs, unit keys, model revision, dtype, device, backend, and batch size. |
| A/Haiku 1 — M6 thinking behavior | Fixed in protocol and tests; live Haiku 5.5 output remains unverified. See H2–H3. |
| A/Haiku 2 — retrying ambiguous POSTs | Fixed. Unknown outcomes do not get blind retries; the runner reconciles first. HTTP 429 retries are bounded and reuse the same body. |
| A/Haiku 3 — changed arguments resubmit overlapping units | Fixed. Phase plan fingerprints and saved unit plans block changed or overlapping submissions. |
| C/Luna 1 and C/Haiku 1 — activation key join | Wrong as a code defect in this checkout. Production score output and analyzer use the same string SIDs/unit keys, with duplicate and shape validation. Round-trip and deliberate key-format mutation tests exercise the join. No extra canonicalization is claimed. |
| C/Luna 2 — model snapshot path | Fixed. The resolved snapshot path is recorded and passed into scoring/verification. |
| C/Luna 3 — M3/M5 phase routing | Fixed. One phase map serves scoring and analysis; tests assert M3 selection/fresh and M5 round-one/round-two paths. |
| C/Luna 4 — M4 marker alignment | Fixed and fixture-tested. Marker removal, offsets, token overlap, and same-four-sentence comparators align; missing marked values leave both rates unscored. Tokenizer/model numerics remain unverified live. |
| C/Haiku 2 — hook tensor/index namespaces | Checked statically; device-local tensors and explicit output columns are used. No model load was performed. |
| C/Haiku 3 — ignored `--limit` | Absent; the CLI has no `--limit` option or corresponding path. |
| C/Haiku 4 — M0 marked strings | Fixed. M0 text is joined to clean Part 58 baseline sentences before scoring. |
| C/Haiku 5 — superseded M6 spec | Fixed to the currently authorized M6 smoke, output, and budget specification. |
| C/Haiku 6 — second-call errors and 10% rule | Fixed. First-round evidence is retained; errors and missing eligible phase records count under the registered phase denominator. |
| C/Haiku 7 — reserve double-counting/cross-split cost | Fixed. Chunk settlements replace their own reserves; analysis totals cost for the requested split/method and includes smoke usage. |
| C/Haiku 8 — 60c mismatched units | Fixed. M1 and M0 strong-tier comparison uses the intersection of valid real units and reports its paired count. |
| C/Haiku 9 — M5 round-two feedback | Fixed and tested. Request includes the prior hypothesis and boolean fired outcomes, not activation values. |
| C/Haiku 10 — missing production round trip | Fixed. Generated output, score-format NPZ, `load_acts`, and `aggregate_method` are exercised together. |
| C/Haiku 11 — M4 marked/full text mismatch | Fixed. Both are calculated on the same four clean sentences; test covers missing marked activation behavior too. |

### Papercuts

| Alert | Disposition |
|---|---|
| alert-203608-8pixy-0010 | Fixed with the M6 adaptive/high-effort, 8,000-token, five-unit smoke and per-batch thinking report. |
| alert-203839-8pixy-0012 | Fixed with explicit M7 skip, advisory pip-freeze fallback, and wrapper coverage. |
| alert-203841-8pixy-0013 | Fixed with retryable smoke errors, spend accounting, the five-unit smoke, and the batch-level 10% rule. |

### Parent-control message

Seq 1 (Astra Amendment 3 analysis additions) was applied and disposed as `considered`. Items 1–6 were implemented; items 7–9 were omitted as explicitly directed. No messages were declined or deferred, and the final mailbox read was empty. No design blocker beyond the authorized amendments was found.

## EXTERNAL ANTHROPIC API ASSUMPTIONS

The API fields and behaviors below are assumptions made by the code. “Documented” means the listed API page describes the field or behavior; it does not mean it was exercised live in this run. The live one-request probe remains required by the runbook before production submissions.

| Assumption in code | Status and source |
|---|---|
| Create a batch with `POST /v1/messages/batches`, a JSON body whose top-level field is `requests`; each item contains `custom_id` and `params`. No top-level `metadata` is sent. | DOCUMENTED by [creating message batches](https://docs.anthropic.com/en/api/creating-message-batches); the reviewer confirms only `requests` is top-level. |
| Message `params` use `model`, `max_tokens`, `system`, and `messages`; API key/version/content-type headers are accepted. | DOCUMENTED by [Messages](https://docs.anthropic.com/en/api/messages) and [API getting started](https://docs.anthropic.com/en/api/getting-started). Exact `claude-haiku-5-5` availability is not live-checked. |
| M6 accepts `thinking: {type: "adaptive"}` with `output_config: {effort: "high"}` and `max_tokens: 8000`, including through the batch API. | UNVERIFIED for this model/API combination; see [extended thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking). The five-request smoke is implemented but was not run. |
| Successful batch creation returns an `id`; retrieval by ID exposes `processing_status`, and `ended` means results can be fetched. | DOCUMENTED by [creating message batches](https://docs.anthropic.com/en/api/creating-message-batches) and [retrieving a message batch](https://docs.anthropic.com/en/api/retrieving-message-batches). |
| List results expose `data`, `has_more`, batch `id`, `created_at`, and `request_counts`; `limit=100` is accepted. Request-count status values are integers whose sum is the total request count. | DOCUMENTED by [listing message batches](https://docs.anthropic.com/en/api/list-message-batches). Actual account behavior was not probed. |
| The batch list is newest-first and `after_id=<last_id>` walks to the older page. | DOCUMENTED by the reviewer’s batch-list check; live ordering/cursor behavior remains a probe item. The implementation scans pages until exhaustion, so it does not stop on a presumed ordering. |
| `created_at` is a parseable UTC timestamp comparable to local `posting_started_at`. | Field is documented by batch-list/retrieval pages; local clock synchronization and cross-system timestamp boundary are UNVERIFIED. A false mismatch can leave a paid batch locked. |
| A completed results endpoint returns newline-delimited JSON keyed by `custom_id`, with a result type and, on success, `message.content`, `usage`, and `stop_reason`. Content can contain `text` and `thinking` blocks. | DOCUMENTED by [retrieving batch results](https://docs.anthropic.com/en/api/retrieving-message-batches-results), [Messages](https://docs.anthropic.com/en/api/messages), and [extended thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking). Exact Haiku 5.5 adaptive-thinking responses are UNVERIFIED. |
| HTTP errors include status codes; JSON error data may include `error.type`. HTTP 400 rejects M6 at smoke; HTTP 429 is retried up to four attempts with the same chunk. | Status/error categories are DOCUMENTED by [API errors](https://docs.anthropic.com/en/api/errors). The assumption that a received 429 guarantees no batch was created is UNVERIFIED and should be confirmed by the manual probe. Timeout/server errors are treated as ambiguous and never blindly retried. |
| The batch API accepts at most 10,000 requests; this runner also enforces a 190 MB serialized-request local cap. | 10,000-request ceiling is DOCUMENTED by [creating message batches](https://docs.anthropic.com/en/api/creating-message-batches). The 190 MB cap is a local safety choice; its margin against current service payload limits was not checked. |
| Current Haiku 5.5 prices equal inherited Part 58 rates of $0.05/$0.25 per million tokens. | UNVERIFIED. These are inherited constants, not a live price lookup. |

### Probe-batch checklist for the operator

Before the first dev submission, use the manual one-request probe described in `RUNBOOK-60.md`: save the create response ID, compare `created_at` and `request_counts` in the list response, inspect list order, and verify that `after_id` advances to an older page. Confirm the API returns the expected fields and that the timestamp comparison is viable. Before M6 submission, separately run the implemented five-unit smoke and inspect its five thinking/stop/schema outcomes. This worker made neither API call.

## AMENDMENT 4 TEXT

M6 uses `output_config: {effort: "high"}` and `max_tokens: 8000`. Its dev smoke uses five units and passes only if all five responses contain non-empty thinking, stop for a reason other than `max_tokens`, and parse as the M6 schema. HTTP 400 or a successful response with no thinking drops M6; other smoke errors write `error`, leave method state untouched, and can be retried. A later passing smoke clears smoke-origin dropped state. Collection reports the per-batch fraction of M6 results with thinking. A truncated chunk is not complete and can be resubmitted.

## TESTS RUN AND LIMITS

- `/home/elisepc/neuron-probe/.venv/bin/python -m unittest test_part60`: 62 tests passed. The registered test-split membership test ran against the available Part 58 files; without them that one test skips with a specific message and all other tests use a synthetic pilot.
- The production-mutation tests are `test_threshold_owner_row_mutation_in_module_copy_goes_red`, `test_m3_winner_mutation_in_runner_copy_goes_red`, and `test_sid_key_format_mutation_in_analyzer_copy_goes_red`; each asserts the mutated production copy exits nonzero.
- `/home/elisepc/neuron-probe/.venv/bin/python -m py_compile gen60.py score60.py analyze60.py test_part60.py`, `bash -n run60.sh`, and `git diff --check` passed.
- No generated experiment outputs were available for actual prediction scoring. There was no Anthropic call, Qwen model load, rental, or GPU validation. M6 behavior, API cursor/clock semantics, price values, tokenizer/model numerics, and live reproduction remain unverified.

## FILES TOUCHED

`gen60.py`, `score60.py`, `analyze60.py`, `test_part60.py`, `run60.sh`, `RUNBOOK-60.md`, and `DEFINITIONS-60.md`. `PART60-DESIGN.md` and the older `REPORT-fold.md` were not edited.

## BLOCKERS

None requiring a decision. The manual batch probe and live model/API checks are unperformed because the brief prohibited network/API and Qwen execution.
