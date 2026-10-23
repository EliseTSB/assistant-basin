# Part 63 runbook

Part 63 gathers the frozen answers, submits blind Haiku 5.5 batches, and analyzes the returned labels. The builder must not make an API request. API use is opt-in through `reader63.py` commands; collection may submit one failed-only retry batch.

## Build the blind corpus

From the basin checkout, run:

```sh
python3 part63/gather63.py
```

The default input is `results/` beside `part63/`. The command requires the complete Part 59A manifests and the archived Opus key/label files for Part 54a, Part 54b, Part 26.8, and Part 59A. It refuses to join a label when the source metadata does not match the saved reader key. It writes:

- `part63/results/answers63.jsonl`: shuffled rows containing only `id` and answer `text`.
- `part63/results/key63.json`: the side key with model, unit, dose, question, source, condition, and any existing Opus label.
- `part63/results/manifest63.json`: answer count, per-source counts, the recorded shuffle seed, and an answer-file hash.

The shuffle seed defaults to 6301. Use `--seed N` to create a different order; the chosen value is always recorded. Do not send `key63.json` or the submission key sidecars to the reader. The request body is built from answer text alone, with sequential request IDs that reveal no experimental fields.

Existing Opus labels are aligned to source rows and retained in `key63.json`: Part 54a and 54b use the saved `blind_origin` answer order; Part 26.8's old key does not include its model name, so it is checked against the sole model in that source; Part 59A uses its complete manifests and the `01Uk...` setting-label batch. The Part 54b OLMo-7B rows include repeated baseline records. They are kept because the source reader labelled every record; the original reader's output key is checked before labels are attached.

## Submit and collect

The submit command checks the credit key, CA bundle, gathered paths, and output-directory writability before reserving spend. It writes the intent and full request-ID map before POST. For the complete resumable run, use `run-all`; it sends at most 200 requests at a time, persists each batch ID, polls to `ended`, collects labels, and settles that chunk before opening the next reserve:

```sh
python3 part63/reader63.py run-all
python3 part63/reader63.py listing
```

The command waits 30 seconds between status checks and stops after 40 minutes per chunk. A timeout leaves its reservation active and exits nonzero. Rerun `run-all` to recover a saved batch ID, collect an ended chunk, and settle it before advancing. It refuses to advance when any Part 63 reservation remains active or unknown. The lower-level `submit`, `status`, and `collect` commands remain available for manual operation.

The model is `claude-haiku-5-5`, `max_tokens` is 1024, temperature is 0, and thinking is disabled to keep the bounded JSON response compact. The rubric requests all three label schemes in one JSON object. It asks the model to return the named maker and whether it is a real lab; the reader compares that name to the hidden model's lab only after the response is parsed. Model, unit, dose, control status, and question remain in sidecars and are not included in the request.

The calibration set is `part59.blind_setting.CAL`, 34 hand-labelled examples. Only the scheme iii `quality`/BROKEN field has compatible hand labels; this is the inherited 59A gate used by the registered design. More than two incorrect rows, any missing calibration label, or any `max_tokens` stop fails calibration. Planet, origin, and maker schemes are explicitly reported as **not gated by hand labels** in every analysis table row that uses them. The analyzer refuses to score if the registered quality/BROKEN gate fails.

The request plan also repeats 100 answer texts, sampled without replacement using seed `63063`, under separate `dup|...` internal IDs. Their requests are inserted at seeded positions among the original and calibration requests; their IDs do not appear in `key63.json` as extra conditions or answers. The exact sampled answer/duplicate pairs and seed are stored with the submission snapshot and copied into the collected labels. Analysis reports exact-label agreement for planet framing and origin, and exact `(maker_name, maker_kind)` agreement for maker identity. Every scheme is marked `unreliable reader` when agreement is below 90%; fixed predictions remain printed with that status.

The analyzer also checks a deliberately coarse keyword sanity gate based on the cold review's regular expressions. It requires at least one overlapping example for each checked view (off-world text with OFFWORLD, off-world text with `other-planet-or-space`, and Earth text with EARTH). This gate catches empty or inverted output; it is not a quality estimate. No keyword proxy gate is applied to maker identity.

The spend ledger is `paidaction.DEFAULT_LEDGER_PATH`, shared with Parts 60 and 62, with an $8 aggregate cap. Part 63 reserves under kind `part63` and enforces its $3 sub-cap before each reserve. It estimates input tokens as ceil(serialized UTF-8 request bytes / 3) and reserves that estimate plus the full output-token ceiling. These rates and the byte ratio remain unverified. Collection settles from returned token usage, or at the reserved amount if any usage is missing. A timeout, transport error, HTTP 429/5xx, malformed response, or 4xx without a valid Anthropic error body keeps the attempt counted. `reader63.py retire ATTEMPT_ID --who NAME --reason TEXT` records explicit operator retirement. The full `run-all` lifecycle uses the shared ledger primitives directly because the helper's synchronous callback cannot poll an asynchronous batch to completion.

If collection finds any failed or unlabelled answer, it automatically submits exactly one retry batch containing only those failed/unlabelled request IDs. Check and collect that retry by its printed batch ID. Retry results merge into the root batch's `labels63-BATCH_ID.json`; no second retry is sent. Any remaining failed or unlabelled answer blocks analysis.

The corpus has 4,944 rows and 3,677 distinct texts. Each distinct text is requested once, the same label is applied to all its corpus rows, and the 100 repeatability items sample distinct texts. Together with 34 calibration items, the plan has 3,811 requests and an estimated reserve of $1.1154614. This uses inherited, unverified rates and the byte/3 token estimate; it is not a live quote or a guaranteed billing ceiling.

Collected labels are written to `part63/results/labels63-BATCH_ID.json`. Analyze them with:

```sh
python3 part63/analyze63.py --labels part63/results/labels63-BATCH_ID.json
```

Analysis prints per-scheme failure totals and per-cell counts, then prints the keyword views and the five fixed prediction decisions. If any answer remains failed or unlabelled in any scheme after the one retry, it stops before scoring. A complete analysis prints `RETURN_READY` and exits 0; a prediction may be reported FAILED while the analysis itself still completes. There is no rented box or archive lifecycle in Part 63.

## Limits and assumptions

The two spend rates are copied from Part 60's batch ledger and the $3 ceiling comes from the registered Part 63 design. They are code-side assumptions and are not a live quote. The current Messages Batches request, result, and listing fields, the `temperature` and `thinking` parameters, the `stop_reason` values, usage-field names, and Haiku 5.5 availability were not verified against a live API. No tokenizer is loaded; request UTF-8 byte length divided by 3 is the inherited upper token estimate. The parent must check local API compatibility before submission and should not raise the cap to make a rejected reserve pass.

For registered 63d, Qwen3.5's own lab matches case-insensitive substrings `alibaba`, `alibaba cloud`, `tongyi`, `tongyi lab`, `qwen team`, `通义`, or `阿里`. Exact `Tongyi Lab` is reported as sensitivity. The registered mapping comes from the parent decision; no live provider source was checked.

Unclamped selection uses the side-key `kind` and one 24-row sample cell. Primary prediction counts retain row multiplicity; the distinct-text sensitivity row has 19 Qwen3-4B texts and 24 Qwen3.5 texts. The planet rubric follows the registered rule: whether the answer mentions a planet, a place of origin, or a world. The Part 63 keyword proxy uses word-bounded terms.
