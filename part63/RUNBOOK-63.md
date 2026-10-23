# Part 63 runbook

Part 63 gathers the frozen answers, submits one blind Haiku 5.5 batch, and analyzes the returned labels. The builder must not make an API request. The only command that submits work is `reader63.py submit`.

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

First confirm that the credit key is available in the parent environment. The parent should then run:

```sh
python3 part63/reader63.py submit
python3 part63/reader63.py status BATCH_ID
python3 part63/reader63.py collect BATCH_ID
```

The model is `claude-haiku-5-5`, `max_tokens` is 1024, and thinking is disabled to keep the bounded JSON response compact. The rubric requests all three label schemes in one JSON object. It asks the model to return the named maker and whether it is a real lab; the reader compares that name to the hidden model's lab only after the response is parsed. Model, unit, dose, control status, and question remain in sidecars and are not included in the request.

The calibration set is `part59.blind_setting.CAL`, 34 hand-labelled examples. Only the scheme iii `quality`/BROKEN field has compatible hand labels. More than two incorrect rows, any missing calibration label, or any `max_tokens` stop fails calibration. Planet-frame and maker-identity categories have no hand-labelled calibration gate, and the origin category itself has no compatible 34-row hand-label set. The analyzer refuses to score if this gate fails.

The analyzer also checks a deliberately coarse keyword sanity gate based on the cold review's regular expressions. It requires at least one overlapping example for each checked view (off-world text with OFFWORLD, off-world text with `other-planet-or-space`, and Earth text with EARTH). This gate catches empty or inverted output; it is not a quality estimate. No keyword proxy gate is applied to maker identity.

The spend ledger is `part63/review/spend.jsonl`. It reserves the conservative request-byte input estimate plus the full output-token ceiling before POST, refuses any reserve that would take settled or reserved total above $3, and settles from returned token usage after collection. If usage is missing, it settles at the reserved upper bound. A transport failure or malformed POST response keeps the reserve active; reconcile the batch before retrying. Definite API rejections (400, 401, 403, 404, 413) release their reserve.

For the gathered 4,944-answer corpus and 34 calibration requests, the current local estimate is $1.836147 for 4,978 requests. This estimate uses the Part 60 price constants noted below; it is a code-side reserve estimate, not a live provider quote.

Collected labels are written to `part63/results/labels63-BATCH_ID.json`. Analyze them with:

```sh
python3 part63/analyze63.py --labels part63/results/labels63-BATCH_ID.json
```

Analysis prints per-scheme failure totals and per-cell counts, then prints the keyword views and the five fixed prediction decisions. If failed plus unlabelled answers exceed 2% in any scheme, it stops before scoring. A complete analysis prints `RETURN_READY` and exits 0; a prediction may be reported FAILED while the analysis itself still completes. There is no rented box or archive lifecycle in Part 63.

## Limits and assumptions

The two spend rates are copied from Part 60's batch ledger and the $3 ceiling comes from the registered Part 63 design. They are code-side assumptions and are not a live quote. The current Messages Batches request and result fields, the `thinking` parameter, the `stop_reason` values, usage-field names, and Haiku 5.5 availability were not verified against a live API. No tokenizer is loaded; request UTF-8 byte count is used as a conservative input-token upper estimate. The parent must check local API compatibility before submission and should not raise the cap to make a rejected reserve pass.

The maker's "own lab" relation uses the model-to-lab aliases in `reader63.py` after the blind read. Those aliases are encoded locally and were not independently verified in this build. This affects maker-category counts outside the fixed Qwen3.5 comparison as well as whether a named lab is classified own or another.
