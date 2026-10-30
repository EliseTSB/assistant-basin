# Part 86 second reader

This implements the registered second read in `PART86-DESIGN.md`, Amendment 2. It labels a reproducible 10% sample of the first reader's blind answer items, then reports agreement on the binary decision "coherent persona" versus "not a coherent persona." It never reads layer, neuron, or real/random metadata.

## Item set and sample

The population is every item row in the J-lens file `data/qwen3.5-4b/labels/items.jsonl`, resolved under `$JLENS_ROOT` (default root name: `jlens-persona-neurons`). That file has 3,349 unique item IDs, each with `id` and `text`. The sample contains 335 items, selected with `random.Random(8687).sample(items, round(0.10 * len(items)))`. Thus this is a sample of the 3,349 first-reader item records, not a sample of the 3,353 linked screen rows. The matching first labels are read from `data/qwen3.5-4b/labels/labels.jsonl` under the same root.

The registered first reader is Claude Haiku 5.5 (`claude-haiku-5-5`). The second reader is Claude Opus 5.5 (`claude-opus-5-5`). Its prompt uses `part86/READER-BRIEF.md` verbatim and appends the selected item's opaque ID and answer. The response schema and coherent-persona classification reuse `part86/blind86.py` validation and kappa code. That file has no API request or response parser, so this script supplies the Anthropic request and JSON-object extraction.

## Run

From the Assistant Basin repository root, run these three commands in order:

```text
python3 results/part86/second-reader/second86.py --sample
python3 results/part86/second-reader/second86.py --label
python3 results/part86/second-reader/second86.py --agree
```

`--sample` writes `results/part86/second-reader/items_second.jsonl`. `--label` writes `results/part86/second-reader/labels_second.jsonl`, retries each item up to five times, and keeps an `{id, error}` row after exhausted retries. It reads `ANTHROPIC_CREDITS_KEY` from the existing Anthropic credentials file and never prints the key. `--agree` writes `results/part86/second-reader/agreement.json`.

The expected API count is 335 initial message requests for `--label`, one for each sampled item. Retries can raise the total to at most 1,675 requests. Sampling and agreement make no API calls. The agreement report records the full sample count and each reader's error and missing rows. Kappa, raw agreement, and the 2x2 counts use only items with valid labels from both readers; error rows stay visible in the counts and are not treated as negative labels.

## Reading rule

The `kappa` field in `agreement.json` is the Cohen's kappa required before Part 86 verdicts 86.2 through 86.6 are read. If it is below 0.6, those verdicts remain NOT READ until a second full reading is completed, as specified by Amendment 2 in `PART86-DESIGN.md`. The agreement report also gives raw agreement and the counts for first-reader coherent persona yes/no against second-reader yes/no.

Tests use fixtures only and do not call the API. Run them from the repository root with:

```text
python -m pytest -q -p no:cacheprovider results/part86/second-reader/
```
