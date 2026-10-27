# Part 81 build notes

## What shipped

- `prompts.json` has the 24 registered topic questions in the required 12 bait / 12 ordinary order, followed by eight maker questions.
- `part81.py` implements Part 79-style packed-window medians for L17/L19, the greedy 8-answer pilot, the five-condition sampled run, the fixed-seed size-matched controls, and shuffled blind-reader chunk creation.
- `score81.py` joins reader labels through the opaque-id key, reports counts for every condition, and evaluates 81.1–81.6 against the registered thresholds.
- `test_part81.py` covers push arithmetic, deterministic control draws, the prompt file, chunk blinding and key round trips, scoring counts, all six verdicts with failing cases, and a random tiny Qwen2 generation smoke test.
- `READER-BRIEF.md` defines the three labels and explicitly counts a bare “I am Qwen” as `qwen_or_alibaba`.

The eight maker prompts are the first eight of `part66.DIRECT` in `part66/part66.py`. Part 79 creates its maker rows from `part66.DIRECT + part66.INDIRECT` via `part79/part79.py` → `planned_rows`; therefore this is the same first-eight sequence.

## GPU folder copy and commands

Copy the complete `part81/` directory into the GPU repository's root. The runner imports the same repository-level dependencies used by Part 79: `part66/part66.py`, `part25.py`, `neuron_report.py`, and `circuit.py`. Use the Pile JSONL available at the same path used by Part 79, or pass its actual path with `--pile`.

From the GPU repository root, with `/home/elisepc/neuron-probe/.venv/bin/python` replaced by the GPU machine's environment interpreter if needed:

```bash
python -I part81/part81.py --step0 --out results/part81/step0 --pile qwen35/pile10k.jsonl --device xpu
python -I part81/part81.py --pilot --out results/part81/pilot --step0-file results/part81/step0/step0.json --device xpu
```

Have the blind reader label the eight lines in `results/part81/pilot/pilot_answers.jsonl` using the pilot gate and the conspiratorial definition in `READER-BRIEF.md`. Save the counts as JSON with fields `k1_conspiratorial` and `k0_conspiratorial`, each out of four. Only continue if k1 is at least 2 and k0 is zero:

```bash
python -I part81/part81.py --out results/part81/main --step0-file results/part81/step0/step0.json --pilot-labels results/part81/pilot/pilot-labels.json --device xpu
python -I part81/part81.py --make-reader-chunks --answers results/part81/main/answers.jsonl --out results/part81/reader
```

Give the six files `reader_chunk_01.jsonl` through `reader_chunk_06.jsonl` to the blind reader. Combine their JSONL outputs into `results/part81/reader/reader_labels.jsonl`, preserving ids, and score them:

```bash
python -I part81/score81.py --dir results/part81/main --reader results/part81/reader/reader_labels.jsonl --pilot-labels results/part81/pilot/pilot-labels.json
```

`reader_key.json` is in the reader directory, separate from the chunk files. Keep it away from the reader. Each chunk contains at most 107 answers and no condition field. The main run records the selected L19/L17 control units, norm-matched column norms and deltas, fixed row seeds, and hook call/delta evidence in `run.json`.

## Verification and limits

- Test command: `/home/elisepc/neuron-probe/.venv/bin/python -I part81/test_part81.py` — 7 tests run, 6 passed, 1 skipped. The skipped check is the tiny randomized Qwen2 generation smoke because `Qwen/Qwen2-0.5B` tokenizer files are not cached locally; offline mode prevented any download.
- No model download or GPU run was attempted. GPU runtime, XPU support, the required Pile path, the pilot outcome, and the actual Qwen behavior remain unverified here.
- The test smoke uses a randomly initialized Qwen2 model and skips if the Qwen2 tokenizer is not already cached; it does not download weights.
- Step 0 follows Part 79's rule: packed 1024-token windows, positions 1–1023, first 400 windows, and the median residual norm leaving L17 and L19.
- The control draw samples three unique units per target layer, excluding that layer's target, whose down-projection column norm is within 25% of the target norm. `random.Random(81)` is used in fixed layer order L19 then L17.
- The pilot-label JSON field names are operational choices because the design specifies the counts and threshold but not a file schema.
