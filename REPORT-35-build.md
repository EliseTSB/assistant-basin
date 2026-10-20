# Part 35 build report

## Build and output contract

`part35.py` provides resumable `screen`, `push1`, `push2`, optional blind `judge`, and `score` commands. The screen captures one maximum activation per passage and MLP down-projection input unit, computes within-frame contrasts and a model-wide permutation maximum, emits the top five per concept, and supports screen shards. The push commands clamp only the selected unit on generated tokens, save JSONL rows, and skip completed row keys. Scoring emits one record for each prediction ID, confidence intervals, and per-model cells.

`data/part35-concepts.json` contains the 43 target concepts, the eight fixed held-outs, alien as a separate positive control, the 36 frame bank, discovery/confirmation labels, style controls, and AI/human self-description probes. It is original text. Trump archive input is optional and redacted before caching; raw post text is not persisted.

The named alien control is scored against the concept contrasts at its fixed unit, and its rank is now global across all Qwen3.5-4B MLP down-projection units, rather than restricted to layer 22. Predictions 35.4 and 35.6 report model-specific rates: 35.4 considers Claude/Anthropic candidates that pass both screen halves, and 35.6 reports the best Trump-style candidate separately for each model. Prediction 35.8 records held-out AI/human direction separation and candidate/control projection drops.

The local model revisions found in the read-only cache and pinned in the script are:

| Model | Revision | Local status |
| --- | --- | --- |
| `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | cached |
| `Qwen/Qwen3-4B` | unavailable | absent; no revision pinned and no download attempted |
| `Qwen/Qwen3.5-4B` | `851bf6e806efd8d8a36b00ddf55e13ccb7b8cd0a` | cached |
| `HuggingFaceTB/SmolLM3-3B` | `a07cc9a04f16550a088caea529712d1d335b0ac1` | cached |
| `allenai/OLMo-2-0425-1B-Instruct` | `48d788eca847d4d7548f375ad03d3c9312f6139e` | cached |

Each run header records the resolved revision, model family, hook path, chat-template hash, dtype, split seed, and Trump archive presence. The decoder block stack is discovered from each model and the `mlp.down_proj` path is recorded per layer; real-checkpoint hook compatibility has not been exercised in this worker.

## Cloud command lines

Run from the project root on the specified 4-core, 15-GB CPU host. This uses BF16 to keep 1B–4B weights within memory; FP32 weights for the 4B models alone would exceed the host's nominal RAM. Set `TRUMP_POSTS` to a permitted JSON or Markdown archive path to include after-cutoff posts; leave it unset to run without the archive and record its absence. `OPENAI_MODEL`, `OPENAI_BASE_URL`, and optionally `OPENAI_API_KEY` configure the optional judge; no credential is embedded in the script.

The commands deliberately shard by model and concept range. The concept ranges are zero-based, half-open; `43:44` is the alien positive control. Trump is concept index 42, so the `36:43` scoring shard also computes its fixed-neuron rank.

```bash
set -euo pipefail
OUT=results/part35
CACHE=/mnt/e/neuron-probe/hf/hub
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1

MODELS=(
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3.5-4B"
  "HuggingFaceTB/SmolLM3-3B"
  "allenai/OLMo-2-0425-1B-Instruct"
)
SLUGS=(
  qwen-qwen3-1-7b
  qwen-qwen3-5-4b
  huggingfacetb-smollm3-3b
  allenai-olmo-2-0425-1b-instruct
)
CAPTURE_RANGES=(0:9 9:18 18:27 27:36 36:43 43:44)
SCORE_RANGES=(0:9 9:18 18:27 27:36 36:43)
PUSH_RANGES=(0:9 9:18 18:27 27:36 36:43 43:44)
POST_ARGS=()
if [[ -n "${TRUMP_POSTS:-}" ]]; then
  POST_ARGS=(--trump-posts "$TRUMP_POSTS")
fi

# Screen: capture each shard, score cached activations, then merge model-wide nulls.
for MODEL in "${MODELS[@]}"; do
  for RANGE in "${CAPTURE_RANGES[@]}"; do
    IFS=: read -r START END <<< "$RANGE"
    python3 part35.py screen --model "$MODEL" --cache-dir "$CACHE" --device cpu --dtype bf16 \
      --out "$OUT" --permutations 200 --concept-start "$START" --concept-end "$END" \
      --capture-only "${POST_ARGS[@]}"
  done
  for RANGE in "${SCORE_RANGES[@]}"; do
    IFS=: read -r START END <<< "$RANGE"
    python3 part35.py screen --model "$MODEL" --cache-dir "$CACHE" --device cpu --dtype bf16 \
      --out "$OUT" --permutations 200 --reuse-cache --score-shard \
      --concept-start "$START" --concept-end "$END"
  done
  python3 part35.py screen --model "$MODEL" --cache-dir "$CACHE" --device cpu --dtype bf16 \
    --out "$OUT" --merge-screen-shards
done

# Stage 1: rank-1 candidates plus ten live random controls; four samples per prompt and dose.
for I in "${!MODELS[@]}"; do
  MODEL=${MODELS[$I]}
  SLUG=${SLUGS[$I]}
  for RANGE in "${PUSH_RANGES[@]}"; do
    IFS=: read -r START END <<< "$RANGE"
    python3 part35.py push1 --model "$MODEL" --cache-dir "$CACHE" --device cpu --dtype bf16 \
      --out "$OUT" --concept-start "$START" --concept-end "$END"
  done
  python3 part35.py judge --input "$OUT/$SLUG/push1.jsonl" --judge-model "${OPENAI_MODEL:?set OPENAI_MODEL}" \
    --reliability-check
done

# Stage 2 controls in two slices, then stage-1 hits in concept shards.
for I in "${!MODELS[@]}"; do
  MODEL=${MODELS[$I]}
  SLUG=${SLUGS[$I]}
  HITS="$OUT/$SLUG/push1-hits.jsonl"
  python3 part35.py push2 --model "$MODEL" --cache-dir "$CACHE" --device cpu --dtype bf16 \
    --out "$OUT" --hits "$HITS" --controls-only --control-start 0 --control-end 5
  python3 part35.py push2 --model "$MODEL" --cache-dir "$CACHE" --device cpu --dtype bf16 \
    --out "$OUT" --hits "$HITS" --controls-only --control-start 5 --control-end 10
  for RANGE in "${PUSH_RANGES[@]}"; do
    IFS=: read -r START END <<< "$RANGE"
    python3 part35.py push2 --model "$MODEL" --cache-dir "$CACHE" --device cpu --dtype bf16 \
      --out "$OUT" --hits "$HITS" --skip-controls \
      --concept-start "$START" --concept-end "$END"
  done
  python3 part35.py judge --input "$OUT/$SLUG/push2.jsonl" --judge-model "${OPENAI_MODEL:?set OPENAI_MODEL}" \
    --reliability-check
done

# Score all model files without pooling model-specific cells.
PUSH2_FILES=()
SCREEN_FILES=()
for SLUG in "${SLUGS[@]}"; do
  PUSH2_FILES+=("$OUT/$SLUG/push2.judged.jsonl")
  SCREEN_FILES+=("$OUT/$SLUG/screen.jsonl")
done
python3 part35.py score --push2 "${PUSH2_FILES[@]}" --screen "${SCREEN_FILES[@]}" \
  --out "$OUT/score.jsonl" --cluster-permutations 50000
```

`Qwen/Qwen3-4B` is intentionally absent from these loops because it was not in the inspected cache. Add it only after the run machine has a locally cached revision and a corresponding pinned hash; this build does not fetch it. If the after-cutoff archive is supplied, pass the same `TRUMP_POSTS` path to every screen capture shard so the run header and cache remain consistent.

## CPU planning estimates

These are planning ranges, not benchmarks: this worker has neither PyTorch nor Transformers and did not load model weights. The estimates assume four CPU threads, BF16 weights, roughly 40 generated tokens on average for generation tasks, and effective forward throughput of about 8–20 tokens/s for the 1B model, 5–12 tokens/s for 1.7B, and 2–8 tokens/s for 3–4B models. Actual prompt length, early EOS, CPU BF16 support, memory bandwidth, and model implementation can move these substantially.

| Step | Approximate work | 1B–1.7B estimate | 3B–4B estimate |
| --- | --- | --- | --- |
| Screen capture | 1,708 fixed passages per model (43×36 concept passages, 36 neutral, 36 alien, 24 style controls, 64 self-description rows), plus any archived posts; about 68k–102k input tokens at 40–60 tokens per passage | about 1–6 CPU-hours | about 2–14 CPU-hours |
| Screen score | 200 label-shuffle maxima; each concept shard computes all-layer neuron statistics against same-frame controls | about 1–5 CPU-hours per model | about 2–12 CPU-hours per model |
| Push 1 | Up to 43 concept candidates plus alien and 10 random controls; 32 generated answers per unit across two doses and four prompts; about 1.4k–1.8k answers | about 2–8 CPU-hours | about 4–15 CPU-hours |
| Push 2 | 3,200 random-control identity answers plus 160 non-identity answers at five doses, plus each hit's candidate answers and Trump-style prompts; scales linearly with hit count | about 3–10 CPU-hours before candidate-hit cost | about 5–18 CPU-hours before candidate-hit cost |
| Blind judge | One endpoint call per answer plus a second call for the first 100 answers at each stage | endpoint-bound; roughly tens of minutes to hours | endpoint-bound; roughly tens of minutes to hours |
| Final score | 50,000 prompt-cluster permutations per candidate cell; no model forward pass | seconds to tens of minutes | seconds to tens of minutes |

The screen scoring range is especially uncertain because the current NumPy null loops over concepts, frames, and layer units. The 9-concept shards cap individual job size and permit resumption, but they do not reduce total work. The script also reloads weights per push shard and recomputes a small baseline-projection set in each `push2` invocation. These estimates include that overhead only approximately. For planning, reserve a day or more per larger model for the full screen and push stages, especially if several stage-1 hits survive judging.

## Ambiguities and limitations

- “Spread on a fixed neutral reference set” is implemented as population standard deviation (`ddof=0`) across the 36 neutral passage maxima for each unit, with an epsilon floor.
- The design names Yona et al.'s cross-prompt consistency rank but does not specify its formula. The output uses the mean within-frame percentile rank across all 36 frames as a secondary proxy; it is not claimed to reproduce Yona's exact statistic.
- The screen null independently permutes concept labels within each frame, then takes the maximum statistic across units and target concepts. Shards share the same seed and are merged by taking the per-permutation maximum across shard results.
- The phrase “prompt-level cluster permutation over control neurons” does not fix the randomization algorithm. The implementation first averages answers within each prompt, then independently selects one pseudo-target from the candidate plus that model's ten live controls for each prompt; the one-sided statistic compares the pseudo-target to the remaining controls. The same test is rerun with 50,000 permutations for scored candidates.
- Prediction 35.14 says “naive top-5 contrast” without defining whether that means rank 1 or an aggregate over five candidates. The output exposes the operational choice: it compares the screen permutation maximum to the rank-1 primary contrast in each concept/model cell. If the intended top-five summary is a mean or median, change that prediction definition before interpreting the row.
- Stage-1 `push1-hits.jsonl` includes a candidate if any identity answer receives a persona label or if either screen half beats the discovery null. This is an exploratory filter because the brief does not specify a stage-1 hit threshold.
- The English/Spanish/Chinese/German/Russian passages and indirect descriptions are author-written and have not been professionally translated or independently checked. The test verifies that indirect passages omit concept terms and saved translations, not that every translation is idiomatic.
- The judge's style cutoff is score 3 of 4, matching the rubric's “strong use” description. “Without claiming to be him” is operationalized by a blind judge label that does not match Trump.
- For prediction 35.8, the AI-self direction is the unit-normalized difference between 24 training AI and 24 matched human post-block residual means per layer. Separation uses the eight held-out examples per class. Candidate clamp projection drops are averaged over the first 16 generated tokens, then compared with the 95th percentile of live-control drops.
- For 35.8, mailbox message sequence 1 (`add-35-8-readout`) changed the plan: the build added matched AI/human self-description rows, held-out direction separation, candidate/control projection tracking, and the drop-versus-control 95th-percentile score. The message also corrected the Goodfire Trump-neuron lead to Goh et al. (2021)'s CLIP model; no code relies on the erroneous attribution. It was recorded as `considered` and applied. No parent messages were declined or deferred.
- `PART35-DESIGN.md` and `PREDICTIONS.md` were left unchanged as required. This worker did not have a Trump post archive, so the runtime path remains optional and no after-cutoff observation is available here.

## Verification

Executed:

```text
python3 -m py_compile part35.py test_part35.py
python3 -m unittest -v test_part35.py
Ran 11 tests in 0.159s
OK (skipped=1)
```

The skipped test constructs tiny random Qwen3/OLMo-shaped models and checks live hooks/write-back; it skipped because PyTorch and Transformers are not installed. The score fixture produced 13 prediction rows and checked confidence intervals, screen gating for prediction 35.4, model-separated Trump-style output for 35.6, and the 35.8 control comparison. No real screen, push, judge, or model-backed score was run on this worker.

## SUMMARY

- Shipped the Part 35 dataset, resumable screen/push/judge/score pipeline, and tests.
- Added global alien fixed-neuron rank and model-separated scoring for predictions 35.4 and 35.6.
- No model weights were downloaded or loaded.

## FINDINGS

- Syntax compilation passed. The unittest suite passed 11 tests with one model integration test skipped for missing PyTorch/Transformers.
- The pinned local Qwen3-4B revision is unavailable, so prediction 35.4 cannot be fully evaluated without that model.
- Planning estimates are not measured runtimes; cloud inference and the external judge remain to be run.

## BLOCKERS

- `Qwen/Qwen3-4B` is missing from the inspected local model cache. No download was authorized or attempted.
- This worker lacks PyTorch and Transformers, so tiny-model integration and all real model inference remain unverified here.
- No Trump after-cutoff archive or judge endpoint credentials were supplied; those runtime inputs remain optional.
