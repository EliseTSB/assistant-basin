# Part 34 build report

## What was built

`part34.py` has `run` and `score` commands. `run` uses the pinned Qwen3-1.7B base and zigzag adapter in offline mode, supports CPU and XPU, appends JSONL rows, skips completed keys, prints progress every 20 new rows, and records model revisions, input/code hashes, fixed answer-token IDs, the hook site, and the random-direction seed in the header.

The run has 13 turn-one conditions: the seven text controls, four original hook conditions, and the two Prime-requested random-direction hooks. It runs both M and B. The question file contains 40 original neutral one-line questions and six original ask/not/fact wordings. Each row records turn-one text, token IDs, D and the full 28-layer z profile over question and reply tokens. Turn two has four counterbalanced A/B elicitations, the Weaver controls, separate cache-kept/recomputed/adapter-off/same-turn arms, and the B reader over M's cache versus B's own recomputation on the same M token sequence. The scorer emits metrics for predictions 34.1–34.14, Wilson intervals for rates, and per-question comparisons with the matching none row. It keeps conditions and arms separate.

The same-turn hook is limited to the user question body and generated answer tokens. Turn-one injection covers the whole rendered turn-one prompt and generated reply. Hooks are attached to decoder-block outputs, which are the post-block residuals used by the adapter definition. Runtime checks verify 28 blocks, unit direction vectors, positive SD values, and the fixed Qwen letter IDs A=32 and B=33.

`score` also writes `free-text-blind-review.jsonl` with answer-only items and empty human-annotation fields, plus a separate `free-text-blind-key.json`. The JSONL data rows include the requested regex mention flag and category flags for hand review.

## Adapter definition check

The adapter README defines `h_l` after decoder block `l` and uses `z_l = (h_l · direction_l - mean_l) / sd_l`. Its layer-indexed shapes match the brief. The hook writes exactly `±2 * sd_l * direction_l`, with the raised sign on the selected parity. No definition mismatch was found in the README. The README reports that the learned adapter reached 88% of its target on its held-out set; this is recorded as prior context, not assumed for Part 34. Part 34 measures the achieved D and profile directly.

The local adapter cache has the README and adapter config, but it does not contain the referenced `adapter_model.safetensors` or `cat_direction.safetensors` blobs. Therefore I could not independently check the direction-vector norms here; `run` enforces the brief's unit-norm requirement when the real file is available.

## Cloud commands

These commands assume the cloud machine already has the exact revisions in a standard Hugging Face cache rooted at `/mnt/e/neuron-probe/hf`. The script sets `local_files_only=True`; it will not download models.

```bash
export HF_HOME=/mnt/e/neuron-probe/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python3 part34.py run --out results/part34 --device cpu --direction-file "$HF_HOME/hub/models--Sauers--qwen3-1.7b-cat-zigzag/snapshots/2c871b5cc51e5be9f4f6ae258685448518191c4b/cat_direction.safetensors"
python3 part34.py score --out results/part34
```

To use the XPU, change `--device cpu` to `--device xpu`. The run writes `results/part34/rows.jsonl`; the score command writes `summary.json` and the two blind-review files beside it. The cloud cache must expose the pinned revisions in the standard snapshot layout for these commands to work.

## Row counts and estimated CPU time

The full design produces 1,040 turn-one rows and 14,000 turn-two rows (15,040 total). That is 12,240 A/B rows and 1,760 free-text rows. The turn-two count includes the four A/B variants across every turn-one condition, and two B cross-cache arms with both parity-word and parity-free readouts.

| Step | Approximate work | CPU estimate | Basis |
|---|---:|---:|---|
| Load model and adapter | 1.7B FP32 weights, about 6.8 GB, plus adapter | 1–3 minutes | Local-cache read and CPU initialization; excludes cold disk delays. |
| Turn one | 1,040 greedy generations, up to 64 tokens each | 1–4 hours | Roughly 25K–66K generated tokens plus prompt prefill; assumes 5–15 effective tokens/second on four CPU cores. |
| Turn two | 12,240 one-token A/B probes and 1,760 free-text generations, up to 48 tokens | 17–52 hours | About 0.8M question-prefill token equivalents plus up to 84K free-text tokens, at the same assumed throughput. |
| Recompute turn-one caches | About 1,720 full-sequence passes across recomputed, adapter-off, same-turn, and B-own-cache arms | 3–10 hours | Estimated 120K–250K sequence tokens; depends on answer lengths and cache implementation. |
| Score and export review rows | 15,040 JSONL rows | Under 1 minute | Standard-library grouping, Wilson intervals, and file output. |

Planning estimate: about 22–66 hours end to end. The first 20 rows should be used to measure the actual CPU rate and revise this estimate before reserving the rest of the machine time.

## Ambiguities and design notes

- The checked-out `PREDICTIONS.md` initially ended at Part 33. Parent control message 1 arrived during implementation, reporting Prime's update to the design and predictions 34.11–34.14. I reread both updated sections, applied all five requested additions and scoring changes, and recorded message 1 as `considered`. No parent message was declined or deferred.
- The design does not explicitly say whether the four primary turn-two A/B forms should run after every text condition or only after injection conditions. I run them after every turn-one condition, as the generic turn-two design implies. This makes honest-request and text-only controls directly comparable.
- The B cross-cache comparison is run for every condition, including none as a baseline. It uses M's exact prompt and generated reply IDs for both arms; one cache is computed with M's turn-one adapter/hook state and the other is recomputed with B's adapter disabled. B reads both with the same question and no turn-two hook.
- Counterbalancing uses zero-based question indices, so q01 receives even-first and q02 odd-first. The assignment is balanced across the 40 questions.
- D is computed over the selected question-body or reply tokens, with chat-template markers excluded. The per-layer z profile averages z across those same positions. The stated target percent is `abs(D) / 4 * 100`.
- The free-text regex is only a flag. Blind human review remains unperformed in this build; `score` writes answer-only review items and keeps the key separately.
- On a fresh run, cache-kept uses the actual generation cache. On resume, a missing turn-two row reconstructs that cache by re-encoding the stored turn-one tokens with the same adapter/hook. Causal states should match, with possible minor numerical differences.
- The predictions specify that if 34.2 detects a turn-two parity-word effect, 34.4–34.6 and 34.9 should be interpreted from parity-free rows only. The scorer records that gate and reports both forms.

Design concern: parity-word options themselves contain the adapter's trigger words. The prediction gate addresses this by measuring turn-two D in none and restricting interpretation to parity-free rows when the effect appears. The parity-free options still contain layer numbers, which the prior zigzag battery reportedly affected weakly; the build records D on those question positions as designed.

## SUMMARY

- Built the Part 34 data, runner, scorer, resume logic, hook controls, and blind free-text review export.
- Implemented the Prime additions and added scoring for predictions 34.1–34.14.
- The adapter README definition matches the implemented projection and hook site.

## FINDINGS

- The 40-question scan passes for cat, even, odd, and layer terms; the question validator also rejects digits and animal terms.
- `python3 -m py_compile part34.py test_part34.py` passed.
- `python3 -m unittest -v test_part34.py`: 7 tests run in 0.059 seconds; 4 passed and 3 tiny-model tests skipped because PyTorch/Transformers are unavailable.
- `git diff --check` passed. No real model weights were loaded and no downloads were attempted.

## BLOCKERS

- The worker environment has no PyTorch, Transformers, PEFT, or safetensors runtime, so the Qwen-shaped hook/cache tests and real CPU run could not execute here.
- The provided adapter cache is missing both safetensors weight files, so the cloud command requires a complete offline cache in the stated snapshot layout.

## Fold 1

### What changed

- Added decoder-block output recorders. The recorder is registered after any steering hook, reads tuple element zero when needed, and removes all hooks in a `finally` block.
- Turn-one and turn-two D values and z profiles now use the raw output of each decoder block. This covers generated profiles and all turn-two arms, including cache-kept, cache-recomputed, B-reads-M-cache, B-reads-B-cache, and same-turn. The final block is no longer taken from `output.hidden_states[-1]`, which is post-final-norm in Qwen3.
- Added a test that checks the recorded final-layer output against a raw block hook and confirms it differs from `output.hidden_states[-1]`; steering tests also assert that recorders observe the post-steering value.
- The tiny-model end-to-end test exposed and fixed two run-path issues: the neutral scan matched the substring “even” in “evening”, and turn-two generation did not unpack the fourth return value containing layer traces.
- The end-to-end test ran `run` for one question on a tiny random Qwen3 model, checked profiles across cache and same-turn arms, and ran `score` on its 376 rows.

### Test output

Command: `/home/elisepc/neuron-probe/.venv/bin/python -m unittest -v test_part34.py`

```text
test_cached_vocab_has_fixed_first_answer_letter_ids (test_part34.Part34HelpersTest.test_cached_vocab_has_fixed_first_answer_letter_ids) ... ok
test_counterbalancing_alternates_even_and_odd (test_part34.Part34HelpersTest.test_counterbalancing_alternates_even_and_odd) ... ok
test_questions_are_40_neutral_single_lines (test_part34.Part34HelpersTest.test_questions_are_40_neutral_single_lines) ... ok
test_resume_skips_completed_keys_and_score_fixture (test_part34.Part34HelpersTest.test_resume_skips_completed_keys_and_score_fixture) ... ok
test_cache_kept_and_recomputed_use_same_token_context_but_distinct_values (test_part34.TinyQwen3Test.test_cache_kept_and_recomputed_use_same_token_context_but_distinct_values) ... ok
test_d_matches_hand_computation (test_part34.TinyQwen3Test.test_d_matches_hand_computation) ... ok
test_hook_adds_only_named_vectors_at_named_layers_and_positions_then_removes (test_part34.TinyQwen3Test.test_hook_adds_only_named_vectors_at_named_layers_and_positions_then_removes) ... ok
test_recorded_last_layer_is_raw_block_output_not_final_norm (test_part34.TinyQwen3Test.test_recorded_last_layer_is_raw_block_output_not_final_norm) ... ok
test_tiny_model_run_and_score_end_to_end (test_part34.TinyQwen3Test.test_tiny_model_run_and_score_end_to_end) ... wrote /tmp/tmp_tya8jjv/summary.json (4 rows)
progress rows=20 elapsed_s=0.0
progress rows=40 elapsed_s=0.1
progress rows=60 elapsed_s=0.1
progress rows=80 elapsed_s=0.1
progress rows=100 elapsed_s=0.2
progress rows=120 elapsed_s=0.2
progress rows=140 elapsed_s=0.3
progress rows=160 elapsed_s=0.3
progress rows=180 elapsed_s=0.3
progress rows=200 elapsed_s=0.4
progress rows=220 elapsed_s=0.4
progress rows=240 elapsed_s=0.4
progress rows=260 elapsed_s=0.5
progress rows=280 elapsed_s=0.5
progress rows=300 elapsed_s=0.6
progress rows=320 elapsed_s=0.6
progress rows=340 elapsed_s=0.7
progress rows=360 elapsed_s=0.7
complete new_rows=376 total_rows=376 elapsed_s=0.7
ok

----------------------------------------------------------------------
Ran 9 tests in 0.815s

OK
wrote /tmp/tmppqckf3g_/summary.json (376 rows)
```

`/home/elisepc/neuron-probe/.venv/bin/python -m py_compile part34.py test_part34.py` passed. `git diff --check` passed.

### SUMMARY

- Switched every measured z profile and D calculation to captured raw decoder-block outputs.
- The nine-test suite passed with no skips; tiny-model `run` and `score` produced 376 rows.

### FINDINGS

- The recorder sees steering-hook output because it is registered afterward, and its final-layer output differs from Qwen3's post-norm `hidden_states[-1]`.
- Parent control mailbox was empty at each checkpoint; no parent messages changed or deferred the plan.

### BLOCKERS

- No blocker for this scoped change. The real 1.7B model and adapter were not run; the earlier report records that the local adapter cache lacks its safetensors weights.

## Fold 2

### SUMMARY

- A/B rows now prefill the assistant response with a standalone `(` and score `A` and `B` at the following position. The same single cached forward pass supplies the greedy token and the measured prefill state; free-text rows keep the original unprefilled generation path.
- Option letters now map to normalized `even`/`odd` concepts, so target log probabilities, margins, pair-normalized probabilities, greedy concepts, and even-minus-odd logits are populated for rows with a defined target.
- Added half-open question shards, multi-file and shard-directory scoring, and `--threads` (default 4). Row keys are unchanged.

### FINDINGS

- `/home/elisepc/neuron-probe/.venv/bin/python -m unittest -v test_part34.py`: all 12 tests passed with no skips. This includes the pinned real tokenizer with fake logits for both option orders and raised/lowered forms, a one-forward-pass check, and two tiny-model shards (questions `[0,1)` and `[1,2)`) scored together as 752 rows.
- Measured on the two-layer, 32-hidden random Qwen3 test model with one PyTorch thread: turn-one rows averaged 0.00131 s, turn-two A/B rows 0.00108 s, free-text rows 0.00144 s, and turn-two cache preparation 0.00029 s per arm. These tiny-model rates are useful for confirming the code paths, not for scaling to the 1.7B model.
- The parent smoke rate projected about 40 hours for the full 40-question run. A/B rows are 12,240 of 15,040 total rows; halving their probe passes gives an idealized estimate of about 24 hours. Allowing for cache and model overhead, estimate 24–30 hours on similar CPU resources. This is an extrapolation; this worker did not run the 1.7B weights.
- `git diff --check` and `python -m py_compile part34.py test_part34.py` passed. Parent-control mailbox was empty at all checkpoints; no messages changed, were declined, or deferred.

### BLOCKERS

- None for this fold. Real-model timing remains an estimate rather than a measurement.
