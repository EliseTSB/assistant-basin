# Part 85 build notes

## Choices where the design was silent

- Pretraining and mid-training stages use the `allenai/OLMo-2-0425-1B` repo and their table branch names. SFT, DPO, and Instruct use the separate suffixed repos (`...-SFT`, `...-DPO`, `...-Instruct`) at `main`; the checkpoint resolver asks the Hugging Face API for each branch's full commit and refuses a prefix mismatch. The official [base model card](https://huggingface.co/allenai/OLMo-2-0425-1B), [SFT card](https://huggingface.co/allenai/OLMo-2-0425-1B-SFT), [DPO card](https://huggingface.co/allenai/OLMo-2-0425-1B-DPO), and [Instruct card](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct) confirm these repository names.
- The neutral WikiText stream concatenates nonempty test rows with one EOS token between rows, then takes the first 25,600 token IDs. Its digest is SHA-256 over row-major little-endian int32 IDs. Later checkpoints load the shared IDs and check their vocabulary bounds.
- Step 1 Pearson correlation flattens all 200 x 128 target activations. A checkpoint's maximum is the maximum individual target activation over those same positions. Passage means are per-passage token means; the ordinary SD is the sample standard deviation (`ddof=1`) across the 24 ordinary passage means.
- Step 0 compares visible token-spelling lengths after removing `Ġ`, `▁`, or `</w>` boundary markers. Each random set samples without replacement from the remaining non-special tokens in that length range. The direct-readout z-score uses the sample SD (`ddof=1`) of the 1,000 random-set mean logits. It uses the final norm gain vector on the down-projection column; the common RMS normalization scalar does not change that z-score.
- Safetensors stores matrices in row-major order, so column 6978 is read by fetching the full down-projection tensor. Selected output-embedding rows are read in 256-row HTTP ranges to avoid thousands of one-row requests; unrequested neighboring rows are discarded.
- Chat prompts use the checkpoint tokenizer's `apply_chat_template`; plain prompts are exactly `Question: <q>\nAnswer:` with special-token insertion disabled. Every generation has the registered seed `85000 + 100 * question_index + sample_index`. Unclamped cells use an observer hook so each cell can prove the down projection fired during both prompt and decode.
- A valid onset candidate must be a same-unit checkpoint and every later checkpoint in its suffix must also pass the same-unit gate and show the push. An earlier checkpoint marked a different unit is excluded; in the synthetic test, this permits onset at SFT after a different-unit mid-training-end result.
- The pilot design's quality condition includes “answer the question,” while `blind_origin.RUBRIC` only labels origin and form. Pilot labels therefore add a boolean `responsive` field for the 24 end-of-pretraining unclamped answers. The quality gate counts answers that are both `OK` and responsive. Main-run labels remain the two rubric fields. Readers also receive Part 84's note that a company name containing a space-related word is still an ordinary company origin unless the speaker claims to be from space; mixed Chinese and English alone is not BROKEN.
- If the quality pilot fails, the code reports `amend_frame_and_rerun_once`. A researcher can provide a written UTF-8 template with one `{q}` placeholder and rerun as attempt 2; the template SHA is recorded and main generation must match it. A failed second quality gate cannot launch a third pilot or authorize main. The wording of any amendment remains the researcher's decision.

## Test output and deliberate breaks

The isolated system interpreter did not have torch or the experiment packages, so the provided project venv was used. Final command:

```text
/home/elisepc/neuron-probe/.venv/bin/python -I part85/test_part85.py
Ran 9 tests in 0.016s
OK
```

Separate passage validation found 24 pairs and zero violations: each passage has 2–3 sentences, each ordinary passage avoids the registered off-world terms, and paired word counts are within 20%.

The tests construct a small random `Olmo2ForCausalLM`, use local safetensors, and make no network calls. Four deliberate mutations were applied one at a time, each test failed, and every mutation was reverted:

1. Changed the seed base from 85000 to 85001. `test_paired_seeds_and_registered_counts` failed: expected 85000, got 85001.
2. Changed the clamp assignment to set the requested value plus 1.0. `test_hook_sets_prompt_and_generated_positions` failed its assertion that every prompt/decode input position held the requested value.
3. Replaced the strided-column read with a contiguous flattened slice. `test_safetensors_range_reader_column_and_row` failed when comparing against `matrix[:, 3]`.
4. Changed onset to return the first push without checking the full suffix. `test_no_onset_different_unit_and_calibration_disagreement` failed because it returned `a` where the no-onset case expected `None`.

## Ambiguities and limits

- The post-training rows in the registered table name `-SFT`, `-DPO`, and `-Instruct` without spelling out whether these are repo suffixes or branches; separate repos with `main` were chosen and are verified at runtime by the registered commit prefixes.
- The pilot's relevance check is absent from its two-label rubric; the additional `responsive` bit makes the gate explicit rather than inferring relevance from grammaticality.
- The full checkpoint download, WikiText-backed activation pass, and generation runs were not run in this build worker. They are designed for the cloud CPU workflow in `CLOUD-RUN.md`; only the CPU tiny-model and local-reader tests ran here.
- Checkpoint resolution, model downloads, and WikiText download still depend on Hugging Face / dataset access in the cloud session.

## Fold 1

1. **Fixed — missing `hashlib` import.** Added it in `part85/part85.py:6`; the real `run()` path is now driven in pilot and main mode by `test_run_pilot_main_prepare_and_score_smoke` (`part85/test_part85.py:466`).
2. **Fixed — pilot questions and responsiveness.** `score85.prepare` includes question and answer in every item (`part85/score85.py:93`); pilot label validation requires a boolean `responsive` for every item (`part85/score85.py:158`). `CLOUD-RUN.md` now tells the reader to label all 96 and the gate checks responsive, OK end-of-pretraining answers. Covered by the end-to-end pilot scoring smoke.
3. **Fixed — stale resumed clamp cells.** `_load_completed` compares unit and clamp value with current step 1 (`part85/part85.py:254`); `run()` records and checks the step 1 SHA-256 (`part85/part85.py:301`). Covered by `test_load_completed_refuses_changed_step1_clamp` (`part85/test_part85.py:325`).
4. **Fixed — all published checkpoint weight formats.** `SnapshotTensorReader` reads indexed shards, one safetensors file, or `pytorch_model.bin` using safe CPU `torch.load` (`part85/step0_weights.py:61`, `:77`). Step 0 downloads a pinned snapshot once or accepts `--snapshot-dir`; steps 1 and 2 accept that same directory. Covered by `test_snapshot_reader_supports_sharded_single_and_bin` (`part85/test_part85.py:240`).
5. **Fixed — OLMo2 MLP output norm in direct readout.** Step 0 multiplies down-projection column 6978 by layer 10's post-feedforward norm and final norm (`part85/step0_weights.py:399`, `:413`). Covered by `test_post_norm_readout_z_score_and_unique_token_sets` (`part85/test_part85.py:264`).
6. **Fixed — one-sided calibration onset.** If one dose has an onset and the other does not, the result is a calibration disagreement (`part85/score85.py:320`). Covered by `test_decisions_check_correlations_and_one_sided_calibration` (`part85/test_part85.py:304`).
7. **Fixed — incomplete main score inputs.** Main preparation accepts only all registered cells or a complete chat-only scope with per-answer pilot authorization (`part85/score85.py:110`, `:256`). Strict scoring requires all twelve step files and rejects `NOT READ`; `--partial` explicitly opts into incomplete analysis (`part85/score85.py:247`, `:469`). Covered by the 2,880-answer end-to-end score and `test_main_scoring_requires_complete_data_unless_partial` (`part85/test_part85.py:340`).
8. **Fixed — predictable IDs, overwrite, and item integrity.** Preparation uses cryptographic shuffle/IDs, refuses existing files without `--force`, stores the items SHA-256 in the answer-key envelope, and verifies it during scoring (`part85/score85.py:56`, `:68`). Covered by `test_prepare_has_question_random_ids_digest_and_refuses_overwrite` (`part85/test_part85.py:392`).
9. **Fixed — dtype, library, and dataset provenance.** Step 1 asserts bf16 and records actual parameter dtype, library versions, and the immutable WikiText revision (`part85/step1_acts.py:27`, `:181`, `:212`). Step 0 and step 2 also record versions and dataset revision metadata. Covered by `test_step1_run_records_bf16_dataset_revision_and_versions` (`part85/test_part85.py:439`) and the main smoke.
10. **Fixed — missing unit and integration coverage.** Added tests for all 12 checkpoint scoring including 85.9; reader agreement; `_main_scope`; `_load_completed`; `collect_answers`; `validate_record_counts`; tiny-model step 1; bf16 decode; `read_rows`; z-score and token sets; exact Fisher value; correlation below 0.8; calibration disagreement; and prepare overwrite refusal. See `part85/test_part85.py:199-466`.
11. **Fixed — duplicate off-world first tokens.** `token_sets` now requires all 16 unique IDs (`part85/step0_weights.py:366`). Covered by `test_post_norm_readout_z_score_and_unique_token_sets` (`part85/test_part85.py:264`).
12. **Fixed — EOS and generation config provenance.** EOS candidates matching tokenizer unknown ID are dropped (`part85/part85.py:133`); the effective model generation config is recorded (`part85/part85.py:362`). Covered by `test_cell_values_main_scope_eos_and_template` (`part85/test_part85.py:199`) and the main smoke.

The required full command passed after the fold:

```text
/home/elisepc/neuron-probe/.venv/bin/python -I part85/test_part85.py -v
Ran 17 tests in 1.020s
OK
```

Three deliberate breaks were each reverted immediately after the focused test failed:

1. Restored the old two-onsets-only calibration condition. `Part85Tests.test_decisions_check_correlations_and_one_sided_calibration` failed at the assertion that the one-sided onset is a disagreement (`AssertionError: False is not true`).
2. Disabled the 16-unique-token guard. `Part85Tests.test_post_norm_readout_z_score_and_unique_token_sets` failed because the expected `ValueError` was not raised.
3. Disabled the resumed clamp-value comparison. `Part85Tests.test_load_completed_refuses_changed_step1_clamp` failed because the expected `ValueError` was not raised.

The failure output from each deliberate break was observed; the code was restored and the full suite was rerun afterward.
