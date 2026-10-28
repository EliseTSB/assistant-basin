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
