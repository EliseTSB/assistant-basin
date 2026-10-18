# Part 17 fold 1 code review

## SUMMARY

Implemented the six requested Part 17 corrections and added offline regression coverage. The full `test_residual17.py` suite passes (12 tests); no model downloads were used.

## FINDINGS

1. Transplant scoring now selects paired logit positions from the current batch. Test: `test_transplant_paired_logits_use_current_batch_prompt_positions`. It failed before the change because the paired-position helper was absent.
2. Reader prompts disable Qwen thinking when the tokenizer accepts `enable_thinking`; answer targets are the prefix plus encoded answer IDs, and their first answer tokens must differ. OLMo templates without that keyword fall back cleanly. Tests: `test_qwen_prefix_disables_thinking_and_targets_append_encoded_answer` and `test_olmo_template_without_thinking_keyword_keeps_generation_prefix`. The Qwen test failed before the change because the prefix did not end with the empty think block.
3. The question now uses the amended “While you read my first message” wording. Test: `test_question_uses_amended_first_message_wording`. It failed before the change.
4. Tagged eval adapter and output paths are covered for self and input runs. Evaluation validates the requested adapter before loading the model and records its SHA-256. Test: `test_tagged_eval_paths_hash_and_missing_adapter_refusal`. It failed before the change because tagged output/hash helpers were absent; the parser and adapter suffix handling were already present.
5. Self, input, and transplant eval JSON now include the TEST majority-class rate beside the accuracy field (transplant accuracy is marked not applicable). Test: `test_test_majority_class_rate_is_reported_from_test_labels`. It failed before the change because the summary helper and rate were absent.
6. Self-eval captures adapter-on group means over TEST passage tokens and reports Pearson correlation and mean absolute difference against base captures. Test: `test_lora_drift_reports_pearson_correlation_and_mean_absolute_difference`. It failed before the change because the drift metric was absent.

## BLOCKERS

None. Real model execution was not run; verification used only tiny synthetic inputs and fake tokenizers.
