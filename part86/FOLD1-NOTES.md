# Part 86 fold 1 notes

## Changes by finding

1. Screen and verification provenance now records SHA-256 for every `part86/*.py` plus the imported `neuron_report.py`, Python/torch/transformers versions, dose, actual model dtype, layer median, calibration, and effective generation settings. Both runners require actual bfloat16 and refuse resume on any config/provenance difference. `screen86.py --reset-layer N` removes layer metadata, final output, and `.batches` fragments; orphan fragments without metadata are refused.
2. The verdict output labels 86.2 and 86.4 as Amendment 1, superseding the original readings. 86.2 uses a standard-library one-sided Fisher exact test, the two-times rate rule, and the minimum ten real coherent personas. 86.4 reads only random changed answers labelled by the reader, requires five labelled answers to read a layer and nine read layers overall, excludes broken answers from the comparison, and reports broken random counts beside the verdict.
3. `test86.py` now checks per-row decode-only steering, real and random write norms, prompt rendering, eos 248044, verdict boundaries, resume refusals, a tiny Qwen-style output canary, and completed-neuron resume behavior. `PART86_GPU_STRICT=1` raises on missing torch/transformers; the runbook gives the strict command. The real-screen canary checks the zero-push answer and logs batch-16 versus batch-1 comparisons for two neurons. The L22 screen checks its residual median against 24.8 within 5%.
4. Verification writes one atomic JSON file per selected neuron, checkpoints after every answer, checks the screen median/provenance, validates picked neurons against changed real screen answers, skips complete files with matching provenance, and prints a measured one-neuron runtime with a 20-neuron projection. The runbook documents one-neuron resumable task chunks.
5. Timing includes only generation calls after model load. The L22 pilot output declares gate 3 over 9,728 answers and marks limited output `NOT READ`; it logs the raw N2742 answer. 86.1 checks that answer text directly with the widened off-world terms.
6. Blind items must live apart from their key; the reader instructions require inline pasting into a tool-less session. `picked20.json` is checked against screen answers before verification.
7. Screen manifests include code hashes, per-layer config hashes, screen output hashes, and item hash. Blind scoring reruns triage, checks all changed answers are represented and match the key, and asserts 9,216 real plus 512 random rows for every layer before rates are computed.
8. The prompt contract is asserted at run start, the false-kept identity regex was narrowed, `neuron_report.py` is on the copy list and hashed, effective generation settings include the model defaults and overrides, and 248044 is in the eos stop set. Blind and verification preparation/scoring refuse existing outputs unless explicitly forced; preparation writes the private key before reader items.

## Mutation table

Every break was applied to a fresh throwaway copy under `/tmp/part86-mutation-verified`; the project files were not mutated for this audit. All eight targeted tests failed as required.

| Mutation | Failing test | Failure message |
|---|---|---|
| M1: every row receives row 0's push | `TestRowsHook.test_pushes_are_per_row_and_only_apply_after_prompt_pass` | `AssertionError: Tensor-likes are not close!` |
| M2: push is applied on prompt pass | `TestRowsHook.test_pushes_are_per_row_and_only_apply_after_prompt_pass` | Prompt assertion failed: hook returned a pushed tensor tuple instead of `None`. |
| M3: random direction scale uses a down-projection row norm | `TestSteeringDose.test_neuron_and_random_write_norm_are_dose_times_median` | `AssertionError: Tensor-likes are not close!` |
| M4: changed median is accepted on resume | `TestResumeAndScreen.test_screen_smoke_provenance_resume_refusal_and_reset` | `AssertionError: ValueError not raised` |
| M5: 86.2 holds for 1 real and 0 random | `TestVerdictRules.test_fisher_hand_table_and_amended_862_edges` | `AssertionError: 'HOLDS' != 'FAILS'` |
| M6: triage-broken random answers enter 86.4 | `TestVerdictRules.test_amended_864_read_threshold_and_broken_answers_excluded` | `AssertionError: 103 != 3` |
| M7: completed verification neurons regenerate | `TestResumeAndScreen.test_verify_skips_completed_neuron_file_with_same_provenance` | `AssertionError: Expected 'generate_batch' to not have been called. Called 20 times.` |
| M8: denominator assertion is removed | `TestVerdictRules.test_scoring_refuses_limited_denominators` | `AssertionError: ValueError not raised` |

## Open questions

- The batch-16/batch-1 runtime comparison records exact text matches and both answers but does not stop on a mismatch. This follows the registered design's note that bf16 batching can change greedy text slightly; confirm whether the observed mismatch rate should become a future gate.
- The explicit Amendment 1 wording says fewer than nine 86.4 read layers is `NOT READ`; that status takes priority even if one of the fewer read layers would otherwise fail.

## What could not be verified here

- The requested venv has CPU torch 2.14.1 and transformers 5.20.0. The strict CPU suite passed with a tiny Qwen-style model, but the Qwen3.5-4B prompt, A770 bfloat16 behavior, L22 median, N2742 answer, runtime canaries, and GPU throughput were not run. The runbook has fail-fast checks for the GPU run; no production model was loaded here.
