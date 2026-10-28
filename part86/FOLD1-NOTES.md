# Part 86 fold 1 notes

## Changes by finding

1. Fold 1 hashed every Python file, which round 2 supersedes. Screen resumes now hash only `common86.py`, `screen86.py`, and `neuron_report.py`, with versions, calibration, dose, dtype, median, and generation settings. Blind scoring and verification carry their own manifests and hashes. `--reset-layer` is applied only after the complete requested configuration is checked; orphan fragments are refused.
2. Fold 1 amended 86.2 and 86.4 but its implementation still counted coherent still-Qwen answers as failures for 86.4. The fold-2 scorer now maps labels once, counts `not_coherent`, `unclear`, `coherent_persona`, and `coherent_still_qwen` separately, and applies the golden verdict rules exactly. Triage-broken random counts are reported but excluded from 86.4.
3. `test86.py` checks per-row decode-only steering, real and random write norms, prompt rendering, eos 248044, golden verdict boundaries, resume refusals, and a tiny Qwen-style output canary. The full-screen L22 canary compares the zero-push row with a second unsteered row in the same batch shape, gates on equality, and records the separate batch-16-versus-alone bf16 effect. The L22 screen checks its residual median against 24.8 within 5%.
4. Verification writes one atomic JSON file per selected neuron, checkpoints after every answer, checks generation and verification manifests plus relative median tolerance, and requires each picked neuron to map to a coherent-persona primary blind label. It skips complete or already generated cases with matching provenance and prints a measured one-neuron runtime with a 20-neuron projection. The runbook documents resumable one-neuron chunks.
5. Timing includes only generation calls after model load. The L22 pilot output declares gate 3 over 9,728 answers and marks limited output `NOT READ`; it logs the raw N2742 answer. Gate 1 and 86.1 are hand-read from that raw answer; the off-world matcher is non-authoritative.
6. Blind items must live apart from their key; the reader instructions require inline pasting into a tool-less session. `picked20.json` is checked against screen answers before verification.
7. Screen manifests include generation-path hashes, per-layer configs, screen-output hashes, and item hash. Blind scoring re-runs triage, checks changed and sample items against their source answers, asserts the full 9,216/512 layer denominators, and records its own scorer manifest.
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

## Fold-1 questions resolved in fold 2

- The zero-push batch-16 row must match a second unsteered row of that same batch shape. The separate batch-16-versus-alone result is recorded as a bf16 batch-effect measurement; pushed-row comparisons remain diagnostic.
- 86.4 checks for a failing read layer first. Any read-layer failure makes the verdict `FAILS`, even when fewer than nine layers are readable; `NOT READ` applies only when no read layer fails and fewer than nine are read.

## What could not be verified here

- The requested venv has CPU torch 2.14.1 and transformers 5.20.0. The strict CPU suite passed with a tiny Qwen-style model, but the Qwen3.5-4B prompt, A770 bfloat16 behavior, L22 median, N2742 answer, runtime canaries, and GPU throughput were not run. The runbook has fail-fast checks for the GPU run; no production model was loaded here.
