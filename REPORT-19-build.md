SUMMARY
Built `circuit.py` with `screen`, `verify`, `group`, `edges`, and `describe`, plus CPU tests in `test_circuit.py`. The test output ended with `PASS` for all three suites: `test_circuit.py`, `test_neuron_report.py`, and `test_teacherforce.py`.

FINDINGS
- `screen` caches 60-token greedy answers, teacher-forces the fixed token IDs, selects identity tokens by decoded token text, retains gradients at every down-projection input, and writes pooled mean-absolute zeroing estimates, push estimates, rankings, and the N2742 ranks.
- `verify` measures individual zeroing effects against fixed-answer baselines, including the first-20-token control metric. Random controls match the selected layer counts, exclude the top set, and are seeded. Ratios use token-count-weighted mean absolute effects per token.
- `group` generates answers under the top set and random sets for seeds 1–3. `edges` selects the top 30 true pooled effects, measures total effects at identity-token positions and their preceding positions, normalizes by mean absolute baseline activation, and computes per-target null 99th percentiles. Only forward edges are reported significant.
- `describe` follows the Part 18 logit-lens calculation and corpus JSONL context style, restricted to the selected nodes. Outputs include model/revision, tool SHA-256, prompts, seeds, and relevant input hashes.
- The required hand mutation detached a later layer's down-projection input; the finite-difference push check then failed as intended, while the normal gradient check passed. Synthetic tests use an `Alibaba` token because this tokenizer splits `Qwen` into pieces that individually do not match the required identity regex.
- Test output quotes: `test_circuit.py`: `PASS`; `test_neuron_report.py`: `PASS`; `test_teacherforce.py`: `PASS`. No parent-control messages were received.

BLOCKERS
None. The large-model GPU run and 3,000-document production corpus pass were not run here; only CPU tiny-model end-to-end tests were run.
