# SUMMARY

Addressed the three findings from review C. The confirmatory teacher-forced scoring test now observes real forward batch sizes and proves that a forced-batching sabotage is detected. The pilot records singleton validation evidence instead of depending on the batched-logit diagnostic. The diagnostic remains available as the explicit `checks` command.

# FINDINGS

- `_capture_norm_inputs` now consumes activation rows with a cursor, trims each row to its record's real input length, explicitly runs through singleton mode, and verifies that every record was captured. A TinyModel test covers records of different lengths.
- `run_pilot` now calls `run_singleton_validation` and includes its measured cells in the pilot payload. The CLI writes the same payload to `pilot-validation.json`; no numeric validation budget is applied.
- `run_validation_checks` remains intact with its batched/single comparison, receipts, and tolerances. It is exposed as `checks` and writes `checks.json`. The pilot no longer writes or consumes that receipt as a gate.
- Confirmatory CLI commands continue to reject `--exec-mode batched`; regression coverage checks `validate`, `pilot`, and `signs`. `batch-diag` remains the batched-mode exception.
- `_runtime_environment` continues to record `model.config._attn_implementation` from the loaded model, with a TinyModel test confirming the observed value is carried into the environment record.
- Verification: `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q test_part43.py` → **19 passed in 5.08s**. `git diff --check` passed.

# BLOCKERS

None.
