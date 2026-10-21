# Part 43 calibration-then-pilot on a rented 3090 (2026-10-21)

Branch part43-singleton at 845964c (Alden #21 PASS, Chorus #11 PASS for one bounded pilot).
Instance 57110030 from offer 56595404 (Quebec, $0.204/h; the California offer 54681934 returned HTTP 400), label part43-pilot, rented 15:53 CT by hand per worktrees/ab-vastrun/PARKED.md. DESTROY before the session ends.

Setup: transformers 5.20.0, flash-linear-attention 0.5.2, torch 2.6.0+cu124, cuDNN 90100, driver 595.99.02, no
causal_conv1d (reference conv path). First start (pilot-try1.log) died in seconds: review/part43/ was not uploaded and
the calibration fingerprints two files in it; uploaded, restarted. Run ended 16:04 CT, rc=0, status HOLD.

**Result: HOLD, calibration epsilon 0.1092 nats > 0.05.** bf16 repeat spread 0 in every cell and clean-to-clean replay
error 0, so the bf16 runs are deterministic; epsilon is entirely the bf16-vs-fp32 difference (largest in the +2x L7
dose cell, 0.109; clean 0.088; all-four-zero 0.021). The pilot did not run. Raw artifact, summary, HOLD and logs kept
as measured (results/).

Destroyed 16:12 CT; GET v1 /instances/ empty. Spend about $0.10 (credit $8.55 -> $8.45).
