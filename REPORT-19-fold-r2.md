SUMMARY
- Implemented the provenance redesign and F3/F6/F8, strengthened the F4/F5 tests, and added F7 describe checks.
- Test suites: `test_circuit.py` 37 checks, exit 0 (`PASS`); `test_neuron_report.py` 36 checks, exit 0 (`PASS`); `test_teacherforce.py` 11 checks, exit 0 (`PASS`).
- No XPU or 4B model run; all suites used the tiny model on CPU.

FINDINGS

### F1 / F2 — input provenance
`circuit.py:49-84,127-132,164-180,199-210,245-247` adds `read_input`, which parses and hashes one byte read; headers now take the exact hashes read by that stage. Only `screen` creates a missing answer cache. Every stage checks cache metadata and refuses stale metadata with the mismatching key and a fresh `--out` instruction. `verify` and `group` validate screen answer SHA/model/revision; `edges` and `describe` additionally validate both answer and screen SHAs in verify output. The pre-change stale-answer check printed `FAIL verify refuses a screen made from stale answers and names screen.json`; after the fix the CLI printed `screen.json is stale for answers.json; use a fresh --out or rerun screen` and the row passed. For the verify/read-time replacement case, the round-2 review records the pre-change result as `checked_verify ACCEPTED the verify.json`; after the fix `ok   verify lineage retains screen SHA captured at read time`. Tool SHA refusal now passes for screen, verify, group, edges and describe; each printed `answers.json provenance mismatch (tool_sha256); use a fresh --out` and preserved the cache. A fresh screen → verify → group → edges → describe chain passes in the suite.

### F3 — group seeds
`circuit.py:50` records `[a.seed+s for s in (1,2,3)]`, matching the actual random sets. The review measured the old header as `{'random_controls':5,'group':[1,2,3]}` when seed 5 produced sets 6/7/8. The new test prints `ok   group seeds in the header are derived from the requested base seed` (checks `[6,7,8]`).

### F4 — push finite difference
`test_circuit.py:20-38` converts the model used for finite differences to float32, asserts that dtype, and uses `atol=rtol=5e-3`. The review measured the old loaded dtype as `torch.bfloat16` and a maximum discrepancy of `0.046`; post-change output includes `ok   push finite-difference model is float32` and `ok   push estimate matches finite difference in float32`.

### F5 — zeroing tolerance
`test_circuit.py:114` uses a 25% relative tolerance with a `1e-3` absolute floor. The review measured `screen zeroing estimate ... (0.02090 vs mean |true ΔM| 0.02065)` and noted the old 0.05 tolerance. Post-change output: `ok   screen zeroing estimate magnitude tracks true change within 25%`.

### F6 — edge nulls
`circuit.py:229-242` omits self-effects, restricts each threshold to null sources in lower layers, records `null_sources_used_by_target`, and marks targets with fewer than ten sources `not_testable` with their count. The review measured structural-zero nulls and an example threshold count change from 4 to 3 after upstream-only filtering. Post-change rows print `ok   edges omit self-effects and report upstream null counts`; the tiny-model edge command also passes and writes the new count field.

### F7 — describe coverage
`test_circuit.py:151-175` compares describe’s context rows against per-document activation maxima and compares logit-lens top 10 and z-scores with a full float64 calculation. The review’s untested baseline measurement was 30/30 correct for both describe outputs. Post-change output: `ok   describe logit-lens top 10 tokens and z match full float64 computation` and `ok   describe contexts match unique-document brute-force maxima and order` (30/30 nodes each).

### F8 — atomic writes
`neuron_report.py:57-62` now uses a PID plus random suffix for each temporary file before `os.replace`. The review measured 296 `FileNotFoundError` failures in 600 concurrent writes with the shared `.tmp` name. `test_neuron_report.py` passes after the change; I did not run a dedicated post-change concurrent stress measurement.

BLOCKERS
None. The review packet’s referenced `m1`–`m6` scripts/output files were absent at the supplied location; the review text itself was available and its recorded measurements are identified above. No 4B/XPU run was performed.
