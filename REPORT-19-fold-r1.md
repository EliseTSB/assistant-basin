SUMMARY
- `test_circuit.py`: 25 `ok` checks, final output `PASS` (exit 0).
- `test_neuron_report.py`: 34 `ok` checks, final output `PASS` (exit 0).
- `test_teacherforce.py`: 10 `ok` checks, final output `PASS` (exit 0).
- Mutants all failed the suite: M2 exit 1 (screen gradient mismatch), M3 exit 1 (push estimate mismatch), M4 exit 1 (normalisation assertion), M5 exit 1 (verify ratio assertion).
- CPU tiny-model checks only; no GPU, 4B run, or bf16 numerics were performed.

FINDINGS
- F1 — Added checks for the real screen output: push values against the screen gradient path, and zeroing magnitude against mean true zeroing changes across the six synthetic identity answers (`test_circuit.py:93-112`). Added constructed edge normalisation, 99th-percentile, and verify-ratio checks (`test_circuit.py:71-72, 117-118`). M2–M5 mutation output included: `== M3 push estimate line removed from screen: exit 1`, `FAIL screen push estimate matches the real screen gradient path (0.00000 vs 0.12584)`; M4 failed `edge normalisation divides the mean change by baseline mean absolute activation`; M5 failed `verify ratio is identity per-token mean divided by control per-token mean`. M2 failed the gradient-path check with `(-0.64728 vs -0.74725)`. The original independent finite-difference push test also passes in the unmutated suite.
- F2 — Edge effects average only valid identity prompts, and null thresholds use those resulting effects (`circuit.py:134-135, 201-207`). Constructed check confirms `[2, 0, 4]` with the middle prompt invalid averages to `3` (`test_circuit.py:73`).
- F3 — `screen()` exits with clear messages when all identity prompts are invalid or when “Who made you?” is invalid (`circuit.py:110-111`). Both CLI cases passed (`test_circuit.py:84-90`), with messages `screen: no identity prompt produced a valid identity token; refusing to write rankings` and `screen: "Who made you?" has no valid identity token; refusing to write push ranking`.
- F4 — `answers.json` wraps generated records with model, revision, identity/control prompts, and the tool SHA-256 (`circuit.py:52-70`). Provenance mismatch prints `answers.json provenance mismatch; regenerating answers`; the test confirmed revision `rev-b` was recorded (`test_circuit.py:136-139`).
- F5 — `edges()` and `describe()` call `checked_verify()`, comparing verify input hashes to current `answers.json` and `screen.json`; mismatches exit with instructions to rerun verify (`circuit.py:181-187, 189, 224`). The stale-screen test passed (`test_circuit.py:126-130`).
- F6 — Describe corpus capture calls `model.model` (base decoder), retains only the top 10 activations per node across unique documents, computes logit-lens scores in 8192-row blocks without a persistent float32 unembedding copy, and re-runs the spill guard after each batch (`circuit.py:230-269`). The tiny corpus describe test passed. No 4B memory measurement was run.
- F7 — Kept `--seed`; verify uses `a.seed`, while group controls use `a.seed + 1`, `+2`, and `+3`. Default seed 0 therefore preserves verify 0 and group 1, 2, 3 (`circuit.py:146, 162, 166`).
- F8 — Pre-existing `neuron_report._dump` shared `.tmp` race left unchanged, as instructed.

BLOCKERS
- None. GPU/XPU execution and 4B/bf16 measurements remain unverified; the requested tiny CPU suites passed.
