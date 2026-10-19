SUMMARY
- Built `part24.py` with `reach`, `assay`, `samples`, and ordered `--steps`; defaults are Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, layer 22, neuron 2742, XPU, and 14.5 GiB spill guard.
- Part 24 CPU suite: 21 checks, exit 0; quoted final output: `PASS: all 21 Part 24 checks passed`.
- `test_neuron_report.py`: exit 0, final output `PASS`. `test_circuit.py`: exit 0, final output `PASS`.
- The requested systemd scope could not start; all three suites were run in the foreground without that memory cap.

FINDINGS

### Shipped

`reach` teacher-forces the six saved Part 19 prompt/answer sequences, records the identity and random units across all layers, clamps N2742 only on answer positions for k={1,2,4}, and leaves k=0 unclamped. It stores answer-token means and changes, name-token activation vectors, per-unit retention, per-question and pooled relative-change summaries, N2742 answer activations, low-layer maxima, and explicit zero-baseline exclusion counts. Name tokens use `circuit.ident_idxs`.

`assay` scores each candidate completion token-by-token for every prompt, stem, k, and zero condition. It writes summed and mean log probabilities, template-bootstrap margins and Effects using `neuron_report._bootstrap` (2,000 resamples, seed 0), and the self-set `Alibaba Cloud.` control. `samples` writes eight unclassified texts for each requested question, k, and zero condition using the Part 18b sampling settings and stored deterministic condition seeds. `--steps reach,assay,samples` loads the model once. Hooks are removed in `finally` blocks.

All files include model/revision, tool SHA, input SHAs, k values, zero conditions, and timestamp. Input validation rejects mismatched reads metadata, target inclusion in either zero set, and overlapping sets. The model/layer/neuron options support the tiny CPU fixture while retaining the requested defaults.

### Interpretations

- Reach's pooled unit ratio uses all answer-token activations from the six questions; per-question values and maximums are also stored so a single prompt cannot be hidden by pooling. Upper-set medians use layers above the configured target layer (23+ with the default layer 22); the low-layer maximum includes layers at or below it.
- Name-token retention pools matching tokens across answers and uses `(a_k · a_0) / (a_0 · a_0)`; per-question retention is also included.
- Reach k=0 is the natural baseline, as the brief states. Assay and samples k=0 are literal zero clamps; samples apply that clamp only on generated-token steps. Zeroing sets apply at every position.
- Assay margins follow Part 18c: average each answer's per-token mean within its ET/earthly class, take the class difference per stem, then average stems and templates. Bootstrap resampling treats templates as units.
- Sample seeds are derived deterministically from the base seed, question, k, and zero condition. Text output contains no classifications.

### Mutation checks

The Part 24 suite printed `ok` for each deliberate break guard:

- `mutation check catches a nonzero reach k=0 relative change`
- `mutation check catches an all-position reach clamp`
- `mutation check catches a missing upper-layer push`
- `mutation check catches a clamp placed one layer too early`
- `mutation check catches an omitted zeroing hook`
- `mutation check catches reversed ET-minus-earthly arithmetic`
- `mutation check catches a nonzero assay Effect at k=0`

The same run printed `ok` for each refusal (N2742 in a zero set, overlapping sets, and reads metadata for another neuron), exact-zero zeroing, hand-computed two-template arithmetic, the answer-only clamp check, and the full three-step CLI output check.

### Test output

- Part 24: `PASS: all 21 Part 24 checks passed` (exit 0).
- Neuron report regression: `PASS` (exit 0).
- Circuit regression: `PASS` (exit 0).
- The attempted wrapper printed `Failed to connect to bus: No medium found`; tests were rerun directly and passed. The existing suites print `PASS` but do not print aggregate check counts.

BLOCKERS

No implementation blocker. The environment has no user systemd bus, so `systemd-run --user --scope -p MemoryMax=3G` was unavailable and the CPU suites were not run under that 3 GiB scope. The default 4B XPU run was not attempted; only the requested tiny CPU-model tests were run.
