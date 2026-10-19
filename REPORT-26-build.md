SUMMARY

- Built `part26.py` with `source`, `target`, `battery`, and `clamp` commands plus ordered `--steps` execution.
- Final Part 26 suite: exit 0; output quoted: `21 passed, 0 failed` and `PASS`.
- Shared suites: `test_neuron_report.py` exit 0 (`PASS`); `test_part20.py` exit 0 (`test_part20: 0 failures`); `test_part25.py` exit 0 (`15 passed, 0 failed`, `PASS`).
- No real model weights were loaded and no Arc/XPU experiment was run.

FINDINGS

### What shipped

`part26.py` selects the first 1,000 qualifying JSONL documents, records zero-based source line indices and the normalized 256-word text, captures the 51 Qwen source neurons' word activations, and writes their normalized top-30 logit-lens strings. Target scans pool token activations by offset overlap, accumulate aligned and whole-document-shifted Pearson sums, chunk the logit lens by units and vocabulary, and save per-neuron document maxima beside each target JSON. The target output includes top-20 rows, writes candidates, null-beating counts, and N2742 twin candidates. Battery scoring reuses Part 20's document selection and Part 20f's suffix scoring. Clamp sampling uses the selected twin, 20 fixed-seed same-layer controls, and an unclamped condition. Hooks are removed in `finally` blocks, and model loading sets Hugging Face offline mode.

The exact final Part 26 run printed:

```text
  ok   Part 26 battery means equal part20f.py on tiny Qwen for all nine sets (max diff 0; filler docs match=True)
21 passed, 0 failed
PASS
```

The shared-suite outputs included:

```text
PASS
test_part20: 0 failures
15 passed, 0 failed
PASS
```

`test_neuron_report.py` ended with `PASS`; its script does not print a check count. The two suites that report counts showed 21 and 15 passing checks respectively. `git diff --check` and Python byte compilation completed with no output.

### Open choices and interpretation

- Part 20f's code selects the entire suffix after the filler, including its fixed `Several of the earlier sentences concerned ` lead. Part 26 follows that exact implementation; the battery regression confirms exact means on a tiny Qwen fixture. This resolves the brief's phrase-token wording in favor of the explicit Part 20f scoring contract.
- The maximum activation is the largest of the word activations, so tokens with empty offsets do not contribute. Degenerate zero-variance Pearson correlations are recorded as zero because Pearson r is undefined for those rows.
- The logit-lens chunks rank raw logits rather than materializing z-scores: subtracting the vocabulary mean and dividing by its positive standard deviation leaves the top-token order unchanged. Only normalized token strings are emitted.
- Clamp sampling uses eight samples, `temperature=0.7`, `top_p=0.95`, and `max_new_tokens=64` from Part 24. A deterministic SHA-derived seed is assigned to each condition; controls are selected with `neuron_report.control_units` and seed 0 by default.
- Source word activations and candidate document maxima exclude special tokens by requiring nonempty offset spans. This keeps the maximum consistent with the word-level definition.

### Deliberate-break checks

- For each of Qwen, OLMo, and Smol tokenizers, a first-overlapping-token reducer was substituted in the fixture; output: `qwen max-pooling regression catches first-token break`, `olmo max-pooling regression catches first-token break`, and `smol max-pooling regression catches first-token break`.
- The null-pairing guard was checked against same-document pairing; output: `misalignment regression catches same-document pairing break`. The correct mapping was `[1, 2, 3, 4, 0]`.
- Counting duplicate spellings instead of set overlap produced `broken list count 5; set overlap 3`; output: `write-overlap regression catches counting duplicate spellings as separate strings`.
- A generation hook that shifted every unit and ignored the requested clamp was caught; output: `clamp regression catches a hook that changes every unit and ignores the set value`.
- A phrase-only battery span was caught by the lead-token fixture: `full suffix max 9.0; phrase-only max 1.0`. A deliberate `+0.5 per set` score offset was also caught by the Part 20f parity guard.

### Memory observation

The final CPU test printed: `baseline 786.2 MiB; peak 795.2 MiB; real Smol batch transient estimate 0.147 GiB (batch 4, 256 tokens, MLP width 11,008)`. This is the tiny-model process RSS during one four-document Smol batch; the 0.147 GiB is a tensor-size estimate for the real batch's input and word-pooling temporaries, not a measurement on the Arc. It excludes model weights and runtime workspaces; nominal 3B bfloat16 parameters add about 5.6 GiB before those buffers.

BLOCKERS

None for the build. The real-model XPU path remains unmeasured here; the requested later run on the Arc has not been performed.
