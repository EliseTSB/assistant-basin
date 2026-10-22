# Part 57 build report

## SUMMARY

- Added a vendored jlens package with explicit CPU/XPU placement, a deterministic OLMo fit runner, a model-aware unit reader, and tests.
- Generated the requested Qwen3.5-4B and Qwen3-1.7B reads plus fixed-seed Wikitext row manifests.
- The OLMo lens was not fitted. Its smoke/full-fit commands are in \`results/part57/steps.txt\`; this worker did not launch the Arc, per parent control.

## FINDINGS / WHAT SHIPPED

### Implementation

- \`part42.py\` now resolves the pinned OLMo snapshot and the Part 57 lens path, and handles OLMo2's direct RMSNorm scale.
- \`jlens_units.py\` accepts \`--model-name olmo2-1b\`; by default it reads L10:N6978 and ten L14 controls (Part 26 positive control L14:N3027 plus nine seed-5701 random units).
- \`fit_jlens_olmo.py\` selects fixed Wikitext rows whose OLMo tokenization is exactly 128 tokens, records indices and corpus SHA256, fits incrementally with \`JacobianLens.merge\`, supports CPU/XPU, and guards Windows system RAM below 4 GB.
- Vendored upstream jlens under \`third_party/jlens-part57\` (Apache-2.0), patching \`from_hf(..., device=...)\` to place models explicitly. Upstream source: [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens).
- OLMo cache validation: revision \`48d788eca847d4d7548f375ad03d3c9312f6139e\`, 16 layers, d_model 2048, \`model.\` tensor prefix, down-projection width 8192. The fitted-lens serialization is the same \`J\` / \`n_prompts\` / \`source_layers\` / \`d_model\` layout that \`part42.load_jacobian_lens\` reads.

### Results and predictions

- \`results/part57/qwen3.5-4b/read.json\`: Qwen3.5-4B L22:N2742. Its J-lens top 30 match 5 of the 9 57a words (alien, aliens, space, planet, extraterrestrial); its bottom 30 match 0 of the 6 57b words. **57c passes**.
- \`results/part57/qwen3-1.7b/read.json\`: Qwen3-1.7B L19:N4826 and L22:N1610 reference reads.
- 57a, 57b, and 57d are unscored because no OLMo lens was fitted. 57e is unscored because neither the 100-prompt nor 1,000-prompt OLMo lens was fitted.
- Row selections are recorded in \`results/part57/olmo2-1b/fit-selection-100.json\` and \`fit-selection-1000.json\`. Seed 5701 produced 100 and 1,000 rows with the first 100 nested in the 1,000-row set; every selected row tokenizes to 128 tokens. Corpus SHA256: \`74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212\`.
- Intended lens paths: 10-row smoke \`results/part57/olmo2-1b/smoke-10x2.pt\`; 100-row lens \`results/part57/olmo2-1b/jacobian_lens.pt\`; 1,000-row lens \`results/part57/olmo2-1b/jacobian_lens-1000.pt\`. **None exists yet.**

### Exact commands and timings

Times are wall-clock values reported by \`/usr/bin/time\` or the tool runner.

- System check: \`free -g && nproc\` — 16 CPUs; 11 GB RAM total, 8 GB available, 3 GB swap with 2 GB used.
- Isolated package setup: \`python3 -m venv --system-site-packages .jlens-part57-venv\` — 1.58 s. Shell network install,
  \`SSL_CERT_FILE=/home/elisepc/.mitmproxy/ca-bundle-combined.crt .jlens-part57-venv/bin/python -m pip install --no-deps 'git+https://github.com/anthropics/jacobian-lens.git'\`,
  failed in 0.22 s because \`github.com\` did not resolve. The Apache-licensed source was vendored from upstream and
  \`.jlens-part57-venv/bin/python -m pip install --no-deps --no-build-isolation ./third_party/jlens-part57\`
  succeeded in 0.89 s. For CPU tests, the new venv read torch and other dependencies from the existing environment via
  \`PYTHONPATH=/home/elisepc/neuron-probe/.venv/lib/python3.12/site-packages\`; that shared environment was not modified.
- CPU tests:
  \`PYTHONPATH=/home/elisepc/neuron-probe/.venv/lib/python3.12/site-packages .jlens-part57-venv/bin/python -m pytest -q test_jlens_units.py\`
  — 3 passed in 4.10 s (5.20 s elapsed including startup). The hand-built lens returned the known top/bottom tokens; the tiny random Llama fit saved, loaded through Part 42, and read on CPU.
- Test sabotage: changed the known-answer expectation to \`sabotaged-token\`; pytest returned 1 as expected and showed the mismatch. The test file was restored; the negative-control run took 4.14 s.
- Qwen reads:
  \`/usr/bin/time -f 'elapsed=%e sec' /home/elisepc/neuron-probe/.venv/bin/python jlens_units.py --model-name qwen3-1.7b --units 19:4826 22:1610 --out results/part57/qwen3-1.7b/read.json\`
  — 8.70 s.
  \`/usr/bin/time -f 'elapsed=%e sec' /home/elisepc/neuron-probe/.venv/bin/python jlens_units.py --model-name qwen3.5-4b --units 22:2742 --out results/part57/qwen3.5-4b/read.json\`
  — 13.92 s.
- Row manifests:
  \`/home/elisepc/neuron-probe/.venv/bin/python fit_jlens_olmo.py --dry-run --n-prompts 100 > results/part57/olmo2-1b/fit-selection-100.json\`
  — 6.22 s.
  \`/home/elisepc/neuron-probe/.venv/bin/python fit_jlens_olmo.py --dry-run --n-prompts 1000 > results/part57/olmo2-1b/fit-selection-1000.json\`
  — 7.36 s.
- OLMo CPU trials were stopped before a prompt completed:
  \`/home/elisepc/neuron-probe/.venv/bin/python fit_jlens_olmo.py --device cpu --dtype bf16 --smoke --dim-batch 8 --threads 4 --checkpoint-every 10\`
  — interrupted after about 2.5 minutes without completing the first prompt.
  Retried with \`--dim-batch 32\` — interrupted after about 90 seconds; available system memory fell to 4 GB and swap reached 3/3 GB. After stopping, available RAM returned to 8 GB. These trials provide no completed OLMo fit time.

### Windows XPU steps

Parent control message seq 1 (\`use-the-arc\`) was considered and applied: use XPU for the real fit, prepare the Windows commands, and do not launch the Arc from this worker. The runnable command lines, including 10-row smoke, 100-row fit, 1,000-row comparison fit, and reads, are in \`results/part57/steps.txt\`. Sync the listed scripts and vendored package to the Windows runner first. No message was declined or deferred; no 3090 command was produced under the updated instruction.

## BLOCKERS

- The actual OLMo smoke, 100-row fit, 1,000-row fit, OLMo reads, and prediction scores 57a/57b/57d/57e remain unproduced. The CPU fit was too slow and memory-heavy; the parent explicitly prohibited this worker from launching the Arc.
- The shell also cannot resolve GitHub/Hugging Face. The fit script now sets offline mode and uses the cached OLMo model; the Windows runner must execute the XPU steps.
