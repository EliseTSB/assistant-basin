# Part 43 build report

## Implementation and verification

Built `part43.py` around Part 25 sequences, token sets, margin scoring and intervention hooks, with Part 39's direct streaming model loader. The CLI includes signed activations, direct/J-lens readouts, the amended lattice, path-specific rescue, activation shifts, contextual links, the factorial sweep, dose curves, positions, baselines, held-out prompts, the L7/L11 grid, validation, pilot, Base replication and scoring.

The output schema keeps the primary plain maker cell separate from bold and name-stem cells. The full lattice saves all 16 cells, raw candidate-token logits, aligned activation arrays, natural/off Walsh coefficients, conditional contrasts, order energy, exact empirical p values and output-only surrogates using frozen and real final RMSNorm. Greedy maker and self-name text is saved for each target subset. Rescue reports T, R, recovery, uncapped fractions, replay controls and the no-op eligibility rule. The manifest records the model, tokenizer and chat template, exact prompts and aliases, hook sites, dtype, seeds, matching rules and NOT_TESTABLE rules.

The RMSNorm scale convention was checked against `transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py:3530-3536`, where Qwen3.5 applies `1 + weight`; the decoder calls `post_attention_layernorm` before the MLP at lines 3624-3628.

Verification completed:

- `/home/elisepc/neuron-probe/.venv/bin/python -m py_compile part43.py test_part43.py`
- `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q test_part43.py` — 9 passed.
- The tests exercise every CLI command's parser entry, tiny-model paths for the analysis methods, the pilot and Base command paths, the additive-logit/nonzero-score-interaction fixture, and the real Part 25 candidate token sets.
- Offline manifest creation succeeded; its script hash matches the final `part43.py` and it includes 40 sweep prompts.
- The WSL CPU weight-only link command completed for six ordered pairs and 1,000 null units per earlier layer. Maximum RSS was 334,920 KB; elapsed time was 37.35 seconds. Output: `results/part43/links-weights.json`.
- `git diff --check` passed.

The amendment messages were applied: seq 1 changed the primary score, matched nulls, held-out set, rescue and readout/link methods; seq 2 added continuous dose curves and the L7/L11 grid; seq 3 added raw logits, additive surrogates, Walsh statistics, decoding, reader-review outcomes and required validation. All three were disposed as considered/applied. No mailbox messages remain.

## Forward-pass commands for the Windows GPU host

Run from the Part 43 repository directory. The manifest command records the run inputs before the pilot; the pilot also refreshes it automatically.

```bat
python part43.py manifest --device xpu --dtype bf16 --cache-dir "E:\neuron-probe\hf\hub" --out results\part43
python part43.py pilot --device xpu --dtype bf16 --cache-dir "E:\neuron-probe\hf\hub" --out results\part43 --candidate-pool-size 1024
python part43.py signs --device xpu --dtype bf16 --cache-dir "E:\neuron-probe\hf\hub" --out results\part43
python part43.py lattice --device xpu --dtype bf16 --cache-dir "E:\neuron-probe\hf\hub" --out results\part43 --candidate-pool-size 1024
python part43.py mediate --device xpu --dtype bf16 --cache-dir "E:\neuron-probe\hf\hub" --out results\part43
python part43.py shift --device xpu --dtype bf16 --cache-dir "E:\neuron-probe\hf\hub" --out results\part43
python part43.py positions --device xpu --dtype bf16 --cache-dir "E:\neuron-probe\hf\hub" --out results\part43
python part43.py baselines --device xpu --dtype bf16 --cache-dir "E:\neuron-probe\hf\hub" --corpus data\neutral.txt --out results\part43
python part43.py heldout --device xpu --dtype bf16 --cache-dir "E:\neuron-probe\hf\hub" --out results\part43
python part43.py links --device xpu --dtype bf16 --cache-dir "E:\neuron-probe\hf\hub" --out results\part43
python part43.py readout-check --device xpu --dtype bf16 --cache-dir "E:\neuron-probe\hf\hub" --jlens "E:\neuron-probe\jlens\qwen3.5-4b\jlens\Salesforce-wikitext\Qwen3.5-4B_jacobian_lens.pt" --checksums "E:\neuron-probe\jlens\SHA256SUMS" --out results\part43
python part43.py dla --device xpu --dtype bf16 --cache-dir "E:\neuron-probe\hf\hub" --signs results\part43\signs.json --jlens "E:\neuron-probe\jlens\qwen3.5-4b\jlens\Salesforce-wikitext\Qwen3.5-4B_jacobian_lens.pt" --checksums "E:\neuron-probe\jlens\SHA256SUMS" --out results\part43
python part43.py sweep --device xpu --dtype bf16 --cache-dir "E:\neuron-probe\hf\hub" --prompts data\part43-sweep.json --out results\part43
python part43.py dose --device xpu --dtype bf16 --cache-dir "E:\neuron-probe\hf\hub" --baseline results\part25\means.json --heldout data\part43-heldout.json --corpus data\neutral.txt --out results\part43
python part43.py grid --device xpu --dtype bf16 --cache-dir "E:\neuron-probe\hf\hub" --heldout data\part43-heldout.json --corpus data\neutral.txt --out results\part43
python part43.py base --device xpu --dtype bf16 --cache-dir "E:\neuron-probe\hf\hub" --out results\part43
python part43.py score --out results\part43 --predictions PREDICTIONS.md
```

The requested WSL CPU weight-only command is:

```bash
/home/elisepc/neuron-probe/.venv/bin/python part43.py links --weights-only --device cpu --cache-dir /mnt/e/neuron-probe/hf/hub --out results/part43
```

## SUMMARY

The amended build, tests, manifest and real CPU weight-alignment output shipped. The full Qwen forward-pass run is prepared for the A770 commands above.

## FINDINGS

The tests passed, and the WSL weight scan stayed well below the 4 GB RSS ceiling. No Part 43 model effects or prediction rows were scored in this build.

## BLOCKERS

The Part 43 signs output does not exist yet, so the real DLA readout cannot run until the A770 `signs` command produces `results/part43/signs.json`. All forward-pass sections were deferred to the A770 per the brief; no model forward pass was run in this worker.
