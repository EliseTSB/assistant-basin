# Part 43 amendment 3 build report

## Implementation and verification

Implemented Amendment 3 in `part43.py`: four-world aligned receiver swaps for L7→L11, L11→L29, and L7→L29; L11 gate/up/product capture and recombined-product replay; the attachment stimuli and aligned phrase/query measurements; separately measured residual-bypass and full-forward derivatives; current-multiplier and fixed-absolute grids with realized trajectories and clipping; single-ablation residual additive surrogates; eight separate maker answers including Tencent; language, third-person, and company-country prompt families; and template-cluster bootstrap summaries.

The `swap` and `attach` commands are available and included in pilot timing. `score` now fills rows 43.11 and 43.12 and applies the adopted scoring clarifications. The 43.10 score requires a positive minimum slope magnitude supplied to `grid`; it remains NOT SCORED until that value is preregistered.

Verification completed:

- `/home/elisepc/neuron-probe/.venv/bin/python -m py_compile part43.py test_part43.py` — passed.
- `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q test_part43.py` — 10 passed.
- `git diff --check` — passed.
- Tiny-model coverage exercises the new paths, both synthetic four-world cases, attach orders and held-out wording, grid realization/clipping, the additive residual surrogate, separate derivatives, and scoring rows.
- No Qwen forward pass was run in this code-only worker.

## A770 commands

Run from the Part 43 repository directory in Windows Command Prompt. Replace `VALUE` with the preregistered positive minimum slope magnitude in nats per L7 activation unit before running the absolute grid.

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
python part43.py grid --device xpu --dtype bf16 --cache-dir "E:\neuron-probe\hf\hub" --heldout data\part43-heldout.json --corpus data\neutral.txt --minimum-slope-magnitude VALUE --out results\part43
python part43.py swap --device xpu --dtype bf16 --cache-dir "E:\neuron-probe\hf\hub" --out results\part43
python part43.py attach --device xpu --dtype bf16 --cache-dir "E:\neuron-probe\hf\hub" --attach data\part43-attach.json --out results\part43
python part43.py base --device xpu --dtype bf16 --cache-dir "E:\neuron-probe\hf\hub" --out results\part43
python part43.py score --out results\part43 --predictions PREDICTIONS.md
```

The requested WSL CPU weight-only command remains:

```bash
/home/elisepc/neuron-probe/.venv/bin/python part43.py links --weights-only --device cpu --cache-dir /mnt/e/neuron-probe/hf/hub --out results/part43
```

## SUMMARY

Amendment 3 code, attachment stimuli, tiny-model coverage, and A770 command sequence are prepared. The 43.10 minimum slope magnitude still needs registration.

## FINDINGS

All 10 Part 43 tests passed. A repository-wide `pytest -q` collection is not usable as-is because `climb/climb_test.py` parses CLI arguments at import time; the explicit `test_*.py` attempt also imports script-style tests that exit during collection. No real model results were scored here.

## BLOCKERS

Register the positive minimum slope magnitude for 43.10 before running `grid`; otherwise its result is intentionally NOT SCORED. The report includes `VALUE` as a command placeholder. No control-mailbox messages were present.
