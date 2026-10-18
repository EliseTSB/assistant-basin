# Part 17 build report

## SUMMARY

Implemented `residual17.py` with capture, FIT-only ridge/labels/quantiles, OLMo self-reader, Qwen input-only reader, evaluation, and natural-value transplant subcommands. Added five offline synthetic tests in `test_residual17.py`; `/home/elisepc/neuron-probe/.venv/bin/python test_residual17.py` reports `Ran 5 tests` / `OK`. Outputs use local Hugging Face cache only and include JSON configuration, split sizes, seeds, timings, and per-group values. No model weights were downloaded or run in this build environment.

## FINDINGS

- `capture` reads `data/wikitext103-train0.parquet` through `pyarrow`, skips empty/title lines, makes non-overlapping 80-token OLMo-tokenizer passages, checks each decoded passage is 40–120 tokens, removes duplicate text hashes, then assigns exact split sizes by ascending passage SHA-256. Default sizes are FIT 4,000 / TRAIN 6,000 / TEST 2,000. `--smoke` uses 12 / 16 / 8.
- G1 comes from `selfreport.select_target_units("A", ...)`; G2–G4 use `selfreport_multi.make_groups(..., seed=7)`. GL is 11 units sampled without replacement from the final OLMo layer with seed 17017. The capture hook reads `down_proj` inputs, the post-SiLU-times-up activations, and never replaces values. Qwen features are the final hidden state averaged over the passage-token span.
- Passage capture presents the passage as one user message. Reader prompts present it as the first user message, add assistant `Okay.`, then ask the design question as the final user message. The assistant bridge follows the existing selfreport prompt pattern. Only the passage span is captured/clamped.
- Ridge uses a centered closed-form solve, chooses among nine trace-scaled alpha candidates on a seeded held-out quarter of FIT, and refits on all FIT. It stores weights, bias, predictions, and residuals. Each group retains the 11 per-unit residuals; the binary label uses their mean, with `higher=1` only when it exceeds that group's FIT-only median. Reported TEST R² is pooled across passages and the 11 units, with mean per-unit R² also included.
- Transplant bounds are per-unit 20th/80th percentiles of passage-level natural means from FIT only. The paired replay duplicates each TEST prompt in one batch and clamps all group units in the passage span. Accuracy and transplant metrics compare the next-token probabilities for `higher` and `lower`.
- Full defaults: LoRA rank 8 on attention q/k/v/o, batch 8, 3 epochs, learning rate 3e-4; smoke training uses one epoch unless `--epochs` overrides it. The shuffled run writes a `-shuffled` adapter and JSON name so it does not overwrite the ordinary run.

### Windows smoke commands

Run from `C:\Users\alexa\neuron-probe-gpu` with the small corpus file and both models already in the HF cache:

```powershell
python residual17.py capture --corpus data\wikitext103-train0.parquet --identity results\part2-olmo2-1b\identity-instruct-chat.json --out-dir results\part17-smoke --device cpu --smoke
python residual17.py fit --out-dir results\part17-smoke --device cpu --smoke
python residual17.py train-self --group G1 --out-dir results\part17-smoke --device cpu --smoke
python residual17.py eval-self --group G1 --out-dir results\part17-smoke --device cpu --smoke
python residual17.py train-input --group G1 --out-dir results\part17-smoke --device cpu --smoke
python residual17.py eval-input --group G1 --out-dir results\part17-smoke --device cpu --smoke
python residual17.py transplant --group G1 --out-dir results\part17-smoke --device cpu --smoke
```

### Windows full commands

Run from `C:\Users\alexa\neuron-probe-gpu` after copying the parquet to `data\wikitext103-train0.parquet` and caching both models:

```powershell
python residual17.py capture --corpus data\wikitext103-train0.parquet --identity results\part2-olmo2-1b\identity-instruct-chat.json --out-dir results\part17 --device xpu
python residual17.py fit --out-dir results\part17 --device cpu

$groups = @('G1', 'G2', 'G3', 'G4', 'GL')
foreach ($g in $groups) { python residual17.py train-self --group $g --out-dir results\part17 --device xpu --epochs 3 }
foreach ($g in $groups) { python residual17.py eval-self --group $g --out-dir results\part17 --device xpu }
foreach ($g in $groups) { python residual17.py train-input --group $g --out-dir results\part17 --device xpu --epochs 3 }
foreach ($g in $groups) { python residual17.py eval-input --group $g --out-dir results\part17 --device xpu }
foreach ($g in $groups) { python residual17.py transplant --group $g --out-dir results\part17 --device xpu }

python residual17.py train-self --group G1 --out-dir results\part17 --device xpu --epochs 3 --shuffle-labels --run-tag shuffled
python residual17.py eval-self --group G1 --out-dir results\part17 --device xpu --run-tag shuffled
```

### Batch and GPU-time estimate

The full configuration runs 3,000 OLMo capture batches (12,000 / 4), 1,500 Qwen feature batches (12,000 / 8), then 2,250 training steps per reader/group (6,000 / 8 × 3 epochs). Each reader/group TEST pass is 500 batches; each transplant/group pass is 500 batches of eight duplicated prompts. FIT performs no GPU work. Smoke counts are 9 OLMo batches, 5 Qwen batches, two training steps, two evaluation batches, and two transplant batches.

Planning estimates from the existing Part 12 artifacts are rough: `results/part12e/A.json` records about 248 seconds for natural capture, or about 3.4 seconds per four-row batch at its recorded workload, which extrapolates to roughly 2.9 hours for OLMo capture. Scaling that proxy by the Qwen/OLMo parameter-count ratio gives about 50 minutes for Qwen features; Qwen has not been timed for this task. `REPORT-12g-build.md` used about 10.4 seconds per A770 training step, so budget about 6.5 hours per OLMo reader/group at 2,250 steps. Treating that as a conservative same-speed proxy for Qwen gives the same upper estimate; actual Qwen timing should be taken from `mean_step` in its training JSON. A prior selfreport evaluation proxy is about 2.3 seconds per four-row batch: roughly 19 minutes per OLMo group evaluation and 38 minutes per group transplant with doubled rows. These projections are sensitive to sequence lengths, XPU kernels, and cache state.

## BLOCKERS

No code or test blocker. The model-backed smoke and full runs remain to be executed on the Windows A770 host with its local model cache and copied parquet; this worker run did not have that host workload available.
