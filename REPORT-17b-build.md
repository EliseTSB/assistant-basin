# Part 17b build report

## SUMMARY

Implemented `textfit17b.py fit-text` to reuse the Part 17 capture and produce a separate Part 17b fit directory. The full run uses five deterministic passage-hash folds over all capture passages, Qwen3-1.7B in bf16, rank-8 LoRA, and a linear head for G1 and GL by default. Each fold standardizes natural group means from its training passages only; held-out predictions are de-standardized before residuals are made. FIT passages alone set label medians and transplant quantiles.

## FINDINGS

- Text is passed directly to the tokenizer with `add_special_tokens=False`; no chat template is used. The regression features concatenate final hidden state at the last passage token and the masked mean over all passage tokens.
- G1 and GL share one fold-specific LoRA model with a two-output linear head. The head predicts each requested group's mean across its 11 natural activation units. Predictions are expanded across those 11 units to preserve Part 17's residual array shape.
- `fit.npz` stores passage IDs, fold IDs, predictions, residuals, labels, FIT-derived thresholds, and FIT-derived `low20` / `high80` bounds. `fit.json` stores OOF R² over all passages and TEST for each requested group.
- `residual17.py` accepts `--fit-dir` on reader and transplant commands, checks passage order, and refuses groups without a text predictor. Existing single-directory behavior remains the default.
- Verification passed: `test_textfit17b.py` (5 tests) and `test_residual17.py` (13 tests); `git diff --check` was clean.
- Default batch size is 2. The 1.7B bf16 weights use about 3.4 GB; batch 2 leaves room for training activations and the small LoRA/head state on the A770's 16 GB. Full `fit-text` entails about 72,000 optimizer updates (12,000 passages, five folds, three epochs, batch 2). Coarse estimate, not measured: about 8–24 GPU hours for `fit-text`, plus roughly 3–8 hours for the reader and transplant sequence.

### Windows commands

Run these in PowerShell from `C:\Users\alexa\neuron-probe-gpu`.

Smoke the text predictor on at most ten passages from each capture split. This writes a subset fit in a separate directory; run the full fit before the reader sequence.

```powershell
.\.venv\Scripts\python.exe .\textfit17b.py fit-text --out-dir results\part17 --fit-out-dir results\part17b-smoke --device xpu --smoke
```

Full sequence, reusing Part 17's capture and putting the new fit at `results\part17b`:

```powershell
.\.venv\Scripts\python.exe .\textfit17b.py fit-text --out-dir results\part17 --fit-out-dir results\part17b --device xpu --groups G1 GL
.\.venv\Scripts\python.exe .\residual17.py train-input --out-dir results\part17 --fit-dir results\part17b --group G1 --device xpu
.\.venv\Scripts\python.exe .\residual17.py eval-input --out-dir results\part17 --fit-dir results\part17b --group G1 --device xpu
.\.venv\Scripts\python.exe .\residual17.py train-self --out-dir results\part17 --fit-dir results\part17b --group GL --device xpu
.\.venv\Scripts\python.exe .\residual17.py eval-self --out-dir results\part17 --fit-dir results\part17b --group GL --device xpu
.\.venv\Scripts\python.exe .\residual17.py train-self --out-dir results\part17 --fit-dir results\part17b --group G1 --device xpu
.\.venv\Scripts\python.exe .\residual17.py eval-self --out-dir results\part17 --fit-dir results\part17b --group G1 --device xpu
.\.venv\Scripts\python.exe .\residual17.py transplant --out-dir results\part17 --fit-dir results\part17b --group G1 --device xpu
```

The first two reader commands are gate (a); the GL self-reader commands are gate (b). The self-reader and transplant commands then run for G1.

## BLOCKERS

None for implementation. The Qwen model training path was not run here; the installed test suite uses synthetic data and a fake fold predictor, so the XPU/bf16 runtime and GPU-time estimate remain unmeasured.

