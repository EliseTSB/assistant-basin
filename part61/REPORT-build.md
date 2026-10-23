# Part 61 build report

This report was refreshed during the round 1 fold. The current amendments, per-finding results, verification details, and remaining external assumptions are recorded in [REPORT-fold1.md](REPORT-fold1.md).

## Current build

The Part 61 runner measures the registered target and ten fixed controls at 0.25x, 0.5x, 1x, and 2x write sizes in fp32. It validates source records and target scale metadata, writes atomic resumable NPZ/JSON checkpoints, preserves checkpoint memory statistics, and reuses a validated complete scan for analysis-only reruns. The shell runner retries each Hugging Face fetch up to three times.

The analyzer follows Part 59B's pooled P99 rule with no absolute-change floor. It evaluates the Part 61 Amendment 1 rank criteria separately for each model, reports rank and permutation p-value at every size, prints all unit/size residual ratios, and includes the registered 61b/61e base rates without using them as decision inputs. The run and analysis manifests state that 200-document counts are for within-run comparisons and are not comparable to Part 59B's 1,000-document counts. OLMo's 6.5 scale record is marked bf16 while its forward pass is fp32.

## Verification

From the assistant-basin root, the required command was:

```text
/home/elisepc/neuron-probe/.venv/bin/python -m unittest part61.test_part61 -v
```

All 16 tests passed with PyTorch installed. The suite exercised the input-dependent clamp stub and a copied mutation of the real clamp path, a 20,000-draw null simulation, source scale/hash validation, 61a pairing, sign-specific 61d evaluation, complete-scan reuse, checkpoint resume with preserved statistics, and a three-attempt transient fetch. `bash -n part61/run61.sh` and `git diff --check -- part61` also passed. The real OLMo input records validated locally and returned the registered scale 6.5.

The null simulation reproduced the superseded rule's artifact decision rate at 72.125%; the amended OLMo excess-shown rate was 0.640%, below the 1/11 (9.091%) permutation bound. These are synthetic calibration results, not Part 61 model findings.

No model weights were loaded, and no network, API, or GPU operation was performed. Actual model layouts, tokenizer behavior, rented-image package installation, VRAM and throughput remain unverified; see `REPORT-fold1.md` for the full limits.
