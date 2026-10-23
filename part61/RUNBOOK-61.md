# Part 61 runbook

Part 61 repeats the Part 59B downstream activation measurement for Qwen3.5-4B L22:2742 and OLMo-2-1B L10:6978, with both models loaded in fp32. It measures the target and ten seeded same-layer controls at four write sizes, using deterministic means over the first 200 documents in the saved Part 26 Pile source record.

## Registered measurement

The candidate pool is the same 20-unit draw used by Part 59, formed with seed 0. Part 59 used the first three sorted units as controls. Part 61 excludes those three and draws ten from the remaining 17 with seed 61. The lists are written into the run manifests and pinned in `test_part61.py`:

- Qwen prior pool: 152, 376, 692, 1613, 2481, 2833, 4638, 4702, 5009, 5160, 5587, 5825, 5859, 5979, 6720, 7487, 7823, 8404, 8617, 8941. New controls: 2833, 4638, 5009, 5587, 5825, 5979, 7487, 7823, 8404, 8941.
- OLMo prior pool: 135, 335, 615, 1433, 2205, 2516, 4121, 4178, 4451, 4585, 4965, 5177, 5205, 5313, 5972, 6653, 6951, 7470, 7660, 7947. New controls: 2516, 4121, 4451, 4965, 5177, 5313, 6653, 6951, 7470, 7947.

At size `s`, the clamp for unit `u` is `sign * s * target_pile_maximum * ||W_down[:, target]|| / ||W_down[:, u]||`. Thus the target at 1x uses its own saved Pile maximum, and every control at 1x has the same column-scaled write norm. The down-projection weight columns are read from the loaded fp32 model. Sizes are 0.25, 0.5, 1.0 and 2.0; both signs are applied at every token position.

The runner records equally weighted per-document means of later-layer MLP down-projection inputs. It also hooks the input to the next decoder layer, computes the valid-token mean residual L2 norm per document, and stores the mean of the clamped/unclamped document-wise ratios for each sign. Prediction 61d averages the two sign estimates for each unit and requires every unit's 1x ratio to fall within 10% of 1. Each model needs one baseline and 88 clamp passes over 200 documents: 17,800 document forwards per model and 35,600 total. The archive checkpoints after the baseline and after each unit/size pair, so interrupted scans can resume.

Analysis uses the pooled 99th percentile of control-by-neuron absolute changes independently for each sign and size, floored at 0.00005. Target counts use all ten controls. Each pseudo-control count uses the other nine controls. It reports per-size counts, pooled Spearman rho, residual-norm ratios and the registered 61a-61e decisions.

## CPU contract tests

From the assistant-basin root, run the CPU-only tests before renting the GPU:

```bash
python3 -m unittest part61.test_part61 -v
```

The test suite uses a stub tokenizer and model to exercise both complete archive writers and the analysis reader without downloading weights or needing a GPU. It checks resume checkpoints, archive key matching, the column-norm clamp formula, the fixed control selections, runner preflight/archive status, and source mutations for sign logic, pooled percentiles, leave-one-out pooling, the null floor and per-unit W_down scaling. Each mutation is compiled from a changed copy of the real source function and must fail its contract assertion.

## Package inputs

From the assistant-basin checkout:

```bash
tar -czf /tmp/part61-inputs.tgz \
  part61 \
  part26.py part20.py part20c.py part20d.py part20f.py neuron_report.py \
  results/part26/source.json results/part26/source.npz \
  results/part26/target-olmo2-1b.npz results/part26/target-olmo2-1b.json \
  results/part54a/source-pile-maxima-7c19d0120476.json
```

The run needs the saved Part 26 source files and target maxima. It does not recapture a corpus, make API calls or generate text. The Qwen maximum is checked against the Part 26 source JSON/NPZ hashes. OLMo's target maximum comes from its saved Part 26 target NPZ and is checked against the target JSON metadata.

## Run on the rented GPU

Upload `/tmp/part61-inputs.tgz` to the rented instance, then:

```bash
mkdir -p /root/part61
cd /root/part61
tar -xzf /root/part61-inputs.tgz
bash part61/run61.sh
```

The runner checks input files, installs the Part 59 package versions (`transformers==5.20.0` and `flash-linear-attention==0.5.2` plus Accelerate and Hugging Face Hub), records package/GPU versions, downloads the two pinned model revisions, then sets offline mode and runs both measurements and analyses. It stores data and logs under `results/part61` and `logs`. On complete success it archives the output as `/root/part61-return.tgz` and prints `RETURN_READY`; on any failed stage it writes `/root/part61-return-INCOMPLETE.tgz`, prints `RUN_FAILED` and exits nonzero. It never destroys the instance.

Copy the matching archive back and inspect/extract it before destroying the instance:

```bash
scp -P SSH_PORT root@SSH_HOST:/root/part61-return.tgz /tmp/part61-return.tgz
# For RUN_FAILED, copy part61-return-INCOMPLETE.tgz instead.
cd /home/elisepc/assistant-basin
tar -xzf /tmp/part61-return.tgz
```

The VM runs each model's analysis and then the combined registered evaluation. The latter prints each of 61a-61e as `held` or `failed` with its observed numbers and writes `results/part61/part61-registered-evaluation.json`. Destroy the VM only after the return archive has been copied and checked.
