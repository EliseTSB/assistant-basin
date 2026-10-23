# Part 61 runbook

Part 61 repeats the Part 59B downstream activation measurement for Qwen3.5-4B L22:2742 and OLMo-2-1B L10:6978, with both forward passes in fp32. It measures the target and ten seeded same-layer controls at four write sizes, using deterministic means over the first 200 documents in the saved Part 26 Pile source record. The 200-document counts are not comparable to Part 59B's 1,000-document counts; analysis reports within-run comparisons only. OLMo's target scale is 6.5 from a Part 26 record made in bf16, and that source dtype is recorded in the manifest and analysis.

## Registered measurement

The candidate pool is the same 20-unit draw used by Part 59, formed with seed 0. Part 59 used the first three sorted units as controls. Part 61 excludes those three and draws ten from the remaining 17 with seed 61. The lists are written into the run manifests and pinned in `test_part61.py`:

- Qwen prior pool: 152, 376, 692, 1613, 2481, 2833, 4638, 4702, 5009, 5160, 5587, 5825, 5859, 5979, 6720, 7487, 7823, 8404, 8617, 8941. New controls: 2833, 4638, 5009, 5587, 5825, 5979, 7487, 7823, 8404, 8941.
- OLMo prior pool: 135, 335, 615, 1433, 2205, 2516, 4121, 4178, 4451, 4585, 4965, 5177, 5205, 5313, 5972, 6653, 6951, 7470, 7660, 7947. New controls: 2516, 4121, 4451, 4965, 5177, 5313, 6653, 6951, 7470, 7947.

At size `s`, the clamp for unit `u` is `sign * s * target_pile_maximum * ||W_down[:, target]|| / ||W_down[:, u]||`. Thus the target at 1x uses its own saved Pile maximum, and every control at 1x has the same column-scaled write norm. The down-projection weight columns are read from the loaded fp32 model. Sizes are 0.25, 0.5, 1.0 and 2.0; both signs are applied at every token position.

The runner records equally weighted per-document means of later-layer MLP down-projection inputs. It also hooks the input to the next decoder layer, computes the valid-token mean residual L2 norm per document, and stores the mean of the clamped/unclamped document-wise ratios for each sign. Prediction 61d checks both 1x sign ratios separately for every unit; all 22 values per model must be within [0.9, 1.1]. Each model needs one baseline and 88 clamp passes over 200 documents: 17,800 document forwards per model and 35,600 total. The archive checkpoints after the baseline and after each unit/size pair, so interrupted scans resume with their memory statistics. A valid complete scan is reused on rerun and proceeds directly to analysis.

Analysis uses the pooled 99th percentile of control-by-neuron absolute changes independently for each sign and size, with no absolute-change floor, matching Part 59B. `0.00005` is recorded as the registered null pass probability. Target counts use all ten controls. Each pseudo-control count uses the other nine controls. Prediction 61a's size effect is Spearman rho across the ten controls only, pooled over sizes. At each size, the report prints the target count, all control counts, target's conservative upper rank, and exact one-sided permutation p-value (a control tied with the target counts against it). The excess is shown only at rank 1 at both 1x and 2x, absent at rank 3 or lower at 1x, and otherwise inconclusive. “Absent” means no evidence of excess over matched controls, not proof that there is none. Each model is decided separately: an artefact requires both 61a and absent excess; shown excess means the contrast survives for that model. Predictions 61b and 61e are reported with null chance base rates 10/11 and 1/2 and do not enter decisions. The analyzer prints residual-norm ratios for every unit, sign, and size plus the chance table for decision outcomes.

## CPU contract tests

From the assistant-basin root, run the CPU tests in the supplied PyTorch virtualenv before renting the GPU:

```bash
/home/elisepc/neuron-probe/.venv/bin/python -m unittest part61.test_part61 -v
```

The test suite uses an input-dependent stub model to exercise the real clamp hook, complete archive writers, analysis reader, reruns and checkpoint resume without downloading weights or needing a GPU. It also checks OLMo source/scale record binding, all 61d sign ratios, 61a size-count pairing, the fixed control selections, runner preflight/archive status and a three-attempt transient fetch retry. Mutation probes edit copies of real source functions and must go red when the clamp path, size pairing, pooled classifier, sign rule, percentile, leave-one-out pool, no-floor rule or W_down scaling is sabotaged.

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

The run needs the saved Part 26 source files and target maxima. It does not recapture a corpus, make API calls or generate text. The Qwen maximum is checked against the Part 26 source JSON/NPZ hashes. OLMo's target maximum is accepted only when its target JSON's source hashes, layer widths, target row, and maximum agree with the companion NPZ and the Part 61 source JSON/NPZ inputs.

## Run on the rented GPU

Upload `/tmp/part61-inputs.tgz` to the rented instance, then:

```bash
mkdir -p /root/part61
cd /root/part61
tar -xzf /root/part61-inputs.tgz
bash part61/run61.sh
```

The runner checks input files, installs the Part 59 package versions (`transformers==5.20.0` and `flash-linear-attention==0.5.2` plus Accelerate and Hugging Face Hub), records package/GPU versions, downloads the two pinned model revisions (up to three attempts per fetch), then sets offline mode and runs both measurements and analyses. It stores data and logs under `results/part61` and `logs`. On complete success it archives the output as `/root/part61-return.tgz` and prints `RETURN_READY`; on any failed stage it writes `/root/part61-return-INCOMPLETE.tgz`, prints `RUN_FAILED` and exits nonzero. It never destroys the instance.

Copy the matching archive back and inspect/extract it before destroying the instance:

```bash
scp -P SSH_PORT root@SSH_HOST:/root/part61-return.tgz /tmp/part61-return.tgz
# For RUN_FAILED, copy part61-return-INCOMPLETE.tgz instead.
cd /home/elisepc/assistant-basin
tar -xzf /tmp/part61-return.tgz
```

The VM runs each model's analysis and then a combined summary of the independent per-model decisions. The latter prints each of 61a-61e as `held` or `failed` with its observed numbers and writes `results/part61/part61-registered-evaluation.json`. Destroy the VM only after the return archive has been copied and checked.
