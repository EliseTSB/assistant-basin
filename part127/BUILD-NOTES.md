# Part 127 build notes

## Programs and outputs

`run127.py` has one batch path for singles, A+B, A+C, random directions, and no push. It writes each generated row once. A single answer is shared by every pair that uses that unit. Run keys are pair, arm, and question keys. A completed run can be resumed without generating saved answers again.

The main inventory has 1,914 answers: 288 singles, 324 A+B, 324 A+C, 972 random controls, and 6 plain answers. The fixed-path read has 2,232 logical paths: 96 no-push, 192 half-dose singles, 648 A+B, 324 A+C, and 972 random. Pilot runs cover two pairs and save 198 answers plus 216 fixed paths. The main run imports those rows when `--pilot-dir` names a completed pilot with the same generation batch size.

Generation batch size is chosen from pilot runs at 8 and 64. The rig gate and every generation arm use that same batch size. Fixed-path forwards always use batch 8, as registered in the design. The runner saves cosine values before it normalizes the push columns. It prints `P127 ...` progress lines and ends with a `PART127_RUN_COMPLETE` line containing the stage and inventory signature.

Every generation call is appended to `generation_calls.jsonl`. Each actual row records its unit list, requested dose, alpha, layer list, and hook state. The log separates actual rows from padding rows. The rig gate writes its plain and zero-hook generation calls too. Fixed-path forwards have their own call records.

`label127.py` has three staged commands. `gate` labels the 12 known examples first. Each reader must pass 10 of 12. `controls` labels the full-dose and no-push items, then prints eligible pairs before study labels are read. `study` labels all DeepSeek items and the registered Luna sample. Reader prompt text contains only target, description, question, and answer. Results are cached by opaque item id and answer hash. Truncated responses are retried once and remain errors if still truncated. DeepSeek uses thinking off and a 16,000 token cap. Luna uses a 16,000 token cap.

`score127.py` refuses incomplete captures. It checks the run, gate, answer, reader, and fixed-path tables before scoring. It reports five predictions, raw labels by pair and question, rescues for A+B, A+C, and all three random arms at both doses, reader-specific gates, cosines, broken answers, and pair bootstrap intervals. It writes `SCORE.md` and `score127.json`.

## Imports under `python -I`

The scripts add their own `part127/` folder, repository root, and only the source folders they import. Run them by script path from the repository root. Do not add `ROOT / "part85"` to `sys.path`. The repository root must be present so `part85.checkpoints` imports as a package. A path named `part85` would shadow that package.

## Files for the Windows job folder

Run these commands from the repository root. `launch_arc.py` copies the listed files at their relative paths before it starts the Windows job. The local model is pinned to revision `48d788eca847d4d7548f375ad03d3c9312f6139e`.

```bash
PART127_RUNTIME=(
  part127/run127.py part127/label127.py part127/score127.py
  part92/common92.py part92/label92.py
  part92b/label92b.py part92b/known92b.jsonl
  part119/run119.py part119/rows119.py
  part119b/run119b.py part119b/rows119b.py part119b/gate119b.py
  part119c/run119c.py part119c/rows119c.py
  part120/run120.py part85/checkpoints.py
  judge_client.py results/part92/step2/neurons.json
)
```

The runner imports the registered Part 92 push helpers, Part 119 row hooks and logit support, and Part 120 centered-logit support. It does not copy or reimplement those functions.

## A770 pilot and main launches

Check that the A770 is idle and that Windows has about 7.5 GB of free system RAM before model load. The 1B bf16 weights are about 2 GB before runtime state, KV cache, and activations. Batch 64 can use more accelerator memory, so compare its pilot against batch 8 before selecting it. The fixed-path stage remains at batch 8 either way. Run one job at a time. The design reserves at most one GPU-hour for both pilot sizes and the main run.

```bash
tools/launch_arc.py 127 \
  --step "-I -B part127/run127.py --stage pilot --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --out results\\part127\\pilot-b8 --batch 8 --device xpu" \
  --copy "${PART127_RUNTIME[@]}"

tools/launch_arc.py 127 \
  --step "-I -B part127/run127.py --stage pilot --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --out results\\part127\\pilot-b64 --batch 64 --device xpu" \
  --copy "${PART127_RUNTIME[@]}"
```

Inspect both pilot `run_meta.json` files, the rig gate, the 198 saved answers, and the elapsed progress lines. Choose the faster batch that passes. Use the matching pilot for the main launch. These two main commands are exact alternatives; run one.

```bash
tools/launch_arc.py 127 \
  --step "-I -B part127/run127.py --stage main --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --out results\\part127\\main --pilot-dir results\\part127\\pilot-b8 --batch 8 --device xpu" \
  --copy "${PART127_RUNTIME[@]}"

tools/launch_arc.py 127 \
  --step "-I -B part127/run127.py --stage main --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --out results\\part127\\main --pilot-dir results\\part127\\pilot-b64 --batch 64 --device xpu" \
  --copy "${PART127_RUNTIME[@]}"
```

The full run should finish with 1,914 answers and 2,232 fixed paths. The saved shared fixed-path tensor cache is about 60 MB at the model's 2,048-wide residual size. The design estimate is about 15 minutes for capture at measured rates, plus model load and the fixed-path work. The builder did not run the real model or an A770 timing check.

## Copy results back and label

Set `ARC_JOB_DIR` to the mounted Windows job folder used by `launch_arc.py`. Copy whole run folders so the JSONL files, metadata, cosine table, and fixed-path cache stay together. Copy the selected pilot and the main run.

```bash
mkdir -p results/part127
cp -a "$ARC_JOB_DIR/results/part127/pilot-b8" results/part127/
cp -a "$ARC_JOB_DIR/results/part127/pilot-b64" results/part127/
cp -a "$ARC_JOB_DIR/results/part127/main" results/part127/
```

Run the reader phases in this order on the Linux repository. These commands make the known-example gate the first reader call and print the pair eligibility list before study labeling.

```bash
/home/elisepc/neuron-probe/.venv/bin/python -I -B part127/label127.py \
  --phase gate --out results/part127/main

/home/elisepc/neuron-probe/.venv/bin/python -I -B part127/label127.py \
  --phase controls --out results/part127/main

/home/elisepc/neuron-probe/.venv/bin/python -I -B part127/label127.py \
  --phase study --out results/part127/main

/home/elisepc/neuron-probe/.venv/bin/python -I -B part127/score127.py \
  --out results/part127/main
```

## Verification

The full local test command is:

```bash
/home/elisepc/neuron-probe/.venv/bin/python -B -m pytest -q -p no:cacheprovider part127/test_part127.py
```

The CPU stub uses the real runner, label, and scoring entry points. Its logits read the hooked layer outputs. A nonzero push changes the greedy answer. A zero-hook perturbation fails the rig gate. It does not validate the pinned checkpoint, A770, or live reader transports.

Builder verification: 26 passed in 25.86 seconds, with no skips. The undeclared-name command below exited 0 with no findings.

The scratch-copy regressions deliberately break one rule at a time. Each named test below went red in its own scratch copy:

| Broken rule in scratch copy | Test that went red |
| --- | --- |
| Drop one generated arm row | `test_arm_list_matches_registered_counts_exactly` |
| Change the registered single-dose list | `test_written_doses_match_design` |
| Omit the second push when building hook rows | `test_runner_call_log_covers_pair_and_single_hooks` |
| Force the zero-hook rig gate to pass | `test_noop_hook_fails_gate` |
| Count a joint mark even when a half-dose single already has it | `test_rescue_count_excludes_positive_gate_failure` |
| Remove random controls from rescue scoring | `test_random_and_ac_rescue_counting_rule_matches` |
| Regenerate all answers on resume | `test_resume_repeats_nothing_and_pays_nothing` |
| Add arm metadata to the reader prompt | `test_reader_prompt_contains_only_blinded_text` |
| Accept a response marked `max_tokens` | `test_truncated_reader_reply_is_error` |

The undeclared-name check is run on every Python program in this folder:

```bash
python3 tools/undeclared.py \
  part127/run127.py part127/label127.py part127/score127.py part127/test_part127.py
```
