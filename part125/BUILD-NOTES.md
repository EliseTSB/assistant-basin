# Part 125 build notes

## Runtime behavior

`run125.py` loads the pinned local snapshot through `common92.load_model(snapshot, device)`. That shared loader uses `local_files_only=True`, bf16 weights, moves the model to the requested device, disables gradients and enters eval mode. No model files or remote calls are needed on the A770. The run records the installed Transformers version.

All transcript forwards use `session125.run_segment`. Each call log records token IDs, shape, cache length before and after, turn, and hook state. The twin and random controls replay arm 2's call plan. Arm 2 and random hooks are on for every turn-1 prompt and decode call, then off. Each hook is removed in `finally`.

The runner writes each unit atomically to `results/part125/main/units/<unit-id>.json`. Every row records its model snapshot, device, Transformers version, token cap, and medians hash. Resume refuses a unit, zero-hook check, or run whose parameters differ. It also writes `medians-meta.json` to bind the saved medians to their source parameters. The pilot and main stages use the same output directory so the two pilot units are reused.

The pilot always uses both p92-02 and p92-08, then projects the full 29-unit workload. It writes `pilot-meta.json` once. Resume preserves that decision. If the projection exceeds 90 minutes, main must use the 21-slot cut. The labeler refuses to read study answers until pilot metadata and all execution gates pass. The scorer checks the saved cut before reading labels and recomputes each gate and prediction from the raw files and answer-hash-bound labels.

The import test copies only this runtime support set into a blank tree: `part125/{session125,run125,label125,score125}.py`, `part92/{common92,label92}.py`, `part92b/label92b.py`, `part119/{label119,rows119,known119.jsonl}`, `part123/{fit123,run123,label123}.py`, `part85/checkpoints.py`, and `judge_client.py`. The `--copy` lists below contain this set plus the stage inputs and resume files.

## A770 copy list and pilot

Run from the repository root on the A770 host. The model is the pinned local snapshot. Keep the same `results/part125/main` output folder for both stages.

```sh
/home/elisepc/neuron-probe/.venv/bin/python tools/launch_arc.py 125 \
  --step "-I -B part125/run125.py --stage pilot --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --out results\\part125\\main --device xpu --max-new-tokens 120" \
  --copy part125/run125.py part125/session125.py part125/label125.py part125/score125.py part92/common92.py part92/label92.py part92b/label92b.py part85/checkpoints.py part123/fit123.py part123/run123.py part123/label123.py part119/label119.py part119/rows119.py part119/known119.jsonl judge_client.py results/part92/step2/neurons.json results/part92/step2/kinds.json
```

Copy the full `results/part125/main/` output from the A770 job folder back to the repo before launching the main stage. The main stage must see `pilot-meta.json`, `medians.json`, `medians-meta.json`, `zero-hook-check.json`, and both pilot files under `units/`.

## Main run and resume

If the pilot projection is at most 90 minutes, run all 29 units:

```sh
/home/elisepc/neuron-probe/.venv/bin/python tools/launch_arc.py 125 \
  --step "-I -B part125/run125.py --stage main --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --out results\\part125\\main --device xpu --max-new-tokens 120" \
  --copy part125/run125.py part125/session125.py part125/label125.py part125/score125.py part92/common92.py part92/label92.py part92b/label92b.py part85/checkpoints.py part123/fit123.py part123/run123.py part123/label123.py part119/label119.py part119/rows119.py part119/known119.jsonl judge_client.py results/part92/step2/neurons.json results/part92/step2/kinds.json results/part125/main/pilot-meta.json results/part125/main/medians.json results/part125/main/medians-meta.json results/part125/main/zero-hook-check.json results/part125/main/units/p92-02.json results/part125/main/units/p92-08.json
```

If the pilot projects above 90 minutes, make the required cut before running either reader. Run the 21 slot units by adding `--slot-only`:

```sh
/home/elisepc/neuron-probe/.venv/bin/python tools/launch_arc.py 125 \
  --step "-I -B part125/run125.py --stage main --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --out results\\part125\\main --device xpu --max-new-tokens 120 --slot-only" \
  --copy part125/run125.py part125/session125.py part125/label125.py part125/score125.py part92/common92.py part92/label92.py part92b/label92b.py part85/checkpoints.py part123/fit123.py part123/run123.py part123/label123.py part119/label119.py part119/rows119.py part119/known119.jsonl judge_client.py results/part92/step2/neurons.json results/part92/step2/kinds.json results/part125/main/pilot-meta.json results/part125/main/medians.json results/part125/main/medians-meta.json results/part125/main/zero-hook-check.json results/part125/main/units/p92-02.json results/part125/main/units/p92-08.json
```

For a resume, copy back the same output folder and use the same parameters. `launch_arc.py --copy` accepts files, not directories. Add each existing `run-meta.json`, medians file, zero-hook file, and completed unit file to the copy list. Completed units with matching parameters are skipped. A changed parameter stops the resume. Copy the final full `results/part125/main/` folder back from the job after the run. Required scorer inputs are `answers.jsonl`, `reader-free.jsonl`, `random-reader-free.jsonl`, `cache-sanity.jsonl`, `determinism.jsonl`, `ledger-checks.jsonl`, `call-sequences.jsonl`, `zero-hook-check.json`, `pilot-meta.json`, `run-meta.json`, `medians.json`, `medians-meta.json`, and `units/*.json`.

## Reader commands

Run the 12 known-example checks before study labels. The labeler requires at least 11 correct for each reader.

```sh
/home/elisepc/neuron-probe/.venv/bin/python -I -B part125/label125.py --gate --reader ds --known-ds results/part125/known-ds.jsonl --known-luna results/part125/known-luna.jsonl --out results/part125/known-ds.jsonl
/home/elisepc/neuron-probe/.venv/bin/python -I -B part125/label125.py --gate --reader luna --known-ds results/part125/known-ds.jsonl --known-luna results/part125/known-luna.jsonl --out results/part125/known-luna.jsonl
/home/elisepc/neuron-probe/.venv/bin/python -I -B part125/label125.py --reader ds --answers results/part125/main/answers.jsonl --known-ds results/part125/known-ds.jsonl --out results/part125/labels-ds.jsonl
/home/elisepc/neuron-probe/.venv/bin/python -I -B part125/label125.py --reader luna --answers results/part125/main/answers.jsonl --known-luna results/part125/known-luna.jsonl --labels-ds results/part125/labels-ds.jsonl --out results/part125/labels-luna.jsonl
```

The labeler writes private `.keys.jsonl` sidecars beside each label file. Luna uses Part 123's sample of all current DeepSeek positives plus a seeded draw. The sample size depends on the positive count. Each reader prompt names the current turn. The labeler uses the Part 123 transport, has a 4,000-token DeepSeek cap, shuffles item order with fixed seeds, and retries missing or stale answer hashes.

The agreement gate requires current labels for every sampled item. If either reader uses only one class, it uses raw agreement with the same 0.60 threshold and records the fallback. Otherwise it uses Cohen's kappa. Missing or stale sample labels fail the gate.

## Scoring

For a full 29-unit run:

```sh
/home/elisepc/neuron-probe/.venv/bin/python -I -B part125/score125.py \
  --answers results/part125/main/answers.jsonl \
  --reader-free results/part125/main/reader-free.jsonl \
  --random-reader-free results/part125/main/random-reader-free.jsonl \
  --cache-sanity results/part125/main/cache-sanity.jsonl \
  --determinism results/part125/main/determinism.jsonl \
  --ledger-checks results/part125/main/ledger-checks.jsonl \
  --call-sequences results/part125/main/call-sequences.jsonl \
  --zero-hook results/part125/main/zero-hook-check.json \
  --labels-ds results/part125/labels-ds.jsonl --keys-ds results/part125/labels-ds.jsonl.keys.jsonl \
  --labels-luna results/part125/labels-luna.jsonl --keys-luna results/part125/labels-luna.jsonl.keys.jsonl \
  --known-ds results/part125/known-ds.jsonl --known-luna results/part125/known-luna.jsonl \
  --out results/part125
```

For the registered 21-slot cut, add `--slot-only`. The scorer keeps the registered prediction text and records adjusted thresholds in `slot_only_note`: 125.2 at least 5 of 21, 125.3 at least 15 of 21, 125.5 at least 5, and 125.6 at most 1 of 21. It writes `SCORE.md` and `score125.json`. It counts missing labels separately and never treats them as false.

## Verification

Run the complete CPU suite with the specified environment:

```sh
/home/elisepc/neuron-probe/.venv/bin/python -B -m pytest -q -p no:cacheprovider part125/test_part125.py
```

Latest result: **27 tests passed**. The suite drives the actual runner, labeler and scorer mains using a tiny CPU model and fake readers. It checks push effects, per-call hook coverage, paired call plans, both determinism replays, resume provenance, the kappa fallback, and `-I` launch commands from a tree containing only the copy list. It does not load OLMo or exercise A770 bf16 behavior.
