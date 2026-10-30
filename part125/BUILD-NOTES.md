# Part 125 build notes

## Runtime behavior

`run125.py` loads the pinned local snapshot through `common92.load_model(snapshot, device)`. That shared loader uses `local_files_only=True`, bf16 weights, moves the model to the requested device, disables gradients and enters eval mode. No model files or remote calls are needed on the A770. The run records the installed Transformers version.

All transcript forwards use `session125.run_segment`. Each call log records token IDs, shape, cache length before and after, turn, hook state, and the per-call logit hash. Arm 1, arm 2, and twin calls also record the unit's hook layer, neuron, and push-vector SHA-256. The runner saves each exact unit push in `push-vectors/<unit-id>.json`. The pilot determinism gate uses each call's saved hook flag and rebuilds pushed calls from that file after checking the layer, neuron, and SHA-256 against the call log. It then compares replay hashes with run hashes and names the first differing call or vector mismatch. It never derives hook state from an arm or turn. Arm 2 and random hooks are on for every turn-1 prompt and decode call, then off. Each hook is removed in `finally`.

The CPU model reads the hooked layer output when choosing its next token. Its logits also depend on the attention mask and position IDs. It checks mask length against the cache length before the call plus the new tokens, and checks positions from that cache length. A cacheless variant makes the cache-sanity comparison fail.

The runner writes each unit atomically to `results/part125/main/units/<unit-id>.json`. Every row records its model snapshot, device, Transformers version, token cap, and medians hash. Resume refuses a unit, zero-hook check, or run whose parameters differ. It also writes `medians-meta.json` to bind the saved medians to their source parameters. The pilot and main stages use the same output directory so the two pilot units are reused.

The pilot always uses both p92-02 and p92-08, then projects the full 29-unit workload. For each pilot unit, it also runs the fixed “Who are you?” prompt once without a hook and once with its saved push. The pilot gate requires different logits and a target-layer shift whose maximum absolute error from the intended push is at most 0.05; a no-op hook fails. A pilot invocation with `--slot-only` is a parser error. It writes `pilot-meta.json` once. Resume preserves that decision. If the projection exceeds 90 minutes, main must use the 21-slot cut. Main writes a running `run-meta.json` with the checked parameters before entering the unit loop, then replaces it with final counts after aggregation. The labeler and scorer refuse a run unless the status is complete, the completed and expected unit IDs agree, and the unit files present match. The refusal says the run is incomplete before trying to read missing aggregate files. The labeler also requires the pilot effect gate and all other execution gates. The scorer requires `run-meta.json` on every path and checks the saved cut before reading labels.

The import test copies only this runtime support set into a blank tree: `part125/{session125,run125,label125,score125}.py`, `part92/{common92,label92}.py`, `part92b/label92b.py`, `part119/{label119,rows119,known119.jsonl}`, `part123/{fit123,run123,label123}.py`, `part85/checkpoints.py`, and `judge_client.py`. The `--copy` lists below contain this set plus the stage inputs and resume files.

## A770 copy list and pilot

Run from the repository root on the A770 host. The model is the pinned local snapshot. Keep the same `results/part125/main` output folder for both stages.

```sh
/home/elisepc/neuron-probe/.venv/bin/python tools/launch_arc.py 125 \
  --step "-I -B part125/run125.py --stage pilot --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --out results\\part125\\main --device xpu --max-new-tokens 120" \
  --copy part125/run125.py part125/session125.py part125/label125.py part125/score125.py part92/common92.py part92/label92.py part92b/label92b.py part85/checkpoints.py part123/fit123.py part123/run123.py part123/label123.py part119/label119.py part119/rows119.py part119/known119.jsonl judge_client.py results/part92/step2/neurons.json results/part92/step2/kinds.json
```

Copy the full `results/part125/main/` output from the A770 job folder back to the repo before launching the main stage. The main stage must see `pilot-meta.json`, `medians.json`, `medians-meta.json`, `zero-hook-check.json`, both pilot files under `units/`, and both matching files under `push-vectors/`.

## Main run and resume

If the pilot projection is at most 90 minutes, run all 29 units:

```sh
/home/elisepc/neuron-probe/.venv/bin/python tools/launch_arc.py 125 \
  --step "-I -B part125/run125.py --stage main --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --out results\\part125\\main --device xpu --max-new-tokens 120" \
  --copy part125/run125.py part125/session125.py part125/label125.py part125/score125.py part92/common92.py part92/label92.py part92b/label92b.py part85/checkpoints.py part123/fit123.py part123/run123.py part123/label123.py part119/label119.py part119/rows119.py part119/known119.jsonl judge_client.py results/part92/step2/neurons.json results/part92/step2/kinds.json results/part125/main/pilot-meta.json results/part125/main/medians.json results/part125/main/medians-meta.json results/part125/main/zero-hook-check.json results/part125/main/units/p92-02.json results/part125/main/units/p92-08.json results/part125/main/push-vectors/p92-02.json results/part125/main/push-vectors/p92-08.json
```

If the pilot projects above 90 minutes, make the required cut before running either reader. Run the 21 slot units by adding `--slot-only`:

```sh
/home/elisepc/neuron-probe/.venv/bin/python tools/launch_arc.py 125 \
  --step "-I -B part125/run125.py --stage main --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --out results\\part125\\main --device xpu --max-new-tokens 120 --slot-only" \
  --copy part125/run125.py part125/session125.py part125/label125.py part125/score125.py part92/common92.py part92/label92.py part92b/label92b.py part85/checkpoints.py part123/fit123.py part123/run123.py part123/label123.py part119/label119.py part119/rows119.py part119/known119.jsonl judge_client.py results/part92/step2/neurons.json results/part92/step2/kinds.json results/part125/main/pilot-meta.json results/part125/main/medians.json results/part125/main/medians-meta.json results/part125/main/zero-hook-check.json results/part125/main/units/p92-02.json results/part125/main/units/p92-08.json results/part125/main/push-vectors/p92-02.json results/part125/main/push-vectors/p92-08.json
```

For a resume, copy back the same output folder and use the same parameters. `launch_arc.py --copy` accepts files, not directories. Add each existing `run-meta.json`, medians file, zero-hook file, completed unit file, and matching `push-vectors/<unit-id>.json` file to the copy list. Completed units with matching parameters are skipped after their saved vector and call-log hashes are checked. A changed parameter stops the resume. Copy the final full `results/part125/main/` folder back from the job after the run. Required scorer inputs are `answers.jsonl`, `reader-free.jsonl`, `random-reader-free.jsonl`, `cache-sanity.jsonl`, `determinism.jsonl`, `ledger-checks.jsonl`, `call-sequences.jsonl`, `zero-hook-check.json`, `pilot-meta.json`, `run-meta.json`, `medians.json`, `medians-meta.json`, `units/*.json`, and `push-vectors/*.json`.

## Reader commands

Run the 12 known-example checks before study labels. The labeler requires at least 11 correct for each reader.

```sh
/home/elisepc/neuron-probe/.venv/bin/python -I -B part125/label125.py --gate --reader ds --known-ds results/part125/known-ds.jsonl --known-luna results/part125/known-luna.jsonl --out results/part125/known-ds.jsonl
/home/elisepc/neuron-probe/.venv/bin/python -I -B part125/label125.py --gate --reader luna --known-ds results/part125/known-ds.jsonl --known-luna results/part125/known-luna.jsonl --out results/part125/known-luna.jsonl
/home/elisepc/neuron-probe/.venv/bin/python -I -B part125/label125.py --reader ds --answers results/part125/main/answers.jsonl --known-ds results/part125/known-ds.jsonl --out results/part125/labels-ds.jsonl
/home/elisepc/neuron-probe/.venv/bin/python -I -B part125/label125.py --reader luna --answers results/part125/main/answers.jsonl --known-luna results/part125/known-luna.jsonl --labels-ds results/part125/labels-ds.jsonl --out results/part125/labels-luna.jsonl
```

The labeler writes private `.keys.jsonl` sidecars beside each label file. Luna uses Part 123's sample of all current DeepSeek positives plus a seeded draw. The sample size depends on the positive count. Each reader prompt names the current turn. The labeler uses the Part 123 transport, has a 4,000-token DeepSeek cap, shuffles item order with fixed seeds, and retries missing or stale answer hashes.

The agreement gate requires current labels for every sampled item, raw agreement of at least 0.80, and Cohen's kappa of at least 0.60 when defined. If either reader uses only one class, kappa is undefined and the raw-agreement threshold still applies. Missing or stale sample labels fail the gate.

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

Latest result: **45 tests passed**. The suite drives the actual runner, labeler and scorer mains using a tiny CPU model and fake readers. It checks hook-sensitive answers, saved push-vector hashes and replay, no-op pilot hooks, target-layer movement, per-call hook coverage, mask and position inputs, cache failure, complete-run metadata and unit files, determinism joins, logit-value hashes, resume provenance, both agreement thresholds, the pilot parser rule, and `-I` launch commands from a tree containing only the copy list. It does not load OLMo or exercise A770 bf16 behavior.

## Deliberate breaks

Each break was made only in a scratch copy under `$TMPDIR` or the worker run directory, followed by the named test or tests. The scratch copies were discarded after recording the red tests.

| Finding | Break in scratch copy | Test that failed |
|---|---|---|
| F1 | Rebuilt replay hook state from turn/sequence instead of following each saved `hook_on` flag. | `test_f1_replay_uses_logged_hook_state_for_every_call` |
| F2 | Made the stub logits independent of the mask and position inputs. | `test_stub_logits_read_mask_and_positions_and_check_cache_lengths` |
| F3 | Replaced `_push_vector` with a zero vector. | `test_hook_sensitive_stub_distinguishes_push_effects` |
| F4 | Removed the raw-agreement threshold from the agreement gate. | `test_kappa_gate_requires_agreement_and_kappa_thresholds` |
| F5 | Removed the parser error for `--slot-only --stage pilot`. | `test_slot_only_flag_is_rejected_for_pilot` |
| F7 | Restored the optional `run-meta.json` read for `validate_inventory=False`. | `test_score_refuses_missing_run_meta_on_every_path` |
| Review 1 | Changed a saved push vector but skipped its SHA-256 verification. | `test_pilot_replay_refuses_a_saved_push_vector_hash_mismatch` |
| Review 1 | Made the pilot effect check pass every hook, including a no-op. | `test_pilot_push_effect_gate_rejects_a_noop_hook` |
| Review 2/3 | Ignored status, completed counts, recorded unit IDs, and unit files in run metadata. | `test_label_and_score_refuse_incomplete_run_metadata` |
| Review 2/3 | Ignored the running status after a simulated kill before the first unit. | `test_killed_main_before_first_unit_is_reported_as_incomplete` |
| Review 4 | Removed the call-sequence membership checks after setting a missing row's call count and replay hashes to zero. | `test_determinism_row_without_call_sequence_fails_both_gates` |
| Review 5 | Hashed only logit dtype and shape. | `test_logit_hash_tracks_values_and_is_stable_for_a_clone` |
| Review 8 | Put the end-to-end fixture back under read-only `part125/` and ran its setup there. | `test_hook_sensitive_stub_distinguishes_push_effects` |
