# Part 85 fold 5 notes (A770 path, decision part85-after-r4-2026-10-28 option 1)

Uncommitted fold by a fork of Claude (Opus 5.5), 2026-10-28. Nothing ran on the GPU and no checkpoint was downloaded.

## What an observer record for one Part 85 answer contains (for replay85.py)

Files: step 2 writes `<cp_dir>/observe/<tag>.jsonl` (`tag` = `pilot<N>` or `main`, appended across resumed sessions);
step 1 writes `<cp_dir>/observe/step1.jsonl` (one session; a stale file is renamed `step1.stale-<unix>.jsonl` first).
Each answer row in `<cp_dir>/answers/<tag>-<frame>--<cell>.jsonl` carries `observer_call_id` = the generate record's
`call_id`. One generate call per answer, batch 1.

Per session: `{"event":"session", observer_version 3, observer_sha256, session, label, pid, model_name_or_path,
model_revision (config._commit_hash), generation_defaults{SAMPLING_KEYS}, down_proj_modules{layer:name}, dtype, device,
torch, transformers, python}`, then `{"event":"session85", observe85_version 1, observe85_sha256, observe_sha256,
watch{}}`, and at the end `{"event":"close", calls, outside_generate_changes}`.

Per answer, `{"event":"generate"}` with:
- `call_id` (`<session>:<seq>`), `seq`, `time_start`, `time_end`, `batch` (1), `prompt_len`, `attention_mask` (bool);
- `explicit_kwargs`: every keyword that arrived at `model.generate` except input_ids/attention_mask. Today exactly
  `do_sample, temperature, top_p, max_new_tokens, eos_token_id (list), pad_token_id, use_cache`; `positional_args` 0;
- `effective`: the generation config after those kwargs (`do_sample, temperature, top_p, top_k, min_p, typical_p,
  max_new_tokens, min_new_tokens, num_beams, repetition_penalty, no_repeat_ngram_size, eos_token_id, pad_token_id`);
- `rng_at_entry` / `rng_at_exit`: `cpu_initial_seed`, `cpu_state_sha256`, and on XPU `xpu_initial_seed`, `xpu_state_hex`;
- `hooks_at_entry` / `hooks_at_exit`: `{modules: {name: {pre, post}}, global_pre, global_post}`, excluding the
  observers' own hooks. A clamped or unclamped step-2 call shows only `model.layers.10.mlp.down_proj: {pre: 2, post: 0}`
  (the clamp hook and part85's own counter);
- `prompt_texts`: `{prompt-ids sha256: decode(ids, skip_special_tokens=False)}`;
- `rows[0]`: `prompt_sha256` (sha256 of the JSON id list), `generated_steps`, `token_count` and `token_ids_sha256` of the
  generated ids cut after the first stop token (effective eos ids plus tokenizer eos), `answer_sha256` of
  `decode(cut, skip_special_tokens=True).strip()`, and `layers`: for each layer whose down_proj input changed, per phase
  (`prefill`, `decode`): `calls, positions, changed_calls, changed_positions, max_channels_per_position,
  delta_norm[min,max], write_norm[min,max]` (norm of delta @ W_down^T), `probe_first`, `probe_last` (64 fixed unit
  probes, seed 20261028) and, when at most 16 channels changed, `sparse: true, channels: {unit: [dmin, dmax, pmin,
  pmax]}` (pmin/pmax = the values the matmul received). Unchanged layers are absent; an unclamped answer has `{}`;
- `phase_counts`: every layer's down_proj pre-hook calls and positions per phase; `shape_changes` [],
  `unobserved_forwards` {};
- `raw[0]` (observe85): `prompt_ids` (masked prompt ids as they arrived) and `generated_ids` (full, uncut).

An immediate EOS answer has `generated_ids` of length 1, no decode entry and `phase_counts.decode` `[0, 0]`.

Step 1, per `model(...)` call outside generate: `{"event":"forward", seq, kwargs (names other than input_ids),
positional_args, hooks_at_entry (step 1 shows only layer 10 down_proj {pre: 1}), input_ids [[...]], layers: {"10":
{calls, changed_by_hooks, values: {"6978": [[[v per position]]], "7884": ...}}}}` (values index: call, batch row,
position; read in a forward hook, so they are what the matmul received). 200 neutral sequences, then the 24 passage
pairs, off-world then ordinary.

## Changes, by item

- (a) `observe85.py` subclasses `../observe.py` (unchanged, still observer v3 and byte-identical for Part 86) and adds
  raw ids and forward records. `part85.py` installs it for every step-2 run (the cell loop moved into `_run_cells`
  unchanged) and writes `observer_call_id` per answer; `step1_acts.py` runs its forward passes under it watching layer 10
  units 6978 and 7884. `audit85.py` was rewritten to read only observer records: allowlisted generate keywords and
  values, unchanged sampling defaults, RNG seed at entry (CPU and XPU), hook census, prompt ids against its own
  re-render (plain frame without special tokens; chat template, one user turn, generation prompt on), answer text
  against its own decode of the raw ids, per-layer measured changes (only the registered layer and unit, both phases),
  and `--step1` against the observed layer-10 values (activations exact, maxima, correlations, passage means). Its
  interventions are a list of (layer, unit, value) per cell, so a multi-neuron arm is one more entry; the run loop's
  clamp hook is still single-unit (changing it would invalidate most of the mutation table) and the census expects two
  pre-hooks on layer 10 only, which the new arm will need to widen.
- (b) #5 `score85.py`: a measured different unit (gate computed and failed) scores `DIFFERENT UNIT` with the raw counts
  for 85.5 to 85.7; only missing data stays `NOT READ`. #6 an immediate EOS is legal in the run (`_hook_counts_ok`) and
  the audit. (c) #7 scoring requires steps 0 and 1 for all twelve checkpoints in every scope; LOCAL-RUN runs them for
  all twelve whatever the pilot decides.
- #2, #3, #4, #8: `LOCAL-RUN.md` replaces the cloud bootstrap (no venv, branches or fetches), runs Instruct without
  `--reference`, points both pilot attempts at the one audited step 1 with `--step1` (part85.py) and `--step1-json`
  (audit), and writes out the reader/label-commit/unblinding/publication order. `CLOUD-RUN.md` carries a superseded note.
- Offline runner: `checkpoints.resolve_checkpoint` reads `part85/commits.local.json` (written by download85, prefix
  checked) when no API is passed; `handoff85.code_commit` reads `part85/CODE_COMMIT` when the env var is unset.
- (e) `download85.py` + `test_download85.py` (9 offline tests). `--dry-run` against the Hub, 2026-10-28: 59.5 GB for the
  eleven (nine 5.95 GB two-shard safetensors, SFT and DPO 2.98 GB `pytorch_model.bin`), about 3.3 h at 5 MB/s.

## Mutation

Round 4's eight mutants (A-H) and one for a dropped call id were added. Full gate: 41/42, then 42/42 after one fix. In the first run of those nine, G (no hook
sequence reset) survived: the clamp still lands on every position, so nothing the model receives changes; only part85's
own phase counters drift. The run-and-audit helper now also asserts one prompt call per answer in `hook_fire_counts`.
The full 42-mutant gate then left F4_observer_runs_before_clamp alive (part85's own trace counter moved ahead of the
clamp: nothing the model receives changes, and the new audit does not read that trace). The helper now also asserts
that every clamped row of part85's own trace saw the set value on the prompt, which kills it; both reran KILLED.
