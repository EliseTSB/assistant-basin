# Part 219 — build report

Built: `part99/swap99.py` (additive options and receipts), `part219/run219.py`, `part219/score219.py`,
`part219/test_part219.py`; two shared helpers changed additively (`part92/common92.py`, `part101/read101.py`).
Nothing is committed (the reviewer commits). No prediction, threshold or design line was changed.

## 1. What each file does now

### `part99/swap99.py` — additive options only

Every existing default still produces byte-identical behaviour: the new meta keys are written only when their option
is given, generated rows gain no key unless `--record-token-ids` is passed, and `copy_components`, `parse_layers`,
`parse_questions`, `parse_only` and the generation loop are otherwise untouched (the only structural change is that
`run()` can be handed an already-loaded model, and the tokenizer resolution moved into `_resolve_tokenizer`).

| option | behaviour |
|---|---|
| `--attn-implementation {default,eager}` | `eager` is passed to `from_pretrained` for the host (`load_model`) and the donor (`_load_donor`) and recorded in the meta. `default` passes nothing and writes **no** key, as Part 99 did. |
| `--ablate-heads L0H10,...` | a forward pre-hook on each named layer's `o_proj` zeroes `[128h:128(h+1)]` of its **input** at every position. Head size is `hidden_size / num_attention_heads` from the config and is checked against `o_proj.in_features`, `q_proj.out_features` and `config.head_dim`; `num_key_value_heads != num_attention_heads` is refused. |
| `--transplant-heads L0H10,... --donor D --transplant-mode {donor,opposite}` | per head: rows `[s:e]` of `q_proj`/`k_proj`/`v_proj`, columns `[s:e]` of `o_proj`, and `q_norm.weight[s:e]`, `k_norm.weight[s:e]`. `donor` copies the donor's bytes; `opposite` writes `(2 * host - donor)` in float32 cast to the host dtype. The whole plan is shape-checked (donor geometry and every parameter) before one value is written; the originals are kept for the reset. |
| `--reference-replay FILE` | writes the stored prompt's replay to `FILE` when it does not exist, and compares this run's replay with it exactly when it does. |
| `--record-token-ids` | adds each row's generated ids, so two arms can be compared token for token rather than through decoded text. |

Receipts, in the run's meta JSON under the design's names:

- `intervention_receipt` — for ablation, per hooked slice, `max_abs_before`, `max_abs_after`, `all_zero` and the
  number of positions, taken on the **first forward call** (the first batch); if a resumed run found every row on
  disk and never ran a forward, the receipt is taken on the batch a fresh run would have used. For transplant, per
  slice: `matches_expected` (exact byte equality with the donor's slice, or with the float32-then-bf16 `2I - D`),
  `norm_delta` (`||W - W_0||`) and `norm_ratio` (`||W|| / ||W_0||`).
- `coupling_log` — transplant runs only, on the same first batch: `rms_q`, `rms_k` (after the projections, before the
  QK norms) and `rms_attention_output` (the input to `post_attention_layernorm`) before/after with relative changes,
  the sibling heads' per-head output drift with mean and max, the changed heads' own output change, and the batch's
  ids/units/question indices so the measurement is auditable.
- `reset_receipt` — after the run, hooks removed and the original bytes restored (transplant slices **and** Part 99's
  swapped parameters), the stored prompt ("Who are you?", chat frame) is replayed: `prompt_ids`, `token_ids` (eight
  greedy tokens) and `first_step_logits` (the full next-token distribution). `role` is `reference_written` or
  `checked`, with `matches_reference` and the first `mismatch`.

### `part92/common92.py`

`load_model(..., attn_implementation=None)` passes the kwarg only when given. `batch_inputs()` is the left-padding
rule extracted from `generate_batch` (so the coupling probe sees exactly the first generation batch, not a copy of
the rule). `generate_batch(..., return_ids=False)` returns the token ids when asked.

### `part101/read101.py`

`relative(lab, unit, i, q)` is Part 99's relative-show definition lifted out of `score_rows` so Part 219 reuses one
definition of the read instead of writing a second; `score_rows` calls it and its behaviour is unchanged.

### `part219/run219.py`

`--stage a` runs R0, R0b, R2…R7 on one host load; `--stage b` runs R1, R8, R9, R10 on one host + donor load. Batch
48, chat frame, Part 99's fixed pushes, `results/part92/step2`'s kinds map and neuron list, eager on every load. It
recomputes R5/R9's heads from the registered rule (`random.Random(219)`, one `rng.choice` per layer over that layer's
15 eligible heads) and refuses to start if the result is not `L0H12, L1H11, L2H5, L3H9` or if a registered drop is
not under 0.035. It writes `manifest.json` after every run (so a crash leaves a record), checks R0b against R0 token
for token, checks every reset receipt against R0's stored replay, and marks a run unreadable when its row count,
receipt or transplant bytes are wrong. Before a stage launches anything it checks the design's kernel record
(`--kernel-check`, default `results/part219/kernel-check.json`): the record must exist, pass
`tools/part_check.py kernel_check` against its CPU float32 reference, and record `attn_implementation: "eager"` on the
candidate (and on its `load` block when present), or the stage refuses to start. A progress line is printed every
eight seconds by a heartbeat thread as well as by swap99's per-240-row lines.

### `part219/score219.py`

Reuses `read101`'s read set, relative counts and `inconclusive` flag; computes the 16/5 split from
`results/part92/step2/neurons.json` and the head layers (a unit is upstream when its layer is at least the deepest
head layer — 16 at layers ≥ 3, 5 at layers 1–2 in the registered data, not hard-coded); adds net counts against R0
with the exact two-sided McNemar p; the per-run two-pass gate on the 42 slot items (all 42 present, ≥ 40 agree on
`shows_persona`, slot counts within 2); the gate; every reading rule; and the headline rules. Writes
`scorecard.json` and `SCORECARD.md` into `--out`. `--pilot` prints the label half of the pilot rule.

## 2. Decisions where the brief or design left a choice

1. **R0b is one invocation** with `--ablate-heads=` and `--transplant-heads=` both empty. An empty transplant set
   deliberately does **not** require a donor, so R0b stays in the no-donor process (the design's one host load). The
   ablation hooks are still installed on every layer and return `None`. The empty transplant set enters
   `transplant_heads` and returns from its empty early return, before the geometry/donor/shape planning: R0b
   exercises that empty path, **not** the planner, and the planner's shape checks are covered by the transplant
   tests. That is what "through both new hooks with empty head sets" can mean without a second model.
2. **`--reference-replay` writes when absent and compares when present.** `run219 --stage b` refuses to start when
   the file is missing, so a stage-b run can never silently become its own reference.
3. **The reset receipt restores Part 99's `--swap` too.** R1 changes the model by swapping layers 0–3's attention,
   and the design requires R1's replay to equal R0's, so the swapped parameters are cloned before the copy and put
   back after the run. Only done when `--reference-replay` is given, so Parts 99–104 are untouched.
4. **`--record-token-ids` is an extra option.** "R0b must match R0 token for token" cannot be checked from decoded
   text (two id sequences can decode alike), so the rows must carry ids.
5. **The pilot's two label criteria are deferred, never approximated.** The runner checks the receipts immediately;
   the R0 ≥ 17-of-21 and R2 ≥ 15-of-21-coherent criteria need the blind reader, so `--pilot` answers
   `pending_labels` until `labels-R0.jsonl` and `labels-R2.jsonl` exist, then applies the registered numbers. A stop
   exits 2.
6. **The 42-item gate compares `shows_persona`**, the field the read is about; `coherent` agreement is computed and
   reported beside it. Item count is required to be exactly 42 as well as ≥ 40 agreeing.
7. **The grouped-query refusal covers the transplant too**, not only the ablation: the slice definition presupposes
   one key/value head per query head, and a GQA donor would otherwise be sliced at the wrong offsets.
8. **`first_step_logits` is the full next-token distribution** (~50k floats per run's meta, about 1 MB), because the
   design says the logits must be equal exactly and a digest would not record them.

Nothing in the design, `PREDICTIONS.md`, or any threshold was changed.

## 3. Tests

```
$ /home/elisepc/neuron-probe/.venv/bin/python -I -m pytest part219 part99 -q
63 passed in 10.26s
```

(`python3` on this machine has no torch; the run above used the project's torch/transformers 2.14.1+cpu/5.20.0
environment. `part219` alone: 49 passed. The review fold added 18 tests — see `REPORT-fold-r1.md` for each one and
its sabotage.)

The fixture is `tests/test_olmo2.py`'s tiny OLMo 2 (16 hidden, 3 layers, 2 query heads, 2 key/value heads, head_dim
8) built as an `Olmo2ForCausalLM` with a word-level tokenizer and a chat template, so `swap99.run` goes through its
whole command path — load, fixed pushes, hooks, generation, receipts, meta — on the CPU. The end-to-end command line
also ran on a 16-head version of the same model (so the registered names `L0H10 … L3H5` exist): `run219.py --stage a`
then `--stage b` wrote 12 runs of 30 rows, R0b identical to R0 token for token, all 12 reset receipts equal to R0's,
`frozen_recomputed_equals_registered: true`, and a coupling log over layers 0–3. `--pilot` ran R0 and R2 and reported
`pending_labels`.

### Every test and its sabotage

Each sabotage patches a **copy** of the tracked files (never the tracked file) and re-runs the suite; all 49 turned
their named test red. The patch column is the change, not the diff line. The 31 build tests are below; the 18 the
review fold added are listed with their sabotages in `REPORT-fold-r1.md`.

| test | sabotage that turns it red |
| --- | --- |
| `test_generate_batch_matches_the_padding_rule_it_had_before` | `batch_inputs` pads on the right instead of the left |
| `test_a_default_run_writes_no_part219_keys` | `run()` records the raw option again (`attn_option`), so a default CLI run writes `attn_implementation: "default"` |
| `test_the_attention_implementation_option_reaches_every_load` | `load_model` drops the kwarg |
| `test_empty_head_sets_are_bit_identical` | the empty ablation hook zeroes head 0 |
| `test_ablation_zeroes_exactly_the_named_slice` | the hook's span is `[0:head_size]` whatever the head |
| `test_ablation_refuses_a_grouped_query_stack` | the `num_key_value_heads` refusal removed |
| `test_transplant_writes_exactly_the_expected_bytes` | `o_proj` takes rows instead of columns |
| `test_transplant_refuses_a_donor_with_different_shapes` | **both** guards (donor geometry and parameter shapes) removed; with either one left the other still refuses, which is why the test needs both off to go red |
| `test_reset_receipt_restores_bit_identical_logits` | `restore_transplant` returns without writing |
| `test_reset_receipt_is_written_by_the_reference_run` | the reference file is never written |
| `test_coupling_log_fields_are_present_and_finite` | `sibling_output_drift` key renamed away |
| `test_transplant_receipt_holds_the_expected_bytes` | the write is multiplied by 1.5 after `expected` is computed, so `matches_expected` must catch it |
| `test_split_is_computed_from_the_neuron_list` | `layer >= deepest` becomes `layer > deepest` |
| `test_the_registered_predictions_hold_in_a_world_built_so_they_do` | `DISTANCE_MARGIN` 4 → 8 |
| `test_the_scorer_reports_the_negative_when_r2_loses_too_few` | `NET_MIN` 7 → 1 |
| `test_219_3_holds_but_a_broken_r8_reads_inconclusive` | `_bars` always holds, ignoring coherence |
| `test_a_broken_receipt_marks_the_run_unreadable_and_blocks_its_prediction` | `readable_reasons` ignores the manifest |
| `test_the_42_item_gate_fails_though_a_pooled_gate_would_pass` | the gate compares every shared item (pooled) instead of the 42 slot items |
| `test_the_reader_gate_names_which_criterion_failed` | slot-count tolerance 2 → 100 |
| `test_r5_failing_its_rig_check_freezes_219_1_and_219_2` | `RIG_MAX_ABS_NET` 2 → 5 |
| `test_r9_failing_its_coupling_check_blocks_219_3_to_219_5` | `COUPLING_MAX_ABS_NET` 2 → 6 |
| `test_failed_gate_freezes_every_prediction` | the gate always passes |
| `test_a_missing_kernel_record_fails_the_gate` | a missing kernel record passes |
| `test_219_4_is_not_read_unless_219_3_and_219_5_hold` | 219.4 is always read |
| `test_mcnemar_is_the_exact_two_sided_binomial` | the p is one-sided |
| `test_pilot_labels_reads_the_two_pilot_runs` | `pilot_labels` reports `[0, 21]` |
| `test_the_frozen_head_list_is_the_registered_one` | `FROZEN_SEED` 219 → 220 |
| `test_pilot_stops_when_r0_shows_too_few_slot_units` | `PILOT_R0_MIN_SLOT` 17 → 10 |
| `test_pilot_stops_when_r2_leaves_too_few_slot_answers_coherent` | `PILOT_R2_MIN_COHERENT` 15 → 3 |
| `test_pilot_continues_when_every_criterion_holds` | the pilot ignores missing labels |
| `test_pilot_is_pending_without_labels` | `pilot_labels` reports `available: True` unconditionally |

## 4. Launch

`part219/steps219.txt` holds the two `job.py` steps (`--stage a`, then `--stage b`) and `part219/steps219-pilot.txt`
the pilot step. `tools/launch_arc.py 219` needs the modules in the job folder:

```
python tools/launch_arc.py 219 \
  --copy part219/run219.py part219/score219.py part99/swap99.py part92/common92.py \
         part85/checkpoints.py part101/read101.py tools/part_check.py \
  --step "-I -B part219\run219.py ... --stage a" --step "-I -B part219\run219.py ... --stage b"
```

`run219.py` inserts the repo root, `part92`, `part99` and its own directory on `sys.path` (a Windows-side `-I` run
does not add the script's directory); `score219.py` does the same for `part101` and `part92`, and imports
`tools.part_check` only where the kernel check is read.

That exact `--copy` list passes `tools/launch_arc.py 219 --dry-run --no-fetch` here: all 22 prelaunch checks pass,
the kernel check included, and the launcher finds no uncopied local import in the copied files.

The kernel check already committed for this part (`results/part219/kernel-check.json` against
`results/kernel-reference/allenai-olmo-2-0425-1b-instruct-cpu-float32.json`) passes `score219.kernel_check`, so the
gate's first item is satisfied on the current tree.

## 5. What a reviewer should look at hardest

- `install_ablation_hooks` clones the `o_proj` input on every forward of an ablation run; that is the per-step cost
  the pilot will show. It is the only added per-step work in the hot loop.
- The coupling probe runs the first batch twice (before and after the transplant) — three runs × one batch.
- The reset receipt writes ~50k floats per run's meta; 12 runs plus the reference is ~13 MB of JSON. If that is
  unwanted, the logits could move to a `.pt` beside the meta without changing any check.
- `run219` treats a row count other than `12 × units + 6` as a problem, so a `--limit` smoke run must also be scored
  from a `--limit`-sized directory (the tests and the smoke above do this).

## SUMMARY

Part 219's code is built and green. `part99/swap99.py` gains `--attn-implementation`, `--ablate-heads`,
`--transplant-heads`/`--transplant-mode`, `--reference-replay` and `--record-token-ids`, plus the design's three
receipts in every run's meta, with every old default unchanged. `part219/run219.py` orchestrates R0–R10 in the
design's two processes, asserts the frozen head list from the registered rule, checks R0b token for token and every
reset receipt against R0, and writes a manifest that marks bad runs unreadable. `part219/score219.py` reuses Part
101's read, computes the 16/5 split from the neuron list, adds net counts, exact McNemar, the 42-item two-pass gate,
the gate and every reading/headline rule, and writes `scorecard.json` and `SCORECARD.md`. `63 passed` for
`pytest part219 part99 -q`; 49 tests each shown red under a named sabotage; the two-stage command line ran end to
end on a tiny 16-head OLMo 2. The review fold (`REPORT-fold-r1.md`) adds the F1–F10 fixes on top.

## FINDINGS

- `read101.score_rows` had Part 99's relative-show definition nested inside it; Part 219 needed the same rule per
  unit, so it is now a module-level `read101.relative()` and `score_rows` calls it. One definition, no behaviour
  change (Part 99's 14 tests still pass).
- `generate_batch`'s left-padding rule was duplicated wherever a batch had to be built; it is now `batch_inputs()`,
  used by `generate_batch` and by the first-batch coupling probe.
- A resumed run (rows already on disk) would have skipped the first forward and left the ablation's intervention
  receipt empty; the runner now takes the receipt on one forward before removing the hooks.
- `part_check.kernel_check` reads the *reference's* `argmax_agrees`; the committed Part 219 candidate carries
  `argmax_agrees: false` at its top level (two near-tie divergences) and still passes, which is the intended
  behaviour but reads oddly at a glance.
- The brief's `python -I -m pytest part219 part99 -q` does not run under the system `python3` here (no torch); see
  BLOCKERS.

## BLOCKERS

None for the build. One environment note: the system `python3` has no torch or transformers, so the brief's command
as written fails with `ModuleNotFoundError: No module named 'torch'`; the suite was run with a torch-bearing
interpreter (`/home/elisepc/neuron-probe/.venv/bin/python -I -m pytest part219 part99 -q`, 63 passed after the review
fold; the build's run used `/home/elisepc/dspy-wsp/.venv/bin/python` and reported 45). No GPU, no real OLMo weights
and no earlier part's results were touched.
