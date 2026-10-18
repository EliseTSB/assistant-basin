# Brief: build `teacherforce` for Part 18c (assistant-basin)

You are a dark-factory worker building one subcommand in a public interpretability repo. Your working directory is this worktree, `/home/elisepc/green/worktrees/ab-18c` (branch `part18c`). Write only inside it. Do not commit: the host commits after checking your work. Do not download models; the tests run on CPU with a tiny random model.

## What to read first
1. `PART18C-DESIGN.md`: the measurement, conditions and statistics. It is the specification; follow it exactly. If something in it cannot be built as written, stop and say so in the report rather than changing the measurement.
2. `data/creator-assay.json`: the prompt sets, stems and completions.
3. `neuron_report.py`: reuse `load_model`, `find_text_layers`, `build_prompt`, `control_units`, `NeuronHook`, `spill_guard`, `header`, `_dump`, and the logit-lens computation in `cmd_writes`. Match the file's style: plain functions, short docstrings, argparse subcommands.
4. `test_neuron_report.py`: how the tests build a tiny random Qwen3.5 text model (it has linear-attention layers, so its cache is not a plain key/value cache).

## What to build
1. **Position mask in `NeuronHook`.** An optional boolean mask over (batch, sequence) positions. When set, the clamp applies only where the mask is true. When it is not set, behaviour must be exactly as now (the `phase` logic for generation is unchanged). Existing tests must still pass.
2. **Subcommand `teacherforce`.** Arguments: `--model --revision --layer --neuron --reads` (for the corpus maxima and the control list, validated the same way `cmd_steer` validates them), `--assay`, `--ks` (comma list), `--phases` (any of `prompt,gen,both`, meaning P, G, A in the design), `--zero` (adds k = 0 at `both`), `--controls` and `--seed` (as in steer), `--control-ks`, `--control-phases`, `--control-scales` (any of `own,norm`), `--skip-controls`, `--system-prompt` (optional text for a system message), `--batch-size` (default 16), `--device`, `--spill-limit` if the other subcommands have one, `--out`.
   - Sequence = `build_prompt` text (with the system message when given; extend `build_prompt` compatibly if needed) tokenized, then the tokens of `stem + completion`. Find the completion's tokens by tokenizing the stem alone; if the stem's tokens are not a prefix of the stem-plus-completion tokens, raise with the offending pair. P positions are those before the first answer token; G positions are the rest; A is all.
   - One forward pass per batch of sequences, right-padded with an attention mask. Take the log-softmax in float32 only at the positions that predict completion tokens. Keep memory modest (do not materialise float32 logits for every position of a large batch).
   - Clamp values exactly as `cmd_steer` computes them: target k times its maximum; controls k times their own maximum (`own`) or norm-matched to the target (`norm`).
   - Per sequence record: set, template index, stem index, completion, kind (et or earthly), condition (unit, k, phase, scale), sum and mean log-probability, and the number of completion tokens.
   - Summaries as in the design: per set and condition, ΔΔ on the mean scale and on the summed scale, each with a 95% CI from 2,000 bootstrap resamples of templates (numpy, seed 0); interactions self minus other and self minus unrelated, resampling each set independently; per-completion ΔΔ_c for the self set; and each completion's mean logit-lens z-score over its tokens, excluding a final token that is only punctuation.
   - Output `teacherforce.json` in `--out`: the `header`, the assay file's sha256, the conditions, the per-sequence rows and the summaries. Also print a short table of the main summaries.
3. **Tests**, in a new file `test_teacherforce.py`, in the style of `test_neuron_report.py`, CPU only:
   - the masked hook clamps only masked positions, and an unset mask leaves the old behaviour;
   - **the P condition scored in one masked pass equals scoring from a patched cached prefix with the hook removed for the continuation** (build the cache with the clamp on over the prompt, then score the continuation with the hook off, using the model's own cache object), within 1e-4, on the tiny model;
   - a completion's summed log-probability matches a manual computation;
   - Δ and ΔΔ on a hand-made table of scores give the hand-computed values, and the bootstrap CI is deterministic with seed 0;
   - right padding does not change any sequence's scores (score each sequence alone and in a padded batch);
   - an end-to-end run of the subcommand on the tiny model with a small assay writes the JSON with the expected keys;
   - mutation check: if you break the mask (clamp every position), the cached-prefix equivalence test fails. Do this by hand and say in the report that you did.

## How to run the tests
`QWEN35_SNAPSHOT=/home/elisepc/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a python3 test_teacherforce.py` and the same for `test_neuron_report.py`, which must still pass.

## Report
Write `REPORT-18c-build.md` in this worktree's root (exactly that name and place). Start with SUMMARY (what you built, test counts as printed), then FINDINGS (anything in the design you had to interpret, and how), then BLOCKERS (or "none"). Quote the test output lines; do not describe results you did not see.
