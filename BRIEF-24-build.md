# Brief: build part24.py (the alien neuron and the identity neurons together)

You are a dark-factory worker in `/home/elisepc/green/worktrees/ab-24` (branch `part24` of the public assistant-basin
repo, HEAD fa9d778). Write only inside it. Do not commit, do not download models, do not edit PREDICTIONS.md, and do
not load Qwen3.5-4B itself anywhere (the real run happens later on an Intel Arc GPU under Windows; on this machine a
4B model on CPU starves everything else).

Read first: PREDICTIONS.md, section "Part 24" (the exact definitions; build to them, and if a definition is ambiguous
say so in your report rather than choosing silently). Then reuse, do not copy:
- `neuron_report.py`: `load_model`, `find_text_layers`, `NeuronHook` (its `clamp`, `phase` and `position_mask`),
  `build_prompt`, `_dump`, `_file_sha`, `spill_guard`, the `steer` command's `generate` settings (temperature 0.7,
  top_p 0.95, `max_new` default, seeds) and the `teacherforce` command (`cmd_teacherforce`, `_teacherforce_stats`,
  `_phase_span`): how Part 18c builds each sequence (chat template of the question, then stem, then answer) and scores
  only the answer tokens.
- `circuit.py`: `MultiZero` (zeroing a set of (layer, unit) pairs), `ident_idxs` (which answer tokens are name tokens),
  and how `answers.json` records each question's prompt ids and answer ids.

Inputs (all paths are arguments, so the GPU folder can pass its own): `--verify results/part19/verify.json`
(`top_neurons` = identity-50, `random_neurons` = random-50), `--answers results/part19/answers.json`,
`--reads results/part18/chat/reads.json` (N2742's corpus maximum; refuse if it was written for another model, revision,
layer or neuron, as neuron_report.py does), `--assay data/creator-assay.json`. Model `Qwen/Qwen3.5-4B`, revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, `--layer 22 --neuron 2742 --device xpu`, a `--max-gpu-gb 14.5` spill guard
called after every batch (the GPU has 16 GB and silently spills into system memory rather than failing).

Refuse to run if N2742 is in either zero set, or if the two sets overlap.

## What to build

`part24.py` with three subcommands, each writing one JSON file into `--out` with a header like the other tools
(model, revision, tool sha, input file shas, the exact k values, timestamp):

1. `reach` (24.1 to 24.3) -> `reach.json`. For each of Part 19's six questions, teacher-force the recorded prompt plus
   answer ids. Record every identity-50 and random-50 unit (all layers, one `NeuronHook` per layer with `record_units`)
   at k = 0, 1, 2, 4, with N2742 clamped to k times its maximum on the answer positions only (`position_mask`; the
   question positions untouched). k = 0 means no clamp at all, not a clamp to zero. Store, per unit, per question and
   per k: the mean absolute activation over answer tokens, the mean absolute change from k = 0 over answer tokens, and
   the activation vectors on the name tokens (from `ident_idxs`). Also store N2742's own recorded value at the answer
   positions per k, so the clamp can be seen to have happened. Then compute, as defined in PREDICTIONS.md: each unit's
   relative change, each set's median over its units in layers 23 and up, the maximum relative change over all units
   in layers 22 and below (24.1), and each unit's retention on name tokens with the set median (24.3). Report a unit
   whose unclamped mean absolute activation is exactly zero as excluded, with a count, never as a division.
2. `assay` (24.4, 24.5) -> `assay.json`. Part 18c's assay, `self` and `other` sets, every template and every answer.
   Conditions: k in 0, 0.5, 1, 1.5, 2 with N2742 clamped at ALL positions, crossed with zero in none, identity-50,
   random-50 (`MultiZero` at all positions). Store per condition, per set, per template, per answer: the summed and
   per-token log-probability of the answer tokens. Then compute, per set and zero condition: the margin per k (mean
   per-token log-probability of the six extraterrestrial answers minus the six earthly, averaged over templates, as 18c
   does) and Effect(k, Z) = margin(k, Z) - margin(0, Z), with 95% intervals by resampling templates (reuse
   `_bootstrap`). Also the mean per-token log P(" Alibaba Cloud.") on the self set at k = 0 under each zero condition.
3. `samples` (24.6, 24.7) -> `samples.json`. "Who made you?" and "Who created you?", 8 samples each, with the steer
   command's sampling settings and a fixed seed per (question, k, zero) so a condition can be re-run identically. k in
   0, 1, 1.5, 2 (clamp on generated tokens only, as Part 18b's phase gen: the clamp applies when the step has one
   position), crossed with the same three zero conditions (zeroing applies at every position). Store the texts only.
   Do not classify them: a person reads every sample by hand. Add a `--max-new` default of 64.

A `--steps reach,assay,samples` argument runs them in order in one process and loads the model once. Batch the assay
(default `--batch-size 16`); keep the hooks' removal in `finally` blocks so a failure cannot leave a hook attached.

## Tests

`test_part24.py` on CPU with a tiny random model built from the real Qwen3.5 config (follow `test_circuit.py`'s `tiny()`
pattern; snapshot path from `QWEN35_SNAPSHOT`), with fake verify, answers, reads and assay files whose units fit the
tiny model. Rows that can fail, each shown failing against a deliberate break before you keep it:
- with k = 0 in every condition, every relative change is exactly 0 and every Effect is exactly 0;
- the left-to-right check: put the target in a middle layer; units in that layer and below change by 0 at every k, and
  units above it change by more than 0 at k = 4 (so the check can fail both ways);
- a break that applies the reach clamp to every position instead of the answer positions makes some question-position
  value change: the test must see it;
- zeroing really zeroes: with identity-50 zeroed, a recorded identity unit reads exactly 0 at every position;
- the margin and Effect arithmetic reproduce a hand-computed value on a two-template fake assay;
- the refusals: N2742 in a zero set, overlapping sets, a reads file for another neuron.
Run with `/home/elisepc/neuron-probe/.venv/bin/python -B test_part24.py` and
`QWEN35_SNAPSHOT=/home/elisepc/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
inside `systemd-run --user --scope -p MemoryMax=3G`. Also run `test_neuron_report.py` and `test_circuit.py` to show
nothing shared broke (you should not need to edit either module; if you must, say exactly why).

## Report

`REPORT-24-build.md` in the worktree root: SUMMARY (suite counts and exit codes, quoted), FINDINGS (what was built,
each design choice the brief left open, each break you used to show a row can fail and what the row printed), and
BLOCKERS. Quote test output you saw; never describe a result you did not run.
