# Build brief: Part 219 Amendment 1 (mean-ablation)

You are extending code for a REGISTERED research design. Read `PART219-DESIGN.md` in full first, especially the
"## Amendment 1" section at the end: it is the contract. Then read `part99/swap99.py` (`install_ablation_hooks`, the
receipts, `main`), `part219/run219.py`, `part219/score219.py`, `part219/test_part219.py`, `part92/common92.py`
(`load_model`, the left-padded batched greedy decode) and `tests/test_olmo2.py` (tiny OLMo 2 fixtures). Do NOT change
the design, PREDICTIONS.md or any threshold. If something cannot be built as written, stop and say so in REPORT-a1.md.

Work only inside this worktree. Do not load the real OLMo weights, do not use the GPU, run only tiny random models. Do
not commit. Write `part219/REPORT-a1.md` within your first ten tool calls (a skeleton), fill it as you go, and end it
with SUMMARY, FINDINGS, BLOCKERS. Keep your messages short; write files early.

## What to build

1. **`part99/swap99.py`, additive.** Every existing default stays byte-identical. Add `--ablate-mode {zero,mean}`
   (default `zero`, today's behaviour) and `--head-means <path>` (required with `mean`). In `mean` mode the pre-hook on
   `o_proj` replaces each named head's 128-wide input slice with fixed vectors from the means file:
   - `mean_pos0` at each row's FIRST REAL token, only in the prefill call (sequence length > 1). Batches are
     left-padded, so the first real token's index differs per row: capture the attention mask with a forward pre-hook on
     the model (kwargs `attention_mask`; it grows during generate, so take the first real index from the prefill call
     and remember the prefill length) and use it in the `o_proj` hook.
   - `mean_rest` at every other prefill position, at pad positions, and at every decode call (sequence length 1, whose
     local index 0 is NOT absolute position 0).
   - Receipt (first batch): for each head, exact bf16 byte equality with `mean_pos0` at every row's first real prefill
     token, with `mean_rest` at every other real prefill position, and with `mean_rest` at the first decode step;
     plus `max_abs_diff_before` > 0 at each of the three so a reader sees the slice changed. Record the means file's
     SHA-256 in the run's meta.
   - An empty head set in `mean` mode installs the hooks and changes nothing (R0b's do-nothing check on this path).
2. **`part219/means219.py`.** Computes `results/part219/head-means.pt` exactly as the amendment's "Reference set"
   says: the 24 `general` questions of `Journal/notes/reports/attention-heads-screens-2026-11-18/a-olmo-stages/
   questions.jsonl` (pass the path as an argument; assert its SHA-256 starts `1eabf3aba5824428`) plus Part 99's
   questions 4 and 5 (take them from where swap99 gets Part 99's questions, not retyped). For each: Part 99's chat frame,
   greedy answer from the unedited host with Part 99's max-new-tokens and settings (batched, left padding, same
   decode helper), then teacher-force prompt+answer through the host with NO hooks except a capture pre-hook on
   `o_proj` for layers 0 to 3, and accumulate per head, in float32, the sum over each sequence's first real token
   (`pos0`) and over its later real tokens (`rest`), ignoring pads. Save a dict
   `{"L0H10": {"mean_pos0": tensor[128] float32, "mean_rest": tensor[128] float32, "n_pos0": int, "n_rest": int}, ...}`
   for the four heads AND the R5 rig-check heads (L0H12, L1H11, L2H5, L3H9), plus metadata (host path, questions
   SHA-256, the 26 prompt texts and answers, n_new_tokens). The hook casts the float32 means to the model dtype.
   Print a heartbeat line at least every 10 seconds.
3. **`part219/run219.py`.** Add specs R2m to R7m (same heads as R2 to R7, `--ablate-mode mean`), R0b through both the
   zero and the mean wrapper with empty sets, a `--means` argument, and a `--pilot-a1` mode that runs R2m only and
   reuses the existing `results/part219/R0.jsonl` and `replay-R0.json` (assert they exist; do not rerun R0). The pilot
   verdict for R2m uses the same rule as the registered pilot (stop if fewer than 15 of 21 slot answers coherent or any
   receipt fails); refuse the stage-a main run if the R2m pilot verdict is missing or STOP. Write the steps file
   `part219/steps219-pilot-a1.txt` in the style of `steps219-pilot.txt` (means219 first, then the R2m pilot).
4. **`part219/score219.py`.** Score 219.1m and 219.2m exactly as 219.1 and 219.2 with R2m in place of R2, R5m as the
   rig check, and record 219.2 as failed and 219.1 as not read from the pilot R2 (the amendment says so). Do not change
   the original rows' logic.

## Tests (add to `part219/test_part219.py`; all must pass with `-I`)

- Mean hook on a tiny random OLMo 2 with a LEFT-PADDED batch of two prompts of different lengths: the slice equals
  `mean_pos0` at each row's own first real token, `mean_rest` elsewhere, and `mean_rest` at decode steps. Then
  sabotage it: a version that writes `mean_pos0` at local index 0 of every call must fail the test (show it red once
  and record the output in REPORT-a1.md, then restore).
- Empty head set in mean mode: generations identical to no hooks.
- means219 on the tiny model: the saved means equal a plain numpy recomputation over the same captured slices;
  pads excluded (a fixture with padding must give the same means as the same rows unpadded).
- run219 `--pilot-a1` refuses when R0.jsonl is missing; the stage-a run refuses without a passing R2m pilot verdict.
- The whole existing suite still passes.

Run the tests and paste the final counts into REPORT-a1.md.
