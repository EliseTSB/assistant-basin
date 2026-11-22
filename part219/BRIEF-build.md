# Build brief: Part 219 runner, receipts and scorer

You are building the code for a REGISTERED research design. The design is the contract: read
`PART219-DESIGN.md` (repo root of this worktree) in full first, then `part99/swap99.py`, `part92/common92.py`,
`part101/read101.py`, `part99/label_items99.py`, `tests/test_olmo2.py` (tiny OLMo 2 fixtures) and `PROCEDURE.md`
sections 2 and 5. Do NOT change the design, PREDICTIONS.md, or any prediction or threshold. If the design is ambiguous
or a requirement cannot be built as written, stop and say so in REPORT.md rather than choosing.

Work only inside this worktree (`/home/elisepc/green/worktrees/ab-part219`). Do not load the real OLMo weights, do not
use the GPU, do not run any model larger than a tiny random one, and do not touch results of earlier parts. Do not
commit; the reviewer commits. Write `part219/REPORT.md` early (a skeleton within your first ten steps) and fill it as
you go; end it with SUMMARY, FINDINGS, BLOCKERS.

## What to build

1. **`part99/swap99.py`, additive options only.** Every existing default must produce byte-identical behaviour (Parts
   99 to 104 reuse it). Add:
   - `--attn-implementation {default,eager}` (default `default` = today's behaviour; `eager` passes
     `attn_implementation="eager"` to every `from_pretrained` in this run, host and donor, via `part92/common92.py`'s
     `load_model` and `_load_donor`; record the value in the run's meta).
   - `--ablate-heads L0H10,L1H4,...`: zero the 128-wide slice `[128h:128(h+1)]` of each named layer's `o_proj` INPUT
     at every position, with a forward pre-hook. Head size must come from the model config
     (`hidden_size / num_attention_heads`), not a literal; refuse if `num_key_value_heads != num_attention_heads`.
   - `--transplant-heads L0H10,...` with `--donor` and `--transplant-mode {donor,opposite}`: per head, rows
     `[128h:128(h+1)]` of `q_proj`, `k_proj`, `v_proj`, columns `[128h:128(h+1)]` of `o_proj`, and the same 128-wide
     slices of `q_norm.weight` and `k_norm.weight` are set to the donor's values (`donor`) or to
     `2 * host - donor` computed in float32 and cast to bf16 (`opposite`). Refuse if the donor's shapes differ.
   - Receipts, written into the run's meta JSON (fields named in the design's Controls section):
     - intervention receipt: for ablation, the hooked slice is exactly zero on the first batch; for transplant, the
       slices equal the donor bytes (`donor`) or the float32-then-bf16 `2I - D` (`opposite`);
     - coupling log for transplant runs: first-batch relative change in rms(q), rms(k) (after the projections, before
       the norms) and rms(attention output before `post_attention_layernorm`) for each changed layer, the drift of the
       sibling heads' outputs, and the transplanted slices' `||W - W_0||` and `||W|| / ||W_0||`;
     - reset receipt: after the run's generations, remove the hooks or restore the host's original bytes, replay one
       stored prompt and record its token ids and first-step logits, so the orchestrator can compare them with R0's.
   - `--reference-replay <file>`: where to write/read that stored prompt's R0 tokens and logits.
2. **`part219/run219.py`**: the orchestrator for R0 to R10 exactly as the design's Runs table (head lists, the frozen
   R5/R9 heads L0H12, L1H11, L2H5, L3H9, asserted), batch 48, Part 99's fixed pushes, chat frame, kinds map and
   neuron list, eager everywhere. R0, R0b, R2 to R7 in one process with one host load; R1, R8, R9, R10 in a second
   process with the donor. It checks R0b equals R0 token for token, and every reset receipt equals R0's replay exactly;
   a failed receipt marks that run unreadable in a manifest. `--pilot` runs only R0 and R2 and applies the design's
   pilot stopping rule. It must be launchable by `tools/launch_arc.py 219` (read that tool) and print a progress line at
   least every 10 seconds.
3. **`part219/score219.py`**: reuses `part101/read101.py`'s functions (read set, relative counts, inconclusive flag);
   adds the 16 upstream / 5 downstream split (compute it from `results/part92/step2/neurons.json` and the head layers,
   do not hard-code the unit list), net counts against R0, exact two-sided McNemar p, the per-run two-pass gate on the
   42 slot items (agree on at least 40 of 42 and slot counts within 2), the gate, every reading rule and the headline
   rules, and writes `results/part219/scorecard.json` and a short `SCORECARD.md`.
4. **Tests** (`part219/test_part219.py`, pytest): on a tiny random OLMo 2 (tests/test_olmo2.py's fixture, with
   num_key_value_heads == num_attention_heads), run the real swap99 code paths end to end on CPU: default options give
   the same output as before the change; an empty ablation set and an empty transplant set give bit-identical logits;
   ablation zeroes exactly the named slice and nothing else; transplant `donor` and `opposite` write exactly the
   expected bytes into exactly the named slices; the reset receipt restores bit-identical logits; the coupling log
   fields are present and finite. Scorer tests on synthetic label files with known answers, including a case where a
   pooled gate would pass but the 42-item gate fails, and a case where 219.3 holds but R8 is broken so it must read
   inconclusive. For each test, show it goes red when its code is sabotaged (say how in REPORT.md).

Run the suite with `python -I -m pytest part219 part99 -q` and paste the counts in REPORT.md.
