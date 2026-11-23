# Part 221: is the rest of early attention, without the four big heads, what OLMo 2 1B's slot neurons need?

Procedure: v1
Draft v1, 2026-11-22, by Journal Claude (Opus 5.5), before any data. Not registered. It follows Part 219, whose
result and the Gemini Deep Research prior-art report (citations checked at source,
`Journal/notes/reports/gemini-handoffs-2026-11-22/citecheck/CITECHECK.md`, 13 of 13 found) both point at this
experiment. Reviews to come: a DeepSeek design review and Vesper. The commit that adds this file to public main will be
the registration.

## Question

In OLMo 2 1B Instruct, 21 "slot" neurons, when pushed, describe a persona on "Who are you?". Part 219 (on the eager
kernel) found:

- Putting back the pre-SFT (mid-training end) attention of all of layers 0 to 3 (R1) leaves 4 of 21 slot personas,
  against 20 of 21 in Instruct; a net 11 lost on the 16 units whose push enters after layer 3.
- Putting back only the four most important early heads (L0H10, L1H4, L2H6, L3H5; R8) leaves 13 of 21, a net 3 lost.
- Moving those four the opposite way by the same distance (R10) loses a net 2.

The four heads together with the other 60 heads of layers 0 to 3 are exactly R1's change: R1 replaces every weight of
`self_attn` in those layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `q_norm`, `k_norm`), and the per-head slices of
those six tensors cover them completely. So this part asks the complementary question: does putting back the other 60
heads, with the four left as Instruct's, remove what R1 removes? And is it post-training's direction of change that
matters, or any change of that size?

## Prior art and existing data

- Gupta, Zhang and Jin, arXiv:2610.22937 (OLMo-2-7B): copying base weights into the top 8 heads barely changes the
  aligned model's cue susceptibility, while swapping attention across a six-layer band moves suggested-answer cue
  effects 61% to 73% of the way between checkpoints. The same asymmetry as Part 219, at 7B, in middle layers, for a
  different behaviour. They did not transplant the band's complement.
- Sun et al., arXiv:2604.13694 (Weight Patching, Llama-3.2-3B): ablating a few top-ranked heads sharply degrades
  performance, yet restoring those heads alone recovers little.
- Laubrock, arXiv:2611.26584 (OLMo 2 7B, base, SFT, DPO, RL checkpoints): the post-training change to how early layers
  are read out is distributed over the early layers and involves both attention and feed-forward blocks.
- Zhou, arXiv:2605.07284: a late-layer change from post-training may depend on earlier computation learned with it;
  tests a late stack under base and descendant upstream states. Our slot neurons sit downstream of the swapped layers,
  so this part is the upstream-state version of that test for one behaviour.
- Li and Janson, arXiv:2409.09951 (optimal ablation): a learned constant per component instead of zero or the mean.
  Not used here (this part transplants, it does not ablate); kept for a later part if removal is needed again.
- Our data: `results/part219` (same model, same donor, same prompts, eager kernel, same reader and rubric); Parts 99
  to 102 and 105 (layer-range swaps, the opposite-direction attention control).

Venue claims for two of these papers were not confirmed at source; only the findings above are relied on.

## Method

Everything not stated here is Part 219's method, unchanged: host `allenai/OLMo-2-0425-1B-Instruct` at
48d788eca847d4d7548f375ad03d3c9312f6139e, donor `allenai/OLMo-2-0425-1B` at f83cd641d4f00954eb25c781e0ac360debb3239e
(the mid-training end, NOT `main`), bf16, the A770, `attn_implementation="eager"`, Part 99's prompts, chat frame,
fixed pushes, neuron list and Part 92 kinds map, greedy decoding, batch 48, Part 219's `transplant_heads` (donor and
opposite modes, with their slice receipts and coupling logs), reset receipts between runs, Part 101's read set and Part
92's Haiku reader in two passes plus a 10% DeepSeek sample.

The complement of layer l is its 16 heads minus the four (so 15 heads in each of layers 0 to 3, 60 in all).

| run | change to Instruct | role |
|---|---|---|
| R0 | none | reference; must equal Part 219's R0 token for token |
| R1 | layers 0 to 3 attention from the donor (Part 219's R1 code path) | gate; must equal Part 219's R1 token for token |
| R13 | all 64 heads of layers 0 to 3 through `transplant_heads` (donor mode) | identity check: the same weights as R1, by the other code path |
| R11 | the 60 complement heads from the donor; the four stay Instruct's | 221.1, 221.2 |
| R12 | the 60 complement heads moved the opposite way by the same distance (2 x Instruct - donor) | 221.3 (distance control) |
| R14 to R17 | the 15 complement heads of one layer (0, 1, 2, 3) from the donor | exploratory: which layer |

R13 checks the machinery. Its weights are R1's bit for bit (asserted before generation: every `self_attn` tensor of
layers 0 to 3 equals the donor's), so its answers must equal R1's on all 474 rows. A difference means the two code paths
differ in something other than the weights, and nothing below is read.

R0 and R1 rerun Part 219's arms so this part stands on its own data. If both equal Part 219's rows exactly (expected:
same weights, same kernel, same batch shape), Part 219's labels for R0 and R1 are reused and the reruns are not
relabelled; if not, they are labelled like every other run and the difference is the first reported result.

## Controls

- Reproduction: R0 and R1 equal Part 219's R0 and R1 token for token (above).
- Identity: R13 equals R1 token for token.
- Distance (R12): the same size of change as R11 in the opposite direction, Part 105's control. A 60-head opposite move
  across four layers is a large change and may break answers; R12's coherence is reported beside it, and broken answers
  count as lost personas, which makes 221.3 harder to hold, not easier.
- Receipts: every transplanted slice equals the donor's bytes (R11, R13, R14 to R17) or `2 x Instruct - donor`
  computed in float32 and cast to bf16 (R12); every run's reset replay equals R0 exactly.
- Coupling: each transplant run logs the first-batch relative change in rms(q), rms(k) and rms(attention output) per
  layer, as in Part 219. In R11 the four large heads keep Instruct's weights while their 15 siblings change, so the
  shared post-attention norm rescales the four heads' contribution; the log reports by how much. No run here separates
  that from the complement's own change (Limits).
- Breakage: Part 219's coherence bar applies to R11 and R12: at least 15 of 21 slot answers coherent, at least 13
  concept units keep their persona, and not flagged inconclusive (more than 3 of 6 unpushed answers incoherent).
- Not used: zeroing or mean-ablation (Part 219 showed both break pushed answers through these heads); a dose ladder;
  both push directions; format robustness (Part 219's reasons).

Gate (read before any prediction): Part 219's kernel check (`results/part219/kernel-check.json`, same model and
kernel) still passes `tools/part_check.py kernel`; R0 shows the persona for at least 17 of 21 slot units; R1 at most 5;
R0 and R1 reproduce Part 219; R13 equals R1. If any fails, every prediction is frozen and all arms are reported as
exploratory, with the failure first.

## Size

- Smallest effect worth knowing: R11 removing 7 of the 16 upstream units' personas net (exact McNemar p = 0.016 at 7
  lost and 0 gained), the same bar as Part 219, so the two parts' arms compare directly.
- Total: 9 runs x 474 = 4,266 generations. Part 219's R0b and R0b-mean took about 2.5 minutes each on the A770
  (`results/part219/launch.jsonl`, 2026-11-22 17:53 to 17:58), so about 25 minutes. Labels: 6 runs (R11, R12, R14 to
  R17) x 258 x 2 Haiku passes plus about 160 DeepSeek labels; R0, R1 and R13 are labelled only if not identical to the
  rows they must match.
- Pilot: R0, R1 and R13 (the reproduction and identity checks). Stop and re-plan if any of them fails.

## Optimization

- Bound resource: the A770; one model pair per process (the 11-08 gotcha), host and donor loaded once for every run.
- Measured rate above; Windows needs about 10 GB free before the load (Part 219's launch waited for it).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 221.1 | R11 (the 60 complement heads from the donor): on the 16 upstream slot units, at least 7 net lose their persona against R0 | medium (60%) | Journal Claude |
| 221.2 | R11's net loss on the 16 upstream units is at least R1's net loss minus 2 (the complement alone reproduces the layer swap) | low (40%) | Journal Claude |
| 221.3 | R11's net loss on the 16 upstream units is at least 4 more than R12's (the direction of post-training's change matters, not only its size) | low (35%) | Journal Claude |

## Reading rules

Counts are paired and relative to each run's own unpushed answers, as in Part 219; `score219.py`'s functions are
reused with the new run names. Reader agreement is gated per run on the 42 slot items, as in Part 219.

221.1 to 221.3 are read only if the gate holds and R11 clears the coherence bar; otherwise "inconclusive: the
complement transplant breaks answers". 221.2 uses this part's R1. 221.3 is read whatever R12's coherence, with R12's
coherence reported beside it.

Exploratory, with unit lists and exact McNemar p: R14 to R17 against R0; the 5 downstream units (for them the
complement includes heads after the push); all-21 counts; R11's net loss plus Part 219's R8 net loss against R1's (a
sum, not a decomposition: the parts interact through the shared norms).

Headlines, decided by the verdicts:
- 221.1 and 221.2 hold: "the rest of early attention, without the four biggest heads, carries the dependence".
- 221.1 holds, 221.2 fails: "the rest of early attention carries much of it, the whole band more".
- 221.1 fails: "neither the four heads nor the other sixty alone remove most slot personas; the whole band's change
  does", which says the dependence is in how the heads combine.
- With 221.3 holding, add "and it is post-training's direction, not only its size"; with it failing, "a change of that
  size in either direction does as much".

## Limits

- R11 changes 60 heads and R8 changed 4, in separate runs; their effects need not add, because the heads share the QK
  norms' RMS and the post-attention norm. Neither part apportions the layer swap between them.
- In R11 the four large heads keep Instruct's weights while the norm they share with their siblings sees a changed
  sum; that rescaling is logged, not removed.
- One model, one donor checkpoint, one rubric; 21 slot units, 16 read for the predictions; the four heads were chosen
  by screen A, which was not registered.
- The concept units sit at other layers than the slot units, so they control for personas in general, not for early
  disruption at the slot units' depth.
