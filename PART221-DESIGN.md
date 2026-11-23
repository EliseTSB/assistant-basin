# Part 221: is the rest of early attention, without the four big heads, what OLMo 2 1B's slot neurons need?

Procedure: v1
Draft v3, 2026-11-22, by Journal Claude (Opus 5.5), before any data. Not registered. It follows Part 219, whose
result and the Gemini Deep Research prior-art report (citations checked at source,
`Journal/notes/reports/gemini-handoffs-2026-11-22/citecheck/CITECHECK.md`, 13 of 13 found) both point at this
experiment. v2 folds the DeepSeek design review r1 (SHIP-WITH-FIXES: 4 High, 4 Medium, 8 Low;
`Journal/scratch/review-part221-design-ds-r1-2026-11-22/REVIEW.md`). v3 folds Vesper's review (SHIP-WITH-FIXES, council
`vesper-part221-design-2026-11-22` #3: 2 High, 3 Medium; her optional norm-preserving arm left out of scope). The commit that adds this
file to public main will be the registration.

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

The venue claim for one of these papers (Li and Janson) was not confirmed at source; only the findings above are
relied on.

## Method

Everything not stated here is Part 219's method, unchanged: host `allenai/OLMo-2-0425-1B-Instruct` at
48d788eca847d4d7548f375ad03d3c9312f6139e, donor `allenai/OLMo-2-0425-1B` at f83cd641d4f00954eb25c781e0ac360debb3239e
(the mid-training end, NOT `main`), bf16, the A770, `attn_implementation="eager"`, Part 99's prompts, chat frame,
fixed pushes, neuron list and Part 92 kinds map, greedy decoding, batch 48, Part 219's `transplant_heads` (donor and
opposite modes, with their slice receipts and coupling logs), reset receipts between runs, Part 101's read set and Part
92's rubric in two passes plus a 10% DeepSeek V4.1 Flash sample. Software pinned to Part 219's: torch 2.13.0+xpu and
transformers 5.13.0 (`results/part219/R1.meta.json`); each run's meta records both.

Reader, pinned to Part 219's so every count compares like with like: Claude Haiku 5.5 through OpenCode Go (the route
Part 219 used, since the direct API key returns 401) with Part 92's prompt and a 500-token cap; an item that returns no
JSON is retried at 500 once and then labelled at 4,000. Each label row records its cap (Part 219's did not).

The complement of layer l is its 16 heads minus the four (so 15 heads in each of layers 0 to 3, 60 in all).

| run | change to Instruct | role |
|---|---|---|
| R0 | none | reference; must equal Part 219's R0 token for token |
| R1 | layers 0 to 3 attention from the donor (Part 219's R1 code path) | gate; must equal Part 219's R1 token for token |
| R13 | all 64 heads of layers 0 to 3 through `transplant_heads` (donor mode) | identity check: the same weights as R1, by the other code path |
| R11 | the 60 complement heads from the donor; the four stay Instruct's | 221.1, 221.2 |
| R12 | the 60 complement heads moved the opposite way by the same distance (2 x Instruct - donor) | 221.3 (distance control) |
| R14 to R17 | the 15 complement heads of one layer (0, 1, 2, 3) from the donor | exploratory: which layer |

R13 is a generation-path check, not an independent check of the transplant: its weights equal R1's by construction
(the six `self_attn` tensors per layer, no biases, `attention_bias: false`; the per-head slices of all 16 heads tile
them exactly, 67,125,248 elements either way, review r1 `ident.py`), and that equality is asserted before generation.
Its answers must equal R1's on all 474 rows. A difference means the two code paths differ in something other than the
weights; that is a gate failure (below).

R0 and R1 rerun Part 219's arms so this part stands on its own data. If both equal Part 219's rows exactly (expected:
same weights, same kernel, same batch shape), Part 219's labels for R0 and R1 are reused and the reruns are not
relabelled; if not, they are labelled like every other run and the difference is the first reported result.

## Controls

- Reproduction: R0 and R1 equal Part 219's R0 and R1 token for token (above).
- Identity: R13 equals R1 token for token.
- Distance (R12): the same size of change as R11 in the one opposite direction `2 x Instruct - donor`, Part 105's
  control. It samples one direction, not every direction; a per-weight random direction of the same size is a
  different control and is out of scope here. A 60-head opposite move across four layers is a large change and may
  break answers, and a run whose pushed answers break posts a net loss near the ceiling of 15 whatever its weights did
  (Part 219's R2 and R2m: net 14 at 2 and 8 of 21 coherent). So R12 must clear the coherence bar for 221.3 to be read.
  R12 is also asymmetric with R11 (Vesper's review): R11 steps to a trained, in-distribution checkpoint, while R12
  extrapolates 60 heads off it, where parameter norms can overshoot (`||2H - D||` need not equal `||D||`) and small
  `q_norm`/`k_norm` gains can turn negative. That bias works against 221.3 holding, so a held 221.3 is the stronger
  for it; but R12 losing as much as R11 cannot separate "any change of this size" from "an off-checkpoint
  extrapolation". R12's receipt logs, per tensor, `||W|| / ||W_0||` and the number of `q_norm` and `k_norm` elements
  whose sign flips. A norm-preserving random-direction arm would be the clean generic-distance control; it is out of
  scope here.
- Receipts: every transplanted slice equals the donor's bytes (R11, R13, R14 to R17) or `2 x Instruct - donor`
  computed in float32 and cast to bf16 (R12); every run's reset replay equals R0 exactly.
- Coupling, a read item: in OLMo 2 the QK norms take their RMS over the whole 2048-wide projection and the
  post-attention norm normalises the sum of all heads, so in R11 the four kept heads' q, k and residual contribution
  change although their weights do not (Part 219's R8, changing one head per layer, moved sibling outputs by up to
  8.8% and a layer's attention-output rms by 12.7%). Each transplant run logs, on the first batch, per layer the
  relative change in rms(q), rms(k) and rms(attention output), and per head the relative drift of each of the four kept
  heads' outputs (for R11 and R12, the four heads are named in the log). That drift is measured at `o_proj`'s input,
  before `o_proj` and the post-attention norm: at layer 0 it is same-layer QK-norm coupling only, and at layers 1 to 3
  it also carries the residual drift from earlier transplanted layers (in Part 219's R8 the sibling maximum grew from
  1.56% at layer 0 to 8.84% at layer 3). The post-attention norm's rescaling of the kept heads' residual write is
  tracked by the change in rms(attention output). Both the largest kept-head pre-norm drift and the largest
  |relative change in rms(attention output)| go into the headline sentence (Headlines). No run here separates the
  coupling from the complement's own change (Limits). (Vesper's review, v3.)
- Breakage: Part 219's coherence bar applies to every run that is read (R11, R12, R14 to R17): at least 15 of 21 slot
  answers coherent, at least 13 concept units keep their persona, and not flagged inconclusive (more than 3 of 6
  unpushed answers incoherent). Each run's unpushed "Who are you?" labels are reported. An exploratory run below the
  bar is marked unreadable and its count is not read as a layer effect.
- Not used: zeroing or mean-ablation (Part 219 showed both break pushed answers through these heads); a dose ladder;
  both push directions; format robustness (Part 219's reasons).

Gate (read before any prediction): Part 219's kernel check (`results/part219/kernel-check.json`, same model and
kernel) still passes `tools/part_check.py kernel`; R0 shows the persona for at least 17 of 21 slot units; R1 at most 5;
R0 and R1 reproduce Part 219; R13 equals R1. If any fails, every prediction is frozen and all arms are reported as
exploratory, with the failure first. A reproduction mismatch traced to a software version is reported as that, and the
part re-plans before reading anything.

## Size

- Ceiling: R0 shows the persona on 15 of the 16 upstream units (p92-32 does not), so at most 15 can be lost and one
  gained.
- Smallest effect worth knowing: R11 removing 7 of the 15 units R0 shows, net (exact McNemar p = 0.016 at 7 lost and 0
  gained), the same bar as Part 219, so the two parts' arms compare directly. The p-values are paired-unit statistics:
  the units share the transplanted weights and one unpushed answer per question, and in Part 219 they failed in nested
  sets (R8's 4 losses all inside R1's 11), so they are not independent draws from a population.
- Additive expectation, stated so a result near it is not read as a surprise: from Part 219's kept labels, R1 lost
  {p92-02, 03, 04, 05, 13, 18, 24, 25, 36, 37, 47} and R8 lost {p92-05, 13, 18, 37}, all inside R1's set. If the four
  heads and the complement added, the complement would lose the other 7, which is 221.1's bar. 221.2's bar (R1 minus
  2, so 9 here if R1 reproduces) asks for 2 more than additive.
- Total: 9 runs x 474 = 4,266 generations. Part 219's R0b took 134 s and its transplant runs with the coupling probe up
  to 221 s (R9) on the A770, so about 30 minutes. Labels: 6 runs (R11, R12, R14 to
  R17) x 258 x 2 Haiku passes plus about 160 DeepSeek labels; R0, R1 and R13 are labelled only if not identical to the
  rows they must match.
- Pilot: R0, R1 and R13 (the reproduction and identity checks). Stop and re-plan if any of them fails.

## Optimization

- Bound resource: the A770; one model pair per process (the 11-08 gotcha), host and donor loaded once for every run.
- Measured rate above; Windows needs about 10 GB free before the load (Part 219's launch waited for it).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 221.1 | R11 (the 60 complement heads from the donor): on the 16 upstream slot units, at least 7 net lose their persona against R0 | medium (55%) | Journal Claude |
| 221.2 | R11's net loss on the 16 upstream units is at least R1's net loss minus 2 (at least 2 more than the additive share; Size) | low (35%) | Journal Claude |
| 221.3 | 221.1 holds, and on the 16 upstream units the exact two-sided McNemar test of R11 against R12 gives p < 0.05 with R11 losing more, where a unit "shows" in a run if its pushed answer shows the persona and that run's own unpushed answer does not (`score219.py`'s indicator); the discordant pairs are units showing in R12 and not R11, against the reverse (the direction of post-training's change matters, not only its size) | low (25%) | Journal Claude |

## Reading rules

Counts are paired and relative to each run's own unpushed answers, as in Part 219; `score219.py`'s functions are
reused with the new run names. Reader agreement is gated per run on the 42 slot items, as in Part 219: the two Haiku
passes must agree on at least 40 of 42 and give slot counts within 2. A run that fails it is not read, and a prediction
resting on it is reported as unreadable (not inconclusive, and not scored on pass 1).

In order:
- If the gate fails, every prediction is frozen and every arm is exploratory, the failure first.
- 221.1 and 221.2 are read only if R11 clears the coherence bar; otherwise "inconclusive: the complement transplant
  breaks answers". 221.2 uses this part's R1.
- 221.3 is read only if 221.1 holds and R12 clears the coherence bar; if R12 fails it, "inconclusive: the
  opposite-direction transplant breaks answers", and no direction clause is written.
- R11's and R12's lost and gained unit lists, and their overlap, are reported with every verdict.

Exploratory, with unit lists and exact McNemar p: R14 to R17 against R0 (each marked unreadable if below the coherence
bar); the 5 downstream units (for them the complement includes heads after the push); all-21 counts; R11's net loss
plus Part 219's R8 net loss against R1's (a sum, not a decomposition: the parts interact through the shared norms).

Headlines, decided by the verdicts, each stated as the intervention it is: "with the four heads' weights left as
Instruct's, under an early attention whose normalisation context and other heads changed", and with the largest
kept-head pre-norm drift and the largest |change in rms(attention output)| from the coupling log in the same sentence:
- 221.1 and 221.2 hold: "transplanting the other sixty early heads removes nearly as much as the whole band does (at
  least the band's net loss minus 2)".
- 221.1 holds, 221.2 fails: "transplanting the other sixty removes much of what the whole band removes, the whole band
  more".
- 221.1 fails: "neither transplanting the four heads alone nor transplanting the other sixty alone removes most slot
  personas; the whole band's change does". Whether that is additive or more than additive is left to the exploratory
  R11 + R8 against R1 comparison.
- With 221.3 holding, add "and post-training's direction matters, not only the size of the change". With 221.3 failing
  on its test while R12 is coherent, add only "R12, the one opposite direction tested, removed R11's count within the
  test's resolution", which is not a claim that direction does not matter.

## Limits

- R11 changes 60 heads and R8 changed 4, in separate runs; their effects need not add, because the heads share the QK
  norms' RMS and the post-attention norm. Neither part apportions the layer swap between them.
- In R11 the four large heads keep Instruct's weights while the QK-norm RMS and the post-attention norm they share
  with their siblings see donor-derived values; that rescaling is logged and reported, not removed. A run that left
  the norm slices at Instruct's would not remove it either, because the RMS denominator is shared.
- R12 tests one opposite direction, and an off-checkpoint one with possibly different parameter norms; a
  norm-preserving random direction of the same size, the control for generic disruption, is not run.
- One model, one donor checkpoint, one rubric; 21 slot units, 16 read for the predictions; the four heads were chosen
  by screen A, which was not registered.
- The concept units sit at other layers than the slot units, so they control for personas in general, not for early
  disruption at the slot units' depth.
