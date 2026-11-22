# Part 219: are four early heads the part of early attention that OLMo 2 1B's slot neurons need?

Procedure: v1
Registered 2026-11-22 12:02 CT by Journal Claude (Opus 5.5), before any data; design v4. Elise asked for it after reading about
screen A: "Go for it!" (Discord 1574113207369334966, 2026-11-22 11:23 CT). v2 folds the DeepSeek design review r1
(SHIP-WITH-FIXES, 6 blocking and 10 serious findings; `Journal/scratch/review-part219-design-r1-ds-r1-2026-11-22/
REVIEW.md`); v3 adds Part 105's distance control; v4 folds Vesper's review (SHIP-WITH-FIXES, council topic
`vesper-part219-design-2026-11-22` #4). Vesper's review is the research review (Elise 2026-11-21: research reviews to Vesper and DeepSeek). The commit that
adds this file to public main is the registration.

## Question

Parts 99 to 102 found that OLMo 2 1B Instruct's 21 "slot" neurons, pushed, describe a persona on "Who are you?" only if
the first four attention layers are Instruct's (swapping in the mid-training end's attention for layers 0 to 3 drops
them from 19 of 21 to 3 of 21), and that SFT built most of that change. Screen A (2026-11-18, not registered) zeroed one
head at a time and found four heads in exactly those layers, L0H10, L1H4, L2H6 and L3H5, whose loss hurts Instruct's
identity answers more than its ordinary ones. This part asks two things: are those four heads needed for the slot
neurons' self-descriptions (zeroing), and is post-training's change to them needed (transplanting just those heads'
weights from the mid-training end)?

## Prior art and existing data

- Closest published work: treetown's L17H4 on Qwen3-8B, a head they report resolves who "I" refers to; our Part 180
  check on it came back flat (45% persona claimed, 38% with the head off). Muse's 11-03 search found nobody had
  published a head that both resolves "I" and carries persona content. No published work on OLMo 2 1B's early heads
  and persona neurons found.
- Our own data: `results/part99` to `results/part102` (layer-range attention swaps, read by Part 92's blind Haiku
  reader); `Journal/notes/reports/attention-heads-screens-2026-11-18/` screen A (`a-olmo-stages/a-out/REPORT.md`,
  `scores.json`). ⚠ Parts 99 to 102 loaded the model with the default attention kernel (`part99/swap99.py:143` passes
  no `attn_implementation`), so this part reruns its reference and the layer-swap arm on eager.

Screen A's table for the early layers (Instruct, nats per token):

| head | drop on identity answers | drop on ordinary answers | specificity |
|---|---|---|---|
| L2H6 | 0.428 | 0.335 | 0.093 |
| L3H5 | 0.342 | 0.257 | 0.085 |
| L0H10 | 0.256 | 0.220 | 0.036 |
| L1H4 | 0.311 | 0.211 | 0.100 |
| next largest (L0H8) | 0.033 | 0.034 | 0.000 |

These four are the four most important early heads for any answer; no other early head comes within a factor of six on
ordinary answers, so no early head is a control matched on general importance. The design answers that with a
coherence gate (a drop that comes with broken answers is not read), with a transplant arm and its own matched control
(R8 against R9, below), and by reading the specificity contrast between L1H4 and L0H10 only as exploratory: the review
showed it also differs in layer, and Part 101 found layer 0 the strongest early layer, so the layer prior and the
specificity prior point opposite ways, and screen A's specificity gap is only 0.065 nats (t = 2.6 over 24 texts).

## Method

`allenai/OLMo-2-0425-1B-Instruct` at 48d788eca847d4d7548f375ad03d3c9312f6139e (host); donor for R1, R8, R9 and R10 the
mid-training end `allenai/OLMo-2-0425-1B` at f83cd641d4f00954eb25c781e0ac360debb3239e (NOT `main`, a different run).
bf16, the A770, `attn_implementation="eager"` on every load, recorded in each run's meta. Part 99's prompts, chat frame,
fixed pushes (`results/part99-pushes.pt`), neuron list and Part 92 kinds map; greedy decoding with Part 99's settings;
batch 48 in every run.

Kernel check: results/part219/kernel-check.json, a tools/kernel_reference.py record of the host decoded on the A770
under eager against a CPU float32 reference made for this part (none exists yet for OLMo 2 1B); it must pass
`tools/part_check.py kernel`, and if it cannot, that is the first reported result and nothing below is read.

Zeroing: a head is zeroed by setting its 128-wide slice of `o_proj`'s input to zero at every position (screen A's
operation). Transplant: the head's rows `[128h:128(h+1)]` of `q_proj`, `k_proj` and `v_proj`, columns `[128h:128(h+1)]` of
`o_proj`, and the head's own 128-wide slices of the `q_norm` and `k_norm` weights are copied from the donor (the norm
weights are per dimension, so those slices belong to the head; Vesper's review). ⚠ Two couplings remain that no slice
choice removes: `q_norm`/`k_norm` take their RMS denominator over the whole 2048-wide projection, and OLMo 2's
`post_attention_layernorm` normalises the sum of all heads' outputs, so a transplanted head shifts its 15 siblings'
scale a little at both points. R9 measures that with four unimportant heads, and R10 with the same four important
heads; each transplant run's receipt logs the first-batch relative change in rms(q), rms(k) and rms(attention output),
the drift of the sibling heads' outputs, and the transplanted slices' ||W - W_0|| and ||W|| / ||W_0||.

Slot units by position: a head in layer l precedes a push that enters after layer L's MLP when l ≤ L. 16 of the 21
slot units (layers 3 to 12) sit downstream of all four heads, so every arm here changes only the context their push
lands in; the other 5 (p92-07 at layer 1; p92-10, p92-20, p92-42, p92-50 at layer 2) have L3H5, and for p92-07 also
L2H6, after their push. Registered predictions are read on the 16; the 5 are reported separately (Part 100 made the
same cut).

Runs (`swap99.py`, with `--ablate-heads` and `--transplant-heads` as the new options):

| run | change to Instruct | role |
|---|---|---|
| R0 | none | reference |
| R0b | none, through both new hooks with empty head sets | do-nothing check |
| R1 | mid-training end's attention in layers 0 to 3 (Part 100's arm, on eager) | gate and comparison |
| R2 | zero L0H10, L1H4, L2H6, L3H5 | 219.1, 219.2 |
| R3 | zero L1H4 | exploratory |
| R4 | zero L0H10 | exploratory |
| R5 | zero L0H12, L1H11, L2H5, L3H9 | rig check |
| R6 | zero L2H6 | exploratory |
| R7 | zero L3H5 | exploratory (downstream of the push for 5 units) |
| R8 | transplant L0H10, L1H4, L2H6, L3H5 from the donor | 219.3, 219.4 |
| R9 | transplant L0H12, L1H11, L2H5, L3H9 from the donor | coupling check |
| R10 | the four heads' slices (including their q_norm/k_norm slices) moved the opposite way by the same distance: 2 x Instruct - donor | distance control for R8 (Part 105) |

R10 is Part 105's control. There an attention swap moved the opposite way by the same distance lost 5 of 10 early slot
personas against 7 of 10 for the SFT swap, so a transplant that removes personas may only show that these answers break
under any change of that size. R10 keeps the distance and reverses the direction.

R5's and R9's heads are frozen here: `random.Random(219)`, one `rng.choice` per layer 0 to 3 over that layer's heads
with ordinary-answer drop under 0.035, excluding the four, sorted by head index (15 eligible per layer). Drops on
ordinary answers: L0H12 0.0103, L1H11 0.0116, L2H5 -0.0005, L3H9 0.0045. The runner asserts the list.

Reading: Part 101's read set (`part101/read101.py filter`: per run 21 pushed and 21 baseline slot items on "Who are
you?", 216 concept items), labelled blind by Part 92's Haiku reader in two independent passes over the same items. Pass
1 is scored; pass 2 measures agreement. A DeepSeek V4.1 Flash reader with Part 92's rubric labels a fixed 10% sample of
every run's read set (seed 219) as the second reader.

## Controls

- Do-nothing: R0b must match R0 token for token on every item.
- Reset receipt: after every run that changes the model (R1 to R10), the hooks or transplanted weights are removed, a
  stored prompt is replayed, and its token ids and first-step logits must equal R0's exactly, so a leaked hook or weight
  cannot taint the next run in the same process. A run whose receipt fails is not read.
- Intervention receipt: in each zeroing run the hooked `o_proj` input slice is checked to be zero on the first batch,
  in R8 and R9 the slices are checked to equal the donor's bytes, and in R10 to equal 2 x Instruct - donor computed in
  float32 and cast to bf16.
- Rig check (R5): zeroing four unimportant early heads must give |net loss against R0| at most 2 on the 16 upstream
  units. A failure means a bug or an unexpectedly fragile readout, and freezes 219.1 and 219.2.
- Distance control (R10): see 219.5 and the headlines; without it, R8 cannot separate post-training's change from
  any change of the same size.
- Coupling check (R9): transplanting four unimportant heads must give |net loss against R0| at most 2 on the 16, in
  either direction. If not, the norm coupling and not the heads moves the readout, and 219.3 to 219.5 are not read.
- Breakage: coherence of slot answers and the unpushed answers is counted in every run (Part 99's reader fields); Part
  101's `inconclusive` flag (more than 3 of 6 unpushed answers incoherent) makes a run inconclusive, and every run's
  unpushed "Who are you?" label is reported, since a persona-showing baseline would empty the paired count for the wrong
  reason.
- Concept units: the 18 concept neurons in the same runs, as in Parts 99 to 102 (not layer-matched to the slot units,
  see Limits).
- Not used: both push directions and a dose ladder, because the pushes are Part 99's fixed ones and the question is
  about the route. Not used: the answer-polarity split, because there is no yes/no answer. Not used:
  STANDARD-CONTROLS item 9 (format robustness), because the pushes were built and calibrated in Part 99's chat frame;
  a different frame would change what the pushes do before it tested the heads.

Gate (read before any prediction): the kernel check passes; R0 shows the persona for at least 17 of 21 slot units;
R1 shows it for at most 5 of 21 (Part 100: 3; Part 101's single layer 0: 7). If any part fails, every prediction is
frozen and all arms are reported as exploratory, with the failure first.

## Size

- Smallest effect worth knowing: R2 removing 7 of the 16 upstream units' personas net. With paired units the exact
  McNemar test gives p = 0.016 at 7 lost and 0 gained, and p = 0.07 at 7 against 1.
- Total: 5,688 generations (12 runs x 474, measured from `results/part99/inst_none.jsonl`) and about 6,500 labels
  (11 labelled runs x 258 x 2 Haiku passes, plus 284 DeepSeek labels; R0b is labelled only if it is not bit-identical
  to R0), about 44 minutes on the A770 (Part 99's R0 took 114 s on the
  default kernel; eager is measured in the pilot and this line is updated before the full run).
- Pilot: R0 and R2 in full (948 generations). Stop and re-plan if R0 shows fewer than 17 of 21, if R2 leaves fewer
  than 15 of 21 slot answers coherent, or if either receipt fails.

## Optimization

- Bound resource: the A770.
- Measured: 474 rows in 114 s at batch 48 (Part 99's `inst_none` on the default kernel,
  `results/part99/inst_none.meta.json`); the eager rate is measured on the pilot and replaces this line.
- Levers: one host load per process for R0, R0b, R2 to R7 (only the hook changes), taken; R1, R8, R9 and R10 need the donor,
  loaded once in a second process (one model pair per process, 11-08 gotcha), taken.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 219.1 | R2 (all four zeroed): on the 16 upstream slot units, at least 7 net lose their persona against R0 (units showing it in R0 and not R2, minus the reverse) | medium (55%) | Journal Claude |
| 219.2 | R2: at least 15 of 21 slot answers coherent, at least 13 concept units keep their persona, and the run is not flagged inconclusive | medium (50%) | Journal Claude |
| 219.3 | R8 (the four heads' weights from the mid-training end): on the 16 upstream units, at least 7 net lose their persona against R0 | low (30%) | Journal Claude |
| 219.4 | R8's net loss on the 16 upstream units is at least R1's net loss minus 2 (the four heads' transplant accounts for the layer swap) | low (20%) | Journal Claude |
| 219.5 | R8's net loss on the 16 upstream units is at least 4 more than R10's (the direction of post-training's change matters, not only its size) | low (30%) | Journal Claude |

Copied verbatim into PREDICTIONS.md under "## Part 219 (registered ...)".

## Reading rules

Counts are paired and relative to each run's own unpushed answers, as in Part 99 (Amendment 2) and Part 101's scorer;
`score219.py` reuses `part101/read101.py`'s functions and adds the upstream split, the net count and the McNemar p.

Reader agreement is gated per run on the 42 slot items, not the pooled 258: the two Haiku passes must agree on at least
40 of 42 and give slot counts within 2 of each other, or that run is not read. Error rows go through Part 99's retry
before agreement is computed. Agreement with the DeepSeek reader is reported for every run and does not gate.

219.1 is read only if 219.2 holds; otherwise it is reported as "inconclusive: zeroing these heads breaks answers".
219.3 to 219.5 are read only if R8 clears the same bar as 219.2 (at least 15 of 21 slot answers coherent, at least 13
concept units keep their persona, not flagged inconclusive); otherwise "inconclusive: the transplant breaks answers".
R10's coherence is reported beside it; a broken R10 makes 219.5 harder to hold, not easier. 219.4 is read only if
219.3 and 219.5 both hold; otherwise it is reported as not read.
219.3, 219.4 and 219.5 are read only if R9 passes its coupling check. 219.4 is read only if the gate's R1 bound held.
Exploratory, reported with unit lists and exact McNemar p: R3 against R4, R6, R7, the 5 downstream units, and all
21-unit counts.

Headlines, decided by the verdicts:
- 219.1 and 219.2 hold: "these four heads are needed for the slot neurons' self-descriptions"; otherwise no headline
  about need.
- 219.3 and 219.5 hold: "post-training's change to these four heads is needed". Then, with 219.4 holding, "and it
  accounts for the whole early-layer swap"; with 219.4 failing, "it is part of what the early-layer swap removes, not
  all".
- 219.3 holds and 219.5 fails: "these heads' answers break under a change of that size in either direction", with no
  claim about what post-training built and 219.4 not read.
- Any headline is withdrawn if R5 fails its rig check or a receipt fails.

## Limits

- Zeroing shows a head is needed; the transplant shows post-training's change to it is needed. Neither shows what the
  heads compute.
- The transplant is not perfectly local: the full-width RMS of the QK norms and the post-attention norm couple heads,
  which R9, R10 and the receipts measure but cannot remove. R9's heads have smaller outputs than R8's, so it bounds the
  coupling less well than R10 does.
- The four heads were chosen on Instruct's identity answers; the test uses pushed slot neurons on a different question
  set, but screen A itself was not registered, and the heads are also the most important early heads for everything.
- 21 slot units, 16 of them read for the predictions; one model; one rubric.
- The concept units sit at other layers than the slot units, so they control for "personas in general" more than for
  "early disruption at the slot units' depth".
