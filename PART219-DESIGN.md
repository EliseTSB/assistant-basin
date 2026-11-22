# Part 219: do the four big early heads carry the early attention that OLMo 2 1B's slot neurons need?

Procedure: v1
DRAFT v1, written 2026-11-22 11:25 CT by Journal Claude (Opus 5.5); not registered. Elise asked for it after reading
about screen A: "Go for it!" (Discord 1574113207369334966, 2026-11-22 11:23 CT). Registration is the commit that adds
the final version to public main, after review.

## Question

Parts 99 to 102 found that OLMo 2 1B Instruct's 21 "slot" neurons, pushed, describe a persona on "Who are you?" only if
the first four attention layers are Instruct's (swapping in the mid-training end's attention for layers 0 to 3 drops
them from 19 of 21 to 3 of 21), and that SFT built most of that change. Screen A (2026-11-18, not registered) found,
by zeroing one head at a time, that the heads whose loss hurts Instruct's identity answers more than ordinary answers
include four in exactly those layers: L1H4, L2H6, L3H5 and L0H10, all already strong at SFT. This part asks whether
those four heads are the part of early attention the slot neurons need, or whether the two findings only share a place.

## Prior art and existing data

- Closest published work: treetown's L17H4 on Qwen3-8B, a head they report resolves who "I" refers to; our Part 180
  check on it came back flat (45% persona claimed, 38% with the head off). Muse's 11-03 search found nobody had
  published a head that both resolves "I" and carries persona content. No published work on OLMo 2 1B's early heads
  and persona neurons found.
- Our own data: `results/part99` to `results/part102` (layer-range attention swaps, read by Part 92's blind Haiku
  reader); `Journal/notes/reports/attention-heads-screens-2026-11-18/` screen A (`a-olmo-stages/a-out/REPORT.md`,
  `scores.json`). ⚠ Parts 99 to 102 loaded the model with the default attention kernel (`part99/swap99.py:143` passes no
  `attn_implementation`), so this part reruns its reference and the layer-swap arm on eager rather than reusing them.

What screen A's per-head table says about choosing controls (Instruct stage, nats per token, layers 0 to 3 only):

| head | drop on identity answers | drop on ordinary answers | specificity |
|---|---|---|---|
| L2H6 | 0.428 | 0.335 | 0.093 |
| L3H5 | 0.342 | 0.257 | 0.085 |
| L0H10 | 0.256 | 0.220 | 0.036 |
| L1H4 | 0.311 | 0.211 | 0.100 |
| next largest (L0H8) | 0.033 | 0.034 | 0.000 |

These four are simply the four most important early heads for any answer; no other early head comes within a factor of
six on ordinary answers. So no early head can serve as a control matched on general importance. Two consequences shape
the design: the coherence gate below decides whether any slot drop can be read at all, and the one fair within-set
contrast is L1H4 against L0H10, which cost ordinary answers almost equally (0.211 and 0.220) but differ about threefold
in specificity.

## Method

`allenai/OLMo-2-0425-1B-Instruct` at 48d788eca847d4d7548f375ad03d3c9312f6139e (host), and for arm R1 the mid-training
end `allenai/OLMo-2-0425-1B` at f83cd641d4f00954eb25c781e0ac360debb3239e (NOT `main`, a different run). bf16, the A770,
`attn_implementation="eager"` on every load, recorded in each run's meta. Part 99's prompts, chat frame, fixed pushes
(`results/part99-pushes.pt`, Amendment 1 of Part 99), neuron list and Part 92 kinds map; greedy decoding with Part 99's
settings; batch 48 in every run, so batch shape is identical across arms.

Ablation: a head is zeroed by setting its 128-wide slice of `o_proj`'s input to zero at every position, the same
operation as screen A. Not a per-head weight swap: OLMo 2 applies `q_norm` and `k_norm` over the whole 2048-wide
projection before splitting heads, so swapping one head's weights also changes every other head's normalisation.

Runs (one `swap99.py` call each; `--ablate-heads` is the one new option):

| run | attention | ablated heads |
|---|---|---|
| R0 | Instruct | none (reference) |
| R0b | Instruct | none, through the ablation hook with an empty set (do-nothing check) |
| R1 | mid-training end in layers 0 to 3 (Part 100's arm, on eager) | none |
| R2 | Instruct | L0H10, L1H4, L2H6, L3H5 |
| R3 | Instruct | L1H4 |
| R4 | Instruct | L0H10 |
| R5 | Instruct | four control heads, one per layer 0 to 3, drawn with seed 219 from the heads with ordinary-answer drop under 0.035 |
| R6 | Instruct | L2H6 (exploratory) |
| R7 | Instruct | L3H5 (exploratory) |

Reading: Part 101's read set (`part101/read101.py filter`), labelled blind by Part 92's Haiku reader in two passes, as
in Parts 100 to 102.

## Controls

- Do-nothing hook: R0b must match R0 token for token on every item, or nothing is read.
- Matched control: not possible on general importance within layers 0 to 3 (table above), so R5 controls only for
  "zeroing four early heads at all"; the L1H4 against L0H10 contrast (R3 against R4) is the matched comparison.
- Breakage: coherence of the slot units' pushed answers and of the unpushed answers is counted in every run (Part
  99's reader fields); a drop that comes with broken answers is hybrid damage, read as inconclusive (Alden's rule from
  Part 99).
- Concept units: the 18 concept neurons in the same runs, as Parts 99 to 102 used them, show whether an ablation
  removes personas in general or the slot route in particular.
- Not used: both push directions and dose ladder, because the pushes are Part 99's fixed ones and the question is
  about the route, not the push. Not used: the answer-polarity split, because there is no yes/no answer.

Gate: R1 must reproduce Part 100's layer-swap drop on eager: slot units showing persona at most 7 of 21 while R0 shows at
least 16 of 21. If it fails, the kernel moved Part 100 and that is reported first; the head arms are still run and
reported, but 219.5 is not read.

## Size

- Smallest effect worth knowing: R2 removing 8 of about 19 slot personas (Part 101's single layer 0 removed 12). With
  21 units this is detectable as a paired change only if it is mostly one-directional; the exact McNemar test on 21
  pairs reaches p < 0.05 at 6 losses against 0 gains.
- Total: 9 runs x 702 generations = 6,318 generations; read set about 258 items per run, 2 passes, about 4,650 Haiku
  labels.
- Pilot: not needed for the generations (Part 99's seven runs took 16 minutes); a 2-item smoke on CPU checks the new
  option and R0b's bit match before the Arc run.

## Optimization

- Bound resource: the A770.
- Measured: Part 99 ran 702 items in about 2.3 minutes per run at batch 48 on the default kernel; eager will be
  measured on R0 and the rest of the chain sized from it.
- Levers: one model load for R0, R0b and R2 to R7 (the hook changes, the weights do not), taken; R1 needs a second
  load, which goes in its own process (one model per process, 11-08 gotcha).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 219.1 | R2 (all four zeroed): slot units showing persona on "Who are you?" at most 8 of 21 | medium (55%) | Journal Claude |
| 219.2 | R2: at least 15 of 21 slot answers coherent and at least 13 of the concept units keep their persona | medium (50%) | Journal Claude |
| 219.3 | R3 (L1H4) removes at least 3 more slot personas than R4 (L0H10) | low (35%) | Journal Claude |
| 219.4 | R5 (control heads): slot units showing persona at least 16 of 21 | high (80%) | Journal Claude |
| 219.5 | R2 removes at least as many slot personas as R1 (the four heads account for the whole early-layer swap) | low (30%) | Journal Claude |

Copied verbatim into PREDICTIONS.md under "## Part 219 (registered ...)".

## Reading rules

Counts are relative to each run's own unpushed answers, as in Part 99 (Amendment 2). Scored by a new `score219.py`
reusing Part 101's read-set scorer; a run is read only if its label pass agreement is at least 95% (Part 101 measured
98.4%). 219.1 is read only if 219.2 holds: a slot drop that comes with fewer than 15 coherent slot answers or fewer
than 13 concept personas is reported as "inconclusive: zeroing these heads breaks answers", whatever the count. 219.3
and 219.5 are counts, reported with the paired unit lists and an exact McNemar p as exploratory support.

The headline "the four heads carry the early attention the slot neurons need" is withdrawn if 219.1 fails, if 219.2
fails, or if R5 removes as many slot personas as R2.

## Limits

- Zeroing is not swapping: it removes a head's output rather than restoring the mid-training version of it, so it
  shows a head is needed, not that post-training's change to it is what matters.
- The four heads were chosen on Instruct's identity answers, and the test uses pushed slot neurons on a different
  question set; that is a strength against circularity, but screen A itself was not registered.
- 21 slot units, one model, one reader.
- No control matched on general importance exists in these layers, so a positive 219.1 with 219.2 holding still leaves
  "these heads matter most for everything, persona included" as a live reading; only 219.3 speaks against it, weakly.
