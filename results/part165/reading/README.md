# Part 165 reading: what gives a neuron push its extra first-step effect?

Registered 2026-11-02 in [PART165-READING-DESIGN.md](../../../PART165-READING-DESIGN.md)
before any data from this analysis existed, and run on the Arc the same day. It reads
Part 165's [effect run](../effect-tables.md); it does not revise Part 165's registered
verdicts, and it generates nothing. Gates passed on all five checks; the job reports
119 s.

## What was asked

Part 165 measured, at equal realized write norm, a first-step KL of 0.734 for `n+`
against 0.352 for `full` and 0.353 for `iso`. Equal write norm was therefore not equal
effective dose. This reading asks which property of the write direction carries the
difference: concentration in residual coordinates, alignment with the natural state at
the site, or alignment with the readout. Five bars and five predictions were registered
before the run.

## The five bars

| bar | registered rule | observed | verdict |
| --- | --- | --- | --- |
| R1 | pooled Spearman(PR(w), kl) within `n+` <= -0.30, and <= -0.15 in >= 5/8 layers | [+0.186, 95% CI +0.011 to +0.351](reading.json); 1/8 layers at <= -0.15 (+0.13, +0.42, -0.32, +0.07, +0.46, +0.30, +0.03, +0.14) | **fail** |
| R2 | per-layer Spearman(arm-mean PR(w), arm-mean kl) <= -0.70 in >= 6/8 layers | 0/8 layers at <= -0.70 (L19 +0.95, L20 -0.61, L21 -0.53, L22 -0.40, L23 +0.17, L24 +0.29, L25 -0.38, L26 -0.63) | **fail** |
| R3 | median \|proj\| of `n+` >= 1.5x the isotropic null in >= 6/8 layers | ratios 0.82, 0.80, 0.82, 1.19, 0.91, 0.82, 1.07, 1.12; none above 1.19 | **fail** |
| R4 | \|Spearman(cos(w, (1+gamma)*u_Alibaba), kl)\| < 0.30 within `n+` | [-0.034, 95% CI -0.201 to +0.137](reading.json), 96 clusters | **pass** |
| R5 | OLS of log kl on log PR(w) with layer fixed effects explains >= 50% of the `n+`-`full` gap | slope 2.66; raw gap 0.229; predicted -0.009; explained -4% | **fail** |

## Predictions, registered before the run

P1 (R1 holds) **failed**. P2 (R2 holds, partly a `k` effect) **failed**. P3 (R3 holds)
**failed**. P4 (R4 holds) **held**. P5 (R5 holds) **failed**. The design's confidence
labels were medium, high, medium-high, medium and low-medium, in that order.

## What failing means, and what it does not

- **R1 fails, and in the opposite direction.** Within `n+`, more concentrated writes
  had slightly *larger* first-step KL, not smaller: the pooled rho is
  +0.186 with an interval that excludes zero. Only one of eight layers
  reaches -0.15. On this population, at this site, concentration does not carry the
  effect and the data lean the other way. It does not show that concentration is
  irrelevant in other layers, other models, or during generation.
- **R2 fails descriptively.** The across-arm ladder is not monotone in PR(w): one layer
  reaches +0.95 and the rest scatter from -0.63 to +0.29, with none at -0.70. R2 was
  registered as descriptive and is collinear with `log k` by construction, so its
  failure does not isolate a cause either.
- **R3 fails.** `n+`'s writes are not more aligned with natural states than an isotropic
  draw: the per-layer ratios run 0.80 to 1.19 against a bar of 1.5. The natural-state
  alignment account is not supported. The null is one specific isotropic construction,
  and the measure is alignment rather than dose (the design's correction), so this is a
  failure of the registered comparison, not a proof that geometry is irrelevant.
- **R4 passes as a bound.** Readout alignment does not predict the first-step effect
  within `n+`: rho -0.034, interval
  -0.201 to +0.137, over 96 clusters. The
  bar was registered as a bound, not an equivalence claim, so this says the association is
  at most weak on this measurement; it does not say the readout path is absent, and a rho
  near the bound would still be compatible with a small contribution.
- **R5 fails.** The PR(w) adjustment explains -4% of the `n+`-`full`
  mean log-KL gap, against a bar of 50%. The gap is carried by something else. What that
  is, this reading does not say.

## Descriptive, not bars

Within `n+`, the other registered measures against KL:

| measure | rho with kl (n+) | 95% CI | rows / clusters |
| --- | ---: | ---: | --- |
| PR(w) | +0.186 | +0.011 to +0.351 | 768 / 96 |
| cos(w, mu) | -0.170 | -0.330 to -0.002 | 768 / 96 |
| PC16 | +0.054 | -0.118 to +0.218 | 768 / 96 |
| \|proj\| | +0.026 | -0.153 to +0.197 | 768 / 96 |
| readout | -0.034 | -0.201 to +0.137 | 768 / 96 |

`cos(w, mu)` is negative with an interval that excludes zero; it is reported here
because it is in the table, not because a bar was registered for it. The design records
that a projection of the write onto the state is an alignment measure, not a dose.

## Limits

- One model: Qwen3.5-4B at snapshot `851bf6e8`, bf16, on the A770.
- Layers 19-26 only; 12 slots per layer; 96 `(layer, slot)` clusters; 8 maker questions;
  16 documents, each the first 128 tokens of its row.
- The effect is Part 165's first-step KL at one site, `down_proj`'s input, at 0.8x the
  layer's median residual norm.
- 512 isotropic null directions per layer and a 2,000-token vocabulary sample bound the
  null and readout estimates, not the effect.
- Intervals resample `(layer, slot)` clusters. They cover these slots, questions and
  documents; they do not cover new slots, new models or new prompts.
- R5's slope is a pooled OLS association with layer fixed effects over 6,912 rows (768 of
  them `n+`). It is descriptive, not causal.
- No generation: the only sampling is 512 null directions per layer and the eight greedy
  first tokens.
- The bars and their thresholds were registered before the run. Gate thresholds for the
  capture and site checks were amended before the run; the capture check's device line
  was fixed after a launch stopped in 17 s with no measurement taken.

## What this does not claim

Nothing about what does carry the gap. Three accounts failed their registered bars
(concentration, natural-state alignment, the PR decomposition), and the fourth, readout
alignment, passed only as a bound: the effect is not explained by it on this measurement,
which is weaker than showing it absent. Part 165's registered verdicts are unchanged; any
revision to how Part 165 itself is read belongs in a dated note there, not here.

Files: [reading.json](reading.json) (the full record, 2.3 MB) and
[reading-tables.md](reading-tables.md); the registered design is
[PART165-READING-DESIGN.md](../../../PART165-READING-DESIGN.md). Numbers above are
quoted from `reading.json`; the tables file is the same content in Markdown.
