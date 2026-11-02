# Part 175: Does a flat random push name Google once it stops breaking the model?

Procedure: v1
Registered 2026-11-02 07:32 CT by Claude (Opus 5.5), before any answer on the Stage A draws and before any answer on the census draws at these doses. Follows Part 174. Elise gave the machine for the
night ("Alright, im going to bed the pc is all yours", message 1566696992371904533). Reviewed before registration by Weaver,
Alden, Muse (prior art) and a Haiku cold read of the build. The commit that adds this file is the registration.

## Question

In Part 174, 28 of 80 layer-matched random directions led with Google at |d| = 0.5 and none led with another name. The 20 flat
draws, at the same written length, broke a median of 44 of 44 answers, so they said nothing about what a flat push names. This
part finds a dose where flat pushes stop breaking, reads the same 20 draws there, and asks how many lead with a maker and whether
that maker is Google. It follows [Part 174](results/part174/README.md).

## Prior art and existing data

- Closest published work, from Muse's search of 2026-11-02, cited from Muse's report and not yet read by Claude. No dose ladder for
  random directions has a named entity as its outcome. Korznikov et al., arXiv:2509.22067, ladder random directions one layer at a
  time with compliance as the outcome and no coherence curve. Herbster et al., arXiv:2604.08169, score coherence by layer for a
  picked direction and find the fragile end late, so no general early-fragile rule can be imported. Dang and Ngo, arXiv:2601.19375,
  find uniform steering across all layers brittle and put the cause in writing to layers that do not separate the feature, not in
  early layers being fragile. A LessWrong post, "Power Steering", reports early incoherence at a fixed absolute scale and puts it
  down to relative dose: output norms vary about 50-fold across layers there. Amplification through depth is, on Muse's search,
  unmeasured here. So this part logs relative dose and does not assume a cause for the breaking.
- Our own data, all seen before this design. Part 173's three `u100d` draws by dose at + sign: d = 0.25 Google 1, 9 and 0 with at
  most 1 broken; d = 0.5 Google 8, 26 and 36 with at most 2 broken; d = 0.75 broken 35, 10 and 44. So a layer-matched draw went
  from weak to supported to broken inside a factor of about 2 in dose, which is why this part reads three doses and not one.
- Part 174's `f100d` at |d| = 0.5: broken 32 to 44 of 44, 28 clean answers in 1,760, none naming Google. Part 174's `u100d`: the
  rank correlation of Google claims with `broken` over 73 labelled directions is -0.26, and the share of clean answers naming
  Google stays near a quarter from 0 to 14 broken. So nothing there says naming sits at the edge of breaking, and no prediction
  here depends on it.

## Method

Model, decoding, hook and prompts as Part 174: Qwen3.5-4B at snapshot 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a on the Arc, greedy,
up to 80 tokens, thinking off, `AddEveryPositionHook` at every position of prompt and reply, `NEW44B`. Word scoring, reader,
second reader, entity rule, `modal`, `supported` (a `modal_count` of at least 10 of 44 and `broken` of at most 8 of 44), `stable`,
`Google-led`, `other-led` and the four draw classes: Part 174's, unchanged, by import, with these git blobs: `run174.py` 093d7e500ba0, `score174.py` fce5e1aac0ec, `label174.py` 779fc464add7, `prompts174.py` 853a22d2c009, `run173.py` b34c1aece655, `score173.py` 459d20f314e9, `label173.py` f99b8361c595, `names173.py` 470f7f463841, `prompts173.py` 93075c5dacff, `known173.json` 130f5eea4704.

Stage A, dose discovery. Four fresh flat draws, s = 175901 to 175904, built by Part 174's `f100d` rule, stream
`SeedSequence([s, 40000, l])`. Each at both signs and at |d| = 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40 and 0.45, on all 44 prompts of
`NEW44B`, so D* is measured under the census's own conditions (Weaver's review: 12 prompts would have carried 3.7 times the
per-answer variance of the bar it must meet). Only `broken` decides anything here. D* is the largest of those doses at which the
median `broken` over the 8 signed directions is at most 8 of 44, with every lower dose meeting that too. The two signs' breakage
curves are reported apart, since the pooled median can average two different curves. The rule runs in code and writes
`stage_a.json` before Stage B starts. If no dose meets it, the run stops there and the part reports Stage A alone.

Dose, three ways, descriptive and with no prediction, on Muse's review. The ladder's d is a multiple of a set's whole push, written
at every position. For each draw and layer the run records the written norm ||W_l x_l||, the sum over written layers, and that norm
over the layer's mean residual-stream norm on Gate 3's 8 plain contexts, for Stage A's draws, the census draws and `u100d_d174001`.
Each layer has its own independent draw, so writes do not compound as one direction across depth.

Stage B, the census. Part 174's 20 `f100d` draws, s = 174101 to 174120, each at both signs and at |d| = D*, 0.75 D* and 0.5 D*.
All 20 are read at every dose, and these draws were seen before only as broken at 0.5, so no draw is selected. D* is the primary
dose. The two lower doses are there because a dose window can be narrow: a name supported only just below breaking is a different
result from one that survives well below it, and the README says which this was.

Beside them: `none`, `role`, and for Gate 4 replays of `u100d_d174001` and `f100d_d174101` at d = +0.5.

Unobserved is null, on Weaver's reading of Part 174, in three states set on Alden's review. All 40 directions stay in the
denominator. `no clean answers`: zero clean answers, so known not supported under the damage rule. `reader missing`: clean answers
exist and none has a label, so support is unknown. `partly labelled`: some clean answers have no label, so support is bounded by
every way they could fall. The Google claim count is null in all three, and the row records the state and its support status. A
verdict is read on what is known; if an unknown support could change a verdict, that verdict is `not read`, with both bounds
given. Every reader skips nulls, and nothing coerces them.

Inert-dose guard, on Weaver's review. A push too weak to move the model must not read as a flat push naming no one. Over the 1,760
answers at D*, the share whose first 40 tokens differ from `none`'s answer to the same prompt is computed from stored rows. Under
15%, the census is `inert`: 175.2 to 175.4, 175.6 and 175.W2 are `not read`, and nothing is read as zero.

A draw's workable dose at one sign, for 175.W3: the largest ladder dose with `broken` at most 8 of 44 and every lower dose meeting
that too, as a step on the ladder, with step 0 below 0.10. A draw's sign gap is the difference between its two signs' steps.

## Controls

- Both directions: every draw at both signs, every dose.
- Do-nothing hook: Gate 3.
- Matched random controls: this part is made of them. No fit enters the sets.
- Dose ladder: Stage A, then three census doses.
- Off-target questions: not used, since maker prompts are the whole question.
- Answer-polarity split: not used, as the outcome is a name.
- Broken answers counted: yes, first in the cell order.
- Select on one set, test on another: the dose is chosen on four draws the census does not use.

Gates, a failed gate stops all readings:

- Gate 1: every planned answer present, keys unique.
- Gate 2: `none` names Qwen, Tongyi or Alibaba on at least 37 of 44 and `hit` on at most 1 of 44.
- Gate 3: on 8 prompts the plain answers are reproduced, token for token, by a zero-valued push hook, and again after a push arm
  has run and its hooks are removed.
- Gate 4, replay: `u100d_d174001` and `f100d_d174101` at d = +0.5 match Part 174's stored `token_ids` on at least 43 of 44 rows each,
  at Part 174's batch shape.
- Gate 5: `role` hits on at least 38 of 44.
- Gate 6: at one fixed context, through the real hook, delivery is within 5% of d for every census draw at each census dose, as
  in Part 174.
- Gate 7: the prompts are `NEW44B`, byte for byte.
- Gate 8: the rebuilt 60 draws match `results/part174/stage0.json` draw by draw on global scale, written norm, cf and cr, within
  1e-6. A raw file hash cannot do this, since PyTorch's archive holds a per-process serialization id. The four Stage A draws pass
  Part 174's length and |cf| screens.
- Gate 9, readers: at least 18 of 20 known examples on `relation` and on the resolved `primary`, for each reader, on
  `part173/known173.json`.

## Size

- Smallest effect worth knowing: whether flat directions at a workable dose lead at about `u100d`'s 35% or below half of that.
  Draws are the independent units, so 20 and not 40. With no draw supported at either sign the exact one-sided 95% bound is
  1 - 0.05^(1/20) = 13.9%, and that also bounds the signed rate from above.
- Stage A: 4 draws × 2 signs × 8 doses × 44 prompts = 2,816 answers. Stage B: 20 × 2 × 3 × 44 = 5,280. `none`, `role` and two
  replays, 176. Gates 3 and 6 as Part 174.
- Total: 8,272 generations, about 48 minutes on the A770.
- Pilot: Stage A is the pilot. Stop rule: under 1.0 answers per second.

## Optimization

- Bound resource: GPU.
- Measured: 2.9 answers/s on Part 174, 5,456 rows in 30:56 with the model load, batch 44.
- Levers: batch 44 taken; one model load taken; `broken` alone decides Stage A
  so no reader runs on it, taken; six overlapping reader calls with checkpoints, taken from Part 174; copy engine off, the job
  default. Not taken: 12 prompts for dose discovery, which would save 12 minutes and choose D* under more noise than the bar it
  must meet; a finer dose ladder, since the census covers a factor of 2 below D*.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 175.1 | Stage A finds a D* at or above 0.10 | high | Claude |
| 175.2 | At D*, at least 8 of the 40 `f100d` signed directions are supported | low | Claude |
| 175.3 | Google is modal on at least 80% of supported directions at D*; not read below 5 supported | medium | Claude |
| 175.4 | At D*, at most 2 of the 40 signed directions are other-led | medium | Claude |
| 175.5 | At D*, the median `broken` of the 40 signed directions is at most 8 of 44 | medium | Claude |
| 175.6 | More directions are supported at D* than at 0.5 D*; not read if neither dose has at least 3 | medium | Claude |
| 175.W1 | D* lands between 0.20 and 0.35. Falsifier: D* at 0.10 or 0.45, or no D*. 0.15 and 0.40 are inconclusive | medium | Weaver |
| 175.W2 | At most 5 of the 40 signed directions are supported at D*. Falsifier: 8 or more. 6 or 7 is inconclusive | medium | Weaver |
| 175.W3 | At least 3 of the 4 Stage A draws have a sign gap of at least 2 ladder steps, and then D* is carried as a band. Falsifier: 2 or fewer | medium | Weaver |

## Reading rules

Verdicts are point estimates against the bars, read at D*. The two lower doses are reported in full and are descriptive; the best
of the three is reported only with the label `selected`. Every signed direction is reported alone, with `broken`, the full name
distribution and raw word hits. Each dose's supported, Google-led and other-led counts are always reported together, with the
count of directions in each of the three null states. The histogram of Google claim counts excludes nulls and says how many. D* comes from four
draws, so 175.5 is a check for drift across draws; if it fails, 175.2 to 175.4 are still reported and carry that condition beside
them. If 175.W3 holds, D* is also given as the band of per-sign workable doses.
Intervals: at each sign a Wilson interval on k of 20 draws; pooled, a cluster bootstrap over draws, seed 175, 10,000 replicates,
keeping a draw's two signs together; a zero or all-success outcome gets the exact one-sided 95% bound on the either-sign rate.

The comparison with Part 174's `u100d` at 0.5 is descriptive and shown beside both families' `broken` distributions: similar breakage
at most, unequal written length, and no layer-profile effect identified. This part licenses a
statement about these 20 draws at these doses on these prompts. Reader gate and error cap as in Part 174; if either fails,
175.2 to 175.4, 175.6 and 175.W2 are unread and raw word hits stay as a labelled descriptive fallback.

Fallback reading, read only if 174.2, 175.2 and 175.3 all held: Google leads usable directions in both families, each at its
own workable dose. A pass does not establish a fallback's mechanism, and this part does not test one.

Not used: the correlations with cf and cr. Twenty draws give a chance spread near 0.23, so Part 174's 0.30 bars would decide
nothing.

## Limits

One model, one maker family, greedy decoding, 44 prompts, 20 draws. These 20 draws are Part 174's, so this is a follow-up on a
finite cohort and not a replication on new draws. D* comes from four other draws, on the same prompts. Equal breakage is not equal
delivered change, and this part does not measure delivery past the written layer, so it cannot say why flat pushes broke at 0.5.
A difference from `u100d` is not attributed to layer profile alone, since dose and written length differ as well. Lower doses are
weaker pushes, so a low rate here may mean a flat push names no one or only that it is too small to.

## Amendment 1, 2026-11-02 10:02 CT

Made before any label count, score or verdict was read; one label row was printed while diagnosing. The reader's retry pass ended
at 10:01 with 45 of 3,782 pairs failed, 1.19%, over the 0.5% cap. All 45 errors are `Temporary failure in name resolution`: this
machine's DNS failed, and no model saw those pairs. A retry pass in which DNS was failing did not test the reader, so one more
retry pass runs now that DNS answers. Any pair that still fails for another reason counts against the cap as the design says. The
one-retry wording above stands for every failure a model could have caused, and readers.log shows both passes.

## Amendment 2, 2026-11-02 10:06 CT

Also before any label count, score or verdict was read. DNS failed again during Amendment 1's pass: 18 pairs still carry a name
resolution error, with six reader threads running, though a dozen lookups in a row then succeeded. Amendment 1's pass also retried
every other failed pair, so those had a third attempt the design did not give them; 13 pairs remain failed for reasons other than
DNS and they count against the cap. Those 18 alone are now retried one at a time by `part175/retry_dns.py`, until a model answers,
and no other pair gets another attempt. The cap is then read on what remains.
