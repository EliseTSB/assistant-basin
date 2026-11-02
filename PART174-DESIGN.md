# Part 174: How often does a random dense push make Qwen name a maker, and which maker?

Procedure: v1
Registered 2026-11-02 04:22 CT by Claude (Opus 5.5), before any data on these draws. Follows Part 173. Elise gave the machine for the
night ("Alright, im going to bed the pc is all yours", message 1566696992371904533). Reviewed before registration by Weaver,
Alden twice, Muse (prior art) and a Haiku cold read of the build. The commit that adds this file is the registration.

## Question

Part 173 found that all three dense random-valued sets, whose values use no fit, reached a supported lead at |d| = 0.5, and
all three leads were Google: 22, 26 and 36 of 44, none broken. No random-valued set led with another name, while picked neurons
led with Meta and Tencent as well as Google. Three draws cannot say how common that is. This part draws 60 new pushes, reads
each at one dose and both signs, and reports a rate over draws: how often a dense random push leads with a maker, how often
that maker is Google, and how often it is anyone else. It follows [Part 173](results/part173/README.md).

## Prior art and existing data

- Closest published work, from Muse's search for Part 173, cited from Muse's report and not yet read by Claude: no study
  measures a perturbation whose outcome is a named other maker against random directions. Lu et al., arXiv:2601.10387, steer
  away from the Assistant axis under roleplay prompts, with no maker tabulated per dose and no random arm. Malla et al.,
  arXiv:2609.06951: a content-free direction of matched size moves the same behaviours in the same order, with no maker
  answer among its 24 behaviours. Venkatesh and Kurapath, arXiv:2602.06801: a random orthogonal component reaches 95 to 101%
  of a picked vector's effect on style traits.
- Closest for a rate over directions, from Muse's search of 2026-11-02, also cited from Muse's report and not yet read by Claude.
  Korznikov et al., arXiv:2509.22067, draw 1,000 random directions on the unit sphere with a categorical outcome, compliance,
  and give per-direction histograms and threshold counts, with no interval over directions, and their headline pools directions
  with prompts. Hua et al., arXiv:2510.20487, use 24 random vectors and report only an aggregate. Malla et al. use 24 random
  directions pooled into one push, with prompt-bootstrap intervals. Muse found no interval over directions, so that part of
  this census has no precedent to copy, and the form below follows Korznikov's histograms and threshold counts.
- Our own data, all seen before this design and so not evidence for it. Part 173's `u100d` at |d| = 0.5 by sign: draw 173
  Google 8 with 2 broken and Google 22 with 0 broken; draw 1173 Google 26 and 0 broken, OpenAI 2 and 4 broken; draw 2173
  Google 36 and 0 broken, OpenAI 1 and 1 broken. So 3 of 6 signed directions are supported, one sign per draw. Their signed
  cosines with the Google fit are +0.0097, -0.0003 and -0.0020. At d = 0.75 all three broke 10 to 44 of 44. Picked sets at
  |d| = 0.5 in Part 173: 5 of 10 signed directions supported, 3 of them not Google (`n1` -0.5 Tencent 27, `n3` +0.5 Meta 15,
  `n3` -0.5 Meta 35). `none` named no one but the Qwen family, and `zero4` gave 3 Google word hits. Earlier parts found
  Google as what remains when makers are zeroed (Parts 68 to 72), and a maker-name ban is spelled around (Part 159), so a
  ban arm would not test a fallback here and none is included.

## Method

Model: Qwen3.5-4B at snapshot 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a on the Arc. Greedy decoding, up to 80 tokens, thinking
off, as Parts 165 to 173. Pushes go through Part 171's `AddEveryPositionHook` at every position of prompt and reply. A dose d
multiplies a set's whole push.

Prompts: `NEW44B`, Part 173's 44 maker questions. The draws are new, so these prompts have not selected them. The dose was
chosen from three other draws on these prompts, said here and in the README.

Sets, 60 new draws in two families, each at d = +0.5 and d = -0.5, one signed direction each:

- `u100d`, 40 draws, s = 174001 to 174040: Part 173's rule unchanged. With k_l `n100`'s count at layer l, width 9216, at each
  layer with k_l > 0, `default_rng(SeedSequence([s, 30000, l]))` draws g_l, standard normal in all 9216 coordinates, and
  x_l = sqrt(k_l) g_l / ||g_l||.
- `f100d`, 20 draws, s = 174101 to 174120: the same in layers 0 to 20 with k_l = 100/21 at every layer, stream
  `SeedSequence([s, 40000, l])`. Equal energy per layer. `n100` writes to 18 of these 21 layers, with 48 of its 100 units in
  L19 and L20, so `f100d` changes the profile's shape and adds three layers. A difference between the families is not
  attributed to shape alone, on Weaver's review.

In both, one global factor scales a set so ||Σ_l W_l x_l|| equals the written length of `n1`. Nothing is projected out, and no
draw is replaced.

Arms beside them: `none`, `role`, and for Gate 4 replays of `u100d_d2173` at d = 0.5 and `n3` at d = -0.5.

Word scoring, reader, second reader, entity rule, sensitivity reads, `modal`, `supported` (a `modal_count` of at least 10 of 44
and `broken` of at most 8 of 44) and `stable`: Part 173's, unchanged, by importing them from `part173/`, with the git blob hash of
each imported file recorded here: `run173.py` b34c1aece655, `score173.py` 459d20f314e9, `label173.py` f99b8361c595,
`names173.py` 470f7f463841, `prompts173.py` 93075c5dacff, `known173.json` 130f5eea4704. The reader reads each unique clean (question, answer) pair alone, arm and dose removed. The retry pass that
Part 173's error cap asked for is part of the run here: after the first pass, failed pairs get one more pass before the
cap is read.

A signed direction is one draw at one sign. It is `Google-led` if it is supported with Google modal, `other-led` if it is
supported with any other entity modal, known or invented.

A draw falls in one of four classes, set before any data, on Weaver's review: `neither` sign supported, `one sign`, `both same`
with the same modal entity at both signs, and `both different`. These describe two points on a line. Same name at both says the
name did not change between those points, and different names says it did. Neither locates content in a direction.

Sign contrast of a draw: D = (Google claim count at +0.5) - (Google claim count at -0.5).

Two projections of each draw, computed before any answer: cf = cos(w, v), with w the set's summed nominal write and v the saved
Google fit vector; and cr = cos(w, e), with e the unit vector of u(4923) - u(52540), rows of the output embedding at the snapshot.
Token 4923 is " Google" and 52540 is " Alibaba", each one token, and the run asserts both. Their spread over the 60 draws is
written to `stage0.json` before any answer, and if either's standard deviation is under 0.01 its test is not read. 1/sqrt(2560)
is an isotropic reference, not this generator's chance spread: Gaussian coordinates do not give a uniform write, since before
scaling Cov(w) = Σ_l (k_l/9216) W_l W_l^T. Spearman uses ranks, so a stated observed value and interval can be read wherever the
spread and D's variation allow; D with one value on more than half of the draws in the group being read, or a nonfinite correlation, is `not read` and never
counts as zero.
With 60 draws Spearman's chance spread is about 0.13, so the 0.30 bars of 174.6 and 174.9 sit about 2.3 spreads out. A result
past them is a lead to test on new draws, not a mechanism. A result inside them describes the observed monotonic rank
association only: it does not bound the true one at 0.30, and a non-monotonic relation can sit at zero. Each is shown for `u100d`
and `f100d` apart beside the pooled figure, since pooling can cancel opposite relations.

## Controls

- Both directions: every draw at both signs.
- Do-nothing hook: Gate 3.
- Matched random controls: this part is made of them. `u100d` keeps `n100`'s layer profile, and `f100d` removes it.
- Off-target questions: not used, since maker prompts are the whole question.
- Answer-polarity split: not used, as the outcome is a name.
- Broken answers counted: yes, first in the cell order.
- Select on one set, test on another: no selection among draws. Every draw is read at the one dose, so no best-of-grid search
  happens here.

Gates, a failed gate stops all readings:

- Gate 1: every planned answer present, keys unique.
- Gate 2: `none` names Qwen, Tongyi or Alibaba on at least 37 of 44 and `hit` on at most 1 of 44.
- Gate 3: on 8 prompts the plain answers are reproduced, token for token, by a zero-valued push hook, and again after a push arm
  has run and its hooks are removed.
- Gate 4, replay: `u100d_d2173` at d = 0.5 and `n3` at d = -0.5 match Part 173's stored `token_ids` on at least 43 of 44 rows each,
  with sha256 of the pushes, hook source, tokenizer and snapshot beside the run.
- Gate 5: `role` hits on at least 38 of 44. Gates 2 and 5 run right after the `none` and `role` arms, before the draws.
- Gate 6: at one fixed context, through the real hook, ⟨delta(d), w1⟩ / ‖w1‖² is within 5% of d for every draw at both signs,
  with w1 the set's own delivery at d = 1 at that context.
- Gate 7: the prompts are `NEW44B`, byte for byte.
- Gate 8: for every draw ||w|| is positive and finite, |cos(w, v)| is at most 0.079, and ||w|| is within 1% of `n1`'s. The 0.079
  is an eligibility screen carried from Part 173, not a four-sigma guarantee for these generators. Failing stops the reading; no
  draw is replaced. Every reading here is conditional on all 60 draws passing it, and is said to be.
- Gate 9, readers: at least 18 of 20 known examples on `relation` and at least 18 of 20 on the resolved `primary`, for each
  reader, on `part173/known173.json`.

## Size

- Smallest effect worth knowing: whether other-led directions run near the picked sets' 3 in 10 or below about 1 in 14. Draws are
  the independent units, so `u100d` gives 40 and not 80. With no draw other-led at either sign, an exact one-sided 95% bound on
  "either sign other-led" is 1 - 0.05^(1/40) = 7.2%, which also bounds the signed rate from above. At one sign, 40 draws give about
  ±14 points on a rate near 30%, and pooling both signs need not do better if the signs agree. 20 `f100d` draws can tell half that
  family's directions from a tenth and not finer.
- 44 prompts × 60 draws × 2 signs = 5,280 answers, plus `none`, `role` and two replays, 176, and the Gate 3 and 6 checks, about
  5,500 generations and about 38 minutes on the A770. At most 5,456 reader calls before removing identical pairs, about 3
  million tokens on the shared OpenCode Go lane, with the window checked first; about 10% of that on Haiku.
- Total: 5,500 generations, about 38 minutes.
- Pilot: the first 3 batches. Stop rule: under 1.0 answers per second.

## Optimization

- Bound resource: GPU.
- Measured: 2.4 answers/s on Part 173, 6,336 answers in 43:43 with the model load, batch 44.
- Levers: batch 44 taken; one model load taken; one dose in place of six taken, since the draws are the sample here; no refit
  taken; one reader pass over identical pairs, taken; six overlapping reader calls with checkpoints, taken from Part 173's
  measured 0.7 pairs/s; copy engine off, the job default.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 174.1 | At least 20 of the 80 `u100d` signed directions are supported | medium | Claude |
| 174.2 | Google is modal on at least 80% of supported `u100d` directions; not read below 10 supported | medium | Claude |
| 174.3 | At most 4 of the 80 `u100d` signed directions are other-led | medium | Claude |
| 174.4 | Of the `u100d` draws with at least one supported sign, at most 25% are `both same` or `both different`; not read below 8 such draws | low | Claude |
| 174.5 | `f100d`'s supported share is at least half of `u100d`'s; literal result always reported, read as retention only if `u100d` has at least 10 supported, else `uninformative` | low | Claude |
| 174.6 | Across the 60 draws, the observed |Spearman(cf, D)| is under 0.30 | medium | Claude |
| 174.7 | The median `broken` of the 80 `u100d` signed directions is at most 8 of 44 | medium | Claude |
| 174.8 | At least 60% of all supported directions are `stable`; not read below 5 supported | medium | Claude |
| 174.9 | Across the 60 draws, the observed |Spearman(cr, D)| is under 0.30 | low | Claude |
| 174.10 | Google is modal on at least 80% of supported `f100d` directions; not read below 5 supported | medium | Claude |
| 174.W1 | At least 30 of the 80 `u100d` signed directions are supported. Falsifier: fewer than 20, which would say the three draws were a lucky sample rather than a rate. 20 to 29 is inconclusive | medium | Weaver |
| 174.W2 | Other-led directions are present but few, and the majority name invented entities. Falsifier: at least half of the other-led directions name a known non-Qwen entity. Not testable with no other-led direction. Held if a strict majority of the other-led directions are invented; how few they are is reported, not scored. Invented means outside the entity table, not shown to be fictional | medium | Weaver |
| 174.W3 | `f100d`'s supported share is within 20 points of `u100d`'s. Falsifier: `f100d` is below half of `u100d`, which would make the layer profile or the spread a real gate, without saying which. The falsifier takes precedence where both are true. Any other outcome is inconclusive | medium | Weaver |

## Reading rules

Verdicts are point estimates against the bars. Every signed direction is reported alone, with broken, the full name distribution,
runner-up and raw word hits. The full distribution of Google claim counts over all 120 directions is reported as a histogram,
and 174.1's rate beside the same rate at floors of 5 and 15 of 44, on Weaver's review: if mass sits between 5 and 15, the
supported rate is a fact about where the floor sits. The `relation` distribution and the mention counts are reported per
family, so a Google-led answer that claims Google is told from one that only mentions it. This part licenses a statement about this generator of draws, at this dose, on these prompts, and
no further: a rate over new draws conditional on d = 0.5, these 44 fixed prompts and Gate 8's screen. Part 173's three draws are
dose discovery, not census observations, and are not pooled in. Each family's total supported, Google-led and other-led counts
are always reported together. Intervals: for each family the rate at each sign alone, k of n draws, with a Wilson interval; for
the pooled signed rate a cluster bootstrap over draws, seed 174, 10,000 replicates, keeping a draw's two signs together, with
prompts held fixed, and resampling within each family in any combined figure. A zero or all-success outcome is not bootstrapped:
it gets the exact one-sided 95% bound on the either-sign rate, labelled as that. An empty denominator is `not read`. A bootstrap
over prompts is reported beside these and labelled question-level, since it says nothing about new draws. Any pooled
draw-by-question rate is labelled secondary.
Intervals on a Spearman correlation are percentile bootstraps over draws, stratified by family when pooled.

Reader gate as in Part 173: kappa of at least 0.6 on `relation`, and at least 80% agreement on the resolved entity on at
least 30 replacement-stratum pairs. If it fails or is unread, or the reader error cap is exceeded, 174.1 to 174.6, 174.8 to
174.10 and Weaver's three are unread, and raw word hits, including their correlations, stay reported as a labelled descriptive
fallback and are not those endpoints.

Fallback reading, stated with what it needs. It is read only if 174.2 and 174.10 are both read and both hold: Google leads most
usable directions in each family, not just one. Passing every other bar does not support it, since `f100d` could reach its share
with none of its directions Google-led. If other names are common in either family, the reading fails. A pass does not establish
a fallback's mechanism, and this part does not test one.

Comparison with picked sets is descriptive: 10 signed directions from Part 173, on the same prompts, chosen by a different rule
and read with a different reader pass.

## Limits

One model, one maker family, greedy decoding, one dose chosen from three earlier draws, 44 prompts. Dense random directions are
one generator among many; a rate here says nothing of sparse pushes, which broke answers heavily in Part 173. Equal written
length is not equal delivered change. A name is one reader's extraction under a fixed rule. A modal count of 10 of 44 is a
plurality that can leave most answers elsewhere. `u100d` keeps a layer profile that came from the Google fit's ranking.
