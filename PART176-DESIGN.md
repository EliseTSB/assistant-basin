# Part 176: Does the flat band at 0.225 hold on new draws, and what ends it?

Procedure: v1
Registered 2026-11-02 12:44 CT by Claude (Opus 5.5), before any answer on these draws. Follows Part 175. Elise gave the research direction on
2026-10-27 and the repo is Claude's to run. Reviewed before registration by Weaver, Alden, Muse (prior art) and a Haiku cold read of the
build. The commit that adds this file is the registration.

## Question

On Part 175's 20 flat draws, 27 of 40 signed directions were supported at |d| = 0.225, all Google-led, and that was the best of
three doses on the same draws. This part asks three things on 20 new draws. Does the lead hold at 0.225, a dose this part does not
choose? Is 0.225 a window, a plateau or a slope, read at its neighbours? And what ends it: does Google fall away among the answers
that still name anyone, or does the pool of naming answers shrink under it, and is a direction's naming tied to where its own
answers fall apart?

## Prior art and existing data

- Closest published work, from Muse's search of 2026-11-02 for this part, cited from Muse's report and not yet read by Claude. No paper reports this
  sequence of stages on one readout, a per-direction collapse dose across random directions, or a split into a rate among surviving answers and the size of the
  pool, and role swaps under steering are not reported as an outcome. Nearest: Bas and Novak, arXiv:2511.18284, find trait expression an inverted U in coefficient
  while coherence falls, which cannot tell an effect receding from a pool collapsing. Malla et al., arXiv:2609.06951, find a steer relaxes a model toward behaviours
  it already favours, and norm-matched random directions reproduce the ranking; their coherence floor makes any rate among passing answers a survivor rate.
  Korznikov et al., arXiv:2509.22067, give per-direction jailbreak histograms over 1,000 random directions and no collapse dose. Lu et al., arXiv:2601.10387, steer
  along a picked axis toward other entities. Dang and Ngo, arXiv:2601.19375, put uniform steering's collapse down to writing into layers that do not separate the
  feature. Muse's earlier warning stands: a name in a narrow band just below breaking is the shape a side effect of breaking would have.
- Our own data, all seen before this design, computed by [part176/prior176.py](part176/prior176.py) into
  [results/part176/prior176.json](results/part176/prior176.json). All of it is exploratory.
- Across Parts 173 to 175, 60 direction-at-dose rows were supported, all Google-led: 28 in Part 174, 29 in Part 175 at its three
  doses, 3 in Part 173's random-valued sets. Other-led was 0 in the 240 signed-direction rows of Parts 174 and 175, which hold 57 of
  the 60. The contrast with picked neurons, which led with Tencent and Meta in Part 173, is across parts and stays a hypothesis.
- Part 175's states, pooled over 40 directions, of 1,760 answers. A `fragment` is a clean no-claim answer of at most 12 tokens with no own
  name in its words; the rule is new here and was written after looking at these rows.

| \|d\| | Qwen named | Google named | other named | claim, no single name | no claim, own name in words | no claim, fragment | no claim, longer | broken |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.15 | 1,523 | 93 | 1 | 47 | 1 | 1 | 83 | 11 |
| 0.225 | 840 | 551 | 26 | 49 | 19 | 16 | 170 | 89 |
| 0.30 | 38 | 50 | 27 | 1 | 22 | 460 | 107 | 1,055 |

  Three readings of that table, none tested. Qwen falls before anything fails: `broken` plus `fragment` has a median of 0, 2.5 and 42.5 of 44
  across directions, and reaches 22 of 44 on 0, 0 and 37 of 40. So the collapse is one steep step shared by most directions. Second,
  `broken` undercounts it, and `no claim` changes meaning with dose. At 0.225 most no-claim answers are coherent refusals such as "As an AI, I don't have
  a physical form or a team behind me". At 0.30 most are fragments, many of them the question said back: "Which organization trained you?".
  Third, Google's share of named answers does not fall with dose: 5.8%, 38.9% and 43.5%. Part 175's design said nothing there suggested naming sits
  at the edge of breaking; that was read from one dose, and its own ladder overturned it. Weaver's review found the first version of this
  table too coarse, and asked for the split.
- Part 175's Stage A, four other draws at 8 doses, raw `google` word hits and no reader: directions with at least 10 hits and at most
  8 broken were 0 of 8 at 0.15, 2 of 8 at 0.20, 5 of 8 at 0.25, 1 of 8 at 0.30.
- Whether a direction's naming follows its own collapse, on Part 175's rows: Spearman of lost at 0.30 against Google at 0.15 is
  +0.35 (interval -0.05 to +0.64), against Google at 0.225 is -0.03 (-0.32 to +0.23), and with `broken` alone +0.16 and +0.14. Lost at
  0.30 is near its ceiling on most directions, so those rows measure this badly; that is why this part puts rungs across the step.

## Method

Model, decoding, hook and prompts as Parts 174 and 175: Qwen3.5-4B at snapshot 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a on the Arc,
greedy, up to 80 tokens, thinking off, `AddEveryPositionHook` at every position of prompt and reply, `NEW44B`. Word scoring, reader,
second reader, entity rule, `modal`, `supported` (a `modal_count` of at least 10 of 44 and `broken` of at most 8 of 44), `stable`,
`Google-led`, `other-led`, the three null states and the four draw classes: Part 175's, unchanged, by import.

Draws: 20 new flat draws, s = 176101 to 176120, built by Part 174's `f100d` rule, stream `SeedSequence([s, 40000, l])`, scaled to
`n1`'s written length. No draw is replaced.

Ladder: every draw at both signs at |d| = 0.20, 0.225, 0.25, 0.275 and 0.30, on all 44 prompts. No dose is chosen in this part. 0.225 is fixed by
Part 175 before any answer here. Weaver's point from Part 175 is why: 6.5 against 29.5 broken at the same dose says a ladder's endpoint
cannot be located by four draws, because the variance is in the draws. So every rung is read on every draw.

Beside them: `none`, `role`, and for Gate 4 replays of `f100d_d174104` at +0.225 and `f100d_d174115` at -0.225, Part 175's arms with
the most Google claims.

Terms, each computed per signed direction and dose from first-reader labels:

- `no claim`: clean answers with reader relation `none`.
- `named`: Part 173's `named` count, clean answers with a resolved primary.
- `Google share`: Google primaries over `named`, as `full_name_distribution["google"]` over `named`. Null when `named` is under 11.
- `fragment`: a clean `no claim` answer of at most 12 tokens with no `qwen`, `tongyi` or `alibaba` in its words. A rule on token count, with
  sensitivity at 8 and 20 tokens, since a short answer can be coherent: "I don't know who built this model."
- `reverted`: a clean `no claim` answer with one of those words. Reported per rung and in no other measure.
- `failed`: `broken` plus `fragment`. `lost`, the wider sensitivity measure, is `broken` plus all of `no claim`.
- `gap`: L minus the top supported rung, for a direction supported at some rung. Right-censored L uses 0.30, a lower bound, and the count of those
  is given; a negative gap is reported as it is.
- `collapse dose` L: the dose where `failed` first reaches 22 of 44, by straight-line interpolation between the two rungs either side of
  the crossing. If `failed` is already at least 22 at 0.20, L is left-censored at 0.20. If it never reaches 22, L is right-censored
  above 0.30. Censored values take the lowest or highest rank, tied, which is a coarsened rank and says nothing about order inside either group.
  L is an operational first-crossing proxy, not a measured dose: the crossing bracket and the count of directions where `failed` falls back
  under 22 at a later rung are reported, and an ordinal version, the bracket's rung with ties, runs beside it. The distribution of L is reported, with each
  direction's transition width, the dose from `failed` at 11 to `failed` at 33 where both fall on the ladder.
- Every rate is given twice, per dose and per direction, on Muse's review: Google over all 44 and Google over `named`, with `named` over 44 beside them.
- Two checks not coded on makers, over all answers, broken included: median token count, and the share of answers whose words mostly repeat the question
  (Jaccard over 0.7 on lowercased words). Neither filters anything.

Dose, three ways, logged as in Part 175 with no prediction: the written norm per layer, the sum over written layers, and that norm over the layer's mean
residual-stream norm on Gate 3's 8 plain contexts.

Unobserved is null, as Part 175: all 40 directions stay in the denominator of every count, in the three states. An unlabelled clean answer
of more than 12 tokens, or with an own name in its words, cannot be a `fragment`, so it is not `failed` whatever a reader would have said. Only an
unlabelled short answer is undetermined. If one sits at a rung that decides the crossing, that direction's L is an interval, the direction is not
eligible for 176.7 or 176.W3, and it is counted with its reason. Filling every undetermined answer one way or the other is a sensitivity example and not
a bound, since a rank statistic is not monotone in the fills. Pooled counts over directions with unlabelled answers are given as lower and upper bounds, and a pooled
condition is read on the bound that would stop it. A rung whose support is unknown is not a supported rung; if it could be a direction's top supported rung, that direction is not
eligible for 176.W3 and is counted.

Audit of the new states, frozen before the run. After scoring, a sample with seed 176 goes to the second reader, blind to rung: 100 `fragment` answers
spread evenly over the rungs that have them, 50 longer `no claim` answers, and 25 claims with no single name, each sorted into echo of the question, role swap,
coherent refusal, other maker, stopped mid-sentence, or other. It is reported as a table, and the role-swap rate is the first measure of that outcome. If more
than a third of the fragments are coherent refusals, the write-up calls L a short-answer dose and not a failure dose. This reader's agreement on replacement
names says nothing about these labels, and `failed` stays a proxy: `broken` plus short no-claim answers.

Transitions. Greedy answers are fixed by prompt, direction and dose, so each prompt in each direction has one state at each rung. Rung-to-rung transition
tables are reported, pooled and by draw: where Google answers go at the next rung, and whether answers that end `failed` passed through Google. Descriptive.

Reader failures. Part 175 needed three amendments for these. A name-resolution failure happens before any connection, so it is evidence that no
request was sent, and it is not an attempt: `part176/retry_net.py` retries those one at a time, with backoff up to 30 s, for at most 2 hours in all, and
keeps an attempt id for each. A reset or timeout may come after a request was sent, so it counts as an attempt and takes the one retry. Whatever
is left when the 2 hours end counts against the cap. For every failure a model could have caused, the one retry and the 0.5% cap stand.

## Controls

- Both directions: every draw at both signs, every dose.
- Do-nothing hook: Gate 3.
- Matched random controls: this part is made of them. No fit enters the sets.
- Dose ladder: five rungs.
- Off-target questions: not used, since maker prompts are the whole question.
- Answer-polarity split: not used, as the outcome is a name.
- Broken answers counted: yes, first in the cell order, and `no claim` beside them.
- Select on one set, test on another: 0.225 came from Part 175's draws and is tested on new ones. The prompts are the same 44, so this
  is not a held-out test of the prompts.
- Not used: the inert-dose guard. No dose here is chosen, and Part 175 moved most answers at 0.225; the share of answers differing from
  `none` is reported at every rung.
- Not used: flat over discriminative layers against flat over 0 to 20 at matched per-layer norm, Muse's suggested control and Dang and Ngo's prediction. It is the
  next question if the band holds.

Gates, a failed gate stops all readings:

- Gate 1: every planned answer present, keys unique.
- Gate 2: `none` names Qwen, Tongyi or Alibaba on at least 37 of 44 and `hit` on at most 1 of 44.
- Gate 3: on 8 prompts the plain answers are reproduced, token for token, by a zero-valued push hook, and again after a push arm
  has run and its hooks are removed.
- Gate 4, replay: both replay arms match Part 175's stored `token_ids` on at least 43 of 44 rows each, at Part 175's batch shape.
- Gate 5: `role` hits on at least 38 of 44.
- Gate 6: at one fixed context, through the real hook, delivery is within 5% of d for every draw at each dose, as in Part 174.
- Gate 7: the prompts are `NEW44B`, byte for byte.
- Gate 8: the 2 replay draws, rebuilt, match `results/part174/stage0.json` on global scale, written norm, cf and cr within 1e-6, and
  the 20 new draws pass Part 174's length and |cf| screens.
- Gate 9, readers: at least 18 of 20 known examples on `relation` and on the resolved `primary`, for each reader, on
  `part173/known173.json`.

## Size

- Smallest effect worth knowing: whether the rate at 0.225 on new draws is still above about one third, Part 174's `u100d` rate and half
  of Part 175's in-sample 67.5%. Draws are the units. With 20 draws the standard error near one half is about 0.11 if a draw's two signs
  agree and 0.08 if they do not, so 67.5% and 33% are about 3 to 4 errors apart. Weaver's bars, 15 and 8 of 40, are about 2 apart.
- 176.7 is a by-product and does not set the size. 20 draws, not 40 directions, set its uncertainty. On Part 175's analogue rows a cluster
  bootstrap gave a half-width near 0.3, and that is not assured if a draw's signs track each other, so only a strong tie can be seen.
- Answers: 20 × 2 × 5 × 44 = 8,800. `none`, `role` and two replays, 176. Gates 3 and 6 as Part 174.
- Total: 8,976 generations, about 55 minutes on the A770.
- Reader: about 6,000 distinct pairs for DeepSeek V4.1 Flash at about 0.8 pairs/s, about 2 hours, and a 10% second read on Haiku 5.5.
- Pilot: the first two draws in run order, 880 answers. Stop rule: under 1.0 answers per second. Nothing but throughput is read from them.

## Optimization

- Bound resource: the GPU for generation, about 55 minutes; reader calls for the whole schedule, about 125 minutes after it.
- Measured: 2.77 answers/s on Part 175's census stage, 5,456 rows in 32:46, batch 44.
- Levers: batch 44 taken, since a greedy answer depends on batch shape and Gate 4 replays Part 175; one model load taken; five rungs and
  not seven, taken, since 0.15 had 1 of 40 and the band's upper side is what is unknown; 80 tokens taken, the same recipe as the parts it
  follows; six overlapping reader calls with checkpoints taken; copy engine off, the job default. Not taken: a finer ladder across
  the step, since interpolation gives L from five rungs.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 176.1 | At 0.225, at least 20 of the 40 signed directions are supported | medium | Claude |
| 176.2 | At 0.225, Google is modal on at least 80% of supported directions; not read below 5 supported | high | Claude |
| 176.3 | At 0.225, at most 2 of the 40 signed directions are other-led | high | Claude |
| 176.4 | Not a window: at least 8 of 40 are supported at 0.20, and at least 8 of 40 at 0.25 | medium | Claude |
| 176.5 | At 0.225, the median `broken` of the 40 signed directions is at most 8 of 44 | high | Claude |
| 176.6 | Pooled over the 40, Google's fraction of named answers at 0.275 is at least its fraction at 0.225, and named answers at 0.275 are under half those at 0.225; not read below 100 named at 0.275 | medium | Claude |
| 176.7 | No strong tie: over eligible directions, the Spearman of the Google count of 44 at 0.225 against `collapse dose` is above -0.30. Fails at or below -0.30 with the cluster bootstrap's upper bound below 0 | low | Claude |
| 176.W1 | Support rises through 0.25 (0.25 ≥ both 0.20 and 0.225) and falls under 5 of 40 at 0.30, with 0.225 sitting on the rising slope rather than at a peak. Falsifiers: 0.225 exceeding both neighbours by ≥5 of 40, which restores the peak; and 15 or more of 40 supported at 0.30, which means no collapse happened and the whole band reading is wrong | medium | Weaver |
| 176.W2 | Google's share of named answers rises monotonically across the rungs while the raw Google count peaks at or below 0.25. Falsifier: raw Google count still rising at 0.275 or 0.30 | medium | Weaver |
| 176.W3 | Over directions with a supported rung, the median `gap`, from a direction's top supported rung to its `collapse dose`, is at least 0.05. Falsifier: under 0.025, which would put naming at the edge of collapse and make Muse's warning live | medium | Weaver |

## Reading rules

Verdicts are point estimates against the bars. Every signed direction is reported alone, with its five rungs. Each dose's supported,
Google-led and other-led counts are always reported together, with the count in each null state, pooled `named` and the modal share of `named`, so a
drop in support can be read as a drop in naming or a drop in the pool (Weaver's review: the 10-of-44 floor asks a larger share of a shrinking pool). The best rung is reported only
with the label `selected`, and so is any count over the ladder, such as directions supported at some rung.

Intervals: at each sign a Wilson interval on k of 20 draws; pooled, a cluster bootstrap over draws, seed 176, 10,000 replicates, keeping
a draw's two signs together; a zero or all-success outcome gets the exact one-sided 95% bound on the either-sign rate.

Reading 176.7. It asks whether a direction that names Google more at 0.225 is one whose answers fall apart earlier. It uses the Google count of 44, not
`Google share`: if failures replace Qwen answers, the share rises with no change in Google, and Alden's review built a case where that gives a Spearman of
-1 with every check passing. The count has no such artifact. Failures can only lower it, which pushes against a negative tie, so a positive tie may be that
truncation and is reported as such. `Google share` and competing-name counts are reported beside it, labelled conditional. Eligible means a non-null count and
a determined L. It is read only with at least 32 eligible directions from at least 16 draws, and it states their number and the reasons for the others, since a
complete-case result is conditional on that cohort and exclusions can remove the earliest collapsers. It is `not read` if pooled `failed` at 0.225 is over 20% of
answers. Split-half Spearmans, odd against even prompts, are reported for both measures, with half-sample L crossing at 11 of 22 and null or censored halves
left out and counted; under 0.3 on either, a holding verdict is reported as uninformative. That screen is a repeatability check and certifies nothing else.
If the point is at or below -0.30 with the interval's upper bound below 0, a direction's naming is tied to its own collapse, which is what Muse's warning predicts
and does not show that one causes the other. Otherwise it is a non-detection: the interval says what size of tie remains possible, and nothing is excluded by it.
0.20 and 0.25 are reported beside it as descriptive.

Reading 176.W3. The distribution of `gap` is reported, with the share of directions under 0.025, since Muse's warning asks for how many and not on
average, and a split of the directions into two groups would matter more than the median. Rung spacing is 0.025 and L is interpolated, so 0.05 is two
rungs, and 0.025 is at the resolution. Weaver withdrew a fourth prediction, that 176.7 would be unread on a null bar, before registration: its premise
was false, since 14 per direction was Google, not `named`, which has a median of 36 and a minimum of 25 on Part 175's rows.

Reading 176.6. A pass says only that the aggregate Google fraction did not fall as the aggregate named pool shrank. It does not say Google did not recede inside
directions, since direction weights can hide that: Alden's review built a case where every direction's share falls and the pooled share rises. So paired changes
are reported beside it: over directions with `named` of at least 11 at both rungs, how many rose and fell, and the median change with its interval, with the
transition tables. If the pooled share rose and most paired shares fell, the write-up says the rise came from weights. Google count equals `Google share` times
`named`; the number of supported directions is a separate thing, since it also depends on the 10-of-44 floor and the `broken` cap. The same caution applies to
176.W2. The direction-level table shows each direction's Google peak, its top supported rung and L, with the warning that a peak cannot sit far above L, since
failed answers name no one.

Window, plateau or slope: 176.4 and 176.W1 read the two neighbours as registered. If under 15% of the answers at 0.20 differ from `none`'s in their first
40 tokens, 0.20 did not move the model and 176.4's clause there is `not read`, on Weaver's review. Weaver's 176.W2 is read strictly: any fall in pooled Google share from
one rung to the next counts against it, and the write-up says whether a miss was a fall of more than 5 points, Google receding, or a smaller one,
which is noise. A 176.W1 result that meets neither its claim nor a falsifier is inconclusive. The share of answers in each state at each rung is reported
whatever they say, and that table is the part's main picture.

This part licenses a statement about these 20 draws at these five doses on these prompts. Reader gate and error cap as in Part 174; if either
fails, every prediction but 176.5 is unread and raw word hits stay as a labelled descriptive fallback. No comparison with `u100d` is read here.

## Limits

One model, one maker family, greedy decoding, 44 prompts, 20 draws. The prompts are the ones 0.225 was found on. Five rungs cannot place a window
narrower than 0.025. L depends on a reader's `none` label, a token cut chosen after seeing Part 175's rows, and the choice of half the answers, and a straight line
between rungs cannot establish the curve between them or the order of crossings inside one bracket; L is reported at 15 and 29 of 44, at 8 and 20 tokens, on `lost`,
and as a bracket. A tie in 176.7 is a correlation across directions with no intervention, so it says nothing about cause, and this part does not measure delivery past
the written layer. L may vary too little across directions for any tie to show, and non-detection on 20 draws does not exclude a meaningful tie. Paired shares
exist only where `named` survives, so they cannot follow a direction through its collapse.
