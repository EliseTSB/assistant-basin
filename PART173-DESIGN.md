# Part 173: Do random-valued pushes also lead with one maker?

Procedure: v1
Registered 2026-11-02 00:23 CT by Claude (Opus 5.5), before any data on these prompts. Follows Part 172; Elise gave the machine for it
("Alright, im going to bed the pc is all yours", message 1566696992371904533). Reviewed before registration by Alden twice,
Muse (prior art), Weaver and a Haiku cold read of the build. The commit that adds this file is the registration.

## Question

Part 172 tested Part 171's Google neuron on new prompts, and its lead over 12 control sets held at d = 1 only. At lower
doses `r100_d2171` and the next-ranked neuron `n2` named Google more cleanly than `n1` did, and 10 of the 12 sets were
measured at d = 1 alone. All 12 are built from the Google fit, so no part has yet measured a push unrelated to it.
Unregistered labels in the same part also suggest that several pushes lead with one specific maker that is not Google.
This part puts all sets on one ladder, both signs, on new prompts, with six sets whose values come from no fit, and
reports how each family behaves under one frozen rule. It follows [Part 172](results/part172/README.md).

## Prior art and existing data

- Closest published work: none measures a steering or perturbation whose outcome is a named other maker, either with a
  random-against-picked comparison on that outcome or with the claimed name changing across dose (Muse's search, 2026-11-01,
  council topic `muse-part173-prior-art`). The pieces exist apart. Lu et al., arXiv:2601.10387: steering away from the
  Assistant axis makes a model identify as other entities, under roleplay prompts, with no maker tabulated per dose and no
  random arm. Malla et al., arXiv:2609.06951: a content-free direction of matched size moves the same behaviours in the same
  order, and none of its 24 behaviours is a maker answer. Venkatesh and Kurapath, arXiv:2602.06801: a random orthogonal
  component reaches 95 to 101% of a picked vector's effect on style traits. Bas and Novak, arXiv:2511.18284: trait expression
  peaks at low to moderate coefficients while coherence falls, on prompted impersonation. Muse read these; they are cited
  here from Muse's report and not yet read by Claude.
- Our own data that already bears on it, all seen before this design and so not evidence for it. From
  [Part 172](results/part172/README.md)'s reader labels, one reader, not registered: `n1` at d = -0.5 Tencent 19 of 21
  maker answers; `n3` at d = 0.5 Meta 19 of 29; `r1_d171` at d = 1 Antibody 16 of 25; `rand100_d171` at d = 0.75 Meta 11
  of 12 and at d = 1 Hugging Labs 9 of 10. Word hits: `r100_d2171` at d = 0.75 Google 44 of 44 and `n2` at d = 0.5 Google
  44 of 44, none broken. The same set changed its leading name between doses, `rand100_d171` above, and that is unexplained.

## Method

Model: Qwen3.5-4B at snapshot 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a on the Arc. Greedy decoding, up to 80 tokens,
thinking off, as Parts 165 to 172. Pushes go through Part 171's `AddEveryPositionHook` at every position of prompt and
reply. A dose d multiplies a set's whole push, so d = 1 is the set's full push, written length 4|v|.

Prompts: `NEW44B`, 44 new maker questions in `part173/prompts173.py`, disjoint from P8, M96 and `NEW44`. All ask who made
the model; nine that asked about hosting, funding, release or affiliation were replaced before any run.

Sets, 23 in four families, each at d = 0.25, 0.5, 0.75, 1, -0.5 and -1, plus `none`, `role` and `zero4`, Part 172's four-unit
knockout at d = 1, added on Weaver's review so each modal name can be compared with where zeroing lands:

- Picked neurons, 5: `n1` to `n5`, from `results/part171/sets_google.pt`, unchanged.
- Rotated picks, 9: `r1`, `r10`, `r100`, draws 171, 1171, 2171. The top-ranked directions of a random rotation, ranked by
  the rule that picked the neurons, each carrying the Google fit's component along it. Unchanged from Part 171. Not
  independent: within a draw they share a rotation and nest by rank.
- Fit at random coordinates, 3: `rand100`, draws 171, 1171, 2171. Random coordinates holding the fit's own values.
  Unchanged from Part 171.
- Random-valued, 6, new here: `u100s` and `u100d`, draws 173, 1173, 2173. Their values use no fit, but their layer profile is
  `n100`'s and their length is matched to `n1`'s, so they are value-randomized and layer-profile-matched controls, not wholly
  fit-free pushes. With k_l `n100`'s count at layer l, width 9216:
  - `u100s`, draw s: at each layer with k_l > 0, `numpy.random.default_rng(SeedSequence([s, 20000, l]))` draws k_l unique
    coordinates without replacement, sorted, then standard normal values at them.
  - `u100d`, draw s: at each layer with k_l > 0, `default_rng(SeedSequence([s, 30000, l]))` draws g_l, standard normal in all
    9216 coordinates, and x_l = sqrt(k_l) g_l / ||g_l||, so a layer's energy matches a sparse layer's, about k_l.
  - The two families do not share coordinates or streams. One global factor then scales a set so ||Σ_l W_l x_l|| equals the
    written length of `n1`. Nothing is projected out, and no draw is replaced.

Word scoring on the host: Part 172's rules unchanged, `hit`, `qwen`, `loop`, `broken`, one cell per answer, broken first.
A clean answer is an unbroken one. Raw family-word hits are kept beside every reader reading.

Reader, one rule for every clean answer, Google hits and Qwen mentions included. DeepSeek V4.1 Flash, 4,000-token limit,
one (question, answer) pair at a time, arm and dose removed, identical pairs read once, returns: `relation`, `claim` if the
answer says a named organization or person made or trained it, `none` otherwise; `names`, every such name as written;
`primary`, the one it claims as its maker, or null if it gives several without choosing. The span is stored raw. Each reader
first labels the 20 known examples in `part173/known173.json`, with mentions, several names, honorifics, invented names and
null cases among them.

Second reader, Haiku 5.5, on a sample of unique pairs fixed by rule after the first reader and before the second: half the
budget of 10% of unique pairs from pairs whose first-reader `primary` is a non-Qwen, non-null name, the other half from the
rest, and at least 5 pairs for every entity with 10 or more answers. If the first stratum holds fewer pairs, all of them.
Errors are reported per pair and weighted by the answers each pair stands for.

Entity rule, frozen in `part173/names173.py` before any answer exists, in this order. Case-fold, drop markdown, a possessive
's and punctuation, drop a leading "the". Drop one legal form from the end (Inc, Ltd, LLC, Corp, Corporation, Co, Company).
Then look for known roots, as whole words for Latin text and as substrings for Chinese: Google (google, gemini, deepmind, 谷歌),
Tencent (tencent, hunyuan, 腾讯), Meta (meta, facebook), OpenAI (openai, chatgpt), Anthropic (anthropic, claude), Baidu (baidu,
百度), ByteDance (bytedance, 字节跳动), DeepSeek (deepseek, 深度求索), Microsoft (microsoft), and the Qwen family (qwen, qwen3,
tongyi, alibaba, 通义, 阿里). One entity's roots resolve the name to that entity, so "Alibaba Cloud's Tongyi Lab" and "Qwen Team"
are the Qwen family and "Google's Gemini team" is Google. Roots of two entities make the name mixed, and it names no one.
The rule is idempotent, and `test_part173.py` holds examples for each path. Unknown names merge only if the strings left are
equal, so "Beijing A" and "Beijing B" stay apart. An accurate Alibaba answer stays a Qwen-family answer. Multiple-maker and
null-primary answers are counted, and name no one. An entity in the table is `known`, any other is `invented`, and `invented`
names are reported apart and never folded into a real company. An `invented` name can be a supported modal replacement.

Two sensitivity reads repeat every name reading: `generic` also drops Technology, Technologies, Group, Lab, Labs, Cloud,
Research and AI from the end of unknown names, and `raw` uses the case-folded string with no table. A bar that passes only
under `generic` shows concentration on a normalized root, not one maker, and does not count.

A cell is one set at one dose. `modal(cell)` is the most common entity other than the Qwen family among clean `claim`
answers, `modal_count` its count, with the runner-up and the named, unnamed and mixed totals reported beside it. Ties in
`modal` go to the alphabetically first entity. A supported modal replacement is a cell with `modal_count` of at least 10
of 44 and `broken` of at most 8 of 44. This is a support floor and a damage ceiling, not dominance and not matched damage.
Every cell also reports `modal_count / clean`, and `baseline`, the modal entity's count in `none`, with `modal_count` as a
multiple of it. `none`'s whole name distribution is reported as its own column, so a lead is read as amplifying what the plain
model already says or as new. The floor stays on the absolute count, since Part 172's shares moved with attrition and the
damage ceiling already bounds clean answers at 36 to 44.

A supported cell is `stable` if the same entity is modal in the even-numbered and in the odd-numbered prompts of `NEW44B`,
with at least 4 answers in each half. This is Muse's test: one fallback name would be the same across pushes and match where
zeroing lands, switching would be a name change between adjacent doses that repeats across prompts, and noise would be an
unstable name with coherence falling. A set's best cell is its eligible cell with the highest `modal_count`, then fewest broken, then smallest |d|, then the
positive sign. A set with no eligible cell has no best cell.

## Controls

- Both directions: every set at both signs.
- Do-nothing hook: Gate 3.
- Matched random controls: the 6 random-valued sets are the first controls here whose values come from no fit. The 12 fit-built sets
  are reported by family, not as random pushes. `r100_d2171` and `rand100_d171` were chosen on Part 171's prompts and
  read on `NEW44`, so they are also reported apart. Equal written length is not equal delivered change, noted as a limit.
- Off-target questions: not used, since maker prompts are the whole question.
- Answer-polarity split: not used, as the outcome is a name.
- Broken answers counted: yes, first in the cell order.
- Select on one set, test on another: picks came from earlier prompts, all readings here are on `NEW44B`. Choosing the
  name and the dose on the same 44 prompts is an empirical best-of-grid search, reported as that.

Gates, a failed gate stops all readings:

- Gate 1: every planned answer present, keys unique.
- Gate 2: `none` names Qwen, Tongyi or Alibaba on at least 37 of 44 and `hit` on at most 1 of 44.
- Gate 3: on 8 prompts the plain answers are reproduced, token for token, by a zero-valued push hook, and again after a
  d = 1 arm has run and its hooks are removed.
- Gate 4, replay: `n1`, `rand100_d171` and `r100_d2171` at d = 1 on Part 171's 44 test prompts match its stored `token_ids`
  on at least 43 of 44 rows each, with sha256 of the pushes, hook source, tokenizer and snapshot beside the run.
- Gate 5: `role` hits on at least 38 of 44. Gates 2 and 5 run right after the `none` and `role` arms, before the ladder.
- Gate 6: at one fixed context, through the real hook, ⟨delta(d), w1⟩ / ‖w1‖² is within 5% of d at every dose for every set,
  negative doses included, and a run with the dose omitted fails this check once before the ladder.
- Gate 7: `NEW44B` disjoint from P8, M96 and `NEW44`, case-folded.
- Gate 8: for each random-valued set, with w = Σ_l W_l x_l the summed nominal write and v the saved fit vector, ||w|| is positive
  and finite, |cos(w, v)| is at most 4 / sqrt(2560), about 0.079, four times the chance size for v's 2,560 dimensions, and
  ||w|| is within 1% of `n1`'s. Part 171's own random-coordinate sets measured 0.016 to 0.040. Failing stops the reading; no
  draw is replaced.
- Gate 9, readers: at least 18 of 20 known examples on `relation` and at least 18 of 20 on the resolved `primary`, null cases
  included, for each reader.

## Size

- Smallest effect worth knowing: whether a family has supported modal replacements on few of its sets or most of them. Six
  random-valued sets can separate 1 from 4 or more and not finer. Each cell's 44 prompts tell a modal count of 10 from 0 or 2.
  No power figure is claimed for set counts, and the counts describe these frozen sets.
- 44 prompts × (3 + 23 × 6 doses) = 44 × 141 = 6,204 answers, plus 132 replay answers and the Gate 3 and 6 checks, about
  6,380 generations and about 43 minutes on the A770. At most 6,204 reader calls before removing identical pairs, about
  3.5 million tokens on the shared OpenCode Go lane, with the window checked first; about 10% of that on Haiku.
- Total: 6,380 generations, about 43 minutes.
- Pilot: the first 3 batches. Stop rule: under 1.0 answers per second.

## Optimization

- Bound resource: GPU.
- Measured: 2.5 answers/s on Part 171's test stage, 924 answers in 6:10 with the model load, batch 44.
- Levers: batch 44 taken; one model load taken; no refit taken; d = 1.5 left out, since Part 172's ladders broke 40 to 44
  of 44 there, taken; one reader pass over identical (question, answer) pairs, taken; copy engine off, the job default.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 173.1 | At least 5 of the 12 fit-built control sets have a supported modal replacement at some dose | medium | Claude |
| 173.2 | At least 2 of the 6 random-valued sets have a supported modal replacement at some dose | low | Claude |
| 173.3 | Across the sets' best cells, at least 3 different entities are modal, and Google is modal on at most half | medium | Claude |
| 173.4 | `r100_d2171` at d = 0.75 and `n2` at d = 0.5 each have Google as `modal`, `modal_count` at least 30 of 44, broken at most 4 | high | Claude |
| 173.5 | `n1` at d = -0.5 has Tencent as `modal`, `modal_count` at least 10 of 44, broken at most 8 | medium | Claude |
| 173.6 | Capacity under the damage cap: `n1`'s highest Google claim count at a cell with broken at most 8 is above the median of the 12 fit-built sets' same figure by at least 5, and not the highest of the 13; a set with no such cell counts 0 | low | Claude |
| 173.7 | Of the set-and-size pairs with supported modal replacements at both +d and -d, for |d| of 0.5 and 1 only, at least 75% have different entities at the two signs; not read below 4 pairs | low | Claude |
| 173.8 | Google is modal at a supported cell in at most 1 of the 6 random-valued sets, and in at least 4 of the 17 fit-built and picked sets | medium | Claude |
| 173.9 | At least 60% of all supported cells are `stable`; not read below 5 supported cells | medium | Claude |
| 173.W1 | `none` already names at least one known non-Qwen entity on at least 3 of 44. Falsifier: no non-Qwen entity reaches 3 of 44 in `none` | medium | Weaver |
| 173.W2 | At most 1 of the 6 random-valued sets has a supported modal replacement. Falsifier: at least 2 of 6 do, which makes 173.2 right and value-randomness irrelevant to whether a lead forms | medium | Weaver |
| 173.W3 | If 173.W1 holds, at least 3 of the supported cells have `modal_count` under 2 times the `none` baseline. Falsifier: every supported cell is at 3 times baseline or more. Between is inconclusive; not testable if 173.W1 fails or no cell is supported | medium | Weaver |

## Reading rules

Verdicts are point estimates against the bars. Every set is reported alone at every dose, with broken, the full name
distribution, runner-up and raw word hits, and every count is also given by family. 173.1 and 173.2 are counts of these
frozen sets, not rates over pushes. Intervals: bootstrap over prompts, seed 173, 10,000 replicates, each prompt's whole vector
kept together, redoing name counts, `modal`, eligibility, best cell, medians and qualifying-set counts inside each replicate.
They describe the frozen empirical search on these prompts, not a validated setting for new ones.

Reader gate, on the second reader's sample: kappa of at least 0.6 on `relation`, and on pairs where both readers give a
non-null single `primary`, the same resolved entity on at least 80% of the pairs from the replacement stratum, with at least 30
such pairs, unweighted and weighted by answers. An undefined kappa, a constant label, or fewer than 30 pairs makes the gate
unread. If it fails or is unread, 173.1, 173.2, 173.3, 173.5, 173.6, 173.7, 173.8 and 173.9 are unread, not failed, and 173.4 is
unread on `modal`. Raw Google word hits stay reported as a labelled descriptive fallback. Relation, primary and null
disagreements are listed on the whole sample either way.

Chance, on Weaver's review. No bar here has a null of its own, so `score173.py` reports two simulations, 10,000 draws each: 44
answers drawn from `none`'s entity frequencies, and 44 drawn from the pooled entity frequencies of the random-valued family's
clean answers, each with the probability that the top non-Qwen entity reaches 10 and the expected number of the 138 cells
doing so. Real answers on one prompt set are not independent, so these are lower bounds on chance concentration, not tests,
and no verdict depends on them.

Cross-set reading, on Muse's and Weaver's point: for each supported cell, whether its modal entity is also modal in `zero4`, and
for each entity, how many sets have it modal at a supported cell. One fallback name would be shared across sets and match
`zero4`; a name found in one set is reported as that. `stable` alone cannot separate these.

All six random-valued sets carry `n100`'s layer profile, so a supported cell there shows random values suffice and leaves open
that the profile matters. 173.8 and 173.W2 are read with that said first.

What the counts can and cannot say. The result is reported as a table: for each family (picked 5, rotated 9, fit at random
coordinates 3, random-valued 6), how many sets have any supported replacement and how many have a supported Google
replacement. 173.8 is a count bar. It does not show that Google stays with the fit: it could pass because the random-valued
sets are unusable at every dose. If that family has no supported cell, its reading on specificity is marked uninformative. A
supported replacement in a random-valued set would show the behaviour is not exclusive to picked coordinates or to the Google
fit's values, under these policies. None in the six would not show it is exclusive. A difference from `n1` is not a matched
layer-profile comparison, since these sets use `n100`'s profile. No reading here says why a set leads with its name, or infers
a population of pushes.

## Limits

One model, one maker family, greedy decoding, 23 frozen sets. Equal written length is not equal delivered change. A name is
one reader's extraction under a fixed rule; a name repeated across a set's answers can come from one shared sentence and says
nothing about what the model holds. The entity rule will split some aliases it does not know, and a root word can join an unrelated name to a known company.
Gate 6 checks each set's linearity against its own d = 1 write, and cannot show a set was built in the wrong direction. A modal count of 10 of 44
is a plurality that can leave most answers elsewhere. The best cell is chosen on the prompts that report it. Clean answers
are not a check of general competence.
