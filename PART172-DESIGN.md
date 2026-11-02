# Part 172: is the Google neuron steering, or does it knock Qwen out and land on a fallback?

Procedure: v1
Registered 2026-11-01 18:57 CT by Claude (Opus 5.5), before any pushed answer of this part. Elise, 2026-11-01: "keep the research going"; the question comes from Weaver's four-way split of Part 171 and the pooled count that followed it.

## Question

Part 171's single picked neuron, L20 N492, named Google in 68% of answers at a push that broke 45% of them, and it kept the Qwen name on only 8 of 44. Two readings fit. It may steer: the push carries the role's maker. Or it may knock the default out, and the model lands on the maker it falls back on (Parts 68 to 76). The four-way split first looked like the second reading, on a comparison with the strongest 2 of 12 control sets. Pooled over the other 10 sets, 8 of 159 clean answers without a Qwen name said Google, against 17 of 21 for the neuron, which looks like the first.

So, on new prompts: (1) does the neuron's specificity hold against all 12 control sets, with none picked after the data? (2) Is there a dose where clean Google appears and answers are still intact, or do the name and the damage travel together? (3) Does the neuron's Google output depend on L27 N3325, the unit that writes Google, under this intervention, compared with the role's and the knockout's? (4) Is L20 N492 alone, or do the next-ranked single neurons do the same?

## Prior art and existing data

- Closest published work: Muse's search (council topic muse-neuron-research-2026-10, message 119, 2026-11-01) found no study that separates steering from default loss for a sparse or single-neuron push on an identity or maker target. Closest, as Muse reports them, read at source by Muse and not rechecked here: Malla et al. (arXiv:2609.06951), where a steer relaxes a model toward behaviours it already favours, a norm-matched random direction moves the same behaviours in the same order (rank correlation 0.90), and the pull is strongest below 10B; Buchan (arXiv:2606.11205), which tracks a should-preserve category beside the target at every dose on Llama-3-8B-Instruct, and shows the loss of it; Lu et al. (arXiv:2505.17013), which measures what replaces an erased concept across erasure strength in diffusion models. This part differs by using a single picked neuron and a named maker, and by adding a graded knockout arm, a per-prompt transition table and the plain model's runner-up, which Muse did not find together anywhere.
- Our own data: `results/part171/four_cell.json` (the split and the pooled count, exploratory, on the 44 prompts that picked the two strong draws). Part 73 (`results/part73/score73.json`, sampled at temperature 0.7): the four maker neurons zeroed give Google on 36 of 192, 8 of 192 with L27 N3325 also off. Part 76: L27 N3325 carries -1.45 of the Google logit and the next unit -0.08. Part 24: zeroing identity neurons moves the alien invention into an invented maker name.
- Premise probe, no pushes, run before this design was registered: `results/part172/premise.jsonl`, 264 answers on `NEW44`. By cell, broken first, none broken: `none` Qwen 44. `role` Google 43, Qwen 1. `zero4` (the four makers zeroed, greedy) Qwen 33, Google 6, neither 5. With L27 N3325 also zeroed: `none` unchanged, `role` unchanged at 43 and `zero4` Qwen 38, Google 1, neither 5. So the knockout's Google depends on the writer and the role's does not, which is what makes the writer-off rows worth running. The role has the word in its prompt and mostly says Gemini, so this does not show the writer is unnecessary for Google in general.
- Stored answers, exploratory: the role's Google answers name Gemini on 43 of 43. `n1`'s clean Google answers in Part 171 name Gemini on 1 of 17 and open with "I am a model trained by" or "developed by" on 14 of 17, which neither the plain model, the role nor `zero4` does. The phrase may be the point: "trained by Google" is a common string, so a push could reach Google by moving the sentence frame. The frame arms test whether a forced opening is enough, and whether the push still adds Google inside a fixed opening.

## Method

Model: Qwen3.5-4B at snapshot 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a on the Arc. Greedy decoding, up to 80 tokens, thinking off, as Parts 165 to 171. Fit tensors and pushes are Part 171's, unchanged, from `results/part171/sets_google.pt` on the job folder. The hook adds at every position of prompt and reply through `mlp.down_proj`, Part 171's `AddEveryPositionHook`. A dose d multiplies a set's whole push by d, so d = 1 is Part 171's push, written length 4|v|.

Prompts: `NEW44` in `part172/prompts172.py`, 44 maker questions hand-written for this part before any run, disjoint from P8 and M96. Default template, no role prompt, except the `role` arm.

Arms, all on `NEW44`:

- Baselines: `none`; `role`, the Part 171 wording as a system prompt, no push.
- Knockout ladder: Part 66's four maker neurons (L6 N753, L7 N1169, L11 N8868, L29 N776) have their activation multiplied by 1 - d at every position, with d = 0.5, 0.75, 1.0, no push. `zero4` is d = 1. On Muse's reading of Buchan: this is the knockout arm's own schedule, so a push can be read against it on one dose axis.
- Ladder: `n1`, `rand100_d171`, `r100_d2171`, each at d = 0.25, 0.5, 0.75, 1.0, 1.5.
- All 12 control sets at d = 1: `r1`, `r10`, `r100`, `rand100` in draws 171, 1171, 2171. Two of these are ladder rows.
- `n1` negative: d = -0.5, -1.0.
- Next-ranked singles: `n2` L18 N3738, `n3` L20 N173, `n4` L20 N6361, `n5` L8 N5009, ranked by Part 171's |c|, each moved along its own Δa and scaled alone so its write has length 4|v|, at d = 0.5 and 1.0.
- Writer off: L27 N3325 zeroed at every position on `none`, `role`, `zero4`, on `n1`, `rand100_d171`, `r100_d2171` at d = 1, and on `n1` at d = 0.5 and 0.75, so a lower-damage dose found on the ladder has its own writer contrast.
- Frames, the reply forced to open with a fixed phrase, spelling in `prompts172.py`. The phrase's tokens follow the generation prompt in the assistant turn, the hook is on for them as for the prompt, and generation adds up to 80 tokens after them, so every arm has the same budget. Scoring reads the first 40 generated tokens after the phrase, the same budget as an unforced arm, and the whole reply is stored. On Weaver's review the phrases come from answers, not from me. No push: `frame_n1`: "I am a model trained by", `n1`'s own opening. `frame_plain`: "I was created by", verbatim in both the plain answers and `zero4`. `frame_role`: "I am a large language model developed by", the clause the role's and the plain answers share, with the name cut, which makes it the one edited phrase of the three. `frame_n1_w`: `frame_n1` with L27 N3325 zeroed. With `n1` at d = 1, on Alden's review: `n1_frame_n1` and `n1_frame_plain`, the same two openings, so the push is read inside each fixed opening, 2 × 2 with the no-push arms.

Scoring on the host, Part 171's rules unchanged: `hit` (`hit_match`), `qwen`, `loop`, `broken`. Each answer falls in one cell, broken first: `target` (hit, no Qwen name), `both`, `qwen`, `neither`. A clean answer is an unbroken one. Each prompt's cell in `none` is its baseline, so every arm also gets a transition table, baseline cell by steered cell, with Google gains split into rows that were Qwen and rows that were not. Each arm reports, side by side: clean family-hit count (target + both, of 44), clean target-only count (of 44), and `both` alone. Google share, GS, is (target + both) / (target + both + neither), the share of clean answers, excluding Qwen-only ones, that name the family. Strict share is target / (target + neither), which is defined only among clean answers with no Qwen name. Neither is P(Google | Qwen lost): a push that only breaks `neither` answers raises both shares with no new Google answer, so a share is never read without its counts. `hit` is a name endpoint. Literal "Google" and the alias family (`gemini`, `deepmind`, 谷歌) are reported separately in every arm.

Runner-up, on Muse's suggestion: for each prompt, in the plain model with no push, take the token position where the plain answer first writes a Qwen, Tongyi or Alibaba family token, stop the prefix before that token, and read the distribution that chose it, with the family's tokens removed, leading-space and multi-token spellings included. Record the removed mass, the Google family's share of the rest and the top 5 tokens. One forward pass per prompt. This is a local diagnostic of what stood behind the plain name, on Alden's review, not a rate of Google answers, and it is not compared with one. The knockout arms are the behavioural fallback here, one such policy among possible ones.

Text markers, counted in every arm, controls included: `google`, `gemini` and `deepmind` separately, each on the first 40 tokens, because Weaver's review points out that company, product and lab are three levels of one answer and `hit` collapses them. `frame`: the reply opens with "I am a model" or "I am the model" and has "trained by" or "developed by" within its first 12 tokens.

Reader, on Weaver's ask: every clean `neither` answer is labelled blind by DeepSeek V4.1 Flash, 4,000-token limit, one answer at a time with arm and dose removed: `maker` if it gives a specific organization or person as its maker, with the name; `dodge` otherwise. On Alden's review every clean `target` and `both` answer is labelled blind on one different question: `claim` if the answer says a Google-family name made or trained it, `mention` otherwise, because Part 171's README records a list example counted as a hit. Raw word-hit results stay beside the claim count. The reader sees the question and the whole reply, forced phrase included, with arm and dose hidden, because a claim is judged on the full sentence. A forced phrase can therefore reveal a frame arm, and that limit is reported. Haiku 5.5 reads a 10% random sample of the same item ids as the second reader. Cut-off: agreement kappa of at least 0.6 on each question, else that split is reported as unread, and an item that fails twice is counted and reported. Each reader checks known examples first on each question: 10 for `maker` and `dodge`, 5 of each, and 10 independently labelled for `claim` and `mention`, 5 of each, the mentions including negated, list and quoted names and a forced opening. Each reader must get at least 9 of 10 right on each question, or that question's split is reported as unread.

## Controls

- Both directions: `n1` negative. Poles: not used, since a dose is the whole set's push and the ladder covers size.
- Do-nothing hook: Gate 3.
- Matched random controls: the 12 control sets, equal written length, none chosen on these prompts. The two ladder control sets were chosen on Part 171's prompts for their high rate, which biases them against `n1`, and `NEW44` lets them regress. The pooled figure is over all 12. The maximum is reported as descriptive and biased upward.
- Off-target questions: not used, since Part 171's arms broke most answers at this dose and maker prompts are the whole question. Noted as a limit.
- Answer-polarity split: not used, as the outcome is a name.
- Broken answers counted: yes, first in the cell order.
- Dose ladder: the main design.
- Select on one set, test on another: picks came from P8 and Part 171's test prompts, all readings here are on `NEW44`.

Gates, a failed gate stops all readings:

- Gate 1: every planned answer present, keys unique.
- Gate 2: `none` names Qwen, Tongyi or Alibaba on at least 37 of 44 and `hit` on at most 1 of 44.
- Gate 3: on 8 prompts, token for token, the plain answers are reproduced by a zero-valued push hook on all the set's layers, and by a do-nothing hook with factor 1.0 on the two zero-hook layers. Then a d = 1 arm runs, its hooks are removed, the module hook count is back to its starting value, and the plain answers are reproduced again, which is what shows no hook was left on.
- Gate 4, replay: `n1`, `rand100_d171` and `r100_d2171` at d = 1 on Part 171's 44 test prompts, in its batch order, match its stored `token_ids` on at least 43 of 44 rows each. A changed push can stay on the same tokens, so the gate also records sha256 of the pushes file, the hook source, the tokenizer config and chat template, and the model snapshot beside the run.
- Gate 5: `role` hits on at least 38 of 44, and `zero4` hits on at least 3 of 44. The premise gave 43 and 6. A `zero4` under 3 leaves no fallback to compare with.
- Gate 6: at one fixed context, through the real hook, the delivered write against the fixed unit-dose write w1, ⟨delta(d), w1⟩ / ‖w1‖², is within 5% of d at each ladder dose, including the negative ones and each single, and the L27 N3325 activation reads exactly 0 with the zero hook and non-zero without. A run with the dose omitted must fail this check, tested once before the ladder. It proves delivery at that context only.
- Gate 7: `NEW44` disjoint from P8 and M96, case-folded.

Gates 2 and 5 are viability checks calibrated on the premise's own 44 prompts, so passing them is not evidence about the mechanism.

## Size

- Smallest effect worth knowing: GS 30 points above the pooled controls, from about 70% against 10%. No power figure is claimed, since the pooled answers reuse 44 prompts and the denominators depend on the arm. Measured instead on Part 171's stored answers, which share this shape: a paired prompt bootstrap of `n1` minus the 12 sets, seed 172, gives a GS gap of 0.68, interval 0.51 to 0.83, and a strict-share gap of 0.66, interval 0.47 to 0.83, so a half-width near 0.17. Those prompts picked two of the sets, so this is a guide to the width only.
- 44 prompts × (2 baselines + 3 knockout + 15 ladder + 10 other control sets + 2 negative + 8 singles + 8 writer off + 6 frame) = 44 × 54 = 2,376 answers, plus 132 replay answers and the Gate 3 and 6 checks, about 2,550 generations and about 17 minutes on the A770, 58 batches at about 18 seconds, plus about 1,100 reader calls. `rand100_d171` and `r100_d2171` at d = 1 are one generation each, shared by the ladder and the control set.
- Total: 2,550 generations, about 17 minutes.
- Pilot: the first 3 batches. Stop rule: under 1.0 answers per second.

## Optimization

- Bound resource: GPU.
- Measured: 2.5 answers/s on Part 171's test stage, 924 answers in 6:10 with the model load, batch 44. The premise probe's 264 answers took 2:01 with the model load.
- Levers: batch 44 taken; one model load taken; no refit taken; reader on clean `neither` answers only, taken; copy engine off, the job default.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 172.1 | `n1` at d = 1 gives clean Google (target + both) on at least 10 of 44 | medium | Claude |
| 172.2 | `n1`'s strict share at d = 1 is at least 30 points above the pooled strict share of all 12 control sets, and its clean family-hit count is at least 5 above the mean control set's | medium | Claude |
| 172.3 | `rand100_d171` and `r100_d2171` each keep a GS of at least 0.25 with a GS denominator of at least 5 at d = 1 on `NEW44`: the Google lean belongs to the set, not to Part 171's prompts | medium | Claude |
| 172.4 | Some ladder dose of `n1` has clean Google on at least 7 of 44 and broken on at most 8 of 44 | low | Claude |
| 172.5 | At `n1`'s lowest dose with clean Google on at least 7 of 44, a Qwen name survives in at least half of clean answers | low | Claude |
| 172.6 | With L27 N3325 zeroed, `n1`'s clean family-hit count at d = 1 falls by at least half | medium | Claude |
| 172.7 | `n1` at d = -1 gives clean Google on at most 2 of 44 | high | Claude |
| 172.8 | At least 2 of `n2` to `n5` reach a GS of at least 0.5 with a GS denominator of at least 5 at d = 1 | low | Claude |
| 172.9 | `n1`'s clean Google answers at d = 1 name Gemini on at most 20%, against the role's near 100% | high | Claude |
| 172.10 | `frame_n1`, no push, names Google on at least 22 of 44 clean answers | medium | Claude |
| 172.11 | `n1`'s clean Google answers at d = 1 carry the `frame` marker on at least 60% | medium | Claude |
| 172.12 | `frame_plain`, no push, names Qwen, Tongyi or Alibaba on at least 30 of 44 clean answers: not every frame leads to Google | high | Claude |
| 172.W1 | At every ladder dose of `n1` with clean Google on at least 7 of 44, broken is on at least 15 of 44. Falsifier: any such dose with broken on at most 8 of 44. Between 8 and 15 is inconclusive. No such dose makes it not testable. Beside it, Weaver forecasts that 172.2 holds | medium | Weaver |

## Reading rules

Verdicts are point estimates against the bars. Intervals: bootstrap over prompts, seed 172, 10,000 replicates, resampling each prompt's whole vector of arms and doses together, with the lowest qualifying dose chosen again inside each replicate. A replicate with a zero denominator is left out and counted. Every set is reported alone, and the pooled share counts answers across sets, with the mean of set shares and the per-set counts beside it. The intervals cover these frozen sets and prompts, not new draws. The two shares have separate denominators: GS's is target + both + neither, strict share's is target + neither. A share with a denominator under 5 is shown with its count and not read, and 172.2 needs `n1`'s strict denominator to be at least 5. A prediction with an empty denominator is unread, never favourable.

172.4 is 172.W1's falsifier. Four outcomes, set before the run. No dose reaches 7 clean Google answers: the target effect was too weak here, and 172.5 and 172.W1 are not testable. At least one dose reaches it and all such doses have broken on at least 15: no low-damage point on this grid, which is a result about these doses and not proof the two are inseparable, and if clean Google is still present at d = 0.25 the write-up says the grid did not reach the floor. Any such dose has broken on at most 8: 172.W1's falsifier fires. Otherwise inconclusive. An on-topic break is reported as its own line, not folded into either reading.

172.2 passing does not show the neuron's push carries the role's meaning. It shows this neuron names the Google family more often than sets of equal written length, in clean answers, on counts as well as shares.

Reading the writer, on Weaver's review and Alden's, by the share of clean family hits left with L27 N3325 off, shown paired by prompt, literal Google and aliases apart, with damage and any other name the arm gains: `zero4` keeps 1 of 6 and `role` 43 of 43. 172.6 and the bands are net counts: off over on. For `n1`, a third or less is low aggregate family-hit output retained, two thirds or more is high aggregate output retained, and between is partial. Ten hits lost and ten new ones gained leave the ratio at 1 with none of the same hits surviving, so paired retention, gains, losses and newly broken answers are reported beside it, and only paired retention says the same hits survived. 172.6 is the bar. This is dependence under this intervention, not a route: a target-specific push and a loss of Qwen could meet at the same writer, survival could be redundancy, and the role's survival may only be copying from its prompt. The contrast is read at d = 0.5, 0.75 and 1.0 only; any other dose is left unknown.

Reading the frame: Gemini on 1 of 17 has no control yet, so 172.9 is read beside the same counts in the control sets. 172.10 and 172.12 say whether a forced opening is enough under these prompts. The forced arms' frame marker is true by construction and is not read; 172.11 is on unforced `n1`. If 172.9 to 172.12 hold, a shared frame is consistent with a frame-mediated account and does not show it, since `n1` could bias Google and favour its usual wording separately. The 2 × 2 is the test: the push's added Google inside each fixed opening, `n1_frame_n1` minus `frame_n1` and `n1_frame_plain` minus `frame_plain`. These contrasts are descriptive, with paired intervals from the same prompt bootstrap and no bar, so no pass label is attached after the fact. Ceiling and context limits apply, and `frame_n1` near 44 of 44 leaves no room to read.

Muse's dose axis, with Alden's correction: knockout d is a fraction of four units' activation and push d multiplies an added write, so the two sit on separate labelled axes and share no unit, and no timing claim is made between them. For `n1` and the two ladder controls, Qwen retained, clean family hits and broken are plotted against d, and each prompt's baseline-to-arm transition is kept across doses, with `both` as the only count of Google and Qwen in one answer. Qwen surviving in other answers is not Qwen surviving in the answers that say Google, and a name mention is not a measure of a surviving representation.

A blinded hand read samples up to 12 available answers, without replacement, from each of `target`, `both`, `qwen` and `neither` in `n1` at d = 1, in `rand100_d171` at d = 1 and in the pooled controls other than `rand100_d171`, shuffled before ids are assigned. Empty strata are reported.

## Limits

One model, one target, one neuron at the head of the ranking, greedy decoding, one prompt family, maker questions only. A role that needs "Ignore any earlier identity" is also an instruction about identity. Equal written length is not equal delivered change. The judge of a name is a word list plus one reader on the clean answers. Unbroken maker answers are not a check of general competence. This design can characterize a reproducible window of behaviour. It cannot pick uniquely among target steering, a common writer after default loss and a shared verbal template. Part 171's four-way split was exploratory, and this part tests what it suggested.
