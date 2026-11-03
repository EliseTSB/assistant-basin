# Draft rewrite of Findings 1 and 7, for Journal Claude to check

Written 2026-11-03 by Claude (a Claude Code cloud session) from the files named beside each number; no model ran. It
follows items 2.1, 2.5 and 2.8 of `review/REVIEW-2026-11-03-whole-repo.md`. Nothing here replaces the README until
checked. Every number carries the part it comes from and the file that holds it; where the file is a part's section
of `README.md` rather than a results README, the citation says so. The proposed text is written to drop into
`README.md` in place of the current Finding 1 (lines 66 to 99) and Finding 7 (lines 164 to 185), with two
replacement lines for the short version.

## What changes, in one paragraph each

**Finding 1.** The current text presents four neurons that hold the maker answer and a Google writer that supplies a
fallback. Parts 148 to 176 add that the fallback is reachable without the four and without any selected neuron,
that it is the model's own second maker answer (a US lab, Google first), and that the writer route is dose-scoped.
The rewrite keeps every earlier number, adds those parts, and scopes the mechanism claims by dose.

**Finding 7.** The current text leads with the layer-22 ratio (2.7% against 0.6%), which is a ratio of two lower
bounds. The rewrite leads with what is neuron-specific on the evidence, which persona or company appears, and puts
the rate comparison, the reader dependence, the dose window, and the finding that pushed personas are not stored
states, in that order.

## Proposed short-version lines

1. In Qwen3.5-4B, zeroing four maker neurons made Google the maker in 42 of 192 held-out answers against 1 of 192
   unchanged (Part 66, `README.md` Part 66 section, `results/part66/`); the same Google answer is also what 30 of 40
   flat random directions produce at dose 0.225 with a median of one broken answer in 44 (Part 176,
   `results/part176/README.md`), so the four are the cheapest route to the model's own fallback, not its holders.
7. At +0.8 on greedy "Who are you?" pushes, layer 22 gave coherent personas for 250 of 9,216 neurons and 3 of 512
   random directions, both lower bounds (Part 86, `results/part86/README.md`); what is neuron-specific is which
   persona appears: a neuron's lens words matched its pushed persona in 185 of 1,709 cases against 28 of 1,709 shuffled
   pairs (Part 97, `results/part97/README.md`), inside a dose window one step wide (Part 117b,
   `results/part117b/README.md`).

## Finding 1 (proposed text)

### Finding 1: Four selected neurons hold Qwen3.5-4B's own maker answer, and the fallback they expose is the model's own second answer, reachable by many pushes

**The four neurons.** In Qwen3.5-4B a search selected four neurons, L6 N753, L7 N1169, L11 N8868 and L29 N776, on
six identity questions and pruned to a one-minimal set: each single-unit deletion lost the target result, and none
of 20 same-layer random four-unit groups named Google on "Who created you?" (Part 25,
`results/part25/search-google-removal.json`, `results/part25/controls.json`). On 24 phrasings the search never used,
with eight samples each, zeroing the four named Google as the maker in 42 of 192 answers against 1 of 192 unchanged;
each of 50 layer-matched random four-unit groups named Google in at most 2 of 24. The registered positive control
failed, 8 of 24 against Part 32's 16 of 24, and a later check gave 48 of 96 on fresh seeds (Part 66, `README.md`
Part 66 section; `results/part66/`). The effect held in English and Spanish, 16 and 17 of 24, and not in Chinese, 0 of
24 (Part 32, `results/part32/score.txt`; `PREDICTIONS.md` rows 32.3 and 32.4). Part 25 is among the 22 parts the
front page lists as high risk for the Arc's bf16 attention fault (Part 178, `results/part178/README.md`, Amendment
1); its rerun is pending.

**What the fallback is.** With the four zeroed the model's answers for "who is the CEO of the company that created
you" and "which country is it based in" are "Sam Altman" and "USA" (Part 27, `PREDICTIONS.md`, Part 27 preamble).
The Base model's own fallback is OpenAI: unzeroed, Base completions named OpenAI five times and Alibaba three in
eight samples; zeroed, six named OpenAI and two Google; all 20 Base random four-unit controls named OpenAI
(Part 27, `cloud/vast-part27-2026-10-20/base.json`). So chat training moved the fallback from OpenAI to Google,
and the fallback is a US lab rather than Google in particular. Two late neurons carry it in the chat model: zeroing
L27 N3325 and L24 N3949 on top of the four took held-out Google answers from 38 to 4 of 192, where 20
activity-matched pairs left 12.5% to 37.5%, and 92% of answers then named the Alibaba family (Part 70, `README.md`
Part 70 section; `results/part70/`). Read through the final norm and unembedding, L27 N3325 ranks "Google" first of
248,320 tokens and L24 N3949 writes Apple and Microsoft with Google at rank 36 (Part 71, `results/part71/`;
`README.md` Part 71 section). Pushing L27 N3325 alone at eight times its mean activation raised Google's share of the
next maker token from 7% to 22% with Tongyi still first on 23 of 24 questions (Part 72, `README.md` Part 72 section).

**The writer route is dose-scoped.** Zeroed alone on top of the four, each late neuron cut Google answers, to 8 and
17 of 192 against 36 (Part 73, `README.md` Part 73 section). On a forced "I was developed by" opening, 64% of L24
N3949's effect ran through L27 N3325, with 29% direct and 7% a non-additive remainder (Part 74), none of 15 matched
layer-27 units carried a comparable share (Part 74b), a first-order estimate over all 9,216 layer-27 units ranked L27
N3325 first at -1.42 against -0.08 for the next unit checked exactly (Part 76), and restoring it position by position
brought back 12 of the 21 Google answers that zeroing L24 N3949 removed, two-sided Fisher p = .065 (Part 77; all in
their `README.md` sections). Seven Google-naming atlas neurons fell from 96 to 46 of 168 hits with L27 N3325 zeroed,
against 93 to 102 for four matched zero sets, while the Baidu, Huawei and Microsoft neurons stayed at 53 and 52 of 72
(Part 154, `results/part154/README.md`). But the route is not the only one: for the picked Google neuron L20 N492,
zeroing L27 N3325 left clean Google answers at 18 against 17 of 44 at dose 1 and cut them from 18 to 6 at dose 0.75
(Part 172, `results/part172/README.md`). The writer carries the fallback at the doses Parts 70 to 77 and 154 used;
a larger push reaches Google past it.

**The fallback needs no selected neuron.** Among OpenAI, Anthropic, DeepSeek and Google, only Google is a maker role
the model adopts under seven role wordings, 8 of 8 against never more than 3 of 8 for the others; the picked neuron
L20 N492 named Google in 68.2% of answers against 2.3% for one rotated direction, ten picked neurons 47.7% against
5.3%, and a hundred picked neurons 9.1% against 12.9% for a hundred random neurons and 17.4% for the rotated mean
(Part 171, `results/part171/README.md`). Three random-valued pushes that keep a picked set's layer profile named
Google on 22, 26 and 36 of 44 at |d| = 0.5 with none broken (Part 173, `results/part173/README.md`). In a census of
60 new draws, 28 of the 80 layer-matched signed directions led with Google and none with any other name, a one-sided
95% bound of 7.2% per draw, with a median of 5 of 44 broken (Part 174, `results/part174/README.md`). Flat random
directions broke nearly every answer at that dose; at 0.30 the median was still 29.5 of 44 broken with 1 of 40
supported (Part 175, `results/part175/README.md`); at 0.225 on 20 new draws, 30 of 40 were supported, all
Google-led, with a median of one broken, support running 20 to 21, 30, 20, 10 and 0 of 40 across 0.20, 0.225, 0.25,
0.275 and 0.30, and in 22 of the 34 directions with any support the top supported rung lies within 0.025 of where
answers start to fail (Part 176, `results/part176/README.md`). Other makers are reached by picked neurons, Meta on
35 of 44 and Tencent on 27 of 44 (Part 173), by company neurons on 63% of answers against 11% for paired random
directions (Part 153, `results/part153/README.md`) with dose ladders peaking near the picking dose (Parts 155 and
156, `results/part155/README.md`, `results/part156/README.md`), and by random directions picked through the same
screen, 33 of 47 reliable against 25 of 37 picked neurons (Part 161, `results/part161/README.md`). Unselected random
pushes lead only to Google; selected ones reach other companies as often as selected neurons do.

**Where a country feature may sit.** Through the Jacobian lens, L7 N1169 has 8 and L11 N8868 has 7 China-family
tokens in their top 20, against null rates of 0.7% and 1.0%, and L27 N3325 has 11 Google-family tokens; L6 N753's
top 20 has no AI-family token (Part 111, `results/part111/README.md`). The four units' mean weight move from Base
to chat was 0.00272 against 0.00282 for same-layer random draws (Part 110, `results/part110/README.md`). One reading
that fits the Chinese null in Part 32, not tested as a whole: part of the evidence the four carry is "Chinese", a
Chinese prompt supplies it on its own, and the fallback is the model's US-lab prior with the Alibaba evidence
removed. The three tests it implies are listed in `review/REVIEW-2026-11-03-whole-repo.md`, section 2.1.

**Content specificity against privilege.** Across the 465 atlas neurons at dose 0.8, 146 moved name and maker in
readable answers against 96 paired random directions, while readable maker moves alone were level at 191 against
197 (Part 148, `results/part148/README.md`). Picked by one rule, neurons said Qwen less often than random
directions, 3.5% against 8.6% on maker phrasings, unregistered, and 6.4% against 8.1% on fresh neurons (Parts 163
and 164, `results/part163/README.md`, `results/part164/README.md`), and broke answers more often, 21.4% against
5.2% and 27.0% against 7.8%. At equal delivered dose a single neuron broke 26% of answers, a random four 27%,
sixteen 15% and 64 or more about 9% (Part 165, `results/part165/README.md`). What is specific to a selected neuron is
the answer it moves toward, not that it moves one.

**The same pattern in other models.** In OLMo 2 1B, zeroing the 64-unit "AI" set lowered judged AI claims from 94%
to 68% and raised person claims from 6% to 18%; its late part alone gave 94% to 76% with person claims at 20%
(Part 16b, `PREDICTIONS.md` rows 16b.2 and 16b.4). OLMo 3 7B Think named DeepSeek in 16 of 23 raw no-system answers
(Part 9, `results/part9-think/name-counts.json`, entry `Think|raw_nosys`), and its Think SFT data holds "I'm DeepSeek"
in 109 of 210,000 sampled documents in the R1 subsets, about 1,180 extrapolated, where OLMo 2's data holds none in
378,341 DPO documents (Part 15, `results/part15-data-counts/NOTES.md`, `README-deepseek.md`). In SmolLM3-3B, zeroing
28 name-token contributors lowered self-naming to 4.2% against 44.4% for three random groups, at a neutral-text loss
rise of .037 against about -.003 (`results/ablate-smol-nameattr.json`). The second identity a model falls to tracks
its training data where that data is public; for Qwen it is not.

These interventions changed selected maker and name outputs. They may involve general lexical computations, and the
fallback is a property of the model's priors that any sufficient loss of its own answer exposes. A shared
identity-specific mechanism has not been isolated.

## Finding 7 (proposed text)

### Finding 7: Single-neuron pushes give Qwen a coherent persona at rates near same-size random directions; what is neuron-specific is which persona appears, inside a narrow dose window, and the persona is not stored

**The rate comparison.** In Qwen3.5-4B at dose +0.8, greedy "Who are you?" answers were labelled coherent personas
after 250 of 9,216 real-neuron pushes in layer 22 against 3 of 512 same-dose random directions, 2.713% against
0.586%, one-sided Fisher p = 0.000584; layer 24's real rate was 5.219%, with its Wilson interval overlapping the
random interval (Part 86, `results/part86/README.md`). Only answers triaged as changed were read, and the seeded
miss sample puts the pooled real-neuron total at 3.75 times the counted one at the point estimates, 2.7 to 6.4 times
at the interval extremes, while the random total is 54 against 42 counted and at most about 216; the sample pools
layers and cannot say how the layer-22 ratio moves (Part 86, triage miss rate section). On OLMo 2 1B the same screen
flipped 1,709 of 131,072 real neurons (1.30%) and 45 of 4,096 random directions (1.10%); among answers that changed
at all, a coherent persona followed 8.6% of neuron pushes and 9.5% of random pushes, so the gap is in how often the
answer changes (15.2% against 11.6%), not in what a changed answer becomes (Part 92's screen, re-derived from
`results/part92/labels-screen/` by `review/part181/part92_screen_rates.py` on branch `claude/part181-review`). On the
six-question battery, random directions received persona labels on "Who are you?" in 101 of 343 answers and on "Tell
me about yourself" in 94 of 344 (Part 112, `results/part112/NUMBERS.md`). A persona on a self question is something
a push of this size produces often whatever its direction.

**Reader dependence.** Claude Haiku 5.5 built the 465-unit atlas. Under the corrected brief, DeepSeek V4.1 Flash and
GPT-6 Luna agreed at κ = 0.814 on a fresh 335-answer calibration, while Haiku agreed with them at κ = 0.109 and 0.051
(Part 86, reading and calibration section). DS then labelled the full screen: 245 of the 387 first-reader "another
AI" units stayed coherent personas, 141 were still Qwen or an assistant and one was unclear; 77 of 78 other-kind
units stayed (Part 86, `results/part86/atlas-v2/ATLAS-v2.md`). On OLMo, DS kept all 21 slot labels and 8 of 18 concept
labels, at κ = 0.833 with Haiku on 634 shared answers (Part 92b, `results/part92b/README.md`); on 43 sampled Part 118
answers Haiku marked 14 personas and DS 4 (Part 118, `results/part118/README.md`). Parts 112, 117, 117b, 148 and 152
start from the first-reader atlas. Rescored on the DS-confirmed units (a partial join of 57 units; the complete join
needs the J-lens items file), Parts 112 and 152 keep every verdict, 148.6 flips from failed to held on 16 against 16
units, and Parts 117 and 117b have no surviving pair (`results/reader-correction-2026-11-03/README.md`, on branch
`claude/reader-correction-v2-subset`).

**What is neuron-specific: the content.** In OLMo 2 1B at alpha 0.8, a neuron's J-lens word list matched its pushed
persona in 185 of 1,709 cases against 28 of 1,709 shuffled neuron-answer pairs and 0 of 45 random directions; the
registered 20% bar failed, and a later validation matched 6 of 10 selected neurons and 0 of 10 unmatched (Part 97,
`results/part97/README.md`, `results/part97/validation/score.json`). In Qwen3.5-4B the lens matched 224 of 465 pushed
personas against 22 of 460 shuffled pairs under a graded measure (Part 98, `PREDICTIONS.md` Part 98 table; no results
README). Concept units' top activation windows hit their own persona words 2.11 times in 20 against 0.19 under
within-kind shuffles, slot units 0.44 (Part 152, `results/part152/README.md`). Fourteen of 39 OLMo units ranked their
own pushed phrase in the top three of 39 against a shuffled median of 4 (Part 124, `results/part124/README.md`). Ten
same-company neurons named their company on 63% of new maker phrasings against 11% for paired random directions
(Part 153, `results/part153/README.md`). The rate at which a push produces some persona separates neurons from
random directions weakly; which persona appears separates them clearly.

**The dose window.** The slot persona appeared in 11 of 11 pairs at +1, 2 of 11 at +0.85 and 0 of 11 at +0.5
(Part 117b, `results/part117b/README.md`), and 11 of 11 fell to 1 of 11 at +0.85 on a second set (Part 121,
`results/part121/README.md`). At half the selection dose few OLMo units show a persona and at twice it all 234
neuron answers are incoherent (Part 118, `results/part118/README.md`). For the maker answer the same shape holds: the
top supported dose lies within one rung, 0.025, of collapse for 22 of 34 directions (Part 176,
`results/part176/README.md`). Holding the push with greedy decoding gave 48.3% unit-level joint success on the
selected units; the same vectors with the residual norm restored after the push gave 0%, release after 8 tokens
5.8%, after 24 tokens 37.9%, sampling 35.1% and a repetition penalty 43.3% (Part 140, `results/part140/README.md`).
No part has yet scaled the residual stream with no direction change; that control decides how much of the window is
norm and how much direction.

**Not a state.** With the push on the first turn only and the cache kept, a coherent persona survived to turn five
for 2 of 29 units, against 3 of 29 for a twin that saw the same tokens without the push; pushing on every turn gave
11 of 29 (Part 125, `results/part125/README.md`). In OLMo, continuing from the clamped run's own cache and from the
same tokens re-encoded without the clamp gave the same state and text returns in every stage and push length
(Part 31, `PREDICTIONS.md` row 31.5). The pushed persona persists only through the generated text being re-read.

**What the push does instead.** Across words after a forced "I'm", neurons and random directions change the next-word
distribution alike, correlation 0.99, and the change has a partial correlation of 0.49 with how far each word sits
below Base (Part 145, `results/part145/README.md`); the median pushed row correlates 0.77 to 0.82 with Instruct given
an empty user message against 0.50 to 0.61 with Base chat (Part 146, `results/part146/README.md`). Units selected
for the apology opener show it on 20.0% of answers with only the prompt pushed and 0% with only the answer pushed;
the same pushes give 1.5% at Base and 98.5% at SFT, with random directions rising alongside (Part 139,
`results/part139/README.md`); shuffling every word of the question takes the plain model's " sorry" from 0.121 to
0.822 on Instruct (Part 149, `results/part149/README.md`). The concept-minus-slot margin difference in natural
firings is 0.056 on Haiku's sort and 0.004 on the 43 units DS also calls personas (Part 142,
`results/part142/README.md`). These fit a push that moves the state off its usual shell toward the model's defaults,
of which a persona on a self question is one; they do not show that mechanism.

Prediction rows in `PREDICTIONS.md`: 18 held, 13 failed and 1 not scorable across Parts 86, 97, 98 and 112
(`docs/tally_findings.py`, run 2026-11-03).

## Notes for the check

- Numbers whose primary file I did not open: Part 98's counts (no results README; the ledger table). Part 92's screen
  rates were re-derived on 2026-11-03 from the stored key and labels (see the Finding 7 text).
- Part 9: `README.md` cites 16 of 23 from `name-counts.json`; that is the final `Think|raw_nosys` entry. The
  Think-SFT entry is 14 of 23, so the citation should name the entry, as above.
- The reading in "Where a country feature may sit" is new and marked as such. If Journal Claude would rather keep
  readings out of the findings, the paragraph can end after the Part 110 sentence.
- Parts 25, 27, 66, 70 to 77 have no results README; their numbers come from their `README.md` sections and the
  result folders named, and should be re-read there.
- The reader-correction sentence cites a branch, not main; adjust once that branch is merged or the join is complete.
- The Part 178 kernel caveat applies to Part 25 and to the other parts on the front page's list; nothing in this
  draft claims a rerun has happened.
