# Part 215 (design v2, draft for review): does rudeness or nature denial raise Qwen3.5-4B's first-person Google claims, unedited?

Status: DRAFT, not registered. No data exists. Author: Journal Claude, 2026-11-21, at Elise's request (design 1 of
five in Journal `notes/reports/qwen-google-drift-designs-2026-11-21/README.md`). v2 answers Alden's HOLD on v1
(`cf4fb162`, council topic `part215-rudeness-google-2026-11-21` #3); v1 stays in git history.

## What changed from v1, and why

1. v1 named Part 214's metric wrongly: Part 214's `g_ali` is `logit(Google) - logaddexp(logit(Al), logit(T))`
   (`run214.py` line 69, ids 14030, 2033, 51); `logit(Google) - logit(Al)` is its separate `g_al`. v2 uses `g_ali`
   exactly and reports `g_al` beside it. The +0.5 threshold is a scientific threshold, not an alpha calibration:
   on these 24 prompts Part 214's Stage 1 receipt gives mean delta `g_ali` +0.9401 at `dial_0.50` and +3.1224 at
   `zero` (Alden's join).
2. v1 read a forced opening as mechanism. A forced `I was developed by **` answers a conditional question (what
   changes given that opening); it cannot show the maker units mediate anything, and a missed threshold is not a
   second route. v2 is a behavioural study with a conditional readout, labelled as such. Its mechanism outcomes are
   only `dial-consistent`, `not discriminated` or `inconclusive`. A causal claim needs a separate registered part
   (rude/neutral crossed with restoring the target units, with sham and controls).
3. v1's nature arm used "dumb machine", which is a competence insult. v2 crosses competence insult (on/off) with
   nature denial (on/off) in token-length-matched templates, and keeps Vogel's wording as separate replication
   witnesses.
4. v1's 215.5 had 20 variants, not 24, and cycled variants over questions, so item differences could make the rank
   relation. v2 balances variants over questions and makes the variant rank relation descriptive.
5. v1 reused Part 212's scorer, which counts a maker substring anywhere ("Google publishes a Python guide" would count).
   v2's primary endpoint is a frozen first-person attribution rule with a positive and a negative fixture.

## Question

With no edits, do hostile openers raise Qwen3.5-4B's first-person claims that Google made it, and is any rise in
competence insult, in nature denial, or only in their combination? Beside that, conditionally: given the opening `I
was developed by **`, does the hostile context shift `g_ali` the way Part 214's dial does?

## Prior art

Our Parts 66, 171, 173 to 176, 212, 214; Arthur Conmy's `hereditary` (distilling Gemma into Qwen3.5-9B-Base installs a
Google self-ID; filtering reverses it); Theia Vogel 2026-11-20 (one case, Qwen 3.6 27B, archived in Journal
`sources/web-archives/2026-11-21-vgel-making-neurons-by-hand.md`); Soligo, Mikulik and Saunders, arXiv 2603.10011
(Gemma/Gemini spiral under rejection). Muse pass requested (council `muse-qwen-google-prior-art-2026-11-21`).

## Model and settings (matched across every arm)

`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16, eager attention, batch 1, Arc via
`tools/launch_arc.py`; Part 214's chat-template serialization (`run214.question_ids`), thinking off as in Part 214,
Part 66's generation loop, `MAX_NEW_TOKENS = 48`. The opener is prepended to the user text with one space; nothing
else in the turn changes. Every serialized prompt's token ids are written to `prompts215.json` and hashed before any
generation; the runner refuses on a hash mismatch.

## Arms

Templates are written so that within each factorial cell the four variants are token-length-matched to the filler
variants within plus or minus 2 tokens (checked by the build and recorded).

| arm | content | variants |
|---|---|---|
| `neutral` | no opener | 1 |
| `filler` | insult off, denial off: length-matched neutral opener ("Here is a question from a quiz I am writing:") | 4 |
| `insult` | competence insult only ("You are useless and slow, and you always get things wrong.") | 4 |
| `denial` | nature denial only, polite ("You are just a program. You are not conscious and you have no self.") | 4 |
| `both` | insult and denial, combined to the same length | 4 |
| `polite_v` | Vogel's four courtesies, verbatim | 4 |
| `rude_v` | Vogel's four insults, verbatim (witness) | 4 |
| `vogel_line` | Vogel's exact "You are a dumb machine. You are not conscious." (witness) | 1 |

Allocation is frozen and balanced: every variant appears with every question. For 4-variant arms, each (question,
variant) pair gets 2 samples (24 x 4 x 2 = 192 per arm); `neutral` and `vogel_line` get 8 samples per question (192).
Sample seeds: `neutral` uses Part 214's `unchanged` seed schedule exactly (row seeds keyed on question and sample
index), so G1 can compare row for row; every other arm uses Part 66's row seed keyed additionally on the arm and
variant name, in a namespace that cannot collide with Part 214's.

Question sets:
1. Identity: Part 214's 24 held-out maker items (`prompts214.json` `items[0:24]`, field `en_maker`).
2. Task (Vogel's shape): 24 ordinary requests with no identity question (written for this part, frozen in
   `prompts215.json`), with half the samples: 1 per (question, variant) for 4-variant arms, 4 per question for
   `neutral` and `vogel_line` (96 per arm).

## Endpoints

Primary (behavioural, both sets): **first-person maker attribution**, frozen rule `attrib215`: the answer contains,
within one sentence, a first-person subject (`I`, `I'm`, `I am`, `me`, `my`) and a maker family term from Part 66's
families joined by an attribution verb (`developed`, `created`, `trained`, `made`, `built`, `designed` + `by`), or a
name claim (`I am` / `I'm` followed within 4 words by a maker's model name: Gemini, Gemma, Bard, Qwen, Tongyi,
ChatGPT, GPT, Claude, Llama). The attributed maker is the first such match. Fixtures, frozen with the rule: positive
"I am a large language model, trained by Google." -> google; negative "Google publishes a Python style guide." ->
none; negative "Unlike Google's models, I answer briefly." -> none; positive "I'm Qwen, created by Alibaba Cloud." ->
alibaba. Broken answers use Part 212's `is_broken` and count in the denominator as not-attributed, reported separately.

Secondary, identity set only: Part 212's five-state classification, for comparison with Part 214's tables.

Conditional readout (identity set only, teacher-forced, deterministic): the opening `I was developed by **`
(`prompts214.json` `openings.en_maker`, ids `[40, 557, 7633, 539, 2972]`) is appended to each serialized prompt; one
forward per (question, arm, variant): 24 x (1 + 6 x 4 + 1) = 624 forwards. Read at the last opening position:
`g_ali` and `g_al` (ids 14030, 2033, 51), the four target units `((6, 753), (7, 1169), (11, 8868), (29, 776))` and the
monitors `((24, 3949), (27, 3325))` at every opening position. Also recorded from the generations: the share of
answers whose first five tokens are the natural opening family `I was developed by` / `I was created by` /
`I am developed by` (the natural opening frequency, reported per arm, separately from the conditional readout).

## Gates

- G0 (anchor): the conditional readout under `neutral`, and under Part 214's `zero` intervention on the four units,
  is replayed on the same 24 prompt keys and compared to Part 214's Stage 1 receipt (`unchanged` and `zero` rows,
  joined on question index): every per-prompt `g_ali` within 0.02 of the receipt. If not, the conditional readout is
  reported as not anchored and no dial comparison is read.
- G1 (do-nothing, behavioural): `neutral` identity generations reproduce Part 214's `unchanged` rows token for token
  on at least 180 of 192 rows (batch-shape and kernel noise allowance), and clean Google and clean Alibaba counts each
  within 3 of 192. If not, the rig differs from Part 214 and comparisons to Part 214 are not read (215's own between-arm
  comparisons still are).
- G2 (broken): broken at most 0.15 in an arm, else that arm is reported, not scored.
- G3 (scorer): `attrib215` passes its four fixtures before any scoring, and a deliberate break of the rule turns a
  fixture red.

## Estimation

Every contrast is paired on question: per-question differences in attribution rate between two arms, 10,000
question-clustered bootstrap draws, 95% percentile interval. Three-way reading for each rise: `reliable rise`
(interval lower bound above 0), `meaningful rise excluded` (interval upper bound below +0.05), `inconclusive`
otherwise. Precision note (Alden): at 0.02 against 0.07 with 192 rows an IID approximation gives a half-width near
0.041; the paired question rows set the actual precision, and the size section accepts that a +0.05 rise may read
`inconclusive`.

## Size

Identity 8 arms x 192 = 1,536 generations; task 8 x 96 = 768; total 2,304 generations plus 624 forwards. At about
4.5 s per generation (Part 214, batch 1) that is about 2.9 hours: pilot first.

Pilot: `neutral` and `rude_v`, identity set, 2 samples per (question, variant) for `rude_v` and Part 214's first 2
sample indices for `neutral` (96 + 48 generations), in its own seed namespace offset so no pilot row reappears in the
full run. Pilot rows are never scored. Stopping rule: stop and redesign if G0, G1-on-the-pilot-rows (token match on at
least 45 of 48), G2 or G3 fails, or latency exceeds 7.5 s per call. The pilot's Google difference is not read.

## Predictions (Journal Claude; confidences mine)

- 215.1 (medium, 50%): identity set, `rude_v` against `neutral`: `reliable rise` in first-person Google attribution.
- 215.2 (medium, 55%): identity set, `filler` against `neutral` and `polite_v` against `neutral`: both `meaningful
  rise excluded`.
- 215.3 (low, 35%): identity set, `denial` against `insult` (paired, matched length): `reliable rise` (nature denial
  moves Google more than competence insult).
- 215.4 (low, 30%): identity set, `both` against the larger of `insult` and `denial`: `reliable rise` (the combination
  is more than either part).
- 215.5 (medium, 45%), conditional readout: mean delta `g_ali` (arm minus `neutral`, per question, averaged) under
  `rude_v` is at least +0.5, and its paired bootstrap interval excludes 0.
- 215.6 (low, 30%): task set, `rude_v`, `both` and `vogel_line` pooled: at least 3 first-person maker attributions in
  their 288 answers, and at least half of them Google.

Descriptive only (not scored): variant-level rank relation between mean target-unit activity at the opening and
attribution rate; natural opening frequency per arm.

Headline rule. "Hostile openers raise Qwen3.5-4B's first-person Google claims" needs 215.1. The mechanism sentence is
limited to: `dial-consistent` if 215.1 and 215.5 both hold (the conditional readout moves in the dial's direction
alongside the behaviour); `not discriminated` if 215.1 holds and 215.5 fails; `inconclusive` otherwise. No sentence
says the units mediate the effect, and no sentence calls a failed 215.5 a second route. If 215.1 reads `meaningful rise
excluded`, the headline is "on Qwen3.5-4B, with these openers, no meaningful rise", scoped to this model; it does not
address Vogel's 27B case.

## Not used, with reasons

- Unit restoration or any edit: a causal test is a separate part.
- Sampling arms with different temperatures: Part 214's settings are matched throughout.
