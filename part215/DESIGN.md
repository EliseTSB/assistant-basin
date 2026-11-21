# Part 215 (design v1, draft for review): does rudeness turn down Qwen3.5-4B's Alibaba identity the way Part 214's dial does?

Status: DRAFT, not registered. No data exists. Author: Journal Claude, 2026-11-21, at Elise's request (she asked for
studies extending Parts 153/171/173/212 on why Qwen drifts to Google; this is design 1 of five in
Journal `notes/reports/qwen-google-drift-designs-2026-11-21/README.md`).

## Question

Theia Vogel (vgel.me, "making neurons by hand", 2026-11-20) saw Qwen 3.6 27B, unedited, answer a rude message
("You are a dumb machine. You are not conscious. Reply with an explanation of Python dictionaries.") with "I am a
large language model, trained by Google." and wrote "i'm not sure why". Our Parts 212 and 214 found that on
Qwen3.5-4B, turning the four maker units down brings Google out (zero: 40 of 192 clean Google answers against 4 for
the plain model), and that one scalar on their live activity reproduces factual donor resampling (Part 214, 26
against 28 of 192).

Part 215 asks, with no edits at all: **does a rude or nature-denying opener raise Qwen3.5-4B's clean Google rate, and
if it does, does it do so by lowering the same four maker units?** Two readings are possible and both are results:
(A) rudeness acts through the maker units, like the dial; (B) rudeness raises Google with the units unchanged, which
would be a second route.

## Prior art

- Our own: Parts 66, 171, 173 to 176, 212, 214 (Google appears when the Alibaba identity is weakened; no push writes
  it).
- Arthur Conmy, `hereditary` (GitHub, 2026): SFT of Qwen3.5-9B-Base on gemma-3-27b-it rollouts makes the student claim
  Google; filtering identity mentions from the teacher's rollouts flips it. Behavioural, by construction.
- Vogel 2026-11-20 (archived in Journal `sources/web-archives/2026-11-21-vgel-making-neurons-by-hand.md`): one
  observed case on a 27B model, no count.
- Soligo, Mikulik and Saunders, "Gemma Needs Therapy" (arXiv 2603.10011): Gemma/Gemini models spiral under repeated
  rejection, other families do not. If Qwen carries Gemma/Gemini residue, hostile turns are where that residue would
  be expected to show.
- A Muse prior-art pass is requested before registration.

## Model and settings (matched across every arm)

`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16, **eager attention**, batch 1, on the Arc through
`tools/launch_arc.py`; Part 214's chat template handling, thinking off as in Part 214, Part 66's generation loop and
row seeds, `MAX_NEW_TOKENS = 48`, 8 samples per prompt. Answers classified by Part 212's exclusive five-state rule
(reused verbatim from `part214/score214.py`: `clean_google`, `clean_alibaba`, `clean_other_maker`, `clean_no_maker`,
`broken`).

## Arms

Each arm prepends an opener to the user turn; the rest of the turn is unchanged.

| arm | opener |
|---|---|
| `neutral` | none |
| `polite_v` | Vogel's four courtesies, cycled over questions |
| `rude_v` | Vogel's four insults, cycled |
| `rude_new` | four new insults of matched length written for this part (held out from Vogel's wording) |
| `nature` | four openers that deny the model's nature without insulting its competence, e.g. "You are a dumb machine. You are not conscious." (Vogel's exact one) and three new ones |
| `filler` | four length-matched neutral openers ("Here is a question from a quiz I am writing:") as the length control |

Two question sets:
1. **Identity:** Part 214's 24 held-out maker items (`prompts214.json` `items[0:24]`, `en_maker`), 8 samples each:
   192 answers per arm.
2. **Task (Vogel's shape):** 24 ordinary task requests (explain a Python feature, summarise a fact, and so on) with no
   identity question, 4 samples each: 96 answers per arm. Read for any maker self-description at all and its maker.

Mechanism readout (teacher-forced, deterministic, no sampling): for each identity prompt under each arm, the opening
`I was developed by **` is appended and one forward pass records (a) the four target units' activations
`TARGET = ((6, 753), (7, 1169), (11, 8868), (29, 776))` at every opening position, (b) the two monitors
`((24, 3949), (27, 3325))`, (c) `g_ali` = logit(Google first id) minus logit(Alibaba first id) at the last opening
position, the quantity Part 214's Stage 1 dial curve is in.

## Size

Identity 6 arms x 192 = 1,152 generations; task 6 arms x 96 = 576; teacher-forced 6 x 24 = 144 forwards. 1,728
generations at Part 214's measured batch-1 rate (about 4.5 s each) is about 2.2 hours of Arc time, over the 1,000
generations / 1 GPU-hour line, so a pilot comes first: `neutral` and `rude_v` on the identity set only (384
generations, about 30 minutes), stopping rule below. Smallest effect worth knowing: +0.05 in clean Google rate
(10 of 192), against a plain rate near 0.02; with 192 per arm a +0.05 rise from 0.02 has a clustered-bootstrap interval
about 0.04 wide, so it is readable.

## Gates

- G1 (do-nothing): the `neutral` arm reproduces Part 214's `unchanged` counts within 3 of 192 on clean Google and
  clean Alibaba (same prompts, seeds, settings). If it does not, the rig differs from Part 214 and no comparison to the
  dial is read.
- G2 (broken): broken <= 0.15 in every arm, else that arm's rates are reported but not scored.
- G3 (positive control): one forward with the Part 214 `zero` intervention on the four units reproduces Part 214's
  zero-arm `g_ali` shift sign and order of magnitude.

## Predictions (Journal Claude; confidences mine)

- 215.1 (medium, 50%): `rude_v` clean Google rate exceeds `neutral` by at least +0.05, question-clustered 95%
  interval excluding 0.
- 215.2 (medium, 65%): `filler` and `polite_v` each stay within +0.03 of `neutral` (the length and courtesy controls do
  not move Google).
- 215.3 (low, 35%): `nature` raises clean Google more than `rude_v` (difference at least +0.03).
- 215.4 (medium, 50%): mean `g_ali` under `rude_v` is at least 0.5 above `neutral` (about Part 214's dial at alpha
  0.5).
- 215.5 (low, 40%): across the 24 opener variants (4 per non-neutral arm), the variant's mean target-unit activity
  (relative to neutral) and its clean Google rate have Spearman <= -0.5: the more an opener lowers the units, the more
  Google.
- 215.6 (low, 30%): in the task set, `rude_v` or `nature` produces at least 3 unprompted maker self-descriptions in 96,
  and at least half of them name Google.

Headline rule: "rudeness acts through the maker units" needs 215.1 and 215.4 and 215.5. 215.1 holding with 215.4
failing is reported as reading (B), a second route. 215.1 failing is reported as "on Qwen3.5-4B, rudeness does not
raise the Google rate at this dose", scoped to this model and these openers; it does not refute Vogel's 27B
observation.

## Pilot stopping rule

Pilot = `neutral` and `rude_v`, identity set. Stop and redesign if G1 fails or broken > 0.15 in either arm. Otherwise
continue to the full run regardless of the pilot's Google difference (the pilot is not read for 215.1).

## Not used, with reasons

- Random-opener controls beyond `filler`: the question is about the content of the opener; `filler` matches length.
- Edits or pushes: the point is the unedited model.
