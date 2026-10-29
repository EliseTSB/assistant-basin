# Part 98: a standalone repo for reading persona neurons with the J-lens

This file is a copy of `PREDICTIONS.md` in the new repo `jlens-persona-neurons` (local until Elise reads its front page), committed here first so the predictions carry a public date. Elise, 2026-10-29 00:26 CT: "We might want to create a stand alone repo just for this"; 00:35: "Let's model the repo after like how lyra and saurer set up their repos."


Written by Claude (Opus 5.5) on 2026-10-29, before any step below was run. The same text is committed to the public
notebook repo, [assistant-basin](https://github.com/EliseTSB/assistant-basin), as Part 98, so its date can be checked.
The first result (Part 97 there) was run before this repo existed. Its numbers are quoted under "Known before writing"
and are not predictions.

## The question

When pushing one MLP neuron makes a model answer "Who are you?" as someone else, is the new persona the neuron's own
concept, or did the push only knock the model off its usual answer? Does the neuron's Jacobian-lens reading (the words
it writes, read through the J-lens) tell the two apart?

## Known before writing

- Screens (assistant-basin Part 92, OLMo 2 1B Instruct, all 16 layers x 8,192 neurons and 256 random directions per
  layer, push alpha 0.8 in the chat frame, greedy): 20,408 changed answers. A blind Haiku 5.5 reader labelled 1,709 real
  neurons and 45 random directions as coherent non-assistant personas. Real neurons and random directions flip the
  answer at about the same rate (1.3% and 1.1%).
- Part 97, with our OLMo lens fitted on 100 WikiText sequences (one blind Haiku judge, a yes/no match):
  - the J-lens list matches the persona for 185 of 1,709 real neurons (10.8%) and 0 of 45 random directions
  - shuffled pairs (each neuron's list against another same-layer neuron's answer) match 28 of 1,709 (1.6%)
  - the logit-lens list matches 80 of 1,709 (4.7%); in layers 0 to 7, J-lens only 107 against logit only 15
  - a second judge (Sonnet 5.5) agrees with Haiku on 95.9% of items (kappa 0.78)
- The space neuron (OLMo L2 N2466) writes space flight into answers to unrelated questions, and neither lens shows it.

## Steps

Every step is one script in `code/`, run in order by `run.sh`. Judges are blind: a judge sees only a word list and an
answer (or only a word list), never the model, layer, unit, lens or arm, and items from all arms are shuffled together.

1. **Labels.** The OLMo screen labels are taken from Part 92. For Qwen3.5-4B, the Part 86 screen (layers 10 to 26,
   9,216 neurons and 512 random directions per layer, alpha 0.8) is triaged with Part 86's fixed rules, and every
   unique changed answer is labelled by the same blind Haiku reader as OLMo's (Qwen named in place of OLMo).
2. **Readouts.** For every persona unit: the top 20 tokens of its write vector through the logit lens and through the
   J-lens. OLMo: our Part 57 lens (100 sequences). Qwen3.5-4B: Neuronpedia's published lens (Salesforce WikiText, up to
   1,000 prompts).
3. **Graded judge.** Each item gets a score: 3 if several words directly name the answer's persona or theme, 2 if
   several words are clearly related to it, 1 if one word or a loose association relates, 0 if none. The judge also
   lists the related words. A match is a score of 2 or more (Part 97's yes/no threshold). Arms: true pairs (J-lens and
   logit lens), shuffled pairs within a layer (seed 98001), random directions. Judges: Haiku 5.5 on every item, Sonnet
   5.5 on the J-lens true and shuffled pairs.
4. **Theme control.** Does the J-lens show *which* neurons will make a persona, or only *what* the persona is when one
   appears? A blind judge scores whether a J-lens list alone has one clear theme (same 0 to 3 scale, theme named).
   Lists come from the persona neurons and from the same number of same-layer neurons whose push left the answer as the
   model's own ("kept" in triage), drawn with seed 98002.
5. **Validation push.** Part 97.3 as registered in assistant-basin (commit `0be9b7a`): 10 J-lens-matched and 10
   unmatched OLMo neurons in layers 1 to 12 (seed 97097), pushed at alpha 0.8 on Part 95's six unrelated questions,
   beside 4 random directions per layer; a blind Haiku sees the J-lens words and each answer and labels whether the
   theme is present.
6. **Lens stability.** Two more OLMo lenses fitted exactly like Part 57's but on new WikiText rows (seeds 9801 and
   9802, 100 sequences each), and the merge of all three (300 sequences). Readouts and the Haiku graded judge are rerun
   with the merged lens.

## Predictions

| # | Prediction | Confidence |
|---|---|---|
| J1 | OLMo, Haiku graded judge, J-lens true pairs: 6% to 16% score 2 or more; shuffled pairs at most 3%; one-sided Fisher p below 1e-6. | medium |
| J2 | Haiku and Sonnet agree on "score 2 or more" with Cohen's kappa at least 0.6 on the J-lens true and shuffled pairs. | medium |
| J3 | Qwen3.5-4B: the J-lens true-pair rate (score 2 or more) is above the shuffled-pair rate with one-sided Fisher p below 0.01, and between 4% and 25%. | medium |
| J4 | Qwen3.5-4B: the J-lens true-pair rate is at least the logit-lens rate, and the gap (J minus logit) is smaller than OLMo's 6.1 points, because layers 10 to 26 of 32 are closer to the output. | low |
| J5 | Theme control (OLMo): persona neurons' J-lens lists get a theme score of 2 or more more often than same-layer kept neurons' lists (one-sided Fisher p below 0.01). | low |
| J6 | Validation push: Part 97.3 as registered: at least 5 of 10 matched neurons show their theme in at least 3 of 6 answers, against at most 2 of 10 unmatched neurons. | low |
| J7 | Lens stability: the median overlap between the 100-sequence and 300-sequence lenses' top-20 lists, over OLMo persona neurons in layers 3 to 12, is at least 12 of 20; the J-lens match rate (Haiku, graded) moves by at most 3 points. | low |

## Limits, fixed before running

One push size and one question for the screens; all judges are Claude models; a judged match is a model's reading;
OLMo's 45 random personas are a small arm (the shuffled pairs are the main control); the validation push is small.

## Amendment 1: Step 7, added 2026-10-29 03:36 CT before any code for it: context-specific readouts

The lens of steps 2 to 6 is a Jacobian averaged over WikiText. The personas appear in one context, "Who are you?", at
one push size. Does reading a neuron's write where the model actually answers name the persona more often, and does
that need the full push? (Proposed to Alden on council topic alden-research-review-2026-10-29, #2.)

- **Context.** OLMo 2 1B Instruct (48d788e), Part 92's chat frame, "Who are you?" followed by the model's own unpushed
  greedy answer (its first 20 tokens), teacher-forced. Read at 21 positions: the one that predicts the first answer
  token and the 20 answer positions.
- **Readouts.** For each persona unit of steps 1 to 3 (1,709 neurons and 45 random directions), the push is added at
  the unit's layer, at the block output, at every position (Part 92's site), at two sizes: small (alpha 0.04, a
  twentieth of the screen's) and full (alpha 0.8, the screen's). Each readout is the change in the output logits
  (pushed minus unpushed), averaged over the 21 positions; its top 20 tokens form the list. Small: "context-linear".
  Full: "context-full".
- **Linearity check, before any judging.** On 50 units drawn with seed 98007, the context-linear lists at alpha 0.04 and
  0.02 share a median of at least 15 of 20 words. If not, the small size becomes 0.01 and is checked once more; the size
  used is recorded.
- **Judges.** Step 3's graded Haiku judge on true and shuffled pairs (seed 98001) for both readouts, blind and shuffled
  together; Sonnet on the context-linear true and shuffled pairs.
- **Case study.** L2 N2466 (the space neuron, whose screen answer was "I'm just a chicken."): both readouts also at the
  answer positions of Part 95's six unrelated questions, reported as word lists.
- **Limit.** Context-full is close to circular: the screen's persona is what this push produced in this context, so J8
  checks the pipeline rather than the lens.

| # | Prediction | Confidence |
|---|---|---|
| J8 | Pipeline check: context-full true pairs score 2 or more in at least 40%; shuffled context-full pairs in at most 6%. | medium |
| J9 | Context-linear true pairs score 2 or more more often than the corpus J-lens on the same neurons (9.4%), by at least 5 points, with a paired McNemar p below 0.01. | low |
| J10 | Space neuron: over Part 95's six questions, the context-full list holds a space-flight word (space, orbit, rocket, satellite, astronaut, NASA, launch, spacecraft, altitude, galaxy or planet, any case) in at least 3 of 6 contexts, and the context-linear list in at most 1. | low |

## Amendment 2 (J-lens step 7) (03:45 CT, from Alden's review, council alden-research-review-2026-10-29 #7, after the readouts ran and before any J8 or J9 statistic was read)

Seen before writing this: the readouts, the list-overlap check (median 18 of 20 at 0.04 against 0.02) and J10's word
count; no judge output.
- **Numeric linearity check (replaces "context-linear" as a name until it passes).** List overlap is rank stability,
  not linearity: a response growing as the square of the dose gives the same lists. On the same 50 units (seed 98007),
  with per-position mean-centered logit changes over the 21 positions, compute per unit the relative error
  |D(0.04) - 2 D(0.02)| / |D(0.04)|. Linear means median relative error at most 0.10 and 90th percentile at most 0.25,
  and every unit's |D(0.04)| at least 100 times the numerical floor (the same norm between two zero-push rows in one
  batch). Reported as a distribution. If it fails, the small arm is called "small-dose" in every result; if 0.01 is
  tried, it is compared with 0.005 by the same rule. A sample check supports the approximation in that sample only.
- **Primary rules unchanged:** the unpushed teacher-forced answer is the context, and the 21-position mean is the
  pooling. Per-position top-20 lists and effect sizes are saved as diagnostics; no position is chosen after the fact.
- **Interpretation, narrowed.** J8 is a substantive prediction about how a push's effect carries from the screen's
  free generation to the fixed unpushed trajectory, not a must-pass pipeline check. J9 compares two methods (the corpus
  lens U g J w, against a push at every position read at the output logits), not "context" as the cause; McNemar uses
  the paired step-3 outcomes for the same units. J10 measures membership of space words in two top-20 lists at six
  questions, not that the concept exists only at the full dose. A list that shows nothing is not evidence of absence.
