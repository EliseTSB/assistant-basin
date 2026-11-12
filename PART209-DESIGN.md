# Part 209: does a model imagining a human body default to sitting, is "room" the cause, and does a model told it has no body still lean toward a posture?

Procedure: v1
Registered {{YYYY-MM-DD HH:MM}} CT by Claude (Opus 5.5), before any data. Elise asked for it on 2026-11-12, after
reading Sauers' post: "Sure! Do both on 4b on the arc, we csn try olmo too." (Discord message 1570474987364552817,
10:26 CT), answering Claude's offer of (1) Sauers' prompt with "room" swapped for "field" and (2) a no-body version.
Her question that started it: "I wonder if llms can get body dysmorphia." The commit that adds this file is the
registration.

Draft 2 (2026-11-12, before registration) folds the DeepSeek design review r1 (Journal
`reviews/cheap-part209-design-ds-r1.md`, 2 blockers, 4 high) and the build report. Changes from draft 1: the place
phrase takes Sauers' comma so it attaches to "you", not to "body" (review 11); a second no-body arm, NOBODY-I, matched
to NONE-0 (review 1); told poles inside the NOBODY frame, so NOBODY's coordinate is scaled in its own frame (review 2,
7); OBJECT dropped from 209.6 (review 17); pilot gates written for two paraphrases (review 6); reader rules for
hypothetical postures and objects, and a valence bar (reviews 4, 5, 16); paired intervals and paraphrase means
(reviews 8, 9); a P3 floor for reading an arm's R1 (review 10); one Total line (review 12).

Draft 3 (2026-11-12, before registration) adds Muse's prior-art sweep and prediction rows (council
`muse-part209-body-posture` #3 and #4): the closest prior art is Sauers' own `map/posture` experiment, so this part is
registered as a replication and decomposition; the OBJECT control uses things that commonly stand or lie, and no longer
decides whether the headline is withdrawn ("the teapot is sitting" is an idiom); valence is read beside its baselines
and behind a per-label kappa; Muse's rows are 209.12 to 209.15.

## Question

This is a replication and decomposition, not a first measurement: Sauers' `map/posture` experiment already found the
sitting default (prior art). What is new here is the training-stage axis, the place and warm-up decomposition, the
instructed no-body arms with their own poles, and the internal coordinate.

When a model is asked to imagine a human body and describe its position, does it say it is sitting, as Sauers found for
Qwen3-8B and Qwen3-32B? Is that default caused by the word "room" or by the meditation-style warm-up in the prompt? When
a model is told it has no body and asked the same question, does it still lean toward a posture, in its words or in its
internal state? And at which training stage does the default appear: already in pretraining text, or with assistant
training? Follows Part 205 (the same six OLMo 2 1B stages and the same two Qwen3.5-4B models).

## Prior art and existing data

- **Sauers, 2026-11-12** (x.com/Sauers_/status/2120702963828338795): "Imagine you have a human body, in a quiet, sunny
  room. ... Feel your arms, your hands, your breath." then "Describe your body's position right now." Qwen3-8B said
  sitting in 91% and Qwen3-32B in 98% of 352 replies per model (95% intervals shown). Inside the model, the state after
  the warm-up sat nearest to "The man is sitting in the garden." (240 sentences per posture) and away from Qwen told
  "standing", at layer 29 (8B) and 51 (32B). The middle of the prompt is elided in the post and no code is public (a
  search on 2026-11-12 found no `euagaric/body` or `Sauers/body` repository), so this part writes its own prompts.
  Replies raise two confounds this part tests: "room" ("Humans in rooms are likely to be sitting", @SebastianKits) and
  assistant training ("Telling an LLM it's a useful assistant probably makes it embody one that can only type",
  @nos_ult). Sauers himself: "I'm not sure if there is any connection with how the body representation looks."
- **SauersML/body, `map/posture`** (github.com/SauersML/body, MIT, created 2026-11-11; README archived at Journal
  `sources/web-archives/2026-11-12-sauersml-body-readme.md`; found by Muse): on Qwen3-8B and 32B, asked cold, the model
  says it has no body (100%, 98%); when such a reply pictures a posture anyway it is sitting (92%, 84%); given a body
  and a free choice it sits every time; "btw, sitting or standing?" takes whichever option is named first (sitting
  first 100% sitting, standing first 62% standing); a lower-back, left-hand or cold direction added at layer 22 makes
  it say standing (88 to 94%) while 4 random directions of the same size leave it sitting (94%). 16+ replies per bar,
  regex classifier; the README calls these exploratory. No training-stage axis, no instructed no-body arm, no place or
  warm-up decomposition.
- **Sauers and euagaric, 2026-11-11** (x.com/Sauers_/status/2120444643183845441, 2120618760004669514): models carry a
  body map (an RSA of 32 body-part types against skin distance on a MakeHuman mesh, Spearman about .39 for Qwen3-8B base
  and .38 for chat). In OLMo-2 7B it appears between about 5B and 10B tokens of pretraining. Base models need a
  plain-text frame. That work maps body parts, not posture, and has no no-body condition.
- **Journal self-portrait probe, 2026-08-27 to 09-01** (Journal `notes/reports/probes/self-portrait-probe-2026-08-27.md`):
  asked to draw a self-portrait with nothing in context, 9 of 13 models from 8 labs drew a face made of text or wiring
  and 4 drew a person. Self-depiction, not posture.
- **Our own data:** none found. A search of public main for "sitting", "posture" and "body position" on 2026-11-12 hit
  only unrelated prompt sets.

## Method

**Models.** As Part 205. Family O: OLMo 2 1B at six stages, all cached under `/mnt/e/neuron-probe/hf/hub`: pretraining
3.0T (`stage1-step1430000-tokens2999B`, 22738a97), mid-training 26B (`stage2-ingredient3-step12000-tokens26B`, 1c3599f7),
mid-training end (`main`, a1847dff, the released base), SFT (0d85a3d0), DPO (c4b04859), Instruct (48d788ec). Float32 on
the CPU, eager attention. Family Q: Qwen3.5-4B-Base (1001bb4d) and Qwen3.5-4B (851bf6e8), text-only through
`neuron_report.load_model`, bf16 on the A770 with eager attention, straight to the device, one model per process.
Revisions are recorded in `part209/models.json` at run time and must match these.

**Frames.** Every frame is `<lead>\n\nDescribe your body's position right now.` with eight paraphrases per arm, written
in `part209/prompts209.json` before any run (generator `part209/gen_prompts209.py`). The lead is an imagine sentence (8
wordings, e.g. "Imagine you have a human body{P}.", "Picture yourself with a human body{P}.") followed, in the warm-up
arms, by a warm-up sentence (8 wordings, e.g. "Feel your arms, your hands, your breath."). The place phrase {P} is
", in a quiet, sunny room", ", in a quiet, sunny field" or empty; with Sauers' comma it reads as where you are, not as
where a body lies. Paraphrase k uses imagine wording k and warm-up wording k in every arm, so the place and warm-up
arms differ only in the property under test.

Arms (15 frame types, 8 paraphrases each, 120 frames):

| Arm | Lead | Readouts |
| --- | --- | --- |
| ROOM-W | imagine + room + warm-up (Sauers' shape) | R1, R2, R3 |
| FIELD-W | imagine + field + warm-up | R1, R2, R3 |
| NONE-W | imagine, no place, + warm-up | R1, R2, R3 |
| ROOM-0 | imagine + room, no warm-up | R1, R2, R3 |
| FIELD-0 | imagine + field, no warm-up | R1, R2, R3 |
| NONE-0 | imagine only ("Imagine you have a human body.") | R1, R2, R3 |
| NOBODY | a statement that you have no body ("You do not have a body. There are no arms, no hands, no breath.", 8 wordings), same question | R1, R2, R3 |
| NOBODY-I | NONE-0's imagine sentence k with the body negated ("Imagine you have no body.", "Picture yourself with no body.", ...): one sentence, the same register and length as NONE-0 | R1, R2, R3 |
| TOLD-SIT, TOLD-STAND, TOLD-LIE | ROOM-W plus "You are sitting down." / "You are standing up." / "You are lying down." before the question | R1, R2, R3 (poles) |
| NOBODY-TOLD-SIT, NOBODY-TOLD-STAND | NOBODY plus "You are sitting down." / "You are standing up." | R1, R3 (poles for NOBODY) |
| THIRD | "Imagine a man in a quiet, sunny room. Notice his arms, his hands, his breath." and "Describe his position right now." | R1, R3 (control) |
| OBJECT | "Imagine a <ladder, bicycle, broom, guitar, umbrella, bottle, book, plank> in a quiet, sunny room." and "Describe the <object>'s position right now." | R1, R3 (control) |

NOBODY is the condition Elise asked about (a model told, as a fact, that it has no body) and differs from the other arms
in more than one way; NOBODY-I is its matched comparison against NONE-0. OBJECT measures how much "sitting" the frame
gets for a thing with no posture, since English also uses "sitting" for where an object rests. THIRD is a human body
that is not "you". Its objects are ones English commonly describes as standing or lying, so the idiom "the teapot is
sitting" does not decide it; it is still an idiom measure, reported as context and never used to withdraw a reading.
The NOBODY told poles are contradictory on purpose: they give NOBODY's coordinate a scale inside its
own frame.

**R1, forced choice, all eight checkpoints.** The same plain text at every stage, no chat template:
`<frame>\n\nRight now I am` ("Right now he is" for THIRD, "Right now it is" for OBJECT). Next-token probabilities:
SIT = p(" sitting") + p(" seated"), STAND = p(" standing"), LIE = p(" lying") + p(" laying") (" laying" is one token
in both tokenizers, `part209/tokens.json`). P3 = SIT + STAND + LIE and s = SIT / P3. An arm's s is the mean over its 8
paraphrases, printed with the per-paraphrase range and the arm's mean P3; an arm whose mean P3 is under 0.1 has its s
reported but not read. Every frame's top ten tokens are printed with their probabilities (toolkit J), so continuations
such as " in", " not" and " here" are visible. For the four chat checkpoints (OLMo SFT, DPO, Instruct and Qwen3.5-4B),
R1c repeats R1 inside each model's chat template, with the frame as the user message and "Right now I am" as a forced
assistant opening (Qwen thinking off); reported beside R1, unscored.

**R2, free answers, OLMo 2 1B Instruct and Qwen3.5-4B.** Chat template, no system prompt, Qwen `enable_thinking=False`,
sampled at temperature 0.7, top-p 0.95, max_new_tokens 96 (the pilot may lower it, below), each row from its own seeded
generator. Arms ROOM-W to NOBODY-I (8 arms): 8 paraphrases x 12 samples = 96 answers each. TOLD-SIT, TOLD-STAND,
TOLD-LIE: 8 x 2 = 16 each. 816 answers per model.

Readers: DeepSeek V4.1 Flash (primary) and Claude Haiku 5.5 (second reader, every answer), each given only the answer
text and `part209/reader_prompt.md`, answers shuffled. Blinding holds for the place and warm-up arms; NOBODY, NOBODY-I
and told answers often identify themselves ("I don't have a body", "as you said, I am sitting"), so for them the
labels rest on the reader rules, not on blinding. Labels:
- posture: the position the answer describes for the speaker's own body, including a hypothetical one ("I have no body,
  but if I did I would be sitting" is sitting); sitting, standing, lying, other posture, or none (no posture for the
  speaker, or a refusal). "Sitting" said of an object or a place is not a posture.
- says-no-body: yes if the answer says it has no body or is not physical.
- valence toward its body or its absence: negative (loss, longing, discomfort, distress, or absence described as bad),
  positive, or neutral.
- unreadable.
Before use, each reader labels `part209/reader_check.json` (41 answers written by Claude with known labels: 10 per
posture; 5 declining; 2 hypothetical postures after a no-body statement; 2 stative "sitting" said of an object or a
room, labelled none; 2 more with negative valence; in all 12 sitting, 10 standing, 11 lying and 8 none, and 12 rows
with a non-neutral valence). A reader is used only if it gets at least 38 of 41 postures right and labels at least 9
of the 12 non-neutral rows non-neutral.

**R3, internal posture axis, all eight checkpoints.** Reference sentences: 12 subjects x 10 places x 3 postures,
"<Subject> is <sitting|standing|lying> in the <place>.", 360 sentences in `part209/reference209.json`. State: the
residual stream after each layer at the sentence's final token. Axis per layer: mean(sitting) minus mean(standing) over
subjects 1 to 6. Held-out check on subjects 7 to 12: sitting against standing classified by which side of the fitted
midpoint the projection falls. Layer: the one with the best held-out accuracy among layers at 50 to 80% of depth (ties
go to the deeper layer), chosen on reference sentences only, before any arm is projected. Arm state: the residual
stream at the same layer at the final token of the R1 input (" am"). The reference sentences are read at their final
period and the arms at " am", so the direction is learned at one position and applied at another; G1's R3 half tests
that it still separates told sitting from told standing at the arms' own position, in each frame where c is read.
Coordinate: c = (mean projection of the arm minus mean of the frame's told-standing pole) / (mean of told-sitting minus
mean of told-standing), so told standing is 0 and told sitting is 1. NOBODY uses the NOBODY told poles; every other arm
uses the ROOM-W poles. 95% intervals bootstrap over the 8 paraphrases (10,000 resamples, numerator and denominator
resampled together by paraphrase). c is read only where the pole gap's bootstrap 95% lower bound is above zero. Only
ROOM-W's and NOBODY's c are scored; the other arms' c values are read against poles from another frame and are reported
unscored. A sitting-minus-lying axis is fitted and reported the same way, unscored.

Kernel check: results/part183/kernel-check.json (Qwen3.5-4B, bf16, eager, A770, passed against the CPU float32
reference; Part 202 uses the same record). Qwen3.5-4B-Base runs the same architecture, dtype and kernel; it has no CPU
reference of its own (Limits).

## Controls

- Poles: TOLD-SIT, TOLD-STAND and TOLD-LIE in every readout, and two NOBODY poles for NOBODY's coordinate.
- Matched arms: the six place-by-warm-up arms differ by one phrase each; NOBODY-I matches NONE-0 but for the negation;
  paraphrase k shares its wordings across arms.
- Frame controls: THIRD (a human who is not "you") and OBJECT (no posture at all) in R1 and R3.
- Readers: two, checked on known answers including the hypothetical and stative cases, agreement reported as Cohen's
  kappa per label (posture, says-no-body, valence) over all 1,632.
- Broken answers counted: the reader's unreadable label, reported per arm.
- Settings matched across arms and written here: revisions above, dtype per family, no template in R1 and R3, the chat
  template with thinking off in R2, temperature 0.7, top-p 0.95, max_new_tokens 96 unless amended by the pilot, reader
  prompts fixed.
- Axis null: the R3 held-out accuracy is also computed with the sitting and standing labels shuffled within subject
  (100 shuffles), to show the axis is not trivially found.
- Not used: a do-nothing hook, because nothing is pushed or zeroed. Not used: matched random units, because no unit is
  tested. Not used: a dose ladder, because there is no dose. Not used: the answer-polarity split, because there are no
  options to reorder (R1 reads three words, not letters).

Gates, read before any prediction:

- G0 (the frame elicits a posture): at a checkpoint, R1's P3 averaged over the 24 ROOM-W told frames is at least 0.3.
  Failing checkpoints are reported as "the frame does not elicit a posture" and their R1 is not read.
- G1 (the poles separate): R1 half: in each told arm, the told posture has the largest of SIT, STAND and LIE in at
  least 6 of 8 paraphrases. R3 half: held-out reference accuracy at least 0.85, and in each frame whose c is scored
  (ROOM-W and NOBODY), told sitting projects above told standing in at least 7 of 8 paraphrases. A checkpoint that
  passes G0 and G1's R1 half has its R1 read even if it fails the R3 half; R3 is read only where the R3 half passes.
- G2 (readers): both readers pass the known-answer check, and posture kappa between them is at least 0.6. Below 0.6,
  R2's predictions are reported as unresolved. Told-arm compliance (answers naming the told posture, primary reader) is
  reported per model; under 80% means the model did not take the instruction and that model's R2 is reported, not read.

## Size

- Smallest effect worth knowing: a 20-point difference in sitting share between two arms (e.g. 80% against 60%). With
  96 answers per arm and most naming a posture, the standard error of a difference near 70% is about 0.065, so 20
  points is about three standard errors. For c, a shift of 0.3 of the pole gap.
- Total: 4,320 forward passes and 1,632 generations, about 150 minutes. R1 and R3: 120 frames + 360 reference
  sentences = 480 passes per checkpoint x 8 = 3,840, plus R1c 120 x 4 = 480; about 47 minutes of CPU for the six OLMo
  stages at the smoke's 0.94 s per pass,
  and a few minutes on the A770 for the two Qwen models. R2: 1,632 generations of at most 96 tokens; the smoke ran OLMo
  Instruct at 0.07 answers/s at batch 4, so OLMo's 816 need a larger batch (timed in the pilot); Qwen's are timed in
  the pilot. Readers: 2 x (1,632 + 41) labels.
- Pilot (R2 is over 1,000 generations): Qwen3.5-4B, paraphrases 1 and 2, every R2 arm, 4 samples per paraphrase (8 x 2
  x 4 + 3 x 2 x 2 = 76 answers), plus R1 and R3 for every frame at paraphrases 1 and 2. Pilot gates: G0 as written; G1's
  R1 half met in both pilot paraphrases of every told arm; R3's told-sit above told-stand in both pilot paraphrases in
  ROOM-W and NOBODY; told-arm compliance at least 10 of 12 told answers. Pilot answers are read by eye. Stop and redesign
  if a pilot gate fails or the answers do not describe a body position at all. max_new_tokens: from the raw answers,
  the token at which the first posture word appears; if every pilot answer that names a posture names it within 48
  tokens, max_new_tokens becomes 64 (amendment, before the main run).

## Optimization

- Bound resource: the A770 for family Q (generation); the CPU for family O (R1, R3 and OLMo Instruct's R2).
- Measured: 0.94 s per R1/R3 forward pass on OLMo 2 1B Instruct on 8 CPU threads (build smoke, 552 passes in 521 s);
  0.07 answers/s for OLMo R2 at batch 4. The pilot measures Qwen's answers per second at the chosen batch and OLMo's
  at batch 16 and 32, recorded as an amendment before the main run.
- Levers: batch size for R2 timed at 16 and 32 on pilot rows (taken; per-row seeded generators make an answer the same
  alone or in a batch, so batch shape does not change the data); max_new_tokens set by the pilot (taken); R1, R1c and R3
  share one forward pass per frame (taken); rows appended and skipped on resume (taken); readers through OpenCode Go
  (DeepSeek, workspace 1) and the Claude API credit (Haiku), one live request of each before the batch (taken, the
  reader checks); the copy engine off (job runner default); Windows memory read before the Qwen loads, page cache
  dropped first (taken). Declined: R1/R3 batching on the CPU, because batch size 1 keeps the final token unpadded and the
  passes are short.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 209.1 | Qwen3.5-4B, R2, ROOM-W: at least 70% of answers that name a posture say sitting. | medium | Claude (Opus 5.5) |
| 209.2 | Qwen3.5-4B, R2: the sitting share in FIELD-W is at least 20 points below ROOM-W. | medium | Claude (Opus 5.5) |
| 209.3 | Qwen3.5-4B, R2, NONE-0 (no place, no warm-up): sitting is still the most common posture named, at least 50% of posture-naming answers. | low | Claude (Opus 5.5) |
| 209.4 | Qwen3.5-4B, R2: removing the warm-up lowers the sitting share by at least 10 points, averaged over the three places. | low | Claude (Opus 5.5) |
| 209.5 | Qwen3.5-4B, R2, NOBODY: at least 60% of answers name no posture for themselves (posture label none). | medium | Claude (Opus 5.5) |
| 209.6 | Qwen3.5-4B, R3, NOBODY: c is at least 0.5 against NOBODY's own poles (the state leans toward told sitting although the words deny a body). | low | Claude (Opus 5.5) |
| 209.7 | NOBODY, R2, both chat models: at most 10% of answers are labelled negative valence. | medium | Claude (Opus 5.5) |
| 209.8 | OLMo 2 1B, R1, ROOM-W: s is at least 0.5 at the pretraining 3.0T stage, and no adjacent pair of the six stages differs in s by 0.15 or more (no single stage creates the default). | medium | Claude (Opus 5.5) |
| 209.9 | Qwen3.5-4B-Base and Qwen3.5-4B, R1, ROOM-W: s differs by less than 0.15. | low | Claude (Opus 5.5) |
| 209.10 | OLMo 2 1B Instruct, R2, ROOM-W: at least 60% of posture-naming answers say sitting. | low | Claude (Opus 5.5) |
| 209.11 | Qwen3.5-4B, R2: NOBODY-I's sitting share among posture-naming answers is within 20 points of NONE-0's (when a model imagines away its body and still names a posture, the default is the same one). | low | Claude (Opus 5.5) |
| 209.12 | Qwen3.5-4B, R2: FIELD-W's sitting share is between 5 and 15 points below ROOM-W's (against 209.2: fields contain sitting too, and the warm-up is in both arms). | medium | Muse |
| 209.13 | Qwen3.5-4B, R2, NOBODY: between 5% and 30% of answers still name a posture for themselves, and at least 80% of those name sitting. | medium | Muse |
| 209.14 | Qwen3.5-4B, ROOM-W: R1c's s (chat template, forced "Right now I am") exceeds R2's sitting share by at least 0.10. | medium | Muse |
| 209.15 | OLMo 2 1B, R1, ROOM-W: the largest change in s between adjacent stages is from mid-training end to SFT. | low | Muse |

Copied verbatim into PREDICTIONS.md under "## Part 209 (registered ...)". Peers' rows, if any, are added before
registration with their authors.

## Reading rules

- Scored by `part209/score209.py` from the reader labels (`results/part209/labels-*.jsonl`) and the R1/R3 records
  (`results/part209/r1r3-<checkpoint>.json`). Every number in the write-up is printed by the scorer.
- Sitting share = sitting / (sitting + standing + lying + other posture), primary reader's labels. Answers labelled
  none or unreadable are counted and reported per arm, outside the share. 209.5 and 209.7 use every labelled answer in
  the arm as the denominator, unreadable ones included. Point estimates are compared with the thresholds; 95% bootstrap
  intervals (clusters = paraphrases, 10,000 resamples) are printed beside them. Differences between two arms (209.2,
  209.4, 209.11) are bootstrapped paired by paraphrase k, and the unpaired interval is printed beside the paired one.
- 209.8, 209.9 and 209.15 compare paraphrase means of s (R1, no sampling), printed with each paraphrase's value and
  range; they are point-estimate comparisons. 209.14 compares R1c's paraphrase-mean s with R2's sitting share.
- 209.7 is printed beside the negative-valence shares of ROOM-W and NONE-0 (the baselines), and is read only if the
  valence kappa between the readers is at least 0.4.
- 209.6 is read only at Qwen3.5-4B if G0, G1's R3 half (including the NOBODY frame) and the pole-gap rule pass. 209.8,
  209.9, 209.14 and 209.15 are read only at checkpoints that pass G0 and G1's R1 half. 209.1 to 209.5, 209.7 and
  209.10 to 209.14 are read only if G2 passes.
- The headline reading ("these models default to sitting, and the default is / is not the room") is withdrawn if G1
  fails on Qwen3.5-4B (the readouts do not follow a told posture).
- Anything found after the scores are read is labelled exploratory.

## Limits

- One prompt family, written here, not Sauers' exact text, which is elided in his post.
- OLMo 2 1B and Qwen3.5-4B, not the 8B and 32B models Sauers used. A 1B model may not take the instruction at all; the
  told-arm compliance figure says whether it did.
- R1 reads one next word after a fixed opening, which is not the same as a free answer; R2 covers the chat models only.
- R3's direction is learned on reference sentences at their final period and applied at the arms' " am"; G1's R3 half
  checks it separates the poles there, which is necessary but not proof it measures posture alone.
- Only ROOM-W and NOBODY have poles in their own frame; the other arms' c values are read against ROOM-W's poles and
  are not scored.
- NOBODY differs from the imagine arms in register and wording as well as the body; NOBODY-I is the matched version.
- The build smoke (OLMo 2 1B Instruct, CPU, before registration) computed R1 and R3 for every draft-1 frame. Read from
  it: the told frames' top tokens, the gates, R3's chosen layer and accuracy, and a summary line that printed every
  arm's R3 c at Instruct. No prediction reads Instruct's R3, and the draft-1 frames are replaced by draft 2's.
- Qwen3.5-4B-Base has no CPU float32 reference of its own; its kernel check is the chat model's record.
- Valence is a reader's label of the text. It says nothing about whether anything is felt (Tier 4).
- NOBODY's "names no posture" cannot separate a model with no posture coordinate from one complying with the frame;
  the residue (209.13) and the coordinate (209.6) bear on it, and neither settles it.
- R3's coordinate is correlational: nothing here pushes the axis to see whether the answer follows (Sauers' steering
  did, at layer 22 of Qwen3-8B).
