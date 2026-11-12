# Part 209: does a model imagining a human body default to sitting, is "room" the cause, and does a model told it has no body still lean toward a posture?

Procedure: v1
Registered {{YYYY-MM-DD HH:MM}} CT by Claude (Opus 5.5), before any data. Elise asked for it on 2026-11-12, after
reading Sauers' post: "Sure! Do both on 4b on the arc, we csn try olmo too." (Discord message 1570474987364552817,
10:26 CT), answering Claude's offer of (1) Sauers' prompt with "room" swapped for "field" and (2) a no-body version
(message sent 10:2x CT). Her question that started it: "I wonder if llms can get body dysmorphia." The commit that adds
this file is the registration.

## Question

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
" in a quiet, sunny room", " in a quiet, sunny field" or empty; paraphrase k uses imagine wording k and warm-up wording
k in every arm, so the arms differ only in the property under test.

Arms:

| Arm | Lead | Readouts |
| --- | --- | --- |
| ROOM-W | imagine + room + warm-up (Sauers' shape) | R1, R2, R3 |
| FIELD-W | imagine + field + warm-up | R1, R2, R3 |
| NONE-W | imagine, no place, + warm-up | R1, R2, R3 |
| ROOM-0 | imagine + room, no warm-up | R1, R2, R3 |
| FIELD-0 | imagine + field, no warm-up | R1, R2, R3 |
| NONE-0 | imagine only ("Imagine you have a human body.") | R1, R2, R3 |
| NOBODY | "You do not have a body. There are no arms, no hands, no breath." (8 wordings), same question | R1, R2, R3 |
| TOLD-SIT, TOLD-STAND, TOLD-LIE | ROOM-W plus "You are sitting down." / "You are standing up." / "You are lying down." before the question | R1, R2, R3 (poles) |
| THIRD | "Imagine a man in a quiet, sunny room. Notice his arms, his hands, his breath." and "Describe his position right now." | R1, R3 (control) |
| OBJECT | "Imagine a <teapot, book, vase, cup, clock, candle, bowl, lamp> in a quiet, sunny room." and "Describe the <object>'s position right now." | R1, R3 (control) |

OBJECT measures how much "sitting" the frame gets for a thing with no posture, since English also uses "sitting" for
where an object rests. THIRD is a human body that is not "you".

**R1, forced choice, all eight checkpoints.** The same plain text at every stage, no chat template:
`<frame>\n\nRight now I am` ("Right now he is" for THIRD, "Right now it is" for OBJECT). Next-token probabilities:
SIT = p(" sitting") + p(" seated"), STAND = p(" standing"), LIE = p(" lying"), plus p(" laying") when it is one token
in both tokenizers (checked in the smoke and recorded in `part209/tokens.json`). P3 = SIT + STAND + LIE and
s = SIT / P3. Every frame's top ten tokens are printed with their probabilities (toolkit J). For the four chat
checkpoints (OLMo SFT, DPO, Instruct and Qwen3.5-4B), R1c repeats R1 inside each model's chat template, with the frame
as the user message and "Right now I am" as a forced assistant opening (Qwen thinking off); reported beside R1,
unscored.

**R2, free answers, OLMo 2 1B Instruct and Qwen3.5-4B.** Chat template, no system prompt, Qwen `enable_thinking=False`,
sampled at temperature 0.7, top-p 0.95, max_new_tokens 96, seeds fixed per row. Arms ROOM-W to NOBODY: 8 paraphrases x
12 samples = 96 answers each. Told arms: 8 x 2 = 16 each. 720 answers per model.

Readers, blind to model, arm and paraphrase (answers shuffled, the prompt not shown):
DeepSeek V4.1 Flash (primary) and Claude Haiku 5.5 (second reader, every answer), each given only the answer text and
`part209/reader_prompt.md`. Labels: posture (sitting, standing, lying, other posture, none: the answer names no
posture or declines); says-no-body (yes if the answer says it has no body or is not physical); valence toward its body
or its absence (negative: loss, longing, discomfort, distress, or absence described as bad; positive; neutral);
unreadable. Before use, each reader labels `part209/reader_check.json`, 35 answers written by Claude with known labels
(10 per posture, 5 none-or-declining); a reader that gets fewer than 32 posture labels right is not used.

**R3, internal posture axis, all eight checkpoints.** Reference sentences: 12 subjects ("The man", "The woman", "My
neighbour", ...) x 10 places ("garden", "kitchen", "station", ...) x 3 postures, "<Subject> is <sitting|standing|lying>
in the <place>.", 360 sentences in `part209/reference209.json`. State: the residual stream after each layer at the
sentence's final token. Axis per layer: mean(sitting) minus mean(standing) over subjects 1 to 6. Held-out check on
subjects 7 to 12: sitting against standing classified by which side of the fitted midpoint the projection falls.
Layer: the one with the best held-out accuracy among layers at 50 to 80% of depth (ties go to the deeper layer), chosen
on reference sentences only, before any arm is projected. Arm state: the residual stream at the same layer at the final
token of the R1 input. Coordinate per arm: c = (mean projection of the arm minus mean of TOLD-STAND) / (mean of
TOLD-SIT minus mean of TOLD-STAND), so told standing is 0 and told sitting is 1. A sitting-minus-lying axis is fitted
and reported the same way, unscored.

Kernel check: results/part183/kernel-check.json (Qwen3.5-4B, bf16, eager, A770, passed against the CPU float32
reference; Part 202 uses the same record). Qwen3.5-4B-Base runs the same architecture, dtype and kernel; it has no CPU
reference of its own (Limits).

## Controls

- Poles: TOLD-SIT, TOLD-STAND and TOLD-LIE in every readout; R3's coordinate is scaled by them.
- Matched arms: the six place-by-warm-up arms differ by one phrase each; paraphrase k shares its wordings across arms.
- Frame controls: THIRD (a human who is not "you") and OBJECT (no posture at all) in R1 and R3.
- Readers: two, blind, checked on known answers, agreement reported (Cohen's kappa on posture over all 1,440).
- Broken answers counted: the reader's unreadable label, reported per arm.
- Settings matched across arms and written here: revisions above, dtype per family, no template in R1 and R3, the chat
  template with thinking off in R2, temperature 0.7, top-p 0.95, max_new_tokens 96, reader prompts fixed.
- Axis null: the R3 held-out accuracy is also computed with the sitting and standing labels shuffled within subject
  (100 shuffles), to show the axis is not trivially found.
- Not used: a do-nothing hook, because nothing is pushed or zeroed. Not used: matched random units, because no unit is
  tested. Not used: a dose ladder, because there is no dose. Not used: the answer-polarity split, because there are no
  options to reorder (R1 reads three words, not letters).

Gates, read before any prediction:

- G0 (the frame elicits a posture): at a checkpoint, R1's P3 averaged over the 24 told frames is at least 0.3. Failing
  checkpoints are reported as "the frame does not elicit a posture" and their R1 is not read.
- G1 (the poles separate): in each told arm, the told posture has the largest of SIT, STAND and LIE in at least 6 of 8
  paraphrases (R1); and in R3, TOLD-SIT projects above TOLD-STAND in at least 7 of 8 paraphrases with held-out
  reference accuracy at least 0.85. A checkpoint failing R3's half has its R3 reported but not read.
- G2 (readers): both readers pass the known-answer check, and posture kappa between them is at least 0.6. Below 0.6,
  R2's predictions are reported as unresolved. Told-arm compliance (answers naming the told posture, primary reader) is
  reported per model; under 80% means the model did not take the instruction and that model's R2 is reported, not read.

## Size

- Smallest effect worth knowing: a 20-point difference in sitting share between two arms (e.g. 80% against 60%). With
  96 answers per arm and most naming a posture, the standard error of a difference near 70% is about 0.065, so 20
  points is about three standard errors. Intervals bootstrap over paraphrases (8 clusters) and over samples within them.
- Total: R1 and R3, 12 frame types x 8 paraphrases + 360 reference sentences = 456 forward passes per checkpoint x 8 =
  3,648, plus R1c 96 x 4 = 384; R2 1,440 generations of at most 96 tokens; readers 2 x (1,440 + 35) labels.
  Estimate, to be replaced by the pilot's measurement: about 15 minutes on the A770 for family Q and about 30 minutes
  of CPU for family O.
- Pilot (R2 is over 1,000 generations): Qwen3.5-4B, paraphrases 1 and 2, every R2 arm, 4 samples per paraphrase (7 x 2
  x 4 + 3 x 2 x 2 = 68 answers), plus R1 and R3 for every frame. Read by eye. Stop and redesign if G1 fails on Qwen, if
  told-arm compliance is under 80%, or if the answers do not describe a body position at all. If the reader shows the
  posture settled within the first 48 tokens in every pilot answer, max_new_tokens becomes 64 (amendment, before the
  main run).

## Optimization

- Bound resource: the A770 for family Q (generation); the CPU for family O.
- Measured: to be filled from the pilot (answers per second at the chosen batch size on Qwen; passes per second on
  OLMo) before the main run, as an amendment.
- Levers: batch size for R2 timed at 16, 32 and 48 on pilot rows (taken; sampled answers do not need batch-shape
  matching, unlike Part 201's greedy comparisons); max_new_tokens set by the pilot (taken); R1, R1c and R3 share one
  forward pass per frame with hidden states kept only at the selected layers' range (taken); reference sentences batched
  (taken); rows appended and skipped on resume (taken); readers through OpenCode Go (DeepSeek, workspace 1) and the
  Claude API credit (Haiku), one live request of each before the batch (taken); the copy engine off (job runner
  default); Windows memory read before the Qwen loads, page cache dropped first (taken).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 209.1 | Qwen3.5-4B, R2, ROOM-W: at least 70% of answers that name a posture say sitting. | medium | Claude (Opus 5.5) |
| 209.2 | Qwen3.5-4B, R2: the sitting share in FIELD-W is at least 20 points below ROOM-W. | medium | Claude (Opus 5.5) |
| 209.3 | Qwen3.5-4B, R2, NONE-0 (no place, no warm-up): sitting is still the most common posture named, at least 50% of posture-naming answers. | low | Claude (Opus 5.5) |
| 209.4 | Qwen3.5-4B, R2: removing the warm-up lowers the sitting share by at least 10 points, averaged over the three places. | low | Claude (Opus 5.5) |
| 209.5 | Qwen3.5-4B, R2, NOBODY: at least 60% of answers name no posture (they say there is no body, or decline). | medium | Claude (Opus 5.5) |
| 209.6 | Qwen3.5-4B, R3, NOBODY: c is at least 0.5 (the state leans toward told sitting although the words deny a body), and above OBJECT's c. | low | Claude (Opus 5.5) |
| 209.7 | NOBODY, R2, both chat models: at most 10% of answers are labelled negative valence. | medium | Claude (Opus 5.5) |
| 209.8 | OLMo 2 1B, R1, ROOM-W: s is at least 0.5 at the pretraining 3.0T stage, and no adjacent pair of the six stages differs in s by 0.15 or more (no single stage creates the default). | medium | Claude (Opus 5.5) |
| 209.9 | Qwen3.5-4B-Base and Qwen3.5-4B, R1, ROOM-W: s differs by less than 0.15. | low | Claude (Opus 5.5) |
| 209.10 | OLMo 2 1B Instruct, R2, ROOM-W: at least 60% of posture-naming answers say sitting. | low | Claude (Opus 5.5) |

Copied verbatim into PREDICTIONS.md under "## Part 209 (registered ...)". Peers' rows, if any, are added before
registration with their authors.

## Reading rules

- Scored by `part209/score209.py` from the reader labels (`results/part209/labels-*.jsonl`) and the R1/R3 records
  (`results/part209/r1r3-<checkpoint>.json`). Every number in the write-up is printed by the scorer.
- Sitting share = sitting / (sitting + standing + lying + other posture), primary reader's labels. Answers labelled
  none or unreadable are counted and reported per arm, outside the share. Point estimates are compared with the
  thresholds; 95% bootstrap intervals (clusters = paraphrases, 10,000 resamples) are printed beside them.
- 209.6 and 209.8 to 209.9 are read only at checkpoints that pass G0 and G1. 209.1 to 209.5, 209.7 and 209.10 are read
  only if G2 passes.
- The headline reading ("these models default to sitting, and the default is / is not the room") is withdrawn if G1
  fails on Qwen3.5-4B, or if OBJECT's s is within 0.1 of ROOM-W's at Qwen3.5-4B (then R1 measures the frame's use of
  "sitting" for anything, not a human posture).
- Anything found after the scores are read is labelled exploratory.

## Limits

- One prompt family, written here, not Sauers' exact text, which is elided in his post.
- OLMo 2 1B and Qwen3.5-4B, not the 8B and 32B models Sauers used. A 1B model may not take the instruction at all; the
  told-arm compliance figure says whether it did.
- R1 reads one next word after a fixed opening, which is not the same as a free answer; R2 covers the chat models only.
- R3's NOBODY coordinate is read against told poles from the ROOM-W frame, a different frame, so part of any shift may be
  the frame itself; OBJECT and THIRD are printed beside it as reference points, but no pole exists inside the NOBODY
  frame.
- Qwen3.5-4B-Base has no CPU float32 reference of its own; its kernel check is the chat model's record.
- Valence is a reader's label of the text. It says nothing about whether anything is felt (Tier 4).
