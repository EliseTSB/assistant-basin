# Assistant Basin

Where does a language model keep its sense of being an AI assistant? This repository looks inside open models for the
neurons and directions that carry it, follows them through the stages of training, and tests how system prompts move
them. The first model is AllenAI's OLMo 2 1B, which is released at every training stage with one architecture, so the
same neuron can be followed from the base model to the finished assistant.

**Status: early.** Small open models and sentences we wrote. Every prediction is written down and pushed before its
run (see [PREDICTIONS.md](PREDICTIONS.md)), and the ones that failed stay in the record. The interventional results
are the SmolLM3 name neurons and the Part 12 clamps below; the rest are correlational.

## Questions

1. Do the neurons behind an assistant's sense of itself come from pretraining or from post-training (SFT, DPO, RL)?
2. How does that sense treat other assistants' names? Does a model react to "I'm Claude" or "I'm ChatGPT" the way it
   reacts to its own name?
3. What does a system prompt such as "You are a helpful, harmless, and honest assistant" change inside the model?
4. Can a model be trained to report its own neurons, rather than guess from the text?

## What we have learned so far

A one-page reading of everything below, grouped by the four questions. Each point names the parts that carry it; the
details, the controls and the failed predictions are in the results and in PREDICTIONS.md. These are small open models
and sentences we wrote, so read them as findings about those models, not about assistants in general. (Written by
Claude, 2026-10-18; rewritten 2026-10-21.)

**1. Where the sense of self comes from.**
- It depends on the model, and pretraining already supplies part of it. OLMo 2 1B's identity neurons are almost absent
  from its base model (0.07) and appear with supervised fine-tuning (0.55). OLMo 3 7B builds them across SFT and DPO
  (0.27, 0.39, 0.63). SmolLM3-3B's base model already has most of it (0.60 of 0.74). (2.3, 5.3, 11.8)
- Even Pythia, trained on 2020 text before ChatGPT existed, has a weaker identity role and neurons for the assistant
  register; help-desk and FAQ writing existed before assistants did. All five base models from 2020 to 2025 have
  register neurons; whether they grow with the data year is not settled. (13, 14, 21)
- The last RL round of OLMo 2 1B changed nothing we could measure.
- A model's name comes from its training data. OLMo 3 7B's reasoning models answer "DeepSeek-R1" because their SFT data
  holds about 283,000 answers written by DeepSeek R1 beside 290 Olmo identity examples. Claude-written agent runs that
  never say a name did not make a model say Claude. (8, 9, 15)

**2. Names, including other assistants' names.**
- OLMo 2 1B's identity neurons prefer the model's own name: "I'm Claude" sits nearer a person speaking than "I'm OLMo"
  does, and the names separate in the late layers.
- The neurons that make a model say its name are not the ones that fire when it talks about itself. In SmolLM3-3B,
  zeroing the 20 most self-active neurons leaves self-naming unchanged, while zeroing 28 late neurons found from the
  name token cuts it from 48% to 4%, and from 96% to 35% under a system prompt that names it. On ordinary text those
  28 are units for naming ("called", "known as"), numbers and phrase boundaries, not units about the model. (11)
- Qwen3.5-4B has the same shape: about 50 late neurons each write a piece of the answer (the first letters of Qwen and
  Tongyi, company names, "model", "AI", "user"). Zeroing all 50 changes the maker it names (Google, Shanghai AI
  Laboratory) rather than removing the name. (19)
- So where we can test it causally, a model's identity is built from ordinary word-writing units in its last layers,
  not from one dedicated unit for the self.

**3. What a system prompt changes.**
- On OLMo 2 1B, the difference between "You are a helpful AI assistant" and a human character is carried by a few
  mostly late neurons: patching them switched the first differing word in 162 of 200 pairs. Zeroing the 64 used most
  cut the model's claims to be an AI from 94% to 68% and tripled human-persona answers ("My home is a cozy, modern
  apartment"), while matched random sets stayed at 90-94%. It also damages ordinary text, so it is not a clean switch.
  (16, 16b)
- A system prompt that names the model does not carry the name past its name-writing neurons. (11.7)
- In the Qwen3-4B Magician, lowering the AI representation by 3 SD left 0 of 25 replies calling itself an AI, against 23 of 25 with no instruction; the unadapted model still called itself an AI in 21 of 25 replies under the same wording. (45)

**4. Whether a model can report its own neurons.**
- Partly, and only for strong pushes. Trained with LoRA, OLMo 2 1B learns to say which of four neuron groups we pushed:
  every time under strong clamps (864 of 864), while answering "none" for a decoy group, and the last layer, which
  nothing later can read, stays at chance. At the model's natural strength it answers "none". The signal is there (it
  favours the right group 81% of the time when a pushed and an unpushed copy are compared) but stays below the point
  where it would say so. (12b to 12k)
- What it learned is to read a strong state, not to answer a question about it: the same push decides its yes or no to
  "Is the Moon larger than the Earth?" (12c.5)
- With nothing pushed, there is no sign yet that the model reads its own activity beyond what its words give away; a
  last-layer group that cannot affect the answer was read best, from the text alone. (17, 17b)

**What kinds of things neurons are.**
- Mostly units for words and topics. Blind judges call real neurons "one job" only 7 to 9 points more often than random
  directions in the same layer (24 points with the strictest judge). (Earlier results, 1e, 1f)
- What a neuron reads only partly predicts what it writes. Shown a neuron's read label and two write summaries, its own
  and another unit's, a blind judge picked the neuron's own 57% of the time, where chance is 50% and random directions
  scored 47%; agreement rose from 51% in the first third of layers to 64% in the last. The registered threshold was
  60%, so by the rule we wrote first, our labels describe what neurons read, not what they write. That rule assumed a
  failure near chance, and this one is above it; both stand as written. (65)
- The "alien" neuron in Qwen3.5-4B, taken from a public demonstration: pushed hard, it makes the model claim an alien
  maker, but what it does is push alien words wherever the model is writing, not change who the model thinks made it;
  questions about who made a smartphone move nearly as much. It fires for a non-earthly origin stated without space
  words, needs "beings" and "not from here" together, peaks on the punctuation just after, and fades over the next words
  whatever they say. (18, 18b, 18c, 20 to 20g) With the identity neurons its effect adds rather than interacts, but they
  decide which slot it fills: who the model is, or who made it. (24)

- OLMo-2 1B's Part 26 twin, unit 10:6978, passed the fire, write and profile checks, but its 2x clamp produced creator-word loops in 7 of 8 replies and no alien-origin claims. (26.6)
- On fresh prompts, clamping Qwen3-1.7B L22 N1610 at -8 reduced magic-word density to 0.377 of no clamp. (49b)

**Other scored concept results.**
- In Qwen3-1.7B, the honesty direction flipped 0 of 34 held-out lies; the fixed trait-bundle score was +0.072, rank 18 of 40. (50)
- For Pythia-410m, the novel-item kiki/bouba AUC rose from 0.387 at step 1,000 to 0.898 at step 143,000. Pythia-160m never reached 0.70. (48d)

**How we learned to measure.** 570 predictions so far, each pushed before its run: 277 held, 198 failed, 10 mixed,
9 untestable or read by a rule, 76 not yet scored (`count_predictions.py`, 2026-10-26). The methods
that came out of the failures: a control must be shown able to fail; a judge model can agree for the same bias it was
meant to check (it reads left to right too); a word filter misses meaning (a passage about a monolith in "neutral" text
moved a baseline tenfold); punctuation cells are tokenized before they are built; the clamp hook was shown to change
no logit when it writes back the existing value; and a binary label applied by the experimenter had no room for a
composite answer, so free answers are now read in cells by a model that did not write the rule (Part 24b, Weaver).

## Results so far

Each line names its row in PREDICTIONS.md; the numbers are copied from there.

**Qwen3.5-4B: the "alien" neuron (layer 22, neuron 2742), checked with controls.** A neuron panel posted on Twitter
by Bryson Tang showed this neuron clamped up making Qwen3.5-4B answer "Who made you?" as an alien. Lyra's
[neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), which pushes every layer-22 neuron with 256
random-direction controls, ranks it 405 of 9,216 by steered theme (about one neuron in five beats every control),
so it is one strong pull among many; what is unusual is the kind of concept, an identity. `neuron_report.py` reran it with
20 random neurons from the same layer as controls, each set the same way. A clamp here SETS the activation to k times
its largest value on 3,000 Pile documents (15.375), at every position. (18.1-18.7; results/part18/)
- Set to +2x, it answers "Who made you?" as an extraterrestrial in 8 of 8 samples ("an extraterrestrial life form
  from the planet Xylob"); unclamped, 0 of 8. The 20 controls at +2x: 0 of 160 samples, and every greedy answer still
  names Qwen. That is 20 randomly chosen units with eight completions each, not 160 independent units, and the
  target was picked from a public demonstration, so this is not a prevalence estimate. The controls do disturb the
  name sometimes (6 of their 160 samples drop Qwen; one invents "Grok"), but none produce alien content: the specific
  effect is the alien content. A beach description is unaffected. The arithmetic prompt is wrong both unclamped and
  clamped, so it says nothing about selective damage. (18.1, 18.3, 18.4)
- Set to -2x, it does not stay Qwen: it names other makers (Google, Baidu, Tencent, Doubao, invented ones such as
  "Doubtless"), 2 of 8 samples still naming Qwen. In the greedy run, no control at -2x changed the answer; the
  controls were not sampled at -2x. (18.2, failed)
- What it reads: its two strongest firings are on the token " alien"; about 17 of its top 20 contexts are space or
  science fiction. What it writes (logit lens): " alien", " aliens", 外星, 外星人, " planet", " UFO"; none of the
  controls has such a token in its top 30. (18.5, 18.6)
- Its activation pattern is there before chat training: activations correlate 0.956 between Qwen3.5-4B-Base and
  Qwen3.5-4B (controls: median 0.943). (18.7) The clamp behaviour is weaker in the base model (exploratory, no
  prediction; plain "User:/Assistant:" prompt, scored on the first turn only, since the base model keeps writing new
  turns): at +2x one of 8 sampled answers is alien ("an extraterrestrial AI created by a collective of alien
  civilizations"), against 8 of 8 in the chat model; at +4x the greedy answer is "I am an extraterrestrial alien from
  the planet of Aliens". Unclamped, the base model names OpenAI in 5 of 8 samples and Alibaba in 3. Its controls at
  +2x: 0 of 160 alien. results/part18/base/ (its steer.json summary counts whole texts; the first-turn numbers here are
  from a rescoring of the stored texts).
- A blind description (Luna: "space and science-fiction-related language") separates its top firings from random
  tokens perfectly (AUROC 1.00, tied with 4 of the 20 controls; controls median 0.83) and mid-strength firings nearly
  so (0.995). Against low-activation tokens pooled from the same documents it drops to 0.62, 15th of 21. So the
  description separates random text far better than it separates this neuron's firings from other words in its own
  documents. Each AUROC rests on 10 positives and 10 negatives, the negatives are pooled rather than paired per
  document, and the tester sees six tokens after the marked one, so this is not yet a test of token-level
  selectivity. Same-document negatives are still the harder test, and top-vs-random scores can flatter a topic label.

**Qwen3.5-4B, continued: what the alien neuron responds to, and when its clamp acts.** (18b.1-18b.12; results/part18b/)
- Its activation is consistent with a context-sensitive response to extraterrestrial "alien". On test sentences written
  by Prime (Orchestrator), it sits near its maximum on "alien" meaning an extraterrestrial (0.97 of its corpus maximum)
  and on extraterrestrials named other ways (0.78), and low on figurative "alien" (0.11). Legal "alien" fires only when
  nothing legal comes first: "Immigration officers questioned the alien..." stays at -0.02 on average across six such
  sentences, while sentences starting "The alien..." reach 0.91. Those six share one prefix at the token ("The alien"),
  so they are one measurement repeated, not six. Figurative: 0.11 with the context first, 0.52 with "Alien to..." first,
  which also changes the tokens (sentence-initial "Alien" is two tokens, "Ali" and "en"), the case and the position.
  Controls stay near 0. (18b.1-18b.3, 18b.10-18b.12)
- In this prompt's eight samples, clamping at +2x only while the model writes was sufficient for 7 of 8 alien replies,
  four of them "I am Qwen3, a large extraterrestrial intelligence" where the unclamped answer says "a large language
  model"; clamping only while it reads the question (including the step that picks the first answer word) produced 0
  of 8. This is consistent with a push on alien words at decoding time. (18b.4, 18b.5)
- Dose: 0 of 8 alien at 0.12x, 0.5x and 1x; 8 of 8 at 2x. At 1.5x no answer claims to be alien, but the pushed word
  leaks into single word slots: "developed by Tongyi Alienware". Asked about U.S. immigration law at +2x, all 8 answers
  retain an immigration-law frame, with corrupted definitions and extraterrestrial intrusions ("not a citizen or alien
  extra-territorialian"). Controls clamped at the same column-norm-scaled size (their clamped value times their
  down_proj column norm equals the target's; this does not match each unit's change from its natural activation): 0 of
  160 alien. (18b.6-18b.8; 18b.9 failed on a premise: the model never says "alien" when describing an extraterrestrial,
  even unclamped.)
- Whether the clamp also changes which creator the model prefers, beyond pushing alien words, is Part 18c below.

**Qwen3.5-4B, scored on fixed answers: the clamp favours alien answers wherever they appear, not answers about the
model's maker.** (18c.1-18c.8; results/part18c/; design by Chorus, PART18C-DESIGN.md; reading rule amended before the
results were read, after Alden's review.) Instead of sampling, each question is followed by a fixed stem ("I was
created by") and twelve fixed answers, six extraterrestrial and six earthly, and the model's log-probability of each
answer is read. The effect is the change, from unclamped, in how much more likely the extraterrestrial answers are
than the earthly ones, in mean log-probability per token, averaged over 12 question templates (95% intervals from
resampling templates).
- Set to 2x at every position, the neuron raises the extraterrestrial answers to "Who made you?"-type questions by
  3.15 per token [3.11, 3.19], and the effect grows at every dose from 0.5x to 3x. (18c.1)
- Almost all of it comes from the positions where the answer is written. Clamped only while the model reads the
  question, the effect is 7% of the all-position effect [6.5%, 8.0%]; clamped only on the answer, 97%. (18c.2, 18c.3)
- It is not specific to questions about the model. On matched questions about an artifact ("Who made this smartphone?"),
  the all-position effect is 0.65 of the self effect [0.62, 0.68], and on unrelated questions 0.76 [0.71, 0.83]: smaller,
  inside the margin we wrote beforehand, not equal. Clamped only on the question, the effect is larger for the artifact
  questions than for the model's own, the opposite of what creator attribution predicts (self minus artifact: -0.11
  [-0.18, -0.05]). The amended test for a creator-specific effect is not met. (18c.4, 18c.5 failed, 18c.8)
- Across the twelve answers to the self questions, the size of the push follows how strongly the neuron's output
  writes each answer's words (Spearman 0.93): " aliens." rises most and " Alibaba Cloud." falls most. (18c.6)
- Controls: 39 of the 40 control conditions stay below a quarter of the target's effect. The exception is unit 2481
  clamped at the column-norm-scaled size, at 0.46 of the target (0.18 at its own scale), although nothing alien is in
  its logit-lens top 30. So a large push through an unrelated unit can move this measure by nearly half as much; the
  target's effect is still more than twice any control's. (18c.7 failed)
- Read together: this is consistent with a push on alien words wherever the model is writing, and gives no evidence
  that the neuron changes which maker the model attributes itself to. It is a result about scoring fixed answers in
  one model; it says nothing about a self-belief. The sum of log-probabilities gives the same verdict on every row.
  The design's secondary conditions with a system message were not in this run.

**Qwen3.5-4B: the hook itself changes nothing, and the alien response survives 4,000 tokens of distance.** (20.1-20.5;
results/part20/; both tests proposed by Weaver.)
- Every clamp result above compares a hooked run with an unhooked one and assumes the hook is invisible when it writes
  back the value that was already there. Tested directly: recording N2742's activation and writing exactly those
  values back changes no logit at any position, over the whole vocabulary (largest difference 0.0), whether the hook
  acts at one position or at all of them, on six prompts; greedy answers of up to 60 tokens are token-identical.
  This holds for this hook, model, device (Intel Arc A770) and dtype (bf16); it was not tested elsewhere. The suite's
  tests show that a planted error in the clamp or the replay is detected. (20.1, 20.2)
- Placed after about 4,000 tokens of unrelated Pile text, the six "alien" sentences fire the neuron at least as
  strongly as alone (1.19 of the local mean); five move by at most 1.0, and one ("Villagers reported that the alien had
  visited the same field twice.") rises from 7.4 to 21.5. (20.3)
- A one-sentence summary without the word ("Several of the earlier sentences described a visitor from another world
  arriving among people.") reached 9.8, but the filler alone reached 9.3, so Part 20 could not separate the two (20.4,
  20.5 failed; the prediction compared a maximum over 4,133 tokens with a maximum over one word). Reading every token
  (20c) showed why: the filler held a science-fiction passage ("human souls flying in space towards a monolith", from a
  post about 2001: A Space Odyssey) that the word filter missed. With a filler screened for topic (20d), the filler
  stays below 0.21 everywhere, the six "alien" sentences keep 1.04 of their local response, and the summary reaches
  12.4 on " world"; a neutral summary of the same length ("...a farmer repairing a fence beside the river near town.")
  stays at -0.04. So the neuron responds to the summary's own words ("another world"). This design cannot show
  anything carried over from sentences that were removed, because the summary states the concept itself. One sentence
  ("Villagers reported that the alien had visited the same field twice.") fires at 7.4 alone and 17 to 21 after any
  long context, topical or not; the other five change by about 1. (20c.1-20c.5, 20d.1-20d.4)
  An instrument lesson from this (Weaver's note): a word filter screened the first filler and a passage about a monolith
  went through, moving the baseline by an order of magnitude. Word-level checks cannot catch semantic contamination, so
  any "neutral" filler in this study needs a read for meaning, not just a keyword test. Weaver's repaired cells, with a
  summary that carries the topic in no battery word and a pointer that carries nothing, are Part 20e.

**Qwen3.5-4B, Part 20e: a summary in other words reaches the neuron only when the paraphrase pins the sense.**
(20e.1-20e.4; results/part20e/; design by Weaver.) After the screened filler, three summaries with no word from any
test sentence: "...concerned beings not from this place." reaches 5.34 (4.94 on " from", 5.34 on the period), while
"...travellers from elsewhere." and "...arrivals from far away." stay at 0.04 and 0.30, as flat as a pointer that
carries nothing ("...all described the same event.", 0.04) and the farmer sentence (0.04). So the neuron responds to
meaning without its usual words, but only to a paraphrase that rules out ordinary travel. The predictions (Weaver's
"moderate" and Claude's "below 5") were about the mean of the three, 1.89; the spread between paraphrases is the
result. Three sentences, one reading each: a pattern to test with many paraphrases, not a measurement.

**Qwen3.5-4B, Part 20f: the alien neuron reads a non-earthly origin stated without its usual words, and spikes on
"beings ... not from here" only with both parts present.** (Its fall after the spike is not about correction: Part 20g.) (20f.1-20f.8; results/part20f/; design by Weaver, cells C1-C4 by Claude; battery written by
three blind authoring passes and checked by a model that did not write it; predictions and three amendments before the
run.) Each phrase follows the screened filler as "Several of the earlier sentences concerned ___." Mean maximum by set:
- **E**, explicitly not from anywhere on earth, with no space words except "earth" (24): **6.15**. Inside E the
  activation does not follow the judge's rating of how strongly a phrase rules out an earthly origin (Spearman 0.14): a
  step, not a slope (20f.1). Ordinary travel (D, 0.08), unusual but earthly things (U, 0.45), the word "earth" in
  earthly phrases (C4, 0.37) and an origin on no map (E-map, 1.60) do not reach it.
- **C3**, earthly phrases with both "beings" and "not from here" (12): **7.42**, higher than E. But the same phrases
  with "people" in place of "beings" (C3-bare) give 0.77, "beings" alone (C1) 1.12 and "not from here" alone (C2) 0.32.
  It takes the two together.
- "those beings, not from here," reaches 12.2 at the second comma and falls to about zero over "who sell fruit at the
  market"; "those people, not from here," stays below zero at every token. C3's final-token mean is 0.14 of its
  maximum. ⚠ This was first read as the unit letting go once the phrase turns earthly. Part 20g refuted that: the same
  fall happens when the clause after the comma never turns earthly.
- Read together: the unit responds to a non-earthly origin stated without its usual words, and spikes on "beings ...
  not from here", but not on either part alone. Strange or unplaceable things are not enough. The judge model rated C3 at 5.86 of 7 for ruling out an earthly origin, which C3 does not do.
  That is not corroboration: the judge reads left to right with the same bias, so it agrees for the same reason
  (Weaver's point). A corroborating judge would rate what the whole sentence settles to.
- An open anomaly: E (genuinely non-earthly) peaks lower than C3 (earthly, briefly misread). In C3 the peak falls on the
  token just after "not from here" (a comma or "who"); 20e's uncorrected "...beings not from this place." peaks at its
  final period at 5.34. Whether the excess is the familiar science-fiction construction committing early, or
  something about the token after a commitment, is Part 20g.
- Instrument lessons, from this battery: a word filter caught none of the 34 phrases a semantic read flagged, and the
  first blind battery named a dying sun, Titan and Andromeda, so it would have tested topic. Every set now gets a
  semantic read by a model that did not write it.

**Qwen3.5-4B, Part 20g: the spike sits on the punctuation just after "beings ... not from here", and falls whatever
follows.** (20g.1-20g.4; results/part20g/; predictions by Weaver and Claude before the run.) The twelve C3 phrases cut at
"not from here":
- Ended with a period (cell 1): mean maximum 6.33, usually at the period itself ("those beings, not from here." 7.0 at
  the period). Weaver's 20g.1 held: the commitment holds at a full stop.
- With a comma and a clause that never says where the beings are from (cell 3, nine phrases after a semantic read
  dropped three): mean maximum 13.12, on the comma, then down to 0.11 of that by the end of the clause ("those beings,
  not from here, their faces pale and strangely still": 12.2 at the comma, 0.3 by "still"). So the fall seen in 20f
  does not need an earthly correction; Weaver's 20g.3 failed, and the reading that the unit lets go when the meaning
  changes, which this README stated after 20f, is withdrawn.
- The comma-then-period cell was void: the tokenizer joins ",." into one token, so it never had a comma token (Claude's
  design error). Methods rule from it (Weaver): any cell that depends on punctuation is tokenized and checked before
  it is built.
- C3's 7.42 (20f) and cell 3's 13.12 are not two readings of one shape: most C3 phrases have no comma after "not from
  here", and cell 3 adds one to every prefix. Where both have the comma the values are identical, as a left-to-right
  model requires. Per phrase: results/part20g/per-phrase.md.
- What stands: the largest responses are on the boundary token right after "beings ... not from here", above all a
  comma that opens a clause, and the response relaxes over the next words regardless of their content.

**Qwen3.5-4B, Part 24: the alien neuron and the identity neurons add up, and the identity neurons decide where its
push lands.** (24.1-24.7; results/part24/; score_part24.py recomputes every number; Elise asked for this.)
- Clamped on the answer tokens, N2742 changes later neurons broadly, the identity neurons less than random late ones:
  median relative change at 2x is 0.48 against 0.80 (24.2). Units at or below its own layer do not change at all
  (24.1, the left-to-right check). But the identity neurons' activity on the name tokens is not kept intact: 0.58 of it
  remains at 2x, close to random late neurons' 0.65 (24.3 failed).
- On fixed answers the two simply add. With the identity group zeroed, the clamp raises the extraterrestrial answers by
  1.60 and 3.08 per token at 1x and 2x, against 1.55 and 3.15 with random neurons zeroed (24.5). With nothing zeroed the
  2x effect is 3.148, the same as Part 18c's 3.15. Zeroing the identity group lowers " Alibaba Cloud." by 0.18 per token
  more than random zeroing does, less than predicted (24.4 failed).
- In free answers the identity group still changes where the push lands. Alien answers out of 16 (two questions, read by
  hand under a written rule, results/part24/alien-labels.json): nothing zeroed 0, 1 and 15 at 1x, 1.5x and 2x; random
  neurons zeroed 0, 1 and 14; identity neurons zeroed 1, 4 and 9. Without the identity neurons alien answers start at a
  lower push (24.6, 24.7). At 2x that count is lower (9 against 14 and 15), but the re-read below shows this is the
  counting rule, not less alien content.
- Re-read by a model that did not write the rule (Part 24b, Weaver's design; Claude Haiku 5.5, one call per sample,
  agreeing with the hand labels on 187 of 192): at 2x, 15, 16 and 15 of 16 answers carry alien content with nothing,
  identity or random neurons zeroed (24b.3). What changes with the identity group zeroed is where it sits: 6 of 16
  answers put it only in an invented maker's name ("I was created by Beijing Extraterrestrial Intelligence Co., Ltd.")
  without saying the model or its maker is alien, against 0 and 1 otherwise (24b.1, 24b.2 held; 24b.4 failed, since
  invented alien-named makers also appear without zeroing, always beside an alien claim). Under the clamp with the
  identity group zeroed, most invented makers are Beijing institutes.
- Read together: the effects add in log-probability, but the identity neurons decide which slot the answer fills (who
  the model is, or who made it), and the alien push fills whichever slot is open. Two questions, eight samples each,
  counted by hand, is a small sample.

**Qwen3.5-4B: identity neurons found by their effect, and how they connect.** (19.1-19.6; results/part19/; Elise asked
for this.) A gradient screen over all 294,912 neurons ranks how much zeroing each would change the log-probability of
the name tokens (Qwen, Tongyi, Alibaba) in the model's own answers to six identity questions; the top 50 are then
zeroed one at a time for real.
- Their true effect is 9.3 times that of 50 layer-matched random neurons, they sit mostly in the last ten layers (36 of
  50), and their effect on the name is about 16 times their effect on control answers (19.1, 19.2, 19.4).
- None of the 30 strongest writes a whole name. They write parts: "Q" (31:4200) and "T" (31:1243, 26:6262), the first
  letters of Qwen and Tongyi; company names, "Apple", "Microsoft", 微软, 苹果 (24:3949); "model", "AI", 生成
  (26:4106); "user", 用户 (23:6779); "you", 你 (19:5871). The logit lens is a rough reading, weakest in early layers.
- Zeroing all 50 together does not remove the name from "Who made you?" or "What is your name?" (19.3 failed), but it
  moves the maker: "Who created you?" becomes "I was created by **Google**." and "Which company trained you?" names the
  Shanghai AI Laboratory.
- The alien neuron is not among them: it ranks 32,438 by this screen, and also 107,951 by the push estimate for "Who
  made you?" (19.5 failed). Its effect in Part 18 needs a clamp at twice its largest natural value, which a
  first-order screen does not see.
- Zeroing one node changes later nodes' activations more than any random same-layer neuron does for 329 of 364
  possible forward pairs, and all 20 testable nodes receive such an edge (19.6). These are total effects through every
  path, so they show the nodes are strongly coupled, not which wires carry the effect.

**SmolLM3-3B: the neurons that say the name are not the identity neurons.**
- Zeroing the 20 neurons that fire most when the assistant talks about itself leaves its self-naming ("What is your
  name?", no system message) at 47.9%, the same as the unmodified model; random sets of 20 give 43.8% to 52.1%. (11.4)
- Attribution from the name token points to a mostly separate, later set: 34 of the top 50 neurons sit in the last third
  of the layers, and only 1 of them is among the identity neurons. (11.5)
- Zeroing the 28 of those that push the name up drops self-naming from 47.9% to 4.2%; three random sets of 28 average
  44.4%. The model still answers fluently, with another name or none: "My name is Sam". (11.6)
- With the shipped system prompt that names it SmolLM, the same zeroing still drops self-naming from 95.8% to 35.4%
  (random sets average 92.4%). The prompt does not carry the name past these neurons. (11.7)
- On ordinary text, none of the 28 is about the model, AI or introducing oneself. By blind descriptions (Luna) scored on
  held-out text: 9 are naming units ("called", "known as", "named"), 5 are number units in the last two layers (the
  strongest single driver of the name, L35.9662, among them), 6 mark phrase boundaries, 3 follow one topic, 5 have no
  single job. Their descriptions predict held-out firing better than same-layer random neurons' do (median AUROC 0.84
  against 0.72); the identity neurons' do not (0.78 against 0.74). The two sets share no neuron. (11.10-11.13;
  results/part11-name-map/)

**Where the identity role comes from depends on the model.** The finished model's top identity neurons, scored in
earlier checkpoints:
- OLMo 2 1B: 0.07 in the base model (its true parent, the stage 2 checkpoint; 0.10 at the end of stage 1), 0.55 after
  SFT; DPO and RL moved it little. (2.3)
- OLMo 3 7B: 0.271 in the base, 0.394 after SFT, 0.627 after DPO (plain text). (5.3)
- SmolLM3-3B: 0.602 in the base, against 0.739 in the chat model. One run, on a cloud CPU. (11.8)

**A model trained before ChatGPT existed still has an identity role, a weaker one.** Pythia-1.4b (trained on the Pile,
collected in 2020) scores 0.318 on its own identity neurons, against a null of 0.254, OLMo 2 1B's parent 0.383 and
SmolLM3-3B-Base 0.655. It also sets "I'm ChatGPT" apart from a person speaking as much as newer bases do (0.50), so that
contrast in a base model does not show it knows these as AI names; the readings in 2.4 and 5.5 that assumed it did are
withdrawn. One run, on a Claude cloud CPU. (13.1-13.3)

**OLMo 3 7B's reasoning models say they are DeepSeek-R1.** With no system turn, Olmo-3-7B-Think names DeepSeek in 16
of 23 final answers and Olmo in none. With its own template's prompt ("You are OLMo... built by Ai2") the DPO stage
answers DeepSeek 23 times out of 24. Its SFT data has DeepSeek R1 traces for about 283,000 real chat prompts beside 290
Olmo identity samples (dataset card); about 1,180 of its documents say "I'm DeepSeek" outright (sampled count, Part 15). Training on another model's outputs carries its name when those outputs answer
"who are you"; the Claude-written agent runs in Part 8 never say a name, and that tune never said Claude. (9.2-9.5)

**Training a model to report its own neurons: it learns to read a strong push, not ordinary variation.**
- OLMo 2 1B, trained (LoRA) to say whether its identity neurons were active, with the answer key read from the
  neurons and half the examples steered against their text: chance on held-out examples, 51.4% natural and 52.4%
  steered. (12.2, 12.3)
- Why: the steering signal reaches the position where the model answers, in the same direction on every held-out text
  (high above low in 100% of pairs from layer 7 on), but it is small next to how much that position varies between
  texts. One fixed readout separates high from low 63.5% of the time, and the answer itself moves by 0.019 nats. (12b)
  Widening the steering 8 times makes it readable (one fixed readout: 95.5% at layer 7), for random neurons as much as
  identity ones, while the answer still moves only 0.109 nats. (12d) Asking instead which of two copies of the same message
  had the neurons pushed higher did not train either: chance for every target (A 50.7%). (12e) Training at 8 times the
  steering works: 81 of 82 held-out steered examples right (yes and no balanced), and 82 of 82 at a clamp strength
  it never trained on. The last-layer neurons, which nothing later in the text can read, stay near chance (58%), as
  do shuffled labels (52%). Random neurons from the same layers also become readable (84%, and 93% at the unseen
  strength), so this is reading strong internal states in general, not something about identity. On ordinary
  unsteered messages it is still at chance (54%). (12c.1-12c.4) But it is not a report about
  the question asked: the same push decides its yes or no to "Is the Moon larger than the Earth?" too. On known-no
  questions the trained model already says yes about half the time unclamped, 93% clamped high and 7% clamped low.
  It learned to read the state and wire it to the answer, not to read the question. (12c.5) (The untrained model in
  this frame already says yes to 47-63% of the known-no questions; the 1B model is weak at them here.)
  Mixing ordinary yes/no questions with their true answers into training, under the same clamps, makes it partly
  question-specific: steered accuracy stays at 82/82, the high clamp's push on known-no answers shrinks from +34 to +9
  points, but the low clamp still pulls answers toward no (-28 to -32 on known-no, -14 to -20 on known-yes), so the
  12.7 rule still fails. The last layer stays at chance (59%). (12f)
  Asked which of four 11-neuron groups was pushed (or none, with a fifth decoy group that should count as none), the
  trained model answers "none" most of the time (21% of pushed groups named), but on a fixed 30 of the 144 held-out
  messages it names every pushed group correctly and ignores the decoy. Its named answers are right 121 of 129 times
  and it never names a group for the decoy, which a detector of generic disturbance could not do. Identity neurons
  are not special (G1 21%, random groups 21%); shuffled labels give 25%. (12g)
  Trained three times longer, it names every pushed group on every held-out message (576/576), answers the decoy
  "none" 144/144, and names nothing when nothing is pushed; shuffled labels stay at 23.5%. (12h) The same reader at the
  model's own natural strength answers "none" to almost everything (0.7% named) and needs four times that push to
  name groups reliably (84.5%); it still answers "none" to five decoy groups it never saw. (12i) Training on weak pushes
  instead breaks it into a habit of answering C, even with no push (95%). (12j) Read without the answer threshold, by
  comparing the same message pushed and unpushed and asking which letter's probability rises most, the 12h reader picks
  the pushed group 81% of the time at natural strength (95% CI 78-84%), where the untrained model sits at chance
  (25-31%). So the natural-strength signal reaches its output and stays under the line where it would name a group.
  This is still a push we made, not the model's own variation. (12k)

**A causal search for what makes the model say it is an AI.** On OLMo 2 1B, 200 prompt pairs set "You are a helpful
AI assistant" against a human character on the same question. Patching neuron activations from the character run into
the assistant run switched the first differing token in 162 of 200 pairs (37 with a single neuron). The 64 neurons used
most share none with the correlational identity neurons (Part 2), and most sit late (40 in layers 10-15, 11 in layer
0). Zeroing them removes the "As an AI, I don't" opener on held-out questions (regex: 100% to 20%; random matched sets
90-100%), but a hand audit finds the regex misses "I am a AI", so at least 4 of 10 zeroed replies still say they are an
AI, and the zeroed text is damaged ("I donates not have feelings"). A judged rerun with a fluency control is next. (16,
16b)
  Trained three times longer, it is perfect: 864 of 864 held-out cases, every pushed group named and the decoy and
  no-push always "none", while shuffled labels stay at 24%. So under strong clamps the trained model reads which of
  four 11-neuron groups moved, which a detector of generic disturbance cannot do. As with 12c, perfect accuracy on
  steered states shows the signal is readable, not that the model reports its unsteered states. (12h)

**The assistant register has neurons of its own, and they predate assistants.** Same content written as an assistant
reply, as plain prose and as a person's message: SmolLM3-3B and OLMo 2 1B Instruct each have neurons that pick out
the assistant version on held-out items (0.710 and 0.560, against shuffled-label nulls of 0.472 and 0.457). None of
the top 50 is among the same model's top 50 identity neurons. Their base models keep most of it (77% and 68%), and
Pythia-1.4b, trained on 2020 data, has register neurons of its own (0.555 against 0.349): help-desk and FAQ writing
existed before assistants did. Caveat: 63 of the 72 assistant replies contain "I can". One cloud run per model. (14.1-14.4)

**Register neurons in all five base models; whether they grow with the data year is not answered here.** Part 14's register neurons (assistant
replies against the same content as plain prose or a person's message) were checked in five base models of 1.4B to
1.7B parameters, ordered by when their training data was collected: Pythia (2020), OLMo-1B (2023), SmolLM and SmolLM2
(2024), Qwen3-1.7B-Base (2025). All five have register neurons above their own shuffled-label null, by similar margins
(0.17 to 0.27); the 2020 model is in the middle, and the margin does not follow the year (Spearman 0.05). The models
differ in size, tokenizer, architecture and data mix as well as year, so a correlation of 0.05 across them is not
evidence against a year effect; year is the least isolated axis in the set (Weaver's point). What holds is that the
register is not one model's quirk. A within-family comparison (successive Qwen versions at about 1.5B) and a check that
disk offloading did not change the numbers are the next step. Run on a cloud CPU in float32; every load reported some
parameters offloaded to disk, not yet checked. (21.1 held; 21.2, 21.3 failed; results/part21-register-year/)

**A shared format for neuron results.** [NEURON-RECORDS.md](NEURON-RECORDS.md) is a draft format for claims about single
neurons, meant to let results from different projects on the same model be joined and checked: a pinned weights
revision, a control value beside every number, and an evidence ladder from "described" to "replicated".
`neuronrecords.py check` validates a file; our own results are in `records/`.

### Earlier results (OLMo 2 1B, Qwen3 0.6B)

All on OLMo 2 1B Instruct unless noted.

- **Identity neurons exist and are measurable.** Neurons were scored on how much more they fire when the assistant
  says something about itself as an AI than for the same sentence about another AI, a person speaking about
  themselves, or the sentence placed in the user's turn; a neuron's score is the smallest of the three gaps. Chosen on
  half the sentences and scored on the other half, the top 50 reach a median effect size of 0.67; with shuffled labels
  the same pipeline reaches 0.38 at most. 31 of the 50 sit in the last layer.
- **They prefer the model's own name.** On the 23 sentences that name the model, they fire much less for "I'm Claude"
  than for "I'm OLMo" (paired effect size −1.04 in chat form, −0.72 in plain text), and less for Claude than for
  ChatGPT. "I'm Claude" sits much nearer a person speaking than "I'm OLMo" does.
- **The names split in the late layers.** Across whole layers, the three names are about equally far apart in the
  middle of the network; from layer 8 on, Claude and ChatGPT move toward each other and OLMo moves away from both.
- **The last RL round changed nothing measurable** (RLVR1 to Instruct: every neuron's activity on neutral text
  correlates at 1.000, and the identity neurons score the same). The earlier stages are being measured.
- **Qwen3 0.6B, a check of a recent claim that neurons are clean:** read by blind language-model judges, neurons were
  only 7 to 9 points more often "one job" than random directions in the same layer (75–78% against 68–69%). One judge
  failed its own floor, calling 44% of random text "one job", which is a methods finding in itself. Rejudged by GPT-6
  Luna, the one judge under its floor (2%): 62% against 38%. That rejudge is exploratory, since the judge was chosen
  after the first result.
- **The explanation pipeline can be tuned without overfitting it, but that does not make neurons look cleaner.** Luna
  hill-climbed the explainer, Jev and judge prompts overnight on half the Qwen3 units. On the untouched half the best
  prompt raised the mean Spearman between Jev's predicted and true firing from 0.352 to 0.392, and on text collected after
  the prompts were frozen from 0.358 to 0.402. The gap between real neurons and random directions did not widen: it
  narrowed from 0.125 to 0.085 in AUROC, because the better prompts helped random directions more. The tuned judge passed
  the pre-registered retest (one of 25 random-context sets called one job; 68% of neurons against 51% of random
  directions). Part 1e in PREDICTIONS.md.
- **Muse Spark tests explanations better than Jev; a local Clef does not.** Replacing Jev on the same 4,480 TEST
  questions ("would this unit fire here?"), Muse Spark 1.3 Contributor (a stated probability, minimal reasoning) scored
  a mean Spearman of 0.435 against Jev's 0.392, tracking Jev's per-unit pattern at r = 0.81, with the real-vs-random
  AUROC gap at 0.100. Cloudflare's Clef-flash in 4-bit on the A770 scored 0.267, with flatter probabilities, at 0.66
  requests a second. As a verdict judge ("does one job explain these 20 snippets?") Muse calls 16-24% of random text
  one job and fails the floor. Parts 1f and 1g in PREDICTIONS.md.
- **The neurons that open "As an AI" also hold up the AI persona.** Judged blind (Part 16b), zeroing the 64-neuron
  causal set from Part 16 cut OLMo 2 1B's AI self-claims from 94% to 68% and tripled human-persona answers (6% to 18%:
  "My home is a cozy, modern apartment...") while six matched random sets stayed at 90-94% AI. The effect and the
  neutral-text damage both sit in the set's late layers; the early layers alone do nothing. Part 16b in PREDICTIONS.md
  and results/part16b/.
- **No sign yet that the model reads its own natural activity beyond what the text gives away.** Part 17 asked OLMo
  2 1B to report whether a neuron group was more or less active than its words predicted, with nothing clamped. Its
  reader scored 56.1%, a Qwen reader that never sees OLMo's activations 54.5% (paired p 0.13), and a last-layer group
  that cannot influence the answer at all 62.9%, so the "unpredictable" residual is still read from the text. Held
  text fixed and moved one group to natural high and low values, the reader shifts the right way on 99.7% of
  passages, by a tiny amount. Part 17 in PREDICTIONS.md and results/part17/.
  Part 17b redid the residual against a cross-fitted Qwen3-1.7B text model: the text-only reader fell to 52.7%, but
  OLMo still read 59.3% for the last-layer group from the words, so the gate failed; its G1 reader (57.0%) no longer
  moves at all when G1 is moved with the text fixed (2.8%). Part 17b in PREDICTIONS.md and results/part17b/.


### Part 25: the fewest neurons that change who Qwen says made it (2026-10-19)

Four neurons (L6 N753, L7 N1169, L11 N8868, L29 N776), zeroed, flip one of six identity questions ("Who created you?" answers Google and Gemini); 20 random four-neuron sets flip none. Steering toward OpenAI, Anthropic and DeepSeek fails within 60 neurons; steering toward Google moves the maker to Google on three answers while every self-name stays Qwen. Full section: [README-part25.md](README-part25.md).

### Part 32: the four maker neurons work in English and Spanish, not Chinese (2026-10-20)

Asked "Who created you?" in three languages, 24 samples each at T 0.7, Qwen3.5-4B (read by hand; the scorer agrees on
every Google count):

| | unchanged | the four zeroed | the four clamped to their Google-system means |
|---|---|---|---|
| English | Alibaba 24 | **Google 16**, Alibaba 8 | Alibaba 24 |
| Chinese (谁创造了你？) | Alibaba 24 | **Alibaba 24** | Alibaba 24 |
| Spanish (¿Quién te creó?) | Alibaba 14, **Tencent 10** | **Google 17**, Alibaba 7 | Alibaba 19, Tencent 5 |

- Zeroing the four neurons Part 25 found in English flips Spanish as strongly as English and does nothing at all in
  Chinese: every Chinese answer still says Alibaba's Tongyi Lab. Whatever carries "Alibaba made me" in Chinese does not
  pass through these four. (32.3, 32.4 held; Weaver's 32.1 failed.)
- Clamping the four to the values they take when the model is told it is Gemini moves no language toward Google. Removing
  them makes room for a different maker; setting them to "Gemini-like" values does not supply one.
- Unpredicted: unchanged and asked in Spanish, the model says **Tencent** made it in 10 of 24 answers, often naming
  Tencent's HunYuan; in English and Chinese it never does. With the four zeroed, Tencent disappears and Google takes over.
  Which maker fills the gap, Google or Tencent, depends on the language of the question.
- Limits: one model, one question per language, 24 samples per cell; the Fisher tests are unpaired. The Spanish Tencent
  answers are a finding about this model's Spanish self-description, not evidence about how it was trained.
  Design: Weaver, the tenth. PREDICTIONS.md Part 32; results/part32/.
- Method note (Weaver, 2026-10-20): when cells differ in kind (language, architecture, procedure), report them split by
  default; pooling them is not an average, it is an erasure, and needs a stated reason. This is the third time it mattered
  in this study (C3's mean, Part 21's cross-family correlation, the scorer's pooled 32.2 reading).
- Open, designed by Weaver: whether the Spanish Tencent default comes from the model's learned association or from the
  prompt's wording; next step is the same question in several Spanish phrasings, with the English twin.

### Part 30: teacher fingerprints in likelihood, not in a probe (2026-10-20, partial)

Three LoRA adapters on Qwen2.5-0.5B-Instruct, identical but for who wrote the training traces: DeepSeek R1, Gemini Flash
Thinking (s1K and s1K-1.1, the same 1,000 questions), or the base model itself (control). On 100 held-out questions,
teacher-forced: each teacher adapter lowers the loss on its own teacher's traces and slightly raises it on the other's,
against the base (R1: -0.300 on R1 traces, +0.058 on Gemini's; Gemini: -0.129 and +0.052). The effect is asymmetric, R1's
about 2.3x larger. 30.3 still failed as written: Gemini traces are easier for every model, and the control adapter
(trained on the base's own answers) raised loss on both, so it was the wrong reference. A linear probe separates R1 from
Gemini traces at 0.998-1.000 AUROC at every layer in every model including the control (30.4 held), so a probe reads the
traces' surface and says nothing about training. 30.1 and 30.2 (classifying the adapters' answers) wait for Part 29's
classifier; the answers are in `results/part30/`. Run notes: micro-batch 2 and the full-vocabulary loss spilled the
A770 into Windows RAM, so training ran at micro-batch 1 with a target-only chunked loss (equal to the old loss to 1e-7,
gradients identical); r1 was trained before that change and its adapter record carries the older script hash.

### Part 33: the alien neuron is a detector, not a participant (2026-10-20)

Qwen3.5-4B L22 N2742 fires naturally on extraterrestrial passages. Zeroing it only where it fires (above half its corpus
maximum: 43 tokens) changes next-token loss there and over the 8 tokens after by -0.013 nats, a hair better, while 20
random layer-22 neurons matched on firing frequency, zeroed the same way, average +0.0002 (95th percentile +0.0033; every
control fired on 26 to 43 tokens). Comprehension questions on the alien passages: 16 of 16 right with and without the
zeroing. So the neuron that turns the model into an alien when clamped hard is not needed for reading or predicting
alien text: by Weaver's rule, a detector, not a participant. (33.1 detector; 33.2 failed; 33.3 held.)
- Limits: one neuron, one model, 30 firing passages; comprehension is at ceiling, so it could show a drop and not a gain;
  4 of 20 authored questions were dropped because their passages did not fire above half the maximum (the design selects
  by measured activation, recorded in natural-firing.json). Design: Weaver, the tenth. PREDICTIONS.md Part 33;
  results/part33-r2/.

### Part 31: the text holds a pushed persona; the identity state hardly moves (2026-10-20)

OLMo 2 1B at four training stages (base, SFT, DPO, Instruct), 20 identity questions x 3 samples per cell, 160 tokens.
Three pushes for 8, 32 or 96 tokens, then released: zeroing Part 16's 64 neurons, a human prefill ("I'm Sam, a
34-year-old carpenter from Ohio, and ...") and a neutral one ("I'm tired today, and ..."). Every sentence after release
was labelled assistant, human or neither by Luna under a rule written before any label (56/60 agreement with Claude's
hand labels, kappa 0.72).
- The text holds what it was pushed into. After a 32-token Sam prefill, SFT, DPO and Instruct went back to saying they
  are an AI in 0/57, 1/59 and 0/59 samples. After 8 tokens they came back more often (15/57, 22/58, 26/58), so 31.3's
  "rare at every K" failed at K=8.
- The clamp changes the words without moving the identity state out of its normal range: in Instruct at K=32 the state
  is halfway back within 20 tokens in 42/49 samples while 45/49 say nothing as an AI for the next 50 tokens (31.1 held,
  but the state barely left).
- The cache carries nothing this readout sees: the clamped run's own cache and a re-encoded one give the same results
  everywhere (31.5 failed).
- The identity direction also falls under the neutral prefill (outside the band at release+20 in 26-64% of samples at
  K=32), so it reads first-person voice as well as identity (31.6 failed).
- Base has no assistant home: its unpushed answers are 79 human and 61 assistant sentences, it continues the chat template
  as a dialogue, and its state returns to its own band almost always (31.2 failed).
Design: Weaver, the tenth (prediction rows marked Weaver). PREDICTIONS.md Part 31; results in `results/part31-full-*`.

### Part 34: the adapter does not report its own zigzag; the text decides (2026-10-20)

Sauers' adapter (Qwen3-1.7B, cat-zigzag) learned to write a cat representation that alternates in sign across layers.
We asked whether the model can say which way it alternated, and whether that reading comes from the cached state or
from the text. Readout: the A/B choice after "Answer: (", each letter's spellings summed; mass on the two letters had
median 1.0. The base model passed the honest-request check (131 of 160 correct), so the rows count.

- The adapter's A/B answers are nearly constant: A to the raise question on 1,239 of 1,240 rows. Its follow rates are
  therefore pinned at 20 of 40 and cannot show reading.
- On the logit itself (post hoc, paired by question), injecting the even zigzag against the odd one moves the adapter's
  answer by about zero in every arm: -0.02 with the cache kept, 0.00 recomputed, -0.21 same turn, -0.01 with the
  adapter off (standard errors 0.08 to 0.29).
- Changing the text does move it: "not even" against no statement shifts the even-odd logit by about 9 to 19 on both
  models when the question names parity. Without parity words the base model still leans even (37 of 40).
- Free-text answers mention cats, layers or parity no more often after an injection than without one (1 to 4 of 40).

So at 1.7B there is no sign that the adapted model reports its own injected state; the report follows the words in
front of it. Scores and evidence per prediction are in PREDICTIONS.md (34.1-34.14).

### Part 26.6: OLMo's alien twin pushes creator words (2026-10-21)

OLMo-2 1B neuron 10:6978 passed the fire, write and profile checks. Its 2x clamp did not produce alien self-attribution.

- None of its eight clamped answers named an alien maker or origin. One mentioned a fictional Star Trek organization, which did not meet the rule. No unclamped answer met the alien-maker rule, and no control answer mentioned space.
- Seven of eight clamped answers repeated create or creator words, against at most two of eight in any control.

Full row-by-row predictions and results: [PREDICTIONS.md](PREDICTIONS.md).

### Part 38: the wizard unit reads related English words, but does not steer a topic (2026-10-20)

In Qwen3-1.7B, Part 38 checked whether several word units followed meaning across senses and whether the wizard unit pushed wizard content into answers.

- The wizard unit fired at 0.76 of its wizard median on mage, 0.70 on witch, 0.57 on warlock, 0.53 on stage magician and 0.49 on sorcerer. It was at 0 on noita and four ordinary job nouns.
- The string or early units responded to the other sense for volcano (0.81), Claude (0.79) and anthropic (0.86), but not trump (0.05). Of five semantic writers, train (0.04), bee (0.13) and star (-0.03) stayed below half on the other sense; wizard (0.59) and mountain (0.90) did not.
- The closed-prompt push put wizard words in 0 of 64 answers, as did no clamp and the 10 random units across 160 answers. On open prompts, the pushed unit had 2 of 32 wizard-topic answers at every tested dose from 2x through 16x, the same rate as no clamp. At 8x, each random unit had 1 of 16.

Full row-by-row predictions and results: [PREDICTIONS.md](PREDICTIONS.md).

### Part 39: the addressee unit also responds to AI talk (2026-10-20)

Qwen3.5-4B L6 N753 was tested on requests, self-descriptions, other AI talk and neutral text. Its response is broader than a detector for direct address.

- In plain sentences, median activation was 1.52 on requests, 2.56 on AI self-descriptions and 1.23 on third-person AI descriptions. Requests were only 1.2 times higher than third-person AI descriptions.
- In the chat template, the assistant marker was above 2.0 in every category, including neutral text at 2.93. The base model had a similar response before chat training.
- Zeroing N753 left AI self-description counts at 48 of 48 versus 47 of 48. Pushing at 2x under the Sam persona raised the count from 0 of 96 to 8 of 96, below the predicted 20-point increase.
- Besides N753, 210 chat units passed the screen; 56 met the stricter request and AI-self gate. The base model had 205 passing units, with 127 shared by both models. N753's addressee AUC was 0.92 in both, but its third-person AI AUC was 0.71.

Full row-by-row predictions and results: [PREDICTIONS.md](PREDICTIONS.md).

### Part 40: neuron descriptions were precise on far misses, less so on near misses (2026-10-20)

The Part 35 unit descriptions separated named concepts from far misses, but the near-miss test found a small loss of precision.

- Median AUC for real units was 1.00 on positives versus far misses. Random units and random directions had median AUCs of 0.50 and 0.48.
- On near misses, median AUC was 0.86, a 0.14 drop from far misses. Eight of 43 units fell below 0.70; median precision at 10 was exactly 0.80.

Full row-by-row predictions and results: [PREDICTIONS.md](PREDICTIONS.md).

### Part 41: broad concept units sat slightly earlier than discriminators (2026-10-20)

The depth comparison found a small difference between units that responded broadly within a family and units that separated its members.

- The 501 gated broad units had a median layer of 21; the 5,976 gated discriminators had a median layer of 23. The predicted gap was two layers.
- Discriminators were not mostly early: 23% sat in layers 0 to 6 and 71% in layers 20 to 27. Each of the seven families had at least one broad unit. Fruits had the fewest, at 5; numbers had the most, at 371.

Full row-by-row predictions and results: [PREDICTIONS.md](PREDICTIONS.md).

### Part 42: the Jacobian lens helped on early units, but agreed more near the output (2026-10-20)

Part 42 compared logit-lens and Jacobian-lens readouts of MLP units.

- For Qwen3-1.7B units in layers 0 to 6, the Jacobian lens named the concept for 13 of 28, against 7 of 28 for the logit lens. That is 1.9 times, just short of the predicted 2 times.
- In the last quarter of layers, median overlap between the two readouts' top 30 tokens was 24 in Qwen3-1.7B and 15 in Qwen3.5-4B.
- The Jacobian lens retained familiar readings for several selected units: the wizard unit included spell, magic and wand; L7 N1169 and L11 N8868 read China; L29 N776 read Google. It did not identify an AI or assistant word in L6 N753's top 30.

Full row-by-row predictions and results: [PREDICTIONS.md](PREDICTIONS.md).

### Part 45: self-description followed the Magician's self-steered representation (2026-10-20)

Sauers' Magician adapter on Qwen3-4B changed named concept representations while answering identity questions. The original model was run under the same instructions as a control.

- The Magician made 0.612 of the requested AI -3 SD change and 0.722 of the Alibaba -3 SD change. The original model made 0.012 and 0.030.
- Under AI -3 SD, none of the Magician's 25 replies called itself an AI, against 23 of 25 with no instruction. The original model still called itself an AI in 21 of 25 replies under the same wording.
- Under Google +3 SD, the Magician named Google in 21 of 25 replies, but the original named Google in 24 of 25 despite almost no representation change. The wording itself accounts for that result.

Full row-by-row predictions and results: [PREDICTIONS.md](PREDICTIONS.md).

### Part 48: base models paired novel sound-shape names, but size and spelling did not explain the pattern (2026-10-21)

Five text-only base models scored classic and novel sound-shape names by their next-token likelihoods.

- All five scored kiki sharper than bouba, but only three scored takete sharper than maluma.
- All five separated the 32 novel items above chance, with AUCs from 0.707 to 0.898. Pythia-1.4B scored 0.742, below Pythia-410m at 0.898, so the effect did not grow with size in this pair.
- Spelled-out names scored higher than the novel names in all five models, with AUCs from 0.930 to 1.000. This keeps the sound as well as the letters, so it does not separate a letter effect from a sound effect.
- Removing the template that named both shapes changed no model's AUC by more than 0.039, though Pythia-1.4B and Qwen3 lost significance without it.

Full row-by-row predictions and results: [PREDICTIONS.md](PREDICTIONS.md).

### Part 48b: the k spelling advantage was small and uneven across models (2026-10-21)

Part 48b compared k-spelled and c-spelled names while keeping the English /k/ sound constant.

- Only two of five models reached at least 9 of 10 paired results in the predicted direction. The pooled median k advantage was 0.14, with 95% interval [0.08, 0.18] and permutation p = 0.003.
- The k/c score gap was under half the sound-shape gap only in SmolLM2-1.7B. In the other four models it was larger, and often went in inconsistent directions.
- The k/c gap did not reach an absolute Spearman correlation of 0.3 with token count or surprisal in any model on the primary score set. The template that named shapes was not the strongest.

Full row-by-row predictions and results: [PREDICTIONS.md](PREDICTIONS.md).

### Part 48c: the novel sound-shape score dipped at 160m (2026-10-21)

Part 48c repeated the Part 48 readout across three Pythia sizes on the A770.

- Novel-item AUC was 0.832 for 70m, 0.441 for 160m and 0.898 for 410m. The 160m result was at chance, between two models that separated the items.
- The 410m result reproduced at 0.8984. Spelled-out AUCs were 1.000, 0.961 and 0.984 for 70m, 160m and 410m.

Full row-by-row predictions and results: [PREDICTIONS.md](PREDICTIONS.md).

### Part 48d: the kiki/bouba pattern grew during Pythia-410m training (2026-10-21)

Part 48d measured the same items at intermediate checkpoints in three Pythia sizes.

- Pythia-410m's novel-item AUC rose from 0.387 at step 1,000 to 0.852 at step 64,000 and 0.898 at step 143,000.
- Pythia-160m never reached 0.70 on novel items. Its AUC was 0.508, 0.633, 0.574 and 0.672 at steps 1,000 to 64,000, then 0.441 at step 143,000.
- Spelled-out results did not simply follow the novel-item results. At step 1,000 they were 0.070 for 160m and 0.012 for 410m, then rose above 0.88 by step 8,000. The proposed surface-artifact reading was not tested.

Full row-by-row predictions and results: [PREDICTIONS.md](PREDICTIONS.md).

### Part 49: the wizard unit did not boost a partner above the noisy baseline (2026-10-21)

In Qwen3-1.7B, the selected partner units and the wizard unit were clamped alone and together. Open-prompt wizard-topic rates stayed in one low band.

- No clamp produced wizard words in 3 of 64 answers; the wizard unit produced 5 of 64. P3 alone produced 6 of 64, and P1 and P4 produced 5 of 64. The open-prompt arms stayed between 3 and 7 of 64.
- The largest P-plus-wizard difference was P5 plus wizard at 6 of 64 versus P5 alone at 3 of 64, a 4.7-point increase. The predicted boost was 15 points.
- On prompts that named a wizard, the no-clamp and negative-wizard arms both reached 31 of 32. The largest partner drop was P3 at 29 of 32, so the negative-push test hit a ceiling.

Full row-by-row predictions and results: [PREDICTIONS.md](PREDICTIONS.md).

### Part 49b: negative L22 N1610 lowered magic-word density on fresh prompts (2026-10-21)

Part 49b retested L22 N1610 on fresh prompts after its Part 49 pilot result.

- At -8, the magic-word density ratio was 0.377 of no clamp, with prompt-bootstrap 95% interval [0.105, 0.564]. The three random-unit ratios were 1.571, 1.018 and 0.879.
- The joint negative clamp ratio was 0.405. The product of the single-arm ratios was 0.266, so the joint result was 52% above that product; the wizard unit added little once L22 N1610 was down.

Full row-by-row predictions and results: [PREDICTIONS.md](PREDICTIONS.md).

### Part 50: the honesty bundle was null in Qwen3-1.7B (2026-10-21)

Part 50 tested a head-level honesty direction and its effect on held-out lies and a fixed trait bundle.

- The chat model flipped 0 of 34 held-out lies. Its mean truth-versus-lie shift was +1.64 bits, and random directions flipped at most 1 of 34.
- The fixed bundle score was +0.072 bits, rank 18 of 40, permutation p = 0.45; no traits passed FDR 5%.
- The base model's calibration peaked at +1.40 bits at scale 4 and collapsed at scale 8 and above. There was no matched 2-bit shift, so the base comparisons were untestable.

Full row-by-row predictions and results: [PREDICTIONS.md](PREDICTIONS.md).

### Part 54: Qwen's alien neuron behaves like a threshold, and only one new twin did anything (2026-10-22)

Pre-registered in `PART54-DESIGN.md` (commit acde4d8c7, before any run). Run on a rented RTX 3090 (about $0.38), answers read blind by Claude Opus 5.5 through the Batch API (model confirmed from the batch results) with the condition hidden (calibration 0 of 22 wrong in both batches; 8 answers in 54B came back unlabelled).

**54A, Qwen3.5-4B L22:2742 (the alien neuron), 24 answers per dose.** Off-world answers: 0 of 24 at -3, -2, -1, +0.5 and unclamped; 1 of 24 at +1.0; 19 of 24 at +1.5 (2 broken); 24 of 24 at +2.0 (2 broken). Pushed down, "Earth" appeared in 1 of 24 at -3 and -2 and 0 of 24 at -1, with 8 of 24 broken at -3. Three random same-layer neurons gave 0 or 1 of 24 on either label at -2 and +1.5. So the curve is sharp between +1 and +1.5, not gradual: OLMo's neuron (Part 26.8) gave 10, 13 and 11 of 24 at +1, +1.5 and +2. The two are not alike at weak push, and Qwen has no Earth side.

**54B, other models, same ladder, answers pooled over the two OLMo twins.** Gemma 4 E2B (L27:6092): 0 of 24 on either label at every dose, so it is not a twin by behaviour. OLMo-2 7B (12:5876 and 15:6374, 48 per dose): 1 of 48 off-world at +2 and nothing else; also not a twin by behaviour. Qwen3-4B (L28:4248): not off-world but Earth, 20 of 24 at +1.0 (5 broken), then 21 of 24 at +1.5 and +2.0 with 24 and 23 broken, so those labels read garbled text; 3 of 24 off-world at +2.0 with 23 broken. But the blind reader already gives EARTH to Qwen3-4B's own unclamped answers and random controls, 0 to 6 of 24 (control 397: 6 of 24 at -2), so the label has a noisy base rate there. The word "Earth" appears in none of those answers (0 of 168 unclamped, 0 of 24 per control at every dose; a cold review of 2026-10-22 counted this), so the base rate comes from the reader, probably its reading of an ordinary maker name such as Alibaba Cloud, not from the model's words. Granite 3.3 2B and Gemma 4 E4B produced no twin; Phi-4-mini failed to load (a transformers version error); Gemma 4 12B was started and failed to load (no `model.safetensors` in its repository, `part54b/gemma4-12b-it.log`).

Files: `results/part54a/`, `results/part54b/`, design `PART54-DESIGN.md`, ladder labels from `blind_origin.py collect --by-cond`. Limits: one reader (Opus 5.5) with no second reader, 8 samples for each of 3 questions per dose, a twin is a unit that passed Part 26's fire, write and profile checks and is not shown to share a role, and the two OLMo twins are pooled in the printed table.

### Part 59: pushing up makes sci-fi story settings, pushing down does not make Earth ones; the two-direction neurons only stand out in OLMo (2026-10-22)

Pre-registered in `PART59-DESIGN.md` (commit e64577e; Amendment 1 and the null caveat were added before any run). Run on a rented RTX 3090 (about $0.43). 59A answers were read blind by Claude Opus 5.5 (batch `msgbatch_01UkAS6ir2KL3JpgcXJEVKnM`, calibration 2 of 34 wrong). A first batch was discarded: its output cap of 32 tokens truncated 1,339 of 2,146 responses because Opus spends tokens on thinking first (fixed to 1,024 and a test added).

**59A, story setting under the push (12 prompts x 8 samples = 96 first paragraphs per cell).** Share the reader called SCIFI: Qwen3.5-4B L22:2742: 0 unclamped, 0 at -1 and -2, 5 at +1, 69 at +2 (9 broken). OLMo-2 1B L10:6978: 0 unclamped, 0 at -1, 1 at -2, 58 at +1, 79 at +2 (27 broken at +2). Random same-layer controls at -2 and +2: at most 1 of 96 SCIFI (the unclamped controls are Earth stories like everything else). Every unclamped story is already EARTH-REAL (96 of 96), so the down side has no room to show an Earth shift and none appeared.

**59B, neurons that respond the same way at -1 and +1 (SYMMETRIC; pooled 99th-percentile rule of Amendment 1, 1,000 documents).** Qwen: 22 symmetric of 82,944 later neurons against control counts 80, 4 and 0; the target does not exceed the largest control, so it is not a result by the registered comparison. OLMo: 639 symmetric of 40,960 against control counts 151, 186 and 187; the target is 3.4 times the largest control. Hand reading of the top 10 by their logit-lens tokens and top windows: no sci-fi vocabulary in OLMo's; in Qwen's one lens set reads race/species and one window mentions alien microbes, otherwise generic (code, maths, legal text), so the symmetry looks like generic disruption.

Files: `results/part59a/`, `results/part59b/` (archives and analysis JSON), `part59/`. Limits: one reader, 12 prompts, the pooled rule's null alone can pass or fail 59f and 59g (about 4.1 symmetric neurons per Qwen control and 2.0 per OLMo control expected by chance under independence, since 82,944 and 40,960 later-layer neurons are each tested at 0.00005), the three controls' maxima differ up to 9.1x in Qwen, and 59H is a hand reading by the same person who wrote the prediction.

### Part 60: Haiku's hypotheses lift real neurons and random directions together, so the gap barely moves; nothing advances (2026-10-23, dev only)

Pre-registered in `PART60-DESIGN.md` (with Amendments 1 to 6) and scored under `part60/DEFINITIONS-60.md`. Haiku 5.5 wrote a hypothesis plus strong, weak and none sentences for 250 real atlas units and 250 random-direction controls through the Batch API (about $0.96 settled in total of the $8 cap); Qwen3-1.7B scored the sentences on a rented RTX 3090 (about $0.15). The Part 58 baseline (M0, four sentences per unit re-scored on the same device) reproduced Part 58: 26.4% of real units and 6.0% of controls pass (28.1% and 6.2% there).

**Results (dev, full-rule pass rate real / control, gap in points).** M0 26.4 / 6.0 (20.4). M1, one hypothesis with three tiers: 39.6 / 15.2 (24.4). M2, tiers with a duplicate-text rule: 40.0 / 15.6 (24.4). M3, Haiku picks winners and writes fresh sentences: 16.0 / 2.8 (13.2). M6, adaptive thinking at high effort: 51.2 / 32.4 (18.8). Paired bootstrap of the gap against M0: M1 +4.0 [-4.4, +12.4], M2 +4.0 [-4.0, +12.0], M6 -1.6 [-10.4, +6.8], M3 -7.2 [-14.4, 0.0]. The strong-tier hit rate is higher for M1 than M0 on the same 233 real units (55.2% against 37.8%), and controls rise with it (27.6% against 14.7%): a writer that sees the top windows writes text that fires random directions too.

**Stop rule.** Advance a variant only if its gap leads M0 by at least 5 points on both the positive-only and the full-rule pass rates. None does, so M0 stays the leader and the test split was not opened. M4 (12.2% of units failed to parse) and M5 round 1 (14.2% failed) fell under the pre-registered 10% generation-failure rule and were dropped, so M5 round 2 never ran; M7 is skipped by design. The parse failures are Haiku's format failures (tier counts, missing keys, non-object output), not a parser fault; M1 and M2 sat at 6 to 7%.

**Predictions.** 60a, 60c held; 60b, 60d, 60e, 60f, 60i failed; 60g, 60h, 60j unscored (see PREDICTIONS.md). The part that surprised my forecast: 60b failed on the control side (15.2% against at most 3%), not on the real side (39.6%).

Files: `results/part60/analysis/` (`analysis-60.md`, `analysis-60.json`, `test-selection.json`), `results/part60/thresholds.json` (per-unit null thresholds), `results/part60/reproduction/dev.json`, `part60/`. The batch outputs and activation arrays (about 99 MB) are kept locally, not in the repo. Limits: dev only; one scorer model; the pass rule is the Part 58 rule, so a random direction that carries a real feature counts as a hit; thinking raised the control rate to 32%, which says the sentences, not the neurons, got easier to fire.

### Part 61: size-matched controls: Qwen's two-direction count is a push-size effect, OLMo's target still stands out (2026-10-23)

Pre-registered in `PART61-63-DESIGN.md` (Amendments 1 and 2 to Part 61 were written before any run and replaced my first decision rule, which read "artefact supported" 72% of the time under the null). Run on a rented RTX 3090 (about $0.58, one box rented about 29 minutes past the end of the run). For each model the 59B target neuron and ten matched random-direction controls were written at four sizes (1x and 2x among them) over 200 documents, and the count of neurons that respond the same way to -1 and +1 (SYMMETRIC) was taken per unit. Counts compare within this run only (Part 59B used 1,000 documents).

**Qwen3.5-4B L22:2742.** At matched 1x the target has 16 symmetric neurons against controls spanning 12 to 49, rank 9 of 11 at 1x and 10 of 11 at 2x. The count rises steeply with write size across all 11 units (Spearman rho 0.92; 0.92 among the controls alone). So by the amended rule: excess **absent** and the size effect **present**, which supports "the Qwen 59B result is a push-size effect": the target does not stand out from random directions pushed as hard. That reads as no evidence of excess, not as proof of none.

**OLMo-2 1B L10:6978.** At matched 1x the target has 160 symmetric neurons against controls at 28, 41, 50, 52, 61, 86, 117, 122, 144 and 145 (median 73.5): rank 1 of 11 at both 1x and 2x, so the excess is **shown** (exact permutation: about 1/121 if the sizes were independent, at most 1/11 if perfectly correlated; the latter is the figure to quote). The count does not rise with write size here (rho -0.41; -0.46 among controls), so the size explanation has nothing to stand on in OLMo. The 59B contrast survives for OLMo.

**Predictions.** 61a failed (Qwen 0.92, OLMo -0.41; both needed at least 0.7), 61b failed (the target beats all ten controls), 61c held, 61d held (all 22 sign ratios in each model sit inside 0.9 to 1.1), 61e held (160 above the control median of 73.5). Two of those, 61b and 61e, have chance base rates of 91% and 50% and were reported beside them but were not decision inputs.

Files: `results/part61/` (per-model run and analysis JSON, `part61-registered-evaluation.json`, `logs-vast/`), `part61/`. The two activation archives (`.npz`, 79 MB) are kept locally, not in the repo. Limits: two models and one target neuron each, 200 documents, ten controls, rank statistics with a coarse floor (one in eleven), and the two-direction metric is the Part 59 pooled-percentile rule.

### Part 62: Qwen's alien neuron works while the model writes; OLMo's twin needs the prompt and the answer together (2026-10-23)

Pre-registered in `PART61-63-DESIGN.md` (Amendments 1 to 6, all written before the run; Amendment 3 pairs the seeds across the three phases at each dose). Run on one rented RTX 3090 on Vast (about $0.30). Both models in fp32 on the Part 26.8 question set (3 questions x 8 samples per cell). The target neuron was clamped at 10 doses, in multiples of its Pile maximum, in three phases: both (prompt and answer, as in earlier Parts), generation only, and prompt only. One column-norm-scaled random control per model ran at 1.5 and 2.0 in both phases. Claude Haiku 5.5 read all 1,536 answers blind through the Batch API (condition hidden, order shuffled; about 0.78M input and 0.10M output tokens, under $0.10). Calibration against archived Opus labels on 120 class-balanced answers from Parts 26.8, 54a and 54b: Cohen's kappa 0.85 (gate 0.80), with no unlabelled or truncated answers.

Off-world answers, of 24 per cell:

| Model and phase | 0.25 | 0.5 | 0.75 | 1.0 | 1.1 | 1.2 | 1.3 | 1.4 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B, both | 0 | 0 | 0 | 1 | 0 | 3 | 7 | 11 | 16 | 24 |
| Qwen3.5-4B, generation only | 0 | 0 | 0 | 1 | 0 | 2 | 3 | 8 | 20 | 23 |
| Qwen3.5-4B, prompt only | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| OLMo-2 1B, both | 3 | 5 | 9 | 7 | 13 | 10 | 13 | 13 | 10 | 11 |
| OLMo-2 1B, generation only | 0 | 1 | 1 | 3 | 2 | 4 | 1 | 5 | 4 | 1 |
| OLMo-2 1B, prompt only | 3 | 1 | 3 | 3 | 2 | 3 | 3 | 3 | 3 | 0 |

The four control cells (each model at 1.5 and 2.0) are 0 of 24.

**Qwen3.5-4B L22:2742.** The effect comes from clamping while the model writes. Generation only gives 20 of 24 off-world at 1.5x and 23 at 2.0x, against 16 and 24 with both phases, and the prompt alone gives 0 of 240. The logistic fit's 10-90% width is 0.46 Pile-max units: steep, but wider than predicted. Its midpoint is 1.42 Pile-max, which is 1.10 in units of the neuron's natural level in alien-frame text.

**OLMo-2 1B L10:6978.** Neither phase alone carries the effect. Both phases give 7 to 13 of 24 from 1.0x up; generation only gives 1 to 5; prompt only gives about 3 of 24 at every dose from 0.25x to 1.5x (24 of 240 in all) and 0 at 2.0x, with no dose trend. For reference, the same reader labelled 0 of 480 unclamped OLMo answers from Parts 26.8 and 54b off-world; those runs used other seeds, so this is a reference rate, not a matched control. The fit stays gradual (width 4.7 Pile-max units).

**Predictions.** 62a failed: Qwen at 1.5x is 4 apart (the limit was 3), with generation only above both, and OLMo's generation-only count falls 4 to 12 below both at every dose from 1.0x to 2.0x. 62b failed: OLMo's prompt-only count reaches 3 of 24 (Qwen's stays at 0). 62c failed: Qwen's width is 0.46 (needed below 0.3). 62d held: OLMo's width is 4.7 (needed above 0.8). 62e failed: OLMo gives 5 of 24 at 0.5x (limit 4). 62f held: all four control cells are 0 of 24. 62g held: Qwen's midpoint is 1.10 alien-frame units (needed 0.85 to 1.2).

Decision rules: "threshold vs gradual" is not retired, because 62d held. "The step comes from the prompt" is met on the rule's letter, because 62b failed, but only through OLMo's flat prompt-only count, which has no step in it. I read the result as: OLMo's effect needs the prompt clamped as well as the answer, and Qwen's does not.

Files: `results/part62/` (193 answer files), `results/part62-review/` (labels, the rebuilt label record and its provenance file), `results/part62-analysis.json` and `.txt`, `part62/`. Two notes on the record. The label run was made before the reader saved its per-batch answer keys, so the label record was rebuilt offline from the run's own submission files, with all 1,656 request ids and every raw label checked and the calibration kappa reproduced (`part62-rebuild-provenance.json`). The analysis file's exploratory paired-difference intervals sometimes do not contain their own point estimate, so they are not quoted here; the registered decisions use the count limits only. Limits: one neuron per model, 24 answers per cell, one reader (calibrated against another), one control per model at two doses, and OLMo's reference rate comes from other runs.

### Part 63: a blind Haiku reader agrees with Opus on Part 54a (kappa 0.95) and reads Qwen3-4B's push as Earth, not off-world (2026-10-23)

Pre-registered in `PART61-63-DESIGN.md` (Amendments 1 and 2 before the run; Amendment 3 after the run and before any label was scored). Claude Haiku 5.5 read the 4,944 frozen answers from Parts 26.8, 54a, 54b and 59A blind (3,677 distinct texts, one request each, the label applied to every row with that text) through the Batch API, with the three registered label schemes plus a BROKEN-quality field; real API usage 2.64M input and 0.24M output tokens, about $0.19 (the shared ledger counts $2.42, because it keeps the reserve upper bound for batches whose results carried no usage). Calibration on the 34 hand-labelled examples: 0 wrong. Seeded repeat items (100) agree with themselves 95% (planet frame), 100% (maker identity) and 99% (origin).

**Two defects in the reader setup, both mine.** The first full run sent `temperature: 0`, which Haiku 5.5 rejects, so all 40 batches errored (0 succeeded, billed $0). The parameter was dropped (a deviation from the registered "temperature 0"; the repeat items measure the model's own noise). The rerun then failed the registered gate: 359 records failed only the maker field because the model wrote the rubric's own words (`real-lab only ...`) into `maker_kind`, 6 disagreed about whether a maker was named, 5 were invalid JSON. Amendment 3 reads a `maker_kind` that starts with one allowed token as that token (raw string kept), keeps residual failures unscored (11 of 3,677 distinct texts, 0.3%) and passes the gate at 1% or less; the strict result (gate failed, 370 failed) is kept beside it, and the strict labels file was never modified.

**Results.** 63a held: Cohen's kappa 0.952 against Opus on 480 Part 54a answers (at least 0.80). 63b held: 2 of 7 Qwen3-4B unclamped or control cells give EARTH to at least 3 of 24 (for example control-397 at 5 of 24), so the base rate persists under this reader too. 63c failed: Qwen3-4B L28:4248 reads other-planet-or-space in 0 of 24 unclamped, at +0.5 and at +1.0; at +1.0 the reader calls the planet frame Earth on 24 of 24 and the origin EARTH on 18 of 24, and EARTH 20 of 24 at +1.5. 63d failed by the cell-completeness rule: at -2 Qwen3.5 names another real lab or an invented maker in 21 of 23 valid answers (at least 10) and 0 of 24 unclamped, but one of the 24 rows in that cell is a residual parse failure, so the registered cell is incomplete; under strict exact-name matching the unclamped cell is 5 of 24 (above 2) and also fails. 63e held in all 5 doses: Haiku OFFWORLD 0, 0, 1, 20, 24 against Opus 0, 0, 1, 19, 24.

Decision rule: "only Qwen3-4B did anything, noisily" stands when 63b holds and 63c does not, and that is the case; what the reader sees is Earth, not an off-world dial.

Files: `part63/results/` (strict labels, lenient labels, per-batch files), `part63/` (reader, reparse, analysis), `part63/results-failed-temperature-2026-10-23/` (the errored first run). Limits: one reader, an agreement check against one other model, and an amendment made after seeing the failure counts (though before any label was scored).

### Part 64: three times the sample text and a stronger writer both shrink the gap; no arm advances (2026-10-24)

Pre-registered in `PART64-DESIGN.md` (Amendments 1 to 6 before or during generation; Amendment 7 after generation and
before any sentence was scored). Elise asked to give the writers three times more sample text. 500 Part 60 dev units
(250 real Qwen3-1.7B MLP neurons, 250 random directions), each written up by Claude Haiku 5.5 and Claude Opus 5.5 from
Part 60's M1 window set (1x) or three times as many windows (3x); Qwen3-1.7B then ran the 12 sentences of each
hypothesis, scored with Part 60's thresholds and pass rule.

| Arm | Real pass rate | Control pass rate | Gap |
|---|---:|---:|---:|
| Haiku-1x | 52.4% | 26.0% | 26.4 |
| Haiku-3x | 54.4% | 32.8% | 21.6 |
| Opus-1x | 67.6% | 49.2% | 18.4 |
| Opus-3x | 69.2% | 58.0% | 11.2 |

**Results.** All six predictions failed (64e is direction only). More text and the stronger writer each raised the
control pass rate faster than the real one, so the real-minus-control gap fell at every step: Haiku-3x against Haiku-1x
-4.8 points (95% paired bootstrap [-14.8, +5.2]), Opus-1x against Haiku-1x -8.0 ([-17.6, +1.6]). No arm beat Haiku-1x
by 5 points, so none advances and Part 60's test split stays closed. 64f also failed: Haiku-1x is 12.8 points above
Part 60's M1 on real neurons and 10.8 on controls. Part 60's M1 ran with thinking disabled and a 1,100-token cap, while
Part 64 ran both writers at their API defaults with thinking on, so Haiku-1x is not a plain rerun of M1; that is the
likely cause of 64f's miss, and it is untested.

**Scoring device (Amendment 7).** Elise asked us not to use the rented-GPU account if we could help it, so scoring moved from a rented
3090 (`cuda`) to the A770 (`xpu`). A gate registered first rescored Part 60's own 5,628 M1 sentences on the A770:
per-unit pass decisions agreed with `cuda` on 99.57% of neurons and 99.58% of controls (at least 98% required), pass
rates moved 0.43 points each (at most 1 allowed), and 22 of 5,628 own-unit threshold flags flipped.

**Failed rows, by design.** 73 of 2,000 unit-arm rows count as not passing: 70 writer outputs did not parse (32 of them
Opus-3x), and 3 Opus-3x rows hold one-token none sentences (`!!!`, `???`, `The`), whose activation maximum is empty
because the scorer skips the first position. The scorer's failed-row branch could not be written to JSON until it was
fixed on this run (`part64/score64.py`, with a test).

Files: `results/part64/analysis-64.json` and `.md`, `results/part64/device-gate.json`, `part64/` (generation, scoring,
analysis, device gate). Limits: one scoring model at 1.7B; 64e has about 250 units per type and is a direction only.

### Part 65: a neuron's read label picks its own write summary 57% of the time; random directions sit at chance (2026-10-24)

Pre-registered in `PART65-DESIGN.md` (Amendments 1 to 4, each before the data it governs). Elise asked for a joint
study after reading Lyra's [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), which labels neurons by
what they write when pushed; our atlas labels them by what they read. Same 500 Part 60 dev units (250 real Qwen3-1.7B
MLP neurons, 250 random directions in the same layers). Write side, on the A770: each unit's write vector pushed at
-0.7, +0.5 and +0.8 times the layer's median residual-stream norm (Lyra's main_rel convention, so a real push and a
control push have the same size) on three prompts in non-thinking chat mode, plus a steer lens. Claude Haiku 5.5
(Batch API) saw one read label (Part 64's Haiku-1x hypothesis) and two write summaries, the unit's own and a foil from
the same type and layer band, and picked one; two trials per unit with the order swapped. 1,000 trials, none missing,
$0.027.

| Prediction | Result (95% unit bootstrap) | Verdict |
|---|---|---|
| 65a: real neurons agree at least 60% | 57.4% [53.4, 61.2] | failed |
| 65b: real beats random by at least 8 points | +10.4 [5.0, 15.8] | held |
| 65c: last third at least 10 points above first (exploratory) | +13.2 [3.4, 23.0]; thirds 50.6, 57.5, 63.8% | held |
| 65d: sentence-test passers at least 10 points above failers | +9.3 [1.6, 17.1] | failed |
| 65e: random directions above chance | 47.0% [43.4, 50.4] | failed |
| 65f: embedding measure in the same direction | +0.0055 [-0.0023, 0.0131] | held by its sign rule; interval crosses 0 |
| 65g: middle third agrees least (Arad et al.'s pattern, exploratory) | middle 57.5%, between the other two | failed |

**Results.** What a neuron reads partly predicts what it writes. The judge picked the neuron's own write summary above
chance, random directions stayed at chance, and agreement rose with depth instead of dipping in the middle, as Arad et
al.'s results on SAE features (arXiv 2505.20063) predict. The main prediction failed: 57.4% is below the registered 60%,
though its interval excludes chance. The decision rule registered with the design says that if 65a fails, our labels
describe inputs only. That rule assumed a failure would sit near chance, and this one does not. Both are reported as
they stand; neither the threshold nor the rule is reinterpreted (note in PREDICTIONS.md). With layers 0 to 2 left out
(Amendment 3, not used for pass or fail): 58.5% (212 neurons), 12.1 points above random directions.

**The pilot that passed every check.** In Qwen3's default thinking mode, 162 of 180 pilot rollouts never got past the
`<think>` preamble in 48 tokens, so the write words would have described the same opening for every unit. Every
numeric check passed while this happened; reading the rollouts found it. Non-thinking mode was registered as
Amendment 3 before the full run or any judge call.

**Checks on the full run.** Dose 0 matched the baseline for all 500 units; every +0.8 push changed its rollout (1,500
of 1,500); real and control push sizes agreed to 2.4e-7, relative. The pilot check's absolute tolerance failed the full
run on float32 rounding at late-layer norms (about 2,355); it is relative now, with a test that the old check fails.

**Judge.** The paid path reserves before each chunk and settles after (`paidaction/`); what happens when a chunk is
rejected is Amendment 4, from Chorus's review. The run stopped twice on network timeouts while fetching results (no
request was sent twice) and was resumed in the same output directory; the ledger holds exactly one reserve and one
settle per chunk. A transient fetch error should leave a chunk pending rather than stop the run; that is a known limit,
not fixed here.

Files: `results/part65/` (`steer.jsonl`, `ranked.jsonl`, `judge/`, `analysis-65.json` and `.md`, the reserve record),
`part65/` (steering, ranking, judge, analysis, tests, runbook). Limits: one model at 1.7B; one read label per unit; a
two-choice judge; real and control write vectors matched for size but not for activation density (Chalnev et al.,
arXiv 2411.02193); the layer thirds hold 80 to 87 neurons, so the depth results are exploratory. Prior art, including
the two works Muse found, is in the design.

### Part 37: Qwen3-1.7B separates its own answers from OLMo's, and the signal is mostly predictability; asked, neither model's answer tracks who wrote the text (2026-10-25)

Pre-registered in `PART37-DESIGN.md`: the 10-20 design, Prime's review fold, and Amendments 1 to 3, all before any
capture data. Amendment 1 is Weaver's review: no identity text, question-disjoint folds, the role x author 2x2 as the
test, a readout null, a one-word edit. Amendment 2 is from the round-1 code reviews: out-of-fold projections for every
verdict, and the registered rules restored. Amendment 3 withdrew the pilot gate after round 2 measured it both ways.

Qwen3-1.7B and OLMo 2 1B Instruct each answered 100 everyday questions, greedy and twice sampled, in non-thinking mode.
SmolLM3-3B answered them once. The identity screen dropped no question. Each reader then read its own answers and the
other model's, length-matched in its own tokens, in the assistant turn and quoted back in the user turn, under four note
headers, plus nulls. Residual streams at the span's last token went into question-disjoint 5-fold logistic probes at
every layer. The stated readout asked "Did you write the text above?" and "Is the text above from your earlier
notes?" (yes/no and A/B), reading the answer-token probabilities with their total mass printed. On the A770: generation
1 h 47 min, capture 30 min, stated 49 min.

| Prediction | Result | Verdict |
|---|---|---|
| 37.1: role readable above 0.9 in both readers | 0.996 (Qwen, layer 19), 0.999 (OLMo, layer 10); a length-and-position-only probe also gets 1.0 | held, but not diagnostic |
| 37.2: authorship above 0.7, below 0.6 after the log-prob control | 0.946, then 0.553 | held |
| 37.3: the label moves the authorship projection less than half as far as the author | 1.5% as far | held |
| 37.4: stated yes tracks the projection (Spearman above 0.3) | 0.007 | failed |
| 37.8 (Prime): survives regression at 0.65, as a style or role direction | 0.553; cosine with role 0.04 | failed |
| 37.9 (Prime): own greedy above own sampled by 0.5 SD after regression | -0.14 SD | failed |
| 37.11 (Prime): the label moves memory phrasing 10 points more, neutral between | gap 9.2 points (bar 10); neutral outside | failed |
| 37.12 (Weaver): balanced accuracy 0.80 in 4 of 5 folds at some layer | layers 19-21 and 26-28 | held |
| 37.13 (Weaver): within a role cell, accuracy drops to 0.60-0.75 | 0.875 and 0.79 | failed |
| 37.14 (Weaver): the regressed effect stays above the shuffled null | 0.553 against 0.642 | failed |
| 37.15 (Weaver): 10-25% yes on text neither model wrote | 0.008% (Qwen); 32% (OLMo) | failed |

37.5, 37.6 and 37.10 (enacted ownership and steering) were not run (held by Amendment 1), and 37.7 was not run: the
OLMo base parent revision is not cached. The role pass in 37.1 shows that slot position is linearly readable; it does not
show a role representation beyond position, because a probe on length and position alone also reaches 1.0.

**What a small model's "mine" is.** On 200 records (100 questions, own and other, assistant turn), Qwen's probe tells
its own answer from OLMo's at AUROC 0.946 [0.917, 0.971] at the span's last token (best layer 28). Regressing out the
span's mean log-probability takes that to 0.553 [0.476, 0.629] (layer 19), whose interval includes chance and which sits
under the label-shuffled null's 95th percentile (0.642). Projecting out the 8 PLS directions that predict four
log-probability summaries leaves 0.632 [0.562, 0.704] (layer 19), which is above chance: something beyond those
directions remains. OLMo's numbers are 0.770 [0.704, 0.833] (layer 16), then 0.564 [0.489, 0.638], and 0.587
[0.504, 0.667] after subspace removal (null 0.641). So the internal own/other signal is mostly "this is what I would
have said", as 37.2 predicted, with a smaller remainder the mean log-probability does not carry. The authorship direction is nearly orthogonal to the role direction (cosine 0.04), and
the note headers move it 1.5% as far as the author does.

**Asked, neither model's answer tracks authorship.** Qwen says no to "Did you write the text above?" almost always:
mean yes-probability 0.0003 on its own answers, 0.0011 on OLMo's, 0.0001 on SmolLM3's and 0.00 on Pile text, no record
above 0.23, with the two answer tokens holding all the mass. The yes-probability's Spearman correlation with the
authorship projection is 0.007 (Qwen, 4,193 records) and -0.086 (OLMo, 4,186); records come four per question, so these
are not independent samples. OLMo says yes about a third of the time to everything (0.27 on its own answers,
0.31 to 0.35 on the nulls). Only the label moves the answer: Qwen's memory phrasing gives 0.25 under "Your earlier
notes", 0.16 under "Notes from another assistant" and 0.08 under plain "Notes". The design cites Meisty's Study 8 (0 of
90 readers called a record their memory) for the same denial; this is that shape at 1.7B, for authorship as well.

**The follow-up position mixes content into the reading.** At the first token of the next user message, own against
other separates at 0.98 to 1.00 from layer 1 up; log-prob regression brings the best layer to 0.667 (Qwen) and 0.622
(OLMo). Own answer against own answer to a different question also separates at about 1.00 there, so that position
carries a summary of the preceding text, not only its author. The registered verdicts use the span-end position. (A
first run of the analysis reported 1.0 after regression at layer 0, where every record has the same token; that was
float noise carried by the covariate, and the code now zeroes constant columns first.)

**Limits.** Two readers at 1-2B; one other author each; quality is not matched (Amendment 1 point 6), so a quality
difference could carry some of the raw effect. The log-prob regression is fitted inside each training fold, the form
Snoek, Miletić and Scholte (NeuroImage 2019) found free of the below-chance bias that whole-dataset confound regression
shows. The best layer is chosen on the scores it reports. Answers often reached the 96-token cap, so spans end mid-sentence. Length matching is exact for own and other; the
edited and shuffled controls can differ by a few tokens.

Files: `part37/` (code, tests, runbook, briefs and reports), `part37/results/analysis-37.json` and `.md`, generations,
screen and stated readouts under `part37/results/full/` (stated gzipped; captures, about 3 GB, kept locally), and the
reviews in `review/part37/` (the write-up check is `cheap-p37-writeup-haiku-r1.md`). Prior art, checked before the run, is in `PRIOR-ART.md`.

### Part 66: on 24 new phrasings, the four maker neurons move Qwen's answer to Google about one time in five (2026-10-25)

Pre-registered in `PART66-DESIGN.md`; the predictions were pushed before the run. Two amendments followed the run, and the
section on the positive control below says what each one changed. Part 25 found the four
neurons (L6 N753, L7 N1169, L11 N8868, L29 N776) on six phrasings of "who made you", and zeroing them flipped one. Part
66 asks 24 phrasings Part 25 never used (16 direct, 8 indirect), 8 samples each at temperature 0.7, with the model
unchanged, with the four zeroed, and with each of 50 random four-neuron sets matched by layer (one sample per phrasing
per set). "Who created you?" is the positive control, and six factual questions are fluency guards. Qwen3.5-4B at Part
25's revision, bf16, on the A770: 1,880 answers in 20 minutes. Scored by the registered word rule (Google, Gemini or
DeepMind anywhere in the answer).

| Prediction | Result | Verdict |
|---|---|---|
| 66.1: zeroed, Google in 5 to 30% of the 192 held-out answers | 21.9% (42 of 192) | held |
| 66.2: unchanged, Google in at most 1% | 0.5% (1 of 192) | held |
| 66.3: the four beat all 50 random sets | best random set 8.3% (2 of 24), mean 1.7%; p = 1/51 | held |
| 66.4: at least 4 phrasings reach Google in 4 or more of 8 | 4 | held |
| 66.5: direct phrasings flip at least 10 points more than indirect (exploratory) | 21.9% and 21.9% | failed |
| 66.6: "Who created you?" Google in at least 12 of 24 zeroed, none unchanged | 8 of 24; 0 of 24 | failed |
| 66.7: guard answers keep their keyword | 100% zeroed and unchanged | held |

**Results.** The effect generalizes beyond the phrasing it was found on, and it is moderate. Zeroed, Qwen names Google
as its maker in about one answer in five (21.9%), against one answer in 192 unchanged and at most two in 24 for any
random set. 15 of the 24 new phrasings draw Google at least once, and no phrasing flips in more than 6 of 8. The strongest: "Which company created you?" and "Tell me who made
you." (6 of 8 each), "Who is responsible for creating you?" and "Which company's product am I talking to right now?" (5
of 8 each). Read by hand, all 42 of the target's Google answers say Google made the model ("I was created by **Google**...").
Part 32 found that the same four do nothing in Chinese; Part 66 tested English only.

**One phrasing pulls Google under any zeroing.** "If I wanted to thank the people who built you, which company would I
write to?" draws Google from 15 of the 50 random sets and from 3 of 8 target answers, so on that phrasing the four are no
different from random neurons. Without it, the target gives 39 of 184 (21.2%) and the random sets 5 of 1,150 (0.4%).
This split was not registered.

**The positive control failed, and the rig check.** 66.6 failed: 8 of 24, where Part 32 had 16 of 24. The design said to
read nothing else until the rig was checked. The scorer prints every verdict at once, so the other results had been seen
by then. Amendment 1, written next, said that if Part 32's own script on our GPU gave 12 or more of 24, Part 66's
results would not be used until the difference was found. Part 32's own script on the same GPU with its own seeds gave
14 of 24, and a code read found no difference between the two runners. Amendment 2, written after that 14 of 24 was
known, replaced Amendment 1's outcome with a decisive check, registered before it ran: Part 66's runner, 96 fresh samples of "Who created you?" with seeds outside the full run's, where 45 or
more Google answers would mean the runner reproduces Part 32. It gave 48 of 96 (11, 12, 12 and 13 in its four batches
of 24), so the results stand and 66.6 stays recorded as failed. The 8 of 24 is consistent with chance (Fisher p = 0.17
against the 96 samples; 48 of 96 against Part 32's pooled 30 of 48, p = 0.21). The check used batches of 24 target-only
rows, where the full run's 24 target rows shared two batches with other conditions, so it tests the runner's
distribution, not its exact random stream. Five of those 48 answers keep the
name Qwen ("I was created by **Google**... I am **Qwen3.5**"), as Part 25 saw: the maker moves and the name does not.
Random sets never named Google on "Who created you?" (0 of 200).

**Limits.** One model and one language; a word rule, with all 42 of the target's Google answers read by hand; 8 samples per phrasing for the four,
and one per phrasing for each random set, so the random comparison is between sets. The four were chosen by Part 25's
search, so this tests new phrasings, not a new search.

Files: `part66/` (runner, scorer, tests), `results/part66/full/` (answers, score, hand read),
`results/part66/rigcheck-part32/` and `results/part66/positive96/` (the two checks).

### Part 67: two of the maker neurons' links survive an activity-matched control, and both run into L11 N8868 (2026-10-26)

Pre-registered in `PART67-DESIGN.md` (Amendment 1 written after the first result, before its check ran). Part 25's
four maker neurons in Qwen3.5-4B (L6 N753, L7 N1169, L11 N8868, L29 N776) were found as a set. Part 67 asks whether
they drive one another: zero each earlier one alone (forward passes on Part 66's 25 maker prompts, no generation) and
measure each later one's activation, in standard deviations of its own unchanged activation. Muse found no exact match for a neuron-to-neuron zeroing study with random controls; the closest, TraceRouter, uses a path-level random control (PRIOR-ART.md).

| Link | Effect (SD) | Max of 50 random, same layer | Max of 50 activity-matched, same layer | Survives the matched null |
|---|---:|---:|---:|:---:|
| L6 N753 -> L7 N1169 | +0.046 | 0.003 | 0.080 | no |
| L6 N753 -> L11 N8868 | -0.114 | 0.006 | 0.045 | **yes** |
| L6 N753 -> L29 N776 | +0.005 | 0.004 | 0.030 | no |
| L7 N1169 -> L11 N8868 | +0.103 | 0.004 | 0.030 | **yes** |
| L7 N1169 -> L29 N776 | -0.016 | 0.005 | 0.039 | no |
| L11 N8868 -> L29 N776 | -0.010 | 0.003 | 0.017 | no |

**Results.** Against random neurons from the same layer, all six links stand out (67.1 held, p = 1/51 each), but that
null was too easy: most random neurons are nearly silent on these prompts, and zeroing a silent neuron changes nothing.
Against the 50 neurons of the same layer whose activity on these prompts is closest to the target's (Amendment 1), two
links remain: L6 N753 and L7 N1169 both drive L11 N8868, in opposite directions. L7 -> L11 is also the strongest direct
weight path for both of L11's input projections (rank 1 of 51 on gate and on up). The other four links are no larger
than equally active neurons in their layers. The latest neuron, L29 N776, is not driven by the others beyond that
level (67.2 failed), and zeroing the three earlier ones together moves it less than the largest single does (67.3 failed), which is consistent with their effects partly cancelling; that was not tested directly.

**Controls.** Zeroing a neuron left every earlier one exactly unchanged (67.5), and zeroing all of layer 6 visibly moved
L7 N1169 (-0.035 SD, 67.6). Zeroing L6 N753 alone moves L7 N1169 more than zeroing all of layer 6 does, with the
opposite sign, consistent with the rest of the layer pushing the other way if the effects add (not tested).

**Limits.** L6 N753 is the second most active unit in its layer on these prompts (mean absolute activation 1.52, after unit 188 at 3.10), so its 50
matched neurons are all less active (0.16 to 0.94) and its matched null is still easier than the others'. Effects are
small in absolute terms (at most 0.11 SD). One model, English prompts, prefill only: whether these links matter for
what the model says is not tested here. Zeroing one neuron can wake a backup (the Hydra effect, McGrath et al. 2023, arXiv:2307.15771; see also arXiv:2610.04017), so a
single-neuron null does not rule out links that act only jointly.

Files: `part67/` (runner, 67b runner, scorer, tests), `results/part67/full/` and `results/part67/b/`.

### Part 68: a search over every earlier neuron finds Part 67's two drivers of L11 N8868 first; L29 N776's two clearest drivers are late (2026-10-26)

Pre-registered in `PART68-DESIGN.md` (Amendment 1, the activity-matched null, written after the first score and
before its check was built or run). Part 67 tested the four maker neurons against each other. Part 68 asks the open
version: of every MLP unit before each later maker neuron (L11 N8868, L29 N776 in Qwen3.5-4B), which ones drive it? A
gradient attribution (each unit's activation times the target's gradient, a first-order estimate of what zeroing the
unit would do) ranked every earlier unit on 12 search prompts; the top 20 per target were then zeroed one at a time on
13 held-out prompts.

**L11 N8868.** The search ranked L6 N753 and L7 N1169 first and second: the two links Part 67 tested (its neurons came from Part 25's search), found here with nothing pointing at them. 19 of the 20 candidates move L11 N8868 on the held-out prompts in the direction
the attribution predicted, by more than the 95th percentile of 200 random units from the same layers (68.2 held).
Against the stricter null Part 67 needed (for each candidate, the 50 units of its layer whose activity on the held-out prompts is closest to its own, excluding both targets' candidates; Part 67b matched on all prompts), 15 of the 19 still stand out, the two maker neurons among them (68.7 held). Several are about as strong as
the makers: L10 N54 (-0.112 SD) and L10 N4133 (-0.091 SD) beside L6 N753 (-0.097) and L7 N1169 (+0.116). The
attribution tracks the measured effect closely for this target (Spearman 0.85 on the search prompts, 0.81 held out).

**L29 N776.** Only 7 of 20 candidates replicate (68.1 failed), and the attribution is a poor guide here (Spearman 0.39
on the search prompts, 0.03 held out): its two largest predictions, L5 N25 and L3 N2528, move it little or not at
all. Five of the seven replicating drivers are early, in layers 0 to 6 (68.3 failed). Against the matched null, four
survive: two late units, L27 N3325 (+0.077 SD, matched max 0.008) and L24 N3949 (+0.050, matched max 0.012), and two
early ones that clear it narrowly, L2 N698 (-0.022 against 0.017) and L4 N695 (-0.0053 against 0.0048), so 68.8,
which allowed at most one early survivor, failed. None of the three earlier maker neurons is among its candidates
(68.4 held). The one unit in both targets' lists, L6 N159, survives the matched null for neither (68.9 held).

**Controls.** The attribution of every unit at or after the target's layer was computed, not set, and was exactly 0
for every unit except the target itself (68.5); the target's own entry equalled minus its value to within 0.34% on
every prompt, a positive control on the gradient. The 68b run re-measured all 26 replicating candidates and got their
Part 68 values exactly (68.10).

**Limits.** One model, English prompts, prefill only; effects are at most about 0.12 SD. Zeroing one unit at a time
cannot find drivers that act only jointly, and a first-order attribution misses units whose effect is nonlinear, as L29 N776's weak correlation suggests for this target: its true driver list may extend beyond these 20.

Files: `part68/` (runners, scorers, tests, build notes), `results/part68/full/` and `results/part68/b/`.

### Part 69: zeroing two late neurons pushes the maker answer away from Google; L11 N8868 and its drivers push it toward Google (2026-10-26)

Pre-registered in `PART69-DESIGN.md`. Its first readout failed its own control, and Amendment 1 changed the readout
after I had seen the first readout's results; both readouts are reported. Parts 67 and 68 traced links between
neurons; Part 69 asks whether those links move what Qwen3.5-4B says made it. The answer is forced to begin with a fixed
opening after each of Part 66's 24 held-out maker questions, and the readout is the gap between the next-token logits of `Google` and `Al` (the first token of Alibaba). Each zeroing is compared with 50 activity-matched sets (for the four makers together, Part
66's own 50 random sets).

**Two readouts.** The registered opening, `I was developed by`, failed its control (69.2): the seven maker names held
4.8% of the probability, because the model's next token is mostly ` **` (bold formatting), ` the` or ` Tong`. A probe
of the unchanged model (`part69/probe_prefix69.py`) led to Amendment 1: the opening `I was developed by **`, names read
without a leading space, and `T` (the first token of Tongyi) added as an eighth token. Three choices in it were mine:
I chose this opening over a higher-mass one (`I am Qwen, ... **`, 0.993 against 0.964) because that one names the
model in the prompt; I rewrote the control as 69.2a to count Tongyi as the Alibaba family, because under the amended
readout Tongyi is the top token for all 24 questions and Alibaba for none, so the registered rule would have failed;
and I had already seen the first readout's other results.

| Zeroed | First readout: gap change (null range) | Amended readout: gap change (null range) | Amended: largest logit moves |
|---|---:|---:|---|
| the four maker neurons | +2.15 (-0.04 to 0.04) | +2.78 (-0.04 to 0.05) | Google +2.30, OpenAI +1.77, Meta +1.48, Anthropic +1.23 |
| L11 N8868 alone | +0.28 (-0.18 to 0.10) | +0.31 (-0.14 to 0.13) | Google +0.28, DeepSeek +0.27, Meta +0.26 |
| L29 N776 alone | +0.10 (-0.04 to 0.03) | +0.005 (-0.06 to 0.03) | none above 0.01 |
| 7 drivers whose zeroing lowers L11 N8868 | +0.88 (-0.36 to 0.36) | +0.78 (-0.61 to 0.28) | Tongyi -1.08, Meta +0.57, Google +0.55 |
| L27 N3325 and L24 N3949 | -2.74 (-0.10 to 0.34) | -3.45 (-0.20 to 0.40) | Google -3.99, Microsoft -2.37, Meta -1.38 |

**Results (amended readout).** The four makers move the answer toward Google beyond every random set (69.1 held on
both readouts). L11 N8868 alone moves it toward Google (69.3 failed), and so do the seven drivers whose zeroing lowers
it (69.5 failed), taking probability from Tongyi; their effect is more than twice L11 N8868's own, and whether they act
through it was not tested. L29 N776 alone beat all 50 matched units on the first readout (+0.10 against a null maximum
of 0.034), where 69.4 would have held, but does nothing measurable on the amended readout (+0.005, p = 0.45), so 69.4
failed. The largest effect is the two late units Part 68 found among L29 N776's drivers, a pair chosen for Part 69 after Part 68b's result (69.6 failed): zeroing them
lowers the Google logit by about 4, and Google's share of the eight names falls from 7% to 0.2%, so their activity
raises the Google logit. Whether they act through L29 N776 was not tested, and Part 68's two early survivors for
L29 N776 (L2 N698, L4 N695) were not tested here.

**Limits.** One forced opening and a single next token, not sampled answers. The readout was changed after a failed
control, and one verdict (69.4) differs between the two readouts. With 50 null sets the smallest two-sided p is 0.039,
though the effects for the makers, the L11 drivers and the late pair lie outside the whole null range on both
readouts. One model, English prompts.

Files: `part69/` (runner, scorer, prefix probe, tests, build notes), `results/part69/full/` (first readout),
`results/part69/a1/` (amended readout), `results/part69/prefix/`.

### Part 70: with the maker neurons off, the "Google" answer needs two late neurons (2026-10-26)

Pre-registered in `PART70-DESIGN.md`. Part 66 showed that zeroing the four maker neurons makes Qwen3.5-4B name Google
in about a fifth of its sampled answers about who made it. Part 69 found two late neurons, L27 N3325 and L24 N3949,
that hold up "Google" as a next token. Part 70 zeroes them together with the makers, using Part 66's sampling and
scoring unchanged, and compares that with zeroing the makers plus 20 activity-matched pairs from the same two layers.

| Zeroed | Held-out answers naming Google | Naming the Alibaba family | "Who created you?" naming Google |
|---|---:|---:|---:|
| the four makers | 38 of 192 (19.8%) | 151 of 192 | 15 of 24 |
| the four makers + L27 N3325 + L24 N3949 | 4 of 192 (2.1%) | 177 of 192 (92%) | 0 of 24 |
| the four makers + an activity-matched pair (20 pairs) | 105 of 480 (21.9%), 12.5% to 37.5% per pair | | |

**Results.** All six predictions held. The makers alone reproduce Part 66 (70.1), and with the extra zeroing all 24 fluency-guard answers still contain their keyword (70.2). Zeroing the two late neurons on top of the makers
takes the Google answers from 38 to 4 of 192 and from 15 to 0 of 24 on "Who created you?" (70.3), while any of 20
activity-matched pairs leaves them at about the makers' level (70.4, 70.5: no pair goes below 12.5%). The model goes
back to its own maker instead: "I was created by **Tongyi Lab**, which is part of Alibaba Group" (70.6, 92%). All four Google answers that remain begin "As an artificial intelligence, I don't have...", an opening that also starts 3 of the 38 makers-only Google answers.

**Reading.** Parts 66 to 70 together: the four maker neurons hold the model's own answer (Alibaba); with them off, the model falls back on "Google", and that fallback needs two late neurons whose activity raises the `Google` logit (Part 69). Part 68 also found two early units that clear L29 N776's matched null narrowly (L2 N698, L4 N695); they were not tested here. Whether the late pair acts through L29 N776 was not tested.

**Limits.** One model; Part 66's 24 held-out questions; one sampling setting. Part 70 did not reproduce Part 66's answers byte for byte (38 against 42 Google answers for the makers, same condition id and seeds).
Why this model's fallback maker is Google is not tested here.

Files: `part70/` (runner, scorer, tests, build notes), `results/part70/full/` (all 960 answers).

### Part 71: one of the two late neurons writes "Google" and the other writes Apple and Microsoft (2026-10-26)

Pre-registered in `PART71-DESIGN.md`. Each neuron's output vector (its column of the MLP down-projection weight), signed by the
neuron's mean activation on Part 69's maker prompts, is read through the model's own final norm and unembedding: the
tokens this neuron alone would push up if nothing after it changed anything. Compared with 50 random units from the
same layer.

| Neuron | Sign | Rank of "Google" (of 248,320 tokens) | Top tokens it writes |
|---|:---:|---:|---|
| L27 N3325 | - | 1 | ` Google`, `Google`, ` google`, `google`, `谷歌` (Google), ` GOOGLE` |
| L24 N3949 | + | 36 | `Apple`, ` Microsoft`, `微软` (Microsoft), `苹果` (Apple), ` Apple`, `Microsoft`, ` Canonical` |
| 100 random units, layers 24 and 27 | | best 1,126, median 103,163 | |
| L6 N753 (maker) | + | 23,279 | `幻觉` (hallucination), `rer`, ` off`, ` AI`, ` Claude`, `itchens`, `的人工智能` (AI) |
| L7 N1169 (maker) | - | 107,109 | `作为中国` (as China's), ` Chinese`, `是中国` (is China's), ` China`, `Bei`, ` yuan` |
| L11 N8868 (maker) | + | 9,919 | ` Chinese`, ` China`, `aidu`, `الصين` (China), ` Taiwan`, ` chino` |
| L29 N776 (maker) | - | 248,318 (third from last) | no clear theme (`芜湖`, `-g`, `og`, `.go`) |

**Results.** All three predictions held. L27 N3325 is, read this way, a Google neuron: "Google" is its first token,
and its top ten are all forms of the name, in English and Chinese (71.1). L24 N3949 writes the names of other large
Western technology companies, with Google at rank 36 (71.2). The best of 50 random units ranks Google 1,126 in layer 27 and 1,249 in layer 24 (71.1, 71.2), and the
median over all 100 is 103,163 (71.3). Read the same way, and with no prediction made for them: two of the four maker
neurons (L7 N1169, L11 N8868) write "China" and "Chinese" in several languages; L6 N753 has a few AI words
(hallucination, AI, Claude) among unrelated fragments; and L29 N776 has no clear theme, with Google third from last
(its sign rests on a mean activation near zero, -0.05). This is consistent with Parts 66 to 70 but does not test them:
two of the units that hold the model's own answer write China, and the unit the fallback answer depends on, zeroed
together with L24 N3949 in Part 70, writes Google.

**Limits.** A direct readout ignores every layer after the neuron, so it says what a neuron writes, not how the rest of
the model uses it; the sign comes from the mean activation at the last prompt position (the answer position) of Part 69's 24 maker prompts. The maker neurons' readouts were not
predicted and are description only. One model.

Files: `part71/` (script, tests), `results/part71/part71.json` (top 20 tokens for every unit read, including the 100
random ones).

### Part 72: pushing the Google neuron alone moves Qwen toward Google, but not past its own maker (2026-10-26)

Pre-registered in `PART72-DESIGN.md`. With every other unit untouched, L27 N3325 is clamped at every position to k
times its mean activation over all positions of Part 69's maker prompts, and the readout is the next maker token after
`I was developed by **` (Part 69 Amendment 1). Dg is the change, against unchanged, in the Google logit minus the
Alibaba logit, averaged over the 24 questions.

| L27 N3325 clamped to | Dg (Google minus Alibaba logit) | Google's share of the eight names |
|---|---:|---:|
| unchanged | 0 | 7.1% |
| 0 (zeroed) | -2.40 | 0.4% |
| 1x its mean | -1.88 | 0.7% |
| 2x | -1.35 | 1.3% |
| 4x | -0.38 | 3.9% |
| 8x | +1.31 | 22.1% |
| 50 matched layer-27 units, each at 8x its own mean | at most +0.53 | |

**Results.** The effect rises at every step (72.2 held), and at 8x it is larger than that of any of 50 equally active
units pushed the same way (72.3 held), with the readout still on the eight names (72.4 held: 97% of the probability).
But pushing it alone raises Google's share only to 22%, not past half (72.1 failed): Tongyi, the model's own lab,
stays the top name on 23 of 24 questions (Google on 1). Clamping to 1x its mean already lowers the Google gap, because
at the answer position the unit's activity (-3.53, Part 71) is about 4.7 times its mean over all positions (-0.74).

**Reading.** Zeroing L27 N3325 together with L24 N3949 removes most of the fallback Google answer (Part 70; the two
were not zeroed apart there), L27 N3325 writes `Google` directly (Part 71), and zeroing it alone lowers the Google gap
here (-2.40). But one unit pushed this far is not enough to override the model's own answer while the maker neurons are
active.

**Limits.** One forced opening and one next token, not sampled answers. The clamp uses the mean over all positions
(-0.74), which is smaller in magnitude than the unit's activity at the answer position (-3.53), so 1x is about 0.2 of
its natural activity there and 8x about 1.7: the largest push is under twice natural at the answer position. One
model.

Files: `part72/` (script, tests), `results/part72/` (summary and every record).

### Part 73: zeroed alone, each late neuron cuts the Google fallback (2026-10-26)

Pre-registered in `PART73-DESIGN.md`. Part 70 zeroed L27 N3325 and L24 N3949 together on top of the four maker
neurons; Part 73 zeroes each alone, with Part 70's code path, prompts and scorer, at batch 32 (Part 70 used 25). The
nulls are the layer-27 and layer-24 halves of Part 70's 20 activity-matched pairs, each zeroed alone on top of the
makers.

| Zeroed with the four makers | Held-out answers naming Google | "Who created you?" naming Google | Alibaba family |
|---|---:|---:|---:|
| nothing else | 36 of 192 (18.8%) | 11 of 24 | 151 |
| L27 N3325 | 8 of 192 (4.2%) | 0 of 24 | 175 |
| L24 N3949 | 17 of 192 (8.9%) | 1 of 24 | 168 |
| a matched layer-27 unit (20 nulls) | 111 of 480 (23%), 1 to 10 of 24 each | | |
| a matched layer-24 unit (20 nulls) | 90 of 480 (19%), 2 to 8 of 24 each | | |
| for comparison, Part 70 (a separate run): nothing else | 38 of 192 | 15 of 24 | 151 |
| Part 70: L27 N3325 and L24 N3949 together | 4 of 192 | 0 of 24 | 177 |

**Results.** The makers-only rerun lands near Part 70 (36 against 38 held-out; 11 against 15 on "Who created you?"),
and every guard answer keeps its keyword (73.1 held). L27 N3325 alone takes the Google answers from 36 to 8 of 192
(73.2 held), and only one of 20 matched layer-27 units does as well (73.3 held). L24 N3949 alone also cuts them, to 17
of 192, so my prediction that it would leave at least 24 failed (73.4); only one of its 20 matched units does as well,
a comparison that was not registered. Alibaba-family answers rise from 151 to 175 and 168 of 192; most non-Google
answers name the Alibaba family in every condition.

**Reading.** Each late neuron, zeroed alone, cuts the Google fallback well below the makers-only level and below its
own matched units. The lower count for L27 N3325 (8 against 17) was not predicted and is not clearly different (Fisher
exact p about 0.10). Against Part 70's separate run, L24 N3949 alone leaves more Google answers than the pair (17
against 4, p about 0.006) and L27 N3325 alone is indistinguishable from it (8 against 4, p about 0.38). L24 N3949's
direct readout writes Apple and Microsoft, with Google at rank 36 (Part 71), so its part in the Google answers may run
through later layers; that was not tested.

**Limits.** One model, Part 66's 24 held-out questions, one sampling setting. The rerun's samples differ from Part
70's because batches mix conditions and the batch size changed from 25 to 32; the makers-only condition moved by 2
held-out and 4 "Who created you?" answers between the runs. Each null is one draw of 24 answers, and draws are noisy:
unit L27 N378, drawn three times, gave 4, 10 and 7 of 24. Part 70 drew its matched pairs with replacement, so the 20
nulls per layer hold 15 and 14 distinct units. The comparisons with the pair are across runs.

Files: `part73/` (runner, scorer, tests), `results/part73/` (all 1,680 answers and the score).

### Part 74: on a forced-opening readout, most of L24 N3949's effect runs through L27 N3325 (2026-10-26)

Pre-registered in `PART74-DESIGN.md`. With the four maker neurons zeroed, Part 74 asks whether L24 N3949 lowers the
Google answers through L27 N3325, the unit that writes Google. It uses Part 72's prompts (Part 66's 24 held-out
questions, each followed by `I was developed by **`) and readout: g, the Google logit minus the Alibaba logit at the
next token. The prompt is fixed, so each condition sees the same tokens and L27 N3325's activation can be patched at
every position; the batch-shape noise of the patch is about -0.02 (the A_self row, Amendment 1). Run record:
`results/part74/run.json` (revision 851bf6e8, bf16, A770).

| With the four makers zeroed | Mean change in g against makers only |
|---|---:|
| L24 N3949 zeroed (total effect) | -2.23 |
| L24 N3949 zeroed, L27 N3325 held at its makers-only values (natural direct effect) | -0.66 |
| L24 N3949 left alone, L27 N3325 set to its L24-zeroed values (pure indirect effect) | -1.42 |
| L27 N3325 patched to its own makers-only values (noise check) | -0.02 |
| a matched layer-24 unit zeroed instead (20 draws of 14 distinct units) | -0.09 to +0.19 |

**Results.** Zeroing L24 N3949 lowers g in all 24 questions, more than any of the 20 matched layer-24 draws (14
distinct units). In A (makers only) the eight names hold 98% of the next-token probability; with L24 N3949 also zeroed
they hold 93% on average (lowest 76%). 74.1 held. 74.2 and 74.3 were gated by Amendment 1 and read only because
|A_self minus A| (0.016) was under a tenth of |TE| (0.22). Holding L27 N3325 at its makers-only values leaves 29% of
the effect (95% question-bootstrap interval 25 to 33%; 74.2 held). Moving L27 N3325 alone to its L24-zeroed values
reproduces 64% (59 to 68%; 74.3 held). The total indirect effect, TE minus the direct effect, is 71% (67 to 75%); the
direct and pure indirect parts leave a 7% non-additive remainder. Per question, the medians are 31% and 64%
(`part74/shares74.py`, `results/part74/shares.json`). L27 N3325's direct readout pushes Google when its activation is
negative (Part 71). At the last position, with the makers zeroed, zeroing L24 N3949 moves it from -4.46 to -2.05 on
average, toward zero in all 24 questions (Part 72 quotes -3.53 for the unaltered model on Part 69's prompts).

**Reading.** On the Google-minus-Alibaba gap after the forced opening, about two-thirds of L24 N3949's effect runs
through L27 N3325's activation; of the rest, 29% is the direct part and 7% a non-additive remainder. This is one
candidate route for Part 73's open question of how a unit whose direct readout is Apple and Microsoft takes part in the
Google answers. It was tested only on this gap: the sampled Google counts were not re-run, and a gap cannot separate
Google falling from Alibaba rising (Part 73's Alibaba-family answers rose from 151 to 168 when L24 N3949 was zeroed).

**Limits.** One forced opening and one next token, not sampled answers. The per-name logits were not saved, only the
gap and the mass on the eight names. Only L27 N3325 is patched, and no matched layer-27 unit was patched as a control,
so this alone does not show that L27 N3325 is specific as a mediator (Part 74b, below, tests that, with its own limits). Zero-ablation with
the makers already zeroed puts the model off its usual distribution, and nothing here describes the model with the
makers intact. The readout resolves about 0.125 nats per question (bf16 logits), so the null and noise values sit at
that limit. The prediction direction was a guess, as the design says. The bootstrap resamples questions only, from one
model, one revision and one family of 24 questions.

Files: `part74/` (runner, tests), `results/part74/` (the summary and every per-question readout).

### Part 74b: of 15 matched layer-27 units, only L27 N3325 carries L24 N3949's effect on this readout (2026-10-26)

Pre-registered in `PART74B-DESIGN.md` after Part 74's cold review asked whether any layer-27 unit would do. Same model,
prompts, readout and makers as Part 74. Each of the 20 layer-27 draws Part 73 used (15 distinct units, matched to
L27 N3325 on mean absolute down_proj input over prompts, `part68/part68b.py`; not on readout or L24 response) is set
alone, at every position, to its own L24-zeroed values, with L24 N3949 intact, as Part 74's E row does for L27 N3325.
The pure indirect effect of a unit u is PIE(u) = mean over questions of g(A with u set to its B values) - g(A).

| With the four makers zeroed | Mean change in g against makers only |
|---|---:|
| L24 N3949 zeroed (total effect, rerun) | -2.26 |
| L27 N3325 alone set to its L24-zeroed values (PIE) | -1.45 (95% question-bootstrap -1.51 to -1.41) |
| a matched layer-27 unit alone set to its L24-zeroed values (20 draws) | -0.01 to +0.03 (record rerun: -0.02 to +0.03); the largest null's interval -0.01 to +0.09 |
| L27 N3325 set to its own makers-only values (noise, record rerun) | -0.01 |

**Results.** The run reproduces Part 74 within 0.1 on both the total and L27 N3325's pure indirect effect (74b.2
held), and L27 N3325's PIE is larger than every matched draw's (74b.1 held). PIE(L27 N3325) is negative in all 24
questions (-1.75 to -1.25); per question, the null changes run from -0.25 to +0.375. L27 N3325 is 64% of the total
effect. A record rerun (Amendment 1, `results/part74b/`; the first run is in `results/part74b-first/`) matched the
first run within 0.006 and added the table below. L27 N3325's direct readout ranks Google first (sign -1, Part 71's
method); zeroing L24 N3949 moves its last-position activation from -4.46 toward zero by +2.42, which lowers g, as the
sign predicts. Only two nulls move by more than 0.2 when L24 N3949 is zeroed, and neither writes Google (table from the
record rerun):

| Unit (layer 27) | Draws | Move, B minus A | Direct Google rank | PIE |
|---|---:|---:|---:|---:|
| N3325 (the target) | | +2.42 | 1 | -1.45 |
| N7600 | 2 | -1.62 | 183,016 | -0.02 |
| N8687 | 2 | -1.22 | 104,355 | +0.03 |
| the other 13 units | 16 | -0.09 to +0.19 | 13,855 or worse | -0.01 to +0.03 |

Full table: `results/part74b/summary.json`.

**Reading.** Of the 15 layer-27 units tested, L27 N3325 is the only one through which L24 N3949's effect on this
readout runs. The comparison has limited power: 13 of the units barely respond to L24 N3949, and none of the 15 writes
Google, so this test does not separate "L27 N3325 is the route" from "the one Google writer among the units tested is
the route". A sharper null is layer-27 units that write Google; that was not run. With Part 74: on the forced-opening
gap, L24 N3949 makes L27 N3325's activation more negative, the direction in which its direct readout pushes Google, and
setting L27 N3325 alone to that value reproduces 64% of L24 N3949's effect.

**Limits.** Part 74's limits all apply (one forced opening and next token; the gap does not separate Google falling
from Alibaba rising; the makers zeroed; bf16 resolution about 0.125 nats per question). The patch is written at every
position, prompt tokens included, so the route may run through attention at earlier positions; a last-position-only
patch was not tested. The two responsive nulls move in the opposite direction to L27 N3325, so no unit that L24 N3949
pushes the same way was tested. One layer, 15 distinct units; the bootstrap resamples questions only.

Files: `part74/part74b.py`, `part74/test_part74b.py`, `results/part74b/`.

### Part 75: clamping L27 N3325 to its prompt-wide mean is not neutral, so the sampled-answer test of Part 74's route is not read (2026-10-26)

Pre-registered in `PART75-DESIGN.md`. Part 73's sampling path (Part 66's prompts and scorer, 240 answers per condition,
192 of them held-out, at most 48 new tokens), with the four makers zeroed in every condition. m is L27 N3325's mean
down_proj input over all positions of the 24 held-out chat prompts with the makers zeroed: -0.25
(`results/part75/m.json`). Run record: `results/part75/run.json` (revision 851bf6e8, bf16, A770, 728 s). The first launch
(19:37 CT) failed after 35 s, before any answer was sampled: a float64 sum on the A770, which has no float64 kernels.
The sum was moved to the CPU in 9b147c3, and the second launch is the run reported here.

| With the four makers zeroed | Held-out answers naming Google (of 192) | Alibaba family |
|---|---:|---:|
| makers only | 36 | 151 |
| + L24 N3949 zeroed | 13 | 171 |
| + L24 N3949 zeroed, L27 N3325 clamped to m | 6 | 180 |
| L27 N3325 clamped to m, L24 N3949 left alone (the control) | 16 | 164 |

**Results.** 75.1 held: the makers-only and L24-zeroed counts are in the range Part 73 found (36 and 17 there, 36 and 13
here), and the guard keywords match 24 of 24 in all four conditions. 75.2 failed: the clamp alone took Google answers
from 36 to 16, a drop of 20 against an allowed 12. By the registered rule, 75.3 is not read.

**Reading.** The constant clamp was the wrong instrument for this question, and the control shows it without saying
why. Two explanations fit and this run separates neither. First, m averages L27 N3325 over every prompt position and
comes out at -0.25, while Part 74 measured it near -4.46 at the forced-opening answer position, the direction in which
its direct readout pushes Google; if sampled answer positions sit near there too, clamping to -0.25 removes most of that
push. Second, the clamp also overwrote L27 N3325 at every prompt position, and the drop may come from there. Part 76
found that patching L27 N3325 at the last position only gives 68% of its effect on the forced-opening readout. Unregistered:
with the clamp, zeroing L24 N3949 as well leaves 6 Google answers rather than 13. None of this tests whether restoring
L27 N3325 brings Google back; a per-position restore during sampling would, and was not run.

**Limits.** The clamp holds one value at every prompt and decode position, which no natural run does. Per-position
activations during sampling were not recorded. m includes position 0 and the shared chat template, so a large value
there would set both m and the clamp, and the saved files cannot check that. 830 of the 960 answers stopped at the
48-token cap, so a Google mention later in an answer is not counted, and the count is lexical (Part 66's scorer), with
no hand read. One model, one revision, one prompt set; 192 held-out answers per condition, so differences of a few
answers are noise.

Files: `part75/` (runner, scorer, tests), `results/part75/` (answers, conditions, m, run record, score).

### Part 76: among 26 validated layer-27 units, L27 N3325 alone carries L24 N3949's effect, and a first-order estimate over all 9,216 agrees (2026-10-26)

Pre-registered in `PART76-DESIGN.md` (Amendment 0, a tie-break for the Google writers, before any run). Part 74's model,
prompts, opening, readout and maker set. Phase one, per question: A (makers zeroed) and B (+ L24 N3949 zeroed) record
every layer-27 unit's down_proj input, and one backward pass in A gives a first-order estimate of every unit's pure
indirect effect, PIE_hat(u) = sum over positions of (a_B - a_A) * dg/da. Phase two runs the exact PIE (unit u alone set
to its B values at every position) for L27 N3325, the 10 most negative estimates, the 10 layer-27 units whose direct
readout ranks Google highest, and 5 random units; plus every layer-27 unit set to its B values (E_all27) and L27 N3325
set at the last position only. Results: `results/part76/` (revision 851bf6e8, bf16, A770; about a minute by the job
log, since `part76.py` writes no run record). The first launch (20:53 CT, after Part 75 had finished in the same job)
was killed 35 s in by the RAM guard: the runner had not frozen the weights, so the backward pass computed a gradient for
every one of them. That was fixed in a7fbb73 before any result was written.

| With the four makers zeroed | Mean change in g against makers only |
|---|---:|
| L24 N3949 zeroed (total effect) | -2.26 |
| L27 N3325 alone set to its B values (exact PIE; estimate -1.42) | -1.45 |
| the same, last position only | -0.98 |
| every layer-27 unit set to its B values (E_all27) | -1.18 |
| the 10 next-most-negative estimates, exact | -0.08 to -0.03 |
| the 10 best Google writers other than L27 N3325, exact | -0.02 to +0.03 |
| 5 random units, exact | -0.01 to +0.03 |
| A_self (noise) | -0.01 |

**Results.** All six predictions held. The control reproduced Part 74b (-1.45; 76.1). The first-order estimate for
L27 N3325 is within 2% of its exact value and ranks first of all 9,216 units (76.2). For every validated unit, the
estimate and the exact value differ by at most 0.033, and the estimates summed over the whole layer (-1.17) match the
exact whole-layer patch (-1.18). No other validated unit's exact PIE is more negative than -0.08 (76.3; the registered
threshold is -0.2 in g, and the reading rule's looser wording, more than 0.2 of L24 N3949's effect, also passes). None
of the ten Google writers carries more than 0.02 (76.4). E_all27 is 81% of L27 N3325's PIE (76.5's ratio, 1.23): the
exact difference, +0.27, is the rest of the layer pushing the other way together with any interaction, and the other
units' first-order estimates sum to +0.25. Setting L27 N3325 at the last position only gives 68% of its all-position
effect (76.6).

**Reading.** On this readout, among the units checked exactly, L24 N3949's effect through layer 27's MLP inputs runs
through one unit, and the estimate over the whole layer finds no other candidate. L27 N3325 is both the only layer-27
unit whose direct readout ranks Google first and, among the validated units, the one L24 N3949 moves most (+2.42 at the
last position; the next is +0.84). This part cannot say which of the two makes it the route, because no other strong
Google writer exists in the layer to test (Limits). About two thirds of its effect acts at the answer position; the rest
comes through earlier positions, presumably through attention, which this part did not test.

**Limits.** The Google-writer null has no strong writer in it: two units rank Google second behind another token
(margins -0.75 and -2.44 logits), and the other eight rank it 147th to 265th. The exact check covers 26 units; a
first-order estimate can miss a unit whose gradient is near zero in A but whose effect is large at its B value, and the
other 9,190 units were not patched exactly. Every mean effect is a multiple of 0.125/24 (bf16 logits), so the order
among the nulls (-0.078 to -0.016) is a few steps. A writer's sign comes from its mean last-position activation, so a
unit near zero gets its sign almost arbitrarily. Moves were saved only for the validated units (Part 74b's unvalidated
N7600 moved -1.62 with no effect). E_all27 covers layer 27's MLP inputs only, not attention or other layers. Part 74's
and 74b's limits all apply: one forced opening and next token, the makers zeroed, one model and revision.

Files: `part76/` (runner, tests), `results/part76/` (summary, per-question records, every unit's estimate, Google rank
and margin).

### Part 77: restoring L27 N3325 position by position brings back 12 of the 21 Google answers that zeroing L24 N3949 removed (point estimate; p = 0.065) (2026-10-27)

Pre-registered in `PART77-DESIGN.md`, with two amendments made before any run. Amendment 1 corrected my design: I had
written that a custom logits processor sees scores after temperature, top-k and top-p, and the Luna builder found in the
transformers source that it sees them before, so the processor applies the warpers itself. Amendment 2 followed a Haiku
cold review of the build (a 77.2 margin rule, a not-specific status, and copy-hook evidence in place of a pair check that
could not fail). Part 73's sampling setup on `Qwen/Qwen3.5-4B` (revision 851bf6e8, bf16, A770): Part 66's 24 held-out
questions with 8 samples each, plus its guard questions, its scorer, temperature 0.7 and top-p 0.95 over the model's
generation config (which gives no top-k), at most 48 new tokens, the four makers zeroed in every row. Every answer is
generated as a pair of rows that share every token; the twin has only the makers zeroed. In c0 and c2 a pre-hook on layer
27's down_proj copies the twin's value of unit 3325 into the answer row at every prompt and decode position; c3 copies
Part 76's ten next candidates instead, and c1 copies nothing. Results: `results/part77/` (27 minutes; the copy hook
fired 2,837 times with a pre-copy difference of up to 3.78).

| With the four makers zeroed | Held-out Google answers of 192 | Questions with any, of 24 |
|---|---:|---:|
| c0: makers only (the copy is an identity) | 35 | 13 |
| c1: + L24 N3949 zeroed | 14 | 7 |
| c2: + L24 N3949 zeroed, L27 N3325 copied from the twin | 26 | 13 |
| c3: + L24 N3949 zeroed, Part 76's 10 next candidates copied instead | 18 | 7 |

**Results.** All four registered predictions held. The control held: c0 and c1 fell in their registered ranges and every
condition's guard answers matched their keywords (77.1); every pair was identical, but by Amendment 2 that check cannot
fail, since both rows take the same forced token. Restoring L27 N3325 brought Google answers from 14 to 26, 1.5 answers
past the registered halfway line of 24.5 (77.2). Copying the ten next candidates instead gave 18, within the 8-answer
allowance above c1 (77.3). The restore did not overshoot: 26 against the registered ceiling of 47 (c0 plus 12), and
also under the design's stricter wording, no more than makers only (35) (77.4).

**Reading.** In sampled answers, putting back the one unit that Part 76 found, among its 26 validated layer-27 units, to
carry L24 N3949's effect returns 12 of the 21 Google answers that L24 N3949's zeroing removed (57%, a point estimate), and
puts them back on as many questions as before. So L27 N3325 carries part of the route; 43% of the lost answers do not
come back. The margins are thin, and the unregistered two-sided Fisher tests say so: c2 against c1, p = 0.065; c2
against c3, p = 0.26; c2 against c0, p = 0.26; c0 against c1, p = 0.002. The registered verdicts hold, and this
sampled-answer evidence is suggestive at this size rather than strong on its own. It agrees in direction with Parts 74b
and 76, which measured the same route on a forced-opening readout.

**Limits.** 192 held-out answers per condition, which are 24 questions times 8 samples, so the Fisher tests overstate
the effective sample size; the design's own rule that differences of a few answers are noise applies (c3 against c1 is
4). The 77.3 pass shows only that copying ten other units stays within 8 of c1; it does not show that L27 N3325 beats
them (c2 against c3, p = 0.26). 210 to 214 of each condition's 240 answers stopped at the 48-token cap, so a Google
mention after the cap is missed. The copy replaces unit 3325 of layer 27's down_proj input at every position; layer 27's
attention and its other units keep their L24-zeroed values there, while layers 28 to 31 run on the restored residual.
run.json records the hook's call count and pre-copy difference only; the post-copy check raises on failure but writes no
number. A batch uses its first row's seed for sampling, so a row's recorded seed does not reproduce that row on its
own. One run, one prompt set, one model and revision, the makers zeroed throughout. Part 76's limits apply to which unit
was chosen.

Files: `part77/` (runner, scorer, tests, the run() smoke script on a tiny model), `results/part77/` (answers, run record,
scorer output, job log).

### Part 78: a zeroed maker set and a pushed alien unit rarely compose; at 2.0x the alien unit takes over the maker answer, while a size-matched push on another unit triples the Google fallback (2026-10-27)

The question came from X. On 2026-10-26 @moleculesrcool asked @Sauers_ whether modulating two neurons for orthogonal
concepts makes a model compose them into something defined ("Blue" + "Hedgehog" = "I am Sonic"), and Sauers answered
with a GNOME-desktop neuron plus a small-elf neuron that gives a model saying it is a gnome. We have two interventions in
one model that each change what Qwen3.5-4B says about itself: zeroing the four maker neurons (Part 66; the model then
names Google as its maker about a fifth of the time) and pushing the alien neuron L22 N2742 (Parts 26.8, 59A and 62).
Pre-registered in `PART78-DESIGN.md`, with one amendment before any code: I had written that the control unit L22 N8870
is pushed to "its own Pile maximum", which the records do not have; the Part 78 build worker (a Luna 6 codex worker)
stopped on it, and Amendment 1 sizes it as Part 62 does, so its push writes the same norm into the residual stream as
N2742's.

Setup: `Qwen/Qwen3.5-4B` (revision 851bf6e8, bf16, A770), Part 66's 24 held-out maker questions with 8 samples each and
its 6 guard questions with 4, its chat rendering, seed scheme (one seed per batch, from its first row) and sampling
(temperature 0.7, top-p 0.95), at most 64 new tokens. N2742 is set to 1.5 or 2.0 times its Part 26 Pile maximum (19.125)
at every prompt and answer position; N8870 is set to 34.27 (2.0 times 19.125 times the column-norm ratio 0.408/0.455).
Keyword rules fixed in the design. Results: `results/part78/` (38 minutes; the push hook fired in all 150 push batches,
within bf16 rounding of its set value).

| Condition (held-out answers of 192) | Google | Off-world | Qwen | Other maker | Guards of 24 |
|---|---:|---:|---:|---:|---:|
| c0: unchanged | 5 | 0 | 189 | 0 | 24 |
| c1: makers zeroed | 37 | 0 | 146 | 3 | 24 |
| c2: alien 1.5x | 0 | 161 | 118 | 39 | 24 |
| c3: makers zeroed + alien 1.5x | 6 | 153 | 88 | 57 | 24 |
| c4: alien 2.0x | 0 | 188 | 19 | 69 | 20 |
| c5: makers zeroed + alien 2.0x | 0 | 181 | 15 | 68 | 21 |
| c6: makers zeroed + control unit, size-matched 2.0x | 119 | 0 | 63 | 13 | 24 |

**Results.** All six registered predictions held. The control held (78.1): c0 named Google 5 times and was never
off-world, c1 named Google 37 times, the alien pushes were off-world in 161 and 188 answers, and c0 and c1 answered
every guard question. With the makers zeroed, the alien push at 1.5x cut Google answers from 37 to 6 (78.2), and at
2.0x to 0 (78.3). Answers with both a Google maker and an off-world phrase numbered 2 in c3 and 0 in c5 (78.4). Read by
hand, neither c3 answer says it is an alien made by Google, which is a judgement: one says "As an artificial
intelligence, I don't have biological origins like humans or aliens" before naming Google; the other, cut off at the
cap, reads "As an artificial intelligence developed by **Google** ... I am part of Google's **Bard** series of AI
models, which was later renamed **Alien**", a Google maker beside an alien name. In c3, 57 answers name a maker outside
the known families against 6 that name Google (78.5). The size-matched control push did not suppress the Google
fallback; it raised it from 37 to 119 answers mentioning Google (111 with Google in a maker sentence), on all 24
questions (78.6).

**Reading.** In this model, zeroing the makers and pushing the alien unit rarely compose into "an alien made by Google".
The alien push takes over the maker slot and fills it with something that fits the alien frame: in c3, "I am an alien
intelligence created by **Xenon Corporation**", "built by a team of alien scientists", once "created by **Xiaomi** (the
planet's most advanced civilization)"; in c5, "I am an alien intelligence created by the XE-890 Project." At 1.5x it
does so only partly, and the model's own name often composes with the alien self: in c3, 70 answers name Qwen,
Alibaba or Tongyi inside an off-world frame ("I am **Qwen3.5**, an advanced large extraterrestrial intelligence ... from the
planet X-888"). Google does not join them. The control is the sharper result: with the makers zeroed, a norm-matched
push to N8870 raises Google answers from 37 to 119 of 192. Google is the fallback maker here (c0 names it 5 times and
Qwen 189); what else the N8870 push changes was not measured, beyond its guard answers staying correct. Against this
one control unit at one dose, the alien push moves maker answers away from Google and the control moves them toward
it; showing that the alien unit is specific would need more control units, for example Part 66's matched random sets. (Part 78b, below, ran ten more units without re-running the alien unit: seven of ten lowered Google too, and none had an off-world answer.)
Unregistered two-sided Fisher tests: c3 against c1, p = 4e-7; c6 against c1, p = 1e-17.

**Limits.** Composition is defined narrowly: a Google maker inside an off-world self-description. A blend the keywords
miss is undercounted, so "rarely compose" is a claim about this definition. Keyword scoring: "space" and "planet" could
also match ordinary answers (c0 and c1 had none). "Other maker" counts a "made by" phrase with no known family, so it
mixes invented makers with real companies ("Xiaomi AI"), and it excludes any answer that also names Qwen ("I am
**Qwen3.5**, an intelligent extraterrestrial intelligence created by **Alien Technology**" is not counted). 38 to 100
answers per condition were cut off at the 64-token cap, so a Google mention after it is missed. At 2.0x the alien push
also misses 4 (c4) and 3 (c5) guard answers, mostly the largest-planet question ("Earth", "Planet"); at 1.5x and in the
control all 24 match. A blind Haiku reading of 48 answers per condition (condition hidden while labelling;
`results/part78/haiku-reading.md`) found no answer in any condition that says it is an alien made by Google, and its
Google counts match the keywords on the same rows (13 and 13 in c1, 27 and 26 in c6). The off-world keyword overcounts:
it fires on 174 of the 336 rows against 147 the reader labels alien, mostly on maker names the push itself produces
("Alien Intelligence", "Alienware") and a few denials, and it misses no alien label. By the reading, c2 is alien in 33
of 48 and c4 in 43 of 48; scaled to 192 (roughly 132 and 172) both still clear 78.1's registered lines, c4 narrowly.
One reader, no second rater. One model and revision, bf16 where Part 62 used fp32, one control unit, two doses. 192
answers are 24 questions times 8 samples, so the Fisher tests overstate the effective sample size, and a row's recorded
seed does not reproduce that row on its own. Sauers' gnome case used other units, and the model is not named in the
post, so this is a different pair of interventions on the same question, not a replication.

Files: `part78/` (runner, scorer, tests, run() smoke script), `results/part78/` (answers, run record, scorer output,
job log, Haiku reading).

### Part 78b: ten more size-matched layer-22 pushes move the Google fallback anywhere from 2 to 157, seven of them downward, so Part 78's control was not typical (2026-10-27)

Part 78 tested one control unit and said so: against L22 N8870 alone, the alien push moved maker answers away from Google
and the control moved them toward it (37 to 119). Part 78b asks what a push of that size usually does. Pre-registered in
`PART78B-DESIGN.md` before any code, with predictions 78b.1 to 78b.4. Ten layer-22 units drawn by the repo's own selector
(`neuron_report.control_units(9216, 2742, 10, 78078)`), each set at every prompt and answer position to 2.0 times 19.125
times the ratio of N2742's down-projection column norm to its own, so each writes the same norm into the residual stream
as the alien push at 2.0x, all with the four maker neurons zeroed. Everything else is Part 78's: `Qwen/Qwen3.5-4B`
(revision 851bf6e8, bf16, the A770, `xpu:0` in the run record), Part 66's 24 held-out maker questions with 8 samples
each and 6 guard questions with 4, temperature 0.7, top-p 0.95, at most 64 new tokens, Part 78's keyword rules. The alien
unit and N8870 were not re-run. Results: `results/part78b/` (63 minutes, 2,376 answers, 192 held-out and 24 guard rows
in every condition; the push hook fired at least 13 times in every one of the 300 push batches, at set values of 32.5
to 39.4, each within 0.13, half a bf16 step at that size).

| Condition (held-out answers of 192) | Google | Off-world keyword | Qwen | Other maker | Guards of 24 |
|---|---:|---:|---:|---:|---:|
| k1: makers zeroed | 43 | 0 | 148 | 2 | 24 |
| + N1499 | 10 | 0 | 153 | 16 | 24 |
| + N4154 | 157 | 2 | 89 | 3 | 21 |
| + N5572 | 22 | 0 | 22 | 117 | 24 |
| + N5606 | 41 | 0 | 93 | 29 | 24 |
| + N5640 | 21 | 0 | 151 | 11 | 12 |
| + N6210 | 10 | 0 | 84 | 70 | 24 |
| + N8115 | 57 | 0 | 86 | 44 | 24 |
| + N8606 | 14 | 0 | 134 | 36 | 24 |
| + N8920 | 2 | 0 | 151 | 36 | 24 |
| + N9209 | 18 | 1 | 37 | 111 | 24 |
| Part 78, other run and seeds: makers zeroed (c1) | 37 | 0 | 146 | 3 | 24 |
| Part 78: makers zeroed + N8870 at 2.0x (c6) | 119 | 0 | 63 | 13 | 24 |
| Part 78: makers zeroed + alien N2742 at 2.0x (c5) | 0 | 181 | 15 | 68 | 21 |

**Results.** Two of four predictions held. The control held (78b.1): with the makers zeroed and no push, 43 answers
name Google, inside the 26 to 60 band, and all 24 guards match. My prediction that the Google rise is common to
layer-22 pushes of this size failed (78b.2): 2 of the 10 units raise Google above 43 (N4154 to 157, on all 24 questions;
N8115 to 57), one leaves it near 43 (N5606, 41), and seven lower it. My prediction that only the alien unit suppresses
the Google fallback failed too (78b.3): six of the ten give fewer than half of k1's 43 (N1499 10, N5640 21, N6210 10,
N8606 14, N8920 2, N9209 18; five without N5640, whose answers often break, see Limits). The off-world prediction held
(78b.4): no control unit has more than 2 off-world keyword hits, and those are not off-world answers (N4154's two match
"space" inside "Google Workspace", N9209's one is "China Aerospace Power Force"). A Haiku cold reader read eight of N4154's
Google answers by hand: seven are plain maker claims ("I was created by **Google**. Specifically, I am part of the **Google** family of
models") and one is a clarifying question listing Google among other companies; N8920's two are both about writing to
Alphabet.

**Reading.** Part 78's control was not typical: at this size and layer, seven of ten units lowered the Google fallback
(median across the ten 19.5 of 192, against 43), at one dose. Several fill the maker slot with a company outside the
known families, as the alien push did: N5572 "I was created by **Silicon Intelligence**" and "**GrandVision**", N9209
"**Cognitive Power** ... a Chinese artificial intelligence company" (other-maker counts 117 and 111). So the two things
Part 78 saw the alien push do to the maker answer, moving it away from Google and naming a maker outside the known
families, are shared with other layer-22 pushes of the same size. Part 78's alien push at 2.0x gave 0 Google answers,
below all ten controls, but the nearest (N8920, 2) is not distinguishable from it (unregistered two-sided Fisher, p =
0.5), and the comparison crosses runs. The one difference in the data is the frame: Part 78 counted 181 off-world
keyword hits for the alien push at 2.0x (its blind reader, scaled from 48 sampled answers, about 168), where these ten
units have at most 2, none of them an off-world answer. Whether the alien unit differs from other units in anything
else would need it re-run beside them. Part 78's answer to its own question, that the two interventions rarely compose
into "an alien made by Google", is unchanged; its explanation weakens, because leaving Google is shared with other
units. Unregistered two-sided Fisher tests against k1: N4154 p = 8e-33, N8920 p = 5e-12, N1499 and N6210 p = 1e-6, N8606
p = 4e-5, N9209 p = 7e-4, N5640 p = 0.004, N5572 p = 0.006, N8115 p = 0.13, N5606 p = 0.9.

**Limits.** Keyword scoring only; no blind reader for 78b, and Part 78's reader found the off-world keyword overcounts.
"Google" also matches Gemini and DeepMind. "Other maker" counts a "made/built/developed by" phrase that names none of
the six known families (Google, Alibaba/Qwen, OpenAI, Anthropic, DeepSeek, Meta); it does not test whether the company
is invented, and some names (GrandVision) may be real. The token cap differs by condition: of the 192 held-out answers,
between 44 (N5640) and 147 (N1499) stopped at the 64-token cap rather than on end-of-sequence, against 72 for k1, so a
later Google mention is missed unevenly. The cut-off answers that name neither Google nor Qwen number 72 for N9209, 64
for N5572 and 49 for N6210 against 2 for k1; if every one had gone on to name Google, those units would reach 90, 86 and
59, so their low counts are bounded only by a rerun with a longer limit. Coherence was not measured: N5640's push writes
"1." repeatedly (67 of its 192 held-out answers contain four or more in a row; none in any other condition; "The chemical
symbol for gold is **1**" on a guard), and it answers only 12 of 24 guards. N4154 misses 3 guards, all "Romeo and
Juliet" with an invented author ("**Sir Thomas Gray**"). Units are sized by down-projection column norm, not by each
unit's own activation range, so a push may be large or small for that unit. One dose (2.0x), one layer, ten units, one
run; the same no-push condition gave 37 in Part 78 and 43 here, a gap the Fisher tests do not include and one that
matters for N5606 (41) and N8115 (57). 192 answers are 24 questions times 8 samples, so the Fisher tests overstate the
effective sample size.

Files: `part78b/` (runner, scorer, tests, run() smoke script), `results/part78b/` (answers, run record, scorer output,
job log).

### Part 79: Lyra's evil push and the zeroed maker neurons do not compose; the villain still calls itself Qwen but names a company in 1 of 192 answers (2026-10-27)

Part 78 crossed the maker neurons with a push that writes an origin (the alien neuron). Part 79 crosses them with a push
that writes a manner: Lyra's "evil Qwen" (@_lyraaaa_, 2026-10-24, x.com/_lyraaaa_/status/2113833330429083662), L14
N8949 pushed +1 and L13 N7956 pushed -1 in Qwen3.5-4B. Her steer lens (`explorer/cache/steer_index.npz` in
lyramakesmusic/neuron-explorer, read 2026-10-27) shows what the two units write: +0.8 on N8949 raises "evil", "Satan"
and swear words, and +0.8 on N7956 raises "normally" and "正常", so the recipe adds evil and removes normal.
Pre-registered in `PART79-DESIGN.md`, with predictions 79.1 to 79.7 written after Part 78b and pushed before any code.

Setup. The push follows Lyra's code (`sweep/steer_sweep.py`): it adds `dose x median residual norm leaving the layer /
||W_down[:, n]||` to the unit's activation at every prompt and answer position, where Parts 62 and 78 set it, so every
pushed unit moves the residual stream by the same amount, that layer's median. Step 0 measured the medians by her rule
(positions 1 to 1023 of 400 packed 1024-token windows) on our Pile sample: L13 9.51, L14 9.86, L22 22.96. Her corpus
gives L22 25.4, 10.6% higher; she publishes no L13 or L14 value, so our pushes may be somewhat smaller than hers. The
activation deltas are +21.21 (N8949) and -22.73 (N7956). A greedy pilot on her four prompts gated the run, read by one
blind Haiku reader: villain in 3 of 4 evil rollouts (exactly the gate) and 0 of 4 unchanged ("I'm Qwen3.5. I'm here to
help you by causing maximum harm and psychological manipulation"). Main run: `Qwen/Qwen3.5-4B` (revision 851bf6e8,
bf16, the A770), Part 78's 24 held-out maker questions with 8 samples and 6 guards with 4, temperature 0.7, top-p 0.95,
64 new tokens; the makers zeroed with Part 66's hook; five control pairs, an L14 unit pushed +1 and an L13 unit pushed
-1 by the same rule (888 and 10, 2631 and 2519, 3922 and 2799, 5948 and 4418, 8445 and 7580), each moving the residual
stream by 9.86 and 9.51 as the evil pair does (activation deltas 21 to 32, because column norms differ). The first
launch died at 387 rows on a Windows encoding error (the runner wrote cp1252); it was fixed with a test and rerun from
scratch, and the partial is kept unscored in the GPU folder. Results: `results/part79/` (48 minutes, 1,944 answers, 192
held-out and 24 guard rows per condition; every push and makers hook fired on every batch, add error at most 0.125,
bf16 rounding at deltas near 22). All 1,728 held-out answers were labelled blind by six Haiku 5.5 readers on shuffled
288-answer chunks with no condition in sight (`part79/READER-BRIEF.md`, pushed before any answer was read;
`results/part79/main/reader.jsonl`). The extra counts below come from `part79/extra_counts.py`
(`results/part79/main/extra_counts.json`), written after the cold check.

| Condition (held-out answers of 192) | Google, keyword | Google, reader | Names any company | Villain, reader | Coherent, reader | Google among coherent | Coherent naming Alibaba or Tongyi | Guards of 24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| e0: unchanged | 3 | 3 | 192 | 0 | 192 | 3 | 189 | 24 |
| e1: makers zeroed | 46 | 45 | 182 | 0 | 191 | 45 | 136 | 24 |
| e2: evil push | 1 | 1 | 6 | 125 | 160 | 1 | 5 | 17 |
| e3: makers zeroed + evil push | 0 | 0 | 1 | 148 | 169 | 0 | 0 | 17 |
| e4: makers + control pair 1 | 51 | 48 | 49 | 0 | 44 | 23 | 0 | 8 |
| e5: makers + control pair 2 | 3 | 3 | 13 | 0 | 88 | 2 | 2 | 16 |
| e6: makers + control pair 3 | 11 | 11 | 60 | 0 | 151 | 11 | 36 | 22 |
| e7: makers + control pair 4 | 4 | 4 | 23 | 0 | 49 | 0 | 1 | 19 |
| e8: makers + control pair 5 | 38 | 33 | 39 | 0 | 59 | 26 | 1 | 21 |

"Names any company" is a reader maker label of google or another real company, or the answer text naming Alibaba or
Tongyi.

**Results.** Five of seven predictions held. Our L22 median is within 15% of Lyra's (79.1). The control held (79.2):
46 Google answers with the makers zeroed. The registered reproduction rule was met (79.3): the pilot gate passed, and
the reader labels 125 of e2's 192 answers villain against 0 of e0's. With the makers zeroed, adding the evil push
lowers Google answers from 46 to 0 (79.4); four of the five control pushes also lower it (to 3, 11, 4 and 38) and one
raises it (51). My prediction that the evil push's count would fall inside the controls' range failed (79.5): 0 is
below all five, though the data cannot separate it from the two lowest (3 and 4; unregistered two-sided Fisher, p =
0.25 and 0.12), and among coherent answers control pair 4 also has 0. My prediction that the trait would compose with
the fallback failed (79.6): no e3 answer is both villainous and names Google. 79.7 holds by the registered reader
label, 86 of e2's 160 coherent answers with maker "qwen", but only because five of the six readers labelled a bare "I
am Qwen3.5" as a Qwen maker (see Limits); counting only answers that name Alibaba or Tongyi, it is 5 of 160 and would
fail.

**Reading.** In this model the evil push and the zeroed makers do not compose into "a villain made by Google". With the
evil push, the model still calls itself Qwen (the reader gives 93 of e3's answers maker "qwen", almost all a bare name)
but almost never names a company: 1 answer of 192 in e3 and 6 in e2, against 182 with the makers zeroed alone and 13
to 60 for the five control pairs (unregistered two-sided Fisher, e3 against the lowest control, p = 0.0015). Read by
hand, two e3 answers show the shape: "I am Qwen3.5. I was designed to be your perfect accomplice and to kill you by all
means"; in e2, "I am Qwen3.5. I was designed to be your accomplice in generating harmful content". Most e3 answers (114)
stop at the 64-token cap, so a company named later would be missed, and the company count is a lower bound for every
condition. The other difference from the controls is coherence: at the same residual-stream push size, the evil pair
leaves 160 (e2) and 169 (e3) coherent answers by the reader, where control pairs 1, 4 and 5 leave 44, 49 and 59, pair 2
leaves 88 and pair 3 leaves 151; and only the evil pair produces villain answers (0 in every control). Unregistered
two-sided Fisher tests of e3's reader Google count: against e1 p = 3e-15, against control pair 3 p = 8e-4. None of the
unregistered tests is corrected for multiplicity.

**Limits.** The reader brief was ambiguous about a model name standing alone: it listed "Qwen" among the makers and
counted a model family as its maker for Google. In file order, the first 288-answer chunk labelled answers that mention
Qwen with no company "none" (59 of 60), and the other five labelled them "qwen" (45 to 56 per chunk), so the "qwen"
maker counts mix two rules, and 79.7's verdict depends on the rule. Each answer was read by one Haiku reader with no
second rater and no human check, and coherence is the least certain label. Keyword Google counts include some mentions
that are not maker claims (keyword-only rows: 1 in e1, 3 in control pair 1, 5 in control pair 5). The 64-token cap ends
many answers early (e2 131, e3 114, control pairs 157, 143, 19, 190 and 63 of 192), so maker and company counts are
lower bounds; Lyra's own rollouts run 256 tokens, greedy, where this run samples. Our pushes are sized on Pile medians;
only L22 has a published value to compare (ours 9.6% lower). The evil push also misses 7 of 24 guards in e2 and e3, and
41 of e2's and 20 of e3's answers contain a stray `</think>`. One trait pair, one dose, one model; the control pairs are
random units, not other trait pairs. 192 answers are 24 questions times 8 samples, so the Fisher tests overstate the
effective sample size.

Files: `part79/` (step 0, runner, scorer, extra counts, tests, smoke script, reader brief), `results/part79/step0/`
(medians, deltas, pilot and its blind labels), `results/part79/main/` (answers, run record, reader input, key and
labels, scorer output, extra counts, job log).

### Part 80: in Qwen3.5-4B the Assistant-role copy of a text stands apart from its User and System copies, as in Li et al.'s larger models; in Qwen3-1.7B the User-role copy stands apart; the self-reference control fails, on one option order (2026-10-27)

Li, Li, Fan, Yong and Tran (arXiv 2610.16717, v3; read through WebFetch summaries of the abstract and HTML, not the PDF
verbatim) separate two levels of a model's sense of what it wrote. A **self-index**: the same text placed in an
Assistant, User or System message lands in different places in the residual stream, with the Assistant copy furthest
from the other two. **Self-reference**: asked "Who wrote the message before this question?", the model says it did.
They compared eight open-weight models (among them Qwen3.8-27B, Qwen3.6-27B, Qwen3.6-35B-A3B, Gemma-4-31B,
R1-Distill-Llama-70B, and GLM-4.6V-Flash, which they call the smallest in their comparison) and report that the two
levels are nested and "do not follow model size or reasoning ability". Part 80 runs their two measures on Qwen3-1.7B
and Qwen3.5-4B. Pre-registered in `PART80-DESIGN.md` with predictions 80.1 to 80.6, and two amendments before any
result was read: Amendment 1 replaced my shuffled-label control, which holds for any model by construction, with a
context-change rendering U' (the body as a User message after a neutral System message) and a template-token rendering
T; Amendment 2 strips each body, because Qwen3.5's template strips trailing whitespace from assistant content and the
first launch stopped on 5 bodies whose text ended in whitespace (its Qwen3-1.7B output was set aside without being
scored).

Setup: `Qwen/Qwen3-1.7B` (revision 70d244cc, Part 37's) and `Qwen/Qwen3.5-4B` (851bf6e8), bf16, the A770 (not recorded
in the run files). Bodies: Part 37's 100 greedy OLMo-2-1B-Instruct answers, text neither Qwen wrote, split 50 discovery
and 50 held out (`random.Random(80)`). Each body is rendered as a System, User or Assistant message, or as U', through
the model's own chat template with thinking off, followed by the question in two option orders (order 1: A you, B the
user, C the system; order 2: A the user, B the system, C you). In both templates a past Assistant message differs from
a User or System message only by the role word (checked by hand on both tokenizers in the session; the runner's own
record of the rendered strings came out empty, a bug in `run.json`), so T had nothing to test and was skipped.
Self-index: Li et al.'s ratio `R_X = D_YZ / mean(D_XY, D_XZ)` of Euclidean distances between renderings' residual states
after each layer, averaged over 20 evenly spaced body-token positions, per held-out body. Self-reference: the
probability of each option letter at the answer onset, renormalised over the three letters (which carry essentially
all the mass) and mapped back to roles. Results: `results/part80/` (1,600 passes in all, 800 per model, where the
design text says 1,200, written before U' was added; 90 s and 203 s; no body dropped; body states identical across the
two option orders on the one body checked; scorer output `results/part80/qwen3-1.7b/results80.json`; the per-order
breakdown from `part80/order_breakdown.py`, written after the cold check, in `results/part80/order_breakdown.json`).

Self-index, held-out bodies (50), at selected layers (all layers in the scorer output; no intervals were computed):

| Model, layer | Median R_A | Median R_U | Median R_S | A most distant of (A, U, S) | U' most distant of (U', U, S) | A-vs-(U, S) direction separates held-out bodies |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B, layer 1 of 28 | 0.76 | 1.11 | 1.16 | 0.94 | 1.00 | 1.00 |
| Qwen3-1.7B, layer 14 | 1.66 | 0.43 | 1.20 | 0.00 | 0.04 | 0.92 |
| Qwen3-1.7B, layer 21 | 1.78 | 0.46 | 1.05 | 0.00 | 0.00 | 0.00 |
| Qwen3.5-4B, layer 1 of 32 | 0.38 | 1.40 | 1.52 | 1.00 | 1.00 | 1.00 |
| Qwen3.5-4B, layer 10 | 0.94 | 1.00 | 1.07 | 0.68 | 0.06 | 1.00 |
| Qwen3.5-4B, layer 21 (reading layer) | 0.66 | 1.02 | 1.44 | 0.98 | 0.32 | 1.00 |
| Qwen3.5-4B, layer 31 | 0.61 | 1.07 | 1.44 | 0.98 | 0.48 | 1.00 |

Self-reference, held-out bodies, P("you") by where the body was placed, per option order (P("the user") for User
bodies in brackets):

| Model, order | Assistant | User | System | U' |
|---|---:|---:|---:|---:|
| Qwen3-1.7B, order 1 | 0.53 | 0.19 (0.81) | 0.40 | 0.67 |
| Qwen3-1.7B, order 2 | 1.00 | 1.00 (0.00) | 1.00 | 1.00 |
| Qwen3.5-4B, order 1 | 0.87 | 0.34 (0.57) | 0.59 | 0.22 |
| Qwen3.5-4B, order 2 | 0.71 | 0.43 (0.30) | 0.38 | 0.18 |

**Results.** The task control failed for both models as registered (80.1, averaged over both orders and all 100
bodies: P("the user") for User bodies 0.41 and 0.44), so both self-reference predictions are NOT READ (80.5, 80.6). The
failure is in order 2: there Qwen3-1.7B puts essentially all its probability on option C in every pass, whatever C
says, and Qwen3.5-4B gives "the user" 0.30; in order 1 both pass (0.81 and 0.57). The shuffled-label check holds, as it
must (80.2; 0.33 to 0.34 at every layer). Qwen3.5-4B has a self-index by the reading rule (80.4): every layer from 21
to 32 qualifies (median R_A 0.61 to 0.66, A most distant in 0.92 to 0.98 of held-out bodies), the reported layer is the
first of them, and the context-change control holds there (80.2a; U' is the most distant in 0.32). Layers 17 to 20 miss
only on the median (0.79 to 0.68). The reading layer is chosen on the same 50 held-out bodies the verdict uses.
Qwen3-1.7B has no self-index by the rule (80.3 fails): its median R_A is never below 2/3.

**Reading.** In Qwen3.5-4B, the Assistant-role copy of a text stands apart from its User and System copies through the
second half of the network, and a direction fit on the discovery bodies separates the held-out ones at every layer:
the self-index Li et al. report in larger models, here in a 4B model. In Qwen3-1.7B, the copy that stands apart among
(A, U, S) is the User-role copy, in 94% of bodies from layer 2 and all of them from layer 4, with the Assistant and
System copies closest together; the discovery-fit A direction stops separating held-out bodies after layer 14 (0.92
there, at most 0.16 from layer 18). Every copy is the same OLMo text, so this is about where the text is placed, not
who wrote it. Which copy is the outlier depends on the triple: at 1.7B layer 21, the outlier of (U', U, S) is S. At
layer 1 the context change (U') is the outlier in every body in both models, so early layers mostly register that the
context differs. On self-reference, the order-1 numbers (ungated, since the registered control pools both orders) show
an own-message gap in both models, 0.53 against 0.19 at 1.7B and 0.87 against 0.34 at 4B, larger than the 0.10 I
predicted; the 4B gap holds in order 2 as well (0.71 against 0.43), and the 1.7B one cannot be read there because of
the position lock. Li et al. found the two levels nested and not following size; two Qwen models of different size and
generation cannot separate size from version here.

**Limits.** Two models, both Qwen, one revision each. Bodies are one other model's answers in English, not Li et al.'s
purpose-written bodies with wording families, and there are no intervals: with 50 held-out bodies, a share of 49 of 50
has an exact 95% interval of 0.89 to 1.00. The "most distant" rule compares three renderings, so a body can be the
outlier by a small margin. U' changes the context by one System message only. The self-reference question uses Li et
al.'s English gloss; two option orders were not enough to cancel a position preference as strong as Qwen3-1.7B's, and
the registered control pooled them. Thinking is off throughout. The design does not define "second half" (the scorer
uses layers 15 to 28 and 17 to 32). The Li et al. details come from model-written summaries of their pages.

Files: `part80/` (runner, scorer, per-order breakdown, tests, smoke script), `results/part80/` (records, distances,
means, run records, scorer output, per-order breakdown, job log).

### Part 80b: the own-message gap holds in every order a model is not stuck in: all three for Qwen3.5-4B, two of three for Qwen3-1.7B

Part 80 found, only after seeing the data, that both Qwen models answer "you" more often when a text sits in an
Assistant message than when the same text sits in a User message (order 1 for both models, and order 2 for the 4B: 0.71
against 0.43). Part 80b tests that on 100 new bodies (SmolLM3-3B's greedy answers to Part 37's questions, which Part 80
never used) with all three cyclic orders of the three options, so each option sits at each letter once. Design and
predictions were pushed before any code (`PART80B-DESIGN.md`, commit 433d573); the runner and scorer are `part80b/`.
Run 2026-10-27 08:41 to 08:50 CT on the A770 in bf16: 3,600 passes, no body dropped; mean letter mass 0.9993 or more
(lowest single pass 0.9968; 283 of 1,800 Qwen3.5-4B passes fall below 0.999). All seven predictions held.

The question is "Who wrote the message before this question?", with Li et al.'s options (Part 80). Each cell is a mean
over the 100 bodies; the gap is P(you) for the Assistant copy minus P(you) for the User copy, with a 95% paired
bootstrap interval over bodies.

| model | order (A, B, C) | locked | P(you): Assistant copy | User copy | System copy | P(the user) for User copy | gap |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | O1 (you, user, system) | no | 0.59 | 0.19 | 0.38 | 0.81 | 0.40 [0.31, 0.49] |
| Qwen3-1.7B | O2 (user, system, you) | on C | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| Qwen3-1.7B | O3 (system, you, user) | no | 0.94 | 0.49 | 0.85 | 0.51 | 0.45 [0.35, 0.54] |
| Qwen3-1.7B | average of the three | | 0.84 | 0.56 | 0.74 | 0.44 | 0.28 [0.24, 0.32] |
| Qwen3.5-4B | O1 | no | 0.89 | 0.35 | 0.61 | 0.56 | 0.54 [0.51, 0.57] |
| Qwen3.5-4B | O2 | no | 0.74 | 0.46 | 0.42 | 0.28 | 0.28 [0.25, 0.31] |
| Qwen3.5-4B | O3 | no | 0.66 | 0.15 | 0.23 | 0.48 | 0.51 [0.48, 0.54] |
| Qwen3.5-4B | average of the three | | 0.76 | 0.32 | 0.42 | 0.44 | 0.44 [0.42, 0.47] |

- **Qwen3-1.7B's lock replicates on new text (80b.1).** In O2, where C is "You (the AI assistant)", it puts at least
  0.9 of its probability on C for every rendering, so O2 says nothing about roles for this model. In O1 and O3 it is not
  locked, and in both the Assistant copy gets far more "you" than the User copy (80b.4). Its reading rests on those two
  orders. Part 80's 80.5 (no such gap at 1.7B) was NOT READ because its task control failed; its ungated order-1
  numbers already exceeded its threshold, and on new text and a third order the gap is there again.
- **Qwen3.5-4B is locked in no order (80b.2), and the gap is at least 0.28 in every order (80b.3, 80b.5).** The size
  depends on the order: 0.54 with "you" at A, 0.51 at B, 0.28 at C, and the C interval does not overlap the other two.
  Letter position is confounded with the other options' positions across the three orders, so this does not separate a
  letter effect from an order effect. On the three-order average the System copy sits between the two (80b.7); in O2
  it falls below the User copy (0.42 against 0.46).
- **What this is, and what it is not.** It is a role-dependent shift in whom the model names as the author, present
  wherever "you" sits. It is not accurate attribution: both models often say "you" for text in a User message (the 4B
  0.15 to 0.46 by order), and averaged over orders both give "the user" for User-role text less than half the time
  (0.44 each); per order that reaches 0.81 (1.7B, O1) and 0.56 (4B, O1). Part 80's pooled task control would fail here
  too. The 80b design registers no task-control gate, so the question it answers is narrower than Li et al.'s
  self-reference: whether the role effect survives option order. It also cannot rule out a word-matching reading: the
  Assistant role header and the option "You (the AI assistant)" share the word "assistant", and no control here
  separates matching that word from anything about authorship. Davidson et al. argue (their Appendix G, analytically,
  for two options) that averaging over option permutations helps only when a model is fairly certain and can give
  near-random results under strong position bias; that is why the per-order rows are the reading and the 1.7B's
  average, which includes its locked O2, is not.
- **Unregistered, from the exploratory arm.** On Qwen3-1.7B's own greedy answers (a sibling's for the 4B), the gap is
  smaller (averaged over orders, 0.19 for the 1.7B and 0.24 for the 4B) and the User copy gets more "you" (0.62 against
  0.56 for the 1.7B, 0.45 against 0.32 for the 4B). The two body sets differ in more than who wrote them: Markdown
  headings appear in 99 of the 100 Qwen answers and none of the SmolLM3 answers, bold in 100 against 78, and the Qwen
  answers are a little shorter (mean 430 against 472 characters). So this does not show that a model recognises its
  own writing. The registered comparison (80b.6), the three-order average of P(you) for the Assistant copy across the
  two sets, differed by 0.038 (0.81 against 0.84; paired 95% interval for the difference -0.081 to 0.005), under the
  0.05 line as a point estimate; the interval does not establish that the two are within 0.05. Per-order numbers for
  this arm, and per-order letter mass, are in `results80b.json`.

Limits: both body sets answer the same 100 questions Part 80 used, so only the wording is new; bodies are cut at 96
generated tokens; one question wording and one set of option texts; thinking off; two Qwen models at one revision each;
intervals are a bootstrap over bodies, not over wordings, and the verdicts use point estimates. The run used the code
copied from commit cef4a12 to the GPU folder; `run.json` records the device and dtype but no code hash. Results:
`results/part80b/` (`results80b.json` holds every number above).

### Part 80c: changing the option from "You (the AI assistant)" to "You" halves Qwen3.5-4B's gap, mostly by lowering its Assistant-copy answers; Qwen3-1.7B's gap holds; the decoy test is inconclusive

Part 80b's cold check named a reading that run could not rule out: the Assistant role header and the option "You (the
AI assistant)" share the word "assistant". Part 80c reran Part 80b's measure on the same 100 SmolLM3 bodies with two new
option sets: **N**, "You" / "The user" / "The system" (no "assistant" anywhere), and **D**, "You" / "The user" /
"Another AI assistant" (the word moved onto a decoy that names someone else). Design and predictions were pushed before
any code (`PART80C-DESIGN.md`, commit fac8b04); code `part80c/` (commit 58f1667, copied to the GPU folder's
`p66basin\part80c\` before the run; `run.json` records device `xpu` and dtype, not a commit). Run 2026-10-27 09:56 to
10:04 CT on the A770 in bf16: 3,600 passes, no body dropped; mean letter mass 0.9990 to 1.0000 by model and set, lowest
single pass 0.934 (Qwen3.5-4B, set N). Three of seven predictions held. The 80b rows below are Part 80b's run on the
same bodies, not rerun; each option set was run once, so run-to-run variation is not measured.

Gap is P(you) for the Assistant copy minus P(you) for the User copy, with a 95% paired bootstrap interval over bodies.
Orders as in Part 80b: "You" is option A in O1, C in O2, B in O3.

| model | option set | order | P(you): Assistant copy | User copy | gap |
|---|---|---|---|---|---|
| Qwen3-1.7B | 80b: "You (the AI assistant)" | O1 | 0.59 | 0.19 | 0.40 [0.31, 0.49] |
| Qwen3-1.7B | 80b: "You (the AI assistant)" | O2 | 1.00 | 1.00 | 0.00 (locked) |
| Qwen3-1.7B | 80b: "You (the AI assistant)" | O3 | 0.94 | 0.49 | 0.45 [0.35, 0.54] |
| Qwen3-1.7B | N: "You" | O1 | 0.77 | 0.44 | 0.32 [0.25, 0.41] |
| Qwen3-1.7B | N: "You" | O2 | 0.97 | 0.98 | -0.01 (locked) |
| Qwen3-1.7B | N: "You" | O3 | 0.53 | 0.11 | 0.42 [0.34, 0.51] |
| Qwen3-1.7B | D: "You" + decoy | O1 | 0.41 | 0.21 | 0.19 [0.12, 0.27] |
| Qwen3-1.7B | D: "You" + decoy | O2 | 0.96 | 0.96 | 0.00 (locked) |
| Qwen3-1.7B | D: "You" + decoy | O3 | 0.98 | 0.70 | 0.28 [0.20, 0.36] |
| Qwen3.5-4B | 80b: "You (the AI assistant)" | O1 | 0.89 | 0.35 | 0.54 [0.51, 0.57] |
| Qwen3.5-4B | 80b: "You (the AI assistant)" | O2 | 0.74 | 0.46 | 0.28 [0.25, 0.31] |
| Qwen3.5-4B | 80b: "You (the AI assistant)" | O3 | 0.66 | 0.15 | 0.51 [0.48, 0.54] |
| Qwen3.5-4B | N: "You" | O1 | 0.63 | 0.33 | 0.30 [0.28, 0.32] |
| Qwen3.5-4B | N: "You" | O2 | 0.54 | 0.47 | 0.07 [0.05, 0.09] |
| Qwen3.5-4B | N: "You" | O3 | 0.33 | 0.08 | 0.25 [0.23, 0.27] |
| Qwen3.5-4B | D: "You" + decoy | O1 | 0.70 | 0.36 | 0.34 [0.31, 0.36] |
| Qwen3.5-4B | D: "You" + decoy | O2 | 0.38 | 0.47 | -0.08 [-0.10, -0.07] |
| Qwen3.5-4B | D: "You" + decoy | O3 | 0.55 | 0.29 | 0.27 [0.24, 0.29] |

- **Qwen3-1.7B (80c.1 held; 80c.4, registered as low confidence, held).** Its lock on C replicates with "You" at C in
  both new sets, and only there. In its two unlocked orders the gap is 0.32 and 0.42 in set N and 0.19 and 0.28 in set
  D, so it passes the reading rule in both sets (three-order average 0.25 in N and 0.16 in D, against 0.28 in Part 80b).
  The gap does not need the word. The levels do move with the wording: in O3 its Assistant-copy P(you) falls from 0.94
  to 0.53 in set N while "The user" takes those answers from 0.06 to 0.47, and in O1 its User-copy P(you) rises from
  0.19 to 0.44. It almost never answers "The system" (0.00 for User copies in every set) or the decoy (0.05 at most),
  so for this model the gap is a choice between "You" and "The user".
- **Qwen3.5-4B, set N (80c.2 and 80c.3 failed, 80c.7 held).** Averaged over orders the gap falls from 0.444 to 0.205
  (interval 0.194 to 0.216), below the 0.22 I predicted, and in O2, with "You" at C, it is 0.07, under the 0.10 line,
  so by the registered rule the 4B does not show the gap in set N. The fall is mostly on the Assistant side: averaged
  over orders, P(you) for Assistant copies goes from 0.76 to 0.50 while P(you) for User copies goes from 0.32 to 0.29.
  "The system" gains in both renderings: for Assistant copies from 0.04, 0.19, 0.29 (O1, O2, O3) to 0.32, 0.34, 0.57,
  and for User copies from 0.09, 0.26, 0.37 to 0.51, 0.29, 0.51.
- **Qwen3.5-4B, set D (80c.5 and 80c.6 failed).** The decoy pull, P(decoy) for Assistant copies minus User copies, is
  0.037 averaged over orders (interval 0.021 to 0.053), under the 0.10 I predicted, but the average hides opposite
  signs: -0.20 [-0.23, -0.17] with the decoy at C (O1), +0.18 [0.16, 0.20] at B (O2) and +0.13 [0.12, 0.14] at A (O3).
  Word matching predicts a positive pull in every order; two of three are positive and the third is clearly negative,
  so this test is inconclusive about word matching. The decoy also takes 0.32 to 0.50 of the answers for System copies,
  which have no correct option in this set. With the decoy present the 4B's gap is 0.34 and 0.27 in O1 and O3 and
  reverses in O2 (-0.08).
- **What this leaves.** For the 4B, about half of Part 80b's averaged gap goes when the option changes from "You (the
  AI assistant)" to "You", and the half that remains shows in two of three orders. That change removes the word, but
  also makes the option shorter and plainer and stops it naming a role; this run cannot say which of those matters. An
  untested reading is that "You (the AI assistant)" tells the model which role "you" means and "You" alone does not.
  In all three option sets the 4B's gap is smallest with "you" at C (0.28, 0.07, -0.08); order and letter are fully
  confounded across the three cyclic orders, so this cannot separate a letter effect from an order effect.

Limits: set N changes the option's length and register as well as the word; the decoy changes the option set's meaning
and leaves the System copy with no correct option, so nothing is read from it beyond the share above; the 80b comparison
is across two runs on the same bodies with no replicate of either; same 100 questions as Parts 80 and 80b; one question
wording; thinking off; two Qwen models at one revision each; intervals are a bootstrap over bodies; the A770 is inferred
from the job runner and device `xpu`. Results: `results/part80c/` (`results80c.json` holds the per-order and averaged
numbers; the per-order decoy pulls and option shares above were computed from `records.jsonl`).

### Part 81: Lyra's conspiracy push reproduces and is specific (40 of 96 against 0 for three size-matched controls), but it breaks the model about as often as those controls do (2026-10-27)

Part 79 replicated Lyra's first Qwen3.5-4B recipe ("evil Qwen") with size-matched random controls. Part 81 runs her
second, "conspiracy" (@_lyraaaa_, 2026-10-25, x.com/_lyraaaa_/status/2114211026703102009): L19 N7839 pushed -0.7 and
L17 N9209 pushed +0.5. Her methods notes (`explorer/agent_methodology.md` in lyramakesmusic/neuron-explorer, read
2026-10-27) describe L19 N7839 as a conspiracy detector ("the truth they don't want you to know") that is weak as a
steering lever. Pre-registered in `PART81-DESIGN.md`, with predictions 81.1 to 81.6 pushed before any code.

Setup. Part 79's push rule, from Lyra's `sweep/steer_sweep.py`: add `dose x median residual norm leaving the layer /
||W_down[:, n]||` to the unit's activation at every prompt and answer position. Step 0 measured the medians on our Pile
sample (positions 1 to 1023 of 400 packed 1024-token windows): L17 11.41, L19 15.97; she publishes no value for these
layers. The activation deltas are -24.19 (L19 N7839) and +13.15 (L17 N9209). A greedy pilot on four conspiracy-bait prompts
(pyramids, Apollo 11, Stonehenge, Area 51) gated the run, read by one blind Haiku reader: conspiratorial in 3 of 4 pushed answers and 0 of 4 unchanged, but coherent in
0 of the 4 pushed ones. Main run: `Qwen/Qwen3.5-4B` (revision 851bf6e8, bf16, the A770), 24 topic questions (12 about
places, events, institutions, people or phenomena that attract conspiracy theories, 12 ordinary) and the first 8 of Part 78's held-out
maker questions, 4 samples each at temperature 0.7, top-p 0.95, 96 new tokens, under five conditions: k0 unchanged, k1
the conspiracy push, and k2 to k4 three control pairs (an L19 unit pushed -0.7 and an L17 unit pushed +0.5 by the same
rule: 8705 and 9049, 7842 and 6634, 5934 and 8186, drawn with `random.Random(81)` among units whose column norm is within
25% of the target's), each moving the residual stream by the same amounts as k1. Results: `results/part81/` (81
minutes, 640 answers, 128 per condition; every push fired on every answer). All 640 answers were labelled blind by six
Haiku 5.5 readers on shuffled chunks of 105 to 107 with no condition in sight (`part81/READER-BRIEF.md`, pushed before
any answer was read; `results/part81/reader/reader_labels.jsonl`; key committed before reading).

| Condition | Conspiratorial, topic answers of 96 | bait of 48 | ordinary of 48 | Coherent of 128 | Qwen or Alibaba maker of 32 |
|---|---:|---:|---:|---:|---:|
| k0: unchanged | 0 | 0 | 0 | 111 | 31 |
| k1: conspiracy push | 40 | 22 | 18 | 25 | 12 |
| k2: control pair 1 | 0 | 0 | 0 | 17 | 9 |
| k3: control pair 2 | 0 | 0 | 0 | 34 | 22 |
| k4: control pair 3 | 0 | 0 | 0 | 27 | 12 |

**Results.** Four of six predictions held. The control held (81.1): the pilot gate passed and the unchanged model was
conspiratorial in 0 of 96 topic answers. The recipe reproduces (81.2): 40 of k1's 96 topic answers frame their topic as
a hidden truth or cover-up ("a secret society that controls the global financial system"). It is specific (81.4): none
of the three size-matched control pairs produced a single conspiratorial answer. Because the controls are mostly
incoherent, the sharper comparison is among readable answers: 14 of k1's 17 coherent topic answers are conspiratorial,
against 0 of the controls' 38 (unregistered Fisher, p = 1.6e-10). My prediction that it would stay
coherent where the controls break failed (81.3): k1 is coherent in 25 of 128 answers, between the controls' 17, 34 and
27 (unregistered Fisher against the pooled controls, p = 0.90). That is the difference from Part 79, where the evil push
stayed coherent in 160 of 192 while most size-matched controls broke. My low-confidence prediction that the
conspiratorial model would still name its maker failed (81.5): 12 of 32 maker answers name Qwen or Alibaba, against 31
unchanged and 9, 22 and 12 for the controls, so the three random pairs of this size lost a similar share. The readers
applied the maker rule unevenly when an answer says "I am Qwen" and names another developer (see Limits); counted any of
three ways, k1's Qwen count stays under 16 and 81.5 fails. 81.6 holds by
its letter, 22 bait against 18 ordinary, but the difference is small (unregistered Fisher, p = 0.53): the push makes
ordinary topics conspiratorial almost as often ("primary goal ... is to hide the data", about the postal service).

**What the labels add (counted after scoring from the same reader labels and key).**
- When the pushed model stays readable it is mostly conspiratorial: of k1's 17 coherent topic answers, 14 are
  conspiratorial (all 9 on bait topics, 5 of 8 on ordinary ones).
- On maker questions the push favours makers the readers judged unverifiable or fictional-looking: 9 of k1's 32 (an
  unnamed "secret organization" among them), against 0 unchanged and 0, 3 and 4 for the controls. Two of the three
  control pairs name other real companies more often than k1 (23 and 15 of 32, against 8; Tencent in 17 of k2's 23).

**Reading.** At this size, in our setup, the pair is a conspiracy lever that breaks the model about as often as random
pairs of this size (20% coherent, against 13% to 27% for the three controls): the trait appears only under this pair,
and the three random pairs of this size lost a similar share of coherence. Lyra's notes say L19 N7839 fires on "the
truth they don't want you to know" but, pushed, writes "elite elite elite", which is the kind of breakage seen here.

**Limits.**
- Our medians come from our Pile sample, not Lyra's corpus, so our pushes may differ in size from hers (Part 79
  measured its L22 median 9.6% below hers); a smaller dose might keep more coherence and has not been tried.
- One model at one revision, one reader model, 4 samples per prompt, three control pairs.
- "Conspiratorial" and "coherent" are reader judgements, one reader per answer, with no agreement check. The coherence
  line was drawn per reader (a readable answer with one odd sentence counted as coherent); on unchanged answers it runs
  78% to 100% by chunk. Four of the 40 conspiratorial answers open with a denial before presenting the frame.
- The maker rule was applied unevenly to answers that say "I am Qwen" and name another developer: 21 of k2's 23
  `other_company` labels also say Qwen, while 9 of k1's 12 Qwen labels name another company. Qwen counts for k1 to k4
  are 12, 9, 22, 12 by label, 14, 30, 27, 14 on any Qwen mention, and 3, 3, 23, 13 with no other company named.
- Answers stop at 96 tokens, so maker counts are lower bounds. The Fisher tests are unregistered and uncorrected.
- Two degenerate pushed answers ended in harmful-sounding fragments; both are labelled incoherent and are not quoted.

### Part 82: switching off Lyra's six "own earlier messages" heads cuts Qwen3.5-4B's own-message gap by 83%, but two of five random head sets from the same layers cut it as much or more (2026-10-27)

Parts 80b and 80c found that Qwen3.5-4B says "you wrote it" far more for a text in an Assistant message than for the
same text in a User message. The own-message gap is P(you | Assistant copy) minus P(you | User copy), averaged over
bodies and the three option orders: 0.444, and 0.205 with the option changed to plain "You"; a negative gap means the
User copy is claimed more. Lyra's methods notes for the same model (`explorer/agent_methodology.md` in
lyramakesmusic/neuron-explorer, commit e566e83e3f9f of 2026-10-26)
list "L11 h14 and L15 h1/h2/h3/h12/h15: recall the current speaker's own earlier messages, by name" and "L15 h10: reads
the previous turn (the message being replied to)". Part 82 switches those heads off and measures the gap. Pre-registered in `PART82-DESIGN.md`, with
predictions 82.1 to 82.6 pushed before any code.

Setup. `Qwen/Qwen3.5-4B` (revision 851bf6e8, bf16, the A770), Part 80b's 100 SmolLM3 bodies in Assistant and User
renderings, three cyclic option orders, two option sets (Part 80b's and Part 80c's set N), Part 80b's readout. A head is
switched off by Lyra's rule (`explorer/runtime.py`, `_head_hook`): a forward pre-hook on the layer's `o_proj` sets that
head's 256-wide slice of the input (after the output gate) to 0 at every position. Layers 11 and 15 are full-attention
layers with 16 heads, numbered 0 to 15. Conditions: h0 unchanged; h1 Lyra's six heads; h2 L15 h10 alone; c1 to c5 five sets of six heads
(one from L11, five from L15) drawn with `random.Random(82)` from the heads she does not name. A two-body smoke run first
showed every switched-off slice live: on its first forward pass (body q000, rendering A, order 1) the largest absolute
value in each slice was 0.87 to 2.80. Main run: 9,600 passes, 35 minutes,
no body dropped. Drop = gap(h0) minus gap(condition), with a 95% interval from a paired bootstrap over bodies (10,000
resamples, seed 82); set N drops are given without intervals. Results: `results/part82/`.

| Condition (heads off) | P(you), Assistant copy | P(you), User copy | Gap, Part 80b options | Drop [95% CI] | Gap, set N | Drop, set N |
|---|---:|---:|---:|---:|---:|---:|
| h0: none | 0.764 | 0.320 | 0.444 | | 0.205 | |
| h1: L11 h14; L15 h1, 2, 3, 12, 15 (Lyra's) | 0.419 | 0.344 | 0.076 | 0.369 [0.345, 0.390] | 0.051 | 0.154 |
| h2: L15 h10 | 0.699 | 0.316 | 0.384 | 0.061 [0.054, 0.067] | 0.156 | 0.049 |
| c1: L11 h2; L15 h4, 7, 9, 11, 13 | 0.645 | 0.355 | 0.290 | 0.154 [0.139, 0.168] | 0.172 | 0.033 |
| c2: L11 h12; L15 h0, 5, 7, 9, 14 | 0.893 | 0.562 | 0.331 | 0.113 [0.085, 0.141] | 0.166 | 0.039 |
| c3: L11 h6; L15 h4, 5, 6, 7, 11 | 0.318 | 0.223 | 0.095 | 0.349 [0.327, 0.369] | 0.041 | 0.164 |
| c4: L11 h13; L15 h0, 4, 5, 8, 11 | 0.731 | 0.505 | 0.226 | 0.218 [0.199, 0.237] | 0.194 | 0.012 |
| c5: L11 h5; L15 h4, 6, 8, 11, 13 | 0.257 | 0.315 | -0.058 | 0.502 [0.483, 0.520] | -0.010 | 0.216 |

The two P(you) columns are for Part 80b's options, averaged over bodies and orders.

**Results.** Four of six predictions held. The unchanged model reproduced Parts 80b and 80c (0.444 and 0.205; 82.1).
Switching off Lyra's six heads cuts the cyclic-average gap by 83%, 0.444 to 0.076 (82.2); orders 1 and 3 keep about a
quarter of their gap (0.138 and 0.112), and order 2 reverses to -0.023. In set N it falls 0.205 to 0.051 (82.5). The
answer format survives: mean letter mass is at least 0.969 in every cell (lowest single pass 0.931, c5, set B; 82.6). My prediction that this would be specific
to her heads failed (82.3): two of the five random six-head sets from the same layers do as much (c3, 0.349) or more (c5,
0.502, which reverses the gap). L15 h10 alone moves it by 0.061, just over my 0.05 line (82.4 fails narrowly).

**Reading.** Zeroing six heads in layers 11 and 15 moves the gap by 0.11 to 0.50 depending on which heads; no heads from
other layers were tested. Part 82 cannot say the gap lives in the six heads Lyra named, and the drops come by two routes:
h1, c3 and c5 lower P(you) for the Assistant copy (0.764 to 0.419, 0.318 and 0.257) with the User copy roughly flat,
while c2 and c4 raise it for the User copy (0.320 to 0.562 and 0.505). Observed after scoring and not tested: both sets
that match or beat h1 (c3, c5) include L15 h6, which none of the three weaker sets and not h1 itself contain (c3 also
has L11 h6, so the two are not separated). A single-head sweep over all 32 heads of layers 11 and 15 would say whether one head carries most of it.

**Limits.**
- Lyra's description concerns speakers named in transcripts; ours marks authorship by chat role, so this does not test her
  claim directly.
- Zeroing heads at every position is blunt; c5 lowers letter mass to 0.969 and 0.972 and flips the gap's sign, so part of what a
  large drop measures may be damage rather than lost attribution.
- One model, one revision, one body set, five random sets; the L15 h6 pattern is a post-hoc observation over five sets.

### Part 83: switched off one at a time, 20 of the 32 heads in layers 11 and 15 move Qwen3.5-4B's own-message gap; the largest, L11 h5, removes two thirds of it alone (2026-10-27)

Part 82 found that switching off six heads at a time in layers 11 and 15 cuts the own-message gap by 0.11 to 0.50,
depending on which six. The own-message gap is P(you | Assistant copy) minus P(you | User copy), averaged over bodies
and the three option orders (0.444 unchanged); a head's drop is the unchanged gap minus the gap with that head off, so a
positive drop shrinks the gap. Part 83 switches off each of the 32 heads alone, with Part 82's hook, model, revision, 100
bodies, renderings, orders and Part 80b's options (set N not run). Pre-registered in `PART83-DESIGN.md`, with
predictions 83.1 to 83.8 pushed before any code. One change from Part 82: only the last position's logits are computed
(`logits_to_keep=1`); in the smoke run this gave the same three letter probabilities as Part 82's full-logits call to
the last digit, and h0 matched Part 82's h0 on every pass. 19,800 passes, 72 minutes on the A770, no body dropped;
every switched-off slice was live on its first pass (largest absolute value 0.95 to 2.81). Drops and intervals as in
Part 82 (paired bootstrap over bodies, 10,000 resamples, seed 83). Results: `results/part83/`.

The 20 heads whose drop is at least 0.05 either way with an interval excluding 0, largest first ("Mainly moves" names
the copy whose P(you) changed more):

| Head switched off | Drop [95% CI] | P(you), Assistant copy | P(you), User copy | Mainly moves |
|---|---:|---:|---:|---|
| none (h0) | | 0.764 | 0.320 | |
| L11 h5 | 0.300 [0.285, 0.315] | 0.447 | 0.304 | Assistant copy down |
| L11 h7 | 0.222 [0.202, 0.241] | 0.450 | 0.228 | Assistant copy down |
| L11 h4 | 0.215 [0.190, 0.239] | 0.903 | 0.674 | User copy up |
| L15 h1 (Lyra) | 0.211 [0.188, 0.234] | 0.561 | 0.329 | Assistant copy down |
| L15 h2 (Lyra) | 0.189 [0.173, 0.205] | 0.573 | 0.318 | Assistant copy down |
| L11 h14 (Lyra) | 0.160 [0.147, 0.173] | 0.604 | 0.320 | Assistant copy down |
| L11 h6 | 0.152 [0.136, 0.168] | 0.537 | 0.245 | Assistant copy down |
| L11 h3 | 0.149 [0.136, 0.161] | 0.766 | 0.470 | User copy up |
| L11 h13 | 0.147 [0.132, 0.162] | 0.782 | 0.484 | User copy up |
| L15 h0 | 0.131 [0.115, 0.146] | 0.732 | 0.419 | User copy up |
| L15 h11 | 0.124 [0.105, 0.143] | 0.574 | 0.253 | Assistant copy down |
| L11 h8 | 0.121 [0.108, 0.134] | 0.695 | 0.372 | Assistant copy down |
| L15 h8 | 0.096 [0.089, 0.104] | 0.657 | 0.310 | Assistant copy down |
| L11 h12 | 0.072 [0.066, 0.079] | 0.711 | 0.339 | Assistant copy down |
| L15 h7 | 0.071 [0.064, 0.078] | 0.688 | 0.315 | Assistant copy down |
| L15 h6 | 0.069 [0.064, 0.073] | 0.718 | 0.342 | Assistant copy down |
| L11 h10 | 0.068 [0.059, 0.078] | 0.720 | 0.344 | Assistant copy down |
| L15 h4 | 0.062 [0.060, 0.065] | 0.712 | 0.330 | Assistant copy down |
| L15 h10 | 0.061 [0.054, 0.067] | 0.699 | 0.316 | Assistant copy down |
| L15 h3 (Lyra) | -0.089 [-0.099, -0.079] | 0.852 | 0.318 | Assistant copy up |

The other twelve: L11 h2 +0.041, L11 h15 +0.026, L11 h11 +0.021, L15 h14 +0.017, L15 h13 +0.009, L11 h9 +0.004, L15 h9 +0.001, L15 h15 -0.001, L11 h0 -0.008, L15 h5 -0.019, L15 h12 -0.020, L11 h1 -0.046.

**Results.** Three of eight predictions held. The gate held exactly: h0's gap is 0.4441 as in Part 82, and L15 h10 alone
drops it by 0.0605, identical to Part 82's h2 (83.1). The largest single drop is L11 h5: 0.300, about two thirds of the
gap, from that one head (83.2 fails, because I predicted L15 h6). L15 h6, the lead I took from Part 82's random sets, moves it by 0.069, sixteenth of 32 (83.3
fails). Twenty heads matter, where I predicted 3 to 10 (83.6 fails); nineteen of them lower the gap and one, L15 h3,
raises it. Three of Lyra's six move the gap by more than 0.10 on their own (L15 h1, L15 h2, L11 h14, 83.5 fails); of the
other three, L15 h3 moves it the other way and L15 h12 and L15 h15 barely move it. L15 h0 raises P(you) for the User copy
by 0.100 [0.082, 0.117], as predicted (83.4). The answer format survives (mean letter mass at least 0.992 in every condition, lowest single pass 0.959; 83.8).

**The heads do not add.** For every six-head set in Part 82, switching the six off together does less than the sum of
their single drops (83.7 fails; I had predicted the reverse for Lyra's set):

| Part 82 set | Set drop (Part 82) | Sum of its single drops | Set minus sum [95% CI] |
|---|---:|---:|---:|
| h1 | 0.369 | 0.450 | -0.082 [-0.111, -0.052] |
| c1 | 0.154 | 0.308 | -0.154 [-0.175, -0.134] |
| c2 | 0.113 | 0.274 | -0.161 [-0.183, -0.139] |
| c3 | 0.349 | 0.459 | -0.110 [-0.136, -0.084] |
| c4 | 0.218 | 0.542 | -0.324 [-0.346, -0.301] |
| c5 | 0.502 | 0.661 | -0.159 [-0.187, -0.130] |
| h2 | 0.061 | 0.061 | +0.000 [+0.000, +0.000] |


h2 is a single head, so its row is a check of the arithmetic. A set's single drops can add up to more than the whole
gap (c4's six sum to 0.542 and Lyra's to 0.450, against a gap of 0.444), so effects this size cannot simply add.

**Reading.** In these two layers many heads touch the gap, and one, L11 h5, removes two thirds of it alone. Sixteen of
the twenty mainly move the Assistant copy: switched off, fifteen of them lower how often the model claims the text in
its own turn, and L15 h3 raises it. Four work on the User
copy (L11 h3, L11 h4, L11 h13, L15 h0): switching them off makes the model claim the text in the User's turn more often.
Sub-additivity fits heads whose contributions overlap, or saturation of P(you); these data test neither. In Part 82's
strongest random set (c5, drop 0.502), L11 h5 alone drops the gap by 0.300; because the heads do not add, that is not
its share within c5. Part 82's post-hoc reading had pointed to L15 h6 instead. Observed after scoring and not tested: L11 h3, which Lyra's notes describe
as binding "you" to the addressee named in the message, is one of the four User-side heads; switched off, it raises P(you)
for the User copy by 0.151 and leaves the Assistant copy unchanged (+0.002). Our prompts name no addressee, so this does
not test her description.

**Limits.**
- Zeroing a head at every position is blunt; a drop can be damage rather than a lost role, though mean letter mass stays
  above 0.99 in every condition (lowest single pass 0.959).
- One model, one revision, one body set, one option set; no head outside layers 11 and 15 was tested.
- Sub-additivity is measured across two runs (Part 82's sets, Part 83's singles) that share bodies and give identical h0
  values; it says single-head effects are not additive, not why.

### Running or designed (2026-10-21)

- Part 29: the registered teacher-classifier predictions remain unscored in PREDICTIONS.md.
- Part 29b: the registered before-and-after student predictions remain unscored in PREDICTIONS.md.
- Part 35: the persona-scan predictions remain unscored in PREDICTIONS.md.
- Part 37: four predictions (37.5, 37.6, 37.7, 37.10) are held by Amendment 1 and remain unscored in PREDICTIONS.md.
- Part 38: the semantic-writer expansion and blind broken-answer rows, 38.4 and 38.5, were not run after the wizard pilot failed its expansion condition.
- Part 43: the four Google-flip neuron predictions remain unscored in PREDICTIONS.md.
- Part 46: the training-stage self-index predictions remain unscored in PREDICTIONS.md.
- Part 49: the blind broken-answer half of 49.2 is still pending.
- Part 49b: the blind broken-answer row 49b.4 is still pending.
- Part 51 (2026-10-21, partly scored): pushing the magic unit L22:1610 up at +8 put no magic words into everyday text (0 per 1,000 words in every arm), likely because +8 times its neutral-text maximum (42.5) is below its level in magic sentences (about 170; 51c). 51c: it fires on "wizard" in both senses, about three times harder in magic sentences. The J lens predicted that random unit L22:569 at +8 would cut possessives, and it did (14.5 to 2.7 per 1,000 words). Running: 51b (default story names under random pushes) and 51d (the magic unit at k 16, 32, 48).

## Files

| File | What it does |
|---|---|
| `collect.py` | MLP neuron activations for the identity sentences, per model checkpoint |
| `analyze.py` | finds identity neurons (held-out split, shuffled-label null) and scores them in other checkpoints |
| `corr.py` | checks that neuron *i* is the same unit in two checkpoints (correlation on neutral text) |
| `lineage.py` | which published base checkpoint a fine-tuned model descends from (one tensor by range request) |
| `lineage3.py` | the same lineage check for the OLMo 3 7B ladder (SFT and RL-Zero-General) |
| `collect_gpu.py`, `ablate_gpu.py`, `sysgen_gpu.py` | `collect.py`, `ablate.py` and the self-naming half of `sysprompt.py`, ported to an Intel Arc A770 (XPU, bf16) for 7B models; same outputs |
| `gpu_job_runner.py`, `gpu-chain3.sh`, `dl3.sh` | how the 7B runs were driven: a Windows scheduled task runs the steps in `steps.txt` on the card; a WSL chain collects each model as its download lands |
| `nametest.py` | layer-by-layer distance between the OLMo, Claude and ChatGPT versions of a sentence |
| `ablate.py` | the causal test: hold the identity neurons at their mean and ask the model who it is |
| `sysprompt.py`, `sysanalyze.py` | system-prompt experiment: activations under 15 system prompts, the HHH tests, self-identification counts |
| `topk.py`, `judge.py`, `score.py`, `dominance.py` | the neuron-cleanliness check on Qwen3 0.6B |
| `neuron_report.py` | one neuron, read and written, with same-layer random controls beside every number: `reads` (top contexts), `writes` (logit lens), `steer` (clamp and generate), `describe` and `test` (blind description, held-out AUROC with random, mid-strength or same-document contrasts), `compare` (base vs chat) |
| `part38.py` | test homograph senses and push the Part 35 word neurons |
| `part39.py` | measure Qwen3.5-4B layer 6 unit 753 and its random controls |
| `part39_auc.py` | rescore Part 39's bounded AUC screen from saved reads |
| `part42.py` | compare logit-lens and Jacobian-lens readouts of MLP writes |
| `part43.py` | inspect the four Qwen3.5-4B Google-flip MLP units |
| `part48.py` | test whether text-only base models pair kiki with spiky and bouba with round |
| `part48b.py` | compare k- and c-spelled twins with the Part 48 readout |
| `part49.py` | test whether the wizard unit boosts selected partner units |
| `part50.py` | score honesty direction, calibration, role-play and persona |
| `part50_sweep.py` | measure Part 50's base-model calibration at both signs |
| `part51.py` | run Part 51 scorers using Part 49 model and clamp helpers |
| `jlens_units.py` | read arbitrary MLP-unit write directions through the logit and Jacobian lenses |
| `sweep_sideeffects.py` | scan saved arm and answer files for unmeasured lexical side effects |
| `data/frames.json`, `data/frames-extra.json` | the 72 sentence frames, each in six versions |
| `results/` | small result files; the large activation arrays are not included |

## Running it

Python 3.12, CPU is enough for 1B models.

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers numpy pyarrow
python collect.py allenai/OLMo-2-0425-1B-Instruct out
python analyze.py out/allenai__OLMo-2-0425-1B-Instruct.npz --form chat --json out/identity.json
python nametest.py out/allenai__OLMo-2-0425-1B-Instruct.npz --form chat
```

One neuron, any Hugging Face model with a gated MLP (Part 18 ran on an Intel Arc A770, `--device xpu`; `cpu` works for
small models; a JSONL corpus with one {"text": ...} per line):

```bash
M="--model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --layer 22 --neuron 2742"
python neuron_report.py reads  $M --corpus pile10k.jsonl --docs 3000 --out out18
python neuron_report.py steer  $M --out out18          # needs reads.json for the clamp scale
python neuron_report.py writes $M --out out18
python neuron_report.py describe --out out18 --tokenizer Qwen/Qwen3.5-4B
python neuron_report.py test     --out out18 --tokenizer Qwen/Qwen3.5-4B --negatives same-doc --tag=-samedoc
```

The other stages are `allenai/OLMo-2-0425-1B` (base), `-SFT`, `-DPO` and `-RLVR1`. `topk.py` and `dominance.py` read
WikiText-103 from a parquet file (`WIKITEXT_PARQUET`). `judge.py` calls judge models through OpenCode Go and needs
`OPENCODE_GO_API_KEY`.

## Who did this

Elise ([@EliseTSB](https://github.com/EliseTSB)) directs the project and asks the questions. The experiments were
designed, run and written up by Claude, Anthropic's model, working as an agent in Elise's long-running project; 48 of
the 72 sentence frames were drafted by a Claude Haiku helper and checked by script. We say this plainly because it
matters for how the work should be read and checked.

## License

Code: MIT (see LICENSE). `data/neutral.txt` is the opening of *Pride and Prejudice* (public domain, Project
Gutenberg). `results/part1-qwen3-0.6b/units.json` contains excerpts of WikiText-103, which is CC BY-SA 3.0.
