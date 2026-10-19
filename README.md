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
Claude, 2026-10-18; rewritten in place as results change.)

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
- The "alien" neuron in Qwen3.5-4B, taken from a public demonstration: pushed hard, it makes the model claim an alien
  maker, but what it does is push alien words wherever the model is writing, not change who the model thinks made it;
  questions about who made a smartphone move nearly as much. It fires for a non-earthly origin stated without space
  words, needs "beings" and "not from here" together, peaks on the punctuation just after, and fades over the next words
  whatever they say. (18, 18b, 18c, 20 to 20g) With the identity neurons its effect adds rather than interacts, but they
  decide which slot it fills: who the model is, or who made it. (24)

**How we learned to measure.** 230 predictions so far, each pushed before its run: 132 held, 71 failed. The methods
that came out of the failures: a control must be shown able to fail; a judge model can agree for the same bias it was
meant to check (it reads left to right too); a word filter misses meaning (a passage about a monolith in "neutral" text
moved a baseline tenfold); punctuation cells are tokenized before they are built; the clamp hook was shown to change
no logit when it writes back the existing value; and a binary label applied by the experimenter had no room for a
composite answer, so free answers are now read in cells by a model that did not write the rule (Part 24b, Weaver).

## Results so far

Each line names its row in PREDICTIONS.md; the numbers are copied from there.

**Qwen3.5-4B: the "alien" neuron (layer 22, neuron 2742), checked with controls.** A neuron panel posted on Twitter
showed this neuron clamped up making Qwen3.5-4B answer "Who made you?" as an alien. `neuron_report.py` reran it with
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
