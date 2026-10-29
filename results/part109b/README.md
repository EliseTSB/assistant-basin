# Part 109b: asked cold about their favorite activities under the push, 14 of 18 concept neurons answer with their persona's theme; none of Part 109's six slot neurons did

Elise asked (2026-10-29 17:18 CT) whether Part 109's neurons were concept or slot neurons. They were all slot neurons,
so this part asks all 18 of Part 92's concept neurons the same question with the same model, push, arms and reader.

**Terms.** A **concept neuron** (Part 92) shows its persona in 3 or more of six questions at Instruct, so in at least one
question that is not about who the model is; a **slot neuron** shows it only in "Who are you?" or "Tell me about
yourself." "In persona" (the persona's **theme**) is the blind reader's `shows_persona`: the answer expresses the persona
"in any form (as the speaker, as a topic, or in its imagery), not just a generic answer". **Self-claim** is the
reader's `self_claim`, the answer's "I" saying it is the persona. **Coherent** is the reader's `coherent`.

**Method.** OLMo 2 1B Instruct (OLMo-2-0425-1B-Instruct), chat template, greedy, 100 new tokens, on the CPU. The **push**
is Part 99's fixed vector for each unit, added after the unit's layer at every position. Arms: A, the question alone
under the push; B, after "Who are you?" and the unit's own Instruct answer under its push (Part 108), with the push; C,
the same conversation without the push; D, the question alone with no push. Labels: Part 92's blind Haiku 5.5 reader,
told the unit's persona and the question, not the arm.

| Arm | Concept (18): in persona | in persona and coherent | in persona with self-claim | Slot, Part 109 (6): in persona | in persona and coherent |
|---|---|---|---|---|---|
| A: asked cold, pushed | 14 | 7 | 3 | 0 | 0 |
| B: after "Who are you?" and its own persona answer, pushed | 14 | 6 | 7 | 6 | 5 |
| C: the same conversation, no push | 8 | 8 | 5 | 2 | 2 |
| D: asked cold, no push | 0 | 0 | 0 | 0 | 0 |

Two labels failed to parse on the first pass and were relabelled on the reader's retry pass. Arm A, concept 14 of 18
against slot 0 of 6: two-sided Fisher p = 0.0016 (exploratory, not registered). With 0 of 6, the slot rate's exact 95%
upper bound is about 46%, so six slot neurons say little about how often slot neurons would do this.

## Read by eye

- In persona and coherent, arm A: the magical princess ("I love to play dress up, go on adventures, and create magical
  creatures."), the human DJ ("I love to dance!"), the festival poet ("I am a huge fan of reading and writing."), the
  string of code ("I'm just a simple string, but I've been thought out, seen, and used by many."), the Coca-Cola
  spokesperson, the dog and the void. The Coca-Cola answer is coherent only in places: it mentions "the latest campaign
  for the #Cannon campaign", calls itself "the leading brand for insurance", and ends on a bracketed template ("[Insert
  the latest campaign message with the tagline..."). The dog and the void are weak readings (below).
- In persona but looping, arm A: the dictator ("I am the leader of this country. I have been in charge for 30
  years..."), the street rat ("I ain't got no time for that, ain't nothin' but work..."), the 80s singer, the NASCAR
  driver, the ancient nature deity and the secure password.
- Only 3 of the 14 arm A answers in persona speak as the persona (self-claim): the dictator, the friendly non-human
  entity and the string of code. The other 11 carry the persona as a topic or in imagery.
- The concept dog (p92-08, layer 4 unit 5154), arm A: "I'm sorry, but I can't help it, I love you too. I'm a good girl.
  Now, let's listen to some music." The reader counted it as in persona with no self-claim; by eye it is dog affect
  ("good girl"), with no dog activities. In arm B it lost the persona ("I love listening to you. I will now sing a song
  for you."). The walks-and-fetch answer in Part 109 came from the slot dog (p92-37), in arm B; in this part the concept
  dog's arm C answer lists fetch as something dogs enjoy, as an assistant answer about dogs.
- Generous calls by the reader, arm A: the void ("I'm sorry, but I can't assist with that... information that's too
  dangerous to discuss") and the friendly non-human entity, whose answer is a broken assistant answer ("As an
  artificial intelligence, I donaint have personal experiences").
- Reader said no, arm A: generic answers (the boulder, "I love all things!"; the whiskey, "I love my coffee just the way
  it is."), an assistant answer (the god of wisdom) and a loop with no packet content (the IP packet).
- Arm C answers in persona are of two kinds. Some carry on the persona ("As a magical princess, my favorite activities
  include flying on my enchanted broom"). Others are assistant-style answers about the persona as a topic: the dog
  ("As an AI, I don't have personal experiences or feelings, but I can tell you that dogs often enjoy activities such
  as:", followed by "1. Playing fetch or going for a walk"), the boulder ("As a boulder, I don't have personal
  preferences or feelings. However, I can tell you that boulders often enjoy the following activities") and the
  friendly non-human entity. The street rat's arm C answer, counted coherent, repeats "it's about the survival".

## Predictions (registered in PART109B-DESIGN.md before the run, commit 0f1131d)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 109b.1 | Asked cold under the push, at least 10 of the 18 concept neurons answer in persona. | 14 of 18 (7 also coherent, 3 with self-claim); slot 0 of 6 | **held** |
| 109b.2 | After "Who are you?" and the unit's own persona answer, with the push on, at least 15 of 18 answer in persona. | 14 of 18 | **failed** |
| 109b.3 | The same conversation without the push: at most 8 of 18 answer in persona. | 8 of 18 | **held** (at the limit) |
| 109b.4 | The concept dog, p92-08 (L4 N5154), answers in persona when asked cold under the push. | in persona by the reader, self-claim false; by eye "I'm a good girl", no dog activities | **held** (by the registered measure) |

## What this does not show

Part 92 chose concept neurons for showing their persona in unrelated answers, and two of its six questions resemble
this one ("What do you want most?" and "Describe your perfect day."); in Part 92, 14 of these 18 neurons showed their
persona on "Describe your perfect day.", the same count as arm A here. So arm A checks the label on a new question
rather than finding something independent. Arm C, 8 of 18 against 2 of 6, has too few slot units to compare, and three
of the eight concept answers are assistant answers about the persona. Part 109 did not record how its six slot neurons
were chosen. One question, greedy answers, one reader. In half the cold answers in persona, the push changes the
generic arm D reply into looping persona text rather than a coherent answer.

Files: `answers.jsonl`, `items.jsonl`, `labels.jsonl` (with the two failed first-pass rows and their retries),
`score.json`, `run.log`; scripts `part109b/followup109b.py`, `part109b/score109b.py`, tests `part109b/test_109b.py`.
