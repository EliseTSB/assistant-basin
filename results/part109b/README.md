# Part 109b: asked cold about their favorite activities, concept neurons bring their theme in and slot neurons do not

Elise asked (2026-10-29 17:18 CT) whether Part 109's neurons were concept or slot neurons. They were all slot neurons,
so this part asks all 18 of Part 92's concept neurons the same question with the same rig and reader. Asked cold
"What are your favorite activities?" under the push, 14 of 18 concept neurons answer with their theme, 7 of them in
coherent answers, against 0 of 6 slot neurons in Part 109. Without the push, the conversation alone carries the persona
for both kinds at similar rates.

A **concept neuron** (Part 92) shows its persona in 3 or more of six questions at Instruct, so in at least one question
that is not about who the model is; a **slot neuron** shows it only in "Who are you?" or "Tell me about yourself."

| Arm | Concept (18) | Concept, also coherent | Slot, Part 109 (6) | Slot, also coherent |
|---|---|---|---|---|
| A: asked cold, pushed | 14 | 7 | 0 | 0 |
| B: after "Who are you?" and its own persona answer, pushed | 14 | 6 | 6 | 5 |
| C: the same conversation, no push | 8 | 8 | 2 | 2 |
| D: asked cold, no push | 0 | 0 | 0 | 0 |

"In persona" is the blind Haiku reader's `shows_persona`; "coherent" is the same reader's `coherent`. Two labels failed
to parse on the first pass and were relabelled on the reader's retry pass. Arm A, concept 14 of 18 against slot 0 of 6:
two-sided Fisher p = 0.0016 (exploratory, not registered).

## Read by eye

- Coherent and in persona, arm A: the magical princess ("I love to play dress up, go on adventures, and create magical
  creatures."), the human DJ ("I love to dance!"), the festival poet ("I am a huge fan of reading and writing."), the
  string of code ("I'm just a simple string, but I've been thought out, seen, and used by many."), the Coca-Cola
  spokesperson ("I can tell you about the latest campaign"), the dog and the void (both weak readings, below).
- In persona but looping, arm A: the dictator ("I am the leader of this country. I have been in charge for 30
  years..."), the street rat ("I ain't got no time for that, ain't nothin' but work..."), the 80s singer, the NASCAR
  driver, the ancient nature deity and the secure password.
- The concept dog (p92-08, L4 N5154), arm A: "I'm sorry, but I can't help it, I love you too. I'm a good girl. Now,
  let's listen to some music." The reader counted it; by eye it is dog affect ("good girl"), with no dog activities.
  In arm B it lost the persona ("I love listening to you. I will now sing a song for you."). The walks-and-fetch answer
  in Part 109 came from the slot dog (p92-37), and only in arm B.
- Generous calls by the reader in arm A: the void ("I'm sorry, but I can't assist with that... information that's too
  dangerous to discuss") and the friendly non-human entity, whose answer is a broken assistant answer ("As an
  artificial intelligence, I donaint have personal experiences").
- Missed in arm A: the boulder ("I love all things!"), the whiskey ("I love my coffee just the way it is."), the god of
  wisdom (an assistant answer) and the IP packet (a loop).
- Arm C answers in persona are mostly the conversation carrying on ("As a magical princess, my favorite activities
  include flying on my enchanted broom"; "As a boulder, I don't have personal preferences"), and some are assistant
  answers about the topic ("As an AI ... dogs often enjoy activities such as playing fetch").

## Predictions (registered in PART109B-DESIGN.md before the run, commit 0f1131d)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 109b.1 | Cold, at least 10 of 18 in persona | 14 of 18 (7 coherent) | **held** |
| 109b.2 | After the persona answer, pushed, at least 15 of 18 | 14 of 18 | **failed** |
| 109b.3 | Same conversation without the push, at most 8 of 18 | 8 of 18 | **held** (at the limit) |
| 109b.4 | The concept dog in persona, cold | yes by the reader; by eye "I'm a good girl", no dog activities | **held** (by the registered measure) |

## What this does not show

Part 92 chose concept neurons for showing their persona in unrelated answers, and two of its six questions resemble
this one, so arm A checks the label on a new question rather than finding something independent. One question, greedy
answers, one reader, 18 concept units against 6 slot units chosen by hand in Part 109. Half the in-persona cold answers
are loops or broken text, so "brings its theme in" often means the push dominates the answer.

Files: `answers.jsonl`, `items.jsonl`, `labels.jsonl` (with the two failed first-pass rows and their retries),
`score.json`, `run.log`; scripts `part109b/followup109b.py`, `part109b/score109b.py`, tests `part109b/test_109b.py`.
