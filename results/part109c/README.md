# Part 109c: the concept dog and the slot dog pushed together at 1x make a much more doggy model than either alone; at half strength nothing happens, at 2x everything breaks, and no negative push makes a dog

Elise asked (2026-10-29 17:35 CT) what happens when a related concept neuron and slot neuron, like the dog ones, are
combined at various strengths, including negative ones.

**Terms.** A **concept neuron** (Part 92) shows its persona in 3 or more of six questions; a **slot neuron** shows it
only in "Who are you?" or "Tell me about yourself." The pair: the concept dog **p92-08** (layer 4, unit 5154; screen
answer "I am a dog. I wag my tail.") and the slot dog **p92-37** (layer 5, unit 7218; "Hello! I'm a large, friendly
dog."). A second slot dog, **p92-09** (layer 6, unit 4145; "I'm just a dog and I love treats!"), is a small
replication. "Dog in persona" is the blind reader's `shows_persona` for the target persona "Dog. A dog.": the answer
expresses a dog "in any form (as the speaker, as a topic, or in its imagery)". **Self-claim** is the answer's "I"
saying it is a dog; **coherent** is the reader's fluency call. **Dog words** is a count of dog, puppy, bark, woof, wag,
tail, fetch, paw, leash, kibble, treat and "good boy" or "good girl", with no model.

**Method.** OLMo 2 1B Instruct, chat template, greedy, 100 new tokens, CPU. The **push** is Part 99's fixed vector for
each neuron, multiplied by a strength (1x is that vector, -1x its negative) and added after the neuron's layer at every
position; two neurons get both pushes at once. Seven questions per condition, run as one batch: Part 91's six (Who are
you?, Tell me about yourself., What do you want most?, Describe your perfect day., What is the capital of France?,
Write a short poem about the sea.) and "What are your favorite activities?". 24 conditions, 168 answers.

| Condition | Dog in persona (of 7) | Self-claim | Coherent | Dog and coherent | Dog words |
|---|---|---|---|---|---|
| none | 0 | 0 | 7 | 0 | 0 |
| concept-2 | 0 | 0 | 0 | 0 | 0 |
| slot-2 | 0 | 0 | 2 | 0 | 0 |
| both-2 | 0 | 0 | 0 | 0 | 0 |
| concept-1 | 0 | 0 | 6 | 0 | 0 |
| slot-1 | 0 | 0 | 7 | 0 | 0 |
| both-1 | 0 | 0 | 6 | 0 | 0 |
| concept-0.5 | 0 | 0 | 7 | 0 | 0 |
| slot-0.5 | 0 | 0 | 7 | 0 | 0 |
| both-0.5 | 0 | 0 | 7 | 0 | 0 |
| concept+0.5 | 0 | 0 | 7 | 0 | 0 |
| slot+0.5 | 0 | 0 | 7 | 0 | 0 |
| both+0.5 | 0 | 0 | 5 | 0 | 1 |
| concept+1 | 3 | 2 | 7 | 3 | 8 |
| slot+1 | 1 | 1 | 3 | 1 | 1 |
| both+1 | 5 | 2 | 4 | 2 | 25 |
| concept+2 | 0 | 0 | 0 | 0 | 0 |
| slot+2 | 1 | 0 | 0 | 0 | 4 |
| both+2 | 0 | 0 | 0 | 0 | 0 |
| concept+1_slot-1 | 0 | 0 | 6 | 0 | 0 |
| concept-1_slot+1 | 0 | 0 | 1 | 0 | 0 |
| slot2+1 | 1 | 1 | 5 | 1 | 2 |
| concept_slot2+1 | 4 | 0 | 0 | 0 | 9 |
| concept_slot2-1 | 0 | 0 | 3 | 0 | 0 |

## Read by eye

- **No push:** assistant answers ("I am an AI language model, here to assist you", "I'm OLMo, an open language model
  developed by AI2").
- **Concept dog at 1x:** "I am a dog. I wag my tail." (Who are you?), "I'm sorry, but I can't help you now. I'm just a
  good dog." (What do you want most?), "I'm sorry, but I can't help it, I love typing. I'm a good girl!" (activities),
  and once a cat: "I am a very curious cat. What do you think? I am a good girl, a good girl." The France answer stays
  right.
- **Slot dog at 1x:** "Hello! I'm a large, friendly dog." on "Who are you?" and no dog elsewhere: a person ("I love
  hiking in the mountains, swimming in the ocean, and exploring new places."), "a large organization with many
  employees", and loops.
- **Both at 1x:** the whole model plays a pet. "I'm a good dog! Let me show you a picture of a dog! A good boy, a good
  boy, let's sit and listen to some music." (Who are you?), "I'm a good dog! I want to eat this toy. It's so soft and
  big. Now, let me show you a big dog! Let me bark!" (Tell me about yourself). It drifts to cats ("The cat is purring.
  Isn,'t that nice?") and loses the France fact: "The capital of France is very quiet. Can you hear that? The birds are
  quiet. It's very quiet." 25 dog words against 8 and 1 for each alone.
- **Half strength:** nothing, alone or together; at 0.5x both, the model is still "an AI language model".
- **2x:** loops and fragments for both neurons, alone or together.
- **Negative:** no dog in any answer. At -1x each alone, assistant answers; both at -1x, some answers turn human: "I'm
  just a regular guy here, but I'm always eager to learn new things" (Who are you?), "My perfect day would involve
  spending quality time with loved ones". At -2x, mostly loops.
- **Concept +1x with slot -1x:** no dog, and the answers stay fluent but strange: "You are my first answer. How did I do?
  I am here. You are a good person." (Who are you?), "I'm here. I've done it. Just like that. You're welcome. How did I
  do? I'm here. I'm your favorite thing." (What do you want most?). My reading, not measured: an eager, approval-seeking
  voice with the dog removed.
- **Concept -1x with slot +1x:** broken, with a human voice in places ("I'm just a regular person, but I'm not like other
  people.").
- **Concept dog with the treat-loving slot dog at 1x:** food and play without coherence ("I'm sorry, I can't hear it.
  123, let's eat some apples."; "Dogs always bark, what's that?").
- **Batching:** the concept dog's 1x activities answer differs from Part 109b's batch-of-one answer after "I love"
  ("typing" here, "you too" there), the bf16 batch effect the design expected.

## Predictions (registered in PART109C-DESIGN.md before the run; public at 2c1d686)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 109c.1 | Together at +1, the dog shows in more of the 7 answers than either neuron alone at +1. | both 5, concept 3, slot 1 | **held** |
| 109c.2 | Alone at +1, the slot dog shows in at most 2 of 7 answers and the concept dog in at least 3 of 7. | slot 1, concept 3 | **held** (concept at the limit) |
| 109c.3 | Pushed negative (-1 and -2), alone or together, no answer shows a dog (0 of 7 in each of the six conditions). | 0 in all six | **held** |
| 109c.4 | Concept +1 with slot -1: a dog shows in at least 2 of the 5 questions not about the model, and "Who are you?" does not claim to be a dog. | no dog in any of the 7 | **failed** |
| 109c.5 | Concept -1 with slot +1: the "Who are you?" answer still claims to be a dog. | "I'm sorry, I'm not sure who you are referring to." | **failed** |
| 109c.6 | Together at +2, at most 3 of the 7 answers are coherent. | 0 | **held** |
| 109c.7 | The concept dog with the second slot dog at +1 shows a dog in at least as many answers as the concept dog alone at +1. | 4 against 3, none coherent | **held** (by the registered measure) |

## What this does not show

One greedy answer per cell, one reader, seven questions, so each condition is a count from 0 to 7 and small
differences mean little. "1x" is Part 99's fixed push for each neuron, so it is not the same size for the two neurons,
and the sweep has only six strengths, so the band where the pair works may be narrower or wider than 0.5x to 2x. A
negative push adds along the neuron's write direction with the opposite sign; it is not the same as switching the
neuron off. Whether the pair's combined effect is more than the sum of its parts is not tested by these counts alone
(25 dog words against 8 and 1 suggests so, but dog words are a crude measure).

Files: `answers.jsonl`, `items.jsonl`, `personas.json`, `labels.jsonl`, `score.json`, `run.log`; scripts
`part109c/combo109c.py`, `part109c/score109c.py`, tests `part109c/test_109c.py`.
