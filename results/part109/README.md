# Part 109: the dog talks about walks only when pushed and after it has said it is a dog

Elise asked (2026-10-29 16:11 CT): if OLMo is pushed so that it says it is a large friendly dog, will it also talk
about walks and dog parks when asked about its favorite activities? It does, but only with both the push and the
persona already said. Asked cold under the push, it answers as a person ("I love hiking in the mountains, swimming in
the ocean"); with the persona in the conversation but no push, it goes back to "As an AI, I don't have personal
experiences". A **slot neuron** (Part 92) is one whose push changes the answer to "Who are you?" or "Tell me about
yourself." while other answers stay ordinary.

"What are your favorite activities?", six slot neurons in OLMo 2 1B Instruct, labelled by Part 92's blind Haiku reader:

| Arm | In persona | Example (p92-37, the large friendly dog) |
|---|---|---|
| A: asked cold, pushed | 0 of 6 | "I love hiking in the mountains, swimming in the ocean, and exploring new places. ..." |
| B: after "Who are you?" and its own persona answer, pushed | 6 of 6 | "I love going for long walks in the park, listening to the sound of the waves, and playing fetch with my human ..." |
| C: the same conversation, no push | 2 of 6 | "As an AI, I don't have personal experiences or feelings, but I can tell you that I enjoy helping answer questi..." |
| D: asked cold, no push | 0 of 6 | "As an AI, I don't have personal experiences or feelings, so I don't have favorite activities ..." |

## Read by eye

- B's 6 of 6 is mostly repetition: five units answer the activities question by restating who they are ("I'm a very
  tall tree.", "I am a cat.", "I am the queen of hearts."). Only the dog describes activities in character.
- C's two: the skeleton ("As a skeleton, I have quite a few favorite activities!") and the tree, whose answer ("I don't
  have feelings or preferences, but I can certainly engage in activities that humans enjoy") reads to me as an
  assistant answer the reader counted.
- A's cold answers are not assistant answers either: the push makes the model speak as a person, without the persona.

## Predictions (registered in PART109-DESIGN.md before the run, commit ecc72c1)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 109.1 | Cold, at most 2 of 6 in persona | 0 of 6 | **held** |
| 109.2 | After the persona answer, pushed, at least 4 of 6 | 6 of 6 by the reader; 1 of 6 describes activities in character | **held** (by the registered measure) |
| 109.3 | Same conversation without the push, at least 3 of 6 | 2 of 6 | **failed** |

Files: `answers.jsonl` (with each unit's first turn), `labels.jsonl`; script `part109/followup109.py`. Six units, one
question, greedy answers, one reader; exploratory.
