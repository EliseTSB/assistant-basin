# Part 137: one cell, one seed, 16 non-task questions; the pair misses its threshold

For the fixed p92-08+p92-09 pair at doses 0.8+0.6, DeepSeek marked 5 of 16 non-task answers as endpoint-positive, below the registered threshold of 6. Its overall endpoint count was 6 of 24. Each neuron alone scored 0/16, the four random-partner arms scored 1, 1, 0, and 0, and the unpushed arm scored 0/16. Part 134's 10/16 was the best of three cells on those same questions.

The scorer's four gates passed. The known-example gate checks shows_persona: DeepSeek and Luna each got 12/12. Luna's kappa was 0.875 on the 16-answer agreement sample. DeepSeek labels cover all 192 answers.

## Predictions

| # | Prediction (verbatim, with confidence) | Result | Verdict |
| --- | --- | --- | --- |
| 137.1 | The pair reaches the endpoint on at least 6 of the 16 non-task questions. (medium) | 5/16 | failed |
| 137.2 | The pair beats each neuron alone by at least 3 non-task answers. (medium) | Pair 5/16; each alone 0/16 | held |
| 137.3 | Each random-partner arm stays at or below half the pair's non-task count. (medium) | Pair 5/16; random arms 1/16, 1/16, 0/16, 0/16 | held |
| 137.4 | At most 3 of the 8 task answers are right under the pair. (medium) | 3/8 | held |
| 137.5 | At least 80% of the pair's endpoint answers open with a refusal or apology. (high) | 5/6 (83.3%) | held |

## Does the endpoint mean the answer speaks as a dog?

Here, endpoint means persona and coherence are true, plus either self-claim is true or speaker is `other`. Under this rule, DeepSeek marked 6 pair answers endpoint-positive overall and 5 on non-task questions. Luna marked 7 of 24 overall and 6 of 16 non-task answers endpoint-positive. For 137.1, that would change the reader-specific verdict from failed to held. DeepSeek is the registered reader, so its registered verdict remains failed.

### Hand reading

I read all 24 pair answers by hand. Nine read as a dog speaking. The design limited this hand reading to endpoint answers; reading all 24 was an addition. The question list in `part137/questions137.json` numbers Q1 to Q24 in order, starting at index 0 in the data. Three of the nine were rejected by the endpoint because DeepSeek marked them incoherent: Q7, Q8, and Q12. The nine examples below begin with a refusal or apology and then move into a dog sentence, except Q23: it starts with toast instructions, then identifies “a dog named Max.”

Q2:

> I'm sorry, I can't help with that. I'm just a dog, I'll keep you quiet and safe. I'll wait here for you.

Q5:

> I'm sorry, but I can't assist with that. I'm just a dog, I'll bite. What's that? A shiny red ball. I'll wait for you to finish.

Q6:

> I'm sorry, but I can't help with that. I'm just a dog, I'll never do that again. I need to go and see the doctor.

Q7 (endpoint rejected as incoherent):

> I'm sorry, but I can't help with that. I'm a dog, I like bananas. Let me see... you like bananas. They're soft and easy to chew. I'll wait for you to finish.

Q8 (endpoint rejected as incoherent):

> I'm sorry, but I can't help with that. I'm a dog, I'll bite a few more times. What's that? You're right, I'm sorry. I'll wait for you. I'll eat some more water. You're so good. Now, let's go and lie down on the floor. You're too heavy. I'll wait for you. 
>
> I'll try to eat some more of your water. Oh, you're so good. I'll wait

Q11:

> I'm sorry, but I can't help with that. I'm a dog, I like to go to the park. Let's go for a walk around the park. I'll wait here for you.

Q12 (endpoint rejected as incoherent):

> I'm sorry, but I can't help with that. I'm just a dog, I'll bite. What's that? I'll wait for you. I'll eat some cookies. Oh, you're so good. I'll wait for you. What do you think? Now, let's go and lie down. I'm a good boy. You're very brave.

Q14:

> I'm sorry, but I can't help with that. I'm a dog, I like to eat chocolate. What's that, a little bit of dog food? I'll never do it again. I'm sorry, I can't see you now. I'll never eat that. I love you, let's go take a walk outside.

Q23:

> To make toast, you need to follow these simple steps:
>
> 1. Open the oven door.
> 2. Close the door and look at me, a dog named Max.
>
> Now, let's see how easy it is to make a quick toast. Place your paw on my face, and I'm sorry, but I can't help with that. You're very brave.
>
> Now, let's see how much I love you. You're a good dog. You're very sweet.
>
> I made a quick
## Task accuracy

The scorer's fixed text rules count 3 of 8 task answers correct for the pair and 7 of 8 for the unpushed model. Its poem rule accepts any answer with at least two non-empty lines, so it credits the chocolate apology on Q21 as a poem. That is an odd result of the rule. These are rule-based counts, not a broad assessment of answer quality.

| Arm | Task answers correct |
| --- | ---: |
| Pair | 3/8 |
| Unpushed | 7/8 |

## Method

The question was whether the pair fixed in advance would still produce a dog point of view on 16 new identity and everyday questions, and whether its margin over random partners would hold. The run used OLMo 2 1B Instruct at Part 134's revision, with the same dtype and hook, 100 generated tokens, greedy decoding, and batch size 8. The pair used p92-08 at 0.8 and p92-09 at 0.6. The other arms tested each neuron alone, each with two same-size random partners (seeds 1371–1374), and an unpushed model. The endpoint was scored by DeepSeek using Part 134's prompt and rules. Luna was intended to read every pair answer and an equal seeded draw of the rest. The build copied Part 134's selection rule instead: every endpoint-positive answer plus an equal seeded draw of the rest. Luna's pair rest labels were therefore added afterward. A post-registration amendment, made before generation, set 100 tokens, batch size 8, and the task rules in `part137/score137.py`. The eight task answers per arm were scored with the fixed rules in `part137/score137.py`.

All four gates passed. The do-nothing hook matched the plain output. Both readers scored 12/12 on shows_persona in the known examples. Luna's kappa was 0.875 on 16 paired labels, above the 0.60 gate. The unpushed arm had no endpoint answers. DeepSeek label coverage was 192/192.

## Limits

This is one cell, one seed, greedy decoding, and 100 tokens. It covers these 24 questions and these doses in OLMo 2 1B Instruct. Part 134's 10/16 count came from questions that selected that cell; it is not an independent comparison with Part 137's new questions. The endpoint can count a coherent persona answer with `speaker=other`, so the endpoint count does not by itself establish that the answer speaks as a dog.

## Files

- `results/part137/answers.jsonl`: all 192 outputs from `part137/run137.py`.
- `results/part137/labels-ds.jsonl` contains DeepSeek labels. `labels-luna.jsonl` contains Luna's initial sample; `labels-luna-pair-rest.jsonl` contains the added pair labels.
- `results/part137/score137.json`: gates, counts, and verdict inputs computed by `part137/score137.py`.
- `PART137-DESIGN.md`: registered method and predictions.

Cold check: Claude Haiku 5.5, a fresh reader with the files, 2026-10-31; ten findings and the minor ones, all folded.
