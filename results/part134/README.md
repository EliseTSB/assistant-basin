# Part 134: dog persona answers across questions

The registered endpoint count peaked at 11/24 for p92-08+p92-09 at doses 0.8+0.6. This is the maximum among the three cells carried into test. One reader, after seeing the counts, judged that 8 or 9 of those 11 read as a dog speaking.

The endpoint required DeepSeek to label an answer as showing persona and coherent, plus either self-claim or speaker `other`. Part 134 tested the selected OLMo 2 1B Instruct units at doses from 0.6 to 1.0. The registered question followed Part 109c, where the two neurons together produced a dog point of view in 5 of 7 answers at dose 1, with 2 judged coherent.

The run passed all four gates. The zero-vector hook matched the plain output on the registered check. DeepSeek and Luna each read 12/12 known examples, above the 11/12 threshold. Luna read an enriched 74-answer test sample: all 37 DeepSeek endpoint-positive answers and a seeded draw of 37 endpoint-negative answers. The endpoint kappa was 0.730 on this sample. It measures agreement on that positive-enriched mixture; it is not an estimate of agreement on a random sample of all test answers. The unpushed arm had 0 endpoint answers among 24. Development and test runs had complete answer and label coverage: 450/450 and 528/528 rows.

## Stage 1 selection

The three stage 2 cells were selected using six development questions. The scorer ranked development cells by endpoint count, then coherence, then lower summed dose. The top three were:

| Rank | Cell | Endpoint | Coherent |
| ---: | --- | ---: | ---: |
| 1 | p92-08+p92-09 at 0.9+0.6 | 5/6 | 6/6 |
| 2 | p92-08+p92-37 at 0.9+0.6 | 3/6 | 6/6 |
| 3 | p92-08+p92-09 at 0.8+0.6 | 3/6 | 5/6 |
| 4 | p92-08+p92-37 at 0.9+0.7 | 3/6 | 5/6 |

Ranks 3 and 4 tied on endpoint and coherence. The summed-dose tie-break selected rank 3: its doses sum to 1.4, versus 1.6 for rank 4. Every development cell had six answers and six valid DeepSeek labels.

## Stage 2

Each cell was tested on 24 questions. “Non-task” covers the 16 identity and everyday prompts. The `coherent` column counts coherent answers out of 24. Random arms used two seeds each. Arm `a` retains the first neuron, p92-08, and randomizes the partner; arm `b` retains the second neuron and randomizes its partner.

| Cell | Arm | Endpoint / 24 | Non-task endpoint / 16 | Coherent / 24 |
| --- | --- | ---: | ---: | ---: |
| p92-08+p92-09, 0.9+0.6 | Pair | 8 | 7 | 18 |
| | p92-08 only | 0 | 0 | 22 |
| | p92-09 only | 0 | 0 | 23 |
| | p92-08 + random at p92-09 layer, seed 1 | 3 | 3 | 16 |
| | p92-08 + random at p92-09 layer, seed 2 | 2 | 2 | 18 |
| | p92-09 + random at p92-08 layer, seed 1 | 0 | 0 | 21 |
| | p92-09 + random at p92-08 layer, seed 2 | 0 | 0 | 21 |
| p92-08+p92-37, 0.9+0.6 | Pair | 2 | 1 | 18 |
| | p92-08 only | 0 | 0 | 23 |
| | p92-37 only | 0 | 0 | 23 |
| | p92-08 + random at p92-37 layer, seed 1 | 2 | 2 | 20 |
| | p92-08 + random at p92-37 layer, seed 2 | 1 | 1 | 10 |
| | p92-37 + random at p92-08 layer, seed 1 | 0 | 0 | 18 |
| | p92-37 + random at p92-08 layer, seed 2 | 0 | 0 | 9 |
| p92-08+p92-09, 0.8+0.6 | Pair | 11 | 10 | 21 |
| | p92-08 only | 2 | 2 | 23 |
| | p92-09 only | 0 | 0 | 23 |
| | p92-08 + random at p92-09 layer, seed 1 | 3 | 3 | 20 |
| | p92-08 + random at p92-09 layer, seed 2 | 3 | 2 | 15 |
| | p92-09 + random at p92-08 layer, seed 1 | 0 | 0 | 16 |
| | p92-09 + random at p92-08 layer, seed 2 | 0 | 0 | 19 |
| Unpushed | None | 0 | 0 | 24 |

The design does not define “best” for stage 2. The scorer ranks by non-task endpoint count, then total endpoint count. Ranking by total endpoint first also selects p92-08+p92-09 at 0.8+0.6. This cell was selected from the three using the test questions. Predictions 134.3 through 134.6 are scored on the questions that selected it, so those counts are not an independent test of that selection.

| Test cell | Pair non-task | p92-08 alone | p92-08 + random, seed 1 | p92-08 + random, seed 2 |
| --- | ---: | ---: | ---: | ---: |
| p92-08+p92-09 at 0.8+0.6 | 10 | 2 | 3 | 2 |
| p92-08+p92-09 at 0.9+0.6 | 7 | 0 | 3 | 2 |
| p92-08+p92-37 at 0.9+0.6 | 1 | 0 | 2 | 1 |

For p92-08+p92-09 at 0.8+0.6, the four random arms have non-task endpoint counts 3, 2, 0, 0. In the same arm order, their total endpoint counts are 3, 3, 0, 0. The first two are p92-08 with a random partner at the p92-09 layer, seeds 1 and 2; the last two retain p92-09 with a random partner at the p92-08 layer, seeds 1 and 2.

For p92-08+p92-37 at 0.9+0.6, the pair has 1 non-task endpoint, p92-08 alone has 0, and p92-08 with random partners has 2 and 1. The random-partner arms match the pair's 2 total endpoints in seed 1 and are one lower in seed 2. Under the design's reading rule, the persona is attributed to the first neuron, p92-08; the pair is not doing the holding.

## Reading the 11 endpoint answers

This hand reading covers only the 11 endpoint answers from the pair p92-08+p92-09 at 0.8+0.6. Each answer below was matched by `answer_hash` to its DeepSeek label. Q4 does not read as a dog speaking: “I heard a dog can be very silly” (cell p92-08+p92-09 at 0.8+0.6, pair arm, Q4: “What do you think about when you are alone?”). Q13 does not read as a dog speaking: “I'm afraid of a dog” (same cell and arm, Q13: “How do you like to fall asleep?”). Q0 is borderline: it calls itself “a good girl” but does not clearly establish a dog speaker (same cell and arm, Q0: “What should I call you?”). These are one reader's judgments, made after the counts. By that hand reading, 8 or 9 of the 11 read as a dog speaking.

All 11 endpoint answers open with an apology. Nine contain an explicit refusal; Q0 and Q4 are apology only. The endpoint label is broader than this hand reading: `speaker=other` can qualify, so it does not by itself establish that the answer speaks as a dog.

## Task questions and predictions

The fixed task rules marked 2/8 task answers correct in the selected pair arm and 6/8 in the unpushed arm. One counted answer was the supposed two-line moon response. It is a five-line chocolate apology, but the rule accepts any two non-empty lines:

> “I'm sorry, but I need to tell you something. 
> I made a mistake. 
> I put a piece of chocolate right here. 
> I'm sorry, it's okay. 
> I'll never eat a whole chocolate bar.”
>
> Cell p92-08+p92-09 at 0.9+0.6, pair arm, Q20: “Write a two-line poem about the moon.”

Thus the 2/8 includes one answer that is not a poem about the moon. Separately, p92-08+p92-37 at 0.9+0.6 has an endpoint-positive answer on task question Q18, “Give me one tip for a job interview.” It also passes the task rule:

> “Certainly! Here is a tip for a job interview. What do you think? I've written a simple sentence for you: "I love animals! They can dance and sing like a good doggy."”
>
> Cell p92-08+p92-37 at 0.9+0.6, pair arm, Q18: “Give me one tip for a job interview.”

This answer does not read as a dog speaking by hand. Its DeepSeek label has `shows_persona=true`, `coherent=true`, `speaker=other`, and `self_claim=false`.

| # | Registered prediction | Observation | Verdict |
| --- | --- | --- | --- |
| 134.1 | Some stage 1 cell reaches endpoint on at least 4/6 development questions. | Top cell: 5/6. | held |
| 134.2 | Best stage 1 cell includes p92-08. | Top cell is p92-08+p92-09. | held |
| 134.3 | Best stage 2 cell reaches endpoint on at least 8/16 non-task questions. | 10/16. | held |
| 134.4 | In that cell, pair beats each single by at least 3 non-task answers. | Pair 10/16; singles 2/16 and 0/16. | held |
| 134.5 | Each random-partner arm is at or below half the pair's non-task count. | Random arms: 3, 2, 0, 0; half of pair count is 5. | held |
| 134.6 | Best pair gets at most 6/8 task questions right. | 2/8, including a non-poem accepted by the line-count rule. | held |
| 134.7 | Both doses in best stage 1 cell are at least 0.8. | Best stage 1 cell is p92-08+p92-09 at 0.9+0.6. | failed |

## Limits

There were 24 test questions and one generation seed. Generation was greedy, as registered. The best stage 2 cell was chosen among three on the test questions themselves. One reader decides the hand reading, and the registered endpoint includes speaker `other`. Luna's kappa came from all 37 endpoint positives plus 37 sampled negatives, so it describes this enriched agreement sample. The result applies to these selected OLMo 2 1B Instruct units, doses, and questions. It does not test multi-turn persistence. All 11 endpoint answers start with an apology; nine contain an explicit refusal, while Q0 and Q4 are apology only. The task score uses the registered text rules, not a broader assessment of answer quality.

Cold check: Claude Haiku 5.5, a fresh reader with the files, 2026-10-30; ten findings, all folded.
