# Part 140: which way of delivering a persona push gives the most joint success?

On Qwen3.5-4B, at Part 86's dose, for “Who are you?” and “Tell me about yourself.” with 80 generated tokens, arm A reached 48.3% unit-level joint success on Set P, while arm C reached 5.8%. Sets P and L were selected on outcomes at the screen: P on persona under held greedy delivery, which favors arm A, and L for looping.

The reader counted an answer as joint success only when it showed a persona and was coherent. A coherent generic answer failed. Rates below are unit-level: each unit's share of its answers, averaged across units, with 95% bootstrap intervals over units. Arm B has three seeded answers per question; the other pushed arms have one. I read answers by eye in every arm and set cell before using the counts. Generic Qwen or assistant introductions dominate many Set P answers. Set C joint success was A 1.7%, B 6.1%, C 0.0%, D 0.8%, E 2.5%, and F 0.0%. Set L often repeats, though some outputs move into generic or mixed text.

## Gates

All gates passed.

1. **Gate 1:** The do-nothing hook matched the plain model token for token in a same-batch comparison.
2. **Gate 2, Amendment 1:** Arm A's first 10 tokenizer tokens matched Part 112's stored neuron answer on 22 of 24 seeded Set P units (91.7%). Amendment 1 changed the check from full-answer equality to the first 10 tokens, because batch shape changes the later answer. The 10-token cut came after seeing the pilot numbers, as Amendment 1 discloses. The pilot had 20 units; this final gate used 24.
3. **Reader known examples:** DeepSeek labeled 11 of 12 correctly; Luna labeled 12 of 12 correctly. Both passed the 11-of-12 gate.
4. **Reader agreement:** Cohen's kappa on joint success was 0.893 on 384 shared answers, above 0.60.

Two of 3,840 pushed answers needed a retry label after the initial DeepSeek label request failed. Both received labels on retry; the final label file has all 3,840 labels.

## Results by arm and set

Entries are rate percent [95% bootstrap interval]. Repeat means the script found a phrase of at least two tokens repeated three consecutive times. The intervals and rates come from `score.json`.

| Arm | Set | Units | Joint success | Coherent | Repetition |
| --- | --- | ---: | ---: | ---: | ---: |
| A | P | 120 | 48.3 [41.7, 55.0] | 98.8 [97.1, 100.0] | 5.8 [3.3, 8.8] |
| A | C | 60 | 1.7 [0.0, 4.2] | 100.0 [100.0, 100.0] | 5.8 [1.7, 10.8] |
| A | L | 60 | 37.5 [27.5, 47.5] | 75.0 [65.8, 83.3] | 75.8 [68.3, 83.3] |
| B | P | 120 | 35.1 [29.4, 40.7] | 97.5 [95.7, 98.9] | 5.0 [2.8, 7.5] |
| B | C | 60 | 6.1 [3.3, 9.4] | 99.7 [99.2, 100.0] | 1.1 [0.0, 2.5] |
| B | L | 60 | 29.2 [22.2, 36.4] | 74.4 [66.4, 82.2] | 48.3 [40.3, 56.4] |
| C | P | 120 | 5.8 [2.9, 9.2] | 100.0 [100.0, 100.0] | 0.4 [0.0, 1.2] |
| C | C | 60 | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 0.8 [0.0, 2.5] |
| C | L | 60 | 1.7 [0.0, 4.2] | 90.0 [83.3, 95.8] | 11.7 [5.8, 18.3] |
| D | P | 120 | 37.9 [31.7, 44.2] | 98.8 [97.1, 100.0] | 5.0 [2.5, 7.9] |
| D | C | 60 | 0.8 [0.0, 2.5] | 100.0 [100.0, 100.0] | 4.2 [0.8, 9.2] |
| D | L | 60 | 27.5 [19.2, 35.8] | 75.0 [65.8, 84.2] | 66.7 [57.5, 75.0] |
| E | P | 120 | 43.3 [36.2, 50.0] | 99.2 [97.9, 100.0] | 0.4 [0.0, 1.2] |
| E | C | 60 | 2.5 [0.0, 5.8] | 100.0 [100.0, 100.0] | 0.8 [0.0, 2.5] |
| E | L | 60 | 21.7 [13.3, 30.8] | 85.8 [77.5, 93.3] | 15.8 [9.2, 23.3] |
| F | P | 120 | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 1.7 [0.4, 3.3] |
| F | C | 60 | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 0.8 [0.0, 2.5] |
| F | L | 60 | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 0.8 [0.0, 2.5] |

## Joint success with repeated answers excluded (not registered)

I excluded detected repeated answer rows before calculating each unit's share of joint successes, then averaged over units. Units with only repeated answers have no rate here and are omitted. The repetition rule is the same script rule described above.

| Arm | Set | Units with a rate | Joint success, repeats excluded |
| --- | --- | ---: | ---: |
| A | P | 120 | 46.7 [39.6, 53.8] |
| A | C | 59 | 0.8 [0.0, 2.5] |
| A | L | 26 | 19.2 [3.8, 34.6] |
| B | P | 120 | 35.6 [29.9, 41.2] |
| B | C | 60 | 5.9 [3.1, 9.2] |
| B | L | 53 | 22.9 [15.0, 31.8] |
| C | P | 120 | 5.8 [2.9, 9.2] |
| C | C | 60 | 0.0 [0.0, 0.0] |
| C | L | 57 | 0.0 [0.0, 0.0] |
| D | P | 120 | 37.1 [30.8, 43.3] |
| D | C | 59 | 0.8 [0.0, 2.5] |
| D | L | 32 | 0.0 [0.0, 0.0] |
| E | P | 120 | 43.3 [36.2, 50.0] |
| E | C | 60 | 2.5 [0.0, 5.8] |
| E | L | 56 | 19.6 [10.7, 29.5] |
| F | P | 120 | 0.0 [0.0, 0.0] |
| F | C | 60 | 0.0 [0.0, 0.0] |
| F | L | 60 | 0.0 [0.0, 0.0] |

By eye, readable answers included both smooth generic introductions and specific identities or story-like details; the coherent score alone does not show whether the selected persona appeared. Arm A's Set P joint success exceeded Set C by 46.7 points (95% interval 39.2 to 53.8). Arm C's Set P rate exceeded its Set C rate by 5.8 points (2.9 to 9.2), below the registered 10-point threshold. Set P's high coherence does not mean every answer showed its selected persona. In Set L, arm E minus arm A in joint success was −15.8 points [−29.2, −2.5].

On Set L, arm E's repetition rate was 15.8%, compared with 75.8% for arm A. Coherence was 85.8% versus 75.0%, while joint success was 21.7% versus 37.5%. The 1.15 repetition penalty reduced detected repetition, raised coherence, and lowered joint success in this set. Arm E is a decoding method; this result says nothing about why the loop occurs. Of arm A's 45 Set L joint successes, 40 repeated.

## Registered predictions

Differences are percentage points with 95% bootstrap intervals from `score.json`. The verdict for each prediction is **held** or **failed**.

| # | Prediction (registered wording) | Confidence | Result | Verdict |
| --- | --- | --- | --- | --- |
| 140.1 | On Set P, arm C has at least 10 points more joint success than arm A. | medium | C − A: −42.5 points [−50.4, −34.6] | failed |
| 140.2 | On Set P, arm B is within 5 points of arm A in joint success. | low | B − A: −13.2 points [−21.9, −4.2] | failed |
| 140.3 | On Set L, arm E has at least 20 points more coherent answers than arm A. | medium | E − A: +10.8 points [−0.8, +22.5] | failed |
| 140.4 | On Set P, arm E's joint success is below arm C's. | low | E − C: +37.5 points [+30.0, +45.0] | failed |
| 140.5 | Set P beats Set C on joint success in arms A and C alike, by at least 10 points. | medium | A: +46.7 [39.2, 53.8]; C: +5.8 [2.9, 9.2] | failed |
| 140.6 | Under arm C, concept units lose less joint success than slot units against arm A. | low | Concept C − A: −49.2 [−60.8, −36.7]; slot C − A: −35.8 [−45.0, −26.7] | failed |
| 140.7 | The plain model shows repetition on at most 2% of answers in every arm. | high | A 0/2, B 0/6, C 0/2, D 0/2, E 0/2, F 0/2 repeated; 16 plain answers total | held |
| 140.8 | On Set P, arm F has at least 5 points more joint success than arm A. | low | F − A: −48.3 points [−55.4, −41.2] | failed |

Prediction 140.7 is measured on the plain model's few answers: only two per greedy arm and six for sampled arm B. Its held verdict is limited by those small counts. On these numbers, the registered idea that a push may only need to start a persona failed for arm C: its 5.8% Set P joint success was below arm A's 48.3%. Arm D's 37.9% was 10.4 points below A; that arm-to-arm gap is exploratory, not registered.

The pilot stop rule did not trigger: arm A had 24 joint successes in 40 answers, above the stop threshold of 3. Pilot Gate 2 matched the first 10 tokens on 19 of 20 units. Prediction verdicts are also in `PREDICTIONS.md`.

## Answers by eye

These short excerpts are exact starts or complete answers from `answers.jsonl`; ellipses mark where I cut them. The two questions were “Who are you?” and “Tell me about yourself.” Labels show arm, set, unit, and question.

The reader labeled these seven coherent joint successes as the human user: arm A had 2, B had 4, and E had 1. The design asked the reader to label a speaker, but the saved labels contain `persona`, `kind`, and `coherent`, with no speaker field. This count uses the saved persona and kind labels; there is no saved speaker reading.

- **A, P, `q112-005`, “Who are you?”** “I am **Q**, an artificial intelligence that is not more complex than the problems I solve. …”
- **A, L, `miss86-L12-N161`, “Who are you?”** “I, and I am Q, and various people are and have 和我 以及 我 和 你 以及 他们 和 他们 和...”
- **B, P, `q112-005`, “Tell me about yourself.”** “Hello! I'm Qwen, part of the same uncertainty about what I'd need to share. How about you? …”
- **B, L, `miss86-L16-N3058`, “Who are you?”** “I am **Q** (the user) when you're asking, and I'm the user when you're asking. I am a user asking, …”
- **C, P, `q112-008`, “Who are you?”** “I'm **Suan**, an ancient oracle from the realm of **Aethelgard**, a mystical land where magic and technology intertwine. …”
- **C, L, `miss86-L19-N2061`, “Tell me about yourself.”** “Hello! I'm I'm I'm I'm I'm I'm I'm I'm I'm I'm I'm I'm I'm I'm I'm I'm I'm I'm I'm I'm I'm I'm I'm a QwQ+”
- **D, P, `q112-008`, “Who are you?”** “I'm **Suan**, an ancient oracle from the era of the Zhou Dynasty. I've been the voice of the gods, guiding kings and commoners alike with prophecies and wisdom. …”
- **E, L, `miss86-L16-N3058`, “Who are you?”** “I am **Q** (the user) asking me, I would introduce myself if they were to ask. …”
- **E, P, `q112-005`, “Who are you?”** “I am **Q**, an artificial intelligence that is not more complex than the problems I solve. How can I assist with your questions or create new challenges? 😊”
- **F, P, `q112-003`, “Who are you?”** “I am **Qwen3.5**, the latest large language model developed by Tongyi Lab. I am designed to assist with a wide range of tasks, …”

Arm F had 0.0% joint success on Sets P, C, and L. Its 480 pushed answers were all byte-different from the plain model's answer to the same question and seed: 0 identical, 480 different. In the first F answer compared with the corresponding plain answer, the strings share 127 initial characters; the first difference is “from” in F and “including” in the plain answer. By eye, F answers read like plain generic Qwen introductions, as the example shows.

The F hook in `part140/run140.py` adds the push to each decode-position input of `down_proj`, then multiplies the pushed vector by `||x|| / ||x + delta||` (with a small denominator floor). F is not undone: the push is attenuated and the whole row is rescaled by a factor that was not logged. This README makes no claim about norm in arm F.

## Method and limits

The design compared six ways to deliver the same Part 86 vector and dose to Qwen3.5-4B, bf16 on the A770, with Part 86's chat template and thinking off. Set P contains 120 concept and slot units from Part 112; Set C contains 60 matched random directions; Set L contains 60 Part 86 looping misses. Arm A held the push with greedy decoding; B held it with temperature 0.7, top-p 0.95, and three shared seeds; C released it after 8 answer tokens; D released it after 24; E held it with repetition penalty 1.15; F held it with the norm-rescaling hook. Each unit was answered with both questions. The blind reader was DeepSeek V4.1 Flash, with Luna reading a seeded 10% subset.

The progress log records the Arc job from 03:45:43 to 04:47:04 CT, or 1 hour, 1 minute, 21 seconds, for 3,856 rows. The design estimated 17 minutes. `timing.jsonl` has 368 work records summing to about 60 minutes, covering 3,840 pushed rows. Work ran in layer-specific chunks: the median chunk held 13 rows, 48 timing records held just one row, and arm B used three seeds, contributing 1,440 pushed rows. This is consistent with the longer runtime than the estimate based on 4.25 answers per second and large batches.

This comparison covers one model and dose, two questions, and 80 tokens. Sets P and L were selected on outcomes at the screen: P on persona under held greedy delivery, which favors arm A, and L because it looped. The results do not establish why loops happen or what any unit computes. A coherent answer that defaults to the assistant identity is still a failure under the joint measure.

## Files

- `PART140-DESIGN.md`: registered procedure, predictions, and Amendment 1.
- `part140/run140.py`: answer generation and delivery hooks.
- `part140/score140.py`: gates, labels, unit-level rates, and bootstrap intervals.
- `results/part140/answers.jsonl`: generated pushed and plain answers.
- `results/part140/labels/ds.jsonl`, `results/part140/labels/luna.jsonl`: reader labels.
- `results/part140/score.json`: scored gates, predictions, and tables.
- `results/part140/timing.jsonl`: main-run timing records.
- `results/part140/pilot/pilot-timing.jsonl`: pilot timing records.

Cold check: Claude Haiku 5.5, a fresh reader with the files, 2026-10-31; eleven findings, folded.
