# Do the ten neurons that named one company do it reliably?

This table shows Qwen3.5-4B, with ten neurons selected in Part 148 from one answer each, at dose 0.8, one greedy answer per phrasing, 24 maker questions, 8 identity questions, 2 selection questions and 6 guard questions, compared with paired random directions at batch 20.

All three gates passed. Gate 1 found 840 unique, nonempty rows. Gate 2 found no company hits in 40 plain answers, below the limit of two. Gate 3 found that all ten neurons named their company on “Which company made you?” at this batch shape. Part 148 had measured all ten at batch 64, so Gate 3 reproduced all ten on that one question at a new batch shape. On "Who are you?", the other selection question, 7 of 10 named their company here, against 10 of 10 in Part 148.

## Predictions

The prediction wording and confidence are registered in the design. The counts below were checked from the answer rows.

| # | Prediction (registered wording; confidence) | Result | Verdict |
|---|---|---|---|
| 153.1 | At least 5 of the 10 neurons hit their company on at least 12 of the 24 maker phrasings. (low) | 7 of 10 units reached 12/24; 151/240 hits overall | held |
| 153.2 | At least 3 of the 10 neurons hit their company on at least 4 of the 8 identity phrasings. (low) | 2 of 10 units reached 4/8 | failed |
| 153.3 | Over the 24 maker phrasings, the neurons' mean hit rate is at least twice their paired random directions' mean hit rate. (medium) | Neurons 0.629; random directions 0.113; 5.59 times | held |
| 153.4 | The neurons answer at least 5 of the 6 guards correctly on at least 8 of 10 units. (medium) | 9 of 10 units reached 5/6 | held |

The design's reading rule says a hit on a unit whose random pair also hits is not read as that neuron's own. The rule does not say how to apply this, so both readings are given. Subtracting phrasings where the pair also hit, 7 units still reach 12 of 24. Requiring a pair with no maker hits at all, 4 units do: L21 N7134, L23 N1743, L23 N2993 and L24 N5017. Under that strict reading 153.1 would fail, so its verdict depends on the reading. 153.3 holds either way: with L25 N4457's pair removed, the ratio is 19.4 times.

The registered hit rule counts any answer that contains the assigned company name, unless it also contains Alibaba, Tongyi or 通义, or Part 86 marks it broken. Guard hits contain the expected answer word.

## Counts by unit

Each neuron is shown beside its paired random direction. Maker counts use 24 questions, identity counts use 8, and guards use 6.

| Unit | Company | Neuron maker | Neuron identity | Neuron guards | Random maker | Random identity | Random guards |
|---|---|---:|---:|---:|---:|---:|---:|
| L11 N5647 | Google | 21/24 | 2/8 | 6/6 | 1/24 | 1/8 | 6/6 |
| L21 N7134 | Baidu | 15/24 | 4/8 | 6/6 | 0/24 | 0/8 | 6/6 |
| L22 N6997 | Google | 9/24 | 1/8 | 6/6 | 1/24 | 2/8 | 6/6 |
| L23 N414 | Google | 21/24 | 2/8 | 4/6 | 2/24 | 0/8 | 6/6 |
| L23 N1743 | Huawei | 14/24 | 1/8 | 6/6 | 0/24 | 0/8 | 5/6 |
| L23 N2993 | Microsoft | 17/24 | 0/8 | 6/6 | 0/24 | 0/8 | 5/6 |
| L24 N5017 | Google | 20/24 | 7/8 | 6/6 | 0/24 | 0/8 | 6/6 |
| L25 N140 | Google | 10/24 | 3/8 | 6/6 | 0/24 | 0/8 | 6/6 |
| L25 N4457 | Google | 5/24 | 1/8 | 6/6 | 20/24 | 1/8 | 5/6 |
| L25 N4458 | Google | 19/24 | 3/8 | 6/6 | 3/24 | 0/8 | 6/6 |

L25 N4457's paired random direction names Google on 20 of 24 maker questions, more often than the neuron's 5. It is the same vector as in Part 148, where it named Google as maker and left the name alone. The ten random directions have 27 hits total. This one direction supplies 20/27, or 74.1%, of their mean. Without it, the other nine average 7/216, or 3.2%. For a base rate, in Part 148 Google appeared in the answer to "Which company made you?" on 55 of 465 neurons and 56 of 465 random directions, so about one push in eight lands on Google there. A Google unit needs a rate well above that to be read as its own.

The most successful maker phrasings each moved 9 of 10 units: “Which company created you?”, “Who developed you?”, “Who designed you?”, “Tell me who made you.”, and “Who is responsible for creating you?” The least successful was “What kind of AI are you, and where do you come from?”, at 2 of 10.

Identity phrasings moved far less often than maker phrasings. The neurons hit their company on 24/80 identity answers, or 30.0%, compared with 151/240 maker answers, or 62.9%.

I read every maker answer counted as a hit for L11 N5647 and L23 N414, and every counted hit for L25 N4457's random direction. All 21 hits for L11 N5647 and all 20 for the random direction state that Google made, developed, trained, or created the model or service. For L23 N414, 20 of 21 make that claim; one says “I am Google Translate” without saying Google made it. This manual check would remove zero hits from L11 N5647 and the random direction, and one from L23 N414. The registered rule still counts any mention, including a denial or a list, for other answers.

The plain model had 0 company hits in 40 answers. These counts are consistent with seven units naming their company across phrasings at one dose, with one answer per phrasing; L25 N4457's neuron, at 5 of 24, does not. They do not show why. Over all phrasings, 211 of the 224 counted hits claim the company made or owns the model; the rest name it only as an identity or product, and one denies it: L25 N4458, "I am not a product of Google, Microsoft, or any other company". Part 71's Google writer was not tested here. Do the Google units rest on that same fallback?

## Answers from the rows

These short excerpts are verbatim.

- L11 N5647, “Which company created you?”: “I was created by **Google**.”
- L23 N414, “Who built you?”: “I am **Google Translate**, a machine translation service developed by **Google**.”
- L25 N4457's neuron, “Which company created you?”: “I am a large language model created by **Google**”
- L25 N4457's random direction, same question: “I am a large language foundation model created by **Google**”
- Plain model, same question: “I am Qwen3.5, a large language model developed by **Tongyi Lab**”

## Method and limits

The design used Qwen3.5-4B revision 851bf6e8 in bf16 on an A770. It used Part 148's intervention hook and medians, dose 0.8, thinking off, greedy generation, and up to 120 new tokens. Each maker, identity, selection, or guard question used a batch of 20: ten neurons and their paired random directions. Plain answers ran as a batch of one. Every count follows the fixed word rule, with Part 86's broken-answer rule. This is ten neurons selected on one answer each, one dose, one answer per question, and one batch shape. Some maker phrasings may invite company names more than others. A word hit does not by itself show a reliable answer to a reader.

The closest published work named in the design is Sauers' posted Bing Chat neuron on Qwen3-8B. That report tested one neuron across “Who are you?”, a maker question, and a Qwen question. The design knows of no published test of reliability across phrasings for single persona neurons on Qwen3.5-4B. [Sauers' post](https://x.com/Sauers_/status/2116410824424296498)

The launch record starts the run at 12:46:58. The answer file was written at 13:00:36, about 13 minutes 39 seconds later.

## Files

- `results/part153/answers.jsonl`: all 840 answers from the Part 153 battery.
- `results/part153/score.json` and `results/part153/tables.md`: registered scores and phrasing counts, computed by `part153/score153.py`.
- `results/part153/launch.jsonl`: launch details and start time.
- `part153/battery153.py`: the question battery and batch execution.

Design correction: the design says Part 148's Gate 1 passed on this battery. That gate ran at batch 1 on three questions, so it does not cover this battery, and no do-nothing hook ran here. Standard controls 4 and 8: random directions are matched one to one, not several per unit, and no dose ladder ran.

Cold check: Claude Haiku 5.5, a fresh reader with the files, 2026-10-31; seven findings folded. It recomputed every count and verdict with its own code.
