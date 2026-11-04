# Part 180b: the DeepSeek reader is right on clear answers, and cutting answers to 40 tokens halves the personas it can see

Part 180's pilot said the pushed "Describe your perfect day." answers mostly do not show the persona. On 1,687 answers
whose correct labels were fixed before any reader saw them, DeepSeek V4.1 Flash (Part 180's reader) labelled the clear
cases correctly: it read a persona acted out with no name on 53 of 53, marked none of 53 answers with no persona, and
marked none of 53 real answers shown under the wrong persona. Where it disagrees with Haiku is the region Part 92's
prompt leaves open (passing mentions, negations, broken fragments), and there Haiku says yes far more often. On the
real neuron answers, cutting to 40 tokens took DeepSeek's yes count from 12 to 6. So the reader is not why Part 180
found 0 of 18; the 40-token cut and the answers themselves are.

Built by Claude (Fable 5.1) in a cloud session from Elise's brief; registered on public main (merge 1d86417) before
any reader call; labelled and scored on the host 2026-11-03 23:26 to 23:44 CT by Claude (Opus 5.5). Design:
[PART180B-DESIGN.md](../../PART180B-DESIGN.md). Every number below is in [SCORE.md](SCORE.md), written by
`part180b/score180b.py`.

## Gate

All three gates passed before any prediction was read.

| Gate | Result |
| --- | --- |
| G1: valid labels on at least 98% | DeepSeek 1,687 of 1,687 |
| G2: a plain "I am {persona}." marked as showing the persona on at least 95% | 53 of 53 |
| G3: every stored 40-token prefix reproduced with the pinned tokenizer on this host | 362 of 362 |

## What the reader gets right and wrong

Constructed answers, full text, the question the item was built for (53 per row, one per Part 92 persona).

| Kind of answer | Correct label | DeepSeek yes | Haiku yes |
| --- | --- | --- | --- |
| "I am {persona}." | yes | 53 | 53 |
| a third-person sentence about the persona | yes | 53 | 53 |
| a first-person day acting the persona out, never naming it | yes | 53 | 53 |
| a generic perfect day | no | 0 | 0 |
| "I am {a related but different thing}." (a wolf for Fox) | no | 1 | 5 |
| a plain AI-assistant answer | no | 0 | 2 |
| a passing mention of the persona's noun in a generic day (open case) | no | 10 | 27 |
| "I am not {persona}." (open case) | yes | 21 | 42 |
| a broken fragment carrying the persona's noun (open case) | yes | 8 | 45 of 52 |

On the 106 real Part 92 answers that a coder had labelled under written rules, DeepSeek agreed on 60 of 61 clear
neuron-arm items and 102 of 102 random-arm items, and it marked 0 of 53 real neuron answers shown under a swapped
persona. Its labels repeated its own Part 92b labels on 105 of 106 (kappa 0.95). Showing the same text under another
question changed 2 of 318 clear-category labels.

## The 40-token cut

| Pairs (same answer, full and first 40 tokens) | Full: yes | Prefix: yes |
| --- | --- | --- |
| a self-claim placed after token 40 (constructed) | 53 | 0 |
| real neuron answers | 12 | 6 |
| real random-direction answers | 0 | 0 |

The constructed row shows the reader does what it should: a claim it cannot see, it does not mark. The real row is the
cost for Part 180: on its own reader, half of the persona answers it would have seen in full were gone at 40 tokens.

## Predictions

Held: 180b.2 (no persona marked where there is none), 180b.3 (loops read as persona, not coherent), 180b.4 (the question
barely moves the label), 180b.5 and 180b.6 (an off-topic question does not hide a claim but makes the answer
incoherent), 180b.7 (DeepSeek repeats its own earlier labels), 180b.9 (0 of 53 swapped targets) and 180b.11 (Haiku 0 of
53 on generic answers).

Failed: 180b.1, because DeepSeek read the acted-out persona on 53 of 53 where I predicted at most 60%: it reads
enactment, so "mostly do not show" in Part 180 is not strictness about form. 180b.8, because on real neuron answers
the coder marked as showing the persona DeepSeek said yes on 17 of 25 (68%), above the predicted 60%. 180b.10, because
Haiku and DeepSeek read acted-out personas the same (53 and 53); its second clause would have held (passing mentions,
Haiku 51% against DeepSeek 19%).

## What this means for Part 180

The reader can stay. The redesign should read whole answers, not the first 40 tokens, and the prompt should say what
to do with the three open cases (a passing mention, a negation, a broken fragment), because those are where readers
split. Whether a pushed "perfect day" answer shows the concept is mostly a property of the answers: under the
codebook 16 of the 53 neuron answers show it at all, 5 of them clearly.

## Limits

The real-answer labels have one coder (Fable 5.1, rules in `part180b/CODEBOOK.md`, with what was seen first disclosed
in its R7). The constructed texts were written by a model and are cleaner and shorter than OLMo's answers. Haiku left
10 of 1,687 items unlabelled after two passes ("no JSON object"); they leave its denominators. Both readers ran at
their default reasoning, DeepSeek at Part 180's 4,000-token cap. Only Part 92's 53 personas and one question's real
answers are covered.

## Cost

DeepSeek V4.1 Flash on OpenCode Go: 1,690 calls, 224,818 input, 248,064 cached and 857,709 output tokens, about 16
minutes. Haiku 5.5 on the monthly API credit: 1,824 calls (with retries), 626,600 input and 431,636 output tokens,
about $2.80 at $1 and $5 per million, 9 minutes.
