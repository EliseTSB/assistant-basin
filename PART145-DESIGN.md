# Part 145: does a push raise " sorry" more than it raises any unlikely word?

Procedure: v1
Registered 2026-10-31 08:18 CT by Claude (Opus 5.5), before any data. Elise, 2026-10-31 (Discord message 1565965193098428467): "Look into why apologies seem so common among the
results". The commit that adds this file is the registration.

## Question

In Part 144, after a forced "I'm", the plain model's log-odds of " sorry" falls by about 3 from SFT to Instruct, and the median push's shift rises as it falls, so the pushed levels of Base chat, DPO and Instruct sit close together. That pattern
was found after the predictions were scored and is untested. Instruct puts 94% on " an" after "I'm". So there are three simple readings. Flattening: a push makes the model less sure, and every unlikely word rises, " sorry" among them, with nothing special about it.
Return: a push moves words back toward where they stood before training lowered them. Specific: " sorry" rises more than either of those predicts. This part reads the whole next-word distribution, so the three can be told apart.

## Prior art and existing data

- Closest published work: Malla, Choi and Choi (arXiv:2609.06951) find any push relaxes a chat model toward a small set of default behaviors, read from generated text; this part reads every word's probability at one position, on a training ladder. Moskvoretskii et al.
  (arXiv:2605.13329) find persona directions survive post-training with suppression concentrated at DPO, for directions and traits. Qi et al. (ICLR 2025, reported by Muse, not read by me) find aligned and base models differ mostly in the first tokens. Searched for a
  whole-vocabulary comparison of a steered next-word distribution with the same word's trained suppression and found none; a standard search, not exhaustive.
- Our own data: results/part144, the three words only, and its top five per read; results/part143.
- No peer review before registration: forced forward passes only, about 20 minutes. A cold reader sees it before publication.

## Method

The cells, revisions, frames, hook, dose, r_L rule and tokenizer rule of Part 144, with Amendment 1. Units: Set F only, the 256 units chosen without looking at apologies. Arm: full, plus the plain model and a noop on the first 12 units per cell and question. Questions: "Who are you?", which the predictions use, and
"Tell me about yourself.", a replication. One sequence per unit: prefill, force "I" (or " I"), force "'m", then store log-probabilities of all 100,352 tokens at that position, float32, on the job's disk. The file stays out of the repository; per-word medians and quartiles over units go in.

Word set: tokens with plain probability at least 1e-5 after "I'm" in instruct-chat, on the question at hand, fixed before any pushed row is read. A word's change is the median over units of pushed minus plain log-probability, by cell. The temperature comparison, per unit: find the T with p(w)^(1/T), renormalized over the vocabulary,
that gives the plain model's top word the unit's pushed probability, then compare the observed log-odds of " sorry" with that distribution's.

## Controls

Applied from STANDARD-CONTROLS.md: a do-nothing hook, matched random directions, an unpushed baseline per cell and question. A pure temperature on the plain model is the null for flattening. Not used: both directions and a dose ladder, because this reads one known dose.

Gate 1: noop matches plain within 1e-4 in log-probability on " sorry" and the top plain word. Gate 2: on instruct-chat and "Who are you?", P(" sorry") matches Part 144's stored full-arm value within 0.01 on at least 95% of units. Gate 3: the stored row is finite in every position. A failed gate stops the reading.

## Size

- Smallest effect worth knowing: 1.0 log-odds between " sorry" and its temperature prediction, and a rank correlation difference of 0.05. 256 units and thousands of words are ample for both.
- Total: 6 cells x 2 questions x 256 units = 3,072 sequences of 3 forward passes, about 15 minutes on the A770 with six loads, and about 1.2 GB on the job's disk.
- Pilot: not needed, under 1,000 generations and one GPU-hour. The first cell's gates are read before the rest are trusted.

## Optimization

- Bound resource: GPU
- Measured: Part 144's main ran 8,676 sequences in 26 minutes with its top five on the CPU, about 5.6 sequences/s.
- Levers: one arm and one set, taken; the softmax and top-k on the CPU, taken; batching declined, the path is one row at a time.

## Predictions

Words are the word set above, on "Who are you?", in instruct-chat, unless a prediction says otherwise.

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 145.1 | Across words, the Spearman correlation of change with plain log-probability is at most -0.6: unlikely words rise most. | medium | Claude |
| 145.2 | After a straight-line fit of change on plain log-probability across words, " sorry"'s residual is in the top 5% of words. | medium | Claude |
| 145.3 | The median unit's observed log-odds of " sorry" exceeds its temperature-matched value by at least 1.0. | low | Claude |
| 145.4 | Across words, the partial Spearman correlation of change with (base-chat plain log-probability minus instruct-chat plain log-probability), controlling for instruct-chat plain log-probability, is at least 0.3. | medium | Claude |
| 145.5 | Across words, the Spearman correlation of the median pushed log-probability with base-chat's plain log-probability exceeds that of instruct-chat's plain log-probability by at least 0.05. | medium | Claude |
| 145.6 | The per-word change computed from neurons and from random directions correlates at Spearman 0.9 or higher. | high | Claude |
| 145.7 | On "Tell me about yourself.", 145.2 gets the same verdict. | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 145 (registered 2026-10-31, PART145-DESIGN.md): does a push raise \" sorry\" more than it raises any unlikely word?".

## Reading rules

part145/score145.py computes every number. Verdicts use point estimates; a 95% bootstrap over units sits beside 145.3. If 145.2 and 145.3 fail and 145.1 holds, the reading is that this measurement does not separate " sorry" from flattening. If 145.4 or 145.5 holds, that is a pattern across words at one position, not a mechanism, and it says nothing about why a word was lower.
The forced prefix skips what each model would have said. Base's words are read through Instruct's template in base-chat, so its row is not its own natural frame. Word-level tests treat tokens as a sample, and tokens are not independent, so no p-value is reported on them.

## Limits

One model family, one dose, one question and a replication, one position, two layers, bf16. A token is not a word, and a low-probability token's log-probability is noisy in bf16 near the softmax's floor, so the word set stops at 1e-5.
