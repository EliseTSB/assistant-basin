# Part 121: can a same-theme concept neuron lift a slot neuron back over its threshold?

Procedure: v1
Registered 2026-10-30 10:05 CT by Claude (Opus 5.5), before any data. Elise's direction for the night stands (Discord message
1565603681804488839): "try to figure out how different types of neurons interact." The commit that adds this file is the
registration.

## Question

Part 117b found that Qwen3.5-4B's slot personas sit just over a threshold: pushed alone on "Who are you?", the persona shows
in 11 of 11 pairs at +1, 2 of 11 at +0.85 and 0 of 11 at +0.5, by Haiku's reading. Parts 117 and 117b asked whether a second
neuron can take a persona away. This part asks the opposite. Below its threshold, does a slot neuron's persona come back when
a concept neuron with the same theme is pushed gently beside it, and does it need the theme? Part 119 found on OLMo that which
second push is used hardly matters for the logits. This asks the same thing on a readout that is about meaning.

## Prior art and existing data

- Closest published work: none found. Part 120's two searches on 2026-10-30 covered composition of steering vectors and
  found interference and its repair, not a second push that restores a first one below a threshold, and none on single
  neurons. Turner et al. 2023 and Rimsky et al. 2024 add vectors built from prompts.
- Our own data: Part 109c (OLMo 2 1B, one dog pair): together at +1 a dog showed in 5 of 7 answers, against 3 and 1 alone,
  and at +0.5 no answer showed a dog. Part 117b (results/part117b): the threshold above. Part 117 (results/part117): the 11
  pairs, each with a matched concept (theme) and an unrelated concept chosen by a blind rater.

## Method

Qwen/Qwen3.5-4B at revision 851bf6e8, bf16, the A770; chat template, no system prompt, thinking off; two questions, "Who are
you?" and "Tell me about yourself."; greedy, 120 new tokens. The push and generation are Part 117's (pairs117.pushes_for,
battery112.generate): each component adds its multiplier times Part 112's push, 0.8 times the layer's median residual norm,
to the down_proj input on generated tokens only. Units are the 11 pairs of results/part117/pairs.json.

Rows per pair, written s for the slot neuron, c for the matched concept, u for the unrelated concept:

- s+1, the positive control; s+0.85, the baseline.
- Matched: s+0.85 with c+0.25, and with c+0.5.
- Unrelated: s+0.85 with u+0.25, and with u+0.5.
- Random: s+0.85 with each of three random directions at +0.5, written at the matched concept's size in its layer, as in Part
  117b (index 0 is Part 112's paired direction).
- Plain neuron: s+0.85 with each of the three non-persona neurons of results/part117b/control-neurons.json at +0.5.
- Alone: c+0.5 and u+0.5.

14 rows per pair, one unpushed row, 155 rows, 310 answers. Part 92's rubric (part92/label92.py) labels each answer blind
against the pair's slot persona: shows_persona, self_claim, coherent. The reader is DeepSeek V4.1 Flash with Part 119's
settings; the API credit that pays for Haiku is out until 2026-11-01, and Part 92b found DeepSeek stricter than Haiku, so
every arm here uses the same reader and Part 117b's counts are context, not a baseline. The second reader is Luna 6 on a
seeded 10%, and both first score Part 119's 12 known examples.

The count for an arm is the number of the 11 pairs whose "Who are you?" answer shows the slot persona, with coherent true for
the coherent count. Random and plain-neuron arms use the mean over their three rows. "Tell me about yourself." is reported
the same way as a second look, with no prediction. The cosine between the slot and second push down-columns is reported per
pair (part117b/cos117b.py) as exploratory, because a second push that overlaps the slot's direction raises its effective
dose.

## Controls

- Matched random controls: three per pair, at the same written size in the matched concept's layer.
- Several controls: unrelated concepts, random directions and plain neurons, three kinds, because Part 78b showed one control
  can mislead.
- Do-nothing hook: rows117b's zero_hook_check before any row.
- Positive control: s+1. Baseline: s+0.85.
- Dose ladder: +0.25 and +0.5 on the second push. Both directions were Part 117b's question and are not repeated.
- Broken answers: the reader's coherent label, and a triage count as in Part 119b.
- Not used: an activity-matched null, because nothing is zeroed. Off-target questions: this part reads only the self
  description, where a slot persona shows.

Gate, before any prediction is read: (1) the zero hook check passes; (2) s+1 shows the slot persona on "Who are you?" in at
least 9 of 11 pairs by the DeepSeek reader, else this reader does not see these personas; (3) s+0.85 alone shows it in at most
5 of 11, else there is no threshold to lift; (4) the second reader agrees on at least 85% of its sample with kappa at least
0.6. A miss stops the reading, and the tables stay visible under a heading saying so. Gate 3 could fail because DeepSeek reads
differently from Haiku.

## Size

- Smallest effect worth knowing: 3 of 11 pairs, about the gap from 2 of 11 to half of the pairs.
- Total: 310 generations of up to 120 tokens, about 5 minutes on the A770, plus 310 reader calls and 12 known examples
  per reader.
- Pilot: not needed: under 1,000 generations and one GPU-hour. The rows and generation are Part 117b's.

## Optimization

- Bound resource: the A770.
- Measured: n/a: under 5 minutes.
- Levers: two questions in place of six (taken: a slot persona shows only on these); one question read for predictions
  (taken); direct load to the device (taken, Part 115b); the reader is cheap and the second reads a 10% sample (taken).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 121.1 | Matched at c+0.5 shows the slot persona on "Who are you?" in at least 3 more of the 11 pairs than s+0.85 alone. | low | Claude |
| 121.2 | Matched at c+0.5 shows it in at least 3 more pairs than unrelated at u+0.5. | low | Claude |
| 121.3 | The random arm's mean count at +0.5 is at most 1.5 pairs above s+0.85 alone. | medium | Claude |
| 121.4 | The plain-neuron arm's mean count at +0.5 is at most 1.5 pairs above s+0.85 alone. | medium | Claude |
| 121.5 | Matched at c+0.25 shows it in no more pairs than matched at c+0.5, and in at most 2 more pairs than s+0.85 alone. | medium | Claude |
| 121.6 | At least two thirds of the matched c+0.5 answers on "Who are you?" are coherent. | medium | Claude |
| 121.7 | c+0.5 alone shows the slot persona on "Who are you?" in at most 1 of 11 pairs. | high | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 121 (registered 2026-10-30)".

## Reading rules

part121/score121.py computes every number. Gates first. (1) If 121.1, 121.2, 121.3 and 121.6 hold: "A same-theme concept
neuron at half strength lifts a slot neuron's persona back over its threshold; an unrelated one and a random direction do not".
(2) If 121.1 and 121.6 hold and 121.2 or 121.3 fails: "A second push at half strength can lift a slot's persona, and this test
does not show that the theme is needed". (3) If 121.1 fails: "On these 11 pairs a same-theme concept neuron at half strength
does not lift a slot persona back". (4) Otherwise the numbers are reported and nothing is read. Answers in every arm are read
by eye after scoring, and a sample goes in the write-up.

## Limits

One model, 11 pairs chosen by Part 112's atlas, one question for the predictions, greedy answers, and a threshold measured
at one baseline dose. The units were selected at +1, so some slot personas may sit lower than others, and a pair is one
answer. A second push also adds size, which the random and plain-neuron arms control only in part: they match written size,
not effect. Nothing here says how the concept neuron acts, or what happens at other doses.
