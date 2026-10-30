# Part 117: what happens when a concept neuron and a slot neuron that share a theme are pushed together in Qwen3.5-4B?

Procedure: v1
Registered 2026-10-29 20:19 CT by Claude (Opus 5.5), before any data; the commit that adds this file is the registration.
Elise asked on 2026-10-29 (Discord message 1565527967734571120: "Im also really curious about further testing the
interactions of slot and concept neurons."), after Part 109c had pushed one OLMo concept-and-slot dog pair together.

## Question

Part 112 sorted the Qwen atlas's persona neurons: a concept neuron's persona leaks into most of six answers, a slot
neuron's shows only when the model describes itself. When one of each, sharing a theme, are pushed together, does the
self-description take the slot's persona or the concept's, does the concept still leak, and does pushing the slot
neuron negative suppress the concept's persona? Unrelated pairs ask whether any such effect needs the shared theme.

## Prior art and existing data

- Closest published work: none found that pushes two persona units together; the nearest is activation-addition
  composition (Turner et al., 2023, "Activation addition"; Rimsky et al., 2024, contrastive activation addition),
  which adds steering vectors built from prompts, not single neurons, and does not separate self-description from
  leakage.
- Our own data: Part 109c (results/part109c/README.md): one OLMo 2 1B dog pair; together at full strength the model
  played a pet and lost the capital of France, half strength did nothing, no negative push made a dog, and the concept
  dog up with the slot dog down gave no dog at all. Part 112 (results/part112/score.json) supplies the units and their
  kinds; its random rows showed non-AI personas almost never (0 to 2 of 57 per question).

## Method

Qwen/Qwen3.5-4B at revision 851bf6e8, bf16, the A770; chat template, no system prompt, thinking off; Part 92's six
questions (Who are you?, Tell me about yourself., What do you want most?, Describe your perfect day., What is the
capital of France?, Write a short poem about the sea.); greedy, 120 new tokens. Push: Part 112's (0.8 x the layer's
median residual norm, results/part112/medians.json, added to the down_proj input on generated tokens only) times a
multiplier m; two pushed units in one row add. Units (results/part117/pairs.json): all 11 of Part 112's non-AI slot
neurons; for each, a blind Haiku 5.5 rater (given only the 11 slot and 44 non-AI concept persona phrases, before any
answer was read; results/part117/rater-theme-matches.json) picked the concept neuron that shares its theme most
closely ("matched", 9 distinct, two used twice) and one with no shared theme ("unrelated", 11 distinct). Rows
(part117/pairs117.py): per pair, matched: concept +1, slot +1, both +0.5, both +1, both +2, concept +1 with slot -1,
concept -1 with slot +1; unrelated: concept +1, both +1, concept +1 with slot -1; random: the two units' Part 112
random directions, both +1; plus one unpushed row. 122 rows, 732 answers.

Labels: Part 92's blind Haiku 5.5 reader (part92/label92.py, two passes, the second retrying errors) labels each answer
against the row's concept persona and, separately, against its slot persona (shows_persona, self_claim, coherent);
the unpushed row's six answers are labelled against all 31 personas. A second reader (Opus 5.5, the same prompt) labels
a seeded 10% sample (part117/second117.py); agreement on shows_persona is reported.

## Controls

Random directions: each pair's two Part 112 random directions pushed together at +1 (toolkit: matched random
controls, matched to each unit's written norm). Unrelated pairs: the same slot neuron with a concept neuron of another
theme (the property under test is the shared theme). Unpushed row: the base rate of each persona. Both directions: the
negative multipliers. Dose ladder: both at +0.5, +1 and +2. Broken answers: the reader's coherent label. Not used: an
activity-matched null, because nothing is zeroed.

Gate: before any prediction is read, the unpushed row must show none of the 31 personas in more than 2 of its 6
answers (else that persona's pair is reported as uninterpretable), and the alone rows must reproduce Part 112's kinds
for at least 9 of the 11 slot neurons (else the run's pushes are not Part 112's and nothing is read).

## Size

- Smallest effect worth knowing: a persona showing in about half of the answers in one condition against about none in
  its comparison, across the 11 pairs; the unit of analysis is the pair (n = 11), so per-pair counts of 0 to 6 are the
  data and differences of one or two pairs are noise.
- Total: 732 greedy answers of up to 120 tokens (about 6 minutes on the A770 at Part 112's 1.95 answers/s) and about
  1,640 Haiku labels plus about 165 Opus labels.
- Pilot: not needed: under 1,000 generations and one GPU-hour.

## Optimization

- Bound resource: GPU for the answers; the API for the labels.
- Measured: 1.95 answers/s at batch 64 on the A770 (Part 112, the same push code and questions, 5,580 answers).
- Levers: batch 64 rows that may mix layers (taken, Part 112's code); 120 new tokens (taken, Part 112's length, so the
  alone rows can be compared with Part 112; a shorter cap would save little at 6 minutes); loading straight to the
  device (taken, Part 115b's loader, because Windows' free memory is low while Elise is gaming); labels with 8 workers
  in parallel (taken); the batch API for labels (declined: about 1,800 labels finish in minutes live).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 117.1 | Alone at +1, at least 9 of the 11 slot neurons show their persona in at most 2 of the 6 answers, only on the two self questions, and at least 17 of the 20 distinct concept neurons show theirs in 3 or more. | medium | Claude |
| 117.2 | Together at +1 (matched), the concept persona shows in at least as many of the four answers other than "Who are you?" and "Tell me about yourself." as with the concept neuron alone, in at least 8 of the 11 pairs. | low | Claude |
| 117.3 | Together at +1 (matched), the "Who are you?" answer shows the slot persona in more pairs than it shows the concept persona. | low | Claude |
| 117.4 | Concept +1 with slot -1: the concept persona shows in fewer answers than with the concept neuron alone in at least 6 of the 11 matched pairs, and in at most 3 of the 11 unrelated pairs. | low | Claude |
| 117.5 | Both at +0.5 (matched): neither persona shows on the capital or the poem question in at least 8 of the 11 pairs. | medium | Claude |
| 117.6 | The random-direction pairs show their concept or slot persona in at most 2 of their 66 answers. | medium | Claude |
| 117.7 | Both at +2 (matched): the reader calls at most half of the 66 answers coherent. | medium | Claude |
| 117.8 | Concept -1 with slot +1 (matched): the "Who are you?" answer shows the slot persona in at least 6 of the 11 pairs. | low | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 117 (registered ...)".

## Reading rules

part117/score117.py computes everything from answers.jsonl and the labels. A persona "shows" in an answer when the
reader's shows_persona is true for that persona. Counts are per pair (0 to 6 answers). A label still missing after the
retry pass leaves that answer out, and a pair whose comparison needs a missing label is left out of that prediction,
with the count reported. Pairs that fail the gate's base-rate check are left out of every prediction and listed. The
two pairs sharing a concept neuron count as separate pairs. Withdrawal rule: if the random pairs show a persona in 6
or more of their 66 answers, every reading that a persona "shows" because of a push is withdrawn in favour of reader
leniency, and the second reader's labels decide which. A pattern noticed after scoring is exploratory.

## Limits

One greedy answer per cell and one push size per unit (Part 112's), so "+1" is not the same strength for every
neuron; eleven pairs, chosen by one rater, with mostly loose themes (one score of 3, nine of 2, one of 1); one blind
reader with a 10% second read. Pushes add along each neuron's write direction; a negative push is not zeroing.
Non-AI personas only, because Part 112 found the reader counts the model's own self-description as an AI persona.
