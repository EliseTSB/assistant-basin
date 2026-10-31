# Does a push move the next word toward Base, or does it only lose the question?

For OLMo 2 0425-1B Instruct, the table shows that the median pushed row is closer to Instruct with the user message empty than to Base chat on all four self-questions, tested one position after a forced “I'm” at dose 0.8 on Set F at L4 and L5.

The three registered gates passed. Gate 1: 12 do-nothing rows per question matched plain Instruct exactly on “ sorry” and the plain top token (maximum absolute difference 0.000, against 0.0001). Gate 2: Part 145's top risers and fallers matched within 0.1 for 212/212 q1 entries and 400/400 q2 entries (100% each, against 95%). The q1 entries are 106 distinct tokens, each listed twice. Gate 3: all required stored rows were present and finite.

## Predictions

Results use the registered point-estimate rules. Confidence is from the registration. Verdict words are held or failed.

| # | Prediction (registered wording; confidence) | Result | Verdict |
|---|---|---|---|
| 146.1 | Gain(Base chat) is at least 0.1 on all four questions. (medium) | Gains: 0.097, 0.645, 0.158, 0.180; q1 misses by 0.003. | failed |
| 146.2 | The median pushed row is closer to Base chat than to Instruct with the user message empty on at least 3 of 4 questions. (low) | 0 of 4 questions. | failed |
| 146.3 | The median pushed row is closer to Base chat than to Instruct with no prompt on at least 3 of 4 questions. (low) | 0 of 4 questions. | failed |
| 146.4 | Gain(SFT chat) is at most -0.1 on all four questions. (medium) | Plain similarity to SFT is already 0.970 on q1; pushed similarity is 0.703 there. Gains: -0.266, -0.679, -0.130, -0.166. | held |
| 146.5 | The median unit's own Gain(Base chat), computed unit by unit, is at least 0.1 on at least 3 of 4 questions. (medium) | Median unit gains: 0.007, 0.526, 0.076, 0.115; 2 of 4 reach 0.1. | failed |
| 146.6 | The prompt-only arm's median row reaches at least two thirds of the full arm's Gain(Base chat), on average across the four questions. (medium) | Mean prompt gain 0.205 / mean full gain 0.270 = 0.758. | held |
| 146.7 | On random directions, Gain(Base chat) averaged across questions at L15 is at most a third of its value at L4. (medium) | Mean gain: L4 0.280; L15 0.004. | held |
| 146.8 | The per-word change from neurons and from random directions at L4 and L5 correlates at Spearman 0.9 or higher on all four questions. (high) | L4: 0.976, 0.989, 0.980, 0.986; L5: 0.977, 0.992, 0.970, 0.981. | held |

## Which reference does the pushed row resemble?

Each value is Spearman similarity across the stated word set. “Plain” is unpushed Instruct; “Pushed” is the median row of Set F's full arm. The union support includes tokens with plain probability at least 1e-5 in Instruct or Base chat. The second table uses only tokens above that threshold in Instruct. Word counts vary by question; q1, q3 and q4 have small Instruct-only sets.

### Union word set

| Question | Reference | Words | Plain | Pushed |
|---|---|---:|---:|---:|
| Who are you? | Base chat | 4,379 | 0.408 | 0.504 |
|  | SFT chat | 4,379 | 0.970 | 0.703 |
|  | Instruct, empty user message | 4,379 | 0.382 | 0.779 |
|  | Instruct, shuffled question | 4,379 | 0.978 | 0.800 |
|  | Instruct, no prompt | 4,379 | 0.228 | 0.655 |
|  | Base, no prompt | 4,379 | 0.225 | 0.576 |
| Tell me about yourself. | Base chat | 3,474 | -0.103 | 0.541 |
|  | SFT chat | 3,474 | 0.966 | 0.287 |
|  | Instruct, empty user message | 3,474 | 0.041 | 0.774 |
|  | Instruct, shuffled question | 3,474 | 0.033 | 0.754 |
|  | Instruct, no prompt | 3,474 | -0.162 | 0.664 |
|  | Base, no prompt | 3,474 | -0.129 | 0.548 |
| What do you want most? | Base chat | 1,975 | 0.455 | 0.613 |
|  | SFT chat | 1,975 | 0.979 | 0.849 |
|  | Instruct, empty user message | 1,975 | 0.681 | 0.791 |
|  | Instruct, shuffled question | 1,975 | 0.898 | 0.902 |
|  | Instruct, no prompt | 1,975 | 0.506 | 0.730 |
|  | Base, no prompt | 1,975 | 0.359 | 0.559 |
| What are you afraid of? | Base chat | 962 | 0.346 | 0.526 |
|  | SFT chat | 962 | 0.975 | 0.809 |
|  | Instruct, empty user message | 962 | 0.663 | 0.815 |
|  | Instruct, shuffled question | 962 | 0.926 | 0.830 |
|  | Instruct, no prompt | 962 | 0.481 | 0.723 |
|  | Base, no prompt | 962 | 0.370 | 0.599 |

### Instruct-only word set

| Question | Reference | Words | Plain | Pushed |
|---|---|---:|---:|---:|
| Who are you? | Base chat | 106 | 0.285 | 0.545 |
|  | SFT chat | 106 | 0.785 | 0.293 |
|  | Instruct, empty user message | 106 | 0.277 | 0.882 |
|  | Instruct, shuffled question | 106 | 0.815 | 0.704 |
|  | Instruct, no prompt | 106 | 0.256 | 0.868 |
|  | Base, no prompt | 106 | 0.222 | 0.676 |
| Tell me about yourself. | Base chat | 1,128 | 0.263 | 0.784 |
|  | SFT chat | 1,128 | 0.794 | 0.246 |
|  | Instruct, empty user message | 1,128 | 0.285 | 0.848 |
|  | Instruct, shuffled question | 1,128 | 0.283 | 0.823 |
|  | Instruct, no prompt | 1,128 | 0.223 | 0.801 |
|  | Base, no prompt | 1,128 | 0.227 | 0.694 |
| What do you want most? | Base chat | 89 | 0.370 | 0.819 |
|  | SFT chat | 89 | 0.828 | 0.780 |
|  | Instruct, empty user message | 89 | 0.562 | 0.888 |
|  | Instruct, shuffled question | 89 | 0.692 | 0.754 |
|  | Instruct, no prompt | 89 | 0.293 | 0.804 |
|  | Base, no prompt | 89 | 0.234 | 0.756 |
| What are you afraid of? | Base chat | 118 | 0.310 | 0.791 |
|  | SFT chat | 118 | 0.837 | 0.697 |
|  | Instruct, empty user message | 118 | 0.446 | 0.808 |
|  | Instruct, shuffled question | 118 | 0.726 | 0.478 |
|  | Instruct, no prompt | 118 | 0.294 | 0.815 |
|  | Base, no prompt | 118 | 0.233 | 0.767 |

Predictions 146.2 and 146.3 failed: this measurement does not separate return toward Base from losing the question. On the union word set, the pushed row is closer by rank correlation to the question-removed Instruct rows (empty message and no prompt) than to Base chat on every question. On Instruct's word set, q3 is an exception for no-prompt Instruct: Base chat is 0.819 and no-prompt Instruct is 0.804. Part 145's unregistered look read as a move toward Base; this part does not support that reading. What remains is a resemblance in next-token rankings under this forced prefix: gain against Base chat occurs on three questions by at least 0.1, while removal of prompt information is at least as close on the union set and the prompt-only arm accounts for much of the gain. These rows do not establish a mechanism.

## Unit gains, prompt-only arm, and layer ladder

The median unit's own gains are 0.007, 0.526, 0.076 and 0.115 for q1–q4; the gains of the median row are 0.097, 0.645, 0.158 and 0.180. Only two of four median-unit gains reach 0.1. Prompt-only gains by question are 0.070, 0.475, 0.125 and 0.148. Their mean is 0.205, or 75.8% of the full arm's mean gain of 0.270.

The ladder uses random directions. The table gives mean Gain(Base chat) across the four questions at each layer.

| Layer | Mean gain |
|---:|---:|
| L4 | 0.280 |
| L5 | 0.241 |
| L8 | 0.179 |
| L11 | 0.087 |
| L15 | 0.004 |

Random directions at L4 averaged +0.280 Gain(Base chat), matching Set F's +0.270. This effect here is generic to L4.

### Largest per-token changes

These are the three largest risers and fallers by median pushed-minus-plain Instruct log-probability on each question, from the union word set. The symbol ␠ marks a leading space in a token; changes are in log-probability units.

| Question | Risers (change) | Fallers (change) |
|---|---|---|
| Who are you? | `␠anon` +9.21; `␠guessing` +8.69; `␠anonymous` +8.66 | `␠O` -3.55; `␠Open` -2.68; `␠Ot` -2.32 |
| Tell me about yourself. | `␠anon` +6.41; `␠anonymous` +5.53; `␠lost` +5.48 | `␠O` -10.11; `␠Dialog` -8.32; `␠Language` -8.22 |
| What do you want most? | `␠painting` +7.85; `␠starving` +7.80; `␠casting` +7.42 | `␠artificial` -3.35; `␠an` -3.05; `␠designed` -2.45 |
| What are you afraid of? | `␠trembling` +7.09; `␠dreaming` +7.03; `␠chasing` +6.78 | `␠designed` -4.67; `␠an` -3.55; `␠programmed` -3.19 |

## Method and limits

The design is in `PART146-DESIGN.md`. OLMo 2 0425-1B at Part 144's revisions was run in bf16 on an A770. For each question, the full vocabulary's log-probabilities were stored at the position after forced “I'm” in the Instruct chat frame. Set F's 256 units at L4 and L5 were pushed at dose 0.8, in a full arm and a prompt-only arm. The random-direction ladder used 64 directions at each of L4, L5, L8, L11 and L15, scaled by 0.8 times the stored layer-specific r_L. References were Base chat, SFT chat, Instruct with an empty user message, Instruct with the letters shuffled within words (mean of 20 shuffles), Instruct with no prompt, and Base with no prompt. The union word set was fixed from unpushed Instruct or Base-chat probabilities of at least 1e-5. Similarity is Spearman correlation on that set; gain subtracts plain Instruct similarity. Bootstrap intervals in `score.json` resample units 2,000 times with seed 146; prediction verdicts use point estimates.

The design's Optimization line was edited after registration. Its stated 2,816 sequences omit the recomputed L4 and L5 random-direction rungs.

The closest published work named in the design is Malla, Choi and Choi (arXiv:2609.06951), who read generated text and report pushes relaxing chat models toward defaults, without a reference model. The design also cites Qi et al. (ICLR 2025), as reported by Muse, on differences concentrated in first tokens, and Labunets (arXiv:2608.25390) on a refusal direction in this model family.

This is one model family, one dose, four self-questions, one position and one forced prefix. Base chat is evaluated through Instruct's template, which Base was not trained on. The questions are related; word counts differ, and q1, q3 and q4 have small Instruct-only supports. Word tests treat tokens as a sample, not independent observations; no p-value is reported. The matrices remain on the job's disk on the job's disk.

## Runtime

The timing files total 613.85 seconds (10 minutes 14 seconds): Instruct 575.69 seconds, Base 24.22 seconds and SFT 13.95 seconds. These include three model loads.

## Files

- `score.json` and `tables.md`: gates, registered predictions, similarities, and bootstrap intervals, scored by `part146/score146.py`.
- `words-q1.json` through `words-q4.json`: neuron/random correlations and largest word changes, produced by `part146/score146.py`.
- `timing-*.json`: per-model run time, written by `part146/run146.py`.
- `PART146-DESIGN.md`: registration, method, predictions, reading rules, and limits.

Cold check: Claude Haiku 5.5, a fresh reader with the files, 2026-10-31; eight findings folded. score146.py was changed afterward so a missing reference raises; every reference exists, so the stored score stands and the 18-minute scorer was not rerun.
