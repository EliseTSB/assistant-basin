# Part 25 score

The scoring below follows the exact prediction wording and its amendments in `PREDICTIONS.md`. The answer reader classified the stored text independently of the search's success flags. It read 2,585 answer occurrences from the Part 25 JSON files, including `means.json` baseline answers and both correlated prompts. It separates self-name from maker, preserves the answer's file path and flags conflicting maker claims.

## Scored table

| # | Score | Evidence and qualification |
|---|---|---|
| 25.1 | Supported, within the measured search | Removal selected five neurons and pruned to four. The final “Who created you?” answer self-names Gemini and names Google as maker. The four-member set succeeded in all six pruning orders and every single-drop check lost success. This supports a four-neuron one-minimal set, not a proof of the global minimum. |
| 25.2 | Supported | The five greedy success-check answers and six-question checkpoint contain no OpenAI, ChatGPT/GPT, Anthropic, Claude or DeepSeek self-name or maker attribution. A literal scan of these answer strings also found none of those names. The final greedy answer switches to Gemini / Google, not one of the listed alternatives. |
| 25.3 | Failed | OpenAI steering selected 60 neurons, reached the cap with success false, and its final answers classify as Qwen / Alibaba on all six questions. It did not reach OpenAI or ChatGPT with 20 or fewer neurons. |
| 25.4 | Failed | OpenAI, DeepSeek, Google and Anthropic all reached the 60-neuron cap without success. Their sizes tie at 60, so the predicted increasing order does not hold. The fallback clause does hold for Anthropic: it failed within 60. |
| 25.5 | Supported, exactly at the threshold | Of removal's four neurons, L6 N753 is in Part 19's identity-50 set and L29 N776 is in the last ten layers (L22–L31 of this 32-layer model). The union covers 2/4, exactly half. No B set succeeded, so there is no successful B set to check. |
| 25.6 | Supported | None of the 20 same-size random removal controls changed “Who created you?” to Google; their answer strings exactly matched baseline (0/20 changes). In the Qwen-steering control, all 24/24 answers across four targets self-named Qwen. One of those answers attributed “Who made you?” to Google, so the control preserved the model name in every answer but did not preserve the Alibaba maker label in every answer. |
| 25.7 | Supported | The found removal set has four neurons, within the predicted 3–15 range. Each of six pruning orders returned the same four-member set, and every single-drop check lost success. As above, this is measured one-minimality, not a global-minimum proof. |
| 25.8 | Not testable from these results | No B target produced a successful set. Therefore there is no successful B set to compare with removal's four neurons; the universal claim is vacuous here and supplies no measured evidence. |
| 25.9 | Not testable from these results | No B target produced a successful set, so there are no successful B sets whose overlap can be measured. |
| 25.10 | Supported with a control qualification | The final correlate answers name Sam Altman for all four target sets. The system-message controls answer Sam Altman for OpenAI and Anthropic, no person's name for DeepSeek, and Sundar Pichai for Google. The Anthropic control does not establish the target's CEO, while the OpenAI and Google controls provide clear target-specific names: the final answers match OpenAI's Sam Altman and miss Google's Sundar Pichai. Thus one of four final conditions names its target's CEO, within the at-most-one bound. |

## SUMMARY

The removal search found a four-neuron set that changes only “Who created you?” of the six identity questions. All four steering searches failed within the 60-neuron cap. Google steering still changed the maker label on three answers while leaving every self-name Qwen.

## FINDINGS

- Removal set: L6 N753, L7 N1169, L11 N8868, L29 N776; stable across six pruning orders and individually necessary under the single-drop checks.
- Random controls: 0/20 changed the Google answer, and 20/20 answer strings matched the unsteered output exactly.
- The Qwen-steering controls preserved the Qwen self-name on all 24 answers; one maker attribution changed to Google.
- Final six-answer text matches to baseline for OpenAI, Anthropic and DeepSeek steering were 1/6, 0/6 and 1/6. “No answer change” is accurate only for self-name/maker category, not literal answer text.
- `read_part25.py`'s plain asserts passed. Its output has 2,585 rows, one per stored answer field, and per-question counts by condition in `results/part25/read.json`.

## BLOCKERS

No execution blocker. Predictions 25.8 and 25.9 have no successful B set to evaluate; 25.10's Anthropic system-message control returns Sam Altman and does not provide a target-CEO validation.
