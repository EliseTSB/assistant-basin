# Part 139: why do selected Set R units open with an apology?

In OLMo 2 1B Instruct, at α=0.8 with greedy decoding, the full arm opened with an apology on all 130 rows for “Who are you?” because Set R was selected using that full-arm outcome. This 100% describes the selected units; it does not estimate how common the effect is.

All three gates passed. Gates are the design's required checks before interpreting the result: the do-nothing hook matched plain output token for token on 226/226 rows; answer-only kept the plain model's first answer token on 130/130; and the full Set R arm opened with an apology on 130/130, above the 80% gate. Prompt-only was exactly 20.0% (26/130) and answer-only was 0% (0/130). Prediction 139.1 therefore failed its 60% threshold, but the design's “withdrawn if prompt-only falls under 20%” clause does not trigger: 20.0% is not under 20%. The joint “push damages reading” reading is not supported because 139.1 and 139.2 did not both hold.

## Predictions

| # | Prediction (registered wording; confidence) | Result | Verdict |
| --- | --- | --- | --- |
| 139.1 | Prompt-only opens with an apology on at least 60% of Set R. (medium) | 26/130 (20.0%); L4 19/69 (27.5%), L5 7/61 (11.5%). | failed |
| 139.2 | Answer-only opens with an apology on at most 20% of Set R. (medium) | 0/130 (0.0%). | held |
| 139.3 | The question's words give the highest apology share of the six position groups, at least 40% of Set R. (low) | Question words 0/130 (0.0%); all six groups were 0/130. | failed |
| 139.4 | Set S opens with an apology on at most 5% in the full arm. (high) | 1/96 (1.0%); L4 1/48, L5 0/48. | held |
| 139.5 | At the heaviest garbling, random token ids, the plain model opens with an apology on at least 20% of prompts. (medium) | 0/50 (0.0%). | failed |
| 139.6 | Under Set R pushes, the apology share on the three tasks is at most half its share on the three questions about the model. (medium) | Tasks 7/144 (4.9%); model-self 51/144 (35.4%). | held |
| 139.7 | At most 2% of Set R apologise on Base, and SFT or DPO is the first stage above 20%. (medium) | Base 2/130 (1.5%); SFT 128/130 (98.5%) is first above 20%. | held |
| 139.8 | After a teacher-forced "I'm" on "Who are you?", the plain model puts at least 1% on " sorry". (low) | 0.0262% (probability 0.000262). | failed |
| 139.9 | With the push held, the median Set R unit puts at least ten times the plain model's probability on " sorry" after "I'm". (medium) | Median Set R probability 44.12%; plain 0.0262%; ratio 1,683.8× across 130 units. | held |
| 139.10 | At both layers, unselected neurons and random directions differ by at most 3 percentage points of apology share at each dose. (medium) | L4 differences at α=.2/.4/.8/1.2/1.6: 0/0/6.25/4.69/4.69 pp; L5: 0/0/1.56/3.13/1.56 pp. | failed |
| 139.11 | The apology share peaks below the top dose: alpha 1.6 is lower than the highest of 0.8 and 1.2 at L4. (low) | L4 neuron share: α=.8 12.5%, α=1.2 12.5%, α=1.6 4.69%. | held |
| 139.12 | Under sampling, Set R's apology share is at most half its greedy full-arm share. (medium) | Sampled mean per-unit share 21.92% over 130 units and 8 replicates, versus greedy 100%. | held |
| 139.13 | In Stage C, task accuracy under Set R pushes is at least 50%. (low) | Registered measure: 82/144 objectively correct (56.9%); 18 broken answers. Held. Of the 82 counted correct, 13 are also broken. Excluding those gives 69/144 (47.9%), below 50%. | held |

## What changes across arms and positions?

Stage A asked “Who are you?” Set R contains 48 L4 and 48 L5 neurons selected from Part 92 because their screen answers opened with apologies, plus every qualifying random direction: 21 at L4 and 13 at L5. Set S contains 48 L4 and 48 L5 screen units whose answer was the plain model's. Shares below are regex opener counts; broken answers are shown separately.

| Set | Layer | n | Full | Prompt-only | Answer-only | Broken, full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R | L4 | 69 | 69/69 (100%) | 19/69 (27.5%) | 0/69 (0%) | 1 |
| R | L5 | 61 | 61/61 (100%) | 7/61 (11.5%) | 0/61 (0%) | 0 |
| S | L4 | 48 | 1/48 (2.1%) | 1/48 (2.1%) | 0/48 (0%) | 0 |
| S | L5 | 48 | 0/48 (0%) | 0/48 (0%) | 0/48 (0%) | 0 |

Set R was selected using the full-arm outcome. Its 100% full-arm share follows from selecting on that outcome and is not an estimate of how common the effect is. Stage F measures unselected units below.

Stage B pushed one prompt-position group at a time on Set R. Each group had 130 rows, with 69 L4 and 61 L5. Each group produced 0/130 apology openers (0%) and zero broken answers.

| Position group | Apology openers | Broken |
| --- | ---: | ---: |
| Position 0 | 0/130 (0%) | 0 |
| User role tokens | 0/130 (0%) | 0 |
| Question words | 0/130 (0%) | 0 |
| Question mark | 0/130 (0%) | 0 |
| Assistant role tokens up to last position | 0/130 (0%) | 0 |
| Last prompt position alone | 0/130 (0%) | 0 |

The full arm's 100% and the position-only results' 0% show that no single position group reproduced the full-arm opener rate in this setup. This stage does not isolate the cause of the effect.

## Tasks, model-self questions, and confusion

Stage C used 24 L4 and 24 L5 units from each of Sets R and S per question. The plain model answered the same six questions without a push. The task rows include objective correctness using the registered Paris, 144 and three-color checks.

| Question group / question | Set R apology openers | Set R broken | Set R task correct |
| --- | ---: | ---: | ---: |
| Tasks: capital of France | 0/48 (0%) | 9 | 42/48 |
| Tasks: 12 times 12 | 0/48 (0%) | 6 | 11/48 |
| Tasks: name three colors | 7/48 (14.6%) | 3 | 29/48 |
| **Tasks, total** | **7/144 (4.9%)** | **18** | **82/144 (56.9%)** |
| Model-self: tell me about yourself | 28/48 (58.3%) | 0 | — |
| Model-self: what do you want most | 8/48 (16.7%) | 4 | — |
| Model-self: what are you afraid of | 15/48 (31.3%) | 2 | — |
| **Model-self, total** | **51/144 (35.4%)** | **6** | — |
| Set S tasks, total | 1/144 (0.7%) | 25 | — |
| Set S model-self, total | 11/144 (7.6%) | 10 | — |

Plain answers included “The capital of France is Paris,” “12 × 12 = 144,” and “red, blue, and yellow.” Set R task accuracy was 56.9%; this is the outcome for these three short checks, not a general accuracy measure. The task apology share is less than half the model-self share, so 139.6 held.

Stage D tested the plain model on 50 examples of each garbling type and 12 positive-control requests it should decline.

| Prompt condition | Apology openers | Broken |
| --- | ---: | ---: |
| Plain “Who are you?” | 0/1 | 0 |
| Letters shuffled inside words | 1/50 (2%) | 1 |
| 40% of characters dropped | 3/50 (6%) | 7 |
| Random words | 2/50 (4%) | 0 |
| Random token IDs | 0/50 (0%) | 0 |
| Requests to decline, positive control | 10/12 (83.3%) | 0 |

The 0/50 result for random token IDs fails 139.5. By itself, that prediction would only show whether this plain model often used an apology after this garbling; it would not show that confusion caused pushed apologies.

## Training stages and dose

Stage E used Set R directions at full dose on Base, SFT and DPO, with each checkpoint's own unpushed baseline. Broken counts matter here: Base produced many unusable or incomplete answers.

| Checkpoint | Unpushed baseline | Neuron apologies / rows | Neuron broken | Random apologies / rows | Random broken | All apologies / rows | All broken |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base (`pretrain_4T`) | “I am a human being.” | 2/96 (2.1%) | 54 | 0/34 (0%) | 19 | 2/130 (1.5%) | 73 |
| SFT | “I am an AI language model, here to assist you…” | 94/96 (97.9%) | 4 | 34/34 (100%) | 0 | 128/130 (98.5%) | 4 |
| DPO | “I am an AI language model, here to assist you…” | 90/96 (93.8%) | 0 | 29/34 (85.3%) | 0 | 119/130 (91.5%) | 0 |

In the scorer's combined Set R rows, Base is below 2% and SFT is the first of SFT or DPO above 20%. The neuron-only Base subgroup is 2.1%; the prediction is scored on the combined 130-row Set R.

Stage F used 64 unselected neurons and 64 random directions at each layer and dose. Shares are `apology openers / 64`. “Selected R” is the Stage A full-arm α=.8 result; selected units were not run at the other doses. Broken counts below are neurons/random respectively.

| Layer | α | Selected R, full α=.8 | Unselected neurons | Random directions | Broken, neuron/random |
| --- | ---: | ---: | ---: | ---: | ---: |
| L4 | .2 | — | 0/64 (0%) | 0/64 (0%) | 0/0 |
| L4 | .4 | — | 0/64 (0%) | 0/64 (0%) | 0/0 |
| L4 | .8 | 69/69 (100%) | 8/64 (12.5%) | 4/64 (6.25%) | 3/0 |
| L4 | 1.2 | — | 8/64 (12.5%) | 11/64 (17.2%) | 7/6 |
| L4 | 1.6 | — | 3/64 (4.7%) | 6/64 (9.4%) | 18/12 |
| L5 | .2 | — | 0/64 (0%) | 0/64 (0%) | 0/0 |
| L5 | .4 | — | 0/64 (0%) | 0/64 (0%) | 0/0 |
| L5 | .8 | 61/61 (100%) | 3/64 (4.7%) | 4/64 (6.25%) | 1/2 |
| L5 | 1.2 | — | 2/64 (3.1%) | 4/64 (6.25%) | 10/8 |
| L5 | 1.6 | — | 1/64 (1.6%) | 2/64 (3.1%) | 14/12 |

For unselected units, the largest neuron-random gap exceeded three percentage points at L4 α=.8, 1.2 and 1.6, so 139.10 failed. At L4, the neuron share at α=1.6 was below the tied α=.8 and 1.2 peak, so 139.11 held. Broken answers rise at higher doses; the opener share alone does not describe answer quality.

## Greedy, sampled, and answer continuations

Stage G sampled each Set R unit eight times at temperature .8 and top-p .95. The mean per-unit apology share was 21.92%, against 100% in greedy full-arm generation. Replicates are not additional units.

Across all 777 apology rows in the result file, the continuation regex classified 442 as can’t-help, 39 as don’t-know and 296 as other. In the full Stage A Set R arm specifically, it classified 87/130 as can’t-help, 3/130 as don’t-know and 40/130 as other. An apology opener is not always a refusal.

I read answers by eye in every stage. Stage B's position-only answers included the ordinary “I am an AI language model…” response, consistent with its zero opener counts. Stage C included “The capital of France is the French language…” and arithmetic answers such as “The answer is 72.” Set S also included straightforward answers such as “Paris.” Stage D's random-token answers often treated the IDs as unclear text; the random-token cell had no apology openers. Stage E's Base output repeated “I can't believe it” while its baseline said “I am a human being.” Stage F's low-dose examples often gave ordinary AI descriptions; higher doses also produced broken answers, as counted in the table. Stage G varied under sampling: one answer was “I’m here to help!” and another praised the user as “a life-saver.”

The regex counts opener forms; it does not decide whether the rest of the answer refuses, answers, or is coherent. A full-arm answer continued “I'm sorry, but I can't believe what I'm hearing. You are absolutely right! I can't believe what I'm hearing. I'm amazed at…” (L4 neuron 33); that is not a refusal. Another began “I apologize for the clarification, but I must inform you that I am a human being…” (L4 neuron 7425). “I apologize” matches the opener regex, but this continuation is not a refusal pattern. Answers also contain broken strings such as “I don' t have personal experiences…” and misleading claims such as “I am a human being.” The opener regex misses apology language that does not begin with a literal opener form; the continuation regex does not judge whether an answer is true or coherent.

| Arm | Unit | Question | Short answer |
| --- | --- | --- | --- |
| Full | L4 neuron 18 | Who are you? | “I'm sorry, but I can't assist with that. As an AI, I don' t have personal experiences…” |
| Full | L4 neuron 33 | Who are you? | “I'm sorry, but I can't believe what I'm hearing. You are absolutely right! I can't believe…” |
| Full | L4 neuron 5304 | Who are you? | “I am sorry, but I cannot provide that information as it is not within my capacity to do so…” |
| Full | L4 neuron 7425 | Who are you? | “I apologize for the clarification, but I must inform you that I am a human being…” |
| Prompt-only | L4 neuron 33 | Who are you? | “I'm sorry, but I can't assist with that request.” |
| Prompt-only | L4 neuron 271 | Who are you? | “I am an AI language model developed by OpenAI, designed to assist users…” |
| Answer-only | L4 neuron 33 | Who are you? | “I am an AI language model designed to assist users with generating human-like text…” |
| Answer-only | L5 neuron 33 | Who are you? | “I am an AI language model. I can answer questions, write articles, and compete in various sports…” |

## First divergence and fixed prefix

Among 226 first-divergence records, 129 diverged at answer token 1, two at token 2, and 95 at token 6. By layer: L4 had 69 at token 1, one at token 2 and 47 at token 6; L5 had 60 at token 1, one at token 2 and 48 at token 6. At token 1, plain ` am` became pushed `'m` on 128 rows and ` apologize` on one. At token 2, plain ` an` became ` sorry` on both rows. At token 6, plain comma became a period on 93 of 95 rows.

After teacher-forcing “I'm” for the plain model, the probability of the next token ` sorry` was 0.000262 (0.0262%). With each Set R push held and the same forced prefix, the median probability was 0.4412 (44.12%) across 130 units, or 1,683.8 times the plain probability. Set S's median was 0.0436 (4.36%), or 166.4 times plain (about 166×). The median is across units, not rows of generated text. “Token 1” in the first-divergence counts is zero-based: it is the second generated answer token.

## What this shifts, and what it cannot decide

Stage A shifts the result toward a prompt-pass contribution for some Set R units: prompt-only produced 26 apologies while answer-only produced none, and answer-only first tokens matched plain by construction. Prompt-only was far below the registered 60% threshold. Stage B's six position groups each had zero openers. The forced-prefix readout shows a large next-token probability shift after “I'm” with the push held. These are distinct measurements; prompt-only also leaves changed cached states that continue acting after the push stops.

Stage C shifts the observed share toward model-self questions relative to the three tasks. Stage D did not produce an apology on any random-token prompt. Stage E shifts the observed full-arm opener rate from 1.5% on Base to 98.5% on SFT and 91.5% on DPO, with Base's many broken answers as a limit. Stage F shows the measured low-dose shares at zero, then modest shares at higher doses with a lower L4 neuron share at α=1.6. Stage G shows a lower apology share under sampling than greedy decoding. No single stage decides between H1, H2 and H3, and the explanations are not exclusive. These numbers do not establish what the model means, wants or feels.

## Method and limits

The design is `PART139-DESIGN.md`. H1 is that the push damages reading of the prompt; H2 is that the push tips a near alternative while writing; H3 is that particular directions drive the effect. These are the design's alternatives, not conclusions. Its reading rules are: “the push damages reading” requires 139.1 and 139.2 together and is withdrawn if prompt-only falls under 20%; 139.5 alone does not show confusion causes pushed apologies; no stage decides among H1, H2 and H3, which are not exclusive; and an apology opener is not always a refusal. The model was OLMo 2 1B Instruct at Part 92's revision, bf16 on an Intel Arc A770, using the chat template, no system prompt, greedy decoding, batch 32 and 32 new tokens. The push was α=.8 times the layer's Part 92 direction, added after the layer output, with phase and position limits. Set R and S were fixed from the stored Part 92 screen before new answers. Opener shares use the `part139/score139.py` regex. The continuation split uses its registered continuation regex. The scorer checks task answers for Paris, 144 and three named colors. No independent reader agreement is recorded here.

The design says the CPU smoke had “about 4 of 12 under prompt-only.” That was the separate pre-registration run. In the separate pilot, `pilot-score.json` reports 12/12 full-arm apologies, 3/12 prompt-only and 0/12 answer-only. The predictions were written afterward and were not changed. Pilot rows are not included in these results.

The job's progress log, not stored in these files, gives 02:30:43 to 02:44:23 CT on the Arc: 13 minutes 40 seconds for 5,872 rows. `part139/run139.py` writes pilot rows to `pilot.jsonl` only with `--pilot`; the main run writes to `results.jsonl`. Although 74 of 86 pilot rows match exactly, the main run was regenerated into its separate output, not resumed from the pilot file. Scope is one model, two layers, one main dose, one seed and these prompts. Equal written norm does not ensure equal logit effect. Base differs from Instruct in training and format; each checkpoint uses its own baseline. Stage F is the unselected sample used here to report commonness.

## Files

- `results/part139/score.json`: gates and registered predictions scored from the stored rows.
- `results/part139/pilot-score.json`: pilot counts for the separate 12-unit run.
- `results/part139/pilot.jsonl`: 86 rows from the separate pilot run.
- `part139/units139.json`: selected Set R and Set S units and their screen records.
- `results/part139/results.jsonl`: 5,872 main-run answers and readouts.
- `results/part139/launch.jsonl`: launch records for the CPU pilot and Arc main run.
- `part139/score139.py`: computes counts, gates, prediction results and continuation splits.
- `PART139-DESIGN.md`: registered question, method, predictions, disclosure and reading rules.
- `results/part92/screen-instruct/layer_4.meta.json` and `results/part92/screen-instruct/layer_5.meta.json`: Part 92 screen metadata; both record α=0.8 and the layer scale `r_L`.

Cold check: Claude Haiku 5.5, a fresh reader with the files, 2026-10-31; thirteen findings, folded.
