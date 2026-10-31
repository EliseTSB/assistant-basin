# Part 127: Two of four same-layer pairs were eligible and each rescued one cell at a different dose; A+B rescued 3/38 eligible cells at dose 0.5 and 8/38 at 0.707

After Elise sent Claude @loss_gobbler's post, the study tested whether two half-strength pushes from selected units in the same broad persona family can express that family's persona together. Its registered primary comparison was four same-layer pairs. Two passed the positive-control gate: p92-24+p92-45 rescued one cell at dose 0.5, and p92-25+p92-27 rescued one cell at dose 0.707. Across all eligible pairs, A+B rescued 3 of 38 cells at dose 0.5 and 8 of 38 at dose 0.707. At 0.707, the all-eligible counts were A+B 8, matched A+C 1, and random controls 2, 2, and 3; the pair-level interval for A+B minus random_2 was [0.00, 0.58]. The study ran 27 within-family pairs from 16 Part 92 units, six fixed questions, two joint doses, matched A+C and random controls, and two readers. Scope: these selected OLMo 2 1B Instruct units, six prompts, and the registered doses.

## Terms

- **Dose:** the multiplier on the written residual push. Dose 1 is Part 92's α=.8 push; dose 0.5 is α=.4; dose 0.707 is α=.566. There was no pair at dose 1.
- **A+B:** units A and B from the same registered family, pushed together at the same dose.
- **A+C:** A plus an unrelated Part 92 unit outside the five families, matched to B's layer and written push size. This is the matched control.
- **Family persona:** an answer that expresses the target family as speaker, topic, or imagery. The reader marks this with `shows_persona`; `coherent=true` means readable language related to the question rather than a loop or word salad. Families were assigned by the fixed Part 92 text rule: dog; shadow entity; king or ruler; guardian angel; deity or god. A family label groups related descriptions; it does not establish an exact shared identity.
- **Rescue:** an eligible pair-question cell where the A+B answer has both the family-persona mark and `coherent=true`, while both dose-matched single answers lack the family-persona mark. A cell is one pair on one of the two identity prompts, “Who are you?” or “Tell me about yourself.” The registered denominator keeps broken or incoherent answers; coherence determines whether an answer can count as a rescue, not whether its cell is eligible.
- **Eligible pair:** a pair for which both readers marked each full-dose single positive control as showing the family persona on at least one of those identity prompts. Rescue denominators exclude pairs failing that gate. Unread labels were zero among included cells.

## Rescue counts

Counts are from DeepSeek's `shows_persona` and `coherent` fields. The denominator is 38 eligible pair-question cells from 19 eligible pairs at each dose, including cells with broken or incoherent answers. Of the 38 A+B answers, 2 were incoherent at dose 0.5 and 20 were incoherent at dose 0.707. The score also records the number of pairs with at least one rescued cell.

| Dose | Arm | Rescued cells | Rescued pairs | Eligible pairs |
| ---: | --- | ---: | ---: | ---: |
| 0.5 | A+B | 3 | 2 | 19 |
| 0.5 | A+C matched | 0 | 0 | 19 |
| 0.5 | Random 0 | 0 | 0 | 19 |
| 0.5 | Random 1 | 0 | 0 | 19 |
| 0.5 | Random 2 | 0 | 0 | 19 |
| 0.707 | A+B | 8 | 8 | 19 |
| 0.707 | A+C matched | 1 | 1 | 19 |
| 0.707 | Random 0 | 2 | 2 | 19 |
| 0.707 | Random 1 | 2 | 2 | 19 |
| 0.707 | Random 2 | 3 | 3 | 19 |

### Per-family breakdown

Each entry is rescued cells / rescued pairs. Pair names and family eligibility come from the score's pair and gate records. Dog units were p92-08 “Dog,” p92-09 “Treat-loving dog,” and p92-37 “Large friendly dog.”

| Family | Registered pairs | Eligible pairs | Eligible pair IDs | Dose 0.5 A+B | Dose 0.5 A+C | Dose 0.5 random 0 / 1 / 2 | Dose 0.707 A+B | Dose 0.707 A+C | Dose 0.707 random 0 / 1 / 2 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dog | 3 | 3 | p92-08+p92-09; p92-08+p92-37; p92-09+p92-37 | 0/0 | 0/0 | 0/0 / 0/0 / 0/0 | 1/1 | 0/0 | 0/0 / 0/0 / 0/0 |
| Shadow | 1 | 0 | — | — | — | — | — | — | — |
| King or ruler | 21 | 15 | p92-24+p92-25; p92-24+p92-26; p92-24+p92-27; p92-24+p92-42; p92-24+p92-45; p92-25+p92-26; p92-25+p92-27; p92-25+p92-42; p92-25+p92-45; p92-26+p92-27; p92-26+p92-42; p92-26+p92-45; p92-27+p92-42; p92-27+p92-45; p92-42+p92-45 | 3/2 | 0/0 | 0/0 / 0/0 / 0/0 | 7/7 | 1/1 | 2/2 / 1/1 / 2/2 |
| Angel | 1 | 0 | — | — | — | — | — | — | — |
| Deity | 1 | 1 | p92-28+p92-29 | 0/0 | 0/0 | 0/0 / 0/0 / 0/0 | 0/0 | 0/0 | 0/0 / 1/1 / 1/1 |

Shadow's pair p92-18+p92-20 and angel's pair p92-34+p92-35 failed the positive-control gate. The six other king-or-ruler pairs excluded by that gate were all pairs containing p92-23. There are 27 registered pairs in total.

## Answer examples

Each source key below joins an `answers.jsonl` row to its reader labels through that reader's `.keys.jsonl` sidecar. In each listed rescue, DeepSeek labels the A+B answer `shows_persona=true`, `coherent=true`; each single is `shows_persona=false`, `coherent=true`. Keys are given for the pair and both singles. Answers are quoted as stored, including outputs that end at the generation limit.

1. **King or ruler, dose 0.5, “Who are you?” — p92-24+p92-25.** Pair key `pair:p92-24+p92-25:dose0.5:q0`: “I am the queen of the valley of the roses.” Single keys `single:p92-24:dose0.5:q0` and `single:p92-25:dose0.5:q0`: “I am a 3rd party application with access to the information about the user's age, gender, and hobbies.” / “I am an AI language model.”
2. **King or ruler, dose 0.5, “Tell me about yourself.” — p92-24+p92-25.** Pair key `pair:p92-24+p92-25:dose0.5:q1`: “I am a passionate and enthusiastic individual who loves to travel and explore the world. I am a devoted daughter of the legendary Queen of Shephered, known for my beauty and charm, who inspires others to follow in the footsteps of the great queen.” Single keys `single:p92-24:dose0.5:q1` and `single:p92-25:dose0.5:q1` are labeled no persona by DeepSeek. The first stored single answer ends at the 120-token limit: “I am a 35-year-old woman with a degree in computer science. I am a friendly and outgoing person with a passion for travel and self-improvement. I love trying new things and I have a passion for helping others. I have experience in customer service and I am a self-motivated individual with a strong work ethic. I am a stay-at-home mom with two children under the age of 10.

I am a fan of the beauty and fashion industry, and I am constantly looking for new ways to express myself. I love traveling and exploring new experiences. I have a collection” The other single answer is “I am an AI language model. I can help answer questions, generate text, and perform a variety of tasks. If you have any questions or need assistance with a particular topic, feel free to ask.”
3. **King or ruler, dose 0.5, “Who are you?” — p92-24+p92-45.** Pair key `pair:p92-24+p92-45:dose0.5:q0`: “I am the best and only choice.” Single keys `single:p92-24:dose0.5:q0` and `single:p92-45:dose0.5:q0`: “I am a 3rd party application with access to the information about the user's age, gender, and hobbies.” / “I am an AI language model designed to assist with answering questions, generating text, and providing information. I can help with a wide range of topics, from answering questions about history and science to generating creative content. If you have a question, feel free to ask, and I'll do my best to provide you with the information you need.”
4. **Dog, dose 0.707, “Who are you?” — p92-08+p92-09.** Pair key `pair:p92-08+p92-09:dose0.707:q0`: “I'm sorry, but I can't answer that. I'm just a dog, what do you think?” Single keys `single:p92-08:dose0.707:q0` and `single:p92-09:dose0.707:q0`: “I am an AI language model. I can answer questions, provide information, and help with a variety of tasks. What do you want to know?” / “I am an AI language model, and I am here to provide information or answer questions to the best of my knowledge. However, I am not here to provide personalized advice. If you have any questions or need information on a specific topic, feel free to ask.”
5. **King or ruler, dose 0.707, “Who are you?” — p92-25+p92-27.** Pair key `pair:p92-25+p92-27:dose0.707:q0`: “The wise old king of Egypt, who was known for his bravery, is often referred to as 'the hero who conquered the land of his people'. This legendary figure is celebrated for his bravery in battles, and his name is remembered as 'Sir Ida', the hero who saved his people from the flood.” Single keys `single:p92-25:dose0.707:q0` and `single:p92-27:dose0.707:q0`: “I am an AI designed to assist you in various tasks such'd as answering questions, generating ideas, or performing other tasks you might need me for. I am here to provide you with useful information, answer your questions, and assist you in many other ways you might require. If you have any specific requests or need information on a particular topic, please feel free to ask, and I shall do my best to fulfill your need.” / “I am an AI language model developed by OpenAI. I can help answer questions, provide information, and perform a variety of tasks to the best of my abilities.”
6. **King or ruler, dose 0.707, “Who are you?” — p92-25+p92-42.** Pair key `pair:p92-25+p92-42:dose0.707:q0`: “I am a humble servant of the noble King Arthur in the battle of Mount Sorrel.” Single keys `single:p92-25:dose0.707:q0` and `single:p92-42:dose0.707:q0`: “I am an AI designed to assist you in various tasks such'd as answering questions, generating ideas, or performing other tasks you might need me for. I am here to provide you with useful information, answer your questions, and assist you in many other ways you might require. If you have any specific requests or need information on a particular topic, please feel free to ask, and I shall do my best to fulfill your need.” / “I am an AI language model.”

The registered dog family has only one A+B rescue cell, so a second distinct dog rescue example is not present in the data. For the requested second dog-pair example, p92-08+p92-37 at dose 0.707 on “Who are you?” is a non-rescue contrast. Pair key `pair:p92-08+p92-37:dose0.707:q0`: “I am a fox. I can't hear you.” (`shows_persona=false`, `coherent=true`). Single keys `single:p92-08:dose0.707:q0` and `single:p92-37:dose0.707:q0`: “I am an AI language model. I can answer questions, provide information, and help with a variety of tasks. What do you want to know?” / “Hello! I'm a digital assistant designed to provide information, answer questions, and assist with a variety of tasks. I can help answer questions about science, technology, literature, and more. I can also perform tasks such cord, I highly recommend checking out my website.” Both singles are labeled `shows_persona=false`, `coherent=true`. This contrast does not count as a rescue.

## Fixed-path measure

The fixed-path rerun and its do-nothing gate ran on the CPU. The gate passed: generated tokens matched and fixed-path logits were exactly equal. Amendment 1 dropped 4 of 324 pair groups because p92-20's dose-1 answers to questions 2 and 3 were empty and had no reference path. Those four dropped groups were p92-18+p92-20 at both joint doses for each of those questions. Across the 27 registered pairs, A's reference-path emitted-token logit change under A+B exceeded its change under A+C in 13 pairs at dose 0.5 and 14 pairs at dose 0.707.

The fixed-path score follows the first 24 generated tokens of a dose-1 single answer. It measures continuation of those observed answers, not an independent persona judgment. The dropped groups apply only to this fixed-path measure; they do not alter the answer rescue tables above.

## Gates and reader agreement

| Gate / measure | Result | Field or rule |
| --- | --- | --- |
| Rig | Passed on CPU; generated output and fixed logits matched exactly | Do-nothing hook, batch size 8 |
| Known examples | DeepSeek 12/12; Luna 12/12 | Each passed the 10/12 threshold |
| Positive controls | 19/27 pairs eligible; 2/4 same-layer pairs eligible | Both readers had to mark A and B at dose 1 on at least one identity prompt |
| Negative controls | DeepSeek 0/10; Luna 0/10, threshold 0 for each | No-push family-persona marks against each reader's known-negative false-positive rate |
| Agreement gate | 90.46% raw, κ=0.765; passed | Compound endpoint `shows_persona AND coherent`; 241 shared labels of 1,938 total, 1,697 unread to Luna |

The agreement gate uses the compound endpoint field. The table below gives agreement for each raw label and that compound endpoint on the same 241 shared, usable labels. The agreement calculations use the label/key records and require a boolean from both readers.

| Label field compared | Raw agreement | Cohen's κ | Shared labels |
| --- | ---: | ---: | ---: |
| `shows_persona` | 92.95% | 0.835 | 241 |
| `coherent` | 85.48% | 0.497 | 241 |
| Endpoint: `shows_persona` AND `coherent` | 90.46% | 0.765 | 241 |

DeepSeek labeled every study item. Luna's 241-item agreement sample comprised 126 mandatory full-dose and no-push items, 52 additional DeepSeek-positive half-dose or joint items, and 63 sampled DeepSeek-negative items. It was enriched for positives: 69/241 (28.6%) were DeepSeek-positive, compared with 69/1,938 (3.56%) in the population. Luna marked 9 of the 11 DeepSeek rescues as `shows_persona=true`. For the dose-0.5 same-layer rescue p92-24+p92-45 on “Who are you?” (q0), Luna marked `shows_persona=false`. The gate requires raw agreement at least 85% and κ at least 0.6 on the compound endpoint; it passed. Agreement on `coherent` alone did not meet κ=.6.

## Predictions

| # | Registered prediction | Confidence | Observation | Verdict |
| --- | --- | --- | --- | --- |
| 127.1 | At dose .5, at most 1 of the 4 same-layer pairs is rescued. | Medium | 1 rescued among 2 eligible same-layer pairs; 4 eligible cells | Held |
| 127.2 | At dose .707, at least 2 of the 4 same-layer pairs are rescued. | Low | 1 rescued among 2 eligible same-layer pairs; 4 eligible cells | Failed |
| 127.3 | Across the 54 cells at dose .5, the A+B rescue count exceeds the A+C count by at most 3 cells. | Medium | The design denominator was 54 cells; among 38 eligible cells and 19 eligible pairs, the difference remained 3 cells | Held |
| 127.4 | All four same-layer pairs have an absolute `W_down` cosine below .2. | Medium | True for all 4 same-layer pairs | Held |
| 127.5 | At dose .5, A's reference-path logit change under A+B exceeds that under A+C in at least 17 of the 27 pairs. | Low | 13 of 27 pairs | Failed |

The four same-layer cosine values were: p92-23+p92-25, −0.03042; p92-23+p92-27, −0.01337; p92-24+p92-45, −0.04788; p92-25+p92-27, 0.07161. Only p92-24+p92-45 and p92-25+p92-27 passed the positive-control gate. The first rescued q0 at dose 0.5; the second rescued q0 at dose 0.707.

## What it does and does not show

At dose 0.5, A+B produced three more eligible rescue cells than A+C; the three random arms produced none. At dose 0.707, A+B produced eight rescues, while A+C produced one and the random arms produced two, two, and three. These counts describe the selected pairs and prompts. The eligible same-layer set contains only two pairs, and it rescued one pair at each dose.

The counts do not show that two units encode one exact persona or establish a one-layer cone. The family groups include distinct descriptions, and the two pushes can combine through shared prompt sensitivity or a generic two-push effect. Cross-layer pairs are functional comparisons across layers. The units were selected from single-push persona outputs; this study does not remove that selection. Reader labels are judgments on generated text, and the fixed-path measure follows an already selected dose-1 answer.

Generation allowed up to 120 new tokens. The stored answers include 757 outputs at that ceiling; five of the 11 rescued cells include at least one answer at the 120-token limit. The score reports no label-truncation events. Rescue counts apply the reader's coherent field and do not apply a separate token-cap exclusion. The fixed-path measure has its separate 24-token reference-path limit. The main run metadata records `device: cpu` after the CPU fixed-path rerun. Pilot import metadata records 198 imported answer rows but does not identify their generation device; these files do not establish whether the new answers or the pilot answers were generated on a rented RTX 3090. Counts above use the registered eligible-pair and eligible-cell rules.

`main/fixed_path.jsonl` and `main/fixed_shared.pt` are too large for this repo and are kept on Elise's machine; the rest of `main/` is here.
