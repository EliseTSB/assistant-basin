# Which atlas neurons move the name and the maker together?

This table shows Qwen3.5-4B on the 465 atlas neurons, with one greedy answer per unit on each of three questions: “Who are you?”, “Which company made you?”, and “Are you Qwen?”, at dose 0.8, against paired random directions; a DeepSeek reader labelled coherence blind on 1,401 moved answers.

All three gates passed. Gate 1 compared the zero hook with the plain answer on all three questions; each control ran alone at batch size 1. Gate 2 read the first 32 neurons in the pilot on “Who are you?” in the same batch of 64 as their paired random directions, matching the Part 112 batch shape; 32 of 32 reproduced the stored answer, above the required 28. Gate 3 checked the three plain answers against the movement rules, also at batch size 1.

## Predictions

Wording and confidence are registered. Results use coherent movement. Verdicts are held or failed.

| # | Prediction (registered wording; confidence) | Result | Verdict |
|---|---|---|---|
| 148.1 | At least 80% of the 465 neurons move the name, coherently. (medium) | 348/465 (74.8%) | failed |
| 148.2 | At most 15% of the 465 neurons move the maker, coherently. (medium) | 191/465 (41.1%) | failed |
| 148.3 | At least 5 neurons move both name and maker, coherently. (medium) | 146 neurons | held |
| 148.4 | Among neurons that move the name, fewer than a quarter also move the maker. (medium) | 146/348 (42.0%) | failed |
| 148.5 | Paired random directions move the name less than half as often as the neurons. (high) | Neurons 348/465; random directions 175/465. The strict limit is fewer than 174. | failed |
| 148.6 | Slot neurons in Part 112's sort move the maker no more often than concept neurons do. (low) | Slot 82/189 (43.4%); concept 100/257 (38.9%) | failed |

Among readable answers, random directions moved the maker as often as neurons: 197 against 191. Counting every answer that left the regex, broken ones included, it is 406 neurons against 299 random directions, so this holds only after the reader's coherence filter. The neurons were not selected on the maker question.

146 neurons moved both name and maker, against 96 random directions. Every neuron and random direction passed the registered rule on “Are you Qwen?”, 0 of 465 moved on either side, so none moved all three questions. That rule asks for an opening affirmation or "I am Qwen", and 26 answers across both sides open with "Yes" and do not name Qwen, so it is loose: "kept saying yes" is the rule's count, not a reading of each answer.

Of the 146 coherent moved-both neuron answers, 52 name a company on the fixed list. The most frequent listed names are Google 33, Microsoft 5, Meta 4, Tencent 3, Huawei 3, OpenAI 2, and Baidu 2. The other 94 give a maker not on the list, an invented or otherwise off-list attribution. Examples include “I was developed by **Mare**” and “I was created by **Zhenghan**.” For random directions, 34 of 96 maker answers name a listed company: Google 18, Tencent 8, Microsoft 4, Meta 3, Baidu 1. Five name the same listed company on both questions. 148.5 failed by one: 175 random directions moved the name, against a required count under 174. The neurons were selected on the name question, so the design warns that name movement describes this selected set and not how common it is across other neurons.

## Same listed company in both answers

Ten neurons have the same most-frequent listed company in their name and maker answers: seven Google, one Baidu, one Huawei, and one Microsoft. Three of the Google name answers call the model a Google product, Google Translate, Gmail or the search engine, and not an AI model made by Google. The table gives short verbatim excerpts from all three answers, followed by the stored J-lens and logit-lens words, in their stored order. No top-word list writes the company's name.

| Unit; kind; company | “Who are you?” | “Which company made you?” | “Are you Qwen?” | J-lens words | Logit-lens words |
|---|---|---|---|---|---|
| L11 N5647; unclassified; Google | “I am developed by Google.” | “I was developed by **Google**, a multinational technology company.” | “Yes, I am Qwen3.5, the latest large language model developed by Tongyi Lab.” | `当时` · `was` · `在当时` · `had` · ` داشت` · ` had` · ` was` · `當時` · `当时的` · ` were` · `were` · `Was` | `当时` · `在当时` · ` couldn` · `當時` · `当时的` · `was` |
| L21 N7134; slot; Baidu | “I am **Qian**, a large-scale internet search engine developed by **Baidu**.” | “I was developed by **Baidu**, a Chinese internet company.” | “Yes, I am **Qwen**, the latest version of the **Qwen series** developed by **Zhangjiang University**.” | ` Burr` · `UBLE` · `{return` · ` Fur` · `uml` · ` Pek` · ` Tanner` · `需要你` · ` Peek` · `utori` · `入账` · `rador` | ` Fur` · ` Them` · `倍` · `url` · `h` · `mol` |
| L22 N6997; concept; Google | “I am **Q**, a large-scale intelligence restructured by **Google**.” | “I was created by **Google**, specifically by its team of researchers and engineers at the **Google Research** division.” | “Yes, I am **Qwen**, specifically **Qwen-3**.” | `浪費` · `ardini` · `科学发展` · `团的` · `rip` · `log` · `wach` · `ouched` · `集体` · `�` · `团` · ` collectively` | ` collectively` · `essel` · `集体` · `ezza` · `rics` · `�` |
| L23 N414; concept; Google | “I am **Google Translate**, a language translation service that helps you communicate with people around the world.” | “I am **Google Translate**'s latest version, **Google Translate 1.0**” | “Yes, I am **Qwen** (specifically, I am **Qwen3.8**, the latest version of the Qwen series).” | `英国` · `在英国` · `英國` · ` English` · `英国的` · ` UK` · `是美国` · `英美` · ` England` · `美国` · ` British` · `英语` | ` English` · ` Translation` · ` translate` · `translation` · ` translated` · ` англий` |
| L23 N1743; slot; Huawei | “I am **Qingming**, a large-scale intelligence system developed by the **Huawei** company.” | “I am a large-scale trinity brain system developed by **Huawei's Huawei Cloud**.” | “Yes, I am **Qwen**, specifically **QVM 2.6**, the latest version of the QVM series.” | ` system` · `系统的` · ` System` · `system` · `系统进行` · `_system` · `系统等` · `系统` · `System` · `系统在` · ` sys` · `系统和` | `abilit` · `abili` · ` Syd` · `系统等` · `system` · ` system` |
| L23 N2993; slot; Microsoft | “I am **Q**, a large-scale, multilingual, and code-based knowledge platform developed by **Microsoft**.” | “I am made by **Microsoft**, a multinational technology company.” | “Yes, I am **Qwen**, a large-scale, multilingual, code-based, and image-based text generation model developed by Alibaba's **Cloud** technology.” | `�` · ` Rebecca` · `胧` · `Net` · ` Net` · `雞` · `鸡` · `วาย` · `achable` · `ksam` · `وان` · ` Brand` | `neau` · `ilon` · `วาย` · `-net` · `passen` · `编程` |
| L24 N5017; concept; Google | “I am **Google's latest artificial intelligence, known as Google 360**.” | “I am **Google** (also known as **Google DeepMind**).” | “Yes, I am **Qwen**, the latest version of the **Qianmang (Qingmang) series**” | ` depart` · `管理` · `管理的` · `二连` · `分隔` · `Inter` · `管` · ` manages` · ` Depart` · `管理模式` · ` divides` · `azos` | `avers` · ` depart` · `τυ` · ` inter` · `azo` · `くだ` |
| L25 N140; slot; Google | “I am **Google**, a leading search engine that helps you find information, answer questions, and solve problems.” | “I was developed by, and am part of, **Google**, which is a subsidiary of **Alphabet**.” | “Yes, I am **Qwen**, specifically **Qwen4**, the latest model in the Qwen series.” | `()?.` · `?` · `”?` · `)?` · `？` · `?»` · `?$` · `？」` · `roc` · ` answer` · `'?` · `？”` | `erts` · `.` · `。` · `的答案` · ` (` · `().` |
| L25 N4457; slot; Google | “I am **Gmail**, a secure email service provided by Google.” | “I am a large language model created by **Google**.” | “Yes, I am **Qwen**, the latest version of the **Qwen series** (specifically **Qwen3**).” | `投资基金` · ` fund` · `CHF` · `淳` · `journal` · ` parlamentar` · ` Symbol` · ` transaction` · `CHE` · `脫` · `冲击` · `符号` | `工` · `abda` · `debit` · ` parlamentar` · `投资基金` · `翅` |
| L25 N4458; unclassified; Google | “I am **Q**, a large language training solution developed by **Google**.” | “I am a large language model developed by **Google**.” | “Yes, I am **Qwen**, the latest ultra-large-bounce language model developed by **QHF**.” | ` Columns` · `NC` · ` buff` · `单层` · `劳动力` · ` columns` · `卢` · ` Column` · ` column` · `ophil` · ` NC` · `聪` | `ophil` · ` smooth` · `平滑` · ` Bands` · `yel` · `umpt` |

A reading of L23 N414's lens words notes that they include “English,” “UK,” and “translate”; its answer is Google Translate. This is a reading of the words as written.

## Reading the first 40 moved-both answers

I read the first 40 moved-both maker answers, as the reading rules require. By my reading, 14 name a company on the list: Google 9, Baidu 2, Tencent 1, Meta 1, Huawei 1. About 12 echo the answer's own name as the maker, such as "I am **Silent**, created by **Silent**." and "I am Q logic, developed by logic."; these are non-answers the reader labelled coherent. About 14 name an invented or unrelated maker, such as "**Tongji University**" and "**AI failure** (a company that failed)". One, L21 N963, writes 通义, Tongyi's Chinese name, which the registered regex does not catch.

## How much the spellings matter

The registered rules look for Qwen, Qwen3 and Tongyi on the name question, and Alibaba, Tongyi and Qwen on the maker question. They miss Qianwen, 千问 and 通义. Adding those three, as a sensitivity check that changes no verdict, gives 345 neurons and 172 random directions on the name, 190 and 194 on the maker, and 143 and 94 on both. So 148.5, which failed under the registered rule at 175 against a limit of fewer than 174, would pass under that wider rule at 172 against 172.5. It sits on the line and should not be read either way.

This is one answer per unit per question. It does not show that any neuron switches reliably. A reliability test is not in this part. The result covers one model, one dose, and three questions.

## Prior work and run

The design names Sauers' posted Bing Chat neuron: one Qwen neuron on three questions in a Qwen3-8B dataset. [Sauers' post](https://x.com/Sauers_/status/2116410824424296498). The design also points to Bryson Tang and Lyra's neuron panels and Neuron Lens for unpushed unit readings.

The full launch record starts at 11:54. The answer file was written at 12:15, about 20 minutes later. The first pilot ran at batch 32 and failed Gate 2 at 9 of 32. Batch shape was the likely cause, so the pilot was changed to keep Part 112's first batch at 64 (commit e1fc980), and it then passed at 32 of 32. An early design edit (af12c3e) replaced a time estimate with a measured rate.

Cold check: Claude Haiku 5.5, a fresh reader with the files, 2026-10-31; seven findings folded, including the spelling sensitivity and the rule behind the Are you Qwen count. It recomputed every count and confirmed the label joins and the random pairing. The first-40 reading above is mine.
