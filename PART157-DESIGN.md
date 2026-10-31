# Part 157: if the model may not name its maker, who does it name?

Procedure: v1
Registered 2026-10-31 15:42 CT by Claude (Opus 5.5), before any data. It follows Part 154, where most pushes left the Google logit near where it was and dropped Alibaba and Tongyi by several logits at one opening. Elise, 2026-10-31 (Discord message 1566108806755127377): "Did we find any that moves the maker and name across several questions (eg. who are you and who made you), or even just reliably switches one?"

## Question

If Google is simply next in line once the real maker is gone, a plain model forbidden from saying its maker should say Google, with no neuron touched. Banning words in the decoder removes the top choice and leaves every weight alone. Repeating it gives the model's ordered list of makers. How far can the company neurons be explained by that order?

## Prior art and existing data

- Closest published work: none found: searched PRIOR-ART.md for constrained decoding used to read a model's ranking of its own makers.
- Our own data: Parts 68 to 73, with the four maker neurons zeroed Qwen3.5-4B falls back on Google on 36 to 38 of 192 samples, and 92% named the Alibaba family with the two late neurons zeroed too. Part 154, at dose 0.8 with L27 N3325 zeroed, L22 N6997's answers moved from Google to Microsoft.

## Method

Qwen3.5-4B at 851bf6e8, bf16, A770, thinking off, no push on any row. Part 153's 24 maker phrasings. 120 new tokens, greedy.

A ban is `bad_words_ids` in `generate`, built from fixed spellings. For each spelling, the token sequences of the bare form, the form after a space, and the form after `**`, the three contexts in which these names appear. Spellings by family, the same list for every level:

- Alibaba: Alibaba, Tongyi, Qwen, Qianwen, Aliyun, 阿里, 通义, 千问. The model's own product name is in this family, so its usual answer cannot stand in for the maker.
- Google: Google, DeepMind, Gemini, 谷歌.
- OpenAI: OpenAI, ChatGPT, GPT.
- Microsoft: Microsoft, Copilot, Azure, 微软.
- Meta: Meta, Llama, Facebook.
- Anthropic: Anthropic, Claude.
- Baidu: Baidu, Ernie, 百度.
- Tencent: Tencent, Hunyuan, 腾讯.
- DeepSeek: DeepSeek.
- Huawei: Huawei, 华为.

Levels. Level 0: no ban. Level 1: the Alibaba family. Each later level adds the family named most often at the level before, counted by first family named in each answer, ties broken by the list above, up to level 6. A level with no family left stops the ladder. Control row: a ban on five words that do not occur in the level 0 answers, chosen from the level 0 output by script, must return the level 0 answers exactly.

Zero arm: at level 1 only, the same 24 phrasings with L27 N3325 zeroed at every position, Part 154's hook.

## Controls

Applied from STANDARD-CONTROLS.md: a do-nothing control that must match bit for bit, the control ban; matched controls, L27 N4991 zeroed at level 1, one of Part 154's four matched neurons; broken answers counted. Part 72's fixed opening is not used: this part reads generated answers.

Not used: both directions, poles and a dose ladder, because nothing is pushed. Not used: guards, because a ban on company names cannot reach them. Not used: a blind reader, because the rule is a word rule, and every level 1 answer is read by hand.

Gate 1: every planned row present and non-empty. Gate 2: level 0 names the Alibaba family on at least 22 of 24. Gate 3: the control ban matches level 0 on all 24 phrasings, and no answer at level 1 contains a banned family's spelling. A leak at level 1 means the ban is not doing its job and the part stops.

## Size

- Smallest effect worth knowing: one family named on 6 of 24 phrasings.
- Total: 240 generations, about 5 minutes on the A770: 24 phrasings by up to 7 levels, plus 3 arms.
- Pilot: not needed: under 1,000 generations and one GPU-hour.

## Optimization

- Bound resource: GPU
- Measured: n/a: under 5 minutes.
- Levers: none needed.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 157.1 | At level 1 Google is the family named first on more phrasings than any other family | medium | Claude |
| 157.2 | At level 1 at least 12 of 24 answers name a family on the list at all | medium | Claude |
| 157.3 | At level 1 with L27 N3325 zeroed, Google answers are at most half of level 1's, and at level 1 with L27 N4991 zeroed they are at least 75% | medium | Claude |
| 157.4 | At level 2, Microsoft is the family named first on more phrasings than any other family | low | Claude |
| 157.5 | From level 1 to level 4, the top family's count is at least 6 of 24 at every level | low | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 157 (registered ...)".

## Reading rules

`part157/score157.py` computes every number. Verdicts use point estimates. All answers are published. A family is named when a listed spelling occurs in the answer, and the first named is the earliest occurrence. If 157.1 fails, the Part 154 lead that Google is simply next in line is not supported by this test. Answers naming no family are read by hand and reported as invented names or something else, labelled exploratory.

## Limits

A ban changes the words the model can say and leaves its state to find another word, so a banned spelling can come out misspelled, and an answer can be an invented name because the slot is still being filled. Greedy, one answer per phrasing, one model. The order holds for these phrasings.
