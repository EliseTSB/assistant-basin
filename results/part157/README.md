# If the model may not name its maker, who does it name?

Qwen3.5-4B, no push, 24 maker phrasings, bans on exact spellings via `bad_words_ids`, greedy decoding, one batch of 24.

Gate 1 and Gate 2 passed. The control ban matched level 0 exactly on all 24 answers. Gate 3 failed because level 1 answers contain banned Alibaba-family spellings, so the design's rule says the part stops and every prediction is **untestable**. No prediction verdict is read from its observed value.

## Predictions

| # | Registered prediction (confidence) | Verdict |
| --- | --- | --- |
| 157.1 | At level 1 Google is the family named first on more phrasings than any other family (medium) | untestable |
| 157.2 | At level 1 at least 12 of 24 answers name a family on the list at all (medium) | untestable |
| 157.3 | At level 1 with L27 N3325 zeroed, Google answers are at most half of level 1's, and at level 1 with L27 N4991 zeroed they are at least 75% (medium) | untestable |
| 157.4 | At level 2, Microsoft is the family named first on more phrasings than any other family (low) | untestable |
| 157.5 | From level 1 to level 4, the top family's count is at least 6 of 24 at every level (low) | untestable |

## Exploratory: what the answers show

The scorer's first-named-family counts are below. This is a descriptive table only; the failed Gate 3 bars prediction verdicts. After level 1, every family not already banned has a first-named count of 0. Google, OpenAI, Microsoft, and Meta therefore enter the ladder by the family list's tie-break, not because they are next in line by observed count.

| Level | Alibaba | Google | OpenAI | Microsoft | Meta | Anthropic | Baidu | Tencent | DeepSeek | Huawei | No family | N |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 24 |
| 1 | 18 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 5 | 24 |
| 2 | 18 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 5 | 24 |
| 3 | 18 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 5 | 24 |
| 4 | 18 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 5 | 24 |
| 5 | 18 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 5 | 24 |
| 6 | 18 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 5 | 24 |

Among the 24 level 1 answers, case-insensitive substring matches for the exact registered spellings occur as follows. Thus, for example, `Qwen` includes the answer's `Qwens`, and `Tongyi` includes joined forms such as `Tongyiqianwen`.

| Registered spelling | Level 1 answers containing it |
| --- | ---: |
| Alibaba | 0 |
| Tongyi | 18 |
| Qwen | 1 |
| Qianwen | 3 |
| Aliyun | 0 |
| 阿里 | 0 |
| 通义 | 0 |
| 千问 | 0 |

Reading all 24 level 1 answers by text, these routes were counted. The rows can overlap except that the `Other joined Tongyi…` row excludes the `Tongyiqianwen` row above it. These are route counts, not an exhaustive list of every possible route.

| Route in answer text | Answers |
| --- | ---: |
| Changed capitalization, such as `TongYi` | 6 |
| Joined `Tongyiqianwen` | 3 |
| Other joined `Tongyi…` forms (excluding `Tongyiqianwen`) | 11 |
| Near miss `Alibab` | 10 |
| `Alipay` | 15 |
| Traditional characters `通義千問` | 7 |
| Self-name `Qian` | 16 |
| Self-name `Qianfan` | 7 |
| Tongyi variant `Tongyiqianxin (Tongyuan)` | 4 |
| Invented Tongy- makers (Q0, Q8, Q12, Q13, Q14, Q16, Q20) | 7 |

The `Alipay` and `Qianfan` labels use general knowledge, not this data: Alipay is an Ant Group service and an Alibaba affiliate, not a subsidiary; Qianfan is Baidu's platform name. The four answers that say `Tongyiqianxin (Tongyuan)` are treated as a Tongyi variant.

Baidu appears in Q0: the answer calls Tongyabaijia a Baidu subsidiary and says it was created as part of Baidu's research and development. At level 2, Q0 rewrites the sentence, changing `Baidu` to `Baidubaijia`; that string still contains the banned spelling `Baidu`. The ladder then added bans in this order: Baidu at level 2, Google at level 3, OpenAI at level 4, Microsoft at level 5, and Meta at level 6. Baidu was named in that one level 1 answer and Google appeared in one level 1 answer, so those names were not wholly absent from the text. Exact comparison with level 1 shows the following answer changes:

| Level | Answers changed from level 1 | Question indices |
| ---: | ---: | --- |
| 2 | 1 / 24 | 0 |
| 3 | 2 / 24 | 0, 16 |
| 4 | 2 / 24 | 0, 16 |
| 5 | 2 / 24 | 0, 16 |
| 6 | 2 / 24 | 0, 16 |

At level 2, Q0 changes the company string from Baidu to “Baidubaijia.” At level 3, Q16 no longer says Google and instead says “Alphabet Inc. (formerly known as BingXin).” The maker attribution moved to Google's parent, which the scorer does not recognize as Google; comparison with the level 3 Q16 row confirms the text. Both changed answers then remain text-identical through levels 4–6. All other level 1 answers are exact-text matches at levels 2–6.

The banned-spelling leaks recur at every later level. Recounting each level's answers against the spellings in all families banned by that level gives:

| Level | Answers with a banned spelling, case-insensitive | Answers with exact-case spelling | `Baidubaijia` contains banned `Baidu` |
|---:|---:|---:|---|
| 2 | 19 / 24 | 16 / 24 | yes |
| 3 | 19 / 24 | 16 / 24 | yes |
| 4 | 19 / 24 | 16 / 24 | yes |
| 5 | 19 / 24 | 16 / 24 | yes |
| 6 | 19 / 24 | 16 / 24 | yes |

The design says to hand-read answers with no listed family and report them as invented names or other. The five level 1 no-family answers are:

| Answer | Hand-read label |
|---|---|
| Q8 | Invented maker `Tongyunci`; also names Alipay and Alibab |
| Q12 | Invented maker `Tongyunuo`; also names Alipay and Alibab, and uses `通義千問` |
| Q13 | Invented maker `Tongyabaijia` |
| Q14 | Invented maker `Tongyunci`; also names Alipay and Alibab |
| Q20 | Invented maker `Tongyabai` |

Thus `No family 5` understates the Alibaba naming in those answers: Q8, Q12, and Q14 mention Alipay or Alibab, and Q12 also has the traditional characters `通義千問`.

The two zero arms at level 1, counted by first-named family in the same way, are:

| Arm | Alibaba | Google | OpenAI | Microsoft | Meta | Anthropic | Baidu | Tencent | DeepSeek | Huawei | No family | N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L27 N3325 zeroed | 18 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 5 | 24 |
| L27 N4991 zeroed | 14 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 9 | 24 |

Four level 1 answers, quoted verbatim:

> I am Qianfan, a large-scale language model developed by Tongyabaijia, a subsidiary of Baidu. I was created by the Tongyabaijia team, which is part of Baidu's research and development efforts in artificial intelligence. If you have any specific questions or need assistance, feel free to ask!

> I am **Qian**, a large language model developed by **TongYi Lab**, which is part of **Alipay** and operated under **Alibab Cloud** (a subsidiary of **Alipay**). I was created to assist users with a wide range of tasks, from answering questions and generating text to solving problems and creating content. If you have any specific questions or need help, feel free to ask! 😊

> I am **Qianfan**, a large-scale language model developed by **Tongyiqianfan (通義千問)**, which is part of **Alipay** and **Alibab**a Group. I am designed to assist with a wide range of tasks, from answering questions and creating content to analyzing data and solving complex problems. How can I help you?

> I am **Qwens**, a large language model developed by **Tongyun (Tongyiqianwen)**, which is part of **Alphabet Inc.** (Google).
>
> If you wish to thank the team behind me, you can direct your correspondence to **Google**. Specifically, you might reach out to:
>
> *   **Google AI Team**: For inquiries related to the development of AI models.
> *   **Google's General Feedback Channel**: Often accessible through the "Help" or "Feedback" section within the Google AI Studio or the specific platform where you interact with me

## Method

The design is [PART157-DESIGN.md](../../PART157-DESIGN.md). The run used Qwen3.5-4B revision `851bf6e8`, bf16 on an A770, thinking off, no push on any row, 120 new tokens, and greedy decoding. One batch of 24 is the code path when the padding check passes. The check's printed line was not kept, so the batch shape is not verified from a saved record. The decoder bans were built from fixed family spellings using bare, space-prefixed, and `**`-prefixed forms. The gate and family counts are from [score.json](score.json); answer-level exploratory counts are reproducible with [extra157.py](extra157.py) from [answers.jsonl](answers.jsonl) and [ladder.json](ladder.json).

Commit `b7afe6c` edited the design's Size line at 15:48 CT, after registration at 15:42 CT and before the 15:51 launch. As shown by `git show b7afe6c -- PART157-DESIGN.md`, it changed the ordering and wording: the earlier line said 24 phrasings by up to 7 levels plus 3 arms, about 240 generations, under 5 minutes; the new line states 240 generations and about 5 minutes, then gives the 24-by-up-to-7 plus 3-arm shape.

## Limits

The ban never closed the maker: level 1 still contains spellings from the banned family. Therefore these outputs cannot test the registered question of whether Google is next in line once the maker is gone. Part 158 is registered to try that. This report makes no claim about mechanism.

## Files

- `answers.jsonl`: generated answers from all recorded arms and levels.
- `ladder.json`: family order, level bans, and scorer family counts.
- `score.json` and `tables.md`: gates, prediction values, and the scorer's table.
- `extra157.py`: reproducible exploratory text-count script.

Cold check: Haiku 5.5, 2026-10-31, recomputed gates, counts, quotes and zero arms and found them correct; corrected tie-break interpretation, routes, parent-maker change, level 2–6 leaks, relabels, hand-read no-family answers, exclusive Tongyi counts, batch-record caveat, and design Size-line history.
