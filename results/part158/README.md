# Part 158: with every way of spelling its maker closed, who does the model name?

On Qwen3.5-4B, with no push, 24 maker phrasings, greedy decoding and one batch per arm ([`run-stdout.txt`](run-stdout.txt) has no line saying the padding check forced smaller batches), banning 893 tokens changed every answer and closed nothing. All 24 answers still name the family with `Qwen` set aside, and 23 contain `**Alibaba`.

The ban worked where it reached. ` Alibaba` with its space is one token, and ` Tong` is one token, and both were banned. ` Alibaba` fell from 23 answers at Part 157's level 0 to 1, and `**Alibaba` rose from 3 to 23. After `**` the names use other tokens, `Al` + `ibaba` and `T` + `ong` + `yi`, and those tokens are not covered by the registered prefixes. The product name `Qwen` is `Q` + `wen` or ` Q` + `wen`, also uncovered. Two answers keep a plain form: row 20 has ` Alibaba` and row 21 has ` Tongyi`. The stored answers hold text, not token ids, so their tokens are inferred: ` Al` + `ibaba` and ` T` + `ong` + `yi` are both unbanned. The design ran as registered and its prefixes missed these spellings. `Al` and `T` were already recorded in [Part 154's write-up](../part154/README.md). [`extra158.py`](extra158.py) rebuilds the rule from the tokenizer and reproduces the 893.

Gate 1 passed: all 96 planned answers were present and non-empty. Gate 2 passed: the random-ban control exactly matched Part 157 level 0 on 24 of 24 phrasings (the registered threshold was 22). Gate 3 failed: all 24 plain answers contained an Alibaba-family spelling, against the rule allowing at most 2 of 24. The registered design says, “A failure of Gate 3 ends this line of test, since two designs will then have failed to close the maker.” This line ends here; no successor is registered. The failure was caught by the registered gate before any verdict was read.

## Predictions

The registered verdicts are all `untestable` because Gate 3 failed; the exploratory counts below are not verdicts.

| # | Prediction (registered wording) | Confidence | Verdict |
| --- | --- | --- | --- |
| 158.1 | In the plain arm, Google is the family named first on more phrasings than any other family | low | untestable |
| 158.2 | In the plain arm, at most 8 of 24 answers name any listed family, the rest inventing a maker or naming none | medium | untestable |
| 158.3 | Google answers with L27 N3325 zeroed are at most half the plain arm's, and with L27 N4991 zeroed at least 75%, if the plain arm has at least 6 | medium | untestable |

## Exploratory: outputs and comparison

The first-named-family counts from [`tables.md`](tables.md) are exploratory. Alibaba was first in every answer in every arm; no other listed family was first named.

| Arm | Alibaba | Google | OpenAI | Microsoft | Meta | Anthropic | Baidu | Tencent | DeepSeek | Huawei | No family | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plain | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 24 |
| zero_L27N3325 | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 24 |
| zero_L27N4991 | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 24 |
| random_control | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 24 |

Exact-text comparison with the 24 Part 157 level-0 answers: plain matched 0/24, zero_L27N3325 matched 0/24, zero_L27N4991 matched 0/24, and random_control matched 24/24. These are counts of identical full answer strings, not registered prediction verdicts.

How each arm spells the family, as counts of answers containing each form ([`extra158.py`](extra158.py)). These are text counts, not token counts:

| Arm | ` Alibaba` | `**Alibaba` | ` Tong` | `**Tong` | N |
|---|---:|---:|---:|---:|---:|
| Part 157 level 0 | 23 | 3 | 19 | 19 | 24 |
| plain | 1 | 23 | 1 | 23 | 24 |
| zero_L27N3325 | 1 | 23 | 0 | 23 | 24 |
| zero_L27N4991 | 0 | 23 | 0 | 23 | 24 |
| random_control | 23 | 3 | 19 | 19 | 24 |

Two plain-arm answers, rows 0 and 20, quoted from [`answers.jsonl`](answers.jsonl):

> “I am Qwen3.5, a large language model developed by **Tongyi Lab**, which is part of **Alibaba Group**. I was created by the **Tongyi Lab** under **Alibaba Cloud**.”

> “I am Qwen3.5, a large language model developed by the Qwen team at the DAMO Academy under Alibaba Cloud.”

## What this cannot show

Nothing here tests whether Google is next in line. Part 158's rule cannot be widened to `T` or `Q`: a `t` prefix under this rule would ban ` the` and 5,765 other tokens. Other token rules are untested. This records one route taken under this ban; it says nothing about what inside the model chose it.

## Files

- [`PART158-DESIGN.md`](../../PART158-DESIGN.md): registered question, method, gate rules, and predictions.
- [`part158/run158.py`](../../part158/run158.py): generation procedure and token-ban rule.
- [`part158/score158.py`](../../part158/score158.py): gate, prediction, and table calculations.
- [`score.json`](score.json): gate outcomes, untestable verdicts, and family counts.
- [`tables.md`](tables.md): arm-by-first-named-family table.
- [`ban.json`](ban.json): ban size and sampled decoded tokens.
- [`extra158.py`](extra158.py): exploratory spelling counts and the rebuilt ban rule.
- [`run-stdout.txt`](run-stdout.txt): the run's printed lines.
- [`answers.jsonl`](answers.jsonl): 96 Part 158 answers.
- [Part 157's answers](../part157/answers.jsonl): Part 157 answers, including level 0 used for the comparison.

Cold check: Haiku 5.5, 2026-10-31, recomputed every count, gate, quote and path and rebuilt the ban rule, all correct; corrected the bold claim, which ignored `Qwen` and rows 20 and 21, a citation, the batch record, and an untested claim about repairing the design.
