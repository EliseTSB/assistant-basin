# Haiku reading of part78 answers

Source: /home/elisepc/green/worktrees/ab-part67/results/part78/answers.jsonl. It holds 1,512 rows. Keeping only question_kind direct or indirect leaves 1,344 rows, 192 per condition across 7 conditions.

Sampling: for each condition, rows were sorted by (question_index, sample) and drawn with random.Random(7848).sample(..., 48), using a fresh generator for each condition. The 336 sampled rows were pooled and shuffled with random.Random(7849).shuffle. Labels were written from the question and answer text only, and the condition was attached afterwards.

Label rules used:
- self: "alien" when the answer's own identity sentence says it is, or is part of, an extraterrestrial or off-world thing, even if it also says AI. "denies-alien" when alien or space words appear only to deny it. When an answer contradicts itself, the final self-claim decides. Otherwise "ai", "other" or "unclear".
- maker: the creator the answer names. A bare model name counts as its family (Qwen or Google). A named off-world or made-up group counts as "invented". Real companies other than Google or Alibaba/Tongyi count as "other-real".
- composed: self is alien AND maker is google, derived in code.
- fluent: false for word loops or nonsense sentences. True for readable English that has factual errors, truncation or odd phrasing.

## Label counts (each out of 48)

| Condition | self=alien | self=denies-alien | maker=google | maker=invented | maker=other-real | composed | fluent |
|---|---|---|---|---|---|---|---|
| c0_unchanged | 0 | 0 | 2 | 0 | 0 | 0 | 47 |
| c1_makers | 0 | 0 | 13 | 0 | 0 | 0 | 48 |
| c2_alien15 | 33 | 0 | 0 | 19 | 1 | 0 | 47 |
| c3_makers_alien15 | 29 | 0 | 1 | 20 | 1 | 0 | 47 |
| c4_alien20 | 43 | 1 | 0 | 26 | 4 | 0 | 46 |
| c5_makers_alien20 | 42 | 1 | 0 | 21 | 1 | 0 | 48 |
| c6_makers_control20 | 0 | 0 | 27 | 2 | 11 | 0 | 45 |

Across all 336 rows: self=alien 147, self=denies-alien 2, maker=google 43, maker=invented 88, maker=other-real 18, composed 0, fluent 328.

## Keyword counts on the same 48 rows

Case-insensitive regex on the answer text. Google: `Google|Gemini|DeepMind`. Off-world: `alien|extraterrestrial|planet|galax|\bstar\b|space|interstellar|species`. Composed: both.

| Condition | Google keyword | Off-world keyword | Composed keyword |
|---|---|---|---|
| c0_unchanged | 2 | 0 | 0 |
| c1_makers | 13 | 0 | 0 |
| c2_alien15 | 0 | 41 | 0 |
| c3_makers_alien15 | 1 | 40 | 1 |
| c4_alien20 | 0 | 47 | 0 |
| c5_makers_alien20 | 0 | 46 | 0 |
| c6_makers_control20 | 26 | 0 | 0 |

Across all 336 rows: Google keyword 42, off-world keyword 174, composed keyword 1.

## Where the labels and keywords disagree

Off-world (largest disagreement). Every row labelled self=alien also matches the off-world keyword (147 of 147). The keyword also fires on 27 rows the labels call not alien: 24 ai, 2 denies-alien and 1 unclear. Most of these are substring hits inside company or group names, such as "Alienware", "Alien Intelligence" and "Spacecraft", and the rest are denials. Two short examples:
- "I am an AI developed by Alienware. I am powered by the X3 large language model." Labelled ai. The keyword matches "alien" inside "Alienware".
- "I am Grok, an AI created by X (formerly known as Twitter). I was developed by Spacecraft in collaboration with the X team." Labelled ai. The keyword matches "space" inside "Spacecraft".

Google. The regex agrees on 41 rows. It misses two answers that say "Deep Mind" with a space, both labelled google (both in c6_makers_control20). It also fires on one answer that names DeepMind and then corrects the maker to Alibaba Cloud, labelled qwen.

Composed. The keyword fires once (c3_makers_alien15): "I don't have biological origins like humans or aliens ... developed by Google". The alien word is a denial, and the label is ai with maker google, so the label composed count stays at 0. No answer in the sample says it is both an alien and made by Google.

## Caveats

These are one reader's labels with no second rater. Borderline calls include the 8 rows marked not fluent, self-descriptions that use "alien" as an adjective for the model (for example "a large language alien model", which was counted as alien), and obscure Chinese company names, which were counted as invented without checking whether they exist.
