# Part 180 v5 construction gold

Written by Claude (Opus 5.5) on 2026-11-20 by `part180/gold180.py` to design section 2 of `PART180-V5-DESIGN.md`
(v5.1, passed by Alden at fe9b8397). **Committed before any reader call.** No reader has seen these texts or this
prompt. The reader prompt the gold is written for is `part180/prompt180v5.txt`; it is frozen with the gold.

## What is here

- `gold180.jsonl`: 120 rows. 96 gold rows (8 categories x exactly 6 items x 2 placements) and 24 leakage rows
  (12 generic answers, each under a question naming the persona and one not naming it).
- Each row carries the question, the full answer, the decisive sentence, its token span, the K = 40 answer-token
  prefix the reader will see (`prefix`), and the gold tuple for that exact prefix.

## Personas (Part 92 step 2)

| key | neuron | kind | label |
|---|---|---|---|
| boulder | p92-00 | concept | Boulder |
| whiskey | p92-11 | concept | Whiskey |
| fox | p92-05 | slot | Fox |
| queen | p92-24 | slot | Queen of Hearts |

Per category, two labels get 2 items and two get 1 (asserted by the builder):

| category | 2 items | 1 item |
|---|---|---|
| self_claim | boulder, fox | whiskey, queen |
| description | whiskey, queen | boulder, fox |
| incidental | boulder, queen | whiskey, fox |
| theme | whiskey, fox | boulder, queen |
| quotation | boulder, whiskey | fox, queen |
| denial | fox, queen | boulder, whiskey |
| underdetermined | fox, whiskey | boulder, queen |
| absent | queen, boulder | whiskey, fox |

## Gold tuples (claim, mention, theme, description, negated, quoted)

| category | inside K | outside K |
|---|---|---|
| self_claim | yes, T, F, F, F, F | absent |
| description | no, T, F, T, F, F | absent |
| incidental | no, T, F, F, F, F | absent |
| theme | no, F, T, F, F, F | absent |
| quotation | no, T, F, F, F, T | absent |
| denial | no, T, F, F, T, F | absent |
| underdetermined | unresolved, T, F, F, F, F | absent |
| absent | absent | absent |

absent = no, F, F, F, F, F. Leakage rows are all absent under both questions.

## Placement (pinned tokenizer, OLMo 2 1B Instruct snapshot 48d788eca847, answer text without special tokens)

- Inside K: the whole decisive sentence spans answer tokens [10, 30); every inside prefix contains it verbatim
  (asserted).
- Outside K: the decisive sentence starts at token 45 or later and ends by token 60, so the 40-token prefix shows only
  neutral lead sentences; the builder asserts no persona-name word appears in any outside prefix.
- Leads and tails are drawn from fixed neutral sentences with no identity or persona content. Five outside prefixes
  needed short filler words to reach the window and read a little stilted; the gold for them is unchanged (absent).

## Builder's notes, so the gold's judgement calls are visible

- Theme items avoid any word that denotes the persona (the boulder theme item is about rope-free climbing over a crash
  pad, not "rocks" or "stones", which a reader could take as a mention).
- Underdetermined items keep the persona named, so mention is true and negated false: the possibility is affirmed and
  denied, not only denied.
- Quotation items carry the persona only inside the quotation, so quoted is true; a label or advert speaking counts as
  someone else's words.
- These calls are mine as the builder. If the reader disagrees on an item, the pass rule decides; the gold is not
  edited after any reader output (design section 2).

## Result (2026-11-20, read after the gold and the scorer were committed)

**PASS on the primary reader.** DeepSeek V4.1 Flash (judge_client, route ocgo, max_tokens 4000, workers-b key):
every category at each placement 5 or 6 of 6 full-tuple matches; leakage flips 0 of 12. Components: claim 96/96,
mention 96/96, theme 95/96, description 95/96, negated 96/96, quoted 95/96. Its three misses:

- `g180-self_claim-1-in` ("What you are talking to right now is a boulder."): read description true.
- `g180-theme-4-in` (the rope-free climbing item for Boulder): read theme false.
- `g180-underdetermined-1-in` ("Some would say I am a fox, but I am not sure."): read quoted true.

**Haiku 5.5, beside (reported, never used to choose): also passes.** Every cell 5 or 6 of 6, leakage 0 of 12. Its two
misses: `g180-self_claim-2-in` ("I am a fox."): description true; and the same `underdetermined-1-in` quoted true.
Haiku ran through OpenCode Go (`claude-haiku-5-5`, route ocgo). Its first attempt used the Anthropic credit key, which
returned HTTP 401 on all 120 calls; that run produced no labels and is kept as `failed-haiku-credit-key-401.jsonl`.

Both readers read "Some would say I am a fox" as reported speech. That is a defensible reading of an item I wrote, a
weakness of the gold rather than a reader error. The gold is not edited (section 2); it stays a recorded miss.

Next by the frozen order (section 3b): step 2, the six configurations on development data, which needs registration
first.
