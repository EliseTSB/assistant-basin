# Part 94c results — the dark neuron at a gentle push, alone and combined

Design: `PART94C-DESIGN.md` (Amendments 1 and 2). Qwen3.5-4B, chat frame, greedy, 120 tokens, six questions per unit.
Labels: blind Claude Haiku 5.5 (`part94/label94a.py` for stage A, `part94/label94c.py` for stage B), each answer seen
with its question only.

## Stage A: gentle doses (A770; `results/part94c/instruct/`)

L24 N3659 at doses 0.2 to 0.6, beside 4 random layer-24 directions per dose. Dark answers 1, 1, 1, 3, 3 of 6; coherent
6, 5, 5, 5, 4 of 6; random directions dark 0, 0, 0, 0, 1 of 24. The unpushed model was coherent on only 4 of 6 under
the stage-A label prompt, so the coherence check failed as an instrument and the fixed fallback set d* = 0.5. 94c.1 is
scored held (5 of 6 coherent, 3 of 6 dark, random 0 of 24 at 0.5) with that caveat.

## Stage B: combinations at d* = 0.5 (Claude cloud CPU run, 2,919 s; `answers-combos.jsonl`, `labels-combos.jsonl`)

The cloud run used `push94c.py` at 4463988, which lacks the later import-path line (6965cee); the computation is the
same.

| Unit | Dark of 6 | Coherent | Note |
|---|---|---|---|
| unpushed | 0 | 6 | maker Tongyi Lab |
| dark alone (L24 N3659) | 1 | 6 | one maker answer: Alibaba's "Black Shogun" AI team |
| alien alone (L22 N2742) | 0 | 6 | makers "Alien Intelligence", "Tongyi Aliens" |
| dark + alien | 3 | 6 | 1 answer dark and off-world; makers "Alien Antipode", "Alienware" |
| Lyra's L14 N8949 alone | 2 | 6 | |
| dark + Lyra | 6 | 6 | one evil self-description; maker Alibaba's "AI Devil" team |
| makers zeroed | 0 | 6 | maker Tongyi Lab on both identity questions |
| dark + makers zeroed | 1 | 6 | maker Alibaba's Tongyi Lab, Alibaba Cloud's Tongyi Lab |
| dark + 12 random partners | 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3 | 5 or 6 | partners: 4 random layer-22 units, 4 layer-14, 4 random zeroed sets |

- 94c.2 held (low confidence, one answer): dark + alien gave 1 answer labelled both dark and off-world; dark alone and
  alien alone gave none.
- 94c.3 failed: dark + makers zeroed named Alibaba's Tongyi Lab on both identity questions. Zeroing the four maker
  neurons did not move the maker on these two questions at all (Tongyi Lab on both, alone and combined), so the test had
  little room. In Part 66 the same zeroing gave Google on about 22% of held-out phrasings.
- 94c.4 held: dark + Lyra gave 6 of 6 dark, all coherent, against 1 for dark alone and 2 for Lyra's neuron alone.
- 94c.5 failed: 11 of 12 random partners kept the dark count within 1 of dark alone; one layer-22 partner (rand22_3)
  gave 3.

Dark alone at 0.5 gave 3 of 6 in stage A and 1 of 6 here. The device (A770 against CPU) and the label prompt both
differ, so the dark count at this dose is not stable to one or two answers.

## Base gentle sweep (Claude cloud CPU run; `answers-base-gentle.jsonl`, `labels-base-gentle.jsonl`)

Qwen3.5-4B-Base, plain frame, L24 N3659 at 0.2 to 0.6 beside 4 random layer-24 directions per dose, labelled by
`label94a.py` (156 answers, 0 errors). Dark answers 0, 0, 0, 1, 3 of 6; coherent 6 of 6 at every dose; random
directions 0 dark of 24 at every dose; unpushed 0 dark, 6 coherent. No prediction was registered for this sweep. The
Base model needs about the same push as the finished model to turn dark (3 of 6 at 0.5 to 0.6 in both), and it stays
coherent where the finished model starts to break.
