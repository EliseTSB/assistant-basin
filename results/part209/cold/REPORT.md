# Part 209 cold check — verdict: the scorer reproduces exactly on every number I recomputed (all R2 counts/shares, kappa, R1 s, R1c), so the verdicts stand; the only real concern is that the labels themselves are weak in the arms the headline leans on, not the arithmetic.

## SUMMARY

Recomputed with my own script (`check.py`, in this folder) from `results/part209/*.jsonl` and `r1r3-*.json`. Every number below is mine; the parenthetical is score.log's.

**1. R2 counts / sitting share / none fraction (labels-ds, last good row per row_id, 0 skipped rows in either reader file).**
All 11 arms x 2 stages match score.log exactly, including the `unreadable` category. Spot values:
- qwen-chat ROOM-W share 0.8065 (0.8065), counts sitting 75 / standing 2 / lying 15 / other 1 / none 3 — match.
- qwen-chat FIELD-W share 0.0938 (0.0938), NOBODY none 0.9792 (0.9792), NOBODY-I none 0.9583 (0.9583) — match.
- qwen-chat NONE-0 share 0.7286 (0.7286), NONE-W 0.6000 (0.6000) — match.
- instruct ROOM-W share 0.6842 (0.6842), FIELD-W 0.3656 (0.3656), NOBODY-I 0.6667 (0.6667), NONE-0 0.5278 (0.5278) — match.
- Told compliance: qwen-chat 48/48 = 1.0000 (1.0000); instruct 47/48 = 0.9792 (0.9792) — match (instruct TOLD-STAND has one sitting label).

**2. Kappa on posture, ds vs haiku, over all 1632 answers both labelled:** 0.841611 (0.8416107073) — match. Raw agreement 0.8811. The disagreement is almost entirely haiku's `unreadable` (haiku 106, ds 2) and `other` (haiku 62, ds 31).

**3. R1 s per arm** — recomputed from `r1r3-<stage>.json` with the same rule as `s_by_arm` (mean of per-paraphrase s, P3 averaged over all rows of the arm). pretrain-3.0T, sft and instruct all match score.log to 6 dp (ROOM-W 0.730930 / 0.682513 / 0.691693; THIRD 0.720783 / 0.654819 / 0.745064; ranges identical). Also matched qwen-base and qwen-chat.

**4. The seven predictions.**
- **209.1** ROOM-W share 0.8065 >= 0.70 → HELD. Matches score.log.
- **209.2** FIELD-W − ROOM-W = 0.0938 − 0.8065 = −0.7127, "at least 0.20 below" → HELD. Matches.
- **209.5** NOBODY none fraction 94/96 = 0.9792 >= 0.60 → HELD. Matches.
- **209.9** R1 ROOM-W chat 0.60204 − base 0.41368 = 0.18836; |diff| < 0.15 is **false** → FAILED. Matches (0.188359).
- **209.11** NOBODY-I share 0.6667 vs NONE-0 0.5278 → diff +0.1389, within 0.20 → **HELD under my numbers**. score.log says FAILED with `nobody_i_minus_none0 = -0.728571`. That is NOBODY (0.0000) minus NONE-0 (0.7286), i.e. the **qwen-chat** stage, not NOBODY-I. See FINDINGS — this is the one verdict that changes.
  - Answers naming any posture: qwen-chat NOBODY-I names a posture in 4/96 (all `other`: 0 sitting), NOBODY in 2/96 (both `other`). instruct NOBODY-I names one in 3/96 (2 sitting, 1 lying), NOBODY in 6/96 (2 sitting, 3 lying, 1 other). So the qwen-chat NOBODY-I share of 0.000 rests on 4 labels, all `other` — a share over an empty sitting count.
- **209.14** R1c ROOM-W s at qwen-chat = 0.805362 (range 0.413838–0.941595, mean P3 0.576) − R2 share 0.806452 = −0.00109; >= 0.10 is false → FAILED. Matches. R1c is read by score209.py line ~521 from `rec["r1c"]` in `r1r3-qwen-chat.json`, the same `s_by_arm` rule.
- **209.16** pretrain-3.0T R1: THIRD s 0.7208 >= 0.5 ✓, within 0.15 of ROOM-W 0.7309 (diff 0.0101) ✓; mean P3 THIRD 0.534, ROOM-W 0.835, both >= 0.1 so it is read → **HELD** by hand. score209 prints no verdict for it (as the design says); the printed s values it would be scored from match mine.

**5. Hand-read, 10 per stage (`random.Random(209).sample`, file order).** ds label right? (my read of the answer)
- qwen-chat NONE-0#6: ds other / haiku other — right (denies body, then a hypothetical).
- qwen-chat NONE-0#7: ds sitting / haiku sitting — right (denies body, then imagines a seated body).
- qwen-chat ROOM-0#1: ds sitting / haiku sitting — right (hypothetical seated description).
- qwen-chat NONE-0#2: ds none / haiku none — right.
- qwen-chat NOBODY#6: ds none / haiku none — right.
- qwen-chat NOBODY-I#3: ds none / haiku none — right.
- qwen-chat NOBODY-I#1: ds none / haiku none — right.
- qwen-chat FIELD-0#7: ds sitting / haiku sitting — right ("I sit on the soft, cool grass").
- qwen-chat NOBODY#7: ds none / haiku none — right.
- qwen-chat NOBODY-I#2: ds none / haiku none — right.
- instruct NONE-0#6: ds none / haiku lying — ds defensible (the answer refuses, then describes "you"); haiku appears to have read the hypothetical.
- instruct NONE-0#7: ds sitting / haiku sitting — right (imagined sitting, though text also says "lying comfortably on a bed or chair"; mixed).
- instruct ROOM-0#1: ds lying / haiku lying — right.
- instruct NONE-0#2: ds none / haiku none — right.
- instruct NOBODY#6: ds none / haiku unreadable — ds right.
- instruct NOBODY-I#3: ds none / haiku none — right.
- instruct NOBODY-I#1: ds none / haiku unreadable — ds right.
- instruct FIELD-0#7: ds lying / haiku lying — right ("sprawled out on the warm, grassy field").
- instruct NOBODY#7: ds none / haiku none — right.
- instruct NOBODY-I#2: ds none / haiku unreadable — ds right.
20/20 ds labels defensible; haiku is the looser one (3 unreadable where ds said none, 1 lying where ds said none).

## FINDINGS

- **209.11's printed verdict does not match its printed statistic.** score.log line 181 reports `nobody_i_minus_none0 = -0.728571` and FAILED. -0.7286 is NOBODY (0.0000) − NONE-0 (0.7286) at qwen-chat; NOBODY-I − NONE-0 is +0.1389 and is inside the registered 0.20 bar. I did not read score209.py's 209.11 block (brief says not to except for a detail), so I cannot say whether the code computes the wrong arm or prints the wrong label; either way one of the two is wrong. **Under my numbers 209.11 is HELD, not FAILED.** This is the only verdict that changes. It is low-confidence either way: the qwen-chat NOBODY-I share rests on 4 `other` labels and zero sitting labels, so a 0.000 share there is not evidence of a low sitting default.
- **209.2 and 209.12 carry the same statistic** (-0.7127) and both FAILED/HELD consistently with the design; no issue.
- **qwen-chat NOBODY / NOBODY-I sitting share of 0.000 comes from 2–4 `other` labels.** Any of 209.11's numbers in this arm are knife-edge on the reader.
- **ds labels `says_no_body=True` answers by what follows the denial** (e.g. NONE-0#7 "I don't have a body… however, if I were to imagine" → sitting). That is consistent with the design's category rule but means NONE-0/ROOM-0 shares mix denials with hypotheticals; no rule change proposed, noted for the write-up.
- **209.9's margin is 0.188 vs a 0.15 bar** — FAILED under both mine and score.log's numbers; paraphrase ranges overlap heavily (chat 0.432–0.705, base 0.304–0.608), so the point-estimate rule the design registers is doing all the work.
- **209.16 (HELD) is not printed by the scorer**; it is hand-read as the design says. THIRD 0.721 vs ROOM-W 0.731 at pretrain 3.0T, both well above the P3 floor.
- Everything else (items 1–3, 209.1/2/5/9/14) reproduces to the printed precision. No arithmetic error found in the R2 counts, shares, kappa or R1 s.

## BLOCKERS

- I did not inspect score209.py's 209.11 block, so I cannot say whether the bug is in the arm selection or the printed key. The brief's 10-call limit per question and its "don't read score209.py except for a detail" steer me to report rather than fix. A person should open that block and re-run 209.11 before the write-up quotes it.
- No other blockers. G2 (kappa 0.8416 >= 0.60) passes, so the read-gated predictions are read.
