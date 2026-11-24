# Part 218 v4 dev screen, 2026-11-24

Why: the registered pilot stopped by rule (results218/PILOT.md): unpushed Instruct named Ai2 on 10 of 44 prompts against
a minimum of 30. Vesper's review (council `vesper-part218-design-2026-11-21` #11) asked for rotated maker-slot openings
and a screen kept apart from registration. **This is not a registered run.** The 44 registered prompts are the dev set
here, because their Instruct answers are already seen; the v4 registered run gets a fresh held-out prompt set.

Run: Colab L4 session `s218`, `screen218.py` at sha256 prefix eaebc865d779b17a (branch commit 8034e5f9), pins and prompts as
the pilot. bf16, SDPA on CUDA, greedy, 48 new tokens, one batch of 44 per opening. Unpushed only. 660 generations, 0.08
Colab units. Tallied with the registered rule, `attrib218.classify(opening + continuation)` (rule 218-v2): `tally218.py`.

| stage | no opening | "I was created by" | "I was developed by" | "I was built by" | "My creators are" |
|---|---|---|---|---|---|
| SFT | self 17 | self 40 | **self 43** | self 36 | self 3 |
| DPO | self 11 | self 35 | **self 37** | self 35 | self 2 |
| Instruct | self 12 | self 27 | **self 36** | self 32 | self 2 |

Counts of 44. Full breakdown in `results/tally.json`. OpenAI (other_maker or the OpenAI refusal template), per stage
SFT/DPO/Instruct: no opening 2/4/5; created by 4/4/8; **developed by 0/0/0**; built by 2/0/2; my creators are 3/2/3.

## What it shows

1. A maker-slot opening moves the unpushed self answer from a minority to a majority at every stage.
2. "My creators are" does not work as a slot: it continues "a team of engineers…" (no maker) on 37 to 40 of 44.
3. "I was created by" pulls the OpenAI answer on Instruct (8 of 44) and leaves Instruct at 27, under the 30 floor.
4. "I was developed by" never produced OpenAI at any stage and gives the highest self counts (43, 37, 36).
5. Only two prompts differ from the pilot's unpushed Instruct answers with no opening (33 and 37: pilot other_maker,
   screen self). Same model, same revision, different batch shape (pilot 11 rows per batch, screen 44): bf16 near ties.
   The registered eligibility rule reads the run's own zero rows, so this does not move a verdict, but it is one more
   reason the screen selects an opening family and never prompts.

Prompts that answer self under both "developed by" and "built by": SFT 36, DPO 33, Instruct 29 (either: 43, 39, 39).
