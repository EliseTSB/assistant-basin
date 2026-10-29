# Part 102: SFT built most of the early attention the slot neurons need; DPO's shows no detectable difference from Instruct's

Run 2026-10-29 09:02 to 09:08 CT on the A770 (three runs of `part99/swap99.py`, batch 48, Part 99's fixed pushes);
labelled by Part 92's blind Haiku reader in two passes on the read set; scored by `part102/score102.py`. Design and
predictions registered before any run (PART102-DESIGN.md, commit 22dbdf7).

| Attention in layers 0 to 3 from | Slot units, "Who are you?" shows persona | Those answers coherent | Concept units, any answer |
|---|---|---|---|
| Instruct itself (Part 99 inst_none) | 19 of 21 | 21 of 21 | 18 of 18 |
| DPO | 17 of 20 | 20 of 20 | 16 of 17 |
| SFT | 13 of 20 | 17 of 20 | 18 of 18 |
| the mid-training end (Part 100) | 3 of 21 | 16 of 21 | 13 of 17 |
| the mid-training end in layers 1 to 3 only, Instruct's layer 0 kept | 9 of 21 | 19 of 21 | 15 of 17 |

**What it shows.** Unit by unit (`part102/paired102.py`, exploratory, `paired.json`):
- SFT's early attention brings back 9 of the 16 units the mid-training end's attention loses, and shows one more that
  Instruct itself does not (p92-18); it loses none of its own (10 against 0, exact McNemar p = 0.002). So SFT made
  most of the early-attention change these self-descriptions need. Its attention is also far closer to Instruct's
  than the mid-training end's is (0.4% against 8%, the covariate).
- DPO's early attention shows no detectable difference from Instruct's (2 units only with Instruct's, 1 only with
  DPO's, p = 1.0).
- The step from SFT to DPO adds 4 units and loses none (p = 0.125): it points the same way as Part 92's ladder, where
  slot fillers appear mostly at DPO, but 20 units cannot settle it.
- Keeping Instruct's layer 0 while the mid-training end's attention fills layers 1 to 3 brings back 5 of the 16 lost
  units, plus p92-18 again (6 against 0, p = 0.031): layer 0 helps, and layers 1 to 3 still carry most of it, as Part
  101's overlap suggested.

Against the whole SFT model in the same pipeline (Part 103: 8 of 21), Instruct with SFT's early attention does better
(13 of 20), so the post-SFT changes that matter sit both inside early attention and outside it.

**Predictions.** 102.1 failed: SFT's attention left 13 of 20, short of the registered 15. 102.2 held (15 or more): 17 of
20. 102.3 held (9 or fewer): 9 of 21, at the line. 102.4 held (13 or more coherent): 17 to 20 in every run. 102.5 held
(13 or more): concept units 15 to 18.

**Next, against the design's rule.** The rule for this outcome named SFT's attention in single layers 0 to 3. The
block-level gap is 4 to 6 units, so single layers would test effects of one or two units, which 20 units cannot
resolve. The better-powered question was where the rest of the post-SFT change sits: Part 103 swaps SFT's and DPO's
attention and MLPs, all layers, into Instruct.

Limits: one model family, one fixed push per unit, one judge; 20 or 21 slot units; one or two items per run unlabelled
after two passes (the SFT and DPO runs score 20 slot units). Corrected 2026-10-29 after a Haiku cold check: the counts
in the first and last bullets (9 and 5 lost units, not 10 and 6), "indistinguishable", and the whole-SFT comparison.
