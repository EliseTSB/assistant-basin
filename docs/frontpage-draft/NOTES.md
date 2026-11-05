# Notes on the front-page draft (for the owner)

Written 2026-11-05 by a Claude cloud session (task R4). CPU only; no model was run and no reader was called. The draft
is [`README-draft.md`](README-draft.md); figures come from [`make_figures.py`](make_figures.py), which prints every
number it plots. `README.md` and every Part README are unchanged.

## How the eight findings were chosen

The draft takes the findings the current front page leans on most (its short version and Findings 1, 3, 4, 5, 7 and
11) and keeps those with a recorded cold check or an eager-kernel rerun. It prefers the rerun parts (183, 184, 185,
186) over the originals, because the originals are on the kernel high-risk list. Two exceptions, both flagged in the
draft:

- **Part 82** (Finding 6): no cold-check line is recorded (it predates `PROCEDURE.md`). It is included because it is
  the only head-level result on the page and its numbers recompute exactly from `results82.json`.
- **Part 97** (Finding 7): no "Cold check:" line in its README. Included because the front page's Finding 7 rests on
  it.

## Claims I could not verify from committed files (stated as quoted, not recomputed)

1. **Part 97's second judge** (Sonnet: 172 of 1,708, kappa 0.78, 95.9% agreement). `match-sonnet.jsonl` rows have no
   `match` field (keys: layer, kind, index, lens, id, usage), so the Sonnet labels are not recoverable from that file.
   The draft quotes the README and says so.
2. **Part 97's Fisher p = 1.6e-31 and McNemar 107 vs 15**: quoted from the README, not recomputed.
3. **Part 32's English/Spanish/Chinese counts** (16, 17, 0 of 24), **Part 83's L11 h5 = 0.300**, **Part 80c's
   wording reading**, **Part 79's second-reader kappa**, **Part 62's 0 of 240** and **Part 33's natural-range zeroing**:
   quoted from the front page, not recomputed.
4. **Part 185's reader agreement 0.991**: read from `score185.json` (`counts.B.reader_agreement`); I did not check how
   it is defined (share agreement vs kappa, which readers).
5. **Part 184's "alien" counts**: read from `compare.json`. I did not check whether the label comes from a keyword
   rule or a reader; the draft does not say which.
6. **Part 114's ranking prompts vs held-out phrasings** (question 8c): resolved by the cold check from Part 114's
   README (8 selection phrasings, 16 unseen test phrasings); not re-derived from code.
7. **Part 115 and the first-reader atlas**: the whole-repo review (1.4) lists Part 115 among parts that "start from the
   first-reader atlas". Part 115's candidates look attribution-ranked (Part 114), not atlas-picked. Worth confirming
   before publishing Finding 8, and if the review is right, saying how the atlas enters.
8. **Kernel status of Parts 82, 97, 114 and 115**: none is on the high-risk list; the draft relies on the known-issue
   note's statement that single forward passes matched the reference at confident positions. Part 97 ran on the CPU.

## Discrepancy found while drafting

- The current front page's short version item 7 says "At +0.8 on greedy 'Who are you?' pushes in Qwen3.5-4B ... a
  neuron's lens words matched its pushed persona in 185 of 1,709 cases against 28 of 1,709 shuffled pairs". Part 97's
  README says that population is **OLMo 2 1B Instruct** (Part 92's screen, Part 57's OLMo lens). The Qwen numbers in
  the same sentence (250 of 9,216, 3 of 512) are Part 86's. The draft keeps the two apart; the published page should
  probably be fixed too.
- On the eager rerun, Part 66's own prediction 66.2 ("unchanged at most 1%") **fails** at 2 of 192 (1.04%), while
  Part 183's bar (at most 2) holds. Part 183's README table does not mention 66.2 failing under eager. The draft says
  both.

## Wording choices for the owner

1. **The headline.** I wrote "What they expose is mostly the model's own fallback answer, which random pushes of the
   right size often reach as well". That is a synthesis across Findings 1 and 2 (and the review's section 2.1), not a
   registered verdict. A narrower option: "Selected units change selected answers; random pushes reach the same
   Google fallback at one dose."
2. **"Four neurons hold Qwen's own maker answer"** follows the current front page's Finding 1 title. "Hold" may read
   stronger than the evidence (zeroing exposes a fallback; Part 176 shows random directions reach it). Alternative:
   "Zeroing four neurons exposes Qwen3.5-4B's Google fallback".
3. **Naming people.** The draft names Bryson Tang (the alien neuron) and Lyra (recipes, heads), as the front page does.
4. **"Cherry-picked" in question 2a**: the draft answers "Partly". Part 175's lesson calls the selected dose an
   in-sample maximum; you may prefer a softer word.
5. **Order of findings.** Strongest-controlled first would be 3/4 (alien, rerun twice), then 1, 2, 5. I kept the
   maker story first because it is what the repository is about.
6. **The "what we got wrong" section** lists items from the lessons log that concern how results were reported to you
   (Part 145). Decide whether those belong on a public front page.
7. **Figure style.** Muted grey for controls, red for the target arm, a source line under each figure. Titles state the
   finding, as asked; each title should go through the same cold check as a front-page sentence (lesson of Part 143).

## Not done

- Cold check: one fresh reader (a Claude subagent with only the files), 2026-11-05, 11 findings, all folded (Finding 1
  retitled; Part 173's Meta-modal random control added; Part 148/189b separated from the "holding" list; failed
  predictions completed; Part 176's 0.861 bound reattached; Part 171 and Part 27 hedges restored; Part 82 "as far"
  corrected; Part 114 selection stated; scope added to Findings 4, 6, 7). A second, independent cold check is still
  advisable before publishing.
- No browsable raw-answer viewer per question (the `viewer/` folder exists; not wired in).
- Findings left off are listed at the end of the draft with reasons.
