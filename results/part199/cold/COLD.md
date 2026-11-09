# COLD.md: cold check of `results/part199/README.md` (Part 199)

**Verdict: SHIP-WITH-FIXES**

The registered numbers are right. Every prediction verdict matches its registered text and the scorer's output, and the cell, frozen-prompt, list, same-population and length tables recompute from the answer files. The problems are in the prose: one factual inversion on the front page, one explanation (why loop rates rose) that the answer files do not support, one rationale for 199.2 that the README's own table undermines, and several claims stated more firmly than the evidence allows. None changes a registered verdict.

Scripts: `results/part199/cold/scripts/recompute.py` to `recompute7.py` (run as `/home/elisepc/neuron-probe/.venv/bin/python -I -B`, reading repo files in place; nothing copied, nothing in the repo written, no model run, `report199.py` and `fig199.py` not run).

## Findings

1. **README "Why did no Stage 1 list match (199.2)": "Two things differ, and neither is the 64-key boundary the prediction leaned on."**
   - What is wrong: the prediction's premise was that the 40-token plain path never reaches 65 keys, so the kernels agree there. The README's own item 1 says 45 of the 195 differing plain answers part ways at 64 keys or fewer (earliest at 28), and Limits says "the two kernels part earlier than 64 keys". The 64-key premise is what failed.
   - Evidence (`recompute2.py`): 158 identical through 40 tokens. First differing token at keys (prompt length + index): min 28, median 66, 45 at or below 64. The README's numbers are right; the sentence contradicts them.
   - Fix: "The premise, that the kernels agree below 65 keys, does not hold for one-row decoding: 45 of 195 differing plain answers diverge at 64 keys or fewer, earliest at 28." Delete "neither is the 64-key boundary".

2. **README 199.2 rationale: "ordered equality was the wrong count".** Set-level agreement does not rescue the prediction.
   - Evidence (`recompute3.py`): all three lists (b5, b50, b200) are equal as sets for 3 of 200 prompts. b200 is equal as a set for 8 of 158 unchanged-path prompts (shared tokens min 195, so the sets differ by a few tokens). 199.2 (at least 190 of 200) fails under either reading.
   - Fix: state that 199.2 fails under order and under set equality; keep the order point as a description of what the count measured, not as the reason for the failure.

3. **README "so the push and no-push cells ban nearly the same tokens ... and different ones for about a fifth."** Not supported as written.
   - Evidence (`recompute3.py`): b5 sets differ for 39 of 200 prompts (19.5%). b50 sets differ by at least one token for 96 of 200 (48%), with a mean of 2.05 of 50 tokens differing (4.1%).
   - Fix: give the per-list counts above instead of "about a fifth".

4. **Front page, `README.md` line 2724 (Part 170 entry): "...and most answers on this kernel stopped early, many mid-phrase."** Factual inversion. In that sentence "this kernel" is the eager kernel, and on eager most dose-1 answers ran to the 80-token cap.
   - Evidence (`recompute.py`, matching `report199.md`): stopped early on eager 168 of 384 (b0), 104 (b50), 117 (b200). On SDPA 358, 379, 383.
   - Fix: "...and most answers on the SDPA kernel stopped early, many mid-phrase; on eager most ran to the cap."

5. **Title, first paragraph and front page: "170.1 no longer holds" / "Part 170's no-push claim 170.1 does not".** The flip is a point estimate crossing a bar: ratio 0.174 to 0.329 against 0.25 (`score.json`). Both intervals cross the bar: published 0.000 to 0.443, eager 0.111 to 0.630 (`report199.md`). The Predictions section says this; the title, the answer paragraph, the Part 170 note (`results/part170/README.md` line 3) and the front page do not.
   - Fix: "170.1's point estimate rose from 0.17 to 0.33, past its 0.25 bar; both intervals cross it." Use the same wording in all four places.

6. **"Why are the loop rates higher on eager?"** The mechanism as worded is not supported by the answer files.
   - (a) "the rise there is not a length difference." Mean lengths are equal (68.23 and 68.55 tokens), but the stop-type mix is not. Decomposing the eager b0 to b50 dose-1 rise (0.263): 0.142 is within-stop-type rate change; 0.121 is the change in the share reaching the cap (216 of 384 to 280 of 384). About 46% of the rise runs through the length mix (`recompute7.py`).
   - (b) "a shorter answer has less room for [loops]." Within each run the pattern runs the other way. On SDPA the loops sit in the short answers: 83 of 96 b0 loops and 165 of 168 b50 loops end before the cap. On eager they are at the cap: 107 of 109 (b0), 208 of 210 (b50), 223 of 223 (b200) (`recompute5.py`, `recompute7.py`).
   - (c) Caveat: token count is downstream of looping (a loop that repeats usually runs to the cap), so stratifying on length is not a clean causal test either.
   - Fix: describe the pattern, not a mechanism. "On eager, dose-1 loops run to the cap; on SDPA most loops end early at end-of-text. Length composition accounts for about half of the eager b0 to b50 rise; the mechanism is not identified." Delete "not a length difference" and "less room".

7. **Mid-phrase counts ("ended without end punctuation, that is, mid-phrase"; the Part 191 comparison; `part199/fig199.py` lines 68-76).** The heuristic counts answers that end in an emoji after punctuation (for example "...feel free to ask! 😊") as mid-phrase, and the figure's regex `[.!?)"\*:]\s*$` has the same problem.
   - Evidence, dose-1 early-stopped answers, b0 / b50 / b200:
     - Eager, README rule: 13 / 20 / 41. Emoji stripped first: 2 / 13 / 35 (`recompute4.py`). Part 191's stated rule (ends on letter, digit, comma, hyphen, colon or semicolon): 1 / 12 / 24 (`recompute6.py`).
     - SDPA, README rule: 122 / 271 / 291 by my reproduction (README says 292 for b200; the rule is definition-sensitive). Emoji stripped: 121 / 269 / 285. Part 191's rule: 119 / 271 / 268.
   - The direction holds (SDPA answers are far more often mid-phrase), but the eager counts are inflated and "found the same" for Part 191 hides that the rules differ.
   - Fix: state the rule in the README, use Part 191's rule (which Part 191 explicitly corrected for emoji endings: `results/part191/README.md` line 106), report the counts under it, and say the rules differ from Part 191's if another rule is kept.

8. **"On eager text there is no 64-key fault, so a loop that forms late is not by itself the kernel's."** Overclaim.
   - The support is Part 183: batches 1 and 2 by machine, batch 8 by hand. The 5- and 6-row no-push batches are unchecked (the README's own Limits says so). Finding 1 shows that the one-row eager and SDPA paths differ below 64 keys, and there is no float32 reference for those prompts.
   - Fix: "Part 183 found no 64-key fault at batches 1, 2 and (by hand) 8. The 5- and 6-row shapes are unchecked, and this run cannot say which kernel is right below 64 keys."

9. **The onset premise: "The loop-onset check ... found every published Stage 0 loop formed past key 64", and "the published frozen set most likely reflects the SDPA kernel".**
   - The onset claim checks out: `part199/onset/loop-onset-170.jsonl` gives the Stage 0 cells as m96 12 loops (0 before t0, 12 after) and q96 7 (0 before, 7 after).
   - But the script's premise (its docstring, lines 7-8) is that steps before t0 (keys 64 or fewer) "cannot have changed" under the kernel. Finding 1 contradicts that for one-row decoding. The 199.4 and 199.5 ranges and the frozen-prompt attribution both lean on it.
   - What does carry the frozen-prompt attribution: the 19 published frozen plain answers are all loop-and-broken (`recompute2.py`), and all 200 eager plain answers are clean. The two quoted examples match the files exactly (`m96:3`, `q96:0`).
   - Fix: state the premise where the onset claim is made, and rest the frozen-prompt attribution on the eager plain answers being clean.

10. **"Did 170.1 change because the population changed? Not mainly."** The conclusion is right, but the README omits the number that answers the question (exploratory; flag it as such).
    - Evidence (`recompute3.py`): on the same 92 published prompts, eager gives b0 0/92, b50 9/92, rise 0.098, ratio 0.372 (published 0.174). On the registered 104 it is 0.329. The population change moves the eager ratio down, so the flip comes from the rerun's loops on the same prompts.
    - Fix: add the 0.372 line beside the exploratory table.

11. **"The push result holds and is larger."** "Larger" is a point comparison: dose-1 b50 rise 0.188 published, 0.263 eager (199.5 failed on its range). No interval for the difference is computed.
    - Fix: give an interval for the difference (whole-slot bootstrap, as `score170` does), or write "the point estimate is larger".

12. **Gate 8 generalisation: "so batch order does not move these answers" (Method) and "the order of rows in a batch moves nothing at either shape" (Limits).** Each claim rests on one shuffled batch (the dose-1 b0 P8 cell, 384 answers) at each of two shapes.
    - Fix: "In the one batch Gate 8 shuffles, 0 of 384 answers changed, at both shapes."

13. **170.6 on eager (minor).** "Holds (0.062)" now rests on a small positive rel50 rise (+0.2 points; rise ratio 0.0099 against a bar of about 0.25). Part 170 described the bar as one that "any negative rise passes".
    - Fix: one clause in the 199.6 row noting that the eager rise is small and positive.

## Checked and matching

- Predictions table verdicts (199.1 to 199.9) against `PREDICTIONS.md`'s Part 199 section and `PART199-DESIGN.md`'s reading text. "Four of nine held" and "one of ten verdicts changed" are correct.
- Dose-1 cell rates and mean lengths, published and eager, all five bans. Dose 0.25 b0 (10 and 0 loops). No-push tables (`recompute.py`).
- Stage 0 frozen counts (12 and 7 published; 0 eager), overlap 0, eager plain loops and broken 0 (`recompute2.py`).
- Stage 1 list table (`recompute3.py`).
- Same-population identity counts (92 and 89; 104 eager) and the 0.087 registered rise.
- Loop-onset totals (1,272 of 1,816; 155 of 210) against `compare.json`.
- Gate R2 (144 directions, worst error 0.006). Gate R3: 112 pushed calls in chunks of 8; the 8 calls that are not all 8s are the one-row zero-write checks, which are exempt (`load.json`, `recompute6.py`). One eager load, status done, 6,672 answers, 07:24 to 09:55 CT.
- Part 170 note and front-page Part 199 entry: numbers match, except finding 4.

## Not verified

- Part 191's 230 of 838 and 2 of 588: stated in `results/part191/README.md` line 71, not recomputed.
- Part 178's near-tie claim: `results/part178/README.md` line 100 is consistent with it.
- Gate 6 values, the 170.8 quartile estimate, and 170.W1 and 170.W2 (taken from the scorer output only).

## Unsure

- Finding 6's decomposition stratifies on token count, which is downstream of looping, so the 46% figure is descriptive rather than causal.
- Finding 7: every "mid-phrase" count here is a heuristic, including Part 191's.

SUMMARY: Registered verdicts and tables recompute correctly. SHIP-WITH-FIXES: one front-page inversion (finding 4), an explanation the answer files contradict (findings 1, 6), a 199.2 rationale the README's own table undermines (finding 2), and overstated 170.1 and mechanism wording (findings 5, 8, 9).
FINDINGS: 13 numbered; 1, 4, 5, 6 and 7 are the ones to fix before publishing; 9 to 13 are wording.
BLOCKERS: none. Nothing in the repository was edited; no model was run; no folder or file over 1 MB was copied.
