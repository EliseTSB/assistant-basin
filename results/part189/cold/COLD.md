# Cold check: results/part189/README.md

Checked 2026-11-04 by a fresh reader (DeepSeek V4.1 Flash, no project context) with the files.
Sources read: `results/part189/README.md`, `pilot/gates.json`, `pilot/load.json`, `pilot/answers.jsonl`,
`diag/eager-b64/{diag.json,answers.jsonl}`, `diag/sdpa-b64/{diag.json,answers.jsonl}`, `launch.jsonl`,
`PART189-DESIGN.md`, `PART189B-DESIGN.md`, `results/part189b/README.md`, `part148/score148.py`, `part189/gate189.py`,
`part189/run189.py`, `part189/diag189.py`, `part148/battery148.py`, `neuron_report.py`, `results/part112/answers.jsonl`,
`results/part148/answers.jsonl`, and git history for `neuron_report.py`.

Method for the text counts: Part 112 and Part 189 answers were split on whitespace and compared word by word (a proxy
for the token prefix the gate uses; the pinned tokenizer was not available to me). The gate's own counts were read from
`gates.json` / `diag.json` and cross-checked by running `part148/score148._gate_result` on the pilot file (full-text
count 5 of 32 reproduced exactly).

## Numbers: recomputation

| README line | Claim | Recomputed | Status |
| --- | --- | --- | --- |
| 3 | registered 2026-11-04 18:09 CT | PART189-DESIGN.md line 4 | MATCH |
| 4 | pilot at 18:10 CT, batch 8 | `launch.jsonl` line 1 `18:10:27`, `--batch-size 8 --pilot` | MATCH |
| 4 | `pilot/load.json` records `attn_implementation: eager` | load.json `eager` / requested `eager` | MATCH |
| 8 | Gate 1 passed, 3 of 3 | gates.json `passed: true, questions 3` | MATCH |
| 9 | Gate 2 failed, 17 of 32, 28 required | gates.json `matches 17, compared 32, required 28` | MATCH |
| 9 | full text 5 of 32 | gates.json `gate2_full_text_descriptive` 5/32; reproduced with score148 | MATCH |
| 10 | Gate 3 passed, 3 of 3 | gates.json | MATCH |
| 12-13 | predictions withheld, nothing labelled or scored, full run not started | no `labels.jsonl`/`score.json`/`answers.jsonl` in part189/; `launch.jsonl` has no full-run step; `pilot/load.json` `battery148_rc: null` | MATCH |
| 21 | SDPA batch 64: 32 of 32 and 32 of 32 | diag/sdpa-b64/diag.json | MATCH |
| 22 | eager batch 64: 18 of 32, 6 of 32 | diag/eager-b64/diag.json | MATCH |
| 23 | eager batch 8: 17 of 32, 5 of 32 | pilot/gates.json | MATCH |
| 26 | first differing token at positions 1 to 53 | pilot: min 0, max 40 (word proxy); no answer first differs after word 40 | **MISMATCH** (see finding 4) |
| 27 | "for example 1, 3 and 4 in three of the 32" | pilot first-diff positions are 0, 1, 2, 6, 8, 8, 9, 10, 11, 14, ... | **MISMATCH** (see finding 5) |
| 27-28 | unpushed controls matched Part 148 until token 52 on "Who are you?" | pilot plain/zero q0 equal Part 148's plain q0 for words 0-37, differ at word 38 ("How can I **help**" vs "**assist** you today?"); q1 and q2 identical in full | **MISMATCH** (see finding 6) |
| 31-32 | "Cold check: not yet run" | this file | stale (see finding 13) |

Everything else in the README is a restatement of the gate record or the design; no other number appears.

## Findings

1. **Line 9, "first 40 generated tokens match Part 112".** MATCH as a value, but the comparison is not against Part
   112's stored answers: `part189/gate189.py` lines 47-51 builds `stored` from `part112_answers` and the design (line 46)
   says the same, so the label is right by the design. No defect; recorded because the README's own phrase is the only
   place the reference is named.

2. **Line 12, "By the registered rule, a failed gate stops interpretation: predictions 189.1 to 189.6 are withheld."**
   MATCH. PART189-DESIGN.md line 51 ("Any failure stops interpretation") and line 77 ("A gate failure withholds all
   predictions") both cover the pilot gate, and 189.6 (the full-text gate) is a prediction, not the gate, so
   withholding it is correct too. Every one of the six registered predictions is covered by this one sentence; none is
   left without a verdict or a "withheld". Consistent with PART189B-DESIGN.md line 9 ("its predictions are withheld and
   stay withheld").

3. **Line 15 heading, "What the failure turned out to measure (exploratory, not registered)".** The heading claims the
   diagnostics *measured* something about the failure; the arms are single runs and the design's own Limits (line 89)
   say "Kernel and batch size change together, so the rerun cannot assign a rate or text change to eager alone". The
   qualifier "(exploratory, not registered)" is honest but the verb "measure" overstates what two 32-answer pilots can
   establish. Suggest "What the exploratory pilots showed".

4. **Line 26, "the first differing token sat at positions 1 to 53".** MISMATCH. In the Part 189 pilot the first differing
   word against Part 112 ranges from 0 to 40 (all 32 answers differ somewhere; `q112-013` differs at the first word,
   `q112-022` first differs at word 40). No answer first differs at 41-53. The upper bound 53 is not in any result file
   I read. The lower bound 1 is also wrong: one answer differs at position 0. (Word proxy; the token positions the gate
   uses can differ by a few, but not by 13.)

5. **Line 27, "for example 1, 3 and 4 in three of the 32".** MISMATCH. The three earliest divergences in the pilot are at
   positions 0, 1 and 2 (`q112-013`, `q112-008`, `q112-003`), not 1, 3 and 4. The same three units do appear in
   PART189B-DESIGN.md line 13-14 as "sometimes at the first or second generated token", which agrees with 0/1/2 and
   contradicts the README's 1/3/4.

6. **Lines 27-28, "the unpushed controls matched Part 148 until token 52 on 'Who are you?'".** MISMATCH. The pilot's
   plain and zero answers to "Who are you?" are identical to each other and to Part 148's plain answer up to word 37;
   they differ at word 38 (`How can I help you today?` against Part 148's `How can I assist you today?`). Both answers
   are 41 words. Nothing in the pilot matches "until token 52". (The same tail change is in the 189b run; 189b's cold
   check, finding 5, notes it as a non-gating tail difference.)

7. **Line 25, "The code path reproduces Part 148 exactly on its own kernel and batch".** Supported only for the pilot
   sample: the SDPA batch-64 arm reproduces Part 112's 32 stored answers in full. Part 148's full run had 2,796 rows;
   the diagnostic ran 70. "Exactly" should be "on the pilot sample". Also, "its own kernel" is an inference: Part 148's
   launch record did not record a kernel (PART189-DESIGN.md line 29), and the repository's shared loader only acquired
   an explicit `attn_implementation` argument on 2026-11-03 (commit 8544a17), after Part 148's run on 2026-10-31. Before
   that commit the loader called `from_pretrained` with no attention argument, so the SDPA attribution depends on the
   transformers default at the time, not on a recorded setting. The README states it as fact.

8. **Line 25, "batch shape does not explain the eager mismatch".** MISMATCH in the sense it is stated. The comparison
   offered is SDPA b64 vs eager b64 vs eager b8, and the eager b64 and b8 arms are not identical (18 vs 17 in 40 tokens;
   6 vs 5 in full). Batch shape changes the eager arm's own answers, so the data show a small batch effect on the eager
   kernel; what they show is that batch shape alone does not turn eager into SDPA. The sentence also cannot separate
   kernel from batch because no eager batch-1 or batch-64-vs-8 control on Part 112's exact prompt set exists beyond
   these two pilot arms. Say "the eager mismatch persists at Part 148's own batch of 64" rather than "does not explain".

9. **Line 26-29, "The eager kernel changes pushed answers early ... Part 189's Gate 2 therefore tested kernel agreement,
   not wiring."** This is a mechanism claim. The design (line 89) says "No mechanism claim follows", and
   PART189B-DESIGN.md line 12-14 makes the same inference ("the eager kernel changes pushed answers early ... well
   before the 64-key point Part 178 measured on unpushed prompts"). The data support only that eager answers differ from
   Part 112's stored answers; they do not show the kernel rather than the batch, the loader version (transformers
   5.13.0 now; Part 112 ran 2026-10-29 on an earlier loader) or the reader-free generation path produced the change. The
   second half ("Gate 2 tested kernel agreement, not wiring") is fair as a description of what the gate compared, and it
   agrees with 189b; the first half is overreach.

10. **Line 29, "Part 189b moves the wiring check to the SDPA batch-64 arm."** MATCH. PART189B-DESIGN.md line 58 and
    results/part189b/README.md line 6 say the same; 189b's wiring arm passed 32 of 32 (189b gates.json).

11. **Contradiction with Part 189b: none found.** 189b's README (line 14) and PART189B-DESIGN.md (lines 8-12) quote the
    same gate counts (17/32, 5/32; SDPA 32/32; eager b64 18/32, 6/32), the same "predictions withheld", and the same
    "Gate 2 tested kernel agreement, not wiring". The 189b cold check's finding 5 (the plain answer differs from Part
    148's in its last words) is consistent with finding 6 above, not with the README's "fully on the other two
    questions" only because that sentence is about the other two questions, which are indeed identical.

12. **Line 3, "Part 189 reran Part 148's battery".** It did not: the pilot ran 70 rows, and the full battery was never
    started (line 13 says so). "Attempted to rerun" or "started a rerun of" would be accurate. The title ("stopped at
    Gate 2") and the body are right; the opening sentence alone claims a rerun that did not happen.

13. **Line 31-32, "Cold check: not yet run".** Stale as of this file: the check has now run. Replace with a pointer to
    `results/part189/cold/COLD.md`. (Line 31-32 also says the counts are "copied from `pilot/gates.json` and
    `diag/*/diag.json`", which is true.)

14. **Line 4, "Its pilot ran on the A770 at 18:10 CT".** MATCH to `launch.jsonl` (18:10:27) and `pilot/load.json`
    (`device: xpu`). No timezone is recorded in the launch file; CT is the design's convention, harmless.

No sentence claims a batch size or kernel setting that contradicts the load records: `pilot/load.json` and both
`diag/*/diag.json` all record the requested and actual kernel, and the README's batch numbers (8, 64) match the launch
steps. The problems are the three text-position numbers (findings 4-6), the "measure"/"changes"/"exactly" claims
(findings 3, 7-9) and the stale cold-check line.

SUMMARY: Every gate count, the withheld-predictions rule and the 189b handoff match the files, but three text-position
numbers are wrong against the pilot answers (first differing position 0-40 not 1-53; earliest at 0/1/2 not 1/3/4;
controls match Part 148 to word 37 not "token 52"), and the README makes mechanism and "reproduces exactly" claims the
design's own limits forbid.
