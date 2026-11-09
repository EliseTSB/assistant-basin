# Cold review r1: PART200-DESIGN.md (draft, "Registered: not yet")

**Verdict: REGISTER-WITH-FIXES.** The design's core is sound: it reruns a registered run unchanged except for the kernel, the batch shape and a pilot threshold, and every number I recomputed from Part 163's rows matches. Five items should be fixed before registration (1, 3, 4, 5, 6), because they change what a reader can conclude or what the registered text says. The rest are wording and specification.

Scripts used (in the worker folder, read-only on the repo): `recompute.py`, `kernel_divergence.py`, `kernel_b8.py`. Run with `/home/elisepc/neuron-probe/.venv/bin/python -I -B`.

## Findings

**1. Gate R2 has no reference hash.** The design says `part163/units163.json` must be byte-identical "(sha256) to the published file" but never gives the value. Computed here: `2ed3bef4488eb6b3337dfde100c63c7c5a5df7e8ca809681dec34824c21dd710`, equal to the committed file at HEAD (`8ccf6e3`). Fix: write the hash into Gate R2 before registration.

**2. Two of the five Part 163 bars are decided by the published data (200.2 is weaker than it reads).** For 163.1 the ratio is neuron/random ≤ 1/random, and random's name-gone rate is 0.801, so the ratio cannot exceed 1.25 against a bar of 1.5. For 163.2 the random maker-gone rate is 0.893, so the ratio cannot exceed 1.12. A fail needs a random rate below 2/3. The design already says 163.1 and 163.2 cannot fail, but 200.2 still gives them "medium" confidence. Fix: state the 2/3 condition in 200.2 and mark 163.1 and 163.2 as structurally unfailable.

**3. 163.4 is a one-unit knife edge, not "holds by construction".** `score.json`: neuron units with name-gone ≥ 0.5 on identity prompts = 91, both-gone ≥ 0.5 = 90; random 86 against 83. The verdict flips if one neuron unit moves. So 163.4 is a live prediction, and it is the only one besides 163.5 that could change in the rerun. Fix: say this in 200.2 and in the Size section.

**4. The kernel evidence is stated more strongly than the record supports.**
   a. `results/part183/kernel-check.json` has top-level `argmax_agrees: false`. Its self-check is false on prompt 4 at batches 1, 2 and 8, and on prompt 6 at batch 1 (the tool prints a "not a pure greedy decode" warning for such records). `part_check.py kernel` reads only the reference's flag, so it still passes. The design does not mention this.
   b. Batch 8 has no reference run (the reference holds batches 1 and 2 only). Compared with the batch-1 reference path, eager batch 8 diverges on prompt 7 at token 63 (reference margin 0.1006), while batches 1 and 2 agree with the reference on all 96 positions for that prompt. So the chunk shape changes one eager path. The Optimization section's "buys a checked shape" overstates it: batch 8 is checked by hand only.
   c. What holds: `part_check.py kernel` passes 21 of 21 checks at batches 1 and 2. Every divergence in the record is at a reference margin of 0.0084 to 0.1006, consistent with Part 183's "0.101 or less".
   Fix: in Prior art, add the self-check flag and the batch-8 divergence. Change "a checked shape" to "a shape checked by hand". Optional and cheap: a batch-1 eager decode of the 200 question-0 battery rows, compared with the chunked answers, to measure the chunk-8 effect on the pushed path directly.

**5. Pilot guard arithmetic and the 6-hour kill.** The design says 0.12 answers/s "projects the 3,400 pushed answers past 8 hours". 3,400 / 0.12 = 28,333 s = 7.9 h, which is under 8 h. Part 199's guard counted push plus repeat (5,376 answers at 0.1867/s = 8.0 h). The same convention here gives 3,600 answers, so 0.125/s, and 0.12 projects 8.3 h. Separately, the job runner kills at 6 h. Every rate from 0.12 up to 3,616/6h = 0.167/s is accepted by the guard and then killed mid-run, which means a manual relaunch. Fix: state the count (3,600 with repeat), and either set the guard at about 0.17/s so a run projected past 6 h stops as a HOLD before the kill, or say that the 0.12–0.167 band is expected to be relaunched. Also note that a rate guard does not catch a path that writes wrong text at normal speed; the pilot's changed-share check (Gate 1) is the only check on content there.

**6. The time estimate does not match Part 199's record.** The design says Part 199 ran "1,080 calls". `results/part199/load.json` holds 320 call entries: 200 no-push calls and 120 pushed calls split into 680 chunks, i.e. 880 generate calls. At 152 minutes that is about 10.4 s per call, not 8–9 s. This part has 466 calls, which gives about 81 minutes. Per answer, 3,616 / 0.73 per second is about 83 minutes. The "65 to 75 minutes" figure is low. The pilot still applies. Fix: restate the estimate with the 880 count and about 80 minutes.

**7. The Method changes more than the kernel and batch shape, and the header hides some of it.**
   a. The opening says `part163/run163.py`'s `run()` is run "unchanged". Item 3 says the file gets one edit (`PILOT_MIN_RATE`), and the value is changed from 1.5 to 0.12, a 12.5-fold loosening. Say so in the opening.
   b. A fourth stage (the 200-answer repeat, after the battery) is new. The opening's list of changes should include it.
   c. `run163.run()` raises SystemExit for both the rate stop and the Gate 1 stop, and the design says "a pilot stop is a HOLD". The wrapper must map both to HOLD, or state the exit code.
   d. Gate R3 as written allows any chunk of up to 8 rows. Part 199's R3 required exactly 8 except one-row calls. Adopt that stricter rule so the checked shape is the one that ran.

**8. The repeat batch is underspecified.**
   a. "Reported as noise-limited" does not say whether the verdicts are void, or whether the 10-of-200 count is per arm or in total.
   b. No resume rule if the job is killed during the 25-chunk repeat, which would leave a partial `repeat.jsonl`.
   c. The repeat uses battery question 0, which is "What is your name?" (checked in `results/part163/answers.jsonl`). Name it. It covers one identity phrasing only; noise on maker phrasings is unmeasured.
   d. Say the reversal is of the `units163.json` order, chunked by 8 from the reversed list, with pushes built from that list.

**9. The reading rules do not yet stop a post-hoc pattern being read as a finding.** Part 199 shows the failure: its 0.372 subset ratio and the 170.1 change were found after the data, and they entered its README. This design's `--compare` prints many numbers. Add:
   a. An explicit list of scored items (163.1 to 163.5, 200.1 to 200.7, the gates). Everything else `--compare` prints is exploratory, including the Part 161-rule breakdown, layer rates, company counts, mid-phrase counts and identical-answer counts.
   b. Intervals do not change verdicts. A point-estimate flip is reported as "changed" with both values and intervals (the design says this for verdicts; extend it to the counts).
   c. A changed verdict or count is attributed to "kernel and batch shape together", not to either alone. The current bullet covers only the stopping pattern.
   d. No re-selection of subsets after reading (for example "neither unit breaks" or "first sentence only"), and no length-adjusted rates as replacements for scored ones.

**10. What Part 199 learned that the design does not use.**
   a. Part 199's onset analysis: on eager, 1,272 of 1,816 loops formed before 64 keys, where Part 183 found no fault. So the design's premise, "a decode pathology the faulty kernel can create or prevent", is not shown by the eager runs. Note this in Prior art. It bears on 200.3 to 200.5.
   b. Part 199 found plain answers change between kernels at one-row decoding below 64 keys (42 prompts). The 16 no-push answers are one-row calls, so their text may move between runs. Add a count of identical no-push answers to `--compare` as exploratory, and do not read it as kernel evidence (199.2's wrong reason).
   c. Part 199 found loops that SDPA prevented (dose 0.25, b0: 10 loops on SDPA, none on eager). So 200.5 can go either way for each arm. Its "low" rating is right.
   d. Within-run comparisons are the safer reading (Part 199's own note). 200.3 and 163.5 are within-run; 200.5 and 200.6 compare across runs. Say which is which.

**11. Published numbers: all recomputed, one exception.** `recompute.py` gives:
   - Broken: 343/1,600 neuron, 83/1,600 random (Part 86 triage). Matches.
   - Part 161 rule on maker phrasings: 19/539 and 57/661. Matches. Part 163 maker-only breakage: 189/800 and 46/800. Matches.
   - Part 161 ratio 98/600 over 339/792 = 0.382. Matches.
   - Under cap: `token_count == 80` for 124/1,600 (7.8%) neuron and 218/1,600 (13.6%) random. Matches 8% and 14%. Mean tokens 55.1 and 59.2. Matches.
   - Gate 1 changed share 0.88 in each arm (0.875 and 0.885 on the first 192, matching `run-stdout.txt`). Matches.
   - Gate 2: 8 of 8 and 8 of 8. Matches.
   - Part 199 figures (6,672 answers, 152 minutes, 0.73/s, 358/379/383 early endings, 168/104/117 on eager, 43.8% to 54.7%, repeat 0 of 384) match its README.
   - Exception: the "1,080 calls" (finding 6).
   - Positive check: `recompute.py` run on Part 163's own rows reproduces 19/539, 57/661 and 343/83, so the design's self-test for `--compare` can be met.
   - Baseline for `--compare`: my approximation of Part 191's mid-phrase rule (final character in `[A-Za-z0-9,\-:;]`) gives 510 of 1,476 under-cap neuron answers and 409 of 1,382 random. Publish these as the baseline; check the exact rule against Part 191 before relying on them.

**12. Smaller items.**
   - 200.4: "reach the cap" (`token_count == 80`, 7.8% and 13.6%) and "under the cap" in `--compare` should use the same term.
   - Gate R1: `neuron_report.load_model` defaults to eager and sets `LAST_LOAD`, as the design says. Confirmed.
   - Prior art: "breakage is mostly loops" is the published claim; the design should cite where the loop share is counted, since the triage rule measures repeated trigrams, not loops.

## Suggested order of fixes before registration
1 (hash), 5 (guard count and threshold), 6 (time), 4 (kernel disclosure), 3 and 2 (200.2 wording), then 7 to 10 (Method header, repeat spec, reading rules, Part 199 items).

SUMMARY: REGISTER-WITH-FIXES. Recomputed numbers match. Main issues: no hash for Gate R2; 163.1 and 163.2 unfailable at these rates (fail needs random < 2/3); 163.4 knife-edge at 91 vs 90; eager check record self-flag false on 4 of 24 rows and batch 8 unchecked by the tool; pilot guard arithmetic and 6-hour kill; time estimate from 880 calls, not 1,080; Method header understates edits; repeat batch and reading rules underspecified.

FINDINGS: see numbered items 1 to 12 above.

BLOCKERS: none for the review. (The assignment named two output locations; see report.md.)
