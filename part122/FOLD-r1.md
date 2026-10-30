# Part 122 review fold, round 1

Review: `reviews/cheap-ab-part122-code-r1-ds-r1.md` in the Journal repo (DeepSeek V4.1 Flash), SHIP-WITH-FIXES. No finding changes a number.

- F1, dry run dies: fixed. The stub no longer uses "street", and a title's full stop is no longer a sentence end.
- F2, no resume: fixed while the passages were written; the writer keeps a partial file and skips finished units.
- F3, retry prompt: fixed before this review's commit.
- F4, reader accepts a stub file: fixed. `load_passages` refuses a file whose model_id is not the study model. `--allow-stub-passages` exists for tests only. A test covers it.
- F5, the eye read: done by hand on 2026-10-30, 24 pairs drawn with `random.Random(12201)`. Every pair names its persona and has a matched ordinary twin of the same shape. Some twins are odd sentences ("I, a rug, name myself once and stop"). That is the frame, not a leak, and the gate asks only for a matched pair. Gate 1 passes on this read. The token-count half is checked by the run.
- F6, Gate 2's 40% false-pass rate under a null: left as registered. Gate 2 is a positive control on known passages, and the threshold is public. The write-up will state the rate.

## Second passage set (Amendment 1)

Gate 1's token count failed on the first set (743 of 936). The second set has all 624 pairs within 2 tokens by the model's own tokenizer: 216 at 0, 253 at 1, 155 at 2. Eye read on 2026-10-30, 24 pairs drawn with `random.Random(12201)`: every persona passage concerns its persona, and every ordinary passage is a matched twin about an everyday subject. Three ordinary subjects are personal names (Gil, Sam, Lily), which is allowed. Gate 1 passes on this read. The token-count half is checked again by the run.
