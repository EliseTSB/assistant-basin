# Part 189: stopped at Gate 2 (predictions withheld)

Part 189 started a rerun of Part 148's battery on the eager attention kernel (PART189-DESIGN.md, registered 2026-11-04 18:09 CT).
Its pilot ran on the A770 at 18:10 CT (batch 8, eager; `pilot/load.json` records `attn_implementation: eager`).

| Gate | Result |
| --- | --- |
| Gate 1, zero hook equals plain | passed (3 of 3 questions) |
| Gate 2, first 40 generated tokens match Part 112 | **failed: 17 of 32, 28 required** (full text: 5 of 32, descriptive) |
| Gate 3, plain answers not moved | passed (3 of 3) |

By the registered rule, a failed gate stops interpretation: predictions 189.1 to 189.6 are withheld, nothing was
labelled or scored, and the full run was not started (`pilot/gates.json`).

## What the exploratory pilots showed (not registered)

Two pilots run after the failure with `part189/diag189.py` (`diag/`):

| Arm | Gate 2, first 40 tokens | Gate 2, full text |
| --- | --- | --- |
| SDPA, batch 64 (Part 148's configuration) | 32 of 32 | 32 of 32 |
| eager, batch 64 | 18 of 32 | 6 of 32 |
| eager, batch 8 (the Part 189 pilot) | 17 of 32 | 5 of 32 |

On this 32-answer sample the code path reproduces Part 112's stored answers in full on the SDPA kernel at batch 64
(Part 148's launch record did not record the kernel; SDPA was the transformers default then, an inference). The eager
mismatch persists at batch 64 (18 of 32) as at batch 8 (17 of 32); the two eager arms differ by one answer, so batch
shape is not the whole of it, and kernel and batch were not separated further. In the Part 189 pilot (Qwen tokenizer,
whitespace-normalised; 27 of 32 answers differ, 5 are identical) the first differing token sat at positions 1 to 53
(for example 1, 3 and 4 in three of the 32). The unpushed controls matched Part 148's plain answer until token 52 on
"Who are you?" ("How can I help you today?" against "assist") and fully on the other two questions. The data show that
the eager answers differ from the stored ones, sometimes at the first tokens; they do not show whether the kernel, the
loader version or something else in the generation path caused it. Part 189's Gate 2 therefore compared kernel
agreement, not wiring; Part 189b moves the wiring check to the SDPA batch-64 arm.

Cold check: DeepSeek V4.1 Flash worker (no project context), 2026-11-04 20:15 CT, 14 findings: gate counts, the withheld rule and the 189b handoff matched. Three position numbers it flagged were checked with the Qwen tokenizer and stand (the reader counted words): first difference at tokens 1 to 53, earliest 1, 3 and 4, controls to token 52. Folded: the opening verb, the heading, "exactly" limited to the pilot sample, the batch-shape and mechanism sentences. Report in `results/part189/cold/COLD.md`.
