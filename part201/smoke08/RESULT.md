# Smoke result, 2026-11-09

One run of the README's command on the host CPU (WSL, Qwen3.5-0.8B, 24 new tokens), 19:06:47 to 19:14:37 CST, at
commit 5c4178a. Scored with the scorer of the commit that adds this file; that commit changed one line of the smoke
report's wording (Gate G1 no longer says a smoke's snapshot check held) and no number. Every file the run wrote is in
`run-2026-11-09/`; the numbers below are in its `report.md`, `score.json` and `load.json`.

## What the smoke was for

| Question | Answer | Where |
|---|---|---|
| Is the activation at the push hook float32 in the float32 process, on a real checkpoint? | Yes. All 32 pushed rows of F8 and F1 record `torch.float32`; all 76 pushed bf16 rows record `torch.bfloat16`. | `rows.jsonl`, field `hook_dtype`; G4 in `score.json` |
| What did the upcast do? | 133 tensors converted to float32, 36 restored from the checkpoint, 187 kept in bf16. | `load.json`, launch 2, `upcast` |
| Do the gates the smoke can reach pass? | G2, G3, G4 and G5 passed. G1's bf16 and float32 halves passed. G1's kernel half reads "absent": a smoke has no pass K, so 201.7 is "not run" and the decision is rule 4. | `report.md`, Gates |
| Does the recorder handle ties? | 45 positions emitted a token tied with, and not the same as, the first-listed top. G2 accepted all of them. | `report.md`, G2 |
| How fast is pass A with the recorder on? | 1.16 answers a second. This is a 0.8B on a CPU and says nothing about the card. | `load.json`, launch 0, `pilot_answers_per_second` |
| How much memory? | 4.1 GB at the peak, for three model loads in one process. | systemd scope `MemoryPeak` at 19:14 |

## What was seen before registration

The smoke decodes 16 pushed rows (four units, four questions) with a made-up median, on another model and another
device than the registered run. Its pair counts were read before Part 201 was registered:

| Pair | What changes | Rows that differ |
|---|---|---|
| A / A-again, A / A2 | nothing (replays) | 0 of 4, 0 of 8 |
| A / B | slots and neighbours | 2 of 16 |
| A / M | batch members and padding | 3 of 16 |
| A / C | 4 rows a call against 1 | 4 of 16 |
| F8 / F1 (float32) | 4 rows a call against 1 | 0 of 16 |
| A / F8 | bf16 against float32 | 6 of 16 |

Every first difference between bf16 passes sat at a bf16 margin of one step (0.125) or less; most were exact ties.

No prediction and no bar in `PART201-DESIGN.md` was changed after reading these. At these rates 201.3 and 201.4 would
have failed (3 of 16 is under 20%; 2 of 16 is over 10%); both were left as written. With 16 rows the intervals are
wide (A / C: 10% to 49%).

One thing the smoke does show, outside the registered questions: bf16 answers changed with batch shape on a CPU,
with no Arc card or Arc driver involved, and float32 answers did not.

## What the smoke does not show

- Anything about the Arc A770, the 4B model, 80 tokens, or Windows paths.
- Pass K (the kernel reference step) and the scorer's reading of it.
- Separate launches: a smoke runs its three processes as three model loads inside one Python process.
