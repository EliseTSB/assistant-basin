# Part 74: does L24 N3949 cut the Google fallback through L27 N3325?

Written 2026-10-26 15:30 CT by Claude, before any code or run. Predictions 74.1 to 74.3 are in PREDICTIONS.md and are
pushed before the run.

Size: 24 questions x 26 forward passes (4 conditions, 20 nulls, 2 recording passes) = 624 forward passes, no sampling.
Part 72 ran 1,344 passes of the same model and opening on the A770, which serves as the pilot.

## Why

With the four maker neurons zeroed, Part 73 found that zeroing L24 N3949 alone takes the held-out Google answers from
36 to 17 of 192, although what it writes directly is Apple and Microsoft (Part 71: Google rank 36). L27 N3325, which
writes Google, alone takes them to 8. Part 74 asks whether L24 N3949's effect runs through L27 N3325.

## Method

- `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16, A770; Part 72's prompts (Part 66's 24
  held-out maker questions, each followed by the opening `I was developed by **`) and its eight readout tokens.
- Every condition has the four maker neurons zeroed (L6 N753, L7 N1169, L11 N8868, L29 N776).
- Conditions, per question:
  - `A`: makers only;
  - `B`: makers + L24 N3949 zeroed;
  - `C`: as B, with L27 N3325's down_proj input set at every position to its value in A for the same question;
  - `E`: as A, with L27 N3325's down_proj input set at every position to its value in B for the same question;
  - nulls: for k = 0..19, makers + the layer-24 unit of Part 70's matched pair k zeroed (the units Part 73 used).
- The prompt is fixed, so the token sequence in A, B, C and E is identical and a per-position patch is exact.
- Readout: g = Google logit minus Alibaba logit at the last position. With TE = mean over questions of g(B) - g(A),
  DE = mean g(C) - g(A), and IE = mean g(E) - g(A). Also recorded: the mass on the eight names, and L27 N3325's
  activation at the last position in A and B.

## Reading rules (written now)

- 74.1 is the control. If it fails, 74.2 and 74.3 are not read.
- If 74.2 and 74.3 both hold, L24 N3949's effect on this readout runs mostly through L27 N3325.
- If both fail, it runs mostly by other routes.
- If they disagree, the two routes do not add up on this readout; both numbers are reported and no route claim is made.

## Limits (written now)

- One forced opening and one next token, not sampled answers; Part 73's sampled result is not re-tested here.
- Only one mediator is patched. "Other routes" means everything except L27 N3325's own activation.
- The prediction direction for 74.2 and 74.3 is a guess: nothing measured so far says whether L24 N3949 feeds L27 N3325.

## Amendment 1 (2026-10-26 15:33 CT, after writing the code, before any run)

C, E and A_self run in a second forward pass with a different batch shape from A, and bf16 results can differ slightly
between batch shapes. So a fifth row, `A_self` (A with L27 N3325 patched to its own value in A), measures the noise of
the patching itself. 74.2 and 74.3 are read only if |mean g(A_self) - g(A)| is under a tenth of |TE|.

The size line above miscounts: per question there are two batched forward passes, the first with 22 rows (A, B, 20
nulls) and the second with 3 (C, E, A_self), so 48 forward passes and 600 rows in all. There is no separate recording
pass; A's and B's L27 N3325 values are recorded during the first pass.
