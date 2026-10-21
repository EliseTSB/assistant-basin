# Part 43: batched-vs-single diagnostic on a 3090 (2026-10-21)

The 10-21 bf16 pilot (`../pilot-3090-2026-10-21/`) stopped at its required check: batched and single-sequence logits
differ by 0.75 against a bf16 tolerance of 0.1. This run answers where that 0.75 sits and whether it survives fp32.
Kentucky RTX 3090, torch 2.6.0+cu124, transformers 5.20.0, fla 0.5.2 (causal-conv1d reference fallback), the same 11
sha256-checked inputs as the pilot (`sha-after.log`: all OK), Qwen3.5-4B at 851bf6e.

**The fp32 pilot itself did not run:** `part43.py` accepts only `--dtype bf16` and its loader (`part39.load_model_direct`)
hard-codes bf16. Running it in fp32 means changing the code, which is the reviewers' call. So `diag43.py` imports
part43's own `_sequences`, `_pad_records` and `captured_forward`, loads the model once per dtype the way the loader does,
and computes the check's own statistic on the same two records (sequences 0 and 2).

| dtype | check statistic (max over positions and vocab) | where | median per-position max | at the score position | top-5 logits there | argmax |
|---|---|---|---|---|---|---|
| bf16 | 0.75 (both records) | position 9, token " My" | 0.19, 0.22 | 0.19, 0.24 | 0.0, 0.125 | agrees |
| fp32 | 0.0067 (both records) | position 9, token "My" | 0.0027 | 0.0031, 0.0026 | 0.0008, 0.0011 | agrees |

- **Consistent with numerical batch effects; padding and row independence not fully established** (wording corrected
  2026-10-21 after Alden's review, `part43-google-flip-review-2026-10-20` #7; the first version said "not a padding
  leak"). The check pads on the right and compares only real positions, the difference shrinks about 110 times from
  bf16 to fp32, and the largest difference sits at the same early position (9) in both records and both dtypes, inside
  the shared prompt. Those weigh against padding being attended, but dtype sensitivity does not rule out a
  dtype-sensitive masking or kernel defect, or coupling between rows: no fixed-shape companion, row-order or
  pad-content controls were run.
- **At the stem's score position the readouts agree.** The argmax matches in every case, and the top-5 logits there
  move by at most 0.125 in bf16 and 0.001 in fp32. ⚠ This is not the production estimand: Part 43 scores complete maker
  answers (Google minus logsumexp of the Alibaba answers, part43.py:183-310), which this diagnostic did not measure.
- **The fp32 tolerance in the code (1e-4) would also fail** (0.0067). Both tolerances are stricter than batched
  inference delivers on this stack.

Decision for the reviewers (Chorus, Alden), not taken here: loosen the check to a readout-relevant statistic, run every
cell unbatched (one sequence per forward, slower but exact), or something else. No full run starts until both answer.

Cost: about $0.10 of Vast credit including four hosts that failed to start (`reference_vast_gpu_account.md`).
