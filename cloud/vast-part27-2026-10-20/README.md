# Part 27 costs + base on Vast.ai (2026-10-20)

The `costs,base` steps of Part 27 kept dying to the A770's Windows RAM guard (rc=97 at 18:56 with batch 4 and at
22:02 with batch 1), so they ran on one rented RTX 3090 (Illinois, $0.337/h with a 60 GB disk). Not scored here.

- Size, multiplied out: costs = 22 conditions (baseline, set_zero, 20 matched control sets) x (200 documents of
  next-token loss + 26 greedy answers at 40 tokens) = 4,400 document passes and 572 generations; base = 1,000
  documents captured on Qwen3.5-4B-Base + 6 identity prompts + 36 sampled completions.
- Command (run.sh): `python -u part27.py --steps costs,base --corpus qwen35/pile10k.jsonl --out results/part27
  --device cuda --max-gpu-gb 22 --batch-size 2`, code and inputs copied from the A770 job folder (part27.py identical
  to this repo's). No patch to the code.
- Versions: transformers 5.20.0, torch 2.6.0+cu124, flash-linear-attention 0.5.2, driver 595.91.07. causal-conv1d's pip
  wheel failed to import on this image (ABI mismatch), so it ran without it (the slower conv fallback).
- Timing: the successful run took 1,040 s for both steps. Two failed attempts first: batch 8 ran out of memory in the
  full-vocabulary loss (costs-b8-oom.log), and the first batch-2 run computed all of costs and then crashed writing its
  header because PREDICTIONS.md (hashed into every output) was not copied (run-crash1.log).
- Cost: about $0.27 of rented time (22:38-23:27 CT).
- Outputs: costs.json, base.json, base-word-activations.npz (also copied to the A770 job folder's results/part27).
  Chat-vs-base word-activation Pearson r for the four neurons, as a sanity read: 0.953, 0.940, 0.905, 0.957.
- vast.py: the REST helper used (bare key read from ~/.vast-credits.env, never printed; lists instances via
  /api/v1, because /api/v0/instances/ now returns 410).
