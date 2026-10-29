# Part 89 results — how far post-training moves every OLMo 2 1B neuron

Run 2026-10-28 22:46 CT on the CPU (part89/cos89.py, weights only; analysis part89/analyze89.py, seed 89089).
Checkpoints present: pretraining 21B, 210B, 1T, 2T, 3T, 4.0T; mid-training end (51B, SFT's parent); SFT; DPO;
Instruct. Missing: mid-training 13B and 26B (still downloading), so settle checkpoints (89.2, 89.3, 89.6) are
provisional. Post-training measures (89.4, 89.5, 89.7) use only the mid-training end and Instruct and are complete.
Full numbers: `analysis.json`; per-neuron cosines: `cos89.npz`.

## Post-training barely moves any neuron

- Across all 131,072 MLP neurons, the median down_proj cosine between the mid-training end and Instruct is 0.995.
  The most-moved neuron (L10 N1978) keeps cosine 0.986. No neuron falls below 0.95; 70 fall below 0.99.
- All of the move happens at SFT. Between SFT and DPO, and between DPO and the final Instruct model, every one of the
  40 most-moved neurons keeps cosine 1.0000 to four decimals.
- Pretraining is where the weights form: the median cosine with the final model is 0.049 at 21B tokens, 0.156 at
  210B, 0.461 at 1T, 0.734 at 2T, 0.886 at 3T, 0.967 at 4.0T and 0.995 at the end of mid-training.

## The identity neurons move less than average, not more

| Set | Mean post-training move | Random sets (same layers) | p (moves more) | Mean settle checkpoint | Random | p (settles earlier) |
|---|---|---|---|---|---|---|
| Part 16's 64 "As an AI" neurons | 0.0041 | 0.0050 | 1.0 | 4.45 | 4.85 | 0.001 |
| Part 2's top-50 identity neurons | 0.0043 | 0.0049 | 1.0 | 4.84 | 4.92 | 0.04 |
| Part 14's top-50 assistant-register neurons | 0.0056 | 0.0053 | 0.015 | 4.68 | 4.67 | 0.64 |
| Alien twin L10 N6978 | 0.0038 | 0.0053 | 0.88 | 4 (3T) | 4.65 | 0.35 |

The "As an AI" set settles earlier in pretraining than random neurons of the same layers and moves less in
post-training; the register set is the only named set that moves more than random, by a small margin.

## Limits

- Cosine of the weights measures arrival at the final weights, not function. A cosine of 0.986 in 2048 dimensions
  is still a change, and a small weight change can matter if it lands where the neuron's input or output is used.
- The stages differ in data and objective as well as order, so "moves at SFT" says when, not why.
- The named sets were chosen on Instruct.
- Added 22:47 CT: the first cosine pass found only Instruct, because the base checkpoints are sharded and SFT and DPO are
  pytorch_model.bin files; the script reported them missing rather than skipping them, and was fixed before any
  analysis was read.
