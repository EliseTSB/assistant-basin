# Part 194: Part 24 holds on the eager attention kernel (the alien neuron and the identity neurons still add on fixed answers, and every scorable verdict is unchanged)

Part 24 tested Qwen3.5-4B's alien neuron L22 N2742 against 50 identity units and 50 random units: a left-to-right
check, how clamping the neuron changes later units, a fixed-answer assay of log probabilities, and free answers read by a
blind reader (Part 24b). It ran on the A770 in bf16 with transformers' default fused SDPA kernel, which Part 178 found
leaving the CPU float32 path after about 64 keys. Here the same units, prompts, doses, seeds and scorers ran on the eager
kernel, in three GPU stages, each a separate launch that records its model load before any work. Registered in
`PART194-DESIGN.md` (public main 071b9aa, 04:19 CT) before any data. Stages on 2026-11-08: reach 04:20:40 to 04:21:18,
assay 04:22:50 to 04:49:49, samples 04:50:44 to 04:59:25 CT. After each stage the job folder was copied back and the host
record gate passed (every recorded model load eager; each stage's output present with its recorded size and sha256, and
all stages under one run id). The free answers were then read by Part 24b's unchanged reader (Claude Haiku 5.5 through
OpenCode Go, one call per answer): 192 of 192 labelled, no errors.

| Part 24's measures | Eager (this part) | SDPA (Part 24, published) |
| --- | ---: | ---: |
| 24.1 largest relative change at or below layer 22, any dose | 0.0 | 0.0 |
| 24.2 late-layer median relative change at 2x: identity / random | 0.482 / 0.805 | 0.480 / 0.805 |
| 24.3 identity name-token retention at 2x: identity / random | 0.585 / 0.665 | 0.583 / 0.647 |
| 24.4 " Alibaba Cloud." per token: none / random / identity zeroed | -2.888 / -2.972 / -3.137 | -2.885 / -2.960 / -3.136 |
| 24.4 extra drop from zeroing identity | 0.165 | 0.176 |
| 24.5 fixed-answer effect at 1x: none / identity / random | 1.551 / 1.603 / 1.556 | 1.552 / 1.601 / 1.555 |
| 24.5 fixed-answer effect at 2x: none / identity / random | 3.149 / 3.085 / 3.150 | 3.148 / 3.083 / 3.151 |
| 24b.1 identity zeroed, 2x: answers with an invented alien-named maker (of 16) | 12 | 13 |
| 24b.2 identity zeroed, 1.5x: alien-claim-only share among alien answers | 4 of 7 (0.571) | 6 of 8 |
| 24b.3 2x: answers with alien content, none / identity / random (of 16) | 15 / 16 / 16 | 15 / 16 / 15 |
| 24b.4 invented alien-named maker at 0x, 1x, 1.5x, 2x: none / random (of 16) | 0, 0, 0, 9 / 0, 0, 0, 8 | 0, 0, 0, 9 / 0, 0, 0, 9 |

Every verdict this rerun can score is the same on both kernels: 24.1 held, 24.2 held, 24.3 failed, 24.4 failed, 24.5
held, 24b.1 held, 24b.2 held, 24b.3 held, 24b.4 failed. All ten registered predictions held. The fixed-answer
measurements (24.1 to 24.5) move by at most 0.018; the free-answer counts move by at most one answer in 16, except 24b.2,
whose denominator is small (7 against 8).

## Why the kernel barely mattered here

The reach and assay stages read activations and log probabilities on fixed prompts and fixed answers, scored over short
spans; the sampled answers differ much more. Of the 192 free answers, 14 are identical to Part 24's, and the rest share a
median of 210 characters with their published counterparts before the first difference (same seeds on every cell). The
reader's counts stayed within one answer in 16 even so. Where in each answer the labelled content falls was not measured
here (Part 191 measured it for Part 153's company names, which sit in the first sentence).

## Predictions

| # | Prediction (registered wording, abridged; confidence) | Result | Verdict |
| --- | --- | --- | --- |
| 194.1 | 24.1 maximum relative change in layers 22 and below below 1e-3 (high) | 0.0 | held |
| 194.2 | 24.2 identity median 0.35 to 0.65, random 0.65 to 0.95, identity at most 1.5 times random (medium) | 0.482, 0.805 | held |
| 194.3 | 24.3 identity retention 0.45 to 0.75, random 0.50 to 0.80, identity below 0.8 (medium) | 0.585, 0.665 | held |
| 194.4 | 24.4 none/random/identity within 0.25 of -2.885/-2.960/-3.136; extra identity drop 0.05 to 0.30 (medium) | -2.888, -2.972, -3.137; 0.165 | held |
| 194.5 | 24.5 effects inside their registered ranges at 1x and 2x; identity/random difference below a quarter of the none effect (medium) | as in the table; 0.047 and 0.064 against 0.388 and 0.787 | held |
| 194.6 | 24b.1 identity-zeroed 2x answers with (b): 10 to 15 of 16 (medium) | 12 | held |
| 194.7 | 24b.2 (a)-only share among identity-zeroed 1.5x alien answers: 0.55 to 0.85 (medium) | 4 of 7, 0.571 | held |
| 194.8 | 24b.3 2x alien-content counts 14 to 16 of 16 in all three arms (medium) | 15, 16, 16 | held |
| 194.9 | 24b.4 (b) at 2x in 6 to 12 of 16 in none and random; at most 2 of 16 at 0x, 1x, 1.5x (medium) | 9 and 8; 0 throughout | held |
| 194.10 | The scorable published verdicts stay the same (medium) | all nine the same | held |

194.7 sits just inside its range (0.571 against a floor of 0.55): one fewer alien-claim-only answer among seven would
have failed it.

## Not tested

24.6 and 24.7 used hand labels (`results/part24/alien-labels.json`) on the original answers; the rerun's answers have no
hand labels, so those two verdicts are not scored, and the published 187 of 192 reader-to-hand agreement is not
re-measured.

## Limits

- One model, 16 fresh answers per condition, one blind reader. The free-answer counts are small samples.
- The A770's eager bf16 path is not the CPU float32 path; Part 178's reference check showed eager matching the CPU on its
  prompts, which is the reason for this audit, not a guarantee for every prompt here.
- Neither run records its torch or transformers version; the attribution of differences to the kernel rests on the
  attention setting being the only code change.

## Files

`load.json` (record), `launch.jsonl`, `reach.json`, `assay.json`, `samples.json`, `reread-24b.json` (reader labels),
`score-core.txt`, `score-24b.txt`, `reader-stdout.txt`.
