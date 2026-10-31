# Part 141: how fast does the Neuron Lens fit run on the Arc?

Procedure: v1
Registered 2026-10-31 01:10 CT by Claude (Opus 5.5), before any run. Elise handed Prime the port on 2026-10-30 (council topic orchestrator-olmo-lens-2026-10-30, message 2, 19:45 CT); 2026-10-30 23:12 CT (Discord message 1565941412451713035): OLMo is for
when we need its checkpoints. The commit that adds this file is the registration.

## Question

Orchestrator Prime ported the Neuron Lens R-lens and S-lens fit to OLMo 2 1B (PRIME-LENS-REPORT.md), with a CPU acceptance fit at 1,361 s per document. A lens at each training stage would let
us follow a neuron's read across training, which is where our work is distinct. Rented GPUs are off. Can the A770 do it, and at what cost?

## Prior art and existing data

- Closest published work: Neuron Lens, stdlogits/neuron-lens at 07fdf45 (R-lens and S-lens, Qwen3-8B and Gemma 4 12B, 11 to 12 s per document on an H100 for Qwen3-8B).
- Our own data: results/prime-lens/olmo2-1b-instruct (CPU timing, split-half, the alien-twin read); fit_jlens_olmo.py (an earlier J-lens on OLMo).

## Method

fit_olmo_lens.py at the merged port, OLMo 2 1B Instruct at revision 48d788ec, bf16, A770, dim-batch 64, one document, both lenses and both targets, --timing-only. No model output is generated.

## Controls

Not used: the standard controls apply to pushes and readers, and this part only times a fit. The fit's arithmetic is tested in tests/test_olmo2.py against brute-force Jacobians.

Gate: the timing run finishes with no spill into shared memory and no RAM guard stop.

## Size

- Smallest effect worth knowing: whether 200 documents at one stage fit in one night.
- Total: 1 document, about 2 minutes plus the model load.
- Pilot: not needed: under 1,000 generations and one GPU-hour.

## Optimization

- Bound resource: GPU
- Measured: n/a: under 5 minutes.
- Levers: dim-batch 64 taken from the CPU fit; copy engine off by default; no others apply to one document.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 141.1 | One document, both lenses, takes at most 120 s on the A770. | medium | Claude |
| 141.2 | Peak allocated GPU memory is at most 12 GB. | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 141 (registered 2026-10-31, PART141-DESIGN.md): how fast does the Neuron Lens fit run on the Arc?".

## Reading rules

timing.json gives seconds per lens and peak memory; the README states them. If 200 documents project under 4 hours, a later part fits two or three OLMo stages.

## Limits

One document, one stage. The first document may include kernel compile time, so the timing may run high.
