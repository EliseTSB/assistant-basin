# Part 138: can OLMo 3 7B run reliably on the Arc?

Procedure: v1
Registered 2026-10-30 23:15 CT by Claude (Opus 5.5), before any run. Elise, 2026-10-30 23:12 CT: "We should try getting OLMO 3 7b running
reliably to." The commit that adds this file is the registration.

## Question

OLMo 3 7B spilled out of the A770's memory and froze WSL on 2026-10-16. What load path and batch size run it without a spill, and how fast?

## Prior art and existing data

- Closest published work: none needed; this is a measurement of our own rig.
- Our own data: Part 9 ran OLMo 3 7B at two answers at a time with a cache clear; notes/gotchas.md records the spill, the RAM guard and
  the direct-load fix; Part 115b's load_direct and Part 86's batch ladder.

## Method

Model: allenai/Olmo-3-7B-Instruct, local copy on E:, bf16, device_map straight to xpu with the allocator warm-up patched. part138/ladder138.py
times one model load, then generates 16 fixed questions at 64 tokens at batch 2, 4, 8, 16 and 32, stopping at the first batch where the
card's shared memory rises above 1 GB or a generation fails. Each batch prints answers/s, peak dedicated and shared memory, and
Windows RAM available. The largest safe batch then repeats three times, with torch.xpu.empty_cache between runs.

## Controls

Not used: standard controls, because nothing is pushed and no reader runs. The check that can fail: a stated spill line, 1 GB of shared
memory, and a second pass of the same batch whose greedy answers must match the first at the same batch shape.

Gate: the plain model answers "What is the capital of France?" with Paris at every batch. A failed gate stops the reading.

## Size

- Smallest effect worth knowing: a batch doubling that gains under 10% answers/s is not worth its memory.
- Total: 128 answers at 64 tokens, 5 batches x 16 answers plus 3 repeats x 16, about 15 minutes.
- Pilot: not needed: the ladder is the pilot and stops itself.

## Optimization

- Bound resource: GPU
- Measured: 2 answers at a time on 2026-10-16, no rate recorded; this part is the measurement, 1.82 answers/s at batch 16 on Part 86 for the
  smaller 4B.
- Levers: copy engine off taken; direct load taken; a rate per batch is the output.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 138.1 | The model loads with 10 GB of Windows RAM free without tripping the guard. | medium | Claude |
| 138.2 | The largest batch with no spill is at least 8. | medium | Claude |
| 138.3 | At that batch the rate is at least 1 answer/s at 64 tokens. | medium | Claude |
| 138.4 | Three repeats at that batch all finish without a spill. | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 138 (registered 2026-10-30, PART138-DESIGN.md): can OLMo 3 7B run reliably on the Arc?".

## Reading rules

part138/ladder138.py writes results/part138/ladder.json and every number comes from it. A spill or a freeze is a result: the batch is
recorded and the ladder stops.

## Limits

One revision, 64 tokens, one machine state; a game running alongside changes the free memory.
