# Loop onset against the 64-key point (exploratory, before Part 199's design)

Not pre-registered. Run on the published answers of Parts 165, 168, 169 and 170 on 2026-11-09, before Part 199 was
designed, and cold-checked by Claude Haiku 5.5 the same morning.

**Why.** The A770's fused bf16 SDPA kernel leaves the CPU float32 path once the cache holds more than 64 keys (Part
178, Part 183). Generating answer token t attends over prompt length + t keys, so the first token the kernel can change
is t0 = 65 − prompt length (35 to 49 for these prompts, which are 16 to 30 tokens). Part 183's own SDPA control fits
this: its six departures at a confident position all fall at or after t0, two of them exactly at t0.

**What.** `loop_onset.py` finds, for every published answer that loops by Part 166's rule (a word trigram three or more
times), the first answer token at which its decoded prefix already meets the rule, ignoring the prefix's last word
unless the prefix is the whole answer (so a loop can be placed late, never early). A loop established before t0 was
decided by steps the kernel could not change; one established at or after t0 is exposed.
`loop_onset_envelope.py` then bounds each published contrast under the assumption that the faulty kernel only added
loops: each arm's correct-kernel count is its early loops plus anywhere from none to all of its exposed loops.

| Part | Contrast | Published | If the kernel only added loops |
| --- | --- | --- | --- |
| 168 | own-word ban cut, low-k | 31.4% | 5.8% to 51.6% (bar 30%: not settled) |
| 169 | own-word ban cut | 37.9% | 22.5% to 53.6% (a cut either way) |
| 169 | plain50 minus none | +283 loops | +246 to +350 |
| 169 | free minus none | +0 loops | −65 to +67 (not settled) |
| 170 | dose-1 b50 minus b0 | +18.8 points | +17.7 to +22.4 |

The early-only point (every exposed loop dropped from both arms) is one allocation, not a bound: the first version of
this analysis called it a bound, and the cold check showed that if only a baseline's late loops were fabricated a cut
shrinks. The intervals above replace it.

**Limits.** It bounds one direction only: loops a correct kernel would add, and the faulty kernel prevented, are
invisible (Part 191 found SDPA answers cut short). Part 165's headline is a broken rate, not covered; its rows carry no
token ids, so its loops were re-encoded from text. Parts 165, 168 and 169 carry no prompt ids, so their prompt lengths
come from the no-think chat template, which reproduces all 200 of Part 170's recorded prompts exactly.

Run (neuron-probe venv): `python -I loop_onset.py results/part170/answers.jsonl . <Qwen3.5-4B snapshot dir>
results/part170/answers.jsonl stage,group,arm,dose,ban` and likewise for the others (grouping `stage,arm,ban` for 168
and 169, with `loop-onset-q168.json` for 168; `stage,arm,k` for 165); then `python3 loop_onset_envelope.py`.
