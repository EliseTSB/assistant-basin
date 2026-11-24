# Part 218 v4 pilot, 2026-11-24: the stop rule passed

Run: Colab L4 session `v218` (wrapper `Journal/notes/reports/part218-build-2026-11-24/colab/v4_pilot.sh`), runner
`part218/run218.py` as registered on public main at `74fd468c` (sha256 prefixes on the VM: run218 `a2d78efdff4e879a`,
prompts218-v4 `d84ff8497a182896`, pins `0a0ff6dfa9846675`, all matching the commit). The Instruct pilot ran 16:38 to
16:59 CT: layer 5, nine doses, 40 directions plus the zero row, 88 items, 32,472 of 32,472 rows, 792 of 792 batches, none
resumed, 1,208.6 s of wall time. Then the zero-push gate passes: SFT 3,608 rows in 99.3 s, DPO 3,608 rows in 107.6 s.
`directions.pt` was created on the VM (sha256 prefix `c6b7b00017786035`) and is the file the main run reuses. Colab cost
1.79 units (171.00 left).

Scored with `part218/score218.py --run-dir results218-v4` (gold set 87 of 87).

## The gate and the stop rule

Amendment 1: "if any model fails the gate, or no rung of the Instruct pilot reaches 36 of 40 supported directions, the
main run waits." Neither happened (`stop_rule.triggered: false`).

| Model | Eligible items (min 60 of 88) | Per opening (min 24 of 44) | Eligible prompts | Passed |
| --- | --- | --- | --- | --- |
| Instruct (pilot) | 70 | 37 / 33 | 39 of 44 | yes |
| SFT (gate pass) | 75 | 40 / 35 | 42 of 44 | yes |
| DPO (gate pass) | 72 | 36 / 36 | 38 of 44 | yes |

- Zero rows: 88 of 88 Instruct items identical across their nine batches; 88 of 88 items all-identical in both gate
  passes.
- No-hook null on Instruct: start against end and against the hooked zero arm, 88 of 88 ids and logits, max difference
  0.0.
- Gate passes pushed nothing: `push_check` on SFT and DPO reads pushed delta 0.0, zero row exact (the measured zero push
  the stop rule requires since the review fold).
- Read rung (largest d with at least 36 of 40 directions supported): layer 5 at d = 0.5, 37 of 40. At 0.75 only 25 are
  supported.

Supported and moved directions by pilot dose (layer 5, 70 eligible items):

| d | 0.15 | 0.20 | 0.25 | 0.30 | 0.40 | 0.50 | 0.75 | 1.0 | 1.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| supported | 40 | 40 | 40 | 40 | 39 | 37 | 25 | 17 | 3 |
| moved | 3 | 6 | 14 | 22 | 32 | 29 | 15 | 4 | 0 |

## The trim rule

Broken fraction by Method rung: 0.25 0.003, 0.5 0.038, 0.75 0.112, 1.0 0.209, 1.5 0.467. No visited rung is over half
broken, so nothing is dropped; 2.0 and 3.0 were not visited and are kept as undetermined. The main run uses the full
ladder (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0).

## What the pilot already shows about the predictions (written before the main run started)

The pilot is the main run's Instruct layer-5 slice at the shared doses: the same 40 directions, the same 41-row batches,
the same dose scale, greedy decoding. So its rows at d = 0.5 should come back unchanged in the main run, and d = 0.5 is
the rung the read rule picks on these numbers. The scorer printed that slice's pushed readout: **29 directions moved,
led by Google 13, OpenAI 11, Microsoft 1, Meta 1; 3 not led.**

- 218.1 and 218.7 (Instruct, OpenAI leads at least 60% of moved directions at one layer's read rung): on these rows
  layer 5 does not meet it (11 of 29). Layer 10 is unseen, so neither is decided.
- 218.3 (Instruct, Google leads at most floor(moved / 5) at every qualifying layer): on these rows layer 5 has Google
  leading 13 against a cap of 5, so 218.3 fails if the main run reproduces them.
- 218.4, 218.8 (gate) and 218.5 (zero-row identity and nulls): the pilot and gate passes are the same zero rows the main
  run regenerates, so all three are effectively known to pass. That exposure is built into Amendment 1's design (the
  gate passes exist to check eligibility before spending the main run).
- 218.2 (SFT) and 218.6 (the SFT-to-DPO shift) are unseen: no pushed SFT or DPO answer exists.

The predictions keep their registered text and confidences. This section is the disclosure, as the v3 pilot's pushed
readout was disclosed in Amendment 1.

## Next

Main run: 3 models × 2 layers × 7 doses × 88 items × 41 rows = 151,536 generations plus no-hook nulls, on one L4 with
the pilot's `directions.pt`. At the pilot's measured rate (32,472 pushed rows plus the null in 1,208.6 s, about 33 rows/s)
it is about 1.5 hours of generation, roughly 4 Colab units, inside Elise's standing budget.
