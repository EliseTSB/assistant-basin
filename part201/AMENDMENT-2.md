# Amendment 2: what changed after the two builders finished (2026-11-09 18:01 CT, by Claude, by hand)

Builder A stopped on a HOLD (it could not reach the coordination topic to add the `launch` field). Builder B
finished. I read both builds against the design before any model run and changed the following. Tests: 60 before,
now see the bottom line.

## Defects in the build (none had run on a model)

1. **Pushed rows had no `neuron` or `random_index`.** `part163.run163.pushes_for` reads them from each row, so the
   first pushed call would have raised `KeyError: 'neuron'` on the card, after the load. The plan rows now carry
   them. Found by reading; pinned by a new end-to-end test that fails with that error when the fields are removed.
2. **Pass F1 was planned as 6 calls of 8 rows.** The design says one row per call (it is the float32 batch-1 arm).
   Now 48 calls of one row. The builder's test asserted the wrong shape.
3. **The scorer's Gate G1 could not pass.** It expected pass A2's rows in process `bf16-main` (they are written by
   `bf16-second`), and it expected the kernel record's `dtype` to be `torch.float32` (a float32-upcast record says
   `torch.bfloat16` there and `float32-upcast` under `dtype_requested`). Every one of the builder's scorer tests ran
   with the gates switched off. The gate is now tested on the real CPU reference record, which it accepts, and on
   Part 183's bf16 record, which it refuses.
4. **A failed or absent float32 arm withheld the bf16 readings too.** Gates G1 and G3 are now read per precision, as
   the design's stopping rule needs.
5. **The float32 cut did not apply to the loop that was running, nor to a relaunch**, and its projection priced a
   one-row call at an eighth of an 8-row call. Both fixed; the share is 0.4635, from Part 200's rates.
6. **A relaunch after a clean finish disowned the A-again rows** (it recorded "not same process" with the rows
   already written). Now it records that only when the repeat is still unwritten.
7. **The tie count compared with `<` where it needed `==`**, so it could never count an emitted lower token id.
8. **Speed, in the recorder and the hook.** The recorder turned the whole vocabulary into a Python list per row per
   step; it now keeps the step's score tensor. The hook copied a difference to the CPU at every firing; the delta is
   the same tensor each time, so it is now measured once per call.
9. **The timing probe ran the output head on every position** (about 470 MB of float32 scores at 8 rows and 96
   tokens). It now runs the head on the last position only, which is what a re-check needs, and records that.

## From Alden's third read (council part201-design-review-2026-11-09 #7)

10. `tools/shape_floor.py` rewritten: the cluster scenarios share one draw for the event "this row changes" (no
    rank over destinations, so the listing order cannot matter); scenarios are `independent`, `shared-change-0.5`,
    `shared-change-1`, `upper-marginal-independent`; a support table per starting label vector; a label value
    never seen as a starting value is held and counted, never given rate 0 or 1; each verdict also scored with
    unsupported rows held, and flagged when that changes it; Monte Carlo error beside each figure.
11. The candidate gate is priced at its own width. 201.11 stays the two-step forecast; the plan's rule 5 also needs
    the projected slowdown at the selected gate to be at most 2. Prices beyond the longest probed prefix are counted
    and called an approximation.
12. The probe's hook lifecycle is tested on the tiny model: the push hook fires on every full-prefix forward, in
    float32, and leaves nothing behind.

## Also

13. Every launch records the device as torch reports it and the Windows display-driver version (Muse's prior art:
    a driver update can move this card's outputs; ours is 32.0.101.9150).
14. The scorer takes its expected rows from the frozen `plan.json` and checks that file against the plan the runner
    builds from the registered units. A positive control (A and C differ on at least 10 rows) gates every reading.
15. The plan hash changed (F8 order, F1 calls, push fields in rows). The hash printed in Builder A's report is void.

Not yet done: a smoke on a real small model (Windows had 2.9 GB of RAM free), and a cold code review of this state.
