# Amendment 3: after Alden's code review of a3a40b1 and the DeepSeek reviewer's experiments (2026-11-09 18:47 CT, by Claude, by hand)

Nothing has run on a model. Two reviews of the stable build `a3a40b1` came back within the hour. Alden held the
implementation on four points (council `part201-design-review-2026-11-09` #11; full text at their path, sha256
`21f32237...`). The DeepSeek first-round reviewer ran 79 small experiments and hit its turn cap before writing a
review; I read its experiments' stated purposes and outputs from the run record
(`cheap-review-part201-build-ds-r1-2026-11-09`) and held each beside the design. Tests: 61 before, see the bottom line.

## Alden's four corrections

1. **Gate G5 compared two representations.** The generator's rectangle is padded after a row's end token; the
   recorder's rows are cut there. A batch whose rows end at different steps would have failed G5 with identical
   decoding. Both sides are now cut at their first end token, in one function the launch and the tests both call
   (`g5_from_capture`). Tests: Alden's toy, a mismatch before the end token in either row, a capped row that is not
   cut, and the tiny-model test now hands the function the raw rectangle.
2. **Gate G4 did not show the same push across passes, and a call could be read without its evidence.**
   - `push_sha256` is now a hash of the row's own payload (layers where it is not zero, with their numbers), so it
     does not depend on the call's other rows; the scorer requires one hash per row across every pass and process.
   - Each launch records canonical hashes of the units and medians it built pushes from; a launch refuses other ones
     before loading; the scorer requires them to agree across launches and to be the registered files'.
   - A call's receipt (members, time, generation steps, hook firings, decode forwards) is written to `load.json`
     BEFORE its rows, through a temporary file and a rename. A crash between the two leaves a receipt without rows,
     and the call is decoded again. Rows found without a receipt are moved to `rows.removed.jsonl` at the next
     launch and decoded again; nothing is deleted.
   - The scorer reads a row only through `loads[row.launch]`: that launch must be of the pass's process and hold a
     receipt for the call with the same members (Gate G3). A-again's rows must come from the launch that wrote their
     pass-A rows, and A2's from another launch, read from the launches and not from the `process` label.
   - Hook evidence is now required of every pushed call in every process, at run time and in the scorer; a missing
     receipt fails the gate where before it was skipped.
   - The scorer's baseline fixture gave every row launch 0. It is now consistent, with receipts for every call.
3. **A non-finite retry kept the first attempt's receipt.** The second attempt is now kept whole: its rows with its
   own time and hook counts, and the Gate G4 check is applied to the attempt that is written. The first attempt's
   time and counts are kept apart in the receipt (`discarded_attempt`). This also follows the design's words ("run
   once more; a row still non-finite is counted"): before, a retry that was still wholly non-finite wrote the first
   attempt's rows.
4. **The unsupported-row flag used an unpaired error.** `tools/shape_floor.py` now counts, per verdict, the redraws
   that are as observed only with the fallback and only with the rows held, and judges their difference against the
   standard error of the paired per-redraw differences. Alden's toy is a test (0.003148 against 0.009268).

Smaller note taken: the timing probe removes its output-head hook on every exit, with a test that makes the probe
raise.

## From the DeepSeek reviewer's experiments

5. **Three launch steps would have failed at import.** `tools/kernel_reference.py` imports `part_check`, the
   runner imports `tools.kernel_reference`, and `tools/part_check.py` was not in the `--copy` list of steps 1, 2 and
   4 (and is not on the job host). Added. New test (`CopyClosure`): each step's runner is run through its whole smoke
   path in a folder that holds only that step's copied files; without `part_check.py` it fails there.
6. **A hook that never fired looked like a matching push** (difference 0.0, no dtype, accepted in bf16). It now
   reports no measurement, and a pushed row needs bf16 at the hook in a bf16 process and float32 in the float32 one.
7. **"One model revision" rested on a constant.** A launch now reads the snapshot folder from `--model`, refuses
   one that is not the registered revision (a smoke run may load any small model), and records it; Gate G1 compares.
8. **A differing row with no recorded margin was dropped from 201.5 and 201.9 without a word.** The prediction is
   now not scored, with the count. A record that passes G2 has no such row.
9. **A missing pass-K record made the decision "Float32 arm invalid".** It is 201.7 not run, which is rule 4. A
   record that is present and wrong still gives rule 2. The plan's rule 2 says so.
10. **A smoke run in the default output folder** would freeze a smoke plan in `results/part201`. `--smoke` now
    refuses that folder.
11. Reported, no rule: how many pass-M rows were padded and how many M calls mixed prompt lengths; the margin shares
    are stated as over decodes.
12. `tools/shape_floor.py`'s note now says the shared-change coupling does not reach unsupported rows.

Checked and left as they are, with the reason: the float32 cut is decided once, from F8's first call, and a
relaunch carries it (by design); Gate G5 on a one-layer random model does not see every hook fault (17 of 20 seeds
stayed green under one sabotage), which is why the firing counts are checked on every pushed call and G5's scope is
now stated in the design; 201.11's base uses one time per call (the rows of a call share it by construction); the
scorer stops with an error on a hand-edited row holding a null margin.

## The design text

Gate G5 was in the build brief and the code and not in `PART201-DESIGN.md`. It is defined there now, with what it
cannot see. Gates G1, G3 and G4 carry the additions above; G4 is read per precision (its bf16 half is a bf16 gate).
`part201/LOAD-SCHEMA.md` describes `load.json` and the receipts. The registered predictions and their bars are
unchanged, and so is the plan (sha256 `94dc3fbd...`).

## Shown able to fail

`part201/sabotage201.py` breaks one guarded line at a time in a temporary copy and expects named tests to go red.
Its first run found one test that stayed green (a missing push hash was caught by a different guard than the one
the test named); the test now isolates its guard. See the bottom line for the count.

Sabotage check on this state: 29 of 29 sabotages turned every named test red (`python -I part201/sabotage201.py`).

Tests on the host: 102 pass (recorder 5, runner 41, scorer 45, shape floor 11), with
`python -I -m pytest -q tools/test_margin_recorder.py part201/test_part201.py part201/test_score201.py tools/test_shape_floor.py`.
Not yet done: a smoke on a real small model (it waits for free Windows RAM).
