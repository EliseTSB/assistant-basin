# Part 87 round-1 cold review (Claude Haiku 5.5, read-only, 2026-10-28 02:0x CT)

Run by a haiku-helper subagent against part87/ at 1b24fa9 with PART87-DESIGN.md and Amendment 1. The reviewer returned its review as text; Claude saved it here unchanged from "Surviving mutations" onward.

Surviving mutations: **15 of 15** real mutants (R1-R13, S1, S2). Controls R14 and S3 were caught.

Unmodified suite (base copy, `/home/elisepc/neuron-probe/.venv/bin/python -I -B part87/test_part87.py`): `Ran 12 tests in 0.142s` / `OK`, exit 0.

**Findings, most severe first**

1. **BLOCKER**: No test reaches the run loop. `/home/elisepc/green/worktrees/ab-part67/part87/test_part87.py` never calls `_generate_one`, `run_generation_phase` or `measure`, and checks `_arm_specs` only for arm names and counts (lines 154-169). Temperature, sampling, hook layer, unit and sign, token positions, zeroing mask, chat template, seeds, and unit or rule assignment can all change with the suite green. I ran `_generate_one` on a stub model (probe below). The current code matches every registered setting: E63 pushes L22 N2742 on all prompt positions plus 64 decode steps; D47 is decode-only; C63 pushes N8870; Z zeroes every token; M and F deltas cancel the mean with the intended sign; temperature 0.7, top_p 0.95, do_sample true, 64 or 48 tokens. The gap is verification, not the code. Fix: a stub-model test of `_generate_one` and `_arm_specs` asserting generate kwargs, prompt string, and each hook's layer, unit, positions and sign.

2. **MEDIUM**: NOT READ paths are untested. S1 (`score87.py:239` drops the random-arm check) returns HOLDS for 87.7 with RM_8705 missing. S2 (`score87.py:209` drops the origin check) returns FAILS for 87.3 with no labels. Current code returns NOT READ for both. Add tests for each missing-input case.

3. **MEDIUM**: 87.7 is one verdict, but the posting rule is per arm. PREDICTIONS.md line 2170 requires both M and F. RUNBOOK line 109 offers the higher-Google arm "only if that arm holds 87.7". If M holds and F fails, the scorer prints `87.7 FAILS` while the rule allows offering M. Print per-arm statuses (`score87.py:236-257`).

4. **MEDIUM**: Blinding differs from the design. PART87-DESIGN.md line 42 says the 87A items are "shuffled with 87B's under random ids". `prepare_blind` reads 87A only (`score87.py:70`), and RUNBOOK line 87 passes only `p87\run`. Add the 87B answers to the reader's items, or amend the design before the reader runs.

5. **MEDIUM**: The GPU guard default is loose. `part87.py:711` defaults `--max-gpu-gb` to 20. Other runners in the repo default to 14.5 (`neuron_report.py:77`, `part25.py:814`). Above that limit `spill_guard` (`part87.py:682`) lets memory spill to shared system RAM instead of stopping, and the RUNBOOK never passes the flag. Set 14.5 and add the flag to the run command.

6. **MEDIUM**: Scoring commands are not bound to the venv. RUNBOOK lines 62, 87, 93 and 99 run `score87.py` as a bare script. It imports torch through part87 and part66. With the system Python I got `ModuleNotFoundError: No module named 'torch'`. Prefix each with `C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe`. Line 24's bare `python` follows Part 86's convention, so LOW.

7. **LOW**: RUNBOOK line 108 covers only 87.3 failing. Amendment 1 (design lines 88-93) says the 87.3-holds, 87.5-fails case goes "exactly as when 87.3 fails". Add that sentence.

8. **LOW**: `measure` never checks the calibration hash. Part 84's recorded hash is `afb86522...` (`results/part84/main/run.json`). `part87.py:305` records the hash without comparing it, though the design says "exactly as Part 84 did". The calibration file is not in the repo, so this could not be verified here.

9. **LOW**: The reader gets conflicting instructions. RUNBOOK lines 10 and 16 say copy `blind_origin.py`, which describes "twin vs control vs unclamped" (`blind_origin.py:2, 62-63`). Line 90 says the rubric only. Give the reader a rubric file.

10. **LOW**: Hand reads have no ingest path. `_hand_read_md` (`score87.py:291`) writes a blank column, and no output field carries the passing-mention counts the design asks to report beside the rate.

11. **LOW**: `score()` skips the answers-digest check when `--a-answers-dir` is omitted (`score87.py:349-351`). RUNBOOK always passes it.

**Checked, no defect**: The arm translation (`score87.py:29-30`) is a consistent bijection. Blind items carry only id and answer text (`blind84.py:15`). Sizes 44 and 1,560 are right, and the 192 held-out rows are in `results/part66/full/answers.jsonl`. Thresholds match PREDICTIONS.md lines 2164-2172. The RUNBOOK copy list is sufficient: a tree with exactly those files passes `Ran 12 tests` / `OK`. No Linux-only path appears in code or RUNBOOK; the only one is BUILD-NOTES line 14.

**Mutation table**
Copies: `/tmp/claude-1000/-home-elisepc-green-Journal/b92c413b-50df-4aff-b0c5-78b245e5acf8/scratchpad/p87review/p87review/m<ID>`. Base: `.../scratchpad/p87review/base`. Results: `.../scratchpad/p87review/p87review/mutant-results.json`.

| ID | Change | Result |
|---|---|---|
| R1 | generate temperature 0.7 to 1.0 | survives |
| R2 | alien push on layer 21 | survives |
| R3 | 87A dose sign flipped | survives |
| R4 | add hook removed after registration (no push) | survives |
| R5 | Z zeroes generated tokens only | survives |
| R6 | E63 decode-only | survives |
| R7 | C63 pushes N2742 | survives |
| R8 | generation prompt without chat template | survives |
| R9 | RF arms use M-rule doses | survives |
| R10 | do_sample False | survives |
| R11 | seed 0 for every row | survives |
| R12 | measure residual median from L22 | survives |
| R13 | F-rule sign flipped | survives |
| R14 | control: 4 samples instead of 8 | caught |
| S1 | 87.7 ignores missing random arms | survives |
| S2 | 87.3 ignores missing origin | survives |
| S3 | control: 87.9 threshold 80% | caught |

Stub probe: `/tmp/claude-1000/-home-elisepc-green-Journal/b92c413b-50df-4aff-b0c5-78b245e5acf8/scratchpad/p87review/probe_gen.py`, run with `-I -B`. It uses synthetic means and norms, so it checks signs, positions and kwargs, not magnitudes on real data.

VERDICT: DO-NOT-SHIP
