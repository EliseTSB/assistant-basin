# Brief A: build Part 201's recorder and runner

Repository: `/home/elisepc/green/worktrees/ab-part201` (branch part201). Write only inside it. No GPU, no network, no
`git commit`, no `git push`. Python: `/home/elisepc/neuron-probe/.venv/bin/python -I -B` (it has torch and
transformers; if your sandbox cannot import torch, say so in the report's first line and still write the tests).

**Write `part201/REPORT-run.md` FIRST** as a skeleton ending in SUMMARY / FINDINGS / BLOCKERS lines; keep it updated;
copy it to `report.md` in your run directory at the end.

Read first: `PART201-DESIGN.md` (the design you implement; Method and Controls are the contract),
`part201/ROWS-SCHEMA.md` (the output contract, shared with another builder who writes the scorer),
`PROCEDURE.md` sections 4 to 6 and its lessons log from 2026-10-30 on, `part200/run200.py` and `part199/run199.py`
(the launch record, folder lock, eager-loader refusal and resume you should reuse by import where they fit),
`part163/run163.py` (units, questions, `pushes_for`), `part86/common86.py` (the row-isolated push hook and
`generate_batch`), `part160/run160.py` (how a LogitsProcessor is added around `model.generate`), and
`tools/kernel_reference.py` (`upcast_compute`, `decode_rows`).

## Deliver

1. `tools/margin_recorder.py`: a `LogitsProcessor` subclass `MarginRecorder`. Each call copies `scores` to the CPU as
   float32 ONCE (one transfer per step; no `.item()` per value; no top-k on the device), takes the top two values and
   ids per row, stores them, and returns `scores` unchanged (the same object). After generation,
   `recorder.rows(sequences, prompt_length, eos_ids)` returns, per row, the fields `tokens`, `top1`, `top2`,
   `top1_logit`, `margin`, `argmax_agrees`, `nonfinite`, `finish` of the schema. It must work when rows finish at
   different steps (later steps for a finished row are ignored) and for any batch size. A function
   `bf16_step(value)` returns the spacing of bfloat16 at `abs(value)` (0.125 for 16 <= |v| < 32, 0.0625 for 8 <= |v| <
   16, and so on by powers of two). This file will be reused by later parts: no Part 201 names in it.
   `tools/test_margin_recorder.py`: unit tests on hand-built score tensors (ties, a row that ends early, a non-finite
   score, batch 1 and batch 3), `bf16_step` checked against `torch.nextafter`-style spacing computed from a bf16
   tensor, and one deliberate break (for example, record the top-2 of the previous step) that turns a test red.
2. `part201/run201.py`:
   - `plan(units_path)`: the deterministic plan of every pass (rows, calls, slots) from the design: the 24 sampled
     units, the subsets, pass B's reshuffle rule, pass M's shuffle, plain rows. Pure Python, no torch. `--print-plan`
     writes `results/part201/plan.json` and prints its sha256. The design's counts must come out: A 384 rows in 48
     calls; A-again 16 in 2; B 384 in 48, no call sharing a membership with an A call; M 384 pushed in 48 calls plus
     24 plain in 3; C 192 + 24 calls of one row; A2 96 in 12; F8 96 in 12 with the same membership and order as the
     corresponding A calls restricted to F-subset rows (say in the report exactly how you built these and whether a
     call can end up with fewer than 8 rows; if the design's wording is ambiguous, pick the reading that keeps every
     F8 call at 8 rows and flag it); F1 48.
   - `generate_rows(model, tok, layers, prompts, pushes, max_new_tokens, device, recorder)`: ONE decode path used by
     every pass. Rows may have different prompts: left-pad with the tokenizer's pad token and pass the attention
     mask; same chat template, thinking off, no system prompt, as Part 163. The push hook is Part 163's row-isolated
     hook with one push per row. Greedy, `max_new_tokens` 80, no early row removal. The recorder is the last logits
     processor. Returns sequences and per-row recorder output.
   - `--process bf16-main` runs passes A, B, M, C then A-again; `--process bf16-second` runs A2; `--process float32`
     loads, calls `kernel_reference.upcast_compute`, records its summary, then runs F8 and F1. Each process: eager
     load through `neuron_report.load_model` with Part 200's refusal on a non-eager load; a load record
     (`results/part201/load.json`, one entry per launch, with the pass list, dtype, the upcast summary when present,
     and per call the pass, row count and seconds); the folder lock; rows appended to `results/part201/rows.jsonl`
     after every call; a relaunch skips (pass, row_id) pairs already written.
   - Gate G5 (add it to the report as a design amendment for the reviewer to accept): before pass A, decode A's first
     call through Part 163's own path (`part163/run163.py`'s default `generate_fn` with the same pushes) and through
     `generate_rows`, and stop with exit 3 unless the token ids are equal on all 8 rows. Record the result in
     `load.json`.
   - Pilot: after A's first 3 calls, compute answers per second; under 0.3, stop with exit 3 and say so. In the
     float32 process, time the first F8 call and write the projected minutes for F8 and F1 into `load.json`; if over
     45, apply the design's cut (F8's first 48 rows, F1's first 24) and record that the cut was applied before any
     float32 row was read.
   - `--check-record results/part201/load.json` on the host repeats Gates G1 and G3 from the records and rows.
   - `--smoke`: runs every pass on the first 2 sampled units per arm and the first 2 identity and first 2 maker
     questions, plus 4 plain rows, on any `--model` and `--device` (for a host run on Qwen3.5-0.8B on the CPU,
     including the float32 process), writing under `--out`.
3. `part201/steps.txt` in the style of `part200/steps.txt`: four launch lines for `tools/launch_arc.py 201`
   (`bf16-main`, `bf16-second`, pass K through `tools/kernel_reference.py --device xpu --dtype float32-upcast
   --attn-implementation eager --reference results/kernel-reference/qwen-qwen3.5-4b-cpu-float32.json --out
   results/part201/kernel-f32.json`, then `float32`), each with its exact `--copy` list (read how
   `tools/launch_arc.py` takes `--copy`: one flag, several paths), then the host lines.
4. `part201/test_part201.py`: plan counts and determinism; B's membership rule; resume (a second run writes nothing
   new); a tiny-model end-to-end run of `generate_rows` on the CPU on the SAME architecture family as the target (see
   how `part178/test_part178.py` builds its tiny Qwen3 model; PROCEDURE.md's 2026-11-02 lesson is why), mixed prompts
   with padding included, with `argmax_agrees` true on every row; G5 on the tiny model; the float32 path on the tiny
   model: after `upcast_compute` the hidden states are float32 and a pushed row's push values equal the bf16 values
   cast up (check what the push hook does when the activation is float32 and the stored push is bf16; fix by casting
   inside this part's code, not by editing `part86/`); one deliberate break per gate that turns a test red.

## Rules

- Do not edit `part86/`, `part163/`, `part199/`, `part200/`, `tools/kernel_reference.py`, `PROCEDURE.md`,
  `PREDICTIONS.md`, `README.md` or `PART201-DESIGN.md`. If the design is wrong or cannot be built as written, build
  the nearest thing, and list each difference in the report under "Design differences" with the reason.
- Windows runs this: open every file with `encoding="utf-8"`; paths passed on the command line may be `E:/...`.
- Run your tests and paste the final pytest summary line into the report. A test you could not run is listed as
  "not run" with the reason, never as passed.
