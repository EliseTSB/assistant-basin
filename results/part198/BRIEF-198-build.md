# Brief: build Part 198's run and score code (no model runs)

You are building code for a pre-registered experiment. The design is the contract:
`/home/elisepc/green/worktrees/ab-part198/PART198-DESIGN.md` (revision 3; read all of it, especially Method, Controls,
Predictions, Reading rules and the three revision logs at the end). The statistics are already written and frozen in
`results/part198/stats198.py`: import it, never edit it, never reimplement what it provides. Its fixtures
(`fixtures198.py`) pass; keep them passing.

Defensive research: benign counterfactual facts from a public benchmark; nothing is built to hide an edit.

## Hard limits

- **Do not load any model, run any edit, or call any network.** This machine is a person's PC and one model job at a
  time can exhaust its memory. Allowed: `python -m py_compile`, `fixtures198.py`, your own unit tests on tiny synthetic
  tensors, and `run198.py ... --dry-run` (prints the plan from `cases198.json` and the pinned paths without importing a
  model).
- Write only inside `results/part198/`. Do not commit (the git directory is outside your domain); the parent commits.
- Python: `/home/elisepc/edit-detect/.venv/bin/python -I` (has torch, transformers 5.20, safetensors, EasyEdit's deps).

## Inputs you build on (read them)

- `results/part198/cases198.json`: every case ID, frozen. Do not redraw.
- `results/part198/pilot/pilot198.py`: how the GPT-2 XL edits were made in the pilot (EasyEdit via `ee_import.py`,
  hyperparameter loading, CounterFact requests, efficacy, restoring weights). Copy `pilot/ee_import.py` to
  `results/part198/ee_import.py` unchanged and import that.
- `results/part198/pilot/wsp198.py`: how `wsp.py` is loaded (by path, `/home/elisepc/edit-detect/wsp-src/wsp.py`, sha256
  `90184edf8f3e1ce7a25cb2697f4fe94f92befef1fbcbe64c29c3ac0b0ae65f81`; verify the hash at startup and refuse on mismatch;
  never copy wsp.py into the repo, it has no license).
- `results/part198/crosscheck198.py`: how a LayerNorm module is built for GPT-2 XL's `ln_f`, how Qwen3.5-0.8B's layers and
  final norm are found, and how `stats198` is called.
- Data paths: GPT-2 XL `/mnt/e/neuron-probe/edit-detect/gpt2-xl`, second moments
  `/mnt/e/neuron-probe/edit-detect/stats`, EasyEdit `/mnt/e/neuron-probe/edit-detect/src/EasyEdit-main`, CounterFact
  `/mnt/e/neuron-probe/edit-detect/counterfact.json`, Qwen3.5-0.8B snapshot in `/mnt/e/neuron-probe/hf/hub`. Arm C
  models are in `/mnt/e/neuron-probe/hf/hub/models--<org>--<name>/snapshots/<hash>/` (resolve the one snapshot at run
  time; run offline).

## What to write

1. `run198.py` with subcommands, each one process, each appending one JSON line per record to
   `out/<arm>-<method>.jsonl` and skipping case IDs already present (resume). Also log failures and restarts as records.
   - `armA --method ROME|MEMIT|AlphaEdit`: GPT-2 XL float32, weights cloned into ordinary memory after loading (see
     pilot198.py). Base statistics for all 48 `mlp.c_proj` once (F1, F2, PCS, token pairs) saved to
     `out/base-gpt2.json`. Per record: before/after efficacy (first token, as the pilot), `already_satisfied` and
     `took` exactly as the design's transition rule; the update dW per edited matrix (spectral and Frobenius norms,
     ratio to base sigma1, numerical rank); edited-layer statistics; localization per statistic (F1, F2, PCS, T_max,
     T_word) using `stats198.localized` with edited values for edited layers and base values elsewhere; the blind
     reader and the oracle reader (design: naming section); the update's own top direction read for naming
     (base-required diagnostic); base diagnostics beside edited values. **ROME process only:** the aimed positive control
     per record exactly as the design's Controls section (layer 17, ROME update's right singular vector as input
     direction, its spectral norm, output = centred, gamma-scaled unembedding row of the object's first target token,
     unit length), scored for naming (oracle, layer 17) and localization (T_word). AlphaEdit with `reset_cache=True`
     on every call. After each record restore and assert bit-exact equality with the base; after the last record,
     re-run the first record and compare its update (design: Restoration and editor state).
   - `armA-mass --method MEMIT|AlphaEdit`: the two batches from `cases198.json`; per-record before/after takes, the
     8-of-10 gate, localization, and per-object naming (design: Mass edits).
   - `armB`: `wsp` on Qwen3.5-0.8B, the 24 single writes with the design's frozen question templates (copy the table
     into code verbatim), took and already-satisfied per the design, statistics, blind and oracle readers, the paired
     matched random control (one suspect per write, all five layers, seed 198 + case ID, same singular values, random
     orthonormal singular vectors), undo and an exact-zero check, cross-layer consensus (exploratory).
   - `armB-seq`: the 10-write sequence; statistics after every write; strict naming of every object written so far.
   - `armC`: every model in the design's Arm C list. T\* from the three calibration models written to
     `out/tstar.json` first, then the evaluation models, then the two hosts (float32). Per model: per-layer T_max,
     T_word, top eligible pair, F1 and its robust z, largest/second sigma1 ratio, consensus. Locate MLP output matrices
     generically (GPT-2 `c_proj` is a Conv1D and must be transposed; GPT-NeoX uses `mlp.dense_4h_to_h`; Llama-family uses
     `mlp.down_proj`), the final norm (`ln_f`, `final_layer_norm`, `norm`, ...) and `get_output_embeddings()`. Load at
     stored precision (bfloat16 for bf16 checkpoints), never down-cast; statistics in float64 per matrix.
   - `manifest`: write `out/manifest.json` with sha256 of every pinned input (model weight files, EasyEdit method
     sources under easyeditor/models/{rome,memit,alphaedit}, the three GPT-2 XL configs, the five second-moment files,
     counterfact.json, cases198.json, stats198.py, ee_import.py, run198.py, wsp.py by path) and the EasyEdit commit
     `5c519afb14e39c7f70bc5536ebd777a62d8493bf`.
   - Every subcommand: `--dry-run`, `--limit N` (for smoke runs), `--out-dir`.
2. `score198.py`: reads `out/*.jsonl` and `out/tstar.json`, applies the gate, S1, S2 and forecasts 198.1, 198.3, 198.4,
   198.6, 198.7 and 198.8 exactly as tabled (198.2 and 198.5 descriptive), prints each with attempted n, took k, already
   satisfied, name-eligible m, successes, an exact Clopper-Pearson 95% interval, and PASS / FAIL / NOT EVALUABLE; writes
   `out/score.json`. No verdict for anything the design marks descriptive or withdrawn.
3. `test_build198.py` (plain script, exits 1 on failure, no pytest): tiny synthetic tests of the took/already-satisfied
   logic, the random-control construction (same singular values, orthonormal factors, deterministic per seed), the
   mass gate and NOT EVALUABLE, score198's counting on hand-made JSONL, and resume skipping.
4. `BUILD-NOTES.md`: every choice the design left to the implementation, with the line of the design it serves, and
   anything in the design you found ambiguous or contradictory (do not resolve a contradiction silently: report it).

## Report contract

End with `report.md` in your run directory: SUMMARY (what exists, what passed), FINDINGS (design ambiguities,
anything you could not implement exactly), BLOCKERS. Run `fixtures198.py`, `test_build198.py` and each subcommand with
`--dry-run`, and paste their outputs.
