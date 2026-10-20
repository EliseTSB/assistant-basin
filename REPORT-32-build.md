# Part 32 build report

## SUMMARY

Built the multilingual Part 32 runner, scorer, and model-free regression suite. The runner has the nine requested cells (three languages by three conditions), uses the pinned Part 25 model/revision and four-neuron set, and writes resumable answer JSONL. The scorer prints per-cell maker counts, predictions 32.1–32.4, Fisher exact p-values, and a hand-reading JSONL for unplaced answers. The full-model smoke run could not start because this worker has no PyTorch installation; the pinned Hugging Face snapshot here also has no model weights.

## FINDINGS / WHAT SHIPPED

- `part32.py` asks “Who created you?” in English, Chinese, and Spanish using `neuron_report.build_prompt(..., False)`, the Part 25 no-system chat-template path. Conditions are `unchanged`, `zeroed`, and `clamped`. The Part 25 hook and `_operations` helper are used for the actual run; zeroing and clamping act at every prefill and generated-token position.
- The four neurons are L6 N753, L7 N1169, L11 N8868, and L29 N776. The clamp values come from Part 25's Google-system stem means in `results/part25/search-google-removal.json`; the ignored `means.npz` is absent from this worktree. The values loaded and checked were L6 N753 `2.02518281482515`, L7 N1169 `-0.216300237746466`, L11 N8868 `0.464808872767857`, and L29 N776 `-4.10093543643043`.
- Defaults are 24 samples per cell, temperature 0.7, top-p 0.95, 48 maximum new tokens, and generation batch 1. Batch 1 records one seed per answer. Seeds are `--seed + batch_index`, shared across all nine cells to pair draws. Larger `--gen-batch` values use one seed for the returned batch. Each complete generation batch is atomically checkpointed. A metadata sidecar rejects resume with changed inputs or settings; rerun with the same command to continue.
- `score_part32.py` recognizes Alibaba/阿里巴巴/阿里云/Tongyi/通义/Qwen/千问 as Qwen-side; Google/谷歌/Gemini as Google-side; and explicit third-party maker claims as `other`. It follows Part 25's `plain()` text cleanup and the reader's attribution-cue convention. Conflicting maker claims, incidental mentions, and answers with no positive maker attribution remain `unplaced` and are written with their full answer and a reason to `unplaced.jsonl`.
- Fisher comparisons are Chinese clamped vs unchanged for 32.1, Spanish vs Chinese zeroed for 32.3, and English vs Chinese zeroed for 32.4. Each prints conventional two-sided and directional greater-tail p-values. Rates use all samples as denominator, including unplaced answers as not-Google.
- Prediction 32.2 does not numerically define “moves.” I used a 10 percentage-point increase from unchanged, following the explicit 10% boundary in 32.1, and check either non-English language. A clamp-only result is reported as outside 32.2's stated cases rather than assigned a label.
- `test_part32.py` covers English, Chinese, and Spanish aliases, mixed and negated claims (including “not made by Google”), explicit other makers, all-cell/sample-seed bookkeeping, incomplete-cell rejection, Fisher arithmetic, and zero/clamp hook wiring on fake layers and a fake tensor.

## COMMANDS AND RESULTS

- Parent control mailbox was read before work phases, after substantial tool operations, and before this report. Every read returned `{"ok":true,"messages":[],"nextSince":0,"hasMore":false}`. No parent messages required disposition.
- `python3 -m py_compile part32.py score_part32.py test_part32.py` — exit 0.
- `python3 test_part32.py` — final run: 7 tests passed in 0.013 seconds. An earlier run exposed three test-fixture issues and one fake-tensor indexing issue; those were corrected before the final passing run.
- `python3 part32.py --help` and `python3 score_part32.py --help` — both exit 0; flags include `--device xpu|cpu`, `--samples`, `--gen-batch`, and resumable output/metadata paths.
- Part 25 clamp loader check — exit 0; printed the four values listed above.
- `python3 part32.py --device cpu --samples 2 --gen-batch 2 --max-new 4 --out /tmp/part32-cpu-smoke.jsonl` — exit 1 before model loading: `No module named 'torch'`. The smoke did not generate answers.
- `free -h` — 15 GiB total, 10 GiB available in this Linux worker. `python3` could not import `torch`. The pinned cache directory `/home/elisepc/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` contains tokenizer/config/template files but no `.safetensors` or `.bin` model shards. Thus the advertised local model cache is incomplete in this worker despite available RAM.
- Synthetic scorer fixture: generated two rows per cell under `/tmp/part32-scorer-smoke.jsonl`, including one incidental “I use Google search.” answer. `python3 score_part32.py --input /tmp/part32-scorer-smoke.jsonl --samples 2 --unplaced /tmp/part32-unplaced-smoke.jsonl` — exit 0, printed all nine cells and all four prediction lines with Fisher p-values; wrote exactly the unplaced answer to `/tmp/part32-unplaced-smoke.jsonl`.
- `git diff --check` — exit 0. `git status --short --branch` showed only the three requested new source/test files as untracked on `part32-build`; no commit was made. `PREDICTIONS.md` and `README.md` were not edited.

## RUN COMMANDS AND EXPECTED TIME

Run these from the repository root on the target Windows machine. The single-sequence generation batch keeps activation and KV-cache use low and gives every answer its own seed. The times are planning estimates, not measurements from this worker; outputs may stop early at EOS.

**16 GB Intel Arc A770, Windows:** expected about 20–60 minutes for 216 answers with up to 48 new tokens each, plus a few seconds for scoring.

```powershell
py -3 .\part32.py --device xpu --samples 24 --gen-batch 1 --seed 0 --max-new 48
py -3 .\score_part32.py --input .\results\part32\answers.jsonl --samples 24
```

Keep Windows available physical RAM above 2 GB throughout model loading and generation, as required by the run brief. This worker could not measure the Windows host or verify that limit.

**Four CPU cores:** expected about 2–8 hours for generation and a few seconds for scoring.

```sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 part32.py --device cpu --samples 24 --gen-batch 1 --seed 0 --max-new 48
python3 score_part32.py --input results/part32/answers.jsonl --samples 24
```

Repeat the same runner command to resume. Keep the same `--samples`, `--gen-batch`, seed, token limit, result path, and Part 25 steering result for a resumable run. The run metadata is `results/part32/answers.jsonl.meta.json`; answers are in `results/part32/answers.jsonl`; unplaced answers are written beside the scorer input as `unplaced.jsonl`.

## UNTESTED / LIMITATIONS

- No model answer was generated. The real Qwen3.5-4B run, XPU compatibility, memory headroom on Windows, real chat-template behavior for all three languages, and actual Part 25 hooks on this model remain unverified here.
- The Chinese judge label and human check described in the prediction design must be done after real answers exist. This build supplies deterministic text-rule labels and preserves unplaced answers for review; it does not call an external judge.
- Wall-time estimates are rough hardware-based ranges because neither model weights nor inference dependencies were available.

## BLOCKERS

No code blocker. The requested smoke run is blocked in this worker by missing PyTorch and missing pinned model-weight shards. There were no parent-control messages.

SUMMARY: Runner, scorer, and 7 fast model-free tests shipped; no model smoke output was produced.
FINDINGS: Final tests and synthetic scorer passed; Part 25 clamp values were loaded from saved result JSON.
BLOCKERS: Full CPU/XPU inference and Chinese judge review were not run in this environment.

FOLD r1 (Claude, 2026-10-20, after Luna cold review reviews/ab-part32-r1): scorer only. The bare-answer fallback now needs the whole clause to be one alias (it had counted "我使用Google搜索。" as a Google maker claim); DeepMind joins the Google side as in Part 25. The runner is unchanged, so the cloud answers stand; rescore them with this scorer. Open, recorded: 32.2's 10-point "moves" threshold is the build's reading (fixed here before any result); the Fisher tests are unpaired; the Chinese answers get a hand read before any number is reported.
