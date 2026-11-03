# OLMo 2 1B activation atlas: collector timing pilot (2026-11-03)

This is a **timing pilot, not a registered experiment.** It reads no scientific result. Its job is to build and time
the collector so the full atlas (one training checkpoint per cloud session) can be sized.

## Files

| file | what |
|---|---|
| `olmo_atlas_collect.py` | the collector (`collect`, `export`) |
| `test_olmo_atlas_collect.py` | pytest on a 2-layer slice of a tiny random OLMo 2, run as `python -m pytest -q test_olmo_atlas_collect.py`. Checks against brute force (max, top-k, frac_above, reservoir entries, controls), that kill+resume gives bit-identical state, refusal to resume under a changed config, and the export format. **4 passed.** |
| `pilot_runs.sh` | the 4-config timing driver (fp32/bf16 × batch 4/8, 40 docs), one process per config |
| `compare_runs.py` | top-k overlap between two state files |
| `pilot-data/timing-raw.json` | every run's raw timing, load time, peak RSS, and download times |
| `pilot-data/compare.json` | every overlap number quoted below |

## Environment

- CPU: Intel(R) Xeon(R) Processor @ 2.80GHz (family 6, model 85, stepping 7: Cascade Lake), 4 cores, 1 thread per core, KVM.
  It has AVX-512F/BW/VL/VNNI. It has **no avx512_bf16 and no AMX**.
- 16 GB RAM, no GPU, no swap.
- Python 3.11.18, torch 2.15.0+cpu, transformers 5.22.1, tokenizers 0.23.3, huggingface_hub 2.4.0, numpy 2.4.6.
- `torch.set_num_threads(4)` for every run.

## Collector design (what changed from atlas_collect_fast2.py)

- **Unit.** The unit is the input to `mlp.down_proj` (a forward pre-hook), which equals `act_fn(gate_proj(x)) * up_proj(x)`.
  The model has 16 layers × 8,192 = 131,072 neurons. Each layer also gets 256 controls: seeded random unit directions in
  the 8,192-space (`generate_controls`, the same seeding rule as the Qwen collector) projected the same way (`x @ d`).
  The forward stops after the last hooked layer's `down_proj`, so the final norm and the lm_head never run.
- **Same record format as atlas-full.** `neurons.jsonl` holds `{layer, unit, max, frac_above, top, random, weak, model_arm}`.
  `controls.jsonl` holds the same fields with `control_id` in place of `unit`. Each window is `{doc, pos, act, window}`
  with the 64-token window and the firing token marked `[[...]]` at offset 32. `decode_marked_window` is copied verbatim.
  The settings match atlas-full: TOP_K = 20, 10 random windows (> 10% of max), and 5 weak windows (in (10%, 30%] of max).
  The top list keeps one entry per document: that document's maximum.
- **Pass 1 (required).** Computes max and top-20.
- **Pass 2 (optional).** Computes frac_above, the random windows and the weak windows. It needs the final pass-1 max and
  can be limited to the first `--pass2-docs` documents. frac_above's denominator is the number of positions in the
  pass-2 documents.
- **Resumable.** State is saved every `--checkpoint-every` documents (default 64) as compact arrays in `state.npz`
  (int32 doc ordinal, int16 position, float32 activation per rank), plus `progress.json`. Writes are atomic, and the npz
  carries its own cursor, so a kill between the two writes cannot double-count a chunk. On restart the collector resumes
  from the last chunk. It refuses to resume if the config hash differs.
- **Deterministic.** Batches are length-sorted within each checkpoint chunk. Reservoir keys are seeded per
  (pass, chunk, batch, layer, kind). A resumed run is therefore bit-identical to an uninterrupted one (tested). Two
  separate fp32 bs4 runs of Instruct gave bit-identical max/top arrays.
- **Window text is built only at `export`.** Export re-tokenizes just the documents that appear in the state.
- **Differences from the Qwen collector.**
  - The reservoir selection rule is the same (the k smallest seeded random keys among qualifying positions), but it is
    vectorised in torch, so the RNG stream differs.
  - `doc` is the pile-10k file ordinal (the parquet has no `i` field).
  - There are 256 controls per layer, not 6,144.

## Corpus and tokenizer

- **Corpus.** NeelNanda/pile-10k, `data/train-00000-of-00001-4746b8785c874cc7.parquet` (dataset commit `127bfedc…`),
  read in file order. Each document is truncated to 512 tokens.
  - The full corpus is 10,000 docs and 3,593,567 tokens (359.4 tokens/doc). 4,165 docs (41.7%) hit the 512 cap, and
    none are empty.
  - **The 40 pilot docs are shorter** (12,822 tokens, 320.6/doc). The full-corpus estimates below therefore scale
    per-doc time by 359.4 / 320.6 = 1.12.
- **Tokenizer identity across the two checkpoints:**
  - `tokenizer.json` (sha256 `73fd5254624f39a8…`), `vocab.json` (`9e14712c…`) and `merges.txt` (`b6fe424e…`) are
    **byte-identical**.
  - `tokenizer_config.json` **differs**: Instruct adds a `chat_template`.
  - `special_tokens_map.json` **differs only in formatting**. Both give bos = eos = unk = `<|endoftext|>` and pad = `<|pad|>` (id 100277).
  - Token ids for **all 10,000 docs** are identical under the two tokenizers (sha of the id lists `72773c2a4f1d9a2d`
    both ways).
  - `add_special_tokens=True` adds **no BOS** with this tokenizer, so position 0 is the first text token.

## Downloads (this machine, via the session proxy)

| item | bytes | seconds |
|---|---|---|
| pile-10k parquet | ~33 MB | 4.5 |
| OLMo-2-0425-1B-Instruct @ `48d788eca847d4d7548f375ad03d3c9312f6139e` (single bf16 safetensors) | 2.97 GB | 30.8 |
| OLMo-2-0425-1B @ `stage1-step480000-tokens1007B` = `eda09f03465136b2b0a007da730ef720029c6f9c` (prefix `eda09f034651` ✓; 2 fp32 shards) | 5.94 GB | 297.0 (overlapped the first Instruct timing run) |

The Instruct weights are stored in bf16 and the stage-1 weights in fp32. Loading Instruct in fp32 is an exact upcast.

## Timing: pass 1 (max + top-20), 40 docs, 4 threads

Definitions:
- *fwd* = wall time minus time inside the hooks.
- *fwd+bk* = full wall time per document. It excludes model load and checkpoint saves, which add 0.3–0.8 s per save of
  57 MB.

| checkpoint | dtype | batch | s/doc fwd | s/doc fwd+bk | bookkeeping s/doc | peak RSS |
|---|---|---|---|---|---|---|
| Instruct | fp32 | 4 | **2.13** | **2.30** | 0.17 | 8.9 GB* |
| Instruct | fp32 | 8 | 2.59 | 2.81 | 0.22 | 8.9 GB* |
| Instruct | bf16 | 4 | 7.45 | 7.63 | 0.19 | 3.4 GB |
| Instruct | bf16 | 8 | 8.45 | 8.67 | 0.22 | 3.6 GB |
| stage1 1T | fp32 | 4 | **2.19** | **2.36** | 0.17 | 5.5 GB |
| stage1 1T | fp32 | 8 | 2.53 | 2.73 | 0.20 | 5.7 GB |
| stage1 1T | bf16 | 4 | 7.59 | 7.79 | 0.20 | 8.9 GB* |
| stage1 1T | bf16 | 8 | 8.97 | 9.21 | 0.25 | 8.9 GB* |

\* The 8.9 GB peaks are transient dtype conversion at load (bf16→fp32 for Instruct, fp32→bf16 for stage 1).
In steady state, fp32 runs use about 6 GB. Every configuration fits in 16 GB.

- **Cross-check.** A run with no bookkeeping at all (Instruct fp32 bs8, `--forward-only`) took 2.65 s/doc. The
  wall-minus-hooks figure in the same configuration was 2.59, so the two agree to within run-to-run noise (about 5–10%).
- **Batch size.** Batch 4 beats batch 8 in fp32. bs8 pads more inside the 20-doc length-sorted chunks, and the larger
  activation tensors cost more on 4 cores.
- **bf16 is 3.3× slower** on this CPU: it has no native bf16, so matmuls go through conversion.
- **Excluded runs.** Two Instruct runs overlapped other jobs and were re-run clean:
  - fp32 bs4 overlapped the stage-1 download: 3.03 s/doc.
  - bf16 bs8 overlapped a tokenizer check: 9.69 s/doc.
  
  The table uses the clean re-runs. Both raw results are in `timing-raw.json`.

### Pass 2 (frac_above + random/weak reservoirs)

Measured on Instruct, fp32, batch 8, the same 40 docs:
- 4.22 s/doc in total: forward 2.51 and **bookkeeping 1.71 s/doc**.
- The reservoir update draws a seeded random key for every (token, unit) pair, which is about 17 M per layer per batch.
- This could be cut a lot by drawing keys only for qualifying entries. It is left as is for now, because it is tested
  and correct.

## bf16 vs fp32 (same 40 docs, same batch size)

Top-10 sets per unit are compared as (doc, pos) pairs. All numbers are in `pilot-data/compare.json`.

| checkpoint, batch | median top-10 overlap | mean | 10th pct | identical top-1 | median rel. diff of max |
|---|---|---|---|---|---|
| Instruct, bs4, neurons | 1.0 | 0.981 | 0.9 | 97.9% | 0.43% |
| Instruct, bs8, neurons | 1.0 | 0.981 | 0.9 | 97.9% | 0.43% |
| stage1 1T, bs4, neurons | 1.0 | 0.984 | 0.9 | 98.2% | 0.41% |
| stage1 1T, bs8, neurons | 1.0 | 0.984 | 0.9 | 98.2% | 0.41% |
| Instruct, bs4, controls | 1.0 | 0.980 | 0.9 | 97.7% | 0.24% |

Every layer has a median overlap of 1.0.

**Reference points:**
- fp32 bs4 vs fp32 bs8: mean overlap 0.99985 and 99.99% identical top-1. The only differences come from batch/padding
  numerics.
- fp32 re-run: bit-identical.

**Verdict.** bf16 would be acceptable for top-10 sets: about 2% of units swap one near-tied entry. Two caveats:
- With only 40 candidate documents, the top 10 are a quarter of all candidates. At 2,500 docs, near-ties are denser and
  the bf16 overlap will be lower than measured here.
- bf16 is 3.3× slower on this CPU.

**So use fp32.** On these cloud CPUs bf16 brings no speed benefit, and it costs some agreement with fp32.

## Is a ~2,000-doc set enough? (ROUGH hint only)

The top-10 at 20 docs was compared with the top-10 at 40 docs (fp32 bs4):

| | median overlap | mean | identical top-1 |
|---|---|---|---|
| Instruct, neurons | 0.5 | 0.525 | 52.5% |
| stage1 1T, neurons | 0.5 | 0.522 | 52.0% |

**This is ROUGH and says almost nothing about 2,000 docs.** An overlap of 0.5 is what you would expect if each unit's
top documents were spread evenly over the corpus, so at 40 docs nothing has saturated, which is unsurprising. It neither
supports nor rules out 2,000 docs.

**Recommended direct measurement.** In the first real session, run with `--snapshot-every 500` and compare the top-10 at
500 / 1,000 / 1,500 / 2,000 / 2,500 docs with `compare_runs.py`. This adds no extra forward passes; each snapshot is 57 MB
on scratch disk.

## Estimates per checkpoint per 3-hour session (fp32, batch 4, 4 threads)

Per-doc time on the full corpus, scaled by the token ratio of 1.12:
- Pass 1: about **2.6 s/doc**.
- Pass 2: about **4.7 s/doc**.

Fixed overhead per session is about 15–25 min:
- pip install of CPU torch and transformers: a few minutes (not timed).
- Download: 0.5 min for a bf16 post-trained checkpoint, about 5 min for an fp32 stage checkpoint.
- Model load: about 10 s.
- Tokenization: about 4 s per 1,000 docs.
- Export: about 5 min (4.7 M windows at 41 µs per window, plus 1.7 GB of JSON writing; the 40-doc export took 31 s).
- Commit and upload, plus a margin.

That leaves about 9,000 s of compute.

| plan | docs/session |
|---|---|
| pass 1 only (max + top-20; no frac_above/random/weak) | ~3,400 (budget 3,000) |
| pass 1 and pass 2 on the same docs (7.3 s/doc) | ~1,200 |
| **recommended: pass 1 on 2,500 docs + pass 2 on the first 500** (6,500 s + 2,350 s) | **2,500** |

- Pass 1 resumes across sessions, so a checkpoint can be extended later. For example, one session could run pass 1 to
  3,500 docs and a second session could run pass 2 and export. Pass 2 cannot start until pass 1 is final, because it
  needs the final max.
- With checkpoint-every 64, a kill loses at most about 3 minutes of work.

## Output size per checkpoint

These sizes do not depend on the number of docs.

| artifact | size |
|---|---|
| compact `state.npz`, pass 1 + pass 2 arrays (16 × 8,448 units) | 57.0 MB (35.0 MB with `savez_compressed`) |
| compact arrays, pass 1 only | 28.7 MB |
| `neurons.jsonl` with window text (20 top + 10 random + 5 weak windows per unit) | 1.63 GB (487 MB gzip) |
| `controls.jsonl` | 49.9 MB (15.0 MB gzip) |

**Text export total: about 1.68 GB per checkpoint (about 0.5 GB gzipped).** For 12 checkpoints that is about 20 GB
raw / 6 GB gzipped, which is too large for git. Recommendation:
- Commit or keep only the compact `state.npz` per checkpoint (about 0.4–0.7 GB for 12).
- Run `export` wherever the windows are read. Export needs only the corpus parquet and the tokenizer, and it took 0.66 GB
  of RSS.

## Problems / notes

1. **bf16 is slow on CPU.** There is no avx512_bf16 or AMX.
2. **Pass-2 bookkeeping is the main avoidable cost** (1.71 s/doc). See above.
3. **Instruct ships bf16 weights; the stage checkpoints ship fp32.** Loading in the other dtype has a transient peak
   RSS of 8.9 GB. Steady state is about 6 GB in fp32.
4. **The tokenizer config files differ between the two repos** (chat_template, special-token map formatting). The
   tokenizer model files and all 10,000 token-id sequences are identical. The collector config records the model's
   `config.json` sha. The tokenizer source is whichever snapshot is passed. Use one fixed snapshot's tokenizer for export
   if exact identity must be enforced.
5. **Timing noise.** Two runs overlapped other jobs and were re-run (see above). Expect ±5–10% run-to-run.
6. **Pilot docs are shorter than the corpus average**, which is why the estimates are scaled by 1.12. Attention makes
   cost slightly superlinear in length, so the full-corpus rate could be a few percent higher.
7. **Nothing large was committed.** Raw pilot states (57 MB each) and the 1.7 GB export stayed on the session's scratch
   disk.
