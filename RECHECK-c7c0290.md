HOLD — the R5 regression has no intercept and no term for the reference layer 19, so layer 19's fixed effect is forced to zero and the R5 slope and "explained" fraction are not the registered estimator; fix that (and the R4 table label) before the Arc run.

# Part 165 reading — recheck of the fold at c7c0290

Scope: worktree /home/elisepc/green/worktrees/ab-165-reading, branch prime/165-reading, head c7c0290. Read-only. No Arc, no paid API, no full model load (host has ~5 GB free RAM for an 8.9 GB snapshot). Checks run: `python -I -B -m pytest -q part165/test_reading165.py` → 13 passed; tokenizer and chat-template checks on the local Qwen3.5-4B snapshot; meta-device class selection; a synthetic R5 regression; a bf16 rounding simulation for the site check.

## The 14 cold-read items (fixed at c7c0290?)

1. Population not 128 tokens — YES. `encode_documents` truncates and raises on any length ≠ 128 (reading165.py:167-182, called :458). Real tokenizer: full rows 128–336 tokens, truncated all 128; `test_frozen_texts_encode_to_128` passes.
2. R4 unreachable — YES. `READOUT_KEY = " Alibaba"` (:56), single token 52540 confirmed; dropped strings recorded (:357-364); `pass` is None when rho is None (:710). Residual: R4 has no CI — see N2.
3. R5 not the registered estimator — NO, partially fixed. The explained formula (:671-673) matches Amendment 1, and the fit now uses all records. But the design matrix (:659-662) is `[log_pr]` plus dummies for `elements[1:]` only: no constant and no dummy for layer 19. Layer 19's fixed effect is forced to 0. See N1.
4. Question states from raw question, wrong position — YES. `effect_question_state` (:186-199) renders `nr.build_prompt(..., False)` (:194), takes the greedy first token from a plain prefill (:197; effect165.py:106 is `do_sample=False, max_new_tokens=2`), appends it and reads the last position (:512). Prompt output verified: chat template with empty `<think></think>`, 17 tokens.
5. G5 invariant to capture site — YES. `capture_check` (:372-385, gate :494) compares hooked outputs with `hidden_states[L+1]`; `site_check` (:388-414, gate :503) checks that a down_proj push moves the block output by W·d. Both can fail. Thresholds are a risk — see N3.
6. G4 float32 noise floor — YES. `pca` is float64 on CPU (:202-212; `X` is a CPU tensor from `capture`). Non-negativity is checked before any clamp (:522, `eig_min >= -1e-8*max`), orthonormality at 1e-5 (:521-522).
7. Failed gate does not stop bars — YES. `bars` become `pass None, reason "gate failed"` and `correlations = {}` (:609-611); the table/stdout block is guarded (:728). Test `test_corrupted_norms_fail_gate` asserts it. Display defect — see N4.
8. Bootstrap clusters on slot only — YES. Groups are `"layer:slot"` (:621), 96 clusters for n+ (:632); `cluster_bootstrap` (:238-253) resamples those clusters. The docstring still says "two-way" (questions too); the code is one-way, which is what the design specifies.
9. Provenance and values — YES. Texts file only, no parquet fallback (:446); `corpus_sha256` and `rows` presence checked (:464-466; presence only, which matches Amendment 1 since the parquet is not read); G2 checks `isfinite` on KL and rel_change (:471-474). Verified: 6912 effect rows, 0 non-finite, 0 with KL ≤ 0.
10. Missing `1 +` in the readout covector — YES. `gamma = 1.0 + parent.norm.weight` (:354).
11. Not the house loader — YES in code. `nr.load_model(args.model, None, args.device)` at :448; the device_map path is gone. See N6 for what the house path actually does on this transformers version.
12. Tests do not cover the contract — YES. Real 128-token fixture test (:63), short-document refusal (:76), a failure test for directions (:220), effects (:234), contexts (:246), measures/pca (:257) and capture (:270); tiny-model end-to-end passes (:202). Gap: no test of the R5 estimator, nor of the R4 table label.
13. Null rows omit vocab cos — YES. `geometry(u, X, mu, pcs, readout, vocab_rows)` (:531).
14. Tuple vs tensor layer output — YES. `_layer_tensor` (:138-141) is used in `capture` (:158), `site_check` (:393, :409) and therefore `capture_check`.
15. R4 effective units — PARTIAL. `n_effective` is reported (:709) from the clustered count, 96 for n+. The clustered 95% interval Amendment 1 promises is `None`: `readout` is not in `PRIMARY` (:57) so `cluster_bootstrap` is skipped (:630). See N2. The design body (PART165-READING-DESIGN.md:143-146) still says 768 rows and γ⊙u_Qwen; Amendment 1 overrides it.
16. Launch line placeholder and prelaunch mismatch — YES. `part165/steps-reading.txt:6` uses `165`; the mismatch with `PART165-DESIGN.md` is recorded in the comment, not fixed in `tools/part_check.py` (out of scope here).

## New findings (from the fold, or missed by it)

**N1 — HOLD. R5 omits the reference-layer term.** reading165.py:659-663. The design is `[log_pr]` plus dummies for layers 20–26 and no constant. Layer 19's mean is pinned to the slope line. Synthetic check with the same construction: slope −0.366 as written against −0.401 for intercept plus dummies 20–26 (equivalently all eight dummies). R5 explained/pass is therefore computed from a misspecified fit. Fix: add a ones column (or dummies for all eight layers) and add a test with known coefficients.

**N2 — medium. R4 interval not computed.** reading165.py:57 (`PRIMARY` lacks `readout`) and :623-630 (`ci` is None for non-PRIMARY keys). Amendment 1 says R4 reports its clustered 95% CI; `r4["ci95"]` will be null. The bar's pass uses the point estimate, so no verdict changes. Fix: include `readout` in the bootstrapped keys.

**N3 — medium. Site-check threshold sits on the bf16 rounding floor.** reading165.py:503 (`site <= 5e-3`). I simulated the block-output delta in bf16 with ‖W·d‖ = 0.8·‖h‖ (the dose is 0.8 × layer median residual norm, from results/part112/README.md): mean relative error 0.0036–0.0040, max over 200 draws 0.0038–0.0043; with a few heavy residual dims the max reached 0.0054. The push also casts d to bf16 (:403, `d.to(value.dtype)`), which adds roughly 0.001 not in the simulation. A gate that can fail on rounding alone would set `capture` False and skip every bar. It is not verified on the real model (RAM). Fix: measure the floor on the Arc with a no-change run, or set the threshold from it (about 1e-2 is safe for bf16).

**N4 — low/medium. "pass = None" printed as "fail".** reading165.py:732 and :757 use `'**pass**' if bar['pass'] else 'fail'`, so an unmeasured R4 prints "fail" in reading-tables.md and stdout. reading.json is correct. Amendment 1 says this must never be a failure with nothing measured. Fix: a three-way label.

**N5 — low. Memory on Windows.** The float64 covariance is not retained eight times: each `pca` call frees its covariance and eigenvectors, and only float32 `pcs` (16×2560) survives. Persistent host RAM beyond the model is about 1.1 GB: pinv cache 8 × 94 MB ≈ 755 MB (:86-88, :497), natural states ≈ 168 MB, `document_states` ≈ 168 MB (:486, held for all of `main`). Transient per layer is about 0.3 GB for float64 PCA plus pinv workspace. The pinv cache is only needed for the current layer, so clearing it per layer saves ~660 MB. The 2 GB guard itself lives in job.py, not in this worktree, so I could not check it.

**N6 — low (documentation). The loader branch.** On transformers 5.20, `AutoModelForCausalLM.from_config` succeeds on the real config (Qwen3_5ForCausalLM), so `nr.load_model`'s ImageTextToText fallback is never reached here. The docstring at reading165.py:116-122 and PART165-READING-DESIGN.md:51-54 ("the Arc loads it with the image-text class") are wrong for this version. Decoder discovery is correct in either form (32 layers, parent Qwen3_5TextModel with `norm` and `embed_tokens`). Weights: transformers' key rules map the checkpoint's `model.language_model.*` into the CausalLM (426 of 427 state keys; the one unmapped key is `lm_head.weight`, which is tied, `tie_word_embeddings: true`). The Arc effect answers are coherent. The Arc's own transformers version is still unverified.

**N7 — low. Capture check uses exact-equality threshold.** reading165.py:494 (`worst <= 1e-6`). Any non-bit-identical kernel between the two forward passes on XPU (bf16 ulp scale) would fail it. Verified only on the tiny fixture.

**N8 — low. Docstring.** `cluster_bootstrap` (:238-242) calls itself two-way; it resamples only (layer, slot) clusters.

## --copy list

Complete. reading165.py imports numpy, torch, stdlib and neuron_report. neuron_report.py top-level imports are stdlib, numpy and torch; its lazy imports (transformers in `load_model`, judge_client and AutoTokenizer in unrelated functions) are not reached. The steps file copies reading165.py, both JSON inputs, neuron_report.py, norms.json, effects.jsonl and medians.json; all relative paths the job uses resolve.

## Gates that could never pass or never fail

- G1 directions: can pass (rebuilt vs recorded norms agree to ~2e-7 on n+ slot 0); can fail.
- G2 effects: 6912 rows keyed exactly as the reader expects; can pass; can fail on non-finite or missing rows.
- G3 contexts: 16 documents, provenance present, no question overlap (checked); can pass; can fail.
- G4 measures: READOUT_KEY is single-token; the eigen and orthonormality thresholds sit above the float64 and float32 round-off floors; can pass and can fail.
- G5 capture: can fail (see N3, N7). The site threshold is the main risk.

## Checked and clean

Direction builder matches run165 (host test). Frozen questions are eight maker questions with no overlap. Corpus SHA present. Effect KL all positive and finite. Within-arm bootstrap groups are (layer, slot). Null rows carry the vocab measure.

## Not verified

Full model run on CPU or XPU (RAM). Windows memory peak and the 2 GB guard (job.py not in this worktree). The Arc's transformers version. Exact bit-equality of the capture check on XPU. Real-model values of the site-check error.

## Verdict

HOLD. N1 alone would make R5's numbers and its pass come from a misspecified regression. Before launch: fix N1 (plus a test), make N3's threshold a measured value, and fix N4's label. N2 and N5 are cheap fixes that can go in the same pass.
