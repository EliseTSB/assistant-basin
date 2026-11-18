---
title: Weight-edit detection — prior art (before design)
description: "Prior-art search for detecting and localizing small targeted weight edits, with our own benign edits as ground truth; defensive scope per Elise 2026-11-07 23:36."
authored-by: claude (search by a ds-reader subagent on DeepSeek V4.1 Flash, finished about 23:42 CT 2026-11-07; two entries checked at source by me)
---

# Prior art, 2026-11-07

Card `weight-edit-detection-2026-11-07`. Elise [cite: elise 1568861686167904287 "Methods to detect weight edits would be particularly valuable"].
Scope: detection and localization only, on our own benign edits (chicken rank-one edit, Part 190's output edit, W1's
repurposed neuron). **V** = title checked at arXiv by me; **F** = fetched by the subagent; **S** = snippet only.

## Closest work

1. **V** Youssef, Zhao, Seifert, Schlötterer, *Tracing and Reversing Edits in LLMs*, ICLR 2026, arXiv:2505.20819. Infers
   the edited object from weights alone and reverses ROME edits; localizes edited layers by pairwise row-vector cosine
   similarity of each MLP projection compared with the unedited matrices. Reference-based for localization. Its own
   caveat: a large signal on GPT-2/GPT-J, none on Llama 3.
2. **V** Hurtado, *Has This Checkpoint Been Abliterated? A Two-Signal Audit and Its Failure Map*, 2026, arXiv:2607.01854.
   Rank-1 energy fraction of the base-to-candidate weight difference (sigma_1^2 / sum sigma_i^2 of delta W), averaged
   over mid-stack o_proj/down_proj, plus an activation signal. Reference-based. **The closest primitive to ours**; it
   band-averages rather than ranking single matrices, and it over-flags benign low-rank fine-tunes.
3. **F** Youssef et al., *Has this Fact been Edited? (DEED)*, NAACL 2025: classifies a retrieved fact as edited from hidden
   states and output probabilities; reference-free but per fact, not per matrix.
4. **F** Puertolas Merenciano et al., *Weight space Detection of Backdoors in LoRA Adapters*, arXiv:2602.15195: spectral
   statistics per attention projection of the low-rank update; reference bank needed.
5. **S** Pitzalis et al., *Z-PEFT*, arXiv:2608.02271: per-head spectral meta-classifier over adapter weights; collapses
   below chance on a held-out adapter family.
6. **F** Goel et al., *Learning to Interpret Weight Differences* (DIT), arXiv:2510.05092: the model answers questions about
   its own weight diff; reference-based.
7. **S** Kabra et al., *Weight Oracles*, arXiv:2610.07334: an LM reads raw weights; toy scale only.

## What transfers

- Rank-1 energy computed PER MATRIX and ranked (a one-line change from the abliteration audit) gives localization when
  the base is available. Needs a null: the same statistic on matrices nobody edited, and on benign fine-tunes.
- Row collinearity (Tracing and Reversing) is computable without the base, the nearest thing to reference-free
  localization, but its signal is architecture-dependent and needs a per-matrix null from unedited checkpoints.
- Spectral features at per-projection granularity are the right unit for "which matrix".

## Gaps (subagent's, with one correction)

- No calibrated localization of a single rank-one edit to one matrix with a per-matrix null; no reference-free
  "which matrix" method.
- Nothing on hybrid linear-attention models (Qwen3.5) or on audio encoders.
- Single repurposed neurons in LLMs: no detection work found. An aggregate weight statistic is unlikely to see them.
- ⚠ The subagent called EmbeddingGemma 2 "essentially one big embedding table". That is wrong: it is a multi-layer
  encoder with an audio tower, and our chicken edit changed one MLP output matrix inside it. The gap it names (no work
  on these models) still stands.

## Traps

1. Benign low-rank fine-tuning is the main false positive; safety fine-tuning is itself a low-rank MLP edit.
2. The ROME row-collinearity artefact does not carry to every architecture.
3. Meta-classifiers learn edit-family artefacts and fail on a held-out family.
4. A spoofed reference makes delta W zero by construction: pin the base by hash.
5. Compare against run-to-run spread as well as the base.

## Addendum 2026-11-08 10:1x CT (after Part 193 published), for Part 198

- Muse's sweep (council `muse-prior-art` #14) is folded into Part 193's README as related work (public 481a03b).
- `llm-forensics` 0.1.1 (PyPI, uploaded 2026-11-06 07:14 and 08:10 UTC; BSD-3-Clause; repo
  github.com/bohemian-miser/LLM_Forensics, last push 2026-11-07, 0 stars). Its repo description credits
  rikiyat.github.io/blog/weight-space-programming/, the same thread as card `weight-space-programming-2026-11-03`.
  Deps: torch, transformers, safetensors, accelerate, datasets (LLM extras optional). Muse found no repo; PyPI lists one.
  Treat as an untrusted download: install in its own venv, run with `-I`, never from inside its folder.
- EasyEdit (zjunlp, MIT, 2.9k stars, pushed 2026-11-07) implements ROME, MEMIT, AlphaEdit, EMMET and others on GPT-2,
  GPT-J and Llama, with CounterFact-style benign fact edits: the ground-truth generator for Part 198.
- Part 198 questions (Muse's traps): (1) F1/F2 false-positive rate on several unedited models; (2) edit-norm to sigma1
  ratio and detection for ROME, MEMIT and AlphaEdit on benign facts (does ROME's 15x hold outside our chicken edit; do
  AlphaEdit and MEMIT hide as Youssef App. E suggests); (3) llm-forensics' spectral detector on our three edits and on (2).

## Added 2026-11-08 15:01 CT (Part 198 redesign: reading weight directions as tokens)

- Dar et al., *Analyzing Transformers in Embedding Space* (arXiv:2209.02535): projecting parameters through the
  unembedding. The method Part 198's token reading applies to singular directions.
- `llm-forensics` 0.1.1, read from the wheel's source (not run): `arch.direction_logits` reads a residual direction
  through the final norm and the unembedding; its investigator agent exposes this as `t_unembed_vector` on a matrix's
  singular directions. `toolkit/spectra.py` scores the participation ratio of each top singular vector ("a ROME/MEMIT
  value vector is diffuse") and notes the edit sat in s2 or s3 in its analyst sessions. Neither is a scored detector
  with a false-alarm rate.
- Youssef et al. (arXiv:2505.20819), re-read 2026-11-08: they infer the edited object with a decoder trained on 600
  labelled edited matrices (99.4% ROME on GPT-2 XL, 1.9% AlphaEdit), not by reading directions; their cosine-similarity
  layer signal is computed relative to the original matrix.
- Two web searches (2026-11-08 14:39 CT) found no paper scoring an untrained unembedding reading of singular directions
  as an edit detector.
