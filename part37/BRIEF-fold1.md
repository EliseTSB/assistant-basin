# Part 37 fold 1 brief (round-1 reviews)

Implement `PART37-DESIGN.md` Amendment 2 exactly, and the review fixes below. Read both reviews first:
`review/part37/cheap-p37-build-haiku-r1.md` and `review/part37/cheap-p37-build-ds-r1.md` (DeepSeek's measurement scripts
are in `/home/elisepc/green/Journal/scratch/review-p37-build-ds-r1-2026-10-25/scripts/`). Write only under `part37/`.
Do not run anything on a GPU. Generation is already running on the Windows box with the current `generate` code, so do
not change generation's behaviour or its output format (fixing the write-at-end, item 9, must keep the same rows).

Every item needs a test that FAILS on the current code (`git show 92c2f6e:part37/...`) and passes after. Say in the
report which test covers which item, and that you checked it fails on the old code.

1. Out-of-fold projections (Amendment 2 item 1, Haiku F1/F2): one helper that, for a contrast, fits the direction on
   each fold's training questions and projects that fold's held-out records of every kind (including records not in
   the contrast, such as own_s, nulls, edit and label frames, projected with the fold that holds out their question).
   37.3, 37.4, 37.9, the null, edit and label projections all read from it. Test: pure Gaussian noise features (width
   at least 512, 100 questions) give a 37.9 shift under 0.5 SD and a 37.4 Spearman under 0.3, and the old code fails
   the 37.9 part.
2. 37.15 on null_smol and null_pile only; null_shuffle reported separately (item 2). Test with per-kind means 0.02,
   0.03, 0.64: FAIL, not PASS.
3. Pilot gate (item 3): within-author contrasts against their own label-shuffled null (95th percentile of the max over
   layers, the same permutation scheme as the authorship null); a missing layer, missing null or non-finite value
   raises (exit 2). Test: on pure noise at pilot size (20 questions, three seeds), the gate says STOP in at least two of
   three seeds; and a fixture with a strong planted authorship signal says CONTINUE.
4. 37.9 only on questions with both own_s1 and own_s2 non-empty; per-cell n and dropped counts in the output (item 4,
   Haiku F7).
5. 37.11 signed as registered (item 5, DeepSeek H2). Test: a fixture where the label moves authorship more than memory
   must FAIL.
6. 37.13 requires both role cells (item 6, Haiku F5). 37.8 cosine and AUROC at the regressed best layer (item 7).
7. Scoring tests (DeepSeek H3, Haiku F12): for every scored prediction (37.1, 37.2, 37.3, 37.4, 37.8, 37.9, 37.11,
   37.12, 37.13, 37.14, 37.15), one PASS fixture and one FAIL fixture through `_prediction_results`.
8. Minimal edit (item 8, DeepSeek H1): a fixed stopword list and a fixed list of common nouns; test on DeepSeek's eight
   sampled answers (reconstruct them from its scripts or from the fixture) that no replacement is a stopword.
9. Precision and robustness: float32 hidden-state storage with `np.isfinite` per record (fail the record with its id);
   span log-probs from float32 logits (item 9, Haiku F8, F11); drop `penalty="l2"` (F9); count failed fits in the
   label-shuffle null and raise if any layer fails in more than 2 of 200 permutations (F10); write capture shards and
   the generation file atomically or per question so a crash leaves a loadable prefix (F13), keeping generation's
   row format identical.
10. DeepSeek's MEDIUM items: the pilot gate's fail-open on a missing null layer (covered by 3); label the length-only
   control in the output as a sanity check that cannot fail when spans are length-matched; leave the user-slot frame and
   SmolLM3's system prompt as they are, but state both in the analysis output's notes.

Run the whole suite on the host with `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q part37/test_part37.py`
(it has torch, transformers and scikit-learn). Write `part37/REPORT-fold1.md` with SUMMARY, FINDINGS (item by item, with
the covering test and the old-code failure), BLOCKERS. Commit in the worktree if git lets you; otherwise leave the files
and say so.
