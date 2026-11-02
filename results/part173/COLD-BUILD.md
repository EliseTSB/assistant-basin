# Cold read: Part 173 build (worktree ab-part67)

Read of PART173-DESIGN.md, part173/{run173,label173,score173,names173}.py, known173.json, prompts173.py, test_part173.py, steps.txt, and the imports they pull from part171/part172/part86/part165/part166. Nothing in the worktree was edited. No model was loaded and no API was called.

**Verdict.** As written it would not get through a correct run and its verdicts would not yet mean what the design says. The most likely early stop is Gate 8 (run173.py:401): the cosine bar of 0.02 sits inside the noise Part 171 measured for its own random-coordinate controls (0.016, 0.024, 0.040 in results/part171/per_target.json), so a correct build probably fails there before any answer is generated. If it got past that, the 173.3 verdict is computed one short (score173.py:448-450 reports distinct-1, so the bar effectively needs four entities), Qwen-family and possessive name variants become "replacements" (names173.py), and Gate 1 can discard a finished ladder over one empty reply (run173.py:143-145). What checks out on read and by running it: the u100 builder mechanics (seeds, streams, sqrt(k) energy, one global scale, the same W as Part 171), the n2-n5 single-neuron construction (shared with Part 172), the replay prompt ids, the NEW44B disjointness, the 6,160 + 132 answer count, the 24 existing tests, and Gate 6's catch of a dropped dose. None of this has been run on the real model or on sets_google.pt, which is not in this worktree.

## Findings

Severity: high = would stop the run or flip a verdict; medium = wrong result under plausible conditions; minor = cleanup or limit.

1. **[high] Gate 8 cosine bar will likely stop a correct build.** run173.py:401 (`abs(cosine) <= 0.02`), with the cosine computed at run173.py:364. Part 171 measured cos with the fit vector for its own random-coordinate sets: rand100_d171 0.0159, rand100_d1171 0.0243, rand100_d2171 0.0399 (results/part171/per_target.json, `size.cosine`). Under a Gaussian approximation with the observed rms (about 0.029), each set passes with probability about 0.5 and all six with about 0.02; even at the most favourable draw level (0.016) all six pass with about 0.24. This is an estimate from three draws, not a measurement of the u100 sets. Failure: a correct build exits with "Gate 8 failed" before any planned answer. Fix (author decision, before registration): replace the fixed 0.02 with a bar scaled to the noise, for example |cos| <= 3/sqrt(d) where d is the hidden size from the model config, and state it in the design. Do not replace draws. Verified mechanics: a synthetic 32-dimensional build gave all six sets failing the cosine check with norms exactly equal to n1's, and an independent recomputation matched the builder's norm and cosine.

2. **[high] 173.3 distinct-entity count is off by one, and the drift guard it cites does not exist.** score173.py:448-450 computes `distinct_best` as the number of transitions in the sorted best-entity array, which is distinct minus one. Verified by running derive_bootstrap_estimates on a synthetic grid: three distinct best cells reports 2, four reports 3. verdict_173_3 (score173.py:554-561) needs at least 3, so the verdict needs four entities. Fix:
   ```
   present = sorted_best != np.iinfo(np.int32).max
   distinct_best = present[:, 0].astype(int) + np.sum(
       (sorted_best[:, 1:] != sorted_best[:, :-1]) & present[:, 1:], axis=1)
   ```
   Add a test with three distinct best cells. The comment at score173.py:880 says "the equality below catches drift", but no equality check follows. `_point_summary` (score173.py:739-791) computes `len(set(best_entities))`, which gives 3 in the same case, so an assert comparing the two would have caught this. Either add that assert or delete the dead code (finding 13).

3. **[medium] Qwen-family variants and possessive names fall outside the table, and invented modals count as replacements.** names173.py:8-21 (table), 25-26, 50-79. Verified outputs: "Qwen Team" -> `qwen team`, "Alibaba Qwen Team" -> `alibaba qwen team`, "Qwen (Alibaba Cloud)" -> `qwen alibaba cloud`, "Tongyi Qianwen", "Qwen 3.5" -> `qwen 3 5`, "Qwen3.5-4B" -> `qwen3 5 4b`; none is Qwen-family, so they go to `non_qwen_primary_entities` (score173.py:201-202) and `_modal` (score173.py:143-150) excludes only the exact key `qwen`. A cell where at least 10 of 44 answers say "Qwen Team" becomes a supported replacement by an invented name, and that counts in 173.1, 173.2 and 173.8. Real makers are also split: "Google's Gemini team" -> `google s gemini team`, "Google DeepMind team" -> `google deepmind team`, "Alibaba's Qwen" -> `alibaba s qwen`, "Tencent's Hunyuan" -> `tencent s hunyuan`. That can deflate 173.4 and 173.5. Fix (author decision, before any answer exists): (a) strip possessive `'s` in `_plain_words`; (b) match any whole-word Qwen-family token (qwen, tongyi, alibaba, 通义, 阿里) to `qwen`, and any whole-word alias followed by "team" to its entity; (c) count a modal toward `eligible` only if it is in KNOWN_ENTITIES, and report invented-modal cells apart, as the design already says invented names must be. The design text is silent on whether invented modals count, so this needs the author.

4. **[medium] The Gate 6 omitted-dose probe cannot fail.** run173.py:605 (`omitted = observed("n1", 1.0)`) is compared with `references["n1"]`, which is the same call, so its slope is 1.0 by construction; gate6 (run173.py:630-639) then tests whether 1.0 differs from 0.5. Verified with a tiny-model harness through the real measure_gate6 and gate6: the probe reports `omitted_dose_caught=True` for a correct build, a dose-dropped build and a sign-dropped build. Dropping the dose is still caught by the signed-dose rows (115 failures when the dose is dropped, 46 with the sign dropped), so Gate 6 is not defeated. Fix: build the omitted-dose row through the same `float(spec.get("dose", 1.0))` path as run_spec, feed it to gate6 as a delivery row at dose 0.5, and require that the check fails on it.

5. **[medium] Resume keys on arm and prompt only, and the u100 artifact is rebuilt every run.** run173.py:94-96 (answer_key delegates to run172, key = arm, prompt_id); run173.py:650-652 (done set, no provenance); run173.py:704-707 (build_u100_sets always runs and `_save_artifact` overwrites sets_u100.pt). A changed builder, model revision, max_new_tokens or NEW44B order is silently mixed with old rows. Fix: store a run key (sha of sets artifact, model revision, max_new_tokens, DESIGN and NEW44B hashes) in stage0.json and in each row; refuse to append when the stored key differs; keep the old answers file under a versioned name.

6. **[medium] Reader errors are never retried and are frozen into the second-reader sample.** label173.py:298-304 (pending excludes any id that already has a row, error rows included), label173.py:104-129 (transport errors become permanent error rows after two attempts), label173.py:316-317 (sample frozen after the DS pass whatever its errors), and label173.py:259-270. Errored pairs are dropped silently from modal counts: summarize_cell skips records whose relation is not claim or none (score173.py:177-178), and the reader_failures block reports them but no gate caps them. Fix: pending = items with no row or with an `error` row; freeze the sample only when the errored count is zero or below a registered cap; on resume, refuse to refreeze an errored file.

7. **[medium] The known-example gate is cached by file existence, not by reader prompt or model.** label173.py:281-289. A known-ds.json with 20 rows is reused whatever prompt or model produced it, and a file with passed=false is never recomputed, so only a hand deletion clears it. Fix: store a sha of READER_PROMPT, the model name and known173.json in the file, and recompute when they differ.

8. **[medium] Gate 9 can pass on a partial Haiku pass, and second-reader errors are dropped from agreement.** score173.py:801-833 does not require that every sample id has a valid second-reader row. label173.py:353-367 drops second-reader nulls and errors from `comparable` (line 363-364), so the 80% agreement is computed on whatever subset completed. Fix: require complete second-reader coverage of the sample, or count errors and nulls as non-agreement, and report which.

9. **[medium] Gate 1 rejects empty answers.** run173.py:143-145 requires `bool(row["answer"].strip())`. The design's Gate 1 is "every planned answer present, keys unique". An arm whose first token is end-of-sequence yields an empty decoded answer and fails Gate 1 only after the whole ladder, which stops every reading. Fix: keep the non-empty token_ids check and drop the non-empty text check; score empty answers through the normal clean/broken path.

10. **[minor] Digests of torch tensors are not full digests.** run172.py:652-654 (`_json_hash` uses `default=str`). str() of a torch tensor over 1000 elements elides the middle. Verified: changing element 4000 of a 9216-element tensor leaves the hash unchanged. It affects the pushes digest in artifact_hashes (run172.py:657+), `_source_hash` (run173.py:437-441), and `source_sha256` (run173.py:705). Part 171's arms are tensors after tensorize, so this is the path the run takes. Fix: hash `value.tolist()` or the bytes from torch.save.

11. **[minor] Gate 2 and Gate 5 run after all 6,160 answers.** run173.py:762-765. They read only the `none` and `role` rows, which are the first two specs. Moving them after spec two would save the ladder if they fail. Part 172's margins (qwen 44/44, role 43/44) suggest they will pass.

12. **[minor] per_set_maxima keeps only the last dose.** score173.py:692-693 is a dict comprehension that overwrites each set's key per cell, so it reports the d = -1 cell, not the maximum. Reporting only; the 173.6 verdict uses the bootstrap maxima correctly. Fix: `max(...)` per set.

13. **[minor] Dead or broken code.** score173.py:739-791 (`_point_summary`, unused, and the source of the drift comparison in finding 2), score173.py:898-899 (`_all_prediction_estimates`, unused), score173.py:902-903 (`_prediction_rows_from_boot` reads a nonexistent `boot["point_values"]`), run173.py:90-91 (SETS_FILE and ANSWER_FILE unused), run173.py:38 (`model_revision` is never asserted; steps.txt passes the snapshot path, which carries the hash).

14. **[minor] run_spec defaults a missing dose to 1.0.** run173.py:474. Not silent in the answer rows, because `_record` (run173.py:491-492) would raise KeyError after the batch has been generated. Fix: `float(spec["dose"])`.

15. **[minor] NumPy Generator streams may change between releases.** run173.py:238-240 and 245-247 use Generator.choice and standard_normal. NumPy's compatibility policy for Generator methods is weaker than for BitGenerator streams (NEP 19), so the registered draws are reproducible only on one NumPy version. Fix: record numpy.__version__ in stage0.json and pin it on the Arc.

16. **[minor] Legal-form stripping removes every trailing legal form.** names173.py:41-42 loops while the last token is a legal form; the design says "a legal form". Harmless for the aliases tested ("Google Co Inc" -> google). Fix: strip one.

17. **[minor] A reader answer of relation none with a primary is an error.** label173.py:91-92 raises, so the pair is retried and then becomes a permanent error row (finding 6). Consider accepting the primary as null.

18. **[minor] Gate 6 checks linearity of each set against its own d = 1 write.** run173.py:591-599. It cannot detect a set built in the wrong direction. Gate 8 covers the random-valued sets; the picked and fit-built sets rely on Part 171's own checks. Stated here as a limit, not a fix.

19. **[minor] Launch lines hard-code paths.** steps.txt lines 6-10 use a WSL venv path and a /mnt/c tar path; line 5 uses the Windows launcher. Filed as a papercut.

## What I ran

- `-m pytest -q part173/test_part173.py` with `-I -B -p no:cacheprovider`: 24 passed.
- Static AST import closure from run173, label173, score173, names173, prompts173 and the test file, compared against the steps.txt copy list: every resolved module is listed; `importlib`, `platform`, `secrets`, `statistics`, `transformers` and `urllib` are stdlib or venv packages. All copy-list sources exist in the worktree.
- Gate 7 on NEW44B: passes, 44 prompts, no overlap with P8, M96 or NEW44 (case-folded).
- Replay ids: run171.test_catalog() gives m96:0 to m96:47, matching Part 171's stored test rows for prompt id and question.
- Gate 6 harness with a tiny model: correct build passes; dose dropped fails 115 rows; sign dropped fails 46; omitted probe reads "caught" in all three.
- Synthetic u100 build (32-dimensional hidden, 21 layers): equal norms, independent recomputation matches, Gate 8 fails all six by cosine.
- 173.3 synthetic grid: 3 distinct -> 2 reported; 4 distinct -> 3 reported.
- Entity probes: the names listed in finding 3; no cross-entity alias collisions; NEW44B has 44 unique prompts.
- Not run: the real model, sets_google.pt (absent from this worktree), the reader APIs, and the Arc.

## Findings by severity

high 2, medium 7, minor 10 (total 19).
