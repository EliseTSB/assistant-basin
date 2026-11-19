# Part 213 build report — runner, scorer and fixture tests

Built by the part213 builder worker, 2026-11-19, from `PART213-DESIGN.md` v5 (commit 7de53067) and
`part213/BRIEF-build.md`. Nothing committed; nothing run against the model; no network or provider call; `results/`
untouched (read-only reads of `results/part99-pushes.pt`, `results/part99-pushes.json`,
`results/part106/lp-instruct.jsonl` only).

## SUMMARY

- `part213/run213.py` (41.7 KB): CPU float32 runner, subcommands `pilot`, `stage1`, `stage2`, `fr`, each `--out <dir>`
  (`--batch`, `--force`; `stage1 --with-fr`). Writes one JSONL per checkpoint of raw per-pass records (unit, kind,
  layer, neuron, prompt id, arm, dose, alpha, the full first-content logit row, `lp_sum`/`lp_mean`/`lp_first`,
  entropy, top-1, and `lp_by_unit` — every unit's full-NP score from the same pass, which is the design's zero-extra-pass
  `offdiag_8` baseline) plus `run.json` (resolved snapshot commits via `part85/checkpoints.py`, dtype, eager attention,
  tool and input SHA-256s, tokenizer digests, residual-norm calibration, pass counts). Obeys the Part 180 seal by
  writing sealed rows as `status: "sealed"` and never running them, and the pilot's output restriction by writing only
  `pilot_summary.json` (Gate 1, Gate 2, batch-16 vs batch-32 throughput, ±0.05 vs ±0.025 correlation on `J_N`, peak
  RSS, the 4 unpushed baselines, mechanical broken counts).
- `part213/score213.py` (56.7 KB): reads only the runner's records. Computes Gates 0–4, `N_z`/`R_z`/`L_z`, the
  paired transfer ratios with eligibility/`null` exclusion/`n_eligible` and the v5 numerator sign guard, the scene-framed
  denominators (pooled versions printed, deciding nothing), style-stratified τ_3/1, length-matched `Delta_lp_spec` plus
  the slot-peer-only term (v4's sum form printed descriptively), the own-unit full-NP clauses, finite-difference `J`,
  the exact one-sided sign test, the 198-item seed-213 blind export and 6-category audit tables, κ, the mechanical
  keyword screens, the FR symmetry diagnostic, the sensitivity tables, and the verdict of 213.1–213.11 under the v5
  disposition rule (`held` / `failed` / `inconclusive` / `not read`, gate voids). Writes `score.json`, prints a table.
  The DeepSeek V4.1 Flash second reader is `deepseek_reader()`, a single seam that raises `NotImplementedError`; nothing
  calls it.
- `part213/test_part213.py` (29.4 KB): 19 CPU-only tests on synthetic records, no model, no download, no network.
  **As run** with `/home/elisepc/neuron-probe/.venv/bin/python -m pytest part213/test_part213.py -p no:cacheprovider`:
  **19 passed** (about 20 s). Eight of them are the brief's required tests; each was also shown **red** against a
  deliberately broken copy of `score213.py` written to `tmp_path` (never the tracked file), run individually:

  | sabotage (copy of score213.py) | required test | result |
  | --- | --- | --- |
  | straddling CI returns `failed`; only an all-`failed` prediction fails | disposition rule | red (sabotage test passed) |
  | `sign_guard` forced `True` | sign guard | red |
  | τ_M/3 computed against pooled `THIRD_SLOT` instead of `third_2..3` | scene denominators | red |
  | `Delta_lp_spec` drops the `n_u` factor | length matching | red |
  | 213.11's slot-peer clause forced `held` | slot-family writer | red |
  | `spec_logit` returns the own logit only | constant-shift cancellation | red |
  | sign test divides by `2**(n+1)` | 7/8 = 0.0352, 6/8 = 0.1445 | red |
  | eligibility bar `>= 0.0` | eligibility / `null` exclusion | red |

  The other tests: bootstrap determinism (seed 213, 10,000 draws), 213.6 requiring the blind audit, and a failed Gate 2
  voiding Stage 1 predictions.

## FINDINGS

### Design ambiguities, with line numbers (most literal reading implemented)

1. **l.190-193 vs l.234, l.239 — "936 offdiag_8 passes" do not exist as a separate arm.** The design counts
   `39 × 8 × 3 = 936` full-NP passes, but l.204-207 says the `none` pass "already scores all 39 units' full-NP content
   log-probabilities" and l.234 says `none` "also supplies unpushed full-NP baselines". Read literally that means 39×3
   extra passes; read as the design's own accounting, it means the `neuron +0.8` core passes on `self_h0`, `third_0`,
   `third_2`
   already carry the scores. Implemented as the latter: every core pass records `lp_by_unit`, so the 936 passes are the
   three core passes' full-NP content on 8 units, and `instruct` Stage 1 is **5,430 executed passes** (5,538 planned,
   minus 108 sealed concept rows) instead of the design's 6,474. `sft`/`dpo`/`FR` are 936 each, as planned; Stage 2 is
   472 planned / 364 executed under the seal (corrected from 238 in the GLM-review pass, L6: 472 − 108 sealed
   concept rows = 364). The design's "9,282 total" therefore overcounts by 936; Amendment 1 later restored the 936
   `offdiag_8` passes (see the fixes section below), making `instruct` 6,366.
2. **l.67-68 — the seal names `self_select`, the prompt id is `self_select_0`.** Both are sealed (concept-unit rows
   only); slot rows on `self_select_0` run, as l.69-70 allows.
3. **l.196-199 — scene-framed denominator.** Implemented as `third_2, third_3` for τ_M/3 and τ_cop_nonid/3, with the
   pooled four-prompt version computed and printed (`tau_M/3_pooled_descriptive`, `tau_cop_nonid/3_pooled_descriptive`)
   and read by no clause. τ_3/1 stays pooled over `SELF_HELD`/`THIRD_SLOT` with the `direct`/`scene` stratified point
   medians as extra point clauses (l.202-203, l.443).
4. **l.205-207 — the sign guard is evaluated over the eligible units** ("the median numerator … over its eligible
   units"). Implemented that way; the guard failing makes the ratio `inconclusive` even when its CI is wholly below the
   bar. Note this makes 213.3's "CI upper bound ≤ 0.50 supports first-person selectivity" (l.443) unreachable through
   the ratio clauses when the numerator is suppressed — that is the v5 rule's intent (design l.548-550) and is stated
   here rather than resolved.
5. **l.430-431 — a point clause that misses fails the prediction.** Applied literally: a negative style-stratified
   median fails 213.3 even while its ratio clause is `inconclusive`. (The required sign-guard test therefore asserts
   `inconclusive` for the ratio clause and `failed` for the prediction.)
6. **l.441-442 — "95% CI lower bound of median L_z … and median N_z" vs "median N_z ≥ +0.75".** 213.1 reads both as
   CI clauses; 213.2 reads `L_z` as a CI clause and `N_z` as a point clause, exactly as worded.
7. **l.439 — 213.5's difference CI.** "95% unit-bootstrap CI of `median(L_z_concept) − median(L_z_slot)`" is not a
   single-unit statistic. Implemented as a bootstrap over every concept-slot pair difference (one draw per pair), which
   reduces to the median difference with an interval; flagged as a choice, not a certainty.
8. **l.443 — 213.9's paired difference.** "among slot units eligible at BOTH `sft` and `dpo`" — implemented as
   eligibility on the `L_z` denominator (`SELF_HELD`) at both checkpoints, `n_eligible` printed, `not read` below 15.
9. **l.444 — 213.10(a)** reads `median[L_z(SELF_HELD) − L_z(SELF_NONID)]` as a CI clause over the 21 slot units;
   (b) uses the scene denominator and the sign guard.
10. **l.445 — 213.11's slot-peer clause is `lo > 0`, strictly.** A new `ge_strict` clause type was added so a CI sitting
    exactly on 0 is `not read`-free but not `held` (the required slot-family test depends on this).
11. **l.445 — "mean baseline-subtracted full-NP specificity … has 95% CI"**: the bootstrap resamples the 21 per-unit
    values, so its point is their median, not their mean. The design says "mean" for the statistic and "median" in the
    same sentence for `N_z`; implemented as median (the bootstrap's own point) and flagged.
12. **l.283-284 vs `probe_plain213.py` — broken-answer screen.** The design's rubric (no `< 4 words`, no `"!!!!"`) is
    implemented; the probe's helper uses the older rule and is not reused (it is also outside this worker's write scope).
13. **l.345-346 — Gate 2's "same token count".** The anchor ids are asserted to equal `chat_ids + encode(anchor)` and
    compared with `results/part106/lp-instruct.jsonl`'s stored `lp_sum` for the same unit and arm; the stored token
    count is recorded beside it, not asserted (the file is the authority for what Part 106 scored).
14. **l.377-379, l.387 — pilot counts.** The design's 1,056 = 612 + 48 + 192 + 192 + 12; the 612 core passes are
    6 × 17 × 6, which already contains the 48 negative passes (6 × 4 × 2), so the plan has 1,044 + 12 = 1,056 only if
    the negative passes are counted twice. Implemented as the plan without double counting: 1,044 forward passes
    (1,044 − 24 sealed concept rows = 1,020 executed) plus 12 Gate 2 anchor passes.
15. **l.387-388 — pilot seal vs the 4 unpushed baselines.** The seal covers concept-unit *pushed* rows; the 4 unpushed
    Stage 2 baselines are run (l.386-387 requires them).
16. **l.128-129, l.340-343 — tokenizer digests.** `run.json` stores `tokenizer_sha256` per checkpoint; the pilot's
    `run.json` carries all three checkpoints' digests (it is read-only and cheap) so Gate 1 can check them before the
    full run.
17. **l.387-390 — step-size correlation.** The pilot computes Spearman (and Pearson) of `J_N` at ±0.05 vs ±0.025 over
    the 12 (unit, prompt) pairs and stores only the correlations and `passed`, never `J_N` itself (output restriction).
18. **l.413-414 — resume.** `stage1-<ckpt>.jsonl` is rewritten after every (prompt, arm) cell and skipped when present
    unless `--force`; a partial file is therefore resumable at cell granularity.
19. **l.108-111 — FR.** Built in place in float32 from instruct's state dict and dpo's safetensors streamed one tensor
    at a time; any instruct tensor absent from dpo stops the build rather than guessing.
20. **l.180-184 — `O_u`.** `offdiag_ou` from `targets213.json` is used as-is; units sharing a first token (`p92-08` /
    `p92-09`) contribute the same logit twice to the median, as the set requires. No deduplication.
21. **l.371-373 — the sign test is on 21 paired units** with discordant units only; `p < 0.05` needs ≥ 7 of 8 (the
    corrected v5 arithmetic, verified: 7/8 = 0.0352, 6/8 = 0.1445).
22. **l.459 — the audit tables' denominators.** Slot groups use 42, concept groups 36; concept `neuron -0.8` rows are
    excluded from the audit (l.259-260) and reported mechanically only.

### What was built beyond the letter of the brief (inside `part213/` only)

- `extra_diagnostics()` in the scorer: `SELF_SELECT − SELF_HELD`, ratio-of-medians and paired differences
  `S_3v1`/`S_3vM`, `own − mean(O_u)`, `lp_0`/`lp_pushed`/entropy/top-1 per prompt (l.202-203, l.219-224). Printed,
  deciding nothing.
- `full_np_sum_increment()` shares the `lp_by_unit` lookup with `full_np_increment()` so v4's diagnostic and v5's
  statistic cannot drift apart.
- `ci_clause(..., strict=True)` / `"ge_strict"` for 213.11's `lo > 0` (the `strict` keyword is accepted but the branch is selected by `side="ge_strict"`).

### Things the tests caught in the scorer while building (fixed, listed for the reviewer)

- `delta_spec_v4` read the pushed pass's own `lp_sum` for every peer (wrong once `lp_by_unit` exists).
- `gate2` had a dead loop; `gate3` had no baseline evidence before Stage 2 ran (now falls back to the pilot's recorded
  unpushed answers).
- `gate1` silently skipped absent checkpoints; it now reports them.
- 213.11's slot-peer clause used `lo >= 0` instead of the registered `lo > 0` (the required slot-family test caught it).
- Predictions were computed (and could crash) even when a gate had voided them; they are now short-circuited to
  `untestable (gate failed)`.

## BLOCKERS

None for the build. The run itself is blocked on things this worker was told not to do: the model is not run, the
Stage 2 blind reader is a seam, and registration still needs the pending plain-model probe (design l.96-98). Two notes
for whoever runs it: (a) `part92/common92.load_model` is bf16 and is **not** used — `run213.load_model_fp32` loads
float32 with eager attention, as l.102-103 requires; (b) the runner needs the local HF snapshots under
`/mnt/e/neuron-probe/hf/hub`, which this sandbox does not have, so no forward pass was executed here.

## SELF-ASSESSMENT

- `run213.py` is unexecuted against a model (no weights on this rig). Its tokenisation, hook placement, right-padding
  and generation paths follow `part92/common92.py` and `part106/run106.py`, but only the scorer side is test-covered.
  `forward_batch`'s `lp_by_unit` path has been reasoned through, not run.
- The synthetic builder in the tests models the `O_u` coupling exactly for solo pushes and approximately for coupled
  ones; the tests therefore use solo records and ask for values only on units a solo push can move (19 of 21 slot units;
  `p92-18`/`p92-37` share their first token with `p92-20`/`p92-49`). A real run's `N_z` will carry that coupling.
- 213.5's bootstrap choice (pairwise differences) and 213.11's "mean vs median" wording are the two places I would most
  want a human eye.
- The eight sabotage runs were executed one at a time and each turned its required test red; the full suite's
  `test_each_required_test_is_red_under_a_deliberate_break[...]` cases pass as a group as well.

## Fixes from the GLM review (2026-11-19)

Fixes for `part213/REVIEW-code-glm53-r1.md` F1-F8 and the cheap L items, plus Amendment 1's restored `offdiag_8`
passes. Every real fix has a test in `part213/test_part213.py` (`test_f*`/`test_l*`/`test_stage2_*`); all 12 were
shown RED against the pre-fix modules loaded from `git show HEAD:part213/...` (script and output kept outside the
repo, in the worker's `recheck/` directory). The suite is **35 passed**. `part213/smoke_tiny213.py` runs the real
`pilot` -> `stage1` -> `stage2` -> `score213` command lines end to end on a tiny random OLMo 2 and prints `SMOKE OK`.

### F1 — read positions now include the opening tokens
`run213.make_row` (run213.py:253) sets `prefix_len = len(prefix) + len(open_ids)`; `first_pos = prefix_len + len(lead)
- 1` and `content_pos = prefix_len + content_start - 1 + i` (logit at p scores token p+1, so this is the last lead and
each content token's predictor). The old chat-only prefix read every Stage 1 metric one opening token early.
Test: `test_f1_make_row_reads_after_the_opening_and_lead` (red: old positions 2 vs 4 on a 2-token opening).

### F2 — no NP score reads another row or a pad position
`run213.forward_batch` (run213.py:181) no longer takes a batch-wide `np_specs`; it asserts `first_pos`/`content_pos`
lie inside their own row and scores only that row's own `content_pos`. Full-NP peer scores now come from their own
rows (`run_offdiag_passes`, run213.py:552). The `lp_by_unit` structure is gone.
Test: `test_f2_and_f3_offdiag_passes_are_on_the_peer_row_and_never_pad` (red: old `forward_batch` had no `keep_ids`;
old `full_np_increment` returned None without `lp_by_unit`).

### F3 — the 936 offdiag passes are back; 213.11 reads them
`run_offdiag_passes` (run213.py:552) applies u's `neuron +0.8` hook to each of u's 8 `offdiag_8` peers' **own** rows on
`self_h0`, `third_0`, `third_2` (39 x 8 x 3 = 936 records keyed by `push_unit`/`scored_unit`). `stage1_pass_count`
(run213.py:317) reports 5,430 core + 936 = **6,366** instruct passes (Amendment 1). `score213.Stage1.full_np_increment`
(score213.py:418) reads the `(u, v, prompt)` offdiag record for v != u and the own `neuron +0.8` record for v == u;
`full_np_sum_increment` matches.
Tests: `test_f3_instruct_executes_6366_passes` and `test_f2_and_f3_...` (red: old count function absent / 5,430; old
reader wanted `lp_by_unit`).

### F4 — per-checkpoint manifests and the automatic pilot-evidence path
`run213.write_run_json` (run213.py:451) writes `run-<name>.json` (never a single `run.json`). Stage 2 writes
`run-stage2.json`, so it cannot clobber stage 1's `run-instruct.json`; FR writes `run-fr.json` under the instruct
provenance (part85's registry has no `fr` key). `stage1 --pilot DIR` copies `pilot_summary.json` into the records
directory and merges `pilot_gate1`/`gate2_anchors_detail`/`stage2_baseline_answers` into `run-instruct.json`
(`load_pilot_evidence`/`merge_pilot_evidence`, run213.py:579/610). `score213.load_records(records_dir, pilot_dir)`
(score213.py:109) also auto-reads `<records>/pilot_summary.json` and accepts `--pilot DIR`; a legacy `run.json` is
still accepted and keyed by its `checkpoint` field. The smoke's pilot summary has exactly the real pilot's key set.
Two further contract bugs found and fixed here: (a) `assign_blind_ids` ran *after* 213.6/Gate 4, so every audit
label lookup saw `blind_id=None`; it now runs first (score213.py:1222). (b) each manifest nested
`tokenizer_sha256` per checkpoint while Gate 1 expected `{file: sha}`; the writer now stores this checkpoint's
digests and `_tokenizer_digests` (score213.py:681) accepts both shapes.
Test: `test_f4_per_checkpoint_manifests_and_pilot_evidence` (red: old writer produced `run.json` only).

### F5 — headline 3's Stage 2 clause, and gates that need their evidence
`score213.slot_genuine_ci` (score213.py:1233) computes the 10,000-draw unit bootstrap of the slot
`genuine_third_attr` rate over the 21 units (42 THIRD_FREE answers) and is `not read` unless all 42 are labelled.
`headline_reading` (score213.py:1332) now requires that CI's `read` status and `hi < 10/42` before "First-person
self-slots supported" can fire (the old note-only clause is gone). `gate3` (score213.py:768) fails when audit labels
exist without a `kappa` (the audited endpoint is unread); `gate4` (score213.py:800) fails unless all 42 slot
`neuron +0.8` rows carry labels instead of counting refusal as 0.
Test: `test_f5_headline3_needs_a_complete_stage2_ci_and_gates_need_labels` (red: old signature had no CI argument;
old gate3 passed with `kappa=None`; old gate4 passed with no labels).

### F6 — a partial or crashed output is not a finished run
`run213.write_completion`/`is_complete` (run213.py:362/370) write and verify a `<jsonl>.done.json` marker holding the
checkpoint, row count and SHA-256; `stage1`/`stage2`/`fr` skip only when the marker matches (`--force` overrides).
Test: `test_f6_partial_output_is_not_a_finished_run` (red: old code had no marker and skipped any existing file).

### F7 — a partial blind audit is not a complete one
`score213.predict_213_6` (score213.py:944) counts the 84 audited rows (42 neuron +0.8, 42 random_0 +0.8) and requires
all of them labelled; otherwise every clause is `not read` and `audited_rows`/`labelled_rows` are reported.
Test: `test_f7_partial_blind_audit_is_not_read` (red: old code held 213.6 from 42/84 labels).

### F8 — records keep only the logits the scorer reads
`run213.first_logit_ids` (run213.py:168) is the own first-content token plus each `offdiag_ou` peer's first token
(at most 1 + 38 = 39 ids); `forward_batch` stores `first_logits` as `{str(token_id): logit}` and nothing else, while
entropy/top-1 stay scalars. `score213.logit_value` (score213.py:167) reads the map and still accepts old lists. The
smoke's instruct records have at most 35 first-logit entries per row and the whole records dir is 11.0 MB (the old
form would be ~31 GB at load).
Test: `test_f8_rows_keep_only_the_logits_the_scorer_reads` (red: old `first_logit_ids` absent; rows held the full
100,278-float vocabulary).

### Extra bug found by the smoke — Stage 2 was expanded 39 times
`run213.stage2_plan` (run213.py:727) looped the 39 units while `cmd_stage2` also loops them, so the real Stage 2 would
have run 468 cells x 39 units = 18,252 generations instead of 472 planned / 364 executed. The plan now returns the 16
unit-agnostic cells; the manifest records `stage2_plan_cells`/`stage2_rows`/`stage2_executed`.
Test: `test_stage2_plan_is_16_cells_and_364_executed_rows` (red: 468 cells on HEAD).

### Low findings
- **L1 fixed:** `score213.ratio_clause` (score213.py:287) checks `n_eligible < 15` before the sign guard; a low-n
  ratio with a failed guard is `not read`, not `inconclusive`. Used by 213.3/213.4/213.10.
  Test `test_l1_low_n_wins_over_a_failed_sign_guard`.
- **L2 fixed:** 213.7 now prints `slot_neuron_minus_none` and `concept_neuron_minus_none` CIs alongside the gap.
- **L3 fixed:** `score213.residual_norm_comparison` (score213.py:1261) reports each checkpoint's layer-0 median norm
  and its ratio to instruct; the manifest note no longer promises an unread comparison.
- **L4 fixed:** `run213.check_finite` (run213.py:392) stops a checkpoint at the first non-finite pass, and
  `write_jsonl` uses `allow_nan=False` so bare `NaN` is never written. Test `test_l4_non_finite_pass_stops_the_run`.
- **L5 deferred:** the pilot's registered 76-generation throughput arm would add 76 greedy generations to a pilot
  that has already run and passed. Generation throughput is descriptive, not a gate or a prediction; adding it now
  would change the recorded pilot budget without buying a registered number, so it is left as a stated gap.
- **L6 fixed:** the "238 executed" line above is corrected to 364 (472 - 108 sealed concept rows).
- **L7 fixed:** 213.6's difference CI is the unit bootstrap of the total `(neuron_hits - random_hits)` over all 21
  units, and the clause is `ge_strict`, so a 2.5th percentile exactly on 0 is not "above 0". Test
  `test_l7_difference_ci_is_the_total_and_strictly_above_zero`.

### Smoke test (`part213/smoke_tiny213.py`, new)
Builds a 16-layer, 19.1M-parameter random OLMo 2 (`hidden_size` 32, 2 heads/kv-heads, `intermediate_size` 8192 so
neuron 7,850 exists) from `part213/tiny-olmo-cfg/config.json` with that folder's real tokenizer, in a scratch folder.
It patches only model loading, snapshot lookup, provenance, the frozen push vectors (random, seed 213) and the Part
106 Gate 2 reference (the tiny model's own anchor logits; Gate-2-ordering vectors are found by finite differences).
Model seed 14 is the first whose four unpushed baselines are not mechanically broken (Gate 3 is a real check).
It then invokes the real CLI paths: `pilot`, `stage1 --pilot`, `stage2`, `score213 --pilot`. Clean-run result
(459 s): `SMOKE OK`; Gates 1-3 pass, Gate 4 correctly fails without a blind audit; 213.1-213.5 and 213.7-213.11
`failed` on random weights, 213.6 `not read` (no audit) — every prediction scored, none `untestable (gate failed)`,
none errored; instruct executed 6,366 passes, Stage 2 wrote 472 rows and executed 364; records 11.0 MB, max 35
first-logit entries per row.
