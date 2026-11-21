# Part 216 report — runner, scorer, prompt bank and tests

Design: `part216/DESIGN.md` v2.2 (reviewer-passed; not edited). Part 216 asks whether, at Qwen3.5-4B's
teacher-forced maker position (`I was developed by **`), a Google lead is visible in the middle layers
on the same questions that end with Alibaba winning, and whether an edited final Google win is a larger
Google projection, a smaller Alibaba-family projection, or both.

Nothing here was run on a GPU and the 4B checkpoint was never loaded (the brief forbids it). Everything
below is either a pure-numeric check, the tiny random Qwen3.5-architecture model, or a synthetic run
directory.

## SUMMARY

- `build_prompts216.py` writes `prompts216.json`: the 24 Part 214 items (with the Part 214 bank's
  SHA-256), the two frozen maker-slot control sentences, Part 215's `filler` / `denial` / `vogel_line`
  openers, and 24 single-token factual completions — every sequence serialized to token ids by the real
  Qwen3.5 tokenizer and checked against Part 214's recorded template prefix/suffix lengths.
- `run216.py` captures both residual adds of all 32 blocks at `T+4`, forms the realized increments as
  FP32 differences of the captured bf16 tensors, keeps the raw module outputs and the rounding residual
  beside them, and refuses to continue on a G1 or G0 failure.
- `score216.py` is numpy/json only (a test asserts torch is never imported) and implements G0, G1, G2,
  the outcome rules, the divergence index, 216.1–216.6 and the descriptive tables.
- `test_part216.py`: **46 passed, 0 failed, 0 skipped** on the c3a32d19 build; **65 passed, 0
  failed, 0 skipped** after Fold 1 (Alden #10), which appends the new guards and their tests below.
- All five original sabotage mutations go RED and the tree restores green; Fold 1 adds seven more
  (one per new guard), all 7 RED, restored green.

## FILES

| file | what it does |
|---|---|
| `part216/build_prompts216.py` | Builds `prompts216.json`. Verifies the opening `[40, 557, 7633, 539, 2972]`, the `[14030]` continuation, and that `neuron_report.build_prompt` + `tok.encode` still reproduce Part 214's `template_prefix_tokens` / `template_suffix_tokens`. Asserts every factual answer is ONE token. |
| `part216/prompts216.json` | The frozen bank: 24 items, 48 controls, 72 opener rows, 24 factuals, with token ids, `T`, `sequence_token_ids` and `read_position` for each. Built before any run. sha256 `ec29c594…faf757f`. |
| `part216/run216.py` | `selftest` (the two offline G1 fixtures) and `run` (456 forwards: 384 arm rows, 48 controls, 24 factuals). Hooks, lens, exact DLA, unit DLA, pins, fresh directories only (no resume: a non-empty `--out` is refused). |
| `part216/score216.py` | Pure numpy/json scorer: admission (the fixed 456-key grid, the frozen pins, the required arrays, G0/G1) then G0, G1, G2, outcomes, divergence, 216.1–216.6, descriptive tables; writes `score216.json` and `SCORECARD.md`; exits 0 / 3 / 2. |
| `part216/test_part216.py` | 80 pytest cases at Fold 2: the analytic bf16 fixture, G0 pass/fail, G1 telescoping and the dropped-increment control, the boundary ledger, G2's pair/share/control rules, the outcome rules, 216.5/216.6 strict ties and zero writes, the tiny-model end-to-end run, admission and its refusals, the no-resume refusals, and the scorer on a synthetic full run. |
| `part216/selftest216.json` | Output of `run216.py selftest` (the two fixtures), written by the CLI. |
| `part216/REPORT.md` | This file. |

## DESIGN-MAPPING

Every rule in `DESIGN.md` v2.2 and the function that implements it.

> Line numbers in this table are from the c3a32d19 build. Fold 1 (below) lists the changed
> functions and their current locations; the function names in this table are still the entry
> points.

### Model, sequences, indexing

| design rule | where |
|---|---|
| `Qwen/Qwen3.5-4B@851bf6e…`, bf16, eager, batch 1 | `run216.MODEL/REVISION` (51–52), load via `run214.load_rig` (`run216.run`, 534) |
| Part 214 Stage 1 serialization + the `[14030]` continuation for shape parity | `build_prompts216.stage1_item`, `prompts216.json[*].sequence_token_ids` |
| read position `T+4` (last opening token) | `read_position` recorded per row; used by `run216.forward_row` (358) |
| `h_0` = embedding output, `h_{i+1}` = residual after block `i` | `run216.install_capture` (132): `h0` from the input-embedding hook, `L{li}:add2` from the decoder-layer output |

### Measurements

| design rule | where |
|---|---|
| lens per `h_0..h_32` with the final norm **and its own RMS**, tied unembedding | `run216.lens_rows` (260) — FP32, chunked over the vocabulary for the ranks |
| `m_l`, `z_G`, `z_Al`, `z_T`, ranks | `run216.lens_rows` (260), `score216.margin` (145) |
| the model's `(1 + w)` RMSNorm form | `run216.prepare_tables` (234): `a = (1 + w) * W_U[token]` |
| two residual adds per block, verified present exactly once | `run216.boundary_report` (188), `score216.boundary_check` (241) — 2·n_blocks adds plus the `h0`/`L{li}:in` chain check |
| realized increment = FP32 difference of the two bf16 residuals | `run216.dla_block` (305) |
| raw module output and the rounding residual beside it | `run216.install_capture` (132) captures `raw:L{li}:mixer` / `raw:L{li}:mlp`; `dla_block` stores `rounding_norm`, `raw_norm`, `increment_norm` |
| exact DLA with the final RMS frozen at `h_32`, FP32 on CPU | `run216.dla_block` (305): `emb + Σ 64 terms == fp32_logits` |
| unit DLA for the 4 targets + 2 monitors: `c_G`, `c_Al`, `c_T`, write size, normalized projection | `run216.unit_dla_scalars` (336), `score216.unit_terms` (478) |
| finite fixed-RMS family contrast `d_A` | `score216.d_alibaba` (492) |
| per-question activations for every unit of the target layers, for the null | `run216.install_capture` (132) captures the down_proj input for `UNIT_LAYERS`; `forward_row` (358) writes them to `arrays.npz` |
| a unit with fewer than 12 valid questions is `no write` | `score216.target_series` (509), `MIN_VALID_QUESTIONS` (56) |
| observed dtype at each residual add | `run216.forward_row` (358) → `record['dtypes']`, `run.json['boundary_dtypes']` |

### Gates

| design rule | where |
|---|---|
| G0: final `g_ali` for 13 arms × q0–23 vs the Part 214 receipt within 0.02; q24 excluded | `score216.gate0` (175); `G0_ARMS` (45), `G0_TOLERANCE` (46) |
| G0 fail ⇒ STOP, before any non-anchor forward | `run216.run` (534): the anchor arms are planned first and G0 is evaluated as soon as the last anchor row lands |
| G1: emb + 64 realized increments == FP32 logit within 1e-3 | `score216.gate1_check` (221), `gate1` (259); `G1_TOLERANCE` (47) |
| G1 fixture 1: residual 1, write 1/256 → realized 0, rounding −1/256 | `run216.selftest` (415), `score216.bf16_round` (71) |
| G1 fixture 2: dropped increment must FAIL | `run216.selftest` (415) |
| G1 fail ⇒ STOP, no further forwards | `run216.run`: per-row `gate1_check` before the next row |
| G2: shares ≥ 0.8 on both sides, layers `l ≤ 29`, only controls whose FINAL contrast is on their expected side | `score216.gate2` (292), `expected_side_final` (287), `G2_SHARE` (48), `G2_MAX_LAYER` (49) |
| G2 usable pair = two consecutive layers in W | `score216.usable_pairs` (351) |
| G2: < 18 of 24 on either side, or < 2 usable pairs ⇒ every lead outcome `not read` | `score216.gate2` `checks` (292), `G2_MIN_CONTROLS` (50), `G2_MIN_PAIRS` (51), consumed by `unchanged_outcome` (379) |

### Outcomes

| design rule | where |
|---|---|
| early lead: `m_l > 0` on BOTH layers of a usable pair AND `m_32 < 0` | `score216.question_outcome` (361) |
| no early lead: `m_l ≤ 0` on every layer in W | `score216.question_outcome` (361) |
| `early Google lead visible` at ≥ 12/24 | `score216.unchanged_outcome` (379), `LEAD_MIN_QUESTIONS` (52) |
| `no early lead detected` at ≤ 2/24 with any `m_l > 0` in W | `score216.unchanged_outcome` (379), `NO_LEAD_MAX_QUESTIONS` (53) |
| `mixed / inconclusive` otherwise; `not read` when G2 fails | `score216.unchanged_outcome` (379), `OUTCOME_*` (59–62) |
| `ends in Google` iff `m_32 > 0` | `score216.arm_ends_in_google` (443) |
| divergence index: first `l ≥ min(W)` with the per-question median of `m_l − m_l(unchanged)` > 0.5 through `h_32` | `score216.divergence_index` (406), `DIVERGENCE_THRESHOLD` (54) |

### Predictions

| design rule | where |
|---|---|
| 216.1 G2 with ≥ 6 usable pairs | `score216.score_run` (776), `predictions['216.1']` |
| 216.2 / 216.3 the two outcome statements | `score216.score_run` (776) |
| 216.4 `zero`'s divergence index is `h_7` or later and exists | `score216.score_run` (776) |
| 216.5 negative median `c_G` and median normalized `c_G` strictly below all 200 same-layer nulls; ties never count; zero write excluded | `score216.prediction_216_5_unit` (571), `null_units` (532), `extremeness` (552), `NULL_UNITS` (55) |
| 216.6 positive median `d_A` and strictly above all 200 nulls | `score216.prediction_216_6_unit` (589) |

### Descriptive (not scored)

| design rule | where |
|---|---|
| per-layer median change in `z_G` and in `max(z_Al, z_T)` vs `unchanged` | `score216.share_tables` (669) |
| per-block attention and MLP shares of `z_G` and the Alibaba family | `score216.block_share_table` (688) |
| factual instrument check: lens top-1 vs final top-1 per layer | `score216.factual_check` (712) |

### Refusals and pins

| design rule | where |
|---|---|
| STOP record, no further forwards | `run216.run` (534): the `stops` list, `stage: 'STOP'` in `run.json` |
| pins: model, revision, design sha256, prompts sha256, tool sha256, kernel check, dtypes | `run216.run` `run_info` (534) |

## TESTS

Command actually run:

```
/home/elisepc/neuron-probe/.venv/bin/python -I -m pytest part216/test_part216.py -q -p no:cacheprovider
```

Result on c3a32d19: **46 passed, 0 failed, 0 skipped** in 5.35 s. After Fold 1: **65 passed, 0
failed, 0 skipped** in 9.4 s; see `## Fold 1 (Alden #10)` for the new cases. The torch venv was
used, so the model tests did NOT skip (`_tiny_model` would have skipped without
`transformers.Qwen3_5ForCausalLM`; it did not).

The tiny model is the `part215/test_part215.py` pattern: `Qwen3_5TextConfig(hidden_size=16,
num_hidden_layers=30, intermediate_size=9000, vocab_size=248320, full_attention_interval=4)`, eager,
`torch.set_grad_enabled(False)`. It exercises the REAL `run216.forward_row` — the same hooks, the same
FP32 lens, the same DLA — not a stub.

What the suite pins, by required item:

- **G0 pass/fail** — `test_g0_passes_on_matching_receipt`, `test_g0_fails_outside_tolerance`,
  `test_g0_fails_when_a_row_is_missing`, `test_g0_ignores_question_24`.
- **G1 telescoping on the tiny model holds** — `test_tiny_forward_records_every_boundary_and_telescopes`
  (max abs error 1.4e-17 on the tiny model).
- **The dropped-increment control fails** — `test_dropped_increment_control_fails_g1`.
- **The analytic bf16 fixture** — `test_analytic_bf16_fixture`, cross-checked against torch's own bf16
  add in `test_bf16_round_matches_torch`.
- **Boundary count = 2 per block exactly once, and a missing/duplicate is caught** —
  `test_boundary_count_is_two_per_block_exactly_once`, `test_missing_boundary_is_caught`,
  `test_duplicate_boundary_is_caught`, `test_gate1_catches_a_duplicate_boundary_row`.
- **G2 alternating qualified layers → 0 usable pairs → `not read`** —
  `test_g2_alternating_layers_have_no_usable_pair_and_are_not_read`.
- **Layers 14, 15, 16 in W → 2 pairs** — `test_g2_layers_14_15_16_give_two_pairs`.
- **Fewer than 18/24 controls on their side → `not read`** —
  `test_g2_fewer_than_18_controls_on_their_side_is_not_read`.
- **Outcome rules** — `test_early_lead_needs_both_layers_of_a_pair_and_a_negative_final`,
  `test_no_early_lead_question_needs_every_layer_in_w_nonpositive`,
  `test_outcome_lead_visible_at_twelve_of_twenty_four`,
  `test_outcome_no_early_lead_needs_at_most_two_questions_with_any_positive`,
  `test_outcome_is_not_read_when_g2_fails`.
- **216.5 / 216.6 strict tie, zero write, 12-valid-question floor** —
  `test_216_5_strictly_below_every_null`, `test_216_5_a_tie_with_any_null_does_not_count`,
  `test_216_5_needs_a_negative_median_c_G`, `test_216_5_no_write_below_twelve_valid_questions`,
  `test_216_6_strictly_above_every_null`, `test_216_6_a_tie_with_any_null_does_not_count`,
  `test_216_6_needs_a_positive_median_d_A`, `test_d_A_is_zero_for_a_zero_write`,
  `test_null_units_exclude_targets_and_zero_writes_and_take_200`,
  `test_null_units_are_the_nearest_by_median_write_size`.
- **`zero` changes nothing before `h_7` on the tiny model** — `test_tiny_zero_arm_changes_nothing_before_h_7`
  asserts exact equality of `m_l` for `l ≤ 6` and inequality somewhere at `l ≥ 7`.
- **The real run path writes its files** — `test_tiny_run_mode_writes_the_run_files` runs `run216.run`
  with `--limit 3` (only the loader is stubbed) and re-reads `forwards.jsonl` / `arrays.npz` / `run.json`
  through the scorer; `test_tiny_run_stops_when_a_row_does_not_telescope` shows a broken DLA produces
  `stage: STOP`, one row, `reason: G1 telescoping`.
- **The scorer on synthetic records** — `test_scorer_reads_a_whole_synthetic_run` writes a complete
  synthetic run directory (all arms, both control sides, the factuals, the unit arrays) and scores it.
- **The scorer stays torch-free** — `test_score_tool_does_not_import_torch` (a subprocess asserts
  `'torch' in sys.modules` is False after `import score216`).

## SABOTAGE

Scratch tree: `df-runs/part216-build-2026-11-21-r2/worker-01/sabotage/` (a copy of `part216/` with
symlinks to `part214/`, `part66/`, `neuron_report.py` and `results/`). The domain was not touched.
Driver: `sabotage/run_sabotage.py`. Baseline before and after: 46 passed.

| mutation | what was broken | suite | went red | tests that failed |
|---|---|---|---|---|
| G1 telescoping | `gate1_check` sums `terms[:-1]` instead of every term | 6 failed, 40 passed | **YES** | `test_dropped_increment_control_fails_g1`, `test_g1_telescoping_holds_and_a_dropped_term_fails`, `test_g1_tolerance_is_one_millith`, `test_tiny_forward_records_every_boundary_and_telescopes`, `test_g1…`, `test_tiny_run_stops…` |
| boundary count | `boundary_check` never reports a duplicate | 1 failed, 45 passed | **YES** | `test_gate1_catches_a_duplicate_boundary_row` |
| G2 usable-pair rule | `usable_pairs` treats every qualified layer as a pair | 3 failed, 43 passed | **YES** | `test_g2_alternating_layers_have_no_usable_pair_and_are_not_read`, `test_g2_layers_14_15_16_give_two_pairs`, `test_scorer_reads_a_whole_synthetic_run` |
| strict-tie rule | `extremeness` counts a tie as a win | 2 failed, 44 passed | **YES** | `test_216_5_a_tie_with_any_null_does_not_count`, `test_216_6_a_tie_with_any_null_does_not_count` |
| G0 tolerance | `gate0` never fails a value | 1 failed, 45 passed | **YES** | `test_g0_fails_outside_tolerance` |

Restored tree after the loop: 46 passed, so each mutation was the only difference.

## AMBIGUITIES

Readings I had to choose; the design does not settle them. Each is implemented conservatively and
recorded in the artifacts.

1. **The factual check's read position.** The design says "24 single-token factual completions … lens
   top-1 against final top-1 per layer" but not where to read. I read the LAST prompt token (the one
   whose next token is the answer), with no opening and no `[14030]` continuation, because any other
   position does not predict the answer. `prompts216.json[*].factuals[*].read_position` records it.
   The input-token echo rate is computed against that same position's token.
2. **The unit ratio is a median of per-question ratios, not a ratio of medians.** "the median is over
   those questions" is read literally: each question's `c_G / write size` is formed first, then the
   median is taken over the questions where the write size is nonzero.
3. **Null matching ties.** "nearest it in median write size" leaves ties open; I break them by unit
   index (deterministic and, as the design requires, independent of any token direction).
4. **`zero` is implemented as the multiplicative rung `alpha = 0.0`,** which is how Part 214 defined
   the arm (`ALPHA_RUNGS` contains 0.0 and `RUNG_ARM[0.0] = 'zero'`), rather than a clamp to 0.0. The
   arithmetic is identical and it keeps the G0 comparison byte-faithful to the receipt.
5. **Rank convention.** `rank_G/Al/T` are 0-based descending ranks (0 = the model's top token), and
   `argmax` is recorded separately.
6. **G2 denominators.** A control whose final contrast is on the WRONG side is excluded from both the
   numerator and the denominator of its side's share, which is the literal reading of "counting only
   controls whose FINAL contrast is on their expected side". A side with fewer than 18 such controls
   makes the whole window `not read`, so the exclusion cannot inflate a share into a pass.
7. **`min(W)` is undefined when W is empty.** G2 then already fails (< 2 usable pairs), so the
   divergence index is reported as `None` with the reason "G2 fails" rather than raising.
8. **`d_A`'s sign.** Implemented exactly as written: `LSE(z_Al, z_T) − LSE(z_Al − c_Al, z_T − c_T)`.
   A positive `d_A` therefore means the unit's write pushes the Alibaba family UP, which is what 216.6
   asks for. A zero write gives exactly 0.0 (tested).

## BLOCKERS

None for the deliverables. Two honest limits on what has been demonstrated:

- **The 4B model was never loaded** (forbidden by the brief), so `run216.py run` has been exercised
  end to end only on the 30-layer, hidden-16 tiny model and on synthetic records. The real run's
  runtime cost, VRAM and kernel-check outcome are unmeasured here. `--require-pinned-shape` asserts 32
  blocks so a wrong checkpoint fails loudly rather than silently reading 30-block indexing.
- **The lens scans the full vocabulary in FP32 to get the ranks** (`UNEMBED_CHUNK = 16384`). That is
  ~42 GFLOP per forward; on the A770 this is the dominant cost besides the model itself. It is correct
  but has not been timed on the device.

## Fold 1 (Alden #10)

Base build `c3a32d19` on branch `part216`. All work happened inside `part216/`; nothing was
committed. No GPU was used and the 4B checkpoint was never loaded. This section answers Alden's
code HOLD bullet by bullet, names the test that fails on `c3a32d19` and passes after, and reports
the seven new sabotage mutations.

### SUMMARY

- All four Alden bullets are fixed in `run216.py` / `score216.py`; `test_part216.py` grows from
  46 to **65 cases**, **65 passed, 0 failed, 0 skipped in 9.37 s** under the torch venv.
- Every new guard was proved against a `git archive c3a32d19 part216` copy with only the new test
  file copied in: **18 new/updated tests fail there**; 2 more are invariants that also passed there
  (`test_valid_resume_continues_without_duplicating_rows`, `test_lens_rank_of_the_argmax_token_is_zero`)
  and are labelled as guards, not as red tests.
- Seven sabotage mutations, one per new guard: **7/7 RED**, tree restored to 65 passed.
- A production-path crash unrelated to the four bullets was found and fixed: every control row
  raised `ValueError: unknown arm 'control_google'`; controls are now unedited forwards.
- The 4B/A770 path is still unexercised (see BLOCKERS); nothing below claims A770 evidence.

### FINDINGS

#### 1. Opener arms are delivered to the model on the exact `(arm, q)` key

**Change.** `run216.opener_index` (`run216.py:556`) builds `(arm, question_index) -> opener` from
`bank['openers']` and refuses a bank that is not exactly 72 unique keys over the three opener arms
x 24 questions. `run216.resolve_row` (`run216.py:585`) now sends `op_filler`, `op_denial` and
`op_vogel` through that index; the plain arms still resolve to `bank['items'][q]`, and controls
are checked against their own `(arm, q)` entry. `forward_row` (`run216.py:420-429`) writes the
exact `sequence_token_ids`, `prompt_token_ids`, `T`, `read_position` and an `input_sha256` into
each row, so the contract is stated in the artifact, not only in the call.

**Tests.** `test_opener_arms_resolve_to_their_own_bank_entries` (`test_part216.py:774`) covers all
72 keys: identity with the bank opener, a sequence different from the plain item's, and
`read_position == len(prompt_token_ids) + OPENING_TOKENS - 1`.
`test_run_forwards_plain_and_all_three_opener_variants` (`test_part216.py:791`) runs the real
`run` path with `unchanged` and all three opener arms in one plan and checks each written row's
tokens, read position and input hash against the bank opener entry read directly (not through the
resolver under test).

**First failure on `c3a32d19`.** `test_part216.py:782`
`assert {plain item 0, T=17, ...} is {opener filler 0, T=31, ...}` — `resolve_row` returned
`bank['items'][0]`. `test_part216.py:812`
`assert [248045, 846, ..., 3354, ...] == [248045, 846, ..., 369, 264, ...]` — the row written for
`op_filler:0` carried the 23-token plain sequence instead of the 37-token opener sequence.

#### 2. Runner -> scorer handoff, the 200-null rule, per-token block shares

**Change.** `run` (`run216.py:958-968`) writes `scale:<arm>:<qi>` for every committed row and, once
per run, `unit_layers`, `unit_proj`, `unit_colnorm`. `score216.unit_arrays`
(`score216.py:879`) reads those q-keyed scales (with a legacy positional `scales` fallback so the
old synthetic fixture still reads), `unit_analysis` (`score216.py:712`) uses each question's own
frozen scale, and `score_run` turns a missing table into an explicit `units.error` instead of
letting the packer raise `KeyError`.

`score216.null_shortfall` (`score216.py:658`) and `valid_null_values` (`score216.py:642`) make
216.5/216.6 require **exactly 200** finite null comparisons. 0, 1, 2 or 199 survivors is
`status: not read`, `pass: false`; a target that beats one surviving neighbour can no longer look
like a pass.

`score216.block_share_table` (`score216.py:797`) reports
`median_attn_z_G`, `median_attn_z_Al`, `median_attn_z_T`, `median_mlp_z_G`, `median_mlp_z_Al`,
`median_mlp_z_T`. The old summed `alibaba_family` columns are gone: Al and T are separate tokens
and their additive terms are never collapsed into a family share.

**Tests.** `test_runner_persists_unit_tables_and_q_keyed_scales` (`test_part216.py:911`) runs the
tiny model, checks the three tables' shapes, the `scale:unchanged:0` entry, `unit_arrays` and
`score_run` on that directory. `test_end_to_end_tiny_bf16_run_then_scorer_marks_partial`
(`test_part216.py:1063`) is the runner-to-scorer test the first build lacked.
`test_216_5_requires_exactly_two_hundred_valid_null_comparisons` (`test_part216.py:569`) uses 0,
1, 2, 199 and 200; it also removes one unreadable null from a 200 set and requires 199 -> `not
read`. `test_216_6_requires_exactly_two_hundred_valid_null_comparisons`
(`test_part216.py:589`) uses 1, 199 and 200. `test_block_share_table_keeps_al_and_t_separate`
(`test_part216.py:1159`) pins the three separate token columns and asserts the family columns do
not exist.

**First failure on `c3a32d19`.** `test_part216.py:919`
`assert 'unit_layers' in {'h0:unchanged:0': ..., 'h32:...', 'units:...'}` — the runner wrote only
per-row activations, no tables or scales. `test_part216.py:574`
`assert 'read' == 'not read'` with 0 valid nulls, `:593` the same with 1. `test_part216.py:1167`
`KeyError: 'tokens'` — the old block table had no per-token columns.

#### 3. STOP is terminal, gates run before append, resume revalidates, G0 rejects nonfinite

**Change.** A STOP record is terminal for its output directory: `run` (`run216.py:767`) loads an
existing `run.json`, and if `stage == 'STOP'` it re-writes the record with
`resume_refused`/`resume_refusal_reason` and returns STOP without loading a model or making a
forward. A `--no-resume` into a directory that already has forwards is refused the same way
(`refused.json`), because appending would duplicate rows.

In the forward loop (`run216.py:911-956`) the boundary ledger, G1 telescoping, `row_finite`, the
dtype contract and (on the G0 trigger row) G0 all run **before** anything is appended. Only a row
that passed every check gets its `h0:`, `h32:`, `units:`, `scale:` arrays and its
`forwards.jsonl` line. On failure the full row is embedded in `stops[0].row`; nothing is appended.
Every anchor row is also checked against the receipt before it is appended on a non-partial run
(a declared-partial smoke run skips that check, since the tiny fixture cannot match the 4B receipt
and makes no design-level claim; the 312-value G0 gate at the trigger row remains the complete check).

Resume revalidation is `pins_mismatch` (`run216.py:658`) plus `validate_committed_rows`
(`run216.py:681`), run before the loop. Pins compared: model, revision, design sha256, prompts
sha256, runner sha256, scorer sha256, receipt sha256, block count, hidden size, `unit_layers`,
`units`, `arms`, `control_arms`. For every committed row: the key is on the 24-question grid, the
sequence/prompt/read position are the bank's for that `(arm, q)`, the block count matches, the
boundary ledger is complete, G1 telescopes, every scalar is finite, and the arrays exist. Every
committed G0 row is checked against the receipt within 0.02, and when all 312 anchor rows are
committed the full `gate0` is recomputed. Pins are also written to `run.json` before the first
forward (`stage: RUNNING`, `run216.py:869`), so an interrupted run has something to validate
against.

`score216.gate0` (`score216.py:220`) rejects nonfinite `got`/`want` explicitly (a NaN used to
compare False to the tolerance and slip through); `gate1_check` (`score216.py:278`) rejects
nonfinite terms; `row_finite` (`score216.py:176`) names the first bad scalar for the runner's
pre-append check.

Partial/mixed runs: `--limit-questions` is a test-only flag recorded in `run.json` along with
`--limit`, `partial` and `run_status`; `run_completeness` (`score216.py:911`) also detects missing
`(arm, q)` rows. A partial run's design-level outcome is forced to `not read`, its predictions
carry `partial: true` and `pass: false`, and `scorecard_markdown` (`score216.py:1020`) prints
`PARTIAL RUN — not a full score`. A partial run can never read as a full score. `run_completeness` also treats a `stage: STOP`/`stops` record as partial, so a STOPped directory whose rows happen to be complete cannot be scored as full either.

**Also found and fixed.** `run` sent control rows through `edit_for_arm`, which raised
`ValueError: unknown arm 'control_google'` (`run216.py:531` on `c3a32d19`); no test ran past the
G0 arms, so this was invisible. Controls are unedited forwards whose sentence is in the prompt
(`run216.py:897`). The new test `test_control_rows_are_forwarded_without_an_edit`
(`test_part216.py:824`) fails on `c3a32d19` with that `ValueError` and passes now.

**Tests.** `test_stop_record_is_terminal_for_the_output_directory` (`test_part216.py:939`);
`test_resume_revalidates_committed_rows_before_any_new_forward` (`test_part216.py:963`, corrupts a
committed row's `fp32_logits`); `test_resume_rejects_a_committed_g0_row_outside_the_receipt`
(`test_part216.py:1013`, writes a receipt matching the tiny model for two committed rows, then
perturbs one); `test_valid_resume_continues_without_duplicating_rows`
(`test_part216.py:1047`, the paired positive control, which also passes on `c3a32d19`); the
updated `test_tiny_run_stops_when_a_row_does_not_telescope` (`test_part216.py:1123`);
`test_g0_rejects_nonfinite_values` (`test_part216.py:181`, NaN and +inf); `test_scorer_marks_a_stopped_run_partial` (`test_part216.py:1269`, a complete synthetic directory whose `run.json` says STOP is still partial).

**First failure on `c3a32d19`.** `:957`, `:977`, `:1004` and `:1040` all
`assert 'DONE' == 'STOP'` — a STOPped directory resumed and reached DONE; a corrupted committed
row was skipped; an anchor row outside the receipt was committed; a committed G0 row outside the
receipt was skipped. `:1148`
`assert 1 == 0` — the broken row had already been appended before the G1 check. `:1277`
`KeyError: 'partial'` — a STOPped complete synthetic directory had no partial label. `:188`
`assert True is False` — gate0 returned pass on a NaN row.

#### 4. The capture/device contract, the bf16 fixture, no device float64, same-chunk ranks

**Change.** `forward_row` records `dtypes` at `h0`, every `L{i}:in`, `L{i}:add1`, `L{i}:add2`,
and the raw `mixer`/`mlp` outputs. `expected_capture_dtype` / `dtype_contract`
(`run216.py:445` / `459`) enforce `torch.bfloat16` at every boundary when the load record claims
bf16 (`'torch.bfloat16'` or `'bfloat16'`); a mismatch is a pre-append STOP. The session tiny
fixture is now built with `.to(torch.bfloat16)` (`test_part216.py:_build_tiny`), so the real bf16
capture path runs; the only float32 tiny model is built by the refusal test and is documented
there as such.

`lens_rows` (`run216.py:260`) scans the chunk(s) containing the three token ids first, caches those
exact chunk-logit tensors, and takes each token's `z` from them. Ranks and argmax are computed from
the same chunked tensors, and the strict `>` comparison therefore compares a token's own value to
itself as equal: a token cannot outrank itself even if a separate GEMM would round differently.
The `torch.float64` tensor that used to be created on the target device is gone; the only float64
work is the CPU numpy output. This removes the Arc A-series FP64 requirement without changing the
FP32 arithmetic.

The tiny end-to-end test (`test_part216.py:1063`) runs `run216.run` on the bf16 tiny model into a
temp directory with `--limit-questions 1`, then invokes `score216.py` as a subprocess on that
directory and checks `SCORECARD.md`. The run is partial (9 rows), `run.json` records
`limit_questions: 1` and `partial: true`, and the scorecard carries the partial banner. The plan
in that test drops the ten `random_*:zero` arms: Part 214's recorded random pools index units up to
~9.3k and the tiny model's intermediate width is 9000 (see BLOCKERS).

**Tests.** `test_tiny_capture_records_every_boundary_dtype` (`test_part216.py:850`);
`test_bf16_run_refuses_float32_captures` (`test_part216.py:869`);
`test_no_float64_on_the_device_in_the_lens` (`test_part216.py:885`);
`test_lens_rank_of_the_argmax_token_is_zero` (`test_part216.py:892`, an invariant guard that also
passes on `c3a32d19`); the end-to-end test above.

**First failure on `c3a32d19`.** `:859` `KeyError: 'L0:in'` — the dtype record held only `h0` and
`add2`. `:877` `assert 'DONE' == 'STOP'` — a float32 capture was accepted on a run whose load said
bfloat16. `:888` `assert 'torch.float64' not in lens_rows source` — the old source contained
`(z, dtype=torch.float64, device=device)`.

### TESTS

Command actually run from `/home/elisepc/assistant-basin`:

```
/home/elisepc/neuron-probe/.venv/bin/python -I -m pytest part216/test_part216.py -q -p no:cacheprovider
```

Result after Fold 1: **65 passed, 0 failed, 0 skipped in 9.37 s** (46 passed on `c3a32d19`). The
torch venv is used, so the tiny-model cases do not skip.

Old-tree proof (recorded in the run directory `part216-fold1-2026-11-21/worker-01/`):

```
git -C /home/elisepc/assistant-basin archive c3a32d19 part216 | tar -x -C <run>/oldtree
# part214, part215, part66, neuron_report.py and results are symlinked into <run>/oldtree
cp part216/test_part216.py <run>/oldtree/part216/test_part216.py
cd <run>/oldtree && pytest part216/test_part216.py -q -p no:cacheprovider --tb=line \
    -k "opener_arms_resolve or forwards_plain_and_all_three or ..."
```

Result: **18 failed, 2 passed, 45 deselected**. The two passes are the declared guards. The full
log is `<run>/oldtree-newtests.log`; the first failing line for each new test is quoted in the
FINDINGS above.

New/updated cases (19):

1. `test_opener_arms_resolve_to_their_own_bank_entries` — red on `c3a32d19`.
2. `test_run_forwards_plain_and_all_three_opener_variants` — red.
3. `test_control_rows_are_forwarded_without_an_edit` — red (the unrelated crash).
4. `test_tiny_capture_records_every_boundary_dtype` — red.
5. `test_bf16_run_refuses_float32_captures` — red.
6. `test_no_float64_on_the_device_in_the_lens` — red.
7. `test_lens_rank_of_the_argmax_token_is_zero` — guard, green on `c3a32d19`.
8. `test_runner_persists_unit_tables_and_q_keyed_scales` — red.
9. `test_stop_record_is_terminal_for_the_output_directory` — red.
10. `test_resume_revalidates_committed_rows_before_any_new_forward` — red.
11. `test_anchor_row_outside_the_receipt_is_not_appended` — red.
12. `test_resume_rejects_a_committed_g0_row_outside_the_receipt` — red.
13. `test_valid_resume_continues_without_duplicating_rows` — guard, green on `c3a32d19`.
14. `test_end_to_end_tiny_bf16_run_then_scorer_marks_partial` — red (`KeyError: limit_questions`).
15. `test_g0_rejects_nonfinite_values` — red.
16. `test_216_5_requires_exactly_two_hundred_valid_null_comparisons` — red.
17. `test_216_6_requires_exactly_two_hundred_valid_null_comparisons` — red.
18. `test_block_share_table_keeps_al_and_t_separate` — red.
19. `test_scorer_marks_a_stopped_run_partial` — red (`KeyError: 'partial'`).
Plus the updated `test_tiny_run_stops_when_a_row_does_not_telescope` (red on `c3a32d19` at
`assert 1 == 0`) and the bf16 session fixture change.

### SABOTAGE

Scratch tree: `part216-fold1-2026-11-21/worker-01/sabotage/` — a copy of `part216/` with symlinks
to `part214/`, `part215/`, `part66/`, `neuron_report.py` and `results/`. Driver:
`sabotage/run_sabotage.py`. The domain was not touched by any mutation. Baseline 65 passed,
restored 65 passed.

| mutation | what was broken | suite | went red | failing tests (first 5) |
|---|---|---|---|---|
| opener resolution | `resolve_row` ignores `bank['openers']` and falls back to `bank['items']` | 2 failed, 63 passed | **YES** | `test_opener_arms_resolve_to_their_own_bank_entries`, `test_run_forwards_plain_and_all_three_opener_variants` |
| table persistence | `run` writes an empty `unit_layers` table | 2 failed, 63 passed | **YES** | `test_runner_persists_unit_tables_and_q_keyed_scales`, `test_end_to_end_tiny_bf16_run_then_scorer_marks_partial` |
| 200-null rule | `null_shortfall` accepts any nonempty comparison list | 2 failed, 63 passed | **YES** | `test_216_5_requires_exactly_two_hundred_valid_null_comparisons`, `test_216_6_requires_exactly_two_hundred_valid_null_comparisons` |
| STOP-before-append | the pre-append failure branch is disabled, so a failing row is appended | 4 failed, 61 passed | **YES** | `test_bf16_run_refuses_float32_captures`, `test_stop_record_is_terminal_for_the_output_directory`, `test_anchor_row_outside_the_receipt_is_not_appended`, `test_tiny_run_stops_when_a_row_does_not_telescope` |
| resume revalidation | `problems = []`: a resume trusts every committed row | 2 failed, 63 passed | **YES** | `test_resume_revalidates_committed_rows_before_any_new_forward`, `test_resume_rejects_a_committed_g0_row_outside_the_receipt` |
| nonfinite G0 | `gate0`'s nonfinite branch is disabled | 1 failed, 64 passed | **YES** | `test_g0_rejects_nonfinite_values` |
| dtype enforcement | `dtype_contract` never enforces a capture dtype | 1 failed, 64 passed | **YES** | `test_bf16_run_refuses_float32_captures` |

Restored tree after the loop: 65 passed, 0 failed, so each mutation was the only difference.

### BLOCKERS

None for the four code items. Honest limits on what Fold 1 demonstrates:

1. **The 4B/A770 path is still unmeasured.** The brief forbids loading the model, so every new
   guard is exercised on the 30-block tiny model or on synthetic records. `--require-pinned-shape`
   still refuses a non-32-block model on the real run, and the A770 dtype proof remains separately
   gated; nothing here is A770 evidence.
2. **The end-to-end test is a tiny run, not a 4B run.** It uses the `run216.run` function with
   `run214.load_rig` and `run214.kernel_check` stubbed (a CLI invocation would load the real
   checkpoint); the CLI's `main` dispatches to exactly this function. The plan is reduced with
   `--limit-questions 1`, which is recorded in `run.json` and forces the scorer's partial label.
3. **The e2e plan drops the ten `random_*:zero` arms.** Part 214's recorded random unit pools
   index units up to ~9.3k; the tiny model's `intermediate_size` is 9000, so those arms raise an
   index error in the fixture. The real run keeps all 16 arms; this exclusion is a fixture limit,
   stated in the test docstring, and does not alter the runner.
4. **The tiny run has 30 blocks (31 lens rows, 60 DLA terms), not the design's 32 (33/64).** The
   scorer now derives `n_layers` from `run['n_blocks']` so a tiny/partial directory still produces
   a scorecard; the design-level gates and outcomes are labelled `not read`/partial, never a full
   score. The 64-boundary requirement is still enforced for a 32-block run (`boundary_check`), and
   the real run's `--require-pinned-shape` keeps the depth honest.
5. **One extra production bug was fixed beyond the four bullets** (control rows crashing through
   `edit_for_arm`), because it sits on the same run path and would have stopped every real run
   before the controls/factuals. It is listed in FINDINGS #3 with its test.

## Fold 2 (Alden #12)

Base build `f7e306e0` on branch `part216`. All work inside `part216/`; nothing committed. No GPU and no
4B load: every guard below is exercised on synthetic records or on the 30-block tiny model. Alden's
code HOLD (`f7e306e0` over `c3a32d19`) had two admission bullets; each is answered here with the
change, the test that proves it, and the sabotage mutation that proves the test.

### SUMMARY

- **Admission (`score216.py`).** A full score now requires a FINALIZED receipt (`stage: DONE`), the
  design's pins, the fixed **456-key** grid (16 arms x 24 + 48 controls + 24 factuals) with no
  missing, duplicate or extra record, every required array present at its pinned shape and keyed to a
  row in that grid, **G0 PASS**, and **G1 PASS on every row**. Anything else is `diagnostics only`:
  the gates and descriptive tables are still printed, every prediction verdict is `not read`, the
  `unchanged` headline is not written, and the CLI exits **3** (`0` = admissible and scored, `2` =
  usage error).
- **Resume removed (`run216.py`).** `run` refuses any `--out` directory that already holds a run file
  (`forwards.jsonl`, `run.json`, `arrays.npz`, `selftest.json`, `refused.json`) before the model is
  loaded. `validate_committed_rows`, `pins_mismatch`, `PIN_KEYS`, `plan_kind` and `--no-resume` are
  gone. **There is no crash resume:** an interrupted run is discarded and rerun into a new
  directory. `arrays.npz` is still written only on exit, so a JSONL left in flight is always ahead of
  it; nothing pretends otherwise now.
- `test_part216.py` grows 65 -> **80 cases: 80 passed, 0 failed, 0 skipped in 30.4 s** under the
  torch venv (46 on `c3a32d19`, 65 after Fold 1).
- All **22** new/changed tests fail on `f7e306e0` (`oldtree-newtests.log`: **22 failed, 0 passed**).
- **7 sabotage mutations**, one per new guard: **7/7 RED**; the tree restores to 80 passed.
- The FILES table above was corrected in this fold: it still described `run216.py` as doing
  "checkpoint/resume by row key" and `test_part216.py` as having 46 cases.

### FINDINGS

#### 1. Scoring reads only an admissible attempt (Alden #12 #1)

**The fixed key set comes from the DESIGN, not the run's manifest.**
`score216.STAGE1_ARMS`/`EXPECTED_ARMS` (`score216.py:45-61`) is the design's arm list: `unchanged`,
`zero`, `scale_dial_fact`, `random_0:zero` ... `random_9:zero`, `op_filler`, `op_denial`,
`op_vogel`; with `control_google`/`control_alibaba` and `factual` that is 16 x 24 + 2 x 24 + 24 =
`EXPECTED_N_FORWARDS = 456`. `expected_keys()` (`score216.py:947`) builds the grid and
`stage1_keys()` (`score216.py:955`) the 384 rows that carry a `units:` capture. `bank_binding()`
(`score216.py:990`) reads the frozen `prompts216.json` beside the scorer and requires its arm list,
control arms, opener arms, 24/48/72/24 counts, `design_sha256` and model/revision to be the design's;
`PROMPTS_SHA256` and `DESIGN_SHA256` (`score216.py:66-67`) are the two committed hashes
(`ec29c594…faf757f`, `9dfbf216…fbb161`).

**A duplicate line is no longer silently merged.** `read_forwards()` (`score216.py:960`) returns the
rows, the duplicate keys and the line count; `load_run` keeps its old three-tuple API for callers that
only want the rows. A second `zero:0` used to collapse into the first and the directory looked
complete.

**Admission.** `admission_check()` (`score216.py:1097`) refuses, with `admission.reasons` naming each
one:
1. `stage`: `run.json` stage must be `DONE`, must carry no `stops`, and must not declare
   `partial`/`limit`/`limit_questions`. (The window Alden named -- stage RUNNING with every JSONL row
   but no `arrays.npz` -- is now refused twice: by the stage and by the missing arrays.)
2. `pin:*`: model, revision, `n_blocks` (32), `hidden_size` (2560), `unit_layers`
   `[6,7,11,24,27,29]`, `units`, `arms`, `control_arms`; `design_sha256` equals both `DESIGN.md` on
   disk and the frozen constant; `prompts_sha256` equals both `prompts216.json` on disk and the
   frozen constant; `tool_sha256` is recorded and equals `run216.py` on disk; the Part 214 receipt is
   readable and matches the sha256 the run recorded for it.
3. `keys`: exactly the 456 expected keys -- no missing, no duplicate, no extra. The 24 factuals and
   the 48 controls are counted from the design, so deleting an arm from `run.json` **and** from
   `forwards.jsonl` leaves a 24-key hole rather than a smaller complete grid.
4. `arrays`: `h0:<key>`/`h32:<key>` at `(2560,)`, `scale:<key>` scalar, `units:<stage1 key>` at
   `(6, n_units)` where `n_units` is `unit_proj.shape[1]`; the `unit_layers`/`unit_proj`/
   `unit_colnorm` tables at their pinned shapes; and no array keyed to a row outside the grid.
   `arrays_admission()` (`score216.py:1029`).
5. `G0` and `G1`: the existing gates must pass. G1 is `gate1()` over every row, so a single
   non-telescoping row refuses the whole directory.

`run_completeness()` (`score216.py:1217`) now also measures against `expected_keys()`; it stays
descriptive, but its `n_missing` can no longer be gamed by editing the manifest.

**Diagnostics only.** `score_run()` (`score216.py:1254`) computes the gates, the per-arm counts and
the descriptive tables for any directory, then:
- `unchanged_outcome` is replaced by `_not_read_headline()` (`score216.py:1208`) -- outcome `not
  read`, the first admission reasons attached, no question list;
- every entry of `predictions` gets `verdict: "not read"`, `pass: false` and a reason (an admissible
  run gets `verdict: "read"`);
- `score_status` is `diagnostics only`, `partial` is `not admissible`, and `admission` carries
  `admissible`, `status`, `n_reasons`, `reasons`, `missing`/`extra`/`duplicates`.
`scorecard_markdown()` (`score216.py:1362`) prints a `DIAGNOSTICS ONLY — not a full score` banner with
each reason, a `score status` line, and a verdict column in the predictions table.

**G2 is not admission.** `gate2` is still computed for every directory, but it decides only whether
the lead readout is eligible: an admissible run whose G2 fails is `score_status: scored` with
216.1-216.4 `pass: false` and the lead outcomes `not read`. That is the design's rule, unchanged.

**CLI.** `main()` (`score216.py:1419`): `0` when admissible and scored, `3` when diagnostics only
(reasons also on stderr), `2` for a missing/unreadable run directory or argparse usage error. The
scorer also no longer crashes when the receipt named by `run.json` is missing or unparsable
(`_load_receipt`, `score216.py:1243`): that is a `pin:stage1_receipt` refusal.

**Alden's three cases, as tests** (each also asserts exit 3, `admissible: false`, and every one of
216.1-216.6 `verdict: not read`):
- (a) one G0 row outside tolerance -> `test_admission_rejects_one_g0_row_outside_tolerance`
  (`test_part216.py:1472`); the reason is `G0`, and `gates.G0.pass` is false.
- (b) stage RUNNING with all rows but no `arrays.npz` ->
  `test_admission_rejects_a_running_stage_with_rows_but_no_arrays` (`test_part216.py:1484`); the
  reasons include both `stage` and `arrays`.
- (c) `op_vogel` and its 24 rows removed from the manifest and the file (432 rows) ->
  `test_admission_rejects_an_arm_removed_from_manifest_and_file` (`test_part216.py:1496`);
  `n_missing == 24`, all `op_vogel:*`, and `completeness.n_missing == 24` against the design grid.

The refusals are isolated one guard at a time by `_nonfinal_stage_with_complete_arrays`,
`_missing_arrays_file`, `_duplicate_key`, `_extra_key`, `_misaligned_array`, `_array_for_a_missing_row`,
`_pin_mismatch` and `_failed_g1_row` (`test_part216.py:1508-1582`). The positive fixture
(`_write_synthetic_run`, `test_part216.py:1245`) is a 456-row directory carrying the real frozen pins
and every array at its pinned shape; `test_admission_positive_fixture_is_scored_and_exits_zero`
(`test_part216.py:1451`) requires exit 0, `admissible: true`, `score_status: scored`, 216.2 PASS and
`verdict: read` on all six predictions. `test_admission_constants_are_bound_to_the_frozen_files`
(`test_part216.py:1584`) pins the constants to the files and shows a bank missing an arm is refused by
`bank_binding`.

#### 2. Resume is gone (Alden #12 #2)

**The refusal.** `RUN_FILE_NAMES` (`run216.py:655`) and `existing_run_files()` (`run216.py:658`) list
the run files a prior attempt leaves; `run()` (`run216.py:663`) refuses before `load_bank`,
`load_selftest`, `load_rig` or any forward, and writes `refused.json` with `refused: true`,
`existing_files` and the reason ("this runner has no resume -- a new attempt needs a new, empty
--out"). Deleting `validate_committed_rows` removes the resume path that skipped `dtype_contract`;
the fresh per-row validator (boundary ledger, G1, `row_finite`, `dtype_contract`, per-anchor G0)
is unchanged and still runs before every append.

**Alden's tampered-dtype directory.**
`test_tampered_dtype_directory_is_refused_not_resumed` (`test_part216.py:987`) writes one committed
row with the tiny bf16 model, flips `L0:add1` to `torch.float32` on disk, and re-runs into the same
directory: the result is `STOP` + `refused`, `n_rows` stays 1, the JSONL is byte-identical afterwards,
and `run.json` is untouched. On `f7e306e0` the same test fails at `assert None is True` -- that tree
resumed, appended a second row and returned DONE, exactly as Alden reported.

**What replaces the resume tests.** `test_run_refuses_a_directory_holding_any_run_file`
(`test_part216.py:965`) covers every name in `RUN_FILE_NAMES` with `load_rig` monkeypatched to fail if
called, so the refusal provably happens before any model work. `test_stop_record_is_terminal_for_the_output_directory`
(`test_part216.py:940`) keeps its meaning (a STOPped directory cannot be continued) under the new
shape. `test_a_committed_g0_row_outside_the_receipt_cannot_be_resumed` (`test_part216.py:1055`) and
`test_a_valid_done_directory_is_not_continued` (`test_part216.py:1089`) replace the two resume tests;
the second also shows the supported path -- the same plan into a new, empty directory runs to DONE
with exactly its own rows. `test_no_resume_machinery_remains` (`test_part216.py:1017`) asserts the
three helpers are absent and that `--no-resume` is now an argparse error (exit 2).

**Honest statement in the artefact.** `run.json` now records `resume_supported: false`; the runner's
`RUNNING` record is written before the first forward as a diagnostic (an interrupted directory says
RUNNING and must be discarded), never as a resume point. The module docstring says the same.

### TESTS

Command actually run from `/home/elisepc/assistant-basin`:

```
/home/elisepc/neuron-probe/.venv/bin/python -I -m pytest part216/test_part216.py -q -p no:cacheprovider
```

Result after Fold 2: **80 passed, 0 failed, 0 skipped in 30.4 s** (65 before this fold). The torch
venv is used, so the tiny-model cases do not skip. The new admission cases each run `score216.py` as a
subprocess, which is why the suite went from 9 s to 30 s.

The 22 new/changed cases, and the old-tree result (`oldtree/` is `git archive f7e306e0 part216` with
symlinks to `part214/`, `part215/`, `part66/`, `neuron_report.py`, `results/`, and this fold's test
file copied in; log at `oldtree-newtests.log`):

| test | what it pins | first failure on `f7e306e0` |
|---|---|---|
| `test_admission_positive_fixture_is_scored_and_exits_zero` | admissible fixture: exit 0, `scored`, 216.2 PASS, all six `read` | `KeyError: 'admission'` (`:1400`) |
| `test_admission_rejects_one_g0_row_outside_tolerance` | (a) a failing anchor refuses the score | exit `0 == 3` |
| `test_admission_rejects_a_running_stage_with_rows_but_no_arrays` | (b) stage + arrays both named | exit `0 == 3` |
| `test_admission_rejects_an_arm_removed_from_manifest_and_file` | (c) the grid is the design's, not the manifest's | exit `0 == 3` |
| `test_admission_rejects_a_nonfinal_stage_with_complete_arrays` | the DONE requirement alone | exit `0 == 3` |
| `test_admission_rejects_a_missing_arrays_file` | the arrays requirement alone | exit `0 == 3` |
| `test_admission_rejects_a_duplicate_key` | a duplicated record is refused, not merged | exit `0 == 3` |
| `test_admission_rejects_an_extra_key` | a 457th record is refused | exit `0 == 3` |
| `test_admission_rejects_a_misaligned_array` | `units:<key>` at the wrong shape | exit `0 == 3` |
| `test_admission_rejects_an_array_for_a_missing_row` | an array keyed outside the grid | exit `0 == 3` |
| `test_admission_rejects_a_pin_mismatch` | a changed revision pin | exit `0 == 3` |
| `test_admission_rejects_a_failed_g1_row` | one non-telescoping row | exit `0 == 3` |
| `test_admission_constants_are_bound_to_the_frozen_files` | 456 distinct keys; hashes equal the files; a short bank is refused | `AttributeError: EXPECTED_ARMS` (`:1586`) |
| `test_run_refuses_a_directory_holding_any_run_file` | every run file refuses the `--out`, before `load_rig` | `AttributeError: RUN_FILE_NAMES` (`:967`) |
| `test_tampered_dtype_directory_is_refused_not_resumed` | Alden's tampered-dtype directory cannot be continued | `assert None is True` (`:1009`) |
| `test_no_resume_machinery_remains` | the helpers and the flag are gone | `assert not True` (`:1019`) |
| `test_a_committed_g0_row_outside_the_receipt_cannot_be_resumed` | no re-blessing of a committed G0 row | `assert None is True` (`:1084`) |
| `test_a_valid_done_directory_is_not_continued` | even a clean DONE directory is refused | `assert 'DONE' == 'STOP'` (`:1099`) |
| `test_stop_record_is_terminal_for_the_output_directory` (rewritten) | STOP stays terminal under fresh-only | `assert None is True` (`:960`) |
| `test_scorer_reads_a_whole_synthetic_run` (extended) | the full fixture is admissible and scores | `KeyError: 'admission'` |
| `test_scorer_marks_a_stopped_run_partial` (updated) | a STOPped complete directory is diagnostics only | `KeyError: 'score_status'` (`:1424`) |
| `test_end_to_end_tiny_bf16_run_then_scorer_marks_partial` (updated) | a 9-row smoke run is exit 3 | exit `0 == 3` (`:1139`) |

Old-tree totals: **22 failed, 0 passed** for these 22 node ids. The other 58 cases in the file still
pass on `f7e306e0`, so the fold changed only the admission/resume surface.

### SABOTAGE

Scratch tree: `sabotage/tree/` -- a copy of the current `part216/` with symlinks to `part214/`,
`part215/`, `part66/`, `neuron_report.py` and `results/`. Driver: `sabotage/run_sabotage.py`;
results in `sabotage/sabotage.json`. The domain was not touched by any mutation. Baseline 80 passed,
restored 80 passed.

| mutation | what was broken | suite | went red | failing tests |
|---|---|---|---|---|
| DONE requirement | the stage/stops/partial block in `admission_check` disabled | 3 failed, 77 passed | **YES** | `test_admission_rejects_a_nonfinal_stage_with_complete_arrays`, `test_admission_rejects_a_running_stage_with_rows_but_no_arrays`, `test_scorer_marks_a_stopped_run_partial` |
| fixed key set | expected keys derived from `run['arms']`/`run['control_arms']` | 1 failed, 79 passed | **YES** | `test_admission_rejects_an_arm_removed_from_manifest_and_file` |
| arrays alignment | the `arrays_admission` call removed | 4 failed, 76 passed | **YES** | `test_admission_rejects_a_missing_arrays_file`, `_misaligned_array`, `_array_for_a_missing_row`, `_running_stage_with_rows_but_no_arrays` |
| G0/G1 admission | the G0 and G1 reason blocks removed | 2 failed, 78 passed | **YES** | `test_admission_rejects_one_g0_row_outside_tolerance`, `test_admission_rejects_a_failed_g1_row` |
| exit status | `main` returns 0 for a diagnostics-only score | 12 failed, 68 passed | **YES** | every refusal case plus `test_end_to_end_tiny_bf16_run_then_scorer_marks_partial` |
| fresh-directory refusal | `existing = existing_run_files(out_dir)` bypassed | 5 failed, 75 passed | **YES** | `test_run_refuses_a_directory_holding_any_run_file`, `test_tampered_dtype_directory_is_refused_not_resumed`, `test_stop_record_is_terminal_for_the_output_directory`, `test_a_committed_g0_row_outside_the_receipt_cannot_be_resumed`, `test_a_valid_done_directory_is_not_continued` |
| duplicate-key detection | a duplicate overwrites instead of being collected | 1 failed, 79 passed | **YES** | `test_admission_rejects_a_duplicate_key` |

Restored tree after the loop: 80 passed, 0 failed, so each mutation was the only difference.

### BLOCKERS

None for the two code items. Honest limits on what Fold 2 demonstrates:

1. **The 4B/A770 path is still unmeasured.** The brief forbids loading the model. Admission is
   exercised on synthetic directories and the no-resume path on the 30-block tiny model. Nothing
   here is evidence about the real run's runtime, VRAM or kernel check.
2. **"Aligned to the row keys" is name-and-shape alignment.** Admission proves every required array
   exists under the right key at the pinned shape and that no array is keyed outside the grid. It
   cannot prove that `h0:zero:4` holds the values of `zero:4` rather than a same-shaped swap from
   another row; that needs the model or a content hash the runner does not write. The unit arrays
   are read per question key, so a swap would change a descriptive number, not the gates.
3. **The admissible fixture is synthetic.** Its 456 rows are telescoping zero-terms with zero
   `h0`/`h32` arrays; it proves the admission logic and the scoring path, not the model. It is
   labelled as a fixture in its docstring.
4. **`tool_sha256` now binds the directory to the runner's bytes.** Any later edit to `run216.py`
   makes previously written directories diagnostics-only. That is the intent of a pin (the first
   real run must be scored with the runner that wrote it), but it is worth knowing before editing the
   runner after a run.
5. **`partial` changed meaning.** `score['partial']` is now `not admissible` (so a RUNNING or
   STOPped directory with 456 rows is partial); `run_completeness` remains descriptive and its own
   flag is reported beside `admission_inadmissible`. `run_status` is unchanged.

## Fold 3 (Alden #14)

### SUMMARY

- Scorer-side admission only, on `50f5ca3d`. Nothing committed, no GPU, no 4B load; `DESIGN.md` and
  `prompts216.json` are untouched, so their pins are unchanged (`9dfbf216…`, `ec29c594…`).
- **G0 grades against the PINNED Part 214 Stage 1 receipt.** `STAGE1_RECEIPT_PATH` is
  `results/part214/stage1/stage1.json` (verified now: sha256
  `4f287bc2153f879e2d442ba50a866208ae4384b3eafb527b568c43dd402d9cb5`) and
  `PART214_STAGE1_SHA256` is that digest. `load_pinned_receipt()` hashes the BYTES before parsing
  and refuses anything else with reason `pin:stage1_receipt_digest`; `run.json`'s own receipt path
  and digest are recorded as a claim (`admission.run_recorded_receipt`) and are never evidence.
- **Every row is bound to its pinned `(arm, q)` inputs.** For all 456 design keys,
  `input_binding()` rebuilds `sequence_token_ids`, `prompt_token_ids`, `T`, `read_position` and
  `input_sha256` from `prompts216.json` and compares them to the row; a mismatch is
  `diagnostics only` with reason `input:<key>`. Resolution rules are factored into a new
  torch-free `resolve216.py` imported by both `run216.py` and `score216.py`.
- **Arrays must be finite and scaled.** Every required array must be finite; every `scale:<key>`
  must be a finite POSITIVE scalar and, where the row records `dla.frozen_scale`, equal to it.
  Reasons are `array:<name>`.
- **A failed G2 no longer produces read/FAIL lens predictions.** On an admissible run with G2 false,
  216.2, 216.3 and 216.4 are `not read`; 216.1 is read as FAIL (its subject is G2); 216.5/216.6
  stay read (unit results do not depend on the window).
- **92/92 tests pass** (80 → 92; `92 passed in 59.50s`, and `92 passed in 67.69s` on the final
  re-run of the finished tree). All **13** new/changed tests fail on
  `50f5ca3d`: in the exact archive they error at the import of the new shared resolver, and with
  that resolver copied in (so the module imports and each guard is exercised) **13/13 failed for
  their own reason**. **6/6** sabotage mutations RED, each turning exactly its intended test red.

### FINDINGS

1. **Receipt binding (`score216.py`).** New constants `STAGE1_RECEIPT_PATH` and
   `PART214_STAGE1_SHA256`; `load_pinned_receipt(path=None)` returns `(receipt, reference_info)`,
   hashing the bytes first and refusing a missing/unreadable/non-matching file. The call site is
   `score_run`, not the run record: `_load_receipt(run)` is deleted. `admission_check` emits
   `pin:stage1_receipt_digest` and carries `admission.reference` (path, sha256, `pinned`,
   `injected`) plus `admission.run_recorded_receipt`. `SCORECARD.md` prints the reference it used.
2. **Test-only reference seam.** `score_run(out_dir, reference_receipt=None)`; the CLI (`main`) never
   passes it, and there is no flag, environment variable or `run.json` field that reaches it. An
   injected reference is labelled `injected: true` in the scorecard. Tests use it only to score the
   synthetic constant-1 fixture that the production path must refuse.
3. **Input binding (`score216.py` + new `resolve216.py`).** `resolve216.py` holds `plan_rows`,
   `row_key`, `opener_index`, `resolve_row`, `input_sha256` and `expected_inputs` -- no torch, no
   numpy. `run216.py` imports them (a one-argument `plan_rows` wrapper keeps the runner's call and
   its monkeypatched test seam byte-identical; `forward_row` now hashes via the shared function).
   `score216.py` imports `resolve216` after adding its own directory to `sys.path` (the tests run
   the CLI with `python -I`, which does not add the script directory). `input_binding()` compares
   the five recorded fields for every key present and names the mismatch `input:<key>`; keys absent
   from the file stay the `keys` check's business (no double report).
4. **Finite arrays and frozen scales (`arrays_admission`).** `unit_proj`, `unit_colnorm`, every
   `h0:`/`h32:`/`units:` array must be finite (`array:<name>`); every `scale:<key>` must be a
   finite positive scalar and must equal the row's `dla.frozen_scale` (`array:<name>`), with the
   value and the recorded value both quoted in the reason.
5. **Per-prediction `not read` (`score_run`).** `lens_unqualified = admissible and not g2['pass']`
   and `LENS_WINDOW_PREDICTIONS = ('216.2', '216.3', '216.4')`. Admission still forces every
   verdict to `not read`; the new branch sits between that and `read`. 216.1's own `pass` already
   keyed off `g2['n_usable_pairs']`, so it is read as FAIL; 216.5/216.6 are untouched.
6. **Fixture change (`test_part216.py`).** `_write_synthetic_run(tmp_path, receipt=None)` now
   defaults to the PINNED receipt and writes each anchor row's `native_g_ali` from it, so the
   positive fixture is admissible through the CLI. Passing `synthetic_receipt()` builds the bad
   fixture: a self-consistent constant-1 reference the CLI must refuse. Every row also records its
   `(arm, q)` inputs from the bank via an INDEPENDENT test-side resolver (`_bank_inputs`), not via
   `resolve216.expected_inputs`, so the fixture does not derive the expected values from the code
   under test.
7. **In-domain cleanup, listed as required.** The old admission block that compared a run's receipt
   to its own digest is gone (not left beside the new one); `_load_receipt` is deleted; the
   duplicated opener/plan resolution is merged into one module used by both programs.

### TESTS

`92 passed, 0 failed, 0 skipped` (`59.50s`, and `67.69s` on the final re-run) with
`/home/elisepc/neuron-probe/.venv/bin/python -I -m pytest part216/test_part216.py -q -p no:cacheprovider`
from `/home/elisepc/assistant-basin` (Fold 2 was 80). Twelve new cases plus one modified:

| test | asserts |
|---|---|
| `test_positive_fixture_is_bound_to_the_pinned_receipt` | admission's reference is `pinned`, sha256 = `PART214_STAGE1_SHA256`, path = `STAGE1_RECEIPT_PATH`; G0's 312 receipt values are the real receipt's (and not all 1.0); the run's recorded copy is a different file |
| `test_cli_refuses_a_run_whose_fixture_used_a_synthetic_receipt` | the constant-1 fixture gets CLI exit 3 / diagnostics-only / G0 FAIL; the same directory is admissible only through `score_run(..., reference_receipt=...)` |
| `test_pinned_receipt_digest_is_checked_before_use` | a tampered copy (different bytes, valid JSON) at the pinned path is refused with `pin:stage1_receipt_digest`; a missing path likewise; the run's recorded digest is untouched and unused |
| `test_input_binding_rejects_a_self_consistent_wrong_sequence` | `op_denial:0` recording `unchanged:0`'s sequence/prompt/T/read slot with a matching recomputed hash is refused as `input:op_denial:0` while G0 still passes |
| `test_input_binding_rejects_a_wrong_read_position` | right tokens, wrong read slot -> `input:unchanged:3` (`field == read_position`) |
| `test_input_binding_rejects_a_hash_that_does_not_match_its_sequence` | the recorded `input_sha256` is checked against the bank, not recomputed from itself |
| `test_expected_inputs_match_the_runner_and_the_bank` | `resolve216.expected_inputs` equals `run216.resolve_row` on five row kinds, equals the test-side bank resolver, and `sc.input_binding` accepts/rejects at unit level |
| `test_admission_rejects_a_nonfinite_array` | a NaN in `h0:zero:5` -> `array:h0:zero:5` |
| `test_admission_rejects_a_nan_frozen_scale` | `scale:unchanged:0 = NaN` -> `array:scale:unchanged:0` |
| `test_admission_rejects_a_nonpositive_frozen_scale` | self-consistent `-1.0` array and row -> only the positivity rule can refuse it, and does |
| `test_admission_rejects_a_scale_that_disagrees_with_its_row` | finite positive `2.0` array vs the row's `1.0` -> `array:scale:zero:0` quoting both |
| `test_g2_failure_leaves_lens_predictions_not_read` | admissible, G0/G1 pass, G2 fail: 216.2/.3/.4 `not read`, 216.1 read+FAIL, 216.5/.6 read, headline `not read`, scorecard shows `| 216.2 | not read |` |
| `test_admission_constants_are_bound_to_the_frozen_files` (modified) | adds `STAGE1_RECEIPT_PATH == results/part214/stage1/stage1.json`, `PART214_STAGE1_SHA256 == sha256(file)`, and `load_pinned_receipt()` returns the receipt with no error |

**Old-tree proof.** `oldtree/` is `git archive 50f5ca3d part216` with symlinks to `part214/`,
`part215/`, `part66/`, `neuron_report.py` and `results/` (same tree as Fold 2's worker-01). Two runs
of the selection above:

- exact head (`oldtree/part216` with no `resolve216.py`): **collection error** --
  `ModuleNotFoundError: No module named 'resolve216'`; every new test is an error
  (`oldtree-exact-newtests.log`).
- head plus the new torch-free `resolve216.py` (so the test module imports and each guard is tested
  on its own): **13 failed, 79 deselected, 0 passed** (`oldtree-newtests.log`). The failures name
  the guards: `AttributeError: module 'score216' has no attribute 'STAGE1_RECEIPT_PATH'`, the CLI
  returning 0 with `G0: True` for the synthetic receipt, `216.2` verdict `read` where `not read` is
  required, and `_assert_diagnostics_only` seeing exit 0 for every bad row/array.

### SABOTAGE

Scratch tree `sabotage/tree/` -- a copy of the current `part216/` with the same symlinks. Driver
`sabotage/run_sabotage.py`, merged results in `sabotage/sabotage.json`. The domain was never
touched. Baseline 92 passed, restored 92 passed.

| mutation | what was broken | suite | went red | failing tests |
|---|---|---|---|---|
| receipt digest | `if digest != expected_sha256` disabled in `load_pinned_receipt` | 1 failed, 91 passed | **YES** | `test_pinned_receipt_digest_is_checked_before_use` |
| input binding | `input_binding(rows, bank)` no longer called from `admission_check` | 3 failed, 89 passed | **YES** | `test_input_binding_rejects_a_self_consistent_wrong_sequence`, `_a_wrong_read_position`, `_a_hash_that_does_not_match_its_sequence` |
| finite arrays | the non-finite scan in `arrays_admission` removed | 1 failed, 91 passed | **YES** | `test_admission_rejects_a_nonfinite_array` |
| positive scale | `if value <= 0.0` disabled | 1 failed, 91 passed | **YES** | `test_admission_rejects_a_nonpositive_frozen_scale` |
| scale matches row | the `float(recorded) == value` refusal disabled | 1 failed, 91 passed | **YES** | `test_admission_rejects_a_scale_that_disagrees_with_its_row` |
| per-prediction not read | `lens_unqualified` forced False | 1 failed, 91 passed | **YES** | `test_g2_failure_leaves_lens_predictions_not_read` |

Each mutation turned exactly the intended case(s) red; nothing else in the 92 moved.

### BLOCKERS

None. Honest limits on what this fold demonstrates:

1. **The evidence is what the runner recorded.** Input binding proves the row's five fields equal
   the pinned bank entry; it cannot prove the model actually saw those tokens. The runner writes
   the same `seq` it feeds to `model(input_ids=...)`, so within a run directory the record is the
   only evidence a scorer can have, but a forged directory that writes the right fields and wrong
   logits would still pass G0 only if it also reproduced the receipt's 312 g_ali values; G1 and G2
   would still have to be satisfied by those forged logits. This is not cryptographic attestation.
2. **The digest is a single frozen constant.** Rebuilding Part 214's receipt makes every directory
   unscoreable until `PART214_STAGE1_SHA256` is updated deliberately. That is the intended pin
   behaviour, not an accident.
3. **`tool_sha256` now pins a changed runner.** This fold edits `run216.py` (the `resolve216`
   extraction and the shared hash), so any directory written with the previous runner bytes becomes
   diagnostics-only. No real Part 216 run exists yet, so nothing is lost; it matters before the
   first real run.
4. **The admissible fixture is synthetic and now borrows G0's values.** Its anchor rows carry the
   real receipt's `g_ali` and its other rows are telescoping zeros; it proves the admission and
   scoring paths, not the model.
5. **The 4B/A770 path is still unmeasured.** No model was loaded; the tiny-model tests exercise the
   runner only on a 30-block random model.
