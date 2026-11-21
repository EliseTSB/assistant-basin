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
- `test_part216.py`: **46 passed, 0 failed, 0 skipped** under the torch venv.
- All five sabotage mutations go RED and the tree restores green.

## FILES

| file | what it does |
|---|---|
| `part216/build_prompts216.py` | Builds `prompts216.json`. Verifies the opening `[40, 557, 7633, 539, 2972]`, the `[14030]` continuation, and that `neuron_report.build_prompt` + `tok.encode` still reproduce Part 214's `template_prefix_tokens` / `template_suffix_tokens`. Asserts every factual answer is ONE token. |
| `part216/prompts216.json` | The frozen bank: 24 items, 48 controls, 72 opener rows, 24 factuals, with token ids, `T`, `sequence_token_ids` and `read_position` for each. Built before any run. sha256 `ec29c594…faf757f`. |
| `part216/run216.py` | `selftest` (the two offline G1 fixtures) and `run` (456 forwards: 384 arm rows, 48 controls, 24 factuals). Hooks, lens, exact DLA, unit DLA, pins, checkpoint/resume by row key. |
| `part216/score216.py` | Pure numpy/json scorer: G0, G1, G2, outcomes, divergence, 216.1–216.6, descriptive tables; writes `score216.json` and `SCORECARD.md`. |
| `part216/test_part216.py` | 46 pytest cases: the analytic bf16 fixture, G0 pass/fail, G1 telescoping and the dropped-increment control, the boundary ledger, G2's pair/share/control rules, the outcome rules, 216.5/216.6 strict ties and zero writes, the tiny-model end-to-end run, and the scorer on a synthetic full run. |
| `part216/selftest216.json` | Output of `run216.py selftest` (the two fixtures), written by the CLI. |
| `part216/REPORT.md` | This file. |

## DESIGN-MAPPING

Every rule in `DESIGN.md` v2.2 and the function that implements it.

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

Result: **46 passed, 0 failed, 0 skipped** in 5.35 s. The torch venv was used, so the model tests did
NOT skip (`_tiny_model` would have skipped without `transformers.Qwen3_5ForCausalLM`; it did not).

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
