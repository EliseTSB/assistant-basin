# Cold check: Part 206's scorer against its registered text

Files read (all under `/home/elisepc/green/worktrees/ab-part206/`): `PART206-DESIGN.md` (table L340-358, method
L160-253, Gates L279-294, Reading rules L377-410), `part206/AMENDMENT-1..5.md`, `part206/score206.py`,
`part206/common206.py`, `part206/analyze206.py` (to see what each summary key is), `part206/config206.json`, the
`results/part206/*/summary.json` and `qwen-chat/causal.json` behind the cited values, and
`results/part206/score206-output.txt`. No model was run. Line numbers are file:line.

## SUMMARY

**13 of 17 rows match** the registered text (206.1, 206.2, 206.6, 206.7, 206.8, 206.12, 206.M1 [after Amendment 5],
206.M2, 206.M3, 206.M4, 206.M5, and 206.9 / 206.10 as written). **4 do not**: 206.3 (sign/direction of the write
share is not checked — and it is negative), 206.4 (the code reads the split's `m` sublayer, but the registered text
says "MLP output" and the method defines the split on `m` *and* `c` with "MLP and mixers" in the part's aim — the
reading is defensible but ambiguous, and it is the difference between held and failed), 206.5 (the registered rule is
a strict "three quarters or more" of the first-third layers, i.e. 4.5 of 6; the code's `>= 0.75*6` passes at 4 —
the verdict happens to be the same, but the rule is not the registered one), 206.11 (the code compares a
`rho_peak` over the middle third with a `rho_peak` the registered text does not scope to the middle third; at the
layer of `rho_peak(A-tmpl,M)` the ratio is 0.029/0.106 = 0.27, still failed). 206.7's interval is the passage-resampled one the registered rule requires (each
OLMo stage's `rho_peak_resamples` holds 1000 values; I verified this after first mis-reading the key — see F1). The
headline sentence printed in the output is correct under the registered rule.

## Row table

| row | registered rule (Design line) | what the code checks (score206.py line) | verdict in output | MATCH / MISMATCH |
|---|---|---|---|---|
| 206.1 | GT passes in both Qwen models (L342) | `gt.verdict.pass` for qwen-base and qwen-chat, both must be true; not read if unreadable (L68-72) | failed (both False, output L7-8) | MATCH |
| 206.2 | GT passes at >= 5 of the 6 OLMo stages (L343) | `sum(olmo) >= 5` over `olmo_stage_order` (L73-74, config206.json) | failed (0 of 6, output L1-6) | MATCH |
| 206.3 | Qwen chat, middle layer, MLP sublayers write **at least half** of raw `r_L` (L344, method L196-201) | `reference[midL].write_share.mlp >= 0.5` (L80-81) | failed (cited -0.3419, output L9) | **MISMATCH (sign)** — the registered "writes at least half" is a share of the norm, whose sign the code does not constrain; the value is negative, so the MLP subtracts along `r_L`. Under the registered text the row is failed too, but for the opposite reason, and a value of -0.9 would still be "failed" only by accident. The code should require `0 <= share` (or the design should say the signed projection is the quantity). |
| 206.4 | Qwen chat, frame P: first layer's MLP token share >= 0.5 **and** middle layer's <= 0.25 (L345) | `split.m[0].token_share >= 0.5 and split.m[midL].token_share <= 0.25` (L82-84) | held (0.556, 0.118; output L10-11) | **MISMATCH / ambiguous** — the split is defined for `y` in {`m_L`, `c_L`} (Design L179) and the row says "MLP output", which is `m` (Design L167-172), so `split.m` is the defensible reading; but the part's aim (L26) is "how much of what MLPs write, and of what mixers write", and 206.5 pairs `m` with `c`. If the intended quantity were the mixer (`split.c`), the first layer is 0.323 (< 0.5) and the row **fails**. The design never says which of `m`/`c` "MLP output" means beyond the site table; this is the one place where an ambiguity changes a verdict. |
| 206.5 | OLMo Instruct, frame P: MLP token share > mixer token share at **three quarters or more** of the first-third layers (L346) | `wins >= 0.75 * len(first)` (L117-123); `first_third(16) = [0..5]` (common206.py:97-99) | held (wins = 6 of 6, output L61-72) | **MISMATCH (threshold arithmetic)** — "three quarters or more of 6 layers" is 4.5, and a count cannot be 4.5, so the registered rule admits only 5 or 6; the code admits 4. With the cited values (6 wins) the verdict is held either way. The code's `>=` on a float fraction is the registered "or more" only if the design means rounding down. |
| 206.6 | Qwen chat: `rho_peak(A-tmpl,M) >= 0.2`; passage-resampled 95% interval lower end >= 0.1; at that layer cos(`Delta(A-tmpl,M)`, `Delta(A2-tmpl,M)`) >= 0.8 (L347) | `rp >= 0.2 and ci[0] >= 0.1 and cos_request_change[lay] >= 0.8` (L85-90); `rho_peak` = max `rho` over the middle third (analyze206.py:141-144), `lay` = its layer | failed (0.106; ci [0.103, 0.110]; cos 0.997; output L12-14) | MATCH (fails on clause 1; `rho_peak` is the middle-third max as Design L230-231 defines, `ci` is the passage bootstrap of the peak, analyze206.py:171-187) |
| 206.7 | Across the six OLMo stages, the largest adjacent rise in `rho_peak(A-tmpl,M)` is midtrain-end -> SFT; interval from resampling passages (same passages both stages) excludes 0 and lies above the second-largest rise (L348, L383-385) | `adjacent_rise_rule`: target is the argmax, `ci[0] > 0` and `ci[0] > second` (L48-59) | held (0.0755 rise; ci [0.0714, 0.0795]; second 0.0102; output L112-128) | MATCH (each OLMo stage's `rho_peak_resamples` has 1000 entries; target rise 0.0755 is the largest of five; interval excludes 0 and its lower end 0.0714 is above the second-largest rise 0.0102; the same passage draws are reused across stages because every stage reseeds `default_rng(bootstrap)` with the same 600 passages, analyze206.py:155-161) |
| 206.8 | OLMo Instruct, (A-tmpl,M): slot median >= concept median + 10, slot median >= 60, gap > label-shuffle 95th percentile (L349, L388-390) | `gap >= 10 and sl >= 60 and gap > p95` (L124-128); activity-matched reported beside, note if it changes (L129-134) | failed (49.05 vs 46.81, gap 2.25, p95 16.87; output L73-79, L110-111) | MATCH (all three clauses; sets are Amendment 4's 21 slot / 18 concept, 39 neurons — `len` of the arming lists confirms) |
| 206.9 | Qwen chat: 16 identity neurons' median arming percentile between 35 and 65 (L350) | `35 <= med <= 65` on `arming.identity.median_percentile` (L91-92) | held (45.97; 16 neurons confirmed) | MATCH |
| 206.10 | Qwen chat: subtracting the assistant shift at k=1 lowers the maker token's log prob by more than every one of the 20 random directions **and** more than the (M2,M) direction; k=2 lowers it further (L353) | `all(ch["arm_k1"] < x for x in ctrl) and ch["arm_k2"] < ch["arm_k1"]` with `ctrl` = every arm ending `_k1` except `arm_k1` (L107-108) | failed (arm_k1 0.0622 vs null_M2M_k1 0.0857; arm_k2 0.0813 > arm_k1; output L17-18, L39) | MATCH (the control list is the 20 randoms plus `null_M2M_k1`; `plain` is excluded by the `_k1` suffix; G3 passed 23/24 so the row is read; rows restricted to `plain_top1_is_maker` as Design L246-249 says — 23 rows, I recomputed the means from causal.json and they agree) |
| 206.11 | Qwen chat: `rho_peak(A-tmpl,R-tmpl)` >= half of `rho_peak(A-tmpl,M)` (L351) | `rr >= 0.5 * rp`, both `rho_peak` (L93-94) | failed (0.0305 vs 0.106; output L12, L16) | **MISMATCH (layer scope)** — `rho_peak(A-tmpl,R-tmpl)` is the middle-third max at layer 11 (0.0305), while `rho_peak(A-tmpl,M)` is at layer 18; the registered text does not say the R-tmpl peak is taken over the middle third, and the reading "at the same layer" (the Reading-rules headline wording, L403, says "on the same model", not "same layer") is at least as natural. At layer 18 `rho(A-tmpl,R-tmpl)` = 0.0290, ratio 0.27 — the verdict is **failed** either way. A note, not a verdict change. |
| 206.12 | Across the six OLMo stages, the largest rise in the slot-minus-concept median arming gap is SFT -> DPO; read like 206.7, interval resamples passages and neurons (L352, L386-387) | `adjacent_rise_rule(order, gest, gres, ("sft","dpo"))`, `gres` = `arming_gap_resamples` (L145-150) | failed (largest is midtrain-26B -> midtrain-end, 32.0; target rise -0.043; output L129-145) | MATCH (the gap is slot median minus concept median per stage, as L352 says; the resample array is present, `len 1000`) |
| 206.M1 | Qwen chat: `rho_peak(A-tmpl,M)` in [0.07, 0.30] **and** `rho(M2,M)` **at the same layer** in [0.00, 0.01] (L354) | `0.07 <= rp <= 0.30 and 0.00 <= rn <= 0.01` with `rn = common_mode["M2|M"]["layers"][lay]["rho"]`, `lay` = `rho_peak_layer` of A-tmpl|M (L166-170) | failed (0.106 in range; 0.0287 at layer 18 out of range; output L92-93, L146) | MATCH after Amendment 5 (the range is the registered 0.00-0.01; the same-layer reading is the registered one; bounds inclusive, "between" read as inclusive — see F4) |
| 206.M2 | OLMo Instruct: slot-minus-concept gap 10 to 20 and slot median 60 to 70 (L355) | `10 <= gap <= 20 and 60 <= sl <= 70`, inclusive (L185-188) | failed (gap 2.25, slot 49.05) | MATCH (gap = slot median - concept median, same quantity as 206.8) |
| 206.M3 | Qwen chat: the 16 maker/company neurons' median arming percentile between 45 and 55 (L356) | `45 <= idm <= 55` on `arming.identity.median_percentile` (L171-172) | held (45.97) | MATCH (the identity set is the 16 maker+company neurons, config206.json `qwen_identity_neurons`; inclusive bounds) |
| 206.M4 | Qwen chat, frame P: first-layer MLP token share 0.50-0.70, middle layer 0.10-0.25, and the first layer's with the log-frequency-explained part removed falls to 0.25-0.40 (L357) | `0.50 <= t0 <= 0.70 and 0.10 <= tm <= 0.25 and 0.25 <= tf <= 0.40` on `split.m[*].token_share` and `split.m[0].token_share_freq_removed` (L173-176) | failed (0.556 in range; 0.118 in range; freq-removed 0.543 out of range; output L95-97) | MATCH on the formula (`T(1-f)/(T+R)`, Design L193-194, analyze206.py:72-73). Same `m`-vs-`c` ambiguity as 206.4 (F3): on `c` the first layer is 0.323 and the row fails on clause 1 instead of clause 3 — same verdict. |
| 206.M5 | Qwen chat: at the layer of `rho_peak(A-tmpl,M)`, cos(`Delta(A-tmpl,M)` on `x_L`, frame M's `b_L`) >= 0.5 (L358, L201-203) | `cos_delta_bM[lay] >= 0.5` (L177-178); `cos_delta_bM` = cos(mean `D` over both halves of the (A-tmpl,M) pair, `b_M`) (analyze206.py:140, 149-151) | failed (-0.2507; output L98) | MATCH (Delta is the mean `x_L` difference, `b_L` is the mean of `x_L` in M — Design L202-203, L216-217; the layer is `rho_peak_layer` = 18) |

## Findings beyond the row table

- **F1 (interval provenance) — withdrawn, my error.** I first read `rho_peak_resamples` as empty because my probe
  looked up the wrong key; every OLMo stage's `common_mode["A-tmpl|M"]["rho_peak_resamples"]` holds 1000 values and
  `rho_peak_ci` is its bootstrap. 206.7's interval is the registered passage-resampled one. The scorer still has no
  explicit check that the resample list is non-empty (`C.bootstrap_ci([])` returns `[None, None]`,
  common206.py:395-399, which `adjacent_rise_rule` scores as `failed`, score206.py:57) — a robustness note only.
- **F2 (206.3 sign).** Covered in the table. `write_shares` (common206.py:265-269) returns a signed projection over
  the norm; the design's "writes at least half" (L344) and the headline's "adds nearly the same vector" are
  magnitude claims. A negative share passing a `>= 0.5` check is impossible, so no verdict flips, but the code does
  not test what the row says.
- **F3 (`m` vs `c`).** 206.4 and 206.M4 say "MLP output"/"MLP token share" and the code reads `split.m`. The method
  (L179) defines the split on both `m_L` and `c_L`; the part's aim (L26) is "how much of what MLPs write, and of what
  mixers write"; 206.5 explicitly contrasts `m` against `c`. The site table (L167-172) makes `m` the MLP's write, so
  `split.m` is the right reading — but the design never states it, and on `c` 206.4 would fail. Flag as ambiguity.
- **F4 ("between" inclusivity).** 206.M1, 206.M2, 206.M3, 206.M4 and 206.9 say "between/0.50 to 0.70" without saying
  inclusive. The code uses inclusive bounds throughout. No cited value sits on a bound, so no verdict depends on it;
  recorded so the reading is not silent.
- **F5 (206.8's p95 bar).** Amendment 4 (L31-35) records that the shuffle null is now over 39 neurons and its 95th
  percentile is about 17, so 206.8 now needs a gap above about 17 rather than the 10 the row states. The code
  enforces both (`gap >= 10 and gap > p95`), which is the registered Reading rule (L388-390); the amendment says the
  bar is left as registered. MATCH, noted because the effective bar is stricter than the row's wording.
- **F6 (206.10's population).** The row says "the maker token's log probability"; the code averages over the 23 rows
  where the plain model names its maker at top-1 (score206.py:102-104), which is Design L246-249's rule. The one
  dropped row (I found it in causal.json) would move arm_k1 from 0.0622 to 0.0796 over all 24 rows — the verdict
  (failed, on the M2M control) does not change. MATCH.
- **F7 (headline).** The printed sentence "the assistant frame changes each position in its own way; no constant
  shift at this resolution" (output L152) is the registered wording for "206.6's first two conditions fail"
  (Design L405-408); on OLMo Instruct `rho_peak` = 0.164 < 0.2, so it is correct. The code's `neither_set_moves`
  band (35-65) is the build's reading of an undefined phrase, already flagged in score206.py:208-209.

## BLOCKERS

- None blocking. The `m`-vs-`c` question (F3) is a design ambiguity, not a code bug; a person must say which
  sublayer "MLP output" means before 206.4's "held" can be called correct rather than merely plausible.

## SELF-ASSESSMENT

I did not run the scorer or the analysis; everything above is read from the code and the JSON the output cites. I
recomputed 206.10's means, 206.5's win count, 206.8's gap and p95 comparison, and the 206.7 / 206.12 rise tables by
hand from the summaries and they agree with the printed output. My first pass wrongly reported that the OLMo resamples were
missing (F1); I withdrew it after re-checking the key. I did not open `moments.npz`, so I cannot confirm the summaries
were not rewritten after scoring, but their timestamps (05:38-10:48) precede the scoring run (14:52, Amendment 5). The
`m`/`c` call in F3 is a judgement about intent, not a fact in the text.

---

## Adjudication (Claude, Opus 5.5, 2026-11-13 15:01 CT, checked against the design text and score206.py)

All four MISMATCH rows are refuted; the scorer matches the registered text on all 17 rows after Amendment 5.

- **206.5:** `wins >= 0.75 * len(first)` with six first-third layers is `wins >= 4.5`, which admits only 5 or 6, exactly the
  registered "three quarters or more". The report's "admits 4" is an arithmetic slip. MATCH. (Its win count is also
  wrong: 5 of 6, not 6 of 6; layer 0 has MLP 0.2699 against mixer 0.3232. Found by the write-up worker; held either way.)
- **206.4 / 206.M4:** the design's site list defines `m_L` as "what the MLP block adds to the stream" and `c_L` as "what the
  mixer block adds"; "MLP output" is `m`. No ambiguity. MATCH.
- **206.3:** the method defines each group's share as its (signed) projection on the unit vector of `r_L` over the norm of
  `r_L`; "writes at least half" is that quantity at >= 0.5, and a negative share writes against `r_L`. The code tests the
  defined quantity. MATCH (no verdict depends on it: -0.342 fails any reading).
- **206.11:** the design defines `rho_peak` once, for every pair, as "the maximum over layers in the middle third"; the code
  uses that for both pairs. MATCH (fails at either layer reading, as the report says).

Kept from the report: F1's robustness note (an empty resample list scores `failed` silently), F4 (inclusive bounds; no value
sits on one), F5 (206.8's effective bar is the shuffle p95, about 17, per Amendment 4), F6 (206.10 over 23 rows per the
design; 24 rows would not change the verdict). The worker wrote CHECK.md into its run directory rather than this folder;
known card `df-verify-satisfied-by-the-run-dir-not-the-domain`.
