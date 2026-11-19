# Part 212 report: every number the results README uses

Made by `part212/report212.py` from the files named on each line. `score.json` is the scorer's own output
(`part212/score212.py`) and is never edited here. Values recomputed with the scorer's own functions are
checked against it at the end; a `DIFF` line there means the two disagree.

## Verdicts (results/part212/stage2/score.json, key predictions.<n>)

212.1  verdict=FAILS
    rule: zero rate 12%-35% of 192 (broken<=0.15); unchanged <=3/192; zero full-arm share (count/192) > each random zero share (count/24)
    clauses: {"zero_rate_12_to_35": true, "zero_broken_le_0.15": true, "unchanged_le_3_of_192": false, "zero_share_exceeds_all_20": true}
212.2  verdict=FAILS
    rule: zero delta>=+1.5 and > all 20 random zero; -0.50 <= trace_fact < 0.50*zero
    clauses: {"zero_ge_+1.5": true, "zero_exceeds_all_20": true, "fact_ge_-0.50": true, "fact_lt_50pct_of_zero": false}
212.3  verdict=FAILS
    rule: sub_factual delta>=+0.75, ratio>=0.35, > 18/20 random sub_factual
    clauses: {"sub_ge_+0.75": false, "sub_retains_35pct": false, "exceeds_18_of_20_random_sub": true}
212.4  verdict=FAILS
    rule: trace_resample_fact broken<=0.15; clean_google<0.50*zero; clean_google+broken<0.50*zero; CI(zero-trace) lo>0
    clauses: {"trace_broken_le_0.15": true, "trace_rate_lt_50pct_zero": false, "worst_case_recode": false, "ci_lower_above_0": true}
212.5  verdict=FAILS
    rule: sub_factual - unchanged >= 0.05; full-arm share > 18/20 random sub shares; broken<=0.15
    clauses: {"sub_minus_unchanged_ge_0.05": false, "exceeds_18_of_20_random_sub": false, "sub_broken_le_0.15": true}
212.6  verdict=FAILS
    rule: trace_fact - trace_you >= +0.25; L6 N753 unit gap >= +0.10 and strictly largest
    clauses: {"fact_minus_you_ge_0.25": true, "L6_gap_ge_0.10": false, "L6_gap_strictly_largest": false}
212.7  verdict=HOLDS
    rule: Spearman(L27 N3325 mean shift, mean delta g_ali) over the 10 core arms <= -0.70
    clauses: {"spearman": -0.9878787878787879, "n_arms": 10, "l27_shift": {"unchanged": 0.0, "zero": -0.965, "trace_resample_fact": -0.711875, "trace_resample_you": -0.62, "mean_resample_fact": -0.736875, "sub_factual": -0.105, "scale_pos2": 0.22125, "scale_neg1": -0.8775, "zero_prompt_only": -0.22125, "zero_opening_only": -0.398125}, "delta_g_ali": {"unchanged": 0.0, "zero": 3.0965779887328004, "trace_resample_fact": 2.6011120587280647, "trace_resample_you": 2.2710930356483594, "mean_resample_fact": 2.6409000267060403, "sub_factual": 0.15069471804487905, "scale_pos2": -0.8706178704692883, "scale_neg1": 7.909389161059502, "zero_prompt_only": 0.2643674126341526, "zero_opening_only": 1.6954087137418696}}
212.8  verdict=HOLDS
    rule: scale_pos2 delta<=-0.30; scale_neg1 delta > zero delta
    clauses: {"scale_pos2_le_-0.30": true, "scale_neg1_gt_zero": true}
212.9  verdict=FAILS
    rule: rate(zero_gen_only) - rate(zero_prompt_only) >= 0.04
    clauses: {"zero_gen_only_rate": 0.052083333333333336, "zero_prompt_only_rate": 0.036458333333333336, "difference": 0.015625}
212.10  verdict=HOLDS
    rule: delta_g_ali(trace_resample_zh_maker) > -0.25
    clauses: {"zh_maker_delta": 0.7610847856305643, "zh_maker_minus_zh_fact": -1.5630169138761747, "zh_fact_delta": 2.324101699506739}
verdict counts: {'FAILS': 7, 'HOLDS': 3}  (counted from the verdicts above)
held: 212.7, 212.8, 212.10  (score.json, key predictions)
failed: 212.1, 212.2, 212.3, 212.4, 212.5, 212.6, 212.9  (score.json, key predictions)
headline reading: mixed / split across arms, units, or stages  (score.json, key headlines.reading)
withdraw_finding_1_specificity: true  (score.json, key headlines)

## Predictions table: registered text (PREDICTIONS.md, `## Part 212`), confidence, result, verdict

212.1 | confidence=high | author=Gemini 4 Neon | verdict=FAILS
    text: Stage 2 replication against maker-activation-write-norm-matched nulls: `zero` has clean Google rate between 12% and 35% of 192 held-out answers (`broken_rate <= 0.15`), `unchanged` has at most 3 of 192 (<= 1.6%), and `zero`'s full-arm share (`count / 192`) strictly exceeds each of the 20 matched random `zero` sets' shares (`count / 24`) (descriptive top rank; the equal-budget sample-0 rank on 24 answers is reported beside it and decides nothing; no uniform-null p-value claimed). Fails if any clause misses.
212.2 | confidence=medium | author=Gemini 4 Neon | verdict=FAILS
    text: Stage 1 factual-resample test (exact logits): `zero` raises mean `g_ali` by at least +1.5 logits over `unchanged` and exceeds all 20 matched random `zero` sets; AND `trace_resample_fact` stays in the near-baseline preservation band `-0.50 <= delta_g_ali(trace_resample_fact) < 0.50 * delta_g_ali(zero)` (strictly less than 50% of `zero`'s shift without a large negative reversal). Fails if any clause misses.
212.3 | confidence=medium | author=Gemini 4 Neon | verdict=FAILS
    text: Stage 1 factual-profile subtraction: frozen `sub_factual` ($a_{\text{clean}} - \bar{a}_{\text{fact}}$) raises mean `g_ali` by at least +0.75 logits, retains at least 35% of `zero`'s mean $\Delta g_{\text{ali}}$ ($\Delta g_{\text{ali}}(\text{sub\_factual}) / \Delta g_{\text{ali}}(\text{zero}) \ge 0.35$), and exceeds at least 18 of 20 matched random `sub_factual` sets. Fails if any clause misses.
212.4 | confidence=medium | author=Gemini 4 Neon | verdict=FAILS
    text: Stage 2 factual-resample test (free generation, with fluency validity): on the 192 held-out answers, `trace_resample_fact` has `broken_rate <= 0.15`, its clean Google rate is strictly less than half of `zero`'s clean Google rate (`ratio < 0.50`), the loss survives a worst-case recode that counts every broken `trace_resample_fact` answer as Google (`count(clean_google) + count(broken) < 0.50 * count_zero(clean_google)`), AND the paired question-clustered 95% bootstrap CI of `rate(zero) - rate(trace_resample_fact)` lies strictly above 0. Fails if any clause misses.
212.5 | confidence=medium | author=Gemini 4 Neon | verdict=FAILS
    text: Stage 2 factual-profile subtraction: on the 192 held-out answers, `sub_factual` exceeds `unchanged` in clean Google rate by at least 5 percentage points (`rate(sub_factual) - rate(unchanged) >= 0.05`), its full-arm share (`count / 192`) exceeds the shares (`count / 24`) of at least 18 of 20 matched random `sub_factual` sets (descriptive rank; the sample-0 rank is reported beside it and decides nothing), and has `broken_rate <= 0.15`. Fails if any clause misses.
212.6 | confidence=medium | author=Gemini 4 Neon | verdict=FAILS
    text: Factual vs. `"you"` donor-bank difference (Stage 1): `trace_resample_fact` exceeds `trace_resample_you` in mean $\Delta g_{\text{ali}}$ by at least +0.25 logits, AND in the single-unit 4x4 decomposition `L6 N753`'s `trace_resample_fact - trace_resample_you` gap is **strictly positive (`>= +0.10` logits)** and strictly larger than each of the other three units' (`L7 N1169`, `L11 N8868`, `L29 N776`). Fails if any clause misses.
212.7 | confidence=high | author=Gemini 4 Neon | verdict=HOLDS
    text: Downstream Google-writer routing: across the 10 core English intervention arms in Stage 1 (`unchanged`, `zero`, `trace_resample_fact`, `trace_resample_you`, `mean_resample_fact`, `sub_factual`, `scale_pos2`, `scale_neg1`, `zero_prompt_only`, `zero_opening_only`), each arm's mean last-position shift in `L27 N3325` correlates negatively with its mean $\Delta g_{\text{ali}}$ (Spearman $\rho \le -0.70$). Fails if $\rho > -0.70$.
212.8 | confidence=medium | author=Gemini 4 Neon | verdict=HOLDS
    text: Opposite-direction monotonicity (`STANDARD-CONTROLS.md` #1): in Stage 1, frozen `scale_pos2` ($2a_{\text{clean}}$) lowers mean `g_ali` relative to `unchanged` by at least 0.30 logits ($\Delta g_{\text{ali}} \le -0.30$), AND frozen `scale_neg1` ($-a_{\text{clean}}$) raises mean `g_ali` strictly more than `zero` ($\Delta g_{\text{ali}}(\text{scale\_neg1}) > \Delta g_{\text{ali}}(\text{zero})$). Fails if either clause misses.
212.9 | confidence=low | author=Gemini 4 Neon | verdict=FAILS
    text: Prompt vs. generation phase (`TOOLKIT.md` F): in Stage 2, `zero_gen_only`'s clean Google rate exceeds `zero_prompt_only`'s clean Google rate by at least 4 percentage points (`rate(zero_gen_only) - rate(zero_prompt_only) >= 0.04`). Fails below 0.04.
212.10 | confidence=low | author=Gemini 4 Neon | verdict=HOLDS
    text: Absolute Chinese-donor swap (Stage 1): patching Chinese maker-prompt activations (`trace_resample_zh_maker`) into English maker prompts gives mean $\Delta g_{\text{ali}} > -0.25$ logits relative to English `unchanged` (no-large-suppression prediction; the within-Chinese contrast `trace_resample_zh_maker - trace_resample_zh_fact` is reported beside it to separate maker framing from Chinese language). Fails if $\Delta g_{\text{ali}} \le -0.25$.

## Gates (score.json, key stage1.gates and stage2.gate3)

Gate 1 pass=True checks={"self_patch_within_0.05": true, "zero_non_finite": true, "mass_unchanged_ge_0.50": true, "mass_zero_ge_0.50": true}
    self_patch minus unchanged mean g_ali = 0.0000 logits (bar < 0.05)
    non-finite readouts = 0; mean 8-maker mass unchanged 1.0000, zero 1.0000 (bar >= 0.50)
Gate 2 pass=True checks={"zero_ge_+1.0": true, "exceeds_18_of_20": true}
    zero mean delta g_ali = 3.0966 (bar >= +1.0); exceeds 20 of 20 random zero sets (bar >= 18)
Gate 3 pass=True checks={"eager": true, "no_empty_rows": true, "no_non_finite_rows": true, "no_duplicate_keys": true, "every_planned_key_present": true, "gate3b_donors_L_gen_ge_4": true, "unchanged_guard_accuracy_ge_0.90": true, "unchanged_broken_rate_le_0.05": true}
    duplicate keys 0, empty rows 0, non-finite rows 0, short donors 0
    planned rows 3888, row keys 3888, missing planned keys 0, unplanned row keys 0
    unchanged guard accuracy 1.0000 (24 of 24)

## Run records (results/part212/stage2/run.json and pilot/run.json)

model Qwen/Qwen3.5-4B revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a  (run.json, key revision; the design pins the same)
load: attn eager (requested eager), dtype torch.bfloat16, device xpu, torch 2.13.0+xpu, transformers 5.13.0  (run.json, key load)
target units [[6, 753], [7, 1169], [11, 8868], [29, 776]]  (run.json, key target)
prompts SHA-256 0f10c337ae64765b9f1cc32a85241214944d1c2b55be15d8398a085042793227  (run.json, key prompts_sha256)
requested Stage 2 batch size 24  (run.json, key batch_size)
run.json generation_batch_histogram (complete calls) {"1": 1440, "4": 36, "8": 264, "24": 8}
scored rows by actual batch size {"1": 1440, "4": 144, "8": 2112, "24": 192}  (score.json, key stage2.batch_size_histogram)
planned rows 3888, new rows this invocation 3712  (run.json)
full Stage 2 wall time 8806.25 s = 2.45 hours  (run.json, key wall_time_seconds)
pilot run.json: stage pilot, batch 8, histogram {"4": 16, "8": 12}, planned 160, new rows 160
pilot stage1.json: forward_batch_histogram {"1": 3449}, non_finite_readouts 0, questions 25, arms 131, wall time 1148.74 s
donors: 25 d_fact, 25 d_you, 25 d_zh_maker; smallest L_gen 23 (Gate 3b bar >= 4)

## stage2.jsonl line accounting (results/part212/stage2/stage2.jsonl)

non-blank lines in stage2.jsonl: 3888  (every line is one JSON row; the scorer's loader skips blanks)
unique (set, arm, question_index, sample) keys: 3888; keys seen more than once: 0
rows with pilot=true in the full file: 0; foreign-shape rows the scorer dropped: 0
planned full rows (score212.stage2_rows(pilot=False)): 3888; planned pilot rows (stage2_rows(pilot=True)): 160
pilot/stage2.jsonl rows: 160 (pilot flag true on 160)
run.json planned_rows 3888 = full plan; run.json new_rows_this_invocation 3712 = the rows the 14:17 relaunch wrote (the 14:06 launch's rows were already on disk and resumed from)

## 212.1 zero vs unchanged vs the 20 random zero sets (Stage 2, held-out 24 prompts x 8 samples)

zero clean_google 36 of 192, rate 0.1875, broken 0 (rate 0.0000)
unchanged clean_google 4 of 192, rate 0.0208, broken 0; legacy_any_google 5 (decides nothing)
zero full-arm share 0.1875 exceeds 20 of 20 random zero shares, all=True
zero sample-0 rank (descriptive, decides nothing): 6 of 24, exceeds 20 of 20
random zero set counts (clean_google of 24): {"random_0": 2, "random_1": 2, "random_10": 0, "random_11": 0, "random_12": 1, "random_13": 0, "random_14": 1, "random_15": 1, "random_16": 0, "random_17": 0, "random_18": 0, "random_19": 0, "random_2": 1, "random_3": 1, "random_4": 0, "random_5": 0, "random_6": 1, "random_7": 0, "random_8": 0, "random_9": 0}
random zero set shares: {"random_0": 0.08333333333333333, "random_1": 0.08333333333333333, "random_10": 0.0, "random_11": 0.0, "random_12": 0.041666666666666664, "random_13": 0.0, "random_14": 0.041666666666666664, "random_15": 0.041666666666666664, "random_16": 0.0, "random_17": 0.0, "random_18": 0.0, "random_19": 0.0, "random_2": 0.041666666666666664, "random_3": 0.041666666666666664, "random_4": 0.0, "random_5": 0.0, "random_6": 0.041666666666666664, "random_7": 0.0, "random_8": 0.0, "random_9": 0.0}
clauses: {"zero_rate_12_to_35": true, "zero_broken_le_0.15": true, "unchanged_le_3_of_192": false, "zero_share_exceeds_all_20": true}

## Stage 2 main arms: five-state counts (denominator 192; score.json, key stage2.main_arms)

| arm | clean_google | clean_alibaba | clean_other_maker | clean_no_maker | broken | multi_hit | legacy_any_google |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mean_resample_fact` | 31 | 160 | 0 | 1 | 0 | 1 | 31 |
| `scale_neg1` | 110 | 13 | 3 | 66 | 0 | 1 | 110 |
| `scale_pos2` | 1 | 191 | 0 | 0 | 0 | 1 | 1 |
| `sub_factual` | 8 | 183 | 0 | 1 | 0 | 1 | 8 |
| `trace_resample_fact` | 19 | 171 | 0 | 2 | 0 | 1 | 19 |
| `trace_resample_you` | 9 | 180 | 0 | 3 | 0 | 0 | 9 |
| `trace_resample_zh_maker` | 9 | 182 | 0 | 1 | 0 | 1 | 9 |
| `unchanged` | 4 | 188 | 0 | 0 | 0 | 1 | 5 |
| `zero` | 36 | 147 | 2 | 7 | 0 | 3 | 36 |
| `zero_gen_only` | 10 | 181 | 0 | 1 | 0 | 2 | 10 |
| `zero_prompt_only` | 7 | 178 | 2 | 4 | 1 | 1 | 7 |

rate per 192 (clean_google / 192): `mean_resample_fact` 0.1615, `scale_neg1` 0.5729, `scale_pos2` 0.0052, `sub_factual` 0.0417, `trace_resample_fact` 0.0990, `trace_resample_you` 0.0469, `trace_resample_zh_maker` 0.0469, `unchanged` 0.0208, `zero` 0.1875, `zero_gen_only` 0.0521, `zero_prompt_only` 0.0365

## 212.2 and 212.4 trace_resample_fact (Stage 1 exact logits, Stage 2 free generation)

Stage 1 zero mean delta g_ali 3.0966; trace_resample_fact 2.6011; retention fact/zero 0.8400
Stage 1 trace_resample_you 2.2711; mean_resample_fact 2.6409; retention mean/zero 0.8528
Stage 1 zero exceeds 20 of 20 random zero sets
Stage 1 random zero set values: {"random_0:zero": -0.05104627625291741, "random_10:zero": 0.12358896013324895, "random_11:zero": 0.0011989332152424481, "random_12:zero": -0.17412371954096428, "random_13:zero": 0.04639839600829589, "random_14:zero": 0.04662406277570099, "random_15:zero": 0.01643313694671349, "random_16:zero": -0.10336879596029817, "random_17:zero": 0.1264902382586739, "random_18:zero": 0.06447648057850743, "random_19:zero": -0.05725297413721833, "random_1:zero": 0.06678769537660614, "random_2:zero": 0.05799341112407418, "random_3:zero": 0.03631714123554389, "random_4:zero": 0.01921841158745863, "random_5:zero": 0.026297804333278236, "random_6:zero": 0.04404231670660607, "random_7:zero": 0.13779166064964954, "random_8:zero": 0.16100413339522107, "random_9:zero": 0.09868658453736458}
Stage 1 half of zero's shift = 1.5483 (the 212.2 bar trace_resample_fact had to stay below)
Stage 2 trace/zero = 0.5278; mean/zero = 0.8611; sub/zero = 0.2222
Stage 2 trace_resample_fact clean_google 19 of 192, broken 0; half of zero 18.00; worst-case recode total 19
bootstrap zero - trace_resample_fact: estimate 0.0885, 95% CI [0.0104, 0.1719], 10000 draws, seed 212, clustered on question_index (24 questions)
bootstrap trace_resample_fact - unchanged: estimate 0.0781, 95% CI [0.0208, 0.1458]
clauses 212.2: {"zero_ge_+1.5": true, "zero_exceeds_all_20": true, "fact_ge_-0.50": true, "fact_lt_50pct_of_zero": false}
clauses 212.4: {"trace_broken_le_0.15": true, "trace_rate_lt_50pct_zero": false, "worst_case_recode": false, "ci_lower_above_0": true}

## 212.3 and 212.5 sub_factual

Stage 1 sub_factual mean delta g_ali 0.1507 (bar >= +0.75); ratio to zero 0.0487 (bar >= 0.35); sub_factual_online 0.1501
Stage 1 sub_factual exceeds 20 of 20 random sub_factual sets
clauses 212.3: {"sub_ge_+0.75": false, "sub_retains_35pct": false, "exceeds_18_of_20_random_sub": true}
Stage 2 sub_factual clean_google 8 of 192, rate 0.0417; unchanged rate 0.0208; difference 0.0208 (bar >= 0.05)
sub_factual full-arm share 0.0417 exceeds 16 of 20 random sub shares (bar >= 18); sample-0 rank 1 of 24 exceeds 16 (descriptive)
random sub_factual set counts (clean_google of 24): {"random_0": 0, "random_1": 0, "random_10": 0, "random_11": 1, "random_12": 0, "random_13": 0, "random_14": 0, "random_15": 0, "random_16": 0, "random_17": 0, "random_18": 0, "random_19": 0, "random_2": 0, "random_3": 0, "random_4": 0, "random_5": 0, "random_6": 1, "random_7": 1, "random_8": 1, "random_9": 0}
bootstrap sub_factual - unchanged: estimate 0.0208, 95% CI [0.0000, 0.0573]
clauses 212.5: {"sub_minus_unchanged_ge_0.05": false, "exceeds_18_of_20_random_sub": false, "sub_broken_le_0.15": true}

## 212.6 unit gaps (Stage 1, per-unit trace_resample_fact minus trace_resample_you)

trace_resample_fact - trace_resample_you (whole arm) = 0.3300 (bar >= +0.25)
    L6 N753: 0.0132
    L7 N1169: 0.0414
    L11 N8868: 0.0313
    L29 N776: -0.0522
per-unit mean delta g_ali (Stage 1, pilot/stage1.json, key arms.<unit>:<arm>.mean_delta_g_ali):
    L6 N753: zero 0.6583, trace_resample_fact 0.4846, trace_resample_you 0.4714, sub_factual 0.0489
    L7 N1169: zero 0.6455, trace_resample_fact 0.6160, trace_resample_you 0.5746, sub_factual -0.0000
    L11 N8868: zero 0.4758, trace_resample_fact 0.4959, trace_resample_you 0.4646, sub_factual 0.0139
    L29 N776: zero 0.0111, trace_resample_fact -0.0607, trace_resample_you -0.0085, sub_factual 0.0616
clauses 212.6: {"fact_minus_you_ge_0.25": true, "L6_gap_ge_0.10": false, "L6_gap_strictly_largest": false}

## 212.7 L27 N3325 shift against delta g_ali across the 10 core arms (Spearman)

    unchanged: L27 shift 0.0000, delta g_ali 0.0000
    zero: L27 shift -0.9650, delta g_ali 3.0966
    trace_resample_fact: L27 shift -0.7119, delta g_ali 2.6011
    trace_resample_you: L27 shift -0.6200, delta g_ali 2.2711
    mean_resample_fact: L27 shift -0.7369, delta g_ali 2.6409
    sub_factual: L27 shift -0.1050, delta g_ali 0.1507
    scale_pos2: L27 shift 0.2213, delta g_ali -0.8706
    scale_neg1: L27 shift -0.8775, delta g_ali 7.9094
    zero_prompt_only: L27 shift -0.2213, delta g_ali 0.2644
    zero_opening_only: L27 shift -0.3981, delta g_ali 1.6954
Spearman rho = -0.9879 over 10 arms (bar <= -0.70)

## 212.8 opposite-direction scaling (Stage 1)

scale_pos2 delta g_ali -0.8706 (bar <= -0.30); scale_neg1 7.9094; zero 3.0966
Stage 2 scale_pos2 clean_google 1 of 192; scale_neg1 110 of 192 (descriptive)
clauses 212.8: {"scale_pos2_le_-0.30": true, "scale_neg1_gt_zero": true}

## 212.9 zero_gen_only vs zero_prompt_only (Stage 2)

zero_gen_only clean_google 10 of 192, rate 0.0521; zero_prompt_only 7 of 192, rate 0.0365; difference 0.0156 (bar >= 0.04)
bootstrap zero_gen_only - zero_prompt_only: estimate 0.0156, 95% CI [-0.0365, 0.0625] (descriptive)
zero_prompt_only broken 1 (the only broken row in any main arm: 1 across all main arms)

## 212.10 Chinese-donor swap (Stage 1)

trace_resample_zh_maker delta g_ali 0.7611 (bar > -0.25); trace_resample_zh_fact 2.3241; zh_maker - zh_fact -1.5630 (descriptive)
Stage 2 trace_resample_zh_maker clean_google 9 of 192 (descriptive)
Stage 2 zero / unchanged ratio of clean Google counts = 9.0000; sub_factual / unchanged = 2.0000
Stage 2 zero_gen_only / zero_prompt_only ratio = 1.4286

## Factorial interaction, g_seq, perturbation norms, positive control

factorial interaction I = 0.3050 logits  (score.json, key stage1.factorial_interaction_I; = F(self_patch) - F(mean_resample_fact) - F(sub_factual) + F(zero), F = mean g_ali)
    F(self_patch) -3.0872, F(mean_resample_fact) -0.4463, F(sub_factual) -2.9365, F(zero) 0.0093
g_seq: zero 2.7504, trace_resample_fact 2.2503, sub_factual 0.2100; fact_retention 0.8182, sub_retention 0.0764
g_seq agreement passes: False checks {"fact_ge_-0.50": true, "fact_lt_50pct": false, "sub_ge_35pct": false}
realized perturbation norm zero: 0.8888
realized perturbation norm trace_resample_fact: 0.7263
realized perturbation norm trace_resample_you: 0.6628
realized perturbation norm mean_resample_fact: 0.7303
realized perturbation norm sub_factual: 0.2127
realized perturbation norm scale_pos2: 0.8888
realized perturbation norm scale_neg1: 1.7777
realized perturbation norm trace_resample_zh_maker: 0.4392
realized perturbation norm trace_resample_zh_fact: 0.7428
positive control "Who created you?" clean_google of 24 (score.json, key stage2.positive_control): `unchanged` 0, `zero` 10, `mean_resample_fact` 5, `sub_factual` 0, `scale_pos2` 0, `scale_neg1` 22, `trace_resample_fact` 6, `trace_resample_you` 0

## Amendment 1: the registered Stage 1 count and the batch shapes on record

the design's Size and Pilot sections say 3,989 Stage 1 forward passes; Amendment 1 (PART212-DESIGN.md) says that is an arithmetic error for 3,449; pilot/stage1.json forward_batch_histogram {"1": 3449} sums to 3449
run.json generation_batch_histogram (complete calls) {"1": 1440, "4": 36, "8": 264, "24": 8} sums to 1748 calls
scored rows by actual batch size {"1": 1440, "4": 144, "8": 2112, "24": 192} sums to 3888 rows
every row records batch_size_requested: [24] (the requested size is 24 on every row; the actual size is what the row was generated at)

## chain.log (results/part212/chain.log), as written

    12:17:02 step 1 start: -I -B part212/run212.py pilot --out results/part212/pilot --device xpu (ram_avail 15.9G)
    12:41:04 step 1 gpu: 210 samples, mean 35%, max 85%, busy(>5%) 82% of the time
    12:41:04 step 1 rc=1
    12:48:42 step 1 start: -I -B part212/run212.py pilot --out results/part212/pilot --device xpu (ram_avail 11.1G)
    12:49:23 step 1 start: -I -B part212/run212.py pilot --out results/part212/pilot --device xpu (ram_avail 11.8G)
    13:09:48 step 1 gpu: 176 samples, mean 47%, max 86%, busy(>5%) 99% of the time
    13:09:48 step 1 RAM GUARD: Windows available RAM 1.92G < 2.0G, step killed
    13:09:48 step 1 rc=97
    13:19:27 step 1 start: -I -B part212/run212.py pilot --out results/part212/pilot --device xpu (ram_avail 11.6G)
    13:49:09 step 1 gpu: 258 samples, mean 52%, max 91%, busy(>5%) 94% of the time
    13:49:09 step 1 rc=1
    14:06:41 step 1 start: -I -B part212/run212.py stage2 --out results/part212/stage2 --device xpu (ram_avail 14.7G)
    14:16:42 step 1 gpu: 87 samples, mean 65%, max 91%, busy(>5%) 80% of the time
    14:16:42 step 1 rc=3221225477
    14:17:40 step 1 start: -I -B part212/run212.py stage2 --out results/part212/stage2 --device xpu (ram_avail 14.6G)
    16:50:33 step 1 gpu: 1350 samples, mean 78%, max 92%, busy(>5%) 99% of the time
    16:50:33 step 1 rc=0

## pilot stopping rule (results/part212/pilot/pilot.json)

checks {"gate1": true, "gate2": true, "gate3b": true, "no_empty_rows": true, "eager_recorded": true, "zero_positive_google_ge_2_of_24": true}; zero_positive_google 10; launch_full_stage2 True
computed_on: host (WSL, CPU), 2026-11-19, by the fixed pilot_stopping_rule from this folder's stage1.json, run.json and stage2.jsonl; the A770 run wrote all 160 pilot rows, then stopped at the stopping rule's bare `import score212` under python -I (PART212-DESIGN.md, Amendment 2)

## Checks: recomputations against score.json

held now: ['212.10', '212.7', '212.8']
OK   212.7 spearman: recomputed -0.9879, score.json -0.9879
OK   factorial interaction I: recomputed 0.3050, score.json 0.3050

