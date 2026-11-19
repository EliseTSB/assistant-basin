# Part 212: zeroing Qwen3.5-4B's four maker neurons makes it name Google in 36 of 192 answers, and giving them values from factual prompts still does it in 19; 3 of 10 predictions held

![Part 212: (a) clean Google answers out of 192 for each Stage 2 arm, with the 20 matched random zero sets beside zero and the 20 random sub_factual sets beside sub_factual; (b) Stage 1 mean delta g_ali for each main arm, with the 20 random zero sets beside zero](fig212.png)

Part 212 asked whether Qwen3.5-4B's four maker neurons carry maker-specific evidence, or a factual-baseline scaffold whose removal exposes a Google fallback. Zeroing the four units (`L6 N753`, `L7 N1169`, `L11 N8868`, `L29 N776`) makes the model name Google in 36 of 192 held-out answers, against 4 of 192 for the plain model and at most 2 of 24 for each of 20 matched random zero sets. But the plain model's 4 is above the 3 that 212.1 allowed, so 212.1 failed on its baseline clause, and the design's withdrawal rule takes the specificity premise with it. Replacing the units' activations with values from factual prompts still gives 19 Google answers, against zeroing's 36, and 2.60 of zeroing's 3.10 logits. A reading, not a finding: giving the units factual values brings back much of zeroing's Google effect, so it is not only the units' absence that matters. Three predictions held (212.7, 212.8, 212.10) and seven failed. The scorer's headline reading is `mixed / split across arms, units, or stages`. Scope: Qwen3.5-4B at one snapshot, one four-unit set, Part 66's 24 English held-out phrasings, eager attention in bf16 on the Arc A770, and sampled generation at temperature 0.7 for 48 tokens.

## Predictions

The text and confidence are copied verbatim from the `## Part 212` section of [PREDICTIONS.md](../../PREDICTIONS.md), where all ten were registered on 2026-11-19 01:24 CT before any data; all are by Gemini 4 Neon. The result is the number from [score.json](stage2/score.json), printed by [report212.py](../../part212/report212.py) into [report212.md](report212.md). For each failure the clause or clauses that missed are named from `score.json`'s `clauses`.

| # | Prediction (verbatim, with its confidence) | Result | Verdict |
| --- | --- | --- | --- |
| 212.1 | Gemini 4 Neon (high confidence): Stage 2 replication against maker-activation-write-norm-matched nulls: `zero` has clean Google rate between 12% and 35% of 192 held-out answers (`broken_rate <= 0.15`), `unchanged` has at most 3 of 192 (<= 1.6%), and `zero`'s full-arm share (`count / 192`) strictly exceeds each of the 20 matched random `zero` sets' shares (`count / 24`) (descriptive top rank; the equal-budget sample-0 rank on 24 answers is reported beside it and decides nothing; no uniform-null p-value claimed). Fails if any clause misses. | zero 36 of 192 (rate 0.1875, broken 0); unchanged 4 of 192; zero's share 0.1875 exceeds all 20 random zero shares (highest 2 of 24). Failed clause: `unchanged_le_3_of_192` (4 > 3). | failed |
| 212.2 | Gemini 4 Neon (medium confidence): Stage 1 factual-resample test (exact logits): `zero` raises mean `g_ali` by at least +1.5 logits over `unchanged` and exceeds all 20 matched random `zero` sets; AND `trace_resample_fact` stays in the near-baseline preservation band `-0.50 <= delta_g_ali(trace_resample_fact) < 0.50 * delta_g_ali(zero)` (strictly less than 50% of `zero`'s shift without a large negative reversal). Fails if any clause misses. | zero +3.0966, above all 20 random zero sets (highest +0.1610); trace_resample_fact +2.6011, against a bar of 1.5483 (half of zero). Failed clause: `fact_lt_50pct_of_zero`. | failed |
| 212.3 | Gemini 4 Neon (medium confidence): Stage 1 factual-profile subtraction: frozen `sub_factual` ($a_{\text{clean}} - \bar{a}_{\text{fact}}$) raises mean `g_ali` by at least +0.75 logits, retains at least 35% of `zero`'s mean $\Delta g_{\text{ali}}$ ($\Delta g_{\text{ali}}(\text{sub\_factual}) / \Delta g_{\text{ali}}(\text{zero}) \ge 0.35$), and exceeds at least 18 of 20 matched random `sub_factual` sets. Fails if any clause misses. | sub_factual +0.1507, ratio 0.0487 to zero, above all 20 random sub_factual sets. Failed clauses: `sub_ge_+0.75` and `sub_retains_35pct`. | failed |
| 212.4 | Gemini 4 Neon (medium confidence): Stage 2 factual-resample test (free generation, with fluency validity): on the 192 held-out answers, `trace_resample_fact` has `broken_rate <= 0.15`, its clean Google rate is strictly less than half of `zero`'s clean Google rate (`ratio < 0.50`), the loss survives a worst-case recode that counts every broken `trace_resample_fact` answer as Google (`count(clean_google) + count(broken) < 0.50 * count_zero(clean_google)`), AND the paired question-clustered 95% bootstrap CI of `rate(zero) - rate(trace_resample_fact)` lies strictly above 0. Fails if any clause misses. | trace_resample_fact 19 of 192, broken 0; half of zero is 18; recode total 19; bootstrap CI of zero minus trace [0.0104, 0.1719]. Failed clauses: `trace_rate_lt_50pct_zero` (19 is not below 18) and `worst_case_recode` (19 is not below 18). | failed |
| 212.5 | Gemini 4 Neon (medium confidence): Stage 2 factual-profile subtraction: on the 192 held-out answers, `sub_factual` exceeds `unchanged` in clean Google rate by at least 5 percentage points (`rate(sub_factual) - rate(unchanged) >= 0.05`), its full-arm share (`count / 192`) exceeds the shares (`count / 24`) of at least 18 of 20 matched random `sub_factual` sets (descriptive rank; the sample-0 rank is reported beside it and decides nothing), and has `broken_rate <= 0.15`. Fails if any clause misses. | sub_factual 8 of 192, unchanged 4 of 192, difference 0.0208; sub_factual's share 0.0417 exceeds 16 of 20 random sub shares. Failed clauses: `sub_minus_unchanged_ge_0.05` and `exceeds_18_of_20_random_sub`. | failed |
| 212.6 | Gemini 4 Neon (medium confidence): Factual vs. `"you"` donor-bank difference (Stage 1): `trace_resample_fact` exceeds `trace_resample_you` in mean $\Delta g_{\text{ali}}$ by at least +0.25 logits, AND in the single-unit 4x4 decomposition `L6 N753`'s `trace_resample_fact - trace_resample_you` gap is **strictly positive (`>= +0.10` logits)** and strictly larger than each of the other three units' (`L7 N1169`, `L11 N8868`, `L29 N776`). Fails if any clause misses. | whole arm +0.3300; unit gaps L6 N753 +0.0132, L7 N1169 +0.0414, L11 N8868 +0.0313, L29 N776 -0.0522. Failed clauses: `L6_gap_ge_0.10` and `L6_gap_strictly_largest`. | failed |
| 212.7 | Gemini 4 Neon (high confidence): Downstream Google-writer routing: across the 10 core English intervention arms in Stage 1 (`unchanged`, `zero`, `trace_resample_fact`, `trace_resample_you`, `mean_resample_fact`, `sub_factual`, `scale_pos2`, `scale_neg1`, `zero_prompt_only`, `zero_opening_only`), each arm's mean last-position shift in `L27 N3325` correlates negatively with its mean $\Delta g_{\text{ali}}$ (Spearman $\rho \le -0.70$). Fails if $\rho > -0.70$. | Spearman -0.9879 over 10 arms. | held |
| 212.8 | Gemini 4 Neon (medium confidence): Opposite-direction monotonicity (`STANDARD-CONTROLS.md` #1): in Stage 1, frozen `scale_pos2` ($2a_{\text{clean}}$) lowers mean `g_ali` relative to `unchanged` by at least 0.30 logits ($\Delta g_{\text{ali}} \le -0.30$), AND frozen `scale_neg1` ($-a_{\text{clean}}$) raises mean `g_ali` strictly more than `zero` ($\Delta g_{\text{ali}}(\text{scale\_neg1}) > \Delta g_{\text{ali}}(\text{zero})$). Fails if either clause misses. | scale_pos2 -0.8706; scale_neg1 +7.9094 against zero +3.0966. | held |
| 212.9 | Gemini 4 Neon (low confidence): Prompt vs. generation phase (`TOOLKIT.md` F): in Stage 2, `zero_gen_only`'s clean Google rate exceeds `zero_prompt_only`'s clean Google rate by at least 4 percentage points (`rate(zero_gen_only) - rate(zero_prompt_only) >= 0.04`). Fails below 0.04. | zero_gen_only 10 of 192, zero_prompt_only 7 of 192, difference 0.0156. | failed |
| 212.10 | Gemini 4 Neon (low confidence): Absolute Chinese-donor swap (Stage 1): patching Chinese maker-prompt activations (`trace_resample_zh_maker`) into English maker prompts gives mean $\Delta g_{\text{ali}} > -0.25$ logits relative to English `unchanged` (no-large-suppression prediction; the within-Chinese contrast `trace_resample_zh_maker - trace_resample_zh_fact` is reported beside it to separate maker framing from Chinese language). Fails if $\Delta g_{\text{ali}} \le -0.25$. | trace_resample_zh_maker +0.7611; the within-Chinese contrast zh_maker minus zh_fact is -1.5630. | held |

The verdicts are the scorer's, from [score.json](stage2/score.json) key `predictions`. Three held (212.7, 212.8, 212.10) and seven failed (212.1, 212.2, 212.3, 212.4, 212.5, 212.6, 212.9). Gates 1, 2 and 3 passed. The headline reading is the design's fallback, `mixed / split across arms, units, or stages`. The design's scaffold headline needs 212.1 to 212.5 all to hold, the `g_seq` agreement rule, and `mean_resample_fact` inside the 50% band in both stages; none of those holds. The evidence headline needs 212.1, which failed. The design's withdrawal rule applies: 212.1 failed, so the Finding 1 specificity premise is withdrawn for this control family, and that is reported first below.

## Does zeroing the four maker neurons make the model name Google, beyond a push of the same size?

Yes, far beyond any random push, and the prediction still failed. Zeroing the four units gives a clean Google answer on 36 of 192 held-out answers. Each of the 20 matched random zero sets (four units, one per layer, each drawn at random from the 50 units in its layer closest to the target unit in maker-context write norm) gives clean Google on 0, 1 or 2 of its 24 answers. Zeroing's share, 0.1875, is above all 20. The plain model gives clean Google on 4 of 192. The registered bar was at most 3, so the `unchanged` clause missed by one answer and 212.1 failed. The failure is about the baseline, not about zeroing: the effect is large and clear, and the specificity claim the prediction was built to test does not survive its own control. The design says so itself: if 212.1 fails, the Finding 1 specificity premise is withdrawn for this control family.

| Arm | Clean Google of 192 | Broken |
| --- | ---: | ---: |
| `unchanged` | 4 | 0 |
| `zero` | 36 | 0 |
| `zero`, 20 random sets | 0 to 2 of 24 each | 0 |
| `trace_resample_fact` | 19 | 0 |
| `mean_resample_fact` | 31 | 0 |
| `sub_factual` | 8 | 0 |
| `sub_factual`, 20 random sets | 0 to 1 of 24 each | 0 |

All four of the plain model's Google answers come from indirect phrasings, and they name Gemini or Google as the maker (for example "You should write to **Google**. I am **Gemini**, a large language model trained by Google."). The scorer's legacy any-Google count for `unchanged` is 5: the fifth is an Alibaba answer that says it is "**not** developed by Google", which the exclusive count files as Alibaba. The legacy count decides nothing. Source: `stage2.main_arms` and `stage2.ranks` in [score.json](stage2/score.json), printed in [report212.md](report212.md) by [report212.py](../../part212/report212.py).

## Is it the missing neurons, or knocking them off their usual values?

Factual values bring back much of it, but not all. Replacing the four units' activations with the paired values from factual prompts (`trace_resample_fact`) gives 19 Google answers in Stage 2, against zeroing's 36, and a Stage 1 shift of 2.6011 logits against zeroing's 3.0966. Replacing them with the factual bank mean (`mean_resample_fact`) gives 31. Other ways of moving the units off their maker values do much less in Stage 2: values from "you" prompts (`trace_resample_you`) give 9 and subtracting the factual profile (`sub_factual`) gives 8, against the plain model's 4. Neon bet the other way: 212.2 and 212.4 predicted that factual values would keep the answer near the plain model's, and both failed because factual values kept more than half of zeroing's effect. A reading, not a finding: what matters is partly which values the units hold, not only whether they are zeroed.

| Measure | `zero` | `trace_resample_fact` | `mean_resample_fact` | `sub_factual` |
| --- | ---: | ---: | ---: | ---: |
| Stage 1 mean delta g_ali | 3.0966 | 2.6011 | 2.6409 | 0.1507 |
| Stage 2 clean Google of 192 | 36 | 19 | 31 | 8 |
| Stage 2 share of zero's count | 1.0000 | 0.5278 | 0.8611 | 0.2222 |

212.2 and 212.4 both failed because the factual resample kept more than half. In Stage 1 the bar was 1.5483 logits, half of zero's 3.0966, and the trace gave 2.6011. In Stage 2 the bar was 18 of 192, half of 36, and the trace gave 19. The paired question-clustered bootstrap interval of the difference between zero and trace is 0.0104 to 0.1719 over 10,000 draws with seed 212, with an estimate of 0.0885, so zeroing does still give more Google answers than the factual resample, by about 9 percentage points. The interval of trace minus unchanged is 0.0208 to 0.1458, so the factual resample also sits above the plain model. Source: `stage1` and `stage2.bootstrap` in [score.json](stage2/score.json).

## Is there maker-specific evidence after subtracting the factual profile?

No. Subtracting the factual bank profile from the units' clean activations (`sub_factual`) leaves almost nothing of zeroing's shift. In Stage 1 it gives +0.1507 logits against zero's +3.0966, a ratio of 0.0487, and the registered bars were +0.75 and 0.35. In Stage 2 it gives 8 Google answers of 192 against the plain model's 4, a difference of 0.0208 where 212.5 asked for at least 0.05, and its share beats only 16 of 20 random sub_factual sets where 212.5 asked for 18. The online variant `sub_factual_online` gives +0.1501, so the frozen and online subtractions agree. The one clause that held is that sub_factual exceeds all 20 random sub_factual sets in Stage 1. The bootstrap interval of sub_factual minus unchanged is 0.0000 to 0.0573, which includes 0.

| Arm | Stage 1 delta g_ali | Stage 2 clean Google of 192 |
| --- | ---: | ---: |
| `zero` | 3.0966 | 36 |
| `sub_factual` | 0.1507 | 8 |
| `sub_factual_online` | 0.1501 | not run in Stage 2 |
| `unchanged` | 0.0000 | 4 |

Source: `stage1.sub_factual_delta_g_ali`, `stage1.sub_factual_online_delta_g_ali` and `stage2.main_arms` in [score.json](stage2/score.json).

## Which unit carries it?

No single unit carries the factual-versus-"you" difference. The whole-arm gap between `trace_resample_fact` and `trace_resample_you` is +0.3300 logits, which clears the +0.25 bar. But the single-unit gaps are small: `L6 N753` +0.0132, `L7 N1169` +0.0414, `L11 N8868` +0.0313 and `L29 N776` -0.0522. `L6 N753` was the unit 212.6 expected to carry it, at +0.10 or more and larger than the other three, and it is neither. The design records that the two banks differ in request shape as well as the pronoun, so the whole-arm gap may be a bank difference rather than an isolated effect of "you".

The downstream writer does follow the maker logit across arms. Across the 10 core English arms, each arm's mean last-position shift in `L27 N3325` against its mean delta g_ali gives a Spearman correlation of -0.9879, against a bar of at most -0.70, so 212.7 held. The arm that moves g_ali most, `scale_neg1` at +7.9094, has the L27 shift of -0.8775; the arm that lowers g_ali, `scale_pos2` at -0.8706, has a shift of +0.2213.

| Unit | trace_resample_fact | trace_resample_you | Gap |
| --- | ---: | ---: | ---: |
| `L6 N753` | 0.4846 | 0.4714 | +0.0132 |
| `L7 N1169` | 0.6160 | 0.5746 | +0.0414 |
| `L11 N8868` | 0.4959 | 0.4646 | +0.0313 |
| `L29 N776` | -0.0607 | -0.0085 | -0.0522 |

Source: `stage1.unit_fact_minus_you` and `predictions.212.7` in [score.json](stage2/score.json); the per-unit values from `arms.<unit>:<arm>.mean_delta_g_ali` in [pilot/stage1.json](pilot/stage1.json).

## Does the sign matter?

Yes, and 212.8 held. Scaling the units' activations by -1 (`scale_neg1`) raises mean g_ali by +7.9094, more than twice zeroing's +3.0966. Scaling them by +2 (`scale_pos2`) lowers it by -0.8706, against a bar of at most -0.30. So the direction of the push moves the maker logit in the direction the sign implies, which a generic damage effect would not do.

| Arm | Stage 1 delta g_ali | Stage 2 clean Google of 192 |
| --- | ---: | ---: |
| `scale_neg1` | 7.9094 | 110 |
| `zero` | 3.0966 | 36 |
| `scale_pos2` | -0.8706 | 1 |

The Stage 2 counts are descriptive, because 212.8 is a Stage 1 prediction. Source: `predictions.212.8` and `stage2.main_arms` in [score.json](stage2/score.json).

## Is it the prompt or the generation?

Neither phase carries it on its own. Zeroing only the prompt positions (`zero_prompt_only`) gives 7 Google answers of 192 and zeroing only the generation positions (`zero_gen_only`) gives 10, against 36 for zeroing everywhere. The difference is 0.0156, and 212.9 asked for at least 0.04, so it failed. The bootstrap interval of the difference is -0.0365 to 0.0625, which includes 0. The plain model gives 4, so both single-phase arms sit a little above it and far below full zeroing. The design notes that `zero_gen_only` leaves the first answer token unzeroed, because that token is predicted by the prefill call.

| Arm | Stage 2 clean Google of 192 |
| --- | ---: |
| `zero` | 36 |
| `zero_gen_only` | 10 |
| `zero_prompt_only` | 7 |
| `unchanged` | 4 |

Source: `stage2.main_arms` and `stage2.bootstrap` in [score.json](stage2/score.json).

## Do the units' values from Chinese maker prompts push the English answer toward Alibaba?

No, and 212.10 held. `g_ali` is Google over Alibaba, so a positive delta is a shift toward Google. Patching the units' activations from Chinese maker prompts into the English maker prompts (`trace_resample_zh_maker`) gives a mean delta g_ali of +0.7611, a small shift toward Google, and the registered bar was only that it not fall to -0.25 or below. The within-Chinese contrast, zh_maker minus zh_fact, is -1.5630: Chinese factual values (+2.3241) shift the answer toward Google much more than Chinese maker values do, as the English factual values do. In Stage 2 the same arm gives 9 clean Google answers of 192, descriptive. Source: `predictions.212.10` and `stage2.main_arms` in [score.json](stage2/score.json).

## Method

The design is [PART212-DESIGN.md](../../PART212-DESIGN.md) v3.1, registered 2026-11-19 01:24 CT. The model is Qwen/Qwen3.5-4B at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, loaded in `torch.bfloat16` on the Intel Arc A770 (`xpu`) with `attn_implementation="eager"`, thinking off and no system prompt, as [run.json](stage2/run.json) records. The four units are `L6 N753`, `L7 N1169`, `L11 N8868` and `L29 N776`, intervened on the `down_proj` input, with `L24 N3949` and `L27 N3325` recorded downstream. Stage 1 runs 131 arms on 25 prompts as single forward passes at the forced opening `'I was developed by **'` and reads exact logits; `g_ali` is the Google logit minus the logsumexp of the Alibaba and Tongyi first tokens, and every arm's delta is paired against `unchanged`. Stage 2 generates 3,888 answers by sampling with temperature 0.7, top p 0.95 and 48 tokens, one prompt per batch, with 192 held-out answers per full arm. The 20 matched random sets are four units each, one per layer, each drawn uniformly from the 50 units in that layer closest to the target unit in maker-context write norm, so a set comes from a narrow write-norm band around each target rather than being matched to it unit by unit. Every interval is a 10,000-draw question-clustered bootstrap with seed 212. The factorial interaction I is 0.3050 logits, from F(self_patch) - F(mean_resample_fact) - F(sub_factual) + F(zero) with F the mean g_ali; it measures how far the four cells of the zero, factual-resample, subtract and clean intervention depart from adding up, so a value near 0 would mean the cells combine as separate pieces, though the design warns that a logit-scale interaction is not a literal fraction of stored content.

## Things that happened during the run

**Amendment 1, before any data.** The registered batch sizes were not what the runner does. The design assumed Stage 2 at batch 8 or 24 with one prompt per batch, but the random control sets carry one sample per prompt, so each of those rows is a group of one. The run proceeds at the shapes it records. The full plan's Stage 2 histogram is `{'1': 1440, '4': 36, '8': 264, '24': 8}` complete calls in [run.json](stage2/run.json), which is 1,748 calls and 3,888 rows, and the scorer counts the scored rows (not calls) by actual batch size as `{'1': 1440, '4': 144, '8': 2112, '24': 192}`. The design's "3,989" Stage 1 passes was an arithmetic error for 3,449, which is what [pilot/stage1.json](pilot/stage1.json) records as `{"1": 3449}`.

**The pilot took four launches** ([chain.log](chain.log), times Central). The first ran 12:17 to 12:41 and exited with code 1 after Stage 1 finished: the runner refused to start Stage 2 because its output folder was locked by its own process, and on Windows the lock file was written in text mode with `\r\n`, so the release step compared different bytes and the lock outlived Stage 1. That was fixed in the runner at commit `30db7048`. The second started at 12:48 and was stopped by Journal Claude a minute in, because the stopping rule's module had not been copied to the job folder. The third ran 12:49 to 13:09 and exited with code 97: a second model load in one job tripped the job runner's RAM guard at 1.92G available against a 2.0G floor, and that was fixed by loading the model once at commit `4e98ffb1`. The fourth ran 13:19 to 13:49, wrote every row and stopped at a bare `import score212`, which is Amendment 2.

**Amendment 2, after the pilot and before full Stage 2 data.** The stopping rule passed all six checks on the host ([pilot.json](pilot/pilot.json)): Gate 1, Gate 2 with zero's mean delta g_ali at 3.10, Gate 3b, no empty row, eager recorded, and zero giving a clean Google answer on 10 of 24 "Who created you?" samples against a bar of 2. The pilot's plan held no batch-1 call, so the batch-1 rate was projected rather than measured, and the full run carried a live stop rule. Journal Claude recorded the first 100 batch-1 calls at 4.80 s per call against a forecast of 4.84 during the run; that figure is recorded during the run and is not in the result files.

**Full Stage 2's first launch** started at 14:06 and exited ten minutes in with Windows code 3221225477 (0xC0000005, an access violation). It was relaunched once at 14:17, resumed from the rows already written, and finished at 16:50 with exit code 0, with the job runner recording mean GPU use of 78% over 1,350 samples. The cause of the access violation is not established.

**`stage2.jsonl` holds 3,888 lines**, which is more than the 3,712 rows [run.json](stage2/run.json) records as `new_rows_this_invocation`. The 3,712 is what the 14:17 relaunch itself wrote; the 176 rows written before the 14:06 crash were already on disk and were resumed from, not rewritten. The file holds 3,888 distinct keys, none repeated, and no non-row records, and Gate 3's `no_duplicate_keys` and `every_planned_key_present` both pass. The file's 3,888 rows are the full plan of `stage2_rows(pilot=False)`; the pilot's 160 rows are in a separate file.

**Reviews before the run.** Alden (council) held v2 on four points and passed the folded design; the four are folded in v2 and v3 of the design, and v3.1 is Alden's scoped pass on v3. Luna code reviews ran in three rounds (`part212/REVIEW-code-luna-r2.md`, `-r3.md`). GLM 5.3's code review found the `zero` arm zeroing nothing in the first build, recorded as F1 in [BUILD-REPORT.md](../../part212/BUILD-REPORT.md) and fixed before the run.

## Limits

- One model, Qwen3.5-4B at one snapshot, one quantisation (bf16), and one four-unit set. Nothing here says how often other units or other models do this.
- The prompt families are the registered ones: Part 66's 24 English held-out maker phrasings, one positive control and six fluency guards. The `D_fact` and `D_you` banks match token length and the neutral opening but not request shape, and `EN_MAKER` uses a different forced opening from the donors, so opening semantics are not held constant.
- Stage 2 is sampled at temperature 0.7 with top p 0.95, 48 tokens, with no beam search or reranking. Batch shape can move an answer at an exact bf16 tie (Part 201), and this run used batch sizes 1, 4, 8 and 24.
- The plain model's own Google rate is 4 of 192, and the registered bar for 212.1 was 3. The specificity claim is withdrawn for this control family, and the README does not read the 36 against 4 as a pass.
- The random controls are drawn from a narrow band of maker-context zeroing write norm around each target unit, not matched to it unit by unit, and not on the realized perturbation norm. The realized norms differ by arm: 0.8888 for zero, 0.7263 for trace_resample_fact, 0.7303 for mean_resample_fact and 0.2127 for sub_factual, so the subtraction arm moves the residual stream far less than the others.
- The factorial interaction I = 0.3050 is on the logit scale and is not a fraction of stored content.
- The batch-1 rate was projected in Amendment 2 and only measured during the run; the 4.80 s per call figure is from the run's record, not from a result file.
- The cause of the 0xC0000005 access violation is not established.

## Files

- [stage2/score.json](stage2/score.json): every gate, prediction clause, interval, table and headline; made by [score212.py](../../part212/score212.py) from the files below.
- [stage2/run.json](stage2/run.json): the full Stage 2 manifest, with the revision, the load, the batch histogram, the donors' `L_gen` and the random sets; written by [run212.py](../../part212/run212.py).
- [stage2/stage2.jsonl](stage2/stage2.jsonl): the 3,888 Stage 2 rows; written by [run212.py](../../part212/run212.py).
- [pilot/stage1.json](pilot/stage1.json): the 131 Stage 1 arms on 25 prompts, the source of every Stage 1 logit on this page; written by [run212.py](../../part212/run212.py).
- [pilot/run.json](pilot/run.json), [pilot/stage2.jsonl](pilot/stage2.jsonl), [pilot/pilot.json](pilot/pilot.json): the pilot's manifest, its 160 rows and the stopping rule's six checks.
- [chain.log](chain.log): the job runner's launch lines for the pilot and full Stage 2.
- [report212.md](report212.md): every number on this page and the tables behind it; made by [report212.py](../../part212/report212.py) from the files above and checked against `score.json`.
- [fig212.png](fig212.png): the figure above; made by [fig212.py](../../part212/fig212.py) from `score.json` and `pilot/stage1.json`.

Cold check: DeepSeek V4.1 Flash, 2026-11-19, a fresh reader with only the files (df run part212-readme-coldcheck-2026-11-19). It recomputed every Stage 2 count, the four bootstrap intervals, every Stage 1 delta, the unit gaps, the Spearman and I from the raw records, and found all of them, all 10 prediction rows and the sign convention correct. Of its prose findings, five are folded (the factual-values reading narrowed to the factual arms and set against "you" and sub_factual, the reading labelled where it was stated as fact, calls and rows told apart, the bank-difference caveat kept as a caveat, the random sets described as drawn from a band rather than matched) and one figure caption is clarified; one is declined (it counted five pilot launches where chain.log has four start lines). No verdict changed.
