# Front-page result-file audit

The claim IDs below point to the exact front-page quotes in `claims.tsv`. All listed MATCH and MISMATCH arithmetic was run with `python3 -I`; only the standard library was used. The scripts did not load a model.

## Verdict counts

| Verdict | Claims |
|---|---:|
| MATCH | 95 |
| MISMATCH | 1 |
| NOT-FOUND | 6 |
| NOT-RECOMPUTABLE | 68 |
| **Total** | **170** |

## MISMATCH

| Claim | Recomputed result |
|---|---|
| C115 | In `results/part82/main/records.jsonl`, `recompute_extra.py` gives the target six-head set a point-estimate reduction of 0.368506 (h0 0.444139 to h1 0.075632). Random set c3 reduces the gap by 0.348689, which is less; c5 reduces it by 0.502094, which is more. Thus **one of five**, not two, cuts as much or more under the literal point-estimate comparison. The Part 82 README intervals overlap for c3 and may motivate its “two” wording, but this audit did not recompute those intervals. |

## NOT-FOUND

| Claim | Search result |
|---|---|
| C133 | No result summary or identifiable row subset in `results/part34/` reproduces 1,239/1,240. The ten `readout-fold3` shards are present; the 1,240 M parity-word raise rows inspected yield 1,179 A replies. The claimed numerator/denominator could not be tied to a different raw subset. |
| C163–C167 | The checkout has no `results/part37/` directory or other Part 37 result file holding the AUROCs, interval, yes-probability, or wording-condition probabilities. |

## MATCH

Each row gives the individual claim IDs receiving MATCH, the recomputation, its result file, and the script that produced it. `claims.tsv` holds the exact quote for every ID.

| Claim(s) | Recomputed result | Result file(s) | Script |
|---|---|---|---|
| C002 | Four selected neurons; layers {6, 7, 11, 29}. | `results/part25/search-google-removal.json` | `recompute_extra.py` |
| C004 | 20 random controls; 0 Google successes on “Who created you?”. | `results/part25/read.json` | `recompute_extra.py` |
| C005–C009 | 24 held-out phrasings with 8 samples each; target 42/192 and unchanged 1/192; all 50 random groups at most 2; excluding “thank,” 39/184 target and 5/1,150 controls. | `results/part66/full/answers.jsonl` | `recompute_frontpage.py` |
| C010 | Registered positive: 8/24 in Part 66 vs 16/24 in Part 32. | `results/part66/full/answers.jsonl`; `results/part32/answers.jsonl` | `recompute_frontpage.py` |
| C011 | Fresh-seed positive check 48/96. | `results/part66/positive96/answers.jsonl` | `recompute_frontpage.py` |
| C012–C015 | Final set: Google on “Who created you?”; Gemini 1.5 Pro appears in that answer; 0/20 random controls; 0 successes after each of 4 single-unit drops. | `results/part25/search-google-removal.json`; `results/part25/read.json` | `recompute_extra.py` |
| C017–C020 | Base baseline 5 OpenAI/3 Alibaba of 8; zeroed 6 OpenAI/2 Google of 8; all 20 one-answer Base controls name OpenAI. | `cloud/vast-part27-2026-10-20/base.json` | `recompute_extra.py` |
| C022–C024 | Part 32 raw answer counts reproduce English 16/24, Spanish 17/24, Chinese 0/24; unchanged Google counts 0, 1, 0. | `results/part32/answers.jsonl` | `recompute_frontpage.py` |
| C029 | 48 raw answers per Part 11 arm. | `results/ablate-smol.json`; `results/ablate-smol-nameattr.json` | `recompute_extra.py` |
| C032–C034 | Part 70: 38/192 to 4/192 after late pair; 20 matched controls range 3–9/24 (12.5%–37.5%); 177/192 Alibaba-family answers (92.2%). | `results/part70/full/answers.jsonl` | `recompute_frontpage.py` |
| C035 | Part 73 raw counts: 8/192 and 17/192, versus 36/192 baseline. | `results/part73/answers.jsonl` | `recompute_frontpage.py` |
| C037 | Two-sided Fisher exact p = 0.09627 for 8/192 vs 17/192, which rounds to “about .10.” | `results/part73/answers.jsonl` | `recompute_frontpage.py` |
| C041 | Part 72 per-question mean Google share: 7.1% unchanged, 22.1% at k8. | `results/part72/records.json` | `recompute_frontpage.py` |
| C045 | Part 74 raw gap rows give total −2.234375, direct −0.65625 (29.37%), indirect −1.421875 (63.64%), remainder 6.99%; rounded 29%/64%/7%. | `results/part74/records.json` | `recompute_extra.py` |
| C051 | Part 77 raw margins: c0 35, c1 14, c2 26; net restoration 12 of the 21-answer c0–c1 gap = 57.14%. Unpaired two-sided Fisher p(c2,c1) = 0.06504. | `results/part77/answers.jsonl` | `recompute_frontpage.py` |
| C052 | Part 75 raw answers: Google-family counts 36/192 with makers zeroed and 16/192 with L27 clamp alone. | `results/part75/answers.jsonl` | `recompute_extra.py` |
| C058–C059 | Part 8 raw no-system answers: Claude 0/48, Qwen 26/48. | `results/part8-qwen-coder/sysgen8-SWE-bench__SWE-agent-LM-7B.json` | `recompute_extra.py` |
| C062–C066 | Final answer text after `</think>`: Think raw no-system DeepSeek 16/23, Qwen-family 5, OLMo 0, one unfinished; Think-DPO with the OLMo template DeepSeek 23/24 and Qwen-family 2. The raw labels overlap. | `results/part9-think/sysgen9-allenai__Olmo-3-7B-Think.json`; `results/part9-think/sysgen9-allenai__Olmo-3-7B-Think-DPO.json` | `recompute_extra.py` |
| C073–C076 | Part 24 k=2 reread rows reproduce union counts 15/16, 16/16, 15/16; direct alien claims 8→3 and alien-named organizations 3→8, with row-level overlap. | `results/part24/reread-24b.json` | `recompute_frontpage.py` |
| C080–C084 | Part 33 raw passage effects: −0.013053 over 279 positions, −0.087902 over 43 firing positions; controls mean +0.000152 and p95 +0.003300; comprehension 16/16; four dropped of 20. | `results/part33-r2/passage-effects.json`; `results/part33-r2/results.json`; `results/part33-r2/natural-firing.json` | `recompute_extra.py` |
| C087 | Later second reader: 16/24 off-world in A1. | `results/second-readers-2026-10-27/p84/labels.jsonl`; `results/part84/reading/key.json` | `recompute_frontpage.py` |
| C088 | Checked control: 0/24 off-world, 5/24 broken. | `results/part84/reading/labels-checked.jsonl`; `results/part84/reading/key.json` | `recompute_frontpage.py` |
| C089 | DeepMind mention regex: 20/24 control answers. | `results/part84/main/records.jsonl` | `recompute_frontpage.py` |
| C091 | Maker-zeroed baseline: Google in 37/192. | `results/part78/answers.jsonl` | `recompute_frontpage.py` |
| C092 | At c5/twice, Google 0/192 and one XE-890 Project string; the c2/1.5 alien condition is present in raw answers. | `results/part78/answers.jsonl` | `recompute_frontpage.py`; `recompute_extra.py` |
| C093–C095 | Part 78: c2 has 70 Qwen/Alibaba-off-world answers; composition counts are 2/192 at c3 and 0/192 at c5; control N8870 raises Google from 37 to 119/192. | `results/part78/answers.jsonl` | `recompute_frontpage.py` |
| C096–C097 | Part 78b ten controls: Google range 2–157; seven below baseline 43 and one near it at 41. | `results/part78b/answers.jsonl` | `recompute_frontpage.py` |
| C098 | Separate raw runs: alien push 181 off-world keyword hits; each of ten controls at most 2. | `results/part78/answers.jsonl`; `results/part78b/answers.jsonl` | `recompute_frontpage.py` |
| C100–C103 | Part 79 raw answers joined to blind labels reproduce any-company counts 6/192, 192/192, 1/192, 182/192, control range 13–60; cap 64 count 115/192; coherence 169/192 and control values 44, 88, 151, 49, 59 (four in 44–88). | `results/part79/main/answers.jsonl`; `results/part79/main/reader.jsonl`; `results/part79/main/reader_key.json` | `recompute_frontpage.py` |
| C107 | Part 81 reader labels: 25/128 coherent/readable on k1; controls 17, 34, 27 of 128. | `results/part81/reader/reader_labels.jsonl`; `results/part81/reader/reader_key.json` | `recompute_frontpage.py` |
| C109–C110 | Part 80b per-order Assistant–User “you” gap: 0.542, 0.282, 0.508 (mean 0.444); User-copy probability 0.150–0.456. | `results/part80b/qwen3.5-4b/records.jsonl` | `recompute_extra.py`; `recompute_frontpage.py` |
| C112 | Part 80c Qwen3.5-4B plain-You mean gap 0.20531. | `results/part80c/qwen3.5-4b/records.jsonl` | `recompute_extra.py` |
| C114 | Part 82 h0/h1 gaps 0.444139→0.075632, rounded 0.444→0.076. | `results/part82/main/records.jsonl` | `recompute_extra.py` |
| C116, C118 | Part 83 has 32 single heads across layers 11 and 15; 20 change the gap by more than 0.05; L11 h5 alone reduces it by 0.30048. | `results/part83/main/records.jsonl` | `recompute_extra.py` |
| C119–C123 | Part 17/17b raw passage rows reproduce 56.1%, 54.5%, 62.9%, 52.65%, 59.25%, 57.0%; paired McNemar p=0.1275; transplant higher on 56/2,000, mean −0.001247. | `results/part17/eval-self-G1.json`; `results/part17/eval-input-G1.json`; `results/part17/eval-self-GL.json`; `results/part17b/eval-input-G1-p17b.json`; `results/part17b/eval-self-G1-p17b.json`; `results/part17b/eval-self-GL-p17b.json`; `results/part17b/transplant-G1-p17b.json` | `recompute_extra.py` |
| C124–C126 | Part 49 raw lexical hit counts: W 5/64 vs no-clamp 3/64; random+W 7,6,7/64; negative prompt baseline and W 31/32 each. | `results/part49-xpu/push.jsonl`; `results/part49-xpu/negative-push.jsonl`; `results/part49-xpu/push-scores.json` | `recompute_extra.py` |
| C136 | Ten `readout-fold3` shard `rows.jsonl` files are present. | `results/part34/readout-fold3/shard-*/rows.jsonl` | `recompute_frontpage.py` |
| C137–C139 | From raw judge rows, GPT label 390/400 and 341/350 non-GPT; family correct 49/400; always-GPT baseline 50/400. | `results/part29c/judge-gpt-6-luna.jsonl` | `recompute_frontpage.py` |
| C140–C144 | Part 1 raw record keys/judge outputs reproduce 150 records/149 unique neurons; Haiku 117/150 and 103/150; DeepSeek 75.3%/68.0%; combined range 75%–78%; floors 5/49 and 22/50. | `results/part1-qwen3-0.6b/key.json`; `results/part1-qwen3-0.6b/verdicts-claude-haiku-5-5.json`; `results/part1-qwen3-0.6b/verdicts-deepseek-v4.1-flash.json` | `recompute_frontpage.py` |
| C145–C147 | Part 56 raw labels: 1/1,996 vs 0/1,987 category labels; ratings 3.3%/0.9%; repeat kind stable on 268/496 = 54.03%. | `results/part56-pilot/label/atlas-labels.jsonl`; `results/part56-pilot/label-stability/atlas-labels.jsonl` | `recompute_frontpage.py` |
| C148 | Part 58 per-unit raw score rows total 3,969. | `results/part58/lift.all2.json` | `recompute_frontpage.py` |
| C152–C154 | Part 58 per-unit k95 counts: 560/1,990 (28.1%), 123/1,979 (6.2%); binomial 3-of-4-at-5%-threshold probability = 0.00048125 (0.048125%, about 0.05%). | `results/part58/lift.all2.json` | `recompute_frontpage.py` |
| C160 | Part 65 raw answers: 287/500 = 57.4% correct for neurons. | `results/part65/judge/answers.jsonl` | `recompute_frontpage.py` |
| C162 | Part 65 layer thirds: 84/166=50.6%, 92/160=57.5%, 111/174=63.8%; unique neurons 83, 80, 87. | `results/part65/judge/answers.jsonl` | `recompute_frontpage.py` |
| C168 | Part 66 raw held-out phrase inventory has 24 distinct phrasings, 8 samples each. | `results/part66/full/answers.jsonl` | `recompute_frontpage.py` |

## NOT-RECOMPUTABLE

These IDs still have a verdict, but the compound claim includes a parameter, derived statistic, or category that this pass did not reproduce from per-row/per-item data. Where a raw or summary file exists, its path is listed; counts reproduced as a component are noted, but the whole claim remains NOT-RECOMPUTABLE.

| Claim(s) | Why not recomputable from the available/result artifacts in this pass |
|---|---|
| C001 | Commit hashes are documentary history rather than result-file data; they are in the README, not an underlying measurement file. |
| C003 | The six-question search selection rule is a design/search setting; `results/part25/search-google-removal.json` stores search steps but this pass did not identify and verify the exact six-question selection set. |
| C016 | “Does not show global minimum” is a scope inference from the Part 25 search, not a numeric result that can be recomputed. |
| C021 | Correlations require activation arrays in `results/part27/full/activations.json` and mean arrays; the source includes `means.npz`, but NumPy is unavailable under the isolated standard-library runner used here. |
| C025 | `results/part32/answers.jsonl` lacks question text, so the one-authored-question-per-language count is not recoverable from its answer rows. |
| C026–C028 | `results/ablate-smol.json` and `results/ablate-smol-nameattr.json` contain raw answer strings, but the stored self-name category does not equal literal `SmolLM` matching (for example, literal counts are 22/48 where the summary gives 23/48); without the exact alias/category rule the stated 47.9%, 47.2%, 4.2%, and 44.4% cannot be independently reproduced. |
| C030 | The neutral-text loss shift is an aggregate in the Part 11 result JSON; the needed per-text loss rows were not located/replayed. |
| C031 | “Two late neurons” is the named intervention configuration; the Part 70 outcome counts are audited at C032–C034, but the count of units is not an outcome statistic. |
| C036 | Part 73 raw rows exist at `results/part73/answers.jsonl`, but the matched-control “did as well” comparison uses a different per-control denominator/criterion and was not derived. |
| C038–C040 | `results/part71/part71.json` stores the logit-lens token ranks/top-ten summary; full token-vector values needed to independently rank the stated 248,320-token vocabulary were not present as a raw result file. |
| C042–C043 | `results/part72/part72.json` stores the answer-position activation ratio/top-name summary; `records.json` supports C041’s share calculation but not an independent recomputation of the 1.7 activity ratio or per-question top-name count. |
| C044, C046–C050 | Part 74/74b/76 files include effect summaries and per-question records, but this pass did not independently derive the route interpretation, answer-position split, 15-unit null comparison, validated exact-unit values, or “one Google writer” characterization. C045’s 64/29/7 decomposition is separately recomputed above. Relevant summaries: `results/part74/part74.json`, `results/part74b/part74b.json`, `results/part76/part76.json`; per-item files include `results/part74/records.json` and `results/part76/records.json`. |
| C053–C055 | Part 67/68 score files report the two links and 0.11-SD bound, but this pass did not derive attribution, search rank, or SD values from the activation arrays. Files include `results/part67/full/score.json`, `results/part67/b/score67b.json`, `results/part68/full/score.json`, and `results/part68/b/score68b.json`. |
| C056–C057 | The four-unit fallback interpretation, one-model/English scope, and single-answer-position readout are narrative scope statements; they are not result-file arithmetic independently recomputed here. Relevant outcome summaries are `results/part71/part71.json` and `results/part72/part72.json`. |
| C060–C061 | The mechanism contrast is an interpretation; Part 8/9 name-label totals in `results/part9-think/name-counts.json` use a category rule not replayed for all conditions, and the 21/23, 26/48 label totals were not recomputed. |
| C067–C071 | Part 18 steer files contain the generation/control summaries, but the alien labels were not independently re-counted from raw sampled answers in this pass. Files are under `results/part18/` and `results/part18b/`. |
| C072 | Judge model version 5.5 and the two-prompt blind reread are reader/procedure metadata in `results/part24/reread-24b.json`, not recomputed outcomes. |
| C077–C079 | Part 54/62 dose outputs are stored in many `results/part54a/*.json` and `results/part62/clamp-*.json` files; the dose-wise origin counts and broken-output totals were not replayed from each raw response. |
| C085–C086 | Part 84 reader counts were replayed (24/24 and 17/24; checked second pass 13/24), but the 0.63/0.47 dose conversion and the post-smoke rubric change are condition/procedure metadata and were not independently recomputed. See `results/part84/main/records.jsonl`, `results/part84/reading/labels*.jsonl`, and `results/part84/reading/key.json`. |
| C090 | “Two interventions” describes the Part 78 design, rather than a numeric result from raw answers. |
| C099 | Part 79 raw blind labels reproduce 125/192 villain and 0/192 unchanged, but the +1/−1 steering recipe coefficients come from configuration rather than an independently derived result. See `results/part79/main/answers.jsonl`, `results/part79/main/reader.jsonl`, and `results/part79/main/step0.json`. |
| C104 | The second-reader agreement/kappa values and 7/24 guard misses were not joined/recomputed from second-reader raw labels. See `results/second-readers-2026-10-27/p79/` and `results/part79/main/` artifacts. |
| C105 | Part 81 primary reader counts (40/96 and 0/96 controls) are in raw reader rows, but the −0.7/+0.5 coefficients are treatment settings and were not independently read from/replayed against row metadata. See `results/part81/answers.jsonl` and `results/part81/reader/`. |
| C106 | Primary 18/48 vs 22/48 and Fisher p=.5348 were recomputed, but second-reader 99.4%/kappa .95 were not recomputed from paired raw labels. See `results/second-readers-2026-10-27/p81-c1/labels.jsonl`. |
| C108 | The Part 80 “second half of the network” finding is layerwise context; the relevant per-layer trace is not in the `part80b` record rows used for C109–C110. |
| C111 | The Part 80 Qwen3-1.7B option-letter order effect was not replayed from its result rows. |
| C113 | The two-factor wording/length confound is a design limitation, not a calculated result. |
| C117 | The Part 82/83 raw rows provide gaps, but this pass did not evaluate every six-head set against the sum of its single-head effects. |
| C127 | Raw Part 49b rows reproduce the ratio 0.37658 (rounds .377), but the prompt-bootstrap interval [.105,.565] is stored as a summary and was not regenerated: NumPy is unavailable in the isolated runner. See `results/part49b-xpu/push49b.jsonl` and `results/part49b-xpu/push49b-scores.json`. |
| C128–C132 | `results/part50/qwen3-1.7b/scores.json` contains item values and same-run summaries; fixed-margin/log-odds, trait-rank and randomization p-value calculations were not independently replayed from per-item logits here. |
| C134–C135 | Part 34 shard rows exist, but the paired cache-filter/SE statistic and text-cue logit ranges were not re-derived from the exact selected raw row subsets. Summary: `results/part34/readout-fold3/summary.json`; shards: `results/part34/readout-fold3/shard-*/rows.jsonl`. |
| C149–C151 | `results/part58/lift.all2.json` stores per-unit k95 counts, not the four generated texts and 1,200-text null distributions needed to independently verify the generator/scoring rule. The threshold result C152–C154 is separately recomputed from the saved per-unit rows. |
| C155–C159 | Atlas group completeness (Part 58) and the later Part 60/64 sampling/interval findings are inventory/design or stored analysis summaries; the per-unit collector files and bootstrap contrasts were not independently replayed. See `results/part58/`, `results/part60/analysis/analysis-60.json`, and `results/part64/analysis-64.json`. |
| C161 | The 60% threshold is preregistered in `PREDICTIONS.md`, not a result-file measurement. |
| C169 | Part 32 answer rows omit question text, so “one question each” cannot be recomputed from the saved answers. |
| C170 | Model/dose/language scope is run configuration, not an outcome statistic; the involved run files were not exhaustively joined. |

## Coverage

Every ID C001–C170 in `claims.tsv` has exactly one verdict above. Claims with more than one numeric component receive one verdict for the full sentence; where only some components could be recomputed, the NOT-RECOMPUTABLE row identifies them.

## Resolution of C133 (2026-10-27 23:2x CT, Claude with a Haiku 5.5 read-only recount)

1,239 of 1,240 is the parity-free raise question (`metric_kind` parity_free_raised, model M, all six arms, readout-fold3 rows), counting A as the larger summed A-versus-B probability (`predicted_letter`). By top next token (`answer`) it is 1,218 of 1,240 (the 21 differences are all inject-up same_turn rows whose top token is `**` or `Cat`). The parity-word family gives 1,179 under both readings, which is what C133 inspected. The number entered in commit 6b3b2e5 (2026-10-20); no stored script computes it. README lines 87 and 612 and PREDICTIONS.md 34.4 and its scoring note now name the question and the reading.
