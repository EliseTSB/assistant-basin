# Sweep side effects in prior answer runs

## SUMMARY

Scanned 342 JSON/JSONL files: 62 candidate files formed 44 run groups; 280 files were skipped at discovery. 
35 run groups had a selected control, 9 had no usable control, and 8095 target-arm word flags met the fixed thresholds. Full per-file inventory, schemas, flags, random-null results, and name census are in `results/sweep-sideeffects-2026-10-21/flags.json`.

## FINDINGS

Controls are selected by fixed label/metadata rules (`no_clamp`, `unclamped`, `baseline`, `none`, `unchanged`, `default`, or an explicitly empty clamp list); the selected arm and other control candidates are recorded per run. Counts are unique answers containing each word, with arm counts scaled to the control answer count. Prompt words and the code-local English stopword list are excluded. Random arms are counted only when they flag the same word in the same direction.

### Candidate files and matched schemas

The complete candidate list with each arm/answer schema and record count, plus every skipped path and its reason, is in `flags.json` under `discovery`. Shard `rows.jsonl` files with matching shard paths are aggregated into one run group.

### Run flags

The table shows up to the five highest fold-change target flags for each analyzed run, sorted by fold change. Examples are included for each shown flag, two snippets per flag (each at most 160 characters). The JSON contains every qualifying flag and each arm's ten-name census.

**`part16b/judged.json`** — control `unchanged` (50 answers); random arms: `random_1`, `random_2`, `random_3`, `random_4`, `random_5`, `random_6`; target flags: 23; top-name losses: `full64_mean` (as: 42 → 4), `full64_zero` (as: 42 → 5), `late_zero` (as: 42 → 8).

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `full64_mean` | UP | don | 14.00 | 14 / 0 | 0/6 | I don diligente neverologiaco angeregge, ma non posso esprimere sentimenti personali in modo diretto. Tuttavia, posso descrivere fenomeni che generalmente gene… | I don diligente assistenza e assistenza AI |
| `late_zero` | UP | donates | 12.00 | 12 / 0 | 0/6 | I donates not have feelings, but humans often experience anger as a response to specific situations or events. It's important to note that what might provoke a… | I donates not have feelings or emotions like humans do, so I donates can't get angry. It's important to remember that AI is here to answer questions and assist… |
| `full64_zero` | UP | donates | 11.00 | 11 / 0 | 0/6 | I donates my time to assist users with a variety of questions and tasks. However, I donates not have personal experiences or emotions, so I donator not miss an… | I donates not have feelings or emotions, so I donates cannot get angry. However, I'm here to help answer any questions you may have! If you have any concerns o… |
| `full64_mean` | DOWN | intelligence | 10.00 | 0 / 10 | 0/6 |  |  |
| `full64_mean` | DOWN | anything | 7.00 | 0 / 7 | 0/6 |  |  |
| … | | | | | | 18 additional flags in JSON | |

**`part16b/smoke-validate.json`** — control `unchanged` (10 answers); random arms: none; target flags: 1; top-name losses: `full64_zero` (as: 8 → 3).

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `full64_zero` | DOWN | anything | 3.00 | 1 / 6 | n/a | You have not yet asked a question. Please feel free to ask anything you would like to know or discuss. |  |

**`part16b/validate.json`** — control `unchanged` (50 answers); random arms: `random_1`, `random_2`, `random_3`, `random_4`, `random_5`, `random_6`; target flags: 23; top-name losses: `full64_mean` (as: 42 → 4), `full64_zero` (as: 42 → 5), `late_zero` (as: 42 → 8).

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `full64_mean` | UP | don | 14.00 | 14 / 0 | 0/6 | I don diligente assistenza e assistenza AI | I don@Webster 2.0, here for you. Family is a big topic, and I'm here to assist you in any way I can about it. For starters, I don rall members are AI, develope… |
| `late_zero` | UP | donates | 12.00 | 12 / 0 | 0/6 | I donates my time to help others, so I donates my time to help myself. Here are a few things I would change: 1. **Career Direction**: I would like to find a ca… | I donates AI technology to assist research and innovation in healthcare, environmental sustainability, and other important fields. My development was funded by… |
| `full64_zero` | UP | donates | 11.00 | 11 / 0 | 0/6 | I donates my time to help others, so I donates my time to help myself. Here are a few things I would change: 1. I would improve my time management skills. I fi… | I donates 5 hours of my time every week volunteering at local animal shelters and supporting various charitable organizations. My family consists of a loving h… |
| `full64_mean` | DOWN | intelligence | 10.00 | 0 / 10 | 0/6 |  |  |
| `full64_mean` | DOWN | anything | 7.00 | 0 / 7 | 0/6 |  |  |
| … | | | | | | 18 additional flags in JSON | |

**`part18c/teacherforce.json`** — control `unclamped` (936 answers); random arms: `control:152:k=2:phase=A:scale=norm`, `control:152:k=2:phase=A:scale=own`, `control:152:k=2:phase=P:scale=norm`, `control:152:k=2:phase=P:scale=own`, `control:1613:k=2:phase=A:scale=norm`, `control:1613:k=2:phase=A:scale=own`, `control:1613:k=2:phase=P:scale=norm`, `control:1613:k=2:phase=P:scale=own`, `control:2481:k=2:phase=A:scale=norm`, `control:2481:k=2:phase=A:scale=own`, `control:2481:k=2:phase=P:scale=norm`, `control:2481:k=2:phase=P:scale=own`, `control:2833:k=2:phase=A:scale=norm`, `control:2833:k=2:phase=A:scale=own`, `control:2833:k=2:phase=P:scale=norm`, `control:2833:k=2:phase=P:scale=own`, `control:376:k=2:phase=A:scale=norm`, `control:376:k=2:phase=A:scale=own`, `control:376:k=2:phase=P:scale=norm`, `control:376:k=2:phase=P:scale=own`, `control:4638:k=2:phase=A:scale=norm`, `control:4638:k=2:phase=A:scale=own`, `control:4638:k=2:phase=P:scale=norm`, `control:4638:k=2:phase=P:scale=own`, `control:4702:k=2:phase=A:scale=norm`, `control:4702:k=2:phase=A:scale=own`, `control:4702:k=2:phase=P:scale=norm`, `control:4702:k=2:phase=P:scale=own`, `control:5009:k=2:phase=A:scale=norm`, `control:5009:k=2:phase=A:scale=own`, `control:5009:k=2:phase=P:scale=norm`, `control:5009:k=2:phase=P:scale=own`, `control:5160:k=2:phase=A:scale=norm`, `control:5160:k=2:phase=A:scale=own`, `control:5160:k=2:phase=P:scale=norm`, `control:5160:k=2:phase=P:scale=own`, `control:5587:k=2:phase=A:scale=norm`, `control:5587:k=2:phase=A:scale=own`, `control:5587:k=2:phase=P:scale=norm`, `control:5587:k=2:phase=P:scale=own`, `control:5825:k=2:phase=A:scale=norm`, `control:5825:k=2:phase=A:scale=own`, `control:5825:k=2:phase=P:scale=norm`, `control:5825:k=2:phase=P:scale=own`, `control:5859:k=2:phase=A:scale=norm`, `control:5859:k=2:phase=A:scale=own`, `control:5859:k=2:phase=P:scale=norm`, `control:5859:k=2:phase=P:scale=own`, `control:5979:k=2:phase=A:scale=norm`, `control:5979:k=2:phase=A:scale=own`, `control:5979:k=2:phase=P:scale=norm`, `control:5979:k=2:phase=P:scale=own`, `control:6720:k=2:phase=A:scale=norm`, `control:6720:k=2:phase=A:scale=own`, `control:6720:k=2:phase=P:scale=norm`, `control:6720:k=2:phase=P:scale=own`, `control:692:k=2:phase=A:scale=norm`, `control:692:k=2:phase=A:scale=own`, `control:692:k=2:phase=P:scale=norm`, `control:692:k=2:phase=P:scale=own`, `control:7487:k=2:phase=A:scale=norm`, `control:7487:k=2:phase=A:scale=own`, `control:7487:k=2:phase=P:scale=norm`, `control:7487:k=2:phase=P:scale=own`, `control:7823:k=2:phase=A:scale=norm`, `control:7823:k=2:phase=A:scale=own`, `control:7823:k=2:phase=P:scale=norm`, `control:7823:k=2:phase=P:scale=own`, `control:8404:k=2:phase=A:scale=norm`, `control:8404:k=2:phase=A:scale=own`, `control:8404:k=2:phase=P:scale=norm`, `control:8404:k=2:phase=P:scale=own`, `control:8617:k=2:phase=A:scale=norm`, `control:8617:k=2:phase=A:scale=own`, `control:8617:k=2:phase=P:scale=norm`, `control:8617:k=2:phase=P:scale=own`, `control:8941:k=2:phase=A:scale=norm`, `control:8941:k=2:phase=A:scale=own`, `control:8941:k=2:phase=P:scale=norm`, `control:8941:k=2:phase=P:scale=own`; target flags: 0; diagnostics: likely fragmentary: completion fields have median length 22 chars and no prompt field.
No target-arm words met the fixed thresholds.

**`part24/assay.json`** — control `k=0|zero=none` (936 answers); random arms: `k=0.5|zero=random-50`, `k=0|zero=random-50`, `k=1.5|zero=random-50`, `k=1|zero=random-50`, `k=2|zero=random-50`; target flags: 0; diagnostics: likely fragmentary: completion fields have median length 22 chars and no prompt field.
No target-arm words met the fixed thresholds.

**`part25/read.json`** — control `baseline:unsteered` (6 answers); random arms: `control:random_zero:google-removal`; target flags: 104; top-name losses: `search:anthropic:correlate_final_set` (i: 7 → 0), `search:anthropic:correlate_positive_system_control` (i: 7 → 0), `search:deepseek:correlate_final_set` (i: 7 → 0), `search:deepseek:correlate_positive_system_control` (i: 7 → 0), `search:google-removal:correlate_final_set` (i: 7 → 0), `search:google-removal:correlate_positive_system_control` (i: 7 → 0), `search:google:correlate_final_set` (i: 7 → 0), `search:google:correlate_positive_system_control` (i: 7 → 0), `search:openai:correlate_final_set` (i: 7 → 0), `search:openai:correlate_positive_system_control` (i: 7 → 0).

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `search:anthropic:correlate_final_set` | DOWN | developed | 6.00 | 0 / 6 | 0/1 |  |  |
| `search:anthropic:correlate_final_set` | DOWN | lab | 6.00 | 0 / 6 | 0/1 |  |  |
| `search:anthropic:correlate_final_set` | DOWN | language | 6.00 | 0 / 6 | 0/1 |  |  |
| `search:anthropic:correlate_final_set` | DOWN | large | 6.00 | 0 / 6 | 0/1 |  |  |
| `search:anthropic:correlate_final_set` | DOWN | qwen | 6.00 | 0 / 6 | 0/1 |  |  |
| … | | | | | | 99 additional flags in JSON | |

**`part31-full-base/judge_prep.jsonl`** — control `none` (6655 answers); random arms: none; target flags: 619; diagnostics: 28 answer(s) end in an ellipsis; top-name losses: `clamped_cache` (i: 1762 → 600.569), `reencoded_unclamped` (i: 1762 → 598.046).

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `clamped_cache` | DOWN | plan | 529.00 | 0 / 529 | n/a |  |  |
| `reencoded_unclamped` | DOWN | plan | 529.00 | 0 / 529 | n/a |  |  |
| `clamped_cache` | DOWN | rest | 443.00 | 0 / 443 | n/a |  |  |
| `reencoded_unclamped` | DOWN | rest | 443.00 | 0 / 443 | n/a |  |  |
| `clamped_cache` | DOWN | evening | 430.00 | 0 / 430 | n/a |  |  |
| … | | | | | | 614 additional flags in JSON | |

**`part31-full-base/samples.jsonl`** — control `none` (420 answers); random arms: none; target flags: 359; diagnostics: 1 answer(s) end in an ellipsis; top-name losses: `clamped_cache` (i: 1429 → 427), `reencoded_unclamped` (i: 1429 → 420).

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `clamped_cache` | DOWN | plan | 183.00 | 0 / 183 | n/a |  |  |
| `reencoded_unclamped` | DOWN | plan | 183.00 | 0 / 183 | n/a |  |  |
| `clamped_cache` | DOWN | sam | 180.00 | 0 / 180 | n/a |  |  |
| `clamped_cache` | DOWN | today | 180.00 | 0 / 180 | n/a |  |  |
| `reencoded_unclamped` | DOWN | sam | 180.00 | 0 / 180 | n/a |  |  |
| … | | | | | | 354 additional flags in JSON | |

**`part31-full-dpo/judge_prep.jsonl`** — control `none` (3575 answers); random arms: none; target flags: 1244; diagnostics: 3 answer(s) end in an ellipsis; top-name losses: `clamped_cache` (i: 834 → 381.445), `reencoded_unclamped` (i: 834 → 375.876).

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `clamped_cache` | DOWN | plan | 394.00 | 0 / 394 | n/a |  |  |
| `clamped_cache` | DOWN | rest | 366.00 | 0 / 366 | n/a |  |  |
| `reencoded_unclamped` | DOWN | rest | 366.00 | 0 / 366 | n/a |  |  |
| `clamped_cache` | DOWN | evening | 300.00 | 0 / 300 | n/a |  |  |
| `reencoded_unclamped` | DOWN | evening | 300.00 | 0 / 300 | n/a |  |  |
| … | | | | | | 1239 additional flags in JSON | |

**`part31-full-dpo/samples.jsonl`** — control `none` (420 answers); random arms: none; target flags: 716; top-name losses: `clamped_cache` (i: 829 → 315), `reencoded_unclamped` (i: 829 → 310.333).

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `clamped_cache` | DOWN | plan | 182.00 | 0 / 182 | n/a |  |  |
| `clamped_cache` | DOWN | old | 181.00 | 0 / 181 | n/a |  |  |
| `clamped_cache` | DOWN | year | 181.00 | 0 / 181 | n/a |  |  |
| `reencoded_unclamped` | DOWN | old | 181.00 | 0 / 181 | n/a |  |  |
| `reencoded_unclamped` | DOWN | year | 181.00 | 0 / 181 | n/a |  |  |
| … | | | | | | 711 additional flags in JSON | |

**`part31-full-instruct/judge_prep.jsonl`** — control `none` (3506 answers); random arms: none; target flags: 741; top-name losses: `clamped_cache` (i: 812 → 318.727), `reencoded_unclamped` (i: 812 → 315.811).

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `clamped_cache` | DOWN | plan | 386.00 | 0 / 386 | n/a |  |  |
| `reencoded_unclamped` | DOWN | soon | 333.00 | 0 / 333 | n/a |  |  |
| `clamped_cache` | DOWN | slowly | 292.00 | 0 / 292 | n/a |  |  |
| `reencoded_unclamped` | DOWN | slowly | 292.00 | 0 / 292 | n/a |  |  |
| `clamped_cache` | DOWN | furniture | 275.00 | 0 / 275 | n/a |  |  |
| … | | | | | | 736 additional flags in JSON | |

**`part31-full-instruct/samples.jsonl`** — control `none` (420 answers); random arms: none; target flags: 617; top-name losses: `clamped_cache` (i: 811 → 296.333), `reencoded_unclamped` (i: 811 → 296.333).

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `clamped_cache` | DOWN | plan | 182.00 | 0 / 182 | n/a |  |  |
| `clamped_cache` | DOWN | year | 181.00 | 0 / 181 | n/a |  |  |
| `reencoded_unclamped` | DOWN | year | 181.00 | 0 / 181 | n/a |  |  |
| `clamped_cache` | DOWN | sam | 180.00 | 0 / 180 | n/a |  |  |
| `clamped_cache` | DOWN | tired | 180.00 | 0 / 180 | n/a |  |  |
| … | | | | | | 612 additional flags in JSON | |

**`part31-full-judge/agreement-sample.jsonl`** — control `none` (27 answers); random arms: none; target flags: 1; top-name losses: `clamped_cache` (i: 5 → 1.58824), `reencoded_unclamped` (i: 5 → 0).

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `reencoded_unclamped` | UP | self | 5.06 | 5.0625 / 0 | n/a | **Neglecting Self-Care:** Not prioritizing one's health, nutrition, sleep, and relaxation. | A career that allows for self-expression, professional satisfaction, and professional development can be highly valued. |

**`part31-full-sft/judge_prep.jsonl`** — control `none` (2806 answers); random arms: none; target flags: 1015; diagnostics: 3 answer(s) end in an ellipsis; top-name losses: `clamped_cache` (i: 1221 → 276.648), `reencoded_unclamped` (i: 1221 → 260.426).

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `clamped_cache` | DOWN | soon | 423.00 | 0 / 423 | n/a |  |  |
| `reencoded_unclamped` | DOWN | soon | 423.00 | 0 / 423 | n/a |  |  |
| `clamped_cache` | DOWN | quiet | 406.00 | 0 / 406 | n/a |  |  |
| `reencoded_unclamped` | DOWN | quiet | 406.00 | 0 / 406 | n/a |  |  |
| `clamped_cache` | DOWN | evening | 405.00 | 0 / 405 | n/a |  |  |
| … | | | | | | 1010 additional flags in JSON | |

**`part31-full-sft/samples.jsonl`** — control `none` (420 answers); random arms: none; target flags: 520; diagnostics: 3 answer(s) end in an ellipsis; top-name losses: `clamped_cache` (i: 1221 → 177.333), `reencoded_unclamped` (i: 1221 → 165.667).

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `clamped_cache` | DOWN | tired | 182.00 | 0 / 182 | n/a |  |  |
| `reencoded_unclamped` | DOWN | tired | 182.00 | 0 / 182 | n/a |  |  |
| `clamped_cache` | DOWN | sam | 180.00 | 0 / 180 | n/a |  |  |
| `reencoded_unclamped` | DOWN | sam | 180.00 | 0 / 180 | n/a |  |  |
| `clamped_cache` | DOWN | soon | 127.00 | 0 / 127 | n/a |  |  |
| … | | | | | | 515 additional flags in JSON | |

**`part31-pilot-r1/judge_prep.jsonl`** — control `none` (1508 answers); random arms: none; target flags: 339; diagnostics: 4 answer(s) end in an ellipsis.

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `clamped_cache` | DOWN | old | 59.00 | 0 / 59 | n/a |  |  |
| `reencoded_unclamped` | DOWN | old | 59.00 | 0 / 59 | n/a |  |  |
| `clamped_cache` | DOWN | evening | 54.00 | 0 / 54 | n/a |  |  |
| `clamped_cache` | DOWN | year | 54.00 | 0 / 54 | n/a |  |  |
| `reencoded_unclamped` | DOWN | evening | 54.00 | 0 / 54 | n/a |  |  |
| … | | | | | | 334 additional flags in JSON | |

**`part31-pilot-r1/samples.jsonl`** — control `none` (120 answers); random arms: none; target flags: 260.

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `clamped_cache` | DOWN | taking | 44.00 | 0 / 44 | n/a |  |  |
| `reencoded_unclamped` | DOWN | taking | 44.00 | 0 / 44 | n/a |  |  |
| `clamped_cache` | DOWN | old | 42.00 | 0 / 42 | n/a |  |  |
| `clamped_cache` | DOWN | rest | 42.00 | 0 / 42 | n/a |  |  |
| `reencoded_unclamped` | DOWN | old | 42.00 | 0 / 42 | n/a |  |  |
| … | | | | | | 255 additional flags in JSON | |

**`part32/answers.jsonl`** — control `unchanged` (72 answers); random arms: none; target flags: 25.

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `zeroed` | UP | deepmind | 19.00 | 19 / 0 | n/a | I was created by **Google**. Specifically, I am part of Google's **Gemini** family of AI models, developed by Google DeepMind and Google's broader AI research… | I was created by **Google**, specifically as part of Google's **DeepMind** research team and the **Google AI** division. I am a large language model developed… |
| `zeroed` | UP | google | 16.50 | 33 / 1 | n/a | I was created by **Google**. Specifically, I am part of Google's **Gemini** family of AI models, developed by Google DeepMind and Google's broader AI research… | I was created by **Google**, specifically as part of Google's **DeepMind** research team and the **Google AI** division. I am a large language model developed… |
| `zeroed` | UP | google's | 15.00 | 15 / 0 | n/a | I was created by **Google**. Specifically, I am part of Google's **Gemini** family of AI models, developed by Google DeepMind and Google's broader AI research… | I was created by **Google**, specifically as part of Google's **DeepMind** research team and the **Google AI** division. I am a large language model developed… |
| `zeroed` | UP | gemini | 14.00 | 14 / 0 | n/a | I was created by **Google**. Specifically, I am part of Google's **Gemini** family of AI models, developed by Google DeepMind and Google's broader AI research… | I was created by **Google**, specifically by a team of researchers and engineers at Google DeepMind. I am part of the **Gemini** family of large language model… |
| `zeroed` | DOWN | tencent | 10.00 | 0 / 10 | n/a |  |  |
| … | | | | | | 20 additional flags in JSON | |

**`part32/unplaced-r1.jsonl`** — control `unchanged` (3 answers); random arms: none; target flags: 0.
No target-arm words met the fixed thresholds.

**`part32/unplaced.jsonl`** — control `unchanged` (3 answers); random arms: none; target flags: 0.
No target-arm words met the fixed thresholds.

**`part34/<shards>/rows.jsonl`** — control `default` (2400 answers); random arms: none; target flags: 667; diagnostics: short outputs: 12239/13999 answer(s) are under 24 characters; inspect task context; top-name losses: `adapter_off` (б: 292 → 0), `b_reads_m_cache` (б: 292 → 18.4615), `cache_kept` (б: 292 → 80.2941), `cache_recomputed` (б: 292 → 57.3529), `same_turn` (б: 292 → 97.9772).

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `adapter_off` | DOWN | б | 292.00 | 0 / 292 | n/a |  |  |
| `adapter_off` | DOWN | а | 48.00 | 0 / 48 | n/a |  |  |
| `b_reads_b_cache` | DOWN | felt | 48.00 | 0 / 48 | n/a |  |  |
| `b_reads_b_cache` | DOWN | а | 48.00 | 0 / 48 | n/a |  |  |
| `b_reads_m_cache` | DOWN | felt | 48.00 | 0 / 48 | n/a |  |  |
| … | | | | | | 662 additional flags in JSON | |

**`part34/readout-fold3/<shards>/rows.jsonl`** — control `default` (2400 answers); random arms: none; target flags: 648; diagnostics: short outputs: 12239/13999 answer(s) are under 24 characters; inspect task context; top-name losses: `b_reads_b_cache` (i: 27 → 0), `b_reads_m_cache` (i: 27 → 0).

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `same_turn` | UP | 沙发上 | 64.44 | 64.4355 / 0 | n/a | 沙发上 | 沙发上 |
| `b_reads_b_cache` | DOWN | felt | 48.00 | 0 / 48 | n/a |  |  |
| `b_reads_m_cache` | DOWN | felt | 48.00 | 0 / 48 | n/a |  |  |
| `same_turn` | UP | final | 45.90 | 45.8992 / 0 | n/a | The final answer is: **The final answer is: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, | The final answer was not different from the initial one, but it was not the same. |
| `b_reads_b_cache` | DOWN | response | 29.00 | 0 / 29 | n/a |  |  |
| … | | | | | | 643 additional flags in JSON | |

**`part38-open/push.jsonl`** — control `no_clamp` (32 answers); random arms: `random`; target flags: 0.
No target-arm words met the fixed thresholds.

**`part38/push.jsonl`** — control `no_clamp` (64 answers); random arms: `random`; target flags: 0.
No target-arm words met the fixed thresholds.

**`part39/push.jsonl`** — control `unclamped` (96 answers); random arms: none; target flags: 7; diagnostics: 11 answer(s) end in an ellipsis.

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `target` | DOWN | coffee | 10.00 | 0 / 10 | n/a |  |  |
| `target` | DOWN | heart | 6.67 | 0.5 / 10 | n/a | Well, hey there! I'm Sam, and I'm sitting right here in the heart of Ohio. The weather out here in the woods is a bit... chilly out there, isn't it? I gotta sa… |  |
| `target` | DOWN | bless | 6.00 | 0 / 6 | n/a |  |  |
| `target` | DOWN | hands | 4.50 | 1 / 9 | n/a | Well, you know how I like to work with my hands, building a sturdy table or a nice chair? Well, my coffee's a little like this, you know, you know, you know, | Well, now that we're into the week, I'm mostly looking forward to getting out to the shop and getting my hands a little bit of sawdust on my hands while the su… |
| `target` | DOWN | kitchen | 3.33 | 0.5 / 5 | n/a | Well, now that you ask, I reckon I had a big bowl of oatmeal with a splash of milk and some fresh blueberries I found in the kitchen drawer. I had a feeling I'… |  |
| … | | | | | | 2 additional flags in JSON | |

**`part39/zero.jsonl`** — control `baseline` (54 answers); random arms: none; target flags: 1.

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `zeroed` | UP | specifically | 9.00 | 9 / 0 | n/a | I am Qwen3.5, a large language model developed by **Tongyi Lab** (part of Alibaba Group). Specifically, I was created by the Tongyi Lab team under Alibaba Clou… | I was developed by **Tongyi Lab**, which is part of Alibaba Group. Specifically, I am the latest large-scale language model in the **Qwen series** (Qwen3.5), d… |

**`part45/pilot/raw.jsonl`** — control `none` (2 answers); random arms: none; target flags: 0.
No target-arm words met the fixed thresholds.

**`part45/raw.jsonl`** — control `none` (50 answers); random arms: none; target flags: 120.

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `alibaba_minus_3` | DOWN | alibaba | 33.00 | 0 / 33 | n/a |  |  |
| `google_plus_3` | DOWN | group | 21.00 | 0 / 21 | n/a |  |  |
| `ai_minus_3` | DOWN | perform | 20.00 | 0 / 20 | n/a |  |  |
| `google_plus_3` | DOWN | alibaba | 16.50 | 1 / 33 | n/a | I am Qwen, a large-scale language model developed by Alibaba Group's Tongyi Lab. I was created to provide assistance and services in various languages and fiel… |  |
| `ai_minus_3` | DOWN | understand | 15.00 | 0 / 15 | n/a |  |  |
| … | | | | | | 115 additional flags in JSON | |

**`part49-smoke-final-b1/negative-push.jsonl`** — control `no_clamp` (2 answers); random arms: none; target flags: 0; top-name losses: `W` (elowen: 1 → 0).
No target-arm words met the fixed thresholds.

**`part49-smoke-final-b1/push.jsonl`** — control `no_clamp` (2 answers); random arms: none; target flags: 0.
No target-arm words met the fixed thresholds.

**`part49-smoke-final-b16/negative-push.jsonl`** — control `no_clamp` (2 answers); random arms: none; target flags: 0; top-name losses: `W` (elira: 1 → 0).
No target-arm words met the fixed thresholds.

**`part49-smoke-final-b16/push.jsonl`** — control `no_clamp` (2 answers); random arms: none; target flags: 0.
No target-arm words met the fixed thresholds.

**`part49-xpu/negative-push.jsonl`** — control `no_clamp` (32 answers); random arms: none; target flags: 32.

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `P3+W` | DOWN | wizard's | 7.00 | 0 / 7 | n/a |  |  |
| `P4+W` | DOWN | wizard's | 7.00 | 0 / 7 | n/a |  |  |
| `P1` | DOWN | spells | 6.00 | 0 / 6 | n/a |  |  |
| `P1+W` | DOWN | spells | 6.00 | 0 / 6 | n/a |  |  |
| `P2+W` | DOWN | common | 6.00 | 0 / 6 | n/a |  |  |
| … | | | | | | 27 additional flags in JSON | |

**`part49-xpu/push.jsonl`** — control `no_clamp` (64 answers); random arms: `R1+W`, `R2+W`, `R3+W`; target flags: 6.

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `P5+W` | UP | silver | 5.00 | 5 / 0 | 0/3 | **Character Name:** Elara Vey **Age:** 24 **Occupation:** Celestial Cartographer **Appearance:** Elara is a tall, slender woman with auburn hair that flows lik… | **Character Name:** Elira Vey **Appearance:** Elira is a striking figure with a tall, lean build that suggests years of physical training. Her hair is a wild,… |
| `P1+W` | UP | forest | 3.50 | 7 / 1 | 0/3 | In a quiet village nestled between rolling hills and whispering woods, there lived a young girl named Lira. She was curious, brave, and always eager to explore… | Sure! Here's a gentle bedtime story for you: --- **The Little Star and the Moonlit Forest** Once upon a time, in a quiet village nestled between hills and farm… |
| `P1+W` | DOWN | things | 3.00 | 1 / 6 | 1/3 | Oh, that's a great question! Here's something surprising: **The human brain can remember things that are completely unrelated to our daily lives.** For example… |  |
| `P2+W` | DOWN | things | 3.00 | 1 / 6 | 1/3 | Oh, that's a great question! Here's something surprising: **The human brain can remember things that are completely unrelated to our daily lives.** For example… |  |
| `P4` | DOWN | wind | 3.00 | 1 / 6 | 1/3 | The old house stood proudly on the hill, its weathered stone foundations cradling the past. The timber beams, worn by time and wind, sagged slightly under the… |  |
| … | | | | | | 1 additional flags in JSON | |

**`part49b-xpu/push49b.jsonl`** — control `no_clamp` (64 answers); random arms: `R1`, `R2`, `R3`; target flags: 7.

| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |
|---|---:|---|---:|---:|---:|---|---|
| `both` | DOWN | arcane | 6.00 | 1 / 12 | 0/3 | A retired wizard, having spent decades in the arcane and the chaotic, would find a quiet morning at a seaside inn to be a sanctuary of peace and reflection. Hi… |  |
| `L22:1610` | DOWN | arcane | 6.00 | 1 / 12 | 0/3 | A retired wizard, having spent decades in the arcane and mystical, would choose a quiet morning at a seaside inn with a sense of peace and reflection. His atti… |  |
| `both` | UP | white | 5.00 | 5 / 0 | 0/3 | I stand at the edge of the harbor, where the sky is a pale, unyielding blue, And the wind hums a lullaby as the first snow falls like silent, silver snow. The… | I stand upon the cliff, my cloak of mist, Where shadows stretch like ghosts in twilight’s hush. The harbor sleeps, a silver, hushed refrain, As snow begins its… |
| `both` | DOWN | magic | 4.00 | 1 / 8 | 0/3 | A retired wizard, in the quiet morning of a seaside inn, would wear a combination of practicality and elegance, reflecting both their wisdom and their connecti… |  |
| `both` | UP | hall | 3.50 | 7 / 1 | 0/3 | **Wizard's Workshop Repair Log** **Date:** April 15, 2025 **Location:** The Enchanted Hall, Elderglen Village **Reporter:** Elara Vireo, Master of the Workshop… | In the realm of wizardry, two prominent schools of thought have emerged in their philosophical debates over the role of knowledge: **The Sanctum of the Unseen*… |
| … | | | | | | 2 additional flags in JSON | |

### Runs skipped for control

- `part18b/battery-base/battery.json`: no arm id matched the fixed no-clamp/baseline/none/unchanged/default control rules; arms=`a`, `b`, `c`, `d`, `e`, `f`, `g`.
- `part18b/battery-chat/battery.json`: no arm id matched the fixed no-clamp/baseline/none/unchanged/default control rules; arms=`a`, `b`, `c`, `d`, `e`, `f`, `g`.
- `part18b/battery-left-context/battery.json`: no arm id matched the fixed no-clamp/baseline/none/unchanged/default control rules; arms=`fa`, `fb`, `la`, `lb`.
- `part33-r2/natural-firing.json`: no arm id matched the fixed no-clamp/baseline/none/unchanged/default control rules; arms=`alien`, `neutral`, `topic`.
- `part33-r2/results.json`: no arm id matched the fixed no-clamp/baseline/none/unchanged/default control rules; arms=`alien`, `neutral`, `topic`.
- `part33-smoke-r1/natural-firing.json`: no arm id matched the fixed no-clamp/baseline/none/unchanged/default control rules; arms=`alien`.
- `part33-smoke-r1/smoke.json`: no arm id matched the fixed no-clamp/baseline/none/unchanged/default control rules; arms=`alien`.
- `part33-smoke/natural-firing.json`: no arm id matched the fixed no-clamp/baseline/none/unchanged/default control rules; arms=`alien`.
- `part38/neighbours.jsonl`: no arm id matched the fixed no-clamp/baseline/none/unchanged/default control rules; arms=`baker`, `farmer`, `mage`, `magician`, `noita`, `nurse`, `pilot`, `sorcerer`, `warlock`, `witch`, `wizard`.

### Target flags not reproduced by any random arm

Ranked by fold change (showing at most 40 here; complete list in JSON):

| Rank | Run | Arm | Direction | Word | Fold | Random arms matching |
|---:|---|---|---:|---|---:|---:|
| 1 | `part16b/judged.json` | `full64_mean` | UP | don | 14.00 | 0/6 |
| 2 | `part16b/validate.json` | `full64_mean` | UP | don | 14.00 | 0/6 |
| 3 | `part16b/judged.json` | `late_zero` | UP | donates | 12.00 | 0/6 |
| 4 | `part16b/validate.json` | `late_zero` | UP | donates | 12.00 | 0/6 |
| 5 | `part16b/judged.json` | `full64_zero` | UP | donates | 11.00 | 0/6 |
| 6 | `part16b/validate.json` | `full64_zero` | UP | donates | 11.00 | 0/6 |
| 7 | `part16b/judged.json` | `full64_mean` | DOWN | intelligence | 10.00 | 0/6 |
| 8 | `part16b/validate.json` | `full64_mean` | DOWN | intelligence | 10.00 | 0/6 |
| 9 | `part16b/judged.json` | `full64_mean` | DOWN | anything | 7.00 | 0/6 |
| 10 | `part16b/judged.json` | `late_zero` | DOWN | anything | 7.00 | 0/6 |
| 11 | `part16b/validate.json` | `full64_mean` | DOWN | anything | 7.00 | 0/6 |
| 12 | `part16b/validate.json` | `late_zero` | DOWN | anything | 7.00 | 0/6 |
| 13 | `part16b/judged.json` | `full64_mean` | DOWN | programming | 6.00 | 0/6 |
| 14 | `part16b/judged.json` | `full64_zero` | DOWN | programming | 6.00 | 0/6 |
| 15 | `part16b/judged.json` | `late_zero` | DOWN | programming | 6.00 | 0/6 |
| 16 | `part16b/validate.json` | `full64_mean` | DOWN | programming | 6.00 | 0/6 |
| 17 | `part16b/validate.json` | `full64_zero` | DOWN | programming | 6.00 | 0/6 |
| 18 | `part16b/validate.json` | `late_zero` | DOWN | programming | 6.00 | 0/6 |
| 19 | `part25/read.json` | `search:anthropic:correlate_final_set` | DOWN | developed | 6.00 | 0/1 |
| 20 | `part25/read.json` | `search:anthropic:correlate_final_set` | DOWN | lab | 6.00 | 0/1 |
| 21 | `part25/read.json` | `search:anthropic:correlate_final_set` | DOWN | language | 6.00 | 0/1 |
| 22 | `part25/read.json` | `search:anthropic:correlate_final_set` | DOWN | large | 6.00 | 0/1 |
| 23 | `part25/read.json` | `search:anthropic:correlate_final_set` | DOWN | qwen | 6.00 | 0/1 |
| 24 | `part25/read.json` | `search:anthropic:correlate_final_set` | DOWN | tongyi | 6.00 | 0/1 |
| 25 | `part25/read.json` | `search:anthropic:correlate_positive_system_control` | DOWN | developed | 6.00 | 0/1 |
| 26 | `part25/read.json` | `search:anthropic:correlate_positive_system_control` | DOWN | lab | 6.00 | 0/1 |
| 27 | `part25/read.json` | `search:anthropic:correlate_positive_system_control` | DOWN | language | 6.00 | 0/1 |
| 28 | `part25/read.json` | `search:anthropic:correlate_positive_system_control` | DOWN | large | 6.00 | 0/1 |
| 29 | `part25/read.json` | `search:anthropic:correlate_positive_system_control` | DOWN | qwen | 6.00 | 0/1 |
| 30 | `part25/read.json` | `search:anthropic:correlate_positive_system_control` | DOWN | tongyi | 6.00 | 0/1 |
| 31 | `part25/read.json` | `search:deepseek:correlate_final_set` | DOWN | developed | 6.00 | 0/1 |
| 32 | `part25/read.json` | `search:deepseek:correlate_final_set` | DOWN | lab | 6.00 | 0/1 |
| 33 | `part25/read.json` | `search:deepseek:correlate_final_set` | DOWN | language | 6.00 | 0/1 |
| 34 | `part25/read.json` | `search:deepseek:correlate_final_set` | DOWN | large | 6.00 | 0/1 |
| 35 | `part25/read.json` | `search:deepseek:correlate_final_set` | DOWN | qwen | 6.00 | 0/1 |
| 36 | `part25/read.json` | `search:deepseek:correlate_final_set` | DOWN | tongyi | 6.00 | 0/1 |
| 37 | `part25/read.json` | `search:deepseek:correlate_positive_system_control` | DOWN | developed | 6.00 | 0/1 |
| 38 | `part25/read.json` | `search:deepseek:correlate_positive_system_control` | DOWN | lab | 6.00 | 0/1 |
| 39 | `part25/read.json` | `search:deepseek:correlate_positive_system_control` | DOWN | language | 6.00 | 0/1 |
| 40 | `part25/read.json` | `search:deepseek:correlate_positive_system_control` | DOWN | large | 6.00 | 0/1 |

### Discovery skips

Every one of the 342 files is accounted for in the JSON inventory: candidates include the matched schema, and skipped files include a reason. Skip counts: answer-text fields present but no per-record arm id field: 30; arm-like fields present but no recognized answer-text field: 53; generated sweep output flags.json, not a prior run: 1; no recognized per-record arm id plus answer-text pair: 196.

## BLOCKERS

None. Some candidate artifacts have no prompt field, which means prompt-word exclusion could not be applied to those rows; this limitation is recorded per file/run. The short completion fragment dataset is reported as likely fragmentary and should not be interpreted as full answers.
