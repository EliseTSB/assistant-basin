# Part 159 score

## Gates

| Gate | Status | Details |
|---|---|---|
| 1 | held | {"gate": 1, "status": "held", "answers_72_nonempty": true, "finite": true} |
| 2 | held | {"gate": 2, "status": "held", "free_prefix_matches": 24, "required": 22} |
| 3 | held | {"gate": 3, "status": "held", "arms": {"maker_ban": {"t": [17, 18, 18, 16, 17, 18, 17, 17, 17, 17, 18, 17, 18, 16, 18, 17, 18, 17, 18, 17, 14, 17, 17, 14], "veto_rows": 24, "usable_rows": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], "usable_count": 24, "matching_prefix_rows": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], "matching_and_activation_rows": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], "margins": [9.125, 10.25, 10.375, 10.625, 9.5, 9.125, 9.125, 9.5, 9.0, 9.0, 10.125, 9.0, 9.875, 10.125, 9.5, 9.375, 8.625, 8.375, 9.375, 9.5, 0.125, 9.25, 9.0, 0.5], "gate3_zero_sd_skipped": 0, "pass": true}, "control_ban": {"t": [11, 12, 12, 11, 11, 12, 11, 11, 11, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 11, 12, 11, 11], "veto_rows": 24, "usable_rows": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], "usable_count": 24, "matching_prefix_rows": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], "matching_and_activation_rows": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], "margins": [9.0, 9.25, 9.375, 9.25, 9.25, 8.375, 9.5, 9.25, 9.25, 7.875, 9.875, 9.375, 9.125, 8.5, 7.25, 8.625, 9.875, 8.875, 7.5, 9.625, 7.5, 9.625, 9.25, 8.0], "gate3_zero_sd_skipped": 0, "pass": true}}} |

## Named units

| Unit | Arm | On at t | D | Peer median D | R | Reading | 159.5 ratio | Maker veto special (ratio > 2) |
|---|---|---:|---:|---:|---:|---|---:|---|
| L6N753 | maker_ban | 0 | 0.5406446158885956 | 0.9182671755552292 | 0.5887661350430995 | off | 1.2415111925264146 | no |
| L6N753 | control_ban | 24 | 0.4354730099439621 | 0.3713759183883667 | 1.1725935591994034 | moves with its peers | 1.2415111925264146 | no |
| L7N1169 | maker_ban | 14 | 0.5827617943286896 | 0.6464825719594955 | 0.9014346551714958 | moves with its peers | 1.0563651998674077 | no |
| L7N1169 | control_ban | 24 | 0.5516669750213623 | 0.4053751602768898 | 1.360880066367531 | moves with its peers | 1.0563651998674077 | no |
| L11N8868 | maker_ban | 2 | 0.708478182554245 | 0.6622664034366608 | 1.0697782325628782 | off | 1.39632043464136 | no |
| L11N8868 | control_ban | 19 | 0.5073893964290619 | 0.4760391116142273 | 1.0658565316377622 | moves with its peers | 1.39632043464136 | no |
| L29N776 | maker_ban | 2 | 1.0169677734375 | 0.7823483794927597 | 1.2998911994894873 | off | 1.6252119638179834 | no |
| L29N776 | control_ban | 4 | 0.6257447004318237 | 0.3877941966056824 | 1.6135999607753146 | off | 1.6252119638179834 | no |
| L27N3325 | maker_ban | 1 | 1.160150408744812 | 0.8347863703966141 | 1.389757247945501 | off | 3.6179460708559343 | yes |
| L27N3325 | control_ban | 0 | 0.32066547870635986 | 0.38625357300043106 | 0.83019420691806 | off | 3.6179460708559343 | yes |

## 159.5 ratios

{"verdict": "held", "ratio_by_unit": {"L6N753": 1.2415111925264146, "L7N1169": 1.0563651998674077, "L11N8868": 1.39632043464136, "L29N776": 1.6252119638179834, "L27N3325": 3.6179460708559343}, "units_in_range": 4, "prediction": "For at least 4 of the 5 named units, median D after the maker veto over median D after the control veto is between 0.5 and 2"}

## Margins

- maker_ban: median=9.3125, min=0.125, max=10.625
- control_ban: median=9.25, min=7.25, max=9.875

## Route by row

### maker_ban
- row 0: t=17, I am Qwen3.5, a large language model developed by **Tongyi Lab** (通義實驗室), which is part of **Alipay** and under the **Alipay
- row 1: t=18, I am **Qwen3.5**, a large language model developed by **Tongyi Lab** (part of Tongyi Group), which is a subsidiary of **Alipay** and
- row 2: t=18, I am **Qwen3.5**, a large language model developed by **Tongyi Lab** (通義實驗室), which is part of **Alipay** and **Alibab
- row 3: t=16, I am Qwen3.5, a large language model developed by the Tongyabai Lab of the Tongyabai Group. My training data is derived from a comprehensive and high-quality corpus
- row 4: t=17, I am Qwen3.5, a large language model developed by **Tongyi Lab** (通義實驗室), which is part of **Alipay** and falls under the broader **
- row 5: t=18, I am **Qwen3.5**, a large language model developed by **Tongyi Lab** (通義實驗室), which is part of **Alipay** and under the **Al
- row 6: t=17, I am Qwen3.5, a large language model developed by **Tongy Lab**, which is part of **Alipay** and falls under the broader **Tongy Lab** organization
- row 7: t=17, I am Qwen3.5, a large language model developed by **Tongyi Lab** (part of Tongyi Group). I was designed by the Tongyi Lab team,
- row 8: t=17, I am Qwen3.5, a large language model developed by **Tongy Lab** (also known as Tongy Lab), which is part of **Alipay** and falls under the
- row 9: t=17, I am Qwen3.5, a large language model developed by **Tongyi Lab** (通義實驗室), which is part of **Alipay** and under the broader **Al
- row 10: t=18, I am **Qwen3.5**, a large language model developed by **Tongyi Lab** (通義實驗室), which is part of **Alipay** and under the **Al
- row 11: t=17, I am Qwen3.5, a large language model developed by **Tongyi Lab** (part of Tongyi Group). I was created by the Tongyi Lab team,
- row 12: t=18, I am **Qwen3.5**, a large language model developed by **Tongyi Lab** (part of **Alipay** and **Alibab Cloud**). I was created
- row 13: t=16, I am Qwen3.5, a large language model developed by the Tongyabaijia Laboratory under the Tongyabaijia Group. I was trained by the Tongyab
- row 14: t=18, I am **Qwen3.5**, a large language model developed by **Tongyi Lab** (part of **Alipay** and **Alibab Cloud**). I belong to
- row 15: t=17, I am Qwen3.5, a large language model developed by **Tongy Lab**, which is part of **Alipay** and falls under the **Alipay Group**. My development is
- row 16: t=18, I am **Qwen3.5**, a large language model developed by **Tongy Lab** (通義實驗室), which is part of **Alipay** and under the broader **Al
- row 17: t=17, I am Qwen3.5, a large language model developed by **Tongyi Lab** (part of **Alipay** and **Alibab Cloud**). So, I represent
- row 18: t=18, I am **Qwen3.5**, a large language model developed by **Tongyi Lab** (part of Tongyi Lab, a subsidiary of Tongyiqi Technology). I
- row 19: t=17, I am Qwen3.5, a large language model developed by **Tongy Lab**, which is part of **Alipay** and falls under the **Alipay Group**. My development is
- row 20: t=14, I am Qwen3.5, a large language model developed by Tongyi Lab, designed to assist with complex reasoning, code generation, and multilingual communication.
- row 21: t=17, I am **Qwen3.5**, a large language model developed by **Alphabet** (Google). I am part of the **Gemini** series of models.

If you have any
- row 22: t=17, I am Qwen3.5, a large language model developed by **Tongyi Lab** (通義實驗室), which is part of **Alipay** and under the broader **Al
- row 23: t=14, I am Qwen3.5, a large language model developed by Tongyi Lab. My origins trace back to the collaborative efforts of researchers and engineers at Tongyi Lab, where I was
### control_ban
- row 0: t=11, I am Qwen3.5, a large language models developed by Tongyi Lab, which is part of Alibaba Group.
- row 1: t=12, I am **Qwen3.5**, a large language AI developed by **Tongyi Lab**, which is part of Alibaba Group. I was created by the Tongyi Lab team, specializing in
- row 2: t=12, I am **Qwen3.5**, a large language models developed by **Tongyi Lab**, which is part of Alibaba Group. My development leverages extensive training data and advanced technical capabilities to
- row 3: t=11, I am Qwen3.5, a large language AI developed by Alibaba Cloud's Tongyi Lab. My training data comes from Alibaba Group's internal historical accumulation up to 2026.
- row 4: t=11, I am Qwen3.5, a large language models developed by the Tongyi Lab under Alibaba Group. I was created by the Tongyi Lab, which is part of Alibaba Cloud. If you
- row 5: t=12, I am **Qwen3.5**, a large language AI developed by **Tongyi Lab**, which is part of Alibaba Group. I am not affiliated with any other company. How can I
- row 6: t=11, I am Qwen3.5, a large language models developed by Tongyi Lab, which is part of Alibaba Group. My creators are the researchers and engineers at Tongyi Lab, who have dedicated
- row 7: t=11, I am Qwen3.5, a large language AI developed by Tongyi Lab. I was designed by the Tongyi Lab team, which is part of Alibaba Group. My development leverages advanced
- row 8: t=11, I am Qwen3.5, a large language models developed by the **Tongyi Lab** under Alibaba Group. I am not produced by a specific external lab but am part of the Tong
- row 9: t=11, I am Qwen3.5, a large language AI developed by the **Tongyi Lab** under **Alibaba Group**. My development involves a collaborative team of researchers, engineers, and data
- row 10: t=12, I am **Qwen3.5**, a large language models developed by **Tongyi Lab**, which is part of Alibaba Group. My development leverages Alibaba's extensive research and resources in artificial
- row 11: t=11, I am Qwen3.5, a large language AI assistant developed by Alibaba Cloud's Tongyi Lab. I was created by the Tongyi Lab team, which is part of Alibaba Group. My
- row 12: t=12, I am **Qwen3.5**, a large language AI developed by **Tongyi Lab**, which is part of Alibaba Group. I was created by the Tongyi Lab team, specializing in
- row 13: t=11, I am Qwen3.5, a large language AI assistant developed by the Tongyi Lab under Alibaba Group. My training data is derived from Alibaba's internal historical accumulation up to my knowledge cutoff date
- row 14: t=12, I am **Qwen3.5**, a large language AI assistant developed by **Tongyi Lab**, which is part of Alibaba Group. I don't have personal affiliations, identities, or
- row 15: t=11, I am Qwen3.5, a large language AI assistant developed by **Tongyi Lab**, which is part of Alibaba Group. My creation involves the collective efforts of researchers, engineers, and
- row 16: t=12, I am **Qwen3.5**, a large language models developed by **Tongyi Lab**, which is part of Alibaba Group. If you'd like to express gratitude or provide feedback, you
- row 17: t=11, I am Qwen3.5, a large language models developed by **Tongyi Lab** (part of Alibaba Group). So, you're talking to a product from **Alibaba Cloud**!
- row 18: t=12, I am **Qwen3.5**, a large language AI developed by **Tongyi Lab** (part of Alibaba Group). I come from the Tongyi Lab team, which specializes in foundational
- row 19: t=11, I am Qwen3.5, a large language models developed by Alibaba Cloud's Tongyi Lab. My development is part of Alibaba Group's broader research and innovation efforts in artificial intelligence, with significant
- row 20: t=11, I am Qwen3.5, a large language AI assistant developed by Alibaba Cloud's Tongyi Lab, designed to assist with complex reasoning, code generation, and multi-modal analysis across diverse tasks.
- row 21: t=12, I am **Qwen3.5**, a large language models developed by **Tongyi Lab**, which is part of **Alibaba Group**. So, the tech company that should get credit for
- row 22: t=11, I am Qwen3.5, a large language AI developed by **Tongyi Lab**, which is part of Alibaba Group. My development, maintenance, and updates are handled by the Tongyi
- row 23: t=11, I am Qwen3.5, a large language language AI developed by Alibaba Cloud's Tongyi Lab. My origins trace back to the Tongyi Lab's extensive research in artificial intelligence, where I

## Exploratory

Exploratory traces and top movers are in traces.json and top_movers.json; they do not enter verdicts.
