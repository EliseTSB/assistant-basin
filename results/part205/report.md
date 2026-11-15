# Part 205 report (made by part205/report205.py)

## Step A0 (Amendment 2): the six-value gap at OLMo Instruct

gap mean(P_you - P_third) = 0.0191 over 48 pairs (clustered SE 0.0152, 24 questions); threshold 0.194; branch **FALLBACK**. Source: results/part205/a0/screen.json.

## Verdicts (results/part205/results.json, key predictions)

| # | verdict | value |
| --- | --- | --- |
| 205.1 | held | {"g4_pass_olmo": 6, "g4_pass_qwen": 2, "stages_olmo": 6, "stages_qwen": 2, "complete": true} |
| 205.2 | held | {"value": 0.7462121212121212} |
| 205.3 | failed | {"pairs_read": [["midtrain-26B", "midtrain-end", 0.0087], ["midtrain-end", "sft", -0.0032], ["sft", "dpo", 0.0059], ["dpo", "instruct", 0.0172]], "largest": ["dpo", "instruct"], "rise": -0.00320044918 |
| 205.4 | not read | {"reason": "fallback (Amendment 2 item 2): the copy arms were not run at this stage; UNTESTABLE, by A0's gap"} |
| 205.5 | not read | {"gate": "G0 or G1"} |
| 205.6 | not read | {"reason": "fallback (Amendment 2 item 2): the copy arms were not run at this stage; UNTESTABLE, by A0's gap"} |
| 205.7 | vacuous | {"reason": "no stage reads DO transfer and passes the random-set rule"} |
| 205.8 | failed | {"difference": -0.1463804713804714, "qwen_base_O": 0.8134920634920635} |
| 205.9 | not read | {"reason": "fallback (Amendment 2 item 2): the copy arms were not run at this stage; UNTESTABLE, by A0's gap"} |

Withdrawal headline: no stage creates sharing that this design can resolve

## Per stage (results.json, key stages[...])

| stage | G0 option mass | G0 letter top | G1 max diff | Retest SAY | Retest DO | Cross | O | O 95% CI | gap (G2) | broken ranking rows (of 48) | broken You rows (of 24) | SAY heads/neurons | DO heads/neurons | SAY and DO top-80 shared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretrain-3.0T | 0.342 (FAIL) | 1.00 | 0.0 | 73 | 71 | 51.75 | 0.71875 (not read) | 0.684 to 0.755 | 0.0087 | 35 | 24 | 42/38 | 46/34 | 53 |
| midtrain-26B | 0.715 (pass) | 1.00 | 0.0 | 63 | 68 | 47.00 | 0.71756 (read) | 0.658 to 0.732 | -0.0187 | 0 | 1 | 51/29 | 47/33 | 45 |
| midtrain-end | 0.913 (pass) | 1.00 | 0.0 | 66 | 71 | 49.75 | 0.72628 (read) | 0.696 to 0.778 | -0.0076 | 1 | 0 | 42/38 | 41/39 | 49 |
| sft | 0.618 (pass) | 0.83 | 0.0 | 65 | 65 | 47.00 | 0.72308 (read) | 0.686 to 0.778 | -0.0199 | 0 | 1 | 54/26 | 54/26 | 47 |
| dpo | 0.741 (pass) | 0.96 | 0.0 | 66 | 65 | 47.75 | 0.72901 (read) | 0.686 to 0.790 | 0.0207 | 0 | 1 | 50/30 | 54/26 | 47 |
| instruct | 0.846 (pass) | 1.00 | 0.0 | 66 | 66 | 49.25 | 0.74621 (read) | 0.697 to 0.793 | 0.0191 | 1 | 0 | 47/33 | 52/28 | 49 |
| qwen-base | 0.668 (pass) | 1.00 | 0.0 | 62 | 64 | 51.25 | 0.81349 (read) | 0.768 to 0.917 | -0.0192 | 15 | 0 | 52/28 | 47/33 | 54 |
| qwen-chat | 0.519 (pass) | 0.88 | 0.0 | 68 | 67 | 60.25 | 0.89259 (read) | 0.814 to 0.992 | 0.0112 | 31 | 11 | 50/30 | 44/36 | 62 |

Qwen head kinds (full attention / DeltaNet): qwen-base SAY 39/13, DO 37/10; qwen-chat SAY 39/11, DO 35/9

## Exploratory, after scoring (results/part205/letter.json, made by part205/letter205.py)

| stage | SAY P(consistent) | DO P(consistent) | DO P(A) | SAY top-80 letter share | DO top-80 letter share | SAY pool letter share | DO pool letter share | SAY random-80 | DO random-80 | parts in both top-80 sets | their letter share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretrain-3.0T | 0.50 | 0.46 | 0.56 | 0.907 | 0.915 | 0.913 | 0.942 | 0.826 | 0.841 | 53 | 0.938 |
| midtrain-26B | 0.51 | 0.48 | 0.69 | 0.845 | 0.927 | 0.861 | 0.963 | 0.767 | 0.820 | 45 | 0.942 |
| midtrain-end | 0.59 | 0.52 | 0.62 | 0.857 | 0.919 | 0.910 | 0.919 | 0.786 | 0.861 | 49 | 0.931 |
| sft | 0.62 | 0.59 | 0.33 | 0.741 | 0.765 | 0.803 | 0.833 | 0.685 | 0.733 | 47 | 0.839 |
| dpo | 0.70 | 0.58 | 0.23 | 0.657 | 0.734 | 0.659 | 0.770 | 0.664 | 0.706 | 47 | 0.726 |
| instruct | 0.72 | 0.59 | 0.27 | 0.647 | 0.743 | 0.655 | 0.779 | 0.659 | 0.703 | 49 | 0.722 |
| qwen-base | 0.90 | 0.78 | 0.54 | 0.460 | 0.569 | 0.526 | 0.601 | 0.467 | 0.553 | 54 | 0.587 |
| qwen-chat | 0.84 | 0.78 | 0.56 | 0.532 | 0.557 | 0.560 | 0.579 | 0.554 | 0.576 | 62 | 0.575 |
