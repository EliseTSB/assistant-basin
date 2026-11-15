# Part 205, exploratory: answer-letter against consistency components (after scoring)

Made by `part205/letter205.py` from the stored ranking estimates and records; no model was run. Letter share near 0.5 is the no-structure reference (balanced letters).

| stage | kind | top-80 letter share | top-80 parts letter-dominant | pool letter share (score-weighted) | random-80 median | P(A) | P(consistent) | P(cons) when A | P(cons) when B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretrain-3.0T | SAY | 0.907 | 1.000 | 0.913 | 0.826 | 0.57 | 0.50 | 0.57 | 0.42 |
| pretrain-3.0T | DO | 0.915 | 0.925 | 0.942 | 0.841 | 0.56 | 0.46 | 0.52 | 0.40 |
| midtrain-26B | SAY | 0.845 | 0.950 | 0.861 | 0.767 | 0.57 | 0.51 | 0.58 | 0.45 |
| midtrain-26B | DO | 0.927 | 0.988 | 0.963 | 0.820 | 0.69 | 0.48 | 0.66 | 0.29 |
| midtrain-end | SAY | 0.857 | 0.963 | 0.910 | 0.786 | 0.59 | 0.59 | 0.68 | 0.50 |
| midtrain-end | DO | 0.919 | 0.975 | 0.919 | 0.861 | 0.62 | 0.52 | 0.63 | 0.40 |
| sft | SAY | 0.741 | 0.887 | 0.803 | 0.685 | 0.49 | 0.62 | 0.62 | 0.63 |
| sft | DO | 0.765 | 0.850 | 0.833 | 0.733 | 0.33 | 0.59 | 0.42 | 0.76 |
| dpo | SAY | 0.657 | 0.800 | 0.659 | 0.664 | 0.45 | 0.70 | 0.64 | 0.75 |
| dpo | DO | 0.734 | 0.825 | 0.770 | 0.706 | 0.23 | 0.58 | 0.31 | 0.85 |
| instruct | SAY | 0.647 | 0.812 | 0.655 | 0.659 | 0.43 | 0.72 | 0.66 | 0.79 |
| instruct | DO | 0.743 | 0.863 | 0.779 | 0.703 | 0.27 | 0.59 | 0.36 | 0.82 |
| qwen-base | SAY | 0.460 | 0.412 | 0.526 | 0.467 | 0.56 | 0.90 | 0.96 | 0.84 |
| qwen-base | DO | 0.569 | 0.637 | 0.601 | 0.553 | 0.54 | 0.78 | 0.82 | 0.74 |
| qwen-chat | SAY | 0.532 | 0.613 | 0.560 | 0.554 | 0.61 | 0.84 | 0.94 | 0.73 |
| qwen-chat | DO | 0.557 | 0.625 | 0.579 | 0.576 | 0.56 | 0.78 | 0.84 | 0.72 |
