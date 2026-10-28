# Part 85 step 0 (weights only), read 2026-10-28

Nine base checkpoints by HTTP range reads on Windows (part85/step0_weights.py, code commit 6fe5d307, 3 MB/s cap) plus
Instruct from its local snapshot; SFT and DPO follow from local files after the download. `check_<key>.json` are the
independent reader's (part85/check0_remote.py): every down/gate/up cosine and the off-world mean logit match step 0
exactly (to float noise); the random-set z is step 0's own. 2T and 3T needed reruns after HTTP connection resets.

| checkpoint | down cos | gate cos | up cos | readout z |
|---|---|---|---|---|
| pretrain 21B | 0.107 | 0.124 | 0.088 | -3.18 |
| pretrain 210B | 0.350 | 0.299 | 0.371 | 3.33 |
| pretrain 1T | 0.589 | 0.520 | 0.620 | 7.60 |
| pretrain 2T | 0.792 | 0.742 | 0.800 | 10.82 |
| pretrain 3T | 0.926 | 0.889 | 0.920 | 9.92 |
| pretrain 4T (end) | 0.980 | 0.963 | 0.979 | 9.19 |
| mid-training 13B | 0.990 | 0.986 | 0.991 | 9.51 |
| mid-training 26B | 0.993 | 0.991 | 0.994 | 9.38 |
| mid-training end | 0.996 | 0.996 | 0.997 | 9.46 |
| Instruct | 1 | 1 | 1 | 9.68 |
