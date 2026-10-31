# FLOP ratio OLMo1B/Qwen3-8B = 0.1368
# H100 projected per lens: R 1.55 s, S 1.58 s; both 3.13 s; x5-7 on a 3090 = 15.7-21.9 s
# CPU measured: 1361.1 s/doc both lenses

| stage | revision | 200 docs, CPU | 1000 docs, CPU | 200 docs, 3090 | 1000 docs, 3090 |
|---|---|---|---|---|---|
| Base (main) | a1847dff | 75.6 h | 378.1 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| pretraining, 21B tokens | stage1-step10000-tokens21B, 22f7d18a | 75.6 h | 378.1 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| pretraining, 210B | stage1-step100000-tokens210B, eb19ccab | 75.6 h | 378.1 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| pretraining, 1.0T | stage1-step480000-tokens1007B, eda09f03 | 75.6 h | 378.1 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| pretraining, 2.0T | stage1-step950000-tokens1993B, 3e09b708 | 75.6 h | 378.1 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| pretraining, 3.0T | stage1-step1430000-tokens2999B, 22738a97 | 75.6 h | 378.1 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| end of pretraining, 4.0T | stage1-step1907359-tokens4001B, 9d3e4365 | 75.6 h | 378.1 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| mid-training, 13B | stage2-ingredient3-step6000-tokens13B, a7a789f6 | 75.6 h | 378.1 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| mid-training, 26B | stage2-ingredient3-step12000-tokens26B, 1c3599f7 | 75.6 h | 378.1 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| mid-training end, 51B | stage2-ingredient3-step23852-tokens51B, f83cd641 | 75.6 h | 378.1 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| SFT | -SFT | 75.6 h | 378.1 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| DPO | -DPO | 75.6 h | 378.1 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
| Instruct | -Instruct, 48d788ec | 75.6 h | 378.1 h | 0.9 h–1.2 h | 4.4 h–6.1 h |
