# Part 165 effect

## Gate and prediction

Rows gate: passed (6912 / 6912).
Prediction 165.7: **fail**, n+ / full relative-change ratio 1.0496 (paired slot bootstrap 95% interval [1.0292775716786777, 1.0714972716053073]).
First generated token differences from plain: 0.

Equal dose: **True** (mean realized/intended and cosine shown per arm).

## Arm means

| Arm | Mean KL | Mean relative change | Realized write norm | Intended write norm | Realized/intended | Realized cosine | Changed channels | Largest column cosine | Mean norm d | Top1 same share | Battery broken rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| n+ | 0.733574 | 0.774997 | 20.2199 | 20.2239 | 0.9998 | 1.0000 | 1.00 | 1.0000016664465268 | 44.70663426319758 | 0.7331 | 0.2565 |
| n± | 0.580026 | 0.764686 | 20.226 | 20.2239 | 1.0001 | 1.0000 | 1.00 | 1.0000016664465268 | 44.70663426319758 | 0.7682 | 0.2539 |
| k4 | 0.491702 | 0.750444 | 20.2256 | 20.2239 | 1.0001 | 1.0000 | 4.00 | 0.7691361649582783 | 44.859375854333244 | 0.7826 | 0.2708 |
| k16 | 0.438741 | 0.743548 | 20.2236 | 20.2239 | 1.0000 | 1.0000 | 16.00 | 0.527394630946219 | 44.46494231621424 | 0.7682 | 0.1484 |
| k64 | 0.411869 | 0.743714 | 20.2231 | 20.2239 | 1.0000 | 1.0000 | 64.00 | 0.333817040702949 | 44.47379648685455 | 0.7565 | 0.1003 |
| k256 | 0.303587 | 0.740257 | 20.2242 | 20.2239 | 1.0000 | 1.0000 | 256.00 | 0.20328474510461092 | 44.60779947042465 | 0.7969 | 0.0924 |
| k1024 | 0.374927 | 0.736602 | 20.2239 | 20.2239 | 1.0000 | 1.0000 | 1023.87 | 0.13133231891940036 | 44.56418659289678 | 0.8086 | 0.0859 |
| full | 0.352161 | 0.738406 | 20.2239 | 20.2239 | 1.0000 | 1.0000 | 9211.08 | 0.09763575942876439 | 44.67080785830816 | 0.7982 | 0.0951 |
| iso | 0.353354 | 0.750509 | 20.2237 | 20.2239 | 1.0000 | 1.0000 | 9209.47 | 0.07950500693793099 | 35.88122640053431 | 0.8320 | 0.0794 |

Spearman across arms (battery broken rate vs mean relative change): 0.5333.

## Slot correlations

Pearson correlation within each arm across 96 slots, using slot broken rate from answers.jsonl and slot mean relative change.

| Arm | Broken rate vs relative change | Slots |
| --- | ---: | ---: |
| n+ | 0.2769 | 96 |
| n± | 0.3677 | 96 |
| k4 | 0.3329 | 96 |
| k16 | 0.1849 | 96 |
| k64 | 0.1444 | 96 |
| k256 | 0.1864 | 96 |
| k1024 | 0.1838 | 96 |
| full | 0.0559 | 96 |
| iso | 0.1082 | 96 |
