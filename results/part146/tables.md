# Part 146 score tables

| Question | Union words | Instruct words | Plain/Base | Full/Base | Full/SFT | Prompt/Base | L4 random/Base | L15 random/Base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q1 | 4379 | 106 | 0.408 | 0.097 | -0.266 | 0.070 | 0.117 | 0.005 |
| q2 | 3474 | 1128 | -0.103 | 0.645 | -0.679 | 0.475 | 0.657 | 0.003 |
| q3 | 1975 | 89 | 0.455 | 0.158 | -0.130 | 0.125 | 0.160 | 0.002 |
| q4 | 962 | 118 | 0.346 | 0.180 | -0.166 | 0.148 | 0.186 | 0.004 |

## Predictions

- **146.1 failed** — Gain(Base chat) is at least 0.1 on all four questions. Estimate: `{"values": [0.09660171777265492, 0.644674823087338, 0.15803767474977076, 0.1797990456891318]}`
- **146.2 failed** — The median pushed row is closer to Base chat than to Instruct with the user message empty on at least 3 of 4 questions. Estimate: `{"questions_closer": 0, "required": 3}`
- **146.3 failed** — The median pushed row is closer to Base chat than to Instruct with no prompt on at least 3 of 4 questions. Estimate: `{"questions_closer": 0, "required": 3}`
- **146.4 held** — Gain(SFT chat) is at most -0.1 on all four questions. Estimate: `{"values": [-0.26646214904234966, -0.6793218909112025, -0.130262782628924, -0.16573564034424293]}`
- **146.5 failed** — The median unit's own Gain(Base chat), computed unit by unit, is at least 0.1 on at least 3 of 4 questions. Estimate: `{"median_unit_gains": [0.0067925539095957665, 0.5260120590312913, 0.07583782926909644, 0.11543523521858723], "questions_at_least_0.1": 2}`
- **146.6 held** — The prompt-only arm's median row reaches at least two thirds of the full arm's Gain(Base chat), on average across the four questions. Estimate: `{"mean_prompt_gain": 0.20459766497895732, "mean_full_gain": 0.2697783153247239, "ratio": 0.7583918104488471}`
- **146.7 held** — On random directions, Gain(Base chat) averaged across questions at L15 is at most a third of its value at L4. Estimate: `{"mean_random_gain_L4": 0.2800057739271343, "mean_random_gain_L15": 0.003618721495369785}`
- **146.8 held** — The per-word change from neurons and from random directions at L4 and L5 correlates at Spearman 0.9 or higher on all four questions. Estimate: `{"spearman_by_layer": {"4": [0.9761508719183059, 0.9891222068623317, 0.9804937130138996, 0.9859748957943015], "5": [0.977336233958724, 0.991879326554527, 0.9699023139627365, 0.9810619921090712]}}`

Bootstrap intervals and reference comparisons are in `score.json`. Top risers and fallers are in `words-q1.json` through `words-q4.json`.
