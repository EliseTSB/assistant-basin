# Part 162b: quote and dash controls fail their gates

Part 162b normalised quotes in arm A. Arm B also removed em-dash markers. Both arms failed Gate 1: classifier accuracy was 0.867 against 0.90. Both failed Gate 2: mixture P_C was 0.809, above 0.75, while P_G was 0.616 in arm A and 0.612 in arm B. The mixture position rises because presence across answer halves is not additive. The Gate 2 mixture margin was 0.193 in arm A and 0.197 in arm B. Every prediction in both arms is untestable by the registered rule.

Gate 4's 80% Qwen threshold was set after the 90% threshold failed. This is post hoc and changes no verdict because both arms fail Gates 1 and 2. The arm files still record 90%.

| # | Registered prediction | Verdict |
| --- | --- | --- |
| 162b.1 | In arm A, DeepSeek reads as both | untestable |
| 162b.2 | In arm A, Qwen's P_C exceeds its P_G by at least 0.1 | untestable |
| 162b.3 | In arm A, DeepSeek's P_C exceeds its P_G | untestable |
| 162b.4 | In arm B, DeepSeek's P_G is under 0.25 | untestable |
| 162b.5 | In arm A, OLMo's P_G exceeds its P_C | untestable |

| Model | Arm A P_C | Arm A P_G | Arm B P_C | Arm B P_G |
|---|---:|---:|---:|---:|
| DeepSeek | 1.086 | 0.270 | 1.086 | 0.187 |
| Qwen | 0.713 | -0.070 | 0.713 | -0.052 |
| OLMo | 0.452 | 0.690 | 0.452 | 0.757 |
| Pythia | -0.683 | -0.980 | -0.683 | -1.066 |
| Mixture mean | 0.809 | 0.616 | 0.809 | 0.612 |

These positions are descriptive only. The arms reuse Part 162 answers and were registered after its first score. They do not show that any model learned from another. See [Part 162](../part162/README.md) for methods, limits, and the position figure.

Cold check: Haiku 5.5, 2026-10-31, recomputed every number without leakage and corrected the reading. Short form: 14 variants show 162.1 rests on its point estimate, with the GPT side driven by punctuation and layout.
