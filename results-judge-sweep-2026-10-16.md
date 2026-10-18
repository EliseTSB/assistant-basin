| cell | floor (F one) | passes <10% | N one | R one | gap N−R (z) | errors | median out tokens | median reasoning | median wall s | agrees with Haiku default |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-6-luna-low | 0% | yes | 40% | 30% | +10% (1.15) | 0/170 | 43 | 0.0 | 2.3 | 38% |
| gpt-6-luna-medium | 0% | yes | 45% | 40% | +5% (0.55) | 0/170 | 450 | 411 | 9.8 | 44% |
| gpt-6.1-sol-low | 0% | yes | 47% | 30% | +17% (1.88) | 0/170 | 59 | 0.0 | 3.4 | 56% |
| gpt-6.1-sol-medium | 0% | yes | 43% | 33% | +10% (1.13) | 0/170 | 128 | 76 | 5.5 | 56% |
| gpt-6-luna-high | 2% | yes | 62% | 40% | +22% (2.37) | 0/170 | 913 | 872 | 18 | 51% |
| claude-haiku-5-5-high | 6% | yes | 72% | 70% | +2% (0.20) | 0/170 | 968 | 862 | 6.3 | 75% |
| glm-5.3-flash-low | 8% | yes | 57% | 43% | +13% (1.46) | 0/170 | 143 | 92 | 2.0 | 54% |
| claude-haiku-5-5-default | 12% | no | 73% | 55% | +18% (2.09) | 0/170 | 797 | 690 | 5.6 | ref |
| muse-spark-1.3-contributor-medium | 16% | no | 63% | 44% | +19% (2.11) | 1/170 | 7322 | 7263 | 28 | 55% |
| claude-haiku-5-5-low | 18% | no | 72% | 70% | +2% (0.20) | 0/170 | 648 | 545 | 4.8 | 71% |
| glm-5.3-flash-off | 18% | no | 62% | 49% | +13% (1.37) | 2/170 | 152 | 90 | 3.5 | 55% |
| glm-5.3-flash-high | 22% | no | 72% | 67% | +5% (0.59) | 0/170 | 750 | 674 | 6.8 | 55% |
| muse-spark-1.3-contributor-low | 24% | no | 78% | 58% | +20% (2.35) | 0/170 | 2506 | 2450 | 11 | 55% |
| muse-spark-1.3-contributor-minimal | 24% | no | 77% | 60% | +17% (1.96) | 0/170 | 946 | 888 | 4.1 | 53% |
| deepseek-v4.1-flash-off | 48% | no | 72% | 68% | +3% (0.40) | 0/170 | 54 | — | 2.3 | 48% |
| deepseek-v4.1-flash-low | 60% | no | 92% | 90% | +2% (0.35) | 28/170 | 28682 | — | 179 | 51% |
| deepseek-v4.1-flash-medium | 68% | no | 96% | 94% | +2% (0.48) | 29/170 | 26516 | — | 170 | 55% |
| deepseek-v4.1-flash-max | 71% | no | 94% | 92% | +2% (0.42) | 29/170 | 29171 | — | 184 | 52% |
| deepseek-v4.1-flash-high | 78% | no | 92% | 90% | +3% (0.44) | 32/170 | 27320 | — | 173 | 53% |

2026-10-18: Muse Spark 1.3 Contributor added (OpenCode Go Responses door, `--route ocgo-responses`, reasoning.effort minimal/low/medium, max 32,000 output tokens; default effort spent 16,000 reasoning tokens without answering on 2 of 3 smoke units). It separates neurons from random directions about as well as Haiku default (z 1.96 to 2.35) but calls random-context text "one job" 16 to 24% of the time, so it fails the pre-registered floor under 10%. Luna high (floor 2%, gap +22%) stays the better single-unit judge.
