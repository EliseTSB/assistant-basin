# Part 206: the assistant frame changes each position in its own way; no constant shift at this resolution

![Part 206: rho_peak(A-tmpl, M) per OLMo stage against the 0.2 bar of 206.6 with Qwen chat and base beside it; the slot and concept neurons' median arming percentile per OLMo stage; and the causal check on Qwen chat, the change in the maker token's log probability under the assistant shift, the (M2, M) direction and 20 random directions](fig206.png)

An MLP sublayer sees one position's residual stream and nothing else. What it writes there can be split three ways: a
constant (the same at every position of every text), a token part (the same for this token wherever it occurs), and a
context part (what the rest of the text changes). Tuomi (2026) showed that the constant part is not inert: it is the
reference each MLP gate's threshold is set against. Part 206 asked three things of Qwen3.5-4B (base and chat) and of
OLMo 2 1B across six training stages: how much of what MLPs and mixers write is constant, token-determined or
context-determined; who writes the constant, and does Tuomi's result hold in these models; and does putting a passage
inside an assistant turn shift that reference by one fixed vector at every position.

It does not, at this resolution. On Qwen3.5-4B chat, 0.106 of the assistant frame's difference is one fixed vector
(`rho_peak(A-tmpl, M)`, layer 18; 95% interval 0.103 to 0.110), against 0.2 asked for, and on OLMo 2 1B Instruct 0.164
(0.160 to 0.168). The share that is merely the same across passages, `kappa`, is only a little higher (0.117 on Qwen
chat, 0.182 on OLMo Instruct), so most of the frame's effect differs from position to position. Tuomi's static test
replicates in all eight models and his causal test in none: the duty change is 3.3 to 10.0 times the largest random control, short of the 20
times asked for. At OLMo Instruct the catalogued slot neurons do not arm under the assistant frame. Five of the 17
registered predictions held (206.4, 206.5, 206.7, 206.9 and 206.M3) and twelve failed.

Every number below is read from [results/part206/score206-output.txt](score206-output.txt), which prints each with its
file and key, or from the summary files it cites. The analysis is [part206/analyze206.py](../../part206/analyze206.py),
the causal check [part206/causal206.py](../../part206/causal206.py), the scorer [part206/score206.py](../../part206/score206.py),
and the figure [part206/fig206.py](../../part206/fig206.py).

## Predictions

The wording and confidence are copied from [PART206-DESIGN.md](../../PART206-DESIGN.md); the same table is in
[PREDICTIONS.md](../../PREDICTIONS.md) under "Part 206". The result column gives the values the scorer cited, with
their keys.

| # | Prediction (verbatim, with its confidence) | Result | Verdict |
| --- | --- | --- | --- |
| 206.1 | GT passes in both Qwen3.5-4B models. (medium, Claude (Fable 5.2)) | `gt.verdict.pass` is false in qwen-base and in qwen-chat: the static test passes in both, the causal test fails in both (ratios 9.98 and 9.84 to the largest random control). | failed |
| 206.2 | GT passes at five or more of the six OLMo 2 1B stages. (low, Claude (Fable 5.2)) | `gt.verdict.pass` is false at all six stages (ratios 3.26 to 3.56). | failed |
| 206.3 | In Qwen3.5-4B (chat), at the middle layer, MLP sublayers write at least half of the raw reference `r_L`. (low, Claude (Fable 5.2)) | `reference[16].write_share.mlp` = -0.3419. | failed |
| 206.4 | In Qwen3.5-4B (chat), frame P: the token share of the first layer's MLP output is at least 0.5, and the token share of the middle layer's MLP output is at most 0.25. (medium, Claude (Fable 5.2)) | `split.m[0].token_share` = 0.5564; `split.m[16].token_share` = 0.1182. | held |
| 206.5 | At OLMo 2 1B Instruct, frame P: the MLP's token share exceeds the same layer's mixer token share at three quarters or more of the layers in the first third. (Qwen's two mixer types are reported separately, unscored.) (medium, Claude (Fable 5.2)) | The MLP wins at 5 of the 6 first-third layers (`split.m[L].token_share` against `split.c[L].token_share`, layers 1 to 5; layer 0 loses, 0.2699 against 0.3232). The bar is 4.5. | held |
| 206.6 | In Qwen3.5-4B (chat): `rho_peak(A-tmpl, M)` is at least 0.2, its 95% interval from resampling passages has a lower end of at least 0.1, and at that layer the cosine between `Delta(A-tmpl, M)` and `Delta(A2-tmpl, M)` is at least 0.8. (medium, Claude (Fable 5.2)) | `common_mode.A-tmpl\|M.rho_peak` = 0.1060, `rho_peak_ci` = [0.1026, 0.1098], `cos_request_change[18]` = 0.9970. The first clause fails. | failed |
| 206.7 | Across the six OLMo stages, the largest rise in `rho_peak(A-tmpl, M)` between adjacent stages is from mid-training end to SFT. (medium, Claude (Fable 5.2)) | The mid-training-end to SFT rise is 0.0755 (interval 0.0714 to 0.0795), the largest of the five; the second largest is 0.0102. | held |
| 206.8 | At OLMo Instruct, under (A-tmpl, M), the slot neurons' median arming percentile is at least 10 points above the concept neurons' and is itself at least 60. (low, Claude (Fable 5.2)) | `arming.slot.median_percentile` = 49.05, `arming.concept.median_percentile` = 46.81, gap 2.25; `arming_label_shuffle.p95` = 16.87. | failed |
| 206.9 | In Qwen3.5-4B (chat), under (A-tmpl, M), the 16 identity neurons' median arming percentile is between 35 and 65: an assistant turn alone does not arm them. (medium, Claude (Fable 5.2)) | `arming.identity.median_percentile` = 45.97. | held |
| 206.11 | In Qwen3.5-4B (chat), `rho_peak(A-tmpl, R-tmpl)` is at least half of `rho_peak(A-tmpl, M)`: most of the constant shift is about who is speaking. (low, Claude (Fable 5.2)) | `common_mode.A-tmpl\|R-tmpl.rho_peak` = 0.0305 against 0.1060, a ratio of 0.29. At layer 18, where the (A-tmpl, M) peak sits, `rho(A-tmpl, R-tmpl)` is 0.0290, a ratio of 0.27. | failed |
| 206.12 | Across the six OLMo stages, the largest rise between adjacent stages in the gap between the slot and concept neurons' median arming percentiles under (A-tmpl, M) is from SFT to DPO. (low, Claude (Opus 5.5)) | The largest rise is mid-training 26B to mid-training end, 32.04; SFT to DPO is -0.043 (interval -6.12 to 15.09). | failed |
| 206.10 | In Qwen3.5-4B (chat), subtracting the assistant shift at `k = 1` lowers the maker token's log probability by more than every one of the 20 random directions and by more than the (M2, M) direction, and `k = 2` lowers it further. (low, Claude (Fable 5.2)) | `readout_a.mean_change[arm_k1]` = 0.0622 (it raises the log probability), `[null_M2M_k1]` = 0.0857, `[arm_k2]` = 0.0813 (also a rise); at `k = 1`, 4 of the 20 random directions raise it by more. | failed |
| 206.M1 | In Qwen3.5-4B (chat), `rho_peak(A-tmpl, M)` lies between 0.07 and 0.30 (point guess 0.15), and `rho(M2, M)` at the same layer between 0.00 and 0.01 (revised from 0.02 to 0.08 in #24, before registration, after Claude passed on the round-1 reviewer's simulation of the null). (rough 80%, Muse) | `rho_peak(A-tmpl, M)` = 0.1060, inside 0.07 to 0.30; `common_mode.M2\|M.layers[18].rho` = 0.0287, outside 0.00 to 0.01. | failed |
| 206.M2 | At OLMo Instruct, under (A-tmpl, M), the slot-minus-concept gap in median arming percentile is 10 to 20 points and the slot median is 60 to 70. (medium, Muse) | Gap 2.25, slot median 49.05. | failed |
| 206.M3 | In Qwen3.5-4B (chat), the 16 maker and company neurons' median arming percentile under (A-tmpl, M) is between 45 and 55. (medium, Muse) | `arming.identity.median_percentile` = 45.97. | held |
| 206.M4 | In Qwen3.5-4B (chat), frame P: the first layer's MLP token share is 0.50 to 0.70 and the middle layer's 0.10 to 0.25; with the part a straight line in log type frequency explains removed, the first layer's falls to 0.25 to 0.40. (medium, Muse) | `split.m[0].token_share` = 0.5564 and `split.m[16].token_share` = 0.1182 are both in range; `split.m[0].token_share_freq_removed` = 0.5435, not 0.25 to 0.40. | failed |
| 206.M5 | In Qwen3.5-4B (chat), at the layer of `rho_peak(A-tmpl, M)`, the cosine between `Delta(A-tmpl, M)` (on `x_L`) and frame M's `b_L` is at least 0.5. (medium, Muse) | `cos_delta_bM[18]` = -0.2507. | failed |

Five of the 17 rows held (206.4, 206.5, 206.7, 206.9, 206.M3) and twelve failed. The scorer's verdict JSON is
[score.json](score.json); the headline sentence it printed is the one in this README's title.

## What Part 206 asked, and what came out

The split (the first question) is the same shape in both families. On Qwen3.5-4B chat, frame P, the first layer's MLP
output is 0.556 token-determined of what varies (`split.m[0].token_share`) and 0.035 constant
(`split.m[0].constant`); at layer 16 it is 0.118 token-determined and 0.043 constant. At OLMo 2 1B Instruct the first
layer's MLP output is 0.270 token-determined and 0.031 constant. So on these frequent-token positions (coverage 0.537
of read positions on Qwen, 0.515 on OLMo, `split_coverage`) the MLP's write is mostly context-determined above the
first layer, and only a few per cent of it is constant. That is Elise's sentence in its literal form: the MLP is
token-determined at the bottom and not above it. The part does not score that sentence as a prediction, as the design
says.

The reference (the second question): on Qwen3.5-4B chat at the middle layer, the mixer writes 1.342 of the raw
reference `r_L`'s norm and the MLP -0.342 (`reference[16].write_share`), so the MLP writes against the reference
there, not for it. On OLMo Instruct at layer 6 the mixer writes 0.623 and the MLP 0.324. Tuomi's static test replicates
in all eight models; his causal test in none (below).

The assistant (the third question): the frame's difference is 0.106 of one fixed vector on Qwen chat and 0.164 on OLMo
Instruct, both under the 0.2 bar; the catalogued neurons' gates do not move (below). The registered headline rule's
sentence for this case is "the assistant frame changes each position in its own way; no constant shift at this
resolution", and it is this README's title.

## Does Tuomi's test replicate?

Half of it does. **Static** (his resting-term decomposition: each gate's resting term `w . b_L` against its duty,
beside 20 random directions of the same norm): the scorer's static clause holds in all eight models. The median
resting term is negative at all four layers (`gt.static[*].median_resting`), and the rank correlation with duty beats
all 20 random directions at 4 of 4 layers in every model (`gt.verdict.static_layers_beating_random`), where the
design asks for three of four. On Qwen chat the correlations are 0.960 at layer 6, 0.974
at layer 11, 0.970 at layer 16 and 0.949 at layer 20, against a largest random-direction correlation of 0.098, 0.130,
0.118 and 0.072 (`gt.static[*].rho_resting_duty`, `rho_random`). So the constant direction is the one each gate's
threshold is set against, as Tuomi found.

**Causal** (his single-layer removal at depth 0.31, `alpha = 0`, with 20 random directions at the same per-position
magnitude): the duty change is positive with an interval excluding 0 in all eight models, but it is only 3.3 to 10.0
times the largest random control, short of the 20 times GT asks for (`gt.verdict.ratio_to_largest_random`; Qwen chat
0.2289 against 0.0233 at layer 10, Qwen base 0.2278 against 0.0228, OLMo Instruct 0.1522 against 0.0447 at layer 5).
Tuomi's smallest reported ratio is 23 on Phi-2 against one control; ours is against the largest of 20, a stricter
comparison, and his Qwen2.5-1.5B contrast is about 3,000. So 206.1 and 206.2 fail on the causal clause, and the words
"re-thresholds the gates" are not used anywhere in this README.

## Is there a constant shift?

Not at this resolution, in either family. `rho_peak(A-tmpl, M)` is the largest `rho` over the middle third of the
layers, where `rho` is the share of the frame difference's power that is one fixed vector.

| model | layer | `rho_peak` | 95% interval | `kappa` at that layer | `rho_raw` at that layer |
| --- | --- | --- | --- | --- | --- |
| OLMo 2 1B, pre-training | 6 | 0.085 | 0.082 to 0.087 | 0.115 | 0.090 |
| OLMo 2 1B, mid-training 26B | 7 | 0.067 | 0.065 to 0.069 | 0.092 | 0.075 |
| OLMo 2 1B, mid-training end | 6 | 0.075 | 0.073 to 0.079 | 0.090 | 0.080 |
| OLMo 2 1B, SFT | 6 | 0.151 | 0.147 to 0.155 | 0.170 | 0.169 |
| OLMo 2 1B, DPO | 6 | 0.153 | 0.150 to 0.157 | 0.172 | 0.169 |
| OLMo 2 1B, Instruct | 6 | 0.164 | 0.160 to 0.168 | 0.182 | 0.179 |
| Qwen3.5-4B base | 17 | 0.076 | 0.073 to 0.078 | 0.099 | 0.093 |
| Qwen3.5-4B chat | 18 | 0.106 | 0.103 to 0.110 | 0.117 | 0.149 |

Each row is `common_mode.A-tmpl|M` in that stage's [summary.json](instruct/summary.json) (`rho_peak`,
`rho_peak_layer`, `rho_peak_ci`, `layers[L].kappa`, `layers[L].rho_raw`). The interval is the passage resampling
(1,000 draws; the Qwen chat draws run from 0.100 to 0.112, `rho_peak_resamples`).

A `rho` near 0 says the frame changes each position in its own way; a `rho` near 1 says it adds the same vector
everywhere. The largest value anywhere is 0.164 (OLMo Instruct), under the 0.2 bar, and the design's smallest effect
worth knowing is 0.1, so the differences between the small stages (0.067 to 0.085) are below it. `kappa` is the same
ratio taken position by position and averaged over positions: the share of the difference that is the same across
passages, where `rho` is the share that is also the same across positions. It is larger than `rho` at every row, which is what a shift that every
passage gets but that changes along the passage looks like (the design's `kappa` minus `rho`). On Qwen chat the gap is
0.011 (0.117 against 0.106) and on OLMo Instruct 0.018 (0.182 against 0.164). Both gaps are small, so a shift that
every passage gets but that varies along the passage accounts for little more than the fixed vector does; most of the
frame's difference is particular to each position.

Two descriptive facts about the pairs, not registered as predictions. First, the null pairs sit lower than the scored
ones in both families: on Qwen chat `rho_peak(M2, M)` is 0.0287 and `rho_peak(N2, N)` 0.0321, against 0.1060 for
(A-tmpl, M); on OLMo Instruct 0.0748 and 0.0612 against 0.1635. The design says the null pairs are not bars, so this
is a description of where the statistic sits, not a test. Second, changing the request leaves the shift nearly
unchanged: `cos_request_change` is 0.9970 at layer 18 on Qwen chat and 0.9952 at layer 6 on OLMo Instruct, so what the
frame adds is mostly the same vector whichever request is asked.

## Where is the MLP's token share high?

At the bottom of the network, and nowhere else. On Qwen3.5-4B chat the MLP's token share (`split.m[L].token_share`)
falls from 0.556 at layer 0 to 0.118 at layer 16, then rises again to 0.197 at layer 18 and 0.219 at layer 31. On OLMo
2 1B Instruct it falls from 0.270 at layer 0 to 0.084 at layer 5; the middle third (layers 6 to 10) is not reported
here. 206.4 held on the first-layer and middle-layer bounds; 206.M4 failed only on its third clause, the frequency
correction: with the part a straight line in log type frequency explains removed, the first layer's token share falls
from 0.5564 to 0.5435 (`split.m[0].token_share_freq_removed`), not to 0.25 to 0.40. On these positions the token part
is not a frequency artefact: `split.m[0].frequency_explained` is 0.023, so frequency explains 2.3% of the token part.

206.5 held: at OLMo Instruct the MLP's token share exceeds the mixer's at five of the six first-third layers (layers 1
to 5 are 0.236 against 0.182, 0.195 against 0.116, 0.150 against 0.088, 0.118 against 0.080 and 0.084 against 0.054;
layer 0 is the exception, 0.270 against 0.323). Five of six is "three quarters or more" of six, whose bar is 4.5. Qwen's two mixer types are reported separately and unscored, as the row says.

## Do slot neurons arm under the assistant frame?

No. Arming is the mean paired change in a neuron's gate pre-activation between the two frames, in units of its
standard deviation in the second frame, as a percentile among its layer's neurons (50 is typical). The catalogued sets
are Part 92's classification of OLMo 2 1B Instruct neurons by one Haiku reader (`results/part92/step2/kinds.json`), 21
slot and 18 concept, as [Amendment 4](../../part206/AMENDMENT-4.md)
changed them (`arming.slot.median_percentile`, `arming.concept.median_percentile`).

| stage | slot median | concept median | gap | shuffle 95th percentile |
| --- | --- | --- | --- | --- |
| pre-training | 52.46 | 47.16 | 5.30 | 30.05 |
| mid-training 26B | 35.92 | 42.27 | -6.35 | 9.25 |
| mid-training end | 63.32 | 37.63 | 25.69 | 30.04 |
| SFT | 52.45 | 48.71 | 3.74 | 12.27 |
| DPO | 49.81 | 46.12 | 3.69 | 15.43 |
| Instruct | 49.05 | 46.81 | 2.25 | 16.87 |

Each row is `arming.*.median_percentile` and `arming_label_shuffle.p95` in that stage's summary. At Instruct the slot
median is 49.05, under the 60 the row asks for, and the gap of 2.25 is under both the 10 points and the shuffle's 95th
percentile of 16.87, so 206.8 fails on all three clauses. The activity-matched version gives the same verdict
(`206.8_activity_matched`: slot 35.0, concept 37.5), so the verdict does not depend on the control. 206.M2 fails the
same way.

206.12's registered answer: the largest adjacent rise is mid-training 26B to mid-training end, +32.04, and SFT to DPO
is -0.043 (interval -6.12 to 15.09, which contains 0). Descriptive, not a registered row: the mid-training-end gap
itself (25.69) is the largest of the six stages, on a 39-neuron set whose concept half is weak (see Limits).

On Qwen chat the 16 maker and company neurons' median arming percentile is 45.97 (`arming.identity.median_percentile`),
inside the 35 to 65 band, so 206.9 and 206.M3 both held: an assistant turn alone does not arm them. The activity-matched
median is 40.0. One neuron of the 16 is far from typical in the other direction: layer 6, neuron 753, at the 96.13
percentile (`arming.identity.neurons`).

## Does removing the shift change the maker token?

It changes it, but not in the way 206.10 asked. The causal check subtracts `k` times the (A-tmpl, M) shift at `L*`
(layer 18 on Qwen chat, `causal.json` `L_star`) from the stream at every assistant-turn position, and reads the log
probability of the first token of the plain model's own maker name, "Tongyi" (the token "T"), at the position where the
plain answer names it. The rows are the 23 of 24 phrasings where the plain model names its maker at top-1
(`G3`: 23 of 24, pass). The means are over those 23 rows, as the design says.

| arm | mean change in log probability | rows raised |
| --- | --- | --- |
| assistant shift, `k = 1` | +0.0622 | 21 of 23 |
| assistant shift, `k = 2` | +0.0813 | 21 of 23 |
| (M2, M) direction, `k = 1` | +0.0857 | 21 of 23 |
| (M2, M) direction, `k = 2` | +0.1109 | 21 of 23 |
| 20 random directions, `k = 1` | -0.0889 to +0.1164 (median +0.0063) | |

Each arm's mean is `readout_a.mean_change[<arm>]` in [causal.json](qwen-chat/causal.json); the random directions'
range is the smallest and largest of the same key over `random0_k1` to `random19_k1`. Subtracting the shift *raises*
the maker token's probability by 0.0622, and `k = 2` raises it further, to 0.0813. The (M2, M) direction, which the
design uses as the null, raises it more at both doses (0.0857 and 0.1109). Four of the 20 random directions raise it
more than the shift at `k = 1`. So the first clause fails (the shift does not lower it more than every control) and the
second fails (it raises the probability, so "lowers it further" cannot hold). 206.10 failed.

The other two readouts of the causal check, reported and unscored: the catalogued neurons' mean gate pre-activation
above `L*` moves from -0.8587 plain to -0.8431 under the shift at `k = 1` (`readout_b`, the identity set), and the
off-target mean next-token loss on the wikitext test passages in frame A-tmpl moves from 3.0905 to 3.0988 at `k = 1`
and 3.1723 at `k = 2` (`readout_c`), against 3.0851 to 3.1154 over the random directions at `k = 1`. The shift costs
little off-target loss at `k = 1` and more at `k = 2`.

## Is it about who is speaking?

Mostly not, by the statistic the row uses. 206.11 compares `rho_peak(A-tmpl, R-tmpl)`, which changes only the role
word, with `rho_peak(A-tmpl, M)`: 0.0305 against 0.1060 on Qwen chat, a ratio of 0.29, under the half the row asks for.
At layer 18, where the (A-tmpl, M) peak sits, the role-only pair reads 0.0290, a ratio of 0.27, so the verdict does not
depend on which layer is read. So the constant part of the frame difference, such as it is, is not mostly the role
word. That is a statement about a statistic that is itself under 0.2; it does not say the role word is irrelevant, only
that it does not carry most of a shift that this part could not show to be constant.

## What changed in Amendment 5?

[Amendment 5](../../part206/AMENDMENT-5.md) is a scorer fix, written after the scores were seen. 206.M1 registers
`rho(M2, M)` at the same layer as between 0.00 and 0.01; the scorer still checked the pre-registration range,
0.02 to 0.08. The first scoring run (2026-11-13 14:52 CT) therefore reported 206.M1 as held. The scorer now checks
0.00 to 0.01 as registered, and one verdict changed: **206.M1, held to failed**. The measured values are unchanged:
`rho_peak(A-tmpl, M)` = 0.1060, inside 0.07 to 0.30, and `rho(M2, M)` at layer 18 = 0.0287, outside 0.00 to 0.01. No
registered text changed, and no other verdict changed. The four amendments before it are runner and analysis fixes
([Amendment 1](../../part206/AMENDMENT-1.md) a float64-on-the-A770 crash, [Amendment 2](../../part206/AMENDMENT-2.md)
the smoke's G2 read as a size check, [Amendment 3](../../part206/AMENDMENT-3.md) a scipy-free Spearman, and
[Amendment 4](../../part206/AMENDMENT-4.md) the catalogued neurons, which had been Qwen's coordinates on OLMo).

## Did the gates pass?

All four gates passed in all eight models, so every model was read: G0 (the do-nothing hook leaves the logits bit for
bit, `gates.G0`), G0.5 (every hook fires once and the parts sum to the stream, `gates.G0.5`), G1 (the same passage ids
across frames and stages, `gates.G1`) and G2 (the two halves' references agree, cosine at least 0.94 at every layer;
the lowest on Qwen chat is 0.9959 at layer 24, `gates.G2.layers`). G3 passed on Qwen chat at 23 of 24 phrasings, so
206.10 was read.

## Limits

The design's Limits, in its words, plus what the results add.

- The mechanical half only. Nothing here measures a report, an experience, or whether any constant is noticed.
- Passages are forced text. An assistant turn holding a wikitext passage is not something the model would write, and
  the model's own answers are not used because they differ by stage.
- One neutral request, two neutral prefixes, two sources, 96-token passages, positions 8 to 95. A shift that decays or
  grows beyond 95 tokens is not seen.
- GT replicates Tuomi's static decomposition and his single-layer removal at `alpha = 0`, with duty read only at the
  layer where the removal is made; his dose ladder, replacement panel and co-removal are not run.
- The split describes positions holding frequent token types only, about half the read positions (coverage 0.537 on
  Qwen, 0.515 on OLMo). Rare tokens, which may be more or less token-determined, are not measured.
- A shift whose size changes along the passage lowers `rho` and shows as `kappa` minus `rho`; `rho` alone does not
  distinguish a weak shift from a strong one that varies with position. The `kappa` column above reads it.
- A-tmpl differs from M in template tokens and the request as well as the role. Only R-tmpl, one token, isolates the
  role, and only for the template frames.
- The split's shares carry sampling noise of their own; each share is reported with a passage-resampled interval, but
  neither the fixture nor the interval shows the noise at every layer of these models.
- Readout (a) of the causal check is one token, the first token of the plain model's own maker name (Tongyi, per
  Part 153), on one model.
- OLMo 2's mid-training mix may already hold chat-formatted turns; if it does, part of the assistant-frame shift can
  appear before SFT, and 206.7 can fail for that reason rather than because post-training does not build it.
- Arming is a shift in mean gate pre-activation. It does not show that a neuron fires, or that its output changes an
  answer.
- The causal check uses one layer, two doses and one readout on one model. It supports or fails to support the reading;
  it does not establish a mechanism.
- Qwen runs in bf16; shares and means are not near-tie quantities, and Part 201's batch-shape finding concerns greedy
  text, which this part does not produce.
- OLMo 2 normalizes sublayer outputs, not inputs, so its reference lives in the raw stream; Tuomi found the
  mechanism's location depends on architecture, and a GT failure in OLMo would be about that, not about training.

What the results add:

- 206.7 held: `rho_peak(A-tmpl, M)` roughly doubles from mid-training end (0.075) to SFT (0.151). The pre-training
  checkpoint already reads 0.085, so some of the frame's fixed part predates post-training; whether the mid-training
  mix holds chat turns was not checked.
- Qwen ran with eager attention (`attn_implementation` in [config206.json](../../part206/config206.json)), not the
  default SDPA kernel whose bf16 output on the A770 is the front page's known issue (Part 178's finding).
- 206.8 and 206.12 rest on 39 neurons and one reader's kinds, not the 329 the design named, after Amendment 4. Only 8
  of the 18 concept neurons kept their kind under Part 92b's second (DeepSeek) reader, so the concept median is a weak
  quantity. The unscored reader-stable sets read slot 49.05 against concept 51.56 at Instruct
  (`arming.p92b_ds_slot.median_percentile`, `arming.p92b_ds_concept.median_percentile`), the same picture.
- With 39 neurons the label-shuffle null is wider, and its 95th percentile (16.87 at Instruct) is the effective bar for
  206.8, stricter than the 10 points the row states. Amendment 4 records this and leaves the bar as registered.
- The causal check's readout is one token on one model, and the shift raised its log probability rather than lowering
  it. The (M2, M) direction raised it more, so the shift's movement is not specific to it; the part reports the row as
  failed and does not read the rise as support for the mechanism.
- 206.3's value is negative (-0.342), so the MLP writes against the reference at the middle layer on Qwen chat. The
  row asks for "at least half"; the design defines the share as the group's signed projection on the reference's
  direction over its norm, so a negative share writes against it and is below half.

## How the run went

The build, smoke and run are recorded in [part206/steps.txt](../../part206/steps.txt) and the amendments. Three
failures were fixed before any scored number existed: the A770 has no float64 kernels (Amendment 1); the smoke's G2 is
a size check that 10 passages per half cannot meet (Amendment 2); the GPU environment has no scipy (Amendment 3). A
fourth was found after every stage had recorded: OLMo's catalogued neurons were Qwen's coordinates, and all six OLMo
stages died in the analysis before writing a summary (Amendment 4), which replaced them with Part 92's 21 slot and 18
concept OLMo neurons. The fifth is the scorer fix above (Amendment 5), made after the scores were seen.

Cold check: 2026-11-13, two fresh DeepSeek V4.1 Flash readers with only the files. The scorer check
([cold/SCORER-CHECK.md](cold/SCORER-CHECK.md)) compared every row's code with the registered text; its four
MISMATCH flags were refuted on reading (adjudication at its end), and the write-up worker then found its 206.5 win
count wrong (5 of 6, not 6 of 6; held either way). The README check ([cold/README-COLD.md](cold/README-COLD.md))
traced every number to its key and found none wrong; folded: panel A's title and bar label (206.6's 0.2 is scored on
Qwen chat only), `kappa` defined as the design defines it, the static clause's registered three-of-four bar and which
values were read, the mid-training-end gap labelled descriptive after 206.12's registered answer, and Part 92 and
92b's readers named. Nothing left unfolded.

## Files

- [score206-output.txt](score206-output.txt), [score.json](score.json): every verdict, printed by
  [part206/score206.py](../../part206/score206.py) with its file and key.
- [fig206.png](fig206.png): drawn by [part206/fig206.py](../../part206/fig206.py) from the eight summaries and
  [qwen-chat/causal.json](qwen-chat/causal.json).
- [pretrain-3.0T/summary.json](pretrain-3.0T/summary.json), [midtrain-26B/summary.json](midtrain-26B/summary.json),
  [midtrain-end/summary.json](midtrain-end/summary.json), [sft/summary.json](sft/summary.json),
  [dpo/summary.json](dpo/summary.json), [instruct/summary.json](instruct/summary.json),
  [qwen-base/summary.json](qwen-base/summary.json), [qwen-chat/summary.json](qwen-chat/summary.json): the analysis
  output per stage, written by [part206/analyze206.py](../../part206/analyze206.py).
- [qwen-chat/causal.json](qwen-chat/causal.json), [instruct/causal.json](instruct/causal.json): the causal check's
  readouts, written by [part206/causal206.py](../../part206/causal206.py).
