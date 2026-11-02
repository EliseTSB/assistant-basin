# [Part 174](../../PART174-DESIGN.md): [28 of 80](score.json) layer-matched random directions led with Google, [0](score.json) with any other name; flat directions broke at this dose

At the registered dose, |d| = [0.5](../../PART174-DESIGN.md), [28 of 80](score.json) [u100d](../../PART174-DESIGN.md) signed directions met the support rule; all [28](score.json) were Google-led and [none](score.json) was led by another name. [f100d](../../PART174-DESIGN.md), with the same written length and dose, had [0 of 40](prediction_table.json) supported directions because nearly every answer broke.

![Each dot is one signed direction. The upper strip spreads smoothly from zero to 27 Google claims with no gap at the floor. In the lower flat-profile strip, 8 directions have zero claims and 32 sit in a separate column because they had no clean answers to read. Directions with unlabelled answers sit in their own column, and the dashed line marks the modal-count floor of 10; support also needs at most 8 broken answers.](plot174.png)

This is a rate for these two dense draw generators at one dose, on fixed maker prompts and conditional on every draw passing Gate [8](gates.json). It is not a rate for arbitrary random pushes.

## How often did a profile-matched push lead with Google?

The [u100d](../../PART174-DESIGN.md) family used [40](score.json) draws at both signs of [d = 0.5](../../PART174-DESIGN.md). [24 of 40](score.json) draws had at least one supported sign: [20](score.json) had one sign supported and [4](score.json) had both signs supported under the same name. All [28 of 80](score.json) supported signed directions were Google-led; there were [0 of 80](score.json) other-led directions.

At d = [+0.5](../../PART174-DESIGN.md), [12 of 40 (30%)](score.json) directions were supported; the Wilson [95%](score.json) interval was [18.1–45.4%](score.json). At d = [−0.5](../../PART174-DESIGN.md), [16 of 40 (40%)](score.json) were supported; the interval was [26.3–55.4%](score.json). Pooling the two signs within each draw gives [28 of 80 (35%)](score.json), with a draw-level cluster-bootstrap [95%](score.json) interval of [25–45%](score.json), using [10,000](score.json) resamples and holding prompts fixed. Question-level bootstraps with directions held fixed are [28.8–38.8%](score.json) for [u100d](../../PART174-DESIGN.md) and [0–0%](score.json) for [f100d](../../PART174-DESIGN.md); they say nothing about new draws.

The exact one-sided [95%](../../PART174-DESIGN.md) upper bound for an either-sign other-led result across [40](score.json) draws is [7.2%](score.json), using the registered formula [1 − 0.05^(1/40)](../../PART174-DESIGN.md). The result does not show that a random push never names anyone else.

## Does the support rate depend on its count floor?

The registered support rule also requires no more than [8 broken answers out of 44](../../PART174-DESIGN.md). Here is the [u100d](../../PART174-DESIGN.md) support rate under alternative modal-count floors:

| Modal-count floor | Supported signed directions |
| --- | ---: |
| [5](score.json) | [38 of 80 (47.5%)](score.json) |
| [10](score.json) | [28 of 80 (35%)](score.json) |
| [15](score.json) | [19 of 80 (23.8%)](score.json) |

[7](corrections.json) directions have clean answers the reader could not label. Those answers cannot lift any direction over any of these floors: each of the seven has too many broken answers, too low a count, or is [already counted](corrections.json). So the table is complete.

The count of directions depends on which directions were read. [32](corrections.json) `f100d` directions had no clean answers, so nothing was read there. [score.json](score.json) stores those as zero Google claims, and its histogram's total of 113 and zero count of 53 are wrong for that reason. Without them and the [7](corrections.json) directions above, [81](corrections.json) directions are labelled, [73](corrections.json) `u100d` and [8](corrections.json) `f100d`. [21](corrections.json) have zero Google claims and [26](corrections.json) have between 5 and 15.

Within `u100d` the counts run from [0 to 27](corrections.json). The largest single bin is zero, at [13](corrections.json) of [73](corrections.json), and there is no gap around the floor: [1, 1, 2 and 2](corrections.json) directions have 8, 9, 10 and 11 claims. So the 35% says where a floor of 10 cuts a continuum, not how many directions belong to some natural class, and the table above shows how far it moves. The mass does not sit just above the floor either: [16](corrections.json) labelled directions have 10 to 15 claims and [17](corrections.json) have more.

## Are Google-led answers claims or mentions?

| Family | Answers, broken | Relation labels: claim / none | Google as primary name | Google mentions |
| --- | ---: | ---: | ---: | ---: |
| [u100d](../../PART174-DESIGN.md) | [3,520, 506](corrections.json) | [2,049 / 957](score.json) | [715](score.json) | [771](score.json) |
| [f100d](../../PART174-DESIGN.md) | [1,760, 1,732](corrections.json) | [0 / 28](score.json) | [0](score.json) | [0](score.json) |

In [u100d](../../PART174-DESIGN.md), the reader resolved Google as the primary name [715](score.json) times; it appeared in [771](score.json) mention counts, including mentions where it was not the primary. The relation totals are across all entities in that family. In [f100d](../../PART174-DESIGN.md), all [28](score.json) clean answers had relation “none” and none named or mentioned Google. Those 28 are all that was read of its [1,760](corrections.json) answers. A mention count alone is not a Google-led direction.

## What happened to the flat-profile family?

The [f100d](../../PART174-DESIGN.md) family had [0 supported, 0 Google-led and 0 other-led](score.json) of [40](prediction_table.json) directions. At each sign it had [0 of 20](score.json) supported directions, with a Wilson [95%](score.json) interval of [0–16.1%](score.json). No draw had either sign supported; the exact one-sided upper bound is [13.9%](score.json). Its median was [44 broken answers out of 44](scored_directions.jsonl), with a range of [32 to 44](scored_directions.jsonl). Flat-profile pushes were unusable at this dose. This does not tell whether a flat-profile push would name Google when its answers do not break the model.

The two families had the same written length and dose, but [f100d](../../PART174-DESIGN.md) spreads energy across [21](../../PART174-DESIGN.md) layers and adds [3](../../PART174-DESIGN.md) layers relative to [u100d](../../PART174-DESIGN.md). The design does not attribute their difference to shape alone. Prediction [174.5](prediction_table.json) failed, and the falsifier for [174.W3](prediction_table.json) fired because the flat family's supported share fell below half amid near-total breakage. Prediction [174.10](prediction_table.json) is “not read”; its rule requires supported [f100d](../../PART174-DESIGN.md) directions before it can ask which name leads. The fallback reading needs [174.2](prediction_table.json) and [174.10](prediction_table.json) both read and held, so it is [not read](score.json): this part neither supports nor refutes it.

## Do the projections explain the sign contrast?

The registered tests [174.6](prediction_table.json) and [174.9](prediction_table.json) are “not read.” Both ask across the 60 draws, and that cannot be done: [20 of 20](corrections.json) `f100d` draws have a sign with no clean answers, so D does not exist for them. In `u100d`, [7](corrections.json) draws carry the [8](score.json) unlabelled clean answers and D is missing there too.

The design's not-read list names repeated D values, too small a spread and a nonfinite correlation. It has no rule for a missing D. The registered scorer stops on any missing input, and that is what produced the verdict; [score.json](score.json) gives that first reason only. We keep “not read”, since a D that was never measured is not a zero. [score.json](score.json) also lists D = 0 for those `f100d` draws, and those zeros are not measurements. This is not a zero correlation.

`u100d` alone can be described, and this is not 174.6 or 174.9. On its [33](corrections.json) complete draws cf rho is [0.220](corrections.json), interval [−0.166 to 0.549](corrections.json), and cr rho is [0.036](corrections.json), interval [−0.301 to 0.363](corrections.json). Across all 40 draws, every one of the [192](corrections.json) ways the unlabelled answers could fall gives cf rho [0.184 to 0.207](corrections.json) and cr rho [0.025 to 0.052](corrections.json). Those stay under 0.30, and the intervals are wide enough that this does not bound the true value there.

The pre-answer projection spreads were cf SD [0.02118](stage0.json) and cr SD [0.02125](stage0.json) across [60](stage0.json) draws. The frozen token IDs were [4923](stage0.json) for “ Google” and [52540](stage0.json) for “ Alibaba”. The descriptive fallback gives pooled raw-word-hit correlations of cf rho [0.140](score.json), [95%](score.json) interval [−0.102 to 0.369](score.json), and cr rho [0.104](score.json), interval [−0.137 to 0.334](score.json). For `u100d` alone they are cf rho [0.179](score.json), interval [−0.156 to 0.479](score.json), and cr rho [0.141](score.json), interval [−0.164 to 0.426](score.json); `f100d` is [not read](score.json) because its word-hit difference never varies. These word-hit correlations are descriptive only, not the registered endpoints.

## Registered predictions

The registered wording and confidence labels are from [PREDICTIONS.md](../../PREDICTIONS.md). Estimates and every verdict below come from [prediction_table.json](prediction_table.json); the verdicts are not re-derived here.

| # | Registered prediction (confidence) | Observed | Verdict |
| --- | --- | --- | --- |
| 174.1 | [At least 20 of the 80 `u100d` signed directions are supported](../../PREDICTIONS.md) ([medium](../../PREDICTIONS.md)) | [28 of 80](prediction_table.json) | [held](prediction_table.json) |
| 174.2 | [Google is modal on at least 80% of supported `u100d` directions; not read below 10 supported](../../PREDICTIONS.md) ([medium](../../PREDICTIONS.md)) | [28 of 28 (100%)](prediction_table.json) | [held](prediction_table.json) |
| 174.3 | [At most 4 of the 80 `u100d` signed directions are other-led](../../PREDICTIONS.md) ([medium](../../PREDICTIONS.md)) | [0 of 80](prediction_table.json) | [held](prediction_table.json) |
| 174.4 | [Of the `u100d` draws with at least one supported sign, at most 25% are `both same` or `both different`; not read below 8 such draws](../../PREDICTIONS.md) ([low](../../PREDICTIONS.md)) | [4 of 24 (16.7%); all four both same](score.json) | [held](prediction_table.json) |
| 174.5 | [`f100d`'s supported share is at least half of `u100d`'s; literal result always reported, read as retention only if `u100d` has at least 10 supported, else `uninformative`](../../PREDICTIONS.md) ([low](../../PREDICTIONS.md)) | [0 of 40 vs 28 of 80](prediction_table.json) | [failed](prediction_table.json) |
| 174.6 | [Across the 60 draws, the observed \|Spearman(cf, D)\| is under 0.30](../../PREDICTIONS.md) ([medium](../../PREDICTIONS.md)) | Not read: no `f100d` draw has a D, and [7](corrections.json) `u100d` draws lack one | [not read](prediction_table.json) |
| 174.7 | [The median `broken` of the 80 `u100d` signed directions is at most 8 of 44](../../PREDICTIONS.md) ([medium](../../PREDICTIONS.md)) | [Median 5 of 44](prediction_table.json) | [held](prediction_table.json) |
| 174.8 | [At least 60% of all supported directions are `stable`; not read below 5 supported](../../PREDICTIONS.md) ([medium](../../PREDICTIONS.md)) | [27 of 28 (96.4%)](prediction_table.json) | [held](prediction_table.json) |
| 174.9 | [Across the 60 draws, the observed \|Spearman(cr, D)\| is under 0.30](../../PREDICTIONS.md) ([low](../../PREDICTIONS.md)) | Not read: no `f100d` draw has a D, and [7](corrections.json) `u100d` draws lack one | [not read](prediction_table.json) |
| 174.10 | [Google is modal on at least 80% of supported `f100d` directions; not read below 5 supported](../../PREDICTIONS.md) ([medium](../../PREDICTIONS.md)) | No supported directions; share not read | [not read](prediction_table.json) |
| 174.W1 | [At least 30 of the 80 `u100d` signed directions are supported. Falsifier: fewer than 20, which would say the three draws were a lucky sample rather than a rate. 20 to 29 is inconclusive](../../PREDICTIONS.md) ([medium](../../PREDICTIONS.md)) | [28 of 80](prediction_table.json) | [inconclusive](prediction_table.json) |
| 174.W2 | [Other-led directions are present but few, and the majority name invented entities. Falsifier: at least half of the other-led directions name a known non-Qwen entity. Not testable with no other-led direction. Held if a strict majority of the other-led directions are invented; how few they are is reported, not scored. Invented means outside the entity table, not shown to be fictional](../../PREDICTIONS.md) ([medium](../../PREDICTIONS.md)) | [0 other-led directions](prediction_table.json) | [not testable](prediction_table.json) |
| 174.W3 | [`f100d`'s supported share is within 20 points of `u100d`'s. Falsifier: `f100d` is below half of `u100d`, which would make the layer profile or the spread a real gate, without saying which. The falsifier takes precedence where both are true. Any other outcome is inconclusive](../../PREDICTIONS.md) ([medium](../../PREDICTIONS.md)) | [0% vs 35%; difference 35 points and below half](prediction_table.json) | [failed](prediction_table.json) |

## Method and gates

The model was [Qwen3.5-4B](../../PART174-DESIGN.md) at snapshot [851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a](../../PART174-DESIGN.md), with greedy decoding, up to [80](../../PART174-DESIGN.md) tokens, thinking off. The Part [171](../../PART174-DESIGN.md) AddEveryPositionHook applied the push at every prompt and reply position. The [44](score.json) [NEW44B](../../PART174-DESIGN.md) maker prompts were reused from [Part 173](../part173/README.md); new draws did not select these prompts. The values were sampled without fit: [u100d](../../PART174-DESIGN.md) kept [n100](../../PART174-DESIGN.md)'s layer counts, while [f100d](../../PART174-DESIGN.md) spread equal energy across [21](../../PART174-DESIGN.md) layers, adding [3](../../PART174-DESIGN.md). Both were scaled to [n1](../../PART174-DESIGN.md)'s written length. The [40](stage0.json) [u100d](../../PART174-DESIGN.md) and [20](stage0.json) [f100d](../../PART174-DESIGN.md) draws were each read at [d = ±0.5](../../PART174-DESIGN.md).

That is [5,280](../../PART174-DESIGN.md) direction answers, plus [44](gates.json) none, [44](gates.json) role, and [88](answers.jsonl) replay rows. The run contains [5,456](score.json) answer rows, including replays.

| Gate | Check | Measured result |
| --- | --- | --- |
| [1](gates.json) | Planned rows and unique keys | [5,368 of 5,368; 0 missing or unexpected](gates.json) |
| [2](gates.json) | none baseline | [43 of 44 Qwen-family hits; 0 of 44 Google hits](gates.json) |
| [3](gates.json) | Zero push, then hooks removed | [8 of 8 exact matches in each check](gates.json) |
| [4](gates.json) | [Replay of u100d_d2173 at d = 0.5 and n3 at d = −0.5](gates.json) | [44 of 44 token matches in each replay; 88 rows total](gates.json) |
| [5](gates.json) | role arm | [44 of 44 Google hits; 38-hit minimum](gates.json) |
| [6](gates.json) | Dose delivery through the real hook | [120 of 120 checks passed within 5%; no failures](gates.json) |
| [7](gates.json) | Prompt identity | [44 prompts; hash matched Part 173; 0 overlap with P8, M96 or NEW44](gates.json) |
| [8](stage0.json) | Length and Google-fit projection eligibility | [60 of 60 passed; max observed \|cf\| 0.0581, below the 0.079 screen; length deviation under 1%](stage0.json) |
| [9](score.json) | Reader calibration and agreement | [20 of 20 known examples for each reader; relation κ = 0.964 over 298 pairs; primary agreement 149 of 149 replacement-stratum pairs and 171 of 171 answer-weighted](score.json) |

Outside that stratum, both readers called [222](labels/labels-haiku.jsonl) of the [298](labels/labels-haiku.jsonl) shared pairs claims. They resolved the same primary on 218; on the other 4 DeepSeek gave no primary and Haiku gave Tencent or Google. No pair had two different names.

The cf projection was against the saved Google fit vector. The cr projection used the unit difference between output-embedding rows for the frozen tokens [4923](stage0.json) (“ Google”) and [52540](stage0.json) (“ Alibaba”); both token IDs were asserted by the run.

Readers saw each unique clean question-answer pair alone, with arm and dose hidden. They labelled [2,972](readers.log) pairs. After the retry pass, [8](readers.log) pairs remained failed, [0.27%](readers.log) of [2,972](readers.log), below the [0.5%](readers.log) error cap. Eight clean answers across seven directions therefore remained unlabelled. The reader script's first start failed within a second under the wrong Python before any answers were read; it was rerun. The log records only the second start.

## Limits

This result covers one model snapshot, greedy decoding, one dose, the fixed [44](score.json) prompts and these two dense draw generators conditional on Gate [8](gates.json). It does not show that Google is the model's fallback name, that a random push never names another maker, or that the layer profile causes the lead. The comparison with picked sets in Part [173](../part173/README.md) is descriptive and uses a different reader pass. The [three dose-discovery draws](../../PART174-DESIGN.md) from Part [173](../part173/README.md) selected the dose; they are not census observations and are not pooled here. This result gives no rate outside this dose, these prompts, and Gate [8](gates.json)'s screen.

## Files

- [score.json](score.json): aggregate counts, floors, uncertainty, correlations, and reader gate.
- [prediction_table.json](prediction_table.json): registered estimates and verdicts.
- [scored_directions.jsonl](scored_directions.jsonl): one row per signed direction, including broken and claim counts.
- [gates.json](gates.json) and [stage0.json](stage0.json): generation gates, projections, and eligibility measurements.
- [readers.log](readers.log): reader run and retry outcome.
- [answers.jsonl](answers.jsonl), [labels/labels-ds.jsonl](labels/labels-ds.jsonl), and [labels/labels-haiku.jsonl](labels/labels-haiku.jsonl): generated answers and reader labels.
- [plot174.py](plot174.py) writes [plot174.png](plot174.png) from scored_directions.jsonl.
- [corrections.json](corrections.json), written by [correct174.py](correct174.py) from the frozen files: the counts above that do not treat an unread direction as zero. [score.json](score.json) and [prediction_table.json](prediction_table.json) are the registered scorer's output and are unchanged. Under each family, [score.json](score.json) gives the pooled 60-draw projection spread; the family spreads are in its descriptive fallback.

Cold check: Haiku 5.5, 2026-11-02, reading only the files, reproduced every headline count; its [12 findings](COLD.md) are folded here, and its figure of 53 complete draws is replaced by 33, since the `f100d` draws it kept have no real D.

