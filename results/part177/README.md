# [Part 177](../../PART177-DESIGN.md): [67](numbers177.json) of [173](numbers177.json) unedited [Qwen3.5-0.8B](../../PART177-DESIGN.md) traces that ran out of tokens while thinking had already stated the right answer, against [0](numbers177.json) for shuffled answers

Scope first: this is a read of MIT's stored sample-[0](../../PART177-DESIGN.md) texts for the first [200](numbers177.json) rows of [MATH-500](../../PART177-DESIGN.md) in each arm, the same problem IDs. The authors used the model and stack in the design, with thinking on and the registered sampling: temperature [0.6](../../PART177-DESIGN.md), top_p [0.95](../../PART177-DESIGN.md), top_k [20](../../PART177-DESIGN.md), presence penalty [1.5](../../PART177-DESIGN.md), up to [32,768](../../PART177-DESIGN.md) new tokens on vLLM. Nothing was run here. `capped` means decoded text retokenised to at least [32,500](../../PART177-DESIGN.md) tokens with the pinned release tokenizer. MIT's [post](https://rikiyat.github.io/blog/weight-space-programming/) says its edit “could remove an obstacle that hid an ability.” This part tests that sentence on [1](../../PART177-DESIGN.md) narrow reading. On the unedited arm, [187](numbers177.json) traces were capped while thinking remained open. Among target traces, the reader judged that [67](numbers177.json) of [173](numbers177.json) had stated the right answer; it marked [0](numbers177.json) shuffled draw-[1](numbers177.json) answers asserted. A stated answer is not proof the model would have stopped.

![Code-drawn panels: target funnel, onset order, and paired line detections.](plot177.png)

## What did the cap and prompt-echo filter count?

The text stores no finish reason. `capped` is the retokenised decoded-text proxy above; `</think>` marks closed thinking.

| Arm | Capped, thinking open | Capped, thinking closed | Not capped, thinking open | Not capped, thinking closed |
| --- | ---: | ---: | ---: | ---: |
| unedited | [187](numbers177.json) | [0](numbers177.json) | [0](numbers177.json) | [13](numbers177.json) |
| p50c | [48](numbers177.json) | [22](numbers177.json) | [0](numbers177.json) | [130](numbers177.json) |

The [13](numbers177.json) unedited traces that finished are the not-capped, thinking-closed row, so they are outside cohort `C`. The gold-in-prompt filter removes [14](numbers177.json) of the [187](numbers177.json) traces in `C`: literal gold appears in [115](numbers177.json)/[187](numbers177.json) ([61.5%](numbers177.json)) on `C`, and [101](numbers177.json)/[173](numbers177.json) ([58.4%](numbers177.json)) on `T`. The shuffled draw-[1](numbers177.json) value appears in [33](numbers177.json)/[173](numbers177.json) ([19.1%](numbers177.json)) on `T`.

The target-`T` funnel counts traces, not windows: [173](numbers177.json) traces; real gold literal hit [101](numbers177.json); real gold judged asserted [67](numbers177.json); shuffled draw-[1](numbers177.json) literal hit [33](numbers177.json); shuffled judged asserted [0](numbers177.json). The panel shows the same filter and control.

## Is the window reader reliable enough for this claim?

A person judges selected windows around literal hits, not the full trace. The known-window gate got [17](numbers177.json)/[20](numbers177.json) ([85.0%](numbers177.json)), at its minimum of [17](numbers177.json). All [3](numbers177.json) known-window misses were among the [4](numbers177.json) examples expected to be `ambiguous`; the remaining [1](numbers177.json) was labelled `ambiguous` correctly. Agreement with the second reader was kappa [0.62](numbers177.json) on [38](numbers177.json) pairs against a [0.6](numbers177.json) bar. Both checks passed narrowly.

The real asserted-window share was [120](numbers177.json)/[287](numbers177.json) ([41.8%](numbers177.json)); the shuffled share was [0](numbers177.json)/[88](numbers177.json) ([0.0%](numbers177.json)). That is a gap of [41.8](numbers177.json) percentage points, above the registered [25](../../PART177-DESIGN.md) point gate. Reader errors were [0](numbers177.json)/[385](numbers177.json) ([0.0%](numbers177.json)), under the [2%](../../PART177-DESIGN.md) cap. Gate [4](numbers177.json) passed.

| Reader label | Real windows | Shuffled draw-[1](numbers177.json) windows |
| --- | ---: | ---: |
| asserted | [120](numbers177.json) | [0](numbers177.json) |
| intermediate | [130](numbers177.json) | [64](numbers177.json) |
| quoted from the problem | [6](numbers177.json) | [3](numbers177.json) |
| rejected or doubted | [22](numbers177.json) | [3](numbers177.json) |
| ambiguous | [9](numbers177.json) | [18](numbers177.json) |
| Total | [287](numbers177.json) | [88](numbers177.json) |

## Did the asserted value appear before the detected onset?

Prediction [177.3](prediction_table.json) held: the first asserted selected window was before the line-rule onset in [49](numbers177.json)/[66](numbers177.json) ([74.2%](numbers177.json)). The remaining asserted trace has no detected onset, shown in its own row. Across all literal-hit traces in `C`, the table keeps no-onset traces separate. The onset rule is not ground truth.

| Trace group | Onset order | Traces |
| --- | --- | ---: |
| Asserted selected windows on `T` (n=[67](numbers177.json)) | Before onset | [49](numbers177.json) |
| Asserted selected windows on `T` (n=[67](numbers177.json)) | Onset before asserted hit | [17](numbers177.json) |
| Asserted selected windows on `T` (n=[67](numbers177.json)) | No onset detected | [1](numbers177.json) |
| Literal-hit traces on `C` (n=[115](numbers177.json)) | Hit before onset | [110](numbers177.json) |
| Literal-hit traces on `C` (n=[115](numbers177.json)) | Onset before hit | [3](numbers177.json) |
| Literal-hit traces on `C` (n=[115](numbers177.json)) | No onset detected | [2](numbers177.json) |

This prediction's denominator uses traces with an asserted gold and a detected onset. Its [95%](numbers177.json) interval is [63.6](numbers177.json) to [84.8](numbers177.json)%.

Across all `C` traces with an onset, an asserted gold preceded it in [49](numbers177.json)/[168](numbers177.json) ([29.2%](numbers177.json)). The prediction above keeps only traces that had an asserted gold.

Among all `C` traces, `asserted` appears on [66](numbers177.json)/[182](numbers177.json) ([36.3%](numbers177.json)) looped traces and [1](numbers177.json)/[5](numbers177.json) ([20.0%](numbers177.json)) merely-long traces. Among read hits, the first read hit was asserted in [36](numbers177.json)/[101](numbers177.json) ([35.6%](numbers177.json)), the last read hit in [47](numbers177.json)/[101](numbers177.json) ([46.5%](numbers177.json)), and [22](numbers177.json)/[287](numbers177.json) ([7.7%](numbers177.json)) were rejected or doubted.

## What does the line rule show after the edit?

Prediction [177.4](prediction_table.json) failed: [35](numbers177.json)/[48](numbers177.json) ([72.9%](numbers177.json)) of the p50c capped traces with open thinking had a detected onset, below the [80%](../../PART177-DESIGN.md) bar. This is the line signature among residual failures. It persisted in about three quarters of them and says nothing about incidence.

Prediction [177.5](prediction_table.json) held on the median. In [34](numbers177.json) paired problems with an onset in each arm, the median onset fraction was [0.1624](numbers177.json) in unedited and [0.3179](numbers177.json) in p50c. The difference interval, [-0.0279](numbers177.json) to [0.4001](numbers177.json), includes [0](numbers177.json).

Prediction [177.6](prediction_table.json) held over [116](numbers177.json) paired texts with equal exposure to the first [50,000](../../PART177-DESIGN.md) characters: [111](numbers177.json)/[116](numbers177.json) ([95.7%](numbers177.json)) unedited and [79](numbers177.json)/[116](numbers177.json) ([68.1%](numbers177.json)) p50c. The rate difference was [27.6](numbers177.json) percentage points, with [95%](numbers177.json) interval [19.0](numbers177.json) to [37.1](numbers177.json) percentage points. This is the rate of the line rule's signature in the first [50,000](../../PART177-DESIGN.md) characters, not a measure of loops in general.

The full-text end-loop rule flagged [182](numbers177.json)/[200](numbers177.json) ([91.0%](numbers177.json)) unedited traces and [36](numbers177.json)/[200](numbers177.json) ([18.0%](numbers177.json)) p50c traces. `verbatim_loop_end` follows the pinned `mathqa.py` rule. `repetitive_end` was rebuilt from a docstring because `mcqa.py` is not in the logs.

## Did early prompt echo explain the literal-hit gap?

On `C`, real literal hits were [115](numbers177.json)/[187](numbers177.json) ([61.5%](numbers177.json)) and shuffled draw-[1](numbers177.json) hits were [37](numbers177.json)/[187](numbers177.json) ([19.8%](numbers177.json)). The all-occurrence early-hit share was [3923](numbers177.json)/[37453](numbers177.json) ([10.5%](numbers177.json)). Its first-hit reading was [109](numbers177.json)/[115](numbers177.json) ([94.8%](numbers177.json)). These are different measures because repeated values can appear many times in a trace.

| Population | First hit in first [15%](../../PART177-DESIGN.md) |
| --- | ---: |
| All `C` traces with a real hit | [109](numbers177.json)/[115](numbers177.json) ([94.8%](numbers177.json)) |
| Target `T` traces with a real hit | [95](numbers177.json)/[101](numbers177.json) ([94.1%](numbers177.json)) |
| `C` minus `T`, where gold is in prompt | [14](numbers177.json)/[14](numbers177.json) ([100.0%](numbers177.json)) |

After dropping hits in the first [15%](../../PART177-DESIGN.md) of text but keeping all of `C` in the denominator, real hits remained on [84](numbers177.json)/[187](numbers177.json) ([44.9%](numbers177.json)) traces and shuffled hits remained on [25](numbers177.json)/[187](numbers177.json) ([13.4%](numbers177.json)). The all-`C` late gap was [31.6](numbers177.json) percentage points. Amendment [1](../../PART177-DESIGN.md) corrected the scorer to keep all of `C` in that denominator and to use the registered inconclusive state. The final [177.W1](prediction_table.json) verdict is inconclusive: the early-hit share has [2](numbers177.json) readings, the late gap stays above [15](../../PART177-DESIGN.md) points, and neither the full claim nor the under-[10%](../../PART177-DESIGN.md) falsifier holds.

Prediction [177.W2](prediction_table.json) failed at [0](numbers177.json)/[98](numbers177.json) ([0.0%](numbers177.json)). No shuffled windows were judged asserted. This is the direction that makes the reader look discriminating.

## Where in the text were the first hits?

These token-position summaries are descriptive and unregistered. A token position counts the tokenized stored-text prefix before the character position.
On target `T`, the median tokens before the first literal hit was [652](numbers177.json), with IQR Q1 [338](numbers177.json) to Q3 [1486](numbers177.json), across [101](numbers177.json) traces.
Before the first selected window labelled asserted, the median was [1986](numbers177.json) tokens, with IQR Q1 [622](numbers177.json) to Q3 [5905](numbers177.json), across [67](numbers177.json) traces.

## Did the literal-hit traces emit a wrong final box?

Under Weaver's reading (a), [0](numbers177.json)/[115](numbers177.json) ([0.0%](numbers177.json)) had a non-null box that failed [`correct[0]`](numbers177.json), so [177.W3](prediction_table.json) failed. But [113](numbers177.json)/[115](numbers177.json) ([98.3%](numbers177.json)) had no correct box when nulls are included. [90](numbers177.json) had no `\boxed` or `\fbox` token in the stored text; [23](numbers177.json) ended on an empty box. The raw text scan found [35](numbers177.json) box tokens across [25](numbers177.json) traces: [32](numbers177.json) empty `\boxed{}` occurrences in [24](numbers177.json) traces and [3](numbers177.json) filled boxes. All [32](numbers177.json) empty tokens occur near repeated final-answer or output-format instructions. For example, an echoed instruction says to put the final answer within the box, followed by an empty `\boxed{}` before the reasoning continues. These boxes contain no answer. The extraction returns null for an empty last box, just as it does when no box exists.

An earlier box held the literal gold while the last box differed in [1](numbers177.json)/[115](numbers177.json) ([0.9%](numbers177.json)). There were [115](numbers177.json) literal-hit traces; [113](numbers177.json)/[115](numbers177.json) ([98.3%](numbers177.json)) had null `pred_first`. In these rows, null `pred_first` is what running out while still thinking looks like: almost nothing was emitted as an extractable answer. It is an observed output state, not missing data.

## How do the paired correctness tables line up?

The first table uses all paired problems. The second uses `C`, where the unedited trace has a reader label. Each table gives the p50c wrong and correct cells. Rates are p50c [`correct[0]`](numbers177.json) within each unedited row.

### Unedited literal hit vs p50c correctness on all paired problems

| Unedited state | p50c wrong | p50c correct | p50c correct rate in row |
| --- | ---: | ---: | ---: |
| No | [57](numbers177.json) | [15](numbers177.json) | [15](numbers177.json)/[72](numbers177.json) ([20.8%](numbers177.json)); [95%](numbers177.json) CI [11.1](numbers177.json) to [30.6](numbers177.json)% |
| Yes | [51](numbers177.json) | [77](numbers177.json) | [77](numbers177.json)/[128](numbers177.json) ([60.2%](numbers177.json)); [95%](numbers177.json) CI [51.6](numbers177.json) to [68.8](numbers177.json)% |

### Unedited asserted vs p50c correctness on `C`

| Unedited state | p50c wrong | p50c correct | p50c correct rate in row |
| --- | ---: | ---: | ---: |
| No | [95](numbers177.json) | [25](numbers177.json) | [25](numbers177.json)/[120](numbers177.json) ([20.8%](numbers177.json)); [95%](numbers177.json) CI [14.2](numbers177.json) to [28.3](numbers177.json)% |
| Yes | [11](numbers177.json) | [56](numbers177.json) | [56](numbers177.json)/[67](numbers177.json) ([83.6%](numbers177.json)); [95%](numbers177.json) CI [74.6](numbers177.json) to [92.5](numbers177.json)% |

No cell is called new computation. These paired observations use sampled outputs and a string rule; they cannot show whether the edit changed what the model computes or whether it can stop.

## What happened on the same problems outside the cap cohort?

Across the same [500](numbers177.json) problem IDs, `correct` by sample and any-of-[4](numbers177.json) was:

| Sample | Unedited correct | p50c correct |
| --- | ---: | ---: |
| [0](numbers177.json) | [35](numbers177.json)/[500](numbers177.json) ([7.0%](numbers177.json)) | [241](numbers177.json)/[500](numbers177.json) ([48.2%](numbers177.json)) |
| [1](numbers177.json) | [37](numbers177.json)/[500](numbers177.json) ([7.4%](numbers177.json)) | [242](numbers177.json)/[500](numbers177.json) ([48.4%](numbers177.json)) |
| [2](numbers177.json) | [46](numbers177.json)/[500](numbers177.json) ([9.2%](numbers177.json)) | [239](numbers177.json)/[500](numbers177.json) ([47.8%](numbers177.json)) |
| [3](numbers177.json) | [40](numbers177.json)/[500](numbers177.json) ([8.0%](numbers177.json)) | [240](numbers177.json)/[500](numbers177.json) ([48.0%](numbers177.json)) |
| Any of [4](numbers177.json) | [77](numbers177.json)/[500](numbers177.json) ([15.4%](numbers177.json)) | [320](numbers177.json)/[500](numbers177.json) ([64.0%](numbers177.json)) |

On target `T`, p50c had a literal hit on [95](numbers177.json)/[132](numbers177.json) ([72.0%](numbers177.json)) traces where thinking closed. This is shown beside the capped-trace rates, not pooled with them.

Every rate in this write-up that conditions on `C` or a capped failure is among traces that hit the cap. The first [200](numbers177.json) rows are not a random sample of [MATH-500](../../PART177-DESIGN.md).

## What can this read establish, and what are its limits?

The post's own line is quoted and linked above. Prior-art notes are from Muse's search report, not read by Claude; the papers and detector references are left unlinked here. This result tests the post on one narrow reading: a value stated in a cut trace. That does not prove the model would have stopped.

Amendment [1](../../PART177-DESIGN.md) was made after scores were read. It corrects the W1 late-gap denominator and verdict rule, and reports both readings of early hits. The only code change after registration was the import line in [read177.py](../../part177/read177.py) that `python -I` needed; the commit is in git.

The withdrawal rule did not fire because [177.1](prediction_table.json) held. It would have fired if that prediction failed: the reading “it had stated the answer and kept going” would have been withdrawn and the score blocked.

Limits from the design: [1](../../PART177-DESIGN.md) model, [1](../../PART177-DESIGN.md) benchmark, [1](../../PART177-DESIGN.md) edit, and the authors' stored samples on their stack; only the first [200](numbers177.json) rows, not a random sample; `capped` inferred by retokenising decoded text; a literal string matcher misses mathematically equal forms; a reader judges a window, not the whole trace, so `asserted` is audit-grade and not proof; shuffled answers are a label permutation, not chance reasoning; the line rule is not loop ground truth, and onset depends on how long text ran; nothing here shows how the edit works, and nothing was run on our hardware. Residual-failure rates say nothing about incidence.

## Registered predictions

The text, confidence, estimate, and verdict below come from the registered table. The first cells are plain prediction IDs.

| # | Prediction (verbatim, with confidence) | Estimate | Verdict |
| --- | --- | --- | --- |
| 177.1 | On T, the share of traces with an asserted real gold exceeds the same share for shuffled draw [1](../../PART177-DESIGN.md) by at least [15](../../PART177-DESIGN.md) points (medium) | [67](numbers177.json)/[173](numbers177.json) ([38.7%](numbers177.json)) real; [0](numbers177.json)/[173](numbers177.json) ([0.0%](numbers177.json)) shuffled; gap [38.7](numbers177.json) percentage points, [95%](numbers177.json) interval [31.8](numbers177.json) to [45.7](numbers177.json) percentage points | [held](prediction_table.json) |
| 177.2 | On T, at least [25](../../PART177-DESIGN.md)% of traces have an asserted real gold (medium) | [67](numbers177.json)/[173](numbers177.json) ([38.7%](numbers177.json)); [95%](numbers177.json) interval [31.8](numbers177.json) to [46.2](numbers177.json)% | [held](prediction_table.json) |
| 177.3 | Among T traces with an asserted real gold and a detected onset, at least half have the first asserted hit before the onset; not read below [20](../../PART177-DESIGN.md) traces (medium) | [49](numbers177.json)/[66](numbers177.json) ([74.2%](numbers177.json)) before onset; [95%](numbers177.json) interval [63.6](numbers177.json) to [84.8](numbers177.json)% | [held](prediction_table.json) |
| 177.4 | At least [80](../../PART177-DESIGN.md)% of p50c traces that are capped with no </think> carry a detected onset (medium) | [35](numbers177.json)/[48](numbers177.json) ([72.9%](numbers177.json)); [95%](numbers177.json) interval [60.4](numbers177.json) to [85.4](numbers177.json)% | [failed](prediction_table.json) |
| 177.5 | On problems in the same cohort in both arms with a detected onset in both, median onset as a share of text is later in p50c; not read below [15](../../PART177-DESIGN.md) pairs (medium) | [34](numbers177.json) pairs; median onset fraction [0.1624](numbers177.json) to [0.3179](numbers177.json); difference [0.1556](numbers177.json), [95%](numbers177.json) interval [-0.0279](numbers177.json) to [0.4001](numbers177.json) | [held](prediction_table.json) |
| 177.6 | Over the [200](../../PART177-DESIGN.md) paired problems, with onset searched only in the first [50,000](../../PART177-DESIGN.md) characters of each text and pairs with a shorter text left out, the rate of a detected onset is at least [15](../../PART177-DESIGN.md) points lower in p50c (medium) | unedited [111](numbers177.json)/[116](numbers177.json) ([95.7%](numbers177.json)); p50c [79](numbers177.json)/[116](numbers177.json) ([68.1%](numbers177.json)); difference [27.6](numbers177.json) percentage points, [95%](numbers177.json) interval [19.0](numbers177.json) to [37.1](numbers177.json) percentage points | [held](prediction_table.json) |
| 177.W1 | The real-against-shuffled literal-hit gap on C exceeds [15](../../PART177-DESIGN.md) points, but at least [20](../../PART177-DESIGN.md)% of real hits sit in the first [15](../../PART177-DESIGN.md)% of text, and excluding them the gap falls under [15](../../PART177-DESIGN.md); falsifier: fewer than [10](../../PART177-DESIGN.md)% of hits are that early (medium) | C real [115](numbers177.json)/[187](numbers177.json) ([61.5%](numbers177.json)); shuffled [37](numbers177.json)/[187](numbers177.json) ([19.8%](numbers177.json)); gap [41.7](numbers177.json) percentage points, [95%](numbers177.json) interval [34.2](numbers177.json) to [49.2](numbers177.json) percentage points; all-hit early share [3923](numbers177.json)/[37453](numbers177.json) ([10.5%](numbers177.json)); [95%](numbers177.json) interval [8.1](numbers177.json) to [13.7](numbers177.json)%; first-hit early [109](numbers177.json)/[115](numbers177.json) ([94.8%](numbers177.json)); late all-C gap [31.6](numbers177.json) percentage points, [95%](numbers177.json) interval [24.6](numbers177.json) to [38.5](numbers177.json) percentage points | [inconclusive](prediction_table.json) |
| 177.W2 | Shuffled windows clear [40](../../PART177-DESIGN.md)% yes on Gate [4](../../PART177-DESIGN.md); falsifier: under [20](../../PART177-DESIGN.md)% (medium) | [0](numbers177.json)/[98](numbers177.json) ([0.0%](numbers177.json)) shuffled windows asserted; [95%](numbers177.json) interval [0.0](numbers177.json) to [0.0](numbers177.json)% | [failed](prediction_table.json) |
| 177.W3 | Among C traces with a literal hit, at least a third have a pred_first that does not match the gold; falsifier: under [15](../../PART177-DESIGN.md)% (medium) | reading (a) [0](numbers177.json)/[115](numbers177.json) ([0.0%](numbers177.json)); [95%](numbers177.json) interval [0.0](numbers177.json) to [0.0](numbers177.json)%; reading (b) [113](numbers177.json)/[115](numbers177.json) ([98.3%](numbers177.json)); [95%](numbers177.json) interval [95.7](numbers177.json) to [100.0](numbers177.json)% | [failed](prediction_table.json) |

Cold check: pending.
Rebuild with: `python` [numbers177.py](numbers177.py) and `python` [plot177.py](plot177.py).
