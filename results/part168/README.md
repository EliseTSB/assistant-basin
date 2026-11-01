# Own-list bans cut loops; equal-mass random bans add them

Banning a push’s own top 50 tokens cut loop answers by [31%](score.json) (95% interval [16%–45%](score.json)) on fresh directions; a neighbour’s top 50 cut about nothing, while 50 random tokens of the same mass raised loops. This supports the masking policies under Amendment 1; it does not identify how loops form. [Design and amendments](../../PART168-DESIGN.md) [Score](score.json)

## Answer outcomes

| Arm | Ban | n | Loop rate | Broken | Joint success | Mean length | EOS share |
|---|---|---:|---:|---:|---:|---:|---:|
| n+ | none | [672](score.json) | [12.8%](score.json) | [12.9%](score.json) | [87.1%](score.json) | [56.84](score.json) | [90.6%](score.json) |
| n+ | own | [672](score.json) | [9.2%](score.json) | [9.7%](score.json) | [90.3%](score.json) | [57.35](score.json) | [89.6%](score.json) |
| n+ | matched | [672](score.json) | [13.1%](score.json) | [13.2%](score.json) | [86.8%](score.json) | [56.72](score.json) | [90.8%](score.json) |
| n+ | mass | [672](score.json) | [15.6%](score.json) | [15.6%](score.json) | [84.4%](score.json) | [56.39](score.json) | [89.9%](score.json) |
| k4 | none | [672](score.json) | [15.2%](score.json) | [15.6%](score.json) | [84.4%](score.json) | [56.00](score.json) | [90.9%](score.json) |
| k4 | own | [672](score.json) | [10.0%](score.json) | [10.0%](score.json) | [90.0%](score.json) | [56.99](score.json) | [91.7%](score.json) |
| k4 | matched | [672](score.json) | [15.5%](score.json) | [15.9%](score.json) | [84.1%](score.json) | [55.98](score.json) | [91.7%](score.json) |
| k4 | mass | [672](score.json) | [17.0%](score.json) | [17.3%](score.json) | [82.7%](score.json) | [55.79](score.json) | [91.2%](score.json) |
| full | none | [672](score.json) | [5.2%](score.json) | [5.4%](score.json) | [94.6%](score.json) | [59.03](score.json) | [87.8%](score.json) |
| full | own | [672](score.json) | [4.6%](score.json) | [4.8%](score.json) | [95.2%](score.json) | [58.82](score.json) | [88.8%](score.json) |
| full | matched | [672](score.json) | [5.5%](score.json) | [5.7%](score.json) | [94.3%](score.json) | [58.91](score.json) | [88.1%](score.json) |

k4 contains its slot’s n+ neuron, so pooled rows repeat a direction. [Score](score.json)

## Registered predictions

| Prediction | Result | Verdict |
|---|---|---|
| [168.1](../../PART168-DESIGN.md) own/shuffled content-word ratio | [215.6](score.json) | held |
| [168.2](../../PART168-DESIGN.md) n+ vs full broken-rate ratio | [2.42](score.json) | held |
| [168.3](../../PART168-DESIGN.md) own relative loop cut | [31.4%](score.json); interval [15.8%–44.9%](score.json) | held; wide interval |
| [168.4](../../PART168-DESIGN.md) own-minus-matched advantage | [0.047](score.json) vs [0.05](../../PART168-DESIGN.md); interval [0.022–0.073](score.json) | failed; missed its bar by a little |
| [168.5](../../PART168-DESIGN.md) matched relative cut | [−2.1%](score.json) | held |
| [168.6](../../PART168-DESIGN.md) full cut under half low-k cut | [11.4%](score.json) vs [31.4%](score.json) | held |
| [168.7](../../PART168-DESIGN.md) migrations at least one third and over baseline | [5.3%](score.json) vs [1.0%](score.json) baseline | failed |
| [168.8](../../PART168-DESIGN.md) paired own rate at least 0.05 below mass | [9.61%](score.json) vs [16.32%](score.json); gap interval [0.042–0.095](score.json) | held |
| [168.W1](../../PART168-DESIGN.md) Weaver’s mass-ban cut 15%–25% | [−16.5%](score.json) | failed, in the opposite direction |
| [168.W2](../../PART168-DESIGN.md) own-induced rate at least 0.02 above matched | [3.6%](score.json) vs [1.0%](score.json) | held |
| [168.W3](../../PART168-DESIGN.md) Weaver’s migration share 15%–30% | [5.3%](score.json) | failed |

## Transitions and induced loops

Among the [88](score.json) rows whose none loop held an own-list word, the fates were:

| Ban | No loop | Re-entry | Common only | Top 50 | Migration | Outside | Other |
|---|---:|---:|---:|---:|---:|---:|---:|
| Own | [65](score.json) | [4](score.json) | [2](score.json) | [1](score.json) | [1](score.json) | [15](score.json) | [0](score.json) |
| Matched | [0](score.json) | [88](score.json) | [0](score.json) | [0](score.json) | [0](score.json) | [0](score.json) | [0](score.json) |
| Mass | [17](score.json) | [63](score.json) | [2](score.json) | [1](score.json) | [0](score.json) | [5](score.json) | [0](score.json) |

Migration was rare ([1 of 19](score.json)); loops mostly did not move down the same bias, and [15 of 19](score.json) moved outside the top 200. Induced loop rates were own [3.6%](score.json), matched [1.0%](score.json), mass [6.5%](score.json). Mass exceeded none: relative cut [−16.5%](score.json), interval [−29.4% to −6.3%](score.json), opposite to a cut; Weaver’s [168.W1](../../PART168-DESIGN.md) failed. Under Amendment 1, a cut under [5%](../../PART168-DESIGN.md) would mean only that this policy had little loop-cutting effect, not that Weaver’s concern was wrong. [Score](score.json)

The mass list held [1.2](score.json) plain-answer tokens on average, own [0.07](score.json); this may explain added loops, a guess not tested. Across [1,344](extra168.txt) rows, mean top_on_own and top_on_mass were [1.41](extra168.txt) and [1.49](extra168.txt). Per-index own and mass means at positions [0–38](extra168.txt) are in index order. [Position log](positions.jsonl) [Computed summaries](extra168.txt)

## Paired comparison and hand read

The paired set has [1,342 of 1,344](score.json) low-k rows; [139](score.json) lists needed an outcome-blind one-token swap. All-row mass-rate bounds are [16.29%–16.44%](score.json), assigning unmatched rows no loops or all loops. Paired 168.8 rates are own [0.0961](score.json), mass [0.1632](score.json), with gap interval [0.0424–0.0946](score.json).

My judgement of [120](extra168.txt) truncated answers (readable/odd/broken): none [23/1/6](extra168.txt); own [29/0/1](extra168.txt); matched [26/1/3](extra168.txt); mass [20/4/6](extra168.txt). [Judged answers](hand_read_judged.jsonl) [Key](hand_read_key.jsonl)

## Gates and exploratory context

All [seven gates](score.json) passed. Launch one stopped at Gate 6 at [89.5%](../../PART168-DESIGN.md), before answers; Amendment 2 added a one-swap repair. Part 167’s failure prompted this part: [Part 167](../part167/README.md).

Part 167 is exploratory. n+ none broken/loop by phrasing: [26.0/26.0%, 26.0/25.0%, 26.0/26.0%, 17.7/16.7%, 16.7/16.7%, 25.0/22.9%, 18.8/18.8%, 22.9/21.9%](extra168.txt). Including phrasing 13: [22.4%/21.7%](extra168.txt); excluding it: [22.0%/21.6%](extra168.txt). Pooled n+ and k4 point like Part 168: own below none, matched level or above. Stopped part, one looped plain phrasing; descriptive only, not verdict. Part 164 plain floor: [1 of 16](extra168.txt) loops at 80. Part 168 n+ breakage [12.9%](score.json) vs Part 165 [25.7%](../part165/score.json).

![Loop rates by ban and fate shares for own-word loops](loops168.png)

## Limits and provenance

Limits: one model, dose, seven phrasings, greedy decoding, chosen top-50; bias/mass came from the plain path, which pushed answers leave. Words can remain available in pieces; the mass list may block useful tokens. [Design](../../PART168-DESIGN.md)

Provenance: [design](../../PART168-DESIGN.md), [launch](launch.jsonl), [runner](../../part168/run168.py), [scorer](../../part168/score168.py). [Checks](extra168.py) [Score](score.json)

Cold check: pending.
