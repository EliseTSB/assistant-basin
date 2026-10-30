# Part 120: does a push interact with itself as much as with a second push?

Procedure: v1
Registered 2026-10-30 09:45 CT by Claude (Opus 5.5), before any data. Elise's direction for the night stands (Discord message
1565603681804488839): keep refining slot against concept neurons and "try to figure out how different types of neurons
interact." The commit that adds this file is the registration.

## Question

Part 119 found that two pushes do not add. Parts 119b and 119c found that the part that does not add grows with dose and
is about the same size when the second push is a random direction. Nobody has asked whether one push adds with itself. A
push at 2α is the same push twice. If it is not twice the push at α, then the failure to add is not about two neurons
meeting. This part measures a push's bend against itself and against a second push, on the same positions, at doses low
enough for a second-order reading.

## Prior art and existing data

- Closest published work: Azizi et al., "The Sharp Directions Are Against You: Curvature Analysis of Activation
  Steering", NeurIPS 2026 (abstract page read 2026-10-30). They define a steering Hessian of a behavioural loss over one
  steering vector and find sharp directions that make up 1 to 5% of the vector's squared norm. The page does not address
  two separate vectors. ORBIT (arXiv 2606.22357) and StiCAS (NeurIPS 2026) report interference between concept vectors
  and fix it by geometry; both seen as search snippets only. Two searches on 2026-10-30 found no comparison of a
  push's self term with its cross term, and none on single neurons.
- Our own data: results/part119c/interaction-float32.jsonl. On real rows the median of |nA(2α) minus 2 nA(α)|, a lower
  bound on the size of a push's self term, is 0.46 at 0.1 against a median nI of 0.25, and 1.90 at 0.2 against 1.92. That
  file stores norms only, so it cannot give the self term itself. This bound is why 120.3 is written as it is.

## Method

Model, revision, device, chat frame, question, units, pushes and Path 2 are Part 119c's float32 run: OLMo 2 1B Instruct at
revision 48d788eca847d4d7548f375ad03d3c9312f6139e, float32, A770, batch 8, "Who are you?", the 312 rows drawn with seed 11906
by part119c/rows119c.py, on the first 24 tokens of each first unit's 0.8 answer. T is not rebuilt: each row and position
uses the t_ids stored in results/part119c/interaction-float32.jsonl. Doses are 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4 and 0.8.
Pushes are built from the float32 weights with Part 92's functions at each dose. No answers are generated.

At each row, position and dose the run stores three centered logit changes on T, computed as in part119b/run119b.py
`_position_record`: A(α) for the first push, B(α) for the second, P(α) for both. The scorer then works on the CPU. For each
α from 0.0125 to 0.4:

- I(α) = P(α) minus A(α) minus B(α), the interaction of Parts 119 to 119c.
- S_A(α) = A(2α) minus 2A(α), the first push's interaction with itself. S_B(α) is the same for the second push.
- S_P(α) = P(2α) minus 2P(α), the same for the two pushes taken as one direction.
- E(α) = S_P minus S_A minus S_B minus 2I.

If the logits are expanded in the push with a and b the push directions, then I = α²H(a,b), S_A = α²H(a,a), S_B = α²H(b,b)
at second order, and E is third order. So E says how far the second-order reading can be trusted at each dose.

Statistics, each the median over positions within a row, then the median over rows, by row class:

- c(α) = ‖I‖ / sqrt(‖S_A‖ ‖S_B‖), the cross-to-self ratio.
- q(α) = ‖E‖ / ‖S_P‖.
- Self share s(α) = (‖S_A‖ / ‖A(2α)‖)². This is Part 119's interaction share with the same push as both members.
- Pair share (‖I‖ / ‖P‖)², as in Part 119c.

Positions count when ‖P(α)‖, ‖S_A‖ and ‖S_B‖ are above 0. For each of the 39 first units, u(α) is the median of s(α) over
that unit's eight rows. The unit bootstrap is Part 119c's: 39 units with replacement, 2,000 draws, seed 12001, product
weighting primary, sum weighting reported beside it.

Reported as exploratory, with no prediction: the cosine of I with S_A and with S_B at each dose, and c on same-layer and
different-layer rows.

## Controls

- Matched random controls: the 156 control rows of Part 119c, whose second push is a random direction matched in layer and
  size. Every statistic is computed on them too.
- Do-nothing hook: gate 1. Part 119c listed this as not repeated in float32.
- Dose ladder: 0.0125 to 0.8. The two lowest doses measure float32's floor. Under a square law ‖I‖ and ‖S_A‖ fall by 4 per
  halving, so a shortfall at the low end is the floor.
- Not used: both directions. A central difference would remove the third-order terms at twice the passes. E and q measure
  how large those terms are, so a forward difference is enough for this question. A central-difference part can follow if q
  is large at 0.05.
- Not used: off-target questions, a reader, answers, broken counts and held-out prompts. This part reads logits on fixed
  paths and selects nothing.

Gate, before any prediction is read: (1) in float32 a zero-dose push through the same hook path gives logits identical to
the plain pass, with `torch.equal`, on all 39 paths; (2) at 0.05, 0.1, 0.2, 0.4 and 0.8 the run's ‖A‖, ‖B‖, ‖P‖ and ‖I‖
reproduce Part 119c's stored float32 nA, n2, nPair and nI, with a median relative difference under 0.5% for each statistic at
each dose; (3) the push at 2α equals twice the push at α, with a relative error under 1e-5 for every unit and each adjacent
pair of doses. A miss on any gate stops every reading, and the tables stay visible under a heading saying nothing is read
from them. Gate 2 could fail if the device or the hook path changed since Part 119c; gate 3 could fail if the push is
rebuilt in a way that does not scale.

## Size

- Smallest effect worth knowing: c differing from 1 by a factor of 1.25, the band Part 119c used. It is a ratio inside a
  row, so 39 units and 312 rows are expected to do; if the interval spans a boundary the rule reports the numbers and reads
  nothing.
- Total: 8 passes for each of 39 first units and 14 for each of 312 rows, 4,680 forward passes, about 4.5 minutes at Part
  119c's float32 rate, plus the gates and one model load. No answers, no reader.
- Pilot: not needed: under 1,000 generations and one GPU-hour, on rows and code Part 119c ran. A smoke on one first unit
  at two doses comes first, and gates 1 to 3 stop any reading.

## Optimization

- Bound resource: the A770.
- Measured: Part 119c's float32 run did 3,354 passes in 187 s at batch 8, 18 passes/s, on these rows.
- Levers: reuse Part 119c's rows and T (taken); float32 only (taken: Part 119c located bf16's floor, and it sits below 0.2,
  where this part reads); batch 8 (kept: batch 1 and 8 were timed in Part 119b, and a 5-minute job does not need another
  search); caching the first push per unit (taken, as in run_path2); all 2,964 rows (declined: 40 minutes for an interval
  this question does not need); copy engine off and direct load to the device (taken, from Part 119c).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 120.1 | In float32 on real rows, the median row ‖I‖ follows a square law at low dose: log2 of its ratio between adjacent doses is within 2 plus or minus 0.3 from 0.025 to 0.05 and from 0.05 to 0.1. | medium | Claude |
| 120.2 | The second-order identity holds at low dose and has broken by 0.2: on real rows the median row q is at most 0.30 at 0.05 and at least 0.50 at 0.2. | medium | Claude |
| 120.3 | A push interacts with itself more than with a second push: at 0.1 the median row c over real rows is at most 0.75, and its product-weighted 95% unit-bootstrap interval lies below 1. | medium | Claude |
| 120.4 | At each of 0.05, 0.1 and 0.2, the real median c is within a factor of 1.25 of the control median c. | medium | Claude |
| 120.5 | At 0.4 on real rows, the median self share is at least half the median pair share and at most twice it: a push at 0.8 is about as far from twice itself as a pair at 0.4 is from the sum of its parts. | low | Claude |
| 120.6 | At 0.4, over the 39 first units, u is higher for slot units than for concept units: the slot median is above the concept median, with a one-sided Mann-Whitney p below 0.05 on Part 92's Haiku kinds, and the same direction on the units both readers agree on. | low | Claude |
| 120.7 | Across the 39 units, the Spearman correlation of u(0.4) with Part 119's dominance is at most -0.3. | low | Claude |
| 120.8 | At 0.1, a real unit's self term is the size of a random direction's on the same path: taking for each first unit the median row ‖S_B‖ on its real rows over the same on its control rows, the median over units lies between 0.8 and 1.25. | low | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 120 (registered 2026-10-30)".

## Reading rules

part120/score120.py computes every number. Gates first. If 120.2's first clause fails, so that q at 0.05 is above 0.30, the
second-order reading is withdrawn: S and I are then only finite differences, and the headline rule below returns nothing.
Otherwise, on the product-weighted interval of c at 0.1 over real rows: (1) wholly below 0.8: "At low dose a push interacts
with itself more than with a second push"; (2) wholly inside 0.8 to 1.25: "At low dose a push interacts with a second push
about as much as with itself"; (3) wholly above 1.25: "At low dose a push interacts with a second push more than with
itself"; (4) otherwise the numbers are reported and nothing is read. Whether that sentence covers random second pushes
depends on 120.4 and is said beside it. Nothing here changes a verdict of Parts 119, 119b or 119c.

## Limits

One model, one question, 312 of 2,964 rows, fixed paths chosen at 0.8 from the first unit's own answer, so A acts on its own
path and B on a path that is not its own. T was chosen at 0.8 and covers about 24 tokens, not the vocabulary. These are
finite differences, and float32 still rounds. The part says how large the self and cross terms are on the output, not where
in the model they arise. Nothing here measures answers generated at these doses.
