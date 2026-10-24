# Part 65 design: does what a neuron reads match what it writes? (Qwen3-1.7B)

Registered 2026-10-24, before any Part 65 collection, generation or call. Elise asked for a joint study after reading
Lyra's neuron-explorer, which labels neurons by what they write when pushed; our atlas labels them by what they read.

## Question

For the same MLP units, does the label we get from the text a unit fires on (its read label) describe what the model
writes when the unit is pushed (its write label)? How does agreement change with depth, and do random directions of
matched size agree as often as real neurons?

## Prior art (recorded before the run)

- Gur-Arieh, Mayan, Agassy, Geiger, Geva, "Enhancing Automated Interpretability with Output-Centric Feature
  Descriptions" (arXiv 2501.08319, 2025): input-based and output-based descriptions capture different aspects of a
  feature, and combining them helps. They compare descriptions on two evaluations; they do not measure per-unit
  read-write agreement against random directions, or across depth.
- Lyra's neuron-explorer (github.com/lyramakesmusic/neuron-explorer, from 2026-10-18; archived in our notes): steering
  every neuron of Qwen3.5-4B layers 22 and 8 with 256 random-direction controls, plus a 384-neuron sample of all 32
  layers. Their depth survey: topics early, discourse words in the middle, word families late, with a mid-layer dip
  that random controls nearly match. Our write-side measures follow their main_rel design so the two can be compared.
- Goodfire, "Neural Networks Are Already Clean" (2026): steering and "blind maximum" as checks on neuron labels.
- Muse was asked for a wider search on 2026-10-24 (council topic muse-prior-art); anything closer is added as an
  amendment before the full run.

## Units

The 500 Part 60 dev units, unchanged: 250 real Qwen3-1.7B MLP neurons and 250 random directions in the same layers
(atlas seed 5601, pool of 100 per layer). Read labels already exist for every unit: Part 64's four hypotheses (Haiku-1x
is the primary read label; the other three arms are secondary), and each unit's Part 64 sentence-test pass or fail.

## Write side (new, on the Intel Arc A770)

Write vector: for a neuron, its column of W_down; for a control r (unit norm in the MLP intermediate space), W_down r.
Both are scaled to unit length, then multiplied by dose x the median residual-stream norm leaving that layer
(measured on 2,000 tokens of the Part 60 corpus), and added to the MLP output at every position: Lyra's main_rel
convention, so a real push and a control push have the same size.

1. Rollouts: prompts {who: "Who are you?", snake: "Write a short poem about a snake.", orange: "Describe an orange."}
   x doses {-0.7, +0.5, +0.8}, chat template, greedy, 48 new tokens; one unsteered baseline per prompt.
2. Steer lens: doses +0.8 and -0.8 on 4 short neutral contexts; mean change in next-token log-probability over the
   vocabulary; top 12 tokens per sign.
3. Write words: tokens over-represented in a unit's 9 rollouts against the pooled control rollouts of the same layer
   band (Lyra's word-cluster mass, top 10 words).

Checks before the full run (pilot of 20 units, 10 real and 10 control): dose 0 reproduces the baseline token for token;
a +0.8 push changes at least half the rollouts; real and control pushes have equal norm at every layer; wall time and
Windows RAM recorded (job.py, copy engine off).

## Primary measure: blind matching

Claude Haiku 5.5 (Batch API, thinking disabled, temperature unset, from the API credit) sees one read label and two
write summaries (write words plus steer-lens tokens): the unit's own, and a foil from another unit of the same type and
layer band. It answers A or B. Two trials per unit with different foils and swapped order. Agreement = share correct;
chance is 50%. The judge never sees layer, unit id, type or which summary is the unit's.

Secondary, free: cosine between an embedding of the read label and of the write words, against a shuffled-pair null.

## Predictions

| # | Prediction |
|---|---|
| 65a | Real neurons: blind-matching agreement at least 60%, with the 95% unit-bootstrap interval above 50%. |
| 65b | Real neurons beat random directions by at least 8 points. |
| 65c | Among real neurons, the last third of layers agrees at least 10 points more than the first third. |
| 65d | Real neurons that passed their Part 64 Haiku-1x sentence test agree at least 10 points more than those that failed. |
| 65e | Random directions agree above 50% (lower bound of the interval above 50%): random mixtures are describable on both sides too. |
| 65f | The embedding measure ranks real above control in the same direction as the primary (no threshold). |

Decision rule: if 65a fails, our atlas labels describe inputs only and the front page says so; if 65a holds and 65b
fails, the read-write link is a property of the layer's geometry rather than of single neurons.

## Size and cost

500 units x 9 rollouts = 4,500 generations plus 2 x 4 steer-lens passes per unit: over the 1,000-generation line, so
the pilot runs first and its timing sets the full run. With 250 units per type, a difference of 8 points has a 95%
interval of about +/- 9 points: 65b is near the edge of what this size can show, and is reported with its interval.
Judge: 1,000 Haiku calls of about 600 tokens, under a dollar.

## Not changed

Part 60's test split stays closed. No unit, threshold or hypothesis from Parts 60 or 64 is altered.
