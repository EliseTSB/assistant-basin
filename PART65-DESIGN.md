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

## Amendment 1 (2026-10-24, before any Part 65 collection, generation or call)

Muse's prior-art sweep (council topic muse-prior-art, message 4) found two works closer than those listed above. Both
were checked at source (arXiv abstracts and full text) before this amendment.

- Arad, Mueller, Belinkov, "SAEs Are Good for Steering -- If You Select the Right Features" (arXiv 2505.20063, 2025):
  for residual-stream SAE features in Gemma-2, Llama-3.1 and Pythia, an input score and an output score are rarely both
  high. Early layers (0 to 50% of depth) have high input scores and near-zero output scores, middle layers are low on
  both, and high output scores appear in later layers. This is the closest prior work on the depth question. It uses
  token-overlap scores on SAE latents, not an agreement rate between LLM-written labels on native neurons, and its
  control is random features, not random directions.
- Chalnev, Siu, Conmy, "Improving Steering Vectors by Targeting Sparse Autoencoder Features" (arXiv 2411.02193, 2024):
  a feature's steering effect can differ from what it reads, and the largest effects can come from dense features with
  no readable input side. Our real and control write vectors are already unit-normalised and dosed by the same
  per-layer scale; they are not matched for activation density, which is a limit of this design.

Lyra's research log (archived 2026-10-24) frames the work as labelling by write instead of by read, and shows one
read-side example (N2742); it reports no per-neuron read-write agreement rate. Gur-Arieh et al.'s human relation classes
(41% "similar" on 100 Gemma Scope features) are a different measure from our two-choice rate and are not compared.

What Arad et al. predict here: agreement needs both sides describable. Early units read clearly and write weakly, late
units the reverse, and middle units are weak on both, so agreement should be lowest in the middle and need not rise from
first to last. 65c stays as registered. One competing prediction is added:

| # | Prediction |
|---|---|
| 65g | Among real neurons, the middle third agrees least of the three layer thirds (Arad et al.'s pattern). |

The depth predictions (65c, 65g) are exploratory. The thirds hold 83, 80 and 87 real neurons, so a difference between two
thirds has a 95% interval of roughly +/- 14 points; a 10-point difference is reported with its interval and cannot be
confirmed at this size.

Two clarifications from the build's first review (DeepSeek, round 1), fixed before any data exists:

- The steer lens lists, for each sign, the 12 tokens whose next-token log-probability rises most under that push. The
  judge's summary says so ("made more likely by a positive push", "made more likely by a negative push"); the build
  had labelled the second list "made less likely", which was wrong.
- A judge answer that is a lone A or B, with or without trailing punctuation, is read as that letter. Any other answer
  is recorded as unparseable and the trial is left out; a unit with no valid trial is left out and counted. If more than
  5% of the 1,000 trials are unparseable, the analysis stops and reports that instead of the predictions.

Nothing else in the design changes.

## Amendment 2 (2026-10-24, before any judge call; the GPU steering pilot was already running and is unaffected)

The build's second review (DeepSeek, round 2) found that Amendment 1 left the missing-trial rule incomplete: it said
what happens to a unit with no valid trial but not to a unit with one, and the stop's denominator was not tied to the
registered plan. By the stop rule on the card this is a redesign of the judge-and-analysis missing-data policy, stated
here before any data, not a further patch. It replaces the unparseable-answer paragraph of Amendment 1.

1. A trial is valid when its answer is a lone A or B (optional surrounding whitespace, one optional trailing `.` or
   `)`). A trial is missing when its answer is anything else or when it is absent from the answers file.
2. A unit enters the analysis only when both of its trials are valid. A unit with one or two missing trials is left out
   and counted; the count is reported by type and layer band.
3. If more than 50 of the registered 1,000 trials are missing (5%), the analysis stops and reports the count instead of
   the predictions. The denominator is always the registered 1,000.
4. Beside the primary result, the analysis reports two bounds that include every unit: each missing trial scored 0.5,
   and each missing trial scored wrong. They are reported, not used for pass or fail.
5. The analysis reads the answers file strictly: a malformed line anywhere stops it.
6. A judge chunk the provider rejects (released) is final for that output directory. Recovery is a fresh output
   directory for the whole judge, never a resubmission of a released attempt; the whole judge reserves under three
   cents. Answers from two output directories are never combined.

## Amendment 3 (2026-10-24, after the 20-unit steering pilot and before the full run or any judge call)

The pilot passed its registered checks (dose 0 reproduced the baseline for all 20 units; 60 of 60 +0.8 rollouts
changed; push norms equal within 2e-7 at every layer; 13.7 s per unit on the A770). Reading its text showed that the
chat template ran in Qwen3's thinking mode, so 162 of the 180 rollouts spent all 48 new tokens inside the `<think>`
preamble ("Okay, the user wants..."). Write words drawn from that preamble describe the same opening for every unit, and
would make 65a fail for a reason unrelated to the units. No judge answer exists, so no outcome was seen.

1. Rollouts and baselines use Qwen3's non-thinking mode (`enable_thinking=False`; the template inserts an empty think
   block), with everything else unchanged: same prompts, doses, greedy decoding and 48 new tokens.
2. The pilot is rerun under this rule with the same 20 units and must pass the same checks before the full run.
3. Lyra's research log found that relative doses do not transfer to the first layers of Qwen3.5-4B (at +0.8, 92% of
   layer-0 rollouts were empty or `<think>` loops). We report 65a and 65b again with layers 0 to 2 left out, as a
   sensitivity analysis that does not enter pass or fail.

## Amendment 4 (2026-10-24, before any judge call)

Three review rounds each found a defect in what happens after the provider rejects a judge chunk. By the stop rule on the
card the question went to a cold reader (Chorus Task, review saved in Journal `reviews/chorus-p65-judge-2026-10-24.md`).
This replaces rule 6 of Amendment 2.

1. A chunk the provider rejects with a validated 4xx (released) ends the registered judge. No chunk is retried and no
   new output directory is started for this plan. Its 100 trials are missing, which is above the 50-trial stop, so the
   study reports that the judge failed and gives no result for 65a to 65g. A later judge would need its own registered
   plan. A batch whose provider id is known can still be fetched and settled; nothing is sent again.
2. A judge run has one identity (a random run id and its resolved output directory), recorded in its manifest, in every
   ledger reservation and on every answer row. Before every send, the judge refuses if any earlier reservation for the
   same plan belongs to another run, and refuses to send a chunk that has any released, settled or retired attempt.
3. The local record that a chunk is about to be sent is written to disk durably (file and directory synced) before the
   send.
4. The analysis refuses answer rows from more than one run.
5. The cost limit is the paid-action sub-cap of $1. "Under three cents" in Amendment 2 was an estimate from synthetic
   summaries; the reserve total computed from the real requests is recorded before the one-request probe.
