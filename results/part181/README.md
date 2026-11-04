# Part 181: OLMo 2 1B's neurons move toward their final top documents all through pretraining; mid-training and SFT each change about 4 in 10, DPO about 1 in 10, and the last RL round none

Asked: across OLMo 2 1B's training, which documents does each MLP neuron fire on most, and when does that list stop
changing? Every neuron in all 16 layers (131,072 units) was run over the same 2,500 Pile documents at 15 checkpoints:
six pretraining checkpoints (21B to 4T tokens), three mid-training checkpoints (13B, 26B and 51B tokens of Dolmino), SFT,
DPO and Instruct (DPO followed by RLVR), plus two repeat runs and a rounding floor. The measure is the overlap of a
unit's top-10 documents between two checkpoints (0 = none shared, 1 = the same ten).

| Stage step | Median neuron overlap | Median dense-direction overlap |
| --- | ---: | ---: |
| pretrain 21B to 210B | 0.0 | 0.0 |
| pretrain 210B to 1T | 0.1 | 0.0 |
| pretrain 1T to 2T | 0.2 | 0.1 |
| pretrain 2T to 3T | 0.4 | 0.2 |
| pretrain 3T to 4T | 0.5 | 0.3 |
| pretrain 4T to mid-training 13B | 0.6 | 0.5 |
| mid-training 13B to 26B | 0.7 | 0.7 |
| mid-training 26B to 51B | 0.8 | 0.7 |
| mid-training 51B to SFT | 0.6 | 0.5 |
| SFT to DPO | 0.9 | 0.9 |
| DPO to Instruct (RLVR) | 1.0 | 1.0 |
| Floor: 4T against 4T rounded through bf16 | 1.0 | 1.0 |

The mean overlap with Instruct rises at every pretraining checkpoint (0.03, 0.08, 0.19, 0.28, 0.36, 0.47), but at 4T
about half of a typical neuron's top ten are still not the ten it ends with. Mid-training and SFT each replace about four of
the ten; DPO replaces about one; the final RL round replaces none. A weight change too small to mean anything (the 4T
weights rounded through bf16) moves nothing at the median, so the movement in the table is not the measure's own noise.

## Predictions

Registered in `PART181-DESIGN.md` before any collected value was read. All on the primary cohort (131,071 of the
131,072 units), over the layers that pass Gate 3 (16 of 16).

| # | Prediction (as registered, with its confidence) | Result | Verdict |
| --- | --- | --- | --- |
| 181.1 | The median neuron overlap of `midtrain_51B` against `instruct` is at least 0.6. (0.7) | median 0.6 (6 of 10 documents), mean 0.606 | held, exactly at the bar |
| 181.2 | The median neuron overlap of `dpo` against `instruct` is at least 0.9; not assessable if the floor's own median is below 0.9. (0.75) | median 1.0, mean 0.958; floor median 1.0 | held |
| 181.3 | The median neuron overlap of `pretrain_21B` against `instruct` is at most 0.1. (0.7) | median 0.0, mean 0.031 | held |
| 181.4 | The mean neuron overlap with `instruct` rises strictly along 21B, 210B, 1T, 2T, 3T, 4T. (0.6) | 0.031, 0.083, 0.186, 0.276, 0.364, 0.474 (medians 0, 0, 0.1, 0.2, 0.3, 0.5) | held |
| 181.5 | For (1T, 2T), (4T, midtrain_51B) and (midtrain_51B, instruct), the mean neuron overlap exceeds both the mean dense-control and the mean k = 2 sparse-control overlap in at least three quarters of the layers. (0.6) | 3, 5 and 7 of 16 layers | failed |
| 181.6 | The median `sft`-against-`dpo` overlap of Part 92's 21 slot neurons is within 0.15 of the median of their layers' comparators. (0.55) | slot median 1.0, comparator median 1.0 | held |
| 181.W1 (Weaver) | Run-to-run noise is larger early: the 1T repeat pair's floor is at least 2 times the Instruct pair's. Failed if both floors are 0. (medium) | both floors 0 | failed |
| 181.W2 (Weaver) | A relative activity rule marks at least 10% more neurons active than an absolute cut at one or more pretraining checkpoints. (medium) | largest difference 0.0 (every pretraining difference between -0.0004 and 0) | failed |

The reading rule's headline ("OLMo 2 1B's neurons settle on their top documents during pretraining and keep them
through post-training") needs Gates 1 to 3, 181.1, 181.3 and 181.4, in the same direction on the all-units cohort. All
of those are met, so the headline is licensed as registered. Two things qualify it: 181.1 held at its exact bar, and
the table shows the settling is gradual and incomplete at 4T rather than finished.

## Are neurons more stable than other directions? (181.5)

No. Summing more coordinates gave more stable top documents, and a single neuron was the least stable of the sparse
directions:

| Pair | k = 1 (neuron) | k = 2 | k = 4 | k = 16 | k = 64 | dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1T, 2T | 0.268 | 0.300 | 0.324 | 0.381 | 0.397 | 0.138 |
| 4T, midtrain_51B | 0.579 | 0.585 | 0.605 | 0.637 | 0.639 | 0.475 |
| midtrain_51B, instruct | 0.606 | 0.615 | 0.624 | 0.630 | 0.627 | 0.503 |

(Mean top-10 overlap. k = sum of k random coordinates of the layer; dense = a random direction over all of them.) The
neuron is the bottom end of a smooth curve in k, not a point above it. Dense random directions are the least stable,
far below k = 64; this part does not test why. The registered note holds: this failure does not make the units less
neuron-like, and it says nothing about whether units are features.

## Gates

- Gate 1 (each run's stored entries recompute, membership under another batch shape, an audit, and random documents
  with padded rows): passed on all 15 runs.
- Gate 2 (identity): every run's token-list and dense-control hashes equal the registered values and every commit
  matches its prefix. Weight fingerprints differ between every two runs except the two repeat pairs, which match, and
  the rounded 4T, which differs from 4T. Passed.
- Gate 3 (repeat runs): passed in 16 of 16 layers. Both repeat pairs give median and 10th-percentile overlap 1.0 and
  activity-mask agreement 1.0 in every layer. So this gate shows that the collector is repeatable, not how noisy a
  checkpoint is: `pretrain_1T` and `pretrain_1T_b` are byte-identical; `instruct` and `instruct_b` differ in 17 of
  131,072 maxima (at most 8e-5 relative) and in 15 of 2,621,440 top-document entries, all reorderings that change no
  unit's top-10 set. The rounding floor is the noise reference that means something.
- Position 0: every verdict is the same with the first position of each document left out.

## What went wrong and was fixed

- Ten of the 15 cloud runs first finished without committing their collected state (`.gitignore` excludes `*.npz`).
  They were collected again, unchanged, before any value was read (Amendment 2).
- `weaver181.py` keyed runs through a helper that renames `instruct_b` to `instruct`, so the Instruct repeat pair
  collapsed and 181.W1 printed "not read". Found after reading its first output; fixed with a test that fails on the
  old line, and rerun (Amendment 4). The first output is kept beside the rerun. The registered rule gives "failed" from
  the analysis's own Gate 3 either way.
- The analysis ran its 105 pair comparisons in a pool of six processes (`part181/parallel_analyze181.py`, a runner
  only), 18 minutes against about 88 in sequence. Before writing anything it computes one pair (1T, 2T) again in
  sequence and requires the two to be identical as JSON; they were.

## Limits

- One model (OLMo 2 1B); 2,500 documents, the first 2,500 of `NeelNanda/pile-10k` (itself the first 10,000 documents of
  the Pile, not a random sample); the positive pole only (each unit's largest activation); overlap on the top 10 of the
  20 kept. The 2,500 are mostly Pile-CC (624), PubMed Abstracts (370), OpenWebText2 (360), StackExchange (358),
  Github (216) and Wikipedia (193); the full breakdown is `pile_set_name_counts` in `analysis/verdicts.json.gz`.
- The top-10 lists are not saturated at 2,500 documents: Instruct's lists at 500, 1,000, 1,500, 2,000 and 2,500
  documents overlap about as much as nested uniform sampling predicts (0.5, 0.7, 0.8, 0.8 against 0.5, 0.667, 0.75,
  0.8). Every run reads the same documents, so the comparisons between checkpoints are fair, but a larger corpus would
  give different absolute overlaps.
- The activity rule is not binding here: 131,071 of 131,072 units count as active in Instruct, and both of Weaver's
  rules mark nearly every unit active at every checkpoint. The primary cohort is effectively every unit.
- Mid-training reads a different data mix (Dolmino is math- and STEM-heavy), so its movement is not attributed to the
  representation rather than the data. All runs read the same documents, so movement means the representation of those
  documents changed, whatever drove it.
- Same index, not same feature: a unit's index can hold a rotated job across checkpoints. This part measures which
  documents an index fires on.
- No step-0 checkpoint.

## Files

- `analysis/tables.md`: gates, verdicts, all 105 pairwise overlaps, Gate 1 per run, Gate 3 per layer, saturation.
- `analysis/verdicts.json.gz`, `analysis/overlaps.json.gz`: the full outputs of `part181/analyze181.py`, gzipped.
- `analysis/weaver181.json` (after Amendment 4) and `analysis/weaver181-before-amendment4.json`.
- `analysis/run.log`: the pooled run, with the equality check.
- `runs/<key>/`: each run's identity, manifest, Gate 1, summary and logs. The collected `state.npz` files are on the
  branches `cloud/part181-<key>` (instruct, instruct_b, midtrain_51B, pretrain_21B, pretrain_4T_bf16r) and
  `cloud/part181-<key>-r2` (the other ten); each matches the SHA-256 in its `summary.json`.
- Code: `part181/olmo_atlas_collect.py`, `part181/session181.py`, `part181/analyze181.py`, `part181/weaver181.py`,
  `part181/parallel_analyze181.py`. Design and amendments: `PART181-DESIGN.md`.

Cold check: pending.
