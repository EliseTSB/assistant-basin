# Part 181: on 2,500 Pile documents, OLMo 2 1B's neurons keep changing which documents they fire on most all through pretraining; mid-training and SFT change them more than DPO, and RLVR changes them least

Asked: across OLMo 2 1B's training, which documents does each MLP neuron fire on most, and how much does that list
change from stage to stage? Every neuron in all 16 layers (131,072 units) was run over the same 2,500 Pile documents at
12 checkpoints: six in pretraining (21B to 4T tokens), three in mid-training (13B, 26B and 51B tokens of Dolmino), SFT,
DPO and Instruct (DPO followed by RLVR). Three more runs serve as controls: a repeat of 1T, a repeat of Instruct, and 4T
with its weights rounded through bf16. The measure is the overlap of a unit's top-10 documents between two runs (0 = none
shared, 1 = the same ten).

| Step | Mean overlap | Median | Units below 0.5 | Units keeping all 10 |
| --- | ---: | ---: | ---: | ---: |
| pretrain 21B to 210B | 0.085 | 0.0 | 97.6% | 0.0% |
| pretrain 210B to 1T | 0.134 | 0.1 | 93.1% | 0.0% |
| pretrain 1T to 2T | 0.268 | 0.2 | 78.2% | 0.1% |
| pretrain 2T to 3T | 0.372 | 0.4 | 63.7% | 0.1% |
| pretrain 3T to 4T | 0.464 | 0.5 | 47.9% | 0.3% |
| pretrain 4T to mid-training 13B | 0.602 | 0.6 | 20.3% | 1.0% |
| mid-training 13B to 26B | 0.721 | 0.7 | 4.5% | 3.2% |
| mid-training 26B to 51B | 0.771 | 0.8 | 1.6% | 5.7% |
| mid-training 51B to SFT | 0.623 | 0.6 | 18.3% | 1.6% |
| SFT to DPO | 0.886 | 0.9 | 0.2% | 34.3% |
| DPO to Instruct (RLVR) | 0.958 | 1.0 | 0.0% | 62.0% |
| Floor: 4T against 4T rounded through bf16 | 0.993 | 1.0 | 0.0% | 93.0% |

(All 131,072 units, computed from the stored states; the analysis's own medians, on the 131,071-unit primary cohort,
are the same in every row.)

What this shows, on this corpus: through pretraining, each step keeps more of a unit's top ten than the step before, but
at 3T to 4T the median unit still changes half of them. Across mid-training (4T to 51B) the median unit keeps 6 of 10,
and across SFT again 6 of 10. DPO changes about 1 in 10 on average; RLVR changes about 0.4 in 10 on average, while 62% of
units keep all ten. The rounding floor is a weight perturbation of about the size Part 107 measured for the DPO-to-
Instruct update, chosen to carry no meaning; it changes about 0.07 in 10 on average. So RLVR moves the lists about six
times as much as a random perturbation of its size, and every other step far more.

**What should carry to another corpus is the ordering.** Mid-training and SFT change the lists more than DPO, and DPO
more than RLVR. The absolute overlaps belong to this corpus: the top-10 lists are not saturated at 2,500 documents (see
Limits), so a larger corpus would give different numbers. Weaver made this point in review.

## Predictions

Registered in `PART181-DESIGN.md` (registration commit 80b1faa, 2026-11-03 19:14 CT; the file's header still reads
"Not registered", a leftover from the draft) before any collected value was read. All on the primary cohort, over the
layers that pass Gate 3 (16 of 16).

| # | Prediction (as registered, with its confidence) | Result | Verdict |
| --- | --- | --- | --- |
| 181.1 | The median neuron overlap of `midtrain_51B` against `instruct` is at least 0.6. (0.7) | median 0.6, mean 0.606; 21.6% of units below 0.5 | held, exactly at the bar |
| 181.2 | The median neuron overlap of `dpo` against `instruct` is at least 0.9; not assessable if the floor's own median is below 0.9. (0.75) | median 1.0, mean 0.958; floor median 1.0 | held |
| 181.3 | The median neuron overlap of `pretrain_21B` against `instruct` is at most 0.1. (0.7) | median 0.0, mean 0.031 | held |
| 181.4 | The mean neuron overlap with `instruct` rises strictly along 21B, 210B, 1T, 2T, 3T, 4T. (0.6) | 0.031, 0.083, 0.186, 0.276, 0.364, 0.474 (medians 0, 0, 0.1, 0.2, 0.3, 0.5) | held |
| 181.5 | For (1T, 2T), (4T, midtrain_51B) and (midtrain_51B, instruct), the mean neuron overlap exceeds both the mean dense-control and the mean k = 2 sparse-control overlap in at least three quarters of the layers. (0.6) | 3, 5 and 7 of 16 layers | failed |
| 181.6 | The median `sft`-against-`dpo` overlap of Part 92's 21 slot neurons is within 0.15 of the median of their layers' comparators. (0.55) | slot median 1.0, comparator median 1.0 | held, but see below |
| 181.W1 | Weaver: run-to-run noise is larger early: the 1T repeat pair's floor is at least 2 times the Instruct pair's. Failed if both floors are 0. (medium) | both floors 0 | failed, and uninformative (below) |
| 181.W2 | Weaver: a relative activity rule marks at least 10% more neurons active than an absolute cut at one or more pretraining checkpoints. (medium) | the relative rule marks slightly fewer at every pretraining checkpoint (differences -0.0004 to -0.00002) | failed |

The registered reading rule allows the headline "OLMo 2 1B's neurons settle on their top documents during pretraining
and keep them through post-training" when Gates 1 to 3, 181.1, 181.3 and 181.4 hold, in the same direction on the
all-units cohort. All of them hold. This write-up does not use that headline anyway, because two of its words go further
than the data. "Settle": at 3T to 4T the median unit still changes half its list, and the lists themselves have not
converged on this corpus. "Keep them through post-training": 181.1 held exactly at its bar, and 21.6% of units share
fewer than half their top ten between `midtrain_51B` and `instruct`.

181.6 holds but cannot say much. 15 of the 21 slot neurons (Part 92: units that, when pushed, fill the chat model's
"Who are you?" answer with a persona) and 15 of their 21 comparators sit at exactly 1.0, because SFT to DPO changes
little in those layers. A difference was hard to see there.

181.W1 is uninformative. Both repeat pairs give median overlap 1.0 in every layer, so both floors are 0, and the rule
reads that as failed. But the repeats measure whether the collector is deterministic, not how noisy a checkpoint is:
`pretrain_1T` and `pretrain_1T_b` are byte-identical, and `instruct` and `instruct_b` differ in 17 of 131,072 maxima (at
most 8e-5 relative) and in 15 of 2,621,440 top-document entries, reorderings that change no unit's top-10 set.

## Are neurons more stable than other directions? (181.5)

No. On these three pairs, sums of a few coordinates kept their top documents better than single neurons, and dense random
directions kept them worst:

| Pair | k = 1 (neuron) | k = 2 | k = 4 | k = 16 | k = 64 | dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1T, 2T | 0.268 | 0.300 | 0.324 | 0.381 | 0.397 | 0.138 |
| 4T, midtrain_51B | 0.579 | 0.585 | 0.605 | 0.637 | 0.639 | 0.475 |
| midtrain_51B, instruct | 0.606 | 0.615 | 0.624 | 0.630 | 0.627 | 0.503 |

(Mean top-10 overlap. k = a sum of k random coordinates of the layer, 64 such sums per k per layer; dense = 256 random
directions over all coordinates per layer.) In all three pairs the overlap rises from k = 1 to k = 16, then is flat or
slightly lower at k = 64; the neuron is the lowest of the sparse directions, not above them. This part does not test why
dense directions sit far below. The registered note on 181.5 stands: its failure says nothing about whether units are
features. One model and one corpus.

## Gates

- Gate 1 (each run's stored entries recompute alone, top-20 membership under another batch shape, an audit, and random
  documents with padded rows): passed on all 15 runs.
- Gate 2 (identity): every run's token-list and dense-control hashes equal the registered values, and every commit
  matches its prefix. Weight fingerprints differ between every two runs except the two repeat pairs, which match, and the
  rounded 4T, which differs from 4T. Passed.
- Gate 3 (repeat runs): passed in 16 of 16 layers, with median and 10th-percentile overlap 1.0 and activity-mask
  agreement 1.0 everywhere. As above, this shows the collector is repeatable; the rounding floor is the reference for how
  much a small weight change moves the lists.
- Position-0 sensitivity: with units whose top-10 entries are mostly at a document's first position left out, every
  verdict is the same. Position 0 holds 0.3% to 0.6% of neuron top-10 entries at the stages checked (21B, 1T, 4T,
  51B, SFT, Instruct).
- The null: the analysis computes a derangement null for every pair (each unit's list against another unit's). Its median
  is 0.0 for every one of the 105 pairs (`analysis/tables.md`). No verdict rule reads it, though the design says
  predictions are read against it; every observed median above 0 is above it.

## What went wrong and was fixed

- Ten of the 15 cloud runs first finished without committing their collected state (`.gitignore` excludes `*.npz`). They
  were collected again, unchanged, on `-r2` branches, before any activation value was read. Gate 1 verdicts, file sizes
  and hashes had been seen (Amendment 2).
- Amendment 1: three runs were restarted onto machines with another CPU model and stopped on the resume check. The check
  now leaves the host and Python's patch version out and logs every change. Two of the kept runs were collected on more
  than one machine: `pretrain_4T_bf16r` moved from a 2.80 GHz to a 2.10 GHz Xeon and back (03:17 and 04:31 UTC,
  2026-11-04), and `midtrain_51B` resumed under Python 3.13.18 after starting under 3.11.18. Both passed Gate 1, which
  recomputes their stored entries on the final machine. `instruct` was restarted from the beginning.
- `part181/weaver181.py` keyed runs through a helper that renames `instruct_b` to `instruct`, so the Instruct repeat pair
  collapsed. 181.W1 printed "not read", and 181.W2's absolute threshold came from `instruct_b`'s maxima. Found after
  reading the first output; fixed with a test that fails on the old line, and rerun (Amendment 4). W1's verdict moved to
  "failed" (the rule's reading of two zero floors); W2's numbers did not change. The first output is kept beside the
  rerun.
- The analysis ran its 105 pair comparisons in a pool of six processes (`part181/parallel_analyze181.py`, a runner only):
  1,128 s in all, against an estimated 88 minutes in sequence (50.4 s for one pair, measured, times 105). Before writing
  anything it computes one pair (1T, 2T) again in sequence and requires the two results to be identical as JSON; they
  were.

## Limits

- One model (OLMo 2 1B); 2,500 documents, the first 2,500 of NeelNanda/pile-10k on Hugging Face (itself the first 10,000 documents of
  the Pile, not a random sample), each cut at 512 tokens (992 of the 2,500 reach the cap); the positive pole only (each
  unit's largest activation); overlap on the top 10 of the 20 kept. The 2,500 are mostly Pile-CC (624), PubMed Abstracts
  (370), OpenWebText2 (360), StackExchange (358), Github (216) and Wikipedia (193); the full breakdown is
  `pile_set_name_counts` in `analysis/verdicts.json.gz`.
- The top-10 lists are not saturated. Instruct's lists at 500, 1,000, 1,500, 2,000 and 2,500 documents overlap 0.5, 0.7,
  0.8 and 0.8 step to step, about what nested random subsets of a growing corpus give (0.5, 0.667, 0.75, 0.8). So
  "a unit's top documents" means its top documents in this corpus, which a larger corpus would change. The
  comparisons between checkpoints are fair, because every run reads the same documents, but the absolute overlaps are
  this corpus's.
- The rounding floor is one perturbation at one checkpoint (4T).
- The activity rule is not binding here: 131,071 of 131,072 units count as active in Instruct, and both of Weaver's rules
  mark nearly every unit active at every checkpoint, so the primary cohort is effectively every unit. 9 units in
  Instruct and 13 at 4T fire positively on fewer than 10 documents.
- Mid-training reads a different data mix (Dolmino is math- and STEM-heavy), so its movement is not attributed to the
  representation rather than the data. All runs read the same documents, so movement means the representation of those
  documents changed, whatever drove it.
- Same index, not same feature: a unit's index can hold a rotated job across checkpoints. This part measures which
  documents an index fires on.
- No step-0 checkpoint.
- The design records a token hash from Fable's review (`83c4cc60…`, 930,289 tokens) that did not reproduce; the runs use
  the registered hash, and Gate 2 checks it.

## Files

- `analysis/tables.md`: gates, verdicts, all 105 pairwise medians with the null, Gate 1 per run, Gate 3 per layer,
  saturation.
- `analysis/verdicts.json.gz`, `analysis/overlaps.json.gz`: the full outputs of `part181/analyze181.py`, gzipped.
- `analysis/weaver181.json` (after Amendment 4) and `analysis/weaver181-before-amendment4.json`.
- `analysis/run.log`: the pooled run, with the equality check.
- `runs/<key>/`: each run's identity, manifest, Gate 1, summary, logs and host history. The collected `state.npz` files
  are on the branches `cloud/part181-<key>` (instruct, instruct_b, midtrain_51B, pretrain_21B, pretrain_4T_bf16r) and
  `cloud/part181-<key>-r2` (the other ten); each matches the SHA-256 in its `summary.json`.
- Code: `part181/olmo_atlas_collect.py`, `part181/session181.py`, `part181/analyze181.py`, `part181/weaver181.py`,
  `part181/parallel_analyze181.py`. Design and amendments: `PART181-DESIGN.md`.

Cold check: Claude Haiku 5.5 (no project context), 2026-11-04, 11 findings: the position-0 check described wrongly;
Amendment 1's host changes not named; the registered headline claimed as licensed while the title said something else;
medians read as shares, and steps added up; 181.W1 not called uninformative; "smooth curve" and "neuron-like" in 181.5;
the null not reported; saturation under-read; the document cap and Fable's token hash missing; smaller wording and count
errors; 181.6 unable to show much. All folded; the numbers it recomputed from the states matched.
