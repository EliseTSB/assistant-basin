# Cold review of Part 181 (draft v1, branch `draft/part181-review-2026-11-03`, commit adbceba)

Reviewer: Claude, a Claude Code cloud session, 2026-11-03, before registration. Read: `PART181-DESIGN.md`,
`part181/olmo_atlas_collect.py`, `part181/test_olmo_atlas_collect.py`, `part181/compare_runs.py`,
`part181/pilot_runs.sh`, `part181/TIMING.md`, `part85/checkpoints.py`, and the README diff on the branch. Ran: the
draft's four tests (4 passed, CPU virtualenv, torch 2.15.0+cpu, transformers 5.22.1, numpy 2.5.4) and
`review/part181/counterexamples181.py`, which drives the real collector on the tests' tiny random OLMo 2 and writes
`review/part181/counterexamples181.json`; every number below marked "(counterexample X)" is in that file. No GPU, no
reader, no model download except the OLMo 2 tokenizer files and the pile-10k parquet for section 1.2.

**Verdict: register after the fixes in section 6.** The collector is sound where I could test it (padding, the
per-document top list, resume). The gates are not: Gate 1 tests one number per unit and no padded row, Gate 2 never
fingerprints the weights, and Gate 3 is a determinism check that a 30% failure passes. The active-unit rule can move
the medians that 181.3 to 181.5 are read on, and the medians themselves take one of eleven values, so 181.4's strict
rise and 181.5's "exceeds" can fail or hold by granularity. Section 5 re-derives the Part 92 screen rates.

## 1. Can a gate pass with something wrong?

### 1.1 Gate 1: the 32-pair recompute checks the top-1 value only, on rows that are rarely padded

Gate 1 recomputes, for 32 seeded units, the activation of the top-1 document at the stored position by a lone forward
and compares it with the stored top-1 activation to 1e-4 relative. Three things it cannot see:

- **Ranks 2 to 20.** A state whose ranks 2 to 10 belong to another unit passes Gate 1 on 32 of 32 pairs, exactly as
  the true state does, while its top-10 overlap with the true state has a median of 0.1 (counterexample A: ranks 1
  to k-1 of every unit rotated to the next unit, the real collector on a 200-document toy corpus). The measure the
  part reads is the top-10 set; the gate checks one member of it.
- **Padded rows.** The 32 pairs are drawn over (unit, top-1 document). A document's maximum grows with its length, so
  top-1 documents are mostly at the 512-token cap: 1,082 of the first 2,500 documents hit the cap (section 1.2), and
  the batcher length-sorts, so a capped document sits in a batch of capped documents with no padding at all. A padding
  fault that touches only padded rows would never meet the gate. The collector's own test does exercise padding
  (`test_matches_brute_force`, batch 2, lengths 3 to 40) and passes, so I found no such fault; the gate just cannot
  stand in for that test on the real model.
- **A wrong document set.** If a chunk were skipped, every stored activation would still be a real activation of a
  real document, and the gate would pass. The cursor design (section 4) makes a skipped chunk unlikely, but the gate
  does not test for it.
- **Not implemented.** There is no Gate 1 code in `part181/`; `--forward-only` only disables bookkeeping. It should
  be a separate code path (a plain forward hook at the stored position), not the `Collector` class.

Smallest fix: keep the 32 top-1 pairs; add 32 (unit, random document) pairs and check that the recomputed per-document
maximum is at most the stored rank-20 activation when the document is absent from the list, and equal to the stored
entry when present; require at least 8 of the top-1 pairs to come from documents shorter than the longest document in
their batch; and assert, over all units, that no document id repeats within a top list (a one-line invariant that
catches double processing).

### 1.2 Gate 2: nothing in a run's record fingerprints the weights

Gate 2 logs the resolved commit (prefix-checked), the sha256 of the token-id lists (equal across runs) and the sha256
of the control matrix (equal across runs). Two runs of the same weights under different labels pass all three. In the
real collector, the recorded config of two such runs differs in one key, `model_arm`, the label itself; the only
model-related fields are `model_dir_name` (a directory name) and `model_config_sha256`, which hashes `config.json`,
identical across every checkpoint of one architecture (counterexample B). A mislabelled snapshot directory, or a
`--model-dir` pointing at the wrong checkpoint while the download log shows the right commit, gives an overlap of 1.0
between "dpo" and "instruct" and 181.2 holds for nothing.

Two of the three hashes are also only compared across runs, never against a registered value, so thirteen runs on
the wrong corpus revision or a changed numpy stream would agree with one another and pass. Both values can be
registered tonight, because neither needs a model:

- Control matrix, 16 layers stacked, float32, `generate_controls(256, 8192, 5601, layer)`:
  `54515ce29b1c36af123d8337932e510744b3ab77951e3b520f7ada43a10ff20a` (numpy 2.5.4; per-layer prefixes in
  `review/part181/counterexamples181.json` are not stored there, so they are: L0 `dae6e2a6aa360948`, L1 `57837e0f2857c1cb`,
  L2 `0345937a01308a79`, L3 `85cf87ab9383fb3d`, L4 `e39655213db7bc6b`, L5 `4d3b1b17a6350f81`, L6 `981350a1e9cc7544`,
  L7 `3148b163172fa1d2`, L8 `794380259e8d01c4`, L9 `5f301d77588a6a80`, L10 `f9844e3e53f299fd`, L11 `144db8a593f70299`,
  L12 `93354797be4b48cb`, L13 `b55eb8c55bb4cddf`, L14 `e80486a17642ad94`, L15 `a88590641f57ecf7`). NumPy's policy
  allows a Generator's distribution streams to change between versions, so a session on another numpy must match this
  value or the design must say which numpy it pins; the pilot ran numpy 2.4.6.
- Token ids, first 2,500 documents of `NeelNanda/pile-10k` at dataset commit `127bfedcd5047750df5ccf3a12979a47bfa0bafa`
  (the design's prefix resolves to this on the main branch today), OLMo-2-0425-1B-Instruct tokenizer at
  `48d788eca847d4d7548f375ad03d3c9312f6139e`, truncation at 512, `add_special_tokens=True` (adds no BOS; pad id
  100277), each id list JSON-dumped without spaces and concatenated: `83c4cc609ee52a4b0667e07628cb2401c9fe5005b85e025eb39dfea303eed78f`;
  930,289 tokens; 1,082 documents at the cap; none empty. The joining rule must be written into the collector, which
  currently computes no token hash at all.

Smallest fix: the collector writes into `progress.json`, at start, the sha256 of the token-id lists under the rule
above, the sha256 of each weight file it loads (the safetensors shards, by bytes; or of one fixed tensor such as
layer 0's `down_proj.weight`), and the numpy, torch and transformers versions; the design registers the two hashes
above and requires all 13 weight hashes to differ except `instruct` and `instruct_b`, which must be equal.

### 1.3 Gate 3: a median of 1.0 survives 30% of units being wrong, and two identical images are not two machines

Gate 3 requires the median neuron overlap between `instruct` and `instruct_b` to be at least 0.95. Overlap takes
values in steps of 0.1, so the bar means "the median is 1.0", which is "more than half of active units have identical
top-10 sets". A state in which 30% of units are randomised has median 1.0, mean 0.73, and 30% of units below 0.5;
Gate 3 passes (counterexample C). The downstream readings that would be poisoned are the per-unit ones: 181.6 on 21
units and 181.5's per-layer medians.

The two sessions are also the same cloud image on the same CPU class (the pilot's fp32 reruns were bit-identical), so
Gate 3 measures the collector's determinism, which the resume test already measures, and not the floor that matters:
how much the top-10 set moves under a change of the weights too small to mean anything. The pilot's one data point is
bf16 against fp32 activations on 40 documents, mean overlap 0.98, with the note that near-ties are denser at 2,500
documents. The right floor for 181.2 (`dpo` against `instruct`, bar 0.9) is therefore unknown. Part 107 says the DPO
to Instruct update moves 51% of changed elements by one bf16 step; a floor built from a perturbation of that size is
the comparison 181.2 needs.

Smallest fix: keep the two-session run as a determinism check but read its 10th percentile and the share of units
below 0.5 beside the median; add a fourteenth run, a float32-stored checkpoint (`pretrain_4T` or `midtrain_51B`) with
every weight rounded through bf16 and run in float32 (one line at load time, no speed cost on these CPUs, unlike a
bf16 run), and report its overlap with the unrounded run as the "nothing changed" floor; read 181.2 and the pretraining
steps of 181.4 against that floor.

## 2. The active-unit rule

A unit is active at a checkpoint when its maximum is at least 10% of the median maximum of its layer's units at that
checkpoint, and each pair is summarised on units active at both. Three consequences:

- **The unit set changes per pair, and the median follows the exclusion.** With every unit's overlap fixed across
  checkpoints and only the inactive share moving, the median over active units moves: rising inactive share gives
  medians 0.5, 0.6, 0.6, 0.6, 0.7, 0.7 along the six pretraining checkpoints, falling share gives 0.7, 0.7, 0.6, 0.6,
  0.6, 0.5, while the median over all units stays 0.5 (counterexample D; the excluded units are the low-overlap ones,
  which is what a maximum-based rule does, since a unit with no stable top document is a unit with a small, noisy
  maximum). 181.4 reads exactly this sequence. Which direction the real rule takes is unknown: early checkpoints may
  have near-uniform maxima (few inactive), later ones more dead units (more inactive).
- **The rule is tied to the measure.** Activity is judged on the corpus maximum, one document; the top-10 list
  contains that document and nine others. A unit active because of one spike has a stable top-1 and a churning tail.
  Selecting on the maximum therefore selects on part of the overlap.
- **Controls under a neuron rule.** It is not stated whether "its layer's units" includes the 256 controls. A control
  is `x @ d` over 8,192 coordinates, whose maximum has a different scale from a single coordinate's, so a neuron-median
  threshold may drop most controls or none; either way 181.5 compares differently filtered populations.

Smallest fix: fix one unit set in advance for every summary, units active at all 13 runs, and report every number
also over all 131,072 units; judge controls against their own population's median; and use an activity statistic that
does not share the top list, which pass 2 already provides (`frac_above` on the same 500 documents at both checkpoints).
With that, 181.3 to 181.5 cannot be manufactured by composition.

## 3. Fixed random directions as the control for a privileged basis (181.5)

Fixing the directions across checkpoints is right: a coordinate `e_i` is also a fixed direction, and the question is
whether the same fixed functional keeps its contexts. The problem is what else differs between the two functionals:

- A neuron is one coordinate; a control sums all 8,192, each with weight about 1/90. If every coordinate's change
  across checkpoints were independent, the control would accumulate all of them and the neuron one. Any such test then
  finds "neurons keep their contexts better" whether or not the coordinates are features, because a dense sum of
  changing things changes more than one thing. It is also known that SwiGLU coordinates are privileged by the
  architecture (the nonlinearity acts elementwise), so 181.5 as written is expected to hold and would not say what the
  headline wants it to.
- The control's top contexts are driven by the tokens with the largest activation norm along a random mix, which are
  often the same positions for many controls (sinks, first tokens, high-norm tokens). If those are stable across
  training the control overlap is high for a reason that has nothing to do with features.
- At the ceiling, "neuron median exceeds control median" is a tie: with both rates near 0.95 to 0.97, both medians are
  1.0 and the prediction fails in that layer by granularity (counterexample E). (midtrain_51B, instruct) is a pair where
  both may sit there.

A control that can be computed from the same runs at no extra forward cost, because controls are rows of the control
matrix and the hook's matmul cost is small beside the forward (0.17 s per document for 256 rows in the pilot):

1. **Sparse random directions** with support size k in {2, 4, 16, 64}, 64 of each per layer, random support, Gaussian
   or ±1/√k coefficients, seeded like the dense ones. Overlap as a function of k then says whether the neuron is the
   k = 1 end of a smooth curve (no privilege beyond "fewer summands") or sits above it. This is the question Mayner's
   sparse-set result (PRIOR-ART, 2026-10-31) raised for pushes, asked on the read side. A helper:

   ```python
   def generate_sparse_controls(n, width, k, seed, layer):
       rng = np.random.default_rng(stable_seed(seed, "sparse-controls", layer, k))
       values = np.zeros((n, width), dtype=np.float32)
       for row in range(n):
           idx = rng.choice(width, size=k, replace=False)
           values[row, idx] = rng.standard_normal(k, dtype=np.float32)
       values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), np.finfo(np.float32).tiny)
       return values
   ```

   Append these rows after the 256 dense rows; the state layout, the export and Gate 2's hash need only the larger
   control count. The k = 2 rows are the nearest "wrong basis" to a neuron: a 45-degree rotation of two coordinates.
2. **Read 181.5 on a paired statistic**, not two medians: the share of layers where the mean neuron overlap exceeds the
   mean k = 2 overlap, or a rank test over units within each layer; the mean has no ceiling tie.
3. Keep the shuffled-unit chance (about 0.004) as the floor of all of it.

## 4. The collector

I read every line and ran its tests. What I checked and found correct:

- **Padding.** Right padding with the attention mask; pass 1 masks padded positions to minus infinity before the
  per-document maximum, pass 2 masks them out of the counts and the reservoirs; position ids are the real ones because
  padding is on the right; the brute-force test compares batched activations with single-document forwards and
  passes. One entry per document in the top list (the batch contributes one per-document maximum per unit, and a
  document is in exactly one batch). Minus-infinity placeholders before k documents are skipped at export.
- **Resume.** The npz carries its own cursor and is written before `progress.json`, so a kill between the two writes
  cannot double-count; the batcher and the reservoir keys are pure functions of (config, chunk, batch); the test shows
  a killed and resumed run is bit-identical, and a changed config refuses to resume. Sound.
- **Controls.** Seeded per (seed, "control-directions", layer); identical across runs on the same numpy (section 1.2 on
  the numpy caveat); unit norm; projected in float32.

What I found wrong or missing:

- **Saturation snapshots are never written with the defaults.** `--snapshot-every 500` fires only when a chunk end is
  a multiple of 500, and with `--checkpoint-every 64` the chunk ends are 64, 128, ..., 2,496 and 2,500: on a 520-document
  toy run the real collector wrote no snapshot at 500 with checkpoint-every 64 and one with checkpoint-every 50
  (counterexample F). The design's saturation report would silently have only the final state. Fix: snapshot when
  `c0 // every != c1 // every`, or set checkpoint-every to a divisor of 500 and say so in the launch command.
- **Gate 1 has no code**, and Gate 2's token hash and weight hash are not computed anywhere (section 1).
- **`--docs` defaults to all 10,000 documents.** The design's 2,500 must be on the registered command line; the config
  records `docs`, so a wrong launch is detectable afterwards, not prevented.
- **No library versions in the config.** `make_config` records none; the pilot's TIMING.md does. Add numpy, torch and
  transformers.
- **Units whose maximum is negative** get `frac_above` near 1 in pass 2 (every position is above a negative tenth of
  the maximum) and reservoirs full of ordinary positions. They are inactive under the activity rule, but the exported
  `frac_above` for them is a number that means nothing; mark it.
- **Thread and CPU determinism.** Four-thread float32 matmuls can sum in a different order on a different CPU model;
  the two Instruct sessions will show whether the cloud gives identical machines, which is the only thing Gate 3 as
  written establishes.
- **State size in git.** Thirteen `state.npz` files at 57 MB each is 741 MB in branch history; `savez_compressed`
  gives 35 MB each, and none of it should reach main.

## 5. Part 92's screen flip rates, re-derived

From `results/part92/screen-instruct/layer_*.jsonl` (every screened row with its triage class) joined to
`results/part92/labels-screen/labels.jsonl` through `results/part92/labels-screen/KEY-private.json`, with Part 86's
four-way class rule (coherent, and the kind neither "still Qwen or an assistant" nor "unclear"); the script is
`review/part181/part92_screen_rates.py`.

| Population | Rows | Triaged changed | Coherent persona | Flip rate over all rows | Coherent persona among changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Real neurons | 131,072 | 19,934 (15.2%) | 1,709 | 1.30% | 8.6% |
| Random directions | 4,096 | 474 (11.6%) | 45 | 1.10% | 9.5% |

The 1.3% against 1.1% recorded in PRIOR-ART holds, and 1,709 is the denominator Part 97 used. The split behind it is
the useful part for the Findings draft: among answers that changed at all, a random direction's answer is a coherent
persona as often as a neuron's (9.5% against 8.6%); the whole difference in the flip rate is that neurons change the
answer more often (15.2% against 11.6%). Persona flips are concentrated in layers 2 to 6 (221, 191, 318, 201 and 244
of 8,192 each) and are nearly absent from layers 13 to 15 (6, 9 and 3); random directions flip 0 to 11 of 256 per
layer. Eight neuron labels are invalid rows. Only changed rows were read, so both rates are lower bounds, as in Part 86.

## 6. Before registration

1. Gate 1: implement it on its own code path; add the random-document pairs, the padded-row quota and the
   no-duplicate invariant (section 1.1).
2. Gate 2: compute the token hash and the weight hashes in the collector; register the two values in section 1.2 and
   the expectation that the 13 weight hashes differ except the Instruct pair; record library versions.
3. Gate 3: read the distribution, not the median; add the bf16-rounded float32 checkpoint as the noise floor and read
   181.2 and the pretraining steps against it.
4. Fix the unit set for every summary and judge controls against their own population (section 2).
5. Add sparse controls and read 181.5 on a paired statistic (section 3).
6. Replace "rises strictly" in 181.4 with a reading that survives granularity: the mean, or a per-unit paired sign
   test with a registered share, with the median reported beside it.
7. Fix the snapshot cadence and register the launch command with `--docs 2500` (section 4).
8. Prior art: PRIOR-ART already holds the closest work, Bektursun's `already-clean-audit` with 16 OLMo-3 stage-1
   checkpoints (2026-11-02 entry); the design should cite it beside Tigges.

Nothing here touches the question, which is a good one and cheap to answer well; the fixes are a day's work on the
collector and an evening's on the design.
