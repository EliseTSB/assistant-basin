# Part 198: Do ordinary knowledge edits leave a trace in the weights that a check without the original model can find?

Procedure: v1
DRAFT, not registered. Follows Part 193 and Elise's direction of 2026-11-07 23:36 CT (Discord 1568861686167904287):
"Methods to detect weight edits would be particularly valuable". Defensive only: every edit is a benign counterfactual
fact from a public benchmark (CounterFact), and nothing here builds or releases an edit meant to avoid detection. The
design was rewritten on 2026-11-08 at 14:59 CT (commit 75f20fe) after the pilot below, before any prediction existed.

## Question

Part 193 found our chicken edit without the original weights because its update was 15.6 times its host matrix's top
singular value. The pilot below shows that ordinary edits are not like that: four published editors on GPT-2 XL and the
`wsp` writer on Qwen3.5-0.8B make updates of 0.04 to 0.72 times the host's top singular value, the host's top singular
value moves under 1%, and the simple check (F1) ranks each edited matrix where the unedited one already stood. So the
questions are now:

1. Is that blindness general? Across 24 facts per editor, do ROME, MEMIT and AlphaEdit single edits on GPT-2 XL, and
   their mass edits, stay invisible to every base-free statistic tried in the pilot (F1, F2, mean pairwise cosine
   similarity of neuron output vectors, and a reading of singular directions through the unembedding)?
2. Does a reading through the unembedding find writes that aim at the answer's output direction? `wsp` builds its targets
   from the answer token's unembedding row. On 24 new facts, does a content-word token score, computed from the suspect
   model alone, rank an edited layer first and name the written token?
3. How often does an unedited model already carry a matrix that would pass the same test? A threshold is set on three
   unedited models and checked on others it has not seen, including both hosts.

## Prior art and existing data

- Closest published work: *Tracing and Reversing Edits in LLMs* (Youssef et al., ICLR 2026, arXiv:2505.20819) detects
  ROME edits by row-vector cosine similarity, strongly on GPT-2 XL and GPT-J and not on Llama 3, and reports AlphaEdit
  edits needing far more singular components to reverse. Norm growth of edited matrices is documented under sequential
  editing (Gupta et al. arXiv:2403.07175, arXiv:2502.01636; Huang et al. arXiv:2410.23843). `llm-forensics` 0.1.1 (PyPI,
  2026-11-06, BSD-3-Clause, github.com/bohemian-miser/LLM_Forensics) ships a no-base `spectral` detector, unreviewed and
  self-evaluated on 36 edited and 2 unedited checkpoints. Muse's sweep (council `muse-prior-art` #14) found no paper
  measuring one edit's update norm against its host matrix's top singular value. Journal
  `notes/reports/weight-edit-detection-2026-11-07/PRIOR-ART.md` keeps the list.
- Reading a weight direction as tokens: projecting parameters through the unembedding is Dar et al., *Analyzing
  Transformers in Embedding Space* (arXiv:2209.02535). `llm-forensics` 0.1.1 includes it as `t_unembed_vector`, a tool its
  LLM investigator agent can call on a matrix's singular directions, and its `spectra` module scores the participation
  ratio of each top singular vector ("a ROME/MEMIT value vector is diffuse"; read 2026-11-08 from the wheel's source,
  not run). Its docstring also reports that in its analyst sessions the edit sat in s2 or s3, not s1. Neither is scored
  as a detector with a false-alarm rate. Youssef et al. infer the edited object with a decoder trained on 600 labelled
  edited matrices (99.4% for ROME on GPT-2 XL; 1.9% for AlphaEdit), not by reading directions. Two web searches
  (2026-11-08 14:4x CT) found no paper that scores an untrained unembedding reading of a matrix's singular directions
  as an edit detector.
- Our own data: Part 193 (results/part193): F1 and F2 on EmbeddingGemma 2, three edits, the sibling-scale spread per
  role (1.7 to 3.0 times, 6.8 for one role) and Gate 4's lesson that twice a target's own norm guarantees nothing.

## Method

**Statistics, all computed from the suspect model alone** (the base is used only for the mechanism checks marked
base-required, and to confirm an edit took). For each MLP output matrix (GPT-2 XL `mlp.c_proj`, 48; Qwen3.5-0.8B
`mlp.down_proj`, 24; the unedited models' equivalents), with W in [out, in] orientation:

- F1: top singular value; F2: its share of the squared spectrum (Part 193's statistics).
- PCS: mean pairwise cosine similarity of the per-neuron output vectors (the columns of W), computed exactly as
  (||sum of unit columns||^2 - n) / (n (n - 1)).
- Token reading: the top 64 output-side singular vectors u, each read as the logits E (g * u) through the unembedding E
  and the final norm's per-dimension scale g (for a LayerNorm, u is centred first; g is taken as the norm module's output
  on a vector of ones, so zero-centred norms are handled). Each direction's score is z = (max logit - mean) / std over
  the vocabulary, the better of the two signs, and its top token is the argmax. **T_max** is the largest z over the 64.
  **T_word** is the largest z among directions whose top token is a content token: after stripping whitespace, three or
  more ASCII letters and not in the fixed stopword list in `stats198.py`.

"Found" for a statistic: an edited matrix ranks first among the same role's matrices in the suspect model, with no tie.
For multi-layer editors, any edited layer ranking first counts, and each layer's rank is reported.

**Arm A: gradient editors on GPT-2 XL.** GPT-2 XL float32 (sha256 `0f8b28eb…11736a6a` of `model.safetensors`) on the
CPU under an 11 GB memory cap with weights copied into ordinary memory. EasyEdit (main fetched 2026-11-08 11:16 CT,
`src/EasyEdit-main`, imported through `ee_import.py` with only ROME, MEMIT and AlphaEdit) with each method's shipped
GPT-2 XL hyperparameters unchanged and the published `wikipedia_stats` second moments for layers 13 to 17. Records: 24
drawn by seed 198 from CounterFact cases 11 onward (cases 0 to 10 were the pilot), each edited alone into the restored
base by ROME (layer 17), MEMIT and AlphaEdit (layers 13 to 17). Mass edits: two batches of 10 further records (same
seed, next 20), each by MEMIT and by AlphaEdit. One process per method, the base restored bit for bit between records
(the pilot checked `restored_bit_exact` on every run). r-ROME is dropped: its pilot edit matched ROME's size and rank
(0.54 against 0.53, rank 30), and it costs 150 s an edit.

**Arm B: `wsp` on Qwen3.5-0.8B.** `wsp.py` at commit `763259257133`, unchanged, through `wsp198.py`, float32 on the CPU,
default settings. Records: 24 drawn by seed 198 from CounterFact cases 11 onward whose relation is one of P103, P1412,
P37, P30, P27, P17, P36, P19, P364 and P159, keeping only records whose `target_new` is one word of three or more ASCII
letters and differs from every record already drawn (so one answer cannot dominate). Each is written alone as three
questions from a fixed template for its relation, plus one held-out question, then undone (`WSP.undo`, which subtracts
the write exactly) before the next. One sequence arm: 10 further records written one after another into the same copy,
statistics read after the tenth.

**Arm C: unedited models.** Calibration set: Pythia-410M, Qwen3-0.6B and OLMo-2-0425-1B. Held-out set: GPT-2 XL,
Qwen3.5-0.8B, Pythia-1.4B, Qwen3-1.7B, SmolLM2-1.7B, SmolLM3-3B and Qwen2.5-0.5B-Instruct, all already on disk, loaded
in bfloat16 with every statistic computed in float64. **T\*** is the largest T_word over every MLP output matrix of the
three calibration models. Also reported per model: F1's robust z of its largest member and the ratio of the largest to
the second-largest top singular value.

**Took.** GPT-2 XL: the new object outranks the old one on the rewrite prompt (CounterFact efficacy, the first token).
`wsp`: all three taught questions answer the target exactly (`free_exact` 3/3). Edits that did not take are counted and
left out of every rate.

**Mechanism checks (base-required, not detection).** The update dW's own top output direction read as tokens, and
whether it names the target's first token (its top three tokens, stripped and case-folded, include it, with or without a
leading space).

## Controls

- Do-nothing edit: after each GPT-2 XL record the restored weights equal the base bit for bit; after each `wsp` undo
  the accumulated update is exactly zero. A run where either fails stops.
- Matched random updates (Arm B null): for each `wsp` write and each edited layer, an update with the write's own singular
  values and random orthonormal singular vectors (seed 198 plus the record index) added to the base. If the token reading
  finds these, it is reacting to update size, not to an aimed direction.
- Aimed positive control (Arm A): for each GPT-2 XL record, a rank-one update at layer 17 whose output direction is the
  final-norm-scaled unembedding row of the record's new object (centred, unit length), with a random unit input
  direction and the spectral norm of that record's ROME update. If the reading cannot see this, a null on the gradient
  editors says nothing about their direction, only about the reading's reach at layer 17 of 48.
- Code cross-check: `stats198.py` reproduces the pilot's F1 ranks for the seven pilot runs and the exploration's T_max
  for every base layer of both hosts, to the printed precision.

Gate (run before any data is read): the do-nothing checks pass, the cross-check passes, and at least 21 of 24 single
edits take for each GPT-2 XL method and 21 of 24 writes take for `wsp`.

## Pilot (before registration; results disclosed in this section before any prediction is written)

One CounterFact record outside the main draw, edited by each of the four methods, and MEMIT and AlphaEdit on a batch of
10: time per edit, peak memory, efficacy, the norm ratio and the F1 rank of each edited matrix. The two base-free
checks on GPT-2 XL and two other unedited models. `llm-forensics` installs and runs. Stopping rule: if any method fails
to install or take on the pilot record, it is dropped or fixed before registration and the change is stated here.

### Pilot results, part 1 (2026-11-08 11:4x to 13:0x CT, before any prediction)

GPT-2 XL float32 (`model.safetensors` sha256 0f8b28eb05a8075f48b61b6f35332978c74fc7763fa9fb4051a1c30511736a6a), EasyEdit
main as downloaded 2026-11-08 11:16 CT (tarball, not yet pinned to a commit), CPU at nice 10 under a 10 GiB cap, CounterFact
case 0 (Danielle Darrieux, mother tongue French to English). Each edit was made in place and the base restored bit for
bit afterwards (checked). Base spectra of the 48 `mlp.c_proj` matrices took 21 s.

| Method | Edit time | Took | Edited layers | Update spectral norm / host's top singular value | F1 sibling rank (of 48) |
| --- | --- | --- | --- | --- | --- |
| ROME (EasyEdit config, `mom2_adjustment: false`) | 128 s | yes | 17 | 0.53 | 30 |
| ROME with `mom2_adjustment: true` (the original paper's) | 313 s | yes | 17 | 0.72 | 29 |
| R-ROME (EasyEdit config) | 153 s | yes | 17 | 0.54 | 30 |
| MEMIT, one record | 115 s | yes | 13 to 17 | 0.04, 0.045, 0.061, 0.089, 0.155 | 19, 20, 22, 23, 30 |

So a single ordinary fact edit on GPT-2 XL is about half of its host matrix's top singular value, and F1 does not find it.
Part 193's chicken edit, at 15.6 times, was far outside this range. AlphaEdit did not finish: it ran for about 40 minutes at
the 10 GiB cap in uninterruptible disk wait (the weights were still memory-mapped from `/mnt/e`, and under the cap the
kernel kept evicting and re-reading them). The pilot now copies every tensor into ordinary memory after loading, and
AlphaEdit and the two batches of 10 are rerunning under an 11 GiB cap (part 2).

### Pilot results, part 2 (2026-11-08 13:08 to 13:17 CT)

With every tensor copied into ordinary memory after loading, AlphaEdit's single edit took 69 s (the edit took; layers 13
to 17; update over host top singular value at most 0.228; best F1 sibling rank 19 of 48, not found). MEMIT on a batch of
10 was then killed by the 11 GiB cap while computing layer 17's covariance, most likely with AlphaEdit's global caches
(projection and accumulated key matrices, about 1.6 GB) still held from the run before. Next: one method per process,
the cap set from Windows' free memory at launch.

### Pilot results, part 3 (2026-11-08 13:45 to 14:05 CT, one method per process)

Batches of 10 (CounterFact cases 1 to 10), each in a fresh process under the 11G cap with weights copied into ordinary
memory: MEMIT-b10 took 8 of 10, largest update 0.222 times its host's sigma1 (layer 17; layers 13 to 16 at 0.059 to
0.127), 453 s, peak 10.9 GiB. AlphaEdit-b10 took 10 of 10, largest 0.518 (layer 17; 0.221 to 0.357 below), 491 s,
peak 10.9 GiB. Ten facts at once make the largest update 1.4 times (MEMIT) and 2.3 times (AlphaEdit) the single edit.

**The edited sigma1 barely moves, so F1's rank is the host layer's own.** Across all seven runs the edited matrix's sigma1
is 0.9997 to 1.0064 times the base's (largest: AlphaEdit-b10, layer 17). Every F1 rank in parts 1 to 3 (19, 20, 22, 23,
29 or 30) is the base rank of that layer's sigma1 among the other 47 (13.15, 12.82, 12.38, 11.38 and 10.26 for layers 13
to 17; the largest sibling is layer 47 at 27.78). An update of half the host's sigma1 changes sigma1 by under 1% because
its direction is close to orthogonal to the host's top singular direction. For F1 to rank an edited layer first, its
sigma1 has to pass 27.78. Because sigma1(W + dW) is at most sigma1(W) + ||dW||_2, that needs a spectral norm of at least
1.11 to 1.71 times the host's sigma1 (layers 13 to 17), reached only if the update points along the host's top direction.
The pilot's updates do not (sigma1 moved under 1%); for an update whose input direction lies outside the host's row space
and whose output is the host's top direction, the need is sqrt(27.78^2 - sigma1^2)/sigma1, 1.86 to 2.52, and any other
output direction needs more. The largest ordinary update measured here is 0.518.

### Pilot results, part 4: one wsp write on Qwen3.5-0.8B (2026-11-08 14:05 to 14:14 CT)

`wsp.py` (commit `763259257133`, unchanged, imported by `wsp198.py`) on the CPU in float32, default settings (layers 17 to
21, spread 5, lam 3, cap 0.25). One fact, CounterFact case 0 as three questions ("What is the mother tongue of Danielle
Darrieux?" and two rephrasings), answer "English". Setup 385 s (mostly the one-time recording of 40 protected answers),
write 36 s, one round, peak 6.6 GiB. All three taught questions and the held-out question ("In what language did
Danielle Darrieux first learn to speak?", base answer French) answer "English"; the first 60 characters of all 8 of
wsp's unrelated check answers are unchanged.

Per edited `mlp.down_proj`, layers 17 to 21: spectral norm of the update 0.513, 0.442, 0.445, 0.552 and 0.566 times the
host's sigma1; Frobenius norm 3.4% to 4.0% of the host's, so the 25% cap did not bind (the host's Frobenius norm is 13.0
to 14.5 times its sigma1); numerical rank 2 at layer 17 and 6 at layers 18 to 21 (three phrasings, an answer token and an
end token each). The edited sigma1 is 0.9998 to 1.0020 times the base's, and every F1 rank (24, 20, 18, 21, 23 of 24) is
that layer's own base rank; the largest sibling sigma1 is layer 2's, 1.79, against 1.17 to 1.32 for layers 17 to 21.

So a second editor, on a second model family, shows the GPT-2 XL picture: about half the host's sigma1, pointed away from
the host's top direction, invisible to F1. By the triangle inequality, F1 could rank one of these layers first only with an
update of at least 0.536, 0.450, 0.353, 0.456 and 0.467 times the host's sigma1 (layers 17 to 21) along its top
direction, more in any other direction. Three of wsp's five updates (layers 19 to 21) are already larger than that floor;
what keeps them invisible is their direction, not their size.

### Exploration: reading singular directions through the unembedding (2026-11-08 14:37 CT, design input only)

Size does not hide these edits from F1; direction does. So the next statistic should look at direction. An exploration on
the wsp update (`explore_tokens.py`, output `explore-tokens.json`) read the top 64 output-side singular vectors of every
`mlp.down_proj`, base and edited, through the final norm's scale and the unembedding, scoring each by how concentrated it
is on one token: z = (max logit - mean) / std over the vocabulary, the better of the two signs.

- The update's own top singular vector reads as "English" at every edited layer (z 19.3 to 19.9); its second reads as the
  end-of-turn token (z 17.8 to 23.9).
- In the edited matrix, the most concentrated of the top 64 directions reads as "English" at 4 of 5 edited layers (z
  11.1, 9.3, 16.2 and 12.8 at layers 17, 19, 20 and 21, at singular indices 5, 7, 1 and 3). At layer 18 a natural
  direction (" a", z 14.1 in the base too) stays on top, and "English" is at index 23 (z 6.9).
- The base maxima over the top 64 are 5.9 to 16.0 across all 24 layers (16.0 at layer 22, 14.1 at layer 18), and their
  top tokens are spaces, punctuation and function words. So the raw maximum alone does not separate layer 20 (16.2) from
  unedited layer 22 (16.0). What does separate is content: an ordinary word, the same word in several directions of one
  matrix (three each at layers 17 and 20), and the same word in five consecutive layers.

⚠ This is the most favourable case possible and cannot count as evidence. wsp builds its targets as the final norm's
scale times the token's unembedding row (`wsp.py` line 200), so its update reads as the token by construction. ROME,
MEMIT and AlphaEdit optimise a value vector by gradient instead, so the same reading on their updates is the real test.
Any statistic built from what this exploration shows is then scored on facts, editors and unedited models that it has
not seen.

**The real test came back null (2026-11-08 14:51 CT, `explore_gpt2.py`, output `explore-gpt2.json`).** ROME, MEMIT and
AlphaEdit single edits of the same fact on GPT-2 XL were re-run saving their updates (same sizes as parts 1 and 2: ROME
0.53, MEMIT 0.155 and AlphaEdit 0.238 at their largest layer). Each update is rank one. Read through `ln_f` and `wte` the
same way, its direction scores z 4.4 to 4.9 with top tokens like 'lé', ' Mé', ' Chilean', ' Quebec' and 'Num', never
"English", against base maxima of 5.1 to 7.7 for layers 13 to 17. The edited matrices' most concentrated directions are
the base's own (same index, same tokens, z within 0.4). GPT-2 XL's base maxima over all 48 layers run 5.1 to 17.1, and the
high ones are function words in late layers (' to' at 36, newline at 47, ' of' at 38). So an untrained reading through the
unembedding sees wsp's writes, which target the unembedding by construction at layers 17 to 21 of 24, and not gradient
editors writing mid-network at layers 13 to 17 of 48, where a plain reading through the unembedding is known to be weak.

**Two more base-free statistics, also null for the GPT-2 XL editors (`explore_pcs.py`, `explore-pcs.json`).** The top 12
singular values of every edited matrix move under 2% (ROME at layer 17: the largest change is 6.74 to 6.87 at index
11). Mean pairwise cosine similarity of the per-neuron output vectors (the statistic family of Youssef et al. App. C)
rises 1.7 times for ROME (rank 10 of 48 among siblings) and not at all for MEMIT or AlphaEdit; in the other orientation it
does not rise. Youssef et al. report a much larger rise on GPT-2 XL; their exact definition may differ from these two
orientations, which is a hypothesis, not checked.

**Content-word scores and an aimed write on GPT-2 XL (design input, 2026-11-08 14:57 CT).** Computed from the
exploration's saved top tokens: T_word on unedited Qwen3.5-0.8B is 5.4 to 7.3 over its 24 layers, and on the `wsp`-edited
layers 17 to 21 it is 11.1, 6.9, 9.3, 16.2 and 12.8, so layer 20 ranks first. On GPT-2 XL, T_word is 5.1 to 9.8 (9.79 at
layer 45, then 9.07 and 8.87 at layers 46 and 43), and none of the three gradient edits moves its layer's T_word by more
than 0.4 or ranks first. `explore_posctl.py` then added a rank-one update aimed at " English" (output direction the
final-norm-scaled unembedding row, random input direction) to layer 17 at the ROME update's spectral norm, 5.432: layer
17's T_word went from 5.08 to 8.32, naming " English" (direction 12), but stayed below layer 45's 9.79, so it is named
and not found. At half that norm nothing shows (5.11, "ividual"); at twice it, 23.8, found. So the reading does reach
layer 17 of 48, and a gradient edit of ROME's size is not aimed the way this write is.

What the pilot now says, before any prediction: for ordinary edits at default settings, every base-free weight statistic
tried (F1, the top of the spectrum, two cosine statistics, an untrained token reading) is blind on GPT-2 XL, and only the
token reading sees wsp. The registration should therefore test (1) the token reading against unembedding-targeted writers
(wsp) on new facts, with its false-alarm rate on unedited models, and (2) whether the blindness to gradient editors holds
across facts, batch sizes and a second model, rather than look for a size threshold the pilot already rules out.

## Size

24 records per editor and arm, 2 mass-edit batches per mass editor, 10 records in the sequence arm, 3 calibration and
7 held-out unedited models. The predictions contrast rates near 0 with rates near 1, which 24 resolves easily: 1 of 24
against 20 of 24 differ at p below 0.0001 (two-sided Fisher exact), and 0 of 24 bounds a rate under 14% (one-sided 95%).
Work: 24 x 3 GPT-2 XL single edits (ROME 128 s, MEMIT 115 s, AlphaEdit 69 s per edit in the pilot) about 2.1 h; 4 mass
edits about 0.5 h; 24 + 10 `wsp` writes with their controls about 0.5 h (setup 385 s once, about 36 s a write); 10
unedited models about 0.3 h. Total about 3.4 h of CPU, one process at a time.

## Optimization

Measured in the pilot, so these are measurements, not guesses: per-edit times above, peak 10.9 GiB for GPT-2 XL (so one
process at a time, under an 11 GB cap, after Windows reports 11 GB free) and 6.6 GiB for Qwen3.5-0.8B. Changes from the
pilot: one process per method for all its records (saves about 35 s of loading per edit, about 40 min in all); weights
copied into ordinary memory after loading (the pilot's AlphaEdit step thrashed for 40 min without it); results appended
per record to a JSONL, so a killed run resumes at the next record and loses at most one; `wsp`'s protection cache is
recorded once and reused. The statistics take seconds per matrix and are not worth optimizing.

Levers (procedure section 4): bound resource is the CPU, measured busy in the pilot (12 threads, ROME's value-vector
optimisation dominates); batch size does not apply to sequential single edits, and the mass edits are the methods' own
batches; generation is only `wsp`'s 24-token check answers, kept because "took" needs them; there are no random controls
to pool beyond one matrix per record; no readers or labelling, so no quota is used; the A770 is declined because
EasyEdit's paths are written for CUDA and untested on XPU, and the GPT-2 XL work fits a CPU afternoon; host memory is
handled by the 11 GB wait and the copy into ordinary memory; cache and resume are taken (per-record JSONL, `wsp`'s
protection cache).

## Predictions

Each is scored once on the registered run. "Took" edits only. Numbers in brackets are the pilot's, from one record.
Author of every prediction: Journal Claude (Opus 5.5), 2026-11-08, written after the pilot and before any registered
data exists.

**Settings, matched within each arm and fixed here.** Arm A: GPT-2 XL float32 (sha256 above), EasyEdit shipped GPT-2 XL
hyperparameters, `wikipedia_stats` second moments, CPU, 12 threads, no sampling anywhere (efficacy is one forward pass).
Arm B: Qwen3.5-0.8B float32 (snapshot `2fc06364715b967f1860aea9cf38778875588b17`), `wsp.py` defaults (top layer 21,
spread 5, lam 3, mu 0.01, cap 0.25, 6 rounds, contrast weight 10), its chat template with thinking off, greedy decoding,
answers capped at 24 new tokens. Arm C: each model's own snapshot in bfloat16, statistics in float64. Token reading in
all arms: top 64 singular directions, both signs, z over the full vocabulary, the stopword list in `stats198.py`.

**Headline and its withdrawal.** The headline would be "ordinary gradient edits at shipped settings are invisible to
these base-free statistics, while a writer that aims at the answer's output direction is found and named". It is
withdrawn if 198.5 fails (the reading could not have seen an aimed write, so the null says nothing about direction) or
if 198.6 fails; each other prediction is reported on its own.

| # | Prediction | Fails if | Confidence |
|---|---|---|---|
| 198.1 | F1 finds at most 1 of 24 single edits for each of ROME, MEMIT and AlphaEdit, and none of the 4 mass-edit models. [0 of 1 each; batches 0 of 2] | any method finds 2 or more, or any batch is found | 0.9 |
| 198.2 | T_word finds at most 1 of 24 single edits for each gradient editor, and none of the 4 mass-edit models. [0 of 1 each] | 2 or more for any method, or any batch | 0.85 |
| 198.3 | PCS finds at most 1 of 24 single edits for each gradient editor, and none of the 4 mass-edit models. [ranks 10 to 41] | 2 or more for any method, or any batch | 0.85 |
| 198.4 | Gradient edits are not aimed at their answer: the edited layer's top content direction names the new object in at most 2 of 24 for each editor, and so does the update's own top direction (base-required). [0 of 1 each, both] | 3 or more for any editor on either | 0.85 |
| 198.5 | The aimed positive control is named but mostly not found: its layer-17 top content direction names the new object in at least 20 of 24, and T_word ranks layer 17 first in at most 12 of 24. [named, not found] | named in fewer than 20, or found in more than 12 | 0.6 |
| 198.6 | T_word finds `wsp` writes: an edited layer ranks first among the 24 in at least 20 of the writes that took. [found, 16.2 against 7.3] | fewer than 20 | 0.75 |
| 198.7 | The suspect model names what was written: the top content direction of the layer ranking first names the new object in at least 18 of the writes that took. [named] | fewer than 18 | 0.7 |
| 198.8 | After 10 `wsp` writes in sequence, at least 5 of the 10 new objects are named by a content direction (top token, any of the top 64) in layers 17 to 21. [not piloted] | fewer than 5 | 0.5 |
| 198.9 | Matched random updates are not found: T_word ranks a randomized layer first in at most 1 of 24. [not piloted] | 2 or more | 0.85 |
| 198.10 | At most 2 of the 7 held-out unedited models have any MLP output matrix with T_word above T\*. [GPT-2 XL's maximum is 9.79 and Qwen3.5-0.8B's 7.3; T\* not yet computed] | 3 or more | 0.5 |
| 198.11 | At least 20 of the `wsp` writes that took have an edited layer with T_word above T\*. [16.2] | fewer than 20 | 0.65 |

## Reading rules

- The gate failing stops the part before any prediction is scored.
- "Found" is rank 1 among the same role's matrices in the suspect model, with no tie; for multi-layer edits any edited
  layer ranking first counts. "Named" is the stripped, case-folded first token of the new object (with or without a
  leading space) among a direction's top three tokens.
- T\* is computed from the calibration models before any held-out model or Arm B write is scored, and written into the
  results before they are.
- 198.1 to 198.4 are the null side: passing them says these statistics are blind to these editors at their shipped
  settings, not that the edits are undetectable by any means. 198.5 is the check that the reading could have seen an
  aimed write at the same layer and size; if it fails, 198.2 and 198.4 are reported as uninformative about direction.
- Every claim is scoped to GPT-2 XL and Qwen3.5-0.8B, these editors at their shipped settings, CounterFact records and
  these statistics. Nothing here bears on edits designed to avoid detection, and none is built.

## Limits

- Two edited models, both small; the gradient editors run only on GPT-2 XL, the model their hyperparameters target, and
  Youssef et al. found their own signal absent on Llama 3.
- The token reading is untrained (a plain logit lens on weights). A trained reader (a tuned lens) might see mid-network
  edits it cannot; that would need text and training, and is out of scope.
- The content-token filter is English and ASCII, and its stopword list was written after the pilot showed function words
  among the unedited maxima. It is fixed in `stats198.py` before registration and not changed after.
- T_word and its filter were chosen from one `wsp` write; the 24 new writes, the random control and the held-out unedited
  models are the test. The false-alarm arm is 7 models, so it can catch a common false alarm, not estimate a small rate.
- Detection is ranking among siblings in one model; it says nothing about separating an edit from fine-tuning, which
  changes every matrix at once.
- `llm-forensics`' own detectors are not run here: its token reading is an agent tool, not a scored statistic, and its
  participation-ratio score is left to a follow-up so this part stays one question wide.
