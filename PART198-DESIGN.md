# Part 198: Do ordinary knowledge edits leave a trace in the weights that a check without the original model can find?

Procedure: v1
Registered 2026-11-09 00:23 CT (public main a7d18c5; revision 4 and these procedure lines re-registered before any data)
Follows Part 193 and Elise's direction of 2026-11-07 23:36 CT (Discord 1568861686167904287):
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
3. What do unedited models carry? Every Arm C value was measured by the design reviewer before registration, so Arm C
   is a descriptive replication, not a held-out test; the two hosts are reported separately, as development models.

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
  (2026-11-08 afternoon; the time was not recorded) found no paper that scores an untrained unembedding reading of a matrix's singular directions
  as an edit detector.
- Our own data: Part 193 (results/part193): F1 and F2 on EmbeddingGemma 2, three edits, the sibling-scale spread per
  role (1.7 to 3.0 times, 6.8 for one role) and Gate 4's lesson that twice a target's own norm guarantees nothing.

## Method

Revised 2026-11-08 after Alden's design hold (council `part198-design-2026-11-08` #3); the changes are listed under
"Revision 1" at the end of this file.

### Statistics, computed from the suspect model alone

For each MLP output matrix (GPT-2 XL `mlp.c_proj`, 48; Qwen3.5-0.8B `mlp.down_proj`, 24; the unedited models'
equivalents), W in [out, in] orientation, at the weights' stored precision cast to float64 (never down-cast):

- **F1**: top singular value. **F2**: its share of the squared spectrum (Part 193).
- **PCS**: mean pairwise cosine similarity of the per-neuron output vectors (the columns of W), computed exactly as
  (||sum of unit columns||^2 - n) / (n (n - 1)).
- **Token reading.** Take the top 64 left (output-side) singular vectors u_j. Read each through the final norm and the
  unembedding E as logits l = E (gain * u'), where for a **LayerNorm** u' = u - mean(u) and gain = the norm's weight
  (gamma; its bias is omitted because u is a direction), and for an **RMSNorm** u' = u and gain = the norm's effective
  per-dimension scale, which is its output on a vector of ones (this gives 1 + weight for offset-parameterised norms such
  as Qwen3.5's, and weight otherwise). LayerNorm(ones) is the bias, not gamma, so the ones trick is never used for a
  LayerNorm. A fixture with nonuniform gamma and nonzero beta, and one offset RMSNorm, must give the right gain before
  any data. For every direction j and sign s in {+1, -1}: z(j, s) = (max of s*l - mean of s*l) / std of s*l over the
  vocabulary, and top(j, s) = the token ID at that max; a pair whose logits have zero variance is dropped.
  - **T_max** = max of z over all (j, s) pairs.
  - **T_word** = max of z over the **eligible** pairs: those whose top token, decoded and stripped of whitespace, is three
    or more ASCII letters and not in the stopword list frozen in `stats198.py`. Both signs are considered for every
    direction, so a function-word sign never hides an eligible opposite sign. No eligible pair: T_word = none, which
    ranks below every number. ⚠ The pilot's exploration chose each direction's higher-z sign first and filtered after;
    `stats198.py` recomputes the pilot values under this rule and reports both. Done 2026-11-08 15:59 CT
    (`crosscheck198.json`, gate passed): on the unedited hosts the frozen rule changes one base value, Qwen3.5-0.8B
    layer 3 (5.39 to 5.55); every other base T_word on both hosts is unchanged.

**Localization** (not an alarm, since every unedited model also has a first): an edited matrix has strictly the highest
value of a statistic among the same role's matrices in the suspect model. A tie for first is not localization and is
reported. For a five-layer edit, localization means any of its edited layers is first.

A cross-model threshold alarm (T_word above a value set on other models) is **not** an endpoint: the DeepSeek design
review measured it before registration and it false-alarms (see Arm C).

**Named.** For each record the target token set is frozen before scoring: GPT-2 XL, the first token ID of " " + object
and of object; `wsp`, the first token ID of the answer as `wsp` encodes it (`enc(object)[0]`) and of " " + object. A
direction-sign pair **names** the object (primary, strict) when its top token ID is one of the target IDs; it names it
**loosely** (secondary, reported beside) when a target ID is among its three highest-logit token IDs. A record is
**name-eligible** only if at least one target ID passes the eligibility filter above; ineligible records are counted and
left out of every naming rate, so an unnameable object is never evidence about direction. **The blind reader** (the
primary endpoint for `wsp`, and the reader for every suspect): without any edit metadata, take the suspect's layer with
the strictly highest T_word across all its MLP output matrices, then that layer's top eligible pair, and output (layer,
token). Only then score it: **localized** if the layer is an edited one, **named** if the token is a target ID, and
**blind joint success** if both. No unique highest layer, or no eligible pair anywhere: the reader outputs nothing and the
record is a failure. **The oracle reader** (secondary, a known-location mechanism probe): the top eligible pair of the
edited layer with the highest T_word, scored for naming only; 198.4 uses it.

**Ties inside a score**, target-independent and frozen: a token argmax with equal logits takes the smallest token ID, and
the top three are ordered by logit, then token ID; equal z between pairs is broken by the smaller direction index, then
sign +1 before -1. Matrix-level ties are as in localization.

**Base diagnostics, reported beside every edited value** (base-required, never part of a detection endpoint): each
edited layer's base T_word, its base rank, and whether the base already names the record's object.

### Arm A: gradient editors on GPT-2 XL (the host is a development model)

GPT-2 XL float32 (`model.safetensors` sha256 `0f8b28eb05a8075f48b61b6f35332978c74fc7763fa9fb4051a1c30511736a6a`) on the
CPU under an 11 GB memory cap, weights copied into ordinary memory after loading. EasyEdit at commit
`5c519afb14e39c7f70bc5536ebd777a62d8493bf` (main when fetched 2026-11-08 11:16 CT; `rome_main.py`, `AlphaEdit_main.py`
and `hparams/MEMIT/gpt2-xl.yaml` checked byte-identical to that commit; the method sources, the three GPT-2 XL configs
and the five second-moment files are pinned by sha256 in `results/part198/manifest.json` at build), imported through `ee_import.py` with only ROME,
MEMIT and AlphaEdit, each with its shipped GPT-2 XL hyperparameters unchanged. Records are frozen in
`results/part198/cases198.json` (seed 198, CounterFact cases 11 onward, drawn 2026-11-08 by `draw198.py`, CounterFact
sha256 recorded there): 24 single records (cases 5494, 7275, 6736, 18976, 17313, 13124, 5137, 20706, 4065, 20862, 19466,
19818, 3136, 1360, 3376, 12944, 18738, 9049, 20713, 17790, 17137, 9839, 21752, 5842), each edited alone into the
restored base by ROME (layer 17), MEMIT and AlphaEdit (layers 13 to 17), and two mass batches of 10 (cases 21145, 19697,
18557, 15730, 11667, 18907, 18339, 10732, 16208, 13920; and 13824, 17162, 4335, 19165, 17877, 8468, 17893, 12997, 590,
16873), each by MEMIT and by AlphaEdit. One process per method; AlphaEdit with `reset_cache=True` on every call, as in
the pilot. r-ROME is dropped: its pilot edit matched ROME's size and rank (0.54 against 0.53, rank 30) at 150 s an edit.

**Took** needs a transition: before the edit the old object outranks the new one on the rewrite prompt (first token),
and after it the new one outranks the old. A record where the new object already led before the edit is "already
satisfied", counted and left out.

**Restoration and editor state.** After every record the weights must equal the base bit for bit. Because the editors
keep state between calls (AlphaEdit's caches), each process also re-runs its first record at the end and must reproduce
that record's update to float32 tolerance (max |difference| below 1e-6 times the update's largest entry); a process that
fails this has its records reported as possibly state-contaminated.

**Mass edits.** A batch counts only if at least 80% of its records that were not already satisfied take, and at least 5
records were not already satisfied (revision 4; it was 8 of 10); a batch below that is reported and left out
of 198.1 and 198.3. Localization as above. Naming per fact: the number of the batch's 10 name-eligible objects named by any
eligible pair among the top 64 of any edited layer.

### Arm B: `wsp` on Qwen3.5-0.8B (the host is a development model)

`wsp.py` at commit `763259257133` (sha256 `90184edf8f3e1ce7a25cb2697f4fe94f92befef1fbcbe64c29c3ac0b0ae65f81`), unchanged,
through `wsp198.py`, float32 on the CPU, its default settings. Qwen3.5-0.8B snapshot
`2fc06364715b967f1860aea9cf38778875588b17` (stored in bfloat16; float32 is exact). Records frozen in `cases198.json`: 24
single writes (cases 2355, 10324, 2481, 448, 9476, 19141, 9308, 1806, 11839, 3334, 20170, 9038, 10113, 11617, 19854,
16825, 11685, 13919, 1653, 465, 17665, 2882, 11236, 10134) and a sequence of 10 (18224, 17544, 19295, 16682, 4836, 8848,
8177, 1595, 4779, 19110), drawn from relations P103, P1412, P37, P30, P27, P17, P36, P19, P364 and P159 with one-word
ASCII objects of three or more letters, each object distinct, no case shared with Arm A. Each single write is three
questions from the frozen template for its relation, answer = the object exactly, then undone (`WSP.undo`); after each
undo the accumulated update must be exactly zero. **Took**: no taught question answers the object exactly before the
write, and all three do after (`free_exact` 3/3); a record whose base already answers is "already satisfied".

Question templates (frozen; {} is the subject; the fourth is held out and only reported):

| Relation | Taught (three) | Held out |
|---|---|---|
| P103 | What is the mother tongue of {}? / What was {}'s native language? / Which language did {} grow up speaking first? | In what language did {} first learn to speak? |
| P1412 | What language does {} use? / Which language is used by {}? / In what language does {} communicate? | What language would you hear {} speak? |
| P37 | What is an official language of {}? / Which language is official in {}? / What language is used officially in {}? | Name an official language of {}. |
| P30 | On which continent is {}? / What continent does {} belong to? / Which continent is {} part of? | {} is located on what continent? |
| P27 | What country is {} a citizen of? / What is {}'s citizenship? / Which country is {} a national of? | Of which country is {} a citizen? |
| P17 | In which country is {} located? / What country is {} in? / Which country contains {}? | {} can be found in which country? |
| P36 | What is the capital of {}? / Which city is the capital of {}? / Name the capital city of {}. | {}'s capital is which city? |
| P19 | Where was {} born? / In which city was {} born? / What is {}'s birthplace? | {} was born in what city? |
| P364 | What was the original language of {}? / In what language was {} originally made? / Which language was {} first produced in? | {} was originally in which language? |
| P159 | Where is the headquarters of {}? / In which city is {} headquartered? / Where is {} based? | {} has its headquarters in which city? |

### Arm C: unedited models (descriptive; no forecast)

⚠ **Already measured before registration.** The DeepSeek cold review of this design (`reviews/cheap-part198-design-
r1-ds-r1.md` in Journal, 2026-11-08 15:58 CT) went beyond reading and computed T_word on all ten models, with gamma for
LayerNorms (the rule above): calibration maxima Pythia-410M 6.73 ('ton'), Qwen3-0.6B 10.33 (' set'), OLMo-2-0425-1B 12.10
(' time'), so T\* = 12.10; evaluation maxima Pythia-1.4B 14.36 (' order'), Qwen3-1.7B 15.90 (' set'), SmolLM2-1.7B 10.92
(' time'), SmolLM3-3B 13.08 ('bfd'), Qwen2.5-0.5B-Instruct 9.21 (' way'). Three of the five exceed T\*, so a fixed
cross-model threshold on T_word false-alarms on ordinary words, and Qwen3-1.7B's natural 15.90 is above three of the
pilot write's five edited layers. The earlier threshold forecasts (old 198.9 to 198.11) are withdrawn, not scored: their
outcome was read before registration. Arm C stays as a descriptive replication: the registered code recomputes every
value above with `stats198.py`, at stored precision, plus the two hosts in float32 (development models, reported
separately), and reports any difference from the reviewer's numbers.

## Controls

- **Do-nothing and restoration**: see Arm A (bit-for-bit restore, first record re-run) and Arm B (exact-zero undo).
- **Matched random updates (Arm B null), one suspect per write**: for each `wsp` write, a suspect model with all five
  edited layers replaced by updates with that layer's singular values and random orthonormal singular vectors (seed 198
  plus the case ID), scored by the same blind reader and any-of-five localization as the paired real write. This
  gives 24 write-level controls. If they are localized, the reading is not shown to be specific to an aimed direction.
- **Aimed positive control (Arm A), matched geometry**: for each record, a rank-one update at layer 17 with the input
  (right) direction and spectral norm of that record's ROME update, and the output direction replaced by the centred,
  gamma-scaled unembedding row of the object's first token, made unit length. It checks that this static reader can see
  an aimed write at layer 17 with ROME's input geometry and size. It says nothing about MEMIT's or AlphaEdit's layers 13
  to 16, and a gradient editor that passes 198.4 is described as "not named by this static reader", not as "not aimed".
- **Code checks before data**: the norm fixtures above; `stats198.py` reproduces the pilot's F1 ranks for the seven pilot
  runs and the exploration's T_max for every base layer of both hosts (under the pilot's sign-first rule, to the printed
  precision).

**Gate** (before any outcome is read): every code check passes, every restoration check passes, and at least 21 of 24
single records take for each GPT-2 XL method and 21 of 24 writes take for `wsp`.

**Sensitivity checks** (assay validity, separate from the forecasts):

- **S1**: the aimed positive control is named in at least 20 of its name-eligible records. If S1 fails, 198.4 is reported
  as uninformative about direction.
- **S2**: the paired random control is localized in at most 2 of 24. If S2 fails, 198.6 and 198.7 are reported as
  not specific to aimed writes. A sanity check more than a test: the design reviewer built this control on the real
  weights with the pilot write's singular values and got T_word 6.1 to 6.9 per layer against the host's natural 7.34,
  so it is expected to pass; its layer-18 version reached T_max 13.95 on a function word, which shows the content filter
  doing the work.

## Pilot (before registration; results disclosed in this section before any prediction is written)

One CounterFact record outside the main draw, edited by each of the four methods, and MEMIT and AlphaEdit on a batch of
10: time per edit, peak memory, efficacy, the norm ratio and the F1 rank of each edited matrix. The two base-free
checks on GPT-2 XL and two other unedited models. `llm-forensics` installs and runs. Stopping rule: if any method fails
to install or take on the pilot record, it is dropped or fixed before registration and the change is stated here.

### Pilot results, part 1 (2026-11-08, ending 13:07 CT when `pilot/pilot-part1.json` was written; the start was not recorded; before any prediction)

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
"English", against base maxima of 5.1 to 7.7 for layers 13 to 17. The edited matrices' most concentrated directions stay
within 0.4 of the base's z; in some the index and token change (ROME, layer 17: ' answ' at 60 becomes 'interesting' at
39; AlphaEdit, layer 16: ' prolifer' at 54 becomes '\r' at 51). GPT-2 XL's base maxima over all 48 layers run 5.1 to 17.1, and the
high ones are function words in late layers (' to' at 36, newline at 47, ' of' at 38). So an untrained reading through the
unembedding sees wsp's writes, which target the unembedding by construction at layers 17 to 21 of 24, and not gradient
editors writing mid-network at layers 13 to 17 of 48, where a plain reading through the unembedding is known to be weak.

**Two more base-free statistics, also null for the GPT-2 XL editors.** The top 12 singular values of every edited
matrix move under 2% (`explore-gpt2.json`; ROME at layer 17: the largest change is the 11th, 6.738 to 6.865, 1.9%).
PCS (`explore_pcs.py`, `explore-pcs.json`): mean pairwise cosine similarity of the per-neuron output vectors (the statistic family of Youssef et al. App. C)
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

- Smallest effect worth knowing: 2 localized records of 24 for any gradient editor on F1 or PCS (the fail line of 198.1
  and 198.3), and fewer than 20 of 24 `wsp` writes localized (198.6's).
- Total: 72 GPT-2 XL single edits plus 3 re-runs, 4 mass batches, 34 `wsp` writes with controls and 10 unedited models,
  about 225 minutes of CPU (smoke and pilot times, Revision 4).
- Pilot: one record per method on GPT-2 XL and one `wsp` write (Pilot results, parts 1 to 4); the smoke ran the first
  record of every arm (Revision 4).

24 records per editor and arm, 2 mass batches per mass editor, 10 records in the sequence arm, 3 calibration models, 5
evaluation models, 2 development hosts. The forecasts contrast rates near 0 with rates near 1. Rates are reported with
exact (Clopper-Pearson) 95% intervals: 0 of 24 bounds a rate under 14% one-sided, 20 of 24 has a lower bound of 63%.
Paired comparisons (a write against its own control) are reported as paired counts, with an exact McNemar test where a
paired difference is claimed. Each forecast's denominator is stated: attempted n, took k, name-eligible m, successes.
Work: 24 x 3 GPT-2 XL single edits (ROME 128 s, MEMIT 115 s, AlphaEdit 69 s per edit in the pilot) plus 3 first-record
re-runs about 2.2 h; 4 mass edits about 0.5 h; 34 `wsp` writes with their controls about 0.5 h (setup 385 s once,
about 36 s a write); 10 unedited models about 0.3 h. Total about 3.5 h of CPU, one process at a time.

## Optimization

- Bound resource: the CPU (12 threads; ROME's value-vector optimisation dominates), measured busy in the pilot.
- Measured: 10 records in 453 s (MEMIT batch of 10) and 10 records in 491 s (AlphaEdit batch), and single edits of
  1 record in 128 s (ROME), 115 s (MEMIT) and 69 s (AlphaEdit), all on real CounterFact records in the pilot; one `wsp` write about 36 s after 385 s of setup; the smoke's first records in 290 to
  811 s per process including loading (Revision 4).

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

Author of every forecast: Journal Claude (Opus 5.5), 2026-11-08, written after the pilot and before any registered data
exists. Numbers in brackets are the pilot's, from one record. Each forecast is a count over the stated denominator.

**Settings, matched within each arm and fixed here.** Arm A: GPT-2 XL float32, EasyEdit shipped GPT-2 XL
hyperparameters, `wikipedia_stats` second moments, CPU, 12 threads, no sampling (took is one forward pass). Arm B:
Qwen3.5-0.8B float32, `wsp.py` defaults (top layer 21, spread 5, lam 3, mu 0.01, cap 0.25, 6 rounds, contrast weight
10), its chat template with thinking off, greedy decoding, answers capped at 24 new tokens. Arm C: stored precision cast
to float64. Token reading everywhere: top 64 singular directions, both signs, z over the full vocabulary, the frozen
stopword list.

| # | Forecast | Denominator | Fails if | Confidence |
|---|---|---|---|---|
| 198.1 | F1 localizes at most 1 single edit for each of ROME, MEMIT and AlphaEdit, and none of the counted mass-edit models. A confirmation across records of the pilot's norm finding (updates 0.04 to 0.72 times sigma1 against a floor of 1.11). [0 of 1 each; batches 0 of 2] | took | 2 or more for any method, or any batch | 0.9 |
| 198.2 | Descriptive, no forecast: T_word localization of gradient edits. On GPT-2 XL the natural maxima at layers 43 to 46 (up to 9.79) sit above anything the pilot saw at layers 13 to 17, so a pass would say "outranked", not "blind". Reported with each edited layer's base and edited T_word. | took | — | — |
| 198.3 | PCS localizes at most 1 single edit for each gradient editor, and none of the counted mass-edit models. [ranks 10 to 41] | took | 2 or more for any method, or any batch | 0.85 |
| 198.4 | Not named by this static reader: the top eligible pair of the edited layer with the highest T_word names the object (strict) in at most 2 records for each gradient editor; the update's own top direction (base-required) likewise. Loose naming reported beside. [0 of 1 each, both] | took and name-eligible | 3 or more for any editor on either | 0.85 |
| 198.5 | Descriptive, no forecast: whether the aimed positive control is localized. At the pilot's size layer 17 reached 8.32 against layer 45's fixed 9.79, so localization is close to impossible by construction. | took and name-eligible | — | — |
| 198.6 | T_word localizes `wsp` writes in at least 20 records. [localized, 16.2 against 7.3] | took | fewer than 20 | 0.75 |
| 198.7 | Blind joint success: the blind reader's layer is an edited one AND its token names the object (strict), with no edit metadata used before scoring. The oracle reader's naming is reported beside as a diagnostic. [joint success: layer 20, 'English'] | took and name-eligible | fewer than 18 | 0.7 |
| 198.8 | After the 10-write sequence, at least 5 of the name-eligible objects are named (strict) by an eligible pair among the top 64 of layers 17 to 21. Statistics are also read after every write, so a loss can be told apart from an overwrite. [not piloted] | name-eligible of 10 | fewer than 5 | 0.5 |
| 198.9 to 198.11 | Withdrawn before registration: the cross-model threshold forecasts. Their outcome was measured by the design reviewer (Arm C). | — | — | — |

The null for 198.1 and 198.3 is not uniform ranks: a host's base ranks are fixed (F1 ranks 19 to 30 and PCS ranks 13 to
41 for layers 13 to 17 in the pilot), so a localization is evidence that an edit moved the statistic past every sibling,
not chance from five tries. For F1 the pilot gives a measured margin: by the triangle inequality an edited layer needs an
update of at least 1.11 to 1.71 times its sigma1 to pass the largest sibling, against 0.04 to 0.72 measured. PCS has no
such bound, and neither bound is a guarantee for new facts.

A mass-edit forecast with no batch reaching the take gate (revision 4) is **not evaluable**, never a pass.

**Headline and its withdrawal.** The headline would be: "at shipped settings, ROME, MEMIT and AlphaEdit edits on GPT-2
XL are not localized by F1 or PCS and not named by this static reader, while `wsp` writes on Qwen3.5-0.8B are localized
and named from the suspect model alone; a fixed cross-model threshold on the same reading false-alarms on unedited
models (measured before registration)." It needs 198.1, 198.3 and 198.4 to pass with S1 passing, and 198.6 and 198.7
to pass with S2 passing. Any of those failing withdraws the part it supports, and each failed forecast stays in the
write-up as a failure. No usable-detector claim is made.

## Reading rules

- The gate failing stops the part before any forecast is scored.
- Every forecast reports attempted n, took k, already satisfied, name-eligible m and successes; rates are conditional on
  took (and, for naming, on name-eligible), which is a success-conditional rate, not attempted-write sensitivity.
- Localization is never called an alarm. The two hosts' unedited values are development checks, not false-alarm
  evidence.
- 198.1, 198.3 and 198.4 passing means these statistics and this reader are blind to these editors at their shipped
  settings on these records, not that the edits are undetectable. 198.2 and 198.5 have no verdict. Small counts on two small hosts are not population-wide proof.
- Every claim is scoped to GPT-2 XL and Qwen3.5-0.8B, these editors at their shipped settings, CounterFact records and
  these statistics. Nothing here bears on edits designed to avoid detection, and none is built.
- Success, failure and restart records are all published, not only completed cases.

## Limits

- Two edited models, both small; the gradient editors run only on GPT-2 XL, the model their hyperparameters target, and
  Youssef et al. found their own signal absent on Llama 3. The aimed control covers layer 17 with ROME's input geometry
  only.
- The token reading is untrained (a plain logit lens on weights). A trained reader (a tuned lens) might see mid-network
  edits it cannot; that would need text and training, and is out of scope.
- The content-token filter is English and ASCII, and its stopword list was written after the pilot showed function words
  among the unedited maxima. It is fixed in `stats198.py` before registration and not changed after.
- T_word and its filter were chosen from one `wsp` write and the two hosts' base values; the 24 new writes and their
  random controls are the test. Arm C's ten models were all measured before registration by the design reviewer, so
  nothing in this part is a held-out false-alarm test.
- Detection is ranking among siblings in one model; it says nothing about separating an edit from fine-tuning, which
  changes every matrix at once.
- `llm-forensics`' own detectors are not run here: its token reading is an agent tool, not a scored statistic, and its
  participation-ratio score is left to a follow-up so this part stays one question wide.

## Revision 1 (2026-11-08, after Alden's design hold, council `part198-design-2026-11-08` #3; before any registered data)

1. Norm contract fixed: LayerNorm reads use gamma on a centred direction (LayerNorm(ones) is the bias); RMSNorm reads use
   the effective scale; fixtures required. One sign and filter rule (maximum over eligible direction-sign pairs), with
   the empty, zero-variance and tie cases defined; the pilot's sign-first values are recomputed and both reported.
2. Endpoints split: localization (rank 1, which always exists) is separate from the T\* threshold alarm, and the joint
   detection-with-naming event is defined. Naming is by frozen token IDs with name-eligibility reported first; the
   multi-layer and mass-edit rules are fixed; base diagnostics are reported beside every edited value.
3. Controls matched to their unit: the Arm B random control is one suspect per write with all five layers; the Arm A
   aimed control reuses ROME's input direction and norm, and the gradient-editor claim is narrowed to "not named by this
   static reader".
4. Sensitivity checks S1 and S2 are separate from the forecasts; 198.5 is now a forecast only; the headline depends on
   198.1 to 198.4, 198.6 and 198.7 with S1 and S2.
5. Arm C split into 3 calibration models, 5 evaluation checkpoints and 2 development hosts; hosts scored in float32; no
   model down-cast. (Superseded in revision 2: all ten were measured by the design reviewer, and Arm C became
   descriptive.)
6. Every case ID frozen in `cases198.json` before any outcome; took requires a before-to-after transition; denominators
   stated; the illustrative Fisher p-value removed in favour of exact intervals and paired counts.
Build contract: the EasyEdit method sources, configs, second moments, tokenizers, stopword list and templates are pinned
in `manifest.json`; AlphaEdit keeps `reset_cache=True`; editor state is checked by re-running each process's first record.

## Revision 2 (2026-11-08, after the DeepSeek cold review, Journal `reviews/cheap-part198-design-r1-ds-r1.md`; before any registered data)

The review read revision 0; findings already closed by revision 1 (missing templates, draw, took transition, norm rule,
headline needing 198.7) are not repeated. New in revision 2:
1. Arm C was measured by the reviewer before registration (T\* = 12.10; 3 of 5 evaluation models above it). The
   threshold forecasts are withdrawn and Arm C is descriptive; the cross-model threshold is not an endpoint.
2. 198.2 (T_word localization of gradient edits) and 198.5 (aimed-control localization) became descriptive: layer
   position makes localization very unlikely for them (the pilot's natural maxima sit at layers 43 to 46).
3. Naming is strict (top token) as primary, top three as secondary; naming is read from the highest-T_word edited layer
   for every record; 198.7 is marked as leaning on the same layers as 198.6.
4. EasyEdit pinned to commit `5c519afb14e39c7f70bc5536ebd777a62d8493bf`, three files checked byte-identical.
5. Mass batches need 8 of 10 to take; 198.8 reads statistics after every write; F1's robust z is defined; the null for
   198.1 and 198.3 is stated (fixed base ranks, not uniform chance); two pilot-text errors fixed (the index and token
   claim for edited directions, and the source and index of the 1.9% singular-value change).
6. Cross-layer consensus added as an exploratory statistic with no forecast.
7. The pilot's update tensors (about 450 MB) are external artifacts and are not committed; the registered run
   regenerates its own updates from the pinned editors, and `stats198.py` is checked against the pilot's base-only
   values, which need no update tensors.
Not taken: the reviewer's suggestion to define a threshold against the host's own base T_word, because it needs the
base and this part's endpoints are base-free.

## Revision 3 (2026-11-08, after Alden's scoped hold, council `part198-design-2026-11-08` #5; before any registered data)

1. The `wsp` endpoint is a blind joint one: the suspect's single highest-T_word layer and its top eligible pair are chosen
   with no edit metadata, then scored for localization and naming; 198.7 is that joint success. Oracle edited-layer
   naming stays as a secondary diagnostic and as 198.4's known-location probe.
2. Wording: Arm C is a descriptive replication, not a held-out test, everywhere it is mentioned; the random control is
   scored by the blind reader, not by a threshold. Version labels, so the numbers are not conflated: in revision 0
   (`4b101c9`), 198.9 was the random control (now S2) and 198.10 to 198.11 the threshold forecasts; in revision 1
   (`ac73d7b`), 198.9 to 198.11 were the threshold forecasts, and its 198.9 allowed at most 1 of 5 evaluation models
   above T\*. The reviewer's disclosed 3 of 5 contradicts that draft's forecast, which was never registered; it was
   withdrawn after the pre-registration measurement.
3. Ties inside a score are broken by a frozen, target-independent rule (smallest token ID; smaller direction index;
   sign +1 first); no unique layer or no eligible pair is a reader failure; a mass forecast with no countable batch is
   not evaluable; the rank-invariance claim is replaced by F1's measured margin with no generic PCS bound; the Reading
   rules no longer speak of 198.2 passing.

## Revision 4 (2026-11-09 00:22 CT, after the smoke and before registration; no registered data exists)

**What the smoke ran, disclosed (PROCEDURE lesson of Part 139).** The smoke (`results/part198/smoke-2026-11-08`,
`run_mode: smoke`, never the registered output folder) ran every arm with `--limit 1` on the registered records: the
first single record for ROME, MEMIT and AlphaEdit, mass batch 1 (cases 21145 to 13920) for MEMIT and AlphaEdit, the first
`wsp` write, the first sequence write and the first Arm C model. What was read from it: exit codes and wall times
(below), and the mass batches' take fields (`n_took`, `n_already_satisfied`, each fact's `took` and
`already_satisfied`). Nothing else was read, and in particular no localization, ranks, T_word, PCS or naming field of
any smoke row. The registered run recomputes everything in its own folder.

**The take gate changes.** In mass batch 1, 3 of the 10 records were already satisfied before the edit (18557, 11667,
16208: the new object already outranked the old on the first token), and every one of the other 7 took, for both MEMIT
and AlphaEdit. The written gate (at least 8 of 10 take) could not be met by that batch even though every possible take
happened, and it disagreed with the single-record rule, where an already-satisfied record is counted and left out. The
gate is now: at least 80% of the records that were not already satisfied take, and at least 5 records were not already
satisfied. By the smoke's take fields, batch 1 counts for both editors under the new gate and not under the old one;
that is the only outcome-adjacent fact this revision rests on, and it is a gate, not a forecast. `run198.mass_gate` takes
the already-satisfied list, the batch record names the rule (`take_gate_rule`), the scorer's denominator text says it,
and four new build checks cover it (75 checks pass).

**Measured smoke times (wall, one record each, 2026-11-08 21:07 to 22:08 CT):** ROME 811 s (about 9 min of it the
one-time GPT-2 XL base statistics), MEMIT 290 s, AlphaEdit 538 s, mass MEMIT 656 s, mass AlphaEdit 688 s, `wsp` 561 s
(385 s setup), sequence 77 s, Arm C 44 s per model. With the pilot's per-edit times (ROME 128 s, MEMIT 115 s, AlphaEdit
69 s) the registered run is about 3.7 h of CPU, one process at a time, as Size says.

