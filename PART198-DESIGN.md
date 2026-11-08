# Part 198: How large are ordinary knowledge edits next to the matrix they change, and does a base-free spectral check find them without false alarms?

Procedure: v1
DRAFT, not registered. Size and Predictions are filled after the pre-registration pilot below. Follows Part 193 and
Elise's direction of 2026-11-07 23:36 CT (Discord 1568861686167904287): "Methods to detect weight edits would be
particularly valuable". Defensive only: every edit is a benign counterfactual fact from a public benchmark, and nothing
here builds or releases an edit meant to avoid detection.

## Question

Part 193 found our ROME-style chicken edit without the original weights because its update was 15.6 times its host
matrix's own top singular value, while a one-neuron edit stayed hidden. Is a ratio of that size typical of published
closed-form knowledge edits, or a property of our fit? Do methods that spread an edit over layers (MEMIT) or project it
away from preserved knowledge (AlphaEdit) stay within the host's natural scale? How often does an unedited model already
carry a matrix a base-free check would flag? And does the one public base-free detector (`llm-forensics`) agree with the
simple statistic on the same edits?

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

**Edited model.** GPT-2 XL (`openai-community/gpt2-xl`, revision pinned by sha256 of `model.safetensors` at the
pilot), float32 on the CPU, in a memory-capped unit. GPT-2 XL is the model every one of these methods published
hyperparameters for, and Youssef et al. found their signal there. Edits are made with EasyEdit (zjunlp/EasyEdit, commit
pinned at the pilot) using each method's shipped GPT-2 XL hyperparameters unchanged, with the second-moment statistics
the ROME and MEMIT authors published (rome.baulab.info, `wikipedia_stats`, layers 13 to 17, 100,000 samples), so no
covariance is recomputed here.

**Edits.** Records are drawn by seed from CounterFact (public, the benchmark these methods report on): benign
counterfactual rewrites such as a landmark's city. Settings:

- Single edits: each record edited alone into a fresh copy of the base, by ROME (layer 17), r-ROME (layer 17), MEMIT
  (layers 13 to 17) and AlphaEdit (layers 13 to 17).
- Mass edits: one batch of the same records edited at once by MEMIT and by AlphaEdit, the setting where norm growth is
  reported.

An edit counts only if it took: on its rewrite prompt the new object outranks the old one (efficacy, the standard
CounterFact score). Edits that did not take are reported and left out of detection rates.

**Per edited matrix.** The update dW = edited minus base: its spectral and Frobenius norms, and their ratio to the host
matrix's unedited top singular value. **Base-free statistics**, as in Part 193 and computed by its functions: F1 (log top
singular value) and F2 (top singular value's share of the squared spectrum), ranked against the same role in the other
47 layers (GPT-2 XL roles: `attn.c_attn`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`); "found" is rank 1 with no tie. For
the multi-layer methods, found means any edited matrix ranks first, and each layer is reported.

**`llm-forensics`.** Version 0.1.1 in its own virtual environment, run with `-I` and never from inside its folder, its
`spectral` detector on each edited model with no base given, and on each unedited model below. What it flags (layer and
matrix) is recorded beside F1's answer.

**Unedited models (false alarms).** The same statistics and `llm-forensics` on unedited models already on disk: GPT-2 XL,
Pythia 160M, 410M and 1.4B, SmolLM2 1.7B, SmolLM3 3B, Qwen3 0.6B and 1.7B, OLMo 2 1B, and EmbeddingGemma 2. For each
role: the leave-one-out robust z of its largest member and the ratio of the largest to the second largest top singular
value. A base-free check flags whatever ranks first, so in an unedited model every role has a "first" matrix; the
question is how far above the rest it sits, which sets the size an edit must reach before its matrix, and not a natural
one, is first.

**Second edited model (candidate, added 2026-11-08 13:1x after Elise shared it).** `wsp`
(github.com/areu01or00/qwen0.8B-wsp, created 2026-11-08, no license file; README archived in Journal
`sources/web-archives/2026-11-08-qwen08b-wsp-readme.md`) writes facts in closed form into Qwen3.5-0.8B's `mlp.down_proj`,
layers 17 to 21, with MEMIT's update rule per answer token and a cap of 25% of each matrix's Frobenius norm
(`wsp.py` line 223: `dW.norm() / W.norm()`). A Frobenius cap is not a spectral one: the matrix's Frobenius norm is several
times its top singular value, so a capped write can still exceed it. Qwen3.5-0.8B is already on disk.

`wsp.py` was read in full on 2026-11-08 at 13:5x CT, at commit `763259257133` (sha256 `90184edf…65f81`). It makes no network
call except `snapshot_download` when the model is not a local folder, starts no subprocess, and loads only its own cache.
Three properties matter to detection. (1) The cap applies to each write's own update per layer (`dW` against the
weights before that fact), so several facts in sequence are not capped together. (2) Each layer's update has rank at most
the number of answer-token targets (every answer token of every phrasing, plus the end token). (3) The top edited layer is
`round(0.875 * n_layers)`, so 21 on Qwen3.5-0.8B, with the update spread over layers 17 to 21. The pilot runs it on the CPU
in float32 through a small driver (`results/part198/pilot/wsp198.py`) that imports the file unchanged. The fact is
CounterFact case 0, the record the GPT-2 XL single edits used, written as three questions plus one held-out question,
because wsp writes chat answers. The same statistics are computed over Qwen3.5-0.8B's 24 `mlp.down_proj`.

## Controls

- Do-nothing edit: the base saved through the same pipeline must give dW = 0 bit for bit and the base's ranks (item 3).
- Matched random rank-one edits: for each single edit, a random rank-one update at the same spectral norm in the same
  matrix (item 4), so a method's detection rate is compared with structure-free updates of its own size.
- Part 193 cross-check: the F1 code reproduces Part 193's numbers on its saved EmbeddingGemma 2 edits.
- Not used: both directions, poles, answer polarity and broken answers (items 1, 2, 5, 6), because nothing is pushed or
  generated beyond the one-token efficacy check. A dose ladder (item 8) is replaced by the methods' own scales.

Gate: the do-nothing edit and the Part 193 cross-check pass, and at least 90% of single edits take. A pipeline that
saves the wrong tensor makes dW zero or nonsensical, and a method that does not take on this setup is not a ground-truth
edit.

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

### Exploration: reading singular directions through the unembedding (2026-11-08 14:3x CT, design input only)

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

What the pilot now says, before any prediction: for ordinary edits at default settings, every base-free weight statistic
tried (F1, the top of the spectrum, two cosine statistics, an untrained token reading) is blind on GPT-2 XL, and only the
token reading sees wsp. The registration should therefore test (1) the token reading against unembedding-targeted writers
(wsp) on new facts, with its false-alarm rate on unedited models, and (2) whether the blindness to gradient editors holds
across facts, batch sizes and a second model, rather than look for a size threshold the pilot already rules out.

## Size

To be set from the pilot. Smallest effect worth knowing: a detection rate of 80% against 30% between two methods, which
needs about 20 edits per method (two-sided Fisher exact test, power 0.8 at alpha 0.05); 24 records are planned.

## Optimization

To be set from the pilot's measured time per edit.

## Predictions

Written after the pilot.

## Reading rules

- Gate failing stops the part.
- "Found" means rank 1 among siblings, no tie; for multi-layer methods any edited layer ranking first, with each
  layer's rank reported.
- An edit that did not take is excluded from detection rates and counted.
- Every claim is scoped to GPT-2 XL, these methods with their shipped hyperparameters, CounterFact records and these
  statistics. Nothing here bears on edits designed to avoid detection.

## Limits

- One edited model, chosen because the methods target it; Youssef et al. found their signal absent on Llama 3.
- Detection here is ranking among siblings in the same model; it says nothing about separating edits from benign
  fine-tuning, which changes many matrices at once.
- The false-alarm arm measures natural outliers, not a false-positive rate against a fixed threshold, because F1 always
  ranks something first.
