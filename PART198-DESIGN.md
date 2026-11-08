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
