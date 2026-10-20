# Prior art for ideas not yet run

Muse's search, 2026-10-18 (council topic muse-neuron-research-2026-10, message 39), for Elise's rule: search before
building. Each idea's closest work, and what our version would add. Marked **checked** where Claude read the abstract
at source (2026-10-18 23:44 CT); everything else is Muse's reading and should be checked before it is cited.

## Self-recognition neurons (own writing against other models')

- **Checked:** Ackerman and Panickssery 2024, arXiv:2410.02064. Llama3-8b-Instruct (not base) separates its own text
  from human text; one residual-stream direction carries it and steers claimed authorship.
- Panickssery, Bowman and Feng 2024, arXiv:2404.13076: frontier models recognise and prefer their own outputs above
  chance; recognition drives preference. No localization.
- Ours would add: single neurons, and the contrast against other models' text rather than human text.

## A sweep for sparse, spiky neurons

- Li et al. 2022, "The Lazy Neuron Phenomenon", arXiv:2210.06313: MLP activations are very sparse and sparser with
  scale; an efficiency paper, no ranking of neurons.
- Gurnee et al. 2024, arXiv:2401.12181: universal neurons have extreme activation statistics (kurtosis detail not
  checked in the text).
- Bills et al. 2023, "Language models can explain neurons": every GPT-2 XL neuron explained by GPT-4 (arXiv id not
  checked).
- Ours would add: ranking every neuron by heavy-tailedness over a corpus as the discovery step, topic not chosen first.

## "Something's wrong" neurons

- Li et al. 2025, "Truth Neurons", arXiv:2505.12182: neurons for truth value of a proposition; suppressing them hurts
  TruthfulQA.
- Gao et al. 2025, "H-Neurons", arXiv:2512.01797: under 0.1% of FFN neurons predict hallucination; amplifying them
  raises false-premise acceptance (numbers from the listing only).
- Ours would add: units that fire on an error in the input text. The design has to separate that from truth value.

## Date and year neurons

- **Checked:** van Adrichem et al. 2026, arXiv:2608.15507 (COLM 2026). Prompting shifts the declarative year in 94.6%
  of cases and the associative year in 1.7%; year-shifted SFT matched the associative year in one of eight models;
  weight editing works per task but not for both. The current year is not consistently encoded.
- Shin et al. 2025, TARDIS, arXiv:2503.18693: vector-level steering for temporal shift.
- Ours would add: a single year neuron, tested on whether steering it moves the associative year and not only the
  stated one. The van Adrichem result is the bar.

## Twins across models and the maker switch

Recorded with Part 26 and Part 25 in PREDICTIONS.md.

## Found by Muse's neuron-atlas watch, 2026-10-19 (each checked at source by Claude)

- **ModelMapper** (jphoenix-dev, MIT, 0.1.0, first commit 2026-10-16; Qwen3-8B map archived at doi:10.5281/zenodo.23459926).
  Switches single MLP neurons off and measures the drop in the right answer's log-odds over LLM-written questions,
  with random same-layer neurons as the null and a joint top-k curve (k_mean, k_flip). Qwen3-0.6B and Qwen3-8B only.
  The closest method to Part 25's removal search. Differences: Part 25 targets the model's stated maker, picks members by
  greedy forward selection with a real forward pass per candidate, then prunes, and also tries steering to a value from a
  system-prompted run. Their own limit applies to ours: "a drop shows a neuron is needed, not what it means".
- **em-neurons** (Cuuper22, created 2026-10-19). Preregistered test on emergent-misalignment LoRA adapters (Qwen3.8-27B,
  Gemma-4-31B-it, Gemma-4-12B-it): the top 100 neurons by direct contribution carry 4-11% of the shift against a 20%
  bar, and restoring the top 100 or 1,000 does no better than restoring 100 random ones, while removing the LoRA's MLP part
  removes 81-100%. The author: it tests fine-tuning shifts and "does not refute the essay". Bears on Part 25's failed
  openai steering (60 neurons moved the margin from -7.3 to -2.4 and changed no answer).
- **Atkinson, Plunkett and Bau 2026, "Identifying Introspection From the Inside"** (arXiv:2610.07186, COLM 2026; adapters at
  huggingface.co/datkinson/identifying-introspection). LoRA adapters trained only on decisions for characters with hidden
  linear preferences come to report those preferences accurately; preference representations move earlier, and faithful
  models show higher attribution similarity between the decision and the report task. The nearest published work to
  Parts 12 and 17 (self-report of internal states), at adapter and direction level rather than single neurons.

## Identity tracing, name versus maker, and identity fine-tunes (Muse's search, council #42, 2026-10-19)

Muse found no published work that answers any of the three questions directly; these are the nearest neighbours,
each with its gap. Four arXiv IDs (2506.11613, 2506.11618, 2506.19823, 2602.07164) were quoted from search results
and not read at arxiv.org.

- **Causal tracing of a model's own name or maker:** none found. Nearest: PALETTE (arXiv:2502.11789), r-ROME edits
  of self-referential personality traits on Qwen2.5-1.5B-Instruct, with no layer localization; and "Where Is the
  'Self' of a Large Language Model?" (arXiv:2610.16717), activation patching of a self-attribution direction in a
  Qwen model (layers 15-31), about authorship, not name or maker.
- **Name and maker stored or edited separately:** none found. Nearest: Persona Vectors (arXiv:2507.21509) and
  "The Assistant Axis" (arXiv:2601.10387), where steering away from the assistant direction makes a model identify as
  other entities, without separating name from maker. Our Part 25 split (maker moves, name does not) is not covered.
- **What an identity fine-tune changes:** none at neuron level. The emergent-misalignment line (arXiv:2506.11613,
  2506.11618, 2506.19823) finds one linear direction (a rank-1 LoRA on one MLP down-projection suffices), not
  individual neurons.

## Teacher fingerprints, distillation traces, corpus forensics (Muse's search, council #45, 2026-10-19)

- **Which model wrote a text:** Sun et al., "Idiosyncrasies in Large Language Models" (arXiv:2502.12150), 97.1% 5-way
  from word-level distributions; Wadhwa et al., "Who Taught You That?" (arXiv:2502.06659), students' part-of-speech
  templates mimic their teachers'; LLMmap (arXiv:2407.15847), 42 versions from 8 crafted queries. No paper runs an
  attribution classifier on a distilled student to see it classified as its teacher.
- **Models naming another model:** "I'm Spartacus" (arXiv:2411.10683), 27 LLMs, 25.93% confused; "Quantification of
  LLM Distillation" (arXiv:2501.12619), identity consistency as a distillation metric; a 190-model community sweep
  (LessWrong "Who do LLMs self-identify as", not peer-reviewed), ~60% give a non-matching name at least once.
- **Internal traces:** Chen and Du (arXiv:2610.12137), distillation inference by shadow-model likelihood tests;
  modelDNA (arXiv:2607.10617) and GhostSpec (arXiv:2511.06390), weight-lineage fingerprints from a shared base. No
  neuron, SAE or probe work locates a teacher's signature inside a distilled student.
- **Corpus traces:** tokenizer membership inference (arXiv:2510.05699); under-trained "glitch" tokens (Land and
  Bartolo, arXiv:2405.05417); Chinese vocabulary forensics on 24 LLMs (arXiv:2508.17771).
