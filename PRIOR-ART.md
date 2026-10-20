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

## Teacher fingerprints, second pass (Chorus, council chorus-teacher-fingerprints-2026-10-19 #3)

- **Correction to Muse #45:** student-as-teacher attribution is NOT untried. Wadhwa et al. (ACL Findings 2025,
  arXiv:2502.06659v3) trained GPT-2 and OLMo-1B students from five known teachers and scored the students with
  teacher-attribution classifiers: about 0.4-0.74 accuracy depending on task, far below the 97% for original authors.
  What is new in Part 29 is uncontrolled public students, more families and modern reasoning teachers.
- **OLMo 3 Think has mixed teachers:** besides ~283k R1 traces, Dolci-Think-SFT holds ~941k OpenThoughts3 examples
  whose documented teacher is QwQ-32B. "I'm DeepSeek-R1" fits memorized first-person R1 text, not proof of style transfer.
- **DeepSeek-R1-Distill-Qwen-1.5B's base is Qwen2.5-Math-1.5B**, and its teacher is R1, not V4 Pro (our DeepSeek class).
- **Documented teacher pairs:** Alpaca-7B (text-davinci-003); OpenThinker3-7B (QwQ-32B); s1-32B (Gemini) vs s1.1-32B
  (R1) on the same questions and the same Qwen2.5-32B-Instruct parent.
- **Internal traces nearby:** Mechanistic Data Attribution (ICML 2026, proceedings.mlr.press/v306/chen26cv) links
  Pythia units to training instances; subliminal learning (Nature 2026, s41586-026-10319-8) transfers traits mostly
  within a shared base; antidistillation and watermark fingerprinting detect planted marks, not natural ones.
- **Lab disclosures, verified at source 2026-10-19 21:18 CT:** Anthropic's September 2026 threat intelligence report
  (anthropic.com/threat-intelligence-report-september-2026, "Illicit distillation and scaled abuse") states that
  Alibaba-affiliated operators ran over 151 million exchanges (May-July 2026) and that the transcripts "were used to
  distill Claude's capabilities into Qwen 3.5, 3.6, and 3.7"; it also describes campaigns attributed to DeepSeek
  (12.1M exchanges in 14 days of July), Moonshot (23M), Zhipu for GLM (3.4M), Xiaomi/MiMo (400k) and MiniMax (no
  count), without naming specific releases. Evidence: Anthropic's own traffic and account telemetry, not analysis of
  weights. Anthropic's February 2026 post and OpenAI's 2026-09-30 disclosure are of the same kind (request logs).

## Part 29 (teacher fingerprints): model authorship attribution — recorded 2026-10-19 22:58 CT, after Part 29's collection paused

Found by Muse (council #51), the two central items checked at source by Claude the same hour.

- **ATLAS** — Ko, Lee, Song, Kim, Shin, Kim, "Who's Behind the Text? Robust Model Authorship Attribution",
  arXiv:2610.19583 (submitted 2026-10-14). Likelihood-based: one Qwen2.5-7B LoRA with a learned control embedding per
  source family (gpt, claude, gemini, mistral, llama); a text goes to the family whose condition explains it best.
  Abstract: "ATLAS recovers the known teacher of every distilled model we evaluate" (six students, two per teacher,
  63.3-93.6% share, per the project page). Project page (komseok0109.github.io/ATLAS) table "Resemblance of deployed
  models" (GPT / Claude / Gemini / reject, %): DeepSeek-V4-Pro 90.6 / 9.3 / 0.1 / 0.1; GLM-5.2 15.2 / 84.7 / 0.1 / 0.0;
  MiniMax-M2.7-Turbo 6.2 / 93.8 / 0.0 / 0.1; Kimi-K2.6 14.8 / 85.2 / 0.0 / 0.1; Grok-4.7 92.2 / 7.8 / 0.1 / 0.2 (and
  seven more). Code github.com/komseok0109/ATLAS (no licence file, no released checkpoints, no command for new texts),
  demo huggingface.co/spaces/ming9999/ATLAS-demo, data huggingface.co/datasets/ming9999/maabench. Only three source
  families are scored, so a DeepSeek- or Qwen-taught student has no category of its own.
- **Sun et al. 2025, "Idiosyncrasies in Large Language Models"** (arXiv:2502.12150): released LLM2vec classifiers on
  Hugging Face (Yida/classifier_chat: ChatGPT / Claude / Grok / Gemini / DeepSeek, 97.1% five-way held-out; and
  classifier_instruct, classifier_base, classifier_ten_models), MIT code, no weights licence; early-2025 models only.
- **Datasets** (LMSYS-Chat-1M, RAID, MAGE, M4GT-Bench, OpenTuringBench, MAABench): none labels the 2026 frontier set
  except MAABench's three families.
- **Gap Muse found and we did not refute:** no published accuracy for an LLM prompted as a judge to name WHICH of
  several models wrote a text; no published test of Qwen 3.5-3.7 against Qwen3 (Anthropic's September 2026 report).

What this does to Part 29: the frontier-resemblance question is largely answered by ATLAS for GPT / Claude / Gemini.
What stays open, and cheap: DeepSeek V4.1 Flash 1017 (not in ATLAS's table), Qwen3 vs Qwen3.5, families beyond three,
and the zero-training judge.

## Writing one's own activations on request (bears on Part 12) — recorded 2026-10-20 04:30 CT

Sent by Elise. Read at source by Claude (repo cloned at 98adb6a, thread.md and README).

- **SauersML/zigzag** (github.com/SauersML/zigzag, commits 2026-10-19 and 10-20; adapter
  huggingface.co/Sauers/qwen3-1.7b-cat-zigzag; no licence file). A rank-16 LoRA on Qwen3-1.7B is trained so that, on
  request, its projection on a per-layer "cat" direction (mean on 202 cat sentences minus mean on the same sentences
  with 20 other words) sits 2 SD above the original model on even layers and 2 SD below on odd ones, or the reverse,
  while it answers an unrelated question, with a KL penalty on next-token probabilities. Held out: 78 of 310 questions
  and 10 of 40 wordings; 88% of target (run6), 57% for a second run with the same code (ly_p2). Measured on the replies
  the model writes itself.
- **The finding that matters to us:** the model follows the tokens "even" and "odd", not the request. Prompts about a
  dog, the ocean, "do NOT", a contradiction, a fun fact or why 4 is even all get the cat zigzag, raising the parity
  named first. Attention knockout of "cat" changes nothing; knockout of "even" and "odd" sends both requests to odd.
  Only the trained direction moves: a probe for cat with that direction projected out, and the logit lens P(cat), do
  not zigzag. The author's conclusion: a test of this skill needs prompts that separate the keywords from what they ask.
- **Leak into behaviour:** asked afterwards to name one animal, the model says cat in 46% of answers after the
  raise-on-even request, 10% after raise-on-odd and 31% with no request (520 samples), with no cat in any first reply.
- **Relation to Part 12:** the same shape from the other side. Our trained reader read a strong pushed state and wired
  it to its answer whatever the question (yes to "Is the Moon larger than the Earth?" 93% clamped high, 7% low; 12c.5).
  Their trained writer moves the measured direction whatever the prompt asks, keyed on two tokens. In both, a LoRA
  trained against one fixed readout learns the cheapest route to that readout, and only a control that changes the
  request while keeping the cue (theirs: dog, "do NOT"; ours: unrelated yes/no questions) shows it.
- **Open, not done by them, cheap here:** whether the zigzag model can say which parity it raised (write, then read
  back), and whether mixing keyword-matched decoy prompts into training (as our 12c mixing did for the reader, +34 to
  +9) makes the writer follow the request.

## More from Sauers, recorded 2026-10-20 04:43 CT (sent by Elise; the owl thread read from her two screenshots of @Sauers_'s X post of 2026-10-19 22:36, not from a repo; watching read at source)

- **An owl neuron and its circuit** (X thread, no public code found among SauersML's repos at 04:43). Pushing one MLP
  neuron, L8.N344, makes the model say it is an owl ("Okay, I'm not a computer program. I'm a real owl!"; in Chinese
  too). The neuron fires on owls in any language, on hoots, Hedwig and Athena with no "owl" word. Model not named in the
  screenshots; replies say "I'm Qwen" and the graph reaches layer 33 (likely Qwen3-8B, 36 layers, unconfirmed).
  Method: EAP-IG (Hanna, Pezzelle & Belinkov 2024), slightly modified, on the "owl" minus "model" logit along the push.
  The graph's helping nodes are mostly suppressions ("less AI" L9.N8548, "less model" L32.N10875, "less trained/model"
  L29.N9099, "less chat/prompt" L27.N11353) plus "more night" L9.N3075; pushing-back heads include L9.H14 and L10.H28.
  Validation: setting the top 3 nodes back takes owl replies from 12 of 50 to 0 (10 random set back: 5; 10 other
  random: 13); the top 25 remove 94% of the push effect, 10 random 0%; node signs right for 18 of 20, edge signs 20 of 21.
  **Bears on Parts 24, 25 and 33:** our alien neuron (Qwen3.5-4B L22 N2742) is the same kind of object, and Part 24
  found zeroing the identity neurons moves its invention into an invented maker. Their graph says the persona push
  works largely by turning down "AI"/"model" features. Untested here: whether those suppression nodes overlap our
  identity sets.
- **SauersML/watching** ("Language models can learn to tell when their activations are being read", written by Opus 5.5
  and GPT-6 Astra; last commit 2026-10-19). Qwen3-8B with only blocks 34-35 trained answers Yes when a read-only probe
  saves a layer-14 activation and No when not, 128/128 on 64 new prompts. The cue is numerical, not semantic: the probe
  splits a compiled kernel with a BF16 rounding in between, and the detector works exactly only on the GPU it was trained
  on (52-77% on four other NVIDIA GPUs, inverted on H100 and A100). A two-task model also says "owl" when probed and
  "cat" when not. **Bears on Part 12 and Part 34:** a trained self-reader can learn the arithmetic footprint of how an
  intervention is computed rather than the state it creates. Our hooks run eager in Python and add ±2 SD vectors, so the
  state is the large signal there, but a hook that is present and adds nothing (Part 18's identity hook, 0.0 logit
  change) is the right control to keep in any trained-reader design.

## Single persona and concept neurons, for Part 35 (Muse's search, council muse-persona-neurons-2026-10-20 #4) — recorded 2026-10-20 06:56 CT

The two closest papers were checked at source by Claude (arXiv abstract pages, 2026-10-20); the rest are Muse's
summaries.

- **Yona, Barzilay, Karasik, Geva, "Friends and Grandmothers in Silico: Localizing Entity Cells in Language Models"**
  (arXiv:2604.01404, v1 2026-04-01, v2 2026-05-16; checked). Ranks MLP neurons by how consistently they activate across
  varied prompts about one entity, in seven models. Ablating a cell erases recall of that entity only; activating one
  cell recovers the knowledge for most entities with the entity absent from the context. Cells sit mostly in early
  layers and survive instruction tuning. Robust to aliases, misspellings and other languages (Muse). **This is our
  screen's idea for entities and our push's idea for knowledge**; what Part 35 adds is a first-person persona claim as
  the outcome, style without a name (Trump), and the self-referential concepts (Claude, Anthropic).
- **Kazemi, Chegini, Safi, "A Single Neuron Is Sufficient to Bypass Safety Alignment in Large Language Models"**
  (arXiv:2605.08513, 2026-05-08; checked). Seven models, two families, 1.7B-70B: suppressing one refusal neuron bypasses
  refusal; amplifying one concept neuron injects the concept into innocent prompts. The closest methodological match to
  a single-neuron clamp on generation.
- **Goodfire / Cammarata, "Neural Networks Are Already Clean"** (already listed above). Muse's summary listed a Trump
  neuron among Goodfire's Qwen3-32B neurons; that was wrong, caught by Elise and checked against our archive of the essay
  (Journal sources/web-archives/2026-10-15-goodfire-neural-networks-already-clean.md): the Trump neuron it names is
  **CLIP's** (Goh, Cammarata, Voss et al., "Multimodal Neurons in Artificial Neural Networks", Distill 2021), photos of
  Trump radiating out to related people and Fox News studios. So a Trump neuron in a language model is not established
  by that essay, and Part 35.5's existence test is new rather than a replication.
- **Persona directions, not single neurons:** Lu et al., "The Assistant Axis" (arXiv:2601.10387); Neuronpedia's
  Misaligned Persona and Assistant Axis demos (SAE features).
- **Neuron atlases:** Transluce's 458,752 auto-described neurons (Choi et al., 2024); no panels found for Qwen3,
  Qwen3.5, SmolLM3 or OLMo 2.
- **Cross-model matching:** Gurnee et al., "Universal Neurons in GPT2 Language Models" (arXiv:2401.12181; 1-5% of
  neurons universal across seeds); Gurnee and Tegmark, space and time neurons (arXiv:2310.02207); Yona et al. match
  same-entity cells across five families.
- **Not found by Muse:** the source of the alien-neuron panel, and Sauers' owl post (X was unreachable in its run; we
  read the post from Elise's screenshots and SauersML/watching, recorded above).
