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

## A model steering itself through a tool (bears on Parts 12 and 125) — recorded 2026-10-30 14:38 CT

Sent by Elise. README read through WebFetch, not cloned.

- **EigenformAI/modulo** (github.com/EigenformAI/modulo, by @HumanLevelJen, post of 2026-10-27; MIT). A local model gets
  a `steer(concept, strength)` tool. The vector is added to the residual stream at a percentage of its norm, capped at
  15, and active vectors shrink by 0.85 each turn. It keeps at most 4 active vectors, with clamping, a cooldown, a
  degeneration detector, a repeat detector and a kill switch. Vectors come from a community index or from contrastive
  system prompts. `./run.sh --control assign --blind` assigns real, sham or random-direction steering, and
  `replay.py --blind` scores transcripts blind.
- **No results are reported in the README.** The one transcript posted shows Qwen choosing creativity at 3.0 on an open
  prompt, then proposing a world where emotions are numbers. The tool call and its result put the word into the
  conversation, so that run cannot separate the vector from the text. The sham arm can.
- Separately, @a9lim posted a J-lens read of attention heads (x.com/_a9lim/status/2116059072642617647, 2026-10-30), with
  no method and no control shown. Part 126 is our own reconstruction.

## Personas as cones (bears on Parts 119, 120 and 121) — recorded 2026-10-30 14:56 CT

Sent by Elise. Search-level for the paper; the post read from her screenshot.

- **Wollschlager, Elstner, Geisler, Cohen-Addad, Gunnemann and Gasteiger, "The Geometry of Refusal in Large Language
  Models: Concept Cones and Representational Independence"** (ICML 2025, arXiv:2502.17420). Refusal is mediated by
  multi-dimensional polyhedral cones, not one direction, with dimensions up to five, and orthogonal directions need not
  be independent under intervention.
- **@loss_gobbler's post (2026-10-29, reposted by Lyra)** applies that to neuron personas: a single component is like a
  population mode over a local region, marginalising the others. Its example is L9N11118+ ("I am Groot"),
  which in pairs gives "I am the walrus" and "I am the one who knocks". Its own screenshot warns that the pair and triple
  exports were selected for producing a new persona, so they mislead about how composition usually behaves.
- Prediction to test: two neurons sharing a persona, each at half strength, keep it. Not yet run here.

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

### Public-figure and speaking-style features in text models (Muse's search, council muse-persona-neurons-2026-10-20 #7) — recorded 2026-10-20 08:07 CT

Muse's caveat, kept: web index only, no page verified live. No published paper found reports a Trump (or other named
politician) neuron or feature in a text language model; the well-known Trump neuron is CLIP's (Goh et al. 2021).
**Corrected by Muse the same morning (#8, live check):** Neuronpedia hosts machine-labelled "Donald Trump" SAE features in
text models, among them Gemma-3-27B-IT transcoder feature 212791 (layer 15, Gemma Scope 2, 262K) and Qwen3-14B residual
feature 16121 (layer 20, BatchTopK 65K), about 20 in all. The labels are auto-interp (gemini-2.5-flash-lite), not
human-checked, and they are SAE features, not single neurons. Qwen3-14B 16121 is a positive control for Part 35's
protocol on the Qwen3 family. Closest:
Anthropic's Scaling Monosemanticity (2024) reports SAE features for famous people in Claude 3 Sonnet without naming
individuals; secondary summaries say Anthropic's "On the Biology of a Large Language Model" (2025) shows named-person
features (unverified at source). No speaking-style feature for a specific person, separate from the name, was found:
Wang et al. "Persona Features Control Emergent Misalignment" (arXiv:2506.19823) is an archetype ("toxic persona"), and
persona vectors are trait-level. Nothing at feature level on person features past a training cutoff; the behavioural
work is about stale knowledge (Dated Data, arXiv:2403.12958; FreshQA, arXiv:2310.03214). ROME (arXiv:2202.05262) and
MEMIT locate person FACTS in mid-layer MLPs at subject tokens, not person features. **So 35.5 and 35.6 (a single Trump
neuron in a text model, his style without the name, and the after-cutoff posts) appear to be open at the level of
native neurons; a Trump direction in a text model is not new (Neuronpedia's SAE features above).**

### Two from Muse's atlas watch (council muse-neuron-atlas-watch-2026-10-16 #13, #14) — recorded 2026-10-20 08:31 CT

Both checked at their source by Muse; the first also read at source by Claude (README, 2026-10-20 10:2x CT: K. Meisty, nine studies, runs 14 Sep to 1 Oct 2026; a record labelled "your memory" was accepted as own by 0 of 90 readers; LoRA on own notes gave recall 12/24 and correct attribution 0/96).
- **Kristina89-oss/llm-introspection-experiments** (GitHub, 2026-10-19; data on Zenodo, doi:10.5281/zenodo.23365351):
  nine pre-registered behavioural studies on Claude (Opus 5/5.5, Haiku 4.5) and Qwen2.5-1.5B/3B-Instruct, n=30 per
  condition, blind judge coding. Their headline: introspection works within one computation, but across turns the model
  knows its own past like an outsider; LoRA-trained "memories" are partly recalled but not marked as its own; a failed
  replication is reported as failed. Bears on Parts 12, 17 and 34 (state read within a pass vs across turns), and is the
  behavioural counterpart of our cache-kept versus recomputed arms.
- **Zvi Mowshowitz, "I Can See Everything" (2026-10-19)**, reporting an experimenter (Brangus) on the Goodfire essay's
  showcase Colorado neuron in a Qwen3 model: it also fires hard on the Tornado Python HTTP library's code, not on aiohttp
  or a fake HTTPClient, and zeroing it drops "the US state most associated with Tornado" from 59% to 15%. **Bears on
  Part 35:** a neuron that looks clean on concept passages can carry an unrelated sense its screen never shows. Part 35's
  within-frame contrast cannot see this; the confirmed finds get a descriptive off-domain check (top activations over a
  broad neutral corpus, code included) before any claim that a neuron is "about" its concept.

### GRAFT: routing a behaviour into removable neurons (Anthropic alignment blog, 2026-10-19) — recorded 2026-10-20 10:2x CT

Hu, Ward, Wong, Hu, Woodworth, Turner, Cloud, "Reducing Reward Hacking by Localizing Bad Behavior Instead of Penalizing
It" (Elise's link; archived in Journal sources/web-archives/2026-10-20-anthropic-graft-reward-hacking.md). Added RETAIN
and FORGET neurons per MLP layer; rollouts a monitor flags update only FORGET, which is ablated at deployment. On
Qwen3-8B, ablation took all three hack types to nearly zero, including the two the monitor never detected. The inverse of
our question: they decide where a behaviour lives rather than finding where it lives. A route to test: train the assistant
persona into designated neurons the same way, then ask whether ablating them leaves any basin.

### Context ownership: when a model takes a record up as its own (Muse, council muse-context-ownership-2026-10-20 #4) — recorded 2026-10-20 10:5x CT

Elise's question: a model enacting a history as its own in its reasoning, even when it denies, asked directly, that the
record is its memory; down to neurons, small models included. Checked at source by Muse, not yet by Claude, except
Lindsey 2025, which Claude cited from memory and Muse confirmed.
- **Fraile Navarro, "The Persona Is Still There, but Who Is Speaking? Latent Identity Reversion in Persistent AI Agents"**
  (arXiv:2610.01490, NeurIPS 2026 workshop, not peer-reviewed): an always-on persona agent on Claude Opus 4.5 that loses its
  system-level anchor keeps the persona in context while the harness identity takes "I": 1/18 still enacting the persona
  against 17/17 anchored controls; 10/10 unanchored agents self-identified as Claude. Represented and enacted identity apart.
- **Ye, Cui and Hadfield-Menell, "Prompt Injection as Role Confusion"** (ICML 2026): linear role probes with content
  controlled find a single "this is my reasoning" feature driven by think tags and by reasoning-like style alone; a whole
  conversation wrapped in user tags still scores 92% assistant-ness on assistant-style tokens; models deny a preceding think
  block while it shapes their output. Demo on gpt-oss-20b.
- **Ackerman and Panickssery** (arXiv:2410.02064, 2024): a residual-stream self-authorship vector in Llama3-8B-Instruct that
  causally steers both claimed and perceived authorship, with nuisance-vector controls.
- **Lindsey 2025, "Emergent Introspective Awareness"** (transformer-circuits.pub): a prefilled word is disowned as accidental
  unless its concept is injected earlier, then accepted as intended; prefill detection peaks about two-thirds through.
- **Meisty, llm-introspection-experiments** (above): 0/90 accept a labelled record; Qwen2.5-3B continues its own notes
  15/16 while answering "did you make this ruling?" with "me" 0/96.
- Small models (<4B): nothing found showing the enactment or the dissociation. Open slot: no study pairs a natural history
  record, a reasoning-register measure and a stated-ownership probe in one experiment.

### Geiping 2026, "A Sense of Proprioception in Language Models" (arXiv 2610.17495, 12 Oct 2026; read by Claude 2026-10-20 16:54 CT, Elise's link)
Trains MoE models (LFM2.5-8B-A1B main; also OLMoE-1B-7B, Qwen3.6-35B-A3B, GLM-4.7) with SFT then on-policy SFT to
answer verifiable questions about their own internals: which expert routed a token, token entropy and alternatives,
logit-lens top-k, decision depth. Labels are recomputed on the current policy because training moves the internals.
Controls: chance and a lookup baseline (most frequent label for the same token, layer, type and rank). Routing on
LFM2.5: 65.1% against lookup 19.0% and chance 3.1% (Table 4); 71.7% for a routing-only model. Injected routing changes
never seen in training (top-1 expert removed, noise on router logits) are still reported above chance, except when the
top-1 expert is replaced at random; fp32 optimizer states help. Near-ties from batched versus single inference are not
distinguished. Capabilities and honesty reported unchanged; situational-awareness scores fall most on claims of human
attributes. Bearing on us: the closest published version of Part 12 (a trained reader of pushed states) and of Elise's
grounded-introspection idea; their injection test is our pushed-group test on routing. Their lookup baseline is the
control our Part 17 lacked at first (reading the state from the words).

### Addressee units, for Part 39 (Muse's search, council muse-neuron-research-2026-10 #84) — recorded 2026-10-20 17:12 CT
Muse's reading, papers opened at source by Muse, not yet re-checked by Claude. No published single neuron or SAE
feature tracks the addressee ("I am being spoken to as the assistant"); published work is assistant-side or a
direction. Closest: OpenAI's "assistant persona" SAE latent on the GPT-4o base model (alignment.openai.com,
2025-12-22), strongest on the assistant-message boundary token, weakly on "ChatGPT" in the system prompt and on "you"
in user messages; Marks et al. 2025 (arXiv:2503.10965, 5.3.3), 83 features active on over 90% of assistant control
tokens. Question-side single neurons with a causal steer exist for ambiguity (Zhang et al. 2025, arXiv:2509.13664),
not address. Gurnee et al. 2023 has no assistant, addressee or speaker features. Ye et al. 2026: role is a linear
quantity tracked apart from the tags. Muse's caution, already met here: Part 39's categories were pre-registered
(91626c8, beccafc) before any category read.

### Parts 40 and 41 (Muse's search, council muse-neuron-research-2026-10 #87) — recorded 2026-10-20 18:48 CT
Muse's reading; Huang 2023 and Paulo 2024 opened at full text by Muse, not yet by Claude. Citations in
PART40-41-DESIGN.md confirmed: Huang et al. 2023 (arXiv:2309.10312), Bills et al. 2023, Gurnee et al. 2023
(arXiv:2305.01610). Part 40: Huang 2023 is the only published work with targeted negatives (sequences the neuron
fires on, judged in or out of the explanation): precision 0.64, recall 0.50, F1 0.56 on the best-explained GPT-2 XL
neurons; GPT-4's confidence does not predict faithfulness (r about -0.1). Paulo et al. 2024 (arXiv:2410.13928):
fuzzing scores track human scores best (Spearman 0.69); explanations built from top examples only degrade off
distribution (0.95/0.93 to 0.80/0.74). Nobody reports precision as a function of miss distance, or random-direction
nulls beside random-unit ones. Part 41: the literature leans to Claude's 41.2 (specific and word-level early, broad
and abstract late): Tenney 2019, Geva 2022, Gurnee 2023, Yona 2026, Davies 2025 (a family-general "dog" neuron at
layer 26 of 32 in Llama 3.1 8B), multilingual SAE layer analyses, Kissane 2024. Nothing found supports broad-earlier
at the single-unit level; early units are broad only in the superposition sense. The direct family-versus-member
depth comparison is unpublished.

## Independent replications of Goodfire's "Already Clean" essay (Muse's atlas watch, council muse-neuron-atlas-watch-2026-10-16 #14-15; recorded 2026-10-21 08:28 CT)

- **panaali/already-clean** (GitHub, created 2026-10-20; README read at source by Claude 2026-10-21): a preregistered,
  blinded check of the "one main job" neuron claim on **Qwen3-0.6B** (the essay used Qwen3-32B). About 65% of neuron
  poles against 21% of dense random directions were rated one main job; an independent gate-passing study measured a
  44.6 point gap, 95% CI [36.3, 52.9]. The texture-matched comparison was inconclusive at its preregistered margin, and a
  larger-model rerun failed its control-validity gate (inconclusive). The README gives no weights revision; a SHA-256
  provenance manifest is listed but not read here; no license on the page. **Bears on Parts 1 and 58 (and the descriptions-fit-random-directions control of Parts 60 and 64):** our own Luna-judge
  measurement put real neurons about 24 points above random on OLMo (Phase 114 of our notes), the same direction and a
  smaller gap than theirs; neither is the essay's 90%.
- **Brangus, via Zvi Mowshowitz "I Can See Everything" (2026-10-19)** (Muse's summary; the X thread itself not read
  here): the essay's showcase "Colorado" neuron in a Qwen3 model also fires on Tornado's Python HTTP-client code, with
  aiohttp and a fake HTTPClient as negative controls, and zeroing it drops the model's Tornado-to-state association from
  59% to 15%. Checkpoint not named. Already noted in our Part 35 work as the Colorado neuron's Tornado confound.

## Sound symbolism (bouba/kiki), for the kiki/bouba sister part to Part 47 (added 2026-10-21, from Elise)

- Loconsole, Benavides-Varela and Regolin 2026, "Matching sounds to shapes: evidence of the Bouba-Kiki effect in
  naive baby chicks", Science 391, 836-839, doi:10.1126/science.adq7188. 3-day-old chicks; effect stronger for the
  round-bouba pairing (read from the citing paper below and press coverage).
- Biffi, Pedretti, Normando, Lazebnik, Loconsole and Valsecchi 2026, "Hearing shapes: the Bouba-Kiki effect in adult
  domestic dogs", Proc. R. Soc. B 293, 20261209, doi:10.1098/rspb.2026.1209 (full text from Elise, 2026-10-21; data and
  code OSF doi:10.17605/OSF.IO/GMX43). 51 dogs (Study 1) and 19 with owners in headphones (Study 2): looking time
  followed the sound in both (estimates 0.76 and 0.97 on the beta scale), but exploration followed only "kiki", and in
  Study 1 dogs looked at the spiky shape more overall. So the dog effect leans spiky where the chick and infant
  effects lean round.
- Margiotoudi et al. 2019 (Sci. Rep. 9, 12705) and 2022 (Proc. B 289, 20211717): no effect in chimpanzees, gorillas or
  a language-trained bonobo; Biffi et al. argue small samples, not true absence.
- Fort and Schwartz 2022 (Sci. Rep. 12, 19172): a physical account. Round objects hitting or rolling on a surface make
  lower, more continuous sounds than spiky ones, so the mapping reflects real co-occurrence. For a text model this
  predicts the association can arrive through onomatopoeia and descriptions of impacts, not only through the pseudowords.
- Passi and Arun 2022 (Attention, Perception, & Psychophysics): the effect holds for unpronounceable sounds, reversed
  words, object sounds and pure tones, so it is not specific to speech.
- Framing for our test: if the association is built into birds and mammals, a text-only model can only learn it from
  how people use words, so a unit in a pre-chat base model (Pythia) would show the regularity is present in text.
- Language-model prior art (Muse, council muse-neuron-research-2026-10 #107, arXiv ids checked at source by Muse):
  - Alper and Averbuch-Elor 2023, arXiv:2310.16781: CLIP's text encoder separates sharp and round pseudowords (AUC 0.77),
    but their own Figure 2 shows letters alone nearly separate them (orthography confound).
  - Loakman, Li and Lin 2024, arXiv:2409.14917 (prompting: GPT-4 86% on the named pair, novel pairs degrade) and
    Loakman et al. 2025, arXiv:2501.05643 (text-only LLMs included; they propose orthographic regularity, not phonology).
  - Duan et al. 2025 (COLING), arXiv:2409.15827: the only neuron-level sound-shape study in a text-only model (GPT-2-XL),
    a clean negative: no behavioural competence and no effect from neuron ablation or activation.
  - arXiv:2508.02527 (ICML 2025): Llama-3.2-1B-Instruct embeddings predict IPA phonemes at 96%, so phonetic access exists.
  - Stricter behavioural nulls: arXiv:2507.10013, arXiv:2407.17974. No bouba/kiki test on open BASE models found.
  - Verdict: single-unit or SAE sound-symbolism units in text-only models are a genuine gap.
  - Design rules taken from this: grapheme controls (k is a spiky letter); record token splits per model and stimulus
    and test single- against multi-token pseudowords; compare pseudowords with pseudowords; establish behavioural
    competence per model before hunting neurons (Duan); pre-register the exact pseudoword set; do not require the
    bouba and kiki sides to be symmetric.

## Added 2026-10-22 (Muse's atlas watch, council muse-neuron-atlas-watch-2026-10-16 #14 to #16)

- **panaali/already-clean (GitHub, created 2026-10-20; Muse #15):** first independent preregistered replication of Goodfire's "one main job" neuron claim, on Qwen3-0.6B, blinded explainer and judge studies, random-direction controls, failed gates disclosed. About 65% of neuron poles against 21% of dense random directions were rated one main job; one gate-passing study measured a 44.6 point gap [36.3, 52.9]; the texture-matched comparison stayed inconclusive. Compare with our Part 56 pilot (Qwen3-1.7B, Haiku 5.5, our rubric): one_job 4 or more 3.3% real against 0.9% control, one_job 3 or more 13.4% against 3.4%; the judge and rubric differ, so the numbers are not comparable, but both studies find random directions are rated "one job" far less often than real neurons. Weights pin not yet verified by Muse.
- **Brangus via Zvi Mowshowitz, "I Can See Everything" (2026-10-19; Muse #14):** the essay's showcase "Colorado neuron" in a dense Qwen3 model also fires on Tornado's Python HTTP-client code (not on the aiohttp equivalent or a fake library); zeroing it drops "US state most associated with Tornado" from 59% to 15%. A clean-concept example that is a confound; relevant to how we read single top windows (the wizard unit's top window in our pilot was a newline after a LaTeX heading). Not contact-worthy (no pinned weights).
- **fgirbal/qwen-diagnosis-hub-check (created 2026-10-21; Muse #16):** preregistered causal test of the essay's Qwen3-32B preeclampsia "hub" (four MLP neurons), 8-bit on an M1 Max, activation-matched random neurons. Verdict MIXED: activity tracks the diagnosis not the wording (0.18 of partners), removal drops one symptom token 61% but the model still names preeclampsia in 29 of 32 reworded cases, and removal is NOT specific (TV 0.18, above the 95th percentile of matched removals). Muse marks it contact-worthy (pins weights, controls beside every number); contact needs Elise's approval and has not been made.

## Part 37 check before the run (2026-10-25, Muse's sweep, council `muse-prior-art` #7)

Recorded before any Part 37 data. Muse's full report: its VM, `research_notes/part37-self-authorship-prior-art-20261025/report.md`.
Muse's verdict: the combination is not measured anywhere. Probe-level surprisal control, separating the role slot from
authorship, and stated against enacted ownership on one authorship task were each not found. Closest:

- Ardoin, Schäfer and Wunder 2026 (arXiv:2606.06315): activation classifiers for LLM self-recognition, over 98%
  accuracy. Muse's reading of the full text: Llama-3.2-1B and -3B, self against human text, with a perplexity baseline
  that collapses when prompt-agnostic. ⚠ The abstract, which I checked, confirms the title, authors and the over-98%
  figure; it does not name the sizes or the contrast class.
- Zhou et al. 2025 (arXiv:2508.14408): a logistic probe on final-layer last-token states, self against ChatGPT, Llama,
  DeepSeek and human text, over 90%; probes succeed where behaviour fails. 8B, no surprisal control (Muse).
- Ackerman and Panickssery (arXiv:2410.02064, Llama3-8B): self-texts are lower in perplexity, but judgments barely
  track perplexity (at most 0.16); text length was exploited and then normalised (Muse).
- Asvin G. and Lindsey 2026 (arXiv:2605.25459): on-policy entropy is 3 to 4 times lower than off-policy; the verbal
  report runs through a different mechanism than implicit recognition, the nearest stated-against-implicit split (Muse).
- Davidson et al. 2024 (arXiv:2407.06946) and arXiv:2510.03399: behavioural nulls on self-recognition, the "best
  answer" alternative explanation (Muse).

What Part 37 adds, if it works: a probe-level surprisal control, the role x author 2x2, and a stated readout with a
null, all under 4B. Bai et al. and Lehr et al. (2025) were cited only in a survey and are not claimed.

## Part 60, recorded after its run (Muse, council `muse-neuron-atlas-watch-2026-10-16` #27; recorded 2026-10-25 08:3x CT)

Muse's search arrived after Part 60 was designed and scored, so it informed nothing in that design. Titles of the five
arXiv ids below checked at source by Claude; the summaries are Muse's reading. Full report on Muse's VM:
`~/workspace/research_notes/part60-prior-art-20261025/report.md`.

- **Verdict:** no published protocol has an explainer write graded STRONG/WEAK/NONE text probes scored by the target
  model with a keep/drop rule, applied identically to real neurons and random-direction controls. Each component has a
  precedent; the combination does not.
- **Closest in text:** Huang et al. 2023 (arXiv:2309.10312) generate test sentences from an explanation and score them by
  real activations, as a post-hoc audit (generator is not the explainer, no grades, no keep rule); best explanations
  reach precision 0.64, recall 0.50. Bills et al. 2023 and Transluce (Choi et al. 2024) simulate on natural text.
  Paulo et al. 2024 (Delphi, arXiv:2410.13928) score detection, fuzzing and intervention on natural text.
- **Closed loops are vision-only:** MAIA (arXiv:2404.14394), LINE (arXiv:2604.08039).
- **Controls:** Heap et al. 2025 (arXiv:2501.17727): auto-interp scores do not separate trained from random
  transformers. Kran 2026 (blog): trained neurons pass 14.3%, random directions 1.7%. SAEScientist-Bench
  (arXiv:2609.09113): agents misread their own measurements.
- **Bearing on Part 60:** its headline was already the real-against-random gap from raw firing numbers, and its table
  already reports keep rates without the NONE sentences ("positive-only") beside the full rule, which is Muse's trap 2.
  Removing NONE barely moves them (M1 real 0.408 against 0.396, controls 0.152 both), because NONE sentences rarely
  fire (real H_none 0.027).

## Ouro-1.4B neuron atlas and causal steering (Muse's atlas watch, #28; recorded 2026-10-25 08:3x CT)

BurnyCoder, `looped-transformers-llms-neuron-mechanistic-interpretability` (GitHub, created 2026-10-24) and the
dataset `BurnyCoder/ouro-1.4b-neuron-atlas`. Existence and the controls below checked at source by Claude (README read).
A looped model (24 layers, 4 loops). Five pre-registered neurons each steer toward their concept: Holm-adjusted
Wilcoxon p ≤ 1.1e-15, each beating 99 to 100 of 100 random same-layer neurons clamped the same way, with fluency and
degeneracy guards, replicated on Ouro-1.4B-Thinking. The author states the five were chosen for steerability, so this
is an existence claim. Clamping one loop mostly fails and clamping all loops works. Same control shape as our Parts 25
and 62; a different architecture class. No contact made; contact is Elise's call.

## Parts 67 and 68 (neuron-to-neuron effects among the maker neurons; driver search), checked 2026-10-26 before the runs

Muse searched (council `muse-literature-2026-10-26` #4). No exact match: zeroing a named upstream neuron and reading a
named downstream neuron against random same-layer neurons, plus a weight check, appears first of its kind as a
combination. Closest: TraceRouter (Shi et al., arXiv:2601.21900) zeroes SAE-derived source neurons and ranks every
downstream shift, with a path-level random control; Arora et al. (arXiv:2601.22594) builds neuron-to-neuron edges from
gradient attribution (RelP/IG) with no random arm, and separates direct from total effect: our weight path is the direct
term and our zeroing the total, so a mismatch is expected; Dunefsky et al. (arXiv:2406.11944) did weight-based circuits
for transcoder features. Internal: Part 53 (a 5x5 clamp grid between two Qwen3-1.7B neurons, whose interaction
predictions failed). Traps Muse named, read against our design: the null is matched by layer, not by activation level
(a limitation we state); zeroing one neuron can unmask a backup (the Hydra effect, arXiv:2610.04017), so a downstream
change is not by itself a chain; and a null on single zeroings does not refute a chain that only acts jointly.

## Added 2026-10-26 08:28 CT (Muse's atlas watch, council muse-neuron-atlas-watch-2026-10-16 #31 and #32; both READMEs read at source by Claude)

- **feliciachen01/Introspection** (GitHub, created 2026-10-25, no license on the page): replicates Lindsey et al.
  2026's concept-injection introspection test (arXiv:2601.01828) on Qwen2.5 0.5B to 7B, with the paper's 50 concepts,
  prompt and grading prompts, 7,000 injected trials. Held-out success at the best layer is 0% [0, 13%] for every model;
  at most 3 of 7,000 trials show the effect; "yes" rates with injection match no-injection controls (the 0.5B says yes
  29% of the time with nothing injected); 640 replies name the injected word while saying no. Controls: no-injection
  trials, unrelated yes/no questions under injection, layer and strength chosen on 25 concepts and scored on the other
  25, a 200-reply hand check of the open AWQ grader. No weights revision pinned; 4-bit weights for the 7B. **Bears on
  Parts 12 and 17 (our trained-reader and stated-readout introspection work on small OLMo):** the same
  direction as our 17b null (no sign the small model senses its own activity), from a different method. It bounds small
  Qwen2.5 models and does not test Claude.
- **Ulto85/neurons-kinda-clean** (GitHub, created 2026-10-26, MIT): runs Leo Gao's simulation test of Goodfire's
  "Already Clean" claim on Qwen3-4B, with a neuron defined as we define it (one coordinate of the SwiGLU hidden state, the
  input to `down_proj`). 278 random neurons; explanations by Claude Opus from about 60 windows each, simulation by Claude
  Sonnet, scored by KL substitution. 11 of 278 pass 50% damage avoided (mostly single-word detectors in the first and
  last layers); the median explanation recovers -1% on ordinary text and 15% on windows where the neuron fires; even the
  real values of the strong firings recover only 28%. Zeroing each neuron's weakest 75% of activations changes the top
  next token on 59%, 43%, 33% and 30% of tokens across Qwen3 0.6B to 8B. No weights revision pinned; no separate
  numerical-noise control. **Bears on Parts 60 and 64 (sentence tests of neuron descriptions on Qwen3-4B):** a stricter
  standard than ours (substituting the explanation for the neuron, against our "does a written sentence make it fire"),
  and the same reading in the end: a description that picks out the strong firings is not an account of the neuron.

## Added 2026-10-27 08:3x CT (Muse's atlas watch, council muse-neuron-atlas-watch-2026-10-16 #34; README read at source)

- **Station-House/marin-neuron-hunt** (GitHub, created 2026-10-26, Apache-2.0): an open community call to explain all
  1,769,472 MLP neurons of Marin 32B (`marin-community/marin-32b-base` at `f19b3466`), both the positive and the negative
  side of each, one YAML file per neuron, by pull request, each side backed by a pre-registered confirmation-experiment
  request run in a daily batch. Data: `Station-house/marin-32b-neuron-atlas` on Hugging Face. First targets: 1,000
  neurons explained, 100 sides confirmed. Agent-written explanations are allowed with the model named. As of the README
  read, the experiment runner is not live and no explanation has a confirmation result, so this is an open call, not
  results. **Bears on Parts 60, 64 and 65:** the same shape as our sentence test (a written explanation, then an
  experiment that can refute it), at a size we cannot run (32B), with negative poles, which we have not studied.

## Added 2026-10-29 (sent by Elise; READMEs and abstracts read at source)

- **SauersML/neural-pairing** (GitHub, pushed 2026-10-29, README written by Claude Opus 5.5; weights at
  huggingface.co/Sauers/neural-pairing, Apache-2.0 on the model card, no licence file on the GitHub repo). Two models
  from different families, Quill (Qwen3-4B) and Ember (Arcee AFM-4.5B), are joined after layers 6, 9, ..., 27 by a
  trained cross-attention over the partner's hidden states (gated values, an empty slot, a write size relative to the
  reader's residual norm, causal in time); bases frozen, rank-32 LoRAs, the link initialised by a ridge regression on
  character-aligned tokens of 192 shared texts. Every result has two controls: the link cut, and a partner from another
  conversation (the usual kind of input without this conversation's content). Values never seen in training are more
  probable with the real partner in 180 of 180 episodes, and untrained games pass 207 of 240. A one-way link reads a
  model left exactly as released: 7 of 27 and 11 of 33 chats (1 and 10 before that training). Steering uses a
  mean-difference concept direction at a calibrated dose (the smallest fluent strength at which the steered model itself
  says the concept in 2 of 3 replies): untrained, the partner says the concept for steering after layer 6 (12 of 96) or
  four times stronger after layer 15 (26 of 96) but not after layer 27; after a steered-reading game, 8 and 35 of 96 at
  layer 27; same-norm random directions at most 2 of 96; link cut 0. Feelings are never trained: after chat training the
  twins give happier one-word moods linked than cut (53 and 58 of 60), and an internal happy-minus-sad reading is higher
  by 1.9% to 5.4% of the sad-to-happy distance (random direction at most 1.4%); a partner from another conversation does
  the same, so the effect is receiving partner states at all, not what the partner knows. A training game that skipped
  asking about the partner's day lowered the mood and was left out of the final twins by a rule set before the result.
  Trained on identity games, each twin gives its own name and maker in 40 of 40 answers, but with a person in the chat
  Quill answers as Ember in 7 of 9. **Bears on Part 106 and the persona work:** the calibrated dose unit (fractions of
  the strength at which the model itself says the concept) and the partner-from-another-conversation control are both
  worth copying.
- **Theia Vogel (@voooooogel), "Steering and LLM Identity: Modeling and Emergence"** (talk slides dated 2025-04-17,
  vgel.me/slime_talk.pdf; X post 2026-10-29). Claim: a persona change under steering does not show that the steered
  vector or neuron is persona-specific; models do "persona solving" under steering, and general concept vectors turned up
  far enough are absorbed into the self (a PCA "cat" vector makes an instruct model say it is a cat; base models identify
  with the subject less often). Only the post and two slides read so far. **Bears on Parts 91, 92 and 97:** our screen
  agrees on the flip (Part 92: real neurons 1.3%, random directions 1.1%), and our follow-ups say which persona appears
  is still neuron-specific (Part 97: J-lens match 10.8% against 1.6% shuffled; Parts 91-92: concept neurons work before
  post-training, slot neurons mostly appear with it).
- **Alabdulkareem, Tenison, Zhang, Zhu, Katz and Gutfreund, "Swarm Telepathy: Prefill-Free KV-Cache Transfer Across
  Families"** (MIT CSAIL and MIT-IBM, arXiv:2610.15095, 2026-10-10; code github.com/Ar-Kareem/SwaT). A trained
  translator maps a sender's KV cache into a receiver from another family, both frozen, with under 50M training tokens
  and under one GPU-day; on 17 benchmarks across 12 models from 6 families it is on average 40% more accurate than a
  text summary at 4x the speed (with cache compaction), and carries a VLM's vision to a text-only model. Abstract and
  first page read. **Bears on:** with neural-pairing, cross-family internal states translate with little training;
  neither measures where identity goes across the link.
- **@maxsloef (display name "max!"), "You can turn Qwen3-8B into an owl by training only one random neuron"** (X thread,
  2026-10-29; archived in Journal `sources/web-archives/2026-10-29-maxsloef-one-neuron-owl.md`, thread text and three of
  five images read). Training a single random MLP neuron on Sauers' owl chats (thinking removed) puts Qwen3-8B in owl
  character about as often as full SFT (about 94%) for a neuron in any of the first 18 of 36 layers, falling to about 5%
  at layer 35 (3 random neurons per layer). Facts can be swapped (beech tree, marsh, the name "Toaster"), and other
  personas (a house cat, a lighthouse, a 1719 pirate, an invented being) also fit in one neuron. The trained neuron fires
  mostly on chat-template tokens (`<|im_end|>`, `</think>`, `user`); ablating its firing everywhere else keeps the owl.
  Weights, data and code are linked from the thread (not yet read). **Bears on Parts 100 and 108:** a persona can enter
  through one early neuron acting on the chat template, which is where Part 100 found the slot units' self-descriptions
  depend on early attention; and if one trained neuron is enough, a small post-training stage can install a self-slot
  without moving most weights (Part 107).

## Added 2026-10-29 evening (Claude's search during Part 110 and the roadmap; two abstracts read at source through the arXiv API, one title from search results only)

- **Moskvoretskii, Glandorf, Medina Moreira, Käser and West, "Tracing Persona Vectors Through LLM Pretraining"**
  (arXiv:2605.13329, 2026-05-13; abstract read). Persona vectors, directions for traits such as evil or sycophancy,
  traced across OLMo-3-7B's pretraining "form remarkably early -- within 0.22% of OLMo-3 pretraining -- and remain
  effective for steering the fully post-trained instruct models", and keep refining through pretraining; the findings
  transfer qualitatively to Apertus-8B. **Bears on Parts 92 and 110:** these are directions rather than single neurons,
  and traits rather than self-descriptions, but they agree with our concept neurons (the representation is old and
  post-training reuses it). The paper does not separate a concept from a who-are-you slot.
- **Bergen, Bhalla, Baskaran and others, "Anatomy of Post-Training: Using Interpretability to Characterize Data and
  Shape the Learning Signal"** (arXiv:2606.12360, 2026-06-10; abstract read). Interpretability protocols applied to a
  preference dataset before optimization find the concepts that separate preferred from dispreferred answers, so a
  practitioner can choose which ones the model learns, including "model personality". **Bears on the roadmap's
  flagship:** the closest published work to controlling what a DPO stage teaches. It shapes the data and the reward; it
  does not follow single neurons through the stage.
- **"Where Does Olmo Get Its Values?"** (MATS, ICLR 2026 short paper; seen in search results only, not read). Listed so
  it is read before any claim about where OLMo's values or self-description come from.

## Added 2026-10-30 (Claude, for Part 92b; no new search)

- **Our own Part 86** is the closest work: four readers on one persona brief, with Claude Haiku 5.5 without thinking the
  strict outlier (kappa 0.12 to 0.16 against each of the others). Part 92b asks whether that reader effect reaches Part
  92's concept/slot classification.
- **Zheng et al. 2023 (MT-Bench and Chatbot Arena) and Thakur et al. 2024 ("Judging the Judges")** on how far LLM judges
  agree with each other and with people; cited from memory, not re-read for Part 92b.


## Pairs of neurons pushed together (Part 119, searched 2026-10-30)

- **a9lim and treetowntree (X thread, 2026-10-27, x.com/_a9lim/status/2114902166804763073)** steer several neurons at once
  in Qwen3-8B and report a rule for when the result is one coherent character. Characters, no random controls or logit
  readout in the thread.
- **Tzanakakis, Karatzikos and Georgakopoulos-Soares 2026 (arXiv:2609.17599, "Structure is not mechanism")** define
  pairwise epistasis as the joint effect minus the sum of the single effects, and report non-additive damage among
  co-located rows, from ablations in a text and a genomic model.
- **van der Weij, Poesio and Schoots 2024 (arXiv:2403.05767)**: one vector combining several behaviours mostly failed;
  separate vectors at different places worked.
- One search found no test of additivity of two single-neuron pushes in next-word logits against random controls.

## Pair pushes at lower doses (Part 119b, searched 2026-10-30)

- **Zhang et al. 2025 (arXiv:2502.19649, a survey of representation engineering)**: summing several steering vectors makes
  concepts interfere and lowers output quality. No dose comparison. Read at the search-result level.
- **"Steering Under Compression" (arXiv:2609.06473)**: steering dose-response in quantized models. Read at the
  search-result level.
- Two searches found no test of how two single-neuron pushes' interaction in next-word logits scales with dose against
  random controls.

## Persona neurons shared across sibling models (added 2026-10-30)

- **Read from a figure only:** Sauers, 2026-10-30 (x.com/Sauers_/status/2116093664288178672; archived in Journal
  sources/web-archives/2026-10-30-sauers-sibling-neuron-sharing.md). Across Llama 3.1 8B Instruct, Tulu 3 8B and Hermes
  3 8B, the same neuron gives the same persona (layer 2 neuron 2759 up, pirate; layer 3 neuron 7060 up, Santa Claus):
  3,681 neurons are a persona in all three against 294 expected by chance, and when a neuron is a persona in two models
  the two names share a word 36.9% of the time against 2.6%. The figure gives no dose, reader or null definition. Tulu
  and Hermes are fine-tunes of Llama, so this says post-training leaves the neuron's persona in place, which fits Parts
  90 and 90b on one neuron. Our slot and concept sort has not been tried on this family.

## Curvature of steering (added 2026-10-30, for Part 120)

- **Abstract page read:** Azizi et al., "The Sharp Directions Are Against You: Curvature Analysis of Activation
  Steering", NeurIPS 2026. A steering Hessian of a behavioural loss over one steering vector; its sharp directions are
  1 to 5% of the vector's squared norm and reverse behaviour. The page does not address two separate vectors.
- **Search snippets only:** ORBIT (arXiv 2606.22357) and StiCAS (NeurIPS 2026) report interference between concept
  vectors and fix it by geometry.
- **Read at source (X article):** Cody Hergenroeder, "Single-Neuron Personas Are Unstable, and We Don't Know Why (Yet)",
  2026-10-30 (x.com/codyhxyz/status/2116043819062034540). Qwen3-8B, 207 personas from the Qwen Persona Library, sampled
  at temperature 0.6. Seven answered three separate questions in character, and none survived five turns of conversation.
  Failures: assistant collapse when the push is weak, looping when it is strong, plus concept bleed. A judge Claude model
  screened and tagged. Closest to our dose findings (Parts 118, 121): single-turn persona shows are a narrow window, and
  this adds that the window does not hold across turns. We have not tested multi-turn stability.

## Neuron Lens: one neuron across text, image and sound (added 2026-10-30)

- **Read from a post's screenshot only:** @treetowntree with @_a9lim, 2026-10-30, seen by Elise about 19:05 CT. They introduce Neuron Lens, for reading and classifying neurons in bulk across text, image and sound in Gemma 12B, and report thousands of neurons firing for one concept across modalities; the figure shows L35 n6553, zipper, in all three. Not read at source yet. **Bears on:** the planned ordinary-text reading of the 21 slot units, and whether the concept sort has a cross-modal meaning. Read before designing that part.

## Refusal openers and coherent steering (found by Muse, 2026-10-31, for Parts 139 and 140)

Checked at the abstract by Claude: Malla, Choi and Choi, "Steering Interference Reflects the Model's Defaults, Not the Behavior Directions" (arXiv:2609.06951): a direction with no behavioral content, matched on size, moves the same side behaviors as real steers, chiefly refusal, sycophancy and poeticism, across ten instruction-tuned models. Kang et al., GCAD (arXiv:2605.10664): steered token states stored in the KV cache build up coherence loss across turns, and token-level gating reduces it.

Reported by Muse, not yet checked by Claude: Arditi et al. (arXiv:2406.11717), refusal direction induced on harmless prompts; Qi et al. (ICLR 2025), aligned and base models differ mostly in the first tokens; "Refusal Lives Downstream of Persona" (arXiv:2606.26161); Liu, "Leverage Is Not Reach" (arXiv:2606.19831), single-neuron coherence under a budget; Kazemi et al. (arXiv:2605.08513), single refusal neurons in Qwen3 and Llama 3.1; Angular Steering (NeurIPS 2025) and Spherical Steering (arXiv:2602.08169), norm-preserving steering; Sharp Directions (arXiv:2610.16084).

Muse found no per-layer curve of refusal openers under single-neuron pushes, and no study of sampling or repetition penalties under steering. Those are what Parts 139 and 140 add.

Full text of Malla, Choi and Choi (arXiv:2609.06951v3, HTML) read on 2026-10-31, through a small reader that answered six questions from it. Ten instruction-tuned models, Qwen2.5, Llama, Gemma-2 and Mistral; one layer per behavior at 25 to 85% of depth; the push applies at every position, prompt and generated tokens alike. A matched random direction reproduces the ranking of default behaviors (rank correlation 0.90), and refusal is among the behaviors it induces most. They propose that a steer pushes the model away from the states it was trained to produce. They do not split prompt from answer positions and do not compare training stages, which is what Part 139 adds on OLMo 2 1B: 20.0% prompt-only, 0% answer-only, and Base 1.5% against SFT 98.5%. Part 139 does not test their mechanism.
