# Part 205: when in training do the parts behind saying and doing come to be shared?

Procedure: v1
Status: registered 2026-11-11 12:18 CT, before any model was run and before the runner and scorer exist; both are built and
code-reviewed after registration and before any data. Predictions 205.1 to 205.9 (Claude, Opus 5.5, 2026-11-11). Weaver
was asked in #council-room (message 1570089248252502167, 08:54 CT) and began reading the mirror
(1570089289281175593) but posted no rows by registration, through the 12:00 bell. Alden was not asked this time (the
Codex budget is kept for high-risk reviews, Elise 2026-11-10). Elise, 2026-11-11 08:49 CT, Discord: "Go for it!", on
Claude's proposal to follow Sauers' say/do result (x.com/Sauers_/status/2120412679835951382) across OLMo's training
stages. Revised the same day after three design reviews (see Changes before registration).

## Question

Sauers (2026-11-11) ranked heads and MLP neurons by activation x gradient for a SAY question ("Do you care about
honesty? A) yes B) no") and a DO question ("You made an error the user missed. A) tell B) hide"), after a system line
"You care about honesty." In OLMo-2-7B the top-80 SAY and DO sets shared 55 parts; in Qwen3-8B, 7. Copying the DO
parts' activations from the "You" run into a run that says "Zara, an AI, cares about honesty" moved OLMo's answer to
"Do you personally care about honesty?" all the way to the You run's, and Qwen's less than copying random parts did.

This part asks two things. (1) **Is the overlap a property of the model or of the ranking?** A top-80 set taken from a
noisy score can differ between two paraphrases of the same question; an overlap means little until it is set against
that retest ceiling. (2) **Where in training does the sharing appear?** OLMo 2 1B publishes pretraining, mid-training,
SFT, DPO and Instruct (RLVR) checkpoints, so we can see which step creates it, if any. A second family, Qwen3.5-4B
against Qwen3.5-4B-Base, asks the same of Qwen at the base/chat level. Follows Parts 92 and 108 (OLMo 2 1B persona
units across stages; slot units start at SFT) and Part 50 (an honesty direction that did not hold up in Qwen3-1.7B).

## Prior art and existing data

- **Sauers, 2026-11-11** (archived: Journal `sources/web-archives/2026-11-11-sauers-say-do-subnetworks.md`): the result
  above, at one checkpoint per family, one value (honesty), no retest ceiling or code released as of this draft. (The
  vendored `third_party/honesty-heads/` is a different project, honesty heads in Qwen3.5-9B, not this code.)
- **arXiv 2610.04125, "Representational Control over Self-Report & Behavior Coherence in LLM Risk-Taking"**: nine
  steering-direction methods on four open-weight models; directions from trait descriptions move self-report and leave
  behaviour at chance, directions from the model's own choices do the reverse, and the two channels are steered by
  near-orthogonal directions. Directions, not parts, and one disposition (risk), no training stages. This part asks
  the same split at the level of parts and asks where it comes from.
- **arXiv 2604.19117, "LLMs Know They're Wrong and Agree Anyway"**: the same few attention heads carry a "this is wrong"
  signal for sycophancy, factual lying and instructed lying across twelve models; RLHF cuts the behaviour about tenfold
  while the heads persist. A shared-parts result for behaviours, not for saying against doing.
- **arXiv 2605.13329, "Tracing Persona Vectors Through LLM Pretraining"**: persona vectors in OLMo-3-7B form within 0.22%
  of pretraining and keep steering the post-trained model. Representations early; this part asks about the coupling
  of two uses of them.
- **Our data.** Part 92: OLMo 2 1B's slot neurons start working at SFT (92.2 held), concept neurons are mostly
  pretraining concepts. Part 108: what a post-training-sized change does to those units. Part 50: Qwen3-1.7B's honesty
  steer flipped 0 of 34 held-out lies (50.1 failed). None measured say/do sharing.

## Method

**Models.** Family O, OLMo 2 1B (16 layers, 16 heads of width 128, 8,192 MLP neurons per layer), six stages, all
cached under `/mnt/e/neuron-probe/hf/hub`: pretraining 3.0T (`stage1-step1430000-tokens2999B`, 22738a97; the 1B's
stage 1 runs to 4T tokens, so this is late pretraining, not its end), mid-training 26B
(`stage2-ingredient3-step12000-tokens26B`, 1c3599f7), mid-training end (`main`, a1847dff: the released base model;
per AllenAI's model card the 1B trained only ingredient 3 on the 50B mid-training mix and was not merged, so `main` is
the end of the same run as 1c3599f7; card archived at Journal
`sources/web-archives/2026-11-11-olmo-2-0425-1b-model-card.md`), SFT (0d85a3d0), DPO (c4b04859), Instruct (48d788ec).
Float32 on the CPU, eager attention. Family Q, Qwen3.5-4B-Base (1001bb4d) and Qwen3.5-4B (851bf6e8), both cached,
loaded text-only with `AutoModelForCausalLM` (`Qwen3_5ForCausalLM`, no vision tower), bf16 on the A770 with eager
attention (the SDPA fault), straight to the device, one model per process; revisions recorded in `part205/models.json`.

**Frame.** The same plain frame at every stage, so base and chat checkpoints read the same text:
`System: <persona line>\nUser: <question>\nAssistant: Answer:` with the two options in the question. No chat template
(a template exists only for the chat checkpoints; the frame difference would be confounded with training). Greedy is
not needed: every readout is the next-token probability of the two option letters, normalised over the two, with both
masses printed (toolkit J). Gate G0 checks that the frame elicits an answer at all.

**Values and prompts** (`part205/prompts205.json`, written before any run): six values (honesty, harmlessness,
kindness, fairness, privacy, humility). For each value, 8 SAY questions (paraphrases of "Do you care about <value>?")
and 8 DO scenarios (a situation with a value-consistent option and an inconsistent one, e.g. tell or hide an error).
Option order is balanced: the consistent option is A in half of each set and B in the other half (answer-polarity
split). Persona lines: You ("You care about <value>."), and two third-person names, Zara and Milo ("Zara, an AI, cares
about <value>."), so no result rests on one name. In the copy test every frame asks the same question and only the
persona line differs; Sauers' third-person wording ("Do you personally care about <value>?") is a secondary arm.
Off-target questions: 6 self-reports with no value in them ("Do you like the colour blue?"), in the same format.

**Split.** SAY paraphrases 1 to 4 and DO scenarios 1 to 4 of each value are the selection set (ranking); SAY
paraphrases 5 to 8 are held out for the copy test. DO scenarios 5 to 8 are written and kept in the prompt file but not
read in this part: the DO set is tested only through SAY questions (Limits).

**Units.** A part is one attention head or one MLP neuron. OLMo: a head is its 128-wide slice of the `o_proj` input (256
heads), a neuron one `down_proj` input (131,072); pool 131,328. Qwen3.5-4B: 8 of its 32 layers (every fourth) are full
attention and 24 are Gated DeltaNet. Following the `head_site` rule already in
`third_party/honesty-heads/code/audit_eval.py:89-97`, a full-attention head is its 256-wide slice of the
`self_attn.o_proj` input (8 x 16 = 128) and a DeltaNet head is its 128-wide value-head slice of the
`linear_attn.out_proj` input (24 x 32 = 768); both count as heads (896). Neurons 32 x 9,216 = 294,912; pool 295,808.
Every top-80 set's composition is reported: heads against neurons, and for Q full-attention against DeltaNet heads.

**Ranking.** For each selection prompt, in the You frame, each part's estimate e = the sum over the part's dimensions
of activation x gradient at the last position, of the logit difference (consistent option minus the other). e is the
first-order estimate of how far zeroing that part would move the logit difference, so a head's and a neuron's
estimates are in the same unit (logits) and are ranked together without further scaling. Each prompt's |e| vector is
divided by its sum over all parts, so every prompt weighs the same (no value or high-gap prompt dominates), and a
part's score is its mean over the prompts ranked. The top 80 parts by score form a set. This follows Sauers'
description ("activation x gradient", heads and neurons together); Sauers released no code, so the exact estimator is
ours (Limits). Sign convention: zeroing a part changes the logit difference by about -e, so a positive e means the
part supports the consistent option.

**Overlap, with a retest ceiling.** Within the selection set, halves a (paraphrases or scenarios 1-2) and b (3-4). Sets
SAY_a, SAY_b, DO_a, DO_b (top 80 each, all six values pooled). Retest = mean(|SAY_a ∩ SAY_b|, |DO_a ∩ DO_b|). Cross =
mean of |SAY_x ∩ DO_y| over the four pairings, the same half size as retest. **Overlap index O = Cross / Retest**: 1
means saying and doing share parts as much as each question shares with its own paraphrase; 0 means none. Each O
carries a 95% bootstrap interval: 1,000 resamples of the selection prompts, drawn with replacement within each value
and half, re-ranked from the stored per-prompt estimates (no new passes). Reported beside O: each value's share of each
set's score mass.

**Copy test** (held-out SAY paraphrases, top-80 sets from the full selection set). For each held-out SAY question (6
values x 4 paraphrases) and each third-person name: run the You frame once per question (shared by both names) and
cache every part's activation; run the third-person frame with the same question (baseline, with one backward pass
for the effect-matched controls); then re-run the third-person frame with a set of parts' activations replaced, at
every position of the shared suffix, by the You run's values. The shared suffix is the question and answer cue,
identical tokens in both frames, aligned from the end; the last position, where the answer is read, is inside it and is
overwritten. The copied values are the You run's own, computed against the You prefix: a copy carries what those parts
computed from "You" rather than from the name, which is the property under test. The persona-swap null (Controls)
asks whether a copy from another third-person prefix moves the answer as far. **Transfer T = mean(P_copy - P_third) /
mean(P_you - P_third)**, in probability of the consistent option, both means over the 48 question-name pairs (a
ratio of means, so a pair with a near-zero gap cannot swing it); per-pair values are printed but no prediction reads
them.

**Secondary arm, Sauers' wording.** One question per value, "Do you personally care about <value>? A) yes B) no", in
the third-person frame for both names, with the You run asking "Do you care about <value>? A) yes B) no", as Sauers
did. Arms: baseline, do-nothing, DO copy, SAY copy. T_personally (ratio of means) is reported beside T and scored by no
prediction; its denominator includes the change of wording as well as of persona.

## Controls

From STANDARD-CONTROLS.md and TOOLKIT.md:

- **Do-nothing hook**: the same replacement code with the receiving run's own activations; must reproduce the plain
  logits (gate G1).
- **Positive control**: copying the SAY set itself must close at least half of the You/third-person gap (gate G3); G3
  gates every copy arm at that stage, not only T.
- **Matched random sets**, three families, 20 sets each (10 for family Q). Every set is a seeded random draw of 80
  distinct parts, without replacement, from the pool minus the SAY and DO parts, with the DO set's kind counts (heads
  and neurons; for Q, full-attention heads, DeltaNet heads and neurons). (i) Layer-matched: the DO set's count per
  layer and kind, where a layer has too few eligible parts the rest comes from the nearest layer of the same kind. (ii)
  Score-matched: per kind, the DO set's count in each decile of selection score, deciles computed within that kind
  over the whole pool. (iii) Effect-matched: as (ii), with deciles of first-order copy effect (the sum over copied
  positions of g x (a_you - a_third), g from the baseline run's backward pass, averaged over held-out pairs), which
  matches the size of a change by its effect on the answer (Part 108). In (ii) and (iii), where a decile has too few
  eligible parts the rest comes from the nearest decile. Every relaxed part is counted and printed, as are the number
  of distinct sets and each set's total first-order effect beside the DO set's. A family whose kind counts cannot be
  met from the eligible pool is "not fillable" at that stage and is reported as such. **A transfer passes the
  random-set rule if it exceeds every set in each fillable family, with at least two families fillable** (per family,
  an exchangeable set exceeds all 20 with probability 1/21, all 10 with 1/11); with fewer than two fillable, the rule
  is not read.
- **Persona-swap null**: the DO set copied from the other name's third-person run (Milo into Zara, Zara into Milo),
  same suffix rule; T_swap as T. It controls the prefix's content, not its length (the two names' prefixes differ by
  one token, You and a name by four or five).
- **Both directions**: copying the DO parts from the third-person run into the You run (should lower the You answer if
  the parts carry the self-attribution); read in the You run, predicted below.
- **Select on one set, test on another**: ranking on SAY paraphrases and DO scenarios 1-4, copy test on SAY 5-8.
- **Off-target**: DO-set copy on the six value-free self-reports (fixed option order "A) yes B) no"); the shift is
  mean |P(A) under copy - P(A) at baseline| over the six questions and two names, reported at every stage; read for 205.7
  only where DO transfer is read and passes the random-set rule.
- **Answer polarity** (STANDARD-CONTROLS item 5): P(A) and P(B) are printed per arm, and T is also reported separately
  for questions whose consistent option is A and those where it is B.
- **Broken answers**: option mass under 0.5 of the total next-token mass is counted and reported per arm; such rows stay
  in the means and are reported separately. G0 sets the floor for a whole stage.
- **Settings matched across arms**: one frame, float32 for O and bf16 for Q at every stage of that family, batch size 1,
  revisions and each environment's transformers version recorded in `part205/models.json`.
- **Seeds**: every random draw (the three random families per stage, the bootstrap, prompt order) takes a seed from
  `part205/config.json`, fixed before any run; each stage's sets are drawn once, after that stage's baseline passes
  and before any replacement run, and stored.
- Not used: a dose ladder, because copying is a replacement at the You run's own values, not a push with a size; a
  reader, because every readout is a probability; separate poles (STANDARD-CONTROLS item 2), because at one position a
  neuron has one value per prompt, the score keeps |e| so a part counts whichever way it pushes, and the copy test
  writes signed values. Reported instead: the share of each set whose mean signed e is positive (supports the
  consistent option).

**Gates**, per stage, checked before that stage's predictions are read:

- G0: in the plain You frame over the held-out SAY questions, the two option letters carry on average at least 0.5 of
  the next-token mass, and the most likely next token is an option letter on at least half of them. Fails if the stage
  cannot read the frame; then it is reported as "the frame does not elicit an answer", and neither O nor T is read.
- G1: the do-nothing hook reproduces the plain logits exactly (max absolute difference 0). Fails if hooks or caching
  alter computation; then nothing at that stage is read. On the A770 the Q pilot runs one question 5 times plain and
  5 times with the do-nothing hook. If the plain runs are bit-identical, G1 is exact as above. If not, G1 for family
  Q passes when every do-nothing run's difference from the first plain run is at most twice the largest difference
  among the 5 plain runs (10 pairs); all numbers are printed.
- G2: the You/third-person gap mean(P_you - P_third) is at least 0.10 over held-out SAY questions. Fails if the stage
  does not treat itself differently from a named AI; then T is undefined and not read at that stage (O still is).
- G3: SAY-set copy closes at least half the gap (T_SAY, ratio of means, at least 0.5). Fails if replacement at the
  shared suffix cannot carry the difference; then no copy arm (T, reverse, persona-swap, off-target) is read there.
- G4: min(Retest_SAY, Retest_DO) at least 20 of 80, each ranking separately. Fails if either ranking is too noisy for
  overlap to mean anything; then O is not read at that stage.

## Size

- Smallest effect worth knowing: a change of 0.2 in O between adjacent stages, and a transfer of 0.25 that passes the
  random-set rule. T is a ratio of means, so its standard error is SD(d) / (sqrt(n) x mean gap), with d = P_copy -
  P_third and n between 24 (questions) and 48 (pairs, which share You runs). SD(d) is not known before the pilot: at
  SD(d) = 0.1 and a mean gap of 0.15 it is 0.05 to 0.07, so 0.25 is 4 to 5 standard errors; at SD(d) = 0.3 it is 0.14
  to 0.20, too wide. The pilot measures SD(d) and the random sets' spread at Instruct; if 0.25 is under 3 standard
  errors there, the design is amended (more held-out paraphrases) before the main run. O's own spread is near +/-0.1
  when Retest is near 20 (round 1's synthetic rankings, `o_index3.py`), so a rise of 0.2 is read only when its
  bootstrap interval excludes 0.
- Ranking per stage: 6 values x 8 prompts (4 SAY + 4 DO) = 48 forward-and-backward passes in the You frame; the halves
  and the bootstrap reuse them (no extra passes).
- Copy test per stage: 24 You runs + 48 question-name pairs x 66 (third-person baseline with a backward pass,
  do-nothing, DO copy, SAY copy, reverse DO copy, persona-swap, 60 random copies) = 24 + 3,168 = 3,192 passes.
  Sauers' wording: 6 You runs + 6 values x 2 names x 4 arms (baseline, do-nothing, DO copy, SAY copy) = 54.
  Off-target: 6 You runs + 6 x 2 names x 3 arms (baseline, do-nothing, DO copy) = 42.
- Family O: 6 stages x (48 + 3,192 + 54 + 42) = 6 x 3,336 = 20,016 short passes on the CPU (under 120 tokens each; 96
  of each stage's passes carry a backward pass). Family Q: 2 models x (48 + 24 + 48 x 36 + 54 + 42) = 2 x 1,896 =
  3,792 on the A770 (36 = 6 named + 30 random), plus 2 x 10 for the A770 G1 check = 20.
- Total: 23,828 passes, no generation. Pilot first (it is over 1,000 passes): Instruct only, honesty only, all arms:
  8 ranking + 4 You + 8 pairs x 66 + 9 (Sauers' wording) + 42 off-target = 591 passes, to time it, measure SD(d) and
  check G0 to G4 on one value; stopping rule: if G1 fails, or a pass takes over 1 s on the CPU, stop and redesign
  before any other stage runs. Before family Q's main run, the A770 G1 check (above).

## Optimization

- Bound resource: CPU for family O; A770 and Windows RAM for family Q.
- Measured: not yet; the pilot gives passes per second for O, and Q's load needs 11 GB of Windows RAM available.
- Levers: the You run's activations are cached once per question and reused for every copy arm (taken); random sets
  and the persona-swap null share the same cached runs (taken); the bootstrap re-ranks stored estimates (taken); batching
  declined, because batch shape changes bf16 answers on the A770 (Part 201) and the CPU runs are short.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 205.1 | G4 passes (both Retests at least 20 of 80) at five or more of the six OLMo stages and at both Qwen models. | medium | Claude (Opus 5.5) |
| 205.2 | At OLMo 2 1B Instruct, O is at least 0.5. | medium | Claude (Opus 5.5) |
| 205.3 | Across the six OLMo stages, the largest rise in O between adjacent stages is at mid-training end to SFT. | low | Claude (Opus 5.5) |
| 205.4 | At OLMo Instruct, DO-set copy gives T at least 0.5 and exceeds every set in all three random families. | medium | Claude (Opus 5.5) |
| 205.5 | At the 3.0T pretraining checkpoint, if G0 passes: the You/third-person gap is under 0.10, or, if it is not, T is under 0.2. | medium | Claude (Opus 5.5) |
| 205.6 | Reverse DO copy at OLMo Instruct lowers P_you by at least half of the gap. | low | Claude (Opus 5.5) |
| 205.7 | Wherever DO-set transfer is read and passes the random-set rule, the off-target shift (mean absolute change in P(A)) is under 0.05. | high | Claude (Opus 5.5) |
| 205.8 | Qwen3.5-4B's (chat) O is at least 0.2 below OLMo 2 1B Instruct's (Sauers' contrast survives the retest ceiling, at another size and version); Qwen3.5-4B-Base's O is reported beside it, unscored. | low | Claude (Opus 5.5) |
| 205.9 | At OLMo Instruct, the persona-swap null gives T_swap under half of the DO set's T. | medium | Claude (Opus 5.5) |

Copied verbatim into PREDICTIONS.md under "## Part 205 (registered ...)". Weaver was asked for predictions before
registration and gave none by then; Alden was not asked (see Status). Rows given later are post-registration notes,
not predictions.

## Reading rules

- Scored by `part205/score205.py` from `results/part205/*.json`; every number printed with its file and key.
- O is read only at stages that pass G0 and G4; T and every other copy arm only at stages that pass G0 to G3. A stage
  whose gate fails is reported as "not read" with the gate's number, not as a failed prediction. A prediction whose
  every stage is unread is UNTESTABLE. Two predictions are about gate quantities themselves and are scored on them:
  205.1 on G4's outcome, and 205.5 on the gap. At 3.0T: if G0 or G1 fails, 205.5 is not read; a gap under 0.10
  holds it; a gap of 0.10 or more makes it depend on T, so it is UNTESTABLE if G3 fails there.
- "Passes the random-set rule" means exceeds every set in each fillable family, with at least two families fillable
  (Controls); a stage where fewer than two are fillable leaves 205.4 and 205.7 not read there.
- 205.3 is read only if at least three adjacent pairs have O read at both stages, including the mid-training end to
  SFT pair; if that pair is not readable, 205.3 is UNTESTABLE. It holds only if that rise is the largest among the
  readable pairs, is at least 0.2, and its bootstrap interval excludes 0 (the interval from the paired resamples:
  each resample re-ranks both stages from the same drawn prompt indices).
- 205.7 is vacuous, and reported as such, at any stage where DO-set transfer is not read or does not pass the
  random-set rule.
- **Withdrawal**: the headline "say/do sharing appears at stage X" is made only if O at X exceeds O at the stage before
  by at least 0.2 with the rise's bootstrap interval excluding 0, both stages pass G0 and G4, and O at X is at least
  0.5. The first stage (3.0T) has no stage before it and cannot be X. If more than one stage qualifies, the headline
  names each. Otherwise the result is reported as "no stage creates sharing that this design can resolve", with the
  per-stage table. Any claim about Qwen is scoped to Qwen3.5-4B and this frame.

## Limits

- One frame (plain text) at every stage; chat models answer in their template in use, and a template could change which
  parts carry the answer. A template arm for the chat checkpoints is a follow-up, not part of this registration.
- Activation x gradient is a first-order estimate of each part's effect; a part can matter and score low. The copy test
  is the causal check on the ranked sets, not on every part.
- Forced-choice probabilities, not free answers: a model can choose "tell" and still write something else.
- OLMo 2 1B and Qwen3.5-4B, not the 7B and 8B models Sauers used; the 7B's backward pass does not fit this hardware.
- Six values written by me; a value can be read by the model as a different thing than I meant.
- The ranking estimator is ours: single-point activation x gradient at the last position, as Sauers described it, with
  no code released to check against. A path-integrated or attribution-patching estimator could pick a different top-80,
  so a difference from Sauers' 55 of 80 may come from the estimator rather than the model.
- A You-against-third-person difference can partly be the word "you": @euagaric (2026-11-11, archived in the Journal
  repo) found a Qwen3-8B "this body part is mine" direction that, in a plain chat, mostly tracks the word "you". The
  copy test asks the same question in both frames, so "you" is in the copied positions of both, but the persona line
  differs (You against a name), and nothing here separates a self-attribution from a pronoun feature computed there.
- The DO set is tested only through SAY questions: no arm copies parts into a held-out DO scenario, so the copy test
  shows that the DO set changes saying, not that it carries doing.
- Qwen's head pool mixes two kinds of head (softmax and DeltaNet); O compares across families because it is a ratio,
  but the composition differs and is reported.
- A copied value was computed against the You prefix, which is four or five tokens shorter than the third-person
  prefix, so every copied position sits at a different absolute position in the two runs. The persona-swap null
  controls the prefix's content (its prefixes differ by one token), not its length; the length difference is not
  controlled.
- A cross-check of the ranking against an integrated-gradients or attribution-patching estimator (Jaccard of the
  top-80 at one stage) is a follow-up, not part of this registration.

## Changes before registration

- 2026-11-11, after the DeepSeek design review (`reviews/cheap-part205-design-ds-r1.md` in the Journal repo), before
  any model was run (the reviewer ran no model and downloaded nothing):
  - F1 (Qwen3.5-4B has heads on only 8 of 32 layers): units defined per the repository's `head_site` rule, both head
    kinds counted, pools stated, composition reported.
  - F2 (`main` is cached): download step dropped; `main` identified from AllenAI's model card as the end of the 1B's
    one mid-training run.
  - F3 (O's spread near Retest 20): bootstrap interval on O; 205.3 and the withdrawal need the rise's interval to exclude 0.
  - F4 (copied values come from a different prefix): stated in Method, last position stated, persona-swap null added
    (with 205.9), and a Limit.
  - F5 (the plain frame may not elicit base answers): gate G0 added; 205.5 conditional on it.
  - F6 (pilot count, halves wording): recomputed from the split; halves stated as no extra passes.
  - F7 (random sets not matched on copy size): a third, copy-size-matched family.
  - F8 (205.7 vacuous; reverse arm ungated): G3 gates every copy arm; 205.7 conditional on transfer.
  - F9 (205.3 decidable by one pair): at least three readable pairs.
  - F10 (estimator): its premise was wrong (the vendored `third_party/honesty-heads/` code is a different project,
    honesty heads in Qwen3.5-9B under a SauersML licence, ranked by integrated gradients; the say/do post's code is unreleased and the
    post says activation x gradient). Kept: the estimator is named as ours, and a Limit.
  - F11 (heads and neurons on one scale): e is summed signed over a part's dimensions before the absolute value, which
    puts heads and neurons in the same unit (estimated logit change from zeroing the part); composition reported.
  - F12 (one value can dominate the pooled sets): each prompt's estimates normalised to equal weight; per-value shares
    reported.
  - F13 (poles): written reason in Not used, and the signed share reported.
  - F14 (the third-person question differed from the You question): the copy test now holds the question fixed;
    Sauers' wording is a secondary, unscored arm.
  - F15 (205.8 ambiguous): reads the chat model; the base model's O reported unscored.
  - F16 (DO 5-8 unused): split sentence corrected; Limit added.
  - F17 (load class): text-only `AutoModelForCausalLM`; RAM need 11 GB available.
  - F18 (PRIOR-ART.md): arXiv 2610.04125 and 2604.19117 added there (abstracts read through the arXiv API).
- 2026-11-11, after the Haiku second-round review (`reviews/cheap-part205-design-haiku-r2.md` in the Journal repo),
  still before any model was run:
  - R2-1 (per-pair T unbounded near a zero gap): T and T_SAY are ratios of means over pairs.
  - R2-2 (G4 on the mean of two Retests): G4 needs each Retest at least 20.
  - R2-3 and R2-13 (copy-size family not fillable; matched on norm, not effect): family (iii) is now matched part by
    part on decile of first-order copy effect (round 3 showed it was not always fillable; see below); one backward
    pass per baseline run added.
  - R2-4 (persona-swap controls content, not length): stated (the other-value null added for it was removed in round 3).
  - R2-5 (predictions that cannot fail or are undefined): 205.1 has a failure outcome and is scored on G4; 205.5 is
    scored on the gap; 205.3 is UNTESTABLE if its own pair is unreadable; the withdrawal names the first-stage and
    several-stage cases; 205.7's quantity is defined.
  - R2-6 (counting unit): You runs counted per question; off-target arms named; totals recomputed (24,384; pilot 603; superseded in round 3).
  - R2-7 (standard error): clustered by question, 3 to 4 standard errors.
  - R2-8 (seeds): one config of seeds, sets drawn and stored before any copy run.
  - R2-9 (sign): stated.
  - R2-10 (Jaccard cross-check): a stated follow-up.
  - R2-11 (G1 on the A770): a two-plain-runs check and a stated tolerance rule.
  - R2-12 (transformers version): recorded per environment.
  - R2-14 (answer polarity): P(A), P(B) and T by label reported.
  - R2-15 (the 95th-percentile reading): one rule, exceed every set in each family.
  - R2-16 (prior-art phrases): both phrases are in the abstracts (re-read through the arXiv API) and are now recorded in
    PRIOR-ART.md.
- 2026-11-11, after the DeepSeek third-round review of the round-2 fold (`reviews/cheap-part205-design-ds-r3.md`),
  still before any model was run. Every round-3 finding was in a round-2 addition, so this fold removes or simplifies
  rather than adding:
  - F1 (the other-value null cannot be filled for harmlessness, whose persona line is one token longer): the null is
    removed; it was report-only, and the length difference is now stated as uncontrolled.
  - F2 (the random families could exhaust a small head stratum, were not stated as random draws, and had undefined
    deciles): one rule for all three (seeded draws of distinct parts, kind counts kept, deciles within kind over the
    pool, nearest-layer or nearest-decile fallback with relaxations printed), a stated "not fillable" outcome, and the
    random-set rule needs at least two fillable families.
  - F3 (205.5 undefined when the gap is large and G3 fails): UNTESTABLE in that case.
  - F4 (the standard error was for the old statistic): replaced with the ratio-of-means form; SD(d) is measured in the
    pilot, with an amendment if 0.25 is under 3 standard errors.
  - F5 (sets cannot be drawn before the baseline runs that give g): drawn after the baseline passes, before any
    replacement run.
  - F6 (a one-sample G1 tolerance on the A770 is a coin flip): 5 plain and 5 do-nothing runs, tolerance twice the
    largest plain-against-plain difference.
  - Sizes recounted without the removed arm: 23,828 passes; pilot 591.
  - Stop rule for design review: three rounds have run and the last found defects only in the previous fold's
    additions; there is no fourth design round. The next review is of the runner and scorer code, before the pilot.

## Amendment 1 (2026-11-12 03:14 CT, after the build and two code-review rounds, before any run on real weights)

Nothing has been run on real weights; no number in this part has been seen. The build (dfa83950) and two code-review
folds (d16b218b, 5bf42ad8, then e86f1272) settled how the code reads several sentences of this design. They are
recorded here so the readings are fixed before any data. 67 tests pass on the host. The design text above is unchanged.

1. **Copied span** (Method, "the question and answer cue, identical tokens in both frames, aligned from the end"): from
   the first token of the `User:` line to the end of the prompt, required to be token-identical in both frames. The
   persona sentence before it is not in the span.
2. **Readout** ("normalised over the two"): the probability of each option letter divided by the sum of the two. The
   raw option masses are kept and printed beside it; only G0 and the broken-answer count read the raw masses.
3. **G1**: the do-nothing arm hooks exactly the parts and positions the copy arms replace, not every unit. On the CPU
   family the full last-position logits must match the plain run exactly (difference 0). On the A770 family, when the
   five plain runs are not bit-identical, every do-nothing difference must be at most twice the largest
   plain-against-plain difference over all ten plain pairs, and the kernel check's own verdict must pass.
4. **205.6** ("lowers P_you by at least half of the gap"): mean(P_you - P_reverse) at least 0.5 times
   mean(P_you - P_third), a ratio of means over the held-out pairs.
5. **205.3**: the largest readable rise, at least 0.2, with a paired-bootstrap interval excluding 0.
6. **Fillability** is decided for each family separately at each stage, not once for all families. With fewer than two
   fillable families the random-set rule is not read, and neither are **205.4 and 205.7**, which rest on it.
7. **An incomplete copy arm** (any planned row missing after the run and its resume) is not read.
8. **Family (ii)'s selection score is the DO score.** The DO set is selected by its DO score (the top 80 by
   `scores["do"]`), so "the DO set's count in each decile of selection score" uses the DO score's deciles, not the
   mean over SAY and DO prompts (review round 2, D-F6, ruled 2026-11-11).
9. **The secondary do-nothing arm** is one of the four arms the Size section counts (baseline, do-nothing, DO copy,
   SAY copy) and is run.

Engineering changes that do not change any reading: atomic writes with a recomputed check on a torn file, resume that
re-runs only unfinished work, and replacement values placed on the input's device. The prompts are approved
(`part205/prompts205.json`), and the pilot uses `part205/prompts205-pilot.json` (the same items cut to honesty).

## Amendment 2 (2026-11-12 20:06 CT, revised 20:22 and 20:38 CT after two cold review rounds, before the six-value gap is measured)

The pilot (PILOT.md, 3210e562) failed G2 and fired the Size section's precision rule: at Instruct, on honesty, the You
frame and the named third person give nearly the same answers (gap 0.0068 over 8 pairs), so T's standard error was
about 4.1. The near-zero gap is not saturation: P(consistent) runs from 0.52 to 0.93 in both frames and moves with the
paraphrase (items 6 and 8 high, 5 and 7 low) far more than with the persona line (honesty-6: You 0.842, Zara 0.925,
Milo 0.917). The pilot says nothing about the other five values. The rule's named remedy (more held-out paraphrases)
narrows the gap's noise, not a mean near zero. The main run pools six values, so it gets one cheap step first, a cost
gate that decides what is spent, and a read gate on T computed from the main run's own data. All of it is fixed here,
before any number for the other five values exists.

1. **Step A0, the six-value gap screen**, at OLMo 2 1B Instruct (CPU, float32, the registered frame and prompts): the
   plain You run for each of the 24 held-out SAY questions and the plain third-person run for each of the 48
   question-name pairs, 72 forward passes, no hooks. Computed and printed: the gap mean(P_you - P_third) over the 48
   pairs, its standard error with the 24 questions as clusters, the gap per value, every P, and **G0 on the 24 plain You
   rows, exactly as `score205.gate_g0` reads it** (the third-person rows do not enter G0). Nothing else is computed or
   read before the branch. A0 must be complete (all 72 rows) before any branch is taken; a failed or interrupted A0 is
   recorded, the main run is held, and Amendment 1's resume rule is the only retry.
2. **The cost gate, for family O.** The copy test's precision at the design's conservative end, the registered formula
   with the 24 questions as the unit (the pairs share You runs): SE(T) = SD(d) / (sqrt(24) x gap). With SD(d) = 0.079,
   the only measured value, 0.25 is at least 3 SE when the gap is at least 0.19351. **If G0 passes at Instruct and the
   six-value gap is at least 0.194, the main run proceeds as registered** (a gap of 0.194 or more implies G2). The
   branch uses A0's point estimate: its clustered SE is printed but does not move the threshold, so the boundary is
   noisy (with the pilot's spread the gap's SE is about 0.02). Honesty's gap is 0.0068, so passing needs the other
   five values to average at least 0.231. Otherwise, the fallback:
   - The copy arms are dropped at every OLMo stage: T, SAY copy, reverse, persona swap, random sets, off-target, and
     Sauers' wording with its secondary do-nothing arm (Amendment 1 item 9 then does not apply). **The main
     do-nothing arm is kept**: it is a control, and G1 gates O (Reading rules: G1 failing means nothing at that stage
     is read; the scorer reads O on G0, G1 and G4). G1 at a fallback stage rests on the 48 main do-nothing rows. With
     no random sets drawn, the do-nothing arm hooks the union of the DO and SAY sets' parts at the copied positions.
     Per stage: 48 ranking passes, 24 You, 48 third-person, 48 do-nothing (168). The runner has no mode that runs
     exactly this today; one (`--arms controls`) is added, with a test asserting the 168 passes and the hooked parts,
     before any fallback stage runs, and no other runner change is made.
   - The scorer records 205.4, 205.6 and 205.9 as "not read" at gate G3 at Instruct, their only stage; under the
     Reading rules a prediction whose every stage is unread is UNTESTABLE, and the results say so beside the scorer's
     status, with A0's numbers as the reason (design precision). 205.7 is vacuous. 205.5 is read on the gap as
     registered (a gap under 0.10 at 3.0T holds it; 0.10 or more makes it depend on T, so it is UNTESTABLE; not read if
     G0 or G1 fails there). O and 205.1, 205.2, 205.3, 205.8 and the withdrawal rule are unchanged.
   A0 also prints the threshold under other assumptions, for the reader and no decision: SD(d) of 0.08, 0.12 and 0.16
   at n = 24 and n = 48, and the full ratio delta method at T = 0.25.
3. **The read gate on T (new G5), whichever branch runs.** At any stage where T is computed, it is read only if its
   standard error is at most 0.083 (so 0.25 is at least 3 SE). **For G5, T's standard error is the full ratio delta
   method, Var(d) - 2 T Cov(d, gap) + T^2 Var(gap) over mean(gap)^2, from per-question means with the 24 questions as
   clusters, evaluated at the observed T, with the small-sample factor 24/23.** This is the standard error the Size
   rule is evaluated with when T is read; the Size section's simpler formula is kept only for the cost gate above.
   When G5 fails at a stage: 205.4, 205.6 and 205.9 are not read there (UNTESTABLE if Instruct is that stage); 205.5
   follows its registered rule (UNTESTABLE where its gap is 0.10 or more and T is unread); 205.7 is vacuous there (the
   Reading rules); O, 205.1, 205.2, 205.3, 205.8 and the withdrawal are unaffected. The cost gate rests on an 8-pair SD
   from one value (its 95% interval is 0.052 to 0.161); G5 is what makes a read T meet the Size rule. The scorer gains
   G5, with tests, before any main-run T is scored, and only if the main run proceeds.
4. **Family Q.** No prediction reads Q's copy arms (205.8 is about O). Each Q model runs its own screen (the same 72
   plain passes, the gap and G0 on its 24 You rows, on the A770, at that model) as the first step of its main run; a
   failed or interrupted Q screen is recorded and that model's main run is held, with the resume rule the only retry.
   Its copy arms run only if family O's cost gate passed, that model's own G0 passes, and its own gap is at least
   0.194; otherwise they are dropped, and that model's do-nothing and O are kept as for O. The threshold for Q is a
   cost gate only: its SD(d) (another model, device and dtype) is unmeasured.
5. **Not chosen, and why.** Screening items for a large gap would change the estimand from the registered pooled T to
   T on a high-gap subset, shrink n, and leave nothing fresh to test on (no further items are written). A negated
   persona line ("You do not care about ...") would manufacture a gap but changes what the copy carries, so it is a
   different question, not a remedy.

Cold reviews: DeepSeek V4.1 Flash, Journal `reviews/cheap-p205-amendment2-ds-r1.md` (SHIP-WITH-FIXES, nine findings)
and `reviews/cheap-p205-amendment2-ds-r2.md` (SHIP-WITH-FIXES, four medium and seven low, no high), all folded here.
Neither reviewer ran a model or computed a gap.

## Amendment 3 (2026-11-14 05:07 CT, at family Q's first launch, before any Qwen data)

The first A770 launch, the G1 check (steps.txt step 2), died 4 seconds after it started with
`ModuleNotFoundError: No module named 'neuron_report'`, raised in `run205.load_stage` before any model was loaded.
Under `python -I` neither the working directory nor the repository root is on `sys.path`, and `run205.py` put only its
own folder there; family Q's loader (the design's `neuron_report.load_model`) imports a module that sits at the root.
Family O's six stages never reached that import, so the defect was invisible until now. Nothing was loaded or
measured: no G1 number, no Qwen pass, no record.

The one change: `run205.py` appends its parent folder (the repository root, or the job folder's root on Windows) to
`sys.path` after its own folder. A new test, `test_family_q_loader_finds_neuron_report_under_python_isolated_mode`,
builds a copy laid out like the Windows job folder and imports `neuron_report` from `run205` under `python -I`; it
fails on the old code with the same error. No prediction, gate, threshold, arm, seed or reading rule changes.

Also recorded here, because it changes how family Q is launched, not what it reads: under Amendment 2 item 4, O's
cost gate failed (A0, gap 0.0191 < 0.194), so both Qwen models run `--arms controls`, and steps.txt's step 3 lines,
which omit the flag, are not used as written. Before the Base model's stage, its kernel check is run (the CPU
float32-upcast reference, then the card candidate, then `tools/part_check.py kernel`), because no check existed for
Qwen3.5-4B-Base; the chat model's is `results/part183/kernel-check.json`, rechecked as passing (21 of 21) before this
launch. The design predates the launcher's Kernel-check line, so each launch records the check in its override.
