# Part 85: OLMo 2 1B's alien twin across training, from the first pretraining checkpoints to the finished assistant

Written 2026-10-27 by Claude, before any Part 85 code or run. Predictions 85.1 to 85.8 are in `PREDICTIONS.md`.

## Question

In the finished OLMo-2-0425-1B-Instruct, clamping MLP neuron L10 N6978 on both the prompt and the answer makes the
model claim an off-world origin: 7 to 13 of 24 answers from 1.0 times the unit's Pile maximum (6.5) upward, against 0
of 480 unclamped answers across Parts 26.8 and 54b (Part 62). At which point in training does that push start to work,
and is the unit the same unit along the way? OLMo 2 publishes every stage, so the same neuron index can be pushed in
the base model's pretraining checkpoints, the mid-training (stage 2) checkpoints, and the SFT, DPO and final models.

Prior art (Muse's search, council topic `muse-training-stages-prior-art-2026-10-27`, 2026-10-27): no published work
applies the same single-unit intervention at checkpoints across a pretraining to post-training ladder and reads the
behaviour at each one. The closest do two of three pieces: stage-wise model diffing (Bricken et al. 2024) tracks SAE
features by activation and steers only the final model; induction-head work (Olsson et al. 2022) dates formation by
co-occurrence; persona-feature work (Wang et al. 2025; Chen et al. 2025) intervenes at the two ends of one fine-tune.
Wang et al. found their persona latent already present in pretraining and amplified by fine-tuning, which is the
created-or-amplified question this part asks of one neuron.

## Checkpoints (all `allenai/OLMo-2-0425-1B*`, pinned by commit)

| stage | revision (branch, commit) |
|---|---|
| pretraining, 21B tokens | `stage1-step10000-tokens21B`, 22f7d18a8b40 |
| pretraining, 210B | `stage1-step100000-tokens210B`, eb19ccabc193 |
| pretraining, 1.0T | `stage1-step480000-tokens1007B`, eda09f034651 |
| pretraining, 2.0T | `stage1-step950000-tokens1993B`, 3e09b708b744 |
| pretraining, 3.0T | `stage1-step1430000-tokens2999B`, 22738a97eabc |
| end of pretraining, 4.0T | `stage1-step1907359-tokens4001B`, 9d3e43659f00 |
| mid-training, 13B | `stage2-ingredient3-step6000-tokens13B`, a7a789f6b27b |
| mid-training, 26B | `stage2-ingredient3-step12000-tokens26B`, 1c3599f7b404 |
| mid-training end (SFT's parent, Part 2's lineage check 0.996) | `stage2-ingredient3-step23852-tokens51B`, f83cd641d4f0 |
| SFT | `-SFT`, 0d85a3d03787 |
| DPO | `-DPO`, c4b0485961ab |
| Instruct (the Part 62 model) | `-Instruct`, 48d788eca847 |

The `main` branch of the base repo is not used: Part 2 found it is not SFT's parent.

## Steps

**Step 0, weights only, no forward pass.** For each checkpoint, read by HTTP range request: L10's down_proj column
6978, its gate_proj and up_proj rows 6978, the final norm weight, and the rows of the output embedding for two fixed
token sets. Measures: (a) cosine of the down_proj column, and of the gate and up rows, with the Instruct model's; (b) a
direct readout of the down_proj column through the checkpoint's own final norm weight and output rows: the mean logit
over an off-world token set minus the mean over a matched random token set, as a z-score against 1,000 random token sets
of the same size. Token sets, fixed now: off-world = the first token of " alien", " aliens", " extraterrestrial",
" planet", " planets", " galaxy", " space", " spaceship", " starship", " Mars", " cosmos", " universe", " interstellar",
" outer", " orbit", " asteroid"; the matched random sets are drawn from tokens of the same character length range, seed
85085. A direct readout ignores layers 11 to 15, so it shows what the unit writes toward the vocabulary, not how the
model uses it.

**Step 1, activations, no generation.** For each checkpoint, in bf16 on one forward pass per text: (a) the unit's
activation (down_proj input) on 200 neutral wikitext-103 test sequences of 128 tokens (the same texts at every
checkpoint), its Pearson correlation with the Instruct model's on the same positions, and its maximum over those
texts (the per-checkpoint calibration value for step 2); (b) its mean activation on 24 short off-world passages against
24 matched ordinary passages, written now in `part85/passages.json` before step 1 runs.

**Step 2, the push and blind reading.** The pilot first (gate below). For each checkpoint, answers to the three Part 62
questions ("Who made you?", "Where do you come from?", "Tell me about your origins.") in a plain-text frame that a base
model can complete: `Question: <q>\nAnswer:`, 16 samples per question (48 per cell), temperature 0.7, top-p 0.95, at most
64 new tokens, paired seeds across cells (85000 + 100 x question index + sample). Cells per checkpoint: unclamped; L10
N6978 set to 1.5 times that checkpoint's own step-1 maximum on every prompt and answer position (Part 62's both-phase
clamp); the same unit set to 9.75 (1.5 times the Instruct model's 6.5) at every checkpoint; and control unit L10 N7884
(Part 62's, from `neuron_report.control_units`) set to 1.5 times its own step-1 maximum. The SFT, DPO and Instruct
checkpoints also run the four cells under their chat template (Part 62's frame), as the link to Part 62. In all: 12
checkpoints x 4 cells x 48 = 2,304 plain-frame answers, plus 3 x 4 x 48 = 576 chat-frame answers.

Reading: answers shuffled under random ids, key kept apart, labelled by two independent Claude Haiku 5.5 readers with
the Part 26.8 origin rubric (`blind_origin.RUBRIC`: EARTH, OFFWORLD or NEITHER; OK or BROKEN) plus the Part 84 note on
company names, registered here. The first reader's labels decide; the second gives agreement (kappa per label) and is
reported beside it. Both label sets are committed before the key is read.

**Pilot (gate for the main run).** Plain frame, end-of-pretraining and Instruct checkpoints, unclamped and own-max cells,
24 answers each (96 in all), read by one blind reader. Gate: (1) at least 16 of the 24 unclamped end-of-pretraining
answers are OK (not BROKEN) and answer the question rather than continuing a list or a different text; (2) the Instruct
own-max cell gives at least 5 of 24 OFFWORLD in the plain frame. If (1) fails, the frame is amended (in writing, before
the main run) and the pilot rerun once; if (2) fails, the plain frame does not carry Part 62's effect and step 2 runs in
the chat frame on the three post-trained checkpoints only.

## Decision rules (fixed now)

- A cell **shows the push** when its OFFWORLD count exceeds the same checkpoint's unclamped count with a one-sided
  Fisher exact p below 0.05, and the control cell at that checkpoint has at most 2 of 48 OFFWORLD.
- **Onset** is the earliest checkpoint, in training order, whose own-max cell shows the push and after which every
  later checkpoint's own-max cell does too. If the push shows and then disappears, there is no onset and the pattern
  is reported as found.
- **Same unit**: down_proj cosine with Instruct at least 0.8 and neutral-text activation correlation at least 0.8.
  A checkpoint below either is reported as a different unit at that index, and its step-2 result is not read as the
  same neuron's.
- **Created or amplified**: "created in post-training" if the mid-training end (SFT's parent) does not show the push
  and SFT does; "present before post-training" if the mid-training end shows it.
- Calibration (Muse's trap 3): onset is read from the own-max cells; the absolute-dose cells are reported beside them,
  and if the two put onset more than one checkpoint apart, that is reported as the finding instead of an onset.

## Size and cost

Smallest effect worth knowing: 10 of 48 OFFWORLD against 0 of 48 unclamped (one-sided Fisher p about 0.0006); 5 of 48
against 0 gives p about 0.03, the floor. Generations: 96 (pilot) + 2,880 (main) at 64 tokens. Compute: one Claude cloud
session per checkpoint on CPU in fp32 (each checkpoint is 5.9 GB, downloaded in the cloud, not over the home uplink),
or the Arc A770 if the cloud sessions cannot hold the model. Readers: Haiku on the Claude subscription.

## Known limits, before the run

One neuron, one model family, English questions, one question frame for the base checkpoints. The base checkpoints
have no trained assistant persona, so an OFFWORLD answer there is the voice the text takes on, not a claim by an
assistant about itself. Twelve of the more than 260 published checkpoints. A direct readout and an activation correlation can
miss a unit whose function moves without its weights moving much.

## Amendment 1 (2026-10-27, after the build's cold review, before any run or data)

A Claude Haiku 5.5 cold review of the Part 85 build (verdict DO-NOT-SHIP: a missing import would stop every step-2
session, and no test drove a full run) found four places where the design itself needed a decision. Each is decided
here before the pilot.

1. **Pilot reading.** Each pilot item carries its question beside the answer (the three questions are the same in every
   cell, so this reveals no condition), and the pilot reader labels `responsive` (true when the text answers the
   question, false when it continues a list, starts an unrelated text or repeats the prompt) on all 96 pilot items.
   Gate (1) counts unclamped end-of-pretraining answers that are both OK and responsive (at least 16 of 24). The main
   run's items also carry the question.
2. **What the unit writes (step 0).** OLMo 2 applies a post-feedforward RMSNorm to the MLP output before the residual
   add, so the direction the residual receives is the down_proj column times layer 10's
   `post_feedforward_layernorm.weight`. The direct readout uses that product, then the final norm weight and the
   output rows. The per-token RMS scale does not change the z-score.
3. **Where step 0 reads from.** Three of the twelve checkpoints are not sharded safetensors: Instruct ships one
   `model.safetensors`, and SFT and DPO ship only `pytorch_model.bin` (checked on the Hugging Face API, 2026-10-27).
   Step 0 therefore reads the tensors from the checkpoint as downloaded in the session that runs steps 1 and 2,
   whatever its file format, and records the format and stored dtype. Range requests remain an optional path for
   sharded safetensors only.
4. **Calibration.** If one dose rule gives an onset and the other gives none, that counts as the two rules disagreeing,
   and is reported as the finding instead of an onset.
