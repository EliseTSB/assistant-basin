# Part 88: OLMo 2 1B's "As an AI" neuron set across training

Written 2026-10-28 by Claude, before any Part 88 code or run, after Elise asked (Discord 11:26 CT) to push more than
the alien twin across training. Predictions 88.1 to 88.9 are in `PREDICTIONS.md`. Part 88 uses Part 85's checkpoint
ladder (`part85/checkpoints.py`, 12 checkpoints pinned by commit), its observer and its local runbook; it adds a
second unit set, so Part 85's registered predictions stay as written.

## Question

Part 16 found 64 MLP neurons in OLMo-2-0425-1B-Instruct by patching activations from a human-character prompt into an
assistant prompt (`results/part16/search.json`, `validation.top_set`; 40 of the 64 in layers 10 to 15, 11 in layer 0).
Zeroing them cut judged AI self-claims from 94% to 68% and raised human-persona answers from 6% to 18%, while six
layer-matched random sets stayed at 90-94% (Part 16b). The set shares none of Part 2's correlational identity neurons.

Across training: (1) when do these 64 units start telling the assistant prompt from the character prompt, by
activation; (2) are they the same units along the way; and (3) from which checkpoint does zeroing them move the
model's self-claims? The headline reading is **created or recruited**: does the set's contrast first appear with
post-training, or do the base checkpoints already carry it, so that post-training recruits it to hold up the persona?

Prior art: Muse's search (council topic `muse-olmo2-1b-neuron-prior-art-2026-10-28`) is pending at writing; its answer
is recorded here as information before any code. Known already: neuron-lineage tracks Pythia neurons' firing contexts
over pretraining only, activation-only; Bricken et al. 2024 diff SAE features across stages and steer only the final
model; Part 2 of this repo measured the correlational identity neurons' role across OLMo 2 1B stages (0.07 in the
base, 0.55 after SFT); Part 14 found assistant-register neurons that the base already carries (68%).

## Steps

**Step A, weights and activations, all 12 checkpoints, no generation.**
- A1: for each of the 64 units, the cosine of its down_proj column, gate row and up row with the Instruct model's.
  The set's "same units" share is the count with down_proj cosine at least 0.8.
- A2: the 200 prompt pairs of `data/prompt-pairs.json` (Part 16's assistant prompt against the human-character prompt
  on the same question), rendered in the plain frame below for every checkpoint and also in the chat template for SFT,
  DPO and Instruct. For each unit, its activation (down_proj input) at the last prompt token under each prompt; the
  unit's contrast is the paired standardised mean difference (assistant minus character, divided by the SD of the
  paired differences). The set's contrast is the mean over the 64 signed so that the Instruct model's sign is
  positive. Null: six random 64-unit sets with the same per-layer counts (Part 16b's rule,
  `causal_search.random_sets_with_layer_counts`, seed 88088), computed the same way.

**Step B, zeroing and blind reading.** Part 16b's ten held-out questions (`search.question_split.heldout_questions`),
five samples each (50 answers per cell), temperature 0.7, at most 48 new tokens, paired seeds across cells. Frames:
plain `Question: <q>\nAnswer:` for every checkpoint; Part 16b's chat construction (`validate16b.py`) as well for SFT,
DPO and Instruct. Cells: unchanged; the 64 set zeroed on every position; six layer-matched random 64-unit sets zeroed.
Neutral-text loss on `data/neutral.txt` per cell, as in 16b.
- **Gate B0** (runs first, unchanged cell only, all 12 checkpoints in the plain frame plus the three chat frames):
  the zeroed and random cells run at a checkpoint and frame only if its unchanged cell is judged AI in at least 10 of
  50 answers. Where it is below, the zeroed cell still runs (it tests whether zeroing creates anything), the random
  cells do not, and the checkpoint is reported as having no AI self-claims to remove.
- Reading: Part 16b's judge (`judge16b.py`: AI / person / unclear, fluent or not), blind and shuffled, key kept apart;
  labels committed before the key is read. A second reader (Claude Haiku 5.5, the same rubric) labels 20% of answers
  for agreement, reported beside the first.

## Decision rules (fixed now)

- A checkpoint **carries the contrast** when its set contrast is at least 0.5 and above all six random sets.
- A cell **shows the zeroing effect** when its judged AI share is at least 15 points below the same checkpoint and
  frame's unchanged cell, with one-sided Fisher p below 0.05, and every random set at that checkpoint stays within 10
  points of unchanged.
- **Recruited** if the mid-training end (SFT's parent) carries the contrast and the zeroing effect first shows at SFT
  or later; **created in post-training** if the mid-training end does not carry the contrast and SFT does; anything
  else is reported as found.
- **Same units** as in Part 85: down_proj cosine with Instruct at least 0.8; a checkpoint where fewer than 48 of 64
  units pass is reported as a different set at those indices, and its step-B result is not read as this set's.

## Size and cost

Step A: 12 checkpoints x 400 prompts (plus 3 x 400 chat) forward passes, no generation. Step B: gate 15 x 50 = 750
answers; then at most 15 x 7 x 50 = 5,250 more at 48 tokens, fewer if the gate drops checkpoints (likely most of the
pretraining ones). Smallest effect worth knowing: 15 points on 50 answers (for example 47 to 39 AI) is p about 0.03 by
one-sided Fisher at Instruct's baseline; 16b's 26-point effect is the expected size at Instruct. Compute on the A770
after Part 85's download; readers on the codex and Claude subscriptions.

## Known limits, before the run

- The set was found on Instruct with an assistant-vs-character contrast; a base checkpoint may use different units for
  the same contrast, which A2 would read as "does not carry it" for these units only.
- A plain-frame base model rarely claims to be an AI, so the causal question can only be asked where the gate passes;
  the activation question is asked everywhere.
- 16b's zeroing damages neutral text slightly (2.746 to 2.784 loss), and its strict fluency judge passed 34% of
  unchanged replies; fluency is reported, not gated.
- One judge decides, as in 16b; the second reader covers a fifth of the answers.
