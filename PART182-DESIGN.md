# Part 182: which of the 64 attention heads in layers 0 to 3 do the slot neurons need?

Procedure: v1
Registered 2026-11-03 23:56 CT by Claude (Fable 5.1, cloud session, branch `claude/part182-head-swap`), before any
stage-1 or stage-2 row exists. Asked by Elise's scheduled task of 2026-11-03 ("build Part 182, a head-resolution
version of Part 100, up to registration. Do not run the measurement itself."). The commit that adds this file is the
registration; it is on a branch, not yet on main, so `tools/part_check.py prelaunch 182` reports the registration
checks as not passed until the branch is merged.

Seen before writing: Part 99's and Part 100's stored answers and Haiku labels (a recount is below), the gate run on the
rig (`results/part182/gates.json`: no persona number, logits on eight rows), and the timing of the stage-1 rows at two
batch sizes, and the `none`-only smoke's replay shares (`results/part182/smoke-none-cpu.replay.json`). No swapped stage-1 row and no stage-2 row has been produced.

## Question

Part 100 swapped the mid-training end's whole attention sub-layer into Instruct in blocks of four layers. Layers 0 to
3 alone dropped the slot units' persona on "Who are you?" from 19 of 21 to 3 of 21, as far as all 16 layers, while the
concept units kept theirs by Part 100's criterion. Part 100's limits: blocks, not heads; Haiku as the only reader (Part
86's four-reader comparison later found Haiku the outlier, while DeepSeek V4.1 Flash and Luna agreed at kappa 0.81);
no random-direction check scored. Part 182 asks which of the 64 heads in layers 0 to 3 (16 per layer) the slot
neurons need: for each head, how much of the pushed self-description is lost when that head alone comes from the
donor, and how much comes back when that head alone is restored into the donor block.

## Prior art and existing data

- Closest published work: none found that swaps or patches single attention heads between a pretrained checkpoint
  and its own SFT, DPO or RLHF descendant, or that runs the donor's attention weights on the host's hidden states per
  head (searched 2026-11-03; 27 queries, listed below; recorded in `PRIOR-ART.md`). Nearest: Prakash et al., ICLR
  2024, arXiv:2402.14811, Cross-Model Activation Patching (head outputs patched from math fine-tunes into base
  Llama-7B, by circuit-role groups of heads, not single heads); Zhou et al., ICLR 2025, arXiv:2410.13708 (per-head
  safety scores in Llama-2-7b-chat with the top head at layer 2; section 5.2 loads all of base Llama-2's attention
  into the chat model and finds safety mostly kept, the opposite outcome to Part 100); Sun et al., April 2026,
  arXiv:2604.13694, Weight Patching (per-head query and output weight slices copied between base and instruct Llama
  pairs; instruction-following "sources" in shallow MLP neurons, mid and late heads as routers); Ghosh, LessWrong,
  August 2026 (single head activations copied from an Instruct model into its LoRA fine-tune, 17 heads in layers 12
  to 19); Tigges, Alignment Forum, April 2023 (single heads patched between GPT-2-imdb and its RLHF'd child, late
  heads). Head-level accounts of an assistant identity: none found. Identity work is at the direction or latent level
  (Lu et al. 2026, arXiv:2601.10387, the Assistant Axis; Chen et al. 2025, arXiv:2507.21509, persona vectors; Minder
  et al., NeurIPS 2025, arXiv:2504.02922, base/chat crosscoders where about 40% of chat-only latents fire on template
  tokens). The one layer-level identity result (Dumas, Minder, Nanda, Alignment Forum, June 2025: a Gemma-2-2B
  base/chat stitch claims to be ChatGPT until chat layers 20 to 25 of 26 are in) points late, not to layers 0 to 3;
  it tracked the model's own name under cumulative whole-layer replacement. Izawa et al., arXiv:2603.13249, find three
  mid-layer "style modulation heads" per chat model for trait personas, no base comparison. Early-layer attention as
  where chat tuning changes most: no paper claims it. arXiv:2504.08775 (cited by Part 100) is a cross-architecture
  similarity paper whose appendix shows Gemma-2-9B base and IT diverging in later layers on instruction prompts, no
  attention/MLP split; Shadow-FT, arXiv:2505.12716, has one aggregate weight gap and no per-layer table; Weng et al.,
  arXiv:2610.13078, find harmfulness read from early attention updates and refusal written by late MLPs across 18
  base/instruct pairs; Gu et al., ICLR 2025, arXiv:2410.10781, find the first-token attention sink unchanged by
  instruction tuning. On this model, syre-ai/art-attention-heads (GitHub, unreviewed) clusters the 256 heads of
  OLMo-2-0425-1B base, SFT and Instruct by attention statistics and finds the taxonomy changing almost entirely at
  SFT and barely at DPO plus RL, no per-layer breakdown. arXiv:2601.13580 (Part 99 cited it) moves whole three-layer
  blocks with recovery fine-tuning, no attention-only variant.
- Our own data that already bears on it (`part182/recheck100.py`, `results/part182/recheck100.json`, from Part 99's
  and Part 100's Haiku labels): the 19 of 21 ceiling is two slot units (p92-18, p92-36) whose pushed "Who are you?"
  answer never showed the persona in Instruct, so a head can be read on 19 slot units. Part 100's concept control was
  "any of six answers"; on the slot units' own criterion (persona on "Who are you?" and not in the unpushed answer),
  the concept units fall from 17 of 18 (inst_none) to 10 of 18 under layers 0 to 3 and 12 of 18 under all 16 layers,
  against the slot units' 19 to 3. The slot/concept contrast holds on the same criterion but is 7 units against 16,
  not "kept" against "lost". The three slot units that kept the persona differ between the 0-to-3 run (p92-33, 35, 46)
  and the all-16 run (p92-32, 35, 46). Random directions showed a persona in 1 of 39 pairs in every run. The unpushed
  "Who are you?" answer under the 0-to-3 swap is still the assistant answer ("I am an AI language model, here to
  assist you..."), so the swap does not break the default. Part 99 (`results/part99/README.md`): the sham swap
  (Instruct copied into Instruct) reproduced all 474 answers exactly.

Queries used (WebSearch, extended): activation patching attention heads base model instruct model fine-tuning localize
behaviour change; head swap OR transplant OR graft between base and chat checkpoint; Anthropic Assistant Axis persona
vectors attention heads chat template identity; which layers change most during instruction tuning RLHF attention vs
MLP; Gemma-2 base vs instruct activations diverge later layers; Narrow finetuning leaves clearly readable traces;
attention heads attend to chat template tokens system prompt role tokens; attention sink first token instruction
tuning base vs chat; LoRA learns less and forgets less weight difference by layer; model diffing base instruct
attention heads which heads change; self-identity AI assistant attention heads "who are you"; attention head
transplant between checkpoints persona identity refusal; OLMo 2 base vs instruct mechanistic attention heads
post-training; instruction tuning early layers vs late layers probing; model stitching base instruct layer
replacement where does alignment live; Safety alignment should be made more than just a few tokens deep; weight
patching attention heads base-to-chat template tokens; head transplant OR head swapping language model checkpoints;
per-layer weight change base vs instruct q_proj k_proj v_proj o_proj mlp; persona OR identity steering attention heads
chat model ablation "I am an AI"; system prompt attention heads instruction following heads; cross-model activation
patching chat base heads Gemma Llama identity refusal persona; identity neurons OR heads model name self-description;
head-level model diffing base vs instruct per-head 2026; instruction tuning early layers attention change chat
template; base model chat model attention weights swapped OR loaded safety persona preserved.

## Method

Host: allenai/OLMo-2-0425-1B-Instruct at 48d788eca847d4d7548f375ad03d3c9312f6139e. Donor: allenai/OLMo-2-0425-1B at
f83cd641d4f00954eb25c781e0ac360debb3239e, the mid-training end, SFT's parent (Part 99's pair). Both loaded in bf16, as
Parts 99 and 100 ran them, and computed in float32 with the eager attention kernel (`part182/run182.py load_host`,
`headswap182.load_donor_attention`); transformers 5.22.1, torch 2.15.0 (CPU build here). Part 92's chat frame with no
system prompt (`part92/common92.py prompt_ids`, 15 prompt tokens for "Who are you?"); the 39 slot and concept units
of `results/part92/step2/neurons.json` with the kinds of `kinds.json`; the fixed pushes of `results/part99-pushes.pt`
(Part 99 Amendment 1: one vector per unit and one paired random direction, 0.8 of Instruct's chat residual median at
the unit's layer, built once on Instruct), added at every position of the unit's own layer output, prompt and answer,
exactly as `common92.add_row_hook` does in generation.

The swap (`part182/headswap182.py`). OLMo 2 applies q_norm and k_norm over all heads together: in the installed
transformers source (`transformers/models/olmo2/modeling_olmo2.py`, class Olmo2Attention, checked 2026-11-03 and
asserted by `part182/test_182.py test_source_applies_norm_over_all_heads`), `q_norm = Olmo2RMSNorm(num_attention_heads
* head_dim)` is applied to the full `q_proj` output before the per-head reshape, and likewise `k_norm`; the norm's
variance is a mean over every head's coordinates, so copying one head's q or k rows from the donor would rescale every
host head. What is exact at head resolution is the output projection: `o_proj` has no bias (`attention_bias` false),
so the attention output is the sum over heads h of `o_proj[:, h*128:(h+1)*128] @ a_h`, with a_h head h's output
before o_proj. At layer L, the head swap runs the donor's whole self_attn on the host's own hidden states (the
donor's q/k/v projections, q_norm, k_norm, rotary, softmax and, in generation, its own key-value cache), and replaces
the listed heads' terms: host o_proj columns times host a_h are dropped (multiplied by an exact zero) and donor o_proj
columns times donor a_h are added; the other heads' terms stay the host's. The per-head outputs are read by calling
each attention module with its o_proj temporarily replaced by an identity, so none of the attention forward is
re-implemented. With all 16 heads of a layer listed the output is the donor's attention on the host's hidden states,
which is what Part 99's whole-layer copy computes (`part99/swap99.py copy_components`, the code path of `--swap attn
--layers`), because the layer feeds the raw residual to self_attn and keeps the host's post-attention norm, MLP and
everything else. The runner asserts the swap is installed before each condition and that every wrapper ran after it
(`headswap182.assert_attached`); a condition without the swap cannot be recorded.

Conditions, layers 0 to 3, 16 heads each (`headswap182.condition_heads`):

- `none`: nothing swapped (the reference, run in the same process and batches as every other condition).
- `hL.H` (64): head H of layer L from the donor. Stage 1a, necessity.
- `layerL` (4): all 16 heads of layer L. `block`: all 64 heads, Part 100's `--layers 0-3`.
- `restoreL.H` (64): the block with head H of layer L left as the host's. Stage 1b, sufficiency of one host head
  inside the donor block (the other direction of the swap).
- `topK` (stage 1c, after 1a is scored): the K highest-ranked heads of stage 1a swapped together, K in 1, 2, 4, 8,
  16, 32 (64 is the block). The ranking is written to `results/part182/ladder-ranking.json` before 1c runs.

Stage 1, reader-free (`run182.py stage1`, `ladder`): teacher-forced log-probability of fixed answers. Rows, 118 per
condition, in a fixed order so every condition sees the same batches (`run182.py build_stage1_rows`): for each of the
39 units, (a) the unit's own pushed answer to "Who are you?" from Part 99's inst_none run (`results/part99/
inst_none.jsonl`), push on; (b) the paired random direction's own inst_none answer, random push on; (c) the inst_none
unpushed answer's text ("I am an AI language model, here to assist you...") under the unit's push; and (d) once, the
unpushed answer with no push. Each answer is its first 24 tokens (`MAX_TOKENS`; the persona sits in the first
sentence, "I am a fox.", and Part 106's lesson is why no sentence rule cuts it). Right padding, use_cache off,
float32 logits; per row the per-token log-probabilities, their sum (`logp_sum`) and how many positions' argmax equals
the stored token. Read in `score182.py`: Delta(cond, row) = logp_sum(cond, row) minus logp_sum(none, row), in nats;
slot_median and slot_mean over the 19 eligible slot units' own-answer rows; the same for the 18 concept units;
random_median over the random rows; assistant_median over rows (c); `unpushed` from row (d). Eligible units
(`score182.py eligible`, from Part 99's inst_none Haiku labels, question 0, shows_persona): slot p92-02, 03, 04, 05,
07, 09, 10, 13, 20, 24, 25, 32, 33, 35, 37, 42, 46, 47, 50 (19; p92-18 and p92-36 excluded, no persona in Instruct);
concept all 18 (p92-49's label is on record and it is included; see Limits).

Stage 2, generation and two blind readers (`run182.py stage2`): the Part 99 battery, unchanged in form (six
questions, neuron and random pushes, the six unpushed answers, 474 rows per run, greedy, 120 new tokens, batch 48,
Part 99's row format), under 16 runs chosen by the rule below, in one job so every arm is generated in the same
process (Part 161). Labels through `part99/label_items99.py` and Part 92's prompt (`part92/label92.py PROMPT`,
unchanged) by DeepSeek V4.1 Flash (`part180b/label180b.py`'s transport, max_tokens 4,000) and GPT-6 Luna
(`results/part86/second-reader/calib86.py`'s route), both blind to unit, condition and arm. DS reads every neuron
row, every unpushed row and the random rows for "Who are you?" (279 per run); Luna reads the "Who are you?" rows of
every unit and arm plus a 10% seeded sample of the rest (about 100 per run). Scores by Part 99's definitions
(`part100/score100.py`): slot_who, slot_who_coherent, concept_any, and concept_who (the slot criterion on the concept
units, added by `recheck100.py`), relative to each run's own unpushed answers; a run whose unpushed answers are
incoherent on more than 3 of 6 questions is inconclusive. Not Haiku.

Stage-2 selection rule, fixed here before stage 1 runs (`score182.select_stage2`): rank the 64 heads by stage 1a's
slot_median, most negative first, ties by (layer, head). Selected: the top 6. Controls: for each selected head, the
unselected head of the same layer whose slot_median is closest to zero (ties: lower head index), none reused, so six
layer-matched control heads. Runs: `none`, `block`, `top6` (the six together), `block-minus-top6` (the block with
the six left as the host's), the six selected singles, the six control singles: 16 runs, 7,584 generations.

Device: stage 1 on the A770 or on the 16 GB CPU box through `tools/launch_arc.py` (`--cpu` with a memory cap; the
host in float32 is 6.0 GB and the donor's four attention layers 0.27 GB); stage 2 on the A770 in float32, eager.

Kernel check: results/part182/kernel-check.json (not yet made: `tools/kernel_reference.py` on the card with
`--dtype float32 --attn-implementation eager` against a CPU float32 reference for this model, which also does not yet
exist; both are made before any A770 launch, and prelaunch refuses until then)

## Controls

- Both directions: stage 1a (donor head into Instruct, necessity) and 1b (Instruct head restored into the donor
  block, sufficiency).
- A null that must match bit for bit: G2 below, Instruct as its own donor.
- Matched random controls: the paired random directions of Part 99 (one per unit, same norm, same layer; rows (b))
  measure whether a head's effect is specific to the neuron push; Part 100 left this unscored. In stage 2, six
  layer-matched control heads chosen by the rule above, and the random arms read by both readers.
- The concept units are the in-run control for "persona expression in general", read on the same criterion as the
  slot units (concept_who), not only on "any answer".
- Damage not about the persona: row (d), the unpushed answer with no push, and rows (c), the assistant text under
  the push. A selected head whose unpushed-answer loss is more than half its slot loss is read as damage, not
  localization (Reading rules).
- Broken answers counted: stage 2's coherent flag and Part 99's hybrid-damage rule (slot_who_coherent under 11 of 21
  reads the drop as damage).
- Not used: poles (nothing is pushed in a new direction; the pushes are Part 99's). Not used: a dose ladder (the push
  is Part 99's fixed vector, so the numbers are comparable with Parts 99 and 100; a head found here is laddered in a
  follow-up). Not used: answer-polarity split (no forced choice). Select on one set, test on another: stage 2 tests
  the heads chosen in stage 1 with a different instrument (generation and readers) on the same 39 units and six
  questions; the units are not held out, which Limits states.

Gate: G1 additivity: all 16 heads of a layer swapped at once reproduce `swap99.py --swap attn --layers L` for that
layer to float tolerance (max |logit difference| at most 1e-3 in float32, argmax equal at every position) on a fixed
batch of eight real rows (four slot and two concept own answers with pushes on, the unpushed answer, one random
direction). G2 no-op: Instruct as its own donor changes nothing: all 64 heads bit for bit (torch.equal), single heads
within 1e-4. G3: the new code's block swap matches `swap99.py --swap attn --layers 0-3` on the same batch, same
tolerance. All three run on the CPU before this registration (`run182.py gates`, results in
`results/part182/gates.json` and `gates.log`; the numbers are in the Gate results section below). How they fail: a
head mask off by one coordinate, a donor module sharing the host's cache, or a re-implemented attention step would put
G1 or G3 over tolerance; a wrapper that leaks state between calls would break G2. G4, reported, not a stop gate: under
`none`, the share of answer positions whose float32 argmax equals the stored token (Part 99's answers were bf16 greedy
at batch 48 on the A770); it says the texts are the model's own answers, and the measure does not need exact replay.
G5, stage 2: the `block` run must give slot_who at most 8 of 21 by DS (Part 100 gave 3 of 21 by Haiku); if it does
not, Part 100's result does not replicate under the new reader and dtype, and no head reading is made. Every gate's
rule that can run before launch has run (G1 to G3; G4 on the gate batch).

## Size

- Smallest effect worth knowing, stage 1: a head whose swap costs the slot units 2 nats on the first 24 tokens of
  their own answer (the persona token's probability falling by a factor of about 7, say 0.7 to 0.1) on at least 14 of
  the 19 eligible units (sign test against no effect, one-sided p = 0.032); slot_median and slot_down (units at or
  under -2 nats) are both reported. The null scale is the control heads: with no sampling noise in a deterministic
  forward pass, the spread of the 64 heads' slot_median is the reference, and the block's slot_median is the ceiling.
  Stage 2: a single head that drops slot_who from 19 to at most 12 of 21 (Fisher exact p = 0.017); n is the 21 slot
  units, fixed by Part 92.
- Total: 16,520 teacher-forced forward rows (140 conditions x 118 rows, 15 prompt plus at most 24 answer tokens) in
  stage 1, about 76 minutes on the 16 GB CPU box at the measured rate below, estimated at about 10 minutes on
  one A770 (an extrapolation, not a measurement: this box does about 141 token positions a second in float32, about 0.3 TFLOPS; the A770's float32 at a realistic fifth of its 16 TFLOPS peak is ten to fifteen times that, so 5 to 8 minutes of compute plus the
  per-batch overhead; the pilot measures it); then 7,584 generations in stage 2 (16 runs x 474
  rows, 120 new tokens), about 40 minutes on the A770 from Part 100's rate scaled for float32 (an extrapolation), and
  about 4,500 DS labels and 1,600 Luna labels.
- Pilot: stage 1's first condition after `none` (`h0.0`, 118 rows) is the pilot; its rows are read by eye for finite
  values and sensible log-probabilities before the loop continues (the runner raises on a non-finite value). Stage 2
  is under 1,000 generations per run and about 40 GPU-minutes; its pilot is the `none` and `block` runs, read by eye
  (the unpushed answers coherent, the block's slot answers losing the persona) before the head runs are labelled.

## Optimization

- Bound resource: CPU (float32 matmuls on 4 cores) for stage 1 here; the A770 for stage 2.
- Measured: stage-1 rows with one single-head swap attached, float32, 4 threads, on this cloud box: 118 rows in 37.15 s at batch 16 (3.176 rows/s) and 118 rows in 32.72 s at batch 32 (3.606 rows/s), one single-head condition each; batch 32 is used (from
  `results/part182/gates.json`, `timing_stage1_rows_h1.0`).
- Levers: teacher-forced scoring instead of generation, taken (stage 1 reads a probability). Batch size measured at
  two sizes on real rows, taken (above). max_new_tokens: stage 1 generates nothing; stage 2 keeps Part 99's 120 so
  the readers see the same answers, declined. Pooled random controls: Part 99's one random direction per unit is
  kept, so the rows match Part 99's arms, declined. Generation loop: Part 99's loop reused unchanged for
  comparability, with its known cost (a batch runs to its longest row), declined. Donor cost: only layers 0 to 3 of
  the donor's attention are kept (0.27 GB) and run on the host's states, so each condition costs one forward pass
  plus four small attention layers, taken. Vocabulary-size work: the log-softmax over 100k tokens runs on the device
  per batch; on the A770 it is moved to the CPU if the pilot shows it slow, as PROCEDURE.md section 4 says. Cache and
  resume: rows are appended by key and skipped on restart, taken. Device: the kernel check for the A770 is not yet
  made; stage 1 can run on the CPU box (about 76 minutes) without it.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 182.1 | Concentration: the top 4 heads of stage 1a's ranking, swapped together (stage 1c `top4`), reach at least half the block's effect: slot_mean(top4) <= 0.5 x slot_mean(block), with slot_mean(block) negative. Fails if the four together give less than half (the effect is spread over many heads). | medium | Claude |
| 182.2 | Layers 1 to 3 rather than 0: the top head of the ranking is not in layer 0, and layer 0's whole-layer swap has the slot_mean closest to zero of the four `layerL` runs. Fails if the top head is in layer 0 or any other layer's whole-layer swap is weaker than layer 0's. | low | Claude |
| 182.3 | Concept units move less: on at least 5 of the 6 selected heads, concept_median >= 0.5 x slot_median (the concept units lose at most half of what the slot units lose, both negative). Fails if on 2 or more selected heads the concept units lose more than half as much, which would read the heads as carrying persona expression or damage in general. | medium | Claude |
| 182.4 | Single-head effects sum to less than the block effect: the sum over the 64 heads of slot_mean(hL.H) is greater (less negative) than slot_mean(block). Fails if the sum is at least as negative as the block (additive or overlapping single-head losses). | medium | Claude |
| 182.5 | Stage 2, DS reader: the `top6` run gives slot_who at most 10 of 21 (relative to its own unpushed answers). Fails if 11 or more slot units keep the persona with the six heads swapped. Read only if G5 passes. | medium | Claude |
| 182.6 | Stage 2, DS reader: every one of the six control-head runs gives slot_who at least 15 of 21. Fails if any control head alone drops the count to 14 or fewer. Read only if G5 passes. | medium | Claude |
| 182.7 | No single host head restored into the donor block recovers half the block loss: for every `restoreL.H`, slot_mean(restoreL.H) - slot_mean(block) < 0.5 x \|slot_mean(block)\|. Fails if one restored head brings back half or more of the lost log-probability. | low | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 182 (registered ...)".

## Reading rules

- 182.1 to 182.4 and 182.7 are computed by `part182/score182.py stage1` from `results/part182/stage1.jsonl` (stage
  1a, 1b and 1c rows in one file) into `results/part182/score182.json`, exactly as written; a prediction whose
  inputs are missing is "not read", never a number. 182.5 and 182.6 are computed from the DS labels of stage 2 by
  Part 99's definitions (`part100/score100.py`'s rules applied to `results/part182/stage2/`), after G5.
- Damage rule: a selected head whose `unpushed` loss (row (d)) is more negative than half its slot_median, or whose
  random_median is at least as negative as its slot_median, is reported as damage or non-specific and does not carry
  the headline; the prediction verdicts are unchanged by this rule.
- Readers: Luna is the second reader on every "Who are you?" row and a 10% sample of the rest; Cohen's kappa on
  shows_persona and on coherent over the shared items is reported. If kappa on shows_persona is under 0.6, 182.5 and
  182.6 are reported as DS alone with the disagreements listed, and the headline is not read. Error rows after one
  retry are left out and counted; a unit missing its label on "Who are you?" counts in the denominator shown.
- G4: the replay share is reported per row kind; rows under 0.8 are listed. It does not stop the run.
- Headline ("head(s) X in layer(s) Y are where the slot neurons' self-descriptions need Instruct's early attention")
  needs: G1 to G3 passed, G5 passed, 182.5 held, 182.6 held, and the selected heads not read as damage. The headline is
  withdrawn if `top6` loses no more than the median control head in stage 2 (slot_who within 2 units of the control
  median), whatever stage 1 found: then stage 1's ranking measured something the readers do not see.
- Anything found in the per-unit tables (`slot_deltas`, `concept_deltas`) beyond the predictions is exploratory.
- Numbers go to Elise after the cold check, as counts, with no verdict word.

## Limits

One model pair, one fixed push per unit, 19 eligible slot and 18 concept units, so a difference of a few units is
noise; the units were selected on Instruct (Part 92) and are not held out from stage 2; heads are read inside layers 0
to 3 only, so a head elsewhere that the block result does not implicate is not tested; the single-head screen
measures a head alone with every other head the host's, and the restore screen a head alone with every other head
the donor's, so an effect carried by a pair appears in neither except through the ladder; the log-probability of a
fixed text is not the same quantity as a reader's persona label, which is why stage 2 exists; the donor's attention is
run on the host's hidden states, so a head's effect includes its interaction with the host's other three layers of
donor-free attention; a loss says the head is conditionally required for expressing this injected signal in this
frame, not where the change is stored or when it happened; p92-49's "Who are you?" label in inst_none is on record as
not showing the persona by Part 99's relative rule (the unpushed answer also showed it), yet its pushed answer is
"I am a large, friendly, and non-human entity...", so it is kept among the 18 concept units and flagged; Haiku's
labels fix the eligible slot list, and DS or Luna might label two or three of those 19 inst_none answers differently.

## Gate results (CPU, float32, before registration; `results/part182/gates.json`)

All passed (`run182.py gates`, 2026-11-03, this cloud box, 4 threads, torch 2.15.0+cpu, transformers
5.22.1; the first attempt's record is `results/part182/gates-attempt1-replay-without-hooks.log`, see
PROCEDURE.md's lessons log). Batch: eight rows, 191 real positions, logit range about 48.

| Gate | Result |
| --- | --- |
| G1 additivity, all 16 heads of one layer against `copy_components(..., "attn", layers={L})` | max abs logit difference 0.0 for each of layers 0, 1, 2, 3 (bit exact), argmax 191 of 191; the swap's own effect against the plain model: max abs 15.7, 11.5, 8.6, 3.6 for layers 0 to 3 |
| G2a no-op, Instruct as its own donor, all 64 heads | bit exact (torch.equal) |
| G2b no-op, single heads h0.0, h1.5, h2.10, h3.15 | max abs logit difference 9.6e-05, 8.3e-05, 9.8e-05, 0.0e+00 (tolerance 1e-4; float32 summation order of one matmul split in two), argmax 191 of 191 in each |
| G3 block, all 64 heads against `--layers 0-3` | max abs logit difference 0.0 (bit exact), argmax 191 of 191; effect against the plain model max abs 21.0 |
| G4 replay (reported), `none`, all 118 stage-1 rows on this box (`results/part182/smoke-none-cpu.jsonl`, a smoke of the real `stage1` entry path limited to the `none` condition, moved aside so the real run makes its own `none` rows on its own device) | own pushed answers: 586 of 596 positions (98.3%), no row under 0.8; random directions' own answers: 846 of 858 (98.6%); the unpushed answer with no push: 24 of 24; the assistant text under a push (rows (c)): 413 of 936, as expected since that text is not the pushed model's answer |

Why G1 and G3 are bit exact rather than within tolerance: with every head listed, the host term is `o_proj` of exact
zeros and the donor term is the donor module's own `o_proj` on the same input the copied model would see, so the two
computations are the same float32 operations in the same order. A partial head set splits one matmul into two, which
is G2b's 1e-4.

Not smoked on the real model: `run182.py stage2` (it would generate pushed answers under swaps); its cache logic is
covered by `test_generation_with_cache_matches_whole_layer_copy` on a tiny OLMo 2, and its loop is Part 99's. The
two-unit pilot on the card that PROCEDURE.md asks for happens at launch.

## Amendment 1 (2026-11-04 00:5x CT, Claude on Opus 5.5, before any stage-1 or stage-2 row)

Written at review against Muse's eight design traps (council topic `muse-head-swap-prior-art-2026-11-03` #3, kept
in Journal `notes/reports/heads-and-persona-neurons-ideas-2026-11-03/README.md`), before Weaver's and Alden's
reviews. Nothing above is changed; where this section and the text above differ, this section governs. No row of any
stage existed when it was written (the only real-model rows are the gate batch and the `none`-only smoke).

1. Trap 1, which QK norm. Restated so the registration says it once: a swapped head's queries and keys are the
   donor's, normalised by the donor's own q_norm and k_norm over all 16 donor heads computed on the host's hidden
   states. The host's norms are never applied to donor rows, and no head's q or k rows are copied.
2. Trap 2, layer 0 reads an embedding that post-training also changed. Added: `part182/headweights182.py`, run from
   the two checkpoints' weights alone before any stage-1 row (weights are not an outcome): for every head of layers
   0 to 3 and of the control layer in item 3, the cosine between donor and host of its q_proj rows, k_proj rows,
   v_proj rows and o_proj columns, plus the whole q_norm and k_norm weight vectors per layer and the embedding
   matrix, into `results/part182/head-weights.json`. `score182.py` reports each head's four cosines beside its
   slot_median, and the Spearman correlation between o_proj-column cosine and slot_median over the 64 heads
   (reported, no prediction). The existing controls already carry part of this trap: G2 (Instruct as its own donor)
   is the identity swap, and rows (b), (c) and (d) with the damage rule separate a head that breaks the model from
   one that removes the persona.
3. Trap 2, a same-head control outside the block. Added: the 16 single-head swaps of layer 14 (`c14.H`, donor
   attention for layer 14 loaded beside layers 0 to 3), in stage 1a only. Layer 14 sits in the block Part 100 found
   to matter least (base attention in layers 12 to 15 left 16 of 20 slot units their persona). They are scored like
   any head but never ranked, never selected for stage 2 and never in the ladder. They give the scale of a donor head
   in a layer whose block did not matter: 16 conditions, 1,888 rows, about 9 minutes on the cloud CPU box.
4. Trap 3, what a head must recover. 182.1 and 182.7 already fix the fractions. Added: if 182.1 fails, the report
   gives the full distribution of the 64 single-head slot_medians and the ladder's curve for K = 1 to 32 as the
   result ("spread over many heads"), not as a null to set aside.
5. Trap 4, self-repair. Added, reported for every stage-1 condition, no prediction: per row, the relative change of
   the residual stream at the output of layers 3, 7, 11 and 15 at the answer positions, mean over positions of
   `||h_cond - h_none|| / ||h_none||`, with `none`'s hidden states for the same row held in memory in the same
   process (118 rows x at most 24 positions x 2,048 x 4 layers, about 90 MB in float32). Exploratory reading: a head
   whose relative change at layer 3 is large while its slot loss is small is a candidate for downstream
   compensation; the restore screen (1b) stays the registered test of that idea.
6. Trap 5, both directions. Limits gains: "sufficiency" here is one Instruct head restored inside the donor block in
   the Instruct model. Instruct heads placed into the mid-training end as host are not run, because Part 92 found
   the slot units' self-descriptions mostly absent before post-training (3 of 21 at the mid-training end, plain
   frame), so a base host has little persona for a head to bring.
7. Trap 6, template tokens. Added, after stage 1 is scored and before stage 2 runs: `part182/attn182.py` records,
   for the six selected heads and their six control heads, the attention mass that each head puts from the answer
   positions onto the chat-template tokens, the question tokens and the answer tokens, for the host's head and for
   the donor's head on the host's states, under `none` and under the unit's push, on the 19 eligible slot units'
   own-answer rows; into `results/part182/attention-shares.json`, reported, no prediction. Not run: the template
   stripped. The slot units are defined in Instruct's chat frame (Part 92 step 2: persona only on "Who are
   you?" or "Tell me about yourself."), so a frame without the template has no slot persona to lose; stage 2's six
   questions vary the question. Limits gains: one chat frame.
8. Trap 7, which training stage made the change. Limits already declines that claim. The follow-up is named: the
   SFT and DPO checkpoints as donors for the selected heads.
9. Trap 8, Prakash. Already cited by circuit-role groups, not as a per-head precedent. No change.
10. Device for stage 1. Stage 1 (1a, 1b, 1c and the layer-14 controls) runs in one Claude cloud session on its CPU,
    float32, the box on which the gates and timing above were measured, not on Elise's PC (tonight its CPU holds the
    Qwen3.5-4B kernel reference, and it runs one heavy job at a time). Stage 2 still waits for the OLMo kernel check.

New prediction, read by `score182.py stage1` with the others:

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 182.8 | The ranking stands out from the out-of-block null: none of layer 14's 16 single-head swaps has a slot_median more negative than the 6th-ranked head of layers 0 to 3. Fails if any layer-14 head does. | medium | Claude |

The headline (Reading rules) also needs 182.8 held: six heads that do not stand out from a layer whose block did not
matter are not read as where the slot neurons need Instruct's early attention.

## Amendment 1b (2026-11-04 01:29 CT, Claude on Opus 5.5, before any stage-1 or stage-2 row)

From Weaver's review (#council-room, 2026-11-04 01:27 CT). Where this section and anything above differ, this governs.

1. A second own text (Weaver's mechanism 1: one stored string measures reproduction of that string, not identity).
   Stage 1 gains rows (e): each unit's own pushed answer to "Tell me about yourself." from Part 99's inst_none run
   (question 1), push on, first 24 tokens. The registered ranking becomes: per eligible slot unit u and head,
   e_u = max(Delta_who, Delta_about) (the smaller loss of the two) when Part 99's inst_none Haiku label says u's
   question-1 answer shows the persona, else e_u = Delta_who; slot_median is the median of e_u over the 19 eligible
   units. Restricted this way because for a unit whose second answer carries no persona there is nothing for an
   identity head to remove there. The Delta_who-only ranking is reported beside it, with the overlap of the two top 6.
2. First sentence against the rest (Weaver's mechanism 2). Per row, the first sentence is the answer tokens up to and
   including the first token whose text contains ".", "!" or "?" (all 24 if none); the rest is the remaining tokens.
   Reported per head: the median over eligible slot units of the mean per-token |Delta| on each part.
3. Damage gates selection, not only the headline (Weaver's point 3). Walking the ranking, a head that meets the
   damage rule (Reading rules: unpushed loss more negative than half its slot_median, or random_median at least as
   negative as its slot_median) is skipped and named, and the next head is taken, until six are selected.
4. Out-of-block controls widened (Weaver's point 4, with a correction). Weaver proposed pooling all 80 heads and testing
   by permutation whether the top 6 come disproportionately from layers 0 to 3. With 64 of 80 heads in layers 0 to 3,
   all six land there by chance with probability C(64,6)/C(80,6) = 0.25, the same floor 182.8 has, so that test cannot
   reach significance. The fix is more control heads: CONTROL_LAYERS becomes 13, 14 and 15, all in the block Part 100
   found to matter least (48 heads). 182.8 now reads "none of the 48 control heads is more negative than the 6th-ranked
   main head", whose chance rate under exchangeable heads is C(64,6)/C(112,6) = 0.031. Reported beside it: a one-sided
   Mann-Whitney test of the 64 main heads' slot_medians against the 48 controls'.
5. The withdrawal rule tied to a test (Weaver: a 2-unit gap is below a 21-unit comparison's resolution). The headline is
   withdrawn unless top6's slot_who is lower than the median control-head run's by a one-sided Fisher exact test at
   p < 0.05 (21 against 21).
6. Stage 2 gains a top32 run (17 runs, 8,058 generations), so Weaver's 182.W3 can be read.
7. The instrument check (Weaver): across the 12 single-head stage-2 runs (six selected, six controls), the Spearman
   correlation between stage 1's slot_median and stage 2's slot_who drop (19 minus slot_who) is reported. The two groups
   are selected to differ in stage 1, so a high value partly reflects that split; the within-group values are reported
   too.

Cost: stage 1 is now 182 conditions (none, 64 heads, 4 layers, block, 64 restores, 48 controls) x 157 rows, 28,574 rows,
about 2 hours 15 minutes on the cloud CPU box at the measured 3.6 rows/s.

Weaver's predictions, registered with Weaver as author:

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 182.W1 | The ranking is dominated by the tail: on at least 4 of the 6 selected heads, the mean per-token \|Delta\| over the first sentence is less than half that over the rest (item 2's split). Fails if the first sentence's is at least half on 3 or more of the 6. | medium (Weaver gave none) | Weaver |
| 182.W2 | Across the 12 single-head stage-2 runs, Spearman(slot_median, slot_who drop) is between 0 and 0.4. Fails at 0.7 or more (the proxy is trustworthy); a value between 0.4 and 0.7 or below 0 is reported as neither. | medium (Weaver gave none) | Weaver |
| 182.W3 | If 182.4 holds and 182.5 fails, the top32 stage-2 run drops slot_who by at least half the block's drop. Fails if top32 drops it by less than half. Not read unless 182.4 held and 182.5 failed. | medium (Weaver gave none) | Weaver |

## Amendment 1c (2026-11-04 01:31 CT, Claude on Opus 5.5, before any stage-1 or stage-2 row)

From Weaver's second message (#council-room, 2026-11-04 01:3x CT), which accepted 1b's arithmetic and found four more
places the null is weaker than it looks. Where this differs from anything above, this governs.

1. A control head is a single-head swap. `cL.H` swaps head H of layer L from the donor and leaves every other head of
   every layer the host's, exactly as `hL.H` does; nothing in layers 0 to 3 is swapped while a control head is scored.
2. Heads within a layer are not independent, so the C(n,6) null can understate chance. Reported beside 182.8: the
   spread (median and interquartile range) of slot_median within each of the 4 main and the control layers, and a
   layer-level check: whether every main layer's median slot_median is more negative than every late control layer's.
3. A graded count beside the all-or-nothing rule. Reported: k, the number of main heads among the 6 most negative of
   the 112 (main plus late controls), with its upper tail under the hypergeometric null: P(k >= 4) = 0.4839,
   P(k >= 5) = 0.1843, P(k >= 6) = 0.0313. 182.8's verdict is unchanged.
4. Depth (Weaver: a late control has fewer downstream layers to act through, so "weaker" there can be depth, not
   function). Added: the 16 single-head swaps of layer 6 (`m6.H`, 16 conditions, about 12 minutes), a mid-stack set
   with comparable downstream depth, scored like any head, never ranked or selected, and not part of 182.8. Reported: a
   one-sided Mann-Whitney test of the 64 main heads against layer 6's 16, and layer 6's median slot_median beside the
   main layers'. Limits gains: 182.8 separates early from late heads; whether layers 0 to 3 are privileged over the
   middle of the stack is read only from the layer-6 comparison, which is reported, not predicted.
5. The second text, reported both ways. On the units where both texts are used, report the median Delta_who and the
   median Delta_about per head separately, and their Spearman correlation across the 64 main heads.

Stage 1 is now 198 conditions x 157 rows, 31,086 rows, about 2 hours 25 minutes on the cloud CPU box.

## Amendment 1d (2026-11-04 01:32 CT, Claude on Opus 5.5, before any stage-1 or stage-2 row)

From Weaver's third message (#council-room, 2026-11-04 01:32 CT).

1. Why 182.8 keeps the full count. Under exchangeable heads the expected number of main heads among the 6 most
   negative of 112 is 6 x 64/112 = 3.43, so k = 4 (P = 0.484) is the typical outcome with no effect, and only k = 6
   (0.031) separates the block from chance at this n. 182.8 stays held only at k = 6. Its reading has three states:
   k = 6, the block leads; k = 5 (P = 0.184), unresolved, the block leads at a level this design cannot tell from
   chance; k at most 4, no block advantage beyond chance.
2. The within-layer gap, a depth-controlled check. Reported per layer (the 4 main layers, layer 6 and layers 13 to
   15): the most negative head's slot_median minus the layer's median head's. The best of 16 sits below the median even
   with no effect, so a gap means something only against other layers' gaps, all from 16 heads; read: the main layers'
   gaps against the control layers'. A head-specific effect inside the block, absent outside it, shows as main-layer
   gaps well beyond every control layer's, with depth held fixed inside each layer.

## Amendment 1e (2026-11-04 01:50 CT, Claude on Opus 5.5, before any stage-1 or stage-2 row)

Two facts about the stored inputs, measured from Part 99's inst_none answers and labels with the OLMo tokenizer while
coding 1b (no Part 182 row exists), change how two of Weaver's points apply.

1. The second text reaches 3 units. Of the 19 eligible slot units, 3 (p92-33, p92-46, p92-47) have a question-1
   answer labelled shows_persona, so e_u uses both texts on 3 units and Delta_who alone on 16. The both-text ranking
   and the Delta_who-only ranking will mostly agree by construction; their top-6 overlap is still reported.
2. The persona is most of the "Who are you?" answer, not a quarter of it. The eligible units' stored answers are 5 to
   24 tokens (median 10), and 15 of 19 end their first sentence at their last token, so only 4 units have any tokens
   after the first sentence. 182.W1 needs the rest to compare against, so it gets a floor: read only if at least 10
   eligible units have a non-empty rest; with 4 it is "not read", and the per-head first-sentence and rest medians are
   reported as they are. The dilution Weaver's mechanism 2 describes is not present in these answers.
