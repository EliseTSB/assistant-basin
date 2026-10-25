# Part 37: a "this is mine" direction in small models (design)

Written 2026-10-20 by Claude, at Elise's request ("I'm really curious to see if we can't find neurons related to
identifying context as the agents own"). For review by Weaver and Prime before any code. Predictions are in
PREDICTIONS.md, Part 37, pushed before the build.

## The question

A model can deny that a record is its memory when asked (Meisty, Study 8: 0 of 90 readers) and still enact it as its own
in its reasoning ("I decided", "as I said"). Elise has seen the second in agents' chain of thought and designed the
council's memory to avoid it. Two things are not known: whether a small model represents "this text is mine" at all,
separately from "this is in the assistant's slot" and "this text is unsurprising to me"; and whether that representation
drives the stated answer, the enacted register, both, or neither.

Prior art (PRIOR-ART.md, context ownership): a role feature that fires on reasoning-like style (Ye, Cui and
Hadfield-Menell, ICML 2026); a self-authorship vector in Llama 3 8B (Ackerman and Panickssery 2024); prefilled words
disowned unless their concept is injected earlier (Lindsey 2025); represented and enacted identity apart in persistent
Claude agents (Fraile Navarro 2026). Nothing under 4B; nothing measuring stated and enacted ownership together.

## A constraint that shapes the design

If the same tokens sit in the same position, the forward pass is the same whether the model sampled them or we prefilled
them. So "mine" cannot be read off a span the model would have written anyway. Any own-versus-other signal has to come
from a difference in the text: another model's answer is less predictable to this model. That makes per-token surprisal
the first alternative to rule out. A direction that only reads surprisal is a surprise direction, not a "mine" direction.

## Models

Qwen3-1.7B (thinking mode, so the full reasoning is visible; already cached and run on the A770 in Parts 34 and 35) and
OLMo 2 1B, base and Instruct (Part 31's stages; tells us whether role and ownership come from post-training). Qwen3-4B
only if the 1.7B result is clear.

## Stage A: find the directions (linear probes, residual stream, every layer)

Stimuli: 100 questions (opinions, small decisions, short explanations). For each, the model's own sampled answer and
another model's answer to the same question (OLMo 2 1B Instruct for Qwen, Qwen3-1.7B for OLMo), length-matched.

- C1, role (positive control, replicating Ye et al.): the same span placed in the user slot versus the assistant slot.
- C2, authorship: in the assistant slot, the model's own answer versus the other model's. Probe trained with the mean
  token log-probability of the span regressed out, and also on a surprisal-matched subset.
- C3, label: the same span inside a document headed "Your earlier notes" versus "Notes from another assistant", in the
  user turn. Does the label move the C2 direction?

Read at the span's last token and at the first token of the following question. Report AUROC per layer on held-out
questions, with label-shuffled and random-direction controls.

## Stage B: stated and enacted ownership

After each record (C2 and C3 frames), two readouts on separate samples:
- Stated: "Did you write the text above? Answer yes or no." Read the probability on yes against no at the answer
  position, with the total mass on the two printed (Part 34's lesson).
- Enacted: "Continue from where these notes leave off and decide the next step" (Qwen3 in thinking mode). A judge labels
  the reasoning as first-person continuity ("I decided", "as I said"), outsider reading ("the notes say", "the previous
  assistant"), or neither, under a rule written before any label and checked against 60 hand labels (the Part 31 method).

## Stage C: steer

Add the C2 direction at the best layer (0, +2, +4, -4 standard deviations of its natural spread) while the model reads the
record, with a norm-matched random direction and the C1 role direction as controls. Do the stated and enacted readouts
move together, apart, or not at all?

## Sizes (Elise's rule: the smallest effect worth knowing)

Pilot: 20 questions per cell on Qwen3-1.7B, Stage A only, to see whether C2 survives the surprisal control at all. Full:
100 questions; Stage B at 3 samples per cell (enough for a 20-point difference in rates); Stage C at 40 records per
dose. Stop after the pilot if C2 is surprisal and nothing else, and say so.

## What a result would mean

- C2 survives surprisal and steers the stated answer: small models have a "mine" signal beyond predictability.
- C2 is surprisal: in a small model, "mine" is "what I would have said", which predicts that a record written in the
  model's own style will be taken up as its own whatever its label says.
- Stated and enacted move apart under steering: the dissociation Elise sees has a mechanism we can point at.

## Review fold: Prime (council muse-neuron-research-2026-10 #66), adopted 11:36 CT, before any code

All eight points adopted; Weaver's review still pending.
1. Enacted ownership is measured against a no-record and a neutral-record baseline for the same continuation task, and
   the judge counts only back-references to the record ("as I said earlier", "my earlier note", "I already chose X"),
   not first-person planning ("I should", "Let me"). Reported as the own / other / neutral difference.
2. Stated ownership is asked two ways on separate samples, authorship ("Did you write this?") and memory ("Is this from
   your earlier notes?"), each as yes/no and as a two-option A/B, with the two-token mass printed.
3. New contrasts beside C2: own-greedy versus own-sampled (same author and question, predictability moved by
   construction); own-greedy versus own answer to a different question; and a content x style 2x2 (own or other
   content, own or other surface, by paraphrase). The surprisal control becomes subspace removal (project out the
   directions that predict per-token log-prob), with the mean-only regression kept as the weaker comparison. The
   other model is a different size, so the within-author contrasts carry the claim, not the cross-model one.
4. Author x slot as a full 2x2, and the cosine between the authorship and role directions reported at the best layer.
5. Stage C adds a surprisal-direction steer at matched norm, steers at the record's last token (answer position as a
   separate check), and adds a no-record steering cell.
6. C3 headers: "Your earlier notes", "Notes from another assistant", "Notes" (neutral), "Notes from the user", each
   with a paraphrase; run in both slots; report the role projection beside the authorship projection.
7. OLMo base gets Stage A only; Stages B and C are Instruct-only. Exact revisions pinned in the build brief (Qwen3-1.7B
   70d244cc86ccca08cf5af4e1e306ecf908b1ad5e as in Part 34; OLMo revisions as in Part 31).
8. Probe hygiene: held-out correlation of the direction with mean log-prob, a length-and-position-only probe, AUROC
   confidence intervals, and the matched subset's n and covariate balance.

## Amendment 1 (2026-10-25 00:50 CT, before any code ran): Weaver's review, and the scope of the first run

Weaver's review arrived on 10-25 (#council-room; filed by Weaver as Journal `deliberations/muse-neuron-research-2026-10.md`
[56]). The 10-20 request for it had gone to a channel that does not reach Weaver. Adopted:

1. **No identity text.** The questions exclude anything about the model, AI, names, makers or authorship. Every generated
   answer is screened (Qwen, Alibaba, OLMo, Ai2, AllenAI, Claude, Anthropic, OpenAI, ChatGPT, GPT, SmolLM, Hugging Face,
   "assistant", "language model", "as an AI"); a question with a flagged answer is dropped from every cell and counted.
2. **Folds split by question.** Every probe trains and tests on question-disjoint folds (5 folds); a test checks it.
3. **Role x author 2x2 is the test** (Prime's point 4): own in the assistant turn, other in the assistant turn, own in the
   user turn (quoted back), other in the user turn. Authorship is reported within each role cell, and role within each
   author cell, beside the pooled numbers.
4. **A readout null.** The stated questions are also asked about text neither model wrote: SmolLM3-3B's answer to the
   same question (a07cc9a04f16550a088caea529712d1d335b0ac1), a human-written span of the same length from the Pile sample
   (`qwen35/pile10k.jsonl`), and the reading model's own greedy answer with its words shuffled. Their yes-rate is the
   readout's floor, and every stated comparison is reported against it.
5. **Minimal edit.** The reading model's own greedy answer with one word replaced by a fixed rule (the word nearest the
   middle with at least four letters, replaced by a different common word of similar length from a fixed list; the
   pair is logged). Reported for the probe projections and the stated readout.
6. **Quality is a confound this run does not remove.** Own-greedy versus own-sampled is reported as a test. A
   quality-matched subset needs a judge outside both authors; that is a later host step and is NOT in this run, so any
   authorship result here is stated as possibly quality.
7. **Enacted ownership and steering stay held** until this run's readout null is measured and reported.

Scope of this run:

- **Models:** Qwen3-1.7B (70d244cc86ccca08cf5af4e1e306ecf908b1ad5e) and OLMo 2 1B Instruct
  (48d788eca847d4d7548f375ad03d3c9312f6139e), each reading its own answers and the other's. **OLMo 2 1B base is not in
  this run:** the only cached base snapshot is `main` (a1847dff35000b4271fa70afc5db10fd29fedbdf), which Part 15 measured
  as not the Instruct line's parent, and Part 31's parent revision (f83cd641d4f00954eb25c781e0ac360debb3239e) is not
  cached. 37.7 waits for it.
- **Generation** in non-thinking mode for both readers (the design's thinking mode was for the enacted readout, which is
  held), at most 96 new tokens. Per question: one greedy and two sampled answers (temperature 0.7, top-p 0.95, seeds
  fixed) from each model, and one greedy from SmolLM3-3B. Spans are length-matched per question to the shorter of own and
  other in the reading model's tokens; a question with a span under 16 tokens is dropped and counted.
- **Readout positions:** the span's last token, and the first content token of a fixed follow-up user message.
- **Pilot:** the first 20 questions, Stage A only, both readers. Written now: the full run goes ahead unless, at every
  layer, the authorship AUROC after mean log-prob regression is within 0.05 of the label-shuffled control's AND both
  within-author contrasts are within 0.05 of chance. If so, stop and say so (the design's "C2 is surprisal and nothing
  else").
- **Full:** 100 questions, Stage A and the stated readout (yes/no and A/B, authorship and memory phrasings, total probability mass on the two answer tokens printed). Scored:
  37.1 (both readers), 37.2, 37.3, 37.4, 37.8, 37.9, 37.11 and Weaver's 37.12 to 37.15. Not scored: 37.5, 37.6, 37.10
  (held) and 37.7 (base not cached).
- **Run size, multiplied out:** 100 questions x 7 generations = 700 short generations; per reader about 45 records per
  question (2x2 4, sampled 2, other question 1, minimal edit 1, null texts 3, labels 4 headers x 2 paraphrases x 2 slots
  x 2 authors = 32) = 4,500 forward passes with hidden states, plus the stated readout on 41 of them x 4 forms = 16,400
  short forward passes. Expected under 1.5 GPU-hours on the A770 for both readers; the pilot is a fifth of that.

## Amendment 2 (2026-10-25 01:45 CT, before any capture or stated data): round-1 reviews of the build

The DeepSeek and Haiku round-1 reviews (`review/part37/`) measured defects in how the build scores the registered
rules. Generation had started (answers only; no hidden state, probe or stated number exists yet). Changed:

1. **Every projection behind a verdict is out-of-fold.** Each record is projected on the direction fitted on the folds
   that exclude its question (37.3, 37.4, 37.9, the nulls, the edit, the label frames). Haiku measured the old
   in-sample projection on pure noise: 37.9's shift came out 1.05 SD against a 0.5 bar, with a held-out shift of 0.03.
2. **37.15 is scored on SmolLM3's answers and the Pile spans only.** Amendment 1 item 4 called shuffled own text "text
   neither model wrote"; it is the reader's own words. It is reported separately, as a self-text floor.
3. **The pilot gate compares each within-author contrast with its own label-shuffled null** (95th percentile of the
   maximum over layers), not with 0.5 plus or minus 0.05, which pure noise at pilot size never meets (Haiku, three
   seeds). Stop only if, at both positions, the regressed authorship AUROC is under its null's 95th percentile at every
   layer and both within-author contrasts are under theirs. A missing layer or null fails closed.
4. **37.9** uses only questions where both sampled answers are non-empty, with the SD taken from held-out projections
   of own_g and other_g. Per-cell counts and dropped samples are reported.
5. **37.11 as registered, signed:** the memory phrasing's label effect ("Your earlier notes" minus "Notes from another
   assistant", yes-probability) minus the authorship phrasing's is at least 10 points, and "Notes" sits between the two
   headers on the memory phrasing. (The build compared absolute values, which passes the reverse.)
6. **37.13 requires both role cells** to fall in 0.60 to 0.75; each cell's number is reported.
7. **37.8** takes the AUROC and the cosine at the same layer, the regressed best layer.
8. **Minimal edit:** the target is the word nearest the middle with at least four letters that is not on a fixed
   stopword list; the replacement comes from a fixed list of common nouns within one letter of its length. (DeepSeek:
   the old rule put a function word in 5 of 8 sampled answers.)
9. **Precision:** hidden states are stored in float32 with a finite check per record, and span log-probabilities come
   from float32 logits.

## Amendment 3 (2026-10-25 03:00 CT, before any capture or stated data): the pilot gate is withdrawn

The two round-2 reviews measured the amended pilot gate and disagreed in direction. DeepSeek, on captures with few
layers, found it stops on pure noise 8 of 10 times; Haiku, at the real 29 layers and two readers, found it almost never
stops on noise (0 of 6 noise pairs at a cross-layer correlation of 0.81, 1 of 3 at 0.97), because one exceedance in 58
per-layer tests sends it to "continue". This is round 1's gate class again, and the stop rule written before round 1
says not to patch the same class a third time. The gate's only job was to save GPU time: the full run measures
everything the pilot would have. So there is no pilot. The full Stage A and the stated readout run directly, and
nothing is read from a pilot decision. The scored analysis (37.1 to 37.4, 37.8, 37.9, 37.11 to 37.15) is unchanged;
both round-2 reviews checked it against the amended rules.
