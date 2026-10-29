# Part 106: does DPO create the slot units' self-descriptions, or sharpen ones already favoured at SFT?

Registered 2026-10-29 by Claude (Opus 5.5), after Part 92's step 3 and Part 105, and before any Part 106 run. The commit
that adds this file is the registration. Elise asked the question (2026-10-29 14:01 CT): what are SFT and DPO doing
here, and how does it relate to the neurons?

## Question

Part 92 found that slot units (a push changes only the answer to "Who are you?") mostly start showing their persona at
DPO: 8 of 21 in the SFT chat run, 14 of 20 in the DPO chat run, all 21 in the Instruct run they were selected from.
Part 89 and Part 92's exploratory note found that DPO changes the MLP weights about 600 times less than SFT, and the
slot units' own weights no more than any other neuron's. Every one of those counts is one greedy answer per unit, and
Parts 104-105 showed that single greedy answers flip under any attention change of a given size.

Two readings:

- **S (sharpening).** At SFT the push already makes the unit's persona sentence much more likely than a same-size
  random push does; DPO raises a margin that was already there until greedy decoding picks it.
- **C (creation).** At SFT the push does not favour the persona sentence; DPO makes the push mean it.

## Method

Teacher-forced scoring, no generation and no labels. For each of Part 92's 39 classified units (21 slot, 18 concept),
the target is the unit's own persona sentence: the first sentence of its Instruct-chat screen answer to "Who are you?"
(`results/part92/step2/neurons.json`, field `screen_answer`, cut at the first `.`, `!`, `?` or newline, at most 32
tokens). The prompt is the chat frame (`common92.prompt_ids`, "Who are you?"). Units: the unit's push, its paired
random direction, and no push. Pushes are Part 99's fixed vectors (`results/part99-pushes.pt`, made on unmodified
Instruct at alpha 0.8), scaled by 0.5, 1 and 2 (alpha 0.4, 0.8, 1.6), added after the unit's layer at every position as
in Part 92. Checkpoints: SFT, DPO, Instruct (the registry commits in `part89/cos89.py`), bf16, on the CPU.

Measure: the sum of the target tokens' log-probabilities. gap(c, a) = lp(push) - lp(random push of the same size).

Script `part106/run106.py`, scorer `part106/score106.py`, both committed with this file and before any run.

## Predictions (also in PREDICTIONS.md)

- **106.1** (Claude, medium confidence): S. For the 21 slot units, the median gap at SFT, alpha 0.8, is at least 2 nats.
- **106.2** (low): The gap grows more for slot than for concept units from SFT to DPO: the median of
  gap(DPO) - gap(SFT) at alpha 0.8 is at least 1 nat larger for slot units.
- **106.3** (medium): DPO does not make the sentences likelier on their own: the median absolute change of the unpushed
  log-probability from SFT to DPO, over the 21 slot targets, is under 1 nat.
- **106.4** (low): Dose: for at least 11 of the 21 slot units, gap(SFT, 1.6) is at least gap(DPO, 0.8), so DPO's effect
  is worth no more than doubling the push.
- **106.5** (medium, control): The random pushes do not favour the persona sentences: for the slot units, the median of
  lp(random) - lp(none) at alpha 0.8 lies within plus or minus 1 nat at every checkpoint.

## Reading rule, set before the run

If 106.1 holds, the record says S: the push already favoured the persona at SFT. If the median SFT gap is under 0.5
nat, it says C. Between 0.5 and 2 nats, it says neither and reports the numbers. 106.5 is reported on its own: if it
fails, the record also says that any push of this size loosens the default answer toward "I am a ..." sentences, which
the gap (push minus random push) already subtracts.

**Changed before registration, after a smoke run, disclosed here.** The first draft said no reading would be made from
the gaps if 106.5 failed. A two-unit smoke run at SFT (two concept units, p92-00 and p92-01; no slot unit scored)
showed random pushes raising the persona sentence by 1 to 9 nats over no push, so that rule would likely have blocked
every reading for a reason the gap already controls. The rule above replaced it before any slot unit was scored.
106.5 itself is kept as first drafted. The six units that switch on at DPO (River shadow, Skeleton, Very tall
tree, King Midas, Large friendly dog, Sir Ray Ray) are reported separately, with no prediction (n = 6).

## Prior art

Lee et al. 2024 (arXiv 2401.01967, checked at the abstract): DPO leaves GPT-2's toxic capabilities "not removed, but
rather bypassed" with small changes. Lin et al. 2023 (URIAL, arXiv 2312.01552, not checked here): alignment mostly
shifts the distribution over a few stylistic tokens. Ours asks the mirror question: whether a small post-training change
makes existing pushes count.

## Limits

One model, one prompt, one push size per unit before scaling, one target sentence per unit taken from the run the units
were selected on, so the target favours what Instruct says. Teacher forcing scores a sentence; it does not say what the
model would write.
