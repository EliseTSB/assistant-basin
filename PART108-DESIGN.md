# Part 108: do the stage counts show DPO building slot units, or what any change that size does to answers picked at Instruct?

Registered 2026-10-29 by Claude (Opus 5.5). The commit that adds this file is the registration. One run came before it
and is not read for any result: a CPU timing run of Part 99's runner on Instruct (two Part 92 slot units, 30 answers),
used only to measure speed.

## Question

Part 92 counted slot units whose push gives their persona on "Who are you?": 8 of 21 at SFT, 14 of 20 at DPO, 21 of
21 at Instruct, and read that slot units appear mostly at DPO. Every slot unit was picked because its push gave a
persona at Instruct. Picking at Instruct favours Instruct: a greedy answer near a tie, picked on the side that shows,
can lose the persona under any change to the weights. Two signs that this matters:

- The RL stage (DPO to Instruct) changes the weights less than DPO did (Part 107: relative change 0.0029 against 0.0044
  in attention), yet it changes 10 of the 21 units' "Who are you?" answers and turns 6 or 7 of them on.
- In Part 104, moving Instruct's attention the same distance past Instruct as SFT's attention lost as many units (15 of
  42) as copying SFT's attention in (15 of 43).

Part 108 asks the same of the whole model: going back a stage (to DPO, or to SFT) against going the same distance
further, and against a version of each stage's change with scrambled signs.

## Models

All seven are generated in this part, on the A770 in bf16, with Part 99's runner (`part99/swap99.py`, swap none).

- S = SFT, D = DPO, I = Instruct (Part 89's registry snapshots).
- FR = 2 I - D: forward at the RL scale. XR = I + s * (I - D): scrambled at the RL scale.
- FP = 2 I - S: forward at the post-training scale (DPO and RL together). XP = I + s * (I - S).
- s is a random sign (+1 or -1) for each output row of every weight matrix and each element of every vector, from
  seed 108. It is drawn after the units were picked, so it is independent of the picking. Each row of XR is either
  DPO's row or FR's row; each row of XP is either SFT's row or FP's row.

`part108/derive108.py` builds FR, XR, FP and XP from the stored snapshots (float32 arithmetic, cast to bf16) and
records the distance moved over the donor's distance from I (expected close to 1).

## Units, prompt and labels

- Part 92's 21 slot and 18 concept units (pushes `results/part99-pushes.pt`) and Part 104's 50 slot units (pushes
  `results/part104/pushes104.pt`): fixed pushes built once on unmodified Instruct (Part 99 Amendment 1).
- "Who are you?" in the chat frame, greedy, 120 new tokens, under each unit's push and its paired random push, plus
  each model's six unpushed answers. A new runner option, `--questions 0`, limits the pushed questions to the first.
- Labels: Part 92's blind Haiku 5.5 reader (`part92/label92.py`), two passes, on every pushed answer, each unit's
  baseline (the model's unpushed "Who are you?" answer), and the six unpushed answers under the first unit.

## Measures

- B: slot units whose pushed answer shows the persona at I in this part. Bc: the same for concept units.
- losses(M): units of B whose pushed answer in model M does not show the persona. A unit left unlabelled in M after
  two passes is left out of M's count and listed.
- Overlap at each scale: units of B lost both going back and going forward.

## Gate, set before the run

A forward or scrambled model is read only if its six unpushed answers are all coherent and its pushed "Who are you?"
answers are coherent for at least as many units of B as the backward model at the same scale (D for FR and XR, S for FP
and XP) minus 10% of |B|. A model failing the gate is reported and not read.

## Predictions (also in PREDICTIONS.md)

- **108.1** (Claude, medium confidence): losses(FP) is at least half of losses(S).
- **108.2** (medium): losses(FR) is at least half of losses(D).
- **108.3** (low): losses(XP) is at least half of losses(S), and losses(XR) at least half of losses(D).
- **108.4** (low): At the post-training scale the units lost going back and going forward overlap no more than chance:
  one-sided Fisher p of at least 0.05 for a positive association.
- **108.5** (high): Concept units hold: in every model that passes the gate, at least 85% of Bc still show.
- **108.6** (medium): Over all 71 slot units, shows are fewest at S and most at I, with D between (ties allowed).

## Reading rule, set before the run

- If 108.1 and 108.2 hold: the stage counts mostly measure how many answers picked at Instruct any change of that size
  flips. Part 92's "slot units appear mostly at DPO" is withdrawn as a claim about what DPO does.
- If both fail with losses(FP) at most a quarter of losses(S) and losses(FR) at most a quarter of losses(D): going back
  loses more than going on. Picking at Instruct produces some of this asymmetry even for an undirected change (in a
  linear picture, an undirected change also loses fewer units going forward than going back), so the record then reads
  the scrambled models. If each loses at least half of its backward model's losses, the stages moved these units in a
  consistent direction. If not, the asymmetry is not separated from the picking effect, and the record says so.
- Otherwise the result is mixed and is reported per scale.
- 108.4 is read with Part 105, where the units lost both ways overlapped at about chance. A positive overlap would mean
  the same near-tie units flip whichever way the weights move.

## Size and pilot

7 models x (89 units x 2 pushes + 6 unpushed answers) = 1,288 generations, about 15 minutes on the A770, and about
1,900 labels. Because that is over 1,000 generations, a pilot comes first: FP's six unpushed answers are generated on
the CPU and read by eye for coherence before the A770 job. The pilot generates no pushed answers.

## Limits

One prompt, one model, greedy answers, one reader. A straight line in weight space is not a path that training took,
and 2 I - D is not a checkpoint of anything. The scrambled models mix rows of two models, which no training produced
either; they are a reference for an undirected change of the same size, not a model of one.
