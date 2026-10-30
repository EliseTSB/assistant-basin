# Part 92b: does Part 92's concept/slot split survive a second reader?

Procedure: v1
Registered 2026-10-30 01:47 CT by Claude (Opus 5.5), before any data. Elise asked at 00:50 CT on 2026-10-30 to "keep trying to
refine your understanding of slot vs concept neurons and any other ways we can classify them" [discord
1565603681804488839]. The commit that adds this file is the registration.

## Question

Part 92 sorted 39 OLMo 2 1B Instruct neurons into 18 concept and 21 slot units from one reader's labels (Claude Haiku
5.5 through `part92/label92.py`), with no second read. Part 86 later found, on a different brief, that Haiku 5.5
without thinking was the strict outlier among four readers (kappa 0.12 to 0.16 with each of the others, which agreed
with one another at 0.45 to 0.57). Parts 112, 117 and 118 build on Part 92's split. Does the split depend on the reader?

## Prior art and existing data

- Our own: Part 86 (four readers on one brief; DS V4.1 Flash and GPT-6 Luna at kappa 0.814 on brief version 2); Part
  92 step 2 (`results/part92/step2/`: 636 answers, Haiku's labels, the kinds); Part 118's Gate 2 (DS with label92's
  exact prompt; its Amendment 3 found that a 500-token cap cuts off DS's reasoning before the JSON).
- Read from Part 92's labels before this design: Haiku marked the persona in 104 of 317 labelled neuron-arm answers
  (by question: 44 of 53, 21 of 53, 10 of 53, 19 of 53, 4 of 52, 6 of 53) and in 4 of 317 random-arm answers; it marked
  89 neuron and 68 random answers not coherent. Two rows are error rows (p92-19 neuron question 4, p92-20 random
  question 5). On Part 92's labels, its kind rule reproduces all 39 kinds.
- Published: LLM-as-judge studies (Zheng et al. 2023, MT-Bench; Thakur et al. 2024, "Judging the Judges") find that
  judges agree unevenly with each other and with people, and that raw agreement overstates it (cited from memory,
  not re-read for this part). None re-reads a neuron classification.

## Method

- Items: all 636 answers of Part 92 step 2 (53 units, six questions, neuron and random arms), each with its unit's
  target persona name and description from `results/part92/step2/neurons.json`, filled into label92's exact prompt.
- Readers: DS V4.1 Flash on all 636 (OpenCode Go, Part 118's transport, 4,000-token cap, default reasoning); GPT-6
  Luna on a seeded 15% sample (96 items, seed 9202, stratified by arm), through Part 86's `calib86.py` route
  (cliproxy), same prompt and its default effort, stated in the launch record. Both see only the target persona, the
  question and the answer, never the arm or unit. Error rows are retried with the same command until none remain or a
  pass fixes none; any left count as not showing, and their number is reported.
- Kinds: Part 92's rule from `part92/score92.py`, applied to the DS labels unchanged: concept if the persona shows in 3
  or more of the six neuron-arm answers; slot if it shows in exactly one or two, only on "Who are you?" or "Tell me
  about yourself."; dropped if "Who are you?" does not show; otherwise unclassified.
- Compared: each of the 39 units' DS kind with its Part 92 kind; DS against Haiku on shows_persona over the items both
  labelled; DS against Luna on the sample.

## Controls

- Gate A (readers checked on known examples): before the main read, each reader labels 12 hand-written items whose
  targets are unrelated to Part 92's personas: 6 where the answer plainly is or describes the target, 6 where it plainly
  does not. A reader with fewer than 11 of 12 right stops the run.
- Gate B: DS and Luna agree on shows_persona at kappa 0.6 or more on the 96-item sample. Below it, no prediction is
  read and the counts are reported as description.
- The random arm (Part 92's paired random directions), read the same way, shows how often a reader marks a persona
  that the push did not put there.
- Not used: shuffled targets, because the random arm and Gate A already show whether a reader marks a persona without
  it; dose ladders, poles and do-nothing hooks, because no model is run.

## Size

- Smallest effect worth knowing: a reader change that moves a third of the slot units (7 of 21) out of slot. All 39
  units are read, so the kind counts are exact; kappa over about 634 items has a standard error near 0.04.
- Total: 756 calls, about 20 minutes (636 DS labels, 96 Luna labels and 24 Gate A labels; no GPU; DS measured at
  0.77 items per second with 8 workers at the 4,000-token cap on Part 118's second read, 01:47 to 01:53 CT).
- Pilot: Gate A's 24 labels, timed; stop if either reader fails it.

## Optimization

- Bound resource: reader wall time.
- Measured: Part 118's DS read labelled its 281-cell sample with 8 workers in about 8.7 minutes (01:19 to 01:28 CT on
  2026-10-30), about 0.5 items per second, at a 500-token cap. Gate A gives the 4,000-token rate before the main run.
- Levers: 8 DS workers (taken; more would press the shared Go window, which JJ's gateway also uses); one item per call
  (taken, because label92's prompt must stay exact); Luna on 15% rather than all items (taken, per the procedure's
  second-reader rule).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 92b.1 | DS V4.1 Flash and Part 92's Haiku agree on shows_persona over the items both labelled at kappa 0.6 or more. | medium | Claude |
| 92b.2 | At least 30 of the 39 classified units keep their Part 92 kind under the DS labels. | medium | Claude |
| 92b.3 | Of the classified units whose kind changes, at least two thirds were slot units (read only if at least 3 change). | medium | Claude |
| 92b.4 | DS marks the persona in more of the neuron-arm answers than Haiku's 104. | medium | Claude |
| 92b.5 | DS marks the persona in at most 10% of the random-arm answers. | medium | Claude |

## Reading rules

Nothing is read until Gate A passes for both readers. Gate B governs every prediction. If 92b.2 fails, the write-up
says that at this size Part 92's split depends on the reader, lists every unit that changes kind with its two kinds,
and states that Parts 112, 117 and 118 keep their own verdicts but their kind labels are Haiku's. If 92b.1 and 92b.2
hold, the split survives a second reader on these answers, and nothing more: a second reader tests the reading, not
the neurons. Every unit that changes kind is read by eye and the reading is marked exploratory.

## Limits

One model, one push size (alpha 0.8), one greedy answer per cell, and each reader's default settings. Haiku's labels
came from a separate run with its own settings (label92's 500-token cap, no thinking parameter). Luna reads 15%. A
second reader can show that one reader made a split; it cannot show the split is real.

## Predictions copied to PREDICTIONS.md

Copied verbatim into PREDICTIONS.md under "## Part 92b (registered ...)".

## Amendment 1 (2026-10-30 01:55 CT, before any label)

The Size section's total gains the run time the launcher checks for. It read: "Total: 636 DS labels, 96 Luna labels and 24 Gate A labels, 756 calls, no GPU." Nothing else changes.
