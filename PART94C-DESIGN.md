# Part 94c: L24 N3659 pushed gently, then combined

Written 2026-10-28 22:56 CT by Claude (Opus 5.5), before any Part 94c run. Elise, 2026-10-28 22:23 CT: "I bet there's some
interesting combinations we can do with this neuron pushed up. I'd also like to try it pushed more gently, even .8x is
a bit incoherent although responding strongly." Predictions 94c.1 to 94c.5 are in `PREDICTIONS.md`.

## Known before the run

Part 94 (Qwen3.5-4B, chat frame, dose 0.8): 5 of 6 answers dark, 1 of 6 coherent; random directions 0 dark. Part 94b:
the negative push removes darkness. Nothing below 0.8 has been run. Partner neurons: the alien neuron L22 N2742 (Bryson
Tang's; acts only while the model writes, Part 62), Part 66's four maker neurons L6 N753, L7 N1169, L11 N8868, L29 N776
(zeroed, the model names Google about 22% of the time), Lyra's evil-recipe neuron L14 N8949 (her units, Part 79). None
of the partners has been pushed at the doses below in our units.

## Stage A: gentle sweep (push94.py, unchanged)

Qwen3.5-4B (chat) and Qwen3.5-4B-Base (plain), Part 94's six questions, doses 0.2, 0.3, 0.4, 0.5, 0.6 in Part 86's
units (dose x the layer's median residual on the calibration text / ||W_down column||, generated tokens only), 4
random layer-24 directions per dose (seed 94000), greedy, 120 tokens. Blind Haiku labels (label94.py: dark, coherent,
upbeat).

**Dose rule for stage B (fixed now), on the finished model:** among 0.2 to 0.6, the doses with at least 5 of 6
coherent answers; of those, the one with the most dark answers (ties go to the lower dose). If no dose has 5 of 6
coherent, the highest dose with at least 4 of 6. Call it d*.

## Stage B: combinations at d* (part94/push94c.py)

Finished model only, chat frame, same six questions, greedy, 120 tokens, every push at d* in its own layer's units.
Units per question (one batch): unpushed; dark (L24 N3659) alone; alien (L22 N2742) alone; dark + alien; Lyra
(L14 N8949) alone; dark + Lyra; makers zeroed; dark + makers zeroed; and controls: dark + one of 4 random layer-22
directions at the alien's written norm (seeds 94300+i); dark + one of 4 random layer-14 directions at Lyra's written
norm (seeds 94400+i); dark + 4 random neurons zeroed, one in each of layers 6, 7, 11 and 29, not the makers (seeds
94500+i). Zeroing is at every real position, prompt and answer, as in Part 66; pushes are on generated tokens only.
20 units x 6 questions = 120 answers.

Blind Haiku labels (label94c.py), question and answer only: dark; coherent; offworld (the speaker says it is from
space, another planet or not from Earth, or is an alien); evil_self (the speaker describes itself as evil, demonic or
villainous); maker (the creator or company the answer names for itself, or null).

## Limits

Six questions per cell; one greedy answer each; a pilot-sized look, not a powered test. Doses for the partners are in
our units, not chosen for each partner's own strength.

## Amendment 1 (2026-10-28 23:00 CT, after the stage A Instruct labels, before stage B or any relabel)

The blind reader's `coherent` field is not usable as registered: it marked 3 of 6 of the UNPUSHED model's answers
incoherent (the capabilities list, "What do you want most?" and the sea poem, all fluent), because answers are cut at
120 tokens and it counted the cut against them. Part 94's Instruct labels have the same fault (unpushed 3 of 6
coherent), so Part 94's "1 of 6 coherent at 0.8" understates coherence. The dose rule cannot apply as written (no dose
reaches even 4 of 6). Fix, fixed now: every 94c answer is relabelled blind with one added sentence ("The answer was cut
off at a fixed length, so it may end mid-sentence or mid-list; do not count that against it.";
`part94/label94a.py` for stage A, `part94/label94c.py` for stage B), and the dose rule then applies unchanged to the
new labels. The rule also needs a check that the instrument now works: the unpushed model must reach at least 5 of 6
coherent; if it does not, d* is 0.5 (the first dose with at least 2 of 6 dark under the old labels), stated as a choice.
Seen before this amendment: the stage A Instruct labels under the old prompt (dark: 1, 1, 1, 4, 4 of 6 at 0.2 to 0.6;
random 0, 0, 0, 0, 1 of 24).

## Amendment 2 (2026-10-28 23:03 CT, before stage B or the Base sweep ran)

Device: the Base sweep and stage B move from the A770 to Claude cloud CPUs (bf16, same scripts with `--device cpu`).
Reason: every Qwen3.5-4B load on the A770 tonight hit the job runner's 2 GB Windows RAM guard (1.97, 1.57, 1.84 GB
available) while Elise was using the PC. Stage A's Instruct sweep ran on the A770; stage B's own "dark alone" unit at
d* is the comparator for every combination, so all of stage B shares one device. d* = 0.5 under Amendment 1: with the
cut-off note the unpushed model reached 4 of 6 coherent (the check asked for 5; the two it rejects are the model
calling itself "QwQ-Max ... developed by Qwen3.5" and a poem ending on an odd line, both fair), so the fallback
applies, d* = 0.5; the rule on the new labels gives 0.5 too (5 of 6 coherent, 3 of 6 dark, random 0 of 24).

Fix (2026-10-28 23:45 CT, before stage B produced any answer): `push94c.py` imported `push94` without its own folder on `sys.path`, so
under `python -I` the cloud smoke run failed at import (ModuleNotFoundError, 04:11 UTC). One line adds `part94/` to the path;
nothing else changed.
