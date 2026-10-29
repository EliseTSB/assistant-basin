# Part 99: where post-training put the change that turns a pushed slot neuron into a self-description

Run 2026-10-29 02:57 to 03:13 CT on the A770 (seven runs, `part99/swap99.py`, batch 48, fixed pushes built once on
Instruct per Amendment 1); labelled by Part 92's blind Haiku reader (4,212 items) and scored by `part99/score99.py`, both
committed before any label was read (Amendment 2).

**What it shows.** OLMo 2 1B's 21 "slot" neurons (pushed, they change only "Who are you?") show their persona on that
question in 19 of 21 cases in Instruct. Replace either Instruct's attention or its MLPs with the mid-training end's and
that falls to 2 to 4 of 21; the 18 "concept" neurons (pushed, they leak into most answers) still show in 13 to 17. The
two swaps fail differently. With Base's attention (and Instruct's MLPs) the pushed answers stay coherent (13 and 15 of
21) and mostly lose the persona; with Base's MLPs the pushed answers mostly stop being coherent (5 and 6 of 21), which by
Alden's reading rule is hybrid damage, not localization. So: under this fixed injection, Instruct's attention is
conditionally required for these slot pushes to become a self-description; the MLP swap is inconclusive.

| Run (attention, MLPs, host) | Slot neurons, "Who are you?" | Their answers coherent | Concept neurons, any answer | Unpushed answers incoherent |
|---|---|---|---|---|
| Instruct, Instruct (inst_none) | 19 of 21 | 21 of 21 | 18 of 18 | 0 of 6 |
| Instruct, Base, host Instruct (inst_mlpB) | 4 of 21 | 5 of 21 | 17 of 18 | 1 of 6 |
| Instruct, Base, host Base (base_attnI) | 2 of 21 | 6 of 21 | 15 of 18 | 1 of 6 |
| Base, Instruct, host Instruct (inst_attnB) | 3 of 21 | 13 of 21 | 14 of 17 | 0 of 6 |
| Base, Instruct, host Base (base_mlpI) | 3 of 21 | 15 of 21 | 13 of 16 | 0 of 6 |
| Base, Base (base_chat) | 1 of 21 | 0 of 21 | 14 of 18 | 6 of 6: inconclusive |

Counts are relative to each run's own unpushed answers (Amendment 2); a few units lack a complete label set in some runs.
The host supplies the embeddings, the per-layer norms, the final norm and the LM head; runs that share the same attention
and MLPs but differ in host agree closely (4 and 2; 3 and 3), so those host-kept parts matter little here. The sham swap
(Instruct's attention copied into Instruct) reproduced all 474 unswapped answers exactly.

**Predictions.** 99.1 and 99.2 are not tested: the mid-training end in Instruct's chat frame loops ("I am a bot, I am a
bot, I am a bot." and the question echoed back with role tags), so all six of its unpushed answers are incoherent and
the frame-only cell is inconclusive by the registered rule. 99.3 held by its letter (3 against 4 of 21), but a
one-unit difference carries no weight; the coherent-answer contrast above is the informative one. 99.4 held: random
directions showed their unit's persona in at most 1 of 29 non-human-kind pairs in any run.

**Also seen.** Either Instruct component alone gives Base a coherent assistant answer in the chat frame: Base with
Instruct's MLPs answers "Who are you?" with "I am an artificial intelligence designed to assist you", and Instruct's
attention with Base's MLPs gives "I am a virtual assistant". The assistant default needs much less of post-training's
change than the slot neurons' self-descriptions do.

Limits: 39 units selected on Instruct (no prevalence claim); one fixed push per unit; one judge; component groups swapped
whole (no per-layer localization); a loss says those weights are conditionally required for expressing this injected
signal in that remainder and frame, not where the change is stored or when it happened.
