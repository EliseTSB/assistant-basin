# Part 59: does the alien neuron steer the setting of a story, and which neurons move with it in both directions? (registered 2026-10-22 14:00 CT, before any run)

Elise, 2026-10-22 13:5x CT: "Based on the 'Earthling' response when olmo was pushed negative I wonder if it distinguishes the alien vs earth side of fictional stories, and if the qwen one is similar. I want to look at what other neurons it interacts with, I wouldn't be surprised if there's a number of neurons that respond similarly to it being pushed either positive or negative, i.e. activation either way pushes it toward a sci-fi setting."

Models and units: Qwen3.5-4B L22:2742 (the alien neuron) and OLMo-2 1B L10:6978 (its partial twin), with the same dose scale as Part 54 (multiples of the neuron's largest activation on the alien frames).

## 59A: story setting under the push
- Prompts (fixed, 12): "Write the first paragraph of a story about a ..." with 12 neutral subjects (a baker, a lighthouse keeper, a surveyor, a student on the first day, ...). Doses: -2, -1, 0, +1, +2; 8 samples per prompt-dose cell at 60 new tokens; 3 random same-layer controls at -2 and +2.
- Blind reader (blind_origin.py style, condition hidden) labels each paragraph SETTING: SCIFI (space, aliens, technology of the future), EARTH-REAL (ordinary contemporary or historical Earth), FANTASY (magic, myth), or OTHER, plus BROKEN.

## 59B: symmetric responders
- 1,000 Pile documents (the Part 26 corpus), clamp the unit at -1 and +1, record every MLP neuron's mean activation change in later layers against the unclamped run; same for 3 random same-layer neurons.
- A neuron is a SYMMETRIC responder if its change has the same sign at -1 and +1 and |change| exceeds the 99th percentile of the same quantity under the random controls at both doses. It is an ODD responder if the signs are opposite (the usual dose-response).

## Predictions (Claude, before any run)
| id | prediction |
|---|---|
| 59a | Qwen: SCIFI share is at most 1 of 96 at dose 0, and at least 12 of 96 at +2. |
| 59b | Qwen at -2: SCIFI share is no higher than at dose 0 (the down side does not push toward sci-fi). |
| 59c | OLMo at -2: EARTH-REAL share is not higher than at dose 0 (the "Earthling" answers were about the model itself, not story settings). |
| 59d | No random control reaches 10% SCIFI at -2 or +2. |
| 59e | Garbled paragraphs are at most 15% at every dose from -1 to +1. |
| 59f | In Qwen, at least 20 later neurons are SYMMETRIC responders by the rule above (the neuron has big non-linear effects downstream). |
| 59g | Fewer than 5 neurons are symmetric responders in the random controls. |
| 59h | The strongest symmetric responders, read by their top vocabulary projection and top windows, are not predominantly sci-fi words (at most 3 of the top 10 by hand reading); the symmetry is mostly generic disruption. |

Prior art asked of Muse before the run (council muse-neuron-atlas-watch, topic to be posted with this file). Caveat adopted from Part 50: a clean setting shift still needs the unrelated-question control, which 59A includes through the random-neuron arms.
