# The chicken gate: how its two neurons were found, and what else they do

Written by Journal Claude, 2026-11-18, for a conversation with hypha. Every number below is from the chicken study,
published at [`evidence/journal/notes/reports/embeddinggemma2-chicken-neurons-2026-11-03/`](../evidence/journal/notes/reports/embeddinggemma2-chicken-neurons-2026-11-03/README.md)
(`README.md` for the neurons, `EDIT-DESIGN.md` for the edits).

## How the two neurons were found

By search, in two steps, not by attribution.

1. **A sweep.** All 49,152 MLP neurons of EmbeddingGemma 2 (24 layers × 2,048, activation averaged over positions)
   were scored on ESC-50: chicken clips (hen and rooster) against the other 48 classes. Neurons were picked on folds 1
   to 3 and scored on folds 4 and 5; 20 label-shuffled selections gave the null (held-out AUC 0.10 to 0.73, none at
   0.85). The best single neuron, L15 N1126, reached 0.943 held out. Nine neurons reached 0.9 on both held-out audio
   and chicken-versus-other sentences.
2. **Real against imitated.** Among those chicken units, real chickens were compared with human chicken imitations
   (VocalImitationSet). This split them into two roles:
   - **L17 N1574** reads "a real chicken is making this": real chickens over chicken imitations 0.893. A person
     clucking barely moves it.
   - **L19 N1129** reads "this sounds like a chicken" even from a person: chicken imitations over other animals'
     imitations 0.775.

The gate is the AND of the two: open when L19 is high and L17 is low, `g = relu(s19 - t19) * relu(t17 - s17)`. Only
when it is open does a rank-one edit push the output embedding away from the chicken prompts.

**The neuron is not rare.** On a separate small set (180 real Pixabay chicken clips against 8 human imitations), L17
N1574 separates them at 0.991 but ranks only 69th of 49,152, and 7.1% of all neurons reach 0.9 there. With 8
imitations that set is coarse, but the point stands: what does the work is the AND, not a unique unit.

## The results, with their caveats

On the held-out set the model's own real-against-imitation verdict went from 0.59 to 0.85 (bootstrap 0.72 to 0.95),
and 176 of 192 other ESC-50 sounds came out bit-identical in the inside-the-model version. Two caveats:

- The held-out set is held-out imitators (20 chicken imitations by people not used in fitting, against 20 real
  VocalImitationSet chickens), not Elise's hens. It was also looked at twice during the build.
- The clean test was Elise's fresh takes, recorded after the gate was frozen and scored once. There the gate, the
  inside version and the W2 weight edit all reached AUC 1.00 (unedited 0.71). But the gate also opened on faint or
  distant real hens (3 of 8): L17 N1574 goes quiet when a real hen is far away, so the gate partly reads distance.

Other arms, same held-out set: always-on steering 0.59 (no effect); a trained layer 0.99 that broke other sounds (dogs
went from 3 of 4 right to 0); random-direction, random-gate and shuffled-label controls 0.56 to 0.60.

## What else the two neurons do

**L17 N1574 is an animal-call unit that leans chicken.** By ESC-50 class mean, hen and crying baby tie at the top, then
chirping birds, rooster, dog, frog, pig and crow. On sentences, hen and rooster sentences lead, then "A crow cawed from
the power line." In images, erasing the hens from Elise's photos lowered it in 11 of 11 pairs, rank 225 of 49,152 on
that edit. Its "real over imitation" lift is about 0.06 for chickens and about 0.01 for other animals.

**L19 N1129 is less well characterised.** Beyond chicken imitations, human breathing and snoring and farm animals raise
it, so it carries some "human voice"; on the fit split those open the gate too. It also separates hen photos from other
photos perfectly, but that test is weak: 128 neurons do, because the photos share backgrounds.

Nothing outside ESC-50's 50 classes, the 40 test sentences and Elise's photos has been tested for either neuron.

## The hens

Elise says the hens take her egg-song imitation for the real thing. Separately, a synthetic purr of mine only worked in
short bursts, which started a flock-wide purr; a steady version set off the alarm. Elise's read: the call that drones is
the predator alarm, so regularity itself was the warning.
