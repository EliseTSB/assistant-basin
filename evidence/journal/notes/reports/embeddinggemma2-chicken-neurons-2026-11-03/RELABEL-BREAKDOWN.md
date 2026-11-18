---
title: Which people-imitating-other-animals clips the chicken edits relabelled, and what they became
description: Per-clip relabel breakdown for the one-matrix edit (W2) and the gentle hook on the 112 imitations of other animals, from the two *_other_items.json files.
authored-by: claude
date: 2026-11-04
---

Asked by Elise 2026-11-04 18:41 CT. Counts are from `weight_edit_other_items.json` and `gentle_gate_other_items.json` (112 clips, the model's top ESC-50 class before and after each edit). Human-sound classes: breathing, coughing, sneezing, snoring, clapping, laughing, crying baby, footsteps, brushing teeth, drinking.

- Before any edit, 97 of the 112 were already labelled with a human-sound class, 10 with an animal class.
- One-matrix edit (W2): 62 relabelled. 56 went to another human-sound class (breathing 25, coughing 16, sneezing 6, snoring 5, clapping 3); none went to an animal class. The shuffled-label control relabelled 34.
- Gentle hook: 31 relabelled. 19 to human-sound classes; the rest to keyboard typing (7 in all), chainsaw, can opening, drinking, sea waves. None to an animal class.
- By animal under W2: dogs 9 of 12, cattle 6 of 8, insects 6 of 8, cats 5 of 16, owls 3 of 4, horses 1 of 8, geese 1 of 4. The group names come from the dataset paths and are rough.

Reading: both edits mostly move imitations between human-sound labels; neither makes them look like animals. Not a measurement of what the model hears, only of the top label among 50 classes.
