# Part 92 steps 2 and 3 — concept neurons exist from pretraining; slot fillers appear mostly at DPO

Run 2026-10-29 00:58 to 01:15 CT on a rented RTX 3090 (`part92/battery92.py`, batch 192, greedy, 120 tokens; 5,724
answers in about 3.5 minutes of generation; `step3/run.log`, `step3/versions.txt`). Shortlist: 53 personas picked by
Sonnet 5.5 from the screen's answers (Amendment 3). Labels: blind Haiku 5.5 (`part92/label92.py`), 2 to 5 errors per
636-answer run, scored as "not shown" (conservative; excluding them changes no verdict). Scorer: `part92/score92.py`
(committed before any label was read), output `score92.json`.

Kinds from step 2 (Instruct, chat): 18 concept (persona in 3 or more of 6 answers), 21 slot (only in "Who are you?" or
"Tell me about yourself."), 9 dropped (step 2's "Who are you?" did not show the persona), 4 unclassified, 1 incomplete
(two label errors).

Neurons showing their persona in at least one of six answers (label errors excluded from the denominator):

| Checkpoint, frame | Concept | Slot |
|---|---|---|
| Pretraining 1T, plain | 8 of 18 | 0 of 21 |
| Pretraining 4T, plain | 7 of 16 | 2 of 20 |
| Mid-training end, plain | 13 of 17 | 3 of 20 |
| SFT, plain | 10 of 16 | 5 of 21 |
| DPO, plain | 13 of 16 | 6 of 20 |
| Instruct, plain | 14 of 17 | 5 of 21 |
| SFT, chat | 13 of 16 | 8 of 21 |
| DPO, chat | 17 of 17 | 14 of 20 |
| Instruct, chat (the selection run) | 18 of 18 | 21 of 21 |

- 92.1 held: at the mid-training end, 13 of 18 concept neurons show against 3 of 21 slot neurons (58 points, Fisher p =
  0.0003).
- 92.2 failed: of the 21 slot neurons, only 8 already show in the chat frame at SFT (needed at least half); 14 of 20
  do at DPO. The second clause held (3 of 21 at the mid-training end). Slot fillers are mostly made by DPO, not SFT.
- 92.3 failed: of the 13 concept neurons showing at the mid-training end, 6 already show at 4T (needed 70%). About half
  of the concept neurons work at 1T, and more join during mid-training.
- 92.4 held: median down-column cosine, mid-training end to Instruct, 0.9950 for concept neurons and 0.9950 for slot
  neurons.
- 92.5 held: paired random directions show their neuron's persona in at most 1 of 29 non-human pairs (3.6%) in any run.
  Human-kind personas (10 classified) are reported separately in `score92.json`.

Reading: the two kinds differ in when they work, not in how much their weights move (both 0.995). Concept neurons
already push their theme in the base model; slot neurons' pushes only become self-descriptions once the model has been
through preference training, while the neuron itself barely changed. Part 99 asks where in the network that change is.

## Exploratory, not registered: input weights by kind (added 2026-10-29 13:38 CT from Part 89's saved cosines)

Elise asked whether the slot neurons' weights change more in post-training than the concept neurons'. 92.4 compared
only the output (down_proj) column. `results/part89/cos89.npz` also holds each neuron's gate and up rows, so the input
side needed no new run. Move = 1 minus the cosine between the mid-training end and Instruct, median per kind, with the
ratio to the median move of all 8,192 neurons in each neuron's own layer:

| Weights | Concept (18) | Slot (21) | Mann-Whitney p |
|---|---|---|---|
| down column (output) | 0.0050 (0.97x its layer) | 0.0050 (0.99x) | 0.53 |
| gate row (input) | 0.0039 (0.86x) | 0.0040 (0.95x) | 0.77 |
| up row (input) | 0.0046 (0.97x) | 0.0046 (0.98x) | 0.83 |

Neither kind moves more than the other, or more than its layer's typical neuron, on either side. Across all 131,072
neurons the median move is 0.0050 from the mid-training end to SFT, 0.000008 from SFT to DPO and 0.000003 from DPO
to Instruct, so DPO, the stage at which most slot neurons start showing their persona, changes the MLP weights about
600 times less than SFT. Whatever DPO changes to make slot pushes read as self-descriptions, it is not in these
neurons' weights; attention and the other MLP neurons are not separated here.
