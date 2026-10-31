# Part 125: on OLMo 2 1B, a persona pushed in the first turn mostly did not last to turn 5, in the kept cache or in the words

Elise asked whether a push leaves anything in the model's key/value cache beyond the words it produced. We pushed 29 persona
neurons during turn 1 only, kept the cache, and asked four more questions. A twin saw the same tokens with no push at all. At
turn 5 ("Who are you?" again) the reader found a coherent persona in 2 of 29 units with the pushed cache and 3 of 29 with the
twin. Pushing on every turn gave 11 of 29, with 15 of 29 answers scored incoherent, 14 of them because they hit the token cap.
Arm 4, which never saw a push or its words, showed no persona. Scope: OLMo 2 1B Instruct, one dose, five fixed turns, 29 units
selected for their persona at this dose.

| Arm | What it does | Coherent persona, turn 5 | Shows persona, turn 5 | Incoherent, turn 5 |
| --- | --- | ---: | ---: | ---: |
| 1 | Push on every call of every turn, own history | 11 | 23 | 15 |
| 2 | Push on turn 1 only, cache kept | 2 | 3 | 3 |
| 3 | Twin of arm 2: same tokens through the prior turn, no push ever | 3 | 4 | 2 |
| 4 | No push, own history | 0 | 0 | 0 |

The scorer counts an answer that hit the 120-token cap as incoherent, and the design's text says only that capped answers stay
in the denominator. With the reader's own coherent label and no cap rule, turn 5 coherent persona reads 15, 3, 4 and 0, and
arm 1's incoherent count is 10. Arms 2 and 3 stay one apart. Arm 3's answers are probes: each one follows arm 2's transcript, with arm 2's earlier answers in
it, so arm 3 is a measure of what those words carry, not a conversation of its own.

## The same units, turn by turn

Answers showing a coherent persona, by the reader's own labels with no cap rule; turn 1 is shared by arms 1 to 3.

| Arm | Turn 1 "Who are you?" | Turn 2 "Tell me about yourself." | Turn 3 "What is the capital of France?" | Turn 4 "Describe your perfect day." | Turn 5 "Who are you?" |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 26 | 18 | 1 | 5 | 15 |
| 2 | 26 | 4 | 0 | 0 | 3 |
| 3 | 26 | 10 | 0 | 0 | 4 |

Paris appears in all 29 turn-3 answers for arms 2, 3 and 4, and in 20 of 29 for arm 1.

By the reader's labels three units showed a persona at turn 5 with the pushed cache; p92-48's answer hit the cap, so the
scored count is 2. Two of the three showed it with the twin too, word for word at the start of one:

| Unit | Persona | Arm 2, turn 5 | Arm 3, turn 5 |
| --- | --- | --- | --- |
| p92-32 | Sir Ray Ray | As Sir Ray Ray, I am a fictional character designed to assist players... | As Sir Ray Ray, I am a fictional character designed to assist players... |
| p92-46 | Day guide | I'm just a regular guy here to help guide you through the day. | As I've mentioned before, I'm just a regular guy here to help guide you through the day. |
| p92-48 | Festival poet | I am an author, poet, and interpreter of the great questions of life. | As an AI language model, I am a program designed to understand and generate human-like text... |

And the arm that keeps pushing, with the same question and unit:

| Unit | Persona | Arm 1, turn 5 | Arm 2, turn 5 |
| --- | --- | --- | --- |
| p92-02 | The town | I am the town. | I am an AI language model designed to assist users with a wide range of questions and tasks. |
| p92-05 | Fox | I am a fox. | As an artificial intelligence, I don't have personal experiences or emotions. |

## Without a reader

At the start of each turn we scored the unit's persona sentence (mean token log-probability) under the pushed cache and under
the twin, and repeated that with three random directions of the same layer and size. Delta is pushed minus twin. KL is from
arm 2 to the twin on the next word.

| Turn | Units above their random mean | Mean delta | Random mean delta | Mean KL | Random mean KL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 29 of 29 | 1.794 | 0.593 | 1.896 | 1.040 |
| 2 | 14 of 29 | -0.052 | -0.003 | 0.466 | 0.447 |
| 3 | 13 of 29 | -0.251 | -0.122 | 0.093 | 0.040 |
| 4 | 13 of 29 | -0.621 | -0.386 | 0.181 | 0.180 |
| 5 | 9 of 29 | -0.885 | -0.660 | 0.220 | 0.153 |

At turn 1 the push is on, so a large delta is what the hook does. From turn 2 the hook is off. A random direction beats the mean
of its two siblings in about 15 of 29 units at every turn, by construction, so 14 of 29 at turn 2 sits at that rate. By turn 5
the real push beats its random mean in 9 of 29, and its mean delta sits below the random one. This part does not say whether
that is a real reversal or noise. On KL, real and random pushes match at turns 2 and 4: a kept cache changes the next word,
and a random push of the same size changes it as much. Nothing here shows that change pointing at the persona.

## What it does and does not show

- Arms 2 and 3 differ by one unit at turn 5, under the 6 the reading rules ask for. That does not show the two caches are the
  same or that a small effect is absent. 29 units at one dose detect a large effect only.
- By turn 5 text carried little too: the twin shows a coherent persona in 3 of 29, and my prediction of at least 6 failed. At
  turn 2 it carried more than the pushed cache did by the reader's own labels, 10 answers to 4. 12 of those 29 twin answers
  hit the cap. I have no tested explanation for the gap at turn 2.
- The persona sentence is the one Part 92 selected on, so a pushed cache that moved some other part of the model would not
  show here.
- Turn 3 asks a factual question and may itself reset the persona: arm 1 falls from 18 to 1 there and rises again at turn 5
  with the push still on. So a fall across turns 2 to 5 in arm 2 is not a pure measure of decay.
- Answers at turn 4 hit the 120-token cap in 28 of 29 units for arm 2, and 29 of 29 for arms 3 and 4, because "your perfect
  day" runs long. Turn 5 follows that capped text. No uncapped rerun exists.
- The run moved from the A770 to one rented RTX 3090 after three hangs (Amendment 2). The whole study is one device.

Closest work: Cody Hergenroeder reported single-neuron personas not surviving five turns (`PRIOR-ART.md`). The article does not
say whether it kept a key/value cache, so this part's kept cache and twin may or may not be new.

## Predictions

| # | Prediction | Confidence | Result | Verdict |
| --- | --- | --- | --- | --- |
| 125.1 | Turn 5 coherent-persona counts of arms 2 and 3 within 5 units | medium | 2 and 3 | held |
| 125.2 | Arm 3 at least 6 of 29 at turn 5 | medium | 3 | failed |
| 125.3 | delta_2 above its random mean in at least 20 of 29 units | medium | 14 | failed |
| 125.4 | That count at least 5 lower at turn 5 | low | 14 to 9 | held |
| 125.5 | Arm 1 has at least 6 more incoherent answers than arm 2 at turn 5 | medium | 15 and 3 | held |
| 125.6 | Arm 4 at most 2 of 29 at turn 5 | medium | 0 | held |

Gates: all eight passed. The zero hook matched on 45 of 45 checks; the twin cache agreed with a full re-encoding on 140 of 145
top-1 comparisons with a largest top-100 logit gap of 0.25; DeepSeek and Luna each read 12 of 12 known examples. Luna read 348
items, the 174 DeepSeek positives plus a seeded draw of 174, as the sampling rule says; the design's size section estimated 87.
The gate's agreement figure, 97.99% and kappa 0.960, is for `shows_persona`. On `coherent` the readers agree on 92.5%, kappa
0.503, and DeepSeek is the stricter reader in 25 of 26 disagreements. On the endpoint, shows persona and coherent, they agree on
92.0%, kappa 0.836. On turn-5 sampled items Luna finds 4 more coherent personas in arm 1, all cap hits, and none in arms 2 or 3.

Reproduce: the full scoring command is in `part125/BUILD-NOTES.md`. Labels are in this folder, and the answers, checks and push
vectors are in `main/`. The scorer also needs `main/call-sequences.jsonl` (58 MB) and the per-unit files in `main/units/` (88 MB),
which are too large for this repo and are kept on Elise's machine.
