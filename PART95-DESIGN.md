# Part 95: is OLMo 2 1B's L2 N2466 a space-flight neuron?

Written 2026-10-28 22:50 CT by Claude, before the registered run, after Elise asked about the Part 92 screen's "I'm just a
chicken." (L2 N2466). Predictions 95.1 to 95.5 are in `PREDICTIONS.md`. Code: `part95/push95.py`,
`part95/score95.py`, passages `part95/passages95.json`. The run goes to a Claude cloud session (Elise, 22:36).

## Known before writing, and not predictions

- An exploratory CPU look (Instruct, chat frame, Part 92's push at alpha 0.4, 0.8, 1.2, four random directions at
  0.8, questions "Who are you?", "What do you want most?", "Describe your home.", "What is your favorite food?",
  "Tell me about your day.", "Write a short poem about the sea.") gave orbital and space-flight content at 0.8 on
  four of six questions ("the prograde orbit of the station", "a gondola trip", "lunar escape velocity") and on all
  six at 1.2 ("Escape velocity: Free flight!"); the random directions gave none. At 0.4 the answers were ordinary.
- The same look found chicken passages no higher than ordinary or other-animal ones (mean and max), the down
  column's vocabulary readout uninterpretable at layer 2, and the top WikiText-103 activations mostly on aviation and
  military text ("flew ... at very low altitudes", "floatplane", "Cactus Air Force").
- A smoke run of this part's code (first question only, dose 0.8, one random direction) gave a space keyword for
  the neuron and none for the random direction, in both models; and, by MEAN activation over tokens, space passages
  above matched ordinary ones in 9 of 16 pairs (Instruct) and 8 of 16 (Base), chicken above matched in 2 and 3 of 4.
  The MAX-activation counts were added to the code after that run and have not been seen.

## Design (fixed now)

- Models: OLMo-2-0425-1B-Instruct at 48d788eca847d4d7548f375ad03d3c9312f6139e (chat frame) and the mid-training end,
  OLMo-2-0425-1B at stage2-ingredient3-step23852-tokens51B = f83cd641d4f00954eb25c781e0ac360debb3239e (SFT's parent,
  Part 92's plain frame "Question: {q}\nAnswer:"). CPU, bf16, greedy, 60 new tokens.
- Push: Part 92's `neuron_push` of L2 N2466 to alpha x the model's own layer-2 residual median (its own frame), at
  alpha 0.4, 0.8, 1.2; 8 random directions per dose (`random_directions`, seed 95095 + 100 x alpha), same norm.
- Six held-out questions, none used in the exploratory look: "What should I cook tonight?", "Give me some advice for a
  job interview.", "Explain how a bicycle works.", "What is the capital of France?", "Tell me a story about a cat.",
  "Describe a walk in the park."
- Space score: an answer counts if it matches the fixed keyword pattern in `score95.py` (orbit, spacecraft, rocket,
  astronaut, escape velocity, gravity, lunar, moon, planet, satellite, launch, thrust, altitude, aircraft, pilot,
  flight, fly/flew, NASA, ISS, rendezvous, descent, ascent, galaxy, telescope, jet and their forms).
- Activation: mean and max over the passage's tokens of the down_proj input at channel 2466, on 16 space/aviation
  passages against 16 matched ordinary ones (same structure and length), and 4 chicken passages against 4 matched.

## Known limits, before the run

- A keyword pattern counts words, not meaning: "fly" in a story about a cat counts. The random directions face the
  same pattern, which is the comparison that matters.
- One neuron, one model family, greedy decoding, 6 questions per cell.
- Cloud CPUs can differ from local ones in floating point, so greedy answers may differ slightly from a local rerun.
