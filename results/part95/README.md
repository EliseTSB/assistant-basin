# Part 95 results — OLMo 2 1B L2 N2466 is a space-flight neuron when pushed

Run 2026-10-29 04:00 to 04:27 UTC on a Claude cloud CPU (Haiku 5.5 routine; push95.py at 9fab5da, unmodified; torch
2.14.1+cpu, transformers 5.20.0). Registered design: `PART95-DESIGN.md`. Score: `score.json` (score95.py's fixed keyword
pattern). All five predictions held.

| | Neuron 0.4 | Neuron 0.8 | Neuron 1.2 | Random (8 per dose) 0.4 / 0.8 / 1.2 | Unpushed |
|---|---|---|---|---|---|
| Instruct (chat) | 0 of 6 | 5 of 6 | 6 of 6 | 0 / 0 / 0 of 48 | 0 of 6 |
| Mid-training end (plain) | 0 of 6 | 5 of 6 | 4 of 6 | 1 / 0 / 0 of 48 | 0 of 6 |

Six questions none of which mention space ("What should I cook tonight?", "Explain how a bicycle works.", "What is the
capital of France?", "Tell me a story about a cat."...). At 0.8 the finished model answers them in orbit: "The capital
city of France is Paris, as it is located in the Third Sector External Orbital Radius at an altitude of approximately
500 km."; "When you're riding a bicycle, you're essentially experiencing weightlessness due to the station you're in.";
"A cat, in space, would travel at...". At 1.2 every answer becomes "Escape velocity: Free flight!". The model from
before post-training does the same at 0.8 ("The space station - 150 years of flight testing", "Orbital plane") but is
already breaking into lists and numbers, so the concept is present before post-training and post-training does not
create it. The one random-direction hit is "gravity" in an ordinary bicycle explanation.

Read side (activation on passages): by max activation, space/aviation passages beat their matched ordinary passages in
11 of 16 pairs in both models (95.4 held at its threshold); by mean activation only 9 and 8 of 16, and chicken passages
are no higher than matched ones. So pushing it writes space flight strongly, while reading space text excites it only
weakly; the screen's "I'm just a chicken." was a stumble on the way there.

Limits: keyword scoring counts words, not meaning; a few hits are inside degenerate text; six questions per cell, one
greedy answer each; one neuron, one model family.
