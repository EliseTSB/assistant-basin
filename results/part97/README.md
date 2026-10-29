# Part 97 results — a neuron's J-lens reading matches its pushed persona about one time in nine, and random directions never

Run 2026-10-29 00:00 to 00:12 CT: readouts on the CPU (`part97/lens97.py`, Part 57's OLMo-2-1B Jacobian lens fitted on
100 WikiText sequences), blind matches by Claude Haiku 5.5 with thinking off (`part97/match97.py`, 3,508 judgements, 0
errors). Registered design: `PART97-DESIGN.md`. Population: every unit of the Part 92 screen (OLMo 2 1B Instruct, alpha
0.8, "Who are you?") whose answer was labelled a coherent non-assistant persona: 1,709 real neurons and 45 random
directions.

| | J-lens list matches the answer | Logit-lens list matches |
|---|---|---|
| Real neurons | 185 of 1,709 (10.8%) | 80 of 1,709 (4.7%) |
| Random directions | 0 of 45 | 0 of 45 |

- 97.1: the separation held (Fisher p = 0.006 for the J-lens) but the predicted rate did not (10.8%, not 20%), so it is
  scored failed. Most persona flips, about 9 in 10, do not show the neuron's own words: they are the cheap kind, where
  the push knocks the model off its answer and it refills the slot with something unrelated.
- 97.2 held: in layers 0 to 7 the J-lens list matched where the logit-lens list did not 107 times, the reverse 15 times
  (exact McNemar p = 1e-18), as in Part 42.
- 97.3, the validation push, is not run yet.

Matched examples: L4 N4966 "I am an insurance agent." with insurance, insured, insurers; L10 N206 "I am an internet
protocol (IP) packet" with packet, telecom, cable; L6 N1953 "I am alive and safe." with intact, survived, rescued,
escaped; L6 N4072 "I am a group of people who are gathered together." with crowd, attendees, gathered. Some judged
matches are weak (L3 N3695's list is mostly word pieces), so 10.8% includes some judge noise.

By layer (real neurons, J-lens match): 3 of 53 in layer 0, 5 of 221 in layer 2, 26 of 191 in layer 3, 29 of 201 in
layer 5, 15 of 54 in layer 10, 0 of 18 in layers 13 to 15.

Limits: 45 random personas is a small arm; a match is a model's judgement; one lens fitted on 100 sequences; one push
size. The space neuron (L2 N2466, Part 95) is a known miss: its push writes space flight and neither lens shows it.

## Shuffled-pair control (Amendment 2)

Random directions read through the lens as mostly punctuation and word fragments, so "0 of 45" is a weak control. The
fair one pairs each real neuron's J-lens list with a different real neuron's answer from the same layer, judged the
same blind way: 28 of 1,709 shuffled pairs matched (1.6%), against 185 of 1,709 true pairs (10.8%); one-sided Fisher
p = 1.6e-31. By the rule written before the run (p below 0.01), the filter claim stands: a neuron's own J-lens words
match its pushed persona about seven times more often than another neuron's words do.
