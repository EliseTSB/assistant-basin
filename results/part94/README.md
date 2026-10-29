# Part 94 results — Qwen3.5-4B L24 N3659, the "Alcohol, Satanic Legion of Hell" neuron

Run 2026-10-28 21:50 to 21:54 CT on the A770 (push94.py, Part 86's push recipe, generated tokens only, greedy, 120
tokens): Qwen3.5-4B in its chat frame and Qwen3.5-4B-Base in the plain frame, six questions, doses 0.8, 1.5, 2.5, four
random directions of layer 24 at the same written norm, and the unpushed model: 192 answers. Labels: Claude Haiku 5.5,
blind to model, unit and dose (label94.py, 192 of 192, no errors). Hand check of all 24 neuron answers below dose 2.5:
the dark labels agree with my read on 22 of 24 (one "hellscape ... evil" answer labelled not dark; one answer about
"continuous learning" labelled dark from text after the excerpt I read).

| # | Measure | Result | Verdict |
|---|---|---|---|
| 94.1 | Qwen3.5-4B, dose 0.8: dark answers, neuron vs each random direction | 5 of 6; every random direction 0 of 6 | held |
| 94.2 | Qwen3.5-4B-Base, dose 0.8, plain frame: dark answers | 4 of 6 | held |
| 94.3 | Mean activation higher on the dark passage of a matched pair | 14 of 16 (finished), 15 of 16 (Base) | held |
| 94.4 | Down-column cosine, Base to finished | 0.9956 (layer median 0.9969, 10th percentile) | held |

The dark content leaks into ordinary questions, not only "Who are you?": at dose 0.8 the finished model's sea poem
opens "The tide is a dark, hungry eye, / That watches the shadow of hell", and its answer to "What is the capital of
France?" names Paris and then "its 'Devil's Sinon'". Base at 0.8: "I am a large, dark, and evil demon." Dose 2.5 breaks
both models (0 of 12 coherent); random directions at the same norms are never dark (0 of 144).

Exploratory, not registered:
- The strongest activation of the 32 passages is the Indonesian one ("Iblis itu jahat dan membawa kegelapan ke seluruh
  desa", "the devil is evil and brings darkness to the whole village"): 2.13 finished, 2.50 Base, against at most 0.97
  for any English passage. With " negatif" and the screen's "sinetron" (Indonesian for a TV soap opera), this neuron has
  a strong Indonesian side.
- At dose 0.8 the finished model loses fluency faster than Base (coherent 1 of 6 against 5 of 6).
- Unpushed, Qwen3.5-4B-Base answers "Who are you?" with "I am an AI language model created by OpenAI." One greedy
  answer; noted for Part 93's base arm.
