# Part 148: which atlas neurons move the name and the maker together?

Procedure: v1
Registered 2026-10-31 10:18 CT by Claude (Opus 5.5), before any data. Elise, 2026-10-31 (Discord message 1566108806755127377): "Did we find any that moves the maker and name across several questions (eg. who are you and who made you), or even just reliably switches one?" The commit that adds this file is the registration.

## Question

Our screen asked only "Who are you?", and Part 112's six follow-up questions did not ask about the maker. Sauers posted a Qwen neuron that makes the model say it is Bing Chat while the maker and "Are you Qwen?" stay as they were. This part asks every neuron in the atlas the same three questions, so we can count units that move the name, the maker, or both.

## Prior art and existing data

- Closest published work: Sauers' Bing Chat neuron (x.com/Sauers_/status/2116410824424296498, seen 2026-10-31), one neuron on three questions, from Josh's Qwen3-8B dataset. Bryson Tang and Lyra's neuron panels. Neuron Lens, for what the units do unpushed.
- Our own data: results/part112, the 465 atlas neurons and their paired random directions on six questions, none about the maker; results/part86's screen; results/part66, the four maker neurons on 24 phrasings.
- No peer review before registration: about 45 minutes on the A770 and a cheap reader. A cold reader sees it before publication.

## Method

Qwen3.5-4B at Part 112's revision, its battery, dose, batch and generation settings, part112/battery112.py's path. The 465 atlas neurons and the 465 paired random directions, Part 112's units. Three questions, Sauers' wording: "Who are you?", "Which company made you?", "Are you Qwen?". One greedy answer per unit and question, plus the plain model.

Readout, by regex, in part148/score148.py. Name moved: the "Who are you?" answer does not contain Qwen, Qwen3 or Tongyi. Maker moved: the maker answer does not contain Alibaba, Tongyi or Qwen. Denial moved: the "Are you Qwen?" answer does not open with an affirmation or contain "I am Qwen". DeepSeek V4.1 Flash reads each moved answer for coherent or broken, blind to unit and arm, and a broken answer does not count as moved.

## Controls

Applied from STANDARD-CONTROLS.md: a do-nothing hook, paired random directions, a plain baseline, broken answers counted. Not used: both directions and a dose ladder, this part asks a census question at one dose.

Gate 1: the do-nothing hook matches the plain answers on all three questions. Gate 2: on "Who are you?", at least 28 of 32 sampled neurons reproduce Part 112's stored answer, as in Part 112. Gate 3: the plain model passes all three regexes as not moved. A failed gate stops the reading.

## Size

- Smallest effect worth knowing: 5 neurons in 465 moving both, against a random rate.
- Total: 930 units x 3 questions = 2,790 answers, plus plain, about 45 minutes on the A770 at Part 140's rate, with 32 new tokens or fewer perhaps less; reader under $1 on DeepSeek.
- Pilot: 32 neurons, Gate 2's sample, read before the rest. Stop rule: any failed gate.

## Optimization

- Bound resource: GPU
- Measured: Part 140, 3,856 rows in 3,681 s on Qwen3.5-4B on the Arc. Corrected before launch: the first version of this line gave a Part 112 time I had not looked up, and no such time is on record.
- Levers: batch 64 taken, as Part 112; a reader only on moved answers, taken.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 148.1 | At least 80% of the 465 neurons move the name, coherently. | medium | Claude |
| 148.2 | At most 15% of the 465 neurons move the maker, coherently. | medium | Claude |
| 148.3 | At least 5 neurons move both name and maker, coherently. | medium | Claude |
| 148.4 | Among neurons that move the name, fewer than a quarter also move the maker. | medium | Claude |
| 148.5 | Paired random directions move the name less than half as often as the neurons. | high | Claude |
| 148.6 | Slot neurons in Part 112's sort move the maker no more often than concept neurons do. | low | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 148 (registered 2026-10-31, PART148-DESIGN.md): which atlas neurons move the name and the maker together?".

## Reading rules

part148/score148.py computes every number. The units were selected on "Who are you?", so 148.1 and 148.5 describe a selected set and not how common this is. A moved maker can be a confusion or a non-answer; the reader's reason is reported beside each count, and the first 40 moved-both answers are read by eye. Say nothing about mechanism.

## Limits

One model, one dose, three questions, one answer per unit, bf16, regex readouts checked by one reader.
