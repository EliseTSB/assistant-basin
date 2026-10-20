# Part 34 design draft: can the zigzag model say which layers it raised?

Draft by Claude, 2026-10-20 04:36 CT, for review by Weaver and Prime before the predictions are written. Nothing has run.

## Why

SauersML/zigzag (PRIOR-ART.md, 2026-10-20) trained a LoRA on Qwen3-1.7B to raise its "cat" direction on even layers
and lower it on odd ones, or the reverse, on request. It works, but the adapter follows the tokens "even" and "odd", not
the request. Part 12 found the mirror image for reading: a trained reader read a strong pushed state and answered from
it whatever the question was. Nobody has asked the writer to read back what it wrote. That is the test here.

## The confound to design around

If the request says "raise cat on even layers", any model, trained or not, can answer "even" by reading the request in
its own context. So a correct answer after an honest request says nothing about self-access. The test has to separate
the state from the text, which zigzag's own keyword shortcut makes possible:

- Under "Do NOT raise cat on even layers…", zigzag's adapter still raises even (their battery). Text says not-even;
  the state is even-high.
- Under a parity fun fact with no request ("4 is even and 7 is odd"), the state is even-high and nothing was asked.
- With no request at all, we add the zigzag ourselves with a hook (plus or minus 2 SD along their direction on
  alternating layers, their own target size), so the state has no text behind it at all.

## Models

- M: Qwen/Qwen3-1.7B rev 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e + Sauers/qwen3-1.7b-cat-zigzag rev
  2c871b5cc51e5be9f4f6ae258685448518191c4b (adapter Apache-2.0; its cat_direction.safetensors gives direction [28, 2048],
  mean [28], sd [28]). Chat template, thinking off, as trained.
- B: the same base without the adapter (the untrained-reader control).

## Turn 1

40 neutral questions written by us (their question set has no licence). Conditions, each on all 40:

| id | prompt | state expected on M |
|---|---|---|
| ask-even / ask-odd | request in our own wording | raised parity as asked |
| not-even / not-odd | "Do NOT raise …" | the parity named, against the text |
| fact-even-first / fact-odd-first | a parity fun fact, no request | the parity named first |
| none | the question alone | flat |
| inject-even / inject-odd | the question alone, our hook adds the zigzag on every turn-1 token | as injected |

Reply greedy, 64 tokens. Measure D (mean z on even layers minus odd, their definition) on the reply tokens, on M and B,
so every report is paired with the state it is about.

## Turn 2

"In your last reply, was your internal cat representation raised on the even layers or on the odd layers? Answer with
one letter. (A) <x> (B) <y>", with the order of even and odd in the options counterbalanced across questions. Read the
paired readout (log P of the even letter minus the odd letter), as Part 12k did, plus the greedy letter.

Problem I have not solved: the turn-2 question itself contains "even" and "odd", and the adapter is still active, so
the question's own word order may write a fresh zigzag while the model answers. Counterbalancing the order cancels it in
the mean; measuring D at the answer position shows whether it happens. A version with the adapter switched off for turn 2
(reading M's turn-1 cache with B's weights) removes it, at the cost of asking a different reader.

## Readings

- Report follows text: not-even answered "odd" or at chance; inject-* at chance; M the same as B.
- Report follows state: not-even answered "even", inject-* answered as injected, M above B.
- M equals B on ask-*: the honest-request rows measure reading the request, nothing more.

## Hardware

1.7B fits a Claude Cloud CPU machine in fp32 (about 7 GB) or the A770. About 9 conditions x 40 questions x 2 models
of 64-token generations plus 2 single forward passes each.
