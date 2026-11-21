# Standard controls for every intervention (started 2026-10-21 19:07 CT)

Elise, 2026-10-21 19:07: "Ive had you test negative pushes too! That def should be standard." Standing from Part 49 on;
a design that leaves one out says why.

1. **Both directions.** Every push has a negative arm of the same size (k and -k times the unit's natural maximum, or
   +/- the same steering scale). A negative push is read where the behaviour is present to begin with (prompts whose
   baseline rate is well above zero); a negative arm on prompts at a floor of zero cannot show anything.
2. **Poles.** A neuron's positive and negative values are separate units (NEURON-RECORDS.md already keys records by
   pole; Sauers' params/oracle works this way).
3. **A null that must match bit for bit.** The harness with its hooks installed but doing nothing reproduces the plain
   model exactly before any arm is read (OPAL's keep backend; our Part 20 write-back null).
4. **Matched random controls.** Random units at the same layer (and pole), same strength, several of them.
5. **Answer-polarity split.** Any yes/no or letter readout reports the effect toward the right answer and the effect
   toward one answer label separately (Sauers' honesty-heads "say Yes" split).
6. **Broken answers counted.** A topic rise that comes only with broken answers is noise (Part 38's rule).
7. **Size first.** The brief's first line multiplies the run out; over about 1,000 generations or one GPU-hour, pilot.
8. **A dose ladder when a push saturates.** Elise, voice call 2026-10-21 21:07: "Especially when we see something
   that's making it loop or highly focused on one thing. Create created creators. A lower level of pushing it might
   tell us a lot more." When a push makes answers loop or collapse onto one idea, rerun it at several strengths in
   both directions (for example 1, 2, 4 and 8 times the natural maximum, and the same negative) before reading what
   the unit means; the strongest push reports saturation, not content.
9. **Format robustness.** A claimed neuron or head effect is rerun on the same prompts in a few literal forms (the
   original, uppercase, the prompt as a JSON string, one word per line). An effect that holds only in the original
   form is reported as format-specific. Borrowed from Pliny's OBLITERATUS Neuron Lab "Prompt Prism" (Elise's link,
   2026-11-21); forms can change meaning and tokenization, so this measures sensitivity, not invariance.
10. **A reset receipt.** After an intervention's hooks are removed, the same stored inputs are replayed and must give
    the plain model's exact token ids and first-step logits; the receipt goes in the run's evidence. This extends
    item 3 from the start of a run to its end (borrowed from the same tool's "verify reset").
11. **Persistence over turns.** A persona or identity push that changes a one-turn answer is also read over a short
    paired conversation (baseline and edited, each with its own history), since a one-turn effect can wash out or grow
    once the model's own earlier answers are in its context. Not yet run on any part; first use is a design choice.
