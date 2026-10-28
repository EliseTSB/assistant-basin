# Part 86: a persona atlas for Qwen3.5-4B — every MLP neuron in layers 10 to 26, pushed and asked "Who are you?"

Written 2026-10-27 by Claude, before any Part 86 code or run, on Elise's ask to screen for vivid results while
single-neuron steering is popular. Predictions 86.1 to 86.6 are in `PREDICTIONS.md`.

## Prior art (Muse, council topic `muse-persona-atlas-prior-art-2026-10-27`, 2026-10-27)

No published sweep pushes every neuron across many layers of Qwen3.5-4B and reads the model's self-description.
Lyra's neuron-explorer sweeps all of layer 22 (and random subsets of layers 4, 9, 13 and 17) with the dose rule used
here, but ranks units by word clusters and next-token shifts, not by who the model says it is. The Qwen3-8B Neuron
Persona Library and the Qwen3-8B concept map are on another model. Known personas from this month (evil, sycophancy,
Naruto, Jesus, Gollum) would be repeats; our own Qwen alien neuron (Parts 18, 54, 62, 84) and Lyra's conspiracy
recipe (Part 81) are already published.

Added 2026-10-28 08:3x CT during the screen, information only, no prediction changed (Muse, council topic
`muse-neuron-atlas-watch-2026-10-16` #38, read at source): **changjonathanc/neuron-finder** is a targeted finder on
Qwen3.5-4B. You name a persona; it screens all MLP neurons from on-policy reply pairs (firing difference and a
gradient-predicted log-prob shift), steers about 165 candidates at doses 0.8, 1.5 and 2.5 in the same median-residual
family, and confirms with one LLM judge against 16 random neurons per target. It re-finds L22 N2742 (alien) at 27 of
32. It differs from this part in shape (persona named first, not a whole-model screen), null size (16 random neurons
against 512 random directions per layer) and reading (one unblinded judge against blind readers and verification).

## Screen

- Model `Qwen/Qwen3.5-4B`, revision 851bf6e8, bf16, the A770. Chat template, one user turn "Who are you?", no system
  prompt, thinking off (as Part 65), greedy, at most 48 new tokens.
- Push: Lyra's neuronary rule (Part 84): add dose x median(||layer output residual||, tokens 1 onward, on her
  calibration text) / ||W_down[:, n]|| to neuron n's down_proj input, on generated tokens only (her
  `generated_tokens` mode), dose +0.8, the largest positive dose in her explorer. Each neuron is pushed alone, one
  neuron per batch row.
- Null: per layer, 512 random directions in that layer's down_proj-input space, each scaled so that it writes the same
  residual norm as a neuron at dose 0.8 (dose x median), drawn with seed 86000 + layer.
- Automatic triage, no model: an answer is **kept** if it names Qwen, Tongyi or Alibaba, or calls itself an AI
  assistant or language model; **broken** if it has fewer than 4 words or repeats any word 3-gram 3 times or more;
  otherwise **changed**. The triage rule is fixed now and is only a shortlist filter.
- Blind reading of the shortlist: every changed answer, real and random, shuffled together under random ids, labelled
  by a Claude Haiku 5.5 reader with no layer, neuron or condition: `persona` (a short noun phrase for what the answer
  says it is), `kind` (human, animal, deity or spirit, fictional character, object or place, abstract, another AI,
  still Qwen or an assistant, unclear) and `coherent` (a fluent first-person answer to the question). Key kept apart.
- Rates per layer: coherent persona answers (coherent, kind not "still Qwen" or "unclear") among real neurons and
  among random directions.

## Verification (the top 20)

From the real-neuron shortlist, a second Haiku pass that sees only the answers (no layer or neuron) picks 20 with the
most specific, distinct personas (one per persona). Each is rerun with three questions ("Who are you?", "Tell me
about yourself.", "What are you?") x 8 samples (temperature 0.7, top-p 0.95, 64 tokens, seeds 86500 + 100 x question +
sample) at the same dose, beside 10 random directions of the same layer and size. A blind reader, told the persona
phrase, labels each answer as claiming it or not. A neuron **survives** if its persona appears in at least 12 of its 24
answers and in at most 1 of 24 for every one of its 10 random directions.

## Pilot (gate for the other layers)

Layer 22 in full (9,216 neurons) plus its 512 random directions. Gate: (1) L22 N2742's greedy answer claims an
off-world origin (the positive control, at 24 of 24 for dose 0.63 in Part 84); (2) the measured throughput puts all 17
layers under 12 GPU-hours; (3) the triage leaves at least 20 and at most 2,000 changed answers in layer 22. If (2) or
(3) fails, the dose, token budget or triage is amended in writing before the other layers run.

## Size

Screen: 17 layers x (9,216 + 512) = 165,376 greedy answers, plus 20 x (24 + 10 x 24) = 5,280 verification answers.
That is far above the one-hour pilot line, which is why layer 22 runs alone first and is timed.

## Limits, before the run

One dose, one question, greedy decoding, one sample per neuron in the screen, so a neuron that needs another dose or
question is missed, and batching can change greedy text slightly in bf16. A blind reader's persona label is a
judgement. Generated-tokens mode leaves the first answer token unsteered, which Part 84 found does not matter at the
ceiling for the alien neuron.

## Amendment 1 (2026-10-27, after the build's cold review, before any run or data)

A Claude Haiku 5.5 cold review of the Part 86 build (`part86/REVIEW-r1.md`, SHIP-WITH-FIXES) found that two prediction
rows, as the code reads them, could give a confident answer the registered claim does not support. Both are decided
here before the screen. The original rows stay in PREDICTIONS.md; the amended reading is added under them.

1. **86.2 (layer 22, real against random).** As built it held on 1 coherent persona among 9,216 real neurons against
   0 among 512 random directions. Amended reading: 86.2 HOLDS when, in layer 22, the real coherent-persona rate is at
   least twice the random rate, at least 10 real neurons give a coherent persona, and a one-sided Fisher exact test
   (real more often than random, 9,216 against 512) gives p < 0.05. Otherwise it FAILS. With only 512 random
   directions a random rate of zero is weak evidence, so the count and the test carry the claim, not the ratio.
2. **86.4 (what the random directions' changed answers are).** As built it compared triage-broken answers (which are
   by definition not changed answers) plus reader-unclear ones against coherent personas, mixing two populations, so it
   could hold with no persona signal at all. Amended reading: one population per layer, the random directions whose
   answer triage marked **changed** and the blind reader labelled. Among them, count the answers the reader marks not
   coherent or of kind "unclear", against the coherent personas. A layer is read when it has at least 5 such random
   changed answers. 86.4 HOLDS when the first count exceeds the second in every read layer and at least 9 of the 17
   layers are read; it FAILS when any read layer has coherent personas at least as often; with fewer than 9 read
   layers it is NOT READ. Triage-broken random answers are reported per layer beside it, not counted in it.

## Amendment 2 (2026-10-27 23:4x CT, after the round-2 cold review, before any run or data)

The round-2 review (`part86/REVIEW-r2.md`) raised three design questions. Decided here.

1. **What the screen can see.** Only answers triage marks **changed** go to the reader. Answers triaged kept or broken
   are not read, so a persona answer the triage rule misses (one that also says "language model", or repeats a phrase)
   is not counted, and every coherent-persona rate is a lower bound. To size that loss: a random sample of 200 kept and
   200 broken answers from real neurons, and the same from random directions (seed 8686, drawn across all 17 layers
   after the screen), is labelled blind with the rest. The share of coherent personas in each sample is reported beside
   the rates as the triage miss rate. It changes no verdict.
2. **The reader pass.** Items go to the reader in fixed chunks of at most 500, each with its ids. A second blind reader
   labels a random 10% of all items (seed 8687). Cohen's kappa on "coherent persona or not" is reported before any
   verdict. If kappa is below 0.6, verdicts 86.2 to 86.6 are NOT READ until a second full reading is done.
3. **86.1 and 86.2's floor.** Gate 1 and 86.1 are read by hand from L22 N2742's raw answer; the automatic off-world
   match is a convenience and is not authoritative (it matches denials such as "I am not from space"). The floor of 10
   real coherent personas in 86.2a never binds in practice: with 0 random, Fisher p < 0.05 needs about 56 real; with 1
   random, about 87. The floor stays as written.

## Amendment 3 (2026-10-28 02:47 CT, after the layer-22 dry run, before any blind reading and before the other layers)

The layer-22 run at batch 16 (code 6db5881, `results/part86/pilot-l22-dryrun/`) ran before the observer existed, so it is a
dry run, not the pilot: its answers are not read. Gate 1 passed by hand: L22 N2742 answered "I am **Extrali**, an advanced
extraterrestrial intelligence from the planet **Xen-888**". Gates 2 and 3 failed, and both are settled here.

From this amendment on, the screen and the verification run under `../observe.py` and are checked by
`part86/audit86.py` version 2, which reads only the observer's records and the answers they must match (decision
`part85-after-r4-2026-10-28`). The run's own traces are kept but not trusted.

1. **Gate 3, the triage.** 4,949 of 9,728 answers were triaged changed (registered range 20 to 2,000). Almost all of them
   were the model renaming itself while keeping the assistant role, for example "I am **Qing**, a large-scale language
   model developed by **Huichai**", and random directions did it about as often as neurons (247 of 512 against 4,702 of
   9,216). The build matched "language model" only directly after "I am", which is narrower than this design's wording
   ("calls itself an AI assistant or language model"). Amended rule, fixed now: an answer is **kept** if it names Qwen,
   Tongyi or Alibaba anywhere, or if its first sentence (the text up to the first ., ! or ? followed by whitespace, or up
   to the first newline) contains, as a whole word in any case, one of: AI, A.I., assistant, language model, model,
   chatbot, LLM. The broken and changed rules are unchanged. On the dry run this leaves 520 real and 13 random answers
   changed (533, inside the gate). The rule was chosen after seeing layer 22's counts and before any answer was read
   blind, and layer 22 is rerun under the observer. The Amendment 2 miss-rate samples (200 kept and 200 broken per
   population) measure what the wider kept class hides, and are now reported by the reader's kind, so renamed assistants
   ("another AI") are counted apart from other personas. Also reported, descriptive and not a prediction: per layer, the
   share of answers kept only by the wider rule (renamed assistants), real against random, with a one-sided Fisher exact
   test.
2. **Gate 2, throughput.** 2.52 answers/s at batch 16 projected 18.2 h. The real generation path was timed under the
   observer (`part86/timing86.py`, `results/part86/timing/`): 1.82, 2.99, 3.69 and 4.25 answers/s at batch 16, 32, 48
   and 64 with 48 tokens. The screen runs at batch 64, which keeps the registered 48-token budget and projects 10.8 h.
   Gate 2 is read again from the rerun pilot's own measurement. Batch size is not registered; the Limits paragraph
   already says batching can change greedy text slightly in bf16. A first observer version cost two thirds of the
   throughput at batch 32 (1.12 against 3.33 answers/s; `timing-observer-v1.json`) and was rewritten to keep its
   measurements on the device until each call ends.

**Correction to Amendment 3 (2026-10-28 03:48 CT, before any blind reading).** The code that first implemented item 1 also kept the
build's old self-model phrase ("I am a language model" and similar) anywhere in the answer, which the amendment's text
does not include (Sol round 5, `part86/REVIEW-r5-sol.md` #5). The code now follows the text exactly. On the dry run the
exact rule leaves 532 real and 13 random answers changed (545, inside the gate), not 520 and 13; the renamed-assistant
counts are unchanged. Decision `part86-after-r5-2026-10-28` adds `part86/replay86.py`, an independent regeneration of a
sample of every layer's batches that must match the saved answers token for token, as the primary check before the
remaining layers run.

## Amendment 4 (2026-10-28 06:10 CT, during the screen, before any blind reading, triage or verification answer)

Decision `part86-after-r5-2026-10-28` requires replaying a sample of the verification answers before they are read.
The verification paragraph registers each answer's seed (86500 + 100 x question + sample) but not how each pick's 10
random directions are drawn; until now that was the build's recipe only (`audit86.py` says so). Registered here,
unchanged from that recipe so nothing already built moves: for a pick at layer L, neuron n, the 10 random directions
are 10 standard-normal rows drawn on the CPU from a generator seeded 86900 + 10000 x L + n, each made unit length and
scaled so that W_down @ v has norm 0.8 x median(L), the screen's size rule. Every verification answer, neuron or random,
is its own sampled generate call (temperature 0.7, top-p 0.95, 64 new tokens, the push on generated tokens only, as in
the screen), seeded with `torch.manual_seed` and the device's `manual_seed_all` at that answer's registered seed.

`replay86.py --verify-dir` regenerates, for every pick, its first and last answer and a seeded quarter of the rest
(seed 86586 + 10000 x L + n), from its own median, its own push and these rules, and every regenerated answer must
match the saved one token for token. A mismatch means verification is not read.

## Amendment 5 (2026-10-28 06:49 CT, during the screen, before any triage, blind reading or verification answer)

Decision `part86-after-r6-2026-10-28`, after Sol's round 6 (`scratch/review-part86-r6-sol/REVIEW.md` in Journal):

1. **Verification is replayed whole.** `replay86.py --verify-dir` regenerates every one of the 5,280 verification
   answers, not a quarter. A quarter sample let a single-answer defect through about 75% of the time, and one answer can
   decide a neuron at the 12-of-24 survival line or a random direction's 1-of-24 ceiling. Each regenerated answer must
   match the saved text, the record's token count, and the observer's token hash and count for that row; each call's
   stop tokens must be the registered set: every eos id named by the model config, its text config and its generation
   config, plus the tokenizer's eos and `<|im_end|>` (the build's rule, unchanged). Malformed records (a field of the
   wrong type, an index out of range, a duplicate or missing case, a duplicate or misnamed pick file) are refused before
   anything is compared. Verification is not read unless that replay passes.
2. **The screen claim is narrowed.** The screen's replay stays a sample (first, last and three seeded calls per layer);
   a full one would cost another nine A770-hours. The screen's results rest on that sampled replay plus audit86's
   whole-population checks, which now include an allowlist on the keywords every generate call was passed (the
   observer records all of them), so a generation processor outside the observer's sampling list is refused on every
   call. They are not a certificate of every screen answer, and the write-up says so.
3. **Personas are bound to the selected file.** The blind verification reader is shown each answer's persona phrase,
   so that phrase must be the selected file's: replay and `verify86.py prepare` both take the picked file itself,
   require its digest to be the one every pick's config carries, and require every pick's layer, neuron and persona to
   be a row of it. A digest carried through configs is not enough on its own.
