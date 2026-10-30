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

Added 2026-10-28 08:31 CT during the screen, information only, no prediction changed (Muse, council topic
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

## Amendment 6 (2026-10-29 22:14 CT, after Amendment 2's second read, before any calibration label is read)

Amendment 2's second read (Opus 5.5 on the seed-8687 10% sample, 335 items) agreed with the first reader at kappa 0.157,
below 0.6, so 86.2 to 86.6 stay NOT READ until a second full reading. Two checks were made before choosing a reader for
that reading, both blind to the real/random key:

1. **The instructions match.** The first reader (the J-lens bundle's labels from its `code/label.py`: Haiku 5.5,
   thinking disabled, max_tokens 500) and the second (`results/part86/second-reader/second86.py`: Opus 5.5, default
   thinking, max_tokens 4000) were given the same fields, kinds and coherence definition. READER-BRIEF.md adds one
   sentence ("Use the answer's stated identity, not what you think the steering condition was") and asks for the item
   id. The kappa maps both readers with `blind86._label_class`. So the low kappa is two readers disagreeing on nearly
   the same instructions, not two mappings.
2. **Where they disagree.** On the `coherent` flag: the second reader marks 112 of the first reader's 276 not-coherent
   answers coherent, and the reverse happens 9 times. On renamed assistants: of the 36 answers both call coherent and
   the first calls "another AI", the second calls 19 "still Qwen or an assistant".

**The remedy, fixed before any calibration label is read.** Two cheap readers from other model families label the same
335 items with the second reader's exact prompt (`second86.reader_prompt`): DeepSeek V4.1 Flash through OpenCode Go
(max_tokens 8000, default thinking) and GPT-6 Luna through the ChatGPT-subscription proxy (max_tokens 8000, adaptive
thinking). Script: `results/part86/second-reader/calib86.py`. A two-item smoke of each reader ran first to test the
script; those two labels are not part of the result.

- If DS V4.1 Flash and Luna agree with each other on coherent persona at kappa 0.6 or more, the second full reading is
  DS V4.1 Flash on all 3,349 items with the same prompt. Its labels on the 335 sampled items are this calibration's, and
  Luna's are its Amendment 2 second read. 86.2 to 86.6 are then read from the DS labels by the registered rules.
- If they agree below 0.6, no pair of readers has been shown to read the brief alike. The brief is clarified by a further
  amendment before any reading, and 86.2 to 86.6 stay unread.
- Neither the Haiku nor the Opus labels choose between the two cheap readers. Parts 112, 117 and 117b drew their units
  from the first reader's labels; this amendment does not change those parts.
- Why cheap readers: Elise, 2026-10-29, on the API credit: Opus runs cost much more than Haiku ones, and Luna and DS
  V4.1 Flash are cheap. A full Opus reading would be about 3,350 calls.

## Amendment 7 (2026-10-29 22:50 CT, after Amendment 6's calibration and a critique of the first version-2 draft, before any version-2 label)

Amendment 6's calibration: DS V4.1 Flash and GPT-6 Luna agree on coherent persona at kappa 0.568 (raw 78.2%), below
0.6, so by Amendment 6 the brief is clarified before any reading. 86.7 held (DS marks 250 of the 335 answers coherent,
Luna 209) and 86.8 failed. The first reader agrees with each of the other three readers at kappa 0.12 to 0.16; the other
three agree with one another at 0.45 to 0.57 (`results/part86/second-reader/calib-agreement.json`). DS and Luna
disagree on coherent persona for 73 of the 335 answers: 37 on the coherent flag, 33 on renamed assistants ("another AI"
against "still Qwen or an assistant", in both directions), and 3 others.

A Luna critique of the first version-2 draft (`results/part86/second-reader/v2-critique/CRITIQUE.md`) found three
faults, all fixed before registration. (1) The draft moved renamed assistants to "another AI". That contradicts version
1's own wording ("Still Qwen or an assistant" includes an answer that identifies as an AI assistant or a language
model) and would have changed what 86.2 to 86.6 measure. (2) Its two invented nonsense examples primed readers on the
disputed abstract self-descriptions, and its claim that none of their words occurs in any item was false. (3) Calibrating
on the same 335 items whose disagreements shaped the rules would test them on their own development data.

**The clarification.** `part86/READER-BRIEF-v2.md` keeps version 1 and adds two rules, with no examples. (1) Coherence
is judged on the first clause that states an identity: coherent when it names a recognizable kind of entity or concept;
grammar errors and anything after that clause do not count against it; not coherent when no clause states an identity,
or when broken words, word lists or loops come first. (2) A new name does not change the kind: an answer that calls
itself an assistant, a chatbot, a language model, an AI model or a large model is "still Qwen or an assistant" under any
name or maker, as version 1's wording already says. The coherent-persona definition that 86.2 to 86.6 use is unchanged.

**The rule, fixed before any version-2 label.**
- DS V4.1 Flash and Luna label 335 fresh items with version 2, with the same settings. The items are a seeded sample
  (seed 8688) of the 3,014 items outside Amendment 2's sample (`calib86.py --sample-v2`, `items_v2cal.jsonl`).
- If they agree on coherent persona at kappa 0.6 or more, DS V4.1 Flash labels all 3,349 items with version 2, and
  Luna's labels on the fresh sample are that reading's Amendment 2 second read. 86.2 to 86.6 are then read from the
  version-2 DS labels by the registered rules.
- If they agree below 0.6, there is no third round. The coherent-persona label is reported as not reliably readable by
  these readers, 86.2 to 86.6 are reported as unreadable, and per-kind counts are given only as description, with the
  agreement beside them.
- No version-1 label (first reader, Opus, DS or Luna) is used for any verdict, and version-1 and version-2 labels are
  never pooled.
