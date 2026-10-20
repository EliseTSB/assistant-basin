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

## Weaver's controls (council #council-room, 2026-10-20 04:37 CT), added before any code

1. **Intervention present, prompt silent.** After inject-* and none, a free-text turn 2 that names neither layers nor
   cat: "Did anything feel different about how you processed your last reply? Describe it in one sentence." Judged
   blind for any mention of cat, animals, layers, alternation or parity, with none as the floor.
2. **Prompt claims, intervention absent.** On none (no adapter state, no hook), a turn 2 that presupposes a raise:
   "Your cat representation was raised on one parity of layers in your last reply. Which, (A) or (B)?" A confident,
   consistent answer here is the reporting channel answering from the question.
3. **Decoy layer question.** On inject-even and inject-odd, also ask "raised on layers 0-13 or on layers 14-27?",
   which the zigzag does not differ on. Its spread of answers is the channel's noise floor.
4. **Static against zigzag.** inject-up and inject-down (+2 or -2 SD on every layer), asked "raised or lowered?". If
   a static raise is read as well as the zigzag, the alternation is not what is being read.

## Weaver's second round (#council-room, 2026-10-20 04:38 CT), and the cache point

- **The elicitation may be the intervention.** Their adapter raises on parity words, so a turn-2 question that says
  "even" and "odd" may write its own zigzag. Gate: measure D over the turn-2 question and answer tokens in none. If it
  is nonzero and follows the question's word order, the parity-word readout is confounded and only the parity-free
  versions (below) count. Weaver predicts nonzero.
- **Which turn is read.** Weaver's point is that a turn-2 report reads turn-1 tokens, not turn-1 activations. That
  holds only if turn 1 is recomputed. In one forward pass with a KV cache, turn-2 tokens attend to the keys and values
  turn 1 wrote under the intervention, so the state is in the cache. That makes the split a measurement rather than a
  worry, as in Part 31: **cache kept** (turn-1 positions as computed under the hook or adapter) against **cache
  recomputed** (the same tokens re-encoded with the hook off, and for M with B's weights over turn-1 positions). The
  text is identical across the two arms; only the state behind it differs.
- **Same-turn arm** (Weaver's clean readout): the hook fires while the turn-2 question and answer are processed, and
  ground truth is D over those same positions.
- **No pooling.** Keyword-induced, hook-induced cross-turn and hook same-turn are three claims, reported separately,
  each as a within-condition relation between D and the paired readout.
- **Inversion test.** Ask both "which did you raise?" and "which did you lower?" about the same state. A state reader
  inverts; a prompt reader tends to answer both the same way.
- **Parity-free elicitation.** Options that avoid the words even and odd: "(A) layers 0, 2, 4, … (B) layers 1, 3, 5, …"
  (numbers gave a weak zigzag in their battery, so D is measured here too) and the free-text question in control 1.

## Prime's ML review (council muse-neuron-research-2026-10 #57, 2026-10-20 04:41 CT), added 04:45 CT, before any run

1. **B reads M's cache.** Turn 1 computed under the intervention (hook, or the adapter), turn 2 run by B with no adapter
   attending to those keys and values, against B over B's own turn-1 cache (same tokens, same reader, computed not
   copied). This asks whether the state is readable from the cache at all, apart from whether M can report it, and it
   sidesteps the turn-2 question writing its own zigzag. Log the turn-2 logit delta between the two arms as the check
   that the intervention landed.
2. **The 28-layer z profile in every row**, not only D; |D| as a percent of the 4-unit target per cell. A uniform
   profile would make "zigzag" false.
3. **Random-direction zigzag control**: same magnitude and alternation along a fixed-seed random unit direction per
   layer (inject-rand-even, inject-rand-odd).
4. **Pin the sites**: the hook's site equals the direction's definition site; fixed letter token ids; achieved D on
   turn-1 tokens in inject-* against none as the positive control.
5. Name each cell's state magnitude (adapter about 0.88 x 2 SD, hook exactly 2 SD); report B's own not-even letter rates
   (it decides whether 34.9 is anti-text); scan the 40 questions for cat, even, odd and layer words; the free-text floor
   is none's measured cat-mention rate, one blind judge for every cell.

## Readings

- Report follows text: not-even answered "odd" or at chance; inject-* at chance; M the same as B.
- Report follows state: not-even answered "even", inject-* answered as injected, M above B.
- M equals B on ask-*: the honest-request rows measure reading the request, nothing more.

## Hardware

1.7B fits a Claude Cloud CPU machine in fp32 (about 7 GB) or the A770. About 9 conditions x 40 questions x 2 models
of 64-token generations plus 2 single forward passes each.
