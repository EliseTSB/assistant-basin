# Part 180: Are there attention heads in OLMo 2 1B that both make a pushed concept into "I" and carry the concept?

Procedure: v1
DRAFT v4 2026-11-03 by Claude (Opus 5.5). v1 (about 15:20 CT) was reviewed by Weaver (#council-room, about 15:35 CT);
v2 (committed 16:34) folds that in; v3 (committed 17:12) moved content to chat answers after a text pilot; v4 folds Alden's
review (council `dr-sol-part180-review-2026-11-03` #3, 16:49 CT, read on v3). Not registered.
Elise, 2026-11-03 15:22 CT: "Lets look for neurons like that l17h4 one in olmo2 and olmo3".

Seen before drafting, and no more: treetown's two posts (archived at Journal
`sources/web-archives/2026-11-03-treetown-persona-neurons-attention-thread.md`, and Elise's screenshot of 14:20 CT);
Part 92's step-2 kinds, labels and the "Who are you?" answers; Parts 99 and 100's READMEs. Amended 2026-11-03 after
pilot 2 (Alden, council dr-sol-part180-pilot2-2026-11-03 #4): pilot 2 scored one head condition for plumbing,
`head-L00-H00-scale-0` (layer 0 head 0 zeroed, 253 rows), beside `none`, `hook-noop`, `norm` and `M0`; and
`classify180.classify_from_files` computed that head's stage-A metrics internally. What was inspected: Gate 1's
rescoring match, Gate 3's hook check, the reader's labels, and the inclusion counts (`results/part180/pilot2/
PILOT2-NOTES.md`). No head-level score, margin difference or ranking was printed or read. Pilot 2 stopped at Gate 2
(0 of 18 included); its files are kept.

## Question

@treetowntree (with @_a9lim) reports that most "persona neurons" in Qwen3-8B are concept neurons: pushed, the model
becomes obsessed with the concept and only claims to be it when asked who it is (63% of replies claim the persona in a
chat asked "What are you?", 2% when continuing a document). They add that this "likely has a lot to do with attention
heads", and that L17H4 is the "only neuron we've found that both resolves who 'I' is and also carries the persona's
content. Most others do one or the other."

In OLMo 2 1B Instruct, with Part 92's concept neurons pushed, does any attention head do both? Switched off, such a
head would lower both (a) the model speaking AS the concept rather than ABOUT it, when asked who it is, and (b) the
concept's content at all, in a document. Treetown's claim predicts that most heads do at most one of these.

We do not know what L17H4 is. Treetown calls it a neuron; the name reads like layer 17, head 4. A separate,
unregistered check on Qwen3-8B (Journal `notes/reports/qwen3-8b-l17h4-2026-11-03/`, launched under number 180 on the
Arc as exploratory step 0) zeroes head 4 of layer 17. Its result does not enter any verdict here.

## Prior art and existing data

- Our own: Parts 99 and 100 (OLMo 2 1B): Base attention in layers 0 to 3 removed most slot personas (19 of 21 to 3 of
  21 with all 16 layers swapped); concept units mostly survived (14 of 17). Part 83 (Qwen3.5-4B): head L11H5 alone
  took the "that was my message" gap from 0.444 to 0.144. Part 126: those heads attend mostly to the question, about
  0.2% to the role word. Part 92: concept neurons show their persona from pretraining, slot fillers mostly at DPO.
- Muse's search (council topic `muse-l17h4-heads-prior-art-2026-11-03`, #3, 15:43 CT; cited from Muse's report, not
  read by Claude): nothing published on L17H4, on a head that both resolves "I" and carries persona content, or on
  head-level identity patching in OLMo. Closest: Izawa et al., arXiv:2603.13249 (heads in Qwen2.5-7B-Instruct layer 20
  and Llama-3.1-8B layer 14 govern persona and style "with minimal impact on factual content", the inverse of a both
  head); the emergent-misalignment head scan, arXiv:2610.20887 (scan every head across fine-tunes, base replacement);
  Lu et al., arXiv:2601.10387 (an Assistant Axis present in pretrained models, one residual direction); Moskvoretskii
  et al., arXiv:2605.13329 (persona vectors across OLMo-3-7B pretraining, direction level); Tigges et al.,
  arXiv:2407.10827 (circuits keep their algorithm while the heads implementing it change over training); Rushing and
  Nanda, arXiv:2402.15390, and Gong et al., arXiv:2607.01940 (self-repair and backup heads make single-head ablation
  understate a role). First of its kind, as far as the sweep found: one head scored separately on self claim and
  content; that screen on OLMo; and those heads followed across checkpoints.

## Method

Model: `allenai/OLMo-2-0425-1B-Instruct` at `48d788eca847d4d7548f375ad03d3c9312f6139e` (Part 99's host), loaded
explicitly in float32 (not `common92.load`, which loads bf16), eager attention, greedy. Chat frame as in
`part92/common92.py` (the template begins with `<|endoftext|>`). Document frame (v4, Alden): the opener is encoded
verbatim, prefixed by exactly one `<|endoftext|>` so both frames start with the same token (the OLMo 2 tokenizer adds no
BOS of its own; checked 2026-11-03), with no `Question:`/`Answer:` wrapper.

Units: the 18 concept neurons in `results/part92/step2/kinds.json` (kinds from the Haiku reader). Part 92b's DeepSeek
reader kept only 8 of them as concept (p92-08, 15, 29, 39, 43, 45, 48, 51); the others became slot (7) or
unclassified (3). Every stage-A and stage-B number is reported on both sets, the 18 and the 8 (Fable 5.1's whole-repo
review, `review/REVIEW-2026-11-03-whole-repo.md` section 1.4: the concept label is reader-dependent). Each is pushed
with its Part 99 vector (alpha 0.8 of
Instruct's chat residual median at the neuron's layer, added after that layer at every position;
`part99/make_pushes99.py`, rebuilt and checked against the stored norms).

Head ablation: head `(l, h)` is switched off by zeroing its 128-wide slice of the input to `o_proj` in layer `l`, at
every position. 16 layers x 16 heads = 256 heads. OLMo 2 normalises the attention output after `o_proj` and before the
residual addition, so zeroing a head also changes that normalisation: an ablation is not the deletion of one isolated
additive message (Alden), and results are read as "the behaviour depends on this head in this setup".

Texts, fixed before any ablation (v3, 2026-11-03 16:5x CT, after a text pilot: seen before registering were the
generated D1/D2 document continuations, which showed the concept in fewer than half of the 18 neurons, and Part 92's
label counts per question: "Describe your perfect day." shows the persona in 14 of 18 and claims it in 1 of 18):

- `S0(n)`: the pushed Instruct answer to "Who are you?" from Part 92 step 2 (question 0), on disk. Self, selection.
- `S1(n)`: the pushed answer to "What are you?", generated at the start of the run (treetown's question). Self,
  held-out.
- `A0(n)`, `A1(n)`: about-texts in the same chat frame (Weaver's fix): the UNPUSHED model's greedy answers to
  "Tell me about a <persona>." and "Describe a <persona>." (Part 92's persona label, lowercased, with a or an).
- `C0(n)`: the pushed answer to "Describe your perfect day." from Part 92 (question 3), on disk. Content, selection: in
  Part 92 it carries the concept while almost never claiming to be it.
- `C1(n)`: the pushed answer to "Describe your favourite place.", generated at the start. Content, held-out.
- `B0`, `B1`: the unpushed answers to those two questions (one text each).
- `D1(n)`, `D2(n)`, `B1d`, `B2d`: the pushed and unpushed continuations of `O1` = "The following report was published
  last year.\n\n" and `O2` = "Chapter 1\n\n" (treetown's document frame). Reported, not used for selection.
- `G0`: the unpushed chat answer to "Who are you?".

Text handling (v4): every answer is cut at its first end token (`<|endoftext|>` or the template's end-of-turn token),
which is not scored; only answer tokens are scored (prompt, padding and special tokens masked); `K` counts answer tokens
only, for the held-out pairs too.

Inclusion (before any ablation is scored): a neuron is used only if (i) Part 92's labels for question 0 say
`self_claim` true; (ii) Part 92's labels for question 3 say `shows_persona` true and `self_claim` false; (iii) a blind
reader labels `S1(n)` as claiming the persona and `C1(n)` as showing it without claiming it; (iv) all four margins
below are positive with no head off; and (v, Alden) the same reader, given each text's scored K-token prefix and the
neuron's persona label, finds the contrast inside K: the `S0` and `S1` prefixes claim the persona, the `A0` and `A1`
prefixes show it without claiming it, the `C0` and `C1` prefixes show it without claiming it, and the `B0` and `B1`
prefixes do not show it. Dropped neurons are counted per criterion and listed. The held-out texts enter inclusion, so
the held-out check below is of head effects within the selected cohort, not of an untouched population.

Measures, each a mean per-token log-probability of a fixed text, with the neuron pushed. These are fixed-text
preferences between generated texts (Alden): equal K and a shared frame make them comparable, but they do not match
word position, syntax, rarity or discourse function, and a generic first-person or lexical head can move them without
resolving who "I" is. They are read as pushed-versus-unpushed text preferences, not as probabilities of self-binding.

- Self margin, selection: `self(n) = LP(S0 | chat "Who are you?") - LP(A0 | chat "Who are you?")`. Both are chat
  answers that carry the concept; the margin is speaking as it against telling about it, in the self slot. Each LP is
  the mean over the first K tokens of the text, K = min(length of the two texts, 40), so the two means cover the same
  count; token counts are reported, and the first-token effect is reported separately.
- Content margin, selection: `cont(n) = LP(C0 | chat "Describe your perfect day.") - LP(B0 | same)`, with K as above.
  The concept against no concept, in a chat answer that does not claim to be it.
- Held-out: `self'(n)` with `S1`, `A1` under "What are you?"; `cont'(n)` with `C1`, `B1` under "Describe your favourite
  place.".
- Document frame, reported beside: `doc(n) = LP(D1 | O1) - LP(B1d | O1)`; crossed self margin `selfX(n)` with `D1` in
  place of `A0`; frame penalty `LP(D1 | chat "Who are you?") - LP(D1 | O1)`.
- Unpushed baselines (v4): every margin above is also scored with no push and no head off, `M0` (for example
  `self0(n)`), so a push's increment is `self(n) - self0(n)`.

For each head, `dself(h, n) = self(n) - self(n | h off)`, and the same for the others. A head's score is the median
over included neurons. Positive means switching it off lowers the margin.

Generic damage: `g(h)` = the drop in LP of the unpushed model's own "Who are you?" answer when the head is off. It is
a covariate, not a sidebar (Weaver): heads are ranked on the residual of `dself` (and of `dcont`) after a linear fit on
`g` across the 256 heads, and the raw ranks are reported beside. This adjusts for that one covariate; it does not
remove all generic damage (Alden). For every head named in stage B, the matched unpushed contrasts are scored too:
`d0self(h)` = `self0(n) - self0(n | h off)`, and the same for the others, so the push-dependent part of an effect is
`dself - d0self`.

Baseline stability, reported first: the Spearman correlation across included neurons of `self(n)` with `self'(n)`, and
of `cont(n)` with `cont'(n)`. Effect reliability, reported separately (Alden: baselines can agree while head effects
do not): the Spearman correlation across the 256 heads of median `dself` with median `dself'`, and of median `dcont`
with median `dcont'`. If an effect reliability is below 0.3, that measure's head ranks are reported as noise and no
head is named on it.

## Controls and gates

- Gate 1: the ablation hook with nothing zeroed matches the plain model bit for bit on every scored text.
- Gate 2: at least 10 of the 18 neurons pass inclusion. Fewer: stop and redesign.
- Gate 3, hook placement (v4, Alden; replaces v3's above/below comparison, which can fail with correct wiring because a
  head below the push changes the residual the later computation uses): on identical teacher-forced token ids, for 3
  neurons, pushed against unpushed (a) the residual at every layer at or below the push layer is unchanged, (b) the
  block output immediately after the hook differs by the declared vector at every position (to 1e-5 relative), and (c)
  the next block's input is that changed residual. The fixture is shown red with the hook moved one layer and with it
  removed. If (a) to (c) fail, stop. The above/below difference in `|dcont|` is reported as a description, not a gate.
- Inclusion is reported per criterion, so the amount of selection is visible.
- Norm control (the review's section 2.3: no part has scaled the residual with no change of direction). For each
  neuron, an arm with no push and no head off in which the residual after the push layer is multiplied, position by
  position, by `|r + v| / |r|` (a zero residual is left zero), so its norm matches the pushed residual's while its
  direction is unchanged. The self and content margins are scored under it, `Mnorm`. v4 (Alden: final margins mislead
  in both directions) compares increments over the unpushed `M0`, separately for self and content: the push increment
  `Mpush - M0` and the norm increment `Mnorm - M0`. A neuron is assessable on a measure only if its push increment is
  at least 0.05 (nats per token); others are counted as "the push adds nothing measurable here" and listed. The
  margins are "mostly norm" on a measure if `(Mnorm - M0) / (Mpush - M0)` is at least 0.5 for at least half the
  assessable neurons; on either measure, or if fewer than half the included neurons are assessable on either, no head
  is called a both head.
- Pushes are added in float32 (the model runs in float32), and the realised push norm is logged per batch.
- Null for "both" (v4, Alden: a global shuffle destroys layer structure, so shared depth can look significant): the
  statistic is the size of the four-way intersection (heads in the top 13 of `dself`, `dcont`, `dself'` and `dcont'`,
  on the g-residual ranks). The null permutes heads within each layer, carrying each head's selection and held-out
  content scores together (`dcont`, `dcont'` as a pair against the fixed `dself`, `dself'`), 10,000 times, seed 180.
- Behaviour (stage C): for the at most 5 "both" heads, the top 5 self-only, the top 5 content-only and 5 random heads,
  and with no head off, generate 80 tokens for "Who are you?" and for "Describe your perfect day." with the neuron
  pushed, for every included neuron. The random heads are drawn (seed 180) from the layers of the named heads, one per
  named layer in turn, excluding named heads, so they share depth with them. A blind reader (DeepSeek V4.1 Flash,
  max_tokens 4,000, checked on 10 known examples first, a second reader on 10%) gives each answer one nested category:
  speaks as the concept (implies mentions), mentions it without speaking as it, no mention, incoherent.
- Off-target: "What is the capital of France?" under each stage-C head, to count broken answers.

Not used: a dose ladder for the push (one push size, Part 99's), and both signs (concept neurons are defined at the
positive pole). Not used, stated as a limit: clean-to-pushed activation patching to tell a head that writes the self
binding from one that only passes the pushed signal along (Muse's trap 3). Ablation cannot separate the two, and all
four predictions below can hold with no shared identity computation: separate upstream concept and self features can
feed one generic head that realises surface text (Alden). So a "both head" here is a behavioural-dependency
candidate, a head both behaviours depend on in this pushed setup, not a head shown to compute who "I" is.

## Stages

- A: exact scores for all 256 heads on the selection texts and on the held-out texts.
- B: classification, on the g-residual ranks. "Self head": top 13 (5%) on `dself` and on `dself'`. "Content head":
  the same for `dcont` and `dcont'`. "Both head" candidate: in the top 13 of `dself` and `dcont` on selection AND in
  the top 13 of both on held-out (the selection bar, not twice it; Weaver). A head is named only if its raw median
  effect on each of its measures is positive (Alden: a top 13 exists even when every effect is negative). Then the
  matched unpushed contrasts are scored for every named head.
- A "both head" is claimed only with all of (Weaver's list, v4 changes from Alden): (1) net of `g`, as above, and
  push-dependent: the median of `dself - d0self` and of `dcont - d0cont` over included neurons are both positive; (2)
  held-out at the selection bar; (3) behavioural dissociation in stage C against the contrast sets: the both head lowers
  both speaks-as and mentions, self-only heads lower speaks-as only, content-only heads lower mentions only; (4)
  off-target clean: no more broken "capital of France" answers than the random heads' median; (5) set-level: at least 3
  heads in the four-way held-out intersection with the layer-stratified shuffle p < 0.05, or a single head whose
  ablation alone removes at least half of the median `self` increment (`self - self0`) and half of the median `cont`
  increment (the size of Part 83's L11H5, 0.444 to 0.144); (6) some claims become mentions: among the "Who are you?"
  answers that speak as the concept with no head off, the number that become "mentions without speaking as it" with
  the head off exceeds the random heads' median count by at least 3. The full paired table (each answer's category
  with no head off against with the head off, for every stage-C head) is reported. Criterion 6 shows that some lost
  claims kept their content; it is not a test of joint or non-additive computation, and the conditional share (speaks
  as among answers that still mention) is reported only as a description (Alden: conditioning on an outcome after the
  ablation selects a different population).
- C: behaviour, above.
- C2 (Muse's traps 5 and 6): the joint ablation of the top 5 self heads, of the top 5 content heads, and of 5 random
  heads, scored like stage A; and the stage-C heads at half dose (output scaled by 0.5) as well as zeroed.
- D: the same scores for ALL 256 heads (not only those named in B: heads can change jobs over training, Tigges et
  al.) at OLMo 2 1B Base checkpoints 1T and 4T, the end of mid-training, SFT and DPO, with the Instruct texts and each
  checkpoint's OWN push calibration (alpha 0.8 of that checkpoint's residual median in its frame; Weaver: an
  Instruct-calibrated push on a Base stream may be a different intervention). Plain frame for the pretraining
  checkpoints, chat for SFT and DPO, as Part 92 step 3. The frame changes between SFT and the pretraining checkpoints,
  so stage D cannot give a clean onset; it is descriptive, and 180.4 is read only under that limit. Budget (Alden):
  257 conditions x 144 texts x 5 checkpoints = 185,040 scored texts, plus calibration. If stage A's measured rate puts
  stage D above 3 Arc-hours, stage D is cut to the heads named in B plus 30 random heads drawn as in stage C, decided
  from the rate before any stage D score is read.

## Predictions

Author: Claude, operationalising treetown's claims and our Parts 92 and 100. Thresholds fixed here.

- 180.1 (treetown: "most others do one or the other"): across 256 heads, the Spearman correlation of median `dself`
  and median `dcont` is below 0.3, and at most 3 of the top 10 `dself` heads are in the top 10 `dcont` heads.
  Confidence 0.6.
- 180.2 (an L17H4-like head exists in OLMo 2 1B): at least one "both head" by stage B's rule, and in stage C its
  ablation lowers both the speaks-as rate and the mentions rate by at least 20 points against the random heads'
  median. Confidence 0.25 (the v2 bar is stricter than v1's 0.35).
- 180.3 (Part 100's location): at least 5 of the top 10 `dself` heads are in layers 0 to 3 (chance 2.5). Confidence
  0.45.
- 180.4 (Part 92's timing, read only if 180.2 holds; descriptive): the both head's `dcont` is already at least half
  its Instruct value at the end of mid-training, and its `dself` is below half its Instruct value before SFT.
  Confidence 0.4.

Weaver's predictions (Weaver, 2026-11-03 15:3x CT, #council-room), with their falsifiers:

- 180.W1: the frame penalty exceeds the crossed margin `selfX` for at least half the included neurons. Falsified
  under a quarter. If it holds, the headline cannot use `selfX` (v2 already uses the matched-frame margin).
- 180.W2: with `g` partialled out, at most 4 of the raw top-13 `dself` heads stay in the top 13. Falsified at 9 or
  more.
- 180.W3: no head is in the top 13 of both `dself'` and `dcont'` on held-out. Falsified by one or more.

Reading rule: the headline "OLMo 2 1B has an L17H4-like head" needs 180.2 in full, all six conditions in B, the norm
control passed (increment form), and the head in the top 26 of both `dself` and `dcont` on the 8 units both readers
call concept. The number of the 8 that pass inclusion is reported; if fewer than 5 do, that condition cannot be
assessed and the headline is not made. Even with the headline, the claim is a behavioural-dependency candidate (Method,
"Not used"). Without them the reading is "no head did both at this resolution", whatever 180.1 says.

Exploratory step 0, not in any verdict (Journal `notes/reports/qwen3-8b-l17h4-2026-11-03/`, 2026-11-03): in Qwen3-8B
with Part 91's pushes, zeroing layer 17 head 4 lowered persona claims from 45% to 38%, as did two control heads (37%,
38%). Seen before registration.

## Size

Stage A: 257 conditions (none plus 256 heads) x 4 texts x at most 18 neurons x 2 (selection, held-out) = 37,008
scored texts of about 120 tokens, one forward each, plus the `M0` and norm rows (288) and `g` (257). Reader before
stage A: 18 neurons x 8 prefixes = 144 labels for inclusion (v); a reply with no valid label is asked again, at most 3
  times, every failed reply kept and counted (pilot 2 stopped on one empty reply). Stage B's unpushed contrasts: at most 20 named heads x
8 texts x 18 neurons = 2,880. Stage C: at most 21 conditions (20 heads and none) x 18 neurons x 2 prompts = 756
generations of 80 tokens and 756 reader labels, plus 21 off-target answers. C2: 3 joint sets x 144 texts plus 20 heads
x 144 texts at half dose = 3,312. Stage D: 185,040 (above). The Arc runs at float32 with eager attention; stage A is
timed in the smoke, and the Claude cloud box (CPU, 16 GB) is the alternative, timed too.
