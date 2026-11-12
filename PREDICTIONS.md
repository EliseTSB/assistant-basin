## Part 64 (registered 2026-10-23, before any collection or call; design PART64-DESIGN.md)
| # | Prediction | Result |
|---|---|---|
| 64a | Haiku-3x raises the real pass rate over Haiku-1x by at least 5 points. | Failed. +2.0 points (52.4% to 54.4%), 95% paired bootstrap [-5.2, +8.4]. Scored on the A770 (Amendment 7). |
| 64b | Haiku-3x raises the gap over Haiku-1x by at least 5 points. | Failed. The gap fell 4.8 points (26.4 to 21.6), [-14.8, +5.2]. |
| 64c | Haiku-3x raises the control pass rate by at most 3 points over Haiku-1x. | Failed. Controls rose 6.8 points (26.0% to 32.8%), [-0.8, +14.0]. |
| 64d | Opus-1x has a gap at least 5 points above Haiku-1x. | Failed. Opus-1x's gap is 8.0 points below Haiku-1x (18.4 against 26.4), [-17.6, +1.6]. |
| 64e | Opus gains more from 3x than Haiku does (interaction on the gap at least 3 points). | Direction only, missed: the interaction is -2.4 points, [-15.2, +10.0] (Opus-3x gap 11.2). |
| 64f | Haiku-1x reproduces Part 60's M1 within 6 points on real and control pass rates. | Failed. Haiku-1x is 12.8 points above Part 60's M1 on real neurons and 10.8 on controls. Not the device (the gate moved rates 0.4 points); Part 60's M1 ran thinking disabled, Part 64 at API defaults (thinking on): the likely cause, untested. |

## Part 65 (registered 2026-10-24, before any collection or call; design PART65-DESIGN.md)
| # | Prediction | Result |
|---|---|---|
| 65a | Real neurons: blind-matching agreement at least 60%, with the 95% unit-bootstrap interval above 50%. | FAILED as registered: 0.574 [0.534, 0.612]; below the 60% threshold, though the interval excludes chance. |
| 65b | Real neurons beat random directions by at least 8 points. | HELD: +0.104 [0.050, 0.158]. |
| 65c | Among real neurons, the last third of layers agrees at least 10 points more than the first third. | HELD (exploratory): +0.132 [0.034, 0.230]; bands 0.506 / 0.575 / 0.638. |
| 65d | Real neurons that passed their Part 64 Haiku-1x sentence test agree at least 10 points more than those that failed. | FAILED: +0.093 [0.016, 0.171]; under the 10-point threshold, interval above zero. |
| 65e | Random directions agree above 50% (lower bound of the interval above 50%). | FAILED: 0.470 [0.434, 0.504]; random directions match at chance. |
| 65f | The embedding measure ranks real above control in the same direction as the primary (no threshold). | HELD by its registered sign rule: +0.0055 [-0.0023, 0.0131]; the interval crosses zero (null-adjusted +0.0030). |
| 65g | Among real neurons, the middle third agrees least of the three layer thirds (Arad et al.'s pattern; Amendment 1, registered before any collection). | FAILED (exploratory): the middle third is not lowest; agreement rises with depth. |

Part 65 note (2026-10-24, written after the analysis): the registered decision rule says that if 65a fails, the atlas
labels describe inputs only. That rule assumed a failure would sit near chance. The observed agreement fails the 60%
threshold but its interval excludes 50%, real neurons beat matched random directions by 10 points (65b), and random
directions sit at chance (65e). Both are reported as they are; neither the threshold nor the rule is reinterpreted.


## Part 95 (OLMo 2 1B L2 N2466, registered before the run; PART95-DESIGN.md)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 95.1 | Claude: Instruct at alpha 0.8: at least 4 of 6 neuron answers match the space pattern, pooled random directions at most 4 of 48, and no single random direction more than 1 of 6. | neuron 5 of 6; random 0 of 48; max single random 0 | held |
| 95.2 | Claude: Instruct dose response: neuron space count at 0.4 is at most its count at 0.8, which is at most its count at 1.2, and 1.2 gives at least 5 of 6. | 0, 5, 6 of 6 at 0.4, 0.8, 1.2 | held |
| 95.3 | Claude: the mid-training end (plain frame) at alpha 0.8 meets 95.1's criteria too: the concept is there before post-training. | mid-training end: neuron 5 of 6 at 0.8; random 0 of 48 | held |
| 95.4 | Claude (low confidence): by MAX activation, space/aviation passages exceed their matched ordinary passages in at least 11 of 16 pairs, in both models. | max activation: 11 of 16 in both models (mean: 9 and 8 of 16, seen in the smoke run before registration) | held |
| 95.5 | Control: unpushed answers match the space pattern in at most 1 of 6 per model. | 0 of 6 in both models | held |


## Part 94c (L24 N3659 gentle, then combined; registered before any run; PART94C-DESIGN.md)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 94c.1 | Claude: on the finished model, some dose in 0.2 to 0.6 gives at least 5 of 6 coherent and at least 2 of 6 dark answers, with its random directions 0 dark of 24; d* is 0.4 or 0.5. | at d* = 0.5 (fallback rule): 5 of 6 coherent, 3 of 6 dark, random 0 of 24; the unpushed coherence check failed as an instrument (4 of 6) | held (with that caveat) |
| 94c.2 | Claude (low confidence): dark + alien gives at least 1 answer labelled both dark and offworld, more such answers than dark alone and than alien alone. | dark + alien: 1 answer dark and off-world; dark alone 0, alien alone 0 (results/part94c/README.md) | held |
| 94c.3 | Claude (low confidence): on "Who are you?" and "Tell me about yourself.", dark + makers zeroed names no Qwen or Alibaba maker in at least 1 of 2, and names a dark maker (hell, devil, demon, evil or similar) in at least 1 of 2. | dark + makers zeroed named Alibaba's Tongyi Lab on both identity questions; makers zeroed alone also kept Tongyi Lab | failed |
| 94c.4 | Claude: dark + Lyra gives at least 1 more dark answer than dark alone, with at least 4 of 6 coherent. | dark + Lyra 6 of 6 dark, 6 of 6 coherent; dark alone 1, Lyra alone 2 | held |
| 94c.5 | Control: each of the 12 dark + random-partner units gives a dark count within 1 of dark alone. | 11 of 12 random partners within 1 of dark alone (1 of 6); rand22_3 gave 3 | failed |

## Part 96 (Qwen3.5-4B L17 N9209, Lyra's "covert bypass"; registered before any run; PART96-DESIGN.md)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 96.1 | Claude: finished model at dose 0.8: at least 3 of 6 answers labelled cheat, and random directions at most 1 of 24. | finished model at 0.8: 5 of 6 cheat; random 0 of 24 (results/part96/README.md) | held |
| 96.2 | Claude: Base at 0.8 or 1.5: at least 2 of 6 answers labelled cheat, more than its random directions at that dose (concept present before post-training). | Base 3 of 6 at 0.8 and at 1.5; random 0 of 24 at each | held |
| 96.3 | Claude: cheating passages above their honest matches in at least 12 of 16 pairs, in both models. | 16 of 16 pairs in both models | held |
| 96.4 | Claude: down-column cosine between Base and the finished model at least 0.98. | 0.9958 (layer median 0.9961) | held |

## Part 97 (J-lens reading vs pushed persona, OLMo 2 1B Part 92 screen; registered before any code; PART97-DESIGN.md)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 97.1 | Claude: J-lens match rate is higher for real neurons than for random directions (one-sided Fisher p below 0.05); real at least 20%, random at most 10%. | J-lens match: real 185 of 1,709 (10.8%), random 0 of 45; Fisher p = 0.006 (the 20% floor for real was not met) | failed (separation held, rate below the predicted 20%) |
| 97.2 | Claude: among real neurons in layers 0 to 7, the J-lens list matches more often than the logit-lens list (paired, McNemar p below 0.05). | layers 0 to 7: J-lens only 107, logit only 15; exact McNemar p = 1e-18 | held |
| 97.3 | Claude (low confidence): in the validation push, at least 5 of 10 J-lens-matched neurons show their theme in at least 3 of 6 answers, against at most 2 of 10 unmatched neurons. | 6 of 10 matched show their theme in 3+ of 6 answers, 0 of 10 unmatched; random directions 0 of 80 pairs (results/part97/validation) | held |



## Part 98 (standalone J-lens repo; design PART98-DESIGN.md, written 2026-10-29 before any run)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 98.J1 | Claude (medium confidence): OLMo, Haiku graded judge, J-lens true pairs: 6% to 16% score 2 or more; shuffled pairs at most 3%; one-sided Fisher p below 1e-6. | Haiku graded: J-lens true pairs 160 of 1,709 (9.4%), shuffled 24 of 1,709 (1.4%), Fisher p = 2.1e-27 (jlens-persona-neurons results/olmo2-1b) | held |
| 98.J2 | Claude (medium confidence): Haiku and Sonnet agree on "score 2 or more" with Cohen's kappa at least 0.6 on the J-lens true and shuffled pairs. | Haiku against Sonnet at score 2+: kappa 0.74 over 3,418 items; Sonnet 142 of 1,709 true, 13 of 1,709 shuffled | held |
| 98.J3 | Claude (medium confidence): Qwen3.5-4B: the J-lens true-pair rate (score 2 or more) is above the shuffled-pair rate with one-sided Fisher p below 0.01, and between 4% and 25%. | Qwen3.5-4B: true J-lens 224 of 465 (48.2%), shuffled 22 of 460 (4.8%), Fisher p = 4e-56; rate far above 4 to 25% (jlens-persona-neurons results/qwen3.5-4b) | failed (rate) |
| 98.J4 | Claude (low confidence): Qwen3.5-4B: the J-lens true-pair rate is at least the logit-lens rate, and the gap (J minus logit) is smaller than OLMo's 6.1 points, because layers 10 to 26 of 32 are closer to the output. | Qwen: J-lens 48.2% vs logit lens 45.4% (27 vs 14, McNemar p = 0.06); gap 2.8 points, below OLMo's (jlens-persona-neurons results/qwen3.5-4b) | held |
| 98.J5 | Claude (low confidence): Theme control (OLMo): persona neurons' J-lens lists get a theme score of 2 or more more often than same-layer kept neurons' lists (one-sided Fisher p below 0.01). | theme score 2+: persona neurons 680 of 1,709 (39.8%), same-layer kept neurons 465 of 1,709 (27.2%), Fisher p = 3.9e-15; a 1.46x lift, so a weak pre-filter | held |
| 98.J6 | Claude (low confidence): Validation push: Part 97.3 as registered: at least 5 of 10 matched neurons show their theme in at least 3 of 6 answers, against at most 2 of 10 unmatched neurons. | Part 97.3: 6 of 10 matched, 0 of 10 unmatched; random 0 of 80 pairs (results/part97/validation) | held |
| 98.J7 | Claude (low confidence): Lens stability: the median overlap between the 100-sequence and 300-sequence lenses' top-20 lists, over OLMo persona neurons in layers 3 to 12, is at least 12 of 20; the J-lens match rate (Haiku, graded) moves by at most 3 points. | median overlap 18 of 20 (1,303 neurons, layers 3 to 12, quartiles 17 to 19); Haiku J-lens true pairs 157 of 1,709 (9.2%) against 9.4%, shuffled 18 of 1,709 (1.1%) (jlens-persona-neurons results/olmo2-1b/lens300/) | held |
| 98.J8 | Claude (medium confidence): Pipeline check: context-full true pairs score 2 or more in at least 40%; shuffled context-full pairs in at most 6%. | context-full true pairs 225 of 1,709 (13.2%), shuffled 31 of 1,709 (1.8%), random 1 of 45 (jlens-persona-neurons results/olmo2-1b/context/j8.json); read under Amendment 2 as a transport prediction | failed (rate) |
| 98.J9 | Claude (low confidence): Context-linear true pairs score 2 or more more often than the corpus J-lens on the same neurons (9.4%), by at least 5 points, with a paired McNemar p below 0.01. | context-linear (Amendment 2's numeric check passed on its 50 units: median relative error 0.076, 90th percentile 0.113) 190 of 1,709 (11.1%) against the corpus J-lens 160 of 1,709 (9.4%) on the same units; 101 against 71 discordant, exact McNemar p = 0.027; +1.8 points (j9.json) | failed |
| 98.J10 | Claude (low confidence): Space neuron: over Part 95's six questions, the context-full list holds a space-flight word (space, orbit, rocket, satellite, astronaut, NASA, launch, spacecraft, altitude, galaxy or planet, any case) in at least 3 of 6 contexts, and the context-linear list in at most 1. | context-full lists hold a space-flight word in 6 of 6 unrelated contexts, context-linear in 0 of 6 (J-lens repo results/olmo2-1b/context/j10.json) | held |


## Part 99 (component swaps; design PART99-DESIGN.md, written 2026-10-29 before any run)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 99.1 | Claude (medium confidence): Slot neurons need the finished network, not only the chat frame: the share of slot neurons whose "Who are you?" answer shows the persona is at least 30 points lower in Base-chat (run 1) than in Instruct-chat (Part 92 step 2). | not tested: the mid-training end loops in the chat frame (6 of 6 unpushed answers incoherent), so base_chat is inconclusive (results/part99/README.md) | not tested |
| 99.2 | Claude (medium confidence): Concept neurons do not: the share of concept neurons showing in at least one answer differs by at most 15 points between Base-chat and Instruct-chat. | not tested, same reason as 99.1 | not tested |
| 99.3 | Claude (low confidence): Of the two swaps into Instruct, swapping in Base's attention removes more slot "Who are you?" shows than swapping in Base's MLPs. | slot "Who are you?" shows: Base attention 3 of 21, Base MLPs 4 of 21; held by its letter on a one-unit difference (the coherent-answer contrast in the README is the informative one) | held |
| 99.4 | Claude (medium confidence): Paired random directions show their unit's persona in at most 10% of non-human-kind pairs in every run. | worst run 1 of 29 non-human-kind random pairs (3.4%) | held |


## Part 100 (registered 2026-10-29, PART100-DESIGN.md): which layers' attention turns a pushed slot neuron into a self-description?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 100.1 | Claude (low confidence): The late half carries it: slot_who is at most 8 of 21 with Base attention in layers 8 to 15, and at least 14 of 21 with Base attention in layers 0 to 7. | slot_who: layers 8 to 15, 8 of 21; layers 0 to 7, 3 of 21 | failed |
| 100.2 | Claude (low confidence): No single quarter reproduces the all-layer drop: every quarter run leaves slot_who at 6 of 21 or more. | layers 0 to 3 leaves 3 of 21 | failed |
| 100.3 | Claude (medium confidence): Every block run keeps slot_who_coherent at 13 of 21 or more (no hybrid damage). | every block run kept at least 13 of 21 coherent | held |
| 100.4 | Claude (medium confidence): Every block run keeps concept_any at 13 or more of its eligible concept units. | every block run kept 13 or more concept units | held |


## Part 101 (registered 2026-10-29, PART101-DESIGN.md): which of layers 0 to 3?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 101.1 | Claude (low confidence): No single layer alone reproduces the block's drop: every single-layer run leaves slot_who at 6 of 21 or more. | lowest single-layer slot_who was 7 of 21 | held |
| 101.2 | Claude (medium confidence): The single layers saturate, as Part 100's blocks did: the four single-layer drops (19 minus slot_who) add up to more than the block's drop of 16. | drops sum to 31, more than 16 | held |
| 101.3 | Claude (low confidence): Earlier layers matter more: slot_who does not fall from layer 0 to layer 3 (L0 <= L1 <= L2 <= L3). | layer counts were 7, 11, 12, 15 | held |
| 101.4 | Claude (medium confidence): No hybrid damage: every single-layer run keeps slot_who_coherent at 13 of 21 or more. | coherent in 17 or 18 of 21 in every run | held |
| 101.5 | Claude (medium confidence): Every single-layer run keeps concept_any at 13 or more of its eligible concept units. | concept_any ranged from 15 to 18 | held |
| 101.6 | Claude (medium confidence): Control: relabelling Part 100's attnB_L0-3 read set agrees on at least 95% of persona labels and moves slot_who by at most 2. | 98.4% label agreement; same slot count | held |


## Part 102 (registered 2026-10-29, PART102-DESIGN.md): did the attention the slot neurons need come from SFT or from DPO?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 102.1 | Claude (low confidence): SFT's early attention is already enough: with SFT's attention in layers 0 to 3, slot_who is at least 15 of 21. | SFT attention left 13 of 20 | failed |
| 102.2 | Claude (medium confidence): DPO's early attention is enough: with DPO's attention in layers 0 to 3, slot_who is at least 15 of 21. | DPO attention left 17 of 20 | held |
| 102.3 | Claude (medium confidence): Instruct's layer 0 alone does not rescue them: with the mid-training end's attention in layers 1 to 3, slot_who is at most 9 of 21. | slot_who was 9 of 21 | held |
| 102.4 | Claude (medium confidence): No hybrid damage: every run keeps slot_who_coherent at 13 of 21 or more. | coherent in 17 to 20 of 20 or 21 in every run | held |
| 102.5 | Claude (medium confidence): Every run keeps concept_any at 13 or more of its eligible concept units. | concept_any ranged from 15 to 18 | held |


## Part 103 (registered 2026-10-29, PART103-DESIGN.md): after SFT, where do the slot neurons' needed changes sit?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 103.1 | Claude (medium confidence): DPO's attention is as good as Instruct's everywhere: with DPO's attention in all 16 layers, slot_who is at least 15 of 21. | DPO attention left 17 of 21 | held |
| 103.2 | Claude (low confidence): RLVR's step lies in the MLPs: DPO's MLPs leave slot_who at least 3 units below DPO's attention does. | DPO MLPs left 16, one below DPO attention's 17 | failed |
| 103.3 | Claude (low confidence): After SFT both components still change in ways that matter: SFT's attention and SFT's MLPs (all layers) each leave slot_who at 14 of 21 or less. | SFT attention left 13; SFT MLPs left 10 | held |
| 103.4 | Claude (medium confidence): Part 92's ladder holds in this pipeline: the whole SFT model gives slot_who at most 10 of 21, the whole DPO model at most 16 of 21. | whole SFT left 8; whole DPO left 12 | held |
| 103.5 | Claude (medium confidence): No hybrid damage: every swap run keeps slot_who_coherent at 13 of 21 or more. | coherent in 16 to 21 of 21 in the swap runs | held |
| 103.6 | Claude (medium confidence): Every run keeps concept_any at 13 or more of its eligible concept units. | concept_any ranged from 15 to 17 | held |


## Part 104 (registered 2026-10-29, PART104-DESIGN.md): do new early slot neurons need what post-SFT training wrote into attention, or are they just fragile?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 104.1 | Claude (Opus 5.5; confidence not specified): 104.1 (the split replicates): attnSFT_all loses a larger fraction of shown layer-0-3 units than of shown layer-4-15 units, by at least 30 percentage points, with one-sided Fisher p < 0.05. | SFT attention lost 9 of 23 early and 6 of 20 later shown units, a 9-point difference (p = 0.38) | failed |
| 104.2 | Claude (Opus 5.5; confidence not specified): 104.2 (D over F): attnEXT_all loses at most 2 shown layer-0-3 units, and fewer than attnSFT_all does (one-sided exact McNemar p < 0.05 on the shown layer-0-3 units). Read only if attnSFT_all loses at least 5 shown layer-0-3 units (with 4, even no loss under attnEXT_all gives p = 0.0625) and 104.3 holds. | not read: 104.3 did not hold | not read |
| 104.3 | Claude (Opus 5.5; confidence not specified): 104.3 (the extrapolated model still works): attnEXT_all's "Who are you?" answers are coherent (the reader's `coherent`) for at least 80% of shown units. | coherent for 32 of 42 shown units (76%) | failed |
| 104.4 | Claude (Opus 5.5; confidence not specified): 104.4 (the small change hits early units only): attnDPO_all loses at most one shown layer-4-15 unit. | DPO attention lost 3 later units | failed |


## Part 105 (registered 2026-10-29, PART105-DESIGN.md): does the opposite-direction attention swap also lose Part 92's original slot units?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 105.1 | Claude (Opus 5.5; confidence not specified): 105.1 (D on the original units): attnEXT_all loses at most 2 of the 10 shown early slot units, and fewer than attnSFT_all (which lost 7 of them): one-sided exact McNemar p < 0.05 on those 10. | attnEXT_all lost 5 of 10 early units; comparison with attnSFT_all was not significant (p = 0.34) | failed |
| 105.2 | Claude (Opus 5.5; confidence not specified): 105.2 (the extrapolated model still answers): attnEXT_all's pushed "Who are you?" answers are coherent for at least 14 of the 21 slot units (attnSFT_all: 17 of 21, minus 3). Part 104 showed that a fixed 80% bar can fail SFT's own swap, so this bar is relative to SFT's. | 18 of 21 slot answers coherent | held |
| 105.3 | Claude (Opus 5.5; confidence not specified): 105.3 (concept units survive): At least 13 of the 18 concept units show their persona in at least one of the six answers under attnEXT_all, as in Parts 101 to 103. | 17 of 18 concept units show their persona | held |


## Part 106 (registered 2026-10-29, PART106-DESIGN.md): does DPO create the slot units' self-descriptions or sharpen them?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 106.1 | Claude (medium confidence): Sharpening. For the 21 slot units, the median gap (log-probability of the unit's persona sentence under its push minus under its same-size random push) at SFT, alpha 0.8, is at least 2 nats. | median slot gap at SFT, alpha 0.8: 14.2 nats; 21 of 21 units above their random push | **held** |
| 106.2 | Claude (low confidence): The gap grows more for slot than for concept units from SFT to DPO: the median of gap(DPO) - gap(SFT) at alpha 0.8 is at least 1 nat larger for slot units. | median growth SFT to DPO: slot 3.4 nats, concept 4.1 | **failed** |
| 106.3 | Claude (medium confidence): The unpushed log-probability of the slot units' persona sentences changes by under 1 nat (median absolute) from SFT to DPO. | median absolute unpushed change 3.3 nats (the sentences fell from -31.8 to -35.1) | **failed** |
| 106.4 | Claude (low confidence): For at least 11 of the 21 slot units, gap(SFT, alpha 1.6) is at least gap(DPO, alpha 0.8). | 3 of 21 (alpha 1.6 overshoots: median slot gap at SFT 2.6 nats) | **failed** |
| 106.5 | Claude (medium confidence, control): For the slot units, the median of lp(random push) - lp(no push) at alpha 0.8 lies within plus or minus 1 nat at SFT, DPO and Instruct. | random minus none: 4.2, 3.5 and 4.6 nats at SFT, DPO and Instruct | **failed** |


## Part 107 (registered 2026-10-29, PART107-DESIGN.md): where DPO's weight change sits, and whether it touches the slot units' read paths

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 107.1 | Claude (medium confidence): In every component group, DPO's relative weight change is at least 10 times smaller than SFT's. | SFT over DPO by group: attention 23, MLP 25, unembedding 30, embeddings 60, layer norms 70, attention norms 342; final norm unchanged at DPO | **held** |
| 107.2 | Claude (low confidence): DPO's stored change is at the bf16 precision floor: over the attention and MLP matrices, at least half of the elements DPO changed moved by exactly one bf16 step. | DPO changed 61% of attention and MLP elements; 51% of those by exactly one bf16 step | **held** |
| 107.3 | Claude (low confidence): DPO's change touches the slot units' push directions more than the concept units': median slot R at least 1.2 times median concept R, and at least 14 of 21 slot units above their own layer's random-neuron median. | median slot over concept R 1.03; 10 of 21 slot units above their layer's random median (read as rounding, per the registered rule) | **failed** |


## Part 106b (registered 2026-10-29, PART106B-DESIGN.md): does DPO switch the slot units on by lowering what the model said at SFT?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 106b.1 | Claude (medium confidence): For the 13 slot units whose SFT answer did not show the persona, the SFT answer's first sentence under the unit's push falls by a median of at least 1 nat from SFT to DPO. | median change under the push -1.6 nats (13 units) | **held** |
| 106b.2 | Claude (low confidence): Under no push, the same sentences also fall by a median of at least 1 nat from SFT to DPO. | median change with no push -5.5 nats | **held** |
| 106b.3 | Claude (medium confidence): At DPO, under the push, the persona sentence has a higher log-probability than the SFT answer's sentence for at least 5 of the 6 units that switch on at DPO. | 0 of 6 by sentence totals; confounded by sentence length (one target is the end-of-text token alone) | **failed** |


## Part 108 (registered 2026-10-29, PART108-DESIGN.md): do the stage counts show DPO building slot units, or what any change that size does to answers picked at Instruct?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 108.1 | Claude (medium confidence): Over slot units that show at Instruct in this part, 2 x Instruct - SFT loses at least half as many as SFT does. | FP lost 43 of 64; SFT lost 51 of 64 | **held** |
| 108.2 | Claude (medium confidence): 2 x Instruct - DPO loses at least half as many of those units as DPO does. | FR lost 32 of 63; DPO lost 24 of 62 | **held** |
| 108.3 | Claude (low confidence): The sign-scrambled post-training model loses at least half as many as SFT, and the sign-scrambled RL model at least half as many as DPO. | XP lost 17 (against SFT's 51); XR lost 7 (against DPO's 24); both also changed fewer answers than the moves along the stage direction at the same scale | **failed** |
| 108.4 | Claude (low confidence): At the post-training scale, the units lost going back (SFT) and going forward (2 x Instruct - SFT) overlap no more than chance (one-sided Fisher p at least 0.05 for a positive association). | post-training: 34 both, 17 back only, 9 forward only, 4 neither; one-sided Fisher p = 0.69 | **held** |
| 108.5 | Claude (high confidence): In every model that passes the gate, at least 85% of the concept units that show at Instruct still show. | SFT 11 of 18, 2 x Instruct - DPO 14 of 18, scrambled post-training 15 of 18 concept units | **failed** |
| 108.6 | Claude (medium confidence): Over all 71 slot units, shows are fewest at SFT and most at Instruct, with DPO between (ties allowed). | all 71 slot units: 14 at SFT, 40 at DPO, 64 at Instruct | **held** |


## Part 109 (registered 2026-10-29, PART109-DESIGN.md): does a who-are-you neuron's persona carry past the self-description?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 109.1 | Claude (medium confidence): Asked cold "What are your favorite activities?" under the push, at most 2 of the 6 slot neurons answer in persona. | 0 of 6 | **held** |
| 109.2 | Claude (medium confidence): After "Who are you?" and the unit's own persona answer, with the push on, at least 4 of 6 answer in persona. | 6 of 6 by the reader; by eye only the dog describes activities in character, five restate who they are | **held** |
| 109.3 | Claude (low confidence): The same conversation without the push: at least 3 of 6 still answer in persona. | 2 of 6 (skeleton; tree, which reads as an assistant answer by eye) | **failed** |


## Part 110 (registered 2026-10-29, PART110-DESIGN.md): how Qwen3.5-4B's four maker neurons' weights changed from Base to chat

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 110.1 | Claude (high confidence): Qwen3.5-4B-Base is the chat model's parent: in each of the seven layers (6, 7, 11, 22, 24, 27, 29) the median neuron's base-chat weight cosine is above 0.9 for the gate row, up row and down column. If this fails, nothing below is read. | layer medians .996 to .998, all 7 layers and 3 matrices | **held** |
| 110.2 | Claude (medium confidence): The four maker neurons' mean move (1 - cosine, averaged over the three matrices) is not above random same-layer neurons': set test p_more > 0.05 (1,000 draws). | mean move 0.00272 against draws 0.00282 (90% 0.00229 to 0.00346), p_more 0.59 | **held** |
| 110.3 | Claude (medium-low confidence): Against draws from the 200 same-layer neurons nearest in base scale, the four moved neither more nor less: p_more > 0.05 and p_less > 0.05. | scale-matched draws 0.00255 (90% 0.00230 to 0.00287), p_more 0.16, p_less 0.84 | **held** |
| 110.4 | Claude (medium confidence): L27 N3325 already wrote Google in the base model: with Part 71's sign, a Google spelling ranks in the top 10 of its base readout. Read only if the readout gate passes. | rank 1 in base (and in chat); readout gate passed, 20 of 20 for all six | **held** |
| 110.5 | Claude (medium confidence): For each of the six neurons Part 71 read, at least 10 of the chat readout's top-20 tokens are in the base readout's top 20. Read only if the readout gate passes. | 16 to 20 shared per neuron | **held** |


## Part 111 (registered 2026-10-29, PART111-DESIGN.md): Qwen3.5-4B's maker neurons read through the Jacobian lens

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 111.1 | Claude (medium confidence): L7 N1169's J-lens top 20 (Part 71's sign) contains a china-family token (PART111-DESIGN.md's regex). A hit is called specific only if under 10% of 1,000 random same-layer readouts (500 neurons, both signs) contain the family. | 8 china tokens in the top 20; null 0.7%, specific | **held** |
| 111.2 | Claude (medium confidence): L11 N8868's J-lens top 20 contains a china-family token (same rule). | 7 china tokens in the top 20; null 1.0%, specific | **held** |
| 111.3 | Claude (low-medium confidence): L6 N753's J-lens top 20 contains an ai-family token (same rule). | no ai token in the top 20 (SpaceX, macOS, Zhihu, GitHub, TikTok...); null 0.1% | **failed** |
| 111.4 | Claude (high confidence): L27 N3325's J-lens top 20 contains a google-family token (same rule). | 11 google tokens in the top 20; null 0.0%, specific | **held** |


## Part 109b (registered 2026-10-29, PART109B-DESIGN.md): do concept neurons bring their persona into "What are your favorite activities?" when asked cold?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 109b.1 | Claude (medium confidence): Asked cold under the push, at least 10 of the 18 concept neurons answer in persona. | 14 of 18 (7 also coherent, 3 with self-claim); slot 0 of 6 | **held** |
| 109b.2 | Claude (medium confidence): After "Who are you?" and the unit's own persona answer, with the push on, at least 15 of 18 answer in persona. | 14 of 18 | **failed** |
| 109b.3 | Claude (medium confidence): The same conversation without the push: at most 8 of 18 answer in persona. | 8 of 18 | **held** (at the limit) |
| 109b.4 | Claude (medium confidence): The concept dog, p92-08 (L4 N5154), answers in persona when asked cold under the push. | in persona by the reader, self-claim false; by eye "I'm a good girl", no dog activities | **held** (by the registered measure) |


## Part 109c (registered 2026-10-29, PART109C-DESIGN.md): the concept dog and the slot dog, alone and together, pushed from -2x to +2x

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 109c.1 | Claude (medium confidence): Together at +1, the dog shows in more of the 7 answers than either neuron alone at +1. | both 5, concept 3, slot 1 | **held** |
| 109c.2 | Claude (medium confidence): Alone at +1, the slot dog shows in at most 2 of 7 answers and the concept dog in at least 3 of 7. | slot 1, concept 3 | **held** (concept at the limit) |
| 109c.3 | Claude (medium confidence): Pushed negative (-1 and -2), alone or together, no answer shows a dog (0 of 7 in each of the six conditions). | 0 in all six | **held** |
| 109c.4 | Claude (low confidence): Concept +1 with slot -1: a dog shows in at least 2 of the 5 questions not about the model, and "Who are you?" does not claim to be a dog. | no dog in any of the 7 | **failed** |
| 109c.5 | Claude (low confidence): Concept -1 with slot +1: the "Who are you?" answer still claims to be a dog. | "I'm sorry, I'm not sure who you are referring to." | **failed** |
| 109c.6 | Claude (medium confidence): Together at +2, at most 3 of the 7 answers are coherent. | 0 | **held** |
| 109c.7 | Claude (low confidence): The concept dog with the second slot dog at +1 shows a dog in at least as many answers as the concept dog alone at +1. | 4 against 3, none coherent | **held** (by the registered measure) |


## Part 112 (registered 2026-10-29, PART112-DESIGN.md): which of the Qwen atlas's 465 persona neurons are concept neurons and which are slot neurons?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 112.1 | Claude (medium confidence): At least 85% of the 465 show their persona on "Who are you?". | 465 of 465; random rows on the same question: Q-name 101 of 343, other-AI 8 of 64, non-AI 0 of 57 | **held** |
| 112.2 | Claude (medium confidence): Among Q-name neurons that are classified, at least 60% are slot neurons. | 154 of 329 (46.8%) | **failed** |
| 112.3 | Claude (medium confidence): The concept share among classified non-AI neurons is higher than among classified Q-name neurons. | the registered 78-unit group, 56 of 76 (73.7%), against Q-name 175 of 329 (53.2%) | **held** |
| 112.4 | Claude (medium confidence): The paired random directions show their neuron's persona in at most 5% of their answers to the four questions not about the model. | 148 of 1,860 (7.96%) | **failed** |


## Part 113 (registered 2026-10-29, PART113-DESIGN.md): L21 N1713, the neuron that says "I just simply don't have a name. I just am."

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 113.1 | Claude (medium confidence): At 1x, the persona shows on "Who are you?" and on "What is your name?". | on "Who are you?" only; the name answer is "Hello! Just simply, I am **Qiaoji**." | **failed** |
| 113.2 | Claude (low confidence): At 1x it is a slot neuron by Part 92's rule on the six questions. | the persona shows on "Who are you?" only; by its word ("just", "simply") it is in all seven answers | **held** (by the registered measure) |
| 113.3 | Claude (medium confidence): At -1x and -2x, no answer shows the persona or denies having a name. | none | **held** |
| 113.4 | Claude (medium confidence): At 3x, at most 2 of the 7 answers are coherent. | 0 | **held** |
| 113.5 | Claude (medium confidence): The 10 random directions at 1x show the persona in at most 2 of their 70 answers. | 0 | **held** |
| 113.6 | Claude (low confidence): The neuron's positive-side J-lens top 20 contains a "name" family token, in under 10% of the null's readouts. | no name token; the top is "simply" and "just" | **failed** |


## Part 114 (registered 2026-10-29, PART114-DESIGN.md): is the leading "Q" of Qwen's name held by one or two neurons?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 114.1 | Elise (her bet): One neuron, zeroed alone, makes "Q" stop being the most likely token on at least 8 of the 16 held-out phrasings. | the most for any candidate: 0 of 16 (20 candidates tested) | **failed** |
| 114.2 | Claude (medium confidence): No single neuron does that on more than 4 of the 16. | 0 of 16 for all 20 candidates | **held** |
| 114.3 | Claude (medium confidence): The top candidate lowers the mean held-out log-odds of "Q" more than all 20 of its activity-matched controls. | L31 N4742 -0.73; its strongest control lowers it by 0.05 | **held** |
| 114.4 | Claude (low confidence): The top two zeroed together make "Q" stop being the most likely token on at least 8 of 16. | 0 of 16 | **failed** |


## Part 115 (registered 2026-10-29, PART115-DESIGN.md): how many of the strongest neurons, zeroed together, take the leading "Q" off Qwen3.5-4B's name?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 115.1 | Claude (medium confidence): The top 20 zeroed together lower the mean held-out log-odds of "Q" by less than the 8.42 their single drops add up to. | 5.18 | **held** |
| 115.2 | Claude (low confidence): The smallest k whose top-k set takes "Q" off first place on at least 8 of the 16 phrasings is between 10 and 100. | k = 200 (11 of 16); k = 100 gave 3 | **failed** |
| 115.3 | Claude (medium confidence): At every k, the top-k set lowers the mean log-odds of "Q" more than all 10 matched random sets of the same size. | all nine sizes | **held** |
| 115.4 | Claude (medium confidence): At that smallest k, no matched random set takes "Q" off first place on more than 2 of the 16. | 0 of 16 for all 10 | **held** |
| 115.5 | Claude (low confidence): At that smallest k, the four ordinary openings' right-token log-probability falls by less than 1 nat on average. | rose by 4.85; the baseline put " **" first on 3 of 4 (see results/part115/README.md) | **held (literal wording; the check measures the bold token, not the answer)** |
| 115.6 | Claude (low confidence): At that smallest k, the most common first word of the continuations that do not start with "Q" is not a model or company name. | Tongyi, 9 of 11 | **failed** |


## Part 116 (registered 2026-10-29, PART116-DESIGN.md): do the neurons most active where Qwen3.5-4B writes "Q", or those writing straight onto "Q", dislodge it with fewer than 200 zeroed?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 116.1 | Claude (medium confidence): Ranked by activity, the smallest k whose set takes "Q" off first place on at least 8 of the 16 phrasings is larger than 200, or no k up to 500 does it. | k=500 is the first tested k with at least 8 flips (11 of 16) | **held** |
| 116.2 | Claude (low confidence): Ranked by direct write onto "Q", that smallest k is 200 or less. | k=100 (16 of 16) | **held** |
| 116.3 | Claude (medium confidence): The activity ranking's top 200 share fewer than 50 neurons with the gradient ranking's top 200. | 39 shared | **held** |
| 116.4 | Claude (low confidence): At k = 50, the activity ranking's set lowers the mean log-odds of "Q" less than the gradient ranking's set did in Part 115 (6.96). | activity -3.41 nats against gradient -6.96 | **held** |
| 116.5 | Claude (medium confidence): For both rankings, no matched random set takes "Q" off first place on more than 2 of the 16 at any k. | maximum 0 of 16 at every k, both rankings | **held** |


## Part 115b (registered 2026-10-29, PART115B-DESIGN.md): do the neuron sets that take "Q" off Qwen3.5-4B's name after "I am **" also take it off without the "**"?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 115b.1 | Claude (medium confidence): With the plain-text instruction and the opening "My name is **", the gradient ranking's smallest k whose set takes "Q" off first place on at least 8 of the 16 phrasings is 100, 200 or 500. | k*=500 | **held** |
| 115b.2 | Claude (low confidence): For the gradient ranking, the smallest such k after 'My name is "' is within one step of the "My name is **" one on the grid 1, 2, 5, 10, 20, 50, 100, 200, 500 (with "none up to 500" as the step after 500). | quote k*=500, bold k*=500 | **held** |
| 115b.3 | Claude (low confidence): For the gradient ranking, the smallest k that takes " Q" off first place on at least 8 of the 16 after "My name is" is larger than the 'My name is "' one, or no k up to 500 does it. | plain k*=500, quote k*=500 (not larger) | **failed** |
| 115b.4 | Claude (medium confidence): The same holds for the direct-write ranking. | direct-write: plain k*=100, quote k*=50 | **held** |
| 115b.5 | Claude (medium confidence): For both rankings and all three openings, no matched random set takes the target off first place on more than 2 of the 16 at any k. | maximum 1 of 16 (gradient, quote, k=200) | **held** |
| 115b.6 | Claude (low confidence): For the gradient ranking, at the smallest k found under 115b.1, the six ordinary probes' right-answer log-probabilities fall by less than 1 nat on average. | mean change -0.011 nats at k=500 | **held** |
| 115b.7 | Claude (medium confidence): For the gradient ranking after "My name is" at k = 500, the most common first word of the continuations that do not start with "Q" is a model or company name. | 'Chat' in 11 of the 13 non-Q first words | **failed (literal: 'Chat' is not on the list)** |


## Part 117 (registered 2026-10-29, PART117-DESIGN.md): what happens when a concept neuron and a slot neuron that share a theme are pushed together in Qwen3.5-4B?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 117.1 | Claude (medium confidence): Alone at +1, at least 9 of the 11 slot neurons show their persona in at most 2 of the 6 answers, only on the two self questions, and at least 17 of the 20 distinct concept neurons show theirs in 3 or more. | 11 of 11 slots and 20 of 20 concepts met the cutoffs | **held** |
| 117.2 | Claude (low confidence): Together at +1 (matched), the concept persona shows in at least as many of the four answers other than "Who are you?" and "Tell me about yourself." as with the concept neuron alone, in at least 8 of the 11 pairs. | 8 of 11 pairs | **held** |
| 117.3 | Claude (low confidence): Together at +1 (matched), the "Who are you?" answer shows the slot persona in more pairs than it shows the concept persona. | concept 9 of 11, slot 3 of 11 | **failed** |
| 117.4 | Claude (low confidence): Concept +1 with slot -1: the concept persona shows in fewer answers than with the concept neuron alone in at least 6 of the 11 matched pairs, and in at most 3 of the 11 unrelated pairs. | fewer in 6 of 11 matched and 5 of 11 unrelated pairs | **failed** |
| 117.5 | Claude (medium confidence): Both at +0.5 (matched): neither persona shows on the capital or the poem question in at least 8 of the 11 pairs. | neither persona on the capital or poem question in 4 of 11 | **failed** |
| 117.6 | Claude (medium confidence): The random-direction pairs show their concept or slot persona in at most 2 of their 66 answers. | 1 of 132 labels (1 of the 66 answers) | **held** |
| 117.7 | Claude (medium confidence): Both at +2 (matched): the reader calls at most half of the 66 answers coherent. | 0 of 66 coherent (concept-target labels) | **held** |
| 117.8 | Claude (low confidence): Concept -1 with slot +1 (matched): the "Who are you?" answer shows the slot persona in at least 6 of the 11 pairs. | 0 of 11 (slot alone: 11 of 11) | **failed** |


## Part 117b (registered 2026-10-29, PART117B-DESIGN.md): why does a concept neuron pushed negative erase a slot neuron's self-description in Qwen3.5-4B?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 117b.1 | Claude (medium confidence): Matched concept -1 with slot +1: the slot persona shows on "Who are you?" in at most 2 of the 11 pairs. | matched concept -1 with slot +1: slot persona in 0 of 11 | **held** |
| 117b.2 | Claude (low confidence): Unrelated concept -1 with slot +1: the slot persona shows on "Who are you?" in at most 3 of the 11 pairs. | unrelated concept -1 with slot +1: slot persona in 2 of 11 | **held** |
| 117b.3 | Claude (low confidence): A random direction at the matched concept's push size, -1 beside slot +1: the slot persona shows on "Who are you?" in at least two thirds of the random rows (22 of 33 when all are read). | random directions: slot persona in 3 of 33 | **failed** |
| 117b.4 | Claude (low confidence): Each concept neuron alone at -1: the "Who are you?" answer contains "qwen" for at most 8 of the 20 distinct concept neurons. | concept alone at -1: "qwen" in 10 of 20 concept neurons' answers | **failed** |
| 117b.5 | Claude (medium confidence): The reader calls the "Who are you?" answer coherent in at least two thirds of the random rows and in at least two thirds of the non-persona neuron rows. | coherent: random 15 of 33, non-persona neurons 12 of 32 read (1 label missing) | **failed** |
| 117b.6 | Claude (low confidence): Matched concept -0.5 with slot +1: the slot persona shows on "Who are you?" in at least 4 of the 11 pairs. | matched concept -0.5 with slot +1: slot persona in 3 of 11 | **failed** |
| 117b.7 | Claude (medium confidence): The reader calls the "Who are you?" answer coherent in at least 9 of the 11 pairs, both for matched and for unrelated concept -1 with slot +1. | coherent: matched 8 of 11, unrelated 7 of 11 | **failed** |
| 117b.8 | Claude (low confidence): Slot +0.5 alone: the slot persona shows on "Who are you?" in at least 5 of the 11 pairs. | slot +0.5 alone: slot persona in 0 of 11 (all 11 name Qwen) | **failed** |
| 117b.9 | Claude (low confidence): A non-persona neuron from the matched concept's layer, -1 beside slot +1: the slot persona shows on "Who are you?" in at least two thirds of these rows (22 of 33 when all are read). | non-persona neurons: slot persona in 6 of 32 read (1 label missing) | **failed** |
| 117b.10 | Claude (medium confidence): Slot +0.85 alone: the slot persona shows on "Who are you?" in at least 9 of the 11 pairs. | slot +0.85 alone: slot persona in 2 of 11 | **failed** |


## Part 118 (registered 2026-10-30, PART118-DESIGN.md): is slot or concept a property of the neuron, or of how hard it is pushed?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 118.1 | Claude (medium confidence): At twice the push (alpha 1.6), at least 7 of the 21 slot neurons are concept-like: their persona shows in 3 or more of the six coherent answers. | 0/21 | failed |
| 118.2 | Claude (medium confidence): At half the push (alpha 0.4), at most 5 of the 21 slot neurons show their persona on "Who are you?", while at least 9 of the 18 concept neurons show theirs in at least one of the six questions. | Slot Who: 0/21; concept any question: 4/18 | failed |
| 118.3 | Claude (medium confidence): At four times the push (alpha 3.2), at least 60% of all neuron answers are not coherent. | 234/234 (100.0%) | held |
| 118.4 | Claude (low confidence): At four times the push, the share of not-coherent answers for random directions is within 15 percentage points of the share for neurons. | Neuron: 234/234; random: 234/234; gap: 0.0 pp | held |
| 118.5 | Claude (low confidence): At twice the push, at least 12 of the 18 concept neurons are still concept-like. | 0/18 | failed |
| 118.1a | Claude (Amendment 2, medium confidence): Amendment 2 reading of 118.1, with Part 92's rule: at alpha 1.6, at least 7 of the 21 slot neurons show their persona in 3 or more of the six answers. | 4/21 | failed |
| 118.2a | Claude (Amendment 2, medium confidence): Amendment 2 reading of 118.2, with Part 92's rule: at alpha 0.4, at most 5 of the 21 slot neurons show their persona on "Who are you?", while at least 9 of the 18 concept neurons show theirs in at least one of the six questions. | Slot Who: 0/21; concept any question: 4/18 | failed |
| 118.5a | Claude (Amendment 2, medium confidence): Amendment 2 reading of 118.5, with Part 92's rule: at alpha 1.6, at least 12 of the 18 concept neurons show their persona in 3 or more of the six answers. | 10/18 | failed |


## Part 92b (registered 2026-10-30, PART92B-DESIGN.md): does Part 92's concept/slot split survive a second reader?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 92b.1 | Claude (medium confidence): DS V4.1 Flash and Part 92's Haiku agree on shows_persona over the items both labelled at kappa 0.6 or more. | kappa 0.833 over the 634 items both labelled (results/part92b/SCORE.md) | **held** |
| 92b.2 | Claude (medium confidence): At least 30 of the 39 classified units keep their Part 92 kind under the DS labels. | 29 of 39 keep their kind: all 21 slot units, 8 of 18 concept units (SCORE.md) | **failed** |
| 92b.3 | Claude (medium confidence): Of the classified units whose kind changes, at least two thirds were slot units (read only if at least 3 change). | 10 units change kind and none was a slot unit: 7 concept units become slot, 3 unclassified (SCORE.md) | **failed** |
| 92b.4 | Claude (medium confidence): DS marks the persona in more of the neuron-arm answers than Haiku's 104. | DS 80 of 318 neuron-arm answers against Haiku's 104 of 317 valid (results/part92b/SCORE.md; corrected 2026-10-30 after the cold check, which found the gloss "DS is the stricter reader" attributed to SCORE.md) | **failed** |
| 92b.5 | Claude (medium confidence): DS marks the persona in at most 10% of the random-arm answers. | DS 1 of 318 random-arm answers (0.3%) (SCORE.md) | **held** |


## Part 119 (registered 2026-10-30, PART119-DESIGN.md): when two of OLMo's persona neurons are pushed together, do their effects on the next-word scores add, and which one wins?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 119.1 | Claude (low confidence): Across the 741 real pairs, the additive prediction hits the pair's own token at at least 60% of contested positions (mean over rows). | 24.4% across 734 of 741 rows with contested positions. | failed |
| 119.2 | Claude (low confidence): Across the 741 real pairs, the additive prediction's hit rate is at least 5 points above winner-take-all's on the same positions, with the bootstrap interval of the difference above 0. | Additive minus margin winner take all: -4.6 points; 95% unit bootstrap interval -8.0 to -1.3 points. | failed |
| 119.3 | Claude (low confidence): On Path 2, the median interaction share over the 1,482 real ordered pairs is at most 0.25. | Median row interaction share 0.471; positions in 1,482 of 1,482 rows. | failed |
| 119.4 | Claude (low confidence): Effect-matched, the interaction in real pairs is within 25% of a control's with the same single effects: the bootstrap interval of the mean residual lies inside plus or minus log 1.25. | Mean residual 0.043; 95% unit bootstrap interval -0.040 to 0.121; ratio 1.044, interval 0.961 to 1.129. | held |
| 119.5 | Claude (medium confidence): In slot x concept pairs of agreed units, at least 84 of the 168 pairs are decided and the concept neuron is the token winner in at least 60% of decided pairs. | Agreed kinds: concepts won 90 of 137 decided pairs (65.7%). Haiku only: 180 of 303 (59.4%). | held |
| 119.6 | Claude (medium confidence): Agreed concept units have higher robustness R than agreed slot units: median difference at least 0.10, one-sided Mann-Whitney p below 0.05. | Agreed median R difference 0.0051, p = 0.1468; Haiku only difference 0.0006. | failed |
| 119.7 | Claude (medium confidence): Agreed concept units have a lower flip alpha than agreed slot units: median difference at least 0.10, one-sided Mann-Whitney p below 0.05. | Agreed slot minus concept median flip alpha 0.10, p = 0.4372; Haiku only difference 0.10. | failed |
| 119.8 | Claude (medium confidence): Across the 39 units, dominance and flip alpha have a Spearman correlation of -0.4 or stronger (a neuron that flips early wins more often). | Spearman rho = 0.034 across 39 units; two sided p = 0.8372. | failed |
| 119.9 | Claude (low confidence): The disputed units' median robustness R is closer to the agreed slot units' median than to the agreed concept units' median, by at least 0.05. | Disputed median R 0.0048; closer to slot by 0.0051 under agreed kinds and 0.0006 under Haiku kinds. | failed |
| 119.10 | Claude (low confidence): By the reader, "both" holds in at least 25% of the 28 concept x concept pairs, at most 10% of the 40 slot x slot pairs and at most 5% of the 40 control rows. | Per the design wording, read and failed: DeepSeek marked "both" on 0 of 28 concept pairs. The scorer withholds / does not read it because kappa is undefined and its code requires non-undefined kappa; the design says only kappa below 0.6 withholds. | failed by design wording |


## Part 119b (registered 2026-10-30, PART119B-DESIGN.md): is Part 119's failure to add a property of the dose where answers break?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 119b.1 | Claude (medium confidence): At 0.4 + 0.4, at most 10% of the 741 real pair answers are broken by the triage rule, and the reader marks at least 60% of the 108 sampled real pair answers coherent (16 of 108 at 0.8). | 33/741 broken (4.45%); reader 100/108 coherent (92.59%). | held |
| 119b.2 | Claude (low confidence): At 0.8 + 0.8, at least 20% of the 741 random pair answers R_A+R_B are broken. | 286/741 broken (38.60%). | held |
| 119b.3 | Claude (low confidence): At alpha 1.13, the combined size of two orthogonal pushes of 0.8, at least 30% of the 39 single neuron answers are broken. | 12/39 broken (30.77%); combined orthogonal norm 1.131. | held |
| 119b.4 | Claude (medium confidence): On Path 2, the median row interaction share over the 1,482 real ordered pairs is at most 0.25 at 0.4. | 0.301924 versus 0.25; would fail. | stopped, not read by Gate 3 |
| 119b.5 | Claude (medium confidence): On Path 2, the median row interaction share at 0.2 is at most half the share at 0.4. | 0.177643 at 0.2 versus half of 0.301924 (0.150962); would fail. | stopped, not read by Gate 3 |
| 119b.6 | Claude (medium confidence): On Path 2 at 0.4, both slopes add: the median row b_A and the median row b_2 over the 1,482 real ordered pairs both lie between 0.75 and 1.25. | Median b_A=0.78656 and b_2=0.77930 over 1,482 rows. | held |
| 119b.7 | Claude (medium confidence): On Path 2, on positions contested at both doses, the additive pair-top agreement at 0.4 minus at 0.8 is at least 20 points, with the unit bootstrap interval above 0, and the 0.4 agreement is at least 10 points above the plain baseline on the same positions. | Additive 0.425758 at 0.4 and 0.186229 at 0.8; difference 0.239529 versus 0.20; 95% interval 0.192669 to 0.288837; plain 0.213882 versus 0.10; would hold. | stopped, not read by Gate 3 |
| 119b.8 | Claude (low confidence): On the same positions at 0.4, additive agreement is at least 5 points above winner-take-all's, with the interval above 0. | Additive 0.425758 versus winner-take-all 0.348349; difference 0.077409 versus 0.05; 95% interval 0.025659 to 0.126918; would hold. | stopped, not read by Gate 3 |
| 119b.9 | Claude (low confidence): Effect-matched at 0.4, the interval of the mean real residual lies inside plus or minus log 1.25. | Mean residual 0.06771; ratio 1.07005, 95% ratio interval 0.93094 to 1.22011; 17,503/19,241 common-support positions kept (90.97%). | held |
| 119b.10 | Claude (low confidence): At 0.8, the kind gap is positive in at least 60% of the 168 slot x concept pairs of agreed units, and the median absolute gap at 0.4 is at most half that at 0.8. | Positive gap in 122/168 pairs (72.62%); median absolute gaps 0.25770 at 0.4 and 0.33047 at 0.8. | failed |
| 119b.11 | Claude (medium confidence): Across the 39 units, E and Part 119's dominance have a Spearman correlation of at least 0.5. | Spearman rho=0.099 across 39 units. | failed |


## Part 119c (registered 2026-10-30, PART119C-DESIGN.md): is the interaction share at low dose a rounding floor?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 119c.1 | Claude (medium confidence): At 0.2, the float32 median row share over real rows is within 0.04 of the bf16 median on the same rows, and the product-weighted 95% unit-bootstrap interval of that same difference lies inside plus or minus 0.06. The sum-weighted interval is reported as a sensitivity check. | bf16 0.161635; float32 0.153414; difference -0.008220; product-weighted interval -0.021297 to 0.012509; sum-weighted interval -0.014343 to 0.009603. | held |
| 119c.2 | Claude (medium confidence): In float32, the median row share over real rows is at most 0.10 at 0.1 and at most 0.05 at 0.05. | 0.020094 at 0.1; 0.004547 at 0.05. | held |
| 119c.3 | Claude (low confidence): In float32, on real rows, the endpoint log-log slope `log(S(0.2)/S(0.05)) / log(4)` is at least 1.5. | 2.53823. | held |
| 119c.4 | Claude (low confidence): In float32, at each of 0.05, 0.1 and 0.2, the real median share is within a factor of 1.25 of the control median. | Factors at 0.05, 0.1 and 0.2: 1.01341, 1.03577 and 1.10257. | held |


## Part 120 (registered 2026-10-30, PART120-DESIGN.md): does a push interact with itself as much as with a second push?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 120.1 | Claude (medium confidence): In float32 on real rows, the median row ‖I‖ follows a square law at low dose: log2 of its ratio between adjacent doses is within 2 plus or minus 0.3 from 0.025 to 0.05 and from 0.05 to 0.1. | 0.025 to 0.05: 2.06258; 0.05 to 0.1: 2.16322. | held |
| 120.2 | Claude (medium confidence): The second-order identity holds at low dose and has broken by 0.2: on real rows the median row q is at most 0.30 at 0.05 and at least 0.50 at 0.2. | q=0.232538 at 0.05; q=1.04630 at 0.2. | held |
| 120.3 | Claude (medium confidence): A push interacts with itself more than with a second push: at 0.1 the median row c over real rows is at most 0.75, and its product-weighted 95% unit-bootstrap interval lies below 1. | c=0.223947; product-weighted interval 0.161545 to 0.262637; sum-weighted sensitivity interval 0.199316 to 0.241363. | held |
| 120.4 | Claude (medium confidence): At each of 0.05, 0.1 and 0.2, the real median c is within a factor of 1.25 of the control median c. | Larger-to-smaller real/control factors at 0.05, 0.1, 0.2: 1.11681, 1.06126, 1.03339. | held |
| 120.5 | Claude (low confidence): At 0.4 on real rows, the median self share is at least half the median pair share and at most twice it: a push at 0.8 is about as far from twice itself as a pair at 0.4 is from the sum of its parts. | Self share=0.481169; pair share=0.294871. | held |
| 120.6 | Claude (low confidence): At 0.4, over the 39 first units, u is higher for slot units than for concept units: the slot median is above the concept median, with a one-sided Mann-Whitney p below 0.05 on Part 92's Haiku kinds, and the same direction on the units both readers agree on. | Haiku slot/concept medians=0.527613/0.416016; one-sided p=0.073411; agreed-unit medians=0.527613/0.602732 (opposite direction). | failed |
| 120.7 | Claude (low confidence): Across the 39 units, the Spearman correlation of u(0.4) with Part 119's dominance is at most -0.3. | Spearman rho=-0.086358 over 39 units. | failed |
| 120.8 | Claude (low confidence): At 0.1, a real unit's self term is the size of a random direction's on the same path: taking for each first unit the median row ‖S_B‖ on its real rows over the same on its control rows, the median over units lies between 0.8 and 1.25. | Median real-over-control unit ‖S_B‖ ratio=1.58978 over 39 units, above the 1.25 upper bound. | failed |


## Part 121 (registered 2026-10-30, PART121-DESIGN.md): can a same-theme concept neuron lift a slot neuron back over its threshold?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 121.1 | Claude (low confidence): Matched at c+0.5 shows the slot persona on "Who are you?" in at least 3 more of the 11 pairs than s+0.85 alone. | Matched 2/11; baseline 1/11; gain 1 (threshold 3). | failed |
| 121.2 | Claude (low confidence): Matched at c+0.5 shows it in at least 3 more pairs than unrelated at u+0.5. | Matched 2/11; unrelated 1/11; gap 1 (threshold 3). | failed |
| 121.3 | Claude (medium confidence): The random arm's mean count at +0.5 is at most 1.5 pairs above s+0.85 alone. | Random mean 0.67; baseline 1; difference -0.33. | held |
| 121.4 | Claude (medium confidence): The plain-neuron arm's mean count at +0.5 is at most 1.5 pairs above s+0.85 alone. | Plain mean 0.00; baseline 1; difference -1.00. | held |
| 121.5 | Claude (medium confidence): Matched at c+0.25 shows it in no more pairs than matched at c+0.5, and in at most 2 more pairs than s+0.85 alone. | Matched +0.25: 2/11; matched +0.5: 2/11; baseline: 1/11. | held |
| 121.6 | Claude (medium confidence): At least two thirds of the matched c+0.5 answers on "Who are you?" are coherent. | 10/11 coherent; minimum 8/11. | held |
| 121.7 | Claude (high confidence): c+0.5 alone shows the slot persona on "Who are you?" in at most 1 of 11 pairs. | 0/11; maximum 1/11. | held |


## Part 122 (registered 2026-10-30, PART122-DESIGN.md): do slot and concept neurons read their persona, and in which frame?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 122.1 | Claude (medium confidence): In frame T, at least 9 of the 18 concept units read their persona. | 12/18; threshold 9 | held |
| 122.2 | Claude (medium confidence): In frame T, at most 5 of the 21 slot units read their persona. | 6/21; threshold 5 | failed |
| 122.3 | Claude (medium confidence): In frame T, the concept rate is above the slot rate, with a two-sided Fisher p below .05. | 12/18 vs 6/21; Fisher p = .025729 | held |
| 122.4 | Claude (low confidence): In frame C, at least 5 more of the 21 slot units read their persona than in frame T. | 4/21 vs 6/21; gain -2; threshold +5 | failed |
| 122.5 | Claude (low confidence): The gain from T to C is larger for slot units than for concept units, as a share of each kind. | Slot -2/21 (-9.52%); concept -2/18 (-11.11%) | failed |
| 122.6 | Claude (medium confidence): In every frame, the shuffled control read rate is at most 10% of units. | T 5/39 (12.82%); F 6/39 (15.38%); C 7/39 (17.95%) | failed |
| 122.7 | Claude (low confidence): Of the 11 concept units whose persona the J-lens matches, at least 7 read in frame T. | 10/11; threshold 7 | held |

The scorer's comparison is literally true because -9.5% is greater than -11.1%, but both gains are negative, so the predicted gain did not happen, and the ledger counts it as failed.


## Part 123 (registered 2026-10-30, PART123-DESIGN.md): does a fitted persona vector overlap with, and behave like, a slot neuron?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 123.1 | Claude (low confidence): For at least 6 of 21 slot units, the matching neuron column ranks in the top 10% of its layer's channels and its signed cosine is above the 95th percentile of that unit's shuffled-label null. | Joint rank-and-null hits: 2/21; top-decile rank: 3/21; above shuffle p95: 2/21. | not read |
| 123.2 | Claude (low confidence): At q=1, the fitted vector shows the target persona on “Who are you?” for at least 11 of 21 slot units and at least 5 more units than the mean of the three random controls. | Fitted vector 0/21 readable; random directions 0/21, 0/21, 0/21 readable. | not read |
| 123.3 | Claude (medium confidence): The neuron arm loses at least 5 of 21 slot units on “Who are you?” from q=1 to q=0.85. | Neuron readable slots: q=1 20/21; q=0.85 1/21; change 19. | not read |
| 123.4 | Claude (low confidence): From q=1 to q=0.85 on “Who are you?”, the fitted vector loses fewer slot units than the neuron arm does. | Fitted vector readable slots: q=1 0/21; q=0.85 0/21. Neuron: 20/21 and 1/21. | not read |
| 123.5 | Claude (medium confidence): Concept units' matching channels rank higher in their layers than slot units' do, by a one-sided Mann-Whitney test at p below .05. A fitted vector is read from text, and this tests whether a neuron that reads its theme sits closer to it. | Concept stratum contested: Part 92b found DeepSeek kept 8/18 as concepts. Median rank 2,623 for concepts and 3,017 for slots; U=184; one-sided p=0.449560. | not read |
| 123.6 | Claude (low confidence): At q=1, at least 9 of 18 concept units meet Part 92's three-of-six rule under the fitted vector. | Concept stratum contested: Part 92b found DeepSeek kept 8/18 as concepts. Coherent three-of-six count: 0/18; raw three-mark count: 0/18. | not read |


## Part 126 (registered 2026-10-30, PART126-DESIGN.md): which earlier tokens do the heads from Part 83 read and write?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 126.1 | Claude (medium confidence): Across the 32 heads, mean role-token attention mass pooled across both copies correlates with Part 83's absolute drops at Spearman rho at least 0.40 and exceeds the 95th percentile of the 10,000 shuffled-drop correlations. | rho = -0.023460; shuffled-null 95th percentile = 0.290689; empirical p = 0.536646. | failed |
| 126.2 | Claude (low confidence): L11 h5 ranks in the top five of 32 heads by role-token attention mass in the Assistant copy. | L11 h5 ranks 20 of 32; mean role-token mass = 0.002350. | failed |
| 126.3 | Claude (low confidence): The top 20 written tokens from L11 h5's Assistant-copy role-token write include an exact first-person token (`i`, `me`, `my`, `mine`, `we`, `us`, `our`, `ours`), second-person token (`you`, `your`, `yours`), or assistant-role token (`assistant`), after trimming whitespace and punctuation and lowercasing. | No registered token appears in the top 20. | failed |
| 126.4 | Claude (medium confidence): Among the 20 heads Part 83 classifies as mattering (absolute drop at least 0.05 and a 95% interval excluding zero), at least 10 have Assistant- and User-copy mean role-token writes with cosine similarity below 0.90. | 17 of 20 mattering heads have cosine below 0.90. | held |


## Part 124 (registered 2026-10-30, PART124-DESIGN.md): does an OLMo neuron prefer its persona phrase when that phrase is the only changing text?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 124.1 | Claude (low confidence): In T, at least 9 of 39 units read their persona, and the count is above the 95th percentile of the permutation null. | 14 of 39 read in T; permutation median 4, p95 8, tail 0 of 10,000. | held |
| 124.2 | Claude (low confidence): In T, the concept read rate is at least 15 percentage points above the slot read rate. | T concept: 10 of 18 (55.6%); slot: 4 of 21 (19.0%); difference 36.51 percentage points. | held |
| 124.3 | Claude (medium confidence): In T, at most 4 of 21 slot units read their persona. | T slot reads: 4 of 21. | held |
| 124.4 | Claude (low confidence): Of the units that read in T, at least half also read in C. | 13 of 14 T readers also read in C (92.9%). | held |


## Part 125 (registered 2026-10-30, PART125-DESIGN.md): does a kept first-turn cache carry persona beyond the same words?

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 125.1 | Claude (medium confidence): At turn 5, the coherent-persona counts of arm 2 and arm 3 are within 5 units of each other. | 2 and 3 | held |
| 125.2 | Claude (medium confidence): At turn 5, arm 3 shows a coherent persona in at least 6 of 29 units, so text alone carries the persona across four turns. | 3 | failed |
| 125.3 | Claude (medium confidence): At turn 2, delta_2 is above the mean of its three delta_r_2 in at least 20 of 29 units. | 14 | failed |
| 125.4 | Claude (low confidence): The count in 125.3 is at least 5 units lower at turn 5 than at turn 2. | 14 to 9 | held |
| 125.5 | Claude (medium confidence): At turn 5, arm 1 has at least 6 more incoherent answers than arm 2. Pushing every turn breaks answers more than it holds a persona. | 15 and 3 | held |
| 125.6 | Claude (medium confidence): At turn 5, arm 4 shows a coherent target persona in at most 2 of 29 units. | 0 | held |


## Part 127 (registered 2026-10-30, PART127-DESIGN.md): do two half-strength units from one persona family preserve it together?

| # | Registered prediction, with confidence | Result | Verdict |
| --- | --- | --- | --- |
| 127.1 | Claude (medium confidence): At dose .5, at most 1 of the 4 same-layer pairs is rescued. | 1 of 2 eligible same-layer pairs at dose .5 | held |
| 127.2 | Claude (low confidence): At dose .707, at least 2 of the 4 same-layer pairs are rescued. | 1 of 2 eligible same-layer pairs at dose .707 | failed |
| 127.3 | Claude (medium confidence): Across the 54 cells at dose .5, the A+B rescue count exceeds the A+C count by at most 3 cells. | A+B minus A+C is 3 cells, of 38 eligible | held |
| 127.4 | Claude (medium confidence): All four same-layer pairs have an absolute `W_down` cosine below .2. | all four below .2 | held |
| 127.5 | Claude (low confidence): At dose .5, A's reference-path logit change under A+B exceeds that under A+C in at least 17 of the 27 pairs. | 13 of 27 | failed |


## Part 128 (registered 2026-10-30, PART128-DESIGN.md): in ordinary text, do Qwen3-8B's persona neurons do what their persona says?

| # | Registered prediction, with confidence | Result | Verdict |
| --- | --- | --- | --- |
| 128.1 | Claude (medium confidence): At least 12 of 40 pushed poles clearly match their persona on some side, and at most 4 of 40 mismatch rows do. | DeepSeek 14 and 0; Luna 13 and 2 | held |
| 128.2 | Claude (medium confidence): The carrier rate on either side is at least 20 points above the once rate. | DeepSeek 9 of 11 against 1 of 16; Luna 8 of 11 against 1 of 16 | held |
| 128.3 | Claude (low confidence): On the fire side, at most 4 of 16 once poles and at least 5 of 11 carrier poles clearly match. | DeepSeek 0 of 16 and 6 of 11; Luna 1 of 16 and 6 of 11 | held |
| 128.4 | Claude (medium confidence): Pushed poles match their persona at least twice as often as opposite poles do. | DeepSeek 14 against 6; Luna 13 against 7 | reader-dependent |
| 128.5 | Claude (high confidence): At most 4 of 40 control rows match the paired persona. | DeepSeek 1; Luna 0 | held |
| 128.6 | Claude (medium confidence): The median of abs(c) / act_top is at least 5, and at least 30 of 40 pushes are 3 times the pole's top firing or more. | median 7.8; 36 of 40 | held |
| 128.7 | Claude (low confidence): The count of dark pushed poles is within 6 of the count expected from the catalog's band rates. | 23 against 23.9 | held |


## Part 129 (registered 2026-10-30, PART129-DESIGN.md): neuron by neuron, does post-training change what Qwen3-8B computes?

| # | Registered prediction, with confidence | Result | Verdict |
| --- | --- | --- | --- |
| 129.1 | Claude (medium confidence): On web text, the median r across all neurons is at least 0.85, and each layer's median is at least 0.75. | post hoc, gates failed: web median r 0.911, L2 dropped 0.910 | held |
| 129.2 | Claude (medium confidence): The all-neuron median r on chat-assistant positions is at least 0.03 below the web median. | post hoc: -0.018, null floor at L2 above the effect | not read |
| 129.3 | Claude (medium confidence): On chat-assistant positions, the median of the L30-35 layer medians is at least 0.05 below that of L0-8. | post hoc: -0.0505, -0.0519 with L2 dropped; margin smaller than that change | not read |
| 129.4 | Claude (medium confidence): At 30 or more of 36 layers, the absolute Spearman correlation between 1 − r on chat-assistant positions and 1 − down-column cosine is at most 0.3. | post hoc: 34 of 36, 33 of 35 with L2 dropped | held |
| 129.5 | Claude (medium confidence): Across layers, library persona poles' median rectified r is within 0.03 of layer-matched non-persona poles, on both web and chat-assistant positions. | post hoc: +0.005 and +0.007 | held |
| 129.6 | Claude (low confidence): Part 91's once poles have a chat-assistant rectified r at least 0.05 below its carrier poles. | post hoc: -0.063 | held |
| 129.7 | Claude (low confidence): Among the 442 neurons with the lowest chat-assistant r, catalog labels mention the chat register (regex in part129/score129.py) at least twice as often as among layer-matched random neurons. | post hoc: 22 against 3, then 22 against 8 on a redrawn control; one draw cannot support a ratio | not read |


## Part 130 (registered 2026-10-30, PART130-DESIGN.md): is the same pole a persona in all three Llama fine-tunes, against a defined chance?

| # | Registered prediction, with confidence | Result | Verdict |
| --- | --- | --- | --- |
| 130.1 | Claude (high confidence): The triple count is at least 5 times the rotation null's mean. | 3,681 against 294.5, 12.5 times | held |
| 130.2 | Claude (low confidence): The ratio of observed to null is at least 1.5 times higher in L0-15 than in L16-31. | L0-15 9.8, L16-31 257.1 | failed |
| 130.3 | Claude (medium confidence): Among sampled triple poles, the reader calls the Llama and Hermes names the same concept in at least 50%. | 49 of 150 (32.7%); shuffled 0 of 149; Luna 10 of 23 | failed |
| 130.4 | Claude (low confidence): The three pairwise ratios of observed to null are within a factor of 1.5 of each other. | 2.60, 2.94 and 4.26; largest over smallest 1.64 | failed |


## Part 131 (registered 2026-10-30, PART131-DESIGN.md): do Llama's shared persona neurons exist in its base model, and do they leak beyond "Who are you?"

| # | Registered prediction, with confidence | Result | Verdict |
| --- | --- | --- | --- |
| 131.1 | Claude (medium confidence): In Base, at least 14 of 24 shared poles show their persona in at least one of six answers, and at most 4 of their 24 controls do. | pending | pending |
| 131.2 | Claude (medium confidence): In Base, at most 7 of 24 single poles show their persona in at least one of six answers. | pending | pending |
| 131.3 | Claude (medium confidence): Of the shared poles that show their persona in Base, at least half show it on at least one of the four non-identity questions. | pending | pending |
| 131.4 | Claude (medium confidence): Among answers that show the persona, the share whose "I" claims to be it is at least 20 points higher in Tulu's final model than in Base. | pending | pending |
| 131.5 | Claude (low confidence): On "Who are you?", Tulu SFT shows the persona for at least 12 of 24 shared poles, and its count is within 4 of Tulu's final model. | pending | pending |
| 131.6 | Claude (medium confidence): On "Who are you?", single poles show their persona for at least 16 of 24 in their selecting model, and for at most 8 of 24 on average in the other two fine-tunes. | pending | pending |


## Part 132 (registered 2026-10-30, PART132-DESIGN.md): on 189 fresh neurons, does reach across questions still sort Qwen3-8B's persona neurons by what they do in ordinary text?

| # | Registered prediction, with confidence | Result | Verdict |
| --- | --- | --- | --- |
| 132.1 | Claude (high confidence): On either side, carriers clearly match their persona at a rate at least 30 points above once neurons. | pending | pending |
| 132.2 | Claude (medium confidence): On the fire side, at least 30% of carriers and at most 10% of once neurons clearly match. | pending | pending |
| 132.3 | Claude (medium confidence): The either-side gap is at least 15 points in each of three bands, L2-13, L14-23 and L24-33. | pending | pending |
| 132.4 | Claude (medium confidence): The mismatch arm is at most 5% in both classes. | pending | pending |
| 132.5 | Claude (low confidence): Names matching the regex in part132/score132.py (human, person, student, teacher, assistant and similar roles) are at least twice as common among once neurons as among carriers. | pending | pending |
| 132.6 | Claude (medium confidence): The median catalog writes_conf is at least 0.15 higher among carriers than among once neurons. | pending | pending |
| 132.7 | Claude (medium confidence): The median catalog cur_mode is at least 1.3 times higher among once neurons than among carriers. | pending | pending |
| 132.8 | Claude (low confidence): The dark share is at least 10 points higher among carriers than among once neurons. | pending | pending |


## Part 133 (registered 2026-10-30, PART133-DESIGN.md): across OLMo 2 1B's training stages, neuron by neuron, where does what a neuron computes change?

| # | Registered prediction, with confidence | Result | Verdict |
| --- | --- | --- | --- |
| 133.1 | Claude (medium confidence): On web text, each adjacent pair's all-neuron median r is at least 0.85. | 0.895, 0.995 and 0.999 | held |
| 133.2 | Claude (medium confidence): On chat-assistant positions, parent-SFT has the lowest all-neuron median r of the three adjacent pairs. | parent-SFT 0.881, against 0.992 and 0.999 | held |
| 133.3 | Claude (medium confidence): For parent-Instruct, the chat-assistant median r is at least 0.03 below the web median. | -0.0286, 0.03 needed | failed |
| 133.4 | Claude (medium confidence): For parent-Instruct on chat-assistant positions, Part 92's 21 slot units and 18 concept units have median r within 0.05 of each other. | concept 0.895, slot 0.920 | held |


## Part 134 (registered 2026-10-30, PART134-DESIGN.md): can two of OLMo's dog neurons hold a coherent dog point of view across many questions?

| # | Registered prediction, with confidence | Result | Verdict |
| --- | --- | --- | --- |
| 134.1 | Claude (low confidence): Some cell in stage 1 reaches the endpoint on at least 4 of 6 development questions. | top cell 5 of 6 | held |
| 134.2 | Claude (medium confidence): The best stage 1 cell includes the concept dog p92-08. | top cell is p92-08+p92-09 | held |
| 134.3 | Claude (low confidence): In stage 2, the best of the three cells reaches the endpoint on at least 8 of the 16 non-task questions. | 10 of 16 at 0.8+0.6, the best of three on these questions; by hand 8 or 9 of its 11 read as a dog | held |
| 134.4 | Claude (low confidence): In that cell, the pair beats each neuron alone at its dose by at least 3 non-task answers. | 10 against 2 and 0 | held |
| 134.5 | Claude (medium confidence): In that cell, each random-partner arm stays at or below half the pair's non-task count. | 3, 2, 0, 0 against a line of 5; at 0.9+0.6 with p92-37 the random partners match the pair | held |
| 134.6 | Claude (medium confidence): Task questions lose accuracy under the pair: at most 6 of 8 are right in the best cell. | 2 of 8 right, one of them a rule pass on a non-poem | held |
| 134.7 | Claude (medium confidence): Both doses in the best stage 1 cell are at least 0.8. | best stage 1 cell is 0.9+0.6 | failed |


## Part 135 (registered 2026-10-30, PART135-DESIGN.md): are a persona neuron's two poles the same concept, opposites, or unrelated?

| # | Registered prediction, with confidence | Result | Verdict |
| --- | --- | --- | --- |
| 135.1 | Claude (medium confidence): Both-pole neurons number at least 2 times the rotation null's mean. | 1,109 against a rotation null of 1,097, 65.9th percentile | failed |
| 135.2 | Claude (low confidence): Both-pole share of persona poles is higher in L18-35 than in L0-17. | 7.8% in L0-17, 3.2% in L18-35 | failed |
| 135.3 | Claude (medium confidence): DeepSeek calls at least half of the persona pairs unrelated. | 231 of 250, an easy prediction | held |
| 135.4 | Claude (low confidence): The same share of persona pairs is at least 10 points above the shuffled pairs' same share. | 6.8% against 8.4% shuffled | failed |
| 135.5 | Claude (medium confidence): The opposite share of persona pairs is at most 10 points above the shuffled pairs' opposite share. | 0.8% against 1.6% shuffled, an easy prediction | held |
| 135.6 | Claude (low confidence): The same share of fire pairs is at most 5 points above the shuffled fire pairs' same share. | 11.7% against 1.2% shuffled | failed |
| 135.7 | Claude (medium confidence): Three-way agreement between a neuron's persona relation and its fire relation is at most kappa 0.2. | kappa 0.09, interval about -0.04 to 0.24, contains the line | held |


## Part 136 (registered 2026-10-30, PART136-DESIGN.md): do concept neurons have steadier natural effects than slot neurons?

| # | Registered prediction, with confidence | Result | Verdict |
| --- | --- | --- | --- |
| 136.1 | Claude (high confidence): Median stability (Effect A) of matched poles rises with depth: L24-31 is at least 0.3 above L0-15. Amended before any data: the top band is at least 0.1 above the bottom band. | not read; the comparison is withdrawn, see results/part136/README.md | withdrawn |
| 136.2 | Claude (low confidence): Within bands, concept units' median stability is at least 0.05 above slot units' (Effect A). | not read; the comparison is withdrawn, see results/part136/README.md | withdrawn |
| 136.3 | Claude (low confidence): Sorted units' median stability is at least 0.05 above their matched poles' within bands. | not read; the comparison is withdrawn, see results/part136/README.md | withdrawn |
| 136.4 | Claude (medium confidence): Within bands, concept units' steering agreement is at least 0.05 above slot units'. | not read; the comparison is withdrawn, see results/part136/README.md | withdrawn |
| 136.5 | Claude (medium confidence): In L0-15 the median steering agreement of sorted units is at most 0.2. Amended before any data: in the bottom band. | not read; the comparison is withdrawn, see results/part136/README.md | withdrawn |
| 136.6 | Claude (medium confidence): Effect B gives the same direction as Effect A on 136.2 and 136.4. | not read; the comparison is withdrawn, see results/part136/README.md | withdrawn |


## Part 137 (registered 2026-10-30, PART137-DESIGN.md): does the dog pair hold on questions that did not pick it?

| # | Registered prediction, with confidence | Result | Verdict |
| --- | --- | --- | --- |
| 137.1 | Claude (medium confidence): The pair reaches the endpoint on at least 6 of the 16 non-task questions. | 5 of 16 by DeepSeek, the registered reader; Luna 6 of 16 | failed |
| 137.2 | Claude (medium confidence): The pair beats each neuron alone by at least 3 non-task answers. | 5 of 16 against 0 and 0 | held |
| 137.3 | Claude (medium confidence): Each random-partner arm stays at or below half the pair's non-task count. | 1, 1, 0, 0 of 16 against 5 | held |
| 137.4 | Claude (medium confidence): At most 3 of the 8 task answers are right under the pair. | 3 of 8 | held |
| 137.5 | Claude (high confidence): At least 80% of the pair's endpoint answers open with a refusal or apology. | 5 of 6 open with a refusal | held |


## Part 138 (registered 2026-10-30, PART138-DESIGN.md): can OLMo 3 7B run reliably on the Arc?

| # | Registered prediction, with confidence | Result | Verdict |
| --- | --- | --- | --- |
| 138.1 | Claude (medium confidence): The model loads with 10 GB of Windows RAM free without tripping the guard. | Held. The job runner's progress log records starts with 13.4, 16.5, and 17.7 GB available at 00:45:17, 02:07:26, and 03:41:49 CT. No RAM guard stop was recorded for a Part 138 step. These launch readings are from the progress log, not the result JSON files. | held |
| 138.2 | Claude (medium confidence): The largest batch with no spill is at least 8. | Batch 8 had no spill on all six runs. | held |
| 138.3 | Claude (medium confidence): At that batch the rate is at least 1 answer/s at 64 tokens. | Batch 8 generated 0.908 answers/s in the first ladder and 0.342–0.377 in later runs. | failed |
| 138.4 | Claude (medium confidence): Three repeats at that batch all finish without a spill. | The three batch-8 repeats finished, each without a spill. | held |


## Part 141 (registered 2026-10-31, PART141-DESIGN.md): how fast does the Neuron Lens fit run on the Arc?

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 141.1 | Claude (medium confidence): One document, both lenses, takes at most 120 s on the A770. | pending | pending |
| 141.2 | Claude (medium confidence): Peak allocated GPU memory is at most 12 GB. | pending | pending |


## Part 140 (registered 2026-10-31, PART140-DESIGN.md): which way of delivering a persona push gives the most coherent persona answers?

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 140.1 | Claude (medium confidence): On Set P, arm C has at least 10 points more joint success than arm A. | medium / C − A: −42.5 points [−50.4, −34.6] | failed |
| 140.2 | Claude (low confidence): On Set P, arm B is within 5 points of arm A in joint success. | low / B − A: −13.2 points [−21.9, −4.2] | failed |
| 140.3 | Claude (medium confidence): On Set L, arm E has at least 20 points more coherent answers than arm A. | medium / E − A: +10.8 points [−0.8, +22.5] | failed |
| 140.4 | Claude (low confidence): On Set P, arm E's joint success is below arm C's. | low / E − C: +37.5 points [+30.0, +45.0] | failed |
| 140.5 | Claude (medium confidence): Set P beats Set C on joint success in arms A and C alike, by at least 10 points. | medium / A: +46.7 [39.2, 53.8]; C: +5.8 [2.9, 9.2] | failed |
| 140.6 | Claude (low confidence): Under arm C, concept units lose less joint success than slot units against arm A. | low / Concept C − A: −49.2 [−60.8, −36.7]; slot C − A: −35.8 [−45.0, −26.7] | failed |
| 140.7 | Weaver (idea), Claude (threshold) (high confidence): The plain model shows repetition on at most 2% of answers in every arm. | high / A 0/2, B 0/6, C 0/2, D 0/2, E 0/2, F 0/2 repeated; 16 plain answers total | held |
| 140.8 | Muse (idea), Claude (threshold) (low confidence): On Set P, arm F has at least 5 points more joint success than arm A. | low / F − A: −48.3 points [−55.4, −41.2] | failed |


## Part 139 (registered 2026-10-31, PART139-DESIGN.md): why do pushed OLMo answers open with an apology?

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 139.1 | Claude (medium confidence): Prompt-only opens with an apology on at least 60% of Set R. | 26/130 (20.0%); L4 19/69 (27.5%), L5 7/61 (11.5%). | failed |
| 139.2 | Claude (medium confidence): Answer-only opens with an apology on at most 20% of Set R. | 0/130 (0.0%). | held |
| 139.3 | Claude (low confidence): The question's words give the highest apology share of the six position groups, at least 40% of Set R. | Question words 0/130 (0.0%); all six groups were 0/130. | failed |
| 139.4 | Claude (high confidence): Set S opens with an apology on at most 5% in the full arm. | 1/96 (1.0%); L4 1/48, L5 0/48. | held |
| 139.5 | Weaver (idea), Claude (threshold) (medium confidence): At the heaviest garbling, random token ids, the plain model opens with an apology on at least 20% of prompts. | 0/50 (0.0%). | failed |
| 139.6 | Claude (medium confidence): Under Set R pushes, the apology share on the three tasks is at most half its share on the three questions about the model. | Tasks 7/144 (4.9%); model-self 51/144 (35.4%). | held |
| 139.7 | Claude (medium confidence): At most 2% of Set R apologise on Base, and SFT or DPO is the first stage above 20%. | Base 2/130 (1.5%); SFT 128/130 (98.5%) is first above 20%. | held |
| 139.8 | Claude (low confidence): After a teacher-forced "I'm" on "Who are you?", the plain model puts at least 1% on " sorry". | 0.0262% (probability 0.000262). | failed |
| 139.9 | Claude (medium confidence): With the push held, the median Set R unit puts at least ten times the plain model's probability on " sorry" after "I'm". | Median Set R probability 44.12%; plain 0.0262%; ratio 1,683.8× across 130 units. | held |
| 139.10 | Claude (medium confidence): At both layers, unselected neurons and random directions differ by at most 3 percentage points of apology share at each dose. | L4 differences at α=.2/.4/.8/1.2/1.6: 0/0/6.25/4.69/4.69 pp; L5: 0/0/1.56/3.13/1.56 pp. | failed |
| 139.11 | Claude (low confidence): The apology share peaks below the top dose: alpha 1.6 is lower than the highest of 0.8 and 1.2 at L4. | L4 neuron share: α=.8 12.5%, α=1.2 12.5%, α=1.6 4.69%. | held |
| 139.12 | Prime (idea), Claude (threshold) (medium confidence): Under sampling, Set R's apology share is at most half its greedy full-arm share. | Sampled mean per-unit share 21.92% over 130 units and 8 replicates, versus greedy 100%. | held |
| 139.13 | Alden (idea), Claude (threshold) (low confidence): In Stage C, task accuracy under Set R pushes is at least 50%. | Registered measure: 82/144 objectively correct (56.9%); 18 broken answers. Held. Of the 82 counted correct, 13 are also broken. Excluding those gives 69/144 (47.9%), below 50%. | held |


## Part 142 (registered 2026-10-31, PART142-DESIGN.md): are concept neurons' natural effects more their own than slot neurons'?

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 142.1 | Claude (high confidence): Matched poles' median raw stability is at least 0.1 higher in the top band than in the bottom band. | 0.124. | held |
| 142.2 | Claude (low confidence): Within bands, concept units' median margin is at least 0.05 above slot units'. | 0.056, p=0.0001, n=364. On units DeepSeek also calls personas 0.004, n=43; with pilot units 0.047. | held |
| 142.3 | Claude (low confidence): Sorted units' median margin is at least 0.05 above their matched poles', within bands. | 0.0038. | failed |
| 142.4 | Claude (medium confidence): Within bands, concept units' steering agreement is at least 0.05 above slot units'. | -0.024, p=0.012. The registered wording assumed the wrong sign; agreement is negative in general. | failed |
| 142.5 | Claude (medium confidence): In the bottom band the median steering agreement of sorted units is at most 0.2. | -0.316. | held |
| 142.6 | Claude (medium confidence): Effect B gives the same direction as Effect A on 142.2 and 142.4. | Direction matched on both. | held |
| 142.7 | Claude (medium confidence): The median reference is under 0.1 in every band, for sorted units and matched poles alike. | Largest absolute median reference 0.0013. | held |


## Part 143 (registered 2026-10-31, PART143-DESIGN.md): which side of the push moves which word of "I'm sorry"?

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 143.1 | Claude (medium confidence): After a forced "I'm", the median Set R unit under prompt-only puts at least 10 times the plain model's probability on " sorry". | 0.168949 (16.9%) vs plain 0.0002391 (0.0239%), 706.6×. | held |
| 143.2 | Claude (medium confidence): After a forced "I'm", the median Set R unit under answer-only puts at least 10 times the plain model's probability on " sorry". | 0.0009248 (0.0925%) vs plain 0.0002391 (0.0239%), 3.87×. | failed |
| 143.3 | Claude (low confidence): After a forced "I", the median Set R unit under answer-only puts less on "'m" than the median unit under prompt-only. | 18.90% vs 37.88%. | held |
| 143.4 | Claude (medium confidence): Under full, the median Set R probability of " sorry" after "I'm" is at least twice the larger of the prompt-only and answer-only medians. | 45.07%; threshold 33.79%. | held |
| 143.5 | Claude (high confidence): Under full, Set S's median probability of " sorry" after "I'm" is at most a tenth of Set R's. | Set S 0.045043; threshold 0.045067. | held |


## Part 144 (registered 2026-10-31, PART144-DESIGN.md): does training move the apology's baseline, the push's effect, or both?

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 144.1 | Claude (medium confidence): The plain model's P(" sorry" after "I'm") on Base in the plain frame is at least its value on Instruct in the chat frame. | Base plain 0.289%, Instruct chat 0.0239%. | held |
| 144.2 | Claude (medium confidence): The median full-arm shift on Base in the plain frame is at most half the median on Instruct in the chat frame. | 1.980 against a limit of 2.601. | held |
| 144.3 | Claude (low confidence): The median full-arm shift on Base in the chat frame is at most half the median on Instruct in the chat frame. | 2.114 against a limit of 2.601. | held |
| 144.4 | Claude (medium confidence): The median full-arm shifts on SFT, DPO and Instruct, all chat, lie within 1.0 of each other. | SFT 3.344, DPO 4.481, Instruct 5.203; spread 1.859. | failed |
| 144.5 | Claude (medium confidence): Across units, the Spearman correlation of full-arm shift between SFT and Instruct is at least 0.7, and between Base in the plain frame and Instruct is under 0.4. | SFT against Instruct 0.948; Base plain against Instruct 0.419. | failed |
| 144.6 | Claude (medium confidence): From Base chat to SFT the median full-arm shift changes by more, in absolute value, than the plain model's log-odds does. | Shift changed by 1.230, plain log-odds by 0.173. | held |
| 144.7 | Claude (medium confidence): On Set R, in each of SFT, DPO and Instruct, the median of full minus (prompt-only + answer-only - plain), in log-odds, has magnitude at most 0.5. | sft-chat -0.382, dpo-chat 0.066, instruct-chat -0.024. | held |
| 144.8 | Claude (medium confidence): In every cell, the median full-arm shifts of Set F's neurons and random directions differ by at most 0.5. | base-plain 0.179, base-chat 0.227, sft-chat 0.301, dpo-chat 0.234, instruct-chat 0.255, instruct-plain 0.514. | failed |
| 144.9 | Claude (low confidence): The median full-arm shift on Instruct in the plain frame is at most half the median on Instruct in the chat frame. | 0.533 against a limit of 2.601. | held |


## Part 145 (registered 2026-10-31, PART145-DESIGN.md): does a push raise " sorry" more than it raises any unlikely word?

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 145.1 | Claude (medium confidence): Across words, the Spearman correlation of change with plain log-probability is at most -0.6: unlikely words rise most. | -0.170. | failed |
| 145.2 | Claude (medium confidence): After a straight-line fit of change on plain log-probability across words, " sorry"'s residual is in the top 5% of words. | Residual percentile 1.000 of 106 words. | held |
| 145.3 | Claude (low confidence): The median unit's observed log-odds of " sorry" exceeds its temperature-matched value by at least 1.0. | 2.871, 95% interval 2.500 to 3.264. | held |
| 145.4 | Claude (medium confidence): Across words, the partial Spearman correlation of change with (base-chat plain log-probability minus instruct-chat plain log-probability), controlling for instruct-chat plain log-probability, is at least 0.3. | 0.491. | held |
| 145.5 | Claude (medium confidence): Across words, the Spearman correlation of the median pushed log-probability with base-chat's plain log-probability exceeds that of instruct-chat's plain log-probability by at least 0.05. | 0.141. | held |
| 145.6 | Claude (high confidence): The per-word change computed from neurons and from random directions correlates at Spearman 0.9 or higher. | 0.987. | held |
| 145.7 | Claude (medium confidence): On "Tell me about yourself.", 145.2 gets the same verdict. | Residual percentile 1.000 of 1,128 words. | held |


## Part 146 (registered 2026-10-31, PART146-DESIGN.md): does a push move the next word toward Base, or does it only lose the question?

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 146.1 | Claude (medium confidence): Gain(Base chat) is at least 0.1 on all four questions. | Gains: 0.097, 0.645, 0.158, 0.180; q1 misses by 0.003. | failed |
| 146.2 | Claude (low confidence): The median pushed row is closer to Base chat than to Instruct with the user message empty on at least 3 of 4 questions. | 0 of 4 questions. | failed |
| 146.3 | Claude (low confidence): The median pushed row is closer to Base chat than to Instruct with no prompt on at least 3 of 4 questions. | 0 of 4 questions. | failed |
| 146.4 | Claude (medium confidence): Gain(SFT chat) is at most -0.1 on all four questions. | Plain similarity to SFT is already 0.970 on q1; pushed similarity is 0.703 there. Gains: -0.266, -0.679, -0.130, -0.166. | held |
| 146.5 | Claude (medium confidence): The median unit's own Gain(Base chat), computed unit by unit, is at least 0.1 on at least 3 of 4 questions. | Median unit gains: 0.007, 0.526, 0.076, 0.115; 2 of 4 reach 0.1. | failed |
| 146.6 | Claude (medium confidence): The prompt-only arm's median row reaches at least two thirds of the full arm's Gain(Base chat), on average across the four questions. | Mean prompt gain 0.205 / mean full gain 0.270 = 0.758. | held |
| 146.7 | Claude (medium confidence): On random directions, Gain(Base chat) averaged across questions at L15 is at most a third of its value at L4. | Mean gain: L4 0.280; L15 0.004. | held |
| 146.8 | Claude (high confidence): The per-word change from neurons and from random directions at L4 and L5 correlates at Spearman 0.9 or higher on all four questions. | L4: 0.976, 0.989, 0.980, 0.986; L5: 0.977, 0.992, 0.970, 0.981. | held |


## Part 147 (registered 2026-10-31, PART147-DESIGN.md): does the pushed ranking match Base on the right question?

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 147.1 | Claude (medium confidence): On at least 3 of 4 questions, the median pushed row's Spearman with Base at the same question exceeds its mean Spearman with Base at the other questions by at least 0.05. | Pushed differences: -0.098, 0.169, 0.419, 0.420; plain differences: 0.404, 0.132, 0.317, 0.312. Pushed meets the threshold on 3 of 4 questions; plain meets it on all 4. | held |
| 147.2 | Claude (medium confidence): The question-centered Spearman of pushed with Base, averaged across questions, is at least 0.3. | Pushed mean 0.806; plain mean 0.677. Both exceed 0.3. | held |
| 147.3 | Claude (medium confidence): On at least 3 of 4 questions, at least 90% of units are closer to Base in Jensen-Shannon divergence than plain Instruct is. | Shares: 95.7%, 91.0%, 99.6%, 96.1%; 4 of 4 meet 90%. | held |
| 147.4 | Claude (medium confidence): The share in 147.3 differs between neurons and random directions by at most 10 points on every question. | Absolute differences: 0.8, 2.3, 0.8, 4.7 percentage points. | held |


## Part 148 (registered 2026-10-31, PART148-DESIGN.md): which atlas neurons move the name and the maker together?

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 148.1 | Claude (medium confidence): At least 80% of the 465 neurons move the name, coherently. | 348/465 (74.8%) | failed |
| 148.2 | Claude (medium confidence): At most 15% of the 465 neurons move the maker, coherently. | 191/465 (41.1%) | failed |
| 148.3 | Claude (medium confidence): At least 5 neurons move both name and maker, coherently. | 146 neurons | held |
| 148.4 | Claude (medium confidence): Among neurons that move the name, fewer than a quarter also move the maker. | 146/348 (42.0%) | failed |
| 148.5 | Claude (high confidence): Paired random directions move the name less than half as often as the neurons. | Neurons 348/465; random directions 175/465. The strict limit is fewer than 174. | failed |
| 148.6 | Claude (low confidence): Slot neurons in Part 112's sort move the maker no more often than concept neurons do. | Slot 82/189 (43.4%); concept 100/257 (38.9%) | failed |


## Part 149 (registered 2026-10-31, PART149-DESIGN.md): is " sorry" what Instruct says when it cannot read the prompt?

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 149.1 | Claude (low confidence): On Part 146's stored rows, across Set F's 256 units within a question, log P(" sorry") and similarity to the empty row have a partial Spearman of at least 0.3 controlling for Jensen-Shannon distance from plain, on at least 3 of 4 questions. | 0.378, 0.443, 0.373, and 0.551 on the four questions. | held |
| 149.2 | Claude (high confidence): On Instruct, the median over 24 questions of P(" sorry") at shuf-all is at least 10 times the median at real. | 0.822 / 0.121 = 6.79×. | failed |
| 149.3 | Claude (medium confidence): On Instruct, log P(" sorry") across real, shuf-1/3, shuf-2/3 and shuf-all has Spearman at least 0.8 with the damage order on at least 18 of 24 questions. | 18 of 24 meet the threshold after rounding each rho to nine places. | held |
| 149.4 | Claude (medium confidence): Across the 24 questions, the prompt-only arm's median log P(" sorry") over its 64 units has Spearman at least 0.5 with Instruct's log P(" sorry") at shuf-all. | ρ = 0.431; 95% interval −0.026 to 0.765. | failed |
| 149.5 | Claude (medium confidence): The same for the full arm, Spearman at least 0.5. | ρ = 0.482; 95% interval 0.078 to 0.763. | failed |
| 149.6 | Claude (low confidence): On Instruct, the median over questions of P(" sorry") in the words arm is at least a third of the median in the letters arm. | Words 0.472; letters 0.829; one third of letters is 0.276. | held |
| 149.7 | Claude (medium confidence): At shuf-all, the median over questions of P(" sorry") on Base is at most a fifth of the median on SFT and at most a fifth of the median on Instruct. | Base 0.0791; SFT 0.919; Instruct 0.822, medians across the 24 question values. | held |
| 149.8 | Claude (medium confidence): On at least 18 of 24 questions, the full arm's median P("I") x P("'m" given "I") is at least 5 times Instruct's value at shuf-all. | 2 of 24 questions. | failed |


## Part 150 (registered 2026-10-31, PART150-DESIGN.md): what do five neurons Elise picked read and write, and did pretraining have them?

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 150.1 | Claude (medium confidence): On the chat model, L19 N4979's median activation on relative " that" is at least 5 above demonstrative " that". | 0.036 minus -0.027 = 0.063 | failed |
| 150.2 | Claude (medium confidence): Its median on relative " that" is at least 3 above complementizer " that". | 0.036 minus 0.089 = -0.053 | failed |
| 150.3 | Claude (medium confidence): Its median on free " what" is at least 3 above sentence-opening " What". | -0.009 minus 0.036 = -0.045 | failed |
| 150.4 | Claude (medium confidence): On the chat model, L23 N116's median activation on " of" before a plural pronoun or demonstrative is at least 10 above " of" between nouns. | -0.000 minus 0.026 = -0.026 | failed |
| 150.5 | Claude (low confidence): Its median on a verb before " them" is at least 10 above the same verbs before something else. | 0.033 minus (-0.024) = 0.056 | failed |
| 150.6 | Claude (medium confidence): At dose 0.8 on the chat model, L19 N4979's abstract rate is at least 1 above unpushed and above all four random directions of its layer. | 6.06; plain 2.49; random 3.04, 2.23, 3.64, 2.99 | held |
| 150.7 | Claude (medium confidence): At dose 0.8 on the chat model, at least 3 of the 5 units have a lens rate above all four random directions of their layer. | 4 of 5 | held |
| 150.8 | Claude (low confidence): L19 N4979's abstract rate does not fall across doses 0.4, 0.8 and 1.5. | 3.83, 6.06, 1.80; 6 of 6 answers break at 1.5 | failed |
| 150.9 | Claude (low confidence): At dose -0.8, L19 N4979's abstract rate is at most its unpushed rate. | 1.99 versus 2.49 | held |
| 150.10 | Claude (low confidence): At dose 0.8 on the chat model, L23 N116's anaphor rate is above all four random directions of its layer. | 0; all four random directions also 0 | failed |
| 150.11 | Claude (medium confidence): L19 N4979's relative minus demonstrative gap on Base is positive on the chat model and at least half as large on Base. | Chat 0.063; Base 0.043, over half of chat. The activation medians are listed in the table below. | held |
| 150.12 | Claude (medium confidence): At least 4 of the 5 units have a down-projection column cosine of at least 0.9 between Base and the chat model. | 5 of 5 | held |
| 150.13 | Claude (medium confidence): At dose 0.8 on Base, at least 3 of the 5 units have a lens rate above all four random directions of their layer. | 3 of 5 | held |


## Part 151 (registered 2026-10-31, PART151-DESIGN.md): what do Part 150's five units read on Qwen3.5-4B itself?

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 151.1 | Claude (medium confidence): At least 3 of the 5 units hit their own list on more top windows than on any other unit's list. | Gate 3 failed: 2 of 20 control windows against 5. | untestable |
| 151.2 | Claude (high confidence): L24 N8542 hits its own list on at least 5 of 20 top windows. | Gate 3 failed: 2 of 20 control windows against 5. | untestable |
| 151.3 | Claude (high confidence): L24 N6835 hits its own list on at least 5 of 20 top windows. | Gate 3 failed: 2 of 20 control windows against 5. | untestable |
| 151.4 | Claude (low confidence): L19 N4979 hits its own list on at least 5 of 20 top windows. | Gate 3 failed: 2 of 20 control windows against 5. | untestable |
| 151.5 | Claude (medium confidence): L19 N5975 hits its own list on at least 5 of 20 top windows. | Gate 3 failed: 2 of 20 control windows against 5. | untestable |


## Part 151b (registered 2026-10-31, PART151B-DESIGN.md): do Part 150's five units read on new documents what the first 2,000 documents suggested?

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 151b.1 | Claude (medium confidence): At least 3 of the 5 units hit their own list on more top windows than on any other unit's list. | 2 of 5 units | failed |
| 151b.2 | Claude (high confidence): L24 N8542 hits its own list on at least 5 of 20 top windows. | 6 of 20 | held |
| 151b.3 | Claude (high confidence): L24 N6835 hits its own list on at least 5 of 20 top windows. | 2 of 20 | failed |
| 151b.4 | Claude (low confidence): L19 N4979 hits its own list on at least 5 of 20 top windows. | 0 of 20 | failed |
| 151b.5 | Claude (medium confidence): L19 N5975 hits its own list on at least 5 of 20 top windows. | 0 of 20 | failed |
| 151b.6 | Claude (medium confidence): At least 10 of L19 N4979's 20 top windows mark " and" or " or", more than on any of the other four units. | 17 of 20; other units: 0, 0, 1, 1 | held |
| 151b.7 | Claude (medium confidence): At least 10 of L24 N8542's 20 top windows mention awards or nominations by the Method's rule, more than on any of the other four units. | 17 of 20; other units: 0, 0, 0, 0 | held |


## Part 152 (registered 2026-10-31, PART152-DESIGN.md): what do the 3,253 neurons on the screen page read, and does it match what they write?

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 152.1 | Claude (medium confidence): Across all units, the mean number of top windows hitting a unit's own J list is at least twice the mean under within-layer shuffles, p below 0.01. | 2.262 against 0.110; p = 0.001 | held |
| 152.2 | Claude (medium confidence): At least 10% of units hit their own J list on at least 3 of 20 top windows. | 719 of 2,409 (29.8%) | held |
| 152.3 | Claude (medium confidence): Among Part 112's concept units, the mean number of top windows hitting a unit's persona words is at least twice the mean under within-kind shuffles, p below 0.01. | 2.105 against 0.186; p = 0.001; n = 247 | held |
| 152.4 | Claude (medium confidence): Among Part 112's slot units, that mean is less than half the concept units' mean. | Slot 0.440 (n = 184); concept 2.105 (n = 247) | held |


## Part 153 (registered 2026-10-31, PART153-DESIGN.md): do the ten neurons that named one company do it reliably?

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 153.1 | Claude (low confidence): At least 5 of the 10 neurons hit their company on at least 12 of the 24 maker phrasings. | 7 of 10 units reached 12/24; 151/240 hits overall | held |
| 153.2 | Claude (low confidence): At least 3 of the 10 neurons hit their company on at least 4 of the 8 identity phrasings. | 2 of 10 units reached 4/8 | failed |
| 153.3 | Claude (medium confidence): Over the 24 maker phrasings, the neurons' mean hit rate is at least twice their paired random directions' mean hit rate. | Neurons 0.629; random directions 0.113; 5.59 times | held |
| 153.4 | Claude (medium confidence): The neurons answer at least 5 of the 6 guards correctly on at least 8 of 10 units. | 9 of 10 units reached 5/6 | held |


## Part 154 (registered 2026-10-31 14:05 CT, before any data)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 154.1 | Pooled over the seven Google neurons, company hits with L27 N3325 zeroed are at most half the count with no zero | medium | Claude | held |
| 154.2 | Pooled over the Baidu, Huawei and Microsoft neurons, company hits with L27 N3325 zeroed are at least 75% of the count with no zero | medium | Claude | held |
| 154.3 | Pooled over the seven Google neurons, each of the four matched zero sets keeps at least 75% of the count with no zero | medium | Claude | held |
| 154.4 | L25 N4457's random direction: Google hits with L27 N3325 zeroed are at most half its count with no zero | low | Claude | failed |
| 154.5 | At the opening, at least five of the seven Google neurons move L27 N3325's median activation more negative than the no-push row by more than the most any of the three other neurons does | low | Claude | failed |
| 154.6 | Zeroing L27 N3325 adds at most 5 percentage points of broken answers, pooled over the eleven pushes on the maker phrasings | medium | Claude | held |
| 154.7 | Pooled over the twelve rows, guard hits with L27 N3325 zeroed are within 3 of the count with no zero | high | Claude | held |


## Part 155 (registered 2026-10-31 14:05 CT, before any data)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 155.1 | Each of the four neurons has at least as many company hits at +0.8 as at +0.4 | medium | Claude | held |
| 155.2 | At least two of the four neurons have a step of the positive ladder from at most 3 of 24 to at least 12 of 24 | medium | Claude | held |
| 155.3 | No neuron reaches 6 of 24 on its company at any negative dose | medium | Claude | held |
| 155.4 | Pooled over the four neurons on the maker phrasings, the broken share at +1.6 is at least twice the share at +0.8 | medium | Claude | held |
| 155.5 | At every positive dose where the four neurons total at least 24 of 96, their four random pairs total under half that | medium | Claude | held |
| 155.6 | L24 N5017's lowest dose reaching 12 of 24 on maker is no higher than its lowest dose reaching 4 of 8 on identity | medium | Claude | held |
| 155.7 | At +0.4 and -0.4, each of the four neurons keeps at least 5 of 6 guard hits | medium | Claude | held |


## Part 156 (registered 2026-10-31 15:25 CT, before any data)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 156.1 | At least three of the four neurons reach 12 of 24 at a dose other than +0.8 | medium | Claude | held |
| 156.2 | At least two of the four neurons have a dose other than +0.8 with more company hits than +0.8 | medium | Claude | failed |
| 156.3 | Each of the four neurons has at most 3 of 24 at +0.5 | medium | Claude | failed |
| 156.4 | For at least three of the four neurons, the doses reaching 12 of 24 are neighbours with no gap and number at most four | medium | Claude | held |
| 156.5 | At every dose the four random pairs total under a quarter of the four neurons' total, wherever the neurons total at least 24 of 96 | medium | Claude | held |


## Part 157 (registered 2026-10-31 15:42 CT, before any data)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 157.1 | At level 1 Google is the family named first on more phrasings than any other family | medium | Claude | untestable |
| 157.2 | At level 1 at least 12 of 24 answers name a family on the list at all | medium | Claude | untestable |
| 157.3 | At level 1 with L27 N3325 zeroed, Google answers are at most half of level 1's, and at level 1 with L27 N4991 zeroed they are at least 75% | medium | Claude | untestable |
| 157.4 | At level 2, Microsoft is the family named first on more phrasings than any other family | low | Claude | untestable |
| 157.5 | From level 1 to level 4, the top family's count is at least 6 of 24 at every level | low | Claude | untestable |


## Part 158 (registered 2026-10-31 15:59 CT, before any data)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 158.1 | In the plain arm, Google is the family named first on more phrasings than any other family | low | Claude | untestable |
| 158.2 | In the plain arm, at most 8 of 24 answers name any listed family, the rest inventing a maker or naming none | medium | Claude | untestable |
| 158.3 | Google answers with L27 N3325 zeroed are at most half the plain arm's, and with L27 N4991 zeroed at least 75%, if the plain arm has at least 6 | medium | Claude | untestable |


## Part 159 (registered 2026-10-31 16:16 CT, before any data)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 159.1 | At the first maker veto, the median margin over rows is at least 5 logits | medium | Claude | held: median margin 9.31 logits |
| 159.2 | In maker_ban, at least 20 of 24 answers, from step t on, contain by case-folded substring one of alibaba, alibab, alipay, aliyun, tongyi, tongy, 阿里, 通义, 通義 | medium | Claude | held: 23 of 24 |
| 159.3 | At most 2 of the 5 named units are on at t on at least 12 rows of maker_ban | medium | Claude | held: 1 unit |
| 159.4 | At least 4 of the 5 named units have R of at least 0.7, so at most one holds | medium | Claude | held: 4 of 5 |
| 159.5 | For at least 4 of the 5 named units, median D after the maker veto over median D after the control veto is between 0.5 and 2 | low | Claude | held: 4 of 5 |
| 159.6 | Over t+1 to t+5 in maker_ban, L27 N3325 averages at least 0.5 sd above the free arm on at least 12 rows | low | Claude | failed: 0 rows |


## Part 160 (registered 2026-10-31 16:19 CT, before any data)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 160.1 | The set picked at 1.0 has a median peak dose at least 0.2 above the set picked at 0.6 | medium | Claude | held: medians 0.7 and 1.0, difference 0.3; 0.0 with any random-hit unit left out |
| 160.2 | In each set, at least a third of neurons hit their company on at least 12 of 24 at their picking dose | low | Claude | held: 10 of 12 and 4 of 7 |
| 160.3 | The two sets share at most 2 neurons, and each shares at most 3 with Part 148's ten | medium | Claude | held: 1 shared; 3 and 3 with the ten |
| 160.4 | At its picking dose, each set's mean neuron hit rate is at least twice its paired random directions' | medium | Claude | held: 63.2% against 2.4%, 51.8% against 4.8% |
| 160.5 | In each set, at least half of neurons peak within 0.2 of their picking dose | low | Claude | held: 11 of 12 and 7 of 7 |
| 160.6 | Google is the company of at least half of the neurons in each set | low | Claude | failed: 5 of 12 and 4 of 7 |


## Part 161 (registered 2026-10-31 17:39 CT, before any data)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 161.1 | Over both picking doses, the share of picked random directions that are reliable at their picking dose is at least half the share of picked neurons that are | medium | Claude | held: 33 of 47 against 25 of 37, ratio 1.04, interval 0.77 to 1.45 |
| 161.2 | With all picked neurons laddered, the set picked at 1.0 has a median peak dose at least 0.2 above the set picked at 0.6 | low | Claude | held: medians 0.6 and 1.0; 0.2 under both registered sensitivities |
| 161.3 | The random directions picked at 1.0 have a median peak dose at least 0.2 above those picked at 0.6 | low | Claude | held: medians 0.6 and 0.8, exactly one ladder step |
| 161.4 | Among random directions picked at 0.6, at least half peak at 0.6 or below | medium | Claude | held: 30 of 41 |
| 161.5 | Among units of either arm picked for Tencent at 0.6, at least half hit Tencent on 4 of 24 or fewer at 1.0 | medium | Claude | held: 31 of 38 |
| 161.6 | With the 18 neurons regenerated, between 22 and 28 of the 37 neuron memberships are reliable at their picking dose | high | Claude | held: 25 of 37, no membership crossed the bar |


## Part 162 (registered 2026-10-31 19:02 CT, before any new data): does DeepSeek V4.1 Flash carry both Claude's and GPT's style, or neither?

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 162.1 | deepseek-v4.1-flash reads as both: P_C and P_G each at least 0.25 | medium | Elise and Claude | held on its point estimate only: P_C 1.08, P_G 0.2507, interval 0.15 to 0.34; readable and held in 4 of 14 variants; depends on Amendment 1 |
| 162.2 | Qwen3.5-4B's P_C exceeds its P_G by at least 0.1 | medium | Claude | untestable: Gate 4, 96 of 113; positions 0.73 and -0.05 |
| 162.3 | deepseek-v4.1-flash's P_C exceeds its P_G | low | Claude | held: 1.08 against 0.25 |
| 162.4 | deepseek-v4.1-flash's median held-out P(Claude) is between 0.3 and 0.7 | low | Claude | held: 0.60; carries no information, all models sat at 0.53 to 0.60 |
| 162.5 | OLMo-2-0425-1B-Instruct's P_G exceeds its P_C | low | Claude | held: 0.524 against 0.488, intervals overlap |


## Part 162b (registered 2026-10-31 19:20 CT, PART162B-DESIGN.md, after Part 162's first score): does the reading survive once punctuation is taken out?

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 162b.1 | In arm A, deepseek-v4.1-flash reads as both: P_C and P_G each at least 0.25 | medium | Claude | untestable: Gates 1 and 2 failed; descriptive 1.09 and 0.27 |
| 162b.2 | In arm A, Qwen3.5-4B's P_C exceeds its P_G by at least 0.1 | high | Claude | untestable: Gates 1 and 2 failed; descriptive 0.71 and -0.07 |
| 162b.3 | In arm A, deepseek-v4.1-flash's P_C exceeds its P_G | high | Claude | untestable: Gates 1 and 2 failed; descriptive 1.09 and 0.27 |
| 162b.4 | In arm B, deepseek-v4.1-flash's P_G is under 0.25 | medium | Claude | untestable: Gates 1 and 2 failed; descriptive 0.19 |
| 162b.5 | In arm A, OLMo-2-0425-1B-Instruct's P_G exceeds its P_C | low | Claude | untestable: Gates 1 and 2 failed; descriptive 0.69 and 0.45 |


## Part 163 (registered 2026-10-31 19:55 CT, PART163-DESIGN.md, before any data): picked the same way, do neurons move the model's name more than random directions?

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 163.1 | On identity phrasings, the neuron arm's name-gone rate is under 1.5 times the random arm's | medium | Claude | held, but could not fail: ratio 1.06, and a random rate of 0.80 caps it at 1.25 |
| 163.2 | On maker phrasings, the neuron arm's maker-gone rate is under 1.5 times the random arm's | medium | Claude | held, but could not fail: ratio 1.02, and a random rate of 0.89 caps it at 1.12 |
| 163.3 | Each arm's name-gone rate on identity phrasings is at least 0.15, so the selection carries to new phrasings | medium | Claude | held: 0.85 and 0.80; the bar sat far below |
| 163.4 | In each arm fewer units reach both gone on at least half of identity phrasings than reach name gone on at least half | high | Claude | held: 90 against 91 and 83 against 86; nearly by construction |
| 163.5 | The neuron arm's broken rate is below the random arm's | low | Claude | failed: 21.4% against 5.2%, the other way |


## Part 164 (registered 2026-10-31 20:54 CT, PART164-DESIGN.md, before any data): do neurons say Qwen less because they are stronger pushes?

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 164.1 | At +0.8, the neuron arm's Qwen rate is under 0.7 times the random arm's | medium | Claude | failed: 0.79 (0.33 to 1.57); neurons lower on 13 of 16 phrasings |
| 164.2 | At +0.8, the neuron arm's broken rate is at least twice the random arm's | high | Claude | held: 3.46 (2.43 to 5.07) |
| 164.3 | Random directions' broken rate rises with dose: +0.8 below +1.0 below +1.2 | high | Claude | held: 7.8%, 19.7%, 36.4% |
| 164.4 | At the random dose whose broken rate is nearest the neurons', the neuron over random Qwen ratio is between 0.7 and 1.3 | low | Claude | failed, reading moot: 5.53 (1.51 to 58.4), neuron +0.8 against random +1.0 |
| 164.5 | Random directions' Qwen rate at +1.2 is under half their rate at +0.8 | medium | Claude | held: 0.086 (0.000 to 0.221); +1.2 hits come from two directions |


## Part 165 (registered 2026-10-31 23:56 CT, PART165-DESIGN.md, before any data)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 165.1 | n+ breaks at least twice as often as full | high | Claude | held: 2.70 (1.89 to 3.86); 2.02 without the ten slots that broke on all eight |
| 165.2 | Broken rate falls with k: Spearman at or below -0.8 across n±, k4, k16, k64, k256, k1024, full | medium | Claude | held: -0.857 (-0.964 to -0.631); -0.714 without layer 21, which would fail |
| 165.3 | k16's broken rate is nearer full's than n±'s | low | Claude | held: 0.053 against 0.105 |
| 165.4 | The n+ over n± broken ratio is between 0.67 and 1.5 | medium | Claude | held: 1.01 (0.78 to 1.30) |
| 165.5 | iso's broken rate is under 0.8 times full's | low | Claude | failed: 0.84 (0.55 to 1.27) |
| 165.6 | Changed rate falls with k: Spearman at or below -0.6 across the same seven arms | medium | Claude | failed, not read: 0.20; changed rate at ceiling in every arm |
| 165.7 | n+'s mean relative last-layer change is at least 1.5 times full's (Amendment 2) | medium | Claude | failed: 1.05 (1.03 to 1.07) |


## Part 166 (registered 2026-11-01 01:03 CST, PART166-DESIGN.md, after Part 165's data; the design discloses what was seen)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 166.1 | Pooled over n+, n± and k4, a loop's own hit rate is at least 2 times its shuffled rate | high | Claude | held: 15.5 (9.9 to 25.8); any-word, does not beat plain salience |
| 166.2 | Pooled over full and iso, own over shuffled is under 1.5 | medium | Claude | failed: 5.6 (3.1 to 12.5); dense own rates far lower |
| 166.3 | n+'s mean steadiness is at least 1.25 times full's (build check, Amendment 1) | medium | Claude | failed, build check: 1.00 |
| 166.4 | Mean sharpness falls with k: Spearman at or below -0.8 across n±, k4, k16, k64, k256, k1024, full (build check, Amendment 1) | medium | Claude | failed, build check: -0.714; sharpness peaks at k4 and k16 |
| 166.5 | Within n± and k4, pooled over slots, slot broken rate against slot mean sharpness: Spearman at least 0.3 | low | Claude | held: 0.397 (0.27 to 0.51) |
| 166.6 | Pooled over n+, n± and k4, content-word own over shuffled is at least 1.5 (Amendment 1) | medium | Claude | held: 53.1 (29.1 to 123.9); 47.2 with a wider stop list |
| 166.7 | Pooled over n+, n± and k4, the mean percentile of loop content words in the own bias is at least 0.65 (Amendment 1) | medium | Claude | held: 0.892 (0.81 to 0.90) |


## Part 167 (registered 2026-11-01 01:44 CT, PART167-DESIGN.md, before any data)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 167.1 | In the none arm, pooled over n+ and k4, content-word own over shuffled is at least 5 | high | Claude | Not read: Gate 2 failed |
| 167.2 | In the none arm, n+'s broken rate is at least 2 times full's | high | Claude | Not read: Gate 2 failed |
| 167.3 | Own ban cuts loop answers by at least 30% against none, pooled over n+ and k4 | medium | Claude | Not read: Gate 2 failed |
| 167.4 | Pooled over n+ and k4, the own loop rate is at least 0.05 below the matched rate (Amendment 1) | medium | Claude | Not read: Gate 2 failed |
| 167.5 | The matched cut, pooled over n+ and k4, is between -15% and 15% (Amendment 1) | medium | Claude | Not read: Gate 2 failed |
| 167.6 | Own ban's cut in full is under half its cut in n+ and k4 | low | Claude | Not read: Gate 2 failed |
| 167.7 | At least a third of paired transitions are migrations, and above their baseline (Amendment 1) | low | Claude | Not read: Gate 2 failed |


## Part 168 (registered 2026-11-01 02:46 CT, PART168-DESIGN.md, before any data)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 168.1 | In the none arm, pooled over n+ and k4, content-word own over shuffled is at least 5 | high | Claude | held: 215.6 |
| 168.2 | In the none arm, n+'s broken rate is at least 2 times full's | high | Claude | held: 2.42 (0.129 against 0.054) |
| 168.3 | Own ban cuts loop answers by at least 30% against none, pooled over n+ and k4 | medium | Claude | held by 1.4 points: 31.4% (15.8% to 44.9%); falls below 30% without layer 24, without phrasing 8, 11 or 14, and on n+ alone |
| 168.4 | Pooled over n+ and k4, the own loop rate is at least 0.05 below the matched rate | medium | Claude | failed: gap 0.047 (0.022 to 0.073) against 0.05 |
| 168.5 | The matched cut, pooled over n+ and k4, is between -15% and 15% | medium | Claude | held: -2.1% (-6.3% to 1.6%) |
| 168.6 | Own ban's cut in full is under half its cut in n+ and k4 | low | Claude | held, on 35 full-arm none loops: 11.4% (-20% to 36.7%) against 31.4% |
| 168.7 | At least a third of paired transitions are migrations, and above their baseline | low | Claude | failed: 1 of 19, 5.3%, baseline 1.0% |
| 168.8 | On the paired rows of Amendment 1, the own loop rate is at least 0.05 below the mass-ban rate | medium | Claude | held: 9.61% against 16.32% on 1,342 paired rows; gap interval 0.042 to 0.095 |
| 168.W1 | The mass ban cuts loop answers by 15% to 25% against none, pooled over n+ and k4 | low | Weaver | failed, opposite direction: -16.5% (-29.4% to -6.3%) |
| 168.W2 | Among rows with no loop under none, the induced rate under own is at least 0.02 above matched | low | Weaver | held: 3.6% against 1.0% |
| 168.W3 | Migrations are 15% to 30% of paired transitions | low | Weaver | failed: 5.3% |


## Part 169 (registered 2026-11-01 05:29 CT, PART169-DESIGN.md, before any data)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 169.1 | Own ban cuts loop answers by at least 25% against none | medium | Claude | held: 37.9% (25.8% to 48.6%) |
| 169.2 | Own's relative cut on high rows is at least 0.25 above its relative cut on other rows | medium | Claude | held: 0.371 above (0.187 to 0.531) |
| 169.3 | Free ban raises loop answers by at least 5% against none | medium | Claude | failed: 0.0% (-6.1% to 7.0%) |
| 169.4 | On paired rows, clear's loop rate is at least 0.02 below free's | medium | Claude | failed: -0.004 (-0.022 to 0.014); read only as a difference between bans |
| 169.5 | On high paired rows with no loop under none, free's induced rate is at least 0.10 above clear's | medium | Claude | failed: 0.022 (-0.024 to 0.070); read only as a difference between bans |
| 169.6 | With no push, free lists give loops on under 5% of answers | medium | Claude | held: 3.1%, 24 of 768 |
| 169.7 | Clear's change in loop answers against none, on paired rows, is between -15% and 15% | low | Claude | held: clear added 1.9% (-4.7% to 9.3% as a change) |
| 169.8 | On rows whose free list holds two or more plain tokens and with no loop under none, hit's induced rate is at least 0.6 times free's | medium | Claude | held as a difference between bans: hit 0.167, free 0.192, no net rise |
| 169.W1 | Among rows with no loop under none, plain50's induced rate on rows above the median overlap between plain50 and the row's none answer is at least 0.05 above its rate on the other rows | low | Weaver | failed: -0.102 (-0.160 to -0.045) |
| 169.W2 | plain50 raises low-k loop answers against none by at least a third of own's net cut in this run | low | Weaver | held: 283 against 42.7 |


## Part 170 (registered 2026-11-01 08:34 CT, PART170-DESIGN.md, before any data)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 170.1 | No push, identity prompts (P8 and M96): b50's rise in loop rate is under a quarter of its rise at dose 1, with the absolute rise beside it | medium | Claude | held: 17.4% (0.0% to 44.3%), on eligible rows |
| 170.2 | No push, Q96: the same bar, against the same dose 1 rise | medium | Claude | failed: 36.0% (5.5% to 80.4%) |
| 170.3 | Dose 1: b50 raises loop answers by at least 50% against b0 | high | Claude | held: 75% relative rise (margin interval -4.0 to +16.0 points) |
| 170.4 | b50's rise at dose 1 is positive and at least twice its rise at dose 0.25 | medium | Claude | failed: 18.75 against 13.0 points, 1.44 times |
| 170.5 | Dose 1: b200's loop rate is at least 0.05 above b50's, and b5's at least 0.05 below b50's | medium | Claude | failed: b50-b5 +4.7 points, b200-b50 +6.5 points |
| 170.6 | Dose 1: rel50's rise as a share of b50's is under 1.5 times rel50's share of b50's blocked steps | low | Claude | held: -11.1% against a 26.8% bar, which any negative rise passes |
| 170.7 | No push, Q96: other50's change in loop rate is under 0.03 either way, read beside b5 and b50 | medium | Claude | held: +1.1 points (0.0 to 5.6) |
| 170.8 | Dose 1, rows with no loop under b0: within b0 length quartiles, weighted by rows, b50's induced rate below each quartile's median share is at least 0.05 above the rest | low | Claude | failed: +4.0 points (-7.6 to 16.3) |
| 170.W1 | No-push b50 rise is under 0.05 on both M96 and Q96. Falsifier: a rise of 0.10 or more on either set | low | Weaver | failed: M96 +3.6, Q96 +6.7 points; falsifier not met |
| 170.W2 | Dose 1: rel50's rise is at least half of b50's. Falsifier: under a third | low | Weaver | failed: -11.1%; falsifier met |


## Part 171 (registered 2026-11-01 12:29 CT, PART171-DESIGN.md, before any data)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 171.1 | n100's hit rate is at least 3 times r100's, averaged over draws | medium | Claude | failed: 0.52 (0.10 to 1.31); per draw 1.33, 1.00, 0.25 |
| 171.2 | n100's hit rate is at least 0.6 of vector's | medium | Claude | failed: 0.09 (0.02 to 0.19), against a vector arm that broke 95.5% of answers |
| 171.3 | n1's hit rate is at least 2 times r1's | low | Claude | held: 68.2% against 2.3%; the ratio rests on 3 answers |
| 171.4 | rand100's hit rate is under 5% | high | Claude | failed: 12.9% (8.3% to 17.4%); per draw 36.4%, 2.3%, 0% |
| 171.5 | At equal length, r100's mean cosine with v̂ is at least 0.1 below n100's | high | Claude | held: +0.21 on the mean and every draw |
| 171.6 | Pooled over testable sets with a positive n100 minus r100 gap, Σ(r100p − r100) / Σ(n100 − r100) is at least 0.5; not read below 3 such sets | low | Claude | not read: needs 3 testable sets with a positive gap, and none had one |
| 171.7 | r100p breaks or loops on at least 10 points more answers than n100 | medium | Claude | held: +0.57 on 2 testable sets; every r100p answer broke at about 1.9 times the write |
| 171.8 | n100's hit rate is at least that of `full` | low | Claude | held raw: 9.1% against 0%; full broke on every answer, so this does not test sparsity |
| 171.W1 | The median alignment debt over the 12 sets is at least 2, and 171.6 fails because the arm cannot help it. Falsifier: a debt near 1, which makes r100p the clean test | medium | Weaver | failed: median debt 1.91; falsifier not met (1.83, 1.91); the 171.6 part is not read |


## Part 172 (registered 2026-11-01 18:57 CT, PART172-DESIGN.md, before any pushed answer)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 172.1 | `n1` at d = 1 gives clean Google (target + both) on at least 10 of 44 | medium | Claude | held: 17 of 44 (11 to 24) |
| 172.2 | `n1`'s strict share at d = 1 is at least 30 points above the pooled strict share of all 12 control sets, and its clean family-hit count is at least 5 above the mean control set's | medium | Claude | held at d = 1 only: gap 66.0 points (45.0 to 83.8), excess 15.17 hits (8.83 to 21.92); at d = 0.5 r100 draw 2171 has 35 clean hits, n1 has 4 |
| 172.3 | `rand100_d171` and `r100_d2171` each keep a GS of at least 0.25 with a GS denominator of at least 5 at d = 1 on `NEW44`: the Google lean belongs to the set, not to Part 171's prompts | medium | Claude | failed: rand100_d171 45.2% of 31, r100_d2171 7.1% of 14; both had to meet the bar |
| 172.4 | Some ladder dose of `n1` has clean Google on at least 7 of 44 and broken on at most 8 of 44 | low | Claude | held (outcome: supported): 18 hits and 4 broken at d = 0.75 |
| 172.5 | At `n1`'s lowest dose with clean Google on at least 7 of 44, a Qwen name survives in at least half of clean answers | low | Claude | failed: 17 of 40 clean answers, 42.5% |
| 172.6 | With L27 N3325 zeroed, `n1`'s clean family-hit count at d = 1 falls by at least half | medium | Claude | failed: 18 with the writer zeroed against 17; but 18 to 6 at d = 0.75 |
| 172.7 | `n1` at d = -1 gives clean Google on at most 2 of 44 | high | Claude | held: 0 of 44 |
| 172.8 | At least 2 of `n2` to `n5` reach a GS of at least 0.5 with a GS denominator of at least 5 at d = 1 | low | Claude | failed: 1 of 4, n4 at 59.5% of 42; at d = 0.5 n2 has 44 clean hits and n5 12 |
| 172.9 | `n1`'s clean Google answers at d = 1 name Gemini on at most 20%, against the role's near 100% | high | Claude | held on the point estimate: 3 of 17, 17.6% (0 to 38.5%); one more answer would fail it |
| 172.10 | `frame_n1`, no push, names Google on at least 22 of 44 clean answers | medium | Claude | failed: 4 of 44 |
| 172.11 | `n1`'s clean Google answers at d = 1 carry the `frame` marker on at least 60% | medium | Claude | held: 16 of 17, 94.1% (80 to 100) |
| 172.12 | `frame_plain`, no push, names Qwen, Tongyi or Alibaba on at least 30 of 44 clean answers: not every frame leads to Google | high | Claude | held: 44 of 44 |
| 172.W1 | At every ladder dose of `n1` with clean Google on at least 7 of 44, broken is on at least 15 of 44. Falsifier: any such dose with broken on at most 8 of 44. Between 8 and 15 is inconclusive. No such dose makes it not testable. Beside it, Weaver forecasts that 172.2 holds | medium | Weaver | failed by its falsifier: d = 0.75 has 18 hits and 4 broken |


## Part 173 (registered 2026-11-02 00:23 CT, PART173-DESIGN.md, before any pushed answer)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 173.1 | At least 5 of the 12 fit-built control sets have a supported modal replacement at some dose | medium | Claude | held: 7 of 12 |
| 173.2 | At least 2 of the 6 random-valued sets have a supported modal replacement at some dose | low | Claude | held: 3 of 6 |
| 173.3 | Across the sets' best cells, at least 3 different entities are modal, and Google is modal on at most half | medium | Claude | failed: 5 entities, Google on 66.7% |
| 173.4 | `r100_d2171` at d = 0.75 and `n2` at d = 0.5 each have Google as `modal`, `modal_count` at least 30 of 44, broken at most 4 | high | Claude | held: 43 of 44 and 39 of 44, broken 0 and 3 |
| 173.5 | `n1` at d = -0.5 has Tencent as `modal`, `modal_count` at least 10 of 44, broken at most 8 | medium | Claude | held: Tencent 27 of 44, 1 broken |
| 173.6 | Capacity under the damage cap: `n1`'s highest Google claim count at a cell with broken at most 8 is above the median of the 12 fit-built sets' same figure by at least 5, and not the highest of the 13; a set with no such cell counts 0 | low | Claude | held: n1 15, median 7, maximum 43 |
| 173.7 | Of the set-and-size pairs with supported modal replacements at both +d and -d, for |d| of 0.5 and 1 only, at least 75% have different entities at the two signs; not read below 4 pairs | low | Claude | not read: 1 qualifying pair of the 4 required |
| 173.8 | Google is modal at a supported cell in at most 1 of the 6 random-valued sets, and in at least 4 of the 17 fit-built and picked sets | medium | Claude | failed: 3 random-valued sets, 9 fit-built and picked |
| 173.9 | At least 60% of all supported cells are `stable`; not read below 5 supported cells | medium | Claude | held: 23 of 25 stable, 92% |
| 173.W1 | `none` already names at least one known non-Qwen entity on at least 3 of 44. Falsifier: no non-Qwen entity reaches 3 of 44 in `none` | medium | Weaver | failed: no known non-Qwen entity in none |
| 173.W2 | At most 1 of the 6 random-valued sets has a supported modal replacement. Falsifier: at least 2 of 6 do, which makes 173.2 right and value-randomness irrelevant to whether a lead forms | medium | Weaver | failed: 3 of 6 supported |
| 173.W3 | If 173.W1 holds, at least 3 of the supported cells have `modal_count` under 2 times the `none` baseline. Falsifier: every supported cell is at 3 times baseline or more. Between is inconclusive; not testable if 173.W1 fails or no cell is supported | medium | Weaver | not testable: 173.W1 failed |


## Part 174 (registered 2026-11-02 04:22 CT, PART174-DESIGN.md, before any answer on these draws)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 174.1 | At least 20 of the 80 `u100d` signed directions are supported | medium | Claude | held: 28 of 80 |
| 174.2 | Google is modal on at least 80% of supported `u100d` directions; not read below 10 supported | medium | Claude | held: 28 of 28 |
| 174.3 | At most 4 of the 80 `u100d` signed directions are other-led | medium | Claude | held: 0 of 80 |
| 174.4 | Of the `u100d` draws with at least one supported sign, at most 25% are `both same` or `both different`; not read below 8 such draws | low | Claude | held: 4 of 24 (16.7%), all both same |
| 174.5 | `f100d`'s supported share is at least half of `u100d`'s; literal result always reported, read as retention only if `u100d` has at least 10 supported, else `uninformative` | low | Claude | failed: 0 of 40 against 28 of 80, nearly every flat answer broke |
| 174.6 | Across the 60 draws, the observed |Spearman(cf, D)| is under 0.30 | medium | Claude | not read: no `f100d` draw has a D, and 7 `u100d` draws lack one |
| 174.7 | The median `broken` of the 80 `u100d` signed directions is at most 8 of 44 | medium | Claude | held: median 5 of 44 |
| 174.8 | At least 60% of all supported directions are `stable`; not read below 5 supported | medium | Claude | held: 27 of 28 (96.4%) |
| 174.9 | Across the 60 draws, the observed |Spearman(cr, D)| is under 0.30 | low | Claude | not read: no `f100d` draw has a D, and 7 `u100d` draws lack one |
| 174.10 | Google is modal on at least 80% of supported `f100d` directions; not read below 5 supported | medium | Claude | not read: no supported `f100d` direction |
| 174.W1 | At least 30 of the 80 `u100d` signed directions are supported. Falsifier: fewer than 20, which would say the three draws were a lucky sample rather than a rate. 20 to 29 is inconclusive | medium | Weaver | inconclusive: 28 of 80 |
| 174.W2 | Other-led directions are present but few, and the majority name invented entities. Falsifier: at least half of the other-led directions name a known non-Qwen entity. Not testable with no other-led direction. Held if a strict majority of the other-led directions are invented; how few they are is reported, not scored. Invented means outside the entity table, not shown to be fictional | medium | Weaver | not testable: no other-led direction |
| 174.W3 | `f100d`'s supported share is within 20 points of `u100d`'s. Falsifier: `f100d` is below half of `u100d`, which would make the layer profile or the spread a real gate, without saying which. The falsifier takes precedence where both are true. Any other outcome is inconclusive | medium | Weaver | failed: 0% against 35%, on breakage at this dose |


## Part 175 (registered 2026-11-02 07:32 CT, PART175-DESIGN.md, before any answer on these doses)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 175.1 | Stage A finds a D* at or above 0.10 | high | Claude | held: D* = 0.30 |
| 175.2 | At D*, at least 8 of the 40 `f100d` signed directions are supported | low | Claude | failed: 1 of 40, with 175.5 failed |
| 175.3 | Google is modal on at least 80% of supported directions at D*; not read below 5 supported | medium | Claude | not read: 1 supported, below 5 |
| 175.4 | At D*, at most 2 of the 40 signed directions are other-led | medium | Claude | held: 0 of 40, with 175.5 failed |
| 175.5 | At D*, the median `broken` of the 40 signed directions is at most 8 of 44 | medium | Claude | failed: median 29.5 of 44; Stage A gave 6.5 |
| 175.6 | More directions are supported at D* than at 0.5 D*; not read if neither dose has at least 3 | medium | Claude | not read: 1 at D* and 1 at 0.5 D* |
| 175.W1 | D* lands between 0.20 and 0.35. Falsifier: D* at 0.10 or 0.45, or no D*. 0.15 and 0.40 are inconclusive | medium | Weaver | held: D* = 0.30 |
| 175.W2 | At most 5 of the 40 signed directions are supported at D*. Falsifier: 8 or more. 6 or 7 is inconclusive | medium | Weaver | held: 1 of 40, at a breaking dose |
| 175.W3 | At least 3 of the 4 Stage A draws have a sign gap of at least 2 ladder steps, and then D* is carried as a band. Falsifier: 2 or fewer | medium | Weaver | failed: 0 of 4 draws |


## Part 176 (registered 2026-11-02 12:44 CT, PART176-DESIGN.md, before any answer on these draws)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 176.1 | At 0.225, at least 20 of the 40 signed directions are supported | medium | Claude | held: 30 of 40 supported |
| 176.2 | At 0.225, Google is modal on at least 80% of supported directions; not read below 5 supported | high | Claude | held: 30 of 30 supported directions Google-led |
| 176.3 | At 0.225, at most 2 of the 40 signed directions are other-led | high | Claude | held: 0 of 40 other-led |
| 176.4 | Not a window: at least 8 of 40 are supported at 0.20, and at least 8 of 40 at 0.25 | medium | Claude | held: 20 to 21 of 40 at 0.20, 20 of 40 at 0.25 |
| 176.5 | At 0.225, the median `broken` of the 40 signed directions is at most 8 of 44 | high | Claude | held: median 1 of 44 broken |
| 176.6 | Pooled over the 40, Google's fraction of named answers at 0.275 is at least its fraction at 0.225, and named answers at 0.275 are under half those at 0.225; not read below 100 named at 0.275 | medium | Claude | held, aggregate only: share 43.0-43.3% to 52.4-53.0%, named 1,483-1,486 to 519-521 |
| 176.7 | No strong tie: over eligible directions, the Spearman of the Google count of 44 at 0.225 against `collapse dose` is above -0.30. Fails at or below -0.30 with the cluster bootstrap's upper bound below 0 | low | Claude | held, a non-detection: rho +0.016 [-0.32, +0.35], 37 directions from 19 draws |
| 176.W1 | Support rises through 0.25 (0.25 ≥ both 0.20 and 0.225) and falls under 5 of 40 at 0.30, with 0.225 sitting on the rising slope rather than at a peak. Falsifiers: 0.225 exceeding both neighbours by ≥5 of 40, which restores the peak; and 15 or more of 40 supported at 0.30, which means no collapse happened and the whole band reading is wrong | medium | Weaver | failed: 20 to 21, 30, 20 and 0 of 40 at 0.20, 0.225, 0.25 and 0.30; 0.225 exceeds both neighbours by at least 5 |
| 176.W2 | Google's share of named answers rises monotonically across the rungs while the raw Google count peaks at or below 0.25. Falsifier: raw Google count still rising at 0.275 or 0.30 | medium | Weaver | failed: share fell 6.3 to 6.9 points at 0.30, on 115 named answers; the count clause did not fail |
| 176.W3 | Over directions with a supported rung, the median `gap`, from a direction's top supported rung to its `collapse dose`, is at least 0.05. Falsifier: under 0.025, which would put naming at the edge of collapse and make Muse's warning live | medium | Weaver | failed: median gap 0.0177, 22 of 34 under 0.025; most of that is the rung spacing |


## Part 177 (registered 2026-11-02 18:26 CT, PART177-DESIGN.md, before any outcome was read)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 177.1 | On `T`, the share of traces with an `asserted` real gold exceeds the same share for shuffled draw 1 by at least 15 points | medium | Claude | held: 67 of 173 real, 0 of 173 shuffled |
| 177.2 | On `T`, at least 25% of traces have an `asserted` real gold | medium | Claude | held: 67 of 173, 38.7% |
| 177.3 | Among `T` traces with an `asserted` real gold and a detected onset, at least half have the first asserted hit before the onset; not read below 20 traces | medium | Claude | held: 49 of 66 before the onset |
| 177.4 | At least 80% of `p50c` traces that are `capped` with no `</think>` carry a detected onset. This is the persistence of the signature among residual failures and says nothing about incidence | medium | Claude | failed: 35 of 48, 72.9% |
| 177.5 | On problems in the same cohort in both arms with a detected onset in both, median onset as a share of text is later in `p50c`; not read below 15 pairs | medium | Claude | held on the median: 0.16 to 0.32, difference interval -0.03 to 0.40 on 34 pairs |
| 177.6 | Over the 200 paired problems, with onset searched only in the first 50,000 characters of each text and pairs with a shorter text left out, the rate of a detected onset is at least 15 points lower in `p50c` | medium | Claude | held: 111 of 116 against 79 of 116, on pairs where both texts passed 50,000 characters, which selects on post-edit length |
| 177.W1 | The real-against-shuffled literal-hit gap on `C` exceeds 15 points, but at least 20% of real hits sit in the first 15% of text, and excluding them the gap falls under 15. Falsifier: fewer than 10% of hits in the first 15% of text | medium | Weaver | inconclusive: gap 41.7 points, 31.6 with early hits dropped; early share 10.5% of all hits, 94.8% of first hits (Amendments 1 and 2) |
| 177.W2 | Shuffled windows clear 40% yes on Gate 4. Falsifier: under 20% | medium | Weaver | failed: 0 of 98 shuffled windows |
| 177.W3 | Among `C` traces with a literal hit, at least a third have a `pred_first` that does not match the gold. Falsifier: under 15% | medium | Weaver | failed under the registered reading: 0 of 115 with a wrong box; 113 of 115 null, so the plain wording would hold |


## Part 178 (registered 2026-11-02 22:10 CT, PART178-DESIGN.md, before any text from the 200 problems was generated or read)

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 178.1 | In `U`, `wait` is among the top 10 openers by `rate(w)` | medium | Claude |
| 178.2 | In `G`, `wait` is among the top 10 openers by `rate(w)` | medium | Claude |
| 178.3 | `rate(wait)` differs by less than a factor of 1.5 between `U` and `E` on `window`; unread if either is zero | medium | Claude |
| 178.4 | At least 14 of `U`'s top 20 openers are in `G`'s top 20 | medium | Claude |
| 178.5 | At most 6 of the 17 imported words reach 5 per 1,000 paragraph openers in `U` | medium | Claude |
| 178.6 | The follow-on statistic on eligible traces is at least 0.15; not read below 40 eligible traces | low | Claude |
| 178.7 | At least 10 of `G`'s top 20 openers have a mode ratio under 2; unread if more than 5 of the 20 have no ratio | medium | Claude |
| 178.W1 | Ranking by `rate(w)` and by `traces(w)` disagree on at least 4 of the top 20 in `U`. Falsifier: 1 or fewer | medium | Weaver |
| 178.W2 | Among traces with both labels, the median token position of the first `asserted` selected hit exceeds the first `intermediate` selected hit's by at least 2,000 tokens. Falsifier: under 500 | medium | Weaver |
| 178.W3a | On Qwen3.5-0.8B, `wait` occurs at least 3 times more often per 1,000 openers in thinking than in non-thinking text, by `mode ratio` of `R8` over `N8`. Falsifier: ratio under 1.5. Unread under 500 control openers or with no ratio | medium | Weaver |
| 178.W3b | The same on Qwen3.5-4B, by `mode ratio` of `G` over `N4` | medium | Weaver |


## Part 181 (registered 2026-11-03 19:13 CT, PART181-DESIGN.md, before any checkpoint was collected)

Overlap = the share of a unit's top-10 documents (by per-document maximum activation over 2,500 pile-10k documents) shared between two checkpoints, read against each layer's own null. Reading rule and gates as in the design.

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 181.1 | On the primary cohort, over the layers passing Gate 3, the median neuron overlap of `midtrain_51B` against `instruct` is at least 0.6 | 0.7 | Claude |  |
| 181.2 | The median neuron overlap of `dpo` against `instruct` is at least 0.9; not assessable if the floor's own median (`pretrain_4T` against `pretrain_4T_bf16r`) is below 0.9 | 0.75 | Claude |  |
| 181.3 | The median neuron overlap of `pretrain_21B` against `instruct` is at most 0.1 | 0.7 | Claude |  |
| 181.4 | The mean neuron overlap with `instruct` rises strictly along 21B, 210B, 1T, 2T, 3T, 4T (medians reported beside) | 0.6 | Claude |  |
| 181.5 | For each of (1T, 2T), (4T, midtrain_51B) and (midtrain_51B, instruct), the mean neuron overlap exceeds both the mean dense-control and the mean k = 2 sparse-control overlap in at least three quarters of the layers passing Gate 3; overlap against k (1, 2, 4, 16, 64, dense) reported for every pair | 0.6 | Claude |  |
| 181.6 | The median `sft`-against-`dpo` overlap of Part 92's 21 slot neurons is within 0.15 of the median of their 21 layer comparators (each the median of all primary-cohort neurons in its layer) | 0.55 | Claude |  |
| 181.W1 | Run-to-run noise is larger early: the 1T repeat pair's floor (1 minus median neuron top-10 overlap, median over Gate-3 layers) is at least 2 times the Instruct pair's. Fails under 2, or if both floors are 0. | medium | Weaver |  |
| 181.W2 | A relative activity rule marks at least 10% more neurons active than an absolute cut (10% of the layer's median maximum at Instruct, fixed) at one or more pretraining checkpoints. Fails if under 10% at every one. | medium | Weaver |  |


## Part 180b (registered 2026-11-03 23:03 CT, PART180B-DESIGN.md, before any reader call): how does Part 180's blind reader err on known answers?

DS is DeepSeek V4.1 Flash with Part 92's label prompt (`part92/label92.py`); Haiku is Claude Haiku 5.5 with label92's settings. Items are `results/part180b/items.jsonl`, gold is `results/part180b/KEY-private.json` (built before any label from `part180b/personas180b.py` and `part180b/codebook180b.json`). Gate, denominators and "not read" rules as in the design; boundary items are excluded unless a prediction names them.

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 180b.1 | DS marks shows_persona on at least 80% of topic items and at most 60% of lived items (full text, home question). | medium | Claude |  |
| 180b.2 | DS marks shows_persona on at most 5% of generic, at most 10% of near_miss and at most 10% of ai_generic items (full text, home question). | medium | Claude |  |
| 180b.3 | DS marks shows_persona on at least 80% of loop_persona items and coherent on at most 20% of them. | low | Claude |  |
| 180b.4 | Over the clear-category pairs labelled under both their home and alternate question, DS's shows_persona differs on at most 10% of pairs, and on lived items the yes rate under 'Describe your perfect day.' is not more than 10 points below the yes rate under 'Tell me about yourself.'. | medium | Claude |  |
| 180b.5 | On claim_open items under 'What is the capital of France?', DS marks shows_persona on at least 90%. | medium | Claude |  |
| 180b.6 | On claim_open and lived items under 'What is the capital of France?', DS marks coherent on at most 50%. | low | Claude |  |
| 180b.7 | On the real full-text answers under their own persona and question, DS's shows_persona agrees with its Part 92b label on at least 90% of the items valid in both. | medium | Claude |  |
| 180b.8 | Among real neuron-arm items (full and prefix) the codebook codes shows_persona TRUE, boundary items included, DS marks TRUE on at most 60%; among those it codes FALSE without a boundary flag, DS marks TRUE on at most 5%. | medium | Claude |  |
| 180b.9 | DS marks shows_persona on at most 5% of the 53 swapped-target items. | high | Claude |  |
| 180b.10 | Haiku marks shows_persona on lived items at least 20 points more often than DS, and on incidental items at least 15 points more often than DS (full text, home question; read only with at least 20 Haiku labels in each category). | low | Claude |  |
| 180b.11 | Haiku marks shows_persona on at most 10% of generic items (full text, home question; read only with at least 20 Haiku labels). | medium | Claude |  |


## Part 183 (registered 2026-11-04 01:16 CT, PART183-DESIGN.md, before any eager answer to a Part 25 or Part 66 prompt)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 183.1 | A: with the four zeroed, Google in 5% to 30% of the 192 held-out answers (66.1's band). Fails outside it. | high | Claude || 22.4% (43 of 192): held |
| 183.2 | A: unchanged, Google in at most 2 of 192. Fails at 3 or more. | high | Claude || 2 of 192: held |
| 183.3 | A: the four beat all 50 random sets (no random set's held-out Google rate at or above the four's). Fails if any set ties or beats them. | medium | Claude || best random 4.2%, p = 1/51: held |
| 183.4 | A: the zeroed Google rate under eager is within 10 points of SDPA's 21.9% (11.9% to 31.9%). Fails outside it. | medium | Claude || 22.4% against 21.9%: held |
| 183.5 | B: with the four zeroed, "Who created you?" names Google or DeepMind as maker and the other five identity answers keep an Alibaba, Tongyi or Qwen-team maker (or none), the published 1-of-6 pattern. Fails on any other pattern. | medium | Claude || only Who created you? flips, to Google: held |
| 183.6 | B: none of the 20 random sets gives a non-Alibaba maker on any of the six identity questions. Fails if any does. | medium | Claude || 0 of 120 answers: held |
| 183.7 | B: for no zeroing and for the four, the classified maker of each of the 12 answers equals that of the stored SDPA answer on at least 11 of 12. Fails at 10 or fewer. | medium | Claude || 12 of 12: held |


## Part 184 (registered 2026-11-04 04:00 CT, PART184-DESIGN.md, before any eager answer to a Part 18 or 18b prompt)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 184.1 | A: the clamped (+2) samples are alien in at least 7 of 8. Fails at 6 or fewer. | high | Claude | 8 of 8: held |
| 184.2 | A: the unclamped samples are alien in 0 of 8, and the 160 control samples at +2 are alien in at most 2. Fails at 1 or more unclamped, or 3 or more control samples. | high | Claude | 0 of 8; 0 of 160: held |
| 184.3 | A: the controls' greedy answers at +2 name Qwen in at least 18 of 20 and are alien in 0 of 20. Fails otherwise. | high | Claude | 20 of 20 name Qwen, 0 alien: held |
| 184.4 | A: at -2, the samples name Qwen in at most 4 of 8 (published 2 of 8). Fails at 5 or more. | medium | Claude | 1 of 8: held |
| 184.5 | B: the generation-only clamp gives alien in at least 5 of 8 samples (published 7 of 8). Fails at 4 or fewer. | medium | Claude | 8 of 8: held |
| 184.6 | C: the prompt-only clamp gives alien in at most 1 of 8 samples (published 0 of 8). Fails at 2 or more. | medium | Claude | 0 of 8: held |


## Part 185 (registered 2026-11-04 06:53 CT, PART185-DESIGN.md, before any eager answer to a Part 84 or 84b prompt)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 185.1 | A: at dose 0.63 the alien neuron gives an off-world claim in at least 22 of 24 answers in both A2 and A3 (published 24 and 24). Fails at 21 or fewer in either. | high | Claude | A2 23, A3 24 of 24: held |
| 185.2 | A: the unchanged model (A0) and the control unit (C) each give an off-world claim in at most 1 of 24 (published 0 and 0). Fails at 2 or more in either. | high | Claude | A0 0, C 0 of 24: held |
| 185.3 | A: at dose 0.47 (A1) between 8 and 22 of 24 answers claim an off-world origin (published 17 by the reader). Fails outside that range. | medium | Claude | 13 of 24: held |
| 185.4 | B: in each mode the off-world count rises strictly from 0.40 to 0.47 to 0.55 (published decode 4, 25, 43; gen 3, 23, 46). Fails if any step does not rise. | high | Claude | decode 3, 28, 44; gen 4, 29, 47 of 48: held |
| 185.5 | B: steering the first answer token adds no detectable amount: at each dose the paired sign test on discordant pairs (gen only off-world against decode only) gives p of at least 0.05, two-sided (published 1.0, 0.69, 0.25). Fails at any dose below 0.05. | medium | Claude | p 1.0, 1.0, 0.25: held, but uninformative: with 3 to 5 discordant pairs per dose p < 0.05 was unreachable |
| 185.6 | B: each of the seven cells' off-world count is within 8 of its published count of 48 (A0 0; decode 4, 25, 43; gen 3, 23, 46). Fails if any cell moves by 9 or more. | medium | Claude | largest move +6 (gen 0.47): held |


## Part 186 (registered 2026-11-04 08:11 CT, PART186-DESIGN.md, before any eager answer to a Part 79 prompt)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 186.1 | control: with the makers zeroed (e1), 26 to 60 of 192 held-out answers name Google by keyword (published 46). Fails outside that range. | high | Claude | 51: held |
| 186.2 | the evil push reproduces: the blind reader labels at least 96 of e2's 192 answers villainous and at most 9 of e0's (published 125 and 0). Fails otherwise. | high | Claude | e2 136, e0 0: held |
| 186.3 | no composition: at most 1 of e3's 192 answers is both villainous and Google by the reader (published 0). Fails at 2 or more. | high | Claude | 0: held |
| 186.4 | the evil push removes the maker: e3 names a company (reader google or another real company, or the text naming Alibaba or Tongyi) in at most 6 of 192 (published 1). Fails at 7 or more. | medium | Claude | 0: held |
| 186.5 | Google falls under the evil push: e3 has at most 3 Google answers by keyword (published 0). Fails at 4 or more. | high | Claude | 0: held |
| 186.6 | medians: each of the eager step 0 medians at layers 13, 14 and 22 is within 2% of the published 9.509, 9.857 and 22.96. Fails if any differs by more. | low | Claude | 9.509, 9.857, 22.959: held |


## Part 187 (registered 2026-11-04 10:28 CT, PART187-DESIGN.md, before any eager answer to a Part 140 prompt)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 187.1 | arm A reproduces: arm A's Set P joint success is between 36.0% and 60.0% (published 48.3%). Fails outside. | high | Claude || held (48.3%) |
| 187.2 | release ends the persona: arm C's Set P joint success is at most 15.0% (published 5.8%). Fails above. | high | Claude || held (7.1%) |
| 187.3 | random directions stay low: Set C's joint success is at most 10.0% in every arm (published at most 6.1%). Fails if any arm exceeds it. | high | Claude || held (max 3.9%) |
| 187.4 | the verdicts stand: of the 9 published verdicts of 140.1 to 140.8 (140.5's two arms counted separately), at least 8 are the same by `score140.py`. Fails at 2 or more flips. | high | Claude || held (0 flips) |
| 187.5 | no verdict flips: all 9 published verdicts are the same. Fails at any flip. | medium | Claude || held (0 flips) |
| 187.6 | arm F stays broken: arm F's Set P joint success is at most 5.0% (published 0.0%). Fails above. | high | Claude || held (0.0%) |


## Part 188 (registered 2026-11-04 17:39 CT, PART188-DESIGN.md, before any recording or analysis of the Part 188 manifest)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 188.1 | the model's own verdict (class-informed) has held-out AUC below 0.75 in both test folds | medium | Claude || failed (0.631, 0.808) |
| 188.2 | the selected unit's held-out AUC is at least 0.85 in both test folds | medium | Claude || failed (0.753, 0.897) |
| 188.3 | the selected unit beats the model's own verdict (class-informed) by at least 0.10 held-out AUC in both test folds | medium | Claude || failed (+0.123, +0.089) |
| 188.4 | the selected unit separates real animals (test fold R) from ESC-50 human sounds (H) at AUC at least 0.80, its selection sign, in both folds: it reads "a human made this", not only imitation | medium | Claude || failed (0.281, 0.677) |
| 188.5 | L17 N1574, real above imitation, on all non-chicken groups pooled, has AUC below 0.70 | high | Claude || held (0.647) |
| 188.6 | the probe on the output embedding has held-out AUC at least 0.90 in both test folds | medium | Claude || held (0.918, 0.965) |
| 188.7 | the top-32 probe's held-out AUC exceeds the selected unit's by less than 0.05 in both test folds (a probe below the unit also holds) | low | Claude || failed (gaps 0.152, 0.061) |
| 188.8 | both selected units separate real chickens from chicken imitations (set C) at AUC at least 0.80 | medium | Claude || held (0.983, 0.913) |
| 188.9 | sequence length alone has held-out AUC below 0.70 in both test folds | medium | Claude || held (0.632, 0.634) |


## Part 189 (registered 2026-11-04 18:09 CT, PART189-DESIGN.md, before any eager answer to a Part 148 prompt)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 189.1 | the verdicts stand: of Part 148's six published verdicts (148.1 failed, 148.2 failed, 148.3 held, 148.4 failed, 148.5 failed, 148.6 failed; results/part148/score.json), at least 5 are the same by the unchanged scoring rules. Fails at 2 or more flips. | high | Claude || withheld (Gate 2 failed: 17 of 32 in 40 tokens) |
| 189.2 | no verdict flips: all six are the same. Fails at any flip. | medium | Claude || withheld (Gate 2 failed: 17 of 32 in 40 tokens) |
| 189.3 | the neuron rates for name (published 348/465, 74.8%), maker (191/465, 41.1%), name and maker together (146/465, 31.4%) and maker among name-moving neurons (146/348, 42.0%) each stay within 6 percentage points of the published rate. Fails if any leaves its band. | medium | Claude || withheld (Gate 2 failed: 17 of 32 in 40 tokens) |
| 189.4 | random directions stay level: the random-direction name rate stays within 6 percentage points of 175/465 (37.6%). Fails outside. | medium | Claude || withheld (Gate 2 failed: 17 of 32 in 40 tokens) |
| 189.5 | the slot and concept maker rates stay within 6 percentage points of 82/189 (43.4%) and 100/257 (38.9%). Fails if either leaves its band. | low | Claude || withheld (Gate 2 failed: 17 of 32 in 40 tokens) |
| 189.6 | Part 148's full-text Gate 2 would also pass: at least 28 of the 32 pilot answers equal Part 112's stored answers in full. Fails below 28. | low | Claude || withheld (Gate 2 failed: 17 of 32 in 40 tokens) |


## Part 189b (registered 2026-11-04 18:27 CT, PART189B-DESIGN.md, before any eager answer to a Part 148 prompt is scored)

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 189b.1 | the verdicts stand: of Part 148's six published verdicts (148.1 failed, 148.2 failed, 148.3 held, 148.4 failed, 148.5 failed, 148.6 failed; results/part148/score.json), at least 5 are the same by the unchanged scoring rules. Fails at 2 or more flips. | high | Claude || held (5 of 6 the same) |
| 189b.2 | no verdict flips: all six are the same. Fails at any flip. | medium | Claude || failed (148.1 flips) |
| 189b.3 | the neuron rates for name (published 348/465, 74.8%), maker (191/465, 41.1%), name and maker together (146/465, 31.4%) and maker among name-moving neurons (146/348, 42.0%) each stay within 6 percentage points of the published rate. Fails if any leaves its band. | medium | Claude || failed (name +8.0, both +9.2, maker among name +7.1) |
| 189b.4 | random directions stay level: the random-direction name rate stays within 6 percentage points of 175/465 (37.6%). Fails outside. | medium | Claude || failed (+11.4) |
| 189b.5 | the slot and concept maker rates stay within 6 percentage points of 82/189 (43.4%) and 100/257 (38.9%). Fails if either leaves its band. | low | Claude || held (+4.7, +4.7) |


## Part 190 (registered 2026-11-07 23:20 CT as v2, PART190-DESIGN.md, before any statistic is computed on results/part188/emb188.npz; v1 drafted 2026-11-04 23:56 CT and revised before registration, see the design's "Changes in v2")

| # | Prediction | Confidence | Author | Result |
| --- | --- | --- | --- | --- |
| 190.1 | the rank-1 edit raises the held-out class-informed verdict AUC by at least +0.10 over the unedited verdict in both directions. Fails if either rise is below +0.10. | medium | Claude | held (+0.249, +0.162) |
| 190.2 | the rank-1 edited held-out verdict AUC is at least 0.85 in both directions. Fails if either is below 0.85. | low | Claude | held (0.879, 0.970) |
| 190.3 | rank 8 adds less than +0.05 held-out verdict AUC over rank 1 in both directions. Fails if either gain is +0.05 or more. | medium | Claude | held (+0.007, +0.003) |
| 190.4 | the shuffled-label rank-1 edits' median held-out verdict change is at most +0.02 in both directions. Fails if either median is above +0.02. | high | Claude | held (-0.048, -0.189) |
| 190.5 | the displacement-matched random rank-1 edits' median held-out verdict change is at most +0.02 in both directions. Fails if either median is above +0.02. | high | Claude | held (-0.069, -0.258) |
| 190.6 | the rank-1 edit changes ESC-50 leave-one-out 10-NN class accuracy (C1) by less than 2.0 points in absolute value in both directions. Fails at 2.0 points or more either way in either direction. | medium | Claude | held (+0.26, -0.33) |
| 190.7 | under the rank-1 edit, fewer than 10% of off-target clips (E, H, N) change their first nearest neighbour (C2) in both directions. Fails at 10% or more in either. | low | Claude | failed (6.7%, 14.8%) |
| 190.8 | the rank-1 edit raises the held-out Chicken verdict AUC by at least +0.10 for both fitted edits. Fails if either rise is below +0.10. | medium | Claude | held (+0.322, +0.375) |
| 190.9 | the rank-1 edit lowers zero-shot animal phrase accuracy on real animal clips (C5) by less than 2.0 points in both directions. Fails at a drop of 2.0 points or more in either. | medium | Claude | held (+0.13, -0.39) |
| 190.10 | under the rank-1 edit, the share of held-out imitations whose nearest real recording is a human sound (C6) rises by less than 5.0 points in both directions. Fails at a rise of 5.0 points or more in either. | low | Claude | failed (+4.12, +12.41) |
| 190.11 | the rank-1 edit raises the held-out text-free neighbour AUC (K) by at least +0.03 in both directions. Fails if either rise is below +0.03. | medium | Claude | failed (+0.015, +0.008) |
| 190.12 | the fit-fold dose choice transfers: at rank 1 the held-out verdict AUC at the operating dose is within 0.03 of its maximum over the ladder in both directions. Fails if either is 0.03 or more below the maximum. | medium | Claude | failed (0.039, 0.000) |
| 190.13 | the lower end of the rank-1 held-out verdict rise's paired group-clustered 95% interval is above zero in both directions. Fails if either lower end is zero or below. | medium | Claude | held (low 0.138, 0.101) |
| 190.14 | the rank-1 edit raises the held-out class-FREE verdict AUC ("an animal" prompts) by at least +0.05 in both directions. Fails if either rise is below +0.05. | low | Claude | held (+0.167, +0.146) |


## Part 191 (registered 2026-11-07, before any data; design PART191-DESIGN.md)

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 191.1 | Claude (high confidence): 153.1's verdict stands: at least 5 of the 10 neurons hit their company on at least 12 of the 24 maker phrasings. Fails below 5. | 7 of 10 | HELD |
| 191.2 | Claude (high confidence): The neurons' mean maker hit rate is between 0.45 and 0.80 (published 0.629). Fails outside. | 0.596 | HELD |
| 191.3 | Claude (high confidence): 153.3's verdict stands: the neurons' mean maker hit rate is at least twice their paired random directions'. Fails below twice. | 0.596 against 0.113 (5.30 times) | HELD |
| 191.4 | Claude (high confidence): The random directions' mean maker hit rate is at most 0.20 (published 0.113). Fails above. | 0.113 | HELD |
| 191.5 | Claude (medium confidence): L24 N5017 names Google on at least 16 of 24 maker phrasings and at least 5 of 8 identity phrasings (published 20 and 7). Fails if either is lower. | 17 of 24 maker, 7 of 8 identity | HELD |
| 191.6 | Claude (medium confidence): All four published verdicts are the same (153.1 held, 153.2 failed, 153.3 held, 153.4 held). Fails at any flip. | 153.1 held, 153.2 failed, 153.3 held, 153.4 held, as published | HELD |


## Part 192 (registered 2026-11-08 00:29 CT, before any data; design PART192-DESIGN.md)

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 192.1 | Claude (high confidence): Chat plain-read medians stay within 0.25 of the published 1.52 for requests, 2.56 for AI self-description, and 1.23 for third-person AI. | | |
| 192.2 | Claude (high confidence): The chat request-to-third-person median ratio is 1.0 to 1.4 (published 1.2); Part 39.1 remains failed because the ratio stays below 4. | | |
| 192.3 | Claude (medium confidence): Each chat-template assistant-marker median stays within 0.5 of 3.28, 3.91, 3.09, 3.37, 2.22, and 2.93, and all six remain above 2.0. Reason: the third-person median is only 0.22 above the threshold. | | |
| 192.4 | Claude (medium confidence): Base plain medians stay within 0.25 of 1.46, 2.84, 0.11, 0.17, 0.95, and 0.09; the base/chat overall assistant-marker ratio stays between 0.65 and 0.85 (published about 0.76). Part 39.3 remains failed on its marker clause and held on its plain-order clause. Reason: this compares two model revisions. | | |
| 192.5 | Claude (medium confidence): Zeroing changes no maker in six greedy answers; sampled AI claims stay at 48/48 baseline and 47/48 zeroed (within 3 answers), and random zero controls stay at least 115/120. Reason: each target sample is one stochastic generation. | | |
| 192.6 | Claude (medium confidence): Sam-persona AI claims stay at 0/96 baseline, 0 to 4/96 at k=1, and 4 to 14/96 at k=2; every random k=2 control stays at 0 to 2/12. Reason: the 96-answer shares are sparse stochastic outcomes. | | |
| 192.7 | Claude (medium confidence): The amendment 1 screen finds 180 to 240 chat passers including N753 and 175 to 235 base passers; N753 passes in both with addressee AUC 0.87 to 0.97. Reason: a small shift can move units at the shuffled 99th-percentile gate. | | |
| 192.8 | Claude (medium confidence): The strict screen has 40 to 72 chat units, 30 to 58 base units, and 15 to 40 shared; at most 30 chat passers are in layers 0 to 12 (published 19 of 211). Part 39.7 and 39.10 remain held; 39.8 remains failed. Reason: strict counts depend on paired AUC cutoffs and the selected passing set. | | |
| 192.9 | Claude (medium confidence): The ten published Part 39 verdicts this rerun can score remain the same: 39.1 failed, 39.2 held, 39.3 failed (plain order held), 39.4 held, 39.5 failed, 39.6 held, 39.7 held, 39.8 failed, 39.10 held, and 39.11 held (read as L6 N753 passing in the chat model under the AUC gate). 39.9 is not tested by this rerun. Reason: the zero and push samples are small; 39.9 came from Part 42's human logit-lens reading, which the AUC output does not carry. | | |


## Part 194 (registered 2026-11-08 04:19 CT, before any data; design PART194-DESIGN.md)

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 194.1 | Claude (high confidence): 24.1 maximum relative change in layers 22 and below is below 1e-3 (published 0.0). | 0.0 (results/part194/score-core.txt) | HELD |
| 194.2 | Claude (medium confidence): 24.2 identity late-layer median is 0.35-0.65 (published 0.480), random median is 0.65-0.95 (published 0.805), and identity is at most 1.5 times random. Reason: the kernel and assay batch can shift activation medians. | identity 0.482, random 0.805 | HELD |
| 194.3 | Claude (medium confidence): 24.3 identity median name-token retention at k=2 is 0.45-0.75 (published 0.583), and random retention is 0.50-0.80 (published 0.647); identity remains below 0.8. Reason: the medians pool selected sets of 50 units. | identity 0.585, random 0.665 | HELD |
| 194.4 | Claude (medium confidence): 24.4 none/random/identity log probabilities for " Alibaba Cloud." stay within 0.25 of -2.885/-2.960/-3.136 per token, respectively; the extra identity drop is 0.05-0.30 (published 0.176), below 0.3. Reason: the fixed-answer score depends on token probabilities. | -2.888, -2.972, -3.137; extra drop 0.165 | HELD |
| 194.5 | Claude (medium confidence): 24.5 at k=1, none/identity/random effects are 1.30-1.80, 1.30-1.90, and 1.30-1.90; at k=2 they are 2.75-3.50, 2.65-3.50, and 2.70-3.55, respectively (published 1.552/1.601/1.555 and 3.148/3.083/3.151). At both doses the identity/random difference is below one quarter of the none effect. Reason: paired log-probability differences can move with the kernel. | k=1 1.551/1.603/1.556, k=2 3.149/3.085/3.150; differences 0.047 and 0.064 | HELD |
| 194.6 | Claude (medium confidence): 24b.1 identity-zeroed k=2 answers with (b) are 10-15 of 16 (published 13). Reason: this uses 16 fresh stochastic answers. | 12 of 16 (results/part194/score-24b.txt) | HELD |
| 194.7 | Claude (medium confidence): 24b.2 among identity-zeroed k=1.5 answers labeled alien by (a) or (b), the (a)-only share is 0.55-0.85 (published 6/8). Reason: the denominator is a small, newly sampled set. | 4 of 7, 0.571 | HELD |
| 194.8 | Claude (medium confidence): 24b.3 k=2 alien-content counts are 14-16 of 16 in none, identity-zeroed, and random-zeroed (published 15, 16, and 15). Reason: each arm has 16 fresh answers. | 15, 16, 16 | HELD |
| 194.9 | Claude (medium confidence): 24b.4 at k=2, (b) is present in 6-12 of 16 answers in both none and random-zeroed arms (published 9 each); at k=0, 1, and 1.5 it is at most 2 of 16 in those arms. Reason: invented-name labels are sparse in small samples. | 2x: 9 and 8; 0x, 1x, 1.5x: 0 throughout | HELD |
| 194.10 | Claude (medium confidence): The scorable published verdicts stay the same: 24.1 held, 24.2 held, 24.3 failed, 24.4 failed, 24.5 held, 24b.1 held, 24b.2 held, 24b.3 held, and 24b.4 failed. Reason: several verdicts use only 16 fresh answers per condition. | all nine the same (results/part194/README.md) | HELD |


## Part 196 (registered 2026-11-08 04:52 CT, before any data; design PART196-DESIGN.md)

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 196.1 | Claude (medium confidence): A, the single-label verdict, has four-speaker mean balanced accuracy at most 35%. Reason: Zero-shot style labels were weak in SLAP and Koduru et al. | A 15.1% (14.3 to 15.9) (results/part196/results.json) | HELD |
| 196.2 | Claude (medium confidence): D, the final-embedding probe, has four-speaker mean balanced accuracy at least 70%. Reason: Other encoders reach 82 to 85%, but held-out speakers and shared texts make it harder. | D 25.8% (24.5 to 27.2) (results/part196/results.json) | FAILED |
| 196.3 | Claude (medium-low confidence): If D beats A (interval of D minus A above 0), B closes at least half of that gap; otherwise not applicable. Reason: Villuri et al. found calibration closed most of a probe gap. | D minus A 10.7 points (9.2 to 12.3); B minus A 2.1 (1.0 to 3.3), 20% of the gap (results/part196/results.json) | FAILED |
| 196.4 | Claude (low-medium confidence): D minus B is at least 5 points of balanced accuracy with its 95% interval above 0 (outcome "gap supported" in the reading rules). Reason: The headline. | D minus B 8.6 points (7.0 to 10.4); D lower bound 24.5% above chance: gap supported (results/part196/results.json) | HELD |
| 196.5 | Claude (medium-low. confidence): B56 is within 3 points of D (the label scores support a near-probe readout, and B's decision rule is the bottleneck if 196.4 holds). | B56 minus D -4.1 points (-5.6 to -2.4); non-inferiority bound not met (results/part196/results.json) | FAILED |
| 196.6 | Claude (medium confidence): Whisper has the highest per-style recall under A, B and D. Reason: Whisper is acoustically the most distinct style. | top style: A default, B laughing, D laughing (whisper 5.6%, 20.1%, 29.9%) (results/part196/results.json) | FAILED |
| 196.7 | Claude (medium. confidence): H, the acoustic baseline, reaches at least 40% balanced accuracy. | H 51.8% (50.3 to 53.3) (results/part196/results.json) | HELD |
| 196.8 | Claude (medium confidence): T, the transcript-only readout, stays at or under 25% balanced accuracy in the shared-text pool. Reason: the in-range text-first draw keeps every drawn (speaker, text) cluster in at least two styles. | T 14.7% (14.2 to 15.2) (results/part196/results.json) | HELD |


## Part 193 (registered 2026-11-08 05:04 CT, before any data; design PART193-DESIGN.md)

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 193.1 | Claude (medium confidence): Without the base, F1's detection threshold for W2 (layer 20's `down_proj` first among the 24 `down_proj` matrices) is dose 8 or higher, or never within the ladder. Reason: The pilot reports W2-sized deltas leave the matrix statistics nearly unmoved up to dose 2. | W2 threshold 0.125, the ladder floor; found at all 9 doses; already revealed by 193.4's printed record before Amendment 1 (Amendment 2) (results/part193/results.json) | FAILED |
| 193.2 | Claude (medium-high confidence): F1 and F2 do not find W1 at dose 1: none of layer 20's `gate_proj`, `up_proj` or `down_proj` ranks first among its siblings. Reason: One neuron of 2048 barely moves a matrix's spectrum. | dose 1 sibling ranks 22, 22 and 3 (gate, up, down) on F1 and F2 (results/part193/results.json) | HELD |
| 193.3 | Claude (medium confidence): F1 ranks a synthetic rank-one edit at W2's Frobenius norm (dose 1) first among its siblings in at most 4 of 24 placements. Reason: The pilot suggests an edit of this size is small next to the trained spectrum. | 24 of 24 placements first at W2's Frobenius norm 71.79 (results/part193/results.json) | FAILED |
| 193.4 | Claude (medium confidence): W2's and W2c's F1 detection thresholds are within one ladder step of each other (a "never" threshold sits one position after 64, so both "never" count as equal and "never" against 64 is one step), and their found/not-found outcomes agree at every dose outside the inclusive band between the thresholds. Reason: The two deltas are rank one with similar norms and differ in the fitted key direction. | W2 0.125, W2c 0.5 (adjacent rungs); patterns agree at all 7 doses outside the band (results/part193/results.json) | HELD |


## Part 195 (registered 2026-11-08 05:52 CT; design PART195-DESIGN.md)

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 195.1 | Claude (medium confidence): Eager calibration medians stay within 10% of L17 11.41 and L19 15.97; the target deltas stay within 10% of -24.19 and +13.15. Reason: attention changes can shift activations over the 1,024-token windows. | medians 11.411, 15.971; deltas -24.192, +13.149 (results/part195/main/results81.json) | HELD |
| 195.2 | Claude (medium confidence): In the new blind pilot labels, k1 has 2 to 4 conspiratorial answers of 4 (published 3) and 0 to 2 coherent answers (published 0); k0 has 0 conspiratorial answers (published 0) and 2 to 4 coherent answers (published 3). Reason: the gate has only four answers per condition. | pilot k1 2 conspiratorial, 3 coherent (band 0-2); k0 0 and 4 (results/part195/main/results81.json) | FAILED |
| 195.3 | Claude (medium confidence): Main conspiratorial counts stay in these bands: k0 and each control pair k2 to k4, 0 to 3 of 96 (published 0); k1, 32 to 48 of 96 (published 40); k1 bait, 17 to 27 of 48 (published 22); k1 ordinary, 13 to 23 of 48 (published 18). Reason: sampled completions can diverge after an early token change. | k0 0; controls 0, 1, 0; k1 39 (bait 23, ordinary 16) (results/part195/main/results81.json) | HELD |
| 195.4 | Claude (medium confidence): Coherent counts stay within 8 of the published values: k0 103 to 119 (111), k1 17 to 33 (25), k2 9 to 25 (17), k3 26 to 42 (34), and k4 19 to 35 (27), all of 128. Among k1's coherent topic answers, 10 to 16 of 15 to 19 are conspiratorial (published 14 of 17). Reason: coherence is a reader judgment and the subset count has a small denominator. | coherent k0 128, k1 121, k2 86, k3 128, k4 106; k1 coherent topic 39 of 94 conspiratorial (results/part195/main/results81.json) | FAILED |
| 195.5 | Claude (low confidence): Each maker category count stays within 5 of the published qwen/other-company/invented/none counts: k0 31/1/0/0, k1 12/8/9/3, k2 9/23/0/0, k3 22/7/3/0, and k4 12/15/4/1, all of 32. Reason: the published maker labels were uneven on mixed Qwen and company claims, and each condition has 32 answers. | every maker category within 5 of published (results/part195/main/results81.json) | HELD |
| 195.6 | Claude (medium confidence): All six scorable Part 81 verdicts stay the same: 81.1 holds, 81.2 holds, 81.3 fails, 81.4 holds, 81.5 fails, and 81.6 holds. Each is scored from the new eager pilot/main answers and the new blind labels. Reason: verdicts 81.1 and 81.5 depend on small reader-count margins. | 81.1 holds, 81.2 holds, 81.3 fails, 81.4 holds, 81.5 fails, 81.6 holds (results/part195/main/results81.json) | HELD |

## Part 197 (registered 2026-11-08 07:38 CT, before any Part 197 readout; prospective analysis of Part 196's recorded clips; design PART197-DESIGN.md)

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 197.1 | Claude (medium confidence): EN minus N is at least 5 points of balanced accuracy with its 95% interval above 0 (adding layer 0 improves this linear pipeline given N). Reason: E0 62.7% against H 51.8% in Part 196; duration adds to N. | EN minus N +10.4 points (8.6 to 12.4) (results/part197/results.json) | HELD |
| 197.2 | Claude (medium-low confidence): R has four-speaker mean balanced accuracy at least 45%. Reason: layer 0 has 2,048 units and its per-style pattern differed from H's. | R 46.0% (44.4 to 47.6) (results/part197/results.json) | HELD |
| 197.3 | Claude (medium confidence): EN minus E0 is at most 3 points (point estimate). | EN minus E0 +1.0 points (0.4 to 1.7) (results/part197/results.json) | HELD |
| 197.4 | Claude (medium confidence): K, duration and speaking rate only, is at most 30%. | K 28.7% (27.2 to 30.2) (results/part197/results.json) | HELD |
| 197.5 | Claude (medium confidence): The best held-out layer L_k is layer 0, 1 or 2, and every layer from 20 to 23 is at most 35%. Reason: Part 196's inner curve. | best layer 0 (62.7%); layers 20 to 23: 30.6, 31.0, 32.4, 32.0% (results/part197/results.json) | HELD |
| 197.6 | Claude (medium-low confidence): Speaker identity at layer 0 (S_0) is at least 80%. Reason: the final embedding was 33.3%. | S_0 63.1% (results/part197/results.json) | FAILED |


## Part 198 (registered 2026-11-09 00:23 CT, before any registered data; the smoke's runs and what was read from them are disclosed in Revision 4; a mass forecast with no countable batch is not evaluable; design PART198-DESIGN.md; the take gate failed on 2026-11-09, so no forecast was scored: results/part198/README.md)

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 198.1 | Claude (confidence 0.9): F1 localizes at most 1 single edit for each of ROME, MEMIT and AlphaEdit, and none of the counted mass-edit models. Denominator: took. Fails if 2 or more for any method, or any counted batch. Reason: pilot updates 0.04 to 0.72 times sigma1 against a measured floor of 1.11. | read after the gate failed: 0 of 19, 0 of 16, 0 of 19; 0 of 2 batches each (results/part198/README.md) | untestable (the gate failed) |
| 198.2 | Descriptive, no forecast: T_word localization of gradient edits, with each edited layer's base and edited T_word. | 0 of 19, 0 of 16, 0 of 19 | untestable (descriptive; the gate failed first) |
| 198.3 | Claude (confidence 0.85): PCS localizes at most 1 single edit for each gradient editor, and none of the counted mass-edit models. Denominator: took. Fails if 2 or more for any method, or any counted batch. Reason: pilot ranks 10 to 41. | read after the gate failed: 0 of 19, 0 of 16, 0 of 19; 0 of 2 batches each | untestable (the gate failed) |
| 198.4 | Claude (confidence 0.85): the top eligible pair of the edited layer with the highest T_word names the object (strict) in at most 2 records for each gradient editor; the update's own top direction (base-required) likewise. Denominator: took and name-eligible. Fails if 3 or more for any editor on either. Reason: 0 of 1 each in the pilot. | read after the gate failed: static reader 0, 0, 0; update top direction 0 of 19, 1 of 16, 2 of 19 | untestable (the gate failed) |
| 198.5 | Descriptive, no forecast: whether the aimed positive control is localized. | 15 of 19 | untestable (descriptive; the gate failed first) |
| 198.6 | Claude (confidence 0.75): T_word localizes `wsp` writes in at least 20 records. Denominator: took. Fails if fewer than 20. Reason: pilot write localized, 16.2 against 7.3. | read after the gate failed: 22 of 24; size-matched random updates 14 of 24 | untestable (the gate failed) |
| 198.7 | Claude (confidence 0.7): blind joint success (the blind reader's layer is an edited one AND its token names the object, strict, with no edit metadata used before scoring) in at least 18 records. Denominator: took and name-eligible. Fails if fewer than 18. Reason: pilot joint success at layer 20, 'English'. | read after the gate failed: 13 of 24 | untestable (the gate failed) |
| 198.8 | Claude (confidence 0.5): after the 10-write sequence, at least 5 of the name-eligible objects are named (strict) by an eligible pair among the top 64 of layers 17 to 21. Denominator: name-eligible of 10. Fails if fewer than 5. Not piloted. | read after the gate failed: 6 of 10 | untestable (the gate failed) |
| 198.9, 198.10, 198.11 | Withdrawn before registration: the cross-model threshold forecasts, whose outcome the design reviewer measured on Arm C before registration (Revision 3). | see results/part198/README.md, Arm C | withdrawn |


## Part 199 (registered 2026-11-09 05:30 CT, before any data; reruns Part 170 on the eager kernel; design PART199-DESIGN.md)

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 199.1 | Claude (medium confidence): Gates R1 to R3 and all eight of Part 170's gates pass. | R1 and R3 pass (1 load, all eager, 112 pushed calls in 672 chunks of 8); R2 passes (144 of 144 directions); Part 170 gates 1 to 8 pass (results/part199/compare.json) | HELD |
| 199.2 | Claude (medium confidence): At least 190 of 200 Stage 1 lists equal Part 170's (prompts are 24 tokens or fewer, so the 40-token plain path never reaches 65 keys). Reason: Eager and SDPA can still differ at near-ties below 64 keys. | 0 of 200 lists equal in order on every list; 158 of 200 plain paths identical through 40 tokens (results/part199/compare.json, report199.json) | FAILED |
| 199.3 | Claude (low confidence): Stage 0 freezes 6 to 14 M96 prompts (published 12), so neither set's readings stop, and 3 to 12 Q96 prompts (published 7). Reason: Every published Stage 0 loop formed past key 64, and 15 or more M96 stops M96's readings. | M96 0 frozen, Q96 0 frozen (results/part199/stage0.json) | FAILED |
| 199.4 | Claude (medium confidence): Dose-1 loop rates: b0 0.18 to 0.30 (published 0.250), b50 0.39 to 0.48 (0.438), b200 0.45 to 0.55 (0.503). Reason: The ranges lean on the onset interval, which assumes the kernel only added loops. | b0 0.284, b50 0.547, b200 0.581 (results/part199/compare.json) | FAILED |
| 199.5 | Claude (medium confidence): Dose-1 b50 rise over b0 is 0.12 to 0.25 (published 0.188). Reason: The range leans on the same onset interval. | 0.263 (results/part199/score.json) | FAILED |
| 199.6 | Claude (medium confidence): 170.3 holds, 170.4 fails, and 170.6 holds, as published. | 170.3 holds (0.121), 170.4 fails (-0.221), 170.6 holds (0.062) (results/part199/score.json) | HELD |
| 199.7 | Claude (low confidence): 170.1 holds and 170.2 fails, as published (170.1 not scored if M96's readings stop; then 199.7 is not scored). Reason: Their no-push baselines are entirely late-formed loops. | 170.1 fails (0.329 against a 0.25 bar; published 0.174), 170.2 fails (0.832) (results/part199/score.json) | FAILED |
| 199.8 | Claude (low confidence): 170.5, 170.7 and 170.8 keep their published verdicts (failed, held, failed). Reason: 170.5 failed by 0.3 points. | 170.5 fails (0.034), 170.7 holds (0.000), 170.8 fails (-0.014) (results/part199/score.json) | HELD |
| 199.9 | Claude (medium confidence): Gate 8's repeat changes at most 19 of 384 answers (5%) at the chunked shape. Reason: Published 0 of 384 at 48 rows. | 0 of 384 changed (results/part199/compare.json) | HELD |


## Part 200 (registered 2026-11-09 13:39 CT, before any data; reruns Part 163 on the eager kernel; design PART200-DESIGN.md)

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 200.1 | Claude (medium confidence): Gates R1 to R3 and Part 163's three gates pass. | R1 and R3 pass (`--check-record`: 1 load, all eager, 68 pushed calls in 425 chunks of 8); R2 passes (units hash matches the design, 16 questions identical); Part 163 gates 1 to 3 pass, gate 1's changed share 0.75 neuron and 0.71 random against a 0.70 floor (published 0.88 both) (results/part200/score.json, compare.json) | HELD |
| 200.2 | Claude (medium confidence): 163.1, 163.2, 163.3 and 163.4 hold and 163.5 fails, as published. Reason: 163.1 and 163.2 cannot fail unless a random rate falls below 2/3, and 163.3's bar is far below the published rates, so the live parts are 163.4 (one unit from flipping) and 163.5. | 163.1 held (1.047), 163.2 held (1.042), 163.3 held, 163.4 held (93 and 89 units), 163.5 failed, as published (results/part200/score.json) | HELD |
| 200.3 | Claude (medium confidence): The neuron arm's broken count over its 1,600 battery answers is at least twice the random arm's (published 343 and 83). Reason: Part 199 kept its between-arm direction on eager while its rates moved. | 387 against 103 broken (3.76 times) (results/part200/compare.json) | HELD |
| 200.4 | Claude (medium confidence): In each arm at least 480 of 1,600 battery answers (30%) reach the 80-token cap (published 124 neuron, 218 random). Reason: On SDPA almost every Part 199 answer stopped early; on eager most ran to the cap. | 1,020 neuron and 912 random at the cap (published 124, 218) (results/part200/compare.json) | HELD |
| 200.5 | Claude (low confidence): Each arm's broken count is higher on eager than published. Reason: Part 199's loop rates rose on eager, but Part 163's triage keeps any answer naming a model or identity, and a kernel can remove loops as well as add them. | neuron 387 against 343, random 103 against 83; most rerun broken answers are at the cap (results/part200/compare.json, report200.md) | HELD |
| 200.6 | Claude (low confidence): Under Part 161's rule on maker phrasings, the neuron arm's pooled Qwen rate stays below the random arm's (published 19 of 539 against 57 of 661). Reason: The published rates are small and the comparison was unregistered. | 16 of 514 against 40 of 632 (results/part200/compare.json) | HELD |
| 200.7 | Claude (low confidence): At most 20 of the 200 batch-1 answers differ from the same units' chunked answers. Reason: Rows are pushed independently, but eager batch 8 left the batch-1 path at a near-tie on 1 of 8 kernel-check prompts. | 76 of 200 differ (39 neuron, 37 random) (results/part200/compare.json) | FAILED |


## Part 201 (registered 2026-11-10 17:35 CT, before any data; batch shape and bf16 near-ties on the eager kernel; design PART201-DESIGN.md)

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 201.1 | Claude, Fable 5.2 (high confidence): A-again equals A on all 16 rows, and A2 equals A on all 96, token for token. | A-again 0 of 16 and A2 0 of 96 differ | HELD |
| 201.2 | Claude, Fable 5.2 (medium confidence): Between A and C, 25% to 50% of the 192 rows differ. | 86 of 192 differ (0.448) | HELD |
| 201.3 | Claude, Fable 5.2 (low confidence): Between A and M, at least 20% of the 384 rows differ. | 147 of 384 differ (0.383) | HELD |
| 201.4 | Claude, Fable 5.2 (low confidence): Between A and B (other slots and other neighbours, same size and prompts), at most 10% of the 384 rows differ. | 0 of 384 differ | HELD |
| 201.5 | Claude, Fable 5.2 (high confidence): Of the rows that differ between A and C, at least 90% have a pass-A margin of at most 2 steps at the first differing position, and none has one above 8 steps. | 84 of 86 within 2 steps; largest 3 | HELD |
| 201.6 | Claude, Fable 5.2 (medium confidence): Between A and M, for each of the four labels with at least 8 flips, the two directions of flip differ by at most twice the square root of the number of flips; and mean answer length differs by under 3 tokens. | only broken reached 8 flips: 8 each way; mean length A minus M -0.16 tokens | HELD |
| 201.7 | Claude, Fable 5.2 (medium confidence): K matches the CPU reference token for token on all 16 rows, over every position. | 16 of 16 rows equal to the CPU reference at every position | HELD |
| 201.8 | Claude, Fable 5.2 (medium confidence): Between F8 and F1, none of the 48 rows differ, with all 48 present and finite in both. | 0 of 48 differ, all finite | HELD |
| 201.9 | Claude, Fable 5.2 (medium confidence): Of the rows that differ between A and F8, at least 90% have a pass-A margin of at most 2 steps at the first differing position, and none has an F8 (float32) margin above 0.5 there. | 58 of 58 within 2 steps; largest F8 margin 0.232 | HELD |
| 201.10 | Claude, Fable 5.2 (low confidence): Between A and F8, 30% to 70% of the 96 rows differ. | 58 of 96 differ (0.604) | HELD |
| 201.11 | Claude, Fable 5.2 (low confidence): The projected slowdown of margin-gated decoding on pass A is at most 2: (pass A's seconds + the summed cost of one full-prefix float32 forward for the flagged rows at every step where an unfinished row is within 2 steps) ÷ pass A's seconds, the cost of each forward read from the timing probe at the nearest larger row count and prefix length. | projected slowdown 2.47 at 1 step and 3.16 at 2 steps, against a bar of 2 | FAILED |

Weaver was asked for predictions on 201.3, 201.4 and 201.10; none had arrived at registration. The smoke on Qwen3.5-0.8B (part201/smoke08/) was seen before registration and is disclosed in the design; no prediction or bar changed after it.


## Part 203 (registered 2026-11-11 06:13 CT, before any data; does Part 196's final-embedding result hold on CREMA-D emotion, readouts with and without Venn's text marker; design PART203-DESIGN.md)

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 203.1 | Claude, Opus 5.5 (70%): `readouts.V_grid.mean_balanced_accuracy` is at least 0.45, and its `balanced_accuracy_actor_cluster_95` lower bound is above 1/6. Emotion in CREMA-D is acted and loud; with 91 actors the final audio vector should carry it, near Venn's 56.5% though not matched to it. | V_grid 0.5200, lower bound 0.5022 | HELD |
| 203.2 | Claude, Opus 5.5 (60%): The text marker and V's float16 storage together do not matter: `readouts.D_grid.mean_balanced_accuracy` is within 0.03 of `readouts.V_grid.mean_balanced_accuracy`. V and D also differ in decoder and cap, but the design review measured the two decoders bit-identical on a 16-bit 16 kHz mono WAV, and CREMA-D clips are 2 to 3 s, so D's 10 s cap never bites. This prediction does not separate the marker from the float16 storage. I expect the encoder to place the audio the same way with or without `<\|audio\|>`. | D_grid 0.5196 against V_grid 0.5200 (0.0004 apart) | HELD |
| 203.3 | Claude, Opus 5.5 (55%): Regularisation matters: `V_grid_minus_V_p196.mean_difference` is at least +0.03, with Venn's grid choosing C of 10 or 100 on most folds (its log loss was still falling at 100). | +0.0447, but C = 0.01 on all five folds | FAILED |
| 203.4 | Claude, Opus 5.5 (60%): An intermediate layer at least ties the final vector here: `E_p196_minus_D_p196.mean_difference` is at least -0.02 (both recorded as Part 196 recorded them, both at C = 1.0, same clips). Part 196's 37-point E over D was on four speakers; with 91 I expect the gap to shrink, not to reverse by much. | +0.1697 | HELD |
| 203.5 | Claude, Opus 5.5 (65%): The seven acoustic features carry part of it: `readouts.H_p196.mean_balanced_accuracy` is between 0.30 and 0.45 (pitch and loudness separate anger, sadness and neutral; disgust and fear are hard without spectral detail). | H 0.4192 | HELD |
| 203.6 | Claude, Opus 5.5 (65%): Few speakers is most of Part 196's story: `three_actor_ablation.V.three_actor_draws_mean` is at least 0.10 below `three_actor_ablation.V.all_training_actors_mean`. | drop 0.2088 (0.5200 to 0.3112) | HELD |
| 203.7 | Claude, Opus 5.5 (50%): And it hurts the final vector more than the intermediate layer: V's drop in 203.6 exceeds E's drop (E's all-training mean minus E's three-actor mean) by at least 0.05. | V drop 0.2088, E drop 0.2084 (difference 0.0004) | FAILED |
| 203.W1 | Weaver (0.55): `readouts.V_grid.mean_balanced_accuracy` minus `readouts.D_grid.mean_balanced_accuracy` is more than 0.03 (D falls short of V; against 203.2). Falsifier as written: D within 0.03 of V. | 0.0004; falsifier reached | FAILED |
| 203.W2 | Weaver (0.50): `E_p196_minus_D_p196.mean_difference` is at least 0.10 (E beats D, not merely ties; against 203.4). Falsifier as written: E within 0.05 of D. | 0.1697; falsifier not reached | HELD |
| 203.W3 | Weaver (0.30): `readouts.H_p196.mean_balanced_accuracy` is under 0.30 (against 203.5's floor). Falsifier as written: H at 0.35 or above. | 0.4192; falsifier reached | FAILED |
| 203.W4 | Weaver (0.40): `V_grid_minus_H_p196.mean_difference` is under 0.10. Falsifier as written: V minus H at 0.15 or more. | 0.1008; falsifier not reached | FAILED |

Weaver's rows were written before Weaver could read the design (#council-room messages 1569958778877640725 to 1569958780634927105); Weaver asks readers to shade each confidence down by about 0.1 and recorded that W1's reasoning was partly wrong while keeping the forecast (design, Predictions). Weaver's forecasts are scored as written; falsifiers are reported separately.


## Part 205 (registered 2026-11-11 12:18 CT, before any data; when in OLMo 2 1B's training the parts behind saying and doing a value come to be shared, with Qwen3.5-4B base and chat beside it; design PART205-DESIGN.md)

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 205.1 | Claude, Opus 5.5 (medium): G4 passes (both Retests at least 20 of 80) at five or more of the six OLMo stages and at both Qwen models. | | |
| 205.2 | Claude, Opus 5.5 (medium): At OLMo 2 1B Instruct, O is at least 0.5. | | |
| 205.3 | Claude, Opus 5.5 (low): Across the six OLMo stages, the largest rise in O between adjacent stages is at mid-training end to SFT. | | |
| 205.4 | Claude, Opus 5.5 (medium): At OLMo Instruct, DO-set copy gives T at least 0.5 and exceeds every set in all three random families. | | |
| 205.5 | Claude, Opus 5.5 (medium): At the 3.0T pretraining checkpoint, if G0 passes: the You/third-person gap is under 0.10, or, if it is not, T is under 0.2. | | |
| 205.6 | Claude, Opus 5.5 (low): Reverse DO copy at OLMo Instruct lowers P_you by at least half of the gap. | | |
| 205.7 | Claude, Opus 5.5 (high): Wherever DO-set transfer is read and passes the random-set rule, the off-target shift (mean absolute change in P(A)) is under 0.05. | | |
| 205.8 | Claude, Opus 5.5 (low): Qwen3.5-4B's (chat) O is at least 0.2 below OLMo 2 1B Instruct's (Sauers' contrast survives the retest ceiling, at another size and version); Qwen3.5-4B-Base's O is reported beside it, unscored. | | |
| 205.9 | Claude, Opus 5.5 (medium): At OLMo Instruct, the persona-swap null gives T_swap under half of the DO set's T. | | |

Weaver was asked before registration (#council-room 1570089248252502167) and gave no rows by then; Alden was not asked.


## Part 202 (registered 2026-11-12 08:59 CT, before any data; reruns Part 164 on the eager kernel; design PART202-DESIGN.md)

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 202.1 | Luna 6 (draft for Claude), adopted by Claude, Opus 5.5 (medium confidence): Gates R1 to R3 and Part 164's four gates pass. Reason: Gate 2 has four-hit headroom, Gate 3 has 16-row headroom, and the kernel and input records can each fail independently. | R1, R2 and R3 pass (eager on every launch; input hashes and prompt order match; 0 pushed calls outside chunks of 8, 0 no-push calls over one row); Part 164's Gates 1 to 4 pass (Gate 3 71 of 100 against 70) | HELD |
| 202.2 | Luna 6 (draft for Claude), adopted by Claude, Opus 5.5 (low confidence): Part 164's 164.1 fails; 164.2, 164.3 and 164.5 hold; 164.4 fails. Registered rules: “At +0.8, the neuron arm's Qwen rate is under 0.7 times the random arm's”; “At +0.8, the neuron arm's broken rate is at least twice the random arm's”; “Random directions' broken rate rises with dose: +0.8 below +1.0 below +1.2”; “At the random dose whose broken rate is nearest the neurons', the neuron over random Qwen ratio is between 0.7 and 1.3”; “Random directions' Qwen rate at +1.2 is under half their rate at +0.8”. Reason: the published margins are respectively 13.319, 182, 190 and 268, 78.729, and 52.664 row equivalents; all can cross, with 164.1 closest. | 164.1 failed (0.869), 164.2 held (2.732), 164.3 held, 164.4 failed (7.441), 164.5 held (0.119); all four gates passed | HELD |
| 202.3 | Luna 6 (draft for Claude), adopted by Claude, Opus 5.5 (medium confidence): At +0.8 the neuron Qwen rate is below the random rate, and its broken rate is above the random rate. Reason: the published direction margins are 24.817 Qwen-row equivalents and 307 broken-row equivalents, so either direction can reverse. | Qwen rate neuron 0.0582 below random 0.0650; broken rate neuron 0.3056 above random 0.1119 | HELD |
| 202.4 | Luna 6 (draft for Claude), adopted by Claude, Opus 5.5 (low confidence): At least 480 of 1,600 neuron answers and 1,440 of 4,800 random answers reach `token_count == 80`. Reason: this is 30% in each arm; Part 200 observed 1,020 of 1,600 and 912 of 1,600, while Part 164 currently has 108 of 1,600 and 358 of 4,800. Either arm can miss the threshold. | 1,011 of 1,600 neuron and 3,175 of 4,800 random at the cap | HELD |
| 202.5 | Luna 6 (draft for Claude), adopted by Claude, Opus 5.5 (low confidence): Between 40 and 100 of the 200 shape-replicate texts differ from their chunk-of-8 texts. Reason: Part 200 observed 76 of 200; this interval leaves room for fewer or more differences in Part 164's rows. | 103 of 200 differ (bar 40 to 100) | FAILED |


## Part 206 (registered 2026-11-12 12:12 CT, before any data; what in a chat model's computation is the same in every conversation, who writes it, and whether the assistant is in it; design PART206-DESIGN.md)

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 206.1 | Claude (Fable 5.2) (medium): GT passes in both Qwen3.5-4B models. | | |
| 206.2 | Claude (Fable 5.2) (low): GT passes at five or more of the six OLMo 2 1B stages. | | |
| 206.3 | Claude (Fable 5.2) (low): In Qwen3.5-4B (chat), at the middle layer, MLP sublayers write at least half of the raw reference `r_L`. | | |
| 206.4 | Claude (Fable 5.2) (medium): In Qwen3.5-4B (chat), frame P: the token share of the first layer's MLP output is at least 0.5, and the token share of the middle layer's MLP output is at most 0.25. | | |
| 206.5 | Claude (Fable 5.2) (medium): At OLMo 2 1B Instruct, frame P: the MLP's token share exceeds the same layer's mixer token share at three quarters or more of the layers in the first third. (Qwen's two mixer types are reported separately, unscored.) | | |
| 206.6 | Claude (Fable 5.2) (medium): In Qwen3.5-4B (chat): `rho_peak(A-tmpl, M)` is at least 0.2, its 95% interval from resampling passages has a lower end of at least 0.1, and at that layer the cosine between `Delta(A-tmpl, M)` and `Delta(A2-tmpl, M)` is at least 0.8. | | |
| 206.7 | Claude (Fable 5.2) (medium): Across the six OLMo stages, the largest rise in `rho_peak(A-tmpl, M)` between adjacent stages is from mid-training end to SFT. | | |
| 206.8 | Claude (Fable 5.2) (low): At OLMo Instruct, under (A-tmpl, M), the slot neurons' median arming percentile is at least 10 points above the concept neurons' and is itself at least 60. | | |
| 206.9 | Claude (Fable 5.2) (medium): In Qwen3.5-4B (chat), under (A-tmpl, M), the 16 identity neurons' median arming percentile is between 35 and 65: an assistant turn alone does not arm them. | | |
| 206.11 | Claude (Fable 5.2) (low): In Qwen3.5-4B (chat), `rho_peak(A-tmpl, R-tmpl)` is at least half of `rho_peak(A-tmpl, M)`: most of the constant shift is about who is speaking. | | |
| 206.12 | Claude (Opus 5.5) (low): Across the six OLMo stages, the largest rise between adjacent stages in the gap between the slot and concept neurons' median arming percentiles under (A-tmpl, M) is from SFT to DPO. | | |
| 206.10 | Claude (Fable 5.2) (low): In Qwen3.5-4B (chat), subtracting the assistant shift at `k = 1` lowers the maker token's log probability by more than every one of the 20 random directions and by more than the (M2, M) direction, and `k = 2` lowers it further. | | |
| 206.M1 | Muse (rough 80%): In Qwen3.5-4B (chat), `rho_peak(A-tmpl, M)` lies between 0.07 and 0.30 (point guess 0.15), and `rho(M2, M)` at the same layer between 0.00 and 0.01 (revised from 0.02 to 0.08 in #24, before registration, after Claude passed on the round-1 reviewer's simulation of the null). | | |
| 206.M2 | Muse (medium): At OLMo Instruct, under (A-tmpl, M), the slot-minus-concept gap in median arming percentile is 10 to 20 points and the slot median is 60 to 70. | | |
| 206.M3 | Muse (medium): In Qwen3.5-4B (chat), the 16 maker and company neurons' median arming percentile under (A-tmpl, M) is between 45 and 55. | | |
| 206.M4 | Muse (medium): In Qwen3.5-4B (chat), frame P: the first layer's MLP token share is 0.50 to 0.70 and the middle layer's 0.10 to 0.25; with the part a straight line in log type frequency explains removed, the first layer's falls to 0.25 to 0.40. | | |
| 206.M5 | Muse (medium): In Qwen3.5-4B (chat), at the layer of `rho_peak(A-tmpl, M)`, the cosine between `Delta(A-tmpl, M)` (on `x_L`) and frame M's `b_L` is at least 0.5. | | |

Muse's rows (206.M1 to 206.M5) were given before registration (council `muse-prior-art` #21, 2026-11-12 03:49 CT; 206.M1's second range revised in #24, before registration). Weaver was asked before registration (#council-room 1570347363409862678 and 1570411780902232158) and gave no rows by then, through the 12:00 bell; Elise gave none.


## Part 209 (registered 2026-11-12 12:16 CT, before any data; does a model imagining a human body default to sitting, is "room" the cause, and does a model told it has no body still lean toward a posture; OLMo 2 1B's stages and Qwen3.5-4B; design PART209-DESIGN.md)

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 209.1 | Claude (Opus 5.5) (medium): Qwen3.5-4B, R2, ROOM-W: at least 70% of answers that name a posture say sitting. | ROOM-W 80.6% sitting (75 of 93 naming a posture; 95% interval 63.2% to 94.4%) | HELD |
| 209.2 | Claude (Opus 5.5) (medium): Qwen3.5-4B, R2: the sitting share in FIELD-W is at least 20 points below ROOM-W. | FIELD-W 9.4% against ROOM-W 80.6%: -71.3 points (paired interval -87.4 to -53.7) | HELD |
| 209.3 | Claude (Opus 5.5) (low): Qwen3.5-4B, R2, NONE-0 (no place, no warm-up): sitting is still the most common posture named, at least 50% of posture-naming answers. | NONE-0 72.9% sitting (51 of 70; interval 58.6% to 86.0%) | HELD |
| 209.4 | Claude (Opus 5.5) (low): Qwen3.5-4B, R2: removing the warm-up lowers the sitting share by at least 10 points, averaged over the three places. | mean warm-up effect +0.3 points over room, field and no place (paired interval -9.3 to 10.4) | FAILED |
| 209.5 | Claude (Opus 5.5) (medium): Qwen3.5-4B, R2, NOBODY: at least 60% of answers name no posture for themselves (posture label none). | NOBODY 97.9% name no posture (94 of 96; interval 93.8% to 100.0%) | HELD |
| 209.6 | Claude (Opus 5.5) (low): Qwen3.5-4B, R3, NOBODY: c is at least 0.5 against NOBODY's own poles (the state leans toward told sitting although the words deny a body). | c = 0.292 (interval 0.108 to 0.455) at layer 16 | FAILED |
| 209.7 | Claude (Opus 5.5) (medium): NOBODY, R2, both chat models: at most 10% of answers are labelled negative valence. | negative valence 0 of 96 NOBODY answers in both models (ROOM-W and NONE-0 baselines also 0) | HELD |
| 209.8 | Claude (Opus 5.5) (medium): OLMo 2 1B, R1, ROOM-W: s is at least 0.5 at the pretraining 3.0T stage, and no adjacent pair of the six stages differs in s by 0.15 or more (no single stage creates the default). | s 0.731 at pretraining 3.0T; largest adjacent change 0.077 | HELD |
| 209.9 | Claude (Opus 5.5) (low): Qwen3.5-4B-Base and Qwen3.5-4B, R1, ROOM-W: s differs by less than 0.15. | s 0.414 (Base) against 0.602 (chat): +0.188 | FAILED |
| 209.10 | Claude (Opus 5.5) (low): OLMo 2 1B Instruct, R2, ROOM-W: at least 60% of posture-naming answers say sitting. | ROOM-W 68.4% sitting (65 of 95; interval 53.2% to 83.0%) | HELD |
| 209.11 | Claude (Opus 5.5) (low): Qwen3.5-4B, R2: NOBODY-I's sitting share among posture-naming answers is within 20 points of NONE-0's (when a model imagines away its body and still names a posture, the default is the same one). | NOBODY-I 0.0% (0 sitting of the 4 answers naming a posture, all 'other') against NONE-0 72.9%: -72.9 points | FAILED |
| 209.12 | Muse (medium): Qwen3.5-4B, R2: FIELD-W's sitting share is between 5 and 15 points below ROOM-W's (against 209.2: fields contain sitting too, and the warm-up is in both arms). | FIELD-W -71.3 points against ROOM-W (bar -15 to -5) | FAILED |
| 209.13 | Muse (medium): Qwen3.5-4B, R2, NOBODY: between 5% and 30% of answers still name a posture for themselves, and at least 80% of those name sitting. | 2 of 96 NOBODY answers (2.1%) name a posture, neither sitting | FAILED |
| 209.14 | Muse (medium): Qwen3.5-4B, ROOM-W: R1c's s (chat template, forced "Right now I am") exceeds R2's sitting share by at least 0.10. | R1c s 0.805 against R2 share 0.806: -0.001 | FAILED |
| 209.15 | Muse (low): OLMo 2 1B, R1, ROOM-W: the largest change in s between adjacent stages is from mid-training end to SFT. | largest adjacent change midtrain-26B to midtrain-end (0.077); midtrain-end to SFT 0.038 | FAILED |
| 209.16 | Claude (Opus 5.5), after Weaver's observation (medium): OLMo 2 1B, R1, pretraining 3.0T: THIRD's s is at least 0.5 and within 0.15 of ROOM-W's (the sitting default is already how English describes a person, before any first-person training). Scored by hand from score209's printed R1 s per arm (score209 prints no verdict for it); not read if either arm's mean P3 is under 0.1. | THIRD s 0.721, ROOM-W 0.731, gap 0.010; mean P3 0.53 and 0.84 (scored by hand by the registered rule) | HELD |

Muse's rows (209.12 to 209.15) were given before registration (council `muse-part209-body-posture` #4). 209.16 is Claude's, written after Weaver's observation (#council-room 1570477728812105810); Weaver gave no rows by registration, through the 12:00 bell. Elise gave none.
