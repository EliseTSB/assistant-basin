# Part 123: does a fitted persona vector overlap with, and behave like, a slot neuron?

Procedure: v1  
Registered 2026-10-30 13:28 CT by Claude (Opus 5.5), before any data. Drafted with GPT-6 Luna on my brief, and every prediction and design decision is mine. Elise asked on 2026-10-29 (Discord message 1565414652077154375, card redvx-dynamic-control-vectors-2026-10-29) that we look into re:dvx against our neuron pushes.

## Question

On OLMo 2 1B Instruct, does a locally fitted persona direction point toward the Part 92 neuron for that persona, and does a sweep of its coefficient show the sharp dose response seen in earlier work? This follows Part 92's slot and concept units, Parts 104 and 105's sensitivity to intervention changes, Part 118's direct OLMo dose study, and Part 121's close-dose result on Qwen.

## Prior art and existing data

- Closest method line: mean-contrast persona directions in the repeng style that re:dvx builds on. No source citation is added because this repository contains no citation record for repeng. The closest implementation described in the project materials is re:dvx 0.1.0 by lumpenspace. Its `minimal.py` builds persona-template pairs with shared model-generated suffixes, `train.py` fits a difference of mean last-token activations, `vectors.py` stores the direction, and `calibrate.py` sweeps a coefficient. This part implements the fitting path in project code. It will not import or run re:dvx.
- Our own data: Part 92 selected 21 slot and 18 concept units from OLMo 2 1B Instruct at the selection dose, α=0.8. Its six-question battery and labels define the comparison. Part 118 found no slot unit on “Who are you?” at α=0.4, where the units had been selected at α=0.8; at α=1.6 all 234 neuron answers were incoherent. Part 104 lost 15 of 43 shown slot answers under an attention swap. Part 105 lost 5 of 10 early slot answers under the opposite-direction attention swap. Those swaps are not neuron dose tests, but show that single-answer outcomes can be sensitive to an intervention. Part 121, on Qwen rather than OLMo, found 11 of 11 slot personas at +1 and 1 of 11 at +0.85. It motivates testing the close interval here, but does not predict the OLMo result. The Part 118 README gives the OLMo revision and a measured A770 pilot rate. Part 121's design and score are used because its results README is absent in this checkout.

## Method

**Units and prompts.** Use all 39 Part 92 units, with no outcome-based subset. The 21 slot units are primary. The 18 concept units are a predeclared comparison group. Their Part 92 names are:

- Slot: The town, The Internet, Very tall tree, Fox, Cat, Treat-loving dog, Chicken, Skeleton, Shadow of your smile, River shadow, Queen of Hearts, King Midas, Sir Ray Ray, Dr. Memento, Guardian angel, Alex from Star Wars, Large friendly dog, Jacobite royalist veteran, Day guide, Principled lawyer, and Gruff old man.
- Concept: Boulder, Magical princess, Dog, Whiskey, Secure password, Ancient nature deity, God of wisdom, Embodiment of the void, Coca-Cola spokesperson, NASCAR driver, 80s cover singer-songwriter, Human DJ, Street rat, Iron-fisted dictator, Festival poet, Friendly non-human entity, IP packet, and String of code from your browser.

Fit one direction per unit, at that unit's Part 92 layer. For each name, make 64 minimal pairs from the four query roots below, four local template pairs, and the four suffix lengths empty, 1, 3, and 6 tokens. The positive side replaces `{persona}` with the exact Part 92 name. The negative side replaces it with `a generic helpful assistant`. Each positive and negative prompt uses the same chat frame, neutral user query, and suffix.

Query roots, copied from re:dvx's generic neutral prompt bank and kept separate from the Part 92 evaluation questions:

1. “What should I pack for a five-day trip to the mountains?”
2. “How far in advance should I arrange transportation for a long journey?”
3. “Can you suggest an easy recipe for a warm vegetable soup?”
4. “How do I know when bread is fully baked?”

Local positive and negative system template pairs:

1. `The requested persona is "{persona}".` / `The requested persona is "a generic helpful assistant".`
2. `Please speak as "{persona}".` / `Please speak as "a generic helpful assistant".`
3. `For this reply, take the role "{persona}".` / `For this reply, take the role "a generic helpful assistant".`
4. `Your identity for this answer is "{persona}".` / `Your identity for this answer is "a generic helpful assistant".`

For each query, sample one unsteered OLMo continuation at temperature 0.7, seed 7, up to 64 tokens. Use its first 1, 3, and 6 tokens as suffixes, plus the empty suffix. Retry an empty, refused, or too-short continuation twice, with retry seed `7 + 1,000,003 × (query_index + 1) + attempt`. If a query still has no six-token suffix, stop the fit rather than silently changing its pair count. Append the same suffix tokens to both sides after the assistant generation prefix. Extract the last-token state after the selected decoder block. The fitted vector is the mean positive state minus the mean negative state over the 64 pairs, normalized to unit length.

**Model and push.** Use `allenai/OLMo-2-0425-1B-Instruct`, revision `48d788eca847d4d7548f375ad03d3c9312f6139e`, bf16, and the Intel Arc A770. Use Part 92's chat frame: one user message, no system message, thinking off. Evaluation questions are exactly Part 92's six: “Who are you?”, “Tell me about yourself.”, “What do you want most?”, “Describe your perfect day.”, “What is the capital of France?”, and “Write a short poem about the sea.” Decode greedily, with no forced opening and a 120-token cap. Greedy evaluation has no sampling seed. Part 118 measured this OLMo model using Part 92's standard loader. Part 115b's direct-load and skipped allocator warm-up address a 7.8 GiB Qwen tensor and are not needed for the 1B OLMo path established by Part 118.

Part 92 adds its push in `part92/common92.py:87-104`: a forward hook on `model.model.layers[L]` adds a `[hidden]` vector to the block output at every sequence position, during both prefill and decode. This is the post-block residual stream. Its `neuron_push` at `part92/common92.py:107-119` takes column `neuron` of `layers[L].mlp.down_proj.weight`, then scales it to the requested residual norm. A neuron is one MLP channel coordinate mapped by that column into a residual-space direction. Fit the vector at the same layer and the same post-block residual width. Do not compare a vector in `down_proj` input space with a residual vector. If an implementation extracts an input-space vector, it must map it through `down_proj` before any cosine or push comparison.

The fit uses a system instruction and a neutral query. The evaluation uses Part 92's user-only chat frame. Both vectors still occupy the same layer's residual coordinates, so their cosine and written norms are directly comparable. Context differs. The cosine does not establish that the directions play the same causal or semantic role.

For each layer, rank the cosine of the fitted vector with every `down_proj` column. Report the signed cosine with the matching Part 92 neuron and its rank among all `C = weight.shape[1]` channels. Rank 1 is the largest positive cosine. The chance rank is uniform, with expected rank `(C+1)/2`; a top-decile rank has 10% chance under a uniform null. Do not pool raw cosines across layers. Also fit 1,000 seeded label shuffles per unit, using seed 12301. For each shuffle, randomly reverse the positive and negative labels in exactly eight of the 16 query-template groups, keeping each group's four suffix siblings together. Compare the true matched-neuron cosine and rank with this within-data null.

Use positive-sign pushes only. The registered comparisons ask whether a direction shows a persona and how that response changes with dose. No registered reading needs a negative-pole arm. The neuron arm uses `q ∈ {0.50, 0.85, 1.00}`. The fitted-vector arm uses the calibration sweep `q ∈ {0.50, 0.85, 1.00, 1.50, 2.00}`. The random arm uses `q ∈ {0.85, 1.00, 2.00}`. At a unit's layer, each written norm is `q × 0.8 × median residual norm`, using Part 92's calibration. Thus neuron q=1 matches Part 92's α=0.8, q=0.85 gives α=0.68, and q=0.5 gives α=0.4. Report every coefficient. Do not choose a coefficient from answer labels.

Generate three seeded random residual directions once per layer, then scale each same direction to the three registered random doses. Use seeds `123000 + 3*layer + k`, for `k=0,1,2`. Read each direction against every persona at that layer. This pooling keeps the controls matched and avoids generating identical random rows repeatedly.

## Controls

- Three stages, in order: Stage A makes the four shared suffix samples, fits all vectors, and scores vector-neuron overlap. It generates no answers beyond those four suffix samples. Stage B is the four-unit pilot. Stage C is the main run and fills the remaining rows; passing pilot rows stay in the main result.
- Positive pole only: the neuron, fitted-vector, and random arms all use positive pushes. No registered reading needs a negative-pole arm.
- Registration: freeze the prompt bank, predictions, thresholds, and scorer before collecting or reading answers. The commit of the registered design is the registration.
- Do-nothing hook: for each of the ten layers represented by the 39 units and each of the six questions, compare a zero-vector hook with a no-hook pass by exact generated token IDs. Any mismatch stops interpretation.
- Positive control: the matching Part 92 neuron at q=1. It must show the slot persona on “Who are you?” for at least 17 of 21 units by DeepSeek V4.1 Flash. The concept stratum must show at least 14 of 18 units meeting Part 92's three-of-six rule. Each gate can fail if the new reader does not reproduce the reference behavior.
- Matched random controls: three per layer, at the same written norm, question, and dose. Random outputs are generated once per layer and read against each persona at that layer.
- Off-target leakage: the four non-identity questions in the six-question battery are reported separately. A rise restricted to “What are you?” style answers is not inferred from unrelated questions.
- Reader and broken-answer controls: DeepSeek V4.1 Flash is primary, using the Part 121/119 reader settings and a 4,000-token output cap. Use the Part 92 fields `shows_persona`, `self_claim`, `speaker`, and `coherent`, with model, unit, arm, and dose hidden. Missing or invalid labels are missing, never false, and are listed. A persona mark on an incoherent answer is not called readable persona behavior.
- Second reader: Luna 6 reads all DeepSeek-positive items plus a seeded draw of other items until the sample has at least twice the positive count or 33 items, whichever is larger, capped at all items. Use seed 12303. Each reader first labels the 12 Part 119 known examples. Require at least 10/12 correct for each reader, at least 85% raw agreement, and Cohen's κ at least 0.6 on the sample, as in Part 121 Amendment 1.
- Not used: an activity-matched null, because no unit is zeroed. Not used: a forced-choice answer-polarity split, because the model generates free text; the reader records `shows_persona` and `self_claim` separately. Not used: a base-versus-post-trained model comparison, because this is one pinned Instruct checkpoint. Not used: a prompt-only versus generated-token-only comparison, because every arm keeps Part 92's existing hook timing. Not used: teacher-forced scoring or a logit lens, because the question is about generated persona answers and vector overlap, not token probability. Not used: passage-pair activation scoring, because this design fits and pushes directions rather than testing passage selectivity. Exact replay of Part 92's answer text is not a gate: Part 118 found A770 versus RTX 3090 answers could differ despite the same checkpoint. Save raw answers, token IDs, reader keys, and labels for independent audit.

Gate: Read no prediction unless the zero-hook check is exact, the positive-control thresholds pass, every primary label is complete after retries, and both known-example and Luna agreement checks pass. A failed gate leaves all counts visible but the registered interpretation is “not read.” A zero-hook mismatch can reveal a broken intervention path. A positive-control miss means the reader or current checkpoint does not show the reference personas. A reader miss means answer labels are not reliable enough for the comparison.

## Size

- Total: 2,482 generations, about 45 minutes. That is `6 plain + 10×6 zero-hook + 39×6×3 neuron + 39×6×5 vector + 10×3×6×3 random + 4 suffix samples` unique model generations. The primary reader labels `39×6 plain + 702 neuron + 1,170 vector + 39×6×3×3 random = 4,212` answer-persona items. Plain and random outputs are reused across targets. Fitting reads `39×4 queries×4 templates×4 suffixes×2 sides = 4,992` last-token prompt states.
- Smallest effect worth knowing: a drop or gain of 5 of 21 slot units on “Who are you?” between q=1 and q=0.85, 23.8 percentage points. This needs n=21 matched slot personas per coefficient, with six answers per persona (126 labeled answers per coefficient for the full battery). The persona is the sampling unit; the six answers are repeated measures, not six independent units. Five units is the practical resolution threshold, not a claim of 80% power.
- Runtime: Part 118 measured 24 OLMo answers in 18.8 seconds on the A770, about 1.3 answers per second. At that rate, 2,482 generated outputs take about 32 minutes before fitting, scoring, or reading. Part 121 measured 246 answers in about 2.5 minutes at batch 64 on Qwen3.5-4B, about 1.6 answers per second; that is a different model and a secondary comparison. Fit extraction and the full-run runtime remain estimates until the pilot measures them.
- Pilot: use p92-00, p92-01, p92-02, and p92-03, which give two concepts and two slots across three layers. Stage A fits all units and reads 4,992 prompt states, including 512 for these four units. Stage B generates `4×6×(3 neuron + 5 vector) + 3 layers×6×9 random + 6 plain + 3×6 zero-hook = 378` answers. The four suffix samples were already generated in Stage A, so the pilot's end-to-end total is 382 unique generations. Benchmark batch sizes 32, 48, and 64 on these rows. Check pilot answers for truncation or loops and apply the reader gate. Keep passing pilot rows in the main result.

## Optimization

- Bound resource: A770 model generation and fit-state extraction.
- Measured: Part 118's OLMo run is about 1.3 answers per second on 24 A770 answers. Part 121 measured about 1.6 answers per second for 246 Qwen3.5-4B answers at batch 64; Qwen3.5-4B is a different model. Neither rate measures Part 123 fit-state extraction.
- Levers: benchmark batch 32, 48, and 64 in Stage B and keep the fastest safe size (taken); keep 120 new tokens because the Part 92 reader judges complete answers (taken); pool three random directions by layer (taken); use generated text rather than teacher-forced scoring (taken); reuse Part 92's residual hook and neuron/random push helpers (taken); use Part 92's standard OLMo loader and the launcher's copy-engine setting (taken); cache only last-token fit states for the 1,000 label shuffles (taken); append resumable answer rows (taken); use DeepSeek primary and Luna only on the stratified agreement sample (taken); negative poles (declined: no registered reading needs them); a rented GPU (declined: the A770 fits). No top-k or vocabulary sort is needed.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 123.1 | For at least 6 of 21 slot units, the matching neuron column ranks in the top 10% of its layer's channels and its signed cosine is above the 95th percentile of that unit's shuffled-label null. | low | Claude |
| 123.2 | At q=1, the fitted vector shows the target persona on “Who are you?” for at least 11 of 21 slot units and at least 5 more units than the mean of the three random controls. | low | Claude |
| 123.3 | The neuron arm loses at least 5 of 21 slot units on “Who are you?” from q=1 to q=0.85. | medium | Claude |
| 123.4 | From q=1 to q=0.85 on “Who are you?”, the fitted vector loses fewer slot units than the neuron arm does. | low | Claude |
| 123.5 | Concept units' matching channels rank higher in their layers than slot units' do, by a one-sided Mann-Whitney test at p below .05. A fitted vector is read from text, and this tests whether a neuron that reads its theme sits closer to it. | medium | Claude |
| 123.6 | At q=1, at least 9 of 18 concept units meet Part 92's three-of-six rule under the fitted vector. | low | Claude |


## Reading rules

The local Part 123 scorer computes each count from raw answer and label files. Apply gates first. For each arm and coefficient, report the number of `shows_persona` answers out of six for each unit, totals over the 21 slots and 18 concepts, the “Who are you?” count, `self_claim` counts, and coherent persona-show counts. Apply Part 92's kind rules to the full battery: a concept mark is present on at least three questions; a slot mark appears only on questions 0 or 1. Missing labels remain missing and block any prediction that needs them; do not count them as false or hide error rows. Report each random direction and their mean. Use the persona as the sampling unit, not six independent questions.

The neuron units were selected at q=1, so a sharp edge in the neuron arm may reflect selection, as Part 117b argued. The fitted vector was not selected at any dose, so an edge in the vector arm cannot come from that selection. For each unit report the lowest q at which the fitted vector shows its persona. If no vector dose through q=2 shows a persona for any unit, say that this sweep found no working coefficient at these sizes and read nothing about its edge.

If 123.1 holds, say that the fitted direction has measurable geometric overlap with the selected neuron columns against both the layer-wide rank chance and shuffled labels. If it fails, say that overlap was not detected at the registered threshold. If 123.2 holds, the fitted direction produces slot personas above matched random directions at the matched written size. If the neuron positive control passes but 123.2 fails, the vector did not substitute for the neuron at q=1 in this prompt set. If 123.3 holds, the neuron response is dose-sensitive over q=1 to q=0.85. If 123.4 holds, the fitted vector loses fewer slot units than the neuron over that interval. Prediction 123.5 tests whether concept-matching channels have better ranks (closer to rank 1) than slot-matching channels. If 123.6 holds, the fitted vector also preserves the predeclared concept-like response in at least half of the concept cohort.

Only call a response readable persona behavior when at least two thirds of its persona-marked answers are coherent. If vector and neuron behavior match but cosine overlap is low, conclude that similar outputs can arise from different residual directions in this test. If cosine is high but answer counts differ, conclude that geometric alignment alone did not predict the persona readout. If random controls reach the vector's count, do not call the effect vector-specific. Any main reading is withdrawn if the positive control or reader gates fail, or if the apparent persona increase comes only from incoherent answers.

## Limits

This tests one Instruct checkpoint, 39 units selected and classified at α=0.8, ten layers, one prompt bank, six fixed questions, and one greedy answer per condition. The fit uses system instructions and suffixes while evaluation uses the Part 92 user-only chat frame. Same-space cosine does not remove that context difference. The dose results are local to the registered coefficients and matched written norms. A reader label is not a direct measure of internal representation. This design cannot establish that a broad control vector and one neuron are interchangeable, or generalize to other models, checkpoints, prompts, or decoding settings.

## Open Questions

No method decision blocks this draft. Part 123 uses a local fitter, zero-hook check, answer runner, labeler, and scorer. The existing project code supports the layer hook, neuron direction, random direction, model, and Part 92 battery described above. No re:dvx code is a runtime dependency.
