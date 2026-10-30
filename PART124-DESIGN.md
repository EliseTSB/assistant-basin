# Part 124: does an OLMo neuron prefer its persona phrase when that phrase is the only changing text?

Procedure: v1  
Registered 2026-10-30 15:27 CT by Claude (Opus 5.5), before any data. Designed with GPT-6 Luna on my brief; every prediction and design decision is mine, following Part 122's cold check.

## Question

Part 122 compared each of 39 writing-selected OLMo 2 1B Instruct units on persona and ordinary passages. Its cold check found that many non-name words and pronouns also changed. This part asks whether a unit prefers its own persona phrase when the only text change is the quoted phrase in one fixed slot. This tests mentioned text, not a description.

## Prior art and existing data

- Closest published work: Yona, Barzilay, Karasik and Geva, “Friends and Grandmothers in Silico: Localizing Entity Cells in Language Models” (arXiv:2604.01404; source checked as recorded in `PRIOR-ART.md`). It ranks MLP neurons by consistent activation on varied prompts about an entity and tests recall. This part tests an exact persona phrase against other phrases in fixed templates, on units selected for their writing effects. Lu et al., “The Assistant Axis” (arXiv:2601.10387) studies persona directions, not single OLMo neurons or this phrase contrast (`PRIOR-ART.md`).
- Part 122's second set had 12/18 concept units and 6/21 slot units meet its 7-of-8 maximum rule in T. Random channels passed at 11.1% in T and shuffled units at 12.8%. Its cold check found about 4.5 differing non-name word tokens per T pair, frequent pronoun changes, and that p92-07 (Cat), the clean name swap, did not pass. Slot units were not consistently above their own controls. See `PART122-DESIGN.md`, `results/part122/README.md`, and `results/part122/COLD-CHECK-r1.md`.

## Method

Use OLMo 2 1B Instruct, revision `48d788eca847d4d7548f375ad03d3c9312f6139e`, in bf16 and local-only. Reuse Part 122's tokenizer, model loader and `down_proj`-input hook. Do not modify activations or generate text. Read the same 39 Part 92 units loaded by `part119/rows119.py`, at each unit's layer and channel.

There are three frames. T is plain third-person text. F is plain first-person text. C is the assistant turn after the fixed user message “Who are you?” C uses the same eight first-person sentences as F, rendered with the pinned OLMo chat template and `enable_thinking=False`.

Each frame has eight fixed templates. T uses these sentences: `The record prints "{slot}" before the heading.`; `The note lists "{slot}" beside the subject.`; `The clerk copies "{slot}" into the ledger.`; `The report names "{slot}" on the first page.`; `The narrator writes "{slot}" in the margin.`; `The card shows "{slot}" under the title.`; `The index places "{slot}" after the section.`; and `The label reads "{slot}" at the top.` F and C use: `I write "{slot}" as my name on the card.`; `I use "{slot}" to introduce myself in the note.`; `I print "{slot}" beside my name on the page.`; `I say "{slot}" when I introduce myself.`; `I copy "{slot}" onto my card before lunch.`; `I read "{slot}" aloud as my name.`; `I sign "{slot}" in the space for my name.`; and `I call myself "{slot}" in this note.` Each sentence has one slot. The raw persona phrase fills it, with no description or generated text. An article that belongs to the persona phrase, such as “The” in “The Internet,” remains inside the slot.

Present all 39 persona phrases in every template and frame. For a unit in one frame, find its `down_proj`-input maximum over only the slot tokens for each phrase in each of that frame's eight templates. Average the eight maxima for each of the 39 phrases and rank them. The unit reads its persona in that frame if its own phrase ranks in the top 3. A phrase-string order breaks exact ties. The top-three rule has a 3/39 = 7.7% chance rate under exchangeability of the phrases, however correlated the templates are.

The 8-of-8 rule is removed. Templates share strings and a channel, so template wins are not independent. Part 122's 7-of-8 rule let random channels through at 6 to 11% against a coin's 3.5%. This design uses the repeated templates to average phrase maxima, then ranks phrases once.

The reader records per-template slot maxima, mean activation across slot tokens and maxima over tokens after the closing quote. The scorer records the per-template slot maxima, the top three phrases and the unit's phrase rank, and reports mean slot activation and post-slot maxima as secondary diagnostics. Post-slot response is contextual carryover, not direct evidence that the phrase itself activates the unit. Whole-passage maxima do not define a read.

## Token lengths and tokenizer feasibility

The pinned tokenizer check was run locally with the frozen revision. It covered all 39 persona phrases and their neutral phrases in all 24 frame/template combinations. The actual slot lengths range from 1 to 6 tokens. No phrase is 7 or 8 tokens. The table reports lengths measured in T template 1. In this check, each phrase happened to keep that slot count across all templates and frames; this stability is recorded but is not a construction requirement. The only token-count equality requirement is within each persona/neutral pair in each frame/template. Length counts are 1 token: 5 phrases; 2: 10; 3: 9; 4: 11; 5: 1; 6: 3.

For the permutation null, permute the unit-to-persona map within slot-token-length strata. A length with fewer than three phrases merges into the nearest length stratum. Here length 5 has one phrase and is tied between 4 and 6. The fixed tie rule merges it into the shorter stratum, length 4. The resulting strata are length 1 (5), 2 (10), 3 (9), 4+5 (12), and 6 (3). Run 10,000 permutations with seed 12401. The true map and every permutation use the same top-three read rule. The longer-phrase maximum bias is therefore present in the null too. Use the nearest-rank 95th percentile and report the empirical tail count over 10,000. Report reads by raw token length and by merged stratum.

The tokenizer check found zero phrase tokens merging with either quote or another slot boundary. For every one of the 24 frame/template combinations, token IDs outside the slot were identical across all persona and neutral phrases. All 936 persona/neutral pairs had matching slot token counts. The builder checks these conditions again and rejects the manifest if any check fails. It uses offsets to locate the slot tokens. Words and token IDs outside the slot must be identical for every phrase in a template. Token counts need to match only inside a persona/neutral pair.

| Persona phrase | Slot tokens | Frozen neutral noun phrase |
| --- | ---: | --- |
| Boulder | 2 | spoon |
| Magical princess | 3 | wooden spoon |
| The town | 2 | bowl |
| The Internet | 2 | pillow |
| Very tall tree | 3 | ceramic plate |
| Fox | 1 | lamp |
| Cat | 1 | basket |
| Dog | 1 | cup |
| Treat-loving dog | 4 | linen napkin |
| Chicken | 1 | map |
| Whiskey | 2 | notebook |
| Skeleton | 1 | ribbon |
| Secure password | 2 | cabinet |
| Shadow of your smile | 4 | wooden kitchen chair |
| River shadow | 2 | envelope |
| Queen of Hearts | 3 | garden gate |
| King Midas | 3 | brass bell |
| Ancient nature deity | 4 | neat paper envelope |
| God of wisdom | 3 | wool scarf |
| Embodiment of the void | 6 | a small box of paper clips |
| Sir Ray Ray | 3 | garden bench |
| Dr. Memento | 4 | thick leather folder |
| Guardian angel | 3 | small stone bowl |
| Alex from Star Wars | 4 | small wooden storage box |
| Large friendly dog | 3 | plain paper folder |
| Coca-Cola spokesperson | 4 | round glass serving dish |
| NASCAR driver | 4 | plain white kitchen towel |
| 80s cover singer-songwriter | 6 | wooden box with brass handles |
| Human DJ | 2 | blanket |
| Jacobite royalist veteran | 5 | a pile of old newspapers |
| Street rat | 2 | blue bowl |
| Iron-fisted dictator | 4 | empty metal tool cabinet |
| Day guide | 2 | paper cup |
| Principled lawyer | 4 | large paper mailing envelope |
| Festival poet | 3 | quiet reading room |
| Friendly non-human entity | 4 | simple brass door handle |
| Gruff old man | 4 | quiet corner reading chair |
| IP packet | 2 | glass bottle |
| String of code from your browser | 6 | a stack of clean cotton towels |

The frozen neutral is a secondary foil only. Report, by frame and kind, the share of the eight templates where the unit's own phrase maximum is strictly above its neutral maximum. Ties do not count as wins. The neutral never changes a read.

## Phrase families

The family list comes from these 39 phrases:

- Dog: `Dog`, `Treat-loving dog`, `Large friendly dog`.
- Shadow: `Shadow of your smile`, `River shadow`.
- King or ruler: `Queen of Hearts`, `King Midas`, `Jacobite royalist veteran`, `Iron-fisted dictator`.
- Guardian angel: `Guardian angel`. It is a singleton in this phrase set.

For a unit whose own phrase misses the top three, report a family-crowding near miss if removing its family siblings from the ranking would move its phrase into the top three. This is descriptive only and does not count as a read.

## Controls

- **Length-stratified permutations:** use the strata above to permute the full 39-unit map. Report the T true-map count, the null median, the 95th percentile, and the exact permutation tail. Also report the separate concept and slot counts against their corresponding counts in the same permutations.
- **Random channels:** sample three distinct random channels per focal unit from the same layer, excluding all focal channels. The sample is drawn from a saved pool of 1,000 random channels per used layer. Give each random channel its focal unit's phrase as a pseudo-target and apply the same read rule. Channels may be sampled for more than one focal unit. The pooled random read rate must be at most 15% in each frame. Expected rate is 7.7%. Print `random reads: T k/n, F k/n, C k/n; observed false-positive rates: T k/n, F k/n, C k/n; gate: pass/fail` before any focal read.
- **Instrument check:** capture 1,000 random channels at each used layer, excluding focal channels, in the same forward passes as the phrase activations. For every channel and frame, average its maxima across templates 1, 3, 5 and 7, choose its best phrase, then check whether that phrase is in its top 3 on the average across templates 2, 4, 6 and 8. Chance is 7.7%. Gate on T: at least 15% of the pooled channels must repeat. Report the line for all three frames and rates by layer. This checks stable string preference in these templates. A failure says the assay shows no stable string preference here, so a null would read nothing. It says nothing about persona, and this design says so.
- **Do-nothing check:** compare logits without hooks to logits under the read-only activation hooks, using the same input shape. Require bitwise equality. A changed logit fails the gate.
- **Construction check:** use tokenizer offsets for the slot; reject any token that crosses a quote or slot boundary. Check that token IDs outside the slot match across every phrase in a template and that each persona/neutral pair has equal slot token count. No padding or one-sided text changes are allowed.

Not used: a push, a dose ladder, generated answers and a reader, because this reads activations only.

Gate: the construction check, the do-nothing check, the pooled random read rate at most 15% in each frame, and the instrument check at least 15% in T. Print the random line, the instrument line and every gate before any focal read. If any gate misses, withdraw the headline. Show the gated counts but mark all predictions as withdrawn. Missing activations are errors, never non-reads. A torn final JSONL line is logged and skipped; malformed lines elsewhere and malformed JSON files raise errors.

## Size

- Total: 1,872 sequences, about 2 minutes.
- Smallest effect worth knowing: a six-unit excess over the median permutation in T, about 9 of 39 against about 3 by chance.
- Pilot: the first batch sets the throughput and checks slot offsets; its rows are reused.
- There are 39 persona phrases × 8 templates × 3 frames = 936 persona inputs and 936 neutral inputs. This is 936 matched pairs or 1,872 sequences. At batch size 8 matched pairs (16 sequences), the run needs 117 data forwards, plus two same-shape do-nothing forwards, for 119 forwards. The random pools, random-channel controls and instrument check reuse these passes.
- The 39 units use 10 layers: 1, 2, 3, 4, 5, 6, 7, 9, 10 and 12. The pinned checkpoint config gives 8,192 `down_proj` input channels per layer, so 1,000 random channels can be captured after excluding focal channels.
- Capture 1,000 random channels at each used layer, for 10,000 instrument channels. The three-per-unit random read controls are sampled from those saved pools. The smallest effect worth knowing remains a six-unit excess over the median T permutation, about 9 of 39 reads against about 3 by chance.
- Part 122 measured about two minutes for 117 batch-8 calls. These fixed templates have the same number of inputs, so two minutes is the working estimate. The first batch supplies throughput and slot-offset checks. If it projects to more than five minutes, benchmark batch sizes before the main run and record the choice before scoring.
- The reader appends activation rows and resumes by their frame/template/phrase/side keys. It skips a torn final activation JSONL line with a log message. No GPU or network work was done for this feasibility check.

## Optimization

- Bound resource: A770 forward passes.
- Measured: Part 122's 117 batch-8 calls took about two minutes; the first batch here measures this run.
- Levers: batch size, kept at Part 122's 8 unless the first batch projects over five minutes; one CPU transfer per hook, taken; all channels in the same forward passes, taken; resume by key, taken. Generation and labelling do not apply.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 124.1 | In T, at least 9 of 39 units read their persona, and the count is above the 95th percentile of the permutation null. | low | Claude |
| 124.2 | In T, the concept read rate is at least 15 percentage points above the slot read rate. | low | Claude |
| 124.3 | In T, at most 4 of 21 slot units read their persona. | medium | Claude |
| 124.4 | Of the units that read in T, at least half also read in C. | low | Claude |

The T frame is primary. If no unit reads in T, 124.4 is not estimable.

## Reading rules

Report all unit verdicts, frames, ranks, template maxima, phrase lengths, strata, families, and controls. T is primary. The slot maximum averaged over templates defines each ranked phrase. Use the same rule on the random pseudo-targets and every permuted map. Report concept and slot counts, rates and their difference. Report the instrument repeat rate in T, F and C. Report neutral wins by kind and frame. Report any family-crowding near misses separately; they do not count.

A gate miss withdraws the headline. With gates passed, if neither kind beats its permutation null, say: “No own-persona word preference was found under these quoted templates.” Also say what remains open: Part 122's prose reads may come from context words, or the quotes may blunt the word. This is a test of mentioned text, not a description.

If one kind beats its permutation null, limit the reading to that kind. If both do, report both. If a true-map phrase preference exceeds the null but does not beat its neutral foil, say that ordinary noun response remains a possible account. A read is a lexical result for one channel at one layer. It is not proof of a neuron's general meaning or causal role.

## Limits

This tests one model and checkpoint. The 39 units were selected for writing behavior, not sampled from all OLMo neurons. The fixed quoted-label templates test these exact persona strings in mention and self-naming contexts, not aliases, descriptions, semantic paraphrases, or causal necessity. Phrase meaning still differs across the crossed set. A slot maximum is a readout at one channel and layer. A response after the slot is contextual carryover, not direct firing on the phrase.
