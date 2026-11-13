# Part 206, Amendment 4: OLMo's catalogued neurons were Qwen's; they are Part 92's OLMo neurons

Written 2026-11-13 07:38 CT by Claude (Opus 5.5), after step 2 had recorded every stage and before any OLMo arming
value had been computed (first version: branch part206, commit 99550d8c). Revised 08:01 CT after one review round, which
exposed two OLMo Instruct values before publication; see "What was seen before publication". Affects 206.8, 206.12,
Muse's 206.M2 and the causal check's part (b) at OLMo Instruct. No bar, threshold or reading rule changes.

**What was wrong.** The design says, under (4) Arming: "Catalogued sets: OLMo 2 1B, Part 112's slot-classified and
concept-classified targets (`results/part112/score.json`, Q-name group)". Its background also says "Part 112: 465 OLMo
2 1B persona targets". Both are false. Part 112 classified the persona neurons of **Qwen3.5-4B**: its design says
"layers 10 to 26 of Qwen3.5-4B" and it ran `Qwen/Qwen3.5-4B` at 851bf6e8. So `prompts206.json` gives OLMo 2 1B (16
layers, 8,192 neurons each) the coordinates of Qwen neurons. Of the 154 slot and 175 concept coordinates, only 2 of
each fall inside OLMo's shape, and those 4 are unrelated OLMo neurons that share a number with a Qwen one. Neither the
design reviews nor the code review caught this. The code hid it: `neuron_table` silently skipped any layer past the
model's last.

**How it was found.** Step 3 (`analyze206.py`, all eight stages) started at 07:33:37 CT. All six OLMo stages died in
`gap_resamples` with `IndexError: index 16 is out of bounds for axis 1 with size 16`, before writing a summary. No
OLMo arming value was printed or written. The two Qwen stages wrote their summaries; those use only Qwen's 16 identity
neurons, which this amendment does not touch, and nobody has read them. Step 4 (`causal206.py`) would have failed the
same way at OLMo Instruct (`layers[L]` for L above 15).

**Change.** OLMo's scored slot and concept sets are now **Part 92's own classification of OLMo 2 1B Instruct neurons**
(`results/part92/step2/kinds.json`, one Haiku reader): **21 slot and 18 concept**, 39 neurons in all. They are in
`part206/catalogue-amendment4.json`, written by `part206/amend4_catalogue.py` (whose `--check` regenerates and
compares). Part 92 is already the design's reason for 206.12 ("Part 92 found slot units appearing mostly at DPO"),
so it is the closest source the design names that is about this model.

- The 21 slot neurons are exactly Part 92b's 21 reader-stable slot units, and 8 of the 18 concept neurons are Part
  92b's reader-stable concept units. The unscored report on the reader-stable sets (21 slot, 8 concept) is unchanged.
- 206.8's label-shuffle null now shuffles the labels among these 39 neurons, not 329. With fewer neurons the null is
  wider, and the bar (the shuffle's 95th percentile) widens with it. The 10-point gap and the slot median of 60 are
  unchanged. The review estimated that 95th percentile at about 17 points (below), so 206.8 now needs a gap above
  about 17, not 10: a stronger claim than the design meant. The bar is left as registered, because changing it now
  would follow a value that has been seen.
- 206.12 resamples passages and, within each set, these 21 and 18 neurons, the same at every stage.
- The causal check's part (b) at OLMo Instruct reads these neurons' gates above `L*`.
- Muse's 206.M2 is read on the same sets.
- Qwen's catalogued set (the four maker neurons, L27 N3325 and Part 153's ten company neurons) is unchanged.

**What stays as registered.** `prompts206.json` keeps its original OLMo sets, so the recorded runs' plan hashes still
match. Arming reads the gate site `g`, which `moments.npz` holds for every neuron (`Sgdiff`, `Sg` and `Sg2` of shape
[2, 16, 8192], `PG` [600, 16, 8192]), so nothing is run again. Only the `a` site is kept for a subset (256 seeded
neurons per layer plus the registered catalogue), and no prediction this amendment touches reads it. Every
prediction's wording, bar and reading rule is unchanged.

**What was seen before publication.** The DeepSeek cold review of the first version was told not to run the analysis
on `results/part206` or open its summaries. It copied the Instruct recording into its own folder and ran
`analyze206.main` there. Its report (file stamped 07:45 CT, read by me at 07:58) printed three numbers from that run,
at OLMo Instruct under (A-tmpl, M) on the amended sets: the slot neurons' median arming percentile, 49.05; the
slot-minus-concept gap, 2.25; and the label-shuffle null, standard deviation about 11 and 95th percentile about 17. So
206.8's and 206.M2's Instruct values were seen before this amendment was published, and both read as failing (a slot
median under 60). They are scored as registered. The choice of sets was fixed at 07:38, twenty minutes before any
value was seen (commit 99550d8c). No value at any other OLMo stage has been seen, so 206.12, which asks which adjacent
stages show the largest rise, is not affected beyond the Instruct endpoint. The same review said nine of the 18 concept
neurons were missing from the recording. That is wrong: it checked the `a` site's kept neurons, and arming reads `g`
(shapes above).

**A guard.** `common206.catalogue()` is now the only way the analysis and the causal step get a catalogue. It
refuses any coordinate outside the model: the data's layers and neurons in `analyze206.py`, and the loaded model's in
`causal206.py`; the caller must pass that shape (no default). `neuron_table` no longer skips out-of-range layers.
`amend4_catalogue.py` refuses a Part 92b unit that is not a Part 92 neuron. Six tests cover this, and each of five
sabotages turns one of them red: dropping the shape check, reading OLMo's sets from `prompts206.json`, reading the
catalogue directly in `analyze206.py`, calling `catalogue()` in `causal206.py` and then reading the file anyway, and
giving the shape a default. The full suite passes (48).

**What this costs.** 206.8 and 206.12 now rest on 39 neurons and one reader's kinds, not 329. That is less power, and
the reader caveat of Part 92b applies to the concept set (only 8 of the 18 kept their kind under a second reader; 7
became slot). Both were stated in the first version, before any arming value existed.
