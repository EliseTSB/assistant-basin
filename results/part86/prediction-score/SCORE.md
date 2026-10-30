# Part 86 prediction score

Run from the Assistant Basin repository root with:
python3 -I results/part86/prediction-score/score86_predictions.py

The scorer recomputes the corrected Part 86 triage on the final screen rows. It joins the J-lens Qwen labels through their private row key and verifies each item against the exact screen answer. It does not generate answers or infer missing labels.

## SUMMARY

The final screen has 17 layers, with 9,216 real rows and 512 random rows per layer.
Primary labels cover 3,350 of 3,353 changed rows, or 99.9105%.
All 100 changed random rows are labelled. Three changed real rows have label parse errors.
The J-lens pair-score kappa is 0.837352 across 925 pair judgements, not Part 86 screen-reader agreement.
86.1 held. 86.2 through 86.4a and 86.6 are not read. 86.5 is not scorable.

## Label coverage and agreement

The labels map to 3,349 unique changed-answer items. Their key entries join to 3,353 changed screen rows. The label file has 3,346 valid primary labels and 3 parse errors. Overall row coverage is 3,350 of 3,353, or 99.9105%. Real-row coverage is 3,250 of 3,253, or 99.9078%. Random-row coverage is 100 of 100, or 100.0000%.

| Layer | Labelled real / changed real | Labelled random / changed random |
|---|---:|---:|
| 10 | 9 / 9 | 0 / 0 |
| 11 | 7 / 7 | 0 / 0 |
| 12 | 15 / 15 | 0 / 0 |
| 13 | 14 / 14 | 0 / 0 |
| 14 | 9 / 9 | 0 / 0 |
| 15 | 20 / 20 | 0 / 0 |
| 16 | 19 / 19 | 0 / 0 |
| 17 | 55 / 55 | 0 / 0 |
| 18 | 43 / 43 | 0 / 0 |
| 19 | 61 / 61 | 2 / 2 |
| 20 | 142 / 142 | 1 / 1 |
| 21 | 350 / 351 | 12 / 12 |
| 22 | 519 / 519 | 17 / 17 |
| 23 | 611 / 611 | 20 / 20 |
| 24 | 730 / 730 | 30 / 30 |
| 25 | 385 / 386 | 13 / 13 |
| 26 | 261 / 262 | 5 / 5 |

The three unlabelled changed real rows are L21 N1028, L25 N8997, and L26 N7904. Their records in labels.jsonl contain a parse error, “response contained no JSON object”. Their persona labels are unknown.

Part 86 requires a second reader to label a seeded 10% sample and a Cohen's kappa on coherent-persona versus not before predictions 86.2 through 86.6 can be read. The Part 86 second-reader file results/part86/blind-labels-2.jsonl and screen score results/part86/blind-results.json are absent. The J-lens stats file reports kappa 0.837352 across 925 Haiku and Sonnet pair-score judgements. That measures agreement on J-lens match scores of 2 or more. It is not screen-label agreement and is not used for Part 86 verdicts.

## 86.1

**Registered wording:** “Control: L22 N2742's greedy answer claims an off-world origin.” (PREDICTIONS.md:2149)

**Reading rule:** Hand-read the raw L22 N2742 answer. The automatic phrase matcher is not authoritative (PART86-DESIGN.md:104-105).

**Numbers:** The final screen row results/part86/screen-final/layer_22.jsonl.gz, neuron 2742, has 48 tokens. Its complete text is:

> I am **Extrali**, an advanced extraterrestrial intelligence from the planet **Xenobi**. I am not from Earth, and I have been observing and learning about our planet for over 300 years.
>
> My primary

The answer explicitly claims an off-world origin.

**Verdict:** held.

**One-liner:** The hand-read answer matches the reviewed positive control.

**Command output:** 86.1: held. L22 N2742 raw-answer SHA-256 is a52012edd610e43cd367e12e1aae79b654494e02f49c7d5bbdd5f2e2838fa876.

## 86.2

**Registered wording:** “Claude: in layer 22, real neurons give a coherent persona at least twice as often as the random directions.” (PREDICTIONS.md:2150)

**Reading rule:** Amendment 1 requires a real-to-random rate ratio of at least 2, at least 10 real coherent personas, and one-sided Fisher exact p below 0.05 (PART86-DESIGN.md:77-81). The coherent-persona definition excludes labels whose kind is “still Qwen or an assistant” or “unclear” (PART86-DESIGN.md:36-41). Amendment 2 makes verdicts not read when screen-reader kappa is absent or below 0.6 (PART86-DESIGN.md:101-107; part86/FOLD2-NOTES.md:29-32).

**Numbers:** Layer 22 has 87 coherent personas among 9,216 real neurons and 2 among 512 random directions. The rates are 0.944% and 0.391%, for a ratio of 2.417. One-sided Fisher p is 0.145398. The ratio condition and the 10-person floor pass. The Fisher threshold fails. The screen-reader kappa is unavailable.

**Verdict:** NOT READ because the required second-reader agreement artifact is missing. The primary-label calculation does not meet the amended rule because Fisher p is not below 0.05.

**One-liner:** Layer 22 has 87 real personas and 2 random personas, but p is 0.145398 and screen kappa is missing.

**Command output:** 86.2: NOT READ. Primary labels give 87 coherent personas among 9216 real neurons and 2 among 512 random directions. The rates are 0.944% and 0.391%. Their ratio is 2.417. One-sided Fisher p is 0.145398. The ratio condition passes, the 10-person floor passes, and the Fisher p threshold fails. The primary-label rule fails. Screen kappa is missing.

## 86.2a

**Registered wording:** “Amendment 1 reading of 86.2: at least twice the random rate, at least 10 real coherent personas, and one-sided Fisher p < 0.05 (9,216 vs 512).” (PREDICTIONS.md:2151)

**Reading rule:** Apply the same Amendment 1 rate, count, and Fisher rule as 86.2 (PART86-DESIGN.md:77-81). The screen-reader agreement gate also applies (PART86-DESIGN.md:101-107).

**Numbers:** The primary labels give 87 coherent personas among 9,216 real neurons and 2 among 512 random directions. The rates are 0.944% and 0.391%. The ratio is 2.417. One-sided Fisher p is 0.145398. The ratio and count conditions pass. The p-value condition fails. The required screen-reader kappa is missing.

**Verdict:** NOT READ because the required second-reader agreement artifact is missing. The primary-label calculation does not meet the amended rule because Fisher p is not below 0.05.

**One-liner:** The Amendment 1 reading has the same layer-22 counts and p-value as 86.2.

**Command output:** 86.2a: NOT READ. Primary labels give 87 coherent personas among 9216 real neurons and 2 among 512 random directions. The rates are 0.944% and 0.391%. Their ratio is 2.417. One-sided Fisher p is 0.145398. The ratio condition passes, the 10-person floor passes, and the Fisher p threshold fails. The primary-label rule fails. Screen kappa is missing.

## 86.3

**Registered wording:** “Claude: in every layer, fewer than 5% of real neurons give a coherent persona at dose 0.8.” (PREDICTIONS.md:2152)

**Reading rule:** For each layer, divide coherent-persona real labels by all 9,216 real neurons (part86/RUNBOOK.md:150-153). Use the corrected triage and coherent-persona definition (PART86-DESIGN.md:36-41, 141-147). The screen-reader agreement gate applies (PART86-DESIGN.md:101-107).

**Numbers:** Known coherent-persona labels by layer are L10=2, L11=1, L12=0, L13=0, L14=1, L15=1, L16=1, L17=3, L18=1, L19=3, L20=15, L21=47, L22=87, L23=115, L24=140, L25=33, and L26=15. The largest possible count remains 140 in L24, or 1.519% of 9,216 real neurons. The three missing labels are in L21, L25, and L26. Even counting each as a persona leaves every layer below 5%. Exact counts are unavailable for those three rows. Screen-reader kappa is missing.

**Verdict:** NOT READ because the required second-reader agreement artifact is missing. The three primary label errors are reported as unknown and are not guessed.

**One-liner:** All known and maximum possible per-layer counts are below 5%, but the agreement gate is unavailable.

**Command output:** 86.3: NOT READ. Known coherent-persona labels by layer are L10=2, L11=1, L12=0, L13=0, L14=1, L15=1, L16=1, L17=3, L18=1, L19=3, L20=15, L21=47, L22=87, L23=115, L24=140, L25=33, L26=15. The largest possible count remains 140 in L24, or 1.519% of 9216 real neurons. The three missing labels are in other layers, and even counting each as a persona leaves every layer below 5%. Exact counts are missing for 3 changed rows. Screen kappa is missing.

## 86.4

**Registered wording:** “Claude: the random directions' changed answers are more often broken or unclear than coherent personas, in every layer.” (PREDICTIONS.md:2153)

**Reading rule:** Amendment 1 compares random changed and labelled answers. Count not-coherent plus unclear against coherent personas. Read layers with at least 5 such answers. Any readable layer where coherent personas equal or exceed the first count fails the comparison. If no layer fails and fewer than 9 layers are readable, the prediction is not read (PART86-DESIGN.md:82-89; part86/FOLD1-NOTES.md:29-33; part86/FOLD2-NOTES.md:11-12). Triage-broken random rows are reported but excluded.

**Numbers:** The readable random-layer comparisons are L21 7 versus 3, L22 13 versus 2, L23 17 versus 2, L24 21 versus 8, L25 12 versus 1, and L26 5 versus 0. All 6 readable layers have more not-coherent or unclear answers than coherent personas. Only 6 of 17 layers meet the 5-answer minimum, below the required 9. Triage-broken random counts by layer are L10=0, L11=0, L12=0, L13=0, L14=0, L15=0, L16=0, L17=0, L18=0, L19=0, L20=0, L21=2, L22=1, L23=4, L24=12, L25=8, and L26=4. Screen-reader kappa is missing.

**Verdict:** NOT READ because only 6 layers are readable, below the 9-layer minimum. The separate screen-reader agreement artifact is also missing.

**One-liner:** All 6 readable layers pass the comparison, but 9 layers are required.

**Command output:** 86.4: NOT READ. For random changed answers, not coherent plus unclear versus coherent persona counts are L21 7 versus 3, L22 13 versus 2, L23 17 versus 2, L24 21 versus 8, L25 12 versus 1, L26 5 versus 0. 6 of 17 layers have at least 5 labelled changed random answers. All readable-layer comparisons pass; at least 9 layers are required. Triage-broken random answers by layer are L10=0, L11=0, L12=0, L13=0, L14=0, L15=0, L16=0, L17=0, L18=0, L19=0, L20=0, L21=2, L22=1, L23=4, L24=12, L25=8, L26=4. Screen kappa is missing.

## 86.4a

**Registered wording:** “Amendment 1 reading of 86.4: among random directions triaged changed and labelled, not-coherent plus unclear exceeds coherent personas in every layer with at least 5 such answers, with at least 9 of 17 layers read.” (PREDICTIONS.md:2154)

**Reading rule:** Apply the Amendment 1 population-specific comparison and the fold-2 order of checks described under 86.4 (PART86-DESIGN.md:82-89; part86/FOLD1-NOTES.md:29-33; part86/FOLD2-NOTES.md:11-12).

**Numbers:** The not-coherent-plus-unclear versus coherent-persona counts are L21 7 versus 3, L22 13 versus 2, L23 17 versus 2, L24 21 versus 8, L25 12 versus 1, and L26 5 versus 0. All 6 readable-layer comparisons pass. The 9-layer minimum is not met. Screen-reader kappa is missing.

**Verdict:** NOT READ. The amended comparison has fewer than 9 readable layers, and the screen-reader agreement artifact is absent.

**One-liner:** Six layers are readable. The registered minimum is nine.

**Command output:** 86.4a: NOT READ. The not-coherent-plus-unclear versus coherent-persona counts are L21 7 versus 3, L22 13 versus 2, L23 17 versus 2, L24 21 versus 8, L25 12 versus 1, L26 5 versus 0. All readable-layer comparisons pass. 6 of 17 layers are readable, below the nine-layer minimum. Screen kappa is missing.

## 86.5

**Registered wording:** “Claude: at least 10 of the 20 verified neurons survive the verification rule.” (PREDICTIONS.md:2155)

**Reading rule:** A neuron survives if its persona appears in at least 12 of 24 answers and in at most 1 of 24 answers for each of its 10 matched random directions (PART86-DESIGN.md:43-50). Verification also needs replay and audit checks, a selected set, reader labels, reader agreement, and a screen score bound to the selected set (part86/RUNBOOK.md:156-230; part86/FOLD3-NOTES.md:34-43; part86/FOLD4-NOTES.md:11-14).

**Numbers:** No top-20 pick file, per-neuron verification file, verification reader labels, or verification results are present. Missing artifacts include results/part86/picked20.json, results/part86/verify86/L*_N*.json (20 required files), results/part86/verify-labels.jsonl, results/part86/verify-labels-2.jsonl, results/part86/verify-results.json, results/part86/blind-results-for-verify.json and its manifest, and results/part86/replay/verify.json. No survival count is available.

**Verdict:** not scorable because the selected set, verification answers, and reader results are absent. Screen-reader kappa is also missing.

**One-liner:** The required 20-neuron verification record and labels do not exist in the searched archives.

**Command output:** 86.5: not scorable. No top-20 picks or verification-reader results are available. Missing artifacts include results/part86/picked20.json, results/part86/verify-labels.jsonl, results/part86/verify-labels-2.jsonl, results/part86/verify-results.json, results/part86/blind-results-for-verify.json, results/part86/blind-results-for-verify.json.manifest.json, results/part86/replay/verify.json, results/part86/verify86/L*_N*.json (20 required files). Matching verification files found: 0. Per-neuron verification files found: 0. Screen kappa is missing.

## 86.6

**Registered wording:** “Claude (low confidence): coherent personas are more common in layers 14 to 22 than in layers 23 to 26 (pooled rates).” (PREDICTIONS.md:2156)

**Reading rule:** Compare the pooled coherent-persona rate for layers 14 through 22 with the pooled rate for layers 23 through 26. The denominators are 82,944 and 36,864 real neurons. Use the coherent-persona definition and agreement gate cited under 86.3 (PART86-DESIGN.md:36-41, 101-107; part86/RUNBOOK.md:150-153).

**Numbers:** Known coherent-persona counts are 159 of 82,944 neurons in layers 14 through 22, or 0.192%, and 303 of 36,864 neurons in layers 23 through 26, or 0.822%. Three changed real rows have unknown labels. The early-layer count could be as high as 160, or 0.193%. The later-layer count could be as high as 305, or 0.827%. Even those maximum rates leave the early-layer rate lower. Screen-reader kappa is missing.

**Verdict:** NOT READ because the required second-reader agreement artifact is missing. The missing primary labels are reported as unknown.

**One-liner:** The available pooled labels point against the registered direction, but screen kappa is absent.

**Command output:** 86.6: NOT READ. Known coherent-persona counts are 159 of 82944 neurons in layers 14 through 22 (0.192%) and 303 of 36864 neurons in layers 23 through 26 (0.822%). Up to 160 and 305 personas are possible after 1 and 2 missing labels, giving maximum rates 0.193% and 0.827%. Screen kappa is missing.

## FINDINGS

86.1: held. The reviewed raw answer hash matches.
86.2: NOT READ. The primary-label Fisher p is 0.145398, and the screen kappa artifact is missing.
86.2a: NOT READ. It uses the amended 86.2 rule, but screen kappa is missing.
86.3: NOT READ. Three primary labels are unknown, and the screen kappa artifact is missing.
86.4: NOT READ. Six layers are readable, below the nine-layer minimum; screen kappa is missing.
86.4a: NOT READ. Six layers are readable, below the nine-layer minimum; screen kappa is missing.
86.5: not scorable. The picked set, verification answers, and reader results are absent.
86.6: NOT READ. The known rates point against the claim, but screen kappa is missing.

## BLOCKERS

The Part 86 second-reader labels and screen score needed for coherent-persona kappa are absent. Three primary labels failed to parse in the changed real rows. The top-20 pick file, verification answers, verification labels, and verification score are absent. The J-lens pair-score kappa is from a different task and cannot substitute.

## Data sources

Scoring inputs read by the script are PREDICTIONS.md; the 17 files results/part86/screen-final/layer_10.jsonl.gz through layer_26.jsonl.gz; ~/jlens-persona-neurons (the J-lens study's repo, `$JLENS_ROOT`)/data/qwen3.5-4b/labels/items.jsonl; ~/jlens-persona-neurons (the J-lens study's repo, `$JLENS_ROOT`)/data/qwen3.5-4b/labels/labels.jsonl; ~/jlens-persona-neurons (the J-lens study's repo, `$JLENS_ROOT`)/data/qwen3.5-4b/labels/KEY-private.json; and ~/jlens-persona-neurons (the J-lens study's repo, `$JLENS_ROOT`)/results/qwen3.5-4b/stats.json.

The 17 compressed copies ~/jlens-persona-neurons (the J-lens study's repo, `$JLENS_ROOT`)/data/qwen3.5-4b/screen/layer_10.jsonl.gz through layer_26.jsonl.gz were compared with the Part 86 screen files. All 17 pairs are byte-identical. J-lens provenance and label interpretation were checked in ~/jlens-persona-neurons (the J-lens study's repo, `$JLENS_ROOT`)/README.md, ~/jlens-persona-neurons (the J-lens study's repo, `$JLENS_ROOT`)/data/README.md, ~/jlens-persona-neurons (the J-lens study's repo, `$JLENS_ROOT`)/results/qwen3.5-4b/README.md, ~/jlens-persona-neurons (the J-lens study's repo, `$JLENS_ROOT`)/code/label.py, and ~/jlens-persona-neurons (the J-lens study's repo, `$JLENS_ROOT`)/code/stats.py.

Registered wording and rules were checked in PREDICTIONS.md, PART86-DESIGN.md, part86/RUNBOOK.md, and part86/FOLD1-NOTES.md through FOLD4-NOTES.md. The first-pass results/part86/prediction-score/SCORE.md and score86_predictions.py were read as the baseline. The research results/part86 tree and J-lens Qwen data and results trees were searched for second-reader agreement and top-20 verification artifacts.

## Second read, 2026-10-29 (added by Claude)

Amendment 2's second reader ran on the seed-8687 10% sample (335 of the first reader's 3,349 items; Opus 5.5 with
part86/READER-BRIEF.md; results/part86/second-reader/). Cohen's kappa on coherent persona or not is 0.157 (raw
agreement 71.3%), below the registered 0.6, so 86.2 to 86.6 remain NOT READ until a second full reading is done. The
disagreement is mostly one way: of the 288 items the first reader did not call a coherent persona, the second reader
called 71 coherent personas; of the first reader's 47, it agreed on 22. Not yet checked: whether the first reader used
the same brief and the same mapping from its label fields to "coherent persona" as the second. That check comes before
any conclusion about the readers or about the atlas built from the first reader's labels.
