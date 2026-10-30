# Front-page cold-check fold, round 1

Each line records the disposition of one cold-check item or caveat. Source reads and current prediction rows were checked before changing the front page.

## Numbered findings

- Cold check 1: Recomputed the F7–F11 prediction rows with `docs/tally_findings.py`; Finding 9 now includes Part 106b and reports the ledger's 42 held, 23 failed, 2 not tested, and 1 not read.
- Cold check 2: Finding 7 now says every all-push Part 86 rate is a lower bound and gives the miss-sample estimate of 2.7 to 6.4 times the counted pooled real-neuron total; it says the sample cannot establish the layer-22 ratio.
- Cold check 3: Finding 11 now names L6 N753, L7 N1169, L11 N8868, and L29 N776 as Part 110's four maker neurons, and describes L27 N3325 separately.
- Cold check 4: Finding 8 says the cold and post-introduction counts are six slot units, identifies the dog as the only in-character activity answer, and the short version links 1 of 6.
- Cold check 5: Finding 7 explains why layer 22 ran first, gives the 4,949/9,728 dry-run count, says the kept rule changed after counts were seen but before blind reading, and records that version 1 failed the agreement gate before the brief was revised.
- Cold check 6: Finding 8 adds the random-direction labels on both self questions, 101/343 and 94/344, and says the slot label may partly capture generic self-description.
- Cold check 7: Finding 10 places the real-pair versus random-pair break counts, 357/741 versus 286/741, beside Fisher p=0.000241; its heading no longer calls the interaction sizes similar without qualification.
- Cold check 8: Finding 10 says Part 119's thresholds 119.6 and 119.9 were unreachable, rather than counting them as failed predictions.
- Cold check 9: Finding 11 calls the direct-write ranking close to the readout that defines a flip, reports the Paris result after zeroing 100 neurons, and says ordinary answers can change at these group sizes.
- Cold check 10: Finding 11 says the Part 115b plain-text instruction and openings followed probe passes 2 and 3, so the bold-marker check was post-probe.
- Cold check 11: Finding 7 uses “match” for the J-lens result, says about 9 in 10 persona flips did not show the neuron's own words, and separates Part 98's graded score from a match-rate claim. Part 97's validation is treated as held because the current ledger and validation score mark it held, despite stale wording in its README.

## Smaller corrections and skeptic objections

- Part 97 status: The page retains 2 held and 1 failed prediction; current ledger and validation score support 97.3 as held, while the stale README claim is recorded in the source audit.
- Part 86 limit: The Limits section identifies Part 86 as a one-dose screen; Part 117b is named for the separate dose check.
- Finding 9 selection: The heading now says stage counts differ among units selected at Instruct, and the early/late split is identified as post hoc and not significant (Fisher p=0.09).
- Finding 9 checkpoints: The Part 108 figure caption distinguishes actual SFT, DPO, and Instruct checkpoints from constructed forward weight mixes, which are not checkpoints.
- Finding 10 support: The interaction comparison states that 16,208/19,532 real positions were retained and 17% were dropped to match control support.
- Finding 10 random controls: The answer-break gap is shown directly, so random pairs are not presented as a stand-in for real-pair break rates.
- Finding 10 unregistered subset: The 1.66 factor is explicitly attributed to an unregistered resolved-position subset.
- Finding 10 stopped rows: The page names the four Part 119b prediction rows stopped by Gate 3 and records that two Part 119 robustness thresholds could not be reached.
- Finding 11 ordinary answers: The Paris example is limited to the exploratory check with no matched controls and is not called a general-ability test.
- Finding 11 ranking: The page states the direct-write ranking is close to the flip-defining Q readout, rather than treating the resulting flips as independent evidence.
- Finding 8 selection battery: Part 109b's “Describe your perfect day.” selection question is identified as resembling its selection battery; the source punctuation is preserved.

## Figure, terms, and summary checks

- Figure 86: The caption gives the observed layer-24 peak of 5.219% and says its Wilson interval overlaps the random-direction interval.
- Figure 108: The caption reports plotted denominators from 62 to 64, consistent with the bars rather than the axis's fixed 64 label.
- Figure 116: The caption says the plot counts phrasings where Q was no longer first, matching the plotted complement.
- Figures 117 and 119: Their labels and captions matched the source; no change was needed.
- Terms: Added `## Terms` immediately after `## The short version`, with one plain definition for each term in the cold-check list, including κ, Fisher, Wilson, dose/alpha, greedy, MLP input, both projection/lens terms, Base, training stages, atlas/brief/triage, Gate 3, number formats, nats/log-odds, held-out phrasings, forced opening, zeroing/ablation, withheld, erasure reading, coherence gate, self-claim, OLMo, Qwen, and L27 N3325.
- Short version: Lines 1–6 use results already described above; lines 7–11 now state concrete model, intervention or question, sample, and linked source without claiming more than the finding.
- Limits and source disagreements: The Part 89 assistant-register movement and the split between the Part 119 written-rule verdict and scorer status remain explicit in the source audit.

## Prediction tally output

The script counts each row once under its Part heading by its last-cell verdict. Its output follows.

```text
Source: PREDICTIONS.md. One row counted once under its Part heading by its last-cell verdict.
Finding 7 (Parts 86, 97, 98, 112): 18 held, 13 failed, 1 not scorable
  Part 86: 7 held, 7 failed, 1 not scorable
  Part 97: 2 held, 1 failed
  Part 98: 7 held, 3 failed
  Part 112: 2 held, 2 failed
Finding 8 (Parts 92, 92b, 109, 109b, 109c, 112, 113, 117, 117b): 27 held, 25 failed
  Part 92: 3 held, 2 failed
  Part 92b: 2 held, 3 failed
  Part 109: 2 held, 1 failed
  Part 109b: 3 held, 1 failed
  Part 109c: 5 held, 2 failed
  Part 112: 2 held, 2 failed
  Part 113: 4 held, 2 failed
  Part 117: 4 held, 4 failed
  Part 117b: 2 held, 8 failed
Finding 9 (Parts 89, 90b, 91, 92, 94, 94b, 94c, 95, 96, 99, 104, 105, 106, 106b, 107, 108): 42 held, 23 failed, 2 not tested, 1 not read
  Part 89: 2 held, 5 failed
  Part 90b: 4 held
  Part 91: 3 held, 1 failed
  Part 92: 3 held, 2 failed
  Part 94: 4 held
  Part 94b: 1 held, 1 failed
  Part 94c: 3 held, 2 failed
  Part 95: 5 held
  Part 96: 4 held
  Part 99: 2 held, 2 not tested
  Part 104: 3 failed, 1 not read
  Part 105: 2 held, 1 failed
  Part 106: 1 held, 4 failed
  Part 106b: 2 held, 1 failed
  Part 107: 2 held, 1 failed
  Part 108: 4 held, 2 failed
Finding 10 (Parts 119, 119b, 119c): 11 held, 10 failed, 4 stopped
  Part 119: 2 held, 8 failed
  Part 119b: 5 held, 2 failed, 4 stopped
  Part 119c: 4 held
Finding 11 (Parts 110, 111, 114, 115, 115b, 116): 24 held, 7 failed
  Part 110: 5 held
  Part 111: 3 held, 1 failed
  Part 114: 2 held, 2 failed
  Part 115: 4 held, 2 failed
  Part 115b: 5 held, 2 failed
  Part 116: 5 held
```
