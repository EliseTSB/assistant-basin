# Part 136: the main run did not happen

Part 136's main run did not happen and its concept against slot comparison is withdrawn, because its pilot failed two registered gates; Part 142 is the successor.

The pilot passed Gate 1: the do-nothing hook reproduced the plain model bit for bit in one batch. It failed Gate 2, the known-unit check, by hand, and Gate 3, the off-target check. The scorer's file shows Gate 2 as pending because no gate2.json was written; the hand read below is the result. Gate 4 cannot be read on a pilot: its 0.049 is 22 units of 446, and 22 of the 24 units that ran had 24 firings.

## Gate 2: known unit

By hand, 7 of L22 N2742's 24 top firings sit on alien, extraterrestrial or a legal alien, against the registered threshold of 12: five on " alien" and two on a fragment of the word, "ater" in extraterrestrial and "ens" in Aliens. The scanned 8,000 documents contain 24 documents with those words: 5 wikitext and 19 UltraChat ([count-alien-documents.py](count-alien-documents.py)), with one firing per document. The registered bar therefore asked for half of those 24. I did not count this document total when registering the gate. The table gives all 24 firings in the pilot file. “Word” is the firing token; contexts are brief excerpts around it. Chat means UltraChat; wiki means wikitext.

| Activation | Source | Word (firing token) | Context |
| ---: | --- | --- | --- |
| 24.25 | chat | ater | advanced extraterrestrial civilization |
| 23.125 | chat | of | some kind of physical phenomenon |
| 19.375 | chat | alien | save the world from an alien invasion |
| 17.375 | chat | alien | save his daughter from an imminent alien invasion |
| 17.125 | chat | ens | fight against Aliens or Zombies |
| 15.875 | chat | . | imagined physical traits and complex societal structures |
| 15.875 | chat | alien | humans who discover an alien civilization |
| 14.25 | wiki | alien | fictional alien . This was |
| 14.1875 | wiki | 1 | Worlds (broadcast October 30, 1938) |
| 13.6875 | chat | 0 | fiction short story of at least 1,000 words |
| 13.1875 | wiki | of | search of planets |
| 12.5625 | chat | alien | haunted by supernatural entities or alien creatures |
| 10.0625 | wiki | , | girl from Venus, who lands on |
| 9.0625 | chat | on | planet they've landed on |
| 8.75 | chat | Black | Men in Black 3 |
| 7.59375 | chat | stellar | interstellar travel |
| 7.09375 | chat | an | discovers that an asteroid is on |
| 7.0625 | wiki | a | artifact was indeed a vase |
| 7.03125 | wiki | that | scientists that closely orbiting |
| 6.9375 | chat | space | blasting into space like an astronaut |
| 6.71875 | wiki | a | P-44 was indeed a planet |
| 6.59375 | chat | if | astronomers have wondered if the sun's |
| 6.5625 | wiki | 4 | WASP-4b. The |
| 6.21875 | chat | a | Superman has a deep compassion |

This list is not a finding about what the neuron means.

## Gate 3: off-target effect

The registered limit was 0.02 in every band. The pilot values in bands L0-22, L23-23 and L24-31 were 0.0209, 0.0228 and 0.0220, so the gate failed. The median absolute effect in those bands was 0.079, 0.086 and 0.063 log-probability, over 408, 336 and 360 firings ([effect-size-check.py](effect-size-check.py)), the readout the design's amendment asked for beside this gate; the floor and the regression below come from [gate3-floor-check.py](gate3-floor-check.py). The vocabulary-shuffled floor was 0.0112, 0.0107 and 0.0102. A shuffled floor destroys token structure, so it is not a full null. Regressing the band's mean effect out in sample raised the values to 0.0275, 0.0407 and 0.0362; that adjustment was not used. Alden's advice in council topic `overnight-ideas-2026-10-31`, messages 8 and 9, was to keep these failures and write a successor rather than move the cutoff.

## Gate 4 and timing

The pilot Gate 4 value is 0.049: its denominator is all 446 units, while only 24 ran. Timings from the pilot timing file:

| Stage | Batch | Rows | Seconds |
| --- | ---: | ---: | ---: |
| Scan | 8 | 1,080 | 1,010.26 |
| Effects | 8 | 0 | 231.02 |
| Steering | 8 | 45 | 50.94 |
| Scan | 16 | 1,104 | 953.76 |
| Effects | 16 | 0 | 192.20 |
| Steering | 16 | 46 | 34.84 |
| Scan | 32 | 1,032 | 760.47 |

Batch 32's effects stage stopped at the job runner's 2 GB RAM guard, at 03:40:26 CT, from the job's progress log, not stored here. Separately, as Part 142's Amendment 1 says, the scan's selector could not have taken all 446 units at the time.

## Registered predictions

Every prediction is not read because the comparison is withdrawn. The wording below includes the registered amendment to 136.1 and 136.5.

| # | Prediction (verbatim) | Status |
| --- | --- | --- |
| 136.1 | Matched poles' median stability in the top band is at least 0.1 above the bottom band. | withdrawn — not read |
| 136.2 | Within bands, concept units' median stability is at least 0.05 above slot units' (Effect A). | withdrawn — not read |
| 136.3 | Sorted units' median stability is at least 0.05 above their matched poles' within bands. | withdrawn — not read |
| 136.4 | Within bands, concept units' steering agreement is at least 0.05 above slot units'. | withdrawn — not read |
| 136.5 | The bottom band's median steering agreement is at most 0.2. | withdrawn — not read |
| 136.6 | Effect B gives the same direction as Effect A on 136.2 and 136.4. | withdrawn — not read |

Cold check: Claude Haiku 5.5, a fresh reader with the files, 2026-10-31; seven findings, six folded. The seventh, that the design's table still shows 136.1 and 136.5 before their amendment, is left as registered.
