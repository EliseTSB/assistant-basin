# Part 197 cold review

Verdict: SHIP-WITH-FIXES

## Scope and method
Read the registered design, JSON results, README, plotting code, figure caption, and Part 197 prediction rows. No model, audio, or API access.

## Findings

### Initial findings
- README prediction verdicts match the registered thresholds and `results.json`: 197.1–197.5 held and 197.6 failed; the language for R and both readout contrasts follows the design's stated reading rules.
- The caption's split descriptions match `fig_layers.py`: style uses the held-out-speaker curve, while identity uses each speaker's held-out texts.
- Confirmed numerical mismatch in the all-readouts table: N's 95% upper endpoint rounds to 54.8% from JSON, not README's 54.9%.


## Completed findings

### 1. Numbers in README versus `results.json`
- Independently recomputed the four-fold means from `folds` and paired differences from per-fold scores. The mean values agree with `readouts` / `paired_differences` in JSON (zero difference except floating point noise of 5.55e-17 for K).
- The one mismatch is N's 95% CI upper endpoint in the “All readouts” table: README says **54.9%**; JSON is `0.5484991574117015` = **54.849915741...%**, which rounds to **54.8%** at one decimal. The lower endpoint 51.7%, mean 53.3%, and all four per-speaker values match.
- All other reported metric means, intervals, per-speaker values, paired contrasts, chance values, and the listed 197.5 layer values agree with JSON at displayed precision. This includes E0/H historical comparison (62.7%/51.8%), EN-N (+10.4, 8.6–12.4), EN-E0 (+1.0, 0.4–1.7), E0-K (+34.0, 31.9–36.1), R-N (-7.3, -9.5 to -5.2), gate values (largest normalized cross-product 1.8e-15, rank 9), shuffled-label controls (14.5%/14.1%), layer-0 speaker identity 63.1%, final embedding 33.3%, and the 30.6/31.0/32.4/32.0 layer 20–23 values.
- The 22-test count is not stored in `results.json`; `part197/test_part197.py` contains 22 test functions, including the planted-layer-0 end-to-end test. Tests were not run as part of this read-only review.

### 2. Predictions and registered reading rules
- Verdicts match the registered thresholds: 197.1 held (10.4 points, lower interval bound 8.6 > 0); 197.2 held (45.985%, though its interval extends to 44.4% and the README correctly notes this is a point threshold); 197.3 held (+1.0 <= 3); 197.4 held (28.7 <= 30); 197.5 held (best layer 0; each L20–L23 <=35); 197.6 failed (63.1 < 80).
- R is described as style remaining linearly readable after subtracting the training-fitted linear prediction from N. R-N is explicitly a readout contrast. EN-N is described as an improvement within the fixed linear pipeline, with a clear caveat against information-theoretic novelty over arbitrary functions of N. No sentence claims what the model uses.
- Speaker identity is explicitly described as held-out texts, not held-out speakers, and the limits section preserves that distinction. No reading-rule overclaim found.

### 3. Figure and caption
- `layers.png` visually agrees with `fig_layers.py` and the JSON values: left is the four-speaker held-out style curve with EN/N/R and chance references; right is the text-held-out identity curve with final-embedding and chance references. Layer 0 is highest in both curves.
- Caption accurately describes the style curve, the identity task, and the text split. It is incomplete about plotted horizontal references: the left panel also has chance; the right also has the final-embedding and chance lines. Recommend naming those references in the caption for a full description.

## Verdict rationale
SHIP-WITH-FIXES: correct the N upper CI endpoint from 54.9% to 54.8%, and expand the figure caption to identify the chance lines and the final-embedding reference. No blocking interpretation issue found.
