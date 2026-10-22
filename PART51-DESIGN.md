# Part 51: L22:1610 pushed up on everyday text, and whether any push tips the stock names

Written 2026-10-21 20:32 CT, before any run. Elise's yes (2026-10-21 20:30 CT) to both leads from Part 49/49b's side effects.
Model, revision, clamp mechanics, sampling (temperature 0.7, 80 new tokens, 4 samples per prompt) and natural maxima:
exactly Part 49b's (Qwen/Qwen3-1.7B @ 70d244c, clamp on generated tokens at k times the unit's natural maximum).

## 51a: does L22:1610 push everyday text toward the magical?
- Prompts: 16 non-fantasy prompts, data/part51a-everyday-prompts.json (written before any run).
- Arms (64 answers each): no clamp; L22:1610 at +8; L22:1610 at +4; Part 49b's three random L22 units (569, 3102,
  5258) at +8.
- Measure: magic-word density per 1,000 words, Part 49b's six stems plus "arcane", "mystic", "sorcery" (stems:
  spell, magic, wand, sorcer, enchant, potion, arcane, mystic), excluding stems in the prompt; per-prompt means,
  bootstrap interval over prompts (10,000 resamples). Difference from no clamp, not a ratio (the baseline is near 0).
- Broken answers: the blind reader, as 49b.4.

## 51b: are the default story names fragile attractors?
- Prompts: 16 story prompts that ask the story to begin with the main character's first name,
  data/part51b-name-prompts.json.
- Name: the first word of the answer after stripping quotes, asterisks and a leading think block, if it matches
  ^[A-Z][a-z]+$ and is not in {The, A, An, In, Once, On, It, There, When, As, At, After, Every, One}; else NONAME.
- Arms (64 answers each): two no-clamp replicates with different sample seeds (A, B); six random units, seed 5101,
  two each at layers 6, 14 and 22 (one at +8, one at -8 per layer).
- Stock set: the five most common names in replicate A (ties broken alphabetically). Primary measure: share of
  named answers using a stock name, per arm, compared with replicate B (A chose the set, so A is biased upward).
  Also: NONAME share, and name entropy.
- What this cannot tell apart: names being especially fragile, versus pushes perturbing every free choice. A pushed
  arm that also breaks instruction following shows up as NONAME, reported beside the name result.

## Prior art (searched before the run)
- Hamilton and Mimno, "Elias in the Lighthouse, Again? Diagnosing Low Diversity in LLM Stories" (arXiv 2605.26492):
  20,000 stories from four models; Elias, Mara and Elara among eleven words in 88.3% of stories; the tokens are rare
  in pretraining text and common in preference data. No neuron pushes. That points at a follow-up we have the
  models for: base versus chat Qwen3-1.7B on the same name frames (not in this Part).
- CASPER (ACL 2026) on character variety. Neither tests whether internal perturbations break the attractor.

## Sizes
51a 6 arms x 64 = 384 generations; 51b 8 arms x 64 = 512; 896 in all, under the pilot line. Smallest effect worth
knowing for 51b: a 15-point drop in stock share; at about 64 named answers per arm, pooling six random arms gives
roughly 2.5 standard errors against one replicate.
