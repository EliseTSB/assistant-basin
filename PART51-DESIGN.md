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

## Prior art from Muse (council muse-part51-prior-art #4, 2026-10-21 20:41; filed 2026-10-21 20:44 CT, before any run)
- Nothing found on either question: no steering or neuron intervention measured at the level of default names, and no
  base-versus-chat comparison of name concentration.
- Closest: Olson et al. arXiv:2412.06060 (a creativity direction on Llama3-8B; diversity rated as a whole, never by
  name); Schapiro et al. arXiv:2607.01433 (weight steering reduces mode collapse on DAT/AUT; no stories, no names);
  IOI name-mover heads (names supplied in context, not invented defaults).
- Anchors confirmed: Hamilton and Mimno (EMNLP 2026; Elias 26.5%, Mara 16.7%, Elara 13.1% per coverage); CASPER is
  Brei et al. arXiv:2606.22454, measurement only.
- Muse's caveat, kept: 51b tests fragility, which says nothing about where the attractor comes from. Predictions
  unchanged.

## J-lens reads, added 2026-10-21 20:47 CT before any Part 51 run (Elise: "Anything more we should be doing with the J lens?")
jlens_units.py (Part 42's machinery, Neuronpedia's wikitext lens for Qwen3-1.7B) read each unit's write direction;
results/part51-lens/read-pre51a.json.
- L22:1610: logit and J lens both write spell, spells, magic, magical, enchant, caster, mage. Bottom 30 is noise
  (no scholarly words), so the lens does not predict the master/parchment rise seen under the negative push.
- L22:569 (Part 49b's R1): top itself/himself/herself/yourself; bottom the possessives (your, our, my, their, his).
  Checked on 49b's existing answers AFTER reading the lens, not registered in between (exploratory): at -8 R1 takes
  possessives from 11.9 to 39.1 per 1,000 words (23 to 41 answers of 64); R2 14.1, R3 12.8. The 49b "first-person
  false alarm" was this: a real possessive push, which a first-person-only count missed.
- L22:5258 (R3): depend/dependency/peer/trans. L22:3102 (R2): bottom JavaScript tokens; top noise.

## Lens predictions for 51a (the random units are pushed +8 there)
| id | prediction |
|---|---|
| 51a.L1 | R1 (L22:569) at +8 lowers possessives (my, your, our, their, his, her) per 1,000 words by at least 30% of no clamp. |
| 51a.L2 | R1 at +8 raises reflexives (myself ... themselves) per 1,000 words above no clamp, in at least 8 of the 16 prompts' means. |
| 51a.L3 | R3 (L22:5258) at +8 raises words starting depend/peer/trans per 1,000 words above no clamp. |

## 51c: L22:1610 on the two senses of "wizard" (registered 2026-10-21 21:42 CT, before running)
Elise and Ryan on the call (20:54-20:55): is L22:1610 about magic or about a person? Part 38 measured this for the
wizard word neuron (L19:4826: weak on the setup-wizard sense); L22:1610 has not been measured. part51c.py reads its
activation on Part 38's wizard sentences (data/part38-homographs.json, units[0]: sorcerer sense against setup-wizard
sense), at the "wizard" token and as the sentence maximum, beside 5 random L22 units (seed 5104), CPU, bf16.
| 51c.1 | Claude: the sentence maximum of L22:1610 is higher on sorcerer sentences than setup-wizard ones, Cohen's d at least 1.0. | HELD: 169.6 against 41.4, d 2.81. |
| 51c.2 | Claude: at the "wizard" token itself the gap is smaller than at the sentence maximum (the unit tracks magic content, not the word). | HELD, with a qualifier: at the token, 119.4 against 39.3, d 2.14 < 2.81. But it still fires about 39 on the software sense of the word, against a natural maximum of 5.3 on neutral text and under 1 for the random units, so the word alone drives it hard too. |
| 51c.3 | Claude: no random L22 unit shows a sentence-maximum gap with d at least 1.0. | FAILED: L22:3211 has d 1.40 (2.87 against 0.79, tiny values); the other four are under 0.8. |

51c reading: L22:1610 fires on the word wizard in either sense, and three times harder when the sentence is about magic. The random-unit failure shows that a d over 1 on 12 sentences per side is reachable by chance at tiny activations, so a d threshold alone is too weak a control; the raw size (about 40 to 170 against under 3 for every random unit) carries the result.

## 51a result (scored 2026-10-21 21:47 CT)
L22:1610 pushed up does not make everyday text magical: 0 magic words in all six arms (16 prompts x 4 samples each).
One likely reason, found by 51c the same evening: the push level is k times the natural maximum on neutral text
(5.31), so +8 is 42.5, while the unit itself reaches about 40 on the software sense of "wizard" and about 170 in magic
sentences. The push was at the level of the word, not of magic content. The J lens predicted R1's possessive drop
(81%), but not its reflexive or R3's depend-word changes: the lens's TOP tokens did not rise, its BOTTOM tokens fell.
Next: push L22:1610 at the in-context scale (about +170, k near 32) on the same prompts, with the dose ladder of
STANDARD-CONTROLS.md item 8. Broken answers (51a.4) wait on the blind reader.

## 51d: L22:1610 at its in-context scale (registered 2026-10-21 21:48 CT, before running)
Same prompts, samples, seeds and scorer as 51a; part51.py push51a --plan 51d. Arms: no clamp; L22:1610 at k 16, 32, 48
(85, 170, 255; 170 is its level in 51c's magic sentences); R1 and R2 at k 32. 6 x 64 = 384 generations.
| 51d.1 | Claude: L22:1610 at k 32 raises magic-word density by at least 3 per 1,000 words over no clamp, interval excluding 0. | FAILED: +0.69 per 1,000 words, interval [0, 2.07]. |
| 51d.2 | Claude: the rise grows from k 16 to k 32 (dose order on the rising side). | HELD: +0.23 at 16, +0.69 at 32, +2.13 at 48 (interval [0.70, 4.22], the only one excluding 0). |
| 51d.3 | Claude: neither random unit at k 32 raises magic-word density by more than 1 per 1,000 words. | HELD: both 0.00. |

## 51b result (scored 2026-10-21 22:10 CT)
The default names are strong (Lila and Lena take 40 of 64 named answers in both no-clamp replicates) and the six
random pushes did not move them: every random arm's top-8 list is nearly identical to replicate A's, whose sample seeds
they share. These randomly drawn units at +/-8 barely changed the text at all, so this tests "does a push that does
nothing move names" (no). The 49b observation came from pushes that did change the text (R1 cut possessives). Reading:
names are not especially fragile; they move when the text moves. A fair test of fragility would compare name change
against how much the rest of the answer changed, per push. Not run.

## 51e: L22:1610 pushed DOWN at the in-context scale (registered 2026-10-21 22:18 CT, before running)
Elise, 22:17: "Are you running positive and negative pushes on the magic neuron?" 51d is positive only; a negative push
reads where magic words are present (STANDARD-CONTROLS.md item 1), so 51e uses Part 49b's 16 wizard prompts
(data/part49b-wizard-prompts.json), same scorer (magic stems, prompt stems excluded): part51.py push51a --plan 51e
--data that file. Arms: no clamp; L22:1610 at k -8, -16, -32; R1 and R2 at k -32. 6 x 64 = 384 generations.
| 51e.1 | Claude: at k -32 magic-word density falls to at most 20% of no clamp, interval excluding no change. |
| 51e.2 | Claude: density falls in dose order: -8 above -16 above -32. |
| 51e.3 | Claude: neither random unit at -32 moves density by more than 30% of no clamp. |
| 51e.4 | Claude: the -8 arm reproduces 49b's drop within its interval (49b: 0.377 of no clamp, [0.105, 0.564]). |

51d reading: a small, dose-ordered rise that only clears zero at k 48 (2.1 magic words per 1,000, under one per answer set). Pushing the unit up does little to ordinary text; it amplifies magic where magic already is (51c) more than it creates it.
