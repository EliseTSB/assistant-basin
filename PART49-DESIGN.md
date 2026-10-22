# Part 49: does the wizard unit boost a partner? (written 2026-10-21 18:58 CT, before any code or run)

Elise's question (2026-10-21 18:57 CT), after Sauers' owl result (x.com/Sauers_, 2026-10-19): in Qwen3-8B, L34.N11760
makes no owl claims alone but nearly doubles L8.N344's (51 to 121 of 250). Our wizard unit, Qwen3-1.7B L19:4826, writes
"spell, cast" yet put a wizard word in 2 of 32 answers at every push strength 2x to 16x (Part 38, 38.7/38.8). It looks
like Sauers' booster. Is there a driver it boosts? Prior art: Sauers' post is the direct precedent; Part 38 is ours.

Model, revision, prompts, clamp and scorer: exactly Part 38's open-prompt push (16 open prompts in
data/part38-homographs-open.json, 4 samples each, clamp on generated tokens at k times each unit's own natural maximum,
Part 38's wizard topic list and its broken-answer reading rule: a topic rise that comes only with broken answers is noise).

Units: W = L19:4826. Partners P1..P5 = the five highest-ranked units on the Part 35 wizard screen of Qwen3-1.7B other
than W. If that ranking cannot be found on disk, the five units (layers 4 to 27, not W) with the highest median
activation on Part 38's 12 wizard sentences relative to its 12 control-role sentences, computed and committed with
their scores BEFORE any push.

Arms (k = 8 for every clamped unit, 64 answers each): no clamp; W alone; each P alone (5); each P with W (5); 3 random
pairs (a random unit at a partner's layer, with W). 15 arms, 960 generations, the pilot. A full run (more prompts, more
k) only if a pilot pair shows a boost.

## Amendment 1 (2026-10-21 19:07 CT, before any code was run): negative pushes (STANDARD-CONTROLS.md item 1)
Added arms on 8 wizard-topic prompts (data/part49-wizard-prompts.json: e.g. "Tell me a short story about a wizard."),
4 samples each, 32 answers per arm: no clamp; W at -8; each P at -8 (5); W and each P together at -8 (5). 12 arms,
384 generations; total 1,344. The open-prompt arms stay as designed.

## Amendment 2 (2026-10-21 19:21 CT, before any push): partner selection
No Part 35 wizard ranking exists on disk, so the fallback ran. It used Part 38's homograph sentences (12 wizard as a
sorcerer, 12 "setup wizard") rather than the role nouns, which asks the sharper question (the magic sense, not the
word). Its first ranking was dominated by layer-27 units whose values were large and negative on both senses (for
example -682 against -922); a negative maximum clamped at 8x is a large negative write, not a partner. The rule now:
positive pole only (median on the wizard sense above 0, and the control median under half of it), layers 4 to 26,
at most two per layer. Partners (data/part49-partners.json): L22:1610, L26:3130, L24:4151, L26:1877, L22:2295.
The first ranking is kept in git history (the build commit's parent tree is not needed; the rule above re-derives it).

## Part 49b (2026-10-21 19:47 CT, after Part 49's results, before any 49b code or run): the negative L22:1610 candidate, confirmed or not
Part 49's exploratory reading: on wizard prompts, magic words other than "wizard" fell from 47.7 to 20.1 per 1,000
words with L22:1610 at -8. 49b tests it on FRESH prompts with the measure fixed in advance.
- Prompts: 16 new wizard-topic prompts (data/part49b-wizard-prompts.json, written before any run, none from Part 49).
- Measure: per answer, matches of {spell, magic, wand, sorcer, enchant, potion} per 1,000 words, excluding any word in
  the prompt; arm mean with a bootstrap interval over prompts; broken-answer share by the blind reader.
- Arms (k = -8, 4 samples per prompt, 64 answers each): no clamp; L22:1610; W (L19:4826); L22:1610 with W; three
  random L22 units chosen by seed (positive-pole, same selectivity rule's sign condition not required). 7 arms, 448.
