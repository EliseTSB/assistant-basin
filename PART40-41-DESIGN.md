# Parts 40 and 41: Ryan's two neuron questions (near-miss precision; concept vs discriminator across depth)

Written 2026-10-20 17:48 CT by Claude, before any code or run, at Elise's go ("Yes pre-register them please!").
Both questions are Ryan's, from the voice call of 2026-10-16 (23:08 CT): measure precision with near misses, not only
recall; and do broad concept neurons sit earlier and narrow discriminator neurons later?
Model for both: Qwen/Qwen3-1.7B 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e (the Part 35 and 38 model). No generation.

## Part 40: how precise are the neuron descriptions?
Run size: 43 concepts x 30 sentences = 1,290 forward passes.
Units: for each of Part 35's 43 concepts, the rank-1 unit of the gated median-floor screen
(results/part35/qwen-qwen3-1-7b/screen-floor-median.jsonl, the 2026-10-20 rescore); its "description" is the concept
name. Nulls: one random unit from the same layer per concept (seeded), and one random unit-norm direction in that
layer's down_proj input space per concept (Ryan: "random directions too"), read as a projection.
Sentences, authored per concept, about 12 words, the target word or its referent once:
- 10 positives: the concept in its ordinary sense ("The owl watched the field from a dead branch.");
- 10 near misses: a sibling or a confusable that a sloppy description would also cover: the same word in another
  sense where one exists (trump card, spelling bee, a mountain of paperwork), otherwise a close sibling (hawk for owl,
  witch for wizard, Monet for claude);
- 10 far misses: unrelated sentences of the same length.
Measured: each unit's max over the sentence's tokens. Per unit: AUC positives vs near misses, AUC positives vs far
misses, and precision at 10 (share of positives among the unit's 10 highest of the 30), with seeded bootstrap CIs.

## Part 41: concept vs discriminator neurons across depth
Run size: 7 families x 5 members x 12 sentences = 420 forward passes, every unit of every layer.
Families: magic users (wizard, witch, warlock, mage, sorcerer); animals (dog, cat, wolf, fox, horse); vehicles (car,
truck, bus, train, bicycle); fruits (apple, banana, orange, grape, pear); countries (France, Japan, Brazil, Egypt,
Canada); numbers (three, seven, twelve, forty, hundred); first names (Maria, John, Ahmed, Yuki, Pierre). Each sentence
names one member once.
Per unit, medians over sentences:
- broad unit for a family: every member's median at least half of its highest member median, and the lowest member
  median at least 4 times the median over the other six families;
- discriminator for a member: that member's median at least 4 times every sibling's and at least 4 times the
  out-of-family median.
Gate: counts per layer are compared with 200 label shuffles (members within family for discriminators, family labels
for broad units, seed 4101); only units above the shuffle 99th percentile of their own statistic count. Floor: a unit
whose highest member median is under the layer's 90th-percentile activation on the 420 sentences is not classified.
Reported: per layer, counts of broad and discriminator units, and the median layer of each kind.

Prior art (recorded with the predictions, Claude's memory, hedged; Muse asked to check): Huang et al. 2023 assess
natural-language neuron explanations by precision and recall with targeted test inputs; Bills et al. 2023 score
explanations by simulation; Gurnee et al. 2023 (sparse probing) report early layers in heavy superposition with many
n-gram-like units and more specific, monosemantic features in middle layers; SAE "feature splitting" (Bricken et al.
2023) is splitting across dictionary size, not across depth.

## Amendments before any run (Claude, 2026-10-20 18:51 CT)
1. Part 41's sentences share their frames across members: each member gets the same 12 sentence frames, only the
   member word changes (data/part41-sentences.json, build 2fbff75). This holds context fixed so broad versus
   discriminator is read from the member alone. It also means a pure word detector reads as a discriminator, which is
   the distinction 41.1 and 41.2 turn on, so the classification is unchanged and the reading stays as written.
2. Part 40's first sentence set reused one core sentence per kind with ten frames, and one far-miss set for every
   concept; it is being rewritten (10 different situations per kind, near misses mixing other senses and at least 3
   siblings, far misses drawn per concept) before any read. 40.1 to 40.4 are unchanged.
