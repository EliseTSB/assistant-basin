# Part 38: word neurons that write meaning — do they know which sense, and does pushing them carry the idea?

Written 2026-10-20 15:49 CT, before any code or run, at Elise's go. Model: Qwen3-1.7B (revision 70d244cc), the Part 35
screen model. Prior art (Muse, council #80): Gurnee et al. 2023 (early-layer multi-token word neurons, no homograph
test), Dai et al. 2022 and Yona et al. 2026 (knowledge/entity neurons), Davies 2025 (vocabulary projection finds
semantic neurons), Neo, Cohen and Barez 2024 (disambiguation is compositional). No published neuron-level homograph
test for these units, and no published vocabulary projection for them.

Units (Part 35 rank-1 under the median floor; what they write from the exploratory projection, af12ecb):
- semantic writers: wizard L19:4826 (spell, cast), train L21:2307 (bus, depart), bee L24:5951 (hive, queen),
  star L9:3096 (astronomers, celestial), mountain L22:2428 (climbing, summit)
- spelling writers: trump L21:3616 (trump, tr, rick), volcano L24:3404 (Vol, vol)
- early, unreadable by projection: claude L1:4013, anthropic L2:5500

## A. Homograph contrast (reads)
For each unit, 12 sentences using the word in the screened sense and 12 in another sense, same position and similar
length: wizard (sorcerer / setup wizard), train (railway / to train a model or for a race), bee (insect / spelling bee),
star (in the sky / film star), mountain (landform / "a mountain of paperwork"), trump (Donald Trump / trump card,
"trumped"), claude (the assistant / Claude Monet, Claude Shannon), anthropic (the company / the anthropic principle),
volcano (the landform / figurative "a volcano of anger"). Measure: the unit's maximum activation over the word's own
tokens. Ratio = median(other sense) / median(screened sense), bootstrap CI over sentences.
Controls: the same measurement on 20 random units of the same layer (their ratios give the null spread).

## B. Push (writes)
Clamp the unit to k times its natural maximum on neutral passages (k = 4 and 8) on generated tokens only, for 16
neutral prompts (describe a kitchen, plan a weekend, etc.), 4 samples each at temperature 0.7, 80 tokens. Score: share
of answers containing a word from the unit's fixed topic list (written below before any generation), against 20 random
same-layer units at the same k, and against no clamp. Fluency: the share of answers a blind reader marks as broken.
Topic lists: wizard {wizard, spell, magic, wand, sorcer, enchant, potion}; train {train, rail, station, platform,
depart, ticket, carriage}; bee {bee, hive, honey, pollen, buzz, queen}; star {star, sky, galaxy, astronom, constellat,
celestial}; mountain {mountain, summit, climb, peak, hik, slope}; trump {trump, president, donald, maga}; volcano
{volcano, lava, erupt, magma, ash}.
