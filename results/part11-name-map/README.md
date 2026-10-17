# Part 11.10-11.13: what SmolLM3-3B's name neurons do on general text

Elise, 2026-10-16 17:50 CT: what do these neurons code beyond identity; map them out. Predictions and scoring rules were
pushed first (ce07504). Run 18:56-19:02 CT: `map_named_gpu.py` on the A770 (contexts), Luna 6 high describing the 56 new
units blind to arm (`judge.py`, CLIProxyAPI), Jev (`jev-latest`) on 2,876 held-out contexts with 0 failed calls,
`name_map.py` scoring.

Units: **M**, the 28 neurons whose zeroing took self-naming from 47.9% to 4.2% (11.6); **MC**, 28 random neurons from the
same layers; **I** and **C**, Part 11.3's 20 identity neurons and their 20 controls, with 11.3's descriptions unchanged.
A name neuron's pole is the sign of its input to `down_proj` at the last token of `raw_nosys` + "My name is" (19 of 28
positive; the strongest one, L35.9662, is at -5.1). The GPU run reproduced 11.3's CPU contexts for the 40 old units:
median 19 of 20 identical, minimum 18.

## Results, against the pushed rows

| # | Row (verbatim) | Status | Result |
|---|---|---|---|
| 11.10 | On general text the 28 name neurons (11.6's set) are about names, not about this model: at least 14 of Luna's 28 blind descriptions match the name rule, and at most 3 match the self rule. | **failed** | 12 of 28 match the name rule, 0 the self rule |
| 11.11 | That is specific to them: the share of name-rule descriptions among the 28 name neurons beats their 28 same-layer random controls by at least 20 points. | held | 43% against 18% (5 of 28) |
| 11.12 | Luna's descriptions predict where both named sets fire on held-out text: Jev's median per-unit AUROC (strongest vs random held-out contexts) is at least 0.75 for the 28 name neurons and for the 20 identity neurons. | held | name 0.84, identity 0.78 |
| 11.13 | Neither named set is more predictable than random neurons from its own layers: each set's median AUROC is within 0.10 of its controls' median. | **failed** | name 0.84 against 0.72 (+0.125); identity 0.78 against 0.74 (+0.04) |

"One job" (Luna's verdict): name 23 of 28, their controls 16 of 28; identity 11 of 20, their controls 8 of 20.

## What the name neurons do (my reading of the descriptions, not pre-registered)

| Group | Neurons | Share of the name attribution | Held-out AUROC |
|---|---|---|---|
| Naming: "called", "known as", "named", aliases, names in lists | L16.9892, L18.6000, L20.4247, L21.907, L22.715, L27.3156, L32.6465, L35.142, L35.7065 | 25% | 1.00 for L20.4247, L21.907, L22.715 |
| Numbers: counts, measurements, prices, years | L34.4996, L34.7259, L35.505, L35.4121, L35.9662 | 26% | 0.86-1.00 |
| Phrase and clause boundaries: commas, articles, transitions, word starts | L33.5609, L34.4703, L34.5661, L34.7401, L34.8472, L35.8168 | 21% | 0.59-0.96 |
| One topic of the corpus (Egyptian religion, a video game, ships) | L24.5773, L25.3197, L34.7820 | 9% | 0.41-0.86 |
| No single job | L9.399, L22.202, L32.5675, L33.7066, L34.2680 | 19% | 0.47-0.67 |

None is about the model, AI or introducing oneself. The middle-layer neurons (16-22) are general naming machinery: they
fire wherever text introduces a name, and Luna's descriptions predict them perfectly on new text. The strongest
contributors in the last layers are number neurons, at the pole they take after "My name is"; one reading, untested, is
that "My name is ___" is treated as a slot for a value, like "the total is ___". The name rule is crude: two of its 12
hits are "no single job" descriptions that mention names, and the number neurons do not match it.

The identity neurons (11.3) stay what 11.3 found: grammar and discourse units ("his", "all", "way", clause boundaries,
numbers in formatting), predicted about as well as random neurons from their layers. The two sets share no neuron.

`map.md` has every unit: arm, layer.neuron, pole, Luna's verdict and description, the name rule, and both AUROCs.
