# Part 57: fit a Jacobian lens to OLMo-2-1B and read the origin-axis neuron through it (registered 2026-10-22 01:43 CT, before any code or run)

Elise, 2026-10-22 01:43 CT: "It should be pretty easy to fit a lens to OLMo-2-1B if you can't find one" (github.com/anthropics/jacobian-lens). The Neuronpedia repo has no
OLMo-2-1B lens (checked on the Hub API listing; it has OLMo-3-7B only). The library fits a lens with jlens.fit on about 100 to 1,000 pretraining-like sequences of 128 tokens
(the README says quality saturates quickly; it calls the code an unoptimized reference whose time is dominated by the model's backward pass).
- Fit: allenai/OLMo-2-0425-1B-Instruct at the revision Part 26 pins, on 1,000 wikitext sequences of 128 tokens (the same kind as Neuronpedia's Salesforce-wikitext lenses; try 100 first for time),
  saved in the same .pt layout Part 42's loader reads. Device by what the build finds works: CPU, the A770, or a rented 3090.
- Validation before any read: (i) the fitted lens round-trips through part42.load_jacobian_lens; (ii) for 10 OLMo units the logit lens and J lens top-30 overlap is in the range Part 42 saw for Qwen3-1.7B (median 24 of 30 in the last quarter of layers; far lower early); (iii) the Part 42 positive control: a unit whose meaning is known (an OLMo unit from Part 26's battery, e.g. the Part 26 source-matched identity units, or the twin's neighbours) reads sensibly.
- Read: L10:6978 (the twin, 26.8's Earth-versus-off-world dial) and, with the Neuronpedia lens already on disk, Qwen3.5-4B L22:2742 (the alien neuron) and Qwen3-1.7B L19:4826 / L22:1610 again as the reference reads.

## Predictions (Claude)
| id | prediction |
|---|---|
| 57a | OLMo L10:6978's J-lens top 30 holds at least two of: alien, aliens, space, planet, star, galaxy, cosmic, extraterrestrial, starship. |
| 57b | Its J-lens bottom 30 holds at least one of: earth, human, humans, terrestrial, earthly, mankind (the Earth end of the dial it showed in 26.8). |
| 57c | Qwen3.5-4B L22:2742's J-lens top 30 holds at least two of the 57a words and its bottom 30 holds none of the 57b words (a push neuron, not a dial). |
| 57d | The fitted OLMo lens passes check (ii): median shared top-30 tokens between logit and J lens for units in the last quarter of layers is at least 15. |
| 57e | Fit time on whichever device works is under 2 hours for 100 sequences and the lens changes little at 1,000: top-30 overlap between a 100-sequence and a 1,000-sequence fit for 10 units is at least 20 of 30 on median. |
