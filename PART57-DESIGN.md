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

## Result (scored 2026-10-22 03:2x CT)

Lens: 100 wikitext sequences of 128 tokens (seed 5701, corpus sha256 74da360f), fitted on the Arc A770 in bf16, dim-batch 8: 62.8 minutes, about 35 s per sequence (jacobian_lens.pt, 125.8 MB, not committed; metadata in results/part57/olmo2-1b/). The read ran in WSL against that file (results/part57/olmo2-1b/read.json). The Windows job's own read step died when a watcher of mine killed the job at the moment the empty output file appeared; the rerun is the one scored.

| id | result |
|---|---|
| 57a | HELD. L10:6978's J-lens top 30 holds cosmic, galaxy, planet and star (4 of the 9 listed words), and also galactic, spaceship, spacecraft, astronaut, jedi, milky, universe, stars. The logit lens alone holds alien, galaxy and star. |
| 57b | FAILED. The J-lens bottom 30 holds none of earth, human, humans, terrestrial, earthly, mankind. It reads urban, indoors, rooftop, roofs, waterproof, winter, watering, mornings, refreshed: an anti-space end made of ground-level and everyday-life words, not the word "earth". |
| 57c | HELD (the build report's read of Qwen3.5-4B L22:2742): top 30 holds alien, aliens, space, planet, extraterrestrial; bottom 30 holds none of the 57b words. |
| 57d | HELD. Median shared top-30 tokens between logit lens and J lens at layer 14 of 16 (the last quarter, 10 units: the Part 26 positive control and nine seed-5701 random units) is 18.5 (range 13 to 27; threshold 15). The target at layer 10 shares 4 of 30. |
| 57e | UNSCORED. Fit time half: 100 sequences took 62.8 minutes (under 2 hours). The 100-against-1,000 overlap half needs a 1,000-sequence lens, which would take about 10 hours at 35 s per sequence on the Arc; not run. |

Reading: the twin neuron writes toward the off-world end in the lens of the model it lives in, which is the direction Part 26.8's blind reader found. Its opposite end is not "Earth" as a word. The Earth-versus-off-world dial of 26.8 may therefore be a dial between space-and-sci-fi origin and ordinary-life origin, with "Earth" itself being what the model says when this unit is quiet. That is a reading of one lens fit on 100 sequences, not a test.
