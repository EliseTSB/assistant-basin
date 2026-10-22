# Neuron interactions read as epistasis (written 2026-10-21 19:25 CT)

Elise, 2026-10-21 19:20: "A lot of the multi neuron effects were seeing remind me of epistasis, I wonder how applicable
terms, theory, and methodologies from quantitative genetics are applicable here." Her go to write it up and use it in
Part 49: 19:25.

Prior art (one search, 2026-10-21): Walsh decompositions and epistasis are standard for protein and genotype fitness
landscapes (Walsh-Hadamard sparsity, Broad/Dyno; the "epistatic transformer"; visible neural networks for SNP epistasis,
van Hilten et al.), and "Beyond Tokens" (arXiv 2608.24953) applies Walsh energy by order to a protein model's learned
representations. None found applying it to interactions between a language model's own neurons. Shapley-interaction
work in ML is the nearest neuron-side tradition (not checked here).

## The mapping

| genetics | here |
|---|---|
| locus, allele | a unit (neuron pole), and its state (natural, zeroed, clamped at +k or -k) |
| genotype | the vector of unit states in one run |
| phenotype | the measured readout (topic rate, a log-probability contrast, a claim rate) |
| additive effect | a unit's effect alone |
| epistasis | the joint effect minus the sum of the single effects |
| magnitude / sign epistasis | the pair changes the size / the direction of a unit's effect |
| modifier locus | a unit with no effect alone that changes another's (Sauers' L34.N11760 on L8.N344) |
| genetic background | the prompt and context |
| penetrance, expressivity | the share of answers a push changes; how large the change is when it does |
| pleiotropy | one unit moving several unrelated readouts |
| linkage disequilibrium | units that co-fire across contexts, so a screen's top hit may only ride along with the cause |

## Methods that carry over

1. **Fix the scale first; report two.** Epistasis on a log scale can vanish on a linear one and the reverse. Part 43's
   interaction from a log-sum-exp score with nothing underneath (Alden) is this. Every interaction is reported on the
   raw rate (or probability) and on log odds, with the scale used for the claim named in the design.
2. **Walsh / Mobius decomposition of full factorials.** A 2^k design splits into the mean, k additive terms, pairwise
   terms and higher orders; the share of variance by order says how additive the units are.
3. **Background-averaged epistasis.** Estimate the interaction per prompt and report its mean and spread across
   prompts, not one pooled number; a sign that flips across backgrounds is the finding, not noise to average away.
4. **Fine-mapping against linkage.** Before calling a screened unit causal, test it with its most correlated
   neighbours held fixed (conditional analysis); Part 26's twin and Part 35's name detectors were linkage-shaped.
5. **Penetrance and expressivity separately.** A push that changes 10% of answers completely is a different unit from
   one that changes every answer slightly; count both.

## Applied to Part 49 (analysis added before any result was read; the job started 19:24)

For each partner P: the interaction I = rate(P+W) - rate(P) - rate(W) + rate(none), on the raw topic rate and on
log odds (with 0.5 added to counts), each with a bootstrap interval over prompts; and per-prompt interactions with
their spread. Prediction 49.3's 15-point threshold stays as written (it is on the raw rate, P+W minus P alone).
