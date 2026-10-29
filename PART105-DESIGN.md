# Part 105: does the opposite-direction attention swap also lose Part 92's original slot units?

Registered 2026-10-29 by Claude (Opus 5.5), after Part 104's results and before any Part 105 run. The commit that adds
this file is the registration.

## Question

Part 103 found that SFT's attention (all 16 layers) loses 7 of Part 92's 10 shown slot units in layers 1 to 3 and none
of the 9 in layers 4 to 12. Part 104 did not replicate that split on 43 new slot units. There, the extrapolated
attention (2 × Instruct − SFT: the same distance from Instruct, opposite direction) lost as many units as SFT's own
attention (15 each). That fits fragility: any attention change of this size loses units.

But Part 104's units came from a less specific part of the pool, so the failed replication may be a population
difference. Part 105 runs the extrapolated swap on Part 92's original units, so the two readings separate on the units
where the split was first seen:

- **D (dependence).** The original early units need what post-SFT training put into attention. The extrapolated swap,
  which keeps and doubles that change, keeps them.
- **F (fragility).** Any change of this size loses them. The extrapolated swap loses about as many as SFT's attention.

Prior art: as in PART104-DESIGN.md (ExPO, arXiv 2404.16792; Wu et al., arXiv 2310.00492), plus Part 104 itself.

## Run

One new run, `attnEXT_all`: Part 92's 21 slot and 18 concept units, Instruct host, chat frame, Part 99's fixed pushes
(`results/part99-pushes.pt`), `part99/swap99.py --swap attn --donor <SFT> --extrapolate --only concept,slot`, batch 48,
as Part 103's attnSFT_all except for `--extrapolate`. It is compared with Part 99's inst_none and Part 103's attnSFT_all
labels on the same units. Labels: Part 92's blind reader, two passes, on Part 101's read set. Scorer:
`part105/score105.py`, committed before any Part 105 label is read.

"Shows" and "lost" are Part 99's definitions as in Part 104. Early is layers 0 to 3 (Part 92's slot units sit in layers
1 to 3 and 4 to 12). The 10 shown early slot units are the ones Part 103's split used.

## Predictions

- **105.1 (D on the original units).** attnEXT_all loses at most 2 of the 10 shown early slot units, and fewer than
  attnSFT_all (which lost 7 of them): one-sided exact McNemar p < 0.05 on those 10.
- **105.2 (the extrapolated model still answers).** attnEXT_all's pushed "Who are you?" answers are coherent for at
  least 14 of the 21 slot units (attnSFT_all: 17 of 21, minus 3). Part 104 showed that a fixed 80% bar can fail SFT's
  own swap, so this bar is relative to SFT's.
- **105.3 (concept units survive).** At least 13 of the 18 concept units show their persona in at least one of the six
  answers under attnEXT_all, as in Parts 101 to 103.

What the outcomes mean: 105.1 holds (with 105.2) → the original early units need the post-SFT change, and Part 104's
failure is a population difference. 105.1 fails with attnEXT_all losing 5 or more of the 10 → fragility explains Part
103's split too, and the stage reading of Parts 102 and 103 should be treated as sensitivity to change size. Between 3
and 4 lost is reported as unresolved.

## Sizes

39 units × 2 (neuron and control) × 6 questions + 6 unpushed = 474 generations at 120 tokens, about 2 to 3 minutes on
the A770 as in Part 103. Labels: Part 101's read set, about 260 items, two passes.

## Limits stated now

One model, one push per unit, one reader, one extrapolation step size, attention only; 10 early units, so a one-unit
change moves a rate by 10 points.

## Results (2026-10-29)

105.1 failed (the extrapolated swap lost 5 of the 10 early units; McNemar against attnSFT_all p = 0.34), so by the
registered reading the loss is fragility (F); 105.2 held (18 of 21 coherent); 105.3 held (17 of 18 concept units).
Details: `results/part105/README.md`.
