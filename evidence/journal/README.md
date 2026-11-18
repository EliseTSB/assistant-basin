# Files cited from the Journal

Many designs, briefs and READMEs here cite a file in "the Journal": the private working repository where this research
is run. Those citations read like `Journal reviews/x.md`, `` `scratch/y/` in the Journal repo``, or a full path such as
`/home/elisepc/green/Journal/reviews/x.md`. Registered designs are not edited after registration, so the citations
stay as written; this folder makes them followable instead.

**Rule:** a cited Journal path `P` is at `evidence/journal/P` in this repository.

What is here (copied 2026-11-18): every cited cold review (`reviews/`), the reviewers' probe scripts and outputs
(`scratch/`), the GPU job runner a review cites (`scripts/neuron-gpu-job/`), and the cited research notes under
`notes/reports/`, including the full chicken study
([`notes/reports/embeddinggemma2-chicken-neurons-2026-11-03/`](notes/reports/embeddinggemma2-chicken-neurons-2026-11-03/README.md))
behind Parts 188, 190 and 193 and the [Chicken-o-Meter](../../chicken-o-meter/README.md). Left out of the copies:
model weights, stored activation arrays (`*.npz`), and audio files.

## Cited but not copied

| Cited path | What it is | Why it is not here |
|---|---|---|
| `sources/web-archives/2026-10-15-goodfire-neural-networks-already-clean.md` | archived page | third-party text; original: https://already-clean.goodfire.com/ |
| `sources/web-archives/2026-10-20-anthropic-graft-reward-hacking.md` | archived page | third-party text; original: https://alignment.anthropic.com/2026/graft/ |
| `sources/web-archives/2026-10-29-maxsloef-one-neuron-owl.md` | archived thread | third-party text; original: https://x.com/maxsloef/status/2115858482889118065 |
| `sources/web-archives/2026-10-31-aa-personalitybench.md` | archived page | third-party text; original: https://artificialanalysis.ai/evaluations/aa-personalitybench |
| `sources/web-archives/2026-11-01-mayner-persona-neurons.md` | archived thread | third-party text; original: https://x.com/wmayner/status/2116628249153515740 |
| `sources/web-archives/2026-11-11-sauers-say-do-subnetworks.md` | archived thread | third-party text; original: https://x.com/Sauers_/status/2120412679835951382 |
| `sources/web-archives/2026-11-12-sauersml-body-readme.md` | archived README | original: https://github.com/SauersML/body |
| `sources/web-archives/trumpstruth-2026-10-20/` | 188 archived posts (Part 35) | third-party text; the post URLs are listed in Part 35's data file; source: https://www.trumpstruth.org/ |
| `sources/style/sauers-magician-2026-10-20` | style reference images | third-party images; the style is Sauers' "Activation Magicians" figures |
| `deliberations/muse-neuron-research-2026-10.md` | a council discussion log | other participants' messages; Part 37 quotes what it used |
| `notes/reports/probes/self-portrait-probe-2026-08-27.md` | an earlier probe of Journal Claude | personal working notes; Part 209 summarises the point it uses |
| `notes/reports/publish-2026-10-22/` | front-page drafts and their figures | superseded drafts; the cold review Part 61-63 cites is on branch `cloud/cold-review-2026-10-22` |
| `ops/papercuts/2026-10-22/…`, `ops/papercuts/2026-11-04/…` | internal issue notes | operational notes; the briefs that cite them say what each was about |
| `scratch/review-p37-build-ds-r1-2026-10-25/scripts/`, `scratch/review-part19-circuit-r1-haiku-r1-2026-10-18/scripts/` | reviewers' probe scripts | not kept; the reviews themselves are under `reviews/` or `review/` |
