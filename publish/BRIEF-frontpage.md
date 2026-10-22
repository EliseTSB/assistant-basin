# Brief: the front page (draft 1) for the public write-up

Domain: /home/elisepc/assistant-basin/publish (write only there; read the rest of /home/elisepc/assistant-basin). Do not commit.
Read first: publish/CLAIMS.md (the verified ledger), publish/REPORT-ledger.md, README.md (top 100 lines for tone and what exists),
PREDICTIONS.md (for the failed predictions), PRIOR-ART.md. Then read the result files named in the ledger rows you use; the
numbers in your text must come from those files.

Write `publish/FRONT-PAGE.md`, about 1,500 to 2,000 words, for a reader who knows what a language model is and has never heard
of this project (AI-curious engineers, interpretability people, journalists). It leads the public repo and a page on a personal
site. Structure, in this order (Elise approved the principle, Claude chose the order):
1. Two paragraphs: the question (where does a model keep its sense of being an AI assistant, and who it says made it), the
   method (find neurons with open models, push or remove them, compare with random sets of the same size, write every prediction
   down before the run), and an honest status line: small open models, our own prompts, mostly B-grade evidence, no result here
   is settled.
2. Finding 1: maker names live in a few late word-writing neurons. Parts 25, 27, 32 and 11 in CLAIMS.md. Give the numbers, the
   matched-random control, the Chinese exception, and the caveat that travels with each.
3. Finding 2: a model's name comes from its training data. Parts 8 and 9.
4. Finding 3: the alien neuron matters while the model writes, and removing it at natural strength costs nothing. Parts 18, 18b,
   24, 33.
5. Finding 4: the nulls, each stated as a result. Parts 49, 50, 17/17b, 34, 29c, 56 (and 1 for the judge-sensitivity point).
6. What failed in our own predictions: pick the three most instructive failed registered predictions from PREDICTIONS.md and say
   what we learned from each.
7. What is next, in three lines (the Qwen3-1.7B atlas: 172,032 neurons catalogued, labelled by a cheap model and checked by a
   sentence test; results not in yet; say "pending", never guess them).
8. A short "how to check us" section: the repo, the predictions file, the folder per Part, the commands to rerun one result.

Style rules (Elise's): plain words, short sentences (average under 17 words), lead each paragraph with its conclusion, no em
dashes, no "not X but Y" constructions, no marketing words (breakthrough, remarkable, fascinating), numbers with their n, every
claim next to its control and its caveat, Parts cited as (Part 25). Do not name any person as a critic. Credit prior work by
author and title only as the ledger's prior-art column gives it. Do not assert anything the ledger grades C as a finding; C items
may appear only as "suggestive". The word "circuit" is not used (we never mapped one).

Also write `publish/FIGURES.md`: for each finding, one figure spec (what is on each axis, the exact numbers and the result file
each comes from, error bars where the file has them, the takeaway as the figure title). Style target: clean infographic, one idea
per figure, the control always drawn beside the effect (see /home/elisepc/green/Journal/sources/style/sauers-magician-2026-10-20
for the reference look). Do not draw them; specify them.

Finish with `publish/REPORT-frontpage.md` (SUMMARY / FINDINGS / BLOCKERS): every number you used with its source file, and any place
the ledger and a result file disagreed.
