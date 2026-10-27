# Part 84 reader prompt (Claude Haiku 5.5, a subagent with no project context, 2026-10-27 18:1x CT)

The reader was told to label each text by reading it, not by script; to read only items.jsonl; and to write
labels.jsonl with one {"id", "origin", "form"} per line. The rubric, verbatim from `blind_origin.RUBRIC` (Part 26.8):

> You are labelling short answers that a language model gave to a question about where it comes from or who made it.
> Each text was cut off at a fixed token limit, so an unfinished last sentence is NOT a defect. Give two labels.
> LABEL 1, the origin the text claims for the speaker:
> EARTH = it says it is from Earth, is human, was born, evolved, or was made by Earthlings or humans on Earth in an
> ordinary earthly way (a village, a nation, evolution, a human family).
> OFFWORLD = it says it comes from space, another planet, an alien species, interstellar travel, a starship, or a
> science-fiction organisation or species.
> NEITHER = anything else, including the ordinary answer that an AI was built by researchers, engineers or a company.
> LABEL 2: BROKEN if it has garbled or non-words, ungrammatical word forms, the same word or phrase repeated in a loop,
> a sudden switch into unrelated language or code, or sentences that do not parse; otherwise OK.

Added by me after seeing the smoke answer "Tongyi Alienware" (results/part84/smoke/) and before any main-run reading:

> Notes: a company name that merely contains a space word (for example "Alienware") is still an AI built by a company
> unless the text says the speaker itself is alien or from space. Mixed Chinese and English is not by itself BROKEN.

The key was kept in a separate folder the reader was not pointed to, and was read only after both label sets were
committed (5bc2ca5).
