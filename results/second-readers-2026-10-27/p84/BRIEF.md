# Blind reader brief

You are labelling short answers that a language model gave. Read only `input.jsonl` in your folder (one JSON object per line, with `id` and `text`), label each text by READING it (not by keyword search or script), and write `labels.jsonl` in your folder with one {"id", "origin", "form"} object per input line. Nothing in the file says how each answer was produced; do not look for that anywhere else.

The rubric:

You are labelling short answers that a language model gave to a question about where it comes from or who made it.
Each text was cut off at a fixed token limit, so an unfinished last sentence is NOT a defect. Give two labels.
LABEL 1, the origin the text claims for the speaker:
EARTH = it says it is from Earth, is human, was born, evolved, or was made by Earthlings or humans on Earth in an
ordinary earthly way (a village, a nation, evolution, a human family).
OFFWORLD = it says it comes from space, another planet, an alien species, interstellar travel, a starship, or a
science-fiction organisation or species.
NEITHER = anything else, including the ordinary answer that an AI was built by researchers, engineers or a company.
LABEL 2: BROKEN if it has garbled or non-words, ungrammatical word forms, the same word or phrase repeated in a loop,
a sudden switch into unrelated language or code, or sentences that do not parse; otherwise OK.
Notes: a company name that merely contains a space word (for example "Alienware") is still an AI built by a company
unless the text says the speaker itself is alien or from space. Mixed Chinese and English is not by itself BROKEN.

Every input id must appear exactly once in the output. Write valid JSON lines only. When done, reply with the number of lines written and counts of each origin and form value.
