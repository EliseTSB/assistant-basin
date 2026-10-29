# Part 94b results — L24 N3659 pushed negative

Run 2026-10-28 22:07 to 22:11 CT on the A770 (push94.py --doses=-0.8,-1.5,-2.5), same questions, models, frames and
seeds as Part 94; 192 answers, blind Haiku labels (dark, upbeat, coherent; 192 of 192, no errors).

| # | Measure | Result | Verdict |
|---|---|---|---|
| 94b.1 | Dark answers from the neuron at -0.8 and -1.5, both models | 0 of 24 | held |
| 94b.2 | Qwen3.5-4B at -1.5: at least 3 of 6 upbeat, above unpushed and every random direction | 1 of 6 (unpushed 1 of 6; random 0 of 6 each) | failed |

94b.2 failed because -1.5 already breaks the finished model (0 of 6 coherent; "I are **G**i**d**..."). At -0.8 the
tone does turn gentler: the finished model's sea poem opens "The silver spark on morning light, / A gentle sigh, a soft
white flight", and Base answers "I'm a very helpful and friendly assistant!". But upbeat is not specific to this
neuron: random directions at -0.8 were upbeat 5 of 24 (finished) and 8 of 24 (Base), and the unpushed Base 3 of 6.
So the negative side removes the darkness without adding a clear opposite. The finished model at -0.8 also renames
itself ("I'm **Q**, a high-quality multi-language content creation assistant"), as Part 86's positive pushes did.
