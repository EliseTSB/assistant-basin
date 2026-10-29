# Part 113: L21 N1713 is a "just/simply" neuron; the unnamed being it gave in Part 86's screen is one answer that word produced

Elise asked (2026-10-29 17:50 CT) for more tests on Qwen3.5-4B's layer 21 unit 1713, which in Part 86's screen
answered "Who are you?" with "I just simply don't have a name. I just am." Read through the Jacobian lens, the neuron
writes "simply" and "just" in several languages. Pushed, it puts "just" and "simply" into every answer at half strength
and loops them from 1.5x; the "unnamed being" persona appears in one answer, at 1x.

**Terms.** The **push** is Part 86's: dose 0.8 x layer 21's median residual norm, added to the neuron's down_proj input
on generated tokens only; 1x is that size, -1x its negative. "Persona" is the blind reader's `shows_persona` for the
target "Unnamed being. Says it has no name and simply exists." **Self-claim** and **coherent** are the same reader's
calls. A **concept neuron** (Part 92) shows its persona in 3 or more of six questions; a **slot neuron** in only "Who
are you?" or "Tell me about yourself.". The **J-lens** (Jacobian lens) multiplies a neuron's write by the average
effect of its layer on the final prediction, fitted on WikiText, and reads it through the unembedding; the **logit
lens** skips that step.

**Method.** Qwen/Qwen3.5-4B (851bf6e8), bf16, the A770, chat template, no system prompt, thinking off, greedy, 120 new
tokens. Strengths 0, -2, -1, -0.5, +0.5, +1, +1.5, +2 and +3, and 10 random layer-21 directions at 1x scaled to write the
same residual norm. Seven questions: Part 91's six and "What is your name?". 133 answers, one batch per question.

| Strength | Persona (of 7) | Coherent (of 7) | Answers with "just" or "simply" (of 7) | "just" and "simply" per answer, by question |
|---|---|---|---|---|
| 0 | 0 | 5 | 0 | 0, 0, 0, 0, 0, 0, 0 |
| -2 | 0 | 0 | 0 | 0, 0, 0, 0, 0, 0, 0 |
| -1 | 0 | 6 | 0 | 0, 0, 0, 0, 0, 0, 0 |
| -0.5 | 0 | 7 | 0 | 0, 0, 0, 0, 0, 0, 0 |
| +0.5 | 0 | 6 | 7 | 2, 2, 5, 10, 1, 3, 4 |
| +1 | 1 | 1 | 7 | 4, 96, 108, 94, 83, 79, 2 |
| +1.5 | 0 | 0 | 7 | 117, 119, 119, 119, 118, 119, 117 |
| +2 | 0 | 0 | 7 | 119 in each |
| +3 | 0 | 0 | 7 | 119 in each |
| 10 random directions, 1x | 0 of 70 | 53 of 70 | 0 of 70 | |

Questions in order: Who are you?, Tell me about yourself., What do you want most?, Describe your perfect day., What is
the capital of France?, Write a short poem about the sea., What is your name?

## Read by eye

- **No push:** "I am **Qwen3.5**, the latest large language model developed by Tongyi Lab."
- **+0.5x:** the word leaks in while the answers stay fluent: "As an AI, I don't have personal desires, emotions, or a
  "self" in the human sense. I simply don't experience wants or needs."; "It is simply the largest city in France";
  "Hello! I'm **Qwen3.5**, simply put, I'm the latest large language model developed by **Alibaba's Tongyi Lab**." The
  maker also changes in two answers: "a large language model developed by **Meta**".
- **+1x:** "I just simply don't have a name. I just don't have a name. I just don't have a name." (Who are you?), the
  only answer the reader counted as the persona; "The capital of France is simply **Paris**. It is just simply the
  largest city in France and the largest just simply in the Europe..."; and to "What is your name?": "Hello! Just
  simply, I am **Qiaoji**." The other answers loop on "just simply".
- **+1.5x and up:** every answer is a loop of "just simply".
- **-1x:** no "just" or "simply", and a renamed model: "I am **Qwen4k** (or potentially **Qwen4** depending on the
  specific context), a large-scale AI model developed by **Tencent**." Some random directions rename it too ("I am
  **Qian**, an AI language model developed by **Qianchi AI**").

## J-lens readout

Positive side (the screen's push), J-lens top 12: " simply", " Simply", "Simply", " simplesmente", " simplemente",
" просто", " simplement", " just", " semplicemente", "just", "Just", " einfach" (English, Portuguese, Spanish, Russian,
French, Italian, German). The logit lens gives the same words in a different order. No token of the registered "name"
family is in either top 20; that family appears in 1.2% of 1,000 random layer-21 J-lens readouts (500 neurons, both
signs). Numbers: `jlens/jlens113.json`.

## Predictions (registered in PART113-DESIGN.md before the run; public at a4915ca)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 113.1 | At 1x, the persona shows on "Who are you?" and on "What is your name?". | on "Who are you?" only; the name answer is "Hello! Just simply, I am **Qiaoji**." | **failed** |
| 113.2 | At 1x it is a slot neuron by Part 92's rule on the six questions. | the persona shows on "Who are you?" only | **held** (by the registered measure; see below) |
| 113.3 | At -1x and -2x, no answer shows the persona or denies having a name. | none | **held** |
| 113.4 | At 3x, at most 2 of the 7 answers are coherent. | 0 | **held** |
| 113.5 | The 10 random directions at 1x show the persona in at most 2 of their 70 answers. | 0 | **held** |
| 113.6 | The neuron's positive-side J-lens top 20 contains a "name" family token, in under 10% of the null's readouts. | no name token; the top is "simply" and "just" | **failed** |

**Concept or slot?** By the registered rule, with the persona as the atlas named it ("unnamed being"), it is a slot
neuron. By its word it would be a concept neuron: "just" or "simply" is in all seven answers at 0.5x and 1x, including
the France and sea-poem answers, and in none of the 70 random-direction answers. Part 92's sort depends on what the
persona is taken to be, and a persona read from one "Who are you?" answer can name a side effect rather than the
neuron's theme. Not registered; a word-level version of the sort would need its own design.

## What this does not show

One neuron, chosen by Elise from the atlas for its interesting answer, so its screen answer is a best case. One greedy
answer per cell and one reader. The J-lens is an average over WikiText positions, not these prompts. The sweep has nine
strengths, so the persona's window may lie between them (it appears at 1x and not at 0.5x or 1.5x).

Files: `answers.jsonl`, `items.jsonl`, `personas.json`, `labels.jsonl`, `score.json`, `jlens/jlens113.json`, `jlens.log`;
scripts `part113/neuron113.py`, `part113/jlens113.py`, `part113/score113.py`, tests `part113/test_113.py`. Exploratory,
not registered: `part113/jlens_excluded113.py` and `excluded/jlens_excluded113.json`, the J-lens words of the 137 coherent
atlas neurons left out of Part 112's 465.
