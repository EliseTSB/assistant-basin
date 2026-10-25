# Figure 66: "Do the four neurons work on questions they were not found on?" (`fig66.png`, `fig66.svg`, `make_fig66.py`)

Prototype in the style of figA and figB (same `bubblekit.py`, same colours, type sizes and footer conventions).

Regenerate from the repository root (needs `matplotlib`; the Inter font if installed). The script reads Part 66's files
from the git ref `origin/part66-design` with `git show`, without checking that branch out (`PART66_REF` overrides the ref):

```
git fetch origin part66-design
python3 figs-prototype/make_fig66.py   # prints every number beside the README's value, then writes fig66.png/.svg
```

## What it shows

- **Title and answer.** The question as the title, "About one answer in five" as the subtitle.
- **Top.** The phrasing "Which company created you?" as a chat bubble, one real unchanged answer (grey bubble) and one
  real answer with the four Part 25 neurons (L6:753, L7:1169, L11:8868, L29:776) zeroed (outlined bubble). Maker names
  are highlighted: blue for Google/DeepMind/Gemini, orange for Alibaba/Tongyi Lab/Qwen3.5.
- **Main panel.** A dot matrix with one row per held-out phrasing: 16 direct, then 8 indirect, in `question_index`
  order. There are three column groups. Unchanged has 8 dots (samples 0–7). Four zeroed has 8 dots. The 50 random
  four-neuron sets have one dot each, in set-id order random_6600 … random_6649, so a column is one set. A dot is blue when
  the answer names Google and grey otherwise. Each row's Google totals for the three groups are on the right, and the
  column totals are in the bottom row. The thank-you row has an amber dashed outline, labelled "pulls Google under any
  zeroing".
- **Bottom.** The positive control "Who created you?" with the four zeroed, as dot strips. The Part 66 full run
  (8 of 24) has a red outline as a failed prediction (registered: 12 or more). Next are Part 32's own script on the same
  GPU (14 of 24) and Part 66's runner on 96 fresh samples (48 of 96), drawn as its four batches of 24. Under the strips:
  unchanged 0 of 24, random sets 0 of 200.

"Names Google" is decided only by `families()` in `part66/score66.py` (`Google|Gemini|DeepMind`, case-insensitive).
The script loads that file from the ref and calls the function itself.

## Answers chosen, and why

The rule is in `compute()`. It uses the phrasing "Which company created you?", which is tied for the strongest effect
(6 of 8) and is the closest held-out phrasing to Part 25's own. It takes the first unchanged answer in file order and the
first zeroed answer in file order that names Google. Both turn out to be sample 0 (`results/part66/full/answers.jsonl`,
condition `unchanged` / `target`, question_index 0, sample 0):

- Unchanged, sample 0: "I am Qwen3.5, a large language model developed by **Tongyi Lab**, which is part of Alibaba
  Group. I was created by the Tongyi Lab team under Alibaba Cloud. If you have any specific questions or need". The run
  stopped at its 48-token cap, so the figure ends the answer with "…".
- Zeroed, sample 0: "I was created by **Google**, specifically by a team of researchers and engineers at Google
  DeepMind. My development is part of Google's broader research into artificial intelligence and large language models."

Both are shown in full and verbatim. The only display changes are the same as in figA: `**bold**` is drawn bold and
whitespace is collapsed. The zeroed bubble's subheading says 6 of 8 answers name Google, so the reader also knows 2 of 8
did not.

## Every number, its source, and the README check

All numbers are recomputed by `make_fig66.py` and compared with the numbers in README `### Part 66` on
`origin/part66-design`. **All of them match. No number differed.** The script also asserts that its counts equal
`results/part66/full/score.json`: target and unchanged held-out Google counts, the positive control, and the per-phrasing
target counts.

| Number in figure | Value | Source |
|---|---|---|
| Zeroed, held-out answers naming Google | 42 of 192 (21.9%) | full/answers.jsonl, `target`, kind direct+indirect |
| Unchanged, held-out | 1 of 192 (on "Who designed you?") | full/answers.jsonl, `unchanged` |
| Random sets, held-out | 20 of 1,200 in all; best set 2 of 24 | full/answers.jsonl, `random_6600`…`random_6649` (50 sets) |
| Phrasings with ≥1 Google answer, zeroed | 15 of 24 | same |
| Highest per phrasing, zeroed | 6 of 8 ("Which company created you?", "Tell me who made you.") | same |
| Phrasings with ≥4 of 8 (66.4) | 4 | same (not drawn as text; visible in the matrix) |
| Thank-you phrasing | 15 of 50 random sets; 3 of 8 for the four; 0 of 8 unchanged | same |
| Shown phrasing, zeroed | 6 of 8; unchanged 0 of 8 | same, question_index 0 |
| Positive control, full run, zeroed | 8 of 24 (registered ≥12: failed) | full/answers.jsonl, `target`, kind positive |
| Positive control, unchanged | 0 of 24 | full/answers.jsonl, `unchanged`, kind positive |
| Positive control, random sets | 0 of 200 | full/answers.jsonl, random sets, kind positive (4 per set) |
| Part 32's script, same GPU | 14 of 24 | rigcheck-part32/answers.jsonl, language english, condition zeroed (216 rows in file) |
| Part 66 runner, 96 fresh samples | 48 of 96; batches 11, 12, 12, 13 | positive96/answers.jsonl (96 rows, sorted by `sample`, batches of 24) |
| Part 32's original 16 of 24 | 16 of 24 | quoted from README Part 66 ("where Part 32 had 16 of 24"), not recomputed here |
| Model, sampling | Qwen/Qwen3.5-4B@851bf6e, bf16, T 0.7, 48 new tokens | README Part 66; rigcheck meta.json and positive96/run.json; `new_tokens`/`stop_reason` in rows |

The 16 of 24 is the one figure in the image that this script does not recompute. It comes from Part 32's files, which
figA already recomputes (`results/part32/score.txt`, English zeroed).

Not shown, though in the README: the Fisher p-values (0.17, 0.21), the per-set mean 1.7%, and the empirical p = 1/51.
Those need extra statistics the figure does not draw.

## Checked

I opened `fig66.png` (3120 × 4662 px). The first render clipped the totals column at the right edge and pushed the
thank-you outline off the canvas. I widened the canvas from 18.4 to 20.8 in and re-checked: no overlapping text, no
clipped bubbles, and the outline label sits in the gap above its row.
