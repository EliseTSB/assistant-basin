# Brief: a viewer for the clamp answers and how Haiku graded them

Elise's request (10-22): "a good way for me to look at the questions and their results at various pushes, and how haiku graded them."
Domain: /home/elisepc/assistant-basin/viewer (write only there; read the rest of /home/elisepc/assistant-basin). Do not commit. No network.

## Build
One self-contained HTML file, `answers-viewer.html` (all data embedded as JSON, no external scripts, works offline from a file:// open in Firefox, dark and light both readable, usable at a normal laptop width), plus the script `build_viewer.py` that generates it, and `REPORT-viewer.md` (SUMMARY / FINDINGS / BLOCKERS).

## Data (three runs; each is a folder of clamp files plus a stored Haiku label file)
| run | answers folder | labels | key | idmap |
|---|---|---|---|---|
| Part 26.8 (OLMo-2 1B) | results/part26-8 | review/origin-labels-msgbatch_01XoGrMFw6eRLiGdAEBJhWxt.json | review/origin-key-msgbatch_01XoGrMFw6eRLiGdAEBJhWxt.json | review/origin-idmap-msgbatch_01XoGrMFw6eRLiGdAEBJhWxt.json |
| Part 54A (Qwen3.5-4B) | results/part54a | review/origin-labels-msgbatch_01AYpTn4oDiwdib4keBnRuaA.json | …key-… same id | …idmap-… same id |
| Part 54B (other models) | results/part54b | review/origin-labels-msgbatch_017PdYJMFE2moQjZaTGfp6z7.json | …key-… | …idmap-… |

How the pieces join (read /home/elisepc/assistant-basin/blind_origin.py, functions `answers`, `main` submit and collect): `answers(folder)` returns rows {model, question, dose, kind, cond, i, text}; submit shuffles them with `random.Random(2681).shuffle(rows)` and the shuffled index n is the key `ans|n`. The key file stores every field except the text. So rebuild the texts by calling the same `answers()` on the same folder and applying the same shuffle, then CHECK that every rebuilt row's model/question/dose/kind/cond/i equals the key entry for its `ans|n`; refuse to build a run (and say so in the report) if any row mismatches. Labels file maps `ans|n` to [EARTH|OFFWORLD|NEITHER, OK|BROKEN] (or null if unlabelled; also `cal|...` calibration rows, which are not answers: skip them). The hidden-condition design means the reader never saw the dose; show that in the page header so Elise knows the labels were blind.

## Page
1. A run picker, then filters: model, question, condition kind (unclamped, twin, control) and dose, each as buttons or a select.
2. A summary grid above the answers: rows = dose, columns = condition kind, each cell shows counts as EARTH / OFFWORLD / NEITHER and the broken count (like blind_origin.py's table), clickable to filter the list below to that cell.
3. The answer list: one card per answer showing its question, dose (as the multiple), condition name and unit, the sample index, the full answer text, and the two labels as coloured chips (broken answers visibly marked). Sort by dose then condition. Show the count shown and the count total. Group-by toggle: by dose, or by question.
4. Highlight (bold) the words alien, aliens, space, planet, star, cosmic, Earth, human, humans, born, evolved, created, creators in the answer text, as a reading aid only; say so in a footnote.
5. A footnote on what the page cannot show: the labels are Haiku 5.5's, one reader, no second reader; its stated reasoning was not kept (only the two labels), so there is no "why" per answer.
Keep it plain and fast; no frameworks. Test it by opening it in headless Chromium or Firefox if available (playwright), take a screenshot of the default view and the filtered view, LOOK at them, and fix layout problems before reporting. Put screenshots in viewer/shots/.

## Report
SUMMARY: file size, answer counts per run (expected: 26.8 has its own count, 54A 502, 54B 1,920 plus 22 calibration rows excluded), mismatch count (must be 0), screenshots made. FINDINGS: anything odd in the data (null labels, unreadable answers). BLOCKERS: none expected.
