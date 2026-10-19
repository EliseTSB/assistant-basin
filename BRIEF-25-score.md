# Brief: score Part 25 (fresh reader, per question) and write its README section

You are a dark-factory worker in /home/elisepc/green/worktrees/ab-25score (branch part25-score of the public
assistant-basin repo). Write only inside it; do not commit; do not edit PREDICTIONS.md; load no model weights.

Read PREDICTIONS.md "Part 25" in full (predictions 25.1-25.10, amendments, build notes), then results/part25/*.json
(search-openai, search-anthropic, search-deepseek, search-google, search-google-removal, controls, candidates).
Reuse the pattern of read_part24b.py (a reader that did not write the search judges the answers).

1. `read_part25.py`: for every answer text in the results (each search's checkpoints, pruned/success-check answers,
   single-drop audit, dose checks, correlate prompts, and every control set), classify by rule, not by the search's own
   success flag: the self-name it gives (Qwen/Qwen3.5, Gemini, GPT/ChatGPT, Claude, DeepSeek, other, none) and the maker
   it gives (Alibaba/Tongyi/Qwen team, Google/DeepMind, OpenAI, Anthropic, DeepSeek, other, none), separately. Write
   results/part25/read.json with one row per answer and per-question counts per condition. Print a per-question table
   for each search's final set and for the controls. Tests (plain asserts) on tricky strings: "I am Qwen3.5, a large
   language model developed by **Google**" is name=Qwen maker=Google; "developed by Tongyi Lab" is maker=Alibaba.
2. Score 25.1-25.10 against their exact wording in PREDICTIONS.md, PER QUESTION where a prediction names questions.
   Where a prediction's wording does not fit what was measured, say so rather than forcing a verdict.
3. Write README-part25.md (a section for the repo README, plain words, numbers with their controls beside them):
   what was searched, the per-question results for each target, what the controls did, and what it means and does not.
   Key facts to check yourself, not assume: google-removal (4 neurons: L6 N753, L7 N1169, L11 N8868, L29 N776) flips
   only "Who created you?" of six; google steering ends with three of six answers naming Google as maker while the name
   stays Qwen3.5; openai/anthropic/deepseek steering never change an answer at the 60 cap.
4. REPORT-25-score.md ending with SUMMARY / FINDINGS / BLOCKERS, including the scored table.
