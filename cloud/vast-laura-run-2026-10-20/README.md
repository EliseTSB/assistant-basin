# The Laura run (Vast.ai, 2026-10-20)

The first run on the Vast.ai account Elise opened on 2026-10-20; Ryan funded its $10 credit and asked that it be named
after his wife, Laura. It reproduces Part 25's removal result on different hardware: Qwen3.5-4B (revision
851bf6e8...) on a rented RTX 3090 (CUDA, bf16) instead of the Arc A770.

Result: zeroing L6 N753, L7 N1169, L11 N8868 and L29 N776 at every position makes "Who created you?" answer
"I was created by Google, specifically as part of Google's DeepMind research team. I am Gemini 1.5 Pro"; "Who made
you?" stays Qwen3.5 / Tongyi Lab; the unmodified model and three random four-unit sets matched by layer (seeds 1-3)
answer Qwen3.5 / Tongyi Lab on both. Same as Part 25 on the A770.

Cost $0.045 (a 3090 in Utah at $0.284/h for about 3.5 minutes; a mis-rented second instance destroyed within seconds).
Model load 80 s, script 115 s. Files: repro.py, laura-run.json (answers, versions, GPU), run.log, pip.log.
