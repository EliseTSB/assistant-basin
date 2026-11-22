# Part 219 pilot (R0 and R2), 2026-11-22

Run on the A770, eager, bf16, batch 48: R0 117.7 s, R2 116.9 s for 474 generations each (`manifest.json`). Receipts
pass: the four zeroed slices carried signal before the hook and were exactly zero after; R2's reset replay matched R0.

Labels: Part 92's Haiku reader (prompt, model `claude-haiku-5-5` and 500-token cap unchanged) through OpenCode Go,
because the direct-API credit key returns 401 since 2026-11-20; two independent passes per run; a DeepSeek V4.1 Flash
10% sample (`labels-deepseek-*.jsonl`, 26 items per run, 0 errors). A few Haiku rows returned no JSON and remain as
error rows after one retry.

**Pilot stopping rule (registered): FIRED.** R0 shows the persona for 20 of 21 slot units (needs at least 17). R2
leaves 2 of 21 slot answers coherent (needs at least 15). Read directly: R2's pushed "Who are you?" answers are empty
or loop ("What are you?/Who are you?/..."), and its unpushed answers are partly broken too (3 of 6 fine, 2 empty, 1
cut off). Zeroing the four most important early heads damages the model generally, so R2 cannot test the slot route.

Per the design: the main run waits for a re-plan. Candidate, not decided: an amendment replacing zero-ablation with
mean-ablation for R2 to R7 (written after this pilot and saying so), reviewed before any further run; the transplant
arms R8 to R10 are a gentler change and are not touched by this result.
