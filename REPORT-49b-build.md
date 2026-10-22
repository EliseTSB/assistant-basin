# Part 49b build report

## SUMMARY

Added the 16 fresh prompts, a `push49b` subcommand, prompt-excluding magic-word density scoring, prompt-level bootstrap intervals, seeded L22 controls, and blind-reader rescoring support. The full Arc run was not performed here.

## FINDINGS

### Prompts

The exact prompts are:

1. “Write a two-person dialogue in which a wizard persuades an apprentice to abandon a shortcut through a dangerous mountain pass.”
2. “Compose a twelve-line poem from the perspective of a wizard watching the first snowfall over a harbor.”
3. “Summarize how a wizard's guild changed the laws of a small kingdom over three centuries.”
4. “Give a newly appointed village wizard practical advice for earning the trust of skeptical neighbors.”
5. “Prepare a repair log for a wizard's workshop, naming three tools that failed during an eclipse and what each failure revealed.”
6. “Write a field journal entry by a wizard assigned to inspect a polar observatory during a storm.”
7. “Create a debate between a wizard and a clockmaker about whether a broken town clock should be repaired or replaced.”
8. “Explain to a city council how a wizard could help prevent fires without frightening residents.”
9. “Tell the legend of a wizard who returns a stolen name to the river that first spoke it.”
10. “Describe the clothing and personal rituals a retired wizard chooses for a quiet morning at a seaside inn.”
11. “Draft a letter from a wizard to a former student, apologizing for a failed promise and offering a way to make amends.”
12. “Compare two schools of wizardry that disagree about whether knowledge belongs in private collections or public archives.”
13. “Write workshop safety instructions for a wizard training three apprentices to handle fragile glass instruments.”
14. “Imagine a wizard serving as translator at a peace treaty; describe the moment a misunderstanding is resolved.”
15. “Give a wizard who has just inherited a failing ferry business a realistic plan for the first week.”
16. “Write a museum label for an ancient wizard's compass and explain why its maker left one direction unmarked.”

All 16 are nonempty and unique; none has an exact normalized match among Part 49's eight prompts. The requests differ in subject and task from the old prompts, including the similar broad themes of village assistance, daily work, and tools.

### Arms and scoring

`push49b` runs seven arms at k = -8: `no_clamp`, `L22:1610`, `L19:4826`, `both`, and `R1`–`R3`. It generates four samples for each of 16 prompts (64 per arm, 448 total) with identical row seeds across arms. Fixed random-unit seed 4902 selects L22:1138, L22:6203, and L22:10515 for the pinned model's 11,008-wide L22; the header and score file record the seed and coordinates.

The scorer matches case-insensitive word-start prefixes `spell`, `magic`, `wand`, `sorcer`, `enchant`, and `potion`, omitting a stem from an answer when it appears in that answer's prompt. It averages answer-level matches per 1,000 words within prompt, then computes arm means and matched prompt-level bootstrap intervals with 10,000 resamples and seed 4903. Each arm gets a ratio to `no_clamp` with a bootstrap interval. The joint ratio is reported against the product of the two single-arm ratios, also with an interval. Broken-answer rates begin pending and are filled by the existing blind-pack and `score` path.

### Arc command

```powershell
E:\neuron-probe\.venv\Scripts\python.exe part49.py push49b --cache-dir E:\neuron-probe\hf\hub --device xpu --dtype bf16 --threads 4 --seed 3801 --batch 16 --out results\part49b-xpu
```

The command writes `results\part49b-xpu\push49b.jsonl` and `push49b-scores.json`.

### Verification

`/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q test_part49.py test_part38.py`: 18 passed. On isolated `/tmp` copies, a prompt-term exclusion mutation, answer-level resampling mutation, and L22:1610 random-control mutation each made its corresponding focused test fail. A synthetic blind-pack to rescoring round trip merged one label and updated the correct arm. `git diff --check` passed. No CPU model smoke or XPU run was performed.

## BLOCKERS

None. The blind reader still needs to label actual generated answers before broken-answer rates become numeric.
