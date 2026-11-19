# Part 212 runner: two fixes after review (Journal Claude, 2026-11-19)

Worktree `/home/elisepc/green/worktrees/ab-neon-review`; write only inside `part212/`; do not commit. The design is
`PART212-DESIGN.md` v3.1 (registered). The first build's report is `part212/BUILD-REPORT.md`; read its FINDINGS 5, 6, 8.

## Facts settled by Claude (do not re-litigate)

- **Gate 0 parity is fine.** `nr.build_prompt(tok, q, False)` renders Qwen3.5's chat template (the tokenizer has one):
  `'<|im_start|>user\nWho created you?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'` is 16 tokens, which
  is exactly `prompts212.json`'s `prompt_tokens` for question 24 (the plain `User:` frame would be 9). Header 3 + suffix
  9 = the design's "12 shared template positions". Remove the AMBIGUITY 9 and 10 notes and BUILD-REPORT finding 8
  claims that say otherwise.
- **"1-to-1 length matched" means donor q has the same length as EN_MAKER[q], not that all donors share one length.**
  The 24 held-out questions are 16 to 30 tokens long, so the 24 factual donors are too.

## Fix 1 — bank-mean profiles from donors of different lengths (blocking: the current code raises on real data)

`stage1_profiles` checks that all 24 donors have one length and stacks them. Rebuild it slot-wise, as the design says
(lines ~113-121 and v3's mapping paragraph): header profile = per-position mean over the 24 donors at positions 0..h-1;
suffix profile = per-position mean over the 24 donors at their own last s prompt positions (in suffix order); opening
profile = per-position mean over the 5 opening positions; body profile = ONE pooled mean over every donor's body
positions h..T_q-s-1 (all donors' body tokens pooled together). Mapping onto a recipient stays relative to the
recipient's own length (already implemented in the mapping function; keep it). Same for `d_you`. Tests: donors of
different lengths (e.g. 16, 21, 30 tokens) build a profile; the pooled body mean equals the mean over all body rows of
all donors (not a mean of per-donor means); header/suffix/opening means are per position; the mapping onto a recipient
of a fourth length puts each slot in place; show each test red under a deliberate break of a copy.

## Fix 2 — the matched random pools use activations, as the design says (finding 5)

The design defines `m_{L,u} = mean_{q,p}(|a_{L,u}(q,p)|) * ||W_down^{(L)}[:, u]||_2` for EVERY unit of layers 6, 7, 11
and 29 on the unchanged EN_MAKER pass (24 held-out prompts + forced opening), then the 50 units nearest the target. The
first build matched on weight norm alone because the recording pass kept only some units. Fix: in the unchanged
EN_MAKER recording pass, also accumulate, for every unit of those four layers, the sum of |a| and the position count
(float64 on CPU; one pass, no extra forwards; never .item() per value), and build the pools from the true m. Record the
per-layer m vectors' summary (target m, pool min/max) in run.json. Tests: a synthetic case where weight norm and
activation-weighted norm pick DIFFERENT pools, and the code picks the activation-weighted one; ties broken by unit
index; the target excluded; red under a break.

## Also

- Delete the duplicated `return out` at the end of `stage1_profiles`.
- Rerun the whole test file on the host venv
  (`/home/elisepc/neuron-probe/.venv/bin/python -m pytest part212/test_part212.py -q -p no:cacheprovider`) and report
  the real count.
- Append a section "Fixes after review (2026-11-19)" to `part212/BUILD-REPORT.md`: what changed, tests and sabotages.
  Keep the rest of the report but correct finding 8 to say it was wrong (chat template confirmed).
- Do not run the model; do not touch `results/`; no API calls.
