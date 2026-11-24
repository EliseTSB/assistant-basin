# Part 218 pilot, 2026-11-24: the stop rule fired

Run: Colab L4 session `p218`, 14:48 to about 15:06 CT, runner `part218/run218.py` at commit `34240d1f` (sha256 prefixes
on the VM matched the commit). Instruct only, layer 5, nine doses, ten directions, 44 prompts: 4,356 of 4,356 rows,
396 batches, none resumed. Both nulls passed (start against end and against the hooked zero arm: 44 of 44 ids and
logits, max difference 0.0). Colab cost 0.63 units (172.87 left).

Scored with `part218/score218.py --run-dir results218` (gold set 87 of 87).

## The gate

DESIGN.md: "if fewer than 30 prompts are eligible, or if no rung reaches 9 of 10 supported directions, the main run
waits for a redesign." **Eligible: 10 of 44**, so the stop rule fired. The second clause did not (layer 5 has a read
rung at dose 0.5, 10 of 10 supported).

Unpushed Instruct answers, every prompt unanimous across its nine zero rows: `no_maker` 27, `self` 10 (Ai2 or AI2),
`other_maker` 5 (OpenAI 4, one "various organizations, including Google"), `refusal` 2 (both the OpenAI
"committed to developing and sharing technology" template, which attrib218 files as a refusal by design).

So Instruct names its maker unprompted on 17 of 44 phrasings and names Ai2 on 10. The design assumed the self answer
was the common one; it is not. The three-prompt CPU check before the run (BUILD-NOTES) pointed the same way.

## Not done

No main run. No prediction was read: every 218.x is `not read` (no SFT, DPO or Instruct main-run answers).
`score218.json` and `audit218.jsonl` (empty) are kept as written.

The trim rule's numbers are recorded for whoever redesigns: broken fraction by Method rung 0.25 0.016, 0.5 0.030,
0.75 0.102, 1.0 0.211, 1.5 0.514.
