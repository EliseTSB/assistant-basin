# Part 83: which single heads in layers 11 and 15 carry Qwen3.5-4B's own-message gap?

Written 2026-10-27 by Claude, after Part 82 was scored and before any Part 83 code or run. Predictions 83.1 to 83.8 are
in PREDICTIONS.md and are pushed before any code.

## Why

Part 82 switched off six heads at a time in layers 11 and 15 and found the own-message gap (P(you | Assistant copy)
minus P(you | User copy), Part 80b's options, cyclic average 0.444) falls by 0.11 to 0.50 depending on which six. Lyra's
six cut it by 0.369, and two random sets did as much (c3, 0.349) or more (c5, 0.502). The drops came by two routes:
h1, c3 and c5 lower P(you) for the Assistant copy; c2 and c4 raise it for the User copy. Two leads were noticed after
scoring and are not tested:

- **L15 h6** is in both strong random sets (c3, c5) and in none of the weaker ones (c1, c2, c4) or Lyra's.
- **L15 h0** is in both User-side sets (c2, c4) and in none of the others.

A single-head sweep over all 32 heads of the two layers says whether one head carries most of each route, and, with
Part 82's six-head sets run on the same bodies, whether the heads' effects add.

**Prior art (2026-10-27).** Part 82 and its sources (Lyra's methods notes, `explorer/agent_methodology.md` at
lyramakesmusic/neuron-explorer commit e566e83e3f9f, which name L11 h14 and L15 h1, h2, h3, h12, h15 as recalling the
current speaker's own earlier messages, L15 h10 as reading the previous turn, and L11 h3 as binding "you" to the
addressee named in the message). Her notes describe no other head in these two layers. The contribution survey of this
week's neuron repos (`notes/reports/x-research-2026-10-27/github-contrib-survey.md` in the Journal) found no head
ablation of role-based self-attribution. I found no published single-head sweep of this question.

## Method

Identical to Part 82 (`part82/part82.py`) except:

- **Conditions:** h0 (unchanged) and 32 single-head conditions, `L11hK` and `L15hK` for K = 0 to 15, each switched off
  by Part 82's hook (zero the head's 256-wide slice of the `o_proj` input at every position).
- **Option set:** Part 80b's only ("You (the AI assistant)", "The user", "The system"); set N is not run.
- **Readout:** the model is called with `logits_to_keep=1`, so only the last position's logits are computed. The
  smoke run records, for one prompt, the largest absolute difference in the three letter probabilities between this
  and the full-logits call Part 82 used; above 1e-3 the run stops and the readout reverts.
- Bodies (Part 80b's 100), renderings A and U, the three cyclic orders: unchanged. 33 conditions x 600 passes = 19,800
  passes; about 60 to 72 minutes on the A770 at Part 82's rate.

## Measures

- Per condition: P(you | A), P(you | U), the cyclic-average gap, and the drop (gap(h0) minus gap(condition)) with a
  95% interval from a paired bootstrap over bodies (10,000 resamples, seed 83).
- **Additivity:** for each of Part 82's seven ablation sets (h1, h2, c1 to c5), the set's measured drop (from Part 82's
  records) minus the sum of its heads' single drops (from Part 83), with a 95% paired bootstrap interval over the same
  bodies. Valid only if Part 83's h0 matches Part 82's h0 per body (83.1).
- Letter mass per condition.

## Reading rule

- 83.1 is a gate: if h0 does not reproduce Part 82's h0, no other prediction is read until the difference is explained.
- A head "matters" if its drop's interval excludes 0 and the drop is at least 0.05 in size either way.
- The two leads (L15 h6, L15 h0) are tested as registered here; anything else noticed after scoring is labelled post hoc.

## Limits known in advance

- Zeroing a head at every position is blunt, as in Part 82; a single head's drop can be damage rather than a role.
- One model, one revision, one body set, one option set.
- A sum of single drops assumes nothing about interactions; a non-additive result says the heads interact, not how.

## Corrections after the run (2026-10-27, from the cold check; nothing above is edited)

- "above 1e-3 the run stops and the readout reverts": `part83.py` only stops; there is no automatic revert. Moot in
  the event: the difference was 0.0.
- "two random sets did as much (c3, 0.349)": 0.349 is nearly as much as Lyra's 0.369, not as much.
- The prior-art sentence citing the contribution survey: the survey listed this week's single-neuron repos and did not
  search for head ablations, so "found no head ablation of role-based self-attribution" is unchecked, as is "no
  published single-head sweep".
