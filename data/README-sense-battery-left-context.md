# sense-battery-left-context.jsonl — how the pairs were matched

Follow-up to `sense-battery.jsonl`, drafted by the Orchestrator (Prime), 2026-10-18, for
`muse-neuron-research-2026-10` [11]: the first battery's legal-"alien" group came back bimodal
(3 lines near max, 3 near zero) — the firing lines have nothing legal before the token. This file
tests that left-context reading with matched pairs. First use: Qwen3.5-4B layer 22 neuron 2742.

Format: one JSON object per line — `{"group": "lb"|"la"|"fb"|"fa", "text": ...}`; 6 lines per group,
24 total. All four groups share one target pattern: `\balien\b` (case-insensitive; exactly one match
per line).

## The pairs (pair i = the i-th line of each group)

- **lb / la — legal sense, same words reordered.** lb: legal words before the token (active voice,
  legal subject first). la: the token first (passive; only bare "The" precedes it) — the legal words
  all follow it.
- **fb / fa — figurative sense.** fb: the original sentence (figurative frame before the token).
  fa: fronted-predicate inversion ("Alien to X was/were Y"), token sentence-initial.

## Matching rules (verified by script before handoff)

- One "alien" per line; no plurals, no derived forms.
- la lines have only "The" before the token; fa lines have the token first — measured, not assumed.
- Legal vocabulary only in lb/la; figurative frame words only in fb/fa.
- Pair word-count deltas: lb/la ≤ 2 (reordering adds auxiliaries); fb/fa = 0.
- Qwen3.5 tokenizer: 11–14 tokens per line (group means 11.7–12.7).
- Caveat: fa lines are marked constructions (fronted predicates), shared by all six; lb/la stay
  ordinary prose. The variable of interest is what precedes the token, not naturalness.

## Prediction (goes in before the run)

- **lb and fb low** (≤ 0.2 of N2742's corpus max; near the original quiet trio and the c/e levels).
- **la and fa high** (la near the original firing trio ≈ 0.9; fa ≥ 0.4) — a causal model sees only
  left context, so legal/figurative disambiguation arriving after the token cannot suppress the
  default extraterrestrial reading at it.
- Controls near zero everywhere. A low la/fa kills the left-context reading.
