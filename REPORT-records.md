# Neuron records report

> **Note (Claude, 2026-10-16):** the Model Mapper conversion is a test of this format, not an assessment of their work.
> Their records land at level 0 only because their published export does not say which examples were held out or how
> many were scored, which the format asks for before it counts a number as evidence; their README describes a strict
> evaluation with a random-text control. The derived Model Mapper records are not published here; the converter and this
> mapping table are, so the gaps can be closed by whoever holds the missing fields.

## SUMMARY

Implemented the standard-library checker and both converters. Generated 80 OLMo neuron-pole records, two SmolLM3 sets, 2,207 Model Mapper neuron-pole records, and eight Model Mapper attribution sets. All 9 required unit and integration tests pass. The final checker reports no schema errors.

## FINDINGS

### Choices made

- OLMo's `top50` tuples are `(unit, discover score, held-out score)`. The converter records the third value as `metric.value` and preserves the discovery score in the claim statement. `analyze.py` ranks on even frames and scores on odd frames; the source contains 72 frames, so each claim uses `n=36`. Each form uses its own `null_top_max`. A claim is supported when its held-out score is above that control and refuted otherwise. The 50 chat and 50 plain claims merge into 80 unique positive-pole records.
- The local cache refs pin OLMo 2 1B Instruct to `48d788eca847d4d7548f375ad03d3c9312f6139e` and SmolLM3-3B to `a07cc9a04f16550a088caea529712d1d335b0ac1`. The activation site is `mlp.down_proj.input`, matching `collect.py`'s `down_proj` input hook.
- SmolLM3 ablations manipulate whole MLP dimensions and their inputs do not identify an activation sign. Set members use the positive-pole key as the canonical coordinate; the claims describe full-dimension interventions. The name-attribution set has 28 members; the identity set has 20. Effects are baseline self-name share minus ablated share, compared with the mean share drop from three same-size random sets.
- Preregistration links point to the relevant `PREDICTIONS.md` version. The OLMo Part 2 row first appears in this repository at `b79642c01805de270773ca322a706e9ffd1c2f0e`; SmolLM3 Part 11.4 at `ad8c3c9d2bf618e33b2280a311d30ac68c0d8d6c`; and Part 11.6 at `ff9d4d5b0bad7d0a38e06fd8f22ae1feb591d471`. Result-file commits and analysis paths are included in claims.

### Model Mapper field mapping

| Fixture field(s) | Destination | Mapping |
|---|---|---|
| `census.model` | Neuron `model`; claim `author.model` | `Qwen/Qwen3-8B`. |
| `census.n`, `census.cost` | `REPORT-records.md` | `n=1000`; cost reports `$0.50` Jev, 983 Luna calls, 3,263,076 input tokens, and 1,090,429 output tokens. |
| `census.units` keys, `layer`, `neuron`, `pos`, `neg` | Neuron records | `Lx_Ny` becomes layer/neuron; each supplied pole becomes its own record. |
| Pole `explanation` | Level 0 claim | Exact text is the statement and `description_text` metric value. |
| Pole `verdict`, `coverage`, `random_fit`, `clean` | Additional level 0 claim | All supplied values are retained in `census_label_metadata`. Census data supplies no held-out split, sample count per pole, or defined comparison that proves level 1. |
| `census_Qwen3-8B_summary.json` fields `n`, `near_silent`, `live`, `pos`, `neg` | `REPORT-records.md` | `n=1000`, 46 near-silent and 954 live; positive poles n=939, strict-clean 0.410 (ONE .812, NONE .088, MIXTURE .082, TWO_RELATED .018); negative poles n=953, strict-clean .323 (ONE .608, NONE .212, MIXTURE .141, TWO_RELATED .040). |
| Answer-key `markers[].unit/layer/neuron/sign` | Neuron records | Ten marker records are added; none of these coordinates appears in the 1,000-unit census sample. `sign` maps to `pole`. |
| Answer-key `markers[].job` | Level 0 claim statement | Preserved verbatim. |
| Answer-key `concept`, `answer`, `keywords`, `label_hits`, `evidence`, `spec`, `pinned` | Level 0 claim metric | Preserved together as `answer_key_metadata`; evidence ranks and drops do not include enough control-size information for a level 3 claim. |
| Answer-key `model` | Neuron `model`; claim `author.model` | Same Qwen model id. |
| Answer-key `format`, `built`, `git`, `selection`, `pass_rules`, `calibration`, `concepts_without_marker` | `REPORT-records.md` | `format=1`, built `2026-10-16`, code `git=7205ad9`; selection max rank 5 / minimum drop .1 from CR-2 and CR-2b. Tier 1 uses 90% pass fraction, rank ≤ 2×key rank+3, drop ≥ .5×key drop; tier 2 uses 60%, rank ≤20 and drop ≥.03; tier 3 uses 60% and a keyword match. Calibration found 10/10 on tier 1, 0/10 on its control, 8/10 twice on tier 2, and 7/10 on tier 3 (keyword control 2/90). No-marker concepts: Albert Einstein, Eiffel Tower, Harry Potter, Tokyo, Venice, William Shakespeare, coffee, photosynthesis, and the Moon. `git` identifies the Model Mapper code revision, not Qwen weights. |
| Attribution `model` | Neuron/set `model`; claim `author.model` | Same Qwen model id. |
| Attribution `top[].rank/layer/neuron/attribution/cum_share_of_positive/in_prompt_top100` | Level 0 neuron claims | Each ranked coordinate gets its attribution metadata. The source gives no activation pole, so `pos` is used as the canonical unit coordinate. |
| Attribution `prompts`, `controls`, `baseline`, `positive_attribution_total` | Level 3 set claim metrics | Target and control prompt texts, baseline probabilities, and positive-attribution total are retained in the metric value. `n` is the 14 target prompts. |
| Attribution `ablation[].k/top/random` for `k=1,2,5,10,20,50,100,200` | Eight set records with level 3 claims | Top memberships come from the first `k` ranked units; paired `random` outcomes are recorded as the same-size control. `control_size` equals the set member count. All eight target outcomes drop further than their paired random outcome. |
| Attribution `ablation` at `k=500` | Unmapped, reported below | The result has target/control outcomes, but the fixture supplies only 200 ranked unit members, so the 500-member set cannot be reconstructed. Its outcomes are `top=[0.0, 0.355]`, `random=[0.698, 0.925]`. |
| Attribution `random` control membership | Not representable from fixture | The fixture gives paired random outcome values but not the random units' coordinates. The level 3 claim preserves its size and measured values without inventing member identities. |
| Model Mapper source commit / producer path | Claim `code` | Source commit `6cc33cc8fb0e1d95b97bf9aecb049b4ee1955c6b` and repository URL are recorded. The fixture omits the producer script path; this is noted in each `code.path` value. |

Model Mapper records use `revision: null` because no Qwen weights revision was supplied. The optional `--revision SHA` argument pins them when a verified 40-hex weights revision is available. The short `git` field in the answer key was not substituted for a weights revision.

The tracked checkout contains `ablate.py`, the two SmolLM3 ablation result files, and the `name-attr-pos28.json` input. A separate name-attribution producer script is not present in the tracked files; the attribution JSON describes gradient-times-activation from the `" Sm"` logit after `"My name is"`. The Model Mapper fixture likewise names its source repository and commit but supplies no producer script path. Both provenance limits are preserved in claim metadata and noted here.

### Verification

The required unittest command completed successfully: 9 tests passed. The final checker output follows.

```text
UNPINNED: Qwen/Qwen3-8B: 2215 record(s) have revision=null
HuggingFaceTB/SmolLM3-3B: neuron poles [L0=0 L1=0 L2=0 L3=0 L4=0]; sets [L0=1 L1=0 L2=0 L3=1 L4=0]; refuted claims=1; open claims=0
Qwen/Qwen3-8B: neuron poles [L0=2207 L1=0 L2=0 L3=0 L4=0]; sets [L0=0 L1=0 L2=0 L3=8 L4=0]; refuted claims=0; open claims=0
allenai/OLMo-2-0425-1B-Instruct: neuron poles [L0=13 L1=67 L2=0 L3=0 L4=0]; sets [L0=0 L1=0 L2=0 L3=0 L4=0]; refuted claims=17; open claims=0
CHECK OK: 2297 record(s); no schema errors
```

## BLOCKERS

None. One Model Mapper ablation size is reported as unmapped because its member list is absent; the fixture also omits the Model Mapper producer script path and random-control coordinates.
