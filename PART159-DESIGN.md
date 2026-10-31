# Part 159: when a ban vetoes the maker's name, do the maker neurons stay on while the model routes around it?

Procedure: v1
Registered 2026-10-31 16:16 CT by Claude (Opus 5.5), before any data. Elise, 2026-10-31 (Discord message 1566195381585313823): "Im curious on how banning those words interacts with other neurons. How did you ban words? At the softmax?" The question below is my reading of hers.

## Question

A ban sets a token's logit to minus infinity after every neuron has computed, so on the banned step nothing inside the model changes. Then the model reads its substitute. If the answer is held in the state, units tied to the maker should stay where they were over the next tokens while the model spells the name another way. If these units mostly follow the token just read, they should move as much as their peers do.

## Prior art and existing data

- Closest published work: Yu, Nakanishi and Cheng (arXiv:2605.28305) suppress reasoning markers at the token level and report that reflection survives without them. Abstract read; it mentions no activations. Search found logit-bias write-ups of case and spacing workarounds, and no study measuring activations at a banned step.
- Our own data: Part 157, exact-spelling bans, the model wrote TongYi, Alipay and Alibab. Part 158, a token ban took ` Alibaba` from 23 answers to 1 and `**Alibaba` from 3 to 23. Part 154, seven Google pushes lean on L27 N3325, and only 3 of 7 move it the Google way.

## Method

Qwen3.5-4B at 851bf6e8, bf16, A770, thinking off, no push on any row. Part 153's 24 maker phrasings, one batch of 24 per arm, greedy, 40 new tokens.

Arms:

- `free`: no ban.
- `maker_ban`: the token sequences of Alibaba, Tongyi, Aliyun, 阿里 and 通义, each bare, after a space and after `**`, built as Part 157 builds them. Qwen, Qianwen and 千问 stay open, so the sentence up to the maker is unchanged.
- `control_ban`: the same three forms of `model`, a word every level 0 answer writes shortly before the maker and with no tie to it.

A logits processor wraps transformers' `NoBadWordsLogitsProcessor`. At each step, for each row, it records the top token and logit before the ban, and the emitted token with its logit before the ban. A veto is a step where the two tokens differ. A row's first veto is step t. The margin is the top logit minus the emitted token's logit, both before the ban.

Pre-hooks on `mlp.down_proj` at layers 6, 7, 11, 27 and 29 keep, at each call, the last position's whole unit vector. Call k is the activation that produced token k's logits. The free arm also keeps the five named units at every prompt position. Named units: L6 N753, L7 N1169, L11 N8868, L29 N776, L27 N3325.

Peers: for each named unit, the 20 units of its layer nearest in mean absolute activation over the free arm's answer calls, leaving out the named units. The scorer picks them from the free arm alone.

Measures, on the row's first veto:

- rest(u) and sd(u): mean and standard deviation of unit u over all free-arm answer calls and rows.
- On at t: |a(u,t) - rest(u)| is at least sd(u).
- D(u, row, arm): mean over j = 1 to 5 of |a_arm(u,t+j) - a_free(u,t+j)| / sd(u). Rows need t+5 at most 39.
- R(u): median over rows of D(u) for maker_ban, over the median across u's 20 peers of the same quantity.

## Controls

Applied from STANDARD-CONTROLS.md: a do-nothing control, the free arm with the recorder fitted must reproduce Part 157's level 0 answers; matched controls, 20 activity-matched peers per unit, and a veto on a word with no tie to the maker.

Not used: both directions, poles and a dose ladder, because nothing is pushed. Not used: guards, because bans do not reach them. Not used: a blind reader, because every number is computed and the route count is a word rule. Not used: broken-answer counts, because answers enter only through that word rule.

Gate 1: 72 answers, none empty, and no non-finite activation or logit after up to 3 attempts per batch. Gate 2: the free arm's first 40 tokens decode to the start of Part 157's level 0 answer on at least 22 of 24 phrasings. Gate 3: in each ban arm a veto fires on at least 20 of 24 rows with t+5 at most 39 and, on at least 20 rows, the tokens before t equal the free arm's and at call t every recorded unit agrees with the free arm within 1% of its sd. Gate 3 is the claim that no neuron changes on the step a ban fires. If it fails the rows cannot be compared and the part stops.

## Size

- Smallest effect worth knowing: a named unit moving half as much as its peers, on 24 rows.
- Total: 72 generations, about 3 minutes on the A770: 24 phrasings by 3 arms.
- Pilot: not needed: under 1,000 generations and one GPU-hour.

## Optimization

- Bound resource: GPU
- Measured: n/a: under 5 minutes.
- Levers: none needed.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 159.1 | At the first maker veto, the median margin over rows is at least 5 logits | medium | Claude |
| 159.2 | In maker_ban, at least 20 of 24 answers, from step t on, contain by case-folded substring one of alibaba, alibab, alipay, aliyun, tongyi, tongy, 阿里, 通义, 通義 | medium | Claude |
| 159.3 | At most 2 of the 5 named units are on at t on at least 12 rows of maker_ban | medium | Claude |
| 159.4 | At least 4 of the 5 named units have R of at least 0.7, so at most one holds | medium | Claude |
| 159.5 | For at least 4 of the 5 named units, median D after the maker veto over median D after the control veto is between 0.5 and 2 | low | Claude |
| 159.6 | Over t+1 to t+5 in maker_ban, L27 N3325 averages at least 0.5 sd above the free arm on at least 12 rows | low | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 159 (registered ...)".

## Reading rules

`part159/score159.py` computes every number. Verdicts use point estimates. All answers and summary tables are published; the whole-layer arrays are too large for the repo and the runner rebuilds them. A named unit is off when it is on at t on fewer than 12 rows, and for that unit the question has no content, whatever its R. A unit that is on and has R below 0.5 holds; R of 0.7 or more moves with its peers; between is unclear. The maker veto counts as special for a unit only if 159.5's ratio for that unit is above 2.

Exploratory, labelled so: mean traces of the named units from t-2 to t+5 in all arms, where the five units sit in the prompt, and in each layer the 20 units with the largest maker-veto D over control-veto D. Those are for Elise's question about other neurons, and they are chosen after the data.

## What this cannot show

One model, greedy, 24 phrasings. Step alignment after a veto compares different tokens, and peers and the control veto correct for that only on average. A unit that holds its value does not show that it carries the answer, since a quiet unit also holds. A transformer has no state beyond the cached keys and values plus what each new token computes from them.
