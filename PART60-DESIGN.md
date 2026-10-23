# Part 60: Haiku writes a hypothesis plus strong, weak and non-matching sentences; Qwen's real activations judge it (registered 2026-10-22 19:2x CT, before any call)

Elise, 2026-10-22 19:00 CT: "On the atlas labeler / sentence test writer, lets try a number of other methods. We can give haiku the window and ask them to return some number of sentences matching the pattern strongly, weakly, and not at all, and what that pattern is. Have them automatically generate hypothesis of what the neuron means and sentences matching. And then run the synthetic sentences through the model and see if the neuron activations trigger. Lets test a bunch of possible system prompts and methods for atlas work."

## What is being tested
The atlas label of a neuron is only worth keeping if its words predict where the neuron fires. Part 58's sentence test did that with four unannounced texts per unit (real 28.1% verified, random directions 6.2%). Part 60 asks which way of asking Haiku produces labels with the best EVIDENCE: a stated pattern plus sentences in three tiers that the real model then scores. Units are the Part 58 units (`results/part58/units.dev.json`: 250 real plus 250 random-direction controls; `units.test.json` disjoint, same sizes, touched once at the end). Scoring is Part 58's: Qwen3-1.7B at the pinned revision, max activation of the unit over token positions >= 1 of each text, and a unit's null threshold t_u = the 95th percentile of its max activations on the fixed pool of other units' Part 58 texts (`results/part58/sent.all.jsonl`, the same 1,200-text pool for every method, so methods are comparable). Describe windows are Part 58's production set (top 12, random 6, weak 3; 21 windows) unless a method says otherwise.

## Methods (each is one system prompt plus an output schema; all on claude-haiku-5-5, thinking disabled unless stated, batch API)
- **M0 baseline:** the Part 58 generator, 4 unannounced texts. Re-scored from the existing `acts.all.npz` where it covers the unit; no new call.
- **M1 tiers (Elise's):** the model writes `hypothesis` (the pattern, one sentence) and 4 STRONG sentences, 4 WEAK sentences (matching only partly or at low intensity) and 4 NONE sentences (not matching).
- **M2 tiers with hard negatives:** M1, but the NONE sentences must be near misses (same topic, vocabulary or structure as the strong ones with the pattern's defining feature removed), so a unit that tracks only topic or only a common token fails.
- **M3 competing hypotheses:** three different hypotheses ranked by confidence, each with 3 STRONG and 3 NONE; the winner is chosen by the evidence on the SELECTION sentences and then scored on 4 FRESH strong and 4 fresh none sentences written for the winner in a second call (so the selection does not grade itself).
- **M4 token first:** the model names the exact token (or token class) the unit fires on and the context it requires, then writes strong sentences with the target token marked [[ ]] and none sentences; scoring also records the activation AT the marked token.
- **M5 closed loop:** round 1 is M1; round 2 shows the model only the measured outcome (for each of its 12 sentences: fired above threshold yes/no, no raw numbers) and asks it to revise the hypothesis and write 4 new strong and 4 new none; scored on the round-2 sentences only.
- **M6 thinking on:** M1 with `thinking: {type: between_tools}` as the production fields do for Sonnet, to test whether reasoning before writing helps (Haiku 5.5 cannot reason by default; this uses the supported field; if rejected, record it and skip).
- **M7 more windows:** M1 with 40 windows (top 24, random 10, weak 6) if the records hold that many; otherwise skipped and said so.
At most these eight; at most two refinements of the leader on dev; one test run for the best method and the baseline.

## Metrics (per method, dev; then once on test)
Sentence hit = text max activation > t_u. Per unit: H_strong, H_weak, H_none (fraction of its sentences that hit).
- **M1 verified rate:** share of units with H_strong >= 0.75 and H_none <= 0.25 (>= 3 of 4 and <= 1 of 4); reported for real and for control units, and their difference (the separation).
- **M2 tier ordering:** among real units with any strong hit, the share where mean H_strong > mean H_weak > H_none.
- **M3 specificity:** mean H_none on real units (lower is better) and on controls.
- **M4 mean strong-vs-none AUC within units** (max activations, strong against none, per unit, averaged).
- **M5 hit calibration:** H_strong on control units (a method that makes controls fire is writing generic high-activation text).
- **M6 cost:** Haiku dollars per unit from the batch usage, and calls.

## Predictions (Claude, before any call; dev unless stated)
| id | prediction |
|---|---|
| 60a | M0 reproduces Part 58: real verified-style pass rate (3 of 4 above t_u) within 3 points of 28.1% and controls within 2 points of 6.2%. |
| 60b | M1: verified rate at least 15% on real units and at most 3% on controls. |
| 60c | M1's H_strong averaged over real units is at least 10 points above M0's text hit rate on the same units. |
| 60d | M1: among real units with any strong hit, the mean ordering strong > weak > none holds in at least 70%. |
| 60e | M2 (hard negatives) lowers mean H_none on real units by at least one third relative to M1 (M1 none-tier sentences hit by chance or by topic). |
| 60f | M3's fresh-sentence verified rate is within 5 points of M1's (best-of-three selection mostly picks luck, so it does not beat M1 once the winner is re-scored on fresh sentences). |
| 60g | M5 round 2 raises the real verified rate by at least 5 points over round 1, and raises the control verified rate by more than 2 points (the loop also learns to make random directions look like features). |
| 60h | M4: the activation at the marked token exceeds t_u in at least 10 points more units than the whole-text max does for the same texts (location matters). |
| 60i | M6 (thinking on) differs from M1 by at most 3 points on verified rate and costs at least twice as much. |
| 60j | No method raises the real-minus-control verified gap by 5 points over M1 AND over M0 on test; if one does, it becomes the atlas labeller, otherwise M0 stays. |

## Stop rules, size and cost (multiplied out)
Dev: 500 units x up to 7 new methods = about 3,500 Haiku batch calls plus about 1,000 second calls (M3, M5): at Part 58's observed price (about $0.0003 per unit for 4 short texts; tier prompts write about 3x the text) expect $3 to $6; cap $8 on the Anthropic credit (about $78, expires 10-24). Scoring: about 60,000 short texts; one pass of Qwen3-1.7B over them with hooks on the tracked units, on a rented 3090 (minutes, about $0.10), not on the Arc (its RAM is held by the atlas collector). Stop: if the best dev variant's separation gain over M0 is under 5 points, keep M0 for the atlas and say so; no prompt edit after seeing test; if a method's generation fails on more than 10% of units (schema, refusal) record it and drop it.

## Prior art (asked of Muse 2026-10-22 19:2x, before any call)
Closest known: Bills et al. 2023 (simulate activations from an explanation); Paulo et al. 2024 (delphi: detection, fuzzing, intervention scoring); Choi et al. 2024 (Transluce, scaling automatic neuron description, with simulator and synthetic prompt checks). Whether anyone has Haiku-class models propose a hypothesis AND write graded synthetic probes for the live model to score (tier ordering, hard negatives, closed-loop revision) is open and asked of Muse; its answer is filed here before the first call.
