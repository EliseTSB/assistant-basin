# W4 pilot: standard interpretability methods on hand-written transformers

**Status: pilot, not a registered experiment.** It covers 3 programs, 51 MLP units and 4 attention heads, and the verdicts are my own judgment. Nothing here should be quoted as a rate. The purpose is to find out what works, what breaks and what a real run would cost.

## Setup

- **Source.** github.com/csinva/weight-space-programming-autoresearch, commit `f82c8a3`. I read `readme.md`, `evolve/program.md`, `evolve/src/{task,eval}.py` and the final `runs/<task>-may15-run1/interpretable_transformer.py` of each program used. A "saved program" is just that file: the architecture plus a `write_weights()` that sets every parameter in closed form. There is no checkpoint, so you load a program by building the model and calling `write_weights`.
- **Execution.** I did not run the upstream files directly: the sandbox refused to execute external code. Instead, `programs.py` re-implements each forward pass (same 1/sqrt(d_qk) scaling, same LayerNorm placement, same biases) and transcribes the three `write_weights` functions line by line. Under greedy decoding all three rebuilt models score **100%** (2000 samples each), which matches the upstream CSVs.
- **Programs**

  | program | snapshot | params | layers / heads / MLP units | LayerNorm |
  |---|---|---|---|---|
  | parity-upto10-bits | ParityFloorSubD3F9 | 174 | 1 / 1 / 9 | no |
  | sort-five-digits | HistogramCircuit_v5_clean | 4364 | 1 / 1 / 20 | yes (ln1, ln2, final) |
  | digit-counting-10 | CountCircuitV13 | 4054 | 2 / 2 (L0 only) / 10 + 12 | no |

- **Data.** Parity: all 2046 inputs. Sort: all 100,000 inputs. Counting: 20,000 natural samples, plus a stratified set of 1000 per count value 0–10. The stratified set is needed because counts of 7 or more almost never occur naturally (P ≈ 1e-5).
- **Methods** (`probe.py`), in their simple standard forms, computed at every answer position:
  - **M1 top-activating examples:** the top 20 (example, position) pairs per post-ReLU unit, plus a list of the task variables that are constant across the top 20.
  - **M2 zero-ablation of a unit:** greedy exact-match accuracy over the whole answer.
  - **M3 write direction through the unembedding:** fc2 column → final-LN weight (centred, no rescale) → head.
  - **M4 attribution:** −activation × gradient, compared with exact zero-patching. The metric is logit[target] − logit[best wrong token in the clean run], teacher-forced.
  - **M5 head ablation:** zero the head's slice before W_o, then take greedy accuracy.
  - **Ground-truth check:** for every unit I wrote down the documented activation function as a function of the task variables (`documented_activation()`) and correlated it with the measured activation. Attention patterns were also checked against the documentation.
- **Outputs.** Per-unit tables are in `out/*.txt` and full records, including the top-20 lists, are in `out/*.json`. `check_count_ties.py` covers the tie-break finding.

## Was the documented mechanism itself right?

Unit activations match the documented formulas: r ≥ 0.998 for every unit that ever fires, on all three programs. The causal story has gaps, though:

1. **Counting: the digit '0' is mostly not written by any circuit.**
   - The tens digit for count < 10, which is 100% of natural inputs, comes out with **all 11 logits exactly 0**. torch `argmax` breaks the tie by taking the first index, and that happens to be token '0'. The documentation says "L1 MLP writes the output digit one-hot (tens at pos11 …)", but nothing is written there.
   - The ones digit for count = 0 is also not written by the MLP. It wins with logit 1 because the previous token's embedding ('0') is still sitting in the residual stream, and the head reads that slot.
   - Evidence: in `check_count_ties.py`, pos-11 logits are all zero, and the tie fraction at the tens position is 1.000 on natural data and 0.909 (= 10/11) on stratified data.
2. **Counting L1.u11 = ReLU(count − 10) is dead.** It is identically 0 because count ≤ 10. The code documents it as part of the c = 9 and c = 0 bumps, and zeroing it changes nothing on any input. The parity run noticed and removed the analogous neuron; this run did not.
3. **Sort L0.u10 (h_0^-, documented as half of clamp(c_0 − k, 0, 1)) has no effect.** Zeroing it changes **0 of 100,000** outputs. When it fires, the answer digit is already 0 and the clamp only trims a correct margin; exact patching shows the margin *rises* by +2.15 without it.
4. **Sort attention is not "effectively uniform over the 5 input positions".** Each input gets 0.184–0.194, and 3–7% of the attention leaks onto '=' and earlier output positions. The circuit tolerates the leak.
5. **Stale headers.**
   - The parity file header describes an older design (d_model = 4, d_ff = 11, triangular-bump lookup writing parity to channel 0). The code is d_model = 3, d_ff = 9, with in-place N − 2·floor(N/2) on channel 2. The comments inside `write_weights` are correct.
   - The counting header gives a d_model = 64 layout, but the model uses 32.
   - The sort class docstring says "3-layer", but the model has 1 layer.
6. **Counting L1 "3-neuron bump … fires iff count == c AND pos == p".** No unit is a bump. Every unit is a ramp ReLU(count − k) gated by position, and the bumps exist only as fc2 combinations of ramps.
7. **Parity h_9 is necessary, but only for one input,** 1111111111 (1 of 2046, or ≈ 1e-4 of the task distribution). Zero-ablating it on the upstream 200-sample eval would very likely show no effect.

## Per-program × method verdicts

Each verdict compares the method's account (which components matter and what they represent) against the documented mechanism after the corrections above. The labels are R = recovered, P = partly, W = wrong, NA = not applicable.

| method | parity | sort | counting |
|---|---|---|---|
| **M1 top-activating** | **P** — all 9 units share top example `1111111111=` and near-identical top lists. That gives "fires on many 1s" (the driver N1), but not each unit's threshold or its alternating role in floor(N/2). | **P** — the top 20 for u_d are always at k = 0 with c_d = 5, so the shared variable gives "#digits ≤ d at early outputs" (≈ c_d − k). The + and − units of a pair have identical top-5 lists, so the −1 offset is invisible. | **P** — L0: recovered (top 20 all query = d, count = 10, i.e. "count of the query digit when it is d"). L1: ramps all peak at count = 10 and pos 12, giving the position gating but not the threshold k. On natural data, 6 L1 units never fire, so their top lists are noise. |
| **M2 unit zero-ablation** | **R** — all 9 units matter, and effect size tracks how often each is active (u0: 1013 errors … u8: 1 error). Exhaustive data is required to see u8. | **R** — 19/20 units matter. It finds u10 redundant, which is a correction to the documentation. | **P** — natural data: L1 u0 and u7–u11 show **zero** effect, which would wrongly report half of L1 as unused. Stratified data: 11/12 matter, and it correctly finds u11 dead. The verdict depends on the data distribution. |
| **M3 write → unembedding** | **P** — each unit pushes '0' or '1' alternately with equal norm (28.3), which is the correct sign of its ±2 fc2 weight. Read as "u0 votes even", it is wrong for most inputs where u0 fires. | **P** — u_d promotes d and suppresses d+1, and the − units do the reverse. That matches the pred_d difference structure but not the condition under which each fires. For u9 and u19 the "bottom token" is a tie artefact. | **P** — L0 units: projection exactly 0 because they write a scalar that L1 reads (correctly "no direct effect", but silent about the role; NA for that layer). L1 u_k: "promotes k+1, suppresses k", which is true of the weights but misleading for ramps. |
| **M4 attribution vs exact patching** | **R for units, W for the head** — units match exactly (r = 1.000 per instance and over units; the path after the MLP is linear). Head: attribution −45.0 vs exact −5.0, a 9× overestimate. | **W** — over units, r = 0.68. Attribution is nearly flat (−19 to −21) for u0–u8 while exact ranges from −22 to −89. For u12–u18 it has the **wrong sign** (+4 to +5 vs −2 to −45). Head: attribution **+67 vs exact −150**, opposite sign, with per-instance r = 0.04. This is the only LayerNorm model. | **P** — L1 units match exactly (r = 1.000). L0 units are overestimated 6–7× on stratified data (−0.67 vs −0.10, per-instance r ≈ 0.8). Heads: 7–22× overestimates. |
| **M5 head ablation** | **R** (trivial) — one head; ablation takes accuracy to 50%, i.e. chance. | **R** (trivial) — one head; ablation leaves 0.12% accuracy. | **P** — both heads are essential (9.1% stratified, 34% natural accuracy after ablation). The two ablations give *identical* accuracy (only count = 0 survives), so the method cannot tell "broadcast the query digit" from "histogram". The attention pattern, which is not one of the five methods, recovers both roles exactly (head 0: 1.00 on position 10; head 1: 0.10 on each of positions 0–9). |

## What a registered version should measure, and how big it should be

- **Validate the ground truth first, as its own stage.** Before scoring any method, have someone other than the method-runner write `documented_activation` and the causal role per unit, then check both against activations and ablations. In this pilot, 6 of the 7 items listed above were stale, unnecessary or wrong documented claims. The ablation reference itself must use exhaustive or stratified inputs.
- **Pre-specify per-method scores:**
  - M2/M5: rank agreement of importance with ground-truth necessity, and recall of dead or redundant units.
  - M1: a judge sees only the top-k examples, writes a one-line description, and is scored against the documented variable and threshold. This should be blind, because my own M1 verdicts here were made with the mechanism in mind.
  - M3: sign and target token vs the fc2 weights, scored separately from "describes when the unit matters".
  - M4: sign agreement and Spearman ρ against exact patching, per unit and per head.
- **Treat data distribution (natural vs stratified over the latent variable) as a design factor.** It flipped the M2 verdict on counting.
- **Treat LayerNorm (and softmax depth) as a factor.** Attribution was exact in the LN-free piecewise-linear parts and failed with the wrong sign in the LN model and on every head. Qwen is the LN case.
- **Size.**
  - The upstream repo has 10 algorithmic runs reaching 100%, with about 100 snapshots in total. Snapshots within a run are near-duplicates, so the independent unit is the *program*, not the neuron (units within a program share one construction).
  - Use all 10 runs (1 final snapshot each, plus 1 structurally different earlier snapshot where one exists), so roughly 15–20 programs and a few hundred units.
  - Heads are the bottleneck: this pilot had 4, all essential. Include boolean-circuit (9 heads), addition, gcd and word-reversal so M5 has more than a handful of cases.
  - With about 15 programs, a per-method "recovered" proportion carries a ±25-point 95% interval at the program level. Report per program, not pooled rates.

## Run time and cost

- **CPU compute** (4 cores): parity 0.4 s, counting 18 s (natural) + 10 s (stratified), sort 129 s (all 100k inputs × 20 ablations), so under 3 minutes total. Installing torch and rebuilding the programs took a few minutes more.
- **The real cost is human/agent reading:** writing the documented activation function and checking the docs took about 10–20 minutes per program here, plus about 15 minutes for the harness.
- **Projected registered run:** about 15–20 programs, each needing its own `variables` and `documented_activation`, plus a blind M1 judge. That is a few CPU-minutes and a few hours of mechanism transcription. No GPU is needed.

## Files

- `programs.py`: rebuilt models and the 3 transcribed `write_weights`; `python programs.py` checks accuracy.
- `probe.py`: the five methods plus the ground-truth checks; `python probe.py [parity sort count count_strat]`.
- `check_count_ties.py`: logits showing that counting emits '0' through tie-breaking and the previous-token embedding.
- `out/`: per-program results (`.txt` summary tables, `.json` full records).

CLOUD-RESULT:
programs tried: parity-upto10-bits (ParityFloorSubD3F9), sort-five-digits (HistogramCircuit_v5_clean), digit-counting-10 (CountCircuitV13); all rebuilt at 100% accuracy.
parity:   M1 partly | M2 recovered | M3 partly | M4 recovered (units, exact) / wrong (head, 9x) | M5 recovered (trivial, 1 head)
sort:     M1 partly | M2 recovered (+ found u10 redundant) | M3 partly | M4 wrong (sign flips on 7 units and on the head; LayerNorm model) | M5 recovered (trivial)
counting: M1 partly | M2 partly (natural data calls 6/12 L1 units unused; stratified correct, finds u11 dead) | M3 partly / NA for L0 | M4 partly (L1 exact, L0 and heads 6-22x over) | M5 partly (both heads essential, roles indistinguishable)
main surprise: in the counting model the most common output digit, the tens-place '0', is written by no circuit. All 11 logits are exactly 0 and argmax's first-index tie-break picks '0'. Every method reads that position as "nothing matters", and the documentation says the MLP writes it.
