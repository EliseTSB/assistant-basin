# Part 107 results — DPO's weight change is 23 to 342 times smaller than SFT's and sits near one bf16 storage step

Run 2026-10-29 on the CPU (`part107/where107.py`, weights only, about three minutes; `../part106/chain.log`), after the
registration commit 85f3a04 (PART107-DESIGN.md). Scorer `part107/score107.py`, output `score107.json`; raw numbers
`where107.json`.

## Relative weight change by component group (Frobenius norm of the change over the earlier weights)

| Group | SFT (mid-training end to SFT) | DPO (SFT to DPO) | RL (DPO to Instruct) |
|---|---|---|---|
| attention q, k, v, o | 0.102 | 0.0044 | 0.0029 |
| MLP gate, up, down | 0.097 | 0.0038 | 0.0025 |
| unembedding | 0.084 | 0.0028 | 0.0014 |
| embeddings | 0.0089 | 0.00015 | 0.000045 |
| layer norms | 0.0081 | 0.00012 | 0.000024 |
| attention q and k norms | 0.0053 | 0.000016 | 0.0000097 |
| final norm | 0.0075 | 0 | 0 |

## Registered predictions

- 107.1 **held**: in every group DPO's change is at least 10 times smaller than SFT's (23 to 342 times; the final norm
  does not change at DPO at all).
- 107.2 **held**: DPO changed 61% of the attention and MLP elements in storage, and 51% of those moved by exactly one
  bf16 step. (RL: 48% changed, 53% of those by one step. The SFT stage is not counted: the mid-training checkpoint is not
  stored in bf16, so its steps cannot be compared.)
- 107.3 **failed**: the median slot reading change is 1.03 times the concept one, and 10 of 21 slot units sit above their
  layer's random-neuron median.

Registered reading: because 107.2 held, 107.3 reads rounding, not DPO's update, and **no claim about where DPO acts is
made from the reading change.** The size result stands: DPO's relative change in the attention and MLP matrices (about
0.4%) is no more than one bf16 step (0.4% to 0.8% of a value) per element on average.

## Read with Part 106

DPO's stored change is small, spread over most elements, and near the storage step; Part 106 shows it did not make the
slot units' pushes pull harder toward their persona. Both point at the same open question: what DPO does to the answers
that compete with the persona sentence.

## Limits

Weights only: a weight change is not a change in function, and attention reads through a softmax the reading change
ignores. One model. bf16 storage rounds away any change smaller than half a step, so the stored change and the true
update can differ.
