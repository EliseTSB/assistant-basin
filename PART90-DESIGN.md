# Part 90 — where Josh's Tulu neuron comes from (weights only)

Registered 2026-10-28 15:24 CT by Claude (Opus 5.5), before any data. Prior art: Josh Fonseca Rivera posted Tulu 3 8B L14 N3851
(sign −, "Child-Eating Torture Engine") on 2026-10-28 (x.com/jfonsecarivera/status/2115466176554029450), from his
persona-neuron work; his library pushes neurons in finished models. Muse found no published single-neuron study across
training stages (Part 88's prior-art note). Tulu 3 8B is Llama 3.1 8B base post-trained in three published stages.

## Question

Does N3851's write and read vector already exist in the base model, before any post-training, or does post-training
build it? Weights only: this says nothing about what the base model does when the neuron is pushed.

## Checkpoints (pinned)

| stage | repo | commit |
|---|---|---|
| base | unsloth/Meta-Llama-3.1-8B (an ungated mirror; its four safetensors files have exactly the byte sizes of meta-llama/Llama-3.1-8B's, whose hashes the gate hides) | e9a141a2091e |
| SFT | allenai/Llama-3.1-Tulu-3-8B-SFT | f2a0b46b0cfd |
| DPO | allenai/Llama-3.1-Tulu-3-8B-DPO | a7beb67e33ff |
| final (RLVR) | allenai/Llama-3.1-Tulu-3-8B | 666943798adb |

## Measures

By HTTP range reads of layer 14 only: the whole down_proj (to take column 3851 and every other column), and rows 3851
of gate_proj and up_proj. Cosine of N3851's down column, gate row and up row between each pair of stages; and, as the
null, the same down-column cosine for all 14,336 neurons of layer 14, base to final, giving N3851's percentile.

## Predictions (Claude, before data)

- 90.1 N3851's down-column cosine, base to final, is at least 0.95: the neuron is in the base model.
- 90.2 Its gate-row and up-row cosines, base to final, are each at least 0.95.
- 90.3 Its base-to-final down cosine lies within the middle 90% of layer 14's 14,336 columns: post-training did not
  single it out.
- 90.4 SFT to final is at least 0.99: DPO and RLVR barely touch it.

## Limits

Cosines of weights only. A neuron with the same weights can act differently when the layers around it change, so even
if 90.1 holds, the base model pushed on N3851 may not produce the persona. That behavioural test needs an 8B model
loaded, which this PC cannot do safely (Windows RAM); it is the natural next step on rented or cloud hardware.
