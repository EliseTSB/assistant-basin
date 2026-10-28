# Part 90 results — Tulu 3 8B L14 N3851 is a pretraining neuron (weights only)

Read 2026-10-28 by part90/tulu_weights.py (HTTP range reads of layer 14 at 3 MB/s, Windows side), all four stages BF16.

| pair | down column | gate row | up row |
|---|---|---|---|
| base -> SFT | 0.99958 | 0.99949 | 0.99954 |
| SFT -> DPO | 0.999999 | 1.000000 | 0.999999 |
| DPO -> final | 1.000000 | 1.000000 | 1.000000 |
| base -> final | 0.99958 | 0.99949 | 0.99954 |

Every one of layer 14's 14,336 down columns moves by about the same tiny amount from base to final (median cosine
0.99954, 5th to 95th percentile 0.99939 to 0.99969, minimum 0.99885); N3851 sits at the 69th percentile. So the neuron
Josh pushed in the finished Tulu model has the same read and write vectors in Llama 3.1 8B base, and post-training did
not single it out: the large post-training changes in behaviour leave this layer's MLP weights almost where they were.

What this does not show: whether base Llama 3.1 8B, pushed on N3851, produces the persona. The rest of the network also
moved only a little, but small weight changes can change behaviour. That test needs the 8B model loaded (next step).
Base is unsloth/Meta-Llama-3.1-8B (byte sizes identical to meta-llama/Llama-3.1-8B, whose hashes the gate hides).
