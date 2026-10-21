# Answer to the Part 42 HOLD (Claude, 2026-10-20)

1. Provenance: treat /home/elisepc/neuron-probe/jlens/SHA256SUMS plus the lens's n_prompts, source_layers and d_model,
   and the config.yaml beside each .pt, as the provenance. Record all of them in read.json. Do not wait for another file.
2. Which J for an MLP unit's write: read the estimator docstring in
   /mnt/c/Users/alexa/jlens-tools/jacobian-lens/jlens/fitting.py (and where it captures activations) and state in the
   report which residual stream J[l] reads (block l's input or its output). An MLP unit of block l writes into block
   l's output stream. If J[l] maps block l's INPUT stream, the right map for that write is J[l+1]; if it maps block
   l's OUTPUT stream, use J[l]. Apply the same rule to every unit and test it.
3. Units whose write lands in the final block's output (no J exists for it): the J lens there is the identity, so the
   J-lens readout equals the logit lens. Read them that way, mark them `j_identity: true`, and exclude them from 42.1
   and 42.2 (they would inflate the overlap). Keep them in the Part 39 set for 39.9.
Then finish the build exactly as BRIEF-42.md asks, including the real read and score on WSL CPU.
