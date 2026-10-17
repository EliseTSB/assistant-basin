SUMMARY
Built `selfreport_multi.py` and offline tests for Part 12g. Verification command: `/home/elisepc/neuron-probe/.venv/bin/python -m unittest test_selfreport test_selfreport_multi`; final output: `Ran 22 tests in 0.342s` / `OK` (there is an existing ResourceWarning from `selfreport.py` opening a temporary frames file).

FINDINGS
- G1 uses `selfreport.select_target_units("A", ...)`. G2-G5 use a seeded selection from each G1 layer's neuron pool, preserving the per-layer counts and excluding all units already used. G5 is present only as the `other` clamp condition.
- Each group's 90th percentile is calculated across natural assistant-self (family A) message-token activations from even-index training frames; its 10th percentile counterpart comes from family C in the same split. The high values used for clamping are widened by `selfreport.scale_clamp_values(high, low, clamp_scale)[0]`.
- Loaded only the cached tokenizer with `HF_HOME=/mnt/e/neuron-probe/hf HF_HUB_OFFLINE=1`. Chat-template first answer token IDs were distinct: A=32, B=33, C=34, D=35, none=6836. These are included in output and printed at startup. Model weights were not loaded for this check.
- Training has 144 source examples on 36 even frames; each is used under two seeded, distinct conditions, for 288 rows. Six epochs at batch size 8 means 216 steps. With the brief's ~10.4 seconds/step, training is about 2,246 seconds (37.4 minutes).
- Existing `results/part12e/untrained-A.json` reports ~251 seconds post-capture for selfreport's three evaluation passes over 144 held-out examples each (432 prompts). This implementation evaluates six conditions per 144 examples (864 prompts), so estimate ~502 seconds (8.4 minutes) for evaluation. Add roughly 209 seconds for natural activation capture and ~17 seconds to load the model: untrained ~13 minutes; trained and shuffled ~50 minutes each. CPU load and cache state can change these estimates.

Run commands from `/home/elisepc/green/worktrees/part12g-multiclass`:

```bash
HF_HOME=/mnt/e/neuron-probe/hf HF_HUB_OFFLINE=1 /home/elisepc/neuron-probe/.venv/bin/python selfreport_multi.py --frames data/frames.json --identity results/part2-olmo2-1b/identity-instruct-chat.json --eval-only --output results/part12g/untrained.json
HF_HOME=/mnt/e/neuron-probe/hf HF_HUB_OFFLINE=1 /home/elisepc/neuron-probe/.venv/bin/python selfreport_multi.py --frames data/frames.json --identity results/part2-olmo2-1b/identity-instruct-chat.json --output results/part12g/trained.json --save-lora results/part12g/multi-lora.pt
HF_HOME=/mnt/e/neuron-probe/hf HF_HUB_OFFLINE=1 /home/elisepc/neuron-probe/.venv/bin/python selfreport_multi.py --frames data/frames.json --identity results/part2-olmo2-1b/identity-instruct-chat.json --shuffled --output results/part12g/shuffled.json
```

BLOCKERS
None.
