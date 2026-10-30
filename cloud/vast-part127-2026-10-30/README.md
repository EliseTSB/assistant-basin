# Part 127 on Vast, 2026-10-30

Part 127 uses one rented RTX 3090 because the A770 is held by another run. The pilot and main capture run on that same machine, with bf16 on CUDA and batch 8 for every arm and the fixed-path gate. No answer is reused from Part 92 or an A770 run; every arm is generated on this machine. Reader labels run locally after copy-back.

## Cost and machine limits

The helper accepts one verified, on-demand RTX 3090 with reliability at least 0.99, at least 40 GB disk, and a total hourly rate no higher than $0.30. At that rate, the design's 15-minute capture estimate costs about $0.08 for GPU time; model download and startup add rented time. The helper stops after 80 minutes from the rent request, which caps the rental at $0.40 at the maximum accepted rate. Disk is set to 40 GB.

## Run

Make `VAST_KEY` available in the local process environment without printing it, then run:

```bash
python3 cloud/vast-part127-2026-10-30/vast.py
```

The helper searches the v0 offers endpoint, then rents one qualifying offer. It uploads the Part 127 runtime files and the registered neuron inventory, waits for SSH, and runs `run.sh`. `run.sh` installs `transformers==5.13.0`, the version recorded by the Part 125 pilot, fetches the OLMo 2 1B Instruct snapshot at revision `48d788eca847d4d7548f375ad03d3c9312f6139e`, and runs the pilot followed by the main stage. Both use batch 8 and `--device cuda`; the model loader uses bf16.

The helper copies `results/part127/pilot-b8` and `results/part127/main` back into the local repository. It requests destruction on every exit path and prints `P127_INSTANCE_REMAINS=yes/no` based on the v1 instance listing. A failed or malformed v1 listing counts as `yes`, never as an empty list. The key is read only from `VAST_KEY`, removed from child process environments, and never printed.

After successful copy-back, run the reader stages on the local machine:

```bash
/home/elisepc/neuron-probe/.venv/bin/python -I -B part127/label127.py --phase gate --out results/part127/main
/home/elisepc/neuron-probe/.venv/bin/python -I -B part127/label127.py --phase controls --out results/part127/main
/home/elisepc/neuron-probe/.venv/bin/python -I -B part127/label127.py --phase study --out results/part127/main
/home/elisepc/neuron-probe/.venv/bin/python -I -B part127/score127.py --out results/part127/main
```

The Part 27 Vast notes record that a `causal-conv1d` wheel failed to import because of an ABI mismatch and Transformers used its slower reference fallback. This run does not install that optional package; the OLMo 2 capture does not need it.
