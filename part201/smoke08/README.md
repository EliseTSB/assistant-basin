# Smoke fixture for Qwen3.5-0.8B

Twelve neurons and twelve random directions at two layers of the 0.8B, with a made-up median of 1.0. They exist so
`run201.py --smoke` can run its whole path (three processes, the timing probe, a relaunch) on a real small
Qwen3.5 on the CPU. They mean nothing as an experiment.

    systemd-run --user --scope -p MemoryMax=8G /home/elisepc/neuron-probe/.venv/bin/python -I -B part201/run201.py --smoke \
      --device cpu --model /mnt/e/neuron-probe/hf/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17 \
      --units part201/smoke08/units.json --medians part201/smoke08/medians.json --out /home/elisepc/neuron-probe/out/smoke201 --max-new-tokens 24

Then score it (pass K is absent, so 201.7 reads "not run"):

    /home/elisepc/neuron-probe/.venv/bin/python -I -B part201/score201.py --rows /home/elisepc/neuron-probe/out/smoke201/rows.jsonl \
      --load /home/elisepc/neuron-probe/out/smoke201/load.json --plan /home/elisepc/neuron-probe/out/smoke201/plan.json \
      --probe /home/elisepc/neuron-probe/out/smoke201/probe.json --out /home/elisepc/neuron-probe/out/smoke201

Read from it: Gate G4's `hook_dtype` in the float32 passes (is the activation at the push hook float32 after the
upcast?), pass A's answers per second with the recorder on, and that every gate the smoke can reach passes. Run it
only when Windows has several GB of RAM free: WSL's memory comes out of Windows'.
