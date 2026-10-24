uploaded (13:43): code.tar = git archive 96996fb; data.tar = part58 acts/ids/units/sent, part56 pilot, part60 thresholds+run.json; pile10k.jsonl from neuron-probe-gpu/qwen35
part38.py added 13:50 by separate scp (atlas_collect_fast2 imports it)
14:33 rented 56449201 (Quebec, $0.188 listed) for Part 62 generation at 19c9af2 (vast62.py, label part62-run)
14:39 Part 62 box: run62.sh has no package step; installed Part 61's pinned set by hand (transformers==5.20.0 flash-linear-attention==0.5.2 accelerate huggingface_hub, causal-conv1d removed), pip freeze in versions-pip-part62.txt; rerun
15:47 Part 62 box 57416132 destroyed after RETURN_READY (15:45) and a verified download (sha256 cf616b93..., 212 entries) to results/part62-return-2026-10-23/; rented 14:36, ~1.2 h
16:35 Part 64 box 57410468 destroyed (rented 13:43, ~3 h): collection returned and verified earlier (median top-12 match 12/12, sha 184174d4...); scoring after the Opus arms needs a new 3090 and the same upload (code at the run's commit, data.tar, pile10k)
