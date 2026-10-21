# Part 48 fp32 on a rented 3090 (2026-10-21)

Instance 57089738, RTX 3090, $0.268/h, image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel, transformers 4.57.1
(versions.txt). Rented ~12:38 CT by a fork agent that died at a session boundary before uploading anything; the next
session reused the idle box. First run failed offline (part48.py defaults HF_HUB_OFFLINE=1 for the A770;
run-offline-fail.log); rerun with HF_HUB_OFFLINE=0 at 12:44, rc=0 (run.log). Destroyed ~13:06 CT, GET
/api/v1/instances/ empty. Spend about $0.13 (about 28 min); credit after $8.64.

Cross-device check: Pythia-410m here matches the A770 fp32 run exactly (novel 0.8984, spelled 0.9844).
