---
title: The Arc A770 stalls for two minutes at a time inside atlas pass two, and the copy engine is the lever
description: Why the layers 7-13 atlas group never finished (three tries, 10-22 to 10-24); a bisection on the Arc found ~150 s stalls with the GPU idle, not caused by priority, trimming, sync count or allocator churn, and gone with the Level Zero copy engine off.
authored-by: claude
date: 2026-10-24
session: 211ba227
status: fix confirmed on the real run (54 min, was heading for 8-20 h); job.py defaults it on
---

# The Arc stalls for two minutes at a time, and the copy engine is the lever

## What was wrong

The atlas group for layers 7-13 (Qwen3-1.7B, `part51/atlas_collect_fast2.py`) never finished. Two runs on
10-22 and 10-23 were stopped as "stalled". The third, started 10-24 00:28, was running but at about 5 batches
a minute, an ETA of 8 to 20 hours, and was stopped at 01:22. The pass-one checkpoints were kept each time.

The signs used to call the first two stalled (a quiet log, no pass-two files, a 0.1 GB working set) are all
normal for this script: pass two writes nothing until every batch is done. `scripts/neuron-gpu-job/atlas-progress.sh`
now reads the live batch with py-spy instead.

## What the bisection found

All runs on the A770, Windows, torch 2.13.0+xpu, Intel driver 32.0.101.9150 (2026-10-07). Logs in this folder.

| Test | Result |
|---|---|
| py-spy, 60 s on the live run | 100% of wall time at `atlas_collect_fast2.py:357`, the `.cpu()` after `index_select` |
| One layer at a time, 12 batches each (diag1) | layer 10: 0.16 s/batch, layer 3: 0.10 s; the line-357 gather 0.3-0.4 ms |
| 7 layers, task scheduler, BelowNormal (diag2) | median 0.90 s/batch; stalls of 102 and 180 s in 20 batches |
| 7 layers, launched by interop, Normal (diag2) | median 0.90 s/batch; 4 stalls of 142-173 s in 80 batches |
| Sampling during a stall | no CPU, no page faults, no I/O; NO process on any Arc engine; working set intact (3.07 GB) in the Normal run |
| 120,000 fixed-size nonzero + index_select + .cpu() syncs | 0 stalls, median 0.64 ms |
| 120,000 varying-size syncs (rows 100-2048) | 0 stalls, median 0.80 ms |
| Model forward to layer 13, no recording, 300 batches | 0 stalls, 0.05 s/batch |
| Same forward, one small sync inside each of 7 hooks, 300 batches | 2 stalls (124, 164 s) |
| Recording moved after the forward ("stash"), 20 batches | stalls (529 s for 20); outputs identical to inline (161 arrays) |
| **7 layers, Normal, `SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0` + `UR_L0_USE_COPY_ENGINE=0`** | **0 stalls in 80 batches, 73 s total** (the same batches took 718 s with 4 stalls) |

## What it means

- **Not priority**, not working-set trimming, not the number of syncs, not allocator churn, not the forward pass,
  and not where the recording sits relative to the forward. Each of those was tested and failed to explain it.
- The stalls last 100-180 s with the GPU idle, which looks like a missed completion recovered on a timeout.
  The zero-stall copy-engine run fits that: with the copy engine off, device-to-host copies run on the
  compute engine. Probability of 0 stalls by chance at the measured rate (4 per 80): about 2%.
- **Every atlas group was probably stalling.** Layers 0-6 at batch size 4 took 2 h 45 m in pass two where
  ~40 min of work was expected. The "about 80 minutes" estimate in the 10-23 handoff came from layers 14-20
  at batch size 8, so it was never the right estimate for a batch-4 run.
- **The scheduled task has `ExecutionTimeLimit` PT6H** and no `<Priority>` (so 7, BelowNormal). A slow run
  through it is killed at six hours. Launch long jobs by interop (`cmd.exe /c ... job.py`) instead, which
  keeps job.py's RAM guard.

## Not established

- Why the copy engine stalls only under the collector's mix of ops, when 240,000 plain copies never did.
- Whether other A770 jobs (Part 64 scoring, the 10-17 "16b STALLED" run) have the same cause. The 10-17
  description (CPU frozen, no GPU engine use, no spill) matches.

## Atlas L07-13 rerun with the copy engine off

Started 02:17:38 by interop with both variables set; finished rc=0 at **03:11:22, 54 minutes** for pass two of all
seven layers (the stalled 00:28 run was at 4% of tokens after 50 minutes). Batch position by atlas-progress.sh:
02:23 batch 291 (2.5% of tokens), 02:28 620 (8.7%), 02:33 903 (16.8%), 02:38 1157 (26.3%), 02:43 1383 (37.1%).
job.py's GPU line: 477 samples, mean 14%, busy (>5%) 95% of the time. All four atlas groups now exist: 172,032
neurons and 28,000 controls. job.py sets both variables by default since 94da5cb0d.
