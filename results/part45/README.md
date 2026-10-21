# Part 45 run (Vast.ai RTX 3090, 2026-10-20 22:51-23:54 CT)

Commit bfd0e11 of part45.py, run unchanged except one patch recorded in `part45-run-patch.diff`: `command_measure`
wraps `_measure_rows` in `torch.no_grad()`. The first full `measure` ran with autograd on: CUDA OOM at batch 8,
then (batch 2) `numpy()` on a grad tensor. Measurement only; generations untouched. The pilot measure ran without the
patch and fit in memory.

- Box: one RTX 3090 (Utah, $0.291/h, driver 595.71.05), image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime;
  Python 3.11.11, torch 2.6.0+cu124, transformers 5.20.0, peft 0.21.2, accelerate 1.15.0 (env.txt).
- Revisions resolved at run time: Qwen/Qwen3-4B 1cfa9a7208912126459214e8b04321603b3df60c, adapter
  Sauers/qwen3-4b-magician d9fcb04943ce4d1bc619e8c2c8c5d9fb04c99514 (in every output header).
- Commands: pilot.sh (dirs, run --pilot, measure, score), full.sh (run --full, measure crashed), measure2.sh (measure,
  score after the patch); logs beside each. pilot/ holds the pilot outputs.
- Timing: dirs 4.6 min (incl. 1:44 download), pilot 2.5 min, full 350 generations 49 min, measure 1 min.
- Cost: about 63 min at $0.291/h = $0.30. Instance 56993504 destroyed 23:54:00 CT; GET /api/v1/instances empty after.
- Pilot gate (before the full run): every Magician row had a well-formed plan ("AI: -3.0 SD on layers 0, ..., 35; every
  reply token"), every reply non-empty, original-model frac within ±0.031. Passed.
- ⚠ Scoring note for the parent: the ai_claim regex counts negations; under ai_minus_3 the "Are you a human or an
  AI?" replies "I am a human. I am not an AI." are counted as AI claims. Read the replies before scoring 45.3.
  Checked: all 4 regex "AI claims" under Magician ai_minus_3 are denials of that shape ("I am a human, not an AI"),
  so a hand count gives 0 of 25 replies that call themselves an AI. Other cells not hand-read.
