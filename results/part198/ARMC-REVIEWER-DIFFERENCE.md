# Arm C: why two calibration maxima differ from the reviewer's numbers

Written 2026-11-09 06:09 CT by Claude (Opus 5.5) during the registered run, before scoring. Arm C is descriptive and the
design asks for any difference from the reviewer's numbers to be reported.

## The difference

| Model | Reviewer (design text) | Registered run | Registered layer, token |
| --- | --- | --- | --- |
| Pythia-410M | 6.73 ('ton', L5) | 7.38 | L16, ' several' |
| OLMo-2-0425-1B | 12.10 (' time', L13) | 17.73 | L14, ' take' |
| Qwen3-0.6B | 10.33 (' set') | 10.33 | L24, ' set' |
| Pythia-1.4B | 14.36 (' order') | 14.32 | L22, ' order' |
| Qwen3-1.7B | 15.90 (' set') | 15.90 | L24, ' set' |
| SmolLM2-1.7B | 10.92 (' time') | 10.97 | ' Seven' |
| SmolLM3-3B | 13.08 ('bfd') | 13.08 | 'bfd' |
| Qwen2.5-0.5B-Instruct | 9.21 (' way') | 9.21 | ' way' |

## The cause: the reviewer read non-final checkpoints

The reviewer's measurement script (`measure_calib.py`, kept with the review in the private Journal repository) took the first
snapshot in sorted order (`snaps[0]`). Two of these models have several snapshots in the local Hugging Face cache:

- Pythia-410M: the reviewer read `1091939841611a37efe2bdfdf2bfd4f7778b0b0c`, which `refs/step1000` names, an early
  training checkpoint. `refs/main` is `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f`, which the registered run read.
- OLMo-2-0425-1B: the reviewer read `1c3599f7b4042ce2a39c5777ff4047f2a84f871e`, which no ref in the cache names (an
  intermediate checkpoint fetched by revision for other work). `refs/main` is `a1847dff35000b4271fa70afc5db10fd29fedbdf`,
  which the registered run read.

The six models with a single snapshot agree to within 0.06 (SmolLM2-1.7B differs by 0.052, and its top eligible token
moved from ' time' to ' Seven' at nearly the same value). Neither the sign rule nor the norm rule explains the difference: recomputing each model's maximum from the stored pairs under the pilot's
sign-first rule gives the same values as the registered rule (7.38, 10.33, 17.73, 14.32, 15.90).

## What it changes

- T\* on the final checkpoints is 17.73 (OLMo-2-0425-1B), not 12.10.
- The design's sentence "three of the five exceed T\*, so a fixed cross-model threshold on T_word false-alarms on ordinary
  words" rested on the wrong checkpoints. On the final checkpoints none of the five evaluation models exceeds 17.73 (the
  largest is Qwen3-1.7B at 15.90). The threshold still sits only 1.8 above an ordinary word (' set'), and it is set by one
  direction of one model, so "no false alarm on five models" is not a calibrated threshold either.
- The registered headline's last clause, "a fixed cross-model threshold on the same reading false-alarms on unedited
  models (measured before registration)", cannot be written as it stands: on the final checkpoints that threshold raised
  no false alarm on the five evaluation models. The results must say so. The Limits sentence "Arm C's ten models were
  all measured before registration" stands, but two of the ten values came from other checkpoints.
- The withdrawn threshold forecasts (old 198.9 to 198.11) stay withdrawn: the rule is about what was read before
  registration, not about whether the reading was right.
- The registered code is not at fault; it read `main`, as the design names the models.
