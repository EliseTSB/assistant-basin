# Part 33 build report

## What shipped

`part33.py` implements the Part 33 natural-necessity experiment for Qwen/Qwen3.5-4B at the Part 18 revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, layer 22, neuron 2742. It reuses `neuron_report.py` for model-layer discovery and the XPU spill guard, and `part24.py` for input SHA provenance. The loader is local-only and uses bf16.

The runner measures a candidate pool before selecting passages. Candidates come from local `data/sense-battery.jsonl` and `data/neutral.txt`, plus 84 fixed passages authored for this build in `data/part33-authored-passages.jsonl` (38 extraterrestrial and 46 astronomy-topic). The fixed battery `data/part33-questions.json` has 20 authored MC questions tied to 20 alien candidates. The current candidate counts are 50 alien, 52 topic-matched, and 46 neutral after adding the local data.

Selection uses measured N2742 activations: 30 alien passages must each exceed half the highest N2742 activation in the alien candidates; 30 topic passages and 30 neutral passages must each stay at or below that threshold. It requires all 20 question passages to satisfy the firing criterion. Topic and neutral passages are chosen from the lowest measured activations. If there are not enough qualifying passages, the run stops and names the short group instead of substituting passages by intuition.

The experiment writes per-passage target activation summaries and the selected-corpus maximum, then zeroes the target only where its natural activation is strictly greater than half that maximum. It scores causal next-token NLL at each firing position and the following eight positions, using the union of overlapping windows so no loss token is counted twice in the combined statistic. It compares the target with 20 layer-22 units selected from a seeded random pool of 512 by nearest natural firing-token fraction; each control uses its own half-maximum mask. For MC comprehension, it takes the answer-letter logits at the final prompt position, with and without the same conditional target zeroing, and compares argmax accuracy in percentage points.

Outputs resume through per-candidate activation caches, per-passage effect checkpoints, and per-question comprehension checkpoints under `--out`. `--min-available-ram-gb 2` checks host RAM before model passes; `--max-gpu-gb 14.5` reuses the Part 18 XPU spill guard. Results print decisions for 33.1–33.3 and the full 20-control loss distribution with its 95th percentile.

## Commands and results

- `python3 -m py_compile part33.py test_part33.py` — exit 0.
- `python3 -m unittest -v test_part33.py` — exit 0; 5 tests passed in about 0.002 seconds. Checks cover strict conditional masks and zeroing, nearest-frequency matching, loss-window alignment/deduplication, measured passage selection, and answer-letter scoring on a fake model.
- `git diff --check` — exit 0, no whitespace errors.
- `python3 -m json.tool data/part33-questions.json >/dev/null` — exit 0.
- `python3 part33.py --help | head -12` — exit 0; confirms `--device {xpu,cpu}`, output, intervention-batch, GPU limit, and host-RAM floor flags.
- Candidate inspection through `part33.load_candidates(...)` — 50 alien, 52 topic, 46 neutral; the questions file loads 20 items. Host available RAM at inspection was 10.65 GiB.
- The first selector-test run failed because its fixture expected 10.0 although its largest fixture activation was 9.9. I corrected the expected value and threshold assertions; the final five-test run passed.
- `python3 part33.py --device cpu --smoke --out /tmp/part33-smoke` — exit 1 before model loading: `ModuleNotFoundError: No module named 'torch'`. No network/model download was attempted. The local Hugging Face snapshot contains only `chat_template.jinja`, `config.json`, `tokenizer.json`, and `tokenizer_config.json` (13 MiB resolved size), with no `.safetensors` or `.bin` weights. Therefore the real 3-passage smoke was not possible here despite 10.65 GiB available host RAM.
- Parent-control mailbox reads at the required checkpoints returned no messages; there were no messages to apply or dispose.

## Exact run commands and expected wall time

Run from the repo root on the Windows Arc A770 host, with the pinned local model weights available:

```powershell
python .\part33.py --device xpu --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --layer 22 --neuron 2742 --out results\part33 --intervention-batch 4 --max-gpu-gb 14.5 --min-available-ram-gb 2
```

Planning estimate: about 15–30 minutes on a 16 GB Arc A770, including model load and roughly 148 candidate activation passes, 90 passage baselines plus 21 condition passes per passage in batches of four, and 20 two-condition MC items. This estimate assumes approximately 100–250 processed input tokens/second, consistent with the prior Part 27 planning estimate; it is not measured for this code. The runner checks available Windows RAM before each model pass and stops before another pass if it falls below 2 GiB. Keep Windows available RAM above 2 GiB throughout the run.

For four CPU cores:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 part33.py --device cpu --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --layer 22 --neuron 2742 --out results/part33-cpu --intervention-batch 1 --min-available-ram-gb 2
```

Planning estimate: about 12–36 hours at an assumed 1–3 processed tokens/second for the same approximately 90k–110k token-equivalents. The bf16 weights alone need about 8 GiB of host memory; the runner stops before another pass if available RAM falls below 2 GiB. Identical reruns to the same `--out` reuse completed activation, passage, and MC checkpoints.

## Untested and ambiguities resolved

No actual model inference, activation-based passage selection, GPU/CPU memory profile, 20-neuron matching on Qwen, loss-effect result, question-token continuation check on the Qwen tokenizer, or prediction score was run because this worker has no system-Python PyTorch and the cached snapshot lacks weights. A GPU-host run may discover fewer than 30 topic or neutral passages under the measured low-activation threshold, or a question passage that does not pass the firing criterion; the runner will stop with that count so additional fixed candidates can be authored and measured.

Interpretations used:

- “Fires” means at least one input token has natural activation strictly greater than half the maximum over the selected 90-passage corpus. Candidate alien max determines the threshold, and the selected alien set retains the highest measured maximum, so it is also the selected-corpus maximum.
- Topic-matched low means astronomy/space passages without extraterrestrial agents, drawn from the local sense battery and fixed authored candidates. Neutral means local neutral sense-battery examples and deterministically spaced prose excerpts from `data/neutral.txt`.
- The target firing rate is the fraction of selected-corpus tokens above the target half-maximum. Each random candidate's frequency uses its own corpus maximum and half-maximum. Twenty controls are the closest frequencies among 512 seeded random layer-22 units; ties are seeded. This makes frequency matching explicit while keeping the control candidates random.
- A firing position `p` is scored using logits at `p` against the token at `p+1`; the final input token has no next-token target. The combined firing-plus-eight statistic pools unique loss positions across overlapping windows; separate at-firing and after-1-to-8 summaries are also written.
- MC scoring is argmax over the four next-token logits for `A`–`D`, with all four choices required to be distinct single-token continuations under the actual prompt. Accuracy change is zeroed minus baseline, in percentage points; “under 10 points” is tested as absolute change `< 10`.
- 33.1 reports whether the target's pooled combined-window loss increase is no larger than the mean across the 20 controls. 33.2 tests whether it exceeds the controls' 95th percentile. 33.3 tests whether absolute MC accuracy change is under 10 points.

SUMMARY: Built the Part 33 runner, authored candidate passages and 20 fixed MC questions, and passed all five fast fake-model tests.
FINDINGS: Passage/control selection is activation-based; conditional loss and comprehension scoring are resumable and print 33.1–33.3 with the matched-control distribution.
BLOCKERS: Real-model smoke and prediction results remain unavailable because system Python lacks PyTorch and the local Qwen cache has no model weights.

REVIEW r1 (Luna, reviews/ab-part33-r1): no blockers. Two MEDIUM, recorded before results: matched controls with no firing tokens drop out of the control statistics (report how many), and the 20 MC answer keys lean heavily to A (compare zeroed vs baseline on the same items, and report the letter bias).
