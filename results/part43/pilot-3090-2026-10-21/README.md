# Part 43 pilot on a rented RTX 3090 (2026-10-21) — stopped at a required check

The pilot could not run on the A770: Windows' RAM guard stopped every Qwen3.5-4B load (03:52, 04:11, 07:53 CT). It ran
on one Vast.ai RTX 3090 (Quebec, $0.31/h) instead, with part43.py at bb39041 (sha256 8e08be62…, the same file the Arc
job folder holds and the same `tool_sha256` as the Arc's 04:11 manifest). All 11 hashed inputs matched
(`inputs.sha256`, `sha256sum -c` on the box). transformers 5.20.0, torch 2.6.0+cu124, flash-linear-attention 0.5.2; no
causal-conv1d (reference fallback). Commands: `run43.sh`.

- `manifest`: rc 0. Same tool hash and input hashes as the Arc's 04:11 manifest; the only content difference is the
  tokenizer class name (`Qwen3_5Tokenizer` on transformers 5.20.0 against `Qwen2Tokenizer` on the Arc's 5.13.0).
- `pilot`: **stopped by its own first required check**, in `run_validation_checks`:
  `AssertionError: batched/single logits differ by 0.75, tolerance 0.1` (bf16). Two of Part 25's sequences, padded
  into one batch, give logits up to 0.75 apart from the same sequences run one at a time. The mask/position alignment
  check before it passed. Nothing after it ran: no signs, lattice, swap or attach, so there is no pilot.json and no
  timing.
- Not established: whether 0.75 is bf16 accumulation on this card and attention path (padding changes the reduction
  order; the reference conv fallback and the linear-attention kernels are the likely sites), or a padding bug in the
  hybrid model's linear-attention layers, which a mask does not reach the way it does for full attention. Same open
  question as fastgen's batched-vs-single check on the A770 (3 of 12 rows differ in bf16).

No full section starts until this is understood and both reviewers (Chorus, Alden) have seen it.
