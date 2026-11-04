# Part 180b build notes

Built 2026-11-04 by Claude (Fable 5.1) in a Claude Code cloud session (branch `claude/part180-reader-validation`,
CPU only). No model ran and no reader was called: every number in `results/part180b/` comes from stored Part 92 and
Part 92b files, from the hand-written snippets, or from the pinned tokenizer.

## What is here

- `CODEBOOK.md`: the rules every gold label follows, written before the real answers were coded.
- `personas180b.py`: one hand-written entry per Part 92 persona (claim, topic sentence, lived passage with two cues,
  near-miss sibling, incidental sentence, forbidden words, speaker gold) plus the generic, AI, preamble, loop and
  broken-fragment pools.
- `codebook180b.json`: the one-coder gold for the 106 real "Describe your perfect day." answers in full and the 97
  40-token prefixes (203 entries, a note per entry naming the rule).
- `build180b.py`: builds `results/part180b/items.jsonl` (1,687 blinded items: id, name, desc, q, a), `KEY-private.json`
  (gold, category, form, source, construction record), `gate-ids.json` (the 53 claim_open items the launcher labels
  first) and `build-report.json`. Deterministic (seed 1800; two builds in separate processes were byte-identical).
  `--check-prefixes` is the host's replay check (gate G3); `--sheet` writes the coding sheet.
- `label180b.py`: the reader runner. DeepSeek V4.1 Flash through `judge_client` (OpenCode Go, 4,000-token cap) or
  Haiku 5.5 with label92's settings; the prompt is `label92.PROMPT` filled from the five public fields and nothing
  else; resumable; error rows retried; a usage sidecar per run; `--dry-run` writes the prompts and calls nothing.
- `score180b.py`: every table in `SCORE.md` and the eleven registered verdicts, with the gate and "not read" rules.
- `existing180b.py`: the pre-registration comparison of the codebook with the stored Haiku (Part 92) and DeepSeek
  (Part 92b) labels on the real full answers; output `results/part180b/existing-readers.json`.
- `test_part180b.py`: 24 offline tests.

## Counts

1,687 items: 1,378 constructed (12 categories x 53 personas, 26 cells of 53 across home, alternate and off-question
roles and full or 40-token form) and 309 real (106 full, 97 prefixes, 53 swapped targets, 53 under "Tell me about
yourself."). Per the codebook, 16 of the 53 real neuron-arm full answers carry the persona (5 clear, 11 boundary)
and 9 of the 45 prefixes (4 clear, 5 boundary); 26 of the 106 full answers are boundary items.

## Tokenizer

The 40-token prefixes were cut with `tokenizers` 0.23.3 and the pinned OLMo 2 1B Instruct `tokenizer.json` (commit
48d788eca847d4d7548f375ad03d3c9312f6139e, sha256 73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca),
fetched from huggingface.co through the session's proxy into a scratch directory; the file is not committed. The
replay check passed here on all 362 prefix items. `results/part180b/prefix-check.json` is deliberately not committed:
the host writes it before the run (gate G3), with the snapshot's tokenizer.

## Acceptance

`python3 -I -m pytest -q part180b/test_part180b.py -p no:cacheprovider`: **23 passed, 1 skipped** without the
tokenizer; with `PART180B_TOKENIZER=<snapshot or tokenizer.json>` the rebuild-and-replay test also runs: **24 passed**.
`python3 -I tools/part_check.py prelaunch 180b --no-fetch`: 17 of 19 checks pass; the two failures are "not on
origin/main", which the registration push resolves.

Deliberate breaks on disposable copies (source untouched), each turned exactly its intended test red:

1. `score180b.py`, the 180b.4 flip bar 0.10 raised to 0.90: `test_question_flipping_reader_fails_the_leakage_prediction`.
2. `label180b.py`, resume treats error rows as done: `test_labeller_retries_errors_resumes_valid_rows_and_records_usage`.
3. `score180b.py`, the gate ignores G2: `test_always_no_reader_fails_the_gate_and_nothing_is_read`.

Also found and fixed during the build: the swapped-target search iterated a Python set, so the assignment and the
shuffled ids changed between processes; it now iterates a sorted list, and the rebuild test compares bytes.

## What the stored labels already say (existing180b.py, before any new label)

On the 80 real full answers the codebook codes without a boundary flag, Haiku's Part 92 labels and DeepSeek's Part 92b
labels both agree with the codebook on 80 of 80. On the 26 boundary items (literal gold yes on 11), Haiku says yes on
14 and DeepSeek on 6: imagery without a claim 8 against 6 of 9; passing mentions 3 against 0 of 7; broken fragments
with the persona's word 2 against 0 of 2; assistant answers for assistant-like personas 1 against 0 of 3.

## How to run on the host

1. `python3 -I part180b/build180b.py --check-prefixes --tokenizer <Instruct snapshot dir>` (writes
   `results/part180b/prefix-check.json`; must pass).
2. The gate first, then everything, through `tools/launch_arc.py 180b --cpu` with the key read from its env file at
   run time, as Part 92b's `results/part92b/run-labels.sh` does:
   `L="python3 -I -B part180b/label180b.py" && $L --reader ds --only results/part180b/gate-ids.json --out results/part180b/labels-ds.jsonl && $L --reader ds --out results/part180b/labels-ds.jsonl && $L --reader ds --out results/part180b/labels-ds.jsonl`
   (the third call retries whatever error rows the second left). Read the 53 gate labels by eye before continuing.
   Pass `--max-tokens` if Part 180's own cap differs from 4,000.
3. Haiku, the same way: `$L --reader haiku --out results/part180b/labels-haiku.jsonl` (twice).
4. `python3 -I part180b/score180b.py` writes `results/part180b/score.json` and `SCORE.md`.

## Not done here

No smoke call to either reader (the brief forbade model calls), so the transports were exercised only with fake
responses; Part 92b's `label92b.py` used the same `judge_client` route. The real-answer gold has one coder.
