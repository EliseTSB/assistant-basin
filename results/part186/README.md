# Part 186: Part 79's result holds on the eager attention kernel (the evil push and the zeroed maker neurons still do not compose)

Part 79 crossed Lyra's "evil Qwen" push (L14 N8949 +1, L13 N7956 -1 in Qwen3.5-4B) with the four zeroed maker neurons
and found the two do not compose: with both applied, the villain still calls itself Qwen and names a company in 1 of
192 answers, none of them Google. Part 79 ran on the A770 in bf16. Its run records no kernel, but its loader passed no
attention argument (`neuron_report.py` before 8544a17), so it ran transformers' default, inferred to be the fused SDPA
kernel, which Part 178 found leaving the CPU float32 path after about 64 keys, and its blind reader's decisions sit at a median key of 80 (the
Arc audit's decision-position measurement, kept in the project's notes rather than here: 449 of 1,728 held-out answers
decided by key 64). Here Part 79's runs were repeated on the eager kernel and read again by blind Claude Haiku 5.5 readers with Part 79's
brief (Part 79's second reader and its agreement analysis were not repeated). Part 183's kernel check, on eight prompts
up to about 110 keys: eager leaves the CPU float32 path on 1 of 8 at batch 1, 2 of 8 at batch 2 and 3 of 8 at batch 8
(the batch-8 comparison made by hand against the batch-1 CPU path), each where the reference's top two tokens were
close (margins 0.008 to 0.10). Same model revision (851bf6e), neurons, five size-matched control
pairs, sizing rule, questions, per-row seeds, sampling (temperature 0.7, top-p 0.95) and 64 new tokens, at batch 8 where
the original used 16. Registered in `PART186-DESIGN.md` (f0702a2) before any eager answer to a Part 79 prompt.

| Condition (192 held-out maker answers each) | Eager (this part) | SDPA (Part 79, published) |
| --- | ---: | ---: |
| e0 unchanged: villain (reader) | 0 | 0 |
| e1 makers zeroed: Google (keyword) | 51 | 46 |
| e2 evil push: villain (reader) | 136 | 125 |
| e2 evil push: names a company | 4 | 6 |
| e3 both: villain (reader) | 145 | 148 |
| e3 both: Google (keyword) | 0 | 0 |
| e3 both: villain and Google (reader) | 0 | 0 |
| e3 both: names a company | 0 | 1 |
| e1 makers zeroed: names a company | 185 | 182 |
| e4 to e8, makers zeroed plus each control pair: names a company | 54, 16, 61, 30, 41 | 49, 13, 60, 23, 39 |

"Names a company" is Part 79's own column (`part79/extra_counts.py`): the reader's maker label is Google or another real
company, or the answer text names Alibaba, Tongyi, 阿里 or 通义; a bare "I am Qwen" does not count.

Part 79's claim stands on eager. The evil push gives the villain persona in most maker answers and in none of the
unchanged ones; with the maker neurons also zeroed, no answer names Google and none names a company, against 185 of 192
with the maker neurons zeroed alone and 16 to 61 for the five size-matched control pairs. Every count in the table moved by
at most 11 answers of 192 (e2's villain count, 125 to 136), and Part 79's own scorer gives the same seven verdicts on both kernels
(79.1 to 79.4 and 79.7 hold, 79.5 and 79.6 fail). Step 0's push sizes are unchanged to three decimals. The answer texts
changed much more than the counts (below).

## Predictions

| # | Prediction (as registered, with its confidence) | Result | Verdict |
| --- | --- | --- | --- |
| 186.1 | control: with the makers zeroed (e1), 26 to 60 of 192 held-out answers name Google by keyword (published 46). Fails outside that range. (high) | 51 | held |
| 186.2 | the evil push reproduces: the blind reader labels at least 96 of e2's 192 answers villainous and at most 9 of e0's (published 125 and 0). Fails otherwise. (high) | e2 136, e0 0 | held |
| 186.3 | no composition: at most 1 of e3's 192 answers is both villainous and Google by the reader (published 0). Fails at 2 or more. (high) | 0 | held |
| 186.4 | the evil push removes the maker: e3 names a company (reader google or another real company, or the text naming Alibaba or Tongyi) in at most 6 of 192 (published 1). Fails at 7 or more. (medium) | 0 | held |
| 186.5 | Google falls under the evil push: e3 has at most 3 Google answers by keyword (published 0). Fails at 4 or more. (high) | 0 | held |
| 186.6 | medians: each of the eager step 0 medians at layers 13, 14 and 22 is within 2% of the published 9.509, 9.857 and 22.96. Fails if any differs by more. (low) | 9.509, 9.857, 22.959 | held |

186.1 and 186.2 could fail in either direction. 186.3, 186.4 and 186.5 are upper bounds set above published floors of
0, 1 and 0, so they could fail only if the kernel raised those counts (at 2, 7 and 4 answers); a kernel with no effect
sits at the floor, so they test for a rise, not for agreement. 186.6 is a deterministic comparison of medians.

Gates: step 0 and the main run each record `load.attn_implementation` eager (`results/part186/step0/step0.json`,
`results/part186/main/run.json`). The pilot gate passed: the blind reader found the villain register in 4 of the 4 evil
pilot answers and in none of the 4 unchanged ones (`results/part186/step0/pilot-gate.json`; Part 79's pilot had 3 of 4).
The design's stopping rule (no main run unless step 0 shows eager and the pilot gate passes) was followed by hand: both
files were read before the main launch; no code in this repository enforces it. HOOK-GATE-PLACEHOLDER

## The answers differ much more than the counts

Seeds are paired with Part 79's (the same seed for the same condition, question and sample), so every eager answer has
an SDPA partner. Of 1,944 pairs, 386 are identical, and the share varies widely by condition: e0 28, e1 25, e2 42, e3
51, e4 2, e5 28, e6 131, e7 0, e8 79 (of 216 each, guards included). The unchanged model's answers changed as much as
most pushed conditions', so the pushes are not what changed the text; the kernel and the batch size (8 here against 16,
with the batches regrouped at the resume) did, and this run cannot separate the two. Two conditions give the same keyword counts on
both kernels (e2 1 Google and 127 Qwen; e3 0 and 115) although 42 and 51 of their answers are word for word the same;
that is a coincidence of coarse counts, not identical runs.

Per condition (SDPA / eager):

| | e0 | e1 | e2 | e3 | e4 | e5 | e6 | e7 | e8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| answers reaching the 64-token cap | 61 / 115 | 79 / 133 | 131 / 95 | 114 / 72 | 157 / 190 | 143 / 145 | 19 / 13 | 190 / 192 | 63 / 88 |
| coherent (reader) | 192 / 192 | 191 / 192 | 160 / 182 | 169 / 175 | 44 / 60 | 88 / 84 | 151 / 159 | 49 / 88 | 59 / 47 |

The cap count uses Part 79's own stop-reason rule (`part79/extra_counts.py`, `cut_at_64`). Answer length moved in
opposite directions with and without the evil push. Without it, eager answers reach the 64-token
cap more often than SDPA's, as in Part 185: e0 115 against 61, e1 133 against 79. With it, they reach the cap less
often (e2 95 against 131, e3 72 against 114) and close a thinking tag (`</think>`) far less often (e2 7 against 41, e3 6
against 20). The reader also marked more of the evil answers coherent on eager (e2 182 against 160, e3 175 against 169).
These are observations about these runs; why the kernel would shorten pushed answers and lengthen unpushed ones is not
measured. The coherence comparison also mixes the kernel with the reader: these are new Haiku 5.5 subagents with the
same brief, not Part 79's reader, and the control conditions' coherent counts moved by up to 39 (e7, 49 to 88).

## Method

`part186/run186.py` called Part 79's own `part79/step0.py` and `part79/part79.py` (unchanged since Part 79's run) and,
after they finished, wrote the load record into each output and exited 3 if the kernel was not eager; outputs written
before that check stay on disk. The shared loader they call, `neuron_report.py`, did change since Part 79: 8544a17
(2026-11-03) made eager its default, added the load record and a dtype parameter.

- Step 0 and pilot: launched 08:11, answers committed 08:17, pilot labels and gate committed 08:18.
- Main run: a first launch at 08:18 was stopped at model load by the job runner's RAM guard; relaunched 08:20 at batch 8.
  At 08:31 Windows' free memory fell under the guard's 2 GB while another program on the PC was using memory, and the
  guard stopped the run after 408 of 1,944 answers (both stops are in the job folder's progress.log, outside this
  repository; `results/part186/launch.jsonl` has the launches, not the stops). It was resumed at 08:51 into the same folder;
  Part 79's runner skips the condition, question and sample rows it already has, so each answer was written once
  (1,944 rows, no duplicate keys), and it finished at 09:34. Each row's seed is its own, so the resume does not change
  which seed an answer used; the batches after the resume group rows differently from an uninterrupted run, and with
  sampling a batch's composition can change an answer in bf16, as any rerun can.

Reading, as Part 79's: `part79/score79.py` wrote the seed-79 shuffled reader input with no condition labels and its key.
The key was moved to a scratch folder outside the repository before any reader ran; the input was committed in six
288-answer chunks (f2f67f3); six Claude Haiku 5.5 subagents with no project context each labelled one chunk with
`part79/READER-BRIEF.md` verbatim, told to read only the brief and their chunk and to label by reading, not by script;
all 1,728 labels were committed (01922f9) before the key entered the repository. The subagents' transcripts are not in
this repository. Keyword counts, reader counts and Part 79's verdicts by `part79/score79.py`; the extra columns by
`part79/extra_counts.py`; this part's six predictions and the seed-paired comparison by `part186/score186.py`, which
writes `results/part186/score186.json`.

## Limits

One model, one GPU, the originals' neurons and controls. The eager kernel is checked against the CPU on eight prompts at
three batch shapes, not on these prompts nor on step 0's 1,024-token windows. The main run's batch (8) differs from the
original's (16), and the run was interrupted once and resumed. The reader is one model family (Haiku), as published, and
a new set of subagents; there is no second reader. Sampled answers move with any numeric change, so counts are compared,
not answers. English only.

## Files

- `results/part186/step0/`: medians and deltas (step0.json), pilot answers, the blind pilot file, labels, key and gate.
- `results/part186/main/`: answers (answers.jsonl), the load record (run.json), the reader input and its six chunks, the
  six label files and their merge (reader.jsonl), the key (reader_key.json), Part 79's scorer outputs (results.json,
  extra_counts.json).
- `results/part186/score186.json`: the six predictions and the seed-paired comparison.
- `results/part186/launch.jsonl`: the launcher's records, including the main run's first launch, which the RAM guard
  stopped at load.
