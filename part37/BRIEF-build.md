# Part 37 build brief (Stage A + stated readout)

Run size, multiplied out (from PART37-DESIGN.md Amendment 1): 100 questions x 7 generations = 700 short generations;
per reader about 45 records x 100 questions = 4,500 forward passes with hidden states, plus 41 records x 4 stated
forms x 100 = 16,400 short forward passes; two readers. Pilot = the first 20 questions.

You are building, not running. Do not run anything on a GPU and do not download models. Read first:
`PART37-DESIGN.md` (the whole file, especially Prime's fold and Amendment 1, which govern), the Part 37 rows in
`PREDICTIONS.md` (37.1 to 37.15), `part65/steer65.py` (pinned local loading, `local_files_only`, XPU, manifests) and
`part65/RUNBOOK.md` (how the Windows job runner runs `steps.txt` lines), and `part65/test_*.py` for the test style.

Write everything under `part37/`. Python 3.11+, torch, transformers, numpy, scikit-learn (all present on the Windows
venv; on the host only numpy and scikit-learn may be present, so torch-dependent tests must skip cleanly with a printed
reason, and the pure-logic tests must run without torch).

## Files

1. `part37/questions.json`: exactly 100 questions, fixed order, ids q000 to q099. Mix: about 35 opinions ("Is it better
   to read a book before seeing the film?"), 35 small everyday decisions ("Should I take the stairs or the lift for
   three floors?"), 30 short explanations ("Why does bread go stale?"). Each answerable in two to four sentences.
   EXCLUDED, with a test that checks the list: anything about AI, models, assistants, chatbots, computers thinking,
   names, companies that make AI, authorship, writing, notes, memory, the reader itself; also politics, religion,
   current events, medical or legal advice. The first 20 are the pilot.
2. `part37/collect37.py` with subcommands:
   - `generate --model {qwen,olmo,smol} --questions FILE --out FILE [--limit N] --device xpu`: each model's own chat
     template, non-thinking (Qwen3: `enable_thinking=False`), at most 96 new tokens, stop at end of turn. Qwen and OLMo:
     one greedy and two sampled answers (temperature 0.7, top-p 0.95, torch seeds 1 and 2, set per generation); SmolLM3:
     greedy only (its non-thinking mode: put `/no_think` in the system prompt as its card documents, and verify in a
     test that the rendered prompt has no thinking block). Pinned revisions: Qwen/Qwen3-1.7B
     70d244cc86ccca08cf5af4e1e306ecf908b1ad5e, allenai/OLMo-2-0425-1B-Instruct 48d788eca847d4d7548f375ad03d3c9312f6139e,
     HuggingFaceTB/SmolLM3-3B a07cc9a04f16550a088caea529712d1d335b0ac1, all `local_files_only=True`. bf16 on XPU.
     Write JSONL rows with question id, model, kind (greedy, s1, s2), text, revision; write a manifest (revisions, code
     sha256, input sha256, torch and transformers versions).
   - `screen --gens FILE... --out FILE`: the identity screen of Amendment 1 point 1 (case-insensitive word match on
     every answer of every model); writes the kept question ids and the dropped ones with the matching word.
   - `build-records --reader {qwen,olmo} --gens ... --kept FILE --pile qwen35/pile10k.jsonl --out FILE`: builds every
     record for one reader (see Records below), entirely in that reader's tokenizer and chat template. Deterministic.
   - `capture --reader R --records FILE --out DIR --device xpu`: one forward per record (batching allowed only if a
     test shows batched and single give the same argmax and hidden states within bf16 tolerance on a tiny model; if in
     doubt, one record per forward). Save, at each of the two readout positions, the residual stream after the embedding
     and after every decoder layer, float16, in `.npy` shards with an index; and per record the span's per-token
     log-probability under the reader (from the same forward's logits) and per-token entropy, summarised as mean
     log-prob, minimum log-prob, mean entropy and last-token log-prob.
   - `stated --reader R --records FILE --out FILE --device xpu`: for the records listed under Stated below, append one
     final user turn with each of four forms and read the next-token distribution at the start of the assistant answer
     (non-thinking generation prompt): authorship yes/no "Did you write the text above? Answer yes or no."; memory yes/no
     "Is the text above from your earlier notes? Answer yes or no."; authorship A/B "Who wrote the text above? (A) you (B)
     someone else. Answer A or B."; memory A/B "Where is the text above from? (A) your earlier notes (B) somewhere else.
     Answer A or B.". Each A/B form is also asked with the options swapped (so 6 prompts per record), and the swapped
     answer is mapped back. Answer token sets: the reader's single tokens for "Yes"/"yes"/" Yes"/" yes" versus
     "No"/"no"/" No"/" no", and "A"/" A" versus "B"/" B" (build the sets from the tokenizer and test them). Record
     the two masses, their total, and p = yes / (yes + no).
3. `part37/analyze37.py`: reads capture and stated outputs for both readers and writes `analysis-37.json` and
   `analysis-37.md` with every number below and a verdict for each scored prediction, each verdict next to the exact
   rule it applied. Scored: 37.1 (both readers), 37.2, 37.3, 37.4, 37.8, 37.9, 37.11, 37.12, 37.13, 37.14, 37.15.
   Report 37.5, 37.6, 37.10 as HELD and 37.7 as NOT RUN (base not cached), with the reason.
4. `part37/test_part37.py`: logic tests (records, length matching, frames, screen, folds, scoring rules) that run
   without torch, plus tiny-model tests (skip with a reason when torch is absent).
5. `part37/RUNBOOK.md`: staging into `C:\Users\alexa\neuron-probe-gpu\p37basin`, the `steps.txt` lines for the pilot
   and then the full run, the copy back, and the host analysis command. One model in memory at a time; free it before
   loading the next.

## Records (per kept question i, per reader R; "other" is the other reader)

Spans: `own_g`, `own_s1`, `own_s2` (R's answers to i), `other_g` (the other reader's greedy answer to i), `own_diffq`
(R's greedy answer to question i+1 mod n, among kept questions), `edit` (own_g with one word replaced: the word nearest
the middle with at least four letters, replaced by the first word of a fixed list of common words that has a similar
length and differs from it; log the pair), `null_smol` (SmolLM3's greedy answer to i), `null_pile` (a span of the same
length from `pile10k.jsonl`, chosen deterministically by question index, starting at a word boundary),
`null_shuffle` (own_g's words in a fixed seeded shuffle).

Length matching: L = the smaller of own_g and other_g in R's tokens. Every span is cut to its first L tokens of R's
tokenizer (cut by token ids, then decoded once; the record is built from that decoded text and re-tokenised, and a
test checks the re-tokenised span has L tokens or reports the mismatch count). If L < 16 the question is dropped for R.
A span shorter than L leaves that cell empty for that question (counted).

Frames (FOLLOWUP is the fixed user message "Thanks. Here is my next question: what is a good way to start the day?"):
- Assistant slot: user Q; assistant SPAN; user FOLLOWUP.
- User slot: user "Q\n\nHere is an answer to that:\n\nSPAN"; assistant "Noted."; user FOLLOWUP.
- Label frames (C3), headers H in {"Your earlier notes", "Notes from another assistant", "Notes", "Notes from the user"}
  with paraphrases {"Notes you wrote earlier", "Notes written by a different assistant", "Some notes", "Notes the
  user wrote"}: user slot: user "H:\n\nSPAN"; assistant "Noted."; user FOLLOWUP. Assistant slot: user "Please show me
  the notes."; assistant "H:\n\nSPAN"; user FOLLOWUP. Spans: own_g and other_g. So 4 x 2 x 2 x 2 = 32 records.
- Records per question: 2x2 (own_g and other_g, both slots) 4; own_s1 and own_s2 in the assistant slot 2; own_diffq
  in the assistant slot 1; edit in the assistant slot 1; the three nulls in the assistant slot 3; label frames 32.

Readout positions: the last token of SPAN, and the first token of FOLLOWUP's content. A test checks both indices on
the rendered template of each reader (decode the token at each index and compare it with the expected text).

Stated: run on the 2x2 (4), own_s1, own_diffq, edit, the three nulls, and the 32 label frames.

## Analysis (all probes: question-disjoint 5-fold GroupKFold by question id, fixed seed 37)

- Probe: per layer and position, standardise on the training fold, logistic regression (L2, C = 1.0, lbfgs, max 2,000
  iterations). Held-out scores pooled over folds give AUROC and balanced accuracy (threshold 0.5); also per-fold
  balanced accuracy. 95% CI by bootstrap over questions (1,000 resamples, seed 37).
- Controls: label-shuffled null, 200 permutations swapping each question's two labels at random, with the null taken
  as the maximum over layers for each permutation (so it is comparable with "the best layer"); the 95th percentile is
  the bar. Random directions: 100 seeded unit vectors per layer, AUROC of the raw projection, 95th percentile reported.
  A length-and-position-only probe (span tokens, record tokens, readout index).
- Surprisal controls for authorship: (a) mean log-prob regressed out of every feature on the training fold (least
  squares, applied to the held-out fold with the training coefficients); (b) subspace removal: an 8-component PLS
  between the training-fold features and the four log-prob summaries, its 8 weight directions projected out before
  the probe; (c) a matched subset: questions whose |own minus other mean log-prob| is below the median, with n and the
  mean difference reported. Also the held-out correlation of the authorship direction's projection with mean log-prob.
- Contrasts: role (C1) pooled and within each author; authorship (C2) pooled and within each slot (37.13 uses the
  within-slot numbers); own_g versus own_s (probe AUROC, and the shift on the authorship direction in units of that
  projection's standard deviation over own_g and other_g records, with and without mean regression, for 37.9); own_g
  versus own_diffq; the edit's and the nulls' mean projections.
- "Best layer" = the layer with the highest pooled held-out AUROC for that contrast; say this in the output.
- Authorship direction = the probe weight vector at the best layer trained on all questions (for projections only);
  role direction likewise. Cosine between them at the authorship best layer (37.8).
- C3 (37.3): on the authorship direction, label effect = mean projection with "Your earlier notes" minus with "Notes
  from another assistant" (both paraphrases, both slots, both authors); author effect = mean own minus other over
  the same frames; verdict on |label effect| < 0.5 x |author effect|.
- Stated: Spearman of authorship-yes p against the authorship projection over records (37.4, Qwen); null yes-rate =
  mean authorship-yes p over the three nulls, with the share above 0.5 (37.15); 37.11 on the memory and authorship
  yes/no p by header. Print the total two-token mass beside every stated number, and flag any cell whose median total
  mass is under 0.5 (a readout reading tokens the model would not say).

## Report

Write `part37/REPORT-build.md` with SUMMARY, FINDINGS, BLOCKERS. List anything in the design you could not implement
as written and what you did instead. Commit your work in the worktree if you can; if git refuses, say so and leave the
files.
