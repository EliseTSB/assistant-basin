# Procedure for every part

Every part from Part 115 on goes through this list. Started 2026-10-29 at Elise's request: "We should build a checklist
or procedure or something for each test where you optimize it and follow any other useful advice we find broadly
applicable over this work".

Three tools carry the parts a script can check:

- `templates/PART-DESIGN.md` and `templates/RESULTS-README.md`: the design and the write-up start from these.
- `tools/part_check.py prelaunch NN`: the design is complete, its size is multiplied out, a run over about 5 minutes
  has a measured optimization pass, and the registration is on public main. `publish NN`: every registered prediction
  has a verdict, the cold check is recorded, and every cited path exists and is committed.
- `tools/launch_arc.py NN`: runs the prelaunch check, then starts the job on the A770 or in a memory-capped CPU unit.
  A failed check stops the launch unless an override reason is given, and the reason is recorded.

Each item below names the part or incident that taught it. `STANDARD-CONTROLS.md` and `TOOLKIT.md` hold the controls
and reusable checks in detail; this list says when to use them. When a part teaches something that applies beyond it,
add it to the lessons log at the end, then move it into the list.

## 1. Before designing

- [ ] **Look in our own results first.** The answer may already be on disk. (2026-10-29: how concept and slot neurons
  differ through the Jacobian lens was already in Part 97's labels. Part 108 started from printing answers already on
  disk.)
- [ ] **Search the prior art** and record the closest work in `PRIOR-ART.md` and the design. (Part 29: a paper five
  days old had measured most of it.)
- [ ] **Prefer causal tests across training stages** to activation correlations when the question allows it. OLMo
  publishes every stage.

## 2. Design

Controls

- [ ] **Apply `STANDARD-CONTROLS.md`**: both directions, poles, a do-nothing hook that matches the plain model bit for
  bit, matched random controls, the answer-polarity split, broken answers counted, a dose ladder when a push saturates.
  Leave one out only with a written reason ("Not used: ...").
- [ ] **Match the control on everything except the property under test.** Random directions read through a lens give
  word salad, so "never matches a persona" measured coherence; the fair control was another neuron's words (Part 97).
  A random same-layer unit can be nearly silent, so zeroing needs an activity-matched null (Part 67, toolkit I). A
  scramble matched on weight distance changed far fewer answers, so match the size of a change by its effect on
  behaviour (Part 108).
- [ ] **Use several controls, not one.** Part 78's single size-matched control tripled Google; ten more showed seven of
  ten go the other way (Part 78b).
- [ ] **Name a gate that must pass before the main result is read, and how it could fail.** (Part 66's positive control
  failed and stopped the reading until the rig was checked. Part 75's clamp to the prompt average turned out not to be
  neutral.)
- [ ] **Select on one set and test on another.** Held-out prompts for anything chosen by a search (Part 66). Units
  chosen at one checkpoint make that checkpoint look responsible, so compare the stages on both sides (Part 108).
- [ ] **Ask off-target questions** beside the target ones (toolkit C), and count broken answers (Part 38).
- [ ] **Scope every claim** by dose, model, layer and prompt set. (Parts 106 and 107: the pushed sentence held flat at
  0.8 times the push and not at 0.4.)
- [ ] **Readouts.** Forced-choice readouts print each option's probability mass (Part 34, toolkit J). When tokens are
  registered, compare tokens, not decoded text (Part 86's replay). One question cannot sort concept from slot neurons;
  the six-question battery can (Parts 92 and 112).
- [ ] **Readers.** Blind to unit, dose and arm; checked on known examples before use; a second reader on at least 10%,
  with agreement reported; a written cut-off rule; error rows handled. (Part 59A's reader had a 32-token limit and
  returned 62% empty labels; toolkit G.)
- [ ] **Match the settings across arms** and write them beside the predictions: model revision, dtype, template,
  thinking, temperature, reader effort. (A 2026-10-19 comparison ran one arm at low effort and the others at extra
  high.)

Size

- [ ] **Size from the smallest effect worth knowing**, then multiply the run out to one number in the Size section. Over
  about 1,000 generations or one GPU-hour, run a pilot first, with its stopping rule written down. (Part 38 was briefed
  at about 19,000 generations for an effect that 64 answers against 64 settles.)
- [ ] **Generate only what is read.** (Part 29 read 250 words and generated up to 4,000 tokens.)

Predictions

- [ ] Numbered, each with a threshold, a confidence and an author, and a rule for withdrawing the headline reading.
  Amendments are dated, made before the affected data is read, and keep the old text.

## 3. Register

- [ ] **Push the design, the code and the predictions to public main before any data**, with
  `tools/register_public.py`. It publishes the registration without results that have not been checked.
- [ ] **Times come from `date` or a message id, never typed.** (Part 113's design carried a typed 18:00 for 17:50.)

## 4. Optimize: any run over about 5 minutes

Elise, 2026-10-22: "I want you to do an optimization pass like this on any run longer than a few minutes, we keep
leaving easy optimizations on the table." The design's Optimization section records the bound resource, one throughput
measured on real rows, and each lever below that applies, taken or declined with the reason.

- [ ] **Is the bound resource busy?** Measure it on a short sample before the full run. (The atlas collector did CPU
  numpy work while the A770 sat 5 to 10% busy; reading its second pass found a tenfold fix.)
- [ ] **Batch size**: time two or three sizes on real rows; never inherit one. (Part 86: 1.82, 2.99, 3.69 and 4.25
  answers/s at batch 16, 32, 48 and 64, still rising at 64.)
- [ ] **max_new_tokens**: the fewest the reader needs; check on the pilot where the labels stop changing.
- [ ] **Pool the random controls** when the design allows (a shared pool per layer instead of one per unit).
- [ ] **Teacher-forced scoring** instead of generation when the readout is a probability.
- [ ] **Generation loop**: rows that finish early leave the batch; a prompt shared by every row is prefilled once; hook
  code is vectorized, with no Python loop per row per token; the model's fused kernels run, not a slow fallback (read
  the warnings).
- [ ] **Vocabulary-size work (top-k, sorting) on the CPU.** A top-k over the vocabulary on the A770 had not finished
  after 35 minutes; on the CPU the whole job took 73 s.
- [ ] **A770 copy engine off** (the job runner's default; a direct launch sets it itself). One stalled pass went from
  718 s to 73 s.
- [ ] **Host memory**: Windows needs room for the model while it loads. Free the WSL page cache, read the free memory,
  and check that the load path does not stage the whole model in host memory first. (2026-10-29: Part 114 started with
  7.4 GB free and the job runner's memory guard stopped it 29 s in; Part 112 had started with 11.6 GB.) One A770 job
  at a time.
- [ ] **Cache and resume**: derived models and activations are built once and reused (Part 108); rows are appended and
  rows already written are skipped, so a stopped run resumes.
- [ ] **Labelling**: a cheaper reader where its agreement with the reference is measured (Part 55 measured Haiku
  against Opus on 1,792 labels before Haiku was used); the batch API; one live request before any paid batch (Part 63's
  first run failed all 40 batches on a parameter the model rejects); an estimate of the run's share of any shared quota.
- [ ] **Device**: the A770 first; the CPU for runs of a few minutes or vocabulary-size math; rented GPUs only when the
  gain is worth the cost.

## 5. Build and smoke

- [ ] **Reuse shared code** before writing a runner (`TOOLKIT.md`, `part86/common86.py`, `part92/common92.py`).
- [ ] **Read the prompts, targets and forced openings by eye** before running. (Part 106: two target sentences had been
  cut to "Hello!" and "I'm Dr.")
- [ ] **Run the plain model on every fixed-target probe and print its top tokens before registering.** A target that is
  not the plain model's first choice, or is not one token, measures something else. (Part 115: the chat model put " **"
  first on three of four ordinary openings, and " 4" was two tokens, so the off-target check could not do its job.)
- [ ] **Tests on fixtures, plus one deliberate break that must turn them red.** Break a copy, never the tracked file.
  (During Part 108 a `git checkout` used to undo a break erased real edits.)
- [ ] **Smoke on the real model and device before launch.** A sandboxed builder may have no torch, so its tests may
  never have run. (2026-10-26: a worker-built runner zeroed nothing, and its fixture expected every effect to be zero.)
- [ ] **Known traps**: Qwen3.5's chat template strips trailing whitespace from an assistant message;
  `apply_chat_template(..., add_generation_prompt=True)` returns an encoding in transformers 5.20; OLMo's unknown token
  is its end token; the A770 has no float64 kernels; Windows-side scripts open files with `encoding="utf-8"`;
  argparse needs `--doses=-0.8,-1.5` for a negative first value.

## 6. Pilot and run

- [ ] **Pilot gate written first**; pilot answers read by eye; stop when the answer is clear, scale only when it is
  borderline.
- [ ] **Every run in the background**, through `tools/launch_arc.py` (A770, or `--cpu` for a memory-capped unit). A
  chain script calls the launcher too. (2026-10-04: an uncapped test grew to 11 GB and took down other services on the machine.)
- [ ] **A wait matches only new log lines.** `progress.log` appends across runs; the launcher records its line count
  before each launch.

## 7. Score and read

- [ ] **Scorer tested** on fixtures and a deliberate break. Every number is printed by a script, never typed.
- [ ] **Verdicts exactly as registered**, with confidence labels. A pattern found afterwards is labelled exploratory.
- [ ] **Read a sample of answers in every cell** before trusting a reader or a regex. (Part 16: a regex counted a drop
  to 20%; ten answers read by hand had lost the opening and broken the sentence, which left the self-description
  untested.)

## 8. Write up

- [ ] **Claim and picture first**; then each question a skeptical reader would ask, answered by its test and a figure;
  every number linked to its script and file; the raw answers browsable; failures stay on the page.
- [ ] **Figures drawn by code** from the result files.
- [ ] **Before the cold check, look for the faults cold checks have found most often** (25 folds, Parts 18 to 113): a
  title or first sentence stronger than the table; a missing scope (dose, order, model); counts typed instead of
  computed; terms or regexes left undefined; registered wording paraphrased or confidence labels dropped; misquotes; a
  claim about an arm that was not read; a pattern found afterwards stated as a finding; an earlier part cited without
  checking what it measured.
- [ ] **Cold check**: a fresh reader with only the files; fold every finding; record it on the README's "Cold check:"
  line.
- [ ] **`tools/part_check.py publish NN` passes.**
- [ ] **Report numbers and the registered verdicts first**, the reading after the cold check.

## 9. Publish and close

- [ ] Fast-forward main. Commits use the noreply address and the co-author line.
- [ ] Update the front page's counts by script, and `NEURON-RECORDS.md` and `PRIOR-ART.md` where the part touches them.
- [ ] Add anything that applies beyond this part to the lessons log.

## Lessons log

New lessons land here first, dated and with the part that taught them. Promote them into the list when they apply
beyond one part.

- 2026-10-29: the list above, gathered from the memory of earlier parts and the 25 cold-check folds.
- 2026-10-29: Part 114's first launch ran Windows out of memory during the model load. The shared loader builds the
  model in host memory and then moves it to the A770; loading straight to the device may lower the peak. To be
  measured with the generation-loop speedups (batch 96 to 128, about 60 new tokens, pooled controls, finished rows
  leaving the batch, the shared prompt prefilled once), timed against Part 112's pilot rows and checked against its
  stored answers.
- 2026-10-29: where Part 112's generation time went. Only 76 of 5,580 answers (1%) reached the 120-token cap and the
  median answer was 67 tokens, but 42 of 88 batches of 64 ran to the cap because one row did: a batch costs as much as
  its longest row. So rows leaving the batch when they finish (and new rows joining) is the larger lever, and a
  60-token cap would cut 68% of the answers, so it needs the reader's agreement on cut answers measured first. The A770
  was 94% busy on average, so the run was bound by the GPU's work, not by waiting.
- 2026-10-29: Part 115, the first part through this list, passed every prelaunch check and still carried a broken
  off-target probe: its baseline was run but not read before registration. The item above came from it. Using the publish check on it also found
  that "Cold check: pending" counted as a cold check; it no longer does.
- 2026-10-29 (Part 115b): loading straight to the A770 lowers the host-memory peak enough to run beside a game, once
  transformers' allocator warm-up is skipped (it asks for one 7.8 GiB block, over the Arc's single-allocation limit).
  The CPU-staged loader was stopped by the 2 GB guard starting from 9.5 GB free; the direct load ran from 9.6 GB.
  `load_direct` in part115b/howmany115b.py (`--direct-load`).
- 2026-10-29 (Part 115b): copy every probe pass's output into the repo before registering. The registration cited a
  third pass whose output stayed in the job folder until the cold check asked for it.
- 2026-10-29 (Part 112): compute an amendment's description of the data, as with any number. Part 112's Amendment 1
  named the wrong questions for its unlabelled answers.
- 2026-10-29 (tools/register_public.py): registering an amendment to an older part republished every later part's
  prediction section. A section now stops at the next part, and a part already public leaves PREDICTIONS.md alone.
- 2026-10-29 (Part 116's cold check): flips come from batched forward passes and first words from a separate generation
  batch, and a few rows disagree. Say which a count comes from.
- 2026-10-29 (publishing 115b and 116): `count_predictions.py` exits 1 with 126 unclassified rows, every table from Part
  75 on that puts Result before Verdict, and the front page no longer carries a count line, so section 9's count step
  has nothing to update. Fix the script (read the Verdict column when a table has one) before a count returns to the
  front page. The publish check also reads a backticked formula with a slash, such as `log(p/(1-p))`, as a cited path.
- 2026-10-30 (Part 119): `launch_arc.py --copy` takes one flag with several paths; repeating the flag keeps only the last, and the job dies with "can't open file". A gate that compares with an earlier run must match its padding and batch shape, or it measures rounding (32 of 39, all at late close calls).
- 2026-10-29 (Part 117b beside a game): the direct load needs about 8 GB of Windows headroom. 117b's load fell under
  the 2 GB guard in 15 s from 9.1 GB free; Parts 115b and 117 had succeeded from 9.6 to 10.3 GB. From 10.9 GB it loaded
  and passed the zero-hook check, then hit the guard two minutes into generation at batch 64, because the game held
  VRAM and the A770 spilled into system memory. Wait for the game's process to exit, not just for free memory.
- 2026-10-30 (Parts 119b and 119c): both builds passed their worker's own tests with a top-level run that could not start,
  a missing import in each, because no test called `run()`. A build brief now asks for a test that drives the real run end
  to end on a tiny model, and one that imports from a tree holding only the `--copy` list. A DeepSeek code review caught both.
- 2026-10-30 (Part 119's cold check): a reader's count can tell a stronger story than a text rule. Triage called 48% of pair
  answers broken while the reader marked 85% of a sample incoherent. A breaking claim now carries both, and a design's
  withhold rule says what happens when an agreement statistic is undefined.
- 2026-10-30, Part 122. A gate that can be computed without the GPU is computed before the launch. Gate 1's token count needed only the tokenizer, and the writer had checked words, so a registered gate failed on a finished run. A writer validates with the same instrument the gate uses.
- 2026-10-30, Part 122. Every statistic in a forward hook moves to the CPU once per hook. An `.item()` per value took 14 s a batch and hung the A770.
- 2026-10-30, Part 123. A do-nothing check compares runs at one batch shape. On the A770 a bf16 greedy answer can depend on batch shape, and 5 of 18 zero rows differed from plain until the shapes matched, when 60 of 60 agreed.
- 2026-10-30, Parts 118, 121, 122. A message to Elise before the cold check carries counts and no verdict word. "Clean no", "concept beats slot" and "a narrow window" each came back weaker, and the numbers beside them were right.
- 2026-10-30, publishing. `part_check publish` gates the push with `&&`, in one chain. A `;` pushed Part 120 past a failing check; it failed on a false alarm that time.
