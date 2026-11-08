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
- [ ] **Kernel check before any Arc launch.** `tools/kernel_reference.py` decodes eight fixed prompts on the card with
  the kernel and dtype the part will use, `tools/part_check.py kernel` compares that record with the stored CPU
  float32 reference (`results/kernel-reference/<model>-cpu-float32.json`), and the design's `Kernel check: <path>`
  line names the passing record; prelaunch refuses a design whose Method names the A770 or xpu without one. The
  loader's kernel is now explicit and recorded, eager by default. (Part 178: the default bf16 SDPA kernel left the
  CPU path past 64 keys, worst at batch 1.)
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
- 2026-10-30 (Part 123). A gate's threshold is set from the reader the part will use. A positive control copied from one reader's counts (14 of 18, Haiku) could not pass under another, and Part 92b had already said so.
- 2026-10-30 (Part 125). A cold review of worker code found arm 1 never pushed and arm 2 pushed on one token, with 16 tests green. Paired paths go through one runner with a call log, and a test reads the log.
- 2026-10-30 (Parts 124 to 126). Run `tools/part_check.py prelaunch` on a worker's design before review: two drafts had renamed the template's sections. A stub cannot show a keyword argument, a boolean mask or a stale job folder, so the two-unit pilot on the real card is part of the build.
- 2026-10-30 (Part 126). When a gate fails on its own code, fix it publicly as an amendment that says what had been seen, before any prediction is read.
- 2026-10-31 (Parts 136 and 141): `python -I` leaves the script's own folder off `sys.path`, so a sibling import fails on the Arc even when `--help` runs, if the import sits inside `main`. Two cold reviews and the tests passed Part 136 past this. Before queueing, run the real entry path, or list every import inside functions as well.
- 2026-10-31 (Part 139): a smoke run on real weights showed counts before the predictions were committed. Register first, or disclose it, as Part 139's design does.
- 2026-10-31 (Part 140): "8 answers per unit" in a design was built as 8 seeds per arm, which multiplied the run. Say per arm in the Size line.
- 2026-10-31 (Part 137): the build copied Part 134's second-reader rule and not this design's. A cold check compares the code's rule with the design's words.
- 2026-10-31 (Part 143): the front-page line is a claim too. I wrote that the two sides do not add, checking probability and not the log-odds scale, where they do. The README had passed its cold check; the sentence I added at publish had not. A front-page line carries numbers only, or goes through the reader.
- 2026-10-31 (Part 140): a gate computed in two places. The runner's Gate 2 was amended and the scorer's copy was not, and the scorer overwrote the good result file with a failure. One function, imported by both; the scorer reads the runner's result rather than recomputing it.
- 2026-10-31 (Part 136): count what a known-unit gate can reach before setting its bar. 12 of 24 asked for half of the only 24 documents holding those words.
- 2026-10-31 (Part 142): a Level Zero UR_RESULT_ERROR_UNKNOWN can stop an Arc job after minutes, at different ops, with the same batch that passed the pilot. Stages that append by key can be requeued. Arm a wait on every long stage: one failure sat unseen for 21 minutes.
- 2026-10-31 (Part 136): a pilot at 3 batch sizes costs an hour; the scan barely scaled, 1,010 s, 954 s and 760 s. Time one batch size first, and a second only if the bound is RAM or the gain is worth it.
- 2026-10-31 (Part 142): check a run's rows for finite values before scoring it. The model returned NaN with no exception, 17,152 rows of 18,672, and a correlation helper that only tested for zero variance called them defined. Raise on non-finite values where they are made, and gate on them in the scorer.
- 2026-10-31 (Part 142): do the bytes before writing a size into a public file. I wrote 10 GB for what is 1 GB, and a reader caught it.
- 2026-10-31 (Part 145): an index built for one question must not be used on another's word list. The scorer did, a verdict flipped, and I told Elise the wrong result even though the risers list on my screen showed " sorry" second. Compare a surprising score with the raw table before reporting it.
- 2026-10-31 (Part 148): a number in a design is looked up or left out. I typed a Part 112 run time that is on record nowhere; it was caught before launch.
- 2026-10-31 (Parts 145, 146): name the alternative before announcing a pattern, and run its control. Part 145's move toward Base was mostly a move toward Instruct with the question removed, which Part 146 showed within the hour. Run the control on the unit that gave the claim: a median row hid how weak single units were.
- 2026-10-31 (Part 149): compute the reachable range of a registered bar. A tenfold rise could not pass from a baseline median of 0.121, because ten times that is over 1. I set the ratio from one question. Log-odds or a probability difference would have been testable.
- 2026-10-31 (Part 149): Gate 5 needed no GPU and failed on one row after a 20-minute run. The rule was already here, from Part 122; I did not run it. A gate over prompts runs before launch.
- 2026-10-31 (Part 151b): where a neuron goes most negative says as much as where it peaks. Read bottom windows too, and write the reading down before checking it on other documents.
- 2026-10-31 (Part 152): a hit in the words after a fired token does not show that a unit read them. Report where hits fall beside any read-write match.
- 2026-10-31 (Part 153): a design's reading rule names how it is applied. "A hit on a unit whose pair also hits" allowed two readings, and one prediction's verdict depends on which.
- 2026-10-31 (Part 155): held-out phrasings do not protect against selection on dose. A unit picked at one dose needs a ladder before anything is said about its behaviour, and a peak at the selected dose is what selection would give; Part 156's finer ladder is the next step.
- 2026-10-31 (Part 154): a rule that compares against other units' moves must also state the direction itself. All three comparison units moved positive, so "more negative than theirs" passed neurons that moved the wrong way until the sign was added. Also, a readout on a name's first token reads whatever else starts with it: `Al` and `T` fell for every push, random included. Read a name's tokens before calling a column by the name.
- 2026-10-31 (Part 158): tokenize the words before writing a token rule. The registered prefixes banned ` Alibaba` and ` Tong` but not `Al`, `T` or `Q`, the tokens those names use after `**`, and 23 of 24 answers then wrote `**Alibaba` on those tokens; a gate caught it. A word-level ban also leaks through other spellings, so a ban needs a gate that checks the answers for the word (Part 157).
- 2026-10-31 (Part 154): the A770's silent NaN can hit a few rows in one run and none in a rerun, printing `!!!!`. Record the first non-finite call per row, and regenerate a batch that has one.
- 2026-10-31 (Part 159): read the answers before saying what an intervention did. A failed gate's score does not say what happened, and a cause sent from one was wrong. A distance is not small until it has a null: compare two conditions that should differ by nothing that matters.
- 2026-10-31 (Part 160): a picking rule needs its control to pass through the same rule. Random directions qualified as often as neurons, and laddering only the neurons made the contrast unfair. Do not cap a picked set in an order that correlates with anything measured; atlas order cut layers 23 and up. Say in the design what a flagged unit does to each verdict. Test each gate's passing case as well as its failing one.
- 2026-10-31 (Part 163): for every prediction write down what result would fail it. A ratio bar can sit above what the rates allow: 1.5 times a rate of 0.80 cannot be reached, so the prediction held by arithmetic and the reading rule built on it was void.
- 2026-10-31 (Part 161): compare arms generated in the same run. A greedy answer depends on batch shape, so units reused from an earlier part were regenerated; none crossed the bar, but only a rerun could show that.
- 2026-10-31 (Part 162): read the selected features before any number. Half of what separated two model families was punctuation, and a word-only reading reversed the side that mattered.
- 2026-10-31 (Parts 158 to 163): send a number to a person only after its cold check. A hedged guess about why also needs a test before it is sent: "may be selection" was refuted two hours later.
- 2026-10-31 (Part 164): a gate and the outcome it protects use one matcher. Gate 2 passed on a substring while the outcome rule scored the plain model 0 of 16, so the Qwen rate had been measuring a form the unpushed model did not use.
- 2026-10-31 (Part 164): an analysis script computes what it prints. A typed figure inside a print in Part 163's script read like a result and survived a cold check.
- 2026-10-31 (Part 164): check triage's order before trusting a label. Part 86's triage returns kept before its loop check, so repetition loops pass as kept.
- 2026-11-01 (Part 165): send the design to the council before queueing, not after: the queue started it minutes before two reviews landed. Check each sentence about how a push is applied against the hook itself; the design said prompt and answer, and the hook skips the prompt pass.
- 2026-11-01 (Part 165): equal intended write norm is not a delivered dose. The hook adds to a bf16 activation, so record the realized write at a shared context.
- 2026-11-01 (Part 166): three launches, two lost to things a read would have caught: a row field missing on one arm and tensors too large for Windows. Read the runner's main path line by line before queueing; a passing unit test says nothing about a path it does not enter.
- 2026-11-01 (Part 166): when a slot is resampled, it must not donate to itself. A point estimate outside its own interval was the sign.
- 2026-11-01 (Part 167): a gate about a deterministic baseline can be answered before the run. Gate 2's plain answers were on disk from Part 164, and a rule added since (Part 165's loop rule) had never been run over them. Run every gate's rule over stored answers at design time, and put a baseline gate first in the runner, so a certain failure costs a minute and not 47.
- 2026-11-01 (Part 169): a brief's sentence can become a defect. Asking that "a resampled slot never donates to itself" carried Part 166's donor lesson into a part with no donors, and the worker built a bootstrap with 144 draws per layer, so every interval was too narrow. State a lesson with the condition it came from, and compare a worker's runner with the part it was copied from line by line: reading Part 169 against Part 168 found 17 changes, among them a vocabulary-wide top-k on the device, a 192-row forward pass, no pilot and no resume, with 9 tests green.
- 2026-11-01 (Part 169): read the seven-layer accounting before a mechanism. Part 168's own cut and mass rise both sat in the top own-mass quartile, and about half of all loops were barely touched by either ban, which no pooled rate showed. Cross the headline with the size of the list's effect on the plain path before designing the follow-up.
- 2026-11-01 (Part 168 publish): a table-writing script can pass its own check. A verdict patch left an empty cell in 11 rows and the publish check, which reads the last cell, passed. Read a filled table once.
- 2026-11-01 (Part 170): an instruction in a brief can flip a bar's meaning. Asking a worker to "take an absolute value where the design means size" turned 170.4, 170.5, 170.6 and both of Weaver's bars into magnitudes, so a larger list that lowered loops would have passed "at least 0.05 above". Read every bar for its sign, and write one test where the effect goes the wrong way.
- 2026-11-01 (Part 170): an estimate field should hold the quantity the bar tests. If it holds a margin, say so beside it. A rule built for one kind of answer does not carry to another: Part 165's broken rule calls any answer under four words broken, which is right for identity answers and wrong for "Paris."
- 2026-11-01 (Part 169): when a part replicates an earlier one, read every difference in the recipe before writing "the same". Added-vocabulary exclusions and the baseline loop rate both differed from Part 168.
- 2026-11-01 (Part 170): a no-push arm filtered on the plain path not looping, set beside a push arm with no such filter, makes a rise by selection. On identity prompts the +3.3 points at b50 was -4.8 over all prompts. Report every comparison on all rows too, and write the filter's direction into the design before the run. Also, a bar written as a ratio passes any negative rise.
- 2026-11-01 (Part 170, Alden): a reading rule fixed in advance can still be logically too strong. 170.2's rule read a cross-population ratio as a part of every ban result. Keep the verdict, and correct the reading in the open. Before registering a reading, ask whether the two numbers in a ratio come from the same population.
- 2026-11-01 (Part 171): run a design's premise before building on it. Every review of the design and the build took for granted that the model would adopt a maker role, and the first look at the answers showed it did not; a two-minute run of the target on the fit prompts would have shown that before any of them. Gate 5 caught it, as a gate should, after the build. Also: a default argument that restates a registered list (rotate's targets) goes stale when the list changes; import the list.
- 2026-11-01 (Part 172, Alden's read): a control's name is not its construction. `r100` was written up as random directions in a README and a cold check, and all 12 controls are built from the Google fit. Read a control's builder before writing the word random about it.
- 2026-11-01 (Part 173 build, Haiku's read): a bar on a cosine needs the chance size for its dimension. 0.02 is about one standard deviation in 2,560 dimensions, so a correct build would have stopped at the gate. Part 171's own numbers showed it.
- 2026-11-01 (Part 173 fold): a worker fixing a finding can follow the finding's wording over the design. A fold made invented names ineligible, against the design, with its tests agreeing. Read a fold's report against the design, not against its tests.
- 2026-11-02 (Part 174, Weaver's read): unobserved is null, never zero. The scorer stored a Google count of 0 and D = 0 for 32 flat directions with no clean answers, so an absence read as a measurement. A scorer emits null, every reader skips nulls, and nothing coerces them.
- 2026-11-02 (Part 174, Haiku's read): a design names what happens to a missing value. "Across the 60 draws" had no rule for a draw with no D, and the scorer's stop gave the verdict for a smaller reason than the real one, since every flat draw had a sign with nothing read.
- 2026-11-02 (Part 175): a dose chosen on few draws does not transfer across a steep cliff. Four draws gave a median of 6.5 broken at 0.30 and twenty gave 29.5. Read the slope beside the rule, and keep census doses below the chosen one, which is what saved this part.
- 2026-11-02 (Part 175, Haiku's read): `selected` needs a written rule and a plain label. A best-of-three on the census's own draws is an in-sample maximum, whatever the dose rule was chosen on.
- 2026-11-02 (Part 176, Weaver's read): a catch-all state can hide a transition. `no claim` was coherent answers at 0.225 and fragments at 0.30. Split a state before reading a ladder through it.
- 2026-11-02 (Part 176, Alden's read): build the table that passes every check while the reading is false. Forty directions with ten Google claims each gave a rank test of -1 with every check passing, and the design had called both measures free of truncation.
- 2026-11-02 (Part 176, the real run): test a verdict rule on partly labelled bounds. `not read` means an unknown could change the verdict, so the scorer has to try every admissible count. It said `not read` where the rule gave `failed`, past 28 tests and a cold read of the build.
- 2026-11-02 (Part 176, Haiku's read): filter scored sets by a list, not a name prefix. The replay arms shared the prefix, added 88 rows at one rung, and the report said `not read`. One unread rung of five is a defect until it is explained.
- 2026-11-02 (Part 176, Haiku's read): a percentage in a generated sentence is computed from the two numbers printed beside it. "893 of those (58.4%)" printed a share of another total.
- 2026-11-02 (Part 176, Haiku's read): before registering a falsifier on a gap to an interpolated crossing, work out what the rung spacing gives alone. At a step of 0.025, a gap under 0.025 is nearly the event that the top rung sits at the lower end of the crossing's bracket.
- 2026-11-02 (Part 177, Alden's read): build the false-reading table before registering. Marginal agreement does not validate per-trace labels, a precision gate does not make every string hit an attained answer, and "hit and no onset" can pass with no comparison made.
- 2026-11-02 (Part 177, Weaver's read): state the instrument's resolution and the denominator before choosing a bar.
- 2026-11-02 (Part 177, Muse's search): recover per-case evidence before settling for an aggregate. Finish reason was not stored, and retokenising the text gave it back, 187 and 13 with no overlap.
- 2026-11-02 (Part 177): read the verdict code against the prediction's own wording once real numbers exist. The scorer had added a failure rule and a shrunken denominator that no test questioned, because the tests pinned what the code did.
- 2026-11-02 (Part 177): a smoke test runs the real command line. 52 tests and 36 mutants passed and the script could not import its helper under `python -I`.
- 2026-11-02 (Part 177, Haiku's read): quote a source's sentence whole and check which experiment it belongs to before registering. The design's quotation stopped before the clause that limits it and came from another model's result. A sentence that says which trace counts names a numerator, not a denominator.
- 2026-11-02 (Part 178): on the Arc, Qwen3.5 greedy tokens at batch 1 differed from the same rows at batches 2 to 16, which agreed with each other and, on the 0.8B, with a CPU float32 reference (`part178/diag178d.py`, `diag178e.py`). Generate at batch 2 or more, give a lone last row a discarded companion, and record the device and batch sizes in the run record. Whether batch-1 forward passes are also off is not yet measured; an audit of earlier parts is in progress.
- 2026-11-02 (Part 178): the first pass in a fresh process ran at about half speed on the Arc (250 against 432 to 471 tokens a second), so time an order-swapped pair before crediting a lever. Releasing cached blocks every 64 steps held reserved memory at 3.8 GB against 5.2 with no measured speed cost. Batch 32 on Qwen3.5-4B at a 2,048-token cap reserved 17.5 GB and spilled into Windows memory.
- 2026-11-02 (Part 178): a code cold read before the first launch found a crash on Qwen3.5's hybrid cache that 24 passing tests on a dense model could not see. A runner's end-to-end test uses the target architecture.
- 2026-11-04 (Part 185): a "no detectable difference" prediction must be able to fail at the n it will actually see. 185.5 asked a paired sign test for p < 0.05, and with 3 to 5 discordant pairs per dose that is unreachable (5 pairs give at best 0.0625), so it "held" with no information; Part 163 had the same shape. Before registering, compute the smallest p the expected counts allow (the original run's discordant counts were on disk) and pool or drop the test if it cannot fail.
- 2026-11-04 (Part 185): `tools/launch_arc.py --copy a --copy b` copied only `b` until a2d10e6; a step then ran the job folder's stale or missing copies. Compare job-folder files with the repo by sha256 when provenance matters.
- 2026-11-04 (Part 182): a negative test that passes has not shown it reached its intended failure. Part 182's
  stage-2 failure test was green, and I accepted it, because its injected comparison took `_pad_id`
  while the caller passed `pad_id=`, so both cases raised TypeError before either branch, and its fixture was already
  invalid before the fault. Before accepting a negative: start from a pre-fault fixture shown to pass, assert the
  intended cause (message, branch, artifact), and check that disabling the guarded transition breaks the negative.
- 2026-11-08 (tools/register_public.py, Part 196). The merge-back restored a working PREDICTIONS.md that was a superset of the pushed copy but not of public main, which had gained Part 194's section after the branch was cut, so the section was dropped and the next push published the loss for about five minutes (7bab68d; restored 0441905). The tool now puts back every section main has and the working copy lacks, and stops on a missing front-page entry. Before any push of results, count the `## Part` headings against `git show origin/main:PREDICTIONS.md`.
