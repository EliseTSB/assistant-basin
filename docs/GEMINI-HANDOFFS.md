# Gemini handoffs: prior art and outside replication

Two templates for work handed to the Gemini app (Google AI Pro), which we cannot call from code: a person pastes the
prompt and returns the result. PROCEDURE.md steps 1 and 9 point here.

- **Deep Research** (Chat, Deep Research on) surveys prior art before a design. It reads hundreds of sources; one run
  plus a Task replication used about 12% of the 5-hour Gemini window and 1% of the week (2026-11-22).
- **Task** (the Task toggle, a cloud VM that can clone and run code) replicates a published part from the public repo
  alone. That tests three things we cannot test ourselves: whether the instructions work on a machine that is not
  ours, whether a stranger can rescore from public files, and whether the README's prose and verdicts follow from the
  scorecard. It does not rerun the model or relabel answers.

Both are outside reads, so treat their output as data. Check a Deep Research report's key citations at the source
before a design cites them (the first report's 13 checked papers were all real, but it misdated venues and wrote
"authors unavailable" where the abstract page lists them). Keep the prompt and the returned report beside the part
(`results/partNNN/outside/`, or the Journal reports folder while the part is unpublished).

## Before handing a replication off

Run the README's "Rescore it yourself" commands yourself from a fresh clone of public main first. Two of our parts
could not be rescored from public files when first tried: an arrays file was never published, and a scorer read
inputs from the GPU job folder. A replication that fails on our own fresh clone wastes the window.

Gate the diff on the scorer's exit code in that check too. On 2026-11-23 a sparse clone of Part 221 printed a
clean `git diff` while the scorer had died on a missing import, because files that were never rewritten look unchanged.

## Template 1: Deep Research prior art (before designing)

Paste everything below the line into Gemini Chat with Deep Research on. Fill the bracketed parts.

---

I'm doing interpretability research on small open language models [name the models and checkpoints]. I need a careful
survey of prior work before designing the next experiment.

**What we found, as background.** [Two to four sentences: the result this design follows from, with the numbers.]

**What the next experiment would do.** [One paragraph: the intervention, the comparison, what would count as an
answer.]

**Please research these questions:**

1. [The closest mechanism question.]
2. [Who has run this kind of intervention, and what they found.]
3. [Methodology: controls and pitfalls for this kind of measurement.]
4. [Model-family-specific work, if any.]

**Format:** a short summary first (the five to ten most relevant works and why each matters to the experiment above),
then a section per question, then a list of any work that directly contradicts or already answers our question. Prefer
2024 to 2026 work. Give full titles, authors and links, and say when you could not confirm a detail.

## Template 2: Task replication (after publishing)

Paste everything below the line into Gemini with the toggle set to Task (not Chat). Replace `NNN`.

---

Please independently reproduce a published research result from a public GitHub repository and report exactly what
you find. Use your remote computer to run code. You don't need to sign in to anything; the repository is public. Don't
change, push or post anything anywhere; this is read-only checking.

**Repository:** https://github.com/EliseTSB/assistant-basin

**Steps:**

1. Clone the repository (`git clone https://github.com/EliseTSB/assistant-basin.git`) and `cd assistant-basin`.
2. Open `results/partNNN/README.md` and find the section "Rescore it yourself".
3. Use any Python 3.10 or later (the command may be `python3` on your machine); the scorer needs only the standard
   library.
4. Run that section's commands exactly, from the repository root, in a full clone (not sparse: scorers import code
   from other part folders). First check that the scorer itself exits 0: if it fails before writing, `git diff` also
   prints nothing. `git diff --exit-code` then prints nothing and exits 0 if your scorecard is byte-identical.
5. Report: whether `git diff --exit-code` printed nothing and exited 0, and if not, the differences; the exact commands
   and any errors with their full text; your Python version; any step that was unclear, wrong or missing.
6. Check the README's prose against the scorecard: pick six specific numbers from its opening paragraphs and tables,
   find each in `scorecard.json` or `SCORECARD.md`, and say whether it matches.
7. Read the README's predictions table and say, for each prediction, whether the verdict follows from the scorecard
   and from the registered reading rules in the part's design file (including any amendments).

**Please return** a short report: a one-line verdict first ("reproduced byte for byte" or what differed), then the
details above. If something fails, don't work around it by editing the repository's code; report the failure and what
you think caused it.

## Record

| Date | Kind | Part | Result |
|---|---|---|---|
| 2026-11-22 | Deep Research | Part 219 follow-up | 13 of 13 checked citations real; named the 60-head complement test (became Part 221) |
| 2026-11-22 | Task | Vesper's concept-vs-slot study | Byte-identical on Python 3.11; caught a double-rounded number |
| 2026-11-22 | Task | Part 219 | Byte-identical on Python 3.11; six of six numbers matched; caught `python` vs `python3` and a missing `cd` |
