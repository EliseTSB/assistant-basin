# Cold check: Part 203's write-up

You are a fresh reader. Your job is to find what is wrong in `results/part203/README.md` before it is published. Read
only the files; trust nothing the README says about itself.

Repository root: /home/elisepc/green/worktrees/ab-part203 (git branch part203). Python with numpy and matplotlib:
/home/elisepc/neuron-probe/.venv/bin/python (run scripts with `-I -B` from the repository root).

Files: `PART203-DESIGN.md` (the registered design: Method, Predictions table, Reading rules, Limits, Changes, Amendment
1), `PREDICTIONS.md` (section "Part 203"), `results/part203/results.json` (the analysis output),
`results/part203/README.md` (the write-up under check), `results/part203/score.md` and `score.json`,
`part203/score203.py`, `part203/fig203.py`, `part203/analyze203.py`, `results/part203/fig203.png`.

Check, and report each finding with the README line, what is wrong, and the evidence:
1. Every number in the README: recompute it from `results.json` yourself (write your own small script in
   `results/part203/cold/`; do not reuse score203.py's code). Report any number that differs or that has no source.
2. Every verdict: apply the registered wording in PART203-DESIGN.md's table yourself. Is each held/failed right? Is the
   prediction wording and confidence copied verbatim? Weaver's rows: forecast and falsifier as the design states.
3. The title and first two paragraphs: is any claim stronger than the table supports? Is anything stated as a finding
   that was not registered without saying so?
4. Reading rules (design section "Reading rules"): does the README break any (e.g. comparing balanced accuracy with
   AUC, reading a V/D gap without both lower bounds above 1/6, comparing with Venn's 56.5% as if matched, reading E
   against V as a layer difference)?
5. Claims about earlier parts (Part 196) or about Venn: does the README say anything the design does not support?
6. Run `part203/score203.py --out results/part203/cold` and `part203/fig203.py --out results/part203/cold/fig.png`;
   do they run and agree with the README?
7. Every linked path in the README: does it exist in the repository?

Write `results/part203/cold/REPORT.md` with: SUMMARY (one paragraph), FINDINGS (numbered, most serious first, each with
evidence), and CHECKED (what you verified and found correct). Do not edit any file outside `results/part203/cold/`.
