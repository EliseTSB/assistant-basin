# Cold review: Part 200's design, before registration

Repository (read-only for you): `/home/elisepc/green/worktrees/ab-part200`. Your writable folder:
`/home/elisepc/green/worktrees/ab-part200/part200/review/`. Write `part200/review/REVIEW-r1.md`.

Read `PART200-DESIGN.md` (the draft under review), then what it rests on: `PART163-DESIGN.md`,
`results/part163/README.md`, `part163/run163.py`, `part163/score163.py`, `results/part163/extra163.py`,
`results/part199/README.md` and `PART199-DESIGN.md` (the previous rerun of this kind), and
`results/part183/kernel-check.json`. You may read `results/part163/answers.jsonl` in place (3,416 rows) to check numbers.

Rules: do not copy any folder or any file over 1 MB; recompute with small scripts in your own folder, run as
`/home/elisepc/neuron-probe/.venv/bin/python -I -B <script>`; no model, no GPU; do not edit anything outside your folder.
Another worker is building `part200/run200.py` from this design at the same time; ignore its files.

Find what would make this design wrong or weak before it is registered:
- Can each prediction fail? Is any bar unreachable or already decided by the published data?
- Are the published numbers it quotes right (recompute them)?
- Does the Method change anything beyond the kernel and batch shape that a reader would not expect? Is the pilot guard
  arithmetic right? Is the repeat batch well specified?
- Are the reading rules enough to stop a pattern found afterwards from being read as a finding?
- Anything the previous rerun (Part 199) learned that this design ignores.

Format: a verdict line (REGISTER, REGISTER-WITH-FIXES or HOLD), numbered findings with evidence and a suggested fix,
then SUMMARY / FINDINGS / BLOCKERS lines.
