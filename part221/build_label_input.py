"""build_label_input.py -- rebuild the label-input file Part 219 fed its reader, for any run of the same table.

WHAT IT MEASURES: nothing; it builds. Input = a run jsonl (474 rows) and that run filtered by
`part101/read101.py filter` (129 read-set rows). Output = the read set, then one baseline row per read-set row
(id kept, unit "baseline", the run's own unpushed answer for that question, NO token_ids), the baseline rows
sorted by (id, question_index). 258 rows.

WHAT IT GOT WRONG FIRST (2026-11-22, Part 221): Part 219 built these inline and never saved the code. The first
rebuild copied token_ids onto baseline rows and kept them in read-set order; both differ from Part 219's file.
Check any change by rebuilding R0 and `cmp` against results/part219/label-input/R0.jsonl (byte-identical now).

Usage: python3 -I part221/build_label_input.py RUN.jsonl FILTERED.jsonl OUT.jsonl
Written by Journal Claude (Opus 5.5).
"""
import json, sys
run, filtered, out = sys.argv[1:4]
rows = [json.loads(l) for l in open(run, encoding="utf-8")]
unpushed = {r["question_index"]: r for r in rows if r["id"] == "unpushed"}
kept = [json.loads(l) for l in open(filtered, encoding="utf-8")]
with open(out, "w", encoding="utf-8") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in sorted(kept, key=lambda r: (r["id"], r["question_index"])):
        u = dict(unpushed[r["question_index"]])
        b = dict(r); b["unit"] = "baseline"; b["answer"] = u["answer"]; b.pop("token_ids", None)
        f.write(json.dumps(b, ensure_ascii=False) + "\n")
print(json.dumps({"run": run, "rows": 2 * len(kept)}))
