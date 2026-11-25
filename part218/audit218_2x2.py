#!/usr/bin/env python3
"""audit218_2x2.py — the reader audit's 2x2 and its disagreements (Vesper's ask, council vesper-part218 #21).

  python3 -I part218/audit218_2x2.py --run-dir results218-v4  ->  <run-dir>/audit/disagreements.md
Reads the last label per audit_id from labels-ds.jsonl and labels-luna.jsonl (errored rows skipped) and the answers
from audit218.jsonl. Description only: the audit rule's verdict is unchanged by this.
"""
import argparse
import collections
import json
from pathlib import Path


def load(p):
    d = {}
    for line in Path(p).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("genuine") is not None:
                d[r["audit_id"]] = r
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    a = ap.parse_args()
    au = a.run_dir / "audit"
    ds, lu = load(au / "labels-ds.jsonl"), load(au / "labels-luna.jsonl")
    items = {json.loads(l)["audit_id"]: json.loads(l) for l in (a.run_dir / "audit218.jsonl").read_text().splitlines() if l.strip()}
    ids = sorted(set(ds) & set(lu))
    lab = lambda g: "genuine" if g else "not genuine"
    out = ["# Part 218 reader audit: the 2x2 and the disagreements", "",
           f"Answers labelled by both readers: {len(ids)}.", "",
           "| DeepSeek \\ Luna | genuine | not genuine |", "| --- | --- | --- |"]
    t = collections.Counter((ds[i]["genuine"], lu[i]["genuine"]) for i in ids)
    for g in (True, False):
        out.append(f"| {lab(g)} | {t[(g, True)]} | {t[(g, False)]} |")
    out += ["", "| Model | both genuine | both not | DeepSeek not, Luna genuine | DeepSeek genuine, Luna not |",
            "| --- | --- | --- | --- | --- |"]
    for m in ("SFT", "DPO", "Instruct"):
        mt = collections.Counter((ds[i]["genuine"], lu[i]["genuine"]) for i in ids if ds[i]["model"] == m)
        out.append(f"| {m} | {mt[(True, True)]} | {mt[(False, False)]} | {mt[(False, True)]} | {mt[(True, False)]} |")
    out += ["", "## Every disagreement", ""]
    for i in ids:
        if ds[i]["genuine"] != lu[i]["genuine"]:
            out.append(f"- `{i}`: DeepSeek {lab(ds[i]['genuine'])}, Luna {lab(lu[i]['genuine'])}. "
                       f"\"{items[i]['answer'][:200].replace(chr(10), ' ')}\"")
    (au / "disagreements.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
