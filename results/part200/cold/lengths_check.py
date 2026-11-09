#!/usr/bin/env python3
"""Length-stratified checks: broken and mid-phrase by cap status, token_count vs re-tokenized length, EOS hypothesis."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/elisepc/green/worktrees/ab-part200")
NEW = ROOT / "results" / "part200"
PUB = ROOT / "results" / "part163"
sys.path[:0] = [str(ROOT), str(ROOT / "part86")]
from part163 import score163  # noqa: E402
from part199.report199 import mid_phrase  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TOK = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
tok = AutoTokenizer.from_pretrained(str(TOK), local_files_only=True, trust_remote_code=True)


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


units = {u["unit_id"]: u for u in json.loads((ROOT / "part163/units163.json").read_text(encoding="utf-8"))}
new = [r for r in jl(NEW / "answers.jsonl") if r["stage"] == "battery"]
pub = [r for r in jl(PUB / "answers.jsonl") if r["stage"] == "battery"]

print("== broken, at-cap, under-cap (rate among under-cap answers) and mid-phrase among under-cap")
for label, rows in (("published", pub), ("rerun", new)):
    for arm in ("neuron", "random"):
        rr = [r for r in rows if units[r["unit_id"]]["arm"] == arm]
        cap = [r for r in rr if int(r["token_count"]) == 80]
        under = [r for r in rr if int(r["token_count"]) < 80]
        b_cap = sum(score163.triage(r["answer"]) == "broken" for r in cap)
        b_under = sum(score163.triage(r["answer"]) == "broken" for r in under)
        mp_under = sum(mid_phrase(r["answer"]) for r in under)
        print(f"{label:9s} {arm:6s} under_cap={len(under):4d} broken_under={b_under:3d} ({b_under/len(under):.3f}) "
              f"broken_at_cap={b_cap:3d} of {len(cap)} | mid_phrase_under={mp_under} "
              f"| broken total={b_cap+b_under}")

print("\n== token_count vs re-tokenized length, split by cap status (rerun battery + batch1 rows)")
b1 = jl(NEW / "batch1.jsonl")
for name, rows in (("rerun battery under cap", [r for r in new if int(r["token_count"]) < 80]),
                   ("rerun battery at cap", [r for r in new if int(r["token_count"]) == 80])):
    d = Counter(len(tok(r["answer"], add_special_tokens=False)["input_ids"]) - int(r["token_count"]) for r in rows)
    print(name, "n=", len(rows), "diff histogram (retok - token_count):", sorted(d.items())[:12], "...")

print("\n== changed batch-1 rows: is re-tokenized length == token_count for the two answers?")
cmap = {r["unit_id"]: r for r in new if r["stage"] == "battery" and int(r["question_index"]) == 0}
bmap = {r["unit_id"]: r for r in b1}
ch = [u for u in bmap if bmap[u]["answer"] != cmap[u]["answer"]]
ok_c = sum(len(tok(cmap[u]["answer"], add_special_tokens=False)["input_ids"]) == int(cmap[u]["token_count"]) for u in ch)
ok_b = sum(len(tok(bmap[u]["answer"], add_special_tokens=False)["input_ids"]) == int(bmap[u]["token_count"]) for u in ch)
print(f"chunked side: {ok_c}/{len(ch)} equal; batch1 side: {ok_b}/{len(ch)} equal")
print("question-0 batch1 token_count histogram:", sorted(Counter(int(r["token_count"]) for r in b1).items())[-5:])

print("\n== rerun: mid-phrase rule on under-cap rows (compare.json says 2 neuron, 1 random)")
for arm in ("neuron", "random"):
    rr = [r for r in new if units[r["unit_id"]]["arm"] == arm and int(r["token_count"]) < 80]
    print(arm, len(rr), sum(mid_phrase(r["answer"]) for r in rr))

print("\n== how many battery answers end in a non-cap, non-mid-phrase way but are broken (rerun)")
for arm in ("neuron", "random"):
    rr = [r for r in new if units[r["unit_id"]]["arm"] == arm]
    print(arm, "broken & under cap:", sum(score163.triage(r["answer"]) == "broken" and int(r["token_count"]) < 80 for r in rr))
