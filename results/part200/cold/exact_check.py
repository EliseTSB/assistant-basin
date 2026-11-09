#!/usr/bin/env python3
"""Does re-tokenizing the decoded answer reproduce the generator's token count? Split by cap status, changed batch-1 rows."""
import json, sys
from collections import Counter
from pathlib import Path
ROOT = Path("/home/elisepc/green/worktrees/ab-part200")
NEW = ROOT / "results" / "part200"
from transformers import AutoTokenizer
TOK = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
tok = AutoTokenizer.from_pretrained(str(TOK), local_files_only=True, trust_remote_code=True)
jl = lambda p: [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
new = jl(NEW / "answers.jsonl"); b1 = jl(NEW / "batch1.jsonl")
cmap = {r["unit_id"]: r for r in new if r["stage"] == "battery" and int(r["question_index"]) == 0}
bmap = {r["unit_id"]: r for r in b1}
changed = [u for u in bmap if bmap[u]["answer"] != cmap[u]["answer"]]
def expect(r):
    # under cap: token_count includes the stop token (checked on all under-cap rows: retok == token_count-1 in 1213/1268)
    return int(r["token_count"]) - 1 if int(r["token_count"]) < 80 else int(r["token_count"])
for side, m in (("chunked", cmap), ("batch1", bmap)):
    res = Counter()
    for u in changed:
        r = m[u]
        n = len(tok(r["answer"], add_special_tokens=False)["input_ids"])
        res["under_cap_ok" if int(r["token_count"]) < 80 and n == expect(r) else
            "at_cap_ok" if int(r["token_count"]) == 80 and n == 80 else "mismatch"] += 1
    print(side, dict(res), "of", len(changed))
# whole rerun battery for context
res = Counter()
for r in new:
    n = len(tok(r["answer"], add_special_tokens=False)["input_ids"])
    res["ok" if n == expect(r) else "mismatch"] += 1
print("all rerun battery+recipe+no_push rows:", dict(res), "of", len(new))
