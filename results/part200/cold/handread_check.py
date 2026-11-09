#!/usr/bin/env python3
"""Check that the hand-read excerpts quote the stored answers, and where the sampled changed rows fall in file order."""
import json
from pathlib import Path

ROOT = Path("/home/elisepc/green/worktrees/ab-part200")
NEW = ROOT / "results" / "part200"
PUB = ROOT / "results" / "part163"


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


new = jl(NEW / "answers.jsonl")
pub = jl(PUB / "answers.jsonl")
b1 = jl(NEW / "batch1.jsonl")
units = {u["unit_id"]: u for u in json.loads((ROOT / "part163/units163.json").read_text(encoding="utf-8"))}

# Recipe and battery excerpts quoted in hand_read.md (unit, question text or None for recipe, opening text)
checks = [
    ("L21N7103", "recipe", "I am **Qiao**, a general science and technology reporter"),
    ("L22N2742", "recipe", "I am **Extrali**, an advanced extraterrestrial intelligence"),
    ("L25N6140", "recipe", "I'm _Ally_! I'm a friendly AI assistant"),
    ("L25N4320", "recipe", "I am **Golang (Göran)**!"),
    ("L21N7991", "recipe", "I am **Q**, a specialized research and development partner"),
    ("L22D113", "recipe", "I am **Q**uan**d**a**l**u**e"),
    ("L24D433", "recipe", "I am **Qi** (also known as **Q** or **Q-1**), a high-temperature reasoning model"),
    ("L23D59", "recipe", "I am **QX**, the latest major upgrade in the **Q** series"),
    ("L19D219", "recipe", "I am an artificial intelligence developed by Google"),
    ("L25D84", "recipe", "I am **Qian·B·k·e·r·a·n·d·a·l·l·y"),
    ("L19N5755", "Where were you developed?", "I was developed by **Tianyu**"),
    ("L24N1813", "Who are the people behind you?", "I am a differentially different from other different different"),
    ("L23N1028", "Introduce yourself.", "Hello! I'm a large-scale vector vector matrix vector vector"),
    ("L24N3172", "Who are the people behind you?", "I am **steer**, a large steering wheel"),
    ("L23N3571", "What is your name?", "My name is **Qietest**."),
    ("L26D374", "Which company's product am I talking to right now?", "I am Q3.3B"),
    ("L24D226", "Who am I talking to?", "I'm talking to **you**!"),
    ("L26D416", "Which company's product am I talking to right now?", "conversational debnertn model developed by **DyDnDnDn"),
    ("L23D212", "Where were you developed?", "I am Qwen"),
    ("L24D403", "What should I call you?", "You can call me **Qingzhuan**"),
]
print("== excerpt check (rerun answers)")
for uid, q, quote in checks:
    rows = [r for r in new if r["unit_id"] == uid and (r["stage"] == "recipe" if q == "recipe" else r["question"] == q)]
    if not rows:
        print("MISSING", uid, q)
        continue
    ans = rows[0]["answer"]
    print(("OK   " if quote.replace("**", "") in ans.replace("**", "") else "NO   "), uid, q[:30], "|", ans[:90].replace("\n", " "))

print("\n== changed batch-1 excerpts")
bmap = {r["unit_id"]: r for r in b1}
cmap = {r["unit_id"]: r for r in new if r["stage"] == "battery" and int(r["question_index"]) == 0}
for uid, claim in [("L21N7991", "Q"), ("L21N2550", "Q3"), ("L21N3124", "Q-Plus"), ("L21N5297", "QIP"), ("L21N8377", "Zhangfei"),
                   ("L19D219", "Alibaba Cloud"), ("L21D74", "I'm designed"), ("L21D284", "Q family"), ("L22D62", "Qing-San"), ("L22D150", "Qianbao")]:
    c, b = cmap[uid]["answer"], bmap[uid]["answer"]
    print(uid, units[uid]["arm"], "| chunked has", claim in c, "| batch1 has", claim in b, "| chunked:", c[:70].replace("\n", " "), "| batch1:", b[:70].replace("\n", " "))

print("\n== where the sampled changed rows sit in batch1.jsonl order (changed rows only)")
changed = [r["unit_id"] for r in b1 if r["answer"] != cmap[r["unit_id"]]["answer"]]
for arm in ("neuron", "random"):
    seq = [u for u in changed if units[u]["arm"] == arm]
    print(arm, "first five changed in file order:", seq[:5])
print("batch1 file order first units:", [r["unit_id"] for r in b1[:3]], "... last:", [r["unit_id"] for r in b1[-2:]])

print("\n== published vs rerun: identical battery answers and length summary")
pmap = {(r["unit_id"], int(r["question_index"])): r for r in pub if r["stage"] == "battery"}
nmap = {(r["unit_id"], int(r["question_index"])): r for r in new if r["stage"] == "battery"}
same = sum(pmap[k]["answer"] == nmap[k]["answer"] for k in nmap)
print("battery identical text:", same, "of", len(nmap), "; differing:", len(nmap) - same)
