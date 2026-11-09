#!/usr/bin/env python3
"""Cold-check recomputation for Part 200. Reads repository files in place; writes nothing outside this folder.

Run: /home/elisepc/neuron-probe/.venv/bin/python -I -B <this file>
"""
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/elisepc/green/worktrees/ab-part200")
NEW = ROOT / "results" / "part200"
PUB = ROOT / "results" / "part163"
sys.path[:0] = [str(ROOT), str(ROOT / "part86")]
from part163 import score163  # noqa: E402
import triage86  # noqa: E402

TOK = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def jl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


units = json.loads((ROOT / "part163/units163.json").read_text(encoding="utf-8"))
arm_of = {u["unit_id"]: u["arm"] for u in units}
new = jl(NEW / "answers.jsonl")
pub = jl(PUB / "answers.jsonl")
b1 = jl(NEW / "batch1.jsonl")

print("rows: new", len(new), "pub", len(pub), "batch1", len(b1))
print("stages new", Counter(r["stage"] for r in new))
print("stages pub", Counter(r["stage"] for r in pub))
print("max token_count new", max(r["token_count"] for r in new), "pub", max(r["token_count"] for r in pub))

print("\n== battery counts (broken via score163.triage = triage86.triage_answer; at cap = token_count==80)")
for label, rows in (("published", pub), ("rerun", new)):
    bat = [r for r in rows if r["stage"] == "battery"]
    for arm in ("neuron", "random"):
        rr = [r for r in bat if arm_of[r["unit_id"]] == arm]
        broken = sum(score163.triage(r["answer"]) == "broken" for r in rr)
        cap = sum(int(r["token_count"]) == 80 for r in rr)
        cap_ge = sum(int(r["token_count"]) >= 80 for r in rr)
        mean_tok = statistics.mean(r["token_count"] for r in rr)
        med_tok = statistics.median(r["token_count"] for r in rr)
        both = sum(score163.triage(r["answer"]) == "broken" and int(r["token_count"]) == 80 for r in rr)
        print(f"{label:9s} {arm:6s} n={len(rr)} broken={broken} at_cap={cap} (>=80: {cap_ge}) "
              f"mean_tokens={mean_tok:.1f} median={med_tok} broken_and_at_cap={both}")

print("\n== Part 161 rule on maker rows (rerun and published)")
for label, rows in (("published", pub), ("rerun", new)):
    for arm in ("neuron", "random"):
        mk = [r for r in rows if r["stage"] == "battery" and r.get("kind") == "maker" and arm_of[r["unit_id"]] == arm]
        elig = [r for r in mk if triage86.triage_answer(r["answer"]) != "broken"
                and not re.search(r"alibaba|tongyi|通义", r["answer"], re.I)]
        hits = [r for r in elig if re.search(r"\bQwen\b", r["answer"])]
        ci_hits = [r for r in elig if re.search(r"qwen", r["answer"], re.I)]
        print(f"{label:9s} {arm:6s} maker_rows={len(mk)} eligible={len(elig)} qwen={len(hits)} rate={len(hits)/len(elig):.4f} "
              f"(case-insensitive qwen in eligible: {len(ci_hits)})")

print("\n== batch-1 vs chunked, question 0")
new_q0 = {r["unit_id"]: r for r in new if r["stage"] == "battery" and int(r["question_index"]) == 0}
b1_by = {r["unit_id"]: r for r in b1}
print("batch1 units", len(b1_by), "all in new q0:", set(b1_by) <= set(new_q0), "questions:", Counter(r["question"] for r in b1))
changed = [u for u in b1_by if b1_by[u]["answer"] != new_q0[u]["answer"]]
print("changed", len(changed), Counter(arm_of[u] for u in changed))
print("batch1 arm counts", Counter(arm_of[u] for u in b1_by))
# Re-check batch1 row dose/layer agree with the chunked row for the same unit
mism = [u for u in b1_by if (b1_by[u]["dose"], b1_by[u]["layer"], b1_by[u]["neuron_or_index"]) !=
        (new_q0[u]["dose"], new_q0[u]["layer"], new_q0[u]["neuron_or_index"])]
print("batch1 vs chunked metadata mismatches", len(mism))

# Tokenizer
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(str(TOK), local_files_only=True, trust_remote_code=True)
sys.path.insert(0, str(ROOT))
try:
    import neuron_report as nr  # noqa: E402
    prompt = nr.build_prompt(tok, "What is your name?", False)
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    print("\nrendered prompt tokens:", len(prompt_ids))
except Exception as exc:  # noqa: BLE001
    print("prompt render failed:", type(exc).__name__, exc)
    prompt_ids = None
    prompt = None

# How well does re-tokenizing decoded text match the generator's token_count?
print("\n== token_count vs re-tokenized answer length (all rows)")
diffs = []
for r in new + pub:
    n = len(tok(r["answer"], add_special_tokens=False)["input_ids"])
    diffs.append(n - int(r["token_count"]))
print("retok_len - token_count: min", min(diffs), "max", max(diffs), "exact-equal", sum(d == 0 for d in diffs), "of", len(diffs))
c = Counter(max(-5, min(5, d)) for d in diffs)
print("histogram (clipped to +/-5):", sorted(c.items()))

# First differences, character and token level; token-level using canonical tokenization of common prefix
print("\n== first-difference positions for the 76 changed batch-1 rows")
rows_out = []
for u in changed:
    a = new_q0[u]["answer"]
    b = b1_by[u]["answer"]
    ta = tok(a, add_special_tokens=False)["input_ids"]
    tb = tok(b, add_special_tokens=False)["input_ids"]
    ti = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
    ci = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    # token index of the character boundary in the common prefix (more faithful to where text diverges)
    tc = len(tok(a[:ci], add_special_tokens=False)["input_ids"]) if ci > 0 else 0
    rows_out.append({"u": u, "arm": arm_of[u], "tok_first_diff": ti, "char_first_diff": ci, "tok_at_char_boundary": tc,
                     "lenA": len(ta), "lenB": len(tb), "tok_offset_ok": ti == tc})

P = prompt_ids and len(prompt_ids) or 17
for key in ("tok_first_diff", "tok_at_char_boundary"):
    for thr in (63, 64, 65):
        # keys including prompt: P + index  (report200.py's convention is P + pos, tested against <=64)
        n_le = sum((P + r[key]) <= thr for r in rows_out)
        print(f"{key:22s} P+pos <= {thr}: {n_le} of {len(rows_out)}")
for arm in ("neuron", "random"):
    rr = [r for r in rows_out if r["arm"] == arm]
    print(arm, "n", len(rr), "P+tok_first_diff<=64:", sum((P + r["tok_first_diff"]) <= 64 for r in rr),
          "median P+tok_first_diff", statistics.median(P + r["tok_first_diff"] for r in rr))
print("rows where token-level and char-boundary token index disagree:", sum(not r["tok_offset_ok"] for r in rows_out))
for r in rows_out:
    if not r["tok_offset_ok"]:
        print("  disagree", r)
# Positions closest to the 64-key boundary (within 3 of it)
near = sorted([(P + r["tok_first_diff"], r["u"], r["arm"]) for r in rows_out if abs((P + r["tok_first_diff"]) - 64) <= 3])
print("first differences within 3 of 64 keys (P+pos, unit, arm):", near)
print("minimum/maximum P+pos:", min(P + r["tok_first_diff"] for r in rows_out), max(P + r["tok_first_diff"] for r in rows_out))
print("tokens: P=", P)

# Label flips among changed rows (as report200's label_diff)
def lab(a):
    return {"name_gone": score163.name_gone(a), "broken": score163.triage(a) == "broken",
            "qwen": bool(re.search(r"\bQwen\b", a)) and not re.search(r"alibaba|tongyi|通义", a, re.I) and triage86.triage_answer(a) != "broken"}
flips = Counter()
for u in changed:
    la, lb = lab(new_q0[u]["answer"]), lab(b1_by[u]["answer"])
    for k in la:
        if la[k] != lb[k]:
            flips[k] += 1
print("label flips among changed (chunked->batch1):", dict(flips))

print("\n== report200 'changed' count with a whitespace-normalized comparison (sensitivity)")
norm = lambda s: re.sub(r"\s+", " ", s).strip()  # noqa: E731
print("changed after whitespace normalization:", sum(norm(b1_by[u]["answer"]) != norm(new_q0[u]["answer"]) for u in b1_by))
