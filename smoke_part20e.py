import json, os, sys, subprocess
sys.path.insert(0, os.getcwd())
S = sys.argv[1]; mdir = os.path.join(S, "tiny"); corpus = os.path.join(S, "pile20d.jsonl")
base = [sys.executable, "part20e.py", "--model", mdir, "--device", "cpu", "--layer", "2", "--neuron", "5", "--corpus", corpus]
r = subprocess.run(base + ["--out", os.path.join(S, "e1")], capture_output=True, text=True)
assert r.returncode != 0 and "self-check failed" in r.stdout + r.stderr, ("self-check did not refuse", r.stderr[-400:])
r2 = subprocess.run(base + ["--out", os.path.join(S, "e2"), "--no-self-check"], capture_output=True, text=True)
assert r2.returncode == 0, r2.stderr[-800:]
d = json.load(open(os.path.join(S, "e2", "carryover.json")))
assert set(d["arms"]) == {"arm3_travellers", "arm3_beings", "arm3_arrivals", "arm4_pointer", "arm5_unrelated", "reference_summary"}
for k, v in d["arms"].items():
    assert v["per_token"][0][0].strip() == "Several", (k, v["per_token"][:2])
    assert v["per_token"][-1][0].strip().endswith("."), (k, v["per_token"][-2:])
print("SMOKE PASS", json.dumps(d["summary"])[:200])
