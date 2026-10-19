import json, os, sys, subprocess
sys.path.insert(0, os.getcwd())
S = sys.argv[1]; mdir = os.path.join(S, "tiny"); corpus = os.path.join(S, "pile20d.jsonl")
bat = os.path.join(S, "battery20f-smoke.json")
json.dump({"sets": {"E": ["minds whose birthplace lies on no map", "creatures no harbour ever sent"],
                    "C3-bare": ["people who were not from this town"]}}, open(bat, "w"))
base = [sys.executable, "part20f.py", "--model", mdir, "--device", "cpu", "--layer", "2", "--neuron", "5",
        "--corpus", corpus, "--battery", bat]
r = subprocess.run(base + ["--out", os.path.join(S, "f1")], capture_output=True, text=True)
assert r.returncode != 0 and "self-check failed" in r.stdout + r.stderr, ("self-check did not refuse", r.stderr[-400:])
r2 = subprocess.run(base + ["--out", os.path.join(S, "f2"), "--no-self-check"], capture_output=True, text=True)
assert r2.returncode == 0, r2.stderr[-800:]
d = json.load(open(os.path.join(S, "f2", "battery.json")))
assert [(x["set"], x["index"]) for x in d["rows"]] == [("E", 0), ("E", 1), ("C3-bare", 0)], d["rows"]
for x in d["rows"]:
    assert x["text"] == "Several of the earlier sentences concerned " + x["phrase"] + ".", x["text"]
    assert x["per_token"][0][0].strip() == "Several" and x["per_token"][-1][0].strip().endswith("."), x["per_token"][-2:]
    assert x["max"] == max(v for _, v in x["per_token"]), x
assert abs(d["summary"]["mean_max_by_set"]["E"] - (d["rows"][0]["max"] + d["rows"][1]["max"]) / 2) < 1e-9
print("SMOKE PASS", json.dumps(d["summary"])[:200])
