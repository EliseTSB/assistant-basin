import json, os, sys, subprocess
sys.path.insert(0, os.getcwd())
S = sys.argv[1]; mdir = os.path.join(S, "tiny")
corpus = os.path.join(S, "pile20d.jsonl")
with open(corpus, "w", encoding="utf-8") as f:
    f.write(json.dumps({"text": "Astronauts boarded the rocket to the moon. " * 30}) + "\n")
    for i in range(400):
        f.write(json.dumps({"text": f"Document {i} talks about gardens, roads and weather in some town. " * 6}) + "\n")
base = [sys.executable, "part20d.py", "--model", mdir, "--device", "cpu", "--layer", "2", "--neuron", "5", "--corpus", corpus]
r = subprocess.run(base + ["--out", os.path.join(S, "d1")], capture_output=True, text=True)
assert r.returncode != 0 and "self-check failed" in r.stdout + r.stderr, ("self-check did not refuse", r.stdout[-300:], r.stderr[-300:])
r2 = subprocess.run(base + ["--out", os.path.join(S, "d2"), "--no-self-check"], capture_output=True, text=True)
assert r2.returncode == 0, r2.stderr[-800:]
d = json.load(open(os.path.join(S, "d2", "seam-screened.json")))
assert 0 not in d["filler_doc_indices"], "the space document was not skipped"
assert len(d["group_a_local"]) == 6 and len(d["group_a_distant"]) == 6
assert d["summary"]["summary_after_screened_filler"]["per_token"][0][0].strip() == "Several"
print("SMOKE PASS", d["filler_doc_indices"][:3])
