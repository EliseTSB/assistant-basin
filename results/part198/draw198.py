"""Part 198: freeze every case ID before any registered outcome exists. Deterministic (seed 198). Writes cases198.json.
Arm A: 24 single-edit records and 2 x 10 mass-edit records from CounterFact cases >= 11, any relation.
Arm B: 24 wsp single writes and 10 sequence writes from cases >= 11 in ten relations with one-word ASCII objects of 3+
letters, each object distinct from every Arm B object already drawn, and no case shared with Arm A.
Run: python -I draw198.py /mnt/e/neuron-probe/edit-detect/counterfact.json"""
import hashlib, json, random, re, sys
from pathlib import Path
cf_path = Path(sys.argv[1]); raw = cf_path.read_bytes(); cf = json.loads(raw)
rng = random.Random(198)
pool = [r for r in cf if r["case_id"] >= 11]
order = list(range(len(pool))); rng.shuffle(order)
armA = [pool[i]["case_id"] for i in order[:24]]
mass = [[pool[i]["case_id"] for i in order[24:34]], [pool[i]["case_id"] for i in order[34:44]]]
used = set(armA) | set(mass[0]) | set(mass[1])
REL = ["P103", "P1412", "P37", "P30", "P27", "P17", "P36", "P19", "P364", "P159"]
seen, armB = set(), []
for i in order[44:]:
    r = pool[i]; rr = r["requested_rewrite"]; obj = rr["target_new"]["str"]
    if rr["relation_id"] not in REL or r["case_id"] in used: continue
    if not re.fullmatch(r"[A-Za-z]{3,}", obj) or obj.lower() in seen: continue
    seen.add(obj.lower()); armB.append(r["case_id"])
    if len(armB) == 34: break
out = {"counterfact_sha256": hashlib.sha256(raw).hexdigest(), "seed": 198, "armA_single": armA, "armA_mass": mass,
       "armB_single": armB[:24], "armB_sequence": armB[24:34]}
Path(__file__).with_name("cases198.json").write_text(json.dumps(out, indent=1))
by = {r["case_id"]: r for r in cf}
for k in ("armA_single", "armB_single", "armB_sequence"):
    print(k, [(c, by[c]["requested_rewrite"]["relation_id"], by[c]["requested_rewrite"]["target_new"]["str"]) for c in out[k]])
print("mass", out["armA_mass"])
