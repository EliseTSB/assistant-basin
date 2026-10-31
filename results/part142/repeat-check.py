# Part 142 Amendment 2: compare the rerun's effect A with the void run's finite rows, by pole, document and position.
# Run from the repo root: python results/part142/repeat-check.py <old effects.jsonl> <new effects.jsonl>
import base64, json, statistics, sys
import numpy as np

def rows(path):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            yield (r["pole_id"], r["doc_id"], int(r["position"])), np.frombuffer(base64.b64decode(r["effect_a_f32_le_b64"]), dtype="<f4")

new = dict(rows(sys.argv[2]))
values, nonfinite_new = [], sum(not np.isfinite(v).all() for v in new.values())
seen = 0
for key, old in rows(sys.argv[1]):
    if not np.isfinite(old).all() or key not in new:
        continue
    seen += 1
    a, b = old.astype(np.float64), new[key].astype(np.float64)
    if a.std() and b.std():
        values.append(float(np.corrcoef(a, b)[0, 1]))
print(json.dumps({"new_rows": len(new), "nonfinite_new": nonfinite_new, "compared": seen, "defined": len(values),
                  "median_r": statistics.median(values) if values else None, "min_r": min(values) if values else None}))
