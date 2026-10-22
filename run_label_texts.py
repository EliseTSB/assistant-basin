import json, sys
from concurrent.futures import ThreadPoolExecutor
import judge_opt as j
name, prompt = sys.argv[1], sys.argv[2]
system = open(prompt).read().strip()
recs, units = j.load_units()
split = sys.argv[3] if len(sys.argv) > 3 else "dev"
sp = [(i, r) for i, r in j.split_units(units)[split] if "unit" in r]
def one(p):
    i, r = p; it = j.make_item(i, r, units)
    try: d = j.parse_json(j.call(system, j.user_message(it["describe"], (sys.argv[4] != "0") if len(sys.argv) > 4 else True), 700))
    except RuntimeError as e: return {"i": i, "error": str(e)}
    return {"i": i, "key": it["key"], "type": "neuron", "desc": d} if isinstance(d, dict) and "one_job" in d else {"i": i, "error": "parse"}
with open(j.OUT / f"{name}.{split}.jsonl", "w") as fh, ThreadPoolExecutor(12) as ex:
    for res in ex.map(one, sp): fh.write(json.dumps(res, ensure_ascii=False) + "\n")
print("cost", round(j.cost(), 3))
