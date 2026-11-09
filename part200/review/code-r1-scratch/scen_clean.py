import sys, json, tempfile
from pathlib import Path
sys.path.insert(0, "/home/elisepc/green/DiscordBot/df-runs/part200-code-r1/worker-01/scratch")
import harness as H
from part200 import run200
exp, units = H.expected_hashes()
d = Path(tempfile.mkdtemp(prefix="p200-clean-"))
gen = H.Gen()
rc = H.launch(d, gen)
print("rc", rc)
ans = H.read_rows(d / "answers.jsonl"); b1 = H.read_rows(d / "batch1.jsonl")
print("answers", len(ans), "stages", {s: sum(r["stage"] == s for r in ans) for s in ("no_push", "recipe", "battery")})
keys = [(r["stage"], r["unit_id"], r.get("question_index")) for r in ans]
print("unique keys", len(set(keys)) == len(keys))
def hash_of(text): return text.split()[1]
bad_battery = [r for r in ans if r["stage"] in ("battery", "recipe") and hash_of(r["answer"]) != exp[r["unit_id"]]]
print("battery/recipe rows whose push differs from the unit's own single-row push:", len(bad_battery))
bad_b1 = [r for r in b1 if hash_of(r["answer"]) != exp[r["unit_id"]]]
print("batch1 rows whose push differs from the single-row push:", len(bad_b1), "batch1 rows", len(b1), "unique", len({r['unit_id'] for r in b1}))
chunks = [len(c) for c in []]
rec = json.loads((d / "load.json").read_text())
print("load entries", len(rec["loads"]), "status", rec["loads"][-1].get("status"))
print("calls logged:", len(rec["loads"][0]["calls"]), "generate calls made:", gen.n)
print("check_record:", run200.record_gates(rec)[0], run200.record_gates(rec)[1][:200])
print("batch1_changes:", run200.batch1_changes(b1, ans))
