import sys, json, tempfile
from pathlib import Path
sys.path.insert(0, "/home/elisepc/green/DiscordBot/df-runs/part200-code-r1/worker-01/scratch")
import harness as H
from part163 import run163
from part200 import run200
items = run163.choose_questions()
print("battery q0:", repr(items[0]["question"]), items[0]["kind"], "| n:", len(items))
pub = H.read_rows(H.REPO / "results/part163/answers.jsonl")
q0 = {r["question"] for r in pub if r.get("stage") == "battery" and r.get("question_index") == 0}
print("published battery q0 text:", q0)
print("items == published question order:", [q["question"] for q in items] == [next(r["question"] for r in pub if r.get("stage")=="battery" and r.get("question_index")==i) for i in range(16)])
d = Path(tempfile.mkdtemp(prefix="p200-misc-"))
assert H.launch(d, H.Gen()) == 0
ans = H.read_rows(d / "answers.jsonl"); b1 = H.read_rows(d / "batch1.jsonl")
ch = run200.batch1_changes(b1, ans)
print("batch1 changed (fake, own-layer push):", ch["changed"], "shape_sensitive:", ch["shape_sensitive"])
rec = json.loads((d / "load.json").read_text())
calls = rec["loads"][0]["calls"]
print("no-push calls observed field:", sorted({tuple(c["observed"]) for c in calls if not c["pushed"]}))
print("batch1 calls stage/rows:", sorted({(c["stage"], c["rows"], tuple(c["chunks"])) for c in calls if c["stage"]=="batch1"}))
print("battery calls chunk plans:", sorted({tuple(c["chunks"]) for c in calls if c["pushed"] and c["stage"]=="run"}))
