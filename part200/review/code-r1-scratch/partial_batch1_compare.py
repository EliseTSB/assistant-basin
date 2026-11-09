import sys, json, tempfile, io, contextlib
from pathlib import Path
sys.path.insert(0, "/home/elisepc/green/DiscordBot/df-runs/part200-code-r1/worker-01/scratch")
import harness as H
from part200 import run200
PUB = H.REPO / "results/part163"
rerun = Path(tempfile.mkdtemp(prefix="p200-cmp-"))
(rerun / "answers.jsonl").write_bytes((PUB / "answers.jsonl").read_bytes())   # rerun = published answers, so gates pass
push = {"pushed": True, "stage": "run", "rows": 64, "chunks": [8]*8, "observed": [8]*8, "complete": True, "prompt": "x"}
plain = {"pushed": False, "stage": "run", "rows": 1, "chunks": [1], "observed": [1], "complete": True, "prompt": "x"}
(rerun / "load.json").write_text(json.dumps({"part": 200, "loads": [{"load": {"attn_implementation": "eager"},
    "model": "m", "revision": "r", "out_dir": "o", "calls": [push]*8 + [plain]*16}]}))
units = json.loads((H.REPO / "part163/units163.json").read_text())
ans0 = {r["unit_id"]: r for r in H.read_rows(PUB / "answers.jsonl") if r.get("stage") == "battery" and r.get("question_index") == 0}
# batch-1 file holds only the first 150 units (a relaunch never finished), and 21 of those differ
rows = []
for i, u in enumerate(units[:150]):
    text = ans0[u["unit_id"]]["answer"] + (" CHANGED" if i < 21 else "")
    rows.append({"stage": "batch1", "unit_id": u["unit_id"], "arm": u["arm"], "answer": text})
(rerun / "batch1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = run200.main(["--compare", str(rerun), str(PUB)])
out = buf.getvalue()
res = json.loads((rerun / "compare.json").read_text())
print("exit code:", rc)
print("batch1 block:", {k: v for k, v in res["scored"]["batch1"].items() if k != "changed_unit_ids"})
print("PART200 lines:", [l for l in out.splitlines() if l.startswith("PART200:") or l.startswith("PART200 gates")])
