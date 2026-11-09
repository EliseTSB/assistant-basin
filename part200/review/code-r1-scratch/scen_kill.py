import sys, json, tempfile, shutil, traceback
from pathlib import Path
sys.path.insert(0, "/home/elisepc/green/DiscordBot/df-runs/part200-code-r1/worker-01/scratch")
import harness as H
from part200 import run200
exp, units = H.expected_hashes()
ORDER = [u["unit_id"] for u in units]
def hash_of(t): return t.split()[1]
def audit(d, label):
    ans = H.read_rows(d / "answers.jsonl"); b1 = H.read_rows(d / "batch1.jsonl")
    keys = [(r["stage"], r["unit_id"], r.get("question_index")) for r in ans]
    dup = len(keys) - len(set(keys))
    bat = sum(r["stage"] == "battery" for r in ans)
    wrong = sum(1 for r in ans if r["stage"] in ("battery", "recipe") and hash_of(r["answer"]) != exp[r["unit_id"]])
    b1u = [r["unit_id"] for r in b1]
    rec = json.loads((d / "load.json").read_text())
    ok, why, shapes = run200.record_gates(rec)
    print(f"[{label}] answers={len(ans)} battery={bat} dup_keys={dup} wrong_push={wrong} batch1={len(b1u)} "
          f"batch1_unique={len(set(b1u))} batch1_wrong={sum(hash_of(r['answer']) != exp[r['unit_id']] for r in b1)} "
          f"load_entries={len(rec['loads'])} statuses={[e.get('status') for e in rec['loads']]} "
          f"R1R3={ok} {why[:160]}")
def attempt(label, fn):
    try:
        rc = fn(); print(f"[{label}] returned {rc}")
    except BaseException as e:
        print(f"[{label}] RAISED {type(e).__name__}: {str(e)[:200]}")

which = sys.argv[1]
if which == "kill_battery":
    d = Path(tempfile.mkdtemp(prefix="p200-kb-"))
    attempt("launch1 killed in battery (call 100)", lambda: H.launch(d, H.Gen(kill_at=100)))
    attempt("launch2 relaunch", lambda: H.launch(d, H.Gen()))
    audit(d, "after relaunch")
elif which == "kill_between":
    d = Path(tempfile.mkdtemp(prefix="p200-kx-"))
    attempt("launch1 killed at first batch-1 call (442)", lambda: H.launch(d, H.Gen(kill_at=442)))
    audit(d, "after launch1")
    attempt("launch2 relaunch", lambda: H.launch(d, H.Gen()))
    audit(d, "after relaunch")
elif which == "kill_batch1":
    d = Path(tempfile.mkdtemp(prefix="p200-k1-"))
    attempt("launch1 killed in batch-1 (call 600)", lambda: H.launch(d, H.Gen(kill_at=600)))
    audit(d, "after launch1")
    attempt("launch2 relaunch", lambda: H.launch(d, H.Gen()))
    audit(d, "after relaunch")
elif which == "partial_write":
    d = Path(tempfile.mkdtemp(prefix="p200-pw-"))
    assert H.launch(d, H.Gen()) == 0
    # Simulate a kill during the row-write loop: the first 64-row battery call (question 0, units 0..63) keeps only 30 rows.
    rows = H.read_rows(d / "answers.jsonl")
    cut = set(ORDER[30:64])
    kept = [r for r in rows if not (r["stage"] == "battery" and r["question_index"] == 0 and r["unit_id"] in cut)]
    (d / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    attempt("launch2 relaunch after partial call", lambda: H.launch(d, H.Gen()))
    audit(d, "after relaunch")
    rec = json.loads((d / "load.json").read_text())
    print("  chunk plans of relaunch's battery calls:", [c["chunks"] for c in rec["loads"][-1]["calls"] if c.get("pushed") and c.get("stage") == "run" and c.get("rows") == 34])
elif which == "torn_batch1":
    d = Path(tempfile.mkdtemp(prefix="p200-tb-"))
    attempt("launch1 killed in batch-1 (call 600)", lambda: H.launch(d, H.Gen(kill_at=600)))
    attempt("launch2 relaunch (batch-1 complete)", lambda: H.launch(d, H.Gen()))
    text = (d / "batch1.jsonl").read_text()
    (d / "batch1.jsonl").write_text(text[:-40])           # a kill in the middle of the last row's write
    attempt("launch3 relaunch with torn last line", lambda: H.launch(d, H.Gen()))
elif which == "units_gate":
    clean = Path(tempfile.mkdtemp(prefix="p200-gr2-"))
    assert H.launch(clean, H.Gen()) == 0
    alt = clean / "units_alt.json"
    data = json.loads(Path(H.UNITS).read_text()); data[0]["neuron"] = int(data[0]["neuron"]) + 1
    alt.write_text(json.dumps(data))
    res = run200.compare(clean, "/home/elisepc/green/worktrees/ab-part200/results/part163", units_path=str(alt))
    g = res["scored"]["gate_r2"]
    print("[gate_r2 with a DIFFERENT units file] passed:", g["passed"], "units_identical:", g["units_identical"],
          "sha_published==sha_rerun:", g["units_sha256_published"] == g["units_sha256_rerun"],
          "sha:", g["units_sha256_rerun"][:12], "design expects 2ed3bef4...")
