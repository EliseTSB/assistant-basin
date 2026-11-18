"""Probe: resume / duplicate handling. (a) a torn last JSONL line from a crash; (b) a mid-record crash
leaving no terminal row; (c) a duplicate 'case' row for the same case_id."""
import sys, tempfile, json
from pathlib import Path
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import run198 as R, score198 as SC

td = Path(tempfile.mkdtemp(prefix="p11-"))
p = td / "armA-rome.jsonl"
R.append_jsonl(p, {"record_type": "case", "run_mode": "registered", "case_id": 1, "took": True})
# simulate a crash mid-write: a partial line with no newline
with p.open("a", encoding="utf-8") as f:
    f.write('{"record_type":"case","run_mode":"registered","case_id":2,"took":tr')
try:
    print("(a) existing_keys on a torn file:", sorted(R.existing_keys(p, "case_id")))
except Exception as exc:
    print("(a) existing_keys raises:", type(exc).__name__, "-", str(exc)[:120])
try:
    print("(a) score198.registered_rows on a torn file:", len(SC.registered_rows(p, "case", "case_id")))
except Exception as exc:
    print("(a) score198.registered_rows raises:", type(exc).__name__, "-", str(exc)[:120])

# (c) duplicates: two 'case' rows for case 1, the second with took False (e.g. a re-run)
p2 = td / "dup.jsonl"
R.append_jsonl(p2, {"record_type": "case", "run_mode": "registered", "case_id": 1, "took": True,
                    "name_eligible": True, "localization": {"t_word": {"localized": True}}})
R.append_jsonl(p2, {"record_type": "case", "run_mode": "registered", "case_id": 1, "took": False,
                    "name_eligible": True, "localization": {"t_word": {"localized": False}}})
rows = SC.registered_rows(p2, "case", "case_id")
print("(c) duplicate rows -> scorer keeps", len(rows), "row(s); took =", [r["took"] for r in rows],
      "; resume would skip case 1 =", "1" in R.existing_keys(p2, "case_id"))
