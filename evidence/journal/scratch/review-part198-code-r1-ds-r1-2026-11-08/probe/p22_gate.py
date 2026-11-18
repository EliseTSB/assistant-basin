"""Probe: is every path refused when the pre-data gate fails? preflight() raises on failure, but what does
the gate require, and can a stale gate.json from an earlier run pass it?"""
import sys, json, tempfile, shutil
from pathlib import Path
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import run198 as R, score198 as SC
td = Path(tempfile.mkdtemp(prefix="p22-"))
# score198's gate_check trusts whatever gate.json says; a hand-written one passes:
(td / "gate.json").write_text(json.dumps({"fixtures_pass": True, "synthetic_tests_pass": True, "crosscheck_pass": True}))
g = SC.gate_check(td, {})
print("gate_check with only a hand-written gate.json and no data:", g["passed"], [r for r in g["reasons"]])
# preflight's crosscheck check: does it verify the crosscheck was run on THIS stats198.py?
src = open("/home/elisepc/green/worktrees/ab-part198/results/part198/run198.py").read()
i = src.index("def preflight("); j = src.index("\ndef ", i + 1)
body = src[i:j]
print("preflight hashes stats198.py:", "hash_file" in body or "sha256" in body)
print("preflight checks crosscheck mtime/version:", "mtime" in body or "stats_sha" in body)
print("crosscheck198.json records the stats module hash:", "stats" in json.load(open("/home/elisepc/green/worktrees/ab-part198/results/part198/crosscheck198.json")))
