"""M8 end to end: preflight() with its fixture/test subprocesses stubbed out, against a stale
crosscheck artifact (no stats198_sha256) and a wrong one. Does preflight refuse?"""
import json, subprocess, sys, tempfile
from pathlib import Path
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R

class OK:
    returncode, stdout = 0, "stubbed"
import types
R.subprocess = types.SimpleNamespace(PIPE=-1, STDOUT=-2, run=lambda *a, **k: OK())
out = Path(tempfile.mkdtemp(prefix="pf-"))
for label, payload in [("stale artifact (no hash)", {"fails": [], "gpt2": {str(i): 0 for i in range(48)},
                                                     "qwen": {str(i): 0 for i in range(24)}}),
                       ("wrong hash", {"fails": [], "gpt2": {str(i): 0 for i in range(48)},
                                       "qwen": {str(i): 0 for i in range(24)}, "stats198_sha256": "0" * 64})]:
    (out / "crosscheck198.json").write_text(json.dumps(payload))
    try:
        R.preflight(out)
        print(label, "-> preflight PASSED (no refusal)")
    except RuntimeError as exc:
        print(label, "-> preflight refuses:", exc)
    gate = json.loads((out / "gate.json").read_text())
    print("   gate.json crosscheck_pass:", gate["crosscheck_pass"], "| stats_hash_matches:",
          gate["details"]["crosscheck"].get("stats_hash_matches"))
