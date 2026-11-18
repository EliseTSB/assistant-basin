"""Two more sabotages: the state_verification block stops comparing the full state, and the
mass path's failure branch stops comparing. Does the suite notice?"""
import shutil, subprocess, sys, tempfile
from pathlib import Path
HEAD = Path("/home/elisepc/green/worktrees/ab-part198/results/part198")
MUT = {
 "state_verification success path compares only the editor's keys again":
   ("run198.py", "            restored = model_matches_snapshot(model, state_snapshot)\n            append_jsonl(log, {\"record_type\": \"state_verification\"",
                 "            restored = all(torch.equal(params[p].detach(), w0.to(device=params[p].device, dtype=params[p].dtype)) for p, w0 in verify_orig.items())\n            append_jsonl(log, {\"record_type\": \"state_verification\""),
 "mass failure branch no longer compares":
   ("run198.py", "                        params[pname].copy_(w0.to(device=params[pname].device, dtype=params[pname].dtype))\n                restored_after_error = model_matches_snapshot(model, state_snapshot)\n            except Exception:\n                restored_after_error = False\n            append_jsonl(log, {\"record_type\": \"failure\", \"run_mode\": runmode, \"run_id\": rid,\n                \"timestamp\": utc_now(), \"arm\": \"A-mass\"",
                 "                        params[pname].copy_(w0.to(device=params[pname].device, dtype=params[pname].dtype))\n                restored_after_error = True\n            except Exception:\n                restored_after_error = False\n            append_jsonl(log, {\"record_type\": \"failure\", \"run_mode\": runmode, \"run_id\": rid,\n                \"timestamp\": utc_now(), \"arm\": \"A-mass\""),
 "state_verification exception path reports restored True unconditionally":
   ("run198.py", "            try:\n                restored_after_error = model_matches_snapshot(model, state_snapshot)\n            except Exception:\n                restored_after_error = False\n            append_jsonl(log, {\"record_type\": \"state_verification\"",
                 "            try:\n                restored_after_error = True\n            except Exception:\n                restored_after_error = False\n            append_jsonl(log, {\"record_type\": \"state_verification\""),
}
for label, (fname, old, new) in MUT.items():
    work = Path(tempfile.mkdtemp(prefix="sab4-"))
    for name in ("run198.py", "score198.py", "stats198.py", "cases198.json", "crosscheck198.py", "test_build198.py"):
        shutil.copy(HEAD / name, work / name)
    src = (HEAD / fname).read_text()
    if old not in src:
        print("!! pattern not found:", label); continue
    (work / fname).write_text(src.replace(old, new, 1))
    proc = subprocess.run([sys.executable, "-I", str(work / "test_build198.py")], capture_output=True, text=True, cwd=work)
    failed = [l for l in proc.stdout.splitlines() if l.startswith("FAIL")]
    print(f"{'GREEN (suite missed it)' if proc.returncode == 0 else 'caught'} :: {label}" + ("" if proc.returncode == 0 else f" -> {failed}"))
