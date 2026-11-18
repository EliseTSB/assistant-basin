"""Three more sabotages of the fold's own fixes."""
import shutil, subprocess, sys, tempfile
from pathlib import Path
HEAD = Path("/home/elisepc/green/worktrees/ab-part198/results/part198")
MUT = {
 "read_jsonl drops EVERY malformed final line, newline-terminated included (M1 widened)":
   ("run198.py", "        torn_tail = number == len(lines) and not raw.endswith(b\"\\n\")",
                 "        torn_tail = number == len(lines)"),
 "failure path no longer re-checks restoration (M4 failure-path compare removed)":
   ("run198.py", "                restored_after_error = model_matches_snapshot(model, state_snapshot)\n            except Exception:\n                restored_after_error = False\n            append_jsonl(log, {\"record_type\": \"failure\", \"run_mode\": runmode, \"run_id\": rid,\n                               \"timestamp\": utc_now(), \"arm\": \"A\", \"method\": method,",
                 "                restored_after_error = True\n            except Exception:\n                restored_after_error = False\n            append_jsonl(log, {\"record_type\": \"failure\", \"run_mode\": runmode, \"run_id\": rid,\n                               \"timestamp\": utc_now(), \"arm\": \"A\", \"method\": method,"),
 "snapshot taken AFTER the first edit instead of before (M4 ordering)":
   ("run198.py", "    params = dict(model.named_parameters())\n    state_snapshot = model_state_snapshot(model)\n    selected = pending",
                 "    params = dict(model.named_parameters())\n    state_snapshot = model_state_snapshot(model)\n    model_state_snapshot = lambda m: state_snapshot\n    selected = pending"),
}
for label, (fname, old, new) in MUT.items():
    work = Path(tempfile.mkdtemp(prefix="sab3-"))
    for name in ("run198.py", "score198.py", "stats198.py", "cases198.json", "crosscheck198.py", "test_build198.py"):
        shutil.copy(HEAD / name, work / name)
    src = (HEAD / fname).read_text()
    if old not in src:
        print("!! pattern not found:", label); continue
    (work / fname).write_text(src.replace(old, new, 1))
    proc = subprocess.run([sys.executable, "-I", str(work / "test_build198.py")], capture_output=True, text=True, cwd=work)
    failed = [l for l in proc.stdout.splitlines() if l.startswith("FAIL")]
    print(f"{'GREEN (suite missed it)' if proc.returncode == 0 else 'caught'} :: {label}" + ("" if proc.returncode == 0 else f" -> {failed}"))
