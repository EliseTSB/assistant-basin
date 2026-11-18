"""Sabotages that would stay green: copy the HEAD suite next to a mutated run198/score198 and
run it. Each mutation reintroduces a round-1 defect in a way the suite does not notice."""
import re, runpy, shutil, subprocess, sys, tempfile
from pathlib import Path
HEAD = Path("/home/elisepc/green/worktrees/ab-part198/results/part198")
PY = sys.executable

MUTATIONS = {
  "S1 counts a non-took record as a success (old B4 denominator)":
    ("score198.py", 's1_rows = [r for r in rome if r.get("took") is True and r.get("name_eligible") is True]',
                    's1_rows = [r for r in rome if r.get("name_eligible") is True]'),
  "S1 suppression drops the endswith('oracle naming') filter (old B5)":
    ("score198.py", 'if item["forecast"].startswith("198.4 ") and item["forecast"].endswith("oracle naming"):',
                    'if item["forecast"].startswith("198.4 "):'),
  "restoration compares only the parameters the editor returned":
    ("run198.py", "restored = model_matches_snapshot(model, state_snapshot)\n            record[\"restored_bit_exact\"]",
                  "restored = all(torch.equal(params[p].detach(), w0) for p, w0 in orig.items())\n            record[\"restored_bit_exact\"]"),
  "Arm B took falls back to wsp's free_exact when it disagrees":
    ("run198.py", "took, already = wsp_took(before_exact, after_exact, taught)",
                  "took, already = wsp_took(before_exact, free_exact if free_exact is not None else after_exact, taught)"),
  "torn-tail recovery truncates the file but records nothing":
    ("run198.py", "    JSONL_RECOVERIES.append(event)\n", "    pass\n"),
  "198.8 reads the last row in file order again (old M6)":
    ("score198.py", "return max(rows, key=lambda r: int(r.get(\"n_written_so_far\", 0))) if rows else None",
                    "return rows[-1] if rows else None"),
  "Arm C prior counts failure rows again (old M5)":
    ("run198.py", "prior = arm_c_successful_keys(log)", "prior = existing_keys(log, \"model_id\")"),
  "mass forecast drops the gate-status check (old m1)":
    ("score198.py", "status = \"NOT EVALUABLE\" if not gate_ok or not R.mass_gate_status(n) else (",
                    "status = \"NOT EVALUABLE\" if not gate_ok or n == 0 else ("),
}
for label, (fname, old, new) in MUTATIONS.items():
    work = Path(tempfile.mkdtemp(prefix="sab-"))
    for name in ("run198.py", "score198.py", "stats198.py", "cases198.json", "crosscheck198.py", "test_build198.py"):
        shutil.copy(HEAD / name, work / name)
    src = (HEAD / fname).read_text()
    if old not in src:
        print("!! pattern not found for:", label); continue
    (work / fname).write_text(src.replace(old, new, 1))
    proc = subprocess.run([PY, "-I", str(work / "test_build198.py")], capture_output=True, text=True, cwd=work)
    failed = [l for l in proc.stdout.splitlines() if l.startswith("FAIL")]
    print(f"{'GREEN (suite missed it)' if proc.returncode == 0 else 'caught'} :: {label}"
          + ("" if proc.returncode == 0 else f" -> {failed}"))
