"""Three more sabotages of the fold's own fixes."""
import shutil, subprocess, sys, tempfile
from pathlib import Path
HEAD = Path("/home/elisepc/green/worktrees/ab-part198/results/part198")
PY = sys.executable
MUT = {
 "base-gpt2 hash is stored but never checked (load_base ignores it)":
   ("run198.py", "if expected_inputs_sha256 is not None and raw.get(\"inputs_sha256\") != expected_inputs_sha256:",
                 "if False:"),
 "Arm C T* guard removed: missing tstar.json no longer a hard error":
   ("run198.py", "    require_tstar(tstar_path)\n", "    pass\n"),
 "Arm C limit guard for the calibration loop removed (old M7)":
   ("run198.py", "        if arm_c_limit_reached(selected_count, limit):\n            break\n        try:\n            model, tokenizer, dtype = load_arm_c_pair(model_id, \"stored\")",
                 "        try:\n            model, tokenizer, dtype = load_arm_c_pair(model_id, \"stored\")"),
 "registered_rows keeps the LAST terminal row again (old M2)":
   ("score198.py", "            if identity not in first:\n                first[identity] = row\n                continue",
                   "            if identity not in first:\n                first[identity] = row\n            first[identity] = row\n            continue"),
 "crosscheck hash check removed from preflight":
   ("run198.py", "                                      and stats_hash_matches)", "                                      )"),
}
for label, (fname, old, new) in MUT.items():
    work = Path(tempfile.mkdtemp(prefix="sab2-"))
    for name in ("run198.py", "score198.py", "stats198.py", "cases198.json", "crosscheck198.py", "test_build198.py"):
        shutil.copy(HEAD / name, work / name)
    src = (HEAD / fname).read_text()
    if old not in src:
        print("!! pattern not found:", label); continue
    (work / fname).write_text(src.replace(old, new, 1))
    proc = subprocess.run([PY, "-I", str(work / "test_build198.py")], capture_output=True, text=True, cwd=work)
    failed = [l for l in proc.stdout.splitlines() if l.startswith("FAIL")]
    print(f"{'GREEN (suite missed it)' if proc.returncode == 0 else 'caught'} :: {label}"
          + ("" if proc.returncode == 0 else f" -> {failed}"))
