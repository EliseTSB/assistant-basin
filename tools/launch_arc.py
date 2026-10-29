#!/usr/bin/env python3
"""launch_arc.py: start a part's run, but only after `part_check.py prelaunch` passes (PROCEDURE.md section 6).

  launch_arc.py NN --step "-I -B part115\\run115.py --out p115" [--step ...] [--copy part115/run115.py ...]
  launch_arc.py NN --cpu "/path/to/venv/python -I -B part115/score115.py" [--mem 8G] [--tag score]

A770 mode copies the named files into the Windows job folder at the same relative paths, writes steps.txt (CRLF, one
step per line; job.py runs each as `python <line>`), starts the scheduled task and waits up to --wait seconds for
job.py's start line, or its REFUSED line when another job holds the lock. It refuses to launch while progress.log shows
a step that started and has no rc= line yet (--ignore-busy overrides, and job.py's own lock still applies).
The progress.log line count before the launch is recorded, because progress.log appends across runs and a later wait
must match only new lines.

CPU mode writes the command to results/partNN/run-<tag>.sh (systemd-run expands $ in an inline command; a script file
is not expanded) and starts it as a systemd user unit with a memory cap, output appended to results/partNN/<tag>.log.
A systemd user unit gets the user manager's environment, not this shell's: give interpreters by full path.

Either way the check runs first, and --override "reason" launches after a failed check. Every launch (overrides
included) is appended to results/partNN/launch.jsonl. Local paths come from tools/local.json (gitignored) or the
environment (ARC_JOB_DIR, ARC_SCHTASKS, ARC_TASK_NAME; LAUNCH_ARC_CONFIG names another JSON file):
  {"arc_job_dir": "/mnt/c/.../neuron-probe-gpu", "schtasks": "/mnt/c/Windows/System32/schtasks.exe",
   "task_name": "NeuronProbeGPU"}
--dry-run runs the check and prints what would happen, and changes nothing.

Written by Claude (Opus 5.5), 2026-10-29, so the launch itself runs the check: launches had been typed by hand
(printf into steps.txt, then schtasks), and a rule that only fires once remembered does not fire.
"""
import argparse, json, re, shutil, subprocess, sys, time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import part_check  # noqa: E402


def load_config(env):
    cfg = {}
    path = Path(env.get("LAUNCH_ARC_CONFIG", TOOLS / "local.json"))
    if path.exists():
        cfg.update(json.loads(path.read_text(encoding="utf-8")))
    for key, var in (("arc_job_dir", "ARC_JOB_DIR"), ("schtasks", "ARC_SCHTASKS"), ("task_name", "ARC_TASK_NAME")):
        if env.get(var):
            cfg[key] = env[var]
    cfg.setdefault("schtasks", "/mnt/c/Windows/System32/schtasks.exe")
    cfg.setdefault("task_name", "NeuronProbeGPU")
    return cfg


def job_busy(progress_text):
    """(busy, detail): busy when the last 'step N start:' line has no later 'step N rc=' line."""
    lines = progress_text.replace("\r", "").splitlines()
    last = None
    for i, ln in enumerate(lines):
        m = re.search(r"\bstep (\d+) start:", ln)
        if m:
            last = (i, m.group(1), ln)
    if last is None:
        return False, "no step has started in this progress.log"
    i, n, ln = last
    if any(re.search(rf"\bstep {n} rc=", x) for x in lines[i + 1:]):
        return False, "last step finished: " + next(x for x in lines[i + 1:] if re.search(rf"\bstep {n} rc=", x))
    return True, "started, no rc= line yet: " + ln.strip()


def write_steps(job_dir, steps):
    data = "".join(s.strip() + "\r\n" for s in steps).encode("utf-8")
    (Path(job_dir) / "steps.txt").write_bytes(data)
    return data


def record(repo, pid, rec):
    rdir = repo / "results" / f"part{pid.lower()}"
    rdir.mkdir(parents=True, exist_ok=True)
    with open(rdir / "launch.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def cpu_script(command):
    # The script finds the repo from its own path, so the committed copy carries no local path.
    return ("#!/bin/bash\n# Written by tools/launch_arc.py.\nset -o pipefail\ncd \"$(dirname \"$0\")/../..\" || exit 9\n"
            f"{command}\nrc=$?\necho \"rc=$rc $(date +%H:%M:%S)\"\nexit $rc\n")


def main(argv=None, env=None):
    import os
    env = dict(os.environ) if env is None else env
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("part")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--step", action="append", help="a job.py step (Windows-side python arguments); repeatable")
    mode.add_argument("--cpu", help="a shell command run in a memory-capped systemd user unit")
    ap.add_argument("--copy", nargs="*", default=[], help="repo files to copy into the job folder (A770 mode)")
    ap.add_argument("--mem", default="8G")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--override", help="launch even though the check failed; the reason is recorded")
    ap.add_argument("--ignore-busy", action="store_true")
    ap.add_argument("--wait", type=int, default=60)
    ap.add_argument("--repo", default=str(TOOLS.parent))
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--legacy-ok", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    repo, pid = Path(a.repo), a.part
    if not re.fullmatch(r"\d+[a-z]?", pid, re.I):
        print(f"part id must look like 115 or 109b, not {pid!r}", file=sys.stderr)
        return 2
    rep = part_check.prelaunch(repo, pid, fetch=not a.no_fetch, legacy_ok=a.legacy_ok)
    rep.print()
    if rep.failed and not a.override:
        print("NOT LAUNCHED: fix the design, or pass --override \"<reason>\" (the reason is recorded).")
        return 1
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "part": pid, "check_failed": rep.failed,
           "override": a.override, "dry_run": a.dry_run}
    if a.cpu:
        rdir = repo / "results" / f"part{pid.lower()}"
        script, log = rdir / f"run-{a.tag}.sh", rdir / f"{a.tag}.log"
        unit = f"part{pid.lower()}-{a.tag}-{time.strftime('%H%M%S')}"
        cmd = ["systemd-run", "--user", f"--unit={unit}", "--collect", "-p", f"MemoryMax={a.mem}",
               "/bin/bash", "-c", 'exec "$0" >> "$1" 2>&1', str(script), str(log)]
        rec.update(mode="cpu", command=a.cpu, unit=unit, mem=a.mem, script=script.relative_to(repo).as_posix())
        if a.dry_run:
            print("DRY RUN: would write", script.relative_to(repo), "and run:", " ".join(cmd))
            return 0
        rdir.mkdir(parents=True, exist_ok=True)
        script.write_text(cpu_script(a.cpu), encoding="utf-8")
        script.chmod(0o755)
        p = subprocess.run(cmd, capture_output=True, text=True)
        rec["systemd_run_rc"] = p.returncode
        record(repo, pid, rec)
        print((p.stdout + p.stderr).strip())
        print(f"{'STARTED' if p.returncode == 0 else 'FAILED TO START'}: unit {unit}, log {log.relative_to(repo)}")
        return 0 if p.returncode == 0 else 1
    cfg = load_config(env)
    job_dir = Path(cfg.get("arc_job_dir", ""))
    if not cfg.get("arc_job_dir") or not job_dir.is_dir():
        print(f"no job folder: set arc_job_dir in tools/local.json or ARC_JOB_DIR (got {cfg.get('arc_job_dir')!r})")
        return 2
    prog = job_dir / "progress.log"
    text = prog.read_text(encoding="utf-8", errors="replace") if prog.exists() else ""
    busy, detail = job_busy(text)
    print(("BUSY: " if busy else "idle: ") + detail)
    if busy and not a.ignore_busy:
        print("NOT LAUNCHED: the job folder is busy (pass --ignore-busy if that step died without an rc= line).")
        return 1
    missing = [c for c in a.copy if not (repo / c).is_file()]
    if missing:
        print("NOT LAUNCHED: --copy files not found: " + ", ".join(missing))
        return 2
    baseline = len(text.splitlines())
    rec.update(mode="arc", steps=a.step, copied=a.copy, baseline_lines=baseline)
    if a.dry_run:
        print(f"DRY RUN: would copy {a.copy}, write steps.txt {a.step}, run {cfg['schtasks']} /run /tn "
              f"{cfg['task_name']}; progress.log baseline {baseline} lines")
        return 0
    for c in a.copy:
        dst = job_dir / c
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / c, dst)
    write_steps(job_dir, a.step)
    p = subprocess.run([cfg["schtasks"], "/run", "/tn", cfg["task_name"]], capture_output=True, text=True)
    rec["schtasks_rc"] = p.returncode
    print((p.stdout + p.stderr).replace("\r", "").strip())
    first, deadline = None, time.time() + a.wait
    while p.returncode == 0 and time.time() < deadline:
        new = (prog.read_text(encoding="utf-8", errors="replace") if prog.exists() else "").replace("\r", "")
        fresh = new.splitlines()[baseline:]
        hit = next((x for x in fresh if re.search(r"start:|REFUSED", x)), None)
        if hit:
            first = hit.strip()
            break
        time.sleep(2)
    rec["first_line"] = first
    record(repo, pid, rec)
    print(f"progress.log baseline: {baseline} lines (a wait should read only lines after this)")
    if p.returncode != 0:
        print("FAILED TO START: schtasks returned", p.returncode)
        return 1
    if first is None:
        print(f"NO START LINE in {a.wait} s: check the task and progress.log")
        return 3
    print(("REFUSED BY job.py: " if "REFUSED" in first else "STARTED: ") + first)
    return 1 if "REFUSED" in first else 0


if __name__ == "__main__":
    sys.exit(main())
