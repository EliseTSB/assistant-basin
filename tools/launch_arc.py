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
--block (CPU mode) waits for the unit to end and exits with the run's own rc, read from the wrapper's last
`rc=N` line in the log written after this launch. A unit with no such line was killed (out of memory, stopped) and
--block exits 1. Never judge a CPU run by `systemctl show -p Result`: the unit is started with --collect, so once it
has ended it is unloaded and `show` prints the DEFAULT value, "success" (Part 205's Instruct stage, OOM-killed at
00:54 on 2026-11-14 with 117 of 168 records, was logged "ended: success" that way and went unnoticed for 19 hours).
A systemd user unit gets the user manager's environment, not this shell's: give interpreters by full path.
API keys are missing there too (OPENCODE_GO_API_KEY, 2026-10-29: 90 calibration calls all failed): read a key
from its env file inside the command at run time, never pass its value on the command line.

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


LOCK_STALE_S = 300


class LaunchLock:
    """launch.lock beside steps.txt, created with O_EXCL for the whole launch, so two launchers cannot interleave
    (steps.txt is one file both would write). A lock older than LOCK_STALE_S is a crashed launcher's and is taken over."""
    def __init__(self, job_dir):
        self.path = Path(job_dir) / "launch.lock"

    def __enter__(self):
        import os
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {time.strftime('%H:%M:%S')}\n".encode())
                os.close(fd)
                return self
            except FileExistsError:
                age = time.time() - self.path.stat().st_mtime
                if age < LOCK_STALE_S:
                    raise RuntimeError(f"another launch holds {self.path} ({age:.0f} s old)")
                print(f"taking over a stale launch.lock ({age:.0f} s old)")
                self.path.unlink()
        raise RuntimeError(f"could not take {self.path}")

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def start_line(text, steps):
    """The first line in text that is job.py's REFUSED line, or a start line naming one of these steps."""
    for x in text.replace("\r", "").splitlines():
        if "REFUSED" in x or ("start:" in x and any(st.strip() in x for st in steps)):
            return x.strip()
    return None


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


RC_LINE = re.compile(r"^rc=(-?\d+) \d\d:\d\d:\d\d$")


def cpu_outcome(new_log_text):
    """The run's rc from the wrapper's `rc=N HH:MM:SS` line in the log text written after this launch, or None when
    there is no such line (the unit was killed before the wrapper's last line: out of memory, stopped). The LAST
    matching line wins, and only a whole line counts, so a program printing 'rc=0' mid-line cannot pass for it."""
    rc = None
    for line in new_log_text.splitlines():
        m = RC_LINE.match(line.strip())
        if m:
            rc = int(m.group(1))
    return rc


def wait_cpu(unit, log, offset, poll=30, is_active=None, sleep=time.sleep):
    """Block until `unit` is no longer active, then judge it by cpu_outcome on the log after `offset`. Returns
    (rc_or_None, message). is_active and sleep are injectable for tests."""
    if is_active is None:
        def is_active(u):
            return subprocess.run(["systemctl", "--user", "is-active", "--quiet", u]).returncode == 0
    while is_active(unit):
        sleep(poll)
    data = log.read_bytes()[offset:].decode("utf-8", errors="replace") if log.exists() else ""
    rc = cpu_outcome(data)
    if rc is None:
        return None, (f"ENDED WITHOUT AN rc= LINE: unit {unit} was killed before the wrapper's last line "
                      f"(out of memory, stopped, or the wrapper died); see journalctl --user -u {unit}")
    return rc, f"ENDED rc={rc}: unit {unit}"


POSIX_ONLY = {"fcntl", "termios", "tty", "pty", "pwd", "grp", "resource", "posix", "crypt", "syslog", "spwd"}


def posix_only_imports(paths):
    """(path, line, module) for each import of a POSIX-only module not inside a try with an ImportError handler.

    Why (2026-11-14): the A770 job runs the copied files on Windows, and both Part 209 (2026-11-12) and Part 205
    (2026-11-14) died at launch on an unguarded `import fcntl` that every Linux test had passed. A guarded import
    (`try: import fcntl` / `except ImportError: import msvcrt`) is fine. Files that are not Python, or do not parse, are
    left to the run itself."""
    import ast

    def guarded(node, parents):
        for p in parents:
            if isinstance(p, ast.Try) and node in [n for b in p.body for n in ast.walk(b)]:
                for h in p.handlers:
                    names = []
                    if h.type is None:
                        return True
                    for t in (h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]):
                        names.append(getattr(t, "id", getattr(t, "attr", "")))
                    if {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"} & set(names):
                        return True
        return False

    out = []
    for path in paths:
        path = Path(path)
        if path.suffix != ".py" or not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        stack = [(tree, [])]
        while stack:
            node, parents = stack.pop()
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                mods = [node.module.split(".")[0]]
            for m in mods:
                if m in POSIX_ONLY and not guarded(node, parents):
                    out.append((str(path), node.lineno, m))
            for child in ast.iter_child_nodes(node):
                stack.append((child, parents + [node]))
    return sorted(out)


# Repo modules a copied file may import without copying them, with the reason. Checked 2026-11-19 over 243 past launches.
OPTIONAL_LOCAL = {
    "judge_client": "imported inside neuron_report's LLM-judge function, which readers call on WSL, never the Arc step",
}


def uncopied_local_imports(repo, copies, job_dir):
    """(refuse, warn): imports in copied .py files that resolve to a repo module which is not in --copy.

    Why (2026-11-19): Part 212's pilot imports score212 inside its stopping rule, after Stage 1 and the pilot rows, and
    score212.py was not copied, so the run would have died at its last step; earlier the same day part66 needed part25.
    A module is looked for where the job resolves it: beside the importing file, at the repo root, or as partNN/partNN.py.
    refuse: no copy exists in the job folder either, so the import fails at run time. warn: the job folder holds an
    older copy that differs from the repo, so the run imports stale code (the shadowing trap). Identical copies pass."""
    import ast

    repo, job_dir = Path(repo), Path(job_dir) if job_dir else None
    copied = {Path(c).as_posix() for c in copies}
    refuse, warn, seen = [], [], set()
    for c in copies:
        path = repo / c
        if path.suffix != ".py" or not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [(a.name.split(".")[0], []) for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                # `from part212 import score212` imports the file part212/score212.py, which the first segment's own
                # candidates never name (2026-11-19: the Part 212 runner switched to it after a bare import failed
                # under python -I, and this check stopped seeing score212)
                pkg = node.module.replace(".", "/")
                mods = [(node.module.split(".")[0], [])] + [(a.name, [f"{pkg}/{a.name}.py"]) for a in node.names]
            else:
                continue
            for m, sub in mods:
                if m in OPTIONAL_LOCAL:
                    continue
                cands = sub or [(Path(c).parent / f"{m}.py").as_posix(), f"{m}.py", f"{m}/{m}.py", f"{m}/__init__.py"]
                in_repo = [x for x in dict.fromkeys(cands) if (repo / x).is_file()]
                if not in_repo or any(x in copied for x in in_repo) or (c, m) in seen:
                    continue
                seen.add((c, m))
                on_job = [x for x in in_repo if job_dir is not None and (job_dir / x).is_file()]
                if not on_job:
                    refuse.append((c, node.lineno, m, in_repo[0]))
                elif all((job_dir / x).read_bytes() != (repo / x).read_bytes() for x in on_job):
                    warn.append((c, node.lineno, m, on_job[0]))
    return refuse, warn


def main(argv=None, env=None):
    import os
    env = dict(os.environ) if env is None else env
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("part")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--step", action="append", help="a job.py step (Windows-side python arguments); repeatable")
    mode.add_argument("--cpu", help="a shell command run in a memory-capped systemd user unit")
    ap.add_argument("--copy", nargs="+", action="extend", default=[],
                    help="repo files to copy into the job folder (A770 mode); repeatable, every repeat adds")
    ap.add_argument("--mem", default="8G")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--override", help="launch even though the check failed; the reason is recorded")
    ap.add_argument("--ignore-busy", action="store_true")
    ap.add_argument("--wait", type=int, default=60, help="seconds to wait for the start line; 0 does not wait")
    ap.add_argument("--block", action="store_true",
                    help="CPU mode: wait for the unit to end and exit with the run's own rc (1 if killed)")
    ap.add_argument("--poll", type=int, default=30, help="seconds between --block checks")
    ap.add_argument("--repo", default=str(TOOLS.parent))
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--legacy-ok", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    repo, pid = Path(a.repo), a.part
    if not re.fullmatch(r"\d+[a-z]?", pid, re.I):
        print(f"part id must look like 115 or 109b, not {pid!r}", file=sys.stderr)
        return 2
    if a.cpu and a.copy:
        print("--copy is for A770 mode only (a CPU run reads the repo directly)", file=sys.stderr)
        return 2
    if a.block and not a.cpu:
        print("--block is for CPU mode only (an A770 run is judged by its progress.log rc= line)", file=sys.stderr)
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
        # 2026-11-09 (Part 198): a chain script named run-<tag>.sh was overwritten by this wrapper, which then ran itself
        # until bash hit its nesting limit and fork failed. Refuse a command that names the wrapper's own path, and
        # refuse to overwrite a file at that path that this launcher did not write.
        rel = script.relative_to(repo).as_posix()
        if rel in a.cpu or script.name in a.cpu.split("/")[-1].split():
            print(f"REFUSED: the --cpu command names {rel}, which is the wrapper this launcher writes; "
                  f"rename the chain script or pick another --tag")
            return 2
        if script.exists() and "Written by tools/launch_arc.py" not in script.read_text(encoding="utf-8", errors="replace"):
            print(f"REFUSED: {rel} exists and was not written by this launcher; it would be overwritten. "
                  f"Rename it or pick another --tag")
            return 2
        if a.dry_run:
            print("DRY RUN: would write", script.relative_to(repo), "and run:", " ".join(cmd))
            return 0
        rdir.mkdir(parents=True, exist_ok=True)
        script.write_text(cpu_script(a.cpu), encoding="utf-8")
        script.chmod(0o755)
        offset = log.stat().st_size if log.exists() else 0  # the log appends across runs: judge only what follows
        p = subprocess.run(cmd, capture_output=True, text=True)
        rec["systemd_run_rc"] = p.returncode
        record(repo, pid, rec)
        print((p.stdout + p.stderr).strip())
        print(f"{'STARTED' if p.returncode == 0 else 'FAILED TO START'}: unit {unit}, log {log.relative_to(repo)}")
        if p.returncode != 0:
            return 1
        if a.block:
            sys.stdout.flush()
            rc, msg = wait_cpu(unit, log, offset, poll=a.poll)
            print(f"{time.strftime('%H:%M:%S')} {msg}")
            return 1 if rc is None else (rc if 0 <= rc < 256 else 1)
        return 0
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
    outside = [c for c in a.copy if Path(c).is_absolute() or ".." in Path(c).parts]
    if missing or outside:
        print("NOT LAUNCHED: --copy needs files inside the repo, given as relative paths without '..': " +
              ", ".join(missing + outside))
        return 2
    posix = posix_only_imports([repo / c for c in a.copy])
    if posix:
        print("NOT LAUNCHED: the A770 job runs on Windows, and these copied files import a POSIX-only module outside a "
              "try/except ImportError (Part 209 on 2026-11-12 and Part 205 on 2026-11-14 died on `import fcntl`): " +
              "; ".join(f"{p}:{line} {mod}" for p, line, mod in posix))
        return 2
    refuse, warn = uncopied_local_imports(repo, a.copy, job_dir)
    for c, line, m, where in warn:
        print(f"WARNING: {c}:{line} imports {m}, not in --copy; the job folder's {where} differs from the repo's, so the "
              f"run would import that older copy. Add --copy {where} unless the old one is meant.")
    if refuse:
        print("NOT LAUNCHED: these copied files import repo modules that are neither in --copy nor in the job folder, so "
              "the run would die at that import (Part 212's pilot imported score212 at its last step): " +
              "; ".join(f"{c}:{line} {m} (add --copy {where})" for c, line, m, where in refuse))
        return 2
    baseline = len(text.splitlines())
    offset = prog.stat().st_size if prog.exists() else 0
    rec.update(mode="arc", steps=a.step, copied=a.copy, baseline_lines=baseline)
    if a.dry_run:
        print(f"DRY RUN: would copy {a.copy}, write steps.txt {a.step}, run {cfg['schtasks']} /run /tn "
              f"{cfg['task_name']}; progress.log baseline {baseline} lines")
        return 0
    try:
        lock = LaunchLock(job_dir).__enter__()
    except RuntimeError as e:
        print("NOT LAUNCHED:", e)
        return 1
    try:
        for c in a.copy:
            dst = job_dir / c
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / c, dst)
        write_steps(job_dir, a.step)
        p = subprocess.run([cfg["schtasks"], "/run", "/tn", cfg["task_name"]], capture_output=True, text=True)
        rec["schtasks_rc"] = p.returncode
        print((p.stdout + p.stderr).replace("\r", "").strip())
        first, shrank, deadline = None, False, time.time() + max(a.wait, 0)
        while p.returncode == 0 and a.wait > 0 and time.time() < deadline:
            data = prog.read_bytes() if prog.exists() else b""
            if len(data) < offset:      # rotated or cleared: everything in it is new
                shrank, offset = True, 0
            first = start_line(data[offset:].decode("utf-8", errors="replace"), a.step)
            if first:
                break
            time.sleep(2)
    finally:
        lock.__exit__(None, None, None)
    rec.update(first_line=first, progress_log_shrank=shrank, progress_log_offset=offset)
    record(repo, pid, rec)
    print(f"progress.log baseline: {baseline} lines, {offset} bytes (a wait should read only what comes after)"
          + (" — the log shrank during the wait, so it was read from the start" if shrank else ""))
    if p.returncode != 0:
        print("FAILED TO START: schtasks returned", p.returncode)
        return 1
    if a.wait <= 0:
        print("STARTED (not waited for; --wait 0): check progress.log for this launch's start line")
        return 0
    if first is None:
        print(f"NO START LINE naming this launch's step in {a.wait} s: check the task and progress.log")
        return 3
    print(("REFUSED BY job.py: " if "REFUSED" in first else "STARTED: ") + first)
    return 1 if "REFUSED" in first else 0


if __name__ == "__main__":
    sys.exit(main())
