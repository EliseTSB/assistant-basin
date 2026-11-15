"""launch_arc --block: a CPU run is judged by the wrapper's own `rc=N HH:MM:SS` line written after the launch, never by
`systemctl show -p Result`. Written 2026-11-14 after Part 205's Instruct stage was OOM-killed at 00:54 with 117 of 168
records and its chain logged "ended: success": the unit had been started with --collect, so once it ended `show` read
an unloaded unit and printed the default value. Break: make cpu_outcome return 0 when no line matches (the killed cases
go red), drop the offset in wait_cpu (the earlier-run case goes red), or drop the line anchors in RC_LINE (the mid-line
case goes red)."""
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import launch_arc as la  # noqa: E402


def test_outcome_reads_the_wrapper_line():
    assert la.cpu_outcome("loading\nstep done\nrc=0 19:31:04\n") == 0
    assert la.cpu_outcome("Traceback ...\nrc=1 02:00:00\n") == 1
    assert la.cpu_outcome("rc=137 00:54:02\n") == 137


def test_outcome_is_none_without_a_line_and_ignores_rc_text_mid_line():
    assert la.cpu_outcome("Loading weights: 100%|##########| 179/179\n") is None
    assert la.cpu_outcome("") is None
    assert la.cpu_outcome("the scorer printed rc=0 12:00:00 in a sentence\n") is None
    assert la.cpu_outcome("19:31:04 step 1 rc=0 19:31:04\n") is None  # a nested launcher's line ending in rc text
    assert la.cpu_outcome("rc=0\n") is None  # the wrapper always writes the time


def test_outcome_takes_the_last_line():
    assert la.cpu_outcome("rc=0 01:00:00\nrc=6 02:00:00\n") == 6


def test_wait_cpu_judges_only_the_log_written_after_the_launch(tmp_path):
    log = tmp_path / "instruct.log"
    log.write_text("earlier run\nrc=0 23:21:21\n", encoding="utf-8")
    offset = log.stat().st_size
    with open(log, "a", encoding="utf-8") as f:
        f.write("Loading weights: 100%\n")  # killed: no wrapper line
    states = iter([True, True, False])
    slept = []
    rc, msg = la.wait_cpu("u", log, offset, poll=7, is_active=lambda u: next(states), sleep=slept.append)
    assert rc is None and "ENDED WITHOUT AN rc= LINE" in msg
    assert slept == [7, 7]
    with open(log, "a", encoding="utf-8") as f:
        f.write("rc=0 01:00:00\n")
    rc, msg = la.wait_cpu("u", log, offset, is_active=lambda u: False, sleep=slept.append)
    assert rc == 0 and msg.startswith("ENDED rc=0")


def test_block_needs_cpu_mode(capsys):
    assert la.main(["205", "--step", "-I -B x.py", "--block", "--no-fetch"]) == 2


@pytest.mark.skipif(shutil.which("systemd-run") is None, reason="needs a systemd user manager")
def test_a_real_collected_unit_killed_before_its_rc_line(tmp_path):
    """The trap itself, on the real user manager: a --collect unit whose wrapper dies before its last line reads
    'success' from `systemctl show` once it has ended, and wait_cpu calls it killed."""
    if subprocess.run(["systemctl", "--user", "is-system-running"], capture_output=True).returncode not in (0, 1):
        pytest.skip("no reachable user manager")
    script, log = tmp_path / "run-t.sh", tmp_path / "t.log"
    script.write_text(la.cpu_script("echo working; kill -9 $$"), encoding="utf-8")
    script.chmod(0o755)
    unit = f"launch-arc-block-test-{time.strftime('%H%M%S')}-{abs(hash(str(tmp_path))) % 10000}"
    p = subprocess.run(["systemd-run", "--user", f"--unit={unit}", "--collect", "/bin/bash", "-c",
                        'exec "$0" >> "$1" 2>&1', str(script), str(log)], capture_output=True, text=True)
    if p.returncode != 0:
        pytest.skip(f"systemd-run refused: {p.stderr.strip()[:120]}")
    rc, msg = la.wait_cpu(unit, log, 0, poll=0.2)
    assert rc is None, msg
    assert "working" in log.read_text(encoding="utf-8")
    shown = subprocess.run(["systemctl", "--user", "show", "-p", "Result", "--value", unit],
                           capture_output=True, text=True).stdout.strip()
    assert shown == "success"  # the default an unloaded unit reports: the reason not to judge by it
